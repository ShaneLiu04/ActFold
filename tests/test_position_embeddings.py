"""Red-phase tests for AR019: ``position_embeddings`` context threading (UT-1901..1908).

Contract: ``specs/changes/AR019-real-checkpoint-e2e/srs.md`` §3.1 FR1 and
``design.md`` §4.2 / §6.1 — ``ManualFoldedForward.forward`` gains a trailing
optional ``position_embeddings: tuple[torch.Tensor, torch.Tensor] | None``
parameter that is threaded into all four construction faces (serial eager,
batched eager, graph-capture static body, AR017 adaptive re-capture), with an
entry-point structural validation (ValueError tier), a capture-time CUDA
check (silent refusal for CPU pe), and a replay-time object-identity check
(``is``, mirroring the ``attention_mask`` contract; mismatch silently falls
back to eager).

UT mapping (design §6.1 table, one row per test function):

* UT-1901 ``test_ut1901_serial_eager_pe_passthrough`` — serial eager face.
* UT-1902 ``test_ut1902_batched_eager_pe_passthrough`` — batched eager face.
* UT-1903 ``test_ut1903_graph_capture_bakes_pe`` — graph static body baking.
* UT-1904 ``test_ut1904_adaptive_recapture_bakes_pe`` — AR017 re-capture face.
* UT-1905 ``test_ut1905_replay_pe_object_identity`` — replay identity, plus
  the CPU companion ``test_ut1905_eager_face_per_call_pe_semantics`` (the
  "eager output follows pe_b" half of the row, covered on the eager face as
  the brief requires; the graph half stays CUDA-gated).
* UT-1906 ``test_ut1906_none_default_and_plain_layer_zero_change`` — three
  sub-assertions: (a) green nail / (b) explicit None / (c) pe + plain layer.
* UT-1907 ``test_ut1907_structural_validation_value_error`` — ValueError tier.
* UT-1908 ``test_ut1908_capture_rejects_cpu_pe_silently`` — CUDA capture gate.

Red state (why every failing test below fails today): production
``ManualFoldedForward.forward`` (architecture_utils.py :499-507) has no
``position_embeddings`` parameter, so every call passing it raises
``TypeError`` — the expected red cause for UT-1901/1902/1903/1904/
1905(+companion)/1906(b,c)/1908. UT-1907 fails because that same TypeError
is raised where the FR1 contract demands ``ValueError``. UT-1906(a) is the
deliberate green nail (no kwarg → current behavior bit-for-bit) and must stay
green after the Green phase. No test fails through an import, fixture, or
test-side bug: every import exists today and every failure surfaces on the
missing-production face.

Stubs (self-contained, CPU-runnable, no transformers dependency; repo
precedent duplicates fixtures per test module):

* ``_ContextLayer`` — decoder layer whose forward signature carries
  ``position_embeddings``; records every call's pe argument and CONSUMES
  cos/sin so wrong pe values change the output. The inject is a
  direction-varying per-dimension vector added to the residual stream (a
  pure per-token constant or per-token scaling would be cancelled by the
  LayerNorm normalization invariance). Device/dtype tolerant: UT-1908 feeds
  a CPU pe to a CUDA model on the eager path, which must stay correct.
* ``_PlainLayer`` — the synthetic-layer nail: identical structure, signature
  WITHOUT ``position_embeddings`` (exercises the accepted/varkw filtering).
* ``_PeStubModel`` — mock LLaMA-layout decoder (embed -> layers -> model.norm
  -> lm_head) built from a layer factory, L=2/H=32/T=16/V=64 (N=4 batched;
  N*T=64 stays a power of two for the exact capacity arithmetic UT-1904
  inherits from the AR017 fixtures).

Tolerances (AR016/AR017 precedent): eager-folded vs welded-pe reference
``allclose(1e-6)`` on CPU (split-path gathered GEMM 1-ULP, #41); graph-vs-
eager 1e-4 plus per-row argmax equality; replay determinism ``torch.equal``;
UT-1906 uses ``split_layers=False`` so the nail is bit-exact (``torch.equal``,
#41 non-split CPU contract). Test-side readbacks are unrestricted — the
single-host-sync contract binds production code, not tests.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.adaptive_capacity import AdaptiveCapacityConfig
from actfold.models.architecture_utils import ManualFoldedForward, detect_architecture

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_VOCAB = 64
_HIDDEN = 32
_NUM_LAYERS = 2
_SEQ = 16
_NUM_CANDIDATES = 4
# N * T stays a power of two (AR017 fixture constraint): UT-1904's capacity
# assertions (ceil(ratio * N * T) == 32) are exact only for power-of-two totals.
_NUM_TOKENS = _NUM_CANDIDATES * _SEQ
assert _NUM_TOKENS > 0 and _NUM_TOKENS & (_NUM_TOKENS - 1) == 0
_TAU = 0.95
_CACHE_BUDGET = 1024
_UNSET = object()


# ---------------------------------------------------------------------------
# Self-contained synthetic decoder (LlamaLikeModel layout; test_cuda_graph.py
# and test_batched_graph_forward.py precedent — fixtures are duplicated per
# test module, never imported across test modules).
# ---------------------------------------------------------------------------


class _MaskableAttention(nn.Module):
    """Token-wise attention stand-in consuming the mask multiplicatively."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = hidden_states * 0.5
        if attention_mask is not None:
            out = out * attention_mask.to(out.dtype).unsqueeze(-1)
        return out


class _ContextLayer(nn.Module):
    """Decoder layer whose signature carries ``position_embeddings``.

    Records every call's ``position_embeddings`` argument (``received_pe``)
    and consumes cos/sin so a wrong pe value observably changes the output:
    the inject ``0.1 * (cos + sin)`` is a direction-varying per-dimension
    vector added to the residual stream. The
    ``post_attention_layernorm -> mlp`` chain stays detectable for the
    split-layer wrapper (graph capture engages the split hooks).
    Device/dtype tolerant: the eager face must numerically consume a CPU pe
    against CUDA activations (UT-1908).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = _MaskableAttention()
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.received_pe: list[Any] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self.received_pe.append(position_embeddings)
        residual = hidden_states
        if position_embeddings is not None:
            cos, sin = position_embeddings
            residual = residual + 0.1 * (
                cos.to(device=residual.device, dtype=residual.dtype)
                + sin.to(device=residual.device, dtype=residual.dtype)
            )
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class _PlainLayer(nn.Module):
    """Synthetic-layer nail: signature WITHOUT ``position_embeddings``.

    Structurally identical to :class:`_ContextLayer` minus the pe parameter
    and the inject, so UT-1906 can pin that a non-None pe is filtered away
    with bit-identical outputs.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = _MaskableAttention()
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class _PeStubModel(nn.Module):
    """Mock LLaMA-layout decoder built from a layer factory."""

    def __init__(self, layer_factory: Callable[[int], nn.Module]) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": _VOCAB})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.model.layers = nn.ModuleList([layer_factory(_HIDDEN) for _ in range(_NUM_LAYERS)])
        self.model.norm = nn.LayerNorm(_HIDDEN)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))


# ---------------------------------------------------------------------------
# Shared scenario helpers
# ---------------------------------------------------------------------------


def _build_model(
    device: str,
    layer_factory: Callable[[int], nn.Module],
    seed: int = 2026,
) -> tuple[nn.Module, Any]:
    """Deterministically build the decoder and its architecture profile."""
    torch.manual_seed(seed)
    model = _PeStubModel(layer_factory).to(device)
    model.eval()
    profile = detect_architecture(model)
    return model, profile


def _make_mff(
    model: nn.Module,
    device: str,
    gate: SimilarityGate | None = None,
    cache: Any = None,
    use_cuda_graph: bool = False,
    graph_capacity_ratio: float = 0.5,
    graph_adaptive_capacity: bool = False,
    graph_adaptive_config: Any | None = None,
    split_layers: bool = True,
) -> ManualFoldedForward:
    """Fresh ManualFoldedForward (own cache + gate) over the shared model.

    Follows the test_batched_graph_forward.py wiring: split layers with a
    lowered ``split_min_tokens`` so the split FFN path engages at the test
    shapes, plus the AR002 graph flags and (UT-1904) the AR017 adaptive
    flags. UT-1906 passes ``split_layers=False`` for a bit-exact nail (the
    #41 non-split CPU contract allows ``torch.equal``).
    """
    if cache is None:
        cache = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    if gate is None:
        gate = SimilarityGate(tau=_TAU, metric="cosine")
    adaptive_kwargs: dict[str, Any] = {}
    if graph_adaptive_capacity or graph_adaptive_config is not None:
        adaptive_kwargs["graph_adaptive_capacity"] = graph_adaptive_capacity
        adaptive_kwargs["graph_adaptive_config"] = graph_adaptive_config
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        split_layers=split_layers,
        split_min_tokens=1,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=graph_capacity_ratio,
        **adaptive_kwargs,
    )


def _make_pe(device: str, seq: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a ``(cos, sin)`` pair ``[1, seq, hidden]`` with a private generator.

    The batch dimension stays 1 (free to broadcast against ``[B, T]`` tokens
    per the FR1 shape contract: only the seq dim and the internal head_dim
    are constrained).
    """
    gen = torch.Generator().manual_seed(seed)
    cos = torch.randn(1, seq, _HIDDEN, generator=gen).to(device)
    sin = torch.randn(1, seq, _HIDDEN, generator=gen).to(device)
    return (cos, sin)


def _change_tokens(tokens: torch.Tensor, positions: list[int]) -> torch.Tensor:
    """Clone ``tokens`` and change the flat ``positions`` to a different id."""
    child = tokens.clone()
    flat = child.reshape(-1)
    for pos in positions:
        flat[pos] = (flat[pos] + 1) % _VOCAB
    return child


def _make_candidates(
    device: str,
    changes_per_row: int = 1,
    seed: int = 4201,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a parent ``[1, T]`` and N equal-length mid-divergent candidates."""
    torch.manual_seed(seed)
    parent = torch.randint(0, _VOCAB, (1, _SEQ), device=device)
    mid = _SEQ // 2
    rows = []
    for i in range(_NUM_CANDIDATES):
        child = parent.clone()
        for j in range(changes_per_row):
            base = int(child[0, mid + j].item())
            child[0, mid + j] = (base + 1 + i + j) % _VOCAB
        rows.append(child)
    return parent, torch.cat(rows, dim=0)


def _ids(tag: str, num: int = _NUM_CANDIDATES) -> list[str]:
    """Build ``num`` branch ids sharing the ``tag`` prefix (disjoint per UT)."""
    return [f"{tag}{i}" for i in range(num)]


def _serial(
    mff: ManualFoldedForward,
    tokens: torch.Tensor,
    branch_id: str,
    parent_branch_id: str | None = None,
    position_embeddings: Any = _UNSET,
    step_idx: int = 1,
) -> torch.Tensor:
    """One serial ``branch_id`` forward under ``no_grad``.

    ``position_embeddings`` uses an ``_UNSET`` sentinel so the helper can
    express all three call forms: kwarg absent, explicit ``None``, and a
    real pe (UT-1906 needs the distinction).
    """
    kwargs: dict[str, Any] = {}
    if position_embeddings is not _UNSET:
        kwargs["position_embeddings"] = position_embeddings
    with torch.no_grad():
        return mff(
            tokens,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            step_idx=step_idx,
            **kwargs,
        )


def _batched(
    mff: ManualFoldedForward,
    stacked: torch.Tensor,
    branch_ids: list[str],
    parent_branch_id: str,
    position_embeddings: Any = _UNSET,
    step_idx: int = 1,
) -> torch.Tensor:
    """One batched ``branch_ids`` forward under ``no_grad``."""
    kwargs: dict[str, Any] = {}
    if position_embeddings is not _UNSET:
        kwargs["position_embeddings"] = position_embeddings
    with torch.no_grad():
        return mff(
            stacked,
            branch_id=None,
            branch_ids=branch_ids,
            parent_branch_id=parent_branch_id,
            step_idx=step_idx,
            **kwargs,
        )


def _raw_forward(
    model: nn.Module,
    tokens: torch.Tensor,
    pe: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Manual "pe welded into the layer" reference path (no folding).

    embed -> ``layer(x, position_embeddings=pe)`` per layer -> final norm ->
    lm_head. Only valid for :class:`_ContextLayer` stacks; the reference the
    FR1 acceptance standard compares the folded output against.
    """
    with torch.no_grad():
        x = model.model.embed_tokens(tokens)
        for layer in model.model.layers:
            x = layer(x, position_embeddings=pe)
        x = model.model.norm(x)
        return model.lm_head(x)


def _context_layers(mff: ManualFoldedForward) -> list[_ContextLayer]:
    """The raw ``_ContextLayer`` objects wrapped by ``mff``."""
    return [wrapped.original_layer for wrapped in mff._wrapped_layers]


def _user_warning_messages(record: Any) -> list[str]:
    """Return the messages of every UserWarning in a catch_warnings log."""
    return [str(entry.message) for entry in record if issubclass(entry.category, UserWarning)]


# ---------------------------------------------------------------------------
# UT-1901: serial eager pe pass-through
# ---------------------------------------------------------------------------


def test_ut1901_serial_eager_pe_passthrough() -> None:
    """UT-1901 (FR1 face 1, srs standard 1): serial eager pe pass-through.

    A parent pass and a folded child pass both supply ``pe``; every wrapped
    layer must receive the SUPPLIED tuple object (``is`` identity — the
    threading must not rebuild the tuple), and the folded output must match
    the manual "pe welded into the layer" reference path (allclose 1e-6:
    split-path gathered GEMM 1-ULP, #41) while genuinely differing from the
    no-pe semantics (the inject makes wrong pe values observable).

    Red reason: ``ManualFoldedForward.forward`` does not accept
    ``position_embeddings`` — the parent call raises TypeError.
    """
    device = "cpu"
    model, _profile = _build_model(device, _ContextLayer)
    torch.manual_seed(9101)
    parent = torch.randint(0, _VOCAB, (1, _SEQ), device=device)
    child = _change_tokens(parent, [4, 11])
    pe = _make_pe(device, _SEQ, seed=9101)

    mff = _make_mff(model, device)
    layers = _context_layers(mff)
    counts_before = [len(layer.received_pe) for layer in layers]

    _serial(mff, parent, "1901-p", position_embeddings=pe)
    out = _serial(mff, child, "1901-c", parent_branch_id="1901-p", position_embeddings=pe)

    for layer, n0 in zip(layers, counts_before):
        entries = layer.received_pe[n0:]
        assert len(entries) == 2, "one layer call per pass (parent + folded child)"
        assert all(entry is pe for entry in entries), "pe must be the supplied object"

    reference = _raw_forward(model, child, pe)
    assert torch.allclose(out, reference, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out, _raw_forward(model, child, None), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# UT-1902: batched eager pe pass-through
# ---------------------------------------------------------------------------


def test_ut1902_batched_eager_pe_passthrough() -> None:
    """UT-1902 (FR1 face 2, srs standard 1): batched eager pe pass-through.

    The ``branch_ids`` path (``[N, T]`` tokens, eager — ``use_cuda_graph``
    off) must thread pe into every wrapped layer for the whole batch, and
    every branch row's output must match the welded-pe reference row
    (allclose 1e-6 on CPU) while differing from the no-pe semantics.

    Red reason: ``ManualFoldedForward.forward`` does not accept
    ``position_embeddings`` — the parent call raises TypeError.
    """
    device = "cpu"
    model, _profile = _build_model(device, _ContextLayer)
    parent, stacked = _make_candidates(device, changes_per_row=1, seed=9202)
    pe = _make_pe(device, _SEQ, seed=9202)

    mff = _make_mff(model, device)
    layers = _context_layers(mff)
    counts_before = [len(layer.received_pe) for layer in layers]

    _serial(mff, parent, "1902-p", position_embeddings=pe)
    out = _batched(mff, stacked, _ids("1902-b"), "1902-p", position_embeddings=pe)
    assert tuple(out.shape) == (_NUM_CANDIDATES, _SEQ, _VOCAB)

    for layer, n0 in zip(layers, counts_before):
        entries = layer.received_pe[n0:]
        assert len(entries) == 2, "one layer call per pass (parent + batched child)"
        assert all(entry is pe for entry in entries), "pe must be the supplied object"

    for row in range(_NUM_CANDIDATES):
        reference = _raw_forward(model, stacked[row : row + 1], pe)
        assert torch.allclose(out[row : row + 1], reference, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out, _raw_forward(model, stacked, None), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# UT-1903: graph capture bakes pe into the static body
# ---------------------------------------------------------------------------


@_cuda
def test_ut1903_graph_capture_bakes_pe(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1903 (FR1 face 3, srs standards 1/4): graph capture baking.

    The first graph-eligible serial child call captures; the static body's
    layer calls must receive the SUPPLIED pe object by reference (``is``
    identity — the bake must not copy, mirroring ``attention_mask_static``).
    Three further replays are bit-stable, match the eager folded reference
    computed with pe (allclose 1e-4 + argmax), and differ from the no-pe
    reference (the baked pe genuinely drives the output). A monkeypatched
    readback spy around one replay asserts at most ONE host readback (the
    budget count) — the bake must not introduce new host syncs (#37).

    Red reason: ``ManualFoldedForward.forward`` does not accept
    ``position_embeddings`` — the eager reference's parent call raises
    TypeError.
    """
    device = "cuda"
    model, _profile = _build_model(device, _ContextLayer)
    model_ref, _ref_profile = _build_model(device, _ContextLayer)
    torch.manual_seed(9303)
    parent = torch.randint(0, _VOCAB, (1, _SEQ), device=device)
    child = _change_tokens(parent, [5, 10])
    pe = _make_pe(device, _SEQ, seed=9303)

    # Eager folded references on a second identically weighted model (a
    # shared model would pollute the layer recording asserted below).
    mff_ref_pe = _make_mff(model_ref, device)
    _serial(mff_ref_pe, parent, "1903-rp", position_embeddings=pe)
    ref_pe = _serial(
        mff_ref_pe, child, "1903-rc", parent_branch_id="1903-rp", position_embeddings=pe
    ).clone()
    mff_ref_none = _make_mff(model_ref, device)
    _serial(mff_ref_none, parent, "1903-np")
    ref_none = _serial(mff_ref_none, child, "1903-nc", parent_branch_id="1903-np").clone()
    assert not torch.allclose(ref_pe, ref_none, atol=1e-4, rtol=1e-4)

    mff = _make_mff(model, device, use_cuda_graph=True, graph_capacity_ratio=0.5)
    layers = _context_layers(mff)
    _serial(mff, parent, "1903-p", position_embeddings=pe)
    counts_before = [len(layer.received_pe) for layer in layers]

    with warnings.catch_warnings(record=True) as record_capture:
        warnings.simplefilter("always")
        out1 = _serial(mff, child, "1903-c1", parent_branch_id="1903-p", position_embeddings=pe)
    assert len(_user_warning_messages(record_capture)) == 0
    assert mff.graph_runner is not None

    # Spy: the capture body consumed the supplied object by reference (no
    # copy — identity is the "pe comes from the static reference" proof).
    for layer, n0 in zip(layers, counts_before):
        entries = layer.received_pe[n0:]
        assert len(entries) >= 1, "the capture body must call the layers with pe"
        assert all(entry is pe for entry in entries), "baked pe must be the supplied object"

    outs = [out1.clone()]
    for k in range(3):
        with warnings.catch_warnings(record=True) as record_replay:
            warnings.simplefilter("always")
            out_k = _serial(
                mff, child, f"1903-r{k}", parent_branch_id="1903-p", position_embeddings=pe
            )
        assert len(_user_warning_messages(record_replay)) == 0
        outs.append(out_k.clone())
    for later in outs[1:]:
        assert torch.equal(outs[0], later), "replays of identical inputs must be bit-equal"
    assert torch.allclose(outs[0], ref_pe, atol=1e-4, rtol=1e-4)
    assert torch.equal(outs[0].argmax(dim=-1), ref_pe.argmax(dim=-1))
    assert not torch.allclose(outs[0], ref_none, atol=1e-4, rtol=1e-4)

    # No new host syncs: one replay performs at most the single budget
    # readback (test_cuda_graph.py spy precedent).
    counts = {"tolist": 0, "int": 0, "item": 0}
    orig_tolist = torch.Tensor.tolist
    orig_int = torch.Tensor.__int__
    orig_item = torch.Tensor.item

    def _count_tolist(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        counts["tolist"] += 1
        return orig_tolist(self, *args, **kwargs)  # type: ignore[arg-type]

    def _count_int(self: torch.Tensor, *args: object, **kwargs: object) -> int:
        counts["int"] += 1
        return orig_int(self, *args, **kwargs)  # type: ignore[arg-type,return-value]

    def _count_item(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        counts["item"] += 1
        return orig_item(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.Tensor, "tolist", _count_tolist)
    monkeypatch.setattr(torch.Tensor, "__int__", _count_int, raising=False)
    monkeypatch.setattr(torch.Tensor, "item", _count_item)

    out_spy = _serial(mff, child, "1903-spy", parent_branch_id="1903-p", position_embeddings=pe)
    total = counts["tolist"] + counts["int"] + counts["item"]
    assert torch.allclose(out_spy, ref_pe, atol=1e-4, rtol=1e-4)
    assert total <= 1, (
        f"a pe-baked replay must add no host readback (budget readback only), "
        f"observed {total}: tolist={counts['tolist']}, int={counts['int']}, "
        f"item={counts['item']}"
    )


# ---------------------------------------------------------------------------
# UT-1904: adaptive re-capture bakes pe into the new runner
# ---------------------------------------------------------------------------


@_cuda
def test_ut1904_adaptive_recapture_bakes_pe() -> None:
    """UT-1904 (FR1 face 4, srs standard 2): the AR017 shrink re-capture bake.

    Fixture mirrors UT-1602 (``changes_per_row=2`` → D = 8 of N*T = 64,
    ``graph_capacity_ratio=0.5`` → C = 32, ``shrink_patience=2``): the first
    batched call captures, the second triggers the shrink re-capture. The
    NEW runner's capture body must receive the SUPPLIED pe object by
    reference (``is`` identity — a re-capture that drops the rotary context
    would silently mis-compute every later replay), the runner swaps
    (capacity < 32, still ≥ the observed divergence), zero warnings are
    emitted, and all three step outputs match the eager batched reference
    computed with pe (allclose 1e-4 + per-row argmax) while differing from
    the no-pe reference.

    Red reason: ``ManualFoldedForward.forward`` does not accept
    ``position_embeddings`` — the eager reference's parent call raises
    TypeError.
    """
    device = "cuda"
    model, _profile = _build_model(device, _ContextLayer)
    model_ref, _ref_profile = _build_model(device, _ContextLayer)
    parent, stacked = _make_candidates(device, changes_per_row=2, seed=9404)
    pe = _make_pe(device, _SEQ, seed=9404)

    mff_ref_pe = _make_mff(model_ref, device)
    _serial(mff_ref_pe, parent, "1904-rp", position_embeddings=pe)
    ref_pe = _batched(
        mff_ref_pe, stacked, _ids("1904-r"), "1904-rp", position_embeddings=pe
    ).clone()
    mff_ref_none = _make_mff(model_ref, device)
    _serial(mff_ref_none, parent, "1904-np")
    ref_none = _batched(mff_ref_none, stacked, _ids("1904-n"), "1904-np").clone()
    assert not torch.allclose(ref_pe, ref_none, atol=1e-4, rtol=1e-4)

    mff = _make_mff(
        model,
        device,
        use_cuda_graph=True,
        graph_capacity_ratio=0.5,
        graph_adaptive_capacity=True,
        graph_adaptive_config=AdaptiveCapacityConfig(shrink_patience=2),
    )
    layers = _context_layers(mff)
    _serial(mff, parent, "1904-p", position_embeddings=pe)

    with warnings.catch_warnings(record=True) as record_all:
        warnings.simplefilter("always")
        out1 = _batched(mff, stacked, _ids("1904-a"), "1904-p", position_embeddings=pe)
        runner_old = mff.graph_runner_batched
        assert runner_old is not None
        assert runner_old.capacity == 32
        counts_before = [len(layer.received_pe) for layer in layers]
        out2 = _batched(mff, stacked, _ids("1904-b"), "1904-p", position_embeddings=pe)
        out3 = _batched(mff, stacked, _ids("1904-c"), "1904-p", position_embeddings=pe)
    assert len(_user_warning_messages(record_all)) == 0

    runner = mff.graph_runner_batched
    assert runner is not None
    assert runner is not runner_old, "the sustained low load must shrink re-capture"
    assert runner.capacity < 32
    divergences = runner.last_layer_divergences
    assert divergences is not None
    assert runner.capacity >= max(divergences)
    assert runner.budget_exceeded is False

    # The re-captured runner baked the supplied pe object by reference (the
    # entries recorded between the two snapshots are exactly the new
    # runner's warmup + capture static-body calls).
    for layer, n0 in zip(layers, counts_before):
        entries = layer.received_pe[n0:]
        assert len(entries) >= 1, "the re-capture body must call the layers with pe"
        assert all(entry is pe for entry in entries), "re-captured pe must be the object"

    for out in (out1, out2, out3):
        assert torch.allclose(out, ref_pe, atol=1e-4, rtol=1e-4)
        assert torch.equal(out.argmax(dim=-1), ref_pe.argmax(dim=-1))
    assert not torch.allclose(out3, ref_none, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# UT-1905: replay-time object identity (graph face, CUDA) + eager companion
# ---------------------------------------------------------------------------


@_cuda
def test_ut1905_replay_pe_object_identity() -> None:
    """UT-1905 (FR1 replay contract, srs standard 5): pe_a captured / pe_b replayed.

    A graph captured with ``pe_a`` must, on a same-shaped call with a
    DIFFERENT pe object ``pe_b``, silently fall back to eager (zero
    UserWarnings — the mask-object-mismatch semantics, never a warning) and
    produce the pe_b-semantics output: the exact eager mirror (parent cached
    with pe_a, child computed with pe_b) on a fresh identically weighted
    stack. The runner stays pinned to pe_a: a later pe_a call still replays
    through the same runner object with pe_a semantics.

    Red reason: ``ManualFoldedForward.forward`` does not accept
    ``position_embeddings`` — the eager mirror's parent call raises
    TypeError. (mutation-② state: deleting the identity check lets pe_b
    replay through the graph with the baked pe_a values — the wrong-semantics
    output breaks the ``allclose(out_b, ref_b)`` assertion; the two red
    causes are recorded separately per design §6.1.)
    """
    device = "cuda"
    model, _profile = _build_model(device, _ContextLayer)
    model_ref, _ref_profile = _build_model(device, _ContextLayer)
    torch.manual_seed(9505)
    parent = torch.randint(0, _VOCAB, (1, _SEQ), device=device)
    child = _change_tokens(parent, [7])
    pe_a = _make_pe(device, _SEQ, seed=9505)
    pe_b = _make_pe(device, _SEQ, seed=9506)

    # Exact eager mirrors on a fresh stack: the parent is ALWAYS cached with
    # pe_a (that is the capture-time state); the child is computed with pe_a
    # respectively pe_b.
    mff_ref = _make_mff(model_ref, device)
    _serial(mff_ref, parent, "1905-rp", position_embeddings=pe_a)
    ref_a = _serial(
        mff_ref, child, "1905-ra", parent_branch_id="1905-rp", position_embeddings=pe_a
    ).clone()
    ref_b = _serial(
        mff_ref, child, "1905-rb", parent_branch_id="1905-rp", position_embeddings=pe_b
    ).clone()
    assert not torch.allclose(ref_a, ref_b, atol=1e-6, rtol=1e-6)

    mff = _make_mff(model, device, use_cuda_graph=True, graph_capacity_ratio=0.5)
    _serial(mff, parent, "1905-p", position_embeddings=pe_a)
    with warnings.catch_warnings(record=True) as record_capture:
        warnings.simplefilter("always")
        out_cap = _serial(
            mff, child, "1905-c1", parent_branch_id="1905-p", position_embeddings=pe_a
        )
    assert len(_user_warning_messages(record_capture)) == 0
    assert mff.graph_runner is not None
    runner = mff.graph_runner
    assert torch.allclose(out_cap, ref_a, atol=1e-4, rtol=1e-4)

    # pe_b (same shape, different object): silent eager fallback that
    # genuinely consumes pe_b (not the baked pe_a values).
    with warnings.catch_warnings(record=True) as record_b:
        warnings.simplefilter("always")
        out_b = _serial(mff, child, "1905-c2", parent_branch_id="1905-p", position_embeddings=pe_b)
    assert len(_user_warning_messages(record_b)) == 0
    assert torch.allclose(out_b, ref_b, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out_b, ref_a, atol=1e-6, rtol=1e-6)
    assert mff.graph_runner is runner, "the fallback must not re-pin or drop the runner"

    # pe_a still pinned: the runner keeps serving pe_a replays.
    with warnings.catch_warnings(record=True) as record_a2:
        warnings.simplefilter("always")
        out_a2 = _serial(mff, child, "1905-c3", parent_branch_id="1905-p", position_embeddings=pe_a)
    assert len(_user_warning_messages(record_a2)) == 0
    assert torch.allclose(out_a2, ref_a, atol=1e-4, rtol=1e-4)
    assert mff.graph_runner is runner


def test_ut1905_eager_face_per_call_pe_semantics() -> None:
    """UT-1905 eager companion (CPU): the eager face uses the per-call pe.

    The graph half of UT-1905 is CUDA-gated; its "output = pe_b semantics"
    half is pinned here on the eager face: one stack whose parent is cached
    with pe_a and whose children are computed with pe_a then pe_b must
    produce each call's pe semantics (matched against exact mirrors on
    fresh identically weighted stacks), i.e. eager never pins a pe object.

    Red reason: ``ManualFoldedForward.forward`` does not accept
    ``position_embeddings`` — the first pe call raises TypeError.
    """
    device = "cpu"
    model, _profile = _build_model(device, _ContextLayer)
    torch.manual_seed(9515)
    parent = torch.randint(0, _VOCAB, (1, _SEQ), device=device)
    child = _change_tokens(parent, [7])
    pe_a = _make_pe(device, _SEQ, seed=9515)
    pe_b = _make_pe(device, _SEQ, seed=9516)

    mff = _make_mff(model, device)
    _serial(mff, parent, "1905e-p", position_embeddings=pe_a)
    out_a = _serial(mff, child, "1905e-a", parent_branch_id="1905e-p", position_embeddings=pe_a)
    out_b = _serial(mff, child, "1905e-b", parent_branch_id="1905e-p", position_embeddings=pe_b)

    # Exact mirrors: fresh stacks replaying the identical call sequences.
    mff_mirror_a = _make_mff(model, device)
    _serial(mff_mirror_a, parent, "1905e-p", position_embeddings=pe_a)
    ref_a = _serial(
        mff_mirror_a, child, "1905e-c", parent_branch_id="1905e-p", position_embeddings=pe_a
    )
    mff_mirror_b = _make_mff(model, device)
    _serial(mff_mirror_b, parent, "1905e-p", position_embeddings=pe_a)
    ref_b = _serial(
        mff_mirror_b, child, "1905e-c", parent_branch_id="1905e-p", position_embeddings=pe_b
    )

    assert torch.allclose(out_a, ref_a, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_b, ref_b, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out_b, ref_a, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# UT-1906: None default + synthetic-layer zero change (three sub-assertions)
# ---------------------------------------------------------------------------


def test_ut1906_none_default_and_plain_layer_zero_change() -> None:
    """UT-1906 (FR1 standard 3): None default + plain-layer zero change.

    Three sub-assertions on a ``_PlainLayer`` stack (signature without pe),
    ``split_layers=False`` so folded == raw is bit-exact on CPU (#41
    non-split contract; the split gather is not this UT's target):

    * (a) GREEN NAIL — no pe kwarg: the folded output equals the raw model
      forward bit-for-bit. Passes today and must stay green after Green.
    * (b) RED today — explicit ``position_embeddings=None``: the current
      signature cannot accept the kwarg (TypeError); after implementation
      the output must equal (a) bit-for-bit.
    * (c) RED today — a real pe against the plain layer (signature lacks
      pe, no varkw): TypeError today; after implementation the pe must be
      filtered away with the output equal to (a) bit-for-bit.

    The Red run fails at (b) with TypeError; (c) shares that red cause (the
    same missing parameter), per design §6.1's attribution note.
    """
    device = "cpu"
    model, _profile = _build_model(device, _PlainLayer)
    torch.manual_seed(9606)
    parent = torch.randint(0, _VOCAB, (1, _SEQ), device=device)
    child = _change_tokens(parent, [3, 9])

    with torch.no_grad():
        raw = model(child)

    # (a) Green nail: no kwarg -> current behavior, bit-for-bit.
    mff_a = _make_mff(model, device, split_layers=False)
    _serial(mff_a, parent, "1906-p")
    out_a = _serial(mff_a, child, "1906-a", parent_branch_id="1906-p")
    assert torch.equal(out_a, raw)

    # (b) Explicit None: today TypeError (red); after Green equals (a).
    mff_b = _make_mff(model, device, split_layers=False)
    _serial(mff_b, parent, "1906-p", position_embeddings=None)
    out_b = _serial(mff_b, child, "1906-b", parent_branch_id="1906-p", position_embeddings=None)
    assert torch.equal(out_b, raw)
    assert torch.equal(out_b, out_a)

    # (c) Real pe + plain layer: today TypeError (red); after Green the pe
    # is filtered away and the output equals (a).
    pe = _make_pe(device, _SEQ, seed=9606)
    mff_c = _make_mff(model, device, split_layers=False)
    _serial(mff_c, parent, "1906-p", position_embeddings=pe)
    out_c = _serial(mff_c, child, "1906-c", parent_branch_id="1906-p", position_embeddings=pe)
    assert torch.equal(out_c, raw)
    assert torch.equal(out_c, out_a)


# ---------------------------------------------------------------------------
# UT-1907: entry-point structural validation (ValueError tier)
# ---------------------------------------------------------------------------


def test_ut1907_structural_validation_value_error() -> None:
    """UT-1907 (FR1 exception handling): illegal pe structures raise ValueError.

    Every sub-case from the design §6.1 row must raise ``ValueError`` at the
    forward entry point (pure host-metadata checks, no sync): a non-tuple
    (bare tensor); a non-tensor element; a tuple of length != 2; a cos seq
    dim (``shape[-2]``) different from the tokens' ``T``; a cos/sin
    head_dim (``shape[-1]``) mismatch; and a 1-D cos (no seq dim at all —
    must stay in the ValueError tier, never leak an IndexError from the
    ``shape[-2]`` access).

    Red reason: the current forward signature rejects the kwarg with
    TypeError, so ``pytest.raises(ValueError)`` lets the TypeError escape —
    the wrong exception tier is the red cause.
    """
    device = "cpu"
    model, _profile = _build_model(device, _ContextLayer)
    torch.manual_seed(9707)
    tokens = torch.randint(0, _VOCAB, (1, _SEQ), device=device)
    cos, sin = _make_pe(device, _SEQ, seed=9707)
    gen = torch.Generator().manual_seed(9708)
    bad_seq_cos, bad_seq_sin = _make_pe(device, _SEQ + 3, seed=9709)
    bad_head = (
        cos,
        torch.randn(1, _SEQ, _HIDDEN + 4, generator=gen).to(device),
    )

    mff = _make_mff(model, device)

    bad_values: list[Any] = [
        cos,  # non-tuple: a bare tensor
        (cos, None),  # non-tensor element
        (cos, sin, cos),  # length != 2
        (bad_seq_cos, bad_seq_sin),  # cos seq dim (shape[-2]) != tokens T
        bad_head,  # cos/sin head_dim (shape[-1]) mismatch
        (cos.reshape(-1), sin.reshape(-1)),  # 1-D: no seq dim (review N-2)
    ]
    for bad in bad_values:
        with pytest.raises(ValueError):
            _serial(mff, tokens, "1907-p", position_embeddings=bad)


# ---------------------------------------------------------------------------
# UT-1908: capture-time CUDA rejection of a CPU pe
# ---------------------------------------------------------------------------


@_cuda
def test_ut1908_capture_rejects_cpu_pe_silently() -> None:
    """UT-1908 (FR1 exception handling): a CPU pe silently refuses capture.

    On a CUDA stack with ``use_cuda_graph=True``: the parent pass with a CPU
    pe simply runs eager (parents never capture); the first graph-eligible
    child call with the CPU pe must silently refuse the capture — zero
    UserWarnings, no runner, the disable flag untouched, and the eager
    output numerically consuming the CPU pe (allclose 1e-4 vs the welded-pe
    reference). The refusal is not a disable: a subsequent child with a CUDA
    pe still captures and replays correctly.

    Red reason: ``ManualFoldedForward.forward`` does not accept
    ``position_embeddings`` — the parent call raises TypeError. (Deleting
    the CUDA check in a mutation lets the capture body hit the cross-device
    inject, surfacing as the capture-failed warning + disable — the zero-
    warning and runner assertions break; Red-phase verification and the
    mutation check are the same face per design §6.1.)
    """
    device = "cuda"
    model, _profile = _build_model(device, _ContextLayer)
    torch.manual_seed(9808)
    parent = torch.randint(0, _VOCAB, (1, _SEQ), device=device)
    child = _change_tokens(parent, [6])
    child2 = _change_tokens(parent, [2, 13])
    pe_cpu = _make_pe("cpu", _SEQ, seed=9808)
    pe_cuda = tuple(t.to(device) for t in pe_cpu)

    mff = _make_mff(model, device, use_cuda_graph=True, graph_capacity_ratio=0.5)

    # Parent passes stay eager regardless: the CPU pe must simply flow.
    with warnings.catch_warnings(record=True) as record_parent:
        warnings.simplefilter("always")
        _serial(mff, parent, "1908-p", position_embeddings=pe_cpu)
    assert len(_user_warning_messages(record_parent)) == 0

    # First graph-eligible child with a CPU pe: silent capture refusal.
    with warnings.catch_warnings(record=True) as record_child:
        warnings.simplefilter("always")
        out = _serial(mff, child, "1908-c1", parent_branch_id="1908-p", position_embeddings=pe_cpu)
    assert len(_user_warning_messages(record_child)) == 0
    assert mff.graph_runner is None
    assert mff._graph_capture_failed is False
    reference = _raw_forward(model, child, pe_cpu)
    assert torch.allclose(out, reference, atol=1e-4, rtol=1e-4)

    # The refusal is not a disable: a CUDA pe still captures afterwards.
    with warnings.catch_warnings(record=True) as record_c2:
        warnings.simplefilter("always")
        out2 = _serial(
            mff, child2, "1908-c2", parent_branch_id="1908-p", position_embeddings=pe_cuda
        )
    assert len(_user_warning_messages(record_c2)) == 0
    assert mff.graph_runner is not None
    reference2 = _raw_forward(model, child2, pe_cuda)
    assert torch.allclose(out2, reference2, atol=1e-4, rtol=1e-4)
