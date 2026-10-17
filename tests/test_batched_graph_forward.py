"""Red-phase tests for AR016: batched CUDA-graph dispatch (UT-1501..UT-1509).

Contract: ``specs/changes/AR016-batched-graph-verification/design.md`` §3 and
``srs.md`` §3.4 FR4 — ``ManualFoldedForward.forward`` must route batched
``branch_ids`` calls through a dedicated ``graph_runner_batched`` capture /
replay path (mirroring the serial AR002 wiring) whenever the batched
preconditions hold, publish per-branch row slices after every validated
replay, and fall back to the untouched AR015 eager batched path otherwise.

Red state (why the new-behavior assertions fail today): the graph gate at
``architecture_utils.py`` (:486, ``elif self.use_cuda_graph and branch_id is
not None``) only covers serial calls, so batched calls silently bypass the
graph; ``graph_runner_batched`` / ``_graph_forward_batched_or_none`` /
``_parent_cache_complete_batched`` / ``_publish_graph_children`` /
``_graph_batched_capture_failed`` do not exist yet. Pure eager-contract
assertions (UT-1502 publish completeness, UT-1506's result-equality
sub-cases, UT-1508(a)) must already pass and must stay green after the
Green phase.

Fixtures follow ``tests/test_cuda_graph.py`` and
``tests/test_batched_branch_forward.py``: a self-contained LLaMA-layout
decoder (L=2, H=32, T=16, V=64) on CUDA, one raw model shared by the graph
stack and the eager-reference stack (identical weights, disjoint caches,
gates and branch-id prefixes), N=4 equal-length candidates diverging
mid-sequence, ``SimilarityGate(tau=0.95, metric="cosine")``, no scheduler,
and the autouse ``seed`` fixture plus explicit builder seeds. The default
cache is ``ActivationCache``; UT-1508 uses ``ChunkedActivationCache``
(design §3, gate Major-N1) and UT-1509 does too (see its docstring for the
publish-isolation observability rationale).

Tolerances (design §3): graph-vs-eager logits ``allclose(atol=1e-4,
rtol=1e-4)`` plus per-row argmax equality; per-branch cached ``ffn_out``
1e-4; layer-0 ``embedding`` rows 1e-6; per-branch ``mean_stable_ratio``
1e-6 (fused-gate vs plain-gate threshold-adjacent flips are covered by the
srs erratum clause); masks are never asserted (ULP differences propagate
through layers — informational only). The whole file is CUDA-gated: graph
capture is a CUDA-only semantic with nothing to test on CPU.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.chunked_cache import ChunkedActivationCache
from actfold.core.cuda_graph import FoldedGraphRunner
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.models.architecture_utils import ManualFoldedForward, detect_architecture
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_VOCAB = 64
_HIDDEN = 32
_NUM_LAYERS = 2
_SEQ = 16
_NUM_CANDIDATES = 4
_TAU = 0.95
_CACHE_BUDGET = 1024


# ---------------------------------------------------------------------------
# Self-contained synthetic decoder (LlamaLikeModel layout, test_cuda_graph
# precedent): token-wise maskable attention makes attention_mask forms
# numerically observable, and the post_attention_layernorm -> mlp chain is
# detectable by the split-layer wrapper engaged during graph capture.
# ---------------------------------------------------------------------------


class _MaskableAttention(nn.Module):
    """Token-wise attention stand-in consuming the mask multiplicatively."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        out = hidden_states * 0.5
        if attention_mask is not None:
            out = out * attention_mask.to(out.dtype).unsqueeze(-1)
        return out, None


class _LlamaLikeLayer(nn.Module):
    """Pre-norm decoder layer with a detectable split FFN chain."""

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
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class _LlamaLikeModel(nn.Module):
    """Mock LLaMA-layout decoder: embed -> layers -> model.norm -> lm_head."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": _VOCAB})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.model.layers = nn.ModuleList([_LlamaLikeLayer(_HIDDEN) for _ in range(_NUM_LAYERS)])
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


def _build_model(device: str, seed: int = 2026) -> tuple[nn.Module, Any]:
    """Deterministically build the decoder and its architecture profile."""
    torch.manual_seed(seed)
    model = _LlamaLikeModel().to(device)
    model.eval()
    profile = detect_architecture(model)
    return model, profile


def _make_mff(
    model: nn.Module,
    device: str,
    gate: SimilarityGate | None = None,
    scheduler: FoldingScheduler | None = None,
    cache: Any = None,
    use_cuda_graph: bool = False,
    graph_capacity_ratio: float = 0.5,
) -> ManualFoldedForward:
    """Fresh ManualFoldedForward (own cache + gate) over the shared model.

    Follows the test_cuda_graph.py wiring: split layers with a lowered
    ``split_min_tokens`` so the split FFN path engages at the test shapes
    (the graph's padded capacity-C gather vs the eager exact gather is the
    mechanistic source of the 1e-4 logits tolerance), plus the graph flags.
    """
    if cache is None:
        cache = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    if gate is None:
        gate = SimilarityGate(tau=_TAU, metric="cosine")
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        scheduler=scheduler,
        split_layers=True,
        split_min_tokens=1,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=graph_capacity_ratio,
    )


def _make_candidates(
    device: str,
    changes_per_row: int = 1,
    seed: int = 4201,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a parent ``[1, T]`` and N equal-length mid-divergent candidates.

    Each candidate row shares the parent's length and replaces
    ``changes_per_row`` consecutive mid-sequence tokens with mutually
    distinct ids, so every layer's gate mask is mixed (stable prefix and
    suffix, a few divergent positions per row) and the folded slow path —
    the publish-observing path — engages on both the eager and graph runs.
    """
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
    attention_mask: torch.Tensor | None = None,
    step_idx: int = 1,
) -> torch.Tensor:
    """One serial ``branch_id`` forward under ``no_grad``."""
    with torch.no_grad():
        return mff(
            tokens,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            attention_mask=attention_mask,
            step_idx=step_idx,
        )


def _batched(
    mff: ManualFoldedForward,
    stacked: torch.Tensor,
    branch_ids: list[str],
    parent_branch_id: str,
    step_idx: int = 1,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One batched ``branch_ids`` forward under ``no_grad``."""
    with torch.no_grad():
        return mff(
            stacked,
            branch_id=None,
            branch_ids=branch_ids,
            parent_branch_id=parent_branch_id,
            step_idx=step_idx,
            attention_mask=attention_mask,
        )


def _batched_eager_reference(
    model: nn.Module,
    device: str,
    parent: torch.Tensor,
    stacked: torch.Tensor,
    branch_ids: list[str],
    parent_branch_id: str,
    gate: SimilarityGate | None = None,
    scheduler: FoldingScheduler | None = None,
    populate: bool = True,
) -> tuple[ManualFoldedForward, torch.Tensor]:
    """Run the AR015 eager batched path once on a fresh stack (no graph).

    A fresh ManualFoldedForward over the shared raw model carries its own
    cache and gate, so the reference can never observe the graph stack's
    activations. ``populate=False`` serves the parent-empty (cache-miss)
    parity branch, where the shared parent id simply has no cache entry.

    Returns:
        ``(mff, logits)`` — the eager stack (for cache/profile comparisons)
        and the batched-eager logits.
    """
    mff = _make_mff(model, device, gate=gate, scheduler=scheduler)
    if populate:
        _serial(mff, parent, branch_id=parent_branch_id)
    logits = _batched(mff, stacked, branch_ids, parent_branch_id)
    return mff, logits


def _user_warning_messages(record: Any) -> list[str]:
    """Return the messages of every UserWarning in a catch_warnings log."""
    return [str(entry.message) for entry in record if issubclass(entry.category, UserWarning)]


# ---------------------------------------------------------------------------
# UT-1501: batched capture + replay vs the AR015 eager batched path
# ---------------------------------------------------------------------------


def test_ut1501_batched_capture_replay_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-1501 (FR1/FR4): first eligible batched call captures, second replays.

    Against a fresh eager stack (identical weights, no graph wiring): the
    replay logits match the AR015 batched-eager logits within
    ``allclose(1e-4)`` plus per-row argmax equality; the per-branch cached
    ``ffn_out`` matches within 1e-4; the per-branch profiler statistic
    ``mean_stable_ratio`` matches within 1e-6. The capture/replay split is
    pinned by a class-level ``FoldedGraphRunner.capture`` spy: exactly one
    capture across both same-shape calls (also the mutation-① observation
    face — deleting the dispatch drops the count to zero).
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4201)

    capture_calls = {"count": 0}
    original_capture = FoldedGraphRunner.capture

    def _counting_capture(self: Any, *args: Any, **kwargs: Any) -> None:
        capture_calls["count"] += 1
        return original_capture(self, *args, **kwargs)

    monkeypatch.setattr(FoldedGraphRunner, "capture", _counting_capture)

    mff_eager, logits_eager = _batched_eager_reference(
        model, device, parent, stacked, _ids("1501-e"), "1501-ep"
    )
    eager_ids = _ids("1501-e")

    mff = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff, parent, branch_id="1501-gp")
    out1 = _batched(mff, stacked, _ids("1501-b"), "1501-gp")
    assert capture_calls["count"] == 1, "first eligible batched call must capture once"
    assert mff.graph_runner_batched is not None
    runner = mff.graph_runner_batched
    assert tuple(runner.tokens_static.shape) == (_NUM_CANDIDATES, _SEQ)

    replay_ids = _ids("1501-r")
    out2 = _batched(mff, stacked, replay_ids, "1501-gp", step_idx=2)
    assert capture_calls["count"] == 1, "second same-shape call must replay, not recapture"
    assert mff.graph_runner_batched is runner

    assert torch.allclose(out1, logits_eager, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out2, logits_eager, atol=1e-4, rtol=1e-4)
    assert torch.equal(out2.argmax(dim=-1), logits_eager.argmax(dim=-1))

    for i in range(_NUM_CANDIDATES):
        for layer_idx in range(_NUM_LAYERS):
            ffn_graph = mff.cache.fetch(branch_id=replay_ids[i], layer_idx=layer_idx)["ffn_out"]
            ffn_eager = mff_eager.cache.fetch(branch_id=eager_ids[i], layer_idx=layer_idx)[
                "ffn_out"
            ]
            assert torch.allclose(ffn_graph, ffn_eager, atol=1e-4, rtol=1e-4)
        profile_graph = GLOBAL_STABILITY_PROFILER.get_profile(replay_ids[i])
        profile_eager = GLOBAL_STABILITY_PROFILER.get_profile(eager_ids[i])
        assert profile_graph is not None, "graph publish must profile every branch"
        assert profile_eager is not None, "eager reference must profile every branch"
        assert math.isclose(
            profile_graph.mean_stable_ratio,
            profile_eager.mean_stable_ratio,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )


# ---------------------------------------------------------------------------
# UT-1502: per-branch publish completeness
# ---------------------------------------------------------------------------


def test_ut1502_per_branch_publish_completeness() -> None:
    """UT-1502 (FR1/FR4): every published branch carries the full activation set.

    After the capture call and after a second (replay) call, every branch
    cache entry must contain ``ffn_out`` at all L layers plus the layer-0
    ``embedding``, and each stored embedding row must equal the
    ``embed_module(tokens)`` row within 1e-6 — a publish contract the
    eager AR015 path already satisfies (so this UT is green in the Red
    state) and the graph publish must keep satisfying. Rejected-branch
    semantics are out of scope here (the graph layer never prunes).
    """
    device = "cuda"
    model, profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4202)

    mff = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff, parent, branch_id="1502-p")
    with torch.no_grad():
        embedding_rows = profile.embed_module(stacked)

    for tag in ("1502-a", "1502-b"):
        branch_ids = _ids(tag)
        out = _batched(mff, stacked, branch_ids, "1502-p")
        assert tuple(out.shape) == (_NUM_CANDIDATES, _SEQ, _VOCAB)
        for i, bid in enumerate(branch_ids):
            for layer_idx in range(_NUM_LAYERS):
                entry = mff.cache.fetch(branch_id=bid, layer_idx=layer_idx)
                assert "ffn_out" in entry, f"{bid} layer {layer_idx} lacks ffn_out"
                assert tuple(entry["ffn_out"].shape) == (1, _SEQ, _HIDDEN)
                if layer_idx == 0:
                    assert "embedding" in entry, f"{bid} lacks the layer-0 embedding"
                    assert tuple(entry["embedding"].shape) == (1, _SEQ, _HIDDEN)
                    assert torch.allclose(entry["embedding"], embedding_rows[i : i + 1], atol=1e-6)


# ---------------------------------------------------------------------------
# UT-1503: serial/batched runner isolation + FR3 literal comparison
# ---------------------------------------------------------------------------


def test_ut1503_serial_batched_runner_isolation() -> None:
    """UT-1503 (FR1/FR2/FR4): serial and batched graph runners coexist.

    On one ``use_cuda_graph=True`` instance: a batched call captures the
    [N, T] workload into ``graph_runner_batched``; a later serial call
    captures the [1, T] workload into ``graph_runner`` — two distinct
    runner objects with different pinned shapes — and subsequent replays
    of either kind stay correct and reuse their own runner. The FR3
    literal comparison (design gate Minor-N2) closes the m-6 transitivity
    argument: each row of the batched graph replay matches the serial
    graph replay of the same candidate row against the same parent,
    directly, with no eager intermediary.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4203)

    _mff_eager, logits_eager = _batched_eager_reference(
        model, device, parent, stacked, _ids("1503-e"), "1503-ep"
    )

    mff = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff, parent, branch_id="1503-p")

    out_b1 = _batched(mff, stacked, _ids("1503-b"), "1503-p")
    assert torch.allclose(out_b1, logits_eager, atol=1e-4, rtol=1e-4)
    assert mff.graph_runner_batched is not None
    runner_b = mff.graph_runner_batched
    assert tuple(runner_b.tokens_static.shape) == (_NUM_CANDIDATES, _SEQ)

    out_b2 = _batched(mff, stacked, _ids("1503-c"), "1503-p")
    assert torch.allclose(out_b2, logits_eager, atol=1e-4, rtol=1e-4)

    out_s1 = _serial(mff, stacked[0:1], branch_id="1503-s0", parent_branch_id="1503-p")
    assert torch.allclose(out_s1, logits_eager[0:1], atol=1e-4, rtol=1e-4)
    assert mff.graph_runner is not None
    runner_s = mff.graph_runner
    assert runner_s is not runner_b
    assert tuple(runner_s.tokens_static.shape) == (1, _SEQ)

    out_s2 = _serial(mff, stacked[1:2], branch_id="1503-s1", parent_branch_id="1503-p")
    assert torch.allclose(out_s2, logits_eager[1:2], atol=1e-4, rtol=1e-4)
    assert mff.graph_runner is runner_s

    out_b3 = _batched(mff, stacked, _ids("1503-d"), "1503-p")
    assert mff.graph_runner_batched is runner_b
    assert torch.allclose(out_b3, logits_eager, atol=1e-4, rtol=1e-4)
    assert torch.equal(out_b3.argmax(dim=-1), logits_eager.argmax(dim=-1))

    for i in range(_NUM_CANDIDATES):
        out_si = _serial(mff, stacked[i : i + 1], branch_id=f"1503-x{i}", parent_branch_id="1503-p")
        assert torch.allclose(out_b3[i : i + 1], out_si, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# UT-1504: post-capture shape mismatch
# ---------------------------------------------------------------------------


def test_ut1504_shape_mismatch_falls_back_eager() -> None:
    """UT-1504 (FR2/FR4): a post-capture shape change warns once, then eager.

    After the [N, T] capture, a legal [2, T] batched call (a valid batch,
    just a different shape) must emit exactly one UserWarning carrying
    both shapes, never re-capture, and return the batched-eager result; a
    second mismatched call stays silent with the same correct result; and
    the pinned [N, T] runner keeps serving same-shape replays.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4204)
    narrow = stacked[:2]
    narrow_ids = _ids("1504-w", 2)

    _mff_eager_w, ref_wide = _batched_eager_reference(
        model, device, parent, stacked, _ids("1504-e"), "1504-ep"
    )
    _mff_eager_n, ref_narrow = _batched_eager_reference(
        model, device, parent, narrow, narrow_ids, "1504-ep"
    )

    mff = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff, parent, branch_id="1504-p")
    with warnings.catch_warnings(record=True) as record_capture:
        warnings.simplefilter("always")
        out_wide = _batched(mff, stacked, _ids("1504-a"), "1504-p")
    assert torch.allclose(out_wide, ref_wide, atol=1e-4, rtol=1e-4)
    assert len(_user_warning_messages(record_capture)) == 0

    with warnings.catch_warnings(record=True) as record_narrow:
        warnings.simplefilter("always")
        out_narrow = _batched(mff, narrow, narrow_ids, "1504-p")
    assert torch.allclose(out_narrow, ref_narrow, atol=1e-4, rtol=1e-4)
    messages = _user_warning_messages(record_narrow)
    assert len(messages) == 1, "the first shape mismatch must warn exactly once"
    assert "(4, 16)" in messages[0], f"captured shape missing from: {messages[0]!r}"
    assert "(2, 16)" in messages[0], f"incoming shape missing from: {messages[0]!r}"

    with warnings.catch_warnings(record=True) as record_again:
        warnings.simplefilter("always")
        out_again = _batched(mff, narrow, _ids("1504-v", 2), "1504-p")
    assert len(_user_warning_messages(record_again)) == 0
    assert torch.allclose(out_again, ref_narrow, atol=1e-4, rtol=1e-4)

    assert mff.graph_runner_batched is not None
    out_replay = _batched(mff, stacked, _ids("1504-c"), "1504-p")
    assert torch.allclose(out_replay, ref_wide, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# UT-1505: divergent-budget exceeded
# ---------------------------------------------------------------------------


def test_ut1505_budget_exceeded_discards_and_recomputes() -> None:
    """UT-1505 (FR2/FR4): an exceeded divergent budget discards the replay.

    With ``graph_capacity_ratio=0.02`` (capacity C = ceil(0.02 * N * T) = 2)
    and candidates diverging at three mid-sequence positions per row
    (D = 12 > C at every layer), the replay output is untrustworthy: the
    call must warn exactly once, discard the replay, recompute on the
    eager batched path (result == eager reference — correctness first),
    and leave ``runner.budget_exceeded`` set. A second over-budget call
    stays silent and still returns the correct eager result.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, changes_per_row=3, seed=4205)

    _mff_eager, ref = _batched_eager_reference(
        model, device, parent, stacked, _ids("1505-e"), "1505-ep"
    )

    mff = _make_mff(model, device, use_cuda_graph=True, graph_capacity_ratio=0.02)
    _serial(mff, parent, branch_id="1505-p")
    with warnings.catch_warnings(record=True) as record_first:
        warnings.simplefilter("always")
        out = _batched(mff, stacked, _ids("1505-a"), "1505-p")
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    messages = _user_warning_messages(record_first)
    assert len(messages) == 1, "the first budget violation must warn exactly once"
    assert "budget" in messages[0].lower(), f"unexpected budget warning: {messages[0]!r}"
    runner = mff.graph_runner_batched
    assert runner is not None
    assert runner.budget_exceeded is True

    with warnings.catch_warnings(record=True) as record_second:
        warnings.simplefilter("always")
        out2 = _batched(mff, stacked, _ids("1505-b"), "1505-p")
    assert len(_user_warning_messages(record_second)) == 0
    assert torch.allclose(out2, ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# UT-1506: precondition degradation matrix
# ---------------------------------------------------------------------------


def test_ut1506_precondition_matrix() -> None:
    """UT-1506 (FR2/FR4): the batched-graph precondition degradation matrix.

    Every sub-case builds a fresh ``use_cuda_graph=True`` stack (the
    degradation-family warning flag is one shared boolean per instance)
    and asserts the eager-contract outcome before any graph-path probe:

    * scheduler attached / non-cosine gate: no capture, result equal to
      the batched-eager run under the same configuration;
    * CPU tokens: the same eager parity plus the one-time degradation
      UserWarning shared with the serial family;
    * empty parent cache: silent whole-batch recompute (cache-miss
      parity), zero warnings, no capture;
    * attention-mask object mismatch: the capturing call pins mask A; a
      later call with a different mask object silently degrades, genuinely
      consumes the new mask, and leaves the runner state untouched (a
      later pinned-mask call still replays);
    * parent cached at batch 2: silent graph rejection (zero warnings, no
      capture) and the eager #41 contract raising RuntimeError.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4206)

    # (a) FoldingScheduler attached: no capture, eager parity.
    sched_eager = FoldingScheduler(base_tau=_TAU, num_layers=_NUM_LAYERS, num_steps=2)
    sched_graph = FoldingScheduler(base_tau=_TAU, num_layers=_NUM_LAYERS, num_steps=2)
    _mff_a, ref_a = _batched_eager_reference(
        model,
        device,
        parent,
        stacked,
        _ids("1506-ea"),
        "1506-eap",
        scheduler=sched_eager,
    )
    mff_a = _make_mff(model, device, scheduler=sched_graph, use_cuda_graph=True)
    _serial(mff_a, parent, branch_id="1506-pa")
    out_a = _batched(mff_a, stacked, _ids("1506-ba"), "1506-pa")
    assert torch.allclose(out_a, ref_a, atol=1e-6, rtol=1e-6)
    assert getattr(mff_a, "graph_runner_batched", None) is None

    # (b) Non-cosine gate: no capture, eager parity.
    gate_l2 = SimilarityGate(tau=_TAU, metric="l2")
    _mff_b, ref_b = _batched_eager_reference(
        model,
        device,
        parent,
        stacked,
        _ids("1506-eb"),
        "1506-ebp",
        gate=SimilarityGate(tau=_TAU, metric="l2"),
    )
    mff_b = _make_mff(model, device, gate=gate_l2, use_cuda_graph=True)
    _serial(mff_b, parent, branch_id="1506-pb")
    out_b = _batched(mff_b, stacked, _ids("1506-bb"), "1506-pb")
    assert torch.allclose(out_b, ref_b, atol=1e-6, rtol=1e-6)
    assert getattr(mff_b, "graph_runner_batched", None) is None

    # (c) CPU tokens: eager parity plus the one-time degradation warning.
    cpu_model, _cpu_profile = _build_model("cpu")
    cpu_parent, cpu_stacked = _make_candidates("cpu", seed=4206)
    _mff_c, ref_c = _batched_eager_reference(
        cpu_model, "cpu", cpu_parent, cpu_stacked, _ids("1506-ec"), "1506-ecp"
    )
    mff_c = _make_mff(cpu_model, "cpu", use_cuda_graph=True)
    _serial(mff_c, cpu_parent, branch_id="1506-pc")
    with warnings.catch_warnings(record=True) as record_c:
        warnings.simplefilter("always")
        out_c = _batched(mff_c, cpu_stacked, _ids("1506-bc"), "1506-pc")
    assert torch.allclose(out_c, ref_c, atol=1e-6, rtol=1e-6)
    assert len(_user_warning_messages(record_c)) == 1
    assert getattr(mff_c, "graph_runner_batched", None) is None

    # (d) Empty parent cache: silent cache-miss whole-batch recompute.
    _mff_d, ref_d = _batched_eager_reference(
        model,
        device,
        parent,
        stacked,
        _ids("1506-ed"),
        "1506-ghost",
        populate=False,
    )
    mff_d = _make_mff(model, device, use_cuda_graph=True)
    with warnings.catch_warnings(record=True) as record_d:
        warnings.simplefilter("always")
        out_d = _batched(mff_d, stacked, _ids("1506-bd"), "1506-ghost")
    assert torch.allclose(out_d, ref_d, atol=1e-6, rtol=1e-6)
    assert len(_user_warning_messages(record_d)) == 0
    assert getattr(mff_d, "graph_runner_batched", None) is None

    # (e) Attention-mask object mismatch: pinned mask A, silent mask B.
    mask_a = 0.5 * torch.ones(1, _SEQ, device=device)
    mask_b = torch.zeros(1, _SEQ, device=device)
    mff_eager_e = _make_mff(model, device)
    _serial(mff_eager_e, parent, branch_id="1506-eep", attention_mask=mask_a)
    ref_ea = _batched(mff_eager_e, stacked, _ids("1506-eea"), "1506-eep", attention_mask=mask_a)
    ref_eb = _batched(mff_eager_e, stacked, _ids("1506-eeb"), "1506-eep", attention_mask=mask_b)
    mff_e = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff_e, parent, branch_id="1506-pe", attention_mask=mask_a)
    with warnings.catch_warnings(record=True) as record_capture:
        warnings.simplefilter("always")
        out_capture = _batched(mff_e, stacked, _ids("1506-be"), "1506-pe", attention_mask=mask_a)
    assert torch.allclose(out_capture, ref_ea, atol=1e-4, rtol=1e-4)
    assert len(_user_warning_messages(record_capture)) == 0
    assert mff_e.graph_runner_batched is not None
    runner_e = mff_e.graph_runner_batched
    with warnings.catch_warnings(record=True) as record_mask_b:
        warnings.simplefilter("always")
        out_mask_b = _batched(mff_e, stacked, _ids("1506-bm"), "1506-pe", attention_mask=mask_b)
    assert len(_user_warning_messages(record_mask_b)) == 0
    assert torch.allclose(out_mask_b, ref_eb, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out_mask_b, ref_ea)
    assert mff_e.graph_runner_batched is runner_e
    out_pinned = _batched(mff_e, stacked, _ids("1506-bg"), "1506-pe", attention_mask=mask_a)
    assert torch.allclose(out_pinned, ref_ea, atol=1e-4, rtol=1e-4)

    # (f) Parent cached at batch 2: silent rejection + eager #41 error.
    parent_batch2 = torch.cat([parent, parent], dim=0)
    mff_f = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff_f, parent_batch2, branch_id="1506-pf")
    with warnings.catch_warnings(record=True) as record_f:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError):
            _batched(mff_f, stacked, _ids("1506-bf"), "1506-pf")
    assert len(_user_warning_messages(record_f)) == 0
    assert getattr(mff_f, "graph_runner_batched", None) is None


# ---------------------------------------------------------------------------
# UT-1507: capture failure permanently disables the batched path only
# ---------------------------------------------------------------------------


def test_ut1507_batched_capture_failure_permanent_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-1507 (FR2/FR4, EX-002 batched): a failed capture never retries.

    A one-shot ``FoldedGraphRunner.capture`` patch (fails on the first
    call, delegates to the real implementation afterwards — a persistent
    class-level failure would also break the later serial capture) makes
    the first eligible batched call fail its capture: the step still
    returns the correct eager result, ``_graph_batched_capture_failed`` is
    set, and the one-time warning names the capture failure. A second
    batched call neither retries the capture nor warns again. With the
    patch then delegating, a serial call on the SAME instance still
    captures (``graph_runner`` unaffected — the two disable flags and the
    two runners are independent).
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4207)

    capture_calls = {"count": 0}
    original_capture = FoldedGraphRunner.capture

    def _fail_first_capture(self: Any, *args: Any, **kwargs: Any) -> None:
        capture_calls["count"] += 1
        if capture_calls["count"] == 1:
            raise RuntimeError("injected batched capture failure")
        return original_capture(self, *args, **kwargs)

    monkeypatch.setattr(FoldedGraphRunner, "capture", _fail_first_capture)

    _mff_eager, ref = _batched_eager_reference(
        model, device, parent, stacked, _ids("1507-e"), "1507-ep"
    )

    mff = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff, parent, branch_id="1507-p")
    with warnings.catch_warnings(record=True) as record_first:
        warnings.simplefilter("always")
        out = _batched(mff, stacked, _ids("1507-a"), "1507-p")
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    assert mff._graph_batched_capture_failed is True
    messages = _user_warning_messages(record_first)
    assert len(messages) == 1, "the failed capture must warn exactly once"
    assert "capture failed" in messages[0].lower()
    assert capture_calls["count"] == 1
    assert getattr(mff, "graph_runner_batched", None) is None

    with warnings.catch_warnings(record=True) as record_second:
        warnings.simplefilter("always")
        out2 = _batched(mff, stacked, _ids("1507-b"), "1507-p")
    assert len(_user_warning_messages(record_second)) == 0
    assert capture_calls["count"] == 1, "EX-002: a failed capture is never retried"
    assert torch.allclose(out2, ref, atol=1e-4, rtol=1e-4)

    out_serial = _serial(mff, stacked[0:1], branch_id="1507-s0", parent_branch_id="1507-p")
    assert mff.graph_runner is not None
    assert mff._graph_capture_failed is False
    assert capture_calls["count"] == 2
    with torch.no_grad():
        raw_logits = model(stacked[0:1])
    assert torch.allclose(out_serial, raw_logits, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# UT-1508: parent batch != 1 — silent rejection at capture AND at replay
# ---------------------------------------------------------------------------


def test_ut1508_parent_batch_not_one_chunked_cache() -> None:
    """UT-1508 (FR1/FR2/FR4, Major-1): parent batch != 1 is rejected twice.

    This UT runs on ``ChunkedActivationCache`` (design §3, gate Major-N1):
    its whole-chunk replacement semantics let a serial same-id pass
    silently overwrite a [1, T, H] parent with [N, T, H] activations — the
    reachable production form of the replay-time hazard — while
    ``ActivationCache.put`` would surface the same overwrite as a
    RuntimeError instead. Both caches need the replay-time re-validation.

    (a) A parent cached at batch 2 followed by the first batched call:
    zero warnings, no capture, and the eager #41 contract raising
    RuntimeError (the graph layer silently rejects; it never swallows or
    rewrites the eager error).

    (b) After a successful capture, overwriting the parent with a serial
    [N, T] same-id pass makes the next same-shape batched call
    replay-rejected: zero warnings, RuntimeError from the eager layer
    BEFORE any store (cache/profiler zero pollution), the runner state
    untouched, and a re-populated [1, T] parent restoring clean replays.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4208)

    # (a) Parent cached at batch 2: silent rejection + eager RuntimeError.
    cache_a = ChunkedActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    mff_a = _make_mff(model, device, cache=cache_a, use_cuda_graph=True)
    parent_batch2 = torch.cat([parent, parent], dim=0)
    _serial(mff_a, parent_batch2, branch_id="1508-pa")
    with warnings.catch_warnings(record=True) as record_a:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError):
            _batched(mff_a, stacked, _ids("1508-ba"), "1508-pa")
    assert len(_user_warning_messages(record_a)) == 0
    assert getattr(mff_a, "graph_runner_batched", None) is None

    # (b) Capture, overwrite the parent at [N, T], replay gets rejected.
    cache = ChunkedActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    mff = _make_mff(model, device, cache=cache, use_cuda_graph=True)
    _mff_eager, ref = _batched_eager_reference(
        model, device, parent, stacked, _ids("1508-e"), "1508-ep"
    )
    _serial(mff, parent, branch_id="1508-pb")
    out1 = _batched(mff, stacked, _ids("1508-b"), "1508-pb")
    assert torch.allclose(out1, ref, atol=1e-4, rtol=1e-4)
    assert mff.graph_runner_batched is not None
    runner = mff.graph_runner_batched

    # Serial same-id pass silently replaces the [1, T, H] parent chunks
    # with [N, T, H] activations (chunked-cache semantics).
    _serial(mff, stacked, branch_id="1508-pb")
    rejected_ids = _ids("1508-r")
    with warnings.catch_warnings(record=True) as record_reject:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError):
            _batched(mff, stacked, rejected_ids, "1508-pb")
    assert len(_user_warning_messages(record_reject)) == 0
    for bid in rejected_ids:
        for layer_idx in range(_NUM_LAYERS):
            assert not cache.contains(bid, layer_idx), f"{bid} polluted the cache"
        assert GLOBAL_STABILITY_PROFILER.get_profile(bid) is None
    parent_embedding = cache.fetch(branch_id="1508-pb", layer_idx=0)["embedding"]
    assert tuple(parent_embedding.shape[:2]) == (_NUM_CANDIDATES, _SEQ)
    assert mff.graph_runner_batched is runner

    # Re-populate the parent at [1, T] and the replay works again.
    _serial(mff, parent, branch_id="1508-pb")
    out2 = _batched(mff, stacked, _ids("1508-c"), "1508-pb")
    assert torch.allclose(out2, ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# UT-1509: publish isolation from the static child buffers
# ---------------------------------------------------------------------------


def test_ut1509_publish_isolation_from_static_buffers() -> None:
    """UT-1509 (FR1/FR4, mutation ②): published items never alias child_buf.

    Red-phase fixture note (documented deviation from design §3's default
    cache): this UT also runs on ``ChunkedActivationCache``. The design
    fixes the default at ``ActivationCache`` except UT-1508, but
    ``ActivationCache.put`` copies values into private buffers at put
    time, so a skipped ``clone`` would be unobservable through it; the
    chunked cache's ``.contiguous()`` no-op on a contiguous row view
    stores the alias itself, making "publish must clone the row slice"
    observably falsifiable (design mutation ②: skipping the clone lets
    the next replay's child_buf overwrite leak into earlier publishes).

    After a capture-and-publish call, an in-place mutation of
    ``runner.child_buf`` must not change any already-published cache item,
    and a second replay (which overwrites the whole static child buffer)
    must leave the first publish intact.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4209)

    cache = ChunkedActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    mff = _make_mff(model, device, cache=cache, use_cuda_graph=True)
    _serial(mff, parent, branch_id="1509-p")
    first_ids = _ids("1509-a")
    second_ids = _ids("1509-b")
    _batched(mff, stacked, first_ids, "1509-p")
    assert mff.graph_runner_batched is not None
    runner = mff.graph_runner_batched

    snapshot: dict[tuple[str, int], torch.Tensor] = {}
    for bid in first_ids:
        for layer_idx in range(_NUM_LAYERS):
            snapshot[(bid, layer_idx)] = cache.fetch(branch_id=bid, layer_idx=layer_idx)[
                "ffn_out"
            ].clone()

    with torch.no_grad():
        runner.child_buf.add_(123.0)
    for bid in first_ids:
        for layer_idx in range(_NUM_LAYERS):
            published = cache.fetch(branch_id=bid, layer_idx=layer_idx)["ffn_out"]
            assert torch.equal(
                published, snapshot[(bid, layer_idx)]
            ), "mutating child_buf must not touch published cache items"

    _batched(mff, stacked, second_ids, "1509-p")
    for bid in first_ids:
        for layer_idx in range(_NUM_LAYERS):
            published = cache.fetch(branch_id=bid, layer_idx=layer_idx)["ffn_out"]
            assert torch.equal(
                published, snapshot[(bid, layer_idx)]
            ), "the second replay must not overwrite the first publish"
    for bid in second_ids:
        for layer_idx in range(_NUM_LAYERS):
            assert cache.contains(bid, layer_idx)
