"""AR015 tests: batched branch forward (``branch_ids`` contract), UT-1401..UT-1412.

New tests for the ``folded_generate`` multi-candidate batched branch forward
(specs/changes/AR015-batched-branch-verification, srs FR4):

- UT-1401: CPU varlen batched vs serial bit-exact parity (logits, per-branch
  cache, per-branch profiler statistics).
- UT-1402: CUDA varlen parity within allclose(1e-6) plus argmax token equality.
- UT-1403: equal-length candidates with split FFN layers, CPU
  allclose(1e-6) + argmax parity (T002 gemv/gemm erratum — NOT bit-exact).
- UT-1404: launch-count proxy — one batched call runs L layer calls vs N*L
  serial ones (no wall-clock assertions).
- UT-1405: ``folded_generate`` end-to-end — the default (batched) run equals a
  forced-serial run; rejected branches leave no activation-cache residue.
- UT-1406: serial fallback matrix — the dispatch predicate rejects every
  unsupported batch; greedy single-candidate generation stays serial.
- UT-1407: layer-level precondition contract (length mismatch, singular and
  plural branch ids given together, parent cache batch != 1) plus no-parent
  and cache-miss parity branches.
- UT-1408: CUDA-graph bypass — batched calls stay eager (no capture) and a
  later serial call on the same instance still captures and replays.
- UT-1409: scheduler-disabled layers recompute the whole batch per branch;
  the folding layer reuses the dynamic-tau logic (review-gate addition).
- UT-1410: the all-stable three-way branch shares the expanded parent FFN
  across branches without recompute (review-gate addition).
- UT-1411: AR007 truncated-prefix donation — a longer parent feeds an
  all-stable batched fast path (review-gate addition).
- UT-1412: AR007 truncated parent FFN inside the mixed merge path
  (review-gate addition).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, FoldedModel, FoldingScheduler, SimilarityGate
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.eval.generation_utils import greedy_generate
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.profiler.stability_profiler import (
    GLOBAL_STABILITY_PROFILER,
    StabilityProfile,
)
from actfold.speculative import folded_generation
from actfold.speculative.branch_tree import BranchNode
from actfold.speculative.draft_generator import DraftGenerator
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
from actfold.speculative.folded_generation import folded_generate

_VOCAB = 32
_HIDDEN = 32
_NUM_LAYERS = 3
_NUM_CANDIDATES = 4
_PARENT_LEN = 6
_SEED = 2026
_T_LAYER = 6


class _CausalBatchLayer(nn.Module):
    """Causal synthetic layer with a split-detectable token-wise FFN chain.

    The ``cumsum`` scan is prefix-stable (position ``t`` only accumulates
    positions ``<= t``), so a child extending a parent reproduces the parent's
    prefix hidden states bit-for-bit — the mathematical basis for prefix
    folding on this model.  The ``post_attention_layernorm -> mlp`` attribute
    names are in :func:`actfold.core.split_layer.detect_split_spec`'s candidate
    table, so ``split_layers=True`` engages the split FFN path.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cum = torch.cumsum(hidden_states, dim=1)
        denom = torch.arange(1, hidden_states.shape[1] + 1, device=hidden_states.device).view(
            1, -1, 1
        )
        h = hidden_states + 0.1 * cum / denom
        return h + self.mlp(self.post_attention_layernorm(h))


class _CausalBatchModel(nn.Module):
    """Causal synthetic decoder with a detectable embedding/layers/head layout.

    The attribute names (``embedding`` / ``layers`` / ``head``) are all in the
    :func:`detect_architecture` default path tables, so
    :class:`ManualFoldedForward` auto-discovers the stack without explicit
    wiring.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(_CausalBatchLayer(hidden_dim) for _ in range(num_layers))
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _build_setup(
    device: str = "cpu",
    split_layers: bool = False,
    use_cuda_graph: bool = False,
    split_min_tokens: int = 512,
    seed: int = _SEED,
    scheduler: FoldingScheduler | None = None,
) -> tuple[
    nn.Module,
    FastDLLMAdapter,
    ManualFoldedForward,
    ActivationCache,
    SimilarityGate,
]:
    """Build a fresh causal model plus its manual folded stack on ``device``.

    Two setups built with the same ``seed`` share identical weights but carry
    fully independent caches, gates, and folded wrappers, so a batched run and
    a serial run can never observe each other's cached activations.

    Args:
        device: Target device string.
        split_layers: Wrap layers with ``SplitFoldedTransformerLayer``.
        use_cuda_graph: Opt the folded wrapper into CUDA-graph capture.
        split_min_tokens: Minimum ``batch * seq`` for the split FFN path.
        seed: Seed for the synthetic model weights.
        scheduler: Optional ``FoldingScheduler`` attached to the folded stack.

    Returns:
        ``(raw, adapter, folded, cache, gate)`` with a shared raw model.
    """
    torch.manual_seed(seed)
    raw = _CausalBatchModel(_VOCAB, _HIDDEN, _NUM_LAYERS).to(device)
    raw.eval()
    cache = ActivationCache(max_entries_per_layer=256, device=device)
    gate = SimilarityGate(tau=0.95)
    folded = ManualFoldedForward(
        raw,
        cache=cache,
        gate=gate,
        scheduler=scheduler,
        split_layers=split_layers,
        split_min_tokens=split_min_tokens,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=0.5,
    )
    adapter = FastDLLMAdapter(raw, folded_model=folded, num_layers=_NUM_LAYERS, hidden_dim=_HIDDEN)
    return raw, adapter, folded, cache, gate


def _make_varlen_candidates(
    device: str,
    parent_len: int = _PARENT_LEN,
    num: int = _NUM_CANDIDATES,
    seed: int = 4100,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build a parent ``[1, parent_len]`` and ``num`` varlen children.

    Each candidate is ``[1, parent_len + 1]``: the parent prefix plus one
    appended token, with the appended tokens mutually distinct across the
    candidates — the generation-loop growth shape of AR003.
    """
    torch.manual_seed(seed)
    parent = torch.randint(1, _VOCAB, (1, parent_len), device=device)
    suffix = torch.arange(num, device=device).reshape(num, 1)
    stacked = torch.cat([parent.repeat(num, 1), suffix], dim=1)
    return parent, [stacked[i : i + 1] for i in range(num)]


def _make_equal_len_candidates(
    device: str,
    seq_len: int = 8,
    num: int = _NUM_CANDIDATES,
    seed: int = 4100,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build a parent ``[1, seq_len]`` and ``num`` equal-length children.

    Each candidate shares the parent's length and diverges at a mid-sequence
    position (mutually distinct replacement tokens), so the equal-length gate
    branch and the mixed three-way split both engage.
    """
    torch.manual_seed(seed)
    parent = torch.randint(1, _VOCAB, (1, seq_len), device=device)
    mid = seq_len // 2
    base_token = int(parent[0, mid].item())
    candidates: list[torch.Tensor] = []
    for i in range(num):
        child = parent.clone()
        child[0, mid] = (base_token + 1 + i) % _VOCAB
        candidates.append(child)
    return parent, candidates


def _make_all_stable_candidates(
    device: str,
    seq_len: int = 8,
    num: int = _NUM_CANDIDATES,
    seed: int = 4110,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build a parent ``[1, seq_len]`` and ``num`` token-identical children.

    Identical children make the gate mask all-True on every layer, driving
    the batched forward into the all-stable fast path of the three-way
    split.
    """
    torch.manual_seed(seed)
    parent = torch.randint(1, _VOCAB, (1, seq_len), device=device)
    return parent, [parent.clone() for _ in range(num)]


def _make_truncated_candidates(
    device: str,
    parent_len: int = 8,
    child_len: int = 6,
    num: int = _NUM_CANDIDATES,
    seed: int = 4111,
    divergent_tail: int = 0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build a longer parent and ``num`` prefix children (AR007 shape).

    Each candidate is the parent's first ``child_len`` tokens.  With
    ``divergent_tail=0`` every candidate is a true prefix (all-stable
    truncated donation); with ``divergent_tail > 0`` the last positions of
    each candidate carry mutually distinct replacement tokens, so the mixed
    merge runs against the truncated parent FFN.
    """
    torch.manual_seed(seed)
    parent = torch.randint(1, _VOCAB, (1, parent_len), device=device)
    prefix = parent[:, :child_len]
    candidates: list[torch.Tensor] = []
    for i in range(num):
        child = prefix.clone()
        for j in range(divergent_tail):
            pos = child_len - divergent_tail + j
            base = int(parent[0, pos].item())
            child[0, pos] = (base + 1 + i + j) % _VOCAB
        candidates.append(child)
    return parent, candidates


def _spy_recompute_all(folded: ManualFoldedForward) -> list[int]:
    """Install per-layer ``_recompute_all`` counters on ``folded`` (AR015).

    The instance-level spy delegates unchanged while counting calls per
    wrapped layer, making "the all-stable fast path does not recompute"
    (srs FR1) observable: a mutation that degrades the fast path to
    ``_recompute_all`` flips the counts even though the recomputed values of
    a true-prefix child are value-identical to the reused parent FFN.
    """
    counts = [0] * len(folded._wrapped_layers)
    for idx, layer in enumerate(folded._wrapped_layers):
        original = layer._recompute_all

        def _spy(*args: Any, _orig: Any = original, _idx: int = idx, **kwargs: Any) -> Any:
            counts[_idx] += 1
            return _orig(*args, **kwargs)

        layer._recompute_all = _spy  # type: ignore[assignment]
    return counts


def _install_layer_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Install a class-level launch-count spy on ``FoldedTransformerLayer.forward``.

    The spy delegates to the original implementation unchanged while recording
    the keyword arguments of every call, so it serves both as the launch-count
    probe (UT-1404) and as the ``branch_ids``-passing probe (UT-1405/UT-1406).
    """
    original_forward = FoldedTransformerLayer.forward
    calls: list[dict[str, Any]] = []

    def _spy_forward(
        self: FoldedTransformerLayer, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        calls.append(dict(kwargs))
        return original_forward(self, hidden_states, *args, **kwargs)

    monkeypatch.setattr(FoldedTransformerLayer, "forward", _spy_forward)
    return calls


def _track_cached_branches(cache: ActivationCache) -> list[Any]:
    """Record every branch id stored at layer 0 via instance-level spies.

    Wraps both publish faces: ``put`` (per-row, the switch-off and serial
    paths) and — when the cache is batch-capable — ``put_batch`` (the
    AR025 ``multi_parent_batch`` publish), whose ``CachePutEntry`` items
    are unwrapped per entry. Existing switch-off callers never trigger
    ``put_batch``, so their ``seen`` lists are unchanged.
    """
    seen: list[Any] = []
    original_put = cache.put

    def _spy_put(
        branch_id: Any,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        if layer_idx == 0 and branch_id not in seen:
            seen.append(branch_id)
        return original_put(
            branch_id=branch_id,
            layer_idx=layer_idx,
            activations=activations,
            step_idx=step_idx,
        )

    cache.put = _spy_put  # type: ignore[assignment]

    original_put_batch = getattr(cache, "put_batch", None)
    if callable(original_put_batch):

        def _spy_put_batch(entries: Any) -> None:
            for entry in entries:
                if entry.layer_idx == 0 and entry.branch_id not in seen:
                    seen.append(entry.branch_id)
            return original_put_batch(entries)

        cache.put_batch = _spy_put_batch  # type: ignore[assignment]
    return seen


def _assert_tensor_close(actual: torch.Tensor, expected: torch.Tensor, atol: float | None) -> None:
    """Assert bitwise equality (CPU discipline) or allclose within ``atol``."""
    if atol is None:
        assert torch.equal(actual, expected)
    else:
        assert torch.allclose(actual, expected, rtol=atol, atol=atol)


def _assert_float_close(actual: float, expected: float, atol: float | None) -> None:
    """Assert exact float equality (CPU) or closeness within ``atol``."""
    if atol is None:
        assert actual == expected
    else:
        assert math.isclose(actual, expected, rel_tol=atol, abs_tol=atol)


def _assert_profile_parity(
    profile_b: StabilityProfile | None,
    profile_s: StabilityProfile | None,
    atol: float | None,
) -> None:
    """Compare two per-branch stability profiles (statistics, not masks)."""
    assert profile_b is not None, "batched branch must carry a stability profile"
    assert profile_s is not None, "serial branch must carry a stability profile"
    assert len(profile_b.layer_stats) == len(profile_s.layer_stats)
    for stat_b, stat_s in zip(profile_b.layer_stats, profile_s.layer_stats):
        assert stat_b.layer_idx == stat_s.layer_idx
        assert stat_b.num_tokens == stat_s.num_tokens
        _assert_float_close(stat_b.stable_ratio, stat_s.stable_ratio, atol)
    _assert_float_close(profile_b.mean_stable_ratio, profile_s.mean_stable_ratio, atol)


def _assert_branch_parity(
    batched_ids: list[str],
    serial_ids: list[str],
    batched_logits: torch.Tensor,
    serial_logits: list[torch.Tensor],
    batched_cache: ActivationCache,
    serial_cache: ActivationCache,
    batched_profiles: dict[str, StabilityProfile | None],
    serial_profiles: dict[str, StabilityProfile | None],
    atol: float | None = None,
) -> None:
    """Compare a batched branch run against the per-candidate serial baseline.

    Row ``i`` of the batched output must match the ``i``-th serial candidate in
    logits (plus per-row argmax tokens when ``atol`` is set), per-branch
    per-layer cached activations (plus the layer-0 embedding), and per-branch
    profiler statistics.
    """
    assert batched_logits.shape[0] == len(serial_logits)
    for i, (bid, sid) in enumerate(zip(batched_ids, serial_ids)):
        _assert_tensor_close(batched_logits[i : i + 1], serial_logits[i], atol)
        if atol is not None:
            assert torch.equal(
                batched_logits[i].argmax(dim=-1),
                serial_logits[i][0].argmax(dim=-1),
            )
        for layer_idx in range(_NUM_LAYERS):
            entry_b = batched_cache.fetch(branch_id=bid, layer_idx=layer_idx)
            entry_s = serial_cache.fetch(branch_id=sid, layer_idx=layer_idx)
            _assert_tensor_close(entry_b["ffn_out"], entry_s["ffn_out"], atol)
            if layer_idx == 0:
                _assert_tensor_close(entry_b["embedding"], entry_s["embedding"], atol)
        _assert_profile_parity(batched_profiles[bid], serial_profiles[sid], atol)


def _run_batched_vs_serial_parity(
    device: str,
    tag: str,
    parent: torch.Tensor,
    candidates: list[torch.Tensor],
    atol: float | None = None,
    split_layers: bool = False,
    split_min_tokens: int = 512,
    scheduler: FoldingScheduler | None = None,
    after_parent: Callable[[ManualFoldedForward], Any] | None = None,
) -> tuple[
    ActivationCache,
    ActivationCache,
    dict[str, StabilityProfile | None],
    dict[str, StabilityProfile | None],
    SimilarityGate,
    Any,
]:
    """Run one batched ``branch_ids`` call against per-candidate serial calls.

    The serial baseline always runs first and uses only the current production
    API, so in the TDD Red state the failure is isolated to the batched call.
    Per the parity discipline (design §3.1): two fresh setups with identical
    weights, disjoint branch-id prefixes (``<tag>-b*`` vs ``<tag>-s*``, parents
    ``<tag>-pb`` vs ``<tag>-ps``), and per-branch profiles captured immediately
    after each side's run.

    Args:
        after_parent: Optional probe invoked on the batched setup right after
            its parent forward and immediately before the batched call (e.g.
            the ``_recompute_all`` spy); its return value is passed through.

    Returns:
        ``(cache_b, cache_s, batched_profiles, serial_profiles, gate_b,
        after_parent_result)`` so callers can pin path-specific extras beyond
        the shared parity face.
    """
    setup_kwargs: dict[str, Any] = {
        "split_layers": split_layers,
        "split_min_tokens": split_min_tokens,
        "scheduler": scheduler,
    }
    _raw_b, _adapter_b, folded_b, cache_b, gate_b = _build_setup(device, **setup_kwargs)
    _raw_s, _adapter_s, folded_s, cache_s, _gate_s = _build_setup(device, **setup_kwargs)
    batched_ids = [f"{tag}-b{i}" for i in range(_NUM_CANDIDATES)]
    serial_ids = [f"{tag}-s{i}" for i in range(_NUM_CANDIDATES)]

    folded_s(parent, branch_id=f"{tag}-ps")
    serial_logits = [
        folded_s(tokens, branch_id=serial_ids[i], parent_branch_id=f"{tag}-ps", step_idx=1)
        for i, tokens in enumerate(candidates)
    ]
    serial_profiles = {sid: GLOBAL_STABILITY_PROFILER.get_profile(sid) for sid in serial_ids}

    folded_b(parent, branch_id=f"{tag}-pb")
    after_parent_result = None if after_parent is None else after_parent(folded_b)
    stacked = torch.cat(candidates, dim=0)
    batched_logits = folded_b(
        stacked,
        branch_id=None,
        branch_ids=batched_ids,
        parent_branch_id=f"{tag}-pb",
        step_idx=1,
    )
    batched_profiles = {bid: GLOBAL_STABILITY_PROFILER.get_profile(bid) for bid in batched_ids}

    child_len = candidates[0].shape[1]
    assert batched_logits.shape == (_NUM_CANDIDATES, child_len, _VOCAB)
    _assert_branch_parity(
        batched_ids,
        serial_ids,
        batched_logits,
        serial_logits,
        cache_b,
        cache_s,
        batched_profiles,
        serial_profiles,
        atol,
    )
    return cache_b, cache_s, batched_profiles, serial_profiles, gate_b, after_parent_result


def test_ut1401_cpu_varlen_bitwise_parity() -> None:
    """UT-1401 (FR1/FR4): CPU varlen batched branch forward is bit-exact vs serial.

    A parent ``[1, T_p]`` fills the cache via a serial forward; four candidates
    ``[1, T_p + 1]`` with mutually distinct appended tokens run once as a single
    ``branch_ids`` batch and once as four serial child forwards on two fresh
    setups. Logits, per-branch caches (``ffn_out`` plus the layer-0
    ``embedding``), and per-branch profiler statistics must match bitwise.
    """
    device = "cpu"
    parent, candidates = _make_varlen_candidates(device, seed=4101)
    _run_batched_vs_serial_parity(device, "1401", parent, candidates)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ut1402_cuda_varlen_allclose_parity() -> None:
    """UT-1402 (FR1/FR4): CUDA varlen parity holds within allclose(1e-6).

    Same scenario as UT-1401 on CUDA: batched GEMM reduction-order differences
    are covered by ``allclose(rtol=1e-6, atol=1e-6)`` plus per-row argmax token
    equality for logits, cache activations, and profiler ratios.
    """
    device = "cuda"
    parent, candidates = _make_varlen_candidates(device, seed=4102)
    _run_batched_vs_serial_parity(device, "1402", parent, candidates, atol=1e-6)


def test_ut1403_equal_length_split_allclose_parity() -> None:
    """UT-1403 (FR1/FR2/FR4): equal-length candidates on split layers.

    Four candidates share the parent's length and diverge at a mid-sequence
    position, so the equal-length gate branch and the mixed three-way split
    both engage. ``split_min_tokens`` is lowered to the sequence length so the
    split FFN path activates for the serial ``[1, T]`` forwards as well as the
    batched ``[N, T]`` forward (the split threshold counts ``batch * seq``).

    Parity is ``allclose(1e-6)`` + per-row argmax token equality — NOT
    bit-exact (T002 erratum of the design's split-generalization
    assumption): the split FFN gathers the divergent rows of the whole
    batch into one GEMM (M = D_total), while the serial candidates gather
    per-candidate rows (M = D_i). When a serial candidate has exactly one
    divergent row (M=1 dispatches the BLAS gemv micro-kernel) the batched
    M>1 GEMM crosses the gemv/gemm boundary and differs by 1 ULP even on
    CPU. Semantics are identical; the masks and profiler statistics stay
    exactly equal (they derive from the gate masks, not the FFN values).
    """
    device = "cpu"
    seq_len = 8
    parent, candidates = _make_equal_len_candidates(device, seq_len=seq_len, seed=4103)
    _run_batched_vs_serial_parity(
        device,
        "1403",
        parent,
        candidates,
        atol=1e-6,
        split_layers=True,
        split_min_tokens=seq_len,
    )


def test_ut1404_launch_count_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1404 (FR3/FR4): batched evaluation launches L vs N*L layer calls.

    A class-level spy counts wrapped-layer forward calls (no wall-clock
    assertions, AGENTS #39): one batched ``branch_ids`` call must reach each of
    the L layers exactly once, while N serial candidate forwards reach them
    N*L times. The serial count is checked first so the Red-state failure is
    isolated to the batched call.
    """
    calls = _install_layer_spy(monkeypatch)
    _raw_s, _adapter_s, folded_s, _cache_s, _gate_s = _build_setup("cpu")
    parent, candidates = _make_varlen_candidates("cpu", seed=4104)
    serial_ids = [f"1404-s{i}" for i in range(_NUM_CANDIDATES)]

    folded_s(parent, branch_id="1404-ps")
    calls.clear()
    for i, tokens in enumerate(candidates):
        folded_s(tokens, branch_id=serial_ids[i], parent_branch_id="1404-ps", step_idx=1)
    assert len(calls) == _NUM_CANDIDATES * _NUM_LAYERS

    _raw_b, _adapter_b, folded_b, _cache_b, _gate_b = _build_setup("cpu")
    batched_ids = [f"1404-b{i}" for i in range(_NUM_CANDIDATES)]
    folded_b(parent, branch_id="1404-pb")
    calls.clear()
    batched_logits = folded_b(
        torch.cat(candidates, dim=0),
        branch_id=None,
        branch_ids=batched_ids,
        parent_branch_id="1404-pb",
        step_idx=1,
    )
    assert len(calls) == _NUM_LAYERS
    assert all(entry.get("branch_ids") is not None for entry in calls)
    assert batched_logits.shape == (_NUM_CANDIDATES, _PARENT_LEN + 1, _VOCAB)


def test_ut1405_folded_generate_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1405 (FR3/FR4): folded_generate batched run equals the forced-serial run.

    A draft-mode multi-candidate generation (``num_branches_per_step=3``,
    ``max_new_tokens=3``) runs twice on fresh setups: once with the default
    dispatch (batched) and once with ``_batched_forward_supported`` forced to
    ``False`` via monkeypatch (``raising=False`` so the Red state does not
    error on the missing symbol). Tokens, stable ratio, and acceptance rate
    must be identical; a layer spy proves the first run threads ``branch_ids``
    into every layer while the second never does; and the activation cache
    retains exactly the accepted chain — no rejected-branch residue.
    """
    prompt = torch.tensor([[1, 5, 9, 13]])
    calls = _install_layer_spy(monkeypatch)

    _raw_b, adapter_b, folded_b, cache_b, _gate_b = _build_setup("cpu")
    put_ids_b = _track_cached_branches(cache_b)
    draft_b = DraftGenerator(vocab_size=_VOCAB, mode="suffix_append", flip_ratio=0.0)
    result_batched = folded_generate(
        adapter_b,
        prompt,
        max_new_tokens=3,
        folded_model=folded_b,
        draft_generator=draft_b,
        num_branches_per_step=3,
    )
    batched_spy_calls = list(calls)
    calls.clear()

    monkeypatch.setattr(
        folded_generation,
        "_batched_forward_supported",
        lambda *args, **kwargs: False,
        raising=False,
    )
    _raw_s, adapter_s, folded_s, _cache_s, _gate_s = _build_setup("cpu")
    draft_s = DraftGenerator(vocab_size=_VOCAB, mode="suffix_append", flip_ratio=0.0)
    result_serial = folded_generate(
        adapter_s,
        prompt,
        max_new_tokens=3,
        folded_model=folded_s,
        draft_generator=draft_s,
        num_branches_per_step=3,
    )
    serial_spy_calls = list(calls)
    calls.clear()

    assert torch.equal(result_batched.tokens, result_serial.tokens)
    assert result_batched.stable_ratio == result_serial.stable_ratio
    assert result_batched.acceptance_rate == result_serial.acceptance_rate

    assert batched_spy_calls, "layer spy must observe the default run"
    assert all(
        entry.get("branch_ids") is not None for entry in batched_spy_calls
    ), "default run must thread branch_ids into every layer"
    assert serial_spy_calls, "layer spy must observe the forced-serial run"
    assert all(
        entry.get("branch_ids") is None for entry in serial_spy_calls
    ), "forced-serial run must never pass branch_ids"

    survivors = [bid for bid in put_ids_b if cache_b.contains(bid, 0)]
    assert len(survivors) == 3
    assert result_batched.final_branch_id in survivors
    rejected = [bid for bid in put_ids_b if bid not in survivors]
    assert rejected, "multi-candidate generation must prune rejected siblings"
    for branch_id in rejected:
        assert not cache_b.contains(branch_id, 0)


def test_ut1406_serial_fallback_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1406 (FR3/FR4): dispatch predicate rejects unsupported batches; greedy stays serial.

    The unit-level pin accesses ``folded_generation._batched_forward_supported``
    via ``getattr`` and checks the five fallback cases (single candidate,
    mismatched candidate shapes, mixed parents, ``folded_model=None``, and a
    deprecated ``FoldedModel`` instance). The greedy single-candidate
    end-to-end run then asserts every layer call carries
    ``branch_ids is None`` while the generated tokens still match the eager
    greedy baseline bit-for-bit.
    """
    predicate = getattr(folded_generation, "_batched_forward_supported", None)
    assert (
        predicate is not None
    ), "AR015 D7: folded_generation._batched_forward_supported is not implemented"

    _raw, _adapter, folded, _cache, _gate = _build_setup("cpu")

    def _node(branch_id: str, parent_id: str | None, tokens: torch.Tensor) -> BranchNode:
        return BranchNode(branch_id=branch_id, parent_id=parent_id, tokens=tokens, depth=1)

    short = torch.tensor([[1, 2, 3]])
    long = torch.tensor([[1, 2, 3, 4]])

    assert predicate(folded, [_node("m-a", "m-p", short)]) is False
    assert predicate(folded, [_node("m-a", "m-p", short), _node("m-b", "m-p", long)]) is False
    assert predicate(folded, [_node("m-a", "m-p1", short), _node("m-b", "m-p2", short)]) is False
    assert predicate(None, [_node("m-a", "m-p", short), _node("m-b", "m-p", short)]) is False

    torch.manual_seed(_SEED)
    raw_fm = _CausalBatchModel(_VOCAB, _HIDDEN, _NUM_LAYERS)
    folded_model = FoldedModel(
        raw_fm,
        cache=ActivationCache(max_entries_per_layer=256, device="cpu"),
        gate=SimilarityGate(tau=0.95),
        scheduler=None,
    )
    assert (
        predicate(folded_model, [_node("m-a", "m-p", short), _node("m-b", "m-p", short)]) is False
    )

    calls = _install_layer_spy(monkeypatch)
    _raw_g, adapter_g, folded_g, _cache_g, _gate_g = _build_setup("cpu")
    prompt = torch.tensor([[1, 5, 9, 13]])
    result = folded_generate(adapter_g, prompt, max_new_tokens=3, folded_model=folded_g)
    assert calls
    assert all(entry.get("branch_ids") is None for entry in calls)
    eager_adapter = FastDLLMAdapter(_raw_g, num_layers=_NUM_LAYERS, hidden_dim=_HIDDEN)
    expected = greedy_generate(eager_adapter, prompt, max_new_tokens=3)
    assert torch.equal(result.tokens, expected)


def _build_layer_setup(seed: int = _SEED) -> tuple[FoldedTransformerLayer, ActivationCache]:
    """Build a standalone layer-0 folded wrapper over the causal synthetic layer."""
    torch.manual_seed(seed)
    base = _CausalBatchLayer(_HIDDEN)
    base.eval()
    cache = ActivationCache(max_entries_per_layer=256, device="cpu")
    gate = SimilarityGate(tau=0.95)
    layer = FoldedTransformerLayer(base, cache, gate, layer_idx=0)
    return layer, cache


def test_ut1407_layer_front_door_contract() -> None:
    """UT-1407 (FR1/FR4): layer-level batched preconditions and parity branches.

    Direct ``FoldedTransformerLayer`` construction pins the front-door
    contract: ``len(branch_ids) != batch`` raises ``ValueError``; giving both
    ``branch_id`` and ``branch_ids`` raises ``ValueError`` at the layer and at
    the ``ManualFoldedForward`` entry; a parent cache entry with batch 2
    violates the expand contract and raises ``RuntimeError``. Two parity
    branches must stay bit-exact against serial semantics: ``branch_ids``
    without a parent (per-branch full recompute) and ``branch_ids`` with a
    parent id whose cache is empty (the generation-loop root shape, cache-miss
    full recompute).
    """
    torch.manual_seed(4107)
    x2 = torch.randn(2, _T_LAYER, _HIDDEN)

    layer_len, _cache_len = _build_layer_setup()
    with pytest.raises(ValueError):
        layer_len(
            x2,
            branch_id=None,
            branch_ids=["1407-a", "1407-b", "1407-c"],
        )

    layer_mutex, _cache_mutex = _build_layer_setup()
    with pytest.raises(ValueError):
        layer_mutex(x2, branch_id="1407-single", branch_ids=["1407-a", "1407-b"])

    layer_batch2, _cache_batch2 = _build_layer_setup()
    torch.manual_seed(4108)
    parent2 = torch.randn(2, _T_LAYER, _HIDDEN)
    layer_batch2(parent2, branch_id="1407-p2")
    with pytest.raises(RuntimeError):
        layer_batch2(
            x2,
            branch_id=None,
            branch_ids=["1407-a", "1407-b"],
            parent_branch_id="1407-p2",
        )

    _raw, _adapter, folded_entry, _cache_entry, _gate_entry = _build_setup("cpu")
    with pytest.raises(ValueError):
        folded_entry(
            torch.tensor([[1, 2, 3, 4]]),
            branch_id="1407-single",
            branch_ids=["1407-a"],
        )

    layer_b, cache_b = _build_layer_setup()
    layer_s, cache_s = _build_layer_setup()
    out_b = layer_b(x2, branch_id=None, branch_ids=["1407-na", "1407-nb"])
    out_s0 = layer_s(x2[0:1], branch_id="1407-sa")
    out_s1 = layer_s(x2[1:2], branch_id="1407-sb")
    assert torch.equal(out_b[0:1], out_s0)
    assert torch.equal(out_b[1:2], out_s1)
    for bid, sid in (("1407-na", "1407-sa"), ("1407-nb", "1407-sb")):
        entry_b = cache_b.fetch(branch_id=bid, layer_idx=0)
        entry_s = cache_s.fetch(branch_id=sid, layer_idx=0)
        assert torch.equal(entry_b["ffn_out"], entry_s["ffn_out"])
        assert torch.equal(entry_b["embedding"], entry_s["embedding"])

    layer_m_b, cache_m_b = _build_layer_setup()
    layer_m_s, cache_m_s = _build_layer_setup()
    out_m_b = layer_m_b(
        x2,
        branch_id=None,
        branch_ids=["1407-ma", "1407-mb"],
        parent_branch_id="1407-ghost",
    )
    out_m_s0 = layer_m_s(x2[0:1], branch_id="1407-msa", parent_branch_id="1407-ghost")
    out_m_s1 = layer_m_s(x2[1:2], branch_id="1407-msb", parent_branch_id="1407-ghost")
    assert torch.equal(out_m_b[0:1], out_m_s0)
    assert torch.equal(out_m_b[1:2], out_m_s1)
    for bid, sid in (("1407-ma", "1407-msa"), ("1407-mb", "1407-msb")):
        entry_b = cache_m_b.fetch(branch_id=bid, layer_idx=0)
        entry_s = cache_m_s.fetch(branch_id=sid, layer_idx=0)
        assert torch.equal(entry_b["ffn_out"], entry_s["ffn_out"])
        assert torch.equal(entry_b["embedding"], entry_s["embedding"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ut1408_cuda_graph_bypass() -> None:
    """UT-1408 (FR2/FR4): batched calls bypass CUDA graph; serial capture is unaffected.

    On a ``use_cuda_graph=True`` instance, the batched ``branch_ids`` call must
    stay eager and leave ``graph_runner`` untouched (state zero-change), while
    its output still matches a serial run on a fresh setup within
    allclose(1e-6) plus argmax token equality. A subsequent same-shape serial
    call on the SAME instance must capture and replay normally — proof that
    the batched eager path does not poison the graph path.
    """
    device = "cuda"
    _raw_b, _adapter_b, folded_b, cache_b, _gate_b = _build_setup(device, use_cuda_graph=True)
    _raw_s, _adapter_s, folded_s, cache_s, _gate_s = _build_setup(device)
    parent, candidates = _make_varlen_candidates(device, seed=4108)
    batched_ids = [f"1408-b{i}" for i in range(_NUM_CANDIDATES)]
    serial_ids = [f"1408-s{i}" for i in range(_NUM_CANDIDATES)]

    folded_s(parent, branch_id="1408-ps")
    serial_logits = [
        folded_s(tokens, branch_id=serial_ids[i], parent_branch_id="1408-ps", step_idx=1)
        for i, tokens in enumerate(candidates)
    ]
    folded_b(parent, branch_id="1408-pb")
    assert folded_b.graph_runner is None

    stacked = torch.cat(candidates, dim=0)
    batched_logits = folded_b(
        stacked,
        branch_id=None,
        branch_ids=batched_ids,
        parent_branch_id="1408-pb",
        step_idx=1,
    )
    assert folded_b.graph_runner is None
    for i in range(_NUM_CANDIDATES):
        assert torch.allclose(batched_logits[i : i + 1], serial_logits[i], rtol=1e-6, atol=1e-6)
        assert torch.equal(
            batched_logits[i].argmax(dim=-1),
            serial_logits[i][0].argmax(dim=-1),
        )
    for i, bid in enumerate(batched_ids):
        for layer_idx in range(_NUM_LAYERS):
            entry_b = cache_b.fetch(branch_id=bid, layer_idx=layer_idx)
            entry_s = cache_s.fetch(branch_id=serial_ids[i], layer_idx=layer_idx)
            assert torch.allclose(entry_b["ffn_out"], entry_s["ffn_out"], rtol=1e-6, atol=1e-6)

    child = parent.clone()
    graph_logits = folded_b(child, branch_id="1408-g0", parent_branch_id="1408-pb")
    assert folded_b.graph_runner is not None
    eager_logits = _raw_b(child)
    assert torch.allclose(graph_logits, eager_logits, rtol=1e-6, atol=1e-6)


def test_ut1409_scheduler_disabled_layers_batched_parity() -> None:
    """UT-1409 (FR1/FR4): scheduler-disabled layers recompute the whole batch.

    A ``FoldingScheduler`` with layer 1 in ``disabled_layers`` rides the
    batched path (layer 2 — the final layer — is disabled by the scheduler's
    own rule). Layers that ``should_fold`` rejects must take the whole-batch
    ``_recompute_all`` + per-branch store branch, while the folding layer
    reuses the dynamic-tau logic. Parity is bit-exact against a serial run
    carrying the same scheduler; the per-branch profiles record only the
    folding layer (index 0), and the disabled layer's cached rows differ
    across branches (per-child recompute, not a shared parent view).
    """
    device = "cpu"
    sched = FoldingScheduler(
        base_tau=0.95,
        num_layers=_NUM_LAYERS,
        num_steps=8,
        disabled_layers={1},
    )
    parent, candidates = _make_varlen_candidates(device, seed=4109)
    cache_b, _cache_s, profiles_b, _profiles_s, gate_b, _probe = _run_batched_vs_serial_parity(
        device, "1409", parent, candidates, scheduler=sched
    )

    for i in range(_NUM_CANDIDATES):
        profile = profiles_b[f"1409-b{i}"]
        assert profile is not None
        assert [stat.layer_idx for stat in profile.layer_stats] == [0]

    entry_a = cache_b.fetch(branch_id="1409-b0", layer_idx=1)["ffn_out"]
    entry_b = cache_b.fetch(branch_id="1409-b1", layer_idx=1)["ffn_out"]
    assert not torch.equal(entry_a, entry_b)

    assert gate_b.tau == sched.get_tau(layer_idx=0, step_idx=1, task_type="general")


def test_ut1410_all_stable_fast_path_parity() -> None:
    """UT-1410 (FR1/FR4): the all-stable batched branch shares the parent FFN.

    Four token-identical candidates make every gate mask all-True, so the
    batched three-way split must take the all-stable fast path: the cached
    parent FFN is expanded along the batch dimension and stored for every
    branch **without any recompute** (pinned by a per-layer ``_recompute_all``
    spy — a mutation degrading the fast path to recompute stays value-identical
    for a true-prefix child, so only the call count makes "不重算" observable).
    Parity is bit-exact against the serial fast path; every per-branch profile
    reports ``stable_ratio == 1.0`` and every cached branch FFN equals the
    parent's stored value.
    """
    device = "cpu"
    parent, candidates = _make_all_stable_candidates(device, seed=4110)
    cache_b, _cache_s, profiles_b, _profiles_s, _gate_b, recompute_counts = (
        _run_batched_vs_serial_parity(
            device, "1410", parent, candidates, after_parent=_spy_recompute_all
        )
    )

    assert recompute_counts == [0] * _NUM_LAYERS

    for i in range(_NUM_CANDIDATES):
        bid = f"1410-b{i}"
        profile = profiles_b[bid]
        assert profile is not None
        assert profile.mean_stable_ratio == 1.0
        for stat in profile.layer_stats:
            assert stat.stable_ratio == 1.0
        for layer_idx in range(_NUM_LAYERS):
            branch_ffn = cache_b.fetch(branch_id=bid, layer_idx=layer_idx)["ffn_out"]
            parent_ffn = cache_b.fetch(branch_id="1410-pb", layer_idx=layer_idx)["ffn_out"]
            assert torch.equal(branch_ffn, parent_ffn)


def test_ut1411_truncated_all_stable_parity() -> None:
    """UT-1411 (FR1/FR4): AR007 longer parent feeds the all-stable fast path.

    A parent of length 8 donates its 6-token prefix to four true-prefix
    children: the batched gate activation and the cached parent FFN are both
    truncated before the all-stable fast path expands and stores them — the
    ``_recompute_all`` spy pins that the truncation fallback (shape-mismatch
    recompute) never fires. Parity is bit-exact against the serial
    truncated-prefix reuse, every profile reports ``stable_ratio == 1.0``,
    and each cached branch FFN equals the parent's truncated FFN.
    """
    device = "cpu"
    child_len = 6
    parent, candidates = _make_truncated_candidates(device, child_len=child_len, seed=4111)
    cache_b, _cache_s, profiles_b, _profiles_s, _gate_b, recompute_counts = (
        _run_batched_vs_serial_parity(
            device, "1411", parent, candidates, after_parent=_spy_recompute_all
        )
    )

    assert recompute_counts == [0] * _NUM_LAYERS

    for i in range(_NUM_CANDIDATES):
        bid = f"1411-b{i}"
        profile = profiles_b[bid]
        assert profile is not None
        assert profile.mean_stable_ratio == 1.0
        for layer_idx in range(_NUM_LAYERS):
            branch_ffn = cache_b.fetch(branch_id=bid, layer_idx=layer_idx)["ffn_out"]
            parent_ffn = cache_b.fetch(branch_id="1411-pb", layer_idx=layer_idx)["ffn_out"]
            assert branch_ffn.shape[1] == child_len
            assert torch.equal(branch_ffn, parent_ffn[:, :child_len])


def test_ut1412_truncated_mixed_merge_parity() -> None:
    """UT-1412 (FR1/FR4): AR007 truncated parent FFN inside the mixed merge.

    Four 6-token children of an 8-token parent replace the last two tokens
    each, so every layer takes the mixed slow path and the batched merge must
    truncate the longer cached parent FFN to the child grid before the
    expand-and-merge. Parity is bit-exact against the serial run and the
    per-branch profiles report a strictly mixed ratio (neither 0.0 nor 1.0),
    proving the three-way split resolved to the slow path.
    """
    device = "cpu"
    parent, candidates = _make_truncated_candidates(
        device, child_len=6, seed=4112, divergent_tail=2
    )
    _cache_b, _cache_s, profiles_b, _profiles_s, _gate_b, _probe = _run_batched_vs_serial_parity(
        device, "1412", parent, candidates
    )

    for i in range(_NUM_CANDIDATES):
        profile = profiles_b[f"1412-b{i}"]
        assert profile is not None
        assert 0.0 < profile.mean_stable_ratio < 1.0
