"""Red-phase tests for AR018: publish-path host-cost compression (UT-1701..UT-1708).

Contract: ``specs/changes/AR018-publish-cost/design.md`` §2/§3/§5 and
``srs.md`` §3.1-§3.3 -- the batched publish loop must (FR1) derive every
(layer, row) stability count from ONE ``mask_buf.sum(dim=2)`` reduction
recorded through the new ``StabilityProfiler.record_precomputed`` entry
(0-dim views, zero launch, zero host readback) and (FR2) publish from ONE
fresh whole-buffer ``child_buf.clone()`` snapshot (``cache.put`` copies, so
per-row clones are pure overhead), with ``debug_enabled`` falling back to
the per-row ``record`` entry (the only one producing
``divergence_positions``) and ``enabled=False`` performing zero reductions
and zero records.

Red state (why the new-behavior assertions fail today):
``StabilityProfiler.record_precomputed`` does not exist (AttributeError,
UT-1701) and ``_publish_graph_children`` still clones and sums per row, so
the spy windows observe bool-sum = L*N = 8 (UT-1702 and UT-1707's normal
branch) and child_buf-identity clone count 0 of a total 8 (UT-1704)
instead of exactly one each.

Double-sided invariant UTs (expected GREEN in the Red run -- the Red value
of this file is carried by UT-1702/UT-1704): UT-1703 (publish content is
bit-exact against a test-local replica of the old per-row clone publish),
UT-1705 (by-reference anti-aliasing: per-row clones isolate today, the
fresh snapshot isolates after the change -- the mutation-③ guard), UT-1706
(UT-1502 publish-completeness + UT-1509 publish-isolation faces, local
re-assertions) and UT-1708 (eager/serial-path zero interference plus the
srs-pinned ``record`` signature). UT-1709 (mutation spot-check) is
reserved for the T002 mutation phase and is intentionally not implemented
in this file.

Spy window semantics (design §5): ``torch.Tensor.sum``/``clone``/``item``/
``tolist``/``cpu`` are replaced with counting wrappers whose window is a
SINGLE ``_publish_graph_children`` invocation -- the shadowing wrapper
installs the patches immediately before delegating to the original bound
publish method and restores them in a ``finally`` (the ``logits.clone``
after the publish lies outside the window by construction). ``sum`` counts
only ``self.dtype is torch.bool`` (mask reductions); ``clone`` counts
``self is runner.child_buf`` (identity) plus an unfiltered total;
``item``/``tolist``/``cpu`` count unfiltered (zero host sync allowed in
the window).

GLOBAL singleton hygiene: the autouse ``_global_profiler_state`` fixture
snapshots and restores ``GLOBAL_STABILITY_PROFILER.enabled``/
``debug_enabled``/``_raw``/``_profiles``/``_history`` around every case,
so enabled-flag flips and branch-id pollution never leak across cases.

Fixtures follow ``tests/test_batched_graph_forward.py`` (self-contained
L=2/H=32/T=16/V=64/N=4 synthetic LLaMA-layout decoder; ``_build_model``/
``_make_mff``/candidate builders copied isomorphically into this file --
no cross-test-module imports, AR017 precedent). ``N * T = 64`` stays a
power of two. CUDA gating is per-function through the module constant
``_cuda`` (a module-level ``pytestmark`` would skip the CPU-runnable
UT-1701 profiler block along with the CUDA cases); UT-1701 is split into a
CPU sub-block and a CUDA sub-block. Seeds are fixed (repo seed=2026
convention for model builders, explicit per-UT candidate seeds).
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.chunked_cache import ChunkedActivationCache
from actfold.models.architecture_utils import ManualFoldedForward, detect_architecture
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER, StabilityProfiler

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_VOCAB = 64
_HIDDEN = 32
_NUM_LAYERS = 2
_SEQ = 16
_NUM_CANDIDATES = 4
# N * T MUST remain a power of two (AR017 §6 fixture convention): capacity
# conversions ceil(ratio * N * T) are exact only for power-of-two totals.
_NUM_TOKENS = _NUM_CANDIDATES * _SEQ
assert _NUM_TOKENS > 0 and _NUM_TOKENS & (_NUM_TOKENS - 1) == 0
_TAU = 0.95
_CACHE_BUDGET = 1024


# ---------------------------------------------------------------------------
# Self-contained synthetic decoder (LlamaLikeModel layout, copied from
# tests/test_batched_graph_forward.py -- repo precedent: fixtures are
# duplicated per test module, never imported across test modules).
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
# Shared scenario helpers (isomorphic copies of the AR016 test module's
# fixtures; the AR017 adaptive-capacity flags stay off -- AR018 is
# orthogonal to the capacity policy).
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
    cache: Any = None,
    use_cuda_graph: bool = False,
    graph_capacity_ratio: float = 0.5,
) -> ManualFoldedForward:
    """Fresh ManualFoldedForward (own cache + gate) over the shared model.

    Follows the test_batched_graph_forward.py wiring: split layers with a
    lowered ``split_min_tokens`` so the split FFN path engages at the test
    shapes, plus the AR002 graph flags (``use_cuda_graph=True`` with
    ``graph_capacity_ratio=0.5`` in the graph cases).
    """
    if cache is None:
        cache = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    if gate is None:
        gate = SimilarityGate(tau=_TAU, metric="cosine")
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
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
    suffix, a few divergent positions per row) and the folded slow path --
    the publish-observing path -- engages on both the eager and graph runs.
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
    step_idx: int = 1,
) -> torch.Tensor:
    """One serial ``branch_id`` forward under ``no_grad``."""
    with torch.no_grad():
        return mff(
            tokens,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            step_idx=step_idx,
        )


def _batched(
    mff: ManualFoldedForward,
    stacked: torch.Tensor,
    branch_ids: list[str],
    parent_branch_id: str,
    step_idx: int = 1,
) -> torch.Tensor:
    """One batched ``branch_ids`` forward under ``no_grad``."""
    with torch.no_grad():
        return mff(
            stacked,
            branch_id=None,
            branch_ids=branch_ids,
            parent_branch_id=parent_branch_id,
            step_idx=step_idx,
        )


# ---------------------------------------------------------------------------
# GLOBAL singleton hygiene (design §5)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _global_profiler_state() -> Iterator[None]:
    """Snapshot/restore the GLOBAL stability profiler around every case.

    ``enabled``/``debug_enabled`` flag flips and branch-id pollution
    (``_raw``/``_profiles``/``_history``) never leak across cases.
    """
    profiler = GLOBAL_STABILITY_PROFILER
    saved_enabled = profiler.enabled
    saved_debug = profiler.debug_enabled
    saved_raw = {key: list(entries) for key, entries in profiler._raw.items()}
    saved_profiles = dict(profiler._profiles)
    saved_history = {key: list(entries) for key, entries in profiler._history.items()}
    try:
        yield
    finally:
        profiler.enabled = saved_enabled
        profiler.debug_enabled = saved_debug
        profiler._raw.clear()
        for key, entries in saved_raw.items():
            profiler._raw[key] = list(entries)
        profiler._profiles.clear()
        profiler._profiles.update(saved_profiles)
        profiler._history.clear()
        for key, entries in saved_history.items():
            profiler._history[key] = list(entries)


# ---------------------------------------------------------------------------
# Publish-window spy (design §5): window = one _publish_graph_children call
# ---------------------------------------------------------------------------


@contextmanager
def _publish_tensor_spy(runner: Any) -> Iterator[dict[str, int]]:
    """Count ``Tensor.sum``/``clone``/``item``/``tolist``/``cpu`` in the window.

    The patches are installed by the caller immediately before a single
    ``_publish_graph_children`` invocation and restored in a ``finally`` so
    they never leak past the publish. ``sum`` counts only bool ``self``
    (mask reductions); ``clone`` counts ``self is runner.child_buf``
    (identity) plus an unfiltered total; ``item``/``tolist``/``cpu`` count
    unfiltered (zero host sync allowed in the window).
    """
    counts = {
        "bool_sum": 0,
        "clone_child": 0,
        "clone_total": 0,
        "item": 0,
        "tolist": 0,
        "cpu": 0,
    }
    original = {
        name: getattr(torch.Tensor, name) for name in ("sum", "clone", "item", "tolist", "cpu")
    }

    def _sum(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.dtype is torch.bool:
            counts["bool_sum"] += 1
        return original["sum"](self, *args, **kwargs)

    def _clone(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        counts["clone_total"] += 1
        if self is runner.child_buf:
            counts["clone_child"] += 1
        return original["clone"](self, *args, **kwargs)

    def _item(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        counts["item"] += 1
        return original["item"](self, *args, **kwargs)

    def _tolist(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        counts["tolist"] += 1
        return original["tolist"](self, *args, **kwargs)

    def _cpu(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        counts["cpu"] += 1
        return original["cpu"](self, *args, **kwargs)

    for name, wrapper in (
        ("sum", _sum),
        ("clone", _clone),
        ("item", _item),
        ("tolist", _tolist),
        ("cpu", _cpu),
    ):
        setattr(torch.Tensor, name, wrapper)
    try:
        yield counts
    finally:
        for name, method in original.items():
            setattr(torch.Tensor, name, method)


def _windowed_publish_spy(
    monkeypatch: pytest.MonkeyPatch,
    mff: ManualFoldedForward,
) -> list[dict[str, int]]:
    """Shadow ``mff._publish_graph_children`` with a spy-windowed wrapper.

    Every production call to the publish method runs inside a fresh
    ``_publish_tensor_spy`` window; the per-call counter dicts are appended
    to the returned list in call order (one dict per publish).
    """
    windows: list[dict[str, int]] = []
    original_publish = mff._publish_graph_children

    def _windowed(*args: Any, **kwargs: Any) -> None:
        with _publish_tensor_spy(mff.graph_runner_batched) as counts:
            original_publish(*args, **kwargs)
        windows.append(counts)

    monkeypatch.setattr(mff, "_publish_graph_children", _windowed)
    return windows


# ---------------------------------------------------------------------------
# UT-1701: record vs record_precomputed equivalence (CPU sub-block + CUDA
# sub-block; the pure-profiler face runs on CPU, the publish-geometry face
# on CUDA -- design §5)
# ---------------------------------------------------------------------------


def test_ut1701_record_precomputed_equivalence_cpu() -> None:
    """UT-1701 (FR1, CPU sub-block): ``record`` vs ``record_precomputed``.

    The same mask collection is recorded into two fresh profilers -- one
    through the per-row ``record`` entry, one through
    ``record_precomputed`` fed each mask's 0-dim ``sum()`` plus the token
    count -- and every observable face must agree field-for-field:
    ``_raw`` entries (branch/parent/layer/step/num_tokens/tau/metric and
    ``stable_sum`` via ``.item()``), ``_history`` key sets and lengths
    (including the ``_max_history_len`` trim after 120 records on one
    (layer, step) key), ``get_profile().to_dict()``, the aggregation faces
    ``mean_stable_ratio`` / ``all_profiles`` / ``get_mean_stable_ratio``,
    and the ``enabled=False`` early return of both entries. Input
    validation: a non-0-dim or non-Tensor ``stable_sum`` raises
    ``ValueError``. Red face: ``record_precomputed`` does not exist yet
    (AttributeError on the first call).
    """
    torch.manual_seed(2026)
    num_steps = 120  # > StabilityProfiler._max_history_len: the trim engages.
    masks: list[tuple[str, int, torch.Tensor]] = []
    for i in range(num_steps):
        for layer_idx in range(_NUM_LAYERS):
            branch_id = "1701-x" if i % 2 == 0 else "1701-y"
            masks.append((branch_id, layer_idx, torch.rand(1, _SEQ) < 0.75))

    per_row = StabilityProfiler()
    precomputed = StabilityProfiler()
    for branch_id, layer_idx, mask in masks:
        per_row.record(
            branch_id=branch_id,
            parent_branch_id="1701-p",
            layer_idx=layer_idx,
            step_idx=7,
            stable_mask=mask,
            tau=_TAU,
        )
        precomputed.record_precomputed(
            branch_id=branch_id,
            parent_branch_id="1701-p",
            layer_idx=layer_idx,
            step_idx=7,
            stable_sum=mask.sum(),
            num_tokens=mask.numel(),
            tau=_TAU,
        )

    assert set(per_row._raw) == set(precomputed._raw) == {"1701-x", "1701-y"}
    for branch_id in ("1701-x", "1701-y"):
        entries_a = per_row._raw[branch_id]
        entries_b = precomputed._raw[branch_id]
        assert len(entries_a) == len(entries_b) == num_steps
        for raw_a, raw_b in zip(entries_a, entries_b):
            assert raw_a.branch_id == raw_b.branch_id
            assert raw_a.parent_branch_id == raw_b.parent_branch_id
            assert raw_a.layer_idx == raw_b.layer_idx
            assert raw_a.step_idx == raw_b.step_idx
            assert raw_a.num_tokens == raw_b.num_tokens
            assert raw_a.tau == raw_b.tau
            assert raw_a.metric == raw_b.metric
            assert raw_a.stable_sum.item() == raw_b.stable_sum.item()
            assert raw_a.divergence_positions is None
            assert raw_b.divergence_positions is None

    assert set(per_row._history) == set(precomputed._history) == {(0, 7), (1, 7)}
    for key in ((0, 7), (1, 7)):
        trimmed_a = per_row._history[key]
        trimmed_b = precomputed._history[key]
        assert len(trimmed_a) == len(trimmed_b)
        assert len(trimmed_a) == per_row._max_history_len
        assert per_row._max_history_len == precomputed._max_history_len
        assert len(trimmed_a) < num_steps, "the history trim must have engaged"

    for branch_id in ("1701-x", "1701-y"):
        profile_a = per_row.get_profile(branch_id)
        profile_b = precomputed.get_profile(branch_id)
        assert profile_a is not None and profile_b is not None
        assert profile_a.to_dict() == profile_b.to_dict()
        assert profile_a.mean_stable_ratio == profile_b.mean_stable_ratio
    profiles_a = per_row.all_profiles()
    profiles_b = precomputed.all_profiles()
    assert set(profiles_a) == set(profiles_b)
    for branch_id in profiles_a:
        assert profiles_a[branch_id].to_dict() == profiles_b[branch_id].to_dict()
    for layer_idx in range(_NUM_LAYERS):
        mean_a = per_row.get_mean_stable_ratio(layer_idx, 7)
        mean_b = precomputed.get_mean_stable_ratio(layer_idx, 7)
        assert mean_a is not None and mean_b is not None
        assert mean_a == mean_b

    off_row = StabilityProfiler(enabled=False)
    off_precomputed = StabilityProfiler(enabled=False)
    for branch_id, layer_idx, mask in masks[:4]:
        off_row.record(
            branch_id=branch_id,
            parent_branch_id="1701-p",
            layer_idx=layer_idx,
            step_idx=7,
            stable_mask=mask,
            tau=_TAU,
        )
        off_precomputed.record_precomputed(
            branch_id=branch_id,
            parent_branch_id="1701-p",
            layer_idx=layer_idx,
            step_idx=7,
            stable_sum=mask.sum(),
            num_tokens=mask.numel(),
            tau=_TAU,
        )
    assert not off_row._raw and not off_precomputed._raw
    assert not off_row._history and not off_precomputed._history

    strict = StabilityProfiler()
    with pytest.raises(ValueError):
        strict.record_precomputed(
            branch_id="1701-v",
            parent_branch_id="1701-p",
            layer_idx=0,
            step_idx=0,
            stable_sum=torch.tensor([1, 2]),
            num_tokens=2,
            tau=_TAU,
        )
    with pytest.raises(ValueError):
        strict.record_precomputed(
            branch_id="1701-v",
            parent_branch_id="1701-p",
            layer_idx=0,
            step_idx=0,
            stable_sum=3,
            num_tokens=2,
            tau=_TAU,
        )
    assert not strict._raw


@_cuda
def test_ut1701_batched_sum_views_equivalence_cuda() -> None:
    """UT-1701 (FR1, CUDA sub-block): 0-dim views of one batched reduction.

    Mirrors the publish-path geometry: an [L, N, T] bool mask buffer is
    reduced ONCE via ``sum(dim=2)`` into an [L, N] table; the per-row
    profiler records each ``mask_buf[l][r:r+1]`` row while the precomputed
    profiler records the 0-dim VIEW ``sums[l][r]`` with ``num_tokens=T``.
    Both must agree on every per-branch ``get_profile().to_dict()`` and on
    the per-(layer, step) ``get_mean_stable_ratio`` aggregation. Red face:
    ``record_precomputed`` does not exist yet (AttributeError).
    """
    torch.manual_seed(2026)
    mask_buf = torch.rand(_NUM_LAYERS, _NUM_CANDIDATES, _SEQ, device="cuda") < 0.75
    sums = mask_buf.sum(dim=2)
    per_row = StabilityProfiler()
    precomputed = StabilityProfiler()
    for layer_idx in range(_NUM_LAYERS):
        for row in range(_NUM_CANDIDATES):
            branch_id = f"1701c-l{layer_idx}r{row}"
            per_row.record(
                branch_id=branch_id,
                parent_branch_id="1701c-p",
                layer_idx=layer_idx,
                step_idx=3,
                stable_mask=mask_buf[layer_idx][row : row + 1],
                tau=_TAU,
            )
            precomputed.record_precomputed(
                branch_id=branch_id,
                parent_branch_id="1701c-p",
                layer_idx=layer_idx,
                step_idx=3,
                stable_sum=sums[layer_idx][row],
                num_tokens=_SEQ,
                tau=_TAU,
            )
    for layer_idx in range(_NUM_LAYERS):
        for row in range(_NUM_CANDIDATES):
            branch_id = f"1701c-l{layer_idx}r{row}"
            profile_a = per_row.get_profile(branch_id)
            profile_b = precomputed.get_profile(branch_id)
            assert profile_a is not None and profile_b is not None
            assert profile_a.to_dict() == profile_b.to_dict()
        mean_a = per_row.get_mean_stable_ratio(layer_idx, 3)
        mean_b = precomputed.get_mean_stable_ratio(layer_idx, 3)
        assert mean_a is not None and mean_b is not None
        assert mean_a == mean_b


# ---------------------------------------------------------------------------
# UT-1702: one bool reduction per publish, zero host syncs (spy window)
# ---------------------------------------------------------------------------


@_cuda
def test_ut1702_single_mask_reduction_and_zero_host_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-1702 (FR1): exactly one bool sum per publish, zero host syncs.

    In the normal state (``profiler.enabled=True``, ``debug_enabled=
    False``) each publish window must contain exactly ONE bool
    ``Tensor.sum`` (the single batched ``mask_buf.sum(dim=2)``; Red: the
    per-row ``record`` sums give L*N=8) and ZERO ``item``/``tolist()``/
    ``cpu()`` calls (no host readback on the publish path). With
    ``GLOBAL_STABILITY_PROFILER.enabled=False`` the publish must perform
    ZERO bool reductions and record nothing for its branches.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4202)
    mff = _make_mff(model, device, use_cuda_graph=True)
    windows = _windowed_publish_spy(monkeypatch, mff)
    _serial(mff, parent, branch_id="1702-p")

    normal_ids = _ids("1702-n")
    _batched(mff, stacked, normal_ids, "1702-p")
    assert len(windows) == 1
    assert windows[0]["bool_sum"] == 1, (
        f"exactly one bool sum per publish (the batched reduction); "
        f"observed {windows[0]['bool_sum']}"
    )
    assert windows[0]["item"] == 0, "zero .item() host syncs in the publish window"
    assert windows[0]["tolist"] == 0, "zero .tolist() host syncs in the window"
    assert windows[0]["cpu"] == 0, "zero .cpu() host syncs in the publish window"

    GLOBAL_STABILITY_PROFILER.enabled = False
    try:
        disabled_ids = _ids("1702-d")
        _batched(mff, stacked, disabled_ids, "1702-p")
    finally:
        GLOBAL_STABILITY_PROFILER.enabled = True
    assert len(windows) == 2
    assert windows[1]["bool_sum"] == 0, "a disabled profiler performs zero reductions"
    for bid in disabled_ids:
        assert bid not in GLOBAL_STABILITY_PROFILER._raw
        assert GLOBAL_STABILITY_PROFILER.get_profile(bid) is None


# ---------------------------------------------------------------------------
# UT-1703: snapshot publish is bit-exact vs the old per-row clone publish
# (double-sided invariant UT -- green in the Red run)
# ---------------------------------------------------------------------------


@_cuda
def test_ut1703_snapshot_publish_bit_exact() -> None:
    """UT-1703 (FR2, double-sided): publish content is bit-exact vs per-row.

    The production publish (cache A) and a test-local reference publish
    (cache B) -- a replica of the OLD per-row loop
    (``child_buf[l][r:r+1].clone()`` + ``cache.put``, deliberately NO
    profiler calls, so the GLOBAL singleton stays unpolluted by the
    reference) -- run against the SAME replay state (the static buffers
    are untouched in between), so every (branch, layer) entry must compare
    ``torch.equal`` on ``ffn_out`` and, at layer 0, on the ``embedding``
    row. Green today (the current implementation IS the reference) and
    green after the change (the snapshot's row views carry identical
    values); the launch-count Red value lives in UT-1702/UT-1704.
    """
    device = "cuda"
    model, profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4203)
    mff = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff, parent, branch_id="1703-p")
    branch_ids = _ids("1703-a")
    _batched(mff, stacked, branch_ids, "1703-p")
    runner = mff.graph_runner_batched
    assert runner is not None

    reference_cache = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    with torch.no_grad():
        embedding = profile.embed_module(stacked)
        for layer_idx in range(_NUM_LAYERS):
            for row, branch_id in enumerate(branch_ids):
                activations: dict[str, Any] = {
                    "ffn_out": runner.child_buf[layer_idx][row : row + 1].clone()
                }
                if layer_idx == 0:
                    activations["embedding"] = embedding[row : row + 1]
                reference_cache.put(
                    branch_id=branch_id,
                    layer_idx=layer_idx,
                    activations=activations,
                )

    for branch_id in branch_ids:
        for layer_idx in range(_NUM_LAYERS):
            published = mff.cache.fetch(branch_id=branch_id, layer_idx=layer_idx)
            reference = reference_cache.fetch(branch_id=branch_id, layer_idx=layer_idx)
            assert torch.equal(
                published["ffn_out"], reference["ffn_out"]
            ), f"ffn_out mismatch for {branch_id} layer {layer_idx}"
            if layer_idx == 0:
                assert torch.equal(
                    published["embedding"], reference["embedding"]
                ), f"embedding mismatch for {branch_id}"


# ---------------------------------------------------------------------------
# UT-1704: exactly one clone per publish, on the whole child_buf
# ---------------------------------------------------------------------------


@_cuda
def test_ut1704_single_child_buf_clone(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1704 (FR2): exactly one clone per publish, on the whole child_buf.

    The publish window must contain exactly ONE ``Tensor.clone`` whose
    ``self`` IS the runner's ``child_buf`` (the fresh whole-buffer
    snapshot) and exactly one clone in TOTAL (no per-row intermediate
    clones; Red: L*N=8 row-slice clones, none identity-equal to
    ``child_buf``). The ``logits.clone`` after the publish lies outside
    the window by construction (the window wraps only the publish call).
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4204)
    mff = _make_mff(model, device, use_cuda_graph=True)
    windows = _windowed_publish_spy(monkeypatch, mff)
    _serial(mff, parent, branch_id="1704-p")
    _batched(mff, stacked, _ids("1704-a"), "1704-p")
    assert len(windows) == 1
    assert windows[0]["clone_child"] == 1, (
        f"exactly one whole-child_buf clone per publish; " f"observed {windows[0]['clone_child']}"
    )
    assert windows[0]["clone_total"] == 1, (
        f"no per-row intermediate clones in the window; "
        f"observed {windows[0]['clone_total']} total clones"
    )


# ---------------------------------------------------------------------------
# UT-1705: by-reference cache anti-aliasing (double-sided invariant UT)
# ---------------------------------------------------------------------------


class _RefCache:
    """Minimal by-reference cache: ``put`` stores activation references."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, int], dict[str, torch.Tensor]] = {}

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
    ) -> None:
        """Store the activation dict as-is (tensor references, no copies)."""
        self.store[(branch_id, layer_idx)] = dict(activations)


@_cuda
def test_ut1705_by_reference_cache_anti_alias() -> None:
    """UT-1705 (FR2, double-sided): published entries never alias child_buf.

    After a real capture, ``mff.cache`` is swapped for a minimal
    by-reference cache and ``_publish_graph_children`` is called DIRECTLY
    (the publish body's only cache use is ``put`` -- the replay-time parent
    fetch is bypassed). Mutating ``runner.child_buf`` in place must leave
    every stored reference bit-identical (per-row clones isolate today;
    the fresh snapshot isolates after AR018 -- the mutation-③ guard), and
    a subsequent replay + publish through the real path must write the
    correct (replay-deterministic) values: the second replay fully
    overwrites the polluted static buffer, so the new publish must equal
    the first publish's values.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4205)
    mff = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff, parent, branch_id="1705-p")
    first_ids = _ids("1705-a")
    _batched(mff, stacked, first_ids, "1705-p")
    runner = mff.graph_runner_batched
    assert runner is not None

    real_cache = mff.cache
    ref_cache = _RefCache()
    mff.cache = ref_cache
    try:
        with torch.no_grad():
            mff._publish_graph_children(runner, stacked, first_ids, "1705-p", 1)
    finally:
        mff.cache = real_cache

    snapshot = {
        key: {name: tensor.clone() for name, tensor in entry.items()}
        for key, entry in ref_cache.store.items()
    }
    with torch.no_grad():
        runner.child_buf.add_(123.0)
    for key, entry in ref_cache.store.items():
        for name, tensor in entry.items():
            assert torch.equal(
                tensor, snapshot[key][name]
            ), f"mutating child_buf must not touch by-reference entry {key}"

    second_ids = _ids("1705-b")
    _batched(mff, stacked, second_ids, "1705-p")
    for row, branch_id in enumerate(second_ids):
        for layer_idx in range(_NUM_LAYERS):
            published = mff.cache.fetch(branch_id=branch_id, layer_idx=layer_idx)
            assert torch.equal(
                published["ffn_out"], snapshot[(first_ids[row], layer_idx)]["ffn_out"]
            ), (
                f"the second publish must write the correct replay values "
                f"for {branch_id} layer {layer_idx}"
            )


# ---------------------------------------------------------------------------
# UT-1706: existing publish contracts (UT-1502 completeness + UT-1509
# isolation faces, local re-assertions; double-sided invariant UT)
# ---------------------------------------------------------------------------


@_cuda
def test_ut1706_publish_completeness_and_isolation() -> None:
    """UT-1706 (FR2/FR3, double-sided): UT-1502 completeness + UT-1509 isolation.

    Local re-assertions of the two existing publish contracts (both faces
    hold for the old per-row clones and must hold for the snapshot
    publish; the Red value of this file lives in UT-1702/UT-1704):

    * completeness (UT-1502 face): after the capture call and after a
      second (replay) call, every branch cache entry carries ``ffn_out``
      at all L layers plus the layer-0 ``embedding`` row, and each stored
      embedding row equals the ``embed_module(tokens)`` row within 1e-6;
    * isolation (UT-1509 face, on ``ChunkedActivationCache`` -- its
      ``put`` stores contiguous row views as-is, making aliasing
      observable): an in-place ``child_buf.add_(123)`` must not change
      any published item, and a second replay must neither overwrite the
      first publish nor drop the second.
    """
    device = "cuda"
    model, profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4206)

    # (a) Publish completeness (UT-1502 face) on the default cache.
    mff = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff, parent, branch_id="1706-p")
    with torch.no_grad():
        embedding_rows = profile.embed_module(stacked)
    for tag in ("1706-a", "1706-b"):
        branch_ids = _ids(tag)
        out = _batched(mff, stacked, branch_ids, "1706-p")
        assert tuple(out.shape) == (_NUM_CANDIDATES, _SEQ, _VOCAB)
        for row, branch_id in enumerate(branch_ids):
            for layer_idx in range(_NUM_LAYERS):
                entry = mff.cache.fetch(branch_id=branch_id, layer_idx=layer_idx)
                assert "ffn_out" in entry, f"{branch_id} layer {layer_idx}: no ffn_out"
                assert tuple(entry["ffn_out"].shape) == (1, _SEQ, _HIDDEN)
                if layer_idx == 0:
                    assert "embedding" in entry, f"{branch_id}: no layer-0 embedding"
                    assert tuple(entry["embedding"].shape) == (1, _SEQ, _HIDDEN)
                    assert torch.allclose(
                        entry["embedding"], embedding_rows[row : row + 1], atol=1e-6
                    )

    # (b) Publish isolation (UT-1509 face) on the chunked cache.
    cache = ChunkedActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    mff_iso = _make_mff(model, device, cache=cache, use_cuda_graph=True)
    _serial(mff_iso, parent, branch_id="1706-ip")
    first_ids = _ids("1706-x")
    second_ids = _ids("1706-y")
    _batched(mff_iso, stacked, first_ids, "1706-ip")
    runner = mff_iso.graph_runner_batched
    assert runner is not None
    snapshot: dict[tuple[str, int], torch.Tensor] = {}
    for branch_id in first_ids:
        for layer_idx in range(_NUM_LAYERS):
            snapshot[(branch_id, layer_idx)] = cache.fetch(
                branch_id=branch_id, layer_idx=layer_idx
            )["ffn_out"].clone()
    with torch.no_grad():
        runner.child_buf.add_(123.0)
    for branch_id in first_ids:
        for layer_idx in range(_NUM_LAYERS):
            published = cache.fetch(branch_id=branch_id, layer_idx=layer_idx)["ffn_out"]
            assert torch.equal(
                published, snapshot[(branch_id, layer_idx)]
            ), "mutating child_buf must not touch published cache items"
    _batched(mff_iso, stacked, second_ids, "1706-ip")
    for branch_id in first_ids:
        for layer_idx in range(_NUM_LAYERS):
            published = cache.fetch(branch_id=branch_id, layer_idx=layer_idx)["ffn_out"]
            assert torch.equal(
                published, snapshot[(branch_id, layer_idx)]
            ), "the second replay must not overwrite the first publish"
    for branch_id in second_ids:
        for layer_idx in range(_NUM_LAYERS):
            assert cache.contains(branch_id, layer_idx)


# ---------------------------------------------------------------------------
# UT-1707: debug_enabled falls back to per-row record; the normal branch
# batches (the Red face lives in the normal-branch assertion)
# ---------------------------------------------------------------------------


@_cuda
def test_ut1707_debug_fallback_and_normal_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1707 (FR1): debug falls back to per-row record; normal batches.

    With ``GLOBAL_STABILITY_PROFILER.debug_enabled=True`` the publish must
    take the per-row ``record`` path: bool-sum = L*N inside the window and
    the mixed-mask layers produce ``divergence_positions`` (the only entry
    that can). With ``debug_enabled=False`` the publish must take the
    batched path: exactly ONE bool sum (the Red face -- the current
    per-row implementation still sums L*N=8 times in the normal state) and
    every recorded ``_raw`` entry carries ``divergence_positions=None``.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4207)
    mff = _make_mff(model, device, use_cuda_graph=True)
    windows = _windowed_publish_spy(monkeypatch, mff)
    _serial(mff, parent, branch_id="1707-p")

    GLOBAL_STABILITY_PROFILER.debug_enabled = True
    debug_ids = _ids("1707-d")
    _batched(mff, stacked, debug_ids, "1707-p")
    assert len(windows) == 1
    assert windows[0]["bool_sum"] == _NUM_LAYERS * _NUM_CANDIDATES
    debug_entries = [
        raw for bid in debug_ids for raw in GLOBAL_STABILITY_PROFILER._raw.get(bid, [])
    ]
    assert debug_entries, "the debug publish must record every branch"
    assert any(
        raw.divergence_positions is not None for raw in debug_entries
    ), "debug records must produce divergence_positions for mixed-mask layers"

    GLOBAL_STABILITY_PROFILER.debug_enabled = False
    normal_ids = _ids("1707-n")
    _batched(mff, stacked, normal_ids, "1707-p")
    assert len(windows) == 2
    assert windows[1]["bool_sum"] == 1, (
        f"the normal (non-debug) publish must batch the reduction into one "
        f"sum; observed {windows[1]['bool_sum']}"
    )
    normal_entries = [
        raw for bid in normal_ids for raw in GLOBAL_STABILITY_PROFILER._raw.get(bid, [])
    ]
    assert normal_entries, "the normal publish must record every branch"
    assert all(
        raw.divergence_positions is None for raw in normal_entries
    ), "the batched path never produces divergence_positions"


# ---------------------------------------------------------------------------
# UT-1708: serial-graph and eager-batched paths stay unaffected (double-
# sided invariant UT) + the srs-pinned record signature
# ---------------------------------------------------------------------------


@_cuda
def test_ut1708_serial_graph_and_eager_batched_zero_interference() -> None:
    """UT-1708 (FR1, double-sided): serial-graph and eager-batched intact.

    A serial graph-path child publish (``branch_id`` singular, through
    ``_publish_graph_child``) and a batched EAGER path step (no graph
    wiring) each produce healthy profiler statistics
    (``mean_stable_ratio`` in (0, 1]) -- the two record call sites this AR
    must not touch. The ``StabilityProfiler.record`` signature is pinned:
    its parameter (keyword) set is exactly the srs-pinned seven names.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4208)

    mff_serial = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff_serial, parent, branch_id="1708-sp")
    _serial(mff_serial, stacked[0:1], branch_id="1708-sc", parent_branch_id="1708-sp")
    serial_profile = GLOBAL_STABILITY_PROFILER.get_profile("1708-sc")
    assert serial_profile is not None, "the serial graph publish must profile the child"
    assert 0.0 < serial_profile.mean_stable_ratio <= 1.0

    mff_eager = _make_mff(model, device)
    _serial(mff_eager, parent, branch_id="1708-ep")
    eager_ids = _ids("1708-e")
    _batched(mff_eager, stacked, eager_ids, "1708-ep")
    for branch_id in eager_ids:
        eager_profile = GLOBAL_STABILITY_PROFILER.get_profile(branch_id)
        assert eager_profile is not None, "the eager batched path must profile"
        assert 0.0 < eager_profile.mean_stable_ratio <= 1.0

    parameters = inspect.signature(StabilityProfiler.record).parameters
    keyword_set = set(parameters) - {"self"}  # the unbound function carries `self`.
    assert keyword_set == {
        "branch_id",
        "parent_branch_id",
        "layer_idx",
        "step_idx",
        "stable_mask",
        "tau",
        "metric",
    }


# ---------------------------------------------------------------------------
# UT-1709: mutation spot-check -- reserved for the T002 mutation phase
# (design §5 table row 1709: the five mutations are applied against the
# Green implementation; not implemented in this Red-phase file).
# ---------------------------------------------------------------------------
