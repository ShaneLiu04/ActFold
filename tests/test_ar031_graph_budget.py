"""AR031 graph-face budget wiring unit tests (UT-3114..3120).

Red-phase contract for the AR031 T005 graph x budget wiring (design A4
frozen): ``FoldedGraphRunner`` gains a ``budget_static`` construction face
(per-layer 0-dim int64 device scalars; ``None`` = inactive, byte-identical),
the capture body applies the A1 exact-count cap (``apply_layer_budget``)
after the forced AND and before the count re-derivation, the capped mask is
written back into ``mask_buf`` (count / merge / publish share one source),
``ManualFoldedForward`` gains a ``budget_policy`` wiring (``None`` = zero
intrusion), replay back-fills ``budget_static[l]`` from the policy table
outside the captured body (one H2D copy, the AR021 ``forced_divergent_static``
precedent), the budget-on path bypasses the fused gate (#38) via the plain
scores path bit-identically, and the AR017 adaptive data flow is untouched
(the cap lowers staged divergences, which naturally suppresses growth and
enables shrink).

Red state (why every test below fails today): none of the wiring exists --
``FoldedGraphRunner.__init__`` rejects ``budget_static`` (TypeError), the
``budget_static`` signature slot / public probe is absent (AssertionError /
AttributeError), and ``ManualFoldedForward.__init__`` rejects ``budget_policy``
(TypeError).  Every UT is expected to fail through one of those missing
surfaces, never through a collection error: all imported production modules
exist at module import time.  Per the AR031 test plan each UT keeps at least
one CPU-collectible face (signature checks, A1-cap device purity, direct
static-body drives with manually allocated buffers -- the UT-2603 precedent);
full capture/replay faces run only when CUDA is available.

Fixtures follow ``tests/test_adaptive_capacity.py`` (self-contained L=2/H=32/
T=16/V=64/N=4 LLaMA-layout decoder; helpers copied isomorphically -- repo
precedent: no cross-test-module imports).  ``N * T = 64`` stays a power of
two (AR017 design fixture constraint: capacity arithmetic exactness).  The
single-host-sync contract binds production code only; test-side readbacks
are unrestricted (AR017 precedent).
"""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.adaptive_capacity import AdaptiveCapacityConfig, AdaptiveCapacityPolicy
from actfold.core.cuda_graph import FoldedGraphRunner
from actfold.core.layer_budget import LayerRecomputeBudgetPolicy, apply_layer_budget
from actfold.models.architecture_utils import ManualFoldedForward, detect_architecture

_VOCAB = 64
_HIDDEN = 32
_NUM_LAYERS = 2
_SEQ = 16
_NUM_CANDIDATES = 4
_NUM_TOKENS = _NUM_CANDIDATES * _SEQ
assert _NUM_TOKENS > 0 and _NUM_TOKENS & (_NUM_TOKENS - 1) == 0
_TAU = 0.95
_CACHE_BUDGET = 4096
_FUSED_GATE_MIN_TOKENS = 1024


# ---------------------------------------------------------------------------
# Self-contained synthetic decoder (LlamaLikeModel layout, copied from
# tests/test_adaptive_capacity.py -- repo precedent: fixtures are duplicated
# per test module, never imported across test modules).
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
# Shared scenario helpers (isomorphic copies of the AR017 test module's
# fixtures, extended with the AR031 budget-policy constructor flag).
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
    *,
    use_cuda_graph: bool = False,
    graph_capacity_ratio: float = 0.5,
    budget_policy: Any | None = None,
    graph_adaptive_capacity: bool = False,
    graph_adaptive_config: Any | None = None,
    split_layers: bool = False,
    cache_budget: int = _CACHE_BUDGET,
) -> ManualFoldedForward:
    """Fresh ManualFoldedForward (own cache + gate) over the shared model.

    The AR031 ``budget_policy`` and AR017 adaptive kwargs are forwarded only
    when set: default-off constructions stay call-identical to the existing
    wiring, so a Red failure can only surface on the test's own
    missing-production face (``budget_policy`` TypeError today).
    """
    cache = ActivationCache(max_entries_per_layer=cache_budget, device=device)
    gate = SimilarityGate(tau=_TAU, metric="cosine")
    kwargs: dict[str, Any] = {}
    if budget_policy is not None:
        kwargs["budget_policy"] = budget_policy
    if graph_adaptive_capacity or graph_adaptive_config is not None:
        kwargs["graph_adaptive_capacity"] = graph_adaptive_capacity
        kwargs["graph_adaptive_config"] = graph_adaptive_config
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        split_layers=split_layers,
        split_min_tokens=1,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=graph_capacity_ratio,
        **kwargs,
    )


def _make_candidates(
    device: str,
    changes_per_row: int = 1,
    seed: int = 4201,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a parent ``[1, T]`` and N equal-length mid-divergent candidates.

    Each candidate row replaces ``changes_per_row`` consecutive mid-sequence
    tokens with mutually distinct ids; the token-wise attention keeps the
    per-layer divergent count at exactly ``D = N * changes_per_row``.
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


def _full_divergent(parent: torch.Tensor) -> torch.Tensor:
    """Build ``[N, T]`` candidates differing from the parent at every position."""
    offsets = torch.arange(_NUM_CANDIDATES, device=parent.device).unsqueeze(1)
    return (parent + 1 + offsets) % _VOCAB


def _ids(tag: str) -> list[str]:
    """Build N branch ids sharing the ``tag`` prefix (disjoint per UT)."""
    return [f"{tag}{i}" for i in range(_NUM_CANDIDATES)]


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


def _batched_eager_reference(
    model: nn.Module,
    device: str,
    parent: torch.Tensor,
    stacked: torch.Tensor,
    branch_ids: list[str],
    parent_branch_id: str,
) -> tuple[ManualFoldedForward, torch.Tensor]:
    """Run the AR015 eager batched path once on a fresh stack (no graph)."""
    mff = _make_mff(model, device)
    _serial(mff, parent, branch_id=parent_branch_id)
    logits = _batched(mff, stacked, branch_ids, parent_branch_id)
    return mff, logits


def _user_warning_messages(record: list[warnings.WarningMessage]) -> list[str]:
    """Return the messages of every UserWarning in a catch_warnings log."""
    return [str(entry.message) for entry in record if issubclass(entry.category, UserWarning)]


def _parent_child_tokens(
    device: str,
    changes: int = 2,
    seed: int = 4300,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small serial fixture: ``[2, 6]`` parent and child with ``changes`` edits."""
    torch.manual_seed(seed)
    parent = torch.randint(0, _VOCAB, (2, 6), device=device)
    child = parent.clone()
    flat = child.reshape(-1)
    for pos in range(changes):
        flat[2 * pos + 1] = (flat[2 * pos + 1] + 1) % _VOCAB
    return parent, child


def _make_budget_runner(
    mff: ManualFoldedForward,
    profile: Any,
    budget_static: list[torch.Tensor] | None,
    capacity_ratio: float = 0.5,
) -> FoldedGraphRunner:
    """Construct a FoldedGraphRunner with the frozen A4 constructor face.

    ``budget_static`` is the AR031 wiring under test: ``None`` must capture
    the legacy body byte-for-byte; a per-layer 0-dim int64 list activates the
    in-graph A1 cap.  Today this kwarg does not exist (TypeError -- the Red).
    """
    return FoldedGraphRunner(
        wrapped_layers=mff._wrapped_layers,
        embed_fn=profile.embed_module,
        final_norm_fn=profile.final_norm,
        head_fn=profile.head_module,
        cache=mff.cache,
        gate_tau=mff.gate.tau,
        gate_eps=mff.gate.eps,
        capacity_ratio=capacity_ratio,
        attention_mask_static=None,
        budget_static=budget_static,
    )


def _allocate_static_buffers(runner: FoldedGraphRunner, child: torch.Tensor) -> None:
    """Mirror ``capture()``'s allocation block (the CUDA-only face skipped).

    ``capture()`` hard-requires CUDA tokens (its stream/graph machinery), so
    the CPU-collectible face allocates the static buffers with the exact
    shapes/dtypes ``capture()`` uses and pre-fills the parent through the
    REAL ``_prefill_parent`` (the UT-2603 precedent) -- the static body under
    test is unchanged production code.
    """
    batch, seq_len = child.shape
    num_layers = len(runner._layers)
    runner._num_tokens = batch * seq_len
    runner._capacity = max(
        1,
        min(
            math.ceil(runner._capacity_ratio * runner._num_tokens),
            runner._num_tokens,
        ),
    )
    runner._tokens_static = child.clone()
    probe = runner._embed_fn(runner._tokens_static)
    hidden_dim = probe.shape[-1]
    runner._parent_static = torch.empty(
        num_layers + 1, batch, seq_len, hidden_dim, dtype=probe.dtype, device=child.device
    )
    runner._mask_buf = torch.empty(
        num_layers, batch, seq_len, dtype=torch.bool, device=child.device
    )
    runner._count_buf = torch.zeros(num_layers, dtype=torch.int32, device=child.device)
    runner._child_buf = torch.empty(
        num_layers, batch, seq_len, hidden_dim, dtype=probe.dtype, device=child.device
    )


def _install_sync_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count host-sync entrypoints (``.item``/``.tolist``/``.cpu``) globally."""
    counts = {"item": 0, "tolist": 0, "cpu": 0}
    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist
    orig_cpu = torch.Tensor.cpu

    def _count_item(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        counts["item"] += 1
        return orig_item(self, *args, **kwargs)

    def _count_tolist(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        counts["tolist"] += 1
        return orig_tolist(self, *args, **kwargs)

    def _count_cpu(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        counts["cpu"] += 1
        return orig_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", _count_item)
    monkeypatch.setattr(torch.Tensor, "tolist", _count_tolist)
    monkeypatch.setattr(torch.Tensor, "cpu", _count_cpu)
    return counts


# ---------------------------------------------------------------------------
# UT-3114: capture-body zero synchronization
# ---------------------------------------------------------------------------


def test_ut3114_capture_body_zero_sync(device: str) -> None:
    """UT-3114: the budget-wired capture body adds zero host synchronization.

    CPU faces: (a) the A1 cap ``apply_layer_budget`` is device-pure (a
    monkeypatched ``.item``/``.tolist``/``.cpu`` probe counts zero calls);
    (b) a ``budget_static``-constructed runner driven through the static body
    directly (manual buffer allocation, UT-2603 precedent) stays sync-free
    and re-derives ``count_buf`` from the capped mask.  CUDA face: full
    ``capture`` under the probe stays at zero; a replay allows exactly the
    one ``validate_budgets`` ``tolist`` readback (#37 contract) and nothing
    else.

    Red reason: ``FoldedGraphRunner.__init__`` rejects ``budget_static``
    (TypeError) -- the A4 construction face does not exist yet.
    """
    # (a) A1 cap device purity: no host readback inside the cap.
    stable = torch.zeros(16, dtype=torch.bool, device=device)
    stable[:6] = True
    scores = torch.arange(16, dtype=torch.float32, device=device)
    with pytest.MonkeyPatch.context() as purity_probe:
        purity_counts = _install_sync_probe(purity_probe)
        capped = apply_layer_budget(stable, scores, 3)
    assert sum(purity_counts.values()) == 0
    assert int((~capped).sum()) == 3

    # (b) static-body drive with the budget face (TypeError today -> Red).
    model, profile = _build_model(device, seed=4314)
    parent, child = _parent_child_tokens(device, changes=4)
    mff = _make_mff(model, device)
    with torch.no_grad():
        mff(parent, branch_id="3114-parent")
    budget_static = [torch.tensor(2, dtype=torch.int64, device=device) for _ in range(_NUM_LAYERS)]
    runner = _make_budget_runner(mff, profile, budget_static)
    _allocate_static_buffers(runner, child)
    assert runner._prefill_parent("3114-parent")
    with pytest.MonkeyPatch.context() as body_probe:
        body_counts = _install_sync_probe(body_probe)
        with torch.no_grad():
            runner._static_forward()
    assert sum(body_counts.values()) == 0
    for layer_idx in range(_NUM_LAYERS):
        # count re-derivation reduces over the capped mask (one shared source)
        assert int(runner.count_buf[layer_idx].item()) == int(runner.mask_buf[layer_idx].sum())

    if not torch.cuda.is_available():
        return  # CPU faces above pin the contract; capture faces need CUDA

    device_c = "cuda"
    model_c, profile_c = _build_model(device_c, seed=4315)
    parent_c, child_c = _parent_child_tokens(device_c, changes=4)
    mff_c = _make_mff(model_c, device_c)
    with torch.no_grad():
        mff_c(parent_c, branch_id="3114-cuda-parent")
    budget_c = [torch.tensor(2, dtype=torch.int64, device=device_c) for _ in range(_NUM_LAYERS)]
    runner_c = _make_budget_runner(mff_c, profile_c, budget_c)
    with pytest.MonkeyPatch.context() as capture_probe:
        capture_counts = _install_sync_probe(capture_probe)
        with torch.no_grad():
            runner_c.capture(child_c, branch_id="3114-child", parent_branch_id="3114-cuda-parent")
    assert sum(capture_counts.values()) == 0, "capture body must stay sync-free"
    with pytest.MonkeyPatch.context() as replay_probe:
        replay_counts = _install_sync_probe(replay_probe)
        with torch.no_grad():
            out = runner_c.replay(
                child_c, parent_branch_id="3114-cuda-parent", branch_id="3114-child"
            )
    assert out is not None
    assert replay_counts["tolist"] == 1  # the single #37 validate_budgets readback
    assert replay_counts["item"] == 0
    assert replay_counts["cpu"] == 0


# ---------------------------------------------------------------------------
# UT-3115: capacity-conflict fallback (sum B_l > C)
# ---------------------------------------------------------------------------


def test_ut3115_capacity_conflict_falls_back_eager(device: str) -> None:
    """UT-3115: ``sum B_l > C`` replay falls back eager, correct, degradation +1.

    CPU face: the frozen A4 wiring signatures must exist (``budget_static`` on
    ``FoldedGraphRunner.__init__``, ``budget_policy`` on
    ``ManualFoldedForward.__init__``).  CUDA face: a budget table whose sum
    exceeds the captured per-layer capacity (ratio 0.9 -> sum B_l = 115 >
    C = 32) under a fully divergent load caps at ~57 > C, so the replay is
    discarded, the one-time budget warning fires, the step recomputes eagerly
    (output == the fresh-eager reference), and the degradation-fallback
    counter increments by exactly one.

    Red reason: both signature slots are absent (AssertionError).
    """
    runner_params = inspect.signature(FoldedGraphRunner.__init__).parameters
    assert "budget_static" in runner_params, "A4: runner constructor must gain budget_static"
    mff_params = inspect.signature(ManualFoldedForward.__init__).parameters
    assert "budget_policy" in mff_params, "A4: MFF constructor must gain budget_policy"
    # kwarg acceptance (CPU construction face)
    model, _profile = _build_model("cpu", seed=4316)
    budget = LayerRecomputeBudgetPolicy(
        budget_ratio=0.9, num_layers=_NUM_LAYERS, warmup_steps=1, window_steps=50, adaptive=False
    )
    _make_mff(model, "cpu", budget_policy=budget)

    if not torch.cuda.is_available():
        return  # signature faces above pin the contract; replay faces need CUDA

    device_c = "cuda"
    model_c, _profile_c = _build_model(device_c, seed=4317)
    parent, stacked = _make_candidates(device_c, changes_per_row=2, seed=4318)
    full = _full_divergent(parent)
    _mff_ref, ref_full = _batched_eager_reference(
        model_c, device_c, parent, full, _ids("3115-e"), "3115-ep"
    )
    budget_c = LayerRecomputeBudgetPolicy(
        budget_ratio=0.9, num_layers=_NUM_LAYERS, warmup_steps=1, window_steps=50, adaptive=False
    )
    mff = _make_mff(
        model_c, device_c, use_cuda_graph=True, graph_capacity_ratio=0.5, budget_policy=budget_c
    )
    _serial(mff, parent, branch_id="3115-p")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = _batched(mff, full, _ids("3115-a"), "3115-p")
    messages = _user_warning_messages(record)
    assert len(messages) == 1, "the capacity-conflict violation must warn exactly once"
    assert "budget" in messages[0].lower()
    assert torch.allclose(out, ref_full, atol=1e-4, rtol=1e-4)
    assert torch.equal(out.argmax(dim=-1), ref_full.argmax(dim=-1))
    runner = mff.graph_runner_batched
    assert runner is not None, "the capture itself succeeds; the conflict is per-replay"
    assert sum(budget_c.budget_table([_NUM_TOKENS, _NUM_TOKENS])) > runner.capacity
    assert runner.budget_exceeded is True
    assert mff._graph_batched_capture_failed is False
    stats = mff.graph_pool_stats
    assert stats["degradation_fallback_count"] == 1


# ---------------------------------------------------------------------------
# UT-3116: AR017 shrink linkage (budget re-estimate lowers divergence)
# ---------------------------------------------------------------------------


def test_ut3116_ar017_shrink_linkage(device: str) -> None:
    """UT-3116: the capped divergence flows into the AR017 shrink decision.

    CPU faces: the AR017 policy shrinks on a low-divergence window (pure
    logic sanity of the untouched data flow), and the MFF constructor accepts
    ``budget_policy`` (the A4 wiring face).  CUDA face: with ``D = 20`` and
    ``shrink_threshold * C = 16``, the budget-OFF arm never shrinks (20 >
    16); the budget-ON arm caps the staged divergence at ``max B_l`` (ratio
    0.15 -> B = [10, 9]), which slides under the shrink threshold, so after
    ``shrink_patience`` observations ``policy.events`` gains a
    ``{"kind": "recapture", "trigger": "shrink"}`` entry and the runner swaps
    to a smaller capacity that still carries the budget face.

    Red reason: ``ManualFoldedForward.__init__`` rejects ``budget_policy``
    (TypeError).
    """
    # Pure-policy sanity: the untouched AR017 shrink data flow.
    policy17 = AdaptiveCapacityPolicy(AdaptiveCapacityConfig(shrink_patience=2))
    first = policy17.observe(10, False, 32, _NUM_TOKENS)
    assert first.kind == "noop"  # window not yet full
    second = policy17.observe(10, False, 32, _NUM_TOKENS)
    assert second.kind == "recapture"  # 10 <= 0.5 * 32 -> shrink verdict
    policy17.note_recaptured()
    assert policy17.events[-1]["trigger"] == "shrink"

    # A4 wiring face (TypeError today -> Red).
    model, _profile = _build_model("cpu", seed=4319)
    budget = LayerRecomputeBudgetPolicy(
        budget_ratio=0.15, num_layers=_NUM_LAYERS, warmup_steps=1, window_steps=50, adaptive=False
    )
    _make_mff(model, "cpu", budget_policy=budget)

    if not torch.cuda.is_available():
        return  # policy/wiring faces above pin the contract; linkage needs CUDA

    device_c = "cuda"
    model_c, _profile_c = _build_model(device_c, seed=4320)
    parent, stacked = _make_candidates(device_c, changes_per_row=5, seed=4321)

    # Contrast arm (budget OFF): the raw demand D = 20 > 16 blocks shrink.
    mff_off = _make_mff(
        model_c,
        device_c,
        use_cuda_graph=True,
        graph_capacity_ratio=0.5,
        graph_adaptive_capacity=True,
        graph_adaptive_config=AdaptiveCapacityConfig(shrink_patience=2),
    )
    _serial(mff_off, parent, branch_id="3116-p0")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _batched(mff_off, stacked, _ids("3116-a0"), "3116-p0")
        _batched(mff_off, stacked, _ids("3116-b0"), "3116-p0")
    policy_off = mff_off.graph_adaptive_policy
    assert policy_off is not None
    assert policy_off.events == [], "uncapped D=20 > shrink threshold: no shrink without the cap"
    runner_off = mff_off.graph_runner_batched
    assert runner_off is not None
    assert runner_off.last_layer_divergences is not None
    assert max(runner_off.last_layer_divergences) == 20

    # Feature arm (budget ON): capped divergence <= max B_l = 10 <= 16.
    budget_c = LayerRecomputeBudgetPolicy(
        budget_ratio=0.15, num_layers=_NUM_LAYERS, warmup_steps=1, window_steps=50, adaptive=False
    )
    mff_on = _make_mff(
        model_c,
        device_c,
        use_cuda_graph=True,
        graph_capacity_ratio=0.5,
        budget_policy=budget_c,
        graph_adaptive_capacity=True,
        graph_adaptive_config=AdaptiveCapacityConfig(shrink_patience=2),
    )
    _serial(mff_on, parent, branch_id="3116-p1")
    with warnings.catch_warnings(record=True) as record_on:
        warnings.simplefilter("always")
        _batched(mff_on, stacked, _ids("3116-a1"), "3116-p1")
        runner_step1 = mff_on.graph_runner_batched
        _batched(mff_on, stacked, _ids("3116-b1"), "3116-p1")
    assert _user_warning_messages(record_on) == []
    assert runner_step1 is not None
    max_budget = max(budget_c.budget_table([_NUM_TOKENS, _NUM_TOKENS]))
    divs = runner_step1.last_layer_divergences
    assert divs is not None
    assert max(divs) == max_budget  # the staged divergence is the CAPPED one
    assert max(divs) < 20  # ...and strictly below the uncapped demand
    policy_on = mff_on.graph_adaptive_policy
    assert policy_on is not None
    shrink_events = [
        e for e in policy_on.events if e.get("kind") == "recapture" and e.get("trigger") == "shrink"
    ]
    assert len(shrink_events) == 1
    runner_on = mff_on.graph_runner_batched
    assert runner_on is not None
    assert runner_on is not runner_step1  # the shrink recapture swapped the runner
    assert runner_on.capacity < 32
    assert runner_on.budget_static is not None  # the re-capture threads the budget face


# ---------------------------------------------------------------------------
# UT-3117: no-policy zero regression (AR030 end-state parity)
# ---------------------------------------------------------------------------


def test_ut3117_no_policy_zero_regression(device: str) -> None:
    """UT-3117: without a policy the wiring is byte-identical to AR030.

    CPU faces: the A4 signature slots exist, an explicit ``budget_static=None``
    runner reproduces the legacy static body bit-exactly against the eager
    MFF child forward (UT-2603 parity), and the runner's budget face stays
    ``None``.  CUDA face: the full batched graph path without a policy
    matches the eager reference on both steps, replays deterministically
    (bit-equal), records zero degradation fallbacks, and captures a runner
    whose ``budget_static`` is ``None``.

    Red reason: the signature slots are absent (AssertionError).
    """
    runner_params = inspect.signature(FoldedGraphRunner.__init__).parameters
    assert "budget_static" in runner_params, "A4: runner constructor must gain budget_static"
    mff_params = inspect.signature(ManualFoldedForward.__init__).parameters
    assert "budget_policy" in mff_params, "A4: MFF constructor must gain budget_policy"

    model, profile = _build_model(device, seed=4322)
    parent, child = _parent_child_tokens(device, changes=2)
    mff = _make_mff(model, device)
    with torch.no_grad():
        mff(parent, branch_id="3117-parent")
        eager_out = mff(child, branch_id="3117-child", parent_branch_id="3117-parent")
    runner = _make_budget_runner(mff, profile, None)  # explicit None = inactive
    assert runner.budget_static is None
    _allocate_static_buffers(runner, child)
    assert runner._prefill_parent("3117-parent")
    with torch.no_grad():
        static_out = runner._static_forward()
    assert torch.equal(static_out, eager_out)  # legacy body, byte-for-byte

    if not torch.cuda.is_available():
        return  # parity faces above pin the contract; batched faces need CUDA

    device_c = "cuda"
    model_c, _profile_c = _build_model(device_c, seed=4323)
    parent_c, stacked = _make_candidates(device_c, changes_per_row=2, seed=4324)
    _mff_ref, ref = _batched_eager_reference(
        model_c, device_c, parent_c, stacked, _ids("3117-e"), "3117-ep"
    )
    mff_c = _make_mff(model_c, device_c, use_cuda_graph=True, graph_capacity_ratio=0.5)
    _serial(mff_c, parent_c, branch_id="3117-p")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out1 = _batched(mff_c, stacked, _ids("3117-a"), "3117-p")
        out2 = _batched(mff_c, stacked, _ids("3117-b"), "3117-p")
    assert _user_warning_messages(record) == []
    for out in (out1, out2):
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
        assert torch.equal(out.argmax(dim=-1), ref.argmax(dim=-1))
    assert torch.equal(out1, out2)  # deterministic replay sentinel
    runner_c = mff_c.graph_runner_batched
    assert runner_c is not None
    assert runner_c.budget_static is None
    stats = mff_c.graph_pool_stats
    assert stats["degradation_fallback_count"] == 0


# ---------------------------------------------------------------------------
# UT-3118: static out-of-body backfill (budget_static vs policy table)
# ---------------------------------------------------------------------------


def test_ut3118_static_out_of_body_backfill(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-3118: replay back-fills ``budget_static`` from the policy table.

    CPU faces: with a binding budget the capped mask is written back into
    ``mask_buf`` and ``count_buf`` reduces over it (same-source invariants:
    capped recompute fits ``B_l``, stable count never drops).  CUDA faces:
    every replay back-fills ``budget_static[l]`` with the policy's CURRENT
    table (skewed table after a window re-estimate, flipped table after the
    next one -- the values track the policy, not the capture), and the
    publish loop feeds ``observe_layer`` the CAPPED per-layer count, which
    equals the capped-mask reduction and ``count_buf``.

    Red reason: ``FoldedGraphRunner.__init__`` rejects ``budget_static``
    (TypeError on the reference construction below).
    """
    model, profile = _build_model(device, seed=4325)
    parent, child = _parent_child_tokens(device, changes=4)
    mff = _make_mff(model, device)
    with torch.no_grad():
        mff(parent, branch_id="3118-parent")
    num_tokens = child.numel()

    runner_ref = _make_budget_runner(mff, profile, None)  # TypeError today -> Red
    _allocate_static_buffers(runner_ref, child)
    assert runner_ref._prefill_parent("3118-parent")
    with torch.no_grad():
        runner_ref._static_forward()
    uncapped_counts = [int(runner_ref.count_buf[l].item()) for l in range(_NUM_LAYERS)]

    budgets = [2, 3]  # binding: uncapped divergence (4) exceeds both
    budget_static = [torch.tensor(b, dtype=torch.int64, device=device) for b in budgets]
    runner = _make_budget_runner(mff, profile, budget_static)
    _allocate_static_buffers(runner, child)
    assert runner._prefill_parent("3118-parent")
    with torch.no_grad():
        runner._static_forward()
    for layer_idx in range(_NUM_LAYERS):
        mask_sum = int(runner.mask_buf[layer_idx].sum())
        count = int(runner.count_buf[layer_idx].item())
        assert count == mask_sum  # count re-derivation over the capped mask
        assert count >= uncapped_counts[layer_idx]  # the cap only folds rows
        divergence = num_tokens - count
        assert divergence <= budgets[layer_idx]  # capped recompute fits B_l

    if not torch.cuda.is_available():
        return  # same-source faces above pin the contract; backfill needs CUDA

    device_c = "cuda"
    model_c, _profile_c = _build_model(device_c, seed=4326)
    parent_c, stacked = _make_candidates(device_c, changes_per_row=2, seed=4327)
    budget = LayerRecomputeBudgetPolicy(
        budget_ratio=0.25, num_layers=_NUM_LAYERS, warmup_steps=1, window_steps=1, adaptive=True
    )
    mff_c = _make_mff(
        model_c, device_c, use_cuda_graph=True, graph_capacity_ratio=0.5, budget_policy=budget
    )
    _serial(mff_c, parent_c, branch_id="3118-p")

    # Probe: record every publish-loop observe_layer call (0-dim count clone).
    recorded: list[tuple[int, torch.Tensor, int]] = []
    orig_observe = budget.observe_layer

    def _record_observe(layer_idx: int, stable_count: torch.Tensor, num_tokens: int) -> None:
        recorded.append((layer_idx, stable_count.detach().clone(), int(num_tokens)))
        orig_observe(layer_idx, stable_count, num_tokens)

    monkeypatch.setattr(budget, "observe_layer", _record_observe)

    # Deterministic window driving under ANY production begin_step cadence:
    # every begin_step first receives a dominant (10^6-token) skew feed, so
    # the latest re-estimate always reflects the current skew regardless of
    # who advances the step counter (window_steps=1 re-estimates every step).
    feed_tokens = 10**6
    skew = {"layer_stable": 0}
    orig_begin = budget.begin_step

    def _begin_with_feed() -> None:
        stable_layer = skew["layer_stable"]
        unstable_layer = 1 - stable_layer
        with torch.no_grad():
            orig_observe(
                stable_layer,
                torch.tensor(float(feed_tokens), device=device_c),
                feed_tokens,
            )
            orig_observe(
                unstable_layer,
                torch.tensor(0.0, device=device_c),
                feed_tokens,
            )
        orig_begin()

    monkeypatch.setattr(budget, "begin_step", _begin_with_feed)

    def _table() -> list[int]:
        return [budget.budget_for(l, _NUM_TOKENS) for l in range(_NUM_LAYERS)]

    def _static_values() -> list[int]:
        runner_now = mff_c.graph_runner_batched
        assert runner_now is not None
        bs = runner_now.budget_static
        assert bs is not None and len(bs) == _NUM_LAYERS
        return [int(bs[l].item()) for l in range(_NUM_LAYERS)]

    # Step 1 (skew A: layer 0 all-stable -> B_1 >> B_0).
    skew["layer_stable"] = 0
    budget.begin_step()
    with warnings.catch_warnings(record=True) as record1:
        warnings.simplefilter("always")
        _batched(mff_c, stacked, _ids("3118-a"), "3118-p")
    assert _user_warning_messages(record1) == []
    values1 = _static_values()
    assert values1 == _table()  # backfill == the policy's current table
    assert values1[1] > values1[0]  # the re-estimated skew reached the buffers

    # The publish loop observed the CAPPED counts (one call per layer).
    publish_obs = [(l, c, n) for (l, c, n) in recorded if n == _NUM_TOKENS]
    assert [l for l, _, _ in publish_obs] == [0, 1]
    runner1 = mff_c.graph_runner_batched
    assert runner1 is not None
    for layer_idx, count_tensor, num_tok in publish_obs:
        mask_sum = int(runner1.mask_buf[layer_idx].sum())
        assert int(count_tensor.item()) == mask_sum  # capped-mask reduction
        assert int(runner1.count_buf[layer_idx].item()) == mask_sum
        assert num_tok == _NUM_TOKENS

    # Step 2 (skew B flipped): the next replay back-fills the NEW table.
    skew["layer_stable"] = 1
    budget.begin_step()
    with warnings.catch_warnings(record=True) as record2:
        warnings.simplefilter("always")
        _batched(mff_c, stacked, _ids("3118-b"), "3118-p")
    assert _user_warning_messages(record2) == []
    values2 = _static_values()
    assert values2 == _table()
    assert values2[0] > values2[1]  # flipped skew -> flipped allocation
    assert values2 != values1  # the window re-estimate moved the table


# ---------------------------------------------------------------------------
# UT-3119: growth suppression (cap lowers the violation count)
# ---------------------------------------------------------------------------


def test_ut3119_growth_suppression(device: str) -> None:
    """UT-3119: budget-on violations never exceed budget-off violations.

    CPU faces: the A1 cap is directionally divergence-reducing (a synthetic
    would-be violation ``D > C`` becomes a pass after the cap), and the MFF
    constructor accepts ``budget_policy``.  CUDA face: with ``C = 8`` and a
    ``D = 12`` load, the budget-OFF arm discards every replay (three
    degradation fallbacks); the budget-ON arm caps at ``B_l = 3 <= C`` and
    records zero violations -- the growth signal the AR017 hook observes is
    suppressed by the cap, with no code change on the AR017 side.

    Red reason: ``ManualFoldedForward.__init__`` rejects ``budget_policy``
    (TypeError).
    """
    # Synthetic directionality: the cap never increases divergence.
    torch.manual_seed(4328)
    stable = torch.zeros(16, dtype=torch.bool, device=device)
    stable[:6] = True  # D = 10 > C_synth = 8: a would-be violation
    scores = torch.rand(16, device=device)
    capacity_synth = 8
    assert int((~stable).sum()) > capacity_synth
    capped = apply_layer_budget(stable, scores, 3)
    assert int((~capped).sum()) <= capacity_synth  # capped divergence fits

    # A4 wiring face (TypeError today -> Red).
    model, _profile = _build_model("cpu", seed=4329)
    budget = LayerRecomputeBudgetPolicy(
        budget_ratio=0.05, num_layers=_NUM_LAYERS, warmup_steps=1, window_steps=50, adaptive=False
    )
    _make_mff(model, "cpu", budget_policy=budget)

    if not torch.cuda.is_available():
        return  # directionality/wiring faces above; the A/B arm needs CUDA

    device_c = "cuda"
    model_c, _profile_c = _build_model(device_c, seed=4330)
    parent, stacked = _make_candidates(device_c, changes_per_row=3, seed=4331)  # D = 12
    steps = 3

    # Budget OFF: C = ceil(0.125 * 64) = 8 < 12 -> every replay is discarded.
    mff_off = _make_mff(model_c, device_c, use_cuda_graph=True, graph_capacity_ratio=0.125)
    _serial(mff_off, parent, branch_id="3119-p0")
    for tag in ("3119-a0", "3119-b0", "3119-c0"):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _batched(mff_off, stacked, _ids(tag), "3119-p0")
    violations_off = mff_off.graph_pool_stats["degradation_fallback_count"]
    assert violations_off == steps

    # Budget ON: B = [3, 3] -> capped divergence 3 <= C = 8 -> zero violations.
    budget_c = LayerRecomputeBudgetPolicy(
        budget_ratio=0.05, num_layers=_NUM_LAYERS, warmup_steps=1, window_steps=50, adaptive=False
    )
    mff_on = _make_mff(
        model_c,
        device_c,
        use_cuda_graph=True,
        graph_capacity_ratio=0.125,
        budget_policy=budget_c,
    )
    _serial(mff_on, parent, branch_id="3119-p1")
    for tag in ("3119-a1", "3119-b1", "3119-c1"):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _batched(mff_on, stacked, _ids(tag), "3119-p1")
    violations_on = mff_on.graph_pool_stats["degradation_fallback_count"]
    assert violations_on <= violations_off
    assert violations_on == 0  # the cap absorbs the burst entirely
    runner_on = mff_on.graph_runner_batched
    assert runner_on is not None
    assert runner_on.budget_exceeded is False


# ---------------------------------------------------------------------------
# UT-3120: fused-gate bypass (scores path bit-identical, #38)
# ---------------------------------------------------------------------------


def test_ut3120_fused_gate_bypass(device: str) -> None:
    """UT-3120: budget-on bypasses the fused gate with bit-identical results.

    CPU face: a NON-binding budget (``B_l >= N``) changes nothing -- the
    scores-path static body reproduces the plain-gate static body bit-for-bit
    (logits and per-layer masks).  CUDA face: at ``B * T >= 1024`` (the #38
    fused dispatch window) a fused-path runner and a budget-on runner
    (non-binding) capture and replay to identical masks, counts and logits --
    the bypass must be a pure re-expression of the same gate math.

    Red reason: ``FoldedGraphRunner.__init__`` rejects ``budget_static``
    (TypeError on the plain-runner construction below).
    """
    model, profile = _build_model(device, seed=4332)
    parent, child = _parent_child_tokens(device, changes=2)
    mff = _make_mff(model, device)
    with torch.no_grad():
        mff(parent, branch_id="3120-parent")

    runner_plain = _make_budget_runner(mff, profile, None)  # TypeError today -> Red
    _allocate_static_buffers(runner_plain, child)
    assert runner_plain._prefill_parent("3120-parent")
    with torch.no_grad():
        out_plain = runner_plain._static_forward()

    num_tokens = child.numel()
    budget_static = [
        torch.tensor(num_tokens, dtype=torch.int64, device=device) for _ in range(_NUM_LAYERS)
    ]
    runner_scores = _make_budget_runner(mff, profile, budget_static)
    _allocate_static_buffers(runner_scores, child)
    assert runner_scores._prefill_parent("3120-parent")
    with torch.no_grad():
        out_scores = runner_scores._static_forward()
    assert torch.equal(out_scores, out_plain)
    for layer_idx in range(_NUM_LAYERS):
        assert torch.equal(runner_scores.mask_buf[layer_idx], runner_plain.mask_buf[layer_idx])

    if not torch.cuda.is_available():
        return  # identity face above pins the contract; the fused face needs CUDA

    device_c = "cuda"
    batch, seq = 4, 256  # B * T = 1024 >= _FUSED_GATE_MIN_TOKENS (#38 window)
    model_c, profile_c = _build_model(device_c, seed=4333)
    torch.manual_seed(4334)
    parent_c = torch.randint(0, _VOCAB, (batch, seq), device=device_c)
    child_c = parent_c.clone()
    flat = child_c.reshape(-1)
    # Six scattered edits inside the 1024-token grid (the Red-era fixture
    # carried indices for a larger grid and crashed before any assertion --
    # repaired in Green, contract unchanged).
    for pos in (100, 260, 420, 580, 740, 900):
        flat[pos] = (flat[pos] + 1) % _VOCAB
    mff_c = _make_mff(model_c, device_c)
    with torch.no_grad():
        mff_c(parent_c, branch_id="3120-cuda-parent")

    runner_fused = _make_budget_runner(mff_c, profile_c, None)
    budget_c = [
        torch.tensor(batch * seq, dtype=torch.int64, device=device_c) for _ in range(_NUM_LAYERS)
    ]
    runner_bypass = _make_budget_runner(mff_c, profile_c, budget_c)
    with torch.no_grad():
        runner_fused.capture(child_c, branch_id="3120-f", parent_branch_id="3120-cuda-parent")
        runner_bypass.capture(child_c, branch_id="3120-s", parent_branch_id="3120-cuda-parent")
        out_fused = runner_fused.replay(
            child_c, parent_branch_id="3120-cuda-parent", branch_id="3120-f"
        )
        out_bypass = runner_bypass.replay(
            child_c, parent_branch_id="3120-cuda-parent", branch_id="3120-s"
        )
    assert out_fused is not None and out_bypass is not None
    assert torch.equal(out_bypass, out_fused)  # bit-identical results
    for layer_idx in range(_NUM_LAYERS):
        assert torch.equal(runner_bypass.mask_buf[layer_idx], runner_fused.mask_buf[layer_idx])
        assert torch.equal(runner_bypass.count_buf[layer_idx], runner_fused.count_buf[layer_idx])


# ---------------------------------------------------------------------------
# UT-3165: position-level 1-D forced static on the budget face
# ---------------------------------------------------------------------------


def test_ut3165_position_level_forced_one_dim_static_body(device: str) -> None:
    """UT-3165: the #43 position-level ``[T]`` forced static on the graph budget face.

    The static body ANDs ``~forced`` into the gate mask (broadcast) and the
    in-graph A1 cap consumes the same forced exemption; a 1-D static must
    not raise the cap's same-shape ValueError and must stay bit-identical
    to the explicit ``[batch, seq]`` static form (mask buffers and output).
    """
    model, profile = _build_model(device, seed=4335)
    parent, child = _parent_child_tokens(device, changes=4)
    mff = _make_mff(model, device)
    with torch.no_grad():
        mff(parent, branch_id="3165-parent")
    budget_static = [torch.tensor(2, dtype=torch.int64, device=device) for _ in range(_NUM_LAYERS)]
    forced = torch.zeros(child.shape[1], dtype=torch.bool, device=device)
    forced[-6:] = True

    outs = []
    masks = []
    for form in (forced, forced.unsqueeze(0).expand(child.shape)):
        runner = _make_budget_runner(mff, profile, budget_static)
        runner._forced_divergent_static = form.detach().clone()
        _allocate_static_buffers(runner, child)
        assert runner._prefill_parent("3165-parent")
        with torch.no_grad():
            outs.append(runner._static_forward())
        masks.append(runner.mask_buf.clone())
    for layer_idx in range(_NUM_LAYERS):
        assert torch.equal(masks[0][layer_idx], masks[1][layer_idx])
    assert torch.equal(outs[0], outs[1])
