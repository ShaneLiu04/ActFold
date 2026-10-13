"""Red-phase tests for AR017: adaptive divergent-capacity re-capture (UT-1601..1609).

Contract: ``specs/changes/AR017-adaptive-capacity/design.md`` -- §2 (policy
module contract: ``AdaptiveCapacityConfig`` / ``AdaptiveAction`` /
``AdaptiveCapacityPolicy`` fields, defaults, decision algorithm, window trim,
``events`` structure), §3 (``FoldedGraphRunner.last_layer_divergences``
staging semantics), §4 (hooks / re-capture lifecycle / the
``graph_adaptive_policy`` public probe / the ``_recapture_batched_runner``
prospective guard) and §6 (per-UT assertion faces plus fixture constraints);
acceptance wording from ``srs.md`` §3.4 FR4.

Red state (why every implemented test below fails today): the production
pieces do not exist yet -- ``actfold.core.adaptive_capacity`` (ImportError),
the ``ManualFoldedForward(graph_adaptive_capacity=..., graph_adaptive_config=...)``
constructor parameters (TypeError), the ``graph_adaptive_policy`` public
probe (AttributeError) and ``FoldedGraphRunner.last_layer_divergences``
(AttributeError). All eight implemented UTs (1601-1608) are expected to fail
in the Red run, each through one of those missing-production surfaces and
never through a test-side bug; UT-1609 (mutation spot-check) is reserved for
the T002 mutation phase and is intentionally not implemented here, so the
file leaves only its numbered section space.

Import organization: the AR017 policy module is imported lazily through
``_adaptive_module()`` instead of at module top level. A top-level import of
the missing module would turn the whole file into a collection error and
hide UT-1601's own Red report (UT-1601 is pure CPU logic and must stay
runnable); the deferred import converts the missing module into a per-test
``pytest.fail`` carrying the original ImportError message, so UT-1601
reports independently and each CUDA test surfaces its own missing surface
(ImportError guard, constructor TypeError, or probe AttributeError).

CUDA gating (design §6): a module-level ``pytestmark`` would skip the
CPU-runnable UT-1601 along with the CUDA tests, so gating uses the module
constant ``_cuda`` applied per function; UT-1601 carries no mark.

Fixtures follow ``tests/test_batched_graph_forward.py`` (self-contained
L=2/H=32/T=16/V=64/N=4 synthetic LLaMA-layout decoder; ``_build_model`` /
``_make_mff`` / candidate builders copied isomorphically into this file --
no cross-test-module imports, repo precedent). ``N * T = 64`` is a power of
two and MUST stay one (design §6 fixture constraint, §4.4 exactness): the
exact capacity arithmetic ``ceil(ratio * N * T) == target`` that the
capacity assertions rely on holds only for power-of-two token counts.

Tolerances (design §6, AR016 precedent): graph-vs-eager logits
``allclose(atol=1e-4, rtol=1e-4)`` plus per-row argmax equality.
Test-side readbacks (``count_buf.tolist()``) are unrestricted -- the
single-host-sync contract binds production code, not tests.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.cuda_graph import FoldedGraphRunner
from actfold.models.architecture_utils import ManualFoldedForward, detect_architecture

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_VOCAB = 64
_HIDDEN = 32
_NUM_LAYERS = 2
_SEQ = 16
_NUM_CANDIDATES = 4
# N * T MUST remain a power of two (design §6 fixture constraint): capacity
# conversions ceil(ratio * N * T) are exact only for power-of-two totals.
_NUM_TOKENS = _NUM_CANDIDATES * _SEQ
assert _NUM_TOKENS > 0 and _NUM_TOKENS & (_NUM_TOKENS - 1) == 0
_TAU = 0.95
_CACHE_BUDGET = 1024


def _adaptive_module() -> Any:
    """Import ``actfold.core.adaptive_capacity`` or fail the test (Red).

    Deferred import (see the module docstring for the rationale): a
    module-top-level import of the not-yet-existing AR017 module would
    collection-error the whole file including the CPU-runnable UT-1601.
    """
    try:
        import actfold.core.adaptive_capacity as adaptive_capacity
    except ImportError as exc:
        pytest.fail(f"actfold.core.adaptive_capacity is missing (Red, AR017): {exc}")
    return adaptive_capacity


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
        self.model.layers = nn.ModuleList(
            [_LlamaLikeLayer(_HIDDEN) for _ in range(_NUM_LAYERS)]
        )
        self.model.norm = nn.LayerNorm(_HIDDEN)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))


# ---------------------------------------------------------------------------
# Shared scenario helpers (isomorphic copies of the AR016 test module's
# fixtures, extended with the AR017 adaptive-capacity constructor flags).
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
    graph_adaptive_capacity: bool = False,
    graph_adaptive_config: Any | None = None,
) -> ManualFoldedForward:
    """Fresh ManualFoldedForward (own cache + gate) over the shared model.

    Follows the test_batched_graph_forward.py wiring: split layers with a
    lowered ``split_min_tokens`` so the split FFN path engages at the test
    shapes, plus the AR002 graph flags and the AR017 adaptive-capacity
    flags (``graph_adaptive_capacity`` / ``graph_adaptive_config`` -- the
    Red-phase surfaces under test). The adaptive kwargs are forwarded only
    when set: the default-off construction (used by every eager reference
    and by UT-1606) stays call-identical to the AR016 wiring, so a Red
    failure can only surface on the test's own missing-production face.
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
        split_layers=True,
        split_min_tokens=1,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=graph_capacity_ratio,
        **adaptive_kwargs,
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
    suffix, a few divergent positions per row). The per-layer divergent
    count is ``D = N * changes_per_row`` (UT-1505 precedent: changes=3
    gives D=12), so ``changes_per_row`` tunes the D_frac load.
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
    """Build ``[N, T]`` candidates differing from ``parent`` at every position.

    Row ``i`` shifts every parent token by ``1 + i`` (mod vocab), so the
    gate marks every position divergent at every layer (D = N * T) -- the
    deterministic all-divergent load for growth / budget-exceeded steps.
    """
    offsets = torch.arange(_NUM_CANDIDATES, device=parent.device).unsqueeze(1)
    return (parent + 1 + offsets) % _VOCAB


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


def _batched_eager_reference(
    model: nn.Module,
    device: str,
    parent: torch.Tensor,
    stacked: torch.Tensor,
    branch_ids: list[str],
    parent_branch_id: str,
) -> tuple[ManualFoldedForward, torch.Tensor]:
    """Run the AR015 eager batched path once on a fresh stack (no graph).

    A fresh ManualFoldedForward over the shared raw model carries its own
    cache and gate, so the reference can never observe the graph stack's
    activations. Returns ``(mff, logits)``.
    """
    mff = _make_mff(model, device)
    _serial(mff, parent, branch_id=parent_branch_id)
    logits = _batched(mff, stacked, branch_ids, parent_branch_id)
    return mff, logits


def _user_warning_messages(record: Any) -> list[str]:
    """Return the messages of every UserWarning in a catch_warnings log."""
    return [
        str(entry.message)
        for entry in record
        if issubclass(entry.category, UserWarning)
    ]


def _make_runner(
    profile: Any,
    mff: ManualFoldedForward,
    capacity_ratio: float = 0.5,
) -> FoldedGraphRunner:
    """Directly construct a FoldedGraphRunner (test_cuda_graph.py §4.3.4
    constructor precedent; used by UT-1607 to probe the observation seam
    without going through the ManualFoldedForward graph dispatch)."""
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
    )


# ---------------------------------------------------------------------------
# UT-1601: policy pure logic, table-driven (CPU -- no CUDA mark)
# ---------------------------------------------------------------------------


def test_ut1601_policy_table_driven() -> None:
    """UT-1601 (design §6 row 1601, srs §3.4 FR4): policy table-driven faces.

    Assertion faces: growth target ``ceil(grow_factor * maxD)`` clamped to
    ``num_tokens``; growth no-op once ``target <= C`` (capacity already at
    the N*T ceiling); shrink patience counting / beta trigger / ``target``
    inside ``[1, C-1]``; shrink no-op at ``C <= 1`` (empty clamp interval);
    explicit direction monotonicity (grow ``target >= maxD``, shrink
    ``target < C`` and ``target >=`` the recent-window max); window
    trimming to the last ``patience`` observations; interval throttling
    (``steps_since_recapture`` initialized to the interval so the first
    observe is never throttled; ``note_recaptured`` resets it to zero and
    the counter re-accumulates); ``max_recaptures`` retirement returning
    ``kind="retired"`` followed by permanent no-ops with a frozen window;
    ``AdaptiveCapacityConfig`` boundary ``ValueError`` for every illegal
    knob; and deterministic action sequences across two fresh policy
    instances fed the same observation stream.

    Red reason: ``actfold.core.adaptive_capacity`` does not exist yet, so
    ``_adaptive_module()`` fails this test with the ImportError.
    """
    mod = _adaptive_module()
    config_cls = mod.AdaptiveCapacityConfig
    policy_cls = mod.AdaptiveCapacityPolicy
    num_tokens = _NUM_TOKENS

    table: list[dict[str, Any]] = [
        {
            "name": "grow-basic",
            "config": {},
            "capacity": 8,
            "observations": [(20, True)],
            "expected": [("recapture", 25, "grow")],
        },
        {
            "name": "grow-clamp-num-tokens",
            "config": {},
            "capacity": 8,
            "observations": [(60, True)],
            "expected": [("recapture", 64, "grow")],
        },
        {
            "name": "grow-noop-at-full-capacity",
            "config": {},
            "capacity": 64,
            "observations": [(52, True)],
            "expected": [("noop", None, None)],
        },
        {
            "name": "shrink-needs-patience",
            "config": {},
            "capacity": 32,
            "observations": [(5, False), (5, False), (5, False)],
            "expected": [
                ("noop", None, None),
                ("noop", None, None),
                ("recapture", 7, "shrink"),
            ],
        },
        {
            "name": "shrink-beta-block",
            "config": {},
            "capacity": 32,
            "observations": [(5, False), (5, False), (20, False)],
            "expected": [("noop", None, None)] * 3,
        },
        {
            "name": "shrink-exceeded-in-window-block",
            "config": {},
            "capacity": 32,
            "observations": [(5, False), (3, True), (5, False)],
            "expected": [("noop", None, None)] * 3,
        },
        {
            "name": "shrink-capacity-one-noop",
            "config": {},
            "capacity": 1,
            "observations": [(0, False), (0, False), (0, False)],
            "expected": [("noop", None, None)] * 3,
        },
        {
            "name": "window-trim-old-exceeded",
            "config": {},
            "capacity": 32,
            "observations": [(28, True), (5, False), (5, False), (5, False)],
            "expected": [
                ("recapture", 35, "grow"),
                ("noop", None, None),
                ("noop", None, None),
                ("recapture", 7, "shrink"),
            ],
        },
        {
            "name": "interval-init-not-throttled",
            "config": {"min_recapture_interval": 3},
            "capacity": 8,
            "observations": [(20, True)],
            "expected": [("recapture", 25, "grow")],
        },
        {
            "name": "shrink-patience-two",
            "config": {"shrink_patience": 2},
            "capacity": 32,
            "observations": [(5, False), (5, False)],
            "expected": [("noop", None, None), ("recapture", 7, "shrink")],
        },
    ]
    for case in table:
        policy = policy_cls(config_cls(**case["config"]))
        got: list[tuple[str, int | None]] = []
        for max_d, exceeded in case["observations"]:
            action = policy.observe(max_d, exceeded, case["capacity"], num_tokens)
            got.append((action.kind, action.target_capacity))
        expected = [(kind, target) for kind, target, _trigger in case["expected"]]
        assert got == expected, case["name"]
        patience = case["config"].get("shrink_patience", 3)
        for idx, (max_d, _exceeded) in enumerate(case["observations"]):
            kind, target, trigger = case["expected"][idx]
            if kind != "recapture":
                continue
            assert isinstance(target, int), case["name"]
            if trigger == "grow":
                # Direction monotonicity (srs FR2): grow target >= maxD and
                # the num_tokens clamp keeps it inside the legal range.
                assert target >= max_d, case["name"]
                assert target <= num_tokens, case["name"]
            else:
                # Shrink: target in [1, C-1] and >= the recent-window max.
                assert trigger == "shrink", case["name"]
                recent = case["observations"][idx - patience + 1 : idx + 1]
                recent_max = max(d for d, _flag in recent)
                assert 1 <= target < case["capacity"], case["name"]
                assert target >= recent_max, case["name"]

    # Interval throttle: note_recaptured resets the step counter to zero,
    # the next observe is throttled, the one after is not (default
    # min_recapture_interval=2; the un-throttled first observe above also
    # pins the init-to-interval semantics).
    policy = policy_cls(config_cls())
    action = policy.observe(20, True, 8, num_tokens)
    assert (action.kind, action.target_capacity) == ("recapture", 25)
    policy.note_recaptured()
    throttled = policy.observe(30, True, 25, num_tokens)
    assert throttled.kind == "noop"
    unthrottled = policy.observe(30, True, 25, num_tokens)
    assert (unthrottled.kind, unthrottled.target_capacity) == ("recapture", 38)

    # Retirement at max_recaptures: the boundary observe returns
    # kind="retired", sets _retired, and every later observe is a plain
    # no-op (window frozen, no further events).
    policy = policy_cls(config_cls(min_recapture_interval=1, max_recaptures=2))
    action = policy.observe(20, True, 8, num_tokens)
    assert (action.kind, action.target_capacity) == ("recapture", 25)
    policy.note_recaptured()
    action = policy.observe(30, True, 25, num_tokens)
    assert (action.kind, action.target_capacity) == ("recapture", 38)
    policy.note_recaptured()
    retired = policy.observe(40, True, 38, num_tokens)
    assert retired.kind == "retired"
    assert retired.target_capacity is None
    assert policy._retired is True
    after_retire = policy.observe(10, False, 38, num_tokens)
    assert (after_retire.kind, after_retire.target_capacity) == ("noop", None)
    after_retire = policy.observe(40, True, 38, num_tokens)
    assert (after_retire.kind, after_retire.target_capacity) == ("noop", None)
    recapture_events = [e for e in policy.events if e.get("kind") == "recapture"]
    assert len(recapture_events) == 2
    assert any(e.get("kind") == "retired" for e in policy.events)

    # Manual retire() (the re-capture-failure path, UT-1605): permanent
    # no-op immediately, _retired latched.
    policy = policy_cls(config_cls())
    policy.retire()
    action = policy.observe(20, True, 8, num_tokens)
    assert (action.kind, action.target_capacity) == ("noop", None)
    assert policy._retired is True

    # Config boundary validation: every illegal knob raises ValueError.
    invalid_configs: list[dict[str, Any]] = [
        {"grow_factor": 0.9},
        {"shrink_threshold": 0.0},
        {"shrink_threshold": 1.0},
        {"shrink_patience": 0},
        {"min_recapture_interval": 0},
        {"max_recaptures": 0},
    ]
    for kwargs in invalid_configs:
        with pytest.raises(ValueError):
            config_cls(**kwargs)
    config_cls(
        grow_factor=1.0,
        shrink_threshold=0.99,
        shrink_patience=1,
        min_recapture_interval=1,
        max_recaptures=1,
    )

    # Determinism: two fresh instances fed the same observation stream
    # (with mirrored note_recaptured calls) produce equal action sequences.
    def _run_sequence(instance: Any, observations: list[Any]) -> list[Any]:
        actions = []
        for max_d, exceeded, capacity in observations:
            action = instance.observe(max_d, exceeded, capacity, num_tokens)
            actions.append((action.kind, action.target_capacity))
            if action.kind == "recapture":
                instance.note_recaptured()
        return actions

    mixed = [
        (20, True, 8),
        (5, False, 25),
        (5, False, 25),
        (30, True, 25),
        (8, False, 64),
        (8, False, 64),
        (12, False, 64),
        (50, True, 64),
    ]
    jitter_cfg = {
        "shrink_patience": 2,
        "min_recapture_interval": 1,
        "max_recaptures": 2,
    }
    sequence_a = _run_sequence(policy_cls(config_cls(**jitter_cfg)), mixed)
    sequence_b = _run_sequence(policy_cls(config_cls(**jitter_cfg)), mixed)
    assert sequence_a == sequence_b
    kinds = [kind for kind, _target in sequence_a]
    assert "recapture" in kinds
    assert "retired" in kinds


# ---------------------------------------------------------------------------
# UT-1602: shrink end-to-end
# ---------------------------------------------------------------------------


@_cuda
def test_ut1602_shrink_end_to_end() -> None:
    """UT-1602 (design §6 row 1602, srs §3.4 FR4): sustained low load shrinks.

    D_frac load: ``changes_per_row=2`` gives D = 8 of N*T = 64 (D_frac =
    0.125, the AR016 mid-divergence scenario family). Initial
    ``graph_capacity_ratio=0.5`` (C = 32, exact on the 2^6 fixture) with the
    adaptive path on; ``AdaptiveCapacityConfig(shrink_patience=2)``
    deviates from the default 3 to shorten the patience wait (declared
    here per the design §6 note). The capture call observes once (no-op),
    the second call triggers the shrink re-capture, and the assertions are
    placed after the third call (loose placement per design §6): the
    runner object is new (``is not``), ``capacity < 32`` (below the
    initial ceil(0.5 * 64)), ``capacity >= max(last_layer_divergences)``
    (the shrunk graph still fits the real divergent demand), the budget
    passes on every step, zero warnings are emitted, and all three step
    outputs match the fresh-eager batched reference (allclose 1e-4 plus
    per-row argmax).

    Red reason: ``actfold.core.adaptive_capacity`` does not exist yet, so
    ``_adaptive_module()`` fails this test with the ImportError.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, changes_per_row=2, seed=4212)

    _mff_eager, ref = _batched_eager_reference(
        model, device, parent, stacked, _ids("1602-e"), "1602-ep"
    )

    mod = _adaptive_module()
    config_cls = mod.AdaptiveCapacityConfig
    mff = _make_mff(
        model,
        device,
        use_cuda_graph=True,
        graph_capacity_ratio=0.5,
        graph_adaptive_capacity=True,
        graph_adaptive_config=config_cls(shrink_patience=2),
    )
    _serial(mff, parent, branch_id="1602-p")

    with warnings.catch_warnings(record=True) as record_all:
        warnings.simplefilter("always")
        out1 = _batched(mff, stacked, _ids("1602-a"), "1602-p")
        runner_old = mff.graph_runner_batched
        assert runner_old is not None
        assert runner_old.capacity == 32
        assert runner_old.budget_exceeded is False
        out2 = _batched(mff, stacked, _ids("1602-b"), "1602-p")
        out3 = _batched(mff, stacked, _ids("1602-c"), "1602-p")
    assert len(_user_warning_messages(record_all)) == 0

    runner = mff.graph_runner_batched
    assert runner is not None
    assert runner is not runner_old
    assert runner.capacity < 32
    divergences = runner.last_layer_divergences
    assert divergences is not None
    assert runner.capacity >= max(divergences)
    assert runner.budget_exceeded is False

    for out in (out1, out2, out3):
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
        assert torch.equal(out.argmax(dim=-1), ref.argmax(dim=-1))


# ---------------------------------------------------------------------------
# UT-1603: growth end-to-end
# ---------------------------------------------------------------------------


@_cuda
def test_ut1603_growth_end_to_end() -> None:
    """UT-1603 (design §6 row 1603, srs §3.4 FR4): tiny capacity grows after

    a budget violation without changing the violated step's semantics.

    ``graph_capacity_ratio=0.02`` gives C = 2; ``changes_per_row=3`` gives
    D = 12 > 2, so the first batched call exceeds the budget: the step
    must warn exactly once (the one-time budget warning), discard the
    replay, and return the eager result (== the fresh-eager reference,
    allclose 1e-4 plus per-row argmax -- FR2 "this step's semantics are
    unchanged"). The hook-A grow observation then re-captures at
    ``ceil(1.25 * 12) = 15`` (exact on the 2^6 fixture), visible through
    the public probe ``mff.graph_adaptive_policy``: its ``events`` contain
    exactly one ``trigger="grow"`` recapture whose target is 15 and
    ``>=`` the measured ``max(last_layer_divergences)`` (read back after
    the second step's replay). The second step replays within budget and
    matches the reference with zero new warnings. After the swap, an
    all-divergent candidate set exceeds the budget again and must emit
    zero NEW budget warnings (``_graph_batched_budget_warned`` is not
    reset by the runner swap) while still returning the correct eager
    result for that load.

    Red reason: ``ManualFoldedForward`` does not accept the
    ``graph_adaptive_capacity`` constructor parameter yet (TypeError).
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, changes_per_row=3, seed=4213)
    full = _full_divergent(parent)

    _mff_eager_low, ref_low = _batched_eager_reference(
        model, device, parent, stacked, _ids("1603-e1"), "1603-ep1"
    )
    _mff_eager_full, ref_full = _batched_eager_reference(
        model, device, parent, full, _ids("1603-e2"), "1603-ep2"
    )

    mff = _make_mff(
        model,
        device,
        use_cuda_graph=True,
        graph_capacity_ratio=0.02,
        graph_adaptive_capacity=True,
    )
    _serial(mff, parent, branch_id="1603-p")

    with warnings.catch_warnings(record=True) as record_first:
        warnings.simplefilter("always")
        out1 = _batched(mff, stacked, _ids("1603-a"), "1603-p")
    messages = _user_warning_messages(record_first)
    assert len(messages) == 1, "the first budget violation must warn exactly once"
    assert "budget" in messages[0].lower()
    assert torch.allclose(out1, ref_low, atol=1e-4, rtol=1e-4)
    assert torch.equal(out1.argmax(dim=-1), ref_low.argmax(dim=-1))

    runner = mff.graph_runner_batched
    assert runner is not None
    assert runner.capacity == 15

    policy = mff.graph_adaptive_policy
    assert policy is not None
    grow_events = [
        e
        for e in policy.events
        if e.get("kind") == "recapture" and e.get("trigger") == "grow"
    ]
    assert len(grow_events) == 1
    assert grow_events[0]["target"] == 15

    with warnings.catch_warnings(record=True) as record_second:
        warnings.simplefilter("always")
        out2 = _batched(mff, stacked, _ids("1603-b"), "1603-p")
    assert len(_user_warning_messages(record_second)) == 0
    assert runner.budget_exceeded is False
    assert torch.allclose(out2, ref_low, atol=1e-4, rtol=1e-4)
    assert torch.equal(out2.argmax(dim=-1), ref_low.argmax(dim=-1))

    divergences = runner.last_layer_divergences
    assert divergences is not None
    assert grow_events[0]["target"] >= max(divergences)

    with warnings.catch_warnings(record=True) as record_full:
        warnings.simplefilter("always")
        out3 = _batched(mff, full, _ids("1603-c"), "1603-p")
    assert len(_user_warning_messages(record_full)) == 0, "swap must not reset budget flag"
    assert torch.allclose(out3, ref_full, atol=1e-4, rtol=1e-4)
    assert torch.equal(out3.argmax(dim=-1), ref_full.argmax(dim=-1))


# ---------------------------------------------------------------------------
# UT-1604: jitter cap (max_recaptures) and retirement
# ---------------------------------------------------------------------------


@_cuda
def test_ut1604_recapture_cap_and_retirement() -> None:
    """UT-1604 (design §6 row 1604, srs §3.4 FR4): alternating loads stop at

    ``max_recaptures`` and the policy retires with exactly one warning.

    ``AdaptiveCapacityConfig(shrink_patience=2, min_recapture_interval=1,
    max_recaptures=2)`` (all three knobs deviate from the defaults 3/2/8 to
    compress the schedule -- declared here per the design §6 note). The
    load alternates low (D=8) and all-divergent (D=64): step 1 captures
    and observes (no-op); step 2 shrink-re-captures (C: 32 -> below 32);
    step 3 exceeds the shrunk budget (one budget warning), eagerly
    recomputes, and grow-re-captures (capacity increases again); step 4's
    observe hits the cap and returns ``"retired"`` -- exactly one
    retirement warning whose message is not the budget warning; steps 5-6
    stay silent with the runner object and capacity frozen. Faces: the
    recapture events number at most ``max_recaptures``; ``policy._retired``
    is latched; a ``"retired"`` event is recorded; every step's output
    matches the eager reference for its load (allclose 1e-4 plus per-row
    argmax).

    Red reason: ``actfold.core.adaptive_capacity`` does not exist yet, so
    ``_adaptive_module()`` fails this test with the ImportError.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, changes_per_row=2, seed=4214)
    full = _full_divergent(parent)

    _mff_eager_low, ref_low = _batched_eager_reference(
        model, device, parent, stacked, _ids("1604-e1"), "1604-ep1"
    )
    _mff_eager_full, ref_full = _batched_eager_reference(
        model, device, parent, full, _ids("1604-e2"), "1604-ep2"
    )

    mod = _adaptive_module()
    config_cls = mod.AdaptiveCapacityConfig
    mff = _make_mff(
        model,
        device,
        use_cuda_graph=True,
        graph_capacity_ratio=0.5,
        graph_adaptive_capacity=True,
        graph_adaptive_config=config_cls(
            shrink_patience=2,
            min_recapture_interval=1,
            max_recaptures=2,
        ),
    )
    _serial(mff, parent, branch_id="1604-p")

    step_warnings: list[list[str]] = []

    def _step(load: torch.Tensor, tag: str, ref: torch.Tensor) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = _batched(mff, load, _ids(tag), "1604-p")
        step_warnings.append(_user_warning_messages(record))
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
        assert torch.equal(out.argmax(dim=-1), ref.argmax(dim=-1))

    _step(stacked, "1604-a", ref_low)
    runner1 = mff.graph_runner_batched
    assert runner1 is not None
    assert runner1.capacity == 32

    _step(stacked, "1604-b", ref_low)
    runner2 = mff.graph_runner_batched
    assert runner2 is not None
    assert runner2 is not runner1
    assert runner2.capacity < 32

    _step(full, "1604-c", ref_full)
    runner3 = mff.graph_runner_batched
    assert runner3 is not None
    assert runner3 is not runner2
    assert runner3.capacity > runner2.capacity

    _step(stacked, "1604-d", ref_low)
    assert mff.graph_runner_batched is runner3

    _step(stacked, "1604-e", ref_low)
    _step(full, "1604-f", ref_full)
    assert mff.graph_runner_batched is runner3
    assert mff.graph_runner_batched.capacity == runner3.capacity

    assert [len(msgs) for msgs in step_warnings] == [0, 0, 1, 1, 0, 0]
    assert "budget" in step_warnings[2][0].lower()
    assert "budget" not in step_warnings[3][0].lower()

    policy = mff.graph_adaptive_policy
    assert policy is not None
    assert policy._retired is True
    recapture_events = [e for e in policy.events if e.get("kind") == "recapture"]
    assert len(recapture_events) <= 2
    assert any(e.get("kind") == "retired" for e in policy.events)


# ---------------------------------------------------------------------------
# UT-1605: re-capture failure safety net
# ---------------------------------------------------------------------------


@_cuda
def test_ut1605_recapture_failure_safety_net(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1605 (design §6 row 1605, srs §3.4 FR4): a failed re-capture never

    loses the old runner and retires the adaptive path only.

    ``AdaptiveCapacityConfig(shrink_patience=2)`` deviates from the default
    3 (declared here). The first low-load call captures normally (C=32);
    afterwards -- UT-1507 patch timing, placed only after the first
    capture has succeeded -- ``FoldedGraphRunner.capture`` is monkeypatched
    to raise once and delegate afterwards (one-shot; the failure restores
    delegation immediately). The second low-load call reaches the shrink
    re-capture whose ``capture`` fails: exactly one UserWarning naming the
    adaptive failure, ``graph_runner_batched`` still ``is`` the old
    instance, ``_graph_batched_capture_failed`` stays False (the batched
    path must not be disabled -- req Major-3), the policy is retired
    (``_retired``), no recapture event is recorded, and the step's result
    is the old runner's published replay (== the eager reference). With a
    counting ``replay`` spy wrapped around the old runner, two further
    steps replay through the same runner (spy count 1 then 2, strictly
    increasing), stay warning-free, never retry the capture, and match
    the reference.

    Red reason: ``actfold.core.adaptive_capacity`` does not exist yet, so
    ``_adaptive_module()`` fails this test with the ImportError.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, changes_per_row=2, seed=4215)

    _mff_eager, ref = _batched_eager_reference(
        model, device, parent, stacked, _ids("1605-e"), "1605-ep"
    )

    mod = _adaptive_module()
    config_cls = mod.AdaptiveCapacityConfig
    mff = _make_mff(
        model,
        device,
        use_cuda_graph=True,
        graph_capacity_ratio=0.5,
        graph_adaptive_capacity=True,
        graph_adaptive_config=config_cls(shrink_patience=2),
    )
    _serial(mff, parent, branch_id="1605-p")

    with warnings.catch_warnings(record=True) as record_first:
        warnings.simplefilter("always")
        _batched(mff, stacked, _ids("1605-a"), "1605-p")
    assert len(_user_warning_messages(record_first)) == 0
    runner_old = mff.graph_runner_batched
    assert runner_old is not None
    assert runner_old.capacity == 32

    capture_calls = {"count": 0}
    original_capture = FoldedGraphRunner.capture

    def _fail_once_capture(self: Any, *args: Any, **kwargs: Any) -> None:
        capture_calls["count"] += 1
        if capture_calls["count"] == 1:
            raise RuntimeError("injected adaptive re-capture failure")
        return original_capture(self, *args, **kwargs)

    monkeypatch.setattr(FoldedGraphRunner, "capture", _fail_once_capture)

    with warnings.catch_warnings(record=True) as record_fail:
        warnings.simplefilter("always")
        out2 = _batched(mff, stacked, _ids("1605-b"), "1605-p")
    messages = _user_warning_messages(record_fail)
    assert len(messages) == 1, "the failed re-capture must warn exactly once"
    assert "adaptive" in messages[0].lower()
    assert torch.allclose(out2, ref, atol=1e-4, rtol=1e-4)
    assert torch.equal(out2.argmax(dim=-1), ref.argmax(dim=-1))

    assert mff.graph_runner_batched is runner_old
    assert mff._graph_batched_capture_failed is False
    policy = mff.graph_adaptive_policy
    assert policy is not None
    assert policy._retired is True
    assert capture_calls["count"] == 1
    assert all(e.get("kind") != "recapture" for e in policy.events)

    replay_calls = {"count": 0}
    original_replay = runner_old.replay

    def _counting_replay(*args: Any, **kwargs: Any) -> Any:
        replay_calls["count"] += 1
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(runner_old, "replay", _counting_replay)

    with warnings.catch_warnings(record=True) as record_c:
        warnings.simplefilter("always")
        out3 = _batched(mff, stacked, _ids("1605-c"), "1605-p")
    assert len(_user_warning_messages(record_c)) == 0
    assert replay_calls["count"] == 1

    with warnings.catch_warnings(record=True) as record_d:
        warnings.simplefilter("always")
        out4 = _batched(mff, stacked, _ids("1605-d"), "1605-p")
    assert len(_user_warning_messages(record_d)) == 0
    assert replay_calls["count"] == 2
    assert capture_calls["count"] == 1, "a retired policy must never retry"

    for out in (out3, out4):
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
        assert torch.equal(out.argmax(dim=-1), ref.argmax(dim=-1))
    assert mff.graph_runner_batched is runner_old


# ---------------------------------------------------------------------------
# UT-1606: default-off zero behavior
# ---------------------------------------------------------------------------


@_cuda
def test_ut1606_default_off_zero_behavior() -> None:
    """UT-1606 (design §6 row 1606, srs §3.4 FR4): adaptive off = AR016 bits.

    Default construction (``graph_adaptive_capacity`` left False): five
    same-shape batched steps all replay through the same captured runner
    object with an unchanged capacity, emit zero UserWarnings, and match
    the fresh-eager reference on every step (allclose 1e-4 plus per-row
    argmax). The public probe ``mff.graph_adaptive_policy`` must read
    ``None`` (no policy state is ever built when the feature is off).

    Red reason: the ``graph_adaptive_policy`` public probe does not exist
    yet, so the probe read raises AttributeError (the runner/capacity/
    warning assertions above it already pass today and must stay green
    after the Green phase).
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4216)

    _mff_eager, ref = _batched_eager_reference(
        model, device, parent, stacked, _ids("1606-e"), "1606-ep"
    )

    mff = _make_mff(model, device, use_cuda_graph=True)
    _serial(mff, parent, branch_id="1606-p")

    outs: list[torch.Tensor] = []
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        outs.append(_batched(mff, stacked, _ids("1606-a"), "1606-p"))
        runner = mff.graph_runner_batched
        assert runner is not None
        initial_capacity = runner.capacity
        for tag in ("1606-b", "1606-c", "1606-d", "1606-e2"):
            outs.append(_batched(mff, stacked, _ids(tag), "1606-p"))
            assert mff.graph_runner_batched is runner
            assert mff.graph_runner_batched.capacity == initial_capacity
    assert len(_user_warning_messages(record)) == 0

    for out in outs:
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
        assert torch.equal(out.argmax(dim=-1), ref.argmax(dim=-1))

    assert mff.graph_adaptive_policy is None


# ---------------------------------------------------------------------------
# UT-1607: observation seam (last_layer_divergences)
# ---------------------------------------------------------------------------


@_cuda
def test_ut1607_last_layer_divergences_seam() -> None:
    """UT-1607 (design §6 row 1607, srs §3.4 FR1/FR4): the divergence staging.

    A directly constructed ``FoldedGraphRunner`` (test_cuda_graph.py §4.3.4
    precedent, bypassing the ManualFoldedForward dispatch) captures the
    [N, T] workload (num_tokens = 64, power of two) against a cached
    parent: before any replay, ``last_layer_divergences`` must be ``None``
    (only ``validate_budgets`` -- called inside ``replay`` -- stages it).
    After a low-load replay the staged list has one entry per layer (L),
    each equals ``num_tokens - count_buf.tolist()[l]`` (test-side readback
    is unrestricted by the single-sync contract), the max stays within
    ``num_tokens``, and the values are self-consistent non-negative
    counts. Replaying a different (all-divergent) load changes the staged
    values (they are refreshed per replay, never stale).

    Red reason: ``FoldedGraphRunner.last_layer_divergences`` does not exist
    yet, so the first probe read raises AttributeError.
    """
    device = "cuda"
    model, profile = _build_model(device)
    parent, stacked = _make_candidates(device, changes_per_row=2, seed=4217)
    full = _full_divergent(parent)

    mff = _make_mff(model, device)
    _serial(mff, parent, branch_id="1607-p")
    runner = _make_runner(profile, mff, capacity_ratio=0.5)
    with torch.no_grad():
        runner.capture(stacked, branch_id="1607-c", parent_branch_id="1607-p")

    assert runner.last_layer_divergences is None

    with torch.no_grad():
        logits = runner.replay(stacked, parent_branch_id="1607-p", branch_id="1607-c")
    assert logits is not None

    divergences = runner.last_layer_divergences
    assert divergences is not None
    assert len(divergences) == _NUM_LAYERS
    counts = runner.count_buf.tolist()
    for layer_idx in range(_NUM_LAYERS):
        expected = _NUM_TOKENS - counts[layer_idx]
        assert divergences[layer_idx] == expected
    assert 0 <= max(divergences) <= _NUM_TOKENS
    low_divergences = list(divergences)

    with torch.no_grad():
        runner.replay(full, parent_branch_id="1607-p", branch_id="1607-c2")
    full_divergences = runner.last_layer_divergences
    assert full_divergences is not None
    assert len(full_divergences) == _NUM_LAYERS
    assert full_divergences != low_divergences
    assert 0 <= max(full_divergences) <= _NUM_TOKENS


# ---------------------------------------------------------------------------
# UT-1608: serial zero interference
# ---------------------------------------------------------------------------


@_cuda
def test_ut1608_serial_zero_interference() -> None:
    """UT-1608 (design §6 row 1608, srs §3.4 FR4): serial path untouched by

    batched re-captures.

    ``AdaptiveCapacityConfig(shrink_patience=2)`` deviates from the default
    3 (declared here). The batched side re-captures twice (shrink on the
    second low step, growth on an all-divergent step) -- two distinct
    runner swaps. One extra low step sits between the shrink and the
    all-divergent step: the shrink's ``note_recaptured`` resets the
    throttle counter to zero, and ``min_recapture_interval=2`` would
    otherwise throttle the growth re-capture on the very next step
    (design SS2 increment-then-check semantics). Afterwards serial
    [1, T] calls on the SAME instance capture into ``graph_runner`` and
    replay normally: the serial runner is a different object from the
    batched one, keeps its identity and capacity across a second serial
    call (replay, not re-capture), the batched runner and its capacity
    are untouched by the serial captures, and both serial outputs match
    the eager reference rows and the batched replay rows directly
    (UT-1503 transitivity structure, allclose 1e-4 plus per-row argmax).

    Red reason: ``actfold.core.adaptive_capacity`` does not exist yet, so
    ``_adaptive_module()`` fails this test with the ImportError.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, changes_per_row=2, seed=4218)
    full = _full_divergent(parent)

    _mff_eager_low, ref_low = _batched_eager_reference(
        model, device, parent, stacked, _ids("1608-e1"), "1608-ep1"
    )
    _mff_eager_full, ref_full = _batched_eager_reference(
        model, device, parent, full, _ids("1608-e2"), "1608-ep2"
    )

    mod = _adaptive_module()
    config_cls = mod.AdaptiveCapacityConfig
    mff = _make_mff(
        model,
        device,
        use_cuda_graph=True,
        graph_capacity_ratio=0.5,
        graph_adaptive_capacity=True,
        graph_adaptive_config=config_cls(shrink_patience=2),
    )
    _serial(mff, parent, branch_id="1608-p")

    out_b1 = _batched(mff, stacked, _ids("1608-a"), "1608-p")
    runner1 = mff.graph_runner_batched
    assert runner1 is not None
    assert runner1.capacity == 32

    out_b2 = _batched(mff, stacked, _ids("1608-b"), "1608-p")
    runner2 = mff.graph_runner_batched
    assert runner2 is not None
    assert runner2 is not runner1
    assert runner2.capacity < 32

    # Throttle counter warm-up (design SS2): the shrink re-capture reset
    # ``steps_since_recapture`` to zero, so the growth verdict needs one
    # in-budget step before the all-divergent step to clear
    # ``min_recapture_interval=2``.
    out_b2b = _batched(mff, stacked, _ids("1608-b2"), "1608-p")
    assert torch.allclose(out_b2b, ref_low, atol=1e-4, rtol=1e-4)

    out_b3 = _batched(mff, full, _ids("1608-c"), "1608-p")
    runner3 = mff.graph_runner_batched
    assert runner3 is not None
    assert runner3 is not runner2
    assert runner3.capacity > runner2.capacity

    assert torch.allclose(out_b1, ref_low, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_b2, ref_low, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_b3, ref_full, atol=1e-4, rtol=1e-4)

    out_s1 = _serial(
        mff, stacked[0:1], branch_id="1608-s0", parent_branch_id="1608-p"
    )
    runner_s = mff.graph_runner
    assert runner_s is not None
    assert runner_s is not mff.graph_runner_batched
    serial_capacity = runner_s.capacity

    out_s2 = _serial(
        mff, stacked[1:2], branch_id="1608-s1", parent_branch_id="1608-p"
    )
    assert mff.graph_runner is runner_s
    assert mff.graph_runner.capacity == serial_capacity

    out_b4 = _batched(mff, stacked, _ids("1608-d"), "1608-p")
    assert mff.graph_runner_batched is runner3
    assert mff.graph_runner_batched.capacity == runner3.capacity

    assert torch.allclose(out_s1, ref_low[0:1], atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_s2, ref_low[1:2], atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_s1, out_b4[0:1], atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_s2, out_b4[1:2], atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# UT-1609: mutation spot-check -- RESERVED (design §6 row 1609, srs FR4).
#
# Implemented in the T002 mutation phase after the Green phase lands: each
# spot check applies one production mutation (delete the hook calls; delete
# the growth branch; delete the retirement check; swap to old-first
# ordering), re-runs the pinned UT, and restores the source byte-exactly
# (SHA256). Nothing is implemented here in the Red phase.
# ---------------------------------------------------------------------------
