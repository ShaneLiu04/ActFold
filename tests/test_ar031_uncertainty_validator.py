"""AR031 uncertainty-aware rollout validator unit tests (UT-3121..3133, UT-3164).

T006 Red face: ``actfold/speculative/uncertainty_validator.py`` does not exist
yet, so every test fails on the deferred import inside ``_validator`` (Red
reason: ModuleNotFoundError until the T007 Green implementation lands). Once
Green arrives these assertions bind the frozen contract from design 4.2.2 (A5)
+ erratum E2 + srs FR3: prior-descending validation order with id tie-break,
margin + margin_gap double-condition early stop, losers closed via
``close_candidate`` while the winner is kept and the rollout stays open, and
``validate_fn`` failures rolling the failing candidate back atomically.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from actfold.core.paged_fold_cache import PagedFoldCache, RolloutBudgetError, RolloutSet
from actfold.core.pyramid_plan import build_pyramid_plan

PAGE = 16
BATCH = 1
HIDDEN = 4
SEQ = 32
MARGIN = 0.9
MARGIN_GAP = 0.05


def _cache(total: int = 256) -> PagedFoldCache:
    plan = build_pyramid_plan(
        depth=2, total_tokens=total, block_size=PAGE, max_ratio=1.0, min_ratio=1.0
    )
    return PagedFoldCache(plan, PAGE, device="cpu")


def _acts(seq: int, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {"ffn_out": torch.randn(BATCH, seq, HIDDEN, generator=gen)}


def _base(cache: PagedFoldCache, seq: int = SEQ) -> None:
    cache.put("base", 0, _acts(seq, seed=1))


def _validator(
    cache: PagedFoldCache, margin: float = MARGIN, margin_gap: float = MARGIN_GAP
) -> Any:
    """Build the validator under test.

    The import is deliberately deferred: the module is the T007 Green
    deliverable, so at Red time this helper raises ModuleNotFoundError (the
    Red evidence) while all criterion assertions below stay fully written.
    """
    from actfold.speculative.uncertainty_validator import UncertaintyRolloutValidator

    return UncertaintyRolloutValidator(cache, margin=margin, margin_gap=margin_gap)


def _stats_copy(validator: Any) -> dict[str, Any]:
    stats = validator.stats
    return {
        key: (list(value) if isinstance(value, list) else value) for key, value in stats.items()
    }


def _assert_stats_unchanged(validator: Any, before: dict[str, Any]) -> None:
    stats = validator.stats
    for key, value in before.items():
        current = stats[key]
        if isinstance(value, list):
            assert list(current) == value
        else:
            assert current == value


def _candidate_snapshot(
    cache: PagedFoldCache, rollout: RolloutSet, candidate: str, seq: int = SEQ
) -> tuple[Any, ...]:
    """rollout_stats + page table + availability + cached tensors of one candidate."""
    layers = cache._table._branches[candidate]
    return (
        dict(cache.rollout_stats(rollout)[candidate]),
        {layer: dict(state.page_ids) for layer, state in layers.items()},
        cache.availability(candidate, 0, seq).clone(),
        cache.fetch(candidate, 0)["ffn_out"].clone(),
    )


def _assert_same_snapshot(actual: tuple[Any, ...], expected: tuple[Any, ...]) -> None:
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])


def test_ut3121_tiebreak_full_validation_id_order() -> None:
    """UT-3121: identical priors validate all candidates in ascending id order."""
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c2", "c0", "c1"])
    calls: list[str] = []

    def validate_fn(cid: str) -> float:
        calls.append(cid)
        return 0.6  # below margin: no early stop, identical posteriors

    validator = _validator(cache)
    priors = {"c0": 0.5, "c1": 0.5, "c2": 0.5}
    winner = validator.validate(rollout, ["c2", "c0", "c1"], priors, validate_fn)
    assert calls == ["c0", "c1", "c2"]  # id-ascending order, recorded by validate_fn
    assert winner == "c0"  # posterior tie -> smallest id (argmax first-occurrence)
    stats = validator.stats
    assert stats["validated_count"] == 3
    assert stats["skipped_count"] == 0
    assert stats["early_stops"] == 0
    cache.close_rollout(rollout)
    cache.assert_invariants()


def test_ut3122_early_stop_first_candidate_dominates() -> None:
    """UT-3122: margin + gap both met on the first candidate -> skip the rest.

    The controlled posterior map {c0: 0.95, c1: 0.90, c2: 0.85} makes c0 the
    full-validation winner as well, so the early-stop winner is bit-equal to
    the full-validation winner.
    """
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1", "c2"])
    calls: list[str] = []
    posterior = {"c0": 0.95, "c1": 0.90, "c2": 0.85}

    def validate_fn(cid: str) -> float:
        calls.append(cid)
        return posterior[cid]

    validator = _validator(cache, margin=MARGIN, margin_gap=MARGIN_GAP)
    priors = {"c0": 0.8, "c1": 0.7, "c2": 0.6}
    winner = validator.validate(rollout, ["c0", "c1", "c2"], priors, validate_fn)
    assert winner == "c0"
    assert calls == ["c0"]  # the remaining candidates are skipped wholesale
    stats = validator.stats
    assert stats["validated_count"] == 1
    assert stats["skipped_count"] == 2  # N - 1
    assert stats["early_stops"] == 1
    assert stats["winner_scores"] == pytest.approx([0.95])
    cache.close_rollout(rollout)
    cache.assert_invariants()


def test_ut3123_skipped_candidates_keep_fork_state() -> None:
    """UT-3123: skipped candidates are zero-write (stats/availability/pages intact)."""
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1", "c2"])
    baseline = {
        candidate: _candidate_snapshot(cache, rollout, candidate) for candidate in ("c1", "c2")
    }
    observed: dict[str, tuple[Any, ...]] = {}
    calls: list[str] = []

    def validate_fn(cid: str) -> float:
        calls.append(cid)
        if cid == "c0":
            # A real caller write for the candidate under validation ...
            cache.rollout_put(rollout, "c0", 0, _acts(SEQ, seed=42))
            # ... must leave the not-yet-validated candidates at fork baseline.
            for candidate in ("c1", "c2"):
                observed[candidate] = _candidate_snapshot(cache, rollout, candidate)
        return 0.95

    validator = _validator(cache, margin=MARGIN, margin_gap=MARGIN_GAP)
    priors = {"c0": 0.8, "c1": 0.7, "c2": 0.6}
    winner = validator.validate(rollout, ["c0", "c1", "c2"], priors, validate_fn)
    assert winner == "c0"
    assert calls == ["c0"]  # validate_fn (and thus rollout_put) never ran for c1/c2
    for candidate in ("c1", "c2"):
        _assert_same_snapshot(observed[candidate], baseline[candidate])
    stats = validator.stats
    assert stats["validated_count"] == 1
    assert stats["skipped_count"] == 2
    cache.close_rollout(rollout)
    cache.assert_invariants()


def test_ut3124_single_condition_does_not_stop() -> None:
    """UT-3124: margin without gap (or gap without margin) -> full validation."""
    cache = _cache()
    _base(cache)

    # (a) margin satisfied, gap not: 0.95 >= 0.9 but 0.95 - 0.6 = 0.35 < 0.5.
    rollout = cache.fork_rollouts("base", ["c0", "c1", "c2"])
    calls: list[str] = []
    posterior = {"c0": 0.95, "c1": 0.7, "c2": 0.6}

    def validate_fn(cid: str) -> float:
        calls.append(cid)
        return posterior[cid]

    validator = _validator(cache, margin=0.9, margin_gap=0.5)
    priors = {"c0": 0.85, "c1": 0.6, "c2": 0.5}
    winner = validator.validate(rollout, ["c0", "c1", "c2"], priors, validate_fn)
    assert calls == ["c0", "c1", "c2"]
    assert winner == "c0"
    stats = validator.stats
    assert stats["validated_count"] == 3
    assert stats["skipped_count"] == 0
    assert stats["early_stops"] == 0
    cache.close_rollout(rollout)

    # (b) gap satisfied, margin not: 0.9 - 0.6 = 0.3 >= 0.05 but 0.9 < 0.95.
    rollout = cache.fork_rollouts("base", ["c0", "c1", "c2"])
    calls = []
    posterior = {"c0": 0.9, "c1": 0.7, "c2": 0.6}

    def validate_fn_gap(cid: str) -> float:
        calls.append(cid)
        return posterior[cid]

    validator = _validator(cache, margin=0.95, margin_gap=0.05)
    winner = validator.validate(rollout, ["c0", "c1", "c2"], priors, validate_fn_gap)
    assert calls == ["c0", "c1", "c2"]
    assert winner == "c0"
    stats = validator.stats
    assert stats["validated_count"] == 3
    assert stats["skipped_count"] == 0
    assert stats["early_stops"] == 0
    cache.close_rollout(rollout)
    cache.assert_invariants()


def test_ut3125_stats_conservation_and_winner_score_fifo() -> None:
    """UT-3125: validated + skipped == N per call; winner_scores is a FIFO of 100."""
    cache = _cache()
    _base(cache)
    validator = _validator(cache)

    # Call 1: full validation of three tied-prior candidates.
    rollout = cache.fork_rollouts("base", ["a", "b", "c"])

    def validate_all(cid: str) -> float:
        return 0.6

    priors = {"a": 0.5, "b": 0.5, "c": 0.5}
    winner = validator.validate(rollout, ["a", "b", "c"], priors, validate_all)
    assert winner == "a"
    stats = validator.stats
    assert stats["validated_count"] + stats["skipped_count"] == 3
    assert stats["early_stops"] == 0
    cache.close_rollout(rollout)

    # Call 2: early stop on the dominant first candidate.
    rollout = cache.fork_rollouts("base", ["d", "e"])

    def validate_stop(cid: str) -> float:
        return 0.95 if cid == "d" else 0.5

    winner = validator.validate(rollout, ["d", "e"], {"d": 0.8, "e": 0.5}, validate_stop)
    assert winner == "d"
    stats = validator.stats
    assert stats["validated_count"] == 4
    assert stats["skipped_count"] == 1
    assert stats["validated_count"] + stats["skipped_count"] == 3 + 2  # cumulative
    assert stats["early_stops"] == 1
    cache.close_rollout(rollout)

    # 105 further single-candidate calls push winner_scores past the FIFO cap.
    expected_scores = [0.6, 0.95]
    for i in range(105):
        rollout = cache.fork_rollouts("base", ["s"])
        q = float(i) / 1000.0

        def validate_one(cid: str, _q: float = q) -> float:
            return _q

        assert validator.validate(rollout, ["s"], {"s": 0.5}, validate_one) == "s"
        expected_scores.append(q)
        stats = validator.stats
        assert stats["validated_count"] + stats["skipped_count"] == 5 + (i + 1)
        cache.close_rollout(rollout)

    stats = validator.stats
    # 3 (call 1) + 1 (call 2) + 105 single-candidate calls = 109 validated
    # (+ 1 skipped = the 110 conservation total; the Red-era fixture wrote
    # 110 here, contradicting its own in-loop 5 + (i + 1) assertion —
    # repaired in Green, judgment unchanged).
    assert stats["validated_count"] == 109
    assert stats["skipped_count"] == 1
    assert stats["early_stops"] == 1
    assert len(stats["winner_scores"]) == 100  # FIFO cap
    assert stats["winner_scores"] == pytest.approx(expected_scores[-100:])
    cache.assert_invariants()


def test_ut3126_argument_validation_zero_mutation() -> None:
    """UT-3126: N==0 / duplicate / unknown candidate / missing prior / foreign
    rollout all raise ValueError with zero state mutation."""
    cache = _cache()
    _base(cache)
    cache.put("other", 0, _acts(SEQ, seed=2))  # a branch that is NOT a candidate
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    validator = _validator(cache)
    calls: list[str] = []

    def validate_fn(cid: str) -> float:
        calls.append(cid)
        return 0.5

    before = _stats_copy(validator)
    epoch_before = cache.epoch
    with pytest.raises(ValueError):  # N == 0
        validator.validate(rollout, [], {}, validate_fn)
    with pytest.raises(ValueError):  # duplicate ids
        validator.validate(rollout, ["c0", "c0"], {"c0": 0.5}, validate_fn)
    with pytest.raises(ValueError):  # unknown candidate (no such branch)
        validator.validate(rollout, ["c9"], {"c9": 0.5}, validate_fn)
    with pytest.raises(ValueError):  # existing branch but not a rollout candidate
        validator.validate(rollout, ["other"], {"other": 0.5}, validate_fn)
    with pytest.raises(ValueError):  # prior missing for c1
        validator.validate(rollout, ["c0", "c1"], {"c0": 0.5}, validate_fn)

    # Ownership mismatch: a rollout forked on a different cache.
    cache2 = _cache()
    _base(cache2)
    foreign = cache2.fork_rollouts("base", ["x0"])
    with pytest.raises(ValueError):
        validator.validate(foreign, ["x0"], {"x0": 0.5}, validate_fn)

    assert calls == []  # validate_fn never invoked
    assert cache.epoch == epoch_before  # zero cache mutation
    _assert_stats_unchanged(validator, before)
    cache.close_rollout(rollout)
    cache2.close_rollout(foreign)
    cache.assert_invariants()
    cache2.assert_invariants()


def test_ut3127_margin_domain_validation() -> None:
    """UT-3127: margin / margin_gap outside [0, 1] fail at construction time."""
    cache = _cache()
    for margin in (-0.1, 1.5):
        with pytest.raises(ValueError):
            _validator(cache, margin=margin)
    for gap in (-0.05, 1.5):
        with pytest.raises(ValueError):
            _validator(cache, margin_gap=gap)
    _validator(cache, margin=0.0, margin_gap=0.0)  # closed boundaries are legal
    _validator(cache, margin=1.0, margin_gap=1.0)


def test_ut3128_budget_error_atomic_rollback() -> None:
    """UT-3128: RolloutBudgetError from validate_fn -> rollback_candidate +
    re-raise; the failing candidate returns to its fork baseline."""
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    baseline = {
        candidate: cache.fetch(candidate, 0)["ffn_out"].clone() for candidate in ("c0", "c1")
    }
    calls: list[str] = []

    def validate_fn(cid: str) -> float:
        calls.append(cid)
        if cid == "c0":
            # A real COW write precedes the hard-protection budget conflict.
            cache.rollout_put(rollout, "c0", 0, _acts(SEQ, seed=7))
            raise RolloutBudgetError("hard protection prevents budget eviction")
        return 0.5

    validator = _validator(cache)
    before = _stats_copy(validator)
    with pytest.raises(RolloutBudgetError):
        validator.validate(rollout, ["c0", "c1"], {"c0": 0.8, "c1": 0.5}, validate_fn)
    assert calls == ["c0"]
    # The failing candidate is restored to the fork baseline ...
    assert torch.equal(cache.fetch("c0", 0)["ffn_out"], baseline["c0"])
    # ... and every other candidate is untouched.
    assert torch.equal(cache.fetch("c1", 0)["ffn_out"], baseline["c1"])
    _assert_stats_unchanged(validator, before)  # the failed call counts nowhere
    cache.close_rollout(rollout)
    cache.assert_invariants()


def test_ut3129_generic_failure_rollback() -> None:
    """UT-3129: a generic validate_fn exception rolls the candidate back and
    re-raises (AR030 three-failure semantics)."""
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    c0_write = _acts(SEQ, seed=7)
    baseline_c1 = cache.fetch("c1", 0)["ffn_out"].clone()
    calls: list[str] = []

    def validate_fn(cid: str) -> float:
        calls.append(cid)
        if cid == "c0":
            cache.rollout_put(rollout, "c0", 0, c0_write)
            return 0.5
        cache.rollout_put(rollout, "c1", 0, _acts(SEQ, seed=8))
        raise RuntimeError("caller forward failed")

    validator = _validator(cache)
    before = _stats_copy(validator)
    with pytest.raises(RuntimeError):
        validator.validate(rollout, ["c0", "c1"], {"c0": 0.8, "c1": 0.5}, validate_fn)
    assert calls == ["c0", "c1"]
    # The failed candidate is rolled back to its fork baseline ...
    assert torch.equal(cache.fetch("c1", 0)["ffn_out"], baseline_c1)
    # ... while the already-validated candidate keeps its committed write.
    assert torch.equal(cache.fetch("c0", 0)["ffn_out"], c0_write["ffn_out"])
    _assert_stats_unchanged(validator, before)
    cache.close_rollout(rollout)
    cache.assert_invariants()


def test_ut3130_nan_posterior_never_wins() -> None:
    """UT-3130: a NaN posterior never becomes the winner; all-NaN -> order[0]."""
    # A NaN first candidate loses to any finite posterior.
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    posterior = {"c0": float("nan"), "c1": 0.7}

    def validate_fn(cid: str) -> float:
        return posterior[cid]

    validator = _validator(cache, margin=MARGIN, margin_gap=MARGIN_GAP)
    winner = validator.validate(rollout, ["c0", "c1"], {"c0": 0.9, "c1": 0.8}, validate_fn)
    assert winner == "c1"
    stats = validator.stats
    assert stats["validated_count"] == 2
    assert stats["early_stops"] == 0
    cache.close_rollout(rollout)

    # All-NaN posteriors keep the order[0] default winner.
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1"])

    def validate_nan(cid: str) -> float:
        return float("nan")

    validator = _validator(cache)
    winner = validator.validate(rollout, ["c0", "c1"], {"c0": 0.9, "c1": 0.8}, validate_nan)
    assert winner == "c0"  # order[0] survives the all-NaN case
    assert validator.stats["validated_count"] == 2
    cache.close_rollout(rollout)
    cache.assert_invariants()


def test_ut3131_losers_closed_winner_kept_base_frozen() -> None:
    """UT-3131: validated losers are close_candidate'd; the winner branch stays,
    the rollout stays open, the base stays frozen (erratum E2)."""
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1", "c2"])
    pages_at_fork = cache.page_stats()["physical_pages"]
    pages_after_write: dict[str, int] = {}
    posterior = {"c0": 0.5, "c1": 0.85, "c2": 0.7}  # all below margin: no early stop
    seeds = {"c0": 101, "c1": 102, "c2": 103}

    def validate_fn(cid: str) -> float:
        cache.rollout_put(rollout, cid, 0, _acts(SEQ, seed=seeds[cid]))
        pages_after_write[cid] = cache.page_stats()["physical_pages"]
        return posterior[cid]

    validator = _validator(cache, margin=MARGIN, margin_gap=MARGIN_GAP)
    priors = {"c0": 0.5, "c1": 0.5, "c2": 0.5}
    winner = validator.validate(rollout, ["c0", "c1", "c2"], priors, validate_fn)
    assert winner == "c1"
    stats = validator.stats
    assert stats["validated_count"] == 3
    assert stats["skipped_count"] == 0
    assert stats["early_stops"] == 0

    # Validated losers are gone: branch dropped, pages released.
    branches = cache._table.branches()
    assert "c0" not in branches
    assert "c2" not in branches
    assert pages_after_write["c2"] == pages_at_fork + 6  # 3 COW writes x 2 pages
    assert cache.page_stats()["physical_pages"] == pages_at_fork + 2  # winner only

    # The winner branch exists with its written pages; the rollout stays open.
    assert "c1" in branches
    assert cache.contains("c1", 0)
    assert rollout.closed is False
    assert set(cache.rollout_stats(rollout)) == {"c1"}

    # The base stays frozen while the rollout is open.
    with pytest.raises(ValueError):
        cache.put("base", 0, _acts(SEQ, seed=99))
    cache.close_rollout(rollout)
    cache.assert_invariants()


def test_ut3132_winner_pages_retained_and_forkable() -> None:
    """UT-3132: after validate the winner's write pages are in the cache and the
    caller can continue forking from the retained winner."""
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    winner_write = _acts(SEQ, seed=7)
    loser_write = _acts(SEQ, seed=8)

    def validate_fn(cid: str) -> float:
        cache.rollout_put(rollout, cid, 0, winner_write if cid == "c1" else loser_write)
        return 0.95 if cid == "c1" else 0.7

    validator = _validator(cache, margin=MARGIN, margin_gap=MARGIN_GAP)
    winner = validator.validate(rollout, ["c0", "c1"], {"c0": 0.8, "c1": 0.5}, validate_fn)
    assert winner == "c1"
    # The winner's written pages survive the validate call.
    assert cache.contains("c1", 0)
    assert torch.equal(cache.fetch("c1", 0)["ffn_out"], winner_write["ffn_out"])
    # Caller continuation: a fresh rollout can be forked from the winner.
    nested = cache.fork_rollouts("c1", ["g0"])
    assert nested.candidate_ids == ("g0",)
    cache.close_rollout(nested)
    cache.close_rollout(rollout)
    cache.assert_invariants()


def test_ut3133_non_numeric_prior_rejected() -> None:
    """UT-3133: prior_scores with a non-numeric value -> ValueError."""
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    validator = _validator(cache)
    calls: list[str] = []

    def validate_fn(cid: str) -> float:
        calls.append(cid)
        return 0.5

    before = _stats_copy(validator)
    with pytest.raises(ValueError):  # a string is not float-convertible
        validator.validate(rollout, ["c0", "c1"], {"c0": "high", "c1": 0.5}, validate_fn)
    with pytest.raises(ValueError):  # None is not float-convertible
        validator.validate(rollout, ["c0", "c1"], {"c0": 0.5, "c1": None}, validate_fn)
    assert calls == []
    _assert_stats_unchanged(validator, before)
    cache.close_rollout(rollout)
    cache.assert_invariants()


def test_ut3164_closed_rollout_and_inactive_candidate_rejected() -> None:
    """UT-3164: validate on a closed rollout / non-active candidate -> ValueError
    (public-probe ownership path, zero state mutation)."""
    # (a) an already-closed rollout fails the ownership probe.
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0"])
    cache.close_rollout(rollout)
    validator = _validator(cache)
    calls: list[str] = []

    def validate_fn(cid: str) -> float:
        calls.append(cid)
        return 0.5

    before = _stats_copy(validator)
    with pytest.raises(ValueError):
        validator.validate(rollout, ["c0"], {"c0": 0.5}, validate_fn)
    assert calls == []
    _assert_stats_unchanged(validator, before)
    cache.assert_invariants()

    # (b) a candidate removed via close_candidate is no longer active.
    cache2 = _cache()
    _base(cache2)
    rollout2 = cache2.fork_rollouts("base", ["c0", "c1"])
    cache2.close_candidate(rollout2, "c1")
    validator2 = _validator(cache2)
    before2 = _stats_copy(validator2)
    with pytest.raises(ValueError):  # the id set includes the inactive candidate
        validator2.validate(rollout2, ["c0", "c1"], {"c0": 0.5, "c1": 0.5}, validate_fn)
    with pytest.raises(ValueError):  # the inactive candidate alone is rejected too
        validator2.validate(rollout2, ["c1"], {"c1": 0.5}, validate_fn)
    assert calls == []
    _assert_stats_unchanged(validator2, before2)
    cache2.close_rollout(rollout2)
    cache2.assert_invariants()
