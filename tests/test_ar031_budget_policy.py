"""AR031 budget policy unit tests (UT-3102..3113, UT-3161..3163)."""

from __future__ import annotations

import pytest
import torch

from actfold.core.layer_budget import LayerRecomputeBudgetPolicy

RATIO = 0.5
NUM_LAYERS = 4


def _policy(
    ratio: float = RATIO,
    num_layers: int = NUM_LAYERS,
    warmup: int = 5,
    window: int = 10,
    adaptive: bool = True,
) -> LayerRecomputeBudgetPolicy:
    return LayerRecomputeBudgetPolicy(
        budget_ratio=ratio,
        num_layers=num_layers,
        warmup_steps=warmup,
        window_steps=window,
        adaptive=adaptive,
    )


def _budget_table(policy: LayerRecomputeBudgetPolicy, tokens: list[int]) -> list[int]:
    return policy.budget_table(tokens)


def _expected_total(ratio: float, tokens: list[int]) -> int:
    b_total = round(ratio * sum(tokens))
    active = sum(1 for token in tokens if token > 0)
    return max(b_total, active)


def _observe(policy: LayerRecomputeBudgetPolicy, ratios: list[float], num_tokens: int) -> None:
    for layer, stable_ratio in enumerate(ratios):
        policy.observe_layer(
            layer,
            torch.tensor(stable_ratio * num_tokens, dtype=torch.float32),
            num_tokens,
        )


def test_ut3102_warmup_uniform() -> None:
    """UT-3102: warmup-window allocation is uniform (sum invariant, <=1 spread)."""
    policy = _policy(ratio=0.5, num_layers=4, warmup=5, window=10)
    tokens = [7, 7, 7, 7]
    tables: list[list[int]] = []
    for _ in range(4):  # steps 1..4, all before the warmup_steps boundary
        policy.begin_step()
        tables.append(_budget_table(policy, tokens))
    for table in tables:
        assert sum(table) == _expected_total(0.5, tokens)
        assert max(table) - min(table) <= 1  # equal tokens -> near-uniform per layer
        assert all(3 <= budget <= 4 for budget in table)
    assert tables[0] == tables[1] == tables[2] == tables[3]  # every step, same table
    assert policy.budget_for(0, 7) == tables[0][0]  # O(1) wrapper reads the cached table


def test_ut3103_reestimate_weights() -> None:
    """UT-3103: the window-boundary re-estimate sets w proportional to (1 - s + eps)."""
    policy = _policy(ratio=0.5, num_layers=2, warmup=2, window=10)
    policy.begin_step()  # step 1: observations accumulate inside the window
    _observe(policy, [0.9, 0.5], num_tokens=10)
    policy.begin_step()  # step 2 == warmup_steps: re-estimate consumes the window
    weights = policy.weights
    assert weights is not None
    assert len(weights) == 2
    assert weights[1] > weights[0]  # the less stable layer gets the larger weight
    assert weights[0] == pytest.approx(0.100001 / 0.600002, rel=1e-3)
    assert weights[1] == pytest.approx(0.500001 / 0.600002, rel=1e-3)
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)


def test_ut3104_budget_sum_conservation() -> None:
    """UT-3104: sum_l budget_for(l, T_l) == max(B_total, L') on every allocation path."""
    ratio = 0.4
    num_layers = 6
    # Fixed domain-1 vector: B_total < L' -> every active layer gets exactly 1.
    policy = _policy(ratio=ratio, num_layers=num_layers, warmup=3, window=10)
    policy.begin_step()
    assert _budget_table(policy, [1, 1, 1, 1, 1, 1]) == [1, 1, 1, 1, 1, 1]
    # Random seeds cover floor+largest-remainder, clamp surplus redistribution and
    # floor-lift paths in both the B_total < L' and B_total >= L' domains.
    for seed in range(8):
        gen = torch.Generator().manual_seed(seed)
        for low, high in ((0, 4), (10, 61)):
            policy = _policy(ratio=ratio, num_layers=num_layers, warmup=3, window=10)
            policy.begin_step()
            tokens = torch.randint(low, high, (num_layers,), generator=gen).tolist()
            table = _budget_table(policy, tokens)
            assert sum(table) == _expected_total(ratio, tokens)


def test_ut3105_allocation_determinism() -> None:
    """UT-3105: identical inputs and state yield identical tables (clamp + floor-lift)."""

    def build() -> LayerRecomputeBudgetPolicy:
        policy = _policy(ratio=0.5, num_layers=3, warmup=2, window=10)
        policy.begin_step()
        _observe(policy, [0.99, 0.0, 0.0], num_tokens=100)
        policy.begin_step()  # skewed re-estimate exercises clamp / floor-lift paths
        return policy

    tokens = [10, 10, 10]
    first = _budget_table(build(), tokens)
    second = _budget_table(build(), tokens)
    assert first == second  # two identically-built policies agree bit-for-bit
    assert sum(first) == _expected_total(0.5, tokens)
    policy = build()
    assert _budget_table(policy, tokens) == first  # repeated calls are stable
    assert _budget_table(policy, tokens) == first


def test_ut3106_no_data_degradation() -> None:
    """UT-3106: missing profiler data degrades to uniform weights without raising."""
    partial = _policy(ratio=0.5, num_layers=3, warmup=2, window=10)
    partial.begin_step()
    partial.observe_layer(1, torch.tensor(2.0, dtype=torch.float32), 10)
    partial.begin_step()  # layers 0 and 2 carry no data for this window
    weights = partial.weights
    assert weights is not None
    assert weights[1] > weights[0]  # the observed (unstable) layer weighs more
    assert weights[0] == pytest.approx(weights[2], rel=1e-6)  # no-data layers uniform
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)

    empty = _policy(ratio=0.5, num_layers=3, warmup=2, window=10)
    empty.begin_step()
    empty.begin_step()  # no observations at all -> all-uniform, no exception
    weights = empty.weights
    assert weights is not None
    uniform = 1.0 / 3.0
    assert all(weight == pytest.approx(uniform, rel=1e-6) for weight in weights)
    table = _budget_table(empty, [4, 4, 4])
    assert table == [2, 2, 2]
    assert sum(table) == _expected_total(0.5, [4, 4, 4])


def test_ut3107_zero_tokens_get_zero_budget() -> None:
    """UT-3107: tokens_l == 0 forces B_l == 0."""
    policy = _policy(ratio=0.5, num_layers=3, warmup=2, window=10)
    policy.begin_step()
    assert _budget_table(policy, [0, 0, 0]) == [0, 0, 0]
    policy = _policy(ratio=0.5, num_layers=3, warmup=2, window=10)
    policy.begin_step()
    table = _budget_table(policy, [0, 10, 10])
    assert table[0] == 0
    assert sum(table) == _expected_total(0.5, [0, 10, 10])


def test_ut3108_constructor_validation() -> None:
    """UT-3108: construction-time domain checks raise ValueError per parameter."""
    for ratio in (0.0, -0.5, 1.5):  # ratio must lie in (0, 1]
        with pytest.raises(ValueError):
            _policy(ratio=ratio)
    for num_layers in (0, -1):  # at least one layer
        with pytest.raises(ValueError):
            _policy(num_layers=num_layers)
    for warmup in (0, -2):  # W_warm >= 1
        with pytest.raises(ValueError):
            _policy(warmup=warmup)
    for window in (0, -3):  # W_est >= 1
        with pytest.raises(ValueError):
            _policy(window=window)
    _policy(ratio=1.0, num_layers=1, warmup=1, window=1)  # boundary values are legal
    _policy()  # defaults are legal


def test_ut3109_in_range_queries_are_valid_ints() -> None:
    """UT-3109: policy-side depth consistency (the MFF num_layers wiring is T005's duty).

    Every layer index inside [0, num_layers) answers budget_for with an int in
    [0, tokens_l]; out-of-range rejection is pinned by UT-3161.
    """
    policy = _policy(ratio=0.5, num_layers=4, warmup=5, window=10)
    policy.begin_step()
    tokens = [8, 8, 8, 8]
    table = _budget_table(policy, tokens)
    for budget in table:
        assert isinstance(budget, int)
        assert 0 <= budget <= 8
    assert sum(table) == _expected_total(0.5, tokens)


def test_ut3110_cap_upper_bound_folds_most_similar() -> None:
    """UT-3110: D > B binding caps non-exempt recompute at B_l, folding top scores."""
    policy = _policy()
    stable_mask = torch.zeros(12, dtype=torch.bool)
    stable_mask[:4] = True  # 4 stable (exempt) + 8 divergent, no score ties
    scores = torch.arange(12, dtype=torch.float32)
    out = policy.apply_layer_budget(stable_mask, scores, 5)
    expected = torch.zeros(12, dtype=torch.bool)
    expected[[0, 1, 2, 3, 9, 10, 11]] = True
    assert torch.equal(out, expected)
    non_exempt = ~stable_mask
    assert int((non_exempt & ~out).sum()) == 5  # non-exempt recompute == B_l exactly
    folded = non_exempt & out
    assert set(folded.nonzero().flatten().tolist()) == {9, 10, 11}  # max-score rows


def test_ut3111_cap_non_binding_zero_flips() -> None:
    """UT-3111: D <= B is non-binding -> zero forced flips, output equals input."""
    policy = _policy()
    stable_mask = torch.zeros(8, dtype=torch.bool)
    stable_mask[:5] = True  # 5 stable + 3 divergent <= budget 5
    scores = torch.arange(8, dtype=torch.float32)
    out = policy.apply_layer_budget(stable_mask, scores, 5)
    assert torch.equal(out, stable_mask)  # no row flipped to stable for quota's sake
    assert int((~out).sum()) == 3  # recompute count == D_l


def test_ut3111_cap_accepts_tensor_budget() -> None:
    """UT-3111: the budget argument also accepts a 0-dim tensor host value."""
    policy = _policy()
    stable_mask = torch.zeros(8, dtype=torch.bool)
    stable_mask[:5] = True
    scores = torch.arange(8, dtype=torch.float32)
    out = policy.apply_layer_budget(stable_mask, scores, torch.tensor(5))
    assert torch.equal(out, stable_mask)


def test_ut3112_window_freeze_and_atomic_swap() -> None:
    """UT-3112: weights freeze inside a window and swap atomically at boundaries."""
    policy = _policy(ratio=0.5, num_layers=2, warmup=3, window=4)
    policy.begin_step()  # step 1: window-1 observations
    _observe(policy, [0.9, 0.5], num_tokens=10)
    policy.begin_step()  # step 2
    _observe(policy, [0.8, 0.6], num_tokens=10)
    assert policy.window_events == []  # no re-estimate during warmup
    policy.begin_step()  # step 3 == warmup_steps: first re-estimate
    weights_at_boundary = policy.weights
    assert weights_at_boundary is not None
    assert len(policy.window_events) == 1
    assert isinstance(policy.window_events[0], dict)
    for _ in range(3):  # steps 4..6: the window is frozen
        policy.begin_step()
        _observe(policy, [0.2, 0.2], num_tokens=10)
        assert policy.weights == weights_at_boundary
        assert len(policy.window_events) == 1
    policy.begin_step()  # step 7: (7 - 3) % 4 == 0 -> second re-estimate
    weights_after = policy.weights
    assert weights_after is not None
    assert weights_after != weights_at_boundary  # atomic replacement at the boundary
    assert len(policy.window_events) == 2
    assert isinstance(policy.window_events[1], dict)


def test_ut3113_guard_priority_over_cap() -> None:
    """UT-3113: cache-availability guard rows win over the cap (paged x eager).

    The production guard is ``_apply_cache_availability`` in folded_transformer.py
    (it ANDs the target-layer availability into the mask AFTER the budget cap);
    it needs a full FoldedTransformerLayer + cache, so per the AR031 test plan the
    guard flip-back is simulated directly: rows the guard marks unavailable stay
    divergent, are exempt from the B_l accounting, and match the no-budget path.
    """
    policy = _policy()
    stable_mask = torch.zeros(10, dtype=torch.bool)
    scores = torch.arange(10, dtype=torch.float32)
    budget = 4
    guard = torch.zeros(10, dtype=torch.bool)
    guard[8] = True  # rows whose target-layer pages are unavailable
    guard[9] = True

    capped = policy.apply_layer_budget(stable_mask, scores, budget)
    assert int((~capped).sum()) == budget  # binding cap keeps B_l recomputes
    final = capped & ~guard  # the guard re-diverges its rows after the cap
    no_budget = policy.apply_layer_budget(stable_mask, scores, 10)
    assert torch.equal(no_budget, stable_mask)  # non-binding budget is a no-op

    assert not bool(final[guard].any())  # guard rows stay divergent (cap never refills)
    assert torch.equal(final[guard], no_budget[guard])  # matches the guard-only path
    non_guard = ~guard
    assert torch.equal(final[non_guard], capped[non_guard])  # cap still binds elsewhere
    total_recompute = int((~final).sum())
    assert total_recompute <= budget + int(guard.sum())  # guard rows are exempt
    assert total_recompute == budget + int(guard.sum())


def test_ut3161_interface_boundaries() -> None:
    """UT-3161: layer-index bounds, shape validation, and the N == 0 early exit."""
    policy = _policy(num_layers=4)
    policy.begin_step()
    stable = torch.tensor(1.0, dtype=torch.float32)
    with pytest.raises(ValueError):
        policy.observe_layer(4, stable, 10)
    with pytest.raises(ValueError):
        policy.observe_layer(-1, stable, 10)
    with pytest.raises(ValueError):
        policy.budget_for(4, 10)
    with pytest.raises(ValueError):
        policy.budget_for(-1, 10)

    mask = torch.zeros(2, 4, dtype=torch.bool)
    scores = torch.zeros(2, 4, dtype=torch.float32)
    with pytest.raises(ValueError):
        policy.apply_layer_budget(mask, torch.zeros(2, 5, dtype=torch.float32), 3)
    with pytest.raises(ValueError):
        policy.apply_layer_budget(mask, scores, 3, forced=torch.zeros(2, 5, dtype=torch.bool))

    empty_mask = torch.zeros(0, 4, dtype=torch.bool)
    empty_scores = torch.zeros(0, 4, dtype=torch.float32)
    result = policy.apply_layer_budget(empty_mask, empty_scores, 3)
    assert result is empty_mask  # N == 0 early-exit returns the original tensor


def test_ut3162_nan_scores_are_exempt_and_always_recomputed() -> None:
    """UT-3162: NaN scores stay divergent and are exempt from the B_l accounting."""
    policy = _policy()
    stable_mask = torch.zeros(10, dtype=torch.bool)
    scores = torch.arange(10, dtype=torch.float32)
    scores[8] = float("nan")
    scores[9] = float("nan")
    out = policy.apply_layer_budget(stable_mask, scores, 4)
    assert not bool(out[8]) and not bool(out[9])  # NaN rows always recompute
    non_exempt = ~torch.isnan(scores)
    non_exempt_recompute = int((non_exempt & ~out).sum())
    total_recompute = int((~out).sum())
    nan_count = int(torch.isnan(scores).sum())
    assert non_exempt_recompute == 4  # binding + tie-free -> exactly B_l
    assert non_exempt_recompute <= 4
    assert total_recompute == non_exempt_recompute + nan_count
    assert total_recompute <= 4 + nan_count  # <= B_l + exempt count invariant
    expected = torch.zeros(10, dtype=torch.bool)
    expected[4:8] = True
    assert torch.equal(out, expected)


def test_ut3163_forced_rows_are_never_folded() -> None:
    """UT-3163: forced rows keep diverging under a binding budget.

    Only the most similar non-forced rows fold; forced rows never count against
    (nor consume) the B_l budget.
    """
    policy = _policy()
    stable_mask = torch.zeros(10, dtype=torch.bool)  # already AND ~forced
    scores = torch.arange(10, dtype=torch.float32)  # forced rows carry top scores
    forced = torch.zeros(10, dtype=torch.bool)
    forced[7:] = True
    out = policy.apply_layer_budget(stable_mask, scores, 3, forced=forced)
    assert not bool(out[forced].any())  # forced rows all stay divergent
    non_exempt = ~forced
    non_exempt_recompute = int((non_exempt & ~out).sum())
    assert non_exempt_recompute == 3  # binding + tie-free -> exactly B_l
    assert non_exempt_recompute <= 3
    total_recompute = int((~out).sum())
    assert total_recompute == non_exempt_recompute + int(forced.sum())
    assert total_recompute <= 3 + int(forced.sum())
    folded = non_exempt & out
    assert set(folded.nonzero().flatten().tolist()) == {3, 4, 5, 6}
    expected = torch.zeros(10, dtype=torch.bool)
    expected[3:7] = True
    assert torch.equal(out, expected)
