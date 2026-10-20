"""AR029 plan-unit tests (UT-2901..2910 + UT-2905b)."""

from __future__ import annotations

import math

import pytest

from actfold.core.pyramid_plan import PyramidPlan, build_pyramid_plan


def test_ut2901_plan_is_frozen() -> None:
    """UT-2901: PyramidPlan is immutable and exposes the documented fields."""
    plan = build_pyramid_plan(depth=4, total_tokens=1024, block_size=16)
    assert isinstance(plan, PyramidPlan)
    assert plan.depth == 4
    assert plan.block_size == 16
    assert len(plan.layer_budgets) == 4
    with pytest.raises(Exception):
        plan.depth = 8  # type: ignore[misc]


def test_ut2902_linear_budget_shape_and_sum() -> None:
    """UT-2902: linear budgets are block-aligned, monotone, and plan-wide bounded."""
    plan = build_pyramid_plan(
        depth=8, total_tokens=4096, block_size=16, max_ratio=1.0, min_ratio=0.25
    )
    budgets = plan.layer_budgets
    assert all(b % 16 == 0 for b in budgets)
    assert all(a >= b for a, b in zip(budgets, budgets[1:]))
    assert sum(budgets) <= 4096
    assert plan.total_budget_tokens == sum(budgets)


def test_ut2903_geometric_budget_shape_and_sum() -> None:
    """UT-2903: geometric budgets decay monotonically and respect the total."""
    plan = build_pyramid_plan(
        depth=6,
        total_tokens=2048,
        block_size=16,
        max_ratio=1.0,
        min_ratio=0.125,
        schedule="geometric",
    )
    budgets = plan.layer_budgets
    assert all(b % 16 == 0 and b >= 16 for b in budgets)
    assert all(a >= b for a, b in zip(budgets, budgets[1:]))
    assert sum(budgets) <= 2048
    ratios = [b / budgets[0] for b in budgets]
    assert all(r1 <= r0 + 1e-9 for r0, r1 in zip(ratios, ratios[1:]))


def test_ut2904_depth_one_floor() -> None:
    """UT-2904: depth=1 uses floor(total/block)*block."""
    plan = build_pyramid_plan(depth=1, total_tokens=100, block_size=16)
    assert plan.layer_budgets == (96,)
    assert plan.total_budget_tokens == 96


def test_ut2905_infeasible_total_raises() -> None:
    """UT-2905: a total smaller than depth*block_size is rejected."""
    with pytest.raises(ValueError):
        build_pyramid_plan(depth=8, total_tokens=127, block_size=16)


@pytest.mark.parametrize("depth", [0, -1])
def test_ut2905b_depth_invalid(depth: int) -> None:
    """UT-2905b: non-positive depth is rejected."""
    with pytest.raises(ValueError):
        build_pyramid_plan(depth=depth, total_tokens=1024, block_size=16)


@pytest.mark.parametrize("block", [0, -8])
def test_ut2905b_block_invalid(block: int) -> None:
    """UT-2905b: non-positive block size is rejected."""
    with pytest.raises(ValueError):
        build_pyramid_plan(depth=2, total_tokens=1024, block_size=block)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_ratio": 0.0},
        {"min_ratio": -0.1},
        {"max_ratio": 0.0},
        {"max_ratio": 1.5},
        {"max_ratio": 0.25, "min_ratio": 0.5},
    ],
)
def test_ut2906_ratio_invalid(kwargs: dict[str, float]) -> None:
    """UT-2906: ratio bounds are validated."""
    with pytest.raises(ValueError):
        build_pyramid_plan(depth=4, total_tokens=1024, block_size=16, **kwargs)


def test_ut2907_schedule_invalid() -> None:
    """UT-2907: unknown schedules are rejected."""
    with pytest.raises(ValueError):
        build_pyramid_plan(depth=4, total_tokens=1024, block_size=16, schedule="cosine")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recent_blocks": -1},
        {"anchor_blocks": -1},
        {"anchor_stride": 0},
    ],
)
def test_ut2908_retention_invalid(kwargs: dict[str, int]) -> None:
    """UT-2908: negative retention knobs / zero stride are rejected."""
    with pytest.raises(ValueError):
        build_pyramid_plan(depth=4, total_tokens=1024, block_size=16, **kwargs)


def test_ut2909_retained_blocks_clamped() -> None:
    """UT-2909: anchor and stride sets stay within [0, n_blocks)."""
    plan = build_pyramid_plan(
        depth=2,
        total_tokens=1024,
        block_size=16,
        anchor_blocks=10,
        anchor_stride=3,
    )
    n_blocks = math.ceil(64 / 16)
    retained = plan.retained_blocks(0, 64)
    assert retained == frozenset({0, 1, 2, 3})
    assert all(0 <= idx < n_blocks for idx in retained)
    assert plan.retained_blocks(0, 0) == frozenset()
    assert plan.retained_blocks(1, 64) == retained


def test_ut2910_determinism() -> None:
    """UT-2910: identical inputs yield identical plans."""
    a = build_pyramid_plan(depth=5, total_tokens=2000, block_size=16)
    b = build_pyramid_plan(depth=5, total_tokens=2000, block_size=16)
    assert a == b
    assert a.layer_budgets == b.layer_budgets
