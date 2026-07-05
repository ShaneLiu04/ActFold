"""Tests for the layer-aware stability profiler."""

from __future__ import annotations

import pytest
import torch

from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER, StabilityProfiler


def test_profiler_records_layer_stats() -> None:
    """Recording a stable mask creates a profile with the correct ratio."""
    profiler = StabilityProfiler(enabled=True)
    mask = torch.tensor([[True, True, False, True]])
    profiler.record("child", "parent", layer_idx=1, step_idx=0, stable_mask=mask, tau=0.95)

    profile = profiler.get_profile("child")
    assert profile is not None
    assert len(profile.layer_stats) == 1
    assert profile.layer_stats[0].stable_ratio == pytest.approx(0.75)
    assert profile.layer_stats[0].layer_idx == 1
    assert profile.mean_stable_ratio == pytest.approx(0.75)


def test_profiler_disabled_has_no_effect() -> None:
    """A disabled profiler does not store anything."""
    profiler = StabilityProfiler(enabled=False)
    mask = torch.ones((1, 4), dtype=torch.bool)
    profiler.record("child", "parent", layer_idx=0, step_idx=0, stable_mask=mask, tau=0.95)
    assert profiler.get_profile("child") is None


def test_global_profiler_reset() -> None:
    """Resetting the global profiler clears all state."""
    GLOBAL_STABILITY_PROFILER.record(
        "child",
        "parent",
        layer_idx=0,
        step_idx=0,
        stable_mask=torch.ones((1, 2), dtype=torch.bool),
        tau=0.9,
    )
    assert GLOBAL_STABILITY_PROFILER.get_profile("child") is not None
    GLOBAL_STABILITY_PROFILER.reset()
    assert GLOBAL_STABILITY_PROFILER.get_profile("child") is None


def test_history_mean_returns_none_when_empty() -> None:
    """Historical mean returns None before any recording."""
    profiler = StabilityProfiler(enabled=True)
    assert profiler.get_mean_stable_ratio(0, 0) is None


def test_history_mean_computed_correctly() -> None:
    """Historical mean averages previous stable ratios."""
    profiler = StabilityProfiler(enabled=True)
    profiler.record(
        "b1",
        None,
        layer_idx=0,
        step_idx=0,
        stable_mask=torch.ones((1, 2), dtype=torch.bool),
        tau=0.9,
    )
    profiler.record(
        "b2",
        None,
        layer_idx=0,
        step_idx=0,
        stable_mask=torch.zeros((1, 2), dtype=torch.bool),
        tau=0.9,
    )
    mean = profiler.get_mean_stable_ratio(0, 0)
    assert mean == pytest.approx(0.5)
