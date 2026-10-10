"""Tests for the layer-aware stability profiler."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER, StabilityProfiler


def _raise_item(self: torch.Tensor) -> Any:
    """Stand-in for ``torch.Tensor.item`` that flags a device->host sync."""
    raise AssertionError("sync during record")


def _raise_nonzero(*args: Any, **kwargs: Any) -> Any:
    """Stand-in for ``torch.nonzero`` that flags a data-dependent sync."""
    raise AssertionError("nonzero during record")


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


def test_record_does_not_sync() -> None:
    """record() must not sync: no .item() and no torch.nonzero calls (T009)."""
    profiler = StabilityProfiler()
    mask = torch.tensor([[True, True, False, False]])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "item", _raise_item)
        mp.setattr(torch, "nonzero", _raise_nonzero)
        profiler.record("b", "p", layer_idx=0, step_idx=0, stable_mask=mask, tau=0.95)

    profile = profiler.get_profile("b")
    assert profile is not None
    assert profile.layer_stats[0].stable_ratio == pytest.approx(0.5)
    assert profile.layer_stats[0].num_tokens == 4


def test_record_no_divergence_positions_by_default() -> None:
    """With debug collection off, divergence_positions must stay None (T009)."""
    profiler = StabilityProfiler()
    mask = torch.tensor([[True, False, True, False]])
    profiler.record("b", "p", layer_idx=0, step_idx=0, stable_mask=mask, tau=0.95)

    profile = profiler.get_profile("b")
    assert profile is not None
    assert profile.layer_stats[0].divergence_positions is None


def test_debug_enabled_collects_divergence_positions() -> None:
    """debug_enabled=True collects divergence positions matching nonzero(~mask)."""
    profiler = StabilityProfiler(debug_enabled=True)
    mask = torch.tensor([[True, False, False, True]])
    profiler.record("b", "p", layer_idx=0, step_idx=0, stable_mask=mask, tau=0.95)

    profile = profiler.get_profile("b")
    assert profile is not None
    positions = profile.layer_stats[0].divergence_positions
    assert positions is not None

    expected = torch.nonzero(~mask, as_tuple=False)
    got = {tuple(int(v) for v in row) for row in positions.tolist()}
    want = {tuple(int(v) for v in row) for row in expected.tolist()}
    assert got == want


def test_get_profile_materializes_once_per_branch() -> None:
    """get_profile syncs at most once per branch and caches the result (T009).

    The key contract pinned here: repeated get_profile calls for the same
    branch must not re-sync (zero additional ``.item()`` calls).
    """
    profiler = StabilityProfiler()
    masks = [
        torch.ones((1, 4), dtype=torch.bool),
        torch.tensor([[True, True, False, False]]),
        torch.zeros((1, 4), dtype=torch.bool),
    ]
    for layer_idx, mask in enumerate(masks):
        profiler.record("b", "p", layer_idx=layer_idx, step_idx=0, stable_mask=mask, tau=0.95)

    calls = {"item": 0}
    original_item = torch.Tensor.item

    def _counting_item(self: torch.Tensor) -> Any:
        calls["item"] += 1
        return original_item(self)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "item", _counting_item)
        profile = profiler.get_profile("b")
        first_read_calls = calls["item"]
        assert first_read_calls <= 4
        cached_profile = profiler.get_profile("b")
        assert calls["item"] == first_read_calls

    assert profile is not None
    assert profile.mean_stable_ratio == pytest.approx(0.5)
    assert cached_profile is not None
    assert cached_profile.mean_stable_ratio == pytest.approx(0.5)


def test_history_mean_deferred() -> None:
    """History records must not sync; the mean is computed on read (T009)."""
    profiler = StabilityProfiler()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "item", _raise_item)
        mp.setattr(torch, "nonzero", _raise_nonzero)
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
            stable_mask=torch.tensor([[True, False]]),
            tau=0.9,
        )

    mean = profiler.get_mean_stable_ratio(0, 0)
    assert mean == pytest.approx(0.75)


def test_reset_branch_drops_everything() -> None:
    """reset_branch drops raw + materialized data, including history (T009)."""
    profiler = StabilityProfiler()
    profiler.record(
        "b",
        "p",
        layer_idx=0,
        step_idx=0,
        stable_mask=torch.tensor([[True, True, False, False]]),
        tau=0.95,
    )
    profiler.reset_branch("b")
    assert profiler.get_profile("b") is None
    assert profiler.get_mean_stable_ratio(0, 0) is None


def test_multi_branch_isolation() -> None:
    """Profiles of different branches stay isolated with their own ratios."""
    profiler = StabilityProfiler()
    profiler.record(
        "b1",
        None,
        layer_idx=0,
        step_idx=0,
        stable_mask=torch.ones((1, 4), dtype=torch.bool),
        tau=0.9,
    )
    profiler.record(
        "b2",
        None,
        layer_idx=0,
        step_idx=0,
        stable_mask=torch.tensor([[True, False, False, False]]),
        tau=0.9,
    )

    profile_b1 = profiler.get_profile("b1")
    profile_b2 = profiler.get_profile("b2")
    assert profile_b1 is not None
    assert profile_b2 is not None
    assert profile_b1.mean_stable_ratio != profile_b2.mean_stable_ratio
    assert profile_b1.mean_stable_ratio == pytest.approx(1.0)
    assert profile_b2.mean_stable_ratio == pytest.approx(0.25)


def test_record_preserves_tau_and_metric() -> None:
    """Materialized stats carry the tau and metric passed to record()."""
    profiler = StabilityProfiler()
    profiler.record(
        "b",
        "p",
        layer_idx=2,
        step_idx=1,
        stable_mask=torch.ones((1, 4), dtype=torch.bool),
        tau=0.87,
        metric="l2",
    )

    profile = profiler.get_profile("b")
    assert profile is not None
    stats = profile.layer_stats[0]
    assert stats.tau_used == pytest.approx(0.87)
    assert stats.metric == "l2"
