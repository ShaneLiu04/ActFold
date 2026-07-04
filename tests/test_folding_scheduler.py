"""Tests for actfold.core.folding_scheduler."""

from __future__ import annotations

import pytest

from actfold.core.folding_scheduler import FoldingScheduler


def test_get_tau_range() -> None:
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=12, num_steps=10)
    tau = scheduler.get_tau(layer_idx=0, step_idx=0)
    assert 0.80 <= tau <= 0.99


def test_get_tau_monotonicity() -> None:
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=12, num_steps=10)
    # Deeper layers should tend to have lower tau (less reuse).
    early_tau = scheduler.get_tau(layer_idx=0, step_idx=0)
    late_tau = scheduler.get_tau(layer_idx=11, step_idx=0)
    assert late_tau <= early_tau


def test_get_tau_invalid_indices() -> None:
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=12, num_steps=10)
    with pytest.raises(ValueError):
        scheduler.get_tau(layer_idx=12, step_idx=0)
    with pytest.raises(ValueError):
        scheduler.get_tau(layer_idx=0, step_idx=10)


def test_should_fold_disabled_at_boundaries() -> None:
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=4, num_steps=4)
    assert scheduler.should_fold(layer_idx=0, step_idx=0) is True
    assert scheduler.should_fold(layer_idx=2, step_idx=2) is True
    assert scheduler.should_fold(layer_idx=3, step_idx=0) is False
    assert scheduler.should_fold(layer_idx=0, step_idx=3) is False
    assert scheduler.should_fold(layer_idx=3, step_idx=3) is False


def test_should_fold_invalid_indices() -> None:
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=4, num_steps=4)
    assert scheduler.should_fold(layer_idx=-1, step_idx=0) is False
    assert scheduler.should_fold(layer_idx=4, step_idx=0) is False


def test_set_task_bias() -> None:
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=4, num_steps=4)
    scheduler.set_task_bias("custom", 0.05)
    assert scheduler.task_bias["custom"] == 0.05


def test_invalid_base_tau() -> None:
    with pytest.raises(ValueError):
        FoldingScheduler(base_tau=1.5)


def test_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        FoldingScheduler(num_layers=0)
