"""Tests for the compute-bandwidth-aware cost model."""

from __future__ import annotations

import pytest

from actfold.utils.cost_model import ComputeBandwidthCostModel, HardwareProfile


def test_estimate_compute_decreases_with_higher_stability() -> None:
    """Higher stable ratio reduces compute FLOPs while memory cost rises."""
    hw = HardwareProfile(compute_tflops=100.0, memory_bw_gb_s=600.0)
    model = ComputeBandwidthCostModel(hw)

    cost_low = model.layer_cost(seq_len=64, hidden_dim=128, stable_ratio=0.0)
    cost_mid = model.layer_cost(seq_len=64, hidden_dim=128, stable_ratio=0.5)
    cost_high = model.layer_cost(seq_len=64, hidden_dim=128, stable_ratio=0.9)

    assert cost_low.compute_flops > cost_mid.compute_flops > cost_high.compute_flops
    assert cost_low.memory_bytes < cost_mid.memory_bytes < cost_high.memory_bytes


def test_stable_token_cost_is_nonzero() -> None:
    """Even with 100% stability there is still gate and memory cost."""
    hw = HardwareProfile(compute_tflops=100.0, memory_bw_gb_s=600.0)
    model = ComputeBandwidthCostModel(hw)
    t = model.estimate_layer_time(seq_len=64, hidden_dim=128, stable_ratio=1.0)
    assert t > 0.0


def test_layer_cost_breakdown_sums() -> None:
    """LayerCost fields are consistent with the total estimate."""
    hw = HardwareProfile(compute_tflops=100.0, memory_bw_gb_s=600.0)
    model = ComputeBandwidthCostModel(hw)
    cost = model.layer_cost(seq_len=32, hidden_dim=64, stable_ratio=0.5)
    assert cost.estimated_time_ms > 0.0
    assert cost.compute_flops > 0.0
    assert cost.memory_bytes > 0.0


def test_calibration_updates_throughput() -> None:
    """Calibration adjusts compute_tflops based on measured time."""
    hw = HardwareProfile(compute_tflops=100.0, memory_bw_gb_s=600.0)
    model = ComputeBandwidthCostModel(hw)
    original = model.hw.compute_tflops

    # Provide a measured time that is 2x the initial estimate.
    measured = (
        model.estimate_total_time(num_layers=2, seq_len=32, hidden_dim=64, stable_ratio=0.5)
        * 1000.0
    )
    model.calibrate(
        measured_stable_ratio=0.5,
        measured_seq_len=32,
        measured_hidden_dim=64,
        measured_time_ms=measured * 2,
        num_layers=2,
    )
    assert model.hw.compute_tflops != pytest.approx(original)
