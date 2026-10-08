"""Tests for actfold.utils.gpu_profiler."""

from __future__ import annotations

import time

import torch

from actfold.utils.gpu_profiler import GPUMeasurement, gpu_profile


def test_gpu_profile_cpu_latency_measured() -> None:
    """B8: the CPU fallback must measure real wall-clock latency, not 0."""
    with gpu_profile(device="cpu") as measurement:
        time.sleep(0.05)
        x = torch.randn(4, 4)
        _ = x @ x.T

    assert isinstance(measurement, GPUMeasurement)
    assert measurement.latency_ms > 20.0
    # Peak memory stats are CUDA-allocator specific; CPU reports 0.
    assert measurement.peak_memory_mb == 0.0


def test_gpu_profile_cpu_latency_zero_for_noop() -> None:
    """An empty CPU context still yields a finite, non-negative latency."""
    with gpu_profile(device="cpu") as measurement:
        pass

    assert measurement.latency_ms >= 0.0
    assert measurement.peak_memory_mb == 0.0
