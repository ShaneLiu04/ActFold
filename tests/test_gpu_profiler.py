"""Tests for actfold.utils.gpu_profiler."""

from __future__ import annotations

import torch

from actfold.utils.gpu_profiler import GPUMeasurement, gpu_profile


def test_gpu_profile_cpu_fallback() -> None:
    with gpu_profile(device="cpu") as measurement:
        x = torch.randn(4, 4)
        _ = x @ x.T

    assert isinstance(measurement, GPUMeasurement)
    assert measurement.latency_ms == 0.0
    assert measurement.peak_memory_mb == 0.0
