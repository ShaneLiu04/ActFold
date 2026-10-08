"""GPU profiling utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch


@dataclass(frozen=True)
class GPUMeasurement:
    """GPU timing and memory measurement."""

    latency_ms: float
    peak_memory_mb: float


@contextmanager
def gpu_profile(device: str = "cuda") -> Generator[GPUMeasurement, None, None]:
    """Context manager that profiles GPU latency and peak memory.

    CUDA devices are timed with CUDA events; other devices fall back to
    wall-clock ``time.perf_counter`` timing so CPU runs report real latency
    instead of zeros (peak memory stays 0.0 off-CUDA; it is a
    CUDA-allocator statistic).

    Args:
        device: Target PyTorch device.

    Yields:
        A GPUMeasurement populated after the context exits.
    """
    measurement = GPUMeasurement(latency_ms=0.0, peak_memory_mb=0.0)
    use_cuda = torch.cuda.is_available() and "cuda" in device

    start_event = None
    end_event = None
    start_time = 0.0
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start_event.record()  # type: ignore[no-untyped-call]
    else:
        start_time = time.perf_counter()

    try:
        yield measurement
    finally:
        if use_cuda and start_event is not None and end_event is not None:
            end_event.record()  # type: ignore[no-untyped-call]
            torch.cuda.synchronize(device)
            latency_ms = start_event.elapsed_time(end_event)  # type: ignore[no-untyped-call]
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            # Use object.__setattr__ because dataclass is frozen.
            object.__setattr__(measurement, "latency_ms", latency_ms)
            object.__setattr__(measurement, "peak_memory_mb", peak_memory_mb)
        else:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            object.__setattr__(measurement, "latency_ms", latency_ms)
