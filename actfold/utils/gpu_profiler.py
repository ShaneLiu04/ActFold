"""GPU profiling utilities."""

from __future__ import annotations

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

    If CUDA is unavailable, returns zeros and logs a warning.

    Args:
        device: Target PyTorch device.

    Yields:
        A GPUMeasurement populated after the context exits.
    """
    measurement = GPUMeasurement(latency_ms=0.0, peak_memory_mb=0.0)
    use_cuda = torch.cuda.is_available() and "cuda" in device

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start_event.record()  # type: ignore[no-untyped-call]
    else:
        start_event = None
        end_event = None

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
