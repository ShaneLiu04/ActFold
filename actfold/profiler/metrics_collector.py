"""GPU metrics collection during inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class InferenceMetrics:
    """Collected inference metrics."""

    nfe: int = 0
    tflops: float = 0.0
    latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_tok_per_sec: float = 0.0
    custom: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Context manager for collecting GPU metrics during inference.

    Args:
        device: Target PyTorch device.
        seq_len: Sequence length for throughput estimation.
    """

    def __init__(self, device: str = "cuda", seq_len: int = 1) -> None:
        self.device = device
        self.seq_len = seq_len
        self.metrics = InferenceMetrics()
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None

    def __enter__(self) -> "MetricsCollector":
        self._reset()
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            self._end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            self._start_event.record()  # type: ignore[no-untyped-call]
        return self

    def __exit__(self, *exc: Any) -> None:
        if (
            torch.cuda.is_available()
            and "cuda" in self.device
            and self._start_event is not None
            and self._end_event is not None
        ):
            self._end_event.record()  # type: ignore[no-untyped-call]
            torch.cuda.synchronize(self.device)
            self.metrics.latency_ms = self._start_event.elapsed_time(self._end_event)  # type: ignore[no-untyped-call]
            self.metrics.peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (
                1024 * 1024
            )
        else:
            self.metrics.latency_ms = 0.0
            self.metrics.peak_memory_mb = 0.0

        if self.metrics.latency_ms > 0:
            self.metrics.throughput_tok_per_sec = self.seq_len / (self.metrics.latency_ms / 1000.0)

    def _reset(self) -> None:
        """Reset internal state."""
        self.metrics = InferenceMetrics()
        self._start_event = None
        self._end_event = None

    def record(self, name: str, value: Any) -> None:
        """Record a custom metric."""
        self.metrics.custom[name] = value

    def set_nfe(self, nfe: int) -> None:
        """Set the number of function evaluations."""
        self.metrics.nfe = nfe

    def set_tflops(self, tflops: float) -> None:
        """Set the estimated TFLOPs."""
        self.metrics.tflops = tflops

    def get_metrics(self) -> InferenceMetrics:
        """Return the collected metrics."""
        return self.metrics
