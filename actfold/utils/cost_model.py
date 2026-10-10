"""Compute-bandwidth-aware cost model for ActFold.

This module complements :mod:`~actfold.utils.flops_counter` by modelling the
memory-bandwidth and auxiliary costs that the simple FLOPs estimator ignores:
reading cached activations and the KV working set, merging stable/divergent
outputs, and running the similarity gate.  The resulting estimates are closer
to wall-clock latency, especially in memory-bound regimes.

Hardware profiles are resolved per device via :data:`_DEVICE_TABLE` (known GPU
name substrings → bf16 dense TFLOPS / memory GB/s) with a conservative
fallback, or measured directly with a short micro-benchmark
(``HardwareProfile.from_device(device, calibrate=True)``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

# Uppercase device-name substring → (bf16 dense TFLOPS, achievable memory GB/s).
# Values are vendor peak dense numbers derated to realistic achievable rates.
_DEVICE_TABLE: dict[str, tuple[float, float]] = {
    "H100": (989.0, 3350.0),
    "A100": (312.0, 1555.0),
    "RTX PRO 6000": (400.0, 1600.0),
    "4090": (165.0, 1008.0),
    "RTX 6000": (136.0, 1280.0),
    "L40S": (183.0, 864.0),
    "3090": (71.0, 936.0),
    "V100": (125.0, 900.0),
    "T4": (65.0, 320.0),
    "QUADRO RTX 5000": (60.0, 448.0),
}

_FALLBACK_CUDA_PROFILE = (100.0, 600.0)


def _microbench_cuda(index: int | None) -> tuple[float, float]:
    """Measure bf16 dense matmul TFLOPS and copy bandwidth on a CUDA device.

    Runs a short (~2 s) micro-benchmark: a 4096³ bf16 matmul loop for compute
    throughput and a 512 MiB device-to-device copy loop for memory bandwidth.
    """
    device = torch.device("cuda", index) if index is not None else torch.device("cuda")
    m = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    for _ in range(3):
        m = m @ m
    torch.cuda.synchronize()
    start = time.perf_counter()
    reps = 20
    for _ in range(reps):
        m = m @ m
    torch.cuda.synchronize()
    matmul_s = (time.perf_counter() - start) / reps
    tflops = 2 * 4096**3 / matmul_s / 1e12
    del m

    numel = 256 * 1024 * 1024  # 512 MiB of bf16
    a = torch.empty(numel, dtype=torch.bfloat16, device=device)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()
    copy_s = (time.perf_counter() - start) / 10
    bandwidth_gb_s = 2 * numel * 2 / copy_s / 1e9
    return tflops, bandwidth_gb_s


@dataclass
class HardwareProfile:
    """Hardware performance constants used by the cost model."""

    compute_tflops: float  # Peak compute throughput (TFLOPs/s).
    memory_bw_gb_s: float  # Achievable memory bandwidth (GB/s).
    bytes_per_element: int = 4  # fp32=4, fp16/bf16=2.

    @classmethod
    def from_device(cls, device: torch.device | str, calibrate: bool = False) -> "HardwareProfile":
        """Return a hardware profile for ``device``.

        For CUDA devices the profile is resolved by looking up the actual
        device name in :data:`_DEVICE_TABLE`; unknown devices fall back to a
        conservative mid-range GPU profile.  With ``calibrate=True`` a short
        micro-benchmark (bf16 matmul + device copy) measures the achievable
        throughput instead of trusting the table.  Calibration only runs on
        CUDA; on CPU the defaults are returned unchanged.

        Args:
            device: Torch device (or device string) to profile.
            calibrate: When True, run a micro-benchmark on CUDA to measure
                (rather than look up) compute throughput and bandwidth.

        Returns:
            A conservative-but-device-specific :class:`HardwareProfile`.
        """
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            if calibrate:
                tflops, gbps = _microbench_cuda(device.index)
                return cls(compute_tflops=tflops, memory_bw_gb_s=gbps, bytes_per_element=2)
            name = torch.cuda.get_device_name(device.index if device.index else None)
            upper = name.upper()
            for key, (tflops, gbps) in _DEVICE_TABLE.items():
                if key in upper:
                    return cls(compute_tflops=tflops, memory_bw_gb_s=gbps, bytes_per_element=2)
            tflops, gbps = _FALLBACK_CUDA_PROFILE
            return cls(compute_tflops=tflops, memory_bw_gb_s=gbps, bytes_per_element=2)
        # CPU defaults.
        return cls(compute_tflops=1.0, memory_bw_gb_s=50.0, bytes_per_element=4)


@dataclass
class LayerCost:
    """Cost breakdown for one Transformer layer under folding."""

    compute_flops: float
    memory_bytes: float
    gate_bytes: float
    merge_bytes: float
    estimated_time_ms: float


class ComputeBandwidthCostModel:
    """Estimate layer execution time accounting for compute and memory.

    The model treats divergent tokens as compute-bound (attention + FFN,
    including the quadratic attention score/value term) and stable tokens as
    memory-bound (read cached FFN output + write merged result).  The KV
    working-set read and the gate/merge traffic ActFold introduces are charged
    to the memory-bandwidth term.

    Args:
        hw: Hardware profile.  If ``None``, a default profile is chosen based on
            the current device.
    """

    def __init__(self, hw: Optional[HardwareProfile] = None) -> None:
        self.hw = hw or HardwareProfile.from_device("cuda" if torch.cuda.is_available() else "cpu")
        self._attention_flops_per_token = 4.0  # hidden_dim^2 scaled outside
        self._ffn_flops_per_token = 16.0  # hidden_dim^2 scaled outside

    def estimate_layer_time(
        self,
        seq_len: int,
        hidden_dim: int,
        stable_ratio: float,
        num_heads: int = 1,
    ) -> float:
        """Return estimated layer execution time in seconds.

        Args:
            seq_len: Sequence length.
            hidden_dim: Hidden dimension size.
            stable_ratio: Fraction of stable tokens.
            num_heads: Number of attention heads (currently unused).

        Returns:
            Estimated wall-clock time for one layer in seconds.
        """
        del num_heads
        r = float(stable_ratio)
        t = float(seq_len)
        h = float(hidden_dim)
        b = float(self.hw.bytes_per_element)

        # Compute cost: only divergent tokens run attention + FFN.  The
        # attention term includes the quadratic score/value contribution
        # (2 * T^2 * h) that the per-token constants miss.
        compute_flops = (1.0 - r) * t * h * h * (
            self._attention_flops_per_token + self._ffn_flops_per_token
        ) + 2.0 * (1.0 - r) * t * t * h
        compute_time = compute_flops / (self.hw.compute_tflops * 1e12)

        # Memory cost:
        # - read cached stable activations + write merged result (two tensors,
        #   read + write each);
        # - KV working-set read (K and V, one vector each per position);
        # - gate reads parent + child hidden states;
        # - merge reads parent + child FFN rows and writes the merged output.
        memory_bytes = (
            r * t * h * b * 4.0
            + 2.0 * t * h * b  # KV read
            + 2.0 * t * h * b  # gate: parent + child hidden
            + 3.0 * t * h * b  # merge: parent FFN + child FFN + output write
        )
        memory_time = memory_bytes / (self.hw.memory_bw_gb_s * 1e9)

        return compute_time + memory_time

    def estimate_total_time(
        self,
        num_layers: int,
        seq_len: int,
        hidden_dim: int,
        stable_ratio: float,
        num_steps: int = 1,
        num_heads: int = 1,
    ) -> float:
        """Return estimated total forward time in seconds."""
        layer_time = self.estimate_layer_time(seq_len, hidden_dim, stable_ratio, num_heads)
        return layer_time * num_layers * num_steps

    def layer_cost(
        self,
        seq_len: int,
        hidden_dim: int,
        stable_ratio: float,
        num_heads: int = 1,
    ) -> LayerCost:
        """Return a detailed cost breakdown for one layer."""
        r = float(stable_ratio)
        t = float(seq_len)
        h = float(hidden_dim)
        b = float(self.hw.bytes_per_element)

        compute_flops = (1.0 - r) * t * h * h * (
            self._attention_flops_per_token + self._ffn_flops_per_token
        ) + 2.0 * (1.0 - r) * t * t * h
        gate_bytes = 2.0 * t * h * b
        merge_bytes = 3.0 * t * h * b
        memory_bytes = r * t * h * b * 4.0 + 2.0 * t * h * b + gate_bytes + merge_bytes
        time_s = self.estimate_layer_time(seq_len, hidden_dim, stable_ratio, num_heads)

        return LayerCost(
            compute_flops=compute_flops,
            memory_bytes=memory_bytes,
            gate_bytes=gate_bytes,
            merge_bytes=merge_bytes,
            estimated_time_ms=time_s * 1000.0,
        )

    def calibrate(
        self,
        measured_stable_ratio: float,
        measured_seq_len: int,
        measured_hidden_dim: int,
        measured_time_ms: float,
        num_layers: int,
    ) -> None:
        """Calibrate compute throughput based on a measured layer time.

        This updates ``hw.compute_tflops`` so that future estimates better match
        the observed latency.  It can be called repeatedly with fresh
        measurements.
        """
        estimated = self.estimate_layer_time(
            measured_seq_len,
            measured_hidden_dim,
            measured_stable_ratio,
        )
        if estimated <= 0:
            return
        per_layer_ms = measured_time_ms / max(1, num_layers)
        ratio = estimated / (per_layer_ms / 1000.0)
        # Smooth update to avoid over-fitting to a single measurement.
        self.hw.compute_tflops = 0.9 * self.hw.compute_tflops + 0.1 * self.hw.compute_tflops * ratio
