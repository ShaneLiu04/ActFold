"""Compute-bandwidth-aware cost model for ActFold.

This module complements :mod:`~actfold.utils.flops_counter` by modelling the
memory-bandwidth and auxiliary costs that the simple FLOPs estimator ignores:
gathering cached activations, merging stable/divergent outputs, and running the
similarity gate.  The resulting estimates are closer to wall-clock latency,
especially in memory-bound regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HardwareProfile:
    """Hardware performance constants used by the cost model."""

    compute_tflops: float  # Peak compute throughput (TFLOPs/s).
    memory_bw_gb_s: float  # Achievable memory bandwidth (GB/s).
    bytes_per_element: int = 4  # fp32=4, fp16/bf16=2.

    @classmethod
    def from_device(cls, device: torch.device | str) -> "HardwareProfile":
        """Return a conservative default profile for ``device``."""
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            # Conservative defaults for a mid-range GPU.
            return cls(compute_tflops=100.0, memory_bw_gb_s=600.0, bytes_per_element=2)
        # CPU defaults.
        return cls(compute_tflops=1.0, memory_bw_gb_s=50.0, bytes_per_element=4)


@dataclass
class LayerCost:
    """Cost breakdown for one Transformer layer under folding."""

    compute_flops: float
    memory_bytes: float
    gate_flops: float
    merge_flops: float
    estimated_time_ms: float


class ComputeBandwidthCostModel:
    """Estimate layer execution time accounting for compute and memory.

    The model treats divergent tokens as compute-bound (attention + FFN) and
    stable tokens as memory-bound (read cached FFN output + write merged result).
    It also includes the gate and merge overheads that ActFold introduces.

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

        # Compute cost: only divergent tokens run attention + FFN.
        compute_flops = (
            (1.0 - r) * t * h * h * (self._attention_flops_per_token + self._ffn_flops_per_token)
        )
        compute_time = compute_flops / (self.hw.compute_tflops * 1e12)

        # Memory cost: read cached stable activations + write merged output.
        # We account for both "hidden_states" and "ffn_out" activations.
        memory_bytes = r * t * h * self.hw.bytes_per_element * 4
        memory_time = memory_bytes / (self.hw.memory_bw_gb_s * 1e9)

        # Gate cost: cosine similarity over [batch, seq_len, hidden_dim].
        gate_flops = t * h * 4  # multiply + reduce approximations
        gate_time = gate_flops / (self.hw.compute_tflops * 1e12)

        # Merge cost: copy stable positions from parent and divergent from child.
        merge_flops = r * t * h
        merge_time = merge_flops / (self.hw.compute_tflops * 1e12)

        return compute_time + memory_time + gate_time + merge_time

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

        compute_flops = (
            (1.0 - r) * t * h * h * (self._attention_flops_per_token + self._ffn_flops_per_token)
        )
        memory_bytes = r * t * h * self.hw.bytes_per_element * 4
        gate_flops = t * h * 4
        merge_flops = r * t * h
        time_s = self.estimate_layer_time(seq_len, hidden_dim, stable_ratio, num_heads)

        return LayerCost(
            compute_flops=compute_flops,
            memory_bytes=memory_bytes,
            gate_flops=gate_flops,
            merge_flops=merge_flops,
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
