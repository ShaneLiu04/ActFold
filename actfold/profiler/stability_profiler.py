"""Layer-aware stability profiling for ActFold.

The profiler records per-layer, per-step stability decisions made by
:class:`~actfold.core.folded_transformer.FoldedTransformerLayer`.  These
statistics replace the embedding-level proxy used by the verification engine
with real measurements taken at the locations where folding actually occurs.

Recording is asynchronous: ``record`` only accumulates GPU-side sum tensors
(no ``.item()`` readback, no ``nonzero``), so the folded forward path never
synchronizes with the device for profiling. The host-side ratios are
materialized lazily in a single batched readback when a profile is read
(``get_profile`` / ``get_mean_stable_ratio``). Divergence positions are
collected only in explicit ``debug_enabled`` mode because ``nonzero`` forces
a data-dependent device sync.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class LayerStabilityStats:
    """Stability statistics for a single layer and step."""

    layer_idx: int
    step_idx: int
    stable_ratio: float
    tau_used: float
    metric: str
    num_tokens: int
    divergence_positions: Optional[torch.Tensor] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "layer_idx": self.layer_idx,
            "step_idx": self.step_idx,
            "stable_ratio": self.stable_ratio,
            "tau_used": self.tau_used,
            "metric": self.metric,
            "num_tokens": self.num_tokens,
            "divergence_positions": (
                self.divergence_positions.tolist()
                if self.divergence_positions is not None
                else None
            ),
        }


@dataclass
class StabilityProfile:
    """A complete stability profile for one child-forward pass."""

    branch_id: Any
    parent_branch_id: Optional[Any]
    layer_stats: list[LayerStabilityStats] = field(default_factory=list)

    @property
    def mean_stable_ratio(self) -> float:
        """Average stable ratio across all recorded layers."""
        if not self.layer_stats:
            return 0.0
        return sum(s.stable_ratio for s in self.layer_stats) / len(self.layer_stats)

    @property
    def final_stable_ratio(self) -> float:
        """Stable ratio at the final recorded layer."""
        if not self.layer_stats:
            return 0.0
        return self.layer_stats[-1].stable_ratio

    @property
    def min_stable_ratio(self) -> float:
        """Minimum stable ratio observed across layers."""
        if not self.layer_stats:
            return 0.0
        return min(s.stable_ratio for s in self.layer_stats)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "mean_stable_ratio": self.mean_stable_ratio,
            "final_stable_ratio": self.final_stable_ratio,
            "min_stable_ratio": self.min_stable_ratio,
            "layer_stats": [s.to_dict() for s in self.layer_stats],
        }


@dataclass
class _RawLayerStat:
    """Deferred (device-side) stability measurement for one layer."""

    branch_id: Any
    parent_branch_id: Optional[Any]
    layer_idx: int
    step_idx: int
    stable_sum: torch.Tensor
    num_tokens: int
    tau: float
    metric: str
    divergence_positions: Optional[torch.Tensor] = None


class StabilityProfiler:
    """Global layer-aware stability profiler.

    The profiler is designed as a singleton-like object that can be disabled or
    re-enabled at runtime. When enabled, every folded layer records its
    stability decision without any host readback; consumers (e.g. the
    verification engine) materialize the resulting profile lazily.

    Args:
        enabled: Whether to collect statistics. Disabling removes all overhead.
        debug_enabled: Collect per-layer divergence positions. Off by default:
            ``torch.nonzero`` forces a data-dependent device sync on every
            record, which is only acceptable when explicitly debugging.
    """

    def __init__(self, enabled: bool = True, debug_enabled: bool = False) -> None:
        self.enabled = enabled
        self.debug_enabled = debug_enabled
        self._raw: dict[Any, list[_RawLayerStat]] = {}
        # Materialized profiles, filled lazily by get_profile.
        self._profiles: dict[Any, StabilityProfile] = {}
        # (layer_idx, step_idx) -> deferred history entries (branch-attributed).
        self._history: dict[tuple[int, int], list[_RawLayerStat]] = defaultdict(list)
        self._max_history_len = 100

    def record(
        self,
        branch_id: Any,
        parent_branch_id: Optional[Any],
        layer_idx: int,
        step_idx: int,
        stable_mask: torch.Tensor,
        tau: float,
        metric: str = "cosine",
    ) -> None:
        """Record a single layer's stability decision without host readback.

        Args:
            branch_id: Identifier of the current branch.
            parent_branch_id: Identifier of the parent branch (if any).
            layer_idx: Layer index.
            step_idx: Diffusion step index.
            stable_mask: Boolean tensor ``[batch, seq_len]``.
            tau: Similarity threshold used for the decision.
            metric: Similarity metric name.
        """
        if not self.enabled:
            return

        # Device-side accumulation only: no .item(), no nonzero on this path.
        stable_sum = stable_mask.sum()
        num_tokens = int(stable_mask.numel())

        divergence_positions = None
        if self.debug_enabled and num_tokens > 0 and not bool(stable_mask.all()):
            divergence_positions = torch.nonzero(~stable_mask, as_tuple=False)

        raw = _RawLayerStat(
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            layer_idx=layer_idx,
            step_idx=step_idx,
            stable_sum=stable_sum,
            num_tokens=num_tokens,
            tau=tau,
            metric=metric,
            divergence_positions=divergence_positions,
        )
        self._raw.setdefault(branch_id, []).append(raw)

        history_key = (layer_idx, step_idx)
        self._history[history_key].append(raw)
        if len(self._history[history_key]) > self._max_history_len:
            self._history[history_key].pop(0)

    @staticmethod
    def _ratios(entries: list[_RawLayerStat]) -> list[float]:
        """Materialize per-entry stable ratios in one batched host readback."""
        if not entries:
            return []
        try:
            sums = torch.stack([e.stable_sum for e in entries]).float()
            counts = torch.tensor(
                [max(e.num_tokens, 1) for e in entries],
                dtype=sums.dtype,
                device=sums.device,
            )
            return (sums / counts).tolist()
        except RuntimeError:
            # Mixed devices/dtypes: fall back to per-entry readback.
            return [
                float(e.stable_sum.float().item()) / max(e.num_tokens, 1) for e in entries
            ]

    def get_profile(self, branch_id: Any) -> Optional[StabilityProfile]:
        """Return the stability profile for ``branch_id`` if one exists.

        The first call materializes all of the branch's deferred layer
        statistics in a single batched readback; subsequent calls return the
        cached profile without touching the device.
        """
        cached = self._profiles.get(branch_id)
        if cached is not None:
            return cached
        raw_entries = self._raw.get(branch_id)
        if not raw_entries:
            return None

        ratios = self._ratios(raw_entries)
        profile = StabilityProfile(
            branch_id=branch_id,
            parent_branch_id=raw_entries[0].parent_branch_id,
        )
        profile.layer_stats = [
            LayerStabilityStats(
                layer_idx=e.layer_idx,
                step_idx=e.step_idx,
                stable_ratio=ratio,
                tau_used=e.tau,
                metric=e.metric,
                num_tokens=e.num_tokens,
                divergence_positions=e.divergence_positions,
            )
            for e, ratio in zip(raw_entries, ratios)
        ]
        self._profiles[branch_id] = profile
        return profile

    def all_profiles(self) -> dict[Any, StabilityProfile]:
        """Return a snapshot of every recorded branch profile."""
        for branch_id in list(self._raw):
            self.get_profile(branch_id)
        return dict(self._profiles)

    def get_mean_stable_ratio(
        self,
        layer_idx: int,
        step_idx: int = 0,
    ) -> Optional[float]:
        """Return the historical mean stable ratio for a layer/step pair."""
        history = self._history.get((layer_idx, step_idx))
        if not history:
            return None
        ratios = self._ratios(list(history))
        return sum(ratios) / len(ratios)

    def reset_branch(self, branch_id: Any) -> None:
        """Drop the raw and materialized data for a single branch."""
        self._raw.pop(branch_id, None)
        self._profiles.pop(branch_id, None)
        for key in list(self._history):
            self._history[key] = [e for e in self._history[key] if e.branch_id != branch_id]
            if not self._history[key]:
                del self._history[key]

    def reset(self) -> None:
        """Drop all collected profiles and history."""
        self._raw.clear()
        self._profiles.clear()
        self._history.clear()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable profiling."""
        self.enabled = enabled
        if not enabled:
            self.reset()


# Global profiler instance.  Code that does not need explicit control can import
# this directly; tests or multi-tenant callers may create their own instance.
GLOBAL_STABILITY_PROFILER = StabilityProfiler(enabled=True)
