"""Layer-aware stability profiling for ActFold.

The profiler records per-layer, per-step stability decisions made by
:class:`~actfold.core.folded_transformer.FoldedTransformerLayer`.  These
statistics replace the embedding-level proxy used by the verification engine
with real measurements taken at the locations where folding actually occurs.
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


class StabilityProfiler:
    """Global layer-aware stability profiler.

    The profiler is designed as a singleton-like object that can be disabled or
    re-enabled at runtime.  When enabled, every folded layer records its
    stability decision, and consumers (e.g. the verification engine) can read
    the resulting profile for the most recent forward pass.

    Args:
        enabled: Whether to collect statistics.  Disabling removes all overhead.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._profiles: dict[Any, StabilityProfile] = {}
        self._history: dict[tuple[int, int], list[float]] = defaultdict(list)
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
        """Record a single layer's stability decision.

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

        num_tokens = int(stable_mask.numel())
        stable_ratio = float(stable_mask.float().mean().item())

        # Record positions of divergent tokens for debugging/visualisation.
        divergence_positions = None
        if stable_ratio < 1.0:
            divergence_positions = torch.nonzero(~stable_mask, as_tuple=False)

        stats = LayerStabilityStats(
            layer_idx=layer_idx,
            step_idx=step_idx,
            stable_ratio=stable_ratio,
            tau_used=tau,
            metric=metric,
            num_tokens=num_tokens,
            divergence_positions=divergence_positions,
        )

        if branch_id not in self._profiles:
            self._profiles[branch_id] = StabilityProfile(
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
            )
        self._profiles[branch_id].layer_stats.append(stats)

        history_key = (layer_idx, step_idx)
        self._history[history_key].append(stable_ratio)
        if len(self._history[history_key]) > self._max_history_len:
            self._history[history_key].pop(0)

    def get_profile(self, branch_id: Any) -> Optional[StabilityProfile]:
        """Return the stability profile for ``branch_id`` if one exists."""
        return self._profiles.get(branch_id)

    def get_mean_stable_ratio(
        self,
        layer_idx: int,
        step_idx: int = 0,
    ) -> Optional[float]:
        """Return the historical mean stable ratio for a layer/step pair."""
        history = self._history.get((layer_idx, step_idx))
        if not history:
            return None
        return sum(history) / len(history)

    def reset_branch(self, branch_id: Any) -> None:
        """Drop the profile for a single branch."""
        self._profiles.pop(branch_id, None)

    def reset(self) -> None:
        """Drop all collected profiles and history."""
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
