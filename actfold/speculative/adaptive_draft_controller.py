"""Adaptive draft-growth controller for ActFold.

When no trained draft model is available, this controller uses runtime feedback
(stable ratio, recent acceptance rate, and branch depth) to decide how many
candidate branches to spawn at each generation step.  High-stability contexts
get more speculative branches; low-stability contexts fall back to a single
greedy extension.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from actfold.speculative.branch import Branch
from actfold.speculative.draft_generator import DraftGenerator


@dataclass
class DraftGrowthState:
    """Mutable state tracked by the adaptive controller."""

    acceptance_history: deque[float]
    recent_stable_ratio: float = 0.0


class AdaptiveDraftGrowthController:
    """Dynamically control the number of draft branches per generation step.

    Args:
        draft_generator: Base draft generator used to create candidates.
        max_branches: Maximum number of branches to request from the draft
            generator in a single step.
        min_stable_ratio_to_expand: Minimum recent stable ratio required to
            request more than one branch.
        acceptance_window: Number of recent accept/reject decisions to retain.
        min_acceptance_to_expand: Minimum recent acceptance rate required to
            expand beyond one branch.
    """

    def __init__(
        self,
        draft_generator: DraftGenerator,
        max_branches: int = 8,
        min_stable_ratio_to_expand: float = 0.7,
        acceptance_window: int = 10,
        min_acceptance_to_expand: float = 0.5,
    ) -> None:
        if max_branches < 1:
            raise ValueError(f"max_branches must be >= 1, got {max_branches}")
        if not 0.0 <= min_stable_ratio_to_expand <= 1.0:
            raise ValueError(
                f"min_stable_ratio_to_expand must be in [0, 1], got {min_stable_ratio_to_expand}"
            )
        if not 0.0 <= min_acceptance_to_expand <= 1.0:
            raise ValueError(
                f"min_acceptance_to_expand must be in [0, 1], got {min_acceptance_to_expand}"
            )

        self.draft_generator = draft_generator
        self.max_branches = max_branches
        self.min_stable_ratio_to_expand = min_stable_ratio_to_expand
        self.min_acceptance_to_expand = min_acceptance_to_expand
        self._state = DraftGrowthState(
            acceptance_history=deque(maxlen=acceptance_window),
        )

    def generate_children(
        self,
        parent: Branch,
        max_new_tokens: int = 1,
        seed: int = 0,
        recent_stable_ratio: float | None = None,
    ) -> list[Branch]:
        """Generate a dynamically-sized set of child branches.

        Args:
            parent: Parent branch to extend.
            max_new_tokens: Number of tokens each child may add.
            seed: Random seed for the draft generator.
            recent_stable_ratio: Optional stable ratio from the most recent
                folded forward pass.  If omitted, the controller's internal
                moving average is used.

        Returns:
            List of child branches (length >= 1).
        """
        if recent_stable_ratio is not None:
            self._state.recent_stable_ratio = float(recent_stable_ratio)

        num_branches = self._decide_num_branches()
        return self.draft_generator.generate(
            parent,
            num_branches=num_branches,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

    def _decide_num_branches(self) -> int:
        """Return the number of branches to request for the current step."""
        avg_acceptance = self._average_acceptance()
        stable_ratio = self._state.recent_stable_ratio

        if (
            stable_ratio >= self.min_stable_ratio_to_expand
            and avg_acceptance >= self.min_acceptance_to_expand
        ):
            # Scale requested branches with acceptance rate.
            scale = max(0.0, avg_acceptance)
            return max(1, min(self.max_branches, int(self.max_branches * scale) + 1))
        return 1

    def _average_acceptance(self) -> float:
        """Return the average acceptance rate over the recent window."""
        if not self._state.acceptance_history:
            return 0.5
        return sum(self._state.acceptance_history) / len(self._state.acceptance_history)

    def update_acceptance(self, accepted: bool) -> None:
        """Record the outcome of a verification/selection decision."""
        self._state.acceptance_history.append(1.0 if accepted else 0.0)

    def reset(self) -> None:
        """Clear all runtime state."""
        self._state = DraftGrowthState(
            acceptance_history=deque(maxlen=self._state.acceptance_history.maxlen or 10),
        )
