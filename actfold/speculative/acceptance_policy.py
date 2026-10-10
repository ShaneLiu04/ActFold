"""Acceptance policies for folded generation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from actfold.speculative.branch_tree import BranchNode


class AcceptancePolicy(ABC):
    """Decide which candidate branch to accept at each generation step."""

    @abstractmethod
    def select(
        self,
        candidates: list[BranchNode],
        logits: torch.Tensor | None = None,
    ) -> BranchNode:
        """Return the accepted branch from ``candidates``."""
        ...


class GreedyAcceptancePolicy(AcceptancePolicy):
    """Always accept the candidate with the highest last-token logit."""

    def select(
        self,
        candidates: list[BranchNode],
        logits: torch.Tensor | None = None,
    ) -> BranchNode:
        """Pick the candidate whose last token has the largest logit."""
        if len(candidates) == 1:
            return candidates[0]

        best = candidates[0]
        best_score = float("-inf")
        for node in candidates:
            if node.logits is None:
                continue
            last_logits = node.logits[:, -1, :]  # [batch, vocab]
            score = last_logits.max(dim=-1).values.mean().item()
            if score > best_score:
                best_score = score
                best = node
        return best


class ThresholdAcceptancePolicy(AcceptancePolicy):
    """Accept candidates above a stable-ratio threshold, fall back to greedy."""

    def __init__(self, threshold: float = 0.0) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.greedy = GreedyAcceptancePolicy()

    def select(
        self,
        candidates: list[BranchNode],
        logits: torch.Tensor | None = None,
    ) -> BranchNode:
        """Accept the highest-scoring candidate whose metadata passes threshold."""
        eligible = [
            n
            for n in candidates
            if n.logits is not None and n.metadata.get("stable_ratio", 1.0) >= self.threshold
        ]
        if not eligible:
            eligible = candidates
        return self.greedy.select(eligible, logits)


class TargetMatchAcceptancePolicy(AcceptancePolicy):
    """Select the candidate whose appended token matches the target argmax.

    True speculative-decoding semantics (AR004): each candidate's appended
    token (its last position) is accepted iff it equals the target argmax at
    that same position of the candidate's own forward logits.  The candidate
    with the highest acceptance rate wins; ties resolve to the first
    candidate in list order.  Candidates without logits are skipped; if all
    candidates lack logits, the first candidate is returned.
    """

    def _rate(self, node: BranchNode) -> float:
        """Batch-mean acceptance of the candidate's appended token."""
        assert node.logits is not None
        last_logits = node.logits[:, -1, :]  # [batch, vocab]
        predicted = last_logits.argmax(dim=-1)  # [batch]
        return float((predicted == node.tokens[:, -1]).float().mean().item())

    def select(
        self,
        candidates: list[BranchNode],
        logits: torch.Tensor | None = None,
    ) -> BranchNode:
        """Return the candidate with the highest appended-token acceptance.

        Args:
            candidates: Candidate branch nodes (each with forward logits).
            logits: Unused; kept for interface symmetry.

        Returns:
            The accepted branch.
        """
        best: BranchNode | None = None
        best_rate = float("-inf")
        for node in candidates:
            if node.logits is None:
                continue
            rate = self._rate(node)
            if rate > best_rate:
                best_rate = rate
                best = node
        if best is None:
            return candidates[0]
        return best
