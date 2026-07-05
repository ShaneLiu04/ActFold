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
