"""Vanilla multi-branch speculative decoding baseline (no activation reuse)."""

from __future__ import annotations

import torch

from actfold.speculative.branch import Branch
from actfold.speculative.draft_generator import DraftGenerator
from actfold.speculative.fast_dllm_adapter import DiffusionLLMAdapter


class SpiffyBaseline:
    """Vanilla multi-branch speculative decoding without activation reuse.

    This baseline generates ``num_branches`` candidate continuations of length
    ``max_new_tokens`` and verifies each with a full forward pass. The branch
    with the highest mean logit score is accepted.

    Args:
        model: Model adapter.
        draft_generator: Generator for candidate branches.
    """

    def __init__(
        self,
        model: DiffusionLLMAdapter,
        draft_generator: DraftGenerator,
    ) -> None:
        self.model = model
        self.draft_generator = draft_generator

    def generate(
        self,
        prompt: torch.Tensor,
        num_branches: int = 4,
        max_new_tokens: int = 16,
        seed: int | None = None,
    ) -> Branch:
        """Generate and verify candidate branches, returning the best one.

        Args:
            prompt: Prompt token tensor ``[batch, seq_len]``.
            num_branches: Number of candidate branches.
            max_new_tokens: Number of new tokens to draft per branch.
            seed: Optional random seed for the draft generator.

        Returns:
            The accepted Branch instance.
        """
        root = Branch(
            branch_id="root",
            parent_id=None,
            tokens=prompt,
        )
        branches = self.draft_generator.generate(
            parent=root,
            num_branches=num_branches,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        return self.verify(branches)

    def verify(self, branches: list[Branch]) -> Branch:
        """Verify each branch with a full forward pass and accept the best.

        Args:
            branches: Candidate branches.

        Returns:
            The accepted branch.
        """
        best_branch: Branch | None = None
        best_score = float("-inf")

        for branch in branches:
            logits = self.model.forward(branch.tokens)
            score = logits.float().mean().item()
            branch.metadata["baseline_score"] = score
            if score > best_score:
                best_score = score
                best_branch = branch

        if best_branch is None:
            raise RuntimeError("No branch was accepted.")

        best_branch.accepted = True
        return best_branch
