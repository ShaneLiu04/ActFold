"""Candidate branch generation for speculative decoding."""

from __future__ import annotations

import torch

from actfold.speculative.branch import Branch


class DraftGenerator:
    """Generate multiple candidate child branches from a parent branch.

    For research purposes this generator can operate in three modes:
    - ``random``: sample tokens uniformly from the vocabulary.
    - ``perturb``: sample random tokens to append (intended for research only;
      the previous embedding-space perturbation mode is deprecated because raw
      token IDs are discrete, not continuous embeddings).
    - ``copy_flip``: copy the parent tokens, randomly flip a small fraction,
      and optionally append new random tokens.

    Args:
        vocab_size: Size of the token vocabulary.
        mode: Drafting mode ("random", "perturb", "copy_flip").
        perturbation_std: Deprecated; kept for API compatibility.
        flip_ratio: Fraction of tokens to flip in "copy_flip" mode.
    """

    def __init__(
        self,
        vocab_size: int,
        mode: str = "random",
        perturbation_std: float = 0.1,
        flip_ratio: float = 0.1,
    ) -> None:
        if mode not in {"random", "perturb", "copy_flip"}:
            raise ValueError(f"Unsupported draft mode: {mode}")
        self.vocab_size = vocab_size
        self.mode = mode
        self.perturbation_std = perturbation_std
        self.flip_ratio = flip_ratio
        self._counter = 0

    def generate(
        self,
        parent: Branch,
        num_branches: int = 2,
        seed: int | None = None,
        max_new_tokens: int = 0,
    ) -> list[Branch]:
        """Generate ``num_branches`` child branches.

        Args:
            parent: Parent branch.
            num_branches: Number of children to generate.
            seed: Optional random seed.
            max_new_tokens: Number of new tokens to append to each child. When
                zero (the default), children have the same sequence length as
                the parent.

        Returns:
            List of child Branch instances.
        """
        if seed is not None:
            torch.manual_seed(seed)
            # Reset the deterministic counter so repeated calls with the same
            # seed produce the same branch IDs.
            self._counter = 0

        children: list[Branch] = []
        batch_size, seq_len = parent.tokens.shape

        for idx in range(num_branches):
            child_id = f"{parent.branch_id}:child_{self._counter}"
            self._counter += 1

            if self.mode == "random":
                child_tokens = torch.randint(
                    low=0,
                    high=self.vocab_size,
                    size=(batch_size, seq_len + max_new_tokens),
                    dtype=parent.tokens.dtype,
                    device=parent.tokens.device,
                )
            elif self.mode == "perturb":
                # Research stand-in: sample random tokens; real perturbation in
                # embedding space requires access to the model's embeddings.
                child_tokens = torch.randint(
                    low=0,
                    high=self.vocab_size,
                    size=(batch_size, seq_len + max_new_tokens),
                    dtype=parent.tokens.dtype,
                    device=parent.tokens.device,
                )
            else:  # copy_flip
                child_tokens = parent.tokens.clone()
                if self.flip_ratio > 0 and seq_len > 0:
                    num_flips = max(1, int(seq_len * self.flip_ratio))
                    for b in range(batch_size):
                        flip_positions = torch.randperm(seq_len)[:num_flips]
                        new_tokens = torch.randint(
                            0,
                            self.vocab_size,
                            (num_flips,),
                            dtype=child_tokens.dtype,
                            device=child_tokens.device,
                        )
                        child_tokens[b, flip_positions] = new_tokens
                if max_new_tokens > 0:
                    appended = torch.randint(
                        0,
                        self.vocab_size,
                        (batch_size, max_new_tokens),
                        dtype=child_tokens.dtype,
                        device=child_tokens.device,
                    )
                    child_tokens = torch.cat([child_tokens, appended], dim=1)

            child = Branch(
                branch_id=child_id,
                parent_id=parent.branch_id,
                tokens=child_tokens,
            )
            children.append(child)

        return children
