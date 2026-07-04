"""Lightweight Branch dataclass for speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Branch:
    """A candidate generation trajectory.

    This is the speculative-decoding-level branch abstraction, lighter than
    ``actfold.core.branch_manager.Branch``. It tracks tokens, scores, and
    acceptance status without caching full hidden-state trees.

    Attributes:
        branch_id: Unique branch identifier.
        parent_id: Parent branch identifier, or None for root.
        tokens: Token tensor ``[batch, seq_len]``.
        scores: Optional log-prob-like scores ``[batch, seq_len]``.
        accepted: Whether the branch passed verification.
        metadata: Arbitrary metadata dictionary.
    """

    branch_id: str
    parent_id: str | None
    tokens: torch.Tensor
    scores: torch.Tensor | None = None
    accepted: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokens.dim() != 2:
            raise ValueError(f"tokens must be 2D [batch, seq_len], got {self.tokens.shape}")
        if self.scores is not None and self.scores.shape != self.tokens.shape:
            raise ValueError(
                f"scores shape {self.scores.shape} must match tokens shape {self.tokens.shape}"
            )
