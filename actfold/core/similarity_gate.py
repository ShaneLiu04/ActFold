"""Token-level similarity gating for Branch Folding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityGate(nn.Module):
    """Compute a boolean stability mask from parent and child hidden states.

    A token is considered **stable** (True) when its similarity score exceeds
    the threshold ``tau``. Stable tokens may reuse cached parent activations.

    Args:
        tau: Similarity threshold in [0, 1].
        metric: Similarity metric ("cosine", "l2", "pearson").
        eps: Numerical stability constant.

    Note:
        The "l2" metric returns ``1 - L2_distance / sqrt(hidden_dim)``. Because
        distances can exceed ``sqrt(hidden_dim)``, L2 scores can be strongly
        negative; this makes the gate conservative for that metric.
    """

    def __init__(
        self,
        tau: float = 0.95,
        metric: str = "cosine",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        if metric not in {"cosine", "l2", "pearson"}:
            raise ValueError(f"Unsupported metric: {metric}")

        self.tau = tau
        self.metric = metric
        self.eps = eps

    def forward(
        self,
        h_child: torch.Tensor,
        h_parent: torch.Tensor,
    ) -> torch.Tensor:
        """Return a stability mask.

        Args:
            h_child: Child hidden states, shape ``[batch, seq_len, hidden_dim]``.
            h_parent: Parent hidden states, same shape.

        Returns:
            Boolean tensor of shape ``[batch, seq_len]``. ``True`` means stable.

        Raises:
            ValueError: If shapes do not match or inputs are not 3-D.
        """
        if h_child.ndim != 3 or h_parent.ndim != 3:
            raise ValueError(
                f"SimilarityGate expects 3-D inputs [B, T, H], got "
                f"h_child {h_child.shape} and h_parent {h_parent.shape}"
            )
        if h_child.shape != h_parent.shape:
            raise ValueError(
                f"Shape mismatch: h_child {h_child.shape} vs h_parent {h_parent.shape}"
            )

        # Align parent to child's device/dtype to avoid runtime device errors.
        h_parent = h_parent.to(dtype=h_child.dtype, device=h_child.device)

        sim = self._compute_similarity(h_child, h_parent)
        return sim > self.tau

    def _compute_similarity(
        self,
        h_child: torch.Tensor,
        h_parent: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token similarity scores."""
        if self.metric == "cosine":
            return F.cosine_similarity(h_child, h_parent, dim=-1, eps=self.eps)

        if self.metric == "l2":
            # Negative L2 distance, normalized by hidden dim for threshold compatibility.
            l2 = torch.norm(h_child - h_parent, p=2, dim=-1)
            denom = (h_child.size(-1) ** 0.5) + self.eps
            similarity: torch.Tensor = 1.0 - l2 / denom
            return similarity

        if self.metric == "pearson":
            # Pearson correlation per token across hidden dimension.
            child_centered = h_child - h_child.mean(dim=-1, keepdim=True)
            parent_centered = h_parent - h_parent.mean(dim=-1, keepdim=True)
            numerator: torch.Tensor = (child_centered * parent_centered).sum(dim=-1)
            denominator: torch.Tensor = (
                torch.norm(child_centered, p=2, dim=-1) * torch.norm(parent_centered, p=2, dim=-1)
                + self.eps
            )
            return numerator / denominator

        raise RuntimeError(f"Unreachable metric: {self.metric}")

    def set_tau(self, tau: float) -> None:
        """Update the similarity threshold at runtime."""
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        self.tau = tau

    def extra_repr(self) -> str:
        return f"tau={self.tau}, metric={self.metric}, eps={self.eps}"
