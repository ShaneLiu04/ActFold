"""Similarity analysis between parent and child branch hidden states."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class SimilarityAnalyzer:
    """Computes similarity matrices between parent-child branch hidden states."""

    SUPPORTED_METRICS = {"cosine", "l2", "pearson"}

    def compute_similarity(
        self,
        h_parent: torch.Tensor,
        h_child: torch.Tensor,
        metric: str = "cosine",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute token-level similarity scores.

        Args:
            h_parent: Parent hidden states.
                Expected shape: ``[layers, tokens, steps, hidden_dim]`` or
                ``[batch, seq_len, hidden_dim]``.
            h_child: Child hidden states, same shape as ``h_parent``.
            metric: Similarity metric ("cosine", "l2", "pearson").
            eps: Numerical stability constant.

        Returns:
            Similarity tensor with one fewer dimension (the hidden dimension is
            reduced). For 4D inputs, returns ``[layers, tokens, steps]``.

        Raises:
            ValueError: If metric is unsupported or shapes mismatch.
        """
        if h_parent.shape != h_child.shape:
            raise ValueError(f"Shape mismatch: {h_parent.shape} vs {h_child.shape}")
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric: {metric}")

        if metric == "cosine":
            return F.cosine_similarity(h_parent, h_child, dim=-1, eps=eps)

        if metric == "l2":
            l2 = torch.norm(h_parent - h_child, p=2, dim=-1)
            denom = (h_parent.size(-1) ** 0.5) + eps
            similarity: torch.Tensor = 1.0 - l2 / denom
            return similarity

        # Pearson correlation across hidden dimension.
        parent_centered = h_parent - h_parent.mean(dim=-1, keepdim=True)
        child_centered = h_child - h_child.mean(dim=-1, keepdim=True)
        numerator: torch.Tensor = (parent_centered * child_centered).sum(dim=-1)
        denominator: torch.Tensor = (
            torch.norm(parent_centered, p=2, dim=-1) * torch.norm(child_centered, p=2, dim=-1) + eps
        )
        return numerator / denominator

    def generate_report(
        self,
        sim_matrix: torch.Tensor,
        tau: float = 0.95,
    ) -> dict[str, Any]:
        """Return statistics from a similarity matrix.

        Args:
            sim_matrix: Similarity tensor of any shape.
            tau: Stability threshold.

        Returns:
            Dictionary with keys:
            - ``mean``: mean similarity
            - ``std``: standard deviation
            - ``pct_stable``: percentage of positions above ``tau``
            - ``layer_mean``: per-layer mean (if at least 3D)
            - ``hotspots``: indices of positions with similarity below ``tau``
        """
        stable_mask = sim_matrix > tau
        report: dict[str, Any] = {
            "mean": sim_matrix.mean().item(),
            "std": sim_matrix.std().item(),
            "pct_stable": stable_mask.float().mean().item() * 100.0,
            "tau": tau,
        }

        if sim_matrix.dim() >= 3:
            report["layer_mean"] = sim_matrix.mean(dim=tuple(range(1, sim_matrix.dim()))).tolist()
            report["layer_pct_stable"] = (
                stable_mask.float().mean(dim=tuple(range(1, sim_matrix.dim()))).tolist()
            )

        # Find a few divergence hotspots.
        divergent_positions = (~stable_mask).nonzero(as_tuple=False)
        report["num_divergent"] = divergent_positions.size(0)
        report["hotspots"] = divergent_positions[:20].tolist()

        return report
