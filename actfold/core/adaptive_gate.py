"""Adaptive quantile gate for Branch Folding (optimization #3).

A fixed similarity threshold interacts badly with model- and layer-dependent
similarity scales: the same tau can mean "reuse 97%" on one model and "reuse
64%" on another (see the deep experiment report).  :class:`AdaptiveQuantileGate`
instead selects the ``target_stable_ratio`` most similar tokens at every layer
via a top-k selection, so the reuse budget is spent where the hidden states are
most stable and the divergence is spread evenly across layers.
"""

from __future__ import annotations

import torch

from actfold.core.similarity_gate import SimilarityGate


class AdaptiveQuantileGate(SimilarityGate):
    """Similarity gate that targets a fixed stable ratio per call.

    Args:
        target_stable_ratio: Fraction of tokens to classify as stable in every
            call (0 < ratio <= 1).
        metric: Similarity metric ("cosine", "l2", "pearson").
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        target_stable_ratio: float = 0.97,
        metric: str = "cosine",
        eps: float = 1e-8,
    ) -> None:
        super().__init__(tau=0.95, metric=metric, eps=eps)
        if not 0.0 < target_stable_ratio <= 1.0:
            raise ValueError(f"target_stable_ratio must be in (0, 1], got {target_stable_ratio}")
        self.target_stable_ratio = float(target_stable_ratio)
        # Deferred tau bookkeeping (T011): forward stores only a device-side
        # candidate tensor; the host float is materialized lazily on read.
        self._tau_candidate: torch.Tensor | None = None
        self._last_tau_cache: float | None = None

    @property
    def last_tau(self) -> float:
        """Similarity of the least-stable stable token (lazy readback).

        Materialized once per forward from the stored device-side candidate;
        repeated reads do not touch the device again.
        """
        if self._last_tau_cache is not None:
            return self._last_tau_cache
        if self._tau_candidate is None:
            return 0.95
        self._last_tau_cache = float(self._tau_candidate.item())
        self._tau_candidate = None
        return self._last_tau_cache

    def forward(self, h_child: torch.Tensor, h_parent: torch.Tensor) -> torch.Tensor:
        """Return a stability mask marking the most-similar tokens as stable.

        The mask is computed by selecting the **divergent** candidates with a
        bottom-k pass (``k_div = N - k_stable``) instead of a top-k pass over
        nearly the whole tensor: for ratios close to 1 the divergent set is
        tiny, making the selection far cheaper. The result is identical to
        selecting the k_stable most similar tokens.
        """
        if h_child.ndim != 3 or h_parent.ndim != 3:
            raise ValueError(
                f"AdaptiveQuantileGate expects 3-D inputs [B, T, H], got "
                f"h_child {h_child.shape} and h_parent {h_parent.shape}"
            )
        if h_child.shape != h_parent.shape:
            raise ValueError(
                f"Shape mismatch: h_child {h_child.shape} vs h_parent {h_parent.shape}"
            )
        h_parent = h_parent.to(dtype=h_child.dtype, device=h_child.device)
        sim = self._compute_similarity(h_child, h_parent)
        flat = sim.reshape(-1)
        num_tokens = flat.numel()
        if num_tokens == 0:
            return torch.ones_like(sim, dtype=torch.bool)
        k_stable = int(round(self.target_stable_ratio * num_tokens))
        k_stable = min(num_tokens, max(1, k_stable))
        k_div = num_tokens - k_stable

        # Bottom-k over similarity = the divergent candidates. One extra
        # element gives the boundary value (least-stable stable token).
        boundary_vals, div_indices = torch.topk(flat, k_div + 1, largest=False)
        self._tau_candidate = boundary_vals[-1]
        self._last_tau_cache = None

        mask = torch.ones_like(flat, dtype=torch.bool)
        if k_div > 0:
            mask[div_indices[:k_div]] = False
        return mask.reshape(sim.shape)

    def set_target_stable_ratio(self, ratio: float) -> None:
        """Update the target stable ratio at runtime."""
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"target_stable_ratio must be in (0, 1], got {ratio}")
        self.target_stable_ratio = float(ratio)

    def extra_repr(self) -> str:
        return (
            f"target_stable_ratio={self.target_stable_ratio}, "
            f"metric={self.metric}, last_tau={self.last_tau:.4f}"
        )