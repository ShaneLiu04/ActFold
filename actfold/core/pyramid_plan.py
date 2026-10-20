"""Pyramid budget planning for the edge activation cache (AR029).

The plan freezes the per-layer token budgets and the deterministic anchor layout
before any model is loaded.  Runtime state (recent windows, actual evictions)
lives in :class:`~actfold.core.pyramid_cache.PyramidActivationCache`, never in
the plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_SCHEDULES = ("linear", "geometric")


@dataclass(frozen=True)
class PyramidPlan:
    """Immutable per-layer retention plan.

    Args:
        depth: Number of transformer layers.
        block_size: Tokens per cache page (block).
        layer_budgets: Per-layer token budgets (block-aligned).
        recent_blocks: Blocks protected for the just-written entry.
        anchor_blocks: Leading blocks protected in every entry.
        anchor_stride: Every ``anchor_stride``-th block is an anchor.
        schedule: ``"linear"`` or ``"geometric"`` (informational).
        requested_total: The plan-wide budget requested at build time.
    """

    depth: int
    block_size: int
    layer_budgets: tuple[int, ...]
    recent_blocks: int
    anchor_blocks: int
    anchor_stride: int
    schedule: str
    requested_total: int

    def budget_tokens(self, layer: int) -> int:
        """Return the token budget of ``layer``.

        Raises:
            IndexError: If ``layer`` is outside ``[0, depth)``.
        """
        if layer < 0 or layer >= self.depth:
            raise IndexError(f"layer {layer} out of range for depth {self.depth}")
        return self.layer_budgets[layer]

    def budget_blocks(self, layer: int) -> int:
        """Return the block budget of ``layer``."""
        return max(1, self.budget_tokens(layer) // self.block_size)

    @property
    def total_budget_tokens(self) -> int:
        """Plan-wide token upper bound (after reclaim)."""
        return sum(self.layer_budgets)

    def retained_blocks(self, layer: int, seq_len: int) -> frozenset[int]:
        """Return the deterministic anchor block indices for ``layer``.

        The recent window is runtime state and is deliberately not included.
        """
        self.budget_tokens(layer)  # bounds check
        if seq_len <= 0:
            return frozenset()
        n_blocks = math.ceil(seq_len / self.block_size)
        anchors = set(range(min(self.anchor_blocks, n_blocks)))
        anchors.update(range(0, n_blocks, self.anchor_stride))
        return frozenset(idx for idx in anchors if 0 <= idx < n_blocks)


def _weights(depth: int, max_ratio: float, min_ratio: float, schedule: str) -> list[float]:
    if depth == 1:
        return [float(max_ratio)]
    if schedule == "linear":
        step = (max_ratio - min_ratio) / (depth - 1)
        return [float(max_ratio - step * idx) for idx in range(depth)]
    # geometric: max_ratio * (min_ratio / max_ratio) ** (idx / (depth - 1))
    ratio = float(min_ratio / max_ratio)
    return [float(max_ratio * (ratio ** (idx / (depth - 1)))) for idx in range(depth)]


def build_pyramid_plan(
    depth: int,
    total_tokens: int,
    *,
    block_size: int = 16,
    max_ratio: float = 1.0,
    min_ratio: float = 0.25,
    schedule: str = "linear",
    recent_blocks: int = 2,
    anchor_blocks: int = 1,
    anchor_stride: int = 2,
) -> PyramidPlan:
    """Build an immutable per-layer budget plan.

    ``total_tokens`` is the **plan-wide** budget (sum over all layers).  Weights
    are normalized so the sum never exceeds it, even after block flooring.

    Raises:
        ValueError: On any invalid parameter combination.
    """
    if depth <= 0:
        raise ValueError(f"depth must be > 0, got {depth}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if total_tokens < block_size:
        raise ValueError(f"total_tokens ({total_tokens}) must be >= block_size ({block_size})")
    if depth * block_size > total_tokens:
        raise ValueError(
            "total_tokens must provide at least one block per layer: "
            f"depth*block_size={depth * block_size} > total_tokens={total_tokens}"
        )
    if not (0.0 < min_ratio <= max_ratio <= 1.0):
        raise ValueError(
            "ratios must satisfy 0 < min_ratio <= max_ratio <= 1.0, got "
            f"min_ratio={min_ratio}, max_ratio={max_ratio}"
        )
    if schedule not in _SCHEDULES:
        raise ValueError(f"schedule must be one of {_SCHEDULES}, got {schedule!r}")
    if recent_blocks < 0:
        raise ValueError(f"recent_blocks must be >= 0, got {recent_blocks}")
    if anchor_blocks < 0:
        raise ValueError(f"anchor_blocks must be >= 0, got {anchor_blocks}")
    if anchor_stride < 1:
        raise ValueError(f"anchor_stride must be >= 1, got {anchor_stride}")

    if depth == 1:
        budget = max(block_size, (total_tokens // block_size) * block_size)
        return PyramidPlan(
            depth=depth,
            block_size=block_size,
            layer_budgets=(budget,),
            recent_blocks=recent_blocks,
            anchor_blocks=anchor_blocks,
            anchor_stride=anchor_stride,
            schedule=schedule,
            requested_total=total_tokens,
        )

    weights = _weights(depth, max_ratio, min_ratio, schedule)
    weight_sum = sum(weights)
    budgets = [
        max(block_size, int(total_tokens * w / weight_sum) // block_size * block_size)
        for w in weights
    ]
    # Hard plan-wide bound: reclaim block-by-block from the deepest layer.
    while sum(budgets) > total_tokens:
        progressed = False
        for idx in range(depth - 1, -1, -1):
            if sum(budgets) <= total_tokens:
                break
            if budgets[idx] > block_size:
                budgets[idx] -= block_size
                progressed = True
        if not progressed:
            break
    return PyramidPlan(
        depth=depth,
        block_size=block_size,
        layer_budgets=tuple(budgets),
        recent_blocks=recent_blocks,
        anchor_blocks=anchor_blocks,
        anchor_stride=anchor_stride,
        schedule=schedule,
        requested_total=total_tokens,
    )
