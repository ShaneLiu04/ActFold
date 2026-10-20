"""Factory for constructing ActFold activation caches.

This module centralises cache construction so that callers can switch between
:class:`~actfold.core.activation_cache.ActivationCache`,
:class:`~actfold.core.chunked_cache.ChunkedActivationCache`,
:class:`~actfold.core.vectorized_cache.VectorizedActivationCache` and the
bounded-memory :class:`~actfold.core.pyramid_cache.PyramidActivationCache`
(AR029) via configuration without changing instantiation logic elsewhere.
"""

from __future__ import annotations

from typing import Union

from actfold.core.activation_cache import ActivationCache
from actfold.core.chunked_cache import ChunkedActivationCache
from actfold.core.pyramid_cache import PyramidActivationCache
from actfold.core.pyramid_plan import PyramidPlan, build_pyramid_plan
from actfold.core.vectorized_cache import VectorizedActivationCache

# Union type exported for type annotations.
ActivationCacheType = Union[
    ActivationCache,
    ChunkedActivationCache,
    VectorizedActivationCache,
    PyramidActivationCache,
]


def make_activation_cache(
    max_entries_per_layer: int = 1024,
    device: str = "cuda",
    use_chunked: bool = False,
    chunk_size: int = 64,
    use_vectorized: bool = False,
    max_branch_steps: int | None = 4,
    use_pyramid: bool = False,
    pyramid_plan: PyramidPlan | None = None,
    pyramid_depth: int | None = None,
    pyramid_total_tokens: int = 4096,
    pyramid_block_size: int = 16,
    pyramid_max_ratio: float = 1.0,
    pyramid_min_ratio: float = 0.25,
    pyramid_schedule: str = "linear",
    pyramid_recent_blocks: int = 2,
    pyramid_anchor_blocks: int = 1,
    pyramid_anchor_stride: int = 2,
) -> ActivationCacheType:
    """Create an activation cache.

    Args:
        max_entries_per_layer: Maximum number of token-level entries cached per
            layer.  For the chunked cache this is translated into a chunk budget.
        device: Target device string.
        use_chunked: If ``True``, return a :class:`ChunkedActivationCache`.
        chunk_size: Number of tokens stored in each chunk when using the chunked
            cache.
        use_vectorized: If ``True``, return a
            :class:`~actfold.core.vectorized_cache.VectorizedActivationCache`
            whose contiguous buffers make cache read/write orders of magnitude
            cheaper. Takes precedence over ``use_chunked``.
        max_branch_steps: Maximum number of distinct ``(branch, step)`` keys the
            vectorized cache retains (ring eviction); only used with
            ``use_vectorized``. ``0``/``None`` disables the bound.
        use_pyramid: If ``True``, return a :class:`PyramidActivationCache`.
            Takes precedence over every other switch (the budget bound must
            never be silently downgraded).
        pyramid_plan: Explicit frozen plan; ignores the other ``pyramid_*``
            knobs.
        pyramid_depth: Real model layer count (required when no plan is given).
        pyramid_total_tokens: Plan-wide token budget across all layers.
        pyramid_block_size: Tokens per cache page.
        pyramid_max_ratio / pyramid_min_ratio: Per-layer budget ratio bounds.
        pyramid_schedule: ``"linear"`` or ``"geometric"``.
        pyramid_recent_blocks: Blocks protected for the just-written entry.
        pyramid_anchor_blocks: Leading anchor blocks per entry.
        pyramid_anchor_stride: Additional every-N-th block anchor.

    Returns:
        An initialised activation cache.

    Raises:
        ValueError: When ``use_pyramid`` is set without a plan or a positive
            depth, or when the plan parameters are invalid.
    """
    if use_pyramid:
        plan = pyramid_plan
        if plan is None:
            if pyramid_depth is None or pyramid_depth <= 0:
                raise ValueError(
                    "use_pyramid requires an explicit pyramid_plan or a positive "
                    f"pyramid_depth, got {pyramid_depth!r}"
                )
            plan = build_pyramid_plan(
                depth=pyramid_depth,
                total_tokens=pyramid_total_tokens,
                block_size=pyramid_block_size,
                max_ratio=pyramid_max_ratio,
                min_ratio=pyramid_min_ratio,
                schedule=pyramid_schedule,
                recent_blocks=pyramid_recent_blocks,
                anchor_blocks=pyramid_anchor_blocks,
                anchor_stride=pyramid_anchor_stride,
            )
        return PyramidActivationCache(plan=plan, device=device)
    if use_vectorized:
        return VectorizedActivationCache(
            max_entries_per_layer=max_entries_per_layer,
            device=device,
            max_branch_steps=max_branch_steps,
        )
    if use_chunked:
        return ChunkedActivationCache(
            max_entries_per_layer=max_entries_per_layer,
            chunk_size=chunk_size,
            device=device,
        )
    return ActivationCache(
        max_entries_per_layer=max_entries_per_layer,
        device=device,
    )
