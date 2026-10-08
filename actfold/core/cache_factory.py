"""Factory for constructing ActFold activation caches.

This module centralises cache construction so that callers can switch between
:class:`~actfold.core.activation_cache.ActivationCache` and
:class:`~actfold.core.chunked_cache.ChunkedActivationCache` via configuration
without changing instantiation logic throughout the codebase.
"""

from __future__ import annotations

from typing import Union

from actfold.core.activation_cache import ActivationCache
from actfold.core.chunked_cache import ChunkedActivationCache
from actfold.core.vectorized_cache import VectorizedActivationCache

# Union type exported for type annotations.
ActivationCacheType = Union[ActivationCache, ChunkedActivationCache, VectorizedActivationCache]


def make_activation_cache(
    max_entries_per_layer: int = 1024,
    device: str = "cuda",
    use_chunked: bool = False,
    chunk_size: int = 64,
    use_vectorized: bool = False,
    max_branch_steps: int | None = 4,
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

    Returns:
        An initialised activation cache.
    """
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
