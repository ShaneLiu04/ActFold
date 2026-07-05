"""Chunked tensor activation cache for ActFold.

This module provides :class:`ChunkedActivationCache`, a drop-in replacement for
:class:`~actfold.core.activation_cache.ActivationCache` that stores activations
in contiguous tensor chunks rather than per-token dictionary entries.  The
chunked layout dramatically reduces Python dict overhead and enables vectorised
gather/scatter operations for long sequences.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class ActivationChunk:
    """A contiguous block of activations covering ``chunk_size`` token slots."""

    chunk_id: int
    data: dict[str, torch.Tensor]
    start_token: int
    num_valid: int


class ChunkedActivationCache:
    """LRU activation cache backed by contiguous tensor chunks.

    The public API matches :class:`~actfold.core.activation_cache.ActivationCache`
    so that either implementation can be used without changing consumers.

    Args:
        max_entries_per_layer: Maximum number of **tokens** cached per layer.
            Internally this is translated to a chunk budget.
        chunk_size: Number of tokens stored in each contiguous chunk.
        device: Target device string (informational).
    """

    def __init__(
        self,
        max_entries_per_layer: int = 1024,
        chunk_size: int = 64,
        device: str = "cuda",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if max_entries_per_layer <= 0:
            raise ValueError(f"max_entries_per_layer must be positive, got {max_entries_per_layer}")

        self.chunk_size = chunk_size
        self.max_entries = max_entries_per_layer
        self.max_chunks = max(1, max_entries_per_layer // chunk_size)
        self.device = device
        # OrderedDict key: (branch_id, layer_idx, chunk_idx, step_idx)
        self._caches: dict[int, OrderedDict[tuple[str, int, int, int], ActivationChunk]] = {}

    def _ensure_layer(self, layer_idx: int) -> None:
        """Create an empty OrderedDict for a layer if it does not exist."""
        if layer_idx not in self._caches:
            self._caches[layer_idx] = OrderedDict()

    @staticmethod
    def _key(
        branch_id: str,
        layer_idx: int,
        chunk_idx: int,
        step_idx: int,
    ) -> tuple[str, int, int, int]:
        """Build a cache key."""
        return (branch_id, layer_idx, chunk_idx, step_idx)

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        """Store activations in contiguous chunks.

        Args:
            branch_id: Unique branch identifier.
            layer_idx: Layer index.
            activations: Mapping from activation name to tensor of shape
                ``[batch, seq_len, ...]``.
            step_idx: Diffusion step index.

        Raises:
            ValueError: If ``activations`` is empty or shapes are inconsistent.
        """
        if not activations:
            raise ValueError("activations must not be empty")

        self._ensure_layer(layer_idx)
        cache = self._caches[layer_idx]

        expected_shape: tuple[int, ...] | None = None
        for name, tensor in activations.items():
            if tensor.ndim < 2:
                raise ValueError(
                    f"Activation '{name}' must have at least 2 leading dimensions "
                    f"[batch, seq_len, ...], got shape {tensor.shape}"
                )
            leading = tensor.shape[:2]
            if expected_shape is None:
                expected_shape = leading
            elif leading != expected_shape:
                raise ValueError(
                    f"Activation '{name}' has inconsistent leading shape {leading}; "
                    f"expected {expected_shape}"
                )

        assert expected_shape is not None
        batch_size, seq_len = expected_shape
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            chunk_data = {
                name: tensor[:, start:end, ...].contiguous() for name, tensor in activations.items()
            }
            key = self._key(branch_id, layer_idx, chunk_idx, step_idx)
            chunk = ActivationChunk(
                chunk_id=chunk_idx,
                data=chunk_data,
                start_token=start,
                num_valid=end - start,
            )

            if key in cache:
                cache.move_to_end(key)
            cache[key] = chunk

            if len(cache) > self.max_chunks:
                cache.popitem(last=False)

    def get(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Retrieve activations for tokens marked ``True`` in ``token_mask``.

        Missing positions are zero-filled.  The implementation first concatenates
        all available chunks for the requested branch/layer/step into a dense
        tensor, then indexes with the token mask.

        Args:
            branch_id: Branch identifier to load from.
            layer_idx: Layer index.
            token_mask: Boolean tensor ``[batch, seq_len]``.
            step_idx: Diffusion step index.

        Returns:
            Dictionary of activations with the same leading shape as the stored
            tensors and missing/unstable positions zero-filled.
        """
        self._ensure_layer(layer_idx)
        cache = self._caches[layer_idx]
        batch_size, seq_len = token_mask.shape

        # Determine how many chunks we need.
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        chunks: list[ActivationChunk] = []
        for chunk_idx in range(num_chunks):
            key = self._key(branch_id, layer_idx, chunk_idx, step_idx)
            if key in cache:
                chunks.append(cache[key])
                cache.move_to_end(key)

        if not chunks:
            # No cached chunks at all: return zero-filled tensors.
            return self._zero_fill(token_mask)

        return self._gather_from_chunks(chunks, token_mask, seq_len)

    def _gather_from_chunks(
        self,
        chunks: list[ActivationChunk],
        token_mask: torch.Tensor,
        seq_len: int,
    ) -> dict[str, torch.Tensor]:
        """Concatenate chunks and gather selected tokens."""
        # Sort chunks by start_token to obtain a consistent dense view.
        chunks = sorted(chunks, key=lambda c: c.start_token)
        first_chunk = chunks[0]
        batch_size = token_mask.shape[0]
        device = token_mask.device

        output: dict[str, torch.Tensor] = {}
        for name in first_chunk.data:
            ref_shape = first_chunk.data[name].shape
            trailing = ref_shape[2:]

            # Build a dense tensor covering [0, seq_len) for this activation.
            dense = torch.zeros(
                (batch_size, seq_len, *trailing),
                dtype=first_chunk.data[name].dtype,
                device=device,
            )
            for chunk in chunks:
                start = chunk.start_token
                end = start + chunk.num_valid
                dense[:, start:end] = chunk.data[name].to(device=device)

            # Gather selected positions.  Following the original ActivationCache
            # semantics, a token position is selected if it is marked True in any
            # batch element, and all batches receive the same set of positions.
            valid_indices = torch.nonzero(token_mask.any(dim=0), as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                output[name] = torch.zeros(
                    (batch_size, 0, *trailing),
                    dtype=dense.dtype,
                    device=device,
                )
            else:
                gathered = dense[:, valid_indices, ...]
                output[name] = gathered

        return output

    def _zero_fill(self, token_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return an empty activation dict when nothing is cached."""
        return {}

    def clear_branch(self, branch_id: str) -> None:
        """Evict all entries belonging to ``branch_id``."""
        for cache in self._caches.values():
            keys_to_remove = [key for key in cache if key[0] == branch_id]
            for key in keys_to_remove:
                del cache[key]

    def clear_layer(self, layer_idx: int) -> None:
        """Evict all entries for a specific layer."""
        if layer_idx in self._caches:
            self._caches[layer_idx].clear()

    def clear_all(self) -> None:
        """Evict all entries."""
        self._caches.clear()

    def num_entries(self, layer_idx: int | None = None) -> int:
        """Return total cached chunks, optionally for a single layer."""
        if layer_idx is not None:
            return len(self._caches.get(layer_idx, {}))
        return sum(len(cache) for cache in self._caches.values())

    def num_tokens(self, layer_idx: int | None = None) -> int:
        """Return the number of token slots cached (accounting for chunk size)."""
        total = 0
        empty: OrderedDict[tuple[str, int, int, int], ActivationChunk] = OrderedDict()
        caches: list[OrderedDict[tuple[str, int, int, int], ActivationChunk]] = (
            [self._caches.get(layer_idx, empty)]
            if layer_idx is not None
            else list(self._caches.values())
        )
        for cache in caches:
            for chunk in cache.values():
                total += chunk.num_valid
        return total
