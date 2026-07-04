"""Layer-local LRU cache for parent-branch activations."""

from __future__ import annotations

from collections import OrderedDict

import torch

from actfold.core.fused_ops import gather_cached_activations


class ActivationCache:
    """LRU cache for parent activations, partitioned per layer.

    Each layer maintains its own LRU dictionary. The cache key is a tuple of
    ``(branch_id, layer_idx, token_idx, step_idx)``. Stored values are dicts
    mapping activation names (e.g., "ffn_out", "hidden_states") to tensors.

    Args:
        max_entries_per_layer: Maximum number of token-level entries per layer.
        device: Target device string (informational; tensors keep their own device).
    """

    def __init__(
        self,
        max_entries_per_layer: int = 1024,
        device: str = "cuda",
    ) -> None:
        self.max_entries = max_entries_per_layer
        self.device = device
        self._caches: dict[int, OrderedDict[tuple[str, int, int, int], dict[str, torch.Tensor]]] = (
            {}
        )

    def _ensure_layer(self, layer_idx: int) -> None:
        """Create an empty OrderedDict for a layer if it does not exist."""
        if layer_idx not in self._caches:
            self._caches[layer_idx] = OrderedDict()

    @staticmethod
    def _key(
        branch_id: str,
        layer_idx: int,
        token_idx: int,
        step_idx: int,
    ) -> tuple[str, int, int, int]:
        """Build a cache key."""
        return (branch_id, layer_idx, token_idx, step_idx)

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        """Store activations per token for a branch/layer/step.

        The ``activations`` dict must contain tensors of shape
        ``[batch, seq_len, ...]``. The tensor is iterated over the sequence
        dimension and each token position is stored independently.

        Args:
            branch_id: Unique branch identifier.
            layer_idx: Layer index.
            activations: Mapping from activation name to tensor.
            step_idx: Diffusion step index.

        Raises:
            ValueError: If ``activations`` is empty or contains inconsistent
                leading shapes.
        """
        if not activations:
            raise ValueError("activations must not be empty")

        self._ensure_layer(layer_idx)
        cache = self._caches[layer_idx]

        # Validate that all tensors share the same batch and sequence length.
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

        for token_idx in range(seq_len):
            key = self._key(branch_id, layer_idx, token_idx, step_idx)
            per_token_activations = {
                name: tensor[:, token_idx, ...].contiguous() for name, tensor in activations.items()
            }

            if key in cache:
                cache.move_to_end(key)
            cache[key] = per_token_activations

            if len(cache) > self.max_entries:
                cache.popitem(last=False)

    def get(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Retrieve activations for stable tokens in a branch/layer/step.

        Args:
            branch_id: Branch identifier to load from.
            layer_idx: Layer index.
            token_mask: Boolean tensor of shape ``[batch, seq_len]``.
                ``True`` marks tokens whose activations should be retrieved.
            step_idx: Diffusion step index.

        Returns:
            Dictionary of activations with the same shape as stored tensors
            except missing/unstable positions are zero-filled.

        Raises:
            KeyError: If a requested token key is missing from the cache.
        """
        self._ensure_layer(layer_idx)
        cache = self._caches[layer_idx]

        output = gather_cached_activations(
            cache=cache,
            branch_id=branch_id,
            layer_idx=layer_idx,
            token_mask=token_mask,
            step_idx=step_idx,
        )

        # Touch retrieved entries so LRU ordering is preserved.
        batch_size, seq_len = token_mask.shape
        for token_idx in range(seq_len):
            key = self._key(branch_id, layer_idx, token_idx, step_idx)
            if key in cache and token_mask[:, token_idx].any():
                cache.move_to_end(key)

        return output

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
        """Return total cached entries, optionally for a single layer."""
        if layer_idx is not None:
            return len(self._caches.get(layer_idx, {}))
        return sum(len(cache) for cache in self._caches.values())
