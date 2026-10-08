"""Layer-local LRU cache for parent-branch activations.

Internals were rewritten in AR001 (T012) from a per-token ``OrderedDict`` to
contiguous per-``(branch, step)`` buffers: ``put`` is a single vectorized
slice copy and ``get`` is a mask-based gather with zero-fill. Neither path
performs per-token Python loops or per-token host readbacks (``.item()`` /
``.any()``); the LRU is maintained at the group level with a single touch.

The token budget per layer (``max_entries_per_layer``) is enforced by evicting
the oldest groups; a single group larger than the whole budget is kept (the
old per-token eviction could split groups, which no test relied on).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch


class ActivationCache:
    """LRU cache for parent activations, partitioned per layer.

    Each layer keeps an ordered mapping from ``(branch_id, step_idx)`` groups
    to contiguous buffers ``[capacity, batch, ...]`` per activation name. The
    public API is unchanged from the per-token version.

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
        # layer_idx -> OrderedDict[(branch_id, step_idx)] -> {name: buffer}.
        self._layers: dict[int, OrderedDict[tuple[Any, int], dict[str, torch.Tensor]]] = {}
        # (branch_id, step_idx, layer_idx) -> number of tokens stored.
        self._counts: dict[tuple[Any, int, int], int] = {}

    def _ensure_layer(self, layer_idx: int) -> OrderedDict[tuple[Any, int], dict[str, torch.Tensor]]:
        """Create the ordered group map for a layer if it does not exist."""
        if layer_idx not in self._layers:
            self._layers[layer_idx] = OrderedDict()
        return self._layers[layer_idx]

    def _group_capacity(self, store: dict[str, torch.Tensor]) -> int:
        """Return the row capacity of a group's buffers (0 when empty)."""
        if not store:
            return 0
        return next(iter(store.values())).shape[0]

    def _layer_tokens(self, layer_idx: int) -> int:
        """Return the number of token rows currently stored in a layer."""
        total = 0
        for (branch_id, step_idx), store in self._layers.get(layer_idx, {}).items():
            if not store:
                continue
            count = self._counts.get((branch_id, step_idx, layer_idx), 0)
            total += min(count, self._group_capacity(store))
        return total

    def _evict_over_budget(self, layer_idx: int) -> None:
        """Evict oldest groups until the layer fits its token budget.

        The newest group is always kept, even if it alone exceeds the budget:
        splitting a just-written group would corrupt its contiguous buffer.
        """
        layer = self._layers.get(layer_idx)
        if layer is None:
            return
        while self._layer_tokens(layer_idx) > self.max_entries and len(layer) > 1:
            oldest_key = next(iter(layer))
            del layer[oldest_key]
            branch_id, step_idx = oldest_key
            self._counts.pop((branch_id, step_idx, layer_idx), None)

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        """Store activations for a branch/layer/step as one vectorized copy.

        Args:
            branch_id: Unique branch identifier.
            layer_idx: Layer index.
            activations: Mapping from activation name to a tensor of shape
                ``[batch, seq_len, ...]``.
            step_idx: Diffusion step index.

        Raises:
            ValueError: If ``activations`` is empty or contains inconsistent
                leading shapes.
        """
        if not activations:
            raise ValueError("activations must not be empty")

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

        layer = self._ensure_layer(layer_idx)
        key = (branch_id, step_idx)
        store = layer.get(key)
        if store is not None:
            layer.move_to_end(key)
        else:
            store = {}
            layer[key] = store

        for name, tensor in activations.items():
            if name not in store:
                capacity = min(self.max_entries, max(seq_len, 1))
                shape = (capacity, batch_size) + tuple(tensor.shape[2:])
                store[name] = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
            buffer = store[name]
            if seq_len > buffer.shape[0] and buffer.shape[0] < self.max_entries:
                new_capacity = min(self.max_entries, max(seq_len, buffer.shape[0] * 2))
                grown = torch.empty(
                    (new_capacity,) + tuple(buffer.shape[1:]),
                    dtype=buffer.dtype,
                    device=buffer.device,
                )
                grown[: buffer.shape[0]].copy_(buffer)
                store[name] = buffer = grown

            capacity = buffer.shape[0]
            transposed = tensor.transpose(0, 1)
            if seq_len <= capacity:
                buffer[:seq_len].copy_(transposed)
            else:
                start = seq_len - capacity
                rows = torch.arange(start, seq_len, device=buffer.device) % capacity
                buffer.index_copy_(0, rows, transposed[start:])

        self._counts[(branch_id, step_idx, layer_idx)] = seq_len
        self._evict_over_budget(layer_idx)

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
            KeyError: If nothing is stored for this branch/layer/step.
        """
        layer = self._layers.get(layer_idx)
        store = layer.get((branch_id, step_idx)) if layer is not None else None
        total = self._counts.get((branch_id, step_idx, layer_idx), 0)
        if layer is None or not store or total == 0:
            raise KeyError(
                f"No cached activations for branch={branch_id}, "
                f"layer={layer_idx}, step={step_idx}"
            )

        # Single group-level LRU touch: no per-token loops, no host readback.
        layer.move_to_end((branch_id, step_idx))

        mask = token_mask.to(dtype=torch.bool)
        batch_size, seq_len = mask.shape
        sample = next(iter(store.values()))
        capacity = sample.shape[0]
        start = max(0, total - capacity)

        if start == 0 and total >= seq_len:
            # Complete coverage: one clone plus a masked fill per activation.
            # (No ``bool(mask.all())`` readback: the fill handles full masks.)
            batch_mask = mask.transpose(0, 1).unsqueeze(-1)  # [T, B, 1]
            output: dict[str, torch.Tensor] = {}
            for name, buffer in store.items():
                rest = tuple(buffer.shape[2:])
                data = buffer[:seq_len]
                if data.shape[1] == 1 and batch_size > 1:
                    data = data.expand(seq_len, batch_size, *rest)
                elif data.shape[1] != batch_size:
                    data = data[:, :batch_size]
                result = data.clone()
                result.masked_fill_(~batch_mask, 0)
                output[name] = result.transpose(0, 1)
            return output

        # Ring layout: gather surviving rows; zero-fill everything else.
        index = torch.arange(seq_len, device=mask.device)
        valid = (index >= start) & (index < total)
        requested = mask.any(dim=0) & valid  # tensor-only, no host readback
        requested_index = torch.nonzero(requested, as_tuple=False).squeeze(-1)
        rows = (requested_index % capacity).to(sample.device)

        batch_mask = mask.transpose(0, 1).unsqueeze(-1)  # [T, B, 1]

        output = {}
        for name, buffer in store.items():
            rest = tuple(buffer.shape[2:])
            stored_batch = buffer.shape[1]
            result = torch.zeros(
                (seq_len, batch_size) + rest,
                dtype=buffer.dtype,
                device=buffer.device,
            )
            if rows.numel() > 0:
                gathered = buffer.index_select(0, rows)
                if stored_batch == batch_size:
                    pass
                elif stored_batch == 1:
                    gathered = gathered.expand(-1, batch_size, *rest)
                elif gathered.shape[1] >= batch_size:
                    gathered = gathered[:, :batch_size]
                else:
                    gathered = gathered.expand(-1, batch_size, *rest)
                result.index_copy_(0, requested_index.to(buffer.device), gathered)
            # Batch elements that did not request a copied position must be
            # zeroed; applying the fill unconditionally avoids the host
            # readback a "is the mask full?" check would require.
            result.masked_fill_(~batch_mask, 0)
            output[name] = result.transpose(0, 1)
        return output

    def fetch(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Return the dense raw activations for a branch/layer/step.

        Values at written positions are the stored values; never-written
        positions are zero-filled (complete coverage keeps the buffer
        zero-initialized).  The returned tensors are **views** over the
        internal buffers (transpose to ``[batch, seq, ...]``); callers must
        treat them as read-only.  No host sync and no copy are performed.

        Args:
            branch_id: Branch identifier to load from.
            layer_idx: Layer index.
            step_idx: Diffusion step index.

        Returns:
            Dictionary of raw activations shaped ``[batch, seq, ...]``.

        Raises:
            KeyError: If nothing is stored for this branch/layer/step.
        """
        layer = self._layers.get(layer_idx)
        store = layer.get((branch_id, step_idx)) if layer is not None else None
        total = self._counts.get((branch_id, step_idx, layer_idx), 0)
        if layer is None or not store or total == 0:
            raise KeyError(
                f"No cached activations for branch={branch_id}, "
                f"layer={layer_idx}, step={step_idx}"
            )

        # Single group-level LRU touch: no per-token loops, no host readback.
        layer.move_to_end((branch_id, step_idx))

        sample = next(iter(store.values()))
        capacity = sample.shape[0]
        start = max(0, total - capacity)

        if start == 0:
            return {
                name: buffer[:total].transpose(0, 1) for name, buffer in store.items()
            }
        # Ring layout: gather the surviving rows (still no host sync).
        rows = torch.arange(start, total, device=sample.device) % capacity
        return {
            name: buffer.index_select(0, rows).transpose(0, 1)
            for name, buffer in store.items()
        }

    def fetch_masked(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Mask-selected fetch; backward-compatible alias of :meth:`get`."""
        return self.get(branch_id, layer_idx, token_mask, step_idx)

    def get_all(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Full read-only fetch; alias of :meth:`fetch` for this cache."""
        return self.fetch(branch_id, layer_idx, step_idx)

    def contains(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> bool:
        """Return ``True`` iff something is stored for the key."""
        layer = self._layers.get(layer_idx)
        store = layer.get((branch_id, step_idx)) if layer is not None else None
        total = self._counts.get((branch_id, step_idx, layer_idx), 0)
        return bool(store) and total > 0

    def clear_branch(self, branch_id: str) -> None:
        """Evict all entries belonging to ``branch_id``."""
        for layer_idx in list(self._layers):
            layer = self._layers[layer_idx]
            for key in [k for k in layer if k[0] == branch_id]:
                del layer[key]
                self._counts.pop((key[0], key[1], layer_idx), None)

    def clear_layer(self, layer_idx: int) -> None:
        """Evict all entries for a specific layer."""
        self._layers.pop(layer_idx, None)
        for key in [k for k in self._counts if k[2] == layer_idx]:
            del self._counts[key]

    def clear_all(self) -> None:
        """Evict all entries."""
        self._layers.clear()
        self._counts.clear()

    def num_entries(self, layer_idx: int | None = None) -> int:
        """Return total cached entries, optionally for a single layer."""
        if layer_idx is not None:
            return self._layer_tokens(layer_idx)
        return sum(self._layer_tokens(layer) for layer in self._layers)
