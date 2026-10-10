"""Contiguous-buffer activation cache for Branch Folding.

This module implements :class:`VectorizedActivationCache`, a drop-in
replacement for :class:`~actfold.core.activation_cache.ActivationCache` that
stores per-branch activations in contiguous ``[capacity, batch, ...]`` buffers
instead of per-token dictionaries.  ``put`` becomes a slice/index copy and
``get`` becomes a single ``index_select``, removing the per-token Python loop
that dominates the folded forward pass.

Token positions are addressed by their sequence index; when more tokens are
stored than the configured capacity (``max_entries_per_layer``), a ring layout
keeps the most recent ``capacity`` tokens.  Missing or evicted positions are
zero-filled so the stability gate classifies them as divergent, mirroring the
gather behavior of the legacy cache.
"""

from __future__ import annotations

from typing import Any

import torch


class VectorizedActivationCache:
    """LRU-independent, contiguous activation cache with batch gather.

    Args:
        max_entries_per_layer: Maximum number of token rows kept per layer.
        device: Target device string (informational; tensors keep their device).
        max_branch_steps: Maximum number of distinct ``(branch_id, step_idx)``
            keys retained; the oldest keys are evicted once exceeded, keeping
            memory bounded across unbounded generation steps. ``0`` or ``None``
            disables the eviction (unbounded, legacy behavior).
    """

    def __init__(
        self,
        max_entries_per_layer: int = 1024,
        device: str = "cuda",
        max_branch_steps: int | None = 4,
    ) -> None:
        self.max_entries = int(max_entries_per_layer)
        self.device = device
        self.max_branch_steps = max_branch_steps
        # (branch_id, step_idx) -> layer_idx -> name -> buffer [cap, B, ...].
        self._buffers: dict[tuple[Any, int], dict[int, dict[str, torch.Tensor]]] = {}
        # (branch_id, step_idx, layer_idx) -> number of tokens stored.
        self._counts: dict[tuple[Any, int, int], int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_layer(
        self,
        branch_id: Any,
        step_idx: int,
        layer_idx: int,
    ) -> dict[str, torch.Tensor]:
        key = (branch_id, step_idx)
        is_new_key = key not in self._buffers
        branch = self._buffers.setdefault(key, {})
        if is_new_key:
            self._evict_old_branch_steps()
        return branch.setdefault(layer_idx, {})

    def _evict_old_branch_steps(self) -> None:
        """Evict the oldest ``(branch, step)`` keys beyond ``max_branch_steps``."""
        limit = self.max_branch_steps
        if not limit or limit <= 0:
            return
        while len(self._buffers) > limit:
            oldest_key = next(iter(self._buffers))
            del self._buffers[oldest_key]
            branch_id, step_idx = oldest_key
            for count_key in [k for k in self._counts if k[0] == branch_id and k[1] == step_idx]:
                del self._counts[count_key]

    # ------------------------------------------------------------------
    # Public API (mirrors ActivationCache)
    # ------------------------------------------------------------------
    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        """Store activations for ``branch_id``/``layer_idx``.

        Args:
            branch_id: Branch identifier.
            layer_idx: Layer index.
            activations: Mapping from activation name to a tensor of shape
                ``[batch, seq_len, ...]``.
            step_idx: Diffusion step index.

        Raises:
            ValueError: If ``activations`` is empty, tensors are not at least
                2-D, or leading shapes are inconsistent.
        """
        if not activations:
            raise ValueError("activations must not be empty")
        expected_shape: tuple[int, ...] | None = None
        for name, tensor in activations.items():
            if tensor.ndim < 2:
                raise ValueError(
                    f"Activation '{name}' must have at least 2 leading dimensions "
                    f"[batch, seq_len, ...], got shape {tensor.shape}"
                )
            leading = tuple(tensor.shape[:2])
            if expected_shape is None:
                expected_shape = leading
            elif leading != expected_shape:
                raise ValueError(
                    f"Activation '{name}' has inconsistent leading shape {leading}; "
                    f"expected {expected_shape}"
                )
        assert expected_shape is not None
        batch_size, seq_len = expected_shape

        store = self._ensure_layer(branch_id, step_idx, layer_idx)
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
                # Fast path: row(t) = t, so a single slice copy is enough.
                buffer[:seq_len].copy_(transposed)
            else:
                start = seq_len - capacity
                rows = torch.arange(start, seq_len, device=buffer.device) % capacity
                buffer.index_copy_(0, rows, transposed[start:])

        self._counts[(branch_id, step_idx, layer_idx)] = seq_len

    def get(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Gather cached activations for the requested tokens.

        Args:
            branch_id: Branch identifier to load from.
            layer_idx: Layer index.
            token_mask: Boolean tensor ``[batch, seq_len]``. ``True`` marks
                tokens whose activations should be retrieved.
            step_idx: Diffusion step index.

        Returns:
            Dictionary of tensors shaped ``[batch, seq_len, ...]`` with
            zero-filled positions where tokens are missing or masked out.

        Raises:
            KeyError: If nothing was ever stored for this branch/layer/step.
        """
        layer_store = self._buffers.get((branch_id, step_idx), {}).get(layer_idx)
        total = self._counts.get((branch_id, step_idx, layer_idx), 0)
        if not layer_store or total == 0:
            raise KeyError(
                f"No cache entry for branch={branch_id}, layer={layer_idx}, step={step_idx}"
            )

        mask = token_mask.to(dtype=torch.bool)
        batch_size, seq_len = mask.shape
        sample = next(iter(layer_store.values()))
        capacity = sample.shape[0]
        start = max(0, total - capacity)

        # Fast path: every token is present in the buffers.
        if start == 0 and total >= seq_len:
            if bool(mask.all()):
                # Full mask: return non-contiguous transposed views (callers only
                # read), avoiding any copy.
                return {
                    name: buffer[:seq_len].transpose(0, 1) for name, buffer in layer_store.items()
                }
            # Partial mask over complete coverage: one clone plus a masked fill is
            # several times cheaper than the index_select/zero-fill path below.
            batch_mask = mask.transpose(0, 1).unsqueeze(-1)
            output_partial: dict[str, torch.Tensor] = {}
            for name, buffer in layer_store.items():
                rest = tuple(buffer.shape[2:])
                data = buffer[:seq_len]
                if data.shape[1] == 1 and batch_size > 1:
                    data = data.expand(seq_len, batch_size, *rest)
                elif data.shape[1] != batch_size:
                    data = data[:, :batch_size]
                result = data.clone()
                result.masked_fill_(~batch_mask, 0)
                output_partial[name] = result.transpose(0, 1)
            return output_partial

        index = torch.arange(seq_len, device=mask.device)
        valid = (index >= start) & (index < total)
        requested = mask.any(dim=0) & valid
        requested_index = torch.nonzero(requested, as_tuple=False).squeeze(-1)
        rows = (requested_index % capacity).to(sample.device)

        # Per-batch mask used to zero positions that this batch did not request;
        # the legacy cache zeroes them per batch element as well.
        full_mask = bool(mask.all())
        batch_mask = mask.transpose(0, 1).unsqueeze(-1)  # [T, B, 1]

        output: dict[str, torch.Tensor] = {}
        for name, buffer in layer_store.items():
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
            if not full_mask:
                result.masked_fill_(~batch_mask, 0)
            output[name] = result.transpose(0, 1)
        return output

    def get_all(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Gather every cached activation for a branch/layer without a mask.

        Unlike :meth:`get`, no ``token_mask`` is required and no
        ``bool(mask.all())`` host sync ever happens; the returned tensors are
        non-contiguous **views** over the internal buffers (callers must treat
        them as read-only).

        Args:
            branch_id: Branch identifier to load from.
            layer_idx: Layer index.
            step_idx: Diffusion step index.

        Returns:
            Dictionary of tensors shaped ``[batch, seq_len, ...]``.

        Raises:
            KeyError: If nothing was ever stored for this branch/layer/step.
        """
        layer_store = self._buffers.get((branch_id, step_idx), {}).get(layer_idx)
        total = self._counts.get((branch_id, step_idx, layer_idx), 0)
        if not layer_store or total == 0:
            raise KeyError(
                f"No cache entry for branch={branch_id}, layer={layer_idx}, step={step_idx}"
            )

        sample = next(iter(layer_store.values()))
        capacity = sample.shape[0]
        start = max(0, total - capacity)

        if start == 0:
            return {name: buffer[:total].transpose(0, 1) for name, buffer in layer_store.items()}
        # Ring layout: gather the surviving rows (still no host sync).
        rows = torch.arange(start, total, device=sample.device) % capacity
        return {
            name: buffer.index_select(0, rows).transpose(0, 1)
            for name, buffer in layer_store.items()
        }

    def fetch(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Return the dense raw activations for a branch/layer/step.

        Alias of :meth:`get_all`: the returned tensors are **views** over the
        internal buffers (callers must treat them as read-only) and no host
        sync or copy is performed.

        Raises:
            KeyError: If nothing was ever stored for this branch/layer/step.
        """
        return self.get_all(branch_id, layer_idx, step_idx)

    def fetch_masked(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Mask-selected fetch; backward-compatible alias of :meth:`get`."""
        return self.get(branch_id, layer_idx, token_mask, step_idx)

    def contains(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> bool:
        """Return ``True`` iff something was ever stored for the key."""
        layer_store = self._buffers.get((branch_id, step_idx), {}).get(layer_idx)
        total = self._counts.get((branch_id, step_idx, layer_idx), 0)
        return bool(layer_store) and total > 0

    def fetch_flat(
        self,
        branch_id: str,
        layer_idx: int,
        batch_size: int,
        seq_len: int,
        step_idx: int = 0,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]] | None:
        """Return flat buffer views and row indices for fused gather (F8).

        Enables single-pass ``gather_select`` merges that read this cache's
        internal buffers directly, skipping the intermediate
        ``[batch, seq_len, ...]`` materialization of :meth:`fetch`.

        Args:
            branch_id: Branch identifier to read from.
            layer_idx: Layer index.
            batch_size: Requested batch size (must match the stored batch).
            seq_len: Requested sequence length; the stored token total must
                cover it completely.
            step_idx: Diffusion step index.

        Returns:
            ``None`` when nothing is stored, the stored tokens do not fully
            cover ``[0, seq_len)``, eviction wrapped the ring buffer, or the
            stored batch size differs.  Otherwise a mapping from activation
            name to ``(flat_buffer, flat_rows)`` where ``flat_buffer`` is a
            2-D view ``[capacity * batch, ...]`` of the internal contiguous
            buffer and ``flat_rows`` is an int64 tensor of shape
            ``[batch * seq_len]`` with ``flat_rows[b * seq_len + t] ==
            t * batch + b`` (the flat-buffer row of child token ``(b, t)``).
            No host synchronization is performed.
        """
        layer_store = self._buffers.get((branch_id, step_idx), {}).get(layer_idx)
        total = self._counts.get((branch_id, step_idx, layer_idx), 0)
        if not layer_store or total == 0 or seq_len <= 0 or batch_size <= 0:
            return None
        sample = next(iter(layer_store.values()))
        capacity = sample.shape[0]
        stored_batch = sample.shape[1]
        start = max(0, total - capacity)
        if start > 0 or total < seq_len or stored_batch != batch_size:
            return None

        t_idx = torch.arange(seq_len, device=sample.device)
        b_idx = torch.arange(batch_size, device=sample.device)
        rows = (t_idx.view(1, seq_len) * batch_size + b_idx.view(batch_size, 1)).reshape(-1)

        output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, buffer in layer_store.items():
            flat = buffer.reshape(capacity * stored_batch, *buffer.shape[2:])
            output[name] = (flat, rows)
        return output

    def clear_branch(self, branch_id: str) -> None:
        """Evict all entries belonging to ``branch_id`` (any step)."""
        for buffer_key in list(self._buffers):
            if buffer_key[0] == branch_id:
                del self._buffers[buffer_key]
        for count_key in list(self._counts):
            if count_key[0] == branch_id:
                del self._counts[count_key]

    def clear_layer(self, layer_idx: int) -> None:
        """Evict a specific layer across all branches and steps."""
        for buffer_key in list(self._buffers):
            self._buffers[buffer_key].pop(layer_idx, None)
            if not self._buffers[buffer_key]:
                del self._buffers[buffer_key]
        for count_key in list(self._counts):
            if count_key[2] == layer_idx:
                del self._counts[count_key]

    def clear_all(self) -> None:
        """Evict all entries."""
        self._buffers.clear()
        self._counts.clear()

    def num_entries(self, layer_idx: int | None = None) -> int:
        """Return the number of token rows currently stored."""
        total = 0
        for (branch_id, step_idx, layer), count in self._counts.items():
            if layer_idx is not None and layer != layer_idx:
                continue
            layer_store = self._buffers.get((branch_id, step_idx), {}).get(layer)
            if not layer_store:
                continue
            capacity = next(iter(layer_store.values())).shape[0]
            total += min(count, capacity)
        return total
