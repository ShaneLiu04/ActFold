"""Block-paged, layer-pyramid activation cache (AR029).

Missing token positions are represented by :attr:`missing_value` (NaN) and can
be queried with :meth:`PyramidActivationCache.availability`, so the folded stack
can force a recompute instead of silently reusing stale data.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Iterator, Sequence

import torch

from actfold.core.cache_protocol import CachePutEntry
from actfold.core.pyramid_plan import PyramidPlan


class _Entry:
    """One ``(branch, step, layer)`` block page table."""

    __slots__ = ("seq_len", "batch", "blocks", "block_lens")

    def __init__(self, seq_len: int, batch: int) -> None:
        self.seq_len = seq_len
        self.batch = batch
        self.blocks: dict[int, dict[str, torch.Tensor]] = {}
        self.block_lens: dict[int, int] = {}

    def names(self) -> tuple[str, ...]:
        if not self.blocks:
            return ()
        return tuple(next(iter(self.blocks.values())).keys())


class PyramidActivationCache:
    """Per-layer budgeted block-paged activation cache.

    Args:
        plan: Frozen :class:`PyramidPlan` (per-layer token budgets).
        device: Informational device string (tensors keep their own device).
    """

    graph_incompatible: bool = True
    missing_value: float = float("nan")

    def __init__(self, plan: PyramidPlan, device: str = "cuda") -> None:
        self.plan = plan
        self.device = device
        self._layers: dict[int, OrderedDict[tuple[Any, int], _Entry]] = {}
        # Per-layer global write order over (branch, step, block_idx).
        self._order: dict[int, OrderedDict[tuple[Any, int, int], None]] = {}
        self._tokens: dict[int, int] = {}
        self._evictions = 0
        self._requests = 0
        self._missing_tokens = 0
        self.write_op_count = 0

    # ------------------------------------------------------------------
    # validation / helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_activations(activations: dict[str, torch.Tensor]) -> tuple[int, int]:
        if not activations:
            raise ValueError("activations must not be empty")
        expected: tuple[int, int] | None = None
        for name, tensor in activations.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"activation {name!r} is not a tensor")
            if tensor.ndim < 2:
                raise ValueError(
                    f"activation {name!r} must have at least 2 leading dimensions, "
                    f"got {tuple(tensor.shape)}"
                )
            if not torch.is_floating_point(tensor):
                raise ValueError(
                    f"activation {name!r} must be floating point to support the NaN sentinel"
                )
            leading = (int(tensor.shape[0]), int(tensor.shape[1]))
            if leading[0] <= 0 or leading[1] <= 0:
                raise ValueError(
                    f"activation {name!r} must have a non-empty leading shape, " f"got {leading}"
                )
            if expected is None:
                expected = leading
            elif leading != expected:
                raise ValueError(
                    f"activation {name!r} has leading shape {leading}, expected {expected}"
                )
        assert expected is not None
        return expected

    def _delete_entry(self, layer_idx: int, key: tuple[Any, int], entry: _Entry) -> None:
        for block_idx in list(entry.blocks):
            order = self._order.get(layer_idx)
            if order is not None:
                order.pop((key[0], key[1], block_idx), None)
        self._tokens[layer_idx] = self._tokens.get(layer_idx, 0) - sum(entry.block_lens.values())
        layer = self._layers.get(layer_idx)
        if layer is not None:
            layer.pop(key, None)

    def _evict(self, layer_idx: int, current_key: tuple[Any, int], recent: set[int]) -> None:
        budget = self.plan.budget_tokens(layer_idx)
        order = self._order.setdefault(layer_idx, OrderedDict())
        layer = self._layers.get(layer_idx)
        if layer is None:
            return
        while self._tokens.get(layer_idx, 0) > budget and order:
            victim: tuple[Any, int, int] | None = None
            for candidate in order:
                branch_id, step_idx, block_idx = candidate
                entry = layer.get((branch_id, step_idx))
                if entry is None:
                    victim = candidate
                    break
                anchors = self.plan.retained_blocks(layer_idx, entry.seq_len)
                is_recent = (branch_id, step_idx) == current_key and block_idx in recent
                if block_idx not in anchors and not is_recent:
                    victim = candidate
                    break
            if victim is None:
                # Hard bound wins: reclaim the oldest block even if protected.
                victim = next(iter(order))
            branch_id, step_idx, block_idx = victim
            del order[victim]
            entry = layer.get((branch_id, step_idx))
            if entry is None:
                continue
            block = entry.blocks.pop(block_idx, None)
            length = entry.block_lens.pop(block_idx, 0)
            if block is None:
                continue
            self._tokens[layer_idx] = self._tokens.get(layer_idx, 0) - length
            self._evictions += 1
            if not entry.blocks:
                self._delete_entry(layer_idx, (branch_id, step_idx), entry)

    def _assemble(self, entry: _Entry, seq_len: int) -> dict[str, torch.Tensor]:
        names = entry.names()
        blocks = list(entry.blocks.items())
        first = blocks[0][1]
        sample = next(iter(first.values()))
        rest = tuple(sample.shape[2:])
        out: dict[str, torch.Tensor] = {}
        for name in names:
            buf = torch.full(
                (seq_len, entry.batch) + rest,
                self.missing_value,
                dtype=sample.dtype,
                device=sample.device,
            )
            for block_idx, block in blocks:
                tensor = block.get(name)
                if tensor is None:
                    continue
                start = block_idx * self.plan.block_size
                if start >= seq_len:
                    continue
                length = min(entry.block_lens[block_idx], seq_len - start)
                if length <= 0:
                    continue
                buf[start : start + length] = tensor[:length]
            out[name] = buf.transpose(0, 1)
        return out

    def _entry(self, branch_id: str, layer_idx: int, step_idx: int) -> _Entry:
        layer = self._layers.get(layer_idx)
        entry = layer.get((branch_id, step_idx)) if layer is not None else None
        if entry is None or not entry.blocks:
            raise KeyError(
                f"No cached activations for branch={branch_id}, "
                f"layer={layer_idx}, step={step_idx}"
            )
        return entry

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        """Store activations split into blocks, then enforce the layer budget."""
        if layer_idx < 0 or layer_idx >= self.plan.depth:
            raise IndexError(f"layer {layer_idx} out of range for depth {self.plan.depth}")
        batch, seq_len = self._validate_activations(activations)
        layer = self._layers.setdefault(layer_idx, OrderedDict())
        order = self._order.setdefault(layer_idx, OrderedDict())
        key = (branch_id, step_idx)
        if key in layer:
            self._delete_entry(layer_idx, key, layer[key])

        entry = _Entry(seq_len, batch)
        block_size = self.plan.block_size
        n_blocks = math.ceil(seq_len / block_size)
        for block_idx in range(n_blocks):
            start = block_idx * block_size
            stop = min(start + block_size, seq_len)
            # clone, not contiguous(): at batch 1 the transposed slice is already
            # contiguous, so contiguous() would return a view that pins the whole
            # source tensor and eviction would free nothing.
            block = {
                name: tensor[:, start:stop]
                .transpose(0, 1)
                .clone(memory_format=torch.contiguous_format)
                for name, tensor in activations.items()
            }
            entry.blocks[block_idx] = block
            entry.block_lens[block_idx] = stop - start
            order[(branch_id, step_idx, block_idx)] = None
        layer[key] = entry
        self._tokens[layer_idx] = self._tokens.get(layer_idx, 0) + seq_len
        self.write_op_count += len(activations)

        recent_start = max(0, n_blocks - self.plan.recent_blocks)
        recent = set(range(recent_start, n_blocks))
        self._evict(layer_idx, key, recent)

    def put_batch(self, entries: Sequence[CachePutEntry]) -> None:
        """Atomically validate, then perform sequential ``put`` calls."""
        for entry in entries:
            if entry.layer_idx < 0 or entry.layer_idx >= self.plan.depth:
                raise IndexError(
                    f"layer {entry.layer_idx} out of range for depth {self.plan.depth}"
                )
            self._validate_activations(entry.activations)
        for entry in entries:
            self.put(
                entry.branch_id,
                entry.layer_idx,
                entry.activations,
                entry.step_idx,
            )

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def fetch(self, branch_id: str, layer_idx: int, step_idx: int = 0) -> dict[str, torch.Tensor]:
        """Return dense activations with NaN at missing positions."""
        entry = self._entry(branch_id, layer_idx, step_idx)
        out = self._assemble(entry, entry.seq_len)
        self._requests += 1
        present = int(sum(min(length, entry.seq_len) for length in entry.block_lens.values()))
        self._missing_tokens += max(0, entry.seq_len - present) * entry.batch
        return out

    def get_all(self, branch_id: str, layer_idx: int, step_idx: int = 0) -> dict[str, torch.Tensor]:
        """Alias of :meth:`fetch` (zero-synchronization full read)."""
        return self.fetch(branch_id, layer_idx, step_idx)

    def get(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Return zero-filled unrequested rows and NaN for requested holes."""
        return self.fetch_masked(branch_id, layer_idx, token_mask, step_idx)

    def fetch_masked(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Return mask-selected activations (0 unselected, NaN requested holes)."""
        entry = self._entry(branch_id, layer_idx, step_idx)
        mask = token_mask.to(dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[0] != entry.batch:
            raise ValueError(
                f"token_mask must be [batch, seq_len] with batch={entry.batch}, "
                f"got {tuple(mask.shape)}"
            )
        seq_len = int(mask.shape[1])
        out = self._assemble(entry, seq_len)
        keep = mask.unsqueeze(-1)
        for name, tensor in out.items():
            tensor = tensor.masked_fill(~keep, 0)
            out[name] = tensor
        self._requests += 1
        avail = self.availability(branch_id, layer_idx, seq_len, step_idx)
        if avail.ndim == 1:
            avail2d = avail.view(1, -1).expand(entry.batch, -1)
        else:
            avail2d = avail
        self._missing_tokens += int((mask & ~avail2d.to(mask.device)).sum())
        return out

    def availability(
        self,
        branch_id: str,
        layer_idx: int,
        seq_len: int,
        step_idx: int = 0,
        name: str = "ffn_out",
    ) -> torch.Tensor:
        """Return a boolean mask of stored token positions."""
        layer = self._layers.get(layer_idx)
        entry = layer.get((branch_id, step_idx)) if layer is not None else None
        if entry is None:
            return torch.zeros(seq_len, dtype=torch.bool)
        sample = next(iter(entry.blocks.values()))[name]
        mask = torch.zeros(seq_len, dtype=torch.bool, device=sample.device)
        block_size = self.plan.block_size
        limit = min(seq_len, entry.seq_len)
        for block_idx, length in entry.block_lens.items():
            start = block_idx * block_size
            if start >= limit:
                continue
            stop = min(start + length, limit)
            if stop > start:
                mask[start:stop] = True
        if entry.batch > 1:
            return mask.unsqueeze(0).expand(entry.batch, -1)
        return mask

    # ------------------------------------------------------------------
    # introspection / clearing
    # ------------------------------------------------------------------
    def contains(self, branch_id: str, layer_idx: int, step_idx: int = 0) -> bool:
        """Return True iff a non-empty entry exists."""
        layer = self._layers.get(layer_idx)
        entry = layer.get((branch_id, step_idx)) if layer is not None else None
        return entry is not None and bool(entry.blocks)

    def num_entries(self, layer_idx: int | None = None) -> int:
        """Return stored token count (all layers when ``layer_idx`` is None)."""
        if layer_idx is None:
            return sum(self._tokens.values())
        return self._tokens.get(layer_idx, 0)

    def clear_branch(self, branch_id: str) -> None:
        """Evict every entry of ``branch_id``."""
        for layer_idx in list(self._layers):
            for key in list(self._layers[layer_idx]):
                if key[0] == branch_id:
                    self._delete_entry(layer_idx, key, self._layers[layer_idx][key])

    def clear_layer(self, layer_idx: int) -> None:
        """Evict every entry of ``layer_idx``."""
        layer = self._layers.get(layer_idx)
        for key in list(layer) if layer is not None else []:
            self._delete_entry(layer_idx, key, self._layers[layer_idx][key])
        self._order.pop(layer_idx, None)
        self._tokens[layer_idx] = 0

    def clear_all(self) -> None:
        """Evict everything."""
        self._layers.clear()
        self._order.clear()
        self._tokens.clear()

    def iter_block_tensors(self) -> Iterator[torch.Tensor]:
        """Yield every retained block tensor (for byte accounting)."""
        for layer in self._layers.values():
            for entry in layer.values():
                for block in entry.blocks.values():
                    yield from block.values()

    @property
    def bytes_used(self) -> int:
        """Total bytes of retained block tensors."""
        return sum(t.numel() * t.element_size() for t in self.iter_block_tensors())

    def stats(self) -> dict[str, Any]:
        """Return a deterministic snapshot of cache counters."""
        tokens = {layer: self._tokens.get(layer, 0) for layer in sorted(self._tokens)}
        blocks = {
            layer: sum(len(entry.blocks) for entry in self._layers.get(layer, {}).values())
            for layer in sorted(self._layers)
        }
        return {
            "tokens": tokens,
            "blocks": blocks,
            "bytes": self.bytes_used,
            "evictions": self._evictions,
            "requests": self._requests,
            "missing_tokens": self._missing_tokens,
        }
