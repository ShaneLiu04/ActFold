"""Shared batched-write helpers for the activation caches (AR020).

Single-point isolation of the ``torch._foreach_copy_`` dependency (a private
torch API): the three activation caches route their delayed slice copies
through :func:`copy_pairs`, which groups the ``(dst, src)`` pairs by
``(device, dtype)`` and issues one ``_foreach_copy_`` per group. When the
private op is unavailable (older torch), the module-level
``_FOREACH_AVAILABLE`` flag is ``False`` and every pair degrades to a
per-pair ``Tensor.copy_`` — results stay loop-equivalent, only the batching
benefit is lost (no warning by contract: capability probing is the
consumer's job).
"""

from __future__ import annotations

from typing import Sequence

import torch

from actfold.core.cache_protocol import CachePutEntry

# Probed once at import; tests patch this flag to force the per-pair
# downgrade (UT-2009). Patching ``torch._foreach_copy_`` after import has no
# effect by design — the flag is the only switch.
_FOREACH_AVAILABLE: bool = hasattr(torch, "_foreach_copy_")


def validate_entries(entries: Sequence[CachePutEntry]) -> list[CachePutEntry]:
    """Pre-validate every entry; raise ``ValueError`` before any mutation.

    Replicates the three caches' ``put`` validation exactly (per entry:
    empty activations mapping / tensors below 2 leading dimensions /
    inconsistent leading shapes across names). The error message carries the
    offending entry's index. Device/dtype-mixed batches are legal (the copy
    grouping happens later); the check reads host-side tensor metadata only.
    """
    validated = list(entries)
    for index, entry in enumerate(validated):
        activations = entry.activations
        if not activations:
            raise ValueError(
                f"put_batch entry {index}: activations must not be empty"
            )
        expected_shape: tuple[int, ...] | None = None
        for name, tensor in activations.items():
            if tensor.ndim < 2:
                raise ValueError(
                    f"put_batch entry {index}: activation '{name}' must have "
                    f"at least 2 leading dimensions [batch, seq_len, ...], "
                    f"got shape {tuple(tensor.shape)}"
                )
            leading = tuple(tensor.shape[:2])
            if expected_shape is None:
                expected_shape = leading
            elif leading != expected_shape:
                raise ValueError(
                    f"put_batch entry {index}: activation '{name}' has "
                    f"inconsistent leading shape {leading}; expected "
                    f"{expected_shape}"
                )
    return validated


def copy_pairs(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Execute ``(dst, src)`` copy pairs grouped by ``(device, dtype)``.

    Returns the host write-op count (one per foreach group, or one per pair
    on the downgrade path) — the ``write_op_count`` ledger source. Sources
    may be non-contiguous views (``transpose``), shapes may differ within a
    group; only device and dtype must match, which is exactly the grouping
    key.
    """
    if not pairs:
        return 0
    if not _FOREACH_AVAILABLE:
        for dst, src in pairs:
            dst.copy_(src)
        return len(pairs)
    groups: dict[
        tuple[torch.device, torch.dtype], list[tuple[torch.Tensor, torch.Tensor]]
    ] = {}
    for pair in pairs:
        dst = pair[0]
        groups.setdefault((dst.device, dst.dtype), []).append(pair)
    for group in groups.values():
        torch._foreach_copy_([dst for dst, _ in group], [src for _, src in group])
    return len(groups)
