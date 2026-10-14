"""Unified structural contract for activation caches.

All activation cache implementations (:class:`~actfold.core.activation_cache.ActivationCache`,
:class:`~actfold.core.chunked_cache.ChunkedActivationCache` and
:class:`~actfold.core.vectorized_cache.VectorizedActivationCache`) satisfy this
``typing.Protocol``.  It is structural (no inheritance required) and
runtime-checkable so callers may assert cache capabilities with ``isinstance``.

Method contracts:

======= ============================ ==================================== =========
Method  Arguments                    Returns                              Raises
======= ============================ ==================================== =========
fetch   branch/layer/step            dense raw activations ``[B, T, ...]`` KeyError
fetch_  + ``token_mask [B, T]`` bool mask-selected activations with     KeyError
masked                               missing/unselected positions zeroed
get_all branch/layer/step            full read-only view dict             KeyError
contains branch/layer/step           ``True`` iff something is stored     (none)
======= ============================ ==================================== =========

``fetch`` returns the raw dense activations: values at written positions are
the stored values; never-written positions are zero-filled.  Implementations
that can (vectorized/legacy) return **views** over internal buffers, so
callers must treat the returned tensors as read-only.  The old
``get(token_mask)`` method remains available as a backward-compatible alias
of ``fetch_masked``.

AR020 adds the optional batched-write capability
:class:`BatchedPutCapable` (structural, runtime-checkable): caches that
provide ``put_batch`` accept a sequence of :class:`CachePutEntry` records in
one call, state-equivalent to sequential ``put`` calls (batch-level
pre-validation atomicity aside — an invalid batch raises ``ValueError``
before any mutation, which is deliberately stronger than the partial-write
behavior of a sequential loop hitting an invalid entry mid-way).
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, Sequence, runtime_checkable

import torch


class CachePutEntry(NamedTuple):
    """One cache write: mirrors ``cache.put``'s four arguments.

    The field order matches :meth:`ActivationCacheProtocol.put`'s parameter
    order so a ``(branch_id, layer_idx, activations, step_idx)`` tuple can be
    unpacked positionally; ``step_idx`` defaults to ``0`` exactly like the
    ``put`` signature.
    """

    branch_id: str
    layer_idx: int
    activations: dict[str, torch.Tensor]
    step_idx: int = 0


@runtime_checkable
class BatchedPutCapable(Protocol):
    """Optional batched-write capability (structural, no inheritance)."""

    def put_batch(self, entries: Sequence[CachePutEntry]) -> None:
        """Store all entries; state-equivalent to sequential ``put`` calls.

        Raises ValueError before any mutation when any entry is invalid.
        """
        ...


@runtime_checkable
class ActivationCacheProtocol(Protocol):
    """Structural contract shared by every ActFold activation cache."""

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        """Store activations for ``(branch_id, layer_idx, step_idx)``."""
        ...

    def fetch(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Return the dense raw activations for ``(branch, layer, step)``.

        Values at written positions are the stored values; never-written
        positions are zero-filled.  The returned tensors may be views over
        internal buffers and must be treated as read-only by callers.

        Raises:
            KeyError: If nothing is stored for the key.
        """
        ...

    def fetch_masked(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Return mask-selected activations with zero-filled missing positions.

        Identical semantics to the legacy ``get(token_mask)``.

        Raises:
            KeyError: If nothing is stored for the key.
        """
        ...

    def get_all(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Return every cached activation for ``(branch, layer, step)``.

        Zero-synchronization path; implementations that can return views.
        Callers must treat the returned tensors as read-only.

        Raises:
            KeyError: If nothing is stored for the key.
        """
        ...

    def contains(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> bool:
        """Return ``True`` iff something is stored for the key."""
        ...

    def clear_branch(self, branch_id: str) -> None:
        """Evict all entries belonging to ``branch_id``."""
        ...

    def clear_all(self) -> None:
        """Evict all entries."""
        ...
