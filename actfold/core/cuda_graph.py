"""CUDA graph capture/replay for the Manual folded verification forward.

The :class:`FoldedGraphRunner` (AR002/T007, M4b) captures the *static* folded
verification forward — fixed ``(batch, seq)`` shape, fixed divergent capacity
``C`` per layer — into a CUDA graph and replays it for subsequent
verification steps.  It builds on the static-friendly primitives introduced
by AR002:

* ``fused_gate_mask_count`` produces the stability mask and the stable count
  in one kernel (no separate ``mask.sum()`` reduction);
* ``_padded_divergent_index`` derives a fixed-shape ``[C]`` divergent row
  index from the mask via ``argsort`` (no ``nonzero`` host sync);
* the split-layer resident hooks gather exactly ``[C]`` FFN rows, so every
  tensor shape inside the captured body is data-independent.

The capture body never touches the activation cache, the profiler, or any
host readback.  Per replay the host performs exactly ONE readback — the
per-layer stable counts — to validate the divergent budget
(``D_l = N - count_l <= C``); an exceeded budget means the padded capture
recompute was insufficient for that step, so the caller must discard the
replay output and recompute eagerly (correctness first).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn as nn

from actfold.core.fused_ops import fused_gate_mask_count, merge_stable_divergent
from actfold.core.similarity_gate import SimilarityGate
from actfold.core.split_layer import SplitFoldedTransformerLayer, _padded_divergent_index

__all__ = ["FoldedGraphRunner"]


class FoldedGraphRunner:
    """Capture and replay the fixed-shape folded verification forward.

    The runner owns all static buffers (tokens, parent activations, per-layer
    masks/counts, merged child outputs, logits).  ``capture`` warms the
    kernels up on a side stream and records the graph; ``replay`` copy-fills
    the static inputs from the caller's tensors and the parent branch cache,
    replays the graph, and validates the divergent budget with a single
    ``tolist`` readback.

    Args:
        wrapped_layers: The folded layer stack (e.g.
            ``ManualFoldedForward._wrapped_layers``); every layer must be
            CUDA-resident, scheduler-free, and driven by an exact-type
            ``SimilarityGate(metric="cosine")``.
        embed_fn: Token embedding callable ``[B, T] -> [B, T, H]``.
        final_norm_fn: Optional final normalization applied before the head.
        head_fn: Optional LM head; without it the returned buffer holds hidden
            states instead of logits.
        cache: Activation cache holding the parent branch activations.
        gate_tau: Cosine similarity threshold (stable iff ``sim > tau``).
        gate_eps: Eps for the cosine denominator (per-dtype floors apply).
        capacity_ratio: Fraction of tokens reserved for divergent recompute;
            the fixed capacity is ``C = ceil(ratio * batch * seq)``.
        attention_mask_static: Optional fixed-shape attention mask baked into
            every capture/replay; pass ``None`` for unmasked forwards.
    """

    def __init__(
        self,
        wrapped_layers: nn.ModuleList,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        final_norm_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        head_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        cache: Any,
        gate_tau: float,
        gate_eps: float,
        capacity_ratio: float,
        attention_mask_static: torch.Tensor | None,
    ) -> None:
        if not 0.0 < capacity_ratio <= 1.0:
            raise ValueError(f"capacity_ratio must satisfy 0 < ratio <= 1, got {capacity_ratio}")
        if len(wrapped_layers) == 0:
            raise ValueError("wrapped_layers must contain at least one layer")
        self._layers = wrapped_layers
        self._embed_fn = embed_fn
        self._final_norm_fn = final_norm_fn
        self._head_fn = head_fn
        self._cache = cache
        self._gate_tau = float(gate_tau)
        self._gate_eps = float(gate_eps)
        self._capacity_ratio = float(capacity_ratio)
        self._attention_mask_static = attention_mask_static

        # Populated by capture().
        self._num_layers = len(wrapped_layers)
        self._num_tokens = 0
        self._capacity = 0
        self._tokens_static: torch.Tensor | None = None
        self._parent_static: torch.Tensor | None = None
        self._mask_buf: torch.Tensor | None = None
        self._count_buf: torch.Tensor | None = None
        self._child_buf: torch.Tensor | None = None
        self._logits_buf: torch.Tensor | None = None
        self._graph: Any = None
        self._budget_exceeded = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def capacity(self) -> int:
        """Fixed divergent capacity ``C`` per layer (set by ``capture``)."""
        return self._capacity

    @property
    def budget_exceeded(self) -> bool:
        """Whether the most recent ``validate_budgets`` call failed."""
        return self._budget_exceeded

    # ------------------------------------------------------------------
    # Static-buffer accessors (host-side; caller clones before next replay)
    # ------------------------------------------------------------------
    @property
    def tokens_static(self) -> torch.Tensor:
        """Static token buffer ``[batch, seq]`` (capture-time shape)."""
        self._require_captured()
        return self._tokens_static  # type: ignore[return-value]

    @property
    def parent_static(self) -> torch.Tensor:
        """Static parent activations ``[L+1, batch, seq, hidden]``.

        Index 0 holds the parent embedding; ``parent_static[l + 1]`` holds the
        parent layer-``l`` output.
        """
        self._require_captured()
        return self._parent_static  # type: ignore[return-value]

    @property
    def mask_buf(self) -> torch.Tensor:
        """Per-layer stability masks ``[L, batch, seq]`` (bool)."""
        self._require_captured()
        return self._mask_buf  # type: ignore[return-value]

    @property
    def count_buf(self) -> torch.Tensor:
        """Per-layer stable counts ``[L]`` (int32)."""
        self._require_captured()
        return self._count_buf  # type: ignore[return-value]

    @property
    def child_buf(self) -> torch.Tensor:
        """Per-layer merged child outputs ``[L, batch, seq, hidden]``."""
        self._require_captured()
        return self._child_buf  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def capture(
        self,
        tokens: torch.Tensor,
        branch_id: str,
        parent_branch_id: str,
    ) -> None:
        """Warm up and capture the static folded forward as a CUDA graph.

        Args:
            tokens: Child token ids ``[batch, seq]`` fixing the captured
                shape.
            branch_id: Child branch identifier (informational; the captured
                graph is branch-agnostic).
            parent_branch_id: Parent branch whose cache entries feed the
                static parent buffers.

        Raises:
            RuntimeError: If CUDA is unavailable, the inputs/layers are not
                CUDA-resident, a layer carries a scheduler, a layer's gate is
                not an exact-type cosine ``SimilarityGate``, the parent cache
                is incomplete, or the capture itself fails.
        """
        del branch_id
        if not tokens.is_cuda:
            raise RuntimeError(
                "FoldedGraphRunner requires CUDA tensors; got " f"tokens on {tokens.device}"
            )
        for layer in self._layers:
            if getattr(layer, "scheduler", None) is not None:
                raise RuntimeError(
                    "FoldedGraphRunner cannot capture layers with a folding "
                    "scheduler (dynamic tau / disabled layers are host-side "
                    f"branches); layer {layer} carries one."
                )
            gate = getattr(layer, "gate", None)
            if type(gate) is not SimilarityGate or gate.metric != "cosine":
                raise RuntimeError(
                    "FoldedGraphRunner requires an exact-type "
                    f'SimilarityGate(metric="cosine") on every layer; got {gate!r}.'
                )

        batch, seq_len = tokens.shape
        self._num_tokens = batch * seq_len
        self._capacity = int(math.ceil(self._capacity_ratio * self._num_tokens))
        self._capacity = max(1, min(self._capacity, self._num_tokens))

        # Allocate the static buffers (outside the graph pool: the host writes
        # them between replays).
        self._tokens_static = tokens.clone()
        probe = self._embed_fn(self._tokens_static)
        hidden_dim = probe.shape[-1]
        dtype = probe.dtype
        device = tokens.device
        self._parent_static = torch.empty(
            self._num_layers + 1, batch, seq_len, hidden_dim, dtype=dtype, device=device
        )
        self._mask_buf = torch.empty(
            self._num_layers, batch, seq_len, dtype=torch.bool, device=device
        )
        self._count_buf = torch.zeros(self._num_layers, dtype=torch.int32, device=device)
        self._child_buf = torch.empty(
            self._num_layers, batch, seq_len, hidden_dim, dtype=dtype, device=device
        )

        if not self._prefill_parent(parent_branch_id):
            raise RuntimeError(
                f"Parent cache incomplete for branch={parent_branch_id}: the "
                "runner needs the parent embedding (layer 0) and every "
                "layer's ffn_out."
            )

        # Warm up on a side stream (compiles Triton kernels, allocates cuBLAS
        # workspaces) before capturing.
        self._count_buf.zero_()
        side_stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                self._static_forward()
        torch.cuda.current_stream().wait_stream(side_stream)

        self._count_buf.zero_()
        try:
            self._graph = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
            with torch.cuda.graph(self._graph):
                self._static_forward()
        except Exception as exc:
            self._graph = None
            raise RuntimeError(f"CUDA graph capture failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def replay(
        self,
        tokens: torch.Tensor,
        parent_branch_id: str,
        branch_id: str,
    ) -> torch.Tensor | None:
        """Replay the captured graph for one verification step.

        Args:
            tokens: Child token ids with the captured ``[batch, seq]`` shape.
            parent_branch_id: Parent branch whose cache entries are copied
                into the static parent buffers.
            branch_id: Child branch identifier (informational).

        Returns:
            The static logits buffer (the caller must ``.clone()`` before the
            next replay), or ``None`` when the parent cache is incomplete —
            the caller should fall back to the eager path.

        Raises:
            RuntimeError: If called before ``capture``.
            ValueError: If ``tokens`` has a different shape than the capture.
        """
        del branch_id
        self._require_captured()
        if tokens.shape != self._tokens_static.shape:  # type: ignore[union-attr]
            raise ValueError(
                f"replay tokens shape {tuple(tokens.shape)} differs from the "
                f"captured shape {tuple(self._tokens_static.shape)}"  # type: ignore[union-attr]
            )
        if not self._prefill_parent(parent_branch_id):
            return None
        self._tokens_static.copy_(tokens)  # type: ignore[union-attr]
        self._count_buf.zero_()  # type: ignore[union-attr]
        self._graph.replay()
        # The single per-step host readback: budget validation (T008 semantics
        # are layered on top by the caller, which discards on failure).
        self.validate_budgets()
        return self._logits_buf  # type: ignore[return-value]

    def validate_budgets(self) -> bool:
        """Check ``D_l = N - stable_count_l <= C`` for every captured layer.

        Performs exactly one host readback (``count_buf.tolist()``) and
        updates :attr:`budget_exceeded`.

        Returns:
            ``True`` when every layer's divergent count fits the fixed
            capacity; ``False`` when any layer exceeds it (the replay output
            for that step is untrustworthy and must be recomputed eagerly).
        """
        self._require_captured()
        counts = self._count_buf.tolist()  # type: ignore[union-attr]
        ok = all(self._num_tokens - count <= self._capacity for count in counts)
        self._budget_exceeded = not ok
        return ok

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_captured(self) -> None:
        if self._graph is None:
            raise RuntimeError("FoldedGraphRunner.capture() must succeed before this operation.")

    def _prefill_parent(self, parent_branch_id: str) -> bool:
        """Copy the parent branch activations into the static buffers.

        Returns ``False`` (without partial mutation guarantees) when any
        parent entry is missing.
        """
        parent_static = self._parent_static
        assert parent_static is not None, "static buffers allocated in capture()"
        try:
            embedding = self._cache.fetch(branch_id=parent_branch_id, layer_idx=0).get("embedding")
            if embedding is None:
                return False
            parent_static[0].copy_(embedding)
            for layer_idx in range(self._num_layers):
                ffn_out = self._cache.fetch(branch_id=parent_branch_id, layer_idx=layer_idx).get(
                    "ffn_out"
                )
                if ffn_out is None:
                    return False
                parent_static[layer_idx + 1].copy_(ffn_out)
        except (KeyError, RuntimeError):
            return False
        return True

    def _static_forward(self) -> torch.Tensor:
        """Run the static folded forward (capture body; pure GPU ops).

        Per layer: fused gate+mask+count against the static parent input ->
        fixed-capacity padded divergent index -> split hooks engaged ->
        full original-layer recompute (FFN on ``[C, hidden]`` rows) -> merge
        with the static parent FFN output -> child buffer.  Finally the final
        norm and head produce the static logits buffer.
        """
        tokens_static = self._tokens_static
        parent_static = self._parent_static
        mask_buf = self._mask_buf
        count_buf = self._count_buf
        child_buf = self._child_buf
        assert (
            tokens_static is not None
            and parent_static is not None
            and mask_buf is not None
            and count_buf is not None
            and child_buf is not None
        ), "static buffers allocated in capture()"

        x = self._embed_fn(tokens_static)
        for layer_idx, layer in enumerate(self._layers):
            mask = mask_buf[layer_idx]
            count = count_buf[layer_idx]
            fused_gate_mask_count(
                x,
                parent_static[layer_idx],
                self._gate_tau,
                self._gate_eps,
                mask,
                count,
            )
            use_split = isinstance(layer, SplitFoldedTransformerLayer) and layer.split_enabled
            layer_impl: Any = layer
            if use_split:
                layer_impl._split_state = {
                    "flat_index": _padded_divergent_index(mask, self._capacity),
                    "input_shape": None,
                    "num_divergent": 0,
                }
            try:
                child_out = layer_impl._recompute_all(x, self._attention_mask_static)
            finally:
                if use_split:
                    layer_impl._split_state = None
            merged = merge_stable_divergent(
                parent_static[layer_idx + 1],
                child_out,
                mask,
            )
            child_buf[layer_idx].copy_(merged)
            x = merged
        if self._final_norm_fn is not None:
            x = self._final_norm_fn(x)
        if self._head_fn is not None:
            x = self._head_fn(x)
        self._logits_buf = x
        return x
