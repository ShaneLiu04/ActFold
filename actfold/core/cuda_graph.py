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
import sys
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from actfold.core.fused_ops import fused_gate_mask_count, merge_stable_divergent
from actfold.core.layer_budget import apply_layer_budget, broadcast_forced_to_mask
from actfold.core.similarity_gate import SimilarityGate
from actfold.core.split_layer import SplitFoldedTransformerLayer, _padded_divergent_index

__all__ = ["CaptureDiagnostics", "FoldedGraphRunner", "safe_capture"]


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
        position_embeddings_static: Optional rotary ``(cos, sin)`` tuple
            baked BY REFERENCE into every capture/replay (AR019 — never
            cloned, mirroring ``attention_mask_static``); the constant-
            position scheme (``arange(T)``, no KV cache) makes the values
            shape-fixed so a single reference serves every replay. Pass
            ``None`` when the wrapped layers need no rotary embeddings.
        position_ids_static: Optional ``position_ids`` tensor baked BY
            REFERENCE into every capture/replay (AR023, mirroring the AR019
            ``position_embeddings_static`` contract); the Dream family pins
            one ``compute_position_ids`` payload per ``(T, prompt_len)``
            fold context, so a single reference serves every replay. Pass
            ``None`` when the wrapped layers need no explicit positions.
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
        position_embeddings_static: tuple[torch.Tensor, torch.Tensor] | None = None,
        forced_divergent_static: torch.Tensor | None = None,
        position_ids_static: torch.Tensor | None = None,
        budget_static: list[torch.Tensor] | None = None,
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
        self._position_embeddings_static = position_embeddings_static
        self._position_ids_static = position_ids_static
        self._forced_divergent_static = (
            forced_divergent_static.detach().clone()
            if forced_divergent_static is not None
            else None
        )
        if budget_static is not None:
            if len(budget_static) != len(wrapped_layers):
                raise ValueError(
                    f"budget_static length {len(budget_static)} != wrapped layer "
                    f"count {len(wrapped_layers)}"
                )
            # AR031 A4: the budget face owns its own gate so the scores path
            # bypasses the fused kernel (#38) with the identical cosine math.
            self._budget_static: list[torch.Tensor] | None = list(budget_static)
            self._budget_gate: SimilarityGate | None = SimilarityGate(
                tau=self._gate_tau, metric="cosine", eps=self._gate_eps
            )
        else:
            self._budget_static = None
            self._budget_gate = None

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
        # Divergence staging (AR017): per-layer divergent counts from the
        # last validate_budgets call; None until the first validation (a
        # capture without a replay stages nothing).
        self._last_layer_divergences: list[int] | None = None

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

    @property
    def last_layer_divergences(self) -> list[int] | None:
        """Per-layer divergent counts staged by the last budget validation.

        ``None`` until the first :meth:`validate_budgets` call (a capture
        without a replay stages nothing); refreshed on every validation
        (replays and explicit calls alike), never left stale. Readback-free
        by contract: the values reuse the single ``count_buf.tolist()``
        inside :meth:`validate_budgets` -- no extra host sync (AR017 FR1).
        """
        return self._last_layer_divergences

    @property
    def forced_divergent_static(self) -> torch.Tensor | None:
        """Static forced-divergence buffer, or ``None`` (AR021 FR2).

        ``None`` when the runner was captured without a ``forced_divergent``
        argument; otherwise a fixed-shape boolean buffer whose content is
        refreshed by :meth:`replay`` before every graph launch (outside the
        captured body, zero host sync). The None/not-None identity plus the
        shape form the runner's forced capture identity: the caller routes
        mismatching calls to the eager path instead of replaying.
        """
        return self._forced_divergent_static

    @property
    def budget_static(self) -> list[torch.Tensor] | None:
        """Per-layer budget scalars baked by reference (AR031 A4).

        ``None`` when the runner was captured without a budget; otherwise a
        fixed-length list of 0-dim device tensors whose content is refreshed
        by the caller OUTSIDE the capture body (``fill_`` before every
        replay, zero host sync) so the in-graph A1 cap reads fresh per-layer
        budgets on every launch.
        """
        return self._budget_static

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
        if self._mask_buf is None:
            raise RuntimeError(
                "Static mask buffers not allocated: capture() (or the "
                "manual buffer-allocation direct-drive face) must run first."
            )
        return self._mask_buf

    @property
    def count_buf(self) -> torch.Tensor:
        """Per-layer stable counts ``[L]`` (int32)."""
        if self._count_buf is None:
            raise RuntimeError(
                "Static count buffer not allocated: capture() (or the "
                "manual buffer-allocation direct-drive face) must run first."
            )
        return self._count_buf

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
        forced_divergent: torch.Tensor | None = None,
    ) -> None:
        """Warm up and capture the static folded forward as a CUDA graph.

        Args:
            tokens: Child token ids ``[batch, seq]`` fixing the captured
                shape.
            branch_id: Child branch identifier (informational; the captured
                graph is branch-agnostic).
            parent_branch_id: Parent branch whose cache entries feed the
                static parent buffers.
            forced_divergent: Optional boolean forced-divergence tensor
                (AR021 FR2) fixing the forced capture identity; when given,
                a static buffer of the same shape is allocated (outside the
                graph pool — the host copies into it between replays) and
                the captured body ANDs it into every layer's stability mask
                before re-deriving the stable count device-side. ``None``
                captures the legacy forced-free body byte-for-byte.

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
        if forced_divergent is not None and self._forced_divergent_static is None:
            # AR021 FR2: allocate the static forced buffer OUTSIDE the graph
            # pool (the host copy_fills it between replays); the shape is
            # pinned for the runner's lifetime and forms its forced capture
            # identity together with the not-None bit.
            self._forced_divergent_static = forced_divergent.detach().clone()

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
        forced_divergent: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Replay the captured graph for one verification step.

        Args:
            tokens: Child token ids with the captured ``[batch, seq]`` shape.
            parent_branch_id: Parent branch whose cache entries are copied
                into the static parent buffers.
            branch_id: Child branch identifier (informational).
            forced_divergent: Optional forced-divergence tensor with the
                captured forced shape (AR021 FR2). Its content is copied
                into the static buffer here — OUTSIDE the captured body, a
                pure device-to-device copy with zero host sync — so the
                replayed graph consumes fresh forced bits every step.
                Ignored when the runner was captured without one.

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
        if self._forced_divergent_static is not None and forced_divergent is not None:
            self._forced_divergent_static.copy_(forced_divergent)
        self._tokens_static.copy_(tokens)  # type: ignore[union-attr]
        self._count_buf.zero_()  # type: ignore[union-attr]
        self._graph.replay()
        # The single per-step host readback: budget validation (T008 semantics
        # are layered on top by the caller, which discards on failure).
        self.validate_budgets()
        return self._logits_buf  # type: ignore[return-value]

    def validate_budgets(self) -> bool:
        """Check ``D_l = N - stable_count_l <= C`` for every captured layer.

        Performs exactly one host readback (``count_buf.tolist()``), updates
        :attr:`budget_exceeded`, and stages the per-layer divergent counts in
        :attr:`last_layer_divergences` (AR017 FR1 -- the staging reuses this
        readback; reading it costs no extra host sync).

        Returns:
            ``True`` when every layer's divergent count fits the fixed
            capacity; ``False`` when any layer exceeds it (the replay output
            for that step is untrustworthy and must be recomputed eagerly).
        """
        self._require_captured()
        counts = self._count_buf.tolist()  # type: ignore[union-attr]
        self._last_layer_divergences = [self._num_tokens - count for count in counts]
        ok = all(divergence <= self._capacity for divergence in self._last_layer_divergences)
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
        budget_static = self._budget_static
        budget_gate = self._budget_gate
        for layer_idx, layer in enumerate(self._layers):
            mask = mask_buf[layer_idx]
            count = count_buf[layer_idx]
            budget_scores: torch.Tensor | None = None
            if budget_static is not None:
                # AR031: the budget face derives the mask from the scores
                # face, bypassing the fused kernel (#38) -- the cap consumes
                # the same scores and the plain gate is exactly
                # ``scores > tau`` (bit-identical, UT-3120).
                assert budget_gate is not None, "budget gate built with budget_static"
                budget_scores = budget_gate.scores(x, parent_static[layer_idx])
                mask.copy_(budget_scores > self._gate_tau)
            else:
                fused_gate_mask_count(
                    x,
                    parent_static[layer_idx],
                    self._gate_tau,
                    self._gate_eps,
                    mask,
                    count,
                )
            if self._forced_divergent_static is not None:
                # AR021 FR2: AND the forced region into the gate mask and
                # re-derive the stable count from the AND-ed mask with a
                # fixed-shape device reduction (capturable, no host sync) —
                # the kernel count was computed pre-AND and would let forced
                # bits through the divergent budget validation.
                mask &= ~self._forced_divergent_static
            if budget_scores is not None:
                # AR031 A1: cap AFTER the forced AND (forced rows are exempt
                # by construction, #43) -- the most-similar non-exempt
                # divergent rows fold back to stable until the per-layer
                # recompute count fits the refreshed budget scalar.  The
                # static forced buffer may be position-level ``[T]`` (#43
                # production form); broadcast it to the mask shape so the
                # frozen same-shape cap contract holds (UT-3165).
                capped = apply_layer_budget(
                    mask,
                    budget_scores,
                    budget_static[layer_idx],  # type: ignore[index]
                    forced=broadcast_forced_to_mask(mask, self._forced_divergent_static),
                )
                mask.copy_(capped)
            if self._forced_divergent_static is not None or budget_scores is not None:
                # One shared final reduction over the post-AND/post-cap mask:
                # the kernel count is pre-AND/pre-cap and would under-count
                # the divergent demand (UT-3114: count == mask.sum()).
                count.copy_(mask.sum())
            use_split = isinstance(layer, SplitFoldedTransformerLayer) and layer.split_enabled
            layer_impl: Any = layer
            if use_split:
                layer_impl._split_state = {
                    "flat_index": _padded_divergent_index(mask, self._capacity),
                    "input_shape": None,
                    "num_divergent": 0,
                }
            try:
                # AR019/AR023: attach the baked rotary tuple / position_ids
                # as layer kwargs ONLY when supplied (a ``None`` value must
                # not shadow a layer's own rotary/position computation);
                # ``_recompute_all`` filters each kwarg against the original
                # layer's signature.
                extra_kwargs: dict[str, Any] = {}
                if self._position_embeddings_static is not None:
                    extra_kwargs["position_embeddings"] = self._position_embeddings_static
                if self._position_ids_static is not None:
                    extra_kwargs["position_ids"] = self._position_ids_static
                child_out = layer_impl._recompute_all(
                    x,
                    self._attention_mask_static,
                    **extra_kwargs,
                )
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


# ---------------------------------------------------------------------------
# AR026: capture root-cause fidelity (design.md srs.md FR1)
# ---------------------------------------------------------------------------

# The production capture boundary must never let the capture-unwinding
# ``captures_underway.empty() INTERNAL ASSERT`` secondary failure mask the
# primary exception (the AR023 family-probe llada graph-face signature: a
# primary failure inside the capture body makes ``capture_end``/the caching
# allocator assert during unwinding, and the assert is what surfaces).  The
# diagnostics below record BOTH faces; ``safe_capture`` then re-raises the
# ORIGINAL exception unchanged so the call-site degradation semantics stay
# byte-identical.

_CAPTURE_DIAGNOSTICS_MAX = 8
_capture_diagnostics: "list[CaptureDiagnostics]" = []
_capture_diagnostics_overflow = 0


@dataclass(frozen=True)
class CaptureDiagnostics:
    """Serializable record of one production capture attempt (AR026 FR1).

    ``secondary_error`` is the failure that crossed the capture boundary (the
    one the call site's ``except`` sees); ``primary_error`` is the deepest
    exception in its ``__context__`` chain — the failure that actually
    happened inside the capture body before the unwinding assert, or ``None``
    when the boundary failure has no context (``reset_error`` similarly
    records a failed best-effort post-failure ``synchronize``).  VRAM fields
    are ``None`` on the CPU face (no CUDA call is ever made there).
    """

    ok: bool
    stage: str
    primary_error: str | None
    secondary_error: str | None
    reset_error: str | None
    vram_free_mb: float | None
    vram_total_mb: float | None
    platform: str

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of this record."""
        return {
            "ok": self.ok,
            "stage": self.stage,
            "primary_error": self.primary_error,
            "secondary_error": self.secondary_error,
            "reset_error": self.reset_error,
            "vram_free_mb": self.vram_free_mb,
            "vram_total_mb": self.vram_total_mb,
            "platform": self.platform,
        }


def _deepest_context(exc: BaseException) -> BaseException | None:
    """Walk the ``__context__`` chain to the oldest in-flight exception."""
    current = exc.__context__
    if current is None:
        return None
    while current.__context__ is not None:
        current = current.__context__
    return current


def _platform_stamp() -> str:
    return f"{sys.platform}/torch{torch.__version__}"


def _attempt_reset() -> str | None:
    """Best-effort post-failure device reset (never replaces the failure)."""
    if not torch.cuda.is_available():
        return None
    try:
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - diagnostics-only best effort
        return f"{type(exc).__name__}: {exc}"
    return None


def _vram_snapshot() -> tuple[float | None, float | None]:
    """Free/total VRAM in MB; ``(None, None)`` on the CPU face."""
    if not torch.cuda.is_available():
        return None, None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - diagnostics-only best effort
        return None, None
    return free_bytes / (1024 * 1024), total_bytes / (1024 * 1024)


def _record_capture_diagnostics(diagnostics: CaptureDiagnostics) -> None:
    """Keep the FIRST ``_CAPTURE_DIAGNOSTICS_MAX`` failure records, then count."""
    global _capture_diagnostics_overflow
    if len(_capture_diagnostics) < _CAPTURE_DIAGNOSTICS_MAX:
        _capture_diagnostics.append(diagnostics)
    else:
        _capture_diagnostics_overflow += 1


def capture_diagnostics_snapshot() -> list[dict[str, Any]]:
    """Copies of the recorded failure diagnostics (mutations never leak)."""
    return [record.to_json_dict() for record in _capture_diagnostics]


def capture_diagnostics_overflow() -> int:
    """Failures recorded past the first-``_CAPTURE_DIAGNOSTICS_MAX`` cap."""
    return _capture_diagnostics_overflow


def _reset_capture_diagnostics() -> None:
    """Restore the pristine accumulator state (test isolation only)."""
    global _capture_diagnostics_overflow
    _capture_diagnostics.clear()
    _capture_diagnostics_overflow = 0


def safe_capture(runner: Any, stage: str, *args: Any, **kwargs: Any) -> CaptureDiagnostics:
    """Capture through ``runner.capture`` with root-cause fidelity (AR026).

    Forwards ``*args``/``**kwargs`` verbatim to ``runner.capture(*args,
    **kwargs)``.  On success returns a ``CaptureDiagnostics`` with ``ok=True``
    and records nothing (the success path adds zero behavior beyond the
    returned artifact).  On failure extracts the primary exception from the
    ``__context__`` chain, takes a best-effort VRAM snapshot and device reset,
    records the diagnostics, and then RE-RAISES the original exception object
    unchanged — the call site's existing ``except`` semantics are
    byte-identical whether or not it routes through this wrapper.
    """
    try:
        runner.capture(*args, **kwargs)
    except BaseException as exc:
        primary = _deepest_context(exc)
        reset_error = _attempt_reset()
        vram_free_mb, vram_total_mb = _vram_snapshot()
        _record_capture_diagnostics(
            CaptureDiagnostics(
                ok=False,
                stage=stage,
                primary_error=(
                    f"{type(primary).__name__}: {primary}" if primary is not None else None
                ),
                secondary_error=f"{type(exc).__name__}: {exc}",
                reset_error=reset_error,
                vram_free_mb=vram_free_mb,
                vram_total_mb=vram_total_mb,
                platform=_platform_stamp(),
            )
        )
        raise
    return CaptureDiagnostics(
        ok=True,
        stage=stage,
        primary_error=None,
        secondary_error=None,
        reset_error=None,
        vram_free_mb=None,
        vram_total_mb=None,
        platform=_platform_stamp(),
    )
