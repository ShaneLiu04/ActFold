"""Folded Transformer layer with cross-branch activation reuse."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from actfold.core.adaptive_gate import AdaptiveQuantileGate
from actfold.core.cache_factory import ActivationCacheType
from actfold.core.folding_context import FOLDING_CONTEXT
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.fused_ops import (
    _merge_stable_divergent_torch,
    fused_gate_mask_count,
    gather_select,
    merge_stable_divergent,
)
from actfold.core.similarity_gate import SimilarityGate
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

# Keywords that belong to ActFold and must never be forwarded to the original
# Transformer layer.
_ACTFOLD_KWARGS = {"branch_id", "branch_ids", "parent_branch_id", "step_idx"}

# AR009: AdaptiveQuantileGate selects the top-k most similar tokens and
# ignores fixed thresholds, so a FoldingScheduler overwriting ``gate.tau``
# would silently have no effect. Every constructor accepting a gate plus a
# scheduler rejects the combination with this exact message (AGENTS #29).
_ADAPTIVE_GATE_SCHEDULER_MSG = (
    "AdaptiveQuantileGate cannot be combined with FoldingScheduler: the "
    "scheduler overwrites gate.tau, but the quantile gate ignores fixed "
    "thresholds (AGENTS #29). Use SimilarityGate with the scheduler, or "
    "pass scheduler=None with AdaptiveQuantileGate."
)

# D3 thresholds for the fused single-pass gather/select merge (F8): below
# these shapes the kernel launch overhead outweighs the saved memory pass, so
# the original ``fetch`` + ``merge_stable_divergent`` path is used.
_GATHER_SELECT_MIN_TOKENS = 2048
_GATHER_SELECT_MIN_HIDDEN = 4096
_GATHER_SELECT_REQUIRE_CUDA = True


def _apply_cache_availability(
    layer: "FoldedTransformerLayer",
    parent_branch_id: str,
    stable_mask: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """AND the target-layer availability of a sentinel cache into ``stable_mask``.

    Caches without ``availability`` (every legacy cache) are a no-op.  Folding
    stores/reads entries at step 0, so the query uses the same key as
    ``put``/``fetch``.  A 1-D mask (AR028 per-row path) stays 1-D.

    Returns:
        ``(mask, applied)``; ``applied`` tells the caller to invalidate any
        fused-gate count buffer derived from the pre-guard mask.
    """
    availability = getattr(layer.cache, "availability", None)
    if availability is None:
        return stable_mask, False
    avail = availability(parent_branch_id, layer.layer_idx, int(stable_mask.shape[-1]), 0)
    if avail is None:
        return stable_mask, False
    avail = avail.to(device=stable_mask.device)
    if stable_mask.ndim == 1:
        if avail.ndim != 1:
            avail = avail.reshape(-1)
        return stable_mask & avail, True
    if avail.ndim == 1:
        avail = avail.view(1, -1)
    if avail.shape[0] == 1 and stable_mask.shape[0] > 1:
        avail = avail.expand(stable_mask.shape[0], -1)
    elif avail.shape[0] != stable_mask.shape[0]:
        avail = avail.expand(stable_mask.shape[0], -1)
    return stable_mask & avail, True


class FoldedTransformerLayer(nn.Module):
    """A wrapped Transformer layer that reuses parent activations where stable.

    The wrapped ``original_layer`` must accept ``hidden_states`` and optional
    ``attention_mask`` and return the updated hidden states. This wrapper
    intercepts the forward pass, partitions tokens into stable/divergent sets,
    and merges cached parent activations with freshly computed outputs.

    For divergent tokens, the layer is recomputed on the full child hidden
    states so self-attention context is identical to the baseline; only the
    divergent token positions are then written into the output buffer. Stable
    positions copy the cached parent FFN output.

    Args:
        original_layer: The base Transformer layer to wrap.
        cache: Activation cache for parent activations.
        gate: Similarity gate for token partitioning.
        layer_idx: Index of this layer in the model.
        scheduler: Optional folding scheduler for dynamic tau / per-layer
            folding decisions.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        cache: ActivationCacheType,
        gate: SimilarityGate,
        layer_idx: int,
        scheduler: FoldingScheduler | None = None,
        multi_parent_batch: bool = False,
    ) -> None:
        super().__init__()
        if scheduler is not None and isinstance(gate, AdaptiveQuantileGate):
            raise RuntimeError(_ADAPTIVE_GATE_SCHEDULER_MSG)
        self.original_layer = original_layer
        self.cache = cache
        self.gate = gate
        self.layer_idx = layer_idx
        self.scheduler = scheduler
        # AR025 DD5/DD6: the single behavior switch. Off (default): the
        # eager batched publish stores per row (byte-for-byte the AR015
        # loop). On: a batch-capable cache takes ONE put_batch per layer
        # (N entries, branch_ids order); an incapable cache keeps the
        # per-row loop unchanged.
        self.multi_parent_batch = bool(multi_parent_batch)
        # Set on the first recomputation: some architectures (e.g. LLaDA) return
        # ``(hidden_states, cache)`` tuples that the base model unpacks.
        self._returns_tuple: bool | None = None
        # Reflection is moved out of the hot path (F4d): the original layer's
        # forward signature is inspected exactly once, at construction.
        try:
            sig = inspect.signature(self.original_layer.forward)
            self._layer_params: frozenset[str] = frozenset(sig.parameters)
            self._layer_has_varkw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            self._layer_params = frozenset()
            self._layer_has_varkw = False
        # Fast-path parent fetch (F4b/F7): caches exposing ``get_all`` are
        # dispatched via ``getattr`` at call time (see ``forward``); all
        # caches implement the raw ``fetch`` protocol method, so no all-ones
        # mask is ever allocated for the gate comparison.

    def forward(
        self,
        hidden_states: torch.Tensor,
        branch_id: str | None = None,
        parent_branch_id: str | None = None,
        attention_mask: torch.Tensor | None = None,
        step_idx: int = 0,
        *,
        branch_ids: Sequence[str] | None = None,
        forced_divergent: torch.Tensor | None = None,
        parent_branch_ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        """Run one folded Transformer layer.

        Args:
            hidden_states: Child hidden states ``[batch, seq_len, hidden_dim]``.
            branch_id: Identifier of the current child branch. If not provided,
                the layer reads the thread-local folding context set by
                :class:`~actfold.core.model_wrapper.FoldedModel`.
            parent_branch_id: Identifier of the parent branch. If None, falls
                back to a full recomputation through ``original_layer``.
            attention_mask: Optional attention mask.
            step_idx: Current diffusion step index.
            branch_ids: Batched-branch identifiers (AR015): when set, row
                ``i`` of ``hidden_states`` belongs to ``branch_ids[i]`` and
                every branch shares ``parent_branch_id`` (which must be
                cached at batch 1). Mutually exclusive with ``branch_id``.
            forced_divergent: Optional boolean tensor (AR021 FR2) whose True
                positions are forced divergent in the stability partition
                regardless of the gate: the materialized ``stable_mask`` is
                AND-ed with its complement before the profiler record and
                the stable-count consumption, so forced bits never take the
                all-stable reuse fast path. ``[seq_len]`` / ``[1, seq_len]``
                on serial calls (broadcast), ``[N, seq_len]`` on batched
                calls. ``None`` (the default) attaches no operation at all.
            parent_branch_ids: Per-row parent identifiers (AR025): when set,
                row ``i`` folds against ``parent_branch_ids[i]`` — the
                per-row generalization of the shared-``parent_branch_id``
                expand contract. Mutually exclusive with
                ``parent_branch_id``; requires ``branch_ids``. v1
                equal-length semantics: a parent whose length differs from
                the child canvas degrades its row to a full recompute (no
                AR007 truncation on this face). ``None`` (the default)
                attaches nothing — the single-parent path is byte-for-byte
                unchanged.
            **kwargs: Extra arguments forwarded to ``original_layer``
                (filtered to its signature). This channel carries
                ``position_embeddings`` (AR019): a rotary ``(cos, sin)``
                tuple supplied by ``ManualFoldedForward.forward`` reaches
                layers that accept it and is silently dropped by synthetic
                layers that do not (no layer-interface change).

        Returns:
            Updated hidden states ``[batch, seq_len, hidden_dim]``.
        """
        if branch_ids is not None:
            if branch_id is not None:
                raise ValueError(
                    "FoldedTransformerLayer accepts either branch_id "
                    "(serial) or branch_ids (batched), not both."
                )
            return self._forward_batched(
                hidden_states,
                attention_mask,
                list(branch_ids),
                parent_branch_id,
                step_idx,
                forced_divergent=forced_divergent,
                parent_branch_ids=parent_branch_ids,
                **kwargs,
            )
        # Resolve branch identifiers from explicit kwargs or the thread-local
        # context. Explicit kwargs take precedence.
        if branch_id is None:
            ctx = FOLDING_CONTEXT.get()
            if ctx is not None:
                branch_id = ctx["branch_id"]
                parent_branch_id = ctx.get("parent_branch_id", parent_branch_id)
                step_idx = ctx.get("step_idx", step_idx)

        if branch_id is None:
            raise ValueError(
                "FoldedTransformerLayer requires a branch_id either as an "
                "argument or via the ActFold folding context."
            )

        if parent_branch_id is None:
            # No parent to reuse from; compute normally.
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_activations(branch_id, output, hidden_states)
            return self._pack_output(output)

        # Respect the scheduler if one is attached: disabled layers recompute.
        if self.scheduler is not None and not self.scheduler.should_fold(self.layer_idx, step_idx):
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_activations(branch_id, output, hidden_states)
            return self._pack_output(output)

        # Apply dynamic tau when a scheduler is configured.
        if self.scheduler is not None:
            tau = self.scheduler.get_tau(
                layer_idx=self.layer_idx,
                step_idx=step_idx,
                task_type="general",
            )
            self.gate.set_tau(tau)

        # Retrieve the parent activation for the similarity comparison (F6):
        # the parent's input to layer L equals the parent's layer-(L-1) FFN
        # output in sequential residual models, so the gate reads
        # ``ffn_out`` from layer L-1; layer 0 reads the cached ``embedding``.
        # If not cached, fall back to recomputation.  (F7: ``get_all``/``fetch``
        # replace the masked ``get`` so no all-ones mask is ever allocated.)
        gate_layer = 0 if self.layer_idx == 0 else self.layer_idx - 1
        gate_name = "embedding" if self.layer_idx == 0 else "ffn_out"
        try:
            get_all_fn = getattr(self.cache, "get_all", None)
            if get_all_fn is not None:
                parent_activations = get_all_fn(
                    branch_id=parent_branch_id,
                    layer_idx=gate_layer,
                )
            else:
                parent_activations = self.cache.fetch(
                    branch_id=parent_branch_id,
                    layer_idx=gate_layer,
                )
            h_parent = parent_activations.get(gate_name)
        except (KeyError, RuntimeError):
            h_parent = None

        parent_longer = False
        if h_parent is not None and (
            h_parent.dim() != 3
            or h_parent.shape[0] != hidden_states.shape[0]
            or h_parent.shape[2] != hidden_states.shape[2]
            or h_parent.shape[1] == 0
        ):
            # The parent cannot donate a prefix-aligned activation: a batch
            # or hidden-dim mismatch or a zero-length entry.  Treat the
            # parent as absent and recompute — identical semantics to a
            # cache miss, never a hard error.
            h_parent = None
        elif h_parent is not None and h_parent.shape[1] > hidden_states.shape[1]:
            # AR007 truncated-prefix reuse: a parent LONGER than the child
            # donates its first ``seq_len`` positions.  Causal models are
            # bit-identical on a true prefix, so the gate certifies them
            # all-stable and the parent FFN output is reused after the same
            # truncation; non-prefix children diverge at the gate and are
            # recomputed (the similarity gate is the semantic safety net).
            h_parent = h_parent[:, : hidden_states.shape[1]]
            parent_longer = True

        if h_parent is None:
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_activations(branch_id, output, hidden_states)
            return self._pack_output(output)

        # Align cached parent hidden states to the child's device/dtype before
        # the similarity comparison.
        h_parent = h_parent.to(dtype=hidden_states.dtype, device=hidden_states.device)

        batch, seq_len, _ = hidden_states.shape
        prefix_len = h_parent.shape[1]

        # Compute stability mask entirely on GPU.  When the layer runs the
        # standard cosine gate, the mask and the stable count come out of the
        # single fused kernel (AR002/T006): the count buffer replaces the
        # separate ``mask.sum()`` reduction readback below.
        stable_count_buf: torch.Tensor | None = None
        if prefix_len == seq_len:
            # Equal-length folding: the existing semantics, bit for bit.
            if type(self.gate) is SimilarityGate and self.gate.metric == "cosine":
                stable_mask = torch.empty(
                    hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
                )
                stable_count_buf = torch.empty((), dtype=torch.int64, device=hidden_states.device)
                # ``fill_(0)`` instead of ``torch.zeros``: the split-merge path
                # forbids fresh zero-fill allocations (T017) and the launch cost
                # is identical (zeros is empty + fill internally).
                stable_count_buf.fill_(0)
                fused_gate_mask_count(
                    hidden_states,
                    h_parent,
                    self.gate.tau,
                    self.gate.eps,
                    stable_mask,
                    stable_count_buf,
                )
            else:
                stable_mask = self.gate(hidden_states, h_parent)  # [batch, seq_len]
        else:
            # Variable-length prefix folding (AR003): compare only the child's
            # prefix against the parent and append an all-False suffix — the
            # suffix has no parent activation to reuse, so it is divergent for
            # every tau (an explicit False tail stays robust even for
            # negative thresholds, unlike a zero-padded parent).  The plain
            # gate is stride-aware and handles the non-contiguous prefix
            # slice; the fused kernel's same-shape contiguous contract and
            # the ``B*T >= 1024`` dispatch threshold are both inapplicable on
            # variable-length steps (design decision D2).
            prefix_mask = self.gate(hidden_states[:, :prefix_len], h_parent)
            stable_mask = torch.cat(
                [
                    prefix_mask,
                    torch.zeros(
                        batch,
                        seq_len - prefix_len,
                        dtype=torch.bool,
                        device=hidden_states.device,
                    ),
                ],
                dim=1,
            )

        if forced_divergent is not None:
            # AR021 FR2 single ordering: AND the forced region into the
            # materialized mask BEFORE the profiler record and the count
            # consumption, and discard the fused kernel's stale count (it
            # was derived from the pre-AND mask) so the fallback
            # ``stable_mask.sum()`` reduction re-derives it.
            stable_mask &= ~forced_divergent
            stable_count_buf = None

        # AR029: a sentinel cache (pyramid) may have evicted rows of the
        # TARGET layer while the gate read layer L-1.  AND the target-layer
        # availability before the profiler/count/fast paths so a hole always
        # recomputes.  Legacy caches are a no-op.
        stable_mask, cache_guarded = _apply_cache_availability(self, parent_branch_id, stable_mask)
        if cache_guarded:
            stable_count_buf = None

        # Record real layer-wise stability statistics for downstream consumers.
        GLOBAL_STABILITY_PROFILER.record(
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            layer_idx=self.layer_idx,
            step_idx=step_idx,
            stable_mask=stable_mask,
            tau=self.gate.tau,
            metric=self.gate.metric,
        )

        # Three-way split with a single host sync (F4a): one ``sum`` readback
        # covers all-stable / none-stable / mixed; the old ``.all()`` +
        # ``.any()`` pair cost two syncs on the mixed path.  The fused-gate
        # path (AR002/T006) already carries the count in a scalar buffer, so
        # no extra reduction kernel is launched at all.
        stable_count = (
            int(stable_count_buf) if stable_count_buf is not None else int(stable_mask.sum())
        )
        num_tokens = stable_mask.numel()

        # Fast path: all tokens stable -> reuse the cached parent FFN output
        # directly (F7: raw ``fetch``; no mask application, no zero-fill).
        if stable_count == num_tokens:
            try:
                parent_activations = self.cache.fetch(
                    branch_id=parent_branch_id,
                    layer_idx=self.layer_idx,
                )
            except (KeyError, RuntimeError):
                parent_activations = None
            ffn_out = parent_activations.get("ffn_out") if parent_activations is not None else None
            if (
                ffn_out is not None
                and ffn_out.dim() == 3
                and ffn_out.shape[1] > hidden_states.shape[1]
            ):
                # AR007: a parent FFN longer than the child is truncated to
                # the child length before the reuse check.  The shape test is
                # deliberately on the fetched tensor (not the ``parent_longer``
                # flag): a cache whose per-layer entries disagree in length
                # (an anomalous state) still resolves deterministically here.
                ffn_out = ffn_out[:, : hidden_states.shape[1]]
            if (
                ffn_out is not None
                and ffn_out.shape == hidden_states.shape
                and self._returns_tuple is not None
            ):
                self._store_activations(branch_id, ffn_out, hidden_states)
                contains_fn = getattr(self.cache, "contains", None)
                if (
                    parent_activations is not None
                    and contains_fn is not None
                    and not contains_fn(parent_branch_id, self.layer_idx)
                ):
                    self.cache.put(
                        branch_id=parent_branch_id,
                        layer_idx=self.layer_idx,
                        activations=parent_activations,
                    )
                return self._pack_output(ffn_out)
            # Parent FFN output is missing or shape-mismatched despite all
            # tokens being stable, or the original layer's output structure is
            # not known yet (first call): recompute the whole layer to stay
            # consistent.
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_activations(branch_id, output, hidden_states)
            return self._pack_output(output)

        # If no tokens are stable, there is nothing to reuse. Recompute the full
        # layer to avoid fetching a parent FFN output that may not be cached.
        if stable_count == 0:
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_activations(branch_id, output, hidden_states)
            return self._pack_output(output)

        # Slow path: recompute divergent tokens using full child attention context,
        # then fuse cached parent activations with freshly computed outputs.
        # ``_recompute_merged`` is an extension point: the split-layer subclass
        # uses it to skip the FFN for stable tokens.  The already-synced
        # ``stable_count`` is forwarded so the subclass derives the divergent
        # row count for free (AR002/T001, no second host readback).
        merged_kwargs = {**kwargs, "stable_count": stable_count}
        child_out = self._recompute_merged(
            hidden_states,
            attention_mask,
            stable_mask,
            **merged_kwargs,
        )
        # ``prefix_len`` is None on the equal-length path (unchanged merge
        # semantics); on variable-length steps it routes the parent FFN
        # through prefix alignment before the merge (AR003).
        merge_prefix_len = None if prefix_len == seq_len else prefix_len
        h_out = self._merge_parent_child(
            parent_branch_id,
            stable_mask,
            child_out,
            prefix_len=merge_prefix_len,
            parent_longer=parent_longer,
        )

        # Store child activations for future reuse.
        self._store_activations(branch_id, h_out, hidden_states)

        # Residual + layer norm are folded into original_layer in real models.
        # For this generic wrapper we assume original_layer already handles them.
        return self._pack_output(h_out)

    def _forward_batched(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        branch_ids: list[str],
        parent_branch_id: str | None,
        step_idx: int,
        forced_divergent: torch.Tensor | None = None,
        *,
        parent_branch_ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        """Run one folded layer for N same-parent branches in a single batch.

        Batched-branch semantics (AR015, P1-7 first tranche): row ``i`` of
        ``hidden_states`` belongs to ``branch_ids[i]``; every branch shares
        ``parent_branch_id``, whose cached activations must be stored at
        batch 1 (the generate-loop shape) and are expanded along the batch
        dimension. Every branch mirrors the serial per-child path
        bit-for-bit: parent fetch miss / scheduler-disabled / no-parent
        recompute the whole batch, the gate runs on the plain (non-fused)
        path with the expanded parent, the three-way split is evaluated at
        batch granularity, and stores/profiler records happen per branch
        on row slices. When ``parent_branch_ids`` is set (AR025), the call
        delegates to :meth:`_forward_batched_multi` and this shared-parent
        body is untouched (byte-for-byte).

        Args:
            hidden_states: Batched child hidden states ``[N, seq_len, hidden]``.
            attention_mask: Optional attention mask.
            branch_ids: One branch identifier per row (``len == N``).
            parent_branch_id: Shared parent branch identifier.
            step_idx: Current diffusion step index.
            forced_divergent: Optional ``[N, seq_len]`` boolean tensor
                (AR021 FR2): True positions are forced divergent before the
                profiler records and the stable-count consumption.
            parent_branch_ids: Per-row parent identifiers (AR025): row ``i``
                folds against ``parent_branch_ids[i]``. Mutually exclusive
                with ``parent_branch_id``.
            **kwargs: Extra arguments forwarded to ``original_layer``.

        Returns:
            Updated hidden states ``[N, seq_len, hidden]``.
        """
        batch, seq_len, _ = hidden_states.shape
        if len(branch_ids) != batch:
            raise ValueError(
                f"branch_ids length {len(branch_ids)} does not match the "
                f"hidden-states batch {batch}."
            )
        if parent_branch_ids is not None:
            if parent_branch_id is not None:
                raise ValueError(
                    "FoldedTransformerLayer accepts either parent_branch_id "
                    "(shared) or parent_branch_ids (per-row), not both."
                )
            if len(parent_branch_ids) != batch:
                raise ValueError(
                    f"parent_branch_ids length {len(parent_branch_ids)} does "
                    f"not match the hidden-states batch {batch}."
                )
            return self._forward_batched_multi(
                hidden_states,
                attention_mask,
                branch_ids,
                list(parent_branch_ids),
                step_idx,
                forced_divergent=forced_divergent,
                **kwargs,
            )
        if parent_branch_id is None:
            # No parent to reuse from: whole-batch recompute (serial
            # no-parent semantics), stored per branch.
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_batched(branch_ids, output, hidden_states)
            return self._pack_output(output)
        if self.scheduler is not None and not self.scheduler.should_fold(self.layer_idx, step_idx):
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_batched(branch_ids, output, hidden_states)
            return self._pack_output(output)
        if self.scheduler is not None:
            tau = self.scheduler.get_tau(
                layer_idx=self.layer_idx,
                step_idx=step_idx,
                task_type="general",
            )
            self.gate.set_tau(tau)

        # Parent gate activation fetch (serial gate-layer semantics).
        gate_layer = 0 if self.layer_idx == 0 else self.layer_idx - 1
        gate_name = "embedding" if self.layer_idx == 0 else "ffn_out"
        try:
            get_all_fn = getattr(self.cache, "get_all", None)
            if get_all_fn is not None:
                parent_activations = get_all_fn(
                    branch_id=parent_branch_id,
                    layer_idx=gate_layer,
                )
            else:
                parent_activations = self.cache.fetch(
                    branch_id=parent_branch_id,
                    layer_idx=gate_layer,
                )
            h_parent = parent_activations.get(gate_name)
        except (KeyError, RuntimeError):
            h_parent = None

        if h_parent is not None and (
            h_parent.dim() != 3
            or h_parent.shape[2] != hidden_states.shape[2]
            or h_parent.shape[1] == 0
        ):
            # A hidden-dim mismatch or zero-length entry cannot donate a
            # prefix-aligned activation: identical to a cache miss.
            h_parent = None
        if h_parent is not None and h_parent.shape[0] != 1:
            # Expand contract (AR015): the batched path serves N children
            # from a single parent row; a parent cached at batch != 1 is a
            # contract violation, not a cache miss.
            raise RuntimeError(
                f"Batched-branch folding requires a parent cached at batch "
                f"1, got batch {h_parent.shape[0]} for "
                f"branch={parent_branch_id}, layer={self.layer_idx}."
            )
        if h_parent is None:
            # Cache miss — e.g. the first generate-loop step over an
            # unpopulated root: whole-batch recompute (serial cache-miss
            # semantics), stored per branch.
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_batched(branch_ids, output, hidden_states)
            return self._pack_output(output)

        if h_parent.shape[1] > seq_len:
            # AR007 truncated-prefix donation (serial semantics).
            h_parent = h_parent[:, :seq_len]
        h_parent = h_parent.to(dtype=hidden_states.dtype, device=hidden_states.device)
        num_branches = len(branch_ids)
        h_parent = h_parent.expand(num_branches, -1, -1)
        prefix_len = h_parent.shape[1]

        # Plain gate only (AR015 design D3): the stride-0 expanded parent is
        # not validated inside the fused gate kernel; the serial fused
        # dispatch is untouched.
        if prefix_len == seq_len:
            stable_mask = self.gate(hidden_states, h_parent)  # [N, seq_len]
        else:
            # AR003 variable-length prefix folding: compare the prefix
            # against the parent, force the suffix divergent.
            prefix_mask = self.gate(hidden_states[:, :prefix_len], h_parent)
            stable_mask = torch.cat(
                [
                    prefix_mask,
                    torch.zeros(
                        num_branches,
                        seq_len - prefix_len,
                        dtype=torch.bool,
                        device=hidden_states.device,
                    ),
                ],
                dim=1,
            )

        if forced_divergent is not None:
            # AR021 FR2 single ordering (batched face): AND the forced
            # region in before the per-row profiler records and the batch
            # stable-count readback.
            stable_mask &= ~forced_divergent

        # AR029 target-layer availability guard (same-parent batch: one mask).
        stable_mask, _cache_guarded = _apply_cache_availability(self, parent_branch_id, stable_mask)

        for row_idx, bid in enumerate(branch_ids):
            GLOBAL_STABILITY_PROFILER.record(
                branch_id=bid,
                parent_branch_id=parent_branch_id,
                layer_idx=self.layer_idx,
                step_idx=step_idx,
                stable_mask=stable_mask[row_idx : row_idx + 1],
                tau=self.gate.tau,
                metric=self.gate.metric,
            )

        # One batch-granularity readback covers the three-way split (F4a).
        stable_count = int(stable_mask.sum())
        num_tokens = stable_mask.numel()

        if stable_count == num_tokens:
            # All-stable fast path (serial semantics): reuse the cached
            # parent FFN output without recomputation.
            try:
                ffn_out = self.cache.fetch(
                    branch_id=parent_branch_id,
                    layer_idx=self.layer_idx,
                ).get("ffn_out")
            except (KeyError, RuntimeError):
                ffn_out = None
            if ffn_out is not None and ffn_out.dim() == 3 and ffn_out.shape[1] > seq_len:
                ffn_out = ffn_out[:, :seq_len]
            if (
                ffn_out is not None
                and ffn_out.dim() == 3
                and ffn_out.shape[0] == 1
                and ffn_out.shape[1] == seq_len
                and ffn_out.shape[2] == hidden_states.shape[2]
                and self._returns_tuple is not None
            ):
                reused = ffn_out.expand(num_branches, -1, -1)
                self._store_batched(branch_ids, reused, hidden_states)
                return self._pack_output(reused)
            # Parent FFN missing/shape-mismatched or the original layer's
            # output structure unknown: recompute to stay consistent.
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_batched(branch_ids, output, hidden_states)
            return self._pack_output(output)

        if stable_count == 0:
            # Nothing to reuse: whole-batch recompute.
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_batched(branch_ids, output, hidden_states)
            return self._pack_output(output)

        # Slow path: recompute the whole batch with full child attention
        # context, then merge cached parent activations per row. A fully
        # stable row inside a mixed batch merges to the parent values —
        # bit-identical to the serial fast path for that row.
        merged_kwargs = {**kwargs, "stable_count": stable_count}
        child_out = self._recompute_merged(
            hidden_states,
            attention_mask,
            stable_mask,
            **merged_kwargs,
        )
        merge_prefix_len = None if prefix_len == seq_len else prefix_len
        h_out = self._merge_parent_child_batched(
            parent_branch_id,
            stable_mask,
            child_out,
            prefix_len=merge_prefix_len,
        )
        self._store_batched(branch_ids, h_out, hidden_states)
        return self._pack_output(h_out)

    def _fetch_parent_gate_row(
        self,
        parent_branch_id: str,
        gate_layer: int,
        gate_name: str,
        seq_len: int,
        hidden_dim: int,
    ) -> torch.Tensor | None:
        """Fetch one parent row's gate activation (AR025 DD2 per-row face).

        Returns the cached ``[1, T_p, H]`` activation (``T_p <= seq_len``),
        or ``None`` when the entry is missing / structurally unusable (dim,
        hidden, zero-length, or ``T_p > seq_len`` — a LONGER parent degrades
        its row: no AR007 truncation on the multi-parent face, UT-2409). A
        parent SHORTER than the child canvas donates its prefix (the
        append-only generate-loop shape, mirroring the single-parent
        ``_forward_batched`` face; the prefix is compared and the suffix is
        constructively divergent in the caller). A parent cached at
        batch != 1 is a hard ``RuntimeError`` (the per-row #41 contract
        violation, not a cache miss).
        """
        try:
            get_all_fn = getattr(self.cache, "get_all", None)
            if get_all_fn is not None:
                parent_activations = get_all_fn(
                    branch_id=parent_branch_id,
                    layer_idx=gate_layer,
                )
            else:
                parent_activations = self.cache.fetch(
                    branch_id=parent_branch_id,
                    layer_idx=gate_layer,
                )
            h_parent = parent_activations.get(gate_name)
        except (KeyError, RuntimeError):
            return None
        if h_parent is None:
            return None
        if h_parent.dim() != 3 or h_parent.shape[2] != hidden_dim or h_parent.shape[1] == 0:
            # A hidden-dim mismatch or zero-length entry cannot donate a
            # prefix-aligned activation: identical to a cache miss.
            return None
        if h_parent.shape[0] != 1:
            # Per-row expand contract (AR025, the #41 rule applied row-wise):
            # a parent cached at batch != 1 is a contract violation.
            raise RuntimeError(
                f"Multi-parent batched folding requires every parent cached "
                f"at batch 1, got batch {h_parent.shape[0]} for "
                f"branch={parent_branch_id}, layer={self.layer_idx}."
            )
        if h_parent.shape[1] > seq_len:
            # A LONGER parent degrades its row (UT-2409: no AR007
            # truncated-prefix donation on the multi face). A shorter parent
            # donates its prefix — the caller builds the per-row mask.
            return None
        return torch.as_tensor(h_parent)

    def _forward_batched_multi(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        branch_ids: list[str],
        parent_branch_ids: list[str],
        step_idx: int,
        forced_divergent: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        """Run one folded layer for N branches folding against N parents.

        Multi-parent semantics (AR025, design DD2/DD3/DD4): row ``i`` of
        ``hidden_states`` belongs to ``branch_ids[i]`` and folds against
        ``parent_branch_ids[i]`` (each cached at batch 1). The per-row
        mechanism face (gate / forced AND / profiler / merge ``where``) is
        already row-local, so the only batch-granularity work is the parent
        assembly (per-row fetch + stack, zeros placeholder for rows without
        a usable parent) and the mask-style three-way unification:

        - a row whose parent gate entry is missing / mismatched is forced
          ``stable_mask[row] = False`` CONSTRUCTIVELY (after the gate,
          before the forced AND — metric-independent no-leak, design DD2),
          so its values always come from the child recompute;
        - the all-stable fast path requires every row valid, the mask all
          True, and every row's parent FFN fetchable (per-row ``[1, T]``
          fetch -> stack ``[N, T, H]``); any miss takes the unified slow
          path where ``where(stable_mask, parent_ffn, child_out)`` realizes
          every legacy branch as a row-level degenerate (the :617-620
          contract).

        Args:
            hidden_states: Batched child hidden states ``[N, seq_len, hidden]``.
            attention_mask: Optional attention mask.
            branch_ids: One branch identifier per row.
            parent_branch_ids: One parent identifier per row.
            step_idx: Current diffusion step index.
            forced_divergent: Optional ``[N, seq_len]`` boolean tensor
                (AR021 FR2 ordering, unchanged).
            **kwargs: Extra arguments forwarded to ``original_layer``.

        Returns:
            Updated hidden states ``[N, seq_len, hidden]``.
        """
        batch, seq_len, hidden_dim = hidden_states.shape
        if self.scheduler is not None and not self.scheduler.should_fold(self.layer_idx, step_idx):
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_batched(branch_ids, output, hidden_states)
            return self._pack_output(output)
        if self.scheduler is not None:
            tau = self.scheduler.get_tau(
                layer_idx=self.layer_idx,
                step_idx=step_idx,
                task_type="general",
            )
            self.gate.set_tau(tau)

        gate_layer = 0 if self.layer_idx == 0 else self.layer_idx - 1
        gate_name = "embedding" if self.layer_idx == 0 else "ffn_out"

        # DD2 per-row fetch + validation; missing/mismatched rows degrade
        # (row_valid False), batch != 1 parents raise (hard, row-wise #41).
        row_valid: list[bool] = []
        row_parent: list[torch.Tensor | None] = []
        for pid in parent_branch_ids:
            row = self._fetch_parent_gate_row(pid, gate_layer, gate_name, seq_len, hidden_dim)
            row_parent.append(row)
            row_valid.append(row is not None)
        if not any(row_valid):
            # Every parent missing/mismatched: whole-batch recompute (the
            # serial cache-miss semantics, one batched recompute per layer).
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_batched(branch_ids, output, hidden_states)
            return self._pack_output(output)

        # Assemble the per-row gate comparison. Valid rows carry the parent
        # activation (possibly SHORTER than the child — the append-only
        # generate-loop shape donates its prefix, mirroring the
        # single-parent :577-604 face; a LONGER parent degrades, UT-2409);
        # invalid rows get an all-False mask row directly (the constructive
        # force of the zeros placeholder, without gating against zeros).
        if all(row_valid):
            valid_parent_rows = [row for row in row_parent if row is not None]
            prefix_len = valid_parent_rows[0].shape[1]
            if len({row.shape[1] for row in valid_parent_rows}) == 1:
                # Uniform parent length: one batched gate call (prefix or
                # full canvas, the ``_forward_batched`` mirror).
                h_parent = torch.cat(
                    valid_parent_rows,
                    dim=0,
                ).to(dtype=hidden_states.dtype, device=hidden_states.device)
                # Plain gate only (the AR015 D3 discipline: stacked per-row
                # parents are not validated inside the fused gate kernel).
                if prefix_len == seq_len:
                    stable_mask = self.gate(hidden_states, h_parent)  # [N, seq_len]
                else:
                    # AR003 append-only prefix folding: compare the prefix,
                    # force the appended suffix divergent.
                    prefix_mask = self.gate(hidden_states[:, :prefix_len], h_parent)
                    stable_mask = torch.cat(
                        [
                            prefix_mask,
                            torch.zeros(
                                batch,
                                seq_len - prefix_len,
                                dtype=torch.bool,
                                device=hidden_states.device,
                            ),
                        ],
                        dim=1,
                    )
            else:
                # Mixed parent lengths: per-row prefix masks.
                mask_rows: list[torch.Tensor] = []
                for row_idx, row in enumerate(row_parent):
                    assert row is not None
                    t_p = row.shape[1]
                    row_mask = self.gate(
                        hidden_states[row_idx : row_idx + 1, :t_p],
                        row.to(dtype=hidden_states.dtype, device=hidden_states.device),
                    )
                    if t_p < seq_len:
                        row_mask = torch.cat(
                            [
                                row_mask,
                                torch.zeros(
                                    1,
                                    seq_len - t_p,
                                    dtype=torch.bool,
                                    device=hidden_states.device,
                                ),
                            ],
                            dim=1,
                        )
                    mask_rows.append(row_mask)
                stable_mask = torch.cat(mask_rows, dim=0)
        else:
            # Some rows lack a usable parent: all-False rows for them, the
            # same per-row prefix treatment for the rest.
            mask_rows = []
            for row_idx, row in enumerate(row_parent):
                if row is None:
                    mask_rows.append(
                        torch.zeros(
                            1,
                            seq_len,
                            dtype=torch.bool,
                            device=hidden_states.device,
                        )
                    )
                    continue
                t_p = row.shape[1]
                row_mask = self.gate(
                    hidden_states[row_idx : row_idx + 1, :t_p],
                    row.to(dtype=hidden_states.dtype, device=hidden_states.device),
                )
                if t_p < seq_len:
                    row_mask = torch.cat(
                        [
                            row_mask,
                            torch.zeros(
                                1,
                                seq_len - t_p,
                                dtype=torch.bool,
                                device=hidden_states.device,
                            ),
                        ],
                        dim=1,
                    )
                mask_rows.append(row_mask)
            stable_mask = torch.cat(mask_rows, dim=0)

        # DD2 constructive force: rows without a usable parent are never
        # stable, whatever the gate said (the mixed branches above already
        # build all-False rows; this is the idempotent safety net).
        if not all(row_valid):
            for row_idx, valid in enumerate(row_valid):
                if not valid:
                    stable_mask[row_idx] = False

        if forced_divergent is not None:
            # AR021 FR2 single ordering (batched face), unchanged.
            stable_mask &= ~forced_divergent

        # AR029 per-row target-layer availability guard (each row uses its
        # own parent id; rows without a usable parent stay all-False).  The
        # shared helper keeps the rank/availability contract single-sourced.
        if getattr(self.cache, "availability", None) is not None:
            for row_idx, (pid, valid) in enumerate(zip(parent_branch_ids, row_valid)):
                if not valid:
                    stable_mask[row_idx] = False
                    continue
                row_mask, _ = _apply_cache_availability(
                    self, pid, stable_mask[row_idx : row_idx + 1]
                )
                stable_mask[row_idx] = row_mask[0]

        for row_idx, bid in enumerate(branch_ids):
            GLOBAL_STABILITY_PROFILER.record(
                branch_id=bid,
                parent_branch_id=parent_branch_ids[row_idx],
                layer_idx=self.layer_idx,
                step_idx=step_idx,
                stable_mask=stable_mask[row_idx : row_idx + 1],
                tau=self.gate.tau,
                metric=self.gate.metric,
            )

        # One batch-granularity readback covers the three-way split (F4a).
        stable_count = int(stable_mask.sum())
        num_tokens = stable_mask.numel()

        if stable_count == num_tokens:
            # All-stable fast path (DD3): every row's parent FFN is fetched
            # per row and stacked — no recompute at all. Any per-row fetch
            # miss falls through to the unified slow path below (values stay
            # correct: stable rows merge to the parent values bit-identically).
            ffn_rows: list[torch.Tensor | None] = []
            for pid in parent_branch_ids:
                ffn = self._fetch_parent_ffn_row(pid, seq_len, hidden_dim)
                ffn_rows.append(ffn)
            if all(row is not None for row in ffn_rows):
                reused = torch.cat(
                    [
                        row.to(dtype=hidden_states.dtype, device=hidden_states.device)
                        for row in ffn_rows
                        if row is not None
                    ],
                    dim=0,
                )
                self._store_batched(branch_ids, reused, hidden_states)
                return self._pack_output(reused)

        # Unified slow path (DD3 mask-style): recompute the whole batch with
        # full child attention context, then merge per-row parent FFN values
        # — where(stable_mask, parent_ffn, child_out) realizes the legacy
        # all-zero / mixed branches as row-level degenerates.
        merged_kwargs = {**kwargs, "stable_count": stable_count}
        child_out = self._recompute_merged(
            hidden_states,
            attention_mask,
            stable_mask,
            **merged_kwargs,
        )
        h_out = self._merge_parent_child_multi(
            parent_branch_ids,
            row_valid,
            stable_mask,
            child_out,
        )
        self._store_batched(branch_ids, h_out, hidden_states)
        return self._pack_output(h_out)

    def _fetch_parent_ffn_row(
        self,
        parent_branch_id: str,
        seq_len: int,
        hidden_dim: int,
    ) -> torch.Tensor | None:
        """Fetch one parent row's FFN output (AR025 DD3/DD4 per-row face).

        Returns the cached ``[1, T_p, H]`` FFN output with
        ``T_p <= seq_len`` (a prefix-shorter parent donates its prefix; a
        LONGER parent or any other shape mismatch is ``None`` — no AR007
        truncation on this face). Positions beyond ``T_p`` are pad-aligned
        by the caller and never read (the mask is constructively False
        there). Unlike the gate row, no batch check is needed here: the
        gate-row fetch already enforced batch 1 for a usable row.
        """
        try:
            ffn_out = self.cache.fetch(
                branch_id=parent_branch_id,
                layer_idx=self.layer_idx,
            ).get("ffn_out")
        except (KeyError, RuntimeError):
            return None
        if ffn_out is None:
            return None
        if (
            ffn_out.dim() != 3
            or ffn_out.shape[0] != 1
            or ffn_out.shape[1] == 0
            or ffn_out.shape[1] > seq_len
            or ffn_out.shape[2] != hidden_dim
        ):
            return None
        return ffn_out

    def _merge_parent_child_multi(
        self,
        parent_branch_ids: list[str],
        row_valid: list[bool],
        stable_mask: torch.Tensor,
        child_out: torch.Tensor,
    ) -> torch.Tensor:
        """Merge per-row parent FFN values into the batched child output.

        AR025 DD4: rows with a fetchable parent FFN merge through the
        unchanged :func:`_merge_stable_divergent_torch` ``where`` semantics;
        rows whose parent FFN is missing / mismatched take a
        ``child_out[row]`` placeholder AND a forced-False mask row, so
        placeholder values never flow out (shape alignment only).
        """
        num_branches, seq_len, hidden_dim = child_out.shape
        parent_rows: list[torch.Tensor] = []
        row_ffn_ok: list[bool] = []
        for row_idx, pid in enumerate(parent_branch_ids):
            if not row_valid[row_idx]:
                # No parent at all (gate-level invalid): placeholder, and
                # the constructive force already cleared the mask row.
                parent_rows.append(child_out[row_idx : row_idx + 1])
                row_ffn_ok.append(False)
                continue
            ffn = self._fetch_parent_ffn_row(pid, seq_len, hidden_dim)
            if ffn is None:
                # Parent FFN missing/mismatched: the row recomputes (mask
                # forced False below) — never a hard error on this face.
                parent_rows.append(child_out[row_idx : row_idx + 1])
                row_ffn_ok.append(False)
                continue
            ffn = ffn.to(dtype=child_out.dtype, device=child_out.device)
            if ffn.shape[1] < seq_len:
                # Prefix-shorter parent: pad the appended-suffix region with
                # zeros for shape alignment only — the mask is constructively
                # False there, so ``where`` never reads the padded values.
                ffn = torch.cat(
                    [
                        ffn,
                        torch.zeros(
                            1,
                            seq_len - ffn.shape[1],
                            hidden_dim,
                            dtype=ffn.dtype,
                            device=ffn.device,
                        ),
                    ],
                    dim=1,
                )
            parent_rows.append(ffn)
            row_ffn_ok.append(True)
        parent_ffn = torch.cat(parent_rows, dim=0)
        mask_eff = stable_mask
        if not all(row_ffn_ok):
            mask_eff = stable_mask.clone()
            for row_idx, ok in enumerate(row_ffn_ok):
                if not ok:
                    mask_eff[row_idx] = False
        return _merge_stable_divergent_torch(parent_ffn, child_out, mask_eff)

    def _store_batched(
        self,
        branch_ids: list[str],
        ffn_out: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Store per-branch activations for a batched forward (AR015).

        ``ffn_out`` may be a per-row ``[N, ...]`` tensor (row ``i`` goes to
        ``branch_ids[i]``) or a shared ``[1, ...]`` view (a defensive N==1
        case). The all-stable fast path passes the expanded ``[N, ...]``
        parent view — every row carries the same parent values, exactly
        like the serial fast path stores the parent view per child.

        AR025 DD5: when the ``multi_parent_batch`` switch is on and the
        cache is ``BatchedPutCapable`` (the #42 structural probe,
        re-checked on EVERY call — the cache attribute is swappable and
        the result is never cached on the instance), this layer's N rows
        go through ONE ``put_batch`` (L put_batch calls per forward total,
        one per layer — L·N → L). An incapable cache keeps the per-row
        loop below byte-for-byte. The payload mirrors the per-row face
        exactly (``ffn_out`` row views, plus the ``embedding`` row views
        at layer 0), so the terminal cache state is bit-identical either
        way. Single-parent batches take the same path — the publish face
        is orthogonal to the parent count.
        """
        if self.multi_parent_batch:
            from actfold.core.cache_protocol import BatchedPutCapable, CachePutEntry

            if isinstance(self.cache, BatchedPutCapable):
                entries: list[CachePutEntry] = []
                for row_idx, bid in enumerate(branch_ids):
                    child_ffn = ffn_out if ffn_out.shape[0] == 1 else ffn_out[row_idx : row_idx + 1]
                    activations: dict[str, torch.Tensor] = {"ffn_out": child_ffn}
                    if self.layer_idx == 0:
                        activations["embedding"] = hidden_states[row_idx : row_idx + 1]
                    entries.append(
                        CachePutEntry(
                            branch_id=bid,
                            layer_idx=self.layer_idx,
                            activations=activations,
                        )
                    )
                self.cache.put_batch(entries)
                return
        for row_idx, bid in enumerate(branch_ids):
            child_ffn = ffn_out if ffn_out.shape[0] == 1 else ffn_out[row_idx : row_idx + 1]
            self._store_activations(bid, child_ffn, hidden_states[row_idx : row_idx + 1])

    def _merge_parent_child_batched(
        self,
        parent_branch_id: str,
        stable_mask: torch.Tensor,
        child_out: torch.Tensor,
        prefix_len: int | None = None,
    ) -> torch.Tensor:
        """Merge the cached parent FFN with the batched child output (AR015).

        Three-stage contract (design D4): (1) the fetched parent FFN must
        match the parent's stored grid — ``[1, prefix_len]`` on the varlen
        path, ``[1, seq_len]`` (or longer, AR007-truncated) on the
        equal-length path; (2) it is expanded along the batch dimension;
        (3) the expanded output must match ``[N, prefix_len]`` on the varlen
        path (``stable_mask.shape`` on the equal-length degenerate) — the
        post-expand check makes a missing expand observable as a hard
        error instead of a silent broadcast. The fused ``gather_select``
        path is skipped (``fetch_flat`` rows are computed for the parent's
        stored batch and never cover the ``[N, T]`` child grid) and the
        merge runs on the deterministic PyTorch implementation.
        """
        num_branches, seq_len, _ = child_out.shape
        ffn_out = self.cache.fetch(
            branch_id=parent_branch_id,
            layer_idx=self.layer_idx,
        ).get("ffn_out")
        if ffn_out is None:
            raise RuntimeError(
                f"Parent FFN output missing for branch={parent_branch_id}, "
                f"layer={self.layer_idx}"
            )
        if prefix_len is not None:
            expected: torch.Size = torch.Size([1, prefix_len])
            if ffn_out.shape[:2] != expected:
                raise RuntimeError(
                    f"Parent FFN output shape {tuple(ffn_out.shape[:2])} does "
                    f"not match the expected shape {tuple(expected)} for "
                    f"branch={parent_branch_id}, layer={self.layer_idx}"
                )
        else:
            expected = torch.Size([1, seq_len])
            if ffn_out.shape[:2] != expected:
                if ffn_out.dim() == 3 and ffn_out.shape[0] == 1 and ffn_out.shape[1] > seq_len:
                    # AR007: a longer parent FFN donates its prefix.
                    ffn_out = ffn_out[:, :seq_len]
                else:
                    raise RuntimeError(
                        f"Parent FFN output shape {tuple(ffn_out.shape[:2])} "
                        f"does not match the expected shape "
                        f"{tuple(expected)} for branch={parent_branch_id}, "
                        f"layer={self.layer_idx}"
                    )
        parent_ffn = ffn_out.expand(num_branches, -1, -1)
        # Post-expand contract (design D4 stage 3): checked BEFORE the
        # varlen suffix concat — the grid is [N, prefix_len] on the varlen
        # path and degenerates to stable_mask.shape on the equal-length
        # path.
        expected_grid = (
            torch.Size([num_branches, prefix_len]) if prefix_len is not None else stable_mask.shape
        )
        if parent_ffn.shape[:2] != expected_grid:
            raise RuntimeError(
                f"Expanded parent FFN grid {tuple(parent_ffn.shape[:2])} does "
                f"not match the expected grid {tuple(expected_grid)} for "
                f"branch={parent_branch_id}, layer={self.layer_idx}"
            )
        if prefix_len is not None:
            # Variable-length prefix alignment (serial semantics): the
            # suffix is divergent by construction, so the merge always
            # takes the child value there.
            parent_ffn = torch.cat([parent_ffn, child_out[:, prefix_len:]], dim=1)
        return _merge_stable_divergent_torch(parent_ffn, child_out, stable_mask)

    def _pack_output(self, hidden: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, None]:
        """Match the original layer's output structure.

        Architectures such as LLaDA unpack block outputs as
        ``(hidden_states, cache)``.  When the wrapped layer is known to return a
        tuple, the folded output is wrapped as ``(hidden, None)`` so the base
        model's control flow keeps working with ``use_cache=False``.
        """
        if self._returns_tuple:
            return hidden, None
        return hidden

    def _merge_parent_child(
        self,
        parent_branch_id: str,
        stable_mask: torch.Tensor,
        child_out: torch.Tensor,
        prefix_len: int | None = None,
        parent_longer: bool = False,
    ) -> torch.Tensor:
        """Merge the cached parent FFN output with the recomputed child output.

        Prefers the fused single-pass ``gather_select`` (F8) that reads the
        vectorized cache buffer directly when the D3 shape thresholds are
        met; otherwise falls back to ``fetch`` +
        ``merge_stable_divergent``.  Both paths produce bit-identical
        results.

        Args:
            parent_branch_id: Identifier of the parent branch.
            stable_mask: Boolean stability mask ``[batch, seq_len]``.
            child_out: Recomputed child layer output ``[batch, seq_len, hidden]``.
            prefix_len: Parent prefix length for variable-length folding
                (AR003).  ``None`` selects the equal-length path.  When set,
                the ``gather_select`` fast path is skipped (``fetch_flat``
                indexes parent rows by the child ``seq_len``, which does not
                match a shorter parent) and the parent FFN is aligned to the
                child length with the recomputed suffix before the merge.
            parent_longer: True when the cached parent activation was
                longer than the child and was truncated (AR007).  The fused
                ``gather_select`` path is enabled (AR011): ``fetch_flat``
                rows are computed for the requested child length, so they
                read exactly the truncated parent prefix.  An
                eviction-wrapped parent (``start > 0``) still falls back —
                ``fetch_flat`` returns ``None`` and the parent FFN is
                fetched and truncated instead — bit-identical.
        """
        batch, seq_len, hidden_dim = child_out.shape
        if (
            prefix_len is None
            and seq_len >= _GATHER_SELECT_MIN_TOKENS
            and hidden_dim >= _GATHER_SELECT_MIN_HIDDEN
            and (not _GATHER_SELECT_REQUIRE_CUDA or child_out.is_cuda)
        ):
            fetch_flat = getattr(self.cache, "fetch_flat", None)
            if fetch_flat is not None:
                flat = fetch_flat(parent_branch_id, self.layer_idx, batch, seq_len)
                if flat is not None:
                    pair = flat.get("ffn_out")
                    if pair is not None:
                        flat_buffer, flat_rows = pair
                        return gather_select(
                            flat_buffer,
                            flat_rows.view(batch, seq_len),
                            child_out,
                            stable_mask,
                        )
        parent_ffn = self._get_parent_ffn_output(
            parent_branch_id, stable_mask, prefix_len=prefix_len
        )
        if prefix_len is not None:
            # Variable-length prefix alignment: the suffix positions are
            # divergent by construction, so the merge always takes the child
            # value there — the suffix fill below only satisfies the
            # same-shape contract of ``merge_stable_divergent``.
            parent_ffn = torch.cat([parent_ffn, child_out[:, prefix_len:]], dim=1)
        return merge_stable_divergent(parent_ffn, child_out, stable_mask)

    def _get_parent_ffn_output(
        self,
        parent_branch_id: str,
        stable_mask: torch.Tensor,
        prefix_len: int | None = None,
    ) -> torch.Tensor:
        """Retrieve the raw parent FFN output (no mask, no zero-fill).

        The merge overwrites divergent positions with the child output, so
        zero-filling them in the cache fetch would be a wasted allocation
        (F7).  The returned tensor may be a read-only view over the cache
        buffer.

        Args:
            parent_branch_id: Identifier of the parent branch.
            stable_mask: Boolean stability mask ``[batch, seq_len]``.
            prefix_len: Parent prefix length for variable-length folding
                (AR003).  ``None`` selects the equal-length contract, under
                which a parent FFN **longer** than the mask is accepted and
                truncated to the mask length (AR007 truncated-prefix
                reuse); when set, the FFN shape must match
                ``[batch, prefix_len]`` exactly (the varlen contract is
                unchanged).
        """
        ffn_out = self.cache.fetch(
            branch_id=parent_branch_id,
            layer_idx=self.layer_idx,
        ).get("ffn_out")
        if ffn_out is None:
            raise RuntimeError(
                f"Parent FFN output missing for branch={parent_branch_id}, "
                f"layer={self.layer_idx}"
            )
        if prefix_len is not None:
            expected_shape: torch.Size = torch.Size([stable_mask.shape[0], prefix_len])
            if ffn_out.shape[:2] != expected_shape:
                raise RuntimeError(
                    f"Parent FFN output shape {tuple(ffn_out.shape[:2])} does not match "
                    f"the expected shape {tuple(expected_shape)} for "
                    f"branch={parent_branch_id}, layer={self.layer_idx}"
                )
            return ffn_out
        expected_shape = stable_mask.shape
        if ffn_out.shape[:2] != expected_shape:
            if (
                ffn_out.dim() == 3
                and ffn_out.shape[0] == expected_shape[0]
                and ffn_out.shape[1] > expected_shape[1]
            ):
                # AR007: a longer parent FFN donates its first ``seq_len``
                # positions; shorter or batch-mismatched entries remain hard
                # errors (the varlen path above keeps the exact contract).
                return ffn_out[:, : expected_shape[1]]
            raise RuntimeError(
                f"Parent FFN output shape {tuple(ffn_out.shape[:2])} does not match "
                f"the expected shape {tuple(expected_shape)} for "
                f"branch={parent_branch_id}, layer={self.layer_idx}"
            )
        return ffn_out

    def _recompute_merged(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        stable_mask: torch.Tensor,
        stable_count: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Recompute the full layer for the merge step.

        The base implementation ignores the mask and recomputes every token.
        :class:`~actfold.core.split_layer.SplitFoldedTransformerLayer` overrides
        this to run the FFN only on divergent tokens while keeping the attention
        pass on the full sequence.  ``stable_count`` carries the caller's
        existing three-way readback so subclasses avoid a second host sync.
        """
        del stable_mask, stable_count
        return self._recompute_all(hidden_states, attention_mask, **kwargs)

    def _recompute_all(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the original layer on all provided hidden states.

        Many Transformer layers return a tuple ``(hidden_states, ...)``; the
        first element is always treated as the layer output. Arguments are
        filtered to the signature of ``original_layer.forward`` (inspected once
        at construction) so that ActFold-specific identifiers and unsupported
        kwargs (e.g. ``attention_mask`` for PyTorch ``TransformerEncoderLayer``)
        do not raise errors.
        """
        accepted = self._layer_params
        has_varkw = self._layer_has_varkw

        layer_kwargs = {k: v for k, v in kwargs.items() if k not in _ACTFOLD_KWARGS}
        if not has_varkw:
            layer_kwargs = {k: v for k, v in layer_kwargs.items() if k in accepted}
        if attention_mask is not None and "attention_mask" in accepted:
            layer_kwargs["attention_mask"] = attention_mask

        raw = self.original_layer(
            hidden_states,
            **layer_kwargs,
        )
        out: torch.Tensor
        if isinstance(raw, tuple):
            self._returns_tuple = True
            out = raw[0]
        else:
            self._returns_tuple = False
            out = raw
        return out

    def _store_activations(
        self,
        branch_id: str,
        ffn_out: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Store this layer's activations into the cache.

        Only the layer output is stored (F6): the layer input is redundant
        with the previous layer's ``ffn_out`` (or the embedding at layer 0),
        so dropping the per-layer ``hidden_states`` entry roughly halves the
        activation-cache footprint.
        """
        if self.layer_idx == 0:
            activations = {"ffn_out": ffn_out, "embedding": hidden_states}
        else:
            activations = {"ffn_out": ffn_out}
        self.cache.put(
            branch_id=branch_id,
            layer_idx=self.layer_idx,
            activations=activations,
        )
