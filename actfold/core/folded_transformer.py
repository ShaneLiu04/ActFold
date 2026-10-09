"""Folded Transformer layer with cross-branch activation reuse."""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn

from actfold.core.cache_factory import ActivationCacheType
from actfold.core.folding_context import FOLDING_CONTEXT
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.fused_ops import fused_gate_mask_count, gather_select, merge_stable_divergent
from actfold.core.similarity_gate import SimilarityGate
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

# Keywords that belong to ActFold and must never be forwarded to the original
# Transformer layer.
_ACTFOLD_KWARGS = {"branch_id", "parent_branch_id", "step_idx"}

# D3 thresholds for the fused single-pass gather/select merge (F8): below
# these shapes the kernel launch overhead outweighs the saved memory pass, so
# the original ``fetch`` + ``merge_stable_divergent`` path is used.
_GATHER_SELECT_MIN_TOKENS = 2048
_GATHER_SELECT_MIN_HIDDEN = 4096
_GATHER_SELECT_REQUIRE_CUDA = True


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
    ) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.cache = cache
        self.gate = gate
        self.layer_idx = layer_idx
        self.scheduler = scheduler
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
            **kwargs: Extra arguments forwarded to ``original_layer``.

        Returns:
            Updated hidden states ``[batch, seq_len, hidden_dim]``.
        """
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

        if h_parent is not None and h_parent.shape != hidden_states.shape:
            # Variable-length folding is unsupported (README limitation #7):
            # a parent whose sequence length differs from the child's cannot
            # donate activations (e.g. ``folded_generate`` appends one token
            # per step, so every child is one token longer than its parent).
            # Treat the parent as absent and recompute everything — identical
            # semantics to a cache miss, never a hard error.
            h_parent = None

        if h_parent is None:
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_activations(branch_id, output, hidden_states)
            return self._pack_output(output)

        # Align cached parent hidden states to the child's device/dtype before
        # the similarity comparison.
        h_parent = h_parent.to(dtype=hidden_states.dtype, device=hidden_states.device)

        # Compute stability mask entirely on GPU.  When the layer runs the
        # standard cosine gate, the mask and the stable count come out of the
        # single fused kernel (AR002/T006): the count buffer replaces the
        # separate ``mask.sum()`` reduction readback below.
        stable_count_buf: torch.Tensor | None = None
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
                ffn_out = self.cache.fetch(
                    branch_id=parent_branch_id,
                    layer_idx=self.layer_idx,
                ).get("ffn_out")
            except (KeyError, RuntimeError):
                ffn_out = None
            if (
                ffn_out is not None
                and ffn_out.shape == hidden_states.shape
                and self._returns_tuple is not None
            ):
                self._store_activations(branch_id, ffn_out, hidden_states)
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
        h_out = self._merge_parent_child(parent_branch_id, stable_mask, child_out)

        # Store child activations for future reuse.
        self._store_activations(branch_id, h_out, hidden_states)

        # Residual + layer norm are folded into original_layer in real models.
        # For this generic wrapper we assume original_layer already handles them.
        return self._pack_output(h_out)

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
    ) -> torch.Tensor:
        """Merge the cached parent FFN output with the recomputed child output.

        Prefers the fused single-pass ``gather_select`` (F8) that reads the
        vectorized cache buffer directly when the D3 shape thresholds are
        met; otherwise falls back to ``fetch`` +
        ``merge_stable_divergent``.  Both paths produce bit-identical
        results.
        """
        batch, seq_len, hidden_dim = child_out.shape
        if (
            seq_len >= _GATHER_SELECT_MIN_TOKENS
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
        parent_ffn = self._get_parent_ffn_output(parent_branch_id, stable_mask)
        return merge_stable_divergent(parent_ffn, child_out, stable_mask)

    def _get_parent_ffn_output(
        self,
        parent_branch_id: str,
        stable_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve the raw parent FFN output (no mask, no zero-fill).

        The merge overwrites divergent positions with the child output, so
        zero-filling them in the cache fetch would be a wasted allocation
        (F7).  The returned tensor may be a read-only view over the cache
        buffer.
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
        if ffn_out.shape[:2] != stable_mask.shape:
            raise RuntimeError(
                f"Parent FFN output shape {tuple(ffn_out.shape[:2])} does not match "
                f"the child mask shape {tuple(stable_mask.shape)} for "
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
