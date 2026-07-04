"""Folded Transformer layer with cross-branch activation reuse."""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn

from actfold.core.activation_cache import ActivationCache
from actfold.core.folding_context import FOLDING_CONTEXT
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.fused_ops import merge_stable_divergent
from actfold.core.similarity_gate import SimilarityGate

# Keywords that belong to ActFold and must never be forwarded to the original
# Transformer layer.
_ACTFOLD_KWARGS = {"branch_id", "parent_branch_id", "step_idx"}


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
        cache: ActivationCache,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        branch_id: str | None = None,
        parent_branch_id: str | None = None,
        attention_mask: torch.Tensor | None = None,
        step_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
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
            return output

        # Respect the scheduler if one is attached: disabled layers recompute.
        if self.scheduler is not None and not self.scheduler.should_fold(self.layer_idx, step_idx):
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_activations(branch_id, output, hidden_states)
            return output

        # Apply dynamic tau when a scheduler is configured.
        if self.scheduler is not None:
            tau = self.scheduler.get_tau(
                layer_idx=self.layer_idx,
                step_idx=step_idx,
                task_type="general",
            )
            self.gate.set_tau(tau)

        # Retrieve parent hidden states from cache. We need the parent layer
        # input hidden states for similarity comparison. If not cached, fall
        # back to recomputation.
        try:
            parent_activations = self.cache.get(
                branch_id=parent_branch_id,
                layer_idx=self.layer_idx,
                token_mask=torch.ones(
                    hidden_states.shape[:2],
                    dtype=torch.bool,
                    device=hidden_states.device,
                ),
            )
            h_parent = parent_activations.get("hidden_states")
        except (KeyError, RuntimeError):
            h_parent = None

        if h_parent is None:
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_activations(branch_id, output, hidden_states)
            return output

        # Align cached parent hidden states to the child's device/dtype before
        # the similarity comparison.
        h_parent = h_parent.to(dtype=hidden_states.dtype, device=hidden_states.device)

        # Compute stability mask entirely on GPU.
        stable_mask = self.gate(hidden_states, h_parent)  # [batch, seq_len]

        # Fast path: all tokens stable -> copy cached parent FFN output.
        if stable_mask.all():
            try:
                stable_activations = self.cache.get(
                    branch_id=parent_branch_id,
                    layer_idx=self.layer_idx,
                    token_mask=stable_mask,
                )
                ffn_out = stable_activations.get("ffn_out")
            except (KeyError, RuntimeError):
                ffn_out = None
            if ffn_out is not None:
                self._store_activations(branch_id, ffn_out, hidden_states)
                return ffn_out
            # Parent FFN output is missing despite all tokens being stable;
            # recompute the whole layer to stay consistent with the baseline.
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_activations(branch_id, output, hidden_states)
            return output

        # If no tokens are stable, there is nothing to reuse. Recompute the full
        # layer to avoid fetching a parent FFN output that may not be cached.
        if not stable_mask.any():
            output = self._recompute_all(hidden_states, attention_mask, **kwargs)
            self._store_activations(branch_id, output, hidden_states)
            return output

        # Slow path: recompute divergent tokens using full child attention context,
        # then fuse cached parent activations with freshly computed outputs.
        parent_ffn = self._get_parent_ffn_output(parent_branch_id, stable_mask)
        child_out = self._recompute_all(
            hidden_states,
            attention_mask,
            **kwargs,
        )
        h_out = merge_stable_divergent(parent_ffn, child_out, stable_mask)

        # Store child activations for future reuse.
        self._store_activations(branch_id, h_out, hidden_states)

        # Residual + layer norm are folded into original_layer in real models.
        # For this generic wrapper we assume original_layer already handles them.
        return h_out

    def _get_parent_ffn_output(
        self,
        parent_branch_id: str,
        stable_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve parent FFN output for stable tokens, zero-filled elsewhere."""
        stable_activations = self.cache.get(
            branch_id=parent_branch_id,
            layer_idx=self.layer_idx,
            token_mask=stable_mask,
        )
        ffn_out = stable_activations.get("ffn_out")
        if ffn_out is None:
            raise RuntimeError(
                f"Parent FFN output missing for branch={parent_branch_id}, "
                f"layer={self.layer_idx}"
            )
        return ffn_out

    def _recompute_all(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the original layer on all provided hidden states.

        Many Transformer layers return a tuple ``(hidden_states, ...)``; the
        first element is always treated as the layer output. Arguments are
        filtered to the signature of ``original_layer.forward`` so that
        ActFold-specific identifiers and unsupported kwargs (e.g.
        ``attention_mask`` for PyTorch ``TransformerEncoderLayer``) do not
        raise errors.
        """
        sig = inspect.signature(self.original_layer.forward)
        accepted = set(sig.parameters)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

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
            out = raw[0]
        else:
            out = raw
        return out

    def _store_activations(
        self,
        branch_id: str,
        ffn_out: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Store this layer's activations into the cache."""
        self.cache.put(
            branch_id=branch_id,
            layer_idx=self.layer_idx,
            activations={
                "ffn_out": ffn_out,
                "hidden_states": hidden_states,
            },
        )
