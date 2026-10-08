"""Layer-internal FFN splitting for Branch Folding (optimization #2).

Attention must run on the full child sequence to keep the self-attention
context identical to the baseline, but the feed-forward network is token-wise:
its rows can be computed independently.  :class:`SplitFoldedTransformerLayer`
therefore keeps the attention/norm/residual computation of the original layer
untouched and only feeds the *divergent* rows through the FFN chain, scattering
the result back into the full sequence.  Divergent positions are then
bit-identical to the baseline recompute, while FFN FLOPs scale with the number
of divergent tokens.

The split is implemented with temporary forward hooks on two modules that bound
the token-wise FFN chain (a pre-norm and the MLP/down-projection), so no model
code is modified and the wrapping is fully reversible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from actfold.core.folded_transformer import FoldedTransformerLayer


@dataclass(frozen=True)
class SplitSpec:
    """Modules bounding the token-wise FFN chain of a decoder layer.

    Attributes:
        pre_module: First module of the token-wise chain (e.g. the post-attention
            layer norm). Its input is sliced to the divergent rows.
        post_module: Last module of the token-wise chain (e.g. the MLP or the
            down projection). Its output is scattered back to full length.
        label: Human-readable description used in experiment logs.
    """

    pre_module: nn.Module
    post_module: nn.Module
    label: str


# Supported pre-norm decoder layouts, in detection order:
# 1. Llama/Qwen/Dream-style: post_attention_layernorm -> mlp
# 2. LLaDA-style inline FFN: ff_norm -> ... -> ff_out
_CHAIN_CANDIDATES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("post_attention_layernorm",), ("mlp",)),
    (("ff_norm",), ("ff_out",)),
)


def detect_split_spec(layer: nn.Module) -> SplitSpec | None:
    """Return the split specification for ``layer`` if its layout is supported."""
    for pre_names, post_names in _CHAIN_CANDIDATES:
        pre = next((getattr(layer, name) for name in pre_names if hasattr(layer, name)), None)
        post = next((getattr(layer, name) for name in post_names if hasattr(layer, name)), None)
        if isinstance(pre, nn.Module) and isinstance(post, nn.Module):
            return SplitSpec(
                pre_module=pre,
                post_module=post,
                label=f"{pre_names[0]}->{post_names[0]}",
            )
    return None


class SplitFoldedTransformerLayer(FoldedTransformerLayer):
    """Folded layer that computes the FFN only for divergent tokens.

    The pre/post hooks are registered once at construction and remain
    resident: ``_split_state is None`` makes them fully transparent (the
    wrapped layer behaves exactly like the unwrapped module), so the
    per-forward ``register_forward_hook``/``remove`` pattern is unnecessary.
    Call :meth:`remove_hooks` (or :meth:`FoldedModel.restore`) to detach
    them.

    Args:
        original_layer: The base Transformer layer to wrap.
        cache: Activation cache for parent activations.
        gate: Similarity gate for token partitioning.
        layer_idx: Index of this layer in the model.
        scheduler: Optional folding scheduler.
        split_spec: Optional explicit split specification; detected from
            ``original_layer`` when omitted.
        min_split_tokens: Minimum ``batch * seq`` token count for the split
            path; below it the data-dependent row gathers are not worth the
            device synchronization.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        cache: Any,
        gate: Any,
        layer_idx: int,
        scheduler: Any | None = None,
        split_spec: SplitSpec | None = None,
        min_split_tokens: int = 512,
    ) -> None:
        super().__init__(original_layer, cache, gate, layer_idx, scheduler)
        self.split_spec = (
            split_spec if split_spec is not None else detect_split_spec(original_layer)
        )
        self._split_enabled = self.split_spec is not None
        self.min_split_tokens = int(min_split_tokens)
        self._split_state: dict[str, Any] | None = None
        # Resident hooks (F9b): registered once, guarded by ``_split_state``.
        self._pre_handle: Any = None
        self._post_handle: Any = None
        if self.split_spec is not None:
            self._pre_handle = self.split_spec.pre_module.register_forward_pre_hook(
                self._pre_hook, with_kwargs=True
            )
            self._post_handle = self.split_spec.post_module.register_forward_hook(
                self._post_hook
            )

    @property
    def split_enabled(self) -> bool:
        """Whether the FFN split is active for this layer."""
        return self._split_enabled

    def remove_hooks(self) -> None:
        """Detach the resident pre/post hooks (idempotent)."""
        if self._pre_handle is not None:
            self._pre_handle.remove()
            self._pre_handle = None
        if self._post_handle is not None:
            self._post_handle.remove()
            self._post_handle = None

    def __del__(self) -> None:
        """Last-resort hook cleanup; never raises during teardown."""
        try:
            self.remove_hooks()
        except Exception:  # noqa: BLE001 - interpreter teardown must never raise
            pass

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------
    def _pre_hook(self, module: nn.Module, args: Any, kwargs: Any) -> Any:
        state = self._split_state
        if state is None or not args or not isinstance(args[0], torch.Tensor):
            return None
        hidden = args[0]
        if hidden.ndim != 3:
            return None
        flat_index = state["flat_index"]
        flat = hidden.reshape(-1, hidden.shape[-1])
        x_divergent = flat.index_select(0, flat_index)
        state["input_shape"] = tuple(hidden.shape)
        state["input_dtype"] = hidden.dtype
        state["num_divergent"] = int(x_divergent.shape[0])
        return (x_divergent,) + tuple(args[1:]), kwargs

    def _post_hook(self, module: nn.Module, args: Any, output: Any) -> Any:
        state = self._split_state
        if state is None:
            return output
        is_tuple = isinstance(output, tuple)
        out = output[0] if is_tuple else output
        if not isinstance(out, torch.Tensor):
            return output

        input_shape = state.get("input_shape")
        if input_shape is not None and out.ndim == 2 and len(input_shape) == 3:
            # Scatter base (F9a): ``torch.empty`` instead of ``torch.zeros`` —
            # the stable rows are don't-care because the subsequent merge in
            # ``FoldedTransformerLayer.forward`` overwrites them with the
            # cached parent FFN output.  This removes a full ``[B, T, H]``
            # zero-fill memory pass per folded layer.
            full = torch.empty(
                input_shape,
                dtype=state.get("input_dtype", out.dtype),
                device=out.device,
            )
            full.reshape(-1, input_shape[-1]).index_copy_(0, state["flat_index"], out)
            result: torch.Tensor = full
        else:
            result = out
        return (result,) + tuple(output[1:]) if is_tuple else result

    # ------------------------------------------------------------------
    # Folded recompute
    # ------------------------------------------------------------------
    def _recompute_merged(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        stable_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        spec = self.split_spec
        if not self._split_enabled or spec is None:
            return self._recompute_all(hidden_states, attention_mask, **kwargs)
        # Data-dependent row gathers force a device synchronisation per layer;
        # below this token count the FFN savings do not pay for it (measured on
        # RTX PRO 6000 Blackwell).
        num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
        if num_tokens < self.min_split_tokens:
            return self._recompute_all(hidden_states, attention_mask, **kwargs)

        divergent = ~stable_mask
        flat_index = divergent.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        self._split_state = {
            "flat_index": flat_index,
            "input_shape": None,
            "num_divergent": 0,
        }
        try:
            return self._recompute_all(hidden_states, attention_mask, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._split_enabled = False
            warnings.warn(
                f"Split FFN disabled for layer {self.layer_idx} "
                f"({type(exc).__name__}: {exc}); falling back to full recompute.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._recompute_all(hidden_states, attention_mask, **kwargs)
        finally:
            self._split_state = None

    def extra_repr(self) -> str:
        spec = self.split_spec.label if self.split_spec is not None else "none"
        return f"layer_idx={self.layer_idx}, split={spec}, enabled={self._split_enabled}"
