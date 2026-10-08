"""High-level model wrapper that applies Branch Folding to existing models."""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import torch
import torch.nn as nn

from actfold.core.cache_factory import ActivationCacheType
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.folding_context import folding_scope
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.similarity_gate import SimilarityGate
from actfold.core.split_layer import SplitFoldedTransformerLayer


class FoldedModel(nn.Module):
    """Wrap an existing model so its Transformer layers use Branch Folding.

    The wrapper attempts to find the layer stack via common Hugging Face
    attribute names (``layers``, ``model.layers``, ``transformer.h``,
    ``encoder.layer``, ``gpt_neox.layers``) and replaces each layer with a
    :class:`FoldedTransformerLayer`. If no known layer stack is found, the
    original ``forward`` method is still preserved but folding is disabled.

    Because most base models do not forward arbitrary ``**kwargs`` down to each
    layer, the branch context is also pushed into a thread-local
    :class:`~actfold.core.folding_context.FOLDING_CONTEXT` for the duration of
    the forward pass. Each :class:`FoldedTransformerLayer` reads this context
    when it is not passed the identifiers explicitly.

    The wrapper mutates the base model in place. Prefer the context-manager
    form ``with FoldedModel(model, cache, gate) as folded:`` so the original
    layers are restored on exit (including on exceptions); ``__del__`` is a
    last-resort restore for wrappers that were never restored explicitly.

    Args:
        model: The base model to wrap.
        cache: Activation cache shared across branches.
        gate: Similarity gate for token partitioning.
        layer_names: Optional sequence of attribute names to search for the
            layer ModuleList. Defaults to a list of common names.
    """

    _DEFAULT_LAYER_PATHS: tuple[str, ...] = (
        # Decoder-only models (LLaMA, Qwen, Mistral, Gemma, Yi, Phi, ...)
        "model.layers",
        "transformer.h",
        "transformer.layers",
        "layers",
        "h",
        # GPT-Neo / GPT-J / CodeGen
        "gpt_neox.layers",
        "transformer.blocks",
        "blocks",
        # LLaDA (LLaDAModelLM: model.transformer.blocks)
        "model.transformer.blocks",
        # OPT / BLOOM / Llama-style with explicit decoder
        "model.decoder.layers",
        "decoder.layers",
        # Encoder-only models (BERT, RoBERTa, DeBERTa)
        "encoder.layer",
        "model.encoder.layer",
        "bert.encoder.layer",
        # Seq2seq models (T5, BART, mT5, UL2)
        "decoder.block",
        "model.decoder.block",
        "encoder.block",
        "model.encoder.block",
    )

    def __init__(
        self,
        model: nn.Module,
        cache: ActivationCacheType,
        gate: SimilarityGate,
        layer_names: tuple[str, ...] | None = None,
        scheduler: FoldingScheduler | None = None,
        split_layers: bool = False,
        split_min_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.wrapped_model = model
        self.cache = cache
        self.gate = gate
        self.scheduler = scheduler
        self.split_layers = split_layers
        self.split_min_tokens = split_min_tokens
        self.layer_names = layer_names or self._DEFAULT_LAYER_PATHS
        self._layer_path: str | None = None
        self._original_layers: nn.ModuleList | None = None
        self._wrapped_layers: nn.ModuleList | None = None
        # Reflection is moved out of the forward hot path (F4d): both flags are
        # computed once, at construction time.
        self._actfold_kwargs_accepted = self._compute_accepts_actfold_kwargs()
        self._base_accepts_attention_mask = self._compute_accepts_attention_mask()
        self._apply_folding()

    def _compute_accepts_actfold_kwargs(self) -> bool:
        """Return whether the base model's forward accepts ActFold kwargs.

        Base models that do not accept ``branch_id`` / ``parent_branch_id`` /
        ``step_idx`` still fold correctly through the thread-local context, so
        the kwargs are omitted for them, avoiding an exception-based fallback.
        """
        try:
            sig = inspect.signature(self.wrapped_model.forward)
        except (TypeError, ValueError):
            return False
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        return has_varkw or all(
            key in params for key in ("branch_id", "parent_branch_id", "step_idx")
        )

    def _compute_accepts_attention_mask(self) -> bool:
        """Return whether the base model's forward accepts ``attention_mask``."""
        try:
            sig = inspect.signature(self.wrapped_model.forward)
        except (TypeError, ValueError):
            return False
        return "attention_mask" in sig.parameters

    def _accepts_actfold_kwargs(self) -> bool:
        """Return the construction-time ActFold-kwargs acceptance flag."""
        return self._actfold_kwargs_accepted

    def _apply_folding(self) -> None:
        """Replace discovered Transformer layers with folded equivalents."""
        for path in self.layer_names:
            layer_list = self._get_attr_path(self.wrapped_model, path)
            if isinstance(layer_list, nn.ModuleList):
                self._layer_path = path
                self._original_layers = layer_list
                if self.split_layers:
                    self._wrapped_layers = nn.ModuleList(
                        SplitFoldedTransformerLayer(
                            original_layer=layer,
                            cache=self.cache,
                            gate=self.gate,
                            layer_idx=idx,
                            scheduler=self.scheduler,
                            min_split_tokens=self.split_min_tokens,
                        )
                        for idx, layer in enumerate(layer_list)
                    )
                else:
                    self._wrapped_layers = nn.ModuleList(
                        FoldedTransformerLayer(
                            original_layer=layer,
                            cache=self.cache,
                            gate=self.gate,
                            layer_idx=idx,
                            scheduler=self.scheduler,
                        )
                        for idx, layer in enumerate(layer_list)
                    )
                self._set_attr_path(self.wrapped_model, path, self._wrapped_layers)
                warnings.warn(
                    "FoldedModel wrapped the base model IN PLACE: its layers are "
                    f"replaced under '{path}'. (1) state_dict() keys drift (each "
                    "layer is nested as 'original_layer'); re-save checkpoints "
                    "only after restore(). (2) Calling the raw model directly "
                    "without branch context raises. Use 'with FoldedModel(...) "
                    "as folded:' or call restore() when done.",
                    stacklevel=2,
                )
                return

        warnings.warn(
            "FoldedModel could not find a known layer stack in the base model. "
            f"Searched: {self.layer_names}. Folding is disabled and the wrapper "
            "acts as a thin pass-through.",
            stacklevel=2,
        )

    @staticmethod
    def _get_attr_path(obj: nn.Module, path: str) -> nn.ModuleList | None:
        """Retrieve a nested attribute by dot-separated path."""
        current: Any = obj
        for part in path.split("."):
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current if isinstance(current, nn.ModuleList) else None

    @staticmethod
    def _set_attr_path(obj: nn.Module, path: str, value: nn.ModuleList) -> None:
        """Set a nested attribute by dot-separated path."""
        parts = path.split(".")
        current: Any = obj
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], value)

    @property
    def folding_applied(self) -> bool:
        """Return True if at least one layer stack was wrapped."""
        return self._wrapped_layers is not None

    def forward(
        self,
        tokens: torch.Tensor,
        branch_id: str,
        parent_branch_id: str | None = None,
        attention_mask: torch.Tensor | None = None,
        step_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the wrapped model with optional Branch Folding.

        Args:
            tokens: Input token ids ``[batch, seq_len]``.
            branch_id: Identifier of the current branch.
            parent_branch_id: Optional parent branch identifier for reuse.
            attention_mask: Optional attention mask.
            step_idx: Current diffusion step index.
            **kwargs: Extra arguments forwarded to the base model.

        Returns:
            Model output logits or hidden states.
        """
        actfold_keys = {"branch_id", "parent_branch_id", "step_idx"}

        # When folding is not applied, run a normal forward. Only pass
        # attention_mask if the wrapped model accepts it.
        if not self.folding_applied:
            forward_kwargs: dict[str, Any] = {
                k: v for k, v in kwargs.items() if k not in actfold_keys
            }
            if attention_mask is not None and self._base_accepts_attention_mask:
                forward_kwargs["attention_mask"] = attention_mask
            out = self.wrapped_model(tokens, **forward_kwargs)
            return self._unwrap_output(out)

        # Set the thread-local folding context so that nested layers can read
        # branch identifiers even when the base model does not forward kwargs.
        with folding_scope(branch_id, parent_branch_id, step_idx):
            # Pass branch identifiers through kwargs only for models whose
            # forward accepts them; others rely on the folding context.
            if self._accepts_actfold_kwargs():
                folded_kwargs = {
                    **kwargs,
                    "branch_id": branch_id,
                    "parent_branch_id": parent_branch_id,
                    "step_idx": step_idx,
                }
            else:
                folded_kwargs = dict(kwargs)

            # Only pass attention_mask if the wrapped model accepts it.
            forward_kwargs = {**folded_kwargs}
            if attention_mask is not None and self._base_accepts_attention_mask:
                forward_kwargs["attention_mask"] = attention_mask

            try:
                out = self.wrapped_model(tokens, **forward_kwargs)
            except TypeError as exc:
                # If the base model rejects the ActFold-specific kwargs (e.g. a
                # raw Hugging Face model used directly), fall back to a normal
                # forward. The folding context remains active, so nested layers
                # still receive the branch identifiers.
                if any(key in str(exc) for key in actfold_keys):
                    fallback_kwargs = {k: v for k, v in kwargs.items() if k not in actfold_keys}
                    if attention_mask is not None and self._base_accepts_attention_mask:
                        fallback_kwargs["attention_mask"] = attention_mask
                    out = self.wrapped_model(tokens, **fallback_kwargs)
                else:
                    raise
        return self._unwrap_output(out)

    @staticmethod
    def _unwrap_output(out: Any) -> torch.Tensor:
        """Extract a tensor from a model output object or tuple.

        Many Hugging Face models return a ``ModelOutput`` dataclass (e.g.
        ``CausalLMOutputWithPast``) instead of a plain tensor. This helper
        returns ``logits`` when present and falls back to
        ``last_hidden_state`` or the first tuple element, keeping
        :class:`FoldedModel` API-compatible with the rest of ActFold.
        """
        if isinstance(out, torch.Tensor):
            return out
        logits = getattr(out, "logits", None)
        if isinstance(logits, torch.Tensor):
            return logits
        hidden = getattr(out, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            return hidden
        if isinstance(out, (tuple, list)) and out:
            first = out[0]
            if isinstance(first, torch.Tensor):
                return first
        raise TypeError(f"FoldedModel could not extract a tensor from output of type {type(out)}")

    def restore(self) -> nn.Module:
        """Restore the original model layers and return the base model.

        Idempotent: calling it after a successful restore is a no-op.
        Split-layer resident hooks are detached before the original layer
        list is swapped back so the restored model carries no leftovers.
        """
        if self._layer_path is not None and self._original_layers is not None:
            if self._wrapped_layers is not None:
                for layer in self._wrapped_layers:
                    remove_hooks = getattr(layer, "remove_hooks", None)
                    if callable(remove_hooks):
                        remove_hooks()
            self._set_attr_path(self.wrapped_model, self._layer_path, self._original_layers)
            self._wrapped_layers = None
            self._layer_path = None
            self._original_layers = None
        model: nn.Module = self.wrapped_model
        return model

    def __enter__(self) -> "FoldedModel":
        """Enter the wrap scope; layers are restored when the scope exits."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Restore the base model even when the scope exits with an exception."""
        self.restore()

    def __del__(self) -> None:
        """Last-resort restore when the wrapper is garbage-collected unwrapped."""
        try:
            if self._layer_path is not None and self._original_layers is not None:
                self.restore()
        except Exception:  # noqa: BLE001 - interpreter teardown must never raise
            pass
