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
    ) -> None:
        super().__init__()
        self.wrapped_model = model
        self.cache = cache
        self.gate = gate
        self.scheduler = scheduler
        self.layer_names = layer_names or self._DEFAULT_LAYER_PATHS
        self._layer_path: str | None = None
        self._original_layers: nn.ModuleList | None = None
        self._wrapped_layers: nn.ModuleList | None = None
        self._apply_folding()

    def _apply_folding(self) -> None:
        """Replace discovered Transformer layers with folded equivalents."""
        for path in self.layer_names:
            layer_list = self._get_attr_path(self.wrapped_model, path)
            if isinstance(layer_list, nn.ModuleList):
                self._layer_path = path
                self._original_layers = layer_list
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
            if attention_mask is not None:
                sig = inspect.signature(self.wrapped_model.forward)
                if "attention_mask" in sig.parameters:
                    forward_kwargs["attention_mask"] = attention_mask
            out: torch.Tensor = self.wrapped_model(tokens, **forward_kwargs)
            return out

        # Set the thread-local folding context so that nested layers can read
        # branch identifiers even when the base model does not forward kwargs.
        with folding_scope(branch_id, parent_branch_id, step_idx):
            # Pass branch identifiers through kwargs as well, for models that do
            # forward arbitrary kwargs. Strip them again if the base model
            # rejects them.
            folded_kwargs = {
                **kwargs,
                "branch_id": branch_id,
                "parent_branch_id": parent_branch_id,
                "step_idx": step_idx,
            }

            # Only pass attention_mask if the wrapped model accepts it.
            forward_kwargs = {**folded_kwargs}
            if attention_mask is not None:
                sig = inspect.signature(self.wrapped_model.forward)
                if "attention_mask" in sig.parameters:
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
                    if attention_mask is not None and "attention_mask" in sig.parameters:
                        fallback_kwargs["attention_mask"] = attention_mask
                    out = self.wrapped_model(tokens, **fallback_kwargs)
                else:
                    raise
        return out

    def restore(self) -> nn.Module:
        """Restore the original model layers and return the base model."""
        if self._layer_path is not None and self._original_layers is not None:
            self._set_attr_path(self.wrapped_model, self._layer_path, self._original_layers)
            self._wrapped_layers = None
        model: nn.Module = self.wrapped_model
        return model
