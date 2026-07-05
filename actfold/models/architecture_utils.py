"""Architecture detection and component extraction for Hugging Face models.

This module provides helpers that discover the embedding module, Transformer
layer stack, and language modeling head for a wide range of model families.
It is used by the demo and by generic folding wrappers so that Branch Folding
works without architecture-specific wiring.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import torch.nn as nn

_T = TypeVar("_T", bound=nn.Module)


@dataclass
class ArchitectureProfile:
    """Discovered architecture components of a loaded model.

    Attributes:
        model_type: Canonical model type string (e.g. ``"gpt2"``, ``"llama"``).
        embed_module: Module that maps token ids to hidden states.
        layers: List of Transformer decoder/encoder layers.
        head_module: Optional language modeling head or output projection.
        layer_path: Dot-separated attribute path to ``layers``.
        embed_path: Dot-separated attribute path to ``embed_module``.
        head_path: Dot-separated attribute path to ``head_module``.
        supports_causal_mask: Whether the model expects a causal ``attention_mask``.
    """

    model_type: str
    embed_module: nn.Module
    layers: nn.ModuleList
    head_module: nn.Module | None
    layer_path: str
    embed_path: str
    head_path: str | None
    supports_causal_mask: bool = True


# Ordered list of common layer-stack paths.  Earlier entries take precedence.
_DEFAULT_LAYER_PATHS: tuple[str, ...] = (
    # LLaMA / Qwen / Mistral / Gemma / Yi / InternLM style
    "model.layers",
    "transformer.h",
    "transformer.layers",
    # GPT-Neo / GPT-J / CodeGen
    "gpt_neox.layers",
    "transformer.blocks",
    # OPT
    "model.decoder.layers",
    "decoder.layers",
    # BERT / RoBERTa / DeBERTa
    "encoder.layer",
    "model.encoder.layer",
    "bert.encoder.layer",
    # T5 / UL2 / BART / mT5 (decoder path preferred for generation)
    "decoder.block",
    "model.decoder.block",
    "encoder.block",
    "model.encoder.block",
    # Falcon
    "transformer.h",
    # Phi / Phi-2
    "model.layers",
    # Generic
    "layers",
    "h",
    "blocks",
)

# Ordered list of common embedding paths.
_DEFAULT_EMBED_PATHS: tuple[str, ...] = (
    "model.embed_tokens",
    "transformer.wte",
    "transformer.word_embeddings",
    "transformer.embedding",
    "model.decoder.embed_tokens",
    "decoder.embed_tokens",
    "bert.embeddings.word_embeddings",
    "encoder.embed_tokens",
    "shared",
    "embeddings.word_embeddings",
    "embedding",
    "wte",
    "word_embeddings",
)

# Ordered list of common LM head paths.
_DEFAULT_HEAD_PATHS: tuple[str, ...] = (
    "lm_head",
    "model.lm_head",
    "transformer.lm_head",
    "model.decoder.lm_head",
    "encoder.lm_head",
    "cls.predictions",
    "cls",
    "head",
    "output_projection",
)


def _get_attr_path(model: nn.Module, path: str) -> Any | None:
    """Return the nested attribute at ``path`` or ``None`` if missing."""
    current: Any = model
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _find_path(
    model: nn.Module,
    candidates: tuple[str, ...],
    expected_type: type[_T] | tuple[type[_T], ...],
) -> tuple[str, _T] | None:
    """Find the first candidate path that resolves to an instance of ``expected_type``."""
    for path in candidates:
        obj = _get_attr_path(model, path)
        if isinstance(obj, expected_type):
            return path, obj
    return None


def find_layer_list(
    model: nn.Module,
    candidate_paths: tuple[str, ...] | None = None,
) -> tuple[str, nn.ModuleList] | None:
    """Discover the Transformer layer list in ``model``.

    Args:
        model: The model to inspect.
        candidate_paths: Optional ordered list of dot-separated paths to try.

    Returns:
        A tuple of ``(path, module_list)`` or ``None`` if no layer list is found.
    """
    paths = candidate_paths or _DEFAULT_LAYER_PATHS
    result = _find_path(model, paths, nn.ModuleList)
    return result


def find_embedding_module(
    model: nn.Module,
    candidate_paths: tuple[str, ...] | None = None,
) -> tuple[str, nn.Module] | None:
    """Discover the token embedding module in ``model``.

    Args:
        model: The model to inspect.
        candidate_paths: Optional ordered list of dot-separated paths to try.

    Returns:
        A tuple of ``(path, embedding_module)`` or ``None``.
    """
    paths = candidate_paths or _DEFAULT_EMBED_PATHS
    result = _find_path(model, paths, nn.Module)
    if result is not None:
        return result

    # Last resort: try get_input_embeddings for models that expose it.
    if hasattr(model, "get_input_embeddings"):
        getter = getattr(model, "get_input_embeddings")
        if callable(getter):
            emb = getter()
            if isinstance(emb, nn.Module):
                return "get_input_embeddings()", emb
    return None


def find_lm_head(
    model: nn.Module,
    candidate_paths: tuple[str, ...] | None = None,
) -> tuple[str, nn.Module] | None:
    """Discover the language modeling head in ``model``.

    Args:
        model: The model to inspect.
        candidate_paths: Optional ordered list of dot-separated paths to try.

    Returns:
        A tuple of ``(path, head_module)`` or ``None``.
    """
    paths = candidate_paths or _DEFAULT_HEAD_PATHS
    result = _find_path(model, paths, nn.Module)
    if result is not None:
        return result

    # Last resort: try get_output_embeddings for models that expose it.
    if hasattr(model, "get_output_embeddings"):
        getter = getattr(model, "get_output_embeddings")
        if callable(getter):
            head = getter()
            if isinstance(head, nn.Module):
                return "get_output_embeddings()", head
    return None


def _infer_model_type(model: nn.Module) -> str:
    """Infer a canonical model type from the model object or config."""
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if isinstance(model_type, str):
            return model_type.lower()
    cls_name = type(model).__name__.lower()
    for keyword in (
        "llada",
        "dream",
        "fastdllm",
        "gpt2",
        "gptneo",
        "gptj",
        "llama",
        "qwen",
        "mistral",
        "gemma",
        "opt",
        "falcon",
        "phi",
        "bert",
        "roberta",
        "t5",
        "bart",
    ):
        if keyword in cls_name:
            return keyword
    return "unknown"


def detect_architecture(
    model: nn.Module,
    layer_paths: tuple[str, ...] | None = None,
    embed_paths: tuple[str, ...] | None = None,
    head_paths: tuple[str, ...] | None = None,
) -> ArchitectureProfile:
    """Detect the embedding, layer stack, and head of ``model``.

    Args:
        model: A loaded Hugging Face model or ActFold adapter.
        layer_paths: Optional ordered list of layer-list paths.
        embed_paths: Optional ordered list of embedding paths.
        head_paths: Optional ordered list of LM-head paths.

    Returns:
        An :class:`ArchitectureProfile` describing the discovered components.

    Raises:
        RuntimeError: If the layer list cannot be discovered.
    """
    layer_result = find_layer_list(model, layer_paths)
    if layer_result is None:
        raise RuntimeError(
            "Could not discover the Transformer layer list. "
            f"Searched paths: {layer_paths or _DEFAULT_LAYER_PATHS}."
        )
    layer_path, layers = layer_result

    embed_result = find_embedding_module(model, embed_paths)
    if embed_result is None:
        raise RuntimeError(
            "Could not discover the token embedding module. "
            f"Searched paths: {embed_paths or _DEFAULT_EMBED_PATHS}."
        )
    embed_path, embed_module = embed_result

    head_result = find_lm_head(model, head_paths)
    head_module: nn.Module | None = None
    head_path: str | None = None
    if head_result is not None:
        head_path, head_module = head_result

    model_type = _infer_model_type(model)

    supports_causal_mask = model_type not in {"bert", "roberta", "deberta", "deberta-v2"}

    return ArchitectureProfile(
        model_type=model_type,
        embed_module=embed_module,
        layers=layers,
        head_module=head_module,
        layer_path=layer_path,
        embed_path=embed_path,
        head_path=head_path,
        supports_causal_mask=supports_causal_mask,
    )


def build_manual_folded_forward(
    model: nn.Module,
    cache: Any,
    gate: Any,
    scheduler: Any | None = None,
) -> "ManualFoldedForward":
    """Build a manual folded forward helper for ``model``.

    This is a convenience factory used when :class:`~actfold.core.model_wrapper.FoldedModel`
    cannot auto-discover the layer stack. It uses :func:`detect_architecture` to find
    the embedding, layers, and head, then wraps each layer with
    :class:`~actfold.core.folded_transformer.FoldedTransformerLayer`.

    Args:
        model: The model to wrap.
        cache: Activation cache shared across branches.
        gate: Similarity gate.
        scheduler: Optional folding scheduler.

    Returns:
        A :class:`ManualFoldedForward` instance.
    """
    return ManualFoldedForward(model, cache=cache, gate=gate, scheduler=scheduler)


class ManualFoldedForward(nn.Module):
    """Architecture-agnostic folded forward path for HF models.

    Similar to :class:`~actfold.core.model_wrapper.FoldedModel`, but explicitly
    extracts the embedding, layer stack, and language modeling head using
    :func:`detect_architecture` and runs the layers through
    :class:`~actfold.core.folded_transformer.FoldedTransformerLayer`. This is
    useful as a fallback for models whose internal layout does not match the
    auto-discovery heuristics in :class:`~actfold.core.model_wrapper.FoldedModel`.
    """

    def __init__(
        self,
        model: nn.Module,
        cache: Any,
        gate: Any,
        scheduler: Any | None = None,
    ) -> None:
        super().__init__()
        self.profile = detect_architecture(model)
        self.cache = cache
        self.gate = gate
        self.scheduler = scheduler
        self._wrapped_layers = nn.ModuleList(
            [self._wrap_layer(layer, idx) for idx, layer in enumerate(self.profile.layers)]
        )

    def _wrap_layer(self, layer: nn.Module, idx: int) -> nn.Module:
        """Wrap a single Transformer layer with FoldedTransformerLayer."""
        from actfold.core.folded_transformer import FoldedTransformerLayer

        return FoldedTransformerLayer(
            original_layer=layer,
            cache=self.cache,
            gate=self.gate,
            layer_idx=idx,
            scheduler=self.scheduler,
        )

    def forward(
        self,
        tokens: Any,
        branch_id: str,
        parent_branch_id: str | None = None,
        attention_mask: Any | None = None,
        step_idx: int = 0,
    ) -> Any:
        """Run a folded forward pass and return logits or hidden states.

        Args:
            tokens: Input token ids ``[batch, seq_len]``.
            branch_id: Identifier of the current branch.
            parent_branch_id: Optional parent branch identifier for reuse.
            attention_mask: Optional attention mask.
            step_idx: Current diffusion step index.

        Returns:
            Output of the language modeling head, typically logits.
        """
        emb_fn: Callable[..., Any] = self.profile.embed_module
        x = emb_fn(tokens)

        for wrapped in self._wrapped_layers:
            x = wrapped(
                x,
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                attention_mask=attention_mask,
                step_idx=step_idx,
            )

        if self.profile.head_module is not None:
            head_fn: Callable[..., Any] = self.profile.head_module
            return head_fn(x)

        warnings.warn(
            "ManualFoldedForward did not discover a language modeling head; "
            "returning hidden states instead of logits.",
            stacklevel=2,
        )
        return x

    @property
    def folding_applied(self) -> bool:
        """Return ``True`` because layers are explicitly wrapped."""
        return True
