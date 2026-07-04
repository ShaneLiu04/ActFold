"""Model adapter interface for Fast-dLLM / Diffusion LLMs."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, cast

import torch
import torch.nn as nn

from actfold.core.model_wrapper import FoldedModel
from actfold.models.base import DiffusionLLM


class DiffusionLLMAdapter(ABC):
    """Abstract adapter for Diffusion LLM inference."""

    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run a full forward pass and return logits or hidden states."""
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of Transformer layers."""
        ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Return the hidden dimension."""
        ...

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Return the number of attention heads."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        ...

    @abstractmethod
    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Look up input embeddings for ``tokens``."""
        ...


class FastDLLMAdapter(DiffusionLLMAdapter):
    """Concrete adapter wrapping a raw model or DiffusionLLM for Fast-dLLM usage.

    This adapter accepts either:
    - A ``DiffusionLLM`` instance (recommended for real models).
    - A raw ``nn.Module`` that accepts ``input_ids`` and an optional
      ``attention_mask`` and returns logits of shape
      ``[batch, seq_len, vocab_size]``.

    Args:
        model: The underlying model.
        num_layers: Number of Transformer layers (ignored if model is DiffusionLLM).
        hidden_dim: Hidden dimension (ignored if model is DiffusionLLM).
        num_heads: Number of attention heads (ignored if model is DiffusionLLM).
        vocab_size: Vocabulary size (ignored if model is DiffusionLLM).
        folded_model: Optional FoldedModel that provides a branch-aware forward
            path. When provided and ``branch_id`` is passed in ``forward``,
            activations are reused across branches.
    """

    def __init__(
        self,
        model: nn.Module,
        num_layers: int | None = None,
        hidden_dim: int | None = None,
        num_heads: int | None = None,
        vocab_size: int | None = None,
        folded_model: FoldedModel | None = None,
    ) -> None:
        self._model = model
        self._folded_model = folded_model

        if isinstance(model, DiffusionLLM):
            self._num_layers = model.num_layers
            self._hidden_dim = model.hidden_dim
            self._num_heads = model.num_heads
            self._vocab_size = model.vocab_size
        else:
            if num_layers is None or hidden_dim is None:
                raise ValueError("num_layers and hidden_dim are required for raw nn.Module models.")
            self._num_layers = num_layers
            self._hidden_dim = hidden_dim
            self._num_heads = num_heads or max(1, hidden_dim // 64)
            self._vocab_size = vocab_size or 1000

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the wrapped model.

        If a ``FoldedModel`` was supplied and ``branch_id`` is present in
        ``kwargs``, the folded forward path is used so that parent activations
        can be reused. Otherwise ActFold-specific kwargs are filtered out before
        calling the underlying model.
        """
        # Prefer the folded path when branch identifiers are supplied.
        if self._folded_model is not None and "branch_id" in kwargs:
            logits: torch.Tensor = self._folded_model(
                tokens,
                attention_mask=attention_mask,
                **kwargs,
            )
            return logits

        # Strip ActFold-specific arguments when no folded model is available.
        # This keeps the adapter safe to call from the verification engine even
        # when the underlying model does not accept branch identifiers.
        actfold_kwargs = {"branch_id", "parent_branch_id", "step_idx"}
        forward_kwargs = {k: v for k, v in kwargs.items() if k not in actfold_kwargs}

        model = self._model
        if isinstance(model, DiffusionLLM):
            return cast(
                torch.Tensor, model(tokens, attention_mask=attention_mask, **forward_kwargs)
            )

        # For raw nn.Module, only pass arguments accepted by its forward method.
        sig = inspect.signature(model.forward)
        accepted = set(sig.parameters)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_varkw:
            forward_kwargs = {k: v for k, v in forward_kwargs.items() if k in accepted}
        if "attention_mask" in accepted:
            forward_kwargs["attention_mask"] = attention_mask
        return cast(torch.Tensor, model(tokens, **forward_kwargs))

    @property
    def num_layers(self) -> int:
        return int(self._num_layers)

    @property
    def hidden_dim(self) -> int:
        return int(self._hidden_dim)

    @property
    def num_heads(self) -> int:
        return int(self._num_heads)

    @property
    def vocab_size(self) -> int:
        return int(self._vocab_size)

    @property
    def underlying_model(self) -> nn.Module | DiffusionLLM:
        """Return the wrapped model."""
        return self._model

    @property
    def folded_model(self) -> FoldedModel | None:
        """Return the optional folded model, if any."""
        return self._folded_model

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Look up input embeddings for ``tokens`` using the wrapped model.

        The search order is:
        1. ``model.embed(tokens)`` if the wrapped model is a :class:`DiffusionLLM`.
        2. ``model.get_input_embeddings()(tokens)`` for Hugging Face models.
        3. Common embedding attributes: ``embedding``, ``word_embeddings``, ``wte``.

        Raises:
            RuntimeError: If no recognized embedding layer is found.
        """
        model = self._model

        # DiffusionLLM implementations expose a dedicated embed method.
        if isinstance(model, DiffusionLLM):
            embeddings = model.embed(tokens)
            if isinstance(embeddings, torch.Tensor):
                return embeddings
            raise RuntimeError("DiffusionLLM.embed did not return a tensor.")

        # Hugging Face style accessor.
        if hasattr(model, "get_input_embeddings"):
            getter = cast(Callable[[], nn.Module], getattr(model, "get_input_embeddings"))
            emb = getter()
            if isinstance(emb, nn.Module):
                out = emb(tokens)
                if isinstance(out, torch.Tensor):
                    return out

        # Common embedding attribute names.
        for attr in ("embedding", "word_embeddings", "wte", "embeddings"):
            if hasattr(model, attr):
                submodule = getattr(model, attr)
                if isinstance(submodule, nn.Embedding):
                    out = submodule(tokens)
                    if isinstance(out, torch.Tensor):
                        return out
                if isinstance(submodule, nn.Module):
                    try:
                        out = submodule(tokens)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to call {attr}(tokens) while looking for embeddings."
                        ) from exc
                    if isinstance(out, torch.Tensor):
                        return out
        raise RuntimeError(
            "Could not find input embeddings on the wrapped model. "
            "Expected a DiffusionLLM, a Hugging Face model with get_input_embeddings(), "
            "or one of the attributes: embedding, word_embeddings, wte, embeddings."
        )
