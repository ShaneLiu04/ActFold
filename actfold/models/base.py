"""Abstract base class for Diffusion LLM model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from actfold.core.model_wrapper import FoldedModel
    from actfold.models.diffusion_sampler import DiffusionSampler


class DiffusionLLM(ABC, nn.Module):
    """Abstract wrapper for Diffusion LLM inference.

    This class provides a unified interface for loading and running various
    Diffusion LLM architectures (LLaDA, Dream, Fast-dLLM, etc.) within the
    ActFold framework. Concrete subclasses must implement the forward pass
    and expose model architecture properties.

    Attributes:
        model_name_or_path: Hugging Face model identifier or local path.
        model: The underlying loaded model.
        tokenizer: The associated tokenizer, if available.
    """

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model: nn.Module | None = None
        self.tokenizer: Any | None = None

    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run a forward pass and return logits or hidden states.

        Args:
            tokens: Input token ids ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            **kwargs: Model-specific arguments.

        Returns:
            Output tensor, typically logits ``[batch, seq_len, vocab_size]``.
        """
        ...

    @abstractmethod
    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Look up input embeddings for ``tokens``.

        Args:
            tokens: Input token ids ``[batch, seq_len]``.

        Returns:
            Embedding tensor ``[batch, seq_len, hidden_dim]``.
        """
        ...

    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 16,
        num_steps: int = 10,
        folded_model: "FoldedModel" | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using the Diffusion LLM sampling procedure.

        Args:
            prompt_tokens: Prompt token ids ``[batch, seq_len]``.
            max_new_tokens: Number of tokens to generate.
            num_steps: Number of diffusion steps.
            folded_model: Optional folded model for activation reuse.
            **kwargs: Model-specific sampling arguments.

        Returns:
            Generated token ids ``[batch, seq_len + max_new_tokens]``.
        """
        if num_steps == 1:
            return self._autoregressive_generate(
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                folded_model=folded_model,
                **kwargs,
            )

        sampler = self.get_native_sampler(num_steps=num_steps, num_tokens=max_new_tokens)
        if sampler is None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not implement a native diffusion sampler; "
                "set num_steps=1 for autoregressive fallback."
            )
        return sampler.sample(prompt_ids=prompt_tokens, folded_model=folded_model)

    def _autoregressive_generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 16,
        folded_model: "FoldedModel" | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Default greedy autoregressive fallback for ``num_steps == 1``."""
        del folded_model, kwargs
        generated: torch.Tensor = prompt_tokens.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits: torch.Tensor = self.forward(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
        return generated

    def get_native_sampler(
        self,
        num_steps: int,
        num_tokens: int,
    ) -> "DiffusionSampler" | None:
        """Return a native diffusion sampler for this model family.

        Subclasses that support diffusion sampling should override this method.
        The default implementation returns ``None``, which causes ``generate``
        to fall back to autoregressive decoding when ``num_steps == 1`` or to
        raise an error for ``num_steps > 1``.

        Args:
            num_steps: Number of diffusion timesteps.
            num_tokens: Number of tokens to generate.

        Returns:
            Optional diffusion sampler.
        """
        return None

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

    def get_device(self) -> torch.device:
        """Return the device of the underlying model."""
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")
        return next(self.model.parameters()).device

    def estimate_memory_mb(self) -> float:
        """Estimate model memory footprint in MB."""
        if self.model is None:
            return 0.0
        total = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return total / (1024 * 1024)
