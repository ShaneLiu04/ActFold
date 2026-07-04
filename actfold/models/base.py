"""Abstract base class for Diffusion LLM model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


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

    @abstractmethod
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 16,
        num_steps: int = 10,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using the Diffusion LLM sampling procedure.

        Args:
            prompt_tokens: Prompt token ids ``[batch, seq_len]``.
            max_new_tokens: Number of tokens to generate.
            num_steps: Number of diffusion steps.
            **kwargs: Model-specific sampling arguments.

        Returns:
            Generated token ids ``[batch, seq_len + max_new_tokens]``.
        """
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
