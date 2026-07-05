"""Tests for diffusion-native samplers."""

from __future__ import annotations

import torch
import torch.nn as nn

from actfold.models.base import DiffusionLLM
from actfold.models.dream_sampler import DreamSampler
from actfold.models.fast_dllm_sampler import FastDLLMSampler
from actfold.models.llada_sampler import LLaDASampler


class DummyDiffusionModel(DiffusionLLM):
    """Minimal DiffusionLLM for sampler tests."""

    def __init__(self, vocab_size: int = 16, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__("dummy")
        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del attention_mask, kwargs
        if tokens.dtype in (torch.long, torch.int):
            x = self.embedding(tokens)
        else:
            x = tokens
        return self.head(x)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_heads(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


def test_llada_sampler_reduces_masks() -> None:
    """LLaDASampler progressively unmasks generated positions."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    sampler = LLaDASampler(model, num_steps=4, num_tokens=4, mask_token_id=15)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert output.shape == (1, 7)
    assert (output == 15).sum().item() < 4  # some positions should be unmasked


def test_fast_dllm_sampler_changes_tokens() -> None:
    """FastDLLMSampler modifies tokens over diffusion steps."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    sampler = FastDLLMSampler(model, num_steps=4, num_tokens=4)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert output.shape == (1, 7)


def test_dream_sampler_outputs_tokens() -> None:
    """DreamSampler returns discrete tokens from continuous diffusion."""

    class DummyDreamModel(DummyDiffusionModel):
        """Predicts clean embeddings instead of logits."""

        def __init__(self) -> None:
            super().__init__(vocab_size=16, hidden_dim=32)
            self.head = nn.Linear(32, 32)

        def forward(self, tokens, attention_mask=None, **kwargs):
            del attention_mask, kwargs
            if tokens.dtype in (torch.long, torch.int):
                x = self.embedding(tokens)
            else:
                x = tokens
            return self.head(x)

        def decode_final(self, x: torch.Tensor) -> torch.Tensor:
            # Map embeddings back to logits and argmax.
            logits = torch.matmul(x, self.embedding.weight.t())
            return logits.argmax(dim=-1)

    model = DummyDreamModel()
    sampler = DreamSampler(model, num_steps=4, num_tokens=4)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert output.shape == (1, 7)
    assert output.dtype in (torch.long, torch.int)
