"""Tests for actfold.speculative.fast_dllm_adapter."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.models.base import DiffusionLLM
from actfold.speculative.fast_dllm_adapter import DiffusionLLMAdapter, FastDLLMAdapter


class RawModel(nn.Module):
    """Simple raw model that accepts input_ids and an optional attention_mask."""

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(input_ids)
        logits: torch.Tensor = self.head(x)
        return logits


class MasklessModel(nn.Module):
    """Raw model that does not accept attention_mask."""

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        logits: torch.Tensor = self.head(x)
        return logits


def test_adapter_properties() -> None:
    raw = RawModel(vocab_size=100, hidden_dim=32)
    adapter = FastDLLMAdapter(
        raw,
        num_layers=2,
        hidden_dim=32,
        num_heads=4,
        vocab_size=100,
    )
    assert isinstance(adapter, DiffusionLLMAdapter)
    assert adapter.num_layers == 2
    assert adapter.hidden_dim == 32
    assert adapter.num_heads == 4
    assert adapter.vocab_size == 100


def test_adapter_forward_with_mask(device: str) -> None:
    raw = RawModel(vocab_size=100, hidden_dim=32).to(device)
    adapter = FastDLLMAdapter(
        raw,
        num_layers=2,
        hidden_dim=32,
        num_heads=4,
        vocab_size=100,
    )
    tokens = torch.randint(0, 100, (2, 8), device=device)
    mask = torch.ones(2, 8, dtype=torch.bool, device=device)
    logits = adapter.forward(tokens, attention_mask=mask)
    assert logits.shape == (2, 8, 100)


def test_adapter_forward_without_mask(device: str) -> None:
    raw = MasklessModel(vocab_size=100, hidden_dim=32).to(device)
    adapter = FastDLLMAdapter(
        raw,
        num_layers=2,
        hidden_dim=32,
        num_heads=4,
        vocab_size=100,
    )
    tokens = torch.randint(0, 100, (2, 8), device=device)
    logits = adapter.forward(tokens)
    assert logits.shape == (2, 8, 100)


def test_adapter_missing_dimensions() -> None:
    raw = RawModel(vocab_size=100, hidden_dim=32)
    with pytest.raises(ValueError):
        FastDLLMAdapter(raw, num_layers=None, hidden_dim=32)


def test_adapter_embed_raw_model(device: str) -> None:
    """FastDLLMAdapter.embed finds the embedding on a raw nn.Module."""
    raw = RawModel(vocab_size=100, hidden_dim=32).to(device)
    adapter = FastDLLMAdapter(raw, num_layers=2, hidden_dim=32)
    tokens = torch.randint(0, 100, (2, 8), device=device)
    embeddings = adapter.embed(tokens)
    assert embeddings.shape == (2, 8, 32)


def test_adapter_embed_unknown_fails() -> None:
    """FastDLLMAdapter.embed raises when no embedding layer is found."""

    class NoEmbed(nn.Module):
        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return tokens

    adapter = FastDLLMAdapter(NoEmbed(), num_layers=1, hidden_dim=4)
    with pytest.raises(RuntimeError):
        adapter.embed(torch.randint(0, 10, (1, 4)))


def test_adapter_filters_actfold_kwargs_without_folded_model(device: str) -> None:
    """branch_id/parent_branch_id/step_idx are filtered when no FoldedModel is set."""
    raw = RawModel(vocab_size=100, hidden_dim=32).to(device)
    adapter = FastDLLMAdapter(raw, num_layers=2, hidden_dim=32)
    tokens = torch.randint(0, 100, (2, 8), device=device)
    logits = adapter.forward(
        tokens,
        branch_id="child",
        parent_branch_id="parent",
        step_idx=3,
    )
    assert logits.shape == (2, 8, 100)


class MinimalDiffusionLLM(DiffusionLLM):
    """Tiny DiffusionLLM subclass for adapter tests."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__("minimal/test")
        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_heads = max(1, hidden_dim // 64)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        x = self.embedding(tokens)
        logits: torch.Tensor = self.head(x)
        return logits

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)

    def generate(self, prompt_tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return prompt_tokens

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


def test_adapter_wraps_diffusion_llm(device: str) -> None:
    """FastDLLMAdapter infers dimensions from a DiffusionLLM and calls embed."""
    diffusion = MinimalDiffusionLLM(vocab_size=64, hidden_dim=32, num_layers=3).to(device)
    adapter = FastDLLMAdapter(diffusion)
    assert adapter.num_layers == 3
    assert adapter.hidden_dim == 32
    assert adapter.vocab_size == 64

    tokens = torch.randint(0, 64, (2, 8), device=device)
    embeddings = adapter.embed(tokens)
    assert embeddings.shape == (2, 8, 32)

    logits = adapter.forward(tokens)
    assert logits.shape == (2, 8, 64)
