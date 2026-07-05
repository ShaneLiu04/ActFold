"""Tests for architecture detection and generic folding helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.model_wrapper import FoldedModel
from actfold.models.architecture_utils import (
    ManualFoldedForward,
    detect_architecture,
    find_embedding_module,
    find_layer_list,
    find_lm_head,
)


class LlamaLikeModel(nn.Module):
    """Mock architecture following the LLaMA layout."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": vocab_size})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model.layers = nn.ModuleList([_Layer(hidden_dim) for _ in range(num_layers)])
        self.model.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.model.lm_head(x)


class Gpt2LikeModel(nn.Module):
    """Mock architecture following the GPT-2 layout."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "gpt2", "vocab_size": vocab_size})()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab_size, hidden_dim)
        self.transformer.h = nn.ModuleList([_Layer(hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.transformer.wte(tokens)
        for layer in self.transformer.h:
            x = layer(x)
        return self.lm_head(x)


class BertLikeModel(nn.Module):
    """Mock architecture following the BERT layout."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "bert", "vocab_size": vocab_size})()
        self.bert = nn.Module()
        self.bert.embeddings = nn.Module()
        self.bert.embeddings.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.bert.encoder = nn.Module()
        self.bert.encoder.layer = nn.ModuleList([_Layer(hidden_dim) for _ in range(num_layers)])
        self.cls = nn.Module()
        self.cls.predictions = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.bert.embeddings.word_embeddings(tokens)
        for layer in self.bert.encoder.layer:
            x = layer(x)
        return self.cls.predictions(x)


class _Layer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_find_layer_list_llama() -> None:
    model = LlamaLikeModel()
    result = find_layer_list(model)
    assert result is not None
    path, layers = result
    assert path == "model.layers"
    assert len(layers) == 3


def test_find_embedding_module_llama() -> None:
    model = LlamaLikeModel()
    result = find_embedding_module(model)
    assert result is not None
    path, embed = result
    assert path == "model.embed_tokens"
    assert isinstance(embed, nn.Embedding)


def test_find_lm_head_llama() -> None:
    model = LlamaLikeModel()
    result = find_lm_head(model)
    assert result is not None
    path, head = result
    assert path == "model.lm_head"
    assert isinstance(head, nn.Linear)


def test_detect_architecture_gpt2() -> None:
    model = Gpt2LikeModel()
    profile = detect_architecture(model)
    assert profile.model_type == "gpt2"
    assert profile.layer_path == "transformer.h"
    assert profile.embed_path == "transformer.wte"
    assert profile.head_path == "lm_head"
    assert profile.supports_causal_mask is True


def test_detect_architecture_bert() -> None:
    model = BertLikeModel()
    profile = detect_architecture(model)
    assert profile.model_type == "bert"
    assert profile.layer_path == "bert.encoder.layer"
    assert profile.embed_path == "bert.embeddings.word_embeddings"
    assert profile.supports_causal_mask is False


def test_detect_architecture_missing_layers_raises() -> None:
    model = nn.Linear(10, 10)
    with pytest.raises(RuntimeError, match="Could not discover the Transformer layer list"):
        detect_architecture(model)


def test_manual_folded_forward_llama(device: str) -> None:
    model = LlamaLikeModel().to(device)
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    folded = ManualFoldedForward(model, cache=cache, gate=gate).to(device)

    tokens = torch.randint(0, 100, (1, 4), device=device)
    out = folded(tokens, branch_id="root")
    assert out.shape == (1, 4, 100)


def test_folded_model_auto_discovers_llama(device: str) -> None:
    model = LlamaLikeModel().to(device)
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    folded = FoldedModel(model, cache=cache, gate=gate)
    assert folded.folding_applied is True

    tokens = torch.randint(0, 100, (1, 4), device=device)
    out = folded(tokens, branch_id="root")
    assert out.shape == (1, 4, 100)


def test_folded_model_auto_discovers_gpt2(device: str) -> None:
    model = Gpt2LikeModel().to(device)
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    folded = FoldedModel(model, cache=cache, gate=gate)
    assert folded.folding_applied is True
    assert folded._layer_path == "transformer.h"
