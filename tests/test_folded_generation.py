"""Tests for true end-to-end folded generation."""

from __future__ import annotations

import torch
import torch.nn as nn

from actfold.core import ActivationCache, FoldedModel, SimilarityGate
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.eval.generation_utils import greedy_generate
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
from actfold.speculative.folded_generation import folded_generate


class TinyTransformer(nn.Module):
    """Tiny transformer for folded generation tests."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, hidden_dim // 64),
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _make_adapter(
    vocab_size: int = 16, hidden_dim: int = 64, num_layers: int = 2
) -> FastDLLMAdapter:
    raw = TinyTransformer(vocab_size, hidden_dim, num_layers)
    cache = ActivationCache(max_entries_per_layer=128, device="cpu")
    gate = SimilarityGate(tau=0.95)
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=num_layers, num_steps=1)
    folded = FoldedModel(raw, cache=cache, gate=gate, scheduler=scheduler)
    adapter = FastDLLMAdapter(
        raw, folded_model=folded, num_layers=num_layers, hidden_dim=hidden_dim
    )
    adapter.underlying_model.eval()
    return adapter


def test_folded_generate_runs_with_folded_model() -> None:
    """folded_generate successfully uses the folded forward path."""
    folded_adapter = _make_adapter()
    prompt = torch.tensor([[1, 2, 3]])

    result = folded_generate(folded_adapter, prompt, max_new_tokens=4)

    assert result.tokens.shape == (1, 7)
    assert result.num_folded_steps == 4
    assert 0.0 <= result.stable_ratio <= 1.0


def test_folded_generate_reports_stable_ratio() -> None:
    """folded_generate returns a non-negative stable ratio."""
    adapter = _make_adapter()
    prompt = torch.tensor([[1, 2, 3]])

    result = folded_generate(adapter, prompt, max_new_tokens=3)

    assert result.tokens.shape[1] == 6
    assert 0.0 <= result.stable_ratio <= 1.0
    assert result.num_folded_steps == 3


def test_folded_generate_without_folded_model_falls_back() -> None:
    """Without a folded model, folded_generate still produces greedy output."""
    raw = TinyTransformer(vocab_size=16, hidden_dim=64, num_layers=2)
    raw.eval()
    adapter = FastDLLMAdapter(raw, num_layers=2, hidden_dim=64)
    prompt = torch.tensor([[1, 2, 3]])

    expected = greedy_generate(adapter, prompt, max_new_tokens=3)
    result = folded_generate(adapter, prompt, max_new_tokens=3)

    assert torch.equal(result.tokens, expected)
