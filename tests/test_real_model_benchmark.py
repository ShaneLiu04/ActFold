"""Tests for real model benchmark integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from actfold.eval.benchmark_runner import BenchmarkRunner
from actfold.speculative import FastDLLMAdapter
from actfold.utils.config_manager import ActFoldConfig


def test_benchmark_runner_requires_model_name_or_path() -> None:
    """BenchmarkRunner raises when no model path is set and no model is injected."""
    config = ActFoldConfig(
        num_layers=2,
        hidden_dim=64,
        num_heads=4,
        seq_len=8,
        vocab_size=100,
        device="cpu",
    )
    with pytest.raises(ValueError, match="model_name_or_path is required"):
        BenchmarkRunner(config)


def test_benchmark_runner_with_real_model_config_mocked() -> None:
    """Verify BenchmarkRunner loads a real model when config provides a path."""
    from actfold.models.base import DiffusionLLM

    class DummyRealModel(DiffusionLLM):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            self._embedding = torch.nn.Embedding(100, 64)
            self.tokenizer = MagicMock()
            self.tokenizer.encode = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
            self.tokenizer.decode = MagicMock(return_value="answer")

        def forward(self, tokens, attention_mask=None, **kwargs):
            return torch.randn(tokens.size(0), tokens.size(1), 100)

        def embed(self, tokens):
            return self._embedding(tokens)

        def generate(self, prompt_tokens, **kwargs):
            return prompt_tokens

        @property
        def num_layers(self):
            return 2

        @property
        def hidden_dim(self):
            return 64

        @property
        def num_heads(self):
            return 4

        @property
        def vocab_size(self):
            return 100

    config = ActFoldConfig(
        model_name_or_path="dummy/real-model",
        model_family="causal_lm",
        seq_len=8,
        device="cpu",
    )

    with patch("actfold.eval.benchmark_runner.load_model") as mock_load:
        mock_load.return_value = DummyRealModel("dummy/real-model")
        runner = BenchmarkRunner(config)
        assert isinstance(runner.model, FastDLLMAdapter)
        assert runner.model.vocab_size == 100
        mock_load.assert_called_once()
