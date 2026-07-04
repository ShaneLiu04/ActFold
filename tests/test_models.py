"""Tests for actfold.models package."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from actfold.models import ModelRegistry, load_model
from actfold.models.base import DiffusionLLM
from actfold.models.causal_lm import CausalLMDiffusionLLM


def test_registry_list_models() -> None:
    models = ModelRegistry.list_models()
    assert "llada" in models
    assert "dream" in models
    assert "fast_dllm" in models
    assert "causal_lm" in models


def test_registry_resolve_family() -> None:
    assert ModelRegistry._resolve_family("some/llada-8b", "auto") == "llada"
    assert ModelRegistry._resolve_family("my/dream-7b", "auto") == "dream"
    assert ModelRegistry._resolve_family("fast-dllm-v2", "auto") == "fast_dllm"
    assert ModelRegistry._resolve_family("gpt2", "auto") == "causal_lm"
    assert ModelRegistry._resolve_family("gpt2", "causal_lm") == "causal_lm"


def test_registry_unknown_family() -> None:
    with pytest.raises(ValueError, match="Unknown model family"):
        ModelRegistry.load("dummy", model_family="unknown")


@patch("actfold.models.causal_lm.AutoModelForCausalLM.from_pretrained")
@patch("actfold.models.causal_lm.AutoTokenizer.from_pretrained")
def test_causal_lm_wrapper(mock_tokenizer, mock_from_pretrained) -> None:
    mock_model = MagicMock()
    mock_model.config.vocab_size = 50257
    mock_model.config.hidden_size = 768
    mock_model.config.num_hidden_layers = 12
    mock_model.config.num_attention_heads = 12
    mock_model.parameters.return_value = iter([torch.randn(10, 10)])
    mock_from_pretrained.return_value = mock_model

    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "<|endoftext|>"
    mock_tok.__len__ = lambda _: 50257
    mock_tokenizer.return_value = mock_tok

    model = CausalLMDiffusionLLM("gpt2")
    assert model.vocab_size == 50257
    assert model.hidden_dim == 768
    assert model.num_layers == 12
    assert model.num_heads == 12


def test_load_model_function() -> None:
    with patch("actfold.models.registry.ModelRegistry.load") as mock_load:
        mock_model = MagicMock(spec=DiffusionLLM)
        mock_load.return_value = mock_model
        result = load_model("gpt2", model_family="causal_lm")
        mock_load.assert_called_once_with("gpt2", model_family="causal_lm")
        assert result is mock_model


def test_diffusion_llm_interface() -> None:
    class DummyModel(DiffusionLLM):
        def forward(self, tokens, attention_mask=None, **kwargs):
            return tokens

        def embed(self, tokens):
            return torch.randn(tokens.size(0), tokens.size(1), self.hidden_dim)

        def generate(self, prompt_tokens, **kwargs):
            return prompt_tokens

        @property
        def num_layers(self):
            return 2

        @property
        def hidden_dim(self):
            return 16

        @property
        def num_heads(self):
            return 2

        @property
        def vocab_size(self):
            return 100

    model = DummyModel("dummy")
    model.model = MagicMock()
    model.model.parameters.return_value = iter([torch.randn(2, 2)])
    assert model.estimate_memory_mb() > 0.0
