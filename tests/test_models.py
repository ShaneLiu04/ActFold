"""Tests for actfold.models package."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from actfold.models import ModelRegistry, load_model
from actfold.models.base import DiffusionLLM
from actfold.models.causal_lm import CausalLMDiffusionLLM
from actfold.models.generic import GenericDiffusionLLM


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
    mock_tok.eos_token = "eos"
    mock_tok.__len__ = lambda _: 50257
    mock_tokenizer.return_value = mock_tok

    model = CausalLMDiffusionLLM("gpt2")
    assert model.vocab_size == 50257
    assert model.hidden_dim == 768
    assert model.num_layers == 12
    assert model.num_heads == 12


def _make_generic_mocks() -> tuple[MagicMock, MagicMock]:
    """Build (model, tokenizer) mocks for GenericDiffusionLLM without a head."""
    mock_model = MagicMock()
    # No ``lm_head`` attribute and no output embeddings: forces the random-head
    # fallback path that AR001 T003 turns into an explicit error.
    del mock_model.lm_head
    mock_model.config.vocab_size = 100
    mock_model.config.hidden_size = 16
    mock_model.config.num_hidden_layers = 2
    mock_model.config.num_attention_heads = 2
    mock_model.get_output_embeddings = MagicMock(return_value=None)
    mock_tok = MagicMock()
    mock_tok.pad_token = "pad"
    mock_tok.__len__ = lambda _: 100
    return mock_model, mock_tok


@patch("actfold.models.generic.AutoModel.from_pretrained")
@patch("actfold.models.generic.AutoTokenizer.from_pretrained")
def test_generic_random_head_raises_by_default(mock_tokenizer, mock_from_pretrained) -> None:
    """B3: a missing LM head must raise instead of silently using random weights."""
    mock_model, mock_tok = _make_generic_mocks()
    mock_from_pretrained.return_value = mock_model
    mock_tokenizer.return_value = mock_tok

    with pytest.raises(RuntimeError, match="allow_random_head"):
        GenericDiffusionLLM("dummy/no-head-model")


@patch("actfold.models.generic.AutoModel.from_pretrained")
@patch("actfold.models.generic.AutoTokenizer.from_pretrained")
def test_generic_random_head_allowed_with_flag(mock_tokenizer, mock_from_pretrained) -> None:
    """B3: ``allow_random_head=True`` opts into the random head explicitly."""
    mock_model, mock_tok = _make_generic_mocks()
    mock_from_pretrained.return_value = mock_model
    mock_tokenizer.return_value = mock_tok

    model = GenericDiffusionLLM("dummy/no-head-model", allow_random_head=True)
    assert model.lm_head is not None
    assert model.lm_head.weight.shape == (100, 16)


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


class _GeometryStubModel(DiffusionLLM):
    """Concrete ``DiffusionLLM`` stub for FFN/MoE geometry property tests (UT-408).

    Implements only the abstract members; geometry is reachable exclusively
    through ``self.model.config``.
    """

    def forward(self, tokens, attention_mask=None, **kwargs):
        raise NotImplementedError

    def embed(self, tokens):
        raise NotImplementedError

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


def _geometry_config() -> SimpleNamespace:
    """Build the UT-408 config-like object with the full primary-name geometry.

    Returns:
        SimpleNamespace with a swiglu FFN (``intermediate_size=12288``),
        8 routed experts (top-2, expert intermediate 1536), one shared
        expert (intermediate 2048), and 32 layers of which the first 3 are
        dense (``first_k_dense_replace=3`` -> 29 MoE layers).
    """
    return SimpleNamespace(
        intermediate_size=12288,
        hidden_act="silu",
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=1536,
        shared_expert_intermediate_size=2048,
        num_hidden_layers=32,
        first_k_dense_replace=3,
    )


def test_diffusion_llm_ffn_geometry_properties() -> None:
    """UT-408: the 7 geometry properties read real values from ``model.config``.

    ``config`` forwards to ``self.model.config`` and each geometry property
    resolves through the shared extraction helper, including
    ``moe_num_layers == 32 - 3 == 29``.
    """
    config = _geometry_config()
    model = _GeometryStubModel("dummy")
    model.model = SimpleNamespace(config=config)
    assert model.config is config
    assert model.ffn_intermediate_dim == 12288
    assert model.ffn_type == "swiglu"
    assert model.moe_num_experts == 8
    assert model.moe_top_k == 2
    assert model.moe_intermediate_dim == 1536
    assert model.moe_shared_expert is True
    assert model.moe_num_layers == 29


def test_diffusion_llm_ffn_geometry_defaults_without_model() -> None:
    """UT-408: ``self.model=None`` keeps the default geometry without raising.

    ``config`` is None and every geometry property falls back to its
    default (None / "mlp" / False).
    """
    model = _GeometryStubModel("dummy")
    assert model.model is None
    assert model.config is None
    assert model.ffn_intermediate_dim is None
    assert model.ffn_type == "mlp"
    assert model.moe_num_experts is None
    assert model.moe_top_k is None
    assert model.moe_intermediate_dim is None
    assert model.moe_shared_expert is False
    assert model.moe_num_layers is None
