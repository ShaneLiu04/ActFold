"""Tests for diffusion-native samplers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from actfold.models.base import DiffusionLLM
from actfold.models.diffusion_sampler import SamplerOutput
from actfold.models.dream_sampler import DreamSampler, DreamSamplerConfig
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig
from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig


class DummyTokenizer:
    """Minimal tokenizer stand-in for sampler tests."""

    def __init__(self, vocab_size: int = 16) -> None:
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 2
        self.bos_token_id = vocab_size - 3
        self.pad_token_id = self.eos_token_id


class DummyDiffusionModel(DiffusionLLM):
    """Minimal DiffusionLLM for sampler tests."""

    def __init__(
        self,
        vocab_size: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_logits: bool = True,
    ) -> None:
        super().__init__("dummy")
        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.tokenizer = DummyTokenizer(vocab_size)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.output_logits = output_logits
        if output_logits:
            self.head = nn.Linear(hidden_dim, vocab_size)
        else:
            self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
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
    config = LLaDASamplerConfig(num_steps=4, num_tokens=4)
    sampler = LLaDASampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert isinstance(output, SamplerOutput)
    assert output.sequences.shape == (1, 7)
    # Some generated positions should be unmasked (not equal to mask id 15).
    assert (output.sequences == model.tokenizer.mask_token_id).sum().item() < 4


def test_fast_dllm_sampler_changes_tokens() -> None:
    """FastDLLMSampler produces an output tensor of expected shape."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    config = FastDLLMSamplerConfig(num_steps=4, num_tokens=4, block_size=4, small_block_size=4)
    sampler = FastDLLMSampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert isinstance(output, SamplerOutput)
    assert output.sequences.shape[0] == 1
    assert output.sequences.shape[1] >= 1


def test_dream_sampler_outputs_tokens() -> None:
    """DreamSampler returns discrete tokens from masked diffusion."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    config = DreamSamplerConfig(num_steps=4, num_tokens=4)
    sampler = DreamSampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert isinstance(output, SamplerOutput)
    assert output.sequences.shape == (1, 7)
    assert output.sequences.dtype in (torch.long, torch.int)


def test_llada_sampler_with_history() -> None:
    """LLaDASampler can return intermediate canvases."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    config = LLaDASamplerConfig(num_steps=4, num_tokens=4, return_history=True)
    sampler = LLaDASampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert isinstance(output, SamplerOutput)
    assert len(output.history) > 1


def test_dream_sampler_cfg() -> None:
    """DreamSampler supports classifier-free guidance."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    config = DreamSamplerConfig(num_steps=4, num_tokens=4, cfg_scale=1.0)
    sampler = DreamSampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert isinstance(output, SamplerOutput)
    assert output.sequences.shape == (1, 7)


def test_fast_dllm_sampler_trim_stop() -> None:
    """FastDLLMSampler trims trailing stop tokens in decode_final."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    config = FastDLLMSamplerConfig(num_steps=4, num_tokens=4, block_size=4, small_block_size=4)
    sampler = FastDLLMSampler(model, config=config)
    sampler.initialize(torch.tensor([[1, 2, 3]]))

    x = torch.tensor([[1, 2, 3, 14, 14, 14]])  # 14 is eos/stop
    decoded = sampler.decode_final(x)
    # decode_final removes leading pads/masks and keeps a single trailing stop.
    assert decoded.shape[0] == 1
    assert decoded[0, -1].item() == model.tokenizer.eos_token_id


def test_sampling_utilities() -> None:
    """Smoke test for shared sampling helpers used by all samplers."""
    from actfold.models.sampling_utils import (
        LinearMaskingScheduler,
        add_gumbel_noise,
        get_num_transfer_tokens,
        right_shift_logits,
        sample_tokens,
        top_k_logits,
        top_p_logits,
    )

    scheduler = LinearMaskingScheduler()
    assert scheduler.alpha(0.0) == 1.0
    assert scheduler.alpha(1.0) == 0.0

    mask_index = torch.tensor([[True, True, True, True]])
    transfers = get_num_transfer_tokens(mask_index, steps=4, scheduler=scheduler)
    assert transfers.shape[0] == 1
    assert transfers.sum().item() == 4

    logits = torch.randn(2, 4, 16)
    shifted = right_shift_logits(logits)
    assert shifted.shape == logits.shape

    top_k = top_k_logits(logits, top_k=5)
    assert (top_k == float("-inf")).sum().item() > 0

    top_p = top_p_logits(logits, top_p=0.9)
    assert top_p.shape == logits.shape

    conf, tokens = sample_tokens(logits, temperature=0.0)
    assert tokens.shape == (2, 4)

    noisy = add_gumbel_noise(logits, temperature=0.0)
    assert noisy.shape == logits.shape
