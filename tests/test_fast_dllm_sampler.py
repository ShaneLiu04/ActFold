"""Tests for the Fast-dLLM sampler's attention-mask handling."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from actfold.models.base import DiffusionLLM
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig

_VOCAB_SIZE = 16
_MASK_ID = 15
_STOP_ID = 14
_PAD_ID = 13
_BOS_ID = 12


class _RecorderTokenizer:
    """Minimal tokenizer with distinct pad/mask/stop ids."""

    def __init__(self) -> None:
        self.mask_token_id = _MASK_ID
        self.eos_token_id = _STOP_ID
        self.bos_token_id = _BOS_ID
        self.pad_token_id = _PAD_ID


class _RecordingModel(DiffusionLLM):
    """DiffusionLLM stub that records attention masks received by ``forward``."""

    def __init__(self) -> None:
        super().__init__("recording")
        self.tokenizer = _RecorderTokenizer()
        self.embedding = nn.Embedding(_VOCAB_SIZE, 8)
        self.head = nn.Linear(8, _VOCAB_SIZE)
        self.seen_tokens: list[torch.Tensor] = []
        self.seen_masks: list[torch.Tensor | None] = []

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.seen_tokens.append(tokens.detach().clone())
        self.seen_masks.append(
            attention_mask.detach().clone() if attention_mask is not None else None
        )
        hidden = self.embedding(tokens)
        return self.head(hidden)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def hidden_dim(self) -> int:
        return 8

    @property
    def num_heads(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return _VOCAB_SIZE


def test_fast_dllm_sampler_pads_excluded_from_attention() -> None:
    """Prompt padding is excluded from attention via the mask passed to forward.

    With ``prompt_len=3`` and ``block_size=8`` the sampler prepends
    ``first_block_padding=5`` pad tokens; every model forward call must receive
    an attention mask that is False exactly on those pad positions.
    """
    model = _RecordingModel()
    config = FastDLLMSamplerConfig(
        num_steps=2,
        num_tokens=4,
        block_size=8,
        small_block_size=4,
        mask_token_id=_MASK_ID,
        stop_token_id=_STOP_ID,
        pad_token_id=_PAD_ID,
    )
    sampler = FastDLLMSampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3]])

    sampler.sample(prompt)

    assert len(model.seen_masks) > 0
    for tokens, mask in zip(model.seen_tokens, model.seen_masks):
        assert mask is not None
        expected = tokens != _PAD_ID
        assert torch.equal(mask.to(torch.bool), expected)
        assert not bool(expected.all())
