"""Tests for actfold.utils.flops_counter."""

from __future__ import annotations

import pytest

from actfold.utils.flops_counter import DiffusionLLMFLOPs, count_diffusion_llm_flops


def test_count_flops_baseline() -> None:
    flops = count_diffusion_llm_flops(
        num_layers=4,
        hidden_dim=128,
        num_heads=8,
        seq_len=16,
        vocab_size=1000,
        num_steps=1,
        reuse_ratio=0.0,
    )
    assert isinstance(flops, DiffusionLLMFLOPs)
    assert flops.total_tflops > 0.0
    assert flops.total_tflops == pytest.approx(
        flops.attention_tflops + flops.ffn_tflops + flops.embedding_tflops
    )


def test_count_flops_with_reuse() -> None:
    baseline = count_diffusion_llm_flops(
        num_layers=4,
        hidden_dim=128,
        num_heads=8,
        seq_len=16,
        vocab_size=1000,
        num_steps=1,
        reuse_ratio=0.0,
    )
    reused = count_diffusion_llm_flops(
        num_layers=4,
        hidden_dim=128,
        num_heads=8,
        seq_len=16,
        vocab_size=1000,
        num_steps=1,
        reuse_ratio=0.5,
    )
    assert reused.total_tflops < baseline.total_tflops


def test_count_flops_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(
            num_layers=0,
            hidden_dim=128,
            num_heads=8,
            seq_len=16,
            vocab_size=1000,
            num_steps=1,
        )


def test_count_flops_invalid_reuse_ratio() -> None:
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(
            num_layers=4,
            hidden_dim=128,
            num_heads=8,
            seq_len=16,
            vocab_size=1000,
            num_steps=1,
            reuse_ratio=1.5,
        )


def test_count_flops_head_divisibility() -> None:
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(
            num_layers=4,
            hidden_dim=128,
            num_heads=7,
            seq_len=16,
            vocab_size=1000,
            num_steps=1,
        )
