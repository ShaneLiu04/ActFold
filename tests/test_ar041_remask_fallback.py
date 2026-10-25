"""AR041 tests: confidence-triggered remask fallback (CTR-F)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_ar033_history_cache import _MASK_ID, _build_llm, _cfg, _prompt  # noqa: E402

from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig  # noqa: E402


def _cfg_fb(**overrides):
    base = dict(
        history_cache=True,
        num_block_candidates=2,
        candidate_top_k=4,
        candidate_include_greedy=True,
        remask_fallback=True,
        block_size=8,
        small_block_size=8,
        num_tokens=48,
    )
    base.update(overrides)
    return _cfg(**base)


def test_ut4101_default_off_zero_fallbacks() -> None:
    llm = _build_llm(block_size=8)
    sampler = FastDLLMSampler(
        llm,
        config=_cfg(
            history_cache=True,
            num_block_candidates=2,
            candidate_top_k=4,
            candidate_include_greedy=True,
            block_size=8,
            small_block_size=8,
        ),
    )
    sampler.sample(_prompt())
    assert sampler.history_cache_stats["spec_remask_fallbacks"] == 0


def test_ut4102_always_falls_back_to_remasking() -> None:
    llm = _build_llm(block_size=8)
    sampler = FastDLLMSampler(llm, config=_cfg_fb(remask_fallback_threshold=1.1))
    out = sampler.sample(_prompt())
    st = sampler.history_cache_stats
    assert st["spec_remask_fallbacks"] > 0
    assert st["spec_verify_forwards"] == 0
    # one draft forward is spent to obtain the block confidence, then the
    # block runs the official remasking iterations (no verification).
    assert st["spec_draft_forwards"] == st["spec_remask_fallbacks"]
    assert st["degraded_calls"] == 0
    assert _MASK_ID not in out.sequences


def test_ut4103_never_falls_back_is_byte_identical() -> None:
    llm = _build_llm(block_size=8)
    prompt = _prompt()
    fb = FastDLLMSampler(llm, config=_cfg_fb(remask_fallback_threshold=0.0)).sample(prompt)
    off = FastDLLMSampler(
        llm,
        config=_cfg(
            history_cache=True,
            num_block_candidates=2,
            candidate_top_k=4,
            candidate_include_greedy=True,
            block_size=8,
            small_block_size=8,
            num_tokens=48,
        ),
    ).sample(prompt)
    assert torch.equal(fb.sequences, off.sequences)


def test_ut4104_illegal_combination() -> None:
    with pytest.raises(ValueError, match="remask_fallback"):
        FastDLLMSamplerConfig(remask_fallback=True)
