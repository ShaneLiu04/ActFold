"""AR044 tests: refine-after-fill (fill once, then re-read and rewrite low-confidence positions)."""

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


def _cfg_raf(passes: int, **overrides):
    base = dict(
        history_cache=True,
        num_block_candidates=2,
        candidate_top_k=4,
        candidate_include_greedy=True,
        commit_refine_passes=passes,
        block_size=8,
        small_block_size=8,
        num_tokens=24,
    )
    base.update(overrides)
    return _cfg(**base)


def test_ut4401_default_off_zero_refine_forwards() -> None:
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
    assert sampler.history_cache_stats["spec_refine_forwards"] == 0


def test_ut4402_passes_count_and_mask_free() -> None:
    llm = _build_llm(block_size=8)
    sampler = FastDLLMSampler(llm, config=_cfg_raf(2))
    out = sampler.sample(_prompt())
    st = sampler.history_cache_stats
    assert st["spec_refine_forwards"] > 0
    assert st["spec_refine_forwards"] % 2 == 0
    assert st["degraded_calls"] == 0
    assert _MASK_ID not in out.sequences


def test_ut4403_deterministic() -> None:
    llm = _build_llm(block_size=8)
    prompt = _prompt()
    a = FastDLLMSampler(llm, config=_cfg_raf(1)).sample(prompt)
    b = FastDLLMSampler(llm, config=_cfg_raf(1)).sample(prompt)
    assert torch.equal(a.sequences, b.sequences)


def test_ut4404_illegal_values() -> None:
    with pytest.raises(ValueError, match="commit_refine_passes"):
        FastDLLMSamplerConfig(commit_refine_passes=-1)
    with pytest.raises(ValueError, match="commit_refine_passes"):
        FastDLLMSamplerConfig(commit_refine_passes=1)
