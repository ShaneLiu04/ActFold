"""AR037 tests: adaptive candidate budget (skip verification on confident drafts)."""

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


def _cfg_acb(**overrides):
    base = dict(
        history_cache=True,
        num_block_candidates=2,
        candidate_top_k=4,
        candidate_include_greedy=True,
        adaptive_candidate_budget=True,
    )
    base.update(overrides)
    return _cfg(**base)


def test_ut3701_default_off_zero_skips() -> None:
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
    assert sampler.history_cache_stats["spec_budget_skips"] == 0


def test_ut3702_permissive_threshold_skips_all() -> None:
    llm = _build_llm(block_size=8)
    sampler = FastDLLMSampler(
        llm,
        config=_cfg_acb(
            block_size=8, small_block_size=8, num_tokens=48, candidate_budget_threshold=0.0
        ),
    )
    out = sampler.sample(_prompt())
    stats = sampler.history_cache_stats
    assert stats["spec_budget_skips"] > 0
    assert stats["spec_verify_forwards"] == 0
    assert stats["spec_draft_forwards"] == stats["spec_budget_skips"]
    assert _MASK_ID not in out.sequences
    again = FastDLLMSampler(
        llm,
        config=_cfg_acb(
            block_size=8, small_block_size=8, num_tokens=48, candidate_budget_threshold=0.0
        ),
    ).sample(_prompt())
    assert torch.equal(out.sequences, again.sequences)


def test_ut3703_strict_threshold_never_skips() -> None:
    llm = _build_llm(block_size=8)
    prompt = _prompt()
    strict = FastDLLMSampler(
        llm,
        config=_cfg_acb(
            block_size=8, small_block_size=8, num_tokens=48, candidate_budget_threshold=1.1
        ),
    ).sample(prompt)
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
    assert torch.equal(strict.sequences, off.sequences)


def test_ut3704_illegal_combinations() -> None:
    with pytest.raises(ValueError, match="adaptive_candidate_budget"):
        FastDLLMSamplerConfig(adaptive_candidate_budget=True)
    with pytest.raises(ValueError, match="adaptive_candidate_budget"):
        FastDLLMSamplerConfig(
            history_cache=True,
            num_block_candidates=2,
            adaptive_candidate_budget=True,
            speculative_commit_gate=True,
        )
