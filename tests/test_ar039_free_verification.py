"""AR039 tests: verification-free greedy commit + free verification."""

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

from actfold.models import fast_dllm_sampler as fds  # noqa: E402
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig  # noqa: E402


def _cfg_vfg(**overrides):
    base = dict(
        history_cache=True,
        num_block_candidates=2,
        candidate_top_k=4,
        candidate_include_greedy=True,
        commit_draft_fusion=True,
        verification_free_commit=True,
        block_size=8,
        small_block_size=8,
        num_tokens=64,
    )
    base.update(overrides)
    return _cfg(**base)


def test_ut3901_default_off_zero_counters() -> None:
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
    st = sampler.history_cache_stats
    assert st["spec_verification_free_blocks"] == 0
    assert st["free_verify_blocks"] == 0
    assert st["free_verify_low_conf"] == 0


def test_ut3902_vfg_skips_verification(monkeypatch) -> None:
    llm = _build_llm(block_size=8)
    monkeypatch.setattr(fds, "_crop_history_kv", lambda kv, keep: kv)
    sampler = FastDLLMSampler(llm, config=_cfg_vfg())
    out = sampler.sample(_prompt())
    st = sampler.history_cache_stats
    assert st["spec_verification_free_blocks"] > 0
    assert st["spec_fused_forwards"] > 0
    # The tiny fixture is low-confidence, so the free verification re-arms
    # the full verification repeatedly: both paths must be exercised.
    assert st["spec_verify_forwards"] > 0
    assert st["free_verify_low_conf"] > 0
    assert st["spec_verification_free_blocks"] + st["spec_verify_forwards"] <= 8
    assert _MASK_ID not in out.sequences
    assert st["degraded_calls"] == 0


def test_ut3903_free_verification_matches_fused(monkeypatch) -> None:
    llm = _build_llm(block_size=8)
    monkeypatch.setattr(fds, "_crop_history_kv", lambda kv, keep: kv)
    sampler = FastDLLMSampler(llm, config=_cfg_vfg())
    sampler.sample(_prompt())
    st = sampler.history_cache_stats
    assert st["free_verify_blocks"] == st["spec_fused_forwards"]
    assert st["free_verify_low_conf"] <= st["free_verify_blocks"]


def test_ut3904_deterministic_and_mask_free(monkeypatch) -> None:
    llm = _build_llm(block_size=8)
    monkeypatch.setattr(fds, "_crop_history_kv", lambda kv, keep: kv)
    prompt = _prompt()
    a = FastDLLMSampler(llm, config=_cfg_vfg()).sample(prompt)
    b = FastDLLMSampler(llm, config=_cfg_vfg()).sample(prompt)
    assert torch.equal(a.sequences, b.sequences)
    assert _MASK_ID not in a.sequences


def test_ut3905_illegal_combinations() -> None:
    with pytest.raises(ValueError, match="verification_free_commit"):
        FastDLLMSamplerConfig(verification_free_commit=True, commit_draft_fusion=True)
    with pytest.raises(ValueError, match="commit_draft_fusion"):
        FastDLLMSamplerConfig(
            history_cache=True,
            num_block_candidates=2,
            verification_free_commit=True,
        )
    with pytest.raises(ValueError, match="not composed"):
        FastDLLMSamplerConfig(
            history_cache=True,
            num_block_candidates=2,
            commit_draft_fusion=True,
            verification_free_commit=True,
            speculative_commit_gate=True,
        )
