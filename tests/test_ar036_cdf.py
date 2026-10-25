"""AR036 tests: commit-draft fusion (default off, fallback, ledger identity)."""

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

from test_ar033_history_cache import _build_llm, _cfg, _prompt  # noqa: E402

from actfold.models import fast_dllm_sampler as fds  # noqa: E402
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig  # noqa: E402


def _cfg_fused(**overrides):
    base = dict(
        history_cache=True,
        num_block_candidates=2,
        candidate_top_k=4,
        candidate_include_greedy=True,
        commit_draft_fusion=True,
    )
    base.update(overrides)
    return _cfg(**base)


def test_ut3601_default_off_zero_regression() -> None:
    llm = _build_llm(block_size=8)
    prompt = _prompt()
    cfg = _cfg(
        history_cache=True,
        num_block_candidates=2,
        candidate_top_k=4,
        candidate_include_greedy=True,
        block_size=8,
        small_block_size=8,
    )
    out1 = FastDLLMSampler(llm, config=cfg).sample(prompt)
    out2 = FastDLLMSampler(llm, config=cfg).sample(prompt)
    assert torch.equal(out1.sequences, out2.sequences)
    sampler = FastDLLMSampler(llm, config=cfg)
    sampler.sample(prompt)
    assert sampler.history_cache_stats["spec_fused_forwards"] == 0


def test_ut3602_fixture_cache_has_no_crop_falls_back() -> None:
    llm = _build_llm(block_size=8)
    sampler = FastDLLMSampler(
        llm, config=_cfg_fused(block_size=8, small_block_size=8, num_tokens=64)
    )
    with pytest.warns(RuntimeWarning, match="commit_draft_fusion disabled"):
        sampler.sample(_prompt())
    stats = sampler.history_cache_stats
    assert stats["spec_fused_forwards"] == 0
    assert stats["spec_draft_forwards"] == stats["spec_verify_forwards"]
    assert sampler._hc_fusion_disabled is True


def test_ut3603_fused_ledger_identity_when_crop_works(monkeypatch) -> None:
    llm = _build_llm(block_size=8)
    monkeypatch.setattr(fds, "_crop_history_kv", lambda kv, keep: kv)
    sampler = FastDLLMSampler(
        llm, config=_cfg_fused(block_size=8, small_block_size=8, num_tokens=64)
    )
    out = sampler.sample(_prompt())
    stats = sampler.history_cache_stats
    assert stats["spec_fused_forwards"] > 0
    assert (
        stats["spec_draft_forwards"] + stats["spec_fused_forwards"] == stats["spec_verify_forwards"]
    )
    assert stats["degraded_calls"] == 0
    assert out.sequences.numel() > 0


def test_ut3604_illegal_combinations() -> None:
    with pytest.raises(ValueError, match="commit_draft_fusion"):
        FastDLLMSamplerConfig(commit_draft_fusion=True)
    with pytest.raises(ValueError, match="commit_draft_fusion"):
        FastDLLMSamplerConfig(
            history_cache=True,
            num_block_candidates=2,
            commit_draft_fusion=True,
            speculative_commit_gate=True,
        )
