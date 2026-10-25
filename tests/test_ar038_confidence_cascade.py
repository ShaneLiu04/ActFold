"""AR038 tests: confidence-scheduled verification cascade (skip/standard/escalate)."""

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

from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig  # noqa: E402


def _cfg_cas(**overrides):
    base = dict(
        history_cache=True,
        num_block_candidates=2,
        candidate_top_k=4,
        candidate_include_greedy=True,
        candidate_budget_escalate=True,
        escalate_candidates=4,
        escalate_threshold=1.1,
        block_size=8,
        small_block_size=8,
        num_tokens=48,
    )
    base.update(overrides)
    return _cfg(**base)


def test_ut3801_default_off_zero_escalations() -> None:
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
    assert sampler.history_cache_stats["spec_budget_escalations"] == 0


def test_ut3802_always_escalates_and_ledger_matches() -> None:
    llm = _build_llm(block_size=8)
    sampler = FastDLLMSampler(llm, config=_cfg_cas())
    out = sampler.sample(_prompt())
    st = sampler.history_cache_stats
    assert st["spec_budget_escalations"] > 0
    assert st["spec_budget_escalations"] == st["spec_verify_forwards"]
    assert (
        st["spec_draft_forwards"] + st["spec_fused_forwards"]
        == st["spec_verify_forwards"] + st["spec_budget_skips"]
    )
    assert out.sequences.numel() > 0


def test_ut3803_standard_band_is_byte_identical() -> None:
    llm = _build_llm(block_size=8)
    prompt = _prompt()
    cast = FastDLLMSampler(
        llm,
        config=_cfg_cas(
            escalate_threshold=0.0,
            num_tokens=48,
        ),
    ).sample(prompt)
    assert torch.equal(cast.sequences, cast.sequences)
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
    assert torch.equal(cast.sequences, off.sequences)


def test_ut3804_escalate_must_widen() -> None:
    with pytest.raises(ValueError, match="escalate_candidates"):
        FastDLLMSamplerConfig(
            history_cache=True,
            num_block_candidates=4,
            candidate_budget_escalate=True,
            escalate_candidates=4,
        )


def test_ut3805_illegal_combinations() -> None:
    with pytest.raises(ValueError, match="candidate_budget_escalate"):
        FastDLLMSamplerConfig(candidate_budget_escalate=True)
    with pytest.raises(ValueError, match="not composed"):
        FastDLLMSamplerConfig(
            history_cache=True,
            num_block_candidates=2,
            candidate_budget_escalate=True,
            speculative_commit_gate=True,
        )
