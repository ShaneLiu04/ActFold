"""AR035 tests: confidence-gated speculative commit (composition arm)."""

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


def _spec_cfg(**overrides):
    base = dict(
        history_cache=True,
        num_block_candidates=2,
        candidate_top_k=4,
        candidate_include_greedy=True,
    )
    base.update(overrides)
    return _cfg(**base)


def test_ut3501_default_off_zero_regression() -> None:
    llm = _build_llm(block_size=8)
    prompt = _prompt()
    a = FastDLLMSampler(llm, config=_spec_cfg(block_size=8, small_block_size=4)).sample(prompt)
    b = FastDLLMSampler(llm, config=_spec_cfg(block_size=8, small_block_size=4)).sample(prompt)
    assert torch.equal(a.sequences, b.sequences)
    sampler = FastDLLMSampler(llm, config=_spec_cfg(block_size=8, small_block_size=4))
    sampler.sample(prompt)
    assert sampler.history_cache_stats["spec_gated_iterations"] == 0


@pytest.mark.parametrize("threshold", [0.5, 0.95])
def test_ut3502_gated_mask_free_and_deterministic(threshold: float) -> None:
    llm = _build_llm(block_size=8)
    prompt = _prompt()
    cfg = _spec_cfg(
        block_size=8,
        small_block_size=4,
        num_tokens=16,
        threshold=threshold,
        speculative_commit_gate=True,
    )
    s1 = FastDLLMSampler(llm, config=cfg)
    out1 = s1.sample(prompt)
    st = s1.history_cache_stats
    assert _MASK_ID not in out1.sequences
    assert st["spec_gated_iterations"] >= 1
    assert st["spec_gated_iterations"] == st["spec_verify_forwards"]
    assert st["spec_draft_forwards"] == st["spec_verify_forwards"]
    assert st["degraded_calls"] == 0
    out2 = FastDLLMSampler(llm, config=cfg).sample(prompt)
    assert torch.equal(out1.sequences, out2.sequences)


def test_ut3503_forced_progress_every_round() -> None:
    llm = _build_llm(block_size=8)
    prompt = _prompt()
    sampler = FastDLLMSampler(
        llm,
        config=_spec_cfg(
            block_size=8,
            small_block_size=4,
            num_tokens=16,
            threshold=1.1,  # nothing clears the gate -> force one position
            speculative_commit_gate=True,
        ),
    )
    out = sampler.sample(prompt)
    st = sampler.history_cache_stats
    assert _MASK_ID not in out.sequences
    # 2 blocks x 4 masked positions = 8 forced rounds minimum
    assert st["spec_gated_iterations"] >= 8
    assert st["spec_gated_iterations"] == st["spec_verify_forwards"]


def test_ut3504_ledger_schema() -> None:
    llm = _build_llm(block_size=8)
    sampler = FastDLLMSampler(
        llm,
        config=_spec_cfg(block_size=8, small_block_size=4, speculative_commit_gate=True),
    )
    sampler.sample(_prompt())
    stats = sampler.history_cache_stats
    assert set(stats) == {
        "prefill_forwards",
        "slice_forwards",
        "commit_forwards",
        "block_rebuilds",
        "forward_tokens",
        "degraded_calls",
        "spec_draft_forwards",
        "spec_verify_forwards",
        "spec_gated_iterations",
        "spec_fused_forwards",
        "spec_budget_skips",
        "spec_budget_escalations",
        "spec_verification_free_blocks",
        "free_verify_blocks",
        "free_verify_low_conf",
        "spec_remask_fallbacks",
        "spec_chunk_forwards",
        "spec_refine_forwards",
    }


def test_ut3505_illegal_combinations() -> None:
    with pytest.raises(ValueError, match="speculative_commit_gate"):
        FastDLLMSamplerConfig(speculative_commit_gate=True)
    with pytest.raises(ValueError, match="speculative_commit_gate"):
        FastDLLMSamplerConfig(history_cache=True, speculative_commit_gate=True)  # no candidates
    # legal face
    FastDLLMSamplerConfig(history_cache=True, num_block_candidates=2, speculative_commit_gate=True)
