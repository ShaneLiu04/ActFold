"""AR042 tests: chunked block commit (progressive in-block unmasking)."""

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


def _cfg_cbc(chunk: int, **overrides):
    base = dict(
        history_cache=True,
        num_block_candidates=2,
        candidate_top_k=4,
        candidate_include_greedy=True,
        commit_chunk_size=chunk,
        block_size=8,
        small_block_size=8,
        num_tokens=24,
    )
    base.update(overrides)
    return _cfg(**base)


def test_ut4201_default_off_zero_chunk_forwards() -> None:
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
    assert sampler.history_cache_stats["spec_chunk_forwards"] == 0


def test_ut4202_chunk_one_forward_per_fill() -> None:
    llm = _build_llm(block_size=8)
    sampler = FastDLLMSampler(llm, config=_cfg_cbc(1))
    out = sampler.sample(_prompt())
    st = sampler.history_cache_stats
    blocks = st["spec_verify_forwards"] + st["spec_chunk_forwards"]
    assert st["spec_chunk_forwards"] > 0
    assert st["spec_verify_forwards"] == 0
    assert blocks > 0
    assert _MASK_ID not in out.sequences
    assert st["degraded_calls"] == 0


def test_ut4203_large_chunk_matches_one_shot() -> None:
    llm = _build_llm(block_size=8)
    prompt = _prompt()
    big = FastDLLMSampler(llm, config=_cfg_cbc(64)).sample(prompt)
    assert _MASK_ID not in big.sequences
    one = FastDLLMSampler(
        llm,
        config=_cfg(
            history_cache=True,
            num_block_candidates=2,
            candidate_top_k=4,
            candidate_include_greedy=True,
            block_size=8,
            small_block_size=8,
            num_tokens=24,
        ),
    ).sample(prompt)
    assert torch.equal(big.sequences, one.sequences)


def test_ut4204_deterministic() -> None:
    llm = _build_llm(block_size=8)
    prompt = _prompt()
    a = FastDLLMSampler(llm, config=_cfg_cbc(3)).sample(prompt)
    b = FastDLLMSampler(llm, config=_cfg_cbc(3)).sample(prompt)
    assert torch.equal(a.sequences, b.sequences)


def test_ut4205_illegal_values() -> None:
    with pytest.raises(ValueError, match="commit_chunk_size"):
        FastDLLMSamplerConfig(commit_chunk_size=-1)
    with pytest.raises(ValueError, match="commit_chunk_size"):
        FastDLLMSamplerConfig(commit_chunk_size=4)
