"""AR043 tests: measured-best presets + rejected knobs stay off."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from actfold.models.fast_dllm_sampler import FastDLLMSamplerConfig  # noqa: E402
from actfold.models.sampler_presets import (  # noqa: E402
    FASTDLLM_PRESETS,
    PRESET_REJECTED_KNOBS,
    fastdllm_preset,
)


def test_ut4301_all_presets_construct() -> None:
    for name in FASTDLLM_PRESETS:
        cfg = fastdllm_preset(name)
        assert isinstance(cfg, FastDLLMSamplerConfig)


def test_ut4302_quality_preset_shape() -> None:
    cfg = fastdllm_preset("quality", num_tokens=64)
    assert cfg.history_cache is True
    # AR043: the always-fallback tier was measured slower than kv-only at the
    # same log p, so the quality preset is the KV-cached official recipe.
    assert cfg.remask_fallback is False
    assert cfg.num_block_candidates == 0
    assert cfg.verification_free_commit is False
    assert cfg.commit_chunk_size == 0
    assert cfg.num_tokens == 64


def test_ut4303_latency_preset_shape() -> None:
    cfg = fastdllm_preset("latency")
    assert cfg.commit_draft_fusion is True
    assert cfg.adaptive_candidate_budget is True
    assert cfg.verification_free_commit is True
    assert cfg.remask_fallback is False
    assert cfg.speculative_commit_gate is False


def test_ut4307_balanced_preset_is_refine_after_fill() -> None:
    cfg = fastdllm_preset("balanced")
    assert cfg.history_cache is True
    assert cfg.commit_refine_passes == 2
    assert cfg.commit_refine_threshold == 0.5
    assert cfg.num_block_candidates == 4


def test_ut4304_rejected_knobs_are_off_in_every_preset() -> None:
    for name in FASTDLLM_PRESETS:
        cfg = fastdllm_preset(name)
        assert cfg.speculative_commit_gate is False
        assert cfg.candidate_budget_escalate is False
        assert cfg.commit_chunk_size == 0
        assert cfg.history_small_block_cache is False
        assert cfg.uncertainty_validation_order is False


def test_ut4305_unknown_preset_and_bad_override() -> None:
    with pytest.raises(KeyError, match="unknown preset"):
        fastdllm_preset("turbo")
    with pytest.raises(ValueError):
        fastdllm_preset("quality", commit_chunk_size=-1)


def test_ut4306_rejected_knob_reasons_documented() -> None:
    assert set(PRESET_REJECTED_KNOBS) == {
        "speculative_commit_gate",
        "candidate_budget_escalate",
        "commit_chunk_size",
        "remask_fallback",
        "history_small_block_cache",
        "uncertainty_validation_order",
    }
    for reason in PRESET_REJECTED_KNOBS.values():
        assert reason.strip()
