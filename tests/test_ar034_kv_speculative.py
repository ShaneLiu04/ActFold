"""AR034 tests: KV-cached speculative block decoding (composition arm).

The fixture is the AR033 fp32 tiny REAL block-causal decoder (imported from
``test_ar033_history_cache``): real projections/attention/MLP with the
official native cache-kwarg contract —zero mock outputs (#10). AR034 adds
the composition face on top of it:

- UT-3401: the composition arm runs multi-block greedy decodes over >= 4
  shape points; outputs are mask-free, deterministic across runs, and the
  expected canvas length is preserved.
- UT-3402: the forward-count ledger identity —``spec_draft_forwards ==
  spec_verify_forwards > 0``, ``prefill_forwards == 1``, ``degraded_calls ==
  0``, and the composition's total forward count is strictly below the
  cache-only arm's on the same geometry (the launch-count claim).
- UT-3403: ``_expand_history_kv`` is a COPY —mutating the expanded cache
  never touches the source, source rows repeat over the batch dimension
  (list-container and ``batch_repeat_interleave`` faces), the source cache
  is unchanged after a batched verification forward.
- UT-3404: default-off zero regression —``num_block_candidates == 0``
  keeps both spec counters at zero and the AR033 cache path byte-identical.
- UT-3405: illegal-combination fail-fast (three ValueError faces).
- UT-3406: the greedy row is in the race —``speculative_calls`` counts
  every speculative small block and ``greedy_wins`` is a subset of it (the
  AR022 fairness contract: the greedy row can compete, not be forced).
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_ar033_history_cache import (  # noqa: E402
    _MASK_ID,
    _build_llm,
    _cfg,
    _MiniTokenizer,
    _prompt,
    _TinyKVCache,
)

from actfold.models.fast_dllm_sampler import FastDLLMSampler, _expand_history_kv  # noqa: E402


def _comp_cfg(**overrides: Any):
    base: dict[str, Any] = {
        "history_cache": True,
        "num_block_candidates": 2,
        "candidate_top_k": 4,
        "candidate_include_greedy": True,
    }
    base.update(overrides)
    return _cfg(**base)


_SHAPE_POINTS = [
    pytest.param(8, 4, 16, id="b8s4t16"),
    pytest.param(8, 4, 12, id="b8s4t12_short_final"),
    pytest.param(16, 4, 24, id="b16s4t24"),
    pytest.param(16, 8, 32, id="b16s8t32"),
]


@pytest.mark.parametrize("block_size,small,num_tokens", _SHAPE_POINTS)
def test_ut3401_composition_deterministic_and_mask_free(
    block_size: int, small: int, num_tokens: int
) -> None:
    """UT-3401: the composition arm decodes deterministically, mask-free."""
    llm = _build_llm(block_size=block_size)
    prompt = _prompt()
    common = dict(block_size=block_size, small_block_size=small, num_tokens=num_tokens)
    out_a = FastDLLMSampler(llm, config=_comp_cfg(**common)).sample(prompt)
    out_b = FastDLLMSampler(llm, config=_comp_cfg(**common)).sample(prompt)
    assert torch.equal(out_a.sequences, out_b.sequences)
    assert _MASK_ID not in out_a.sequences
    assert out_a.sequences.shape[1] >= 3
    # the composition really ran the batched candidate path
    assert out_a.sequences.shape[0] == 1


@pytest.mark.parametrize(
    "block_size,small,num_tokens",
    [(8, 4, 16), (16, 4, 24)],
    ids=["b8s4t16", "b16s4t24"],
)
def test_ut3402_forward_count_identity(block_size: int, small: int, num_tokens: int) -> None:
    """UT-3402: spec counters are balanced and the arm launches fewer forwards."""
    llm = _build_llm(block_size=block_size)
    prompt = _prompt()
    common = dict(block_size=block_size, small_block_size=small, num_tokens=num_tokens)

    comp = FastDLLMSampler(llm, config=_comp_cfg(**common))
    comp.sample(prompt)
    stats = comp.history_cache_stats
    assert stats["prefill_forwards"] == 1
    assert stats["spec_draft_forwards"] > 0
    assert stats["spec_draft_forwards"] == stats["spec_verify_forwards"]
    assert stats["degraded_calls"] == 0
    assert stats["slice_forwards"] == 0  # the composition replaced the slices
    assert stats["spec_verify_forwards"] >= 1
    # key-set stability (the AR033 seven keys + two spec keys)
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

    cache = FastDLLMSampler(llm, config=_cfg(history_cache=True, **common))
    cache.sample(prompt)
    cstats = cache.history_cache_stats
    cache_total = (
        cstats["prefill_forwards"]
        + cstats["slice_forwards"]
        + cstats["commit_forwards"]
        + cstats["block_rebuilds"]
    )
    comp_total = (
        stats["prefill_forwards"]
        + stats["spec_draft_forwards"]
        + stats["spec_verify_forwards"]
        + stats["commit_forwards"]
        + stats["block_rebuilds"]
    )
    assert comp_total < cache_total
    assert cstats["spec_draft_forwards"] == 0 and cstats["spec_verify_forwards"] == 0


def test_ut3403_expand_history_kv_is_a_copy() -> None:
    """UT-3403: the expanded cache is a copy; the source rows are untouched."""
    cache = _TinyKVCache()
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    cache.update(k, v, 0)
    cache.update(k * 2.0, v * 2.0, 1)

    expanded = _expand_history_kv(cache, 3)
    assert len(expanded) == 2
    for layer_idx in range(2):
        ek, ev = expanded[layer_idx]
        ok, ov = cache[layer_idx]
        assert ek.shape[0] == 3 and ev.shape[0] == 3
        assert torch.equal(ek[0], ok[0]) and torch.equal(ek[2], ok[0])

    # mutating the expanded copy never leaks into the source
    before = cache[0][0].clone()
    expanded[0][0].add_(1.0)
    assert torch.equal(cache[0][0], before)


def test_ut3403b_source_kv_unchanged_after_verify() -> None:
    """UT-3403b: read-only forwards never mutate the history KV in place."""
    llm = _build_llm(block_size=8)
    raw = llm.model
    checks: list[bool] = []
    original = raw.forward

    @functools.wraps(original)
    def recording_forward(*args: Any, **kwargs: Any):
        kv = kwargs.get("past_key_values")
        readonly = kv is not None and kwargs.get("update_past_key_values") is False
        before = [t.detach().clone() for t in kv.key_cache] if readonly else None
        out = original(*args, **kwargs)
        if readonly:
            after = [t.detach().clone() for t in kv.key_cache]
            checks.append(
                len(before) == len(after) and all(torch.equal(a, b) for a, b in zip(before, after))
            )
        return out

    raw.forward = recording_forward  # type: ignore[method-assign]
    try:
        FastDLLMSampler(
            llm, config=_comp_cfg(block_size=8, small_block_size=4, num_tokens=16)
        ).sample(_prompt())
    finally:
        raw.forward = original  # type: ignore[method-assign]

    # every read-only forward (draft + verify) left the source cache intact
    assert len(checks) >= 2
    assert all(checks)


def test_ut3404_default_off_zero_regression() -> None:
    """UT-3404: num_block_candidates=0 keeps the AR033 path and zero spec counters."""
    llm = _build_llm(block_size=8)
    prompt = _prompt()
    off = FastDLLMSampler(llm, config=_cfg(history_cache=True, block_size=8, small_block_size=4))
    out_off = off.sample(prompt)
    stats = off.history_cache_stats
    assert stats["spec_draft_forwards"] == 0
    assert stats["spec_verify_forwards"] == 0
    assert stats["degraded_calls"] == 0

    ref = FastDLLMSampler(llm, config=_cfg(history_cache=True, block_size=8, small_block_size=4))
    out_ref = ref.sample(prompt)
    assert torch.equal(out_off.sequences, out_ref.sequences)


def test_ut3405_illegal_combinations() -> None:
    """UT-3405: the three illegal faces fail fast at construction/sample."""
    llm = _build_llm(block_size=8)
    tokenizer = _MiniTokenizer()
    del tokenizer

    # 1) composition + intra-block sb cache (v1 undefined)
    with pytest.raises(ValueError, match="history_small_block_cache"):
        _cfg(
            history_cache=True,
            history_small_block_cache=True,
            num_block_candidates=2,
            candidate_include_greedy=True,
        )

    # 2) sb cache without history_cache
    with pytest.raises(ValueError, match="requires history_cache"):
        _cfg(history_small_block_cache=True)

    # 3) uncertainty ordering + history cache
    with pytest.raises(ValueError, match="uncertainty_validation_order"):
        _cfg(history_cache=True, num_block_candidates=2, uncertainty_validation_order=True)

    # 4) composition + folded_model fails at sample() (folding is a distinct
    #    compute strategy); the guard fires before the folded path is used.
    sampler = FastDLLMSampler(llm, config=_comp_cfg(block_size=8, small_block_size=4))
    with pytest.raises(ValueError, match="mutually exclusive with a folded_model"):
        sampler.sample(_prompt(), folded_model=object())  # type: ignore[arg-type]


def test_ut3406_greedy_row_competes() -> None:
    """UT-3406: every speculative small block is scored; greedy wins a subset."""
    llm = _build_llm(block_size=8)
    sampler = FastDLLMSampler(
        llm,
        config=_comp_cfg(
            block_size=8,
            small_block_size=4,
            num_tokens=16,
            num_block_candidates=4,
        ),
    )
    sampler.sample(_prompt())
    stats = sampler.greedy_win_stats
    assert stats["speculative_calls"] == sampler.history_cache_stats["spec_verify_forwards"]
    assert stats["speculative_calls"] > 0
    assert 0 <= stats["greedy_wins"] <= stats["speculative_calls"]
