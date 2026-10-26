"""Fast-dLLM sampler presets: the measured-best stacks (AR043).

Every preset is a validated :class:`FastDLLMSamplerConfig`. The presets exist so
downstream users do not have to rediscover which AR knobs compose and which were
measured to be useless. See ``specs/archive/AR043-stack-consolidation/design.md``
for the evidence behind each choice and the rejected-knob list.

Presets
-------
``baseline``
    Official Fast-dLLM v2 recipe as implemented before AR033 (no history KV, no
    speculation). Reference for quality and latency comparisons.
``quality``
    The quality-safe stack: block-causal history KV + the official thresholded
    remasking, i.e. AR041's fallback tier with an always-fallback threshold.
    Measured: mean log p bit-identical to ``baseline`` at ~1.4-1.5x lower wall
    clock (history KV only, no speculative commit).
``balanced`
    Refine-after-fill stack (AR044): one greedy fill per block followed by two
    full-block refine passes that rewrite positions still below the threshold.
    Measured -0.9483 mean log p vs -1.5499 (kv-only) and -1.3047 (latency) at
    96 tokens on one prompt, 9.6 s vs 5.4 s -> the fast-and-quality point.
    Single-prompt evidence; validate on the matrix before citing as a
    distribution.
`latency``
    The fastest measured stack: history KV + greedy-anchored speculative block
    commit (AR034) + commit-draft fusion (AR036) + adaptive candidate budget
    (AR037) + verification-free greedy commit with free-verification re-arm
    (AR039). Measured ~19x vs ``baseline``; quality is workload-dependent
    (AR040), so validate with ``scripts/ar040_matrix_bench.py`` first.

Rejected knobs (measured not to improve the best stacks; keep them off)
----------------------------------------------------------------------
``speculative_commit_gate`` (AR035)        cost 9.58x vs the composition arm
``candidate_budget_escalate`` (AR038)      zero quality change, +46% wall clock
``commit_chunk_size`` (AR042)              chaotic: no monotone quality/chunk
``history_small_block_cache``              incompatible with speculation in v1
``uncertainty_validation_order``           incompatible with history_cache
"""

from __future__ import annotations

from typing import Any

from actfold.models.fast_dllm_sampler import FastDLLMSamplerConfig

__all__ = ["FASTDLLM_PRESETS", "PRESET_REJECTED_KNOBS", "fastdllm_preset"]

FASTDLLM_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "quality": {
        # AR043 measurement: the AR041 always-fallback tier costs one wasted
        # draft forward per block and lands at 1.45x, while the KV cache alone
        # reaches 1.51x with the SAME (bit-identical) log p. The quality-safe
        # stack is therefore the official remasking recipe running on the
        # block-causal history KV (no speculation at all).
        "history_cache": True,
        "history_small_block_cache": False,
    },
    "balanced": {
        # AR044 measured on the reference checkpoint (96 tokens, single prompt):
        # refine-after-fill reaches mean log p -0.9483 versus -1.5499 for the
        # KV-cached official decode and -1.3047 for the latency stack, at 9.6 s
        # against 5.4 s. That is the missing "fast AND quality" point. Single-
        # prompt evidence only: run scripts.ar040_matrix_bench / ar044_refine_probe
        # on more prompts before quoting it as a distribution.
        "history_cache": True,
        "history_small_block_cache": False,
        "num_block_candidates": 4,
        "candidate_top_k": 10,
        "candidate_include_greedy": True,
        "commit_refine_passes": 2,
        "commit_refine_threshold": 0.5,
    },
    "latency": {
        "history_cache": True,
        "history_small_block_cache": False,
        "num_block_candidates": 4,
        "candidate_top_k": 10,
        "candidate_include_greedy": True,
        "commit_draft_fusion": True,
        "adaptive_candidate_budget": True,
        "candidate_budget_threshold": 0.5,
        "verification_free_commit": True,
        "free_verify_floor": 0.3,
    },
}

PRESET_REJECTED_KNOBS: dict[str, str] = {
    "speculative_commit_gate": "AR035: 9.58x cost vs the composition arm (honest negative)",
    "candidate_budget_escalate": "AR038: zero log-p change, +46% wall clock (measured bound)",
    "commit_chunk_size": "AR042: quality is not monotone in chunk (chaotic, H1 refuted)",
    "remask_fallback": (
        "AR041: identical log p to kv-only but 1.45x vs 1.51x (one wasted draft "
        "per block); only useful as a safety tier on top of speculation"
    ),
    "history_small_block_cache": "incompatible with speculation in v1 (ValueError)",
    "uncertainty_validation_order": "incompatible with history_cache (ValueError)",
}


def fastdllm_preset(name: str, **overrides: Any) -> FastDLLMSamplerConfig:
    """Build a validated :class:`FastDLLMSamplerConfig` for a named preset.

    Args:
        name: one of ``"baseline"``, ``"quality"``, ``"latency"``.
        **overrides: config fields applied on top of the preset (e.g.
            ``num_tokens=160, stop_token_id=-1``).

    Returns:
        The validated sampler config.

    Raises:
        KeyError: if ``name`` is unknown (the message lists valid presets).
        ValueError: if an override violates a config invariant.
    """
    if name not in FASTDLLM_PRESETS:
        raise KeyError(f"unknown preset {name!r}; valid presets: {sorted(FASTDLLM_PRESETS)}")
    fields = dict(FASTDLLM_PRESETS[name])
    fields.update(overrides)
    return FastDLLMSamplerConfig(**fields)
