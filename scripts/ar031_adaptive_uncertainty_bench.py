"""AR031/T011: three-arm adaptive-budget + uncertainty-order benchmark (real ckpt).

Runs the REAL ``FastDLLMSampler.sample`` generation loop on the local
Fast-dLLM-v2-1.5B checkpoint in THREE arms over one shared model instance,
prompt set, and seed (the AR022 protocol carried over; the shared helpers
are IMPORTED from ``scripts.ar021_generation_loop_bench``; the folded arms
run the EAGER primary face -- the real CUDA-graph linkage face is
BLOCKED-DEFERRED, see ``ARTIFACT_LIMITATION_NOTES``):

* ``baseline`` -- the official Fast-dLLM v2 recipe (``folded_model=None``).
* ``adaptive-off`` -- MFF + ``num_block_candidates=4`` +
  ``candidate_include_greedy=True`` (the AR022-aligned config) +
  ``LayerRecomputeBudgetPolicy(adaptive=False)`` (permanent uniform weights,
  the cap ACTIVE) + ``uncertainty_validation_order=False`` (full
  validation).
* ``full`` -- the same construction as ``adaptive-off`` but with
  ``adaptive=True`` (windowed layer-adaptive weights) AND
  ``uncertainty_validation_order=True`` (prior-descending validation with
  the frozen margin/margin-gap early stop). The ②-vs-③ difference is
  exactly the two mechanism switches -- the attribution arm.

Hypotheses (declared before measurement; the verdict functions below are
frozen BEFORE any data is collected; honest negatives are delivered
verbatim, never rerun-until-pass):

* **H1** quality alignment (③ vs ①): the merged absolute posthoc overall
  confidence gap <= 1e-3 (+1e-9 IEEE noise margin); per-prompt decode
  length recovery; merged block-grid exact-match ③ >= ②. Any ``None``
  posthoc mean refutes. The ③-vs-② attribution gap ratio is report-only
  (sentinels 0.0/inf when the ② gap is zero, the AR023 caliber).
* **H2** p50 speedup (① vs ③): ``p50(①)/p50(③) > 1.0`` strictly AND
  ``delta > 2 * merged std`` with the AR017 two-sample merged std.
* **H3** mechanism attribution (② vs ③): the same caliber on
  ``p50(②)/p50(③)``; report-only per-arm stable ratios, uncertainty skip
  stats, and budget window events.
* **H4** honest ledger: early-stop winner equivalence rate == 100%
  (every speculative unit of the full arm is re-scored against the full
  vectorized min-prob argmax on the untimed sidecar pass -- scoring is
  cheap, so the face is measured exhaustively); budget upper bound zero
  violations (per capped layer-forward the non-exempt recompute must stay
  <= ``B_l``; the probe exempts forced and NaN-score positions, the two
  constructive exemption classes); the publish ledger stays within the
  #45 ``LEDGER_C`` caliber; the internal reproduction double-run must be
  strictly row-equal -- any mismatch blocks ALL verdicts (one-vote veto).

Protocol: seed 2026, P=2 pre-padded prompts, num_tokens=128 (W=4) formal /
64 (W=2) quick; warmup one full discarded sample per (arm, prompt); the
untimed sidecar quality pass runs TWICE (pass A records the quality faces,
pass B is the reproduction self-check); per-rep three-arm round-robin (rep
r starts at r%3); host perf_counter + cuda synchronize (#39); reps=30
pre-committed single-shot verdict. ``--quick``: reps=3, single prompt,
num_tokens=64, separate artifact path, never the formal one.

Environment gate (construction time, before any model load or artifact
write): ``transformers==4.53.1`` (the AR019 venv) and a local checkpoint
directory containing ``config.json``; failures exit non-zero and write
nothing.

Run (AR019 venv, transformers==4.53.1)::

    python -m scripts.ar031_adaptive_uncertainty_bench [--quick] [--reps 30]
        [--model-path PATH] [--out PATH] [--hf-home PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from actfold.core import LayerRecomputeBudgetPolicy  # noqa: E402
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig  # noqa: E402
from scripts.algo_experiments import stats_from_samples  # noqa: E402
from scripts.ar021_generation_loop_bench import (  # noqa: E402
    BLOCK,
    CACHE_BUDGET,
    CANDIDATE_TOP_K,
    EPS,
    FULL_NUM_TOKENS,
    LEDGER_C,
    NUM_CANDIDATES,
    QUICK_NUM_TOKENS,
    SEED,
    TAU,
    _BenchDiffusionLLM,
    _BenchMFF,
    _block_exact_match,
    _gpu_driver_version,
    _load_model,
    _posthoc_confidence,
    _prepare_prompts,
    _resolve_model_path,
    _sampler_config,
    write_results,
)

SCHEMA = "ar031_adaptive_uncertainty_bench"
SCHEMA_VERSION = 1
ARMS = ("baseline", "adaptive-off", "full")
DEFAULT_OUTPUT_PATH = Path("results/optimization/ar031_adaptive_uncertainty_bench.json")
QUICK_OUTPUT_PATH = Path("results/optimization/ar031_adaptive_uncertainty_bench_quick.json")
REPS_DEFAULT = 30
QUICK_REPS = 3
REQUIRED_TRANSFORMERS = "4.53.1"
UNCERTAINTY_MARGIN = 0.9
UNCERTAINTY_MARGIN_GAP = 0.05
BUDGET_RATIO = 0.15
BUDGET_WARMUP = 5
BUDGET_WINDOW = 10
_H1_GAP_MAX = 1e-3
_H1_RATIO_EPS = 1e-9
_BASELINE = "baseline"
_OFF = "adaptive-off"
_FULL = "full"

ARTIFACT_LIMITATION_NOTES: tuple[str, ...] = (
    "eager is the primary declaration face of this benchmark: the folded "
    "arms run ManualFoldedForward with use_cuda_graph=False (the WDDM "
    "16GB host protocol); all AR031 mechanisms (budget cap, windowed "
    "re-estimation, uncertainty-ordered validation) are exercised on the "
    "eager path",
    "the real-checkpoint CUDA-graph linkage face (graph x budget, the A4 "
    "replay-time budget backfill) is BLOCKED-DEFERRED: it requires a quiet "
    ">=24 GiB TCC/Linux GPU per the AR030 formal-face standard and is "
    "recorded here as a limitation, never as a measured claim",
    "timing follows the #39 wall-clock discipline (perf_counter + "
    "synchronize, warmup discarded, per-rep round-robin); the Windows WDDM "
    "display-driver mode does not lock clocks -- p50 over reps with "
    "interleaving is the noise mitigation",
)

_HYPOTHESES: dict[str, str] = {
    "h1": (
        "quality alignment (full vs baseline): the merged absolute posthoc "
        "overall-confidence gap <= 1e-3 (+1e-9 IEEE margin); per-prompt "
        "decode length recovery; merged block-grid exact-match full >= "
        "adaptive-off; any None posthoc mean refutes; the full-vs-off gap "
        "ratio is report-only (sentinels 0.0/inf when the off gap is 0, "
        "the AR023 caliber)"
    ),
    "h2": (
        "p50 speedup (baseline vs full): p50(baseline)/p50(full) > 1.0 "
        "strictly AND delta > 2 * merged std (AR017 two-sample caliber, "
        "merged std = sqrt(std_a^2 + std_b^2)); else refuted"
    ),
    "h3": (
        "mechanism attribution (adaptive-off vs full): p50(off)/p50(full) "
        "> 1.0 strictly AND delta > 2 * merged std (the same caliber); "
        "per-arm stable ratios, uncertainty skip stats, and budget window "
        "events are report-only faces"
    ),
    "h4": (
        "honest ledger: early-stop winner equivalence rate == 100% (every "
        "speculative unit re-scored against the full vectorized min-prob "
        "argmax on the untimed sidecar); budget upper bound zero "
        "violations (non-exempt recompute <= B_l per capped layer-forward, "
        "forced and NaN-score positions exempt); publish ledger within the "
        "#45 LEDGER_C caliber; the internal reproduction double-run "
        "strictly row-equal -- any mismatch blocks ALL verdicts"
    ),
}


# ---------------------------------------------------------------------------
# Frozen verdict functions (declared before any data collection; the UT
# suite pins every boundary; honest negatives are returned verbatim)
# ---------------------------------------------------------------------------
def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def verdict_h1(quality_faces: dict[str, Any]) -> dict[str, Any]:
    """H1 quality alignment: three faces ALL passing -> supported."""
    per_prompt = quality_faces.get("per_prompt") or []
    gaps_full: list[float] = []
    gaps_off: list[float] = []
    none_mean = False
    for entry in per_prompt:
        means = entry.get("posthoc_overall_mean", {})
        base_mean = means.get(_BASELINE)
        full_mean = means.get(_FULL)
        off_mean = means.get(_OFF)
        if base_mean is None or full_mean is None or off_mean is None:
            none_mean = True
            gaps_full.append(math.nan)
            gaps_off.append(math.nan)
        else:
            gaps_full.append(full_mean - base_mean)
            gaps_off.append(off_mean - base_mean)
    merged_gap_full = sum(gaps_full) / len(gaps_full) if gaps_full else math.nan
    merged_gap_off = sum(gaps_off) / len(gaps_off) if gaps_off else math.nan
    face_a = _finite(merged_gap_full) and abs(merged_gap_full) <= _H1_GAP_MAX + _H1_RATIO_EPS

    face_b = bool(per_prompt)
    decode_rows: list[dict[str, Any]] = []
    for entry in per_prompt:
        lens = entry.get("decode_length", {})
        len_base = lens.get(_BASELINE)
        len_full = lens.get(_FULL)
        ok = len_base is not None and len_full is not None and len_full == len_base
        face_b = face_b and ok
        decode_rows.append(
            {
                "prompt_index": entry.get("prompt_index"),
                "len_baseline": len_base,
                "len_off": lens.get(_OFF),
                "len_full": len_full,
                "matches_baseline": bool(ok),
            }
        )

    matched_full = sum(
        (entry["block_match_counts"]["full"] or {}).get("matched", 0) for entry in per_prompt
    )
    compared_full = sum(
        (entry["block_match_counts"]["full"] or {}).get("compared", 0) for entry in per_prompt
    )
    matched_off = sum(
        (entry["block_match_counts"]["adaptive-off"] or {}).get("matched", 0)
        for entry in per_prompt
    )
    compared_off = sum(
        (entry["block_match_counts"]["adaptive-off"] or {}).get("compared", 0)
        for entry in per_prompt
    )
    rate_full = matched_full / compared_full if compared_full else None
    rate_off = matched_off / compared_off if compared_off else None
    face_c = (
        rate_full is not None and rate_off is not None and rate_full >= rate_off - _H1_RATIO_EPS
    )
    if none_mean:
        face_a = False

    if _finite(merged_gap_off) and merged_gap_off == 0.0:
        attribution = 0.0 if (_finite(merged_gap_full) and merged_gap_full == 0.0) else math.inf
    elif _finite(merged_gap_off) and _finite(merged_gap_full):
        attribution = abs(merged_gap_full) / abs(merged_gap_off)
    else:
        attribution = None

    supported = bool(face_a and face_b and face_c)
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": {
            "a_abs_gap_le_1e-3": bool(face_a),
            "b_decode_length_recovery": bool(face_b),
            "c_merged_block_match_ge_off": bool(face_c),
        },
        "numbers": {
            "merged_gap_full": (round(merged_gap_full, 6) if _finite(merged_gap_full) else None),
            "per_prompt_gaps_full": [round(g, 6) if _finite(g) else None for g in gaps_full],
            "merged_gap_off": round(merged_gap_off, 6) if _finite(merged_gap_off) else None,
            "rate_full": round(rate_full, 6) if rate_full is not None else None,
            "rate_off": round(rate_off, 6) if rate_off is not None else None,
            "attribution_gap_ratio_vs_off": attribution,
            "per_prompt_decode_lengths": decode_rows,
            "none_mean_present": bool(none_mean),
        },
    }


def verdict_h2(
    p50_base: float | None, p50_full: float | None, std_merged: float | None
) -> dict[str, Any]:
    """H2 p50 speedup: strict ratio > 1.0 AND delta > 2 * merged std."""
    ratio: float | None = None
    delta: float | None = None
    threshold: float | None = None
    if (
        p50_base is not None
        and p50_full is not None
        and std_merged is not None
        and math.isfinite(p50_base)
        and math.isfinite(p50_full)
        and math.isfinite(std_merged)
        and p50_full != 0
    ):
        ratio = p50_base / p50_full
        delta = p50_base - p50_full
        threshold = 2 * std_merged
    supported = (
        ratio is not None
        and delta is not None
        and threshold is not None
        and ratio > 1.0
        and delta > threshold
    )
    return {
        "verdict": "supported" if supported else "refuted",
        "numbers": {
            "p50_base_ms": p50_base,
            "p50_full_ms": p50_full,
            "ratio": ratio,
            "delta_ms": delta,
            "std_merged_ms": std_merged,
            "threshold_2sigma_ms": threshold,
            "rule": "supported iff ratio > 1.0 strictly AND delta > 2 * merged std",
        },
    }


def verdict_h3(
    p50_off: float | None, p50_full: float | None, std_merged: float | None
) -> dict[str, Any]:
    """H3 mechanism attribution: the H2 caliber, off in the base slot.

    The numbers dict mirrors H2's key set exactly (``p50_base_ms`` holds
    the adaptive-off p50 -- the base slot of the comparison); the H3
    wording lives in the declared hypothesis text.
    """
    return verdict_h2(p50_off, p50_full, std_merged)


def repro_mismatch_rows(
    pass_a_seqs: dict[tuple[str, int], torch.Tensor],
    pass_b_seqs: dict[tuple[str, int], torch.Tensor],
    arms: Sequence[str],
    num_prompts: int,
) -> list[dict[str, Any]]:
    """Strict per-(arm, prompt) double-run comparison rows (the H4 veto input).

    Pass A must be keyed by ``(arm, prompt_index)`` exactly as pass B is --
    comparing a loop-leaked last-prompt dict against per-prompt pass-B
    sequences fabricates cross-prompt mismatches (the first formal run's
    66-vs-34 incident, UT-3166).  Rows report arm, prompt index and both
    decode lengths; an empty list means the reproduction face passes.
    """
    mismatches: list[dict[str, Any]] = []
    for pi in range(num_prompts):
        for arm in arms:
            seq_a = pass_a_seqs[(arm, pi)]
            seq_b = pass_b_seqs[(arm, pi)]
            if not bool(torch.equal(seq_a, seq_b)):
                mismatches.append(
                    {
                        "arm": arm,
                        "prompt_index": pi,
                        "pass_a_len": int(seq_a.shape[1]),
                        "pass_b_len": int(seq_b.shape[1]),
                    }
                )
    return mismatches


def verdict_h4(
    equiv_rate: float | None,
    budget_violations: int,
    ledger_ok: bool,
    repro_ok: bool,
) -> dict[str, Any]:
    """H4 honest ledger; a failed reproduction blocks (one-vote veto)."""
    if not repro_ok:
        equiv_ok = equiv_rate is not None and equiv_rate >= 1.0
        return {
            "verdict": "blocked",
            "blocked_reason": (
                "the internal reproduction double-run mismatched -- the "
                "same-environment determinism assumption failed wholesale, "
                "so no single-face verdict is trustworthy; numbers are "
                "preserved verbatim, never fabricated"
            ),
            "faces": {
                "winner_equivalence_100pct": bool(equiv_ok),
                "budget_zero_violations": int(budget_violations) == 0,
                "ledger_within_bound": bool(ledger_ok),
                "reproduction_ok": False,
            },
            "numbers": {"equiv_rate": equiv_rate, "budget_violations": int(budget_violations)},
        }
    equiv_ok = equiv_rate is not None and equiv_rate >= 1.0
    budget_ok = int(budget_violations) == 0
    supported = bool(equiv_ok and budget_ok and ledger_ok)
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": {
            "winner_equivalence_100pct": bool(equiv_ok),
            "budget_zero_violations": bool(budget_ok),
            "ledger_within_bound": bool(ledger_ok),
            "reproduction_ok": True,
        },
        "numbers": {"equiv_rate": equiv_rate, "budget_violations": int(budget_violations)},
    }


def overall_verdict(h1: str, h2: str, h3: str, h4: str) -> str:
    """Precedence: blocked -> refuted -> supported (AR029 caliber)."""
    verdicts = (h1, h2, h3, h4)
    if any(v == "blocked" for v in verdicts):
        return "blocked"
    if all(v == "supported" for v in verdicts):
        return "supported"
    return "refuted"


def assemble_verdicts(faces: dict[str, Any]) -> dict[str, Any]:
    """Pure verdict assembly + the reproduction one-vote veto."""
    h1 = verdict_h1(faces["quality_faces"])
    h2 = verdict_h2(faces["p50_base_ms"], faces["p50_full_ms"], faces["std_merged_h2_ms"])
    h3 = verdict_h3(faces["p50_off_ms"], faces["p50_full_ms"], faces["std_merged_h3_ms"])
    h4 = verdict_h4(
        faces["equiv_rate"], faces["budget_violations"], faces["ledger_ok"], faces["repro_ok"]
    )
    result: dict[str, Any] = {
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "overall": overall_verdict(h1["verdict"], h2["verdict"], h3["verdict"], h4["verdict"]),
    }
    if not faces["repro_ok"]:
        for key in ("h1", "h2", "h3", "h4"):
            result[key]["verdict"] = "blocked"
            result[key]["blocked_reason"] = (
                "the internal reproduction double-run mismatched -- all "
                "verdicts are blocked (one-vote veto); the recorded "
                "numbers are preserved verbatim"
            )
        result["overall"] = "blocked"
    return result


# ---------------------------------------------------------------------------
# Environment gate / output paths (construction time, before any model load)
# ---------------------------------------------------------------------------
def require_environment(transformers_version: str | None, model_path: Path | None) -> None:
    """Raise ``RuntimeError`` unless the real-face environment is satisfied."""
    if transformers_version != REQUIRED_TRANSFORMERS:
        raise RuntimeError(
            f"this benchmark requires transformers=={REQUIRED_TRANSFORMERS} "
            f"(the AR019 venv); got {transformers_version!r}"
        )
    if model_path is None:
        raise RuntimeError(
            "a local Fast-dLLM checkpoint directory is required "
            "(--model-path or the FASTDLLM_MODEL_PATH env var)"
        )
    if not model_path.is_dir():
        raise RuntimeError(f"model path is not an existing directory: {model_path}")
    if not (model_path / "config.json").is_file():
        raise RuntimeError(f"model path has no config.json: {model_path}")


def resolve_output_path(out_arg: Path | None, quick: bool) -> Path:
    """Explicit out wins; quick NEVER resolves to the formal artifact path."""
    if out_arg is not None:
        return out_arg
    return QUICK_OUTPUT_PATH if quick else DEFAULT_OUTPUT_PATH


# ---------------------------------------------------------------------------
# Zero-sync probes (sidecar-only readers; the timed path keeps them disabled)
# ---------------------------------------------------------------------------
class _RecordingBudgetPolicy(LayerRecomputeBudgetPolicy):
    """Budget policy with a zero-sync per-layer cap probe.

    ``budget_for`` and ``apply_layer_budget`` are called in immediate
    sequence per capped layer-forward, so the pending budget row is
    consumed exactly once. All recorded quantities are device 0-dim
    tensors -- the bench syncs them once per sidecar read, never inside a
    timed segment. ``probe_enabled=False`` (the timed-segment state) keeps
    the override to a bare ``super()`` call.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.probe_enabled = True
        self.records: list[dict[str, Any]] = []
        self._pending_layer = -1
        self._pending_budget: Any = None

    def budget_for(self, layer_idx: int, tokens_l: int) -> int:
        budget = super().budget_for(layer_idx, tokens_l)
        if self.probe_enabled:
            self._pending_layer = layer_idx
            self._pending_budget = budget
        return budget

    def apply_layer_budget(
        self,
        stable_mask: torch.Tensor,
        scores: torch.Tensor,
        budget: Any,
        forced: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = super().apply_layer_budget(stable_mask, scores, budget, forced)
        if self.probe_enabled:
            self.records.append(
                {
                    "layer": self._pending_layer,
                    "budget": int(self._pending_budget)
                    if self._pending_budget is not None
                    else None,
                    "tokens": int(stable_mask.numel()),
                    "pre_stable": stable_mask.sum(),
                    "post_stable": out.sum(),
                    "forced": forced.sum() if forced is not None else None,
                    "nan_scores": torch.isnan(scores).sum(),
                }
            )
            self._pending_layer = -1
            self._pending_budget = None
        return out


class _EquivalenceProbeSampler(FastDLLMSampler):
    """Full-arm sampler with the H4 winner-equivalence sidecar probe.

    With ``probe_rescore`` off (the timed-segment state) the override is a
    bare ``super()`` call -- zero perturbation of the timed trajectory. On
    the untimed sidecar pass every speculative unit's early-stopped winner
    is re-scored against the FULL vectorized min-prob argmax (the frozen
    off-branch formula over the same ``[N, T, V]`` verification logits) --
    scoring is cheap, so the equivalence face is measured exhaustively.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.probe_rescore = False
        self.probe_units: list[dict[str, Any]] = []

    def _uncertainty_select_inline(
        self,
        shifted: torch.Tensor,
        cands: torch.Tensor,
        small_logits: torch.Tensor,
        masked_local: torch.Tensor,
        sb_start: int,
        sb_end: int,
    ) -> int:
        j_star = super()._uncertainty_select_inline(
            shifted, cands, small_logits, masked_local, sb_start, sb_end
        )
        if self.probe_rescore:
            commit_logits = shifted[:, sb_start:sb_end, :]
            probs = F.softmax(commit_logits, dim=-1)
            committed = cands[:, sb_start:sb_end]
            scores = torch.gather(probs, -1, committed.unsqueeze(-1)).squeeze(-1)
            j_full = int(scores.min(dim=-1).values.argmax())
            self.probe_units.append(
                {
                    "j_star": int(j_star),
                    "j_full": j_full,
                    "match": int(j_star) == j_full,
                }
            )
        return j_star


def _sync_budget_records(records: list[dict[str, Any]]) -> list[dict[str, int]]:
    """One batched sync of the probe's device tensors -> host rows."""
    rows: list[dict[str, int]] = []
    for rec in records:
        pre_stable = int(rec["pre_stable"].item())
        post_stable = int(rec["post_stable"].item())
        forced = int(rec["forced"].item()) if rec["forced"] is not None else 0
        nan_scores = int(rec["nan_scores"].item())
        tokens = int(rec["tokens"])
        budget = int(rec["budget"]) if rec["budget"] is not None else 0
        nonexempt = (tokens - post_stable) - forced - nan_scores
        rows.append(
            {
                "layer": int(rec["layer"]),
                "tokens": tokens,
                "budget": budget,
                "pre_stable": pre_stable,
                "post_stable": post_stable,
                "forced": forced,
                "nan_scores": nan_scores,
                "nonexempt_recompute": nonexempt,
                "violation": nonexempt > budget,
            }
        )
    return rows


def _budget_probe_summary(rows: list[dict[str, int]]) -> dict[str, Any]:
    violations = [r for r in rows if r["violation"]]
    nonexempt = [r["nonexempt_recompute"] for r in rows] or [0]
    return {
        "capped_layer_forwards": len(rows),
        "budget_violations": len(violations),
        "max_nonexempt_over_budget": (
            max(r["nonexempt_recompute"] - r["budget"] for r in rows) if rows else None
        ),
        "mean_nonexempt_recompute": round(sum(nonexempt) / len(nonexempt), 4),
        "violation_rows": violations[:20],
    }


# ---------------------------------------------------------------------------
# Arm construction / local helper copies (the AR022 coupling-note pattern)
# ---------------------------------------------------------------------------
def _arm_sampler_config(num_tokens: int, full: bool) -> FastDLLMSamplerConfig:
    """The folded-arm config: the AR022-aligned base + the AR031 switches."""
    config: FastDLLMSamplerConfig = dataclasses.replace(
        _sampler_config(num_tokens, NUM_CANDIDATES),
        candidate_include_greedy=True,
    )
    if full:
        config = dataclasses.replace(
            config,
            uncertainty_validation_order=True,
            uncertainty_margin=UNCERTAINTY_MARGIN,
            uncertainty_margin_gap=UNCERTAINTY_MARGIN_GAP,
        )
    return config


def _make_eager_mff(model: Any, policy: _RecordingBudgetPolicy) -> _BenchMFF:
    """Eager primary face: graph off, budget policy attached (AR031 A7)."""
    from actfold.core.activation_cache import ActivationCache
    from actfold.core.similarity_gate import SimilarityGate

    cache = ActivationCache(max_entries_per_layer=CACHE_BUDGET, device="cuda")
    gate = SimilarityGate(tau=TAU, metric="cosine", eps=EPS)
    return _BenchMFF(
        model,
        cache=cache,
        gate=gate,
        scheduler=None,
        split_layers=False,
        use_cuda_graph=False,
        budget_policy=policy,
    )


def _ledger_delta(
    mff: _BenchMFF,
    ops0: int,
    s0: int,
    b0: int,
    num_layers: int,
) -> dict[str, Any]:
    """Publish-ledger delta over one sidecar generation (#45 caliber)."""
    ledger = {
        "ops_delta": mff.cache.write_op_count - ops0,
        "serial_events": mff.serial_calls - s0,
        "batched_events": mff.batched_calls - b0,
    }
    ledger["bound"] = LEDGER_C * (
        (num_layers + 1) * ledger["serial_events"]
        + (num_layers + NUM_CANDIDATES) * ledger["batched_events"]
    )
    ledger["within_bound"] = bool(ledger["ops_delta"] <= ledger["bound"])
    return ledger


def _merged_block_exact_match(
    base_seq: torch.Tensor,
    other_seq: torch.Tensor,
    padded_len: int,
    num_tokens: int,
) -> dict[str, Any]:
    """Raw matched/compared counts on the block grid (AR022 local copy)."""
    b = base_seq[0].tolist()
    c = other_seq[0].tolist()
    n = min(len(b), len(c))
    matched = 0
    total = 0
    for blk in range(num_tokens // BLOCK):
        s = padded_len + blk * BLOCK
        seg = max(0, min(s + BLOCK, n) - s)
        if seg <= 0:
            continue
        matched += sum(1 for i in range(seg) if b[s + i] == c[s + i])
        total += seg
    return {
        "matched": matched,
        "compared": total,
        "rate": round(matched / total, 6) if total else None,
    }


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run_bench(model_path: Path, reps: int, quick: bool) -> dict[str, Any]:
    """Run the three-arm adaptive-budget + uncertainty-order benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("run_bench requires a CUDA device")

    num_tokens = QUICK_NUM_TOKENS if quick else FULL_NUM_TOKENS
    model, tokenizer, model_meta = _load_model(model_path)
    prompts = _prepare_prompts(tokenizer, quick)
    llm = _BenchDiffusionLLM(model, tokenizer)
    num_layers = llm.num_layers
    policy_off = _RecordingBudgetPolicy(
        budget_ratio=BUDGET_RATIO,
        num_layers=num_layers,
        warmup_steps=BUDGET_WARMUP,
        window_steps=BUDGET_WINDOW,
        adaptive=False,
    )
    policy_full = _RecordingBudgetPolicy(
        budget_ratio=BUDGET_RATIO,
        num_layers=num_layers,
        warmup_steps=BUDGET_WARMUP,
        window_steps=BUDGET_WINDOW,
        adaptive=True,
    )
    mff_off = _make_eager_mff(model, policy_off)
    mff_full = _make_eager_mff(model, policy_full)
    sampler_full = _EquivalenceProbeSampler(llm, config=_arm_sampler_config(num_tokens, full=True))
    modes: dict[str, Any] = {
        _BASELINE: {
            "sampler": FastDLLMSampler(llm, config=_sampler_config(num_tokens, 0)),
            "mff": None,
        },
        _OFF: {
            "sampler": FastDLLMSampler(llm, config=_arm_sampler_config(num_tokens, full=False)),
            "mff": mff_off,
        },
        _FULL: {"sampler": sampler_full, "mff": mff_full},
    }

    def _generate(arm: str, prompt: dict[str, Any]) -> dict[str, Any]:
        from typing import cast

        from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

        entry = modes[arm]
        mff = entry["mff"]
        timer = llm.block_timer if mff is None else mff.block_timer
        timer.reset()
        if mff is not None:
            mff.window_calls = []
            mff.recapture_windows = []
            mff.branches_seen = []
        GLOBAL_STABILITY_PROFILER.reset()
        with torch.no_grad():
            out = entry["sampler"].sample(
                prompt["ids"],
                folded_model=cast(Any, mff),
            )
        timer.finalize()
        record: dict[str, Any] = {
            "blocks": list(timer.blocks),
            "sequences": out.sequences.detach().to("cpu"),
        }
        if mff is not None:
            record["window_calls"] = list(mff.window_calls)
            ratios: list[float] = []
            for bid in mff.branches_seen:
                profile = GLOBAL_STABILITY_PROFILER.get_profile(bid)
                if profile:
                    ratios.extend(s.stable_ratio for s in profile.layer_stats)
            record["stable_ratios"] = ratios
        GLOBAL_STABILITY_PROFILER.reset()
        return record

    # -- warmup: one full discarded sample per (arm, prompt) --------------
    for arm in ARMS:
        for prompt in prompts:
            _generate(arm, prompt)
        print(f"warmup done: {arm}")

    # -- sidecar pass A (untimed quality faces) ----------------------------
    sampler_full.probe_rescore = True
    policy_off.probe_enabled = True
    policy_full.probe_enabled = True
    per_prompt_rows: list[dict[str, Any]] = []
    sidecar_detail: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    budget_rows: dict[str, list[dict[str, int]]] = {_OFF: [], _FULL: []}
    ledger_rows: dict[str, dict[str, Any]] = {}
    uncertainty_rows: list[dict[str, Any]] = []
    pass_a_seqs: dict[tuple[str, int], torch.Tensor] = {}
    for pi, prompt in enumerate(prompts):
        padded_len = prompt["padded_len"]
        base_posthoc = None
        seqs: dict[str, torch.Tensor] = {}
        posthocs: dict[str, Any] = {}
        arm_ledgers: dict[str, Any] = {}
        stats_before = dict(sampler_full.uncertainty_stats)
        units_before = len(sampler_full.probe_units)
        for arm in ARMS:
            mff = modes[arm]["mff"]
            if mff is not None:
                ops0 = mff.cache.write_op_count
                s0, b0 = mff.serial_calls, mff.batched_calls
            rec = _generate(arm, prompt)
            seqs[arm] = rec["sequences"]
            posthocs[arm] = _posthoc_confidence(model, seqs[arm][0], padded_len, num_tokens)
            if arm == _BASELINE:
                base_posthoc = posthocs[arm]
            if mff is not None:
                arm_ledgers[arm] = _ledger_delta(mff, ops0, s0, b0, num_layers)
        stats_after = dict(sampler_full.uncertainty_stats)
        units_after = len(sampler_full.probe_units)
        for arm in ARMS:
            pass_a_seqs[(arm, pi)] = seqs[arm]
        uncertainty_rows.append(
            {
                "prompt_index": pi,
                "validated_rows": stats_after["validated_rows"] - stats_before["validated_rows"],
                "skipped_rows": stats_after["skipped_rows"] - stats_before["skipped_rows"],
                "early_stops": stats_after["early_stops"] - stats_before["early_stops"],
            }
        )
        for unit in sampler_full.probe_units[units_before:units_after]:
            equivalence_rows.append({"prompt_index": pi, **unit})
        for arm, policy in ((_OFF, policy_off), (_FULL, policy_full)):
            synced = _sync_budget_records(policy.records)
            policy.records.clear()
            budget_rows[arm].extend(synced)
        ledger_rows[f"prompt{pi}"] = arm_ledgers
        per_prompt_rows.append(
            {
                "prompt_index": pi,
                "posthoc_overall_mean": {
                    _BASELINE: (base_posthoc or {}).get("overall_mean"),
                    _OFF: posthocs[_OFF].get("overall_mean"),
                    _FULL: posthocs[_FULL].get("overall_mean"),
                },
                "decode_length": {
                    _BASELINE: int(seqs[_BASELINE].shape[1]),
                    _OFF: int(seqs[_OFF].shape[1]),
                    _FULL: int(seqs[_FULL].shape[1]),
                },
                "block_match_counts": {
                    _OFF: _merged_block_exact_match(
                        seqs[_BASELINE], seqs[_OFF], padded_len, num_tokens
                    ),
                    _FULL: _merged_block_exact_match(
                        seqs[_BASELINE], seqs[_FULL], padded_len, num_tokens
                    ),
                },
            }
        )
        sidecar_detail.append(
            {
                "prompt_index": pi,
                "posthoc_confidence": posthocs,
                "block_exact_match": {
                    _OFF: _block_exact_match(seqs[_BASELINE], seqs[_OFF], padded_len, num_tokens),
                    _FULL: _block_exact_match(seqs[_BASELINE], seqs[_FULL], padded_len, num_tokens),
                },
            }
        )
        print(
            f"sidecar prompt {pi}: len(base/off/full) = "
            f"{seqs[_BASELINE].shape[1]}/{seqs[_OFF].shape[1]}/{seqs[_FULL].shape[1]}, "
            f"uncertainty early_stops = "
            f"{uncertainty_rows[-1]['early_stops']}"
        )

    # -- sidecar pass B (the internal reproduction self-check) ------------
    repro_seqs: dict[tuple[str, int], torch.Tensor] = {}
    for pi, prompt in enumerate(prompts):
        for arm in ARMS:
            rec = _generate(arm, prompt)
            repro_seqs[(arm, pi)] = rec["sequences"]
    repro_mismatches: list[dict[str, Any]] = repro_mismatch_rows(
        pass_a_seqs, repro_seqs, list(ARMS), len(prompts)
    )
    repro_ok = not repro_mismatches
    sampler_full.probe_rescore = False
    policy_off.probe_enabled = False
    policy_full.probe_enabled = False

    # -- timed reps: per-rep round-robin over the three arms ----------------
    whole_samples: dict[str, dict[int, list[float]]] = {
        arm: {pi: [] for pi in range(len(prompts))} for arm in ARMS
    }
    profiler_ratios: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for rep in range(reps):
        order = list(ARMS[rep % 3 :] + ARMS[: rep % 3])
        for arm in order:
            for prompt in prompts:
                pi = prompts.index(prompt)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                rec = _generate(arm, prompt)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                whole_samples[arm][pi].append((t1 - t0) * 1000.0)
                if modes[arm]["mff"] is not None:
                    profiler_ratios[arm].extend(rec.get("stable_ratios", []))
        summary = " | ".join(f"{arm}: last={whole_samples[arm][0][-1]:.1f}ms" for arm in order)
        print(f"rep {rep + 1}/{reps}: {summary}")

    # -- statistics ----------------------------------------------------------
    def _pooled(arm: str) -> list[float]:
        return [x for pi in range(len(prompts)) for x in whole_samples[arm][pi]]

    whole_stats = {arm: stats_from_samples(_pooled(arm)) for arm in ARMS}
    per_prompt_stats = {
        arm: {
            f"prompt{pi}": stats_from_samples(whole_samples[arm][pi]).as_dict()
            for pi in range(len(prompts))
        }
        for arm in ARMS
    }
    p50 = {arm: whole_stats[arm].p50 for arm in ARMS}
    std_h2 = (whole_stats[_BASELINE].std ** 2 + whole_stats[_FULL].std ** 2) ** 0.5
    std_h3 = (whole_stats[_OFF].std ** 2 + whole_stats[_FULL].std ** 2) ** 0.5
    equiv_units = len(equivalence_rows)
    equiv_matches = sum(1 for u in equivalence_rows if u["match"])
    equiv_rate = (equiv_matches / equiv_units) if equiv_units else None
    budget_violations = sum(1 for arm in (_OFF, _FULL) for r in budget_rows[arm] if r["violation"])
    ledger_ok = bool(ledger_rows) and all(
        entry.get(arm, {}).get("within_bound")
        for entry in ledger_rows.values()
        for arm in (_OFF, _FULL)
    )

    faces = {
        "quality_faces": {"per_prompt": per_prompt_rows},
        "p50_base_ms": p50[_BASELINE],
        "p50_off_ms": p50[_OFF],
        "p50_full_ms": p50[_FULL],
        "std_merged_h2_ms": std_h2,
        "std_merged_h3_ms": std_h3,
        "equiv_rate": equiv_rate,
        "budget_violations": budget_violations,
        "ledger_ok": ledger_ok,
        "repro_ok": repro_ok,
    }
    verdicts = assemble_verdicts(faces)

    stable_summary: dict[str, Any] = {}
    for arm in ARMS:
        ratios = profiler_ratios[arm]
        if ratios:
            mean = sum(ratios) / len(ratios)
            stable_summary[arm] = {
                "mean": round(mean, 4),
                "std": round((sum((r - mean) ** 2 for r in ratios) / len(ratios)) ** 0.5, 4),
                "min": round(min(ratios), 4),
                "max": round(max(ratios), 4),
                "n": len(ratios),
            }
        else:
            stable_summary[arm] = None

    import transformers

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "quick": quick,
            "reps": reps,
            "seed": SEED,
            "prompts": {
                "count": len(prompts),
                "num_tokens": num_tokens,
                "shape_windows_W": num_tokens // BLOCK,
                "texts": [p["text"] for p in prompts],
                "raw_token_lens": [p["raw_len"] for p in prompts],
                "padded_token_lens": [p["padded_len"] for p in prompts],
            },
            "arm_configs": {
                _BASELINE: {
                    "folded_model": None,
                    "num_block_candidates": 0,
                    "description": "official Fast-dLLM v2 recipe (raw sampler)",
                },
                _OFF: {
                    "mff": "eager ManualFoldedForward (use_cuda_graph=False)",
                    "budget_policy": {
                        "budget_ratio": BUDGET_RATIO,
                        "warmup_steps": BUDGET_WARMUP,
                        "window_steps": BUDGET_WINDOW,
                        "adaptive": False,
                    },
                    "sampler": {
                        "num_block_candidates": NUM_CANDIDATES,
                        "candidate_top_k": CANDIDATE_TOP_K,
                        "candidate_include_greedy": True,
                        "uncertainty_validation_order": False,
                    },
                    "description": (
                        "AR022-aligned config + uniform-weight budget cap "
                        "(the cap ACTIVE, window re-estimation OFF) + full "
                        "validation -- the attribution control"
                    ),
                },
                _FULL: {
                    "mff": "eager ManualFoldedForward (use_cuda_graph=False)",
                    "budget_policy": {
                        "budget_ratio": BUDGET_RATIO,
                        "warmup_steps": BUDGET_WARMUP,
                        "window_steps": BUDGET_WINDOW,
                        "adaptive": True,
                    },
                    "sampler": {
                        "num_block_candidates": NUM_CANDIDATES,
                        "candidate_top_k": CANDIDATE_TOP_K,
                        "candidate_include_greedy": True,
                        "uncertainty_validation_order": True,
                        "uncertainty_margin": UNCERTAINTY_MARGIN,
                        "uncertainty_margin_gap": UNCERTAINTY_MARGIN_GAP,
                    },
                    "description": (
                        "the adaptive-off construction + windowed "
                        "layer-adaptive weights + prior-descending "
                        "uncertainty validation with the frozen margin/"
                        "margin-gap early stop"
                    ),
                },
            },
            "warmup_strategy": (
                "one full discarded sample per (arm, prompt) before any "
                "timing; the untimed sidecar pass A records the quality "
                "faces, pass B is the internal reproduction self-check"
            ),
            "interleaving": (
                "per-rep round-robin over the three arms; rep r starts at "
                "position r%3 (anti-drift, WDDM fairness)"
            ),
            "timing_method": (
                "host perf_counter + torch.cuda.synchronize (#39); "
                "whole-generation wall clock; the sidecar probes "
                "(equivalence re-scoring, budget recording) are disabled "
                "during the timed segment -- zero probe overhead on p50"
            ),
            "reproduction_declaration": (
                "the untimed sidecar generation runs TWICE per (arm, "
                "prompt); pass A records the quality faces and pass B must "
                "be strictly sequence-equal on every row (deterministic "
                "single-shot premise); any mismatch blocks ALL verdicts "
                "(one-vote veto); the check binds quick and formal alike"
            ),
            "equivalence_probe_semantics": (
                "every speculative unit of the full arm (sidecar pass A) "
                "re-scores the full candidate set with the frozen "
                "off-branch vectorized min-prob formula over the same "
                "verification logits and compares argmax vs the "
                "early-stopped j_star; equiv_rate = matches/units; None "
                "when no speculative unit ran (honest no-data refutes)"
            ),
            "budget_probe_semantics": (
                "the recording policy captures every capped layer-forward "
                "same-shape eager call: violation := (tokens - post-cap "
                "stable) - forced - nan_scores > B_l; forced and NaN-score "
                "positions are the two constructive FR1 exemption classes; "
                "the dense ActivationCache never trips the availability "
                "guard; all quantities are recorded as device tensors and "
                "synced once per sidecar read (zero timed-path syncs)"
            ),
            "limitation_notes": list(ARTIFACT_LIMITATION_NOTES),
            "single_run_discipline": (
                "the full (reps=30) run is a pre-committed single-shot "
                "verdict, no rerun fishing (#39); this artifact is the "
                "--quick smoke run (separate path, never the formal "
                "artifact) and its H verdicts are non-final"
                if quick
                else "pre-committed single-shot verdict, no rerun fishing"
            ),
            "device": torch.cuda.get_device_name(0),
            "gpu_driver_version": _gpu_driver_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "model": "Fast-dLLM-v2-1.5B (Efficient-Large-Model, remote code)",
            "model_details": model_meta,
            "venv": sys.prefix != sys.base_prefix,
            "run_date_utc": datetime.now(timezone.utc).isoformat(),
            "command": (
                "python -m scripts.ar031_adaptive_uncertainty_bench"
                + (" --quick" if quick else f" --reps {reps}")
                + f" --model-path {model_path}"
            ),
        },
        "hypotheses_declared": _HYPOTHESES,
        "hypotheses_verdicts": verdicts,
        "arms": {
            arm: {
                "whole_gen_ms": whole_stats[arm].as_dict(),
                "per_prompt_ms": per_prompt_stats[arm],
                "stable_ratio_timed": stable_summary[arm],
            }
            for arm in ARMS
        },
        "speedups_p50": {
            "full_vs_baseline": (round(p50[_BASELINE] / p50[_FULL], 4) if p50[_FULL] > 0 else None),
            "full_vs_off": round(p50[_OFF] / p50[_FULL], 4) if p50[_FULL] > 0 else None,
            "off_vs_baseline": (round(p50[_BASELINE] / p50[_OFF], 4) if p50[_OFF] > 0 else None),
        },
        "quality": {
            "per_prompt": per_prompt_rows,
            "per_arm_sidecar": sidecar_detail,
            "uncertainty_stats_per_prompt": uncertainty_rows,
        },
        "equivalence": {
            "units": equiv_units,
            "matches": equiv_matches,
            "equiv_rate": round(equiv_rate, 6) if equiv_rate is not None else None,
            "per_unit": equivalence_rows,
        },
        "budget_probe": {arm: _budget_probe_summary(budget_rows[arm]) for arm in (_OFF, _FULL)},
        "budget_window_events": {
            _OFF: list(policy_off.window_events),
            _FULL: list(policy_full.window_events),
        },
        "budget_weights_tail": {
            _OFF: policy_off.weights[-4:] if policy_off.weights else None,
            _FULL: policy_full.weights[-4:] if policy_full.weights else None,
        },
        "reproduction": {
            "mode": "internal-double-run",
            "ok": repro_ok,
            "rows_compared": len(prompts) * len(ARMS),
            "mismatches": repro_mismatches,
        },
        "ledger": {
            "c": LEDGER_C,
            "num_layers": num_layers,
            "per_prompt": ledger_rows,
            "within_bound": {
                arm: bool(
                    all(
                        ledger_rows[f"prompt{pi}"].get(arm, {}).get("within_bound")
                        for pi in range(len(prompts))
                    )
                )
                for arm in (_OFF, _FULL)
            },
        },
    }
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path (default: the formal artifact for full runs, "
            f"the separate quick path {QUICK_OUTPUT_PATH} for --quick)"
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local checkpoint directory (env fallback: FASTDLLM_MODEL_PATH).",
    )
    parser.add_argument("--reps", type=int, default=REPS_DEFAULT, help="Timed reps")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke subset: reps=3, single prompt, num_tokens=64 (2 blocks)",
    )
    add_hf_env_arguments(parser)
    args = parser.parse_args(argv)

    quick = bool(args.quick)
    reps = QUICK_REPS if quick else int(args.reps)
    import transformers

    try:
        model_path = _resolve_model_path(args.model_path)
    except OSError as error:
        print(f"environment gate failed: {error}", file=sys.stderr)
        return 2
    try:
        require_environment(transformers.__version__, model_path)
    except RuntimeError as error:
        print(f"environment gate failed: {error}", file=sys.stderr)
        return 2
    data = run_bench(model_path, reps=reps, quick=quick)
    out = write_results(data, resolve_output_path(args.out, quick))
    print()
    for key in ("h1", "h2", "h3", "h4"):
        entry = data["hypotheses_verdicts"][key]
        print(f"{key.upper()}: {entry['verdict']}")
    print(f"overall: {data['hypotheses_verdicts']['overall']}")
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
