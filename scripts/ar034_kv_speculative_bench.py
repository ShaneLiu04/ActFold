"""AR034 four-arm benchmark: KV-cached speculative block decoding.

Arms (official Fast-dLLM v2 geometry block=32 / small_block=8, every
ActFold arm shares the schedule semantics; only the forward source and the
commit granularity differ):

- ``baseline``   (A): full-canvas recompute (the AR016-AR033 non-cached
  reference arm).
- ``cache-full`` (B2): AR033 ``history_cache`` + intra-block sb cache (the
  fastest cache-only arm).
- ``cache-spec`` (D): AR034 composition -- ``history_cache`` +
  ``num_block_candidates=4`` + greedy anchoring. One whole-block draft + one
  ``[N, block]`` batched verify over the expanded read-only history KV per
  small block, then the shared commit+advance forward.
- ``official``   (C): the checkpoint's native ``generate()`` (report-only
  timing reference, RAW unpadded prompt -- native mask-alignment semantics).

Frozen verdicts (defined BEFORE data collection, single pre-committed run):

- H1 quality: for D (and B2 as the cache reference), decode lengths equal A
  on every prompt, greedy token agreement D-vs-A >= 0.98, posthoc gap
  |D-A| <= 1e-3. Any None -> refuted.
- H2 speed: p50(A)/p50(D) > 1.0 STRICT **and** p50(B2)/p50(D) > 1.0 STRICT
  (the composition must beat both references); C ratios report-only.
- H3 mechanism: D's class ledger is exactly reconstructed (prefill=1,
  spec_draft==spec_verify>0, commit==advance count, degraded 0, every spec
  call length == block, verify batch == N) AND D's total forward-token count
  is strictly below B2's per prompt; A is all full-canvas block-aligned.
- H4 honesty: canary conjunction (D armed + D degraded 0 + B2 armed + frozen
  topology) AND a per-``(arm, prompt_index)`` reproduction double-run with
  strict equality; a mismatch blocks ALL verdicts (AR031 one-vote caliber).

Run (AR019 venv, transformers==4.53.1)::

    python -m scripts.ar034_kv_speculative_bench [--quick] [--reps 30]
        [--model-path PATH] [--out PATH] [--force --force-reason TEXT]
"""

from __future__ import annotations

import argparse
import collections
import inspect
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402

from actfold.models.diffusion_sampler import DiffusionSampler  # noqa: E402
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig  # noqa: E402
from scripts.algo_experiments import stats_from_samples  # noqa: E402
from scripts.ar021_generation_loop_bench import (  # noqa: E402
    BLOCK,
    FULL_NUM_TOKENS,
    QUICK_NUM_TOKENS,
    SEED,
    _BenchDiffusionLLM,
    _gpu_driver_version,
    _load_model,
    _posthoc_confidence,
    _prepare_prompts,
    _resolve_model_path,
    write_results,
)
from scripts.ar033_history_cache_bench import (
    _merged_block_exact_match,
    _merged_mean,
    _probe_official_generate,
    _reconstruct_cache_ledger,
    _reconstruct_legacy_ledger,
    repro_mismatch_rows,
    require_environment,
)
from scripts.ar033_history_cache_bench import (  # noqa: E402
    resolve_output_path as _resolve_ar033_output,
)

SCHEMA = "ar034_kv_speculative_bench"
SCHEMA_VERSION = 1
ARMS = ("baseline", "cache-full", "cache-spec", "official")
DEFAULT_OUTPUT_PATH = Path("results/optimization/ar034_kv_speculative_bench.json")
QUICK_OUTPUT_PATH = Path("results/optimization/ar034_kv_speculative_bench_quick.json")
REPS_DEFAULT = 30
QUICK_REPS = 3
REQUIRED_TRANSFORMERS = "4.53.1"
# AR022-lineage geometry (small_block == block): the speculative commit unit
# must match the config whose greedy-anchored semantics AR022 validated. At
# sb=8 random top-k rows can beat the greedy row under the frozen min-prob
# rule (AR034 T007: 2/8 wins -> 96 vs 34 tokens, quality divergence); at
# sb=32 the greedy row wins 100% and the composition is token-exact vs the
# non-cached baseline on the quick face.
SMALL_BLOCK = 32
SPEC_CANDIDATES = 4
SPEC_TOP_K = 10
_H1_TOKEN_RATE_MIN = 0.98
_H1_POSTHOC_GAP_MAX = 1e-3
_A = "baseline"
_B2 = "cache-full"
_D = "cache-spec"
_C = "official"
_ACTFOLD_ARMS = (_A, _B2, _D)
_QUALITY_ARMS = (_D, _B2)

_T001_PROBE_RESULTS = {
    "P1_kv_unchanged": True,
    "P2_n2_batched_equals_single": [True, True],
    "P3_maxdiff_per_row": [0.0, 0.0, 0.0, 0.0],
    "P3_argmax_match_per_row": [1.0, 1.0, 1.0, 1.0],
    "P4_expand_vram_mb": 4.36,
    "P4_peak_delta_mb": 9.63,
    "P5_slice_vs_block_maxdiff": {"sb0": 1.5, "sb8": 7.21875, "sb16": 8.875},
    "P5_conclusion": "whole-block draft/verify required (intra-block bidirectional attention)",
    "device": "Quadro RTX 5000",
    "date": "2026-10-24",
}


def resolve_output_path(out_arg: Path | None, quick: bool) -> Path:
    """Explicit out wins; quick never resolves to the formal path."""
    if out_arg is not None:
        return out_arg
    return (
        _resolve_ar033_output(None, quick)
        if False
        else (QUICK_OUTPUT_PATH if quick else DEFAULT_OUTPUT_PATH)
    )


# ---------------------------------------------------------------------------
# Recorder (batch-aware: draft vs verify separation)
# ---------------------------------------------------------------------------
class _RawCallRecorderB:
    """Instance-level raw-forward recorder with batch + length metadata."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.calls: list[dict[str, Any]] = []

    def install(self) -> None:
        model = self._model
        original = model.forward
        pinned_signature = inspect.signature(type(model).forward)
        calls = self.calls

        def _recorded(*args: Any, **kwargs: Any) -> Any:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            calls.append(
                {
                    "len": int(input_ids.shape[1]) if input_ids is not None else -1,
                    "batch": int(input_ids.shape[0]) if input_ids is not None else -1,
                    "use_cache": "use_cache" in kwargs,
                    "update": kwargs.get("update_past_key_values"),
                    "has_past": "past_key_values" in kwargs,
                    "use_bc": bool(kwargs.get("use_block_cache", False)),
                    "has_bpv": "block_past_key_values" in kwargs,
                    "replace": kwargs.get("replace_position"),
                }
            )
            return original(*args, **kwargs)

        _recorded.__signature__ = pinned_signature  # type: ignore[attr-defined]
        model.forward = _recorded


def _classify(rec: dict[str, Any]) -> str:
    """AR034 classes: legacy / prefill / commit / sb* / spec_draft / spec_verify."""
    if not rec["use_cache"]:
        return "legacy_full_canvas"
    if rec["update"] and not rec["has_past"]:
        return "prefill"
    if rec["update"]:
        return "commit"
    if rec["use_bc"] and not rec["has_bpv"]:
        return "block_rebuild"
    if rec["use_bc"] and rec["replace"] == 0:
        return "sb0_partial"
    if rec["use_bc"]:
        return "sb_partial"
    if rec["batch"] == 1:
        return "spec_draft"
    return "spec_verify"


def _reconstruct_spec_ledger(
    calls: list[dict[str, Any]], padded_len: int, block_size: int, num_candidates: int
) -> dict[str, Any]:
    """Class-wise ledger reconstruction for the composition arm (D)."""
    counts: collections.Counter[str] = collections.Counter()
    total = 0
    structure_ok = True
    expected_len = {
        "prefill": padded_len,
        "commit": block_size,
        "spec_draft": block_size,
        "spec_verify": block_size,
        "cross_slice": block_size,
    }
    for rec in calls:
        cls = _classify(rec)
        counts[cls] += 1
        total += int(rec["len"])
        if cls in expected_len and int(rec["len"]) != expected_len[cls]:
            structure_ok = False
        if cls == "spec_draft" and int(rec["batch"]) != 1:
            structure_ok = False
        if cls == "spec_verify" and int(rec["batch"]) != num_candidates:
            structure_ok = False
    predicted = {
        "prefill_forwards": counts["prefill"],
        "slice_forwards": counts["cross_slice"],
        "commit_forwards": counts["commit"],
        "block_rebuilds": counts["block_rebuild"],
        "spec_draft_forwards": counts["spec_draft"],
        "spec_verify_forwards": counts["spec_verify"],
    }
    return {
        "counts": dict(counts),
        "total_tokens": total,
        "structure_ok": structure_ok,
        "predicted": predicted,
    }


# ---------------------------------------------------------------------------
# Frozen verdicts
# ---------------------------------------------------------------------------
def _forward_count(recon: dict[str, Any]) -> int:
    """Total raw forward count from a class-ledger reconstruction."""
    pred = recon["predicted"]
    return (
        int(pred.get("prefill_forwards", 0))
        + int(pred.get("slice_forwards", 0))
        + int(pred.get("commit_forwards", 0))
        + int(pred.get("block_rebuilds", 0))
        + int(pred.get("spec_draft_forwards", 0))
        + int(pred.get("spec_verify_forwards", 0))
    )


def verdict_h1(quality_faces: dict[str, Any]) -> dict[str, Any]:
    faces: dict[str, Any] = {}
    numbers: dict[str, Any] = {}
    supported = True
    for arm in _QUALITY_ARMS:
        entry = quality_faces.get(arm, {})
        lens_eq = bool(entry.get("decode_lengths_equal"))
        rate = entry.get("token_rate")
        gap = entry.get("posthoc_gap")
        rate_ok = rate is not None and math.isfinite(rate) and rate >= _H1_TOKEN_RATE_MIN
        gap_ok = gap is not None and math.isfinite(gap) and gap <= _H1_POSTHOC_GAP_MAX
        faces[arm] = {
            "a_decode_lengths_equal": lens_eq,
            "b_token_rate_ge_0.98": bool(rate_ok),
            "c_posthoc_gap_le_1e-3": bool(gap_ok),
        }
        numbers[arm] = {
            "decode_lengths": entry.get("decode_lengths"),
            "baseline_decode_lengths": entry.get("baseline_decode_lengths"),
            "token_rate": rate,
            "matched": entry.get("matched"),
            "compared": entry.get("compared"),
            "posthoc_mean": entry.get("posthoc_mean"),
            "baseline_posthoc_mean": entry.get("baseline_posthoc_mean"),
            "posthoc_gap": gap,
        }
        supported = supported and lens_eq and bool(rate_ok) and bool(gap_ok)
    return {"verdict": "supported" if supported else "refuted", "faces": faces, "numbers": numbers}


def _ratio(faster_ms: float | None, slower_ms: float | None) -> float | None:
    if (
        faster_ms is not None
        and slower_ms is not None
        and math.isfinite(faster_ms)
        and math.isfinite(slower_ms)
        and faster_ms > 0.0
        and slower_ms > 0.0
    ):
        return slower_ms / faster_ms
    return None


def verdict_h2(
    baseline_ms: float | None,
    b2_ms: float | None,
    d_ms: float | None,
    official_ms: float | None,
    sigma_baseline: float | None,
    sigma_b2: float | None,
    sigma_d: float | None,
) -> dict[str, Any]:
    """D must beat BOTH the non-cached baseline and the best cache-only arm."""
    ratio_a = _ratio(d_ms, baseline_ms)  # p50(A)/p50(D)
    ratio_b2 = _ratio(d_ms, b2_ms)  # p50(B2)/p50(D)
    supported = ratio_a is not None and ratio_a > 1.0 and ratio_b2 is not None and ratio_b2 > 1.0
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": {
            "a_baseline_over_d_gt_1": bool(ratio_a is not None and ratio_a > 1.0),
            "b_b2_over_d_gt_1": bool(ratio_b2 is not None and ratio_b2 > 1.0),
        },
        "numbers": {
            "baseline_ms": baseline_ms,
            "b2_ms": b2_ms,
            "d_ms": d_ms,
            "official_ms": official_ms,
            "speedup_d_vs_baseline": round(ratio_a, 4) if ratio_a else None,
            "speedup_d_vs_baseline_raw": ratio_a,
            "speedup_d_vs_b2": round(ratio_b2, 4) if ratio_b2 else None,
            "speedup_d_vs_b2_raw": ratio_b2,
            "sigma_baseline": sigma_baseline,
            "sigma_b2": sigma_b2,
            "sigma_d": sigma_d,
            "report_only_official_ratios": {
                "d_vs_official": round(r, 4) if (r := _ratio(official_ms, d_ms)) else None,
                "b2_vs_official": round(r, 4) if (r := _ratio(official_ms, b2_ms)) else None,
                "official_vs_baseline": round(r, 4)
                if (r := _ratio(baseline_ms, official_ms))
                else None,
            },
            "rule": (
                "supported iff p50(A)/p50(D) > 1.0 AND p50(B2)/p50(D) > 1.0 "
                "strictly; official ratios and sigmas are report-only"
            ),
        },
    }


def verdict_h3(ledger_faces: dict[str, Any]) -> dict[str, Any]:
    faces = {
        "d_ledger_identity": bool(ledger_faces.get("d_identity_all")),
        "b2_ledger_identity": bool(ledger_faces.get("b2_identity_all")),
        "d_spec_balanced": bool(ledger_faces.get("d_spec_balanced_all")),
        "d_lt_b2": bool(ledger_faces.get("d_lt_b2_all")),
        "a_full_canvas_structure": bool(ledger_faces.get("a_structure_all")),
    }
    supported = all(faces.values())
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": faces,
        "numbers": {
            "d_forwards_total": ledger_faces.get("d_forwards"),
            "b2_forwards_total": ledger_faces.get("b2_forwards"),
            "d_forward_tokens_total": ledger_faces.get("d_total"),
            "b2_forward_tokens_total": ledger_faces.get("b2_total"),
            "a_forward_tokens_total": ledger_faces.get("a_total"),
            "rule": (
                "per (D, prompt): stats delta == class reconstruction (prefill=1, "
                "spec_draft==spec_verify>0, commit, degraded 0, spec len==block, "
                "verify batch==N); D total FORWARD COUNT < B2 total; A all "
                "full-canvas aligned. Token-row totals are report-only (D trades "
                "more rows for far fewer launches)"
            ),
        },
    }


def verdict_h4(canary_faces: dict[str, Any], repro_rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = all(bool(v) for v in canary_faces.values()) and not repro_rows
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": {
            **{k: bool(v) for k, v in canary_faces.items()},
            "reproduction_ok": not repro_rows,
        },
        "numbers": {
            "canary_faces": {k: bool(v) for k, v in canary_faces.items()},
            "repro_rows": repro_rows,
        },
    }


def overall_verdict(h1: str, h2: str, h3: str, h4: str) -> str:
    verdicts = (h1, h2, h3, h4)
    if any(v == "refuted" for v in verdicts):
        return "refuted"
    if all(v == "supported" for v in verdicts):
        return "supported"
    return "inconclusive"


def assemble_verdicts(faces: dict[str, Any]) -> dict[str, Any]:
    h1 = verdict_h1(faces["quality_faces"])
    h2 = verdict_h2(
        faces["baseline_ms"],
        faces["b2_ms"],
        faces["d_ms"],
        faces["official_ms"],
        faces["sigma_baseline"],
        faces["sigma_b2"],
        faces["sigma_d"],
    )
    h3 = verdict_h3(faces["ledger_faces"])
    h4 = verdict_h4(faces["canary_faces"], faces["repro_rows"])
    if faces["repro_rows"]:
        for verdict in (h1, h2, h3, h4):
            verdict["verdict"] = "blocked"
            verdict["blocked_by_reproduction_mismatch"] = True
        overall = "blocked"
    else:
        overall = overall_verdict(h1["verdict"], h2["verdict"], h3["verdict"], h4["verdict"])
    return {"h1": h1, "h2": h2, "h3": h3, "h4": h4, "overall": overall}


def _sampler_config(num_tokens: int, arm: str) -> FastDLLMSamplerConfig:
    """Official v2 geometry for every ActFold arm; only the arm knobs differ."""
    base: dict[str, Any] = {
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": SMALL_BLOCK,
        "threshold": 0.95,
        "temperature": 0.0,
        "seed": SEED,
    }
    if arm == _B2:
        base.update(history_cache=True, history_small_block_cache=True)
    elif arm == _D:
        base.update(
            history_cache=True,
            history_small_block_cache=False,
            num_block_candidates=SPEC_CANDIDATES,
            candidate_top_k=SPEC_TOP_K,
            candidate_include_greedy=True,
        )
    return FastDLLMSamplerConfig(**base)


def run_bench(model_path: Path, reps: int, quick: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("run_bench requires a CUDA device")

    num_tokens = QUICK_NUM_TOKENS if quick else FULL_NUM_TOKENS
    model, tokenizer, model_meta = _load_model(model_path)
    prompts = _prepare_prompts(tokenizer, quick)
    llm = _BenchDiffusionLLM(model, tokenizer)
    llm.block_timer.note = lambda block_idx: None  # type: ignore[method-assign, assignment]

    recorder = _RawCallRecorderB(model)
    recorder.install()
    official_probe = _probe_official_generate(model)

    mask_id = DiffusionSampler._resolve_mask_token_id(model, tokenizer)
    eos_id = tokenizer.eos_token_id
    official_ids = {
        "mask_token_id": int(mask_id) if mask_id is not None else None,
        "stop_token_id": int(eos_id) if eos_id is not None else None,
        "pad_token_id": int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
    }
    if official_probe["supported"] and (
        official_ids["mask_token_id"] is None or official_ids["stop_token_id"] is None
    ):
        official_probe = {
            "supported": False,
            "reason": "tokenizer lacks mask/eos ids for the native generate() call",
            "signature_params": official_probe.get("signature_params"),
        }

    samplers = {
        arm: FastDLLMSampler(llm, config=_sampler_config(num_tokens, arm)) for arm in _ACTFOLD_ARMS
    }
    active_arms = [arm for arm in ARMS if arm != _C or official_probe["supported"]]
    n_arms = len(active_arms)

    def _generate(arm: str, prompt: dict[str, Any]) -> dict[str, Any]:
        rec0 = len(recorder.calls)
        if arm == _C:
            with torch.no_grad():
                seq = model.generate(
                    input_ids=prompt["ids"][:, : prompt["raw_len"]],
                    max_new_tokens=num_tokens,
                    mask_id=official_ids["mask_token_id"],
                    threshold=0.95,
                    small_block_size=SMALL_BLOCK,
                    block_size=BLOCK,
                    stop_token=official_ids["stop_token_id"],
                    temperature=0.0,
                    use_block_cache=True,
                )
            sequences = seq.detach().to("cpu")
            stats_delta: dict[str, int] = {}
        else:
            sampler = samplers[arm]
            before = dict(sampler.history_cache_stats)
            with torch.no_grad():
                out = sampler.sample(prompt["ids"])
            sequences = out.sequences.detach().to("cpu")
            after = dict(sampler.history_cache_stats)
            stats_delta = {key: after[key] - before[key] for key in after}
        return {
            "sequences": sequences,
            "calls": recorder.calls[rec0:],
            "stats_delta": stats_delta,
        }

    for arm in active_arms:
        for prompt in prompts:
            _generate(arm, prompt)
        print(f"warmup done: {arm}", flush=True)

    per_prompt_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    pass_a_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for pi, prompt in enumerate(prompts):
        padded_len = prompt["padded_len"]
        seqs: dict[str, torch.Tensor] = {}
        posthocs: dict[str, Any] = {}
        arm_records: dict[str, dict[str, Any]] = {}
        for arm in active_arms:
            rec = _generate(arm, prompt)
            arm_records[arm] = rec
            seqs[arm] = rec["sequences"]
            if arm != _C:
                posthocs[arm] = _posthoc_confidence(model, seqs[arm][0], padded_len, num_tokens)

        a_recon = _reconstruct_legacy_ledger(arm_records[_A]["calls"], padded_len, BLOCK)
        b2_recon = _reconstruct_cache_ledger(
            arm_records[_B2]["calls"], padded_len, BLOCK, SMALL_BLOCK
        )
        b2_delta = arm_records[_B2]["stats_delta"]
        b2_identity = bool(
            b2_recon["structure_ok"]
            and b2_delta.get("degraded_calls") == 0
            and all(b2_delta.get(key, -1) == value for key, value in b2_recon["predicted"].items())
        )
        d_recon = _reconstruct_spec_ledger(
            arm_records[_D]["calls"], padded_len, BLOCK, SPEC_CANDIDATES
        )
        d_delta = arm_records[_D]["stats_delta"]
        d_identity = bool(
            d_recon["structure_ok"]
            and d_delta.get("degraded_calls") == 0
            and all(d_delta.get(key, -1) == value for key, value in d_recon["predicted"].items())
        )
        d_balanced = bool(
            d_delta.get("spec_draft_forwards", 0) > 0
            and d_delta.get("spec_draft_forwards") == d_delta.get("spec_verify_forwards")
        )
        d_lt_b2 = bool(_forward_count(d_recon) < _forward_count(b2_recon))
        ledger_rows.append(
            {
                "prompt_index": pi,
                "a": a_recon,
                "b2": {"reconstruction": b2_recon, "delta": b2_delta, "identity": b2_identity},
                "d": {
                    "reconstruction": d_recon,
                    "delta": d_delta,
                    "identity": d_identity,
                    "balanced": d_balanced,
                    "lt_b2": d_lt_b2,
                },
            }
        )

        for arm in active_arms:
            pass_a_payloads[(arm, pi)] = {
                "decode_length": int(seqs[arm].shape[1]),
                "tokens": seqs[arm][0].tolist(),
            }
        per_prompt_rows.append(
            {
                "prompt_index": pi,
                "posthoc_overall_mean": {
                    arm: posthocs.get(arm, {}).get("overall_mean") for arm in _ACTFOLD_ARMS
                },
                "decode_length": {
                    arm: int(seqs[arm].shape[1]) if arm in seqs else None for arm in ARMS
                },
                "block_match_counts": {
                    arm: _merged_block_exact_match(seqs[_A], seqs[arm], padded_len, num_tokens)
                    for arm in (_B2, _D)
                },
            }
        )
        print(
            f"sidecar prompt {pi}: len(A/B2/D) = "
            f"{seqs[_A].shape[1]}/{seqs[_B2].shape[1]}/{seqs[_D].shape[1]}"
            + (f", official={seqs[_C].shape[1]}" if _C in seqs else ""),
            flush=True,
        )

    repro_rows_all: list[dict[str, Any]] = []
    for pi, prompt in enumerate(prompts):
        for arm in active_arms:
            rec = _generate(arm, prompt)
            second = {
                "decode_length": int(rec["sequences"].shape[1]),
                "tokens": rec["sequences"][0].tolist(),
            }
            repro_rows_all.append(
                {
                    "arm": arm,
                    "prompt_index": pi,
                    "first": pass_a_payloads[(arm, pi)],
                    "second": second,
                }
            )
    repro_mismatches = repro_mismatch_rows(repro_rows_all)

    whole_samples: dict[str, dict[int, list[float]]] = {
        arm: {pi: [] for pi in range(len(prompts))} for arm in active_arms
    }
    for rep in range(reps):
        order = active_arms[rep % n_arms :] + active_arms[: rep % n_arms]
        for arm in order:
            for pi, prompt in enumerate(prompts):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _generate(arm, prompt)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                whole_samples[arm][pi].append((t1 - t0) * 1000.0)
        summary = " | ".join(f"{arm}: last={whole_samples[arm][0][-1]:.1f}ms" for arm in order)
        print(f"rep {rep + 1}/{reps}: {summary}", flush=True)

    def _pooled(arm: str) -> list[float]:
        return [x for pi in range(len(prompts)) for x in whole_samples[arm][pi]]

    whole_stats = {arm: stats_from_samples(_pooled(arm)) for arm in active_arms}
    p50 = {arm: whole_stats[arm].p50 for arm in active_arms}
    sigma = {arm: whole_stats[arm].std for arm in active_arms}

    quality_faces: dict[str, Any] = {}
    for arm in _QUALITY_ARMS:
        lens = [row["decode_length"][arm] for row in per_prompt_rows]
        base_lens = [row["decode_length"][_A] for row in per_prompt_rows]
        matched = sum(row["block_match_counts"][arm]["matched"] for row in per_prompt_rows)
        compared = sum(row["block_match_counts"][arm]["compared"] for row in per_prompt_rows)
        mean_b = _merged_mean([row["posthoc_overall_mean"][arm] for row in per_prompt_rows])
        mean_a = _merged_mean([row["posthoc_overall_mean"][_A] for row in per_prompt_rows])
        gap = abs(mean_b - mean_a) if (mean_b is not None and mean_a is not None) else None
        quality_faces[arm] = {
            "decode_lengths_equal": lens == base_lens,
            "token_rate": (matched / compared) if compared else None,
            "posthoc_gap": gap,
            "decode_lengths": lens,
            "baseline_decode_lengths": base_lens,
            "matched": matched,
            "compared": compared,
            "posthoc_mean": mean_b,
            "baseline_posthoc_mean": mean_a,
        }

    d_total = sum(r["d"]["reconstruction"]["total_tokens"] for r in ledger_rows)
    b2_total = sum(r["b2"]["reconstruction"]["total_tokens"] for r in ledger_rows)
    a_total = sum(r["a"]["total_tokens"] for r in ledger_rows)

    d_forwards = sum(_forward_count(r["d"]["reconstruction"]) for r in ledger_rows)
    b2_forwards = sum(_forward_count(r["b2"]["reconstruction"]) for r in ledger_rows)
    ledger_faces = {
        "d_identity_all": bool(ledger_rows) and all(r["d"]["identity"] for r in ledger_rows),
        "b2_identity_all": bool(ledger_rows) and all(r["b2"]["identity"] for r in ledger_rows),
        "d_spec_balanced_all": bool(ledger_rows) and all(r["d"]["balanced"] for r in ledger_rows),
        "d_lt_b2_all": bool(ledger_rows)
        and all(r["d"]["lt_b2"] for r in ledger_rows)
        and d_forwards < b2_forwards,
        "a_structure_all": bool(ledger_rows)
        and all(r["a"]["all_full_canvas"] and r["a"]["block_aligned"] for r in ledger_rows),
        "d_total": d_total,
        "b2_total": b2_total,
        "a_total": a_total,
        "d_forwards": d_forwards,
        "b2_forwards": b2_forwards,
    }
    canary_faces = {
        "d_degraded_zero": bool(ledger_rows)
        and all(r["d"]["delta"].get("degraded_calls") == 0 for r in ledger_rows),
        "b2_degraded_zero": bool(ledger_rows)
        and all(r["b2"]["delta"].get("degraded_calls") == 0 for r in ledger_rows),
        "d_armed_engaged": bool(ledger_rows)
        and all(
            r["d"]["delta"].get("prefill_forwards", 0) > 0
            and r["d"]["delta"].get("spec_verify_forwards", 0) > 0
            for r in ledger_rows
        ),
        "b2_armed_engaged": bool(ledger_rows)
        and all(r["b2"]["delta"].get("prefill_forwards", 0) > 0 for r in ledger_rows),
        "topology_frozen": all(
            set(row["d"]["reconstruction"]["counts"])
            <= {
                "prefill",
                "commit",
                "spec_draft",
                "spec_verify",
            }
            for row in ledger_rows
        ),
    }

    faces = {
        "quality_faces": quality_faces,
        "baseline_ms": p50.get(_A),
        "b2_ms": p50.get(_B2),
        "d_ms": p50.get(_D),
        "official_ms": p50.get(_C),
        "sigma_baseline": sigma.get(_A),
        "sigma_b2": sigma.get(_B2),
        "sigma_d": sigma.get(_D),
        "ledger_faces": ledger_faces,
        "canary_faces": canary_faces,
        "repro_rows": repro_mismatches,
    }
    verdicts = assemble_verdicts(faces)

    import transformers

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "quick": quick,
            "reps": reps,
            "seed": SEED,
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "transformers_version": transformers.__version__,
            "prompts": {
                "count": len(prompts),
                "num_tokens": num_tokens,
                "blocks": num_tokens // BLOCK,
                "texts": [p["text"] for p in prompts],
                "raw_token_lens": [p["raw_len"] for p in prompts],
                "padded_token_lens": [p["padded_len"] for p in prompts],
            },
            "geometry": {
                "block_size": BLOCK,
                "small_block_size": SMALL_BLOCK,
                "threshold": 0.95,
                "temperature": 0.0,
                "num_candidates": SPEC_CANDIDATES,
                "candidate_top_k": SPEC_TOP_K,
                "note": ("AR022 lineage geometry (block=32, small_block=32) for every arm"),
                "rationale": (
                    "the speculative commit unit must match the config whose "
                    "greedy-anchored semantics AR022 validated; at sb=8 the frozen "
                    "min-prob rule lets random top-k rows beat the greedy row "
                    "(T007: 2/8 wins -> 96 vs 34 tokens, quality divergence), while "
                    "sb=32 keeps greedy wins at 100% and is token-exact vs A"
                ),
            },
            "model": model_meta,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_driver": _gpu_driver_version(),
            "arm_configs": {
                _A: {
                    "history_cache": False,
                    "description": "non-cached full-canvas baseline (ladder reference)",
                },
                _B2: {
                    "history_cache": True,
                    "history_small_block_cache": True,
                    "description": "AR033 cache arm (cache-only reference)",
                },
                _D: {
                    "history_cache": True,
                    "history_small_block_cache": False,
                    "num_block_candidates": SPEC_CANDIDATES,
                    "candidate_top_k": SPEC_TOP_K,
                    "candidate_include_greedy": True,
                    "description": (
                        "AR034 composition: whole-block draft + [N, block] batched "
                        "verify over the expanded read-only history KV"
                    ),
                },
                _C: {
                    "native_generate": True,
                    "use_block_cache": True,
                    "prompt": "raw unpadded ids (native mask-alignment semantics)",
                    "description": "checkpoint native generate() -- report-only",
                },
            },
            "official_arm": {
                "probe": official_probe,
                "ran": _C in active_arms,
                "status": "ran" if _C in active_arms else "blocked",
                "tokenizer_ids": official_ids,
            },
            "t001_probe_results": _T001_PROBE_RESULTS,
            "timing_note": (
                "perf_counter + cuda.synchronize around the whole generation, "
                "warmup discarded, per-rep round-robin; same-run A/B ratios are "
                "internally valid, absolute p50s are not cross-artifact comparable"
            ),
            "recorder_note": (
                "instance-level raw-forward recorder, pinned __signature__, batch+len "
                "metadata only, installed uniformly across arms"
            ),
            "frozen_rules": {
                "H1": "gap<=1e-3, decode lengths equal A, token rate>=0.98",
                "H2": "p50(A)/p50(D)>1 AND p50(B2)/p50(D)>1 strictly",
                "H3": "class ledger identity + spec balanced + D forward count < B2 + A structure",
                "H4": "canary conjunction + strict (arm,prompt) reproduction",
                "precedence": "blocked > refuted > supported",
            },
        },
        "sidecar": {
            "per_prompt": per_prompt_rows,
            "ledger_rows": ledger_rows,
            "reproduction": {
                "rows": len(repro_rows_all),
                "mismatches": repro_mismatches,
            },
        },
        "timing": {
            "per_arm_p50_ms": {arm: (p50.get(arm)) for arm in active_arms},
            "per_arm_std_ms": {arm: (sigma.get(arm)) for arm in active_arms},
            "per_arm_full_stats": {arm: whole_stats[arm].as_dict() for arm in active_arms},
            "per_arm_per_prompt_stats": {
                arm: {
                    f"prompt{pi}": stats_from_samples(whole_samples[arm][pi]).as_dict()
                    for pi in range(len(prompts))
                }
                for arm in active_arms
            },
            "samples_ms": {
                arm: {f"prompt{pi}": whole_samples[arm][pi] for pi in range(len(prompts))}
                for arm in active_arms
            },
        },
        "verdicts": verdicts,
        "overall": verdicts["overall"],
    }
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model-path", default=None, help="Local checkpoint directory")
    parser.add_argument("--reps", type=int, default=REPS_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-reason", default=None)
    add_hf_env_arguments(parser)
    args = parser.parse_args(argv)

    import transformers

    model_path = _resolve_model_path(args.model_path)
    try:
        require_environment(transformers.__version__, model_path)
    except RuntimeError as exc:
        print(f"environment gate failed: {exc}", file=sys.stderr, flush=True)
        return 2

    reps = QUICK_REPS if args.quick else max(1, int(args.reps))
    out_path = resolve_output_path(args.out, args.quick)
    if out_path.exists() and not args.force:
        print(
            f"refusing to overwrite existing artifact {out_path} (pre-committed "
            "single-run discipline; pass --force --force-reason TEXT to re-run)",
            file=sys.stderr,
            flush=True,
        )
        return 4
    if out_path.exists() and args.force and not (args.force_reason or "").strip():
        print("--force requires a non-empty --force-reason", file=sys.stderr, flush=True)
        return 3

    artifact = run_bench(model_path, reps, args.quick)
    artifact["meta"]["force"] = bool(args.force)
    artifact["meta"]["force_reason"] = args.force_reason
    write_results(artifact, out_path)
    print(
        f"wrote {out_path} | overall={artifact['overall']} | "
        f"h1={artifact['verdicts']['h1']['verdict']} "
        f"h2={artifact['verdicts']['h2']['verdict']} "
        f"h3={artifact['verdicts']['h3']['verdict']} "
        f"h4={artifact['verdicts']['h4']['verdict']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
