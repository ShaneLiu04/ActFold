"""AR033/T006: four-arm block-causal history-cache benchmark (real ckpt).

Runs the REAL Fast-dLLM-v2 generation on the local Fast-dLLM-v2-1.5B
checkpoint in FOUR arms over one shared model instance, prompt set, and
seed (the AR031/AR032 protocol; every ActFold arm uses the OFFICIAL v2
geometry ``block_size=32, small_block_size=8`` so the schedule semantics
are shared and the official arm compares natively):

* ``baseline`` (A) -- the official Fast-dLLM v2 recipe as implemented in
  ActFold (``folded_model=None``, caches off): full-canvas recompute
  forwards, with the §6.5 advance erratum applied (unshifted tail row).
* ``cache-cross`` (B1) -- ``history_cache=True``: the cross-block history
  KV cache (exact face -- the T001 P3 probe measured maxdiff 0.0 vs the
  full-canvas forward on this checkpoint); expected token-identical to A.
* ``cache-full`` (B2) -- B1 + ``history_small_block_cache=True``: the
  intra-block small-block KV cache (official-parity APPROXIMATE face --
  the T001 P4 probe measured 0.3125 bf16 kernel-path drift; absorbed by
  the frozen H1 threshold).
* ``official`` (C) -- the checkpoint's NATIVE ``generate()``
  (modeling.py:652-831, ``use_block_cache=True``) on the raw unpadded
  prompt: report-only positioning arm; incompatible -> blocked with the
  minimal probe signature, zero fabricated numbers (srs FR4).

Hypotheses (declared before measurement; the verdict functions below are
frozen BEFORE any data is collected; honest negatives are delivered
verbatim, never rerun-until-pass):

* **H1** quality alignment (each B arm vs A): (a) per-prompt decode
  lengths strictly equal; (b) generated-region token consistency rate >=
  0.98 (B1 expected 1.0 on the P3 bit-exact basis; B2's P4 drift is what
  the threshold absorbs); (c) merged posthoc overall-confidence gap <=
  0.01. Any None face value refutes that face.
* **H2** p50 speedup: ``p50(A)/p50(B1) > 1.0`` AND ``p50(A)/p50(B2) >
  1.0`` strictly; every ratio against C is report-only (parity
  positioning, no gate); the merged two-sample std is report-only.
* **H3** integer ledgers: for EACH B arm, per prompt, the
  ``history_cache_stats`` delta equals the class-wise reconstruction of
  the observed raw-forward traffic EXACTLY (all five counters; every
  observed call length matches its class's geometric value) AND the
  degraded counter delta is 0; the A arm's traffic is all full-canvas
  block-aligned calls (``sum(iter * T_call)`` == the observed total);
  ``forward_tokens(B) < forward_tokens(A)`` per prompt and aggregate
  (structural, still judged explicitly).
* **H4** honest ledger: the armed-path canary (both B arms: degraded
  delta == 0 everywhere, engagement > 0) AND the internal reproduction
  double-run strictly row-equal per ``(arm, prompt_index)`` -- a
  mismatch blocks ALL verdicts (the AR031 one-vote veto); C-arm blocked
  degradation is a legal recorded outcome, never a fabricated number;
  overwrite protection refuses an existing formal artifact unless BOTH
  ``--force`` and ``--force-reason`` are given.

Protocol: seed 2026, P=2 pre-padded prompts, num_tokens=128 (4 blocks)
formal / 64 (2 blocks) quick; warmup one full discarded sample per
(arm, prompt); the untimed sidecar generation runs TWICE (pass A records
the quality faces + ledgers, pass B is the reproduction self-check);
per-rep round-robin over the active arms (rep r starts at ``r % n``);
host perf_counter + cuda synchronize (#39); reps=30 pre-committed
single-shot verdict. ``--quick``: reps=3, single prompt, num_tokens=64,
separate artifact path, never the formal one.

The raw-forward recorder wraps the remote module's ``forward`` at the
INSTANCE level (the pinned ``__signature__`` keeps the sampler's
capability probe exact); it records host metadata only (input length +
kwargs classes), zero device readback, and is installed uniformly across
all arms so the timing perturbation is shared.

Environment gate (construction time, before any model load or artifact
write): ``transformers==4.53.1`` (the AR019 venv) and a local checkpoint
directory containing ``config.json``; failures exit rc=2 and write
nothing. A mid-run arm failure (OOM etc.) crashes without writing any
artifact (zero fabricated numbers, the AR031 caliber).

Run (AR019 venv, transformers==4.53.1)::

    python -m scripts.ar033_history_cache_bench [--quick] [--reps 30]
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

SCHEMA = "ar033_history_cache_bench"
SCHEMA_VERSION = 1
ARMS = ("baseline", "cache-cross", "cache-full", "official")
DEFAULT_OUTPUT_PATH = Path("results/optimization/ar033_history_cache_bench.json")
QUICK_OUTPUT_PATH = Path("results/optimization/ar033_history_cache_bench_quick.json")
REPS_DEFAULT = 30
QUICK_REPS = 3
REQUIRED_TRANSFORMERS = "4.53.1"
SMALL_BLOCK = 8
_H1_TOKEN_RATE_MIN = 0.98
_H1_POSTHOC_GAP_MAX = 0.01
_BASELINE = "baseline"
_B1 = "cache-cross"
_B2 = "cache-full"
_OFFICIAL = "official"
_ACTFOLD_ARMS = (_BASELINE, _B1, _B2)
_B_ARMS = (_B1, _B2)

ARTIFACT_LIMITATION_NOTES: tuple[str, ...] = (
    "this bench moves every ActFold arm to the OFFICIAL v2 geometry "
    "(block_size=32, small_block_size=8); the AR019-AR032 lineage ran "
    "small_block_size=BLOCK, so cross-artifact p50 / decode-length "
    "comparison against pre-AR033 artifacts is INVALID (the #51 "
    "cross-rung timing caveat; schedule granularity changes the "
    "iteration structure by design)",
    "the baseline arm carries the section-6.5 advance erratum fix "
    "(unshifted tail row raw[T-1]); its block-advance tokens are "
    "official-semantics, unlike every pre-AR033 multi-block artifact "
    "(whose A/B internal comparisons remain valid -- both arms carried "
    "the old convention)",
    "B2's intra-block small-block cache is an official-parity "
    "APPROXIMATE face (T001 P4: 0.3125 bf16 kernel-path drift on the "
    "replace path); B1 (cross-block only) is the exact face (T001 P3: "
    "maxdiff 0.0, the basis for the expected 1.0 token rate); the UT "
    "fixture keeps both bit-pinned on fp32 CPU",
    "the official arm C runs the checkpoint's native generate() on the "
    "RAW unpadded prompt (its own mask-alignment semantics); its p50 "
    "is report-only parity positioning, never a quality face",
    "pad tokens are NOT excluded on the real remote model (eval "
    "unconditionally overwrites the external attention_mask with the "
    "pure block-causal mask -- the B5 erratum note); all arms share "
    "the semantics",
    "timing follows the #39 wall-clock discipline (perf_counter + "
    "synchronize, warmup discarded, per-rep round-robin); the Windows "
    "WDDM display-driver mode does not lock clocks (#46) -- p50 over "
    "reps with interleaving is the noise mitigation",
    "the raw-forward recorder adds one host-only Python frame per "
    "forward, installed uniformly across all arms (shared perturbation)",
)

_HYPOTHESES: dict[str, str] = {
    "h1": (
        "quality alignment (each B arm vs A): (a) per-prompt decode "
        "lengths strictly equal; (b) generated-region token consistency "
        "rate >= 0.98 (B1 expected 1.0 on the P3 bit-exact basis, B2's "
        "P4 drift absorbed by the threshold); (c) merged posthoc "
        "overall-confidence gap <= 0.01; any None face value refutes "
        "that face"
    ),
    "h2": (
        "p50 speedup: p50(A)/p50(B1) > 1.0 AND p50(A)/p50(B2) > 1.0 "
        "strictly; every ratio against the official arm C is "
        "report-only (parity positioning, no gate); the merged "
        "two-sample std is report-only"
    ),
    "h3": (
        "integer ledgers: per (B arm, prompt) the history_cache_stats "
        "delta equals the class-wise reconstruction of the observed "
        "raw-forward traffic EXACTLY (all five counters, every call "
        "length matching its class geometry, degraded delta 0); the A "
        "arm's traffic is all full-canvas block-aligned calls; "
        "forward_tokens(B) < forward_tokens(A) per prompt and aggregate"
    ),
    "h4": (
        "honest ledger: the armed-path canary (both B arms: degraded "
        "delta == 0 everywhere, engagement > 0) AND the internal "
        "reproduction double-run strictly row-equal per (arm, "
        "prompt_index) -- a mismatch blocks ALL verdicts (the AR031 "
        "one-vote veto); a blocked C arm is a legal recorded outcome; "
        "overwrite protection refuses an existing formal artifact "
        "unless BOTH --force and --force-reason are given"
    ),
}

# T001 probe results (design section 7 table) -- recorded verbatim in the
# artifact meta so the evidence chain is self-contained.
_T001_PROBE_RESULTS: dict[str, str] = {
    "p1_remote_cache_kwargs": (
        "all accepted as named parameters: use_cache / past_key_values / "
        "update_past_key_values / block_size / use_block_cache / "
        "block_past_key_values / replace_position"
    ),
    "p2_advance_convention": (
        "teacher-forcing hit_next(R1)=0.9028 / hit_self(R2)=0.0000 -- "
        "the unshifted tail row is the official advance convention"
    ),
    "p3_cache_row_alignment": (
        "prefill/slice cache forward vs full canvas maxdiff=0.000000 "
        "(bitwise exact), argmax match 1.0 -- the cross-block history "
        "cache is exact on this checkpoint"
    ),
    "p4_block_replace_path": (
        "block-cache partial forward maxdiff=0.3125 -- sdpa kernel-path "
        "numerical drift under mask-semantics-equivalent calls (bf16 "
        "28-layer accumulation)"
    ),
}


# ---------------------------------------------------------------------------
# Frozen verdict functions (declared before any data collection; honest
# negatives are returned verbatim, never raised)
# ---------------------------------------------------------------------------
def verdict_h1(quality_faces: dict[str, Any]) -> dict[str, Any]:
    """H1 quality alignment: every B arm passes all three faces -> supported."""
    faces: dict[str, Any] = {}
    numbers: dict[str, Any] = {}
    supported = True
    for arm in _B_ARMS:
        entry = quality_faces.get(arm, {})
        lens_eq = bool(entry.get("decode_lengths_equal"))
        rate = entry.get("token_rate")
        gap = entry.get("posthoc_gap")
        rate_ok = rate is not None and math.isfinite(rate) and rate >= _H1_TOKEN_RATE_MIN
        gap_ok = gap is not None and math.isfinite(gap) and gap <= _H1_POSTHOC_GAP_MAX
        faces[arm] = {
            "a_decode_lengths_equal": lens_eq,
            "b_token_rate_ge_0.98": bool(rate_ok),
            "c_posthoc_gap_le_0.01": bool(gap_ok),
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
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": faces,
        "numbers": numbers,
    }


def _strict_ratio(faster_ms: float | None, slower_ms: float | None) -> float | None:
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
    b1_ms: float | None,
    b2_ms: float | None,
    sigma_b1: float | None,
    sigma_b2: float | None,
    official_ms: float | None,
) -> dict[str, Any]:
    """H2 p50 speedup: BOTH strict ratios > 1.0; C ratios report-only."""
    ratio_b1 = _strict_ratio(b1_ms, baseline_ms)
    ratio_b2 = _strict_ratio(b2_ms, baseline_ms)
    supported = ratio_b1 is not None and ratio_b1 > 1.0 and ratio_b2 is not None and ratio_b2 > 1.0
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": {
            "a_baseline_over_b1_gt_1": bool(ratio_b1 is not None and ratio_b1 > 1.0),
            "b_baseline_over_b2_gt_1": bool(ratio_b2 is not None and ratio_b2 > 1.0),
        },
        "numbers": {
            "baseline_ms": baseline_ms,
            "b1_ms": b1_ms,
            "b2_ms": b2_ms,
            "official_ms": official_ms,
            "speedup_b1_vs_baseline": round(ratio_b1, 4) if ratio_b1 else None,
            "speedup_b1_vs_baseline_raw": ratio_b1,
            "speedup_b2_vs_baseline": round(ratio_b2, 4) if ratio_b2 else None,
            "speedup_b2_vs_baseline_raw": ratio_b2,
            "sigma_b1": sigma_b1,
            "sigma_b2": sigma_b2,
            "report_only_official_ratios": {
                "b1_vs_official": round(ratio, 4)
                if (ratio := _strict_ratio(b1_ms, official_ms))
                else None,
                "b2_vs_official": round(ratio, 4)
                if (ratio := _strict_ratio(b2_ms, official_ms))
                else None,
                "official_vs_baseline": round(ratio, 4)
                if (ratio := _strict_ratio(official_ms, baseline_ms))
                else None,
            },
            "rule": (
                "supported iff p50(A)/p50(B1) > 1.0 AND p50(A)/p50(B2) > 1.0 "
                "strictly; official-arm ratios and sigmas are report-only"
            ),
        },
    }


def verdict_h3(ledger_faces: dict[str, Any]) -> dict[str, Any]:
    """H3 integer ledgers: every face exact -> supported."""
    faces = {
        "b1_ledger_identity": bool(ledger_faces.get("b1_identity_all")),
        "b2_ledger_identity": bool(ledger_faces.get("b2_identity_all")),
        "b1_lt_a": bool(ledger_faces.get("b1_lt_a_all")),
        "b2_lt_a": bool(ledger_faces.get("b2_lt_a_all")),
        "a_full_canvas_structure": bool(ledger_faces.get("a_structure_all")),
    }
    supported = all(faces.values())
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": faces,
        "numbers": {
            "b1_forward_tokens_total": ledger_faces.get("b1_total"),
            "b2_forward_tokens_total": ledger_faces.get("b2_total"),
            "a_forward_tokens_total": ledger_faces.get("a_total"),
            "rule": (
                "per (B arm, prompt): stats delta == class-wise "
                "reconstruction (5 counters exact + call-length geometry "
                "+ degraded 0); A all full-canvas block-aligned; "
                "forward_tokens(B) < forward_tokens(A) per prompt and total"
            ),
        },
    }


def _payload_equal(first: Any, second: Any) -> bool:
    """Strict payload equality (tensors compare via ``torch.equal``)."""
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return bool(torch.equal(first, second))
    return bool(first == second)


def repro_mismatch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strict per-``(arm, prompt_index)`` double-run comparison rows.

    Each input row carries its OWN ``first`` / ``second`` payloads (the
    AR031 leak-shape lesson, UT-3166): the pairing key is the row itself.
    """
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        first = row.get("first")
        second = row.get("second")
        if not _payload_equal(first, second):
            mismatches.append(
                {
                    "arm": row.get("arm"),
                    "prompt_index": row.get("prompt_index"),
                    "first": first,
                    "second": second,
                }
            )
    return mismatches


def verdict_h4(canary_faces: dict[str, Any], repro_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """H4 honest ledger: canary AND an empty mismatch list."""
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
    """Precedence: any refuted -> refuted; all supported -> supported; else inconclusive."""
    verdicts = (h1, h2, h3, h4)
    if any(v == "refuted" for v in verdicts):
        return "refuted"
    if all(v == "supported" for v in verdicts):
        return "supported"
    return "inconclusive"


def assemble_verdicts(faces: dict[str, Any]) -> dict[str, Any]:
    """Pure verdict assembly; a repro mismatch blocks ALL verdicts (AR031)."""
    h1 = verdict_h1(faces["quality_faces"])
    h2 = verdict_h2(
        faces["baseline_ms"],
        faces["b1_ms"],
        faces["b2_ms"],
        faces["sigma_b1"],
        faces["sigma_b2"],
        faces["official_ms"],
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
    return {
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "overall": overall,
    }


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
# Raw-forward recorder (instance-level, probe-exact signature pinning)
# ---------------------------------------------------------------------------
class _RawCallRecorder:
    """Host-metadata recorder over the raw remote module's forward.

    Installed once, uniformly across all arms (one extra Python frame per
    forward, shared perturbation). Records the input length and the
    cache-kwarg CLASS of every call -- host metadata only, zero device
    readback. The pinned ``__signature__`` keeps the sampler's
    ``_probe_history_cache_support`` exact (named-parameter acceptance).
    """

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


def _classify_raw_call(rec: dict[str, Any]) -> str:
    """Classify one recorded raw forward by its cache-kwarg signature.

    The classes are in bijection with the cache orchestration's call
    sites (``_history_cached_sample``): prefill / commit / cross_slice /
    block_rebuild / sb0_partial / sb_partial; the legacy full-canvas path
    carries no cache kwargs at all.
    """
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
    return "cross_slice"


def _reconstruct_cache_ledger(
    calls: list[dict[str, Any]],
    padded_len: int,
    block_size: int,
    small_block_size: int,
) -> dict[str, Any]:
    """Class-wise ledger reconstruction of one B-arm generation.

    Every call's input length must equal its class's geometric value
    (this bench's canvases are always block-aligned: prompts pre-pad to
    block multiples and num_tokens is a block multiple, so the active
    block is always full); a mismatch refutes the structure face. The
    ``predicted`` counters are the sampler's own accounting contract.
    """
    counts: collections.Counter[str] = collections.Counter()
    total = 0
    structure_ok = True
    expected_len = {
        "prefill": padded_len,
        "commit": block_size,
        "block_rebuild": block_size,
        "sb0_partial": small_block_size,
        "sb_partial": small_block_size + 1,
        "cross_slice": block_size,
    }
    for rec in calls:
        cls = _classify_raw_call(rec)
        counts[cls] += 1
        total += int(rec["len"])
        if cls in expected_len and int(rec["len"]) != expected_len[cls]:
            structure_ok = False
    predicted = {
        "prefill_forwards": counts["prefill"],
        "slice_forwards": (
            counts["cross_slice"]
            + counts["block_rebuild"]
            + counts["sb0_partial"]
            + counts["sb_partial"]
        ),
        "commit_forwards": counts["commit"],
        "block_rebuilds": counts["block_rebuild"],
        "forward_tokens": total,
    }
    return {
        "counts": dict(counts),
        "total_tokens": total,
        "structure_ok": structure_ok,
        "max_call_len": max((int(r["len"]) for r in calls), default=0),
        "predicted": predicted,
    }


def _reconstruct_legacy_ledger(
    calls: list[dict[str, Any]],
    padded_len: int,
    block_size: int,
) -> dict[str, Any]:
    """Ledger reconstruction of the A arm: all full-canvas block-aligned calls."""
    total = 0
    all_full_canvas = True
    by_len: collections.Counter[int] = collections.Counter()
    for rec in calls:
        if _classify_raw_call(rec) != "legacy_full_canvas":
            all_full_canvas = False
        by_len[int(rec["len"])] += 1
        total += int(rec["len"])
    block_aligned = all(
        length % block_size == 0 and length >= padded_len + block_size for length in by_len
    )
    return {
        "calls": len(calls),
        "total_tokens": total,
        "all_full_canvas": all_full_canvas,
        "block_aligned": block_aligned,
        "by_canvas_len": {str(length): cnt for length, cnt in sorted(by_len.items())},
        "sum_iter_times_tcall": sum(length * cnt for length, cnt in by_len.items()),
    }


def _probe_official_generate(model: Any) -> dict[str, Any]:
    """Probe the checkpoint's native ``generate()`` (the C arm).

    A missing/unusable native generate is a legal BLOCKED outcome (srs
    FR4): recorded with the minimal signature, zero fabricated numbers.
    """
    generate = getattr(model, "generate", None)
    if not callable(generate):
        return {"supported": False, "reason": "no callable native generate()"}
    try:
        params = set(inspect.signature(generate).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return {"supported": False, "reason": "native generate() signature unreadable"}
    required = {
        "input_ids",
        "max_new_tokens",
        "mask_id",
        "threshold",
        "small_block_size",
        "block_size",
        "stop_token",
        "temperature",
        "use_block_cache",
    }
    missing = sorted(required - params)
    if missing:
        return {
            "supported": False,
            "reason": f"native generate() lacks required parameters: {missing}",
            "signature_params": sorted(params),
        }
    return {"supported": True, "reason": "", "signature_params": sorted(params)}


# ---------------------------------------------------------------------------
# Arm construction / local helpers
# ---------------------------------------------------------------------------
def _arm_sampler_config(
    num_tokens: int, history_cache: bool, history_small_block_cache: bool
) -> FastDLLMSamplerConfig:
    """Official v2 geometry (block=32, small=8) for EVERY ActFold arm.

    All ActFold arms share one schedule semantics (small-block
    granularity); only the forward SOURCE differs. The AR019-AR032
    lineage ran ``small_block_size=BLOCK``; this bench moves to the
    official geometry so the sb-cache arm is meaningful (S=1 degenerates
    the small-block cache to whole-block rewrites) -- cross-artifact
    p50/decode-length comparison against pre-AR033 benches is invalid by
    this change (the limitation notes carry it).
    """
    return FastDLLMSamplerConfig(
        num_tokens=num_tokens,
        block_size=BLOCK,
        small_block_size=SMALL_BLOCK,
        threshold=0.95,
        temperature=0.0,
        seed=SEED,
        history_cache=history_cache,
        history_small_block_cache=history_small_block_cache,
    )


def _merged_block_exact_match(
    base_seq: torch.Tensor,
    other_seq: torch.Tensor,
    padded_len: int,
    num_tokens: int,
) -> dict[str, Any]:
    """Matched/compared counts on the block grid over the generated region."""
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


def _merged_mean(values: list[Any]) -> float | None:
    """Merged mean; ANY None member propagates None (honest no-data)."""
    if not values or any(v is None for v in values):
        return None
    return sum(float(v) for v in values) / len(values)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run_bench(model_path: Path, reps: int, quick: bool) -> dict[str, Any]:
    """Run the four-arm history-cache benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("run_bench requires a CUDA device")

    num_tokens = QUICK_NUM_TOKENS if quick else FULL_NUM_TOKENS
    model, tokenizer, model_meta = _load_model(model_path)
    prompts = _prepare_prompts(tokenizer, quick)
    llm = _BenchDiffusionLLM(model, tokenizer)
    # Neutralize the wrapper's per-block sync anchors: this bench times
    # the whole-generation wall clock only, and the cache arms bypass the
    # wrapper entirely (raw-module calls) -- uniform timing discipline.
    llm.block_timer.note = lambda block_idx: None  # type: ignore[method-assign, assignment]

    recorder = _RawCallRecorder(model)
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
        arm: FastDLLMSampler(
            llm,
            config=_arm_sampler_config(
                num_tokens,
                history_cache=arm in _B_ARMS,
                history_small_block_cache=arm == _B2,
            ),
        )
        for arm in _ACTFOLD_ARMS
    }
    active_arms = [arm for arm in ARMS if arm != _OFFICIAL or official_probe["supported"]]
    n_arms = len(active_arms)

    def _generate(arm: str, prompt: dict[str, Any]) -> dict[str, Any]:
        rec0 = len(recorder.calls)
        if arm == _OFFICIAL:
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

    # -- warmup: one full discarded sample per (arm, prompt) --------------
    for arm in active_arms:
        for prompt in prompts:
            _generate(arm, prompt)
        print(f"warmup done: {arm}", flush=True)

    # -- sidecar pass A (untimed quality faces + ledgers) ------------------
    per_prompt_rows: list[dict[str, Any]] = []
    quality_detail: list[dict[str, Any]] = []
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
            if arm != _OFFICIAL:
                posthocs[arm] = _posthoc_confidence(model, seqs[arm][0], padded_len, num_tokens)

        a_recon = _reconstruct_legacy_ledger(arm_records[_BASELINE]["calls"], padded_len, BLOCK)
        for arm in _B_ARMS:
            recon = _reconstruct_cache_ledger(
                arm_records[arm]["calls"], padded_len, BLOCK, SMALL_BLOCK
            )
            delta = arm_records[arm]["stats_delta"]
            identity = bool(
                recon["structure_ok"]
                and delta.get("degraded_calls") == 0
                and all(delta.get(key, -1) == value for key, value in recon["predicted"].items())
            )
            ledger_rows.append(
                {
                    "arm": arm,
                    "prompt_index": pi,
                    "stats_delta": delta,
                    "reconstruction": recon,
                    "identity": identity,
                    "b_total": recon["total_tokens"],
                    "a_total": a_recon["total_tokens"],
                    "b_lt_a": bool(recon["total_tokens"] < a_recon["total_tokens"]),
                }
            )
        ledger_rows.append(
            {
                "arm": _BASELINE,
                "prompt_index": pi,
                "reconstruction": a_recon,
                "a_structure_ok": bool(a_recon["all_full_canvas"] and a_recon["block_aligned"]),
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
                    arm: _merged_block_exact_match(
                        seqs[_BASELINE], seqs[arm], padded_len, num_tokens
                    )
                    for arm in _B_ARMS
                },
            }
        )
        quality_detail.append(
            {
                "prompt_index": pi,
                "posthoc_confidence": posthocs,
                "official_decode_length": (
                    int(seqs[_OFFICIAL].shape[1]) if _OFFICIAL in seqs else None
                ),
            }
        )
        print(
            f"sidecar prompt {pi}: len(A/B1/B2) = "
            f"{seqs[_BASELINE].shape[1]}/{seqs[_B1].shape[1]}/{seqs[_B2].shape[1]}"
            + (f", official={seqs[_OFFICIAL].shape[1]}" if _OFFICIAL in seqs else ""),
            flush=True,
        )

    # -- sidecar pass B (the internal reproduction self-check) ------------
    repro_rows: list[dict[str, Any]] = []
    for pi, prompt in enumerate(prompts):
        for arm in active_arms:
            rec = _generate(arm, prompt)
            second = {
                "decode_length": int(rec["sequences"].shape[1]),
                "tokens": rec["sequences"][0].tolist(),
            }
            repro_rows.append(
                {
                    "arm": arm,
                    "prompt_index": pi,
                    "first": pass_a_payloads[(arm, pi)],
                    "second": second,
                }
            )
    repro_mismatches = repro_mismatch_rows(repro_rows)

    # -- timed reps: per-rep round-robin over the active arms --------------
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

    # -- statistics ----------------------------------------------------------
    def _pooled(arm: str) -> list[float]:
        return [x for pi in range(len(prompts)) for x in whole_samples[arm][pi]]

    whole_stats = {arm: stats_from_samples(_pooled(arm)) for arm in active_arms}
    per_prompt_stats = {
        arm: {
            f"prompt{pi}": stats_from_samples(whole_samples[arm][pi]).as_dict()
            for pi in range(len(prompts))
        }
        for arm in active_arms
    }
    p50 = {arm: whole_stats[arm].p50 for arm in active_arms}
    sigma_b1 = (
        (whole_stats[_BASELINE].std ** 2 + whole_stats[_B1].std ** 2) ** 0.5
        if _B1 in whole_stats
        else None
    )
    sigma_b2 = (
        (whole_stats[_BASELINE].std ** 2 + whole_stats[_B2].std ** 2) ** 0.5
        if _B2 in whole_stats
        else None
    )

    # -- quality faces -------------------------------------------------------
    quality_faces: dict[str, Any] = {}
    for arm in _B_ARMS:
        lens = [row["decode_length"][arm] for row in per_prompt_rows]
        base_lens = [row["decode_length"][_BASELINE] for row in per_prompt_rows]
        matched = sum(row["block_match_counts"][arm]["matched"] for row in per_prompt_rows)
        compared = sum(row["block_match_counts"][arm]["compared"] for row in per_prompt_rows)
        mean_b = _merged_mean([row["posthoc_overall_mean"][arm] for row in per_prompt_rows])
        mean_a = _merged_mean([row["posthoc_overall_mean"][_BASELINE] for row in per_prompt_rows])
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

    # -- ledger faces --------------------------------------------------------
    b_rows = {arm: [r for r in ledger_rows if r["arm"] == arm] for arm in _B_ARMS}
    a_rows = [r for r in ledger_rows if r["arm"] == _BASELINE]
    b1_total = sum(r["b_total"] for r in b_rows[_B1])
    b2_total = sum(r["b_total"] for r in b_rows[_B2])
    a_total = sum(r["reconstruction"]["total_tokens"] for r in a_rows)
    ledger_faces = {
        "b1_identity_all": bool(b_rows[_B1]) and all(r["identity"] for r in b_rows[_B1]),
        "b2_identity_all": bool(b_rows[_B2]) and all(r["identity"] for r in b_rows[_B2]),
        "b1_lt_a_all": bool(b_rows[_B1])
        and all(r["b_lt_a"] for r in b_rows[_B1])
        and b1_total < a_total,
        "b2_lt_a_all": bool(b_rows[_B2])
        and all(r["b_lt_a"] for r in b_rows[_B2])
        and b2_total < a_total,
        "a_structure_all": bool(a_rows) and all(r["a_structure_ok"] for r in a_rows),
        "b1_total": b1_total,
        "b2_total": b2_total,
        "a_total": a_total,
    }

    # -- canary faces --------------------------------------------------------
    canary_faces = {
        "b1_degraded_zero": bool(b_rows[_B1])
        and all(r["stats_delta"]["degraded_calls"] == 0 for r in b_rows[_B1]),
        "b2_degraded_zero": bool(b_rows[_B2])
        and all(r["stats_delta"]["degraded_calls"] == 0 for r in b_rows[_B2]),
        "b_armed_engaged": bool(b_rows[_B1])
        and bool(b_rows[_B2])
        and all(r["stats_delta"]["prefill_forwards"] > 0 for arm in _B_ARMS for r in b_rows[arm]),
    }

    faces = {
        "quality_faces": quality_faces,
        "baseline_ms": p50.get(_BASELINE),
        "b1_ms": p50.get(_B1),
        "b2_ms": p50.get(_B2),
        "official_ms": p50.get(_OFFICIAL),
        "sigma_b1": sigma_b1,
        "sigma_b2": sigma_b2,
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
                "note": (
                    "official v2 geometry for every arm (the AR019-AR032 "
                    "lineage ran small_block_size=BLOCK -- cross-artifact "
                    "p50/decode-length comparison invalid)"
                ),
            },
            "arm_configs": {
                _BASELINE: {
                    "folded_model": None,
                    "history_cache": False,
                    "history_small_block_cache": False,
                    "description": (
                        "the official Fast-dLLM v2 recipe as implemented in "
                        "ActFold: full-canvas recompute, section-6.5 advance "
                        "erratum applied (unshifted tail row)"
                    ),
                },
                _B1: {
                    "folded_model": None,
                    "history_cache": True,
                    "history_small_block_cache": False,
                    "description": (
                        "cross-block history KV cache -- the EXACT face "
                        "(T001 P3: maxdiff 0.0 vs the full canvas)"
                    ),
                },
                _B2: {
                    "folded_model": None,
                    "history_cache": True,
                    "history_small_block_cache": True,
                    "description": (
                        "cross-block + intra-block small-block KV cache -- "
                        "the official-parity APPROXIMATE face (T001 P4: "
                        "0.3125 bf16 replace-path drift)"
                    ),
                },
                _OFFICIAL: {
                    "native_generate": True,
                    "use_block_cache": True,
                    "prompt": "raw unpadded ids (native mask-alignment semantics)",
                    "description": (
                        "the checkpoint's own generate() (modeling.py:652-831) "
                        "-- report-only parity positioning arm"
                    ),
                },
            },
            "official_arm": {
                "probe": official_probe,
                "ran": _OFFICIAL in active_arms,
                "status": "ran" if _OFFICIAL in active_arms else "blocked",
                "tokenizer_ids": official_ids,
            },
            "t001_probe_results": _T001_PROBE_RESULTS,
            "recorder_note": (
                "the raw-forward recorder wraps the remote module's forward "
                "at the instance level (pinned __signature__, probe-exact); "
                "host metadata only, installed uniformly across arms"
            ),
            "warmup_strategy": (
                "one full discarded sample per (arm, prompt) before any "
                "timing; the untimed sidecar pass A records the quality "
                "faces and the integer ledgers, pass B is the internal "
                "reproduction self-check"
            ),
            "interleaving": (
                "per-rep round-robin over the active arms; rep r starts at "
                "position r%n (anti-drift, WDDM fairness)"
            ),
            "timing_method": (
                "host perf_counter + torch.cuda.synchronize (#39); "
                "whole-generation wall clock; the quality sidecar (posthoc "
                "re-scoring, match grids, ledger reconstruction) runs only "
                "on the untimed passes"
            ),
            "reproduction_declaration": (
                "the untimed sidecar generation runs TWICE per (arm, "
                "prompt); pass A records the quality faces and ledgers, "
                "and pass B must be strictly payload-equal on every (arm, "
                "prompt) row (decode length + the full committed token "
                "list); any mismatch blocks ALL verdicts (the AR031 "
                "one-vote veto); the check binds quick and formal alike"
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
                "python -m scripts.ar033_history_cache_bench"
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
            }
            for arm in active_arms
        },
        "speedups_p50": {
            "b1_vs_baseline": (round(p50[_BASELINE] / p50[_B1], 4) if p50.get(_B1) else None),
            "b2_vs_baseline": (round(p50[_BASELINE] / p50[_B2], 4) if p50.get(_B2) else None),
            "b2_vs_b1": (round(p50[_B1] / p50[_B2], 4) if p50.get(_B1) and p50.get(_B2) else None),
            "report_only": {
                "b1_vs_official": (
                    round(p50[_OFFICIAL] / p50[_B1], 4)
                    if p50.get(_OFFICIAL) and p50.get(_B1)
                    else None
                ),
                "b2_vs_official": (
                    round(p50[_OFFICIAL] / p50[_B2], 4)
                    if p50.get(_OFFICIAL) and p50.get(_B2)
                    else None
                ),
                "official_vs_baseline": (
                    round(p50[_BASELINE] / p50[_OFFICIAL], 4) if p50.get(_OFFICIAL) else None
                ),
            },
        },
        "quality": {
            "faces": quality_faces,
            "per_prompt": per_prompt_rows,
            "per_arm_sidecar": quality_detail,
        },
        "ledger": {
            "per_prompt": ledger_rows,
            "totals": {
                "b1_forward_tokens": b1_total,
                "b2_forward_tokens": b2_total,
                "a_forward_tokens": a_total,
            },
        },
        "canary": {
            "faces": {key: bool(value) for key, value in canary_faces.items()},
            "pass": all(bool(value) for value in canary_faces.values()),
        },
        "reproduction": {
            "mode": "internal-double-run",
            "ok": not repro_mismatches,
            "rows_compared": len(prompts) * n_arms,
            "mismatches": repro_mismatches,
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing formal artifact (requires --force-reason).",
    )
    parser.add_argument(
        "--force-reason",
        default=None,
        help="Non-empty audit reason recorded in the run log when --force is used.",
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

    out = resolve_output_path(args.out, quick)
    if not quick and out == DEFAULT_OUTPUT_PATH and out.exists():
        if not args.force:
            print(
                f"refusing to overwrite the existing formal artifact {out} "
                "(pre-committed single-shot discipline; pass --force with a "
                "--force-reason for a deliberate rerun)",
                file=sys.stderr,
            )
            return 4
        if not (args.force_reason and args.force_reason.strip()):
            print(
                "--force requires a non-empty --force-reason (audit trail)",
                file=sys.stderr,
            )
            return 3
        print(f"overwrite forced: {args.force_reason!r}", file=sys.stderr)

    data = run_bench(model_path, reps=reps, quick=quick)
    written = write_results(data, out)
    print()
    for key in ("h1", "h2", "h3", "h4"):
        entry = data["hypotheses_verdicts"][key]
        print(f"{key.upper()}: {entry['verdict']}")
    print(f"overall: {data['hypotheses_verdicts']['overall']}")
    print(f"results written to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
