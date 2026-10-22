"""AR032/T004: three-arm speculative-advance-reuse benchmark (real ckpt).

Runs the REAL ``FastDLLMSampler.sample`` generation loop on the local
Fast-dLLM-v2-1.5B checkpoint in THREE arms over one shared model instance,
prompt set, and seed (the AR022/AR031 protocol carried over; the shared
helpers are IMPORTED from ``scripts.ar021_generation_loop_bench``; the
folded arms run the EAGER primary face -- the real CUDA-graph linkage face
is BLOCKED-DEFERRED, see ``ARTIFACT_LIMITATION_NOTES``):

* ``baseline`` -- the official Fast-dLLM v2 recipe (``folded_model=None``).
* ``reuse-off`` -- MFF + ``num_block_candidates=4`` +
  ``candidate_include_greedy=True`` (the AR022-aligned config) with
  ``speculative_advance_reuse=False``: every block-advance-eligible block
  runs the independent serial AR forward -- the attribution control.
* ``reuse-on`` -- the IDENTICAL construction with exactly one switch
  flipped, ``speculative_advance_reuse=True``: the block-advance token is
  the last small block's winning verification row tail argmax, so ONE
  serial forward is omitted per eligible block -- the mechanism arm.

Hypotheses (declared before measurement; the verdict functions below are
frozen BEFORE any data is collected -- pinned by UT-3178/UT-3179; honest
negatives are delivered verbatim, never rerun-until-pass):

* **H1** quality alignment (on vs baseline): the merged absolute posthoc
  overall-confidence gap <= 1e-3 (+1e-9 IEEE noise margin); per-prompt
  decode lengths on == off elementwise; merged block-grid exact-match
  on >= off - 1e-9 (slack) - 1e-9 (IEEE margin). Any ``None`` face value
  refutes that face alone.
* **H2** p50 speedup (baseline vs on): ``p50(baseline)/p50(on) > 1.0``
  strictly; the merged two-sample std is recorded (2-sigma is report-only
  -- the frozen gate is the strict ratio, per design draft-3).
* **H3** mechanism attribution (off vs on): ``p50(off)/p50(on) > 1.0``
  strictly AND the forward-count identity
  ``off_serial - on_serial == eligible_blocks`` (the count face alone can
  refute -- a speedup without the omitted-forward bookkeeping is not the
  mechanism).
* **H4** honest ledger: the mechanism-engagement canary (the on arm never
  mints a ``fastdllm_ar_*`` advance branch; eligible blocks > 0;
  per-prompt batched verification counts equal across off/on) AND the
  per-arm actual-event-count-driven publish ledger within the #45
  ``LEDGER_C`` bound AND the internal reproduction double-run strictly
  row-equal per ``(arm, prompt_index)`` (a mismatch refutes H4 and the
  overall verdict; H1-H3 keep their own recomputable verdicts).

Protocol: seed 2026, P=2 pre-padded prompts, num_tokens=128 (W=4) formal /
64 (W=2) quick; warmup one full discarded sample per (arm, prompt); the
untimed sidecar generation runs TWICE (pass A records the quality faces
and the forward-count sidecar, pass B is the reproduction self-check);
per-rep three-arm round-robin (rep r starts at r%3); host perf_counter +
cuda synchronize (#39); reps=30 pre-committed single-shot verdict.
``--quick``: reps=3, single prompt, num_tokens=64, separate artifact path,
never the formal one. The default formal artifact path is
overwrite-protected (an existing formal artifact is refused, the
run_formal_queue semantics).

Environment gate (construction time, before any model load or artifact
write): ``transformers==4.53.1`` (the AR019 venv) and a local checkpoint
directory containing ``config.json``; failures exit rc=2 and write
nothing. A mid-run arm failure (OOM etc.) crashes without writing any
artifact (zero fabricated numbers, the AR031 caliber).

Run (AR019 venv, transformers==4.53.1)::

    python -m scripts.ar032_speculative_advance_bench [--quick] [--reps 30]
        [--model-path PATH] [--out PATH] [--hf-home PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402

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

SCHEMA = "ar032_speculative_advance_bench"
SCHEMA_VERSION = 1
ARMS = ("baseline", "reuse-off", "reuse-on")
DEFAULT_OUTPUT_PATH = Path("results/optimization/ar032_speculative_advance_bench.json")
QUICK_OUTPUT_PATH = Path("results/optimization/ar032_speculative_advance_bench_quick.json")
REPS_DEFAULT = 30
QUICK_REPS = 3
REQUIRED_TRANSFORMERS = "4.53.1"
_H1_GAP_MAX = 1e-3
_H1_RATIO_EPS = 1e-9
_H3_EXACT_MATCH_SLACK = 1e-9
_BASELINE = "baseline"
_OFF = "reuse-off"
_ON = "reuse-on"
_AR_BRANCH_PREFIX = "fastdllm_ar_"

ARTIFACT_LIMITATION_NOTES: tuple[str, ...] = (
    "eager is the primary declaration face of this benchmark: the folded "
    "arms run ManualFoldedForward with use_cuda_graph=False (the WDDM "
    "16GB host protocol); the AR032 reuse mechanism reads the batched "
    "verification forward's return value and is exercised end-to-end on "
    "the eager path (D7: the graph linkage face is identical by "
    "construction -- the stash is written by the verification forward "
    "itself, never by a replay-side effect -- and is BLOCKED-DEFERRED as "
    "a measured claim, requiring a quiet >=24 GiB TCC/Linux GPU per the "
    "AR030 formal-face standard)",
    "timing follows the #39 wall-clock discipline (perf_counter + "
    "synchronize, warmup discarded, per-rep round-robin); the Windows WDDM "
    "display-driver mode does not lock clocks -- p50 over reps with "
    "interleaving is the noise mitigation",
    "the mechanism's headroom is bounded by the per-prompt "
    "eligible-block count: the AR031 artifact's posthoc evidence shows "
    "1-2 speculative blocks per prompt at these geometries, so the H3 "
    "delta is expected to be honest-but-modest; the count face and the "
    "engagement canary record the mechanism's activation regardless of "
    "the delta's size",
    "the forward-count sidecar reads MFF host-side counters differenced "
    "around one untimed sidecar generation per (arm, prompt); the MFF "
    "bookkeeping (counter increments, branch-name appends) also runs "
    "during the timed segment but is host-only and uniform across arms",
)

_HYPOTHESES: dict[str, str] = {
    "h1": (
        "quality alignment (on vs baseline): the merged absolute posthoc "
        "overall-confidence gap <= 1e-3 (+1e-9 IEEE margin); per-prompt "
        "decode lengths on == off elementwise; merged block-grid "
        "exact-match on >= off - 1e-9 (slack) - 1e-9 (IEEE margin); any "
        "None face value refutes that face alone"
    ),
    "h2": (
        "p50 speedup (baseline vs on): p50(baseline)/p50(on) > 1.0 "
        "strictly; the merged two-sample std is recorded (2-sigma is "
        "report-only, the frozen gate is the strict ratio)"
    ),
    "h3": (
        "mechanism attribution (off vs on): p50(off)/p50(on) > 1.0 "
        "strictly AND off_serial_forwards - on_serial_forwards == "
        "eligible_blocks (the count face alone can refute); per-arm "
        "stable ratios and per-prompt forward counts are report-only faces"
    ),
    "h4": (
        "honest ledger: the mechanism-engagement canary (on-arm "
        "fastdllm_ar_* advance branches == 0, eligible blocks > 0, "
        "per-prompt batched verification counts equal across off/on) AND "
        "the per-arm actual-event-count-driven publish ledger within the "
        "#45 LEDGER_C bound AND the internal reproduction double-run "
        "strictly row-equal per (arm, prompt_index) -- a mismatch "
        "refutes H4 and the overall verdict while H1-H3 keep their own "
        "recomputable verdicts"
    ),
}


# ---------------------------------------------------------------------------
# Frozen verdict functions (declared before any data collection; the UT
# suite pins every boundary; honest negatives are returned verbatim)
# ---------------------------------------------------------------------------
def verdict_h1(quality_faces: dict[str, Any]) -> dict[str, Any]:
    """H1 quality alignment: three faces ALL passing -> supported.

    Face (a) is the merged absolute posthoc gap of the on arm against the
    baseline arm (the fixed reference of the whole innovation line); face
    (b) is elementwise decode-length equality between the on and off arms
    (the attribution pair); face (c) is the merged block-grid exact-match
    of the on arm against baseline staying at or above the off arm's (the
    slack + IEEE margins are frozen by UT-3178).
    """
    on_mean = quality_faces.get("on_posthoc_mean")
    base_mean = quality_faces.get("baseline_posthoc_mean")
    on_lens = quality_faces.get("on_decode_lengths")
    off_lens = quality_faces.get("off_decode_lengths")
    on_em = quality_faces.get("on_exact_match")
    off_em = quality_faces.get("off_exact_match")

    gap: float | None = None
    face_a = False
    if (
        on_mean is not None
        and base_mean is not None
        and math.isfinite(on_mean)
        and math.isfinite(base_mean)
    ):
        gap = abs(on_mean - base_mean)
        face_a = gap <= _H1_GAP_MAX + _H1_RATIO_EPS

    face_b = bool(isinstance(on_lens, list) and isinstance(off_lens, list) and on_lens == off_lens)

    if on_em is not None and off_em is not None:
        face_c = bool(on_em >= off_em - _H3_EXACT_MATCH_SLACK - _H1_RATIO_EPS)
    else:
        face_c = False

    supported = bool(face_a and face_b and face_c)
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": {
            "a_posthoc_gap_le_1e-3": bool(face_a),
            "b_decode_lengths_equal": bool(face_b),
            "c_exact_match_ge_off": bool(face_c),
        },
        "numbers": {
            "on_posthoc_mean": on_mean,
            "baseline_posthoc_mean": base_mean,
            "on_decode_lengths": on_lens,
            "off_decode_lengths": off_lens,
            "on_exact_match": on_em,
            "off_exact_match": off_em,
            "posthoc_gap": gap,
        },
    }


def verdict_h2(
    baseline_ms: float | None, on_ms: float | None, sigma: float | None
) -> dict[str, Any]:
    """H2 p50 speedup: the strict ratio gate; sigma is report-only.

    The reported ``speedup`` is rounded to 4 decimals; ``speedup_raw`` is
    kept unrounded for recomputability.  None / non-positive / non-finite
    inputs refute (and never raise).
    """
    speedup_raw: float | None = None
    delta: float | None = None
    threshold: float | None = None
    if (
        baseline_ms is not None
        and on_ms is not None
        and sigma is not None
        and math.isfinite(baseline_ms)
        and math.isfinite(on_ms)
        and math.isfinite(sigma)
        and baseline_ms > 0.0
        and on_ms > 0.0
        and sigma >= 0.0
    ):
        speedup_raw = baseline_ms / on_ms
        delta = baseline_ms - on_ms
        threshold = 2.0 * sigma
    supported = speedup_raw is not None and speedup_raw > 1.0
    return {
        "verdict": "supported" if supported else "refuted",
        "numbers": {
            "baseline_ms": baseline_ms,
            "on_ms": on_ms,
            "sigma": sigma,
            "speedup": round(speedup_raw, 4) if speedup_raw is not None else None,
            "speedup_raw": speedup_raw,
            "delta_ms": delta,
            "threshold_2sigma_ms": threshold,
            "rule": "supported iff p50(baseline)/p50(on) > 1.0 strictly; 2-sigma report-only",
        },
    }


def verdict_h3(
    off_ms: float | None,
    on_ms: float | None,
    off_serial_forwards: int | None,
    on_serial_forwards: int | None,
    eligible_blocks: int | None,
) -> dict[str, Any]:
    """H3 mechanism attribution: strict speedup AND the forward-count face.

    The count face requires non-negative counts and the exact identity
    ``off_serial_forwards - on_serial_forwards == eligible_blocks`` -- one
    omitted serial advance forward per block-advance-eligible block.  The
    count face alone can refute (a speedup without the bookkeeping is not
    the mechanism).
    """
    speedup_raw: float | None = None
    if (
        off_ms is not None
        and on_ms is not None
        and math.isfinite(off_ms)
        and math.isfinite(on_ms)
        and off_ms > 0.0
        and on_ms > 0.0
    ):
        speedup_raw = off_ms / on_ms
    face_a = speedup_raw is not None and speedup_raw > 1.0

    face_b = False
    if (
        off_serial_forwards is not None
        and on_serial_forwards is not None
        and eligible_blocks is not None
        and off_serial_forwards >= 0
        and on_serial_forwards >= 0
        and eligible_blocks >= 0
        and (off_serial_forwards - on_serial_forwards) == eligible_blocks
    ):
        face_b = True

    supported = bool(face_a and face_b)
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": {
            "a_speedup_gt_1": bool(face_a),
            "b_mechanism_count": bool(face_b),
        },
        "numbers": {
            "off_ms": off_ms,
            "on_ms": on_ms,
            "off_serial_forwards": off_serial_forwards,
            "on_serial_forwards": on_serial_forwards,
            "eligible_blocks": eligible_blocks,
            "speedup": round(speedup_raw, 4) if speedup_raw is not None else None,
            "speedup_raw": speedup_raw,
            "rule": (
                "supported iff p50(off)/p50(on) > 1.0 strictly AND "
                "off_serial - on_serial == eligible_blocks (non-negative counts)"
            ),
        },
    }


def _payload_equal(first: Any, second: Any) -> bool:
    """Strict payload equality (tensors compare via ``torch.equal``).

    Bench payloads are JSON-comparable dicts (decode length + the full
    committed token list); the tensor branch is defensive only.
    """
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return bool(torch.equal(first, second))
    return bool(first == second)


def repro_mismatch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strict per-``(arm, prompt_index)`` double-run comparison rows.

    Each input row carries its OWN ``first`` / ``second`` payloads; a row
    mismatches iff its own payloads differ.  Reusing one shared ``first``
    dict across the collection loop contaminates every earlier row with
    the LAST prompt's cargo (the AR031 first-formal-run incident,
    UT-3166/UT-3179) -- the pairing key here is the row itself, so the
    leak shape surfaces as exactly the contaminated rows.  Payloads must
    be JSON-comparable (``_payload_equal``).
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


def verdict_h4(
    canary_pass: bool, ledger_within: bool, repro_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """H4 honest ledger: canary AND ledger AND an empty mismatch list."""
    supported = bool(canary_pass) and bool(ledger_within) and not repro_rows
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": {
            "canary_pass": bool(canary_pass),
            "ledger_within": bool(ledger_within),
            "reproduction_ok": not repro_rows,
        },
        "numbers": {
            "canary_pass": bool(canary_pass),
            "ledger_within": bool(ledger_within),
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
    """Pure verdict assembly; every verdict recomputes from the faces."""
    h1 = verdict_h1(faces["quality_faces"])
    h2 = verdict_h2(faces["baseline_ms"], faces["on_ms"], faces["sigma"])
    h3 = verdict_h3(
        faces["off_ms"],
        faces["on_ms"],
        faces["off_serial_forwards"],
        faces["on_serial_forwards"],
        faces["eligible_blocks"],
    )
    h4 = verdict_h4(faces["canary_pass"], faces["ledger_within"], faces["repro_rows"])
    return {
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "overall": overall_verdict(h1["verdict"], h2["verdict"], h3["verdict"], h4["verdict"]),
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
# Arm construction / local helper copies (the AR022 coupling-note pattern)
# ---------------------------------------------------------------------------
def _arm_sampler_config(num_tokens: int, reuse: bool) -> FastDLLMSamplerConfig:
    """The folded-arm config: the AR022-aligned base + the single AR032 switch."""
    return dataclasses.replace(
        _sampler_config(num_tokens, NUM_CANDIDATES),
        candidate_include_greedy=True,
        speculative_advance_reuse=reuse,
    )


def _make_eager_mff(model: Any) -> _BenchMFF:
    """Eager primary face: graph off, no budget policy (the AR022 caliber)."""
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
    )


def _ledger_delta(
    mff: _BenchMFF,
    ops0: int,
    s0: int,
    b0: int,
    num_layers: int,
) -> dict[str, Any]:
    """Publish-ledger delta over one sidecar generation (#45 caliber).

    Per-arm actual-event driven (the design draft-3 UT-3178 note): the
    bound is parameterized by THIS arm's measured serial/batched event
    counts, so the reuse-on arm's lower serial count shrinks its own bound
    instead of false-failing a fixed formula.
    """
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


def _merged_mean(values: list[Any]) -> float | None:
    """Merged mean; ANY None member propagates None (honest no-data)."""
    if not values or any(v is None for v in values):
        return None
    return sum(float(v) for v in values) / len(values)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run_bench(model_path: Path, reps: int, quick: bool) -> dict[str, Any]:
    """Run the three-arm speculative-advance-reuse benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("run_bench requires a CUDA device")

    num_tokens = QUICK_NUM_TOKENS if quick else FULL_NUM_TOKENS
    model, tokenizer, model_meta = _load_model(model_path)
    prompts = _prepare_prompts(tokenizer, quick)
    llm = _BenchDiffusionLLM(model, tokenizer)
    num_layers = llm.num_layers

    mff_off = _make_eager_mff(model)
    mff_on = _make_eager_mff(model)
    modes: dict[str, Any] = {
        _BASELINE: {
            "sampler": FastDLLMSampler(llm, config=_sampler_config(num_tokens, 0)),
            "mff": None,
        },
        _OFF: {
            "sampler": FastDLLMSampler(llm, config=_arm_sampler_config(num_tokens, reuse=False)),
            "mff": mff_off,
        },
        _ON: {
            "sampler": FastDLLMSampler(llm, config=_arm_sampler_config(num_tokens, reuse=True)),
            "mff": mff_on,
        },
    }

    def _generate(arm: str, prompt: dict[str, Any]) -> dict[str, Any]:
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
            record["ar_advance_forwards"] = sum(
                1 for bid in mff.branches_seen if str(bid).startswith(_AR_BRANCH_PREFIX)
            )
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

    # -- sidecar pass A (untimed quality faces + forward-count sidecar) ----
    per_prompt_rows: list[dict[str, Any]] = []
    sidecar_detail: list[dict[str, Any]] = []
    ledger_rows: dict[str, dict[str, Any]] = {}
    count_rows: list[dict[str, Any]] = []
    pass_a_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for pi, prompt in enumerate(prompts):
        padded_len = prompt["padded_len"]
        seqs: dict[str, torch.Tensor] = {}
        posthocs: dict[str, Any] = {}
        arm_counts: dict[str, dict[str, int]] = {}
        arm_ledgers: dict[str, Any] = {}
        for arm in ARMS:
            mff = modes[arm]["mff"]
            if mff is not None:
                ops0 = mff.cache.write_op_count
                s0, b0 = mff.serial_calls, mff.batched_calls
            rec = _generate(arm, prompt)
            seqs[arm] = rec["sequences"]
            posthocs[arm] = _posthoc_confidence(model, seqs[arm][0], padded_len, num_tokens)
            if mff is not None:
                arm_counts[arm] = {
                    "serial_forwards": mff.serial_calls - s0,
                    "batched_forwards": mff.batched_calls - b0,
                    "ar_advance_forwards": int(rec["ar_advance_forwards"]),
                }
                arm_ledgers[arm] = _ledger_delta(mff, ops0, s0, b0, num_layers)
        eligible = arm_counts[_OFF]["ar_advance_forwards"]
        count_rows.append(
            {
                "prompt_index": pi,
                "off": arm_counts[_OFF],
                "on": arm_counts[_ON],
                "eligible_blocks": eligible,
                "serial_delta_off_minus_on": (
                    arm_counts[_OFF]["serial_forwards"] - arm_counts[_ON]["serial_forwards"]
                ),
                "batched_equal": (
                    arm_counts[_OFF]["batched_forwards"] == arm_counts[_ON]["batched_forwards"]
                ),
                "on_ar_zero": arm_counts[_ON]["ar_advance_forwards"] == 0,
            }
        )
        for arm in ARMS:
            pass_a_payloads[(arm, pi)] = {
                "decode_length": int(seqs[arm].shape[1]),
                "tokens": seqs[arm][0].tolist(),
            }
        ledger_rows[f"prompt{pi}"] = arm_ledgers
        per_prompt_rows.append(
            {
                "prompt_index": pi,
                "posthoc_overall_mean": {
                    _BASELINE: posthocs[_BASELINE].get("overall_mean"),
                    _OFF: posthocs[_OFF].get("overall_mean"),
                    _ON: posthocs[_ON].get("overall_mean"),
                },
                "decode_length": {
                    _BASELINE: int(seqs[_BASELINE].shape[1]),
                    _OFF: int(seqs[_OFF].shape[1]),
                    _ON: int(seqs[_ON].shape[1]),
                },
                "block_match_counts": {
                    _OFF: _merged_block_exact_match(
                        seqs[_BASELINE], seqs[_OFF], padded_len, num_tokens
                    ),
                    _ON: _merged_block_exact_match(
                        seqs[_BASELINE], seqs[_ON], padded_len, num_tokens
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
                    _ON: _block_exact_match(seqs[_BASELINE], seqs[_ON], padded_len, num_tokens),
                },
            }
        )
        print(
            f"sidecar prompt {pi}: len(base/off/on) = "
            f"{seqs[_BASELINE].shape[1]}/{seqs[_OFF].shape[1]}/{seqs[_ON].shape[1]}, "
            f"eligible_blocks = {eligible}"
        )

    # -- sidecar pass B (the internal reproduction self-check) ------------
    repro_rows: list[dict[str, Any]] = []
    for pi, prompt in enumerate(prompts):
        for arm in ARMS:
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

    # -- timed reps: per-rep round-robin over the three arms ----------------
    whole_samples: dict[str, dict[int, list[float]]] = {
        arm: {pi: [] for pi in range(len(prompts))} for arm in ARMS
    }
    profiler_ratios: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for rep in range(reps):
        order = list(ARMS[rep % 3 :] + ARMS[: rep % 3])
        for arm in order:
            for pi, prompt in enumerate(prompts):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                rec = _generate(arm, prompt)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                whole_samples[arm][pi].append((t1 - t0) * 1000.0)
                if modes[arm]["mff"] is not None:
                    profiler_ratios[arm].extend(rec.get("stable_ratios", []))
        summary = " | ".join(f"{arm}: last={whole_samples[arm][0][-1]:.1f}ms" for arm in order)
        print(f"rep {rep + 1}/{reps}: {summary}", flush=True)

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
    std_h2 = (whole_stats[_BASELINE].std ** 2 + whole_stats[_ON].std ** 2) ** 0.5

    off_serial_total = sum(r["off"]["serial_forwards"] for r in count_rows)
    on_serial_total = sum(r["on"]["serial_forwards"] for r in count_rows)
    off_batched_total = sum(r["off"]["batched_forwards"] for r in count_rows)
    on_batched_total = sum(r["on"]["batched_forwards"] for r in count_rows)
    eligible_total = sum(r["eligible_blocks"] for r in count_rows)

    canary_faces = {
        "on_ar_advance_forwards_zero": all(bool(r["on_ar_zero"]) for r in count_rows),
        "eligible_blocks_positive": eligible_total > 0,
        "batched_structure_equal": all(bool(r["batched_equal"]) for r in count_rows),
    }
    canary_pass = all(bool(v) for v in canary_faces.values())

    ledger_within = bool(ledger_rows) and all(
        bool(entry[arm]["within_bound"]) for entry in ledger_rows.values() for arm in (_OFF, _ON)
    )

    matched_on = sum(r["block_match_counts"][_ON]["matched"] for r in per_prompt_rows)
    compared_on = sum(r["block_match_counts"][_ON]["compared"] for r in per_prompt_rows)
    matched_off = sum(r["block_match_counts"][_OFF]["matched"] for r in per_prompt_rows)
    compared_off = sum(r["block_match_counts"][_OFF]["compared"] for r in per_prompt_rows)
    on_exact = matched_on / compared_on if compared_on else None
    off_exact = matched_off / compared_off if compared_off else None

    quality_faces = {
        "on_posthoc_mean": _merged_mean([r["posthoc_overall_mean"][_ON] for r in per_prompt_rows]),
        "baseline_posthoc_mean": _merged_mean(
            [r["posthoc_overall_mean"][_BASELINE] for r in per_prompt_rows]
        ),
        "on_decode_lengths": [r["decode_length"][_ON] for r in per_prompt_rows],
        "off_decode_lengths": [r["decode_length"][_OFF] for r in per_prompt_rows],
        "on_exact_match": on_exact,
        "off_exact_match": off_exact,
    }

    faces = {
        "quality_faces": quality_faces,
        "baseline_ms": p50[_BASELINE],
        "on_ms": p50[_ON],
        "sigma": std_h2,
        "off_ms": p50[_OFF],
        "off_serial_forwards": off_serial_total,
        "on_serial_forwards": on_serial_total,
        "eligible_blocks": eligible_total,
        "canary_pass": canary_pass,
        "ledger_within": ledger_within,
        "repro_rows": repro_mismatches,
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
                    "sampler": {
                        "num_block_candidates": NUM_CANDIDATES,
                        "candidate_top_k": CANDIDATE_TOP_K,
                        "candidate_include_greedy": True,
                        "speculative_advance_reuse": False,
                    },
                    "description": (
                        "the AR022-aligned config -- the attribution "
                        "control; every block-advance-eligible block runs "
                        "the independent serial AR forward"
                    ),
                },
                _ON: {
                    "mff": "eager ManualFoldedForward (use_cuda_graph=False)",
                    "sampler": {
                        "num_block_candidates": NUM_CANDIDATES,
                        "candidate_top_k": CANDIDATE_TOP_K,
                        "candidate_include_greedy": True,
                        "speculative_advance_reuse": True,
                    },
                    "description": (
                        "the identical construction with exactly one "
                        "switch flipped -- the block-advance token is the "
                        "last small block's winning verification row tail "
                        "argmax; one serial forward omitted per eligible "
                        "block"
                    ),
                },
            },
            "warmup_strategy": (
                "one full discarded sample per (arm, prompt) before any "
                "timing; the untimed sidecar pass A records the quality "
                "faces and the forward-count sidecar, pass B is the "
                "internal reproduction self-check"
            ),
            "interleaving": (
                "per-rep round-robin over the three arms; rep r starts at "
                "position r%3 (anti-drift, WDDM fairness)"
            ),
            "timing_method": (
                "host perf_counter + torch.cuda.synchronize (#39); "
                "whole-generation wall clock; the quality sidecar (posthoc "
                "re-scoring, exact-match grids, ledger reads) runs only on "
                "the untimed passes; the MFF host-only bookkeeping is "
                "uniform across arms during the timed segment"
            ),
            "reproduction_declaration": (
                "the untimed sidecar generation runs TWICE per (arm, "
                "prompt); pass A records the quality faces and forward "
                "counts, and pass B must be strictly payload-equal on "
                "every (arm, prompt) row (decode length + the full "
                "committed token list); any mismatch refutes H4 and the "
                "overall verdict while H1-H3 keep their own recomputable "
                "verdicts; the check binds quick and formal alike"
            ),
            "mechanism_probe_semantics": (
                "per (arm, prompt) the MFF serial/batched call counters "
                "are differenced around one untimed sidecar generation; "
                "eligible_blocks := the reuse-off arm's count of "
                "fastdllm_ar_* serial advance branches (the reuse-on arm "
                "never mints them -- D5); the on arm's per-prompt batched "
                "count must equal the off arm's (identical speculative "
                "verification structure); the identity off_serial - "
                "on_serial == eligible_blocks is the H3 count face"
            ),
            "canary_note": (
                "the canary is the mechanism-engagement face (on-arm "
                "advance branches == 0; eligible blocks > 0; batched "
                "structure equal); the AR022 degradation-class graph "
                "canary is moot by construction on the eager primary face "
                "(use_cuda_graph=False) and is recorded as a limitation "
                "note, never as a measured claim"
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
                "python -m scripts.ar032_speculative_advance_bench"
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
            "on_vs_baseline": (round(p50[_BASELINE] / p50[_ON], 4) if p50[_ON] > 0 else None),
            "on_vs_off": round(p50[_OFF] / p50[_ON], 4) if p50[_ON] > 0 else None,
            "off_vs_baseline": (round(p50[_BASELINE] / p50[_OFF], 4) if p50[_OFF] > 0 else None),
        },
        "quality": {
            "faces": quality_faces,
            "per_prompt": per_prompt_rows,
            "per_arm_sidecar": sidecar_detail,
        },
        "mechanism_probe": {
            "per_prompt": count_rows,
            "totals": {
                "off_serial_forwards": off_serial_total,
                "on_serial_forwards": on_serial_total,
                "off_batched_forwards": off_batched_total,
                "on_batched_forwards": on_batched_total,
                "eligible_blocks": eligible_total,
            },
        },
        "canary": {
            "faces": {k: bool(v) for k, v in canary_faces.items()},
            "pass": bool(canary_pass),
        },
        "ledger": {
            "c": LEDGER_C,
            "num_layers": num_layers,
            "per_prompt": ledger_rows,
            "within_bound": {
                arm: bool(
                    all(
                        ledger_rows[f"prompt{pi}"][arm]["within_bound"]
                        for pi in range(len(prompts))
                    )
                )
                for arm in (_OFF, _ON)
            },
        },
        "reproduction": {
            "mode": "internal-double-run",
            "ok": not repro_mismatches,
            "rows_compared": len(prompts) * len(ARMS),
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
        print(
            f"refusing to overwrite the existing formal artifact {out} "
            "(pre-committed single-shot discipline; pass an explicit "
            "--out for a deliberate rerun)",
            file=sys.stderr,
        )
        return 4

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
