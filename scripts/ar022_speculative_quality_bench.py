"""AR022/T003: four-mode speculative quality-alignment benchmark (real ckpt).

Runs the REAL ``FastDLLMSampler.sample`` generation loop on the local
Fast-dLLM-v2-1.5B checkpoint in FOUR modes over one shared model instance,
prompt set, and seed (AR021 protocol fully carried over; helpers are
IMPORTED from ``scripts.ar021_generation_loop_bench`` -- design 4.1.4
option A, zero duplication):

* ``baseline`` -- the raw sampler (``folded_model=None``).
* ``folded-serial`` -- MFF_A (``adaptive=False``): FR2 chain folding.
* ``folded-batched`` -- MFF_B (``adaptive=True``, N=4/top_k=10/greedy off):
  FR3 batched speculative verification; its sidecar quality face is the
  AR021 reproduction self-check (same seed must strictly reproduce the
  AR021 artifact priors).
* ``folded-batched-greedy`` -- the AR022 alignment mode: MFF_C, an
  INDEPENDENT instance constructed identically to MFF_B (own cache/gate/
  pool/canary/AR017 policy), plus sampler
  ``candidate_include_greedy=True`` (row 0 = greedy hypothesis). Same
  ``[N, T]`` batched shape/N/top-k/prompts/seed as folded-batched: the
  speed variable is constructively zeroed (same-shape A/B).

Import couplings (design 4.1.4, recorded honestly): (a)
``_serialize_windows`` reads the AR021 module-level ``_snapshot_events``
dict -- this script assigns that global before the timed segment (safe:
keys are instance ids); (b) AR021's ``_ctx_mask_map`` hardcodes its two
folded mode names -- a local three-folded-mode copy lives here; (c)
importing the AR021 module executes its module-level ``apply_hf_env()``
and heavy torch/transformers imports -- idempotent no-op after ours.

Hypotheses (declared before measurement; verdicts from data only; frozen
wording from srs 3.4):

* **H1** quality alignment (hard gate): the aligned mode's committed-token
  quality aligns with baseline -- three faces ALL passing: (a) merged
  position-level agreement rate > 0.5; (b) merged block-level exact-match
  rate > the folded-batched same-criterion MEASURED aggregate this run;
  (c) per prompt aligned posthoc overall_mean > the folded-batched
  same-prompt MEASURED value. supported iff all three pass; else refuted
  (deterministic output, no inconclusive).
* **H2** speed preservation (hard gate): sigma = max(sigma_greedy,
  sigma_batched); supported iff p50(batched) - p50(greedy) >= -2*sigma;
  else refuted (directional no-regression gate, no inconclusive).
* **H3** early-stop recovery: supported iff BOTH prompts' aligned decode
  lengths == the baseline same-prompt MEASURED lengths; else refuted.
* **H4** full-stack exercise: (i) every aligned-mode window with a batched
  call has batched captures > 0 AND timed replays > 0 (AR021 H4 criterion
  on MFF_C); (ii) degradation-class canary timed == 0, all three folded
  modes; (iii) publish ledger <= 4*(L+N), three folded modes per prompt;
  (iv) AR017 events verified only when triggered (untriggered = neutral;
  MFF_B and MFF_C judged independently). Otherwise refuted.

Reproduction self-check: the folded-batched sidecar face is compared for
STRICT equality against the AR021 artifact priors. On mismatch (full run)
the artifact records ``reproduction_anomaly`` and ALL FOUR verdicts are
set to ``blocked`` (never silently substituted, never fabricated). Under
``--quick`` the face degrades to a structural validation (single prompt /
num_tokens=64 cannot reproduce the full-scale priors) -- recorded
honestly; the strict gate binds the full T004 run.

Protocol: seed 2026, P=2 prompts (pre-pad 32/64), num_tokens=128 (W=4);
warmup one full discarded sample per (mode, prompt); per-rep four-mode
round-robin (rep r starts at r%4); host perf_counter + cuda synchronize
(#39); per-block anchors + whole-generation wall clock; pool snapshot
diffs; reps=30 pre-committed single-shot verdict. ``--quick``: reps=3,
single prompt, num_tokens=64 (W=2), separate artifact path, <= 5 min.

Run (AR019 venv, transformers==4.53.1)::

    python -m scripts.ar022_speculative_quality_bench [--quick] [--reps 30]
        [--model-path PATH] [--out PATH] [--hf-home PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402

import scripts.ar021_generation_loop_bench as _ar021  # noqa: E402
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig  # noqa: E402
from scripts.algo_experiments import stats_from_samples  # noqa: E402
from scripts.ar021_generation_loop_bench import (  # noqa: E402
    BLOCK,
    CACHE_BUDGET,
    CANDIDATE_TOP_K,
    CAPACITY_RATIO,
    EPS,
    FULL_NUM_TOKENS,
    LEDGER_C,
    NUM_CANDIDATES,
    POOL_SIZE,
    QUICK_NUM_TOKENS,
    SEED,
    TAU,
    _BenchDiffusionLLM,
    _BenchMFF,
    _block_exact_match,
    _compare_logits_slices,
    _cuda_profiler_supported,
    _generate,
    _gpu_driver_version,
    _load_model,
    _make_mff,
    _posthoc_confidence,
    _prepare_prompts,
    _resolve_model_path,
    _sampler_config,
    _serialize_windows,
    _snapshot_pool,
    _token_agreement,
    _window_forwards,
    write_results,
)

DEFAULT_OUTPUT_PATH = Path("results/optimization/ar022_speculative_quality_bench.json")
QUICK_OUTPUT_PATH = Path("results/optimization/ar022_speculative_quality_bench_quick.json")

_MODES = ("baseline", "folded-serial", "folded-batched", "folded-batched-greedy")
_FOLDED_MODES = ("folded-serial", "folded-batched", "folded-batched-greedy")

# AR021 artifact priors (results/optimization/ar021_generation_loop_bench.json,
# quality.h3_batched_vs_baseline[*]): the folded-batched sidecar face must
# strictly reproduce these on a full same-environment rerun (design 4.2.2(5)).
_AR021_PRIORS: dict[str, dict[str, Any]] = {
    "prompt0": {
        "block_exact_match": 0.0,
        "len_other": 160,
        "posthoc_mean": 0.289692,
        "posthoc_min": 5e-06,
    },
    "prompt1": {
        "block_exact_match": 0.0,
        "len_other": 77,
        "posthoc_mean": 0.106418,
        "posthoc_min": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Aligned-mode config / local helper copies (design 4.1.4 coupling notes)
# ---------------------------------------------------------------------------
def _ar022_sampler_config(num_tokens: int) -> FastDLLMSamplerConfig:
    """Aligned-mode sampler config: greedy anchoring over the frozen batched one.

    ``dataclasses.replace`` keeps every frozen constant (seed, block sizes,
    threshold, temperature, N, top-k) identical to folded-batched -- the
    single-knob difference isolates the quality variable (same-shape A/B).
    """
    config: FastDLLMSamplerConfig = dataclasses.replace(
        _sampler_config(num_tokens, NUM_CANDIDATES),
        candidate_include_greedy=True,
    )
    return config


def _ctx_mask_map(samplers: dict[str, FastDLLMSampler]) -> dict[int, tuple[int, int]]:
    """Local three-folded-mode copy of the AR021 helper (coupling note (b)).

    The AR021 original hardcodes its two folded mode names; this copy is
    identical except for the mode tuple (design 4.1.4).
    """
    out: dict[int, tuple[int, int]] = {}
    for mode in _FOLDED_MODES:
        cache = getattr(samplers[mode], "_fold_ctx_cache", None) or {}
        for key, ctx in cache.items():
            out[id(ctx.mask)] = (int(key[0]), int(key[1]))
    return out


# ---------------------------------------------------------------------------
# Dual-granularity quality faces (design 4.2.2(5))
# ---------------------------------------------------------------------------
def _position_match(
    base_seq: torch.Tensor,
    other_seq: torch.Tensor,
    padded_len: int,
) -> dict[str, Any]:
    """Position-level token agreement over the generated region (H1(a) face).

    Comparison set = ``[padded_len, min(len_baseline, len_aligned))``: both
    rows hold real generated tokens across that interval (baseline unmasks
    at least one position per iteration so the canvas mask clears within
    ``max_iters = 2*S``; speculative candidates fill every still-masked
    position; the post-stop pad tail is trimmed by ``decode_final`` --
    the AR021 ``_block_exact_match`` assumption family). ``compared == 0``
    -> rate ``None`` (that prompt is excluded from the merged aggregation).
    """
    b = base_seq[0].tolist()
    c = other_seq[0].tolist()
    end = min(len(b), len(c))
    compared = max(0, end - padded_len)
    matched = sum(1 for i in range(padded_len, end) if b[i] == c[i]) if compared else 0
    return {
        "matched": matched,
        "compared": compared,
        "rate": round(matched / compared, 6) if compared else None,
        "len_baseline": len(b),
        "len_other": len(c),
        "comparison_window": f"[{padded_len}, {end})",
    }


def _merged_block_exact_match(
    base_seq: torch.Tensor,
    other_seq: torch.Tensor,
    padded_len: int,
    num_tokens: int,
) -> dict[str, Any]:
    """Raw matched/compared counts on the AR021 block grid (for merged sums).

    Same grid and slicing as AR021 ``_block_exact_match`` (block_size ==
    small_block_size == 32 frozen: the aligned mode's own small-block
    partition coincides with the canvas block grid, satisfying the frozen
    H1(b) "small-block partition per the aligned mode" criterion); returns
    counts so callers can merge prompts as sum(matched)/sum(compared).
    """
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
# H1-H4 verdict functions (frozen formulas, srs 3.4 -> code, verbatim)
# ---------------------------------------------------------------------------
def _h1_verdict(quality_faces: dict[str, Any]) -> dict[str, Any]:
    """H1 quality alignment: three faces ALL passing -> supported (no inconclusive).

    (a) merged position rate > 0.5; (b) merged block rate > the folded-batched
    same-criterion MEASURED aggregate this run; (c) per prompt aligned posthoc
    overall_mean > the folded-batched same-prompt MEASURED value this run.
    """
    merged_position = quality_faces["merged_position"]
    merged_block = quality_faces["merged_block"]
    aligned = quality_faces["aligned_entries"]
    batched = quality_faces["batched_entries"]

    rate_pos = merged_position["rate"]
    face_a = rate_pos is not None and rate_pos > 0.5
    rate_aln = merged_block["folded-batched-greedy"]["rate"]
    rate_bat = merged_block["folded-batched"]["rate"]
    face_b = rate_aln is not None and rate_bat is not None and rate_aln > rate_bat

    per_prompt_posthoc: list[dict[str, Any]] = []
    face_c = True
    for a, b in zip(aligned, batched):
        am = a["posthoc_confidence"]["folded-batched-greedy"]["overall_mean"]
        bm = b["posthoc_confidence"]["folded-batched"]["overall_mean"]
        base_m = a["posthoc_confidence"]["baseline"]["overall_mean"]
        ok = am is not None and bm is not None and am > bm
        face_c = face_c and ok
        per_prompt_posthoc.append(
            {
                "prompt_index": a["prompt_index"],
                "aligned_overall_mean": am,
                "batched_overall_mean": bm,
                "baseline_overall_mean": base_m,
                "aligned_vs_baseline_gap": (
                    round(am - base_m, 6) if am is not None and base_m is not None else None
                ),
                "pass": bool(ok),
            }
        )

    supported = bool(face_a and face_b and face_c)
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": {
            "a_merged_position_rate_gt_0_5": bool(face_a),
            "b_merged_block_rate_gt_batched_measured": bool(face_b),
            "c_per_prompt_posthoc_gt_batched_measured": bool(face_c),
        },
        "numbers": {
            "merged_position_match": merged_position,
            "merged_block_match": merged_block,
            "per_prompt_posthoc": per_prompt_posthoc,
            "per_prompt_position_match": [e["position_match"] for e in aligned],
            "per_prompt_compared_counts": [e["position_match"]["compared"] for e in aligned],
            "greedy_win": quality_faces["greedy_win_total"],
        },
    }


def _h2_verdict(stats_greedy: Any, stats_batched: Any) -> dict[str, Any]:
    """H2 speed preservation: directional no-regression gate, no inconclusive.

    sigma = max(sigma_greedy, sigma_batched); supported iff
    ``p50(batched) - p50(greedy) >= -2*sigma`` (acceleration and parity both
    pass); else refuted. vs-baseline speedup is a report face.
    """
    sigma = max(stats_greedy.std, stats_batched.std)
    delta = stats_batched.p50 - stats_greedy.p50
    supported = delta >= -2 * sigma
    return {
        "verdict": "supported" if supported else "refuted",
        "numbers": {
            "p50_folded_batched_ms": round(stats_batched.p50, 4),
            "p50_folded_batched_greedy_ms": round(stats_greedy.p50, 4),
            "delta_batched_minus_greedy_ms": round(delta, 4),
            "sigma_ms": round(sigma, 4),
            "threshold_minus_2sigma_ms": round(-2 * sigma, 4),
            "sigma_rule": "sigma = max(sigma_greedy, sigma_batched)",
            "rule": "supported iff p50(batched) - p50(greedy) >= -2*sigma",
        },
    }


def _h3_verdict(
    lengths_greedy: list[int],
    lengths_baseline: list[int],
    lengths_batched: list[int],
) -> dict[str, Any]:
    """H3 early-stop recovery: aligned decode lengths == baseline MEASURED ones."""
    per_prompt: list[dict[str, Any]] = []
    all_equal = bool(lengths_greedy)
    for i, (g, b) in enumerate(zip(lengths_greedy, lengths_baseline)):
        ok = g == b
        all_equal = all_equal and ok
        per_prompt.append(
            {
                "prompt_index": i,
                "len_aligned": g,
                "len_baseline_measured": b,
                "len_folded_batched_measured": (
                    lengths_batched[i] if i < len(lengths_batched) else None
                ),
                "matches_baseline": bool(ok),
            }
        )
    return {
        "verdict": "supported" if all_equal else "refuted",
        "numbers": {"per_prompt_decode_lengths": per_prompt},
    }


def _h4_verdict(
    windows_greedy: dict[str, Any],
    canaries: dict[str, bool],
    ledgers: dict[str, bool],
    ar017: dict[str, Any],
) -> dict[str, Any]:
    """H4 full-stack exercise: four faces, AR021 criterion applied to MFF_C.

    (i) every aligned-mode window with a batched call has batched captures
    > 0 AND timed replays > 0; (ii) degradation canary timed == 0 for all
    three folded modes; (iii) ledger within c=4 bound, three folded modes;
    (iv) AR017 events verified only when triggered (untriggered = neutral;
    the policy records only successful re-captures, so a positive count
    means the trigger condition actually fired -- never refutes by itself).
    """
    scope = windows_greedy["scope"]
    pool = windows_greedy["windows"]["batched"]["windows"]
    capture_ok = bool(scope) and all(
        pool.get(label, {}).get("captures_cumulative", 0) > 0 for label in scope
    )
    replay_ok = bool(scope) and all(
        pool.get(label, {}).get("replays_timed", 0) > 0 for label in scope
    )
    canary_ok = all(bool(v) for v in canaries.values())
    ledger_ok = all(bool(v) for v in ledgers.values())
    supported = capture_ok and replay_ok and canary_ok and ledger_ok
    checks = {
        "aligned_batched_scope_windows": scope,
        "aligned_batched_capture_positive": bool(capture_ok),
        "aligned_batched_timed_replay_positive": bool(replay_ok),
        "canary_degradation_timed_zero": {k: bool(v) for k, v in canaries.items()},
        "ledger_within_bound": {k: bool(v) for k, v in ledgers.items()},
        "ar017_events": ar017,
    }
    return {
        "verdict": "supported" if supported else "refuted",
        "numbers": checks,
    }


# ---------------------------------------------------------------------------
# Sidecar ledger helper (AR021 formula, c=4 frozen)
# ---------------------------------------------------------------------------
def _ledger_delta(
    mff: _BenchMFF,
    ops0: int,
    s0: int,
    b0: int,
    num_layers: int,
) -> dict[str, Any]:
    """Publish-ledger delta over one sidecar generation (AR021 formula)."""
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


def _adaptive_event_count(mff: _BenchMFF) -> int:
    """Current AR017 adaptive-policy event count (0 when the feature is off)."""
    policy = mff.graph_adaptive_policy
    return 0 if policy is None else len(policy.events)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run_bench(model_path: Path, reps: int, quick: bool) -> dict[str, Any]:
    """Run the four-mode speculative quality-alignment benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("run_bench requires a CUDA device")

    num_tokens = QUICK_NUM_TOKENS if quick else FULL_NUM_TOKENS
    model, tokenizer, model_meta = _load_model(model_path)
    prompts = _prepare_prompts(tokenizer, quick)
    llm = _BenchDiffusionLLM(model, tokenizer)
    mff_serial = _make_mff(model, adaptive=False)
    mff_batched = _make_mff(model, adaptive=True)
    # MFF_C: constructed identically to MFF_B but as an INDEPENDENT instance
    # (own cache/gate/pool/canary/AR017 policy -- per-mode isolation, H4 face).
    mff_greedy = _make_mff(model, adaptive=True)
    samplers = {
        "baseline": FastDLLMSampler(llm, config=_sampler_config(num_tokens, 0)),
        "folded-serial": FastDLLMSampler(llm, config=_sampler_config(num_tokens, 0)),
        "folded-batched": FastDLLMSampler(llm, config=_sampler_config(num_tokens, NUM_CANDIDATES)),
        "folded-batched-greedy": FastDLLMSampler(llm, config=_ar022_sampler_config(num_tokens)),
    }
    modes: dict[str, Any] = {
        "baseline": {"sampler": samplers["baseline"], "mff": None},
        "folded-serial": {"sampler": samplers["folded-serial"], "mff": mff_serial},
        "folded-batched": {"sampler": samplers["folded-batched"], "mff": mff_batched},
        "folded-batched-greedy": {
            "sampler": samplers["folded-batched-greedy"],
            "mff": mff_greedy,
        },
    }
    num_layers = llm.num_layers

    # -- warmup: one full discarded sample per (mode, prompt) --------------
    for mode_name in _MODES:
        for prompt in prompts:
            _generate(mode_name, prompt, modes, llm)
        print(f"warmup done: {mode_name}")

    # -- sidecar (untimed, deterministic single-shot quality faces) --------
    sidecar: dict[str, Any] = {
        "h1": [],
        "batched_quality": [],
        "aligned_quality": [],
        "ledger": {},
    }
    window_counts: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in _FOLDED_MODES}
    lengths: dict[str, list[int]] = {
        "baseline": [],
        "folded-batched": [],
        "folded-batched-greedy": [],
    }
    for pi, prompt in enumerate(prompts):
        padded_len = prompt["padded_len"]

        # baseline (recorded; H1-serial reference + quality reference)
        llm.recording = True
        llm.records = []
        base_rec = _generate("baseline", prompt, modes, llm)
        llm.recording = False
        base_records, llm.records = llm.records, []
        base_seq = base_rec["sequences"]
        lengths["baseline"].append(int(base_seq.shape[1]))
        base_posthoc = _posthoc_confidence(model, base_seq[0], padded_len, num_tokens)

        # folded-serial (H1-serial face: AR021 argmax + logits-allclose carried over)
        ops0 = mff_serial.cache.write_op_count
        s0, b0 = mff_serial.serial_calls, mff_serial.batched_calls
        mff_serial.recording = True
        mff_serial.records = []
        ser_rec = _generate("folded-serial", prompt, modes, llm)
        mff_serial.recording = False
        ser_records, mff_serial.records = mff_serial.records, []
        ledger_serial = _ledger_delta(mff_serial, ops0, s0, b0, num_layers)
        token_ok, token_rate = _token_agreement(base_seq, ser_rec["sequences"])
        logits_cmp = _compare_logits_slices(base_records, ser_records)
        sidecar["h1"].append(
            {
                "prompt_index": pi,
                "token_match": token_ok,
                "token_match_rate": round(token_rate, 6),
                "forward_counts_match": bool(
                    logits_cmp["ref_forwards"] == logits_cmp["live_forwards"]
                ),
                "logits": logits_cmp,
            }
        )
        window_counts["folded-serial"][pi] = _window_forwards(ser_rec["window_calls"])

        # folded-batched (AR021 h3 face re-measure = reproduction self-check face)
        ops0 = mff_batched.cache.write_op_count
        s0, b0 = mff_batched.serial_calls, mff_batched.batched_calls
        bat_rec = _generate("folded-batched", prompt, modes, llm)
        ledger_batched = _ledger_delta(mff_batched, ops0, s0, b0, num_layers)
        bat_seq = bat_rec["sequences"]
        lengths["folded-batched"].append(int(bat_seq.shape[1]))
        exact_batched = _block_exact_match(base_seq, bat_seq, padded_len, num_tokens)
        sidecar["batched_quality"].append(
            {
                "prompt_index": pi,
                "block_exact_match": exact_batched,
                "block_match_counts": _merged_block_exact_match(
                    base_seq, bat_seq, padded_len, num_tokens
                ),
                "posthoc_confidence": {
                    "baseline": base_posthoc,
                    "folded-batched": _posthoc_confidence(
                        model, bat_seq[0], padded_len, num_tokens
                    ),
                },
            }
        )
        window_counts["folded-batched"][pi] = _window_forwards(bat_rec["window_calls"])

        # aligned mode (greedy_win snapshot diff + dual-granularity faces)
        ops0 = mff_greedy.cache.write_op_count
        s0, b0 = mff_greedy.serial_calls, mff_greedy.batched_calls
        gw0 = samplers["folded-batched-greedy"].greedy_win_stats
        aln_rec = _generate("folded-batched-greedy", prompt, modes, llm)
        gw1 = samplers["folded-batched-greedy"].greedy_win_stats
        ledger_greedy = _ledger_delta(mff_greedy, ops0, s0, b0, num_layers)
        aln_seq = aln_rec["sequences"]
        lengths["folded-batched-greedy"].append(int(aln_seq.shape[1]))
        wins = int(gw1["greedy_wins"] - gw0["greedy_wins"])
        calls = int(gw1["speculative_calls"] - gw0["speculative_calls"])
        exact_aln = _block_exact_match(base_seq, aln_seq, padded_len, num_tokens)
        posm = _position_match(base_seq, aln_seq, padded_len)
        sidecar["aligned_quality"].append(
            {
                "prompt_index": pi,
                "block_exact_match": exact_aln,
                "block_match_counts": _merged_block_exact_match(
                    base_seq, aln_seq, padded_len, num_tokens
                ),
                "position_match": posm,
                "posthoc_confidence": {
                    "baseline": base_posthoc,
                    "folded-batched-greedy": _posthoc_confidence(
                        model, aln_seq[0], padded_len, num_tokens
                    ),
                },
                "len_baseline": int(base_seq.shape[1]),
                "len_aligned": int(aln_seq.shape[1]),
                "greedy_win": {
                    "wins": wins,
                    "calls": calls,
                    "rate": round(wins / calls, 6) if calls else None,
                },
            }
        )
        sidecar["ledger"][f"prompt{pi}"] = {
            "folded-serial": ledger_serial,
            "folded-batched": ledger_batched,
            "folded-batched-greedy": ledger_greedy,
        }
        window_counts["folded-batched-greedy"][pi] = _window_forwards(aln_rec["window_calls"])
        print(
            f"sidecar prompt {pi}: serial token_match={token_ok} "
            f"({logits_cmp['ref_forwards']} vs "
            f"{logits_cmp['live_forwards']} forwards), "
            f"logits allclose_all={logits_cmp['allclose_all']}, "
            f"batched exact={exact_batched['overall_exact_match']} "
            f"(len {exact_batched['len_other']}), "
            f"aligned exact={exact_aln['overall_exact_match']} "
            f"position_rate={posm['rate']} greedy_win={wins}/{calls}"
        )

    sidecar_ledger_all = {
        mode: bool(
            all(
                sidecar["ledger"][f"prompt{pi}"][mode]["within_bound"] for pi in range(len(prompts))
            )
        )
        for mode in _FOLDED_MODES
    }

    # -- merged quality aggregations (sum(matched)/sum(compared)) ----------
    merged_position = {
        "matched": sum(e["position_match"]["matched"] for e in sidecar["aligned_quality"]),
        "compared": sum(e["position_match"]["compared"] for e in sidecar["aligned_quality"]),
    }
    merged_position["rate"] = (
        round(merged_position["matched"] / merged_position["compared"], 6)
        if merged_position["compared"]
        else None
    )
    merged_block: dict[str, Any] = {}
    for key, entries in (
        ("folded-batched-greedy", sidecar["aligned_quality"]),
        ("folded-batched", sidecar["batched_quality"]),
    ):
        matched = sum(e["block_match_counts"]["matched"] for e in entries)
        compared = sum(e["block_match_counts"]["compared"] for e in entries)
        merged_block[key] = {
            "matched": matched,
            "compared": compared,
            "rate": round(matched / compared, 6) if compared else None,
        }
    greedy_win_total = {
        "wins": sum(e["greedy_win"]["wins"] for e in sidecar["aligned_quality"]),
        "calls": sum(e["greedy_win"]["calls"] for e in sidecar["aligned_quality"]),
    }
    greedy_win_total["rate"] = (
        round(greedy_win_total["wins"] / greedy_win_total["calls"], 6)
        if greedy_win_total["calls"]
        else None
    )

    # -- reproduction self-check (strict equality vs AR021 priors) ---------
    reproduction_actual = {
        f"prompt{e['prompt_index']}": {
            "block_exact_match": e["block_exact_match"]["overall_exact_match"],
            "len_other": e["block_exact_match"]["len_other"],
            "posthoc_mean": e["posthoc_confidence"]["folded-batched"]["overall_mean"],
            "posthoc_min": e["posthoc_confidence"]["folded-batched"]["overall_min"],
        }
        for e in sidecar["batched_quality"]
    }
    field_matches: dict[str, dict[str, bool]] = {}
    for pkey, actual in reproduction_actual.items():
        expected = _AR021_PRIORS.get(pkey)
        field_matches[pkey] = (
            {k: bool(expected[k] == actual[k]) for k in expected} if expected is not None else {}
        )
    strict_pass = bool(field_matches) and all(
        matches and all(matches.values()) for matches in field_matches.values()
    )
    if quick:
        structural_ok = bool(reproduction_actual) and all(
            v is not None for a in reproduction_actual.values() for v in a.values()
        )
        reproduction_selfcheck: dict[str, Any] = {
            "mode": "quick_structural",
            "gating": False,
            "expected": _AR021_PRIORS,
            "actual": reproduction_actual,
            "field_matches": field_matches,
            "strict_equality_pass": bool(strict_pass),
            "pass": bool(structural_ok),
            "note": (
                "--quick smoke uses a single prompt and num_tokens=64 (W=2) "
                "while the AR021 priors were recorded at full scale (two "
                "prompts, num_tokens=128); strict prior equality is only "
                "judgeable on the full T004 run -- here the face validates "
                "the measurement machinery structurally (deterministic "
                "single-shot sidecar, all fields present and non-null)"
            ),
        }
        reproduction_anomaly: dict[str, Any] | None = None
    else:
        strict_pass = strict_pass and set(_AR021_PRIORS) == set(reproduction_actual)
        reproduction_selfcheck = {
            "mode": "strict",
            "gating": True,
            "expected": _AR021_PRIORS,
            "actual": reproduction_actual,
            "field_matches": field_matches,
            "strict_equality_pass": bool(strict_pass),
            "pass": bool(strict_pass),
        }
        reproduction_anomaly = None
        if not strict_pass:
            mismatches = [
                {
                    "prompt": p,
                    "field": f,
                    "expected": _AR021_PRIORS[p][f],
                    "actual": reproduction_actual[p][f],
                }
                for p in sorted(field_matches)
                for f, ok in field_matches[p].items()
                if not ok
            ]
            reproduction_anomaly = {
                "expected": _AR021_PRIORS,
                "actual": reproduction_actual,
                "field_mismatches": mismatches,
                "blocked_scope": ["H1", "H2", "H3", "H4"],
                "reason": (
                    "same-environment determinism assumption failed "
                    "wholesale (same seed, different results); no "
                    "single-face verdict is trustworthy, so ALL FOUR H "
                    "verdicts are blocked (design 4.2.2(5)); priors are "
                    "never silently substituted and verdicts are never "
                    "fabricated"
                ),
            }

    # -- snapshots for timed-segment diffs ----------------------------------
    # Coupling note (a): _serialize_windows reads the AR021 module-level
    # _snapshot_events dict (keyed by id(mff)) -- assign it here so the
    # AR017 timed diffs resolve for all three folded modes.
    ctx_map = _ctx_mask_map(samplers)
    snap_serial = _snapshot_pool(mff_serial)
    snap_batched = _snapshot_pool(mff_batched)
    snap_greedy = _snapshot_pool(mff_greedy)
    _ar021._snapshot_events = {
        id(mff_serial): _adaptive_event_count(mff_serial),
        id(mff_batched): _adaptive_event_count(mff_batched),
        id(mff_greedy): _adaptive_event_count(mff_greedy),
    }
    timed_ops_base = {
        "folded-serial": mff_serial.cache.write_op_count,
        "folded-batched": mff_batched.cache.write_op_count,
        "folded-batched-greedy": mff_greedy.cache.write_op_count,
    }

    # -- timed reps: per-rep round-robin over the four modes ----------------
    whole_samples: dict[str, dict[int, list[float]]] = {
        m: {pi: [] for pi in range(len(prompts))} for m in _MODES
    }
    block_samples: dict[str, dict[int, dict[int, list[float]]]] = {
        m: {pi: {} for pi in range(len(prompts))} for m in _MODES
    }
    profiler_ratios: dict[str, list[float]] = {m: [] for m in _MODES}
    peaks: dict[str, list[float]] = {m: [] for m in _MODES}
    timed_window_calls: dict[str, list[tuple[int, int, str]]] = {m: [] for m in _FOLDED_MODES}
    for rep in range(reps):
        order = _MODES[rep % 4 :] + _MODES[: rep % 4]
        for mode_name in order:
            mff = modes[mode_name]["mff"]
            torch.cuda.reset_peak_memory_stats()
            for pi, prompt in enumerate(prompts):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                rec = _generate(mode_name, prompt, modes, llm)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                whole_samples[mode_name][pi].append((t1 - t0) * 1000.0)
                for blk, ms in rec["blocks"]:
                    block_samples[mode_name][pi].setdefault(blk, []).append(ms)
                if mff is not None:
                    profiler_ratios[mode_name].extend(rec["stable_ratios"])
                    if mode_name in timed_window_calls:
                        timed_window_calls[mode_name].extend(rec["window_calls"])
            peaks[mode_name].append(round(torch.cuda.max_memory_allocated() / 1024**2, 1))
        summary = " | ".join(f"{m}: last={whole_samples[m][0][-1]:.1f}ms" for m in order)
        print(f"rep {rep + 1}/{reps}: {summary}")

    # -- statistics ----------------------------------------------------------
    def _pooled(mode: str) -> list[float]:
        return [x for pi in range(len(prompts)) for x in whole_samples[mode][pi]]

    whole_stats = {m: stats_from_samples(_pooled(m)) for m in _MODES}
    per_prompt_stats = {
        m: {
            f"prompt{pi}": stats_from_samples(whole_samples[m][pi]).as_dict()
            for pi in range(len(prompts))
        }
        for m in _MODES
    }
    block_stats: dict[str, Any] = {}
    for m in _MODES:
        merged: dict[int, list[float]] = {}
        for pi in range(len(prompts)):
            for blk, samples in block_samples[m][pi].items():
                merged.setdefault(blk, []).extend(samples)
        block_stats[m] = {
            f"block{blk}": stats_from_samples(samples).as_dict()
            for blk, samples in sorted(merged.items())
        }
    speedups = {
        "serial_vs_baseline": (
            whole_stats["baseline"].p50 / whole_stats["folded-serial"].p50
            if whole_stats["folded-serial"].p50 > 0
            else None
        ),
        "batched_vs_serial": (
            whole_stats["folded-serial"].p50 / whole_stats["folded-batched"].p50
            if whole_stats["folded-batched"].p50 > 0
            else None
        ),
        "batched_vs_baseline": (
            whole_stats["baseline"].p50 / whole_stats["folded-batched"].p50
            if whole_stats["folded-batched"].p50 > 0
            else None
        ),
        "greedy_vs_batched": (
            whole_stats["folded-batched"].p50 / whole_stats["folded-batched-greedy"].p50
            if whole_stats["folded-batched-greedy"].p50 > 0
            else None
        ),
        "greedy_vs_baseline": (
            whole_stats["baseline"].p50 / whole_stats["folded-batched-greedy"].p50
            if whole_stats["folded-batched-greedy"].p50 > 0
            else None
        ),
    }

    # -- pool serialization (three folded modes) -----------------------------
    windows_serial = _serialize_windows(mff_serial, snap_serial, ctx_map)
    windows_batched = _serialize_windows(mff_batched, snap_batched, ctx_map)
    windows_greedy = _serialize_windows(mff_greedy, snap_greedy, ctx_map)
    for mode, windows in (
        ("folded-serial", windows_serial),
        ("folded-batched", windows_batched),
        ("folded-batched-greedy", windows_greedy),
    ):
        sidecar_counts: dict[str, dict[str, int]] = {}
        for pi in range(len(prompts)):
            for label, counts in window_counts[mode][pi].items():
                entry = sidecar_counts.setdefault(label, {"serial": 0, "batched": 0})
                entry["serial"] += counts["serial"]
                entry["batched"] += counts["batched"]
        windows["sidecar_forwards_per_window"] = sidecar_counts

    # -- H verdicts (frozen formulas) ----------------------------------------
    h1_faces = {
        "merged_position": merged_position,
        "merged_block": merged_block,
        "aligned_entries": sidecar["aligned_quality"],
        "batched_entries": sidecar["batched_quality"],
        "greedy_win_total": greedy_win_total,
    }
    h1 = _h1_verdict(h1_faces)
    h2 = _h2_verdict(whole_stats["folded-batched-greedy"], whole_stats["folded-batched"])
    h3 = _h3_verdict(
        lengths["folded-batched-greedy"],
        lengths["baseline"],
        lengths["folded-batched"],
    )
    aligned_scope = [
        label
        for label, counts in windows_greedy["sidecar_forwards_per_window"].items()
        if counts["batched"] >= 1
    ]
    canaries = {
        "folded-serial": windows_serial["degradation_fallback_count"]["timed"] == 0,
        "folded-batched": windows_batched["degradation_fallback_count"]["timed"] == 0,
        "folded-batched-greedy": (windows_greedy["degradation_fallback_count"]["timed"] == 0),
    }
    ar017 = {
        "folded-batched": {
            "cumulative": windows_batched["ar017_recapture_events"]["cumulative"],
            "timed": windows_batched["ar017_recapture_events"]["timed"],
        },
        "folded-batched-greedy": {
            "cumulative": windows_greedy["ar017_recapture_events"]["cumulative"],
            "timed": windows_greedy["ar017_recapture_events"]["timed"],
        },
        "note": (
            "verified only when the trigger condition actually fired "
            "(untriggered = neutral, never refutes); the policy records only "
            "successful re-captures, so a positive count means the trigger "
            "condition actually fired; MFF_B and MFF_C are judged "
            "independently (independent instances by construction)"
        ),
    }
    h4 = _h4_verdict(
        {"scope": aligned_scope, "windows": windows_greedy},
        canaries,
        sidecar_ledger_all,
        ar017,
    )

    # -- verdict texts (frozen wording, srs 3.4) ------------------------------
    hypotheses = {
        "H1": (
            "aligned-mode (folded-batched-greedy) committed-token quality "
            "aligns with baseline: supported iff three faces ALL pass -- "
            "(a) merged position-level agreement rate (aligned vs baseline; "
            "canvas-absolute positions over [padded_len, min(len_aligned, "
            "len_baseline)), the both-mode generated-token positions, "
            "per-position token equality, merged over prompts as "
            "sum(matched)/sum(compared)) > 0.5; (b) merged block-level "
            "exact-match rate > the folded-batched same-criterion MEASURED "
            "aggregate this run (AR021 prior 0.0 -> strictly positive "
            "passes); (c) per prompt aligned posthoc overall_mean > the "
            "folded-batched same-prompt MEASURED value this run (priors "
            "0.289692 / 0.106418 as reference anchors); any face failing -> "
            "refuted (deterministic output, no inconclusive); greedy_win_rate "
            "and the vs-baseline confidence gap are mechanism report faces"
        ),
        "H2": (
            "aligned-mode whole-generation wall clock does not regress vs "
            "folded-batched (same [N,T] shape A/B): sigma = "
            "max(sigma_aligned, sigma_batched); supported iff p50(batched) - "
            "p50(aligned) >= -2*sigma (acceleration and parity both pass); "
            "refuted iff the regression exceeds 2*sigma; no inconclusive "
            "(directional no-regression gate); vs-baseline speedup is a "
            "report face"
        ),
        "H3": (
            "aligned-mode decode length matches baseline (stop-token hit "
            "face restored): supported iff BOTH prompts' aligned decode "
            "lengths == the baseline same-prompt MEASURED lengths this run "
            "(priors 34 / 66 as reference anchors); else refuted; the "
            "improvement vs folded-batched (priors 160 / 77) is a numbers "
            "report subfield"
        ),
        "H4": (
            "aligned-mode full-stack contract faces do not regress vs the "
            "AR021 criterion: (i) every aligned-mode shape window containing "
            "a batched call has batched captures > 0 AND timed replays > 0 "
            "(AR021 H4 criterion applied to MFF_C); (ii) the "
            "degradation-class no-fallback canary is timed == 0 for ALL "
            "THREE folded modes; (iii) the publish ledger stays within "
            "c*(L+N) with c=4 frozen (three folded modes, per prompt "
            "sidecar); (iv) AR017 adaptive events are verified only when "
            "their trigger condition actually fired (untriggered = neutral; "
            "MFF_B and MFF_C judged independently). Otherwise refuted"
        ),
    }
    verdicts: dict[str, dict[str, Any]] = {
        "H1": {"hypothesis": hypotheses["H1"], **h1},
        "H2": {"hypothesis": hypotheses["H2"], **h2},
        "H3": {"hypothesis": hypotheses["H3"], **h3},
        "H4": {"hypothesis": hypotheses["H4"], **h4},
    }
    if reproduction_anomaly is not None:
        for key in ("H1", "H2", "H3", "H4"):
            verdicts[key]["verdict"] = "blocked"
            verdicts[key]["blocked_reason"] = (
                "folded-batched reproduction self-check failed against the "
                "AR021 priors; the same-environment determinism assumption "
                "failed wholesale, so no single-face verdict is trustworthy "
                "(design 4.2.2(5))"
            )

    # -- summaries / environment ---------------------------------------------
    stable_summary: dict[str, Any] = {}
    for m in _MODES:
        ratios = profiler_ratios[m]
        if ratios:
            mean = sum(ratios) / len(ratios)
            stable_summary[m] = {
                "mean": round(mean, 4),
                "std": round((sum((r - mean) ** 2 for r in ratios) / len(ratios)) ** 0.5, 4),
                "min": round(min(ratios), 4),
                "max": round(max(ratios), 4),
                "n": len(ratios),
            }
        else:
            stable_summary[m] = None

    vram_peaks = {m: {"max_over_reps_mb": max(peaks[m]), "per_rep_mb": peaks[m]} for m in _MODES}
    counts_supported = _cuda_profiler_supported()
    driver = _gpu_driver_version()
    import transformers

    timed_ops = {
        "folded-serial": (mff_serial.cache.write_op_count - timed_ops_base["folded-serial"]),
        "folded-batched": (mff_batched.cache.write_op_count - timed_ops_base["folded-batched"]),
        "folded-batched-greedy": (
            mff_greedy.cache.write_op_count - timed_ops_base["folded-batched-greedy"]
        ),
    }

    batch_canary: dict[str, Any] = {}
    for mode, windows in (
        ("folded-serial", windows_serial),
        ("folded-batched", windows_batched),
        ("folded-batched-greedy", windows_greedy),
    ):
        batch_canary[mode] = {
            "timed_degradation_fallbacks": windows["degradation_fallback_count"]["timed"],
            "cumulative_degradation_fallbacks": windows["degradation_fallback_count"]["cumulative"],
            "cumulative_structural_eager": windows["structural_eager_count"]["cumulative"],
            "timed_structural_eager": windows["structural_eager_count"]["timed"],
            "pass": windows["degradation_fallback_count"]["timed"] == 0,
        }
    batch_canary["pass"] = bool(all(batch_canary[m]["pass"] for m in _FOLDED_MODES))
    batch_canary["note"] = (
        "canary = degradation class only, timed segment == 0; structural "
        "eager (by-design window populate/switch) is counted separately and "
        "never fails the canary (AR020-style); all three folded modes"
    )

    mff_kwargs_decl: dict[str, Any] = {
        "use_cuda_graph": True,
        "graph_runner_pool_size": POOL_SIZE,
        "split_layers": False,
        "graph_capacity_ratio": CAPACITY_RATIO,
        "graph_adaptive_capacity": True,
        "graph_adaptive_config": "AdaptiveCapacityConfig()",
        "scheduler": None,
        "gate": f"SimilarityGate(tau={TAU}, cosine, eps={EPS})",
        "cache": f"ActivationCache({CACHE_BUDGET} tokens/layer)",
    }
    artifact: dict[str, Any] = {
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
                "prepad_note": (
                    "prompts pre-padded to exact 32-token multiples (pad=eos) "
                    "so the sampler's own first-block padding is zero and the "
                    "pinned fold-context mask is the pure block-causal mask, "
                    "element-equal to the remote model's internal eval_mask "
                    "(the raw path's [B,T] mask is overwritten internally by "
                    "the remote forward); non-zero padding would break "
                    "cross-mode parity by construction"
                ),
            },
            "block_constants": {
                "block_size": BLOCK,
                "small_block_size": BLOCK,
                "threshold": 0.95,
                "temperature": 0.0,
            },
            "warmup_strategy": (
                "one full discarded sample per (mode, prompt) before any "
                "timing, all four modes: per-shape-window pool pre-capture "
                "(A5 scheme a: the warmup round naturally captures every "
                "window's serial and batched runners, MFF_A/B/C alike) + "
                "Triton/JIT warmup + the AR017 adaptive transition; capture "
                "costs are excluded from p50 (srs FR4); timed-segment "
                "re-captures, if any, are counted and timed honestly"
            ),
            "interleaving": (
                "per-rep round-robin over the four modes; rep r starts at "
                "position r%4 (anti-drift, WDDM fairness)"
            ),
            "timing_method": (
                "host perf_counter + torch.cuda.synchronize (#39); "
                "whole-generation wall clock plus per-block anchors (the "
                "forward wrappers observe the step_idx change as the block "
                "boundary; syncs only at those boundaries)"
            ),
            "mode_configs": {
                "baseline": {
                    "folded_model": None,
                    "num_block_candidates": 0,
                    "description": "raw FastDLLMSampler (unchanged recipe)",
                },
                "folded-serial": {
                    "mff_kwargs": {
                        **mff_kwargs_decl,
                        "graph_adaptive_capacity": False,
                        "graph_adaptive_config": None,
                    },
                    "sampler": {"num_block_candidates": 0},
                    "description": (
                        "FR2 serial chain folding (#26) with the pinned "
                        "fold-context trio and the shape-keyed serial pool"
                    ),
                },
                "folded-batched": {
                    "mff_kwargs": {**mff_kwargs_decl},
                    "sampler": {
                        "num_block_candidates": NUM_CANDIDATES,
                        "candidate_top_k": CANDIDATE_TOP_K,
                    },
                    "description": (
                        "FR3 batched speculative block verification (N=4 "
                        "from the AR019 artifact matrix, candidate_top_k=10, "
                        "A5 scheme a); AR017 adaptive capacity engaged on "
                        "the batched path only (#37); split_layers=False on "
                        "all three MFF instances (Split-layer resident hooks "
                        "would stack on the shared model)"
                    ),
                },
                "folded-batched-greedy": {
                    "mff_kwargs": {**mff_kwargs_decl},
                    "sampler": {
                        "num_block_candidates": NUM_CANDIDATES,
                        "candidate_top_k": CANDIDATE_TOP_K,
                        "candidate_include_greedy": True,
                    },
                    "description": (
                        "AR022 alignment mode: MFF_C, an independent "
                        "instance constructed identically to MFF_B (own "
                        "cache/gate/pool/canary/AR017 policy -- per-mode "
                        "isolation for H4) with candidate_include_greedy="
                        "True (row 0 = greedy hypothesis); same [N,T] "
                        "batched shape, N, top-k, prompt set, and seed as "
                        "folded-batched -- the speed variable is "
                        "constructively zeroed (same-shape A/B)"
                    ),
                },
            },
            "greedy_anchor_declaration": (
                "candidate row 0 = the greedy hypothesis: every still-masked "
                "position is filled with the draft top-k ordered-set head "
                "(the frozen argmax definition; deterministic construction, "
                "zero device readback, no generator); N semantics frozen: "
                "total candidates = config N, greedy occupies row 0, random "
                "candidates = N-1 (j = 1..N-1, the frozen seed formula keeps "
                "j as the row number, so rows 1..N-1 match a greedy-off "
                "same-seed run inside the canvas-consistent region -- the "
                "first speculative small block); branch_ids keep their AR021 "
                "row-number semantics (_cand0 names the greedy row); "
                "min-prob selection is unchanged -- the greedy row competes "
                "with no exemption or weighting and may lose"
            ),
            "reproduction_selfcheck_declaration": (
                "the folded-batched sidecar quality face (same seed, "
                "deterministic trajectory, warmup-post replay pool state) "
                "is compared for STRICT equality against the AR021 artifact "
                "priors (quality.h3_batched_vs_baseline: block_exact_match "
                "0.0/0.0, len_other 160/77, posthoc_mean 0.289692/0.106418, "
                "posthoc_min 5e-06/0.0); on mismatch the artifact records "
                "reproduction_anomaly and ALL FOUR H verdicts are set to "
                "'blocked' (the same-environment determinism assumption "
                "failed wholesale; no single-face verdict is trustworthy; "
                "never silently substituted, never fabricated); the check "
                "binds same-GPU/driver/venv reruns (RERUN_CHECKLIST scope) "
                "-- a different host failing equality is recorded blocked "
                "honestly; under --quick the face degrades to a structural "
                "validation (single prompt / num_tokens=64 cannot reproduce "
                "the full-scale priors) and does not gate"
            ),
            "forced_divergent_declaration": (
                "the sampler pins a per-(T, prompt_len) forced base tensor "
                "whose True positions cover exactly the active block; every "
                "folded forward (denoise iterations, the block-advance AR "
                "step, and the [N,T] batched verification) threads it "
                "through ManualFoldedForward(forced_divergent=...); the "
                "freshly committed / actively denoised region is always "
                "recomputed, never gate-certified stable against a stale "
                "parent"
            ),
            "shape_keyed_pool_declaration": (
                "graph_runner_pool_size=8 per MFF instance: one runner per "
                "(tokens shape, pinned mask identity) window == (T, "
                "prompt_len) serial / (N, T, prompt_len) batched; expected "
                "entries <= 8 per pool (W*P upper bound; folded-batched and "
                "folded-batched-greedy: serial (W-1)*P, batched W*P; "
                "folded-serial <= W*P data-dependent), zero evictions at "
                "cap=8 (LRU is a defensive bound only); MFF_A/B/C are "
                "independent instances, so their pools are fully isolated"
            ),
            "use_cuda_graph": True,
            "a5_scheme_a_note": (
                "warmup pre-capture amortization holds under the bench "
                "protocol (every timed rep replays each window's runners); "
                "in single-shot production generation the capture amortizes "
                "NEGATIVE (one capture ~= 2-3 forwards serving a single "
                "replay) -- recommend use_cuda_graph=False for single-shot "
                "generate scenarios"
            ),
            "pool_cap": POOL_SIZE,
            "cache_budget_tokens_per_layer": CACHE_BUDGET,
            "ledger_c": LEDGER_C,
            "ledger_normalization": (
                "per publish event: a serial forward publishes 1 event "
                "counting L+1 ops (<= 4*(L+1), trivially in bound); a "
                "batched publish counts O(L+N) ops (put_batch foreach groups "
                "+ immediate executions, <= 4*(L+N)); the aggregate over a "
                "generation preserves the bounds (AR020-style, c frozen at "
                "design); verified on the untimed sidecar generation (same "
                "deterministic trajectory as the timed reps)"
            ),
            "sigma_convention": (
                "sigma = max(sigma_a, sigma_b) bound to the mode pair; H2 "
                "(greedy vs batched) uses sigma = max(sigma_greedy, "
                "sigma_batched) with the directional no-regression rule "
                "(supported iff p50(batched) - p50(greedy) >= -2*sigma; no "
                "inconclusive); quality faces are deterministic single-shot "
                "sidecar measurements with no sigma; samples pooled over "
                "prompts x reps per mode"
            ),
            "h4_scope_note": (
                "the aligned-mode batched capture/replay sub-criterion is "
                "verified on every shape window with >= 1 batched call in "
                "the sidecar generation (the AR021 H4 criterion applied to "
                "MFF_C); the degradation canary and the c=4 ledger bound "
                "cover ALL THREE folded modes; AR017 events are verified "
                "only when triggered (untriggered = neutral), MFF_B and "
                "MFF_C judged independently"
            ),
            "canary_note": (
                "the no-fallback canary checks the DEGRADATION class only "
                "(timed segment == 0): every non-structural graph-path eager "
                "fallback; structural eager (window-populate, window-switch, "
                "last-window no-AR) is counted separately and excluded by "
                "design (else the canary is constructively false)"
            ),
            "vram_peak_mb_per_mode": vram_peaks,
            "vram_note": (
                "three MFF instances (MFF_A/B/C) are resident for the whole "
                "run, so every mode's peak includes all three pools; the "
                "design estimate was ~5.8GB + interaction margin <= 6.5GB "
                "on a 16GB card (AR021 folded-batched peak 5374.5MB + one "
                "MFF_C-shaped increment ~427MB); the measured peaks above "
                "are the honest calibration"
            ),
            "device": torch.cuda.get_device_name(0),
            "gpu_driver_version": driver,
            "gpu_driver_version_note": (
                "queried via nvidia-smi; null means the query failed on "
                "this host and the value was omitted rather than fabricated"
            ),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "platform_note": (
                "Windows WDDM display-driver mode (no clock locking); noise "
                "mitigated by warmup + reps + p50 + per-rep round-robin"
            ),
            "kernel_count": None,
            "kernel_count_note": (
                "runtime CUPTI probe: this torch build cannot record CUDA "
                "profiler events (LIBKINETO_NOCUPTI); kernel-launch counts "
                "recorded null + note, never fabricated (#39) - the "
                "write_op_count ledger and graph_pool_stats counters are the "
                "structural evidence"
                if not counts_supported
                else "CUPTI available; wall-clock remains the primary metric"
            ),
            "single_run_discipline": (
                "the full (reps=30) run is a pre-committed single-shot "
                "verdict, no rerun fishing (#39 continuation); this "
                "artifact is the --quick smoke run (separate path, never "
                "the formal artifact) and its H verdicts are non-final"
                if quick
                else "pre-committed single-shot verdict, no rerun fishing"
            ),
            "model": "Fast-dLLM-v2-1.5B (Efficient-Large-Model, remote code)",
            "model_details": model_meta,
            "venv": sys.prefix != sys.base_prefix,
            "run_date_utc": datetime.now(timezone.utc).isoformat(),
            "command": (
                "python -m scripts.ar022_speculative_quality_bench"
                + (" --quick" if quick else f" --reps {reps}")
                + f" --model-path {model_path}"
            ),
        },
        "hypotheses_declared": hypotheses,
        "hypotheses_verdicts": verdicts,
        "mode_comparison": {
            m: {
                "whole_gen_ms": whole_stats[m].as_dict(),
                "per_prompt_ms": per_prompt_stats[m],
                "per_block_ms": block_stats[m],
                "stable_ratio_timed": stable_summary[m],
            }
            for m in _MODES
        },
        "speedups_p50": {k: (round(v, 4) if v is not None else None) for k, v in speedups.items()},
        "per_shape_window": {
            "folded-serial": windows_serial,
            "folded-batched": windows_batched,
            "folded-batched-greedy": windows_greedy,
        },
        "quality": {
            "h1_serial_vs_baseline": sidecar["h1"],
            "batched_vs_baseline": sidecar["batched_quality"],
            "aligned_vs_baseline": sidecar["aligned_quality"],
            "merged_position_match": merged_position,
            "merged_block_match": merged_block,
            "greedy_win_total": greedy_win_total,
            "reproduction_selfcheck": reproduction_selfcheck,
        },
        "reproduction_anomaly": reproduction_anomaly,
        "ledger": {
            "c": LEDGER_C,
            "normalization": (
                "ops <= c*((L+1)*serial_events + (L+N)*batched_events) per "
                "sidecar generation; L = num_hidden_layers, N = "
                f"{NUM_CANDIDATES}"
            ),
            "num_layers": num_layers,
            "per_prompt": sidecar["ledger"],
            "within_bound": sidecar_ledger_all,
            "timed_segment_ops_total": timed_ops,
        },
        "batch_canary": batch_canary,
    }
    return artifact


def main() -> None:
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
    parser.add_argument("--reps", type=int, default=30, help="Timed reps")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke subset: reps=3, single prompt, num_tokens=64 (2 blocks)",
    )
    add_hf_env_arguments(parser)
    args = parser.parse_args()

    quick = args.quick
    reps = 3 if quick else args.reps
    out_path = args.out or (QUICK_OUTPUT_PATH if quick else DEFAULT_OUTPUT_PATH)
    model_path = _resolve_model_path(args.model_path)
    data = run_bench(model_path, reps=reps, quick=quick)
    out = write_results(data, out_path)
    print()
    for key in ("H1", "H2", "H3", "H4"):
        entry = data["hypotheses_verdicts"][key]
        verdict = entry.get("verdict", entry)
        print(f"{key}: {verdict}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
