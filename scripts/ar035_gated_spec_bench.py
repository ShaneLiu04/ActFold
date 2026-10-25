"""AR035 bench: confidence-gated speculative commit (cost/quality knob).

Three arms on the official v2 geometry (block=32, small_block=32):
  A  baseline          non-cached block-diffusion baseline (full recompute)
  D  cache-spec        AR034 KV-cached speculative block decoding
  G  cache-gated       D + speculative_commit_gate (AR035)

Framing (frozen BEFORE data collection): AR035 is delivered as a
**cost/quality knob**, not a speedup rung. G commits only verified positions
whose probability clears ``threshold`` (force-advancing one argmax position
per round), so it trades verification rounds for commit conservatism. The
gates below measure BOTH sides honestly and never rerun-until-pass.

H1 quality   : G mask-free on every prompt, token-identical to A, and
               decode length equal to A (gating must not corrupt output).
H2 cost      : p50(A)/p50(G) > 1.0 (gating still beats the baseline) with
               the p50(D)/p50(G) cost ratio published verbatim; a negative
               H2 is an accepted, pre-declared outcome.
H3 ledger    : G runs ``spec_gated_iterations == spec_verify_forwards`` > 0,
               A/D report zero gated iterations, no degraded calls.
H4 repro     : full double-run equality per (arm, prompt) — a mismatch
               blocks ALL verdicts (AR031 caliber).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env  # noqa: E402

apply_hf_env()

import torch  # noqa: E402

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
    _prepare_prompts,
    _resolve_model_path,
    write_results,
)

SCHEMA = "ar035_gated_spec_bench"
SCHEMA_VERSION = 1
ARMS = ("baseline", "cache-spec", "cache-gated")
DEFAULT_OUTPUT_PATH = Path("results/optimization/ar035_gated_spec_bench.json")
QUICK_OUTPUT_PATH = Path("results/optimization/ar035_gated_spec_bench_quick.json")
REPS_DEFAULT = 30
QUICK_REPS = 2
SMALL_BLOCK = 32
SPEC_CANDIDATES = 4
SPEC_TOP_K = 10
_A, _D, _G = ARMS


def resolve_output_path(out_arg: Path | None, quick: bool) -> Path:
    if out_arg is not None:
        return out_arg
    return QUICK_OUTPUT_PATH if quick else DEFAULT_OUTPUT_PATH


def sampler_config(num_tokens: int, arm: str) -> FastDLLMSamplerConfig:
    base: dict[str, Any] = {
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": SMALL_BLOCK,
        "threshold": 0.95,
        "temperature": 0.0,
        "seed": SEED,
    }
    if arm in (_D, _G):
        base.update(
            history_cache=True,
            history_small_block_cache=False,
            num_block_candidates=SPEC_CANDIDATES,
            candidate_top_k=SPEC_TOP_K,
            candidate_include_greedy=True,
        )
    if arm == _G:
        base["speculative_commit_gate"] = True
    return FastDLLMSamplerConfig(**base)


def verdict_h1(rows: dict[str, Any]) -> dict[str, Any]:
    face = rows["quality"]
    ok = (
        face["mask_free_all"] and face["token_rate_vs_baseline"] == 1.0 and face["decode_equal_all"]
    )
    return {"verdict": "supported" if ok else "refuted", **face}


def verdict_h2(p50: dict[str, float]) -> dict[str, Any]:
    a, g, d = p50[_A], p50[_G], p50[_D]
    speedup_a = a / g if g else 0.0
    cost_vs_d = g / d if d else 0.0
    return {
        "verdict": "supported" if speedup_a > 1.0 else "refuted",
        "p50_a_ms": a,
        "p50_d_ms": d,
        "p50_g_ms": g,
        "speedup_a_over_g": speedup_a,
        "cost_g_over_d": cost_vs_d,
    }


def verdict_h3(ledger: dict[str, Any]) -> dict[str, Any]:
    g = ledger[_G]
    ok = (
        g["spec_gated_iterations"] > 0
        and g["spec_gated_iterations"] == g["spec_verify_forwards"]
        and ledger[_A]["spec_gated_iterations"] == 0
        and ledger[_D]["spec_gated_iterations"] == 0
        and all(ledger[arm]["degraded_calls"] == 0 for arm in ARMS)
    )
    return {"verdict": "supported" if ok else "refuted", "ledgers": ledger}


def verdict_h4(repro: dict[str, Any]) -> dict[str, Any]:
    ok = bool(repro["ok"]) and repro["mismatches"] == 0
    return {"verdict": "supported" if ok else "refuted", **repro}


def run_bench(model_path: Path, reps: int, quick: bool) -> dict[str, Any]:
    num_tokens = QUICK_NUM_TOKENS if quick else FULL_NUM_TOKENS
    torch.manual_seed(SEED)
    model, tokenizer, meta = _load_model(model_path)
    prompts = _prepare_prompts(tokenizer, quick=quick)
    llm = _BenchDiffusionLLM(model, tokenizer)

    samplers = {arm: FastDLLMSampler(llm, config=sampler_config(num_tokens, arm)) for arm in ARMS}
    ids = [p["ids"][:, : p["raw_len"]] for p in prompts]

    samples: dict[str, list[float]] = {arm: [] for arm in ARMS}
    per_prompt: dict[str, dict[int, list[float]]] = {
        arm: {i: [] for i in range(len(prompts))} for arm in ARMS
    }
    sequences: dict[str, list[torch.Tensor]] = {arm: [] for arm in ARMS}
    repro_rows: dict[str, list[torch.Tensor]] = {arm: [] for arm in ARMS}

    for rep in range(reps):
        order = ARMS[rep % len(ARMS) :] + ARMS[: rep % len(ARMS)]
        for arm in order:
            sampler = samplers[arm]
            for idx, row in enumerate(ids):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = sampler.sample(row)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000.0
                samples[arm].append(ms)
                per_prompt[arm][idx].append(ms)
                sequences[arm].append(out.sequences.clone())

    for arm in ARMS:
        for idx, row in enumerate(ids):
            repro_rows[arm].append(samplers[arm].sample(row).sequences.clone())

    p50 = {arm: stats_from_samples(samples[arm]).p50 for arm in ARMS}
    ledger = {arm: dict(samplers[arm].history_cache_stats) for arm in ARMS}

    mask_id = int(getattr(tokenizer, "mask_token_id", 0) or 0)
    quality: dict[str, Any] = {
        "mask_free_all": True,
        "decode_equal_all": True,
        "token_rate_vs_baseline": 0.0,
        "decode_lengths": {},
        "identical_rows": [],
    }
    base_seq = sequences[_A]
    g_seq = sequences[_G]
    matches = 0
    for i, (b, g) in enumerate(zip(base_seq, g_seq)):
        if mask_id and bool((b == mask_id).any()):
            quality["mask_free_all"] = False
        if mask_id and bool((g == mask_id).any()):
            quality["mask_free_all"] = False
        if bool(torch.equal(b, g)):
            matches += 1
            quality["identical_rows"].append(i)
        if b.shape != g.shape:
            quality["decode_equal_all"] = False
    quality["token_rate_vs_baseline"] = matches / len(base_seq) if base_seq else 0.0
    quality["decode_lengths"] = {
        "baseline": sorted({int(s.numel()) for s in base_seq}),
        "cache-spec": sorted({int(s.numel()) for s in sequences[_D]}),
        "cache-gated": sorted({int(s.numel()) for s in g_seq}),
    }

    mismatches = 0
    for arm in ARMS:
        for got, want in zip(repro_rows[arm], sequences[arm]):
            if not bool(torch.equal(got, want)):
                mismatches += 1
    repro = {"ok": mismatches == 0, "mismatches": mismatches, "reps": reps}

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "ar": "AR035",
        "title": "confidence-gated speculative commit (cost/quality knob)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quick": quick,
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": SMALL_BLOCK,
        "reps": reps,
        "seed": SEED,
        "threshold": 0.95,
        "num_prompts": len(prompts),
        "environment": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "driver": _gpu_driver_version(),
            "torch": torch.__version__,
            "model_path": str(model_path),
        },
        "timings": {
            "per_arm_p50_ms": p50,
            "per_arm_full_stats": {arm: stats_from_samples(samples[arm]).as_dict() for arm in ARMS},
            "per_prompt_p50_ms": {
                arm: {i: stats_from_samples(v).p50 for i, v in per_prompt[arm].items()}
                for arm in ARMS
            },
        },
        "h1_quality": verdict_h1({"quality": quality}),
        "h2_cost": verdict_h2(p50),
        "h3_ledger": verdict_h3(ledger),
        "h4_reproduction": verdict_h4(repro),
    }
    verdicts = [
        payload["h1_quality"]["verdict"],
        payload["h2_cost"]["verdict"],
        payload["h3_ledger"]["verdict"],
        payload["h4_reproduction"]["verdict"],
    ]
    payload["overall"] = (
        "blocked"
        if payload["h4_reproduction"]["verdict"] == "refuted"
        else ("supported" if all(v == "supported" for v in verdicts) else "refuted")
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    add_hf_env_arguments(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out = resolve_output_path(args.out, args.quick)
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out} (pass --force)", file=sys.stderr)
        return 4
    reps = args.reps if args.reps is not None else (QUICK_REPS if args.quick else REPS_DEFAULT)
    payload = run_bench(_resolve_model_path(args.model), reps, args.quick)
    write_results(payload, out)
    print(
        f"overall={payload['overall']} h1={payload['h1_quality']['verdict']} "
        f"h2={payload['h2_cost']['verdict']} h3={payload['h3_ledger']['verdict']} "
        f"h4={payload['h4_reproduction']['verdict']} -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
