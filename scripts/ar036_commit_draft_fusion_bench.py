"""AR036 bench: commit-draft fusion (CDF) on the AR034 composition arm.

Arms (official v2 geometry block=32 / small_block=32):
  A  baseline      non-cached block-diffusion baseline (full recompute)
  D  cache-spec    AR034 KV-cached speculative block decoding (3 forwards/block)
  F  cache-fused   D + commit_draft_fusion (2 forwards/block when eligible)

Frozen gates (before data collection):
  H1 quality  : F token-identical to A on every prompt, mask-free, same decode length
  H2 speed    : p50(D)/p50(F) >= 1.15, p50(A)/p50(F) published verbatim
  H3 ledger   : F draft + fused == verify (one draft source per block), D fused == 0,
                no degraded calls, F fused > 0
  H4 repro    : double-run equality per (arm, prompt); mismatch blocks all verdicts
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

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

SCHEMA = "ar036_commit_draft_fusion_bench"
SCHEMA_VERSION = 1
ARMS = ("baseline", "cache-spec", "cache-fused")
DEFAULT_OUTPUT_PATH = Path("results/optimization/ar036_commit_draft_fusion_bench.json")
QUICK_OUTPUT_PATH = Path("results/optimization/ar036_commit_draft_fusion_bench_quick.json")
REPS_DEFAULT = 30
QUICK_REPS = 2
SMALL_BLOCK = 32
SPEC_CANDIDATES = 4
SPEC_TOP_K = 10
_A, _D, _F = ARMS
_H2_MIN_SPEEDUP = 1.15


def resolve_output_path(out_arg: Path | None, quick: bool) -> Path:
    if out_arg is not None:
        return out_arg
    return QUICK_OUTPUT_PATH if quick else DEFAULT_OUTPUT_PATH


def sampler_config(num_tokens: int, arm: str, no_eos: bool = False) -> FastDLLMSamplerConfig:
    base: dict[str, Any] = {
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": SMALL_BLOCK,
        "threshold": 0.95,
        "temperature": 0.0,
        "seed": SEED,
    }
    if arm in (_D, _F):
        base.update(
            history_cache=True,
            history_small_block_cache=False,
            num_block_candidates=SPEC_CANDIDATES,
            candidate_top_k=SPEC_TOP_K,
            candidate_include_greedy=True,
        )
    if arm == _F:
        base["commit_draft_fusion"] = True
    if no_eos:
        # Geometry measurement only: -1 never matches a generated token, so the
        # block loop runs to num_tokens and multi-block fusion can be measured.
        base["stop_token_id"] = -1
    return FastDLLMSamplerConfig(**base)


def verdict_h1(face: dict[str, Any]) -> dict[str, Any]:
    ok = (
        face["mask_free_all"] and face["token_rate_vs_baseline"] == 1.0 and face["decode_equal_all"]
    )
    return {"verdict": "supported" if ok else "refuted", **face}


def verdict_h2(p50: dict[str, float]) -> dict[str, Any]:
    a, d, f = p50[_A], p50[_D], p50[_F]
    speedup_d_f = d / f if f else 0.0
    return {
        "verdict": "supported" if speedup_d_f >= _H2_MIN_SPEEDUP else "refuted",
        "threshold": _H2_MIN_SPEEDUP,
        "p50_a_ms": a,
        "p50_d_ms": d,
        "p50_f_ms": f,
        "speedup_d_over_f": speedup_d_f,
        "speedup_a_over_f": a / f if f else 0.0,
    }


def verdict_h3(ledger: dict[str, dict[str, int]]) -> dict[str, Any]:
    f, d = ledger[_F], ledger[_D]
    ok = (
        f["spec_fused_forwards"] > 0
        and f["spec_draft_forwards"] + f["spec_fused_forwards"] == f["spec_verify_forwards"]
        and d["spec_fused_forwards"] == 0
        and d["spec_draft_forwards"] == d["spec_verify_forwards"]
        and all(ledger[arm]["degraded_calls"] == 0 for arm in ARMS)
    )
    return {"verdict": "supported" if ok else "refuted", "ledgers": ledger}


def verdict_h4(repro: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": "supported" if repro["ok"] and repro["mismatches"] == 0 else "refuted",
        **repro,
    }


def _custom_prompts(tokenizer: Any, texts: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        out.append({"ids": ids, "raw_len": int(ids.shape[1])})
    return out


def run_bench(
    model_path: Path,
    reps: int,
    quick: bool,
    prompt_texts: list[str] | None = None,
    arms: tuple[str, ...] = ARMS,
    num_tokens_override: int | None = None,
    no_eos: bool = False,
) -> dict[str, Any]:
    num_tokens = (
        int(num_tokens_override)
        if num_tokens_override
        else (QUICK_NUM_TOKENS if quick else FULL_NUM_TOKENS)
    )
    torch.manual_seed(SEED)
    model, tokenizer, meta = _load_model(model_path)
    prompts = (
        _custom_prompts(tokenizer, prompt_texts)
        if prompt_texts
        else _prepare_prompts(tokenizer, quick=quick)
    )
    llm = _BenchDiffusionLLM(model, tokenizer)
    samplers = {
        arm: FastDLLMSampler(llm, config=sampler_config(num_tokens, arm, no_eos)) for arm in arms
    }
    ids = [p["ids"][:, : p["raw_len"]] for p in prompts]

    samples: dict[str, list[float]] = {arm: [] for arm in ARMS}
    per_prompt: dict[str, dict[int, list[float]]] = {
        arm: {i: [] for i in range(len(prompts))} for arm in arms
    }
    sequences: dict[str, list[torch.Tensor]] = {arm: [] for arm in arms}

    for rep in range(reps):
        order = arms[rep % len(arms) :] + arms[: rep % len(arms)]
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

    mismatches = 0
    for arm in arms:
        sampler = samplers[arm]
        for idx, row in enumerate(ids):
            again = sampler.sample(row).sequences
            if not bool(torch.equal(again, sequences[arm][idx])):
                mismatches += 1

    mask_id_smoke = int(getattr(tokenizer, "mask_token_id", None) or 0) or 151665
    p50 = {arm: stats_from_samples(samples[arm]).p50 for arm in arms}
    ledger = {arm: dict(samplers[arm].history_cache_stats) for arm in arms}

    if set(arms) != set(ARMS):
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "ar": "AR036",
            "quick": quick,
            "num_tokens": num_tokens,
            "smoke_subset": list(arms),
            "timings": {"per_arm_p50_ms": {a: stats_from_samples(samples[a]).p50 for a in arms}},
            "ledgers": ledger,
            "h4_reproduction": verdict_h4(
                {"ok": mismatches == 0, "mismatches": mismatches, "reps": reps}
            ),
            "decode_lengths": {a: sorted({int(s.numel()) for s in sequences[a]}) for a in arms},
            "masks_left": {
                a: [int((s == mask_id_smoke).sum()) for s in sequences[a]] for a in arms
            },
            "overall": "smoke",
        }

    mask_id = int(getattr(tokenizer, "mask_token_id", None) or 0)
    if not mask_id:
        mask_id = int(tokenizer.convert_tokens_to_ids("|<MASK>|"))
    base_seq, f_seq = sequences[_A], sequences[_F]
    matches = 0
    mask_free = True
    decode_equal = True
    for b, f in zip(base_seq, f_seq):
        if mask_id and (bool((b == mask_id).any()) or bool((f == mask_id).any())):
            mask_free = False
        if bool(torch.equal(b, f)):
            matches += 1
        if b.shape != f.shape:
            decode_equal = False
    quality = {
        "mask_free_all": mask_free,
        "decode_equal_all": decode_equal,
        "token_rate_vs_baseline": matches / len(base_seq) if base_seq else 0.0,
        "decode_lengths": {
            "baseline": sorted({int(s.numel()) for s in base_seq}),
            "cache-spec": sorted({int(s.numel()) for s in sequences[_D]}),
            "cache-fused": sorted({int(s.numel()) for s in f_seq}),
        },
    }
    repro = {"ok": mismatches == 0, "mismatches": mismatches, "reps": reps}

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "ar": "AR036",
        "title": "commit-draft fusion (CDF)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quick": quick,
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": SMALL_BLOCK,
        "reps": reps,
        "seed": SEED,
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
                for arm in arms
            },
        },
        "h1_quality": verdict_h1(quality),
        "h2_speed": verdict_h2(p50),
        "h3_ledger": verdict_h3(ledger),
        "h4_reproduction": verdict_h4(repro),
    }
    verdicts = [
        payload["h1_quality"]["verdict"],
        payload["h2_speed"]["verdict"],
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
    ap.add_argument("--prompt-text", action="append", default=None)
    ap.add_argument("--arms", default=None, help="comma subset for smoke runs")
    ap.add_argument("--num-tokens", type=int, default=None)
    ap.add_argument("--no-eos", action="store_true")
    args = ap.parse_args(argv)

    out = resolve_output_path(args.out, args.quick)
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out} (pass --force)")
        return 4
    reps = args.reps if args.reps is not None else (QUICK_REPS if args.quick else REPS_DEFAULT)
    arms = ARMS if not args.arms else tuple(a.strip() for a in args.arms.split(","))
    payload = run_bench(
        _resolve_model_path(args.model),
        reps,
        args.quick,
        args.prompt_text,
        arms,
        args.num_tokens,
        args.no_eos,
    )
    write_results(payload, out)
    print("overall=%s -> %s" % (payload["overall"], out), flush=True)
    print(payload.get("timings", {}).get("per_arm_p50_ms"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
