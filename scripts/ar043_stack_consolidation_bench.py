"""AR043 consolidation bench: pick the best measured stack and prove the choice.

Five arms over the AR040 prompt set at one length:
  A    baseline            official recipe, no history KV, no speculation
  KV   kv-only             AR033 cross-block history KV + official remasking
  D    spec                KV + greedy-anchored speculative block commit (N=4)
  Q    quality preset      (KV + speculation + AR041 always-fallback tier)
  L    latency preset      (KV + speculation + AR036 CDF + AR037 ACB + AR039 VFG)

Frozen verdicts:
  V1 best-quality  : the fastest arm whose mean log p >= logp(baseline) - 0.05
                     on all cells (reported with its speedup)
  V2 best-latency  : the fastest arm overall (its log-p delta is reported, not judged)
  V3 stacking gain : the latency preset is not slower than D on >= 2/3 cells
                     (ACB + VFG must actually help once stacked)
  V4 reproduction  : Q and L re-run token-identically on every cell
"""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402

from actfold.models.fast_dllm_sampler import FastDLLMSampler  # noqa: E402
from actfold.models.sampler_presets import fastdllm_preset  # noqa: E402
from scripts.ar021_generation_loop_bench import (  # noqa: E402
    BLOCK,
    SEED,
    _BenchDiffusionLLM,
    _gpu_driver_version,
    _load_model,
    _resolve_model_path,
    write_results,
)
from scripts.ar040_matrix_bench import PROMPTS, mean_logp  # noqa: E402

SCHEMA = "ar043_stack_consolidation_bench"
SCHEMA_VERSION = 1
ARMS = ("baseline", "kv-only", "spec", "quality", "latency")
DEFAULT_OUTPUT = Path("results/optimization/ar043_stack_consolidation_bench.json")
QUICK_OUTPUT = Path("results/optimization/ar043_stack_consolidation_bench_quick.json")
_H1_DROP = 0.05


def _config(arm: str, num_tokens: int) -> Any:
    common: dict[str, Any] = {
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": BLOCK,
        "threshold": 0.95,
        "temperature": 0.0,
        "seed": SEED,
        "stop_token_id": -1,
    }
    if arm == "baseline":
        return fastdllm_preset("baseline", **common)
    if arm == "kv-only":
        return fastdllm_preset("baseline", history_cache=True, **common)
    if arm == "spec":
        return fastdllm_preset(
            "baseline",
            history_cache=True,
            num_block_candidates=4,
            candidate_top_k=10,
            candidate_include_greedy=True,
            commit_draft_fusion=True,
            **common,
        )
    return fastdllm_preset(arm, **common)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    add_hf_env_arguments(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--num-tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out = args.out or (QUICK_OUTPUT if args.quick else DEFAULT_OUTPUT)
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out} (pass --force)")
        return 4

    model, tokenizer, meta = _load_model(_resolve_model_path(args.model))
    llm = _BenchDiffusionLLM(model, tokenizer)
    cells: list[dict[str, Any]] = []
    for prompt in PROMPTS[: args.prompts]:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        gen_start = int(ids.shape[1])
        cell: dict[str, Any] = {"prompt_head": prompt[:40], "arms": {}}
        for arm in ARMS:
            sampler = FastDLLMSampler(llm, config=_config(arm, args.num_tokens))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = sampler.sample(ids)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            st = dict(sampler.history_cache_stats)
            cell["arms"][arm] = {
                "ms": ms,
                "logp": mean_logp(model, result.sequences, gen_start),
                "gen_tokens": int(result.sequences.shape[1]) - gen_start,
                "ledger": st,
            }
            if arm in ("quality", "latency"):
                again = FastDLLMSampler(llm, config=_config(arm, args.num_tokens)).sample(ids)
                cell["arms"][arm]["repro_ok"] = bool(torch.equal(again.sequences, result.sequences))
        cells.append(cell)
        print(
            "cell %r | %s"
            % (
                prompt[:24],
                " | ".join(
                    "%s %dms %.4f" % (a, round(cell["arms"][a]["ms"]), cell["arms"][a]["logp"])
                    for a in ARMS
                ),
            ),
            flush=True,
        )

    base_logp = [c["arms"]["baseline"]["logp"] for c in cells]
    speedups = {
        arm: statistics.median([c["arms"]["baseline"]["ms"] / c["arms"][arm]["ms"] for c in cells])
        for arm in ARMS
    }
    quality_ok = {
        arm: all(c["arms"][arm]["logp"] >= b - _H1_DROP for c, b in zip(cells, base_logp))
        for arm in ARMS
    }
    candidates = [a for a in ARMS if quality_ok[a] and a != "baseline"]
    best_quality = max(candidates, key=lambda a: speedups[a]) if candidates else "baseline"
    fastest = max(ARMS, key=lambda a: speedups[a])
    stack_gain = sum(1 for c in cells if c["arms"]["latency"]["ms"] <= c["arms"]["spec"]["ms"])
    repro_ok = all(
        c["arms"][a].get("repro_ok", False) for c in cells for a in ("quality", "latency")
    )
    verdicts = {
        "speedup_vs_baseline": {a: speedups[a] for a in ARMS},
        "logp_parity": {a: quality_ok[a] for a in ARMS},
        "best_quality_arm": best_quality,
        "best_quality_speedup": speedups.get(best_quality, 1.0),
        "best_latency_arm": fastest,
        "best_latency_speedup": speedups[fastest],
        "best_latency_logp_delta": statistics.mean(
            [c["arms"][fastest]["logp"] - c["arms"]["baseline"]["logp"] for c in cells]
        ),
        "v3_stack_gain_cells": stack_gain,
        "v3_pass": stack_gain >= 2,
        "v4_repro_pass": repro_ok,
        "overall": (
            "supported"
            if repro_ok and stack_gain >= 2 and best_quality != "baseline"
            else "refuted"
        ),
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "ar": "AR043",
        "title": "stack consolidation: measured-best preset selection",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quick": bool(args.quick),
        "num_tokens": args.num_tokens,
        "num_prompts": len(cells),
        "arms": list(ARMS),
        "environment": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "driver": _gpu_driver_version(),
            "torch": torch.__version__,
            "model_path": str(_resolve_model_path(args.model)),
        },
        "cells": cells,
        "verdicts": verdicts,
    }
    write_results(payload, out)
    print("verdicts=%s -> %s" % (verdicts, out), flush=True)
    return 0 if verdicts["overall"] == "supported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
