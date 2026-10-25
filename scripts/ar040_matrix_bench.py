"""AR040 matrix benchmark: is the ladder's advantage robust across prompts/lengths?

For every (prompt, length) cell run four arms:
  A  baseline        non-cached block-diffusion baseline (full recompute)
  D  cache-spec      AR034 composition + AR036 commit-draft fusion
  S  cache-adaptive  D + AR037 adaptive candidate budget (tau=0.5)
  V  cache-vfg       S + AR039 verification-free greedy commit

Frozen verdicts (declared before data collection):
  H1 quality   : V mean log p >= A mean log p - 0.05 on >= 90% of cells
  H2 speed     : median(A/V) >= 10x AND V < S < D on >= 80% of cells
  H3 ledger    : every V cell has free_verify_blocks == spec_fused_forwards,
                 spec_verify_forwards <= blocks, degraded_calls == 0
  H4 repro     : each V cell re-runs token-identically (else overall = blocked)

Quality metric is the same-model mean log p (AR037: never baseline agreement).
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig  # noqa: E402
from scripts.ar021_generation_loop_bench import (  # noqa: E402
    BLOCK,
    SEED,
    _BenchDiffusionLLM,
    _gpu_driver_version,
    _load_model,
    _resolve_model_path,
    write_results,
)

SCHEMA = "ar040_matrix_bench"
SCHEMA_VERSION = 1
ARMS = ("baseline", "cache-spec", "cache-adaptive", "cache-vfg")
_A, _D, _S, _V = ARMS
DEFAULT_OUTPUT = Path("results/optimization/ar040_matrix_bench.json")
QUICK_OUTPUT = Path("results/optimization/ar040_matrix_bench_quick.json")
FULL_LENGTHS = (160, 320)
QUICK_LENGTHS = (160,)
QUICK_PROMPTS = 3
_H1_DROP = 0.05
_H1_RATIO = 0.90
_H2_MEDIAN = 10.0
_H2_ORDER_RATIO = 0.80

PROMPTS: tuple[str, ...] = (
    "Write a long detailed technical explanation of how diffusion language models generate text. "
    "Cover masking schedules, block decoding, confidence thresholds, and KV caching.",
    "Explain the design of a modern vector database from first principles. Discuss indexing, "
    "recall trade-offs, segment compaction, and how queries are planned.",
    "Describe how a compiler lowers a high-level language to machine code. Cover parsing, "
    "intermediate representations, optimization passes, and register allocation.",
    "Walk through the internals of a distributed consensus protocol. Explain leader election, "
    "log replication, membership changes, and failure recovery.",
    "Explain how operating systems schedule threads on many-core machines. Cover run queues, "
    "priority inversion, NUMA effects, and real-time guarantees.",
    "Describe the lifecycle of an HTTP request in a large-scale service mesh. Cover DNS, TLS, "
    "load balancing, retries, and observability.",
)


def sampler_config(num_tokens: int, arm: str) -> FastDLLMSamplerConfig:
    base: dict[str, Any] = {
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": BLOCK,
        "threshold": 0.95,
        "temperature": 0.0,
        "seed": SEED,
        "stop_token_id": -1,
    }
    if arm != _A:
        base.update(
            history_cache=True,
            history_small_block_cache=False,
            num_block_candidates=4,
            candidate_top_k=10,
            candidate_include_greedy=True,
            commit_draft_fusion=True,
        )
    if arm in (_S, _V):
        base.update(adaptive_candidate_budget=True, candidate_budget_threshold=0.5)
    if arm == _V:
        base["verification_free_commit"] = True
    return FastDLLMSamplerConfig(**base)


def mean_logp(model: Any, seq: torch.Tensor, gen_start: int) -> float:
    out = model(
        input_ids=seq[-1:, :],
        use_cache=True,
        update_past_key_values=False,
        block_size=BLOCK,
    )
    logits = out.logits[0]
    total = 0.0
    n = 0
    for j in range(gen_start, seq.shape[1]):
        probs = F.softmax(logits[j - 1].float(), dim=-1)
        total += math.log(max(float(probs[int(seq[0, j])].item()), 1e-12))
        n += 1
    return total / max(n, 1)


def run_cell(llm: Any, model: Any, tokenizer: Any, prompt: str, length: int) -> dict[str, Any]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    gen_start = int(ids.shape[1])
    cell: dict[str, Any] = {"prompt_head": prompt[:40], "length": length, "arms": {}}
    for arm in ARMS:
        sampler = FastDLLMSampler(llm, config=sampler_config(length, arm))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = sampler.sample(ids)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        st = dict(sampler.history_cache_stats)
        cell["arms"][arm] = {
            "ms": ms,
            "gen_tokens": int(out.sequences.shape[1]) - gen_start,
            "logp": mean_logp(model, out.sequences, gen_start),
            "ledger": st,
        }
        if arm == _V:
            again = FastDLLMSampler(llm, config=sampler_config(length, arm)).sample(ids)
            cell["arms"][arm]["repro_ok"] = bool(torch.equal(again.sequences, out.sequences))
    return cell


def verdicts(cells: list[dict[str, Any]]) -> dict[str, Any]:
    ok_quality = 0
    ok_order = 0
    ratios: list[float] = []
    ledger_ok = True
    repro_ok = True
    for cell in cells:
        arms = cell["arms"]
        a, d, s, v = (arms[n] for n in ARMS)
        if v["logp"] >= a["logp"] - _H1_DROP:
            ok_quality += 1
        if v["ms"] < s["ms"] < d["ms"]:
            ok_order += 1
        ratios.append(a["ms"] / v["ms"] if v["ms"] else 0.0)
        vs = v["ledger"]
        blocks = vs["spec_verify_forwards"] + vs["spec_verification_free_blocks"]
        if not (
            vs["free_verify_blocks"] == vs["spec_fused_forwards"]
            and vs["spec_verify_forwards"] <= blocks
            and vs["degraded_calls"] == 0
        ):
            ledger_ok = False
        if not v.get("repro_ok", False):
            repro_ok = False
    n = len(cells)
    h1 = ok_quality / n >= _H1_RATIO
    h2 = statistics.median(ratios) >= _H2_MEDIAN and ok_order / n >= _H2_ORDER_RATIO
    h4 = repro_ok
    overall = "blocked" if not h4 else ("supported" if (h1 and h2 and ledger_ok) else "refuted")
    return {
        "h1_quality_cells": ok_quality,
        "h1_pass": h1,
        "h2_median_speedup": statistics.median(ratios),
        "h2_order_cells": ok_order,
        "h2_pass": h2,
        "h3_ledger_pass": ledger_ok,
        "h4_repro_pass": h4,
        "cells": n,
        "overall": overall,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    add_hf_env_arguments(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lengths", default=None, help="comma list, e.g. 160,320")
    ap.add_argument("--prompts", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out = args.out or (QUICK_OUTPUT if args.quick else DEFAULT_OUTPUT)
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out} (pass --force)")
        return 4
    lengths = (
        tuple(int(x) for x in args.lengths.split(","))
        if args.lengths
        else (QUICK_LENGTHS if args.quick else FULL_LENGTHS)
    )
    n_prompts = args.prompts or (QUICK_PROMPTS if args.quick else len(PROMPTS))
    prompts = PROMPTS[:n_prompts]

    model, tokenizer, meta = _load_model(_resolve_model_path(args.model))
    llm = _BenchDiffusionLLM(model, tokenizer)
    cells: list[dict[str, Any]] = []
    for prompt in prompts:
        for length in lengths:
            cells.append(run_cell(llm, model, tokenizer, prompt, length))
            last = cells[-1]["arms"]
            print(
                "cell prompt=%r len=%d ms A/D/S/V=%s logp=%s vfg_blocks=%d rearm=%d"
                % (
                    prompt[:24],
                    length,
                    [round(last[a]["ms"]) for a in ARMS],
                    [round(last[a]["logp"], 4) for a in ARMS],
                    last[_V]["ledger"]["spec_verification_free_blocks"],
                    last[_V]["ledger"]["free_verify_low_conf"],
                ),
                flush=True,
            )

    v = verdicts(cells)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "ar": "AR040",
        "title": "matrix benchmark: robustness of the ladder across prompts and lengths",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quick": bool(args.quick),
        "lengths": list(lengths),
        "num_prompts": len(prompts),
        "block_size": BLOCK,
        "seed": SEED,
        "arms": list(ARMS),
        "environment": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "driver": _gpu_driver_version(),
            "torch": torch.__version__,
            "model_path": str(_resolve_model_path(args.model)),
        },
        "thresholds": {
            "h1_logp_drop": _H1_DROP,
            "h1_cell_ratio": _H1_RATIO,
            "h2_median_speedup": _H2_MEDIAN,
            "h2_order_ratio": _H2_ORDER_RATIO,
        },
        "cells": cells,
        "verdicts": v,
    }
    write_results(payload, out)
    print("verdicts=%s -> %s" % (v, out), flush=True)
    return 0 if v["overall"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
