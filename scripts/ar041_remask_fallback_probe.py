"""AR041 probe: does the low-confidence remask fallback restore baseline quality?

Runs A (official iterative remasking), D (greedy whole-block commit + CDF) and
R (D + remask_fallback) on the AR040 prompt set at one length, reporting wall
clock and the same-model mean log p of the generated region.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402

from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig  # noqa: E402
from scripts.ar021_generation_loop_bench import (  # noqa: E402
    BLOCK,
    SEED,
    _BenchDiffusionLLM,
    _load_model,
    _resolve_model_path,
)
from scripts.ar040_matrix_bench import PROMPTS, mean_logp  # noqa: E402


def _cfg(num_tokens: int, arm: str, threshold: float) -> FastDLLMSamplerConfig:
    base: dict[str, Any] = {
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": BLOCK,
        "threshold": 0.95,
        "temperature": 0.0,
        "seed": SEED,
        "stop_token_id": -1,
    }
    if arm != "A":
        base.update(
            history_cache=True,
            history_small_block_cache=False,
            num_block_candidates=4,
            candidate_top_k=10,
            candidate_include_greedy=True,
            commit_draft_fusion=True,
        )
    if arm == "R":
        base.update(remask_fallback=True, remask_fallback_threshold=threshold)
    return FastDLLMSamplerConfig(**base)


def main() -> int:
    ap = argparse.ArgumentParser()
    add_hf_env_arguments(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--num-tokens", type=int, default=160)
    ap.add_argument("--thresholds", default="0.5,0.3,0.2,0.1")
    ap.add_argument("--prompts", type=int, default=3)
    args = ap.parse_args()

    model, tokenizer, meta = _load_model(_resolve_model_path(args.model))
    llm = _BenchDiffusionLLM(model, tokenizer)
    print(f"model={meta.get('model_path')} prompts={args.prompts}", flush=True)

    for idx, prompt in enumerate(PROMPTS[: args.prompts]):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        gen_start = int(ids.shape[1])
        thresholds = [float(x) for x in args.thresholds.split(",")]
        row: dict[str, Any] = {}
        for arm in ["A", "D"] + [f"R{t}" for t in thresholds]:
            if arm.startswith("R"):
                thr = float(arm[1:])
                cfg = _cfg(args.num_tokens, "R", thr)
            else:
                cfg = _cfg(args.num_tokens, arm, 0.5)
            sampler = FastDLLMSampler(llm, config=cfg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = sampler.sample(ids)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            st = sampler.history_cache_stats
            row[arm] = {
                "ms": ms,
                "logp": mean_logp(model, out.sequences, gen_start),
                "gen": int(out.sequences.shape[1]) - gen_start,
                "fallbacks": st.get("spec_remask_fallbacks", 0),
            }
        parts = ["p%d" % (idx + 1)]
        for arm in row:
            parts.append(
                "%s ms=%d logp=%.4f fb=%d"
                % (arm, round(row[arm]["ms"]), row[arm]["logp"], row[arm]["fallbacks"])
            )
        print(" | ".join(parts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
