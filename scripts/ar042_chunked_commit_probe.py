"""AR042 probe: quality/speed Pareto of chunked (progressive) in-block commit.

Per prompt: run A (official iteration), D (one-shot greedy commit + CDF) and the
chunked arm at several ``commit_chunk_size`` values. Reports wall clock, mean
log p and forward counts so the Pareto point can be read off directly.
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


def _cfg(num_tokens: int, arm: str, chunk: int) -> FastDLLMSamplerConfig:
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
    if chunk > 0:
        base["commit_chunk_size"] = chunk
    return FastDLLMSamplerConfig(**base)


def main() -> int:
    ap = argparse.ArgumentParser()
    add_hf_env_arguments(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--num-tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--chunks", default="4,8,16,32")
    args = ap.parse_args()

    model, tokenizer, meta = _load_model(_resolve_model_path(args.model))
    llm = _BenchDiffusionLLM(model, tokenizer)
    chunks = [int(x) for x in args.chunks.split(",")]
    print(f"model={meta.get('model_path')} chunks={chunks}", flush=True)

    for idx, prompt in enumerate(PROMPTS[: args.prompts]):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        gen_start = int(ids.shape[1])
        row: dict[str, dict[str, float]] = {}
        for arm, chunk in [("A", 0), ("D", 0)] + [(f"C{c}", c) for c in chunks]:
            cfg = _cfg(args.num_tokens, "A" if arm == "A" else "D", chunk)
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
                "chunk_fwd": st.get("spec_chunk_forwards", 0),
                "gen": int(out.sequences.shape[1]) - gen_start,
            }
        parts = ["p%d gen=%d" % (idx + 1, row["A"]["gen"])]
        for arm in row:
            parts.append(
                "%s %dms %.4f cf=%d"
                % (arm, round(row[arm]["ms"]), row[arm]["logp"], row[arm]["chunk_fwd"])
            )
        print(" | ".join(parts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
