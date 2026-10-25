"""AR035 smoke: does the gated arm run on the real checkpoint, and how many
verification rounds does it take vs the ungated composition arm?

Not a benchmark (no verdicts, no artifact). Quick host check only.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env  # noqa: E402

apply_hf_env()

import torch  # noqa: E402

from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig  # noqa: E402
from scripts.ar021_generation_loop_bench import (  # noqa: E402
    BLOCK,
    QUICK_NUM_TOKENS,
    SEED,
    _BenchDiffusionLLM,
    _load_model,
    _prepare_prompts,
    _resolve_model_path,
)

SMALL_BLOCK = 32


def _cfg(num_tokens: int, arm: str) -> FastDLLMSamplerConfig:
    base: dict[str, Any] = {
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": SMALL_BLOCK,
        "threshold": 0.95,
        "temperature": 0.0,
        "seed": SEED,
        "history_cache": True,
        "history_small_block_cache": False,
        "num_block_candidates": 4,
        "candidate_top_k": 10,
        "candidate_include_greedy": True,
    }
    if arm == "gated":
        base["speculative_commit_gate"] = True
    return FastDLLMSamplerConfig(**base)


def main() -> int:
    ap = argparse.ArgumentParser()
    add_hf_env_arguments(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--num-tokens", type=int, default=QUICK_NUM_TOKENS)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    model, tokenizer, meta = _load_model(_resolve_model_path(args.model))
    prompts = _prepare_prompts(tokenizer, quick=True)
    llm = _BenchDiffusionLLM(model, tokenizer)
    print(f"model={meta.get('model_path')} prompts={len(prompts)}", flush=True)

    for arm in ("spec", "gated"):
        sampler = FastDLLMSampler(llm, config=_cfg(args.num_tokens, arm))
        ids = prompts[0]["ids"][:, : prompts[0]["raw_len"]]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = sampler.sample(ids)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        gen = out.sequences[0, -args.num_tokens :]
        print(
            f"{arm}: ms={ms:.1f} gen_tokens={gen.numel()} stats={sampler.history_cache_stats}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
