"""AR044 probe: does refine-after-fill recover quality at >= 5x speed?

Per prompt: kv-only (quality reference, log p == baseline), latency preset (speed
reference) and refine-after-fill at several pass counts.
"""

from __future__ import annotations

import argparse
import time
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
    _load_model,
    _resolve_model_path,
)
from scripts.ar040_matrix_bench import PROMPTS, mean_logp  # noqa: E402


def _cfg(arm: str, num_tokens: int, passes: int) -> Any:
    common: dict[str, Any] = {
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": BLOCK,
        "threshold": 0.95,
        "temperature": 0.0,
        "seed": SEED,
        "stop_token_id": -1,
    }
    if arm == "kv":
        return fastdllm_preset("quality", **common)
    if arm == "latency":
        return fastdllm_preset("latency", **common)
    return fastdllm_preset(
        "baseline",
        history_cache=True,
        num_block_candidates=4,
        candidate_top_k=10,
        candidate_include_greedy=True,
        commit_refine_passes=passes,
        **common,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    add_hf_env_arguments(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--num-tokens", type=int, default=160)
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--passes", default="1,2,3")
    args = ap.parse_args()

    model, tokenizer, meta = _load_model(_resolve_model_path(args.model))
    llm = _BenchDiffusionLLM(model, tokenizer)
    passes = [int(x) for x in args.passes.split(",")]
    print("model=%s passes=%s" % (meta.get("model_path"), passes), flush=True)

    for idx, prompt in enumerate(PROMPTS[: args.prompts]):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        gen_start = int(ids.shape[1])
        row: dict[str, dict[str, float]] = {}
        for arm, p in [("kv", 0), ("latency", 0)] + [(f"R{n}", n) for n in passes]:
            kind = "kv" if arm == "kv" else ("latency" if arm == "latency" else "raf")
            sampler = FastDLLMSampler(llm, config=_cfg(kind, args.num_tokens, p))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = sampler.sample(ids)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            st = sampler.history_cache_stats
            row[arm] = {
                "ms": ms,
                "logp": mean_logp(model, out.sequences, gen_start),
                "refine": st.get("spec_refine_forwards", 0),
                "gen": int(out.sequences.shape[1]) - gen_start,
            }
        print(
            "p%d gen=%d | %s"
            % (
                idx + 1,
                row["kv"]["gen"],
                " | ".join(
                    "%s %dms %.4f rf=%d"
                    % (a, round(row[a]["ms"]), row[a]["logp"], row[a]["refine"])
                    for a in row
                ),
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
