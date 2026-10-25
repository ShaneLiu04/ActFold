"""AR037 ACB probe: how often does the confidence gate skip verification, and
what does it cost/earn on the real checkpoint at long-horizon geometry?

Arms: D (AR034/36 composition) vs S (D + adaptive_candidate_budget).
Reports skip rate, forward counts, wall clock and mean log p (quality proxy,
recomputed under the same model).
"""

from __future__ import annotations

import argparse
import math
import time
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
    _load_model,
    _resolve_model_path,
)

PROMPT = (
    "Write a long detailed technical explanation of how diffusion language models "
    "generate text. Cover masking schedules, block decoding, confidence thresholds, "
    "and why KV caching matters for inference latency."
)


def _cfg(
    num_tokens: int,
    arm: str,
    threshold: float,
    fusion: bool,
    escalate: bool = False,
    esc_threshold: float = 0.9,
    esc_candidates: int = 8,
    vfg: bool = False,
) -> FastDLLMSamplerConfig:
    base: dict[str, Any] = {
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": BLOCK,
        "threshold": 0.95,
        "temperature": 0.0,
        "seed": SEED,
        "stop_token_id": -1,
        "history_cache": True,
        "history_small_block_cache": False,
        "num_block_candidates": 4,
        "candidate_top_k": 10,
        "candidate_include_greedy": True,
    }
    if arm == "S":
        base["adaptive_candidate_budget"] = True
        base["candidate_budget_threshold"] = threshold
    if fusion:
        base["commit_draft_fusion"] = True
    if arm == "V":
        base["verification_free_commit"] = True
    if escalate:
        base["candidate_budget_escalate"] = True
        base["escalate_threshold"] = esc_threshold
        base["escalate_candidates"] = esc_candidates
    return FastDLLMSamplerConfig(**base)


def _mean_logp(model: Any, seq: torch.Tensor, gen_start: int) -> float:
    out = model(
        input_ids=seq,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    add_hf_env_arguments(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--num-tokens", type=int, default=192)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--fusion", action="store_true")
    ap.add_argument("--escalate", action="store_true")
    ap.add_argument("--escalate-threshold", type=float, default=0.9)
    ap.add_argument("--escalate-candidates", type=int, default=8)
    ap.add_argument("--vfg", action="store_true")
    args = ap.parse_args()

    model, tokenizer, meta = _load_model(_resolve_model_path(args.model))
    llm = _BenchDiffusionLLM(model, tokenizer)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()
    gen_start = int(ids.shape[1])
    print(f"model={meta.get('model_path')} gen_start={gen_start}", flush=True)

    arms = ("D", "S", "V") if args.vfg else ("D", "S")
    for arm in arms:
        sampler = FastDLLMSampler(
            llm,
            config=_cfg(
                args.num_tokens,
                arm,
                args.threshold,
                args.fusion,
                args.escalate,
                args.escalate_threshold,
                args.escalate_candidates,
                args.vfg,
            ),
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = sampler.sample(ids)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        st = sampler.history_cache_stats
        logp = _mean_logp(model, out.sequences, gen_start)
        skip_rate = st["spec_budget_skips"] / max(
            st["spec_budget_skips"] + st["spec_verify_forwards"], 1
        )
        print(
            f"arm={arm} ms={ms:.0f} gen={int(out.sequences.shape[1]) - gen_start} "
            f"logp={logp:.4f} skips={st['spec_budget_skips']} verify={st['spec_verify_forwards']} "
            f"draft={st['spec_draft_forwards']} fused={st['spec_fused_forwards']} "
            f"skip_rate={skip_rate:.4f} escalations={st['spec_budget_escalations']} "
            f"degraded={st['degraded_calls']} vfg={st['spec_verification_free_blocks']} "
            f"fv={st['free_verify_blocks']}/{st['free_verify_low_conf']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
