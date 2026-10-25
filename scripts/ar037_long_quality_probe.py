"""AR037 T001 probe: is there a multi-block quality divergence at long horizon?

Generates with EOS disabled (``stop_token_id=-1``) so the block loop runs to the
cap, compares the composition arm (D) against the non-cached baseline (A)
token-by-token, and measures the post-hoc confidence of D's committed positions
(the upper bound of what a self-correcting remask could revise).

Real checkpoint only; no synthetic tensors.
"""

from __future__ import annotations

import argparse
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


def _cfg(num_tokens: int, arm: str) -> FastDLLMSamplerConfig:
    base: dict[str, Any] = {
        "num_tokens": num_tokens,
        "block_size": BLOCK,
        "small_block_size": BLOCK,
        "threshold": 0.95,
        "temperature": 0.0,
        "seed": SEED,
        "stop_token_id": -1,
    }
    if arm == "D":
        base.update(
            history_cache=True,
            history_small_block_cache=False,
            num_block_candidates=4,
            candidate_top_k=10,
            candidate_include_greedy=True,
        )
    return FastDLLMSamplerConfig(**base)


def _confidence(model: Any, seq: torch.Tensor, gen_start: int) -> list[float]:
    """Post-hoc probability of each generated token under the block-causal model."""
    out = model(
        input_ids=seq,
        use_cache=True,
        update_past_key_values=False,
        block_size=BLOCK,
    )
    logits = out.logits[0]
    conf: list[float] = []
    for j in range(gen_start, seq.shape[1]):
        probs = F.softmax(logits[j - 1].float(), dim=-1)
        conf.append(float(probs[int(seq[0, j])].item()))
    return conf


def main() -> int:
    ap = argparse.ArgumentParser()
    add_hf_env_arguments(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--lengths", default="192")
    args = ap.parse_args()

    model, tokenizer, meta = _load_model(_resolve_model_path(args.model))
    llm = _BenchDiffusionLLM(model, tokenizer)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()
    gen_start = int(ids.shape[1])
    print(f"model={meta.get('model_path')} prompt_tokens={gen_start} B={BLOCK}", flush=True)

    for length in [int(x) for x in args.lengths.split(",")]:
        seqs: dict[str, torch.Tensor] = {}
        timings: dict[str, float] = {}
        for arm in ("A", "D", "D2"):
            cfg_arm = "D" if arm.startswith("D") else "A"
            sampler = FastDLLMSampler(llm, config=_cfg(length, cfg_arm))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = sampler.sample(ids)
            torch.cuda.synchronize()
            timings[arm] = (time.perf_counter() - t0) * 1000.0
            seqs[arm] = out.sequences
            if arm == "D":
                stats = sampler.history_cache_stats

        a, d = seqs["A"], seqs["D"]
        n = min(a.shape[1], d.shape[1])
        gen = n - gen_start
        same = int((a[0, gen_start:n] == d[0, gen_start:n]).sum())
        first_div = next((i for i in range(gen_start, n) if int(a[0, i]) != int(d[0, i])), None)
        per_block = []
        for b in range(0, gen, BLOCK):
            hi = min(b + BLOCK, gen)
            per_block.append(
                int(
                    (
                        a[0, gen_start + b : gen_start + hi] != d[0, gen_start + b : gen_start + hi]
                    ).sum()
                )
            )
        conf_a = _confidence(model, a, gen_start)
        conf = _confidence(model, d, gen_start)
        low = sum(1 for c in conf if c < 0.5)
        low_a = sum(1 for c in conf_a if c < 0.5)
        logp_a = sum(__import__("math").log(max(c, 1e-12)) for c in conf_a) / max(len(conf_a), 1)
        logp_d = sum(__import__("math").log(max(c, 1e-12)) for c in conf) / max(len(conf), 1)
        repro = bool(torch.equal(seqs["D"], seqs["D2"]))
        print(
            f"len={length} genA={int(a.shape[1]) - gen_start} genD={d.shape[1] - gen_start} "
            f"agreement={same}/{gen}={same / gen if gen else 0:.4f} "
            f"first_div={first_div} per_block_div={per_block} "
            f"confA={sum(conf_a) / max(len(conf_a), 1):.4f} confD={sum(conf) / max(len(conf), 1):.4f} "
            f"lowA={low_a} lowD={low} logpA={logp_a:.4f} logpD={logp_d:.4f} repro_D={repro} "
            f"tA={timings['A']:.0f}ms tD={timings['D']:.0f}ms "
            f"D_stats=draft{stats['spec_draft_forwards']}/fused{stats['spec_fused_forwards']}/"
            f"verify{stats['spec_verify_forwards']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
