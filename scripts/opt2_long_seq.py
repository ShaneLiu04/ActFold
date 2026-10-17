#!/usr/bin/env python3
"""Optimization #2 end-to-end check at longer sequences.

The hook-based FFN split pays a per-layer device sync, so it only wins when the
FFN work saved per layer exceeds that constant.  This script repeats the
folded-forward comparison at seq_len=512 with several divergent-token counts,
using the vectorized cache and the split auto-enable threshold.

Usage::

    python -m scripts.opt2_long_seq --model fastdllm --out results/optimization/opt2/longseq/fastdllm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from actfold.core import FoldedModel, SimilarityGate  # noqa: E402
from actfold.core.vectorized_cache import VectorizedActivationCache  # noqa: E402
from actfold.models import load_model  # noqa: E402
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER  # noqa: E402
from scripts.algo_experiments import MODEL_SPECS, make_child, time_forward  # noqa: E402

LONG_TEXT = (
    "Diffusion language models generate text by iteratively denoising a masked "
    "sequence, and speculative decoding accelerates them by verifying several "
    "candidate branches in parallel. Branch folding reuses parent activations "
    "for tokens whose hidden states remain similar, which reduces the amount of "
    "compute required during verification. "
) * 4

FLIPS = [1, 8, 32, 128]


def encode_long(tokenizer: Any, target: int, device: str) -> torch.Tensor:
    ids = tokenizer(LONG_TEXT, return_tensors="pt").input_ids
    if ids.shape[1] < target:
        reps = (target // ids.shape[1]) + 1
        ids = ids.repeat(1, reps)
    ids = ids[:, :target]
    return ids.to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    add_hf_env_arguments(parser)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    print(f"=== loading {spec['label']} ===", flush=True)
    model = load_model(spec["repo"], model_family=spec["family"], torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()

    prompt = encode_long(model.tokenizer, args.seq_len, "cuda")
    children = {}
    baselines = {}
    for k in FLIPS:
        child, _ = make_child(prompt, model.vocab_size, k, seed=7)
        children[k] = child
        with torch.no_grad():
            # Keep bf16 logits on the device (a 512-token fp32 vocab tensor is
            # 250+ MB for these vocabularies).
            baselines[k] = model.forward(child).detach()

    baseline_ms = time_forward(lambda: model.forward(children[FLIPS[0]]), warmup=3, reps=10).mean
    payload: dict[str, Any] = {
        "meta": {
            "model_key": args.model,
            "label": spec["label"],
            "seq_len": int(prompt.shape[1]),
            "gpu": torch.cuda.get_device_name(0),
            "baseline_forward_ms": baseline_ms,
        },
        "conditions": {},
    }

    for split in (False, True):
        name = "split" if split else "nosplit"
        folded = FoldedModel(
            model.model,
            cache=VectorizedActivationCache(max_entries_per_layer=65536, device="cuda"),
            gate=SimilarityGate(tau=0.99),
            split_layers=split,
            split_min_tokens=512,
        )
        active = sum(
            1 for layer in (folded._wrapped_layers or []) if getattr(layer, "split_enabled", False)
        )
        try:
            with torch.no_grad():
                folded(prompt, branch_id=f"{name}_p", step_idx=0)
            entries = []
            for k in FLIPS:
                GLOBAL_STABILITY_PROFILER.reset_branch(f"{name}_{k}_c")
                with torch.no_grad():
                    out = folded(
                        children[k],
                        branch_id=f"{name}_{k}_c",
                        parent_branch_id=f"{name}_p",
                        step_idx=0,
                    )
                profile = GLOBAL_STABILITY_PROFILER.get_profile(f"{name}_{k}_c")
                latency = time_forward(
                    lambda: folded(
                        children[k],
                        branch_id=f"{name}_{k}_c",
                        parent_branch_id=f"{name}_p",
                        step_idx=0,
                    ),
                    warmup=2,
                    reps=8,
                ).mean
                entry = {
                    "num_flips": k,
                    "stable_ratio": profile.mean_stable_ratio if profile else 0.0,
                    "latency_ms": latency,
                    "speedup_vs_baseline": baseline_ms / max(latency, 1e-9),
                    "top1_agreement": (out.argmax(-1) == baselines[k].argmax(-1))
                    .float()
                    .mean()
                    .item(),
                    "rel_mse": F.mse_loss(out.float(), baselines[k]).item()
                    / max(baselines[k].var().item(), 1e-12),
                }
                entries.append(entry)
                print(
                    f"  [{name}] k={k:<4} stable={entry['stable_ratio']:.3f} "
                    f"lat={entry['latency_ms']:.1f}ms speedup={entry['speedup_vs_baseline']:.2f}x "
                    f"top1={entry['top1_agreement']:.3f}",
                    flush=True,
                )
            payload["conditions"][name] = {"active_layers": active, "entries": entries}
        finally:
            folded.restore()

    path = outdir / "opt2_longseq.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
