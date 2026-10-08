#!/usr/bin/env python3
"""Optimization #3 validation: adaptive quantile gate vs fixed threshold.

For every model we sweep a fixed tau and an adaptive stable-ratio target on two
children (1 flipped token and 8 flipped tokens), recording the achieved stable
ratio, output fidelity (top-1 agreement / relative MSE) and latency.

Usage::

    python -m scripts.opt3_adaptive_bench --model fastdllm --out results/optimization/opt3/fastdllm
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
from actfold.core.adaptive_gate import AdaptiveQuantileGate  # noqa: E402
from actfold.core.vectorized_cache import VectorizedActivationCache  # noqa: E402
from actfold.models import load_model  # noqa: E402
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER  # noqa: E402
from scripts.algo_experiments import (  # noqa: E402
    MODEL_SPECS,
    PROMPTS,
    encode_prompt,
    make_child,
    time_forward,
)

FIXED_TAUS = [1.0, 0.999, 0.995, 0.99, 0.95, 0.8, 0.5]
ADAPTIVE_TARGETS = [1.0, 0.995, 0.99, 0.97, 0.95, 0.9, 0.8, 0.5]
FLIPS = [1, 8]


def run_condition(
    folded: FoldedModel,
    child: torch.Tensor,
    baseline: torch.Tensor,
    parent_branch: str,
    tag: str,
    reps: int = 8,
) -> dict[str, Any]:
    GLOBAL_STABILITY_PROFILER.reset_branch(f"{tag}_c")
    with torch.no_grad():
        out = folded(child, branch_id=f"{tag}_c", parent_branch_id=parent_branch, step_idx=0)
    profile = GLOBAL_STABILITY_PROFILER.get_profile(f"{tag}_c")
    latency = time_forward(
        lambda: folded(child, branch_id=f"{tag}_c", parent_branch_id=parent_branch, step_idx=0),
        warmup=2,
        reps=reps,
    ).mean
    base = baseline.float()
    return {
        "stable_ratio": profile.mean_stable_ratio if profile is not None else 0.0,
        "min_layer_stable": profile.min_stable_ratio if profile is not None else 0.0,
        "latency_ms": latency,
        "top1_agreement": (out.argmax(-1) == baseline.argmax(-1)).float().mean().item(),
        "rel_mse": F.mse_loss(out.float(), base).item() / max(base.var().item(), 1e-12),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--out", required=True)
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

    prompt = encode_prompt(model.tokenizer, PROMPTS[0], "cuda", max_len=96)
    children = {}
    baselines = {}
    for k in FLIPS:
        child, _ = make_child(prompt, model.vocab_size, k, seed=7)
        children[k] = child
        with torch.no_grad():
            baselines[k] = model.forward(child).detach()
    baseline_ms = time_forward(lambda: model.forward(children[1]), warmup=3, reps=15).mean

    payload: dict[str, Any] = {
        "meta": {
            "model_key": args.model,
            "label": spec["label"],
            "seq_len": int(prompt.shape[1]),
            "gpu": torch.cuda.get_device_name(0),
            "baseline_forward_ms": baseline_ms,
        },
        "fixed": {"tau": FIXED_TAUS, "target": None, "conditions": {}},
        "adaptive": {"tau": None, "target": ADAPTIVE_TARGETS, "conditions": {}},
    }

    # Fixed threshold sweep.
    folded = FoldedModel(
        model.model,
        cache=VectorizedActivationCache(max_entries_per_layer=65536, device="cuda"),
        gate=SimilarityGate(tau=0.95),
    )
    try:
        with torch.no_grad():
            folded(prompt, branch_id="fixed_p", step_idx=0)
        for k in FLIPS:
            entries = []
            for tau in FIXED_TAUS:
                folded.gate.set_tau(tau)
                entry = run_condition(
                    folded, children[k], baselines[k], "fixed_p", f"fixed_{k}_{tau}"
                )
                entry["tau"] = tau
                entries.append(entry)
                print(
                    f"  [fixed k={k}] tau={tau:<6} stable={entry['stable_ratio']:.3f} "
                    f"top1={entry['top1_agreement']:.3f} rel_mse={entry['rel_mse']:.3e}",
                    flush=True,
                )
            payload["fixed"]["conditions"][str(k)] = entries
    finally:
        folded.restore()

    # Adaptive target sweep.
    folded = FoldedModel(
        model.model,
        cache=VectorizedActivationCache(max_entries_per_layer=65536, device="cuda"),
        gate=AdaptiveQuantileGate(target_stable_ratio=0.97),
    )
    try:
        with torch.no_grad():
            folded(prompt, branch_id="adaptive_p", step_idx=0)
        for k in FLIPS:
            entries = []
            for target in ADAPTIVE_TARGETS:
                gate = folded.gate
                assert isinstance(gate, AdaptiveQuantileGate)
                gate.set_target_stable_ratio(target)
                entry = run_condition(
                    folded, children[k], baselines[k], "adaptive_p", f"adaptive_{k}_{target}"
                )
                entry["target"] = target
                entries.append(entry)
                print(
                    f"  [adaptive k={k}] target={target:<6} stable={entry['stable_ratio']:.3f} "
                    f"top1={entry['top1_agreement']:.3f} rel_mse={entry['rel_mse']:.3e}",
                    flush=True,
                )
            payload["adaptive"]["conditions"][str(k)] = entries
    finally:
        folded.restore()

    path = outdir / "opt3_adaptive.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
