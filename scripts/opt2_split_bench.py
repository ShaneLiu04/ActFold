#!/usr/bin/env python3
"""Optimization #2 validation: split FFN (attention full, FFN divergent-only).

Compares folded forward latency and fidelity for:

- baseline full forward (no folding),
- folded layer without splitting (vectorized cache),
- folded layer with FFN splitting.

Two controlled sweeps vary the divergent-token count:

1. tau sweep on a 1-token-flip child.
2. k-flip sweep at tau=0.999, where exactly the flipped tokens are divergent.

Usage::

    python -m scripts.opt2_split_bench --model fastdllm --out results/optimization/opt2/fastdllm
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
from scripts.algo_experiments import (  # noqa: E402
    MODEL_SPECS,
    PROMPTS,
    encode_prompt,
    make_child,
    time_forward,
)

TAUS = [1.0, 0.999, 0.995, 0.99, 0.95, 0.8, 0.5]
FLIP_COUNTS = [0, 1, 2, 4, 8, 16, 32]


def build_folded(model: Any, split: bool) -> FoldedModel:
    cache = VectorizedActivationCache(max_entries_per_layer=65536, device="cuda")
    gate = SimilarityGate(tau=0.95)
    return FoldedModel(model.model, cache=cache, gate=gate, split_layers=split)


def run_condition(
    folded: FoldedModel,
    child: torch.Tensor,
    tau: float,
    tag: str,
    baseline: torch.Tensor,
    parent_branch: str,
    reps: int = 8,
) -> dict[str, Any]:
    folded.gate.set_tau(tau)
    GLOBAL_STABILITY_PROFILER.reset_branch(f"{tag}_c")
    with torch.no_grad():
        out = folded(child, branch_id=f"{tag}_c", parent_branch_id=parent_branch, step_idx=0)
    profile = GLOBAL_STABILITY_PROFILER.get_profile(f"{tag}_c")
    stable = profile.mean_stable_ratio if profile is not None else 0.0
    latency = time_forward(
        lambda: folded(child, branch_id=f"{tag}_c", parent_branch_id=parent_branch, step_idx=0),
        warmup=2,
        reps=reps,
    ).mean
    top1 = (out.argmax(-1) == baseline.argmax(-1)).float().mean().item()
    return {
        "tau": tau,
        "stable_ratio": stable,
        "divergent_fraction": 1.0 - stable,
        "latency_ms": latency,
        "mse": F.mse_loss(out.float(), baseline.float()).item(),
        "top1_agreement": top1,
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
    child, _ = make_child(prompt, model.vocab_size, 1, seed=42)
    with torch.no_grad():
        baseline_child = model.forward(child).float()
    baseline_ms = time_forward(lambda: model.forward(child), warmup=3, reps=15).mean

    # Pre-compute every k-flip child and its baseline BEFORE any wrapping: once
    # the model is wrapped by FoldedModel the raw module cannot be called
    # without a branch id.
    flip_children: dict[int, torch.Tensor] = {}
    flip_baselines: dict[int, torch.Tensor] = {}
    for k in FLIP_COUNTS:
        kk = min(k, prompt.shape[1])
        flip_child, _ = make_child(prompt, model.vocab_size, kk, seed=7)
        with torch.no_grad():
            flip_baselines[kk] = model.forward(flip_child).float()
        flip_children[kk] = flip_child

    payload: dict[str, Any] = {
        "meta": {
            "model_key": args.model,
            "label": spec["label"],
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "seq_len": int(prompt.shape[1]),
            "gpu": torch.cuda.get_device_name(0),
            "baseline_forward_ms": baseline_ms,
        },
        "tau_sweep": {},
        "flip_sweep": {},
        "equivalence": {},
    }

    for split in (False, True):
        name = "split" if split else "nosplit"
        folded = build_folded(model, split)
        active = sum(
            1 for layer in (folded._wrapped_layers or []) if getattr(layer, "split_enabled", False)
        )
        payload["meta"][f"{name}_layers_active"] = active
        try:
            with torch.no_grad():
                folded(prompt, branch_id=f"{name}_p", step_idx=0)

            # Phase A: tau sweep with a 1-flip child.
            payload["tau_sweep"][name] = [
                run_condition(folded, child, tau, f"{name}_tau{i}", baseline_child, f"{name}_p")
                for i, tau in enumerate(TAUS)
            ]

            # Phase B: k-flip sweep at tau=0.99 (only the flipped tokens diverge).
            flips = []
            for k in FLIP_COUNTS:
                kk = min(k, prompt.shape[1])
                entry = run_condition(
                    folded,
                    flip_children[kk],
                    0.99,
                    f"{name}_flip{kk}",
                    flip_baselines[kk],
                    f"{name}_p",
                )
                entry["num_flips"] = kk
                flips.append(entry)
            payload["flip_sweep"][name] = flips

            # Equivalence check at tau=0.95 (partial) and tau=1.0 (all divergent).
            with torch.no_grad():
                folded.gate.set_tau(0.95)
                out_partial = folded(
                    child, branch_id=f"{name}_eq", parent_branch_id=f"{name}_p", step_idx=0
                )
            payload["equivalence"][name] = {
                "partial_output": out_partial.float().cpu(),
            }
        finally:
            folded.restore()

    # Compare split vs nosplit outputs for exactness at divergent positions.
    out_split = payload["equivalence"]["split"].pop("partial_output")
    out_base = payload["equivalence"]["nosplit"].pop("partial_output")
    payload["equivalence"]["max_abs_diff_split_vs_nosplit"] = (
        (out_split - out_base).abs().max().item()
    )
    payload["equivalence"]["mse_split_vs_baseline"] = F.mse_loss(
        out_split, baseline_child.cpu()
    ).item()
    payload["equivalence"]["mse_nosplit_vs_baseline"] = F.mse_loss(
        out_base, baseline_child.cpu()
    ).item()
    print(
        f"equivalence: max|split-nosplit|={payload['equivalence']['max_abs_diff_split_vs_nosplit']:.3e} "
        f"mse split={payload['equivalence']['mse_split_vs_baseline']:.3e} "
        f"mse nosplit={payload['equivalence']['mse_nosplit_vs_baseline']:.3e}",
        flush=True,
    )
    for split in (False, True):
        name = "split" if split else "nosplit"
        print(f"--- {name} (active layers={payload['meta'][f'{name}_layers_active']}) ---")
        for entry in payload["tau_sweep"][name]:
            print(
                f"  tau={entry['tau']:<6} stable={entry['stable_ratio']:.3f} "
                f"div={entry['divergent_fraction']:.3f} lat={entry['latency_ms']:.2f}ms "
                f"top1={entry['top1_agreement']:.3f}",
                flush=True,
            )

    path = outdir / "opt2_split.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
