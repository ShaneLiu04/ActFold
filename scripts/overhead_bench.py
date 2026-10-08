#!/usr/bin/env python3
"""Per-layer overhead decomposition for ActFold's folded forward path.

Measures, on a real diffusion LLM's first Transformer layer:

- original layer forward time
- folded layer time with all tokens stable (fast path)
- folded layer time with no tokens stable (full recompute path)
- similarity gate time
- activation cache ``put`` / ``get`` time
- stable/divergent merge time (Triton and PyTorch)

The decomposition explains why high stable ratios do not automatically
translate into wall-clock speedups in the current implementation.

Usage::

    python -m scripts.overhead_bench --model fastdllm --out results/experiments/fastdllm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402

from actfold.core import ActivationCache, SimilarityGate  # noqa: E402
from actfold.core.folded_transformer import FoldedTransformerLayer  # noqa: E402
from actfold.core.fused_ops import (  # noqa: E402
    _merge_stable_divergent_torch,
    merge_stable_divergent,
)
from actfold.models import load_model  # noqa: E402
from actfold.models.architecture_utils import detect_architecture  # noqa: E402
from scripts.algo_experiments import MODEL_SPECS, PROMPTS, encode_prompt, time_forward  # noqa: E402


def bench_model(key: str, dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
    spec = MODEL_SPECS[key]
    model = load_model(spec["repo"], model_family=spec["family"], torch_dtype=dtype)
    model.to("cuda")
    model.eval()
    raw = model.model
    tokens = encode_prompt(model.tokenizer, PROMPTS[0], "cuda", max_len=96)

    profile = detect_architecture(raw)
    layer0 = profile.layers[0]
    seq_len = int(tokens.shape[1])
    hidden = model.hidden_dim

    # Capture the real layer input/kwargs during one baseline forward pass.
    captured: dict[str, Any] = {}

    def pre_hook(module: Any, args: Any, kwargs: Any) -> None:
        if not captured:
            captured["args"] = [a.detach().clone() if torch.is_tensor(a) else a for a in args]
            captured["kwargs"] = {
                k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()
            }

    handle = layer0.register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.no_grad():
        model.forward(tokens)
    handle.remove()
    assert captured, "layer input capture failed"

    args = captured["args"]
    kwargs = captured["kwargs"]
    hidden_states = args[0]

    def run_orig() -> Any:
        return layer0(*args, **kwargs)

    orig_ms = time_forward(run_orig, warmup=3, reps=20).mean

    # Folded layer timings.
    gate = SimilarityGate(tau=0.95)
    cache = ActivationCache(max_entries_per_layer=65536, device="cuda")
    folded_layer = FoldedTransformerLayer(layer0, cache, gate, layer_idx=0).to("cuda")

    fold_kwargs = {
        k: v for k, v in kwargs.items() if k not in ("branch_id", "parent_branch_id", "step_idx")
    }

    parent_h = hidden_states.clone()
    with torch.no_grad():
        folded_layer(parent_h, branch_id="bench_parent", **fold_kwargs)

    stable_ms = time_forward(
        lambda: folded_layer(
            parent_h.clone(),
            branch_id="bench_child_stable",
            parent_branch_id="bench_parent",
            **fold_kwargs,
        ),
        warmup=2,
        reps=20,
    ).mean

    gate.tau = 1.0
    divergent_ms = time_forward(
        lambda: folded_layer(
            parent_h.clone() * 0.9,
            branch_id="bench_child_div",
            parent_branch_id="bench_parent",
            **fold_kwargs,
        ),
        warmup=2,
        reps=20,
    ).mean
    gate.tau = 0.95

    gate_ms = time_forward(
        lambda: gate(hidden_states, hidden_states * 0.999), warmup=5, reps=50
    ).mean

    put_cache = ActivationCache(max_entries_per_layer=65536, device="cuda")

    def do_put() -> None:
        put_cache.clear_layer(0)
        put_cache.put("bench", 0, {"ffn_out": hidden_states, "hidden_states": hidden_states})

    put_ms = time_forward(do_put, warmup=2, reps=20).mean

    get_cache = ActivationCache(max_entries_per_layer=65536, device="cuda")
    get_cache.put("bench", 0, {"ffn_out": hidden_states, "hidden_states": hidden_states})
    mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device="cuda")
    get_ms = time_forward(lambda: get_cache.get("bench", 0, mask), warmup=3, reps=20).mean

    parent = torch.randn_like(hidden_states)
    child = torch.randn_like(hidden_states)
    merge_mask = torch.rand(hidden_states.shape[:2], device="cuda") > 0.2
    merge_triton_ms = time_forward(
        lambda: merge_stable_divergent(parent, child, merge_mask), warmup=5, reps=50
    ).mean
    merge_torch_ms = time_forward(
        lambda: _merge_stable_divergent_torch(parent, child, merge_mask), warmup=5, reps=50
    ).mean

    model_ms = time_forward(lambda: model.forward(tokens), warmup=3, reps=15).mean

    return {
        "model": spec["label"],
        "seq_len": seq_len,
        "hidden_dim": hidden,
        "num_layers": model.num_layers,
        "baseline_full_forward_ms": model_ms,
        "original_layer_ms": orig_ms,
        "folded_all_stable_ms": stable_ms,
        "folded_all_divergent_ms": divergent_ms,
        "gate_ms": gate_ms,
        "cache_put_ms": put_ms,
        "cache_get_ms": get_ms,
        "merge_triton_ms": merge_triton_ms,
        "merge_torch_ms": merge_torch_ms,
        "estimated_model_speedup_at_full_reuse": (
            model_ms / (model_ms - (orig_ms - stable_ms) * (model.num_layers - 1))
            if orig_ms > stable_ms
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--out", type=str, required=True)
    add_hf_env_arguments(parser)
    args = parser.parse_args()

    torch.manual_seed(42)
    result = bench_model(args.model)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "overhead.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
