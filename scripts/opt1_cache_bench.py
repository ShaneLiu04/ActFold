#!/usr/bin/env python3
"""Optimization #1 validation: vectorized activation cache vs legacy/per-token.

Measures
--------
1. Microbenchmark: ``put``/``get`` latency for legacy, chunked, and vectorized
   caches across sequence lengths.
2. End-to-end: folded forward latency (all-stable fast path and partial-stable
   path) and output parity for each cache implementation.

Usage::

    python -m scripts.opt1_cache_bench --model fastdllm --out results/optimization/opt1/fastdllm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from actfold.core import (  # noqa: E402
    ActivationCache,
    ChunkedActivationCache,
    FoldedModel,
    SimilarityGate,
)
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

CACHE_BUILDERS: dict[str, Callable[[int], Any]] = {
    "legacy": lambda cap: ActivationCache(max_entries_per_layer=cap, device="cuda"),
    "chunked": lambda cap: ChunkedActivationCache(
        max_entries_per_layer=cap, chunk_size=64, device="cuda"
    ),
    "vectorized": lambda cap: VectorizedActivationCache(max_entries_per_layer=cap, device="cuda"),
}


def time_cpu(fn: Callable[[], Any], warmup: int = 3, reps: int = 30) -> float:
    """Mean CPU wall-clock ms per call (with a final CUDA sync for hybrid ops)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1000.0


def micro_bench(model: Any, outdir: Path) -> dict[str, Any]:
    hidden = model.hidden_dim
    dtype = torch.bfloat16
    lengths = [34, 41, 96, 256, 512]
    results: dict[str, Any] = {}
    for seq in lengths:
        acts = {
            "ffn_out": torch.randn(1, seq, hidden, dtype=dtype, device="cuda"),
            "hidden_states": torch.randn(1, seq, hidden, dtype=dtype, device="cuda"),
        }
        mask = torch.ones(1, seq, dtype=torch.bool, device="cuda")
        entry: dict[str, Any] = {"seq_len": seq}
        for name, builder in CACHE_BUILDERS.items():
            cache = builder(65536)
            cache.put("b", 0, acts)

            def do_put(c: Any = cache, a: dict[str, torch.Tensor] = acts) -> None:
                c.put("b", 0, a)

            def do_get(c: Any = cache, m: torch.Tensor = mask) -> Any:
                return c.get("b", 0, m)

            put_ms = time_cpu(do_put, reps=30)
            get_ms = time_cpu(do_get, reps=30)
            entry[f"{name}_put_ms"] = put_ms
            entry[f"{name}_get_ms"] = get_ms
        entry["put_speedup_legacy_over_vec"] = entry["legacy_put_ms"] / max(
            entry["vectorized_put_ms"], 1e-9
        )
        entry["get_speedup_legacy_over_vec"] = entry["legacy_get_ms"] / max(
            entry["vectorized_get_ms"], 1e-9
        )
        results[str(seq)] = entry
        print(
            f"  T={seq:<4} put legacy={entry['legacy_put_ms']:.3f}ms "
            f"vec={entry['vectorized_put_ms']:.3f}ms ({entry['put_speedup_legacy_over_vec']:.1f}x) | "
            f"get legacy={entry['legacy_get_ms']:.3f}ms vec={entry['vectorized_get_ms']:.3f}ms "
            f"({entry['get_speedup_legacy_over_vec']:.1f}x)",
            flush=True,
        )
    return results


def e2e_bench(model: Any, outdir: Path) -> dict[str, Any]:
    prompt = encode_prompt(model.tokenizer, PROMPTS[0], "cuda", max_len=96)
    child, _ = make_child(prompt, model.vocab_size, 1, seed=42)
    with torch.no_grad():
        baseline_parent = model.forward(prompt).float()
        baseline_child = model.forward(child).float()

    results: dict[str, Any] = {}
    for name, builder in CACHE_BUILDERS.items():
        cache = builder(65536)
        gate = SimilarityGate(tau=0.95)
        folded = FoldedModel(model.model, cache=cache, gate=gate)
        try:
            # Parent populates the cache.
            with torch.no_grad():
                folded(prompt, branch_id=f"{name}_p", step_idx=0)

            # All-stable fast path: child == parent.
            GLOBAL_STABILITY_PROFILER.reset_branch(f"{name}_self")
            with torch.no_grad():
                self_out = folded(
                    prompt, branch_id=f"{name}_self", parent_branch_id=f"{name}_p", step_idx=0
                )
            profile = GLOBAL_STABILITY_PROFILER.get_profile(f"{name}_self")
            self_ms = time_forward(
                lambda: folded(
                    prompt, branch_id=f"{name}_self", parent_branch_id=f"{name}_p", step_idx=0
                ),
                warmup=2,
                reps=10,
            ).mean

            # Partial-stable path: one token flipped.
            with torch.no_grad():
                part_out = folded(
                    child, branch_id=f"{name}_part", parent_branch_id=f"{name}_p", step_idx=0
                )
            partial_ms = time_forward(
                lambda: folded(
                    child, branch_id=f"{name}_part", parent_branch_id=f"{name}_p", step_idx=0
                ),
                warmup=2,
                reps=10,
            ).mean
        finally:
            folded.restore()

        results[name] = {
            "self_fold_ms": self_ms,
            "partial_ms": partial_ms,
            "self_fold_mse": F.mse_loss(self_out.float(), baseline_parent).item(),
            "partial_mse": F.mse_loss(part_out.float(), baseline_child).item(),
            "self_fold_stable": profile.mean_stable_ratio if profile else None,
            "num_cache_entries": cache.num_entries(),
        }
        print(
            f"  [{name}] all-stable={self_ms:.2f}ms partial={partial_ms:.2f}ms "
            f"mse_self={results[name]['self_fold_mse']:.3e} entries={cache.num_entries()}",
            flush=True,
        )

    results["speedup_self_fold_legacy_over_vec"] = results["legacy"]["self_fold_ms"] / max(
        results["vectorized"]["self_fold_ms"], 1e-9
    )
    results["speedup_partial_legacy_over_vec"] = results["legacy"]["partial_ms"] / max(
        results["vectorized"]["partial_ms"], 1e-9
    )
    results["baseline_forward_ms"] = time_forward(
        lambda: model.forward(prompt), warmup=3, reps=15
    ).mean
    return results


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

    payload: dict[str, Any] = {
        "meta": {
            "model_key": args.model,
            "label": spec["label"],
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        }
    }
    print("--- microbenchmark put/get ---", flush=True)
    payload["micro"] = micro_bench(model, outdir)
    print("--- end-to-end folded forward ---", flush=True)
    payload["e2e"] = e2e_bench(model, outdir)

    path = outdir / "opt1_cache.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
