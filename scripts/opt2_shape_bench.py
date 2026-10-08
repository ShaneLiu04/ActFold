#!/usr/bin/env python3
"""Optimization #2 shape sweep: where does FFN splitting actually pay off?

The split path needs a data-dependent row gather per layer (one device sync).
Its benefit scales with the number of divergent tokens and the hidden size, so
this script sweeps (batch, seq_len, divergent fraction) on a real LLaDA decoder
layer and reports the per-layer speedup of split vs full recompute.

Usage::

    python -m scripts.opt2_shape_bench --out results/optimization/opt2/shape
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402

from actfold.core import SimilarityGate  # noqa: E402
from actfold.core.activation_cache import ActivationCache  # noqa: E402
from actfold.core.split_layer import SplitFoldedTransformerLayer  # noqa: E402
from actfold.models import load_model  # noqa: E402
from actfold.models.architecture_utils import detect_architecture  # noqa: E402
from scripts.algo_experiments import time_forward  # noqa: E402


class FixedMaskGate(SimilarityGate):
    """Gate that returns a preset stability mask computed from token index."""

    def __init__(self) -> None:
        super().__init__(tau=0.95)
        self.divergent_count = 0
        self.seq_len = 0
        self.batch_size = 0

    def forward(self, h_child: torch.Tensor, h_parent: torch.Tensor) -> torch.Tensor:
        del h_parent
        batch, seq = h_child.shape[:2]
        index = torch.arange(seq, device=h_child.device).unsqueeze(0).expand(batch, -1)
        return index >= self.divergent_count


def bench_shape(
    layer: Any,
    split_layer: SplitFoldedTransformerLayer,
    hidden: int,
    batch: int,
    seq: int,
    div_frac: float,
    dtype: torch.dtype,
) -> dict[str, Any]:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    div_count = max(1, int(round(seq * div_frac))) if div_frac < 1.0 else seq
    if div_frac <= 0.0:
        div_count = 0
    stable_mask = torch.ones(batch, seq, dtype=torch.bool, device="cuda")
    stable_mask[:, :div_count] = False

    base_ms = time_forward(lambda: split_layer.original_layer(x), warmup=3, reps=10).mean
    split_ms = time_forward(
        lambda: split_layer._recompute_merged(x, None, stable_mask),
        warmup=3,
        reps=10,
    ).mean
    return {
        "batch": batch,
        "seq_len": seq,
        "divergent_fraction": div_count / max(seq, 1),
        "divergent_tokens": div_count * batch,
        "full_layer_ms": base_ms,
        "split_layer_ms": split_ms,
        "speedup": base_ms / max(split_ms, 1e-9),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--repo", default="GSAI-ML/LLaDA-8B-Instruct", help="Model used for the layer"
    )
    add_hf_env_arguments(parser)
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    print("loading LLaDA for layer-level sweep", flush=True)
    model = load_model(args.repo, model_family="llada", torch_dtype=torch.bfloat16)
    model.to("cuda")
    profile = detect_architecture(model.model)
    layer = profile.layers[0]
    gate = FixedMaskGate()
    split_layer = SplitFoldedTransformerLayer(
        layer,
        ActivationCache(max_entries_per_layer=8, device="cuda"),
        gate,
        layer_idx=0,
        min_split_tokens=0,
    )
    assert split_layer.split_enabled, "LLaDA layer split spec not detected"

    results = []
    for batch in (1, 2, 4):
        for seq in (64, 128, 256, 512, 1024, 2048):
            for div_frac in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
                entry = bench_shape(
                    layer, split_layer, model.hidden_dim, batch, seq, div_frac, torch.bfloat16
                )
                results.append(entry)
                print(
                    f"  B={batch} T={seq:<5} div={entry['divergent_fraction']:.2f} "
                    f"full={entry['full_layer_ms']:.3f}ms split={entry['split_layer_ms']:.3f}ms "
                    f"speedup={entry['speedup']:.2f}x",
                    flush=True,
                )
    payload = {
        "meta": {
            "model": args.repo,
            "hidden_dim": model.hidden_dim,
            "gpu": torch.cuda.get_device_name(0),
        },
        "results": results,
    }
    path = outdir / "opt2_shape.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
