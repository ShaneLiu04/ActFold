#!/usr/bin/env python3
"""Optimization #4 validation: fused Triton gather+select vs PyTorch.

The folded slow path merges cached parent FFN rows with recomputed child rows.
The unfused path materialises a gathered parent tensor (``index_select``) and
then runs ``torch.where``.  The fused kernel resolves the row and copies it in a
single memory pass.  This script sweeps (seq_len, hidden_dim) and reports the
latency and parity of both implementations.

Usage::

    python -m scripts.opt4_fused_bench --out results/optimization/opt4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from actfold.core.fused_ops import gather_select
from scripts.algo_experiments import time_forward

SEQ_LENS = [128, 512, 2048, 8192]
HIDDEN_DIMS = [1536, 4096, 8192]
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def unfused(
    parent: torch.Tensor, rows: torch.Tensor, child: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    gathered = parent.index_select(0, rows.reshape(-1)).reshape(child.shape)
    return torch.where(mask.unsqueeze(-1), gathered, child)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    dtype = DTYPES[args.dtype]
    device = "cuda"
    assert torch.cuda.is_available()

    results: list[dict[str, Any]] = []
    for seq in SEQ_LENS:
        for hidden in HIDDEN_DIMS:
            parent = torch.randn(seq, hidden, dtype=dtype, device=device)
            rows = torch.randint(0, seq, (1, seq), device=device)
            child = torch.randn(1, seq, hidden, dtype=dtype, device=device)
            mask = torch.rand(1, seq, device=device) > 0.3

            ref = unfused(parent, rows, child, mask)
            fused = gather_select(parent, rows, child, mask)
            max_err = (fused.float() - ref.float()).abs().max().item()

            torch_ms = time_forward(
                lambda: unfused(parent, rows, child, mask), warmup=5, reps=30
            ).mean
            fused_ms = time_forward(
                lambda: gather_select(parent, rows, child, mask), warmup=5, reps=30
            ).mean
            entry = {
                "seq_len": seq,
                "hidden_dim": hidden,
                "dtype": args.dtype,
                "torch_ms": torch_ms,
                "fused_ms": fused_ms,
                "speedup": torch_ms / max(fused_ms, 1e-9),
                "max_abs_err": max_err,
                "traffic_mb": seq * hidden * 2 * 4 / 1e6,
            }
            results.append(entry)
            print(
                f"  T={seq:<5} H={hidden:<5} torch={torch_ms:.4f}ms fused={fused_ms:.4f}ms "
                f"speedup={entry['speedup']:.2f}x max_err={max_err:.1e}",
                flush=True,
            )

    payload = {
        "meta": {"gpu": torch.cuda.get_device_name(0), "dtype": args.dtype},
        "results": results,
    }
    path = outdir / "opt4_fused.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
