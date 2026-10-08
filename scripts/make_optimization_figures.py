#!/usr/bin/env python3
"""Figures for the ActFold optimization phase (opt #1-#4).

Reads ``results/optimization/**/*.json`` and writes PNGs to
``results/optimization/figures/``.

Usage::

    python scripts/make_optimization_figures.py --root results/optimization --out results/optimization/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODEL_ORDER = ["fastdllm", "llada", "dream"]
COLORS = {"fastdllm": "#1f77b4", "llada": "#d62728", "dream": "#2ca02c"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fig_opt1(root: Path, out: Path) -> None:
    models = [m for m in MODEL_ORDER if (root / "opt1" / m / "opt1_cache.json").exists()]
    if not models:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    first = load_json(root / "opt1" / models[0] / "opt1_cache.json")
    taus = sorted(int(k) for k in first["micro"])
    for key in models:
        data = load_json(root / "opt1" / key / "opt1_cache.json")
        label = data["meta"]["label"]
        color = COLORS.get(key, "gray")
        put = [data["micro"][str(t)]["put_speedup_legacy_over_vec"] for t in taus]
        get = [data["micro"][str(t)]["get_speedup_legacy_over_vec"] for t in taus]
        axes[0].plot(taus, put, marker="o", color=color, label=label)
        axes[1].plot(taus, get, marker="o", color=color, label=label)
        e2e = data["e2e"]
        names = ["baseline_forward_ms", "legacy", "chunked", "vectorized"]
        values = [
            e2e["baseline_forward_ms"],
            e2e["legacy"]["self_fold_ms"],
            e2e["chunked"]["self_fold_ms"],
            e2e["vectorized"]["self_fold_ms"],
        ]
        x = np.arange(len(names)) + 0.22 * models.index(key)
        axes[2].bar(x, values, 0.2, color=color, label=label)
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("Cache put speedup vs legacy")
    axes[0].set_ylabel("speedup (x)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("Cache get speedup vs legacy")
    axes[1].set_ylabel("speedup (x)")
    for ax in axes[:2]:
        ax.set_xlabel("sequence length")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[2].set_xticks(np.arange(4) + 0.22)
    axes[2].set_xticklabels(
        ["baseline\n(no folding)", "legacy", "chunked", "vectorized"], fontsize=8
    )
    axes[2].set_title("All-stable folded forward latency")
    axes[2].set_ylabel("ms")
    axes[2].grid(alpha=0.3, axis="y")
    axes[2].legend(fontsize=8)
    fig.suptitle("Optimization #1: vectorized activation cache", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_opt1_cache.png", dpi=160)
    plt.close(fig)


def fig_opt2_shape(root: Path, out: Path) -> None:
    path = root / "opt2" / "shape" / "opt2_shape.json"
    if not path.exists():
        return
    data = load_json(path)
    results = data["results"]
    batches = sorted({r["batch"] for r in results})
    divs = sorted({r["divergent_fraction"] for r in results})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    cmap = plt.get_cmap("viridis")
    for j, batch in enumerate(batches):
        for i, div in enumerate(divs):
            rows = sorted(
                [
                    r
                    for r in results
                    if r["batch"] == batch and abs(r["divergent_fraction"] - div) < 1e-6
                ],
                key=lambda r: r["seq_len"],
            )
            if not rows:
                continue
            color = cmap(i / max(len(divs) - 1, 1))
            axes[0].plot(
                [r["seq_len"] for r in rows],
                [r["speedup"] for r in rows],
                marker="o",
                markersize=3,
                color=color,
                linestyle="-" if batch == 1 else "--",
                label=f"B={batch} div={div:.0%}" if batch == 1 else None,
            )
    axes[0].axhline(1.0, color="k", linestyle=":", linewidth=1)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("sequence length")
    axes[0].set_ylabel("split / full layer speedup (x)")
    axes[0].set_title("FFN split per-layer speedup\n(solid B=1, dashed B=2/4)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, title="colors = divergent fraction", title_fontsize=7)

    min_div = min(r["divergent_fraction"] for r in results)
    for batch in batches:
        rows = sorted(
            [
                r
                for r in results
                if r["batch"] == batch and r["divergent_fraction"] <= min_div + 1e-6
            ],
            key=lambda r: r["seq_len"],
        )
        if rows:
            axes[1].plot(
                [r["seq_len"] for r in rows],
                [r["full_layer_ms"] for r in rows],
                marker="s",
                color="#d62728",
                label="full layer" if batch == 1 else None,
            )
            axes[1].plot(
                [r["seq_len"] for r in rows],
                [r["split_layer_ms"] for r in rows],
                marker="o",
                color="#1f77b4",
                label="split layer (1% divergent)" if batch == 1 else None,
            )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("sequence length (B=1)")
    axes[1].set_ylabel("layer time (ms, log)")
    axes[1].set_title("Layer latency vs sequence length")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Optimization #2: split FFN crossover with sequence length", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_opt2_shape.png", dpi=160)
    plt.close(fig)


def fig_opt2_longseq(root: Path, out: Path) -> None:
    models = [
        m for m in MODEL_ORDER if (root / "opt2" / "longseq" / m / "opt2_longseq.json").exists()
    ]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.3), squeeze=False)
    for ax, key in zip(axes[0], models):
        data = load_json(root / "opt2" / "longseq" / key / "opt2_longseq.json")
        label = data["meta"]["label"]
        baseline = data["meta"]["baseline_forward_ms"]
        for name, color, marker in (("nosplit", "#d62728", "s"), ("split", "#1f77b4", "o")):
            entries = data["conditions"][name]["entries"]
            ax.plot(
                [e["num_flips"] for e in entries],
                [e["latency_ms"] for e in entries],
                marker=marker,
                color=color,
                label=f"{name} (active={data['conditions'][name]['active_layers']})",
            )
        ax.axhline(baseline, color="k", linestyle="--", linewidth=1, label="baseline (no folding)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("flipped tokens (divergent, log)")
        ax.set_ylabel("latency (ms)")
        ax.set_title(f"{label}\nseq={data['meta']['seq_len']}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("End-to-end at seq=512: split vs non-split folded forward", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig_opt2_longseq.png", dpi=160)
    plt.close(fig)


def fig_opt3_frontier(root: Path, out: Path) -> None:
    models = [m for m in MODEL_ORDER if (root / "opt3" / m / "opt3_adaptive.json").exists()]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 4.4), squeeze=False)
    for ax, key in zip(axes[0], models):
        data = load_json(root / "opt3" / key / "opt3_adaptive.json")
        label = data["meta"]["label"]
        for flip, marker in (("1", "o"), ("8", "s")):
            fixed = data["fixed"]["conditions"][flip]
            adaptive = data["adaptive"]["conditions"][flip]
            ax.plot(
                [e["stable_ratio"] for e in fixed],
                [max(e["rel_mse"], 1e-6) for e in fixed],
                color="#d62728",
                marker=marker,
                linestyle="-",
                label=f"fixed tau, k={flip}",
            )
            ax.plot(
                [e["stable_ratio"] for e in adaptive],
                [max(e["rel_mse"], 1e-6) for e in adaptive],
                color="#1f77b4",
                marker=marker,
                linestyle="--",
                label=f"adaptive, k={flip}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("achieved stable ratio")
        ax.set_ylabel("relative MSE (log)")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Optimization #3: fidelity–reuse frontier, fixed tau vs adaptive top-k", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig_opt3_frontier.png", dpi=160)
    plt.close(fig)


def fig_opt4_fused(root: Path, out: Path) -> None:
    path = root / "opt4" / "opt4_fused.json"
    if not path.exists():
        return
    data = load_json(path)
    results = data["results"]
    hiddens = sorted({r["hidden_dim"] for r in results})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    cmap = plt.get_cmap("plasma")
    for i, hidden in enumerate(hiddens):
        rows = sorted([r for r in results if r["hidden_dim"] == hidden], key=lambda r: r["seq_len"])
        color = cmap(i / max(len(hiddens) - 1, 1))
        axes[0].plot(
            [r["seq_len"] for r in rows],
            [r["torch_ms"] for r in rows],
            marker="s",
            linestyle="--",
            color=color,
            label=f"PyTorch H={hidden}",
        )
        axes[0].plot(
            [r["seq_len"] for r in rows],
            [r["fused_ms"] for r in rows],
            marker="o",
            linestyle="-",
            color=color,
            label=f"fused H={hidden}",
        )
        axes[1].plot(
            [r["seq_len"] for r in rows],
            [r["speedup"] for r in rows],
            marker="o",
            color=color,
            label=f"H={hidden}",
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("sequence length")
    axes[0].set_ylabel("gather+select time (ms, log)")
    axes[0].set_title("Fused vs PyTorch")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].axhline(1.0, color="k", linestyle=":", linewidth=1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("sequence length")
    axes[1].set_ylabel("speedup (x)")
    axes[1].set_title("Fused speedup (max abs err = 0)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Optimization #4: fused Triton gather+select", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_opt4_fused.png", dpi=160)
    plt.close(fig)


def fig_summary(root: Path, out: Path) -> None:
    models = [m for m in MODEL_ORDER if (root / "opt1" / m / "opt1_cache.json").exists()]
    if models:
        fig, ax = plt.subplots(figsize=(9.5, 4.4))
        labels = []
        baseline = []
        legacy = []
        vectorized = []
        for key in models:
            data = load_json(root / "opt1" / key / "opt1_cache.json")
            labels.append(data["meta"]["label"])
            e2e = data["e2e"]
            baseline.append(e2e["baseline_forward_ms"])
            legacy.append(e2e["legacy"]["self_fold_ms"])
            vectorized.append(e2e["vectorized"]["self_fold_ms"])
        x = np.arange(len(models))
        w = 0.26
        ax.bar(x - w, baseline, w, label="baseline (no folding)", color="#7f7f7f")
        ax.bar(x, legacy, w, label="folded, legacy cache", color="#d62728")
        ax.bar(x + w, vectorized, w, label="folded, vectorized cache", color="#1f77b4")
        for xi, (b, l, v) in enumerate(zip(baseline, legacy, vectorized)):
            ax.text(xi + w, v * 1.02, f"{b / v:.1f}x", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("all-stable folded forward (ms)")
        ax.set_title(
            "Optimization #1 summary: 100% stable folded forward is now 2.3-2.7x faster than recompute"
        )
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "fig_opt_summary.png", dpi=160)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/optimization")
    parser.add_argument("--out", default="results/optimization/figures")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_opt1(root, out)
    fig_opt2_shape(root, out)
    fig_opt2_longseq(root, out)
    fig_opt3_frontier(root, out)
    fig_opt4_fused(root, out)
    fig_summary(root, out)
    print("optimization figures written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
