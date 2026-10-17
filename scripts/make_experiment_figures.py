#!/usr/bin/env python3
"""Generate explanatory figures from ActFold deep-experiment artifacts.

Reads ``results/experiments/<model>/results.json``, ``similarity.npz`` and
``overhead.json`` for every available model and writes PNG figures to
``results/experiments/figures/``.

Usage::

    python scripts/make_experiment_figures.py --root results/experiments --out results/experiments/figures
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
COLORS = {
    "fastdllm": "#1f77b4",
    "llada": "#d62728",
    "dream": "#2ca02c",
}


def load_models(root: Path) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for key in MODEL_ORDER:
        results_path = root / key / "results.json"
        if not results_path.exists():
            continue
        data: dict[str, Any] = {"results": json.loads(results_path.read_text(encoding="utf-8"))}
        npz_path = root / key / "similarity.npz"
        if npz_path.exists():
            data["sim"] = dict(np.load(npz_path))
        overhead_path = root / key / "overhead.json"
        if overhead_path.exists():
            data["overhead"] = json.loads(overhead_path.read_text(encoding="utf-8"))
        models[key] = data
    return models


def fig_overhead(models: dict[str, dict[str, Any]], out: Path) -> None:
    keys = [k for k, v in models.items() if "overhead" in v]
    if not keys:
        return
    fields = [
        ("original_layer_ms", "Original layer"),
        ("folded_all_stable_ms", "Folded all-stable"),
        ("folded_all_divergent_ms", "Folded all-divergent"),
        ("gate_ms", "Similarity gate"),
        ("cache_get_ms", "Cache get"),
        ("cache_put_ms", "Cache put"),
        ("merge_triton_ms", "Merge (Triton)"),
        ("merge_torch_ms", "Merge (PyTorch)"),
    ]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        ov = models[key]["overhead"]
        values = [max(ov[f], 1e-4) for f, _ in fields]
        labels = [lab for _, lab in fields]
        bars = ax.bar(range(len(fields)), values, color=COLORS.get(key, "gray"), alpha=0.85)
        ax.set_yscale("log")
        ax.set_xticks(range(len(fields)))
        ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
        ax.set_ylabel("time per layer (ms, log scale)")
        ax.set_title(f"{ov['model']}\nseq={ov['seq_len']}, hidden={ov['hidden_dim']}")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.15,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.axhline(ov["original_layer_ms"], color="k", linestyle="--", linewidth=0.8)
    fig.suptitle(
        "Per-layer overhead decomposition: cache retrieval exceeds layer compute",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_overhead_breakdown.png", dpi=160)
    plt.close(fig)


def fig_similarity_heatmap(models: dict[str, dict[str, Any]], out: Path) -> None:
    keys = [k for k, v in models.items() if "sim" in v]
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 2, figsize=(11, 2.8 * len(keys)), squeeze=False)
    for row, key in enumerate(keys):
        sim = models[key]["sim"]
        label = models[key]["results"]["meta"]["label"]
        for col, flip in enumerate((1, 8)):
            matrix = sim[f"flip{flip}"]
            ax = axes[row][col]
            im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xlabel("token position")
            ax.set_ylabel("layer")
            ax.set_title(f"{label} — {flip} token(s) flipped", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="cosine sim")
    fig.suptitle("Parent/child hidden-state similarity map by layer and token", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_similarity_heatmap.png", dpi=160)
    plt.close(fig)


def fig_stable_by_layer(models: dict[str, dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for key, data in models.items():
        if "sim" not in data:
            continue
        matrix = data["sim"]["flip1"]
        color = COLORS.get(key, "gray")
        label = data["results"]["meta"]["label"]
        mean_sim = matrix.mean(axis=1)
        axes[0].plot(mean_sim, label=label, color=color, marker="o", markersize=3)
        for tau, style in ((0.95, "-"), (0.99, "--"), (0.995, ":")):
            axes[1].plot(
                (matrix > tau).mean(axis=1),
                label=f"{label} τ={tau}",
                color=color,
                linestyle=style,
                marker="o",
                markersize=2.5,
            )
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("mean cosine similarity (1 flipped token)")
    axes[0].set_title("Similarity vs depth")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("fraction of stable tokens")
    axes[1].set_title("Per-layer stable ratio at different τ")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_stable_by_layer.png", dpi=160)
    plt.close(fig)


def fig_tau_quality(models: dict[str, dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for key, data in models.items():
        sweep = data["results"].get("tau_sweep")
        if not isinstance(sweep, list):
            continue
        color = COLORS.get(key, "gray")
        label = data["results"]["meta"]["label"]
        taus = [e["tau"] for e in sweep]
        axes[0].plot(taus, [e["mean_stable"] for e in sweep], color=color, marker="o", label=label)
        axes[1].plot(
            taus,
            [e["top1_agreement"] for e in sweep],
            color=color,
            marker="o",
            label=label,
        )
        axes[2].plot(
            taus,
            [max(e["rel_mse"], 1e-8) for e in sweep],
            color=color,
            marker="o",
            label=label,
        )
    axes[0].set_ylabel("mean stable ratio")
    axes[0].set_title("Reuse grows as τ drops")
    axes[1].set_ylabel("top-1 agreement vs baseline")
    axes[1].set_title("Output fidelity")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("relative MSE (log)")
    axes[2].set_title("Approximation error vs τ")
    for ax in axes:
        ax.set_xlabel("τ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_tau_quality.png", dpi=160)
    plt.close(fig)


def fig_cache_budget(models: dict[str, dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for key, data in models.items():
        budget = data["results"].get("cache_budget")
        if not isinstance(budget, dict):
            continue
        color = COLORS.get(key, "gray")
        label = data["results"]["meta"]["label"]
        xs = sorted(int(k) for k in budget)
        axes[0].plot(
            xs, [budget[str(b)]["mean_stable"] for b in xs], color=color, marker="o", label=label
        )
        axes[1].plot(
            xs, [budget[str(b)]["top1_agreement"] for b in xs], color=color, marker="o", label=label
        )
    for ax, ylabel, title in (
        (axes[0], "mean stable ratio", "Reuse vs per-layer cache budget"),
        (axes[1], "top-1 agreement", "Fidelity vs cache budget"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("max entries per layer")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_cache_budget.png", dpi=160)
    plt.close(fig)


def fig_layer_ablation(models: dict[str, dict[str, Any]], out: Path) -> None:
    variants = ["all", "early_only", "late_only", "none"]
    keys = [k for k, v in models.items() if isinstance(v["results"].get("layer_ablation"), dict)]
    if not keys:
        return
    x = np.arange(len(variants))
    width = 0.8 / max(1, len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for idx, key in enumerate(keys):
        data = models[key]["results"]["layer_ablation"]
        label = models[key]["results"]["meta"]["label"]
        color = COLORS.get(key, "gray")
        stable = [data[v]["mean_stable"] for v in variants]
        rel_mse = [data[v]["rel_mse"] for v in variants]
        axes[0].bar(x + idx * width, stable, width, label=label, color=color, alpha=0.85)
        axes[1].bar(x + idx * width, rel_mse, width, label=label, color=color, alpha=0.85)
    for ax, ylabel, title in (
        (axes[0], "mean stable ratio", "Reuse by folded layer range"),
        (axes[1], "relative MSE", "Error by folded layer range"),
    ):
        ax.set_xticks(x + width * (len(keys) - 1) / 2)
        ax.set_xticklabels(variants)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_layer_ablation.png", dpi=160)
    plt.close(fig)


def fig_latency_prediction(models: dict[str, dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), squeeze=False)
    for ax, (key, data) in zip(axes[0], models.items()):
        cost = data["results"].get("cost_model")
        if not isinstance(cost, dict) or "predictions" not in cost:
            ax.axis("off")
            continue
        preds = cost["predictions"]
        stable = [p["stable_ratio"] for p in preds]
        measured = [p["measured_ms"] for p in preds]
        predicted = [max(p["predicted_ms"], 1e-4) for p in preds]
        ax.plot(stable, measured, "o-", color="#d62728", label="measured folded forward")
        ax.plot(stable, predicted, "s--", color="#1f77b4", label="cost-model prediction")
        ax.set_yscale("log")
        ax.set_xlabel("mean stable ratio")
        ax.set_ylabel("latency (ms, log)")
        ax.set_title(data["results"]["meta"]["label"], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "FLOPs model vs wall-clock reality: predicted gains do not materialise "
        "while any token is divergent",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig_latency_vs_prediction.png", dpi=160)
    plt.close(fig)


def fig_sampling(models: dict[str, dict[str, Any]], out: Path) -> None:
    keys = [
        k
        for k, v in models.items()
        if isinstance(v["results"].get("sampling"), dict)
        and "step_stats" in v["results"]["sampling"]
    ]
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    repeats = max(int(models[k]["results"]["sampling"].get("repeats", 1)) for k in keys)
    for key in keys:
        sampling = models[key]["results"]["sampling"]
        steps = sampling["step_stats"]
        color = COLORS.get(key, "gray")
        label = models[key]["results"]["meta"]["label"]
        # New schema carries mean/std over >=3 seeded repeats; old artifacts
        # only have a single-sample latency (std treated as 0).
        baseline_mean = sampling.get("baseline_ms_mean", sampling.get("baseline_ms", 0.0))
        folded_mean = sampling.get("folded_ms_mean", sampling.get("folded_ms", 0.0))
        axes[0].plot(
            range(1, len(steps) + 1),
            [s["mean_stable"] for s in steps],
            color=color,
            marker="o",
            markersize=3,
            label=f"{label} (match={sampling['token_match_rate']:.2f})",
        )
        axes[1].bar(
            label,
            folded_mean / max(baseline_mean, 1e-9),
            color=color,
            alpha=0.85,
        )
    axes[0].set_xlabel("diffusion denoising step")
    axes[0].set_ylabel("mean stable ratio")
    axes[0].set_title("Cross-step stable ratio during diffusion sampling")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("folded / baseline wall-clock ratio")
    axes[1].set_title(f"Folded sampling slowdown (mean ± std, n={repeats})")
    axes[1].axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    for idx, key in enumerate(keys):
        sampling = models[key]["results"]["sampling"]
        baseline_mean = sampling.get("baseline_ms_mean", sampling.get("baseline_ms", 0.0))
        baseline_std = sampling.get("baseline_ms_std", 0.0)
        folded_mean = sampling.get("folded_ms_mean", sampling.get("folded_ms", 0.0))
        folded_std = sampling.get("folded_ms_std", 0.0)
        ratio = folded_mean / max(baseline_mean, 1e-9)
        # First-order error propagation of the latency ratio.
        rel_var = (folded_std / max(folded_mean, 1e-9)) ** 2 + (
            baseline_std / max(baseline_mean, 1e-9)
        ) ** 2
        ratio_std = float(np.sqrt(rel_var)) * ratio
        axes[1].text(idx, ratio * 1.02, f"{ratio:.2f}x ± {ratio_std:.2f}", ha="center", fontsize=9)
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out / "fig_sampling.png", dpi=160)
    plt.close(fig)


def fig_similarity_hist(models: dict[str, dict[str, Any]], out: Path) -> None:
    keys = [k for k, v in models.items() if "sim" in v]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.0), squeeze=False)
    for ax, key in zip(axes[0], keys):
        sim = models[key]["sim"]
        for flip, alpha in ((1, 0.6), (8, 0.4)):
            ax.hist(
                sim[f"flip{flip}"].ravel(),
                bins=60,
                range=(0, 1),
                alpha=alpha,
                label=f"{flip} flipped token(s)",
                color="#1f77b4" if flip == 1 else "#ff7f0e",
            )
        ax.set_yscale("log")
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("count (log)")
        ax.set_title(models[key]["results"]["meta"]["label"], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Similarity distribution: most tokens are near-identical, flipped tokens trail", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_similarity_hist.png", dpi=160)
    plt.close(fig)


def fig_invariants(models: dict[str, dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    keys = list(models)
    x = np.arange(len(keys))
    baseline = [models[k]["results"]["invariants"]["self_fold"]["baseline_ms"] for k in keys]
    folded = [models[k]["results"]["invariants"]["self_fold"]["folded_fast_path_ms"] for k in keys]
    width = 0.38
    axes[0].bar(x - width / 2, baseline, width, label="baseline forward", color="#4c72b0")
    axes[0].bar(
        x + width / 2, folded, width, label="folded fast path (100% stable)", color="#dd8452"
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([models[k]["results"]["meta"]["label"] for k in keys], fontsize=8)
    axes[0].set_ylabel("latency (ms)")
    axes[0].set_title("All-stable fast path is slower than plain recompute")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    for idx, key in enumerate(keys):
        speedup = models[key]["results"]["invariants"]["self_fold"]["measured_speedup"]
        axes[0].text(
            idx,
            max(baseline[idx], folded[idx]) * 1.03,
            f"{speedup:.2f}x",
            ha="center",
            fontsize=9,
        )
    self_mse = [models[k]["results"]["invariants"]["self_fold"]["mse"] for k in keys]
    div_mse = [models[k]["results"]["invariants"]["all_divergent"]["mse"] for k in keys]
    axes[1].bar(
        x - width / 2,
        [max(v, 1e-12) for v in self_mse],
        width,
        label="self-fold MSE",
        color="#55a868",
    )
    axes[1].bar(
        x + width / 2,
        [max(v, 1e-12) for v in div_mse],
        width,
        label="all-divergent MSE",
        color="#c44e52",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([models[k]["results"]["meta"]["label"] for k in keys], fontsize=8)
    axes[1].set_ylabel("logit MSE (log)")
    axes[1].set_title("Algorithm invariants: exactness holds")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig_invariants.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/experiments")
    parser.add_argument("--out", default="results/experiments/figures")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = load_models(root)
    print("models with artifacts:", list(models))
    fig_overhead(models, out)
    fig_similarity_heatmap(models, out)
    fig_stable_by_layer(models, out)
    fig_tau_quality(models, out)
    fig_cache_budget(models, out)
    fig_layer_ablation(models, out)
    fig_latency_prediction(models, out)
    fig_sampling(models, out)
    fig_similarity_hist(models, out)
    fig_invariants(models, out)
    print("figures written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
