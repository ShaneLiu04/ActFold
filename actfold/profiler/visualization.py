"""Visualization utilities for ActFold profiling results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")


def _ensure_path(path: Path | str | None) -> Path | None:
    """Normalize an optional path."""
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_similarity_heatmap(
    sim_matrix: torch.Tensor | np.ndarray[Any, Any],
    title: str = "Layer-Token Similarity",
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Generate a heatmap: x-axis=token position, y-axis=layer index.

    Args:
        sim_matrix: 2D tensor/array of shape ``[layers, tokens]`` or 3D
            ``[layers, tokens, steps]`` (the first step is used for 3D).
        title: Plot title.
        save_path: Optional path to save the figure.
        show: Whether to call ``plt.show()``.
    """
    data = sim_matrix if isinstance(sim_matrix, np.ndarray) else sim_matrix.detach().cpu().numpy()
    if data.ndim == 3:
        data = data[:, :, 0]
    if data.ndim != 2:
        raise ValueError(f"sim_matrix must be 2D or 3D, got {data.ndim}")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Cosine Similarity")

    save_path = _ensure_path(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_pareto_frontier(
    results: list[dict[str, float]],
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Plot speedup vs. accuracy trade-off.

    Args:
        results: List of dictionaries with keys ``speedup`` and ``accuracy_drop``.
        save_path: Optional path to save the figure.
        show: Whether to call ``plt.show()``.
    """
    if not results:
        raise ValueError("results must not be empty")

    speedups = [r["speedup"] for r in results]
    drops = [r["accuracy_drop"] for r in results]
    labels = [r.get("label", str(i)) for i, r in enumerate(results)]

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(speedups, drops, s=100, c=range(len(results)), cmap="plasma")
    for i, label in enumerate(labels):
        ax.annotate(str(label), (speedups[i], drops[i]), textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Speedup")
    ax.set_ylabel("Accuracy Drop (%)")
    ax.set_title("Speedup vs. Accuracy Trade-off")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.colorbar(scatter, ax=ax, label="Configuration Index")

    save_path = _ensure_path(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_tflops_reduction(
    reductions: dict[str, float],
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Plot a bar chart of TFLOPs reduction by configuration.

    Args:
        reductions: Mapping from configuration label to reduction percentage.
        save_path: Optional path to save the figure.
        show: Whether to call ``plt.show()``.
    """
    labels = list(reductions.keys())
    values = list(reductions.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color="steelblue")
    ax.set_ylabel("TFLOPs Reduction (%)")
    ax.set_title("Verification TFLOPs Reduction")
    ax.set_ylim(0.0, 100.0)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    save_path = _ensure_path(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_ablation_table(
    df_data: dict[str, list[Any]],
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Render an ablation study table as a figure.

    Args:
        df_data: Dictionary mapping column names to lists of values.
        save_path: Optional path to save the figure.
        show: Whether to call ``plt.show()``.
    """
    import pandas as pd

    df = pd.DataFrame(df_data)

    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.5), max(4, len(df) * 0.4)))
    ax.axis("off")
    ax.axis("tight")

    table = ax.table(
        cellText=df.round(3).values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    save_path = _ensure_path(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
