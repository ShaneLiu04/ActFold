"""Tests for actfold.profiler modules."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from actfold.profiler import (
    HiddenStateTracker,
    InferenceMetrics,
    MetricsCollector,
    SimilarityAnalyzer,
)


def test_similarity_analyzer_identical() -> None:
    analyzer = SimilarityAnalyzer()
    x = torch.randn(2, 4, 8, 32)
    sim = analyzer.compute_similarity(x, x, metric="cosine")
    assert sim.shape == (2, 4, 8)
    assert torch.allclose(sim, torch.ones_like(sim), atol=1e-6)


def test_similarity_analyzer_report() -> None:
    analyzer = SimilarityAnalyzer()
    x = torch.randn(3, 5, 4, 32)
    y = x + 0.01 * torch.randn_like(x)
    sim = analyzer.compute_similarity(x, y, metric="cosine")
    report = analyzer.generate_report(sim, tau=0.95)
    assert "mean" in report
    assert "pct_stable" in report
    assert 0.0 <= report["pct_stable"] <= 100.0


def test_hidden_state_tracker(tmp_path: Path) -> None:
    model = nn.Sequential(
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )
    tracker = HiddenStateTracker(storage_device="cpu")
    tracker.register_hook(model, target_module_type=nn.Linear)

    x = torch.randn(2, 8)
    _ = model(x)

    states = tracker.get_states()
    assert len(states) > 0

    out_file = tmp_path / "states.safetensors"
    tracker.dump(out_file, format="safetensors")
    assert out_file.exists()
    tracker.clear()
    assert len(tracker.get_states()) == 0


def test_metrics_collector() -> None:
    collector = MetricsCollector(device="cpu", seq_len=16)
    with collector:
        x = torch.randn(4, 16)
        _ = x @ x.T
        collector.set_nfe(10)
        collector.set_tflops(1.5)
        collector.record("custom", 42)

    metrics = collector.get_metrics()
    assert isinstance(metrics, InferenceMetrics)
    assert metrics.nfe == 10
    assert metrics.tflops == 1.5
    assert metrics.custom["custom"] == 42


def test_unsupported_metric() -> None:
    analyzer = SimilarityAnalyzer()
    with pytest.raises(ValueError, match="Unsupported metric"):
        analyzer.compute_similarity(torch.randn(2, 3, 4), torch.randn(2, 3, 4), metric="hamming")
