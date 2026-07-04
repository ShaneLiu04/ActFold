"""Tests for actfold.core.similarity_gate."""

from __future__ import annotations

import pytest
import torch

from actfold.core.similarity_gate import SimilarityGate


@pytest.fixture
def identical_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(2, 8, 32)
    return x, x.clone()


@pytest.fixture
def different_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(2, 8, 32)
    y = torch.randn(2, 8, 32)
    return x, y


def test_identical_is_stable(identical_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = identical_tensors
    gate = SimilarityGate(tau=0.95, metric="cosine")
    mask = gate(x, y)
    assert mask.shape == (2, 8)
    assert mask.all()


def test_different_is_unstable(different_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = different_tensors
    gate = SimilarityGate(tau=0.99, metric="cosine")
    mask = gate(x, y)
    # Random independent vectors are very unlikely to exceed 0.99 cosine similarity.
    assert not mask.any()


@pytest.mark.parametrize("metric", ["cosine", "l2", "pearson"])
def test_supported_metrics(
    identical_tensors: tuple[torch.Tensor, torch.Tensor],
    metric: str,
) -> None:
    x, y = identical_tensors
    gate = SimilarityGate(tau=0.95, metric=metric)
    mask = gate(x, y)
    assert mask.shape == (2, 8)
    assert mask.all()


def test_invalid_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        SimilarityGate(metric="jaccard")


def test_invalid_tau() -> None:
    with pytest.raises(ValueError, match="tau must be in"):
        SimilarityGate(tau=1.5)


def test_shape_mismatch() -> None:
    gate = SimilarityGate()
    with pytest.raises(ValueError, match="Shape mismatch"):
        gate(torch.randn(2, 8, 32), torch.randn(2, 10, 32))


def test_set_tau() -> None:
    gate = SimilarityGate(tau=0.95)
    gate.set_tau(0.8)
    assert gate.tau == pytest.approx(0.8)


def test_set_tau_invalid() -> None:
    gate = SimilarityGate()
    with pytest.raises(ValueError):
        gate.set_tau(-0.1)
