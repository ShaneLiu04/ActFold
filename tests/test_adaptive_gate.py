"""Tests for the adaptive quantile gate."""

from __future__ import annotations

import pytest
import torch

from actfold.core.adaptive_gate import AdaptiveQuantileGate


def test_selects_exact_target_fraction() -> None:
    x = torch.randn(2, 10, 16)
    y = torch.randn(2, 10, 16)
    gate = AdaptiveQuantileGate(target_stable_ratio=0.7)
    mask = gate(x, y)
    assert mask.shape == (2, 10)
    assert int(mask.sum().item()) == 14
    assert mask.dtype == torch.bool


def test_identical_tokens_fully_stable() -> None:
    x = torch.randn(1, 8, 16)
    gate = AdaptiveQuantileGate(target_stable_ratio=1.0)
    mask = gate(x, x.clone())
    assert mask.all()
    assert gate.last_tau == pytest.approx(1.0, abs=1e-3)


def test_tie_handling_selects_exact_count() -> None:
    """Exact ties must not push the stable ratio past the target."""
    x = torch.zeros(1, 4, 8)
    y = torch.zeros(1, 4, 8)
    y[0, 0] = 1.0  # three identical tokens, one different
    gate = AdaptiveQuantileGate(target_stable_ratio=0.5)
    mask = gate(x, y)
    assert int(mask.sum().item()) == 2


def test_target_updates() -> None:
    gate = AdaptiveQuantileGate(target_stable_ratio=0.9)
    gate.set_target_stable_ratio(0.5)
    assert gate.target_stable_ratio == pytest.approx(0.5)
    x = torch.randn(1, 10, 8)
    mask = gate(x, torch.randn(1, 10, 8))
    assert int(mask.sum().item()) == 5


def test_invalid_target_raises() -> None:
    with pytest.raises(ValueError):
        AdaptiveQuantileGate(target_stable_ratio=0.0)
    with pytest.raises(ValueError):
        AdaptiveQuantileGate(target_stable_ratio=1.5)


def test_shape_validation() -> None:
    gate = AdaptiveQuantileGate()
    with pytest.raises(ValueError):
        gate(torch.randn(2, 8, 4), torch.randn(2, 9, 4))
    with pytest.raises(ValueError):
        gate(torch.randn(8, 4), torch.randn(8, 4))


@pytest.mark.parametrize("metric", ["cosine", "l2", "pearson"])
def test_supported_metrics(metric: str) -> None:
    gate = AdaptiveQuantileGate(target_stable_ratio=0.5, metric=metric)
    x = torch.randn(1, 6, 16)
    mask = gate(x, torch.randn(1, 6, 16))
    assert int(mask.sum().item()) == 3


def test_forward_no_sync_and_no_shared_state_write() -> None:
    """T011: forward must not ``.item()`` nor write ``gate.tau``."""
    gate = AdaptiveQuantileGate(target_stable_ratio=0.7)
    x = torch.randn(2, 10, 16)
    y = torch.randn(2, 10, 16)
    tau_before = gate.tau

    def _raise_item(*args: object, **kwargs: object) -> None:
        raise AssertionError("sync during forward")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "item", _raise_item)
        mask = gate(x, y)

    assert int(mask.sum().item()) == 14
    assert gate.tau == tau_before, "forward must not mutate the shared tau state"
    # last_tau is materialized lazily on read (one readback allowed there).
    assert isinstance(gate.last_tau, float)


def test_bottom_k_mask_equivalence() -> None:
    """T011: bottom-k divergent selection equals top-k stable selection."""
    torch.manual_seed(0)
    gate = AdaptiveQuantileGate(target_stable_ratio=0.7)
    x = torch.randn(2, 10, 16)
    y = torch.randn(2, 10, 16)
    mask = gate(x, y)

    sim = torch.nn.functional.cosine_similarity(x.float(), y.float(), dim=-1)
    expected_stable = torch.topk(sim.reshape(-1), 14, largest=True).indices
    expected = torch.zeros(20, dtype=torch.bool)
    expected[expected_stable] = True
    assert torch.equal(mask.reshape(-1), expected)


def test_last_tau_is_boundary_value() -> None:
    """T011: last_tau equals the least-stable stable token's similarity."""
    torch.manual_seed(1)
    gate = AdaptiveQuantileGate(target_stable_ratio=0.5)
    x = torch.randn(1, 8, 16)
    y = torch.randn(1, 8, 16)
    gate(x, y)

    sim = torch.nn.functional.cosine_similarity(x.float(), y.float(), dim=-1)
    boundary = torch.topk(sim.reshape(-1), 4, largest=True).values[-1]
    assert gate.last_tau == pytest.approx(float(boundary), abs=1e-6)


def test_large_ratio_bottom_k_path() -> None:
    """T011: ratio close to 1 exercises the small-k divergent selection."""
    torch.manual_seed(2)
    gate = AdaptiveQuantileGate(target_stable_ratio=0.97)
    x = torch.randn(1, 100, 16)
    mask = gate(x, torch.randn(1, 100, 16))
    assert int(mask.sum().item()) == 97


def test_last_tau_cached_across_reads() -> None:
    """T011: repeated last_tau reads do not re-readback the candidate."""
    gate = AdaptiveQuantileGate(target_stable_ratio=0.5)
    gate(torch.randn(1, 8, 16), torch.randn(1, 8, 16))
    first = gate.last_tau
    calls = {"n": 0}
    original_item = torch.Tensor.item

    def _counting_item(self: torch.Tensor) -> float:
        calls["n"] += 1
        return original_item(self)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "item", _counting_item)
        second = gate.last_tau
    assert second == first
    assert calls["n"] == 0, "cached last_tau must not trigger a new readback"
