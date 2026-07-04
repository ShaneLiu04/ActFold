"""Tests for actfold.core.activation_cache."""

from __future__ import annotations

import pytest
import torch

from actfold.core.activation_cache import ActivationCache


def test_put_and_get() -> None:
    cache = ActivationCache(max_entries_per_layer=4, device="cpu")
    activations = {
        "ffn_out": torch.randn(1, 4, 32),
        "hidden_states": torch.randn(1, 4, 32),
    }
    cache.put("branch_a", layer_idx=0, activations=activations)

    mask = torch.tensor([[True, True, False, False]])
    retrieved = cache.get("branch_a", layer_idx=0, token_mask=mask)

    assert "ffn_out" in retrieved
    assert retrieved["ffn_out"].shape == (1, 4, 32)
    assert torch.allclose(retrieved["ffn_out"][:, :2, :], activations["ffn_out"][:, :2, :])
    assert (retrieved["ffn_out"][:, 2:, :] == 0).all()


def test_lru_eviction() -> None:
    cache = ActivationCache(max_entries_per_layer=2, device="cpu")
    for i in range(3):
        cache.put(
            f"branch_{i}",
            layer_idx=0,
            activations={"ffn_out": torch.randn(1, 1, 16)},
        )
    assert cache.num_entries(layer_idx=0) == 2


def test_clear_branch() -> None:
    cache = ActivationCache(max_entries_per_layer=8, device="cpu")
    cache.put("branch_a", 0, {"ffn_out": torch.randn(1, 2, 16)})
    cache.put("branch_b", 0, {"ffn_out": torch.randn(1, 2, 16)})

    cache.clear_branch("branch_a")
    assert cache.num_entries(layer_idx=0) == 2

    mask = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(KeyError):
        cache.get("branch_a", 0, mask)

    retrieved = cache.get("branch_b", 0, mask)
    assert retrieved["ffn_out"].shape == (1, 2, 16)


def test_missing_branch_raises() -> None:
    cache = ActivationCache(device="cpu")
    mask = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(KeyError):
        cache.get("missing", 0, mask)


def test_num_entries() -> None:
    cache = ActivationCache(device="cpu")
    cache.put("a", 0, {"ffn_out": torch.randn(1, 3, 8)})
    cache.put("a", 1, {"ffn_out": torch.randn(1, 2, 8)})
    assert cache.num_entries() == 5
    assert cache.num_entries(layer_idx=0) == 3


def test_put_empty_raises() -> None:
    cache = ActivationCache(device="cpu")
    with pytest.raises(ValueError, match="activations must not be empty"):
        cache.put("a", 0, {})


def test_put_inconsistent_shapes_raises() -> None:
    cache = ActivationCache(device="cpu")
    with pytest.raises(ValueError, match="inconsistent leading shape"):
        cache.put(
            "a",
            0,
            {
                "ffn_out": torch.randn(1, 4, 8),
                "hidden_states": torch.randn(1, 3, 8),
            },
        )


def test_clear_layer() -> None:
    cache = ActivationCache(device="cpu")
    cache.put("a", 0, {"ffn_out": torch.randn(1, 2, 8)})
    cache.put("a", 1, {"ffn_out": torch.randn(1, 2, 8)})
    cache.clear_layer(0)
    assert cache.num_entries(layer_idx=0) == 0
    assert cache.num_entries(layer_idx=1) == 2


def test_clear_all() -> None:
    cache = ActivationCache(device="cpu")
    cache.put("a", 0, {"ffn_out": torch.randn(1, 2, 8)})
    cache.clear_all()
    assert cache.num_entries() == 0


def test_put_and_get_with_step_idx() -> None:
    cache = ActivationCache(device="cpu")
    activations = {"ffn_out": torch.randn(1, 4, 32)}
    cache.put("branch_a", layer_idx=0, step_idx=1, activations=activations)

    mask = torch.ones((1, 4), dtype=torch.bool)
    retrieved = cache.get("branch_a", layer_idx=0, step_idx=1, token_mask=mask)
    assert torch.allclose(retrieved["ffn_out"], activations["ffn_out"])

    with pytest.raises(KeyError):
        cache.get("branch_a", layer_idx=0, step_idx=0, token_mask=mask)
