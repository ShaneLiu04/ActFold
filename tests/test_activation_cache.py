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


def test_put_get_no_host_sync() -> None:
    cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    activations = {"ffn_out": torch.randn(2, 8, 16)}

    with pytest.MonkeyPatch.context() as monkeypatch:

        def _forbid_sync(*args: object, **kwargs: object) -> None:
            raise AssertionError("sync")

        monkeypatch.setattr(torch.Tensor, "any", _forbid_sync)
        monkeypatch.setattr(torch.Tensor, "all", _forbid_sync)
        monkeypatch.setattr(torch.Tensor, "item", _forbid_sync)

        cache.put("b", 0, activations)
        mask = (torch.arange(8) % 2 == 0).unsqueeze(0).repeat(2, 1)
        out = cache.get("b", 0, mask)

    assert out["ffn_out"].shape == (2, 8, 16)
    assert (out["ffn_out"][~mask] == 0).all()
    assert torch.allclose(out["ffn_out"][mask], activations["ffn_out"][mask])


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones((2, 6), dtype=torch.bool),
        torch.zeros((2, 6), dtype=torch.bool),
        torch.tensor([[True, True, True, False, False, False]] * 2),
        (torch.arange(6) % 2 == 0).unsqueeze(0).repeat(2, 1),
    ],
    ids=["full-true", "full-false", "first-half", "alternating"],
)
def test_get_returns_correct_zero_fill(mask: torch.Tensor) -> None:
    cache = ActivationCache(device="cpu")
    activations = {"ffn_out": torch.randn(2, 6, 8)}
    cache.put("b", 0, activations)

    retrieved = cache.get("b", 0, mask)

    assert retrieved["ffn_out"].shape == (2, 6, 8)
    assert torch.equal(retrieved["ffn_out"][mask], activations["ffn_out"][mask])
    assert (retrieved["ffn_out"][~mask] == 0).all()


def test_get_returns_correct_zero_fill_single_position() -> None:
    cache = ActivationCache(device="cpu")
    activations = {"ffn_out": torch.randn(2, 6, 8)}
    cache.put("b", 0, activations)

    mask = torch.zeros((2, 6), dtype=torch.bool)
    mask[1, 4] = True
    retrieved = cache.get("b", 0, mask)

    assert retrieved["ffn_out"].shape == (2, 6, 8)
    assert torch.equal(retrieved["ffn_out"][mask], activations["ffn_out"][mask])
    assert (retrieved["ffn_out"][~mask] == 0).all()


def test_repeated_put_same_group_overwrites() -> None:
    cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    x1 = torch.randn(1, 6, 8)
    x2 = torch.randn(1, 6, 8)
    cache.put("b", 0, {"ffn_out": x1})
    cache.put("b", 0, {"ffn_out": x2})

    mask = torch.ones((1, 6), dtype=torch.bool)
    retrieved = cache.get("b", 0, mask)

    assert torch.equal(retrieved["ffn_out"], x2)
    assert not torch.equal(retrieved["ffn_out"], x1)
    assert cache.num_entries(layer_idx=0) == 6


def test_repeated_put_refreshes_group_lru_order() -> None:
    cache = ActivationCache(max_entries_per_layer=4, device="cpu")
    a1 = torch.randn(1, 2, 8)
    b = torch.randn(1, 2, 8)
    a2 = torch.randn(1, 2, 8)
    c = torch.randn(1, 2, 8)
    mask = torch.ones((1, 2), dtype=torch.bool)

    cache.put("a", 0, {"ffn_out": a1})
    cache.put("b", 0, {"ffn_out": b})
    cache.put("a", 0, {"ffn_out": a2})
    cache.put("c", 0, {"ffn_out": c})

    with pytest.raises(KeyError):
        cache.get("b", 0, mask)
    assert torch.equal(cache.get("a", 0, mask)["ffn_out"], a2)
    assert torch.allclose(cache.get("c", 0, mask)["ffn_out"], c)
    assert cache.num_entries(layer_idx=0) == 4


def test_group_lru_evicts_oldest_group() -> None:
    cache = ActivationCache(max_entries_per_layer=4, device="cpu")
    groups = {name: torch.randn(1, 2, 8) for name in ("g1", "g2", "g3")}
    for name in ("g1", "g2", "g3"):
        cache.put(name, 0, {"ffn_out": groups[name]})

    mask = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(KeyError):
        cache.get("g1", 0, mask)
    assert cache.num_entries() == 4
    assert torch.allclose(cache.get("g2", 0, mask)["ffn_out"], groups["g2"])
    assert torch.allclose(cache.get("g3", 0, mask)["ffn_out"], groups["g3"])


def test_partial_mask_on_batch_dimension() -> None:
    cache = ActivationCache(device="cpu")
    activations = {"ffn_out": torch.randn(2, 4, 8)}
    cache.put("b", 0, activations)

    mask = torch.zeros((2, 4), dtype=torch.bool)
    mask[0, :] = True
    retrieved = cache.get("b", 0, mask)

    assert retrieved["ffn_out"].shape == (2, 4, 8)
    assert torch.equal(retrieved["ffn_out"][0], activations["ffn_out"][0])
    assert (retrieved["ffn_out"][1] == 0).all()


def test_put_multiple_activations_same_group() -> None:
    cache = ActivationCache(device="cpu")
    activations = {
        "ffn_out": torch.randn(1, 4, 8),
        "hidden_states": torch.randn(1, 4, 8),
    }
    cache.put("b", 0, activations)

    mask = (torch.arange(4) % 2 == 0).unsqueeze(0)
    retrieved = cache.get("b", 0, mask)

    assert set(retrieved.keys()) == {"ffn_out", "hidden_states"}
    for name in ("ffn_out", "hidden_states"):
        assert retrieved[name].shape == (1, 4, 8)
        assert torch.equal(retrieved[name][mask], activations[name][mask])
        assert (retrieved[name][~mask] == 0).all()


def test_no_per_token_python_loop_in_put(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = ActivationCache(max_entries_per_layer=256, device="cpu")
    activations = {"ffn_out": torch.randn(1, 64, 8)}
    original_contiguous = torch.Tensor.contiguous
    contiguous_calls = {"count": 0}

    def _counting_contiguous(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        contiguous_calls["count"] += 1
        return original_contiguous(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", _counting_contiguous)
    cache.put("b", 0, activations)

    assert contiguous_calls["count"] <= 4


def test_large_put_get_roundtrip() -> None:
    cache = ActivationCache(max_entries_per_layer=256, device="cpu")
    activations = {"ffn_out": torch.randn(2, 128, 32)}
    cache.put("b", 0, activations)

    mask = torch.ones((2, 128), dtype=torch.bool)
    retrieved = cache.get("b", 0, mask)

    assert torch.allclose(retrieved["ffn_out"], activations["ffn_out"])
    # Token counting is per sequence row (a row holds the full batch), matching
    # the legacy seq-position keys and VectorizedActivationCache semantics.
    assert cache.num_entries(layer_idx=0) == 128
