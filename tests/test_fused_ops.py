"""Tests for actfold.core.fused_ops."""

from __future__ import annotations

import pytest
import torch

from actfold.core.activation_cache import ActivationCache
from actfold.core.fused_ops import (
    _HAS_TRITON,
    _merge_stable_divergent_torch,
    gather_cached_activations,
    merge_stable_divergent,
)


# ---------------------------------------------------------------------------
# merge_stable_divergent
# ---------------------------------------------------------------------------
def _reference_merge(
    parent: torch.Tensor,
    child: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Plain PyTorch reference for the merge operation."""
    return torch.where(mask.unsqueeze(-1), parent, child)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_merge_stable_divergent_shapes_and_values(dtype: torch.dtype) -> None:
    batch, seq, hidden = 2, 8, 64
    parent = torch.randn(batch, seq, hidden, dtype=dtype)
    child = torch.randn(batch, seq, hidden, dtype=dtype)
    mask = torch.rand(batch, seq) > 0.5

    out = merge_stable_divergent(parent, child, mask)
    expected = _reference_merge(parent, child, mask)

    if dtype == torch.bfloat16:
        assert torch.allclose(out.float(), expected.float(), atol=1e-2)
    elif dtype == torch.float16:
        assert torch.allclose(out.float(), expected.float(), atol=1e-3)
    else:
        assert torch.allclose(out, expected, atol=1e-6)


def test_merge_all_stable() -> None:
    parent = torch.randn(1, 4, 32)
    child = torch.randn(1, 4, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = merge_stable_divergent(parent, child, mask)
    assert torch.allclose(out, parent)


def test_merge_all_divergent() -> None:
    parent = torch.randn(1, 4, 32)
    child = torch.randn(1, 4, 32)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    out = merge_stable_divergent(parent, child, mask)
    assert torch.allclose(out, child)


def test_merge_shape_mismatch_raises() -> None:
    parent = torch.randn(2, 8, 32)
    child = torch.randn(2, 8, 64)
    mask = torch.ones(2, 8, dtype=torch.bool)
    with pytest.raises(ValueError):
        merge_stable_divergent(parent, child, mask)

    child = torch.randn(2, 4, 32)
    with pytest.raises(ValueError):
        merge_stable_divergent(parent, child, mask)

    mask = torch.ones(2, 4, dtype=torch.bool)
    child = torch.randn(2, 8, 32)
    with pytest.raises(ValueError):
        merge_stable_divergent(parent, child, mask)


def test_torch_fallback_matches_reference() -> None:
    parent = torch.randn(2, 8, 32)
    child = torch.randn(2, 8, 32)
    mask = torch.rand(2, 8) > 0.5
    out = _merge_stable_divergent_torch(parent, child, mask)
    expected = _reference_merge(parent, child, mask)
    assert torch.allclose(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_merge_on_cuda(dtype: torch.dtype) -> None:
    parent = torch.randn(2, 8, 128, dtype=dtype, device="cuda")
    child = torch.randn(2, 8, 128, dtype=dtype, device="cuda")
    mask = torch.rand(2, 8, device="cuda") > 0.5

    out = merge_stable_divergent(parent, child, mask)
    expected = _reference_merge(parent, child, mask)

    assert out.device.type == "cuda"
    if dtype == torch.bfloat16:
        assert torch.allclose(out.float(), expected.float(), atol=1e-2)
    elif dtype == torch.float16:
        assert torch.allclose(out.float(), expected.float(), atol=1e-3)
    else:
        assert torch.allclose(out, expected, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_merge_cpu_mask_with_cuda_tensors() -> None:
    """A CPU mask must not crash the CUDA merge; it should fall back to PyTorch."""
    parent = torch.randn(1, 4, 128, device="cuda")
    child = torch.randn(1, 4, 128, device="cuda")
    mask = torch.ones(1, 4, dtype=torch.bool)  # CPU mask
    out = merge_stable_divergent(parent, child, mask)
    expected = _reference_merge(parent, child, mask.to(parent.device))
    assert out.device.type == "cuda"
    assert torch.allclose(out, expected, atol=1e-6)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_kernel_used_on_cuda() -> None:
    """Smoke test ensuring the Triton path is exercised when available."""
    parent = torch.randn(2, 4, 128, device="cuda")
    child = torch.randn(2, 4, 128, device="cuda")
    mask = torch.tensor([[True, False, True, False], [False, True, True, True]], device="cuda")
    out = merge_stable_divergent(parent, child, mask)
    expected = _reference_merge(parent, child, mask)
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# gather_cached_activations
# ---------------------------------------------------------------------------
def _build_sparse_cache(
    batch: int,
    seq: int,
    hidden: int,
    missing: set[int] | None = None,
) -> dict[tuple[str, int, int, int], dict[str, torch.Tensor]]:
    missing = missing or set()
    cache: dict[tuple[str, int, int, int], dict[str, torch.Tensor]] = {}
    for t in range(seq):
        if t in missing:
            continue
        cache[("branch", 0, t, 0)] = {
            "ffn_out": torch.randn(batch, hidden),
            "hidden_states": torch.randn(batch, hidden),
        }
    return cache


def test_gather_cached_activations_dense() -> None:
    batch, seq, hidden = 2, 4, 32
    cache = _build_sparse_cache(batch, seq, hidden)
    token_mask = torch.tensor([[True, True, False, False], [False, True, True, True]])

    out = gather_cached_activations(cache, "branch", 0, token_mask)

    assert out["ffn_out"].shape == (batch, seq, hidden)
    assert out["hidden_states"].shape == (batch, seq, hidden)

    for name in ("ffn_out", "hidden_states"):
        for b in range(batch):
            for t in range(seq):
                if token_mask[b, t]:
                    assert torch.allclose(out[name][b, t], cache[("branch", 0, t, 0)][name][b])
                else:
                    assert (out[name][b, t] == 0).all()


def test_gather_cached_activations_sparse_fallback() -> None:
    batch, seq, hidden = 1, 4, 32
    cache = _build_sparse_cache(batch, seq, hidden, missing={2})
    token_mask = torch.ones(batch, seq, dtype=torch.bool)

    out = gather_cached_activations(cache, "branch", 0, token_mask)
    assert out["ffn_out"].shape == (batch, seq, hidden)
    # Missing token position should remain zero.
    assert (out["ffn_out"][:, 2, :] == 0).all()


def test_gather_cached_activations_matches_activation_cache() -> None:
    """Vectorized gather must match the legacy ActivationCache.get output."""
    cache_obj = ActivationCache(max_entries_per_layer=16, device="cpu")
    activations = {
        "ffn_out": torch.randn(2, 5, 32),
        "hidden_states": torch.randn(2, 5, 32),
    }
    cache_obj.put("branch", layer_idx=0, activations=activations)

    token_mask = torch.rand(2, 5) > 0.3
    out = cache_obj.get("branch", layer_idx=0, token_mask=token_mask)

    assert out["ffn_out"].shape == activations["ffn_out"].shape
    assert out["hidden_states"].shape == activations["hidden_states"].shape

    for name in ("ffn_out", "hidden_states"):
        expected = torch.where(
            token_mask.unsqueeze(-1),
            activations[name],
            torch.zeros_like(activations[name]),
        )
        assert torch.allclose(out[name], expected)


def test_gather_cached_activations_missing_first_token_raises() -> None:
    cache: dict[tuple[str, int, int, int], dict[str, torch.Tensor]] = {}
    token_mask = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(KeyError):
        gather_cached_activations(cache, "branch", 0, token_mask)
