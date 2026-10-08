"""Tests for actfold.core.fused_ops."""

from __future__ import annotations

import warnings

import pytest
import torch

from actfold.core import fused_ops
from actfold.core.activation_cache import ActivationCache
from actfold.core.fused_ops import (
    _HAS_TRITON,
    _merge_stable_divergent_torch,
    gather_cached_activations,
    gather_select,
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


# ---------------------------------------------------------------------------
# gather_select (optimization #4)
# ---------------------------------------------------------------------------
def test_gather_select_cpu_fallback() -> None:
    from actfold.core.fused_ops import gather_select

    parent = torch.randn(8, 16)
    rows = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    child = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
    out = gather_select(parent, rows, child, mask)
    expected = torch.where(mask.unsqueeze(-1), parent[rows.reshape(-1)].reshape(2, 4, 16), child)
    assert torch.allclose(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gather_select_cuda_matches_reference(dtype: torch.dtype) -> None:
    from actfold.core.fused_ops import gather_select

    capacity, seq, hidden = 64, 32, 256
    parent = torch.randn(capacity, hidden, dtype=dtype, device="cuda")
    rows = torch.randint(0, capacity, (1, seq), device="cuda")
    child = torch.randn(1, seq, hidden, dtype=dtype, device="cuda")
    mask = torch.rand(1, seq, device="cuda") > 0.3
    out = gather_select(parent, rows, child, mask)
    expected = torch.where(
        mask.unsqueeze(-1), parent.index_select(0, rows.reshape(-1)).reshape(1, seq, hidden), child
    )
    assert out.shape == (1, seq, hidden)
    assert torch.equal(out.float(), expected.float())


def test_gather_select_validation() -> None:
    from actfold.core.fused_ops import gather_select

    with pytest.raises(ValueError):
        gather_select(
            torch.randn(4, 8),
            torch.zeros(1, 2, dtype=torch.long),
            torch.randn(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError):
        gather_select(
            torch.randn(4, 9),
            torch.zeros(1, 2, dtype=torch.long),
            torch.randn(1, 2, 8),
            torch.ones(1, 2, dtype=torch.bool),
        )


# ---------------------------------------------------------------------------
# T013: merge Triton improvements + independent gather/select disable flag
# ---------------------------------------------------------------------------
def _t013_forbidden_fallback(*args: object, **kwargs: object) -> torch.Tensor:
    """Stand-in for the PyTorch fallback that must never be called."""
    raise AssertionError("fallback used")


class _T013BoomKernel:
    """Fake Triton kernel whose launch always raises RuntimeError('boom')."""

    def __getitem__(self, grid: object) -> None:
        """Pretend the kernel launch fails, like ``kernel[grid](...)`` would."""
        raise RuntimeError("boom")


def _t013_gather_expected(
    parent: torch.Tensor,
    rows: torch.Tensor,
    child: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reference result for ``gather_select`` (PyTorch index_select + where)."""
    batch, seq, hidden = child.shape
    return torch.where(
        mask.unsqueeze(-1),
        parent.index_select(0, rows.reshape(-1)).reshape(batch, seq, hidden),
        child,
    )


def _t013_small_gather_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small deterministic CUDA inputs for ``gather_select`` (capacity, rows, child, mask)."""
    torch.manual_seed(13)
    capacity, batch, seq, hidden = 16, 1, 4, 32
    parent = torch.randn(capacity, hidden, device="cuda")
    rows = torch.randint(0, capacity, (batch, seq), device="cuda")
    child = torch.randn(batch, seq, hidden, device="cuda")
    mask = torch.rand(batch, seq, device="cuda") > 0.5
    return parent, rows, child, mask


def test_t013_gather_select_disabled_flag_exists() -> None:
    """Module must expose an independent gather/select disable flag, default False."""
    assert hasattr(fused_ops, "_TRITON_GATHER_SELECT_DISABLED")
    assert fused_ops._TRITON_GATHER_SELECT_DISABLED is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_t013_merge_bit_exact_h3584(dtype: torch.dtype) -> None:
    """Triton merge must be bit-exact vs the PyTorch fallback at H=3584."""
    torch.manual_seed(1301)
    batch, seq, hidden = 2, 7, 3584
    parent = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    child = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    mask = torch.rand(batch, seq, device="cuda") > 0.5

    out = merge_stable_divergent(parent, child, mask)
    expected = _merge_stable_divergent_torch(parent, child, mask)
    assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_t013_merge_bit_exact_noncontiguous() -> None:
    """Triton merge must be bit-exact on non-contiguous CUDA inputs.

    Case 1 uses stride-2 hidden slices; case 2 uses unusual batch/seq strides
    built with ``torch.as_strided``.
    """
    torch.manual_seed(1302)
    # Case 1: stride-2 hidden slices of a wider buffer.
    wide_parent = torch.randn(2, 7, 256, device="cuda")
    wide_child = torch.randn(2, 7, 256, device="cuda")
    parent = wide_parent[:, :, ::2]
    child = wide_child[:, :, ::2]
    mask = torch.rand(2, 7, device="cuda") > 0.5
    assert not parent.is_contiguous()
    assert parent.stride(2) == 2

    out = merge_stable_divergent(parent, child, mask)
    expected = _merge_stable_divergent_torch(parent, child, mask)
    assert torch.equal(out.cpu(), expected.cpu())

    # Case 2: unusual batch/seq strides via as_strided.
    base_parent = torch.randn(8, 30, 128, device="cuda")
    base_child = torch.randn(8, 30, 128, device="cuda")
    parent2 = base_parent.as_strided((2, 7, 128), (4 * 30 * 128, 2 * 128, 1))
    child2 = base_child.as_strided((2, 7, 128), (4 * 30 * 128, 2 * 128, 1))
    mask2 = torch.rand(2, 7, device="cuda") > 0.5
    assert not parent2.is_contiguous()
    assert parent2.stride(0) == 4 * 30 * 128 and parent2.stride(1) == 2 * 128

    out2 = merge_stable_divergent(parent2, child2, mask2)
    expected2 = _merge_stable_divergent_torch(parent2, child2, mask2)
    assert torch.equal(out2.cpu(), expected2.cpu())


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_t013_merge_triton_path_any_hidden_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Triton merge must handle hidden dims that are not multiples of 128 (H=100)."""
    torch.manual_seed(1303)
    batch, seq, hidden = 2, 7, 100
    parent = torch.randn(batch, seq, hidden, device="cuda")
    child = torch.randn(batch, seq, hidden, device="cuda")
    mask = torch.rand(batch, seq, device="cuda") > 0.5
    expected = _merge_stable_divergent_torch(parent, child, mask)

    monkeypatch.setattr(fused_ops, "_merge_stable_divergent_torch", _t013_forbidden_fallback)
    out = merge_stable_divergent(parent, child, mask)
    assert torch.equal(out, expected)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_t013_merge_triton_path_noncontiguous_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Triton merge must handle non-contiguous inputs (stride-2 hidden) natively.

    Uses H=100 with hidden stride 2 so both tail masking and real stride
    support are required; the PyTorch fallback is forbidden via monkeypatch.
    """
    torch.manual_seed(1304)
    wide_parent = torch.randn(2, 7, 200, device="cuda")
    wide_child = torch.randn(2, 7, 200, device="cuda")
    parent = wide_parent[:, :, ::2]  # [2, 7, 100] with hidden stride 2
    child = wide_child[:, :, ::2]
    mask = torch.rand(2, 7, device="cuda") > 0.5
    assert not parent.is_contiguous()
    assert parent.shape[2] == 100 and parent.stride(2) == 2
    expected = _merge_stable_divergent_torch(parent, child, mask)

    monkeypatch.setattr(fused_ops, "_merge_stable_divergent_torch", _t013_forbidden_fallback)
    out = merge_stable_divergent(parent, child, mask)
    assert torch.equal(out, expected)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_t013_flags_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A disabled merge path must not stop gather_select from attempting Triton.

    With ``_TRITON_MERGE_DISABLED=True`` and a failing gather/select kernel,
    ``gather_select`` must warn, return the correct fallback result, and set
    ``_TRITON_GATHER_SELECT_DISABLED=True`` so later calls skip Triton silently.
    """
    monkeypatch.setattr(fused_ops, "_TRITON_MERGE_DISABLED", True)
    monkeypatch.setattr(fused_ops, "_TRITON_GATHER_SELECT_DISABLED", False, raising=False)
    monkeypatch.setattr(fused_ops, "_gather_select_kernel", _T013BoomKernel())

    parent, rows, child, mask = _t013_small_gather_inputs()
    expected = _t013_gather_expected(parent, rows, child, mask)

    with pytest.warns(RuntimeWarning, match="boom"):
        out = gather_select(parent, rows, child, mask)
    assert torch.equal(out, expected)
    assert fused_ops._TRITON_GATHER_SELECT_DISABLED is True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out2 = gather_select(parent, rows, child, mask)
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings
    assert torch.equal(out2, expected)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_t013_gather_select_failure_does_not_disable_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gather/select Triton failure must leave the merge Triton path enabled."""
    monkeypatch.setattr(fused_ops, "_TRITON_MERGE_DISABLED", False)
    monkeypatch.setattr(fused_ops, "_TRITON_GATHER_SELECT_DISABLED", False, raising=False)
    monkeypatch.setattr(fused_ops, "_gather_select_kernel", _T013BoomKernel())

    parent, rows, child, mask = _t013_small_gather_inputs()
    expected_g = _t013_gather_expected(parent, rows, child, mask)
    with pytest.warns(RuntimeWarning, match="boom"):
        out_g = gather_select(parent, rows, child, mask)
    assert torch.equal(out_g, expected_g)

    # Merge must still use Triton (its fallback is forbidden).
    mp = torch.randn(2, 4, 128, device="cuda")
    mc = torch.randn(2, 4, 128, device="cuda")
    mmask = torch.rand(2, 4, device="cuda") > 0.5
    expected_m = _merge_stable_divergent_torch(mp, mc, mmask)
    monkeypatch.setattr(fused_ops, "_merge_stable_divergent_torch", _t013_forbidden_fallback)
    out_m = merge_stable_divergent(mp, mc, mmask)
    assert torch.equal(out_m, expected_m)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_t013_merge_failure_does_not_disable_gather_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A merge Triton failure must leave the gather/select Triton path enabled."""
    monkeypatch.setattr(fused_ops, "_TRITON_MERGE_DISABLED", False)
    monkeypatch.setattr(fused_ops, "_TRITON_GATHER_SELECT_DISABLED", False, raising=False)

    mp = torch.randn(2, 4, 128, device="cuda")
    mc = torch.randn(2, 4, 128, device="cuda")
    mmask = torch.rand(2, 4, device="cuda") > 0.5
    expected_m = _merge_stable_divergent_torch(mp, mc, mmask)

    monkeypatch.setattr(fused_ops, "_merge_kernel", _T013BoomKernel())
    with pytest.warns(RuntimeWarning, match="boom"):
        out_m = merge_stable_divergent(mp, mc, mmask)
    assert torch.equal(out_m, expected_m)
    assert fused_ops._TRITON_MERGE_DISABLED is True

    # gather_select must still attempt Triton despite the merge failure.
    monkeypatch.setattr(fused_ops, "_gather_select_kernel", _T013BoomKernel())
    parent, rows, child, mask = _t013_small_gather_inputs()
    expected_g = _t013_gather_expected(parent, rows, child, mask)
    with pytest.warns(RuntimeWarning, match="boom"):
        out_g = gather_select(parent, rows, child, mask)
    assert torch.equal(out_g, expected_g)


# ---------------------------------------------------------------------------
# T016 (F8 / P1-3): fused gather_select at the D3 main-path threshold shape.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_t016_gather_select_large_shape_cuda_bit_exact() -> None:
    """T016: gather_select is bit-exact at the D3 threshold shape (T=2048, H=4096).

    Exercises the Triton kernel at the exact main-path activation shape from
    design decision D3: fp16, B=1, ~50% stable tokens, identity row mapping
    over a contiguous [T, H] parent buffer.
    """
    torch.manual_seed(1601)
    batch, seq, hidden = 1, 2048, 4096
    parent = torch.randn(seq, hidden, dtype=torch.float16, device="cuda")
    rows = torch.arange(seq, device="cuda").view(batch, seq)
    child = torch.randn(batch, seq, hidden, dtype=torch.float16, device="cuda")
    mask = torch.rand(batch, seq, device="cuda") > 0.5

    out = gather_select(parent, rows, child, mask)
    expected = torch.where(
        mask.unsqueeze(-1),
        parent[rows.reshape(-1)].reshape(batch, seq, hidden),
        child,
    )
    assert out.shape == (batch, seq, hidden)
    assert torch.equal(out, expected)
