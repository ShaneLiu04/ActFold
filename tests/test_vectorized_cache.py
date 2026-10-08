"""Tests for VectorizedActivationCache parity with the legacy ActivationCache."""

from __future__ import annotations

import pytest
import torch

from actfold.core.activation_cache import ActivationCache
from actfold.core.vectorized_cache import VectorizedActivationCache

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _make_activations(
    batch: int, seq: int, hidden: int, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    return {
        "ffn_out": torch.randn(batch, seq, hidden, dtype=dtype),
        "hidden_states": torch.randn(batch, seq, hidden, dtype=dtype),
    }


def _assert_parity(
    legacy: ActivationCache,
    vectorized: VectorizedActivationCache,
    mask: torch.Tensor,
    atol: float = 1e-6,
) -> None:
    out_legacy = legacy.get("b", 0, mask)
    out_vec = vectorized.get("b", 0, mask)
    for name in out_legacy:
        assert torch.allclose(
            out_legacy[name].float(), out_vec[name].float(), atol=atol
        ), f"mismatch for {name}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch,seq,hidden", [(1, 8, 16), (2, 5, 8), (1, 1, 4)])
def test_full_and_partial_mask_parity(
    dtype: torch.dtype, batch: int, seq: int, hidden: int
) -> None:
    acts = _make_activations(batch, seq, hidden, dtype)
    legacy = ActivationCache(max_entries_per_layer=64)
    vectorized = VectorizedActivationCache(max_entries_per_layer=64)
    for cache in (legacy, vectorized):
        cache.put("b", 0, acts)

    _assert_parity(legacy, vectorized, torch.ones(batch, seq, dtype=torch.bool))
    mask = torch.rand(batch, seq) > 0.4
    _assert_parity(legacy, vectorized, mask)


def test_eviction_parity() -> None:
    """When seq_len > capacity, both caches keep the most recent tokens."""
    batch, seq, hidden = 1, 8, 8
    acts = _make_activations(batch, seq, hidden)
    legacy = ActivationCache(max_entries_per_layer=4)
    vectorized = VectorizedActivationCache(max_entries_per_layer=4)
    for cache in (legacy, vectorized):
        cache.put("b", 0, acts)

    mask = torch.ones(batch, seq, dtype=torch.bool)
    _assert_parity(legacy, vectorized, mask)

    out = vectorized.get("b", 0, mask)
    assert (out["ffn_out"][:, :4, :] == 0).all()
    assert torch.allclose(out["ffn_out"][:, 4:, :], acts["ffn_out"][:, 4:, :])


def test_overwrite_parity() -> None:
    batch, seq, hidden = 1, 6, 8
    first = _make_activations(batch, seq, hidden)
    second = _make_activations(batch, seq, hidden)
    legacy = ActivationCache(max_entries_per_layer=32)
    vectorized = VectorizedActivationCache(max_entries_per_layer=32)
    for cache in (legacy, vectorized):
        cache.put("b", 0, first)
        cache.put("b", 0, second)

    mask = torch.ones(batch, seq, dtype=torch.bool)
    _assert_parity(legacy, vectorized, mask)
    out = vectorized.get("b", 0, mask)
    assert torch.allclose(out["ffn_out"], second["ffn_out"])


def test_num_entries_parity() -> None:
    batch, seq, hidden = 1, 6, 8
    acts = _make_activations(batch, seq, hidden)
    legacy = ActivationCache(max_entries_per_layer=4)
    vectorized = VectorizedActivationCache(max_entries_per_layer=4)
    for cache in (legacy, vectorized):
        cache.put("b", 0, acts)
    assert legacy.num_entries() == vectorized.num_entries() == 4
    assert legacy.num_entries(0) == vectorized.num_entries(0) == 4


def test_clear_operations() -> None:
    acts = _make_activations(1, 4, 8)
    cache = VectorizedActivationCache(max_entries_per_layer=16)
    cache.put("b", 0, acts)
    cache.put("c", 1, acts)
    assert cache.num_entries() == 8

    cache.clear_branch("b")
    assert cache.num_entries() == 4
    with pytest.raises(KeyError):
        cache.get("b", 0, torch.ones(1, 4, dtype=torch.bool))

    cache.clear_all()
    assert cache.num_entries() == 0


def test_batch_expand_parity() -> None:
    """A batch-1 entry fetched with a batch-2 mask expands like the legacy cache."""
    acts = _make_activations(1, 4, 8)
    legacy = ActivationCache(max_entries_per_layer=16)
    vectorized = VectorizedActivationCache(max_entries_per_layer=16)
    for cache in (legacy, vectorized):
        cache.put("b", 0, acts)
    mask = torch.ones(2, 4, dtype=torch.bool)
    _assert_parity(legacy, vectorized, mask)


def test_missing_first_token_still_reusable() -> None:
    """Tokens stored after an evicted first token remain retrievable."""
    acts = _make_activations(1, 6, 8)
    cache = VectorizedActivationCache(max_entries_per_layer=3)
    cache.put("b", 0, acts)
    mask = torch.ones(1, 6, dtype=torch.bool)
    out = cache.get("b", 0, mask)
    assert (out["ffn_out"][:, :3, :] == 0).all()
    assert torch.allclose(out["ffn_out"][:, 3:, :], acts["ffn_out"][:, 3:, :])


def test_get_missing_branch_raises() -> None:
    cache = VectorizedActivationCache(max_entries_per_layer=16)
    with pytest.raises(KeyError):
        cache.get("nope", 0, torch.ones(1, 4, dtype=torch.bool))


def test_validation_errors() -> None:
    cache = VectorizedActivationCache(max_entries_per_layer=16)
    with pytest.raises(ValueError):
        cache.put("b", 0, {})
    with pytest.raises(ValueError):
        cache.put("b", 0, {"x": torch.randn(4)})
    with pytest.raises(ValueError):
        cache.put(
            "b",
            0,
            {"x": torch.randn(1, 4, 8), "y": torch.randn(1, 5, 8)},
        )


def test_ring_eviction_keeps_recent_branch_steps() -> None:
    """Only the newest ``max_branch_steps`` (branch, step) keys survive puts."""
    cache = VectorizedActivationCache(max_entries_per_layer=16, max_branch_steps=2)
    acts = _make_activations(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)

    cache.put("b1", 0, acts, step_idx=0)
    cache.put("b1", 0, acts, step_idx=1)
    cache.put("b2", 0, acts, step_idx=0)

    # The oldest key ("b1", 0) was evicted once the third key was created.
    with pytest.raises(KeyError):
        cache.get("b1", 0, mask, step_idx=0)
    out_b1_s1 = cache.get("b1", 0, mask, step_idx=1)
    out_b2_s0 = cache.get("b2", 0, mask, step_idx=0)
    assert out_b1_s1["ffn_out"].shape == (1, 4, 8)
    assert out_b2_s0["ffn_out"].shape == (1, 4, 8)

    # Only the two surviving keys' rows remain (4 tokens per key).
    assert cache.num_entries() == 8


@pytest.mark.parametrize("max_branch_steps", [0, None])
def test_ring_eviction_disabled_when_zero_or_none(max_branch_steps: int | None) -> None:
    """``max_branch_steps`` of 0 or None disables (branch, step) eviction."""
    cache = VectorizedActivationCache(
        max_entries_per_layer=16, max_branch_steps=max_branch_steps
    )
    mask = torch.ones(1, 4, dtype=torch.bool)
    for step in range(5):
        cache.put(f"b{step}", 0, _make_activations(1, 4, 8), step_idx=step)

    for step in range(5):
        out = cache.get(f"b{step}", 0, mask, step_idx=step)
        assert out["ffn_out"].shape == (1, 4, 8)
    assert cache.num_entries() == 20


def test_ring_eviction_reput_same_key_does_not_evict() -> None:
    """Re-putting an existing (branch, step) key must not evict it."""
    cache = VectorizedActivationCache(max_entries_per_layer=16, max_branch_steps=1)
    acts = _make_activations(1, 4, 8)

    cache.put("b1", 0, acts, step_idx=0)
    cache.put("b1", 0, acts, step_idx=0)

    out = cache.get("b1", 0, torch.ones(1, 4, dtype=torch.bool), step_idx=0)
    assert out["ffn_out"].shape == (1, 4, 8)


def test_num_entries_bounded_across_steps() -> None:
    """Memory boundedness: fresh (branch, step) keys never exceed the ring cap."""
    cache = VectorizedActivationCache(max_entries_per_layer=16, max_branch_steps=2)
    acts = _make_activations(1, 16, 8)

    for step in range(10):
        cache.put(f"b{step}", 0, acts, step_idx=step)

    # At most 2 surviving (branch, step) keys x 16 token rows per key.
    assert cache.num_entries() <= 2 * 16


def test_get_all_returns_views_no_sync() -> None:
    """T010 (b): get_all returns full non-contiguous views with no mask.all() sync.

    ``get_all(branch_id, layer_idx)`` mirrors the full-mask fast path of
    ``get`` but takes no mask argument and must never call ``.all()``. It
    raises KeyError when nothing was stored for the branch/layer.
    """
    cache = VectorizedActivationCache(max_entries_per_layer=16)
    # Batch 2: with batch 1 the transposed view is trivially contiguous and
    # cannot witness the zero-copy view contract.
    cache.put("b", 0, {"h": torch.randn(2, 4, 8)})

    reference = cache.get("b", 0, torch.ones(2, 4, dtype=torch.bool))["h"]

    def _raise_sync(*args: object, **kwargs: object) -> None:
        raise AssertionError("sync")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "all", _raise_sync)
        result = cache.get_all("b", 0)

    assert result["h"].shape == (2, 4, 8)
    assert torch.equal(result["h"], reference)
    assert not result["h"].is_contiguous(), "get_all must return non-contiguous views"

    with pytest.raises(KeyError):
        cache.get_all("missing", 0)


# ---------------------------------------------------------------------------
# T014: storing only ffn_out (+ a layer-0 embedding) roughly halves the
# per-branch buffer bytes compared to the old {ffn_out, hidden_states} scheme.
# ---------------------------------------------------------------------------


def _buffer_bytes(cache: VectorizedActivationCache) -> int:
    """Total bytes across all internal storage buffers of the cache."""
    return sum(
        buf.numel() * buf.element_size()
        for branch in cache._buffers.values()
        for layer_store in branch.values()
        for buf in layer_store.values()
    )


def _simulate_scheme(num_layers: int, scheme: str) -> VectorizedActivationCache:
    """Store one branch of an ``num_layers`` model under the given scheme.

    ``new``: layer 0 stores ``{ffn_out, embedding}``, layers 1..L-1 only
    ``{ffn_out}`` (L+1 tensors total). ``old``: every layer stores
    ``{ffn_out, hidden_states}`` (2L tensors total).
    """
    batch, seq, hidden = 2, 8, 16
    sample = torch.randn(batch, seq, hidden)
    cache = VectorizedActivationCache(max_entries_per_layer=seq, max_branch_steps=0)
    for layer_idx in range(num_layers):
        if scheme == "new":
            activations: dict[str, torch.Tensor] = {"ffn_out": sample}
            if layer_idx == 0:
                activations["embedding"] = sample
        else:
            activations = {"ffn_out": sample, "hidden_states": sample}
        cache.put("b", layer_idx, activations)
    return cache


def test_t014_buffer_bytes_halved() -> None:
    """T014: the new scheme stores L+1 tensors per branch instead of 2L.

    For L=4 the new scheme needs exactly 5 tensors of ``[B, T, H]`` (layer 0:
    ffn_out + embedding; layers 1-3: ffn_out) versus 8 tensors for the old
    ``{ffn_out, hidden_states}`` scheme. The relative saving approaches 50% as
    L grows; it crosses below 60% of the old bytes once L >= 6.
    """
    batch, seq, hidden = 2, 8, 16
    per_tensor = batch * seq * hidden * torch.empty(1, dtype=torch.float32).element_size()

    num_layers = 4
    new_cache = _simulate_scheme(num_layers, "new")
    old_cache = _simulate_scheme(num_layers, "old")
    assert _buffer_bytes(new_cache) == (num_layers + 1) * per_tensor
    assert _buffer_bytes(old_cache) == 2 * num_layers * per_tensor
    assert _buffer_bytes(new_cache) < 0.7 * _buffer_bytes(old_cache)

    # The design's < 60% target is reached at deeper stacks; check it at L=8
    # (9/16 = 56.25% of the old bytes).
    deep = 8
    deep_new = _simulate_scheme(deep, "new")
    deep_old = _simulate_scheme(deep, "old")
    assert _buffer_bytes(deep_new) == (deep + 1) * per_tensor
    assert _buffer_bytes(deep_new) < 0.6 * _buffer_bytes(deep_old)


# ---------------------------------------------------------------------------
# T015 (F7): protocol fetch on complete coverage must be zero-allocation.
# ---------------------------------------------------------------------------


def test_t015_fetch_zero_allocation() -> None:
    """T015: a complete-coverage ``fetch`` returns views without any clone.

    ``torch.Tensor.clone`` is monkeypatched to raise, so ``fetch`` on a fully
    covered [B, T, H] entry must return non-contiguous views over the internal
    buffers (batch > 1 witnesses the view contract) and the values must equal
    what was put.
    """
    cache = VectorizedActivationCache(max_entries_per_layer=16)
    x = torch.randn(2, 4, 8)
    cache.put("b", 0, {"ffn_out": x})

    def _raise_clone(*args: object, **kwargs: object) -> None:
        raise AssertionError("clone allocated during fetch")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "clone", _raise_clone)
        out = cache.fetch("b", 0)

    assert out["ffn_out"].shape == x.shape
    assert torch.equal(out["ffn_out"], x)
    assert not out["ffn_out"].is_contiguous(), "complete-coverage fetch must return views"


# ---------------------------------------------------------------------------
# T016 (F8 / P1-3): fetch_flat exposes the raw [cap, B, ...] buffers as flat
# 2-D views plus the interleaved row indices needed by fused gather_select.
# ---------------------------------------------------------------------------


def test_t016_fetch_flat_complete_coverage() -> None:
    """T016: complete-coverage fetch_flat returns flat views plus row indices.

    After storing a [B=2, T=4, H=8] activation, ``fetch_flat`` returns per
    name a ``(flat_buffer, flat_rows)`` tuple where ``flat_buffer`` is a 2-D
    ``[cap * B, H]`` view sharing storage with the internal contiguous
    ``[cap, B, H]`` buffer, and ``flat_rows[b * T + t] == t * B + b`` so that
    ``flat_buffer[flat_rows].reshape(B, T, H)`` reproduces the stored tensor.
    """
    batch, seq, hidden = 2, 4, 8
    cache = VectorizedActivationCache(max_entries_per_layer=16)
    acts = _make_activations(batch, seq, hidden)
    cache.put("b", 0, acts)

    result = cache.fetch_flat("b", 0, batch_size=batch, seq_len=seq)

    assert result is not None
    assert set(result.keys()) == set(acts.keys())
    expected_rows = torch.tensor(
        [t * batch + b for b in range(batch) for t in range(seq)], dtype=torch.int64
    )
    for name, tensor in acts.items():
        internal = cache._buffers[("b", 0)][0][name]
        flat_buffer, flat_rows = result[name]
        capacity = internal.shape[0]
        assert flat_buffer.shape == (capacity * batch, hidden)
        # View contract: shares storage with the internal contiguous buffer.
        assert flat_buffer.data_ptr() == internal.data_ptr()
        assert (
            flat_buffer.untyped_storage().data_ptr()
            == internal.untyped_storage().data_ptr()
        )
        assert flat_rows.dtype == torch.int64
        assert flat_rows.shape == (batch * seq,)
        assert torch.equal(flat_rows, expected_rows)
        reconstructed = flat_buffer[flat_rows].reshape(batch, seq, hidden)
        assert torch.equal(reconstructed, tensor)


def test_t016_fetch_flat_none_cases() -> None:
    """T016: fetch_flat returns None whenever coverage is not complete.

    None is expected when (a) nothing was stored, (b) the stored token total
    is below the requested seq_len, (c) ring/eviction occurred (start > 0),
    or (d) the stored batch size differs from the requested one.
    """
    # (a) Nothing stored for the key.
    cache = VectorizedActivationCache(max_entries_per_layer=16)
    assert cache.fetch_flat("b", 0, batch_size=2, seq_len=4) is None

    # (b) Stored token total below the requested seq_len.
    cache.put("b", 0, _make_activations(2, 4, 8))
    assert cache.fetch_flat("b", 0, batch_size=2, seq_len=4) is not None
    assert cache.fetch_flat("b", 0, batch_size=2, seq_len=5) is None

    # (c) Ring/eviction: storing T=8 tokens with capacity 4 wraps, so even a
    # request within the capacity cannot be served.
    wrapped = VectorizedActivationCache(max_entries_per_layer=4)
    wrapped.put("b", 0, _make_activations(1, 8, 8))
    assert wrapped.fetch_flat("b", 0, batch_size=1, seq_len=4) is None

    # (d) Stored batch size differs from the requested one.
    assert cache.fetch_flat("b", 0, batch_size=3, seq_len=4) is None


def test_t016_fetch_flat_no_sync() -> None:
    """T016: fetch_flat performs no host synchronization.

    ``torch.Tensor.item``/``.any``/``.all`` are monkeypatched to raise for the
    duration of the call; a complete-coverage fetch_flat must complete and
    return a usable entry without triggering any of them.
    """
    cache = VectorizedActivationCache(max_entries_per_layer=16)
    cache.put("b", 0, _make_activations(2, 4, 8))

    def _raise_sync(*args: object, **kwargs: object) -> None:
        raise AssertionError("host sync in fetch_flat")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "item", _raise_sync)
        mp.setattr(torch.Tensor, "any", _raise_sync)
        mp.setattr(torch.Tensor, "all", _raise_sync)
        result = cache.fetch_flat("b", 0, batch_size=2, seq_len=4)

    assert result is not None
    assert "ffn_out" in result
