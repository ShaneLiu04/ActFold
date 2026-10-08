"""Tests for the chunked activation cache."""

from __future__ import annotations

import pytest
import torch

from actfold.core.chunked_cache import ChunkedActivationCache


def _make_activations(batch: int, seq_len: int, hidden_dim: int) -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.randn(batch, seq_len, hidden_dim),
        "ffn_out": torch.randn(batch, seq_len, hidden_dim),
    }


def test_put_and_get_matches_original() -> None:
    """Stored activations can be retrieved and match the original values."""
    cache = ChunkedActivationCache(max_entries_per_layer=128, chunk_size=4, device="cpu")
    acts = _make_activations(batch=1, seq_len=8, hidden_dim=16)
    cache.put("branch", layer_idx=0, activations=acts)

    mask = torch.ones((1, 8), dtype=torch.bool)
    retrieved = cache.get("branch", layer_idx=0, token_mask=mask)

    assert retrieved is not None
    assert torch.allclose(retrieved["hidden_states"], acts["hidden_states"])
    assert torch.allclose(retrieved["ffn_out"], acts["ffn_out"])


def test_partial_mask_returns_zero_filled_full_length() -> None:
    """A partial mask keeps [B, T, H] and zero-fills unrequested positions."""
    cache = ChunkedActivationCache(max_entries_per_layer=128, chunk_size=4, device="cpu")
    acts = _make_activations(batch=1, seq_len=8, hidden_dim=16)
    cache.put("branch", layer_idx=0, activations=acts)

    mask = torch.tensor([[False, True, False, True, False, False, False, False]])
    retrieved = cache.get("branch", layer_idx=0, token_mask=mask)

    assert retrieved["hidden_states"].shape == (1, 8, 16)
    expected = torch.where(
        mask.unsqueeze(-1), acts["hidden_states"], torch.zeros_like(acts["hidden_states"])
    )
    assert torch.allclose(retrieved["hidden_states"], expected)


def test_clear_branch_removes_entries() -> None:
    """clear_branch evicts all chunks for a branch."""
    cache = ChunkedActivationCache(max_entries_per_layer=128, chunk_size=2, device="cpu")
    cache.put("b1", 0, _make_activations(1, 4, 8))
    cache.put("b2", 0, _make_activations(1, 4, 8))

    cache.clear_branch("b1")
    mask = torch.ones((1, 4), dtype=torch.bool)
    with pytest.raises(KeyError):
        cache.get("b1", 0, mask)
    assert cache.get("b2", 0, mask)


def test_chunked_matches_legacy_parity() -> None:
    """Chunked cache must match the legacy cache contract on all mask shapes."""
    from actfold.core.activation_cache import ActivationCache

    for batch, seq_len, hidden in ((1, 8, 16), (2, 5, 8), (1, 1, 4)):
        acts = _make_activations(batch, seq_len, hidden)
        legacy = ActivationCache(max_entries_per_layer=128, device="cpu")
        chunked = ChunkedActivationCache(max_entries_per_layer=128, chunk_size=4, device="cpu")
        for cache in (legacy, chunked):
            cache.put("branch", layer_idx=0, activations=acts)

        for mask in (
            torch.ones(batch, seq_len, dtype=torch.bool),
            torch.rand(batch, seq_len) > 0.4,
        ):
            out_legacy = legacy.get("branch", 0, mask)
            out_chunked = chunked.get("branch", 0, mask)
            for name in out_legacy:
                assert torch.allclose(
                    out_legacy[name], out_chunked[name]
                ), f"parity failure batch={batch} seq={seq_len} name={name}"


def test_chunked_batch_expand_parity() -> None:
    from actfold.core.activation_cache import ActivationCache

    acts = _make_activations(1, 4, 8)
    legacy = ActivationCache(max_entries_per_layer=128, device="cpu")
    chunked = ChunkedActivationCache(max_entries_per_layer=128, chunk_size=2, device="cpu")
    for cache in (legacy, chunked):
        cache.put("branch", 0, acts)
    mask = torch.ones(2, 4, dtype=torch.bool)
    out_legacy = legacy.get("branch", 0, mask)
    out_chunked = chunked.get("branch", 0, mask)
    for name in out_legacy:
        assert torch.allclose(out_legacy[name], out_chunked[name])


def test_num_tokens_accounts_for_chunks() -> None:
    """num_tokens returns the actual number of valid token slots."""
    cache = ChunkedActivationCache(max_entries_per_layer=128, chunk_size=4, device="cpu")
    acts = _make_activations(batch=1, seq_len=10, hidden_dim=8)
    cache.put("branch", 0, acts)
    assert cache.num_tokens() == 10


def test_invalid_chunk_size_raises() -> None:
    """A non-positive chunk size is rejected."""
    with pytest.raises(ValueError):
        ChunkedActivationCache(chunk_size=0)
