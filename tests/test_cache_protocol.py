"""T015 (F7) tests: the unified ``ActivationCacheProtocol`` contract.

These tests specify the new cache protocol module
``actfold/core/cache_protocol.py`` (created in the Green phase): every cache
implementation must expose ``fetch`` / ``fetch_masked`` / ``get_all`` /
``contains`` on top of the existing ``put`` / ``clear_branch`` / ``clear_all``,
while the legacy ``get(token_mask)`` remains a backward-compatible alias of
``fetch_masked``.  All three caches (legacy ``ActivationCache``,
``ChunkedActivationCache``, ``VectorizedActivationCache``) are parametrized
over small CPU instances.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from actfold.core import ActivationCache, ChunkedActivationCache, VectorizedActivationCache

CACHE_FACTORIES: dict[str, Callable[[], Any]] = {
    "legacy": lambda: ActivationCache(max_entries_per_layer=64, device="cpu"),
    "chunked": lambda: ChunkedActivationCache(
        max_entries_per_layer=64,
        chunk_size=4,
        device="cpu",
    ),
    "vectorized": lambda: VectorizedActivationCache(max_entries_per_layer=64, device="cpu"),
}

CACHE_KINDS = sorted(CACHE_FACTORIES)


def _make_activations(batch: int, seq: int, hidden: int) -> dict[str, torch.Tensor]:
    """Build a small two-name activation dict with complete token coverage."""
    return {
        "ffn_out": torch.randn(batch, seq, hidden),
        "hidden_states": torch.randn(batch, seq, hidden),
    }


def _mixed_mask(batch: int, seq: int) -> torch.Tensor:
    """Return a boolean [batch, seq] mask with at least one True and one False."""
    mask = torch.rand(batch, seq) > 0.4
    assert 0 < int(mask.sum()) < mask.numel(), "test requires a partially-True mask"
    return mask


@pytest.mark.parametrize("cache_kind", CACHE_KINDS)
def test_fetch_returns_dense_values(cache_kind: str) -> None:
    """``fetch`` returns the dense raw activations that were put.

    The result keeps the stored names, the full ``[batch, seq, hidden]``
    shape, and (under complete coverage) values equal to what was written.
    """
    cache = CACHE_FACTORIES[cache_kind]()
    acts = _make_activations(2, 6, 8)
    cache.put("b", 0, acts)

    out = cache.fetch("b", 0)

    assert set(out.keys()) == set(acts.keys())
    for name, tensor in acts.items():
        assert out[name].shape == tensor.shape
        assert torch.equal(out[name], tensor)


@pytest.mark.parametrize("cache_kind", CACHE_KINDS)
def test_fetch_missing_raises_keyerror(cache_kind: str) -> None:
    """``fetch`` raises ``KeyError`` when nothing is stored for the key."""
    cache = CACHE_FACTORIES[cache_kind]()
    with pytest.raises(KeyError):
        cache.fetch("missing", 0)

    cache.put("b", 0, _make_activations(1, 4, 8))
    with pytest.raises(KeyError):
        cache.fetch("b", 0, step_idx=1)


@pytest.mark.parametrize("cache_kind", CACHE_KINDS)
def test_fetch_masked_matches_get(cache_kind: str) -> None:
    """``fetch_masked`` has the semantics of the legacy ``get``.

    With a partially-True mask, ``fetch_masked`` must return exactly what
    ``get`` returns, and masked-out positions must be zero-filled.
    """
    cache = CACHE_FACTORIES[cache_kind]()
    acts = _make_activations(2, 6, 8)
    cache.put("b", 0, acts)
    mask = _mixed_mask(2, 6)

    masked = cache.fetch_masked("b", 0, mask)
    legacy = cache.get("b", 0, mask)

    assert set(masked.keys()) == set(acts.keys())
    for name in acts:
        assert torch.equal(masked[name], legacy[name])
    assert bool((masked["ffn_out"][~mask] == 0).all()), "masked-out positions must be zero"


@pytest.mark.parametrize("cache_kind", CACHE_KINDS)
def test_get_all_full(cache_kind: str) -> None:
    """``get_all`` returns every stored name with the full values.

    Missing keys raise ``KeyError``.
    """
    cache = CACHE_FACTORIES[cache_kind]()
    acts = _make_activations(2, 6, 8)
    cache.put("b", 0, acts)

    out = cache.get_all("b", 0)

    assert set(out.keys()) == set(acts.keys())
    for name, tensor in acts.items():
        assert out[name].shape == tensor.shape
        assert torch.equal(out[name], tensor)

    with pytest.raises(KeyError):
        cache.get_all("missing", 0)


@pytest.mark.parametrize("cache_kind", CACHE_KINDS)
def test_contains(cache_kind: str) -> None:
    """``contains`` tracks stored (branch, layer, step) keys exactly."""
    cache = CACHE_FACTORIES[cache_kind]()
    assert cache.contains("b", 0) is False

    cache.put("b", 0, _make_activations(1, 4, 8))
    assert cache.contains("b", 0) is True
    assert cache.contains("b", 0, step_idx=1) is False
    assert cache.contains("other", 0) is False

    cache.clear_branch("b")
    assert cache.contains("b", 0) is False

    cache.put("b", 0, _make_activations(1, 4, 8))
    cache.clear_all()
    assert cache.contains("b", 0) is False


@pytest.mark.parametrize("cache_kind", CACHE_KINDS)
def test_clear_all_and_clear_branch_exist(cache_kind: str) -> None:
    """``clear_branch`` removes only its branch; ``clear_all`` empties the cache."""
    cache = CACHE_FACTORIES[cache_kind]()
    cache.put("b0", 0, _make_activations(1, 4, 8))
    cache.put("b1", 1, _make_activations(1, 4, 8))

    cache.clear_branch("b0")
    assert cache.contains("b0", 0) is False
    assert cache.contains("b1", 1) is True

    cache.clear_all()
    assert cache.contains("b1", 1) is False


@pytest.mark.parametrize("cache_kind", CACHE_KINDS)
def test_get_alias_backward_compatible(cache_kind: str) -> None:
    """The legacy ``get(token_mask)`` stays callable and matches ``fetch_masked``."""
    cache = CACHE_FACTORIES[cache_kind]()
    acts = _make_activations(2, 6, 8)
    cache.put("b", 0, acts)
    mask = _mixed_mask(2, 6)

    via_get = cache.get("b", 0, mask)
    via_fetch_masked = cache.fetch_masked("b", 0, mask)

    for name in acts:
        assert torch.equal(via_get[name], via_fetch_masked[name])


@pytest.mark.parametrize("cache_kind", CACHE_KINDS)
def test_protocol_runtime_checkable(cache_kind: str) -> None:
    """Every cache instance satisfies ``ActivationCacheProtocol`` at runtime.

    The protocol lives in the new ``actfold.core.cache_protocol`` module and
    must be ``@runtime_checkable`` so ``isinstance`` validates the method set.
    """
    from actfold.core.cache_protocol import ActivationCacheProtocol

    cache = CACHE_FACTORIES[cache_kind]()
    assert isinstance(cache, ActivationCacheProtocol)
