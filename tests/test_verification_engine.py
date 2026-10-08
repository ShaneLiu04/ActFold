"""Tests for actfold.speculative.verification_engine."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.speculative import ActFoldVerificationEngine, FastDLLMAdapter
from actfold.speculative.branch import Branch


class TinyModel(nn.Module):
    """Minimal embedding + head model for cache-population tests."""

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(tokens))


def test_t014_verification_engine_embedding_key() -> None:
    """T014: ``_ensure_parent_cache`` stores layer-0 embeddings as "embedding".

    The layer-0 entry must be readable under the name ``"embedding"`` and must
    not contain ``"hidden_states"`` or ``"ffn_out"``.
    """
    vocab_size = 16
    hidden_dim = 8
    model = TinyModel(vocab_size, hidden_dim)
    adapter = FastDLLMAdapter(model, num_layers=1, hidden_dim=hidden_dim)
    cache = ActivationCache(device="cpu")
    gate = SimilarityGate(tau=0.95)
    engine = ActFoldVerificationEngine(adapter, cache, gate)

    tokens = torch.randint(0, vocab_size, (1, 4))
    parent = Branch(branch_id="root", parent_id=None, tokens=tokens)
    engine._ensure_parent_cache(parent)

    full_mask = torch.ones(1, 4, dtype=torch.bool)
    activations = cache.get("root", layer_idx=0, token_mask=full_mask)
    assert "embedding" in activations
    assert "hidden_states" not in activations
    assert "ffn_out" not in activations
    assert activations["embedding"].shape == (1, 4, hidden_dim)
    assert torch.allclose(activations["embedding"], model.embedding(tokens))


def test_t015_ensure_parent_cache_uses_contains(monkeypatch: pytest.MonkeyPatch) -> None:
    """T015: ``_ensure_parent_cache`` checks presence via ``contains()``.

    ``cache.get`` is monkeypatched to raise so the old try/get/except-KeyError
    control flow cannot run. When ``contains`` returns False the engine must
    ``put`` the layer-0 ``{"embedding": ...}`` entry exactly once; when it
    returns True it must neither ``put`` nor raise.
    """
    vocab_size = 16
    hidden_dim = 8
    model = TinyModel(vocab_size, hidden_dim)
    adapter = FastDLLMAdapter(model, num_layers=1, hidden_dim=hidden_dim)
    cache = ActivationCache(device="cpu")
    gate = SimilarityGate(tau=0.95)
    engine = ActFoldVerificationEngine(adapter, cache, gate)

    tokens = torch.randint(0, vocab_size, (1, 4))
    parent = Branch(branch_id="root", parent_id=None, tokens=tokens)
    hidden = model.embedding(tokens)

    contains_calls: list[tuple[str, int, int]] = []
    put_calls: list[tuple[str, int, dict[str, torch.Tensor]]] = []

    def _contains_false(branch_id: str, layer_idx: int, step_idx: int = 0) -> bool:
        contains_calls.append((branch_id, layer_idx, step_idx))
        return False

    def _contains_true(branch_id: str, layer_idx: int, step_idx: int = 0) -> bool:
        contains_calls.append((branch_id, layer_idx, step_idx))
        return True

    real_put = cache.put

    def _record_put(
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        put_calls.append((branch_id, layer_idx, dict(activations)))
        real_put(branch_id, layer_idx, activations, step_idx=step_idx)

    def _raise_get(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy cache.get used by _ensure_parent_cache")

    monkeypatch.setattr(cache, "get", _raise_get)
    monkeypatch.setattr(cache, "put", _record_put)
    monkeypatch.setattr(cache, "contains", _contains_false, raising=False)

    # contains -> False: exactly one put of the real embeddings.
    engine._ensure_parent_cache(parent)

    assert ("root", 0, 0) in contains_calls
    assert len(put_calls) == 1
    branch_id, layer_idx, activations = put_calls[0]
    assert branch_id == "root"
    assert layer_idx == 0
    assert set(activations.keys()) == {"embedding"}
    assert torch.allclose(activations["embedding"], hidden)

    # contains -> True: no put, and no raise despite get being patched.
    monkeypatch.setattr(cache, "contains", _contains_true, raising=False)
    contains_calls.clear()
    put_calls.clear()

    engine._ensure_parent_cache(parent)

    assert ("root", 0, 0) in contains_calls
    assert put_calls == []
