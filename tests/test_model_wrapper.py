"""Tests for actfold.core.model_wrapper."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.model_wrapper import FoldedModel


class TinyTransformer(nn.Module):
    """Minimal Transformer with a discoverable ``layers`` attribute."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, hidden_dim // 64),
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        logits: torch.Tensor = self.lm_head(x)
        return logits


def test_folded_model_wraps_layers(device: str) -> None:
    hidden_dim = 64
    model = TinyTransformer(vocab_size=100, hidden_dim=hidden_dim, num_layers=2).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedModel(model, cache, gate)

    assert folded.folding_applied is True
    assert folded._layer_path == "layers"
    assert folded._wrapped_layers is not None
    assert len(folded._wrapped_layers) == 2


def test_folded_model_forward_no_folding(device: str) -> None:
    class NoLayerModel(nn.Module):
        def forward(
            self,
            tokens: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            return tokens.float()

    model = NoLayerModel()
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    with pytest.warns(UserWarning, match="could not find a known layer stack"):
        folded = FoldedModel(model, cache, gate)

    assert folded.folding_applied is False
    x = torch.randint(0, 10, (2, 4), device=device)
    out = folded.forward(x, branch_id="root")
    assert torch.equal(out, x.float())


def test_folded_model_restore(device: str) -> None:
    hidden_dim = 64
    model = TinyTransformer(vocab_size=100, hidden_dim=hidden_dim, num_layers=2).to(device)
    original_layers = model.layers

    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedModel(model, cache, gate)

    assert model.layers is not original_layers
    restored = folded.restore()
    assert restored is model
    assert model.layers is original_layers
    assert folded._wrapped_layers is None


def test_folded_model_context_reaches_layers_when_kwargs_rejected(device: str) -> None:
    """Layers read branch context via contextvars when the base model drops kwargs."""

    class HFLikeModel(nn.Module):
        """Base model that does not accept ActFold-specific kwargs."""

        def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=max(1, hidden_dim // 64),
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            )
            self.lm_head = nn.Linear(hidden_dim, vocab_size)

        def forward(
            self,
            tokens: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            x = self.embedding(tokens)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(x)

    hidden_dim = 32
    model = HFLikeModel(vocab_size=100, hidden_dim=hidden_dim, num_layers=2).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedModel(model, cache, gate)
    assert folded.folding_applied is True

    tokens = torch.randint(0, 100, (1, 4), device=device)
    # Parent forward populates layer caches.
    _ = folded(tokens, branch_id="parent")

    # Child forward with parent_branch_id should reuse activations via context.
    out = folded(
        tokens,
        branch_id="child",
        parent_branch_id="parent",
        step_idx=0,
    )
    assert out.shape == (1, 4, 100)


def test_folded_model_preserves_user_kwargs_in_fallback(device: str) -> None:
    """User kwargs are preserved when falling back from ActFold kwargs."""

    class KwargsModel(nn.Module):
        def forward(
            self,
            tokens: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            custom_scale: float = 1.0,
            **kwargs: Any,
        ) -> torch.Tensor:
            return tokens.float() * custom_scale

    model = KwargsModel()
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    with pytest.warns(UserWarning, match="could not find a known layer stack"):
        folded = FoldedModel(model, cache, gate)

    x = torch.randint(0, 10, (2, 4), device=device)
    out = folded.forward(x, branch_id="root", custom_scale=2.0)
    assert torch.equal(out, x.float() * 2.0)


def test_folded_model_as_context_manager(device: str) -> None:
    """B12-lite: ``with FoldedModel(...)`` restores the base model on exit."""
    hidden_dim = 64
    model = TinyTransformer(vocab_size=100, hidden_dim=hidden_dim, num_layers=2).to(device)
    original_layers = model.layers
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)

    with FoldedModel(model, cache, gate) as folded:
        assert folded.folding_applied is True
        assert model.layers is not original_layers

    assert model.layers is original_layers
    assert folded.folding_applied is False


def test_folded_model_context_manager_restores_on_exception(device: str) -> None:
    """B12-lite: an exception inside the ``with`` block still restores layers."""
    hidden_dim = 64
    model = TinyTransformer(vocab_size=100, hidden_dim=hidden_dim, num_layers=2).to(device)
    original_layers = model.layers
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)

    with pytest.raises(RuntimeError, match="boom"):
        with FoldedModel(model, cache, gate):
            raise RuntimeError("boom")

    assert model.layers is original_layers


def test_folded_model_wrap_warns_inplace_mutation(device: str) -> None:
    """B12-lite: wrapping a foldable model warns about in-place mutation."""
    hidden_dim = 64
    model = TinyTransformer(vocab_size=100, hidden_dim=hidden_dim, num_layers=2).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)

    with pytest.warns(UserWarning, match="state_dict"):
        folded = FoldedModel(model, cache, gate)

    assert folded.folding_applied is True
    folded.restore()


def test_folded_model_del_restores_layers(device: str) -> None:
    """B12-lite: ``__del__`` is a last-resort restore when no ref remains."""
    import gc

    hidden_dim = 64
    model = TinyTransformer(vocab_size=100, hidden_dim=hidden_dim, num_layers=2).to(device)
    original_layers = model.layers
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)

    folded = FoldedModel(model, cache, gate)
    assert model.layers is not original_layers
    del folded
    gc.collect()

    assert model.layers is original_layers


def test_folded_model_no_reflection_in_forward(device: str) -> None:
    """T010 (d): FoldedModel.forward must not call inspect.signature per call.

    Construction-time reflection is allowed; after the wrapper is built,
    ``inspect.signature`` is patched to raise and both the folding-applied
    forward (branch_id only) and the attention_mask forward must complete.
    """
    hidden_dim = 64
    model = TinyTransformer(vocab_size=100, hidden_dim=hidden_dim, num_layers=2).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedModel(model, cache, gate)
    assert folded.folding_applied is True

    tokens = torch.randint(0, 100, (1, 4), device=device)

    def _raise_reflection(*args: object, **kwargs: object) -> None:
        raise AssertionError("reflection in forward")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(inspect, "signature", _raise_reflection)
        out_a = folded(tokens, branch_id="a")
        out_b = folded(
            tokens,
            branch_id="b",
            parent_branch_id="a",
            attention_mask=torch.ones(1, 4, dtype=torch.bool, device=device),
        )

    assert out_a.shape == (1, 4, 100)
    assert out_b.shape == (1, 4, 100)
