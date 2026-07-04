"""Tests for actfold.core.folded_transformer."""

from __future__ import annotations

import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.folded_transformer import FoldedTransformerLayer


class DummyLayer(nn.Module):
    """Identity-like layer with small transformation for testing."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.linear(hidden_states)


def test_folded_layer_without_parent(device: str) -> None:
    hidden_dim = 32
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    x = torch.randn(2, 8, hidden_dim, device=device)
    out = folded(x, branch_id="root")
    expected = layer(x)
    assert torch.allclose(out, expected, atol=1e-5)


class TupleOutputLayer(nn.Module):
    """Layer that returns a tuple like many Hugging Face layers."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states * 2, hidden_states


def test_folded_layer_handles_tuple_output(device: str) -> None:
    """FoldedTransformerLayer extracts the first element from tuple outputs."""
    hidden_dim = 16
    layer = TupleOutputLayer().to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    x = torch.randn(2, 4, hidden_dim, device=device)
    out = folded(x, branch_id="root")
    assert out.shape == x.shape
    assert torch.allclose(out, x * 2)


def test_folded_layer_with_parent(device: str) -> None:
    hidden_dim = 32
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    # Parent forward populates cache.
    parent = torch.randn(2, 8, hidden_dim, device=device)
    parent_out = folded(parent, branch_id="parent")
    cache.put(
        branch_id="parent",
        layer_idx=0,
        activations={
            "ffn_out": parent_out,
            "hidden_states": parent,
        },
    )

    # Child nearly identical -> most tokens stable.
    child = parent + 1e-5 * torch.randn_like(parent)
    child_out = folded(child, branch_id="child", parent_branch_id="parent")

    assert child_out.shape == (2, 8, hidden_dim)
    # Stable path copies parent output, divergent path recomputes; both should
    # be close for near-identical inputs.
    assert torch.allclose(child_out, parent_out, atol=1e-3)


def test_folded_layer_divergent_only(device: str) -> None:
    """When no tokens are stable the layer recomputes without parent FFN."""
    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.99)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    parent = torch.randn(2, 4, hidden_dim, device=device)
    folded(parent, branch_id="parent")

    # Very different child should produce no stable tokens.
    child = torch.randn(2, 4, hidden_dim, device=device)
    child_out = folded(child, branch_id="child", parent_branch_id="parent")

    expected = layer(child)
    assert torch.allclose(child_out, expected, atol=1e-5)


def test_folded_layer_scheduler_disables_layer(device: str) -> None:
    """A scheduler that disables the layer forces full recomputation."""
    from actfold.core.folding_scheduler import FoldingScheduler

    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=2, num_steps=1)
    scheduler.disable_layers([0])
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0, scheduler=scheduler).to(device)

    parent = torch.randn(2, 4, hidden_dim, device=device)
    folded(parent, branch_id="parent")

    child = parent.clone()
    child_out = folded(child, branch_id="child", parent_branch_id="parent", step_idx=0)
    assert torch.allclose(child_out, layer(child), atol=1e-5)
