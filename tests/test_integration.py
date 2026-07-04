"""End-to-end integration tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, FoldedModel, SimilarityGate
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.speculative import (
    ActFoldVerificationEngine,
    DraftGenerator,
    FastDLLMAdapter,
    SpiffyBaseline,
)
from actfold.speculative.branch import Branch
from actfold.utils.flops_counter import count_diffusion_llm_flops


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(tokens))


def test_verification_engine_validates_threshold() -> None:
    """ActFoldVerificationEngine rejects invalid acceptance thresholds."""
    model = TinyModel(10, 8)
    cache = ActivationCache(device="cpu")
    gate = SimilarityGate(tau=0.95)
    with pytest.raises(ValueError):
        ActFoldVerificationEngine(model, cache, gate, acceptance_threshold=-0.1)
    with pytest.raises(ValueError):
        ActFoldVerificationEngine(model, cache, gate, acceptance_threshold=1.5)


def test_verification_engine_parent_cache_stores_embeddings() -> None:
    """The engine stores only input embeddings (not ffn_out) in layer-0 cache."""
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

    activations = cache.get("root", layer_idx=0, token_mask=torch.ones(1, 4, dtype=torch.bool))
    assert "hidden_states" in activations
    assert "ffn_out" not in activations
    assert activations["hidden_states"].shape == (1, 4, hidden_dim)


def test_verification_engine_uses_folded_model(device: str) -> None:
    """When the adapter has a FoldedModel, the engine populates parent layer caches."""
    vocab_size = 32
    hidden_dim = 16
    num_layers = 2

    class TinyLayer(nn.Module):
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return hidden_states * 1.01

    class TinyLayersModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList(TinyLayer() for _ in range(num_layers))
            self.head = nn.Linear(hidden_dim, vocab_size)

        def forward(
            self,
            tokens: torch.Tensor,
            branch_id: str = "",
            parent_branch_id: str | None = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            x = self.embedding(tokens)
            for layer in self.layers:
                x = layer(
                    x,
                    branch_id=branch_id,
                    parent_branch_id=parent_branch_id,
                    **kwargs,
                )
            return self.head(x)

    raw = TinyLayersModel().to(device)
    cache = ActivationCache(max_entries_per_layer=128, device=device)
    gate = SimilarityGate(tau=0.95)
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=num_layers, num_steps=4)
    folded = FoldedModel(raw, cache, gate, scheduler=scheduler)
    adapter = FastDLLMAdapter(
        raw, num_layers=num_layers, hidden_dim=hidden_dim, folded_model=folded
    )

    engine = ActFoldVerificationEngine(adapter, cache, gate, scheduler)
    parent_tokens = torch.randint(0, vocab_size, (1, 4), device=device)
    child_tokens = parent_tokens.clone()
    parent = Branch(branch_id="root", parent_id=None, tokens=parent_tokens)
    child = Branch(branch_id="child", parent_id="root", tokens=child_tokens)

    result = engine.verify_branch(parent, child, step_idx=0)
    assert result.accepted
    assert 0.0 <= result.stable_ratio <= 1.0
    # Parent layer caches should have been populated.
    assert cache.num_entries(layer_idx=0) > 0
    assert cache.num_entries(layer_idx=num_layers - 1) > 0


def test_demo_pipeline(device: str) -> None:
    vocab_size = 100
    hidden_dim = 32
    num_layers = 2
    seq_len = 8
    num_branches = 2

    torch.manual_seed(42)
    model = TinyModel(vocab_size, hidden_dim).to(device)
    adapter = FastDLLMAdapter(model, num_layers=num_layers, hidden_dim=hidden_dim)
    draft_generator = DraftGenerator(vocab_size=vocab_size, mode="random")

    parent_tokens = torch.randint(0, vocab_size, (1, seq_len), device=device)
    parent = Branch(branch_id="root", parent_id=None, tokens=parent_tokens)
    children = draft_generator.generate(parent, num_branches=num_branches, seed=42)

    # Baseline.
    baseline = SpiffyBaseline(adapter, draft_generator)
    accepted_baseline = baseline.verify(children)
    assert accepted_baseline.accepted

    # ActFold.
    cache = ActivationCache(max_entries_per_layer=128, device=device)
    gate = SimilarityGate(tau=0.95)
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=num_layers, num_steps=4)
    engine = ActFoldVerificationEngine(adapter, cache, gate, scheduler)

    results = [engine.verify_branch(parent, child, step_idx=0) for child in children]
    best = max(results, key=lambda r: r.child_branch.metadata.get("actfold_score", 0))
    assert best.accepted
    assert 0.0 <= best.stable_ratio <= 1.0
    assert best.tflops >= 0.0

    # TFLOPs counter sanity check.
    flops = count_diffusion_llm_flops(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=4,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_steps=1,
        reuse_ratio=best.stable_ratio,
    )
    assert flops.total_tflops >= 0.0
