"""Tests for architecture detection and generic folding helpers."""

from __future__ import annotations

from typing import get_args, get_type_hints

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.model_wrapper import FoldedModel
from actfold.core.split_layer import SplitFoldedTransformerLayer
from actfold.models.architecture_utils import (
    ManualFoldedForward,
    detect_architecture,
    find_embedding_module,
    find_layer_list,
    find_lm_head,
)
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
from actfold.speculative.folded_generation import folded_generate


class LlamaLikeModel(nn.Module):
    """Mock architecture following the LLaMA layout."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": vocab_size})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model.layers = nn.ModuleList([_Layer(hidden_dim) for _ in range(num_layers)])
        self.model.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.model.lm_head(x)


class LlamaLikeModelWithFinalNorm(nn.Module):
    """Mock LLaMA architecture with a final norm before the LM head.

    Follows the real LLaMA layout ``embed -> layers -> model.norm -> lm_head``
    so that skipping the final norm produces systematically biased logits.
    """

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": vocab_size})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model.layers = nn.ModuleList([_Layer(hidden_dim) for _ in range(num_layers)])
        self.model.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        return self.lm_head(x)


class Gpt2LikeModel(nn.Module):
    """Mock architecture following the GPT-2 layout."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "gpt2", "vocab_size": vocab_size})()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab_size, hidden_dim)
        self.transformer.h = nn.ModuleList([_Layer(hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.transformer.wte(tokens)
        for layer in self.transformer.h:
            x = layer(x)
        return self.lm_head(x)


class BertLikeModel(nn.Module):
    """Mock architecture following the BERT layout."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "bert", "vocab_size": vocab_size})()
        self.bert = nn.Module()
        self.bert.embeddings = nn.Module()
        self.bert.embeddings.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.bert.encoder = nn.Module()
        self.bert.encoder.layer = nn.ModuleList([_Layer(hidden_dim) for _ in range(num_layers)])
        self.cls = nn.Module()
        self.cls.predictions = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.bert.embeddings.word_embeddings(tokens)
        for layer in self.bert.encoder.layer:
            x = layer(x)
        return self.cls.predictions(x)


class _Layer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_find_layer_list_llama() -> None:
    model = LlamaLikeModel()
    result = find_layer_list(model)
    assert result is not None
    path, layers = result
    assert path == "model.layers"
    assert len(layers) == 3


def test_find_embedding_module_llama() -> None:
    model = LlamaLikeModel()
    result = find_embedding_module(model)
    assert result is not None
    path, embed = result
    assert path == "model.embed_tokens"
    assert isinstance(embed, nn.Embedding)


def test_find_lm_head_llama() -> None:
    model = LlamaLikeModel()
    result = find_lm_head(model)
    assert result is not None
    path, head = result
    assert path == "model.lm_head"
    assert isinstance(head, nn.Linear)


def test_detect_architecture_gpt2() -> None:
    model = Gpt2LikeModel()
    profile = detect_architecture(model)
    assert profile.model_type == "gpt2"
    assert profile.layer_path == "transformer.h"
    assert profile.embed_path == "transformer.wte"
    assert profile.head_path == "lm_head"
    assert profile.supports_causal_mask is True


def test_detect_architecture_bert() -> None:
    model = BertLikeModel()
    profile = detect_architecture(model)
    assert profile.model_type == "bert"
    assert profile.layer_path == "bert.encoder.layer"
    assert profile.embed_path == "bert.embeddings.word_embeddings"
    assert profile.supports_causal_mask is False


def test_detect_architecture_missing_layers_raises() -> None:
    model = nn.Linear(10, 10)
    with pytest.raises(RuntimeError, match="Could not discover the Transformer layer list"):
        detect_architecture(model)


def test_manual_folded_forward_llama(device: str) -> None:
    model = LlamaLikeModel().to(device)
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    folded = ManualFoldedForward(model, cache=cache, gate=gate).to(device)

    tokens = torch.randint(0, 100, (1, 4), device=device)
    out = folded(tokens, branch_id="root")
    assert out.shape == (1, 4, 100)


def test_folded_model_auto_discovers_llama(device: str) -> None:
    model = LlamaLikeModel().to(device)
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    folded = FoldedModel(model, cache=cache, gate=gate)
    assert folded.folding_applied is True

    tokens = torch.randint(0, 100, (1, 4), device=device)
    out = folded(tokens, branch_id="root")
    assert out.shape == (1, 4, 100)


def test_folded_model_auto_discovers_gpt2(device: str) -> None:
    model = Gpt2LikeModel().to(device)
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    folded = FoldedModel(model, cache=cache, gate=gate)
    assert folded.folding_applied is True
    assert folded._layer_path == "transformer.h"


def test_detect_architecture_finds_final_norm() -> None:
    """detect_architecture must surface the final norm of a LLaMA-style model."""
    model = LlamaLikeModelWithFinalNorm()
    profile = detect_architecture(model)
    assert profile.final_norm is model.model.norm


def test_detect_architecture_no_final_norm_returns_none() -> None:
    """Models without any final norm must report ``final_norm is None``."""
    model = LlamaLikeModel()
    profile = detect_architecture(model)
    assert profile.final_norm is None


def test_manual_folded_forward_applies_final_norm(device: str) -> None:
    """Folded logits must match the original forward when a final norm exists (srs 3.1 B2).

    Without applying ``model.norm`` before the LM head, the logits are
    systematically biased relative to the model's own forward pass.
    """
    model = LlamaLikeModelWithFinalNorm().to(device)
    model.eval()
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    folded = ManualFoldedForward(model, cache=cache, gate=gate).to(device)
    folded.eval()

    tokens = torch.randint(0, 100, (2, 8), device=device)
    ref_logits = model(tokens)
    folded_logits = folded(tokens, branch_id="root")

    assert folded_logits.shape == ref_logits.shape
    assert torch.allclose(folded_logits, ref_logits, rtol=1e-5, atol=1e-5)
    assert torch.equal(folded_logits.argmax(dim=-1), ref_logits.argmax(dim=-1))


def test_manual_folded_forward_without_norm_still_correct(device: str) -> None:
    """Models without a final norm must keep matching the original forward."""
    model = LlamaLikeModel().to(device)
    model.eval()
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    folded = ManualFoldedForward(model, cache=cache, gate=gate).to(device)
    folded.eval()

    tokens = torch.randint(0, 100, (2, 8), device=device)
    ref_logits = model(tokens)
    folded_logits = folded(tokens, branch_id="root")

    assert folded_logits.shape == ref_logits.shape
    assert torch.allclose(folded_logits, ref_logits, rtol=1e-5, atol=1e-5)
    assert torch.equal(folded_logits.argmax(dim=-1), ref_logits.argmax(dim=-1))


def test_manual_path_never_reads_folding_context(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """AR002 srs 3.2 (UT-004a): the Manual path folds with a poisoned context.

    ``ManualFoldedForward`` passes branch identifiers down explicitly, so the
    thread-local ``FOLDING_CONTEXT`` (legacy ``FoldedModel`` fallback) is never
    consulted: with its ``get`` patched to raise, a parent->child folded
    forward still engages folding (child cache entries exist, the all-stable
    fast path returns the cached parent FFN) and stays bit-exact versus the
    unpoisoned reference run.
    """
    from actfold.core import folded_transformer as ft_module

    class _PoisonedContext:
        @staticmethod
        def get(*args: object, **kwargs: object) -> dict[str, object] | None:
            raise AssertionError("FOLDING_CONTEXT.get must not be called on the Manual path")

    torch.manual_seed(3)
    reference_model = LlamaLikeModel().to(device)
    model = LlamaLikeModel().to(device)
    model.load_state_dict(reference_model.state_dict())

    tokens = torch.randint(0, 100, (1, 4), device=device)

    def run(m: nn.Module) -> tuple[torch.Tensor, ActivationCache]:
        cache = ActivationCache(max_entries_per_layer=16, device=device)
        gate = SimilarityGate(tau=0.95, metric="cosine")
        folded = ManualFoldedForward(m, cache=cache, gate=gate).to(device)
        with torch.no_grad():
            folded(tokens, branch_id="parent")
            child_out = folded(tokens, branch_id="child", parent_branch_id="parent")
        return child_out, cache

    expected_out, _ = run(reference_model)

    monkeypatch.setattr(ft_module, "FOLDING_CONTEXT", _PoisonedContext())
    out, cache = run(model)

    assert torch.equal(out, expected_out)
    # Folding actually engaged through the explicit kwargs: the child branch
    # has its own layer-0 cache entry (stored via the folded fast path).
    child_entry = cache.fetch(branch_id="child", layer_idx=0)
    assert child_entry["ffn_out"] is not None


# --- AR002/T005 (srs 3.4): Manual 常态化 ---


def test_t005_manual_new_constructor_params(device: str) -> None:
    """ManualFoldedForward exposes split/graph constructor params (srs 3.4).

    The promoted constructor accepts ``split_layers`` / ``split_min_tokens`` /
    ``use_cuda_graph`` / ``graph_capacity_ratio`` while staying non-mutating:
    the base model's ``state_dict`` keys are untouched and a direct raw
    ``model(tokens)`` forward still runs after wrapping.
    """
    model = LlamaLikeModel().to(device)
    keys_before = set(model.state_dict().keys())
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    mff = ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        split_layers=True,
        split_min_tokens=64,
        use_cuda_graph=False,
        graph_capacity_ratio=0.5,
    )

    assert isinstance(mff._wrapped_layers[0], SplitFoldedTransformerLayer)
    assert mff.use_cuda_graph is False
    assert mff.graph_capacity_ratio == 0.5
    assert getattr(mff, "graph_runner", "unset") is None

    # Non-mutating wrap: the base model is unchanged.
    assert set(model.state_dict().keys()) == keys_before
    tokens = torch.randint(0, 100, (1, 4), device=device)
    with torch.no_grad():
        raw_out = model(tokens)
    assert isinstance(raw_out, torch.Tensor)
    assert raw_out.shape == (1, 4, 100)


def test_t005_manual_graph_ratio_validation(device: str) -> None:
    """graph_capacity_ratio must be validated to lie strictly in (0, 1]."""
    model = LlamaLikeModel().to(device)
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    with pytest.raises(ValueError):
        ManualFoldedForward(model, cache=cache, gate=gate, graph_capacity_ratio=0.0)
    with pytest.raises(ValueError):
        ManualFoldedForward(model, cache=cache, gate=gate, graph_capacity_ratio=1.5)


def test_t005_manual_split_fallback_without_chain(device: str) -> None:
    """split_layers on a chain-less model silently falls back to full recompute.

    ``BertLikeModel``'s mock layers expose no ``post_attention_layernorm->mlp``
    or ``ff_norm->ff_out`` chain, so ``SplitFoldedTransformerLayer`` cannot
    hook anything; construction must not raise and the folded forward must
    still produce full-sequence logits.
    """
    model = BertLikeModel().to(device)
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    mff = ManualFoldedForward(model, cache=cache, gate=gate, split_layers=True, split_min_tokens=64)

    tokens = torch.randint(0, 100, (1, 4), device=device)
    with torch.no_grad():
        out = mff(tokens, branch_id="root")
    assert out.shape == (1, 4, 100)


def test_t005_manual_bit_exact_vs_folded_model(device: str) -> None:
    """ManualFoldedForward must be bit-exact with the legacy FoldedModel stack.

    Two identically-seeded models run the same parent->child folded sequence,
    one through the context-managed in-place ``FoldedModel`` and one through
    the non-mutating ``ManualFoldedForward``; logits must match exactly on a
    fully-child pass and on a second child with different tokens.
    """
    torch.manual_seed(7)
    mA = LlamaLikeModel().to(device)
    snapshot = {k: v.clone() for k, v in mA.state_dict().items()}
    mB = LlamaLikeModel().to(device)
    mB.load_state_dict(snapshot)

    tokens = torch.randint(0, 100, (1, 4), device=device)
    tokens2 = torch.randint(0, 100, (1, 4), device=device)
    tokens3 = torch.randint(0, 100, (1, 4), device=device)
    num_layers = len(mA.model.layers)

    cacheA = ActivationCache(max_entries_per_layer=16, device=device)
    gateA = SimilarityGate(tau=0.95, metric="cosine")
    schedA = FoldingScheduler(base_tau=0.95, num_layers=num_layers, num_steps=2)
    with torch.no_grad():
        with FoldedModel(
            mA,
            cacheA,
            gateA,
            scheduler=schedA,
            split_layers=True,
            split_min_tokens=64,
        ) as fm:
            fm(tokens, branch_id="parent")
            outA = fm(tokens2, branch_id="child", parent_branch_id="parent")
            outA2 = fm(tokens3, branch_id="child2", parent_branch_id="parent")

    cacheB = ActivationCache(max_entries_per_layer=16, device=device)
    gateB = SimilarityGate(tau=0.95, metric="cosine")
    schedB = FoldingScheduler(base_tau=0.95, num_layers=num_layers, num_steps=2)
    mff = ManualFoldedForward(
        mB,
        cacheB,
        gateB,
        scheduler=schedB,
        split_layers=True,
        split_min_tokens=64,
    )
    with torch.no_grad():
        mff(tokens, branch_id="parent")
        outB = mff(tokens2, branch_id="child", parent_branch_id="parent")
        outB2 = mff(tokens3, branch_id="child2", parent_branch_id="parent")

    assert torch.equal(outA, outB)
    assert torch.equal(outA2, outB2)


def test_t005_folded_model_deprecated_docstring() -> None:
    """FoldedModel's docstring must flag it as deprecated vs ManualFoldedForward."""
    doc = (FoldedModel.__doc__ or "").lower()
    assert "deprecated" in doc
    assert "manualfoldedforward" in doc


def test_t005_folded_generate_annotation_union() -> None:
    """folded_generate and FastDLLMAdapter accept ManualFoldedForward too.

    ``get_type_hints`` resolves the string annotations via module globals;
    today ``Optional[FoldedModel]`` resolves, so the assertion fails only
    because ``ManualFoldedForward`` is not yet part of the union.
    """
    hints = get_type_hints(folded_generate)
    assert any(arg is ManualFoldedForward for arg in get_args(hints["folded_model"]))

    init_hints = get_type_hints(FastDLLMAdapter.__init__)
    assert any(arg is ManualFoldedForward for arg in get_args(init_hints["folded_model"]))
