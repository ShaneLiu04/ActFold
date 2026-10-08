"""Tests for the split-layer FFN optimization."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.fused_ops import merge_stable_divergent
from actfold.core.model_wrapper import FoldedModel
from actfold.core.split_layer import SplitFoldedTransformerLayer, detect_split_spec


class RecordingMLP(nn.Module):
    """MLP that records the leading dimension of every input it sees."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.seen_shapes: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_shapes.append(tuple(x.shape))
        return self.fc2(self.act(self.fc1(x)))


class DummyAttention(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        return hidden_states * 0.5, None


class LlamaLikeLayer(nn.Module):
    """Pre-norm decoder layer with the standard module layout."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_attn = DummyAttention()
        self.mlp = RecordingMLP(hidden_dim)
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class PlainLayer(nn.Module):
    """Layer without a detectable split layout."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 3.0


def test_detect_split_spec_llama_like() -> None:
    layer = LlamaLikeLayer(16)
    spec = detect_split_spec(layer)
    assert spec is not None
    assert spec.pre_module is layer.post_attention_layernorm
    assert spec.post_module is layer.mlp


def test_detect_split_spec_plain_returns_none() -> None:
    assert detect_split_spec(PlainLayer()) is None


def test_split_matches_full_recompute_on_divergent_rows() -> None:
    torch.manual_seed(0)
    hidden_dim = 16
    layer = LlamaLikeLayer(hidden_dim)
    parent = torch.randn(1, 6, hidden_dim)

    # Two divergent tokens, four stable ones.
    child = parent.clone()
    child[:, :2, :] = torch.randn(1, 2, hidden_dim)

    expected_full = layer(child)

    cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    gate = SimilarityGate(tau=0.99)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=0)
    assert split.split_enabled
    with torch.no_grad():
        split(parent, branch_id="parent")
        out_split = split(child, branch_id="child", parent_branch_id="parent")

    # Divergent positions must match the full recompute to floating-point
    # tolerance (identical inputs and weights, possibly different GEMM blocking).
    assert torch.allclose(out_split[:, :2, :], expected_full[:, :2, :], atol=1e-6)
    # Stability is decided by the gate; with tau=0.99 the unchanged tokens are
    # stable and copy the parent output.
    assert torch.allclose(out_split[:, 2:, :], layer(parent)[:, 2:, :], atol=1e-6)


def test_split_mlp_receives_only_divergent_rows() -> None:
    torch.manual_seed(1)
    hidden_dim = 8
    layer = LlamaLikeLayer(hidden_dim)
    parent = torch.randn(1, 5, hidden_dim)
    child = parent.clone()
    child[:, 3, :] = torch.randn(hidden_dim)

    cache = ActivationCache(max_entries_per_layer=32, device="cpu")
    gate = SimilarityGate(tau=0.99)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=0)
    with torch.no_grad():
        split(parent, branch_id="parent")
        split(child, branch_id="child", parent_branch_id="parent")

    # The final MLP call during the split recompute must see a 2-D, single-row
    # tensor (the divergent token), not the full [1, 5, H] sequence.
    assert len(layer.mlp.seen_shapes[-1]) == 2
    assert layer.mlp.seen_shapes[-1][0] == 1
    assert layer.mlp.seen_shapes[-1][-1] == hidden_dim


def test_split_falls_back_without_spec() -> None:
    layer = PlainLayer()
    cache = ActivationCache(max_entries_per_layer=16, device="cpu")
    gate = SimilarityGate(tau=0.95)
    folded = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=0)
    assert not folded.split_enabled
    x = torch.randn(1, 3, 8)
    out = folded(x, branch_id="b")
    assert torch.allclose(out, x * 3.0)


def test_split_all_stable_fast_path() -> None:
    hidden_dim = 8
    layer = LlamaLikeLayer(hidden_dim)
    parent = torch.randn(1, 4, hidden_dim)

    cache = ActivationCache(max_entries_per_layer=32, device="cpu")
    gate = SimilarityGate(tau=0.5)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=0)
    with torch.no_grad():
        split(parent, branch_id="parent")
        out = split(parent.clone(), branch_id="child", parent_branch_id="parent")
    assert torch.equal(out, layer(parent))


def test_min_split_tokens_auto_disables() -> None:
    """Small token counts fall back to full recompute to avoid gather syncs."""
    torch.manual_seed(2)
    hidden_dim = 8
    layer = LlamaLikeLayer(hidden_dim)
    parent = torch.randn(1, 5, hidden_dim)
    child = parent.clone()
    child[:, 2, :] = torch.randn(hidden_dim)

    cache = ActivationCache(max_entries_per_layer=32, device="cpu")
    gate = SimilarityGate(tau=0.99)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=256)
    with torch.no_grad():
        split(parent, branch_id="parent")
        out = split(child, branch_id="child", parent_branch_id="parent")

    # The MLP saw the full 3-D sequence (no slicing) and the output equals a
    # plain folded recompute.
    assert layer.mlp.seen_shapes[-1] == (1, 5, hidden_dim)
    assert out.shape == child.shape


def test_split_matches_base_folded_layer_partial_mask() -> None:
    """Split and non-split folded layers agree exactly on real partial masks."""
    torch.manual_seed(3)
    hidden_dim = 12
    parent = torch.randn(1, 7, hidden_dim)
    child = parent.clone()
    child[:, 1, :] = torch.randn(hidden_dim)
    child[:, 4, :] = torch.randn(hidden_dim)

    results = {}
    for name, cls in (
        ("base", FoldedTransformerLayer),
        ("split", SplitFoldedTransformerLayer),
    ):
        torch.manual_seed(0)
        layer = LlamaLikeLayer(hidden_dim)
        cache = ActivationCache(max_entries_per_layer=64, device="cpu")
        gate = SimilarityGate(tau=0.99)
        kwargs = {"min_split_tokens": 0} if cls is SplitFoldedTransformerLayer else {}
        folded = cls(layer, cache, gate, layer_idx=0, **kwargs)
        with torch.no_grad():
            folded(parent, branch_id="parent")
            results[name] = folded(child, branch_id="child", parent_branch_id="parent")
    assert torch.allclose(results["base"], results["split"], atol=1e-6)


# ---------------------------------------------------------------------------
# T017 (F9): zero-free split merge + resident hooks
# ---------------------------------------------------------------------------


class TinyLlamaModel(nn.Module):
    """Minimal llama-style model exposing a ``layers`` ModuleList.

    The branch identifiers are accepted so ``FoldedModel`` classifies the
    model as kwargs-capable; the wrapped layers read them from the
    thread-local folding context instead.
    """

    def __init__(self, hidden_dim: int, num_layers: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(LlamaLikeLayer(hidden_dim) for _ in range(num_layers))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        branch_id: str | None = None,
        parent_branch_id: str | None = None,
        step_idx: int = 0,
    ) -> torch.Tensor:
        """Run the tiny model; branch identifiers are consumed by folded layers."""
        del branch_id, parent_branch_id, step_idx
        hidden = self.embed(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


def test_t017_no_zeros_allocation_in_split_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """T017(a): the split merge path never allocates via ``torch.zeros``.

    The scatter base in ``_post_hook`` must be ``torch.empty``; stable rows
    are don't-care because the subsequent merge overwrites them.  The final
    output stays bit-exact vs an unwrapped ``FoldedTransformerLayer``.
    """
    torch.manual_seed(0)
    hidden_dim = 16
    layer = LlamaLikeLayer(hidden_dim)
    parent = torch.randn(1, 8, hidden_dim)
    child = parent.clone()
    child[:, 1, :] = torch.randn(1, 1, hidden_dim)
    child[:, 5, :] = torch.randn(1, 1, hidden_dim)

    cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    gate = SimilarityGate(tau=0.99)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=4)
    with torch.no_grad():
        split(parent, branch_id="parent")

    # Baseline: identical non-split folded layer with identical weights.
    torch.manual_seed(0)
    base_layer = LlamaLikeLayer(hidden_dim)
    base = FoldedTransformerLayer(
        base_layer,
        ActivationCache(max_entries_per_layer=64, device="cpu"),
        SimilarityGate(tau=0.99),
        layer_idx=0,
    )
    with torch.no_grad():
        base(parent, branch_id="parent")
        expected = base(child, branch_id="child", parent_branch_id="parent")

    def _no_zeros(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("torch.zeros must never be called by the split merge path")

    monkeypatch.setattr(torch, "zeros", _no_zeros)
    with torch.no_grad(), warnings.catch_warnings():
        # A zero-fill fallback would disable the split with a RuntimeWarning.
        warnings.simplefilter("error", RuntimeWarning)
        out = split(child, branch_id="child", parent_branch_id="parent")

    assert split.split_enabled
    # The split path actually engaged: the MLP saw only the divergent rows.
    assert len(layer.mlp.seen_shapes[-1]) == 2
    assert torch.equal(out, expected)


def test_t017_hooks_registered_once_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """T017(b): hooks are resident from construction; no per-forward registration."""
    torch.manual_seed(1)
    hidden_dim = 16
    layer = LlamaLikeLayer(hidden_dim)
    parent = torch.randn(1, 8, hidden_dim)
    child = parent.clone()
    child[:, 3, :] = torch.randn(1, 1, hidden_dim)

    cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    gate = SimilarityGate(tau=0.99)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=4)
    spec = split.split_spec
    assert spec is not None

    # Resident hooks: present immediately after construction, before any forward.
    assert len(spec.pre_module._forward_pre_hooks) == 1
    assert len(spec.post_module._forward_hooks) == 1

    # Correctness without any monkeypatch interference: two consecutive mixed
    # forwards through the resident hooks stay deterministic and split-engaged.
    with torch.no_grad():
        split(parent, branch_id="parent")
        out1 = split(child, branch_id="child", parent_branch_id="parent")
        out2 = split(child, branch_id="child2", parent_branch_id="parent")
    assert torch.equal(out1, out2)
    assert len(layer.mlp.seen_shapes[-1]) == 2

    # No hook registration/removal may happen during the forwards.
    calls = {"pre": 0, "post": 0}
    orig_pre = nn.Module.register_forward_pre_hook
    orig_post = nn.Module.register_forward_hook

    def counting_pre(mod: nn.Module, *args: Any, **kwargs: Any) -> Any:
        calls["pre"] += 1
        return orig_pre(mod, *args, **kwargs)

    def counting_post(mod: nn.Module, *args: Any, **kwargs: Any) -> Any:
        calls["post"] += 1
        return orig_post(mod, *args, **kwargs)

    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", counting_pre)
    monkeypatch.setattr(nn.Module, "register_forward_hook", counting_post)
    with torch.no_grad():
        split(child, branch_id="child3", parent_branch_id="parent")
        split(child, branch_id="child4", parent_branch_id="parent")
    assert calls["pre"] == 0
    assert calls["post"] == 0


def test_t017_transparent_without_folding_context() -> None:
    """T017(b): resident hooks are fully transparent without a folding context.

    With no parent branch (``parent_branch_id=None``) and on direct calls to
    the wrapped original layer, the output must be bit-exact vs the same
    module never being wrapped.
    """
    torch.manual_seed(2)
    hidden_dim = 16
    layer = LlamaLikeLayer(hidden_dim)
    # Twin with identical weights, never wrapped by a split layer.
    torch.manual_seed(2)
    twin = LlamaLikeLayer(hidden_dim)

    hidden = torch.randn(2, 6, hidden_dim)
    child = hidden.clone()
    child[:, 0, :] = torch.randn(2, hidden_dim)

    cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    gate = SimilarityGate(tau=0.99)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=4)

    # Parent-pass forward (no parent branch): resident hooks must be inert.
    with torch.no_grad():
        out = split(hidden, branch_id="orphan")
    assert torch.equal(out, twin(hidden))

    # After a folded (state-setting) forward, the state is cleared and direct
    # calls through the still-hooked original layer remain bit-exact.
    with torch.no_grad():
        split(hidden, branch_id="p")
        split(child, branch_id="c", parent_branch_id="p")
    assert split._split_state is None
    assert torch.equal(split.original_layer(child), twin(child))


def test_t017_remove_hooks_idempotent_and_restore_cleans() -> None:
    """T017(b): ``remove_hooks`` is idempotent; ``FoldedModel.restore`` cleans up."""
    torch.manual_seed(3)
    hidden_dim = 16
    layer = LlamaLikeLayer(hidden_dim)
    parent = torch.randn(1, 8, hidden_dim)
    child = parent.clone()
    child[:, 2, :] = torch.randn(1, 1, hidden_dim)

    cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    gate = SimilarityGate(tau=0.99)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=4)
    with torch.no_grad():
        split(parent, branch_id="parent")

    # Idempotent removal: safe to call twice, and the defensive __del__ must
    # never raise even after the handles are already gone.
    split.remove_hooks()
    split.remove_hooks()
    split.__del__()

    # With the hooks gone, a mixed forward equals the FULL recompute + merge
    # (the split state is set but no hook fires, so nothing is sliced).
    stable_mask = gate(child, parent)
    expected = merge_stable_divergent(layer(parent), layer(child), stable_mask)
    with torch.no_grad():
        out = split(child, branch_id="child", parent_branch_id="parent")
    assert torch.equal(out, expected)
    # Full recompute ran: the MLP saw the whole 3-D sequence.
    assert layer.mlp.seen_shapes[-1] == (1, 8, hidden_dim)

    # FoldedModel.restore() removes resident hooks BEFORE swapping the
    # original layer list back: restored models carry no leftover hooks and
    # produce bit-exact outputs vs a never-wrapped model.
    torch.manual_seed(4)
    wrapped = TinyLlamaModel(hidden_dim, num_layers=2)
    pristine = TinyLlamaModel(hidden_dim, num_layers=2)
    pristine.load_state_dict(wrapped.state_dict())
    tokens = torch.randint(0, 32, (1, 8))
    expected_logits = pristine(tokens)

    fm = FoldedModel(
        wrapped,
        ActivationCache(max_entries_per_layer=64, device="cpu"),
        SimilarityGate(tau=0.99),
        split_layers=True,
        split_min_tokens=4,
    )
    child_tokens = tokens.clone()
    child_tokens[:, 4:] = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        fm(tokens, branch_id="parent")
        fm(child_tokens, branch_id="child", parent_branch_id="parent")
    restored = fm.restore()
    with torch.no_grad():
        out_restored = restored(tokens)
    assert torch.equal(out_restored, expected_logits)
    for lyr in restored.layers:
        assert len(lyr.post_attention_layernorm._forward_pre_hooks) == 0
        assert len(lyr.mlp._forward_hooks) == 0


def test_t017_no_sync_guards_in_recompute_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    """T017(b): the mixed path must not call ``Tensor.all()``/``.any()`` syncs.

    The forward slow path only reaches ``_recompute_merged`` with a mixed
    mask, so the two-sync guard is redundant and must be gone.
    """
    torch.manual_seed(5)
    hidden_dim = 16
    layer = LlamaLikeLayer(hidden_dim)
    parent = torch.randn(1, 8, hidden_dim)
    child = parent.clone()
    child[:, 6, :] = torch.randn(1, 1, hidden_dim)

    cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    gate = SimilarityGate(tau=0.99)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=4)

    def _no_all(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("Tensor.all() must not be called on the split path")

    def _no_any(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("Tensor.any() must not be called on the split path")

    monkeypatch.setattr(torch.Tensor, "all", _no_all)
    monkeypatch.setattr(torch.Tensor, "any", _no_any)
    with torch.no_grad():
        split(parent, branch_id="parent")
        out = split(child, branch_id="child", parent_branch_id="parent")
    assert out.shape == child.shape
    # The split actually engaged (mixed mask, above min_split_tokens).
    assert len(layer.mlp.seen_shapes[-1]) == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_t017_split_output_bit_exact_vs_baseline(dtype: torch.dtype) -> None:
    """T017: mixed-stability split output is bit-exact vs the unsplit merge."""
    torch.manual_seed(6)
    batch, seq_len, hidden_dim = 1, 8, 16
    layer = LlamaLikeLayer(hidden_dim).to(dtype)
    parent = torch.randn(batch, seq_len, hidden_dim, dtype=dtype)
    child = parent.clone()
    child[:, 0::2, :] = torch.randn(1, 4, hidden_dim, dtype=dtype)

    cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    gate = SimilarityGate(tau=0.99)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=4)
    with torch.no_grad():
        split(parent, branch_id="parent")
        out = split(child, branch_id="child", parent_branch_id="parent")

    stable_mask = gate(child, parent)
    assert 0 < int(stable_mask.sum()) < stable_mask.numel()
    reference = merge_stable_divergent(layer(parent), layer(child), stable_mask)
    assert torch.equal(out, reference)
