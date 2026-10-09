"""Tests for the split-layer FFN optimization."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core import split_layer as split_layer_module
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


# ---------------------------------------------------------------------------
# AR002 / T001: nonzero-sync-free divergent indexing
# ---------------------------------------------------------------------------


def _exact_divergent_index() -> Callable[[torch.Tensor], torch.Tensor]:
    """Lazily import the sync-free exact divergent-index helper.

    ``_exact_divergent_index`` is a design deliverable of AR002/T001 in
    ``actfold.core.split_layer``.  Importing it lazily keeps the
    pre-implementation Red state scoped to the new tests instead of breaking
    collection of the whole module.
    """
    from actfold.core.split_layer import _exact_divergent_index as fn

    return fn


def _padded_divergent_index() -> Callable[[torch.Tensor, int], torch.Tensor]:
    """Lazily import the fixed-capacity padded divergent-index helper."""
    from actfold.core.split_layer import _padded_divergent_index as fn

    return fn


def _ut001a_build_mask(batch: int, seq_len: int, case: str) -> torch.Tensor:
    """Build a deterministic ``[batch, seq_len]`` bool mask for the given case.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        case: One of ``random``, ``all_false``, ``all_true``, ``single_true``
            or ``single_false``.

    Returns:
        Boolean stability mask with the requested fill pattern.
    """
    torch.manual_seed(batch * 100 + seq_len)
    if case == "random":
        return torch.rand(batch, seq_len) > 0.5
    flat = torch.ones(batch * seq_len, dtype=torch.bool)
    if case == "all_false":
        flat[:] = False
    elif case == "single_true":
        flat[:] = False
        flat[flat.numel() // 2] = True
    elif case == "single_false":
        flat[flat.numel() // 2] = False
    return flat.reshape(batch, seq_len)


@pytest.mark.parametrize("batch, seq_len", [(1, 1), (1, 7), (2, 3), (3, 5), (4, 2)])
@pytest.mark.parametrize(
    "case", ["random", "all_false", "all_true", "single_true", "single_false"]
)
def test_ut001a_exact_divergent_index_matches_nonzero_reference(
    batch: int, seq_len: int, case: str, device: str
) -> None:
    """UT-001a: ``_exact_divergent_index`` equals the ``nonzero`` reference.

    Across several mask shapes, the degenerate fills (all-False, all-True,
    single True, single False) and random masks, on CPU and the test device,
    the helper must return the ascending flat divergent indices as an int64
    tensor elementwise-identical to
    ``(~mask).reshape(-1).nonzero(as_tuple=False).squeeze(-1)``.
    """
    exact = _exact_divergent_index()
    mask_cpu = _ut001a_build_mask(batch, seq_len, case)
    devices = ["cpu"] if device == "cpu" else ["cpu", device]
    for dev in devices:
        mask = mask_cpu.to(dev)
        result = exact(mask)
        expected = (~mask).reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        assert result.dtype == torch.int64
        assert result.shape == expected.shape
        assert torch.equal(result, expected)


def test_ut001b_mixed_split_path_eradicates_nonzero_sync(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """UT-001b: mixed split forwards perform no ``nonzero`` host sync.

    The only tolerated per-layer host readback is the three-way stable-count
    check in ``FoldedTransformerLayer.forward`` (``int(stable_mask.sum())``,
    dispatched through ``Tensor.__int__``; a Python-level ``Tensor.item``
    patch cannot see ``int(...)``).  The ``nonzero`` flat-index call in
    ``_recompute_merged`` is the synchronization point this AR eradicates:
    its call count must be zero, where the legacy implementation calls it
    once per split forward.
    """
    hidden_dim = 16
    batch, seq_len = 2, 6
    layer = LlamaLikeLayer(hidden_dim).to(device)
    parent = torch.randn(batch, seq_len, hidden_dim, device=device)
    child = parent.clone()
    child[:, 0, :] = torch.randn(batch, hidden_dim, device=device)

    counts = {"nonzero": 0, "readback": 0}
    orig_nonzero = torch.Tensor.nonzero
    orig_item = torch.Tensor.item

    def counting_nonzero(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        counts["nonzero"] += 1
        return orig_nonzero(self, *args, **kwargs)

    def counting_item(self: torch.Tensor) -> Any:
        counts["readback"] += 1
        return orig_item(self)

    def counting_int(self: torch.Tensor) -> Any:
        counts["readback"] += 1
        return orig_item(self)

    monkeypatch.setattr(torch.Tensor, "nonzero", counting_nonzero)
    monkeypatch.setattr(torch.Tensor, "item", counting_item)
    monkeypatch.setattr(torch.Tensor, "__int__", counting_int)

    cache = ActivationCache(max_entries_per_layer=64, device=device)
    gate = SimilarityGate(tau=0.99)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=4)
    with torch.no_grad():
        split(parent, branch_id="parent")
        out = split(child, branch_id="child", parent_branch_id="parent")

    assert out.shape == child.shape
    # The split actually engaged on the mixed mask: the MLP saw only the two
    # divergent rows (token 0 of each batch element).
    assert len(layer.mlp.seen_shapes[-1]) == 2
    assert layer.mlp.seen_shapes[-1][0] == 2
    # The eradicated sync point: no nonzero call anywhere in the folded forward.
    assert counts["nonzero"] == 0
    # The retained three-way stable-count readback: exactly one per layer.
    assert counts["readback"] == 1


class _NonzeroReferenceSplitLayer(SplitFoldedTransformerLayer):
    """Reference split layer pinned to the legacy ``nonzero`` index path.

    AR002/T001 replaces the ``nonzero`` flat-index computation in
    ``SplitFoldedTransformerLayer._recompute_merged`` with the sync-free
    ``_exact_divergent_index`` helper.  This subclass copies the legacy
    implementation verbatim so the old and new index paths can be compared
    bit-exactly on identical inputs and weights.
    """

    def _recompute_merged(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        stable_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        spec = self.split_spec
        if not self._split_enabled or spec is None:
            return self._recompute_all(hidden_states, attention_mask, **kwargs)
        num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
        if num_tokens < self.min_split_tokens:
            return self._recompute_all(hidden_states, attention_mask, **kwargs)
        divergent = ~stable_mask
        flat_index = divergent.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        self._split_state = {
            "flat_index": flat_index,
            "input_shape": None,
            "num_divergent": 0,
        }
        try:
            return self._recompute_all(hidden_states, attention_mask, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._split_enabled = False
            warnings.warn(
                f"Split FFN disabled for layer {self.layer_idx} "
                f"({type(exc).__name__}: {exc}); falling back to full recompute.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._recompute_all(hidden_states, attention_mask, **kwargs)
        finally:
            self._split_state = None


@pytest.mark.parametrize("divergent_positions", [(0, 1, 2), (1, 4)])
def test_ut001c_split_bit_exact_vs_nonzero_reference(
    divergent_positions: tuple[int, ...],
) -> None:
    """UT-001c: the sync-free index path is bit-exact vs the legacy one.

    ``divergent_positions=(0, 1, 2)`` makes the divergent count D=3 equal a
    representative fixed capacity; ``(1, 4)`` is a general mixed spread.  The
    new path and the legacy ``nonzero`` reference subclass see identical
    inputs and weights; their outputs must be ``torch.equal``.
    """
    hidden_dim = 16
    seq_len = 8
    torch.manual_seed(0)
    parent = torch.randn(1, seq_len, hidden_dim)
    child = parent.clone()
    child[:, list(divergent_positions), :] = torch.randn(
        1, len(divergent_positions), hidden_dim
    )

    # Precondition: the gate marks exactly the constructed positions divergent.
    stable_mask = SimilarityGate(tau=0.99)(child, parent)
    assert int((~stable_mask).sum()) == len(divergent_positions)

    outputs: dict[str, torch.Tensor] = {}
    for name, cls in (
        ("new", SplitFoldedTransformerLayer),
        ("reference", _NonzeroReferenceSplitLayer),
    ):
        torch.manual_seed(7)
        layer = LlamaLikeLayer(hidden_dim)
        cache = ActivationCache(max_entries_per_layer=64, device="cpu")
        gate = SimilarityGate(tau=0.99)
        folded = cls(layer, cache, gate, layer_idx=0, min_split_tokens=0)
        with torch.no_grad():
            folded(parent, branch_id="parent")
            outputs[name] = folded(child, branch_id="child", parent_branch_id="parent")
        # The split engaged in both layers: the MLP saw only the divergent rows.
        assert len(layer.mlp.seen_shapes[-1]) == 2
    assert torch.equal(outputs["new"], outputs["reference"])


@pytest.mark.parametrize(
    "batch, seq_len, num_divergent, capacity",
    [
        (1, 8, 3, 6),  # D < capacity: stable padding head + exact tail
        (2, 5, 4, 4),  # D == capacity: equals the exact index
        (1, 8, 6, 3),  # D > capacity: last `capacity` divergent ranks
        (1, 6, 0, 4),  # D == 0: pure stable padding
        (3, 4, 9, 5),  # D > capacity, batched
        (1, 8, 8, 8),  # D == capacity == N: all divergent
    ],
)
def test_ut001d_padded_divergent_index_contract(
    batch: int, seq_len: int, num_divergent: int, capacity: int, device: str
) -> None:
    """UT-001d: the fixed-capacity padded index obeys the padding contract.

    ``_padded_divergent_index`` always returns an int64 tensor of shape
    ``[capacity]`` with ``capacity`` distinct flat positions.  For
    D < capacity the head entries are stable positions and the tail equals
    the exact divergent index; for D == capacity the result equals the exact
    index; for D > capacity it holds the ascending divergent ranks
    (D - capacity)..(D - 1), i.e. the last ``capacity`` exact indices.
    """
    exact = _exact_divergent_index()
    padded_fn = _padded_divergent_index()

    n = batch * seq_len
    generator = torch.Generator().manual_seed(
        batch * 1000 + seq_len * 10 + num_divergent
    )
    perm = torch.randperm(n, generator=generator)
    flat = torch.ones(n, dtype=torch.bool)
    flat[perm[:num_divergent]] = False

    devices = ["cpu"] if device == "cpu" else ["cpu", device]
    for dev in devices:
        mask = flat.reshape(batch, seq_len).to(dev)
        exact_ref = (~mask).reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        assert exact_ref.numel() == num_divergent
        assert torch.equal(exact(mask), exact_ref)

        padded = padded_fn(mask, capacity)
        assert padded.dtype == torch.int64
        assert padded.shape == (capacity,)
        # The argsort construction yields `capacity` distinct flat positions.
        assert torch.unique(padded).numel() == capacity

        if num_divergent < capacity:
            head = padded[: capacity - num_divergent]
            assert bool(mask.reshape(-1)[head].all())
            if num_divergent > 0:
                assert torch.equal(padded[capacity - num_divergent :], exact_ref)
        elif num_divergent == capacity:
            assert torch.equal(padded, exact_ref)
        else:
            assert torch.equal(padded, exact_ref[num_divergent - capacity :])


def test_ut001e_all_stable_fast_path_skips_divergent_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-001e: the all-stable fast path never computes a divergent index.

    With the child identical to the parent the mask is all-True and the fast
    path returns the cached parent FFN output directly: no FFN recompute, no
    divergent-index gather.
    """
    hidden_dim = 8
    layer = LlamaLikeLayer(hidden_dim)
    parent = torch.randn(1, 4, hidden_dim)

    cache = ActivationCache(max_entries_per_layer=32, device="cpu")
    gate = SimilarityGate(tau=0.99)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=0)
    with torch.no_grad():
        split(parent, branch_id="parent")

    child = parent.clone()
    assert bool(gate(child, parent).all())

    calls = {"n": 0}

    def _forbidden(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls["n"] += 1
        raise AssertionError(
            "_exact_divergent_index must not be called on the all-stable fast path"
        )

    monkeypatch.setattr(split_layer_module, "_exact_divergent_index", _forbidden)
    seen_before = len(layer.mlp.seen_shapes)
    with torch.no_grad():
        out = split(child, branch_id="child", parent_branch_id="parent")

    parent_ffn = cache.fetch(branch_id="parent", layer_idx=0)["ffn_out"]
    assert torch.equal(out, parent_ffn)
    # No FFN compute and no divergent-index gather happened at all.
    assert len(layer.mlp.seen_shapes) == seen_before
    assert calls["n"] == 0
    assert split._split_state is None


def test_ut001f_all_divergent_full_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-001f: an all-divergent mask (stable_count == 0) full-recomputes.

    No token is stable, so the layer takes the full recompute path: the
    output equals a direct call to the original layer and the split state
    never activates.
    """
    hidden_dim = 8
    layer = LlamaLikeLayer(hidden_dim)
    parent = torch.randn(1, 5, hidden_dim)
    child = torch.randn(1, 5, hidden_dim)

    cache = ActivationCache(max_entries_per_layer=32, device="cpu")
    # tau=1.0: the gate's `sim > tau` is never true, so every token diverges.
    gate = SimilarityGate(tau=1.0)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0, min_split_tokens=0)

    assert not bool(gate(child, parent).any())

    calls = {"n": 0}

    def _forbidden(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls["n"] += 1
        raise AssertionError(
            "_exact_divergent_index must not be called when no token is stable"
        )

    monkeypatch.setattr(split_layer_module, "_exact_divergent_index", _forbidden)
    with torch.no_grad():
        split(parent, branch_id="parent")
        out = split(child, branch_id="child", parent_branch_id="parent")

    # Full 3-D recompute through the original layer, no row slicing.
    assert layer.mlp.seen_shapes[-1] == (1, 5, hidden_dim)
    assert torch.equal(out, layer(child))
    assert split._split_state is None
    assert calls["n"] == 0


def test_ut001g_min_split_boundary_and_exception_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-001g: min_split_tokens boundary and the one-shot exception fallback.

    ``N == min_split_tokens - 1`` stays on the full-recompute path (the exact
    divergent index is never computed and the output equals the full
    recompute); ``N == min_split_tokens`` enters the split with exactly one
    index computation.  A single injected failure in the original layer
    degrades with a ``RuntimeWarning`` and a correct fallback output.
    """
    hidden_dim = 8

    def build(
        seq_len: int, min_split_tokens: int
    ) -> tuple[SplitFoldedTransformerLayer, LlamaLikeLayer, torch.Tensor, torch.Tensor]:
        """Build a split layer over a fresh layer with one divergent token."""
        torch.manual_seed(11)
        layer = LlamaLikeLayer(hidden_dim)
        parent = torch.randn(1, seq_len, hidden_dim)
        child = parent.clone()
        child[:, 2, :] = torch.randn(1, hidden_dim)
        cache = ActivationCache(max_entries_per_layer=64, device="cpu")
        gate = SimilarityGate(tau=0.99)
        split = SplitFoldedTransformerLayer(
            layer, cache, gate, layer_idx=0, min_split_tokens=min_split_tokens
        )
        with torch.no_grad():
            split(parent, branch_id="parent")
        return split, layer, parent, child

    real_exact = split_layer_module._exact_divergent_index
    calls = {"n": 0}

    def counting_exact(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls["n"] += 1
        return real_exact(*args, **kwargs)

    monkeypatch.setattr(split_layer_module, "_exact_divergent_index", counting_exact)

    # N == min_split_tokens - 1: no split, no divergent-index computation.
    split, layer, parent, child = build(seq_len=7, min_split_tokens=8)
    with torch.no_grad():
        out = split(child, branch_id="child", parent_branch_id="parent")
    stable_mask = SimilarityGate(tau=0.99)(child, parent)
    expected = merge_stable_divergent(layer(parent), layer(child), stable_mask)
    assert torch.equal(out, expected)
    assert layer.mlp.seen_shapes[-1] == (1, 7, hidden_dim)
    assert calls["n"] == 0

    # N == min_split_tokens: the split engages with exactly one index computation.
    calls["n"] = 0
    split, layer, parent, child = build(seq_len=8, min_split_tokens=8)
    stable_mask = SimilarityGate(tau=0.99)(child, parent)
    expected = merge_stable_divergent(layer(parent), layer(child), stable_mask)
    shapes_before = len(layer.mlp.seen_shapes)
    with torch.no_grad():
        out = split(child, branch_id="child", parent_branch_id="parent")
    assert torch.allclose(out, expected, atol=1e-6)
    # Only the single divergent row went through the MLP (shape [D, hidden]).
    assert len(layer.mlp.seen_shapes) == shapes_before + 1
    assert layer.mlp.seen_shapes[-1] == (1, hidden_dim)
    assert calls["n"] == 1

    # One-shot failure inside the split recompute: RuntimeWarning + fallback.
    calls["n"] = 0
    split, layer, parent, child = build(seq_len=8, min_split_tokens=4)
    stable_mask = SimilarityGate(tau=0.99)(child, parent)
    expected = merge_stable_divergent(layer(parent), layer(child), stable_mask)

    orig_forward = layer.forward
    failures = {"remaining": 1}

    def flaky_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        if failures["remaining"] > 0:
            failures["remaining"] -= 1
            raise RuntimeError("injected one-shot failure")
        return orig_forward(*args, **kwargs)

    monkeypatch.setattr(layer, "forward", flaky_forward)
    with pytest.warns(RuntimeWarning):
        with torch.no_grad():
            out = split(child, branch_id="child", parent_branch_id="parent")
    assert failures["remaining"] == 0
    assert torch.allclose(out, expected, atol=1e-6)
    assert split._split_state is None
    assert split.split_enabled is False
