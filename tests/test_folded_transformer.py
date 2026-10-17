"""Tests for actfold.core.folded_transformer."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, FoldedModel, SimilarityGate
from actfold.core import folded_transformer as folded_transformer_module
from actfold.core import fused_ops as fused_ops_module
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.fused_ops import merge_stable_divergent
from actfold.core.vectorized_cache import VectorizedActivationCache


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
    """FoldedTransformerLayer preserves tuple output arity (e.g. LLaDA blocks)."""
    hidden_dim = 16
    layer = TupleOutputLayer().to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    x = torch.randn(2, 4, hidden_dim, device=device)
    out = folded(x, branch_id="root")
    assert isinstance(out, tuple)
    hidden, cache_out = out
    assert cache_out is None
    assert hidden.shape == x.shape
    assert torch.allclose(hidden, x * 2)


def test_folded_layer_tuple_fast_path_with_parent(device: str) -> None:
    """All-stable fast path keeps the tuple structure for tuple layers."""
    hidden_dim = 16
    layer = TupleOutputLayer().to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    parent = torch.randn(2, 4, hidden_dim, device=device)
    parent_out = folded(parent, branch_id="parent")
    assert isinstance(parent_out, tuple)
    cache.put(
        branch_id="parent",
        layer_idx=0,
        activations={"ffn_out": parent_out[0], "embedding": parent},
    )

    child_out = folded(parent.clone(), branch_id="child", parent_branch_id="parent")
    assert isinstance(child_out, tuple)
    assert child_out[1] is None
    assert torch.allclose(child_out[0], parent_out[0])


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
            "embedding": parent,
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


# ---------------------------------------------------------------------------
# T010: single-sync three-way branch, get_all fast path, ones-mask reuse,
# and reflection-free forward.
# ---------------------------------------------------------------------------

_STABLE_SCENARIOS = ["all_stable", "all_divergent", "mixed"]


class _SyncFreeCache:
    """Duck-typed activation cache whose reads never call ``.all()``/``.any()``.

    Both real caches call ``.any()``/``bool(mask.all())`` inside ``get``
    (``ActivationCache.get`` LRU-touch loop and ``gather_cached_activations``;
    ``VectorizedActivationCache.get`` full-mask fast path), which would trip
    the sync monkeypatch in ``test_three_way_branch_single_sync``. This fake
    mirrors the public ``put``/``get``/``get_all`` interface but returns the
    stored tensors directly (the semantics of the full-mask fast path), so
    only ``FoldedTransformerLayer``'s own device syncs are observed.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        self._store[(branch_id, layer_idx, step_idx)] = dict(activations)

    def get(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        key = (branch_id, layer_idx, step_idx)
        if key not in self._store:
            raise KeyError(f"No cache entry for branch={branch_id}, layer={layer_idx}")
        return dict(self._store[key])

    def get_all(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        key = (branch_id, layer_idx, step_idx)
        if key not in self._store:
            raise KeyError(f"No cache entry for branch={branch_id}, layer={layer_idx}")
        return dict(self._store[key])

    def fetch(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        key = (branch_id, layer_idx, step_idx)
        if key not in self._store:
            raise KeyError(f"No cache entry for branch={branch_id}, layer={layer_idx}")
        return dict(self._store[key])


class _RecordingVectorizedCache:
    """Duck-typed cache recording get/get_all calls around a real vectorized cache."""

    def __init__(self, wrapped: VectorizedActivationCache) -> None:
        self._wrapped = wrapped
        self.get_calls: list[tuple[str, int, int]] = []
        self.get_all_calls: list[tuple[str, int, int]] = []

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        self._wrapped.put(branch_id, layer_idx, activations, step_idx=step_idx)

    def get(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        self.get_calls.append((branch_id, layer_idx, step_idx))
        return self._wrapped.get(branch_id, layer_idx, token_mask, step_idx=step_idx)

    def get_all(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        self.get_all_calls.append((branch_id, layer_idx, step_idx))
        return self._wrapped.get_all(branch_id, layer_idx, step_idx=step_idx)


def _make_child(parent: torch.Tensor, scenario: str) -> torch.Tensor:
    """Build a child input for the requested stability scenario.

    ``all_stable``: identical to the parent (cosine 1.0 > tau).
    ``all_divergent``: parent plus a large perturbation (cosine ~ 0).
    ``mixed``: first half of the sequence perturbed, second half identical.
    """
    if scenario == "all_stable":
        return parent.clone()
    if scenario == "all_divergent":
        return parent + 100.0 * torch.randn_like(parent)
    if scenario == "mixed":
        child = parent.clone()
        half = child.shape[1] // 2
        child[:, :half] = child[:, :half] + 100.0 * torch.randn(
            child.shape[0], half, child.shape[2], device=child.device, dtype=child.dtype
        )
        return child
    raise ValueError(f"Unknown scenario: {scenario}")


def _expected_child_output(
    layer: DummyLayer,
    gate: SimilarityGate,
    parent: torch.Tensor,
    parent_out: torch.Tensor,
    child: torch.Tensor,
    scenario: str,
) -> torch.Tensor:
    """Oracle for the folded child output under each stability scenario.

    all_stable -> cached parent FFN output; all_divergent -> original_layer on
    the child; mixed -> merge_stable_divergent of the parent FFN output and the
    recomputed child output under the gate's stability mask.
    """
    if scenario == "all_stable":
        return parent_out
    if scenario == "all_divergent":
        return layer(child)
    stable_mask = gate(child, parent)
    return merge_stable_divergent(parent_out, layer(child), stable_mask)


@pytest.mark.parametrize("scenario", _STABLE_SCENARIOS)
def test_three_way_branch_single_sync(device: str, scenario: str) -> None:
    """(a) The three-way stable branch must not call .all()/.any() in forward.

    ``torch.Tensor.all`` and ``torch.Tensor.any`` are monkeypatched to raise
    for the duration of the child forward; the branch decision must be made
    with a single ``int(stable_mask.sum())`` sync instead. Behavior must stay
    identical in all three scenarios (output matches the per-scenario oracle).
    """
    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = _SyncFreeCache()
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    parent = torch.randn(1, 4, hidden_dim, device=device)
    parent_out = folded(parent, branch_id="parent")
    child = _make_child(parent, scenario)
    expected = _expected_child_output(layer, gate, parent, parent_out, child, scenario)

    def _raise_sync(*args: object, **kwargs: object) -> None:
        raise AssertionError("sync")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.Tensor, "all", _raise_sync)
        mp.setattr(torch.Tensor, "any", _raise_sync)
        out = folded(child, branch_id="child", parent_branch_id="parent")

    assert torch.allclose(out, expected, atol=1e-5)


def test_parent_fetch_uses_get_all_when_available(device: str) -> None:
    """(b) The parent hidden-states fetch goes through cache.get_all when present.

    Uses a fully divergent child: the parent hidden-states fetch is the only
    cache read on that path (the none-stable branch never fetches parent FFN),
    so ``get`` must not be called at all when ``get_all`` is available.
    """
    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = _RecordingVectorizedCache(VectorizedActivationCache(device=device))
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    parent = torch.randn(1, 4, hidden_dim, device=device)
    folded(parent, branch_id="parent")

    child = _make_child(parent, "all_divergent")
    out = folded(child, branch_id="child", parent_branch_id="parent")

    assert any(
        branch == "parent" and layer_idx == 0 for branch, layer_idx, _ in cache.get_all_calls
    ), "parent hidden-states fetch must use get_all when the cache provides it"
    assert cache.get_calls == [], "parent hidden-states fetch must not fall back to get()"
    assert torch.allclose(out, layer(child), atol=1e-5)


def test_fallback_ones_mask_cached_by_shape(device: str) -> None:
    """(c) Legacy-cache gate path never allocates an all-ones mask (T015).

    Since F7 the folded layer fetches the parent activation via
    ``get_all``/``fetch`` instead of a masked ``get``; the legacy
    ``ActivationCache`` implements both.  Consecutive child forwards must
    therefore allocate zero ``torch.ones`` masks and produce outputs equal to
    the cached parent FFN output.
    """
    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    assert hasattr(cache, "get_all")
    assert hasattr(cache, "fetch")
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    parent = torch.randn(1, 4, hidden_dim, device=device)
    parent_out = folded(parent, branch_id="parent")
    child = _make_child(parent, "all_stable")

    def _raise_ones(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("torch.ones allocated in the folded forward")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch, "ones", _raise_ones)
        out1 = folded(child, branch_id="child", parent_branch_id="parent")
        out2 = folded(child, branch_id="child", parent_branch_id="parent")

    assert torch.allclose(out1, parent_out, atol=1e-5)
    assert torch.allclose(out1, out2)


def test_no_reflection_in_forward(device: str) -> None:
    """(d) FoldedTransformerLayer.forward must not call inspect.signature.

    The accepted-params set of ``original_layer.forward`` must be precomputed
    at construction time; a mixed-scenario child forward (which recomputes via
    ``_recompute_all``) completes with ``inspect.signature`` patched to raise.
    """
    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    parent = torch.randn(1, 4, hidden_dim, device=device)
    parent_out = folded(parent, branch_id="parent")
    child = _make_child(parent, "mixed")

    def _raise_reflection(*args: object, **kwargs: object) -> None:
        raise AssertionError("reflection in forward")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(inspect, "signature", _raise_reflection)
        out = folded(child, branch_id="child", parent_branch_id="parent")

    expected = _expected_child_output(layer, gate, parent, parent_out, child, "mixed")
    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.parametrize("scenario", _STABLE_SCENARIOS)
def test_three_way_outputs_unchanged(device: str, scenario: str) -> None:
    """Regression guard: the three-way branch keeps original_layer semantics.

    all_divergent -> ``original_layer(child)``; all_stable -> cached parent FFN
    output; mixed -> ``merge_stable_divergent(parent_ffn, original_layer(child),
    stable_mask)``. These oracles must hold before and after the T010 changes.
    """
    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    parent = torch.randn(1, 4, hidden_dim, device=device)
    parent_out = folded(parent, branch_id="parent")
    child = _make_child(parent, scenario)
    out = folded(child, branch_id="child", parent_branch_id="parent")

    expected = _expected_child_output(layer, gate, parent, parent_out, child, scenario)
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# T014: only ffn_out (plus a layer-0 embedding) is cached; the gate reads the
# parent's previous-layer ffn_out, or the layer-0 embedding at layer 0.
# ---------------------------------------------------------------------------


class _TinyStackedModel(nn.Module):
    """Minimal token-wise model with a discoverable ``layers`` stack."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(DummyLayer(hidden_dim) for _ in range(num_layers))
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
        logits: torch.Tensor = self.head(x)
        return logits


def test_t014_store_schema(device: str) -> None:
    """T014: a parent pass stores only ffn_out plus a layer-0 embedding.

    Layer 0 must cache ``{"ffn_out", "embedding"}`` and every deeper layer only
    ``{"ffn_out"}``. The name ``"hidden_states"`` must never appear in a cache
    entry written by a folded layer.
    """
    vocab_size = 32
    hidden_dim = 16
    num_layers = 2
    model = _TinyStackedModel(vocab_size, hidden_dim, num_layers).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedModel(model, cache, gate)

    tokens = torch.randint(0, vocab_size, (1, 4), device=device)
    folded(tokens, branch_id="parent")

    full_mask = torch.ones(1, 4, dtype=torch.bool, device=device)
    layer0 = cache.get("parent", layer_idx=0, token_mask=full_mask)
    assert set(layer0.keys()) == {"ffn_out", "embedding"}
    assert "hidden_states" not in layer0

    for layer_idx in range(1, num_layers):
        entry = cache.get("parent", layer_idx=layer_idx, token_mask=full_mask)
        assert set(entry.keys()) == {"ffn_out"}
        assert "hidden_states" not in entry


def test_t014_gate_reads_previous_layer_ffn(device: str) -> None:
    """T014: the gate of layer L reads the parent's ffn_out from layer L-1.

    The parent cache holds ``{"ffn_out": P0}`` at layer 0 and ``{"ffn_out":
    P1}`` at layer 1 (the new scheme). The child forward at layer_idx=1 must
    equal the manual reference ``where(gate(child, P0), P1, layer(child))``:
    the previous layer's output is numerically the parent's input to layer 1.
    """
    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=1).to(device)

    p0 = torch.randn(1, 6, hidden_dim, device=device)
    p1 = torch.randn(1, 6, hidden_dim, device=device)
    cache.put(branch_id="parent", layer_idx=0, activations={"ffn_out": p0})
    cache.put(branch_id="parent", layer_idx=1, activations={"ffn_out": p1})

    # Mixed stability: first half identical to P0 (stable), second half far away.
    child = p0.clone()
    child[:, 3:, :] += 100.0 * torch.randn(1, 3, hidden_dim, device=device)

    out = folded(child, branch_id="child", parent_branch_id="parent")

    stable_mask = folded.gate(child, p0)
    expected = merge_stable_divergent(p1, layer(child), stable_mask)
    assert torch.allclose(out, expected, atol=1e-5)


def test_t014_gate_layer0_reads_embedding(device: str) -> None:
    """T014: the gate of layer 0 reads the parent's layer-0 "embedding".

    The parent cache holds ``{"embedding": E, "ffn_out": P0}`` at layer 0. The
    child forward at layer_idx=0 must equal
    ``where(gate(child, E), P0, layer(child))``.
    """
    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    emb = torch.randn(1, 6, hidden_dim, device=device)
    p0 = torch.randn(1, 6, hidden_dim, device=device)
    cache.put(
        branch_id="parent",
        layer_idx=0,
        activations={"embedding": emb, "ffn_out": p0},
    )

    # Mixed stability: first half identical to the embedding, second half far
    # away.
    child = emb.clone()
    child[:, 3:, :] += 100.0 * torch.randn(1, 3, hidden_dim, device=device)

    out = folded(child, branch_id="child", parent_branch_id="parent")

    stable_mask = folded.gate(child, emb)
    expected = merge_stable_divergent(p0, layer(child), stable_mask)
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# T015 (F7): the folded forward migrates to the unified cache protocol
# (fetch / fetch_masked / get_all / contains); the ones-mask machinery and
# all legacy cache.get() calls inside forward are removed.
# ---------------------------------------------------------------------------


def test_t015_merge_path_no_zero_fill(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """T015: the slow path reads the parent FFN output without legacy ``get``.

    Mixed-stability child forward with the parent cache pre-populated under
    the T014 schema. The cache instance's legacy ``get`` is monkeypatched to
    raise, so the forward must complete through the new fetch/get_all paths
    (whose mask-free reads eliminate ``get``'s zero-fill-for-divergent-position
    allocation; the merge overwrites those positions anyway) and produce the
    exact merged output.
    """
    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=1).to(device)

    p0 = torch.randn(1, 6, hidden_dim, device=device)
    p1 = torch.randn(1, 6, hidden_dim, device=device)
    cache.put(branch_id="parent", layer_idx=0, activations={"ffn_out": p0})
    cache.put(branch_id="parent", layer_idx=1, activations={"ffn_out": p1})

    # Mixed stability: first half identical to the parent layer-0 output,
    # second half far away.
    child = p0.clone()
    child[:, 3:, :] += 100.0 * torch.randn(1, 3, hidden_dim, device=device)

    def _raise_get(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy cache.get used in the folded forward")

    monkeypatch.setattr(cache, "get", _raise_get)

    out = folded(child, branch_id="child", parent_branch_id="parent")

    stable_mask = gate(child, p0)
    expected = merge_stable_divergent(p1, layer(child), stable_mask)
    assert torch.allclose(out, expected, atol=1e-5)


def test_t015_no_ones_mask_allocation(device: str) -> None:
    """T015: the gate path never allocates an all-ones mask.

    ``torch.ones`` is monkeypatched to raise during a parent+child folded
    forward through a legacy cache: the gate fetch must use the mask-free
    paths (get_all/fetch) instead of ``get`` with a fabricated full mask.
    The layer must also no longer carry the ``_full_masks`` attribute (the
    ones-mask machinery is deleted).
    """
    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    parent = torch.randn(1, 4, hidden_dim, device=device)
    parent_out = folded(parent, branch_id="parent")
    child = _make_child(parent, "all_stable")

    def _raise_ones(*args: object, **kwargs: object) -> None:
        raise AssertionError("torch.ones allocated in the folded forward")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch, "ones", _raise_ones)
        out = folded(child, branch_id="child", parent_branch_id="parent")

    assert torch.allclose(out, parent_out, atol=1e-5)
    assert not hasattr(folded, "_full_masks"), "the ones-mask machinery must be deleted"


def test_t015_all_stable_fast_path_uses_fetch(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """T015: the all-stable fast path reads the parent FFN via the new API.

    With every token stable the child output must equal the cached parent
    ``ffn_out`` while the legacy ``get`` is monkeypatched to raise, proving
    the fast path goes through the mask-free fetch/get_all paths.
    """
    hidden_dim = 16
    layer = DummyLayer(hidden_dim).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    parent = torch.randn(1, 4, hidden_dim, device=device)
    parent_out = folded(parent, branch_id="parent")
    child = _make_child(parent, "all_stable")

    def _raise_get(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy cache.get used in the all-stable fast path")

    monkeypatch.setattr(cache, "get", _raise_get)

    out = folded(child, branch_id="child", parent_branch_id="parent")
    assert torch.allclose(out, parent_out, atol=1e-5)


# ---------------------------------------------------------------------------
# T016 (F8 / P1-3): the mixed-stability slow path prefers the fused
# gather_select kernel above the D3 thresholds (T >= 2048, H >= 4096, CUDA),
# and stays bit-exact with the original fetch + merge path below them.
# ---------------------------------------------------------------------------


def _t016_parent_cache_with(p0: torch.Tensor, p1: torch.Tensor) -> VectorizedActivationCache:
    """Vectorized cache pre-populated per the T014 schema for a layer_idx=1 child.

    Layer 0 holds ``{"ffn_out": p0}`` (the parent's input to layer 1) and
    layer 1 holds ``{"ffn_out": p1}``.
    """
    cache = VectorizedActivationCache(max_entries_per_layer=p0.shape[1], max_branch_steps=0)
    cache.put("parent", 0, {"ffn_out": p0})
    cache.put("parent", 1, {"ffn_out": p1})
    return cache


def _t016_mixed_child(p0: torch.Tensor) -> torch.Tensor:
    """Child input that is stable on the first half, divergent on the second."""
    child = p0.clone()
    half = child.shape[1] // 2
    child[:, half:] += 100.0 * torch.randn(child.shape[0], child.shape[1] - half, child.shape[2])
    return child


def test_t016_thresholds_default_values() -> None:
    """T016: the fused-path gating constants match design decision D3.

    ``_GATHER_SELECT_MIN_TOKENS``/``_GATHER_SELECT_MIN_HIDDEN`` gate the fused
    path at T >= 2048 and H >= 4096, and ``_GATHER_SELECT_REQUIRE_CUDA``
    restricts it to CUDA tensors.
    """
    assert folded_transformer_module._GATHER_SELECT_MIN_TOKENS == 2048
    assert folded_transformer_module._GATHER_SELECT_MIN_HIDDEN == 4096
    assert folded_transformer_module._GATHER_SELECT_REQUIRE_CUDA is True


def test_t016_slow_path_fused_bit_exact_cpu() -> None:
    """T016: the fused slow path is bit-exact with the original slow path.

    Two identical layer_idx=1 folded layers with separately (identically)
    populated parent caches run the same mixed-stability child: once with the
    default module constants (original ``fetch`` + ``merge_stable_divergent``
    path) and once with the thresholds lowered and the CUDA requirement
    disabled (fused ``gather_select`` path on CPU). The outputs must be equal.
    """
    hidden_dim = 16
    batch, seq = 2, 6
    gate = SimilarityGate(tau=0.95)
    layer = DummyLayer(hidden_dim)
    torch.manual_seed(161)
    p0 = torch.randn(batch, seq, hidden_dim)
    p1 = torch.randn(batch, seq, hidden_dim)
    child = _t016_mixed_child(p0)

    # Sanity: the scenario really is mixed-stability (slow path exercised).
    stable_mask = gate(child, p0)
    assert 0 < int(stable_mask.sum()) < stable_mask.numel()

    # Original path: default module constants (T=6 < 2048 -> below threshold).
    folded_original = FoldedTransformerLayer(
        layer, _t016_parent_cache_with(p0, p1), gate, layer_idx=1
    )
    out_original = folded_original(child, branch_id="child", parent_branch_id="parent")

    # Fused path: thresholds lowered + CUDA requirement disabled.
    folded_fused = FoldedTransformerLayer(layer, _t016_parent_cache_with(p0, p1), gate, layer_idx=1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(folded_transformer_module, "_GATHER_SELECT_MIN_TOKENS", 0)
        mp.setattr(folded_transformer_module, "_GATHER_SELECT_MIN_HIDDEN", 0)
        mp.setattr(folded_transformer_module, "_GATHER_SELECT_REQUIRE_CUDA", False)
        out_fused = folded_fused(child, branch_id="child", parent_branch_id="parent")

    assert torch.equal(out_original, out_fused)


def test_t016_slow_path_fused_uses_gather_select() -> None:
    """T016: the slow path calls gather_select only above the thresholds.

    With the thresholds monkeypatched to enable the fused path, a recording
    wrapper around ``fused_ops.gather_select`` must observe exactly one call
    during a mixed-stability forward. With the default thresholds restored,
    a second forward must not add further calls (original path).
    """
    hidden_dim = 16
    batch, seq = 2, 6
    gate = SimilarityGate(tau=0.95)
    layer = DummyLayer(hidden_dim)
    torch.manual_seed(162)
    p0 = torch.randn(batch, seq, hidden_dim)
    p1 = torch.randn(batch, seq, hidden_dim)
    folded = FoldedTransformerLayer(layer, _t016_parent_cache_with(p0, p1), gate, layer_idx=1)
    child = _t016_mixed_child(p0)

    real_gather_select = fused_ops_module.gather_select
    calls: list[tuple[object, ...]] = []

    def _recording_gather_select(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(args)
        return real_gather_select(*args, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(folded_transformer_module, "_GATHER_SELECT_MIN_TOKENS", 0)
        mp.setattr(folded_transformer_module, "_GATHER_SELECT_MIN_HIDDEN", 0)
        mp.setattr(folded_transformer_module, "_GATHER_SELECT_REQUIRE_CUDA", False)
        mp.setattr(fused_ops_module, "gather_select", _recording_gather_select)
        mp.setattr(
            folded_transformer_module,
            "gather_select",
            _recording_gather_select,
            raising=False,
        )
        out_fused = folded(child, branch_id="child", parent_branch_id="parent")
        assert len(calls) == 1

    # Default thresholds: T=6 < 2048 -> original path, no further calls.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fused_ops_module, "gather_select", _recording_gather_select)
        mp.setattr(
            folded_transformer_module,
            "gather_select",
            _recording_gather_select,
            raising=False,
        )
        out_original = folded(child, branch_id="child2", parent_branch_id="parent")

    assert len(calls) == 1
    assert torch.equal(out_original, out_fused)
