"""Tests for AR011: gather_select on the parent-longer (truncated) path.

UT-1001 (srs 3.1) pins the fused single-pass dispatch for a parent longer
than its child (AR007 truncated-prefix reuse); UT-1002/1003 (srs 3.2) are
behaviour nails for the safety nets: ring-buffer eviction (``fetch_flat``
returns None when ``start > 0``) and the varlen prefix path.

Shared fixtures follow the T016 precedents in ``tests/test_folded_transformer.py``:
the T014 cache schema (layer 0 holds ``{"ffn_out": p0}`` as the gate input,
layer 1 holds ``{"ffn_out": p1}`` as the parent FFN), the three-constant
monkeypatch that lowers the D3 thresholds for CPU runs, and the recording
wrapper that patches ``gather_select`` in both ``fused_ops`` and
``folded_transformer``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from actfold.core import SimilarityGate
from actfold.core import folded_transformer as folded_transformer_module
from actfold.core import fused_ops as fused_ops_module
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.vectorized_cache import VectorizedActivationCache
from test_folded_transformer import DummyLayer, _t016_parent_cache_with

_PERTURB_POSITIONS = [1, 3]


def _perturbed_child(base: torch.Tensor) -> torch.Tensor:
    """Child cloned from ``base`` with fixed-position ``+100*randn`` perturbations.

    Generalizes the ``_t016_mixed_child`` precedent: the large perturbation
    guarantees the tau=0.95 cosine gate marks the perturbed tokens divergent
    (a small drift could stay under the threshold, turn the scenario
    all-stable, and route the forward away from the merge path entirely).
    """
    child = base.clone()
    for pos in _PERTURB_POSITIONS:
        child[:, pos] += 100.0 * torch.randn(child.shape[0], child.shape[2])
    return child


def _ring_parent_cache_with(
    p0: torch.Tensor, p1: torch.Tensor, capacity: int
) -> VectorizedActivationCache:
    """Vectorized cache whose T014-schema parent entries wrap the ring buffer.

    Local generalization of the imported T016 helper
    ``_t016_parent_cache_with``: its hardcoded ``capacity = seq_len`` cannot
    trigger eviction, so UT-1002/1003 build independent caches with an
    explicit ``max_entries_per_layer`` (srs 3.2, req-gate Minor-3).
    """
    cache = VectorizedActivationCache(
        max_entries_per_layer=capacity, max_branch_steps=0
    )
    cache.put("parent", 0, {"ffn_out": p0})
    cache.put("parent", 1, {"ffn_out": p1})
    return cache


def _patch_fused_path(mp: pytest.MonkeyPatch, calls: list[tuple[Any, ...]]) -> None:
    """Lower the D3 thresholds to 0/CPU and record every gather_select call.

    Mirrors the T016 three-constant monkeypatch plus the recording wrapper
    that patches both ``fused_ops.gather_select`` and the module-level
    reference inside ``folded_transformer`` (T016 :764-809 precedent).
    """
    mp.setattr(folded_transformer_module, "_GATHER_SELECT_MIN_TOKENS", 0)
    mp.setattr(folded_transformer_module, "_GATHER_SELECT_MIN_HIDDEN", 0)
    mp.setattr(folded_transformer_module, "_GATHER_SELECT_REQUIRE_CUDA", False)
    real_gather_select = fused_ops_module.gather_select

    def _recording_gather_select(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(args)
        return real_gather_select(*args, **kwargs)

    mp.setattr(fused_ops_module, "gather_select", _recording_gather_select)
    mp.setattr(
        folded_transformer_module, "gather_select", _recording_gather_select,
        raising=False,
    )


def _assert_mixed_positions(
    gate: SimilarityGate, child: torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    """Pre-forward sanity: the scenario is mixed-stable at the pinned positions.

    Guards against a flaky construction (T016 :749-751 precedent): the
    unperturbed columns must be stable and the perturbed ones divergent,
    otherwise the forward never reaches the merge under test.
    """
    stable_mask = gate(child, base)
    assert 0 < int(stable_mask.sum()) < stable_mask.numel()
    keep = [p for p in range(child.shape[1]) if p not in _PERTURB_POSITIONS]
    assert bool(stable_mask[:, keep].all())
    assert not bool(stable_mask[:, _PERTURB_POSITIONS].any())
    return stable_mask


def _assert_merge_composition(
    out: torch.Tensor,
    stable_mask: torch.Tensor,
    parent_ffn: torch.Tensor,
    child_full: torch.Tensor,
) -> None:
    """Stable positions reuse the parent FFN; divergent ones the child values."""
    assert torch.equal(out[stable_mask], parent_ffn[stable_mask])
    assert torch.equal(out[~stable_mask], child_full[~stable_mask])


def test_ut1001_truncated_path_uses_gather_select_bit_exact() -> None:
    """UT-1001 (srs 3.1): the parent-longer merge dispatches to gather_select once.

    A T_p=8 parent in a non-wrapping vectorized cache (capacity=T_p) with a
    T_c=4 mixed-stable child must take the fused single-pass path when the
    D3 thresholds are lowered: exactly one ``gather_select`` call, output
    bit-identical to the default-threshold fetch + truncate + merge run, and
    the merged composition must reuse the parent FFN on stable positions
    while carrying the recomputed child values on perturbed ones.
    """
    hidden_dim = 16
    batch, t_parent, t_child = 2, 8, 4
    gate = SimilarityGate(tau=0.95)
    layer = DummyLayer(hidden_dim)
    torch.manual_seed(1001)
    p0 = torch.randn(batch, t_parent, hidden_dim)
    p1 = torch.randn(batch, t_parent, hidden_dim)
    child = _perturbed_child(p0[:, :t_child])
    stable_mask = _assert_mixed_positions(gate, child, p0[:, :t_child])

    # Default thresholds on CPU: the original fetch + truncate + merge path.
    folded_default = FoldedTransformerLayer(
        layer, _t016_parent_cache_with(p0, p1), gate, layer_idx=1
    )
    out_default = folded_default(child, branch_id="child", parent_branch_id="parent")

    calls: list[tuple[Any, ...]] = []
    folded_fused = FoldedTransformerLayer(
        layer, _t016_parent_cache_with(p0, p1), gate, layer_idx=1
    )
    with pytest.MonkeyPatch.context() as mp:
        _patch_fused_path(mp, calls)
        out_fused = folded_fused(child, branch_id="child", parent_branch_id="parent")
        assert len(calls) == 1

    assert torch.equal(out_default, out_fused)
    _assert_merge_composition(
        out_default, stable_mask, p1[:, :t_child], layer(child)
    )


def test_ut1002_ring_wrapped_parent_falls_back_without_gather_select() -> None:
    """UT-1002 (srs 3.2): ring eviction keeps the truncated path off the fused kernel.

    An independently built cache (capacity 6, parent T_p=8 -> ring wrap with
    ``start=2``) serves only the surviving token window, so the child is
    built from the surviving-window prefix to keep the scenario on the
    parent-reuse merge (a child built from the original 0..3 prefix would be
    all-divergent and silently take the full-recompute branch).
    ``fetch_flat`` must refuse the wrapped ring: ``gather_select`` is never
    called even with the thresholds lowered, the composition proves the
    merge reused the surviving parent FFN, and the output matches the
    default-threshold run bit for bit.
    """
    hidden_dim = 16
    batch, t_parent, t_child = 2, 8, 4
    gate = SimilarityGate(tau=0.95)
    layer = DummyLayer(hidden_dim)
    torch.manual_seed(1002)
    p0 = torch.randn(batch, t_parent, hidden_dim)
    p1 = torch.randn(batch, t_parent, hidden_dim)
    cache = _ring_parent_cache_with(p0, p1, capacity=6)
    base = cache.fetch("parent", 0)["ffn_out"][:, :t_child]
    assert base.shape == (batch, t_child, hidden_dim)
    child = _perturbed_child(base)
    stable_mask = _assert_mixed_positions(gate, child, base)

    calls: list[tuple[Any, ...]] = []
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=1)
    with pytest.MonkeyPatch.context() as mp:
        _patch_fused_path(mp, calls)
        out = folded(child, branch_id="child", parent_branch_id="parent")
        assert len(calls) == 0

    surviving_ffn = cache.fetch("parent", 1)["ffn_out"][:, :t_child]
    _assert_merge_composition(out, stable_mask, surviving_ffn, layer(child))

    folded_default = FoldedTransformerLayer(
        layer, _ring_parent_cache_with(p0, p1, capacity=6), gate, layer_idx=1
    )
    out_default = folded_default(child, branch_id="child", parent_branch_id="parent")
    assert torch.equal(out_default, out)


def test_ut1003_varlen_shorter_parent_never_calls_gather_select() -> None:
    """UT-1003 (srs 3.2): the varlen prefix path skips the fused kernel.

    A parent shorter than the child (T_p=3 < T_c=5) takes the AR003
    variable-length prefix path; with the thresholds lowered the fused
    dispatch must still be skipped (behaviour nail: the explicit
    ``prefix_len is None`` guard and ``fetch_flat``'s ``total < seq_len``
    rejection are indistinguishable here), and the output keeps the
    prefix-reuse semantics — the all-stable prefix reuses the parent FFN
    and the appended suffix carries the recomputed child values.
    """
    hidden_dim = 16
    batch, t_parent, t_child = 2, 3, 5
    gate = SimilarityGate(tau=0.95)
    layer = DummyLayer(hidden_dim)
    torch.manual_seed(1003)
    p0 = torch.randn(batch, t_parent, hidden_dim)
    p1 = torch.randn(batch, t_parent, hidden_dim)
    cache = _ring_parent_cache_with(p0, p1, capacity=t_child)
    child = torch.cat(
        [p0, torch.randn(batch, t_child - t_parent, hidden_dim)], dim=1
    )
    assert bool(gate(child[:, :t_parent], p0).all())

    calls: list[tuple[Any, ...]] = []
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=1)
    with pytest.MonkeyPatch.context() as mp:
        _patch_fused_path(mp, calls)
        out = folded(child, branch_id="child", parent_branch_id="parent")
        assert len(calls) == 0

    child_full = layer(child)
    assert torch.equal(out[:, :t_parent], p1)
    assert torch.equal(out[:, t_parent:], child_full[:, t_parent:])
