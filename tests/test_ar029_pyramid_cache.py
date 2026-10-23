"""AR029 cache-unit tests (UT-2911..2938, 2945..2958, 2968..2972)."""

from __future__ import annotations

import math

import pytest
import torch

from actfold.core.activation_cache import ActivationCache
from actfold.core.cache_protocol import CachePutEntry, MissingSentinelCapable
from actfold.core.pyramid_cache import PyramidActivationCache
from actfold.core.pyramid_plan import build_pyramid_plan


def _tokens(seq_len: int, batch: int = 1, hidden: int = 8, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq_len, hidden, generator=gen)


def _acts(seq_len: int, batch: int = 1, hidden: int = 8, seed: int = 0) -> dict:
    return {
        "embedding": _tokens(seq_len, batch, hidden, seed),
        "ffn_out": _tokens(seq_len, batch, hidden, seed + 1),
    }


def _plan(depth: int, per_layer: int, block: int = 16, **kwargs) -> object:
    return build_pyramid_plan(
        depth=depth,
        total_tokens=depth * per_layer,
        block_size=block,
        max_ratio=1.0,
        min_ratio=1.0,
        **kwargs,
    )


def test_ut2911_blocks_and_partial_tail() -> None:
    """UT-2911: blocks split correctly and a partial tail round-trips."""
    cache = PyramidActivationCache(_plan(depth=2, per_layer=64))
    acts = _acts(40)
    cache.put("b", 0, acts)
    got = cache.fetch("b", 0)
    assert got["ffn_out"].shape == (1, 40, 8)
    assert torch.equal(got["ffn_out"], acts["ffn_out"])
    assert cache.stats()["blocks"][0] == 3
    assert cache.num_entries(0) == 40


def test_ut2912_per_layer_hard_bound() -> None:
    """UT-2912: each layer never exceeds its (pyramid) token budget."""
    plan = build_pyramid_plan(
        depth=3, total_tokens=3 * 64, block_size=16, max_ratio=1.0, min_ratio=0.25
    )
    cache = PyramidActivationCache(plan)
    for step in range(4):
        cache.put("b", 0, _acts(48, seed=step))
        cache.put("b", 1, _acts(48, seed=step + 10))
        cache.put("b", 2, _acts(48, seed=step + 20))
    for layer in range(3):
        assert cache.stats()["tokens"][layer] <= plan.budget_tokens(layer)


def test_ut2913_plan_wide_bound() -> None:
    """UT-2913: the sum over layers never exceeds the plan bound."""
    plan = _plan(depth=4, per_layer=32)
    cache = PyramidActivationCache(plan)
    for step in range(6):
        for layer in range(4):
            cache.put("b", layer, _acts(64, seed=step + layer))
    assert sum(cache.stats()["tokens"].values()) <= plan.total_budget_tokens


def test_ut2914_anchor_blocks_retained() -> None:
    """UT-2914: anchor blocks are protected ahead of unprotected old blocks."""
    plan = _plan(depth=1, per_layer=80, anchor_blocks=1, anchor_stride=2, recent_blocks=0)
    cache = PyramidActivationCache(plan)
    cache.put("old", 0, _acts(64, seed=1))
    cache.put("new", 0, _acts(64, seed=2))
    retained = plan.retained_blocks(0, 64)
    avail_old = cache.availability("old", 0, 64)
    for block in retained:
        assert avail_old[block * 16 : min((block + 1) * 16, 64)].all()


def test_ut2915_recent_window_only_for_newest_entry() -> None:
    """UT-2915: recent protection applies to the just-written entry only."""
    plan = _plan(depth=1, per_layer=48, anchor_blocks=0, anchor_stride=100, recent_blocks=2)
    cache = PyramidActivationCache(plan)
    cache.put("old", 0, _acts(64, seed=1))
    cache.put("new", 0, _acts(64, seed=2))
    assert cache.availability("new", 0, 64)[32:].all()
    assert not cache.availability("old", 0, 64)[32:].all()


def test_ut2916_evictions_increase() -> None:
    """UT-2916: over-budget writes increment the eviction ledger."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=32, recent_blocks=0))
    for step in range(5):
        cache.put(f"b{step}", 0, _acts(32, seed=step))
    stats = cache.stats()
    assert stats["evictions"] > 0
    assert stats["tokens"][0] <= 32


def test_ut2917_hard_bound_wins_over_protection() -> None:
    """UT-2917: even protected blocks are reclaimed when the bound requires it."""
    plan = _plan(depth=1, per_layer=32, anchor_blocks=10, anchor_stride=100, recent_blocks=10)
    cache = PyramidActivationCache(plan)
    cache.put("b", 0, _acts(128, seed=1))
    assert cache.stats()["tokens"][0] <= 32


def test_ut2918_bytes_exact() -> None:
    """UT-2918: bytes_used equals the retained block tensor bytes."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=64))
    cache.put("b", 0, _acts(40))
    total = sum(t.numel() * t.element_size() for t in cache.iter_block_tensors())
    assert cache.bytes_used == total == cache.stats()["bytes"]


def test_ut2919_determinism() -> None:
    """UT-2919: identical write sequences yield identical stats."""

    def run():
        cache = PyramidActivationCache(_plan(depth=2, per_layer=48, recent_blocks=1))
        for step in range(3):
            cache.put("b", 0, _acts(64, seed=step))
            cache.put("b", 1, _acts(64, seed=step + 5))
        return cache.stats()

    assert run() == run()


@pytest.mark.parametrize(
    "activations",
    [
        {},
        {"ffn_out": torch.randn(4)},
        {"a": torch.randn(1, 4, 8), "b": torch.randn(1, 5, 8)},
        {"ffn_out": torch.randn(1, 0, 8)},
    ],
)
def test_ut2920_invalid_put(activations: dict) -> None:
    """UT-2920: empty/low-rank/shape-mismatched writes are rejected."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=32))
    with pytest.raises(ValueError):
        cache.put("b", 0, activations)


def test_ut2921_missing_entry_keyerror() -> None:
    """UT-2921: querying a never-written entry raises KeyError."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=32))
    with pytest.raises(KeyError):
        cache.fetch("nope", 0)
    assert cache.contains("nope", 0) is False


def test_ut2922_put_batch_equivalent() -> None:
    """UT-2922: put_batch is state-equivalent to sequential put."""
    entries = [
        CachePutEntry("a", 0, _acts(32, seed=1)),
        CachePutEntry("b", 0, _acts(32, seed=2), 1),
        CachePutEntry("a", 1, _acts(32, seed=3)),
    ]
    cache_a = PyramidActivationCache(_plan(depth=2, per_layer=64))
    cache_b = PyramidActivationCache(_plan(depth=2, per_layer=64))
    cache_a.put_batch(entries)
    for entry in entries:
        cache_b.put(entry.branch_id, entry.layer_idx, entry.activations, entry.step_idx)
    assert cache_a.stats() == cache_b.stats()
    assert cache_a.write_op_count == cache_b.write_op_count
    for entry in entries:
        got_a = cache_a.fetch(entry.branch_id, entry.layer_idx, entry.step_idx)
        got_b = cache_b.fetch(entry.branch_id, entry.layer_idx, entry.step_idx)
        assert all(torch.equal(got_a[k], got_b[k]) for k in got_a)


def test_ut2923_put_batch_atomic_validation() -> None:
    """UT-2923: an invalid batch writes nothing."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=64))
    entries = [
        CachePutEntry("a", 0, _acts(16)),
        CachePutEntry("b", 0, {"bad": torch.randn(4)}),
    ]
    with pytest.raises(ValueError):
        cache.put_batch(entries)
    assert cache.stats()["tokens"] == {}
    assert cache.write_op_count == 0


def test_ut2924_two_dimensional_activations() -> None:
    """UT-2924: 2-D activations are accepted like ActivationCache."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=32))
    acts = {"ffn_out": torch.randn(1, 8, 4)}
    cache.put("b", 0, acts)
    got = cache.fetch("b", 0)["ffn_out"]
    assert got.shape == (1, 8, 4)


def test_ut2925_layers_are_independent() -> None:
    """UT-2925: a small deep budget does not evict a shallow layer."""
    plan = build_pyramid_plan(
        depth=2, total_tokens=2 * 64, block_size=16, max_ratio=1.0, min_ratio=0.25
    )
    cache = PyramidActivationCache(plan)
    cache.put("b", 0, _acts(64, seed=1))
    cache.put("b", 1, _acts(64, seed=2))
    assert cache.stats()["tokens"][0] == 64
    assert plan.budget_tokens(0) > plan.budget_tokens(1)


def test_ut2926_fetch_nan_for_missing() -> None:
    """UT-2926: evicted token rows are NaN; surviving rows match."""
    cache = PyramidActivationCache(
        _plan(depth=1, per_layer=48, recent_blocks=0, anchor_blocks=0, anchor_stride=100)
    )
    cache.put("old", 0, _acts(32, seed=1))
    cache.put("new", 0, _acts(32, seed=2))
    got = cache.fetch("old", 0)
    avail = cache.availability("old", 0, 32)
    # Block 0 is a stride anchor (design formula includes k=0), so the tail
    # block is evicted first.
    assert avail[:16].all()
    assert not avail[16:].any()
    assert not torch.isnan(got["ffn_out"][:, :16]).any()
    assert torch.isnan(got["ffn_out"][:, 16:]).all()


def test_ut2927_get_semantics() -> None:
    """UT-2927: unrequested positions are 0; requested missing are NaN."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=32, recent_blocks=0))
    cache.put("old", 0, _acts(40, seed=1))
    mask = torch.ones(1, 40, dtype=torch.bool)
    got = cache.get("old", 0, mask)
    assert torch.isnan(got["ffn_out"][:, 16:32]).all()
    assert not torch.isnan(got["ffn_out"][:, 0:16]).any()

    keep = torch.zeros(1, 4, dtype=torch.bool)
    keep[:, 1] = True
    got2 = cache.get("old", 0, keep)
    assert torch.equal(got2["ffn_out"][:, 1], cache.fetch("old", 0)["ffn_out"][:, 1])
    assert got2["ffn_out"][:, 0].abs().sum() == 0


def test_ut2928_fetch_masked_alias() -> None:
    """UT-2928: fetch_masked and get agree."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=32))
    cache.put("b", 0, _acts(16))
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[:, 3:6] = True
    a = cache.get("b", 0, mask)
    b = cache.fetch_masked("b", 0, mask)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_ut2929_availability_truth() -> None:
    """UT-2929: availability is exactly the set of stored token positions."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=32, recent_blocks=0))
    cache.put("b", 0, _acts(32))
    assert cache.availability("b", 0, 32).all()
    assert not cache.availability("b", 0, 48)[32:].any()
    assert not cache.availability("none", 0, 32).any()


def test_ut2930_availability_parent_id_semantics() -> None:
    """UT-2930: a child id has no entry, so availability is all False."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=32))
    cache.put("parent", 0, _acts(32))
    assert cache.availability("parent", 0, 32).all()
    assert not cache.availability("child", 0, 32).any()


def test_ut2931_ut2933_similarity_metrics_missing() -> None:
    """UT-2931..2933: cosine/l2/pearson all mark NaN rows unstable."""
    from actfold.core.similarity_gate import SimilarityGate

    child = torch.randn(1, 16, 8)
    parent = child.clone()
    parent[:, 8:] = float("nan")
    for metric in ("cosine", "l2", "pearson"):
        gate = SimilarityGate(tau=0.9, metric=metric)
        mask = gate(child, parent)
        assert not mask[:, 8:].any(), metric


def test_ut2937_tau_boundaries() -> None:
    """UT-2937: missing rows stay unstable at tau=0.9 and tau=0.0."""
    from actfold.core.similarity_gate import SimilarityGate

    child = torch.randn(1, 8, 8)
    parent = torch.full_like(child, float("nan"))
    for tau in (0.9, 0.0):
        assert not SimilarityGate(tau=tau, metric="cosine")(child, parent).any()


def test_ut2968_availability_step_key() -> None:
    """UT-2968: folded entries live at step 0; a nonzero step is empty."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=32))
    cache.put("b", 0, _acts(32))
    assert cache.availability("b", 0, 32, 0).all()
    assert not cache.availability("b", 0, 32, 3).any()


def test_ut2969_entry_deleted_when_empty() -> None:
    """UT-2969: evicting the last block deletes the entry."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=16, recent_blocks=0))
    cache.put("old", 0, _acts(16, seed=1))
    cache.put("new", 0, _acts(16, seed=2))
    assert not cache.contains("old", 0)
    with pytest.raises(KeyError):
        cache.fetch("old", 0)


def test_ut2970_stats_schema() -> None:
    """UT-2970: stats has the six documented keys with non-negative values."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=16))
    cache.put("b", 0, _acts(32))
    cache.fetch("b", 0)
    stats = cache.stats()
    assert set(stats) == {"tokens", "blocks", "bytes", "evictions", "requests", "missing_tokens"}
    assert stats["requests"] == 1
    assert stats["missing_tokens"] == 16
    for key in ("bytes", "evictions", "requests", "missing_tokens"):
        assert stats[key] >= 0
    assert all(v >= 0 for v in stats["tokens"].values())
    assert all(v >= 0 for v in stats["blocks"].values())


def test_ut2976_partial_last_block_availability() -> None:
    """UT-2976: positions beyond the written tail are unavailable."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=64))
    cache.put("b", 0, _acts(35))
    avail = cache.availability("b", 0, 48)
    assert avail[:35].all()
    assert not avail[35:].any()


def test_ut2945_full_budget_matches_activation_cache() -> None:
    """UT-2945: with a full budget the pyramid read equals ActivationCache."""
    base = ActivationCache(max_entries_per_layer=128)
    pyr = PyramidActivationCache(_plan(depth=1, per_layer=128))
    acts = _acts(64, batch=2)
    base.put("b", 0, acts)
    pyr.put("b", 0, acts)
    got_base = base.fetch("b", 0)
    got_pyr = pyr.fetch("b", 0)
    assert all(torch.equal(got_base[k], got_pyr[k]) for k in got_base)


def test_ut2947_write_op_count() -> None:
    """UT-2947: write_op_count mirrors ActivationCache's put ledger."""
    base = ActivationCache(max_entries_per_layer=128)
    pyr = PyramidActivationCache(_plan(depth=1, per_layer=128))
    acts = _acts(16)
    base.put("b", 0, acts)
    pyr.put("b", 0, acts)
    assert pyr.write_op_count == base.write_op_count == len(acts)


def test_ut2948_num_entries_tokens() -> None:
    """UT-2948: num_entries returns stored tokens."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=64))
    cache.put("b", 0, _acts(32))
    assert cache.num_entries(0) == 32
    assert cache.num_entries() == 32


def test_ut2950_clear_operations() -> None:
    """UT-2950: clear_branch/clear_layer/clear_all remove state."""
    cache = PyramidActivationCache(_plan(depth=2, per_layer=64))
    cache.put("a", 0, _acts(16))
    cache.put("b", 1, _acts(16))
    cache.clear_branch("a")
    assert not cache.contains("a", 0)
    assert cache.contains("b", 1)
    cache.clear_layer(1)
    assert not cache.contains("b", 1)
    cache.put("c", 0, _acts(16))
    cache.clear_all()
    assert cache.stats()["tokens"] == {}


def test_ut2951_contains() -> None:
    """UT-2951: contains tracks non-empty entries."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=32))
    assert not cache.contains("b", 0)
    cache.put("b", 0, _acts(16))
    assert cache.contains("b", 0)
    assert not cache.contains("b", 1)


@pytest.mark.parametrize("batch", [1, 2])
def test_blocks_own_their_storage(batch: int) -> None:
    """Stored blocks never alias the caller's tensor, so eviction really frees memory."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=64))
    acts = _acts(48, batch=batch)
    src = {name: t.untyped_storage().data_ptr() for name, t in acts.items()}
    cache.put("b", 0, acts)
    for block in cache._layers[0][("b", 0)].blocks.values():
        for name, tensor in block.items():
            assert tensor.untyped_storage().data_ptr() != src[name]
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def test_ut2967b_missing_sentinel_protocol() -> None:
    """UT-2967b: the cache advertises the explicit sentinel capability."""
    cache = PyramidActivationCache(_plan(depth=1, per_layer=32))
    assert isinstance(cache, MissingSentinelCapable)
    assert math.isnan(cache.missing_value)
    assert cache.graph_incompatible is True
