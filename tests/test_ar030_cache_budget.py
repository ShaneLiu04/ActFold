"""AR030 cache budget/COW unit tests (UT-3013..3028)."""

from __future__ import annotations

import pytest
import torch

from actfold.core.paged_fold_cache import PagedFoldCache, RolloutBudgetError
from actfold.core.pyramid_plan import build_pyramid_plan

PAGE = 16
BATCH = 2
HIDDEN = 4


def _cache(depth: int = 2, total: int = 256, page: int = PAGE) -> PagedFoldCache:
    plan = build_pyramid_plan(
        depth=depth,
        total_tokens=total,
        block_size=page,
        max_ratio=1.0,
        min_ratio=1.0,
        recent_blocks=1,
        anchor_blocks=1,
    )
    return PagedFoldCache(plan, page, device="cpu")


def _tokens(seq: int, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {"ffn_out": torch.randn(BATCH, seq, HIDDEN, generator=gen)}


def test_ut3013_shared_page_counts_once_against_budget() -> None:
    cache = _cache(total=PAGE * 2 * 2)
    cache.put("base", 0, _tokens(PAGE))
    cache.fork_branch("base", "child")
    before = cache.page_stats()["physical_pages"]
    cache.put("child", 0, _tokens(PAGE))
    assert cache.page_stats()["physical_pages"] == before
    assert cache.page_stats()["shared_pages"] == 1


def test_ut3014_budget_demotes_to_hard_bound() -> None:
    cache = _cache(total=PAGE * 2)
    for layer in (0, 1):
        for idx in range(4):
            cache.put(f"b{layer}{idx}", layer, _tokens(PAGE, seed=layer * 10 + idx))
    for layer in (0, 1):
        hot = [p for p in cache._store._hot if cache._store.page(p).layer == layer]
        assert len(hot) <= cache._budgets[layer]
    assert cache.stats()["evictions"] > 0
    assert cache.stats()["demotions"] >= cache.stats()["evictions"]


def test_ut3015_protected_pages_are_preferred_but_not_exempt() -> None:
    cache = _cache(total=PAGE * 2)
    for idx in range(3):
        cache.put("base", 0, _tokens(PAGE, seed=idx))
    assert cache.stats()["hot_pages"] <= cache._budgets[0]
    cache.assert_invariants()


def test_ut3016_explicit_evict_returns_false_for_protected_or_cold() -> None:
    cache = _cache(total=PAGE * 4)
    cache.put("base", 0, _tokens(PAGE))
    page_id = cache._table._branches["base"][0].page_ids[0]
    assert cache.evict(page_id) is False
    cache.demote(page_id)
    assert cache.evict(page_id) is False


def test_ut3017_explicit_demote_counts_only_demotions() -> None:
    cache = _cache(total=PAGE * 4)
    cache.put("base", 0, _tokens(PAGE))
    page_id = cache._table._branches["base"][0].page_ids[0]
    before = cache.stats()
    cache.demote(page_id)
    after = cache.stats()
    assert after["demotions"] == before["demotions"] + 1
    assert after["evictions"] == before["evictions"]
    assert cache.availability("base", 0, PAGE).sum() == 0


def test_ut3018_demote_promote_round_trip_is_bit_exact() -> None:
    cache = _cache(total=PAGE * 4)
    cache.put("base", 0, _tokens(PAGE, seed=5))
    expected = cache.fetch("base", 0)
    page_id = cache._table._branches["base"][0].page_ids[0]
    cache.demote(page_id)
    assert torch.isnan(cache.fetch("base", 0)["ffn_out"]).all()
    cache.promote(page_id)
    got = cache.fetch("base", 0)
    assert torch.equal(got["ffn_out"], expected["ffn_out"])
    assert cache.stats()["promotions"] == 1


def test_ut3019_promote_keeps_hard_bound_and_counts_success_only() -> None:
    cache = _cache(total=PAGE * 2)
    for idx in range(3):
        cache.put(f"b{idx}", 0, _tokens(PAGE, seed=idx))
    cold = cache._store._cold.copy()
    assert cold
    page_id = next(iter(cold))
    cache.promote(page_id)
    assert cache.stats()["hot_pages"] <= cache._budgets[0]
    assert cache.stats()["promotions"] == 1


def test_ut3020_cow_reuses_identical_pages_and_allocates_on_diff() -> None:
    cache = _cache(total=PAGE * 8)
    cache.put("base", 0, _tokens(32, seed=1))
    cache.fork_branch("base", "child")
    pages_before = cache.page_stats()["physical_pages"]
    cache.put("child", 0, _tokens(32, seed=1))
    assert cache.page_stats()["physical_pages"] == pages_before
    changed = _tokens(32, seed=1)
    changed["ffn_out"][:, -3:] += 2.0
    cache.put("child", 0, changed)
    assert cache.page_stats()["physical_pages"] == pages_before + 1


def test_ut3021_cow_off_forces_allocation() -> None:
    plan = build_pyramid_plan(
        depth=1, total_tokens=256, block_size=PAGE, max_ratio=1.0, min_ratio=1.0
    )
    cache = PagedFoldCache(plan, PAGE, device="cpu", enable_cow=False)
    acts = _tokens(32, seed=2)
    cache.put("base", 0, acts)
    cache.fork_branch("base", "child")
    cache.put("child", 0, acts)
    assert cache.stats()["shared_pages"] == 0
    assert cache.page_stats()["physical_pages"] == 4


def test_ut3022_put_batch_is_atomic_on_validation_failure() -> None:
    from actfold.core.cache_protocol import CachePutEntry

    cache = _cache(total=PAGE * 8)
    cache.put("base", 0, _tokens(PAGE))
    before = (cache.stats(), cache.page_stats(), cache.epoch)
    bad = {"ffn_out": torch.randn(BATCH, PAGE, HIDDEN + 1)}
    with pytest.raises(ValueError):
        cache.put_batch(
            [
                CachePutEntry("base", 0, _tokens(PAGE)),
                CachePutEntry("base", 0, bad),
            ]
        )
    assert (cache.stats(), cache.page_stats(), cache.epoch) == before


def test_ut3023_put_batch_increments_epoch_once() -> None:
    from actfold.core.cache_protocol import CachePutEntry

    cache = _cache(total=PAGE * 8)
    before = cache.epoch
    cache.put_batch(
        [
            CachePutEntry("a", 0, _tokens(PAGE)),
            CachePutEntry("b", 0, _tokens(PAGE, seed=1)),
        ]
    )
    assert cache.epoch == before + 1


def test_ut3024_eviction_trace_is_deterministic() -> None:
    def build() -> PagedFoldCache:
        cache = _cache(total=PAGE * 2)
        for idx in range(5):
            cache.put(f"b{idx}", 0, _tokens(PAGE, seed=idx))
        return cache

    left, right = build(), build()
    assert left.eviction_trace() == right.eviction_trace()


def test_ut3025_eviction_trace_marks_shared_pages() -> None:
    cache = _cache(total=PAGE * 2)
    cache.put("base", 0, _tokens(PAGE))
    cache.fork_branch("base", "child")
    for idx in range(4):
        cache.put(f"extra{idx}", 0, _tokens(PAGE, seed=idx))
    trace = cache.eviction_trace()
    assert trace
    assert {"page_id", "shared", "reason", "failed"} <= set(trace[0])


def test_ut3026_rollout_budget_protection_raises_without_changing_others() -> None:
    cache = _cache(total=PAGE * 2)
    cache.put("base", 0, _tokens(PAGE))
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    cache.put("filler", 0, _tokens(PAGE, seed=4))
    cache.put("filler2", 0, _tokens(PAGE, seed=5))
    before = cache.stats()
    with pytest.raises(RolloutBudgetError):
        cache.put("c0", 0, _tokens(PAGE + 1, seed=6))
    assert cache.stats()["hot_pages"] == before["hot_pages"]
    cache.close_rollout(rollout)


def test_ut3027_hot_bytes_match_page_bytes() -> None:
    cache = _cache(total=PAGE * 4)
    cache.put("base", 0, _tokens(PAGE))
    stats = cache.stats()
    pages = sum(cache._store.page_bytes(pid) for pid in cache._store._hot)
    assert stats["hot_bytes"] == pages
    assert stats["hot_bytes"] + stats["cold_bytes"] == sum(
        cache._store.page_bytes(p.page_id) for p in cache._store.iter_pages()
    )


def test_ut3028_defensive_all_protected_still_satisfies_hard_bound() -> None:
    cache = _cache(total=PAGE * 2)
    for idx in range(2):
        cache.put(f"b{idx}", 0, _tokens(PAGE, seed=idx))
    assert cache.stats()["hot_pages"] <= 1
    cache.assert_invariants()


def test_ut3079_exact_budget_boundary_has_zero_evictions() -> None:
    """UT-3079: filling to exactly the budget page count evicts nothing."""
    cache = _cache(depth=1, total=PAGE * 2)
    cache.put("a", 0, _tokens(PAGE, seed=0))
    cache.put("b", 0, _tokens(PAGE, seed=1))
    assert cache.page_stats()["physical_pages"] == 2
    assert cache.stats()["evictions"] == 0
    assert cache.stats()["hot_pages"] == 2
    # One page over the bound: the hard bound holds via a single demotion.
    cache.put("c", 0, _tokens(PAGE, seed=2))
    assert cache.stats()["hot_pages"] == 2
    assert cache.stats()["evictions"] == 1
    cache.assert_invariants()


def test_ut3080_failed_transaction_keeps_trace_marked_failed() -> None:
    """UT-3080: a rolled-back transaction keeps its trace entries (design 4.2.2(1b))."""
    from actfold.core.cache_protocol import CachePutEntry

    cache = _cache(depth=1, total=PAGE * 2)
    cache.put("a", 0, _tokens(PAGE, seed=0))
    cache.put("b", 0, _tokens(PAGE, seed=1))
    before_stats = cache.stats()
    before_pages = cache.page_stats()
    before_epoch = cache.epoch
    trace_len = len(cache.eviction_trace())
    entries = [
        CachePutEntry("c", 0, _tokens(PAGE, seed=2)),  # forces a budget demotion
        CachePutEntry("a", 0, {"other": torch.randn(BATCH, PAGE, HIDDEN)}),  # bad names
    ]
    with pytest.raises(ValueError):
        cache.put_batch(entries)
    trace = cache.eviction_trace()
    assert len(trace) == trace_len + 1
    assert trace[-1]["failed"] is True
    # The live eviction counter is NOT rolled back (design 6.5: counters keep
    # semantic continuity across failed transactions).
    assert cache.stats()["evictions"] == before_stats["evictions"] + 1
    assert cache.page_stats() == before_pages
    assert cache.epoch == before_epoch
    cache.assert_invariants()


def test_ut3081_empty_put_batch_is_a_noop() -> None:
    """UT-3081: an empty put_batch commits nothing and leaves the epoch alone."""
    cache = _cache(total=PAGE * 8)
    cache.put_batch([])
    assert cache.epoch == 0
    assert cache.page_stats()["physical_pages"] == 0
    cache.put("a", 0, _tokens(PAGE))
    before = cache.epoch
    cache.put_batch([])
    assert cache.epoch == before
    assert cache.page_stats()["physical_pages"] == 1
    cache.assert_invariants()


def test_ut3082_refcount_corruption_is_detected() -> None:
    """UT-3082: an out-of-band refcount edit shows up in the leak ledger."""
    cache = _cache(total=PAGE * 8)
    cache.put("a", 0, _tokens(PAGE))
    page_id = cache._table._branches["a"][0].page_ids[0]
    assert cache.stats()["refcount_leaks"] == 0
    cache._store.page(page_id).refcount += 1
    assert cache.stats()["refcount_leaks"] > 0
    cache._store.page(page_id).refcount -= 1
    assert cache.stats()["refcount_leaks"] == 0


def test_ut3083_seq_len_shrink_and_grow_reput() -> None:
    """UT-3083: re-put with a shorter/longer seq_len reshapes the page list."""
    cache = _cache(depth=1, total=PAGE * 8)
    cache.put("a", 0, _tokens(PAGE * 3, seed=0))
    assert len(cache._table._branches["a"][0].page_ids) == 3
    cache.put("a", 0, _tokens(PAGE * 2, seed=1))  # shrink: third page released
    state = cache._table._branches["a"][0]
    assert len(state.page_ids) == 2
    assert state.seq_len == PAGE * 2
    assert cache.page_stats()["physical_pages"] == 2
    cache.put("a", 0, _tokens(PAGE * 3, seed=2))  # grow again
    assert len(cache._table._branches["a"][0].page_ids) == 3
    # A single token is a legal one-short-page entry.
    cache.put("b", 0, _tokens(1, seed=3))
    short = cache._table._branches["b"][0]
    assert len(short.page_ids) == 1
    assert short.page_lens[0] == 1
    assert cache.fetch("b", 0)["ffn_out"].shape == (BATCH, 1, HIDDEN)
    # Changing the registered batch for an existing branch layer is rejected.
    with pytest.raises(ValueError):
        cache.put("a", 0, {"ffn_out": torch.randn(3, PAGE, HIDDEN)})
    cache.assert_invariants()


def test_ut3084_cow_off_cold_page_gets_fresh_allocation() -> None:
    """UT-3084: COW-off never writes in place, including sole-owner cold pages."""
    plan = build_pyramid_plan(
        depth=1, total_tokens=256, block_size=PAGE, max_ratio=1.0, min_ratio=1.0
    )
    cache = PagedFoldCache(plan, PAGE, device="cpu", enable_cow=False)
    cache.put("a", 0, _tokens(PAGE, seed=0))
    old_id = cache._table._branches["a"][0].page_ids[0]
    cache.demote(old_id)  # sole owner, cold
    cache.put("a", 0, _tokens(PAGE, seed=7))  # divergent write: fresh page, no in-place
    new_id = cache._table._branches["a"][0].page_ids[0]
    assert new_id != old_id
    assert cache._store.page(new_id).tier == "hot"
    assert torch.equal(cache.fetch("a", 0)["ffn_out"], _tokens(PAGE, seed=7)["ffn_out"])
    assert cache.page_stats()["physical_pages"] == 1  # old cold page reclaimed
    cache.assert_invariants()
