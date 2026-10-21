"""AR030 rollout unit tests (UT-3049..3062, UT-3094..3096, UT-3099)."""

from __future__ import annotations

import pytest
import torch

from actfold.core.paged_fold_cache import PagedFoldCache
from actfold.core.pyramid_plan import build_pyramid_plan
from actfold.speculative.branch_tree import BranchNode, BranchTree

PAGE = 16
BATCH = 1
HIDDEN = 4


def _cache(total: int = 256) -> PagedFoldCache:
    plan = build_pyramid_plan(
        depth=2, total_tokens=total, block_size=PAGE, max_ratio=1.0, min_ratio=1.0
    )
    return PagedFoldCache(plan, PAGE, device="cpu")


def _acts(seq: int, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {"ffn_out": torch.randn(BATCH, seq, HIDDEN, generator=gen)}


def _base(cache: PagedFoldCache, seq: int = 32) -> None:
    cache.put("base", 0, _acts(seq, seed=1))


def test_ut3049_fork_rollouts_creates_candidates_without_copying() -> None:
    cache = _cache()
    _base(cache)
    before = cache.page_stats()["physical_pages"]
    rollout = cache.fork_rollouts("base", ["c0", "c1", "c2"])
    assert rollout.candidate_ids == ("c0", "c1", "c2")
    assert cache.page_stats()["physical_pages"] == before
    assert cache.page_stats()["shared_pages"] == 2
    assert cache.epoch == 1 + 3


def test_ut3050_base_is_frozen_while_rollout_active() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0"])
    with pytest.raises(ValueError):
        cache.put("base", 0, _acts(32, seed=2))
    with pytest.raises(ValueError):
        cache.drop_branch("base")
    with pytest.raises(ValueError):
        cache.clear_all()
    cache.close_rollout(rollout)


def test_ut3051_candidate_writes_are_isolated() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    cache.rollout_put(rollout, "c0", 0, _acts(32, seed=3))
    c0 = cache.fetch("c0", 0)["ffn_out"]
    c1 = cache.fetch("c1", 0)["ffn_out"]
    base = cache.fetch("base", 0)["ffn_out"]
    assert torch.equal(c1, base)
    assert not torch.equal(c0, base)
    cache.close_rollout(rollout)


def test_ut3052_rollback_restores_candidate_fork_state() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0"])
    before = cache.fetch("c0", 0)["ffn_out"].clone()
    cache.rollout_put(rollout, "c0", 0, _acts(32, seed=5))
    cache.rollback_candidate(rollout, "c0")
    assert torch.equal(cache.fetch("c0", 0)["ffn_out"], before)
    cache.close_rollout(rollout)


def test_ut3053_close_candidate_releases_only_that_candidate() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    before = cache.fetch("c1", 0)["ffn_out"].clone()
    cache.close_candidate(rollout, "c0")
    assert "c0" not in cache._table.branches()
    assert torch.equal(cache.fetch("c1", 0)["ffn_out"], before)
    cache.close_rollout(rollout)


def test_ut3054_close_rollout_is_atomic_and_marks_closed() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    before = cache.epoch
    cache.close_rollout(rollout)
    assert cache.epoch == before + 1
    assert rollout.closed is True
    assert "c0" not in cache._table.branches()
    with pytest.raises(ValueError):
        cache.rollout_put(rollout, "c0", 0, _acts(32))


def test_ut3055_record_rollout_masks_and_stats() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0"])
    valid = torch.ones(BATCH, 32, dtype=torch.bool)
    reused = torch.zeros(BATCH, 32, dtype=torch.bool)
    reused[:, :16] = True
    cache.record_rollout_masks(rollout, "c0", valid, reused)
    stats = cache.rollout_stats(rollout)["c0"]
    assert stats["valid_tokens"] == 32
    assert stats["reused_tokens"] == 16
    assert stats["reuse_ratio"] == 0.5
    assert stats["recompute_ratio"] == 0.5
    cache.close_rollout(rollout)


def test_ut3056_record_rejects_non_subset_and_closed_handle() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0"])
    valid = torch.zeros(BATCH, 8, dtype=torch.bool)
    reused = torch.ones(BATCH, 8, dtype=torch.bool)
    with pytest.raises(ValueError):
        cache.record_rollout_masks(rollout, "c0", valid, reused)
    cache.close_rollout(rollout)
    with pytest.raises(ValueError):
        cache.record_rollout_masks(rollout, "c0", valid, valid)


def test_ut3057_rollout_window_stages_and_commits_once() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0"])
    plan = cache.build_plan(["base", "c0"])
    before = cache.epoch
    with cache.enter_plan(plan):
        cache.rollout_put(rollout, "c0", 0, _acts(32, seed=8))
        with pytest.raises(RuntimeError):
            cache.rollback_candidate(rollout, "c0")
    assert cache.epoch == before + 1
    assert torch.equal(cache.fetch("c0", 0)["ffn_out"], _acts(32, seed=8)["ffn_out"])
    cache.close_rollout(rollout)


def test_ut3058_rollout_window_failure_restores_all_candidates() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    plan = cache.build_plan(["base", "c0", "c1"])
    before = (cache.epoch, cache.page_stats())
    cache._fail_after_staged = 1
    with pytest.raises(RuntimeError):
        with cache.enter_plan(plan):
            cache.rollout_put(rollout, "c0", 0, _acts(32, seed=9))
            cache.rollout_put(rollout, "c1", 0, _acts(32, seed=10))
    assert (cache.epoch, cache.page_stats()) == before
    cache.close_rollout(rollout)


def test_ut3059_rollout_budget_error_rolls_back_single_write() -> None:
    cache = _cache(total=PAGE * 2)
    _base(cache, PAGE)
    rollout = cache.fork_rollouts("base", ["c0"])
    cache.put("filler", 0, _acts(PAGE, seed=11))
    before = cache.fetch("c0", 0)["ffn_out"].clone()
    with pytest.raises(ValueError):
        cache.rollout_put(rollout, "c0", 0, _acts(PAGE + 1, seed=12))
    assert torch.equal(cache.fetch("c0", 0)["ffn_out"], before)
    assert cache.rollout_stats(rollout)["c0"]["failures"] == 1
    cache.close_rollout(rollout)


def test_ut3060_candidate_put_is_allowed_and_isolated() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0"])
    cache.put("c0", 0, _acts(32, seed=13))
    assert not torch.equal(cache.fetch("c0", 0)["ffn_out"], cache.fetch("base", 0)["ffn_out"])
    cache.close_rollout(rollout)


def test_ut3061_sync_branch_tree_requires_exact_mapping() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0"])
    tree = BranchTree(BranchNode("base", None, torch.zeros(1, 1, dtype=torch.long)))
    tree.add(BranchNode("c0", "base", torch.zeros(1, 1, dtype=torch.long)))
    cache.sync_branch_tree(tree)
    tree.add(BranchNode("ghost", "base", torch.zeros(1, 1, dtype=torch.long)))
    with pytest.raises(ValueError):
        cache.sync_branch_tree(tree)
    cache.close_rollout(rollout)


def test_ut3062_rollout_stats_are_bounded_and_nonnegative() -> None:
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0", "c1"])
    for candidate in ("c0", "c1"):
        stats = cache.rollout_stats(rollout)[candidate]
        assert 0.0 <= stats["reuse_ratio"] <= 1.0
        assert 0.0 <= stats["recompute_ratio"] <= 1.0
        assert int(stats["failures"]) >= 0
        assert int(stats["hot_bytes"]) >= 0
    cache.close_rollout(rollout)


def test_ut3094_rollout_window_validation_failure_counts_and_discards() -> None:
    """UT-3094: a window validation failure counts the failure and discards only
    that candidate's staged writes (design 4.2.2(10))."""
    cache = _cache()
    _base(cache, PAGE)
    handle = cache.fork_rollouts("base", ["w_a", "w_b"])
    plan = cache.build_plan(["base", "w_a", "w_b"])
    with cache.enter_plan(plan):
        cache.rollout_put(handle, "w_a", 0, _acts(PAGE, seed=5))  # staged, then discarded
        with pytest.raises(ValueError):
            cache.rollout_put(handle, "w_a", 0, {"ffn_out": torch.zeros(4)})
        cache.rollout_put(handle, "w_b", 0, _acts(PAGE, seed=6))  # keeps its staged write
    stats = cache.rollout_stats(handle)
    assert stats["w_a"]["failures"] == 1
    assert stats["w_b"]["failures"] == 0
    assert torch.equal(cache.fetch("w_a", 0)["ffn_out"], cache.fetch("base", 0)["ffn_out"])
    assert not torch.equal(cache.fetch("w_b", 0)["ffn_out"], cache.fetch("base", 0)["ffn_out"])
    cache.close_rollout(handle)
    cache.assert_invariants()


def test_ut3095_external_handle_rewrites_do_not_affect_judgment() -> None:
    """UT-3095: judgment uses internal canonical state, not the public handle."""
    cache = _cache()
    _base(cache, PAGE)
    handle = cache.fork_rollouts("base", ["c0", "c1"])
    handle.closed = True  # hostile external rewrite: must NOT freeze the rollout
    cache.rollout_put(handle, "c0", 0, _acts(PAGE, seed=7))  # still legal
    handle.candidate_ids = ("x",)  # external rewrite: stats use canonical copies
    stats = cache.rollout_stats(handle)
    assert set(stats) == {"c0", "c1"}
    cache.rollback_candidate(handle, "c0")
    cache.close_rollout(handle)
    with pytest.raises(ValueError):
        cache.rollout_stats(handle)
    with pytest.raises(ValueError):
        cache.rollout_put(handle, "c1", 0, _acts(PAGE, seed=8))
    cache.assert_invariants()


def test_ut3096_tier_ops_on_rollout_protected_pages() -> None:
    """UT-3096: rollout hard protection blocks demote; explicit evict refuses."""
    cache = _cache()
    _base(cache, PAGE)
    handle = cache.fork_rollouts("base", ["p0"])
    pid = cache._table._branches["base"][0].page_ids[0]
    with pytest.raises(ValueError):
        cache.demote(pid)  # protected by the active rollout
    assert cache.evict(pid) is False
    cache.close_rollout(handle)
    cache.demote(pid)  # legal after close
    assert cache._store.page(pid).tier == "cold"
    cache.assert_invariants()


def test_ut3099_sync_branch_tree_rejects_cycles_and_duplicate_ids() -> None:
    """UT-3099: the sync traversal rejects self-cycles and duplicate ids
    (paged_fold_cache sync_branch_tree guard), and a clean tree still syncs."""
    cache = _cache()
    _base(cache)
    rollout = cache.fork_rollouts("base", ["c0"])
    tree = BranchTree(BranchNode("base", None, torch.zeros(1, 1, dtype=torch.long)))
    tree.add(BranchNode("c0", "base", torch.zeros(1, 1, dtype=torch.long)))
    # Self-cycle: base lists itself among its children.
    tree.get("base").children.append("base")
    with pytest.raises(ValueError, match="cycle or duplicate"):
        cache.sync_branch_tree(tree)
    # Duplicate: c0 is visited twice in one traversal.
    tree.get("base").children.remove("base")
    tree.get("base").children.append("c0")
    with pytest.raises(ValueError, match="cycle or duplicate"):
        cache.sync_branch_tree(tree)
    # The clean tree (base -> c0) still syncs: the guard rejects only
    # broken graphs, not valid ones.
    tree.get("base").children.remove("c0")
    cache.sync_branch_tree(tree)
    cache.close_rollout(rollout)
    cache.assert_invariants()
