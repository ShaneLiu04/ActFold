"""AR030 plan/run unit tests (UT-3029..3038)."""

from __future__ import annotations

import pytest
import torch

from actfold.core.fold_plan import FoldPagePlan, PlanStaleError
from actfold.core.paged_fold_cache import PagedFoldCache
from actfold.core.pyramid_plan import build_pyramid_plan

PAGE = 16
BATCH = 1
HIDDEN = 3


def _cache(total: int = 128) -> PagedFoldCache:
    plan = build_pyramid_plan(
        depth=2, total_tokens=total, block_size=PAGE, max_ratio=1.0, min_ratio=1.0
    )
    return PagedFoldCache(plan, PAGE, device="cpu")


def _acts(seq: int, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {"ffn_out": torch.randn(BATCH, seq, HIDDEN, generator=gen)}


def test_ut3029_plan_is_frozen_and_serializable() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    plan = cache.build_plan(["base"])
    assert isinstance(plan, FoldPagePlan)
    with pytest.raises(Exception):
        plan.epoch = 9  # type: ignore[misc]
    roundtrip = FoldPagePlan.from_dict(plan.to_dict())
    assert roundtrip.to_dict() == plan.to_dict()
    assert roundtrip.plan_id == plan.plan_id


def test_ut3030_plan_availability_matches_cache_without_cold() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    plan = cache.build_plan(["base"])
    expected = cache.availability("base", 0, 24)
    got = plan.availability_for("base", 0)
    assert tuple(bool(x) for x in expected) == got


def test_ut3031_plan_marks_cold_pages_unavailable() -> None:
    cache = _cache(total=PAGE * 2)
    cache.put("base", 0, _acts(PAGE))
    cache.put("other", 0, _acts(PAGE, seed=1))
    cache.put("third", 0, _acts(PAGE, seed=2))
    plan = cache.build_plan(["base", "other", "third"])
    assert cache.stats()["cold_pages"] > 0
    unavailable = False
    for branch in plan.branches:
        for layer in plan.layers:
            got = plan.availability_for(branch, layer)
            if got is not None and not all(got):
                unavailable = True
    assert unavailable


def test_ut3032_plan_expected_bytes_is_name_and_page_upper_bound() -> None:
    cache = _cache()
    acts = {
        "ffn_out": torch.randn(BATCH, 24, HIDDEN),
        "embedding": torch.randn(BATCH, 24, HIDDEN),
    }
    cache.put("base", 0, acts)
    plan = cache.build_plan(["base"])
    per_page = PAGE * BATCH * HIDDEN * 4 * 2
    assert plan.expected_bytes == per_page * cache._budgets[0]


def test_ut3033_stale_plan_raises_by_default() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    plan = cache.build_plan(["base"])
    cache.put("base", 0, _acts(24, seed=2))
    with pytest.raises(PlanStaleError):
        with cache.enter_plan(plan):
            pass


def test_ut3034_plan_fallback_rebuilds_and_counts() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    plan = cache.build_plan(["base"])
    cache.put("base", 0, _acts(24, seed=2))
    with cache.enter_plan(plan, allow_plan_fallback=True):
        out = cache.fetch("base", 0)
    assert cache.plan_fallback_count == 1
    assert cache.last_plan_fallback_reason == "epoch_mismatch"
    assert out["ffn_out"].shape == (BATCH, 24, HIDDEN)


def test_ut3035_window_stages_writes_and_commits_epoch_once() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    plan = cache.build_plan(["base"])
    before = cache.epoch
    with cache.enter_plan(plan):
        cache.put("base", 0, _acts(24, seed=4))
        assert cache.epoch == before
        assert torch.equal(cache.fetch("base", 0)["ffn_out"], _acts(24)["ffn_out"])
    assert cache.epoch == before + 1
    assert torch.equal(cache.fetch("base", 0)["ffn_out"], _acts(24, seed=4)["ffn_out"])


def test_ut3036_window_rejects_mutators_and_nested_plan() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    plan = cache.build_plan(["base"])
    with cache.enter_plan(plan):
        with pytest.raises(RuntimeError):
            cache.demote(cache._table._branches["base"][0].page_ids[0])
        with pytest.raises(RuntimeError):
            cache.fork_branch("base", "child")
        with pytest.raises(RuntimeError):
            with cache.enter_plan(plan):
                pass


def test_ut3037_window_fetch_masked_skips_missing_accounting() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    before = cache.stats()["missing_tokens"]
    plan = cache.build_plan(["base"])
    mask = torch.ones(BATCH, 24, dtype=torch.bool)
    with cache.enter_plan(plan):
        cache.fetch_masked("base", 0, mask)
        assert cache.stats()["missing_tokens"] == before


def test_ut3038_window_commit_failure_restores_state() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    plan = cache.build_plan(["base"])
    before = (cache.epoch, cache.stats(), cache.page_stats())
    cache._fail_after_staged = 1
    with pytest.raises(RuntimeError):
        with cache.enter_plan(plan):
            cache.put("base", 0, _acts(24, seed=7))
            cache.put("base", 0, _acts(24, seed=8))
    assert (cache.epoch, cache.stats(), cache.page_stats()) == before
    assert torch.equal(cache.fetch("base", 0)["ffn_out"], _acts(24)["ffn_out"])


def test_ut3089_missing_entry_faces() -> None:
    """UT-3089: missing branch/layer reads follow the AR029 KeyError convention."""
    cache = _cache()
    cache.put("a", 0, _acts(PAGE, seed=1))
    with pytest.raises(KeyError):
        cache.fetch("missing", 0)
    with pytest.raises(KeyError):
        cache.fetch("a", 1)  # layer 1 has no entry
    assert cache.contains("missing", 0) is False
    assert cache.contains("a", 1) is False
    assert not bool(cache.availability("missing", 0, PAGE).any())
    assert not bool(cache.availability("a", 1, PAGE).any())


def test_ut3090_fetch_masked_boundaries() -> None:
    """UT-3090: long masks extend with NaN; empty masks are all-NaN; bad shapes raise."""
    cache = _cache()
    cache.put("a", 0, _acts(PAGE, seed=1))
    long_mask = torch.ones(1, PAGE + 8, dtype=torch.bool)
    out = cache.fetch_masked("a", 0, long_mask)
    assert out["ffn_out"].shape == (1, PAGE + 8, HIDDEN)
    assert torch.isnan(out["ffn_out"][:, PAGE:]).all()
    assert not torch.isnan(out["ffn_out"][:, :PAGE]).any()
    # Design 4.2.2(8): masked-out positions are filled with 0 (not NaN) — the
    # NaN sentinel covers only missing/unmapped tokens before the fill.
    empty = torch.zeros(1, PAGE, dtype=torch.bool)
    out2 = cache.fetch_masked("a", 0, empty)
    assert not torch.isnan(out2["ffn_out"]).any()
    assert torch.equal(out2["ffn_out"], torch.zeros(1, PAGE, HIDDEN))
    with pytest.raises(ValueError):
        cache.fetch_masked("a", 0, torch.ones(PAGE, dtype=torch.bool))
    with pytest.raises(ValueError):
        cache.fetch_masked("a", 0, torch.ones(3, PAGE, dtype=torch.bool))


def test_ut3091_batched_availability_is_2d() -> None:
    """UT-3091: batch>1 entries expose a [batch, seq_len] availability grid."""
    cache = _cache()
    gen = torch.Generator().manual_seed(3)
    cache.put("a", 0, {"ffn_out": torch.randn(2, PAGE * 2, HIDDEN, generator=gen)})
    avail = cache.availability("a", 0, PAGE * 2)
    assert avail.shape == (2, PAGE * 2)
    assert bool(avail.all())
    pid1 = cache._table._branches["a"][0].page_ids[1]
    cache.demote(pid1)
    avail2 = cache.availability("a", 0, PAGE * 2)
    assert avail2.shape == (2, PAGE * 2)
    assert torch.equal(avail2[0], avail2[1])
    assert bool(avail2[:, :PAGE].all())
    assert not bool(avail2[:, PAGE:].any())


def test_ut3092_nonzero_step_raises_on_every_entry() -> None:
    """UT-3092: the frozen step-0 contract raises ValueError on every face."""
    from actfold.core.cache_protocol import CachePutEntry

    cache = _cache()
    cache.put("a", 0, _acts(PAGE, seed=1))
    mask = torch.ones(1, PAGE, dtype=torch.bool)
    with pytest.raises(ValueError):
        cache.put("a", 0, _acts(PAGE, seed=2), step_idx=1)
    with pytest.raises(ValueError):
        cache.fetch("a", 0, step_idx=1)
    with pytest.raises(ValueError):
        cache.get_all("a", 0, step_idx=1)
    with pytest.raises(ValueError):
        cache.get("a", 0, mask, step_idx=1)
    with pytest.raises(ValueError):
        cache.fetch_masked("a", 0, mask, step_idx=1)
    with pytest.raises(ValueError):
        cache.availability("a", 0, PAGE, step_idx=1)
    with pytest.raises(ValueError):
        cache.contains("a", 0, step_idx=1)
    with pytest.raises(ValueError):
        cache.put_batch([CachePutEntry("a", 0, _acts(PAGE, seed=3), 1)])


def test_ut3093_window_rejects_all_remaining_mutators() -> None:
    """UT-3093: every mutator is rejected inside an enter_plan read window."""
    cache = _cache()
    cache.put("base", 0, _acts(PAGE, seed=1))
    cache.put("base", 1, _acts(PAGE, seed=2))
    cache.fork_branch("base", "child")
    handle = cache.fork_rollouts("base", ["c1"])
    snap = cache.snapshot()
    plan = cache.build_plan(["base", "child"])
    pid = cache._table._branches["base"][0].page_ids[0]
    mask = torch.ones(1, PAGE, dtype=torch.bool)
    with cache.enter_plan(plan):
        with pytest.raises(RuntimeError):
            cache.evict(pid)
        with pytest.raises(RuntimeError):
            cache.promote(pid)
        with pytest.raises(RuntimeError):
            cache.snapshot()
        with pytest.raises(RuntimeError):
            cache.restore(snap)
        with pytest.raises(RuntimeError):
            cache.drop_branch("child")
        with pytest.raises(RuntimeError):
            cache.clear_branch("base")
        with pytest.raises(RuntimeError):
            cache.clear_all()
        with pytest.raises(RuntimeError):
            cache.fork_rollouts("base", ["z"])
        with pytest.raises(RuntimeError):
            cache.rollback_candidate(handle, "c1")
        with pytest.raises(RuntimeError):
            cache.close_candidate(handle, "c1")
        with pytest.raises(RuntimeError):
            cache.close_rollout(handle)
        with pytest.raises(RuntimeError):
            cache.record_rollout_masks(handle, "c1", mask, torch.zeros_like(mask))
    cache.assert_invariants()
