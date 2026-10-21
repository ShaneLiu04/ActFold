"""AR030 tiering/snapshot unit tests (UT-3039..3048, UT-3085..3088, UT-3100)."""

from __future__ import annotations

import copy
import time

import pytest
import torch

from actfold.core.paged_fold_cache import PagedFoldCache
from actfold.core.pyramid_plan import build_pyramid_plan

PAGE = 16
BATCH = 2
HIDDEN = 4


def _cache(total: int = 256) -> PagedFoldCache:
    plan = build_pyramid_plan(
        depth=2, total_tokens=total, block_size=PAGE, max_ratio=1.0, min_ratio=1.0
    )
    return PagedFoldCache(plan, PAGE, device="cpu")


def _acts(seq: int, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {"ffn_out": torch.randn(BATCH, seq, HIDDEN, generator=gen)}


def test_ut3039_cold_pages_are_unavailable_and_nan() -> None:
    cache = _cache(total=PAGE * 2)
    cache.put("base", 0, _acts(PAGE))
    cache.put("other", 0, _acts(PAGE, seed=1))
    cache.put("third", 0, _acts(PAGE, seed=2))
    cold = next(iter(cache._store._cold))
    branch, layer = next(
        (b, l) for b, l, s in cache._table.entries() if cold in s.page_ids.values()
    )
    avail = cache.availability(branch, layer, PAGE)
    assert avail.sum() == 0
    assert torch.isnan(cache.fetch(branch, layer)["ffn_out"]).all()


def test_ut3040_stats_and_page_stats_have_frozen_keys() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    stats = cache.stats()
    assert set(stats) == {
        "hot_pages",
        "cold_pages",
        "shared_pages",
        "hot_bytes",
        "cold_bytes",
        "evictions",
        "demotions",
        "promotions",
        "refcount_leaks",
        "requests",
        "missing_tokens",
    }
    page_stats = cache.page_stats()
    assert set(page_stats) == {
        "branches",
        "physical_pages",
        "shared_pages",
        "hot_pages",
        "cold_pages",
        "hot_bytes",
        "cold_bytes",
        "metadata_entries",
        "refcount_leaks",
    }


def test_ut3041_snapshot_restore_round_trip_and_epoch() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24, seed=3))
    cache.fork_branch("base", "child")
    cache.put("child", 0, _acts(24, seed=4))
    snap = cache.snapshot()
    restored = _cache()
    before = restored.epoch
    restored.restore(copy.deepcopy(snap))
    assert restored.epoch == max(before, snap["epoch"]) + 1
    assert restored.page_stats() == cache.page_stats()
    assert torch.equal(restored.fetch("child", 0)["ffn_out"], cache.fetch("child", 0)["ffn_out"])
    assert torch.equal(restored.availability("child", 0, 24), cache.availability("child", 0, 24))


def test_ut3042_snapshot_restore_preserves_tiers() -> None:
    cache = _cache(total=PAGE * 2)
    for idx in range(3):
        cache.put(f"b{idx}", 0, _acts(PAGE, seed=idx))
    assert cache.stats()["cold_pages"] > 0
    snap = cache.snapshot()
    restored = _cache(total=PAGE * 2)
    restored.restore(snap)
    for page_id, raw in snap["pages"].items():
        assert restored._store.page(int(page_id)).tier == raw["tier"]


def test_ut3043_restore_rejects_non_empty_target() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    snap = cache.snapshot()
    with pytest.raises(ValueError):
        cache.restore(snap)


def test_ut3044_restore_rejects_corrupt_checksum() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    snap = cache.snapshot()
    page_id = next(iter(snap["pages"]))
    snap["pages"][page_id]["sha256"] = "0" * 64
    target = _cache()
    with pytest.raises(ValueError):
        target.restore(snap)
    assert target.epoch == 0


def test_ut3045_restore_rejects_refcount_mismatch() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    snap = cache.snapshot()
    page_id = next(iter(snap["pages"]))
    snap["pages"][page_id]["refcount"] += 1
    target = _cache()
    with pytest.raises(ValueError):
        target.restore(snap)


def test_ut3046_restore_keeps_target_counters_monotonic() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24))
    snap = cache.snapshot()
    target = _cache()
    target._requests = 7
    target.restore(snap)
    assert target._requests == 7


def test_ut3047_restore_then_promote_timing() -> None:
    """UT-3047: restore + promote-all-cold under 50ms (warmup + 5, median).

    The timed unit is the SRS acceptance face: a cross-instance ``restore``
    followed by promoting every cold page back to hot.
    """
    cache = _cache(total=PAGE * 4)
    cache.put("base", 0, _acts(PAGE))
    page_id = cache._table._branches["base"][0].page_ids[0]
    cache.demote(page_id)
    snap = cache.snapshot()
    # Warmup rep (discarded).
    warm = _cache(total=PAGE * 4)
    warm.restore(snap)
    warm.promote(page_id)
    times = []
    for _ in range(5):
        target = _cache(total=PAGE * 4)  # restore requires an empty instance
        start = time.perf_counter()
        target.restore(snap)
        if target._store.page(page_id).tier == "cold":
            target.promote(page_id)
        times.append(time.perf_counter() - start)
    assert sorted(times)[2] < 0.05


def test_ut3048_demote_promote_keeps_metadata() -> None:
    cache = _cache()
    cache.put("base", 0, _acts(24, seed=9))
    page_id = cache._table._branches["base"][0].page_ids[0]
    before = cache._store.snapshot_pages()[page_id]
    cache.demote(page_id)
    cold = cache._store.snapshot_pages()[page_id]
    assert cold["write_seq"] == before["write_seq"]
    assert cold["refcount"] == before["refcount"]
    cache.promote(page_id)
    hot = cache._store.snapshot_pages()[page_id]
    assert hot["write_seq"] == before["write_seq"]
    assert hot["refcount"] == before["refcount"]


def test_ut3085_tier_op_edge_errors() -> None:
    """UT-3085: idempotent tier ops and missing-page tier ops are hard errors."""
    cache = _cache(total=PAGE * 4)
    cache.put("a", 0, _acts(PAGE))
    page_id = cache._table._branches["a"][0].page_ids[0]
    cache.demote(page_id)
    with pytest.raises(ValueError):
        cache.demote(page_id)  # already cold
    cache.promote(page_id)
    with pytest.raises(ValueError):
        cache.promote(page_id)  # already hot
    assert cache.evict(100_000) is False  # nonexistent page
    with pytest.raises(KeyError):
        cache.demote(100_000)
    with pytest.raises(KeyError):
        cache.promote(100_000)
    cache.assert_invariants()


def test_ut3086_window_read_path_is_zero_readback(monkeypatch) -> None:
    """UT-3086: inside a plan window the read path performs zero device readback.

    Design 4.2.2(7)/(8): within the window ``fetch_masked`` skips missing-token
    accounting (the outside-window ``(mask & ~availability).sum()`` sync is the
    frozen AR029-parity behavior and is NOT under test here).
    """
    cache = _cache(total=PAGE * 4)
    cache.put("a", 0, _acts(PAGE * 2, seed=1))
    second_page = cache._table._branches["a"][0].page_ids[1]
    cache.demote(second_page)
    plan = cache.build_plan(["a"])
    mask = torch.ones(BATCH, PAGE * 2, dtype=torch.bool)
    calls: list[str] = []
    for name in ("item", "tolist", "cpu"):
        original = getattr(torch.Tensor, name)

        def _spy(*args, _name=name, _original=original, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, _spy)
    original_int = torch.Tensor.__int__

    def _spy_int(value):  # type: ignore[no-untyped-def]
        calls.append("__int__")
        return original_int(value)

    monkeypatch.setattr(torch.Tensor, "__int__", _spy_int, raising=False)
    original_float = torch.Tensor.__float__

    def _spy_float(value):  # type: ignore[no-untyped-def]
        calls.append("__float__")
        return original_float(value)

    monkeypatch.setattr(torch.Tensor, "__float__", _spy_float, raising=False)
    with cache.enter_plan(plan):
        cache.fetch("a", 0)
        cache.get_all("a", 0)
        cache.get("a", 0, mask)
        cache.fetch_masked("a", 0, mask)
        cache.availability("a", 0, PAGE * 2)
        cache.contains("a", 0)
        cache.num_entries()
        cache.stats()
        cache.page_stats()
        cache.eviction_trace()
        cache.assert_invariants()
    assert calls == [], f"window read path performed device readbacks: {calls}"


def test_ut3088_snapshot_entries_schema_is_frozen() -> None:
    """UT-3088: entries use the design schema {seq_len, pages}; corruption is rejected."""
    cache = _cache(total=PAGE * 4)
    cache.put("a", 0, _acts(PAGE, seed=1))
    snap = cache.snapshot()
    entry = snap["entries"]["a"][0]
    assert set(entry) == {"seq_len", "pages"}
    assert entry["seq_len"] == PAGE
    assert set(entry["pages"]) == {0}
    assert entry["pages"][0] == PAGE
    # A corrupted seq_len no longer consistent with the page lens is rejected.
    bad = copy.deepcopy(snap)
    bad["entries"]["a"][0]["seq_len"] = PAGE + 1
    target = _cache(total=PAGE * 4)
    with pytest.raises(ValueError):
        target.restore(bad)
    # The legacy __seq_len__ payload shape is rejected (schema is frozen).
    legacy = copy.deepcopy(snap)
    legacy["entries"]["a"][0] = {"0": PAGE, "__seq_len__": PAGE}
    target2 = _cache(total=PAGE * 4)
    with pytest.raises(ValueError):
        target2.restore(legacy)


def test_ut3100_restore_rejects_cache_with_active_rollout() -> None:
    """UT-3100: restore is refused outright while a rollout is active
    (outside any plan window) — restore requires an empty cache."""
    cache = _cache()
    cache.put("base", 0, _acts(24))
    snapshot = cache.snapshot()
    rollout = cache.fork_rollouts("base", ["c0"])
    with pytest.raises(ValueError, match="no branches/plan/rollout"):
        cache.restore(snapshot)
    cache.close_rollout(rollout)
    # After close the rollout is gone, but restore still refuses the
    # non-empty cache; a fresh instance accepts the same snapshot.
    with pytest.raises(ValueError, match="no branches/plan/rollout"):
        cache.restore(snapshot)
    fresh = _cache()
    fresh.restore(snapshot)
    assert fresh.epoch > 0
    cache.assert_invariants()
