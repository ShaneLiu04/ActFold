"""AR030 page-table unit tests (UT-3001..3012).

The page layer is deliberately independent of the cache/plan layer: `PageStore`
owns physical pages and their refcounts, `PageTable` owns branch/layer metadata
(logical -> physical page ids) only.  These tests therefore never touch step
indices, plans or rollout state.
"""

from __future__ import annotations

import pytest
import torch

from actfold.core.page_table import PageStore, PageTable

PAGE_TOKENS = 16
BATCH = 2
HIDDEN = 4
NAMES = ("ffn_out",)


def _tokens(page_len: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, page_len, HIDDEN, generator=gen)


def _page_tensors(
    page_len: int, names: tuple[str, ...] = NAMES, seed: int = 0
) -> dict[str, torch.Tensor]:
    return {name: _tokens(page_len, seed + idx) for idx, name in enumerate(names)}


def _allocate(
    store: PageStore,
    layer: int = 0,
    page_len: int = PAGE_TOKENS,
    names: tuple[str, ...] = NAMES,
    seed: int = 0,
) -> int:
    page_id = store.allocate(layer, page_len, _page_tensors(page_len, names, seed))
    assert isinstance(page_id, int)
    return page_id


def _bind(
    table: PageTable,
    branch: str,
    layer: int,
    page_ids: dict[int, int],
    page_lens: dict[int, int],
    names: tuple[str, ...] = NAMES,
) -> object:
    """Populate one branch/layer state through the design's `state()` accessor.

    `_BranchLayer` is specified to carry `seq_len`, `page_ids`, `page_lens` and
    `names`; `state(branch, layer)` is the only page-table accessor for it.
    """

    state = table.state(branch, layer)
    state.page_ids = dict(page_ids)
    state.page_lens = dict(page_lens)
    state.seq_len = sum(page_lens.values())
    state.names = tuple(names)
    return state


def _entry_keys(table: PageTable) -> set[tuple[str, int]]:
    return {(branch, layer) for branch, layer, _ in table.entries()}


def _snapshot(store: PageStore) -> dict:
    pages = store.snapshot_pages()
    assert isinstance(pages, dict)
    return pages


def test_ut3001_allocate_unique_ids_and_exact_page_bytes() -> None:
    """UT-3001: allocation yields unique increasing ids and exact byte counts."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    names = ("embedding", "ffn_out")
    pages = [_allocate(store, page_len=PAGE_TOKENS, names=names, seed=step) for step in range(3)]

    assert len(set(pages)) == 3
    assert pages == sorted(pages)

    expected_bytes = sum(PAGE_TOKENS * BATCH * HIDDEN * 4 for _ in names)
    for page_id in pages:
        assert store.page_bytes(page_id) == expected_bytes
        assert page_id in _snapshot(store)
        assert _snapshot(store)[page_id]["page_len"] == PAGE_TOKENS

    stats = store.stats()
    assert stats["physical_pages"] == 3
    assert stats["hot_pages"] == 3
    assert stats["cold_pages"] == 0


@pytest.mark.parametrize("layer,page_len", [(-1, PAGE_TOKENS), (0, 0), (0, -1), (0, 17)])
def test_ut3002_invalid_page_length_and_layer_rejected(layer: int, page_len: int) -> None:
    """UT-3002: invalid page length or negative layer is rejected atomically."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    with pytest.raises(ValueError):
        store.allocate(layer, page_len, _page_tensors(PAGE_TOKENS))

    assert store.stats()["physical_pages"] == 0
    assert _snapshot(store) == {}


def test_ut3003_duplicate_child_fork_rejected() -> None:
    """UT-3003: forking onto an existing child id raises and changes nothing."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    table = PageTable()
    page_id = _allocate(store)
    _bind(table, "base", 0, {0: page_id}, {0: PAGE_TOKENS})

    table.fork("base", "child", layers=[0])
    before_entries = _entry_keys(table)
    before_refs = table.refcounts()
    before_pages = _snapshot(store)

    with pytest.raises(ValueError):
        table.fork("base", "child", layers=[0])

    assert _entry_keys(table) == before_entries
    assert table.refcounts() == before_refs
    assert _snapshot(store) == before_pages


def test_ut3004_fork_copies_metadata_not_tensors() -> None:
    """UT-3004: fork copies host metadata only; physical pages are untouched."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    table = PageTable()
    p0 = _allocate(store, seed=1)
    p1 = _allocate(store, seed=2)
    _bind(table, "base", 0, {0: p0, 1: p1}, {0: PAGE_TOKENS, 1: PAGE_TOKENS})

    bytes_before = {p0: store.page_bytes(p0), p1: store.page_bytes(p1)}
    pages_before = _snapshot(store)
    parent_state = table.state("base", 0)

    table.fork("base", "child", layers=[0])

    assert table.mapping("child", 0) == table.mapping("base", 0) == {0: p0, 1: p1}
    child_state = table.state("child", 0)
    assert child_state is not parent_state
    assert child_state.page_ids is not parent_state.page_ids
    assert child_state.page_lens is not parent_state.page_lens

    assert _snapshot(store) == pages_before
    assert store.stats()["physical_pages"] == 2
    assert {pid: store.page_bytes(pid) for pid in (p0, p1)} == bytes_before
    assert table.refcounts() == {p0: 2, p1: 2}


def test_ut3005_fork_layer_subset_copies_only_selected_layers() -> None:
    """UT-3005: `layers=[...]` forks only the requested branch layers."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    table = PageTable()
    p0 = _allocate(store, layer=0, seed=1)
    p1 = _allocate(store, layer=1, seed=2)
    _bind(table, "base", 0, {0: p0}, {0: PAGE_TOKENS})
    _bind(table, "base", 1, {0: p1}, {0: PAGE_TOKENS})

    table.fork("base", "child", layers=[1])

    assert _entry_keys(table) == {("base", 0), ("base", 1), ("child", 1)}
    assert table.mapping("child", 1) == {0: p1}
    assert table.refcounts() == {p0: 1, p1: 2}
    assert "child" in {branch for branch, _, _ in table.entries()}


def test_ut3006_unknown_parent_and_unknown_layer_rejected() -> None:
    """UT-3006: unknown parents and layers outside the parent state are rejected."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    table = PageTable()
    page_id = _allocate(store)
    _bind(table, "base", 0, {0: page_id}, {0: PAGE_TOKENS})

    with pytest.raises(ValueError):
        table.fork("missing", "child", layers=None)

    with pytest.raises(ValueError):
        table.fork("base", "child", layers=[3])

    assert _entry_keys(table) == {("base", 0)}
    assert table.refcounts() == {page_id: 1}


def test_ut3007_acquire_and_release_adjust_refcount_while_positive() -> None:
    """UT-3007: acquire increments and release decrements without reclaiming early."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    page_id = _allocate(store)
    initial = _snapshot(store)[page_id]["refcount"]

    store.acquire(page_id)
    store.acquire(page_id)
    assert _snapshot(store)[page_id]["refcount"] == initial + 2

    store.release(page_id)
    assert _snapshot(store)[page_id]["refcount"] == initial + 1
    assert page_id in _snapshot(store)
    assert store.page_bytes(page_id) > 0
    assert store.stats()["physical_pages"] == 1


def test_ut3008_zero_refcount_reclaims_page_immediately() -> None:
    """UT-3008: releasing the last reference removes the page from the store."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    page_id = _allocate(store)
    store.acquire(page_id)

    store.release(page_id)

    assert page_id not in _snapshot(store)
    stats = store.stats()
    assert stats["physical_pages"] == 0
    assert stats["hot_pages"] == 0
    assert stats["cold_pages"] == 0
    with pytest.raises((KeyError, ValueError)):
        store.page_bytes(page_id)


def test_ut3009_drop_releases_all_branch_references() -> None:
    """UT-3009: dropping a branch removes its mappings and per-page references."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    table = PageTable()
    p0 = _allocate(store, seed=1)
    p1 = _allocate(store, seed=2)
    _bind(table, "base", 0, {0: p0, 1: p1}, {0: PAGE_TOKENS, 1: PAGE_TOKENS})
    table.fork("base", "child", layers=[0])

    table.drop("child")

    assert _entry_keys(table) == {("base", 0)}
    assert table.refcounts() == {p0: 1, p1: 1}

    table.drop("base")
    assert _entry_keys(table) == set()
    assert table.refcounts() == {}
    assert list(table.entries()) == []


def test_ut3010_zero_refcount_reclaims_from_hot_and_cold() -> None:
    """UT-3010: reclaim removes pages from both the hot and cold tiers."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    hot = _allocate(store, seed=1)
    cold = _allocate(store, seed=2)
    store.acquire(hot)
    store.acquire(cold)

    store.demote(cold)
    assert _snapshot(store)[cold]["tier"] == "cold"
    assert store.stats()["hot_pages"] == 1
    assert store.stats()["cold_pages"] == 1

    store.release(hot)
    assert hot not in _snapshot(store)
    assert store.stats()["hot_pages"] == 0
    assert store.stats()["cold_pages"] == 1

    store.release(cold)
    assert cold not in _snapshot(store)
    assert _snapshot(store) == {}
    assert store.stats()["physical_pages"] == 0
    assert store.stats()["hot_pages"] == 0
    assert store.stats()["cold_pages"] == 0


def test_ut3011_refcounts_match_actual_references() -> None:
    """UT-3011: `refcounts()` equals the number of branch mappings per page."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    table = PageTable()
    p0 = _allocate(store, layer=0, seed=1)
    p1 = _allocate(store, layer=0, seed=2)
    p2 = _allocate(store, layer=1, seed=3)
    _bind(table, "base", 0, {0: p0, 1: p1}, {0: PAGE_TOKENS, 1: PAGE_TOKENS})
    _bind(table, "base", 1, {0: p2}, {0: PAGE_TOKENS})

    table.fork("base", "c1", layers=None)
    table.fork("base", "c2", layers=[0])

    assert table.refcounts() == {p0: 3, p1: 3, p2: 2}

    table.drop("c2")
    assert table.refcounts() == {p0: 2, p1: 2, p2: 2}

    actual: dict[int, int] = {}
    for _, _, state in table.entries():
        for page_id in state.page_ids.values():
            actual[page_id] = actual.get(page_id, 0) + 1
    assert table.refcounts() == actual
    assert all(count > 0 for count in table.refcounts().values())


def test_ut3012_logical_pages_contiguous_and_last_page_short() -> None:
    """UT-3012: logical pages are contiguous, lengths sum and only the tail is short."""
    store = PageStore(device="cpu", page_tokens=PAGE_TOKENS)
    table = PageTable()
    p0 = _allocate(store, page_len=PAGE_TOKENS, seed=1)
    p1 = _allocate(store, page_len=PAGE_TOKENS, seed=2)
    p2 = _allocate(store, page_len=8, seed=3)
    lens = {0: PAGE_TOKENS, 1: PAGE_TOKENS, 2: 8}
    _bind(table, "base", 0, {0: p0, 1: p1, 2: p2}, lens)

    state = table.state("base", 0)
    logical = sorted(state.page_ids)
    assert logical == list(range(len(logical)))
    assert sorted(state.page_lens) == logical
    assert state.seq_len == sum(state.page_lens.values()) == 40
    assert all(state.page_lens[p] == PAGE_TOKENS for p in logical[:-1])
    assert 1 <= state.page_lens[logical[-1]] <= PAGE_TOKENS
    assert state.page_lens[logical[-1]] == 8
    assert table.mapping("base", 0) == {0: p0, 1: p1, 2: p2}
    assert store.page_bytes(p2) == 8 * BATCH * HIDDEN * 4
