"""Physical page store and branch page table for the AR030 paged cache.

The page layer is deliberately independent of the cache/plan layer.  ``PageStore``
owns physical pages (tensors, tier, refcounts, byte accounting); ``PageTable``
owns only host metadata (branch/layer -> logical page -> physical page id).  A
caller that needs both (the cache) keeps the two refcount ledgers in sync by
calling ``acquire``/``release`` alongside table mutations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import torch


@dataclass
class _Page:
    page_id: int
    layer: int
    page_len: int
    names: tuple[str, ...]
    tensors: dict[str, torch.Tensor]
    tier: str
    refcount: int
    write_seq: int
    nbytes: int

    def metadata(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "layer": self.layer,
            "page_len": self.page_len,
            "names": tuple(self.names),
            "tier": self.tier,
            "refcount": self.refcount,
            "write_seq": self.write_seq,
            "bytes": self.nbytes,
        }


class PageStore:
    """Own physical pages and their hot/cold residency."""

    def __init__(self, device: str = "cuda", page_tokens: int = 16) -> None:
        if page_tokens <= 0:
            raise ValueError(f"page_tokens must be > 0, got {page_tokens}")
        self.device = device
        self.page_tokens = page_tokens
        self._pages: dict[int, _Page] = {}
        self._hot: set[int] = set()
        self._cold: set[int] = set()
        self._next_page_id = 0
        self._next_write_seq = 0

    # ------------------------------------------------------------------
    # allocation / references
    # ------------------------------------------------------------------
    def allocate(self, layer: int, page_len: int, tensors: dict[str, torch.Tensor]) -> int:
        if layer < 0:
            raise ValueError(f"layer must be >= 0, got {layer}")
        if page_len < 1 or page_len > self.page_tokens:
            raise ValueError(f"page_len must be in [1, {self.page_tokens}], got {page_len}")
        self._validate_tensors(page_len, tensors)
        names = tuple(sorted(tensors))
        moved = {name: tensor.detach().to(self.device).clone() for name, tensor in tensors.items()}
        nbytes = sum(int(t.numel()) * t.element_size() for t in moved.values())
        page_id = self._next_page_id
        self._next_page_id += 1
        page = _Page(
            page_id=page_id,
            layer=layer,
            page_len=page_len,
            names=names,
            tensors=moved,
            tier="hot",
            refcount=0,
            write_seq=self._next_write_seq,
            nbytes=nbytes,
        )
        self._next_write_seq += 1
        self._pages[page_id] = page
        self._hot.add(page_id)
        return page_id

    def _validate_tensors(self, page_len: int, tensors: dict[str, torch.Tensor]) -> None:
        if not tensors:
            raise ValueError("tensors must not be empty")
        batch: int | None = None
        dtype: torch.dtype | None = None
        for name, tensor in tensors.items():
            if not isinstance(name, str) or not name:
                raise ValueError("tensor names must be non-empty strings")
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
                raise ValueError(f"tensor {name!r} must be a >=2-D torch tensor")
            if int(tensor.shape[1]) != page_len:
                raise ValueError(
                    f"tensor {name!r} has seq len {tensor.shape[1]}, expected {page_len}"
                )
            if batch is None:
                batch = int(tensor.shape[0])
                dtype = tensor.dtype
            elif int(tensor.shape[0]) != batch:
                raise ValueError("all page tensors must share the same batch size")
            elif tensor.dtype != dtype:
                raise ValueError("all page tensors must share the same dtype")

    def acquire(self, page_id: int) -> None:
        page = self._page(page_id)
        page.refcount += 1

    def release(self, page_id: int) -> None:
        page = self._page(page_id)
        if page.refcount <= 0:
            raise ValueError(f"page {page_id} refcount underflow")
        page.refcount -= 1
        if page.refcount == 0:
            self._reclaim(page_id)

    def _reclaim(self, page_id: int) -> None:
        self._pages.pop(page_id, None)
        self._hot.discard(page_id)
        self._cold.discard(page_id)

    # ------------------------------------------------------------------
    # residency
    # ------------------------------------------------------------------
    def demote(self, page_id: int) -> None:
        page = self._page(page_id)
        if page.tier == "cold":
            raise ValueError(f"page {page_id} is already cold")
        page.tensors = {name: tensor.detach().to("cpu") for name, tensor in page.tensors.items()}
        page.tier = "cold"
        self._hot.discard(page_id)
        self._cold.add(page_id)

    def promote(self, page_id: int, protected: Any = None) -> None:
        page = self._page(page_id)
        if page.tier == "hot":
            raise ValueError(f"page {page_id} is already hot")
        page.tensors = {
            name: tensor.detach().to(self.device) for name, tensor in page.tensors.items()
        }
        page.tier = "hot"
        self._cold.discard(page_id)
        self._hot.add(page_id)

    def tier(self, page_id: int) -> str:
        return self._page(page_id).tier

    def page(self, page_id: int) -> _Page:
        return self._page(page_id)

    def _page(self, page_id: int) -> _Page:
        page = self._pages.get(page_id)
        if page is None:
            raise KeyError(f"unknown page id {page_id}")
        return page

    # ------------------------------------------------------------------
    # accounting
    # ------------------------------------------------------------------
    def page_bytes(self, page_id: int) -> int:
        return self._page(page_id).nbytes

    def page_tensors(self, page_id: int) -> dict[str, torch.Tensor]:
        return dict(self._page(page_id).tensors)

    def replace_tensors(
        self, page_id: int, page_len: int, tensors: dict[str, torch.Tensor]
    ) -> None:
        page = self._page(page_id)
        self._validate_tensors(page_len, tensors)
        if tuple(sorted(tensors)) != page.names:
            raise ValueError("replacement tensors must keep the page name set")
        target = "cpu" if page.tier == "cold" else self.device
        moved = {name: tensor.detach().to(target).clone() for name, tensor in tensors.items()}
        page.tensors = moved
        page.page_len = page_len
        page.nbytes = sum(int(t.numel()) * t.element_size() for t in moved.values())

    def clone_state(self) -> tuple[dict[int, _Page], set[int], set[int], int, int]:
        pages = {
            page_id: _Page(
                page_id=page.page_id,
                layer=page.layer,
                page_len=page.page_len,
                names=tuple(page.names),
                tensors={name: t.clone() for name, t in page.tensors.items()},
                tier=page.tier,
                refcount=page.refcount,
                write_seq=page.write_seq,
                nbytes=page.nbytes,
            )
            for page_id, page in self._pages.items()
        }
        return pages, set(self._hot), set(self._cold), self._next_page_id, self._next_write_seq

    def restore_state(
        self,
        state: tuple[dict[int, _Page], set[int], set[int], int, int],
    ) -> None:
        pages, hot, cold, next_id, next_seq = state
        self._pages = {pid: page for pid, page in pages.items()}
        self._hot = set(hot)
        self._cold = set(cold)
        self._next_page_id = next_id
        self._next_write_seq = next_seq

    def stats(self) -> dict[str, int]:
        hot_bytes = sum(self._pages[pid].nbytes for pid in self._hot)
        cold_bytes = sum(self._pages[pid].nbytes for pid in self._cold)
        leaks = sum(1 for page in self._pages.values() if page.refcount <= 0)
        return {
            "physical_pages": len(self._pages),
            "hot_pages": len(self._hot),
            "cold_pages": len(self._cold),
            "shared_pages": sum(1 for p in self._pages.values() if p.refcount >= 2),
            "hot_bytes": hot_bytes,
            "cold_bytes": cold_bytes,
            "refcount_leaks": leaks,
        }

    def snapshot_pages(self) -> dict[int, dict[str, Any]]:
        return {page_id: page.metadata() for page_id, page in self._pages.items()}

    def iter_pages(self) -> Iterator[_Page]:
        return iter(self._pages.values())

    def assert_invariants(self) -> None:
        for page_id, page in self._pages.items():
            if page.refcount < 0:
                raise AssertionError(f"page {page_id} has negative refcount")
            if page.tier not in {"hot", "cold"}:
                raise AssertionError(f"page {page_id} has invalid tier {page.tier}")
            in_hot = page_id in self._hot
            in_cold = page_id in self._cold
            if in_hot == in_cold:
                raise AssertionError(f"page {page_id} must be in exactly one tier")
            if (page.tier == "hot") != in_hot:
                raise AssertionError(f"page {page_id} tier set mismatch")
            if page.page_len < 1 or page.page_len > self.page_tokens:
                raise AssertionError(f"page {page_id} has invalid page_len")
            if page.nbytes != sum(int(t.numel()) * t.element_size() for t in page.tensors.values()):
                raise AssertionError(f"page {page_id} byte ledger mismatch")
        if len(self._pages) != len(self._hot) + len(self._cold):
            raise AssertionError("page tier sets do not cover the store")


@dataclass
class _BranchLayer:
    seq_len: int = 0
    page_ids: dict[int, int] = field(default_factory=dict)
    page_lens: dict[int, int] = field(default_factory=dict)
    names: tuple[str, ...] = ()
    batch: int = 0
    feat_shape: tuple[int, ...] = ()
    dtype: torch.dtype | None = None

    def clone(self) -> "_BranchLayer":
        return _BranchLayer(
            seq_len=self.seq_len,
            page_ids=dict(self.page_ids),
            page_lens=dict(self.page_lens),
            names=tuple(self.names),
            batch=self.batch,
            feat_shape=tuple(self.feat_shape),
            dtype=self.dtype,
        )


class PageTable:
    """Branch/layer metadata mapping logical pages to physical page ids."""

    def __init__(self, page_tokens: int | None = None) -> None:
        self.page_tokens = page_tokens
        self._branches: dict[str, dict[int, _BranchLayer]] = {}

    def state(self, branch: str, layer: int) -> _BranchLayer:
        if not isinstance(branch, str) or not branch:
            raise ValueError("branch id must be a non-empty string")
        if layer < 0:
            raise ValueError(f"layer must be >= 0, got {layer}")
        layers = self._branches.setdefault(branch, {})
        return layers.setdefault(layer, _BranchLayer())

    def mapping(self, branch: str, layer: int) -> dict[int, int]:
        state = self._branches.get(branch, {}).get(layer)
        if state is None:
            return {}
        return dict(state.page_ids)

    def entries(self) -> Iterator[tuple[str, int, _BranchLayer]]:
        for branch, layers in self._branches.items():
            for layer, state in layers.items():
                yield branch, layer, state

    def fork(self, parent: str, child: str, layers: list[int] | None = None) -> None:
        if parent not in self._branches:
            raise ValueError(f"unknown parent branch {parent!r}")
        if not isinstance(child, str) or not child:
            raise ValueError("child id must be a non-empty string")
        if child in self._branches:
            raise ValueError(f"branch {child!r} already exists")
        parent_layers = self._branches[parent]
        selected = sorted(parent_layers) if layers is None else list(layers)
        if not selected:
            raise ValueError("fork requires at least one layer")
        for layer in selected:
            if layer not in parent_layers:
                raise ValueError(f"parent {parent!r} has no layer {layer}")
        child_layers: dict[int, _BranchLayer] = {}
        for layer in selected:
            child_layers[layer] = parent_layers[layer].clone()
        self._branches[child] = child_layers

    def drop(self, branch: str) -> None:
        if branch not in self._branches:
            raise ValueError(f"unknown branch {branch!r}")
        del self._branches[branch]

    def clear(self) -> None:
        self._branches.clear()

    def clone_all(self) -> dict[str, dict[int, _BranchLayer]]:
        return {
            branch: {layer: state.clone() for layer, state in layers.items()}
            for branch, layers in self._branches.items()
        }

    def replace_all(self, branches: dict[str, dict[int, _BranchLayer]]) -> None:
        self._branches = {
            branch: {layer: state.clone() for layer, state in layers.items()}
            for branch, layers in branches.items()
        }

    def refcounts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, _, state in self.entries():
            for page_id in state.page_ids.values():
                counts[page_id] = counts.get(page_id, 0) + 1
        return counts

    def branches(self) -> tuple[str, ...]:
        return tuple(self._branches)

    def assert_invariants(self) -> None:
        for branch, layer, state in self.entries():
            logical = sorted(state.page_ids)
            if logical != list(range(len(logical))):
                raise AssertionError(f"{branch}/{layer} logical pages are not contiguous")
            if sorted(state.page_lens) != logical:
                raise AssertionError(f"{branch}/{layer} page_lens keys mismatch")
            if state.seq_len != sum(state.page_lens.values()):
                raise AssertionError(f"{branch}/{layer} seq_len mismatch")
            for pos, length in state.page_lens.items():
                if length < 1:
                    raise AssertionError(f"{branch}/{layer} page {pos} length < 1")
                if self.page_tokens is not None and length > self.page_tokens:
                    raise AssertionError(f"{branch}/{layer} page {pos} too long")
            if self.page_tokens is not None and len(logical) > 1:
                for pos in logical[:-1]:
                    if state.page_lens[pos] != self.page_tokens:
                        raise AssertionError(f"{branch}/{layer} non-tail page {pos} is not full")
