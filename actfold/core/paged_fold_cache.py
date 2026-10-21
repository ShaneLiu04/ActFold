"""AR030 page-shared fold memory (PSFM) cache."""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import torch

from actfold.core.cache_protocol import CachePutEntry
from actfold.core.fold_plan import (
    FoldPagePlan,
    PlanStaleError,
    availability_words,
    budget_pages,
    compute_plan_id,
    freeze_mapping,
    page_bytes_for,
)
from actfold.core.page_table import PageStore, PageTable, _BranchLayer
from actfold.core.pyramid_plan import PyramidPlan

try:  # pragma: no cover - optional import cycle guard
    from actfold.speculative.branch_tree import BranchTree
except Exception:  # pragma: no cover
    BranchTree = Any  # type: ignore[assignment,misc]


class RolloutBudgetError(ValueError):
    """Raised when the rollout hard-protection set prevents budget eviction."""


@dataclass
class RolloutSet:
    base_id: str
    candidate_ids: tuple[str, ...]
    closed: bool = False


@dataclass
class _RolloutState:
    handle: RolloutSet
    base_id: str
    candidates: set[str]
    baseline: dict[str, dict[int, _BranchLayer]]
    candidate_order: tuple[str, ...] = ()
    closed: bool = False
    failures: dict[str, int] = field(default_factory=dict)
    valid_tokens: dict[str, int] = field(default_factory=dict)
    reused_tokens: dict[str, int] = field(default_factory=dict)


class PagedFoldCache:
    """Branch-shared, paged activation cache with COW and Tier-0/1 residency."""

    graph_incompatible = True
    missing_value = float("nan")

    def __init__(
        self,
        plan: PyramidPlan,
        page_tokens: int,
        device: str = "cuda",
        *,
        enable_cow: bool = True,
        plan_config: Mapping[str, Any] | None = None,
    ) -> None:
        if page_tokens <= 0 or page_tokens != plan.block_size:
            raise ValueError(
                f"page_tokens must equal plan.block_size ({plan.block_size}), got {page_tokens}"
            )
        self._plan = plan
        self.page_tokens = page_tokens
        self.device = device
        self.enable_cow = enable_cow
        self._store = PageStore(device=device, page_tokens=page_tokens)
        self._table = PageTable(page_tokens=page_tokens)
        self._epoch = 0
        self._write_op_count = 0
        self._requests = 0
        self._missing_tokens = 0
        self._evictions = 0
        self._demotions = 0
        self._promotions = 0
        self._plan_build_count = 0
        self._plan_fallback_count = 0
        self._last_plan_fallback_reason: str | None = None
        self._active_plan: FoldPagePlan | None = None
        self._staged: list[tuple[str, int, dict[str, torch.Tensor], str | None]] = []
        self._plan_sync_count = 0
        self._trace: list[dict[str, Any]] = []
        self._evict_seq = 0
        self._rollouts: dict[int, _RolloutState] = {}
        self._frozen_bases: set[str] = set()
        self._fail_after_staged: int | None = None
        self._budgets = budget_pages(plan, page_tokens)
        self._config = self._derive_plan_config(plan_config)
        self._created_epoch_zero = True

    # ------------------------------------------------------------------
    # properties / introspection
    # ------------------------------------------------------------------
    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def write_op_count(self) -> int:
        return self._write_op_count

    @property
    def plan(self) -> PyramidPlan:
        return self._plan

    @property
    def plan_build_count(self) -> int:
        return self._plan_build_count

    @property
    def plan_fallback_count(self) -> int:
        return self._plan_fallback_count

    @property
    def last_plan_fallback_reason(self) -> str | None:
        return self._last_plan_fallback_reason

    def _derive_plan_config(self, supplied: Mapping[str, Any] | None) -> Mapping[str, Any]:
        base = {
            "depth": self._plan.depth,
            "page_tokens": self.page_tokens,
            "total_tokens": int(self._plan.requested_total),
            "max_ratio": 1.0,
            "min_ratio": 1.0,
            "schedule": str(self._plan.schedule),
            "recent_pages": int(self._plan.recent_blocks),
            "anchor_pages": int(self._plan.anchor_blocks),
            "anchor_stride": int(self._plan.anchor_stride),
            "layer_budgets": [self._budgets[l] for l in range(self._plan.depth)],
        }
        if supplied:
            base.update({str(k): v for k, v in supplied.items()})
        return freeze_mapping(base)

    # ------------------------------------------------------------------
    # validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_step(step_idx: int) -> None:
        if step_idx != 0:
            raise ValueError(f"PagedFoldCache only supports step_idx=0, got {step_idx}")

    def _validate_activations(
        self, activations: dict[str, torch.Tensor]
    ) -> tuple[int, int, tuple[int, ...], torch.dtype, tuple[str, ...]]:
        if not isinstance(activations, dict) or not activations:
            raise ValueError("activations must be a non-empty dict")
        names = tuple(sorted(activations))
        first = activations[names[0]]
        if not isinstance(first, torch.Tensor) or first.ndim < 2:
            raise ValueError("activations must be >=2-D torch tensors")
        batch = int(first.shape[0])
        seq_len = int(first.shape[1])
        dtype = first.dtype
        feat = tuple(int(d) for d in first.shape[2:])
        for name in names:
            tensor = activations[name]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
                raise ValueError(f"activation {name!r} must be a >=2-D tensor")
            if int(tensor.shape[0]) != batch:
                raise ValueError("activation batch sizes differ")
            if int(tensor.shape[1]) != seq_len:
                raise ValueError("activation sequence lengths differ")
            if tensor.dtype != dtype:
                raise ValueError("activation dtypes differ")
            if tuple(int(d) for d in tensor.shape[2:]) != feat:
                raise ValueError("activation feature shapes differ")
        return batch, seq_len, feat, dtype, names

    def _validate_layer(self, layer_idx: int) -> None:
        if not isinstance(layer_idx, int) or layer_idx < 0 or layer_idx >= self._plan.depth:
            raise ValueError(f"layer {layer_idx} outside [0, {self._plan.depth})")

    def _reject_in_window(self) -> None:
        if self._active_plan is not None:
            raise RuntimeError("mutation is not allowed inside an enter_plan read window")

    def _check_branch_writable(self, branch_id: str) -> None:
        if not isinstance(branch_id, str) or not branch_id:
            raise ValueError("branch id must be a non-empty string")
        if branch_id in self._frozen_bases:
            raise ValueError(f"branch {branch_id!r} is frozen by an active rollout")

    # ------------------------------------------------------------------
    # state / transaction helpers
    # ------------------------------------------------------------------
    def _capture_state(self) -> tuple[Any, ...]:
        return (
            self._store.clone_state(),
            self._table.clone_all(),
            self._epoch,
            self._trace_len(),
        )

    def _trace_len(self) -> int:
        return len(self._trace)

    def _restore_state(self, state: tuple[Any, ...]) -> None:
        store_state, table_state, epoch, trace_len = state
        self._store.restore_state(store_state)
        self._table.replace_all(table_state)
        self._epoch = epoch
        # Failed transactions keep their trace entries marked ``failed=True``
        # (design 4.2.2(1b)); deterministic replay / state-equality assertions
        # must exclude the failed entries.
        for item in self._trace[trace_len:]:
            item["failed"] = True

    def _refs(self) -> dict[int, list[tuple[str, int, int]]]:
        refs: dict[int, list[tuple[str, int, int]]] = {}
        for branch, layer, state in self._table.entries():
            for logical, page_id in state.page_ids.items():
                refs.setdefault(page_id, []).append((branch, layer, logical))
        return refs

    def _logical_extent(self, page_id: int) -> tuple[int, int, int]:
        refs = self._refs().get(page_id, [])
        if not refs:
            return 0, 0, 0
        branch, layer, logical = min(refs, key=lambda item: (item[1], item[2], item[0]))
        return layer, logical, logical + 1

    def _protected_pages(self, layer: int) -> set[int]:
        protected: set[int] = set()
        for _branch, state_layer, state in self._table.entries():
            if state_layer != layer or not state.page_ids:
                continue
            n_blocks = int(math.ceil(state.seq_len / self.page_tokens))
            anchors = self._plan.retained_blocks(layer, state.seq_len)
            recent_start = max(0, n_blocks - int(self._plan.recent_blocks))
            for logical, page_id in state.page_ids.items():
                if logical in anchors or logical >= recent_start:
                    protected.add(page_id)
        return protected

    def _unmovable_pages(self) -> set[int]:
        if not self._rollouts:
            return set()
        refs = self._refs()
        protected: set[int] = set()
        for rollout in self._rollouts.values():
            branches = {rollout.base_id} | rollout.candidates
            for page_id, uses in refs.items():
                if any(branch in branches for branch, _, _ in uses):
                    protected.add(page_id)
        return protected

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        self._validate_step(step_idx)
        self._validate_layer(layer_idx)
        self._check_branch_writable(branch_id)
        batch, seq_len, feat, dtype, names = self._validate_activations(activations)
        if self._active_plan is not None:
            self._staged.append((branch_id, layer_idx, dict(activations), None))
            return
        state = self._capture_state()
        try:
            self._put_locked(branch_id, layer_idx, activations, batch, seq_len, feat, dtype, names)
        except Exception:
            self._restore_state(state)
            raise

    def _put_locked(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        batch: int,
        seq_len: int,
        feat: tuple[int, ...],
        dtype: torch.dtype,
        names: tuple[str, ...],
    ) -> None:
        self._check_branch_writable(branch_id)
        old = self._table._branches.get(branch_id, {}).get(layer_idx)
        if old is not None and old.names and tuple(old.names) != names:
            raise ValueError(f"activation names for {branch_id}/{layer_idx} must stay {old.names}")
        if old is not None and old.batch not in (0, batch):
            raise ValueError("activation batch changed for an existing branch layer")
        if old is not None and old.feat_shape not in ((), feat):
            raise ValueError("activation feature shape changed for an existing branch layer")
        state = self._table.state(branch_id, layer_idx)
        old_ids = dict(state.page_ids)
        new_ids: dict[int, int] = {}
        new_lens: dict[int, int] = {}
        n_pages = int(math.ceil(seq_len / self.page_tokens)) if seq_len else 0
        refs = self._refs()
        for logical in range(n_pages):
            start = logical * self.page_tokens
            length = min(self.page_tokens, seq_len - start)
            chunk = {
                name: tensor[:, start : start + length].detach()
                for name, tensor in activations.items()
            }
            old_page_id = old_ids.get(logical)
            reused = False
            if old_page_id is not None and old_page_id in refs:
                old_page = self._store.page(old_page_id)
                same_len = old_page.page_len == length
                same_names = tuple(old_page.names) == names
                if self.enable_cow and same_len and same_names:
                    old_tensors = self._store.page_tensors(old_page_id)
                    equal = True
                    for name in names:
                        left = old_tensors[name]
                        right = chunk[name]
                        if left.device != right.device:
                            right = right.to(left.device)
                        if left.shape != right.shape or not torch.equal(left, right):
                            equal = False
                            break
                    if equal:
                        new_ids[logical] = old_page_id
                        new_lens[logical] = length
                        reused = True
                    elif old_page.refcount == 1:
                        self._store.replace_tensors(old_page_id, length, chunk)
                        new_ids[logical] = old_page_id
                        new_lens[logical] = length
                        reused = True
                if not reused:
                    if old_page.refcount == 1 and old_page.page_len == length and self.enable_cow:
                        self._store.replace_tensors(old_page_id, length, chunk)
                        new_ids[logical] = old_page_id
                        new_lens[logical] = length
                        reused = True
                    elif (
                        old_page.refcount == 1
                        and self.enable_cow
                        and old_page.tier == "cold"
                        and same_names
                        and old_page.page_len == length
                    ):
                        # Cold-page storage recycling is legal only for a sole
                        # owner: a shared cold page (refcount >= 2) is still
                        # live content for its co-owners and must never be
                        # replaced in place. The ``enable_cow`` gate keeps the
                        # COW-off contract (design 4.2.2(6)): every write
                        # takes a fresh page, skipping comparison and in-place
                        # mutation entirely.
                        self._store.replace_tensors(old_page_id, length, chunk)
                        new_ids[logical] = old_page_id
                        new_lens[logical] = length
                        reused = True
            if not reused:
                page_id = self._store.allocate(layer_idx, length, chunk)
                self._store.acquire(page_id)
                new_ids[logical] = page_id
                new_lens[logical] = length
        for logical, page_id in old_ids.items():
            if new_ids.get(logical) != page_id:
                self._store.release(page_id)
        state.page_ids = new_ids
        state.page_lens = new_lens
        state.seq_len = seq_len
        state.names = names
        state.batch = batch
        state.feat_shape = feat
        state.dtype = dtype
        self._enforce_budget(layer_idx)
        self._epoch += 1
        self._write_op_count += len(activations)

    def put_batch(self, entries: Sequence[CachePutEntry]) -> None:
        prepared: list[tuple[CachePutEntry, tuple[Any, ...]]] = []
        for entry in entries:
            self._validate_step(entry.step_idx)
            self._validate_layer(entry.layer_idx)
            self._check_branch_writable(entry.branch_id)
            prepared.append((entry, self._validate_activations(entry.activations)))
        if self._active_plan is not None:
            for entry, _ in prepared:
                self._staged.append(
                    (entry.branch_id, entry.layer_idx, dict(entry.activations), None)
                )
            return
        state = self._capture_state()
        try:
            for entry, meta in prepared:
                batch, seq_len, feat, dtype, names = meta
                self._put_locked(
                    entry.branch_id,
                    entry.layer_idx,
                    entry.activations,
                    batch,
                    seq_len,
                    feat,
                    dtype,
                    names,
                )
        except Exception:
            self._restore_state(state)
            raise
        if prepared:
            self._epoch -= len(prepared) - 1

    # ------------------------------------------------------------------
    # budget / residency
    # ------------------------------------------------------------------
    def _enforce_budget(self, layer: int, extra_unmovable: set[int] | None = None) -> None:
        budget = self._budgets.get(layer)
        if budget is None:
            raise ValueError(f"layer {layer} outside plan depth")
        protected = self._protected_pages(layer)
        unmovable = set(self._unmovable_pages())
        if extra_unmovable:
            unmovable.update(extra_unmovable)
        hot = [pid for pid in self._store._hot if self._store.page(pid).layer == layer]
        while len(hot) > budget:
            candidates = [pid for pid in hot if pid not in unmovable]
            if not candidates:
                raise RolloutBudgetError(
                    f"layer {layer} cannot evict enough pages under rollout protection"
                )
            candidates.sort(
                key=lambda pid: (
                    1 if pid in protected else 0,
                    self._store.page(pid).refcount,
                    self._store.page(pid).write_seq,
                    pid,
                )
            )
            victim = candidates[0]
            self._demote_page(victim, reason="budget", count_eviction=True)
            hot.remove(victim)

    def _demote_page(self, page_id: int, reason: str, count_eviction: bool) -> None:
        page = self._store.page(page_id)
        self._store.demote(page_id)
        self._demotions += 1
        if count_eviction:
            self._evictions += 1
        layer, logical, _ = self._logical_extent(page_id)
        refs = self._refs().get(page_id, [])
        branch = min(refs)[0] if refs else ""
        self._trace.append(
            {
                "seq": self._evict_seq,
                "layer": layer,
                "page_id": page_id,
                "logical": logical,
                "branch": branch,
                "shared": len(refs) > 1,
                "refcount": page.refcount,
                "write_seq": page.write_seq,
                "reason": reason,
                "failed": False,
            }
        )
        self._evict_seq += 1

    def demote(self, page_id: int) -> None:
        self._reject_in_window()
        page = self._store.page(page_id)
        if page.tier == "cold":
            raise ValueError(f"page {page_id} is already cold")
        if page_id in self._unmovable_pages():
            raise ValueError(f"page {page_id} is protected by an active rollout")
        self._demote_page(page_id, reason="explicit", count_eviction=False)
        self._epoch += 1

    def evict(self, page_id: int) -> bool:
        self._reject_in_window()
        try:
            page = self._store.page(page_id)
        except KeyError:
            return False
        if page.tier != "hot" or page_id in self._unmovable_pages():
            return False
        if page_id in self._protected_pages(page.layer):
            return False
        self._demote_page(page_id, reason="explicit", count_eviction=False)
        self._epoch += 1
        return True

    def promote(self, page_id: int) -> None:
        self._reject_in_window()
        page = self._store.page(page_id)
        if page.tier == "hot":
            raise ValueError(f"page {page_id} is already hot")
        state = self._capture_state()
        try:
            self._store.promote(page_id)
            self._enforce_budget(page.layer, extra_unmovable={page_id})
        except Exception:
            self._restore_state(state)
            raise
        self._promotions += 1
        self._epoch += 1

    # ------------------------------------------------------------------
    # branches
    # ------------------------------------------------------------------
    def fork_branch(
        self, parent_id: str, child_id: str, layers: Sequence[int] | None = None
    ) -> None:
        self._reject_in_window()
        self._check_branch_writable(child_id)
        if parent_id in self._frozen_bases:
            raise ValueError(f"cannot fork from frozen base {parent_id!r}")
        selected = None if layers is None else list(layers)
        self._table.fork(parent_id, child_id, selected)
        for _branch, _layer, state in self._table.entries():
            if _branch != child_id:
                continue
            for page_id in state.page_ids.values():
                self._store.acquire(page_id)
        self._epoch += 1

    def drop_branch(self, branch_id: str) -> None:
        self._reject_in_window()
        if branch_id in self._frozen_bases:
            raise ValueError(f"branch {branch_id!r} is frozen by an active rollout")
        for _branch, _layer, state in list(self._table.entries()):
            if _branch != branch_id:
                continue
            for page_id in state.page_ids.values():
                self._store.release(page_id)
        self._table.drop(branch_id)
        self._epoch += 1

    def clear_branch(self, branch_id: str) -> None:
        self.drop_branch(branch_id)

    def clear_all(self) -> None:
        self._reject_in_window()
        if self._frozen_bases or self._rollouts:
            raise ValueError("cannot clear all branches during an active rollout")
        for _branch, _layer, state in list(self._table.entries()):
            for page_id in state.page_ids.values():
                self._store.release(page_id)
        self._table.clear()
        self._epoch += 1

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def _entry(self, branch_id: str, layer_idx: int) -> _BranchLayer:
        state = self._table._branches.get(branch_id, {}).get(layer_idx)
        if state is None or not state.page_ids:
            raise KeyError(
                f"No cached activations for branch={branch_id}, layer={layer_idx}, step=0"
            )
        return state

    def _assemble(self, state: _BranchLayer, seq_len: int) -> dict[str, torch.Tensor]:
        sample = None
        for page_id in state.page_ids.values():
            sample = next(iter(self._store.page_tensors(page_id).values()))
            break
        if sample is None:
            raise KeyError("no pages to assemble")
        out = {
            name: torch.full(
                (state.batch, seq_len, *state.feat_shape),
                self.missing_value,
                dtype=state.dtype or sample.dtype,
                device=self.device,
            )
            for name in state.names
        }
        limit = min(seq_len, state.seq_len)
        for logical, page_id in state.page_ids.items():
            page = self._store.page(page_id)
            start = logical * self.page_tokens
            if start >= limit or page.tier != "hot":
                continue
            stop = min(start + page.page_len, limit)
            if stop <= start:
                continue
            tensors = self._store.page_tensors(page_id)
            for name in state.names:
                out[name][:, start:stop] = tensors[name][:, : stop - start].to(self.device)
        return out

    def fetch(self, branch_id: str, layer_idx: int, step_idx: int = 0) -> dict[str, torch.Tensor]:
        # Read path / plan-window equivalence (design 4.2.2(8), line 368):
        # inside an enter_plan window this reads the LIVE page table rather
        # than the frozen plan's tables/availability. The two views are
        # identical by construction — writes taken inside the window are
        # staged and commit only on exit_plan, so the live table never
        # diverges from the plan snapshot for the duration of the window.
        self._validate_step(step_idx)
        state = self._entry(branch_id, layer_idx)
        out = self._assemble(state, state.seq_len)
        self._requests += 1
        present = sum(
            min(self._store.page(pid).page_len, state.seq_len)
            for pid in state.page_ids.values()
            if self._store.page(pid).tier == "hot"
        )
        self._missing_tokens += max(0, state.seq_len - present) * state.batch
        return out

    def get_all(self, branch_id: str, layer_idx: int, step_idx: int = 0) -> dict[str, torch.Tensor]:
        return self.fetch(branch_id, layer_idx, step_idx)

    def get(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        return self.fetch_masked(branch_id, layer_idx, token_mask, step_idx)

    def fetch_masked(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        self._validate_step(step_idx)
        state = self._entry(branch_id, layer_idx)
        mask = token_mask.to(dtype=torch.bool)
        if mask.ndim != 2 or int(mask.shape[0]) != state.batch:
            raise ValueError(
                f"token_mask must be [batch, seq_len] with batch={state.batch}, got {tuple(mask.shape)}"
            )
        seq_len = int(mask.shape[1])
        out = self._assemble(state, seq_len)
        keep = mask.unsqueeze(-1)
        for name, tensor in out.items():
            out[name] = tensor.masked_fill(~keep, 0)
        self._requests += 1
        if self._active_plan is None:
            avail = self.availability(branch_id, layer_idx, seq_len, 0)
            avail2d = avail.view(1, -1).expand(state.batch, -1) if avail.ndim == 1 else avail
            self._missing_tokens += int((mask & ~avail2d.to(mask.device)).sum())
        return out

    def availability(
        self,
        branch_id: str,
        layer_idx: int,
        seq_len: int,
        step_idx: int = 0,
        name: str = "ffn_out",
    ) -> torch.Tensor:
        # Same plan-window equivalence note as fetch(): the live table equals
        # the frozen plan snapshot for the whole window (staged writes commit
        # only on exit_plan).
        self._validate_step(step_idx)
        state = self._table._branches.get(branch_id, {}).get(layer_idx)
        if state is None:
            return torch.zeros(seq_len, dtype=torch.bool)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        limit = min(seq_len, state.seq_len)
        for logical, page_id in state.page_ids.items():
            page = self._store.page(page_id)
            if page.tier != "hot":
                continue
            start = logical * self.page_tokens
            if start >= limit:
                continue
            stop = min(start + page.page_len, limit)
            if stop > start:
                mask[start:stop] = True
        if state.batch > 1:
            return mask.unsqueeze(0).expand(state.batch, -1)
        return mask

    def contains(self, branch_id: str, layer_idx: int, step_idx: int = 0) -> bool:
        self._validate_step(step_idx)
        state = self._table._branches.get(branch_id, {}).get(layer_idx)
        return state is not None and bool(state.page_ids)

    def num_entries(self, layer_idx: int | None = None) -> int:
        total = 0
        for _branch, layer, state in self._table.entries():
            if layer_idx is None or layer == layer_idx:
                total += state.seq_len
        return total

    # ------------------------------------------------------------------
    # plan / run
    # ------------------------------------------------------------------
    def _plan_config_dict(self) -> dict[str, Any]:
        return {str(k): v for k, v in self._config.items()}

    def build_plan(
        self,
        branch_ids: Sequence[str],
        layers: Sequence[int] | None = None,
        seq_len: Mapping[str, Mapping[int, int]] | None = None,
    ) -> FoldPagePlan:
        if not branch_ids:
            raise ValueError("branch_ids must not be empty")
        layers_t = (
            tuple(range(self._plan.depth)) if layers is None else tuple(int(l) for l in layers)
        )
        for layer in layers_t:
            self._validate_layer(layer)
        tables: dict[str, dict[int, dict[int, int]]] = {}
        avail: dict[str, dict[int, tuple[tuple[int, ...], int]]] = {}
        lens_map: dict[str, dict[int, int]] = {}
        for branch in branch_ids:
            if branch not in self._table.branches():
                raise ValueError(f"unknown branch {branch!r}")
            tables[branch] = {}
            avail[branch] = {}
            lens_map[branch] = {}
            for layer in layers_t:
                state = self._table._branches[branch].get(layer)
                if state is None or not state.page_ids:
                    continue
                length = state.seq_len
                if seq_len is not None:
                    requested = int(seq_len.get(branch, {}).get(layer, length))
                    if requested > length:
                        raise ValueError("planned seq_len exceeds the stored length")
                    length = requested
                tables[branch][layer] = dict(state.page_ids)
                lens_map[branch][layer] = length
                hot_lens = {
                    logical: self._store.page(pid).page_len
                    for logical, pid in state.page_ids.items()
                    if self._store.page(pid).tier == "hot"
                }
                words = availability_words(hot_lens, length, self.page_tokens)
                avail[branch][layer] = (words, length)
        expected = 0
        for layer in layers_t:
            names: set[str] = set()
            batch = 0
            feat: tuple[int, ...] = ()
            dtype: Any = None
            for _branch, l, state in self._table.entries():
                if l != layer:
                    continue
                names.update(state.names)
                batch = max(batch, state.batch)
                feat = state.feat_shape or feat
                dtype = state.dtype or dtype
            if names and batch:
                expected += (
                    page_bytes_for(self.page_tokens, batch, feat, tuple(sorted(names)), dtype)
                    * self._budgets[layer]
                )
        plan = FoldPagePlan(
            plan_id="",
            epoch=self._epoch,
            branches=tuple(branch_ids),
            layers=layers_t,
            seq_len=freeze_mapping(lens_map),
            tables=freeze_mapping(tables),
            availability=freeze_mapping(avail),
            expected_bytes=expected,
            budget_pages=freeze_mapping(self._budgets),
            config=freeze_mapping(self._plan_config_dict()),
        )
        plan = FoldPagePlan(
            plan_id=compute_plan_id(plan),
            epoch=plan.epoch,
            branches=plan.branches,
            layers=plan.layers,
            seq_len=plan.seq_len,
            tables=plan.tables,
            availability=plan.availability,
            expected_bytes=plan.expected_bytes,
            budget_pages=plan.budget_pages,
            config=plan.config,
        )
        self._plan_build_count += 1
        return plan

    @contextmanager
    def enter_plan(self, plan: FoldPagePlan, allow_plan_fallback: bool = False) -> Iterator[None]:
        if self._active_plan is not None:
            raise RuntimeError("nested enter_plan is not allowed")
        if self._epoch != plan.epoch:
            if not allow_plan_fallback:
                raise PlanStaleError(plan.epoch, self._epoch)
            self._plan_fallback_count += 1
            self._last_plan_fallback_reason = "epoch_mismatch"
            plan = self.build_plan(self._table.branches())
        self._active_plan = plan
        self._staged = []
        state = self._capture_state()
        try:
            yield
        except Exception:
            self._staged = []
            self._active_plan = None
            self._restore_state(state)
            raise
        staged = self._staged
        self._staged = []
        try:
            if staged:
                if self._fail_after_staged == 0:
                    self._fail_after_staged = None
                    raise RuntimeError("injected commit failure")
                applied = 0
                for branch, layer, acts, _candidate in staged:
                    batch, seq_len, feat, dtype, names = self._validate_activations(acts)
                    self._put_locked(branch, layer, acts, batch, seq_len, feat, dtype, names)
                    applied += 1
                    if (
                        self._fail_after_staged is not None
                        and self._fail_after_staged > 0
                        and applied >= self._fail_after_staged
                    ):
                        self._fail_after_staged = None
                        raise RuntimeError("injected commit failure")
                self._epoch -= len(staged)
                self._epoch += 1
        except Exception:
            self._restore_state(state)
            self._active_plan = None
            raise
        self._active_plan = None

    # ------------------------------------------------------------------
    # snapshot / restore
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        self._reject_in_window()
        pages: dict[int, dict[str, Any]] = {}
        for page in self._store.iter_pages():
            names = {}
            data = {}
            digest = hashlib.sha256()
            for name in sorted(page.tensors):
                tensor = page.tensors[name].detach().to("cpu").clone()
                data[name] = tensor
                names[name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "bytes": int(tensor.numel() * tensor.element_size()),
                }
                digest.update(tensor.contiguous().numpy().tobytes())
            pages[page.page_id] = {
                "tier": page.tier,
                "refcount": page.refcount,
                "write_seq": page.write_seq,
                "page_len": page.page_len,
                "names": names,
                "data": data,
                "sha256": digest.hexdigest(),
            }
        entries: dict[str, dict[int, dict[str, Any]]] = {}
        tables: dict[str, dict[int, dict[int, int]]] = {}
        for branch, layer, state in self._table.entries():
            tables.setdefault(branch, {})[layer] = dict(state.page_ids)
            # Design 4.2.2(8) frozen schema: {seq_len, pages{logical: page_len}}.
            entries.setdefault(branch, {})[layer] = {
                "seq_len": state.seq_len,
                "pages": dict(state.page_lens),
            }
        return {
            "version": 1,
            "epoch": self._epoch,
            "next_page_id": self._store._next_page_id,
            "next_write_seq": self._store._next_write_seq,
            "pages": pages,
            "entries": entries,
            "tables": tables,
            "checksum_alg": "sha256",
            "plan": {str(k): v for k, v in self._config.items()},
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self._reject_in_window()
        if self._table.branches() or self._active_plan is not None or self._rollouts:
            raise ValueError("restore requires a cache with no branches/plan/rollout")
        try:
            if snapshot.get("version") != 1 or snapshot.get("checksum_alg") != "sha256":
                raise ValueError("unsupported snapshot schema")
            plan_cfg = dict(snapshot["plan"])
            budgets = [int(v) for v in plan_cfg["layer_budgets"]]
            if budgets != [self._budgets[l] for l in range(self._plan.depth)]:
                raise ValueError("snapshot plan budgets do not match this cache")
            pages_raw = snapshot["pages"]
            tables_raw = snapshot["tables"]
            entries_raw = snapshot["entries"]
            page_layers: dict[int, int] = {}
            for branch_key, layers in tables_raw.items():
                for layer_raw, mapping in layers.items():
                    for raw_page in mapping.values():
                        pid = int(raw_page)
                        layer = int(layer_raw)
                        if page_layers.get(pid, layer) != layer:
                            raise ValueError("page is referenced from multiple layers")
                        page_layers[pid] = layer
            store = PageStore(device=self.device, page_tokens=self.page_tokens)
            table = PageTable(page_tokens=self.page_tokens)
            pages_by_id: dict[int, dict[str, Any]] = {}
            for raw_id, raw in pages_raw.items():
                page_id = int(raw_id)
                if page_id < 0 or page_id >= int(snapshot["next_page_id"]):
                    raise ValueError("page id is outside next_page_id")
                page_len = int(raw["page_len"])
                datas = {str(name): torch.as_tensor(t) for name, t in raw["data"].items()}
                digest = hashlib.sha256()
                for name in sorted(datas):
                    digest.update(datas[name].contiguous().numpy().tobytes())
                if digest.hexdigest() != raw["sha256"]:
                    raise ValueError(f"page {page_id} checksum mismatch")
                page_id_new = store.allocate(int(page_layers.get(page_id, 0)), page_len, datas)
                if page_id_new != page_id:
                    raise ValueError("snapshot page ids are not contiguous")
                page = store.page(page_id)
                page.refcount = int(raw["refcount"])
                page.write_seq = int(raw["write_seq"])
                if int(raw["write_seq"]) >= int(snapshot["next_write_seq"]):
                    raise ValueError("write_seq is outside next_write_seq")
                if raw["tier"] == "cold":
                    store.demote(page_id)
                elif raw["tier"] != "hot":
                    raise ValueError("invalid tier in snapshot")
                pages_by_id[page_id] = raw
            for branch, layers in tables_raw.items():
                for layer_raw, mapping in layers.items():
                    layer = int(layer_raw)
                    state = table.state(str(branch), layer)
                    state.page_ids = {int(k): int(v) for k, v in mapping.items()}
                    lens_raw = entries_raw[str(branch)][layer_raw]
                    if (
                        not isinstance(lens_raw, dict)
                        or "seq_len" not in lens_raw
                        or "pages" not in lens_raw
                    ):
                        raise ValueError("invalid snapshot entries schema")
                    state.page_lens = {int(k): int(v) for k, v in lens_raw["pages"].items()}
                    state.seq_len = int(lens_raw["seq_len"])
                    if state.seq_len != sum(state.page_lens.values()):
                        raise ValueError("snapshot seq_len mismatch")
                    first_id = next(iter(state.page_ids.values()))
                    first_page = store.page(first_id)
                    state.names = tuple(first_page.names)
                    state.batch = int(first_page.tensors[state.names[0]].shape[0])
                    state.feat_shape = tuple(
                        int(d) for d in first_page.tensors[state.names[0]].shape[2:]
                    )
                    state.dtype = first_page.tensors[state.names[0]].dtype
                    page_layer = first_page.layer
                    if page_layer != layer:
                        raise ValueError("page layer does not match its table")
            counts: dict[int, int] = {}
            for _, _, state in table.entries():
                for page_id in state.page_ids.values():
                    counts[page_id] = counts.get(page_id, 0) + 1
            for page_id, raw in pages_raw.items():
                pid = int(page_id)
                if counts.get(pid, 0) != int(raw["refcount"]):
                    raise ValueError(f"page {pid} refcount does not match tables")
                if int(raw["refcount"]) <= 0:
                    raise ValueError(f"page {pid} has no live reference")
            for page_id, raw in pages_raw.items():
                page = store.page(int(page_id))
                page.refcount = counts[int(page_id)]
            store._next_page_id = int(snapshot["next_page_id"])
            store._next_write_seq = int(snapshot["next_write_seq"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid snapshot: {exc}") from exc
        self._store = store
        self._table = table
        self._epoch = max(self._epoch, int(snapshot["epoch"])) + 1

    # ------------------------------------------------------------------
    # rollouts
    # ------------------------------------------------------------------
    def _rollout_state(self, rollout: RolloutSet) -> _RolloutState:
        state = self._rollouts.get(id(rollout))
        if state is None or state.handle is not rollout:
            raise ValueError("unknown rollout handle")
        # Design 4.2.2(10): judgment is made against the *internal* closed
        # flag and canonical candidate tuple; a caller rewriting the public
        # handle's fields (e.g. ``handle.closed = False``) must not affect it.
        if state.closed:
            raise ValueError("rollout is closed")
        return state

    def fork_rollouts(self, base_id: str, candidate_ids: Sequence[str]) -> RolloutSet:
        self._reject_in_window()
        if base_id not in self._table.branches():
            raise ValueError(f"unknown base branch {base_id!r}")
        if base_id in self._frozen_bases:
            raise ValueError(f"base {base_id!r} is already frozen")
        candidates = tuple(str(c) for c in candidate_ids)
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("candidate ids must be non-empty and unique")
        if base_id in candidates:
            raise ValueError("base id cannot be a candidate id")
        for candidate in candidates:
            if candidate in self._table.branches():
                raise ValueError(f"candidate branch {candidate!r} already exists")
        for candidate in candidates:
            self.fork_branch(base_id, candidate)
        handle = RolloutSet(base_id=base_id, candidate_ids=candidates)
        baseline = {
            branch: layers
            for branch, layers in self._table.clone_all().items()
            if branch == base_id
        }
        self._rollouts[id(handle)] = _RolloutState(
            handle=handle,
            base_id=base_id,
            candidates=set(candidates),
            baseline=baseline,
            candidate_order=candidates,
            failures={c: 0 for c in candidates},
            valid_tokens={c: 0 for c in candidates},
            reused_tokens={c: 0 for c in candidates},
        )
        self._frozen_bases.add(base_id)
        return handle

    def rollout_put(
        self,
        rollout: RolloutSet,
        candidate_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        state = self._rollout_state(rollout)
        if candidate_id not in state.candidates:
            raise ValueError(f"{candidate_id!r} is not an active candidate of this rollout")
        try:
            self._validate_step(step_idx)
            self._validate_layer(layer_idx)
            self._validate_activations(activations)
        except Exception:
            # Design 4.2.2(10): a validation failure counts into the
            # candidate's failure ledger; inside a plan window the
            # candidate's staged writes are discarded (other candidates'
            # staged writes stay untouched).
            state.failures[candidate_id] = state.failures.get(candidate_id, 0) + 1
            if self._active_plan is not None:
                self._staged = [item for item in self._staged if item[3] != candidate_id]
            raise
        if self._active_plan is not None:
            if candidate_id == state.base_id:
                raise ValueError("cannot write the base during a rollout")
            self._staged.append((candidate_id, layer_idx, dict(activations), candidate_id))
            return
        try:
            self.put(candidate_id, layer_idx, activations, 0)
        except Exception:
            state.failures[candidate_id] = state.failures.get(candidate_id, 0) + 1
            raise

    def _restore_candidate(self, state: _RolloutState, candidate: str, bump: bool) -> None:
        if candidate not in state.candidates:
            raise ValueError(f"{candidate!r} is not an active candidate")
        current = self._table._branches.get(candidate, {})
        for layer, branch_state in current.items():
            for page_id in branch_state.page_ids.values():
                self._store.release(page_id)
        self._table.drop(candidate)
        baseline = state.baseline[state.base_id]
        layers: dict[int, _BranchLayer] = {
            layer: branch_state.clone() for layer, branch_state in baseline.items()
        }
        self._table._branches[candidate] = layers
        for branch_state in layers.values():
            for page_id in branch_state.page_ids.values():
                self._store.acquire(page_id)
        if bump:
            self._epoch += 1

    def rollback_candidate(self, rollout: RolloutSet, candidate_id: str) -> None:
        self._reject_in_window()
        state = self._rollout_state(rollout)
        self._restore_candidate(state, candidate_id, bump=True)

    def close_candidate(self, rollout: RolloutSet, candidate_id: str) -> None:
        self._reject_in_window()
        state = self._rollout_state(rollout)
        if candidate_id not in state.candidates:
            raise ValueError(f"{candidate_id!r} is not an active candidate")
        self._restore_candidate(state, candidate_id, bump=False)
        for layer, branch_state in list(self._table._branches[candidate_id].items()):
            for page_id in branch_state.page_ids.values():
                self._store.release(page_id)
        self._table.drop(candidate_id)
        state.candidates.discard(candidate_id)
        self._epoch += 1

    def close_rollout(self, rollout: RolloutSet) -> None:
        self._reject_in_window()
        state = self._rollout_state(rollout)
        for candidate in list(state.candidates):
            for layer, branch_state in list(self._table._branches.get(candidate, {}).items()):
                for page_id in branch_state.page_ids.values():
                    self._store.release(page_id)
            self._table.drop(candidate)
        state.candidates.clear()
        self._frozen_bases.discard(state.base_id)
        state.closed = True
        state.handle.closed = True
        self._rollouts.pop(id(rollout), None)
        self._epoch += 1

    def record_rollout_masks(
        self,
        rollout: RolloutSet,
        candidate_id: str,
        valid_mask: torch.Tensor,
        reused_mask: torch.Tensor,
    ) -> None:
        if self._active_plan is not None:
            raise RuntimeError("record_rollout_masks is only allowed outside a plan window")
        state = self._rollout_state(rollout)
        if candidate_id not in state.candidates:
            raise ValueError(f"{candidate_id!r} is not an active candidate")
        if not isinstance(valid_mask, torch.Tensor) or not isinstance(reused_mask, torch.Tensor):
            raise ValueError("masks must be torch tensors")
        if valid_mask.shape != reused_mask.shape:
            raise ValueError("valid_mask and reused_mask shapes differ")
        if bool((reused_mask.to(torch.bool) & ~valid_mask.to(torch.bool)).any()):
            raise ValueError("reused_mask must be a subset of valid_mask")
        state.valid_tokens[candidate_id] += int(valid_mask.to(torch.bool).sum())
        state.reused_tokens[candidate_id] += int(reused_mask.to(torch.bool).sum())

    def rollout_stats(self, rollout: RolloutSet) -> dict[str, dict[str, float | int]]:
        state = self._rollout_state(rollout)
        out: dict[str, dict[str, float | int]] = {}
        for candidate in state.candidate_order:
            if candidate not in state.candidates:
                continue
            hot = cold = shared = 0
            seen: set[int] = set()
            for layer_state in self._table._branches.get(candidate, {}).values():
                for page_id in layer_state.page_ids.values():
                    if page_id in seen:
                        continue
                    seen.add(page_id)
                    page = self._store.page(page_id)
                    if page.tier == "hot":
                        hot += page.nbytes
                    else:
                        cold += page.nbytes
                    if page.refcount >= 2:
                        shared += page.nbytes
            valid = state.valid_tokens.get(candidate, 0)
            reused = state.reused_tokens.get(candidate, 0)
            ratio = reused / max(valid, 1)
            out[candidate] = {
                "hot_bytes": hot,
                "cold_bytes": cold,
                "shared_bytes": shared,
                "reused_tokens": reused,
                "valid_tokens": valid,
                "reuse_ratio": ratio,
                "recompute_ratio": 1.0 - ratio,
                "failures": state.failures.get(candidate, 0),
            }
        return out

    def sync_branch_tree(self, tree: BranchTree) -> None:
        seen: set[str] = set()
        stack: list[Any] = [tree.root_id]
        while stack:
            branch_id = stack.pop()
            node = tree.get(branch_id)
            if node is None:
                raise ValueError(f"tree node {branch_id!r} cannot be resolved")
            key = str(branch_id)
            if key in seen:
                raise ValueError(f"cycle or duplicate id in branch tree: {key}")
            seen.add(key)
            if key not in self._table.branches():
                raise ValueError(f"tree branch {key!r} is missing from the cache")
            stack.extend(node.children)
        cache_branches = set(self._table.branches())
        if cache_branches != seen:
            raise ValueError(
                f"cache branches {sorted(cache_branches)} do not match tree {sorted(seen)}"
            )

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, int]:
        store = self._store.stats()
        return {
            "hot_pages": store["hot_pages"],
            "cold_pages": store["cold_pages"],
            "shared_pages": store["shared_pages"],
            "hot_bytes": store["hot_bytes"],
            "cold_bytes": store["cold_bytes"],
            "evictions": self._evictions,
            "demotions": self._demotions,
            "promotions": self._promotions,
            "refcount_leaks": self._refcount_leaks(),
            "requests": self._requests,
            "missing_tokens": self._missing_tokens,
        }

    def page_stats(self) -> dict[str, int]:
        store = self._store.stats()
        return {
            "branches": len(self._table.branches()),
            "physical_pages": store["physical_pages"],
            "shared_pages": store["shared_pages"],
            "hot_pages": store["hot_pages"],
            "cold_pages": store["cold_pages"],
            "hot_bytes": store["hot_bytes"],
            "cold_bytes": store["cold_bytes"],
            "metadata_entries": sum(len(state.page_ids) for _, _, state in self._table.entries()),
            "refcount_leaks": self._refcount_leaks(),
        }

    def _refcount_leaks(self) -> int:
        counts: dict[int, int] = {}
        for _, _, state in self._table.entries():
            for page_id in state.page_ids.values():
                counts[page_id] = counts.get(page_id, 0) + 1
        leaks = 0
        for page in self._store.iter_pages():
            expected = counts.get(page.page_id, 0)
            if expected == 0 or page.refcount != expected:
                leaks += 1
        for page_id in counts:
            if page_id not in self._store.snapshot_pages():
                leaks += 1
        return leaks

    def eviction_trace(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._trace]

    def assert_invariants(self) -> None:
        self._store.assert_invariants()
        self._table.assert_invariants()
        if self._refcount_leaks() != 0:
            raise AssertionError("refcount ledger has leaks")
        for branch, layer, state in self._table.entries():
            for logical, page_id in state.page_ids.items():
                page = self._store.page(page_id)
                if page.layer != layer:
                    raise AssertionError("page layer mismatch")
                if state.page_lens[logical] != page.page_len:
                    raise AssertionError("page length mismatch")
            if state.seq_len != sum(state.page_lens.values()):
                raise AssertionError("seq_len mismatch")
