"""AR030 mechanism benchmark: paged fold cache for agent workloads.

This script measures real tensor computation only: a small decoder stack built
from real ``nn.Linear``/softmax attention/FFN layers, driven through
``ManualFoldedForward`` with the AR030 ``PagedFoldCache``.  The agent pattern
under measurement is the N-candidate rollout: one shared root branch, eight
forked candidates whose token suffix diverges, COW page sharing on the
unchanged prefix, and a hard per-layer page budget.

It is a mechanism/infrastructure benchmark (``evidence_level="mechanism"``,
``baseline_claim=false``); the real-checkpoint A/B protocol lives in
``docs/RERUN_CHECKLIST.md`` as BLOCKED-DEFERRED.

Frozen geometry (SRS): ``T_base=128``, ``page_tokens=16``, ``depth=2``,
``hidden=32``, ``candidates=8``, each candidate diverges its last 8 tokens,
``paged_total_tokens=512`` (uniform: 256 tokens/layer = 16 pages/layer —
the root's 8 pages plus 8 candidates' 1 new page per layer land exactly at
the budget boundary with zero evictions), ``max_ratio=min_ratio=1.0``.

Frozen verdicts (declared before data collection):
    E1 exactness (bit-exact vs a dense ActivationCache oracle on both faces),
    E2 memory bound (per-layer hot pages within budget;
       ``paged_hot_bytes <= plan_expected_bytes``; ``paged_hot_bytes`` <= the
       dense oracle's independently recomputed resident bytes; the per-layer
       page ledger equals the store total; ``hot+cold`` bytes equal the
       independent per-page byte sum; ``sharing_ratio >= 0.5`` against the
       dense oracle's per-entry page recomputation),
    E3 fork/COW semantics (fork adds zero bytes; each candidate's new
       physical-page SET equals its content-differing-page SET, per layer and
       logical position; shared pages bit-equal; per-candidate metadata
       entries equal the per-layer page-count sum),
    E4 honesty (injected commit failures roll back atomically; stale plans
       are rejected; demote/promote round-trips bit-exactly with availability
       flips; snapshot/restore round-trips bit-exactly with tier/availability
       equality; the frozen plan's availability equals the executor's live
       availability; window reads see the frozen plan view; COW-off skips
       comparison and in-place mutation; refcount corruption is detected;
       rollout window validation failures are counted and discard only the
       failing candidate's staged writes; rollout rollback restores the fork
       baseline; all stats finite and non-negative).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from actfold.core import ActivationCache, PlanStaleError, make_activation_cache
from actfold.core.paged_fold_cache import PagedFoldCache
from actfold.core.pyramid_plan import build_pyramid_plan
from actfold.core.similarity_gate import SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward

SCHEMA = "ar030_paged_agent_bench"
SCHEMA_VERSION = 1
QUICK_PATH = Path("results/optimization/ar030_paged_agent_bench_quick.json")
FORMAL_PATH = Path("results/optimization/ar030_paged_agent_bench.json")
_REPO_ROOT = Path(__file__).resolve().parents[1]

HIDDEN = 32
VOCAB = 64
DEPTH = 2
SEQ = 128
PAGE = 16
CANDIDATES = 8
FORCED_TOKENS = 8
TAU = 0.9
PRESSURE_TOTAL = 512  # SRS frozen: 256 tokens/layer = 16 pages/layer (at-boundary)
FULL_TOTAL = 8192  # uniform: 4096 tokens/layer = 256 pages/layer
MIN_SHARING_RATIO = 0.5
QUICK_TIME_BUDGET_S = 60.0

HONESTY_KEYS = (
    "commit_fail_k0_atomic",
    "commit_fail_k1_atomic",
    "stale_plan_rejected",
    "demote_promote_roundtrip_bit_exact",
    "snapshot_restore_bit_exact",
    "snapshot_restore_tier_availability_equal",
    "plan_availability_matches_executor",
    "window_reads_match_plan_view",
    "cow_off_skips_comparison_and_inplace",
    "refcount_corruption_detected",
    "rollout_window_failure_discards_staged",
    "rollout_rollback_restores_baseline",
    "rollout_close_unfreezes_base",
    "refcount_ledger_clean",
)


class _Block(nn.Module):
    """A real decoder block: causal-free attention + FFN with residual norms."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(h), dim=-1)
        x = x + self.proj(attn @ v)
        return x + self.ff(self.norm2(x))  # type: ignore[no-any-return]


class TinyDecoder(nn.Module):
    """Real 2-layer decoder used by the mechanism benchmark."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type(
            "Config",
            (),
            {"model_type": "llama", "vocab_size": VOCAB, "num_hidden_layers": DEPTH},
        )()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.model.layers = nn.ModuleList([_Block(HIDDEN) for _ in range(DEPTH)])
        self.model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.model.lm_head(x)  # type: ignore[no-any-return]


def _paged_cache(total_tokens: int, enable_cow: bool = True) -> PagedFoldCache:
    if enable_cow:
        cache = make_activation_cache(
            device="cpu",
            use_paged=True,
            paged_depth=DEPTH,
            paged_page_tokens=PAGE,
            paged_total_tokens=total_tokens,
            paged_max_ratio=1.0,
            paged_min_ratio=1.0,
        )
        assert isinstance(cache, PagedFoldCache)
        return cache
    # COW disabled (design 4.2.2(6)): no content comparison, no in-place
    # mutation, a fresh page per write. Same plan geometry as the factory face.
    plan = build_pyramid_plan(
        depth=DEPTH,
        total_tokens=total_tokens,
        block_size=PAGE,
        max_ratio=1.0,
        min_ratio=1.0,
    )
    return PagedFoldCache(plan=plan, page_tokens=PAGE, device="cpu", enable_cow=False)


def _tokens(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, SEQ), generator=gen)


def _candidate(root: torch.Tensor, j: int) -> torch.Tensor:
    child = root.clone()
    child[:, -FORCED_TOKENS:] = (child[:, -FORCED_TOKENS:] + j + 1) % VOCAB
    return child


def _forced_mask() -> torch.Tensor:
    mask = torch.zeros(1, SEQ, dtype=torch.bool)
    mask[:, -FORCED_TOKENS:] = True
    return mask


# ---------------------------------------------------------------------------
# Frozen verdicts
# ---------------------------------------------------------------------------


def verdict_e1(full_diffs: list[float], pressure_diffs: list[float]) -> bool:
    """E1: bit-exact against the dense oracle on both faces."""
    return all(d == 0.0 for d in full_diffs + pressure_diffs)


def verdict_e2(
    physical_per_layer: dict[int, int],
    budget_pages: dict[int, int],
    paged_hot_bytes: int,
    paged_total_bytes: int,
    plan_expected_bytes: int,
    independent_page_bytes: int,
    nonshared_hot_bytes: int,
    nonshared_pages: int,
    physical_pages: int,
) -> bool:
    """E2 (SRS): budget, plan-byte and dense-footprint bounds, ledger equality.

    Conditions (all frozen):
      1. per-layer physical pages within the per-layer budget,
      2. the per-layer ledger sums to the store's physical-page total,
      3. ``hot+cold`` bytes equal the independent per-page byte sum,
      4. ``paged_hot_bytes <= plan_expected_bytes``,
      5. ``paged_hot_bytes <= nonshared_hot_bytes`` (dense oracle recompute),
      6. ``sharing_ratio = 1 - physical/nonshared >= 0.5``.
    """
    within_budget = all(
        physical_per_layer.get(layer, 0) <= budget_pages[layer] for layer in budget_pages
    )
    ledger_pages = sum(physical_per_layer.values()) == physical_pages
    ledger_bytes = paged_total_bytes == independent_page_bytes
    plan_bound = paged_hot_bytes <= plan_expected_bytes
    dense_bound = paged_hot_bytes <= nonshared_hot_bytes
    page_sharing = physical_pages <= MIN_SHARING_RATIO * nonshared_pages
    return (
        within_budget
        and ledger_pages
        and ledger_bytes
        and plan_bound
        and dense_bound
        and page_sharing
    )


def verdict_e3(
    fork_byte_deltas: list[int],
    new_coords: list[set[tuple[int, int]]],
    diff_coords: list[set[tuple[int, int]]],
    shared_ok: list[bool],
    meta_entries_ok: list[bool],
) -> bool:
    """E3: fork is byte-free; new-page SET == content-diff SET, per candidate."""
    return (
        all(d == 0 for d in fork_byte_deltas)
        and all(new == diff for new, diff in zip(new_coords, diff_coords))
        and all(len(new) >= 1 for new in new_coords)
        and all(shared_ok)
        and all(meta_entries_ok)
    )


def verdict_e4(checks: dict[str, bool], stat_values: list[float]) -> bool:
    """E4: every honesty check holds and every stat is finite/non-negative."""
    return all(checks.values()) and all(math.isfinite(v) and v >= 0 for v in stat_values)


def overall_verdict(e1: bool, e2: bool, e3: bool, e4: bool) -> str:
    """Frozen precedence: blocked (honesty) > supported > refuted."""
    if not e4:
        return "blocked"
    if e1 and e2 and e3:
        return "supported"
    return "refuted"


# ---------------------------------------------------------------------------
# Measurement faces
# ---------------------------------------------------------------------------


def _run_agent_face(model: TinyDecoder, cache: Any) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Root forward + 8 forked candidate forwards through the paged cache."""
    root = _tokens(0)
    forced = _forced_mask()
    mff = ManualFoldedForward(model, cache=cache, gate=SimilarityGate(tau=TAU))
    outputs: list[torch.Tensor] = []
    fork_byte_deltas: list[int] = []
    new_coords: list[set[tuple[int, int]]] = []
    diff_coords: list[set[tuple[int, int]]] = []
    shared_ok: list[bool] = []
    meta_entries_ok: list[bool] = []

    with torch.no_grad():
        mff(root, branch_id="root")
        root_ids = {
            layer: dict(cache._table._branches["root"][layer].page_ids) for layer in range(DEPTH)
        }
        root_pages = {layer: cache.fetch("root", layer)["ffn_out"] for layer in range(DEPTH)}
        for j in range(CANDIDATES):
            before = cache.page_stats()
            cache.fork_branch("root", f"cand{j}")
            after_fork = cache.page_stats()
            fork_byte_deltas.append(
                int(after_fork["hot_bytes"] + after_fork["cold_bytes"])
                - int(before["hot_bytes"] + before["cold_bytes"])
            )
            out = mff(
                _candidate(root, j),
                branch_id=f"cand{j}",
                parent_branch_id="root",
                forced_divergent=forced,
            )
            outputs.append(out)
            # Content ground truth, computed independently of the page ledger:
            # which (layer, logical) pages differ from the root's pages, and
            # which physical pages the candidate no longer shares with root.
            cand_new: set[tuple[int, int]] = set()
            cand_diff: set[tuple[int, int]] = set()
            shared_pages = 0
            meta_entries = 0
            for layer in range(DEPTH):
                cand_state = cache._table._branches[f"cand{j}"][layer]
                meta_entries += len(cand_state.page_ids)
                cand_t = cache.fetch(f"cand{j}", layer)["ffn_out"]
                root_t = root_pages[layer]
                for logical in range(SEQ // PAGE):
                    start = logical * PAGE
                    if torch.equal(
                        root_t[:, start : start + PAGE], cand_t[:, start : start + PAGE]
                    ):
                        shared_pages += 1
                    else:
                        cand_diff.add((layer, logical))
                    if cand_state.page_ids[logical] != root_ids[layer][logical]:
                        cand_new.add((layer, logical))
            total_pages = (SEQ // PAGE) * DEPTH
            new_coords.append(cand_new)
            diff_coords.append(cand_diff)
            shared_ok.append(shared_pages >= MIN_SHARING_RATIO * total_pages)
            meta_entries_ok.append(meta_entries == total_pages)
    return outputs, {
        "fork_byte_deltas": fork_byte_deltas,
        "new_coords": new_coords,
        "diff_coords": diff_coords,
        "shared_ok": shared_ok,
        "meta_entries_ok": meta_entries_ok,
    }


def _run_dense_face(model: TinyDecoder) -> tuple[list[torch.Tensor], int, int]:
    """Same computation through a plain dense ActivationCache (the oracle).

    Returns the outputs, the oracle's resident byte total, and the oracle's
    per-entry nonshared page count (SRS: ``branches x sum_layer
    ceil(seq_len/page_tokens)``, recomputed from actual entries — not from the
    geometric formula).
    """
    root = _tokens(0)
    forced = _forced_mask()
    cache = ActivationCache(max_entries_per_layer=SEQ * 64)
    mff = ManualFoldedForward(model, cache=cache, gate=SimilarityGate(tau=TAU))
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        mff(root, branch_id="root")
        for j in range(CANDIDATES):
            outputs.append(
                mff(
                    _candidate(root, j),
                    branch_id=f"cand{j}",
                    parent_branch_id="root",
                    forced_divergent=forced,
                )
            )
    byte_total = 0
    page_total = 0
    for layer in cache._layers.values():  # reference accounting only
        for entry in layer.values():
            byte_total += sum(t.numel() * t.element_size() for t in entry.values())
            # ActivationCache's internal layout is [seq_len, batch, hidden].
            any_tensor = next(iter(entry.values()))
            page_total += int(math.ceil(int(any_tensor.shape[0]) / PAGE))
    return outputs, byte_total, page_total


def _honesty_face(model: TinyDecoder) -> dict[str, bool]:
    """E4: atomicity, staleness, tiering, snapshot, plan view, COW-off, rollouts."""
    checks: dict[str, bool] = {}
    cache = _paged_cache(FULL_TOTAL)

    def _acts(seed: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        return {"ffn_out": torch.randn(1, PAGE, HIDDEN, generator=gen)}

    # Atomic staged-commit failure (k=0: fail before any write).
    cache.put("base", 0, _acts(1))
    plan = cache.build_plan(["base"])
    before = cache.page_stats()
    before_epoch = cache.epoch
    cache._fail_after_staged = 0
    try:
        with cache.enter_plan(plan):
            cache.put("base", 0, _acts(2))
        checks["commit_fail_k0_atomic"] = False
    except RuntimeError:
        checks["commit_fail_k0_atomic"] = (
            cache.page_stats() == before and cache.epoch == before_epoch
        )
    cache.assert_invariants()

    # Atomic staged-commit failure (k=1: fail after one applied write).
    plan = cache.build_plan(["base"])
    before = cache.page_stats()
    cache._fail_after_staged = 1
    try:
        with cache.enter_plan(plan):
            cache.put("base", 0, _acts(3))
            cache.put("base", 1, _acts(4))
        checks["commit_fail_k1_atomic"] = False
    except RuntimeError:
        checks["commit_fail_k1_atomic"] = cache.page_stats() == before
    cache.assert_invariants()

    # Stale plan rejection.
    plan = cache.build_plan(["base"])
    cache.put("base", 0, _acts(5))  # bumps the epoch: plan is now stale
    try:
        with cache.enter_plan(plan):
            pass
        checks["stale_plan_rejected"] = False
    except PlanStaleError:
        checks["stale_plan_rejected"] = True

    # Demote/promote round trip: availability flips, content bit-exact.
    tp = _paged_cache(FULL_TOTAL)
    tp.put("base", 0, _acts(20))
    tp_pid = tp._table._branches["base"][0].page_ids[0]
    original = tp.fetch("base", 0)["ffn_out"].clone()
    tp.demote(tp_pid)
    cold_avail = tp.availability("base", 0, PAGE)
    tp.promote(tp_pid)
    checks["demote_promote_roundtrip_bit_exact"] = (
        bool((~cold_avail).all())
        and bool(tp.availability("base", 0, PAGE).all())
        and torch.equal(tp.fetch("base", 0)["ffn_out"], original)
    )
    tp.assert_invariants()

    # Snapshot / restore round trip on a populated agent-pattern cache.
    pop = _paged_cache(FULL_TOTAL)
    root = _tokens(0)
    mff = ManualFoldedForward(model, cache=pop, gate=SimilarityGate(tau=TAU))
    with torch.no_grad():
        mff(root, branch_id="root")
        for j in range(CANDIDATES):
            pop.fork_branch("root", f"cand{j}")
            mff(
                _candidate(root, j),
                branch_id=f"cand{j}",
                parent_branch_id="root",
                forced_divergent=_forced_mask(),
            )
    snap = pop.snapshot()
    target = _paged_cache(FULL_TOTAL)
    target.restore(snap)
    same = True
    for branch in ["root"] + [f"cand{j}" for j in range(CANDIDATES)]:
        for layer in range(DEPTH):
            same = same and torch.equal(
                pop.fetch(branch, layer)["ffn_out"], target.fetch(branch, layer)["ffn_out"]
            )
    checks["snapshot_restore_bit_exact"] = same and target.epoch == snap["epoch"] + 1
    tiers_same = all(
        pop._store.page(page.page_id).tier == target._store.page(page.page_id).tier
        for page in pop._store.iter_pages()
    )
    avail_same = all(
        torch.equal(
            pop.availability(branch, layer, SEQ),
            target.availability(branch, layer, SEQ),
        )
        for branch in ["root"] + [f"cand{j}" for j in range(CANDIDATES)]
        for layer in range(DEPTH)
    )
    checks["snapshot_restore_tier_availability_equal"] = tiers_same and avail_same
    target.assert_invariants()

    # Frozen plan availability vs the executor's live availability.
    pop_plan = pop.build_plan(["root"] + [f"cand{j}" for j in range(CANDIDATES)])
    plan_ok = pop_plan.availability_for("root", 0) is not None
    for branch in pop_plan.branches:
        for layer in range(DEPTH):
            decoded = pop_plan.availability_for(branch, layer)
            live = pop.availability(branch, layer, SEQ)
            if decoded is None:
                plan_ok = False
                continue
            plan_ok = plan_ok and list(decoded) == [bool(v) for v in live.tolist()]
    checks["plan_availability_matches_executor"] = plan_ok

    # Window read path: reads inside enter_plan see the frozen plan view;
    # staged writes do not disturb reads and commit atomically on exit.
    wr = _paged_cache(FULL_TOTAL)
    wr.put("base", 0, _acts(30))
    before_fetch = wr.fetch("base", 0)["ffn_out"].clone()
    wr_plan = wr.build_plan(["base"])
    wr_decoded = wr_plan.availability_for("base", 0)
    with wr.enter_plan(wr_plan):
        inside_ok = torch.equal(before_fetch, wr.fetch("base", 0)["ffn_out"])
        inside_avail = wr.availability("base", 0, PAGE)
        if wr_decoded is None:
            inside_ok = False
        else:
            inside_ok = inside_ok and list(wr_decoded) == [bool(v) for v in inside_avail.tolist()]
        wr.put("base", 0, _acts(31))  # staged: must not disturb reads
        staged_ok = torch.equal(before_fetch, wr.fetch("base", 0)["ffn_out"])
    committed = wr.fetch("base", 0)["ffn_out"]
    checks["window_reads_match_plan_view"] = (
        inside_ok and staged_ok and not torch.equal(before_fetch, committed)
    )
    wr.assert_invariants()

    # COW-off contract: identical AND divergent re-writes both allocate fresh
    # pages (no comparison, no in-place mutation); content stays correct.
    co = _paged_cache(FULL_TOTAL, enable_cow=False)
    co.put("base", 0, _acts(40))
    co_pid0 = co._table._branches["base"][0].page_ids[0]
    co.put("base", 0, _acts(40))  # identical content
    co_pid1 = co._table._branches["base"][0].page_ids[0]
    co.put("base", 0, _acts(41))  # divergent content
    co_pid2 = co._table._branches["base"][0].page_ids[0]
    checks["cow_off_skips_comparison_and_inplace"] = (
        co_pid1 != co_pid0
        and co_pid2 not in (co_pid0, co_pid1)
        and torch.equal(co.fetch("base", 0)["ffn_out"], _acts(41)["ffn_out"])
    )
    co.assert_invariants()

    # Refcount corruption is detected by the leak ledger.
    lk = _paged_cache(FULL_TOTAL)
    lk.put("base", 0, _acts(50))
    lk_pid = lk._table._branches["base"][0].page_ids[0]
    lk._store.page(lk_pid).refcount += 1
    checks["refcount_corruption_detected"] = lk.stats()["refcount_leaks"] > 0

    # Rollout window: a validation failure counts into the candidate's failure
    # ledger and discards ONLY that candidate's staged writes.
    rw = _paged_cache(FULL_TOTAL)
    rw.put("base", 0, _acts(60))
    handle = rw.fork_rollouts("base", ["w_a", "w_b"])
    rw_plan = rw.build_plan(["base", "w_a", "w_b"])
    with rw.enter_plan(rw_plan):
        rw.rollout_put(handle, "w_a", 0, _acts(61))  # staged, then discarded
        try:
            rw.rollout_put(handle, "w_a", 0, {"ffn_out": torch.zeros(4)})
            failure_counted = False
        except ValueError:
            failure_counted = True
        rw.rollout_put(handle, "w_b", 0, _acts(62))  # keeps its staged write
    rw_stats = rw.rollout_stats(handle)
    checks["rollout_window_failure_discards_staged"] = (
        failure_counted
        and rw_stats["w_a"]["failures"] == 1
        and rw_stats["w_b"]["failures"] == 0
        and torch.equal(rw.fetch("w_a", 0)["ffn_out"], rw.fetch("base", 0)["ffn_out"])
        and not torch.equal(rw.fetch("w_b", 0)["ffn_out"], rw.fetch("base", 0)["ffn_out"])
    )
    rw.close_rollout(handle)
    rw.assert_invariants()

    # Rollout: fork candidates, write one, roll it back to the fork baseline.
    rc = _paged_cache(FULL_TOTAL)
    rc.put("base", 0, _acts(10))
    rc.put("base", 1, _acts(11))
    rc_handle = rc.fork_rollouts("base", ["r_a", "r_b"])
    rc.rollout_put(rc_handle, "r_a", 0, _acts(12))
    rc.rollback_candidate(rc_handle, "r_a")
    checks["rollout_rollback_restores_baseline"] = all(
        torch.equal(rc.fetch("r_a", layer)["ffn_out"], rc.fetch("base", layer)["ffn_out"])
        for layer in range(DEPTH)
    )
    rc.close_rollout(rc_handle)
    try:
        rc.put("base", 0, _acts(13))  # base is writable again after close
        checks["rollout_close_unfreezes_base"] = True
    except ValueError:
        checks["rollout_close_unfreezes_base"] = False
    rc.assert_invariants()

    # Refcount ledger sanity on every uncorrupted cache built in this face.
    checks["refcount_ledger_clean"] = (
        cache.stats()["refcount_leaks"] == 0
        and tp.stats()["refcount_leaks"] == 0
        and pop.stats()["refcount_leaks"] == 0
        and target.stats()["refcount_leaks"] == 0
        and wr.stats()["refcount_leaks"] == 0
        and co.stats()["refcount_leaks"] == 0
        and rw.stats()["refcount_leaks"] == 0
        and rc.stats()["refcount_leaks"] == 0
    )
    return checks


def run(quick: bool) -> dict[str, Any]:
    """Run the frozen mechanism benchmark and return the artifact payload."""
    started = time.perf_counter()
    torch.manual_seed(1234)
    model = TinyDecoder().eval()

    dense_out, dense_bytes, dense_pages = _run_dense_face(model)

    full_cache = _paged_cache(FULL_TOTAL)
    full_out, full_aux = _run_agent_face(model, full_cache)
    pressure_started = time.perf_counter()
    pressure_cache = _paged_cache(PRESSURE_TOTAL)
    pressure_out, pressure_aux = _run_agent_face(model, pressure_cache)
    pressure_ms = (time.perf_counter() - pressure_started) * 1000

    full_diffs = [
        float((a - b).abs().max().item()) if a.shape == b.shape else float("inf")
        for a, b in zip(full_out, dense_out)
    ]
    pressure_diffs = [
        float((a - b).abs().max().item()) if a.shape == b.shape else float("inf")
        for a, b in zip(pressure_out, dense_out)
    ]

    ps = pressure_cache.page_stats()
    physical_pages = int(ps["physical_pages"])
    paged_hot_bytes = int(ps["hot_bytes"])
    paged_total_bytes = int(ps["hot_bytes"] + ps["cold_bytes"])
    independent_page_bytes = sum(int(page.nbytes) for page in pressure_cache._store.iter_pages())
    naive_pages = (CANDIDATES + 1) * (SEQ // PAGE) * DEPTH
    # SRS: the nonshared footprint is independently recomputed from the dense
    # oracle's actual entries; the geometric formula is only a cross-check.
    nonshared_pages = dense_pages
    nonshared_hot_bytes = dense_bytes
    assert nonshared_pages == naive_pages, (
        f"dense oracle page recomputation ({nonshared_pages}) diverged from the "
        f"geometric formula ({naive_pages}): measurement setup is broken"
    )
    budget_pages = {layer: PRESSURE_TOTAL // DEPTH // PAGE for layer in range(DEPTH)}
    physical_per_layer: dict[int, int] = {layer: 0 for layer in range(DEPTH)}
    for page in pressure_cache._store.iter_pages():
        physical_per_layer[page.layer] = physical_per_layer.get(page.layer, 0) + 1
    plan = pressure_cache.build_plan(["root"] + [f"cand{j}" for j in range(CANDIDATES)])
    plan_expected_bytes = int(plan.expected_bytes)

    honesty = _honesty_face(model)

    e1 = verdict_e1(full_diffs, pressure_diffs)
    e2 = verdict_e2(
        physical_per_layer,
        budget_pages,
        paged_hot_bytes,
        paged_total_bytes,
        plan_expected_bytes,
        independent_page_bytes,
        nonshared_hot_bytes,
        nonshared_pages,
        physical_pages,
    )
    e3 = verdict_e3(
        pressure_aux["fork_byte_deltas"],
        pressure_aux["new_coords"],
        pressure_aux["diff_coords"],
        pressure_aux["shared_ok"],
        pressure_aux["meta_entries_ok"],
    )
    pressure_stats = pressure_cache.stats()
    full_stats = full_cache.stats()
    stat_values = [float(v) for v in pressure_stats.values()] + [
        float(v) for v in full_stats.values()
    ]
    outputs_finite = all(
        torch.isfinite(x).all().item() for x in full_out + pressure_out + dense_out
    )
    e4 = verdict_e4(honesty, stat_values) and outputs_finite
    overall = overall_verdict(e1, e2, e3, e4)
    duration_s = time.perf_counter() - started

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evidence_level": "mechanism",
        "baseline_claim": False,
        "status": "ok" if overall != "blocked" else "blocked",
        "mode": "quick" if quick else "formal",
        "geometry": {
            "hidden": HIDDEN,
            "vocab": VOCAB,
            "depth": DEPTH,
            "seq": SEQ,
            "page_tokens": PAGE,
            "candidates": CANDIDATES,
            "forced_tokens": FORCED_TOKENS,
            "tau": TAU,
            "pressure_total_tokens": PRESSURE_TOTAL,
            "full_total_tokens": FULL_TOTAL,
            "budget_pages_per_layer": PRESSURE_TOTAL // DEPTH // PAGE,
        },
        "metrics": {
            "full": {
                "max_abs_diff_vs_dense": max(full_diffs),
                "physical_pages": int(full_cache.page_stats()["physical_pages"]),
                "paged_bytes": int(
                    full_cache.page_stats()["hot_bytes"] + full_cache.page_stats()["cold_bytes"]
                ),
                "new_pages_per_candidate": [len(s) for s in full_aux["new_coords"]],
                "fork_byte_deltas": full_aux["fork_byte_deltas"],
            },
            "pressure": {
                "wall_ms": round(pressure_ms, 3),
                "max_abs_diff_vs_dense": max(pressure_diffs),
                "physical_pages": physical_pages,
                "physical_pages_per_layer": physical_per_layer,
                "budget_pages_per_layer": budget_pages,
                "paged_hot_bytes": paged_hot_bytes,
                "paged_total_bytes": paged_total_bytes,
                "plan_expected_bytes": plan_expected_bytes,
                "independent_page_bytes": independent_page_bytes,
                "nonshared_hot_bytes": int(nonshared_hot_bytes),
                "nonshared_pages": nonshared_pages,
                "byte_sharing_ratio": round(1 - paged_total_bytes / max(nonshared_hot_bytes, 1), 6),
                "page_sharing_ratio": round(1 - physical_pages / max(nonshared_pages, 1), 6),
                "new_pages_per_candidate": [len(s) for s in pressure_aux["new_coords"]],
                "diff_pages_per_candidate": [len(s) for s in pressure_aux["diff_coords"]],
                "fork_byte_deltas": pressure_aux["fork_byte_deltas"],
                "evictions": int(pressure_stats["evictions"]),
            },
            "dense_reference": {
                "cache_bytes": int(dense_bytes),
                "page_equivalents": dense_pages,
            },
        },
        "honesty_checks": honesty,
        "verdicts": {"E1": e1, "E2": e2, "E3": e3, "E4": e4},
        "overall": overall,
        "duration_s": round(duration_s, 3),
        "meta": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "note": (
                "mechanism-level evidence; real-checkpoint A/B is BLOCKED-DEFERRED "
                "and must not be inferred from this artifact"
            ),
        },
    }
    if quick and duration_s >= QUICK_TIME_BUDGET_S:
        raise RuntimeError(
            f"quick face exceeded its {QUICK_TIME_BUDGET_S}s wall-clock budget "
            f"({duration_s:.1f}s)"
        )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AR030 mechanism benchmark")
    parser.add_argument("--quick", action="store_true", help="write the quick artifact")
    args = parser.parse_args(argv)
    if not args.quick:
        print(
            "formal face is BLOCKED-DEFERRED (user-authorized no-large-experiments "
            "scope); run with --quick for the mechanism artifact"
        )
        return 3
    artifact = run(quick=True)
    out = _REPO_ROOT / QUICK_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {out} overall={artifact['overall']}")
    return 0 if artifact["overall"] in ("supported", "blocked") else 1


if __name__ == "__main__":
    sys.exit(main())
