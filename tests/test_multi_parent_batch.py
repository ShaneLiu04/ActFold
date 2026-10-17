"""AR025 Red tests: multi-parent batch folding, UT-2400..UT-2427.

New tests for the per-row parent-assembly batched forward
(specs/changes/AR025-multi-parent-batch-folding, design rev 3):

- UT-2400: frozen sanity -- the per-parent reference helper (current
  API) matches per-candidate serial calls (green anchor for the Red
  file's baselines).
- UT-2401: API contract -- ``parent_branch_ids`` mutual exclusion with
  ``parent_branch_id``, plural-parent without ``branch_ids``, and
  length mismatch all raise ``ValueError``.
- UT-2402: ``parent_branch_ids=None`` attaches nothing -- byte-equal to
  the call that omits the kwarg entirely.
- UT-2403: a homogeneous repeated parent id list equals the single
  ``parent_branch_id`` expand path bit-for-bit.
- UT-2404: all-distinct-parent parity -- one multi-parent ``[N, T]``
  call equals per-parent batched calls in logits, per-branch cache
  entries, and profiler statistics.
- UT-2405: a missing parent row degrades to that row's full recompute
  while the present row folds normally.
- UT-2406: missing-row value no-leak is CONSTRUCTIVE (design DD2): with
  an ``l2``/``tau=0`` gate (where a zeros placeholder can score stable)
  the missing row still equals its own recompute, never another row's
  parent values.
- UT-2407: all parents missing -- whole-batch recompute, one
  ``_recompute_all`` per layer (spy), values equal the unparented
  reference.
- UT-2408: a parent cached at batch != 1 among good parents is a hard
  ``RuntimeError`` (per-row contract violation).
- UT-2409: a parent whose length differs from the child canvas (T_p !=
  T_c) degrades that row (v1 equal-length semantics -- no AR007
  truncation on the multi-parent face).
- UT-2410: the all-stable fast path stacks multi-parent FFN rows with
  zero recompute (spy) and reuses each parent's own values.
- UT-2411: a mixed batch's fully stable row merges to its parent values
  (bit-identical to the serial fast path for that row, the :617-620
  contract) while divergent rows match the per-parent reference.
- UT-2412: fully divergent rows equal the whole-batch child recompute.
- UT-2413: a hidden parent FFN entry (fetch KeyError at one layer)
  degrades the fast path to the unified slow path with correct values.
- UT-2414: merge-path placeholder rows never leak values (the missing
  row's merged output equals its own recompute).
- UT-2415: ``_merge_stable_divergent_torch`` zero-change regression on
  an ``[N, T]`` mask (green).
- UT-2416: switch-on publish equivalence -- ``put_batch`` and per-row
  ``put`` leave bit-identical cache terminal states.
- UT-2417: a cache without ``put_batch`` falls back to per-row ``put``
  with unchanged values (the #42 explicit-AttributeError wrapper).
- UT-2418: the write-op ledger of one multi-parent batched call is
  strictly smaller than per-parent batched calls (switch on vs off).
- UT-2419: single-parent batches are ``put_batch``-ized too when the
  switch is on (strictly smaller ledger than the off path).
- UT-2420: ``folded_generate_batched`` two-chain parity vs per-chain
  ``folded_generate`` (tokens / stable ratio / acceptance, winner
  continuation across steps).
- UT-2421: unequal-length chains fall back to per-chain generation with
  identical results.
- UT-2422: the default (off) switch falls back to per-chain generation
  with identical results.
- UT-2423: a single chain degenerates to the single-chain result.
- UT-2424: prune is per chain -- rejected candidates leave no cache
  residue in either chain.
- UT-2425: ``ActFoldConfig.multi_parent_batch`` defaults to ``False``.
- UT-2426: YAML consumption of ``multi_parent_batch`` raises no
  unknown-key ``UserWarning``.
- UT-2427: the dispatch predicate generalizes only under the switch --
  multi-parent candidates merge iff on; a single candidate never
  batches.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.fused_ops import _merge_stable_divergent_torch
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from actfold.speculative.branch_tree import BranchNode
from actfold.speculative.draft_generator import DraftGenerator
from actfold.speculative.folded_generation import folded_generate
from actfold.utils.config_manager import ActFoldConfig, load_config
from tests.test_batched_branch_forward import (
    _NUM_LAYERS,
    _VOCAB,
    _assert_tensor_close,
    _CausalBatchModel,
    _make_equal_len_candidates,
    _spy_recompute_all,
    _track_cached_branches,
)

_SEED = 2026


# ---------------------------------------------------------------------------
# Local builders (AR025): a custom-gate / extra-kwargs variant of the AR015
# _build_setup, a two-parent candidate world, and the multi-parent call
# helpers. The per-parent reference always uses only the current production
# API, so in the TDD Red state the failure is isolated to the multi-parent
# call (the parity discipline of _run_batched_vs_serial_parity).
# ---------------------------------------------------------------------------


def _build_setup_mp(
    device: str = "cpu",
    gate: SimilarityGate | None = None,
    mff_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, ManualFoldedForward, ActivationCache, SimilarityGate]:
    """Build a fresh causal model plus a manual folded stack (AR025 variant).

    Args:
        device: Target device string.
        gate: Optional custom gate (e.g. the l2/tau=0 leak probe of UT-2406).
        mff_kwargs: Extra ``ManualFoldedForward`` constructor kwargs (the
            ``multi_parent_batch`` switch face of T002/T004).

    Returns:
        ``(raw, folded, cache, gate)``.
    """
    torch.manual_seed(_SEED)
    raw = _CausalBatchModel(_VOCAB, 32, _NUM_LAYERS).to(device)
    raw.eval()
    cache = ActivationCache(max_entries_per_layer=256, device=device)
    if gate is None:
        gate = SimilarityGate(tau=0.95)
    folded = ManualFoldedForward(
        raw,
        cache=cache,
        gate=gate,
        split_layers=False,
        split_min_tokens=512,
        **(mff_kwargs or {}),
    )
    return raw, folded, cache, gate


def _make_two_parent_world(
    device: str,
    num_per: int = 2,
    seq_len: int = 8,
    seed: int = 4300,
) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
    """Build two distinct ``[1, seq_len]`` parents with mid-divergent children.

    Each parent gets ``num_per`` equal-length children that diverge at the
    mid-sequence position (mutually distinct replacement tokens), so the
    mixed merge path engages for both groups.
    """
    torch.manual_seed(seed)
    parents: list[torch.Tensor] = []
    groups: list[list[torch.Tensor]] = []
    mid = seq_len // 2
    for _ in range(2):
        parent = torch.randint(1, _VOCAB, (1, seq_len), device=device)
        base = int(parent[0, mid].item())
        group: list[torch.Tensor] = []
        for i in range(num_per):
            child = parent.clone()
            child[0, mid] = (base + 1 + i) % _VOCAB
            group.append(child)
        parents.append(parent)
        groups.append(group)
    return parents, groups


def _multi_call(
    folded: ManualFoldedForward,
    parent_ids: list[str],
    groups: list[list[torch.Tensor]],
    tag: str,
    step_idx: int = 1,
) -> tuple[torch.Tensor, list[str]]:
    """One multi-parent ``[N, T]`` batched call (the AR025 API face)."""
    stacked = torch.cat([child for group in groups for child in group], dim=0)
    branch_ids = [f"{tag}-r{i}" for i in range(stacked.shape[0])]
    parent_branch_ids = [pid for pid, group in zip(parent_ids, groups) for _ in group]
    logits = folded(
        stacked,
        branch_id=None,
        branch_ids=branch_ids,
        parent_branch_ids=parent_branch_ids,
        step_idx=step_idx,
    )
    return logits, branch_ids


def _per_parent_reference(
    folded: ManualFoldedForward,
    parent_ids: list[str],
    groups: list[list[torch.Tensor]],
    tag: str,
    parented: list[bool] | None = None,
) -> tuple[list[torch.Tensor], list[str]]:
    """Current-API baseline: one ``branch_ids`` batched call per parent group.

    ``parented[g] == False`` runs group ``g`` without a parent (whole-group
    recompute), covering missing / shape-mismatched parent rows.
    """
    if parented is None:
        parented = [True] * len(parent_ids)
    outs: list[torch.Tensor] = []
    ids: list[str] = []
    for g, (pid, group) in enumerate(zip(parent_ids, groups)):
        stacked = torch.cat(group, dim=0)
        bids = [f"{tag}-g{g}r{i}" for i in range(len(group))]
        out = folded(
            stacked,
            branch_id=None,
            branch_ids=bids,
            parent_branch_id=pid if parented[g] else None,
            step_idx=1,
        )
        outs.append(out)
        ids.extend(bids)
    return outs, ids


def _populate_parents(
    folded: ManualFoldedForward,
    parents: list[torch.Tensor],
    parent_ids: list[str],
) -> None:
    """Cache each parent's activations with a batch-1 serial forward."""
    for parent, pid in zip(parents, parent_ids):
        folded(parent, branch_id=pid)


def _atol(device: str) -> float | None:
    """Bit-exact on CPU (the #41 discipline); allclose(1e-6) on CUDA."""
    return None if device == "cpu" else 1e-6


def _assert_cache_entry_close(
    cache_a: ActivationCache,
    id_a: str,
    cache_b: ActivationCache,
    id_b: str,
    atol: float | None,
) -> None:
    """Compare per-layer cached activations of two matched branches."""
    for layer_idx in range(_NUM_LAYERS):
        entry_a = cache_a.fetch(branch_id=id_a, layer_idx=layer_idx)
        entry_b = cache_b.fetch(branch_id=id_b, layer_idx=layer_idx)
        _assert_tensor_close(entry_a["ffn_out"], entry_b["ffn_out"], atol)
        if layer_idx == 0:
            _assert_tensor_close(entry_a["embedding"], entry_b["embedding"], atol)


class _NoBatchPutCache:
    """Delegating cache WITHOUT ``put_batch`` (the #42 wrapper discipline).

    Every consumed method is explicit; there is no blanket ``__getattr__``
    (a blanket delegate would leak the inner cache's ``put_batch`` and both
    sides of an A/B would take the same branch). ``put_batch`` raises
    ``AttributeError`` explicitly.
    """

    def __init__(self, inner: ActivationCache) -> None:
        self._inner = inner

    def fetch(self, branch_id: str, layer_idx: int) -> dict[str, torch.Tensor]:
        return self._inner.fetch(branch_id=branch_id, layer_idx=layer_idx)

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        self._inner.put(
            branch_id=branch_id,
            layer_idx=layer_idx,
            activations=activations,
            step_idx=step_idx,
        )

    def contains(self, branch_id: str, layer_idx: int) -> bool:
        return self._inner.contains(branch_id, layer_idx)

    def clear_branch(self, branch_id: str) -> None:
        self._inner.clear_branch(branch_id)

    def put_batch(self, entries: Any) -> None:
        raise AttributeError("put_batch")


class _DropEntryCache(_NoBatchPutCache):
    """A delegating cache that hides ONE ``(branch, layer)`` fetch entry.

    ``fetch`` of the hidden pair raises ``KeyError`` -- the parent-FFN
    precondition breaker of UT-2413 (gate-layer fetches at other layers
    still succeed, isolating the fast-path FFN face).
    """

    def __init__(
        self,
        inner: ActivationCache,
        hidden_branch: str,
        hidden_layer: int,
    ) -> None:
        super().__init__(inner)
        self._hidden_branch = hidden_branch
        self._hidden_layer = hidden_layer

    def fetch(self, branch_id: str, layer_idx: int) -> dict[str, torch.Tensor]:
        if branch_id == self._hidden_branch and layer_idx == self._hidden_layer:
            raise KeyError(branch_id)
        return super().fetch(branch_id=branch_id, layer_idx=layer_idx)


def _import_batched_generation_api() -> tuple[Any, Any]:
    """Lazy import of the AR025 multi-chain generation API (Red-safe)."""
    from actfold.speculative.folded_generation import (  # noqa: F401
        ChainInput,
        folded_generate_batched,
    )

    return ChainInput, folded_generate_batched


# ---------------------------------------------------------------------------
# API face
# ---------------------------------------------------------------------------


def test_ut2400_per_parent_reference_matches_serial(device: str) -> None:
    """UT-2400 (frozen sanity, green): the reference helper is sound."""
    parent, candidates = _make_equal_len_candidates(device, num=4)
    _raw_a, folded_a, _cache_a, _gate = _build_setup_mp(device)
    _raw_s, folded_s, _cache_s, _gate = _build_setup_mp(device)
    folded_s(parent, branch_id="ps")
    serial = [
        folded_s(tokens, branch_id=f"s{i}", parent_branch_id="ps", step_idx=1)
        for i, tokens in enumerate(candidates)
    ]
    folded_a(parent, branch_id="pa")
    outs, _ids = _per_parent_reference(folded_a, ["pa"], [candidates], "ref")
    for row, out in enumerate(serial):
        _assert_tensor_close(outs[0][row : row + 1], out, _atol(device))


def test_ut2401_api_contract(device: str) -> None:
    """UT-2401 (FR1): ``parent_branch_ids`` validation contract."""
    _raw, folded, _cache, _gate = _build_setup_mp(device)
    tokens = torch.randint(1, _VOCAB, (2, 4), device=device)
    with pytest.raises(ValueError):
        folded(
            tokens,
            branch_id=None,
            branch_ids=["a", "b"],
            parent_branch_id="p",
            parent_branch_ids=["p1", "p2"],
        )
    with pytest.raises(ValueError):
        folded(tokens, branch_id="x", parent_branch_ids=["p1", "p2"])
    with pytest.raises(ValueError):
        folded(
            tokens,
            branch_id=None,
            branch_ids=["a", "b"],
            parent_branch_ids=["p1"],
        )


def test_ut2402_none_attach_zero_perturbation(device: str) -> None:
    """UT-2402 (FR1): ``parent_branch_ids=None`` attaches nothing."""
    _raw, folded, _cache, _gate = _build_setup_mp(device)
    parent = torch.randint(1, _VOCAB, (1, 6), device=device)
    child = parent.clone()
    child[0, 3] = (int(parent[0, 3].item()) + 1) % _VOCAB
    folded(parent, branch_id="p")
    out_none = folded(
        child,
        branch_id="c1",
        parent_branch_id="p",
        parent_branch_ids=None,
        step_idx=1,
    )
    out_plain = folded(child, branch_id="c2", parent_branch_id="p", step_idx=1)
    assert torch.equal(out_none, out_plain)


def test_ut2403_homogeneous_ids_equal_single_parent(device: str) -> None:
    """UT-2403 (FR1): a repeated parent id list equals the single-parent path."""
    parent, candidates = _make_equal_len_candidates(device, num=4)
    _raw_a, folded_a, _cache_a, _gate = _build_setup_mp(device)
    _raw_b, folded_b, _cache_b, _gate = _build_setup_mp(device)
    folded_a(parent, branch_id="pa")
    ids_a = [f"sp-r{i}" for i in range(4)]
    out_single = folded_a(
        torch.cat(candidates, dim=0),
        branch_id=None,
        branch_ids=ids_a,
        parent_branch_id="pa",
        step_idx=1,
    )
    folded_b(parent, branch_id="pb")
    out_multi, ids_b = _multi_call(folded_b, ["pb"], [candidates], "hm")
    assert torch.equal(out_single, out_multi)
    for id_a, id_b in zip(ids_a, ids_b):
        _assert_cache_entry_close(_cache_a, id_a, _cache_b, id_b, None)


# ---------------------------------------------------------------------------
# Assembly face
# ---------------------------------------------------------------------------


def test_ut2404_all_distinct_parents_parity(device: str) -> None:
    """UT-2404 (FR2): one multi-parent call equals per-parent batched calls."""
    atol = _atol(device)
    parents, groups = _make_two_parent_world(device)
    _raw_a, folded_a, cache_a, _gate = _build_setup_mp(device)
    _raw_b, folded_b, cache_b, _gate = _build_setup_mp(device)
    _populate_parents(folded_a, parents, ["pa", "pb"])
    ref_outs, ref_ids = _per_parent_reference(folded_a, ["pa", "pb"], groups, "ref2404")
    _populate_parents(folded_b, parents, ["pa", "pb"])
    logits, ids = _multi_call(folded_b, ["pa", "pb"], groups, "mp2404")
    row = 0
    for out in ref_outs:
        for i in range(out.shape[0]):
            _assert_tensor_close(logits[row : row + 1], out[i : i + 1], atol)
            _assert_cache_entry_close(cache_b, ids[row], cache_a, ref_ids[row], atol)
            profile_b = GLOBAL_STABILITY_PROFILER.get_profile(ids[row])
            profile_a = GLOBAL_STABILITY_PROFILER.get_profile(ref_ids[row])
            assert profile_b is not None and profile_a is not None
            assert len(profile_b.layer_stats) == len(profile_a.layer_stats)
            row += 1
    assert row == logits.shape[0]


def test_ut2405_missing_parent_row_degrades(device: str) -> None:
    """UT-2405 (FR2): a missing parent row degrades to its own recompute."""
    atol = _atol(device)
    parents, groups = _make_two_parent_world(device)
    _raw_a, folded_a, _cache_a, _gate = _build_setup_mp(device)
    _raw_b, folded_b, _cache_b, _gate = _build_setup_mp(device)
    # Only parent A is populated; parent B is a cache miss.
    folded_a(parents[0], branch_id="pa")
    ref_outs, _ref_ids = _per_parent_reference(
        folded_a, ["pa", "pb"], groups, "ref", parented=[True, False]
    )
    folded_b(parents[0], branch_id="pa")
    logits, _ids = _multi_call(folded_b, ["pa", "pb"], groups, "mp")
    for i in range(ref_outs[0].shape[0]):
        _assert_tensor_close(logits[i : i + 1], ref_outs[0][i : i + 1], atol)
    for i in range(ref_outs[1].shape[0]):
        _assert_tensor_close(
            logits[ref_outs[0].shape[0] + i : ref_outs[0].shape[0] + i + 1],
            ref_outs[1][i : i + 1],
            atol,
        )


def test_ut2406_missing_row_no_leak_constructive(device: str) -> None:
    """UT-2406 (FR2, design DD2): the no-leak guarantee is constructive.

    With an ``l2``/``tau=0`` gate a zeros placeholder can score stable, so
    the implicit cosine argument does not hold; the explicit
    ``stable_mask[~row_valid] = False`` force must keep the missing row at
    its own recompute (never another row's parent values).
    """
    atol = _atol(device)
    parents, groups = _make_two_parent_world(device)
    gate = SimilarityGate(tau=0.0, metric="l2")
    _raw_a, folded_a, _cache_a, _gate = _build_setup_mp(device, gate=gate)
    _raw_b, folded_b, _cache_b, _gate = _build_setup_mp(device, gate=gate)
    folded_a(parents[0], branch_id="pa")
    ref_outs, _ref_ids = _per_parent_reference(
        folded_a, ["pa", "pb"], groups, "ref", parented=[True, False]
    )
    folded_b(parents[0], branch_id="pa")
    logits, _ids = _multi_call(folded_b, ["pa", "pb"], groups, "mp")
    offset = ref_outs[0].shape[0]
    # Observability guard: the rows carry distinct values by construction,
    # so a leak (the missing row inheriting another row's merged parent
    # values) would be value-observable.
    assert not torch.equal(logits[offset : offset + 1], logits[0:1])
    for i in range(ref_outs[1].shape[0]):
        _assert_tensor_close(logits[offset + i : offset + i + 1], ref_outs[1][i : i + 1], atol)


def test_ut2407_all_missing_whole_batch_recompute(device: str) -> None:
    """UT-2407 (FR2): all parents missing -- one whole-batch recompute."""
    atol = _atol(device)
    parents, groups = _make_two_parent_world(device)
    _raw_a, folded_a, _cache_a, _gate = _build_setup_mp(device)
    _raw_b, folded_b, _cache_b, _gate = _build_setup_mp(device)
    ref_outs, _ref_ids = _per_parent_reference(
        folded_a, ["pa", "pb"], groups, "ref", parented=[False, False]
    )
    counts = _spy_recompute_all(folded_b)
    logits, _ids = _multi_call(folded_b, ["pa", "pb"], groups, "mp")
    assert counts == [1] * _NUM_LAYERS, "one batched recompute per layer"
    row = 0
    for out in ref_outs:
        for i in range(out.shape[0]):
            _assert_tensor_close(logits[row : row + 1], out[i : i + 1], atol)
            row += 1


def test_ut2408_parent_batch_not1_hard_error(device: str) -> None:
    """UT-2408 (FR2): a batch-2 parent among good parents is a hard error."""
    parents, groups = _make_two_parent_world(device)
    _raw, folded, _cache, _gate = _build_setup_mp(device)
    folded(parents[0], branch_id="pa")
    # Parent B cached at batch 2 (a [2, T] forward stores [2, T, H]).
    folded(parents[1].repeat(2, 1), branch_id="pb")
    with pytest.raises(RuntimeError):
        _multi_call(folded, ["pa", "pb"], groups, "mp")


def test_ut2409_parent_length_mismatch_row_degrades(device: str) -> None:
    """UT-2409 (FR2, v1 equal-length semantics): T_p != T_c degrades the row.

    Unlike the single-parent face (AR007 truncation), the multi-parent v1
    face requires ``T_p_i == T_c``: a longer parent degrades its row to a
    full recompute instead of donating a truncated prefix.
    """
    atol = _atol(device)
    parents, groups = _make_two_parent_world(device, seq_len=8)
    _raw_a, folded_a, _cache_a, _gate = _build_setup_mp(device)
    _raw_b, folded_b, _cache_b, _gate = _build_setup_mp(device)
    folded_a(parents[0], branch_id="pa")
    ref_outs, _ref_ids = _per_parent_reference(
        folded_a, ["pa", "pb"], groups, "ref", parented=[True, False]
    )
    # Parent B cached on a canvas three tokens longer than the children.
    longer_b = torch.cat([parents[1], torch.randint(1, _VOCAB, (1, 3), device=device)], dim=1)
    folded_b(parents[0], branch_id="pa")
    folded_b(longer_b, branch_id="pb")
    logits, _ids = _multi_call(folded_b, ["pa", "pb"], groups, "mp")
    for i in range(ref_outs[0].shape[0]):
        _assert_tensor_close(logits[i : i + 1], ref_outs[0][i : i + 1], atol)
    offset = ref_outs[0].shape[0]
    for i in range(ref_outs[1].shape[0]):
        _assert_tensor_close(logits[offset + i : offset + i + 1], ref_outs[1][i : i + 1], atol)


# ---------------------------------------------------------------------------
# Three-branch face
# ---------------------------------------------------------------------------


def test_ut2410_all_stable_fast_path_multi_parent(device: str) -> None:
    """UT-2410 (FR3): the fast path stacks multi-parent FFN rows, zero recompute."""
    atol = _atol(device)
    torch.manual_seed(4310)
    parents = [torch.randint(1, _VOCAB, (1, 8), device=device) for _ in range(2)]
    groups = [[parent.clone() for _ in range(2)] for parent in parents]
    _raw, folded, _cache, _gate = _build_setup_mp(device)
    parent_logits = [folded(parent, branch_id=pid) for parent, pid in zip(parents, ["pa", "pb"])]
    counts = _spy_recompute_all(folded)
    logits, _ids = _multi_call(folded, ["pa", "pb"], groups, "as")
    assert counts == [0] * _NUM_LAYERS, "the all-stable fast path never recomputes"
    for row in range(4):
        out = parent_logits[0] if row < 2 else parent_logits[1]
        _assert_tensor_close(logits[row : row + 1], out, atol)


def test_ut2411_mixed_batch_stable_row_bit_identical(device: str) -> None:
    """UT-2411 (FR3, the :617-620 contract): a fully stable row inside a mixed
    batch merges to its parent values -- bit-identical to the serial fast
    path for that row."""
    atol = _atol(device)
    torch.manual_seed(4311)
    parents = [torch.randint(1, _VOCAB, (1, 8), device=device) for _ in range(2)]
    # Group A: token-identical children (fully stable rows); group B:
    # mid-divergent children (mixed rows).
    group_a = [parents[0].clone() for _ in range(2)]
    group_b = []
    for i in range(2):
        child = parents[1].clone()
        child[0, 4] = (int(parents[1][0, 4].item()) + 1 + i) % _VOCAB
        group_b.append(child)
    groups = [group_a, group_b]
    _raw_a, folded_a, _cache_a, _gate = _build_setup_mp(device)
    _raw_b, folded_b, _cache_b, _gate = _build_setup_mp(device)
    parent_a_logits = folded_a(parents[0], branch_id="pa")
    folded_a(parents[1], branch_id="pb")
    ref_outs, _ref_ids = _per_parent_reference(folded_a, ["pa", "pb"], [group_a, group_b], "ref")
    folded_b(parents[0], branch_id="pa")
    folded_b(parents[1], branch_id="pb")
    logits, _ids = _multi_call(folded_b, ["pa", "pb"], groups, "mx")
    # Stable rows: the parent's own values (the serial fast-path result).
    for i in range(2):
        _assert_tensor_close(logits[i : i + 1], parent_a_logits, atol)
    # Divergent rows: the per-parent mixed-merge reference.
    for i in range(2):
        _assert_tensor_close(logits[2 + i : 3 + i], ref_outs[1][i : i + 1], atol)


def test_ut2412_fully_divergent_rows_equal_recompute(device: str) -> None:
    """UT-2412 (FR3): fully divergent rows equal the whole-batch recompute.

    The reference is a single same-shape ``[4, T]`` unparented batched call
    (the current API), so the comparison isolates the mechanism claim —
    the merge with an all-False mask returns ``child_out`` unchanged —
    from CUDA M-shape GEMM noise (a per-group ``[2, T]`` reference would
    compound batch-shape differences across layers on CUDA).
    """
    parents, _groups = _make_two_parent_world(device)
    # GUARANTEED fully-divergent children: every position's token differs
    # from the parent's (a fixed non-zero offset per row). Fresh random
    # tokens can collide with the parent's at individual positions -- at
    # layer 0 the gate compares pure token embeddings, so a collision is
    # cosine-exactly-1.0 stable and the merge legitimately reuses the
    # parent value there, breaking the "fully divergent" premise (found by
    # the CUDA run; the CPU RNG happened not to collide).
    groups = [[(parents[g] + 13 + 2 * g + i) % _VOCAB for i in range(2)] for g in range(2)]
    _raw_a, folded_a, _cache_a, _gate = _build_setup_mp(device)
    _raw_b, folded_b, _cache_b, _gate = _build_setup_mp(device)
    stacked = torch.cat([child for group in groups for child in group], dim=0)
    ref_ids = [f"rec2404-r{i}" for i in range(stacked.shape[0])]
    ref_logits = folded_a(
        stacked,
        branch_id=None,
        branch_ids=ref_ids,
        parent_branch_id=None,
        step_idx=1,
    )
    _populate_parents(folded_b, parents, ["pa", "pb"])
    logits, _ids = _multi_call(folded_b, ["pa", "pb"], groups, "dv2412")
    for row in range(stacked.shape[0]):
        _assert_tensor_close(logits[row : row + 1], ref_logits[row : row + 1], _atol(device))


def test_ut2413_fast_path_ffn_missing_slow_path(device: str) -> None:
    """UT-2413 (FR3): a hidden parent FFN entry degrades to the slow path.

    The dropped ``(pb, layer 1)`` fetch breaks the fast path's FFN
    precondition while every gate-layer fetch still succeeds; the unified
    slow path recomputes once and merges, producing values equal to the
    full-cache per-parent reference.
    """
    atol = _atol(device)
    torch.manual_seed(4313)
    parents = [torch.randint(1, _VOCAB, (1, 8), device=device) for _ in range(2)]
    groups = [[parent.clone() for _ in range(2)] for parent in parents]
    _raw_a, folded_a, _cache_a, _gate = _build_setup_mp(device)
    _raw_b, folded_b, cache_b, _gate = _build_setup_mp(device)
    _populate_parents(folded_a, parents, ["pa", "pb"])
    ref_outs, _ref_ids = _per_parent_reference(folded_a, ["pa", "pb"], groups, "ref")
    _populate_parents(folded_b, parents, ["pa", "pb"])
    # Drop at the TOP layer: a drop at an interior layer perturbs the
    # downstream layers' child values (the recomputed row diverges from the
    # parent, so the downstream gate goes mixed) -- the top layer isolates
    # the fast-path FFN face with every upstream layer staying fast.
    folded_b.cache = _DropEntryCache(cache_b, "pb", _NUM_LAYERS - 1)
    counts = _spy_recompute_all(folded_b)
    logits, _ids = _multi_call(folded_b, ["pa", "pb"], groups, "fp2413")
    assert all(count == 0 for count in counts[:-1]), "layers without the drop stay fast"
    assert counts[-1] >= 1, "the dropped layer must take the slow path"
    row = 0
    for out in ref_outs:
        for i in range(out.shape[0]):
            _assert_tensor_close(logits[row : row + 1], out[i : i + 1], atol)
            row += 1


# ---------------------------------------------------------------------------
# Merge face
# ---------------------------------------------------------------------------


def test_ut2414_merge_placeholder_row_no_leak(device: str) -> None:
    """UT-2414 (FR3, design DD4): a merge-path placeholder row never leaks.

    Mixed rows with one parent missing: the missing row's merged output
    equals its own recompute (the ``child_out[:1]`` placeholder is shape
    alignment only -- values must not flow).
    """
    atol = _atol(device)
    parents, groups = _make_two_parent_world(device)
    _raw_a, folded_a, _cache_a, _gate = _build_setup_mp(device)
    _raw_b, folded_b, _cache_b, _gate = _build_setup_mp(device)
    folded_a(parents[0], branch_id="pa")
    ref_outs, _ref_ids = _per_parent_reference(
        folded_a, ["pa", "pb"], groups, "ref", parented=[True, False]
    )
    folded_b(parents[0], branch_id="pa")
    logits, _ids = _multi_call(folded_b, ["pa", "pb"], groups, "mg")
    offset = ref_outs[0].shape[0]
    for i in range(ref_outs[1].shape[0]):
        _assert_tensor_close(logits[offset + i : offset + i + 1], ref_outs[1][i : i + 1], atol)


def test_ut2415_merge_torch_zero_change_regression() -> None:
    """UT-2415 (green): ``_merge_stable_divergent_torch`` keeps [N, T] row semantics."""
    torch.manual_seed(4315)
    parent = torch.randn(3, 4, 5)
    child = torch.randn(3, 4, 5)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    assert torch.equal(_merge_stable_divergent_torch(parent, child, mask), child)
    mask[:] = True
    assert torch.equal(_merge_stable_divergent_torch(parent, child, mask), parent)
    mask[:] = False
    mask[0, :] = True
    mask[2, 0] = True
    merged = _merge_stable_divergent_torch(parent, child, mask)
    assert torch.equal(merged[0], parent[0])
    assert torch.equal(merged[1], child[1])
    assert torch.equal(merged[2, 0], parent[2, 0])
    assert torch.equal(merged[2, 1:], child[2, 1:])


# ---------------------------------------------------------------------------
# Publish face (switch on: the ``multi_parent_batch`` MFF kwarg, T004)
# ---------------------------------------------------------------------------


def test_ut2416_put_batch_publish_equivalence(device: str) -> None:
    """UT-2416 (FR4): put_batch and per-row put leave identical cache states."""
    parents, groups = _make_two_parent_world(device)
    _raw_on, folded_on, cache_on, _gate = _build_setup_mp(
        device, mff_kwargs={"multi_parent_batch": True}
    )
    _raw_off, folded_off, cache_off, _gate = _build_setup_mp(device)
    _populate_parents(folded_on, parents, ["pa", "pb"])
    logits_on, ids_on = _multi_call(folded_on, ["pa", "pb"], groups, "on")
    _populate_parents(folded_off, parents, ["pa", "pb"])
    logits_off, ids_off = _multi_call(folded_off, ["pa", "pb"], groups, "off")
    assert torch.equal(logits_on, logits_off)
    assert len(ids_on) == len(ids_off)
    for id_on, id_off in zip(ids_on, ids_off):
        _assert_cache_entry_close(cache_on, id_on, cache_off, id_off, None)


def test_ut2417_no_put_batch_capability_fallback(device: str) -> None:
    """UT-2417 (FR4): a cache without put_batch falls back to per-row put."""
    atol = _atol(device)
    parents, groups = _make_two_parent_world(device)
    _raw_a, folded_a, _cache_a, _gate = _build_setup_mp(device)
    _raw_b, folded_b, cache_b, _gate = _build_setup_mp(
        device, mff_kwargs={"multi_parent_batch": True}
    )
    _populate_parents(folded_a, parents, ["pa", "pb"])
    ref_outs, _ref_ids = _per_parent_reference(folded_a, ["pa", "pb"], groups, "ref")
    _populate_parents(folded_b, parents, ["pa", "pb"])
    folded_b.cache = _NoBatchPutCache(cache_b)
    logits, _ids = _multi_call(folded_b, ["pa", "pb"], groups, "np")
    row = 0
    for out in ref_outs:
        for i in range(out.shape[0]):
            _assert_tensor_close(logits[row : row + 1], out[i : i + 1], atol)
            row += 1


def test_ut2418_ledger_multi_parent_smaller(device: str) -> None:
    """UT-2418 (FR4): one multi-parent call has a strictly smaller write ledger."""
    parents, groups = _make_two_parent_world(device)
    _raw_on, folded_on, cache_on, _gate = _build_setup_mp(
        device, mff_kwargs={"multi_parent_batch": True}
    )
    _raw_off, folded_off, cache_off, _gate = _build_setup_mp(device)
    _populate_parents(folded_on, parents, ["pa", "pb"])
    before_on = cache_on.write_op_count
    _multi_call(folded_on, ["pa", "pb"], groups, "on")
    delta_on = cache_on.write_op_count - before_on
    _populate_parents(folded_off, parents, ["pa", "pb"])
    before_off = cache_off.write_op_count
    _per_parent_reference(folded_off, ["pa", "pb"], groups, "off")
    delta_off = cache_off.write_op_count - before_off
    assert delta_on < delta_off


def test_ut2419_single_parent_batch_put_batchized(device: str) -> None:
    """UT-2419 (FR4, the DD5 amplifier): single-parent batches put_batch too."""
    parent, candidates = _make_equal_len_candidates(device, num=4)
    _raw_on, folded_on, cache_on, _gate = _build_setup_mp(
        device, mff_kwargs={"multi_parent_batch": True}
    )
    _raw_off, folded_off, cache_off, _gate = _build_setup_mp(device)
    stacked = torch.cat(candidates, dim=0)
    ids = [f"sp-r{i}" for i in range(4)]
    folded_on(parent, branch_id="pa")
    before_on = cache_on.write_op_count
    out_on = folded_on(stacked, branch_id=None, branch_ids=ids, parent_branch_id="pa", step_idx=1)
    delta_on = cache_on.write_op_count - before_on
    folded_off(parent, branch_id="pb")
    before_off = cache_off.write_op_count
    out_off = folded_off(stacked, branch_id=None, branch_ids=ids, parent_branch_id="pb", step_idx=1)
    delta_off = cache_off.write_op_count - before_off
    assert delta_on < delta_off
    assert torch.equal(out_on, out_off)


# ---------------------------------------------------------------------------
# Wiring face (folded_generate_batched, T003)
# ---------------------------------------------------------------------------


def _run_two_chain_reference(
    device: str,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
) -> list[Any]:
    """Per-chain ``folded_generate`` reference runs on fresh setups."""
    results = []
    for prompt in prompts:
        raw, folded, _cache, _gate = _build_setup_mp(device)
        draft = DraftGenerator(vocab_size=_VOCAB, mode="suffix_append", flip_ratio=0.0)
        results.append(
            folded_generate(
                _build_adapter(raw, folded),
                prompt,
                max_new_tokens=max_new_tokens,
                folded_model=folded,
                draft_generator=draft,
                num_branches_per_step=2,
            )
        )
    return results


def _build_adapter(raw: torch.nn.Module, folded: ManualFoldedForward) -> Any:
    from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter

    return FastDLLMAdapter(raw, folded_model=folded, num_layers=_NUM_LAYERS, hidden_dim=32)


def test_ut2420_batched_vs_per_chain_parity(device: str) -> None:
    """UT-2420 (FR5): two chains merged equal per-chain generation."""
    ChainInput, folded_generate_batched = _import_batched_generation_api()
    prompts = [
        torch.tensor([[1, 5, 9, 13]], device=device),
        torch.tensor([[2, 6, 10, 14]], device=device),
    ]
    ref = _run_two_chain_reference(device, prompts, max_new_tokens=3)
    _raw, folded, _cache, _gate = _build_setup_mp(device, mff_kwargs={"multi_parent_batch": True})
    draft = DraftGenerator(vocab_size=_VOCAB, mode="suffix_append", flip_ratio=0.0)
    results = folded_generate_batched(
        _build_adapter(_raw, folded),
        [ChainInput(input_ids=p) for p in prompts],
        max_new_tokens=3,
        folded_model=folded,
        draft_generator=draft,
        num_branches_per_step=2,
    )
    assert len(results) == 2
    for res, ref_res in zip(results, ref):
        assert torch.equal(res.tokens, ref_res.tokens)
        assert res.stable_ratio == ref_res.stable_ratio
        assert res.acceptance_rate == ref_res.acceptance_rate


def test_ut2421_unequal_length_fallback(device: str) -> None:
    """UT-2421 (FR5): unequal-length chains fall back to per-chain generation."""
    ChainInput, folded_generate_batched = _import_batched_generation_api()
    prompts = [
        torch.tensor([[1, 5, 9, 13]], device=device),
        torch.tensor([[2, 6, 10, 14, 7, 3]], device=device),
    ]
    ref = _run_two_chain_reference(device, prompts, max_new_tokens=3)
    _raw, folded, _cache, _gate = _build_setup_mp(device, mff_kwargs={"multi_parent_batch": True})
    draft = DraftGenerator(vocab_size=_VOCAB, mode="suffix_append", flip_ratio=0.0)
    results = folded_generate_batched(
        _build_adapter(_raw, folded),
        [ChainInput(input_ids=p) for p in prompts],
        max_new_tokens=3,
        folded_model=folded,
        draft_generator=draft,
        num_branches_per_step=2,
    )
    for res, ref_res in zip(results, ref):
        assert torch.equal(res.tokens, ref_res.tokens)
        assert res.stable_ratio == ref_res.stable_ratio


def test_ut2422_off_switch_fallback(device: str) -> None:
    """UT-2422 (FR6): the default off switch stays per-chain."""
    ChainInput, folded_generate_batched = _import_batched_generation_api()
    prompts = [
        torch.tensor([[1, 5, 9, 13]], device=device),
        torch.tensor([[2, 6, 10, 14]], device=device),
    ]
    ref = _run_two_chain_reference(device, prompts, max_new_tokens=3)
    _raw, folded, _cache, _gate = _build_setup_mp(device)
    draft = DraftGenerator(vocab_size=_VOCAB, mode="suffix_append", flip_ratio=0.0)
    results = folded_generate_batched(
        _build_adapter(_raw, folded),
        [ChainInput(input_ids=p) for p in prompts],
        max_new_tokens=3,
        folded_model=folded,
        draft_generator=draft,
        num_branches_per_step=2,
    )
    for res, ref_res in zip(results, ref):
        assert torch.equal(res.tokens, ref_res.tokens)
        assert res.stable_ratio == ref_res.stable_ratio


def test_ut2423_single_chain_degenerate(device: str) -> None:
    """UT-2423 (FR5): a single chain equals single-chain generation."""
    ChainInput, folded_generate_batched = _import_batched_generation_api()
    prompt = torch.tensor([[1, 5, 9, 13]], device=device)
    ref = _run_two_chain_reference(device, [prompt], max_new_tokens=3)
    _raw, folded, _cache, _gate = _build_setup_mp(device, mff_kwargs={"multi_parent_batch": True})
    draft = DraftGenerator(vocab_size=_VOCAB, mode="suffix_append", flip_ratio=0.0)
    results = folded_generate_batched(
        _build_adapter(_raw, folded),
        [ChainInput(input_ids=prompt)],
        max_new_tokens=3,
        folded_model=folded,
        draft_generator=draft,
        num_branches_per_step=2,
    )
    assert len(results) == 1
    assert torch.equal(results[0].tokens, ref[0].tokens)
    assert results[0].stable_ratio == ref[0].stable_ratio


def test_ut2424_prune_per_chain(device: str) -> None:
    """UT-2424 (FR5): rejected candidates leave no residue in either chain."""
    ChainInput, folded_generate_batched = _import_batched_generation_api()
    prompts = [
        torch.tensor([[1, 5, 9, 13]], device=device),
        torch.tensor([[2, 6, 10, 14]], device=device),
    ]
    _raw, folded, cache, _gate = _build_setup_mp(device, mff_kwargs={"multi_parent_batch": True})
    put_ids = _track_cached_branches(cache)
    draft = DraftGenerator(vocab_size=_VOCAB, mode="suffix_append", flip_ratio=0.0)
    results = folded_generate_batched(
        _build_adapter(_raw, folded),
        [ChainInput(input_ids=p) for p in prompts],
        max_new_tokens=3,
        folded_model=folded,
        draft_generator=draft,
        num_branches_per_step=2,
    )
    survivors = [bid for bid in put_ids if cache.contains(bid, 0)]
    # Two chains x three accepted nodes, exactly like UT-1405's single chain.
    assert len(survivors) == 6
    for res in results:
        assert res.final_branch_id in survivors
    rejected = [bid for bid in put_ids if bid not in survivors]
    assert rejected, "multi-candidate chains must prune rejected siblings"
    for branch_id in rejected:
        assert not cache.contains(branch_id, 0)


# ---------------------------------------------------------------------------
# Config face (T004)
# ---------------------------------------------------------------------------


def test_ut2425_config_default_off() -> None:
    """UT-2425 (FR6): ``ActFoldConfig.multi_parent_batch`` defaults to False."""
    cfg = ActFoldConfig()
    assert cfg.multi_parent_batch is False


def test_ut2426_yaml_consumption_no_warning(tmp_path: Any) -> None:
    """UT-2426 (FR6, #5): the YAML key is consumed without the unknown-key warning."""
    yaml_path = tmp_path / "ar025_cfg.yaml"
    yaml_path.write_text("multi_parent_batch: false\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(yaml_path)
    assert cfg.multi_parent_batch is False
    assert not any(
        "unknown" in str(entry.message).lower()
        for entry in caught
        if issubclass(entry.category, UserWarning)
    )


def test_ut2427_predicate_generalization(device: str) -> None:
    """UT-2427 (FR6): the dispatch predicate merges multi-parent rows iff on."""
    from actfold.speculative.folded_generation import _batched_forward_supported

    tokens = torch.randint(1, _VOCAB, (1, 6), device=device)
    cands = [
        BranchNode(
            branch_id=f"c{i}",
            parent_id=f"p{i}",
            tokens=tokens.clone(),
            depth=1,
        )
        for i in range(2)
    ]
    _raw_off, folded_off, _cache, _gate = _build_setup_mp(device)
    assert _batched_forward_supported(folded_off, cands) is False
    _raw_on, folded_on, _cache, _gate = _build_setup_mp(
        device, mff_kwargs={"multi_parent_batch": True}
    )
    assert _batched_forward_supported(folded_on, cands) is True
    assert _batched_forward_supported(folded_on, cands[:1]) is False
