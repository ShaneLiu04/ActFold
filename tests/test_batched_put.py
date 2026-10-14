"""Red-phase tests for AR020: publish-path batched cache put (UT-2001..UT-2009).

Contract: ``specs/changes/AR020-batched-cache-put/design.md`` §3/§4/§6.1 --
the three activation caches must grow a generic ``put_batch(entries)``
capability (state-equivalent to sequential ``put`` calls on the exact
comparison face of design §4.2.2(1): per ``(branch, step, layer, name)``
``buffer[:final _counts value]`` ``torch.equal`` + ``_counts`` equality +
per-layer ``OrderedDict`` key order + ``num_entries``; NOT buffer rows
beyond the count and NOT capacity), with batch-level pre-validation
atomicity (any invalid entry -> ``ValueError`` before any mutation), a
``write_op_count`` host-call ledger (excluded from the equivalence face),
and the publish path ``ManualFoldedForward._publish_graph_children`` must
probe ``isinstance(self.cache, BatchedPutCapable)`` on EVERY publish (no
instance caching) and dispatch to one ``put_batch`` call whose per-entry
payload is field-identical to the loop variant (snapshot row views, layer-0
embedding row views), falling back to the original double loop for
incapable caches.

UT -> test function mapping (one UT may map to several functions):

===== ========================================== ================================
UT    Function(s)                                Asserted points
===== ========================================== ================================
2001  test_ut2001_fixed_scenario_equivalence     ActivationCache put_batch vs
      test_ut2001_randomized_property_equivalence put terminal-state equivalence
                                                 (fixed + seeded randomized
                                                 property test: mixed
                                                 batch/seq/step, repeated keys,
                                                 duplicate entries, small-budget
                                                 eviction, mixed dtype/device on
                                                 CUDA); fetch face companion; the
                                                 write_op_count probe asserted to
                                                 be OUTSIDE the equivalence face.
2002  test_ut2002_fixed_scenario_equivalence     VectorizedActivationCache same
      test_ut2002_randomized_property_equivalence face incl.
                                                 _evict_old_branch_steps
                                                 trigger-point equivalence.
2003  test_ut2003_fixed_scenario_equivalence     ChunkedActivationCache face:
      test_ut2003_randomized_property_equivalence chunk data, key order,
                                                 num_entries and num_tokens,
                                                 cross-chunk-boundary seqs.
2004  test_ut2004_atomicity_zero_writes          Any invalid entry (empty dict /
                                                 ndim<2 / leading-shape mismatch)
                                                 at first/middle/last position ->
                                                 ValueError and the cache's full
                                                 internal state deep-compares
                                                 equal to the pre-call snapshot
                                                 (zero writes; message carries
                                                 the entry index).
2005  test_ut2005_ring_grow_overlap_equivalence  Ring entries, buffer grow,
                                                 simple+ring mixed batches, and
                                                 repeated logical destinations
                                                 with overlapping dst (incl. the
                                                 flush-order-sensitive grow and
                                                 ring variants) stay
                                                 loop-equivalent.
2006  test_ut2006_protocol_and_isinstance        CachePutEntry field order and
                                                 default; BatchedPutCapable
                                                 isinstance on the three caches
                                                 (wrapper negative);
                                                 put_batch returns None (empty
                                                 batch is a no-op);
      test_ut2006_publish_dispatch_batched_payload capable cache -> put_batch
                                                 exactly once with L*N entries
                                                 whose payload is field-identical
                                                 to the loop rows;
      test_ut2006_publish_incapable_fallback_loop incapable wrapper -> original
                                                 loop (put called L*N times;
                                                 double-sided green in Red);
      test_ut2006_probe_each_publish_and_cache_swap the probe re-runs every
                                                 publish: swapping the cache
                                                 switches the branch next time;
      test_ut2006_profiler_face_unchanged        record_precomputed call count
                                                 and parameters identical to the
                                                 loop variant;
      test_ut2006_publish_exception_propagates   a raising put_batch propagates
                                                 unchanged out of the publish.
2007  test_ut2007_loop_put_ledger                write_op_count starts at 0; put
                                                 increments by len(activations);
                                                 chunked counts materialized
                                                 chunks only;
      test_ut2007_put_batch_direct_ledger        put_batch ledger: one foreach
                                                 group for a uniform batch, two
                                                 for mixed dtype, per-pair counts
                                                 for ring and repeated
                                                 destinations;
      test_ut2007_publish_ledger_loop_and_batch  publish loop branch delta ==
                                                 L*N+N; batch branch delta <=
                                                 2*(L+N) (c=2 frozen).
2008  test_ut2008_zero_host_sync_and_freshness   the batched publish window
                                                 performs zero
                                                 .item()/.tolist()/.cpu(); the
                                                 published cache contents survive
                                                 an in-place child_buf mutation;
      test_ut2008_debug_path_per_row_record      debug_enabled keeps the per-row
                                                 record face while still
                                                 dispatching to put_batch.
2009  test_ut2009_foreach_downgrade_equivalence_ monkeypatching
      and_ledger                                 batched_write._FOREACH_AVAILABLE
                                                 to False keeps put_batch
                                                 loop-equivalent with
                                                 write_op_count == per-pair count;
      test_ut2009_publish_downgrade_dispatch     the publish dispatch is
                                                 unaffected by the downgrade flag.
===== ========================================== ================================

Red state (why the new-behavior assertions fail today) -- every failure is a
production-missing face, never a swallowed exception:

* ``actfold.core.cache_protocol.CachePutEntry`` / ``BatchedPutCapable`` do
  not exist (ImportError: UT-2001..2005, UT-2007's direct ledger, UT-2009,
  and the protocol sub-block of UT-2006);
* ``actfold.core.batched_write`` does not exist (ImportError: UT-2009);
* the three caches have no ``write_op_count`` (AttributeError: UT-2007);
* ``_publish_graph_children`` has no batched branch, so the capable spy
  caches never observe a ``put_batch`` call and the raising-cache exception
  never fires (assertion failures: UT-2006 dispatch/swap/exception sub-blocks
  and UT-2008).

Double-sided invariant functions (green in the Red run by design, mirroring
the AR018 UT-1703/1705/1706/1708 precedent): the incapable-wrapper fallback
(loop face), the profiler-face comparison, and the zero-host-sync /
freshness sub-assertions all hold for the current loop publish and must keep
holding after the batch branch lands.

Fixtures follow ``tests/test_publish_cost.py`` (self-contained L=2/H=32/T=16
/V=64/N=4 synthetic LLaMA-layout decoder; helpers copied isomorphically into
this file -- no cross-test-module imports). ``N * T = 64`` stays a power of
two. CUDA gating is per-function via ``_cuda``; cache-level equivalence
tests are parametrized over CPU + CUDA. Every random face is pinned to a
fixed seed. The autouse ``_global_profiler_state`` fixture snapshots and
restores the GLOBAL stability profiler around every case.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate, VectorizedActivationCache
from actfold.core.chunked_cache import ChunkedActivationCache
from actfold.models.architecture_utils import ManualFoldedForward, detect_architecture
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_DEVICES: list[str] = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# Entry tuple face: (branch_id, layer_idx, activations, step_idx).
_Spec = list[tuple[str, int, dict[str, torch.Tensor], int]]

_VOCAB = 64
_HIDDEN = 32
_NUM_LAYERS = 2
_SEQ = 16
_NUM_CANDIDATES = 4
# N * T MUST remain a power of two (AR017 §6 fixture convention).
_NUM_TOKENS = _NUM_CANDIDATES * _SEQ
assert _NUM_TOKENS > 0 and _NUM_TOKENS & (_NUM_TOKENS - 1) == 0
_TAU = 0.95
_CACHE_BUDGET = 1024


class _MaskableAttention(nn.Module):
    """Token-wise attention stand-in consuming the mask multiplicatively."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        out = hidden_states * 0.5
        if attention_mask is not None:
            out = out * attention_mask.to(out.dtype).unsqueeze(-1)
        return out, None


class _LlamaLikeLayer(nn.Module):
    """Pre-norm decoder layer with a detectable split FFN chain."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = _MaskableAttention()
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class _LlamaLikeModel(nn.Module):
    """Mock LLaMA-layout decoder: embed -> layers -> model.norm -> lm_head."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": _VOCAB})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.model.layers = nn.ModuleList(
            [_LlamaLikeLayer(_HIDDEN) for _ in range(_NUM_LAYERS)]
        )
        self.model.norm = nn.LayerNorm(_HIDDEN)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))


def _build_model(device: str, seed: int = 2026) -> tuple[nn.Module, Any]:
    """Deterministically build the decoder and its architecture profile."""
    torch.manual_seed(seed)
    model = _LlamaLikeModel().to(device)
    model.eval()
    profile = detect_architecture(model)
    return model, profile


def _make_mff(
    model: nn.Module,
    device: str,
    gate: SimilarityGate | None = None,
    cache: Any = None,
    use_cuda_graph: bool = False,
    graph_capacity_ratio: float = 0.5,
) -> ManualFoldedForward:
    """Fresh ManualFoldedForward (own cache + gate) over the shared model."""
    if cache is None:
        cache = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    if gate is None:
        gate = SimilarityGate(tau=_TAU, metric="cosine")
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        split_layers=True,
        split_min_tokens=1,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=graph_capacity_ratio,
    )


def _make_candidates(
    device: str,
    changes_per_row: int = 1,
    seed: int = 4201,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a parent ``[1, T]`` and N equal-length mid-divergent candidates."""
    torch.manual_seed(seed)
    parent = torch.randint(0, _VOCAB, (1, _SEQ), device=device)
    mid = _SEQ // 2
    rows = []
    for i in range(_NUM_CANDIDATES):
        child = parent.clone()
        for j in range(changes_per_row):
            base = int(child[0, mid + j].item())
            child[0, mid + j] = (base + 1 + i + j) % _VOCAB
        rows.append(child)
    return parent, torch.cat(rows, dim=0)


def _ids(tag: str, num: int = _NUM_CANDIDATES) -> list[str]:
    """Build ``num`` branch ids sharing the ``tag`` prefix (disjoint per UT)."""
    return [f"{tag}{i}" for i in range(num)]


def _serial(
    mff: ManualFoldedForward,
    tokens: torch.Tensor,
    branch_id: str,
    parent_branch_id: str | None = None,
    step_idx: int = 1,
) -> torch.Tensor:
    """One serial ``branch_id`` forward under ``no_grad``."""
    with torch.no_grad():
        return mff(
            tokens,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            step_idx=step_idx,
        )


def _batched(
    mff: ManualFoldedForward,
    stacked: torch.Tensor,
    branch_ids: list[str],
    parent_branch_id: str,
    step_idx: int = 1,
) -> torch.Tensor:
    """One batched ``branch_ids`` forward under ``no_grad``."""
    with torch.no_grad():
        return mff(
            stacked,
            branch_id=None,
            branch_ids=branch_ids,
            parent_branch_id=parent_branch_id,
            step_idx=step_idx,
        )


@pytest.fixture(autouse=True)
def _global_profiler_state() -> Iterator[None]:
    """Snapshot/restore the GLOBAL stability profiler around every case."""
    profiler = GLOBAL_STABILITY_PROFILER
    saved_enabled = profiler.enabled
    saved_debug = profiler.debug_enabled
    saved_raw = {key: list(entries) for key, entries in profiler._raw.items()}
    saved_profiles = dict(profiler._profiles)
    saved_history = {key: list(entries) for key, entries in profiler._history.items()}
    try:
        yield
    finally:
        profiler.enabled = saved_enabled
        profiler.debug_enabled = saved_debug
        profiler._raw.clear()
        for key, entries in saved_raw.items():
            profiler._raw[key] = list(entries)
        profiler._profiles.clear()
        profiler._profiles.update(saved_profiles)
        profiler._history.clear()
        for key, entries in saved_history.items():
            profiler._history[key] = list(entries)


@contextmanager
def _publish_tensor_spy(runner: Any) -> Iterator[dict[str, int]]:
    """Count ``Tensor.sum``/``clone``/``item``/``tolist``/``cpu`` in the window.

    The patches are installed immediately before a single
    ``_publish_graph_children`` invocation and restored in a ``finally``.
    ``item``/``tolist``/``cpu`` count unfiltered (zero host sync allowed in
    the publish window).
    """
    counts = {
        "bool_sum": 0,
        "clone_child": 0,
        "clone_total": 0,
        "item": 0,
        "tolist": 0,
        "cpu": 0,
    }
    original = {
        name: getattr(torch.Tensor, name)
        for name in ("sum", "clone", "item", "tolist", "cpu")
    }

    def _sum(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.dtype is torch.bool:
            counts["bool_sum"] += 1
        return original["sum"](self, *args, **kwargs)

    def _clone(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        counts["clone_total"] += 1
        if self is runner.child_buf:
            counts["clone_child"] += 1
        return original["clone"](self, *args, **kwargs)

    def _item(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        counts["item"] += 1
        return original["item"](self, *args, **kwargs)

    def _tolist(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        counts["tolist"] += 1
        return original["tolist"](self, *args, **kwargs)

    def _cpu(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        counts["cpu"] += 1
        return original["cpu"](self, *args, **kwargs)

    for name, wrapper in (
        ("sum", _sum),
        ("clone", _clone),
        ("item", _item),
        ("tolist", _tolist),
        ("cpu", _cpu),
    ):
        setattr(torch.Tensor, name, wrapper)
    try:
        yield counts
    finally:
        for name, method in original.items():
            setattr(torch.Tensor, name, method)


def _windowed_publish_spy(
    monkeypatch: pytest.MonkeyPatch,
    mff: ManualFoldedForward,
) -> list[dict[str, int]]:
    """Shadow ``mff._publish_graph_children`` with a spy-windowed wrapper."""
    windows: list[dict[str, int]] = []
    original_publish = mff._publish_graph_children

    def _windowed(*args: Any, **kwargs: Any) -> None:
        with _publish_tensor_spy(mff.graph_runner_batched) as counts:
            original_publish(*args, **kwargs)
        windows.append(counts)

    monkeypatch.setattr(mff, "_publish_graph_children", _windowed)
    return windows


# ---------------------------------------------------------------------------
# Cache-level helpers: spec runners, equivalence faces, state snapshots
# ---------------------------------------------------------------------------


def _run_loop(cache: Any, spec: _Spec) -> None:
    """Reference path: sequential ``put`` per entry, in spec order."""
    for branch_id, layer_idx, activations, step_idx in spec:
        cache.put(branch_id, layer_idx, activations, step_idx=step_idx)


def _run_batch(cache: Any, spec: _Spec) -> None:
    """New path: one ``put_batch`` call with positionally built entries."""
    from actfold.core.cache_protocol import CachePutEntry

    cache.put_batch([CachePutEntry(*item) for item in spec])


def _make_cache_by_kind(kind: str, device: str, budget: int | None = None) -> Any:
    """Build a cache of the given kind (``budget`` fixes the token budget)."""
    if kind == "activation":
        return ActivationCache(max_entries_per_layer=budget or 64, device=device)
    if kind == "vectorized":
        return VectorizedActivationCache(
            max_entries_per_layer=budget or 64,
            device=device,
            max_branch_steps=None if budget is not None else 4,
        )
    if kind == "chunked":
        return ChunkedActivationCache(
            max_entries_per_layer=budget or 64, chunk_size=8, device=device
        )
    raise AssertionError(f"unknown cache kind {kind!r}")


def _assert_activation_equivalent(
    loop_cache: ActivationCache,
    batch_cache: ActivationCache,
) -> None:
    """Design §4.2.2(1) comparison face for ActivationCache.

    Compares, per layer: the ``OrderedDict`` key order, the per-group name
    sets, ``buffer[:final count]`` bit-equality and the full ``_counts``
    mapping, plus ``num_entries``. Buffer rows beyond the count and buffer
    capacity are deliberately NOT compared; the ``write_op_count`` probe is
    likewise outside this face.
    """
    assert dict(loop_cache._counts) == dict(batch_cache._counts)
    assert set(loop_cache._layers) == set(batch_cache._layers)
    for layer_idx in set(loop_cache._layers):
        loop_layer = loop_cache._layers[layer_idx]
        batch_layer = batch_cache._layers[layer_idx]
        assert list(loop_layer.keys()) == list(batch_layer.keys()), (
            f"layer {layer_idx}: OrderedDict key order differs "
            f"(loop={list(loop_layer)} batch={list(batch_layer)})"
        )
        for key in loop_layer:
            count = loop_cache._counts[(key[0], key[1], layer_idx)]
            loop_store = loop_layer[key]
            batch_store = batch_layer[key]
            assert set(loop_store) == set(batch_store)
            for name in loop_store:
                assert torch.equal(
                    loop_store[name][:count], batch_store[name][:count]
                ), (
                    f"buffer mismatch for {(key[0], key[1], layer_idx, name)} "
                    f"within the first {count} rows"
                )
    assert loop_cache.num_entries() == batch_cache.num_entries()
    for layer_idx in set(loop_cache._layers):
        assert loop_cache.num_entries(layer_idx) == batch_cache.num_entries(layer_idx)


def _assert_activation_fetch_face(
    loop_cache: ActivationCache,
    batch_cache: ActivationCache,
) -> None:
    """Companion fetch face: ``contains``/``fetch`` agree on every key."""
    keys = set(loop_cache._counts) | set(batch_cache._counts)
    for branch_id, step_idx, layer_idx in sorted(keys):
        contains_a = loop_cache.contains(branch_id, layer_idx, step_idx)
        contains_b = batch_cache.contains(branch_id, layer_idx, step_idx)
        assert contains_a == contains_b, (
            f"contains mismatch for {(branch_id, step_idx, layer_idx)}: "
            f"loop={contains_a} batch={contains_b}"
        )
        if not contains_a:
            with pytest.raises(KeyError):
                loop_cache.fetch(
                    branch_id=branch_id, layer_idx=layer_idx, step_idx=step_idx
                )
            with pytest.raises(KeyError):
                batch_cache.fetch(
                    branch_id=branch_id, layer_idx=layer_idx, step_idx=step_idx
                )
            continue
        face_a = loop_cache.fetch(
            branch_id=branch_id, layer_idx=layer_idx, step_idx=step_idx
        )
        face_b = batch_cache.fetch(
            branch_id=branch_id, layer_idx=layer_idx, step_idx=step_idx
        )
        assert set(face_a) == set(face_b)
        for name in face_a:
            assert torch.equal(face_a[name], face_b[name]), (
                f"fetch mismatch for {(branch_id, step_idx, layer_idx, name)}"
            )


def _assert_vectorized_equivalent(
    loop_cache: VectorizedActivationCache,
    batch_cache: VectorizedActivationCache,
) -> None:
    """Design §4.2.2(1) face for VectorizedActivationCache.

    The outer ``_buffers`` key order is the (branch, step) eviction order
    and MUST compare equal (``_evict_old_branch_steps`` trigger-point
    equivalence, UT-2002).
    """
    assert dict(loop_cache._counts) == dict(batch_cache._counts)
    assert list(loop_cache._buffers.keys()) == list(batch_cache._buffers.keys()), (
        f"_buffers key order differs (loop={list(loop_cache._buffers)} "
        f"batch={list(batch_cache._buffers)})"
    )
    for key in loop_cache._buffers:
        loop_branch = loop_cache._buffers[key]
        batch_branch = batch_cache._buffers[key]
        assert set(loop_branch) == set(batch_branch)
        for layer_idx in set(loop_branch):
            count = loop_cache._counts[(key[0], key[1], layer_idx)]
            loop_store = loop_branch[layer_idx]
            batch_store = batch_branch[layer_idx]
            assert set(loop_store) == set(batch_store)
            for name in loop_store:
                assert torch.equal(
                    loop_store[name][:count], batch_store[name][:count]
                ), (
                    f"buffer mismatch for {(key[0], key[1], layer_idx, name)} "
                    f"within the first {count} rows"
                )
    assert loop_cache.num_entries() == batch_cache.num_entries()


def _assert_vectorized_fetch_face(
    loop_cache: VectorizedActivationCache,
    batch_cache: VectorizedActivationCache,
) -> None:
    """Companion fetch face for VectorizedActivationCache."""
    keys = set(loop_cache._counts) | set(batch_cache._counts)
    for branch_id, step_idx, layer_idx in sorted(keys):
        contains_a = loop_cache.contains(branch_id, layer_idx, step_idx)
        contains_b = batch_cache.contains(branch_id, layer_idx, step_idx)
        assert contains_a == contains_b
        if not contains_a:
            with pytest.raises(KeyError):
                loop_cache.get_all(branch_id, layer_idx, step_idx)
            with pytest.raises(KeyError):
                batch_cache.get_all(branch_id, layer_idx, step_idx)
            continue
        face_a = loop_cache.get_all(branch_id, layer_idx, step_idx)
        face_b = batch_cache.get_all(branch_id, layer_idx, step_idx)
        assert set(face_a) == set(face_b)
        for name in face_a:
            assert torch.equal(face_a[name], face_b[name]), (
                f"fetch mismatch for {(branch_id, step_idx, layer_idx, name)}"
            )


def _assert_chunked_equivalent(
    loop_cache: ChunkedActivationCache,
    batch_cache: ChunkedActivationCache,
) -> None:
    """Chunked face: chunk payloads, key order, num_entries and num_tokens."""
    assert set(loop_cache._caches) == set(batch_cache._caches)
    for layer_idx in set(loop_cache._caches):
        keys_a = list(loop_cache._caches[layer_idx].keys())
        keys_b = list(batch_cache._caches[layer_idx].keys())
        assert keys_a == keys_b, (
            f"layer {layer_idx}: chunk key order differs "
            f"(loop={keys_a} batch={keys_b})"
        )
        for key in keys_a:
            chunk_a = loop_cache._caches[layer_idx][key]
            chunk_b = batch_cache._caches[layer_idx][key]
            assert chunk_a.chunk_id == chunk_b.chunk_id
            assert chunk_a.start_token == chunk_b.start_token
            assert chunk_a.num_valid == chunk_b.num_valid
            assert set(chunk_a.data) == set(chunk_b.data)
            for name in chunk_a.data:
                assert torch.equal(chunk_a.data[name], chunk_b.data[name]), (
                    f"chunk data mismatch for {(key, name)}"
                )
    assert loop_cache.num_entries() == batch_cache.num_entries()
    assert loop_cache.num_tokens() == batch_cache.num_tokens()
    for layer_idx in set(loop_cache._caches):
        assert loop_cache.num_entries(layer_idx) == batch_cache.num_entries(layer_idx)
        assert loop_cache.num_tokens(layer_idx) == batch_cache.num_tokens(layer_idx)


def _assert_chunked_fetch_face(
    loop_cache: ChunkedActivationCache,
    batch_cache: ChunkedActivationCache,
) -> None:
    """Companion fetch face for ChunkedActivationCache."""
    keys: set[tuple[str, int, int]] = set()
    for cache in (loop_cache, batch_cache):
        for layer_idx, odict in cache._caches.items():
            for key in odict:
                keys.add((key[0], key[3], key[2]))
    for branch_id, step_idx, layer_idx in sorted(keys):
        contains_a = loop_cache.contains(branch_id, layer_idx, step_idx)
        contains_b = batch_cache.contains(branch_id, layer_idx, step_idx)
        assert contains_a == contains_b
        if not contains_a:
            with pytest.raises(KeyError):
                loop_cache.fetch(
                    branch_id=branch_id, layer_idx=layer_idx, step_idx=step_idx
                )
            with pytest.raises(KeyError):
                batch_cache.fetch(
                    branch_id=branch_id, layer_idx=layer_idx, step_idx=step_idx
                )
            continue
        face_a = loop_cache.fetch(
            branch_id=branch_id, layer_idx=layer_idx, step_idx=step_idx
        )
        face_b = batch_cache.fetch(
            branch_id=branch_id, layer_idx=layer_idx, step_idx=step_idx
        )
        assert set(face_a) == set(face_b)
        for name in face_a:
            assert torch.equal(face_a[name], face_b[name])


def _snapshot_activation_state(cache: ActivationCache) -> dict[str, Any]:
    """Deep snapshot of every internal container (atomicity face)."""
    layers: dict[int, tuple[list[Any], dict[Any, dict[str, torch.Tensor]]]] = {}
    for layer_idx, odict in cache._layers.items():
        layers[layer_idx] = (
            list(odict.keys()),
            {
                key: {name: tensor.clone() for name, tensor in store.items()}
                for key, store in odict.items()
            },
        )
    return {"layers": layers, "counts": dict(cache._counts)}


def _assert_activation_state_unchanged(
    cache: ActivationCache,
    snapshot: dict[str, Any],
) -> None:
    """Full deep compare against the pre-call snapshot (zero writes)."""
    assert dict(cache._counts) == snapshot["counts"], "_counts must be untouched"
    assert set(cache._layers) == set(snapshot["layers"])
    for layer_idx, (keys, stores) in snapshot["layers"].items():
        odict = cache._layers[layer_idx]
        assert list(odict.keys()) == keys, "group key order must be untouched"
        for key, store_snapshot in stores.items():
            store = odict[key]
            assert list(store.keys()) == list(store_snapshot.keys())
            for name, tensor in store_snapshot.items():
                assert torch.equal(store[name], tensor), (
                    f"buffer {(key, layer_idx, name)} must be byte-identical"
                )


def _snapshot_vectorized_state(cache: VectorizedActivationCache) -> dict[str, Any]:
    """Deep snapshot of every internal container (atomicity face)."""
    buffers: dict[Any, dict[int, dict[str, torch.Tensor]]] = {}
    for key, branch in cache._buffers.items():
        buffers[key] = {
            layer_idx: {name: tensor.clone() for name, tensor in store.items()}
            for layer_idx, store in branch.items()
        }
    return {"buffers": buffers, "counts": dict(cache._counts)}


def _assert_vectorized_state_unchanged(
    cache: VectorizedActivationCache,
    snapshot: dict[str, Any],
) -> None:
    """Full deep compare against the pre-call snapshot (zero writes)."""
    assert dict(cache._counts) == snapshot["counts"], "_counts must be untouched"
    assert list(cache._buffers.keys()) == list(snapshot["buffers"].keys())
    for key, branch_snapshot in snapshot["buffers"].items():
        branch = cache._buffers[key]
        assert list(branch.keys()) == list(branch_snapshot.keys())
        for layer_idx, store_snapshot in branch_snapshot.items():
            store = branch[layer_idx]
            assert list(store.keys()) == list(store_snapshot.keys())
            for name, tensor in store_snapshot.items():
                assert torch.equal(store[name], tensor), (
                    f"buffer {(key, layer_idx, name)} must be byte-identical"
                )


def _snapshot_chunked_state(cache: ChunkedActivationCache) -> dict[str, Any]:
    """Deep snapshot of every internal container (atomicity face)."""
    layers: dict[int, tuple[list[Any], dict[Any, Any]]] = {}
    for layer_idx, odict in cache._caches.items():
        layers[layer_idx] = (
            list(odict.keys()),
            {
                key: (
                    chunk.chunk_id,
                    chunk.start_token,
                    chunk.num_valid,
                    {name: tensor.clone() for name, tensor in chunk.data.items()},
                )
                for key, chunk in odict.items()
            },
        )
    return {"layers": layers}


def _assert_chunked_state_unchanged(
    cache: ChunkedActivationCache,
    snapshot: dict[str, Any],
) -> None:
    """Full deep compare against the pre-call snapshot (zero writes)."""
    assert set(cache._caches) == set(snapshot["layers"])
    for layer_idx, (keys, chunks) in snapshot["layers"].items():
        odict = cache._caches[layer_idx]
        assert list(odict.keys()) == keys, "chunk key order must be untouched"
        for key, (chunk_id, start_token, num_valid, data_snapshot) in chunks.items():
            chunk = odict[key]
            assert chunk.chunk_id == chunk_id
            assert chunk.start_token == start_token
            assert chunk.num_valid == num_valid
            assert list(chunk.data.keys()) == list(data_snapshot.keys())
            for name, tensor in data_snapshot.items():
                assert torch.equal(chunk.data[name], tensor), (
                    f"chunk {(key, name)} must be byte-identical"
                )


def _snapshot_cache_state(cache_kind: str, cache: Any) -> dict[str, Any]:
    if cache_kind == "activation":
        return _snapshot_activation_state(cache)
    if cache_kind == "vectorized":
        return _snapshot_vectorized_state(cache)
    return _snapshot_chunked_state(cache)


def _assert_cache_state_unchanged(
    cache_kind: str,
    cache: Any,
    snapshot: dict[str, Any],
) -> None:
    if cache_kind == "activation":
        _assert_activation_state_unchanged(cache, snapshot)
    elif cache_kind == "vectorized":
        _assert_vectorized_state_unchanged(cache, snapshot)
    else:
        _assert_chunked_state_unchanged(cache, snapshot)


def _fixed_spec_2001(device: str) -> _Spec:
    """Fixed mixed scenario: repeats, duplicate entries, grow, ring, eviction."""
    torch.manual_seed(2101)

    def _t(batch: int, seq: int) -> torch.Tensor:
        return torch.randn(batch, seq, 4, device=device)

    spec: _Spec = [
        ("alpha", 0, {"a": _t(1, 3)}, 0),
        ("alpha", 1, {"a": _t(1, 3), "b": _t(1, 3)}, 0),
        ("beta", 0, {"a": _t(1, 6)}, 1),
        ("alpha", 0, {"a": _t(1, 5)}, 0),
        ("gamma", 0, {"a": _t(1, 9)}, 2),
        ("beta", 0, {"a": _t(1, 2)}, 1),
        ("alpha", 0, {"a": _t(1, 4)}, 0),
    ]
    spec.append(spec[0])
    spec.append(("delta", 2, {"a": _t(1, 7), "b": _t(1, 7)}, 0))
    spec.append(("epsilon", 1, {"a": _t(1, 4)}, 3))
    return spec


def _fixed_spec_2003(device: str) -> _Spec:
    """Fixed chunked scenario: cross-chunk-boundary seqs and chunk eviction."""
    torch.manual_seed(2103)

    def _t(batch: int, seq: int) -> torch.Tensor:
        return torch.randn(batch, seq, 4, device=device)

    spec: _Spec = [
        ("c-alpha", 0, {"a": _t(1, 10)}, 0),
        ("c-alpha", 1, {"a": _t(1, 6)}, 0),
        ("c-beta", 0, {"a": _t(1, 3), "b": _t(1, 3)}, 1),
        ("c-alpha", 0, {"a": _t(1, 10)}, 2),
        ("c-beta", 0, {"a": _t(1, 5)}, 1),
    ]
    spec.append(spec[0])
    spec.append(("c-gamma", 2, {"a": _t(2, 9)}, 0))
    return spec


def _random_spec(
    device: str,
    seed: int,
    *,
    num_entries: int = 48,
    mixed_dtype: bool = False,
    mixed_device: bool = False,
) -> _Spec:
    """Seeded randomized property-test scenario (design UT-2001 table row).

    Mixes batch sizes, sequence lengths (some beyond any small budget, the
    ring path), steps, repeated keys, repeated names and exact duplicate
    entries. Per logical key the (batch, dtype, device) triple is pinned so
    the scenario never relies on the legacy ``copy_`` broadcast behavior;
    dtype/device variety lives ACROSS keys, which is exactly the
    ``(device, dtype)`` grouping face.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    dtypes = [torch.float32, torch.float16] if mixed_dtype else [torch.float32]
    key_attrs: dict[tuple[str, int, int], tuple[int, torch.dtype, str]] = {}
    spec: _Spec = []
    for _ in range(num_entries):
        branch_id = f"rb{rng.randrange(4)}"
        step_idx = rng.choice([0, 1, 2])
        layer_idx = rng.randrange(3)
        key = (branch_id, step_idx, layer_idx)
        if key not in key_attrs:
            tensor_device = device
            if mixed_device and rng.random() < 0.3:
                tensor_device = "cpu"
            key_attrs[key] = (rng.choice([1, 1, 2]), rng.choice(dtypes), tensor_device)
        batch, dtype, tensor_device = key_attrs[key]
        seq = rng.choice([1, 2, 3, 5, 7, 9])
        names = ["a"] if rng.random() < 0.5 else ["a", "b"]
        activations = {
            name: torch.randn(batch, seq, 4, device=tensor_device, dtype=dtype)
            for name in names
        }
        spec.append((branch_id, layer_idx, activations, step_idx))
    spec.append(spec[0])
    spec.append(spec[0])
    spec.append(spec[7])
    return spec


# ---------------------------------------------------------------------------
# UT-2001: ActivationCache batched equivalence (fixed + randomized, CPU/CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_ut2001_fixed_scenario_equivalence(device: str) -> None:
    """UT-2001: put_batch terminal state == sequential put (fixed scenario).

    Red face: ``CachePutEntry`` does not exist yet (ImportError).
    """
    spec = _fixed_spec_2001(device)
    loop_cache = ActivationCache(max_entries_per_layer=6, device=device)
    _run_loop(loop_cache, spec)
    batch_cache = ActivationCache(max_entries_per_layer=6, device=device)
    _run_batch(batch_cache, spec)
    _assert_activation_equivalent(loop_cache, batch_cache)
    _assert_activation_fetch_face(loop_cache, batch_cache)
    # The probe is explicitly OUTSIDE the equivalence face (design §4.2.2(3)):
    # the loop ledger counts one write per activation name while the batch
    # ledger counts foreach groups plus immediate writes, so the two probes
    # are allowed to (and for this multi-entry scenario do) differ.
    assert isinstance(loop_cache.write_op_count, int)
    assert isinstance(batch_cache.write_op_count, int)
    assert loop_cache.write_op_count != batch_cache.write_op_count


@pytest.mark.parametrize("device", _DEVICES)
def test_ut2001_randomized_property_equivalence(device: str) -> None:
    """UT-2001: seeded randomized property test (eviction-dense config).

    Budget 6 against sequence lengths up to 9 forces eviction and ring
    paths on nearly every entry; CUDA adds mixed dtype and mixed device
    entries across keys (the ``(device, dtype)`` grouping face).
    """
    is_cuda = device.startswith("cuda")
    spec = _random_spec(device, seed=2001, mixed_dtype=is_cuda, mixed_device=is_cuda)
    loop_cache = ActivationCache(max_entries_per_layer=6, device=device)
    _run_loop(loop_cache, spec)
    batch_cache = ActivationCache(max_entries_per_layer=6, device=device)
    _run_batch(batch_cache, spec)
    _assert_activation_equivalent(loop_cache, batch_cache)
    _assert_activation_fetch_face(loop_cache, batch_cache)


# ---------------------------------------------------------------------------
# UT-2002: VectorizedActivationCache batched equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_ut2002_fixed_scenario_equivalence(device: str) -> None:
    """UT-2002: vectorized put_batch equivalence incl. branch-step eviction.

    ``max_branch_steps=3`` against five distinct ``(branch, step)`` keys
    forces ``_evict_old_branch_steps`` to fire on new-key inserts; the batch
    path must trigger it at exactly the same points (key-order face).
    """
    spec = _fixed_spec_2001(device)
    loop_cache = VectorizedActivationCache(
        max_entries_per_layer=6, device=device, max_branch_steps=3
    )
    _run_loop(loop_cache, spec)
    batch_cache = VectorizedActivationCache(
        max_entries_per_layer=6, device=device, max_branch_steps=3
    )
    _run_batch(batch_cache, spec)
    _assert_vectorized_equivalent(loop_cache, batch_cache)
    _assert_vectorized_fetch_face(loop_cache, batch_cache)


@pytest.mark.parametrize("device", _DEVICES)
def test_ut2002_randomized_property_equivalence(device: str) -> None:
    """UT-2002: seeded randomized property test (12 keys vs max_branch_steps=3)."""
    is_cuda = device.startswith("cuda")
    spec = _random_spec(device, seed=2002, mixed_dtype=is_cuda, mixed_device=is_cuda)
    loop_cache = VectorizedActivationCache(
        max_entries_per_layer=6, device=device, max_branch_steps=3
    )
    _run_loop(loop_cache, spec)
    batch_cache = VectorizedActivationCache(
        max_entries_per_layer=6, device=device, max_branch_steps=3
    )
    _run_batch(batch_cache, spec)
    _assert_vectorized_equivalent(loop_cache, batch_cache)
    _assert_vectorized_fetch_face(loop_cache, batch_cache)


# ---------------------------------------------------------------------------
# UT-2003: ChunkedActivationCache batched equivalence
# ---------------------------------------------------------------------------


def _cloned_spec(spec: _Spec) -> _Spec:
    """Clone every activation tensor (chunked ``put`` stores by reference)."""
    return [
        (
            branch_id,
            layer_idx,
            {name: tensor.clone() for name, tensor in acts.items()},
            step_idx,
        )
        for branch_id, layer_idx, acts, step_idx in spec
    ]


@pytest.mark.parametrize("device", _DEVICES)
def test_ut2003_fixed_scenario_equivalence(device: str) -> None:
    """UT-2003: chunked put_batch equivalence (chunk data/order/tokens).

    ``chunk_size=4`` with sequence lengths 10/6/9 exercises the
    cross-chunk-boundary splits; ``max_chunks=2`` forces per-insert
    eviction. The loop reference is fed cloned tensors so the chunk-data
    comparison never compares a tensor with itself (chunked ``put`` stores
    ``.contiguous()`` results by reference).
    """
    spec = _fixed_spec_2003(device)
    loop_cache = ChunkedActivationCache(
        max_entries_per_layer=8, chunk_size=4, device=device
    )
    _run_loop(loop_cache, _cloned_spec(spec))
    batch_cache = ChunkedActivationCache(
        max_entries_per_layer=8, chunk_size=4, device=device
    )
    _run_batch(batch_cache, spec)
    _assert_chunked_equivalent(loop_cache, batch_cache)
    _assert_chunked_fetch_face(loop_cache, batch_cache)


@pytest.mark.parametrize("device", _DEVICES)
def test_ut2003_randomized_property_equivalence(device: str) -> None:
    """UT-2003: seeded randomized chunked property test (chunk_size=4 budget)."""
    spec = _random_spec(device, seed=2003)
    loop_cache = ChunkedActivationCache(
        max_entries_per_layer=8, chunk_size=4, device=device
    )
    _run_loop(loop_cache, _cloned_spec(spec))
    batch_cache = ChunkedActivationCache(
        max_entries_per_layer=8, chunk_size=4, device=device
    )
    _run_batch(batch_cache, spec)
    _assert_chunked_equivalent(loop_cache, batch_cache)
    _assert_chunked_fetch_face(loop_cache, batch_cache)


# ---------------------------------------------------------------------------
# UT-2004: batch-level atomicity (zero writes on any invalid entry)
# ---------------------------------------------------------------------------


def _invalid_entry(
    kind: str, device: str
) -> tuple[str, int, dict[str, torch.Tensor], int]:
    """Build one invalid entry of the given kind (design §4.2.2(2))."""
    if kind == "empty":
        return ("bad", 0, {}, 0)
    if kind == "ndim":
        return ("bad", 0, {"a": torch.randn(8, device=device)}, 0)
    return (
        "bad",
        0,
        {
            "a": torch.randn(1, 4, 8, device=device),
            "b": torch.randn(2, 3, 8, device=device),
        },
        0,
    )


@pytest.mark.parametrize("position", ["first", "middle", "last"])
@pytest.mark.parametrize("invalid_kind", ["empty", "ndim", "leading"])
@pytest.mark.parametrize("cache_kind", ["activation", "vectorized", "chunked"])
def test_ut2004_atomicity_zero_writes(
    cache_kind: str,
    invalid_kind: str,
    position: str,
) -> None:
    """UT-2004: an invalid entry anywhere -> ValueError, zero partial writes.

    The cache is pre-populated (layers 0/1), the batch carries seven valid
    entries (spanning layers 0/1/2 -- a leaky implementation would even
    create the layer-2 container) plus one invalid entry at the first,
    middle or last position. ``put_batch`` must raise ``ValueError`` before
    any mutation: the FULL internal state (every container, every buffer
    byte, every key order) deep-compares equal to the pre-call snapshot.
    The empty-dict middle case additionally pins the message contract: the
    entry index must appear in the message. Red face: ``put_batch`` /
    ``CachePutEntry`` do not exist yet (AttributeError / ImportError --
    neither is a ValueError, so ``pytest.raises(ValueError)`` fails through
    with the production-missing error).
    """
    device = "cpu"
    cache = _make_cache_by_kind(cache_kind, device)
    cache.put("pre0", 0, {"a": torch.randn(1, 4, 8, device=device)})
    cache.put("pre1", 1, {"a": torch.randn(1, 5, 8, device=device)}, step_idx=1)
    valid: _Spec = [
        (f"atom{i}", i % 3, {"a": torch.randn(1, 4, 8, device=device)}, i % 2)
        for i in range(7)
    ]
    index = {"first": 0, "middle": 3, "last": 7}[position]
    spec = list(valid)
    spec.insert(index, _invalid_entry(invalid_kind, device))
    snapshot = _snapshot_cache_state(cache_kind, cache)
    with pytest.raises(ValueError) as excinfo:
        _run_batch(cache, spec)
    if invalid_kind == "empty" and position == "middle":
        assert str(index) in str(excinfo.value), (
            f"the ValueError message must carry the entry index {index}; "
            f"got: {excinfo.value}"
        )
    _assert_cache_state_unchanged(cache_kind, cache, snapshot)


# ---------------------------------------------------------------------------
# UT-2005: ring and grow paths, overlapping repeated destinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cache_kind", ["activation", "vectorized"])
@pytest.mark.parametrize("device", _DEVICES)
def test_ut2005_ring_grow_overlap_equivalence(device: str, cache_kind: str) -> None:
    """UT-2005: ring entries, buffer grow and overlapping-dst batches.

    The ``overlap_grow_flush_order`` and ``ring_repeat_flush_order`` rows
    are order-sensitive within the comparison face: a batch path that wrote
    the second occurrence of a logical destination BEFORE flushing its
    first-occurrence pair would leave the first entry's rows visible in
    the final state and fail the face (design §4.2.2(1) flush semantics).
    """
    torch.manual_seed(2105)

    def _t(seq: int, batch: int = 1) -> torch.Tensor:
        return torch.randn(batch, seq, 4, device=device)

    scenarios: list[tuple[str, int, _Spec]] = [
        (
            "ring",
            4,
            [("r", 0, {"a": _t(9)}, 0), ("r2", 0, {"a": _t(7)}, 1)],
        ),
        (
            "grow",
            16,
            [
                ("g", 0, {"a": _t(2)}, 0),
                ("g", 0, {"a": _t(6)}, 0),
                ("g", 0, {"a": _t(3)}, 0),
            ],
        ),
        (
            "mixed_simple_ring",
            6,
            [
                ("m1", 0, {"a": _t(3)}, 0),
                ("m2", 0, {"a": _t(9)}, 0),
                ("m1", 1, {"a": _t(4)}, 0),
            ],
        ),
        (
            "overlap_grow_flush_order",
            16,
            [("o", 0, {"a": _t(3)}, 0), ("o", 0, {"a": _t(10)}, 0)],
        ),
        (
            "overlap_shrink",
            16,
            [("s", 0, {"a": _t(8)}, 0), ("s", 0, {"a": _t(4)}, 0)],
        ),
        (
            "ring_repeat_flush_order",
            4,
            [("q", 0, {"a": _t(9)}, 0), ("q", 0, {"a": _t(9)}, 0)],
        ),
        (
            "multi_name_overlap",
            16,
            [
                ("n", 0, {"a": _t(4), "b": _t(4)}, 0),
                ("n", 0, {"a": _t(6), "b": _t(6)}, 0),
            ],
        ),
    ]
    for name, budget, spec in scenarios:
        loop_cache = _make_cache_by_kind(cache_kind, device, budget)
        _run_loop(loop_cache, spec)
        batch_cache = _make_cache_by_kind(cache_kind, device, budget)
        _run_batch(batch_cache, spec)
        if cache_kind == "activation":
            _assert_activation_equivalent(loop_cache, batch_cache)
            _assert_activation_fetch_face(loop_cache, batch_cache)
        else:
            _assert_vectorized_equivalent(loop_cache, batch_cache)
            _assert_vectorized_fetch_face(loop_cache, batch_cache)


# ---------------------------------------------------------------------------
# Publish-path spy caches (UT-2006/2007/2008/2009 CUDA faces)
# ---------------------------------------------------------------------------


class _SpyCapableCache(ActivationCache):
    """Batch-capable spy: records put/put_batch, delegates to the real impl.

    ``put_batch`` delegates to the (Green) production implementation, so the
    publish's batched semantics are exercised for real while the calls and
    payloads are recorded. In the Red run the publish never calls it (no
    batched branch), which is exactly the red face of the dispatch tests.
    """

    def __init__(self, max_entries_per_layer: int, device: str) -> None:
        super().__init__(max_entries_per_layer=max_entries_per_layer, device=device)
        self.batch_calls: list[list[Any]] = []
        self.put_calls: list[tuple[str, int, int, tuple[str, ...]]] = []

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        self.put_calls.append((branch_id, layer_idx, step_idx, tuple(activations)))
        super().put(branch_id, layer_idx, activations, step_idx=step_idx)

    def put_batch(self, entries: Sequence[Any]) -> None:
        self.batch_calls.append(list(entries))
        super().put_batch(entries)


class _RaisingCapableCache(ActivationCache):
    """Batch-capable cache whose put_batch always raises (exception face)."""

    def put_batch(self, entries: Sequence[Any]) -> None:
        raise RuntimeError("ar020-put-batch-boom")


class _IncapableCacheWrapper:
    """Delegating wrapper exposing every legacy method EXCEPT ``put_batch``.

    Every legacy method is delegated explicitly (no ``__getattr__``): a
    ``runtime_checkable`` isinstance probe uses ``hasattr``, so attribute
    fallback delegation would fake the capability whenever the inner cache
    grows ``put_batch``.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.put_calls: list[tuple[str, int, int, tuple[str, ...]]] = []

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        self.put_calls.append((branch_id, layer_idx, step_idx, tuple(activations)))
        self.inner.put(branch_id, layer_idx, activations, step_idx=step_idx)

    def fetch(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        return self.inner.fetch(
            branch_id=branch_id, layer_idx=layer_idx, step_idx=step_idx
        )

    def fetch_masked(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        return self.inner.fetch_masked(branch_id, layer_idx, token_mask, step_idx)

    def get(
        self,
        branch_id: str,
        layer_idx: int,
        token_mask: torch.Tensor,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        return self.inner.get(branch_id, layer_idx, token_mask, step_idx)

    def get_all(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        return self.inner.get_all(branch_id, layer_idx, step_idx)

    def contains(
        self,
        branch_id: str,
        layer_idx: int,
        step_idx: int = 0,
    ) -> bool:
        return self.inner.contains(branch_id, layer_idx, step_idx)

    def clear_branch(self, branch_id: str) -> None:
        self.inner.clear_branch(branch_id)

    def clear_layer(self, layer_idx: int) -> None:
        self.inner.clear_layer(layer_idx)

    def clear_all(self) -> None:
        self.inner.clear_all()

    def num_entries(self, layer_idx: int | None = None) -> int:
        return self.inner.num_entries(layer_idx)


# ---------------------------------------------------------------------------
# UT-2006: capability protocol + publish dispatch
# ---------------------------------------------------------------------------


def test_ut2006_protocol_and_isinstance() -> None:
    """UT-2006① + interface face (design §4.3/§6.2).

    ``CachePutEntry`` is a NamedTuple whose field order mirrors ``put``'s
    parameter order (positional compatibility) with ``step_idx`` defaulting
    to 0; ``BatchedPutCapable`` is runtime-checkable and the three caches
    satisfy it structurally while the delegating wrapper (and a bare
    object) does not. ``put_batch`` returns ``None`` and an empty batch is
    a legal no-op. Red face: the protocol names do not exist yet
    (ImportError).
    """
    from actfold.core.cache_protocol import BatchedPutCapable, CachePutEntry

    assert CachePutEntry._fields == (
        "branch_id",
        "layer_idx",
        "activations",
        "step_idx",
    )
    activations = {"a": torch.randn(1, 3, 4)}
    entry = CachePutEntry("b", 1, activations, 3)
    assert entry.branch_id == "b"
    assert entry.layer_idx == 1
    assert entry.activations is activations
    assert entry.step_idx == 3
    assert CachePutEntry("b", 1, activations).step_idx == 0

    caches = [
        ActivationCache(max_entries_per_layer=16, device="cpu"),
        ChunkedActivationCache(max_entries_per_layer=16, device="cpu"),
        VectorizedActivationCache(max_entries_per_layer=16, device="cpu"),
    ]
    for cache in caches:
        assert isinstance(cache, BatchedPutCapable), (
            f"{type(cache).__name__} must satisfy BatchedPutCapable"
        )
    assert not isinstance(
        _IncapableCacheWrapper(
            ActivationCache(max_entries_per_layer=16, device="cpu")
        ),
        BatchedPutCapable,
    )
    assert not isinstance(object(), BatchedPutCapable)

    for cache in caches:
        assert cache.put_batch([]) is None
        assert (
            cache.put_batch([CachePutEntry("solo", 0, {"a": torch.randn(1, 2, 4)})])
            is None
        )
        assert cache.contains("solo", 0)


@_cuda
def test_ut2006_publish_dispatch_batched_payload() -> None:
    """UT-2006③: capable cache -> one put_batch with loop-identical payload.

    Every published entry must carry the snapshot row view (values equal to
    the static ``child_buf`` row but NOT aliasing its storage), layer-0
    entries additionally the embedding row view, and the branch/layer/step
    fields of the loop variant; the batch branch must not fall back to
    per-entry ``put``. Red face: the publish has no batched branch, so the
    spy observes zero ``put_batch`` calls (production missing).
    """
    device = "cuda"
    model, profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4306)
    spy = _SpyCapableCache(_CACHE_BUDGET, device)
    mff = _make_mff(model, device, cache=spy, use_cuda_graph=True)
    _serial(mff, parent, branch_id="2006p")
    spy.put_calls.clear()
    branch_ids = _ids("2006a")
    _batched(mff, stacked, branch_ids, "2006p")
    runner = mff.graph_runner_batched
    assert runner is not None

    assert len(spy.batch_calls) == 1, (
        f"the publish must call put_batch exactly once for a capable cache; "
        f"observed {len(spy.batch_calls)} calls (batched branch missing)"
    )
    assert not spy.put_calls, "the batch branch must not call per-entry put"
    entries = spy.batch_calls[0]
    assert len(entries) == _NUM_LAYERS * _NUM_CANDIDATES, (
        f"put_batch must receive L*N={_NUM_LAYERS * _NUM_CANDIDATES} entries; "
        f"observed {len(entries)}"
    )
    with torch.no_grad():
        embedding_rows = profile.embed_module(stacked)
    seen: set[tuple[str, int]] = set()
    for entry in entries:
        layer_idx = entry.layer_idx
        row = branch_ids.index(entry.branch_id)
        assert (entry.branch_id, layer_idx) not in seen
        seen.add((entry.branch_id, layer_idx))
        assert entry.step_idx == 0, (
            "the entry must mirror the loop variant's put call, which omits "
            "step_idx: the folded stack stores every activation at step 0 "
            "(eager layers, both graph publishes, and all parent fetches), "
            "so a step-threaded batch entry would orphan the published "
            "children from every step-0 consumer"
        )
        ffn = entry.activations["ffn_out"]
        static_row = runner.child_buf[layer_idx][row : row + 1]
        assert torch.equal(ffn, static_row), (
            f"ffn_out payload must equal the child_buf row for "
            f"{(entry.branch_id, layer_idx)}"
        )
        assert ffn.data_ptr() != static_row.data_ptr(), (
            "ffn_out must be a snapshot row view, not a static-buffer alias"
        )
        if layer_idx == 0:
            assert set(entry.activations) == {"ffn_out", "embedding"}
            assert torch.equal(
                entry.activations["embedding"], embedding_rows[row : row + 1]
            )
        else:
            assert set(entry.activations) == {"ffn_out"}
    assert len(seen) == _NUM_LAYERS * _NUM_CANDIDATES

    for row, branch_id in enumerate(branch_ids):
        for layer_idx in range(_NUM_LAYERS):
            stored = spy.fetch(branch_id=branch_id, layer_idx=layer_idx)
            assert torch.equal(
                stored["ffn_out"], runner.child_buf[layer_idx][row : row + 1]
            ), f"the delegated put_batch must populate {branch_id} layer {layer_idx}"


@_cuda
def test_ut2006_publish_incapable_fallback_loop() -> None:
    """UT-2006② (double-sided, green in the Red run): incapable -> original loop.

    A wrapper cache without ``put_batch`` must keep the publish on the
    original double loop: exactly L*N ``put`` calls whose payloads match the
    loop shape (two names at layer 0, one elsewhere) and whose stored
    content equals the static child buffer rows.
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4326)
    inner = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    wrapper = _IncapableCacheWrapper(inner)
    mff = _make_mff(model, device, cache=wrapper, use_cuda_graph=True)
    _serial(mff, parent, branch_id="2006fp")
    wrapper.put_calls.clear()
    branch_ids = _ids("2006fa")
    _batched(mff, stacked, branch_ids, "2006fp")
    runner = mff.graph_runner_batched
    assert runner is not None

    assert len(wrapper.put_calls) == _NUM_LAYERS * _NUM_CANDIDATES, (
        f"the incapable fallback must put L*N={_NUM_LAYERS * _NUM_CANDIDATES} "
        f"entries; observed {len(wrapper.put_calls)}"
    )
    layer0_calls = [call for call in wrapper.put_calls if call[1] == 0]
    assert len(layer0_calls) == _NUM_CANDIDATES
    for _branch_id, layer_idx, _step, names in wrapper.put_calls:
        if layer_idx == 0:
            assert set(names) == {"ffn_out", "embedding"}
        else:
            assert set(names) == {"ffn_out"}
    for row, branch_id in enumerate(branch_ids):
        for layer_idx in range(_NUM_LAYERS):
            stored = inner.fetch(branch_id=branch_id, layer_idx=layer_idx)
            assert torch.equal(
                stored["ffn_out"], runner.child_buf[layer_idx][row : row + 1]
            )


@_cuda
def test_ut2006_probe_each_publish_and_cache_swap() -> None:
    """UT-2006④: the capability probe re-runs on EVERY publish.

    Swapping ``mff.cache`` between capable spies and an incapable wrapper
    must switch the publish branch on the very next publish (the probe is
    never cached on the MFF instance). The parent is re-serialised into the
    freshly swapped cache before each replay so the replay-time parent
    validation passes. Red face: the first capable spy never observes a
    ``put_batch`` call (batched branch missing).
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4336)

    spy_a = _SpyCapableCache(_CACHE_BUDGET, device)
    mff = _make_mff(model, device, cache=spy_a, use_cuda_graph=True)
    _serial(mff, parent, branch_id="2006swp")
    _batched(mff, stacked, _ids("2006s1"), "2006swp")
    assert len(spy_a.batch_calls) == 1, (
        "the first publish must dispatch to spy_a.put_batch "
        "(batched branch missing)"
    )

    spy_b = _SpyCapableCache(_CACHE_BUDGET, device)
    mff.cache = spy_b
    _serial(mff, parent, branch_id="2006swp")
    _batched(mff, stacked, _ids("2006s2"), "2006swp")
    assert len(spy_b.batch_calls) == 1, "the swapped-in cache must be re-probed"
    assert len(spy_a.batch_calls) == 1, "the swapped-out cache must not be touched"

    wrapper = _IncapableCacheWrapper(
        ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    )
    mff.cache = wrapper
    _serial(mff, parent, branch_id="2006swp")
    wrapper.put_calls.clear()
    _batched(mff, stacked, _ids("2006s3"), "2006swp")
    assert len(wrapper.put_calls) == _NUM_LAYERS * _NUM_CANDIDATES, (
        "the incapable wrapper must fall back to the per-entry loop"
    )
    assert len(spy_b.batch_calls) == 1

    spy_c = _SpyCapableCache(_CACHE_BUDGET, device)
    mff.cache = spy_c
    _serial(mff, parent, branch_id="2006swp")
    _batched(mff, stacked, _ids("2006s4"), "2006swp")
    assert len(spy_c.batch_calls) == 1, (
        "switching back to a capable cache must restore the batched branch"
    )


@_cuda
def test_ut2006_profiler_face_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2006⑤: record_precomputed count/parameters identical to the loop.

    The batched publish (capable spy cache) and the loop publish (incapable
    wrapper) run the same replay workload; the recorded
    ``record_precomputed`` faces must agree call-for-call (compared
    order-agnostically per (layer, branch): layer_idx, step_idx, parent,
    num_tokens, tau, metric and the stable_sum value). The profiler face is
    double-sided in the Red run; the red value is carried by the
    put_batch-dispatch check.
    """
    calls: list[dict[str, Any]] = []
    original = GLOBAL_STABILITY_PROFILER.record_precomputed

    def _spy_record(**kwargs: Any) -> None:
        calls.append(kwargs)
        original(**kwargs)

    monkeypatch.setattr(GLOBAL_STABILITY_PROFILER, "record_precomputed", _spy_record)
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4346)

    spy = _SpyCapableCache(_CACHE_BUDGET, device)
    mff_batch = _make_mff(model, device, cache=spy, use_cuda_graph=True)
    _serial(mff_batch, parent, branch_id="2006pp")
    _batched(mff_batch, stacked, _ids("2006pa"), "2006pp")
    assert len(spy.batch_calls) == 1, (
        "the batched publish must dispatch to put_batch (branch missing)"
    )
    batch_face = list(calls)
    calls.clear()

    wrapper = _IncapableCacheWrapper(
        ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    )
    mff_loop = _make_mff(model, device, cache=wrapper, use_cuda_graph=True)
    _serial(mff_loop, parent, branch_id="2006pp")
    _batched(mff_loop, stacked, _ids("2006pb"), "2006pp")
    loop_face = list(calls)

    expected = _NUM_LAYERS * _NUM_CANDIDATES
    assert len(batch_face) == expected, (
        f"the batched publish must record_precomputed L*N={expected} times; "
        f"observed {len(batch_face)}"
    )
    assert len(loop_face) == expected

    def _face_key(call: dict[str, Any]) -> tuple[int, str]:
        return (call["layer_idx"], call["branch_id"])

    batch_sorted = sorted(batch_face, key=_face_key)
    loop_sorted = sorted(loop_face, key=_face_key)
    for call_b, call_l in zip(batch_sorted, loop_sorted):
        assert call_b["layer_idx"] == call_l["layer_idx"]
        assert call_b["step_idx"] == call_l["step_idx"]
        assert call_b["parent_branch_id"] == call_l["parent_branch_id"]
        assert call_b["num_tokens"] == call_l["num_tokens"]
        assert call_b["tau"] == call_l["tau"]
        assert call_b["metric"] == call_l["metric"]
        assert call_b["stable_sum"].item() == call_l["stable_sum"].item(), (
            f"stable_sum mismatch at layer {call_b['layer_idx']} "
            f"for {call_b['branch_id']}"
        )


@_cuda
def test_ut2006_publish_exception_propagates() -> None:
    """UT-2006⑥: a raising put_batch propagates unchanged out of the publish.

    Red face: the publish has no batched branch, the raising ``put_batch``
    is never invoked and the forward completes without raising, so
    ``pytest.raises`` fails (production missing).
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4356)
    mff = _make_mff(
        model,
        device,
        cache=_RaisingCapableCache(_CACHE_BUDGET, device),
        use_cuda_graph=True,
    )
    _serial(mff, parent, branch_id="2006ep")
    with pytest.raises(RuntimeError, match="ar020-put-batch-boom"):
        _batched(mff, stacked, _ids("2006ea"), "2006ep")


# ---------------------------------------------------------------------------
# UT-2007: write_op_count host-call ledger
# ---------------------------------------------------------------------------


def test_ut2007_loop_put_ledger() -> None:
    """UT-2007 (loop face): the probe counts per-activation-name writes.

    ``write_op_count`` starts at 0; every ``put`` increments it by
    ``len(activations)`` regardless of the slice/ring path; the chunked
    cache counts only materialized ``.contiguous()`` results (batch-1
    contiguous chunk slices materialize nothing, batch-2 slices materialize
    one tensor per (chunk, name)). Red face: the attribute does not exist
    yet (AttributeError).
    """
    cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    assert cache.write_op_count == 0, "the probe must start at 0"
    cache.put("x", 0, {"a": torch.randn(1, 4, 8), "b": torch.randn(1, 4, 8)})
    assert cache.write_op_count == 2
    ring = ActivationCache(max_entries_per_layer=2, device="cpu")
    ring.put("r", 0, {"a": torch.randn(1, 5, 3)})
    assert ring.write_op_count == 1, "a ring put still counts one op per name"

    vec = VectorizedActivationCache(max_entries_per_layer=64, device="cpu")
    assert vec.write_op_count == 0
    vec.put("v", 0, {"a": torch.randn(1, 3, 4), "b": torch.randn(1, 3, 4)})
    assert vec.write_op_count == 2

    chunked = ChunkedActivationCache(
        max_entries_per_layer=64, chunk_size=4, device="cpu"
    )
    assert chunked.write_op_count == 0
    chunked.put("c", 0, {"a": torch.randn(1, 10, 4)})
    assert chunked.write_op_count == 0, (
        "batch-1 contiguous chunk slices store by reference (zero ops)"
    )
    chunked.put("c", 1, {"a": torch.randn(2, 10, 4)})
    assert chunked.write_op_count == 3, (
        "batch-2 chunk slices materialize one tensor per chunk (3 chunks)"
    )


def test_ut2007_put_batch_direct_ledger() -> None:
    """UT-2007 (direct batch face): groups + immediate writes.

    A uniform (device, dtype) batch of distinct destinations collapses to
    ONE foreach group; two dtypes give two groups; a ring entry and a
    repeated destination execute immediately (per-pair counts). The loop
    and batch probes differ for the same workload, pinning that the probe
    is not part of the equivalence face. Red face: ``CachePutEntry`` /
    ``put_batch`` do not exist yet (ImportError / AttributeError).
    """
    spec: _Spec = [
        (f"b{i}", 0, {"a": torch.randn(1, 3, 4), "b2": torch.randn(1, 3, 4)}, 0)
        for i in range(4)
    ]
    loop_cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    _run_loop(loop_cache, spec)
    assert loop_cache.write_op_count == 8
    batch_cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    _run_batch(batch_cache, spec)
    assert batch_cache.write_op_count == 1, (
        "a uniform batch of distinct destinations is one foreach group"
    )
    assert loop_cache.write_op_count != batch_cache.write_op_count

    mixed: _Spec = [
        ("m1", 0, {"a": torch.randn(1, 3, 4)}, 0),
        ("m2", 1, {"a": torch.randn(1, 3, 4, dtype=torch.float16)}, 0),
    ]
    mixed_cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    _run_batch(mixed_cache, mixed)
    assert mixed_cache.write_op_count == 2, "two dtypes give two groups"

    ring_cache = ActivationCache(max_entries_per_layer=2, device="cpu")
    _run_batch(ring_cache, [("r", 0, {"a": torch.randn(1, 5, 3)}, 0)])
    assert ring_cache.write_op_count == 1, "a ring entry executes immediately"

    repeat: _Spec = [
        ("q", 0, {"a": torch.randn(1, 3, 4)}, 0),
        ("q", 0, {"a": torch.randn(1, 3, 4)}, 0),
    ]
    repeat_cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    _run_batch(repeat_cache, repeat)
    assert repeat_cache.write_op_count == 2, (
        "a repeated destination flushes its first pair then runs the current "
        "pair immediately (two writes)"
    )


@_cuda
def test_ut2007_publish_ledger_loop_and_batch() -> None:
    """UT-2007 (publish face): loop delta == L*N+N; batch delta <= 2*(L+N).

    The loop publish performs L*N puts with L*N+N activation-name writes;
    the batched publish must stay within the frozen c=2 bound of L+N host
    write calls. Red face: ``write_op_count`` does not exist yet
    (AttributeError on the first read).
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4367)

    inner = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    wrapper = _IncapableCacheWrapper(inner)
    mff_loop = _make_mff(model, device, cache=wrapper, use_cuda_graph=True)
    _serial(mff_loop, parent, branch_id="2007p")
    loop_before = inner.write_op_count
    _batched(mff_loop, stacked, _ids("2007a"), "2007p")
    loop_delta = inner.write_op_count - loop_before
    assert loop_delta == _NUM_LAYERS * _NUM_CANDIDATES + _NUM_CANDIDATES, (
        f"the loop publish must account for L*N+N="
        f"{_NUM_LAYERS * _NUM_CANDIDATES + _NUM_CANDIDATES} name writes; "
        f"observed {loop_delta}"
    )

    spy = _SpyCapableCache(_CACHE_BUDGET, device)
    mff_batch = _make_mff(model, device, cache=spy, use_cuda_graph=True)
    _serial(mff_batch, parent, branch_id="2007p")
    batch_before = spy.write_op_count
    _batched(mff_batch, stacked, _ids("2007b"), "2007p")
    assert len(spy.batch_calls) == 1
    batch_delta = spy.write_op_count - batch_before
    bound = 2 * (_NUM_LAYERS + _NUM_CANDIDATES)
    assert 1 <= batch_delta <= bound, (
        f"the batched publish must cost at most 2*(L+N)={bound} host write "
        f"calls (c=2 frozen); observed {batch_delta}"
    )


# ---------------------------------------------------------------------------
# UT-2008: zero host-sync + freshness in the batched publish window
# ---------------------------------------------------------------------------


@_cuda
def test_ut2008_zero_host_sync_and_freshness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-2008: the batched publish stays sync-free and snapshot-fresh.

    The publish window must contain zero ``.item()``/``.tolist()``/``.cpu()``
    calls (double-sided: AR018 already delivers this on the loop path) and
    must dispatch to ``put_batch`` (red face: zero observed calls). After
    the publish, an in-place ``runner.child_buf`` mutation must leave every
    published cache entry bit-identical (the snapshot's freshness makes the
    by-reference row views safe).
    """
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4378)
    spy = _SpyCapableCache(_CACHE_BUDGET, device)
    mff = _make_mff(model, device, cache=spy, use_cuda_graph=True)
    windows = _windowed_publish_spy(monkeypatch, mff)
    _serial(mff, parent, branch_id="2008p")
    branch_ids = _ids("2008a")
    _batched(mff, stacked, branch_ids, "2008p")
    assert len(windows) == 1
    assert windows[0]["item"] == 0, "zero .item() host syncs in the publish window"
    assert windows[0]["tolist"] == 0, "zero .tolist() host syncs in the window"
    assert windows[0]["cpu"] == 0, "zero .cpu() host syncs in the publish window"
    assert len(spy.batch_calls) == 1, (
        "the publish must dispatch to put_batch for the capable spy cache "
        "(batched branch missing)"
    )
    runner = mff.graph_runner_batched
    assert runner is not None

    snapshot: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    for branch_id in branch_ids:
        for layer_idx in range(_NUM_LAYERS):
            entry = spy.fetch(branch_id=branch_id, layer_idx=layer_idx)
            snapshot[(branch_id, layer_idx)] = {
                name: tensor.clone() for name, tensor in entry.items()
            }
    with torch.no_grad():
        runner.child_buf.add_(123.0)
    for branch_id in branch_ids:
        for layer_idx in range(_NUM_LAYERS):
            entry = spy.fetch(branch_id=branch_id, layer_idx=layer_idx)
            for name, tensor in entry.items():
                assert torch.equal(tensor, snapshot[(branch_id, layer_idx)][name]), (
                    f"mutating child_buf must not touch the published entry "
                    f"{(branch_id, layer_idx, name)}"
                )


@_cuda
def test_ut2008_debug_path_per_row_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2008 (debug face): debug_enabled keeps the per-row record entries.

    With ``debug_enabled`` the publish must keep calling the per-row
    ``record`` entry (the only ``divergence_positions`` producer) L*N times
    with bool ``[1, T]`` masks while still dispatching the cache writes to
    ``put_batch`` (red face: zero observed calls). The record count is
    double-sided in the Red run (the loop publish also records per row).
    """
    record_calls: list[dict[str, Any]] = []
    original = GLOBAL_STABILITY_PROFILER.record

    def _spy_record(**kwargs: Any) -> None:
        record_calls.append(kwargs)
        original(**kwargs)

    monkeypatch.setattr(GLOBAL_STABILITY_PROFILER, "record", _spy_record)
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4388)
    spy = _SpyCapableCache(_CACHE_BUDGET, device)
    mff = _make_mff(model, device, cache=spy, use_cuda_graph=True)
    _serial(mff, parent, branch_id="2008dp")
    record_calls.clear()
    GLOBAL_STABILITY_PROFILER.debug_enabled = True
    try:
        _batched(mff, stacked, _ids("2008da"), "2008dp")
    finally:
        GLOBAL_STABILITY_PROFILER.debug_enabled = False
    assert len(spy.batch_calls) == 1, (
        "the debug publish must still dispatch to put_batch "
        "(batched branch missing)"
    )
    assert len(record_calls) == _NUM_LAYERS * _NUM_CANDIDATES, (
        f"debug_enabled must keep the per-row record face "
        f"({_NUM_LAYERS * _NUM_CANDIDATES} calls); "
        f"observed {len(record_calls)}"
    )
    for call in record_calls:
        mask = call["stable_mask"]
        assert mask.dtype is torch.bool
        assert tuple(mask.shape) == (1, _SEQ)


# ---------------------------------------------------------------------------
# UT-2009: foreach-unavailable downgrade
# ---------------------------------------------------------------------------


def test_ut2009_foreach_downgrade_equivalence_and_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-2009: ``_FOREACH_AVAILABLE = False`` downgrades copy_pairs per-pair.

    With the module flag forced off, ``put_batch`` must stay
    loop-equivalent (the downgrade is a mechanism-level fallback: per-pair
    ``copy_`` inside each group) and the ledger must count every pair
    (3 entries x 2 names = 6). Red face: ``actfold.core.batched_write``
    does not exist yet (ImportError).
    """
    import actfold.core.batched_write as batched_write

    monkeypatch.setattr(batched_write, "_FOREACH_AVAILABLE", False)
    spec: _Spec = [
        (
            f"d{i}",
            i % 2,
            {"a": torch.randn(1, 4, 5), "b": torch.randn(1, 4, 5)},
            i % 3,
        )
        for i in range(3)
    ]
    loop_cache = ActivationCache(max_entries_per_layer=64, device="cpu")
    _run_loop(loop_cache, spec)
    downgraded = ActivationCache(max_entries_per_layer=64, device="cpu")
    _run_batch(downgraded, spec)
    _assert_activation_equivalent(loop_cache, downgraded)
    _assert_activation_fetch_face(loop_cache, downgraded)
    assert downgraded.write_op_count == 6, (
        "the per-pair downgrade must count one write per (entry, name) pair"
    )


@_cuda
def test_ut2009_publish_downgrade_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2009: the publish dispatch is unaffected by the downgrade flag.

    With ``_FOREACH_AVAILABLE`` forced off the capable spy cache must still
    receive exactly one ``put_batch`` call with L*N entries from the
    publish. Red face: ``actfold.core.batched_write`` does not exist yet
    (ImportError).
    """
    import actfold.core.batched_write as batched_write

    monkeypatch.setattr(batched_write, "_FOREACH_AVAILABLE", False)
    device = "cuda"
    model, _profile = _build_model(device)
    parent, stacked = _make_candidates(device, seed=4399)
    spy = _SpyCapableCache(_CACHE_BUDGET, device)
    mff = _make_mff(model, device, cache=spy, use_cuda_graph=True)
    _serial(mff, parent, branch_id="2009p")
    _batched(mff, stacked, _ids("2009a"), "2009p")
    assert len(spy.batch_calls) == 1, (
        "the publish dispatch must not depend on the foreach availability flag"
    )
    assert len(spy.batch_calls[0]) == _NUM_LAYERS * _NUM_CANDIDATES
