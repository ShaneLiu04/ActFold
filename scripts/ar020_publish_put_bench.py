"""AR020/T003: publish-path cache-put batching benchmark (real checkpoint).

Measures the AR020 ``put_batch`` capability on the Fast-dLLM-v2-1.5B
block-verification workload (the AR019 benchmark's configs, workload
builder, and folded-stack construction are reused verbatim) at three
layers:

1. **put microbenchmark** -- real-shape entries (bf16 ``[1, T, H]`` row
   views plus the layer-0 embedding row) written through sequential
   ``ActivationCache.put`` calls vs one ``put_batch`` call. Entry
   construction stays OUTSIDE the timed region (this layer isolates the
   cache write cost itself); the ``write_op_count`` ledger is read
   around every timed call.
2. **publish-window microbenchmark** -- one untimed ``runner.replay``
   repopulates the static child buffer, then
   ``ManualFoldedForward._publish_graph_children`` alone is timed
   (sync-bounded host ``perf_counter``; the window naturally includes
   the batched-entry construction -- H2 is judged at this layer, per the
   frozen design). A/B is realized by swapping ``mff.cache`` between two
   symmetric wrappers (the capability probe runs on EVERY publish, so a
   swap re-probes): ``loop`` wraps an ``ActivationCache`` delegating
   every method but never exposing ``put_batch`` (the production loop
   publish), ``batch`` adds a pass-through ``put_batch``. Both wrappers
   add the same one-hop Python delegation to every call.
3. **end-to-end macrobenchmark** -- baseline / fixed / adaptive(loop) /
   adaptive(batch) step wall-clocks, round-robin interleaved (AR019
   timing family; the two adaptive paths differ ONLY in the cache
   wrapper; ``fixed`` keeps the plain ``ActivationCache``, i.e. the
   production default with the batched publish active).

Correctness gate (H1, real-model face): after one replay the loop
variant publishes to its check branches and the batch variant to its
own; every fetched ``(branch, layer, name)`` tensor pair must be
bit-equal. The UT face (``tests/test_batched_put.py``, 56 cases) is the
other H1 component and is recorded as provenance.

Hypotheses (frozen in srs 3.3 BEFORE measurement; sigma =
max(sigma_loop, sigma_batch) throughout):

* H1 -- put_batch is bit-equivalent to sequential put (UT face + this
  run's publish gate). supported <=> all bit-equal, else refuted.
* H2 -- publish-window p50: batch beats loop. supported <=>
  p50(loop) - p50(batch) > 2*sigma; refuted <=> p50(batch) - p50(loop)
  > 2*sigma; inconclusive <=> |dp50| <= 2*sigma.
* H3 -- end-to-end adaptive: batch never falls back and the minimal
  config strictly improves. supported <=> no fallback everywhere AND
  p50(batch) <= p50(loop) + 2*sigma on every config AND
  p50(loop) - p50(batch) > 2*sigma on the minimal config; refuted <=>
  any config regresses beyond 2*sigma; inconclusive otherwise.
* H4 -- host-call ledger: put-side copy host calls drop from L*N+N to
  O(L+N) (``write_op_count``; CUPTI is null on this host, #39, so the
  ledger is the structural evidence, never a kernel-launch count).
  supported <=> batch ledger <= c*(L+N) with c = 2 frozen at design
  time and written into the artifact meta before the run.

Pre-registered single-run discipline (#39 lineage): reps=30, one run,
no rerun fishing; WDDM noise swamping verdicts honestly as
``inconclusive``. Per-config failures (capture/OOM) are recorded with
their reason, never fabricated.

Run (AR019 venv, transformers==4.53.1)::

    python -m scripts.ar020_publish_put_bench [--quick] [--reps 30]
        [--warmup 8] [--model-path PATH] [--out PATH]

``--model-path`` defaults to the ``FASTDLLM_MODEL_PATH`` environment
variable; neither present raises ``FileNotFoundError`` (AGENTS #10: no
synthetic-model fallback for a real-checkpoint benchmark).

Import-order discipline (design 4.2.2(5)): importing
``scripts.ar019_real_e2e_bench`` applies the HF endpoint/cache
environment at module import time and imports transformers FIRST; this
module imports it before any of its own torch/transformers-dependent
imports so the remote-code checkpoint environment is identical.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import torch  # noqa: E402

from actfold.core import ActivationCache, SimilarityGate  # noqa: E402
from actfold.core.adaptive_capacity import AdaptiveCapacityConfig  # noqa: E402
from actfold.core.batched_write import _FOREACH_AVAILABLE  # noqa: E402
from actfold.core.cache_protocol import CachePutEntry  # noqa: E402
from actfold.models.architecture_utils import ManualFoldedForward  # noqa: E402
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER  # noqa: E402
from scripts._hf_env import add_hf_env_arguments  # noqa: E402
from scripts.algo_experiments import stats_from_samples  # noqa: E402
from scripts.ar019_real_e2e_bench import (  # noqa: E402  (import order!)
    _CONFIGS,
    BLOCK,
    CAPACITY_RATIO,
    EPS,
    MASK_ID,
    SEED,
    TAU,
    TOPK,
    _build_context,
    _cuda_profiler_supported,
    _gpu_driver_version,
    _load_model,
    _make_mff,
    _make_workload,
    _overall,
    _resolve_model_path,
    _timed_ms,
)

DEFAULT_OUTPUT_PATH = Path("results/optimization/ar020_publish_put_bench.json")
DEFAULT_MODEL_ENV = "FASTDLLM_MODEL_PATH"

CACHE_BUDGET = 8192
C_CONSTANT = 2
MIN_CONFIG = (256, 4)

_UT_FACE = {
    "file": "tests/test_batched_put.py",
    "cases": 56,
    "result": "56 passed (T002 green; full regression 920 passed / 3 skipped)",
    "note": "UT face of H1 (three caches x CPU/CUDA, randomized property "
    "tests, publish dispatch, ledger, foreach-downgrade); recorded as "
    "provenance from the committed suite, not re-measured in this run",
}

_VERDICT_FORMULAS = {
    "H1": "put_batch bit-equivalent to sequential put (UT face + real-model "
    "publish gate); supported <=> all bit-equal; otherwise refuted (no "
    "inconclusive)",
    "H2": "publish-window p50: batch beats loop; supported <=> p50(loop) - "
    "p50(batch) > 2*sigma; refuted <=> p50(batch) - p50(loop) > 2*sigma; "
    "inconclusive <=> |dp50| <= 2*sigma; sigma = max(sigma_loop, sigma_batch)",
    "H3": "end-to-end adaptive: batch never falls back and the minimal "
    "config strictly improves; supported <=> no fallback everywhere AND "
    "p50(batch) <= p50(loop) + 2*sigma on every config AND p50(loop) - "
    "p50(batch) > 2*sigma on the minimal config; refuted <=> any config "
    "regresses beyond 2*sigma; inconclusive otherwise",
    "H4": "host-call ledger (write_op_count, structural evidence under "
    "CUPTI null, #39): batch ledger <= c*(L+N) with c = 2 frozen at design "
    "time; otherwise refuted; measurement object = ActivationCache",
    "aggregation": "H2/H3 overall = most conservative aggregation over "
    "per-config verdicts (any refuted -> refuted; all supported -> "
    "supported; otherwise inconclusive)",
}


# ---------------------------------------------------------------------------
# A/B wrapper caches (symmetric one-hop delegation, design 4.2.2(5))
# ---------------------------------------------------------------------------
class _NoBatchCache:
    """Loop variant: delegates every method, never exposes ``put_batch``.

    The capability probe (``isinstance(cache, BatchedPutCapable)``) is an
    attribute-existence check, so the blanket ``__getattr__`` delegation
    must explicitly raise for ``put_batch`` -- otherwise the inner
    ActivationCache's method would leak through and both variants would
    take the same branch.
    """

    def __init__(self, inner: ActivationCache) -> None:
        self._inner = inner
        self.put_calls = 0

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        self.put_calls += 1
        self._inner.put(branch_id, layer_idx, activations, step_idx)

    def __getattr__(self, name: str) -> Any:
        if name == "put_batch":
            raise AttributeError(name)
        return getattr(self._inner, name)


class _BatchCache:
    """Batch variant: same delegation hop plus a pass-through ``put_batch``."""

    def __init__(self, inner: ActivationCache) -> None:
        self._inner = inner
        self.put_calls = 0
        self.put_batch_calls = 0

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        self.put_calls += 1
        self._inner.put(branch_id, layer_idx, activations, step_idx)

    def put_batch(self, entries: Sequence[CachePutEntry]) -> None:
        self.put_batch_calls += 1
        self._inner.put_batch(entries)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _make_mff_with_cache(model: Any, cache: Any, adaptive: bool) -> ManualFoldedForward:
    """Build a graph-capture folded stack around a caller-supplied cache."""
    gate = SimilarityGate(tau=TAU, metric="cosine", eps=EPS)
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        split_layers=True,
        split_min_tokens=1,
        use_cuda_graph=True,
        graph_capacity_ratio=CAPACITY_RATIO,
        graph_adaptive_capacity=adaptive,
        graph_adaptive_config=AdaptiveCapacityConfig() if adaptive else None,
    )


def _h2_verdict(loop_p50: float, loop_std: float, batch_p50: float, batch_std: float) -> str:
    """Frozen H2 rule (sigma = max of the two stds, srs 3.3)."""
    sigma = max(loop_std, batch_std)
    if loop_p50 - batch_p50 > 2 * sigma:
        return "supported"
    if batch_p50 - loop_p50 > 2 * sigma:
        return "refuted"
    return "inconclusive"


# ---------------------------------------------------------------------------
# Layer 1: put microbenchmark
# ---------------------------------------------------------------------------
def _bench_put_micro(
    num_layers: int,
    hidden: int,
    seq_len: int,
    num_candidates: int,
    warmup: int,
    reps: int,
) -> dict[str, Any]:
    """Sequential put vs one put_batch on real-shape entries.

    Entries mirror the publish payload (per-row ``[1, T, H]`` bf16 ffn_out
    views, plus the embedding row at layer 0) and are built OUTSIDE the
    timed region. Repeated reps re-put the same destinations: after the
    first rep the cache reuses its buffers (in-place slice overwrite), so
    the timed face is the steady-state write cost with identical
    destination patterns on both variants.
    """
    src = torch.randn(
        num_layers,
        num_candidates,
        seq_len,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
    )
    emb = torch.randn(num_candidates, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
    entries: list[CachePutEntry] = []
    for layer_idx in range(num_layers):
        for row in range(num_candidates):
            acts: dict[str, torch.Tensor] = {"ffn_out": src[layer_idx][row : row + 1]}
            if layer_idx == 0:
                acts["embedding"] = emb[row : row + 1]
            entries.append(
                CachePutEntry(
                    branch_id=f"micro{row}",
                    layer_idx=layer_idx,
                    activations=acts,
                    step_idx=0,
                )
            )
    loop_cache = ActivationCache(max_entries_per_layer=CACHE_BUDGET, device="cuda")
    batch_cache = ActivationCache(max_entries_per_layer=CACHE_BUDGET, device="cuda")

    def run_loop() -> None:
        for entry in entries:
            loop_cache.put(
                entry.branch_id,
                entry.layer_idx,
                entry.activations,
                entry.step_idx,
            )

    def run_batch() -> None:
        batch_cache.put_batch(entries)

    samples: dict[str, list[float]] = {"loop": [], "batch": []}
    ledger: dict[str, list[int]] = {"loop": [], "batch": []}
    caches = {"loop": loop_cache, "batch": batch_cache}
    for i in range(warmup + reps):
        order = (
            (("loop", run_loop), ("batch", run_batch))
            if i % 2 == 0
            else (("batch", run_batch), ("loop", run_loop))
        )
        for name, fn in order:
            before = caches[name].write_op_count
            samples[name].append(_timed_ms(fn))
            ledger[name].append(caches[name].write_op_count - before)

    result = {
        "timings_ms": {k: stats_from_samples(v[warmup:]).as_dict() for k, v in samples.items()},
        "write_ops_p50": {
            "loop": int(statistics.median(ledger["loop"][warmup:])),
            "batch": int(statistics.median(ledger["batch"][warmup:])),
        },
        "entries": len(entries),
        "note": "entry construction outside the timed region; steady-state "
        "in-place overwrite (same destinations every rep); write_ops = "
        "write_op_count delta per timed call",
    }
    # Locals (source tensors, entry list, both caches) are released when
    # this function returns; the caching allocator is trimmed explicitly.
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Layer 2: publish-window microbenchmark + correctness gate
# ---------------------------------------------------------------------------
def _bench_publish_window(
    model: Any,
    seq_len: int,
    num_candidates: int,
    warmup: int,
    reps: int,
    mask: Any,
    pe: Any,
    parent_ids: Any,
    cands: Any,
    num_layers: int,
) -> dict[str, Any]:
    """Time ``_publish_graph_children`` alone, A/B by swapping ``mff.cache``.

    Script-layer private-face probe (AR018/19 precedent): one untimed
    ``runner.replay`` repopulates the static child buffer before every
    timed publish so the measured window is exactly the production
    publish (snapshot clone + entry construction + cache writes +
    profiler records). The two variants publish to the same branch ids
    every rep -- steady-state in-place overwrites on both sides.
    """
    loop_w = _NoBatchCache(ActivationCache(max_entries_per_layer=CACHE_BUDGET, device="cuda"))
    batch_w = _BatchCache(ActivationCache(max_entries_per_layer=CACHE_BUDGET, device="cuda"))
    mff = _make_mff_with_cache(model, loop_w, adaptive=False)
    pub_ids = [f"pub{i}" for i in range(num_candidates)]

    with torch.no_grad():
        # Parent prefill + capture + one loop publish (steady-state setup).
        mff(parent_ids, branch_id="parent", attention_mask=mask, position_embeddings=pe)
        mff(
            cands,
            branch_id=None,
            branch_ids=pub_ids,
            parent_branch_id="parent",
            attention_mask=mask,
            position_embeddings=pe,
        )
        # Swap to the batch wrapper: the property setter re-propagates the
        # cache to the wrapped layers and the runner, the parent is
        # re-serialized into the new inner cache, and the next child call
        # replays + publishes through put_batch (steady-state setup).
        mff.cache = batch_w
        mff(parent_ids, branch_id="parent", attention_mask=mask, position_embeddings=pe)
        mff(
            cands,
            branch_id=None,
            branch_ids=pub_ids,
            parent_branch_id="parent",
            attention_mask=mask,
            position_embeddings=pe,
        )
        runner = mff.graph_runner_batched
        if runner is None:
            raise RuntimeError(
                "publish-window benchmark: batched graph runner did not "
                "capture (recorded as a config failure, never fabricated)"
            )

        # Correctness gate (H1 real-model face, untimed): one replay, then
        # each variant publishes to its own check branches; every fetched
        # (branch, layer, name) pair must be bit-equal.
        chk_loop = [f"chk_l{i}" for i in range(num_candidates)]
        chk_batch = [f"chk_b{i}" for i in range(num_candidates)]
        mff.cache = loop_w
        runner.replay(cands, "parent", chk_loop[0])
        mff._publish_graph_children(runner, cands, chk_loop, "parent", 0)
        mff.cache = batch_w
        runner.replay(cands, "parent", chk_batch[0])
        mff._publish_graph_children(runner, cands, chk_batch, "parent", 0)
        checked = 0
        bit_equal = True
        for row in range(num_candidates):
            for layer_idx in range(num_layers):
                left = loop_w._inner.fetch(chk_loop[row], layer_idx)
                right = batch_w._inner.fetch(chk_batch[row], layer_idx)
                if set(left) != set(right):
                    bit_equal = False
                    continue
                for name in left:
                    checked += 1
                    if not torch.equal(left[name], right[name]):
                        bit_equal = False
        correctness_gate = {
            "branches": num_candidates,
            "layers": num_layers,
            "checked_tensors": checked,
            "bit_equal": bit_equal,
        }

        # Timed A/B: alternate the variant order every rep (round-robin
        # interleaving); the swap, the replay and the ledger reads stay
        # outside the timed region.
        samples: dict[str, list[float]] = {"loop": [], "batch": []}
        ledger: dict[str, list[int]] = {"loop": [], "batch": []}
        budget_flags: dict[str, list[bool]] = {"loop": [], "batch": []}
        wrappers: dict[str, Any] = {"loop": loop_w, "batch": batch_w}
        for i in range(warmup + reps):
            order = ("loop", "batch") if i % 2 == 0 else ("batch", "loop")
            for name in order:
                mff.cache = wrappers[name]
                runner.replay(cands, "parent", pub_ids[0])
                budget_flags[name].append(bool(runner.budget_exceeded))
                inner = wrappers[name]._inner
                before = inner.write_op_count
                elapsed = _timed_ms(
                    lambda: mff._publish_graph_children(runner, cands, pub_ids, "parent", 0)
                )
                samples[name].append(elapsed)
                ledger[name].append(inner.write_op_count - before)

    loop_stats = stats_from_samples(samples["loop"][warmup:])
    batch_stats = stats_from_samples(samples["batch"][warmup:])
    loop_ops = int(statistics.median(ledger["loop"][warmup:]))
    batch_ops = int(statistics.median(ledger["batch"][warmup:]))
    result = {
        "timings_ms": {
            "loop": loop_stats.as_dict(),
            "batch": batch_stats.as_dict(),
        },
        "write_ops_p50": {"loop": loop_ops, "batch": batch_ops},
        "budget_exceeded_replays": sum(sum(flags) for flags in budget_flags.values()),
        "batch_put_batch_calls": batch_w.put_batch_calls,
        "loop_put_calls": loop_w.put_calls,
        "sigma_ms": max(loop_stats.std, batch_stats.std),
        "h2_verdict": _h2_verdict(loop_stats.p50, loop_stats.std, batch_stats.p50, batch_stats.std),
        "h4_verdict": (
            "supported" if batch_ops <= C_CONSTANT * (num_layers + num_candidates) else "refuted"
        ),
        "correctness_gate": correctness_gate,
        "note": "one untimed replay repopulates the static child buffer "
        "before each timed publish; A/B via mff.cache swap (capability "
        "re-probed on every publish); steady-state in-place overwrite "
        "destinations",
    }
    # The folded stack, runner, wrappers and inner caches are all function
    # locals -- released on return; the allocator is trimmed explicitly.
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Layer 3: end-to-end macrobenchmark
# ---------------------------------------------------------------------------
def _bench_end_to_end(
    model: Any,
    seq_len: int,
    num_candidates: int,
    warmup: int,
    reps: int,
    mask: Any,
    pe: Any,
    parent_ids: Any,
    cands: Any,
) -> dict[str, Any]:
    """Four interleaved paths: baseline / fixed / adaptive(loop) / (batch).

    ``fixed`` reuses the AR019 builder (plain ActivationCache -- the
    production default, batched publish active); the two adaptive paths
    differ ONLY in the cache wrapper. The warmup covers the captures and
    the AR017 adaptation schedule (shrink patience 3 + throttle
    interval 2); canaries (budget flag, pe identity, publish dispatch
    counters) are read between timed steps, zero timing pollution.
    """
    mffs: dict[str, ManualFoldedForward] = {
        "graph_fixed": _make_mff(model, adaptive=False),
        "adaptive_loop": _make_mff_with_cache(
            model,
            _NoBatchCache(ActivationCache(max_entries_per_layer=CACHE_BUDGET, device="cuda")),
            adaptive=True,
        ),
        "adaptive_batch": _make_mff_with_cache(
            model,
            _BatchCache(ActivationCache(max_entries_per_layer=CACHE_BUDGET, device="cuda")),
            adaptive=True,
        ),
    }
    parents = {
        "graph_fixed": "p_fx",
        "adaptive_loop": "p_al",
        "adaptive_batch": "p_ab",
    }
    ids = {
        "graph_fixed": [f"fx{i}" for i in range(num_candidates)],
        "adaptive_loop": [f"al{i}" for i in range(num_candidates)],
        "adaptive_batch": [f"ab{i}" for i in range(num_candidates)],
    }
    with torch.no_grad():
        for key, mff in mffs.items():
            mff(parent_ids, branch_id=parents[key], attention_mask=mask, position_embeddings=pe)

    step_fns: dict[str, Callable[[], Any]] = {
        "baseline_eager": lambda: [model(cands[i : i + 1]) for i in range(num_candidates)],
        "graph_fixed": lambda: mffs["graph_fixed"](
            cands,
            branch_id=None,
            branch_ids=ids["graph_fixed"],
            parent_branch_id=parents["graph_fixed"],
            attention_mask=mask,
            position_embeddings=pe,
        ),
        "adaptive_loop": lambda: mffs["adaptive_loop"](
            cands,
            branch_id=None,
            branch_ids=ids["adaptive_loop"],
            parent_branch_id=parents["adaptive_loop"],
            attention_mask=mask,
            position_embeddings=pe,
        ),
        "adaptive_batch": lambda: mffs["adaptive_batch"](
            cands,
            branch_id=None,
            branch_ids=ids["adaptive_batch"],
            parent_branch_id=parents["adaptive_batch"],
            attention_mask=mask,
            position_embeddings=pe,
        ),
    }
    order = (
        "baseline_eager",
        "graph_fixed",
        "adaptive_loop",
        "adaptive_batch",
    )
    samples: dict[str, list[float]] = {name: [] for name in order}
    budget_flags: dict[str, list[bool]] = {name: [] for name in order if name != "baseline_eager"}
    pe_mismatches: dict[str, int] = {name: 0 for name in order if name != "baseline_eager"}
    with torch.no_grad():
        for _ in range(warmup + reps):
            for name in order:
                samples[name].append(_timed_ms(step_fns[name]))
                if name == "baseline_eager":
                    continue
                mff = mffs[name]
                runner = mff.graph_runner_batched
                exceeded = bool(runner.budget_exceeded) if runner is not None else False
                budget_flags[name].append(exceeded)
                if cast(Any, mff)._graph_batched_pe is not pe:
                    pe_mismatches[name] += 1

    timings = {name: stats_from_samples(samples[name][warmup:]) for name in order}
    ab_wrapper = cast(Any, mffs["adaptive_batch"].cache)
    al_wrapper = cast(Any, mffs["adaptive_loop"].cache)
    expected_steps = warmup + reps
    steady_ab_violations = sum(budget_flags["adaptive_batch"][warmup:])
    batch_publishes = ab_wrapper.put_batch_calls
    no_fallback = bool(
        batch_publishes >= expected_steps
        and pe_mismatches["adaptive_batch"] == 0
        and steady_ab_violations == 0
    )
    al = timings["adaptive_loop"]
    ab = timings["adaptive_batch"]
    sigma = max(al.std, ab.std)

    def _final_capacity(key: str) -> int | None:
        runner = mffs[key].graph_runner_batched
        return int(cast(Any, runner).capacity) if runner is not None else None

    def _recaptures(key: str) -> int:
        policy = cast(Any, mffs[key]).graph_adaptive_policy
        if policy is None:
            return 0
        return sum(1 for e in policy.events if e.get("kind") == "recapture")

    speedups = {
        "adaptive_batch_vs_loop": (al.p50 / ab.p50 if ab.p50 > 0 else float("inf")),
        "adaptive_loop_vs_baseline": (
            timings["baseline_eager"].p50 / al.p50 if al.p50 > 0 else float("inf")
        ),
        "fixed_vs_baseline": (
            timings["baseline_eager"].p50 / timings["graph_fixed"].p50
            if timings["graph_fixed"].p50 > 0
            else float("inf")
        ),
    }
    result = {
        "timings_ms": {name: timings[name].as_dict() for name in order},
        "speedups_p50": {k: round(v, 4) for k, v in speedups.items()},
        "budget_violations": {name: sum(flags) for name, flags in budget_flags.items()},
        "pe_identity_mismatches": pe_mismatches,
        "adaptive_state": {
            "adaptive_loop": {
                "final_capacity": _final_capacity("adaptive_loop"),
                "recaptures": _recaptures("adaptive_loop"),
            },
            "adaptive_batch": {
                "final_capacity": _final_capacity("adaptive_batch"),
                "recaptures": _recaptures("adaptive_batch"),
            },
        },
        "batch_canary": {
            "batch_publishes": batch_publishes,
            "expected_steps": expected_steps,
            "loop_put_calls": al_wrapper.put_calls,
            "batch_put_calls": ab_wrapper.put_calls,
            "steady_state_budget_violations": steady_ab_violations,
            "batch_no_fallback": no_fallback,
        },
        "h3_components": {
            "sigma_ms": sigma,
            "regressed_beyond_2sigma": bool(ab.p50 > al.p50 + 2 * sigma),
            "strictly_improved_beyond_2sigma": bool(al.p50 - ab.p50 > 2 * sigma),
        },
        "note": "round-robin per rep: baseline -> fixed -> adaptive(loop) "
        "-> adaptive(batch); the two adaptive paths differ only in the "
        "cache wrapper",
    }
    # All three folded stacks and their caches are function locals --
    # released on return; the allocator is trimmed explicitly.
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Per-config benchmark
# ---------------------------------------------------------------------------
def _run_config(
    model: Any,
    tokenizer: Any,
    seq_len: int,
    num_candidates: int,
    warmup: int,
    reps: int,
) -> dict[str, Any]:
    """Run the three measurement layers for one (T, N) config."""
    torch.manual_seed(SEED + seq_len + num_candidates)
    torch.cuda.reset_peak_memory_stats()
    GLOBAL_STABILITY_PROFILER.reset()
    num_layers = int(model.config.num_hidden_layers)
    hidden = int(model.config.hidden_size)

    mask, pe = _build_context(model, seq_len)
    parent_ids, cands = _make_workload(model, tokenizer, seq_len, num_candidates)

    put_micro = _bench_put_micro(num_layers, hidden, seq_len, num_candidates, warmup, reps)
    print(
        f"T={seq_len} N={num_candidates} put-micro: "
        f"loop p50={put_micro['timings_ms']['loop']['p50']:.3f}ms "
        f"batch p50={put_micro['timings_ms']['batch']['p50']:.3f}ms "
        f"(ledger {put_micro['write_ops_p50']['loop']} -> "
        f"{put_micro['write_ops_p50']['batch']})"
    )

    publish = _bench_publish_window(
        model,
        seq_len,
        num_candidates,
        warmup,
        reps,
        mask,
        pe,
        parent_ids,
        cands,
        num_layers,
    )
    print(
        f"T={seq_len} N={num_candidates} publish: "
        f"loop p50={publish['timings_ms']['loop']['p50']:.3f}ms "
        f"batch p50={publish['timings_ms']['batch']['p50']:.3f}ms "
        f"sigma={publish['sigma_ms']:.3f}ms "
        f"H2={publish['h2_verdict']} H4={publish['h4_verdict']} "
        f"gate_bit_equal={publish['correctness_gate']['bit_equal']}"
    )

    end_to_end = _bench_end_to_end(
        model,
        seq_len,
        num_candidates,
        warmup,
        reps,
        mask,
        pe,
        parent_ids,
        cands,
    )
    timings = end_to_end["timings_ms"]
    print(
        f"T={seq_len} N={num_candidates} e2e: "
        + " | ".join(f"{name} p50={timings[name]['p50']:.3f}ms" for name in timings)
        + f" | no_fallback={end_to_end['batch_canary']['batch_no_fallback']}"
    )

    return {
        "seq_len": seq_len,
        "num_candidates": num_candidates,
        "num_layers": num_layers,
        "hidden_size": hidden,
        "num_tokens": num_candidates * seq_len,
        "put_microbenchmark": put_micro,
        "publish_window": publish,
        "correctness_gate": publish["correctness_gate"],
        "end_to_end": end_to_end,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run_bench(
    model_path: Path,
    warmup: int = 8,
    reps: int = 30,
    quick: bool = False,
) -> dict[str, Any]:
    """Run the three-layer A/B benchmark over the frozen (T, N) scan.

    Raises:
        RuntimeError: If no CUDA device is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("run_bench requires a CUDA device")

    configs = [(256, 4)] if quick else list(_CONFIGS)

    model, tokenizer, model_meta = _load_model(model_path)
    driver = _gpu_driver_version()
    counts_supported = _cuda_profiler_supported()
    import transformers

    records: list[dict[str, Any]] = []
    for seq_len, num_candidates in configs:
        try:
            record = _run_config(model, tokenizer, seq_len, num_candidates, warmup, reps)
        except Exception as exc:  # noqa: BLE001 - record, never fabricate
            record = {
                "seq_len": seq_len,
                "num_candidates": num_candidates,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"T={seq_len} N={num_candidates}: FAILED "
                f"({type(exc).__name__}: {exc}) -- recorded, not fabricated"
            )
        records.append(record)

    ok = [r for r in records if "error" not in r]
    if not ok:
        raise RuntimeError("every config failed; nothing to verdict (see per-config errors)")

    # H1: UT face (provenance) + every config's real-model publish gate.
    h1 = "supported" if all(r["correctness_gate"]["bit_equal"] for r in ok) else "refuted"
    # H2 / H4: most conservative aggregation over per-config verdicts.
    h2 = _overall([r["publish_window"]["h2_verdict"] for r in ok])
    h4 = _overall([r["publish_window"]["h4_verdict"] for r in ok])
    # H3: no-fallback everywhere + no regression anywhere + strict
    # improvement on the frozen minimal config (srs 3.3).
    min_records = [r for r in ok if (r["seq_len"], r["num_candidates"]) == MIN_CONFIG]
    regressed_any = any(r["end_to_end"]["h3_components"]["regressed_beyond_2sigma"] for r in ok)
    no_fallback_all = all(r["end_to_end"]["batch_canary"]["batch_no_fallback"] for r in ok)
    min_strict = bool(
        min_records
        and min_records[0]["end_to_end"]["h3_components"]["strictly_improved_beyond_2sigma"]
    )
    if regressed_any:
        h3 = "refuted"
    elif no_fallback_all and not regressed_any and min_strict:
        h3 = "supported"
    else:
        h3 = "inconclusive"

    return {
        "meta": {
            "device": torch.cuda.get_device_name(0),
            "gpu_driver_version": driver,
            "gpu_driver_version_note": (
                "queried via nvidia-smi (AR016 Minor-2 closure); null means "
                "the query failed on this host and the value was omitted "
                "rather than fabricated"
                if driver is None
                else "queried via nvidia-smi (AR016 Minor-2 closure)"
            ),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": cast(Any, torch.version).cuda,
            "platform_note": (
                "Windows WDDM display-driver mode (no clock locking); noise "
                "mitigated by warmup + reps + p50 + interleaved A/B"
            ),
            "run_date_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "model": "Fast-dLLM-v2-1.5B (Efficient-Large-Model, remote code)",
            "model_details": model_meta,
            "venv": sys.prefix != sys.base_prefix,
            "workload": {
                "mask_id": MASK_ID,
                "block": BLOCK,
                "workload": "parent = K_PROMPT prompt tokens + one all-mask "
                "block; children = top-10 model-sampled block fillings "
                "(AR019 probe P3 recipe, reused verbatim)",
                "tau": TAU,
                "capacity_ratio": CAPACITY_RATIO,
                "topk": TOPK,
                "cache_budget": CACHE_BUDGET,
            },
            "foreach_available": _FOREACH_AVAILABLE,
            "foreach_note": (
                "torch._foreach_copy_ present (probe P1-P6); when absent "
                "copy_pairs degrades to per-pair copy_ (same write order, "
                "ledger counts pairs)"
            ),
            "c_constant": C_CONSTANT,
            "c_constant_note": (
                "H4 threshold constant c=2 frozen at design time (srs 3.3 / "
                "design 4.2.2(5)); written into this artifact before the "
                "pre-registered single run, not adjustable after"
            ),
            "min_config": list(MIN_CONFIG),
            "min_config_note": (
                "minimal N*T config for H3's strict-improvement clause, "
                "frozen from the AR019 artifact matrix at design time"
            ),
            "verdict_formulas": _VERDICT_FORMULAS,
            "ut_face": _UT_FACE,
            "kernel_count": None if not counts_supported else "CUPTI available",
            "kernel_count_note": (
                "this torch build cannot record CUDA profiler events "
                "(LIBKINETO_NOCUPTI); kernel-launch counts omitted rather "
                "than fabricated -- the write_op_count ledger is the H4 "
                "structural evidence (rerun on a CUPTI-capable host for "
                "launch counts)"
                if not counts_supported
                else "counts recordable via torch.profiler (CUPTI); the "
                "write_op_count ledger remains the H4 evidence"
            ),
            "warmup": warmup,
            "reps": reps,
            "pre_registered_single_run": (
                "reps as given, one run, no rerun fishing (#39 lineage); "
                "WDDM noise swamping verdicts recorded as inconclusive"
            ),
            "timing_method": (
                "time.perf_counter around torch.cuda.synchronize() "
                "boundaries (host-side publish cost included -- the AR020 "
                "measurement target; AR019's CUDA-event face excluded it by "
                "construction)"
            ),
            "ab_design": (
                "symmetric one-hop wrapper caches: loop delegates every "
                "method but never exposes put_batch; batch adds a "
                "pass-through put_batch; A/B via mff.cache swap, capability "
                "re-probed on every publish"
            ),
            "command": "python -m scripts.ar020_publish_put_bench"
            + (f" --model-path {model_path}" if model_path else "")
            + (" --quick" if quick else "")
            + (f" --reps {reps}" if reps != 30 else "")
            + (f" --warmup {warmup}" if warmup != 8 else ""),
        },
        "hypotheses_declared": {
            "H1": "put_batch is bit-equivalent to sequential put on the UT "
            "face and the real-model publish face",
            "H2": "publish-window p50: batch beats loop beyond 2*sigma "
            "(sigma = max of the two stds)",
            "H3": "end-to-end adaptive: batch never falls back and the "
            "minimal config (256, 4) strictly improves beyond 2*sigma",
            "H4": "host-call ledger: batch write_op_count <= 2*(L+N) " "(vs the loop's L*N+N)",
        },
        "hypotheses_verdicts": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "failed_configs": [
            {"seq_len": r["seq_len"], "num_candidates": r["num_candidates"], "error": r["error"]}
            for r in records
            if "error" in r
        ],
        "configs": records,
    }


def write_results(data: dict[str, Any], path: str | Path) -> Path:
    """Write the benchmark record as JSON, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Local checkpoint directory (env fallback: {DEFAULT_MODEL_ENV}).",
    )
    parser.add_argument("--reps", type=int, default=30, help="Timed reps per config")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup reps per config")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Single-config smoke subset (T=256, N=4, reps=3, warmup=2)",
    )
    add_hf_env_arguments(parser)
    args = parser.parse_args()

    reps = 3 if args.quick else args.reps
    warmup = 2 if args.quick else args.warmup
    model_path = _resolve_model_path(args.model_path)
    data = run_bench(model_path, warmup=warmup, reps=reps, quick=args.quick)
    out = write_results(data, args.out)
    print()
    for key in ("H1", "H2", "H3", "H4"):
        print(f"{key}: {data['hypotheses_verdicts'][key]}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
