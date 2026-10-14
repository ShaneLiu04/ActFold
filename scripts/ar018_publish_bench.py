"""AR018/T003: publish-path host-cost benchmark (batched reduction + snapshot).

Measures the same-shape multi-candidate diffusion verification workload on
THREE paths per (H, L, N, T=512) -- the AR017 config matrix, so the
cross-artifact anchor stays valid:

* ``baseline_eager`` -- the raw (unfolded) model, N full forwards per step.
* ``graph_fixed`` -- the AR016 batched CUDA graph path, fixed ratio 0.5.
* ``graph_adaptive`` -- the AR017 adaptive-capacity path, now publishing
  through the AR018 compressed loop (ONE ``mask_buf.sum(dim=2)`` reduction
  recorded via ``record_precomputed`` + ONE whole-buffer ``child_buf.clone``
  snapshot; the pre-AR018 loop spent 3*L*N launches on per-row
  clone/put-copy/sum -- the probe measured the publish phase at
  60.7%/71.2%/29.7%/31.5% of the full adaptive step across the four
  configs).

Hypotheses (declared here, before measurement; verdicts come from data only
and are recorded as ``supported`` / ``refuted`` / ``inconclusive`` /
``no_anchor``):

* **H1** -- AR018 adaptive p50 < AR017 artifact adaptive p50 per config
  (cross-artifact comparison: same machine class, seed, method and config
  matrix; two-sample 2-sigma criterion via ``_verdict``; the AR017 anchor
  values are embedded into this artifact for auditability).
* **H2** -- AR018 speedup (baseline/adaptive) >= AR017 artifact speedup
  (the P1-7 fourth rung: the publish compression must not regress the
  folded framework's advantage over the baseline; equality is supported --
  the no-regression boundary).
* **H3** -- the publish phase's share of the full adaptive step drops
  below the AR018 host-cost probe reference (60.7%/71.2%/29.7%/31.5%).
* **H4** -- the three paths agree (exact argmax; adaptive-vs-fixed graph
  allclose 1e-4) and the profiler statistics are consistent (each branch's
  last-step stable ratio equals an independent per-row ``mask_buf`` sum
  recomputation).

Timing is ``time.perf_counter`` around ``torch.cuda.synchronize()``
boundaries (host-side prefill, readback and publish are part of the graph
paths' real cost), round-robin interleaved per repetition (RERUN_CHECKLIST
SS4.2), warmup=5 covering capture plus the adaptation schedule, reps=30
(the n=10 under-powered precedent pre-commits to a single high-power run).

The phase decomposition (six sync-bounded phases driven through private
APIs, probe method fixed as a script-level measurement) runs OUTSIDE the
timed window; ``phase_sum_ms`` and ``full_step_p50_ms`` are both recorded
with a two-way deviation note -- sync-bounded phases serialize the pipeline
(positive bias) while host work between phases is missed (negative bias);
the probe measured the negative side dominating (3 of 4 configs), so no
direction is assumed. H3 judges the share, not the absolute value.

Peak memory (``torch.cuda.max_memory_allocated`` around one untimed
adaptive step) quantifies the snapshot transient (srs risk-table promise).

Run::

    python -m scripts.ar018_publish_bench [--quick] [--reps 30] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import torch

from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from scripts.algo_experiments import TimingStats, stats_from_samples
from scripts.ar017_adaptive_bench import (
    _CAPACITY_RATIO,
    _PATH_LABELS,
    _PATH_ORDER,
    _SEED,
    _SEQ_LEN,
    _Decoder,
    _GEOMETRIES,
    _NUM_CANDIDATES,
    _argmax_agreement,
    _cuda_profiler_supported,
    _gpu_driver_version,
    _make_candidates,
    _make_mff,
    _overall,
    _timed_ms,
    _verdict,
)

DEFAULT_OUTPUT_PATH = Path("results/optimization/ar018_publish_bench.json")
AR017_ANCHOR_PATH = Path("results/optimization/ar017_adaptive_bench.json")

# Probe reference (H3): publish-phase share of the full adaptive step, from
# ``ar018_hostcost_probe`` (2026-10-14, n=30, AR017 fixture/steady capacity,
# same seed/method) -- the pre-AR018 publish cost this AR compresses.
_PROBE_PUBLISH_SHARE: dict[tuple[int, int], float] = {
    (256, 4): 0.607,
    (256, 8): 0.712,
    (1024, 4): 0.297,
    (1024, 8): 0.315,
}
_PROBE_PROVENANCE = (
    "ar018_hostcost_probe (temp probe, 2026-10-14, n=30, AR017 "
    "fixture/seed/steady capacity, sync-bounded six-phase decomposition)"
)

_PHASES = (
    "revalidate_host",
    "prefill",
    "replay",
    "readback",
    "publish",
    "logits_clone",
)


def _load_ar017_anchor(path: Path) -> dict[str, Any] | None:
    """Load the AR017 artifact as the cross-artifact anchor (best effort).

    A missing or unreadable anchor never blocks the run: H1/H2 degrade to
    ``no_anchor`` verdicts instead of failing.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("configs"), list):
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None


def _anchor_stats(
    anchor: dict[str, Any] | None,
    hidden_dim: int,
    num_candidates: int,
    path_name: str,
) -> TimingStats | None:
    """Rebuild one AR017 TimingStats from the anchor artifact."""
    if anchor is None:
        return None
    for record in anchor["configs"]:
        if record.get("hidden_dim") != hidden_dim:
            continue
        if record.get("num_candidates") != num_candidates:
            continue
        raw = record.get("timings_ms", {}).get(path_name)
        if not isinstance(raw, dict):
            return None
        return TimingStats(
            mean=float(raw["mean"]),
            std=float(raw["std"]),
            p50=float(raw["p50"]),
            n=int(raw["n"]),
        )
    return None


def _phase_decomposition(
    mff: Any,
    runner: Any,
    candidates: torch.Tensor,
    branch_ids: list[str],
    parent_branch_id: str,
    step_idx: int,
    reps: int,
) -> tuple[dict[str, TimingStats], int]:
    """Six-phase sync-bounded decomposition of one steady-state step.

    Replicates the AR018 host-cost probe method (private-API drive, script
    layer): revalidate -> prefill -> replay -> budget readback -> publish
    -> logits clone. Phases are separated by ``torch.cuda.synchronize()``
    boundaries, so their sum carries the two-way sync-vs-omission deviation
    documented in the module docstring.

    Raises:
        RuntimeError: If any production precondition fails mid-decomposition
            (parent cache incomplete, prefill refused, budget exceeded) --
            the same hard asserts the probe used.
    """
    samples: dict[str, list[float]] = {name: [] for name in _PHASES}
    with torch.no_grad():
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if not mff._parent_cache_complete_batched(parent_branch_id, candidates):
                raise RuntimeError("parent cache incomplete during decomposition")
            t1 = time.perf_counter()
            if not runner._prefill_parent(parent_branch_id):
                raise RuntimeError("prefill refused during decomposition")
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            runner._tokens_static.copy_(candidates)
            runner._count_buf.zero_()
            runner._graph.replay()
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            runner.validate_budgets()
            t4 = time.perf_counter()
            if runner.budget_exceeded:
                raise RuntimeError("budget exceeded during decomposition")
            mff._publish_graph_children(
                runner, candidates, branch_ids, parent_branch_id, step_idx
            )
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            runner._logits_buf.clone()
            torch.cuda.synchronize()
            t6 = time.perf_counter()
            step_idx += 1
            samples["revalidate_host"].append((t1 - t0) * 1000.0)
            samples["prefill"].append((t2 - t1) * 1000.0)
            samples["replay"].append((t3 - t2) * 1000.0)
            samples["readback"].append((t4 - t3) * 1000.0)
            samples["publish"].append((t5 - t4) * 1000.0)
            samples["logits_clone"].append((t6 - t5) * 1000.0)
    return {name: stats_from_samples(vals) for name, vals in samples.items()}, step_idx


def _profiler_consistency(
    mff: Any,
    runner: Any,
    candidates: torch.Tensor,
    branch_ids: list[str],
    parent_branch_id: str,
    num_layers: int,
    seq_len: int,
) -> bool:
    """Check the recorded stable ratios against per-row mask_buf re-sums.

    One untimed adaptive step is published after a profiler reset; the last
    ``num_layers`` entries of each branch profile must equal an independent
    per-row ``mask_buf[layer][row].sum() / T`` recomputation (the pre-AR018
    per-row path's own arithmetic) within 1e-6.
    """
    GLOBAL_STABILITY_PROFILER.reset()
    with torch.no_grad():
        mff(
            candidates,
            branch_id=None,
            branch_ids=branch_ids,
            parent_branch_id=parent_branch_id,
        )
    if cast(Any, runner).budget_exceeded:
        return False
    for row, branch_id in enumerate(branch_ids):
        profile = GLOBAL_STABILITY_PROFILER.get_profile(branch_id)
        if profile is None:
            return False
        stats = profile.layer_stats
        if len(stats) < num_layers:
            return False
        for layer_idx in range(num_layers):
            entry = stats[-num_layers + layer_idx]
            if entry.layer_idx != layer_idx:
                return False
            recomputed = (
                float(
                    cast(Any, runner).mask_buf[layer_idx][row].sum().item()
                )
                / seq_len
            )
            if abs(entry.stable_ratio - recomputed) > 1e-6:
                return False
    return True


def _run_config(
    hidden_dim: int,
    num_layers: int,
    num_candidates: int,
    warmup: int,
    reps: int,
    anchor: dict[str, Any] | None,
) -> dict[str, Any]:
    """Benchmark one (H, L, N) config: three paths + decomposition + H1-H4."""
    torch.manual_seed(_SEED + hidden_dim + num_candidates)
    model = _Decoder(
        vocab_size=1000, hidden_dim=hidden_dim, num_layers=num_layers
    ).to("cuda")
    model.eval()

    seq_len = _SEQ_LEN
    mffs: dict[str, Any] = {
        "graph_fixed": _make_mff(model, adaptive=False),
        "graph_adaptive": _make_mff(model, adaptive=True),
    }
    parent_tokens = torch.randint(0, 1000, (1, seq_len), device="cuda")
    candidates = _make_candidates(parent_tokens, num_candidates, seq_len)
    fixed_ids = [f"f{i}" for i in range(num_candidates)]
    adaptive_ids = [f"a{i}" for i in range(num_candidates)]

    for mff in mffs.values():
        mff(parent_tokens, branch_id="parent")

    step_fns: dict[str, Callable[[], Any]] = {
        "baseline_eager": lambda: [
            model(candidates[i : i + 1]) for i in range(num_candidates)
        ],
        "graph_fixed": lambda: mffs["graph_fixed"](
            candidates,
            branch_id=None,
            branch_ids=fixed_ids,
            parent_branch_id="parent",
        ),
        "graph_adaptive": lambda: mffs["graph_adaptive"](
            candidates,
            branch_id=None,
            branch_ids=adaptive_ids,
            parent_branch_id="parent",
        ),
    }

    # Per-repetition round-robin interleaving; warmup covers capture and the
    # adaptation schedule so the timed steady state reflects the adapted
    # capacity plus the AR018 compressed publish.
    samples: dict[str, list[float]] = {name: [] for name in _PATH_ORDER}
    with torch.no_grad():
        for _ in range(warmup + reps):
            for name in _PATH_ORDER:
                samples[name].append(_timed_ms(step_fns[name]))
    timings = {name: stats_from_samples(samples[name][warmup:]) for name in _PATH_ORDER}

    # Correctness sidecar (H4, untimed): one extra call per path.
    logits: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in _PATH_ORDER:
            out = step_fns[name]()
            if name == "baseline_eager":
                logits[name] = torch.cat([o for o in out], dim=0)
            else:
                logits[name] = out
    pairwise: dict[str, float] = {}
    parity = True
    for i, a in enumerate(_PATH_ORDER):
        for b in _PATH_ORDER[i + 1 :]:
            agreement = _argmax_agreement(logits[a], logits[b])
            pairwise[f"{a}__vs__{b}"] = round(agreement, 6)
            parity = parity and agreement == 1.0
    adaptive_close = bool(
        torch.allclose(
            logits["graph_adaptive"],
            logits["graph_fixed"],
            atol=1e-4,
            rtol=1e-4,
        )
    )

    adaptive_runner = mffs["graph_adaptive"].graph_runner_batched
    graph_captured = {
        "graph_fixed": mffs["graph_fixed"].graph_runner_batched is not None
        and tuple(
            mffs["graph_fixed"].graph_runner_batched.tokens_static.shape
        )
        == (num_candidates, seq_len),
        "graph_adaptive": adaptive_runner is not None
        and tuple(adaptive_runner.tokens_static.shape)
        == (num_candidates, seq_len),
    }

    # Profiler consistency (H4, untimed, outside the timed window).
    profiler_ok = _profiler_consistency(
        mffs["graph_adaptive"],
        adaptive_runner,
        candidates,
        adaptive_ids,
        "parent",
        num_layers,
        seq_len,
    )

    # Peak memory sidecar: reset, one untimed adaptive step, read the peak.
    # The snapshot transient (whole child_buf cloned once) shows up here.
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        step_fns["graph_adaptive"]()
    torch.cuda.synchronize()
    peak_memory_bytes = int(torch.cuda.max_memory_allocated())

    # Phase decomposition (probe method, n=reps, untimed private-API drive).
    phase_stats, _next_step = _phase_decomposition(
        mffs["graph_adaptive"],
        adaptive_runner,
        candidates,
        adaptive_ids,
        "parent",
        warmup + reps,
        reps,
    )
    full_step_p50 = timings["graph_adaptive"].p50
    phase_sum = sum(phase_stats[name].p50 for name in _PHASES)
    publish_share = phase_stats["publish"].p50 / full_step_p50
    probe_share = _PROBE_PUBLISH_SHARE.get((hidden_dim, num_candidates))

    # H1/H2: cross-artifact anchor verdicts (no_anchor degrades gracefully).
    ar017_adaptive = _anchor_stats(anchor, hidden_dim, num_candidates, "graph_adaptive")
    ar017_baseline = _anchor_stats(anchor, hidden_dim, num_candidates, "baseline_eager")
    if ar017_adaptive is None or ar017_baseline is None:
        h1: str = "no_anchor"
        h2: str = "no_anchor"
        ar017_speedup: float | None = None
    else:
        h1 = _verdict(timings["graph_adaptive"], ar017_adaptive)
        ar017_speedup = ar017_baseline.p50 / ar017_adaptive.p50
        ar018_speedup = timings["baseline_eager"].p50 / timings["graph_adaptive"].p50
        h2 = "supported" if ar018_speedup >= ar017_speedup else "refuted"

    h3 = (
        "supported"
        if probe_share is not None and publish_share < probe_share
        else "refuted"
        if probe_share is not None
        else "no_reference"
    )
    h4 = "supported" if (parity and adaptive_close and profiler_ok) else "refuted"

    speedups = {
        "adaptive_vs_fixed": (
            timings["graph_fixed"].p50 / timings["graph_adaptive"].p50
            if timings["graph_adaptive"].p50 > 0
            else float("inf")
        ),
        "adaptive_vs_baseline": (
            timings["baseline_eager"].p50 / timings["graph_adaptive"].p50
            if timings["graph_adaptive"].p50 > 0
            else float("inf")
        ),
        "fixed_vs_baseline": (
            timings["baseline_eager"].p50 / timings["graph_fixed"].p50
            if timings["graph_fixed"].p50 > 0
            else float("inf")
        ),
    }

    return {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_candidates": num_candidates,
        "seq_len": seq_len,
        "num_tokens": num_candidates * seq_len,
        "timings_ms": {name: timings[name].as_dict() for name in _PATH_ORDER},
        "speedups_p50": {k: round(v, 4) for k, v in speedups.items()},
        "graph_captured": graph_captured,
        "graph_validated_last_step": {
            "graph_fixed": (
                bool(not mffs["graph_fixed"].graph_runner_batched.budget_exceeded)
                if mffs["graph_fixed"].graph_runner_batched is not None
                else None
            ),
            "graph_adaptive": (
                bool(not adaptive_runner.budget_exceeded)
                if adaptive_runner is not None
                else None
            ),
        },
        "logits_parity_argmax_exact": parity,
        "adaptive_vs_fixed_allclose_1e-4": adaptive_close,
        "argmax_agreement_pairwise": pairwise,
        "profiler_consistent": profiler_ok,
        "peak_memory_allocated_bytes": peak_memory_bytes,
        "phase_decomposition": {
            "phases_ms": {name: phase_stats[name].as_dict() for name in _PHASES},
            "phase_sum_ms": round(phase_sum, 4),
            "full_step_p50_ms": round(full_step_p50, 4),
            "publish_share_of_full_step": round(publish_share, 6),
            "probe_publish_share_reference": probe_share,
            "probe_provenance": _PROBE_PROVENANCE,
            "deviation_note": (
                "phase_sum_ms and full_step_p50_ms are both recorded; their "
                "deviation is two-way (sync boundaries serialize the phases "
                "positively while host work between phases is missed "
                "negatively) and no direction is assumed"
            ),
            "method_note": (
                "six sync-bounded phases driven through private APIs "
                "(runner._prefill_parent/_graph.replay/validate_budgets, "
                "mff._publish_graph_children), script-layer measurement "
                "outside the timed window"
            ),
        },
        "h1_vs_ar017_adaptive": h1,
        "h2_speedup_no_regression_vs_ar017": h2,
        "ar017_anchor_speedup_p50": (
            round(ar017_speedup, 4) if ar017_speedup is not None else None
        ),
        "h3_publish_share_below_probe": h3,
        "h4_correctness_and_profiler": h4,
    }


def run_bench(warmup: int = 5, reps: int = 30, quick: bool = False) -> dict[str, Any]:
    """Run the three-path publish-cost benchmark over the config scan.

    Raises:
        RuntimeError: If no CUDA device is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("run_bench requires a CUDA device")

    configs: list[tuple[int, int, int]] = [
        (hidden, layers, n)
        for hidden, layers in _GEOMETRIES
        for n in _NUM_CANDIDATES
    ]
    if quick:
        configs = [(256, 8, 4)]

    driver = _gpu_driver_version()
    counts_supported = _cuda_profiler_supported()
    anchor = _load_ar017_anchor(AR017_ANCHOR_PATH)
    config_records = []
    for hidden_dim, num_layers, num_candidates in configs:
        GLOBAL_STABILITY_PROFILER.reset()
        record = _run_config(
            hidden_dim, num_layers, num_candidates, warmup, reps, anchor
        )
        config_records.append(record)
        decomp = record["phase_decomposition"]
        print(
            f"H={hidden_dim:4d} L={num_layers:2d} N={num_candidates}: "
            + " | ".join(
                f"{label:8s} p50={record['timings_ms'][name]['p50']:.3f}ms"
                for label, name in _PATH_LABELS
            )
        )
        print(
            f"    publish {decomp['phases_ms']['publish']['p50']:.3f}ms "
            f"({decomp['publish_share_of_full_step'] * 100:.1f}% of full, "
            f"probe {decomp['probe_publish_share_reference'] * 100:.1f}%) "
            f"H3={record['h3_publish_share_below_probe']} | "
            f"H1={record['h1_vs_ar017_adaptive']} "
            f"H2={record['h2_speedup_no_regression_vs_ar017']} "
            f"H4={record['h4_correctness_and_profiler']}"
        )

    anchor_available = anchor is not None
    h1 = (
        _overall([r["h1_vs_ar017_adaptive"] for r in config_records])
        if anchor_available
        else "no_anchor"
    )
    h2 = (
        _overall([r["h2_speedup_no_regression_vs_ar017"] for r in config_records])
        if anchor_available
        else "no_anchor"
    )
    h3 = _overall([r["h3_publish_share_below_probe"] for r in config_records])
    h4 = _overall([r["h4_correctness_and_profiler"] for r in config_records])

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
            "cuda_version": cast(Any, torch.version).cuda,
            "platform_note": (
                "Windows WDDM display-driver mode (no clock locking); noise "
                "mitigated by warmup + 30 reps + p50 + per-rep round-robin"
            ),
            "run_date_utc": datetime.now(timezone.utc).isoformat(),
            "seed": _SEED,
            "model": "synthetic LLaMA-layout V=1000, per-config (H, L)",
            "capacity_ratio": _CAPACITY_RATIO,
            "warmup": warmup,
            "reps": reps,
            "warmup_note": (
                "warmup covers capture plus the adaptation schedule (shrink "
                "patience 3 + throttle interval 2); timed steady state "
                "reflects the adapted capacity and the AR018 publish"
            ),
            "interleaving": (
                "round-robin per repetition: " + " -> ".join(_PATH_ORDER)
            ),
            "timing_method": (
                "time.perf_counter around torch.cuda.synchronize() boundaries "
                "(host-side prefill/readback/publish included; CUDA-event "
                "timing would exclude them)"
            ),
            "kernel_count": None if not counts_supported else "CUPTI available",
            "kernel_count_note": (
                "this torch build cannot record CUDA profiler events "
                "(LIBKINETO_NOCUPTI); kernel counts omitted rather than "
                "fabricated - rerun on a CUPTI-capable host to record them"
                if not counts_supported
                else "counts recordable via torch.profiler (CUPTI); wall-clock "
                "remains the primary metric"
            ),
            "peak_memory_method": (
                "torch.cuda.reset_peak_memory_stats() before one untimed "
                "adaptive step, max_memory_allocated() after synchronize; "
                "quantifies the snapshot transient (srs risk-table promise)"
            ),
            "cross_artifact_anchor": {
                "path": str(AR017_ANCHOR_PATH),
                "available": anchor_available,
                "nature": (
                    "cross-artifact comparison (H1/H2): same machine class, "
                    "seed, method, and config matrix as the AR017 run; the "
                    "run_date differs, so day-to-day machine state is a "
                    "declared confound carried by the 2-sigma criterion"
                ),
                "ar017_run_date_utc": (
                    anchor["meta"]["run_date_utc"] if anchor_available else None
                ),
                "anchor_values_embedded": (
                    "per-config h1/h2 verdicts are computed against the AR017 "
                    "artifact's stored p50/std/n (reproducible offline); the "
                    "anchor speedup is recorded as ar017_anchor_speedup_p50"
                ),
            },
            "command": "python -m scripts.ar018_publish_bench"
            + (" --quick" if quick else "")
            + f" --reps {reps}"
            + (f" --warmup {warmup}" if warmup != 5 else ""),
        },
        "hypotheses_declared": {
            "H1": "AR018 adaptive p50 < AR017 artifact adaptive p50 per "
            "config (cross-artifact, two-sample 2-sigma via _verdict)",
            "H2": "AR018 baseline/adaptive speedup >= AR017 artifact speedup "
            "(P1-7 fourth rung: no regression; equality is supported)",
            "H3": "publish-phase share of the full adaptive step < the AR018 "
            f"host-cost probe reference ({_PROBE_PROVENANCE})",
            "H4": "three-path logits consistency (exact argmax + adaptive-vs-"
            "fixed allclose 1e-4) and profiler statistics equal to an "
            "independent per-row mask_buf re-sum",
        },
        "hypotheses_verdicts": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "configs": config_records,
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
    parser.add_argument("--reps", type=int, default=30, help="Timed reps per config")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup reps per config")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Single-config smoke subset (H=256, L=8, N=4, T=512) for CI",
    )
    args = parser.parse_args()

    data = run_bench(warmup=args.warmup, reps=args.reps, quick=args.quick)
    out = write_results(data, args.out)
    print(f"\nH1 (vs AR017 adaptive p50): {data['hypotheses_verdicts']['H1']}")
    print(f"H2 (speedup no-regression vs AR017): {data['hypotheses_verdicts']['H2']}")
    print(f"H3 (publish share below probe): {data['hypotheses_verdicts']['H3']}")
    print(f"H4 (consistency + profiler): {data['hypotheses_verdicts']['H4']}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
