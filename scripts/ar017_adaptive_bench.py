"""AR017/T003: adaptive divergent-capacity benchmark (decisive P1-7 re-answer).

Measures the same-shape multi-candidate diffusion verification workload on
THREE paths (per geometry x N x T):

* ``baseline_eager`` -- the raw (unfolded) model, N full forwards per step.
* ``graph_fixed`` -- the AR016 batched CUDA graph path with the fixed
  ``capacity_ratio=0.5`` capacity (the AR017 status quo).
* ``graph_adaptive`` -- the AR017 adaptive path: same initial ratio 0.5, the
  observation-driven policy re-captures the runner when the capacity is
  persistently over-provisioned (mid-divergence loads sit at D_frac ~13%,
  so a fixed C = 50% pays 3x+ FFN recompute waste -- srs SS1.1 evidence).

Geometries: (H=256, L=8) -- the small shape where graph overhead dominates
(the AR016 P1-7 inconclusive regime) -- and (H=1024, L=16) -- the deep shape
where the AR016 probes measured graph-vs-baseline at 1.36-1.41x and the
capacity probe lifted it to 2.32-2.43x. N in {4, 8} with T=512, so
``N * T`` stays a power of two (design SS4.4 capacity exactness).

Hypotheses (declared here, before measurement; verdicts come from data only
and are recorded as ``supported`` / ``refuted`` / ``inconclusive``):

* **H1** -- graph_adaptive is significantly faster than graph_fixed (the
  right-sized capacity cuts FFN recompute at mid-divergence loads).
* **H2** -- graph_adaptive is significantly faster than baseline_eager (the
  decisive P1-7 re-answer at deep geometry).
* **H3** -- the three paths produce consistent logits (exact argmax
  agreement; adaptive-vs-fixed graph allclose 1e-4).
* **H4** -- the adaptive capacity tracks the real demand: the final capacity
  lands below the fixed capacity, at or above the measured maximum layer
  divergence, with at least one shrink recapture and zero budget violations
  after the adaptation.

Timing is ``time.perf_counter`` around ``torch.cuda.synchronize()``
boundaries (host-side prefill, count readback, and per-branch publish are
part of the graph paths' real cost; re-captures land inside the warmup so
the timed steady state reflects the adapted capacity). Paths are interleaved
round-robin within every repetition (RERUN_CHECKLIST SS4.2). ``warmup=5``
covers the adaptation schedule (shrink patience 3 + throttle interval 2);
the adaptation events are recorded per config as a separate table.

Run::

    python -m scripts.ar017_adaptive_bench [--quick] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.adaptive_capacity import AdaptiveCapacityConfig
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from scripts.algo_experiments import TimingStats, stats_from_samples

DEFAULT_OUTPUT_PATH = Path("results/optimization/ar017_adaptive_bench.json")

_VOCAB = 1000
_TAU = 0.95
_EPS = 1e-8
_CAPACITY_RATIO = 0.5
_SEED = 2026
_SEQ_LEN = 512

_GEOMETRIES: tuple[tuple[int, int], ...] = (
    (256, 8),
    (1024, 16),
)
_NUM_CANDIDATES: tuple[int, ...] = (4, 8)

_PATH_ORDER = (
    "baseline_eager",
    "graph_fixed",
    "graph_adaptive",
)
_PATH_LABELS = tuple(
    (label, name)
    for label, name in zip(("baseline", "fixed", "adaptive"), _PATH_ORDER)
)


# ---------------------------------------------------------------------------
# Synthetic LLaMA-layout decoder (detectable post_attention_layernorm->mlp
# chain), identical in structure to the AR002/AR016 benchmark fixtures.
# ---------------------------------------------------------------------------
class _Attention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        out = hidden_states * 0.5
        if attention_mask is not None:
            out = out * attention_mask.to(out.dtype).unsqueeze(-1)
        return out, None


class _Layer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = _Attention()
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


class _DecoderModel(nn.Module):
    """Typed container so attribute access stays strict under mypy."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([_Layer(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)


class _Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = _VOCAB,
        hidden_dim: int = 256,
        num_layers: int = 8,
    ) -> None:
        super().__init__()
        self.model = _DecoderModel(vocab_size, hidden_dim, num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return cast(torch.Tensor, self.lm_head(self.model.norm(x)))


# ---------------------------------------------------------------------------
# Workload construction (AR016 precedent): parent [1, T] cached once; N
# same-parent equal-length candidates with a shared stable prefix, a 4-token
# divergent mid block and an independent ~T/8 random suffix -- the measured
# mid-divergence band (D_frac ~13%, srs SS1.1) that makes the fixed 0.5
# capacity over-provisioned.
# ---------------------------------------------------------------------------
def _make_candidates(
    parent_tokens: torch.Tensor,
    num_candidates: int,
    seq_len: int,
) -> torch.Tensor:
    """Build [N, T] candidates: shared stable prefix, divergent mid + suffix."""
    gen = torch.Generator(device=parent_tokens.device).manual_seed(_SEED + 7)
    candidates = parent_tokens.expand(num_candidates, seq_len).clone()
    mid_start = max(1, seq_len // 3)
    for row in range(num_candidates):
        for pos in range(mid_start, min(mid_start + 4, seq_len)):
            candidates[row, pos] = (candidates[row, pos] + 1 + row) % _VOCAB
    suffix_start = max(mid_start + 4, seq_len - max(1, seq_len // 8))
    suffix = torch.randint(
        0,
        _VOCAB,
        (num_candidates, seq_len - suffix_start),
        generator=gen,
        device=parent_tokens.device,
    )
    candidates[:, suffix_start:] = suffix
    return candidates


def _make_mff(model: nn.Module, adaptive: bool) -> ManualFoldedForward:
    cache = ActivationCache(max_entries_per_layer=8192, device="cuda")
    gate = SimilarityGate(tau=_TAU, metric="cosine", eps=_EPS)
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        split_layers=True,
        split_min_tokens=1,
        use_cuda_graph=True,
        graph_capacity_ratio=_CAPACITY_RATIO,
        graph_adaptive_capacity=adaptive,
        graph_adaptive_config=AdaptiveCapacityConfig() if adaptive else None,
    )


def _timed_ms(step_fn: Callable[[], Any]) -> float:
    """Wall-clock one step in ms, synchronize-bounded (perf_counter)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _gpu_driver_version() -> str | None:
    """Query the display driver version via nvidia-smi (best effort)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        first = out.stdout.strip().splitlines()[0].strip()
        return first or None
    except Exception:  # noqa: BLE001 - metadata probe must never fail the run
        return None


def _cuda_profiler_supported() -> bool:
    """Probe whether this build records CUDA profiler events (CUPTI, #39)."""
    try:
        x = torch.ones(4, device="cuda")
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            x = x + 1
            torch.cuda.synchronize()
        return any(
            evt.device_type == torch.profiler.DeviceType.CUDA
            for evt in prof.key_averages()
        )
    except Exception:  # noqa: BLE001 - probe must never raise
        return False


def _argmax_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of positions where two [N, T] logits agree on argmax."""
    return float((a.argmax(-1) == b.argmax(-1)).float().mean().item())


def _verdict(faster: TimingStats, slower: TimingStats) -> str:
    """supported iff the p50 delta exceeds 2x the merged std (else decide)."""
    delta = slower.p50 - faster.p50
    merged_std = (faster.std**2 + slower.std**2) ** 0.5
    if delta > 2 * merged_std:
        return "supported"
    if delta < -2 * merged_std:
        return "refuted"
    return "inconclusive"


def _overall(verdicts: list[str]) -> str:
    """Strict aggregation: supported everywhere, refuted anywhere, else mixed."""
    if all(v == "supported" for v in verdicts):
        return "supported"
    if any(v == "refuted" for v in verdicts):
        return "refuted"
    return "inconclusive"


# ---------------------------------------------------------------------------
# Per-config benchmark
# ---------------------------------------------------------------------------
def _run_config(
    hidden_dim: int,
    num_layers: int,
    num_candidates: int,
    warmup: int,
    reps: int,
) -> dict[str, Any]:
    """Benchmark one (H, L, N) config on the three interleaved paths."""
    torch.manual_seed(_SEED + hidden_dim + num_candidates)
    model = _Decoder(
        vocab_size=_VOCAB, hidden_dim=hidden_dim, num_layers=num_layers
    ).to("cuda")
    model.eval()

    seq_len = _SEQ_LEN
    mffs: dict[str, ManualFoldedForward] = {
        "graph_fixed": _make_mff(model, adaptive=False),
        "graph_adaptive": _make_mff(model, adaptive=True),
    }
    parent_tokens = torch.randint(0, _VOCAB, (1, seq_len), device="cuda")
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

    # Per-repetition round-robin interleaving (RERUN_CHECKLIST 4.2); the
    # warmup covers both capture and the adaptation schedule. The capacity
    # trajectory and divergence samples are recorded AFTER the timed window
    # closes -- zero timing pollution (the staged divergences persist until
    # the next replay, so the post-call read sees this step's values).
    samples: dict[str, list[float]] = {name: [] for name in _PATH_ORDER}
    capacity_trajectory: list[int] = []
    divergence_samples: list[int] = []
    with torch.no_grad():
        for _ in range(warmup + reps):
            for name in _PATH_ORDER:
                elapsed = _timed_ms(step_fns[name])
                samples[name].append(elapsed)
                if name == "graph_adaptive":
                    traj_runner = mffs["graph_adaptive"].graph_runner_batched
                    if traj_runner is not None:
                        capacity_trajectory.append(int(traj_runner.capacity))
                        traj_divergences = cast(
                            Any, traj_runner
                        ).last_layer_divergences
                        if traj_divergences:
                            divergence_samples.append(max(traj_divergences))
    timings = {
        name: stats_from_samples(samples[name][warmup:]) for name in _PATH_ORDER
    }

    # Correctness sidecar: one extra call per path, logits collected.
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

    # Graph state probes: capture shape, budgets, and the adaptive trail.
    num_tokens = num_candidates * seq_len
    fixed_runner = mffs["graph_fixed"].graph_runner_batched
    adaptive_runner = mffs["graph_adaptive"].graph_runner_batched
    graph_captured = {
        "graph_fixed": fixed_runner is not None
        and tuple(fixed_runner.tokens_static.shape) == (num_candidates, seq_len),
        "graph_adaptive": adaptive_runner is not None
        and tuple(adaptive_runner.tokens_static.shape)
        == (num_candidates, seq_len),
    }
    policy = mffs["graph_adaptive"].graph_adaptive_policy
    divergences = (
        cast(Any, adaptive_runner).last_layer_divergences
        if adaptive_runner is not None
        else None
    )
    max_divergence = max(divergences) if divergences else None
    final_capacity = (
        int(cast(Any, adaptive_runner).capacity)
        if adaptive_runner is not None
        else None
    )
    fixed_capacity = int(_CAPACITY_RATIO * num_tokens)
    recapture_events = (
        [e for e in cast(Any, policy).events if e.get("kind") == "recapture"]
        if policy is not None
        else []
    )
    budget_exceeded_last = (
        bool(cast(Any, adaptive_runner).budget_exceeded)
        if adaptive_runner is not None
        else None
    )
    adaptive_validated = (
        (not budget_exceeded_last) if budget_exceeded_last is not None else None
    )

    h4_supported = bool(
        final_capacity is not None
        and final_capacity < fixed_capacity
        and max_divergence is not None
        and final_capacity >= max_divergence
        and len(recapture_events) >= 1
        and budget_exceeded_last is False
    )

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
        "num_tokens": num_tokens,
        "timings_ms": {name: timings[name].as_dict() for name in _PATH_ORDER},
        "speedups_p50": {k: round(v, 4) for k, v in speedups.items()},
        "graph_captured": graph_captured,
        "graph_validated_last_step": {
            "graph_fixed": (
                bool(not cast(Any, fixed_runner).budget_exceeded)
                if fixed_runner is not None
                else None
            ),
            "graph_adaptive": adaptive_validated,
        },
        "logits_parity_argmax_exact": parity,
        "adaptive_vs_fixed_allclose_1e-4": adaptive_close,
        "argmax_agreement_pairwise": pairwise,
        "adaptive_state": {
            "fixed_capacity": fixed_capacity,
            "final_capacity": final_capacity,
            "last_layer_divergences": divergences,
            "observed_max_layer_divergence": {
                "p50": (
                    statistics.median(divergence_samples)
                    if divergence_samples
                    else None
                ),
                "min": min(divergence_samples) if divergence_samples else None,
                "max": max(divergence_samples) if divergence_samples else None,
                "n": len(divergence_samples),
            },
            "capacity_trajectory": capacity_trajectory,
            "budget_exceeded_last_step": budget_exceeded_last,
            "recapture_count": len(recapture_events),
            "recapture_events": recapture_events,
            "capacity_tracks_demand": h4_supported,
        },
        "h1_adaptive_vs_fixed": _verdict(
            timings["graph_adaptive"], timings["graph_fixed"]
        ),
        "h2_adaptive_vs_baseline": _verdict(
            timings["graph_adaptive"], timings["baseline_eager"]
        ),
        "h4_capacity_tracking": "supported" if h4_supported else "refuted",
    }


def run_bench(warmup: int = 5, reps: int = 10, quick: bool = False) -> dict[str, Any]:
    """Run the three-path benchmark over the geometry x N scan.

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
        configs = [(1024, 16, 4)]

    driver = _gpu_driver_version()
    counts_supported = _cuda_profiler_supported()
    config_records = []
    for hidden_dim, num_layers, num_candidates in configs:
        GLOBAL_STABILITY_PROFILER.reset()
        record = _run_config(hidden_dim, num_layers, num_candidates, warmup, reps)
        config_records.append(record)
        print(
            f"H={hidden_dim:4d} L={num_layers:2d} N={num_candidates}: "
            + " | ".join(
                f"{label:8s} p50={record['timings_ms'][name]['p50']:.3f}ms"
                for label, name in _PATH_LABELS
            )
        )
        state = record["adaptive_state"]
        demand = state["observed_max_layer_divergence"]
        print(
            f"    capacity {state['fixed_capacity']} -> "
            f"{state['final_capacity']} (demand p50={demand['p50']} "
            f"max={demand['max']}), "
            f"{state['recapture_count']} recaptures, "
            f"H1={record['h1_adaptive_vs_fixed']} "
            f"H2={record['h2_adaptive_vs_baseline']}"
        )

    h1 = _overall([r["h1_adaptive_vs_fixed"] for r in config_records])
    h2 = _overall([r["h2_adaptive_vs_baseline"] for r in config_records])
    h3 = (
        "supported"
        if all(
            r["logits_parity_argmax_exact"]
            and r["adaptive_vs_fixed_allclose_1e-4"]
            for r in config_records
        )
        else "refuted"
    )
    h4 = _overall([r["h4_capacity_tracking"] for r in config_records])

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
                "mitigated by warmup + >=10 reps + p50 + per-rep round-robin"
            ),
            "run_date_utc": datetime.now(timezone.utc).isoformat(),
            "seed": _SEED,
            "model": "synthetic LLaMA-layout V=1000, per-config (H, L)",
            "capacity_ratio": _CAPACITY_RATIO,
            "adaptive_config": {
                "grow_factor": 1.25,
                "shrink_threshold": 0.5,
                "shrink_patience": 3,
                "min_recapture_interval": 2,
                "max_recaptures": 8,
            },
            "warmup": warmup,
            "reps": reps,
            "warmup_note": (
                "warmup covers capture plus the adaptation schedule (shrink "
                "patience 3 + throttle interval 2); timed steady state "
                "reflects the adapted capacity"
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
            "command": "python -m scripts.ar017_adaptive_bench"
            + (" --quick" if quick else "")
            + (f" --reps {reps}" if reps != 10 else "")
            + (f" --warmup {warmup}" if warmup != 5 else ""),
        },
        "hypotheses_declared": {
            "H1": "graph_adaptive significantly faster than graph_fixed "
            "(right-sized capacity cuts FFN recompute at mid-divergence loads)",
            "H2": "graph_adaptive significantly faster than baseline_eager "
            "(decisive P1-7 re-answer at deep geometry)",
            "H3": "three paths produce consistent logits (exact argmax "
            "agreement; adaptive-vs-fixed allclose 1e-4)",
            "H4": "adaptive capacity tracks the real demand (final capacity "
            "below the fixed capacity, at or above the measured max layer "
            "divergence, >=1 shrink recapture, zero budget violations after "
            "adaptation)",
        },
        "hypotheses_verdicts": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "p1_7_verdict": {
            "question": "is adaptive folded batched-graph verification "
            "faster than the raw baseline (P1-7, decisive re-answer)?",
            "overall": h2,
            "note": "recorded either way; per-config verdicts and speedups "
            "in configs[].h2_adaptive_vs_baseline / speedups_p50",
        },
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
    parser.add_argument("--reps", type=int, default=10, help="Timed reps per config")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup reps per config")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Single-config smoke subset (H=1024, L=16, N=4, T=512) for CI",
    )
    args = parser.parse_args()

    data = run_bench(warmup=args.warmup, reps=args.reps, quick=args.quick)
    out = write_results(data, args.out)
    print(f"\nH1 (adaptive vs fixed): {data['hypotheses_verdicts']['H1']}")
    print(f"H2 (adaptive vs baseline): {data['hypotheses_verdicts']['H2']}")
    print(f"H3 (three-path consistency): {data['hypotheses_verdicts']['H3']}")
    print(f"H4 (capacity tracks demand): {data['hypotheses_verdicts']['H4']}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
