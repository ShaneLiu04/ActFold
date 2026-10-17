"""AR016/T003: batched CUDA-graph verification benchmark (P1-7 wall-clock).

Measures the same-shape multi-candidate diffusion verification workload on
FOUR paths (per config N x T):

* ``baseline_eager`` — the raw (unfolded) model, N full forwards per step.
* ``folded_serial_eager`` — per-candidate serial folded calls (pre-AR015
  baseline of the folded engine).
* ``folded_batched_eager`` — one AR015 batched-branch eager call.
* ``folded_batched_graph`` — the AR016 batched CUDA graph path (capture on
  the first eligible call, replay afterwards).

Hypotheses (declared here, before measurement; verdicts come from data only
and are recorded as ``supported`` / ``refuted`` / ``inconclusive``):

* **H1** — batched eager is significantly faster than folded serial
  (direction ~N x).
* **H2** — batched graph further reduces per-step wall clock vs batched
  eager.
* **H3** — the four paths produce consistent logits (argmax agreement).

The P1-7 question ("is folded verification actually faster than the raw
baseline?") is answered by the separate ``p1_7_verdict`` field comparing
``folded_batched_graph`` with ``baseline_eager`` — recorded either way.

Timing is ``time.perf_counter`` around ``torch.cuda.synchronize()``
boundaries: host-side parent prefill, count readback, and per-branch publish
are part of the batched graph path's real cost, and CUDA-event timing would
exclude them. Paths are interleaved round-robin within every repetition
(RERUN_CHECKLIST §4.2) so GPU state drift affects all paths equally. Kernel
counts require a CUPTI-capable build; on ``LIBKINETO_NOCUPTI`` Windows builds
they are ``null`` with an explicit note (never fabricated, #39).

Run::

    python -m scripts.ar016_batched_bench [--quick] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from scripts.algo_experiments import TimingStats, stats_from_samples

DEFAULT_OUTPUT_PATH = Path("results/optimization/ar016_batched_bench.json")

_LAYERS = 8
_HIDDEN = 256
_VOCAB = 1000
_TAU = 0.95
_EPS = 1e-8
_CAPACITY_RATIO = 0.5
_SEED = 2026

_PATH_ORDER = (
    "baseline_eager",
    "folded_serial_eager",
    "folded_batched_eager",
    "folded_batched_graph",
)
_PATH_LABELS = tuple(
    (label, name) for label, name in zip(("baseline", "serial", "batch", "graph"), _PATH_ORDER)
)
_FULL_CONFIGS: tuple[tuple[int, int], ...] = (
    (2, 128),
    (2, 512),
    (4, 128),
    (4, 512),
    (8, 128),
    (8, 512),
)
_QUICK_CONFIGS: tuple[tuple[int, int], ...] = ((4, 512),)


# ---------------------------------------------------------------------------
# Synthetic LLaMA-layout decoder (detectable post_attention_layernorm->mlp
# chain), identical in structure to the AR002 benchmark fixture.
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
        hidden_dim: int = _HIDDEN,
        num_layers: int = _LAYERS,
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
# Workload construction (FR5: parent [1, T] cached once; N same-parent
# equal-length candidates with mid-sequence divergence + independent suffix;
# same shape across steps so the graph can replay).
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


def _make_mff(model: nn.Module, use_cuda_graph: bool) -> ManualFoldedForward:
    cache = ActivationCache(max_entries_per_layer=8192, device="cuda")
    gate = SimilarityGate(tau=_TAU, metric="cosine", eps=_EPS)
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        split_layers=True,
        split_min_tokens=1,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=_CAPACITY_RATIO,
    )


def _timed_ms(step_fn: Callable[[], Any]) -> float:
    """Wall-clock one step in ms, synchronize-bounded (perf_counter)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _cuda_profiler_supported() -> bool:
    """Probe whether this build records CUDA profiler events (CUPTI)."""
    try:
        x = torch.ones(4, device="cuda")
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            x = x + 1
            torch.cuda.synchronize()
        return any(evt.device_type == torch.profiler.DeviceType.CUDA for evt in prof.key_averages())
    except Exception:  # noqa: BLE001 - probe must never raise
        return False


# ---------------------------------------------------------------------------
# Per-config benchmark
# ---------------------------------------------------------------------------
def _build_step_fns(
    model: nn.Module,
    num_candidates: int,
    seq_len: int,
) -> tuple[
    dict[str, Callable[[], Any]],
    dict[str, ManualFoldedForward],
    torch.Tensor,
]:
    """Build the four per-step callables (one verification of N candidates)."""
    mffs: dict[str, ManualFoldedForward] = {
        "folded_serial_eager": _make_mff(model, use_cuda_graph=False),
        "folded_batched_eager": _make_mff(model, use_cuda_graph=False),
        "folded_batched_graph": _make_mff(model, use_cuda_graph=True),
    }
    parent_tokens = torch.randint(0, _VOCAB, (1, seq_len), device="cuda")
    candidates = _make_candidates(parent_tokens, num_candidates, seq_len)
    branch_ids = [f"b{i}" for i in range(num_candidates)]
    serial_ids = [f"s{i}" for i in range(num_candidates)]
    graph_ids = [f"g{i}" for i in range(num_candidates)]

    for mff in mffs.values():
        mff(parent_tokens, branch_id="parent")

    step_fns: dict[str, Callable[[], Any]] = {
        "baseline_eager": lambda: [model(candidates[i : i + 1]) for i in range(num_candidates)],
        "folded_serial_eager": lambda: [
            mffs["folded_serial_eager"](
                candidates[i : i + 1],
                branch_id=serial_ids[i],
                parent_branch_id="parent",
            )
            for i in range(num_candidates)
        ],
        "folded_batched_eager": lambda: mffs["folded_batched_eager"](
            candidates,
            branch_id=None,
            branch_ids=branch_ids,
            parent_branch_id="parent",
        ),
        "folded_batched_graph": lambda: mffs["folded_batched_graph"](
            candidates,
            branch_id=None,
            branch_ids=graph_ids,
            parent_branch_id="parent",
        ),
    }
    return step_fns, mffs, candidates


def _argmax_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of positions where two [N, T] logits agree on argmax."""
    return float((a.argmax(-1) == b.argmax(-1)).float().mean().item())


def _mean_stable_ratio(branch_ids: Sequence[str]) -> float | None:
    """Mean stable ratio over a path's branches (profiler is global)."""
    ratios = []
    for branch_id in branch_ids:
        profile = GLOBAL_STABILITY_PROFILER.get_profile(branch_id)
        if profile is not None and profile.layer_stats:
            ratios.append(profile.mean_stable_ratio)
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def _verdict(faster: TimingStats, slower: TimingStats) -> str:
    """supported iff the p50 delta exceeds 2x the merged std (else decide)."""
    delta = slower.p50 - faster.p50
    merged_std = (faster.std**2 + slower.std**2) ** 0.5
    if delta > 2 * merged_std:
        return "supported"
    if delta < -2 * merged_std:
        return "refuted"
    return "inconclusive"


def _run_config(
    model: nn.Module,
    num_candidates: int,
    seq_len: int,
    warmup: int,
    reps: int,
) -> dict[str, Any]:
    """Benchmark one (N, T) config on the four interleaved paths."""
    step_fns, mffs, candidates = _build_step_fns(model, num_candidates, seq_len)

    # Per-repetition round-robin interleaving (RERUN_CHECKLIST 4.2).
    samples: dict[str, list[float]] = {name: [] for name in _PATH_ORDER}
    with torch.no_grad():
        for _ in range(warmup + reps):
            for name in _PATH_ORDER:
                elapsed = _timed_ms(step_fns[name])
                samples[name].append(elapsed)
    timings = {name: stats_from_samples(samples[name][warmup:]) for name in _PATH_ORDER}

    # Correctness sidecar: one extra call per path, logits collected.
    logits: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in _PATH_ORDER:
            out = step_fns[name]()
            if name in ("baseline_eager", "folded_serial_eager"):
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
    folded_close = bool(
        torch.allclose(
            logits["folded_batched_graph"],
            logits["folded_serial_eager"],
            atol=1e-4,
            rtol=1e-4,
        )
        and torch.allclose(
            logits["folded_batched_graph"],
            logits["folded_batched_eager"],
            atol=1e-4,
            rtol=1e-4,
        )
    )

    # The runner materializes during the first (warmup) call, so probe it
    # only after the timing loop.
    runner = mffs["folded_batched_graph"].graph_runner_batched
    graph_captured = runner is not None and tuple(runner.tokens_static.shape) == (
        num_candidates,
        seq_len,
    )
    graph_validated = bool(runner is not None and not cast(Any, runner).budget_exceeded)

    stable_ratios = {
        "folded_serial_eager": _mean_stable_ratio([f"s{i}" for i in range(num_candidates)]),
        "folded_batched_eager": _mean_stable_ratio([f"b{i}" for i in range(num_candidates)]),
        "folded_batched_graph": _mean_stable_ratio([f"g{i}" for i in range(num_candidates)]),
    }

    h1 = _verdict(timings["folded_batched_eager"], timings["folded_serial_eager"])
    h2 = _verdict(timings["folded_batched_graph"], timings["folded_batched_eager"])
    p1_7 = _verdict(timings["folded_batched_graph"], timings["baseline_eager"])
    speedup_vs_baseline = (
        timings["baseline_eager"].p50 / timings["folded_batched_graph"].p50
        if timings["folded_batched_graph"].p50 > 0
        else float("inf")
    )

    return {
        "num_candidates": num_candidates,
        "seq_len": seq_len,
        "timings_ms": {name: timings[name].as_dict() for name in _PATH_ORDER},
        "graph_captured": graph_captured,
        "graph_validated_last_step": graph_validated,
        "logits_parity_argmax_exact": parity,
        "folded_paths_allclose_1e-4": folded_close,
        "argmax_agreement_pairwise": pairwise,
        "mean_stable_ratio": {k: v for k, v in stable_ratios.items()},
        "h1_batched_vs_serial": h1,
        "h2_graph_vs_batched_eager": h2,
        "p1_7_graph_vs_baseline": {
            "verdict": p1_7,
            "speedup_p50": round(speedup_vs_baseline, 4),
        },
    }


def _overall(verdicts: Sequence[str]) -> str:
    """Strict aggregation: supported everywhere, refuted anywhere, else mixed."""
    if all(v == "supported" for v in verdicts):
        return "supported"
    if any(v == "refuted" for v in verdicts):
        return "refuted"
    return "inconclusive"


def run_bench(warmup: int = 3, reps: int = 10, quick: bool = False) -> dict[str, Any]:
    """Run the four-path benchmark over the (N, T) scan.

    Raises:
        RuntimeError: If no CUDA device is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("run_bench requires a CUDA device")

    torch.manual_seed(_SEED)
    model = _Decoder().to("cuda")
    model.eval()
    configs = _QUICK_CONFIGS if quick else _FULL_CONFIGS
    counts_supported = _cuda_profiler_supported()

    config_records = []
    for num_candidates, seq_len in configs:
        GLOBAL_STABILITY_PROFILER.reset()
        record = _run_config(model, num_candidates, seq_len, warmup, reps)
        config_records.append(record)
        print(
            f"N={num_candidates:2d} T={seq_len:3d}: "
            + " | ".join(
                f"{label:8s} p50={record['timings_ms'][name]['p50']:.3f}ms"
                for label, name in _PATH_LABELS
            )
        )

    h1 = _overall([r["h1_batched_vs_serial"] for r in config_records])
    h2 = _overall([r["h2_graph_vs_batched_eager"] for r in config_records])
    h3 = "supported" if all(r["logits_parity_argmax_exact"] for r in config_records) else "refuted"
    p1_7 = _overall([r["p1_7_graph_vs_baseline"]["verdict"] for r in config_records])

    return {
        "meta": {
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": cast(Any, torch.version).cuda,
            "platform_note": (
                "Windows WDDM display-driver mode (no clock locking); noise "
                "mitigated by warmup + >=10 reps + p50 + per-rep round-robin"
            ),
            "run_date_utc": datetime.now(timezone.utc).isoformat(),
            "seed": _SEED,
            "model": f"synthetic LLaMA-layout L={_LAYERS} H={_HIDDEN} V={_VOCAB}",
            "capacity_ratio": _CAPACITY_RATIO,
            "warmup": warmup,
            "reps": reps,
            "interleaving": ("round-robin per repetition: " + " -> ".join(_PATH_ORDER)),
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
            "command": "python -m scripts.ar016_batched_bench" + (" --quick" if quick else ""),
        },
        "hypotheses_declared": {
            "H1": "batched eager significantly faster than folded serial (~Nx)",
            "H2": "batched graph further reduces per-step wall clock vs batched eager",
            "H3": "four paths produce consistent logits (exact argmax agreement)",
        },
        "hypotheses_verdicts": {"H1": h1, "H2": h2, "H3": h3},
        "p1_7_verdict": {
            "question": "is folded batched-graph verification faster than the "
            "raw baseline (P1-7)?",
            "overall": p1_7,
            "note": "recorded either way; per-config verdicts and speedups in "
            "configs[].p1_7_graph_vs_baseline",
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
    parser.add_argument("--warmup", type=int, default=3, help="Warmup reps per config")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Single-config smoke subset (N=4, T=512) for CI",
    )
    args = parser.parse_args()

    data = run_bench(warmup=args.warmup, reps=args.reps, quick=args.quick)
    out = write_results(data, args.out)
    print(f"\nH1 (batched vs serial): {data['hypotheses_verdicts']['H1']}")
    print(f"H2 (graph vs batched eager): {data['hypotheses_verdicts']['H2']}")
    print(f"H3 (four-path consistency): {data['hypotheses_verdicts']['H3']}")
    print(f"P1-7 (graph vs baseline): {data['p1_7_verdict']['overall']}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
