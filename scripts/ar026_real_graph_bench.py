"""AR026/T005: real-checkpoint four-mode graph bench (production capture).

Runs the REAL ``LLaDASampler`` / ``DreamSampler`` generation loop on the
local LLaDA-8B / Dream-v0-7B checkpoints in FOUR modes over one shared
model instance, prompt set, and seed -- the AR023 protocol carried over
with the graph dial as the ONLY inter-mode difference (design 4.2.2):

* ``baseline`` -- the family official recipe, ``folded_model=None``.
* ``folded-batched`` -- N=4 / top_k=10 eager: the AR023 anchor mode.
* ``graph-fixed`` -- the anchor's MFF plus ``use_cuda_graph=True`` with a
  fixed divergent capacity (ratio 0.5, pool 4, adaptive off).
* ``graph-adaptive`` -- the same plus ``graph_adaptive_capacity=True``.

Single-dial discipline: the two graph modes differ from the eager anchor
ONLY in graph knobs (``MODE_KNOBS``); the sampling config (N / top-k /
seed / prompts) is identical across the three folded modes.

Hypotheses (declared before measurement; frozen wording from srs 3.5 /
design 4.2.2; verdicts from data only):

* **H1 graph parity** (mechanism-swap invariance,
  ``h1_graph_parity_verdict``): the graph replay is a mechanism swap,
  not a quality mechanism -- each graph mode's committed rows must be
  indistinguishable from the eager anchor's: (a) per-prompt posthoc
  overall-mean within ``_H1_PARITY_ATOL`` of the anchor's; (b)
  decode-length equality on every prompt; (c) merged block-grid
  exact-match vs the anchor at rate 1.0. Any ``None`` posthoc mean
  refutes (no fabricated numbers). A graph mode with ZERO captures is
  NOT a parity pass: its verdict leg is ``inconclusive`` (the mode ran
  eager by fallback -- the engagement face, never a silent substitute).
  The AR023 ``h1_quality_verdict`` threshold semantics do NOT transfer
  (the 0.5-gap-shrink face presumes a quality mechanism; the graph is
  not one) -- it stays re-exported for cross-artifact reconciliation.
* **H2** speed: ``p50(baseline) / p50(graph-adaptive) > 1.0`` (strict;
  no magnitude presumption). The graph-vs-anchor ratio is REPORT-ONLY
  (pre-registered no-direction: WDDM + small-geometry capture
  amortization risk).
* **H4** full-stack exercise: publish-ledger within the AR021 c=4 bound
  per folded mode; the degradation canary is the REAL face now --
  non-structural graph-path eager fallbacks during the timed window
  must stay at ZERO per folded mode; stable-ratio recording (report).

Reproduction self-check: the baseline double-run strict row equality
(AR023 semantics); a mismatch blocks ALL verdicts. Degraded-family and
failure semantics: the AR023 blocked-artifact face verbatim (status
``blocked``, zero fabricated numbers, probe cross-reference).

Protocol: seed 2026, the AR023 probe prompt set (P=2; one under
``--quick``), num_tokens 128 (quick 64), warmup one discarded sample per
(mode, prompt) -- the graph modes capture during warmup -- per-rep
four-mode round-robin (rep r starts at r % 4), host perf_counter + cuda
synchronize (#39), reps=30 pre-committed single-shot verdict. ``--quick``
writes a separate ``_quick`` artifact, never the formal one.

Geometry inheritance (pre-registered waiver): the EVR >=4-shape floor is
explicitly waived via ``GEOMETRY_INHERITANCE_NOTE`` (the bench inherits
the AR023 anchor geometry; the reason is recorded in the artifact meta).

Run (AR019 venv, transformers==4.53.1)::

    python -m scripts.ar026_real_graph_bench --family llada|dream
        [--quick] [--reps 30] [--models-root PATH] [--out PATH]
        [--probe-artifact PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import scripts.ar023_cross_family_bench as _ar023

# ---------------------------------------------------------------------------
# Frozen protocol constants (AR023 carried over; identity re-exports)
# ---------------------------------------------------------------------------
SEED = _ar023.SEED
REPS_DEFAULT = _ar023.REPS_DEFAULT
QUICK_REPS = _ar023.QUICK_REPS
FULL_NUM_TOKENS = _ar023.FULL_NUM_TOKENS
QUICK_NUM_TOKENS = _ar023.QUICK_NUM_TOKENS
BLOCK_GRID = _ar023.BLOCK_GRID
NUM_CANDIDATES = _ar023.NUM_CANDIDATES
CANDIDATE_TOP_K = _ar023.CANDIDATE_TOP_K
SPECULATIVE_STEPS = _ar023.SPECULATIVE_STEPS
LEDGER_C = _ar023.LEDGER_C
TAU = _ar023.TAU
CACHE_BUDGET = _ar023.CACHE_BUDGET
FAMILY_RECIPES = _ar023.FAMILY_RECIPES
FAMILIES = _ar023.FAMILIES
DEFAULT_PROBE_ARTIFACT = _ar023.DEFAULT_PROBE_ARTIFACT
SCHEMA_VERSION = 1

# Identity re-exports (assign-form, the AR024 pattern): a local
# re-implementation -- even a behaviorally identical one -- is drift.
h1_quality_verdict = _ar023.h1_quality_verdict
h2_speed_verdict = _ar023.h2_speed_verdict
h4_fullstack_verdict = _ar023.h4_fullstack_verdict
reproduction_verdict = _ar023.reproduction_verdict
mark_all_blocked = _ar023.mark_all_blocked
build_blocked_artifact = _ar023.build_blocked_artifact
write_results = _ar023.write_results
probe_conclusion = _ar023.probe_conclusion
_load_family_model = _ar023._load_family_model
_prepare_prompts = _ar023._prepare_prompts
_decode_length = _ar023._decode_length
_posthoc_face = _ar023._posthoc_face
_block_exact_counts = _ar023._block_exact_counts
_position_match_counts = _ar023._position_match_counts
_stats_dict = _ar023._stats_dict

# ---------------------------------------------------------------------------
# Frozen four-mode matrix (single-dial discipline, design 4.2.2 point 5)
# ---------------------------------------------------------------------------
MODES = ("baseline", "folded-batched", "graph-fixed", "graph-adaptive")
FOLDED_MODES = ("folded-batched", "graph-fixed", "graph-adaptive")
GRAPH_MODES = ("graph-fixed", "graph-adaptive")
_ANCHOR = "folded-batched"
_TOP = "graph-adaptive"

MODE_KNOBS: dict[str, dict[str, Any]] = {
    "baseline": {"folded": False},
    "folded-batched": {"folded": True, "use_cuda_graph": False},
    "graph-fixed": {
        "folded": True,
        "use_cuda_graph": True,
        "graph_capacity_ratio": 0.5,
        "graph_runner_pool_size": 4,
        "graph_adaptive_capacity": False,
    },
    "graph-adaptive": {
        "folded": True,
        "use_cuda_graph": True,
        "graph_capacity_ratio": 0.5,
        "graph_runner_pool_size": 4,
        "graph_adaptive_capacity": True,
    },
}

SCHEMA_NAME = "ar026_real_graph_bench"
DEFAULT_OUTPUT_PATH = Path("results/optimization/ar026_real_graph_bench.json")

GEOMETRY_INHERITANCE_NOTE = (
    "Pre-registered EVR >=4-shape floor waiver: this bench inherits the AR023 "
    "anchor geometry (2 prompts x one num_tokens => <=2 pool keys per mode) "
    "because the binding constraint on this hardware is VRAM headroom, not "
    "shape coverage -- a 16GB card carrying a bf16 8B checkpoint (~15GB) "
    "leaves ~1GB, and every graph pool entry costs capture-side VRAM (the A5 "
    "amortization caveat; the AR021 LRU eviction exists for exactly this). "
    "Formal multi-geometry expansion (>=4 shape points) is recorded as a "
    "follow-up AR face, never silently skipped."
)

# The per-prompt posthoc parity tolerance (H1 face a): the graph replay is
# bit-exact against the eager anchor per the AR002 contract, so any real
# divergence exceeds this by orders of magnitude -- it is afloat-noise
# guard, not a quality budget.
_H1_PARITY_ATOL = 1e-6


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
def quick_output_path() -> Path:
    """The formal artifact path with a ``_quick`` suffix."""
    return DEFAULT_OUTPUT_PATH.with_name(
        DEFAULT_OUTPUT_PATH.stem + "_quick" + DEFAULT_OUTPUT_PATH.suffix
    )


def resolve_output_path(family: str, quick: bool, out_override: str | Path | None = None) -> Path:
    """Explicit override wins; else quick path under ``--quick``, else formal.

    ``family`` keeps the AR023 signature but does not key the path: the
    AR026 artifact is a single shared file (the four-mode matrix is the
    commitment; the family is recorded in the artifact meta).
    """
    _ = family
    if out_override is not None:
        return Path(out_override)
    return quick_output_path() if quick else DEFAULT_OUTPUT_PATH


# ---------------------------------------------------------------------------
# Frozen verdict: H1 graph parity (pre-registered, mechanism-swap invariance)
# ---------------------------------------------------------------------------
def h1_graph_parity_verdict(
    per_prompt: list[dict[str, Any]],
    engagement: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """H1 graph parity: each graph mode vs the eager anchor (frozen).

    Faces per graph mode (all three, else the mode leg refutes):
    (a) per-prompt posthoc overall-mean within ``_H1_PARITY_ATOL`` of the
    anchor's; (b) decode-length equality on every prompt; (c) merged
    block-grid exact-match vs the anchor at rate 1.0. Any ``None``
    posthoc mean refutes (no fabricated numbers). A graph mode with zero
    captures is ``inconclusive`` (eager fallback -- never a silent parity
    pass); overall precedence: refuted > inconclusive > supported.
    """
    mode_verdicts: dict[str, Any] = {}
    any_refuted = False
    any_not_engaged = False
    for mode in GRAPH_MODES:
        eng = engagement.get(mode) or {}
        engaged = int(eng.get("captures", 0) or 0) > 0
        face_a = True
        face_b = True
        none_mean = False
        deltas: list[float | None] = []
        for entry in per_prompt:
            means = entry.get("posthoc_overall_mean", {})
            anchor_mean = means.get(_ANCHOR)
            mode_mean = means.get(mode)
            if anchor_mean is None or mode_mean is None:
                none_mean = True
                deltas.append(None)
                face_a = False
            else:
                delta = mode_mean - anchor_mean
                deltas.append(delta)
                if abs(delta) > _H1_PARITY_ATOL:
                    face_a = False
            lens = entry.get("decode_length", {})
            if lens.get(mode) != lens.get(_ANCHOR):
                face_b = False

        def _merged_rate(key: str) -> float | None:
            matched = 0
            compared = 0
            for entry in per_prompt:
                counts = (entry.get(key) or {}).get(mode) or {}
                matched += int(counts.get("matched", 0) or 0)
                compared += int(counts.get("compared", 0) or 0)
            return (matched / compared) if compared else None

        rate = _merged_rate("block_exact_vs_anchor")
        face_c = rate is not None and rate == 1.0

        if not engaged:
            verdict = "inconclusive"
            any_not_engaged = True
        elif not (face_a and face_b and face_c):
            verdict = "refuted"
            any_refuted = True
        else:
            verdict = "supported"
        mode_verdicts[mode] = {
            "verdict": verdict,
            "engaged": engaged,
            "faces": {
                "a_posthoc_parity_atol_1e-6": bool(face_a),
                "b_decode_length_equality": bool(face_b),
                "c_block_exact_full_match_vs_anchor": bool(face_c),
            },
            "numbers": {
                "posthoc_delta_vs_anchor": [None if d is None else round(d, 6) for d in deltas],
                "block_exact_rate_vs_anchor": None if rate is None else round(rate, 6),
                "captures": int(eng.get("captures", 0) or 0),
                "replays": int(eng.get("replays", 0) or 0),
                "evictions": int(eng.get("evictions", 0) or 0),
                "none_posthoc_mean": bool(none_mean),
            },
        }

    if any_refuted:
        overall = "refuted"
    elif any_not_engaged:
        overall = "inconclusive"
    else:
        overall = "supported"
    return {
        "verdict": overall,
        "modes": mode_verdicts,
        "rule": (
            "supported iff every graph mode captured >=1 runner AND holds all "
            "three parity faces vs the eager anchor; a zero-capture mode is "
            "inconclusive (eager fallback, never a silent parity pass); "
            "precedence: refuted > inconclusive > supported"
        ),
    }


# ---------------------------------------------------------------------------
# Artifact schema (own assembler, AR023 key set, AR026 schema name)
# ---------------------------------------------------------------------------
def build_artifact(
    family: str,
    status: str,
    modes: dict[str, Any] | None,
    quality: dict[str, Any] | None,
    verdicts: dict[str, Any],
    reproduction: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the benchmark artifact (the ST recomputation surface)."""
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "status": status,
        "modes": modes,
        "quality": quality,
        "verdicts": verdicts,
        "reproduction": reproduction,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """``--family`` defaults to llada (dream allowed; meta carries the choice)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="llada", choices=FAMILIES)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke subset: reps=3, single prompt, num_tokens=64",
    )
    parser.add_argument("--reps", type=int, default=REPS_DEFAULT, help="Timed reps")
    parser.add_argument(
        "--models-root",
        default=None,
        help="Checkpoint root (default: <HF_HOME>/ar023_models; the probe layout)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: the formal artifact, or the _quick path)",
    )
    parser.add_argument(
        "--probe-artifact",
        default=DEFAULT_PROBE_ARTIFACT,
        help="The probe artifact carrying the family conclusions",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Lazy run layer (heavy imports stay inside; the CI surface is pure logic)
# ---------------------------------------------------------------------------
def _make_mode_mff(raw_model: Any, mode: str) -> Any:
    """One instrumented ManualFoldedForward per folded mode (MFF_C pattern).

    Knobs come from the frozen ``MODE_KNOBS`` row minus the ``folded``
    semantic marker (the ctor has no ``folded`` parameter -- every folded
    mode constructs an MFF; baseline passes ``folded_model=None``): the
    eager anchor and the two graph modes share cache/gate/split settings,
    so the graph knobs are the ONLY dial.
    """
    from actfold.core import ActivationCache, SimilarityGate
    from actfold.models.architecture_utils import ManualFoldedForward

    class _BenchMFF(ManualFoldedForward):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.serial_calls = 0
            self.batched_calls = 0

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("branch_ids") is not None:
                self.batched_calls += 1
            else:
                self.serial_calls += 1
            return super().forward(*args, **kwargs)

    cache = ActivationCache(max_entries_per_layer=CACHE_BUDGET, device="cuda")
    gate = SimilarityGate(tau=TAU, metric="cosine")
    knobs = {k: v for k, v in MODE_KNOBS[mode].items() if k != "folded"}
    return _BenchMFF(raw_model, cache=cache, gate=gate, scheduler=None, split_layers=False, **knobs)


def _make_mode_sampler(
    family: str, mode: str, llm: Any, num_tokens: int, mff: Any
) -> tuple[Any, Any]:
    """Family sampler for one mode (official recipe; spec knobs per mode).

    Unlike AR023's mode-coupled ``_make_sampler``, ALL three folded modes
    carry N=4 / top_k=10 here (single-dial discipline: the sampling
    config is identical, only the graph knobs differ).
    """
    import dataclasses

    from actfold.models.dream_sampler import DreamSampler, DreamSamplerConfig
    from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig

    recipe = FAMILY_RECIPES[family]
    if family == "llada":
        config: Any = LLaDASamplerConfig(
            num_steps=int(recipe["num_steps"]),
            num_tokens=num_tokens,
            block_size=int(recipe["block_size"]),
            remasking=str(recipe["remasking"]),
            seed=SEED,
        )
    else:
        config = DreamSamplerConfig(
            num_steps=num_tokens,
            num_tokens=num_tokens,
            seed=SEED,
            speculative_steps=SPECULATIVE_STEPS,
        )
    if mode in FOLDED_MODES:
        config = dataclasses.replace(
            config, num_block_candidates=NUM_CANDIDATES, candidate_top_k=CANDIDATE_TOP_K
        )
    sampler_cls = LLaDASampler if family == "llada" else DreamSampler
    folded = mff if mode != "baseline" else None
    return sampler_cls(llm, config=config), folded


def _engagement(stats: dict[str, Any]) -> dict[str, int]:
    """Merged captures/replays/evictions over both pool sections."""
    captures = 0
    replays = 0
    evictions = 0
    for section in ("serial", "batched"):
        for key, entry in (stats.get(section) or {}).items():
            if key == "pool_size":
                continue
            captures += int(entry.get("captures", 0) or 0)
            replays += int(entry.get("replays", 0) or 0)
            evictions += int(entry.get("evictions", 0) or 0)
    return {"captures": captures, "replays": replays, "evictions": evictions}


def _serializable_pool_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe copy of ``graph_pool_stats``.

    Pool keys are ``(shape, mask-object)`` tuples; the mask identity is
    not artifact-relevant (the shape is the semantic key), so keys become
    ``str(shape)``. The capture diagnostics carry ``to_json_dict``
    (T002); the overflow is a plain int.
    """
    out: dict[str, Any] = {}
    for section in ("serial", "batched"):
        sec = stats.get(section) or {}
        converted: dict[str, Any] = {}
        for key, entry in sec.items():
            if key == "pool_size":
                continue
            shape = key[0] if isinstance(key, tuple) else key
            converted[str(shape)] = dict(entry)
        out[section] = {"pool_size": int(sec.get("pool_size", 0) or 0), **converted}
    out["degradation_fallback_count"] = stats.get("degradation_fallback_count")
    out["structural_eager_count"] = stats.get("structural_eager_count")
    out["capture_diagnostics_overflow"] = stats.get("capture_diagnostics_overflow")
    out["capture_diagnostics"] = [
        d.to_json_dict() if hasattr(d, "to_json_dict") else d
        for d in stats.get("capture_diagnostics") or []
    ]
    return out


def run_graph_bench(
    family: str,
    models_root: Path,
    reps: int,
    quick: bool,
    conclusion: dict[str, Any],
    probe_ref: str,
) -> dict[str, Any]:
    """Run the four-mode graph benchmark for one family (CUDA required)."""
    import torch

    from scripts.algo_experiments import stats_from_samples

    if not torch.cuda.is_available():
        raise RuntimeError("run_graph_bench requires a CUDA device")
    from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

    num_tokens = QUICK_NUM_TOKENS if quick else FULL_NUM_TOKENS
    llm, dtype, load_meta = _load_family_model(family, models_root)
    device = "cuda"
    prompts = _prepare_prompts(llm, quick, device)
    eos_token_id = llm.tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("tokenizer has no eos_token_id")

    sampler_pairs: dict[str, tuple[Any, Any]] = {}
    mffs: dict[str, Any] = {}
    for mode in MODES:
        if mode == "baseline":
            mff = None
        else:
            # Independent instances per folded mode (the AR022 MFF_C
            # pattern: own cache/gate/counters -- per-mode isolation).
            mff = _make_mode_mff(llm.model, mode)
            mffs[mode] = mff
        sampler, folded = _make_mode_sampler(family, mode, llm, num_tokens, mff)
        sampler_pairs[mode] = (sampler, folded)

    def _sample(mode: str, prompt: dict[str, Any]) -> Any:
        sampler, folded = sampler_pairs[mode]
        return sampler.sample(prompt["ids"], folded_model=folded)

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(
        f"[ar026-bench] family={family} dtype={dtype} "
        f"num_tokens={num_tokens} reps={reps} "
        f"vram_free={free_bytes / 1e6:.0f}MB",
        flush=True,
    )

    # -- reproduction self-check: baseline double-run (untimed, strict rows)
    repro_rows_a: list[list[int]] = []
    repro_rows_b: list[list[int]] = []
    for prompt in prompts:
        row_a = _sample("baseline", prompt).sequences[0].tolist()
        row_b = _sample("baseline", prompt).sequences[0].tolist()
        repro_rows_a.append(row_a)
        repro_rows_b.append(row_b)
    reproduction = reproduction_verdict(repro_rows_a, repro_rows_b)
    print(f"[ar026-bench] reproduction double-run ok={reproduction['ok']}", flush=True)

    # -- warmup: one full discarded sample per (mode, prompt); the graph
    #    modes run their production capture in this window
    for prompt in prompts:
        for mode in MODES:
            _sample(mode, prompt)
    print("[ar026-bench] warmup complete (graph capture window)", flush=True)

    degradation_before = {
        mode: int(mffs[mode].graph_pool_stats["degradation_fallback_count"])
        for mode in FOLDED_MODES
    }

    # -- timed reps: per-rep four-mode round-robin (rep r starts at r % 4)
    samples: dict[tuple[str, int], list[float]] = {
        (mode, i): [] for mode in MODES for i in range(len(prompts))
    }
    first_rows: dict[tuple[str, int], list[int]] = {}
    cross_rep_deterministic = True
    for rep in range(reps):
        order = MODES[rep % len(MODES) :] + MODES[: rep % len(MODES)]
        for prompt_index, prompt in enumerate(prompts):
            for mode in order:
                folded = sampler_pairs[mode][1]
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = sampler_pairs[mode][0].sample(prompt["ids"], folded_model=folded)
                torch.cuda.synchronize()
                samples[(mode, prompt_index)].append((time.perf_counter() - start) * 1000.0)
                row = out.sequences[0].tolist()
                key = (mode, prompt_index)
                if key not in first_rows:
                    first_rows[key] = row
                elif row != first_rows[key]:
                    cross_rep_deterministic = False
        print(f"[ar026-bench] rep {rep + 1}/{reps} done", flush=True)

    # -- canaries: the REAL degradation face (non-structural graph-path
    #    eager fallbacks during the timed window must stay at ZERO)
    canaries: dict[str, bool] = {
        mode: (
            int(mffs[mode].graph_pool_stats["degradation_fallback_count"])
            == degradation_before[mode]
        )
        for mode in FOLDED_MODES
    }

    # -- sidecar faces: ledger + stable ratio per folded mode (untimed)
    ledgers: dict[str, bool] = {}
    stable_ratios: dict[str, float] = {}
    num_layers = int(sampler_pairs["baseline"][0].model.model.config.num_hidden_layers)
    for mode in FOLDED_MODES:
        mff = mffs[mode]
        ops0 = mff.cache.write_op_count
        serial0 = mff.serial_calls
        batched0 = mff.batched_calls
        # The samplers reset their folding chain every ``sample()`` call, so
        # branch ids repeat across calls -- reset the global profiler
        # instead: every profile recorded below belongs to this mode's
        # sidecar window by construction.
        GLOBAL_STABILITY_PROFILER.reset()
        for prompt in prompts:
            _sample(mode, prompt)
        ops_delta = mff.cache.write_op_count - ops0
        serial_events = mff.serial_calls - serial0
        batched_events = mff.batched_calls - batched0
        bound = LEDGER_C * (
            (num_layers + 1) * serial_events + (num_layers + NUM_CANDIDATES) * batched_events
        )
        ledgers[mode] = bool(ops_delta <= bound)
        window_profiles = GLOBAL_STABILITY_PROFILER.all_profiles()
        if window_profiles:
            ratios = [profile.mean_stable_ratio for profile in window_profiles.values()]
            stable_ratios[mode] = round(sum(ratios) / len(ratios), 6)
        print(
            f"[ar026-bench] sidecar {mode}: ops={ops_delta} bound={bound} "
            f"serial={serial_events} batched={batched_events}",
            flush=True,
        )

    # -- quality faces (untimed): decode lengths, posthoc, block exact
    rows: dict[tuple[str, int], list[int]] = first_rows
    quality_details: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        region_start = prompt["prompt_len"]
        lengths: dict[str, int] = {}
        posthoc: dict[str, Any] = {}
        for mode in MODES:
            row = rows[(mode, prompt_index)]
            lengths[mode] = _decode_length(row, region_start, num_tokens, eos_token_id)
        for mode in MODES:
            row = rows[(mode, prompt_index)]
            posthoc[mode] = _posthoc_face(llm, row, region_start, lengths[mode], family)
        base_row = rows[("baseline", prompt_index)]
        anchor_row = rows[(_ANCHOR, prompt_index)]
        block_exact: dict[str, Any] = {}
        position_match: dict[str, Any] = {}
        block_exact_vs_anchor: dict[str, Any] = {}
        position_match_vs_anchor: dict[str, Any] = {}
        for mode in FOLDED_MODES:
            other_row = rows[(mode, prompt_index)]
            common = min(lengths["baseline"], lengths[mode])
            block_exact[mode] = _block_exact_counts(base_row, other_row, region_start, common)
            position_match[mode] = _position_match_counts(base_row, other_row, region_start, common)
        for mode in GRAPH_MODES:
            other_row = rows[(mode, prompt_index)]
            common_anchor = min(lengths[_ANCHOR], lengths[mode])
            block_exact_vs_anchor[mode] = _block_exact_counts(
                anchor_row, other_row, region_start, common_anchor
            )
            position_match_vs_anchor[mode] = _position_match_counts(
                anchor_row, other_row, region_start, common_anchor
            )
        quality_details.append(
            {
                "prompt_index": prompt_index,
                "prompt_text": prompt["text"],
                "decode_length": lengths,
                "posthoc_confidence": posthoc,
                "block_exact": block_exact,
                "position_match": position_match,
                "block_exact_vs_anchor": block_exact_vs_anchor,
                "position_match_vs_anchor": position_match_vs_anchor,
            }
        )

    # -- verdicts
    engagement = {mode: _engagement(mffs[mode].graph_pool_stats) for mode in GRAPH_MODES}
    h1_input = [
        {
            "prompt_index": entry["prompt_index"],
            "posthoc_overall_mean": {
                mode: entry["posthoc_confidence"][mode]["overall_mean"] for mode in MODES
            },
            "decode_length": entry["decode_length"],
            "block_exact_vs_anchor": entry["block_exact_vs_anchor"],
        }
        for entry in quality_details
    ]
    h1 = h1_graph_parity_verdict(h1_input, engagement)

    def _mode_p50(mode: str) -> float:
        per_prompt_p50 = [stats_from_samples(samples[(mode, i)]).p50 for i in range(len(prompts))]
        return sum(per_prompt_p50) / len(per_prompt_p50)

    h2 = h2_speed_verdict(
        p50_baseline_ms=_mode_p50("baseline"),
        p50_aligned_ms=_mode_p50(_TOP),
        p50_batched_ms=_mode_p50(_ANCHOR),
    )

    h4 = h4_fullstack_verdict(canaries, ledgers, True, stable_ratios)

    verdicts = {"h1_graph_parity": h1, "h2": h2, "h4": h4}
    if not reproduction["ok"]:
        verdicts = mark_all_blocked(
            verdicts,
            "reproduction anomaly: baseline double-run mismatch "
            f"(first diff prompt {reproduction['mismatch']['first_diff_prompt_index']})",
        )

    # -- artifact
    mode_wall: dict[str, Any] = {}
    for mode in MODES:
        entry: dict[str, Any] = {
            "wall_ms": {
                f"prompt{i}": _stats_dict(stats_from_samples(samples[(mode, i)]))
                for i in range(len(prompts))
            },
            "decode_lengths": {
                f"prompt{i}": _decode_length(
                    rows[(mode, i)],
                    prompts[i]["prompt_len"],
                    num_tokens,
                    eos_token_id,
                )
                for i in range(len(prompts))
            },
            "mode_knobs": dict(MODE_KNOBS[mode]),
        }
        if mode in GRAPH_MODES:
            entry["graph_pool_stats"] = _serializable_pool_stats(mffs[mode].graph_pool_stats)
            entry["engagement"] = engagement[mode]
        mode_wall[mode] = entry

    free_end_bytes, _ = torch.cuda.mem_get_info()
    import transformers

    meta = {
        "checkpoint_dir": load_meta["checkpoint_dir"],
        "checkpoint_sha256": load_meta["checkpoint_sha256"],
        "dtype": dtype,
        "quantization": "none" if dtype == "bf16" else "bitsandbytes int8",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "family_constants": {
            **FAMILY_RECIPES[family],
            "num_tokens": num_tokens,
            "seed": SEED,
            "reps": reps,
            "num_candidates": NUM_CANDIDATES,
            "candidate_top_k": CANDIDATE_TOP_K,
            "speculative_steps": SPECULATIVE_STEPS,
            "gate_tau": TAU,
            "cache_budget_per_layer": CACHE_BUDGET,
            "block_grid_measurement": BLOCK_GRID,
            "h1_parity_atol": _H1_PARITY_ATOL,
        },
        "mode_knobs": {mode: dict(knobs) for mode, knobs in MODE_KNOBS.items()},
        "graph_enabled": True,
        "graph_note": (
            "the graph path is exercised by construction (graph-fixed / "
            "graph-adaptive run production capture); the probe conclusion's "
            "graph_enabled is superseded by the run itself"
        ),
        "geometry_inheritance_note": GEOMETRY_INHERITANCE_NOTE,
        "vram_headroom": {
            "free_mb_after_load": round(free_bytes / 1e6, 1),
            "free_mb_at_end": round(free_end_bytes / 1e6, 1),
            "total_mb": round(total_bytes / 1e6, 1),
        },
        "h2_no_direction_pre_registration": (
            "graph-vs-anchor speed carries NO direction presumption (WDDM + "
            "small-geometry capture-amortization risk); the strict H2 face "
            "is baseline vs graph-adaptive only"
        ),
        "probe_conclusion_snapshot": dict(conclusion),
        "cupti_note": (
            "#39 wall-clock discipline: host perf_counter + cuda synchronize; "
            "no kernel-launch claims (CUPTI not probed)"
        ),
        "seed": SEED,
        "reps": reps,
        "quick": quick,
        "prompts": [prompt["text"] for prompt in prompts],
        "probe_artifact_ref": probe_ref,
        "cross_rep_deterministic": cross_rep_deterministic,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    artifact = build_artifact(
        family,
        "ok",
        mode_wall,
        {"per_prompt": quality_details},
        verdicts,
        reproduction,
        meta,
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    family = args.family
    quick = args.quick
    reps = QUICK_REPS if quick else args.reps
    out_path = resolve_output_path(family, quick, args.out)

    probe_path = Path(args.probe_artifact)
    probe_artifact: dict[str, Any] = {}
    if probe_path.exists():
        probe_artifact = json.loads(probe_path.read_text(encoding="utf-8"))
    conclusion = probe_conclusion(probe_artifact, family)

    if conclusion["degraded"]:
        artifact = build_blocked_artifact(family, conclusion, str(probe_path))
        written = write_results(artifact, out_path)
        print(f"[ar026-bench] family '{family}' degraded per probe: benchmark blocked", flush=True)
        print(f"results written to {written}")
        return 0

    models_root = (
        Path(args.models_root)
        if args.models_root
        else Path(probe_artifact.get("meta", {}).get("models_root", "ar023_models"))
    )
    try:
        artifact = run_graph_bench(family, models_root, reps, quick, conclusion, str(probe_path))
    except Exception as exc:  # noqa: BLE001 - blocked artifact, never fabricated numbers
        artifact = build_artifact(
            family,
            "blocked",
            None,
            None,
            {},
            {"ok": False, "reproduction_anomaly": False, "skipped": True},
            {
                "blocked_reason": f"{type(exc).__name__}: {exc}",
                "probe_artifact_ref": str(probe_path),
                "dtype_choice": conclusion.get("dtype_choice"),
                "graph_enabled": True,
            },
        )
        artifact["blocked_reason"] = artifact["meta"]["blocked_reason"]
        written = write_results(artifact, out_path)
        print(f"[ar026-bench] family '{family}' benchmark blocked: {exc}", flush=True)
        print(f"results written to {written}")
        return 1

    written = write_results(artifact, out_path)
    print()
    for key in ("h1_graph_parity", "h2", "h4"):
        entry = artifact["verdicts"][key]
        print(f"{key.upper()}: {entry['verdict']}")
    print(f"results written to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
