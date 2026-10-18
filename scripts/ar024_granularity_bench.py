"""AR024/T004: LLaDA commit-granularity four-mode benchmark (real ckpt).

Runs the REAL ``LLaDASampler`` generation loop on the local LLaDA-8B
checkpoint in FOUR modes over one shared model instance, prompt set, and
seed (the AR023 protocol carried over via module-level import reuse --
``scripts.ar023_cross_family_bench`` is imported as ``_ar023`` and its
scaffolding/verdict/wall-clock helpers are reused BY IDENTITY; srs FR5
"no copy-paste re-implementation"):

* ``baseline`` -- the family reference sampler (official recipe: LLaDA
  block 32 / steps 32 / low_confidence), ``folded_model=None``.
* ``folded-serial`` -- speculation off (N=0) + one MFF serial folding
  chain (the AR023 reference face, unchanged).
* ``folded-batched`` -- the AR023 semantic A/B anchor: N=4 / top_k=10 /
  greedy off / ``speculative_sub_block_size=BLOCK_GRID(32)`` /
  ``candidate_eos_aware=False`` (unit-granularity commit, EOS-blind
  continuation -- byte-for-byte the AR023 ``folded-batched`` behavior).
* ``folded-batched-greedy`` -- the AR024 recommended granularity: N=4 /
  top_k=10 / greedy on / ``speculative_sub_block_size=8`` /
  ``candidate_eos_aware=True`` (the aligned mode).

**Constructive A/B isolation**: batched vs greedy differ ONLY in the
granularity stack (``speculative_sub_block_size`` +
``candidate_eos_aware`` + ``candidate_include_greedy``) -- same N /
top-k / shape / seed / prompts / independently-constructed MFF (the
AR022 MFF_C pattern), so H1's hard faces (posthoc gap shrink, decode
length recovery, block-grid exact-match strict improvement) attribute
directly to the commit-granularity mechanism.

Hypotheses (declared before measurement; the frozen AR023 wording, verdict
predicates reused unchanged from ``_ar023``):

* **H1** hierarchical quality gate. Hard gate layer (all three faces,
  else refuted): (a) the aligned-vs-baseline posthoc gap shrinks to
  <= 0.5 of the folded-batched gap; (b) decode-length recovery on every
  prompt; (c) merged block-grid exact-match strictly improves over
  folded-batched. Strong conclusion layer (non-blocking): ``gap == 0``
  -> full alignment; exact all-hit; length equality.
* **H2** speed: ``p50(baseline) / p50(aligned) > 1.0`` (strict;
  aligned-vs-batched report-only).
* **H4** full-stack exercise: publish-ledger within the AR021 c=4 bound
  per folded mode; degradation canary (graph path off per the probe
  conclusion -> eager by construction, recorded as the neutral note);
  stable-ratio recording (report face).

Reproduction self-check: the baseline mode is sampled twice per prompt
(untimed) and the rows must match strictly; on mismatch ALL verdicts are
set to ``blocked`` (``_ar023.mark_all_blocked`` reuse -- never silently
substituted, never fabricated). Degraded-family semantics: a degraded
probe conclusion blocks the artifact with the probe cross-reference and
zero fabricated numbers.

**``--refill-face`` report face** (design 4.1.7 frozen boundary): a
quick-smoke-scale run of the same four modes with
``progressive_refill=True`` added to the aligned stack (k=None -> s//2),
writing the separate ``_refill`` artifact -- NEVER the formal verdict
path (the two-phase refill is opt-in mechanism face; its acceptance is
the UT/mutation surface, not the formal recommendation stack).

Protocol: seed 2026 (inherited), P=2 probe prompts (single prompt under
``--quick`` / ``--refill-face``), num_tokens=128 (quick/refill: 64),
warmup one discarded sample per (mode, prompt), per-rep four-mode
round-robin (rep r starts at r%4), host perf_counter + cuda synchronize
(#39 -- wall clock only, no kernel-launch claims), reps=30 pre-committed
single-shot verdict. ``--quick`` / ``--refill-face``: reps=3, separate
artifact paths, never the formal one. The CLI family domain is llada-only
(the AR024 granularity knobs are LLaDA-scoped; Dream/Fast-dLLM carry no
new config fields).

Run (AR019 venv, transformers==4.53.1)::

    python -m scripts.ar024_granularity_bench --family llada
        [--quick] [--refill-face] [--reps 30] [--models-root PATH]
        [--out PATH] [--probe-artifact PATH]
        [--hf-endpoint URL] [--hf-home PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import scripts.ar023_cross_family_bench as _ar023
from scripts._hf_env import add_hf_env_arguments
from scripts.ar023_cross_family_bench import (
    BLOCK_GRID,
    CACHE_BUDGET,
    CANDIDATE_TOP_K,
    FULL_NUM_TOKENS,
    LEDGER_C,
    NUM_CANDIDATES,
    QUICK_NUM_TOKENS,
    QUICK_REPS,
    REPS_DEFAULT,
    SEED,
    TAU,
    _block_exact_counts,
    _decode_length,
    _load_family_model,
    _position_match_counts,
    _posthoc_face,
    _prepare_prompts,
    _stats_dict,
    build_artifact,
    build_blocked_artifact,
    h2_speed_verdict,
    h4_fullstack_verdict,
    probe_conclusion,
    reproduction_verdict,
    write_results,
)

# ---------------------------------------------------------------------------
# Frozen protocol constants (design 4.1.7)
# ---------------------------------------------------------------------------
MODES = _ar023.MODES
FOLDED_MODES = _ar023.FOLDED_MODES
_BATCHED = _ar023._BATCHED
_ALIGNED = _ar023._ALIGNED
FAMILY = "llada"
LLADA_RECIPE = _ar023.FAMILY_RECIPES[FAMILY]

DEFAULT_OUTPUT_PATH = Path("results/optimization/ar024_llada_granularity_bench.json")
DEFAULT_PROBE_ARTIFACT = _ar023.DEFAULT_PROBE_ARTIFACT

# Reused BY IDENTITY (UT-2411): never re-implemented locally.
mark_all_blocked = _ar023.mark_all_blocked
h1_quality_verdict = _ar023.h1_quality_verdict
_make_bench_mff = _ar023._make_bench_mff

#: The frozen granularity constants recorded in the artifact meta
#: (``anchor_semantics`` names the AR023 A/B anchor: unit-granularity
#: commit at s=block_size with EOS-aware continuation off).
GRANULARITY_CONSTANTS: dict[str, Any] = {
    "sub_block_size": 8,
    "candidate_eos_aware": True,
    "progressive_refill": False,
    "refill_top_k": None,
    "anchor_semantics": "ar023 (s=block_size, eos off)",
}

#: The frozen four-mode granularity stacks (the speculative modes only;
#: baseline/folded-serial inherit the official recipe untouched). The
#: batched stack IS the AR023 anchor semantics; the greedy stack is the
#: AR024 recommended granularity. Constructive A/B isolation: the two
#: stacks differ ONLY in the granularity knobs.
GRANULARITY_STACKS: dict[str, dict[str, Any]] = {
    "folded-batched": {
        "num_block_candidates": NUM_CANDIDATES,
        "candidate_top_k": CANDIDATE_TOP_K,
        "candidate_include_greedy": False,
        "speculative_sub_block_size": BLOCK_GRID,
        "candidate_eos_aware": False,
        "progressive_refill": False,
    },
    "folded-batched-greedy": {
        "num_block_candidates": NUM_CANDIDATES,
        "candidate_top_k": CANDIDATE_TOP_K,
        "candidate_include_greedy": True,
        "speculative_sub_block_size": 8,
        "candidate_eos_aware": True,
        "progressive_refill": False,
    },
}

SCHEMA_NAME = "ar024_granularity_bench"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
def quick_output_path() -> Path:
    """The formal artifact path with a ``_quick`` suffix."""
    return DEFAULT_OUTPUT_PATH.with_name(
        DEFAULT_OUTPUT_PATH.stem + "_quick" + DEFAULT_OUTPUT_PATH.suffix
    )


def refill_output_path() -> Path:
    """The ``--refill-face`` report artifact path (``_refill`` suffix).

    Always distinct from the formal verdict artifact -- the refill face
    never writes the formal path.
    """
    return DEFAULT_OUTPUT_PATH.with_name(
        DEFAULT_OUTPUT_PATH.stem + "_refill" + DEFAULT_OUTPUT_PATH.suffix
    )


def resolve_output_path(
    quick: bool, refill_face: bool, out_override: str | Path | None = None
) -> Path:
    """Explicit override wins; else refill path, else quick path, else formal."""
    if out_override is not None:
        return Path(out_override)
    if refill_face:
        return refill_output_path()
    return quick_output_path() if quick else DEFAULT_OUTPUT_PATH


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """``--family`` is required and llada-only (the AR024 scope)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=(FAMILY,))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke subset: reps=3, single prompt, num_tokens=64",
    )
    parser.add_argument(
        "--refill-face",
        action="store_true",
        help=(
            "Report face: quick-scale four-mode run with progressive_refill=True "
            "on the aligned stack (k=None -> s//2); writes the separate _refill "
            "artifact, never the formal verdict path"
        ),
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
        help=(
            "Output JSON path (default: the formal artifact, the _quick path, "
            "or the _refill path)"
        ),
    )
    parser.add_argument(
        "--probe-artifact",
        default=DEFAULT_PROBE_ARTIFACT,
        help="The probe artifact carrying the family conclusions",
    )
    add_hf_env_arguments(parser)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Lazy run layer (heavy imports stay inside; the CI surface is pure logic)
# ---------------------------------------------------------------------------
def _make_granularity_sampler(
    mode: str, llm: Any, num_tokens: int, mff: Any, refill_face: bool
) -> tuple[Any, Any]:
    """The LLaDA sampler for one mode (granularity stacks per design 4.1.7).

    baseline / folded-serial: the official recipe (N=0), the AR023 faces
    unchanged. The two speculative modes carry the frozen granularity
    stacks from ``GRANULARITY_STACKS``; under ``--refill-face`` the
    aligned stack additionally enables ``progressive_refill=True``
    (k=None -> the s//2 frozen default).
    """
    import dataclasses

    from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig

    config: Any = LLaDASamplerConfig(
        num_steps=int(LLADA_RECIPE["num_steps"]),
        num_tokens=num_tokens,
        block_size=int(LLADA_RECIPE["block_size"]),
        remasking=str(LLADA_RECIPE["remasking"]),
        seed=SEED,
    )
    if mode in GRANULARITY_STACKS:
        config = dataclasses.replace(config, **GRANULARITY_STACKS[mode])
    if mode == _ALIGNED and refill_face:
        config = dataclasses.replace(config, progressive_refill=True)
    folded = mff if mode != "baseline" else None
    return LLaDASampler(llm, config=config), folded


def run_granularity_bench(
    models_root: Path,
    reps: int,
    quick: bool,
    refill_face: bool,
    conclusion: dict[str, Any],
    probe_ref: str,
) -> dict[str, Any]:
    """Run the four-mode granularity benchmark for LLaDA (CUDA required)."""
    import torch

    from scripts.algo_experiments import stats_from_samples

    if not torch.cuda.is_available():
        raise RuntimeError("run_granularity_bench requires a CUDA device")
    from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

    num_tokens = QUICK_NUM_TOKENS if quick else FULL_NUM_TOKENS
    llm, dtype, load_meta = _load_family_model(FAMILY, models_root)
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
            # Independent instances per folded mode (the AR022 MFF_C pattern:
            # own cache/gate/counters -- per-mode isolation, H4 ledger face).
            mff = _make_bench_mff(llm.model)
            mffs[mode] = mff
        sampler, folded = _make_granularity_sampler(mode, llm, num_tokens, mff, refill_face)
        sampler_pairs[mode] = (sampler, folded)

    def _sample(mode: str, prompt: dict[str, Any]) -> Any:
        sampler, folded = sampler_pairs[mode]
        return sampler.sample(prompt["ids"], folded_model=folded)

    face = "refill" if refill_face else ("quick" if quick else "formal")
    print(
        f"[ar024-bench] family={FAMILY} face={face} dtype={dtype} "
        f"num_tokens={num_tokens} reps={reps}"
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
    print(f"[ar024-bench] reproduction double-run ok={reproduction['ok']}")

    # -- warmup: one full discarded sample per (mode, prompt)
    for prompt in prompts:
        for mode in MODES:
            _sample(mode, prompt)
    print("[ar024-bench] warmup complete")

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
        print(f"[ar024-bench] rep {rep + 1}/{reps} done", flush=True)

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
        # branch ids repeat across calls -- "branches new since the last
        # snapshot" cannot identify this window. Reset the global profiler
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
            f"[ar024-bench] sidecar {mode}: ops={ops_delta} bound={bound} "
            f"serial={serial_events} batched={batched_events}"
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
            posthoc[mode] = _posthoc_face(llm, row, region_start, lengths[mode], FAMILY)
        base_row = rows[("baseline", prompt_index)]
        block_exact: dict[str, Any] = {}
        position_match: dict[str, Any] = {}
        for mode in FOLDED_MODES:
            other_row = rows[(mode, prompt_index)]
            common = min(lengths["baseline"], lengths[mode])
            block_exact[mode] = _block_exact_counts(base_row, other_row, region_start, common)
            position_match[mode] = _position_match_counts(base_row, other_row, region_start, common)
        quality_details.append(
            {
                "prompt_index": prompt_index,
                "prompt_text": prompt["text"],
                "decode_length": lengths,
                "posthoc_confidence": posthoc,
                "block_exact": block_exact,
                "position_match": position_match,
            }
        )

    # -- verdicts (the frozen _ar023 predicates, reused unchanged)
    h1_input = {
        "per_prompt": [
            {
                "prompt_index": entry["prompt_index"],
                "posthoc_overall_mean": {
                    mode: entry["posthoc_confidence"][mode]["overall_mean"] for mode in MODES
                },
                "decode_length": entry["decode_length"],
                "block_exact": entry["block_exact"],
            }
            for entry in quality_details
        ]
    }
    h1 = h1_quality_verdict(h1_input)

    def _mode_p50(mode: str) -> float:
        per_prompt_p50 = [stats_from_samples(samples[(mode, i)]).p50 for i in range(len(prompts))]
        return sum(per_prompt_p50) / len(per_prompt_p50)

    h2 = h2_speed_verdict(
        p50_baseline_ms=_mode_p50("baseline"),
        p50_aligned_ms=_mode_p50(_ALIGNED),
        p50_batched_ms=_mode_p50(_BATCHED),
    )

    canaries: dict[str, bool] = {mode: True for mode in FOLDED_MODES}
    h4 = h4_fullstack_verdict(
        canaries, ledgers, bool(conclusion.get("graph_enabled", False)), stable_ratios
    )

    verdicts = {"h1": h1, "h2": h2, "h4": h4}
    if not reproduction["ok"]:
        verdicts = mark_all_blocked(
            verdicts,
            "reproduction anomaly: baseline double-run mismatch "
            f"(first diff prompt {reproduction['mismatch']['first_diff_prompt_index']})",
        )

    # -- artifact
    mode_wall: dict[str, Any] = {}
    for mode in MODES:
        mode_wall[mode] = {
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
        }

    import transformers

    meta = {
        "checkpoint_dir": load_meta["checkpoint_dir"],
        "checkpoint_sha256": load_meta["checkpoint_sha256"],
        "dtype": dtype,
        "quantization": "none" if dtype == "bf16" else "bitsandbytes int8",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "family_constants": {
            **LLADA_RECIPE,
            "num_tokens": num_tokens,
            "seed": SEED,
            "reps": reps,
            "num_candidates": NUM_CANDIDATES,
            "candidate_top_k": CANDIDATE_TOP_K,
            "gate_tau": TAU,
            "cache_budget_per_layer": CACHE_BUDGET,
            "block_grid_measurement": BLOCK_GRID,
            "approximate_folding": (
                "similarity-gated folding (cosine tau=0.95): near-exact reuse by gate"
            ),
        },
        "granularity_constants": dict(GRANULARITY_CONSTANTS),
        "granularity_stacks": {mode: dict(stack) for mode, stack in GRANULARITY_STACKS.items()},
        "refill_face": bool(refill_face),
        "refill_note": (
            "report face: aligned stack run with progressive_refill=True "
            "(k=None -> s//2); not the formal verdict artifact"
            if refill_face
            else None
        ),
        "graph_enabled": bool(conclusion.get("graph_enabled", False)),
        "graph_note": (
            "graph path off per the probe conclusion (eager folding by construction)"
            if not conclusion.get("graph_enabled", False)
            else None
        ),
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
        FAMILY,
        "ok",
        mode_wall,
        {"per_prompt": quality_details},
        verdicts,
        reproduction,
        meta,
    )
    # Traceability: the assembled artifact carries the reused ar023 schema
    # name; the AR024 bench overrides it with its own schema identifier
    # (assembly logic reused by identity, schema name owned here).
    artifact["schema"] = SCHEMA_NAME
    artifact["schema_version"] = SCHEMA_VERSION
    return artifact


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from scripts._hf_env import apply_hf_env

    apply_hf_env()

    # The refill face is a quick-smoke-scale report (design 4.1.7).
    quick = args.quick or args.refill_face
    reps = QUICK_REPS if quick else args.reps
    out_path = resolve_output_path(args.quick, args.refill_face, args.out)

    probe_path = Path(args.probe_artifact)
    probe_artifact: dict[str, Any] = {}
    if probe_path.exists():
        probe_artifact = json.loads(probe_path.read_text(encoding="utf-8"))
    conclusion = probe_conclusion(probe_artifact, FAMILY)

    if conclusion["degraded"]:
        artifact = build_blocked_artifact(FAMILY, conclusion, str(probe_path))
        written = write_results(artifact, out_path)
        print(f"[ar024-bench] family '{FAMILY}' degraded per probe: benchmark blocked")
        print(f"results written to {written}")
        return 0

    models_root = (
        Path(args.models_root)
        if args.models_root
        else Path(probe_artifact.get("meta", {}).get("models_root", "ar023_models"))
    )
    try:
        artifact = run_granularity_bench(
            models_root, reps, quick, args.refill_face, conclusion, str(probe_path)
        )
    except Exception as exc:  # noqa: BLE001 - blocked artifact, never fabricated numbers
        artifact = build_artifact(
            FAMILY,
            "blocked",
            None,
            None,
            {},
            {"ok": False, "reproduction_anomaly": False, "skipped": True},
            {
                "blocked_reason": f"{type(exc).__name__}: {exc}",
                "probe_artifact_ref": str(probe_path),
                "dtype_choice": conclusion.get("dtype_choice"),
                "graph_enabled": conclusion.get("graph_enabled", False),
            },
        )
        artifact["schema"] = SCHEMA_NAME
        artifact["blocked_reason"] = artifact["meta"]["blocked_reason"]
        written = write_results(artifact, out_path)
        print(f"[ar024-bench] family '{FAMILY}' benchmark blocked: {exc}")
        print(f"results written to {written}")
        return 1

    written = write_results(artifact, out_path)
    print()
    for key in ("h1", "h2", "h4"):
        entry = artifact["verdicts"][key]
        print(f"{key.upper()}: {entry['verdict']}")
    print(f"results written to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
