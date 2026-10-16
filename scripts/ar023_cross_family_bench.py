"""AR023/T005: cross-family four-mode speculative benchmark (real ckpts).

Runs the REAL ``LLaDASampler`` / ``DreamSampler`` generation loop on the local
LLaDA-8B / Dream-v0-7B checkpoints in FOUR modes over one shared model
instance, prompt set, and seed (the AR021/AR022 protocol carried over;
design 4.1.8):

* ``baseline`` -- the family reference sampler (official recipe: LLaDA
  block 32 / steps 32 / low_confidence; Dream maskgit_plus / eps 1e-3 /
  1:1 steps), ``folded_model=None``.
* ``folded-serial`` -- speculation off (N=0) + one MFF serial folding
  chain (the family ctx threads end to end).
* ``folded-batched`` -- N=4 / top_k=10 / greedy off: the AR023 batched
  speculative verification.
* ``folded-batched-greedy`` -- the AR022 alignment mode: an INDEPENDENT
  MFF instance constructed identically to folded-batched plus
  ``candidate_include_greedy=True`` (same [N, T] shape / N / top-k /
  prompts / seed -- the speed variable is constructively zeroed).
  Dream modes share ``speculative_steps=8``.

Hypotheses (declared before measurement; frozen wording from srs 3.5 /
design 4.1.8; verdicts from data only):

* **H1** hierarchical quality gate (OQ1 rev 3). Hard gate layer (all three
  faces, else refuted): (a) the aligned-vs-baseline posthoc gap shrinks to
  <= 0.5 of the folded-batched gap; (b) decode-length recovery (every
  prompt: ``|len_aln - len_base| <= |len_bat - len_base|``); (c) merged
  block-grid exact-match strictly improves over folded-batched. Strong
  conclusion layer (non-blocking, recorded): ``gap == 0`` -> full
  alignment; exact all-hit; length equality.
* **H2** speed: ``p50(baseline) / p50(aligned) > 1.0`` (strict; no
  magnitude presumption -- AR022's 2.9x was Fast-dLLM bf16 specific).
* **H4** full-stack exercise: publish-ledger within the AR021 c=4 bound
  per folded mode; the degradation canary (graph path off per the probe
  conclusion -> eager by construction, recorded as the neutral note);
  stable-ratio recording of the approximate folding (report face).

Reproduction self-check: the baseline mode is sampled twice per prompt
(untimed) and the rows must match strictly (temperature-0 greedy paths:
LLaDA low_confidence argmax / Dream maskgit_plus greedy). On mismatch the
artifact records ``reproduction_anomaly`` and ALL verdicts are set to
``blocked`` (never silently substituted, never fabricated).

Degraded-family semantics: a family whose probe conclusion is degraded
skips the benchmark entirely -- the artifact records ``status: blocked``
with the probe cross-reference and NO fabricated numbers (srs FR5). A
single-mode failure (e.g. OOM) blocks that family's artifact the same way.

Protocol: seed 2026, P=2 probe prompts (single prompt under ``--quick``),
num_tokens=128 (quick: 64), warmup one discarded sample per (mode, prompt),
per-rep four-mode round-robin (rep r starts at r%4), host perf_counter +
cuda synchronize (#39 -- wall clock only, no kernel-launch claims), reps=30
pre-committed single-shot verdict. ``--quick``: reps=3, separate artifact
path, never the formal one.

Run (AR019 venv, transformers==4.53.1)::

    python -m scripts.ar023_cross_family_bench --family llada|dream
        [--quick] [--reps 30] [--models-root PATH] [--out PATH]
        [--probe-artifact PATH] [--hf-endpoint URL] [--hf-home PATH]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments

# ---------------------------------------------------------------------------
# Frozen protocol constants (design 4.1.8)
# ---------------------------------------------------------------------------
SEED = 2026
REPS_DEFAULT = 30
QUICK_REPS = 3
FULL_NUM_TOKENS = 128
QUICK_NUM_TOKENS = 64
BLOCK_GRID = 32
NUM_CANDIDATES = 4
CANDIDATE_TOP_K = 10
SPECULATIVE_STEPS = 8
LEDGER_C = 4
TAU = 0.95
CACHE_BUDGET = 64

MODES = (
    "baseline",
    "folded-serial",
    "folded-batched",
    "folded-batched-greedy",
)
FOLDED_MODES = ("folded-serial", "folded-batched", "folded-batched-greedy")
FAMILIES = ("llada", "dream")

FAMILY_RECIPES: dict[str, dict[str, Any]] = {
    "llada": {"block_size": 32, "num_steps": 32, "remasking": "low_confidence"},
    "dream": {"alg": "maskgit_plus", "eps": 1e-3, "num_steps": "1:1 with num_tokens"},
}

DEFAULT_OUTPUT_PATHS: dict[str, Path] = {
    "llada": Path("results/optimization/ar023_llada_bench.json"),
    "dream": Path("results/optimization/ar023_dream_bench.json"),
}
DEFAULT_PROBE_ARTIFACT = "results/optimization/ar023_family_probe.json"
SCHEMA_VERSION = 1

_ALIGNED = "folded-batched-greedy"
_BATCHED = "folded-batched"
# The h1 face-(a) ``<= 0.5`` comparison carries this epsilon: decimal-gap
# fixtures at the exact boundary (e.g. 0.30-0.40 vs 0.20-0.40) land at
# 0.5000000000000002 under IEEE-754 representation noise, not 0.5.
_H1_RATIO_EPS = 1e-9


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
def quick_output_path(family: str) -> Path:
    """The family's formal artifact path with a ``_quick`` suffix."""
    formal = DEFAULT_OUTPUT_PATHS[family]
    return formal.with_name(formal.stem + "_quick" + formal.suffix)


def resolve_output_path(family: str, quick: bool, out_override: str | Path | None = None) -> Path:
    """Explicit override wins; else quick path under ``--quick``, else formal."""
    if out_override is not None:
        return Path(out_override)
    return quick_output_path(family) if quick else DEFAULT_OUTPUT_PATHS[family]


# ---------------------------------------------------------------------------
# Probe conclusion extraction (safe degrade)
# ---------------------------------------------------------------------------
def probe_conclusion(probe_artifact: dict[str, Any], family: str) -> dict[str, Any]:
    """Extract the family's probe conclusions with a safe degrade.

    A missing family entry or missing conclusions dict degrades to
    ``degraded=True`` with an explicit reason -- the benchmark never runs
    on an unprobed family (srs FR5 trigger condition).
    """
    families = probe_artifact.get("families") or {}
    entry = families.get(family)
    conclusions = (entry or {}).get("conclusions") if isinstance(entry, dict) else None
    if not isinstance(conclusions, dict):
        return {
            "degraded": True,
            "degrade_reason": (
                f"probe artifact has no conclusions for family '{family}' "
                "(family not probed or probe artifact missing)"
            ),
            "dtype_choice": None,
            "graph_enabled": False,
        }
    return {
        "degraded": bool(conclusions.get("degraded", True)),
        "degrade_reason": conclusions.get("degrade_reason"),
        "dtype_choice": conclusions.get("dtype_choice"),
        "graph_enabled": bool(conclusions.get("graph_enabled", False)),
    }


# ---------------------------------------------------------------------------
# Frozen verdict predicates (design 4.1.8 -> code, verbatim)
# ---------------------------------------------------------------------------
def h1_quality_verdict(quality_faces: dict[str, Any]) -> dict[str, Any]:
    """H1 hierarchical quality gate (hard layer + non-blocking strong layer).

    Hard faces (all three, else refuted): (a) the aligned-vs-baseline
    posthoc gap shrinks to <= 0.5 of the folded-batched gap; (b) decode
    length recovery on EVERY prompt; (c) merged block-grid exact-match
    strictly improves over folded-batched. Any ``None`` posthoc mean
    refutes (no fabricated numbers). ``gap_bat == 0`` sentinels: ratio 0.0
    when the aligned gap is also 0, else ``inf``. The ``<= 0.5`` boundary
    comparison carries a 1e-9 epsilon against IEEE-754 representation
    noise (``_H1_RATIO_EPS``).
    """
    per_prompt = quality_faces.get("per_prompt") or []
    scoreable = (_BATCHED, _ALIGNED)

    gaps: dict[str, list[float]] = {mode: [] for mode in scoreable}
    none_mean = False
    for entry in per_prompt:
        means = entry.get("posthoc_overall_mean", {})
        base_mean = means.get("baseline")
        if base_mean is None:
            none_mean = True
            continue
        for mode in scoreable:
            mode_mean = means.get(mode)
            if mode_mean is None:
                none_mean = True
                gaps[mode].append(math.nan)
            else:
                gaps[mode].append(mode_mean - base_mean)

    merged_gap = {mode: (sum(v) / len(v) if v else math.nan) for mode, v in gaps.items()}
    gap_bat = merged_gap[_BATCHED]
    gap_aln = merged_gap[_ALIGNED]
    if none_mean or math.isnan(gap_bat) or math.isnan(gap_aln):
        ratio: float = math.inf
    elif gap_bat == 0.0:
        ratio = 0.0 if gap_aln == 0.0 else math.inf
    else:
        ratio = abs(gap_aln) / abs(gap_bat)
    face_a = ratio <= 0.5 + _H1_RATIO_EPS

    face_b = bool(per_prompt)
    lengths: list[dict[str, Any]] = []
    for entry in per_prompt:
        lens = entry.get("decode_length", {})
        len_base = lens.get("baseline")
        len_bat = lens.get(_BATCHED)
        len_aln = lens.get(_ALIGNED)
        ok = (
            len_base is not None
            and len_bat is not None
            and len_aln is not None
            and abs(len_aln - len_base) <= abs(len_bat - len_base)
        )
        face_b = face_b and ok
        lengths.append(
            {
                "prompt_index": entry.get("prompt_index"),
                "len_baseline": len_base,
                "len_folded_batched": len_bat,
                "len_aligned": len_aln,
                "recovered": bool(ok),
            }
        )

    def _merged_rate(mode: str) -> float | None:
        matched = 0
        compared = 0
        for entry in per_prompt:
            counts = (entry.get("block_exact") or {}).get(mode) or {}
            matched += int(counts.get("matched", 0) or 0)
            compared += int(counts.get("compared", 0) or 0)
        return (matched / compared) if compared else None

    rate_bat = _merged_rate(_BATCHED)
    rate_aln = _merged_rate(_ALIGNED)
    face_c = rate_aln is not None and rate_bat is not None and rate_aln > rate_bat

    supported = bool(face_a and face_b and face_c)
    strong_full = (
        "supported"
        if (gaps[_ALIGNED] and all(g == 0.0 for g in gaps[_ALIGNED]) and not none_mean)
        else None
    )
    strong_exact = "supported" if rate_aln == 1.0 else None
    strong_length = (
        "supported"
        if (lengths and all(item["len_aligned"] == item["len_baseline"] for item in lengths))
        else None
    )
    return {
        "verdict": "supported" if supported else "refuted",
        "faces": {
            "a_gap_shrink_ratio_le_0_5": bool(face_a),
            "b_length_recovery": bool(face_b),
            "c_block_exact_strict_improvement": bool(face_c),
        },
        "strong": {
            "quality_full_alignment": strong_full,
            "exact_all_hit": strong_exact,
            "length_equals_baseline": strong_length,
        },
        "numbers": {
            "gap": {
                mode: (None if math.isnan(merged_gap[mode]) else round(merged_gap[mode], 6))
                for mode in scoreable
            },
            "ratio": round(ratio, 6),
            "exact": {
                _BATCHED: None if rate_bat is None else round(rate_bat, 6),
                _ALIGNED: None if rate_aln is None else round(rate_aln, 6),
            },
            "per_prompt_decode_lengths": lengths,
            "none_posthoc_mean": bool(none_mean),
        },
    }


def h2_speed_verdict(
    p50_baseline_ms: float, p50_aligned_ms: float, p50_batched_ms: float
) -> dict[str, Any]:
    """H2 speed: ``p50(baseline) / p50(aligned) > 1.0`` strict.

    The aligned-vs-batched same-shape ratio is a REPORT face only (the
    [N, T] batched shape/N/top-k are identical by construction).
    """
    speedup = p50_baseline_ms / p50_aligned_ms
    return {
        "verdict": "supported" if speedup > 1.0 else "refuted",
        "numbers": {
            "speedup_p50_vs_baseline": round(speedup, 4),
            "p50_baseline_ms": p50_baseline_ms,
            "p50_aligned_ms": p50_aligned_ms,
            "aligned_vs_batched_ratio": round(p50_aligned_ms / p50_batched_ms, 4),
            "rule": "supported iff p50(baseline)/p50(aligned) > 1.0",
        },
    }


def h4_fullstack_verdict(
    canaries: dict[str, bool],
    ledgers: dict[str, bool],
    graph_enabled: bool,
    stable_ratios: dict[str, float],
) -> dict[str, Any]:
    """H4 full-stack exercise: canary/ledger conjunction, eager-neutral graph.

    ``graph_enabled=False`` records the eager note and never refutes (the
    probe conclusion disables the graph path by construction); stable
    ratios are the approximate-folding record face (report only).
    """
    canary_ok = all(bool(v) for v in canaries.values())
    ledger_ok = all(bool(v) for v in ledgers.values())
    supported = bool(canary_ok and ledger_ok)
    ratios = dict(stable_ratios)
    mean_ratio = round(sum(ratios.values()) / len(ratios), 6) if ratios else None
    return {
        "verdict": "supported" if supported else "refuted",
        "numbers": {
            "canary_degradation_zero": {k: bool(v) for k, v in canaries.items()},
            "ledger_within_bound": {k: bool(v) for k, v in ledgers.items()},
            "graph_enabled": bool(graph_enabled),
            "graph_note": (
                None
                if graph_enabled
                else "graph path off per the probe conclusion: eager folding by "
                "construction, no degradation fallback path exists (neutral)"
            ),
            "stable_ratios": ratios,
            "stable_ratio_mean": mean_ratio,
        },
    }


def reproduction_verdict(
    baseline_rows_a: list[list[int]], baseline_rows_b: list[list[int]]
) -> dict[str, Any]:
    """Strict equality of the two baseline double-run row lists.

    Each list holds one row (list of token ids) per prompt, in prompt
    order. Any content or length divergence fails with the first
    diverging prompt index.
    """
    if len(baseline_rows_a) != len(baseline_rows_b):
        common = min(len(baseline_rows_a), len(baseline_rows_b))
        for i in range(common):
            if list(baseline_rows_a[i]) != list(baseline_rows_b[i]):
                break
        else:
            i = common
        return {
            "ok": False,
            "reproduction_anomaly": True,
            "mismatch": {
                "first_diff_prompt_index": i,
                "a_len": len(baseline_rows_a[i]) if i < len(baseline_rows_a) else None,
                "b_len": len(baseline_rows_b[i]) if i < len(baseline_rows_b) else None,
            },
        }
    for i, (row_a, row_b) in enumerate(zip(baseline_rows_a, baseline_rows_b)):
        if list(row_a) != list(row_b):
            return {
                "ok": False,
                "reproduction_anomaly": True,
                "mismatch": {
                    "first_diff_prompt_index": i,
                    "a_len": len(row_a),
                    "b_len": len(row_b),
                },
            }
    return {"ok": True}


def mark_all_blocked(verdicts: dict[str, dict[str, Any]], reason: str) -> dict[str, dict[str, Any]]:
    """Every verdict -> ``blocked`` with the reason (input never mutated)."""
    out: dict[str, dict[str, Any]] = {}
    for key, entry in verdicts.items():
        blocked = copy.deepcopy(entry)
        blocked["verdict"] = "blocked"
        blocked["blocked_reason"] = reason
        out[key] = blocked
    return out


# ---------------------------------------------------------------------------
# Artifact schema
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
        "schema": "ar023_cross_family_bench",
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "status": status,
        "modes": modes,
        "quality": quality,
        "verdicts": verdicts,
        "reproduction": reproduction,
        "meta": meta,
    }


def build_blocked_artifact(
    family: str, conclusion: dict[str, Any], probe_ref: str
) -> dict[str, Any]:
    """The degraded-family face: status blocked, zero fabricated numbers."""
    reason = conclusion.get("degrade_reason") or (
        f"family '{family}' degraded per the probe conclusion"
    )
    artifact = build_artifact(
        family,
        "blocked",
        None,
        None,
        {},
        {
            "ok": False,
            "reproduction_anomaly": False,
            "skipped": True,
            "note": "degraded family per probe conclusion",
        },
        {
            "blocked_reason": reason,
            "probe_artifact_ref": probe_ref,
            "dtype_choice": conclusion.get("dtype_choice"),
            "graph_enabled": conclusion.get("graph_enabled", False),
        },
    )
    artifact["blocked_reason"] = reason
    return artifact


def write_results(artifact: dict[str, Any], output_path: str | Path) -> Path:
    """JSON dump (indent=2, UTF-8); parent directories are created."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """``--family`` is required with exactly the two family choices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=FAMILIES)
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
    add_hf_env_arguments(parser)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Lazy run layer (heavy imports stay inside; the CI surface is pure logic)
# ---------------------------------------------------------------------------
def _load_family_model(family: str, models_root: Path) -> tuple[Any, str, dict[str, Any]]:
    """Family wrapper at the first viable dtype (bf16, then int8).

    Mirrors the probe's ``dtype_choice`` priority; the achieved dtype is
    recorded honestly in the artifact meta.
    """
    import torch

    from scripts.ar023_family_probe import _family_wrapper_cls, sha256_file

    local_dir = models_root / family
    if not local_dir.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory not found: {local_dir} (run the probe first; "
            "no synthetic-model fallback for a real-checkpoint benchmark)"
        )
    wrapper_cls = _family_wrapper_cls(family)
    attempts: list[str] = []
    model: Any = None
    dtype: str
    try:
        model = wrapper_cls(str(local_dir), torch_dtype=torch.bfloat16)
        model.model.cuda()
        dtype = "bf16"
    except Exception as exc:  # noqa: BLE001 - load-face isolation mirrors the probe
        attempts.append(f"bf16: {type(exc).__name__}: {exc}")
        model = None
        torch.cuda.empty_cache()
    if model is None:
        model = wrapper_cls(str(local_dir), load_in_8bit=True, device_map={"": 0})
        dtype = "int8"
    model.model.eval()
    weights = local_dir / "model.safetensors"
    load_meta = {
        "checkpoint_dir": str(local_dir),
        "checkpoint_sha256": sha256_file(weights) if weights.exists() else None,
        "dtype_achieved": dtype,
        "load_attempts": attempts,
    }
    return model, dtype, load_meta


def _make_bench_mff(raw_model: Any) -> Any:
    """One instrumented ManualFoldedForward (serial/batched call counters).

    Eager only: the probe conclusion disabled the graph path for both
    families (graph_enabled=False), so no graph/pool knobs are set. The
    cache budget is small (64/layer): the parent-chain working set is a
    few entries while the 8B checkpoints leave ~1GB VRAM headroom.
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
    return _BenchMFF(raw_model, cache=cache, gate=gate, scheduler=None, split_layers=False)


def _make_sampler(family: str, mode: str, llm: Any, num_tokens: int, mff: Any) -> Any:
    """Family sampler for one mode (official recipe; spec knobs per mode)."""
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
    if mode in (_BATCHED, _ALIGNED):
        config = dataclasses.replace(
            config,
            num_block_candidates=NUM_CANDIDATES,
            candidate_top_k=CANDIDATE_TOP_K,
        )
    if mode == _ALIGNED:
        config = dataclasses.replace(config, candidate_include_greedy=True)
    sampler_cls = LLaDASampler if family == "llada" else DreamSampler
    folded = mff if mode != "baseline" else None
    return sampler_cls(llm, config=config), folded


def _prepare_prompts(llm: Any, quick: bool, device: str) -> list[dict[str, Any]]:
    """The probe prompt set (P=2; single prompt under ``--quick``)."""
    from scripts.ar023_family_probe import PROBE_PROMPTS

    texts = PROBE_PROMPTS[:1] if quick else list(PROBE_PROMPTS)
    prompts: list[dict[str, Any]] = []
    for text in texts:
        ids = llm.tokenizer(text, return_tensors="pt").input_ids.to(device)
        prompts.append({"text": text, "ids": ids, "prompt_len": int(ids.shape[1])})
    return prompts


def _decode_length(row: list[int], region_start: int, num_tokens: int, eos_id: int) -> int:
    """Tokens until the first EOS inside the generation region (inclusive)."""
    region = row[region_start : region_start + num_tokens]
    for offset, token in enumerate(region):
        if token == eos_id:
            return offset + 1
    return num_tokens


def _posthoc_face(
    llm: Any,
    seq_row: list[int],
    region_start: int,
    decode_len: int,
    family: str,
) -> dict[str, Any]:
    """Post-hoc confidence of the committed generated tokens (report face).

    One untimed raw forward over the final committed row; the token
    probability at position j is the softmax gathered at the committed
    token -- right-shifted for Dream (the family shift convention), direct
    for LLaDA (bidirectional, no shift).
    """
    import torch

    from actfold.models.sampling_utils import right_shift_logits

    with torch.no_grad():
        device = next(llm.model.parameters()).device
        ids = torch.as_tensor(seq_row, dtype=torch.long, device=device)[None, :]
        logits = llm.forward(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))
        if family == "dream":
            logits = right_shift_logits(logits)
        probs = torch.softmax(logits.float(), dim=-1)
        length = int(ids.shape[1])
        gather = torch.arange(length, device=probs.device)
        conf = probs[0, gather, ids[0].to(probs.device)]
        window = conf[region_start : region_start + decode_len]
        return {
            "overall_mean": round(float(window.mean().item()), 6) if window.numel() else None,
            "overall_min": round(float(window.min().item()), 6) if window.numel() else None,
            "window": f"[{region_start}, {region_start + decode_len})",
        }


def _block_exact_counts(
    base_row: list[int], other_row: list[int], region_start: int, decode_len: int
) -> dict[str, int]:
    """Merged matched/compared counts on the BLOCK_GRID measurement grid.

    Comparison window = the common committed prefix of the two rows; the
    grid is the measurement face (both families), not a sampler construct.
    """
    matched = 0
    compared = 0
    end = region_start + decode_len
    for chunk in range(math.ceil(decode_len / BLOCK_GRID)):
        s = region_start + chunk * BLOCK_GRID
        e = min(s + BLOCK_GRID, end)
        matched += sum(
            1
            for i in range(s, e)
            if i < len(base_row) and i < len(other_row) and base_row[i] == other_row[i]
        )
        compared += e - s
    return {"matched": matched, "compared": compared}


def _position_match_counts(
    base_row: list[int], other_row: list[int], region_start: int, decode_len: int
) -> dict[str, Any]:
    """Position-level agreement over the common committed prefix (report)."""
    end = region_start + decode_len
    compared = 0
    matched = 0
    for i in range(region_start, end):
        if i >= len(base_row) or i >= len(other_row):
            break
        compared += 1
        matched += 1 if base_row[i] == other_row[i] else 0
    return {
        "matched": matched,
        "compared": compared,
        "rate": round(matched / compared, 6) if compared else None,
    }


def _stats_dict(stats: Any) -> dict[str, Any]:
    return {
        "mean_ms": round(stats.mean, 4),
        "std_ms": round(stats.std, 4),
        "p50_ms": round(stats.p50, 4),
        "n": int(stats.n),
    }


def run_family_bench(
    family: str,
    models_root: Path,
    reps: int,
    quick: bool,
    conclusion: dict[str, Any],
    probe_ref: str,
) -> dict[str, Any]:
    """Run the four-mode benchmark for one family (CUDA required)."""
    import torch

    from scripts.algo_experiments import stats_from_samples

    if not torch.cuda.is_available():
        raise RuntimeError("run_family_bench requires a CUDA device")
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
            # Independent instances per folded mode (the AR022 MFF_C pattern:
            # own cache/gate/counters -- per-mode isolation, H4 ledger face).
            mff = _make_bench_mff(llm.model)
            mffs[mode] = mff
        sampler, folded = _make_sampler(family, mode, llm, num_tokens, mff)
        sampler_pairs[mode] = (sampler, folded)

    def _sample(mode: str, prompt: dict[str, Any]) -> Any:
        sampler, folded = sampler_pairs[mode]
        return sampler.sample(prompt["ids"], folded_model=folded)

    print(f"[ar023-bench] family={family} dtype={dtype} num_tokens={num_tokens} reps={reps}")

    # -- reproduction self-check: baseline double-run (untimed, strict rows)
    repro_rows_a: list[list[int]] = []
    repro_rows_b: list[list[int]] = []
    for prompt in prompts:
        row_a = _sample("baseline", prompt).sequences[0].tolist()
        row_b = _sample("baseline", prompt).sequences[0].tolist()
        repro_rows_a.append(row_a)
        repro_rows_b.append(row_b)
    reproduction = reproduction_verdict(repro_rows_a, repro_rows_b)
    print(f"[ar023-bench] reproduction double-run ok={reproduction['ok']}")

    # -- warmup: one full discarded sample per (mode, prompt)
    for prompt in prompts:
        for mode in MODES:
            _sample(mode, prompt)
    print("[ar023-bench] warmup complete")

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
        print(f"[ar023-bench] rep {rep + 1}/{reps} done", flush=True)

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
            f"[ar023-bench] sidecar {mode}: ops={ops_delta} bound={bound} "
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
            posthoc[mode] = _posthoc_face(llm, row, region_start, lengths[mode], family)
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

    # -- verdicts
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
            "approximate_folding": (
                "similarity-gated folding (cosine tau=0.95): near-exact reuse by gate"
            ),
        },
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
    from scripts._hf_env import apply_hf_env

    apply_hf_env()

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
        print(f"[ar023-bench] family '{family}' degraded per probe: benchmark blocked")
        print(f"results written to {written}")
        return 0

    models_root = (
        Path(args.models_root)
        if args.models_root
        else Path(probe_artifact.get("meta", {}).get("models_root", "ar023_models"))
    )
    try:
        artifact = run_family_bench(family, models_root, reps, quick, conclusion, str(probe_path))
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
                "graph_enabled": conclusion.get("graph_enabled", False),
            },
        )
        artifact["blocked_reason"] = artifact["meta"]["blocked_reason"]
        written = write_results(artifact, out_path)
        print(f"[ar023-bench] family '{family}' benchmark blocked: {exc}")
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
