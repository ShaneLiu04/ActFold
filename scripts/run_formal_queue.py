"""AR027: server formal-run queue executor.

Runs the seven pre-committed formal faces (AR023/AR024/AR025/AR026) in a fixed
order with:

* hard preflight gates (transformers pin / CUDA / VRAM / checkpoint files /
  probe artifact / GPU silence),
* isolated subprocess execution via ``python -m scripts.<bench>``,
* structured logs (``queue.log`` + ``queue.json``) under a per-run UTC dir,
* artifact validation that re-derives the frozen verdicts with the benches'
  own verdict functions (single source of truth) plus structural self-checks,
* honest failure semantics (stop-on-first-failure, no retry, no overwrite
  without ``--force`` + a non-empty audited reason).

Nothing here writes an artifact: validation only reads and compares.

Usage::

    python -m scripts.run_formal_queue --list
    python -m scripts.run_formal_queue --quick
    python -m scripts.run_formal_queue --models-root /data/ar023_models
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA_AR023 = "ar023_cross_family_bench"
SCHEMA_AR024 = "ar024_granularity_bench"
SCHEMA_AR025 = "ar025_real_bench"
SCHEMA_AR026 = "ar026_real_graph_bench"

TRANSFORMERS_PIN = "4.53.1"
FORMAL_VRAM_GB = 24.0
QUICK_VRAM_GB = 16.0
GPU_MAX_UTIL_PCT = 10.0
GPU_MAX_USED_MIB = 1024.0

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_ARTIFACT = REPO_ROOT / "results/optimization/ar023_family_probe.json"
DEFAULT_RUNS_BASE = REPO_ROOT / "results/optimization/formal_runs"
DEFAULT_MODELS_ROOT = REPO_ROOT / "ar023_models"

EXIT_OK = 0
EXIT_VALIDATION = 2
EXIT_PREFLIGHT = 3
EXIT_FAILURE = 4

AR023_MODES = ("baseline", "folded-serial", "folded-batched", "folded-batched-greedy")
AR026_MODES = ("baseline", "folded-batched", "graph-fixed", "graph-adaptive")
AR025_VERDICT_KEYS = (
    "H1_token_parity",
    "H2_speed",
    "delivery_line",
    "H4_ledger_strictly_smaller",
    "canary_structural_eager_fallback",
    "overall",
    "blocked_reasons",
    "reproduction",
)
DEGRADE_NOTE = "degraded family per probe conclusion"
DEGRADE_REASON_PREFIX = "family degraded per probe"
RUN_FAILED_PREFIX = "real-face run failed:"


@dataclass(frozen=True)
class FaceSpec:
    """One formal face: frozen command, artifacts and expected schemas."""

    face_id: str
    module: str
    family: str
    formal_argv: tuple[str, ...]
    formal_artifact: Path
    quick_argv: tuple[str, ...]
    quick_artifact: Path
    schemas: dict[str, tuple[str, int | None]]
    validator: str


def _schemas_full(
    ok: tuple[str, int | None],
    degraded: tuple[str, int | None],
    exception: tuple[str, int | None],
) -> dict[str, tuple[str, int | None]]:
    return {"ok": ok, "blocked_degraded": degraded, "blocked_exception": exception}


FACE_QUEUE: tuple[FaceSpec, ...] = (
    FaceSpec(
        face_id="ar023-llada",
        module="scripts.ar023_cross_family_bench",
        family="llada",
        formal_argv=("--family", "llada", "--reps", "30"),
        formal_artifact=Path("results/optimization/ar023_llada_bench.json"),
        quick_argv=("--family", "llada", "--reps", "30", "--quick"),
        quick_artifact=Path("results/optimization/ar023_llada_bench_quick.json"),
        schemas=_schemas_full((SCHEMA_AR023, 1), (SCHEMA_AR023, 1), (SCHEMA_AR023, 1)),
        validator="ar023",
    ),
    FaceSpec(
        face_id="ar023-dream",
        module="scripts.ar023_cross_family_bench",
        family="dream",
        formal_argv=("--family", "dream", "--reps", "30"),
        formal_artifact=Path("results/optimization/ar023_dream_bench.json"),
        quick_argv=("--family", "dream", "--reps", "30", "--quick"),
        quick_artifact=Path("results/optimization/ar023_dream_bench_quick.json"),
        schemas=_schemas_full((SCHEMA_AR023, 1), (SCHEMA_AR023, 1), (SCHEMA_AR023, 1)),
        validator="ar023",
    ),
    FaceSpec(
        face_id="ar024",
        module="scripts.ar024_granularity_bench",
        family="llada",
        formal_argv=("--family", "llada", "--reps", "30"),
        formal_artifact=Path("results/optimization/ar024_llada_granularity_bench.json"),
        quick_argv=("--family", "llada", "--reps", "30", "--quick"),
        quick_artifact=Path("results/optimization/ar024_llada_granularity_bench_quick.json"),
        schemas=_schemas_full((SCHEMA_AR024, 1), (SCHEMA_AR023, 1), (SCHEMA_AR024, 1)),
        validator="ar024",
    ),
    FaceSpec(
        face_id="ar025-real-llada",
        module="scripts.ar025_multi_parent_bench",
        family="llada",
        formal_argv=("--mode", "real", "--family", "llada", "--reps", "30"),
        formal_artifact=Path("results/optimization/ar025_real_bench_llada.json"),
        quick_argv=("--mode", "real", "--family", "llada", "--reps", "30", "--quick"),
        quick_artifact=Path("results/optimization/ar025_real_bench_llada_quick.json"),
        schemas=_schemas_full((SCHEMA_AR025, None), (SCHEMA_AR025, None), (SCHEMA_AR025, None)),
        validator="ar025_real",
    ),
    FaceSpec(
        face_id="ar025-real-dream",
        module="scripts.ar025_multi_parent_bench",
        family="dream",
        formal_argv=("--mode", "real", "--family", "dream", "--reps", "30"),
        formal_artifact=Path("results/optimization/ar025_real_bench_dream.json"),
        quick_argv=("--mode", "real", "--family", "dream", "--reps", "30", "--quick"),
        quick_artifact=Path("results/optimization/ar025_real_bench_dream_quick.json"),
        schemas=_schemas_full((SCHEMA_AR025, None), (SCHEMA_AR025, None), (SCHEMA_AR025, None)),
        validator="ar025_real",
    ),
    FaceSpec(
        face_id="ar026-llada",
        module="scripts.ar026_real_graph_bench",
        family="llada",
        formal_argv=(
            "--family",
            "llada",
            "--reps",
            "30",
            "--out",
            "results/optimization/ar026_real_graph_bench.json",
        ),
        formal_artifact=Path("results/optimization/ar026_real_graph_bench.json"),
        quick_argv=("--family", "llada", "--reps", "30", "--quick"),
        quick_artifact=Path("results/optimization/ar026_real_graph_bench_quick.json"),
        schemas=_schemas_full((SCHEMA_AR026, 1), (SCHEMA_AR023, 1), (SCHEMA_AR026, 1)),
        validator="ar026",
    ),
    FaceSpec(
        face_id="ar026-dream",
        module="scripts.ar026_real_graph_bench",
        family="dream",
        formal_argv=(
            "--family",
            "dream",
            "--reps",
            "30",
            "--out",
            "results/optimization/ar026_real_graph_bench_dream.json",
        ),
        formal_artifact=Path("results/optimization/ar026_real_graph_bench_dream.json"),
        quick_argv=(
            "--family",
            "dream",
            "--reps",
            "30",
            "--quick",
            "--out",
            "results/optimization/ar026_real_graph_bench_dream_quick.json",
        ),
        quick_artifact=Path("results/optimization/ar026_real_graph_bench_dream_quick.json"),
        schemas=_schemas_full((SCHEMA_AR026, 1), (SCHEMA_AR023, 1), (SCHEMA_AR026, 1)),
        validator="ar026",
    ),
)

FACE_IDS = tuple(face.face_id for face in FACE_QUEUE)


@dataclass
class Probes:
    """Injectable environment probes (fake instances are used by the tests)."""

    transformers_version: Callable[[], str]
    cuda: Callable[[], tuple[bool, int]]
    total_vram_gb: Callable[[], float]
    model_files_ok: Callable[[Path, str], bool]
    probe_artifact_ok: Callable[[Path], bool]
    gpu_state: Callable[[], tuple[float, float] | None]


Runner = Callable[[Sequence[str]], tuple[int, str, str]]


def _default_probes() -> Probes:
    def _version() -> str:
        import transformers

        return str(transformers.__version__)

    def _cuda() -> tuple[bool, int]:
        import torch

        if not torch.cuda.is_available():
            return False, 0
        return True, int(torch.cuda.device_count())

    def _vram() -> float:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        return float(props.total_memory) / (1024.0**3)

    def _model_files(root: Path, family: str) -> bool:
        directory = Path(root) / family
        if not (directory / "config.json").is_file():
            return False
        return any(directory.glob("*.safetensors"))

    def _probe_ok(path: Path) -> bool:
        return Path(path).is_file()

    def _gpu_state() -> tuple[float, float] | None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    return Probes(
        transformers_version=_version,
        cuda=_cuda,
        total_vram_gb=_vram,
        model_files_ok=_model_files,
        probe_artifact_ok=_probe_ok,
        gpu_state=_gpu_state,
    )


def _default_runner(timeout_hours: float | None) -> Runner:
    def _run(argv: Sequence[str]) -> tuple[int, str, str]:
        timeout = None if timeout_hours is None else timeout_hours * 3600.0
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            cwd=Path(__file__).resolve().parents[1],
        )
        return completed.returncode, completed.stdout, completed.stderr

    return _run


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _stamp(moment: _dt.datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stem = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base / stem
    counter = 1
    while candidate.exists():
        candidate = base / f"{stem}-{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def _float_close(left: float, right: float, ndigits: int = 6) -> bool:
    if math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6):
        return True
    return round(left, ndigits) == round(right, ndigits)


def _compare(recomputed: Any, recorded: Any, path: str, out: list[str], ndigits: int = 6) -> None:
    """Recursive semantic comparison (strict for semantics, tolerant for float noise)."""
    if isinstance(recorded, dict):
        if not isinstance(recomputed, dict):
            out.append(f"{path}: expected dict, got {type(recomputed).__name__}")
            return
        for key, value in recorded.items():
            if key not in recomputed:
                out.append(f"{path}.{key}: missing in recomputed verdict")
                continue
            _compare(recomputed[key], value, f"{path}.{key}", out, ndigits)
        for key in recomputed:
            if key not in recorded:
                out.append(f"{path}.{key}: missing in recorded verdict")
        return
    if isinstance(recorded, list):
        if not isinstance(recomputed, list) or len(recomputed) != len(recorded):
            out.append(f"{path}: list shape mismatch")
            return
        for index, (left, right) in enumerate(zip(recomputed, recorded)):
            _compare(left, right, f"{path}[{index}]", out, ndigits)
        return
    if isinstance(recorded, bool) or isinstance(recomputed, bool):
        if recomputed != recorded:
            out.append(f"{path}: {recomputed!r} != {recorded!r}")
        return
    if isinstance(recorded, float) or isinstance(recomputed, float):
        if not isinstance(recorded, (int, float)) or not isinstance(recomputed, (int, float)):
            out.append(
                f"{path}: type mismatch {type(recomputed).__name__}/{type(recorded).__name__}"
            )
            return
        if not _float_close(float(recomputed), float(recorded), ndigits):
            out.append(f"{path}: {recomputed!r} != {recorded!r}")
        return
    if recomputed != recorded:
        out.append(f"{path}: {recomputed!r} != {recorded!r}")


def _map_ar023_per_prompt(per_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped = []
    for entry in per_prompt:
        if not isinstance(entry, dict):
            continue
        confidence = entry.get("posthoc_confidence")
        confidence = confidence if isinstance(confidence, dict) else {}
        means: dict[str, Any] = {}
        for mode in AR023_MODES:
            cell = confidence.get(mode)
            means[mode] = cell.get("overall_mean") if isinstance(cell, dict) else None
        decode_length = entry.get("decode_length")
        block_exact = entry.get("block_exact")
        mapped.append(
            {
                "prompt_index": entry.get("prompt_index"),
                "posthoc_overall_mean": means,
                "decode_length": decode_length if isinstance(decode_length, dict) else {},
                "block_exact": block_exact if isinstance(block_exact, dict) else {},
            }
        )
    return mapped


def _map_ar026_per_prompt(per_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped = []
    for entry in per_prompt:
        if not isinstance(entry, dict):
            continue
        confidence = entry.get("posthoc_confidence")
        confidence = confidence if isinstance(confidence, dict) else {}
        means: dict[str, Any] = {}
        for mode in AR026_MODES:
            cell = confidence.get(mode)
            means[mode] = cell.get("overall_mean") if isinstance(cell, dict) else None
        decode_length = entry.get("decode_length")
        anchor = entry.get("block_exact_vs_anchor")
        mapped.append(
            {
                "prompt_index": entry.get("prompt_index"),
                "posthoc_overall_mean": means,
                "decode_length": decode_length if isinstance(decode_length, dict) else {},
                "block_exact_vs_anchor": anchor if isinstance(anchor, dict) else {},
            }
        )
    return mapped


def _recompute_and_compare(face: FaceSpec, artifact: dict[str, Any]) -> list[str]:
    from scripts import ar023_cross_family_bench as ar023
    from scripts import ar026_real_graph_bench as ar026

    mismatches: list[str] = []
    verdicts_raw = artifact.get("verdicts")
    verdicts = verdicts_raw if isinstance(verdicts_raw, dict) else {}
    quality_raw = artifact.get("quality")
    quality = quality_raw if isinstance(quality_raw, dict) else {}
    per_prompt_raw = quality.get("per_prompt")
    per_prompt = per_prompt_raw if isinstance(per_prompt_raw, list) else []

    if face.validator in ("ar023", "ar024"):
        recorded_h1 = verdicts.get("h1")
        if not isinstance(recorded_h1, dict):
            mismatches.append("verdicts.h1: missing or not a dict")
        else:
            recomputed_h1 = ar023.h1_quality_verdict(
                {"per_prompt": _map_ar023_per_prompt(per_prompt)}
            )
            _compare(recomputed_h1, recorded_h1, "h1", mismatches)
    elif face.validator == "ar026":
        recorded_h1 = verdicts.get("h1_graph_parity")
        if not isinstance(recorded_h1, dict):
            mismatches.append("verdicts.h1_graph_parity: missing or not a dict")
        else:
            modes_raw = artifact.get("modes")
            modes = modes_raw if isinstance(modes_raw, dict) else {}
            engagement: dict[str, Any] = {}
            for mode in AR026_MODES:
                cell = modes.get(mode)
                eng = cell.get("engagement") if isinstance(cell, dict) else None
                engagement[mode] = dict(eng) if isinstance(eng, dict) else {}
            recomputed_h1 = ar026.h1_graph_parity_verdict(
                _map_ar026_per_prompt(per_prompt), engagement
            )
            _compare(recomputed_h1, recorded_h1, "h1_graph_parity", mismatches)

    if face.validator in ("ar023", "ar024", "ar026"):
        recorded_h2 = verdicts.get("h2")
        if not isinstance(recorded_h2, dict):
            mismatches.append("h2: missing or not a dict")
        else:
            numbers = recorded_h2.get("numbers") or {}
            try:
                p50_baseline = float(numbers["p50_baseline_ms"])
                p50_aligned = float(numbers["p50_aligned_ms"])
            except (KeyError, TypeError, ValueError):
                mismatches.append("h2.numbers: missing p50 inputs")
            else:
                recomputed_h2 = ar023.h2_speed_verdict(p50_baseline, p50_aligned, p50_aligned)
                if recomputed_h2.get("verdict") != recorded_h2.get("verdict"):
                    mismatches.append(
                        f"h2.verdict: {recomputed_h2.get('verdict')!r} != "
                        f"{recorded_h2.get('verdict')!r}"
                    )
                recorded_speedup = numbers.get("speedup_p50_vs_baseline")
                recomputed_speedup = (recomputed_h2.get("numbers") or {}).get(
                    "speedup_p50_vs_baseline"
                )
                if not (
                    isinstance(recorded_speedup, (int, float))
                    and isinstance(recomputed_speedup, (int, float))
                    and _float_close(float(recomputed_speedup), float(recorded_speedup), ndigits=4)
                ):
                    mismatches.append(
                        f"h2.numbers.speedup_p50_vs_baseline: {recomputed_speedup!r} != "
                        f"{recorded_speedup!r}"
                    )
    return mismatches


def _validate_structure(face: FaceSpec, artifact: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    verdicts = artifact.get("verdicts")

    if face.validator == "ar025_real":
        if not isinstance(verdicts, dict):
            return ["verdicts: missing or not a dict"]
        for key in AR025_VERDICT_KEYS:
            if key not in verdicts:
                mismatches.append(f"verdicts.{key}: missing")
        extra = sorted(set(verdicts) - set(AR025_VERDICT_KEYS))
        if extra:
            mismatches.append(f"verdicts: unexpected keys {', '.join(extra)}")
        for key in ("H1_token_parity", "H2_speed", "delivery_line", "H4_ledger_strictly_smaller"):
            entry = verdicts.get(key)
            if not isinstance(entry, dict) or not isinstance(entry.get("pass"), bool):
                mismatches.append(f"verdicts.{key}.pass: missing or not a bool")
        overall = verdicts.get("overall")
        if overall not in ("supported", "blocked"):
            mismatches.append(f"verdicts.overall: invalid value {overall!r}")
        reasons = verdicts.get("blocked_reasons")
        if not isinstance(reasons, list):
            mismatches.append("verdicts.blocked_reasons: missing or not a list")
        elif overall == "blocked" and not reasons:
            mismatches.append("verdicts.blocked_reasons: empty on a blocked overall")
        elif overall == "supported" and reasons:
            mismatches.append("verdicts.blocked_reasons: non-empty on a supported overall")
        if overall == "supported":
            for key in (
                "H1_token_parity",
                "H2_speed",
                "delivery_line",
                "H4_ledger_strictly_smaller",
            ):
                entry = verdicts.get(key)
                if isinstance(entry, dict) and entry.get("pass") is False:
                    mismatches.append(f"verdicts.{key}.pass: false on a supported overall")
            h2 = verdicts.get("H2_speed")
            if not isinstance(h2, dict) or not isinstance(
                h2.get("p50_ratio_b_over_a"), (int, float)
            ):
                mismatches.append("verdicts.H2_speed.p50_ratio_b_over_a: missing numeric face")
            delivery = verdicts.get("delivery_line")
            if not isinstance(delivery, dict) or not isinstance(
                delivery.get("relative_speedup"), (int, float)
            ):
                mismatches.append("verdicts.delivery_line.relative_speedup: missing numeric face")
            h4 = verdicts.get("H4_ledger_strictly_smaller")
            if not isinstance(h4, dict) or not isinstance(h4.get("ops_a_merged"), int):
                mismatches.append(
                    "verdicts.H4_ledger_strictly_smaller.ops_a_merged: missing numeric face"
                )
        reproduction = verdicts.get("reproduction")
        if not isinstance(reproduction, dict) or not isinstance(reproduction.get("ok"), bool):
            mismatches.append("verdicts.reproduction.ok: missing or not a bool")
        canary = verdicts.get("canary_structural_eager_fallback")
        if not isinstance(canary, dict):
            mismatches.append("verdicts.canary_structural_eager_fallback: not a dict")
        elif canary.get("timed_count") is None and not canary.get("note"):
            mismatches.append("verdicts.canary_structural_eager_fallback.note: empty")
        return mismatches

    reproduction = artifact.get("reproduction")
    if not isinstance(reproduction, dict):
        mismatches.append("reproduction: missing or not a dict")
    elif not isinstance(reproduction.get("ok"), bool):
        mismatches.append("reproduction.ok: missing or not a bool")

    if isinstance(verdicts, dict):
        scalar_keys = {"overall", "blocked_reasons", "reproduction"}
        for key, entry in verdicts.items():
            if key in scalar_keys:
                continue
            if not isinstance(entry, dict):
                mismatches.append(f"verdicts.{key}: not a dict")
                continue
            verdict = entry.get("verdict")
            if verdict is not None and verdict not in (
                "supported",
                "refuted",
                "inconclusive",
                "blocked",
            ):
                mismatches.append(f"verdicts.{key}.verdict: invalid value {verdict!r}")
            if artifact.get("status") == "blocked" and verdict == "supported":
                mismatches.append(f"verdicts.{key}.verdict: supported on a blocked artifact")
    return mismatches


def validate_artifact(face: FaceSpec, artifact: dict[str, Any]) -> list[str]:
    """Schema + frozen-verdict recomputation + structural self-consistency."""
    mismatches: list[str] = []
    if not isinstance(artifact, dict):
        return ["artifact: not a JSON object"]
    status = artifact.get("status")
    if status not in ("ok", "blocked"):
        mismatches.append(f"status: invalid or missing ({status!r})")
    classification = classify_blocked(face, artifact)
    if status == "ok":
        state = "ok"
    elif classification == "legit_degraded":
        state = "blocked_degraded"
    elif classification == "legit_negative":
        state = "ok"
    else:
        state = "blocked_exception"
    expected_schema, expected_version = face.schemas[state]
    if artifact.get("schema") != expected_schema:
        mismatches.append(f"schema: {artifact.get('schema')!r} != {expected_schema!r}")
    if artifact.get("schema_version") != expected_version:
        mismatches.append(
            f"schema_version: {artifact.get('schema_version')!r} != {expected_version!r}"
        )

    verdicts = artifact.get("verdicts", {})
    if verdicts is None:
        verdicts = {}
    if not isinstance(verdicts, dict):
        mismatches.append("verdicts: not a dict")
        verdicts = {}
    quality = artifact.get("quality")
    if quality is not None and not isinstance(quality, dict):
        mismatches.append("quality: not a dict")
    has_verdicts = bool(verdicts)
    if status == "blocked" and has_verdicts and classification == "failure":
        mismatches.append("verdicts: blocked artifact carrying verdicts is classified as failure")
    if status == "ok":
        if not has_verdicts:
            mismatches.append("verdicts: missing on a status=ok artifact")
        expected_verdicts = (
            ("h1_graph_parity", "h2", "h4")
            if face.validator == "ar026"
            else (
                ("h1", "h2", "h4") if face.validator in ("ar023", "ar024") else AR025_VERDICT_KEYS
            )
        )
        for key in expected_verdicts:
            if key not in verdicts:
                mismatches.append(f"verdicts.{key}: missing on a status=ok artifact")
        if has_verdicts:
            mismatches.extend(_safe_checks(face, artifact))
        return mismatches

    if status == "blocked" and not has_verdicts:
        if not artifact.get("blocked_reason"):
            mismatches.append("blocked_reason: missing or empty on a blocked artifact")
        if artifact.get("modes") is not None:
            mismatches.append("modes: must be null on a blocked artifact (no fabricated numbers)")
        if artifact.get("quality") is not None:
            mismatches.append("quality: must be null on a blocked artifact (no fabricated numbers)")
        if face.validator in ("ar023", "ar024", "ar026") and classification == "legit_degraded":
            reproduction = artifact.get("reproduction")
            if not isinstance(reproduction, dict):
                mismatches.append("reproduction: must be a dict on a degraded blocked artifact")
            else:
                if not isinstance(reproduction.get("ok"), bool):
                    mismatches.append(
                        "reproduction.ok: missing or not a bool on a degraded blocked artifact"
                    )
                if reproduction.get("skipped") is not True:
                    mismatches.append(
                        "reproduction.skipped: must be true on a degraded blocked artifact"
                    )
                elif not reproduction.get("note"):
                    mismatches.append(
                        "reproduction.note: must be non-empty on a degraded blocked artifact"
                    )
        return mismatches

    if has_verdicts:
        mismatches.extend(_safe_checks(face, artifact))
    return mismatches


def _safe_checks(face: FaceSpec, artifact: dict[str, Any]) -> list[str]:
    """Run recomputation/structure checks; malformed nested scalars become mismatches."""
    try:
        mismatches = _recompute_and_compare(face, artifact)
        mismatches.extend(_validate_structure(face, artifact))
    except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError) as exc:
        return [f"verdicts: malformed nested value ({type(exc).__name__}: {exc})"]
    return mismatches


def classify_blocked(face: FaceSpec, artifact: dict[str, Any]) -> str | None:
    """Classify blocked artifacts (legal probe degradation vs failures)."""
    status = artifact.get("status")
    if status == "ok":
        reproduction = artifact.get("reproduction")
        if (
            face.validator in ("ar023", "ar024", "ar026")
            and isinstance(reproduction, dict)
            and reproduction.get("ok") is False
        ):
            return "failure"
        return None
    if status != "blocked":
        return None

    if face.validator == "ar025_real":
        reason = artifact.get("blocked_reason")
        verdicts = artifact.get("verdicts")
        if isinstance(reason, str) and reason.startswith(DEGRADE_REASON_PREFIX):
            if isinstance(verdicts, dict) and verdicts:
                return "failure"
            return "legit_degraded"
        if isinstance(reason, str) and reason.startswith(RUN_FAILED_PREFIX):
            return "failure"
        if not reason and isinstance(verdicts, dict) and verdicts:
            reproduction = verdicts.get("reproduction")
            if not isinstance(reproduction, dict) or reproduction.get("ok") is not True:
                return "failure"
            if verdicts.get("overall") != "blocked":
                return "failure"
            return "legit_negative"
        return "failure"

    reproduction = artifact.get("reproduction")
    verdicts = artifact.get("verdicts")
    if (
        isinstance(reproduction, dict)
        and reproduction.get("note") == DEGRADE_NOTE
        and not (isinstance(verdicts, dict) and verdicts)
    ):
        return "legit_degraded"
    return "failure"


def _select_faces(only: str | None) -> list[FaceSpec]:
    if only is None:
        return list(FACE_QUEUE)
    tokens = [token.strip() for token in only.split(",") if token.strip()]
    unknown = [token for token in tokens if token not in FACE_IDS]
    if not tokens or unknown:
        raise ValueError(f"unknown --only ids: {', '.join(unknown) or '(empty)'}")
    selected = set(tokens)
    return [face for face in FACE_QUEUE if face.face_id in selected]


def _preflight(
    faces: list[FaceSpec],
    probes: Probes,
    *,
    quick: bool,
    models_root: Path,
    allow_busy: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    items: list[dict[str, Any]] = []

    def _probe_failed(name: str, exc: BaseException) -> dict[str, Any]:
        return {
            "name": name,
            "ok": False,
            "detail": f"probe raised {type(exc).__name__}: {exc}",
        }

    try:
        version = probes.transformers_version()
    except Exception as exc:  # noqa: BLE001 - probes must never crash preflight
        items.append(_probe_failed("transformers_version", exc))
    else:
        items.append(
            {
                "name": "transformers_version",
                "ok": version == TRANSFORMERS_PIN,
                "detail": f"version={version} (required {TRANSFORMERS_PIN})",
            }
        )

    try:
        available, count = probes.cuda()
    except Exception as exc:  # noqa: BLE001
        items.append(_probe_failed("cuda", exc))
    else:
        items.append(
            {
                "name": "cuda",
                "ok": bool(available) and count >= 1,
                "detail": f"available={available} devices={count}",
            }
        )

    required = QUICK_VRAM_GB if quick else FORMAL_VRAM_GB
    try:
        vram = float(probes.total_vram_gb())
    except Exception as exc:  # noqa: BLE001
        items.append(_probe_failed("vram", exc))
    else:
        items.append(
            {
                "name": "vram",
                "ok": vram >= required,
                "detail": (
                    f"total={vram:.1f}GiB required>={required:.1f}GiB "
                    f"({'quick' if quick else 'formal'})"
                ),
            }
        )

    missing: list[str] = []
    for face in faces:
        try:
            ok = probes.model_files_ok(models_root, face.family)
        except Exception as exc:  # noqa: BLE001
            items.append(_probe_failed("model_files", exc))
            missing = []
            break
        if not ok:
            missing.append(face.family)
    else:
        missing = sorted(set(missing))
    if not any(item["name"] == "model_files" for item in items):
        items.append(
            {
                "name": "model_files",
                "ok": not missing,
                "detail": (
                    "ok" if not missing else f"missing checkpoint files for {', '.join(missing)}"
                ),
            }
        )

    try:
        probe_ok = probes.probe_artifact_ok(DEFAULT_PROBE_ARTIFACT)
    except Exception as exc:  # noqa: BLE001 - probes must never crash preflight
        items.append(_probe_failed("probe_artifact", exc))
    else:
        items.append(
            {
                "name": "probe_artifact",
                "ok": probe_ok,
                "detail": f"{DEFAULT_PROBE_ARTIFACT} exists={probe_ok}",
            }
        )

    try:
        state = probes.gpu_state()
    except Exception as exc:  # noqa: BLE001
        items.append(_probe_failed("gpu_silence", exc))
        state = None
    if state is None:
        gpu_ok = False
        gpu_detail = "nvidia-smi unavailable (not overridable)"
    else:
        util, used = state
        gpu_ok = util <= GPU_MAX_UTIL_PCT and used <= GPU_MAX_USED_MIB
        gpu_detail = f"util={util:.1f}% used={used:.0f}MiB"
        if not gpu_ok and allow_busy:
            gpu_ok = True
            gpu_detail += " (allow-busy override)"
    if not any(item["name"] == "gpu_silence" for item in items):
        items.append({"name": "gpu_silence", "ok": gpu_ok, "detail": gpu_detail})

    failed = [item for item in items if not item["ok"]]
    if not failed:
        return items, None
    reasons = "; ".join(f"{item['name']} ({item['detail']})" for item in failed)
    return items, reasons


def _resolve_models_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return _resolve_path(Path(explicit))
    try:
        probe = json.loads(DEFAULT_PROBE_ARTIFACT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_MODELS_ROOT
    if not isinstance(probe, dict):
        return DEFAULT_MODELS_ROOT
    meta = probe.get("meta")
    if not isinstance(meta, dict):
        return DEFAULT_MODELS_ROOT
    meta_root = meta.get("models_root")
    return Path(meta_root) if isinstance(meta_root, str) and meta_root else DEFAULT_MODELS_ROOT


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


class _Logger:
    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "queue.log"
        self._handle = self.path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{_stamp(_utc_now())}] {message}"
        self._handle.write(line + "\n")
        self._handle.flush()
        print(line)

    def close(self) -> None:
        self._handle.close()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _artifacts_plan(faces: list[FaceSpec], quick: bool) -> dict[str, Path]:
    return {
        face.face_id: _resolve_path(face.quick_artifact if quick else face.formal_artifact)
        for face in faces
    }


def _print_queue(faces: list[FaceSpec], *, quick: bool) -> None:
    for face in faces:
        args = face.quick_argv if quick else face.formal_argv
        artifact = face.quick_artifact if quick else face.formal_artifact
        command = " ".join(["python", "-m", face.module, *args])
        print(f"{face.face_id}\t{command}\t{artifact}")


def main(
    argv: list[str] | None = None,
    *,
    probes: Probes | None = None,
    runner: Runner | None = None,
    run_dir: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="AR027 server formal-run queue executor (pre-committed single shot)"
    )
    parser.add_argument("--only", type=str, default=None, help="comma separated face ids")
    parser.add_argument("--list", action="store_true", help="print the queue and exit")
    parser.add_argument("--quick", action="store_true", help="run each face with --quick")
    parser.add_argument("--models-root", type=Path, default=None)
    parser.add_argument(
        "--allow-busy", action="store_true", help="override only the GPU silence gate"
    )
    parser.add_argument("--force", action="store_true", help="allow overwriting an existing target")
    parser.add_argument("--force-reason", type=str, default=None)
    parser.add_argument("--timeout-hours", type=float, default=None)
    args = parser.parse_args(argv)

    try:
        faces = _select_faces(args.only)
    except ValueError as exc:
        print(f"error: {exc}")
        print("legal face ids: " + ", ".join(FACE_IDS))
        return EXIT_PREFLIGHT

    if args.list:
        _print_queue(faces, quick=args.quick)
        return EXIT_OK

    if args.force and not (args.force_reason or "").strip():
        print("error: --force requires a non-empty --force-reason")
        return EXIT_PREFLIGHT
    if args.timeout_hours is not None and args.timeout_hours <= 0:
        print("error: --timeout-hours must be > 0")
        return EXIT_PREFLIGHT

    active_probes = probes if probes is not None else _default_probes()
    active_runner = runner if runner is not None else _default_runner(args.timeout_hours)
    base = Path(run_dir) if run_dir is not None else DEFAULT_RUNS_BASE
    directory = _resolve_run_dir(base)
    logger = _Logger(directory)
    queue_json_path = directory / "queue.json"

    models_root = _resolve_models_root(args.models_root)
    preflight_items, blocked_reason = _preflight(
        faces,
        active_probes,
        quick=args.quick,
        models_root=models_root,
        allow_busy=args.allow_busy,
    )

    queue: dict[str, Any] = {
        "run_id": directory.name,
        "argv": list(sys.argv if argv is None else argv),
        "quick": bool(args.quick),
        "allow_busy": bool(args.allow_busy),
        "force_reason": None,
        "preflight": preflight_items,
        "faces": [],
        "overall": "failed",
    }

    logger.log(f"run {directory.name}: {len(faces)} face(s), quick={args.quick}")
    for item in preflight_items:
        logger.log(
            f"preflight {item['name']}: {'PASS' if item['ok'] else 'FAIL'} ({item['detail']})"
        )
    if args.allow_busy:
        logger.log("allow-busy override recorded by operator")

    if blocked_reason is not None:
        logger.log(f"preflight blocked: {blocked_reason}")
        logger.log(
            "fix per docs/RERUN_CHECKLIST.md section 0 (environment/venv) and section 7 "
            "(server reproduction: model root, GPU silence, clock)"
        )
        queue["overall"] = "failed"
        _write_json(queue_json_path, queue)
        logger.close()
        return EXIT_PREFLIGHT

    _write_json(queue_json_path, queue)
    targets = _artifacts_plan(faces, args.quick)
    for face in faces:
        queue["faces"].append(
            {
                "id": face.face_id,
                "argv": [],
                "start": None,
                "end": None,
                "duration_s": None,
                "exit_code": None,
                "artifact": str(targets[face.face_id]),
                "validation": None,
                "outcome": "not_run",
                "overwrite": False,
            }
        )
    exit_code = EXIT_OK
    overall = "ok"

    for index, face in enumerate(faces):
        target = targets[face.face_id]
        entry = queue["faces"][index]

        if target.exists():
            if args.force and (args.force_reason or "").strip():
                entry["overwrite"] = True
                queue["force_reason"] = args.force_reason
                logger.log(f"overwrite permitted for {target} (force-reason: {args.force_reason})")
            else:
                logger.log(f"overwrite protection: {target} exists (use --force --force-reason)")
                entry["outcome"] = "failure"
                queue["overall"] = "failed"
                for remaining in queue["faces"][index + 1 :]:
                    remaining["outcome"] = "not_run"
                _write_json(queue_json_path, queue)
                logger.close()
                return EXIT_FAILURE

        base_argv = [sys.executable, "-m", face.module]
        args_quick = list(face.quick_argv) if args.quick else list(face.formal_argv)
        if args.models_root is not None:
            args_quick.extend(["--models-root", str(models_root)])
        command = base_argv + args_quick
        entry["argv"] = command
        face_log_path = directory / f"face-{face.face_id}.log"

        started = _utc_now()
        entry["start"] = _stamp(started)
        logger.log(f"face {face.face_id} start: {' '.join(command)}")
        started_perf = time.perf_counter()
        code: int
        stdout: str = ""
        stderr: str = ""
        failure_note: str | None = None
        try:
            code, stdout, stderr = active_runner(command)
        except subprocess.TimeoutExpired as exc:
            code = 124
            failure_note = f"timeout: {exc}"

            def _as_text(value: Any) -> str:
                if value is None:
                    return ""
                return (
                    value.decode("utf-8", errors="replace")
                    if isinstance(value, bytes)
                    else str(value)
                )

            stderr = f"{failure_note}\n{_as_text(exc.stderr)}"
            stdout = _as_text(exc.stdout)
        except OSError as exc:
            code = 125
            failure_note = f"OSError: {exc}"
            stderr = f"{failure_note}\n"

        elapsed = time.perf_counter() - started_perf
        ended = _utc_now()
        entry["end"] = _stamp(ended)
        entry["duration_s"] = round(elapsed, 3)
        entry["exit_code"] = code

        body = [f"argv: {' '.join(command)}"]
        if stdout:
            body.append(stdout.rstrip("\n"))
        if stderr:
            body.append(stderr.rstrip("\n"))
        body.append(f"exit_code={code}")
        face_log_path.write_text("\n".join(body) + "\n", encoding="utf-8")
        logger.log(f"face {face.face_id} done: exit_code={code} duration_s={elapsed:.2f}")

        if failure_note is not None:
            logger.log(f"face {face.face_id} failed: {failure_note}")
        if code != 0:
            entry["outcome"] = "failure"
            exit_code = EXIT_FAILURE
            overall = "failed"
            for remaining in queue["faces"][index + 1 :]:
                remaining["outcome"] = "not_run"
            break

        try:
            artifact = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(artifact, dict):
                raise ValueError("artifact is not a JSON object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.log(f"face {face.face_id} artifact unreadable: {exc}")
            entry["outcome"] = "failure"
            entry["validation"] = "missing"
            exit_code = EXIT_VALIDATION
            overall = "failed"
            for remaining in queue["faces"][index + 1 :]:
                remaining["outcome"] = "not_run"
            break

        classification = classify_blocked(face, artifact)
        if classification == "failure":
            logger.log(f"face {face.face_id} classified as failure (blocked source)")
            entry["outcome"] = "failure"
            exit_code = EXIT_FAILURE
            overall = "failed"
            for remaining in queue["faces"][index + 1 :]:
                remaining["outcome"] = "not_run"
            break

        mismatches = validate_artifact(face, artifact)
        if mismatches:
            existing_lines = face_log_path.read_text(encoding="utf-8").rstrip("\n").splitlines()
            detail_lines = [f"mismatch: {mismatch}" for mismatch in mismatches]
            kept = existing_lines[:-1] if existing_lines else []
            face_log_path.write_text(
                "\n".join(kept + detail_lines + ["exit_code=0"]) + "\n",
                encoding="utf-8",
            )
            for mismatch in mismatches:
                logger.log(f"face {face.face_id} mismatch: {mismatch}")
            entry["outcome"] = "failure"
            entry["validation"] = "mismatch"
            exit_code = EXIT_VALIDATION
            overall = "failed"
            for remaining in queue["faces"][index + 1 :]:
                remaining["outcome"] = "not_run"
            break

        entry["validation"] = "pass"
        if classification == "legit_degraded":
            entry["outcome"] = "blocked"
            overall = "blocked" if overall == "ok" else overall
        elif classification == "legit_negative":
            entry["outcome"] = "blocked"
            overall = "blocked" if overall == "ok" else overall
        else:
            entry["outcome"] = "ok"
        _write_json(queue_json_path, queue)

    queue["overall"] = overall if exit_code == EXIT_OK else "failed"
    _write_json(queue_json_path, queue)
    logger.log(f"queue overall: {queue['overall']}")
    logger.close()
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
