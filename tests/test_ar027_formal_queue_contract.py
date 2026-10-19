"""CPU contract tests for AR027: ``scripts/run_formal_queue.py`` (UT-2701..UT-2764).

Contract source: ``specs/changes/AR027-server-formal-runner/design.md``
(§4.2 flow, §4.3 interfaces / exit codes / queue.json fields, §6.1 UT table)
and ``srs.md`` §3.1-§3.7.

The executor is exercised entirely through injected dependencies:

* a fake ``Probes`` dataclass instance (six callables, no GPU / torch probing);
* a fake ``runner(argv) -> (exit_code, stdout, stderr)`` that records argv and
  materialises fixture artifacts in ``tmp_path`` (no real subprocess is ever
  started);
* ``run_dir`` is the BASE directory under which the executor creates its
  ``<UTC>`` run directory.  This is the assumption the tests record explicitly:
  two consecutive runs under one ``run_dir`` must yield two distinct ``<UTC>``
  child directories (UT-2737).  When ``run_dir`` is omitted the executor uses
  the repository's ``results/optimization/formal_runs`` base; the tests never
  rely on that path.

All ``FaceSpec`` artifact paths are redirected to ``tmp_path`` via
``dataclasses.replace`` + monkeypatching the module-level ``FACE_QUEUE``, so no
test writes anything under the real ``results/`` tree.  The only repository
read is UT-2726/UT-2727's copy of the committed
``results/optimization/ar023_llada_bench.json`` (read-only).

FR -> test mapping
==================
FR1 (queue definition / selection):
    test_ut2701_queue_frozen_seven_faces
    test_ut2702_list_prints_queue_without_running
    test_ut2703_only_subset_preserves_queue_order
    test_ut2704_only_unknown_id_exits_3_and_lists_ids
    test_ut2705_models_root_absent_not_forwarded
    test_ut2706_models_root_present_forwarded
    test_ut2707_quick_ar026_llada_omits_out
    test_ut2708_quick_ar026_dream_uses_dream_quick_out
    test_ut2709_quick_other_faces_use_quick_argv
    test_ut2741_only_empty_string_exits_3
FR2 (preflight):
    test_ut2710_preflight_transformers_version
    test_ut2711_preflight_cuda_unavailable
    test_ut2712_preflight_vram_boundaries
    test_ut2713_preflight_model_files
    test_ut2714_preflight_probe_artifact
    test_ut2715_preflight_gpu_silence_boundaries
    test_ut2716_allow_busy_overrides_gpu_silence
    test_ut2754_unmeasurable_gpu_not_overridable
    test_ut2756_probe_exception_blocks_with_exit_3
    test_ut2757_cuda_unavailable_vram_probe_exception
    test_ut2759_malformed_probe_json_falls_back_to_default_root
FR3 (subprocess execution / quick):
    test_ut2717_execution_logs_and_queue_json
    test_ut2718_first_failure_stops_queue
    test_ut2719_overwrite_protection_without_force
    test_ut2720_force_with_empty_reason_exits_3
    test_ut2721_force_with_reason_overwrites_and_records
    test_ut2733_timeout_exits_4_and_stops
    test_ut2734_runner_oserror_exits_4_and_logs
    test_ut2739_reason_without_force_exits_4
    test_ut2742_invalid_timeout_hours_exits_3
FR4 (artifact validation):
    test_ut2722_schema_table_ok_faces
    test_ut2723_schema_table_degraded_blocked
    test_ut2724_schema_table_exception_blocked
    test_ut2725_tampered_verdict_exits_2_with_diff
    test_ut2726_committed_ar023_artifact_validates
    test_ut2727_h1_mapping_recompute_matches_record
    test_ut2731_structure_self_consistency
    test_ut2735_missing_or_bad_json_exits_2
    test_ut2755_nested_container_malformed_is_mismatch
    test_ut2760_degraded_requires_boolean_reproduction_ok
    test_ut2736_validation_does_not_write_artifact
    test_ut2743_ar025_exact_structure_keys
FR5 (failure semantics / blocked classification / overwrite):
    test_ut2718_first_failure_stops_queue
    test_ut2719_overwrite_protection_without_force
    test_ut2720_force_with_empty_reason_exits_3
    test_ut2721_force_with_reason_overwrites_and_records
    test_ut2728_classify_degraded_blocked
    test_ut2729_classify_exception_blocked
    test_ut2730_classify_ar025_reproduction
    test_ut2732_status_ok_with_reproduction_failure_exits_4
    test_ut2740_blocked_fallback_is_failure
    test_ut2738_quick_not_killed_by_formal_artifact
    test_ut2739_reason_without_force_exits_4
    test_ut2758_force_without_existing_target_is_not_recorded_as_overwrite
    test_ut2761_ar025_negative_requires_blocked_overall_and_true_reproduction
    test_ut2762_ar025_supported_overall_under_blocked_status_exits_4
    test_ut2763_degraded_reason_with_verdicts_is_failure
    test_ut2764_ar026_top_level_engagement_tamper_is_mismatch
FR6 (logs / exit codes / run-dir isolation):
    test_ut2717_execution_logs_and_queue_json
    test_ut2737_run_directory_isolation
FR1..FR7 (review regressions, T010):
    test_ut2744_ok_without_verdicts_is_mismatch
    test_ut2745_allow_busy_cannot_bypass_other_gates
    test_ut2746_face_log_first_line_is_full_argv
    test_ut2747_list_quick_shows_quick_argv
    test_ut2748_invalid_or_missing_status_is_mismatch
    test_ut2749_non_object_json_exits_2
    test_ut2750_h4_non_dict_is_mismatch
    test_ut2751_ar025_extra_key_is_mismatch
    test_ut2752_all_seven_green_exit_zero
    test_ut2753_mixed_blocked_continues_queue
    test_ut2754_unmeasurable_gpu_not_overridable
    test_ut2755_nested_container_malformed_is_mismatch
    test_ut2756_probe_exception_blocks_with_exit_3
    test_ut2757_cuda_unavailable_vram_probe_exception
    test_ut2758_force_without_existing_target_is_not_recorded_as_overwrite
    test_ut2759_malformed_probe_json_falls_back_to_default_root
    test_ut2760_degraded_requires_boolean_reproduction_ok
    test_ut2761_ar025_negative_requires_blocked_overall_and_true_reproduction
    test_ut2762_ar025_supported_overall_under_blocked_status_exits_4
    test_ut2763_degraded_reason_with_verdicts_is_failure
    test_ut2764_ar026_top_level_engagement_tamper_is_mismatch
FR7 (this file: CPU-only injectable contract tests):
    every test above (fake Probes + fake runner + tmp run_dir; no
    subprocess / GPU / network dependency)
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, Callable

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.ar023_cross_family_bench as ar023_bench  # noqa: E402
import scripts.ar026_real_graph_bench as ar026_bench  # noqa: E402
import scripts.run_formal_queue as formal_queue  # noqa: E402
from scripts.run_formal_queue import (  # noqa: E402
    FACE_QUEUE,
    FaceSpec,
    Probes,
    classify_blocked,
    main,
    validate_artifact,
)

# ---------------------------------------------------------------------------
# Frozen expectations (design.md §4.3 tables)
# ---------------------------------------------------------------------------

_EXPECTED_ORDER = (
    "ar023-llada",
    "ar023-dream",
    "ar024",
    "ar025-real-llada",
    "ar025-real-dream",
    "ar026-llada",
    "ar026-dream",
)

_EXPECTED_FORMAL = {
    "ar023-llada": (
        "scripts.ar023_cross_family_bench",
        "llada",
        ("--family", "llada", "--reps", "30"),
        Path("results/optimization/ar023_llada_bench.json"),
    ),
    "ar023-dream": (
        "scripts.ar023_cross_family_bench",
        "dream",
        ("--family", "dream", "--reps", "30"),
        Path("results/optimization/ar023_dream_bench.json"),
    ),
    "ar024": (
        "scripts.ar024_granularity_bench",
        "llada",
        ("--family", "llada", "--reps", "30"),
        Path("results/optimization/ar024_llada_granularity_bench.json"),
    ),
    "ar025-real-llada": (
        "scripts.ar025_multi_parent_bench",
        "llada",
        ("--mode", "real", "--family", "llada", "--reps", "30"),
        Path("results/optimization/ar025_real_bench_llada.json"),
    ),
    "ar025-real-dream": (
        "scripts.ar025_multi_parent_bench",
        "dream",
        ("--mode", "real", "--family", "dream", "--reps", "30"),
        Path("results/optimization/ar025_real_bench_dream.json"),
    ),
    "ar026-llada": (
        "scripts.ar026_real_graph_bench",
        "llada",
        (
            "--family",
            "llada",
            "--reps",
            "30",
            "--out",
            "results/optimization/ar026_real_graph_bench.json",
        ),
        Path("results/optimization/ar026_real_graph_bench.json"),
    ),
    "ar026-dream": (
        "scripts.ar026_real_graph_bench",
        "dream",
        (
            "--family",
            "dream",
            "--reps",
            "30",
            "--out",
            "results/optimization/ar026_real_graph_bench_dream.json",
        ),
        Path("results/optimization/ar026_real_graph_bench_dream.json"),
    ),
}

_EXPECTED_QUICK = {
    "ar023-llada": (
        ("--family", "llada", "--reps", "30", "--quick"),
        Path("results/optimization/ar023_llada_bench_quick.json"),
    ),
    "ar023-dream": (
        ("--family", "dream", "--reps", "30", "--quick"),
        Path("results/optimization/ar023_dream_bench_quick.json"),
    ),
    "ar024": (
        ("--family", "llada", "--reps", "30", "--quick"),
        Path("results/optimization/ar024_llada_granularity_bench_quick.json"),
    ),
    "ar025-real-llada": (
        ("--mode", "real", "--family", "llada", "--reps", "30", "--quick"),
        Path("results/optimization/ar025_real_bench_llada_quick.json"),
    ),
    "ar025-real-dream": (
        ("--mode", "real", "--family", "dream", "--reps", "30", "--quick"),
        Path("results/optimization/ar025_real_bench_dream_quick.json"),
    ),
    "ar026-llada": (
        ("--family", "llada", "--reps", "30", "--quick"),
        Path("results/optimization/ar026_real_graph_bench_quick.json"),
    ),
    "ar026-dream": (
        (
            "--family",
            "dream",
            "--reps",
            "30",
            "--quick",
            "--out",
            "results/optimization/ar026_real_graph_bench_dream_quick.json",
        ),
        Path("results/optimization/ar026_real_graph_bench_dream_quick.json"),
    ),
}

_BATCHED = "folded-batched"
_ALIGNED = "folded-batched-greedy"
_SERIAL = "folded-serial"
_AR023_MODES = ("baseline", _SERIAL, _BATCHED, _ALIGNED)
_AR026_MODES = ("baseline", _BATCHED, "graph-fixed", "graph-adaptive")


# ---------------------------------------------------------------------------
# Fixture artifacts (built with the benches' own frozen verdict functions)
# ---------------------------------------------------------------------------


def _ar023_per_prompt(index: int) -> dict[str, Any]:
    """One ar023/ar024 per-prompt quality entry with recorded raw fields."""
    return {
        "prompt_index": index,
        "decode_length": {
            "baseline": 10,
            _SERIAL: 10,
            _BATCHED: 12,
            _ALIGNED: 10,
        },
        "posthoc_confidence": {
            "baseline": {"overall_mean": 0.5, "overall_min": 0.4, "window": "[0, 10)"},
            _SERIAL: {"overall_mean": 0.5, "overall_min": 0.4, "window": "[0, 10)"},
            _BATCHED: {"overall_mean": 0.4, "overall_min": 0.3, "window": "[0, 12)"},
            _ALIGNED: {"overall_mean": 0.5, "overall_min": 0.4, "window": "[0, 10)"},
        },
        "block_exact": {
            _SERIAL: {"matched": 3, "compared": 4},
            _BATCHED: {"matched": 1, "compared": 4},
            _ALIGNED: {"matched": 3, "compared": 4},
        },
    }


def _mapped_ar023(per_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The h1 input mapping layer: posthoc_confidence -> posthoc_overall_mean."""
    return [
        {
            "prompt_index": entry["prompt_index"],
            "posthoc_overall_mean": {
                mode: entry["posthoc_confidence"][mode]["overall_mean"] for mode in _AR023_MODES
            },
            "decode_length": entry["decode_length"],
            "block_exact": entry["block_exact"],
        }
        for entry in per_prompt
    ]


def _ar023_family_artifact(family: str, schema: str, schema_version: int) -> dict[str, Any]:
    """A status=ok artifact for ar023/ar024 whose h1/h2 records are consistent."""
    per_prompt = [_ar023_per_prompt(0), _ar023_per_prompt(1)]
    h1 = ar023_bench.h1_quality_verdict({"per_prompt": _mapped_ar023(per_prompt)})
    h2 = ar023_bench.h2_speed_verdict(100.0, 80.0, 90.0)
    return {
        "schema": schema,
        "schema_version": schema_version,
        "family": family,
        "status": "ok",
        "modes": {},
        "quality": {"per_prompt": per_prompt},
        "verdicts": {
            "h1": h1,
            "h2": h2,
            "h4": {"verdict": "supported", "numbers": {}},
        },
        "reproduction": {"ok": True},
        "meta": {},
    }


def _ar026_per_prompt(index: int) -> dict[str, Any]:
    """One ar026 per-prompt quality entry (four modes + graph parity fields)."""
    return {
        "prompt_index": index,
        "decode_length": {mode: 10 for mode in _AR026_MODES},
        "posthoc_confidence": {
            mode: {"overall_mean": 0.5, "overall_min": 0.4, "window": "[0, 10)"}
            for mode in _AR026_MODES
        },
        "block_exact_vs_anchor": {
            "graph-fixed": {"matched": 4, "compared": 4},
            "graph-adaptive": {"matched": 4, "compared": 4},
        },
    }


def _mapped_ar026(per_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt_index": entry["prompt_index"],
            "posthoc_overall_mean": {
                mode: entry["posthoc_confidence"][mode]["overall_mean"] for mode in _AR026_MODES
            },
            "decode_length": entry["decode_length"],
            "block_exact_vs_anchor": entry["block_exact_vs_anchor"],
        }
        for entry in per_prompt
    ]


def _ar026_artifact(family: str) -> dict[str, Any]:
    per_prompt = [_ar026_per_prompt(0), _ar026_per_prompt(1)]
    engagement = {
        "graph-fixed": {"captures": 2, "replays": 10, "evictions": 0},
        "graph-adaptive": {"captures": 2, "replays": 10, "evictions": 0},
    }
    h1 = ar026_bench.h1_graph_parity_verdict(_mapped_ar026(per_prompt), engagement)
    h2 = ar023_bench.h2_speed_verdict(100.0, 80.0, 90.0)
    return {
        "schema": "ar026_real_graph_bench",
        "schema_version": 1,
        "family": family,
        "status": "ok",
        "modes": {mode: {"engagement": dict(cell)} for mode, cell in engagement.items()},
        "quality": {"per_prompt": per_prompt},
        "verdicts": {
            "h1_graph_parity": h1,
            "h2": h2,
            "h4": {"verdict": "supported", "numbers": {}},
        },
        "reproduction": {"ok": True},
        "meta": {},
    }


def _ar025_verdicts(*, overall: str, reproduction_ok: bool) -> dict[str, Any]:
    """The exact eight-key ar025 real-face verdict block."""
    return {
        "H1_token_parity": {"pass": True, "rule": "frozen"},
        "H2_speed": {"pass": True, "rule": "frozen", "p50_ratio_b_over_a": 1.2},
        "delivery_line": {"pass": True, "rule": "frozen", "relative_speedup": 0.2},
        "H4_ledger_strictly_smaller": {
            "pass": True,
            "rule": "frozen",
            "ops_a_merged": 10,
            "ops_b_per_chain": 20,
        },
        "canary_structural_eager_fallback": {
            "timed_count": None,
            "note": "graph path off per the probe conclusions (eager-neutral)",
        },
        "overall": overall,
        "blocked_reasons": [] if overall == "supported" else ["frozen negative result"],
        "reproduction": {"ok": reproduction_ok},
    }


def _ar025_artifact(family: str) -> dict[str, Any]:
    return {
        "schema": "ar025_real_bench",
        "family": family,
        "status": "ok",
        "verdicts": _ar025_verdicts(overall="supported", reproduction_ok=True),
        "meta": {},
    }


def _ok_artifact(face: FaceSpec) -> dict[str, Any]:
    if face.validator == "ar023":
        return _ar023_family_artifact(face.family, "ar023_cross_family_bench", 1)
    if face.validator == "ar024":
        return _ar023_family_artifact(face.family, "ar024_granularity_bench", 1)
    if face.validator == "ar025_real":
        return _ar025_artifact(face.family)
    if face.validator == "ar026":
        return _ar026_artifact(face.family)
    raise AssertionError(f"unknown validator {face.validator!r}")


def _degraded_artifact(face: FaceSpec) -> dict[str, Any]:
    if face.validator == "ar025_real":
        return {
            "schema": "ar025_real_bench",
            "family": face.family,
            "status": "blocked",
            "blocked_reason": "family degraded per probe conclusion: probe says so",
            "meta": {},
        }
    return {
        "schema": "ar023_cross_family_bench",
        "schema_version": 1,
        "family": face.family,
        "status": "blocked",
        "blocked_reason": "degraded family per probe conclusion",
        "modes": None,
        "quality": None,
        "verdicts": {},
        "reproduction": {
            "ok": False,
            "reproduction_anomaly": False,
            "skipped": True,
            "note": "degraded family per probe conclusion",
        },
        "meta": {},
    }


def _exception_artifact(face: FaceSpec) -> dict[str, Any]:
    if face.validator == "ar025_real":
        return {
            "schema": "ar025_real_bench",
            "family": face.family,
            "status": "blocked",
            "blocked_reason": "real-face run failed: CUDA out of memory",
            "meta": {},
        }
    if face.validator == "ar024":
        schema, version = "ar024_granularity_bench", 1
    elif face.validator == "ar026":
        schema, version = "ar026_real_graph_bench", 1
    else:
        schema, version = "ar023_cross_family_bench", 1
    return {
        "schema": schema,
        "schema_version": version,
        "family": face.family,
        "status": "blocked",
        "blocked_reason": "RuntimeError: benchmark exploded",
        "modes": None,
        "quality": None,
        "verdicts": {},
        "reproduction": {
            "ok": False,
            "reproduction_anomaly": False,
            "skipped": True,
        },
        "meta": {},
    }


# ---------------------------------------------------------------------------
# Fake probes / fake runner / tmp-path helpers
# ---------------------------------------------------------------------------


def _make_probes(**overrides: Callable[..., Any]) -> Probes:
    fields: dict[str, Callable[..., Any]] = {
        "transformers_version": lambda: "4.53.1",
        "cuda": lambda: (True, 1),
        "total_vram_gb": lambda: 24.0,
        "model_files_ok": lambda root, family: True,
        "probe_artifact_ok": lambda path: True,
        "gpu_state": lambda: (0.0, 0.0),
    }
    fields.update(overrides)
    return Probes(**fields)


def _face_id_from_argv(argv: list[str]) -> str:
    module = argv[argv.index("-m") + 1]
    family = argv[argv.index("--family") + 1]
    if module.endswith("ar023_cross_family_bench"):
        return f"ar023-{family}"
    if module.endswith("ar024_granularity_bench"):
        return "ar024"
    if module.endswith("ar025_multi_parent_bench"):
        return f"ar025-real-{family}"
    if module.endswith("ar026_real_graph_bench"):
        return f"ar026-{family}"
    raise AssertionError(f"unexpected module in argv: {argv!r}")


class FakeRunner:
    """Recording runner: returns canned codes and materialises fixtures.

    ``runner(argv) -> (exit_code, stdout, stderr)``; records every argv and
    never starts a subprocess.  ``raises`` injects startup / timeout errors.
    """

    def __init__(
        self,
        targets: dict[str, Path],
        artifacts: dict[str, Any],
        *,
        write: bool = True,
        failures: dict[str, int] | None = None,
        raises: dict[str, BaseException] | None = None,
    ) -> None:
        self.targets = targets
        self.artifacts = artifacts
        self.write = write
        self.failures = failures or {}
        self.raises = raises or {}
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> tuple[int, str, str]:
        self.calls.append(list(argv))
        face_id = _face_id_from_argv(argv)
        exc = self.raises.get(face_id)
        if exc is not None:
            raise exc
        code = self.failures.get(face_id, 0)
        if code != 0:
            return code, f"{face_id} failed\n", f"{face_id} stderr\n"
        if self.write and face_id in self.targets:
            path = self.targets[face_id]
            path = path if path.is_absolute() else Path.cwd() / path
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.artifacts[face_id]
            if isinstance(payload, str):
                path.write_text(payload, encoding="utf-8")
            else:
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0, f"{face_id} ok\n", ""


def _patch_faces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FaceSpec, ...]:
    """Redirect every face's artifacts under tmp_path and patch FACE_QUEUE."""
    faces = tuple(
        replace(
            face,
            formal_artifact=tmp_path / "formal" / f"{face.face_id}.json",
            quick_artifact=tmp_path / "quick" / f"{face.face_id}_quick.json",
        )
        for face in FACE_QUEUE
    )
    monkeypatch.setattr(formal_queue, "FACE_QUEUE", faces)
    return faces


def _make_runner(
    faces: tuple[FaceSpec, ...],
    *,
    quick: bool = False,
    artifacts: dict[str, Any] | None = None,
    write: bool = True,
    failures: dict[str, int] | None = None,
    raises: dict[str, BaseException] | None = None,
) -> FakeRunner:
    targets = {
        face.face_id: (face.quick_artifact if quick else face.formal_artifact) for face in faces
    }
    arts = artifacts or {face.face_id: _ok_artifact(face) for face in faces}
    return FakeRunner(
        targets,
        arts,
        write=write,
        failures=failures,
        raises=raises,
    )


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(path for path in base.iterdir() if path.is_dir())


def _only_run_dir(base: Path) -> Path:
    dirs = _run_dirs(base)
    assert len(dirs) == 1, f"expected exactly one run directory, got {dirs!r}"
    return dirs[0]


def _read_queue_json(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "queue.json").read_text(encoding="utf-8"))


def _face_log(run_dir: Path, face_id: str) -> str:
    return (run_dir / f"face-{face_id}.log").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# FR1: queue definition and selection (UT-2701..2709, UT-2741)
# ---------------------------------------------------------------------------


def test_ut2701_queue_frozen_seven_faces() -> None:
    """UT-2701 (FR1): seven frozen faces in order with full formal argv."""
    assert isinstance(FACE_QUEUE, tuple)
    assert len(FACE_QUEUE) == 7
    assert tuple(face.face_id for face in FACE_QUEUE) == _EXPECTED_ORDER

    for face in FACE_QUEUE:
        module, family, argv, artifact = _EXPECTED_FORMAL[face.face_id]
        assert isinstance(face, FaceSpec)
        assert face.module == module
        assert face.family == family
        assert tuple(face.formal_argv) == argv
        assert Path(face.formal_artifact) == artifact

    with pytest.raises(FrozenInstanceError):
        FACE_QUEUE[0].face_id = "mutated"  # type: ignore[misc]


def test_ut2702_list_prints_queue_without_running(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """UT-2702 (FR1): ``--list`` prints the seven faces and calls runner 0 times."""
    runner = FakeRunner({}, {})
    rc = main(["--list"], probes=_make_probes(), runner=runner, run_dir=tmp_path / "runs")
    output = capsys.readouterr()
    combined = output.out + output.err

    assert rc == 0
    assert runner.calls == []
    face_lines = [
        line for line in combined.splitlines() if any(fid in line for fid in _EXPECTED_ORDER)
    ]
    assert len(face_lines) == 7
    for face_id in _EXPECTED_ORDER:
        assert face_id in combined


def test_ut2703_only_subset_preserves_queue_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2703 (FR1): a ``--only`` subset executes in queue order."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces)
    rc = main(
        ["--only", "ar026-dream,ar023-llada"],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )

    assert rc == 0
    assert [_face_id_from_argv(call) for call in runner.calls] == [
        "ar023-llada",
        "ar026-dream",
    ]


def test_ut2704_only_unknown_id_exits_3_and_lists_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """UT-2704 (FR1): an unknown ``--only`` ID exits 3 and lists legal IDs."""
    runner = FakeRunner({}, {})
    rc = main(
        ["--only", "ar023-llada,does-not-exist"],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert rc == 3
    assert runner.calls == []
    for face_id in _EXPECTED_ORDER:
        assert face_id in combined


def test_ut2705_models_root_absent_not_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2705 (FR1): without ``--models-root`` no such flag reaches the child."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces)
    rc = main(
        ["--only", "ar023-llada"],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )

    assert rc == 0
    assert len(runner.calls) == 1
    assert "--models-root" not in runner.calls[0]


def test_ut2706_models_root_present_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2706 (FR1): an explicit ``--models-root`` is appended to every argv."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces)
    models_root = tmp_path / "models"
    rc = main(
        ["--only", "ar023-llada", "--models-root", str(models_root)],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )

    assert rc == 0
    assert runner.calls[0][-2:] == ["--models-root", str(models_root)]


def test_ut2707_quick_ar026_llada_omits_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2707 (FR1/FR3): ar026-llada quick argv has no ``--out``; formal untouched."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = next(f for f in faces if f.face_id == "ar026-llada")
    _write_json(face.formal_artifact, {"sentinel": "formal-llada"})
    before = _sha256(face.formal_artifact)

    runner = _make_runner(faces, quick=True)
    rc = main(
        ["--quick", "--only", "ar026-llada"],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )

    assert rc == 0
    assert len(runner.calls) == 1
    assert "--quick" in runner.calls[0]
    assert "--out" not in runner.calls[0]
    assert _sha256(face.formal_artifact) == before


def test_ut2708_quick_ar026_dream_uses_dream_quick_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2708 (FR1/FR3): ar026-dream quick ``--out`` targets only ``_dream_quick``."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = next(f for f in faces if f.face_id == "ar026-dream")
    _write_json(face.formal_artifact, {"sentinel": "formal-dream"})
    before = _sha256(face.formal_artifact)

    runner = _make_runner(faces, quick=True)
    rc = main(
        ["--quick", "--only", "ar026-dream"],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )

    argv = runner.calls[0]
    out_values = [argv[i + 1] for i, token in enumerate(argv) if token == "--out"]
    assert rc == 0
    assert len(out_values) == 1
    assert out_values[0].endswith("ar026_real_graph_bench_dream_quick.json")
    assert _sha256(face.formal_artifact) == before


def test_ut2709_quick_other_faces_use_quick_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2709 (FR1/FR3): ar023/ar024/ar025 quick argv is the full replacement."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces, quick=True)
    other_faces = [
        face_id for face_id in _EXPECTED_ORDER if face_id not in ("ar026-llada", "ar026-dream")
    ]
    for index, face_id in enumerate(other_faces):
        rc = main(
            ["--quick", "--only", face_id],
            probes=_make_probes(),
            runner=runner,
            run_dir=tmp_path / f"runs-{index}",
        )
        assert rc == 0, face_id

    assert len(runner.calls) == len(other_faces)
    expected_by_face = {face_id: _EXPECTED_QUICK[face_id][0] for face_id in other_faces}
    for call in runner.calls:
        face_id = _face_id_from_argv(call)
        assert call[3:] == list(expected_by_face[face_id]), face_id
        assert "--quick" in call
        assert "--out" not in call


def test_ut2741_only_empty_string_exits_3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """UT-2741 (FR1): an empty ``--only`` value is a parameter error (exit 3)."""
    runner = FakeRunner({}, {})
    rc = main(
        ["--only", ""],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert rc == 3
    assert runner.calls == []
    for face_id in _EXPECTED_ORDER:
        assert face_id in combined


# ---------------------------------------------------------------------------
# FR2: preflight (UT-2710..2716)
# ---------------------------------------------------------------------------


def test_ut2710_preflight_transformers_version(tmp_path: Path) -> None:
    """UT-2710 (FR2): transformers != 4.53.1 blocks with exit 3, no runner call."""
    runner = FakeRunner({}, {})
    base = tmp_path / "runs"
    rc = main(
        ["--only", "ar023-llada"],
        probes=_make_probes(transformers_version=lambda: "5.0.0"),
        runner=runner,
        run_dir=base,
    )
    assert rc == 3
    assert runner.calls == []
    guide = (base / _only_run_dir(base).name / "queue.log").read_text(encoding="utf-8")
    assert "RERUN_CHECKLIST" in guide


def test_ut2711_preflight_cuda_unavailable(tmp_path: Path) -> None:
    """UT-2711 (FR2): no CUDA device blocks with exit 3, no runner call."""
    runner = FakeRunner({}, {})
    rc = main(
        ["--only", "ar023-llada"],
        probes=_make_probes(cuda=lambda: (False, 0)),
        runner=runner,
        run_dir=tmp_path / "runs",
    )
    assert rc == 3
    assert runner.calls == []


def test_ut2712_preflight_vram_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2712 (FR2): formal 24 GiB and quick 16 GiB are inclusive boundaries."""
    faces = _patch_faces(tmp_path, monkeypatch)
    cases = [
        (False, 23.9, 3),
        (False, 24.0, 0),
        (True, 15.9, 3),
        (True, 16.0, 0),
    ]
    for index, (quick, vram, expected) in enumerate(cases):
        runner = _make_runner(faces, quick=quick)
        argv = ["--only", "ar023-llada"]
        if quick:
            argv.insert(0, "--quick")
        rc = main(
            argv,
            probes=_make_probes(total_vram_gb=lambda v=vram: v),
            runner=runner,
            run_dir=tmp_path / f"runs-{index}",
        )
        assert rc == expected, (quick, vram)
        if expected == 3:
            assert runner.calls == []


def test_ut2713_preflight_model_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2713 (FR2): missing config/safetensors for the family blocks with exit 3."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces)
    seen: list[tuple[Path, str]] = []

    def _model_files_ok(root: Path, family: str) -> bool:
        seen.append((Path(root), family))
        return family != "dream"

    rc = main(
        ["--only", "ar023-dream"],
        probes=_make_probes(model_files_ok=_model_files_ok),
        runner=runner,
        run_dir=tmp_path / "runs",
    )

    assert rc == 3
    assert runner.calls == []
    assert "dream" in {family for _root, family in seen}


def test_ut2714_preflight_probe_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2714 (FR2): a missing probe artifact blocks all seven faces (exit 3)."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces)
    rc = main(
        [],
        probes=_make_probes(probe_artifact_ok=lambda path: False),
        runner=runner,
        run_dir=tmp_path / "runs",
    )
    assert rc == 3
    assert runner.calls == []


def test_ut2715_preflight_gpu_silence_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2715 (FR2): util<=10% and used<=1024MiB pass; either excess blocks."""
    faces = _patch_faces(tmp_path, monkeypatch)
    cases = [
        ((10.0, 1024.0), 0),
        ((11.0, 0.0), 3),
        ((0.0, 1025.0), 3),
    ]
    for index, (state, expected) in enumerate(cases):
        runner = _make_runner(faces)
        rc = main(
            ["--only", "ar023-llada"],
            probes=_make_probes(gpu_state=lambda value=state: value),
            runner=runner,
            run_dir=tmp_path / f"runs-{index}",
        )
        assert rc == expected, state
        if expected == 3:
            assert runner.calls == []


def test_ut2716_allow_busy_overrides_gpu_silence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2716 (FR2): ``--allow-busy`` records the override in queue.log/json."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces)
    base = tmp_path / "runs"
    rc = main(
        ["--allow-busy", "--only", "ar023-llada"],
        probes=_make_probes(gpu_state=lambda: (99.0, 2048.0)),
        runner=runner,
        run_dir=base,
    )

    run_dir = _only_run_dir(base)
    queue = _read_queue_json(run_dir)
    log = (run_dir / "queue.log").read_text(encoding="utf-8")

    assert rc == 0
    assert len(runner.calls) == 1
    assert queue["allow_busy"] is True
    gpu_items = [
        item
        for item in queue["preflight"]
        if "gpu" in item["name"].lower() or "busy" in item["name"].lower()
    ]
    assert gpu_items
    assert any(item["detail"] for item in gpu_items)
    assert "allow" in log.lower()


# ---------------------------------------------------------------------------
# FR3/FR6: execution, logs, failure stop, overwrite protection (UT-2717..2721)
# ---------------------------------------------------------------------------


def test_ut2717_execution_logs_and_queue_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2717 (FR3/FR6): face log structure and complete queue.json face fields."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces)
    base = tmp_path / "runs"
    rc = main(
        ["--only", "ar023-llada"],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )

    run_dir = _only_run_dir(base)
    log_text = _face_log(run_dir, "ar023-llada")
    lines = [line for line in log_text.splitlines() if line.strip()]
    queue = _read_queue_json(run_dir)

    assert rc == 0
    assert lines[0].startswith("argv:")
    assert "scripts.ar023_cross_family_bench" in lines[0]
    assert lines[-1] == "exit_code=0"
    assert queue["run_id"] == run_dir.name
    assert queue["overall"] == "ok"
    assert queue["quick"] is False
    face_entry = queue["faces"][0]
    assert set(face_entry) >= {
        "id",
        "argv",
        "start",
        "end",
        "duration_s",
        "exit_code",
        "artifact",
        "validation",
        "outcome",
    }
    assert face_entry["id"] == "ar023-llada"
    assert face_entry["exit_code"] == 0
    assert face_entry["validation"] == "pass"
    assert face_entry["outcome"] == "ok"
    queue_log = (run_dir / "queue.log").read_text(encoding="utf-8")
    assert queue_log.strip()
    assert queue_log.splitlines()[0].startswith("[")


def test_ut2718_first_failure_stops_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2718 (FR3/FR5): first nonzero face exits 4; later faces are not_run."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces, failures={"ar023-llada": 1})
    base = tmp_path / "runs"
    rc = main(
        ["--only", "ar023-llada,ar026-dream"],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )

    queue = _read_queue_json(_only_run_dir(base))
    assert rc == 4
    assert len(runner.calls) == 1
    assert queue["overall"] == "failed"
    assert queue["faces"][0]["outcome"] == "failure"
    assert queue["faces"][1]["id"] == "ar026-dream"
    assert queue["faces"][1]["outcome"] == "not_run"


def test_ut2719_overwrite_protection_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2719 (FR5): an existing formal target without ``--force`` exits 4."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = faces[0]
    _write_json(face.formal_artifact, {"sentinel": "keep"})
    before = _sha256(face.formal_artifact)
    runner = _make_runner(faces)

    rc = main(
        ["--only", face.face_id],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )

    assert rc == 4
    assert runner.calls == []
    assert _sha256(face.formal_artifact) == before


def test_ut2720_force_with_empty_reason_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2720 (FR5): ``--force`` with an empty reason is a parameter error."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = faces[0]
    _write_json(face.formal_artifact, {"sentinel": "keep"})
    runner = _make_runner(faces)

    rc = main(
        ["--only", face.face_id, "--force", "--force-reason", ""],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )

    assert rc == 3
    assert runner.calls == []


def test_ut2721_force_with_reason_overwrites_and_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2721 (FR5): ``--force`` + non-empty reason overwrites and is audited."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = faces[0]
    _write_json(face.formal_artifact, {"sentinel": "before"})
    reason = "operator explicit recommit for AR027"
    runner = _make_runner(faces)
    base = tmp_path / "runs"

    rc = main(
        ["--only", face.face_id, "--force", "--force-reason", reason],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )

    queue = _read_queue_json(_only_run_dir(base))
    overwritten = json.loads(face.formal_artifact.read_text(encoding="utf-8"))
    assert rc == 0
    assert len(runner.calls) == 1
    assert queue["force_reason"] == reason
    assert overwritten["status"] == "ok"
    assert queue["faces"][0]["outcome"] == "ok"
    assert queue["faces"][0]["overwrite"] is True


def test_ut2758_force_without_existing_target_is_not_recorded_as_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2758 (FR5): force + reason on a fresh target is not an overwrite event."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = faces[0]
    base = tmp_path / "runs"

    rc = main(
        ["--only", face.face_id, "--force", "--force-reason", "precautionary"],
        probes=_make_probes(),
        runner=_make_runner(faces),
        run_dir=base,
    )

    queue = _read_queue_json(_only_run_dir(base))
    assert rc == 0
    assert queue["force_reason"] is None
    assert queue["faces"][0]["overwrite"] is False


# ---------------------------------------------------------------------------
# FR4: schema table and verdict recomputation (UT-2722..2727, UT-2731)
# ---------------------------------------------------------------------------


def test_ut2722_schema_table_ok_faces() -> None:
    """UT-2722 (FR4): the seven status=ok fixtures pass their schema table."""
    assert len(FACE_QUEUE) == 7
    for face in FACE_QUEUE:
        mismatches = validate_artifact(face, _ok_artifact(face))
        assert mismatches == [], (face.face_id, mismatches)


def test_ut2723_schema_table_degraded_blocked() -> None:
    """UT-2723 (FR4): degraded blocked uses ar023 schema (ar025 has no version)."""
    for face in FACE_QUEUE:
        mismatches = validate_artifact(face, _degraded_artifact(face))
        assert mismatches == [], (face.face_id, mismatches)
        if face.validator in ("ar023", "ar024", "ar026"):
            assert face.schemas["blocked_degraded"] == (
                "ar023_cross_family_bench",
                1,
            )
        if face.validator == "ar025_real":
            assert face.schemas["blocked_degraded"] == ("ar025_real_bench", None)


def test_ut2724_schema_table_exception_blocked() -> None:
    """UT-2724 (FR4): exception blocked uses each face's own schema."""
    expected = {
        "ar023": ("ar023_cross_family_bench", 1),
        "ar024": ("ar024_granularity_bench", 1),
        "ar025_real": ("ar025_real_bench", None),
        "ar026": ("ar026_real_graph_bench", 1),
    }
    for face in FACE_QUEUE:
        mismatches = validate_artifact(face, _exception_artifact(face))
        assert mismatches == [], (face.face_id, mismatches)
        assert face.schemas["blocked_exception"] == expected[face.validator]


def test_ut2725_tampered_verdict_exits_2_with_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2725 (FR4): a tampered h1/h2 record is a mismatch and exit 2 with diff."""
    face = FACE_QUEUE[0]
    artifact = _ok_artifact(face)
    artifact["verdicts"]["h2"]["numbers"]["speedup_p50_vs_baseline"] = 9.9999
    mismatches = validate_artifact(face, artifact)
    assert mismatches
    assert any("h2" in item.lower() for item in mismatches)
    assert any("speedup" in item.lower() for item in mismatches)

    tampered_h1 = _ok_artifact(face)
    tampered_h1["verdicts"]["h1"]["verdict"] = (
        "supported" if tampered_h1["verdicts"]["h1"]["verdict"] == "refuted" else "refuted"
    )
    assert validate_artifact(face, tampered_h1)

    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces, artifacts={face.face_id: artifact})
    base = tmp_path / "runs"
    rc = main(
        ["--only", face.face_id],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )

    run_dir = _only_run_dir(base)
    queue = _read_queue_json(run_dir)
    log_text = _face_log(run_dir, face.face_id).lower()
    assert rc == 2
    assert len(runner.calls) == 1
    assert queue["overall"] == "failed"
    assert queue["faces"][0]["validation"] == "mismatch"
    assert "h2" in log_text or "speedup" in log_text


def test_ut2726_committed_ar023_artifact_validates(tmp_path: Path) -> None:
    """UT-2726 (FR4): the committed ar023 artifact (copied to tmp) passes."""
    source = _REPO_ROOT / "results" / "optimization" / "ar023_llada_bench.json"
    assert source.is_file(), f"missing committed artifact: {source}"
    target = tmp_path / "ar023_llada_bench.json"
    target.write_bytes(source.read_bytes())
    artifact = json.loads(target.read_text(encoding="utf-8"))

    face = next(f for f in FACE_QUEUE if f.face_id == "ar023-llada")
    assert validate_artifact(face, artifact) == []


def test_ut2727_h1_mapping_recompute_matches_record() -> None:
    """UT-2727 (FR4): posthoc_confidence -> posthoc_overall_mean is the h1 input."""
    source = _REPO_ROOT / "results" / "optimization" / "ar023_llada_bench.json"
    artifact = json.loads(source.read_text(encoding="utf-8"))
    recorded = artifact["verdicts"]["h1"]

    mapped = [
        {
            "prompt_index": entry["prompt_index"],
            "posthoc_overall_mean": {
                mode: entry["posthoc_confidence"][mode]["overall_mean"] for mode in _AR023_MODES
            },
            "decode_length": entry["decode_length"],
            "block_exact": entry["block_exact"],
        }
        for entry in artifact["quality"]["per_prompt"]
    ]
    recomputed = ar023_bench.h1_quality_verdict({"per_prompt": mapped})

    assert recomputed["verdict"] == recorded["verdict"]
    assert recomputed["numbers"]["ratio"] == recorded["numbers"]["ratio"]
    assert recomputed["numbers"]["gap"] == recorded["numbers"]["gap"]
    assert recomputed["numbers"]["none_posthoc_mean"] is False


def test_ut2731_structure_self_consistency() -> None:
    """UT-2731 (FR4): missing reproduction.ok / blocked_reasons is a mismatch."""
    face = FACE_QUEUE[0]
    broken = copy.deepcopy(_ok_artifact(face))
    broken["reproduction"].pop("ok")
    mismatches = validate_artifact(face, broken)
    assert mismatches
    assert any("reproduction" in item.lower() for item in mismatches)

    ar025 = next(f for f in FACE_QUEUE if f.face_id == "ar025-real-llada")
    broken_025 = copy.deepcopy(_ar025_artifact(ar025.family))
    broken_025["verdicts"].pop("blocked_reasons")
    mismatches_025 = validate_artifact(ar025, broken_025)
    assert mismatches_025
    assert any("blocked_reasons" in item.lower() for item in mismatches_025)


# ---------------------------------------------------------------------------
# FR5: blocked classification (UT-2728..2730, UT-2732, UT-2740)
# ---------------------------------------------------------------------------


def test_ut2728_classify_degraded_blocked() -> None:
    """UT-2728 (FR5): degraded notes (ar023/024/026) and ar025 prefix are legal."""
    for face in FACE_QUEUE:
        assert classify_blocked(face, _degraded_artifact(face)) == "legit_degraded", face.face_id


def test_ut2729_classify_exception_blocked() -> None:
    """UT-2729 (FR5): exception blocked artifacts classify as failure."""
    for face in FACE_QUEUE:
        assert classify_blocked(face, _exception_artifact(face)) == "failure", face.face_id

    ar023_face = FACE_QUEUE[0]
    missing_note = copy.deepcopy(_exception_artifact(ar023_face))
    missing_note.pop("blocked_reason", None)
    assert classify_blocked(ar023_face, missing_note) == "failure"


def test_ut2730_classify_ar025_reproduction() -> None:
    """UT-2730 (FR5): ar025 honest negative vs reproduction anomaly."""
    face = next(f for f in FACE_QUEUE if f.face_id == "ar025-real-llada")

    negative = {
        "schema": "ar025_real_bench",
        "family": face.family,
        "status": "blocked",
        "verdicts": _ar025_verdicts(overall="blocked", reproduction_ok=True),
    }
    assert classify_blocked(face, negative) == "legit_negative"

    anomaly = copy.deepcopy(negative)
    anomaly["verdicts"]["reproduction"]["ok"] = False
    assert classify_blocked(face, anomaly) == "failure"

    ok_face = {
        "schema": "ar025_real_bench",
        "family": face.family,
        "status": "ok",
        "verdicts": _ar025_verdicts(overall="supported", reproduction_ok=True),
    }
    assert classify_blocked(face, ok_face) is None


def test_ut2732_status_ok_with_reproduction_failure_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2732 (FR5): status ok + reproduction.ok False exits 4, not 2."""
    faces = _patch_faces(tmp_path, monkeypatch)
    broken = _ok_artifact(faces[0])
    broken["reproduction"]["ok"] = False
    assert classify_blocked(faces[0], broken) == "failure"

    artifacts = {
        faces[0].face_id: broken,
        "ar026-dream": _ok_artifact(next(f for f in faces if f.face_id == "ar026-dream")),
    }
    runner = _make_runner(faces, artifacts=artifacts)
    base = tmp_path / "runs"
    rc = main(
        ["--only", f"{faces[0].face_id},ar026-dream"],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )

    queue = _read_queue_json(_only_run_dir(base))
    assert rc == 4
    assert len(runner.calls) == 1
    assert queue["faces"][0]["outcome"] == "failure"
    assert queue["faces"][1]["outcome"] == "not_run"


def test_ut2740_blocked_fallback_is_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2740 (FR5): a blocked artifact with a non-degrade note fails the queue."""
    faces = _patch_faces(tmp_path, monkeypatch)
    bad = _degraded_artifact(faces[0])
    bad["reproduction"]["note"] = "something else entirely"
    assert classify_blocked(faces[0], bad) == "failure"

    artifacts = {
        faces[0].face_id: bad,
        "ar026-dream": _ok_artifact(next(f for f in faces if f.face_id == "ar026-dream")),
    }
    runner = _make_runner(faces, artifacts=artifacts)
    base = tmp_path / "runs"
    rc = main(
        ["--only", f"{faces[0].face_id},ar026-dream"],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )

    queue = _read_queue_json(_only_run_dir(base))
    assert rc == 4
    assert len(runner.calls) == 1
    assert queue["faces"][0]["outcome"] == "failure"
    assert queue["faces"][1]["outcome"] == "not_run"


# ---------------------------------------------------------------------------
# FR3/FR5/FR6: timeout, startup, missing/bad JSON, no-write, quick, isolation
# ---------------------------------------------------------------------------


def test_ut2733_timeout_exits_4_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2733 (FR3): a runner timeout exits 4 and stops the queue."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(
        faces,
        raises={
            "ar023-llada": subprocess.TimeoutExpired(
                cmd=["python", "-m"], timeout=0.001, output=b"partial-timeout-output\n"
            )
        },
    )
    base = tmp_path / "runs"
    rc = main(
        ["--only", "ar023-llada,ar026-dream", "--timeout-hours", "0.001"],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )

    run_dir = _only_run_dir(base)
    log_text = _face_log(run_dir, "ar023-llada").lower()
    assert rc == 4
    assert len(runner.calls) == 1
    assert "timeout" in log_text or "timed out" in log_text
    assert "partial-timeout-output" in log_text


def test_ut2734_runner_oserror_exits_4_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2734 (FR3): a runner startup OSError exits 4 and is logged."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(
        faces,
        raises={"ar023-llada": OSError("cannot spawn interpreter")},
    )
    base = tmp_path / "runs"
    rc = main(
        ["--only", "ar023-llada,ar026-dream"],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )

    run_dir = _only_run_dir(base)
    log_text = _face_log(run_dir, "ar023-llada")
    assert rc == 4
    assert len(runner.calls) == 1
    assert "OSError" in log_text or "cannot spawn interpreter" in log_text


def test_ut2735_missing_or_bad_json_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2735 (FR4): a missing artifact or invalid JSON exits 2."""
    faces = _patch_faces(tmp_path, monkeypatch)

    missing_runner = _make_runner(faces, write=False)
    rc_missing = main(
        ["--only", "ar023-llada"],
        probes=_make_probes(),
        runner=missing_runner,
        run_dir=tmp_path / "runs-missing",
    )
    assert rc_missing == 2
    assert len(missing_runner.calls) == 1

    bad_runner = _make_runner(faces, artifacts={"ar023-llada": "{not valid json"})
    rc_bad = main(
        ["--only", "ar023-llada"],
        probes=_make_probes(),
        runner=bad_runner,
        run_dir=tmp_path / "runs-bad",
    )
    assert rc_bad == 2
    assert len(bad_runner.calls) == 1


def test_ut2736_validation_does_not_write_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2736 (FR4): validation never mutates the artifact (sha256 unchanged)."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = faces[0]
    _write_json(face.formal_artifact, _ok_artifact(face))
    before = _sha256(face.formal_artifact)
    runner = _make_runner(faces, write=False)

    rc = main(
        ["--only", face.face_id, "--force", "--force-reason", "sha256 audit"],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )

    assert rc == 0
    assert _sha256(face.formal_artifact) == before


def test_ut2738_quick_not_killed_by_formal_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2738 (FR5): quick protection keys on the quick target, never formal."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = next(f for f in faces if f.face_id == "ar026-llada")
    _write_json(face.formal_artifact, {"sentinel": "formal remains"})
    before = _sha256(face.formal_artifact)
    runner = _make_runner(faces, quick=True)

    rc = main(
        ["--quick", "--only", face.face_id],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )

    assert rc == 0
    assert len(runner.calls) == 1
    assert _sha256(face.formal_artifact) == before
    assert face.quick_artifact.is_file()


def test_ut2739_reason_without_force_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2739 (FR5): a reason without ``--force`` is refused with exit 4."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = faces[0]
    _write_json(face.formal_artifact, {"sentinel": "keep"})
    before = _sha256(face.formal_artifact)
    runner = _make_runner(faces)

    rc = main(
        ["--only", face.face_id, "--force-reason", "reason only"],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )

    assert rc == 4
    assert runner.calls == []
    assert _sha256(face.formal_artifact) == before


def test_ut2737_run_directory_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2737 (FR6): two runs create two distinct, non-overwriting UTC dirs."""
    faces = _patch_faces(tmp_path, monkeypatch)
    base = tmp_path / "runs"

    rc_first = main(
        ["--only", "ar023-llada"],
        probes=_make_probes(),
        runner=_make_runner(faces),
        run_dir=base,
    )
    first_dirs = _run_dirs(base)

    rc_second = main(
        ["--only", "ar023-dream"],
        probes=_make_probes(),
        runner=_make_runner(faces),
        run_dir=base,
    )
    second_dirs = _run_dirs(base)

    assert rc_first == 0
    assert rc_second == 0
    assert len(first_dirs) == 1
    assert len(second_dirs) == 2
    assert second_dirs[0] != second_dirs[1]
    for run_dir in second_dirs:
        assert (run_dir / "queue.json").is_file()
        assert (run_dir / "queue.log").is_file()
    first_queue = _read_queue_json(first_dirs[0])
    assert first_queue["faces"][0]["id"] == "ar023-llada"


# ---------------------------------------------------------------------------
# FR1/FR6: invalid timeout and ar025 exact structure (UT-2742, UT-2743)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "-1", "-0.5"])
def test_ut2742_invalid_timeout_hours_exits_3(tmp_path: Path, value: str) -> None:
    """UT-2742 (FR6): ``--timeout-hours <= 0`` is a parameter error (exit 3)."""
    runner = FakeRunner({}, {})
    rc = main(
        ["--only", "ar023-llada", "--timeout-hours", value],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )
    assert rc == 3
    assert runner.calls == []


def test_ut2743_ar025_exact_structure_keys() -> None:
    """UT-2743 (FR4): ar025 exact key set; a missing H4 key is a mismatch."""
    face = next(f for f in FACE_QUEUE if f.face_id == "ar025-real-llada")
    artifact = _ar025_artifact(face.family)

    assert set(artifact["verdicts"]) == {
        "H1_token_parity",
        "H2_speed",
        "delivery_line",
        "H4_ledger_strictly_smaller",
        "canary_structural_eager_fallback",
        "overall",
        "blocked_reasons",
        "reproduction",
    }
    assert validate_artifact(face, artifact) == []

    broken = copy.deepcopy(artifact)
    broken["verdicts"].pop("H4_ledger_strictly_smaller")
    mismatches = validate_artifact(face, broken)
    assert mismatches
    assert any("H4_ledger_strictly_smaller" in item for item in mismatches)


# ---------------------------------------------------------------------------
# T010 review regressions (UT-2744..2747)
# ---------------------------------------------------------------------------


def test_ut2744_ok_without_verdicts_is_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2744 (FR4): a status=ok artifact without verdicts cannot pass."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = faces[0]
    hollow = {"schema": "ar023_cross_family_bench", "schema_version": 1, "status": "ok"}
    mismatches = validate_artifact(face, hollow)
    assert mismatches
    assert any("verdicts" in item.lower() for item in mismatches)

    runner = _make_runner(faces, artifacts={face.face_id: hollow})
    base = tmp_path / "runs"
    rc = main(
        ["--only", face.face_id],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )
    queue = _read_queue_json(_only_run_dir(base))
    assert rc == 2
    assert queue["faces"][0]["validation"] == "mismatch"


def test_ut2745_allow_busy_cannot_bypass_other_gates(tmp_path: Path) -> None:
    """UT-2745 (FR2): --allow-busy never overrides non-GPU preflight failures."""
    runner = FakeRunner({}, {})
    rc = main(
        ["--only", "ar023-llada", "--allow-busy"],
        probes=_make_probes(transformers_version=lambda: "5.0.0", gpu_state=lambda: (99.0, 2048.0)),
        runner=runner,
        run_dir=tmp_path / "runs",
    )
    assert rc == 3
    assert runner.calls == []


def test_ut2746_face_log_first_line_is_full_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2746 (FR3): the face log header is the exact executed argv."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces)
    base = tmp_path / "runs"
    rc = main(
        ["--only", "ar023-llada"],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )
    run_dir = _only_run_dir(base)
    lines = _face_log(run_dir, "ar023-llada").splitlines()
    assert rc == 0
    assert lines[0] == "argv: " + " ".join(runner.calls[0])


def test_ut2747_list_quick_shows_quick_argv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """UT-2747 (FR1): --list --quick previews the quick argv/artifacts."""
    rc = main(
        ["--list", "--quick"],
        probes=_make_probes(),
        runner=FakeRunner({}, {}),
        run_dir=tmp_path / "runs",
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "ar026_real_graph_bench_dream_quick.json" in captured.out
    assert "--quick" in captured.out


# ---------------------------------------------------------------------------
# T010 review round-2 regressions (UT-2748..2751)
# ---------------------------------------------------------------------------


def test_ut2748_invalid_or_missing_status_is_mismatch() -> None:
    """UT-2748 (FR4): status must be ok|blocked; anything else is a mismatch."""
    face = FACE_QUEUE[0]
    payload = _ok_artifact(face)
    payload["status"] = "pending_t006"
    assert validate_artifact(face, payload)

    missing = _ok_artifact(face)
    missing.pop("status")
    assert validate_artifact(face, missing)


def test_ut2749_non_object_json_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2749 (FR4): a JSON array is not an artifact; exit 2, validation=missing."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = faces[0]
    runner = _make_runner(faces, artifacts={face.face_id: []})
    base = tmp_path / "runs"
    rc = main(
        ["--only", face.face_id],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )
    queue = _read_queue_json(_only_run_dir(base))
    assert rc == 2
    assert queue["faces"][0]["validation"] == "missing"


def test_ut2750_h4_non_dict_is_mismatch() -> None:
    """UT-2750 (FR4): every verdict entry must be a dict (h4 included)."""
    face = FACE_QUEUE[0]
    payload = _ok_artifact(face)
    payload["verdicts"]["h4"] = "totally-not-a-dict"
    mismatches = validate_artifact(face, payload)
    assert mismatches
    assert any("h4" in item for item in mismatches)


def test_ut2751_ar025_extra_key_is_mismatch() -> None:
    """UT-2751 (FR4): the ar025 verdict block has an exact key set."""
    face = next(f for f in FACE_QUEUE if f.face_id == "ar025-real-llada")
    payload = _ar025_artifact(face.family)
    payload["verdicts"]["surprise"] = {"pass": True}
    mismatches = validate_artifact(face, payload)
    assert mismatches
    assert any("unexpected keys" in item for item in mismatches)


# ---------------------------------------------------------------------------
# T010 review round-3 regressions (UT-2752..2753)
# ---------------------------------------------------------------------------


def test_ut2752_all_seven_green_exit_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2752 (FR6): a full all-green queue exits 0 with overall=ok and 7 faces."""
    faces = _patch_faces(tmp_path, monkeypatch)
    runner = _make_runner(faces)
    base = tmp_path / "runs"
    rc = main([], probes=_make_probes(), runner=runner, run_dir=base)

    queue = _read_queue_json(_only_run_dir(base))
    assert rc == 0
    assert len(runner.calls) == 7
    assert queue["overall"] == "ok"
    assert [entry["outcome"] for entry in queue["faces"]] == ["ok"] * 7


def test_ut2753_mixed_blocked_continues_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2753 (FR5): a legal probe-degraded face keeps the queue going."""
    faces = _patch_faces(tmp_path, monkeypatch)
    degraded = _degraded_artifact(faces[0])
    ok_second = _ok_artifact(faces[1])
    runner = _make_runner(
        faces,
        artifacts={faces[0].face_id: degraded, faces[1].face_id: ok_second},
    )
    base = tmp_path / "runs"
    rc = main(
        ["--only", f"{faces[0].face_id},{faces[1].face_id}"],
        probes=_make_probes(),
        runner=runner,
        run_dir=base,
    )

    queue = _read_queue_json(_only_run_dir(base))
    assert rc == 0
    assert len(runner.calls) == 2
    assert queue["overall"] == "blocked"
    assert queue["faces"][0]["outcome"] == "blocked"
    assert queue["faces"][1]["outcome"] == "ok"


# ---------------------------------------------------------------------------
# T010 review round-4 regressions (UT-2754..2757)
# ---------------------------------------------------------------------------


def test_ut2754_unmeasurable_gpu_not_overridable(tmp_path: Path) -> None:
    """UT-2754 (FR2): --allow-busy cannot override an unavailable nvidia-smi."""
    runner = FakeRunner({}, {})
    base = tmp_path / "runs"
    rc = main(
        ["--only", "ar023-llada", "--allow-busy"],
        probes=_make_probes(gpu_state=lambda: None),
        runner=runner,
        run_dir=base,
    )
    assert rc == 3
    assert runner.calls == []
    log_text = (base / _only_run_dir(base).name / "queue.log").read_text(encoding="utf-8")
    assert "not overridable" in log_text


def test_ut2755_nested_container_malformed_is_mismatch() -> None:
    """UT-2755 (FR4): malformed nested containers are mismatches, not crashes."""
    face = FACE_QUEUE[0]

    bad_verdicts = _ok_artifact(face)
    bad_verdicts["verdicts"] = ["x"]
    assert validate_artifact(face, bad_verdicts)

    bad_quality = _ok_artifact(face)
    bad_quality["quality"] = ["x"]
    assert validate_artifact(face, bad_quality)

    bad_prompt = _ok_artifact(face)
    bad_prompt["quality"]["per_prompt"] = [1, 2]
    assert validate_artifact(face, bad_prompt)

    bad_confidence = _ok_artifact(face)
    bad_confidence["quality"]["per_prompt"][0]["posthoc_confidence"] = 5
    assert validate_artifact(face, bad_confidence)

    bad_scalar = _ok_artifact(face)
    bad_scalar["quality"]["per_prompt"][0]["posthoc_confidence"]["baseline"]["overall_mean"] = "bad"
    assert validate_artifact(face, bad_scalar)

    ar026_face = next(f for f in FACE_QUEUE if f.face_id == "ar026-llada")
    bad_parity = _ar026_artifact(ar026_face.family)
    bad_parity["verdicts"]["h1_graph_parity"]["modes"] = {"graph-fixed": 5}
    assert validate_artifact(ar026_face, bad_parity)

    bad_captures = _ar026_artifact(ar026_face.family)
    bad_captures["verdicts"]["h1_graph_parity"]["modes"]["graph-fixed"]["numbers"][
        "captures"
    ] = "bad"
    assert validate_artifact(ar026_face, bad_captures)


def test_ut2756_probe_exception_blocks_with_exit_3(tmp_path: Path) -> None:
    """UT-2756 (FR2): a raising probe becomes a preflight FAIL, never a crash."""
    runner = FakeRunner({}, {})

    def _boom(root: Path, family: str) -> bool:
        raise RuntimeError("checkpoint introspection failed")

    rc = main(
        ["--only", "ar023-llada"],
        probes=_make_probes(model_files_ok=_boom),
        runner=runner,
        run_dir=tmp_path / "runs",
    )
    assert rc == 3
    assert runner.calls == []


def test_ut2757_cuda_unavailable_vram_probe_exception(tmp_path: Path) -> None:
    """UT-2757 (FR2): CPU-only torch (vram probe raises) blocks with exit 3."""
    runner = FakeRunner({}, {})

    def _vram_boom() -> float:
        raise RuntimeError("CUDA error: no kernel image is available")

    rc = main(
        ["--only", "ar023-llada"],
        probes=_make_probes(cuda=lambda: (False, 0), total_vram_gb=_vram_boom),
        runner=runner,
        run_dir=tmp_path / "runs",
    )
    assert rc == 3
    assert runner.calls == []


def test_ut2759_malformed_probe_json_falls_back_to_default_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2759 (FR2): malformed probe JSON shape falls back, never crashes."""
    probe = tmp_path / "probe.json"
    monkeypatch.setattr(formal_queue, "DEFAULT_PROBE_ARTIFACT", probe)
    monkeypatch.setattr(formal_queue, "DEFAULT_MODELS_ROOT", tmp_path / "fallback")

    for payload in ("[1, 2]", '{"meta": [1]}', '{"meta": {"models_root": 5}}', "not-json"):
        probe.write_text(payload, encoding="utf-8")
        assert formal_queue._resolve_models_root(None) == tmp_path / "fallback", payload

    probe.write_text('{"meta": {"models_root": "D:\\\\weights"}}', encoding="utf-8")
    assert formal_queue._resolve_models_root(None) == Path("D:\\weights")


def test_ut2760_degraded_requires_boolean_reproduction_ok() -> None:
    """UT-2760 (FR4): a degraded blocked artifact must carry a boolean reproduction.ok."""
    face = next(f for f in FACE_QUEUE if f.face_id == "ar023-llada")
    degraded = _degraded_artifact(face)
    assert validate_artifact(face, degraded) == []

    missing_ok = copy.deepcopy(degraded)
    missing_ok["reproduction"].pop("ok", None)
    assert validate_artifact(face, missing_ok)

    non_bool = copy.deepcopy(degraded)
    non_bool["reproduction"]["ok"] = "false"
    assert validate_artifact(face, non_bool)


def test_ut2761_ar025_negative_requires_blocked_overall_and_true_reproduction() -> None:
    """UT-2761 (FR5): ar025 honest negative needs overall=blocked + reproduction.ok is True."""
    face = next(f for f in FACE_QUEUE if f.face_id == "ar025-real-llada")
    negative = {
        "schema": "ar025_real_bench",
        "family": face.family,
        "status": "blocked",
        "verdicts": _ar025_verdicts(overall="blocked", reproduction_ok=True),
    }
    assert classify_blocked(face, negative) == "legit_negative"

    supported = copy.deepcopy(negative)
    supported["verdicts"]["overall"] = "supported"
    supported["verdicts"]["blocked_reasons"] = []
    assert classify_blocked(face, supported) == "failure"

    missing_ok = copy.deepcopy(negative)
    missing_ok["verdicts"]["reproduction"].pop("ok", None)
    assert classify_blocked(face, missing_ok) == "failure"


def test_ut2762_ar025_supported_overall_under_blocked_status_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2762 (FR5): blocked status with overall=supported is a failure (exit 4)."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = next(f for f in faces if f.face_id == "ar025-real-llada")
    inconsistent = {
        "schema": "ar025_real_bench",
        "family": face.family,
        "status": "blocked",
        "verdicts": _ar025_verdicts(overall="supported", reproduction_ok=True),
        "meta": {},
    }
    runner = _make_runner(faces, artifacts={face.face_id: inconsistent})
    rc = main(
        ["--only", face.face_id],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )
    assert rc == 4


def test_ut2763_degraded_reason_with_verdicts_is_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2763 (FR5): a degraded reason plus verdicts is contradictory and fails."""
    faces = _patch_faces(tmp_path, monkeypatch)
    face = next(f for f in faces if f.face_id == "ar025-real-llada")
    contradictory = {
        "schema": "ar025_real_bench",
        "family": face.family,
        "status": "blocked",
        "blocked_reason": "family degraded per probe conclusion: probe says so",
        "verdicts": _ar025_verdicts(overall="supported", reproduction_ok=True),
        "meta": {},
    }
    assert classify_blocked(face, contradictory) == "failure"
    assert validate_artifact(face, contradictory)

    runner = _make_runner(faces, artifacts={face.face_id: contradictory})
    rc = main(
        ["--only", face.face_id],
        probes=_make_probes(),
        runner=runner,
        run_dir=tmp_path / "runs",
    )
    assert rc == 4


def test_ut2764_ar026_top_level_engagement_tamper_is_mismatch() -> None:
    """UT-2764 (FR4): h1 recomputation reads top-level modes[*].engagement."""
    face = next(f for f in FACE_QUEUE if f.face_id == "ar026-llada")
    artifact = _ar026_artifact(face.family)
    assert validate_artifact(face, artifact) == []

    tampered = copy.deepcopy(artifact)
    tampered["modes"]["graph-fixed"]["engagement"]["captures"] = 0
    assert validate_artifact(face, tampered)

    dropped = copy.deepcopy(artifact)
    dropped["modes"] = {}
    assert validate_artifact(face, dropped)
