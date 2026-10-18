"""Red-phase tests for AR026: the probe production-graph-face contract (UT-2605).

Contract: ``specs/changes/AR026-real-graph-e2e/design.md`` §4.2.2 point 4 (probe
graph face productionization) and §6.1 UT-2605 — the ``ar023_family_probe``
``graph`` face must stop being a raw ad-hoc CUDA-graph capture and become the
PRODUCTION face (MFF construction + family-sampler mini speculative window
driving the capture), while the old raw capture is demoted to a ``graph_raw``
DIAGNOSTIC face.  ``graph_enabled`` conclusions stay keyed ONLY on the
production ``graph`` face (§4.2.2: "``graph_enabled`` 结论只依据生产面"), and
the ``graph_raw`` face joins ``CUDA_FACE_NAMES`` so the fresh-subprocess
isolation wrapper (§4.2.2: the tuple drives the subprocess wrapping) covers it.

Covered points:

* ``graph_raw`` is registered in ``scripts.ar023_family_probe.CUDA_FACE_NAMES``
  alongside ``graph`` (the subprocess-isolation driver tuple);
* ``build_family_faces`` returns a face registry containing BOTH ``graph`` and
  ``graph_raw``, both callable (the factories only close over ctx at
  construction time, so a bare namespace stand-in ctx is safe);
* ``derive_conclusions`` is keyed ONLY on the production ``graph`` face:
  (a) production ``graph`` ok=True + ``headroom_mb=2048.0`` → ``graph_enabled``
  True even when ``graph_raw`` is ok=False; (b) production ``graph`` ok=False →
  ``graph_enabled`` False even when ``graph_raw`` is ok=True with headroom
  2048.0; (c) production ``graph`` ok=True but ``headroom_mb=512.0`` →
  ``graph_enabled`` False (the strict ``> 1024.0`` MB threshold);
* the production driver ``_drive_production_graph_capture`` does the CUDA check
  FIRST (before any model load): on the CPU face it returns ``{"ok": False,
  error mentioning CUDA, "capture_delta_mb": None, "headroom_mb": None,
  "diagnostics": []}`` and NEVER loads a model (a monkeypatched
  ``_load_probe_model`` raising AssertionError proves the early out);
* ``make_graph_face`` is a thin wrapper ROUTING to the production driver: a
  monkeypatched ``_drive_production_graph_capture`` spy's canned return dict
  (canary key included) passes through EXACTLY, and the spy is called exactly
  once;
* ``make_graph_raw_face`` (the demoted raw-capture diagnostic face) keeps the
  CPU early-out contract: ``ok=False`` with a CUDA-mentioning error, no model
  load.

Red state (why the tests fail today) — the Green implementer will restructure
the probe so that ``make_graph_face(ctx)`` returns
``_drive_production_graph_capture`` (a NEW module-level production driver:
CUDA check FIRST, then MFF construction + family-sampler mini speculative
window), the old raw-capture body moves to a NEW ``make_graph_raw_face(ctx)``,
``build_family_faces`` gains ``"graph_raw"`` and ``CUDA_FACE_NAMES`` gains
``"graph_raw"`` — none of these names exist yet:

* ``scripts.ar023_family_probe`` has NO ``_drive_production_graph_capture`` and
  NO ``make_graph_raw_face`` (AttributeError);
* ``CUDA_FACE_NAMES`` is ``("bf16", "int8", "smoke", "graph", "prompt")`` — no
  ``"graph_raw"`` entry (assertion failure);
* ``build_family_faces`` returns the seven-face registry without
  ``"graph_raw"`` (assertion failure);
* ``make_graph_face`` is currently the raw ad-hoc capture — monkeypatching the
  missing driver name raises AttributeError before the face ever runs.

Honest classification: the ``derive_conclusions`` test (production-face
keying + strict ``> 1024.0`` threshold) is a REGRESSION PIN — the old code
already keys ``graph_enabled`` only on ``"graph"`` with the strict threshold
(``_GRAPH_HEADROOM_THRESHOLD_MB = 1024.0``), so it passes on the old probe and
must keep passing after the restructure.  Every other test is Red on the
missing-name / missing-registration faces.

All faces are pure CPU + monkeypatch (``torch.cuda.is_available`` forced False,
``_load_probe_model`` guarded); no network, no model loading, no CUDA.  The
import mirrors ``scripts/ar024_granularity_bench.py`` (the AR024 module-level
``import scripts.ar023_...`` precedent — ``scripts`` is a namespace package
resolved from the repo root, which this test inserts on ``sys.path`` like
``tests/test_ar026_capture_safety.py``).

UT-2606 (appended below) — the AR026 four-mode real-graph bench contract
========================================================================

Contract: ``specs/changes/AR026-real-graph-e2e/design.md`` §4.2.2 point 5 (the
four-mode real-checkpoint bench: single-knob discipline, geometry-inheritance
note obligation, VRAM headroom face, blocked semantics) and §6.1 UT-2606.  The
Green implementer will create ``scripts/ar026_real_graph_bench.py`` importing
``scripts.ar023_cross_family_bench`` at module level (the AR024
``import ... as _ar023`` precedent) with EXACTLY this pinned API surface:

* ``MODES = ("baseline", "folded-batched", "graph-fixed", "graph-adaptive")``;
* ``MODE_KNOBS: dict[str, dict]`` — the frozen single-dial knob matrix
  (``baseline`` = official recipe off; ``folded-batched`` = the AR023 eager
  anchor; the two graph modes differ from the anchor ONLY in graph knobs from
  {``use_cuda_graph``, ``graph_capacity_ratio``, ``graph_runner_pool_size``,
  ``graph_adaptive_capacity``} with the frozen 0.5 / 4 / True / False values);
* ``SCHEMA_NAME = "ar026_real_graph_bench"`` and ``DEFAULT_OUTPUT_PATH =
  Path("results/optimization/ar026_real_graph_bench.json")``;
* ``quick_output_path()`` / ``resolve_output_path(family, quick, out_override)``
  — same semantics as ar023's (explicit override wins; quick →
  ``ar026_real_graph_bench_quick.json``; else the formal path);
* ``parse_args(argv)`` — ``--family`` choices ("llada", "dream") default
  "llada" (dream ALLOWED — unlike AR024's llada-only guard; the formal
  commitment is llada via meta), ``--reps`` type int default 30, ``--quick``
  store_true, ``--out`` optional override; no HF env args (minimal CLI);
* ``GEOMETRY_INHERITANCE_NOTE`` — a non-empty module constant string that
  artifacts must embed in meta (the EVR >=4-shape floor explicitly waived WITH
  reason: 16GB-card bf16-8B + graph-pool VRAM headroom is the binding
  constraint; AR023 anchor geometry inherited; formal geometry expansion
  listed as follow-up) — the waiving reason must be explicit, never silent;
* assign-form re-exports (identity pins, the pyflakes-safe AR024 pattern):
  ``h1_quality_verdict = _ar023.h1_quality_verdict``, same for
  ``h2_speed_verdict`` / ``h4_fullstack_verdict`` / ``reproduction_verdict`` /
  ``mark_all_blocked`` / ``build_blocked_artifact`` / ``write_results``, and
  the constants ``SEED = _ar023.SEED`` / ``REPS_DEFAULT = _ar023.REPS_DEFAULT``
  — re-implementation drift is a contract failure;
* ``build_artifact(...)`` — its own assembler whose output dict carries
  ``artifact["schema"] == SCHEMA_NAME`` (the AR024 schema-name override
  pattern) and the same key set as ar023's build_artifact (family / status /
  modes / quality / verdicts / reproduction / meta / schema_version).

Red state (why the appended tests fail today): ``scripts/ar026_real_graph_bench.py``
does NOT exist — ``python -c "import scripts.ar026_real_graph_bench"`` raises
ModuleNotFoundError.  Each UT-2606 test therefore imports the bench module
LAZILY inside its own test body: a module-level bench import would fail at
test-module import time and break the UT-2605 tests above (collection of the
whole file).  With the lazy pattern the UT-2605 tests keep collecting and
passing while the script is missing; each UT-2606 test then fails on its own
ModuleNotFoundError.  All UT-2606 faces are pure stdlib-level contract checks:
no CUDA, no model loading, no network, no monkeypatching.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.ar023_family_probe as probe  # noqa: E402 — post-syspath insert


def _force_cpu_face(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the CPU face and forbid any model load.

    ``torch.cuda.is_available`` is monkeypatched False and the probe's
    ``_load_probe_model`` is monkeypatched to raise AssertionError — the
    contract under test is that the CPU early out happens BEFORE any model
    load, so reaching the loader is a failure, not a skip.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _must_not_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must not load on the CPU face")

    monkeypatch.setattr(probe, "_load_probe_model", _must_not_load)


def _benign_family_result(
    graph_face: dict[str, Any],
    graph_raw_face: dict[str, Any],
) -> dict[str, Any]:
    """A minimal valid seven(+raw)-face result dict for ``derive_conclusions``.

    ``derive_conclusions`` reads the ``load`` / ``bf16`` / ``int8`` / ``smoke``
    / ``graph`` keys (the download/prompt faces are never consulted), so benign
    ``ok=True`` values avoid any KeyError and keep the assertions focused on
    the graph-face keying.
    """
    return {
        "download": {"ok": True},
        "load": {"ok": True},
        "bf16": {"ok": True},
        "int8": {"ok": True},
        "smoke": {"ok": True},
        "graph": graph_face,
        "graph_raw": graph_raw_face,
        "prompt": {"ok": True},
    }


# ---------------------------------------------------------------------------
# UT-2605: subprocess-isolation tuple + face registry registrations
# ---------------------------------------------------------------------------


def test_graph_raw_registered_in_cuda_subprocess_tuple() -> None:
    """UT-2605: both ``graph`` and ``graph_raw`` drive the subprocess wrapper.

    ``CUDA_FACE_NAMES`` is the tuple that ``wrap_cuda_faces_with_subprocess``
    keys on — the production face and the demoted raw diagnostic face must BOTH
    run in fresh subprocesses (a failed capture poisons the process-wide CUDA
    allocator state, [T001-2]).
    """
    assert "graph" in probe.CUDA_FACE_NAMES
    assert "graph_raw" in probe.CUDA_FACE_NAMES


def test_face_registry_contains_graph_and_graph_raw() -> None:
    """UT-2605: ``build_family_faces`` registers both graph faces, callable.

    The face factories only close over ctx at construction time (the make_*
    signatures take any ctx and never run the face body here), so a bare
    ``SimpleNamespace`` stand-in ctx is sufficient — this test constructs the
    registry without executing any face.
    """
    faces = probe.build_family_faces(SimpleNamespace())

    assert isinstance(faces, dict)
    assert "graph" in faces
    assert "graph_raw" in faces
    assert callable(faces["graph"])
    assert callable(faces["graph_raw"])


# ---------------------------------------------------------------------------
# UT-2605: conclusions keyed ONLY on the production graph face
# ---------------------------------------------------------------------------


def test_conclusions_keyed_only_on_production_graph_face() -> None:
    """UT-2605: ``graph_enabled`` derives ONLY from the production ``graph`` face.

    (a) production ok=True + headroom 2048.0 MB → enabled even when
    ``graph_raw`` failed; (b) production ok=False → disabled even when
    ``graph_raw`` is ok=True with headroom 2048.0 MB (the diagnostic face
    never gates the conclusion); (c) the strict threshold: production ok=True
    with headroom 512.0 MB → disabled (``> 1024.0`` MB, not ``>=``).
    """
    # (a) production face healthy, raw diagnostic face failed → enabled.
    result_a = probe.derive_conclusions(
        _benign_family_result(
            graph_face={"ok": True, "headroom_mb": 2048.0},
            graph_raw_face={"ok": False, "headroom_mb": None, "error": "raw capture failed"},
        )
    )
    assert result_a["graph_enabled"] is True

    # (b) production face failed, raw diagnostic face healthy → disabled.
    result_b = probe.derive_conclusions(
        _benign_family_result(
            graph_face={"ok": False, "headroom_mb": None, "error": "production capture failed"},
            graph_raw_face={"ok": True, "headroom_mb": 2048.0},
        )
    )
    assert result_b["graph_enabled"] is False

    # (c) strict threshold: 512.0 MB is NOT above the 1024.0 MB threshold.
    result_c = probe.derive_conclusions(
        _benign_family_result(
            graph_face={"ok": True, "headroom_mb": 512.0},
            graph_raw_face={"ok": True, "headroom_mb": 512.0},
        )
    )
    assert result_c["graph_enabled"] is False


# ---------------------------------------------------------------------------
# UT-2605: the production driver — CUDA check FIRST, before any model load
# ---------------------------------------------------------------------------


def test_production_driver_cpu_early_out_no_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-2605: ``_drive_production_graph_capture`` checks CUDA before loading.

    The production driver's contract (design.md §4.2.2 point 4) is that the
    CUDA check happens FIRST — before any model load — so on the CPU face it
    returns a structured ``ok=False`` result (error mentioning CUDA,
    ``capture_delta_mb``/``headroom_mb`` None, an empty ``diagnostics`` list)
    and never reaches ``_load_probe_model`` (monkeypatched to raise
    AssertionError: reaching the loader is a failure, not a skip).
    """
    _force_cpu_face(monkeypatch)

    result = probe._drive_production_graph_capture(SimpleNamespace())

    assert isinstance(result, dict)
    assert result["ok"] is False
    assert "CUDA" in result["error"]
    assert result["capture_delta_mb"] is None
    assert result["headroom_mb"] is None
    assert result["diagnostics"] == []


# ---------------------------------------------------------------------------
# UT-2605: make_graph_face routes to the production driver
# ---------------------------------------------------------------------------


def test_graph_face_routes_to_production_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-2605: ``make_graph_face`` is a thin wrapper over the production driver.

    A monkeypatched ``_drive_production_graph_capture`` spy returns a canned
    dict (canary key included); calling the constructed face must return
    EXACTLY the canned dict — the wrapper adds, drops, or rewrites nothing —
    and the spy must have been called exactly once.
    """
    canned = {"ok": True, "canary": 2604}
    spy_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _spy_driver(*args: Any, **kwargs: Any) -> dict[str, Any]:
        spy_calls.append((args, kwargs))
        return canned

    monkeypatch.setattr(probe, "_drive_production_graph_capture", _spy_driver)

    face = probe.make_graph_face(SimpleNamespace())
    result = face()

    assert result == canned
    assert result["canary"] == 2604
    assert len(spy_calls) == 1


# ---------------------------------------------------------------------------
# UT-2605: the demoted raw diagnostic face keeps the CPU early out
# ---------------------------------------------------------------------------


def test_graph_raw_face_cpu_early_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2605: ``make_graph_raw_face`` returns ``ok=False`` on CPU, no load.

    The demoted raw-capture diagnostic face keeps the probe's existing CPU
    contract (no CUDA → structured ``ok=False`` with a CUDA-mentioning error,
    no exception, no model load — the monkeypatched ``_load_probe_model``
    raising AssertionError proves the early out).
    """
    _force_cpu_face(monkeypatch)

    face = probe.make_graph_raw_face(SimpleNamespace())
    result = face()

    assert isinstance(result, dict)
    assert result["ok"] is False
    assert "CUDA" in result["error"]


# ---------------------------------------------------------------------------
# UT-2606: the AR026 four-mode real-graph bench contract.
#
# The bench script does NOT exist in the Red state, so every test below
# imports it LAZILY inside the test body — a module-level import would
# break the whole test module (UT-2605 included) at collection time.
# ---------------------------------------------------------------------------

#: The frozen single-dial graph-knob name set (design 4.2.2 point 5: the
#: graph modes differ from the eager anchor ONLY in these knobs).
_GRAPH_KNOB_NAMES = {
    "use_cuda_graph",
    "graph_capacity_ratio",
    "graph_runner_pool_size",
    "graph_adaptive_capacity",
}


def test_ut2606_mode_matrix_frozen_single_dial() -> None:
    """UT-2606: the four-mode matrix is frozen and single-dial.

    MODES is the exact four-mode tuple; MODE_KNOBS keys cover exactly the
    modes; every knob name other than ``folded`` is a graph knob from the
    frozen set; the frozen values (0.5 / 4 / True / False) are pinned per
    mode; and the two graph modes differ from the eager anchor
    (``folded-batched``) ONLY in graph knobs (set difference of the knob
    dicts) — the AR024 single-knob discipline.
    """
    import scripts.ar026_real_graph_bench as bench

    assert bench.MODES == ("baseline", "folded-batched", "graph-fixed", "graph-adaptive")
    assert isinstance(bench.MODE_KNOBS, dict)
    assert set(bench.MODE_KNOBS) == set(bench.MODES)

    for mode, knobs in bench.MODE_KNOBS.items():
        for name in knobs:
            if name != "folded":
                assert name in _GRAPH_KNOB_NAMES, (mode, name)

    assert bench.MODE_KNOBS["baseline"] == {"folded": False}
    assert bench.MODE_KNOBS["folded-batched"] == {"folded": True, "use_cuda_graph": False}
    assert bench.MODE_KNOBS["graph-fixed"] == {
        "folded": True,
        "use_cuda_graph": True,
        "graph_capacity_ratio": 0.5,
        "graph_runner_pool_size": 4,
        "graph_adaptive_capacity": False,
    }
    assert bench.MODE_KNOBS["graph-adaptive"] == {
        "folded": True,
        "use_cuda_graph": True,
        "graph_capacity_ratio": 0.5,
        "graph_runner_pool_size": 4,
        "graph_adaptive_capacity": True,
    }

    anchor = bench.MODE_KNOBS["folded-batched"]
    for graph_mode in ("graph-fixed", "graph-adaptive"):
        knobs = bench.MODE_KNOBS[graph_mode]
        assert set(knobs) - set(anchor) <= _GRAPH_KNOB_NAMES, graph_mode
        assert set(anchor) - set(knobs) <= _GRAPH_KNOB_NAMES, graph_mode
        for key in set(knobs) & set(anchor):
            if key not in _GRAPH_KNOB_NAMES:
                assert knobs[key] == anchor[key], (graph_mode, key)


def test_ut2606_schema_name_and_output_paths() -> None:
    """UT-2606: schema name + output path semantics (override/quick/formal).

    Same semantics as ar023's path resolution: an explicit ``--out``
    override ALWAYS wins (even under ``--quick``); ``--quick`` resolves to
    the dedicated ``ar026_real_graph_bench_quick.json`` artifact, never the
    formal one; the default is the formal artifact path.
    """
    import scripts.ar026_real_graph_bench as bench

    assert bench.SCHEMA_NAME == "ar026_real_graph_bench"
    assert bench.DEFAULT_OUTPUT_PATH == Path("results/optimization/ar026_real_graph_bench.json")

    quick = bench.quick_output_path()
    assert quick.name == "ar026_real_graph_bench_quick.json"
    assert quick.parent == bench.DEFAULT_OUTPUT_PATH.parent

    # Explicit override wins — even under --quick, for either family.
    assert bench.resolve_output_path("llada", True, "override.json") == Path("override.json")
    assert bench.resolve_output_path("dream", False, "nested/out.json") == Path("nested/out.json")

    # Quick -> the dedicated quick artifact, never the formal one.
    assert bench.resolve_output_path("llada", True, None) == quick
    assert bench.resolve_output_path("dream", True, None) == bench.quick_output_path()

    # Formal -> the formal artifact path.
    assert bench.resolve_output_path("llada", False, None) == bench.DEFAULT_OUTPUT_PATH
    assert bench.resolve_output_path("dream", False, None) == bench.DEFAULT_OUTPUT_PATH


def test_ut2606_cli_contract() -> None:
    """UT-2606: the minimal CLI — family default llada, dream allowed, reps 30.

    Unlike AR024's llada-only guard, ``--family dream`` is ALLOWED here (the
    formal commitment is llada via the artifact meta, not a CLI rejection).
    No HF env args belong on this CLI (``--repo-override`` must not exist).
    """
    import scripts.ar026_real_graph_bench as bench

    args = bench.parse_args([])
    assert args.family == "llada"
    assert args.reps == 30
    assert args.quick is False
    assert args.out is None

    # dream is an accepted family choice (AR026 allows it; AR024 did not).
    dream = bench.parse_args(["--family", "dream", "--quick"])
    assert dream.family == "dream"
    assert dream.quick is True

    custom = bench.parse_args(["--reps", "5", "--out", "x.json"])
    assert custom.reps == 5
    assert custom.out == "x.json"

    # The minimal CLI carries no HF env surface.
    assert not hasattr(args, "repo_override")


def test_ut2606_reuse_identity_pins() -> None:
    """UT-2606: the ar023 contract pieces are reused BY IDENTITY, never copied.

    Every re-exported callable/constant must be the very same object as
    ``scripts.ar023_cross_family_bench``'s — the assign-form AR024 pattern;
    a local re-implementation (even a behaviorally identical one) is drift
    and fails this pin.
    """
    import scripts.ar023_cross_family_bench as _ar023
    import scripts.ar026_real_graph_bench as bench

    assert bench.h1_quality_verdict is _ar023.h1_quality_verdict
    assert bench.h2_speed_verdict is _ar023.h2_speed_verdict
    assert bench.h4_fullstack_verdict is _ar023.h4_fullstack_verdict
    assert bench.reproduction_verdict is _ar023.reproduction_verdict
    assert bench.mark_all_blocked is _ar023.mark_all_blocked
    assert bench.build_blocked_artifact is _ar023.build_blocked_artifact
    assert bench.write_results is _ar023.write_results
    assert bench.SEED is _ar023.SEED
    assert bench.REPS_DEFAULT is _ar023.REPS_DEFAULT


def test_ut2606_build_artifact_schema_override() -> None:
    """UT-2606: own assembler, ar023 key set, ar026 schema name.

    The AR024 schema-name override pattern: the assembled artifact reuses
    the ar023 artifact key set exactly (family / status / modes / quality /
    verdicts / reproduction / meta / schema_version) while owning its schema
    identifier ``ar026_real_graph_bench``.
    """
    import scripts.ar026_real_graph_bench as bench

    artifact = bench.build_artifact(
        "llada",
        "ok",
        None,
        None,
        {"h1": {"verdict": "supported"}},
        {"ok": True},
        {"note": "dummy"},
    )

    assert artifact["schema"] == "ar026_real_graph_bench"
    assert artifact["family"] == "llada"
    assert artifact["status"] == "ok"
    assert isinstance(artifact["schema_version"], int)
    assert set(artifact) == {
        "schema",
        "schema_version",
        "family",
        "status",
        "modes",
        "quality",
        "verdicts",
        "reproduction",
        "meta",
    }


def test_ut2606_geometry_inheritance_note_obligation() -> None:
    """UT-2606: the geometry-inheritance waiving reason is explicit, not silent.

    ``GEOMETRY_INHERITANCE_NOTE`` is a non-empty module constant string that
    artifacts must embed in meta; it must mention the VRAM constraint (the
    16GB card — bf16-8B + graph-pool VRAM headroom is the binding reason the
    EVR >=4-shape floor is explicitly waived while inheriting the AR023
    anchor geometry).
    """
    import scripts.ar026_real_graph_bench as bench

    note = bench.GEOMETRY_INHERITANCE_NOTE
    assert isinstance(note, str)
    assert note.strip() != ""
    assert "vram" in note.lower()
    assert "16" in note


def test_ut2606_blocked_semantics_zero_fabrication() -> None:
    """UT-2606: blocked semantics — status blocked, zero fabricated numbers.

    The re-exported ``build_blocked_artifact`` (pinned by identity above)
    must keep the AR023 FR5 face: a degraded conclusion produces a blocked
    artifact with ``modes``/``quality`` None and empty verdicts — no
    fabricated measurements (mirrors UT-2332's pin in
    ``tests/test_ar023_bench.py``).
    """
    import scripts.ar026_real_graph_bench as bench

    conclusion = {
        "degraded": True,
        "degrade_reason": "graph capture failed on the real checkpoint",
        "dtype_choice": "bf16",
        "graph_enabled": False,
    }
    artifact = bench.build_blocked_artifact(
        "dream", conclusion, "results/optimization/ar023_family_probe.json"
    )

    assert artifact["status"] == "blocked"
    assert artifact["modes"] is None
    assert artifact["quality"] is None
    assert artifact["verdicts"] == {}
    assert artifact["blocked_reason"] == "graph capture failed on the real checkpoint"


# ---------------------------------------------------------------------------
# UT-2606 Green-phase spec pin (the T003 pinning precedent): the frozen H1
# graph parity rule. The test is derived from the PRE-REGISTERED wording
# (design 4.2.2 / the bench module docstring), not from observed
# implementation behavior: faces (a) posthoc parity within the frozen atol,
# (b) decode-length equality, (c) block-grid exact full match vs the anchor;
# the engagement gate (zero captures -> inconclusive, never a silent parity
# pass); precedence refuted > inconclusive > supported; None mean refutes.
# ---------------------------------------------------------------------------

#: Both graph modes engaged (captures > 0) in every scenario below unless
#: stated otherwise.
_ENGAGED = {
    "graph-fixed": {"captures": 2, "replays": 60, "evictions": 0},
    "graph-adaptive": {"captures": 2, "replays": 60, "evictions": 0},
}


def _parity_prompt(
    idx,
    anchor_mean,
    fixed_mean,
    adaptive_mean,
    fixed_len=34,
    adaptive_len=34,
    fixed_exact=(32, 32),
    adaptive_exact=(32, 32),
):
    """One per-prompt entry in the h1_graph_parity_verdict input layout."""
    return {
        "prompt_index": idx,
        "posthoc_overall_mean": {
            "baseline": 0.5,
            "folded-batched": anchor_mean,
            "graph-fixed": fixed_mean,
            "graph-adaptive": adaptive_mean,
        },
        "decode_length": {
            "baseline": 40,
            "folded-batched": 34,
            "graph-fixed": fixed_len,
            "graph-adaptive": adaptive_len,
        },
        "block_exact_vs_anchor": {
            "graph-fixed": {"matched": fixed_exact[0], "compared": fixed_exact[1]},
            "graph-adaptive": {"matched": adaptive_exact[0], "compared": adaptive_exact[1]},
        },
    }


def test_ut2606_h1_graph_parity_spec_pin() -> None:
    """UT-2606: the frozen graph-parity rule (engagement gate + 3 faces)."""
    import scripts.ar026_real_graph_bench as bench

    good = [
        _parity_prompt(0, 0.369661, 0.369661, 0.369661),
        _parity_prompt(1, 0.570341, 0.570341, 0.570341),
    ]
    verdict = bench.h1_graph_parity_verdict(good, _ENGAGED)
    assert verdict["verdict"] == "supported"
    for mode in ("graph-fixed", "graph-adaptive"):
        assert verdict["modes"][mode]["verdict"] == "supported"
        assert verdict["modes"][mode]["engaged"] is True
        assert all(verdict["modes"][mode]["faces"].values())

    # Face (a) violated: a posthoc drift beyond the frozen atol refutes the
    # drifting mode only (per-mode isolation; the clean leg stays supported).
    drifted = [
        _parity_prompt(0, 0.369661, 0.369661, 0.4),
        _parity_prompt(1, 0.570341, 0.570341, 0.570341),
    ]
    verdict = bench.h1_graph_parity_verdict(drifted, _ENGAGED)
    assert verdict["verdict"] == "refuted"
    assert verdict["modes"]["graph-adaptive"]["verdict"] == "refuted"
    assert not verdict["modes"]["graph-adaptive"]["faces"]["a_posthoc_parity_atol_1e-6"]
    assert verdict["modes"]["graph-fixed"]["verdict"] == "supported"

    # Face (b) violated: decode-length divergence refutes the mode.
    len_break = [
        _parity_prompt(0, 0.369661, 0.369661, 0.369661, fixed_len=33),
        _parity_prompt(1, 0.570341, 0.570341, 0.570341),
    ]
    verdict = bench.h1_graph_parity_verdict(len_break, _ENGAGED)
    assert verdict["verdict"] == "refuted"
    assert not verdict["modes"]["graph-fixed"]["faces"]["b_decode_length_equality"]

    # Face (c) violated: block-grid exact < full match refutes the mode.
    # The miss is deliberately NEAR-full (999/1000 merged with a clean
    # second prompt ~= 0.999): a mutant threshold (>= 0.99, >= 0.999-ish)
    # must not sneak a sub-1.0 rate through as "full match" — the frozen
    # face is rate == 1.0 exactly.
    exact_break = [
        _parity_prompt(0, 0.369661, 0.369661, 0.369661, adaptive_exact=(999, 1000)),
        _parity_prompt(1, 0.570341, 0.570341, 0.570341),
    ]
    verdict = bench.h1_graph_parity_verdict(exact_break, _ENGAGED)
    assert verdict["verdict"] == "refuted"
    assert not verdict["modes"]["graph-adaptive"]["faces"]["c_block_exact_full_match_vs_anchor"]

    # Engagement gate: zero captures -> inconclusive, NEVER a silent parity
    # pass (the mode ran eager by fallback).
    zero_fixed = {
        "graph-fixed": {"captures": 0, "replays": 0, "evictions": 0},
        "graph-adaptive": {"captures": 2, "replays": 60, "evictions": 0},
    }
    verdict = bench.h1_graph_parity_verdict(good, zero_fixed)
    assert verdict["verdict"] == "inconclusive"
    assert verdict["modes"]["graph-fixed"]["verdict"] == "inconclusive"
    assert verdict["modes"]["graph-fixed"]["engaged"] is False
    assert verdict["modes"]["graph-adaptive"]["verdict"] == "supported"

    # Precedence: refuted beats inconclusive (an engaged mode's face failure
    # is stronger information than another leg's missing engagement).
    verdict = bench.h1_graph_parity_verdict(drifted, zero_fixed)
    assert verdict["verdict"] == "refuted"

    # A None posthoc mean refutes (no fabricated numbers) — the anchor-side
    # None breaks both graph legs' face (a).
    none_anchor = [
        _parity_prompt(0, None, 0.369661, 0.369661),
        _parity_prompt(1, 0.570341, 0.570341, 0.570341),
    ]
    verdict = bench.h1_graph_parity_verdict(none_anchor, _ENGAGED)
    assert verdict["verdict"] == "refuted"
    for mode in ("graph-fixed", "graph-adaptive"):
        assert verdict["modes"][mode]["verdict"] == "refuted"
        assert verdict["modes"][mode]["numbers"]["none_posthoc_mean"] is True
