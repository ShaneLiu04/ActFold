"""Red-phase tests for AR023/T001: the cross-family model probe script.

Target (design.md §4.1.7 + §6.2/§6.3, srs FR1, API contract): ``scripts/ar023_family_probe.py``
— a family probe that measures seven faces (download / load / bf16 / int8 / smoke /
graph headroom / prompt adaptation) per checkpoint family
(``GSA-SAI/LLaDA-8B`` and ``Dream-org/Dream-v0-Base-7B``) and derives the
family-level conclusions (``dtype_choice`` / ``graph_enabled`` / ``degraded`` /
``degrade_reason``) that parameterize all later AR023 tasks.

Only the CI-safe pure-logic surface is tested here — no GPU, no network, no
real weights:

* ``PROBE_REPOS`` / ``DEFAULT_OUTPUT_PATH`` constants;
* ``sha256_file`` against hashlib on tmp_path fixtures;
* ``derive_conclusions``: bf16-over-int8 priority, the strict ``> 1024.0`` MB
  graph-headroom threshold, the three independent degrade triggers (load fail /
  bf16+int8 both fail / smoke fail), degrade_reason content;
* ``run_faces_with_isolation``: one face raising must not stop the rest
  (srs FR1 "单项失败不中断" — the mechanism face lives here);
* ``build_artifact``: seven face dicts preserved verbatim + ``conclusions``
  appended + top-level ``schema_version``/``meta``/``families`` structure;
* ``write_results``: JSON round-trip via ``json.load``;
* ``parse_repo_overrides``: argparse ``action="append"`` collected value
  strings of the form ``family=repo``; empty side or missing ``=`` → ValueError.

Interpretation pin: ``parse_repo_overrides(args)`` receives the *collected
value strings* (e.g. ``["llada=Foo/Bar", "dream=Baz/Qux"]``), NOT the raw argv
tokens — the ``--repo-override`` flag itself is consumed by argparse in
``main()``.

The target module is imported lazily inside each test so the Red state —
``scripts.ar023_family_probe`` not existing yet — fails tests individually
with ``ModuleNotFoundError`` instead of aborting module collection. A
subprocess test additionally pins the lightweight-import contract: importing
the module must not pull torch / transformers / bitsandbytes at module top
(the CI host has no CUDA stack and the heavy deps must stay lazy inside the
face functions).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_ALL_FACES = ("download", "load", "bf16", "int8", "smoke", "graph", "prompt")
_GRAPH_HEADROOM_THRESHOLD_MB = 1024.0


# ---------------------------------------------------------------------------
# Fixture-style helpers
# ---------------------------------------------------------------------------


def _make_family_result(**face_overrides: dict) -> dict:
    """Build a single-family seven-face result dict, all faces ok by default.

    Mirrors the API contract: keys ``download/load/bf16/int8/smoke/graph/prompt``,
    each face at least ``{"ok": bool}``, the graph face additionally carrying
    ``headroom_mb``. Individual faces are replaced wholesale via
    ``face_overrides``.
    """
    faces: dict[str, dict] = {
        "download": {"ok": True, "files": []},
        "load": {"ok": True, "note": "loaded"},
        "bf16": {"ok": True, "vram_mb": 16384.0},
        "int8": {"ok": True, "vram_mb": 8192.0},
        "smoke": {"ok": True, "tokens": 16},
        "graph": {"ok": True, "headroom_mb": 2048.0},
        "prompt": {"ok": True, "note": "padded"},
    }
    faces.update(face_overrides)
    return faces


# ---------------------------------------------------------------------------
# (1) derive_conclusions: dtype_choice priority
# ---------------------------------------------------------------------------


def test_ut_2301_dtype_choice_prefers_bf16_when_ok() -> None:
    """dtype_choice is "bf16" whenever bf16.ok is True — even if int8 also ok."""
    from scripts.ar023_family_probe import derive_conclusions

    both_ok = derive_conclusions(_make_family_result())
    assert both_ok["dtype_choice"] == "bf16"

    bf16_only = derive_conclusions(_make_family_result(int8={"ok": False}))
    assert bf16_only["dtype_choice"] == "bf16"


def test_ut_2302_dtype_choice_falls_back_to_int8_when_bf16_fails() -> None:
    """bf16 fail + int8 ok → dtype_choice "int8"."""
    from scripts.ar023_family_probe import derive_conclusions

    result = derive_conclusions(_make_family_result(bf16={"ok": False, "error": "oom"}))
    assert result["dtype_choice"] == "int8"


def test_ut_2303_dtype_choice_none_when_bf16_and_int8_both_fail() -> None:
    """bf16 fail + int8 fail → dtype_choice None (no viable dtype)."""
    from scripts.ar023_family_probe import derive_conclusions

    result = derive_conclusions(
        _make_family_result(bf16={"ok": False}, int8={"ok": False, "error": "bnb missing"})
    )
    assert result["dtype_choice"] is None


# ---------------------------------------------------------------------------
# (2) derive_conclusions: the three independent degrade triggers
# ---------------------------------------------------------------------------


def test_ut_2304_degraded_true_when_load_fails() -> None:
    """load.ok False alone degrades the family; reason names the failed face."""
    from scripts.ar023_family_probe import derive_conclusions

    result = derive_conclusions(
        _make_family_result(load={"ok": False, "error": "transformers too new"})
    )
    assert result["degraded"] is True
    reason = result["degrade_reason"]
    assert isinstance(reason, str) and reason != ""
    assert "load" in reason.lower()


def test_ut_2305_degraded_true_when_bf16_and_int8_both_fail() -> None:
    """bf16+int8 double failure degrades; reason names both failed faces."""
    from scripts.ar023_family_probe import derive_conclusions

    result = derive_conclusions(
        _make_family_result(bf16={"ok": False}, int8={"ok": False, "error": "no wheel"})
    )
    assert result["degraded"] is True
    reason = result["degrade_reason"]
    assert isinstance(reason, str) and reason != ""
    assert "bf16" in reason.lower()
    assert "int8" in reason.lower()


def test_ut_2306_degraded_true_when_smoke_fails() -> None:
    """smoke.ok False alone degrades the family; reason names the failed face."""
    from scripts.ar023_family_probe import derive_conclusions

    result = derive_conclusions(_make_family_result(smoke={"ok": False, "error": "decode NaN"}))
    assert result["degraded"] is True
    reason = result["degrade_reason"]
    assert isinstance(reason, str) and reason != ""
    assert "smoke" in reason.lower()


def test_ut_2307_degraded_false_and_reason_none_when_all_faces_ok() -> None:
    """All seven faces ok → degraded False and degrade_reason None."""
    from scripts.ar023_family_probe import derive_conclusions

    result = derive_conclusions(_make_family_result())
    assert result["degraded"] is False
    assert result["degrade_reason"] is None


# ---------------------------------------------------------------------------
# (3) derive_conclusions: graph_enabled threshold (strict > 1024.0 MB)
# ---------------------------------------------------------------------------


def test_ut_2308_graph_enabled_strict_headroom_threshold() -> None:
    """graph.ok + headroom 1025.0 MB → True; exactly 1024.0 MB → False (strict >)."""
    from scripts.ar023_family_probe import derive_conclusions

    above = derive_conclusions(_make_family_result(graph={"ok": True, "headroom_mb": 1025.0}))
    assert above["graph_enabled"] is True

    boundary = derive_conclusions(_make_family_result(graph={"ok": True, "headroom_mb": 1024.0}))
    assert boundary["graph_enabled"] is False


def test_ut_2309_graph_enabled_false_on_none_headroom_or_face_failure() -> None:
    """graph.ok + headroom None → False; graph not ok → False even with huge headroom."""
    from scripts.ar023_family_probe import derive_conclusions

    none_headroom = derive_conclusions(_make_family_result(graph={"ok": True, "headroom_mb": None}))
    assert none_headroom["graph_enabled"] is False

    face_failed = derive_conclusions(
        _make_family_result(graph={"ok": False, "headroom_mb": 8192.0})
    )
    assert face_failed["graph_enabled"] is False


# ---------------------------------------------------------------------------
# (4) run_faces_with_isolation: one face raising must not stop the rest
# ---------------------------------------------------------------------------


def test_ut_2310_run_faces_with_isolation_first_face_raise_does_not_stop_rest() -> None:
    """A raising face records ok False + error with the exception message; later
    faces still execute and their return dicts are preserved verbatim."""
    from scripts.ar023_family_probe import run_faces_with_isolation

    executed: list[str] = []

    def _boom() -> dict:
        raise RuntimeError("probe face exploded")

    def _fine() -> dict:
        executed.append("smoke")
        return {"ok": True, "tokens": 16}

    result = run_faces_with_isolation({"download": _boom, "smoke": _fine})

    assert set(result) == {"download", "smoke"}
    assert executed == ["smoke"], "the face after the raising one must still run"

    failed = result["download"]
    assert failed["ok"] is False
    error = failed["error"]
    assert isinstance(error, str)
    assert "RuntimeError" in error
    assert "probe face exploded" in error

    assert result["smoke"] == {"ok": True, "tokens": 16}


# ---------------------------------------------------------------------------
# (5) sha256_file
# ---------------------------------------------------------------------------


def test_ut_2311_sha256_file_matches_hashlib_on_known_contents(tmp_path: Path) -> None:
    """sha256_file returns the hashlib SHA256 hex digest for known contents."""
    from scripts.ar023_family_probe import sha256_file

    payload = b"ar023 family probe known payload \x00\x01\xff\n"
    blob = tmp_path / "blob.bin"
    blob.write_bytes(payload)
    assert sha256_file(blob) == hashlib.sha256(payload).hexdigest()

    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert sha256_file(empty) == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# (6) parse_repo_overrides
# ---------------------------------------------------------------------------


def test_ut_2312_parse_repo_overrides_parses_single_and_multiple() -> None:
    """Legal ``family=repo`` value strings parse to a family→repo dict; empty → {}."""
    from scripts.ar023_family_probe import parse_repo_overrides

    assert parse_repo_overrides(["llada=Some/Repo"]) == {"llada": "Some/Repo"}

    multiple = parse_repo_overrides(["llada=GSA-SAI/LLaDA-8B", "dream=Dream-org/Dream-v0-Base-7B"])
    assert multiple == {"llada": "GSA-SAI/LLaDA-8B", "dream": "Dream-org/Dream-v0-Base-7B"}

    assert parse_repo_overrides([]) == {}


def test_ut_2313_parse_repo_overrides_rejects_malformed_specs() -> None:
    """Empty family side, empty repo side, or a missing ``=`` all raise ValueError."""
    from scripts.ar023_family_probe import parse_repo_overrides

    with pytest.raises(ValueError):
        parse_repo_overrides(["=Some/Repo"])
    with pytest.raises(ValueError):
        parse_repo_overrides(["llada="])
    with pytest.raises(ValueError):
        parse_repo_overrides(["lladaSomeRepo"])


# ---------------------------------------------------------------------------
# (7) build_artifact
# ---------------------------------------------------------------------------


def test_ut_2314_build_artifact_preserves_faces_and_appends_conclusions() -> None:
    """build_artifact keeps every face dict verbatim, appends conclusions, and
    assembles the top-level schema_version/meta/families structure for both
    families."""
    from scripts.ar023_family_probe import build_artifact, derive_conclusions

    llada = _make_family_result()
    dream = _make_family_result(
        bf16={"ok": False, "error": "oom"},
        int8={"ok": False, "error": "bnb wheel missing"},
        graph={"ok": False, "headroom_mb": None},
    )
    meta = {"probe": "ar023", "transformers": "4.53.1"}

    artifact = build_artifact({"llada": llada, "dream": dream}, meta)

    assert artifact["schema_version"] == 1
    assert artifact["meta"] == meta
    assert set(artifact["families"]) == {"llada", "dream"}

    for family, faces in (("llada", llada), ("dream", dream)):
        family_dict = artifact["families"][family]
        for face in _ALL_FACES:
            assert face in family_dict, f"family {family!r} lost face key {face!r}"
            assert family_dict[face] == faces[face], (
                f"family {family!r} face {face!r} must be preserved verbatim: "
                f"{family_dict[face]!r} != {faces[face]!r}"
            )
        assert "conclusions" in family_dict
        assert family_dict["conclusions"] == derive_conclusions(faces)

    llada_conclusions = artifact["families"]["llada"]["conclusions"]
    assert llada_conclusions["dtype_choice"] == "bf16"
    assert llada_conclusions["degraded"] is False

    dream_conclusions = artifact["families"]["dream"]["conclusions"]
    assert dream_conclusions["dtype_choice"] is None
    assert dream_conclusions["degraded"] is True
    assert isinstance(dream_conclusions["degrade_reason"], str)


# ---------------------------------------------------------------------------
# (8) write_results: JSON round-trip
# ---------------------------------------------------------------------------


def test_ut_2315_write_results_round_trips_json(tmp_path: Path) -> None:
    """write_results dumps the artifact so json.load(path) == the original dict."""
    from scripts.ar023_family_probe import write_results

    artifact = {
        "schema_version": 1,
        "meta": {"probe": "ar023", "note": "下载/加载实测"},
        "families": {
            "llada": {
                "download": {
                    "ok": True,
                    "files": [{"name": "model.safetensors", "sha256": "ab" * 32, "bytes": 7}],
                },
                "graph": {"ok": True, "headroom_mb": 1536.0},
                "conclusions": {
                    "dtype_choice": "bf16",
                    "graph_enabled": True,
                    "degraded": False,
                    "degrade_reason": None,
                },
            }
        },
    }
    out_path = tmp_path / "ar023_family_probe.json"
    write_results(artifact, out_path)

    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed == artifact


# ---------------------------------------------------------------------------
# (9) constants
# ---------------------------------------------------------------------------


def test_ut_2316_probe_repos_and_default_output_path_constants() -> None:
    """PROBE_REPOS pins the two checkpoint repo ids; DEFAULT_OUTPUT_PATH points
    at results/optimization/ar023_family_probe.json."""
    from scripts.ar023_family_probe import DEFAULT_OUTPUT_PATH, PROBE_REPOS

    assert PROBE_REPOS == {
        "llada": "GSAI-ML/LLaDA-8B-Base",
        "dream": "Dream-org/Dream-v0-Base-7B",
    }

    default_path = Path(DEFAULT_OUTPUT_PATH)
    assert default_path.name == "ar023_family_probe.json"
    assert "results" in default_path.parts
    assert "optimization" in default_path.parts


# ---------------------------------------------------------------------------
# (10) lightweight-import contract (soft constraint, subprocess-isolated)
# ---------------------------------------------------------------------------


def test_ut_2317_module_import_stays_lightweight() -> None:
    """Importing the module must not pull torch/transformers/bitsandbytes
    (heavy deps stay lazy inside the face functions); the pure functions are
    callable afterwards. Runs in a clean subprocess so a pytest session that
    already imported torch elsewhere cannot mask a violation."""
    code = (
        "import sys\n"
        "import scripts.ar023_family_probe as probe\n"
        "heavy = [m for m in ('torch', 'transformers', 'bitsandbytes') if m in sys.modules]\n"
        "assert not heavy, f'module top-level import pulled heavy deps: {heavy}'\n"
        "assert callable(probe.derive_conclusions)\n"
        "assert callable(probe.run_faces_with_isolation)\n"
        "assert callable(probe.sha256_file)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"lightweight-import probe failed (exit {proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# (11) [T001-2] CUDA-face subprocess isolation
# ---------------------------------------------------------------------------


class _FakeCompleted:
    """Minimal stand-in for subprocess.CompletedProcess (str stdout/stderr)."""

    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_recording_runner(
    result: _FakeCompleted,
) -> tuple:
    """Build a fake ``subprocess.run`` that records the command it received."""
    calls: list[list[str]] = []

    def runner(cmd: list[str], **_kwargs: object) -> _FakeCompleted:
        calls.append(list(cmd))
        return result

    return runner, calls


def test_ut_2318_cuda_face_names_is_exactly_the_gpu_faces() -> None:
    """CUDA_FACE_NAMES pins the GPU faces that must run in fresh
    subprocesses; download/load stay in-process. AR026 T004 adds
    ``graph_raw`` (the demoted raw-capture diagnostic face — a failed
    capture poisons process-wide allocator state, so BOTH graph faces need
    the isolation wrapper)."""
    from scripts.ar023_family_probe import CUDA_FACE_NAMES

    assert set(CUDA_FACE_NAMES) == {
        "bf16",
        "int8",
        "smoke",
        "graph",
        "graph_raw",
        "prompt",
    }
    assert "download" not in CUDA_FACE_NAMES
    assert "load" not in CUDA_FACE_NAMES


def test_ut_2319_build_face_worker_command_shape() -> None:
    """The parent re-invokes this very module as a single-face worker:
    ``<python> -m scripts.ar023_family_probe --_face-worker FAMILY FACE
    --repo REPO --local-dir DIR``."""
    from scripts.ar023_family_probe import build_face_worker_command

    cmd = build_face_worker_command(
        "dream", "bf16", "Dream-org/Dream-v0-Base-7B", Path("D:/m/dream")
    )
    assert cmd[0] == sys.executable
    assert cmd[1] == "-m"
    assert cmd[2] == "scripts.ar023_family_probe"
    i = cmd.index("--_face-worker")
    assert cmd[i + 1] == "dream"
    assert cmd[i + 2] == "bf16"
    assert cmd[cmd.index("--repo") + 1] == "Dream-org/Dream-v0-Base-7B"
    assert cmd[cmd.index("--local-dir") + 1] == str(Path("D:/m/dream"))


def test_ut_2320_parse_face_worker_output_finds_last_sentinel_line() -> None:
    """The worker prints its result as a sentinel-prefixed LAST line so stray
    library prints on stdout cannot corrupt the protocol; parsing picks the
    last sentinel line and decodes its JSON payload."""
    from scripts.ar023_family_probe import FACE_RESULT_SENTINEL, parse_face_worker_output

    assert FACE_RESULT_SENTINEL == "AR023_FACE_RESULT:"

    payload = {"ok": True, "vram_mb": 1234.0}
    stdout = (
        "some library warning line\n"
        f"{FACE_RESULT_SENTINEL}{json.dumps({'ok': True, 'stale': True})}\n"
        "intermediate noise\n"
        f"{FACE_RESULT_SENTINEL}{json.dumps(payload)}\n"
    )
    assert parse_face_worker_output(stdout) == payload

    bare = f"{FACE_RESULT_SENTINEL}{json.dumps(payload)}"
    assert parse_face_worker_output(bare) == payload


def test_ut_2321_parse_face_worker_output_raises_without_sentinel() -> None:
    """No sentinel line in the worker output → ValueError (the caller converts
    it into an ok=False face result, never a crash)."""
    from scripts.ar023_family_probe import parse_face_worker_output

    with pytest.raises(ValueError):
        parse_face_worker_output("torch warnings everywhere\nnothing usable\n")
    with pytest.raises(ValueError):
        parse_face_worker_output("")


def test_ut_2322_make_subprocess_face_success_parses_sentinel_result() -> None:
    """rc=0 + sentinel JSON → the face result is the parsed dict; the runner is
    invoked once with captured output."""
    from scripts.ar023_family_probe import FACE_RESULT_SENTINEL, make_subprocess_face

    payload = {"ok": True, "note": "bf16 loaded", "vram_mb": 15265.0}
    runner, calls = _make_recording_runner(
        _FakeCompleted(0, f"noise\n{FACE_RESULT_SENTINEL}{json.dumps(payload)}\n", "")
    )
    face = make_subprocess_face(
        "llada", "bf16", "GSAI-ML/LLaDA-8B-Base", Path("D:/m/llada"), runner=runner
    )
    result = face()
    assert result == payload
    assert len(calls) == 1
    assert calls[0][1:3] == ["-m", "scripts.ar023_family_probe"]


def test_ut_2323_make_subprocess_face_worker_crash_maps_to_ok_false() -> None:
    """rc != 0 → {"ok": False} with the rc and a stderr tail in the error; the
    remaining faces are unaffected because the failure is a value, not a raise."""
    from scripts.ar023_family_probe import make_subprocess_face

    runner, _calls = _make_recording_runner(
        _FakeCompleted(1, "", "Traceback (most recent call last):\nCUDA error: out of memory")
    )
    result = make_subprocess_face(
        "dream", "graph", "Dream-org/Dream-v0-Base-7B", Path("D:/m/dream"), runner=runner
    )()
    assert result["ok"] is False
    assert "rc=1" in result["error"]
    assert "out of memory" in result["error"]


def test_ut_2324_make_subprocess_face_unusable_output_maps_to_ok_false() -> None:
    """rc=0 but no sentinel line → ok=False with a protocol error message (a
    worker that crashed before printing the sentinel is a failed face)."""
    from scripts.ar023_family_probe import make_subprocess_face

    runner, _calls = _make_recording_runner(_FakeCompleted(0, "printed something else\n", ""))
    result = make_subprocess_face(
        "llada", "prompt", "GSAI-ML/LLaDA-8B-Base", Path("D:/m/llada"), runner=runner
    )()
    assert result["ok"] is False
    assert isinstance(result["error"], str) and result["error"]


def test_ut_2325_wrap_cuda_faces_replaces_only_gpu_faces() -> None:
    """wrap_cuda_faces_with_subprocess keeps download/load callables by identity
    and replaces exactly the five CUDA faces with subprocess-backed ones; calling
    a wrapped CUDA face routes through the injected runner."""
    from scripts.ar023_family_probe import FACE_RESULT_SENTINEL, wrap_cuda_faces_with_subprocess

    faces = {name: (lambda n=name: {"ok": True, "face": n}) for name in _ALL_FACES}
    payload = {"ok": True, "via": "worker"}
    runner, calls = _make_recording_runner(
        _FakeCompleted(0, f"{FACE_RESULT_SENTINEL}{json.dumps(payload)}\n", "")
    )
    wrapped = wrap_cuda_faces_with_subprocess(
        faces, "dream", "Dream-org/Dream-v0-Base-7B", Path("D:/m/dream"), runner=runner
    )

    assert set(wrapped) == set(faces)
    assert wrapped["download"] is faces["download"]
    assert wrapped["load"] is faces["load"]
    for gpu_face in ("bf16", "int8", "smoke", "graph", "prompt"):
        assert wrapped[gpu_face] is not faces[gpu_face], f"{gpu_face} must be subprocess-backed"

    assert calls == []
    assert wrapped["smoke"]() == payload
    assert len(calls) == 1
    assert calls[0][cmd_face_index(calls[0])] == "smoke"


def cmd_face_index(cmd: list[str]) -> int:
    """Test helper: index of the FACE argument in a worker command."""
    return cmd.index("--_face-worker") + 2
