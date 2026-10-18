"""AR023/T001 cross-family model probe (design.md §4.1.7, srs FR1).

Measures seven faces (download / load / bf16 / int8 / smoke / graph headroom /
prompt adaptation) for each probe family — ``GSA-SAI/LLaDA-8B`` (``llada``) and
``Dream-org/Dream-v0-Base-7B`` (``dream``) — and derives the family-level
conclusions (``dtype_choice`` / ``graph_enabled`` / ``degraded`` /
``degrade_reason``) that parameterize every later AR023 task.

Face semantics (design.md §4.1.7):

* **download** — ``curl.exe`` per-file download + SHA256 (AR019 P6: the python
  requests stack is unusable behind the corporate proxy);
* **load** — config/tokenizer/remote-code compatibility on the AR019 venv
  (transformers 4.53.1) — the heavy weight loads are measured by the bf16/int8
  faces, not here;
* **bf16** — ``torch.cuda.mem_get_info`` delta around a bf16 load + one short
  forward;
* **int8** — bitsandbytes availability probe + ``load_in_8bit`` load + short
  forward (Windows wheel risk face);
* **smoke** — short generation (16 tokens) through the family reference
  sampler + decode + mask-token resolution (AGENTS.md #25);
* **graph** — one CUDA-graph capture of a batched ``[N, T]`` forward at the
  viable dtype: ``capture_delta_mb`` + ``headroom_mb`` (free VRAM after the
  capture — the ``graph_enabled`` threshold input);
* **prompt** — benchmark-prompt encoding/padding walk-through plus the
  family-specific folding-transparency probe: 2-D ``[T, T]`` bool attention
  mask acceptance (LLaDA) and ``position_ids`` kwarg acceptance (Dream).

Single-face failure never interrupts the remaining faces (srs FR1): each face
result carries ``ok`` and, on failure, an ``error`` string. The five CUDA
faces additionally run in fresh subprocesses — a failed CUDA-graph capture
corrupts the process-wide CUDA caching-allocator state, so exception
isolation alone cannot contain it ([T001-2]). The artifact is written to
``results/optimization/ar023_family_probe.json``.

Module-top imports stay stdlib-only (+ ``scripts._hf_env``): torch /
transformers / bitsandbytes are imported lazily inside the face callables so
CI tests of the pure-logic surface stay lightweight (UT-2317).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

PROBE_REPOS: dict[str, str] = {
    "llada": "GSAI-ML/LLaDA-8B-Base",
    "dream": "Dream-org/Dream-v0-Base-7B",
}

DEFAULT_OUTPUT_PATH = Path("results/optimization/ar023_family_probe.json")

SCHEMA_VERSION = 1

# graph_enabled threshold: graph.ok AND free VRAM after the capture strictly
# above 1 GB (design.md §4.1.7 — "余量 >1GB").
_GRAPH_HEADROOM_THRESHOLD_MB = 1024.0

# Two representative short prompts for the prompt-adaptation face (the AR022
# benchmark prompt set is re-imported by the T005 bench script; the probe only
# verifies that encoding/padding semantics work end to end).
PROBE_PROMPTS: list[str] = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue.",
]

_SMOKE_NUM_TOKENS = 16
_SMOKE_NUM_STEPS = 16
_GRAPH_BATCH_ROWS = 4
_GRAPH_SEQ_LEN = 64

# AR026 production-graph-face knobs: the AR023 bench anchor construction
# (``split_layers=False``, cosine gate tau=0.95 — the ``_make_bench_mff``
# shape) plus the graph knobs the bench's graph modes set (fixed capacity
# 0.5, AR021 shape-keyed pool ON so a second ``(T, prompt_len)`` key never
# hits the pool=0 warn-once degrade). The mini speculative window is two
# candidates at probe scale (16 tokens).
_PROBE_GRAPH_GATE_TAU = 0.95
_PROBE_GRAPH_CANDIDATES = 2
_PROBE_GRAPH_TOP_K = 10
_PROBE_GRAPH_CAPACITY_RATIO = 0.5
_PROBE_GRAPH_POOL_SIZE = 4
_PROBE_GRAPH_CACHE_BUDGET = 64
_PROBE_GRAPH_SEED = 2026


# ---------------------------------------------------------------------------
# Pure-logic surface (CI-tested, no GPU / network / weights)
# ---------------------------------------------------------------------------


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA256 hex digest of ``path`` (streamed, large-file safe)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_conclusions(family_result: dict[str, Any]) -> dict[str, Any]:
    """Derive the family-level conclusions from a seven-face result dict.

    Args:
        family_result: per-face dicts keyed ``download/load/bf16/int8/smoke/
            graph/prompt``; each face at least ``{"ok": bool}``, the graph
            face additionally ``headroom_mb`` (float | None).

    Returns:
        ``dtype_choice`` (``"bf16"`` | ``"int8"`` | ``None``, bf16 first),
        ``graph_enabled`` (strict ``headroom_mb > 1024.0`` MB), ``degraded``
        (load failed OR both dtypes failed OR smoke failed) and
        ``degrade_reason`` (``None`` when not degraded).
    """
    load = family_result.get("load", {})
    bf16 = family_result.get("bf16", {})
    int8 = family_result.get("int8", {})
    smoke = family_result.get("smoke", {})
    graph = family_result.get("graph", {})

    if bf16.get("ok"):
        dtype_choice: str | None = "bf16"
    elif int8.get("ok"):
        dtype_choice = "int8"
    else:
        dtype_choice = None

    headroom = graph.get("headroom_mb")
    graph_enabled = (
        bool(graph.get("ok")) and headroom is not None and (headroom > _GRAPH_HEADROOM_THRESHOLD_MB)
    )

    reasons: list[str] = []
    if not load.get("ok"):
        reasons.append(f"load failed: {load.get('error') or load.get('note') or 'unknown'}")
    if not bf16.get("ok") and not int8.get("ok"):
        reasons.append(
            "no viable dtype: bf16 failed "
            f"({bf16.get('error') or 'unknown'}), int8 failed "
            f"({int8.get('error') or 'unknown'})"
        )
    if not smoke.get("ok"):
        reasons.append(f"smoke failed: {smoke.get('error') or 'unknown'}")

    return {
        "dtype_choice": dtype_choice,
        "graph_enabled": graph_enabled,
        "degraded": bool(reasons),
        "degrade_reason": "; ".join(reasons) if reasons else None,
    }


def run_faces_with_isolation(
    faces: dict[str, Callable[[], dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Run each face callable in isolation; one raising face never stops the rest.

    Args:
        faces: ordered mapping ``face name -> zero-arg callable`` returning the
            face result dict (with ``ok``).

    Returns:
        ``{face name: result dict}``; a face that raised records
        ``{"ok": False, "error": "<ExcType>: <message>"}``.
    """
    results: dict[str, dict[str, Any]] = {}
    for name, fn in faces.items():
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 — probe isolation by design
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return results


# The five faces that touch CUDA. A failed CUDA-graph capture corrupts the
# process-wide CUDA caching-allocator state (empirical [T001-2]: the first real
# run's graph-face INTERNAL ASSERT "captures_underway.empty()" poisoned the
# LLaDA prompt face and every Dream GPU face with the identical error), so
# these faces must run in fresh subprocesses — exception isolation alone
# cannot contain CUDA-state corruption.
CUDA_FACE_NAMES: tuple[str, ...] = (
    "bf16",
    "int8",
    "smoke",
    "graph",
    "graph_raw",
    "prompt",
)

# Worker protocol: the single-face worker prints its result as this
# sentinel-prefixed line so stray library prints on stdout cannot corrupt it.
FACE_RESULT_SENTINEL = "AR023_FACE_RESULT:"


def build_face_worker_command(
    family: str,
    face: str,
    repo: str,
    local_dir: str | os.PathLike[str],
) -> list[str]:
    """Build the single-face worker command (same interpreter, this module)."""
    return [
        sys.executable,
        "-m",
        "scripts.ar023_family_probe",
        "--_face-worker",
        family,
        face,
        "--repo",
        repo,
        "--local-dir",
        str(local_dir),
    ]


def parse_face_worker_output(stdout: str) -> dict[str, Any]:
    """Extract the LAST sentinel-prefixed JSON line from worker stdout.

    Raises:
        ValueError: no sentinel line present in the output.
    """
    for line in reversed(stdout.strip().splitlines()):
        stripped = line.strip()
        if stripped.startswith(FACE_RESULT_SENTINEL):
            parsed: dict[str, Any] = json.loads(stripped[len(FACE_RESULT_SENTINEL) :])
            return parsed
    raise ValueError("no sentinel line in face worker output")


def make_subprocess_face(
    family: str,
    face: str,
    repo: str,
    local_dir: str | os.PathLike[str],
    runner: Callable[..., Any] = subprocess.run,
) -> Callable[[], dict[str, Any]]:
    """Build a face callable that executes the face in a fresh subprocess.

    The worker prints ``FACE_RESULT_SENTINEL + json(result)`` as its last
    stdout line; any worker-level crash (import failure, hard exit) becomes an
    ``ok=False`` face result instead of an exception, so one face's CUDA-state
    corruption cannot invalidate the remaining faces.
    """

    def run() -> dict[str, Any]:
        cmd = build_face_worker_command(family, face, repo, local_dir)
        proc = runner(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-500:]
            return {
                "ok": False,
                "error": f"face worker rc={proc.returncode}: {tail}",
            }
        try:
            return parse_face_worker_output(proc.stdout)
        except ValueError as exc:
            return {"ok": False, "error": f"face worker output unusable: {exc}"}

    return run


def wrap_cuda_faces_with_subprocess(
    faces: dict[str, Callable[[], dict[str, Any]]],
    family: str,
    repo: str,
    local_dir: str | os.PathLike[str],
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Callable[[], dict[str, Any]]]:
    """Return a copy of ``faces`` with the CUDA faces subprocess-backed.

    The non-CUDA faces (download / load) keep their original callables by
    identity; each CUDA face is replaced by a ``make_subprocess_face`` wrapper.
    """
    wrapped: dict[str, Callable[[], dict[str, Any]]] = {}
    for name, fn in faces.items():
        if name in CUDA_FACE_NAMES:
            wrapped[name] = make_subprocess_face(family, name, repo, local_dir, runner=runner)
        else:
            wrapped[name] = fn
    return wrapped


def build_artifact(
    family_results: dict[str, dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the top-level probe artifact.

    Face dicts are preserved verbatim; the derived ``conclusions`` are appended
    per family (the input dicts are never mutated).
    """
    families: dict[str, dict[str, Any]] = {}
    for family, faces in family_results.items():
        families[family] = {
            **faces,
            "conclusions": derive_conclusions(faces),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "families": families,
    }


def write_results(artifact: dict[str, Any], output_path: str | os.PathLike[str]) -> None:
    """Write the artifact as UTF-8 JSON (parents created as needed)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)


def parse_repo_overrides(args: list[str]) -> dict[str, str]:
    """Parse ``family=repo`` override strings collected by argparse ``append``.

    Args:
        args: the collected value strings (NOT raw argv tokens).

    Returns:
        ``{family: repo}``; an empty input yields ``{}``.

    Raises:
        ValueError: a spec without ``=`` or with an empty side.
    """
    overrides: dict[str, str] = {}
    for spec in args:
        if "=" not in spec:
            raise ValueError(f"--repo-override expects family=repo, got {spec!r}")
        family, _, repo = spec.partition("=")
        family = family.strip()
        repo = repo.strip()
        if not family or not repo:
            raise ValueError(f"--repo-override expects non-empty family and repo, got {spec!r}")
        overrides[family] = repo
    return overrides


# ---------------------------------------------------------------------------
# Real-run machinery (heavy imports stay lazy inside the callables)
# ---------------------------------------------------------------------------


def _proxy_from_env() -> str | None:
    """Best-effort proxy URL from the standard environment variables."""
    for var in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        value = os.environ.get(var)
        if value:
            return value
    return None


def _curl_base_args() -> list[str]:
    args = ["curl.exe", "-sS"]
    proxy = _proxy_from_env()
    if proxy:
        args += ["--proxy", proxy]
    return args


class _FamilyContext:
    """Per-family probe context (paths and prompts for the real faces)."""

    def __init__(self, family: str, repo: str, local_dir: Path) -> None:
        self.family = family
        self.repo = repo
        self.local_dir = local_dir


def make_download_face(ctx: _FamilyContext) -> Callable[[], dict[str, Any]]:
    """curl.exe per-file download + SHA256 + byte size (AR019 P6 precedent)."""

    def run() -> dict[str, Any]:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        ctx.local_dir.mkdir(parents=True, exist_ok=True)
        api_cmd = _curl_base_args() + [
            "--max-time",
            "120",
            f"{endpoint}/api/models/{ctx.repo}",
        ]
        proc = subprocess.run(
            api_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "files": [],
                "error": f"curl API call failed (rc={proc.returncode}): {proc.stderr.strip()}",
            }
        try:
            info = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            return {"ok": False, "files": [], "error": f"API response not JSON: {exc}"}
        names = [
            item.get("rfilename") for item in info.get("siblings", []) if item.get("rfilename")
        ]
        has_safetensors = any(name.endswith(".safetensors") for name in names)
        skip_suffixes = (".h5", ".msgpack", ".ot", ".pth", ".gitattributes")
        selected = [
            name for name in names if not (has_safetensors and name.endswith(skip_suffixes))
        ]
        if not selected:
            return {"ok": False, "files": [], "error": "no files listed by the API"}
        files: list[dict[str, Any]] = []
        failures: list[str] = []
        for name in selected:
            target = ctx.local_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"{endpoint}/{ctx.repo}/resolve/main/{name}"
            dl = subprocess.run(
                _curl_base_args()
                + [
                    "-L",
                    "--fail",
                    "-C",
                    "-",
                    "--retry",
                    "3",
                    "--max-time",
                    "14400",
                    "-o",
                    str(target),
                    url,
                ]
            )
            if dl.returncode != 0:
                failures.append(f"{name}: curl rc={dl.returncode}")
                continue
            files.append(
                {
                    "name": name,
                    "sha256": sha256_file(target),
                    "bytes": target.stat().st_size,
                }
            )
        result: dict[str, Any] = {"ok": not failures, "files": files}
        if failures:
            result["error"] = "; ".join(failures[:10])
        return result

    return run


def make_load_face(ctx: _FamilyContext) -> Callable[[], dict[str, Any]]:
    """Config + tokenizer + remote-code compatibility (lightweight load)."""

    def run() -> dict[str, Any]:
        import transformers
        from transformers import AutoConfig, AutoTokenizer

        note = f"transformers=={transformers.__version__}"
        try:
            config = AutoConfig.from_pretrained(str(ctx.local_dir), trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(str(ctx.local_dir), trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001 — probe isolation by design
            return {"ok": False, "note": note, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": True,
            "note": (
                f"{note}; model_type={getattr(config, 'model_type', '?')}; "
                f"vocab={len(tokenizer)}"
            ),
        }

    return run


def _family_wrapper_cls(family: str) -> type:
    if family == "llada":
        from actfold.models.llada import LLaDAModel

        return LLaDAModel
    from actfold.models.dream import DreamModel

    return DreamModel


def _load_probe_model(ctx: _FamilyContext) -> tuple[Any, str]:
    """Load the family wrapper at the first viable dtype (bf16, then int8).

    Mirrors the ``dtype_choice`` priority. Raises ``RuntimeError`` when neither
    dtype loads — the caller (face isolation) records the error.
    """
    import torch

    wrapper_cls = _family_wrapper_cls(ctx.family)
    attempts: list[str] = []
    try:
        model = wrapper_cls(str(ctx.local_dir), torch_dtype=torch.bfloat16)
        model.model.cuda()
        return model, "bf16"
    except Exception as exc:  # noqa: BLE001 — probe isolation by design
        attempts.append(f"bf16: {type(exc).__name__}: {exc}")
        torch.cuda.empty_cache()
    try:
        model = wrapper_cls(str(ctx.local_dir), load_in_8bit=True, device_map={"": 0})
        return model, "int8"
    except Exception as exc:  # noqa: BLE001 — probe isolation by design
        attempts.append(f"int8: {type(exc).__name__}: {exc}")
        torch.cuda.empty_cache()
    raise RuntimeError("no viable dtype: " + "; ".join(attempts))


def _free_probe_model(model: Any) -> None:
    import torch

    del model
    torch.cuda.empty_cache()


def _short_forward_ok(model: Any) -> str | None:
    """One short forward; returns an error string or ``None`` on success."""
    import torch

    device = "cuda"
    ids = torch.randint(0, model.vocab_size, (1, 16), device=device)
    # bool mask: Dream's custom attention rejects int64 masks while LLaDA
    # tolerates them; bool is accepted by both (empirically verified, and
    # generate() internally builds a bool/float mask).
    logits = model.forward(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))
    if not torch.isfinite(logits.float().sum()).item():
        return "non-finite logits"
    return None


def make_bf16_face(ctx: _FamilyContext) -> Callable[[], dict[str, Any]]:
    """bf16 load + VRAM delta (``mem_get_info``) + short forward."""

    def run() -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {"ok": False, "vram_mb": None, "error": "CUDA unavailable"}
        wrapper_cls = _family_wrapper_cls(ctx.family)
        torch.cuda.empty_cache()
        free_before, _total = torch.cuda.mem_get_info()
        model = wrapper_cls(str(ctx.local_dir), torch_dtype=torch.bfloat16)
        try:
            model.model.cuda()
            free_after, _total = torch.cuda.mem_get_info()
            vram_mb = (free_before - free_after) / (1024 * 1024)
            forward_error = _short_forward_ok(model)
            if forward_error is not None:
                return {"ok": False, "vram_mb": round(vram_mb, 1), "error": forward_error}
            return {"ok": True, "vram_mb": round(vram_mb, 1)}
        finally:
            _free_probe_model(model)

    return run


def make_int8_face(ctx: _FamilyContext) -> Callable[[], dict[str, Any]]:
    """bitsandbytes availability + ``load_in_8bit`` + short forward."""

    def run() -> dict[str, Any]:
        import importlib.util

        if importlib.util.find_spec("bitsandbytes") is None:
            return {
                "available": False,
                "ok": False,
                "vram_mb": None,
                "error": "bitsandbytes is not installed",
            }
        import torch

        if not torch.cuda.is_available():
            return {"available": True, "ok": False, "vram_mb": None, "error": "CUDA unavailable"}
        wrapper_cls = _family_wrapper_cls(ctx.family)
        torch.cuda.empty_cache()
        free_before, _total = torch.cuda.mem_get_info()
        model = wrapper_cls(str(ctx.local_dir), load_in_8bit=True, device_map={"": 0})
        try:
            free_after, _total = torch.cuda.mem_get_info()
            vram_mb = (free_before - free_after) / (1024 * 1024)
            forward_error = _short_forward_ok(model)
            if forward_error is not None:
                return {
                    "available": True,
                    "ok": False,
                    "vram_mb": round(vram_mb, 1),
                    "error": forward_error,
                }
            return {"available": True, "ok": True, "vram_mb": round(vram_mb, 1)}
        finally:
            _free_probe_model(model)

    return run


def make_smoke_face(ctx: _FamilyContext) -> Callable[[], dict[str, Any]]:
    """16-token generation through the family reference sampler + decode."""

    def run() -> dict[str, Any]:
        from actfold.models.diffusion_sampler import DiffusionSampler
        from actfold.models.dream_sampler import DreamSampler, DreamSamplerConfig
        from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig

        model, dtype = _load_probe_model(ctx)
        try:
            sampler: DiffusionSampler
            if ctx.family == "llada":
                sampler = LLaDASampler(
                    model,
                    config=LLaDASamplerConfig(
                        num_steps=_SMOKE_NUM_STEPS, num_tokens=_SMOKE_NUM_TOKENS
                    ),
                )
            else:
                sampler = DreamSampler(
                    model,
                    config=DreamSamplerConfig(
                        num_steps=_SMOKE_NUM_STEPS, num_tokens=_SMOKE_NUM_TOKENS
                    ),
                )
            prompt = PROBE_PROMPTS[0]
            prompt_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            output = sampler.sample(prompt_ids)
            text = model.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
            mask_token_id = DiffusionSampler._resolve_mask_token_id(model.model, model.tokenizer)
            if not text.strip():
                return {"ok": False, "tokens": text, "error": "empty decode"}
            return {
                "ok": True,
                "tokens": text[:80],
                "mask_token_id": mask_token_id,
                "dtype": dtype,
            }
        finally:
            _free_probe_model(model)

    return run


def _drive_production_graph_capture(ctx: _FamilyContext) -> dict[str, Any]:
    """Production graph-face driver (AR026 T004, design.md §4.2.2 point 4).

    Builds the AR023-anchor MFF over the loaded checkpoint with the graph
    knobs on (fixed capacity + AR021 shape-keyed pool) and drives ONE mini
    speculative window through the family sampler, so the #37 preconditions
    (batch-1 parent cache, same-shape ``[N, T]`` ``branch_ids`` call,
    mask/pe/position_ids object pinning) thread natively — the face observes
    the PRODUCTION capture path, never a raw-model proxy. ``graph_enabled``
    conclusions read only this face; the pre-AR026 raw capture survives as
    the ``graph_raw`` diagnostic. The CUDA check happens FIRST, before any
    model load.
    """
    import torch

    if not torch.cuda.is_available():
        return {
            "ok": False,
            "capture_delta_mb": None,
            "headroom_mb": None,
            "diagnostics": [],
            "error": "CUDA unavailable",
        }
    from actfold.core import ActivationCache, SimilarityGate
    from actfold.core.cuda_graph import capture_diagnostics_snapshot
    from actfold.models.architecture_utils import ManualFoldedForward
    from actfold.models.dream_sampler import DreamSampler, DreamSamplerConfig
    from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig

    model, dtype = _load_probe_model(ctx)
    try:
        free_before, _total = torch.cuda.mem_get_info()
        cache = ActivationCache(max_entries_per_layer=_PROBE_GRAPH_CACHE_BUDGET, device="cuda")
        gate = SimilarityGate(tau=_PROBE_GRAPH_GATE_TAU, metric="cosine")
        mff = ManualFoldedForward(
            model.model,
            cache=cache,
            gate=gate,
            scheduler=None,
            split_layers=False,
            use_cuda_graph=True,
            graph_capacity_ratio=_PROBE_GRAPH_CAPACITY_RATIO,
            graph_runner_pool_size=_PROBE_GRAPH_POOL_SIZE,
        )
        sampler: Any
        if ctx.family == "llada":
            config: Any = LLaDASamplerConfig(
                num_steps=_SMOKE_NUM_STEPS,
                num_tokens=_SMOKE_NUM_TOKENS,
                seed=_PROBE_GRAPH_SEED,
                num_block_candidates=_PROBE_GRAPH_CANDIDATES,
                candidate_top_k=_PROBE_GRAPH_TOP_K,
            )
            sampler = LLaDASampler(model, config=config)
        else:
            config = DreamSamplerConfig(
                num_steps=_SMOKE_NUM_TOKENS,
                num_tokens=_SMOKE_NUM_TOKENS,
                seed=_PROBE_GRAPH_SEED,
                speculative_steps=8,
                num_block_candidates=_PROBE_GRAPH_CANDIDATES,
                candidate_top_k=_PROBE_GRAPH_TOP_K,
            )
            sampler = DreamSampler(model, config=config)
        prompt_ids = model.tokenizer(PROBE_PROMPTS[0], return_tensors="pt").input_ids.cuda()
        sampler.sample(prompt_ids, folded_model=mff)

        stats = mff.graph_pool_stats
        diagnostics = capture_diagnostics_snapshot()
        free_after, _total = torch.cuda.mem_get_info()
        capture_delta_mb = (free_before - free_after) / (1024 * 1024)
        headroom_mb = free_after / (1024 * 1024)
        captures = 0
        for section in ("serial", "batched"):
            for key, entry in stats.get(section, {}).items():
                if key == "pool_size":
                    continue
                captures += int(entry.get("captures", 0))
        return {
            "ok": captures > 0 and not diagnostics,
            "capture_delta_mb": round(capture_delta_mb, 1),
            "headroom_mb": round(headroom_mb, 1),
            "dtype": dtype,
            "graph_captures": captures,
            "degradation_fallback_count": stats.get("degradation_fallback_count", 0),
            "structural_eager_count": stats.get("structural_eager_count", 0),
            "capture_diagnostics_overflow": stats.get("capture_diagnostics_overflow", 0),
            "diagnostics": diagnostics,
            "note": (
                "production path: AR023-anchor MFF (split_layers=False, cosine "
                "gate) + graph knobs + one family-sampler mini speculative "
                "window (AR026 T004)"
            ),
        }
    finally:
        _free_probe_model(model)


def make_graph_face(ctx: _FamilyContext) -> Callable[[], dict[str, Any]]:
    """Production-path graph face (AR026): routes to the production driver."""

    def run() -> dict[str, Any]:
        return _drive_production_graph_capture(ctx)

    return run


def make_graph_raw_face(ctx: _FamilyContext) -> Callable[[], dict[str, Any]]:
    """Raw-model ad-hoc capture — DIAGNOSTIC face (the pre-AR026 graph face).

    Measures ``capture_delta_mb`` (VRAM consumed by the capture) and
    ``headroom_mb`` (free VRAM after the capture). A raw-model capture
    isolates REMOTE-CODE capture unsafety from the production machinery (the
    AR023 probe failure signatures were recorded here); it never gates the
    ``graph_enabled`` conclusion (AR026 — conclusions read only the
    production ``graph`` face). Capture-unsafe remote code records
    ``ok: False`` without failing the probe.
    """

    def run() -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {
                "ok": False,
                "capture_delta_mb": None,
                "headroom_mb": None,
                "error": "CUDA unavailable",
            }
        model, dtype = _load_probe_model(ctx)
        try:
            rows, seq_len = _GRAPH_BATCH_ROWS, _GRAPH_SEQ_LEN
            static_ids = torch.randint(0, model.vocab_size, (rows, seq_len), device="cuda")
            static_mask = torch.ones(rows, seq_len, dtype=torch.bool, device="cuda")

            def _forward() -> Any:
                return model.model(
                    input_ids=static_ids, attention_mask=static_mask, return_dict=True
                )

            stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    _forward()
            torch.cuda.current_stream().wait_stream(stream)

            free_before, _total = torch.cuda.mem_get_info()
            graph = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
            with torch.cuda.graph(graph):
                _forward()
            free_after, _total = torch.cuda.mem_get_info()
            capture_delta_mb = (free_before - free_after) / (1024 * 1024)
            headroom_mb = free_after / (1024 * 1024)
            return {
                "ok": True,
                "capture_delta_mb": round(capture_delta_mb, 1),
                "headroom_mb": round(headroom_mb, 1),
                "dtype": dtype,
                "note": "raw-model [N,T] forward capture (MFF batched-capture proxy)",
            }
        finally:
            _free_probe_model(model)

    return run


def make_prompt_face(ctx: _FamilyContext) -> Callable[[], dict[str, Any]]:
    """Prompt encoding/padding walk-through + folding-transparency probes.

    LLaDA: right-padded canvas + 2-D ``[T, T]`` bool attention-mask acceptance
    (the AR021 fold-ctx mask form). Dream: left-padded canvas + ``position_ids``
    kwarg acceptance (the T004 folded-branch threading prerequisite).
    """

    def run() -> dict[str, Any]:
        import torch

        from actfold.models.diffusion_sampler import DiffusionSampler
        from actfold.models.sampling_utils import (
            build_left_padded_canvas,
            build_right_padded_canvas,
            compute_position_ids,
        )

        model, dtype = _load_probe_model(ctx)
        try:
            tokenizer = model.tokenizer
            mask_token_id = DiffusionSampler._resolve_mask_token_id(model.model, tokenizer)
            eos_token_id = tokenizer.eos_token_id
            if mask_token_id is None or eos_token_id is None:
                return {"ok": False, "error": "mask/eos token resolution failed"}
            notes: list[str] = []
            for prompt in PROBE_PROMPTS:
                ids = tokenizer(prompt, return_tensors="pt").input_ids
                if ctx.family == "llada":
                    x, attn, _lens, _maxlen = build_right_padded_canvas(
                        [ids[0]], _SMOKE_NUM_TOKENS, eos_token_id, mask_token_id
                    )
                    valid_end = int(attn[0].sum())
                    seq_len = x.shape[1]
                    mask2d = torch.arange(seq_len, device="cuda")[None, :] < valid_end
                    logits = model.model(
                        input_ids=x.cuda(), attention_mask=mask2d, return_dict=True
                    ).logits
                    notes.append(
                        f"right-pad T={seq_len} valid_end={valid_end} "
                        f"2d_mask_accepted shape={tuple(logits.shape)}"
                    )
                else:
                    x, attn, _seq_lens = build_left_padded_canvas(
                        [ids[0]], _SMOKE_NUM_TOKENS, eos_token_id, mask_token_id
                    )
                    position_ids = compute_position_ids(attn).cuda()
                    logits = model.model(
                        input_ids=x.cuda(),
                        attention_mask=attn.bool().cuda(),
                        position_ids=position_ids,
                        return_dict=True,
                    ).logits
                    notes.append(
                        f"left-pad T={x.shape[1]} position_ids_accepted "
                        f"shape={tuple(logits.shape)}"
                    )
                if not torch.isfinite(logits.float().sum()).item():
                    return {"ok": False, "note": "; ".join(notes), "error": "non-finite logits"}
            return {"ok": True, "note": "; ".join(notes), "dtype": dtype}
        finally:
            _free_probe_model(model)

    return run


def build_family_faces(ctx: _FamilyContext) -> dict[str, Callable[[], dict[str, Any]]]:
    """Assemble the seven ordered face callables for one family."""
    return {
        "download": make_download_face(ctx),
        "load": make_load_face(ctx),
        "bf16": make_bf16_face(ctx),
        "int8": make_int8_face(ctx),
        "smoke": make_smoke_face(ctx),
        "graph": make_graph_face(ctx),
        "graph_raw": make_graph_raw_face(ctx),
        "prompt": make_prompt_face(ctx),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the seven-face probe for both families and write the artifact."""
    apply_hf_env()
    parser = argparse.ArgumentParser(
        prog="ar023_family_probe",
        description="AR023 cross-family model probe (seven faces per family).",
    )
    add_hf_env_arguments(parser)
    parser.add_argument(
        "--_face-worker",
        nargs=2,
        metavar=("FAMILY", "FACE"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--repo", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--local-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--repo-override",
        action="append",
        default=[],
        metavar="FAMILY=REPO",
        help="Override the probe repo id for a family (repeatable).",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Root directory for downloaded checkpoints " "(default: $HF_HOME/ar023_models).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Artifact output path (default: %(default)s).",
    )
    args = parser.parse_args()

    if args._face_worker is not None:
        family, face = args._face_worker
        if args.repo is None or args.local_dir is None:
            parser.error("--_face-worker requires --repo and --local-dir")
        ctx = _FamilyContext(family, args.repo, args.local_dir)
        faces = build_family_faces(ctx)
        if face not in faces:
            result: dict[str, Any] = {"ok": False, "error": f"unknown face {face!r}"}
        else:
            try:
                result = faces[face]()
            except Exception as exc:  # noqa: BLE001 — probe isolation by design
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(FACE_RESULT_SENTINEL + json.dumps(result, ensure_ascii=False), flush=True)
        return 0

    overrides = parse_repo_overrides(args.repo_override)
    repos = {**PROBE_REPOS, **overrides}
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    models_root = args.models_root or Path(hf_home) / "ar023_models"

    meta: dict[str, Any] = {
        "probe": "ar023_family_probe",
        "models_root": str(models_root),
        "repos": dict(repos),
        "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    }

    family_results: dict[str, dict[str, Any]] = {}
    for family, repo in repos.items():
        ctx = _FamilyContext(family, repo, models_root / family)
        print(f"[ar023-probe] family={family} repo={repo}", flush=True)
        faces = build_family_faces(ctx)
        family_results[family] = run_faces_with_isolation(
            wrap_cuda_faces_with_subprocess(faces, family, repo, ctx.local_dir)
        )

    artifact = build_artifact(family_results, meta)
    write_results(artifact, args.output)
    for family, faces in artifact["families"].items():
        conclusions = faces["conclusions"]
        print(
            f"[ar023-probe] {family}: dtype_choice={conclusions['dtype_choice']} "
            f"graph_enabled={conclusions['graph_enabled']} "
            f"degraded={conclusions['degraded']}",
            flush=True,
        )
    print(f"[ar023-probe] artifact written to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
