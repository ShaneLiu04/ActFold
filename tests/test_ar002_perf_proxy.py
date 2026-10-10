"""Red-phase tests for AR002/T009: the performance proxy validation package.

Three contracts are pinned:

1. ``scripts/ar002_graph_bench.py`` (BS-007) — a CUDA benchmark script,
   importable as ``scripts.ar002_graph_bench`` and runnable as
   ``python -m scripts.ar002_graph_bench``, exposing ``run_bench() -> dict``
   and ``write_results(data, path) -> Path`` (creates parent directories).
   ``run_bench()`` measures eager vs CUDA-graph child-step latency and
   per-step CUDA kernel-launch counts over a fixed-shape verification loop
   on a synthetic LLaMA-layout decoder.
2. UT-006b — with the Triton fused gate ENABLED vs FORCIBLY DISABLED
   (``actfold.core.fused_ops._TRITON_GATE_DISABLED = True``), a single folded
   mixed-path child forward with ``B*T >= 1024`` must launch at least 3 FEWER
   CUDA kernels in the enabled configuration (the gate cosine chain plus the
   separate ``mask.sum()`` reduction collapse into one kernel).
3. srs §3.1 chain-level claim — a full eager ``ManualFoldedForward`` child
   pass over L mixed layers performs at most L host readbacks.

Plus a demo-baseline regression pin (BS-005 / srs §4 NFR): ``demo.py`` must
keep reproducing the 85.5% FLOPs reduction / <= 2.35e-03 MSE / 93.75%
stable-token-ratio baseline.

The benchmark module is imported lazily inside each test so the Red state —
``scripts.ar002_graph_bench`` not existing yet — fails tests individually
with ``ModuleNotFoundError`` instead of aborting module collection.

Self-contained per repo convention: the small LLaMA-layout decoder below is
copied verbatim from ``tests/test_cuda_graph.py`` (multiplicative-mask
``_MaskableAttention`` included) because the tests package has no
``__init__.py`` to import shared fixture model classes from.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EPS = 1e-8
_TAU = 0.95
_VOCAB = 100
_HIDDEN = 32
_LAYERS = 2
_SEED = 2024


# ---------------------------------------------------------------------------
# Self-contained synthetic decoder (LlamaLikeModel layout) — verbatim copy
# from tests/test_cuda_graph.py
# ---------------------------------------------------------------------------


class _MaskableAttention(nn.Module):
    """Token-wise attention stand-in that consumes the attention mask.

    The mask is applied MULTIPLICATIVELY: an additive per-token constant
    would be cancelled by the final LayerNorm's shift invariance and thus be
    unobservable at the logits level.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        out = hidden_states * 0.5
        if attention_mask is not None:
            out = out * attention_mask.to(out.dtype).unsqueeze(-1)
        return out, None


class _LlamaLikeLayer(nn.Module):
    """Pre-norm decoder layer with a detectable ``post_attention_layernorm->mlp`` chain."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = _MaskableAttention()
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


class LlamaLikeModel(nn.Module):
    """Mock LLaMA-layout decoder: embed -> layers -> model.norm -> lm_head."""

    def __init__(
        self,
        vocab_size: int = _VOCAB,
        hidden_dim: int = _HIDDEN,
        num_layers: int = _LAYERS,
    ) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": vocab_size})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model.layers = nn.ModuleList([_LlamaLikeLayer(hidden_dim) for _ in range(num_layers)])
        self.model.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------


def _build_model(
    device: str,
    num_layers: int = _LAYERS,
    hidden_dim: int = _HIDDEN,
    seed: int = _SEED,
) -> nn.Module:
    """Deterministically build the decoder on ``device``."""
    torch.manual_seed(seed)
    model = LlamaLikeModel(_VOCAB, hidden_dim, num_layers).to(device)
    model.eval()
    return model


def _make_mff(
    model: nn.Module,
    device: str,
    cache_budget: int = 64,
    use_cuda_graph: bool = False,
    graph_capacity_ratio: float = 0.5,
) -> ManualFoldedForward:
    """Fresh eager/graph ``ManualFoldedForward`` (own cache + cosine gate).

    ``split_layers=True, split_min_tokens=1`` keeps the split FFN chain
    engaged at every shape so the folded mixed path really exercises the
    per-layer gather/merge machinery under measurement.
    """
    cache = ActivationCache(max_entries_per_layer=cache_budget, device=device)
    gate = SimilarityGate(tau=_TAU, metric="cosine", eps=_EPS)
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        split_layers=True,
        split_min_tokens=1,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=graph_capacity_ratio,
    )


def _change_tokens(tokens: torch.Tensor, positions: list[int], offset: int = 1) -> torch.Tensor:
    """Clone ``tokens`` and change the flat ``positions`` to a different id."""
    child = tokens.clone()
    flat = child.reshape(-1)
    for pos in positions:
        flat[pos] = (flat[pos] + offset) % _VOCAB
    return child


def _profiled_cuda_launch_count(step_fn: Any) -> int:
    """Run ``step_fn`` under the CUDA profiler and count kernel launches.

    The profiler context wraps ONLY the measured step (the caller warms up /
    runs the parent pass outside it) and a ``torch.cuda.synchronize()`` runs
    before entering the block so no prior work leaks into the record. The
    launch count is the sum of event counts with CUDA device type.
    """
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        step_fn()
        torch.cuda.synchronize()
    return sum(
        evt.count
        for evt in prof.key_averages()
        if evt.device_type == torch.profiler.DeviceType.CUDA
    )


def _cuda_profiler_supported() -> bool:
    """Probe whether this build can record CUDA events at all.

    Some Windows torch builds are compiled with ``LIBKINETO_NOCUPTI``: the
    profiler enters but records zero CUDA events. Kernel-launch assertions
    must skip (and benchmarks must report ``None`` counts with an explicit
    note) on such hosts instead of fabricating numbers.
    """
    if not torch.cuda.is_available():
        return False
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
# (1) BS-007 benchmark script contract
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bench_module_contract(tmp_path: Path) -> None:
    """``scripts.ar002_graph_bench`` exposes run_bench/write_results/DEFAULT_OUTPUT_PATH."""
    from scripts.ar002_graph_bench import DEFAULT_OUTPUT_PATH, run_bench, write_results

    data = run_bench()
    assert isinstance(data, dict)

    required_types: dict[str, type] = {
        "device": str,
        "batch": int,
        "seq": int,
        "num_layers": int,
        "eager_per_step_ms": float,
        "graph_per_step_ms": float,
        "graph_validated_steps": int,
    }
    for key, expected in required_types.items():
        assert key in data, f"run_bench() result missing key {key!r}"
        assert isinstance(data[key], expected), (
            f"run_bench() result key {key!r} must be {expected.__name__}, "
            f"got {type(data[key]).__name__}: {data[key]!r}"
        )

    # Kernel-launch counts are ints on CUPTI-capable hosts; on
    # LIBKINETO_NOCUPTI builds they must be None with an explicit flag and
    # note (honest measurement, never fabricated).
    assert "kernel_launches_supported" in data
    for key in ("eager_kernel_launches", "graph_kernel_launches"):
        assert key in data
        if data["kernel_launches_supported"]:
            assert isinstance(data[key], int) and data[key] >= 1
        else:
            assert data[key] is None
            assert isinstance(data.get("kernel_launch_note", ""), str)

    # Timing fields are positive; validated steps are >= 1
    # (the graph path must actually have been used, not silently degraded).
    assert data["eager_per_step_ms"] > 0.0
    assert data["graph_per_step_ms"] > 0.0
    assert data["graph_validated_steps"] >= 1
    # The bench shape must engage the Triton fused gate (B*T >= 1024).
    assert data["batch"] >= 1
    assert data["seq"] >= 1
    assert data["num_layers"] >= 1
    assert data["batch"] * data["seq"] >= 1024

    # write_results creates parent directories and round-trips the dict.
    out = write_results(data, tmp_path / "sub" / "bench.json")
    out_path = Path(out)
    assert out_path.exists()
    assert out_path.parent.is_dir()
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed == data
    for key in ("eager_per_step_ms", "graph_per_step_ms"):
        assert key in parsed

    # The default CLI output path is results/optimization/ar002_graph_bench.json.
    default_path = Path(DEFAULT_OUTPUT_PATH)
    assert default_path.name == "ar002_graph_bench.json"
    assert "results" in default_path.parts
    assert "optimization" in default_path.parts


# ---------------------------------------------------------------------------
# (2) BS-007 design judgement + evidence artifact
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bs007_graph_replay_not_slower(tmp_path: Path) -> None:
    """Graph replay per-step latency must not exceed eager per-step latency.

    The evidence copy is written under ``tmp_path`` so test runs never
    overwrite the canonical repo artifact
    (``results/optimization/ar002_graph_bench.json``), which is produced by
    running ``python -m scripts.ar002_graph_bench`` — otherwise every test
    run would make the committed numbers drift from the documented ones.
    """
    from scripts.ar002_graph_bench import run_bench, write_results

    data = run_bench()
    assert data["graph_per_step_ms"] <= data["eager_per_step_ms"], (
        f"CUDA graph replay per-step ({data['graph_per_step_ms']:.4f} ms) must not "
        f"exceed eager per-step ({data['eager_per_step_ms']:.4f} ms)"
    )

    artifact = tmp_path / "ar002_graph_bench.json"
    returned = Path(write_results(data, artifact))
    assert returned.exists()
    parsed = json.loads(artifact.read_text(encoding="utf-8"))
    assert "eager_per_step_ms" in parsed
    assert "graph_per_step_ms" in parsed


# ---------------------------------------------------------------------------
# (3) UT-006b: fused-gate kernel-launch reduction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ut006b_launch_count_reduction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fused gate ON launches >= 3 fewer CUDA kernels than gate OFF (B*T >= 1024)."""
    if not _cuda_profiler_supported():
        pytest.skip(
            "this torch build cannot record CUDA profiler events "
            "(LIBKINETO_NOCUPTI); run on a CUPTI-capable host"
        )
    from actfold.core import fused_ops

    device = "cuda"
    batch, seq, hidden, num_layers = 2, 512, 256, 1  # B*T = 1024 -> Triton gate
    model = _build_model(device, num_layers=num_layers, hidden_dim=hidden)
    parent_tokens = torch.randint(0, _VOCAB, (batch, seq), device=device)
    child_tokens = _change_tokens(parent_tokens, [100, 500, 900])  # D=3, mixed path

    def _measure(triton_disabled: bool) -> int:
        """Profiled mixed child-pass launch count; identical runs apart from the flag."""
        monkeypatch.setattr(fused_ops, "_TRITON_GATE_DISABLED", triton_disabled)
        mff = _make_mff(model, device, cache_budget=4096)
        with torch.no_grad():
            # Parent pass (eager) and a warmup child pass outside the profiler
            # so Triton JIT compilation never pollutes the measured record.
            mff(parent_tokens, branch_id="parent")
            mff(child_tokens, branch_id="warmup", parent_branch_id="parent")
        return _profiled_cuda_launch_count(
            lambda: mff(child_tokens, branch_id="child", parent_branch_id="parent")
        )

    enabled_launches = _measure(triton_disabled=False)
    disabled_launches = _measure(triton_disabled=True)
    # monkeypatch auto-restores _TRITON_GATE_DISABLED at teardown.

    assert enabled_launches <= disabled_launches - 3, (
        f"fused gate enabled must launch >= 3 fewer CUDA kernels than the "
        f"disabled fallback: enabled={enabled_launches}, disabled={disabled_launches}"
    )


# ---------------------------------------------------------------------------
# (4) srs §3.1: at most one host readback per layer on the eager child pass
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sync_count_e2e_at_most_one_per_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eager child forward over L=3 mixed layers performs <= 3 host readbacks."""
    device = "cuda"
    num_layers, batch, seq = 3, 1, 8
    model = _build_model(device, num_layers=num_layers, hidden_dim=_HIDDEN)
    mff = _make_mff(model, device, cache_budget=64)  # eager: use_cuda_graph=False
    parent_tokens = torch.randint(0, _VOCAB, (batch, seq), device=device)
    child_tokens = _change_tokens(parent_tokens, [3, 6])  # 2 changed tokens, mixed path

    # Parent pass outside the patch window (NOT counted).
    with torch.no_grad():
        mff(parent_tokens, branch_id="parent")

    counts = {"item": 0, "int": 0, "cpu": 0}
    orig_item = torch.Tensor.item
    orig_int = torch.Tensor.__int__
    orig_cpu = torch.Tensor.cpu

    def _count_item(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        counts["item"] += 1
        return orig_item(self, *args, **kwargs)  # type: ignore[arg-type]

    def _count_int(self: torch.Tensor, *args: object, **kwargs: object) -> int:
        counts["int"] += 1
        return orig_int(self, *args, **kwargs)  # type: ignore[arg-type,return-value]

    def _count_cpu(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        counts["cpu"] += 1
        return orig_cpu(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.Tensor, "item", _count_item)
    monkeypatch.setattr(torch.Tensor, "__int__", _count_int, raising=False)
    monkeypatch.setattr(torch.Tensor, "cpu", _count_cpu)

    # ONE child forward with a small divergence, counted.
    with torch.no_grad():
        child_logits = mff(child_tokens, branch_id="child", parent_branch_id="parent")

    total = counts["item"] + counts["int"] + counts["cpu"]
    assert isinstance(child_logits, torch.Tensor)
    assert tuple(child_logits.shape) == (batch, seq, _VOCAB)
    assert total <= num_layers, (
        f"eager child forward over {num_layers} layers must perform at most one "
        f"host readback per layer; observed {total}: "
        f"item={counts['item']}, int={counts['int']}, cpu={counts['cpu']}"
    )


# ---------------------------------------------------------------------------
# (5) BS-005 demo baseline regression pin (srs §4 NFR)
# ---------------------------------------------------------------------------


def test_bs005_demo_baseline_regression() -> None:
    """demo.py keeps the 85.5% / 2.35e-03 / 93.75% baseline (device auto-selected)."""
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "demo.py"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"demo.py exited with {proc.returncode}\nstdout:\n{proc.stdout}\n" f"stderr:\n{proc.stderr}"
    )
    stdout = proc.stdout

    reduction = re.search(r"Total FLOPs reduction: (\d+(?:\.\d+)?)%", stdout)
    assert reduction is not None, f"no FLOPs-reduction line in demo output:\n{stdout}"
    assert (
        85.4 <= float(reduction.group(1)) <= 85.6
    ), f"FLOPs reduction {reduction.group(1)}% outside the pinned [85.4, 85.6] band"

    mse = re.search(r"Output equivalence \(MSE\): (\d+\.\d+e-\d+)", stdout)
    assert mse is not None, f"no MSE line in demo output:\n{stdout}"
    assert float(mse.group(1)) <= 2.5e-3, f"MSE {mse.group(1)} exceeds 2.5e-3"

    ratio = re.search(r"Estimated stable token ratio: (\d+(?:\.\d+)?)%", stdout)
    assert ratio is not None, f"no stable-token-ratio line in demo output:\n{stdout}"
    assert (
        93.5 <= float(ratio.group(1)) <= 94.0
    ), f"stable token ratio {ratio.group(1)}% outside the pinned [93.5, 94.0] band"
