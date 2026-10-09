"""AR002/T009 (BS-007): CUDA graph verification-loop benchmark.

Measures the fixed-shape diffusion verification folded forward on BOTH paths:

* **eager** — ``ManualFoldedForward(use_cuda_graph=False, split_layers=True)``:
  every child step runs the eager folded forward (per-layer fused gate,
  padded/exact divergent gather, merge, one count readback per layer).
* **graph** — ``ManualFoldedForward(use_cuda_graph=True)``: the first child
  step captures the CUDA graph (via :class:`actfold.core.cuda_graph.FoldedGraphRunner`)
  and later steps replay it (copy-in static buffers, one replay, one count
  readback for budget validation).

The design judgement (design.md BS-007): replay per-step wall-clock must not
exceed eager per-step wall-clock, and the measurement is recorded as the
evidence artifact ``results/optimization/ar002_graph_bench.json``.

Kernel-launch counts are recorded on CUPTI-capable hosts; Windows torch builds
compiled with ``LIBKINETO_NOCUPTI`` cannot record CUDA profiler events, in
which case the counts are ``null`` with an explicit note (never fabricated).

Run::

    python -m scripts.ar002_graph_bench [--out PATH] [--steps N] [--warmup N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward

DEFAULT_OUTPUT_PATH = Path("results/optimization/ar002_graph_bench.json")

_BATCH = 2
_SEQ = 512  # B*T = 1024 -> the Triton fused gate dispatches inside the graph
_HIDDEN = 256
_LAYERS = 4
_VOCAB = 1000
_TAU = 0.95
_EPS = 1e-8
_CAPACITY_RATIO = 0.5


# ---------------------------------------------------------------------------
# Synthetic LLaMA-layout decoder (detectable post_attention_layernorm->mlp chain)
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
# Measurement helpers
# ---------------------------------------------------------------------------
def _cuda_profiler_supported() -> bool:
    """Probe whether this build records CUDA profiler events (CUPTI)."""
    try:
        x = torch.ones(4, device="cuda")
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            x = x + 1
            torch.cuda.synchronize()
        return any(
            evt.device_type == torch.profiler.DeviceType.CUDA
            for evt in prof.key_averages()
        )
    except Exception:  # noqa: BLE001 - probe must never raise
        return False


def _profiled_cuda_launch_count(step_fn: Callable[[], Any]) -> int:
    """Count CUDA kernel launches of one step under the profiler."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        step_fn()
        torch.cuda.synchronize()
    return sum(
        evt.count
        for evt in prof.key_averages()
        if evt.device_type == torch.profiler.DeviceType.CUDA
    )


def _timed_step_ms(step_fn: Callable[[], Any]) -> float:
    """Wall-clock one step in milliseconds (device synchronized)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _next_child_tokens(tokens: torch.Tensor, step: int) -> torch.Tensor:
    """Chain a new child: change one token position per step."""
    child = tokens.clone()
    flat = child.reshape(-1)
    pos = (3 * step + 7) % flat.numel()
    flat[pos] = (flat[pos] + 1) % _VOCAB
    return child


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


def run_bench(warmup_steps: int = 5, measured_steps: int = 20) -> dict[str, Any]:
    """Run the fixed-shape verification-loop benchmark on both paths.

    Returns:
        A dict with per-step wall-clock means for the eager and graph paths,
        kernel-launch counts (``None`` on non-CUPTI builds, with an explicit
        note), and the shape metadata. The graph path's ``graph_validated_steps``
        counts measured steps whose divergent budget validated (replay kept).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("run_bench requires a CUDA device")

    torch.manual_seed(2024)
    model = _Decoder().to("cuda")
    model.eval()
    device = torch.cuda.get_device_name(0)

    counts_supported = _cuda_profiler_supported()

    # --- eager path -------------------------------------------------------
    eager_mff = _make_mff(model, use_cuda_graph=False)
    with torch.no_grad():
        parent_tokens = torch.randint(0, _VOCAB, (_BATCH, _SEQ), device="cuda")
        eager_mff(parent_tokens, branch_id="parent")

        eager_times: list[float] = []
        prev_branch = "parent"
        prev_tokens = parent_tokens
        for step in range(warmup_steps + measured_steps):
            step_tokens = _next_child_tokens(prev_tokens, step)
            branch = f"eager_{step}"
            mff_call = _one_step(eager_mff, step_tokens, branch, prev_branch)
            elapsed = _timed_step_ms(mff_call)
            if step >= warmup_steps:
                eager_times.append(elapsed)
            prev_branch, prev_tokens = branch, step_tokens

        eager_launches = (
            _profiled_cuda_launch_count(
                _one_step(
                    eager_mff,
                    _next_child_tokens(prev_tokens, 0),
                    "eager_profiled",
                    prev_branch,
                )
            )
            if counts_supported
            else None
        )

    # --- graph path -------------------------------------------------------
    graph_mff = _make_mff(model, use_cuda_graph=True)
    validated_steps = 0
    with torch.no_grad():
        graph_mff(parent_tokens, branch_id="parent")

        graph_times: list[float] = []
        prev_branch = "parent"
        prev_tokens = parent_tokens
        for step in range(warmup_steps + measured_steps):
            step_tokens = _next_child_tokens(prev_tokens, step)
            branch = f"graph_{step}"
            mff_call = _one_step(graph_mff, step_tokens, branch, prev_branch)
            elapsed = _timed_step_ms(mff_call)
            runner = graph_mff.graph_runner
            if step >= warmup_steps:
                graph_times.append(elapsed)
                if runner is not None and not runner.budget_exceeded:
                    validated_steps += 1
            prev_branch, prev_tokens = branch, step_tokens

        graph_launches = (
            _profiled_cuda_launch_count(
                _one_step(
                    graph_mff,
                    _next_child_tokens(prev_tokens, 0),
                    "graph_profiled",
                    prev_branch,
                )
            )
            if counts_supported
            else None
        )

    return {
        "device": device,
        "batch": _BATCH,
        "seq": _SEQ,
        "num_layers": _LAYERS,
        "capacity_ratio": _CAPACITY_RATIO,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "eager_per_step_ms": sum(eager_times) / len(eager_times),
        "graph_per_step_ms": sum(graph_times) / len(graph_times),
        "kernel_launches_supported": counts_supported,
        "eager_kernel_launches": eager_launches,
        "graph_kernel_launches": graph_launches,
        "graph_validated_steps": validated_steps,
        "kernel_launch_note": (
            "counts recorded via torch.profiler (CUPTI)"
            if counts_supported
            else "this torch build cannot record CUDA profiler events "
            "(LIBKINETO_NOCUPTI); kernel-launch counts omitted rather than "
            "fabricated — rerun on a CUPTI-capable host to record them"
        ),
    }


def _one_step(
    mff: ManualFoldedForward,
    tokens: torch.Tensor,
    branch: str,
    parent_branch: str,
) -> Callable[[], Any]:
    """Bind one verification child step as a zero-argument callable."""

    def step() -> Any:
        return mff(tokens, branch_id=branch, parent_branch_id=parent_branch)

    return step


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
    parser.add_argument("--steps", type=int, default=20, help="Measured steps per path")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps per path")
    args = parser.parse_args()

    data = run_bench(warmup_steps=args.warmup, measured_steps=args.steps)
    out = write_results(data, args.out)
    print(f"device: {data['device']}")
    print(f"shape: B={data['batch']} T={data['seq']} L={data['num_layers']}")
    print(f"eager per-step: {data['eager_per_step_ms']:.4f} ms")
    print(f"graph per-step: {data['graph_per_step_ms']:.4f} ms")
    print(f"graph validated steps: {data['graph_validated_steps']}/{data['measured_steps']}")
    if data["kernel_launches_supported"]:
        print(f"eager kernel launches: {data['eager_kernel_launches']}")
        print(f"graph kernel launches: {data['graph_kernel_launches']}")
    else:
        print(f"kernel launches: {data['kernel_launch_note']}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
