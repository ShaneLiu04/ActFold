"""AR029 mechanism benchmark: bounded-memory pyramid activation cache.

This script measures real tensor computation only: a small decoder stack built
from real ``nn.Linear``/softmax attention/FFN layers, driven through
``ManualFoldedForward``.  It is a mechanism/infrastructure benchmark
(``evidence_level="mechanism"``, ``baseline_claim=false``); the real-checkpoint
A/B protocol lives in ``docs/RERUN_CHECKLIST.md`` as BLOCKED-DEFERRED.

Frozen verdicts (declared before data collection):
    E1 exactness, E2 memory bound, E3 reuse effectiveness, E4 honesty.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from actfold.core import ActivationCache
from actfold.core.pyramid_cache import PyramidActivationCache
from actfold.core.pyramid_plan import build_pyramid_plan
from actfold.core.similarity_gate import SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

SCHEMA = "ar029_pyramid_cache_bench"
SCHEMA_VERSION = 1
QUICK_PATH = Path("results/optimization/ar029_pyramid_cache_bench_quick.json")
FORMAL_PATH = Path("results/optimization/ar029_pyramid_cache_bench.json")
_REPO_ROOT = Path(__file__).resolve().parents[1]

HIDDEN = 32
VOCAB = 64
DEPTH = 2
SEQ = 32
BLOCK = 16
MIN_REUSE_RATIO = 0.01


class _Block(nn.Module):
    """A real decoder block: causal-free attention + FFN with residual norms."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(h), dim=-1)
        x = x + self.proj(attn @ v)
        return x + self.ff(self.norm2(x))  # type: ignore[no-any-return]


class TinyDecoder(nn.Module):
    """Real 2-layer decoder used by the mechanism benchmark."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type(
            "Config",
            (),
            {"model_type": "llama", "vocab_size": VOCAB, "num_hidden_layers": DEPTH},
        )()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.model.layers = nn.ModuleList([_Block(HIDDEN) for _ in range(DEPTH)])
        self.model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.model.lm_head(x)  # type: ignore[no-any-return]


def build_plan(*, pressure: bool) -> Any:
    """Full (no eviction) or genuine pyramid (deeper layers evict) plan.

    The pressure plan is deliberately **not** uniform: a linear schedule with
    a large layer-0 budget keeps the whole gate-layer parent resident while
    the narrower deepest layer loses its tail.  A uniform-budget plan cannot
    exercise the AR029 target-layer guard at all — a missing gate-layer parent
    already makes the gate unstable, so the deeper hole would be masked by an
    unrelated full recompute.  The real pyramid shape (more budget early, less
    late) is exactly the regime the guard exists for.
    """
    if not pressure:
        per_layer = SEQ * 4
        return build_pyramid_plan(
            depth=DEPTH,
            total_tokens=DEPTH * per_layer,
            block_size=BLOCK,
            max_ratio=1.0,
            min_ratio=1.0,
            recent_blocks=0,
            anchor_blocks=1,
            anchor_stride=100,
        )
    # depth=2, total=96 -> layer0=64 (4 blocks: root 2 + child 2 fit),
    # layer1=16 (1 block: the tail of the parent is evicted).
    return build_pyramid_plan(
        depth=DEPTH,
        total_tokens=6 * BLOCK,
        block_size=BLOCK,
        max_ratio=1.0,
        min_ratio=0.25,
        schedule="linear",
        recent_blocks=0,
        anchor_blocks=1,
        anchor_stride=100,
    )


def verdict_e1(full_diff: float, pressure_diff: float) -> bool:
    """E1: bit-exact against both references (CPU, non-split)."""
    return full_diff == 0.0 and pressure_diff == 0.0


def verdict_e2(
    tokens: dict[int, int], budgets: dict[int, int], cache_bytes: int, block_bytes: int
) -> bool:
    """E2: per-layer budget + independent byte ledger."""
    return all(tokens.get(layer, 0) <= budgets[layer] for layer in budgets) and (
        cache_bytes == block_bytes
    )


def verdict_e3(reuse_tokens: int, valid_tokens: int) -> bool:
    """E3: the cache is actually reused under pressure."""
    ratio = reuse_tokens / max(valid_tokens, 1)
    return reuse_tokens > 0 and ratio >= MIN_REUSE_RATIO


def verdict_e4(outputs: list[torch.Tensor], stat_values: list[float]) -> bool:
    """E4: every output finite and every stat finite/non-negative."""
    return all(torch.isfinite(x).all().item() for x in outputs) and all(
        math.isfinite(v) and v >= 0 for v in stat_values
    )


def overall_verdict(e1: bool, e2: bool, e3: bool, e4: bool) -> str:
    """Frozen precedence: blocked (honesty) > supported > refuted."""
    if not e4:
        return "blocked"
    if e1 and e2 and e3:
        return "supported"
    return "refuted"


class _Trace:
    """Capture guarded stable masks for reuse accounting."""

    def __init__(self) -> None:
        self.reuse = 0
        self.valid = 0
        self._original: Any = None

    def __enter__(self) -> "_Trace":
        self._original = GLOBAL_STABILITY_PROFILER.record

        def record(**kwargs: Any) -> Any:
            mask = kwargs["stable_mask"]
            self.reuse += int(mask.sum())
            self.valid += int(mask.numel())
            return self._original(**kwargs)

        GLOBAL_STABILITY_PROFILER.record = record  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._original is not None:
            GLOBAL_STABILITY_PROFILER.record = self._original  # type: ignore[assignment]


def _tokens(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, SEQ), generator=gen)


def _measure(out: torch.Tensor, started: float) -> dict[str, Any]:
    return {
        "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        "finite": bool(torch.isfinite(out).all()),
    }


def _run_face(
    cache: Any,
    root: torch.Tensor,
    child: torch.Tensor,
    *,
    forced: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Write ``root``, then fold the ``child`` branch against it."""
    model = _MODEL
    mff = ManualFoldedForward(model, cache=cache, gate=SimilarityGate(tau=0.9))
    with torch.no_grad():
        started = time.perf_counter()
        mff(root, branch_id="root")
        out = mff(child, branch_id="child", parent_branch_id="root", forced_divergent=forced)
    return out, started


_MODEL: TinyDecoder


class _HoleMaskedDense:
    """Dense cache oracle that injects the pyramid's factual layer holes.

    The pressure face needs a reference whose **per-layer** hole pattern is the
    same as the pyramid's.  A single ``forced_divergent`` mask cannot express
    that (it applies to every layer), and forcing the deepest hole everywhere
    would recompute the shallower layer too, which is a different computation.
    This oracle keeps real full-precision parents in an ``ActivationCache`` and
    exposes exactly the pyramid's unavailable rows as the NaN sentinel, so the
    gate and the AR029 guard see bit-identical information while the storage
    implementation is independent of the block-page cache.
    """

    def __init__(self, cache: ActivationCache, holes: dict[int, torch.Tensor]) -> None:
        self._cache = cache
        self._holes = holes

    def _masked(
        self, layer_idx: int, activations: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        hole = self._holes.get(layer_idx)
        if hole is None:
            return activations
        mask = hole.view(1, -1, 1)
        return {
            name: torch.where(mask, torch.full_like(t, float("nan")), t)
            for name, t in activations.items()
        }

    def fetch(
        self, branch_id: str, layer_idx: int, step_idx: int = 0, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        return self._masked(layer_idx, self._cache.fetch(branch_id, layer_idx, step_idx))

    def get_all(
        self, branch_id: str, layer_idx: int, step_idx: int = 0, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        return self.fetch(branch_id, layer_idx, step_idx)

    def availability(
        self, branch_id: str, layer_idx: int, seq_len: int, step_idx: int = 0
    ) -> torch.Tensor:
        hole = self._holes.get(layer_idx)
        if hole is None:
            return torch.ones(seq_len, dtype=torch.bool)
        return ~hole.view(-1)[:seq_len]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cache, name)


def _dense_reference_stats(cache: ActivationCache) -> dict[str, Any]:
    """Account the dense oracle's retained tensors (reference-only metrics)."""
    token_rows = 0
    byte_total = 0
    for layer in cache._layers.values():  # reference accounting only
        for entry in layer.values():
            if entry:
                sample = next(iter(entry.values()))
                token_rows += int(sample.shape[0] * sample.shape[1])
            byte_total += sum(t.numel() * t.element_size() for t in entry.values())
    return {"cache_bytes": int(byte_total), "tokens_stored": int(token_rows)}


def run(quick: bool) -> dict[str, Any]:
    """Run the frozen mechanism benchmark and return the artifact payload."""
    global _MODEL
    torch.manual_seed(1234)
    _MODEL = TinyDecoder().eval()
    root = _tokens(0)
    child = root.clone()
    child[:, -8:] = (child[:, -8:] + 7) % VOCAB

    # Full face: no eviction; pyramid must equal the dense cache bit-for-bit.
    full_plan = build_plan(pressure=False)
    dense_cache = ActivationCache(max_entries_per_layer=SEQ * 4)
    dense_out, dense_start = _run_face(dense_cache, root, child)
    dense_ms = (time.perf_counter() - dense_start) * 1000

    full_cache = PyramidActivationCache(full_plan)
    with _Trace() as full_trace:
        full_out, full_start = _run_face(full_cache, root, child)
    full_ms = (time.perf_counter() - full_start) * 1000
    full_diff = float(
        (full_out - dense_out).abs().max().item()
        if full_out.shape == dense_out.shape
        else float("inf")
    )

    # Pressure face: genuine pyramid (wide early layers, narrow deep layer).
    # Snapshot the per-layer hole pattern BEFORE the measured forward; the
    # oracle replays the same branch/tokens with those exact holes.
    pressure_plan = build_plan(pressure=True)
    pressure_cache = PyramidActivationCache(pressure_plan)
    with torch.no_grad():
        ManualFoldedForward(_MODEL, cache=pressure_cache, gate=SimilarityGate(tau=0.9))(
            root, branch_id="root"
        )
    holes = {layer: ~pressure_cache.availability("root", layer, SEQ, 0) for layer in range(DEPTH)}
    with _Trace() as trace:
        pressure_out, pressure_start = _run_face(pressure_cache, root, child)
    pressure_ms = (time.perf_counter() - pressure_start) * 1000

    dense_ref_cache = ActivationCache(max_entries_per_layer=SEQ * 4)
    with torch.no_grad():
        ManualFoldedForward(_MODEL, cache=dense_ref_cache, gate=SimilarityGate(tau=0.9))(
            root, branch_id="root"
        )
    oracle = _HoleMaskedDense(dense_ref_cache, holes)
    with _Trace() as ref_trace:
        with torch.no_grad():
            started = time.perf_counter()
            ref_out = ManualFoldedForward(_MODEL, cache=oracle, gate=SimilarityGate(tau=0.9))(
                child,
                branch_id="ref",
                parent_branch_id="root",
            )
        ref_ms = (time.perf_counter() - started) * 1000
    pressure_diff = float((pressure_out - ref_out).abs().max().item())

    stats = pressure_cache.stats()
    budget = {layer: pressure_plan.budget_tokens(layer) for layer in range(DEPTH)}
    block_bytes = sum(t.numel() * t.element_size() for t in pressure_cache.iter_block_tensors())
    reference = _dense_reference_stats(dense_ref_cache)

    e1 = verdict_e1(full_diff, pressure_diff)
    full_stats = full_cache.stats()
    full_budget = {layer: full_plan.budget_tokens(layer) for layer in range(DEPTH)}
    full_block_bytes = sum(t.numel() * t.element_size() for t in full_cache.iter_block_tensors())
    e2 = verdict_e2(
        {k: int(v) for k, v in full_stats["tokens"].items()},
        full_budget,
        int(full_stats["bytes"]),
        int(full_block_bytes),
    ) and verdict_e2(
        {k: int(v) for k, v in stats["tokens"].items()},
        budget,
        int(stats["bytes"]),
        int(block_bytes),
    )
    e3 = verdict_e3(trace.reuse, trace.valid)
    e4 = verdict_e4(
        [full_out, pressure_out, ref_out],
        [float(stats[k]) for k in ("bytes", "evictions", "requests", "missing_tokens")]
        + [float(v) for v in stats["tokens"].values()]
        + [float(v) for v in stats["blocks"].values()]
        + [float(full_stats[k]) for k in ("bytes", "evictions", "requests", "missing_tokens")]
        + [float(v) for v in full_stats["tokens"].values()]
        + [float(v) for v in full_stats["blocks"].values()],
    )
    overall = overall_verdict(e1, e2, e3, e4)

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evidence_level": "mechanism",
        "baseline_claim": False,
        "status": "ok" if overall != "blocked" else "blocked",
        "config": {
            "hidden": HIDDEN,
            "vocab": VOCAB,
            "depth": DEPTH,
            "seq": SEQ,
            "block_size": BLOCK,
            "quick": bool(quick),
            "full_budgets": list(full_plan.layer_budgets),
            "pressure_budgets": list(pressure_plan.layer_budgets),
            "tau": 0.9,
        },
        "measurements": {
            "full": {
                "cache_bytes": int(full_cache.bytes_used),
                "tokens_stored": int(sum(full_cache.stats()["tokens"].values())),
                "reuse_ratio": round(full_trace.reuse / max(full_trace.valid, 1), 6),
                "missing_ratio": round(
                    full_stats["missing_tokens"]
                    / max(full_trace.valid + full_stats["missing_tokens"], 1),
                    6,
                ),
                "stable_ratio": round(full_trace.reuse / max(full_trace.valid, 1), 6),
                "wall_ms": round(full_ms, 3),
                "dense_wall_ms": round(dense_ms, 3),
                "max_abs_diff_vs_dense": full_diff,
                "child_max_abs_diff_vs_dense": full_diff,
            },
            "pressure": {
                "cache_bytes": int(stats["bytes"]),
                "tokens_stored": int(sum(stats["tokens"].values())),
                "reuse_ratio": round(trace.reuse / max(trace.valid, 1), 6),
                "missing_ratio": round(
                    stats["missing_tokens"] / max(trace.valid + stats["missing_tokens"], 1), 6
                ),
                "stable_ratio": round(trace.reuse / max(trace.valid, 1), 6),
                "wall_ms": round(pressure_ms, 3),
                "reuse_tokens": int(trace.reuse),
                "valid_tokens": int(trace.valid),
                "max_abs_diff_vs_hole_oracle": pressure_diff,
                "evictions": int(stats["evictions"]),
            },
            "reference": {
                "cache_bytes": reference["cache_bytes"],
                "tokens_stored": reference["tokens_stored"],
                "reuse_ratio": round(ref_trace.reuse / max(ref_trace.valid, 1), 6),
                "missing_ratio": 0.0,
                "stable_ratio": round(ref_trace.reuse / max(ref_trace.valid, 1), 6),
                "wall_ms": round(ref_ms, 3),
                "reuse_tokens": int(ref_trace.reuse),
                "valid_tokens": int(ref_trace.valid),
            },
        },
        "verdicts": {"E1": e1, "E2": e2, "E3": e3, "E4": e4},
        "overall": overall,
        "meta": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "note": (
                "mechanism-level evidence; real-checkpoint A/B is BLOCKED-DEFERRED "
                "and must not be inferred from this artifact"
            ),
        },
    }
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AR029 mechanism benchmark")
    parser.add_argument("--quick", action="store_true", help="write the quick artifact")
    args = parser.parse_args(argv)
    if not args.quick:
        print(
            "formal face is BLOCKED-DEFERRED (user-authorized no-large-experiments "
            "scope); run with --quick for the mechanism artifact"
        )
        return 3
    artifact = run(quick=True)
    out = _REPO_ROOT / QUICK_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {out} overall={artifact['overall']}")
    return 0 if artifact["overall"] in ("supported", "blocked") else 1


if __name__ == "__main__":
    sys.exit(main())
