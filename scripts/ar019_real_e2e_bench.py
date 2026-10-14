"""AR019/T003: real-checkpoint end-to-end wall-clock benchmark (P1-7 rung 5).

Runs the Fast-dLLM-v2-1.5B block-verification workload (the AR019
feasibility probe P3 design: parent = K_PROMPT real prompt tokens + one
all-mask block, children = the same prompt with the block filled by N
different model-sampled top-10 tokens) on THREE paths per config:

* ``baseline_eager`` -- the raw (unfolded) remote-code model, N full
  forwards per step (the model's own eval forward builds the identical
  block-causal mask and rotary internally, so both sides share one
  context semantics).
* ``graph_fixed`` -- the AR016 batched CUDA graph path with the fixed
  ``capacity_ratio=0.5`` capacity, now running on a REAL checkpoint via
  the AR019 ``position_embeddings`` threading (rotary is required by the
  remote Qwen layers; without it every folded forward raises).
* ``graph_adaptive`` -- the AR017 adaptive-capacity path on the same
  workload (the divergence demand is N * 32 tokens, far below the fixed
  C = 0.5 * N * T, so a right-sized capacity should cut the padded FFN
  recompute).

Configs: (T=256, N=4), (T=256, N=8), (T=512, N=4) -- the block-verification
geometry (bd_size=32, T a multiple of 32, K_PROMPT = T - 32).

Hypotheses (declared before measurement; verdicts from data only):

* **H1** -- graph_adaptive is significantly faster than baseline_eager
  (the decisive P1-7 answer on a real checkpoint: the folded framework
  must beat the raw baseline end to end, not only on synthetic stacks).
* **H2** -- all three paths agree: exact argmax agreement against the
  baseline AND ``allclose(rtol=1e-2, atol=1e-1)`` (bf16 1-ULP at the
  logits scale is ~2^-7 relative; the probe P3 measured bit-exactness,
  which this tolerance band covers).
* **H3** -- the gate splits exactly along block lines: the stable ratio
  mean is within 0.02 of K_PROMPT/T and the per-(layer, branch) minimum
  stays at or above K_PROMPT/T - 0.02 (block-causal semantics make the
  prompt blocks bit-stable and the fill block divergent).
* **H4** -- the adaptive capacity tracks the real demand: capture
  succeeds at [N, T], the trajectory converges (no new events after the
  last one), zero budget violations in the steady state, and the final
  capacity is right-sized (>= the observed max divergence, < the fixed
  capacity -- AR017 H4 wording).

Timing is ``time.perf_counter`` around ``torch.cuda.synchronize()``
boundaries with per-repetition round-robin interleaving (RERUN_CHECKLIST
SS4.2); warmup covers the capture plus the adaptation schedule. CUPTI
kernel counts are probed (#39): recorded when available, ``null`` +
note when not, never fabricated.

Run (AR019 venv, transformers==4.53.1)::

    python -m scripts.ar019_real_e2e_bench [--quick] [--reps 30]
        [--warmup 8] [--model-path PATH] [--out PATH]

``--model-path`` defaults to the ``FASTDLLM_MODEL_PATH`` environment
variable; neither present raises ``FileNotFoundError`` (AGENTS #10: no
synthetic-model fallback for a real-checkpoint benchmark).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
)

from actfold.core import ActivationCache, SimilarityGate  # noqa: E402
from actfold.core.adaptive_capacity import AdaptiveCapacityConfig  # noqa: E402
from actfold.core.fused_ops import _TRITON_FAILED_KEYS  # noqa: E402
from actfold.models.architecture_utils import ManualFoldedForward  # noqa: E402
from actfold.profiler.stability_profiler import (  # noqa: E402
    GLOBAL_STABILITY_PROFILER,
)
from scripts.algo_experiments import TimingStats, stats_from_samples  # noqa: E402

DEFAULT_OUTPUT_PATH = Path("results/optimization/ar019_real_e2e_bench.json")
DEFAULT_MODEL_ENV = "FASTDLLM_MODEL_PATH"

MASK_ID = 151665
BLOCK = 32
TAU = 0.95
EPS = 1e-8
CAPACITY_RATIO = 0.5
SEED = 2026
TOPK = 10

_CONFIGS: tuple[tuple[int, int], ...] = (
    (256, 4),
    (256, 8),
    (512, 4),
)
_PATH_ORDER = (
    "baseline_eager",
    "graph_fixed",
    "graph_adaptive",
)
_PATH_LABELS = tuple(
    (label, name)
    for label, name in zip(("baseline", "fixed", "adaptive"), _PATH_ORDER)
)

_PROMPT_TEXT = (
    "The theory of relativity describes how space and time are linked. "
    "Explain the key ideas in a few sentences. Write a short paragraph about "
    "why the ocean appears blue. List the first five prime numbers and "
    "explain what makes a number prime. Describe how photosynthesis works "
    "in plants and why chlorophyll matters. Summarize the plot of a classic "
    "novel in your own words. Explain why the sky is red at sunset."
)


# ---------------------------------------------------------------------------
# Model / context / workload construction
# ---------------------------------------------------------------------------
def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Stream-hash a (potentially large) checkpoint file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block_bytes = handle.read(chunk)
            if not block_bytes:
                break
            digest.update(block_bytes)
    return digest.hexdigest()


def _load_model(model_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Load the remote-code checkpoint in bf16 on CUDA (eval mode)."""
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True
    )
    weights = model_path / "model.safetensors"
    meta = {
        "model_path": str(model_path),
        "weights_sha256": _sha256_of(weights) if weights.exists() else None,
        "config": {
            "hidden_size": int(model.config.hidden_size),
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "vocab_size": int(model.config.vocab_size),
        },
    }
    return model, tokenizer, meta


def _build_context(model: Any, seq_len: int) -> tuple[torch.Tensor, tuple]:
    """Build the block-causal mask and rotary embeddings for one T.

    Both objects are constructed ONCE and reused for every call of every
    path (the AR019 graph identity contract pins them by reference).
    The bool ``[T, T]`` mask equals the remote ``eval_block_diff_mask``
    element-wise (probe P4: ``block_q >= block_kv`` -- bidirectional
    inside a 32-token block, causal across blocks).
    """
    arange = torch.arange(seq_len, device="cuda")
    mask = (arange[:, None] // BLOCK) >= (arange[None, :] // BLOCK)
    hidden = int(model.config.hidden_size)
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(
            torch.zeros(1, seq_len, hidden, device="cuda", dtype=torch.bfloat16),
            arange.unsqueeze(0),
        )
    return mask, (cos, sin)


def _make_workload(
    model: Any,
    tokenizer: Any,
    seq_len: int,
    num_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the parent ids and the [N, T] candidate block fillings.

    The prompt text is tokenized once, then truncated or cycle-filled to
    ``K_PROMPT = T - 32``; the parent appends one all-mask block; the N
    children fill that block with different seeded top-10 picks from the
    parent's own logits at the mask positions (probe P3 recipe).
    """
    k_prompt = seq_len - BLOCK
    ids = tokenizer(_PROMPT_TEXT, return_tensors="pt").input_ids.cuda()
    if ids.shape[1] < k_prompt:
        reps = -(-k_prompt // ids.shape[1])
        ids = ids.repeat(1, reps)
    ids = ids[:, :k_prompt]
    # The parent spans the full T: prompt part + one all-mask block.
    parent_ids = torch.cat(
        [
            ids,
            torch.full(
                (1, BLOCK), MASK_ID, device=ids.device, dtype=ids.dtype
            ),
        ],
        dim=1,
    )

    with torch.no_grad():
        base_parent = model(parent_ids)
        logits_at_mask = base_parent.logits[:, k_prompt:, :]
        topk = logits_at_mask.float().topk(TOPK, dim=-1).indices  # [1, 32, 10]
    generator = torch.Generator(device="cpu").manual_seed(SEED + num_candidates)
    children = []
    for _ in range(num_candidates):
        sel = torch.randint(0, TOPK, (BLOCK,), generator=generator)
        pick = topk[0, torch.arange(BLOCK), sel].cuda()
        child = parent_ids.clone()
        child[:, k_prompt:] = pick
        children.append(child)
    cands = torch.cat(children, dim=0)  # [N, T]
    return parent_ids, cands


def _make_mff(model: Any, adaptive: bool) -> ManualFoldedForward:
    """Build a folded stack with the graph path on (AR017 precedent)."""
    cache = ActivationCache(max_entries_per_layer=8192, device="cuda")
    gate = SimilarityGate(tau=TAU, metric="cosine", eps=EPS)
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        split_layers=True,
        split_min_tokens=1,
        use_cuda_graph=True,
        graph_capacity_ratio=CAPACITY_RATIO,
        graph_adaptive_capacity=adaptive,
        graph_adaptive_config=AdaptiveCapacityConfig() if adaptive else None,
    )


# ---------------------------------------------------------------------------
# Timing / probes / verdicts
# ---------------------------------------------------------------------------
def _timed_ms(step_fn: Callable[[], Any]) -> float:
    """Wall-clock one step in ms, synchronize-bounded (perf_counter)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _gpu_driver_version() -> str | None:
    """Query the display driver version via nvidia-smi (best effort)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        first = out.stdout.strip().splitlines()[0].strip()
        return first or None
    except Exception:  # noqa: BLE001 - metadata probe must never fail the run
        return None


def _cuda_profiler_supported() -> bool:
    """Probe whether this build records CUDA profiler events (CUPTI, #39)."""
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


def _argmax_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of positions where two [N, T, V] logits agree on argmax."""
    return float((a.argmax(-1) == b.argmax(-1)).float().mean().item())


def _verdict(faster: TimingStats, slower: TimingStats) -> str:
    """supported iff the p50 delta exceeds 2x the merged std (else decide)."""
    delta = slower.p50 - faster.p50
    merged_std = (faster.std**2 + slower.std**2) ** 0.5
    if delta > 2 * merged_std:
        return "supported"
    if delta < -2 * merged_std:
        return "refuted"
    return "inconclusive"


def _overall(verdicts: list[str]) -> str:
    """Strict aggregation: supported everywhere, refuted anywhere, else mixed."""
    if all(v == "supported" for v in verdicts):
        return "supported"
    if any(v == "refuted" for v in verdicts):
        return "refuted"
    return "inconclusive"


# ---------------------------------------------------------------------------
# Per-config benchmark
# ---------------------------------------------------------------------------
def _run_config(
    model: Any,
    tokenizer: Any,
    seq_len: int,
    num_candidates: int,
    warmup: int,
    reps: int,
) -> dict[str, Any]:
    """Benchmark one (T, N) config on the three interleaved paths."""
    torch.manual_seed(SEED + seq_len + num_candidates)
    torch.cuda.reset_peak_memory_stats()
    k_prompt = seq_len - BLOCK

    mask, pe = _build_context(model, seq_len)
    parent_ids, cands = _make_workload(
        model, tokenizer, seq_len, num_candidates
    )

    # Baseline reference logits (untimed): the raw model's own eval forward
    # builds the same block-causal mask and rotary, so both sides share one
    # context semantics (probe P3 measured bit-exact folded-vs-baseline).
    with torch.no_grad():
        base_children = torch.cat(
            [model(cands[i : i + 1]).logits for i in range(num_candidates)],
            dim=0,
        )

    mffs: dict[str, ManualFoldedForward] = {
        "graph_fixed": _make_mff(model, adaptive=False),
        "graph_adaptive": _make_mff(model, adaptive=True),
    }
    with torch.no_grad():
        for mff in mffs.values():
            mff(
                parent_ids,
                branch_id="parent",
                attention_mask=mask,
                position_embeddings=pe,
            )

    fixed_ids = [f"f{i}" for i in range(num_candidates)]
    adaptive_ids = [f"a{i}" for i in range(num_candidates)]
    step_fns: dict[str, Callable[[], Any]] = {
        "baseline_eager": lambda: [
            model(cands[i : i + 1]) for i in range(num_candidates)
        ],
        "graph_fixed": lambda: mffs["graph_fixed"](
            cands,
            branch_id=None,
            branch_ids=fixed_ids,
            parent_branch_id="parent",
            attention_mask=mask,
            position_embeddings=pe,
        ),
        "graph_adaptive": lambda: mffs["graph_adaptive"](
            cands,
            branch_id=None,
            branch_ids=adaptive_ids,
            parent_branch_id="parent",
            attention_mask=mask,
            position_embeddings=pe,
        ),
    }

    # Per-repetition round-robin interleaving (RERUN_CHECKLIST 4.2). The
    # warmup covers both captures and the adaptation schedule (shrink
    # patience 3 + throttle interval 2). Sidecar probes (budget flag, pe
    # identity canary, capacity trajectory) are read BETWEEN timed steps --
    # zero timing pollution; ``budget_exceeded`` / the pinned object are
    # host-side state already resolved by the replay's single readback.
    samples: dict[str, list[float]] = {name: [] for name in _PATH_ORDER}
    budget_violation_flags = {"graph_fixed": [], "graph_adaptive": []}
    pe_identity_mismatches = {"graph_fixed": 0, "graph_adaptive": 0}
    capacity_trajectory: list[int] = []
    divergence_samples: list[int] = []
    with torch.no_grad():
        for _ in range(warmup + reps):
            for name in _PATH_ORDER:
                elapsed = _timed_ms(step_fns[name])
                samples[name].append(elapsed)
                if name == "baseline_eager":
                    continue
                mff = mffs[name]
                runner = mff.graph_runner_batched
                exceeded = bool(runner.budget_exceeded) if runner is not None else False
                budget_violation_flags[name].append(exceeded)
                if runner is not None:
                    divergences = cast(Any, runner).last_layer_divergences
                    if divergences:
                        divergence_samples.append(max(divergences))
                # pe identity canary (script-layer private probe, AR018
                # precedent): the graph path pins the pe object at capture;
                # a mismatch would mean a silent eager fallback happened.
                if cast(Any, mff)._graph_batched_pe is not pe:
                    pe_identity_mismatches[name] += 1
                if name == "graph_adaptive":
                    if runner is not None:
                        capacity_trajectory.append(int(runner.capacity))
    timings = {
        name: stats_from_samples(samples[name][warmup:]) for name in _PATH_ORDER
    }

    # Parity sidecar (untimed): one extra call per graph path against the
    # baseline reference. The profiler is reset first so the per-(layer,
    # branch) stable ratios describe exactly these calls (H3).
    parity: dict[str, Any] = {}
    stable_ratios: dict[str, Any] = {}
    GLOBAL_STABILITY_PROFILER.reset()
    with torch.no_grad():
        for name in ("graph_fixed", "graph_adaptive"):
            out = step_fns[name]()
            # Per-row float casting caps the transient at one [1, T, V]
            # fp32 pair (the whole-tensor cast would spike ~3x [N, T, V]
            # fp32 on a 16 GB card at the (512, 4) config).
            maxdiff = 0.0
            close = True
            for row in range(num_candidates):
                o = out[row : row + 1].float()
                b = base_children[row : row + 1].float()
                maxdiff = max(maxdiff, float((o - b).abs().max().item()))
                close = close and bool(
                    torch.allclose(o, b, rtol=1e-2, atol=1e-1)
                )
                del o, b
            agreement = _argmax_agreement(out, base_children)
            parity[name] = {
                "argmax_agreement": round(agreement, 6),
                "maxdiff": round(maxdiff, 6),
                "allclose_rtol1e-2_atol1e-1": close,
            }
            del out
            ratios: list[float] = []
            branch_ids = fixed_ids if name == "graph_fixed" else adaptive_ids
            for bid in branch_ids:
                profile = GLOBAL_STABILITY_PROFILER.get_profile(bid)
                if profile:
                    ratios.extend(s.stable_ratio for s in profile.layer_stats)
            if ratios:
                stable_ratios[name] = {
                    "mean": round(sum(ratios) / len(ratios), 4),
                    "min": round(min(ratios), 4),
                    "max": round(max(ratios), 4),
                    "n": len(ratios),
                }
            GLOBAL_STABILITY_PROFILER.reset()

    # Graph state probes: capture shape, adaptive trail, H4 right-sizing.
    num_tokens = num_candidates * seq_len
    fixed_runner = mffs["graph_fixed"].graph_runner_batched
    adaptive_runner = mffs["graph_adaptive"].graph_runner_batched
    graph_captured = {
        "graph_fixed": fixed_runner is not None
        and tuple(fixed_runner.tokens_static.shape) == (num_candidates, seq_len),
        "graph_adaptive": adaptive_runner is not None
        and tuple(adaptive_runner.tokens_static.shape)
        == (num_candidates, seq_len),
    }
    policy = mffs["graph_adaptive"].graph_adaptive_policy
    recapture_events = (
        [e for e in cast(Any, policy).events if e.get("kind") == "recapture"]
        if policy is not None
        else []
    )
    final_capacity = (
        int(cast(Any, adaptive_runner).capacity)
        if adaptive_runner is not None
        else None
    )
    fixed_capacity = (
        int(cast(Any, fixed_runner).capacity)
        if fixed_runner is not None
        else num_tokens // 2  # exact for the in-scope power-of-two configs
    )
    observed_max_divergence = max(divergence_samples) if divergence_samples else None
    # Trajectory convergence: the capacity is constant over the trailing
    # post-warmup window (no new adaptation events once the steady state
    # is reached).
    steady_trajectory = capacity_trajectory[warmup:]
    trajectory_converged = bool(
        steady_trajectory and len(set(steady_trajectory)) == 1
    )
    # The warmup/timed split aligns with ``samples``: the adaptive path
    # runs once per round, so flag ``i`` belongs to round ``i``; the timed
    # steady state is rounds >= warmup.
    steady_violations = sum(budget_violation_flags["graph_adaptive"][warmup:])
    h4_supported = bool(
        graph_captured["graph_adaptive"]
        and trajectory_converged
        and final_capacity is not None
        and observed_max_divergence is not None
        and final_capacity >= observed_max_divergence
        and final_capacity < fixed_capacity
        and steady_violations == 0
        and len(recapture_events) >= 1
    )

    speedups = {
        "adaptive_vs_baseline": (
            timings["baseline_eager"].p50 / timings["graph_adaptive"].p50
            if timings["graph_adaptive"].p50 > 0
            else float("inf")
        ),
        "fixed_vs_baseline": (
            timings["baseline_eager"].p50 / timings["graph_fixed"].p50
            if timings["graph_fixed"].p50 > 0
            else float("inf")
        ),
        "adaptive_vs_fixed": (
            timings["graph_fixed"].p50 / timings["graph_adaptive"].p50
            if timings["graph_adaptive"].p50 > 0
            else float("inf")
        ),
    }

    return {
        "seq_len": seq_len,
        "num_candidates": num_candidates,
        "k_prompt": k_prompt,
        "num_tokens": num_tokens,
        "timings_ms": {name: timings[name].as_dict() for name in _PATH_ORDER},
        "speedups_p50": {k: round(v, 4) for k, v in speedups.items()},
        "graph_captured": graph_captured,
        "budget_violations": {
            name: sum(flags) for name, flags in budget_violation_flags.items()
        },
        "pe_identity_mismatches": pe_identity_mismatches,
        "parity": parity,
        "stable_ratios": stable_ratios,
        "expected_stable_ratio": round(k_prompt / seq_len, 4),
        "adaptive_state": {
            "fixed_capacity": fixed_capacity,
            "final_capacity": final_capacity,
            "observed_max_divergence": observed_max_divergence,
            "divergence_p50": (
                statistics.median(divergence_samples)
                if divergence_samples
                else None
            ),
            "capacity_trajectory": capacity_trajectory,
            "trajectory_converged": trajectory_converged,
            "recapture_count": len(recapture_events),
            "recapture_events": recapture_events,
            "steady_state_violations": steady_violations,
        },
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "h1_adaptive_vs_baseline": _verdict(
            timings["graph_adaptive"], timings["baseline_eager"]
        ),
        "h2_parity": (
            "supported"
            if all(
                p["argmax_agreement"] == 1.0
                and p["allclose_rtol1e-2_atol1e-1"]
                for p in parity.values()
            )
            else "refuted"
        ),
        "h3_block_split": (
            "supported"
            if stable_ratios
            and all(
                abs(r["mean"] - k_prompt / seq_len) <= 0.02
                and r["min"] >= k_prompt / seq_len - 0.02
                for r in stable_ratios.values()
            )
            else "refuted"
        ),
        "h4_capacity_tracking": "supported" if h4_supported else "refuted",
    }


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run_bench(
    model_path: Path,
    warmup: int = 8,
    reps: int = 30,
    quick: bool = False,
) -> dict[str, Any]:
    """Run the three-path real-checkpoint benchmark over the (T, N) scan.

    Raises:
        RuntimeError: If no CUDA device is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("run_bench requires a CUDA device")

    configs = list(_CONFIGS)
    if quick:
        configs = [(256, 4)]

    model, tokenizer, model_meta = _load_model(model_path)
    driver = _gpu_driver_version()
    counts_supported = _cuda_profiler_supported()
    import transformers

    config_records = []
    for seq_len, num_candidates in configs:
        GLOBAL_STABILITY_PROFILER.reset()
        record = _run_config(
            model, tokenizer, seq_len, num_candidates, warmup, reps
        )
        config_records.append(record)
        print(
            f"T={seq_len} N={num_candidates}: "
            + " | ".join(
                f"{label:8s} p50={record['timings_ms'][name]['p50']:.3f}ms"
                for label, name in _PATH_LABELS
            )
        )
        state = record["adaptive_state"]
        print(
            f"    capacity {state['fixed_capacity']} -> "
            f"{state['final_capacity']} (max demand {state['observed_max_divergence']}), "
            f"{state['recapture_count']} recaptures, "
            f"peak VRAM {record['peak_vram_mb']:.0f} MB, "
            f"H1={record['h1_adaptive_vs_baseline']} "
            f"H2={record['h2_parity']} "
            f"H3={record['h3_block_split']} "
            f"H4={record['h4_capacity_tracking']}"
        )

    h1 = _overall([r["h1_adaptive_vs_baseline"] for r in config_records])
    h2 = _overall([r["h2_parity"] for r in config_records])
    h3 = _overall([r["h3_block_split"] for r in config_records])
    h4 = _overall([r["h4_capacity_tracking"] for r in config_records])

    return {
        "meta": {
            "device": torch.cuda.get_device_name(0),
            "gpu_driver_version": driver,
            "gpu_driver_version_note": (
                "queried via nvidia-smi (AR016 Minor-2 closure); null means "
                "the query failed on this host and the value was omitted "
                "rather than fabricated"
                if driver is None
                else "queried via nvidia-smi (AR016 Minor-2 closure)"
            ),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": cast(Any, torch.version).cuda,
            "platform_note": (
                "Windows WDDM display-driver mode (no clock locking); noise "
                "mitigated by warmup + reps + p50 + per-rep round-robin"
            ),
            "run_date_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "model": "Fast-dLLM-v2-1.5B (Efficient-Large-Model, remote code)",
            "model_details": model_meta,
            "venv": sys.prefix != sys.base_prefix,
            "workload": {
                "mask_id": MASK_ID,
                "block": BLOCK,
                "mask_formula": "(q // 32) >= (kv // 32) [bool T x T, equals "
                "the remote eval_block_diff_mask element-wise]",
                "workload": "parent = K_PROMPT prompt tokens + one all-mask "
                "block; children = top-10 model-sampled block fillings",
                "prompt_note": "fixed text, tokenized once, truncated or "
                "cycle-filled to K_PROMPT per config",
            },
            "tau": TAU,
            "capacity_ratio": CAPACITY_RATIO,
            "adaptive_config": {
                "grow_factor": 1.25,
                "shrink_threshold": 0.5,
                "shrink_patience": 3,
                "min_recapture_interval": 2,
                "max_recaptures": 8,
            },
            "warmup": warmup,
            "reps": reps,
            "warmup_note": (
                "warmup covers both captures plus the adaptation schedule "
                "(shrink patience 3 + throttle interval 2); the timed steady "
                "state reflects the adapted capacity"
            ),
            "interleaving": "round-robin per repetition: " + " -> ".join(
                _PATH_ORDER
            ),
            "timing_method": (
                "time.perf_counter around torch.cuda.synchronize() "
                "boundaries (host-side prefill/readback/publish included; "
                "CUDA-event timing would exclude them)"
            ),
            "kernel_count": None if not counts_supported else "CUPTI available",
            "kernel_count_note": (
                "this torch build cannot record CUDA profiler events "
                "(LIBKINETO_NOCUPTI); kernel counts omitted rather than "
                "fabricated - rerun on a CUPTI-capable host to record them"
                if not counts_supported
                else "counts recordable via torch.profiler (CUPTI); "
                "wall-clock remains the primary metric"
            ),
            "triton_failed_keys": sorted(str(k) for k in _TRITON_FAILED_KEYS),
            "command": "python -m scripts.ar019_real_e2e_bench"
            + (f" --model-path {model_path}" if model_path else "")
            + (" --quick" if quick else "")
            + (f" --reps {reps}" if reps != 30 else "")
            + (f" --warmup {warmup}" if warmup != 8 else ""),
        },
        "hypotheses_declared": {
            "H1": "graph_adaptive significantly faster than baseline_eager "
            "on the real checkpoint (decisive P1-7 end-to-end answer)",
            "H2": "all paths agree with the baseline: exact argmax "
            "agreement AND allclose(rtol=1e-2, atol=1e-1)",
            "H3": "the gate splits along block lines: stable mean within "
            "0.02 of K_PROMPT/T and per-(layer,branch) min at or above "
            "K_PROMPT/T - 0.02",
            "H4": "adaptive capacity tracks the demand: capture at [N, T], "
            "trajectory converged, zero steady-state violations, final "
            "capacity >= max divergence and < fixed capacity (right-sized)",
        },
        "hypotheses_verdicts": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "p1_7_real_checkpoint_verdict": {
            "question": "is the folded framework faster than the raw "
            "baseline on a real checkpoint end to end (P1-7 rung 5)?",
            "overall": h1,
            "note": "recorded either way; per-config verdicts and speedups "
            "in configs[].h1_adaptive_vs_baseline / speedups_p50",
        },
        "configs": config_records,
    }


def write_results(data: dict[str, Any], path: str | Path) -> Path:
    """Write the benchmark record as JSON, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def _resolve_model_path(cli_value: str | None) -> Path:
    """Resolve the checkpoint path from the CLI flag or the env fallback.

    Raises:
        FileNotFoundError: If neither the flag nor ``FASTDLLM_MODEL_PATH``
            yields a path (AGENTS #10: a real-checkpoint benchmark never
            silently falls back to a synthetic model).
    """
    value = cli_value or os.environ.get(DEFAULT_MODEL_ENV)
    if not value:
        raise FileNotFoundError(
            "no model path given: pass --model-path or set "
            f"{DEFAULT_MODEL_ENV} (no synthetic fallback, AGENTS #10)"
        )
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"model directory not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Local checkpoint directory (env fallback: {DEFAULT_MODEL_ENV}).",
    )
    parser.add_argument("--reps", type=int, default=30, help="Timed reps per config")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup reps per config")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Single-config smoke subset (T=256, N=4, reps=3, warmup=2)",
    )
    add_hf_env_arguments(parser)
    args = parser.parse_args()

    reps = 3 if args.quick else args.reps
    warmup = 2 if args.quick else args.warmup
    model_path = _resolve_model_path(args.model_path)
    data = run_bench(model_path, warmup=warmup, reps=reps, quick=args.quick)
    out = write_results(data, args.out)
    print()
    for key in ("H1", "H2", "H3", "H4"):
        print(f"{key}: {data['hypotheses_verdicts'][key]}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
