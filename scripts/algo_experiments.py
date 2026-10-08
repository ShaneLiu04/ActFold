#!/usr/bin/env python3
"""Deep algorithm experiments for ActFold on real diffusion LLMs.

This script runs a battery of interpretability and ablation experiments on a
real diffusion language model (Fast-dLLM v2, LLaDA, or Dream) and saves
machine-readable artifacts for figure generation and reporting.

Experiments
-----------
1. ``similarity``      per-layer per-token parent/child cosine similarity maps.
2. ``tau_sweep``       stable ratio / output fidelity / latency vs threshold.
3. ``invariants``      self-fold fast path and all-divergent exactness checks.
4. ``layer_ablation``  folding restricted to early / late / all layers.
5. ``cache_budget``    partial-cache reuse under tight per-layer budgets.
6. ``dynamic_tau``     FoldingScheduler per-layer thresholds vs fixed tau.
7. ``merge_bench``     Triton vs PyTorch stable/divergent merge latency.
8. ``sampling``        diffusion-native generation with cross-step folding.
9. ``cost_model``      measured hardware calibration and latency prediction.

Usage::

    python -m scripts.algo_experiments --model fastdllm --out results/experiments/fastdllm
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from actfold.core import ActivationCache, SimilarityGate  # noqa: E402
from actfold.core.folding_scheduler import FoldingScheduler  # noqa: E402
from actfold.core.fused_ops import (  # noqa: E402
    _merge_stable_divergent_torch,
    merge_stable_divergent,
)
from actfold.core.model_wrapper import FoldedModel  # noqa: E402
from actfold.models import load_model  # noqa: E402
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER  # noqa: E402
from actfold.utils.cost_model import ComputeBandwidthCostModel, HardwareProfile  # noqa: E402

MODEL_SPECS: dict[str, dict[str, str]] = {
    "fastdllm": {
        "repo": "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
        "family": "fast_dllm",
        "label": "Fast-dLLM-v2-1.5B",
    },
    "llada": {
        "repo": "GSAI-ML/LLaDA-8B-Instruct",
        "family": "llada",
        "label": "LLaDA-8B-Instruct",
    },
    "dream": {
        # Allow a local checkpoint (e.g. a ModelScope download) to override the
        # Hugging Face hub identifier without editing the script.
        "repo": os.environ.get("DREAM_MODEL_PATH", "Dream-org/Dream-v0-Instruct-7B"),
        "family": "dream",
        "label": "Dream-7B-Instruct",
    },
}

PROMPTS = [
    "The theory of relativity describes how space and time are linked. Explain the key ideas in a few sentences:",
    "Write a short paragraph about why the ocean appears blue:",
    "List the first five prime numbers and explain what makes a number prime:",
]

TAUS = [0.50, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class RecordingGate(SimilarityGate):
    """Similarity gate that records the raw cosine matrix of every call."""

    def __init__(self, tau: float = 0.95) -> None:
        super().__init__(tau=tau, metric="cosine")
        self.records: list[torch.Tensor] = []

    def forward(self, h_child: torch.Tensor, h_parent: torch.Tensor) -> torch.Tensor:
        sim = self._compute_similarity(h_child, h_parent)
        self.records.append(sim.detach().float().cpu())
        return sim > self.tau


def encode_prompt(tokenizer: Any, text: str, device: str, max_len: int = 96) -> torch.Tensor:
    """Tokenize ``text``, preferring the model chat template when available."""
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        ids = tokenizer(text, return_tensors="pt").input_ids
    if ids.shape[1] > max_len:
        ids = ids[:, :max_len]
    return ids.to(device)


def make_child(
    tokens: torch.Tensor, vocab_size: int, num_flips: int, seed: int = 42
) -> tuple[torch.Tensor, list[int]]:
    """Return a copy of ``tokens`` with ``num_flips`` random positions changed."""
    if num_flips <= 0:
        return tokens.clone(), []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    seq_len = tokens.shape[1]
    num_flips = min(num_flips, seq_len)
    positions = torch.randperm(seq_len, generator=generator)[:num_flips].tolist()
    rand = torch.randint(0, vocab_size, (num_flips,), generator=generator, dtype=tokens.dtype).to(
        tokens.device
    )
    child = tokens.clone()
    child[0, torch.tensor(positions, device=tokens.device)] = rand
    return child, positions


def fidelity(out: torch.Tensor, base: torch.Tensor) -> dict[str, float]:
    """Output-fidelity metrics between folded and baseline logits."""
    out_f = out.float()
    base_f = base.float()
    mse = F.mse_loss(out_f, base_f).item()
    var = base_f.var().item()
    cosine = F.cosine_similarity(out_f.flatten(), base_f.flatten(), dim=0).item()
    top1 = (out_f.argmax(dim=-1) == base_f.argmax(dim=-1)).float().mean().item()
    return {
        "mse": mse,
        "rel_mse": mse / max(var, 1e-12),
        "cosine": cosine,
        "top1_agreement": top1,
    }


def build_folded(
    model: Any,
    tau: float = 0.95,
    max_entries: int = 65536,
    scheduler: FoldingScheduler | None = None,
    gate: SimilarityGate | None = None,
) -> FoldedModel:
    cache = ActivationCache(max_entries_per_layer=max_entries, device="cuda")
    gate = gate if gate is not None else SimilarityGate(tau=tau)
    return FoldedModel(model.model, cache=cache, gate=gate, scheduler=scheduler)


def run_folded_child(
    folded: FoldedModel,
    parent_tokens: torch.Tensor,
    child_tokens: torch.Tensor,
    tag: str,
    step_idx: int = 0,
) -> tuple[torch.Tensor, Any]:
    """Populate the parent cache then run one folded child forward."""
    with torch.no_grad():
        folded(parent_tokens, branch_id=f"{tag}_p", step_idx=step_idx)
    GLOBAL_STABILITY_PROFILER.reset_branch(f"{tag}_c")
    with torch.no_grad():
        out = folded(
            child_tokens,
            branch_id=f"{tag}_c",
            parent_branch_id=f"{tag}_p",
            step_idx=step_idx,
        )
    profile = GLOBAL_STABILITY_PROFILER.get_profile(f"{tag}_c")
    return out, profile


def profile_summary(profile: Any, num_layers: int) -> dict[str, float]:
    if profile is None or not profile.layer_stats:
        return {"mean_stable": 0.0, "min_stable": 0.0, "all_stable_fraction": 0.0}
    ratios = [s.stable_ratio for s in profile.layer_stats]
    all_stable = sum(1 for r in ratios if r >= 1.0)
    return {
        "mean_stable": float(np.mean(ratios)),
        "min_stable": float(np.min(ratios)),
        "all_stable_fraction": all_stable / max(1, num_layers),
    }


@dataclass(frozen=True)
class TimingStats:
    """Summary statistics for repeated wall-clock measurements (ms).

    Fields:
        mean: Arithmetic mean over ``n`` samples.
        std: Population standard deviation (ddof=0).
        p50: Median (50th percentile).
        n: Number of timed repetitions (>= 3 for statistical validity).
    """

    mean: float
    std: float
    p50: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable dict view of the statistics."""
        return {"mean": self.mean, "std": self.std, "p50": self.p50, "n": self.n}


def stats_from_samples(samples: Sequence[float]) -> TimingStats:
    """Summarize a non-empty sequence of timing samples (ms).

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if len(samples) == 0:
        raise ValueError("stats_from_samples requires at least one sample")
    arr = np.asarray(samples, dtype=np.float64)
    return TimingStats(
        mean=float(arr.mean()),
        std=float(arr.std()),
        p50=float(np.median(arr)),
        n=int(arr.size),
    )


def time_forward(fn: Any, warmup: int = 3, reps: int = 10) -> TimingStats:
    """Time ``fn()`` with per-repeat CUDA events and return summary statistics.

    Runs under ``torch.no_grad()`` so that timed forwards do not retain
    autograd graphs; without this, long-sequence forwards on 7B/8B models
    exhaust GPU memory during repeated timing runs.

    Args:
        fn: Zero-argument callable to time.
        warmup: Untimed warm-up calls before measurement.
        reps: Number of timed repetitions. Must be >= 3.

    Returns:
        :class:`TimingStats` with mean/std/p50 over the per-repeat times.

    Raises:
        ValueError: If ``reps < 3`` (statistical validity floor).
    """
    if reps < 3:
        raise ValueError(f"time_forward requires reps >= 3, got {reps}")
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
    samples = [starts[i].elapsed_time(ends[i]) for i in range(reps)]
    return stats_from_samples(samples)


def repeat_with_seed(
    fn: Callable[[], tuple[torch.Tensor, float]],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Run ``fn()`` repeatedly with an explicit per-run seed.

    Before every call the global CPU RNG is reseeded with ``torch.manual_seed(seed)``,
    so a deterministic ``fn`` produces identical outputs on every repeat. The
    caller-visible global RNG state is snapshotted at entry and restored before
    returning, so the experiment never perturbs downstream randomness.

    Args:
        fn: Zero-argument callable returning ``(tokens, elapsed_ms)``.
        repeats: Number of repetitions. Must be >= 1.
        seed: Seed applied via ``torch.manual_seed`` before each call.

    Returns:
        Dict with keys ``"ms"`` (list of per-repeat ms), ``"tokens"`` (list of
        token tensors) and ``"identical"`` (True iff all token tensors are
        ``torch.equal`` to the first).

    Raises:
        ValueError: If ``repeats < 1``.
    """
    if repeats < 1:
        raise ValueError(f"repeat_with_seed requires repeats >= 1, got {repeats}")
    rng_state = torch.get_rng_state()
    try:
        ms: list[float] = []
        tokens: list[torch.Tensor] = []
        for _ in range(repeats):
            torch.manual_seed(seed)
            out, elapsed_ms = fn()
            tokens.append(out)
            ms.append(float(elapsed_ms))
    finally:
        torch.set_rng_state(rng_state)
    identical = all(torch.equal(t, tokens[0]) for t in tokens[1:])
    return {"ms": ms, "tokens": tokens, "identical": identical}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def exp_similarity(
    model: Any,
    prompt_tokens: torch.Tensor,
    children: dict[int, tuple[torch.Tensor, list[int]]],
    outdir: Path,
) -> dict[str, Any]:
    """Per-layer, per-token parent/child similarity heatmaps (tau=1 -> no folding)."""
    gate = RecordingGate(tau=1.0)
    folded = build_folded(model, gate=gate)
    if not folded.folding_applied:
        raise RuntimeError("FoldedModel could not discover the layer stack")
    sims: dict[int, np.ndarray] = {}
    try:
        with torch.no_grad():
            folded(prompt_tokens, branch_id="sim_parent", step_idx=0)
        for num_flips, (child, _) in children.items():
            gate.records.clear()
            GLOBAL_STABILITY_PROFILER.reset_branch("sim_child")
            with torch.no_grad():
                folded(child, branch_id="sim_child", parent_branch_id="sim_parent", step_idx=0)
            matrix = torch.cat([r.reshape(1, -1) for r in gate.records], dim=0).numpy()
            sims[num_flips] = matrix
    finally:
        folded.restore()
    np.savez(
        outdir / "similarity.npz",
        tokens=prompt_tokens.cpu().numpy(),
        **{f"flip{nf}": sims[nf] for nf in sims},
    )
    summary = {}
    for num_flips, matrix in sims.items():
        summary[f"flip{num_flips}"] = {
            "mean_sim": float(matrix.mean()),
            "median_sim": float(np.median(matrix)),
            "p05_sim": float(np.percentile(matrix, 5)),
            "shape": list(matrix.shape),
        }
    return summary


def exp_tau_sweep(
    model: Any,
    prompt_tokens: torch.Tensor,
    child: torch.Tensor,
    baseline_child: torch.Tensor,
    outdir: Path,
) -> list[dict[str, Any]]:
    """Fold the same child at different tau thresholds and measure quality/latency."""
    results = []
    folded = build_folded(model, tau=0.95)
    try:
        with torch.no_grad():
            folded(prompt_tokens, branch_id="sweep_parent", step_idx=0)
        gate = folded.gate
        for tau in TAUS:
            gate.set_tau(tau)
            GLOBAL_STABILITY_PROFILER.reset_branch("sweep_child")
            with torch.no_grad():
                out = folded(
                    child, branch_id="sweep_child", parent_branch_id="sweep_parent", step_idx=0
                )
            profile = GLOBAL_STABILITY_PROFILER.get_profile("sweep_child")
            metrics = fidelity(out, baseline_child)
            entry: dict[str, Any] = {"tau": tau}
            entry.update(profile_summary(profile, model.num_layers))
            entry.update(metrics)
            entry["latency_ms"] = time_forward(
                lambda: folded(
                    child, branch_id="sweep_child", parent_branch_id="sweep_parent", step_idx=0
                ),
                warmup=2,
                reps=8,
            ).mean
            results.append(entry)
            print(
                f"  tau={tau:<6} stable={entry['mean_stable']:.3f} "
                f"top1={entry['top1_agreement']:.4f} rel_mse={entry['rel_mse']:.3e} "
                f"latency={entry['latency_ms']:.1f}ms",
                flush=True,
            )
    finally:
        folded.restore()
    return results


def exp_invariants(
    model: Any,
    prompt_tokens: torch.Tensor,
    child: torch.Tensor,
    baseline_child: torch.Tensor,
    baseline_parent: torch.Tensor,
) -> dict[str, Any]:
    """Self-fold (all stable) and all-divergent exactness invariants."""
    out: dict[str, Any] = {}

    # Baseline latency must be measured before any layer wrapping: once a raw
    # module is wrapped by FoldedModel it cannot be called without a branch id.
    baseline_ms = time_forward(lambda: model.forward(prompt_tokens), warmup=3, reps=10).mean

    # 1) self-fold: child == parent -> every token identical -> all layers stable.
    folded = build_folded(model, tau=0.95)
    try:
        with torch.no_grad():
            folded(prompt_tokens, branch_id="inv_self_p", step_idx=0)
        GLOBAL_STABILITY_PROFILER.reset_branch("inv_self_c")
        with torch.no_grad():
            self_out = folded(
                prompt_tokens, branch_id="inv_self_c", parent_branch_id="inv_self_p", step_idx=0
            )
        profile = GLOBAL_STABILITY_PROFILER.get_profile("inv_self_c")
        folded_ms = time_forward(
            lambda: folded(
                prompt_tokens, branch_id="inv_self_c", parent_branch_id="inv_self_p", step_idx=0
            ),
            warmup=2,
            reps=10,
        ).mean
    finally:
        folded.restore()
    out["self_fold"] = {
        **fidelity(self_out, baseline_parent),
        **profile_summary(profile, model.num_layers),
        "baseline_ms": baseline_ms,
        "folded_fast_path_ms": folded_ms,
        "measured_speedup": baseline_ms / max(folded_ms, 1e-9),
    }

    # 2) all-divergent: tau=1.0 forces full recomputation -> bit-close to baseline.
    folded = build_folded(model, tau=1.0)
    try:
        with torch.no_grad():
            folded(prompt_tokens, branch_id="inv_div_p", step_idx=0)
        GLOBAL_STABILITY_PROFILER.reset_branch("inv_div_c")
        with torch.no_grad():
            div_out = folded(child, branch_id="inv_div_c", parent_branch_id="inv_div_p", step_idx=0)
        profile = GLOBAL_STABILITY_PROFILER.get_profile("inv_div_c")
    finally:
        folded.restore()
    out["all_divergent"] = {
        **fidelity(div_out, baseline_child),
        **profile_summary(profile, model.num_layers),
    }
    return out


def exp_layer_ablation(
    model: Any,
    prompt_tokens: torch.Tensor,
    child: torch.Tensor,
    baseline_child: torch.Tensor,
    tau: float = 0.99,
) -> dict[str, Any]:
    """Restrict folding to early / late / all layers via disabled_layers."""
    num_layers = model.num_layers
    early_cut = num_layers // 3
    late_cut = 2 * num_layers // 3
    variants = {
        "all": set(),
        "early_only": set(range(early_cut, num_layers)),
        "late_only": set(range(0, late_cut)),
        "none": set(range(num_layers)),
    }
    results: dict[str, Any] = {}
    for name, disabled in variants.items():
        scheduler = FoldingScheduler(
            base_tau=0.99, num_layers=num_layers, num_steps=10, disabled_layers=disabled
        )
        folded = build_folded(model, scheduler=scheduler)
        try:
            out_t, profile = run_folded_child(folded, prompt_tokens, child, f"ab_{name}")
            entry = {**profile_summary(profile, num_layers), **fidelity(out_t, baseline_child)}
        finally:
            folded.restore()
        results[name] = entry
        print(
            f"  layer_ablation[{name}] stable={entry['mean_stable']:.3f} "
            f"top1={entry['top1_agreement']:.4f} rel_mse={entry['rel_mse']:.3e}",
            flush=True,
        )
    return results


def exp_cache_budget(
    model: Any,
    prompt_tokens: torch.Tensor,
    child: torch.Tensor,
    baseline_child: torch.Tensor,
) -> dict[str, Any]:
    """Measure reuse under tight per-layer cache budgets (LRU eviction)."""
    results: dict[str, Any] = {}
    budgets = [1, 4, 16, 64, 256, 65536]
    for budget in budgets:
        folded = build_folded(model, tau=0.99, max_entries=budget)
        try:
            out_t, profile = run_folded_child(folded, prompt_tokens, child, f"cb_{budget}")
            entry = {
                **profile_summary(profile, model.num_layers),
                **fidelity(out_t, baseline_child),
            }
        finally:
            folded.restore()
        results[str(budget)] = entry
        print(
            f"  cache_budget[{budget}] stable={entry['mean_stable']:.3f} "
            f"top1={entry['top1_agreement']:.4f}",
            flush=True,
        )
    return results


def exp_dynamic_tau(
    model: Any,
    prompt_tokens: torch.Tensor,
    child: torch.Tensor,
    baseline_child: torch.Tensor,
) -> dict[str, Any]:
    """Compare a fixed gate threshold with the FoldingScheduler."""
    results: dict[str, Any] = {}
    folded = build_folded(model, tau=0.95)
    try:
        out_t, profile = run_folded_child(folded, prompt_tokens, child, "dyn_fixed")
        results["fixed_0.95"] = {
            **profile_summary(profile, model.num_layers),
            **fidelity(out_t, baseline_child),
        }
    finally:
        folded.restore()

    scheduler = FoldingScheduler(base_tau=0.95, num_layers=model.num_layers, num_steps=10)
    folded = build_folded(model, scheduler=scheduler)
    try:
        out_t, profile = run_folded_child(folded, prompt_tokens, child, "dyn_sched")
        results["scheduler"] = {
            **profile_summary(profile, model.num_layers),
            **fidelity(out_t, baseline_child),
        }
        if profile is not None:
            results["scheduler"]["tau_per_layer"] = [
                round(s.tau_used, 4) for s in profile.layer_stats
            ]
    finally:
        folded.restore()
    return results


def exp_merge_bench(model: Any, prompt_tokens: torch.Tensor) -> dict[str, Any]:
    """Triton vs PyTorch stable/divergent merge latency at model hidden size."""
    seq_len = int(prompt_tokens.shape[1])
    hidden = model.hidden_dim
    dtype = torch.bfloat16
    parent = torch.randn(1, seq_len, hidden, dtype=dtype, device="cuda")
    child = torch.randn(1, seq_len, hidden, dtype=dtype, device="cuda")
    mask = torch.rand(1, seq_len, device="cuda") > 0.3
    ref = _merge_stable_divergent_torch(parent, child, mask)
    triton_out = merge_stable_divergent(parent, child, mask)
    max_err = (triton_out.float() - ref.float()).abs().max().item()
    triton_ms = time_forward(
        lambda: merge_stable_divergent(parent, child, mask), warmup=5, reps=50
    ).mean
    torch_ms = time_forward(
        lambda: _merge_stable_divergent_torch(parent, child, mask), warmup=5, reps=50
    ).mean
    return {
        "seq_len": seq_len,
        "hidden_dim": hidden,
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "speedup": torch_ms / max(triton_ms, 1e-9),
        "max_abs_err": max_err,
    }


def exp_sampling(
    model: Any,
    prompt_tokens: torch.Tensor,
    spec: dict[str, str],
    outdir: Path,
    num_tokens: int = 32,
    num_steps: int = 32,
    repeats: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Diffusion-native generation with cross-step folded activations.

    Each variant (baseline and folded) is generated ``repeats`` times with the
    global RNG reseeded to ``seed`` before every run, so tokens are
    reproducible across repeats and the reported latency carries mean/std
    statistics. The caller-visible global RNG state is snapshotted and
    restored, so this experiment never perturbs downstream randomness.

    Raises:
        ValueError: If ``repeats < 3`` (statistical validity floor).
    """
    if repeats < 3:
        raise ValueError(f"exp_sampling requires repeats >= 3, got {repeats}")
    family = spec["family"]
    if family == "fast_dllm":
        from actfold.models.fast_dllm_sampler import FastDLLMSamplerConfig

        cfg: Any = FastDLLMSamplerConfig(
            num_steps=num_steps,
            num_tokens=num_tokens,
            block_size=16,
            small_block_size=16,
            threshold=0.9,
            temperature=0.0,
            seed=seed,
        )
    elif family == "llada":
        from actfold.models.llada_sampler import LLaDASamplerConfig

        cfg = LLaDASamplerConfig(
            num_steps=num_steps,
            num_tokens=num_tokens,
            block_size=16,
            remasking="low_confidence",
            temperature=0.0,
            seed=seed,
        )
    else:
        from actfold.models.dream_sampler import DreamSamplerConfig

        cfg = DreamSamplerConfig(
            num_steps=num_steps,
            num_tokens=num_tokens,
            alg="maskgit_plus",
            temperature=0.0,
            seed=seed,
        )

    def _generate(folded: FoldedModel | None) -> tuple[torch.Tensor, float]:
        # Global RNG seeding is handled by repeat_with_seed (explicit seed).
        start = time.perf_counter()
        with torch.no_grad():
            tokens = model.generate(
                prompt_tokens,
                max_new_tokens=num_tokens,
                num_steps=num_steps,
                sampler_config=cfg,
                folded_model=folded,
            )
        torch.cuda.synchronize()
        return tokens, (time.perf_counter() - start) * 1000.0

    baseline_runs = repeat_with_seed(lambda: _generate(None), repeats=repeats, seed=seed)
    baseline_tokens = baseline_runs["tokens"][0]
    baseline_stats = stats_from_samples(baseline_runs["ms"])
    tokens_reproducible = baseline_runs["identical"]

    GLOBAL_STABILITY_PROFILER.reset()
    folded = build_folded(model, tau=0.95, max_entries=65536)
    try:
        folded_runs = repeat_with_seed(lambda: _generate(folded), repeats=repeats, seed=seed)
        profiles = GLOBAL_STABILITY_PROFILER.all_profiles()
    finally:
        folded.restore()
    folded_tokens = folded_runs["tokens"][0]
    folded_stats = stats_from_samples(folded_runs["ms"])
    tokens_reproducible = tokens_reproducible and folded_runs["identical"]

    step_stats = []
    for branch_id, profile in profiles.items():
        if profile is None or not profile.layer_stats:
            continue
        step_stats.append(
            {
                "branch": str(branch_id),
                "mean_stable": profile.mean_stable_ratio,
                "min_stable": profile.min_stable_ratio,
                "final_stable": profile.final_stable_ratio,
            }
        )

    min_len = min(baseline_tokens.shape[1], folded_tokens.shape[1])
    match = (baseline_tokens[:, :min_len] == folded_tokens[:, :min_len]).float().mean().item()
    tokenizer = model.tokenizer
    return {
        "num_tokens": num_tokens,
        "num_steps": num_steps,
        "repeats": repeats,
        "seed": seed,
        "baseline_ms_mean": baseline_stats.mean,
        "baseline_ms_std": baseline_stats.std,
        "folded_ms_mean": folded_stats.mean,
        "folded_ms_std": folded_stats.std,
        "tokens_reproducible": tokens_reproducible,
        "token_match_rate": match,
        "num_folded_steps": len(step_stats),
        "mean_step_stable": (
            float(np.mean([s["mean_stable"] for s in step_stats])) if step_stats else 0.0
        ),
        "step_stats": step_stats,
        "baseline_text": tokenizer.decode(baseline_tokens[0], skip_special_tokens=True)[:400],
        "folded_text": tokenizer.decode(folded_tokens[0], skip_special_tokens=True)[:400],
    }


def exp_cost_model(
    model: Any,
    prompt_tokens: torch.Tensor,
    tau_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calibrate hardware with microbenchmarks and compare predicted vs measured."""
    # Memory bandwidth: read+write a 512 MB fp16 tensor.
    numel = 256 * 1024 * 1024
    a = torch.empty(numel, dtype=torch.float16, device="cuda")
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()
    copy_s = (time.perf_counter() - start) / 10
    bandwidth_gb_s = 2 * numel * 2 / copy_s / 1e9
    del a, b

    # Compute: 4096^3 matmul in fp16.
    m = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    for _ in range(3):
        m = m @ m
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        m = m @ m
    torch.cuda.synchronize()
    matmul_s = (time.perf_counter() - start) / 20
    tflops = 2 * 4096**3 / matmul_s / 1e12
    del m

    profile = HardwareProfile(
        compute_tflops=tflops, memory_bw_gb_s=bandwidth_gb_s, bytes_per_element=2
    )
    cost = ComputeBandwidthCostModel(profile)
    seq_len = int(prompt_tokens.shape[1])
    predictions = []
    for entry in tau_results:
        predicted_ms = (
            cost.estimate_total_time(
                num_layers=model.num_layers,
                seq_len=seq_len,
                hidden_dim=model.hidden_dim,
                stable_ratio=entry["mean_stable"],
                num_steps=1,
                num_heads=max(1, model.num_heads),
            )
            * 1000.0
        )
        predictions.append(
            {
                "tau": entry["tau"],
                "stable_ratio": entry["mean_stable"],
                "predicted_ms": predicted_ms,
                "measured_ms": entry["latency_ms"],
            }
        )
    return {
        "measured_compute_tflops": tflops,
        "measured_bandwidth_gb_s": bandwidth_gb_s,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--max-prompt-len", type=int, default=96)
    parser.add_argument("--skip-sampling", action="store_true")
    add_hf_env_arguments(parser)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print(f"=== loading {spec['label']} ({spec['repo']}) ===", flush=True)
    t0 = time.time()
    model = load_model(spec["repo"], model_family=spec["family"], torch_dtype=dtype)
    model.to("cuda")
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    results: dict[str, Any] = {
        "meta": {
            "model_key": args.model,
            "label": spec["label"],
            "repo": spec["repo"],
            "family": spec["family"],
            "num_layers": model.num_layers,
            "hidden_dim": model.hidden_dim,
            "num_heads": model.num_heads,
            "vocab_size": model.vocab_size,
            "dtype": args.dtype,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
        }
    }

    prompt_tokens = encode_prompt(model.tokenizer, PROMPTS[0], "cuda", args.max_prompt_len)
    results["meta"]["prompt_len"] = int(prompt_tokens.shape[1])
    print(f"prompt tokens: {prompt_tokens.shape[1]}", flush=True)

    children: dict[int, tuple[torch.Tensor, list[int]]] = {}
    baselines: dict[int, torch.Tensor] = {}
    for num_flips in (0, 1, 8):
        child, positions = make_child(prompt_tokens, model.vocab_size, num_flips, seed=42)
        children[num_flips] = (child, positions)
        with torch.no_grad():
            baselines[num_flips] = model.forward(child).float().cpu()
    parent_baseline = baselines[0]
    child = children[1][0]
    baseline_child = baselines[1]

    experiments: list[tuple[str, Any]] = [
        ("similarity", lambda: exp_similarity(model, prompt_tokens, children, outdir)),
        (
            "tau_sweep",
            lambda: exp_tau_sweep(model, prompt_tokens, child, baseline_child.to("cuda"), outdir),
        ),
        (
            "invariants",
            lambda: exp_invariants(
                model,
                prompt_tokens,
                child,
                baseline_child.to("cuda"),
                parent_baseline.to("cuda"),
            ),
        ),
        (
            "layer_ablation",
            lambda: exp_layer_ablation(model, prompt_tokens, child, baseline_child.to("cuda")),
        ),
        (
            "cache_budget",
            lambda: exp_cache_budget(model, prompt_tokens, child, baseline_child.to("cuda")),
        ),
        (
            "dynamic_tau",
            lambda: exp_dynamic_tau(model, prompt_tokens, child, baseline_child.to("cuda")),
        ),
        ("merge_bench", lambda: exp_merge_bench(model, prompt_tokens)),
    ]

    for name, fn in experiments:
        print(f"--- {name} ---", flush=True)
        t1 = time.time()
        try:
            results[name] = fn()
        except Exception as exc:  # keep going; store the error for the report
            import traceback

            results[name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(),
            }
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
        print(f"  ({time.time() - t1:.1f}s)", flush=True)
        (outdir / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if not args.skip_sampling:
        print("--- sampling ---", flush=True)
        t1 = time.time()
        try:
            results["sampling"] = exp_sampling(model, prompt_tokens, spec, outdir)
        except Exception as exc:
            import traceback

            results["sampling"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(),
            }
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
        print(f"  ({time.time() - t1:.1f}s)", flush=True)

    if "tau_sweep" in results and isinstance(results["tau_sweep"], list):
        print("--- cost_model ---", flush=True)
        try:
            results["cost_model"] = exp_cost_model(model, prompt_tokens, results["tau_sweep"])
        except Exception as exc:
            import traceback

            results["cost_model"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(),
            }
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)

    (outdir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"=== DONE in {time.time() - t0:.1f}s -> {outdir / 'results.json'} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
