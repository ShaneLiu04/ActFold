"""Systematic ablation study framework.

All sweeps are real measurements: the study wraps the adapter's raw module
in a context-managed :class:`~actfold.core.model_wrapper.FoldedModel`, runs a
parent forward to populate the activation cache, then a folded child forward
whose per-layer stability is recorded by the global stability profiler.
Layer-wise ablations disable folding outside the requested range via
:class:`~actfold.core.folding_scheduler.FoldingScheduler.disabled_layers`,
and the cache sweep uses budgets small enough to force real eviction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.model_wrapper import FoldedModel
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from actfold.speculative.branch import Branch
from actfold.speculative.draft_generator import DraftGenerator
from actfold.speculative.fast_dllm_adapter import DiffusionLLMAdapter, FastDLLMAdapter
from actfold.utils.flops_counter import count_diffusion_llm_flops, model_ffn_flops_kwargs
from actfold.utils.logger import get_logger

logger = get_logger("ablation")

#: Base tau for sweeps that are not explicitly parameterized on tau.
_SWEEP_TAU = 0.95

#: Diffusion-step horizon for the per-measurement scheduler.
_SWEEP_NUM_STEPS = 10


@dataclass(frozen=True)
class FoldedMeasurement:
    """Result of one real folded measurement.

    Attributes:
        stable_ratio: Mean stable ratio over the layers that actually folded
            (0.0 when no layer folded).
        per_layer_stable: Measured stable ratio per folded layer index.
            Layers that were disabled or recomputed are absent (they save
            nothing).
        folded_layer_count: Number of layers that actually folded, i.e.
            ``len(per_layer_stable)``.
        actfold_tflops: Baseline TFLOPs minus the measured per-layer savings.
        baseline_tflops: TFLOPs of the full recomputation.
        reduction_pct: ``100 * (baseline - actfold) / baseline``.
    """

    stable_ratio: float
    per_layer_stable: dict[int, float]
    folded_layer_count: int
    actfold_tflops: float
    baseline_tflops: float
    reduction_pct: float


class AblationStudy:
    """Run ablation experiments across thresholds, layers, and cache sizes.

    Args:
        model: Model adapter wrapping the raw module to measure. It must not
            carry its own ``folded_model`` (the study manages its own folded
            stack) and must expose the raw module via ``underlying_model``.
        baseline: Optional baseline adapter (kept for API compatibility).
        draft_generator: Draft generator used to create child branches.
            When ``None``, a ``suffix_append`` generator is used so sweeps
            measure realistic parent-like children.
        vocab_size: Vocabulary size used for FLOPs accounting.
        seq_len: Sequence length of the synthetic parent branch.
        device: Device on which caches and gates are allocated.
    """

    def __init__(
        self,
        model: DiffusionLLMAdapter,
        baseline: DiffusionLLMAdapter | None,
        draft_generator: DraftGenerator | None = None,
        vocab_size: int = 1000,
        seq_len: int = 16,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.baseline = baseline
        self.draft_generator = draft_generator or DraftGenerator(
            vocab_size=vocab_size, mode="suffix_append"
        )
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device

    def _create_parent_branch(self, seed: int = 42) -> Branch:
        """Create a deterministic parent branch for ablation inputs."""
        torch.manual_seed(seed)
        tokens = torch.randint(0, self.vocab_size, (1, self.seq_len), device=self.device)
        return Branch(branch_id="root", parent_id=None, tokens=tokens)

    def _underlying_module(self) -> nn.Module:
        """Return the raw module to wrap, enforcing the measurement contract.

        Raises:
            ValueError: If the adapter already carries a ``folded_model``;
                the study must manage its own folded stack.
            TypeError: If the adapter exposes no ``underlying_model``
                ``nn.Module`` to wrap.
        """
        if getattr(self.model, "folded_model", None) is not None:
            raise ValueError(
                "AblationStudy manages its own folded stack for measurement; pass "
                "the raw adapter without a folded_model."
            )
        raw: Any = getattr(self.model, "underlying_model", None)
        if not isinstance(raw, nn.Module):
            raise TypeError(
                "AblationStudy requires an adapter that exposes the raw module via "
                "'underlying_model' (e.g. FastDLLMAdapter) so it can wrap it in a "
                "FoldedModel."
            )
        return raw

    def _flops_budget(self) -> tuple[float, float]:
        """Return ``(baseline_total_tflops, per_layer_reusable_tflops)``.

        The reusable share is the attention + FFN work of one layer: exactly
        the part that folding one layer with stable ratio ``s`` saves
        ``s * per_layer_reusable`` of.
        """
        flops = count_diffusion_llm_flops(
            num_layers=self.model.num_layers,
            hidden_dim=self.model.hidden_dim,
            num_heads=max(1, self.model.num_heads),
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            num_steps=1,
            reuse_ratio=0.0,
            **model_ffn_flops_kwargs(self.model),
        )
        reusable = flops.attention_tflops + flops.ffn_tflops
        per_layer_reusable = reusable / self.model.num_layers
        return flops.total_tflops, per_layer_reusable

    def measure_folding(
        self,
        tau: float = _SWEEP_TAU,
        disabled_layers: set[int] | None = None,
        max_entries_per_layer: int = 1024,
        seed: int = 42,
    ) -> FoldedMeasurement:
        """Run one real folded measurement and return per-layer statistics.

        The raw module is wrapped in a fresh :class:`FoldedModel` (restored
        on exit), a parent forward populates the cache, and a folded child
        forward records per-layer stability in the global profiler.

        Args:
            tau: Base similarity threshold for the measurement gate.
            disabled_layers: Layers excluded from folding (recompute instead).
            max_entries_per_layer: Token budget per cache layer; small values
                force real eviction during the child pass.
            seed: Seed for the deterministic parent/child branches.

        Returns:
            FoldedMeasurement with measured per-layer stable ratios and the
            FLOPs savings implied by them.

        Raises:
            ValueError: If the adapter already carries a ``folded_model``.
            TypeError: If the raw module cannot be obtained from the adapter.
            RuntimeError: If no Transformer layer stack is discovered in the
                raw module, so no folding can be measured.
        """
        rng_state = torch.get_rng_state()
        try:
            parent = self._create_parent_branch(seed=seed)
            child = self.draft_generator.generate(parent, num_branches=1, seed=seed)[0]

            raw = self._underlying_module()
            cache = ActivationCache(
                max_entries_per_layer=max_entries_per_layer, device=self.device
            )
            gate = SimilarityGate(tau=tau, metric="cosine")
            scheduler = FoldingScheduler(
                base_tau=tau,
                num_layers=self.model.num_layers,
                num_steps=_SWEEP_NUM_STEPS,
                disabled_layers=disabled_layers,
            )
            baseline_total, per_layer_reusable = self._flops_budget()

            with FoldedModel(raw, cache, gate, scheduler=scheduler) as folded:
                if not folded.folding_applied:
                    raise RuntimeError(
                        "AblationStudy could not discover a Transformer layer stack "
                        "in the underlying model; layer-wise folding cannot be "
                        "measured. Check that the model exposes a standard layer "
                        "ModuleList (see FoldedModel._DEFAULT_LAYER_PATHS)."
                    )
                measure_adapter = FastDLLMAdapter(
                    raw,
                    num_layers=self.model.num_layers,
                    hidden_dim=self.model.hidden_dim,
                    num_heads=max(1, self.model.num_heads),
                    vocab_size=self.model.vocab_size,
                    folded_model=folded,
                )
                with torch.no_grad():
                    # Parent pass: populates every layer's cache with real
                    # activations (no folding, no profiler records).
                    measure_adapter.forward(parent.tokens, branch_id=parent.branch_id, step_idx=0)
                    # Child pass: folded layers record real per-layer stability.
                    GLOBAL_STABILITY_PROFILER.reset_branch(child.branch_id)
                    measure_adapter.forward(
                        child.tokens,
                        branch_id=child.branch_id,
                        parent_branch_id=parent.branch_id,
                        step_idx=0,
                    )

            profile = GLOBAL_STABILITY_PROFILER.get_profile(child.branch_id)
            per_layer: dict[int, float] = {}
            if profile is not None:
                for stat in profile.layer_stats:
                    per_layer[int(stat.layer_idx)] = float(stat.stable_ratio)

            savings = sum(per_layer.values()) * per_layer_reusable
            actfold_tflops = max(0.0, baseline_total - savings)
            stable_ratio = (
                sum(per_layer.values()) / len(per_layer) if per_layer else 0.0
            )
            reduction = 100.0 * savings / baseline_total if baseline_total > 0 else 0.0
            return FoldedMeasurement(
                stable_ratio=float(stable_ratio),
                per_layer_stable=per_layer,
                folded_layer_count=len(per_layer),
                actfold_tflops=actfold_tflops,
                baseline_tflops=baseline_total,
                reduction_pct=reduction,
            )
        finally:
            torch.set_rng_state(rng_state)

    def run_threshold_sensitivity(
        self,
        taus: list[float] | None = None,
    ) -> pd.DataFrame:
        """Measure TFLOPs reduction across similarity thresholds.

        Each tau is measured by a real folded forward pass with
        ``suffix_append`` drafts (or the caller-provided generator).

        Args:
            taus: List of tau values to test. Defaults to
                ``[0.90, 0.95, 0.99]``.

        Returns:
            DataFrame with columns: tau, stable_ratio, folded_layer_count,
            actfold_tflops, baseline_tflops, tflops_reduction_pct.
        """
        if taus is None:
            taus = [0.90, 0.95, 0.99]

        records: list[dict[str, Any]] = []
        for tau in taus:
            measurement = self.measure_folding(tau=tau)
            records.append(
                {
                    "tau": tau,
                    "stable_ratio": measurement.stable_ratio,
                    "folded_layer_count": measurement.folded_layer_count,
                    "actfold_tflops": measurement.actfold_tflops,
                    "baseline_tflops": measurement.baseline_tflops,
                    "tflops_reduction_pct": measurement.reduction_pct,
                }
            )

        return pd.DataFrame(records)

    def run_layerwise_folding(
        self,
        layer_ranges: list[tuple[int, int]] | None = None,
    ) -> pd.DataFrame:
        """Measure the impact of folding only specific layer ranges.

        For each range, folding is disabled outside ``[start, end]`` via
        :attr:`FoldingScheduler.disabled_layers` and the reduction is
        measured from the real per-layer stable ratios. The legacy linear
        extrapolation (uniform full-model ratio scaled by the folded
        fraction) is reported alongside as ``linear_estimate_pct``.

        Args:
            layer_ranges: List of ``(start_layer, end_layer)`` inclusive
                ranges. Defaults to lower half / upper half / full model.

        Returns:
            DataFrame with columns: start_layer, end_layer, folded_layers,
            measured_stable_ratio, folded_layer_count,
            estimated_reduction_pct (measured), linear_estimate_pct,
            full_model_stable_ratio.
        """
        num_layers = self.model.num_layers
        if layer_ranges is None:
            layer_ranges = [
                (0, num_layers // 2 - 1),
                (num_layers // 2, num_layers - 1),
                (0, num_layers - 1),
            ]

        full = self.measure_folding(tau=_SWEEP_TAU)
        baseline_total, per_layer_reusable = self._flops_budget()
        reusable_total = per_layer_reusable * num_layers

        records: list[dict[str, Any]] = []
        for start, end in layer_ranges:
            num_folded = end - start + 1
            disabled = {idx for idx in range(num_layers) if not start <= idx <= end}
            measurement = self.measure_folding(tau=_SWEEP_TAU, disabled_layers=disabled)
            # Legacy linear extrapolation for comparison: assume every layer
            # is as stable as the full-model mean and scale by the folded
            # fraction of the (uniform) reusable work.
            linear_pct = (
                100.0
                * full.stable_ratio
                * (num_folded / num_layers)
                * (reusable_total / baseline_total)
                if baseline_total > 0
                else 0.0
            )
            records.append(
                {
                    "start_layer": start,
                    "end_layer": end,
                    "folded_layers": num_folded,
                    "measured_stable_ratio": measurement.stable_ratio,
                    "folded_layer_count": measurement.folded_layer_count,
                    "estimated_reduction_pct": measurement.reduction_pct,
                    "linear_estimate_pct": linear_pct,
                    "full_model_stable_ratio": full.stable_ratio,
                }
            )

        return pd.DataFrame(records)

    def run_all(
        self,
        output_dir: str | Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run all ablation studies and optionally persist CSVs.

        Args:
            output_dir: Optional directory to write
                ``threshold_sensitivity.csv``, ``layerwise_folding.csv``, and
                ``cache_size_impact.csv``.

        Returns:
            Dictionary mapping study name to DataFrame.
        """
        results = {
            "threshold_sensitivity": self.run_threshold_sensitivity(),
            "layerwise_folding": self.run_layerwise_folding(),
            "cache_size_impact": self.run_cache_size_impact(),
        }

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            for name, df in results.items():
                path = output_path / f"{name}.csv"
                df.to_csv(path, index=False)
                logger.info("Saved %s ablation results to %s", name, path)

        return results

    def run_cache_size_impact(
        self,
        cache_sizes: list[int] | None = None,
    ) -> pd.DataFrame:
        """Measure TFLOPs reduction across cache budgets.

        Defaults scale with ``seq_len``: ``[seq_len, 2*seq_len,
        4*seq_len]``. The smallest budget cannot hold both the parent and
        the child activations, so the parent's entries are really evicted
        during the child pass and deeper layers must recompute — the sweep
        measures the resulting degradation instead of a no-op curve.

        Args:
            cache_sizes: List of ``max_entries_per_layer`` values.

        Returns:
            DataFrame with columns: cache_size, stable_ratio,
            folded_layer_count, actfold_tflops, baseline_tflops,
            tflops_reduction_pct.
        """
        if cache_sizes is None:
            cache_sizes = [self.seq_len, 2 * self.seq_len, 4 * self.seq_len]

        records: list[dict[str, Any]] = []
        for size in cache_sizes:
            measurement = self.measure_folding(
                tau=_SWEEP_TAU,
                max_entries_per_layer=size,
            )
            records.append(
                {
                    "cache_size": size,
                    "stable_ratio": measurement.stable_ratio,
                    "folded_layer_count": measurement.folded_layer_count,
                    "actfold_tflops": measurement.actfold_tflops,
                    "baseline_tflops": measurement.baseline_tflops,
                    "tflops_reduction_pct": measurement.reduction_pct,
                }
            )

        return pd.DataFrame(records)
