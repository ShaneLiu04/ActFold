"""Systematic ablation study framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.speculative import ActFoldVerificationEngine, DraftGenerator
from actfold.speculative.branch import Branch
from actfold.speculative.fast_dllm_adapter import DiffusionLLMAdapter
from actfold.utils.flops_counter import count_diffusion_llm_flops
from actfold.utils.logger import get_logger

logger = get_logger("ablation")


class AblationStudy:
    """Run ablation experiments across thresholds, layers, and cache sizes."""

    def __init__(
        self,
        model: DiffusionLLMAdapter,
        baseline: DiffusionLLMAdapter | None,
        draft_generator: DraftGenerator,
        vocab_size: int = 1000,
        seq_len: int = 16,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.baseline = baseline
        self.draft_generator = draft_generator
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device

    def _create_parent_branch(self, seed: int = 42) -> Branch:
        """Create a deterministic parent branch for ablation inputs."""
        torch.manual_seed(seed)
        tokens = torch.randint(0, self.vocab_size, (1, self.seq_len), device=self.device)
        return Branch(branch_id="root", parent_id=None, tokens=tokens)

    def run_threshold_sensitivity(
        self,
        taus: list[float] | None = None,
    ) -> pd.DataFrame:
        """Measure TFLOPs reduction and estimated accuracy across thresholds.

        Args:
            taus: List of tau values to test.

        Returns:
            DataFrame with columns: tau, stable_ratio, tflops, tflops_reduction.
        """
        if taus is None:
            taus = [0.90, 0.95, 0.99]

        records: list[dict[str, Any]] = []
        parent = self._create_parent_branch()
        children = self.draft_generator.generate(parent, num_branches=2, seed=42)

        # Measure the full-model stable ratio once at the default tau so the
        # layer-wise estimate below is grounded in a real observation.
        full_cache = ActivationCache(max_entries_per_layer=512, device=self.device)
        full_gate = SimilarityGate(tau=0.95, metric="cosine")
        full_scheduler = FoldingScheduler(
            base_tau=0.95,
            num_layers=self.model.num_layers,
            num_steps=10,
        )
        full_engine = ActFoldVerificationEngine(self.model, full_cache, full_gate, full_scheduler)
        full_result = full_engine.verify_branch(parent, children[0], step_idx=0)
        full_stable_ratio = full_result.stable_ratio

        for tau in taus:
            cache = ActivationCache(max_entries_per_layer=512, device=self.device)
            gate = SimilarityGate(tau=tau, metric="cosine")
            scheduler = FoldingScheduler(
                base_tau=tau,
                num_layers=self.model.num_layers,
                num_steps=10,
            )
            engine = ActFoldVerificationEngine(self.model, cache, gate, scheduler)
            result = engine.verify_branch(parent, children[0], step_idx=0)

            base_tflops = count_diffusion_llm_flops(
                num_layers=self.model.num_layers,
                hidden_dim=self.model.hidden_dim,
                num_heads=max(1, self.model.num_heads),
                seq_len=self.seq_len,
                vocab_size=self.vocab_size,
                num_steps=1,
                reuse_ratio=0.0,
            ).total_tflops

            reduction = 100.0 * (1.0 - result.tflops / base_tflops) if base_tflops > 0 else 0.0
            records.append(
                {
                    "tau": tau,
                    "stable_ratio": result.stable_ratio,
                    "actfold_tflops": result.tflops,
                    "baseline_tflops": base_tflops,
                    "tflops_reduction_pct": reduction,
                    "full_model_stable_ratio": full_stable_ratio,
                }
            )

        return pd.DataFrame(records)

    def run_layerwise_folding(
        self,
        layer_ranges: list[tuple[int, int]] | None = None,
    ) -> pd.DataFrame:
        """Measure the impact of folding only specific layer ranges.

        Args:
            layer_ranges: List of (start_layer, end_layer) inclusive ranges.

        Returns:
            DataFrame with columns: start_layer, end_layer, estimated_reduction.
        """
        if layer_ranges is None:
            num_layers = self.model.num_layers
            layer_ranges = [
                (0, num_layers // 2 - 1),
                (num_layers // 2, num_layers - 1),
                (0, num_layers - 1),
            ]

        records: list[dict[str, Any]] = []

        # Ground the estimate in a real full-model measurement.
        parent = self._create_parent_branch()
        children = self.draft_generator.generate(parent, num_branches=2, seed=42)
        full_cache = ActivationCache(max_entries_per_layer=512, device=self.device)
        full_gate = SimilarityGate(tau=0.95, metric="cosine")
        full_scheduler = FoldingScheduler(
            base_tau=0.95,
            num_layers=self.model.num_layers,
            num_steps=10,
        )
        full_engine = ActFoldVerificationEngine(self.model, full_cache, full_gate, full_scheduler)
        full_result = full_engine.verify_branch(parent, children[0], step_idx=0)
        full_stable_ratio = full_result.stable_ratio

        for start, end in layer_ranges:
            num_folded = end - start + 1
            # Scale the measured full-model stable ratio by the fraction of
            # layers that are folded, replacing the previous hardcoded 0.7.
            avg_stable = full_stable_ratio * num_folded / self.model.num_layers
            tflops = count_diffusion_llm_flops(
                num_layers=self.model.num_layers,
                hidden_dim=self.model.hidden_dim,
                num_heads=max(1, self.model.num_heads),
                seq_len=self.seq_len,
                vocab_size=self.vocab_size,
                num_steps=1,
                reuse_ratio=avg_stable,
            ).total_tflops
            base_tflops = count_diffusion_llm_flops(
                num_layers=self.model.num_layers,
                hidden_dim=self.model.hidden_dim,
                num_heads=max(1, self.model.num_heads),
                seq_len=self.seq_len,
                vocab_size=self.vocab_size,
                num_steps=1,
                reuse_ratio=0.0,
            ).total_tflops
            reduction = 100.0 * (1.0 - tflops / base_tflops) if base_tflops > 0 else 0.0
            records.append(
                {
                    "start_layer": start,
                    "end_layer": end,
                    "folded_layers": num_folded,
                    "estimated_reduction_pct": reduction,
                    "full_model_stable_ratio": full_stable_ratio,
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

        Args:
            cache_sizes: List of max_entries_per_layer values.

        Returns:
            DataFrame with columns: cache_size, stable_ratio, tflops_reduction.
        """
        if cache_sizes is None:
            cache_sizes = [256, 512, 1024, 2048]

        records: list[dict[str, Any]] = []
        parent = self._create_parent_branch()
        children = self.draft_generator.generate(parent, num_branches=2, seed=42)

        for size in cache_sizes:
            cache = ActivationCache(max_entries_per_layer=size, device=self.device)
            gate = SimilarityGate(tau=0.95, metric="cosine")
            scheduler = FoldingScheduler(
                base_tau=0.95,
                num_layers=self.model.num_layers,
                num_steps=10,
            )
            engine = ActFoldVerificationEngine(self.model, cache, gate, scheduler)
            result = engine.verify_branch(parent, children[0], step_idx=0)

            base_tflops = count_diffusion_llm_flops(
                num_layers=self.model.num_layers,
                hidden_dim=self.model.hidden_dim,
                num_heads=max(1, self.model.num_heads),
                seq_len=self.seq_len,
                vocab_size=self.vocab_size,
                num_steps=1,
                reuse_ratio=0.0,
            ).total_tflops

            reduction = 100.0 * (1.0 - result.tflops / base_tflops) if base_tflops > 0 else 0.0
            records.append(
                {
                    "cache_size": size,
                    "stable_ratio": result.stable_ratio,
                    "actfold_tflops": result.tflops,
                    "tflops_reduction_pct": reduction,
                }
            )

        return pd.DataFrame(records)
