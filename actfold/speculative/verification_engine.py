"""ActFold-accelerated verification engine for speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from actfold.core.cache_factory import ActivationCacheType
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.similarity_gate import SimilarityGate
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER, StabilityProfile
from actfold.speculative.branch import Branch
from actfold.speculative.fast_dllm_adapter import DiffusionLLMAdapter
from actfold.utils.cost_model import ComputeBandwidthCostModel, HardwareProfile
from actfold.utils.flops_counter import count_diffusion_llm_flops
from actfold.utils.gpu_profiler import gpu_profile


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying a child branch against its parent."""

    accepted: bool
    child_branch: Branch
    parent_branch: Branch
    stable_ratio: float
    tflops: float
    latency_ms: float
    estimated_latency_ms: float = 0.0
    stability_profile: StabilityProfile | None = None


class ActFoldVerificationEngine:
    """Verify child branches by reusing parent activations where stable.

    The engine uses real input embeddings to estimate the token-level stable
    ratio between a parent and child branch. The actual layer-wise folding is
    performed by the model's own folded forward path (e.g. via
    :class:`~actfold.core.model_wrapper.FoldedModel`); this engine provides the
    acceptance decision and FLOPs estimate for speculative decoding.

    Args:
        model: Model adapter.
        cache: Activation cache shared across branches.
        gate: Similarity gate.
        scheduler: Optional folding scheduler for dynamic tau.
        acceptance_threshold: Minimum stable ratio for a branch to be accepted.
            A value of ``0.0`` accepts all branches (useful for research
            benchmarking). Increase to make verification more conservative.
    """

    def __init__(
        self,
        model: DiffusionLLMAdapter,
        cache: ActivationCacheType,
        gate: SimilarityGate,
        scheduler: FoldingScheduler | None = None,
        acceptance_threshold: float = 0.0,
        cost_model: ComputeBandwidthCostModel | None = None,
    ) -> None:
        if not 0.0 <= acceptance_threshold <= 1.0:
            raise ValueError(f"acceptance_threshold must be in [0, 1], got {acceptance_threshold}")

        self.model = model
        self.cache = cache
        self.gate = gate
        self.scheduler = scheduler
        self.acceptance_threshold = acceptance_threshold
        self._current_seq_len: int = 1
        self.cost_model = cost_model or ComputeBandwidthCostModel(
            HardwareProfile.from_device(cache.device)
        )

    def verify_branch(
        self,
        parent_branch: Branch,
        child_branch: Branch,
        step_idx: int = 0,
    ) -> VerificationResult:
        """Verify ``child_branch`` using activations from ``parent_branch``.

        Args:
            parent_branch: Parent branch with cached activations.
            child_branch: Child branch to verify.
            step_idx: Current diffusion step index.

        Returns:
            VerificationResult with acceptance decision and metrics.
        """
        # Track the current sequence length so TFLOPs estimates use real values.
        self._current_seq_len = max(1, child_branch.tokens.shape[1])

        # Pre-populate parent cache with real embeddings if not already present.
        self._ensure_parent_cache(parent_branch)

        # If the adapter has a folded model, run the parent forward first so that
        # all layer caches are populated with real activations.
        self._ensure_parent_layers(parent_branch, step_idx)

        # Reset any stale profile for the child branch before running the
        # folded forward pass so the profiler only sees the current verification.
        GLOBAL_STABILITY_PROFILER.reset_branch(child_branch.branch_id)

        with gpu_profile(device=self.cache.device) as measurement:
            logits = self.model.forward(
                child_branch.tokens,
                branch_id=child_branch.branch_id,
                parent_branch_id=parent_branch.branch_id,
                step_idx=step_idx,
            )

        stable_ratio = self._estimate_stable_ratio(
            parent_branch,
            child_branch,
            logits,
            step_idx,
        )

        tflops = self._estimate_tflops(stable_ratio)

        score = logits.float().mean().item()
        child_branch.metadata["actfold_score"] = score
        child_branch.metadata["stable_ratio"] = stable_ratio

        accepted = stable_ratio >= self.acceptance_threshold
        child_branch.accepted = accepted

        if not accepted:
            self.cache.clear_branch(child_branch.branch_id)

        profile = GLOBAL_STABILITY_PROFILER.get_profile(child_branch.branch_id)
        estimated_latency_ms = self._estimate_latency(stable_ratio, profile)

        return VerificationResult(
            accepted=accepted,
            child_branch=child_branch,
            parent_branch=parent_branch,
            stable_ratio=stable_ratio,
            tflops=tflops,
            latency_ms=measurement.latency_ms,
            estimated_latency_ms=estimated_latency_ms,
            stability_profile=profile,
        )

    def _ensure_parent_layers(
        self,
        parent_branch: Branch,
        step_idx: int,
    ) -> None:
        """Run the parent branch through the folded forward path if available.

        This populates layer-wise FFN caches so that child verification can
        reuse real parent activations. If the adapter has no folded model, only
        the layer-0 embedding cache (from :meth:`_ensure_parent_cache`) is used
        and the stable-ratio estimate still relies on real embeddings.
        """
        folded = getattr(self.model, "folded_model", None)
        if folded is None:
            return
        GLOBAL_STABILITY_PROFILER.reset_branch(parent_branch.branch_id)
        with torch.no_grad():
            self.model.forward(
                parent_branch.tokens,
                branch_id=parent_branch.branch_id,
                parent_branch_id=None,
                step_idx=step_idx,
            )

    def _ensure_parent_cache(self, parent_branch: Branch) -> None:
        """Populate the layer-0 cache entry for the parent branch if missing.

        Only the input embeddings are stored; they are used by
        :meth:`_estimate_stable_ratio` to measure the real parent/child
        similarity. Layer-wise FFN outputs must be populated by the model's
        folded forward path (e.g. ``FoldedModel``) prior to verifying children.
        """
        hidden = self._token_to_hidden(parent_branch.tokens)
        dummy_mask = torch.ones(
            hidden.shape[:2],
            dtype=torch.bool,
            device=hidden.device,
        )
        try:
            self.cache.get(
                branch_id=parent_branch.branch_id,
                layer_idx=0,
                token_mask=dummy_mask,
            )
        except (KeyError, RuntimeError):
            self.cache.put(
                branch_id=parent_branch.branch_id,
                layer_idx=0,
                activations={
                    "hidden_states": hidden,
                },
            )

    def _token_to_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Look up real input embeddings for ``tokens``.

        Raises:
            RuntimeError: If the wrapped model does not expose an embedding
                layer and no real hidden states are available.
        """
        return self.model.embed(tokens)

    def _estimate_stable_ratio(
        self,
        parent_branch: Branch,
        child_branch: Branch,
        child_logits: torch.Tensor,
        step_idx: int,
    ) -> float:
        """Estimate the fraction of stable tokens using the similarity gate.

        When the child forward pass was executed through a folded model, the
        :class:`~actfold.profiler.stability_profiler.StabilityProfiler` records
        per-layer stable ratios.  In that case we return the mean stable ratio
        across layers, which reflects the actual folding behaviour.  Otherwise we
        fall back to comparing the input embeddings at layer 0.

        ``child_logits`` is accepted for API symmetry with the forward pass but
        is not needed for the embedding-based estimate.
        """
        del child_logits

        # Prefer real layer-wise profile when available.
        profile = GLOBAL_STABILITY_PROFILER.get_profile(child_branch.branch_id)
        if profile is not None and profile.layer_stats:
            return float(profile.mean_stable_ratio)

        # Fallback: compare input embeddings.
        h_child = self._token_to_hidden(child_branch.tokens)
        h_parent = self._token_to_hidden(parent_branch.tokens)

        if h_child.shape != h_parent.shape:
            raise ValueError(
                f"Parent/child embedding shape mismatch: "
                f"parent {h_parent.shape} vs child {h_child.shape}"
            )

        if self.scheduler is not None:
            tau = self.scheduler.get_tau(
                layer_idx=0,
                step_idx=step_idx,
                task_type="general",
            )
            self.gate.set_tau(tau)

        stable_mask = self.gate(h_child, h_parent)
        stable_ratio = stable_mask.float().mean().item()
        return float(min(max(stable_ratio, 0.0), 1.0))

    def _estimate_tflops(self, stable_ratio: float) -> float:
        """Estimate TFLOPs for verification with the given stable ratio."""
        flops = count_diffusion_llm_flops(
            num_layers=self.model.num_layers,
            hidden_dim=self.model.hidden_dim,
            num_heads=max(1, self.model.num_heads),
            seq_len=self._current_seq_len,
            vocab_size=self.model.vocab_size,
            num_steps=1,
            reuse_ratio=stable_ratio,
        )
        return flops.total_tflops

    def _estimate_latency(
        self,
        stable_ratio: float,
        profile: StabilityProfile | None,
    ) -> float:
        """Estimate latency using the compute-bandwidth-aware cost model.

        If a layer-wise stability profile is available, the mean stable ratio
        across layers is used; otherwise the embedding-level stable ratio is
        used as a fallback.
        """
        if profile is not None and profile.layer_stats:
            effective_ratio = profile.mean_stable_ratio
        else:
            effective_ratio = stable_ratio

        return (
            self.cost_model.estimate_total_time(
                num_layers=self.model.num_layers,
                seq_len=self._current_seq_len,
                hidden_dim=self.model.hidden_dim,
                stable_ratio=effective_ratio,
                num_steps=1,
                num_heads=max(1, self.model.num_heads),
            )
            * 1000.0
        )  # seconds -> ms
