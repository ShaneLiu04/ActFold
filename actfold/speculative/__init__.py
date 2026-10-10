"""Speculative decoding integration."""

from actfold.speculative.acceptance import (
    acceptance_rate,
    draft_region_mask,
    mean_log_prob,
    target_argmax_accept_mask,
)
from actfold.speculative.adaptive_draft_controller import AdaptiveDraftGrowthController
from actfold.speculative.branch import Branch
from actfold.speculative.draft_generator import DraftGenerator
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
from actfold.speculative.folded_generation import FoldedGenerationResult, folded_generate
from actfold.speculative.spiffy_baseline import SpiffyBaseline
from actfold.speculative.verification_engine import ActFoldVerificationEngine, VerificationResult

__all__ = [
    "acceptance_rate",
    "draft_region_mask",
    "mean_log_prob",
    "target_argmax_accept_mask",
    "AdaptiveDraftGrowthController",
    "Branch",
    "DraftGenerator",
    "FastDLLMAdapter",
    "FoldedGenerationResult",
    "SpiffyBaseline",
    "ActFoldVerificationEngine",
    "VerificationResult",
    "folded_generate",
]
