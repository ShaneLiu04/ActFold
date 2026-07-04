"""Speculative decoding integration."""

from actfold.speculative.branch import Branch
from actfold.speculative.draft_generator import DraftGenerator
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
from actfold.speculative.spiffy_baseline import SpiffyBaseline
from actfold.speculative.verification_engine import ActFoldVerificationEngine, VerificationResult

__all__ = [
    "Branch",
    "DraftGenerator",
    "FastDLLMAdapter",
    "SpiffyBaseline",
    "ActFoldVerificationEngine",
    "VerificationResult",
]
