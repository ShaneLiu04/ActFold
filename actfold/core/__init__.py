"""Core Branch Folding engine."""

from actfold.core.activation_cache import ActivationCache
from actfold.core.branch_manager import Branch, BranchManager
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.model_wrapper import FoldedModel
from actfold.core.similarity_gate import SimilarityGate

__all__ = [
    "ActivationCache",
    "Branch",
    "BranchManager",
    "FoldedModel",
    "FoldedTransformerLayer",
    "FoldingScheduler",
    "SimilarityGate",
]
