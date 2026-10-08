"""Core Branch Folding engine."""

from actfold.core.activation_cache import ActivationCache
from actfold.core.adaptive_gate import AdaptiveQuantileGate
from actfold.core.branch_manager import Branch, BranchManager
from actfold.core.cache_factory import make_activation_cache
from actfold.core.chunked_cache import ChunkedActivationCache
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.model_wrapper import FoldedModel
from actfold.core.similarity_gate import SimilarityGate
from actfold.core.split_layer import SplitFoldedTransformerLayer, SplitSpec, detect_split_spec
from actfold.core.vectorized_cache import VectorizedActivationCache

__all__ = [
    "ActivationCache",
    "AdaptiveQuantileGate",
    "Branch",
    "BranchManager",
    "ChunkedActivationCache",
    "FoldedModel",
    "FoldedTransformerLayer",
    "FoldingScheduler",
    "SimilarityGate",
    "SplitFoldedTransformerLayer",
    "SplitSpec",
    "VectorizedActivationCache",
    "detect_split_spec",
    "make_activation_cache",
]
