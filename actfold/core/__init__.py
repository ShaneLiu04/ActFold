"""Core Branch Folding engine."""

from actfold.core.activation_cache import ActivationCache
from actfold.core.adaptive_gate import AdaptiveQuantileGate
from actfold.core.branch_manager import Branch, BranchManager
from actfold.core.cache_factory import make_activation_cache
from actfold.core.chunked_cache import ChunkedActivationCache
from actfold.core.fold_plan import FoldPagePlan, PlanStaleError
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.layer_budget import LayerRecomputeBudgetPolicy, apply_layer_budget
from actfold.core.model_wrapper import FoldedModel
from actfold.core.page_table import PageTable
from actfold.core.paged_fold_cache import PagedFoldCache, RolloutBudgetError, RolloutSet
from actfold.core.pyramid_cache import PyramidActivationCache
from actfold.core.pyramid_plan import PyramidPlan, build_pyramid_plan
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
    "FoldPagePlan",
    "PlanStaleError",
    "LayerRecomputeBudgetPolicy",
    "PageTable",
    "PagedFoldCache",
    "RolloutBudgetError",
    "RolloutSet",
    "PyramidActivationCache",
    "PyramidPlan",
    "SimilarityGate",
    "SplitFoldedTransformerLayer",
    "SplitSpec",
    "VectorizedActivationCache",
    "apply_layer_budget",
    "build_pyramid_plan",
    "detect_split_spec",
    "make_activation_cache",
]
