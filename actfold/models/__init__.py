"""Real Diffusion LLM model adapters and registry."""

from __future__ import annotations

from actfold.models.architecture_utils import (
    ArchitectureProfile,
    ManualFoldedForward,
    build_manual_folded_forward,
    detect_architecture,
    find_embedding_module,
    find_layer_list,
    find_lm_head,
)
from actfold.models.base import DiffusionLLM
from actfold.models.diffusion_sampler import DiffusionSampler, SamplerConfig, SamplerOutput
from actfold.models.dream_sampler import DreamSampler, DreamSamplerConfig
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig
from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig
from actfold.models.registry import ModelRegistry, load_model
from actfold.models.sampling_utils import (
    CosineMaskingScheduler,
    LinearMaskingScheduler,
    MaskingScheduler,
    add_gumbel_noise,
    build_left_padded_canvas,
    build_right_padded_canvas,
    compute_position_ids,
    get_num_transfer_tokens,
    make_masking_scheduler,
    right_shift_logits,
    sample_tokens,
    top_k_logits,
    top_p_logits,
)

__all__ = [
    "ArchitectureProfile",
    "CosineMaskingScheduler",
    "DiffusionLLM",
    "DiffusionSampler",
    "DreamSampler",
    "DreamSamplerConfig",
    "FastDLLMSampler",
    "FastDLLMSamplerConfig",
    "LinearMaskingScheduler",
    "LLaDASampler",
    "LLaDASamplerConfig",
    "ManualFoldedForward",
    "MaskingScheduler",
    "ModelRegistry",
    "SamplerConfig",
    "SamplerOutput",
    "add_gumbel_noise",
    "build_left_padded_canvas",
    "build_manual_folded_forward",
    "build_right_padded_canvas",
    "compute_position_ids",
    "detect_architecture",
    "find_embedding_module",
    "find_layer_list",
    "find_lm_head",
    "get_num_transfer_tokens",
    "load_model",
    "make_masking_scheduler",
    "right_shift_logits",
    "sample_tokens",
    "top_k_logits",
    "top_p_logits",
]
