"""Fast-dLLM model wrapper."""

from __future__ import annotations

from typing import Any

import torch

from actfold.models.causal_lm import CausalLMDiffusionLLM
from actfold.models.diffusion_sampler import DiffusionSampler
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig


class FastDLLMModel(CausalLMDiffusionLLM):
    """Wrapper for the Fast-dLLM (Fast Diffusion LLM) family.

    Fast-dLLM models are typically published as causal LMs with custom
    speculative decoding support.  This wrapper uses
    :class:`~actfold.models.fast_dllm_sampler.FastDLLMSampler` as the
    reference discrete diffusion sampler, which follows the Fast-dLLM v2
    block-wise masked decoding recipe.

    Args:
        model_name_or_path: Hugging Face model identifier or local path.
        trust_remote_code: Whether to trust remote code in the model repo.
    """

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__(model_name_or_path, trust_remote_code=trust_remote_code)
        self._native_sampler: Any | None = None

    def get_native_sampler(
        self,
        num_steps: int,
        num_tokens: int,
        **kwargs: Any,
    ) -> DiffusionSampler | None:
        """Return the Fast-dLLM discrete diffusion sampler."""
        config = kwargs.pop("sampler_config", None)
        if config is None:
            config = FastDLLMSamplerConfig(
                num_steps=num_steps,
                num_tokens=num_tokens,
                **kwargs,
            )
        if not isinstance(config, FastDLLMSamplerConfig):
            raise TypeError("FastDLLMModel expects FastDLLMSamplerConfig")
        return FastDLLMSampler(self, config=config)

    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 16,
        num_steps: int = 10,
        folded_model: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using Fast-dLLM sampling if available.

        Falls back to the causal LM generate method otherwise.
        """
        if self._native_sampler is not None:
            tokens: torch.Tensor = self._native_sampler(
                self.model,
                self.tokenizer,
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                num_steps=num_steps,
                **kwargs,
            )
            return tokens
        return super().generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            num_steps=num_steps,
            folded_model=folded_model,
            **kwargs,
        )
