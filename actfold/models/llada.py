"""LLaDA model wrapper."""

from __future__ import annotations

from typing import Any

import torch

from actfold.models.diffusion_sampler import DiffusionSampler
from actfold.models.generic import GenericDiffusionLLM
from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig


class LLaDAModel(GenericDiffusionLLM):
    """Wrapper for the LLaDA (Large Language Diffusion with mAsking) model.

    Uses the reference masked diffusion sampler from
    :class:`~actfold.models.llada_sampler.LLaDASampler`, which follows the
    official LLaDA/MDLM recipe.  If a native LLaDA sampler is installed it can
    be bound via ``_native_sampler``.

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
        """Return the LLaDA masked diffusion sampler."""
        config = kwargs.pop("sampler_config", None)
        if config is None:
            config = LLaDASamplerConfig(
                num_steps=num_steps,
                num_tokens=num_tokens,
                **kwargs,
            )
        if not isinstance(config, LLaDASamplerConfig):
            raise TypeError("LLaDAModel expects LLaDASamplerConfig")
        return LLaDASampler(self, config=config)

    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 16,
        num_steps: int = 128,
        folded_model: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using LLaDA's masked diffusion sampling.

        Falls back to greedy autoregressive decoding when ``num_steps == 1``.
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
