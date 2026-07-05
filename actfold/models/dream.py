"""Dream model wrapper."""

from __future__ import annotations

from typing import Any

import torch

from actfold.models.diffusion_sampler import DiffusionSampler
from actfold.models.dream_sampler import DreamSampler
from actfold.models.generic import GenericDiffusionLLM


class DreamModel(GenericDiffusionLLM):
    """Wrapper for the Dream Diffusion LLM family.

    When the official Dream implementation is available, this class can be
    extended to call its native diffusion sampling routine. By default it
    falls back to the generic AutoModel wrapper.

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
    ) -> DiffusionSampler | None:
        """Return the reference Dream continuous diffusion sampler."""
        return DreamSampler(self, num_steps=num_steps, num_tokens=num_tokens)

    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 16,
        num_steps: int = 64,
        folded_model: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using Dream's diffusion sampling.

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
