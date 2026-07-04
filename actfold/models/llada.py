"""LLaDA model wrapper."""

from __future__ import annotations

from typing import Any

import torch

from actfold.models.generic import GenericDiffusionLLM


class LLaDAModel(GenericDiffusionLLM):
    """Wrapper for the LLaDA (Large Language Diffusion with mAsking) model.

    When the official LLaDA implementation is available, this class can be
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
        try:
            # If a native LLaDA sampler is installed, bind it here.
            # Example: from llada import generate as llada_generate
            # self._native_sampler = llada_generate
            pass
        except ImportError:
            self._native_sampler = None

    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 16,
        num_steps: int = 128,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using LLaDA's masked diffusion sampling if available.

        Falls back to greedy autoregressive decoding otherwise.
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
            **kwargs,
        )
