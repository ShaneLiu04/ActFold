"""LLaDA mask-diffusion sampler with Branch Folding support.

This is a reference implementation of a masked diffusion sampler that follows
LLaDA's iterative demasking scheme.  It is intended to demonstrate how
Branch Folding can be applied across diffusion timesteps; for production use it
should be validated against the official LLaDA sampling recipe.
"""

from __future__ import annotations

from typing import Any

import torch

from actfold.core.model_wrapper import FoldedModel
from actfold.models.base import DiffusionLLM
from actfold.models.diffusion_sampler import DiffusionSampler
from actfold.utils.logger import get_logger

logger = get_logger("models.llada_sampler")


class LLaDASampler(DiffusionSampler):
    """Masked diffusion sampler for LLaDA-style models.

    Args:
        model: Diffusion model.
        num_steps: Number of demasking steps.
        num_tokens: Number of tokens to generate.
        mask_token_id: Token ID used for masked positions.
    """

    def __init__(
        self,
        model: DiffusionLLM,
        num_steps: int,
        num_tokens: int,
        mask_token_id: int = 126336,  # Common LLaDA mask id; override as needed.
    ) -> None:
        super().__init__(model, num_steps, num_tokens)
        self.mask_token_id = mask_token_id

    def initialize(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Keep the prompt and mask the positions to be generated."""
        batch_size, prompt_len = prompt_ids.shape
        generated = torch.full(
            (batch_size, self.num_tokens),
            self.mask_token_id,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        return torch.cat([prompt_ids, generated], dim=-1)

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        branch_id: Any,
        parent_branch_id: Any | None,
        folded_model: FoldedModel | None,
    ) -> torch.Tensor:
        """Predict tokens for a scheduled number of masked positions."""
        logits = self._forward(
            x_t,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            folded_model=folded_model,
            step_idx=t,
        )
        predictions = logits.argmax(dim=-1)

        mask = x_t == self.mask_token_id
        num_masked = int(mask.sum().item())
        if num_masked == 0:
            return x_t

        # Schedule: unmask a fraction of remaining masked positions.
        confidence = logits.max(dim=-1).values
        confidence = confidence.masked_fill(~mask, float("-inf"))
        flat_conf = confidence.view(-1)
        k = max(1, num_masked * t // self.num_steps)
        _, topk_indices = flat_conf.topk(k)

        x_next = x_t.clone()
        flat_pred = predictions.view(-1)
        x_next.view(-1)[topk_indices] = flat_pred[topk_indices]
        return x_next
