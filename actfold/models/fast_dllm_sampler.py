"""Fast-dLLM discrete diffusion sampler with Branch Folding support.

This module provides a reference discrete-diffusion sampler suitable for
Fast-dLLM-style models.  It iteratively resamples low-confidence tokens based
on model predictions.  As with the other samplers in this package, this is an
illustrative implementation and should be validated against the official
Fast-dLLM recipe for production work.
"""

from __future__ import annotations

from typing import Any

import torch

from actfold.core.model_wrapper import FoldedModel
from actfold.models.diffusion_sampler import DiffusionSampler
from actfold.utils.logger import get_logger

logger = get_logger("models.fast_dllm_sampler")


class FastDLLMSampler(DiffusionSampler):
    """Discrete diffusion sampler for Fast-dLLM-style models.

    Args:
        model: Diffusion model.
        num_steps: Number of diffusion timesteps.
        num_tokens: Number of tokens to generate.
    """

    def initialize(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Keep the prompt and fill generated positions with random tokens."""
        batch_size, prompt_len = prompt_ids.shape
        random_tokens = torch.randint(
            0,
            self.model.vocab_size,
            (batch_size, self.num_tokens),
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        return torch.cat([prompt_ids, random_tokens], dim=-1)

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        branch_id: Any,
        parent_branch_id: Any | None,
        folded_model: FoldedModel | None,
    ) -> torch.Tensor:
        """Replace low-confidence tokens with model predictions."""
        logits = self._forward(
            x_t,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            folded_model=folded_model,
            step_idx=t,
        )
        predictions = logits.argmax(dim=-1)
        confidence = logits.max(dim=-1).values

        # Determine which positions to resample.  Earlier steps resample more.
        frac = t / self.num_steps
        num_positions = max(1, int(frac * x_t.shape[1]))
        _, lowest_conf_indices = confidence.view(-1).topk(num_positions, largest=False)

        x_next = x_t.clone()
        flat_pred = predictions.view(-1)
        x_next.view(-1)[lowest_conf_indices] = flat_pred[lowest_conf_indices]
        return x_next
