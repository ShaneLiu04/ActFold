"""Dream-style continuous diffusion sampler with Branch Folding support.

This is a simplified reference sampler for continuous-diffusion text models
inspired by Dream.  It treats token embeddings as continuous vectors, adds
noise according to a schedule, and denoises them step by step.  Production use
requires alignment with the official Dream sampling recipe.
"""

from __future__ import annotations

from typing import Any

import torch

from actfold.core.model_wrapper import FoldedModel
from actfold.models.base import DiffusionLLM
from actfold.models.diffusion_sampler import DiffusionSampler
from actfold.utils.logger import get_logger

logger = get_logger("models.dream_sampler")


class DreamSampler(DiffusionSampler):
    """Continuous diffusion sampler for Dream-style models.

    Args:
        model: Diffusion model.
        num_steps: Number of diffusion timesteps.
        num_tokens: Number of tokens to generate.
        embedding_dim: Dimensionality of token embeddings.
    """

    def __init__(
        self,
        model: DiffusionLLM,
        num_steps: int,
        num_tokens: int,
        embedding_dim: int | None = None,
    ) -> None:
        super().__init__(model, num_steps, num_tokens)
        self.embedding_dim = embedding_dim or model.hidden_dim

    def initialize(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Start from the prompt embeddings and random noise for new tokens."""
        batch_size, prompt_len = prompt_ids.shape
        prompt_emb = self.model.embed(prompt_ids)
        noise = torch.randn(
            batch_size,
            self.num_tokens,
            self.embedding_dim,
            dtype=prompt_emb.dtype,
            device=prompt_emb.device,
        )
        return torch.cat([prompt_emb, noise], dim=1)

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        branch_id: Any,
        parent_branch_id: Any | None,
        folded_model: FoldedModel | None,
    ) -> torch.Tensor:
        """Predict clean embeddings and interpolate toward them."""
        # The model receives continuous embeddings and should predict clean embeddings.
        predicted_clean = self._forward(
            x_t,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            folded_model=folded_model,
            step_idx=t,
        )

        # Simple linear noise schedule (alpha_t = t / num_steps).
        alpha = t / self.num_steps
        x_next = alpha * x_t + (1.0 - alpha) * predicted_clean
        return x_next

    def decode_final(self, x: torch.Tensor) -> torch.Tensor:
        """Map continuous embeddings back to discrete token IDs.

        This simplified decoder projects embeddings to logits and argmaxes.  A
        real Dream implementation would use the model's native decoder.
        """
        # Use the model's output head if available, otherwise a simple projection.
        logits: torch.Tensor
        if hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(x)
        else:
            projection = torch.nn.Linear(self.embedding_dim, self.model.vocab_size, bias=False).to(
                dtype=x.dtype, device=x.device
            )
            logits = projection(x)
        tokens: torch.Tensor = logits.argmax(dim=-1)
        return tokens
