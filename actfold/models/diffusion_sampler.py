"""Diffusion-native sampling framework for ActFold.

This module provides an abstract base class for diffusion samplers that can be
integrated with :class:`~actfold.core.model_wrapper.FoldedModel`.  The design
 treats each diffusion timestep as a branch-extension step: the state at
 timestep ``t`` is the parent branch and the predicted state at ``t-1`` is a
 child branch whose activations can be folded against the parent.

Concrete samplers for specific model families (LLaDA, Dream, Fast-dLLM) should
 subclass :class:`DiffusionSampler` and implement ``initialize`` and
 ``denoise_step``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from actfold.core.model_wrapper import FoldedModel
from actfold.models.base import DiffusionLLM


class DiffusionSampler(ABC):
    """Abstract sampler for diffusion LLMs with optional Branch Folding.

    Args:
        model: The diffusion model to sample from.
        num_steps: Number of diffusion timesteps.
        num_tokens: Number of tokens to generate.
    """

    def __init__(
        self,
        model: DiffusionLLM,
        num_steps: int,
        num_tokens: int,
    ) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")

        self.model = model
        self.num_steps = num_steps
        self.num_tokens = num_tokens

    @abstractmethod
    def initialize(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Create the initial noisy/masked state ``x_T`` from ``prompt_ids``."""
        ...

    @abstractmethod
    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        branch_id: Any,
        parent_branch_id: Any | None,
        folded_model: FoldedModel | None,
    ) -> torch.Tensor:
        """Run a single denoising step and return ``x_{t-1}``.

        Args:
            x_t: Current state ``[batch, seq_len]``.
            t: Current timestep (decreases from ``num_steps`` to ``1``).
            branch_id: Identifier for the child branch produced by this step.
            parent_branch_id: Identifier for the parent branch (``x_t``).
            folded_model: Optional folded model for activation reuse.

        Returns:
            Denoised state ``[batch, seq_len]``.
        """
        ...

    def sample(
        self,
        prompt_ids: torch.Tensor,
        folded_model: Optional[FoldedModel] = None,
    ) -> torch.Tensor:
        """Run the full diffusion sampling loop.

        Args:
            prompt_ids: Encoded prompt ``[batch, prompt_len]``.
            folded_model: Optional folded model for cross-timestep activation reuse.

        Returns:
            Final token sequence ``[batch, prompt_len + num_tokens]``.
        """
        x: torch.Tensor = self.initialize(prompt_ids)
        parent_id: Any | None = None

        for t in range(self.num_steps, 0, -1):
            branch_id = f"diffusion_t{t}"
            x = self.denoise_step(
                x,
                t,
                branch_id=branch_id,
                parent_branch_id=parent_id,
                folded_model=folded_model,
            )
            parent_id = branch_id

        return self.decode_final(x)

    def decode_final(self, x: torch.Tensor) -> torch.Tensor:
        """Convert the final diffusion state into discrete token IDs.

        The default implementation simply returns ``x`` assuming it already
        contains token IDs.  Subclasses may override this for models that
        produce logits or continuous representations.
        """
        return x

    def _forward(
        self,
        x: torch.Tensor,
        branch_id: Any,
        parent_branch_id: Any | None,
        folded_model: FoldedModel | None,
        step_idx: int,
    ) -> torch.Tensor:
        """Helper to run ``model.forward`` with optional folding context."""
        if folded_model is not None:
            out: torch.Tensor = folded_model(
                x,
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                step_idx=step_idx,
            )
            return out
        logits: torch.Tensor = self.model.forward(
            x,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            step_idx=step_idx,
        )
        return logits
