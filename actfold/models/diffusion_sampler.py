"""Diffusion-native sampling framework for ActFold.

This module provides a base class for diffusion samplers that can be
integrated with :class:`~actfold.core.model_wrapper.FoldedModel`.  The design
 treats each diffusion timestep as a branch-extension step: the state at
timestep ``t`` is the parent branch and the predicted state at ``t-1`` is a
child branch whose activations can be folded against the parent.

Concrete samplers for specific model families (LLaDA, Dream, Fast-dLLM)
should subclass :class:`DiffusionSampler` and implement ``initialize`` and
``denoise_step``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import torch

from actfold.core.model_wrapper import FoldedModel
from actfold.models.base import DiffusionLLM


@dataclass
class SamplerConfig:
    """Base configuration shared by all diffusion samplers."""

    num_steps: int = 128
    num_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    return_history: bool = False
    seed: int | None = None


@dataclass
class SamplerOutput:
    """Output of a diffusion sampler."""

    sequences: torch.Tensor
    history: list[torch.Tensor] = field(default_factory=list)


class DiffusionSampler(ABC):
    """Abstract sampler for diffusion LLMs with optional Branch Folding.

    Args:
        model: The diffusion model to sample from.
        config: Sampler configuration.
    """

    def __init__(
        self,
        model: DiffusionLLM,
        config: SamplerConfig,
    ) -> None:
        self.model = model
        self.config = config
        self._folding_counter: int = 0
        self._folding_parent: str | None = None

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
    ) -> SamplerOutput:
        """Run the full diffusion sampling loop.

        Args:
            prompt_ids: Encoded prompt ``[batch, prompt_len]``.
            folded_model: Optional folded model for cross-timestep activation reuse.

        Returns:
            Final token sequence and optional intermediate history.
        """
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        x: torch.Tensor = self.initialize(prompt_ids)
        parent_id: Any | None = None
        history: list[torch.Tensor] = [x.clone()] if self.config.return_history else []

        for t in range(self.config.num_steps, 0, -1):
            branch_id = f"diffusion_t{t}"
            x = self.denoise_step(
                x,
                t,
                branch_id=branch_id,
                parent_branch_id=parent_id,
                folded_model=folded_model,
            )
            parent_id = branch_id
            if self.config.return_history:
                history.append(x.clone())

        decoded = self.decode_final(x)
        return SamplerOutput(sequences=decoded, history=history)

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
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Helper to run ``model.forward`` with optional folding context.

        For token inputs the model is called with ``input_ids``; for continuous
        embeddings it is called with ``inputs_embeds``.  The branch identifiers
        are forwarded through the folded model or as explicit kwargs.
        """
        if folded_model is not None:
            out: torch.Tensor = folded_model(
                x,
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                step_idx=step_idx,
                attention_mask=attention_mask,
            )
            return out

        kwargs: dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids

        logits: torch.Tensor = self.model.forward(
            x,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            step_idx=step_idx,
            **kwargs,
        )
        return logits

    @staticmethod
    def _resolve_mask_token_id(model: Any, tokenizer: Any) -> int | None:
        """Resolve the mask token id across model-specific naming conventions.

        Checks, in order: ``tokenizer.mask_token_id``, the model config's
        ``mask_token_id``, and the common token spellings used by real
        checkpoints (LLaDA: ``<|mdm_mask|>``, Fast-dLLM: ``|<MASK>|``,
        Dream: ``<|mask|>``).
        """
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is not None:
            return int(mask_token_id)
        raw_model = getattr(model, "model", None)
        config = getattr(raw_model, "config", None)
        config_mask = getattr(config, "mask_token_id", None)
        if config_mask is not None:
            return int(config_mask)
        unk = getattr(tokenizer, "unk_token_id", None)
        for token in ("<|mdm_mask|>", "|<MASK>|", "<|mask|>", "<mask>", "[MASK]"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != unk:
                return int(token_id)
        return None

    def _next_folding_branch(self, tag: str) -> tuple[str, str | None]:
        """Return ``(branch_id, parent_branch_id)`` for cross-step folding.

        Each forward pass in a sampling loop becomes a branch whose parent is
        the previous forward pass.  Chaining the parent lets
        :class:`~actfold.core.folded_transformer.FoldedTransformerLayer` reuse
        activations across denoising iterations, which is the core ActFold
        mechanism applied to diffusion sampling.
        """
        counter = getattr(self, "_folding_counter", 0) + 1
        self._folding_counter = counter
        branch_id = f"{tag}_{counter}"
        parent_id = getattr(self, "_folding_parent", None)
        self._folding_parent = branch_id
        return branch_id, parent_id

    def _reset_folding_chain(self) -> None:
        """Reset the cross-step folding parent chain at the start of sampling."""
        self._folding_counter = 0
        self._folding_parent = None

    def _get_special_token_ids(self) -> tuple[int, int, int | None]:
        """Return ``(mask_token_id, eos_token_id, bos_token_id)`` from the tokenizer.

        Raises:
            RuntimeError: If the model has no tokenizer or required token ids.
        """
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a tokenizer to obtain mask/eos/bos ids."
            )
        mask_token_id = self._resolve_mask_token_id(self.model, tokenizer)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if mask_token_id is None:
            raise RuntimeError(
                "Tokenizer does not define a mask_token_id; required for masked diffusion samplers."
            )
        if eos_token_id is None:
            raise RuntimeError("Tokenizer does not define an eos_token_id.")
        return (
            mask_token_id,
            cast(int, eos_token_id),
            cast(int | None, bos_token_id),
        )

    def _maybe_get_lm_head(self) -> torch.nn.Module | None:
        """Return the model's output head if one is available."""
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            return cast(torch.nn.Module, self.model.lm_head)
        return None
