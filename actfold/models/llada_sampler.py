"""LLaDA masked diffusion sampler with Branch Folding support.

This sampler follows the official LLaDA / MDLM recipe:

- Build a right-padded canvas: prompt left-aligned, generation region filled
  with ``[MASK]``.
- Split the generation region into blocks (default ``block_size`` equal to
  ``num_tokens`` for a single block, matching the original reference).
- Within each block, use a masking schedule to decide how many tokens to
  reveal per step.
- At each step, predict all positions, then commit the highest-confidence
  predictions among currently-masked positions using ``low_confidence`` or
  ``random`` remasking.
- Support classifier-free guidance, top-p/top-k sampling, and Gumbel-Max
  noise for stochastic decoding.

References:
- LLaDA: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
- MDLM reference: https://github.com/ZHZisZZ/dllm
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from actfold.core.model_wrapper import FoldedModel
from actfold.models.base import DiffusionLLM
from actfold.models.diffusion_sampler import DiffusionSampler, SamplerConfig, SamplerOutput
from actfold.models.sampling_utils import (
    MaskingScheduler,
    add_gumbel_noise,
    build_right_padded_canvas,
    get_num_transfer_tokens,
    right_shift_logits,
)


@dataclass
class LLaDASamplerConfig(SamplerConfig):
    """Configuration for the LLaDA masked diffusion sampler."""

    block_size: int | None = None
    remasking: str = "low_confidence"  # "low_confidence" | "random"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] = field(default_factory=list)
    suppress_tokens: list[int] = field(default_factory=list)
    begin_suppress_tokens: list[int] = field(default_factory=list)
    right_shift_logits: bool = False
    scheduler: MaskingScheduler | None = None


class LLaDASampler(DiffusionSampler):
    """Masked diffusion sampler for LLaDA-style models.

    Args:
        model: Diffusion model wrapper.
        config: Sampler configuration. If omitted, sensible defaults are used.
    """

    config: LLaDASamplerConfig

    def __init__(
        self,
        model: DiffusionLLM,
        config: LLaDASamplerConfig | None = None,
    ) -> None:
        super().__init__(model, config or LLaDASamplerConfig())
        if self.config.remasking not in {"low_confidence", "random"}:
            raise ValueError(f"Unsupported remasking: {self.config.remasking}")

    def initialize(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Keep the prompt and mask the positions to be generated."""
        mask_token_id, eos_token_id, _ = self._get_special_token_ids()
        B, prompt_len = prompt_ids.shape
        max_new_tokens = self.config.num_tokens
        config_block = self.config.block_size
        block_size = config_block if config_block is not None else max_new_tokens
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")

        # Build right-padded canvas.
        inputs = [prompt_ids[i] for i in range(B)]
        x, _, _, max_length = build_right_padded_canvas(
            inputs, max_new_tokens, eos_token_id, mask_token_id
        )
        self._max_length = max_length
        self._block_size = block_size
        self._mask_token_id = mask_token_id
        self._eos_token_id = eos_token_id
        return x

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        branch_id: Any,
        parent_branch_id: Any | None,
        folded_model: FoldedModel | None,
    ) -> torch.Tensor:
        """Not used directly; sampling is block-wise inside :meth:`sample`."""
        return x_t

    def sample(
        self,
        prompt_ids: torch.Tensor,
        folded_model: FoldedModel | None = None,
    ) -> SamplerOutput:
        """Run block-wise masked diffusion sampling.

        This overrides the base loop to implement the official LLaDA block-wise
        schedule and remasking strategy.
        """
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        x = self.initialize(prompt_ids)
        B, T = x.shape
        max_new_tokens = self.config.num_tokens
        block_size = self._block_size
        mask_token_id = self._mask_token_id
        eos_token_id = self._eos_token_id

        # Pre-compute per-sample prompt lengths and attention mask.
        prompt_lens = []
        for i in range(B):
            non_eos = (x[i] != eos_token_id).nonzero(as_tuple=False)
            if non_eos.numel() == 0:
                prompt_lens.append(0)
            else:
                # First masked position after prompt.
                mask_positions = (x[i] == mask_token_id).nonzero(as_tuple=False)
                if mask_positions.numel() == 0:
                    prompt_lens.append(T)
                else:
                    prompt_lens.append(int(mask_positions[0].item()))

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=x.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that are given at the start (non-mask, non-EOS, valid).
        unmasked_index = (x != mask_token_id) & attention_mask.bool()
        if self.config.cfg_keep_tokens:
            keep = torch.isin(
                x,
                torch.tensor(self.config.cfg_keep_tokens, device=x.device),
            )
            unmasked_index = unmasked_index & ~keep

        num_blocks = max(1, math.ceil(max_new_tokens / block_size))
        steps_per_block = max(1, math.ceil(self.config.num_steps / num_blocks))
        history: list[torch.Tensor] = [x.clone()] if self.config.return_history else []

        for block_idx in range(num_blocks):
            # Determine which positions in this block are still masked.
            block_mask_index = torch.zeros((B, block_size), dtype=torch.bool, device=x.device)
            for j in range(B):
                start = prompt_lens[j] + block_idx * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = x[j, start:end] == mask_token_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.config.scheduler,
                stochastic=self.config.stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            for _ in range(effective_steps):
                mask_index = x == mask_token_id

                logits = self._forward_with_cfg(
                    x=x,
                    attention_mask=attention_mask,
                    unmasked_index=unmasked_index,
                    folded_model=folded_model,
                    step_idx=block_idx,
                )

                if self.config.suppress_tokens:
                    for token_id in self.config.suppress_tokens:
                        logits[:, :, token_id] = float("-inf")

                if self.config.right_shift_logits:
                    logits = right_shift_logits(logits)

                if self.config.begin_suppress_tokens:
                    for token_id in self.config.begin_suppress_tokens:
                        logits[:, :, token_id] = float("-inf")

                # Greedy prediction with optional Gumbel noise.
                logits_noisy = add_gumbel_noise(logits, self.config.temperature)
                x0 = torch.argmax(logits_noisy, dim=-1)

                # Confidence for choosing which masks to commit.
                if self.config.remasking == "low_confidence":
                    probs = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                else:  # random
                    x0_p = torch.rand(x0.shape, device=x0.device)

                # Restrict selection window to the current block.
                for j in range(B):
                    block_end = prompt_lens[j] + (block_idx + 1) * block_size
                    x0_p[j, block_end:] = float("-inf")

                # Only allow updates at currently masked positions.
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, float("-inf"))

                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, _].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                x[transfer_index] = x0[transfer_index]
                if self.config.return_history:
                    history.append(x.clone())

        decoded = self.decode_final(x)
        return SamplerOutput(sequences=decoded, history=history)

    def _forward_with_cfg(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        unmasked_index: torch.Tensor,
        folded_model: FoldedModel | None,
        step_idx: int,
    ) -> torch.Tensor:
        """Forward pass with optional classifier-free guidance."""
        cfg_scale = self.config.cfg_scale
        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[unmasked_index] = self._mask_token_id
            x_ = torch.cat([x, un_x], dim=0)
            am = torch.cat([attention_mask, attention_mask], dim=0)
            logits = self._forward(
                x_,
                branch_id=f"diffusion_cfg_{step_idx}",
                parent_branch_id=None,
                folded_model=folded_model,
                step_idx=step_idx,
                attention_mask=am,
            )
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = uncond_logits + (cfg_scale + 1.0) * (cond_logits - uncond_logits)
        else:
            logits = self._forward(
                x,
                branch_id=f"diffusion_{step_idx}",
                parent_branch_id=None,
                folded_model=folded_model,
                step_idx=step_idx,
                attention_mask=attention_mask,
            )
        return logits

    def decode_final(self, x: torch.Tensor) -> torch.Tensor:
        """Return token IDs unchanged (LLaDA already produces discrete tokens)."""
        return x
