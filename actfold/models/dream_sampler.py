"""Dream-style masked diffusion sampler with Branch Folding support.

This sampler follows the official Dream recipe (e.g. Dream-v0-7B):

- Build a left-padded canvas: prompts are right-aligned and the generation
  region on the left is filled with ``[MASK]``.
- Compute 1-D position ids that ignore left-side padding.
- Iteratively denoise the whole canvas for a fixed number of steps.
- At each step, score only the currently masked positions with one of the
  official confidence rules (``maskgit_plus``, ``topk_margin``, ``entropy``),
  then commit the scheduled number of highest-confidence positions.
- Support classifier-free guidance, temperature/top-p/top-k token sampling,
  and soft confidence-based selection via ``alg_temp``.

Reference:
- https://huggingface.co/Dream-org/Dream-v0-Base-7B/blob/main/generation_utils.py
- https://github.com/ZHZisZZ/dllm/tree/main/dllm/pipelines/dream
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from actfold.core.model_wrapper import FoldedModel
from actfold.models.base import DiffusionLLM
from actfold.models.diffusion_sampler import DiffusionSampler, SamplerConfig, SamplerOutput
from actfold.models.sampling_utils import (
    MaskingScheduler,
    build_left_padded_canvas,
    compute_position_ids,
    get_num_transfer_tokens,
    right_shift_logits,
    sample_tokens,
)


@dataclass
class DreamSamplerConfig(SamplerConfig):
    """Configuration for the Dream masked diffusion sampler."""

    eps: float = 1e-3
    alg: str = "maskgit_plus"  # "maskgit_plus" | "topk_margin" | "entropy"
    alg_temp: float = 0.0
    stochastic_transfer: bool = False
    right_shift_logits: bool = True
    cfg_scale: float = 0.0
    scheduler: MaskingScheduler | None = None


class DreamSampler(DiffusionSampler):
    """Masked diffusion sampler for Dream-style models.

    Args:
        model: Diffusion model wrapper.
        config: Sampler configuration. If omitted, sensible defaults are used.
    """

    config: DreamSamplerConfig

    def __init__(
        self,
        model: DiffusionLLM,
        config: DreamSamplerConfig | None = None,
    ) -> None:
        super().__init__(model, config or DreamSamplerConfig())
        if self.config.alg not in {"maskgit_plus", "topk_margin", "entropy"}:
            raise ValueError(f"Unsupported Dream alg: {self.config.alg}")

    def initialize(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Build a left-padded canvas with prompts right-aligned."""
        mask_token_id, eos_token_id, _ = self._get_special_token_ids()
        inputs = [prompt_ids[i] for i in range(prompt_ids.shape[0])]
        x, attention_mask, seq_lens = build_left_padded_canvas(
            inputs, self.config.num_tokens, eos_token_id, mask_token_id
        )
        self._attention_mask = attention_mask
        self._seq_lens = seq_lens
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
        """Not used directly; sampling is canvas-wide inside :meth:`sample`."""
        return x_t

    def sample(
        self,
        prompt_ids: torch.Tensor,
        folded_model: FoldedModel | None = None,
    ) -> SamplerOutput:
        """Run Dream-style iterative masked diffusion sampling."""
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        x = self.initialize(prompt_ids)
        B, T = x.shape
        mask_token_id = self._mask_token_id
        attention_mask = self._attention_mask

        pos_id: torch.Tensor | None = None
        if torch.any(attention_mask == 0):
            pos_id = compute_position_ids(attention_mask)

        mask_index = x == mask_token_id
        num_transfer_tokens_list = get_num_transfer_tokens(
            mask_index=mask_index,
            steps=self.config.num_steps,
            scheduler=self.config.scheduler,
            stochastic=self.config.stochastic_transfer,
        )
        effective_steps = num_transfer_tokens_list.size(1)

        # For CFG, only the original prompt positions are masked in the
        # unconditional branch; step-wise revealed tokens are not masked again.
        prompt_index = attention_mask.bool() & (
            torch.arange(T, device=x.device).unsqueeze(0) < T - self.config.num_tokens
        )

        history: list[torch.Tensor] = [x.clone()] if self.config.return_history else []

        for step in range(effective_steps):
            mask_index = x == mask_token_id

            logits = self._forward_with_cfg(
                x=x,
                attention_mask=attention_mask,
                position_ids=pos_id,
                prompt_index=prompt_index,
                folded_model=folded_model,
                step_idx=step,
            )

            if self.config.right_shift_logits:
                logits = right_shift_logits(logits)

            mask_logits = logits[mask_index]
            if mask_logits.numel() == 0:
                break

            confidence, x0 = sample_tokens(
                mask_logits,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                margin_confidence=self.config.alg == "topk_margin",
                neg_entropy=self.config.alg == "entropy",
            )

            full_confidence = torch.full_like(x, float("-inf"), device=x.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence

            for j in range(B):
                k = int(num_transfer_tokens_list[j, step].item())
                if k > 0:
                    if self.config.alg_temp is None or self.config.alg_temp == 0.0:
                        _, transfer_index = torch.topk(full_confidence[j], k=k)
                    else:
                        fc = full_confidence[j] / self.config.alg_temp
                        fc = F.softmax(fc, dim=-1)
                        transfer_index = torch.multinomial(fc, num_samples=k)

                    x_ = torch.full_like(x, mask_token_id, device=x.device)
                    x_[mask_index] = x0
                    x[j, transfer_index] = x_[j, transfer_index]

            if self.config.return_history:
                history.append(x.clone())

        decoded = self.decode_final(x)
        return SamplerOutput(sequences=decoded, history=history)

    def _forward_with_cfg(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        prompt_index: torch.Tensor,
        folded_model: FoldedModel | None,
        step_idx: int,
    ) -> torch.Tensor:
        """Forward pass with optional classifier-free guidance."""
        cfg_scale = self.config.cfg_scale
        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[prompt_index] = self._mask_token_id
            x_ = torch.cat([x, un_x], dim=0)
            am = torch.cat([attention_mask, attention_mask], dim=0)
            pos = (
                torch.cat([position_ids, position_ids], dim=0) if position_ids is not None else None
            )
            logits = self._forward(
                x_,
                branch_id=f"dream_cfg_{step_idx}",
                parent_branch_id=None,
                folded_model=folded_model,
                step_idx=step_idx,
                attention_mask=am,
                position_ids=pos,
            )
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = uncond_logits + (cfg_scale + 1.0) * (cond_logits - uncond_logits)
        else:
            logits = self._forward(
                x,
                branch_id=f"dream_{step_idx}",
                parent_branch_id=None,
                folded_model=folded_model,
                step_idx=step_idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return logits

    def decode_final(self, x: torch.Tensor) -> torch.Tensor:
        """Return token IDs unchanged (Dream sampler already produces discrete tokens)."""
        return x
