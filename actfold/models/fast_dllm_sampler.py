"""Fast-dLLM discrete diffusion sampler with Branch Folding support.

This sampler follows the Fast-dLLM v2 recipe:

- Generation is performed block-by-block.  Each block is initialized to the
  ``[MASK]`` token.
- Within a block, the model predicts all positions.  Positions are unmasked
  in small sub-blocks when their predicted probability exceeds a confidence
  ``threshold``.  The highest-probability position in each small block is
  always unmasked.
- When a block is fully unmasked, an autoregressive step generates the next
  block's first token using KV-cache friendly block processing.
- Support top-p / temperature sampling and early stopping on a stop token.

Reference:
- https://github.com/NVlabs/Fast-dLLM/blob/main/v2/generation_functions.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from actfold.core.model_wrapper import FoldedModel
from actfold.models.base import DiffusionLLM
from actfold.models.diffusion_sampler import DiffusionSampler, SamplerConfig, SamplerOutput
from actfold.models.sampling_utils import right_shift_logits, sample_tokens


@dataclass
class FastDLLMSamplerConfig(SamplerConfig):
    """Configuration for the Fast-dLLM discrete diffusion sampler."""

    block_size: int = 32
    small_block_size: int = 32
    threshold: float = 0.95
    stop_token_id: int | None = None
    pad_token_id: int | None = None
    mask_token_id: int | None = None


class FastDLLMSampler(DiffusionSampler):
    """Discrete diffusion sampler for Fast-dLLM-style models.

    Args:
        model: Diffusion model wrapper.
        config: Sampler configuration. If omitted, sensible defaults are used.
    """

    config: FastDLLMSamplerConfig

    def __init__(
        self,
        model: DiffusionLLM,
        config: FastDLLMSamplerConfig | None = None,
    ) -> None:
        super().__init__(model, config or FastDLLMSamplerConfig())
        if self.config.block_size < 1 or self.config.small_block_size < 1:
            raise ValueError("block_size and small_block_size must be >= 1")
        if self.config.block_size % self.config.small_block_size != 0:
            raise ValueError("block_size must be divisible by small_block_size")

    def initialize(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Return prompt_ids; block-wise state is built inside :meth:`sample`."""
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("FastDLLMSampler requires a tokenizer to obtain special token ids.")

        cfg = self.config
        mask_token_id = cfg.mask_token_id
        stop_token_id = cfg.stop_token_id
        pad_token_id = cfg.pad_token_id

        if mask_token_id is None:
            mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if stop_token_id is None:
            stop_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None and stop_token_id is not None:
                pad_token_id = stop_token_id

        if mask_token_id is None:
            raise RuntimeError(
                "FastDLLMSampler requires a mask_token_id. Set it in the config or tokenizer."
            )
        if stop_token_id is None:
            raise RuntimeError(
                "FastDLLMSampler requires a stop_token_id. Set it in the config or tokenizer."
            )

        self._mask_token_id = int(mask_token_id)
        self._stop_token_id = int(stop_token_id)
        self._pad_token_id = int(pad_token_id) if pad_token_id is not None else int(stop_token_id)
        return prompt_ids.clone()

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        branch_id: Any,
        parent_branch_id: Any | None,
        folded_model: FoldedModel | None,
    ) -> torch.Tensor:
        """Not used directly; generation is block-wise inside :meth:`sample`."""
        return x_t

    def sample(
        self,
        prompt_ids: torch.Tensor,
        folded_model: FoldedModel | None = None,
    ) -> SamplerOutput:
        """Run Fast-dLLM block-wise masked diffusion sampling."""
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        x = self.initialize(prompt_ids)
        B = x.shape[0]
        device = x.device
        block_size = self.config.block_size
        small_block_size = self.config.small_block_size
        num_small_blocks = block_size // small_block_size
        max_new_tokens = self.config.num_tokens
        mask_id = self._mask_token_id
        stop_id = self._stop_token_id
        pad_id = self._pad_token_id

        prompt_len = x.shape[1]
        total_len = prompt_len + max_new_tokens
        num_blocks = max(1, (max_new_tokens + block_size - 1) // block_size)

        # Pad prompt to a multiple of block_size so that subsequent blocks are aligned.
        first_block_padding = (
            (block_size - prompt_len % block_size) if prompt_len % block_size != 0 else 0
        )
        if first_block_padding > 0:
            pad = torch.full((B, first_block_padding), pad_id, dtype=torch.long, device=device)
            x = torch.cat([x, pad], dim=1)
        else:
            first_block_padding = 0

        # Append the first block of masks.
        x = torch.cat(
            [x, torch.full((B, block_size), mask_id, dtype=torch.long, device=device)],
            dim=1,
        )

        history: list[torch.Tensor] = [x.clone()] if self.config.return_history else []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for block_idx in range(num_blocks):
            if finished.all():
                break

            # Denoise the last block.
            for small_block_idx in range(num_small_blocks):
                start = -block_size + small_block_idx * small_block_size
                end = (
                    None
                    if small_block_idx == num_small_blocks - 1
                    else -block_size + (small_block_idx + 1) * small_block_size
                )

                # Keep denoising this small block until all positions are unmasked.
                max_iters = small_block_size * 2
                for _ in range(max_iters):
                    block_view = x[:, -block_size:]
                    mask_idx = block_view == mask_id
                    if mask_idx[:, start:end].sum() == 0:
                        break

                    logits = self._forward(
                        x,
                        branch_id=f"fastdllm_b{block_idx}_s{small_block_idx}",
                        parent_branch_id=None,
                        folded_model=folded_model,
                        step_idx=block_idx,
                    )
                    logits = right_shift_logits(logits)
                    logits = logits[:, -block_size:]
                    logits = logits[:, start:end]

                    confidence, x1 = sample_tokens(
                        logits,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                    )
                    probs = F.softmax(logits, dim=-1)
                    x1_p = torch.gather(probs, dim=-1, index=x1.unsqueeze(-1)).squeeze(-1)

                    small_mask = mask_idx[:, start:end]
                    x1_p = torch.where(small_mask, x1_p, torch.full_like(x1_p, float("-inf")))

                    unmask = x1_p > self.config.threshold
                    max_idx = x1_p.argmax(dim=-1)
                    unmask[torch.arange(B), max_idx] = True
                    unmask = unmask & small_mask

                    x[:, start:end][unmask] = x1[unmask]

                    finished = finished | ((x1 == stop_id) & unmask).any(dim=1)

                    if self.config.return_history:
                        history.append(x.clone())

            # After the block is decoded, generate the next token autoregressively
            # and append a fresh mask block if more tokens remain.
            remaining = total_len + first_block_padding - x.shape[1]
            if remaining <= 0:
                continue

            logits = self._forward(
                x,
                branch_id=f"fastdllm_ar_{block_idx}",
                parent_branch_id=None,
                folded_model=folded_model,
                step_idx=block_idx,
            )
            logits = right_shift_logits(logits)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            next_token[finished] = pad_id
            x = torch.cat([x, next_token], dim=1)

            if remaining > 1:
                fresh_masks = torch.full(
                    (B, min(block_size - 1, remaining - 1)),
                    mask_id,
                    dtype=torch.long,
                    device=device,
                )
                x = torch.cat([x, fresh_masks], dim=1)

            if self.config.return_history:
                history.append(x.clone())

        # Trim padding and stop tokens for finished sequences.
        decoded = self.decode_final(x)
        return SamplerOutput(sequences=decoded, history=history)

    def decode_final(self, x: torch.Tensor) -> torch.Tensor:
        """Trim leading pad/mask positions and trailing stop tokens.

        Fast-dLLM uses left-padded canvas internally.  This decoder removes the
        initial padding and any trailing stop-token suffix.
        """
        B, T = x.shape
        pad_id = self._pad_token_id
        stop_id = self._stop_token_id
        mask_id = self._mask_token_id

        # Find the first non-pad/non-mask position in each row.
        valid = (x != pad_id) & (x != mask_id)
        out_rows: list[torch.Tensor] = []
        for i in range(B):
            pos = valid[i].nonzero(as_tuple=False)
            if pos.numel() == 0:
                out_rows.append(torch.tensor([stop_id], dtype=torch.long, device=x.device))
                continue
            start = int(pos[0].item())
            row = x[i, start:].clone()
            # Trim trailing stop tokens (but keep one).
            stop_positions = (row == stop_id).nonzero(as_tuple=False)
            if stop_positions.numel() > 0:
                end = int(stop_positions[0].item()) + 1
                row = row[:end]
            out_rows.append(row)

        max_len = max(r.shape[0] for r in out_rows)
        out = torch.full((B, max_len), pad_id, dtype=torch.long, device=x.device)
        for i, r in enumerate(out_rows):
            out[i, : r.shape[0]] = r
        return out
