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
from actfold.models.speculative_core import SpecCtx, SpeculativeCore


@dataclass
class LLaDASamplerConfig(SamplerConfig):
    """Configuration for the LLaDA masked diffusion sampler.

    Args:
        num_block_candidates: Number of speculative candidate continuations
            per block (AR023 FR2). ``0`` (the default) keeps the exact
            reference remasking recipe; ``N > 0`` replaces the block's
            ``steps_per_block`` remasking iterations with one draft serial
            forward plus one ``[N, T]`` batched-branch verification call
            (requires a ``folded_model`` that supports ``branch_ids`` and
            batch 1).
        candidate_top_k: Draft-set size per position for the speculative
            candidates (top-k tokens of the draft logits). Consumed only
            when ``num_block_candidates > 0``.
        candidate_include_greedy: AR022 greedy anchoring, shared face.
            ``True`` pins candidate row 0 to the greedy hypothesis — the
            still-masked block positions are filled with the draft's argmax
            deterministically, with zero device readback and zero generator
            consumption; random candidate rows become ``j=1..N-1`` (the
            frozen per-candidate seed formula keeps ``j`` as the row
            number). The greedy row competes under the unchanged min-prob
            rule and CAN LOSE. ``False`` (the default) is byte-identical to
            the speculative-off state; ``True`` requires
            ``num_block_candidates > 0``.
    """

    block_size: int | None = None
    remasking: str = "low_confidence"  # "low_confidence" | "random"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] = field(default_factory=list)
    suppress_tokens: list[int] = field(default_factory=list)
    begin_suppress_tokens: list[int] = field(default_factory=list)
    right_shift_logits: bool = False
    scheduler: MaskingScheduler | None = None
    num_block_candidates: int = 0
    candidate_top_k: int = 10
    candidate_include_greedy: bool = False

    def __post_init__(self) -> None:
        if self.num_block_candidates < 0:
            raise ValueError(f"num_block_candidates must be >= 0, got {self.num_block_candidates}")
        if self.candidate_top_k < 1:
            raise ValueError(f"candidate_top_k must be >= 1, got {self.candidate_top_k}")
        if self.candidate_include_greedy and self.num_block_candidates == 0:
            raise ValueError(
                "candidate_include_greedy=True requires num_block_candidates"
                " > 0 (greedy anchoring is defined only on the speculative"
                f" path; got num_block_candidates={self.num_block_candidates})"
            )


class LLaDASampler(DiffusionSampler, SpeculativeCore):
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
        # AR023: instance-level greedy_win observation counters (AR022
        # precedent) — survive across ``sample`` calls, incremented only on
        # the greedy-anchored speculative path.
        self._greedy_wins = 0
        self._greedy_spec_calls = 0

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

    def _spec_ctx_for(self, x: torch.Tensor, prompt_len: int) -> SpecCtx:
        """Return the pinned LLaDA speculative fold context (AR023 §4.1.3).

        The context is keyed by ``(T, prompt_len)`` and object-pinned through
        the Core's LRU skeleton (cap 32). The ``[T, T]`` bool mask is the
        1-D reference mask ``arange(T) < valid_end`` tiled over rows (the
        existing ``attention_mask`` semantics in 2-D form — pad columns are
        excluded everywhere); ``valid_end = prompt_len + num_tokens`` comes
        from host metadata only, never a device readback. LLaDA carries no
        position payload (``position=None`` — no rotary cargo).
        """
        seq_len = int(x.shape[1])
        key = (seq_len, int(prompt_len))
        cached = self._ctx_cache_get(key)
        if cached is not None:
            return cached
        valid_end = min(int(prompt_len) + int(self.config.num_tokens), seq_len)
        reference_1d = torch.arange(seq_len, device=x.device) < valid_end
        mask = reference_1d.unsqueeze(0).repeat(seq_len, 1)
        ctx = SpecCtx(mask=mask, position=None)
        self._ctx_cache_put(key, ctx)
        return ctx

    def _build_unit_forced(
        self,
        x: torch.Tensor,
        region_start: int,
        region_end: int,
        masked_local: torch.Tensor,
    ) -> torch.Tensor:
        """Return the ``[T]`` bool forced tensor for one block unit.

        True over the whole block window ``[region_start, region_end)`` —
        the conservative superset of the actively committed positions
        (AR021 P5 reasoning: a freshly committed block vs. the last denoise
        iteration is exactly the partial-flip scenario, so the whole block
        window is always recomputed, never gate-certified stable against a
        stale parent). False outside the window.
        """
        forced = torch.zeros(int(x.shape[1]), dtype=torch.bool, device=x.device)
        forced[region_start:region_end] = True
        return forced

    def sample(
        self,
        prompt_ids: torch.Tensor,
        folded_model: FoldedModel | None = None,
    ) -> SamplerOutput:
        """Run block-wise masked diffusion sampling.

        This overrides the base loop to implement the official LLaDA block-wise
        schedule and remasking strategy. With ``num_block_candidates > 0``
        (AR023) each block instead becomes one speculative unit: a draft
        serial forward plus one ``[N, T]`` batched-branch verification call
        through the shared :class:`~actfold.models.speculative_core.SpeculativeCore`
        (default off keeps the reference recipe byte-identical).

        Raises:
            ValueError: If ``num_block_candidates > 0`` without a
                ``folded_model``, with a folded model that does not support
                the ``branch_ids`` kwarg, with a batch larger than 1 (the
                ``[T, T]`` fold-context mask cannot express per-row padding),
                or with ``cfg_scale > 0`` (the speculative path does not
                implement classifier-free guidance; never a silent drop).
        """
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        num_candidates = int(self.config.num_block_candidates)
        if num_candidates > 0:
            if folded_model is None:
                raise ValueError(
                    "num_block_candidates > 0 requires a folded_model for the "
                    "batched speculative verification path."
                )
            if not self._supports_branch_ids(folded_model):
                raise ValueError(
                    "num_block_candidates > 0 requires a folded_model that "
                    "supports the branch_ids batched-branch kwarg."
                )
            if prompt_ids.shape[0] > 1:
                raise ValueError(
                    "LLaDASampler speculative sampling supports batch 1 only; "
                    "the [T, T] fold-context mask cannot express per-row "
                    f"padding (got batch {prompt_ids.shape[0]})."
                )
            if self.config.cfg_scale > 0.0:
                raise ValueError(
                    "num_block_candidates > 0 requires cfg_scale == 0.0 (the "
                    "speculative draft/batched path does not implement "
                    "classifier-free guidance; refusing to silently drop it)."
                )

        self._reset_folding_chain()
        x = self.initialize(prompt_ids)
        B, T = x.shape
        max_new_tokens = self.config.num_tokens
        block_size = self._block_size
        mask_token_id = self._mask_token_id
        eos_token_id = self._eos_token_id

        # Pre-compute per-sample prompt lengths and the attention mask,
        # vectorized (F10): one host readback total instead of one per row.
        is_mask = x == mask_token_id
        any_mask = is_mask.any(dim=1)
        any_non_eos = (x != eos_token_id).any(dim=1)
        first_mask = torch.argmax(is_mask.to(torch.uint8), dim=1)
        full_len = torch.full_like(first_mask, T)
        zero = torch.zeros_like(first_mask)
        prompt_lens_t = torch.where(any_mask, first_mask, torch.where(any_non_eos, full_len, zero))
        prompt_lens = prompt_lens_t.tolist()
        valid_end = (prompt_lens_t + max_new_tokens).clamp(max=T)
        attention_mask = torch.arange(T, device=x.device).unsqueeze(0) < valid_end.unsqueeze(1)
        attention_mask = attention_mask.to(dtype=torch.long)

        # Tokens that are given at the start (non-mask, non-EOS, valid).
        unmasked_index = (x != mask_token_id) & attention_mask.bool()
        if self.config.cfg_keep_tokens:
            keep = torch.isin(
                x,
                torch.tensor(self.config.cfg_keep_tokens, device=x.device),
            )
            unmasked_index = unmasked_index & ~keep

        # Suppression id tensors, precomputed once (F10): a single
        # ``index_fill_`` per step replaces the per-token Python loop.
        suppress_ids = (
            torch.tensor(self.config.suppress_tokens, dtype=torch.long, device=x.device)
            if self.config.suppress_tokens
            else None
        )
        begin_suppress_ids = (
            torch.tensor(self.config.begin_suppress_tokens, dtype=torch.long, device=x.device)
            if self.config.begin_suppress_tokens
            else None
        )

        num_blocks = max(1, math.ceil(max_new_tokens / block_size))
        steps_per_block = max(1, math.ceil(self.config.num_steps / num_blocks))
        history: list[torch.Tensor] = [x.clone()] if self.config.return_history else []
        col_index = torch.arange(T, device=x.device).unsqueeze(0)
        # AR023: the speculative path's early-stop flags (the reference
        # recipe below never touches them — reference callers see no new
        # logic at all with ``num_block_candidates == 0``).
        finished: torch.Tensor | None = None
        if num_candidates > 0:
            finished = torch.zeros(B, dtype=torch.bool, device=x.device)

        for block_idx in range(num_blocks):
            if num_candidates > 0:
                # AR023 FR2: one speculative unit per block — one draft
                # serial forward plus one [N, T] batched verification call
                # commits the whole block (skipping the block's remaining
                # remasking iterations, AR021-isomorphic).
                assert finished is not None  # gated: created when num_candidates > 0
                if bool(finished.all()):
                    break
                region_start = prompt_lens[0] + block_idx * block_size
                region_end = min(region_start + block_size, prompt_lens[0] + max_new_tokens, T)
                if region_start >= region_end:
                    continue
                masked_local = (
                    (x[0, region_start:region_end] == mask_token_id)
                    .nonzero(as_tuple=False)
                    .reshape(-1)
                )
                if int(masked_local.numel()) == 0:
                    continue
                ctx = self._spec_ctx_for(x, prompt_len=prompt_lens[0])
                unit_forced = self._build_unit_forced(
                    x,
                    region_start=region_start,
                    region_end=region_end,
                    masked_local=masked_local,
                )
                draft_branch_id, draft_parent_id = self._next_folding_branch(
                    f"llada_spec_b{block_idx}"
                )
                draft_logits = self._forward(
                    x,
                    branch_id=draft_branch_id,
                    parent_branch_id=draft_parent_id,
                    folded_model=folded_model,
                    step_idx=block_idx,
                    attention_mask=ctx.mask,
                    position_embeddings=ctx.position,
                    forced_divergent=unit_forced,
                )
                if suppress_ids is not None:
                    draft_logits.index_fill_(-1, suppress_ids, float("-inf"))
                if self.config.right_shift_logits:
                    draft_logits = right_shift_logits(draft_logits)
                if begin_suppress_ids is not None:
                    draft_logits.index_fill_(-1, begin_suppress_ids, float("-inf"))
                region_logits = draft_logits[:, region_start:region_end, :]

                def _batched_forward(tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                    return self._forward(
                        tokens,
                        folded_model=folded_model,
                        step_idx=block_idx,
                        attention_mask=ctx.mask,
                        position_embeddings=ctx.position,
                        **kwargs,
                    )

                self._speculative_commit(
                    x=x,
                    finished=finished,
                    draft_logits_region=region_logits,
                    region_start=region_start,
                    masked_local=masked_local,
                    draft_branch_id=draft_branch_id,
                    ctx=ctx,
                    unit_forced=unit_forced,
                    folded_model=folded_model,
                    unit_idx=block_idx,
                    sub_idx=0,
                    history=history,
                    forward_fn=_batched_forward,
                    stop_token_id=eos_token_id,
                )
                continue

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
            # One host readback per block: the batched top-k width below.
            k_max = int(num_transfer_tokens.max().item())
            block_ends = prompt_lens_t + (block_idx + 1) * block_size

            for step in range(effective_steps):
                mask_index = x == mask_token_id
                if not bool(mask_index.any()):
                    # Early stop (F10): nothing left to reveal, skip the
                    # forward pass entirely.
                    break

                logits = self._forward_with_cfg(
                    x=x,
                    attention_mask=attention_mask,
                    unmasked_index=unmasked_index,
                    folded_model=folded_model,
                    step_idx=block_idx,
                )

                if suppress_ids is not None:
                    logits.index_fill_(-1, suppress_ids, float("-inf"))

                if self.config.right_shift_logits:
                    logits = right_shift_logits(logits)

                if begin_suppress_ids is not None:
                    logits.index_fill_(-1, begin_suppress_ids, float("-inf"))

                # Greedy prediction with optional Gumbel noise.
                logits_noisy = add_gumbel_noise(logits, self.config.temperature)
                x0 = torch.argmax(logits_noisy, dim=-1)

                # Confidence for choosing which masks to commit.
                if self.config.remasking == "low_confidence":
                    probs = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                else:  # random
                    x0_p = torch.rand(x0.shape, device=x0.device)

                # Restrict selection window to the current block (vectorized).
                x0_p = torch.where(
                    col_index < block_ends.unsqueeze(1),
                    x0_p,
                    torch.full_like(x0_p, float("-inf")),
                )

                # Only allow updates at currently masked positions.
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, float("-inf"))

                # Batched top-k transfer selection (F10): one ``topk`` for the
                # whole batch; per-row valid counts are applied as a tensor
                # mask, so no per-row host readback is needed.
                if k_max > 0:
                    k_j = num_transfer_tokens[:, step]
                    _, top_idx = torch.topk(confidence, k=k_max, dim=-1)
                    valid = torch.arange(k_max, device=x.device).unsqueeze(0) < k_j.unsqueeze(1)
                    transfer_index = torch.zeros_like(x, dtype=torch.bool)
                    transfer_index.scatter_(1, top_idx, valid)
                    x = torch.where(transfer_index, x0, x)
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
            branch_id, parent_id = self._next_folding_branch("llada_cfg")
            logits = self._forward(
                x_,
                branch_id=branch_id,
                parent_branch_id=parent_id,
                folded_model=folded_model,
                step_idx=step_idx,
                attention_mask=am,
            )
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = uncond_logits + (cfg_scale + 1.0) * (cond_logits - uncond_logits)
        else:
            branch_id, parent_id = self._next_folding_branch("llada")
            logits = self._forward(
                x,
                branch_id=branch_id,
                parent_branch_id=parent_id,
                folded_model=folded_model,
                step_idx=step_idx,
                attention_mask=attention_mask,
            )
        return logits

    def decode_final(self, x: torch.Tensor) -> torch.Tensor:
        """Return token IDs unchanged (LLaDA already produces discrete tokens)."""
        return x
