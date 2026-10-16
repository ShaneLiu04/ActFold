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
from actfold.models.speculative_core import SpecCtx, SpeculativeCore


@dataclass
class DreamSamplerConfig(SamplerConfig):
    """Configuration for the Dream masked diffusion sampler.

    Args:
        num_block_candidates: Number of speculative candidate continuations
            per super-step (AR023 FR3). ``0`` (the default) keeps the exact
            reference per-step recipe; ``N > 0`` merges
            ``speculative_steps`` schedule steps into one super-step — one
            draft serial forward plus one ``[N, T]`` batched-branch
            verification call committing the merged transfer budget
            (requires a ``folded_model`` that supports ``branch_ids`` and
            batch 1).
        candidate_top_k: Draft-set size per position for the speculative
            candidates (top-k tokens of the draft logits). Consumed only
            when ``num_block_candidates > 0``.
        candidate_include_greedy: AR022 greedy anchoring, shared face
            (see :class:`~actfold.models.llada_sampler.LLaDASamplerConfig`).
        speculative_steps: Super-step size ``s`` (the Dream-only family
            knob): the number of schedule steps merged into one
            speculative unit (budget = the sum of their transfer counts,
            truncated by the still-masked count). ``>= 1``; ``s`` covering
            the whole schedule degenerates to a one-shot tail.
    """

    eps: float = 1e-3
    alg: str = "maskgit_plus"  # "maskgit_plus" | "topk_margin" | "entropy"
    alg_temp: float = 0.0
    stochastic_transfer: bool = False
    right_shift_logits: bool = True
    cfg_scale: float = 0.0
    scheduler: MaskingScheduler | None = None
    num_block_candidates: int = 0
    candidate_top_k: int = 10
    candidate_include_greedy: bool = False
    speculative_steps: int = 8

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
        if self.speculative_steps < 1:
            raise ValueError(f"speculative_steps must be >= 1, got {self.speculative_steps}")


class DreamSampler(DiffusionSampler, SpeculativeCore):
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
        # AR023: instance-level greedy_win observation counters (AR022
        # precedent) — survive across ``sample`` calls, incremented only on
        # the greedy-anchored speculative path.
        self._greedy_wins = 0
        self._greedy_spec_calls = 0

    def initialize(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Build a left-padded canvas with prompts right-aligned."""
        mask_token_id, eos_token_id, _ = self._get_special_token_ids()
        inputs = [prompt_ids[i] for i in range(prompt_ids.shape[0])]
        x, attention_mask, seq_lens = build_left_padded_canvas(
            inputs, self.config.num_tokens, eos_token_id, mask_token_id
        )
        # Dream's attention accepts bool masks directly (1 = attend). Passing a
        # long mask raises in torch SDPA, so normalize to bool here.
        self._attention_mask = attention_mask.to(dtype=torch.bool)
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

    def _spec_ctx_for(self, x: torch.Tensor, prompt_len: int) -> SpecCtx:
        """Return the pinned Dream speculative fold context (AR023 §4.1.4).

        The context is keyed by ``(T, prompt_len)`` and object-pinned through
        the Core's LRU skeleton (cap 32). The mask is the family's bool
        ``[1, T]`` left-padded attention mask (the ``initialize`` form —
        Dream's SDPA contract accepts bool masks directly); the position
        payload is ``compute_position_ids(mask)`` — the 1-D positions that
        ignore left-side padding, threaded through ``_forward``'s
        ``position_ids`` tail parameter by the family ``forward_fn`` closure.
        """
        seq_len = int(x.shape[1])
        key = (seq_len, int(prompt_len))
        cached = self._ctx_cache_get(key)
        if cached is not None:
            return cached
        family_mask = self._attention_mask
        position = compute_position_ids(family_mask)
        ctx = SpecCtx(mask=family_mask, position=position)
        self._ctx_cache_put(key, ctx)
        return ctx

    def _build_unit_forced(
        self,
        x: torch.Tensor,
        region_start: int,
        region_end: int,
        masked_local: torch.Tensor,
    ) -> torch.Tensor:
        """Return the ``[T]`` bool forced tensor for one super-step unit.

        True exactly at the selected commit positions
        (``region_start + masked_local``) — the precise "about to change"
        set (the Dream super-step has no block superset counterpart; the
        selected positions are the P5 partial-flip equivalent domain).
        False elsewhere.
        """
        del region_end
        forced = torch.zeros(int(x.shape[1]), dtype=torch.bool, device=x.device)
        forced[(masked_local.reshape(-1) + region_start)] = True
        return forced

    def sample(
        self,
        prompt_ids: torch.Tensor,
        folded_model: FoldedModel | None = None,
    ) -> SamplerOutput:
        """Run Dream-style iterative masked diffusion sampling.

        With ``num_block_candidates > 0`` (AR023) every
        ``speculative_steps``-step window of the transfer schedule becomes
        one speculative super-step unit: one draft serial forward plus one
        ``[N, T]`` batched-branch verification call through the shared
        :class:`~actfold.models.speculative_core.SpeculativeCore` commits
        the merged transfer budget (the family confidence rule selects the
        commit positions; default off keeps the reference per-step recipe
        byte-identical).

        Raises:
            ValueError: If ``num_block_candidates > 0`` without a
                ``folded_model``, with a folded model that does not support
                the ``branch_ids`` kwarg, with a batch larger than 1 (the
                family fold-context mask cannot express per-row padding),
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
                    "DreamSampler speculative sampling supports batch 1 only; "
                    "the family fold-context mask cannot express per-row "
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
        mask_token_id = self._mask_token_id
        attention_mask = self._attention_mask

        # The layer-driven folded path bypasses the raw model's internal
        # position-id computation (remote Dream rotary reads ``position_ids``
        # directly), so every folded call carries the explicit payload even
        # for a batch-1 no-pad canvas — where it is the plain ``arange``,
        # value-identical to what the raw model computes internally.
        pos_id: torch.Tensor | None = None
        if torch.any(attention_mask == 0) or folded_model is not None:
            pos_id = compute_position_ids(attention_mask)

        mask_index = x == mask_token_id
        num_transfer_tokens_list = get_num_transfer_tokens(
            mask_index=mask_index,
            steps=self.config.num_steps,
            scheduler=self.config.scheduler,
            stochastic=self.config.stochastic_transfer,
        )
        effective_steps = num_transfer_tokens_list.size(1)
        # One host readback per sample: the batched top-k width below.
        k_max = int(num_transfer_tokens_list.max().item())

        # For CFG, only the original prompt positions are masked in the
        # unconditional branch; step-wise revealed tokens are not masked again.
        prompt_index = attention_mask.bool() & (
            torch.arange(T, device=x.device).unsqueeze(0) < T - self.config.num_tokens
        )

        history: list[torch.Tensor] = [x.clone()] if self.config.return_history else []

        if num_candidates > 0:
            # AR023 FR3: the super-step speculative path — one draft + one
            # [N, T] batched verification per merged schedule window.
            s = int(self.config.speculative_steps)
            prompt_len = int(prompt_ids.shape[1])
            ctx = self._spec_ctx_for(x, prompt_len=prompt_len)
            eos_token_id = self._eos_token_id
            finished = torch.zeros(1, dtype=torch.bool, device=x.device)
            unit_idx = 0
            step = 0
            while step < effective_steps:
                mask_index = x == mask_token_id
                masked_count = int(mask_index[0].sum().item())
                if masked_count == 0:
                    break
                # s-step budget merge (host integers): the sum of the
                # window's transfer counts, truncated by the still-masked
                # count (the canvas wins over the schedule at the tail).
                budget = int(num_transfer_tokens_list[0, step : step + s].sum().item())
                k_eff_budget = min(budget, masked_count)
                if k_eff_budget <= 0:
                    step += s
                    continue

                draft_branch_id, draft_parent_id = self._next_folding_branch(
                    f"dream_spec_u{unit_idx}"
                )
                draft_logits = self._forward(
                    x,
                    branch_id=draft_branch_id,
                    parent_branch_id=draft_parent_id,
                    folded_model=folded_model,
                    step_idx=step,
                    attention_mask=ctx.mask,
                    position_ids=ctx.position,
                )
                if self.config.right_shift_logits:
                    draft_logits = right_shift_logits(draft_logits)

                # Position selection: the family confidence rule (the same
                # ``sample_tokens`` face as the reference path) picks the
                # top-``k_eff_budget`` still-masked positions.
                mask_logits = draft_logits[mask_index]
                confidence, _x0 = sample_tokens(
                    mask_logits,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    margin_confidence=self.config.alg == "topk_margin",
                    neg_entropy=self.config.alg == "entropy",
                )
                full_confidence = torch.full_like(
                    x, float("-inf"), device=x.device, dtype=draft_logits.dtype
                )
                full_confidence[mask_index] = confidence
                _, top_idx = torch.topk(full_confidence, k=k_eff_budget, dim=-1)
                selected = top_idx[0]
                unit_forced = self._build_unit_forced(
                    x, region_start=0, region_end=T, masked_local=selected
                )

                def _batched_forward(tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                    return self._forward(
                        tokens,
                        folded_model=folded_model,
                        step_idx=step,
                        attention_mask=ctx.mask,
                        position_ids=ctx.position,
                        **kwargs,
                    )

                self._speculative_commit(
                    x=x,
                    finished=finished,
                    draft_logits_region=draft_logits,
                    region_start=0,
                    masked_local=selected,
                    draft_branch_id=draft_branch_id,
                    ctx=ctx,
                    unit_forced=unit_forced,
                    folded_model=folded_model,
                    unit_idx=unit_idx,
                    sub_idx=0,
                    history=history,
                    forward_fn=_batched_forward,
                    stop_token_id=eos_token_id,
                )
                if bool(finished[0]):
                    break
                step += s
                unit_idx += 1

            decoded = self.decode_final(x)
            return SamplerOutput(sequences=decoded, history=history)

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

            # Candidate canvas (F10): built ONCE per step, outside the
            # per-row loop — x0 written at masked positions, x kept elsewhere.
            candidate = x.clone()
            candidate[mask_index] = x0

            if self.config.alg_temp is None or self.config.alg_temp == 0.0:
                # Batched top-k transfer (F10): one ``topk`` for the whole
                # batch with per-row valid counts applied as a tensor mask.
                if k_max > 0:
                    k_j = num_transfer_tokens_list[:, step]
                    _, top_idx = torch.topk(full_confidence, k=k_max, dim=-1)
                    valid = torch.arange(k_max, device=x.device).unsqueeze(0) < k_j.unsqueeze(1)
                    transfer_index = torch.zeros_like(x, dtype=torch.bool)
                    transfer_index.scatter_(1, top_idx, valid)
                    x = torch.where(transfer_index, candidate, x)
            else:
                for j in range(B):
                    k = int(num_transfer_tokens_list[j, step].item())
                    if k > 0:
                        fc = full_confidence[j] / self.config.alg_temp
                        fc = F.softmax(fc, dim=-1)
                        transfer_index = torch.multinomial(fc, num_samples=k)
                        x[j, transfer_index] = candidate[j, transfer_index]

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
            branch_id, parent_id = self._next_folding_branch("dream_cfg")
            logits = self._forward(
                x_,
                branch_id=branch_id,
                parent_branch_id=parent_id,
                folded_model=folded_model,
                step_idx=step_idx,
                attention_mask=am,
                position_ids=pos,
            )
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = uncond_logits + (cfg_scale + 1.0) * (cond_logits - uncond_logits)
        else:
            branch_id, parent_id = self._next_folding_branch("dream")
            logits = self._forward(
                x,
                branch_id=branch_id,
                parent_branch_id=parent_id,
                folded_model=folded_model,
                step_idx=step_idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return logits

    def decode_final(self, x: torch.Tensor) -> torch.Tensor:
        """Return token IDs unchanged (Dream sampler already produces discrete tokens)."""
        return x
