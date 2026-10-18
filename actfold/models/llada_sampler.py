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
from typing import Any, NamedTuple

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
        speculative_sub_block_size: AR024 FR1 sub-block window size ``s``.
            Blocks split into ``s``-sized windows (remainder tail); each
            window runs one draft serial forward plus one ``[N, T]``
            batched verification (the ``s == block_size`` degenerate is the
            AR023 whole-block unit, bit-identical). ``None`` (the default)
            normalizes to ``8`` and is the unvalidated default face — the
            runtime window clamps to the block geometry (zero disturbance
            for existing configs); an explicit ``s`` is user intent and
            validates against the block geometry on the speculative path.
        progressive_refill: AR024 FR2 two-phase refill (opt-in). After the
            Core's partial top-k commit, one extra batch-1 serial forward
            on the committed context fills the window's remaining masked
            positions with the deterministic argmax (one extra forward per
            window upper bound). Requires ``num_block_candidates > 0``.
        refill_top_k: AR024 FR2 committed-position count ``k``.
            ``None`` (the default) resolves to ``s // 2`` at the phase-two
            entry; an explicit ``k`` wins. Validated in ``[1,
            speculative_sub_block_size]`` whenever set (malformed values
            are path-independent).
        candidate_eos_aware: AR024 FR3 EOS-aware continuation row (default
            ``True`` — the recommended AR024 stack). With ``N > 0``,
            candidate row N-1 is the first non-EOS draft-pool entry per
            position (pure device indexing; an EOS-absent pool degenerates
            to the greedy value). ``False`` restores the AR023 semantics
            exactly (the A/B anchor face).
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
    speculative_sub_block_size: int | None = None
    progressive_refill: bool = False
    refill_top_k: int | None = None
    candidate_eos_aware: bool = True

    def __post_init__(self) -> None:
        sub_block = self.speculative_sub_block_size
        self._sub_block_size_explicit = sub_block is not None
        sub_block = 8 if sub_block is None else sub_block
        self.speculative_sub_block_size = sub_block
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
        if sub_block < 1:
            raise ValueError(f"speculative_sub_block_size must be >= 1, got {sub_block}")
        if self.progressive_refill and self.num_block_candidates == 0:
            raise ValueError(
                "progressive_refill=True requires num_block_candidates > 0 "
                "(the two-phase refill is defined only on the speculative "
                f"path; got num_block_candidates={self.num_block_candidates})"
            )
        if self.refill_top_k is not None and (
            self.refill_top_k < 1 or self.refill_top_k > sub_block
        ):
            raise ValueError(
                "refill_top_k must be in [1, speculative_sub_block_size="
                f"{self.speculative_sub_block_size}] when set, got {self.refill_top_k}"
            )
        if (
            self._sub_block_size_explicit
            and self.num_block_candidates > 0
            and self.block_size is not None
            and sub_block > self.block_size
        ):
            raise ValueError(
                f"speculative_sub_block_size ({self.speculative_sub_block_size}) "
                f"must not exceed block_size ({self.block_size}) on the "
                "speculative path"
            )


class _LLaDAJointWindow(NamedTuple):
    """Per-chain prepared window entry for the joint lockstep loop (AR025).

    One chain's segment-A output bundle for a single sub-block window: the
    draft serial forward is done, the candidates are built, and the entry
    waits for the ONE joint ``[sum_chains*N, T]`` verification call.
    """

    chain: int
    masked_local: torch.Tensor
    unit_forced: torch.Tensor
    region_logits: torch.Tensor
    draft_branch_id: str
    cands: torch.Tensor
    cand_branch_ids: list[str]


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
        sub_block = self.config.speculative_sub_block_size
        assert sub_block is not None  # normalized in LLaDASamplerConfig.__post_init__
        self._sub_block_size = sub_block
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
        (AR023/AR024) each block instead becomes a sequence of ``s``-sized
        sub-block speculative windows (``speculative_sub_block_size``; the
        ``s == block_size`` degenerate is the AR023 whole-block unit): one
        draft serial forward plus one ``[N, T]`` batched-branch verification
        call per window through the shared
        :class:`~actfold.models.speculative_core.SpeculativeCore`, with the
        optional two-phase refill (``progressive_refill``) filling the
        window remainder on the committed context (default off keeps the
        reference recipe byte-identical).

        Raises:
            ValueError: If ``num_block_candidates > 0`` without a
                ``folded_model``, with a folded model that does not support
                the ``branch_ids`` kwarg, with a batch larger than 1 (the
                ``[T, T]`` fold-context mask cannot express per-row padding),
                with ``cfg_scale > 0`` (the speculative path does not
                implement classifier-free guidance; never a silent drop), or
                with an explicit ``speculative_sub_block_size`` exceeding the
                resolved ``block_size`` (the sample-entry resolved-geometry
                validation — ``block_size=None`` resolves from the
                generation geometry at entry, before any forward).
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
        if (
            num_candidates > 0
            and getattr(self.config, "_sub_block_size_explicit", True)
            and self._sub_block_size > block_size
        ):
            raise ValueError(
                f"speculative_sub_block_size ({self._sub_block_size}) must not "
                f"exceed the resolved block_size ({block_size}) on the "
                "speculative path"
            )
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
                # AR024 FR1: blocks split into s-sized sub-block windows
                # (remainder tail); one draft serial forward plus one
                # [N, T] batched verification per WINDOW (the AR023
                # whole-block unit is the s == block_size degenerate).
                # The chain continues per window — each window's draft
                # parent is the previous window's winner branch.
                assert finished is not None  # gated: created when num_candidates > 0
                if bool(finished.all()):
                    break
                region_start = prompt_lens[0] + block_idx * block_size
                region_end = min(region_start + block_size, prompt_lens[0] + max_new_tokens, T)
                if region_start >= region_end:
                    continue
                sub_block = self._sub_block_size
                w_start, w_idx = region_start, 0
                while w_start < region_end:
                    if bool(finished.all()):
                        break
                    w_end = min(w_start + sub_block, region_end)
                    masked_local = (
                        (x[0, w_start:w_end] == mask_token_id).nonzero(as_tuple=False).reshape(-1)
                    )
                    if int(masked_local.numel()) == 0:
                        # Window without masked positions: skipped with
                        # zero forwards (w_idx still counts every window).
                        w_start, w_idx = w_end, w_idx + 1
                        continue
                    ctx = self._spec_ctx_for(x, prompt_len=prompt_lens[0])
                    unit_forced = self._build_unit_forced(
                        x,
                        region_start=w_start,
                        region_end=w_end,
                        masked_local=masked_local,
                    )
                    # Frozen branch tags (A/B anchor face): w_idx == 0
                    # keeps the AR023 form; w_idx > 0 carries the window
                    # index suffix.
                    if w_idx == 0:
                        tag = f"llada_spec_b{block_idx}"
                    else:
                        tag = f"llada_spec_b{block_idx}_w{w_idx}"
                    # Frozen branch tags carry NO folding-counter suffix (the
                    # design-pinned A/B anchor face) — the chain bookkeeping
                    # mirrors ``_next_folding_branch`` (parent = the current
                    # chain slot, then the draft owns the slot).
                    draft_parent_id = self._folding_parent
                    draft_branch_id = tag
                    self._folding_parent = draft_branch_id
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
                    region_logits = draft_logits[:, w_start:w_end, :]

                    def _batched_forward(tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                        return self._forward(
                            tokens,
                            folded_model=folded_model,
                            step_idx=block_idx,
                            attention_mask=ctx.mask,
                            position_embeddings=ctx.position,
                            **kwargs,
                        )

                    k_resolved: int | None = None
                    if self.config.progressive_refill:
                        k_resolved = (
                            self.config.refill_top_k
                            if self.config.refill_top_k is not None
                            else sub_block // 2
                        )
                    self._speculative_commit(
                        x=x,
                        finished=finished,
                        draft_logits_region=region_logits,
                        region_start=w_start,
                        masked_local=masked_local,
                        draft_branch_id=draft_branch_id,
                        ctx=ctx,
                        unit_forced=unit_forced,
                        folded_model=folded_model,
                        unit_idx=block_idx,
                        sub_idx=w_idx,
                        history=history,
                        forward_fn=_batched_forward,
                        stop_token_id=eos_token_id,
                        refill_top_k=k_resolved,
                    )
                    if k_resolved is not None:
                        self._refill_phase_two(
                            x,
                            finished,
                            w_start,
                            w_end,
                            block_idx,
                            w_idx,
                            ctx,
                            folded_model,
                            history,
                            eos_token_id,
                        )
                        if self.config.return_history:
                            history.append(x.clone())
                    w_start, w_idx = w_end, w_idx + 1
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

    def sample_joint(
        self,
        prompt_ids_list: list[torch.Tensor],
        folded_model: FoldedModel | None = None,
    ) -> list[SamplerOutput]:
        """W2 lockstep joint speculative sampling (AR025 DD8a, LLaDA face).

        ONE sampler instance drives EVERY chain in a single-threaded
        lockstep loop with per-chain canvas / ``finished`` / history /
        folding-chain state. Per sub-block window: per-chain draft serial
        forward (``[1, T]`` each, chain-qualified branch tags
        ``llada_spec_j{c}_b{block}[_w{w}]``), per-chain candidate
        construction (``_speculative_build`` — the frozen seed formula has
        NO chain component, so per-chain candidates equal the solo build
        bitwise), ONE joint ``[sum_chains*N, T]`` batched verification call
        with per-row parents through the ``parent_branch_ids`` C4 form (the
        F2 merged closure below rewrites the Core's frozen singular call
        form defensively), then per-chain score/commit
        (``_speculative_score_commit``). An EOS-finished chain is REMOVED
        from the joint batch (C5) and the survivors continue alone. Because
        the speculative path consumes no global RNG at ``temperature == 0``
        (candidate draws replay the frozen seed formula on dedicated CPU
        generators), each chain's output is BITWISE equal to a back-to-back
        solo ``sample()`` on the same config — while the joint verification
        costs one ``put_batch`` group per layer per window instead of one
        per chain (#42).

        The joint path is armed only when ALL preconditions hold:
        ``num_block_candidates > 0``, a ``folded_model`` that supports
        ``branch_ids``, and the folded model's ``multi_parent_batch``
        instance switch (DD6) — otherwise this dispatches per chain
        (``[self.sample(p, folded_model) for p in prompt_ids_list]``,
        DD7-isomorphic off-switch) and the solo entry's frozen
        validations/errors surface unchanged.

        Args:
            prompt_ids_list: Per-chain prompts, each ``[1, prompt_len]``
                (batch-1) with EQUAL token lengths across chains (C2: the
                chains share one fold-context key ``(T, prompt_len)``).
            folded_model: The folded model (must carry
                ``multi_parent_batch=True`` to arm the joint path).

        Returns:
            One :class:`~actfold.models.diffusion_sampler.SamplerOutput`
            per chain, in ``prompt_ids_list`` order.

        Raises:
            ValueError: On the joint path (switch armed): an empty
                ``prompt_ids_list``, a batch-2+ prompt (the ``[T, T]``
                fold-context mask cannot express per-row padding), unequal
                prompt token lengths (C2), ``cfg_scale > 0`` (frozen
                mirror), ``temperature > 0`` or truthy
                ``stochastic_transfer`` (F3 — the per-candidate
                construction is deterministic only on the greedy path, and
                an interleaved joint loop could not reproduce
                back-to-back ``sample()`` draws at temperature > 0).
        """
        num_candidates = int(self.config.num_block_candidates)
        joint_armed = (
            num_candidates > 0
            and folded_model is not None
            and self._supports_branch_ids(folded_model)
            and bool(getattr(folded_model, "multi_parent_batch", False))
        )
        if not joint_armed:
            # Off-switch dispatch (DD7-isomorphic): per-chain solo runs;
            # the solo entry's frozen validations/errors surface unchanged.
            return [self.sample(prompt_ids, folded_model) for prompt_ids in prompt_ids_list]

        # Joint-path entry validations (F3/C2 and the frozen cfg mirror).
        if not prompt_ids_list:
            raise ValueError(
                "LLaDASampler.sample_joint requires at least one prompt "
                "chain (got an empty list)."
            )
        for chain_idx, prompt_ids in enumerate(prompt_ids_list):
            if prompt_ids.shape[0] > 1:
                raise ValueError(
                    "LLaDASampler.sample_joint supports batch-1 prompts only; "
                    "the [T, T] fold-context mask cannot express per-row "
                    f"padding (chain {chain_idx} got batch {prompt_ids.shape[0]})."
                )
        prompt_len_first = int(prompt_ids_list[0].shape[1])
        for chain_idx, prompt_ids in enumerate(prompt_ids_list):
            if int(prompt_ids.shape[1]) != prompt_len_first:
                raise ValueError(
                    "LLaDASampler.sample_joint requires equal prompt token "
                    "lengths across chains (the shared fold-context key "
                    f"(T, prompt_len) — chain 0 has {prompt_len_first}, "
                    f"chain {chain_idx} has {int(prompt_ids.shape[1])})."
                )
        if self.config.cfg_scale > 0.0:
            raise ValueError(
                "sample_joint requires cfg_scale == 0.0 (the joint "
                "speculative draft/batched path does not implement "
                "classifier-free guidance; refusing to silently drop it)."
            )
        if self.config.temperature > 0.0:
            raise ValueError(
                "sample_joint requires temperature == 0.0 (F3: the "
                "per-candidate construction is deterministic only on the "
                "greedy path — an interleaved joint loop cannot reproduce "
                "back-to-back sample() draws at temperature > 0)."
            )
        if self.config.stochastic_transfer:
            raise ValueError(
                "sample_joint requires stochastic_transfer == False (F3: "
                "stochastic transfer schedules consume the global RNG and "
                "would desynchronize the joint lockstep's per-chain parity)."
            )

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        self._reset_folding_chain()
        xs = [self.initialize(prompt_ids) for prompt_ids in prompt_ids_list]
        num_chains = len(xs)
        T = int(xs[0].shape[1])
        if any(int(x.shape[1]) != T for x in xs):
            raise ValueError(
                "LLaDASampler.sample_joint resolved unequal canvas lengths "
                f"across chains ({[int(x.shape[1]) for x in xs]}); the chains "
                "must share one fold-context key (T, prompt_len)."
            )
        max_new_tokens = self.config.num_tokens
        block_size = self._block_size
        if (
            getattr(self.config, "_sub_block_size_explicit", True)
            and self._sub_block_size > block_size
        ):
            raise ValueError(
                f"speculative_sub_block_size ({self._sub_block_size}) must not "
                f"exceed the resolved block_size ({block_size}) on the "
                "speculative path"
            )
        mask_token_id = self._mask_token_id
        eos_token_id = self._eos_token_id

        # Per-chain prompt lengths (the solo vectorized block, batch-1 per
        # canvas); C2 requires them equal — the chains share ONE ctx.
        prompt_lens: list[int] = []
        for x in xs:
            is_mask = x == mask_token_id
            any_mask = is_mask.any(dim=1)
            any_non_eos = (x != eos_token_id).any(dim=1)
            first_mask = torch.argmax(is_mask.to(torch.uint8), dim=1)
            full_len = torch.full_like(first_mask, T)
            zero = torch.zeros_like(first_mask)
            prompt_lens_t = torch.where(
                any_mask, first_mask, torch.where(any_non_eos, full_len, zero)
            )
            prompt_lens.append(int(prompt_lens_t[0].item()))
        if len(set(prompt_lens)) != 1:
            raise ValueError(
                "LLaDASampler.sample_joint resolved unequal prompt lengths "
                f"across chains ({prompt_lens}); the chains must share one "
                "fold-context key (T, prompt_len)."
            )
        prompt_len = prompt_lens[0]
        ctx = self._spec_ctx_for(xs[0], prompt_len=prompt_len)

        # Suppression id tensors, precomputed once (the F10 solo form).
        suppress_ids = (
            torch.tensor(self.config.suppress_tokens, dtype=torch.long, device=xs[0].device)
            if self.config.suppress_tokens
            else None
        )
        begin_suppress_ids = (
            torch.tensor(self.config.begin_suppress_tokens, dtype=torch.long, device=xs[0].device)
            if self.config.begin_suppress_tokens
            else None
        )

        finished = [torch.zeros(1, dtype=torch.bool, device=x.device) for x in xs]
        histories: list[list[torch.Tensor]] = [
            [x.clone()] if self.config.return_history else [] for x in xs
        ]
        # Per-chain folding-chain state (F1): the joint loop owns each
        # chain's parent externally; segment B publishes the winner through
        # the instance slot, which the loop reads back per chain.
        parents: list[str | None] = [None] * num_chains

        sub_block = self._sub_block_size
        k_resolved: int | None = None
        if self.config.progressive_refill:
            k_resolved = (
                self.config.refill_top_k if self.config.refill_top_k is not None else sub_block // 2
            )

        num_blocks = max(1, math.ceil(max_new_tokens / block_size))
        for block_idx in range(num_blocks):
            if all(bool(flags.all()) for flags in finished):
                break
            region_start = prompt_len + block_idx * block_size
            region_end = min(region_start + block_size, prompt_len + max_new_tokens, T)
            if region_start >= region_end:
                continue
            w_start, w_idx = region_start, 0
            while w_start < region_end:
                if all(bool(flags.all()) for flags in finished):
                    break
                w_end = min(w_start + sub_block, region_end)

                # Per-chain window prep (segment A): draft serial forward +
                # candidate build. Finished chains are REMOVED (C5); a
                # window without masked positions costs zero forwards.
                window: list[_LLaDAJointWindow] = []
                for chain in range(num_chains):
                    if bool(finished[chain].all()):
                        continue
                    masked_local = (
                        (xs[chain][0, w_start:w_end] == mask_token_id)
                        .nonzero(as_tuple=False)
                        .reshape(-1)
                    )
                    if int(masked_local.numel()) == 0:
                        continue
                    unit_forced = self._build_unit_forced(
                        xs[chain],
                        region_start=w_start,
                        region_end=w_end,
                        masked_local=masked_local,
                    )
                    # Chain-qualified branch tags (the joint namespace):
                    # the value semantics equal the solo tags — ids only key
                    # the cache — and the two chains never collide (C4).
                    if w_idx == 0:
                        tag = f"llada_spec_j{chain}_b{block_idx}"
                    else:
                        tag = f"llada_spec_j{chain}_b{block_idx}_w{w_idx}"
                    draft_parent_id = parents[chain]
                    draft_branch_id = tag
                    parents[chain] = draft_branch_id
                    draft_logits = self._forward(
                        xs[chain],
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
                    region_logits = draft_logits[:, w_start:w_end, :]
                    cands, cand_branch_ids = self._speculative_build(
                        x=xs[chain],
                        draft_logits_region=region_logits,
                        region_start=w_start,
                        masked_local=masked_local,
                        draft_branch_id=draft_branch_id,
                        unit_idx=block_idx,
                        sub_idx=w_idx,
                        stop_token_id=eos_token_id,
                    )
                    window.append(
                        _LLaDAJointWindow(
                            chain=chain,
                            masked_local=masked_local,
                            unit_forced=unit_forced,
                            region_logits=region_logits,
                            draft_branch_id=draft_branch_id,
                            cands=cands,
                            cand_branch_ids=cand_branch_ids,
                        )
                    )

                if window:
                    n = num_candidates
                    joint_cands = torch.cat([entry.cands for entry in window], dim=0)
                    joint_branch_ids = [bid for entry in window for bid in entry.cand_branch_ids]
                    # Per-row parents (C4): chain i's N candidate rows hang
                    # on chain i's OWN draft branch — never on another
                    # chain's winner.
                    joint_parent_ids = [entry.draft_branch_id for entry in window for _ in range(n)]
                    joint_forced = torch.cat(
                        [entry.unit_forced.unsqueeze(0).repeat(n, 1) for entry in window],
                        dim=0,
                    )

                    def _joint_batched_forward(tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                        # F2 (design rev 4): the merged closure face. The
                        # Core's frozen composed call form threads the
                        # SINGULAR ``parent_branch_id``; consume it and
                        # rewrite it as the per-row plural (never forwarded
                        # alongside it — the MFF mutex face), so a
                        # frozen-form caller and the C4 joint form share ONE
                        # closure. The plural form passes through unchanged.
                        singular = kwargs.pop("parent_branch_id", None)
                        if singular is not None:
                            if "parent_branch_ids" in kwargs:
                                raise ValueError(
                                    "the joint merged forward accepts either "
                                    "parent_branch_id (shared singular) or "
                                    "parent_branch_ids (per-row plural), not "
                                    "both."
                                )
                            kwargs["parent_branch_ids"] = [singular] * int(tokens.shape[0])
                        # ``branch_id`` / ``parent_branch_id`` are consumed
                        # positionally: the frozen form threads
                        # ``parent_branch_id`` (popped and rewritten above),
                        # the C4 form carries neither singular — both
                        # resolve to None here (the MFF mutex face).
                        return self._forward(
                            tokens,
                            kwargs.pop("branch_id", None),
                            kwargs.pop("parent_branch_id", None),
                            folded_model=folded_model,
                            step_idx=block_idx,
                            attention_mask=ctx.mask,
                            position_embeddings=ctx.position,
                            **kwargs,
                        )

                    joint_logits = _joint_batched_forward(
                        joint_cands,
                        branch_id=None,
                        parent_branch_ids=joint_parent_ids,
                        branch_ids=joint_branch_ids,
                        forced_divergent=joint_forced,
                    )

                    # Per-chain segment B on the row-sliced joint logits;
                    # the instance parent slot swaps in/out per chain (F1).
                    for slot, entry in enumerate(window):
                        chain_logits = joint_logits[slot * n : (slot + 1) * n]
                        self._speculative_score_commit(
                            x=xs[entry.chain],
                            finished=finished[entry.chain],
                            batched_logits=chain_logits,
                            cands=entry.cands,
                            cand_branch_ids=entry.cand_branch_ids,
                            draft_logits_region=entry.region_logits,
                            region_start=w_start,
                            masked_local=entry.masked_local,
                            history=histories[entry.chain],
                            stop_token_id=eos_token_id,
                            refill_top_k=k_resolved,
                        )
                        parents[entry.chain] = self._folding_parent
                        if k_resolved is not None:
                            # Phase two reads/writes the instance parent
                            # slot — swap the chain's state in and out.
                            self._folding_parent = parents[entry.chain]
                            self._refill_phase_two(
                                xs[entry.chain],
                                finished[entry.chain],
                                w_start,
                                w_end,
                                block_idx,
                                w_idx,
                                ctx,
                                folded_model,
                                histories[entry.chain],
                                eos_token_id,
                                branch_tag=(f"llada_spec_j{entry.chain}_b{block_idx}_w{w_idx}_r"),
                            )
                            parents[entry.chain] = self._folding_parent
                            if self.config.return_history:
                                histories[entry.chain].append(xs[entry.chain].clone())
                w_start, w_idx = w_end, w_idx + 1

        return [
            SamplerOutput(sequences=self.decode_final(x), history=history)
            for x, history in zip(xs, histories)
        ]

    def _refill_phase_two(
        self,
        x: torch.Tensor,
        finished: torch.Tensor,
        w_start: int,
        w_end: int,
        block_idx: int,
        w_idx: int,
        ctx: SpecCtx,
        folded_model: FoldedModel | None,
        history: list[torch.Tensor],
        eos_token_id: int,
        branch_tag: str | None = None,
    ) -> None:
        """Fill the window's still-masked remainder (AR024 FR2, design §4.1.4).

        One extra batch-1 serial forward on the committed context (parent =
        the Core-selected winner branch, published at batch 1 per #41): the
        remaining window positions take the deterministic argmax of the
        phase-two logits with the family semantic corrections applied in
        the same form as the draft face. ``finished`` conjuncts the refill
        fills; the chain slot switches to the phase-two branch. A window
        with no remaining masked positions returns without any forward (the
        ``k >= masked count`` clamp degenerates to FR1). The history list is
        carried for the frozen signature only — the family loop appends the
        per-window snapshot after this call (the Core never appends in
        refill mode).

        Args:
            x: ``[1, T]`` canvas, mutated in place at the remaining masked
                window positions.
            finished: ``[1]`` bool finished flags, mutated in place.
            w_start: Canvas start coordinate of the sub-block window.
            w_end: Canvas end coordinate of the sub-block window.
            block_idx: Block index (branch tag / step_idx cargo).
            w_idx: Window index (the frozen ``_r`` tag suffix).
            ctx: The family fold context (LRU-pinned; the same object the
                window's draft/batched calls used).
            folded_model: The folded model for the serial forward.
            history: History list (signature cargo; the caller appends).
            eos_token_id: The stop token (finished semantics).
            branch_tag: AR025 joint-path override for the phase-two branch
                id. ``None`` (the default) keeps the frozen solo tag
                ``llada_spec_b{block_idx}_w{w_idx}_r``; the joint loop
                passes the chain-qualified form
                ``llada_spec_j{chain}_b{block_idx}_w{w_idx}_r`` so the two
                chains' refill branches never collide in the shared cache.
        """
        remaining_local = (
            (x[0, w_start:w_end] == self._mask_token_id).nonzero(as_tuple=False).reshape(-1)
        )
        if int(remaining_local.numel()) == 0:
            return
        r_branch = branch_tag if branch_tag is not None else f"llada_spec_b{block_idx}_w{w_idx}_r"
        r_parent = self._folding_parent
        self._folding_parent = r_branch
        r_logits = self._forward(
            x,
            branch_id=r_branch,
            parent_branch_id=r_parent,
            folded_model=folded_model,
            step_idx=block_idx,
            attention_mask=ctx.mask,
            position_embeddings=ctx.position,
        )
        if self.config.suppress_tokens:
            r_logits.index_fill_(
                -1,
                torch.tensor(self.config.suppress_tokens, dtype=torch.long, device=x.device),
                float("-inf"),
            )
        if self.config.right_shift_logits:
            r_logits = right_shift_logits(r_logits)
        if self.config.begin_suppress_tokens:
            r_logits.index_fill_(
                -1,
                torch.tensor(self.config.begin_suppress_tokens, dtype=torch.long, device=x.device),
                float("-inf"),
            )
        fill_pos = remaining_local + w_start
        x[0, fill_pos] = r_logits[0, fill_pos, :].argmax(dim=-1)
        finished |= (x[:, fill_pos] == eos_token_id).any(dim=1)
        self._folding_parent = r_branch

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
