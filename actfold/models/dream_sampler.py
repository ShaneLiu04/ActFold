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
from typing import Any, NamedTuple

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
        uncertainty_validation_order: AR031 FR4 opt-in uncertainty-aware
            validation order (the shared SpeculativeCore face — see
            :class:`~actfold.models.llada_sampler.LLaDASamplerConfig`).
            ``False`` (the default) is byte-identical. Requires
            ``num_block_candidates > 0``.
        uncertainty_margin: AR031 early-stop margin, ``[0, 1]``.
        uncertainty_margin_gap: AR031 early-stop gap, ``[0, 1]``.
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
    uncertainty_validation_order: bool = False
    uncertainty_margin: float = 0.9
    uncertainty_margin_gap: float = 0.05

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
        if self.uncertainty_validation_order and self.num_block_candidates == 0:
            raise ValueError(
                "uncertainty_validation_order=True requires num_block_candidates"
                " > 0 (the uncertainty-aware validation order is defined only"
                f" on the speculative path; got num_block_candidates="
                f"{self.num_block_candidates})"
            )
        if not 0.0 <= self.uncertainty_margin <= 1.0:
            raise ValueError(f"uncertainty_margin must be in [0, 1], got {self.uncertainty_margin}")
        if not 0.0 <= self.uncertainty_margin_gap <= 1.0:
            raise ValueError(
                "uncertainty_margin_gap must be in [0, 1], got " f"{self.uncertainty_margin_gap}"
            )


class _DreamJointWindow(NamedTuple):
    """Per-chain prepared super-step entry for the joint lockstep loop (AR025).

    One chain's segment-A output bundle for a single super-step unit: the
    draft serial forward is done, the confidence-selected commit positions
    are fixed, and the candidates are built — the entry waits for the ONE
    joint ``[sum_chains*N, T]`` verification call.
    """

    chain: int
    selected: torch.Tensor
    unit_forced: torch.Tensor
    draft_logits: torch.Tensor
    draft_branch_id: str
    cands: torch.Tensor
    cand_branch_ids: list[str]


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
        # AR031: instance-level uncertainty observation counters (the
        # greedy_win_stats precedent) — survive across ``sample`` calls,
        # incremented only on the uncertainty-armed speculative path.
        self._unc_validated_rows = 0
        self._unc_skipped_rows = 0
        self._unc_early_stops = 0

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

    def sample_joint(
        self,
        prompt_ids_list: list[torch.Tensor],
        folded_model: FoldedModel | None = None,
    ) -> list[SamplerOutput]:
        """W2 lockstep joint speculative sampling (AR025 DD8a, Dream face).

        ONE sampler instance drives EVERY chain in a single-threaded
        lockstep loop over the ``speculative_steps`` super-step schedule,
        with per-chain canvas / ``finished`` / history / folding-chain
        state. Per super-step unit: per-chain draft serial forward
        (``[1, T]`` each, chain-qualified branch tags
        ``dream_spec_j{c}_u{unit}``), per-chain confidence position
        selection and candidate construction (``_speculative_build`` — the
        frozen seed formula has NO chain component), ONE joint
        ``[sum_chains*N, T]`` batched verification call with per-row
        parents through the ``parent_branch_ids`` C4 form (the F2 merged
        closure below rewrites the Core's frozen singular call form
        defensively and threads the Dream ctx ``position_ids`` payload),
        then per-chain score/commit (``_speculative_score_commit``). A
        finished or fully-denoised chain is REMOVED from the joint batch
        (C5) and the survivors continue alone. At ``temperature == 0`` the
        speculative path consumes no global RNG, so each chain's output is
        BITWISE equal to a back-to-back solo ``sample()`` on the same
        config — while the joint verification costs one ``put_batch`` group
        per layer per unit instead of one per chain (#42).

        The joint path is armed only when ALL preconditions hold:
        ``num_block_candidates > 0``, a ``folded_model`` that supports
        ``branch_ids``, and the folded model's ``multi_parent_batch``
        instance switch (DD6) — otherwise this dispatches per chain
        (DD7-isomorphic off-switch) and the solo entry's frozen
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
                ``prompt_ids_list``, a batch-2+ prompt (the family
                fold-context mask cannot express per-row padding), unequal
                prompt token lengths (C2), ``cfg_scale > 0`` (frozen
                mirror), ``temperature > 0`` or truthy
                ``stochastic_transfer`` (F3 — the confidence selection
                consumes temperature draws, and an interleaved joint loop
                could not reproduce back-to-back ``sample()`` randomness).
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
                "DreamSampler.sample_joint requires at least one prompt "
                "chain (got an empty list)."
            )
        for chain_idx, prompt_ids in enumerate(prompt_ids_list):
            if prompt_ids.shape[0] > 1:
                raise ValueError(
                    "DreamSampler.sample_joint supports batch-1 prompts only; "
                    "the family fold-context mask cannot express per-row "
                    f"padding (chain {chain_idx} got batch {prompt_ids.shape[0]})."
                )
        prompt_len_first = int(prompt_ids_list[0].shape[1])
        for chain_idx, prompt_ids in enumerate(prompt_ids_list):
            if int(prompt_ids.shape[1]) != prompt_len_first:
                raise ValueError(
                    "DreamSampler.sample_joint requires equal prompt token "
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
                "confidence position selection consumes temperature draws — "
                "an interleaved joint loop cannot reproduce back-to-back "
                "sample() randomness at temperature > 0)."
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
                "DreamSampler.sample_joint resolved unequal canvas lengths "
                f"across chains ({[int(x.shape[1]) for x in xs]}); the chains "
                "must share one fold-context key (T, prompt_len)."
            )
        mask_token_id = self._mask_token_id
        eos_token_id = self._eos_token_id
        # C2: BOTH canvases are initialized BEFORE the ctx lookup — the
        # Dream ctx content comes from the instance ``_attention_mask``
        # that ``initialize`` overwrites; equal-length prompts resolve to
        # identical masks, so the shared ctx is every chain's own ctx.
        ctx = self._spec_ctx_for(xs[0], prompt_len=prompt_len_first)

        # Per-chain transfer schedules (equal by the C2 prompt-length
        # equality; computed per chain verbatim from each canvas).
        transfer_schedules: list[torch.Tensor] = []
        for x in xs:
            mask_index = x == mask_token_id
            transfer_schedules.append(
                get_num_transfer_tokens(
                    mask_index=mask_index,
                    steps=self.config.num_steps,
                    scheduler=self.config.scheduler,
                    stochastic=self.config.stochastic_transfer,
                )
            )
        effective_steps = transfer_schedules[0].size(1)
        s = int(self.config.speculative_steps)

        finished = [torch.zeros(1, dtype=torch.bool, device=x.device) for x in xs]
        histories: list[list[torch.Tensor]] = [
            [x.clone()] if self.config.return_history else [] for x in xs
        ]
        # Per-chain folding-chain state (F1): the joint loop owns each
        # chain's parent externally; segment B publishes the winner through
        # the instance slot, which the loop reads back per chain.
        parents: list[str | None] = [None] * num_chains

        # Per-chain speculative-unit counters mirroring the solo loop's
        # semantics EXACTLY (solo increments only on a built window — a
        # zero-budget window ``continue``s without incrementing): a shared
        # counter desynchronizes the frozen seed formula's ``unit_idx``
        # component against solo whenever the two chains' zero-budget
        # windows differ, breaking the joint-vs-solo bit-equality premise.
        unit_counts = [0] * num_chains
        step = 0
        while step < effective_steps:
            # Per-chain super-step prep (segment A): draft serial forward +
            # confidence position selection + candidate build. Finished or
            # fully-denoised chains are REMOVED (C5); a zero-budget window
            # skips the chain with zero forwards.
            window: list[_DreamJointWindow] = []
            for chain in range(num_chains):
                if bool(finished[chain][0]):
                    continue
                mask_index = xs[chain] == mask_token_id
                masked_count = int(mask_index[0].sum().item())
                if masked_count == 0:
                    continue
                budget = int(transfer_schedules[chain][0, step : step + s].sum().item())
                k_eff_budget = min(budget, masked_count)
                if k_eff_budget <= 0:
                    continue
                # Chain-qualified branch tags (the joint namespace): value
                # semantics equal the solo counter-suffixed ids — ids only
                # key the cache — and the two chains never collide (C4).
                chain_unit = unit_counts[chain]
                draft_branch_id = f"dream_spec_j{chain}_u{chain_unit}"
                draft_parent_id = parents[chain]
                parents[chain] = draft_branch_id
                draft_logits = self._forward(
                    xs[chain],
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
                    xs[chain], float("-inf"), device=xs[chain].device, dtype=draft_logits.dtype
                )
                full_confidence[mask_index] = confidence
                _, top_idx = torch.topk(full_confidence, k=k_eff_budget, dim=-1)
                selected = top_idx[0]
                unit_forced = self._build_unit_forced(
                    xs[chain], region_start=0, region_end=T, masked_local=selected
                )
                cands, cand_branch_ids = self._speculative_build(
                    x=xs[chain],
                    draft_logits_region=draft_logits,
                    region_start=0,
                    masked_local=selected,
                    draft_branch_id=draft_branch_id,
                    unit_idx=chain_unit,
                    sub_idx=0,
                    stop_token_id=eos_token_id,
                )
                window.append(
                    _DreamJointWindow(
                        chain=chain,
                        selected=selected,
                        unit_forced=unit_forced,
                        draft_logits=draft_logits,
                        draft_branch_id=draft_branch_id,
                        cands=cands,
                        cand_branch_ids=cand_branch_ids,
                    )
                )
                # Solo-mirror increment: only a BUILT window advances the
                # chain's unit counter (zero-budget/finished chains were
                # ``continue``d above without incrementing).
                unit_counts[chain] = chain_unit + 1

            if window:
                n = num_candidates
                joint_cands = torch.cat([entry.cands for entry in window], dim=0)
                joint_branch_ids = [bid for entry in window for bid in entry.cand_branch_ids]
                # Per-row parents (C4): chain i's N candidate rows hang on
                # chain i's OWN draft branch — never on another chain's
                # winner.
                joint_parent_ids = [entry.draft_branch_id for entry in window for _ in range(n)]
                joint_forced = torch.cat(
                    [entry.unit_forced.unsqueeze(0).repeat(n, 1) for entry in window],
                    dim=0,
                )

                def _joint_batched_forward(tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                    # F2 (design rev 4): the merged closure face. The
                    # Core's frozen composed call form threads the SINGULAR
                    # ``parent_branch_id``; consume it and rewrite it as
                    # the per-row plural (never forwarded alongside it —
                    # the MFF mutex face), so a frozen-form caller and the
                    # C4 joint form share ONE closure. The plural form
                    # passes through unchanged; the Dream ctx
                    # ``position_ids`` payload threads through every joint
                    # batched call.
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
                    # the C4 form carries neither singular — both resolve
                    # to None here (the MFF mutex face).
                    return self._forward(
                        tokens,
                        kwargs.pop("branch_id", None),
                        kwargs.pop("parent_branch_id", None),
                        folded_model=folded_model,
                        step_idx=step,
                        attention_mask=ctx.mask,
                        position_ids=ctx.position,
                        **kwargs,
                    )

                joint_logits = _joint_batched_forward(
                    joint_cands,
                    branch_id=None,
                    parent_branch_ids=joint_parent_ids,
                    branch_ids=joint_branch_ids,
                    forced_divergent=joint_forced,
                )

                # Per-chain segment B on the row-sliced joint logits; the
                # instance parent slot swaps in/out per chain (F1).
                for slot, entry in enumerate(window):
                    chain_logits = joint_logits[slot * n : (slot + 1) * n]
                    self._speculative_score_commit(
                        x=xs[entry.chain],
                        finished=finished[entry.chain],
                        batched_logits=chain_logits,
                        cands=entry.cands,
                        cand_branch_ids=entry.cand_branch_ids,
                        draft_logits_region=entry.draft_logits,
                        region_start=0,
                        masked_local=entry.selected,
                        history=histories[entry.chain],
                        stop_token_id=eos_token_id,
                    )
                    parents[entry.chain] = self._folding_parent
            step += s
            if all(bool(flags[0]) for flags in finished):
                break

        return [
            SamplerOutput(sequences=self.decode_final(x), history=history)
            for x, history in zip(xs, histories)
        ]

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
