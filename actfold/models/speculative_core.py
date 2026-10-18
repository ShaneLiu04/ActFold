"""Shared greedy-anchored speculative-candidate mechanism core (AR023).

design.md §4.1.1/§4.1.2: the AR021 generation-loop folding + AR022 greedy
anchoring port onto the LLaDA / Dream families through ONE mechanism core.
The family adapters own the canvas geometry (padding, block windows, logits
shift convention) and the fold-context objects; this core owns the frozen
candidate construction / batched verification / min-prob selection / commit
/ chain-continuation machinery, constant-compatible with the Fast-dLLM
implementation (``fast_dllm_sampler.py`` — the mechanism's first family
face; the per-candidate seed formula, the ``[N, T]`` batch shape, the
``branch_ids`` naming, and the LRU cap are pinned by UT parity against it).

Host contract (design §4.1.2) — the host is a ``DiffusionSampler`` subclass:

- the host exposes ``config`` with ``num_block_candidates`` /
  ``candidate_top_k`` / ``candidate_include_greedy`` (``seed`` and
  ``return_history`` come from ``SamplerConfig``);
- the host initializes ``_greedy_wins`` / ``_greedy_spec_calls`` /
  ``_folding_parent`` in its ``__init__`` — this core deliberately defines
  NO ``__init__`` (the host subclass never needs ``super().__init__`` from
  the mixin);
- the host implements ``_spec_ctx_for(x, prompt_len)`` (the family canvas
  adaptation; the LRU skeleton ``_ctx_cache_get`` / ``_ctx_cache_put`` lives
  here) and ``_build_unit_forced(...)`` (the family unit forced tensor);
- the family adapter constructs the ``forward_fn`` closure at the call site
  (binding the family fold context — mask form, position payload, logits
  shift convention), so the core stays family-agnostic by construction.

Frozen semantics (AR021/AR022, cross-checked with fast_dllm_sampler.py):

- candidate row 0 under ``candidate_include_greedy=True`` is the greedy
  hypothesis — the still-masked unit positions filled with the draft argmax
  (the top-k ordered-set first entry, ``sorted=True`` topk) via pure device
  indexing: zero device readback, zero generator consumption;
- random rows ``j`` replay the seed formula
  ``(seed*1_000_003 + unit_idx*8_191 + sub_idx*131 + j) & 0x7FFF_FFFF`` on
  a dedicated CPU generator (``j`` keeps its row number; ``seed=None`` uses
  the base seed 0 — the global RNG is never consumed);
- the min-prob scoring domain is the FILL position set (the still-masked
  unit positions — AR023 generalization #1);
- the greedy row competes under the unchanged min-prob rule and CAN LOSE.

AR024 extends ``_speculative_commit`` with two optional tail parameters
(``eos_aware`` / ``refill_top_k``): the EOS-aware continuation row and the
draft-confidence top-k partial commit. Both default to the AR023
byte-identical behavior (``None`` resolution — attribute-less hosts stay
untouched); see the method docstring for the frozen semantics.

AR025 (W2 lockstep joint sampling, design DD8a N7) splits the commit into
two reusable segments: ``_speculative_build`` (segment A — candidate
construction: batch-1 guard, draft top-k pool, the frozen per-candidate
seed draws, the greedy anchor row, the EOS-aware continuation row, the
candidate branch ids) and ``_speculative_score_commit`` (segment B — the
min-prob score/select over the FILL positions, the greedy_win counters,
the ``_folding_parent`` chain-continuation write, and the full / partial
commit). The frozen ``_speculative_commit`` recomposes both segments
around the family ``forward_fn`` call — byte-identical to the AR024
monolith — while the family ``sample_joint`` loops call the segments
directly: per-chain segment A, ONE joint ``[sum_chains*N, T]`` batched
verification with per-row parents (the C4 form through
``DiffusionSampler._forward``'s ``parent_branch_ids`` tail parameter),
then per-chain segment B. The seed formula carries NO chain component
(same ``unit_idx``/``sub_idx``/``j`` per chain), which is what makes the
per-chain outputs bitwise-equal to back-to-back solo ``sample()`` runs.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from actfold.models.fast_dllm_sampler import _MAX_FOLD_CTX_ENTRIES


@dataclass
class SpecCtx:
    """Family speculative-forward context (LRU-pinned static pair).

    Attributes:
        mask: The attention-mask form the family folded path accepts
            (LLaDA: ``[T, T]`` bool excluding pad columns; Dream: ``[1, T]``
            bool) — the form is decided by the family adapter, the core is
            agnostic.
        position: The position payload (LLaDA: rotary ``(cos, sin)`` or
            ``None``; Dream: ``position_ids [1, T]``) — threaded through
            ``DiffusionSampler._forward``'s corresponding tail parameter by
            the family ``forward_fn`` closure.
    """

    mask: torch.Tensor
    position: Any | None = None


class SpeculativeCore:
    """Greedy-anchored speculative-candidate mechanism core (mixin).

    The host subclass provides the config namespace and the counters (see
    the module docstring); this class carries the frozen mechanism. It
    defines no ``__init__`` — hosts initialize their own state.
    """

    # Host-provided state (design §4.1.2 host contract): annotation-only,
    # keeping the mixin mypy-clean without introducing an __init__.
    config: Any
    _greedy_wins: int
    _greedy_spec_calls: int
    _folding_parent: str | None

    @property
    def greedy_win_stats(self) -> dict[str, int]:
        """Read-only greedy_win observation face (AR022 FR2, AR023 shared).

        Returns:
            ``{"greedy_wins": int, "speculative_calls": int}`` — instance-
            level cumulative counts surviving across ``sample`` calls. Only
            the ``candidate_include_greedy=True`` speculative path increments
            them (default off stays at ``{0, 0}``); the per-unit greedy_win
            decision is ``j_star == 0``.
        """
        return {"greedy_wins": self._greedy_wins, "speculative_calls": self._greedy_spec_calls}

    @staticmethod
    def _supports_branch_ids(folded_model: Any) -> bool:
        """Probe whether ``folded_model`` accepts the ``branch_ids`` kwarg.

        Verbatim parity with ``FastDLLMSampler._supports_branch_ids``
        (fast_dllm_sampler.py:325-349) — the same probe logic exists in both
        implementations and UT-2301 pins the True/False routes.

        Non-callables and signatures that would reject ``branch_ids`` (an
        explicit parameter list without it and without a ``**kwargs``
        catch-all) fail the probe — the caller raises ``ValueError`` instead
        of letting the batched call degrade silently. A ``**kwargs``
        passthrough (e.g. a recording subclass of
        ``ManualFoldedForward``) is accepted: the kwarg is syntactically
        forwardable and the real folded forward enforces its own contract.
        """
        target = folded_model
        if isinstance(target, torch.nn.Module):
            target = getattr(target, "forward", None)
            if target is None:
                return False
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return False
        params = signature.parameters
        if "branch_ids" in params:
            return True
        return any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())

    def _ctx_cache_get(self, key: tuple[int, int]) -> SpecCtx | None:
        """Return the pinned ``SpecCtx`` for ``key``, or ``None``.

        LRU freshness on hit (``move_to_end``); the cache is lazily created
        (the host ``__init__`` may pre-date this mixin's cache attribute).
        """
        cache: "OrderedDict[tuple[int, int], SpecCtx] | None" = getattr(
            self, "_spec_ctx_cache", None
        )
        if cache is None:
            return None
        ctx = cache.get(key)
        if ctx is not None:
            cache.move_to_end(key)
        return ctx

    def _ctx_cache_put(self, key: tuple[int, int], ctx: SpecCtx) -> None:
        """Insert/refresh ``ctx`` under ``key``; evict past the LRU cap.

        The cap is ``_MAX_FOLD_CTX_ENTRIES`` (32) — the AR021 fold-context
        LRU cap, single-sourced from ``fast_dllm_sampler.py``.
        """
        cache: "OrderedDict[tuple[int, int], SpecCtx] | None" = getattr(
            self, "_spec_ctx_cache", None
        )
        if cache is None:
            cache = OrderedDict()
            self._spec_ctx_cache = cache
        cache[key] = ctx
        cache.move_to_end(key)
        while len(cache) > _MAX_FOLD_CTX_ENTRIES:
            cache.popitem(last=False)

    def _speculative_build(
        self,
        x: torch.Tensor,
        draft_logits_region: torch.Tensor,
        region_start: int,
        masked_local: torch.Tensor,
        draft_branch_id: str,
        unit_idx: int,
        sub_idx: int,
        stop_token_id: int,
        eos_aware: bool | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        """Segment A (AR025 N7): build the ``[N, T]`` candidate canvas.

        Pure construction — no chain-state mutation, no forward: each
        still-masked unit position is filled from the draft's top-k token
        set (row 0 = the greedy hypothesis when ``candidate_include_greedy``
        is on; random rows via the frozen per-candidate seed formula on a
        dedicated CPU generator; row ``N-1`` = the EOS-aware continuation
        row when ``eos_aware`` resolves on). The joint ``sample_joint``
        loops call this per chain with the SAME ``unit_idx``/``sub_idx``
        (the seed formula has no chain component) and own the per-chain
        parents externally.

        Args:
            x: ``[1, T]`` canvas (read-only here — the commit lives in
                segment B).
            draft_logits_region: ``[1, R, V]`` family-corrected draft logits
                over the unit region (``R`` = region length).
            region_start: Canvas start coordinate of the unit region.
            masked_local: ``[M]`` still-masked positions inside the region
                (local coordinates); must be non-empty (the family adapter
                skips empty units before calling — silent zero-fill
                candidates would be a dangerous no-op).
            draft_branch_id: The draft serial forward's branch id (names
                the candidate rows ``f"{draft_branch_id}_cand{j}"``).
            unit_idx: Unit index for the frozen seed formula.
            sub_idx: Sub index for the frozen seed formula.
            stop_token_id: The family stop token (the EOS-aware
                continuation row's exclusion).
            eos_aware: AR024 FR3 override; ``None`` resolves through
                ``getattr(cfg, "candidate_eos_aware", False)`` (verbatim
                the composed resolution).

        Returns:
            ``(cands, cand_branch_ids)`` — the ``[N, T]`` candidate canvas
            and the per-row candidate branch ids.

        Raises:
            ValueError: If the canvas batch is not 1 (the frozen #41
                structural limit message) or ``masked_local`` is empty.
        """
        if x.shape[0] != 1:
            raise ValueError(
                "SpeculativeCore._speculative_commit supports batch 1 only; "
                f"the family fold-context mask cannot express per-row padding "
                f"(got batch {x.shape[0]})."
            )
        masked_local = masked_local.reshape(-1)
        num_masked = int(masked_local.shape[0])
        if num_masked == 0:
            raise ValueError(
                "SpeculativeCore._speculative_build requires a non-empty "
                "masked_local (the family adapter skips empty units before "
                "calling)."
            )

        cfg = self.config
        num_candidates = int(cfg.num_block_candidates)
        vocab = int(draft_logits_region.shape[-1])
        k_eff = max(1, min(int(cfg.candidate_top_k), vocab))
        topk_idx = torch.topk(draft_logits_region, k_eff, dim=-1).indices  # [1, R, k]
        draft_set = topk_idx[0, masked_local, :]  # [M, k]
        fill_positions = masked_local + region_start  # [M] canvas coordinates

        cands = x.repeat(num_candidates, 1)
        if cfg.candidate_include_greedy:
            # Greedy hypothesis (AR022 FR1): the still-masked positions take
            # the draft argmax = the top-k ordered-set first entry
            # (sorted=True topk). The index tensor stays on-device, so the
            # construction is readback-free and deterministic.
            cands[0, fill_positions] = draft_set[:, 0]
        eos_on = (
            bool(eos_aware)
            if eos_aware is not None
            else bool(getattr(cfg, "candidate_eos_aware", False))
        )
        first_random = 1 if cfg.candidate_include_greedy else 0
        last_random = num_candidates - 1 if eos_on else num_candidates
        base_seed = int(cfg.seed) if cfg.seed is not None else 0
        for j in range(first_random, last_random):
            seed_j = (
                base_seed * 1_000_003 + int(unit_idx) * 8_191 + int(sub_idx) * 131 + j
            ) & 0x7FFF_FFFF
            generator = torch.Generator(device="cpu").manual_seed(seed_j)
            choice = torch.randint(0, k_eff, (num_masked,), generator=generator).to(x.device)
            cand_tokens = torch.gather(draft_set, 1, choice.unsqueeze(1)).squeeze(1)
            cands[j, fill_positions] = cand_tokens
        if eos_on:
            # AR024 FR3 continuation row: row N-1 takes the first non-EOS
            # draft-pool entry per position — pure device indexing, zero
            # generator consumption, zero device readback. Assigned LAST so
            # the row is unconditionally the continuation (with N=1 the
            # single row plays both roles — the greedy definition refines
            # to the EOS-excluding greedy); an all-EOS pool degenerates to
            # ``pool[:, 0]`` (argmax of zeros is 0).
            pool = draft_set
            non_eos = pool != stop_token_id
            first_non_eos = non_eos.to(torch.uint8).argmax(dim=1)
            cont_tokens = pool.gather(1, first_non_eos.unsqueeze(1)).squeeze(1)
            cands[num_candidates - 1, fill_positions] = cont_tokens

        cand_branch_ids = [f"{draft_branch_id}_cand{j}" for j in range(num_candidates)]
        return cands, cand_branch_ids

    def _speculative_score_commit(
        self,
        x: torch.Tensor,
        finished: torch.Tensor,
        batched_logits: torch.Tensor,
        cands: torch.Tensor,
        cand_branch_ids: list[str],
        draft_logits_region: torch.Tensor,
        region_start: int,
        masked_local: torch.Tensor,
        history: list[torch.Tensor],
        stop_token_id: int,
        refill_top_k: int | None = None,
    ) -> None:
        """Segment B (AR025 N7): score, select, and commit one unit.

        The AR023/AR024 commit flow isolated from the construction segment:
        min-prob selection over the FILL positions (softmax over the
        family-corrected ``batched_logits`` region slice, gathered at the
        candidate tokens), the greedy_win counters, the ``_folding_parent``
        chain-continuation write (the host reads it back for per-chain
        state on the joint path), and the full / partial
        (``refill_top_k``) commit. The joint ``sample_joint`` loops verify
        all chains in ONE batched call and then run this segment per chain
        on the row-sliced logits.

        Args:
            x: ``[1, T]`` canvas, mutated in place at the fill positions.
            finished: ``[1]`` (batch-1) bool finished flags, mutated in
                place.
            batched_logits: ``[N, T, V]`` family-corrected verification
                logits for THIS chain's candidates (the joint loop slices
                its per-chain block out of the joint call's output).
            cands: The ``[N, T]`` candidate canvas (segment A's output).
            cand_branch_ids: The per-row candidate branch ids (segment A's
                output).
            draft_logits_region: ``[1, R, V]`` family-corrected draft
                logits over the unit region (the refill branch's
                draft-side confidence source).
            region_start: Canvas start coordinate of the unit region.
            masked_local: ``[M]`` still-masked positions inside the region
                (local coordinates, non-empty).
            history: History list; gains one post-commit snapshot when the
                host config's ``return_history`` is on (full-commit mode
                only — the AR024 refill contract).
            stop_token_id: The family stop token (finished semantics).
            refill_top_k: AR024 FR2 partial commit (verbatim the composed
                semantics; ``None`` = the full FR1 commit).
        """
        masked_local = masked_local.reshape(-1)
        num_masked = int(masked_local.shape[0])
        region_len = int(draft_logits_region.shape[1])
        fill_positions = masked_local + region_start  # [M] canvas coordinates

        region_logits = batched_logits[:, region_start : region_start + region_len, :]
        probs = F.softmax(region_logits, dim=-1)
        committed = cands[:, region_start : region_start + region_len]
        scores = torch.gather(probs, -1, committed.unsqueeze(-1)).squeeze(-1)  # [N, R]
        fill_scores = scores[:, masked_local]  # [N, M] — the FILL position domain
        min_scores = fill_scores.min(dim=-1).values
        j_star = int(min_scores.argmax())

        if self.config.candidate_include_greedy:
            self._greedy_spec_calls += 1
            if j_star == 0:
                self._greedy_wins += 1
        self._folding_parent = cand_branch_ids[j_star]
        if refill_top_k is None:
            x[:, fill_positions] = cands[j_star, fill_positions]
            finished |= (x[:, fill_positions] == stop_token_id).any(dim=1)
            if self.config.return_history:
                history.append(x.clone())
            return
        k_eff = min(int(refill_top_k), num_masked)
        if k_eff >= num_masked:
            # Runtime clamp (k >= the unit's masked count): the full commit
            # degenerates to the FR1 semantics.
            x[:, fill_positions] = cands[j_star, fill_positions]
            finished |= (x[:, fill_positions] == stop_token_id).any(dim=1)
            return
        # AR024 FR2 partial commit: the top-k fill positions by DRAFT-side
        # confidence of the winner row's tokens commit; the uncommitted
        # winner tokens are discarded (the family's phase two refills the
        # remainder on the committed context). ``finished`` checks ONLY the
        # committed positions and the Core never appends history in refill
        # mode — the family owns the per-window snapshot (design §4.1.4).
        draft_probs = F.softmax(draft_logits_region, dim=-1)
        winner_tokens = cands[j_star, fill_positions]
        w_conf = draft_probs[0, masked_local, :].gather(-1, winner_tokens.unsqueeze(1)).squeeze(1)
        top_local = torch.topk(w_conf, k_eff).indices
        commit_pos = fill_positions[top_local]
        x[0, commit_pos] = winner_tokens[top_local]
        finished |= (x[:, commit_pos] == stop_token_id).any(dim=1)

    def _speculative_commit(
        self,
        x: torch.Tensor,
        finished: torch.Tensor,
        draft_logits_region: torch.Tensor,
        region_start: int,
        masked_local: torch.Tensor,
        draft_branch_id: str,
        ctx: SpecCtx,
        unit_forced: torch.Tensor,
        folded_model: Any,
        unit_idx: int,
        sub_idx: int,
        history: list[torch.Tensor],
        forward_fn: Callable[..., torch.Tensor],
        stop_token_id: int,
        eos_aware: bool | None = None,
        refill_top_k: int | None = None,
    ) -> None:
        """Run one speculative unit commit (mutates ``x`` / ``finished``).

        Builds ``N`` candidate continuations of the canvas by filling each
        still-masked unit position from the draft's top-k token set (row 0 =
        the greedy hypothesis when ``candidate_include_greedy`` is on; random
        rows via the frozen per-candidate seed formula on a dedicated CPU
        generator), verifies all candidates in ONE ``[N, T]`` batched call
        sharing the draft parent, and commits the row whose per-position
        minimum token probability over the FILL positions (softmax over the
        family-corrected batched logits, gathered at the candidate tokens) is
        maximal. The folding-chain parent switches to the selected candidate
        branch; the stop-token check matches the family's ``finished``
        semantics through ``stop_token_id``.

        The family adapter guarantees ``draft_logits_region`` already carries
        the family's position-prediction convention (LLaDA: position i
        predicts i — no shift; Dream: right-shifted) — the core slices the
        batched logits directly and never shifts.

        Args:
            x: ``[1, T]`` canvas, mutated in place at the fill positions.
            finished: ``[1]`` (batch-1) bool finished flags, mutated in place.
            draft_logits_region: ``[1, R, V]`` family-corrected draft logits
                over the unit region (``R`` = region length).
            region_start: Canvas start coordinate of the unit region.
            masked_local: ``[M]`` still-masked positions inside the region
                (local coordinates). An empty ``masked_local`` is a no-op
                (defensive sentinel — the family adapter should skip such
                units before calling).
            draft_branch_id: The draft serial forward's branch id (the
                batched candidates share it as their parent).
            ctx: The family fold context (pinned by the adapter; the
                ``forward_fn`` closure already binds it — carried for the
                frozen signature).
            unit_forced: ``[T]`` bool forced-divergence tensor for this unit.
            folded_model: The folded model (passed through to the family
                ``forward_fn`` closure's expectations; the core itself only
                requires the closure to accept the batched kwargs).
            unit_idx: Unit index for the frozen seed formula.
            sub_idx: Sub index for the frozen seed formula.
            history: History list; gains one post-commit snapshot when the
                host config's ``return_history`` is on.
            forward_fn: Family closure over ``self._forward`` (binds the
                family ctx / step); called as
                ``forward_fn(cands, branch_id=None, parent_branch_id=
                draft_branch_id, branch_ids=[f"{draft_branch_id}_cand{j}"],
                forced_divergent=unit_forced.repeat(N, 1))`` and returning
                ``[N, T, V]`` batched logits.
            stop_token_id: The family stop token (finished semantics).
            eos_aware: AR024 FR3 EOS-aware continuation row. ``None`` (the
                default) resolves through
                ``getattr(cfg, "candidate_eos_aware", False)`` — an
                attribute-less host (the Dream face) stays eos-off and the
                no-new-param call is byte-identical to the explicit-off
                call; the explicit parameter overrides the cfg. When on,
                candidate row N-1 is the continuation row (the first
                non-EOS draft-pool entry per position, pure device
                indexing) assigned LAST — the random rows yield the
                ``j = N-1`` draw and rows ``1..N-2`` are bit-identical to
                the eos-off build; an EOS-absent pool degenerates to the
                greedy value.
            refill_top_k: AR024 FR2 partial commit. ``None`` (the default)
                is the full FR1 commit. An int commits only the top-k fill
                positions by draft-side confidence of the winner row's
                tokens (a runtime clamp to the masked count degenerates to
                the FR1 full commit); ``finished`` checks ONLY the
                committed positions, the Core never appends history in
                refill mode (the family owns the per-window snapshot), and
                the uncommitted winner tokens are discarded.

        Raises:
            ValueError: If the canvas batch is not 1 (the ``[T, T]``
                family-mask semantics cannot express per-row padding — #41
                structural limit).
        """
        if x.shape[0] != 1:
            raise ValueError(
                "SpeculativeCore._speculative_commit supports batch 1 only; "
                f"the family fold-context mask cannot express per-row padding "
                f"(got batch {x.shape[0]})."
            )
        masked_local = masked_local.reshape(-1)
        num_masked = int(masked_local.shape[0])
        if num_masked == 0:
            return

        # AR025 N7: the frozen composed flow — segment A (construction),
        # the family ``forward_fn`` batched verification call, segment B
        # (score/select/commit) — byte-identical to the AR024 monolith;
        # the joint ``sample_joint`` loops call the two segments directly
        # around ONE multi-chain batched call instead.
        cands, cand_branch_ids = self._speculative_build(
            x=x,
            draft_logits_region=draft_logits_region,
            region_start=region_start,
            masked_local=masked_local,
            draft_branch_id=draft_branch_id,
            unit_idx=unit_idx,
            sub_idx=sub_idx,
            stop_token_id=stop_token_id,
            eos_aware=eos_aware,
        )
        num_candidates = int(self.config.num_block_candidates)
        batched_logits = forward_fn(
            cands,
            branch_id=None,
            parent_branch_id=draft_branch_id,
            branch_ids=cand_branch_ids,
            forced_divergent=unit_forced.unsqueeze(0).repeat(num_candidates, 1),
        )
        self._speculative_score_commit(
            x=x,
            finished=finished,
            batched_logits=batched_logits,
            cands=cands,
            cand_branch_ids=cand_branch_ids,
            draft_logits_region=draft_logits_region,
            region_start=region_start,
            masked_local=masked_local,
            history=history,
            stop_token_id=stop_token_id,
            refill_top_k=refill_top_k,
        )
