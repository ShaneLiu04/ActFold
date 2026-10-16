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

        cfg = self.config
        num_candidates = int(cfg.num_block_candidates)
        vocab = int(draft_logits_region.shape[-1])
        k_eff = max(1, min(int(cfg.candidate_top_k), vocab))
        region_len = int(draft_logits_region.shape[1])
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
        base_seed = int(cfg.seed) if cfg.seed is not None else 0
        for j in range(1 if cfg.candidate_include_greedy else 0, num_candidates):
            seed_j = (
                base_seed * 1_000_003 + int(unit_idx) * 8_191 + int(sub_idx) * 131 + j
            ) & 0x7FFF_FFFF
            generator = torch.Generator(device="cpu").manual_seed(seed_j)
            choice = torch.randint(0, k_eff, (num_masked,), generator=generator).to(x.device)
            cand_tokens = torch.gather(draft_set, 1, choice.unsqueeze(1)).squeeze(1)
            cands[j, fill_positions] = cand_tokens

        cand_branch_ids = [f"{draft_branch_id}_cand{j}" for j in range(num_candidates)]
        batched_logits = forward_fn(
            cands,
            branch_id=None,
            parent_branch_id=draft_branch_id,
            branch_ids=cand_branch_ids,
            forced_divergent=unit_forced.unsqueeze(0).repeat(num_candidates, 1),
        )
        region_logits = batched_logits[:, region_start : region_start + region_len, :]
        probs = F.softmax(region_logits, dim=-1)
        committed = cands[:, region_start : region_start + region_len]
        scores = torch.gather(probs, -1, committed.unsqueeze(-1)).squeeze(-1)  # [N, R]
        fill_scores = scores[:, masked_local]  # [N, M] — the FILL position domain
        min_scores = fill_scores.min(dim=-1).values
        j_star = int(min_scores.argmax())

        if cfg.candidate_include_greedy:
            self._greedy_spec_calls += 1
            if j_star == 0:
                self._greedy_wins += 1
        x[:, fill_positions] = cands[j_star, fill_positions]
        finished |= (x[:, fill_positions] == stop_token_id).any(dim=1)
        self._folding_parent = cand_branch_ids[j_star]
        if cfg.return_history:
            history.append(x.clone())
