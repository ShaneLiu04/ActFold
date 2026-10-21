"""AR031 uncertainty-aware candidate validation over paged rollouts.

``UncertaintyRolloutValidator`` ranks the candidates of an open
:class:`~actfold.core.paged_fold_cache.PagedFoldCache` rollout by a
caller-supplied PRIOR (descending, id-ascending tie-break), validates them
one by one through a caller-supplied posterior scorer, and stops early when
the running best posterior satisfies BOTH the absolute margin and the
prior-gap margin (design A5 + erratum E2):

* the winner is the candidate with the highest posterior — ties keep the
  first-validated candidate (the highest prior / smallest id); a NaN
  posterior never wins, and an all-NaN batch keeps ``order[0]``;
* on return every LOSING candidate is closed via
  :meth:`~actfold.core.paged_fold_cache.PagedFoldCache.close_candidate`
  (its fork references and exclusive pages are released) while the winner
  branch, the open rollout, and the frozen base stay untouched —
  ``close_rollout`` remains the caller's duty (E2);
* a ``validate_fn`` failure rolls the FAILING candidate back to its fork
  baseline (:meth:`~actfold.core.paged_fold_cache.PagedFoldCache.rollback_candidate`)
  and re-raises the original exception unchanged — the failed call counts
  nowhere in the stats and no candidate is closed (AR030 atomic semantics);
* skipped candidates are zero-write: the validator neither writes nor
  evicts anything for them (their fork state stays byte-identical to the
  fork baseline, UT-3123).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from actfold.core.paged_fold_cache import PagedFoldCache, RolloutSet

__all__ = ["UncertaintyRolloutValidator"]

_WINNER_SCORES_CAP = 100


class UncertaintyRolloutValidator:
    """Prior-ordered, margin-early-stopping rollout candidate validator.

    Args:
        cache: The paged fold cache owning the rollouts under validation.
        margin: Absolute early-stop threshold on the best posterior
            (``q_best >= margin``); must lie in ``[0, 1]``.
        margin_gap: Prior-gap early-stop threshold (``q_best - max prior of
            the remaining candidates >= margin_gap``); must lie in ``[0, 1]``.
            Both conditions must hold for the early stop to fire.
    """

    def __init__(
        self,
        cache: PagedFoldCache,
        margin: float = 0.9,
        margin_gap: float = 0.05,
    ) -> None:
        if not 0.0 <= margin <= 1.0:
            raise ValueError(f"margin must lie in [0, 1], got {margin}")
        if not 0.0 <= margin_gap <= 1.0:
            raise ValueError(f"margin_gap must lie in [0, 1], got {margin_gap}")
        self._cache = cache
        self._margin = float(margin)
        self._margin_gap = float(margin_gap)
        self._validated_count = 0
        self._skipped_count = 0
        self._early_stops = 0
        self._winner_scores: list[float] = []

    @property
    def stats(self) -> dict[str, Any]:
        """Read-only cumulative counters (survive across ``validate`` calls).

        Keys: ``validated_count`` / ``skipped_count`` / ``early_stops`` /
        ``winner_scores`` (the per-call best posterior, FIFO-capped at 100).
        """
        return {
            "validated_count": self._validated_count,
            "skipped_count": self._skipped_count,
            "early_stops": self._early_stops,
            "winner_scores": list(self._winner_scores),
        }

    def validate(
        self,
        rollout: RolloutSet,
        candidate_ids: Sequence[str],
        prior_scores: Mapping[str, Any],
        validate_fn: Callable[[str], float],
    ) -> str:
        """Validate rollout candidates in prior order; return the winner id.

        Args:
            rollout: An OPEN rollout owned by this validator's cache.
            candidate_ids: The candidate subset to consider (non-empty, no
                duplicates, every id an ACTIVE candidate of ``rollout``).
            prior_scores: One prior per candidate id (numeric; missing or
                non-numeric values raise ``ValueError`` before any
                validation work).
            validate_fn: Posterior scorer invoked as ``validate_fn(cid)``;
                its return value is coerced with ``float``. Any exception it
                raises rolls the failing candidate back to its fork baseline
                and propagates unchanged.

        Returns:
            The winning candidate id (highest posterior; ties keep the
            first-validated candidate; NaN never wins; all-NaN keeps
            ``order[0]``).

        Raises:
            ValueError: On an empty/duplicate/unknown/inactive candidate
                list, a missing or non-numeric prior, or a rollout that is
                closed or belongs to a different cache — always with zero
                state mutation (``validate_fn`` is never invoked).
        """
        ids = list(candidate_ids)
        if not ids:
            raise ValueError("candidate_ids must be non-empty")
        if len(set(ids)) != len(ids):
            raise ValueError(f"candidate_ids contains duplicates: {ids}")
        # Ownership probe (the PUBLIC face, never the private rollout dict):
        # a closed or foreign rollout raises through ``rollout_stats``, and
        # only ACTIVE candidates appear in the returned map.
        try:
            active = set(self._cache.rollout_stats(rollout))
        except ValueError as exc:
            raise ValueError(f"rollout is not an open rollout of this cache: {exc}") from exc
        inactive = [cid for cid in ids if cid not in active]
        if inactive:
            raise ValueError(f"candidates are not active in this rollout: {inactive}")
        priors: dict[str, float] = {}
        for cid in ids:
            if cid not in prior_scores:
                raise ValueError(f"prior score missing for candidate {cid!r}")
            try:
                priors[cid] = float(prior_scores[cid])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"prior score for {cid!r} is not numeric: " f"{prior_scores[cid]!r}"
                ) from exc

        order = sorted(ids, key=lambda cid: (-priors[cid], cid))
        q_best = float("-inf")
        winner = order[0]
        validated: list[str] = []
        stopped_early = False
        try:
            for idx, cid in enumerate(order):
                posterior = float(validate_fn(cid))
                validated.append(cid)
                # Strict ``>``: a tie keeps the first-validated candidate
                # (highest prior, smallest id) and a NaN posterior never
                # compares True — the all-NaN case keeps ``order[0]``.
                if posterior > q_best:
                    q_best = posterior
                    winner = cid
                remaining = order[idx + 1 :]
                if remaining and q_best >= self._margin:
                    prior_gap = q_best - max(priors[c] for c in remaining)
                    if prior_gap >= self._margin_gap:
                        stopped_early = True
                        break
        except Exception:
            # Atomic failure face (AR030 semantics): the FAILING candidate
            # (not yet in ``validated``) returns to its fork baseline; the
            # already-validated candidates keep their committed writes; the
            # stats count this call nowhere; no candidate is closed.
            self._cache.rollback_candidate(rollout, order[len(validated)])
            raise

        # E2 close-out: every loser is released, the winner and the open
        # rollout survive, and ``close_rollout`` stays the caller's duty.
        for cid in order:
            if cid != winner:
                self._cache.close_candidate(rollout, cid)

        validated_set = set(validated)
        skipped = [cid for cid in order if cid not in validated_set]
        self._validated_count += len(validated)
        self._skipped_count += len(skipped)
        if stopped_early:
            self._early_stops += 1
        self._winner_scores.append(q_best)
        if len(self._winner_scores) > _WINNER_SCORES_CAP:
            self._winner_scores = self._winner_scores[-_WINNER_SCORES_CAP:]
        return winner
