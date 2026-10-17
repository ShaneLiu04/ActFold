"""Adaptive divergent-capacity policy for the batched CUDA graph path (AR017).

The AR016 batched graph runner fixes each layer's divergent capacity at capture
time (``C = ceil(graph_capacity_ratio * N * T)``). Measured workloads stay in a
narrow mid-divergence band (~13-16%), so a fixed ``C = 0.5 * N * T`` over-pays
FFN recompute 3.2-3.8x, while a fixed low ``C`` discards replays on every burst
(eager fallback, zero graph benefit). This module owns the decision layer of
the AR017 fix: a small observation-driven state machine that watches the
per-step divergence demand and emits ``recapture`` actions -- the caller
(:meth:`ManualFoldedForward._recapture_batched_runner`) re-captures the graph
at the new capacity.

Contract (design.md SS2, pinned by tests/test_adaptive_capacity.py UT-1601):

* ``observe`` is called once per eligible batched step with the step's maximum
  per-layer divergence, the budget verdict, the live capacity and the token
  total. Growth fires immediately on a violation at
  ``ceil(grow_factor * maxD)`` (clamped to ``num_tokens``); shrink fires after
  ``shrink_patience`` consecutive violation-free observations whose maximum
  stays at or below ``shrink_threshold * C``, targeting
  ``clamp(ceil(grow_factor * maxD), 1, C - 1)``.
* Re-captures are throttled by ``min_recapture_interval`` steps
  (increment-then-check: the counter initializes to the interval, so the very
  first observe is never throttled; ``note_recaptured`` resets it) and capped
  by ``max_recaptures`` *successful* re-captures, after which the policy
  retires -- the retiring observe appends its observation first, returns
  ``kind="retired"`` and latches; later observes are permanent no-ops with a
  frozen window.
* ``events`` records only outcomes: one recapture entry per successful
  ``note_recaptured`` (never a failed or throttled attempt) plus one
  ``"retired"`` entry at the cap. A failed re-capture is the caller's to
  handle via ``retire()`` (no event -- the failure warning lives at the call
  site).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "ActionKind",
    "AdaptiveAction",
    "AdaptiveCapacityConfig",
    "AdaptiveCapacityPolicy",
]

ActionKind = Literal["noop", "recapture", "retired"]


@dataclass(frozen=True)
class AdaptiveCapacityConfig:
    """Hyper-parameters of the adaptive capacity policy.

    Attributes:
        grow_factor: Multiplier applied to the observed divergence when
            growing (``>= 1.0``; the headroom absorbs the next burst).
        shrink_threshold: Fraction of the live capacity that the recent
            divergence maximum must not exceed for a shrink (in ``(0, 1)``).
        shrink_patience: Consecutive non-violating observations required
            before a shrink fires (``>= 1``).
        min_recapture_interval: Minimum steps between two re-captures
            (``>= 1``); the counter starts pre-loaded, so the first observe is
            never throttled.
        max_recaptures: Successful re-captures before the policy retires
            (``>= 1``); retirement freezes the capacity for good.
    """

    grow_factor: float = 1.25
    shrink_threshold: float = 0.5
    shrink_patience: int = 3
    min_recapture_interval: int = 2
    max_recaptures: int = 8

    def __post_init__(self) -> None:
        if self.grow_factor < 1.0:
            raise ValueError(f"grow_factor must be >= 1.0, got {self.grow_factor}")
        if not 0.0 < self.shrink_threshold < 1.0:
            raise ValueError("shrink_threshold must be in (0, 1), got " f"{self.shrink_threshold}")
        if self.shrink_patience < 1:
            raise ValueError(f"shrink_patience must be >= 1, got {self.shrink_patience}")
        if self.min_recapture_interval < 1:
            raise ValueError(
                "min_recapture_interval must be >= 1, got " f"{self.min_recapture_interval}"
            )
        if self.max_recaptures < 1:
            raise ValueError(f"max_recaptures must be >= 1, got {self.max_recaptures}")


@dataclass(frozen=True)
class AdaptiveAction:
    """One policy verdict.

    Attributes:
        kind: ``"noop"`` (keep the current capacity), ``"recapture"`` (rebuild
            the graph at ``target_capacity``) or ``"retired"`` (the cap was
            reached; the caller warns once and keeps the current capacity).
        target_capacity: The new capacity for ``"recapture"``; ``None``
            otherwise.
    """

    kind: ActionKind
    target_capacity: int | None = None


@dataclass
class AdaptiveCapacityPolicy:
    """Observation-driven state machine deciding when to re-capture.

    The policy is pure bookkeeping: it never touches tensors or the runner.
    The caller owns the observation stream (one ``observe`` per eligible
    batched step), performs the re-capture, and confirms success with
    ``note_recaptured`` (failure with ``retire``).

    Attributes:
        config: The validated :class:`AdaptiveCapacityConfig`.
        events: Append-only outcome log; see the module docstring for the
            entry shapes.
    """

    config: AdaptiveCapacityConfig
    events: list[dict[str, Any]] = field(default_factory=list)

    _window: list[tuple[int, bool]] = field(default_factory=list, init=False)
    _observe_count: int = field(default=0, init=False)
    _steps_since_recapture: int = field(default=0, init=False)
    _recapture_count: int = field(default=0, init=False)
    _retired: bool = field(default=False, init=False)
    # (trigger, target_capacity, from_capacity, observe_index) of the pending
    # recapture verdict; consumed (and cleared) by the next note_recaptured.
    _pending: tuple[str, int, int, int] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Increment-then-check throttle semantics: the counter starts
        # pre-loaded so the first observe is never throttled (UT-1601).
        self._steps_since_recapture = self.config.min_recapture_interval

    def observe(
        self,
        max_layer_divergence: int,
        exceeded: bool,
        current_capacity: int,
        num_tokens: int,
    ) -> AdaptiveAction:
        """Fold one observation into the window and return the verdict.

        Args:
            max_layer_divergence: The step's maximum per-layer divergent
                count (``max(runner.last_layer_divergences)``).
            exceeded: Whether the step violated the divergent budget.
            current_capacity: The live per-layer capacity of the runner.
            num_tokens: The captured token total (``N * T``).

        Returns:
            The :class:`AdaptiveAction` verdict for this step.
        """
        if self._retired:
            return AdaptiveAction("noop")
        self._window.append((max_layer_divergence, exceeded))
        if len(self._window) > self.config.shrink_patience:
            del self._window[0]
        self._observe_count += 1
        self._steps_since_recapture += 1
        self._pending = None
        if self._recapture_count >= self.config.max_recaptures:
            self._retired = True
            self.events.append({"observe": self._observe_count, "kind": "retired"})
            return AdaptiveAction("retired")
        if exceeded:
            target = min(
                num_tokens,
                math.ceil(self.config.grow_factor * max_layer_divergence),
            )
            if target <= current_capacity:
                return AdaptiveAction("noop")
            if self._steps_since_recapture < self.config.min_recapture_interval:
                return AdaptiveAction("noop")
            self._pending = (
                "grow",
                target,
                current_capacity,
                self._observe_count,
            )
            return AdaptiveAction("recapture", target_capacity=target)
        if current_capacity <= 1:
            return AdaptiveAction("noop")
        recent = self._window[-self.config.shrink_patience :]
        if len(recent) == self.config.shrink_patience and all(
            not item_exceeded for _, item_exceeded in recent
        ):
            recent_max = max(item_d for item_d, _ in recent)
            if recent_max <= self.config.shrink_threshold * current_capacity:
                target = max(
                    1,
                    min(
                        current_capacity - 1,
                        math.ceil(self.config.grow_factor * recent_max),
                    ),
                )
                if self._steps_since_recapture < self.config.min_recapture_interval:
                    return AdaptiveAction("noop")
                self._pending = (
                    "shrink",
                    target,
                    current_capacity,
                    self._observe_count,
                )
                return AdaptiveAction("recapture", target_capacity=target)
        return AdaptiveAction("noop")

    def note_recaptured(self) -> None:
        """Confirm the pending re-capture succeeded and reset the throttle.

        Records the pending verdict as a recapture event (a recapture
        without a preceding ``"recapture"`` verdict records nothing), clears
        the observation window (pre-stretch demand must not re-trigger), and
        resets the step counter so the next re-capture waits a full
        interval.
        """
        if self._pending is not None:
            trigger, target, from_capacity, observe_index = self._pending
            self.events.append(
                {
                    "observe": observe_index,
                    "kind": "recapture",
                    "trigger": trigger,
                    "target": target,
                    "from": from_capacity,
                }
            )
            self._pending = None
        self._window.clear()
        self._steps_since_recapture = 0
        self._recapture_count += 1

    def retire(self) -> None:
        """Retire the policy manually (the re-capture-failure path).

        Latches the same frozen state as the ``max_recaptures`` cap but
        records no event: the failure itself is reported by the caller's
        warning, and no re-capture ever succeeded.
        """
        self._retired = True
