"""AR031 layer-adaptive recompute budget: windowed policy + exact-count cap.

``LayerRecomputeBudgetPolicy`` splits a global recompute budget across layers
(design A2: domain rule with floor+largest-remainder, surplus redistribution
and floor-lift reconciliation) and accumulates per-layer stability counts with
one batched readback per re-estimate window (design A3).  ``apply_layer_budget``
is the exact-count cap (design A1): exempt rows (stable / forced / NaN scores)
never fold and never consume the budget; among the remaining rows the most
similar ones are forced stable until the non-exempt recompute count fits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

_EPS = 1e-6


class LayerRecomputeBudgetPolicy:
    """Windowed per-layer recompute budget with frozen allocation semantics."""

    def __init__(
        self,
        budget_ratio: float,
        num_layers: int,
        warmup_steps: int = 5,
        window_steps: int = 10,
        adaptive: bool = True,
    ) -> None:
        if not 0.0 < budget_ratio <= 1.0:
            raise ValueError(f"budget_ratio must lie in (0, 1], got {budget_ratio}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")
        if window_steps < 1:
            raise ValueError(f"window_steps must be >= 1, got {window_steps}")
        self._budget_ratio = float(budget_ratio)
        self._num_layers = int(num_layers)
        self._warmup_steps = int(warmup_steps)
        self._window_steps = int(window_steps)
        self._adaptive = bool(adaptive)
        self._step_count = 0
        self._weights: tuple[float, ...] | None = None
        self._window_events: list[dict[str, object]] = []
        self._acc_stable: dict[int, Tensor] = {}
        self._acc_tokens: list[int] = [0] * self._num_layers
        self._cache_tokens: tuple[int, ...] | None = None
        self._cache_table: tuple[int, ...] | None = None

    @property
    def weights(self) -> tuple[float, ...] | None:
        """Current window weights (``None`` during warmup = uniform allocation)."""
        return self._weights

    @property
    def num_layers(self) -> int:
        """Configured depth (the MFF wiring validates against its own layer count)."""
        return self._num_layers

    @property
    def window_events(self) -> list[dict[str, object]]:
        """Read-only snapshot of the re-estimate event log."""
        return list(self._window_events)

    def begin_step(self) -> None:
        """Advance the step counter and re-estimate at window boundaries."""
        self._step_count += 1
        at_warmup = self._step_count == self._warmup_steps
        at_window = self._step_count > self._warmup_steps and (
            (self._step_count - self._warmup_steps) % self._window_steps == 0
        )
        if at_warmup or at_window:
            self._reestimate()
            # Weights changed: the cached ``budget_table`` result is stale.
            self._cache_tokens = None
            self._cache_table = None

    def observe_layer(self, layer_idx: int, stable_count: Tensor, num_tokens: int) -> None:
        """Accumulate one layer's stability count (device-side add, zero sync).

        Raises:
            ValueError: If ``layer_idx`` is out of range, or the observation
                lands on a different device than this layer's accumulator —
                cross-device observations are rejected, never silently moved
                (UT-3152).
        """
        if not 0 <= layer_idx < self._num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self._num_layers})")
        acc = self._acc_stable.get(layer_idx)
        if acc is None:
            acc = torch.zeros((), dtype=torch.float32, device=stable_count.device)
            self._acc_stable[layer_idx] = acc
        elif acc.device != stable_count.device:
            raise ValueError(
                f"observe_layer device mismatch for layer {layer_idx}: accumulator "
                f"on {acc.device} but observation on {stable_count.device}"
            )
        acc += stable_count.to(dtype=torch.float32)
        self._acc_tokens[layer_idx] += int(num_tokens)

    def budget_table(self, tokens: Sequence[int]) -> list[int]:
        """Full per-layer budget table for one forward call (design A2)."""
        tokens_t = tuple(int(t) for t in tokens)
        if len(tokens_t) != self._num_layers:
            raise ValueError(f"tokens length {len(tokens_t)} != num_layers {self._num_layers}")
        table = self._allocate(tokens_t)
        self._cache_tokens = tokens_t
        self._cache_table = tuple(table)
        return table

    def budget_for(self, layer_idx: int, tokens_l: int) -> int:
        """O(1) lookup into the last ``budget_table`` result (uniform fallback)."""
        if not 0 <= layer_idx < self._num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self._num_layers})")
        if (
            self._cache_table is not None
            and self._cache_tokens is not None
            and self._cache_tokens[layer_idx] == tokens_l
        ):
            return self._cache_table[layer_idx]
        return self._allocate((tokens_l,) * self._num_layers)[layer_idx]

    def apply_layer_budget(
        self,
        stable_mask: Tensor,
        scores: Tensor,
        budget: Tensor | int,
        forced: Tensor | None = None,
    ) -> Tensor:
        """Policy-face wrapper around :func:`apply_layer_budget`."""
        return apply_layer_budget(stable_mask, scores, budget, forced)

    def _reestimate(self) -> None:
        num_layers = self._num_layers
        observed = [layer for layer in range(num_layers) if self._acc_tokens[layer] > 0]
        if self._adaptive:
            if observed:
                stacked = torch.stack([self._acc_stable[layer] for layer in observed])
                stable_values = stacked.to(dtype=torch.float32).tolist()
            else:
                stable_values = []
            stable_map = {
                layer: float(value) / max(self._acc_tokens[layer], 1)
                for layer, value in zip(observed, stable_values)
            }
            utilities: list[float] = []
            for layer in range(num_layers):
                if layer in stable_map:
                    utilities.append(1.0 - float(stable_map[layer]) + _EPS)
                else:
                    utilities.append(1.0 / num_layers + _EPS)
            total = sum(utilities)
            weights = tuple(utility / total for utility in utilities)
        else:
            weights = tuple(1.0 / num_layers for _ in range(num_layers))
        self._weights = weights
        self._acc_stable = {}
        self._acc_tokens = [0] * num_layers
        self._window_events.append(
            {
                "step": self._step_count,
                "weights": list(weights),
                "adaptive": self._adaptive,
            }
        )

    def _allocate(self, tokens: tuple[int, ...]) -> list[int]:
        num_layers = self._num_layers
        active = [layer for layer in range(num_layers) if tokens[layer] > 0]
        table = [0] * num_layers
        if not active:
            return table
        b_total = round(self._budget_ratio * sum(tokens))
        if b_total < len(active):
            for layer in active:
                table[layer] = 1
            return table
        if self._weights is None:
            weights = [1.0 / num_layers] * num_layers
        else:
            weights = list(self._weights)
        weight_sum = sum(weights[layer] for layer in active)
        share = {layer: weights[layer] / weight_sum for layer in active}
        order = sorted(active, key=lambda layer: (-share[layer], layer))
        base = {layer: math.floor(b_total * share[layer]) for layer in active}
        remainder = b_total - sum(base.values())
        for i in range(max(0, remainder)):
            base[order[i % len(order)]] += 1
        surplus = 0
        for layer in active:
            if base[layer] > tokens[layer]:
                surplus += base[layer] - tokens[layer]
                base[layer] = tokens[layer]
        cursor = 0
        while surplus > 0:
            layer = order[cursor % len(order)]
            if base[layer] < tokens[layer]:
                base[layer] += 1
                surplus -= 1
            cursor += 1
        while True:
            zero_layers = [layer for layer in active if base[layer] == 0]
            if not zero_layers:
                break
            target = zero_layers[0]
            donors = [layer for layer in active if base[layer] >= 2]
            donor = max(donors, key=lambda layer: (base[layer], layer))
            base[target] += 1
            base[donor] -= 1
        for layer in active:
            table[layer] = base[layer]
        return table


def apply_layer_budget(
    stable_mask: Tensor,
    scores: Tensor,
    budget: Tensor | int,
    forced: Tensor | None = None,
) -> Tensor:
    """Exact-count budget cap (design A1, frozen).

    Exempt rows (already stable, forced-divergent, or NaN-scored) always stay
    recomputed and never consume the budget; among the non-exempt divergent
    rows the highest-scoring ones flip to stable until the non-exempt recompute
    count fits ``budget``.  Empty inputs return the original tensor unchanged.
    """
    if stable_mask.shape != scores.shape:
        raise ValueError(
            f"stable_mask shape {tuple(stable_mask.shape)} != scores shape"
            f" {tuple(scores.shape)}"
        )
    if forced is not None and forced.shape != stable_mask.shape:
        raise ValueError(
            f"forced shape {tuple(forced.shape)} != stable_mask shape"
            f" {tuple(stable_mask.shape)}"
        )
    n = stable_mask.numel()
    if n == 0:
        return stable_mask
    forced_mask = torch.zeros_like(stable_mask) if forced is None else forced
    exempt = stable_mask | forced_mask | torch.isnan(scores)
    pool = torch.where(exempt, torch.full_like(scores, math.inf), scores)
    sorted_asc = torch.sort(pool.reshape(-1)).values
    if isinstance(budget, Tensor):
        # Tensor budget (the graph face): ``sorted_asc[idx]`` with a 0-dim
        # tensor index performs a host readback internally and is illegal
        # inside a CUDA graph capture; gather is the capture-safe,
        # mathematically identical selector.
        idx = torch.clamp(budget.to(dtype=torch.long), max=n - 1)
        thr = torch.gather(sorted_asc, 0, idx.reshape(1)).squeeze(0)
    else:
        thr = sorted_asc[min(int(budget), n - 1)]
    binding = budget < (~exempt).sum()
    flip = (~exempt) & (scores >= thr) & binding
    return stable_mask | flip
