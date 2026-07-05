"""Shared utilities for diffusion-native samplers.

This module collects helpers that are common across LLaDA, Dream, and Fast-dLLM
reference samplers: masking schedules, number-of-transfers computation, Gumbel
noise, top-p/top-k filtering, and canvas construction.  Keeping them in one
place lets each sampler stay close to its official recipe while sharing
well-tested primitives.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

Number = torch.Tensor | float


# ---------------------------------------------------------------------------
# Masking schedules
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class MaskingScheduler:
    """Base class for masking rate alpha(t) schedulers, t in [0, 1].

    Subclasses implement ``_alpha`` and ``_alpha_derivative``.  The scheduler
    is used by :func:`get_num_transfer_tokens` to decide how many tokens to
    unmask at each reverse-diffusion step.
    """

    __registry__: ClassVar[dict[str, type["MaskingScheduler"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        MaskingScheduler.__registry__[cls.__name__] = cls
        MaskingScheduler.__registry__[cls.__name__.lower()] = cls

    def alpha(self, t: Number) -> Number:
        """Return alpha(t), the masking rate at time t."""
        t_t = torch.as_tensor(
            t,
            dtype=torch.float32,
            device=t.device if isinstance(t, torch.Tensor) else None,
        )
        if not torch.all((0.0 <= t_t) & (t_t <= 1.0)):
            raise ValueError(f"t={t} not in [0, 1]")
        out = self._alpha(t_t)
        return out.item() if isinstance(t, float) else out

    def alpha_derivative(self, t: Number) -> Number:
        """Return d alpha / d t."""
        t_t = torch.as_tensor(
            t,
            dtype=torch.float32,
            device=t.device if isinstance(t, torch.Tensor) else None,
        )
        if not torch.all((0.0 <= t_t) & (t_t <= 1.0)):
            raise ValueError(f"t={t} not in [0, 1]")
        out = self._alpha_derivative(t_t)
        return out.item() if isinstance(t, float) else out

    def reverse_mask_prob(self, s: Number, t: Number) -> Number:
        """Return P(still masked at s | masked at t) = (1 - alpha(s)) / (1 - alpha(t)).

        Args:
            s: Earlier (closer to unmasked) time, in [0, 1).
            t: Later (closer to fully masked) time, in (0, 1].
        """
        s_t = torch.as_tensor(
            s,
            dtype=torch.float32,
            device=s.device if isinstance(s, torch.Tensor) else None,
        )
        t_t = torch.as_tensor(
            t,
            dtype=torch.float32,
            device=t.device if isinstance(t, torch.Tensor) else None,
        )
        if not torch.all((0.0 <= s_t) & (s_t < 1.0) & (0.0 < t_t) & (t_t <= 1.0)):
            raise ValueError(f"(s={s}, t={t}) out of range")
        if not torch.all(s_t < t_t):
            raise ValueError(f"Require s < t elementwise, got (s={s}, t={t})")
        out: torch.Tensor = (1.0 - self._alpha(s_t)) / (1.0 - self._alpha(t_t) + 1e-6)
        if isinstance(s, float) and isinstance(t, float):
            return float(out.item())
        return out

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _alpha_derivative(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclasses.dataclass
class LinearMaskingScheduler(MaskingScheduler):
    """Linear schedule: alpha(t) = 1 - t."""

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sub(1.0, t)

    def _alpha_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)


@dataclasses.dataclass
class CosineMaskingScheduler(MaskingScheduler):
    """Cosine schedule: alpha(t) = 1 - cos(pi/2 * (1 - t))."""

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sub(1.0, torch.cos((math.pi / 2.0) * (1.0 - t)))

    def _alpha_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -(math.pi / 2.0) * torch.sin((math.pi / 2.0) * (1.0 - t))


def make_masking_scheduler(name: str) -> MaskingScheduler:
    """Instantiate a scheduler by name (case-insensitive)."""
    cls = MaskingScheduler.__registry__.get(name) or MaskingScheduler.__registry__.get(name.lower())
    if cls is None:
        available = sorted(k for k in MaskingScheduler.__registry__ if k[0].isupper())
        raise ValueError(f"Unknown masking scheduler '{name}'. Available: {available}")
    return cls()


# ---------------------------------------------------------------------------
# Number of tokens to transfer per step
# ---------------------------------------------------------------------------
def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: MaskingScheduler | None = None,
    stochastic: bool = False,
) -> torch.Tensor:
    """Compute how many masked tokens to reveal at each diffusion step.

    Follows the schedule used in the official LLaDA/Dream reference code.
    Steps with zero transfers are removed and rows are right-padded to the
    same length.

    Args:
        mask_index: Boolean tensor ``[B, L]`` indicating masked positions.
        steps: Total number of reverse-diffusion steps.
        scheduler: Masking scheduler. Defaults to ``LinearMaskingScheduler``.
        stochastic: If True, sample transfers from a binomial distribution;
            otherwise use the deterministic expected value.

    Returns:
        Integer tensor ``[B, effective_steps]`` with the number of tokens to
        unmask at each effective step.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if scheduler is None:
        scheduler = LinearMaskingScheduler()

    mask_num = mask_index.sum(dim=1, keepdim=True)  # [B, 1]
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    )

    for i in range(mask_num.size(0)):
        remaining = int(mask_num[i, 0].item())
        for t_idx, s_idx, j in zip(range(steps, 0, -1), range(steps - 1, -1, -1), range(steps)):
            s_norm = s_idx / steps
            t_norm = t_idx / steps
            reverse_transfer_prob = 1.0 - float(scheduler.reverse_mask_prob(s=s_norm, t=t_norm))
            reverse_transfer_prob = max(0.0, min(1.0, reverse_transfer_prob))

            if remaining <= 0:
                break

            if not stochastic:
                x = remaining * reverse_transfer_prob
                n_tok = int(round(x))
            else:
                n_tok = int(
                    torch.distributions.Binomial(  # type: ignore[no-untyped-call]
                        torch.tensor(remaining, dtype=torch.float64),
                        torch.tensor(reverse_transfer_prob, dtype=torch.float64),
                    )
                    .sample()  # type: ignore[no-untyped-call]
                    .item()
                )

            n_tok = min(n_tok, remaining)
            num_transfer_tokens[i, j] = n_tok
            remaining -= n_tok

    # Remove all-zero columns and right-pad rows to the same effective length.
    rows: list[torch.Tensor] = []
    max_len = 0
    for i in range(num_transfer_tokens.size(0)):
        nonzero = num_transfer_tokens[i][num_transfer_tokens[i] > 0]
        rows.append(nonzero)
        max_len = max(max_len, nonzero.numel())

    if max_len == 0:
        return torch.zeros(mask_num.size(0), 1, device=mask_index.device, dtype=torch.int64)

    padded_rows: list[torch.Tensor] = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad])
        padded_rows.append(r)
    return torch.stack(padded_rows, dim=0)


# ---------------------------------------------------------------------------
# Gumbel noise and top-p/top-k filtering
# ---------------------------------------------------------------------------
def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel-Max noise for low-temperature sampling.

    Following the official MDLM implementation, computations are performed in
    float64 to avoid numerical inconsistencies reported in the literature.
    """
    if temperature == 0.0:
        return logits
    logits_f64 = logits.to(torch.float64)
    noise = torch.rand_like(logits_f64, dtype=torch.float64)
    gumbel_noise: torch.Tensor = (-torch.log(noise)) ** temperature
    out: torch.Tensor = logits_f64.exp() / gumbel_noise
    return out


def top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Set logits outside the top-k to -inf."""
    if top_k <= 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
    min_values = values[..., -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Set logits outside the nucleus to -inf."""
    if top_p >= 1.0 or top_p <= 0.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample or greedily select tokens and return per-token confidence.

    Args:
        logits: Logits tensor of arbitrary shape ending in ``[..., V]``.
        temperature: Sampling temperature. 0 means greedy argmax.
        top_p: Nucleus cutoff.
        top_k: Top-k cutoff.
        margin_confidence: If True, confidence = p(top1) - p(top2).
        neg_entropy: If True, confidence = negative entropy.

    Returns:
        Tuple ``(confidence, tokens)`` with the same leading shape as ``logits``.
    """
    if temperature > 0.0:
        logits = logits / temperature
    if top_p is not None and top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    if top_k is not None and top_k > 0:
        logits = top_k_logits(logits, top_k)

    probs = F.softmax(logits, dim=-1)

    if temperature > 0.0:
        try:
            x0 = torch.distributions.Categorical(probs=probs).sample()  # type: ignore[no-untyped-call]
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[..., 0] - sorted_probs[..., 1]

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = -(probs * log_probs).sum(dim=-1)

    return confidence, x0


# ---------------------------------------------------------------------------
# Canvas / attention-mask helpers
# ---------------------------------------------------------------------------
def right_shift_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shift logits one position to the right.

    Some diffusion LMs are trained to predict token at position i+1 from the
    hidden state at position i (AR alignment).  This helper duplicates the
    first column to keep the sequence length unchanged.
    """
    if logits.shape[1] == 0:
        return logits
    return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)


def build_left_padded_canvas(
    inputs: list[torch.Tensor],
    max_new_tokens: int,
    eos_token_id: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build a left-padded canvas for Dream-style samplers.

    Returns:
        - ``x``: ``[B, T]`` canvas with inputs right-aligned and the generation
          region filled with ``mask_token_id``.
        - ``attention_mask``: ``[B, T]`` with 1 on valid (non-pad) positions.
        - ``seq_lens``: list of total valid lengths per sample.
    """
    prompt_lens = [p.shape[0] for p in inputs]
    max_length = max_new_tokens + max(prompt_lens)
    B = len(inputs)
    T = max_length
    device = inputs[0].device

    x = torch.full((B, T), eos_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    seq_lens: list[int] = []

    for i, p in enumerate(inputs):
        total_len = prompt_lens[i] + max_new_tokens
        seq_lens.append(total_len)
        start = T - total_len
        x[i, start : start + prompt_lens[i]] = p
        x[i, start + prompt_lens[i] : T] = mask_token_id
        attention_mask[i, start:T] = 1

    return x, attention_mask, seq_lens


def build_right_padded_canvas(
    inputs: list[torch.Tensor],
    max_new_tokens: int,
    eos_token_id: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
    """Build a right-padded canvas for LLaDA-style samplers.

    Returns:
        - ``x``: ``[B, T]`` canvas with inputs left-aligned and the generation
          region filled with ``mask_token_id``.
        - ``attention_mask``: ``[B, T]`` with 1 on valid (non-EOS) positions.
        - ``prompt_lens``: list of prompt lengths.
        - ``max_length``: total canvas length.
    """
    prompt_lens = [p.shape[0] for p in inputs]
    max_length = max_new_tokens + max(prompt_lens)
    B = len(inputs)
    T = max_length
    device = inputs[0].device

    x = torch.full((B, T), eos_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)

    for i, p in enumerate(inputs):
        x[i, : prompt_lens[i]] = p
        x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_token_id
        valid_end = min(prompt_lens[i] + max_new_tokens, T)
        attention_mask[i, :valid_end] = 1

    return x, attention_mask, prompt_lens, max_length


def compute_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute 1-D position ids that respect left-side padding.

    Valid positions are numbered from 0 to L-1 within each row; padding
    positions receive position id 1 (a harmless placeholder).
    """
    pos_id = attention_mask.long().cumsum(dim=-1) - 1
    pos_id.masked_fill_(attention_mask == 0, 1)
    return pos_id


__all__ = [
    "MaskingScheduler",
    "LinearMaskingScheduler",
    "CosineMaskingScheduler",
    "make_masking_scheduler",
    "get_num_transfer_tokens",
    "add_gumbel_noise",
    "top_k_logits",
    "top_p_logits",
    "sample_tokens",
    "right_shift_logits",
    "build_left_padded_canvas",
    "build_right_padded_canvas",
    "compute_position_ids",
]
