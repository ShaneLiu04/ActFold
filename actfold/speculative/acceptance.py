"""True speculative-decoding acceptance semantics (AR004, P2-5).

This module hosts the pure functions that implement target-argmax acceptance
for diffusion-LM parallel verification.  The contract is the **same-position
prediction convention**: the target model's forward logits ``logits[:, i]``
predict the token at position ``i`` (the Fast-dLLM/SPIFFY parallel
verification contract; synthetic token-wise models follow the same shape).

A draft token is *accepted* at a position iff it equals the target argmax at
that same position.  Acceptance is only measured over the **draft region**:
positions where the child carries a new claim — differences against the
parent over their common prefix plus the whole appended suffix when the child
is longer.  Positions identical to the parent are already-verified reuse and
do not count.

All functions are stateless and vectorized; there are no per-position Python
loops and no extra forward passes.
"""

from __future__ import annotations

import torch


def _validate_token_tensor(
    name: str,
    tokens: torch.Tensor,
    ndim: int,
) -> None:
    """Validate a token tensor's rank and integral dtype.

    Args:
        name: Argument name used in error messages.
        tokens: Token tensor to validate.
        ndim: Required rank (2 for ``[batch, seq_len]``).

    Raises:
        ValueError: If ``tokens`` has the wrong rank or a non-integer dtype.
    """
    if tokens.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions [batch, seq_len], "
            f"got shape {tuple(tokens.shape)}"
        )
    if (
        torch.is_floating_point(tokens)
        or torch.is_complex(tokens)
        or tokens.dtype == torch.bool
    ):
        raise ValueError(
            f"{name} must have an integer dtype, got {tokens.dtype}"
        )


def target_argmax_accept_mask(
    child_tokens: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Per-position accept mask: child token == same-position target argmax.

    Args:
        child_tokens: ``[batch, seq_len]`` integer draft tokens.
        logits: ``[batch, seq_len, vocab_size]`` target forward logits
            (same-position prediction convention).

    Returns:
        ``[batch, seq_len]`` boolean accept mask.

    Raises:
        ValueError: If shapes are inconsistent or ``child_tokens`` has a
            non-integer dtype.
    """
    _validate_token_tensor("child_tokens", child_tokens, ndim=2)
    if logits.ndim != 3:
        raise ValueError(
            f"logits must have 3 dimensions [batch, seq_len, vocab_size], "
            f"got shape {tuple(logits.shape)}"
        )
    if logits.shape[:2] != child_tokens.shape:
        raise ValueError(
            f"logits leading shape {tuple(logits.shape[:2])} does not match "
            f"child_tokens shape {tuple(child_tokens.shape)}"
        )
    predicted = logits.argmax(dim=-1)
    return predicted == child_tokens


def draft_region_mask(
    parent_tokens: torch.Tensor,
    child_tokens: torch.Tensor,
) -> torch.Tensor:
    """Positions carrying new draft claims: prefix diffs + appended suffix.

    The draft region is the union of (a) positions where the child differs
    from the parent over their common prefix ``[:, :min(T_p, T_c)]`` and
    (b) the appended suffix ``[T_p, T_c)`` when the child is longer than the
    parent.  Positions identical to the parent are already-verified reuse.

    When the child is *shorter* than the parent, only (a) applies over the
    child's own length (there is no appended suffix).

    Args:
        parent_tokens: ``[batch, T_p]`` integer parent tokens.
        child_tokens: ``[batch, T_c]`` integer child tokens.

    Returns:
        ``[batch, T_c]`` boolean draft-region mask.

    Raises:
        ValueError: If the batch sizes differ.
    """
    _validate_token_tensor("parent_tokens", parent_tokens, ndim=2)
    _validate_token_tensor("child_tokens", child_tokens, ndim=2)
    if parent_tokens.shape[0] != child_tokens.shape[0]:
        raise ValueError(
            f"Batch mismatch: parent {parent_tokens.shape[0]} vs child "
            f"{child_tokens.shape[0]}"
        )

    t_parent = parent_tokens.shape[1]
    t_child = child_tokens.shape[1]
    common = min(t_parent, t_child)

    mask = torch.zeros(
        child_tokens.shape, dtype=torch.bool, device=child_tokens.device
    )
    mask[:, :common] = child_tokens[:, :common] != parent_tokens[:, :common]
    if t_child > t_parent:
        mask[:, t_parent:] = True
    return mask


def acceptance_rate(
    accept_mask: torch.Tensor,
    draft_mask: torch.Tensor,
) -> float:
    """Mean accept ratio over the draft positions.

    An empty draft region (no ``True`` positions) is defined as ``1.0``:
    the child makes no new claims, which is semantically full acceptance.

    Args:
        accept_mask: ``[batch, seq_len]`` boolean accept mask from
            :func:`target_argmax_accept_mask`.
        draft_mask: ``[batch, seq_len]`` boolean draft-region mask from
            :func:`draft_region_mask`.

    Returns:
        The acceptance rate in ``[0, 1]``; ``1.0`` for an empty draft region.

    Raises:
        ValueError: If the two masks have different shapes.
    """
    if accept_mask.shape != draft_mask.shape:
        raise ValueError(
            f"accept_mask shape {tuple(accept_mask.shape)} does not match "
            f"draft_mask shape {tuple(draft_mask.shape)}"
        )
    if not bool(draft_mask.any()):
        return 1.0
    return float(accept_mask[draft_mask].float().mean().item())


def mean_log_prob(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """fp32 log-softmax gather mean over the masked positions.

    Args:
        logits: ``[batch, seq_len, vocab_size]`` target forward logits.
        tokens: ``[batch, seq_len]`` integer token ids to score.
        mask: Optional ``[batch, seq_len]`` boolean mask of scored positions.
            ``None`` scores every position.

    Returns:
        The mean token log-probability under the target distribution.  An
        all-``False`` mask returns the documented ``0.0`` sentinel (there
        are no scored positions; this is not a probability claim).

    Raises:
        ValueError: If shapes are inconsistent or ``tokens`` has a
            non-integer dtype.
    """
    if logits.ndim != 3:
        raise ValueError(
            f"logits must have 3 dimensions [batch, seq_len, vocab_size], "
            f"got shape {tuple(logits.shape)}"
        )
    _validate_token_tensor("tokens", tokens, ndim=2)
    if logits.shape[:2] != tokens.shape:
        raise ValueError(
            f"logits leading shape {tuple(logits.shape[:2])} does not match "
            f"tokens shape {tuple(tokens.shape)}"
        )
    if mask is not None and mask.shape != tokens.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match tokens shape "
            f"{tuple(tokens.shape)}"
        )

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(
        -1, tokens.unsqueeze(-1).long()
    ).squeeze(-1)

    if mask is None:
        return float(token_log_probs.mean().item())
    if not bool(mask.any()):
        return 0.0
    return float(token_log_probs[mask].mean().item())
