"""Unit tests for the actfold.speculative.acceptance pure functions (AR004 T001)."""

from __future__ import annotations

import math

import pytest
import torch

from actfold.speculative.acceptance import (
    acceptance_rate,
    draft_region_mask,
    mean_log_prob,
    target_argmax_accept_mask,
)


def _make_logits(argmax_tokens: torch.Tensor, vocab_size: int, high: float = 10.0) -> torch.Tensor:
    """Build logits whose per-position argmax is exactly ``argmax_tokens``.

    Args:
        argmax_tokens: ``[batch, seq_len]`` integer tensor of argmax token ids.
        vocab_size: Vocabulary dimension of the produced logits.
        high: Logit magnitude assigned to each argmax entry.

    Returns:
        ``[batch, seq_len, vocab_size]`` float logits with controlled argmax.
    """
    batch, seq_len = argmax_tokens.shape
    logits = torch.zeros(batch, seq_len, vocab_size)
    logits.scatter_(-1, argmax_tokens.unsqueeze(-1), high)
    return logits


def _reference_mean_log_prob(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor | None,
) -> float:
    """Reference mean log-prob via fp32 ``log_softmax`` + ``gather``.

    Args:
        logits: ``[batch, seq_len, vocab]`` logits.
        tokens: ``[batch, seq_len]`` target token ids.
        mask: Optional ``[batch, seq_len]`` bool mask of averaged positions.

    Returns:
        Mean gathered token log-probability (subset mean when ``mask`` is given).
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    if mask is None:
        return token_log_probs.mean().item()
    return token_log_probs[mask].mean().item()


# ---------------------------------------------------------------------------
# UT-301: target_argmax_accept_mask
# ---------------------------------------------------------------------------


def test_ut301_target_argmax_accept_mask_partial_match_is_exact() -> None:
    """UT-301: accept mask is the exact per-position argmax equality (SRS 3.1-1).

    With controlled logits the mask must be True exactly where the child token
    equals the per-position argmax, and False everywhere else.
    """
    child = torch.tensor([[1, 2, 3], [0, 1, 2]])
    argmax = torch.tensor([[1, 0, 3], [0, 1, 1]])
    logits = _make_logits(argmax, vocab_size=4)

    mask = target_argmax_accept_mask(child, logits)

    assert mask.dtype == torch.bool
    assert mask.shape == child.shape
    assert torch.equal(mask, torch.tensor([[True, False, True], [True, True, False]]))


def test_ut301_target_argmax_accept_mask_all_positions_match() -> None:
    """UT-301: fully matching child tokens yield an all-True mask (SRS 3.1-1)."""
    child = torch.tensor([[2, 0, 3, 1]])
    logits = _make_logits(child, vocab_size=4)

    mask = target_argmax_accept_mask(child, logits)

    assert torch.equal(mask, torch.ones_like(mask))


def test_ut301_target_argmax_accept_mask_no_position_matches() -> None:
    """UT-301: no position matching the argmax yields an all-False mask (SRS 3.1-1)."""
    child = torch.tensor([[0, 1, 2]])
    argmax = torch.tensor([[1, 2, 0]])
    logits = _make_logits(argmax, vocab_size=3)

    mask = target_argmax_accept_mask(child, logits)

    assert torch.equal(mask, torch.zeros_like(mask))


def test_ut301_target_argmax_accept_mask_rows_are_independent() -> None:
    """UT-301: each batch row is evaluated independently (SRS 3.1-1)."""
    child = torch.tensor([[0, 0], [1, 1]])
    argmax = torch.tensor([[0, 1], [1, 1]])
    logits = _make_logits(argmax, vocab_size=2)

    mask = target_argmax_accept_mask(child, logits)

    assert torch.equal(mask, torch.tensor([[True, False], [True, True]]))


# ---------------------------------------------------------------------------
# UT-302: draft_region_mask
# ---------------------------------------------------------------------------


def test_ut302_draft_region_mask_equal_length_marks_only_diff_positions() -> None:
    """UT-302: equal-length branches mark exactly the differing positions (SRS 3.1-2)."""
    parent = torch.tensor([[1, 2, 3, 4]])
    child = torch.tensor([[1, 9, 3, 8]])

    mask = draft_region_mask(parent, child)

    assert mask.dtype == torch.bool
    assert mask.shape == (1, 4)
    assert torch.equal(mask, torch.tensor([[False, True, False, True]]))


def test_ut302_draft_region_mask_append_only_includes_full_suffix() -> None:
    """UT-302: append-only children mark diff positions plus the whole suffix (SRS 3.1-3)."""
    parent = torch.tensor([[1, 2, 3]])
    child = torch.tensor([[1, 9, 3, 4, 5]])

    mask = draft_region_mask(parent, child)

    assert mask.shape == (1, 5)
    assert torch.equal(mask, torch.tensor([[False, True, False, True, True]]))


def test_ut302_draft_region_mask_shorter_child_has_no_suffix_positions() -> None:
    """UT-302: a shorter child marks only common-prefix differences (SRS 3.1-3)."""
    parent = torch.tensor([[1, 2, 3, 4, 5]])
    child = torch.tensor([[1, 2, 0, 4]])

    mask = draft_region_mask(parent, child)

    assert mask.shape == (1, 4)
    assert torch.equal(mask, torch.tensor([[False, False, True, False]]))


def test_ut302_draft_region_mask_batch_mismatch_raises() -> None:
    """UT-302: mismatched batch sizes raise ValueError."""
    parent = torch.zeros(1, 3, dtype=torch.long)
    child = torch.zeros(2, 3, dtype=torch.long)

    with pytest.raises(ValueError):
        draft_region_mask(parent, child)


# ---------------------------------------------------------------------------
# UT-303: acceptance_rate
# ---------------------------------------------------------------------------


def test_ut303_acceptance_rate_partial_match_is_exact_ratio() -> None:
    """UT-303: rate is the exact accepted fraction of draft positions (SRS 3.1-1)."""
    accept = torch.tensor([[True, True, True, False]])
    draft = torch.tensor([[False, True, True, True]])

    rate = acceptance_rate(accept, draft)

    assert isinstance(rate, float)
    assert rate == pytest.approx(2.0 / 3.0)

    accept_b = torch.tensor([[True, False], [False, False]])
    draft_b = torch.tensor([[True, True], [True, False]])
    assert acceptance_rate(accept_b, draft_b) == pytest.approx(1.0 / 3.0)


def test_ut303_acceptance_rate_empty_draft_region_returns_one() -> None:
    """UT-303: an empty draft region is defined as rate 1.0 (SRS 3.1-2)."""
    accept = torch.zeros(2, 3, dtype=torch.bool)
    draft = torch.zeros(2, 3, dtype=torch.bool)

    assert acceptance_rate(accept, draft) == 1.0


def test_ut303_acceptance_rate_full_accept_and_full_reject_boundaries() -> None:
    """UT-303: all-accepted drafts give 1.0 and all-rejected drafts give 0.0 (SRS 3.1-1)."""
    draft = torch.tensor([[True, True, True]])

    assert acceptance_rate(torch.tensor([[True, True, True]]), draft) == 1.0
    assert acceptance_rate(torch.tensor([[False, False, False]]), draft) == 0.0


# ---------------------------------------------------------------------------
# UT-304: mean_log_prob
# ---------------------------------------------------------------------------


def test_ut304_mean_log_prob_matches_reference_over_all_positions() -> None:
    """UT-304: unmasked value equals the fp32 log_softmax gather reference (SRS 3.2-1)."""
    logits = torch.randn(2, 3, 5)
    tokens = torch.randint(0, 5, (2, 3))

    value = mean_log_prob(logits, tokens)

    assert isinstance(value, float)
    assert value == pytest.approx(_reference_mean_log_prob(logits, tokens, None), abs=1e-6)


def test_ut304_mean_log_prob_masked_matches_subset_mean() -> None:
    """UT-304: masked value equals the mean over the masked subset (SRS 3.2-1)."""
    logits = torch.randn(2, 4, 6)
    tokens = torch.randint(0, 6, (2, 4))
    mask = torch.tensor([[True, False, True, True], [False, False, True, False]])

    value = mean_log_prob(logits, tokens, mask)

    assert value == pytest.approx(_reference_mean_log_prob(logits, tokens, mask), abs=1e-6)


def test_ut304_mean_log_prob_empty_mask_returns_zero_sentinel() -> None:
    """UT-304: an all-False mask returns the documented 0.0 sentinel."""
    logits = torch.randn(1, 3, 4)
    tokens = torch.randint(0, 4, (1, 3))
    mask = torch.zeros(1, 3, dtype=torch.bool)

    assert mean_log_prob(logits, tokens, mask) == 0.0


def test_ut304_mean_log_prob_large_logits_stay_finite_in_fp32() -> None:
    """UT-304: 1e4-scale logits produce finite fp32 log-probs (SRS 3.2-1)."""
    logits = torch.full((1, 2, 4), -1e4)
    logits[0, 0, 2] = 1e4
    logits[0, 1, 0] = 1e4
    tokens = torch.tensor([[2, 1]])

    value = mean_log_prob(logits, tokens)

    assert math.isfinite(value)
    assert value == pytest.approx(_reference_mean_log_prob(logits, tokens, None), abs=1e-6)


# ---------------------------------------------------------------------------
# UT-305: shape / dtype validation
# ---------------------------------------------------------------------------


def test_ut305_accept_mask_rejects_three_dimensional_tokens() -> None:
    """UT-305: tokens with ndim != 2 raise ValueError (SRS 3.1-4)."""
    tokens = torch.zeros(1, 1, 3, dtype=torch.long)
    logits = torch.zeros(1, 3, 4)

    with pytest.raises(ValueError):
        target_argmax_accept_mask(tokens, logits)


def test_ut305_accept_mask_rejects_two_dimensional_logits() -> None:
    """UT-305: logits with ndim != 3 raise ValueError (SRS 3.1-4)."""
    tokens = torch.zeros(1, 3, dtype=torch.long)
    logits = torch.zeros(1, 4)

    with pytest.raises(ValueError):
        target_argmax_accept_mask(tokens, logits)


def test_ut305_accept_mask_rejects_batch_mismatch() -> None:
    """UT-305: batch mismatch between tokens and logits raises ValueError (SRS 3.1-4)."""
    tokens = torch.zeros(1, 3, dtype=torch.long)
    logits = torch.zeros(2, 3, 4)

    with pytest.raises(ValueError):
        target_argmax_accept_mask(tokens, logits)


def test_ut305_accept_mask_rejects_sequence_length_mismatch() -> None:
    """UT-305: sequence-length mismatch between tokens and logits raises ValueError (SRS 3.1-4)."""
    tokens = torch.zeros(1, 3, dtype=torch.long)
    logits = torch.zeros(1, 4, 4)

    with pytest.raises(ValueError):
        target_argmax_accept_mask(tokens, logits)


def test_ut305_accept_mask_rejects_non_integer_tokens() -> None:
    """UT-305: float-dtype tokens raise ValueError (SRS 3.1-4)."""
    tokens = torch.zeros(1, 3, dtype=torch.float32)
    logits = torch.zeros(1, 3, 4)

    with pytest.raises(ValueError):
        target_argmax_accept_mask(tokens, logits)


def test_ut305_acceptance_rate_rejects_shape_mismatch() -> None:
    """UT-305: accept/draft mask shape mismatch raises ValueError (SRS 3.1-4)."""
    accept = torch.zeros(1, 4, dtype=torch.bool)
    draft = torch.zeros(1, 3, dtype=torch.bool)

    with pytest.raises(ValueError):
        acceptance_rate(accept, draft)


def test_ut305_mean_log_prob_rejects_token_shape_mismatch() -> None:
    """UT-305: tokens not matching the logits' leading shape raise ValueError (SRS 3.1-4)."""
    logits = torch.zeros(1, 3, 4)
    tokens = torch.zeros(1, 4, dtype=torch.long)

    with pytest.raises(ValueError):
        mean_log_prob(logits, tokens)


def test_ut305_mean_log_prob_rejects_mask_shape_mismatch() -> None:
    """UT-305: a mask not matching the token shape raises ValueError (SRS 3.1-4)."""
    logits = torch.zeros(1, 3, 4)
    tokens = torch.zeros(1, 3, dtype=torch.long)
    mask = torch.zeros(1, 4, dtype=torch.bool)

    with pytest.raises(ValueError):
        mean_log_prob(logits, tokens, mask)


class _FixedForwardModel:
    """Stub adapter returning scripted logits regardless of input tokens."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.logits


def test_spiffy_baseline_score_is_mean_log_prob() -> None:
    """SpiffyBaseline.verify stores mean_log_prob as ``baseline_score`` (AR004 T004).

    The historical ``logits.mean()`` placeholder is replaced by the true
    verification score: the mean log-probability of the branch tokens under
    the model logits, computed over all positions. The metadata key name is
    unchanged by design (D6).
    """
    from actfold.speculative.branch import Branch
    from actfold.speculative.spiffy_baseline import SpiffyBaseline

    vocab_size = 8
    tokens = torch.tensor([[1, 2, 3]])
    # High probability for the actual tokens at every position.
    logits = _make_logits(tokens, vocab_size, high=10.0)
    baseline = SpiffyBaseline(_FixedForwardModel(logits), draft_generator=None)

    accepted = baseline.verify([Branch(branch_id="b", parent_id="root", tokens=tokens)])

    assert accepted.accepted
    assert accepted.metadata["baseline_score"] == pytest.approx(
        mean_log_prob(logits, tokens), abs=1e-6
    )


def test_spiffy_baseline_prefers_higher_log_prob_branch() -> None:
    """SpiffyBaseline.verify selects the branch with the higher mean log-prob."""
    from actfold.speculative.branch import Branch
    from actfold.speculative.spiffy_baseline import SpiffyBaseline

    vocab_size = 8
    good_tokens = torch.tensor([[1, 2, 3]])
    bad_tokens = torch.tensor([[4, 5, 6]])

    class _TwoBranchModel(_FixedForwardModel):
        def __init__(self) -> None:
            super().__init__(torch.zeros(1, 1, 1))
            self.good_logits = _make_logits(good_tokens, vocab_size, high=10.0)
            self.bad_logits = _make_logits(bad_tokens, vocab_size, high=0.5)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            if torch.equal(tokens, good_tokens):
                return self.good_logits
            return self.bad_logits

    good = Branch(branch_id="good", parent_id="root", tokens=good_tokens)
    bad = Branch(branch_id="bad", parent_id="root", tokens=bad_tokens)
    baseline = SpiffyBaseline(_TwoBranchModel(), draft_generator=None)

    accepted = baseline.verify([bad, good])

    assert accepted.branch_id == "good"
    assert accepted.metadata["baseline_score"] == pytest.approx(
        mean_log_prob(_TwoBranchModel().good_logits, good_tokens), abs=1e-6
    )
