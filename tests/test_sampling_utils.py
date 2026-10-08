"""Tests for shared sampling utilities."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from actfold.models.sampling_utils import (
    CosineMaskingScheduler,
    LinearMaskingScheduler,
    MaskingScheduler,
    get_num_transfer_tokens,
    sample_tokens,
)


def test_sample_tokens_nan_logits_raises() -> None:
    """sample_tokens raises RuntimeError on all-NaN logits.

    NaN logits must fail loudly instead of flowing through softmax into NaN
    probabilities and garbage token choices.
    """
    logits = torch.full((2, 5), float("nan"))

    with pytest.raises(RuntimeError, match=r"[Nn]a[Nn]"):
        sample_tokens(logits)


def test_sample_tokens_inf_logits_raises() -> None:
    """sample_tokens raises RuntimeError on infinite logits."""
    logits = torch.full((2, 5), float("inf"))

    with pytest.raises(RuntimeError, match=r"(?i)(inf|finite)"):
        sample_tokens(logits)


def test_sample_tokens_no_silent_greedy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors from the Categorical sampling path propagate to the caller.

    A broken stochastic path must surface its exception rather than being
    swallowed by a broad ``except`` that silently degrades to greedy argmax.
    """

    def _raise(*args: object, **kwargs: object) -> torch.Tensor:
        raise ValueError("synthetic Categorical.sample failure")

    monkeypatch.setattr(torch.distributions.Categorical, "sample", _raise)
    logits = torch.tensor([[0.1, 0.4, 0.2, 0.3], [1.0, 0.0, 0.5, 0.5]])

    with pytest.raises(ValueError, match="synthetic Categorical.sample failure"):
        sample_tokens(logits, temperature=1.0)


def test_sample_tokens_greedy_still_works() -> None:
    """Greedy sampling returns the argmax token and its softmax confidence."""
    logits = torch.tensor([[0.0, 1.0, 3.0, 2.0]])

    confidence, tokens = sample_tokens(logits, temperature=0.0)

    expected_confidence = F.softmax(logits, dim=-1)[0, 2]
    assert torch.equal(tokens, torch.tensor([2]))
    assert confidence.shape == (1,)
    assert confidence.item() == pytest.approx(expected_confidence.item())


# ---------------------------------------------------------------------------
# T018: vectorized get_num_transfer_tokens (bit-exact deterministic numerics)
# ---------------------------------------------------------------------------
# The reference below is a VERBATIM copy of the current per-row Python-loop
# implementation of ``get_num_transfer_tokens`` in
# ``actfold/models/sampling_utils.py``.  The vectorized rewrite must produce
# bit-identical results to this reference in deterministic mode.


def _reference_get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: MaskingScheduler | None = None,
    stochastic: bool = False,
) -> torch.Tensor:
    """Frozen pre-T018 reference: per-row loop with per-row ``.item()`` reads.

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
                    .sample()
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


def _install_sync_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch ``torch.Tensor.item``/``tolist``/``cpu`` to count host readbacks.

    The patched methods keep their original behavior; they only increment a
    per-method counter so tests can measure tensor-to-host synchronization
    counts without breaking the code under test.
    """
    counts: dict[str, int] = {"item": 0, "tolist": 0, "cpu": 0}

    def _make_wrapper(method_name: str, original: Any) -> Any:
        def _counting(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            counts[method_name] += 1
            return original(self, *args, **kwargs)

        return _counting

    for method_name in ("item", "tolist", "cpu"):
        original = getattr(torch.Tensor, method_name)
        monkeypatch.setattr(torch.Tensor, method_name, _make_wrapper(method_name, original))
    return counts


def _total_syncs(counts: dict[str, int]) -> int:
    """Return the total number of counted host readbacks."""
    return counts["item"] + counts["tolist"] + counts["cpu"]


def _t018_mask_cases() -> dict[str, torch.Tensor]:
    """Build deterministic mask_index cases covering shapes and densities.

    Covers B in 1..8, L in 1..64, and the edge densities (all-masked,
    none-masked, single token, one-per-row) plus random densities.
    """
    cases: dict[str, torch.Tensor] = {}
    cases["all_masked_1x1"] = torch.ones((1, 1), dtype=torch.bool)
    cases["all_masked_8x64"] = torch.ones((8, 64), dtype=torch.bool)
    cases["none_masked_4x16"] = torch.zeros((4, 16), dtype=torch.bool)
    single = torch.zeros((5, 32), dtype=torch.bool)
    single[2, 17] = True
    cases["single_token_5x32"] = single
    per_row = torch.zeros((3, 7), dtype=torch.bool)
    per_row[0, 3] = True
    per_row[1, 0] = True
    per_row[2, 6] = True
    cases["one_per_row_3x7"] = per_row
    gen = torch.Generator().manual_seed(1234)
    cases["sparse_8x64"] = torch.rand((8, 64), generator=gen) < 0.15
    cases["dense_6x48"] = torch.rand((6, 48), generator=gen) < 0.85
    cases["mixed_2x63"] = torch.cat(
        [torch.ones((1, 63), dtype=torch.bool), torch.zeros((1, 63), dtype=torch.bool)]
    )
    cases["width1_8x1"] = torch.rand((8, 1), generator=gen) < 0.5
    return cases


_T018_MASK_CASES = _t018_mask_cases()


@pytest.mark.parametrize(
    "scheduler",
    [
        pytest.param(None, id="default"),
        pytest.param(LinearMaskingScheduler(), id="linear"),
        pytest.param(CosineMaskingScheduler(), id="cosine"),
    ],
)
@pytest.mark.parametrize("steps", [1, 5, 32])
@pytest.mark.parametrize("mask_name", list(_T018_MASK_CASES))
def test_t018_num_transfer_tokens_reference_parity(
    mask_name: str,
    steps: int,
    scheduler: MaskingScheduler | None,
) -> None:
    """Vectorized get_num_transfer_tokens is bit-identical to the loop version.

    Deterministic results must match the frozen per-row reference exactly
    (torch.equal) across schedulers, step counts, and mask shapes/densities.
    """
    mask_index = _T018_MASK_CASES[mask_name]

    new = get_num_transfer_tokens(mask_index, steps=steps, scheduler=scheduler)
    reference = _reference_get_num_transfer_tokens(mask_index, steps=steps, scheduler=scheduler)

    assert new.dtype == torch.int64
    assert torch.equal(new, reference)


def test_t018_num_transfer_tokens_host_sync_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic mode performs at most 2 tensor-to-host readbacks.

    The vectorized implementation must not read back per-row or per-step
    values; a B=8, steps=16 call may sync at most twice in total.
    """
    counts = _install_sync_counter(monkeypatch)
    gen = torch.Generator().manual_seed(2024)
    mask_index = torch.rand((8, 64), generator=gen) < 0.5

    out = get_num_transfer_tokens(mask_index, steps=16)

    assert out.shape[0] == 8
    assert out.shape[1] >= 1
    assert _total_syncs(counts) <= 2


def test_t018_num_transfer_tokens_stochastic_invariants() -> None:
    """Stochastic mode keeps its output invariants (stays stochastic).

    With a fixed seed the output stays int64, non-negative, shaped
    ``[B, >=1]``, and each row's total never exceeds its mask count.
    """
    torch.manual_seed(123)
    gen = torch.Generator().manual_seed(99)
    mask_index = torch.rand((6, 32), generator=gen) < 0.6

    out = get_num_transfer_tokens(mask_index, steps=16, stochastic=True)

    assert out.dtype == torch.int64
    assert out.dim() == 2
    assert out.shape[0] == 6
    assert out.shape[1] >= 1
    assert bool((out >= 0).all())
    row_totals = out.sum(dim=1)
    mask_counts = mask_index.sum(dim=1).to(torch.int64)
    assert bool((row_totals <= mask_counts).all())
