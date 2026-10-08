"""Tests for actfold.core.similarity_gate."""

from __future__ import annotations

import pytest
import torch

from actfold.core.similarity_gate import SimilarityGate


@pytest.fixture
def identical_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(2, 8, 32)
    return x, x.clone()


@pytest.fixture
def different_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(2, 8, 32)
    y = torch.randn(2, 8, 32)
    return x, y


def test_identical_is_stable(identical_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = identical_tensors
    gate = SimilarityGate(tau=0.95, metric="cosine")
    mask = gate(x, y)
    assert mask.shape == (2, 8)
    assert mask.all()


def test_different_is_unstable(different_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = different_tensors
    gate = SimilarityGate(tau=0.99, metric="cosine")
    mask = gate(x, y)
    # Random independent vectors are very unlikely to exceed 0.99 cosine similarity.
    assert not mask.any()


@pytest.mark.parametrize("metric", ["cosine", "l2", "pearson"])
def test_supported_metrics(
    identical_tensors: tuple[torch.Tensor, torch.Tensor],
    metric: str,
) -> None:
    x, y = identical_tensors
    gate = SimilarityGate(tau=0.95, metric=metric)
    mask = gate(x, y)
    assert mask.shape == (2, 8)
    assert mask.all()


def test_invalid_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        SimilarityGate(metric="jaccard")


def test_invalid_tau() -> None:
    with pytest.raises(ValueError, match="tau must be in"):
        SimilarityGate(tau=1.5)


def test_shape_mismatch() -> None:
    gate = SimilarityGate()
    with pytest.raises(ValueError, match="Shape mismatch"):
        gate(torch.randn(2, 8, 32), torch.randn(2, 10, 32))


def test_set_tau() -> None:
    gate = SimilarityGate(tau=0.95)
    gate.set_tau(0.8)
    assert gate.tau == pytest.approx(0.8)


def test_set_tau_invalid() -> None:
    gate = SimilarityGate()
    with pytest.raises(ValueError):
        gate.set_tau(-0.1)


def test_cosine_clamped_at_one() -> None:
    """tau=1.0 must make identical tokens divergent (sim clamped to 1.0)."""
    x = torch.randn(2, 8, 32) * 1000.0
    gate = SimilarityGate(tau=1.0, metric="cosine")
    mask = gate(x, x.clone())
    assert not mask.any()


# ---------------------------------------------------------------------------
# B7: low-precision (fp16/bf16) numerical robustness and NaN handling.
#
# These tests pin the dtype-aware eps floor, fp32 accumulation, and the
# NaN -> divergent mapping planned in design.md section 4.3.
# ---------------------------------------------------------------------------


def test_fp16_zero_norm_vector_is_divergent() -> None:
    """fp16 zero-norm parents must be divergent with finite, fp32-consistent scores.

    A fixed ``eps=1e-8`` is far below the fp16 minimum normal (~6.1e-5), so the
    cosine denominator collapses to zero for zero-norm vectors and the fp16
    score degrades to NaN. The fp16 mask must be all False (a zero-norm parent
    cannot be verified as stable) and must equal the fp32 mask for the same
    input. The raw fp16 scores (no dtype cast, so the degradation stays
    observable) must be finite.
    """
    h_child = torch.ones(1, 4, 64, dtype=torch.float16)
    h_parent = torch.zeros(1, 4, 64, dtype=torch.float16)
    gate = SimilarityGate(tau=0.95, metric="cosine")

    mask = gate(h_child, h_parent)
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 4)
    assert not mask.any()

    # Scores computed from the raw fp16 inputs must be finite (pre-fix they
    # are NaN because eps=1e-8 underflows to 0 in fp16).
    sim = gate._compute_similarity(h_child, h_parent)
    assert torch.isfinite(sim).all()

    # Consistency across dtypes: the fp16 result must match the fp32 result.
    mask_fp32 = gate(h_child.to(torch.float32), h_parent.to(torch.float32))
    assert not mask_fp32.any()
    assert torch.equal(mask, mask_fp32)


def test_fp16_large_magnitude_no_overflow() -> None:
    """Identical fp16 vectors with overflowing dot products must be stable.

    ``500**2 * 512 ~= 1.28e8`` exceeds the fp16 max (65504), so an fp16 dot
    product overflows to ``inf`` and the cosine degenerates to ``inf/inf`` or
    NaN. With fp32 accumulation the similarity of identical inputs is exactly
    1.0 and must exceed ``tau``.
    """
    v = torch.full((1, 4, 512), 500.0, dtype=torch.float16)
    gate = SimilarityGate(tau=0.9, metric="cosine")

    mask = gate(v.clone(), v.clone())
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 4)
    assert mask.all()

    # Sanity check that this is a real computation, not an always-True gate:
    # flipping one element to -500 gives cosine ~= 510/512 ~= 0.9961, which
    # must fail a strict tau=0.999 threshold.
    h_parent2 = v.clone()
    h_parent2[..., 0] = -500.0
    strict_gate = SimilarityGate(tau=0.999, metric="cosine")
    assert not strict_gate(v.clone(), h_parent2).any()


def test_fp16_large_magnitude_pearson_no_overflow() -> None:
    """Identical large-magnitude fp16 inputs must be stable under ``pearson``.

    Unlike ``F.cosine_similarity`` (whose torch 2.5 CPU kernel already
    accumulates in fp32), the pearson path multiplies and sums in the input
    dtype: ``500 * 500 = 250000`` overflows fp16 elementwise, so both the
    numerator and denominator become ``inf`` and the score degenerates to
    ``inf/inf = NaN`` -> divergent, even though the true correlation is 1.0.
    """
    x = torch.empty(1, 4, 512, dtype=torch.float16)
    # Alternating signs keep the row non-constant (a constant row would
    # center to zero and be degenerate even in fp32) and the mean at zero.
    x[..., 0::2] = 500.0
    x[..., 1::2] = -500.0
    gate = SimilarityGate(tau=0.9, metric="pearson")

    mask = gate(x, x.clone())
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 4)
    assert mask.all()


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_nan_inputs_map_to_divergent(metric: str) -> None:
    """NaN similarity scores must map to divergent (False) without raising.

    Regression guard: this may already pass pre-fix (``NaN > tau`` evaluates
    to False), but it pins the NaN -> divergent contract for the
    fp32-accumulation rework so the fix cannot regress it.
    """
    h_child = torch.randn(2, 4, 32)
    h_child[0, 2, :] = float("nan")
    # NaN present at the same position in both inputs.
    h_parent = h_child.clone()
    gate = SimilarityGate(tau=0.95, metric=metric)

    mask = gate(h_child, h_parent)  # Must not raise.
    assert mask.dtype == torch.bool

    expected = torch.ones(2, 4, dtype=torch.bool)
    expected[0, 2] = False  # NaN token is divergent...
    assert torch.equal(mask, expected)  # ...while identical finite tokens stay stable.


def test_effective_eps_dtype_floor() -> None:
    """``_effective_eps`` must floor eps by dtype (design.md section 4.3).

    The floor is ``max(user eps, {fp16: 1e-4, bf16: 1e-2, fp32/fp64: 0.0})``.
    Pre-fix this fails with ``AttributeError`` because ``_effective_eps`` does
    not exist yet.
    """
    gate = SimilarityGate(tau=0.5)
    assert gate._effective_eps(torch.float16) >= 1e-4
    assert gate._effective_eps(torch.bfloat16) >= 1e-2
    # Full-precision dtypes keep the user-provided eps unchanged.
    assert gate._effective_eps(torch.float32) == 1e-8
    assert gate._effective_eps(torch.float64) == 1e-8

    # A larger user eps must be respected (the floor is a lower bound).
    custom = SimilarityGate(tau=0.5, eps=1e-3)
    assert custom._effective_eps(torch.float16) >= 1e-3


def test_similarity_computed_in_fp32_internally(monkeypatch: pytest.MonkeyPatch) -> None:
    """fp16 inputs must never reach ``F.cosine_similarity`` as fp16.

    Wraps ``torch.nn.functional.cosine_similarity`` with a spy that records
    the dtype of every call while delegating to the original implementation.
    Tolerant to a Green implementation that avoids the call entirely (its own
    fp32 dot): the assertion is only that *no* recorded call used float16.
    """
    original = torch.nn.functional.cosine_similarity
    recorded_dtypes: list[torch.dtype] = []

    def _spy(
        x1: torch.Tensor,
        x2: torch.Tensor,
        dim: int = 1,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        recorded_dtypes.append(x1.dtype)
        return original(x1, x2, dim=dim, eps=eps)

    monkeypatch.setattr(torch.nn.functional, "cosine_similarity", _spy)

    gate = SimilarityGate(tau=0.9, metric="cosine")
    x = torch.randn(2, 4, 64, dtype=torch.float16)
    mask = gate(x, x.clone())

    # Identical fp16 inputs of moderate magnitude must be stable...
    assert mask.all()
    # ...and any call that reached F.cosine_similarity must have seen fp32
    # inputs (i.e., the gate cast up before dispatch, or bypassed the call).
    for dtype in recorded_dtypes:
        assert dtype == torch.float32
    assert torch.float16 not in recorded_dtypes


# ---------------------------------------------------------------------------
# Regression guards for existing behavior (must pass pre- and post-fix).
# ---------------------------------------------------------------------------


def test_l2_metric_matches_manual_computation() -> None:
    """Regression guard: ``l2`` scores must match the manual formula on fp32."""
    x = torch.randn(2, 4, 32)
    y = x + 0.1 * torch.randn(2, 4, 32)
    gate = SimilarityGate(tau=0.5, metric="l2")

    sim = gate._compute_similarity(x, y)
    expected = 1.0 - torch.norm(x - y, p=2, dim=-1) / (32**0.5 + 1e-8)
    assert torch.allclose(sim, expected)
    assert torch.equal(sim > 0.5, gate(x, y))


def test_pearson_identical_is_stable() -> None:
    """Regression guard: pearson similarity of identical inputs is 1.0."""
    x = torch.randn(2, 4, 32)
    gate = SimilarityGate(tau=0.99, metric="pearson")
    assert gate(x, x.clone()).all()


def test_l2_identical_is_stable() -> None:
    """Regression guard: identical l2 inputs give ``1 - 0/sqrt(H) = 1.0``."""
    x = torch.randn(2, 4, 64)
    gate = SimilarityGate(tau=0.99, metric="l2")
    assert gate(x, x.clone()).all()
