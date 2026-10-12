"""Direct unit tests for the pure helpers in ``scripts/algo_experiments.py``.

AR013 pins for ``make_child`` and ``fidelity`` — the two pure experiment
helpers left uncovered by ``tests/test_experiment_stats.py`` (that file owns
the statistics contract, AGENTS #32).  Every expectation is hand-derived:
independent recomputations mirror the RNG stream and the fidelity formulas
operator-by-operator without importing any other production symbol.
"""

from __future__ import annotations

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# make_child (UT-1201 / UT-1202)
# ---------------------------------------------------------------------------


def test_make_child_deterministic_and_recomputed() -> None:
    """UT-1201: same-seed calls agree bit-for-bit and match a hand-derived RNG stream.

    The recomputation drives its own ``torch.Generator`` through ``randperm``
    then ``randint`` in the production order; distinct seeds must yield
    distinguishable outputs (proves the seed is honored, not silently
    ignored).
    """
    from scripts.algo_experiments import make_child

    tokens = torch.arange(12, dtype=torch.long).reshape(1, 12)
    outputs: dict[int, tuple[torch.Tensor, list[int]]] = {}
    for seed in (7, 123):
        child_a, positions_a = make_child(tokens, vocab_size=10, num_flips=5, seed=seed)
        child_b, positions_b = make_child(tokens, vocab_size=10, num_flips=5, seed=seed)
        assert torch.equal(child_a, child_b)
        assert positions_a == positions_b

        generator = torch.Generator(device="cpu").manual_seed(seed)
        expected_positions = torch.randperm(12, generator=generator)[:5].tolist()
        expected_rand = torch.randint(0, 10, (5,), generator=generator, dtype=tokens.dtype)
        expected_child = tokens.clone()
        expected_child[0, torch.tensor(expected_positions)] = expected_rand
        assert positions_a == expected_positions
        assert torch.equal(child_a, expected_child)
        outputs[seed] = (child_a, positions_a)

    assert outputs[7][1] != outputs[123][1]
    assert not torch.equal(outputs[7][0], outputs[123][0])


def test_make_child_edge_cases() -> None:
    """UT-1202: early exit, clamping, full-length flips, vocab=1, immutability, dtype."""
    from scripts.algo_experiments import make_child

    tokens = torch.arange(12, dtype=torch.long).reshape(1, 12)
    snapshot = tokens.clone()

    child_zero, positions_zero = make_child(tokens, 10, num_flips=0, seed=5)
    assert positions_zero == []
    assert torch.equal(child_zero, tokens)
    assert child_zero is not tokens

    child_neg, positions_neg = make_child(tokens, 10, num_flips=-3, seed=5)
    assert positions_neg == []
    assert torch.equal(child_neg, tokens)
    assert child_neg is not tokens

    child_clamped, positions_clamped = make_child(tokens, 10, num_flips=15, seed=9)
    assert len(positions_clamped) == 12
    assert len(set(positions_clamped)) == 12
    assert all(0 <= p < 12 for p in positions_clamped)
    assert all(0 <= int(child_clamped[0, p]) < 10 for p in positions_clamped)

    child_full, positions_full = make_child(tokens, 10, num_flips=12, seed=9)
    assert len(positions_full) == 12
    assert len(set(positions_full)) == 12
    assert all(0 <= p < 12 for p in positions_full)

    child_vocab1, positions_vocab1 = make_child(tokens, 1, num_flips=5, seed=11)
    assert len(positions_vocab1) == 5
    assert all(int(child_vocab1[0, p]) == 0 for p in positions_vocab1)

    generator = torch.Generator(device="cpu").manual_seed(3)
    float_tokens = torch.randn(1, 12, generator=generator)
    float_snapshot = float_tokens.clone()
    child_float, positions_float = make_child(float_tokens, 10, num_flips=4, seed=13)
    assert child_float.dtype == float_tokens.dtype == torch.float32
    assert len(positions_float) == 4
    assert all(0.0 <= float(child_float[0, p]) < 10.0 for p in positions_float)

    assert torch.equal(tokens, snapshot)
    assert torch.equal(float_tokens, float_snapshot)


# ---------------------------------------------------------------------------
# fidelity (UT-1203 / UT-1204)
# ---------------------------------------------------------------------------

_FIDELITY_KEYS = {"mse", "rel_mse", "cosine", "top1_agreement"}


def test_fidelity_identity_floor_and_dtype() -> None:
    """UT-1203: exact identity metrics, finite rel_mse under the var floor, fp16==fp32.

    The identity tensor ``torch.ones(2, 2) * 2`` is hand-picked so every
    intermediate is exactly representable (norm^2 = 16, sqrt = 4, product =
    16, dot = 16) and cosine is bit-exact 1.0 rather than a norm-roundtrip
    lottery.  A constant ``base`` (var = 0) exercises the ``max(var, 1e-12)``
    denominator floor: rel_mse stays finite.  float16 inputs must equal the
    float32 results for identical values (the internal ``.float()`` upcast).
    """
    from scripts.algo_experiments import fidelity

    base = torch.ones(2, 2) * 2.0
    identity = fidelity(base, base)
    assert identity["mse"] == 0.0
    assert identity["rel_mse"] == 0.0
    assert identity["cosine"] == 1.0
    assert identity["top1_agreement"] == 1.0

    floored = fidelity(torch.ones(2, 2) * 3.0, base)
    assert math.isfinite(floored["rel_mse"])
    assert floored["rel_mse"] == pytest.approx(floored["mse"] / 1e-12)

    out16 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
    base16 = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float16)
    from_fp16 = fidelity(out16, base16)
    from_fp32 = fidelity(out16.float(), base16.float())
    assert from_fp16 == from_fp32

    assert set(identity) == _FIDELITY_KEYS
    assert set(floored) == _FIDELITY_KEYS
    assert set(from_fp16) == _FIDELITY_KEYS


def test_fidelity_hand_computed_example() -> None:
    """UT-1204: four metrics against first-principles hand arithmetic.

    ``out = [[1, 1], [1, 1]]`` vs ``base = [[1, 2], [1, 1]]``:
    mse = 1/4 = 0.25 (exact); unbiased var = 0.75/3 = 0.25 via the explicit
    mirror ``((base - mean)**2).sum() / (n - 1)`` — never ``Tensor.var()``,
    so a torch default flip cannot silently re-taint the expectation;
    rel_mse = 0.25/0.25 = 1.0; cosine = 5/(2*sqrt(7)) ~ 0.944911;
    top1 = 1/2 = 0.5 (exact).
    """
    from scripts.algo_experiments import fidelity

    out = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    base = torch.tensor([[1.0, 2.0], [1.0, 1.0]])

    metrics = fidelity(out, base)

    mean = float(base.sum() / base.numel())
    var_manual = float(((base - mean) ** 2).sum() / (base.numel() - 1))
    assert mean == 1.25
    assert var_manual == 0.25

    expected_rel_mse = 0.25 / max(var_manual, 1e-12)
    assert expected_rel_mse == 1.0

    assert metrics["mse"] == 0.25
    assert metrics["rel_mse"] == pytest.approx(expected_rel_mse, rel=1e-6)
    assert metrics["cosine"] == pytest.approx(5.0 / (2.0 * math.sqrt(7.0)), rel=1e-6)
    assert metrics["cosine"] == pytest.approx(0.944911, rel=1e-6)
    assert metrics["top1_agreement"] == 0.5
