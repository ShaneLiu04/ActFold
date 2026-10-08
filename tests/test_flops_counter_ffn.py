"""Red-phase tests for T022: FFN params, SwiGLU correction, embedding fix in flops_counter.

These tests define the Green contract for ``count_diffusion_llm_flops``:
- ``ffn_intermediate_dim`` / ``ffn_type`` / ``include_attention_t2`` kwargs.
- Embedding FLOPs halved to the LM-head output projection only.
- Duck-typed ``model_ffn_flops_kwargs`` helper.

Red phase: the new kwargs and helper do not exist yet, so every test below
must fail until production code is updated.
"""

from __future__ import annotations

import types

import pytest

# Canonical parameter set shared by all tests.
L = 4
H = 128
HEADS = 8
T = 16
V = 1000
STEPS = 1


def _call(**overrides: object):
    from actfold.utils.flops_counter import count_diffusion_llm_flops

    kwargs: dict[str, object] = {
        "num_layers": L,
        "hidden_dim": H,
        "num_heads": HEADS,
        "seq_len": T,
        "vocab_size": V,
        "num_steps": STEPS,
    }
    kwargs.update(overrides)
    return count_diffusion_llm_flops(**kwargs)  # type: ignore[arg-type]


def test_default_mlp_matches_old_attention_and_ffn() -> None:
    flops = _call()
    assert flops.attention_tflops == pytest.approx(4 * L * H * H * T / 1e12)
    assert flops.ffn_tflops == pytest.approx(16 * L * H * H * T / 1e12)
    # Corrected: LM-head output projection only (halved vs old 2 * V * h * T).
    assert flops.embedding_tflops == pytest.approx(1 * V * H * T / 1e12)


def test_swiglu_ffn_formula() -> None:
    flops = _call(ffn_type="swiglu", ffn_intermediate_dim=432)
    assert flops.ffn_tflops == pytest.approx(2 * 3 * 432 * H * L * T / 1e12)


def test_mlp_custom_intermediate() -> None:
    flops = _call(ffn_type="mlp", ffn_intermediate_dim=512)
    assert flops.ffn_tflops == pytest.approx(2 * 2 * 512 * H * L * T / 1e12)


def test_none_intermediate_defaults_to_4h() -> None:
    flops = _call(ffn_type="swiglu", ffn_intermediate_dim=None)
    assert flops.ffn_tflops == pytest.approx(2 * 3 * (4 * H) * H * L * T / 1e12)


def test_embedding_uses_full_seq_len_with_reuse() -> None:
    flops = _call(reuse_ratio=0.5)
    # LM head runs on all positions regardless of reuse.
    assert flops.embedding_tflops == pytest.approx(1 * V * H * T / 1e12)
    # Attention/FFN use T_eff = T * (1 - reuse) = 8.
    assert flops.attention_tflops == pytest.approx(4 * L * H * H * 8 / 1e12)
    assert flops.ffn_tflops == pytest.approx(16 * L * H * H * 8 / 1e12)


def test_attention_t2_term_optional() -> None:
    base = _call(include_attention_t2=False)
    assert base.attention_tflops == pytest.approx(4 * L * H * H * T / 1e12)

    with_t2 = _call(include_attention_t2=True)
    assert with_t2.attention_tflops == pytest.approx(
        (4 * L * H * H * T + 2 * L * T * T * H) / 1e12
    )

    with_reuse = _call(include_attention_t2=True, reuse_ratio=0.5)
    assert with_reuse.attention_tflops == pytest.approx(
        (4 * L * H * H * 8 + 2 * L * 8 * T * H) / 1e12
    )


def test_invalid_ffn_type_raises() -> None:
    with pytest.raises(ValueError):
        _call(ffn_type="gelu")


def test_nonpositive_intermediate_dim_raises() -> None:
    with pytest.raises(ValueError):
        _call(ffn_intermediate_dim=0)


def test_total_sums_components() -> None:
    flops = _call(ffn_type="swiglu", ffn_intermediate_dim=432, reuse_ratio=0.3,
                  include_attention_t2=True)
    assert flops.total_tflops == pytest.approx(
        flops.attention_tflops + flops.ffn_tflops + flops.embedding_tflops
    )


def test_model_ffn_flops_kwargs_duck_typed() -> None:
    from actfold.utils.flops_counter import model_ffn_flops_kwargs

    # (a) Plain object without FFN attributes -> defaults.
    plain = types.SimpleNamespace(hidden_dim=H)
    kwargs = model_ffn_flops_kwargs(plain)
    assert kwargs == {"ffn_intermediate_dim": None, "ffn_type": "mlp"}

    # (b) Object exposing FFN shape -> extracted values.
    shaped = types.SimpleNamespace(ffn_intermediate_dim=11008, ffn_type="swiglu")
    kwargs = model_ffn_flops_kwargs(shaped)
    assert kwargs == {"ffn_intermediate_dim": 11008, "ffn_type": "swiglu"}


def test_reuse_reduces_total() -> None:
    baseline = _call(ffn_type="swiglu", include_attention_t2=True, reuse_ratio=0.0)
    reused = _call(ffn_type="swiglu", include_attention_t2=True, reuse_ratio=0.5)
    assert reused.total_tflops < baseline.total_tflops
