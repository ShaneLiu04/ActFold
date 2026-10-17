"""Tests for actfold.utils.flops_counter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.models.base import DiffusionLLM
from actfold.utils.flops_counter import (
    DiffusionLLMFLOPs,
    count_diffusion_llm_flops,
    model_ffn_flops_kwargs,
)

_MOE_BASE: dict[str, Any] = {
    "num_layers": 4,
    "hidden_dim": 128,
    "num_heads": 4,
    "seq_len": 64,
    "vocab_size": 1000,
    "num_steps": 2,
    "reuse_ratio": 0.0,
    "ffn_type": "swiglu",
    "moe_num_experts": 8,
    "moe_top_k": 2,
    "moe_intermediate_dim": 256,
}


def _moe_kwargs(**overrides: Any) -> dict[str, Any]:
    """Build the UT-401 baseline MoE kwargs with overrides applied."""
    kwargs = dict(_MOE_BASE)
    kwargs.update(overrides)
    return kwargs


def _dense_kwargs(**overrides: Any) -> dict[str, Any]:
    """Build the dense baseline kwargs (MoE keys stripped) with overrides applied."""
    kwargs = {k: v for k, v in _MOE_BASE.items() if not k.startswith("moe_")}
    kwargs.update(overrides)
    return kwargs


def test_count_flops_baseline() -> None:
    flops = count_diffusion_llm_flops(
        num_layers=4,
        hidden_dim=128,
        num_heads=8,
        seq_len=16,
        vocab_size=1000,
        num_steps=1,
        reuse_ratio=0.0,
    )
    assert isinstance(flops, DiffusionLLMFLOPs)
    assert flops.total_tflops > 0.0
    assert flops.total_tflops == pytest.approx(
        flops.attention_tflops + flops.ffn_tflops + flops.embedding_tflops
    )


def test_count_flops_with_reuse() -> None:
    baseline = count_diffusion_llm_flops(
        num_layers=4,
        hidden_dim=128,
        num_heads=8,
        seq_len=16,
        vocab_size=1000,
        num_steps=1,
        reuse_ratio=0.0,
    )
    reused = count_diffusion_llm_flops(
        num_layers=4,
        hidden_dim=128,
        num_heads=8,
        seq_len=16,
        vocab_size=1000,
        num_steps=1,
        reuse_ratio=0.5,
    )
    assert reused.total_tflops < baseline.total_tflops


def test_count_flops_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(
            num_layers=0,
            hidden_dim=128,
            num_heads=8,
            seq_len=16,
            vocab_size=1000,
            num_steps=1,
        )


def test_count_flops_invalid_reuse_ratio() -> None:
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(
            num_layers=4,
            hidden_dim=128,
            num_heads=8,
            seq_len=16,
            vocab_size=1000,
            num_steps=1,
            reuse_ratio=1.5,
        )


def test_count_flops_head_divisibility() -> None:
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(
            num_layers=4,
            hidden_dim=128,
            num_heads=7,
            seq_len=16,
            vocab_size=1000,
            num_steps=1,
        )


def test_count_flops_moe_per_token_formula() -> None:
    """UT-401: MoE per-token expert FLOPs match the hand-computed formula.

    L=4, h=128, T=64, V=1000, steps=2, swiglu, top_k=2, moe_inter=256,
    reuse=0: ffn_tflops == 2*3*256*128*2*4*64*2/1e12, attention and
    embedding match their dense formulas, and total equals the sum of the
    three components.
    """
    flops = count_diffusion_llm_flops(**_moe_kwargs())
    assert flops.ffn_tflops == 2 * 3 * 256 * 128 * 2 * 4 * 64 * 2 / 1e12
    assert flops.attention_tflops == 4 * 4 * 128 * 128 * 64 * 2 / 1e12
    assert flops.embedding_tflops == 1000 * 128 * 64 * 2 / 1e12
    assert flops.total_tflops == pytest.approx(
        flops.attention_tflops + flops.ffn_tflops + flops.embedding_tflops
    )


def test_count_flops_moe_shared_expert() -> None:
    """UT-402: shared expert adds +1 to the per-token expert coefficient.

    moe_shared_expert=True with top_k=2 gives coefficient (2+1); with
    moe_top_k=None the flag does not activate MoE and the output stays
    bit-identical to the pure dense output.
    """
    flops = count_diffusion_llm_flops(**_moe_kwargs(moe_shared_expert=True))
    assert flops.ffn_tflops == 2 * 3 * 256 * 128 * (2 + 1) * 4 * 64 * 2 / 1e12
    dense = count_diffusion_llm_flops(**_dense_kwargs())
    inactive = count_diffusion_llm_flops(**_dense_kwargs(moe_shared_expert=True))
    assert inactive == dense


def test_count_flops_moe_dense_mixed_layers() -> None:
    """UT-403: mixed MoE/dense layer counts.

    moe_num_layers=2 of 4: FFN equals 2 MoE expert layers (moe_inter=256)
    plus 2 dense layers (ffn_intermediate_dim=384, same swiglu topology);
    moe_num_layers=0 is bit-identical to pure dense; moe_num_layers ==
    num_layers is full MoE (identical to the default L_moe).
    """
    mixed = count_diffusion_llm_flops(**_moe_kwargs(moe_num_layers=2, ffn_intermediate_dim=384))
    moe_part = 2 * 3 * 256 * 128 * 2 * 2 * 64
    dense_part = 2 * 3 * 384 * 128 * 2 * 64
    assert mixed.ffn_tflops == (moe_part + dense_part) * 2 / 1e12
    zero_moe = count_diffusion_llm_flops(**_moe_kwargs(moe_num_layers=0, ffn_intermediate_dim=384))
    dense_ref = count_diffusion_llm_flops(**_dense_kwargs(ffn_intermediate_dim=384))
    assert zero_moe == dense_ref
    full_moe = count_diffusion_llm_flops(**_moe_kwargs(moe_num_layers=4))
    default_layers = count_diffusion_llm_flops(**_moe_kwargs())
    assert full_moe == default_layers


def test_count_flops_moe_reuse_ratio() -> None:
    """UT-404: reuse_ratio semantics are unchanged under MoE.

    With reuse_ratio=0.5 the effective sequence length halves, so the MoE
    FFN component is exactly half the reuse=0 value, attention is exactly
    half (existing behavior), and embedding is unchanged.
    """
    baseline = count_diffusion_llm_flops(**_moe_kwargs())
    reused = count_diffusion_llm_flops(**_moe_kwargs(reuse_ratio=0.5))
    assert reused.ffn_tflops == baseline.ffn_tflops / 2
    assert reused.attention_tflops == baseline.attention_tflops / 2
    assert reused.embedding_tflops == baseline.embedding_tflops


@pytest.mark.parametrize("ffn_type", ["mlp", "swiglu"])
@pytest.mark.parametrize("ffn_intermediate_dim", [None, 384])
def test_count_flops_default_dense_regression(
    ffn_type: str, ffn_intermediate_dim: int | None
) -> None:
    """UT-405: without moe_top_k the output matches the dense formula exactly.

    Covers ffn_type in {"mlp", "swiglu"} x ffn_intermediate_dim in
    {None, 384}: ffn_tflops == 2*n_matmul*intermediate*h*L*T*steps/1e12
    with intermediate = ffn_intermediate_dim or 4*h.
    """
    flops = count_diffusion_llm_flops(
        num_layers=4,
        hidden_dim=128,
        num_heads=4,
        seq_len=64,
        vocab_size=1000,
        num_steps=2,
        ffn_intermediate_dim=ffn_intermediate_dim,
        ffn_type=ffn_type,
    )
    n_matmul = 3 if ffn_type == "swiglu" else 2
    intermediate = 4 * 128 if ffn_intermediate_dim is None else ffn_intermediate_dim
    assert flops.ffn_tflops == 2 * n_matmul * intermediate * 128 * 4 * 64 * 2 / 1e12


def test_count_flops_moe_intermediate_fallback() -> None:
    """UT-406: moe_inter falls back to ffn_intermediate_dim, then 4*h.

    moe_intermediate_dim=None with ffn_intermediate_dim=192 uses 192 for
    the MoE segment; both None uses 4*hidden_dim.
    """
    from_ffn = count_diffusion_llm_flops(
        **_moe_kwargs(moe_intermediate_dim=None, ffn_intermediate_dim=192)
    )
    assert from_ffn.ffn_tflops == 2 * 3 * 192 * 128 * 2 * 4 * 64 * 2 / 1e12
    from_4h = count_diffusion_llm_flops(
        **_moe_kwargs(moe_intermediate_dim=None, ffn_intermediate_dim=None)
    )
    assert from_4h.ffn_tflops == 2 * 3 * (4 * 128) * 128 * 2 * 4 * 64 * 2 / 1e12


@pytest.mark.parametrize("moe_top_k", [0, -1])
def test_count_flops_moe_top_k_nonpositive_raises(moe_top_k: int) -> None:
    """EX-701: moe_top_k <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(**_moe_kwargs(moe_top_k=moe_top_k))


def test_count_flops_moe_num_experts_validation() -> None:
    """EX-702: expert-count domain validation raises ValueError.

    moe_top_k > moe_num_experts, moe_num_experts == 0, and
    moe_num_experts < 0 all raise when moe_top_k is given.
    """
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(**_moe_kwargs(moe_top_k=4, moe_num_experts=2))
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(**_moe_kwargs(moe_num_experts=0))
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(**_moe_kwargs(moe_num_experts=-1))


@pytest.mark.parametrize("moe_intermediate_dim", [0, -5])
def test_count_flops_moe_intermediate_nonpositive_raises(moe_intermediate_dim: int) -> None:
    """EX-703: moe_intermediate_dim <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(**_moe_kwargs(moe_intermediate_dim=moe_intermediate_dim))


def test_count_flops_moe_num_layers_out_of_range() -> None:
    """EX-704: moe_num_layers outside [0, num_layers] raises ValueError.

    num_layers is 4 in the baseline, so -1 and 5 (num_layers + 1) are
    both out of range.
    """
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(**_moe_kwargs(moe_num_layers=-1))
    with pytest.raises(ValueError):
        count_diffusion_llm_flops(**_moe_kwargs(moe_num_layers=5))


def test_count_flops_moe_half_config_inactive() -> None:
    """EX-705: moe_num_experts alone (top_k None) does not activate MoE.

    No error is raised and the output is bit-identical to the dense
    baseline (trigger-key semantics).
    """
    dense = count_diffusion_llm_flops(**_dense_kwargs())
    half = count_diffusion_llm_flops(**_dense_kwargs(moe_num_experts=8))
    assert half == dense


# ---------------------------------------------------------------------------
# AR005 T002: FFN/MoE geometry extraction and helper-chain resolution
# (UT-407/409/410).  The private helpers are imported lazily inside each
# test so the T002 Red state (helpers not implemented yet) fails only the
# new tests while the existing suite stays green.
# ---------------------------------------------------------------------------


def _make_full_config() -> SimpleNamespace:
    """Build a config-like object carrying the full primary-name geometry.

    Returns:
        SimpleNamespace with a swiglu FFN (``intermediate_size=12288``),
        8 routed experts (top-2, expert intermediate 1536), one shared
        expert (intermediate 2048), and 32 layers of which the first 3 are
        dense (``first_k_dense_replace=3`` -> 29 MoE layers).
    """
    return SimpleNamespace(
        intermediate_size=12288,
        hidden_act="silu",
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=1536,
        shared_expert_intermediate_size=2048,
        num_hidden_layers=32,
        first_k_dense_replace=3,
    )


class _GeometryDiffusionStub(DiffusionLLM):
    """Concrete ``DiffusionLLM`` stub whose geometry lives on ``model.config``.

    Implements only the abstract members (UT-409); FFN/MoE geometry is
    reachable exclusively through ``self.model.config``.
    """

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def num_layers(self) -> int:
        return 2

    @property
    def hidden_dim(self) -> int:
        return 16

    @property
    def num_heads(self) -> int:
        return 2

    @property
    def vocab_size(self) -> int:
        return 100


class _UnderlyingModelAdapter:
    """Adapter stub that only exposes ``underlying_model`` (UT-409a/409c)."""

    def __init__(self, underlying: Any) -> None:
        self._underlying = underlying

    @property
    def underlying_model(self) -> Any:
        """Return the wrapped model, hiding all geometry behind the chain."""
        return self._underlying


class _RawConfigModule(nn.Module):
    """HF-style raw module exposing ``.config`` directly (UT-409e)."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config


@pytest.mark.parametrize("hidden_act", ["silu", "swish", "swiglu"])
def test_extract_ffn_geometry_swiglu_family(hidden_act: str) -> None:
    """UT-407: hidden_act in {silu, swish, swiglu} maps to ffn_type "swiglu".

    Also checks that ``ffn_intermediate_dim`` is read from
    ``intermediate_size``.
    """
    from actfold.utils.flops_counter import _extract_ffn_geometry

    geometry = _extract_ffn_geometry(
        SimpleNamespace(intermediate_size=11008, hidden_act=hidden_act)
    )
    assert geometry["ffn_type"] == "swiglu"
    assert geometry["ffn_intermediate_dim"] == 11008


@pytest.mark.parametrize("hidden_act", ["gelu", "gelu_new", "relu", "quickgelu"])
def test_extract_ffn_geometry_mlp_family(hidden_act: str) -> None:
    """UT-407: gelu/gelu_new/relu and unknown activations map to ffn_type "mlp"."""
    from actfold.utils.flops_counter import _extract_ffn_geometry

    geometry = _extract_ffn_geometry(
        SimpleNamespace(intermediate_size=11008, hidden_act=hidden_act)
    )
    assert geometry["ffn_type"] == "mlp"


def test_extract_ffn_geometry_missing_hidden_act() -> None:
    """UT-407: a config without hidden_act defaults to "mlp" (original semantics)."""
    from actfold.utils.flops_counter import _extract_ffn_geometry

    geometry = _extract_ffn_geometry(SimpleNamespace(intermediate_size=512))
    assert geometry["ffn_type"] == "mlp"
    assert geometry["ffn_intermediate_dim"] == 512


def test_model_ffn_flops_kwargs_drills_underlying_model() -> None:
    """UT-409a: geometry resolves through ``underlying_model`` to model.config.

    The adapter exposes nothing but ``underlying_model``; the wrapped
    DiffusionLLM stub holds the geometry only on ``model.config``.  All 7
    keys must come back with real values.
    """
    stub = _GeometryDiffusionStub("dummy")
    stub.model = SimpleNamespace(config=_make_full_config())
    kwargs = model_ffn_flops_kwargs(_UnderlyingModelAdapter(stub))
    assert kwargs["ffn_intermediate_dim"] == 12288
    assert kwargs["ffn_type"] == "swiglu"
    assert kwargs["moe_num_experts"] == 8
    assert kwargs["moe_top_k"] == 2
    assert kwargs["moe_intermediate_dim"] == 1536
    assert kwargs["moe_shared_expert"] is True
    assert kwargs["moe_num_layers"] == 29


def test_model_ffn_flops_kwargs_bare_module_defaults() -> None:
    """UT-409b: a bare nn.Module keeps the current default geometry (zero change)."""
    kwargs = model_ffn_flops_kwargs(nn.Module())
    assert kwargs["ffn_intermediate_dim"] is None
    assert kwargs["ffn_type"] == "mlp"
    assert kwargs["moe_num_experts"] is None
    assert kwargs["moe_top_k"] is None
    assert kwargs["moe_intermediate_dim"] is None
    assert kwargs["moe_shared_expert"] is False
    assert kwargs["moe_num_layers"] is None


def test_model_ffn_flops_kwargs_nested_wrappers() -> None:
    """UT-409c: two levels of ``underlying_model`` wrapping are drilled through."""
    stub = _GeometryDiffusionStub("dummy")
    stub.model = SimpleNamespace(config=_make_full_config())
    inner = _UnderlyingModelAdapter(stub)
    outer = _UnderlyingModelAdapter(inner)
    kwargs = model_ffn_flops_kwargs(outer)
    assert kwargs["ffn_intermediate_dim"] == 12288
    assert kwargs["ffn_type"] == "swiglu"
    assert kwargs["moe_num_experts"] == 8
    assert kwargs["moe_top_k"] == 2
    assert kwargs["moe_intermediate_dim"] == 1536
    assert kwargs["moe_shared_expert"] is True
    assert kwargs["moe_num_layers"] == 29


def test_extract_ffn_geometry_alias_union() -> None:
    """UT-409d: DeepSeek-style alias names resolve; primary names win when both exist.

    Alias-only config (n_routed_experts/num_selected_experts/
    expert_intermediate_size) maps to the canonical keys; when a primary
    name and its alias are both present the primary value is used.
    """
    from actfold.utils.flops_counter import _extract_ffn_geometry

    aliased = _extract_ffn_geometry(
        SimpleNamespace(
            n_routed_experts=64,
            num_selected_experts=6,
            expert_intermediate_size=1408,
        )
    )
    assert aliased["moe_num_experts"] == 64
    assert aliased["moe_top_k"] == 6
    assert aliased["moe_intermediate_dim"] == 1408

    both = _extract_ffn_geometry(
        SimpleNamespace(
            num_experts=8,
            n_routed_experts=64,
            num_experts_per_tok=2,
            num_selected_experts=6,
            moe_intermediate_size=1536,
            expert_intermediate_size=1408,
        )
    )
    assert both["moe_num_experts"] == 8
    assert both["moe_top_k"] == 2
    assert both["moe_intermediate_dim"] == 1536


def test_model_ffn_flops_kwargs_raw_config_module() -> None:
    """UT-409e: an HF-style module exposing ``.config`` directly is read as-is."""
    kwargs = model_ffn_flops_kwargs(_RawConfigModule(_make_full_config()))
    assert kwargs["ffn_intermediate_dim"] == 12288
    assert kwargs["ffn_type"] == "swiglu"
    assert kwargs["moe_num_experts"] == 8
    assert kwargs["moe_top_k"] == 2
    assert kwargs["moe_intermediate_dim"] == 1536
    assert kwargs["moe_shared_expert"] is True
    assert kwargs["moe_num_layers"] == 29


def test_model_ffn_flops_kwargs_keys_additive() -> None:
    """UT-410: the kwargs dict has exactly the 7 geometry keys (additive only).

    On an attribute-less object the two original keys keep their legacy
    semantics (``ffn_intermediate_dim`` None, ``ffn_type`` "mlp") and the
    helper returns the same 7-key set.
    """
    from actfold.utils.flops_counter import _extract_ffn_geometry

    expected_keys = {
        "ffn_intermediate_dim",
        "ffn_type",
        "moe_num_experts",
        "moe_top_k",
        "moe_intermediate_dim",
        "moe_shared_expert",
        "moe_num_layers",
    }
    kwargs = model_ffn_flops_kwargs(SimpleNamespace())
    assert set(kwargs) == expected_keys
    assert kwargs["ffn_intermediate_dim"] is None
    assert kwargs["ffn_type"] == "mlp"

    geometry = _extract_ffn_geometry(SimpleNamespace())
    assert set(geometry) == expected_keys
    assert geometry["ffn_intermediate_dim"] is None
    assert geometry["ffn_type"] == "mlp"
