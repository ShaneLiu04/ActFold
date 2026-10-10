"""Tests for actfold.speculative.verification_engine."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.models.base import DiffusionLLM
from actfold.speculative import ActFoldVerificationEngine, FastDLLMAdapter
from actfold.speculative.branch import Branch
from actfold.utils.flops_counter import count_diffusion_llm_flops, model_ffn_flops_kwargs


class TinyModel(nn.Module):
    """Minimal embedding + head model for cache-population tests."""

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(tokens))


def test_t014_verification_engine_embedding_key() -> None:
    """T014: ``_ensure_parent_cache`` stores layer-0 embeddings as "embedding".

    The layer-0 entry must be readable under the name ``"embedding"`` and must
    not contain ``"hidden_states"`` or ``"ffn_out"``.
    """
    vocab_size = 16
    hidden_dim = 8
    model = TinyModel(vocab_size, hidden_dim)
    adapter = FastDLLMAdapter(model, num_layers=1, hidden_dim=hidden_dim)
    cache = ActivationCache(device="cpu")
    gate = SimilarityGate(tau=0.95)
    engine = ActFoldVerificationEngine(adapter, cache, gate)

    tokens = torch.randint(0, vocab_size, (1, 4))
    parent = Branch(branch_id="root", parent_id=None, tokens=tokens)
    engine._ensure_parent_cache(parent)

    full_mask = torch.ones(1, 4, dtype=torch.bool)
    activations = cache.get("root", layer_idx=0, token_mask=full_mask)
    assert "embedding" in activations
    assert "hidden_states" not in activations
    assert "ffn_out" not in activations
    assert activations["embedding"].shape == (1, 4, hidden_dim)
    assert torch.allclose(activations["embedding"], model.embedding(tokens))


def test_t015_ensure_parent_cache_uses_contains(monkeypatch: pytest.MonkeyPatch) -> None:
    """T015: ``_ensure_parent_cache`` checks presence via ``contains()``.

    ``cache.get`` is monkeypatched to raise so the old try/get/except-KeyError
    control flow cannot run. When ``contains`` returns False the engine must
    ``put`` the layer-0 ``{"embedding": ...}`` entry exactly once; when it
    returns True it must neither ``put`` nor raise.
    """
    vocab_size = 16
    hidden_dim = 8
    model = TinyModel(vocab_size, hidden_dim)
    adapter = FastDLLMAdapter(model, num_layers=1, hidden_dim=hidden_dim)
    cache = ActivationCache(device="cpu")
    gate = SimilarityGate(tau=0.95)
    engine = ActFoldVerificationEngine(adapter, cache, gate)

    tokens = torch.randint(0, vocab_size, (1, 4))
    parent = Branch(branch_id="root", parent_id=None, tokens=tokens)
    hidden = model.embedding(tokens)

    contains_calls: list[tuple[str, int, int]] = []
    put_calls: list[tuple[str, int, dict[str, torch.Tensor]]] = []

    def _contains_false(branch_id: str, layer_idx: int, step_idx: int = 0) -> bool:
        contains_calls.append((branch_id, layer_idx, step_idx))
        return False

    def _contains_true(branch_id: str, layer_idx: int, step_idx: int = 0) -> bool:
        contains_calls.append((branch_id, layer_idx, step_idx))
        return True

    real_put = cache.put

    def _record_put(
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
        step_idx: int = 0,
    ) -> None:
        put_calls.append((branch_id, layer_idx, dict(activations)))
        real_put(branch_id, layer_idx, activations, step_idx=step_idx)

    def _raise_get(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy cache.get used by _ensure_parent_cache")

    monkeypatch.setattr(cache, "get", _raise_get)
    monkeypatch.setattr(cache, "put", _record_put)
    monkeypatch.setattr(cache, "contains", _contains_false, raising=False)

    # contains -> False: exactly one put of the real embeddings.
    engine._ensure_parent_cache(parent)

    assert ("root", 0, 0) in contains_calls
    assert len(put_calls) == 1
    branch_id, layer_idx, activations = put_calls[0]
    assert branch_id == "root"
    assert layer_idx == 0
    assert set(activations.keys()) == {"embedding"}
    assert torch.allclose(activations["embedding"], hidden)

    # contains -> True: no put, and no raise despite get being patched.
    monkeypatch.setattr(cache, "contains", _contains_true, raising=False)
    contains_calls.clear()
    put_calls.clear()

    engine._ensure_parent_cache(parent)

    assert ("root", 0, 0) in contains_calls
    assert put_calls == []


class FixedLogitsModel(nn.Module):
    """Stub model whose forward returns precomputed logits.

    The verification engine only needs an ``embedding`` for ``adapter.embed``
    and a ``[batch, seq_len, vocab]`` logits output whose per-position argmax
    can be controlled exactly by the test.
    """

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.logits: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.logits is None:
            raise AssertionError("FixedLogitsModel.logits must be set before forward")
        return self.logits


def test_ut306_verify_branch_reports_acceptance_metrics() -> None:
    """UT-306: verify_branch computes and reports acceptance metrics (SRS 3.2-1/3.2-3).

    The stub adapter returns logits with a controlled per-position argmax, so
    the draft region (positions where child differs from parent) and every
    metric are hand-computable:

    - draft region = positions {2, 4}; only position 2 is accepted, so the
      acceptance rate is exactly 0.5;
    - ``mean_log_prob`` equals the fp32 log_softmax gather mean over the draft
      region and replaces ``logits.mean()`` as ``actfold_score``;
    - ``ema_acceptance_rate`` initializes to the first call's rate (T002).
    """

    vocab_size = 8
    hidden_dim = 8
    model = FixedLogitsModel(vocab_size, hidden_dim)
    adapter = FastDLLMAdapter(model, num_layers=1, hidden_dim=hidden_dim)
    cache = ActivationCache(device="cpu")
    gate = SimilarityGate(tau=0.95)
    engine = ActFoldVerificationEngine(adapter, cache, gate)

    parent_tokens = torch.tensor([[1, 2, 3, 4, 5]])
    child_tokens = torch.tensor([[1, 2, 0, 4, 7]])
    draft_mask = torch.tensor([[False, False, True, False, True]])
    argmax_tokens = torch.tensor([[1, 2, 0, 7, 4]])
    logits = torch.zeros(1, 5, vocab_size)
    logits.scatter_(-1, argmax_tokens.unsqueeze(-1), 10.0)
    model.logits = logits

    parent = Branch(branch_id="root", parent_id=None, tokens=parent_tokens)
    child = Branch(branch_id="child", parent_id="root", tokens=child_tokens)

    result = engine.verify_branch(parent, child, step_idx=0)

    assert result.acceptance_rate == pytest.approx(0.5)
    assert result.ema_acceptance_rate == pytest.approx(0.5)

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, child_tokens.unsqueeze(-1)).squeeze(-1)
    expected_mlp = token_log_probs[draft_mask].mean().item()
    assert result.mean_log_prob == pytest.approx(expected_mlp, abs=1e-6)

    assert child.metadata["actfold_score"] == result.mean_log_prob
    assert child.metadata["acceptance_rate"] == result.acceptance_rate
    assert child.metadata["stable_ratio"] == result.stable_ratio
    assert 0.0 <= result.stable_ratio <= 1.0
    # Default threshold 0.0 accepts every branch (baseline behavior).
    assert result.accepted is True


# ---------------------------------------------------------------------------
# AR004 T002: EMA acceptance-rate tracking + acceptance-based decision switch.
# ---------------------------------------------------------------------------


def _make_fixed_logits_engine(
    vocab_size: int = 8,
    hidden_dim: int = 8,
    **engine_kwargs: float,
) -> tuple[FixedLogitsModel, ActivationCache, ActFoldVerificationEngine]:
    """Build a verification engine around a :class:`FixedLogitsModel` stub.

    Args:
        vocab_size: Stub vocabulary size (bounds usable argmax token ids).
        hidden_dim: Stub embedding dimension.
        engine_kwargs: Extra keyword arguments forwarded to
            ``ActFoldVerificationEngine`` (e.g. ``acceptance_threshold``,
            ``ema_alpha``).

    Returns:
        A ``(model, cache, engine)`` triple; assign ``model.logits`` through
        :func:`_set_argmax` before invoking ``engine.verify_branch``.
    """
    model = FixedLogitsModel(vocab_size, hidden_dim)
    adapter = FastDLLMAdapter(model, num_layers=1, hidden_dim=hidden_dim)
    cache = ActivationCache(device="cpu")
    gate = SimilarityGate(tau=0.95)
    engine = ActFoldVerificationEngine(adapter, cache, gate, **engine_kwargs)
    return model, cache, engine


def _set_argmax(
    model: FixedLogitsModel,
    argmax_tokens: torch.Tensor,
    vocab_size: int,
) -> None:
    """Point the stub's per-position logits argmax at ``argmax_tokens``.

    Args:
        model: Stub whose ``logits`` should be overwritten.
        argmax_tokens: ``[batch, seq_len]`` desired per-position argmax ids.
        vocab_size: Logits vocabulary dimension.
    """
    batch, seq_len = argmax_tokens.shape
    logits = torch.zeros(batch, seq_len, vocab_size)
    logits.scatter_(-1, argmax_tokens.unsqueeze(-1), 10.0)
    model.logits = logits


def test_ut307_ema_initializes_to_first_rate() -> None:
    """UT-307 (SRS 3.3-1): the first verify sets the EMA to that call's rate.

    A freshly constructed engine reports the idle EMA ``0.0``; after the
    first ``verify_branch`` both ``engine.ema_acceptance_rate`` and
    ``VerificationResult.ema_acceptance_rate`` equal the observed rate
    (no ``alpha * 0`` warm-up blend on the first call).
    """
    model, _, engine = _make_fixed_logits_engine()
    assert engine.ema_acceptance_rate == pytest.approx(0.0)

    parent = Branch(branch_id="root", parent_id=None, tokens=torch.tensor([[1, 2, 3, 4, 5]]))
    child = Branch(branch_id="child", parent_id="root", tokens=torch.tensor([[1, 2, 0, 4, 7]]))
    # Draft region {2, 4}; only position 2 matches the target argmax -> rate 0.5.
    _set_argmax(model, torch.tensor([[1, 2, 0, 7, 4]]), vocab_size=8)

    result = engine.verify_branch(parent, child, step_idx=0)

    assert result.acceptance_rate == pytest.approx(0.5)
    assert engine.ema_acceptance_rate == pytest.approx(0.5)
    assert result.ema_acceptance_rate == pytest.approx(0.5)


def test_ut307_ema_chains_with_alpha_blend() -> None:
    """UT-307 (SRS 3.3-2): consecutive verifies chain alpha * r + (1 - alpha) * EMA.

    With ``ema_alpha = 0.25`` and hand-computed rates ``r1 = 0.5``,
    ``r2 = 1.0``, ``r3 = 0.0``:

    - after the first verify the EMA is ``r1 = 0.5`` (initialization);
    - after the second it is ``0.25 * 1.0 + 0.75 * 0.5 = 0.625``;
    - after the third it is ``0.25 * 0.0 + 0.75 * 0.625 = 0.46875``.
    """
    alpha = 0.25
    model, _, engine = _make_fixed_logits_engine(ema_alpha=alpha)

    parent = Branch(branch_id="root", parent_id=None, tokens=torch.tensor([[1, 2, 3, 4, 5]]))
    child_tokens = torch.tensor([[1, 2, 0, 4, 7]])

    # (branch suffix, argmax row, expected rate, expected EMA afterwards)
    cases = [
        ("c1", [1, 2, 0, 7, 4], 0.5, 0.5),
        ("c2", [1, 2, 0, 4, 7], 1.0, 0.625),
        ("c3", [1, 2, 5, 4, 5], 0.0, 0.46875),
    ]
    for suffix, argmax_row, expected_rate, expected_ema in cases:
        child = Branch(branch_id=f"child-{suffix}", parent_id="root", tokens=child_tokens)
        _set_argmax(model, torch.tensor([argmax_row]), vocab_size=8)
        result = engine.verify_branch(parent, child, step_idx=0)
        assert result.acceptance_rate == pytest.approx(expected_rate)
        assert engine.ema_acceptance_rate == pytest.approx(expected_ema)
        assert result.ema_acceptance_rate == pytest.approx(expected_ema)


def test_ut307_ema_alpha_out_of_domain_raises_value_error() -> None:
    """UT-307 / EX-603 (SRS 3.3-3): ema_alpha outside (0, 1] fails construction.

    ``0``, negative, and ``> 1`` values must raise ``ValueError`` from the
    engine constructor (never be silently clamped).
    """
    for bad_alpha in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            _make_fixed_logits_engine(ema_alpha=bad_alpha)


def test_ut307_ema_alpha_one_degenerates_to_instantaneous_rate() -> None:
    """UT-307 (SRS 3.3-3): ema_alpha = 1.0 is legal and tracks the latest rate.

    ``ema_alpha = 1.0`` is the degenerate instantaneous case: the EMA equals
    the rate of the most recent verify, and a new engine still reports the
    idle value ``0.0`` before any call.
    """
    model, _, engine = _make_fixed_logits_engine(ema_alpha=1.0)
    assert engine.ema_acceptance_rate == pytest.approx(0.0)

    parent = Branch(branch_id="root", parent_id=None, tokens=torch.tensor([[1, 2, 3, 4, 5]]))
    child_tokens = torch.tensor([[1, 2, 0, 4, 7]])

    # Draft region {2, 4}; only position 2 matches -> rate 0.5.
    _set_argmax(model, torch.tensor([[1, 2, 0, 7, 4]]), vocab_size=8)
    child_a = Branch(branch_id="child-a", parent_id="root", tokens=child_tokens)
    engine.verify_branch(parent, child_a, step_idx=0)
    assert engine.ema_acceptance_rate == pytest.approx(0.5)

    # Draft region {2, 4}; both positions match -> rate 1.0.
    _set_argmax(model, torch.tensor([[1, 2, 0, 4, 7]]), vocab_size=8)
    child_b = Branch(branch_id="child-b", parent_id="root", tokens=child_tokens)
    result = engine.verify_branch(parent, child_b, step_idx=0)
    assert engine.ema_acceptance_rate == pytest.approx(1.0)
    assert result.ema_acceptance_rate == pytest.approx(1.0)


def test_ut308_default_threshold_keeps_baseline_behavior() -> None:
    """UT-308 (SRS 3.4-1): threshold 0.0 accepts every rate in [0, 1].

    A child whose draft region is fully rejected (rate = 0.0) is still
    accepted under the default ``acceptance_threshold = 0.0`` — the decision
    switch introduces zero behavior change at the default threshold.
    """
    model, _, engine = _make_fixed_logits_engine()

    parent = Branch(branch_id="root", parent_id=None, tokens=torch.tensor([[1, 2, 3, 4, 5]]))
    child = Branch(branch_id="child", parent_id="root", tokens=torch.tensor([[1, 2, 0, 4, 7]]))
    # Draft region {2, 4}; neither position matches the target argmax -> rate 0.0.
    _set_argmax(model, torch.tensor([[1, 2, 5, 4, 5]]), vocab_size=8)

    result = engine.verify_branch(parent, child, step_idx=0)

    assert result.acceptance_rate == pytest.approx(0.0)
    assert result.accepted is True


def test_ut308_below_threshold_rejects_and_clears_child_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-308 (SRS 3.4-2): rate < threshold rejects and evicts the child cache.

    Regression guard for the decision switch: ``stable_ratio`` is forced high
    (0.9 >= 0.8) while ``acceptance_rate = 0.5 < 0.8``, so only the
    acceptance-rate semantics can reject the branch. The rejected child's
    cache entry must be cleared (``contains`` flips to ``False``) while
    ``stable_ratio`` keeps being reported in metadata unchanged.
    """
    model, cache, engine = _make_fixed_logits_engine(acceptance_threshold=0.8)

    parent = Branch(branch_id="root", parent_id=None, tokens=torch.tensor([[1, 2, 3, 4, 5]]))
    child = Branch(branch_id="child", parent_id="root", tokens=torch.tensor([[1, 2, 0, 4, 7]]))
    # Draft region {2, 4}; only position 2 matches the target argmax -> rate 0.5.
    _set_argmax(model, torch.tensor([[1, 2, 0, 7, 4]]), vocab_size=8)

    cache.put(
        branch_id="child",
        layer_idx=0,
        activations={"embedding": model.embedding(child.tokens)},
    )
    assert cache.contains(branch_id="child", layer_idx=0)

    monkeypatch.setattr(engine, "_estimate_stable_ratio", lambda *args, **kwargs: 0.9)

    result = engine.verify_branch(parent, child, step_idx=0)

    assert result.acceptance_rate == pytest.approx(0.5)
    assert result.accepted is False
    assert cache.contains(branch_id="child", layer_idx=0) is False
    assert result.stable_ratio == pytest.approx(0.9)
    assert child.metadata["stable_ratio"] == pytest.approx(0.9)
    assert child.metadata["acceptance_rate"] == pytest.approx(0.5)


def test_ut308_full_rate_accepted_at_high_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-308 (SRS 3.4-3): rate = 1.0 passes threshold = 0.8 and keeps the cache.

    ``stable_ratio`` is forced low (0.6 < 0.8) while every draft token is
    accepted (rate = 1.0), proving the decision follows the acceptance rate
    rather than the stable ratio; accepted branches are not evicted.
    """
    model, cache, engine = _make_fixed_logits_engine(acceptance_threshold=0.8)

    parent = Branch(branch_id="root", parent_id=None, tokens=torch.tensor([[1, 2, 3, 4, 5]]))
    child = Branch(branch_id="child", parent_id="root", tokens=torch.tensor([[1, 2, 0, 4, 7]]))
    # Draft region {2, 4}; both positions match the target argmax -> rate 1.0.
    _set_argmax(model, torch.tensor([[1, 2, 0, 4, 7]]), vocab_size=8)

    cache.put(
        branch_id="child",
        layer_idx=0,
        activations={"embedding": model.embedding(child.tokens)},
    )

    monkeypatch.setattr(engine, "_estimate_stable_ratio", lambda *args, **kwargs: 0.6)

    result = engine.verify_branch(parent, child, step_idx=0)

    assert result.acceptance_rate == pytest.approx(1.0)
    assert result.accepted is True
    assert cache.contains(branch_id="child", layer_idx=0)


def test_ex604_empty_draft_region_is_full_acceptance() -> None:
    """EX-604 (design 6.4): child == parent means an empty draft region.

    No position carries a new claim, so the rate is 1.0 by definition, the
    draft-region mean log-prob is the documented 0.0 sentinel, and under the
    default threshold the branch is accepted without any exception.
    """
    model, _, engine = _make_fixed_logits_engine()

    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    parent = Branch(branch_id="root", parent_id=None, tokens=tokens)
    child = Branch(branch_id="child", parent_id="root", tokens=tokens.clone())
    _set_argmax(model, tokens, vocab_size=8)

    result = engine.verify_branch(parent, child, step_idx=0)

    assert result.acceptance_rate == pytest.approx(1.0)
    assert result.mean_log_prob == pytest.approx(0.0)
    assert result.accepted is True
    assert engine.ema_acceptance_rate == pytest.approx(1.0)


def test_ex601_engine_propagates_malformed_logits_value_error() -> None:
    """EX-601 (design 6.4): malformed logits raise ValueError through the engine.

    The engine calls :func:`target_argmax_accept_mask` /
    :func:`draft_region_mask` without an extra guard, so shape validation
    happens once in the shared ``_validate_token_tensor`` and must propagate
    out of ``verify_branch``. A rank-2 logits tensor (missing the vocab axis)
    is the malformed case.
    """
    model, _, engine = _make_fixed_logits_engine()

    parent = Branch(branch_id="root", parent_id=None, tokens=torch.tensor([[1, 2, 3]]))
    child = Branch(branch_id="child", parent_id="root", tokens=torch.tensor([[1, 0, 3]]))
    # Rank-2 logits: no per-position vocab axis.
    model.logits = torch.zeros(1, 8)

    with pytest.raises(ValueError):
        engine.verify_branch(parent, child, step_idx=0)


def test_ex602_engine_propagates_batch_mismatch_value_error() -> None:
    """EX-602 (design 6.4): a parent/child batch mismatch raises ValueError.

    The shared shape validation rejects a child whose batch dimension differs
    from the logits (and hence from the parent path), and the engine must
    propagate the error instead of silently broadcasting.
    """
    model, _, engine = _make_fixed_logits_engine()

    parent = Branch(branch_id="root", parent_id=None, tokens=torch.tensor([[1, 2, 3], [4, 5, 6]]))
    child = Branch(branch_id="child", parent_id="root", tokens=torch.tensor([[1, 0, 3]]))
    _set_argmax(model, torch.tensor([[1, 2, 3], [4, 5, 6]]), vocab_size=8)

    with pytest.raises(ValueError):
        engine.verify_branch(parent, child, step_idx=0)


# ---------------------------------------------------------------------------
# AR005 T003: engine consumes real FFN/MoE geometry from the model config.
# ---------------------------------------------------------------------------


class _ConfigGeometryDiffusionModel(DiffusionLLM):
    """DiffusionLLM stub whose geometry is only reachable via ``model.config``.

    Mimics the real checkpoint path (IT-411): concrete subclasses keep the HF
    config on the wrapped module, never on the wrapper itself.
    """

    def __init__(self, config: Any) -> None:
        super().__init__("geometry-stub")
        self.model = SimpleNamespace(config=config)
        self._logits: torch.Tensor | None = None

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._logits is None:
            raise AssertionError("_ConfigGeometryDiffusionModel._logits must be set")
        return self._logits

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(tokens.shape[0], tokens.shape[1], self.hidden_dim)

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


def _it411_config() -> SimpleNamespace:
    """SwiGLU + MoE config distinct from the 4h-MLP default geometry."""
    return SimpleNamespace(
        intermediate_size=48,  # != 4 * 16 default
        hidden_act="silu",  # swiglu: 3 matmuls
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=32,  # shared expert present
        num_hidden_layers=2,
        first_k_dense_replace=1,  # 1 dense prefix layer, 1 MoE layer
    )


def test_it411_engine_tflops_use_config_geometry() -> None:
    """IT-411 (design 6.2): the engine's TFLOPs reflect real config geometry.

    The geometry is only reachable through ``DiffusionLLM.model.config``
    (the real checkpoint path); the engine must pick it up through
    ``model_ffn_flops_kwargs`` without any call-site change and report a
    hand-computed value that differs from the 4h-MLP default estimate.
    """
    config = _it411_config()
    model = _ConfigGeometryDiffusionModel(config)
    adapter = FastDLLMAdapter(model)  # isinstance DiffusionLLM: dims from stub

    argmax_tokens = torch.tensor([[1, 2, 3, 4, 5]])
    logits = torch.zeros(1, 5, 100)
    logits.scatter_(-1, argmax_tokens.unsqueeze(-1), 10.0)
    model._logits = logits

    engine = ActFoldVerificationEngine(
        adapter, ActivationCache(device="cpu"), SimilarityGate(tau=0.95)
    )
    parent = Branch(branch_id="root", parent_id=None, tokens=argmax_tokens.clone())
    child = Branch(branch_id="child", parent_id="root", tokens=torch.tensor([[1, 2, 0, 4, 7]]))

    result = engine.verify_branch(parent, child, step_idx=0)

    kwargs = model_ffn_flops_kwargs(adapter)
    assert kwargs["ffn_intermediate_dim"] == 48
    assert kwargs["ffn_type"] == "swiglu"
    assert kwargs["moe_top_k"] == 2
    assert kwargs["moe_num_layers"] == 1
    assert kwargs["moe_shared_expert"] is True

    expected = count_diffusion_llm_flops(
        num_layers=2,
        hidden_dim=16,
        num_heads=2,
        seq_len=5,
        vocab_size=100,
        num_steps=1,
        reuse_ratio=result.stable_ratio,
        **kwargs,
    ).total_tflops
    assert result.tflops == pytest.approx(expected, rel=1e-12)

    default_estimate = count_diffusion_llm_flops(
        num_layers=2,
        hidden_dim=16,
        num_heads=2,
        seq_len=5,
        vocab_size=100,
        num_steps=1,
        reuse_ratio=result.stable_ratio,
    ).total_tflops
    assert result.tflops != pytest.approx(default_estimate)
