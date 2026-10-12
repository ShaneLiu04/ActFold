"""AR009: silent-misconfiguration guards for Branch Folding.

Covers specs/changes/AR009-silent-misconfig-guards (srs 3.1 / 3.2):

- UT-801/802/803: the embedding-level proxy fallback in
  ``ActFoldVerificationEngine._estimate_stable_ratio`` emits a one-time
  ``UserWarning`` per engine instance, while the layer-profile path stays
  silent (srs 3.1).
- UT-804/805: combining ``AdaptiveQuantileGate`` with a ``FoldingScheduler``
  is rejected at construction time by ``FoldedTransformerLayer``,
  ``SplitFoldedTransformerLayer`` and ``ActFoldVerificationEngine`` (srs 3.2).
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.adaptive_gate import AdaptiveQuantileGate
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.split_layer import SplitFoldedTransformerLayer
from actfold.profiler.stability_profiler import (
    GLOBAL_STABILITY_PROFILER,
    LayerStabilityStats,
    StabilityProfile,
)
from actfold.speculative import ActFoldVerificationEngine, FastDLLMAdapter
from actfold.speculative.branch import Branch


class _FixedLogitsModel(nn.Module):
    """Stub model whose forward returns precomputed logits (UT-306 pattern).

    The verification engine only needs the ``embedding`` attribute for
    ``adapter.embed`` and a ``[batch, seq_len, vocab]`` logits output; both
    are controlled exactly by the test.
    """

    def __init__(self, vocab_size: int = 8, hidden_dim: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.logits: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.logits is None:
            raise AssertionError("_FixedLogitsModel.logits must be set before forward")
        return self.logits


def _make_engine(
    gate: SimilarityGate,
    scheduler: FoldingScheduler | None = None,
) -> tuple[_FixedLogitsModel, ActFoldVerificationEngine]:
    """Build a verification engine around a :class:`_FixedLogitsModel` stub.

    Args:
        gate: Similarity gate handed to the engine (constructor-guard focus).
        scheduler: Optional folding scheduler handed to the engine.

    Returns:
        A ``(model, engine)`` pair; assign ``model.logits`` through
        :func:`_set_argmax` before invoking ``engine.verify_branch``.
    """
    model = _FixedLogitsModel()
    adapter = FastDLLMAdapter(model, num_layers=1, hidden_dim=8)
    cache = ActivationCache(device="cpu")
    engine = ActFoldVerificationEngine(adapter, cache, gate, scheduler)
    return model, engine


def _set_argmax(model: _FixedLogitsModel, argmax_tokens: torch.Tensor) -> None:
    """Point the stub's per-position logits argmax at ``argmax_tokens``.

    Args:
        model: Stub whose ``logits`` should be overwritten.
        argmax_tokens: ``[batch, seq_len]`` desired per-position argmax ids.
    """
    batch, seq_len = argmax_tokens.shape
    logits = torch.zeros(batch, seq_len, 8)
    logits.scatter_(-1, argmax_tokens.unsqueeze(-1), 10.0)
    model.logits = logits


def _proxy_warnings(recwarn: pytest.WarningsChecker) -> list[warnings.WarningMessage]:
    """Return the captured warnings that announce the embedding-level proxy.

    Args:
        recwarn: The pytest warning-capture fixture.

    Returns:
        The captured ``UserWarning`` entries whose message contains the
        ``"embedding-level proxy"`` marker from srs 3.1.
    """
    return [
        w
        for w in recwarn.list
        if issubclass(w.category, UserWarning) and "embedding-level proxy" in str(w.message)
    ]


def _expected_proxy_ratio(
    gate: SimilarityGate,
    model: _FixedLogitsModel,
    parent_tokens: torch.Tensor,
    child_tokens: torch.Tensor,
) -> float:
    """Independently compute the embedding-proxy fallback formula.

    Args:
        gate: The same gate instance the engine was constructed with.
        model: Stub model supplying the real input embeddings.
        parent_tokens: Parent branch tokens.
        child_tokens: Child branch tokens.

    Returns:
        ``gate(h_child, h_parent).float().mean().item()`` — the fallback
        formula itself, computed outside the engine (anti-vacuous pin).
    """
    h_child = model.embedding(child_tokens)
    h_parent = model.embedding(parent_tokens)
    return gate(h_child, h_parent).float().mean().item()


def test_ut801_proxy_fallback_warns_exactly_once_per_engine(
    recwarn: pytest.WarningsChecker,
) -> None:
    """UT-801 (srs 3.1-1): a proxy fallback warns once; the ratio is unchanged.

    The stub adapter has no folded forward path, so both consecutive
    ``verify_branch`` calls take the embedding-proxy fallback. Exactly one
    ``UserWarning`` mentioning ``"embedding-level proxy"`` must be captured,
    and both returned ``stable_ratio`` values must be bit-identical to the
    fallback formula computed independently in this test (pre-change
    semantics pinned), hence equal to each other.
    """
    gate = SimilarityGate(tau=0.95)
    model, engine = _make_engine(gate)
    _set_argmax(model, torch.tensor([[1, 2, 0, 7, 4]]))

    parent_tokens = torch.tensor([[1, 2, 3, 4, 5]])
    child_tokens = torch.tensor([[1, 2, 0, 4, 7]])
    expected = _expected_proxy_ratio(gate, model, parent_tokens, child_tokens)

    parent = Branch(branch_id="root", parent_id=None, tokens=parent_tokens)
    child_a = Branch(branch_id="child-a", parent_id="root", tokens=child_tokens)
    child_b = Branch(branch_id="child-b", parent_id="root", tokens=child_tokens)

    result_a = engine.verify_branch(parent, child_a, step_idx=0)
    result_b = engine.verify_branch(parent, child_b, step_idx=0)

    assert len(_proxy_warnings(recwarn)) == 1
    assert result_a.stable_ratio == expected
    assert result_b.stable_ratio == expected


def test_ut802_once_flag_is_engine_instance_scoped(
    recwarn: pytest.WarningsChecker,
) -> None:
    """UT-802 (srs 3.1-2): two independent engines each warn once (2 total).

    The once flag is instance-level, not module-level: two engines built
    with identical parameters each take the embedding-proxy fallback once,
    and both first fallbacks must be announced — two warnings in total.
    """
    gate_a = SimilarityGate(tau=0.95)
    model_a, engine_a = _make_engine(gate_a)
    gate_b = SimilarityGate(tau=0.95)
    model_b, engine_b = _make_engine(gate_b)
    _set_argmax(model_a, torch.tensor([[1, 2, 0, 7, 4]]))
    _set_argmax(model_b, torch.tensor([[1, 2, 0, 7, 4]]))

    parent_tokens = torch.tensor([[1, 2, 3, 4, 5]])
    child_tokens = torch.tensor([[1, 2, 0, 4, 7]])
    parent_a = Branch(branch_id="root", parent_id=None, tokens=parent_tokens)
    child_a = Branch(branch_id="child-a", parent_id="root", tokens=child_tokens)
    parent_b = Branch(branch_id="root", parent_id=None, tokens=parent_tokens)
    child_b = Branch(branch_id="child-b", parent_id="root", tokens=child_tokens)

    engine_a.verify_branch(parent_a, child_a, step_idx=0)
    engine_b.verify_branch(parent_b, child_b, step_idx=0)

    assert len(_proxy_warnings(recwarn)) == 2


def test_ut803_layer_profile_path_never_warns(
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsChecker,
) -> None:
    """UT-803 (srs 3.1-3): a real layer profile silences the proxy warning.

    ``GLOBAL_STABILITY_PROFILER.get_profile`` is monkeypatched to return a
    real :class:`StabilityProfile` with one
    ``LayerStabilityStats(stable_ratio=0.5)`` entry, so
    ``_estimate_stable_ratio`` takes the profiler-mean path (the function
    under test still executes for real). No ``UserWarning`` may be raised
    and the returned ``stable_ratio`` must equal the profiler mean 0.5.
    """
    gate = SimilarityGate(tau=0.95)
    model, engine = _make_engine(gate)
    _set_argmax(model, torch.tensor([[1, 2, 0, 7, 4]]))

    profile = StabilityProfile(
        branch_id="child",
        parent_branch_id="root",
        layer_stats=[
            LayerStabilityStats(
                layer_idx=0,
                step_idx=0,
                stable_ratio=0.5,
                tau_used=0.5,
                metric="cosine",
                num_tokens=4,
            )
        ],
    )
    monkeypatch.setattr(GLOBAL_STABILITY_PROFILER, "get_profile", lambda branch_id: profile)

    parent = Branch(branch_id="root", parent_id=None, tokens=torch.tensor([[1, 2, 3, 4, 5]]))
    child = Branch(branch_id="child", parent_id="root", tokens=torch.tensor([[1, 2, 0, 4, 7]]))

    result = engine.verify_branch(parent, child, step_idx=0)

    assert len(_proxy_warnings(recwarn)) == 0
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert result.stable_ratio == pytest.approx(0.5)


def test_ut804_layer_guard_rejects_adaptive_gate_with_scheduler() -> None:
    """UT-804 (srs 3.2): layer constructors reject the mutually exclusive combo.

    ``FoldedTransformerLayer`` and its subclass
    ``SplitFoldedTransformerLayer`` must raise ``RuntimeError`` when built
    with ``(AdaptiveQuantileGate, FoldingScheduler())`` — the message names
    ``AdaptiveQuantileGate`` and ``gate.tau``. The legal combinations
    ``(SimilarityGate, FoldingScheduler())`` and
    ``(AdaptiveQuantileGate, None)`` must keep constructing successfully for
    both classes (four legal cases in total).
    """
    cache = ActivationCache(device="cpu")
    original = nn.Module()
    scheduler = FoldingScheduler()

    for layer_cls in (FoldedTransformerLayer, SplitFoldedTransformerLayer):
        with pytest.raises(RuntimeError, match="gate.tau") as excinfo:
            layer_cls(original, cache, AdaptiveQuantileGate(), layer_idx=0, scheduler=scheduler)
        assert "AdaptiveQuantileGate" in str(excinfo.value)
        assert "FoldingScheduler" in str(excinfo.value)

    for layer_cls in (FoldedTransformerLayer, SplitFoldedTransformerLayer):
        layer_cls(original, cache, SimilarityGate(tau=0.95), layer_idx=0, scheduler=scheduler)
        layer_cls(original, cache, AdaptiveQuantileGate(), layer_idx=0, scheduler=None)


def test_ut805_engine_guard_rejects_adaptive_gate_with_scheduler() -> None:
    """UT-805 (srs 3.2): the engine rejects the mutually exclusive combo.

    ``ActFoldVerificationEngine`` must raise ``RuntimeError`` with the same
    message contract as the layer guard (``gate.tau`` +
    ``AdaptiveQuantileGate``) when constructed with
    ``(AdaptiveQuantileGate, FoldingScheduler())``, while the legal
    combinations ``(SimilarityGate, scheduler)`` and
    ``(AdaptiveQuantileGate, scheduler=None)`` keep constructing
    successfully (UT-306 stub-adapter pattern).
    """
    with pytest.raises(RuntimeError, match="gate.tau") as excinfo:
        _make_engine(AdaptiveQuantileGate(), FoldingScheduler())
    assert "AdaptiveQuantileGate" in str(excinfo.value)
    assert "FoldingScheduler" in str(excinfo.value)

    _, engine = _make_engine(SimilarityGate(tau=0.95), FoldingScheduler())
    assert isinstance(engine, ActFoldVerificationEngine)

    _, engine = _make_engine(AdaptiveQuantileGate(), None)
    assert isinstance(engine, ActFoldVerificationEngine)
