"""Red-phase tests for AR002/T006: ``fused_gate_mask_count`` + layer wiring.

Covers the fused cosine+threshold+count op (contract: design.md §4.3.1) and
the ``FoldedTransformerLayer`` wiring that replaces the separate
``gate(...)`` + ``mask.sum()`` calls on the cosine/SimilarityGate path.

The fused function is imported lazily inside each test so that the Red state
(the function not existing yet) fails tests individually instead of aborting
the whole module at collection time.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, fused_ops
from actfold.core.adaptive_gate import AdaptiveQuantileGate
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.fused_ops import merge_stable_divergent
from actfold.core.similarity_gate import SimilarityGate

_EPS = 1e-8
_HAS_TRITON = fused_ops._HAS_TRITON


def _load_fused_gate_mask_count():
    """Import the fused gate op lazily (fails per-test in the Red state)."""
    from actfold.core.fused_ops import fused_gate_mask_count

    return fused_gate_mask_count


class _IdentityLayer(nn.Module):
    """Identity-like layer mirroring tests/test_folded_transformer.py."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.linear(hidden_states)


def _run_folded_child(
    layer: nn.Module,
    gate: SimilarityGate,
    parent: torch.Tensor,
    child: torch.Tensor,
) -> torch.Tensor:
    """Run one parent pass + one mixed child pass through a folded layer."""
    cache = ActivationCache(device="cpu")
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0)
    folded(parent, branch_id="parent")
    return folded(child, branch_id="child", parent_branch_id="parent")


# ---------------------------------------------------------------------------
# Fused op contract (PyTorch fallback path, CPU)
# ---------------------------------------------------------------------------


def test_fused_gate_fallback_bit_exact_cpu() -> None:
    """CPU fallback is bit-exact with the SimilarityGate cosine chain."""
    fused_gate_mask_count = _load_fused_gate_mask_count()

    torch.manual_seed(42)
    h_parent = torch.randn(2, 5, 8, dtype=torch.float32)
    h_child = h_parent.clone()
    h_child[0, 0] = -h_parent[0, 0]  # cosine -1 -> divergent for tau >= 0
    h_child[1, 2] = h_child[1, 2] + 0.1  # slightly below 1 -> boundary-sensitive

    for tau in (0.0, 0.5, 0.99, 1.0):
        out_mask = torch.empty(2, 5, dtype=torch.bool)
        out_count = torch.zeros((), dtype=torch.int64)
        fused_gate_mask_count(h_child, h_parent, tau, _EPS, out_mask, out_count)

        ref_gate = SimilarityGate(tau=tau, metric="cosine", eps=_EPS)
        ref_mask = ref_gate(h_child, h_parent)
        assert torch.equal(out_mask, ref_mask), f"mask mismatch at tau={tau}"
        assert int(out_count) == int(ref_mask.sum()), f"count mismatch at tau={tau}"


def test_fused_gate_accumulate_semantics() -> None:
    """out_count accumulates; out_mask is fully overwritten."""
    fused_gate_mask_count = _load_fused_gate_mask_count()

    torch.manual_seed(43)
    h_parent = torch.randn(2, 5, 8, dtype=torch.float32)
    h_child = h_parent.clone()
    h_child[0, 0] = -h_parent[0, 0]

    ref_mask = SimilarityGate(tau=0.5, metric="cosine", eps=_EPS)(h_child, h_parent)
    assert 0 < int(ref_mask.sum()) < ref_mask.numel(), "scenario must be mixed"

    out_mask = ~ref_mask  # deliberately wrong: False wherever the ref is True
    out_count = torch.full((), 7, dtype=torch.int64)
    fused_gate_mask_count(h_child, h_parent, 0.5, _EPS, out_mask, out_count)

    assert torch.equal(out_mask, ref_mask), "out_mask must be fully overwritten"
    assert int(out_count) == 7 + int(ref_mask.sum()), "out_count must accumulate"


def test_fused_gate_input_validation_value_errors() -> None:
    """Entry validation raises ValueError before any compute."""
    fused_gate_mask_count = _load_fused_gate_mask_count()

    h_child = torch.randn(2, 5, 8, dtype=torch.float32)
    h_parent = torch.randn(2, 5, 8, dtype=torch.float32)
    out_mask = torch.empty(2, 5, dtype=torch.bool)
    out_count = torch.zeros((), dtype=torch.int64)

    # h_parent shape mismatch.
    with pytest.raises(ValueError):
        fused_gate_mask_count(
            h_child,
            torch.randn(2, 4, 8, dtype=torch.float32),
            0.5,
            _EPS,
            out_mask,
            out_count,
        )

    # out_mask shape != h_child.shape[:2].
    with pytest.raises(ValueError):
        fused_gate_mask_count(
            h_child,
            h_parent,
            0.5,
            _EPS,
            torch.empty(2, 4, dtype=torch.bool),
            out_count,
        )

    # out_count with 2 elements.
    with pytest.raises(ValueError):
        fused_gate_mask_count(
            h_child,
            h_parent,
            0.5,
            _EPS,
            out_mask,
            torch.zeros(2, dtype=torch.int64),
        )

    # h_parent dtype mismatch (float64 vs float32).
    with pytest.raises(ValueError):
        fused_gate_mask_count(h_child, h_parent.double(), 0.5, _EPS, out_mask, out_count)

    # h_parent device mismatch (meta vs cpu): must fire before any compute.
    with pytest.raises(ValueError):
        fused_gate_mask_count(
            h_child,
            torch.empty(2, 5, 8, device="meta"),
            0.5,
            _EPS,
            out_mask,
            out_count,
        )


def test_fused_gate_nan_and_tau_boundary() -> None:
    """NaN rows map to divergent; tau=1.0 with identical vectors is all-False."""
    fused_gate_mask_count = _load_fused_gate_mask_count()

    torch.manual_seed(44)
    h_parent = torch.randn(2, 5, 8, dtype=torch.float32)
    h_child = h_parent.clone()
    h_child[1, 3, :] = float("nan")

    out_mask = torch.empty(2, 5, dtype=torch.bool)
    out_count = torch.zeros((), dtype=torch.int64)
    fused_gate_mask_count(h_child, h_parent, 0.5, _EPS, out_mask, out_count)

    ref_mask = SimilarityGate(tau=0.5, metric="cosine", eps=_EPS)(h_child, h_parent)
    assert not out_mask[1, 3], "NaN row must be divergent (False)"
    assert torch.equal(out_mask, ref_mask)
    assert int(out_count) == int(ref_mask.sum())

    # Identical vectors: sim clamps to exactly 1.0 and 1.0 > 1.0 is False.
    out_mask2 = torch.empty(2, 5, dtype=torch.bool)
    out_count2 = torch.zeros((), dtype=torch.int64)
    fused_gate_mask_count(h_parent, h_parent, 1.0, _EPS, out_mask2, out_count2)
    assert not out_mask2.any(), "strict > at tau=1.0 must yield no stable token"
    assert int(out_count2) == 0


def test_fused_gate_min_tokens_constant_exists() -> None:
    """The module exposes the dispatch constants (_FUSED_GATE_MIN_TOKENS...)."""
    from actfold.core import fused_ops

    assert isinstance(fused_ops._FUSED_GATE_MIN_TOKENS, int)
    assert fused_ops._FUSED_GATE_MIN_TOKENS >= 1
    assert hasattr(fused_ops, "_TRITON_GATE_DISABLED")


def test_fused_gate_forced_disabled_fallback_bit_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With _TRITON_GATE_DISABLED forced True the op reproduces the gate chain.

    fp32 and fp16 CPU inputs (the reference gate runs fp16 on CPU fine: it
    upcasts to fp32 internally, so no dtype skip is needed).
    """
    fused_gate_mask_count = _load_fused_gate_mask_count()
    from actfold.core import fused_ops

    monkeypatch.setattr(fused_ops, "_TRITON_GATE_DISABLED", True, raising=False)

    torch.manual_seed(45)
    for dtype in (torch.float32, torch.float16):
        h_parent = torch.randn(2, 5, 8, dtype=dtype)
        h_child = h_parent.clone()
        h_child[0, 0] = -h_parent[0, 0]

        for tau in (0.0, 0.5, 1.0):
            out_mask = torch.empty(2, 5, dtype=torch.bool)
            out_count = torch.zeros((), dtype=torch.int64)
            fused_gate_mask_count(h_child, h_parent, tau, _EPS, out_mask, out_count)

            ref_gate = SimilarityGate(tau=tau, metric="cosine", eps=_EPS)
            ref_mask = ref_gate(h_child, h_parent)
            assert torch.equal(out_mask, ref_mask), f"mask mismatch {dtype} tau={tau}"
            assert int(out_count) == int(ref_mask.sum()), f"count mismatch {dtype} tau={tau}"


# ---------------------------------------------------------------------------
# CUDA-only tests (Triton path); skipped on this CPU-only machine
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_gate_cuda_triton_bit_exact(dtype: torch.dtype) -> None:
    """On CUDA with B*T >= _FUSED_GATE_MIN_TOKENS the Triton path is bit-exact."""
    fused_gate_mask_count = _load_fused_gate_mask_count()

    torch.manual_seed(46)
    h_parent = torch.randn(2, 512, 64, dtype=dtype, device="cuda")
    h_child = h_parent.clone()
    h_child[:, :256] = -h_child[:, :256]  # first half divergent, second stable

    out_mask = torch.empty(2, 512, dtype=torch.bool, device="cuda")
    out_count = torch.zeros((), dtype=torch.int64, device="cuda")
    fused_gate_mask_count(h_child, h_parent, 0.5, _EPS, out_mask, out_count)

    ref_gate = SimilarityGate(tau=0.5, metric="cosine", eps=_EPS)
    ref_mask = ref_gate(h_child, h_parent)
    assert torch.equal(out_mask, ref_mask)
    assert int(out_count) == int(ref_mask.sum())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_fused_gate_launch_count_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer wiring on CUDA calls the fused op once per mixed layer, bit-exact.

    The patched child pass must observe exactly one ``fused_gate_mask_count``
    call and its output must equal the eager reference (gate call + manual
    ``merge_stable_divergent``).
    """
    real_fused = _load_fused_gate_mask_count()

    hidden_dim = 64
    layer = _IdentityLayer(hidden_dim).to("cuda")
    gate = SimilarityGate(tau=0.95, metric="cosine", eps=_EPS)
    cache = ActivationCache(device="cuda")
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0)

    torch.manual_seed(47)
    parent = torch.randn(2, 512, hidden_dim, device="cuda")
    folded(parent, branch_id="parent")
    child = parent.clone()
    child[:, :256] += 100.0 * torch.randn(2, 256, hidden_dim, device="cuda")

    calls: list[tuple] = []

    def _counting_wrapper(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        return real_fused(*args, **kwargs)

    monkeypatch.setattr(
        "actfold.core.folded_transformer.fused_gate_mask_count",
        _counting_wrapper,
        raising=False,
    )
    out = folded(child, branch_id="child", parent_branch_id="parent")

    assert len(calls) == 1, "mixed child pass must call the fused op exactly once"

    stable_mask = gate(child, parent)
    assert 0 < int(stable_mask.sum()) < stable_mask.numel(), "scenario must be mixed"
    parent_ffn = cache.fetch(branch_id="parent", layer_idx=0).get("ffn_out")
    expected = merge_stable_divergent(parent_ffn, layer(child), stable_mask)
    assert torch.equal(out, expected)


# ---------------------------------------------------------------------------
# FoldedTransformerLayer wiring (CPU)
# ---------------------------------------------------------------------------


def test_layer_wiring_uses_fused_for_cosine(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cosine SimilarityGate layer takes the fused path, bit-exactly."""
    real_fused = _load_fused_gate_mask_count()

    hidden_dim = 16
    torch.manual_seed(48)
    layer = _IdentityLayer(hidden_dim)
    parent = torch.randn(2, 8, hidden_dim)
    child = parent.clone()
    child[:, :4] += 100.0 * torch.randn(2, 4, hidden_dim)

    # Sanity: the child pass is mixed-stability (slow path is exercised).
    probe_mask = SimilarityGate(tau=0.95, metric="cosine", eps=_EPS)(child, parent)
    assert 0 < int(probe_mask.sum()) < probe_mask.numel()

    # Unpatched reference run.
    ref_out = _run_folded_child(
        layer, SimilarityGate(tau=0.95, metric="cosine", eps=_EPS), parent, child
    )

    calls: list[tuple] = []

    def _counting_wrapper(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        return real_fused(*args, **kwargs)

    monkeypatch.setattr(
        "actfold.core.folded_transformer.fused_gate_mask_count",
        _counting_wrapper,
        raising=False,
    )
    out = _run_folded_child(
        layer, SimilarityGate(tau=0.95, metric="cosine", eps=_EPS), parent, child
    )

    assert len(calls) == 1, "cosine SimilarityGate child pass must use the fused op"
    assert torch.equal(out, ref_out), "wiring must preserve bit-exact outputs"


def test_layer_wiring_skips_fused_for_non_cosine_or_adaptive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-cosine metrics and AdaptiveQuantileGate keep the old gate path."""
    real_fused = _load_fused_gate_mask_count()

    hidden_dim = 16
    torch.manual_seed(49)
    layer = _IdentityLayer(hidden_dim)
    parent = torch.randn(2, 8, hidden_dim)
    child = parent.clone()
    child[:, :4] += 100.0 * torch.randn(2, 4, hidden_dim)

    calls: list[tuple] = []

    def _counting_wrapper(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        return real_fused(*args, **kwargs)

    monkeypatch.setattr(
        "actfold.core.folded_transformer.fused_gate_mask_count",
        _counting_wrapper,
        raising=False,
    )

    # Non-cosine metric: wrapper must not be called; output matches the
    # unpatched l2 run.
    ref_out = _run_folded_child(
        layer, SimilarityGate(tau=0.95, metric="l2", eps=_EPS), parent, child
    )
    out = _run_folded_child(layer, SimilarityGate(tau=0.95, metric="l2", eps=_EPS), parent, child)
    assert len(calls) == 0, "non-cosine metric must not take the fused path"
    assert torch.equal(out, ref_out)

    # AdaptiveQuantileGate subclasses SimilarityGate but the wiring uses an
    # exact type check, so it must keep the old path too.
    adaptive_gate = AdaptiveQuantileGate(target_stable_ratio=0.5, metric="cosine", eps=_EPS)
    _run_folded_child(layer, adaptive_gate, parent, child)
    assert len(calls) == 0, "AdaptiveQuantileGate must not take the fused path"


# ---------------------------------------------------------------------------
# UT-006c: simulated compile/launch failure -> one RuntimeWarning + disable
# ---------------------------------------------------------------------------


class _ExplodingKernel:
    """Stand-in for the Triton JIT kernel whose launch always raises."""

    def __getitem__(self, grid: object) -> Any:
        def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated Triton compile failure")

        return _raise


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAS_TRITON),
    reason="CUDA + Triton required (the test patches the JIT kernel object)",
)
def test_ut006c_compile_failure_warns_once_and_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    """AR002 design UT-006c: kernel failure -> one RuntimeWarning + fallback.

    The first call falls back to the PyTorch gate chain (bit-exact vs the
    reference gate), sets the module-level disable flag, and warns exactly
    once; subsequent calls stay on the fallback without warning again.
    """
    fused_gate_mask_count = _load_fused_gate_mask_count()
    from actfold.core import fused_ops

    monkeypatch.setattr(fused_ops, "_TRITON_GATE_DISABLED", False, raising=False)
    monkeypatch.setattr(fused_ops, "_fused_gate_kernel", _ExplodingKernel())

    torch.manual_seed(47)
    h_parent = torch.randn(2, 512, 64, device="cuda")
    h_child = h_parent.clone()
    h_child[:, :256] = -h_child[:, :256]

    ref_gate = SimilarityGate(tau=0.5, metric="cosine", eps=_EPS)
    ref_mask = ref_gate(h_child, h_parent)

    out_mask = torch.empty(2, 512, dtype=torch.bool, device="cuda")
    out_count = torch.zeros((), dtype=torch.int64, device="cuda")
    with pytest.warns(RuntimeWarning, match="fused gate kernel unavailable"):
        fused_gate_mask_count(h_child, h_parent, 0.5, _EPS, out_mask, out_count)

    # Fallback result is bit-exact vs the reference gate chain.
    assert torch.equal(out_mask, ref_mask)
    assert int(out_count) == int(ref_mask.sum())
    # The failure permanently disabled the Triton path.
    assert fused_ops._TRITON_GATE_DISABLED is True

    # Second call: no new warning (the kernel is not attempted again) and
    # the result is still correct.
    out_mask2 = torch.empty(2, 512, dtype=torch.bool, device="cuda")
    out_count2 = torch.zeros((), dtype=torch.int64, device="cuda")
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        fused_gate_mask_count(h_child, h_parent, 0.5, _EPS, out_mask2, out_count2)
    assert torch.equal(out_mask2, ref_mask)
    assert int(out_count2) == int(ref_mask.sum())
