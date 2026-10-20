"""AR029 benchmark-verdict tests (UT-2959..2966, UT-2963b, UT-2977..2978)."""

from __future__ import annotations

import math

import pytest
import torch

import actfold.core.folded_transformer as folded_module
import scripts.ar029_pyramid_cache_bench as bench
from actfold.core import ActivationCache
from actfold.core.pyramid_cache import PyramidActivationCache
from actfold.core.similarity_gate import SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward


@pytest.fixture(scope="module")
def artifact() -> dict:
    """Run the quick mechanism face once for the artifact-level tests."""
    return bench.run(quick=True)


def test_ut2959_e1_supported_on_exact_references() -> None:
    """UT-2959: E1 passes only when both references are bit-exact."""
    assert bench.verdict_e1(0.0, 0.0) is True


def test_ut2960_e1_refuted_by_tampered_reference() -> None:
    """UT-2960: any nonzero (even 1e-9) reference difference refutes E1."""
    assert bench.verdict_e1(1e-9, 0.0) is False
    assert bench.verdict_e1(0.0, 1e-9) is False
    assert bench.verdict_e1(float("nan"), 0.0) is False


def test_ut2961_e2_upper_bound_and_byte_ledger() -> None:
    """UT-2961: per-layer budgets hold and the byte ledger matches blocks."""
    tokens = {0: 16, 1: 16}
    budgets = {0: 16, 1: 16}
    assert bench.verdict_e2(tokens, budgets, 4096, 4096) is True
    assert bench.verdict_e2({0: 32, 1: 16}, budgets, 4096, 4096) is False
    assert bench.verdict_e2(tokens, budgets, 4096, 2048) is False


def test_ut2961b_e2_missing_layer_reads_as_zero() -> None:
    """UT-2961b: an absent stats layer is zero, not a KeyError."""
    assert bench.verdict_e2({0: 16}, {0: 16, 1: 16}, 1024, 1024) is True
    assert bench.verdict_e2({}, {0: 16}, 0, 0) is True


def test_ut2962_e3_reuse_threshold() -> None:
    """UT-2962: E3 needs real reuse above the frozen 1% floor."""
    assert bench.MIN_REUSE_RATIO == 0.01
    assert bench.verdict_e3(1, 100) is True
    assert bench.verdict_e3(0, 100) is False
    assert bench.verdict_e3(0, 0) is False


def test_ut2977_e3_ratio_formula() -> None:
    """UT-2977: the ratio is reuse/valid over integer token counts."""
    assert bench.verdict_e3(40, 64) is True
    assert bench.verdict_e3(1, 1000) is False
    assert bench.verdict_e3(5, 10) is True


def test_ut2963_e4_blocks_on_nan_or_negative_stats() -> None:
    """UT-2963: non-finite outputs or negative stats force ``blocked``."""
    good = torch.tensor([1.0, 2.0])
    bad = torch.tensor([1.0, float("nan")])
    assert bench.verdict_e4([good], [0.0, 1.0]) is True
    assert bench.verdict_e4([bad], [0.0, 1.0]) is False
    assert bench.verdict_e4([good], [-1.0]) is False
    assert bench.verdict_e4([good], [math.inf]) is False
    assert bench.overall_verdict(True, True, True, False) == "blocked"
    assert bench.overall_verdict(False, True, True, True) == "refuted"
    assert bench.overall_verdict(True, False, True, True) == "refuted"
    assert bench.overall_verdict(True, True, False, True) == "refuted"
    assert bench.overall_verdict(True, True, True, True) == "supported"


def test_ut2966_verdicts_are_frozen_in_the_script() -> None:
    """UT-2966: the gate thresholds are module constants, not data-driven."""
    assert bench.SCHEMA_VERSION == 1
    assert bench.SCHEMA == "ar029_pyramid_cache_bench"
    assert bench.verdict_e1.__module__ == bench.__name__
    assert bench.verdict_e2.__module__ == bench.__name__
    assert bench.verdict_e3.__module__ == bench.__name__
    assert bench.verdict_e4.__module__ == bench.__name__


def _pressure_scene(monkeypatch=None):
    """Build the pressure scene; optionally disable the target-layer guard."""
    torch.manual_seed(1234)
    model = bench.TinyDecoder().eval()
    bench._MODEL = model
    root = bench._tokens(0)
    child = root.clone()
    child[:, -8:] = (child[:, -8:] + 7) % bench.VOCAB
    cache = PyramidActivationCache(bench.build_plan(pressure=True))
    with torch.no_grad():
        ManualFoldedForward(model, cache=cache, gate=SimilarityGate(tau=0.9))(
            root, branch_id="root"
        )
    factual = {
        layer: ~cache.availability("root", layer, bench.SEQ, 0) for layer in range(bench.DEPTH)
    }
    if monkeypatch is not None:
        monkeypatch.setattr(
            folded_module,
            "_apply_cache_availability",
            lambda layer, parent, mask: (mask, False),
        )
    measured, _ = bench._run_face(cache, root, child)
    return model, root, child, factual, measured


def test_ut2963b_guard_off_output_is_not_honest(monkeypatch) -> None:
    """UT-2963b: disabling the guard leaks NaN (E4 fails), never a silent pass."""
    _, _, _, _, guarded = _pressure_scene()
    assert torch.isfinite(guarded).all()

    _, _, _, _, unguarded = _pressure_scene(monkeypatch)
    assert not torch.isfinite(unguarded).all()
    assert bench.verdict_e4([unguarded], [0.0]) is False
    assert bench.overall_verdict(True, True, True, False) == "blocked"


def test_ut2978_guard_off_face_is_refuted(monkeypatch) -> None:
    """UT-2978: a guard-off face fails E1/E4 instead of passing by accident."""
    model, root, child, factual, measured = _pressure_scene()
    assert torch.isfinite(measured).all()

    dense = ActivationCache(max_entries_per_layer=bench.SEQ * 4)
    with torch.no_grad():
        ManualFoldedForward(model, cache=dense, gate=SimilarityGate(tau=0.9))(
            root, branch_id="root"
        )
    oracle = bench._HoleMaskedDense(dense, factual)
    with torch.no_grad():
        ref = ManualFoldedForward(model, cache=oracle, gate=SimilarityGate(tau=0.9))(
            child, branch_id="ref", parent_branch_id="root"
        )
    assert float((measured - ref).abs().max().item()) == 0.0

    dense2 = ActivationCache(max_entries_per_layer=bench.SEQ * 4)
    with torch.no_grad():
        ManualFoldedForward(model, cache=dense2, gate=SimilarityGate(tau=0.9))(
            root, branch_id="root"
        )
    all_available = bench._HoleMaskedDense(
        dense2, {layer: torch.zeros(bench.SEQ, dtype=torch.bool) for layer in factual}
    )
    with torch.no_grad():
        wrong_availability = ManualFoldedForward(
            model, cache=all_available, gate=SimilarityGate(tau=0.9)
        )(child, branch_id="ref", parent_branch_id="root")
    wrong_diff = float((measured - wrong_availability).abs().max().item())
    assert not (wrong_diff == 0.0)
    assert bench.verdict_e1(0.0, wrong_diff) is False

    _, _, _, _, unguarded = _pressure_scene(monkeypatch)
    guard_off_diff = float((unguarded - ref).abs().max().item())
    assert not (guard_off_diff == 0.0)
    assert bench.verdict_e1(0.0, guard_off_diff) is False
    assert bench.verdict_e4([unguarded], [0.0]) is False


def test_ut2964_quick_path_and_formal_refusal(tmp_path, monkeypatch) -> None:
    """UT-2964: quick writes the quick artifact; formal is BLOCKED-DEFERRED."""
    target = tmp_path / "quick.json"
    monkeypatch.setattr(bench, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(bench, "QUICK_PATH", target)
    assert bench.main(["--quick"]) == 0
    assert target.is_file()
    assert bench.main([]) == 3
    assert not (tmp_path / "results/optimization/ar029_pyramid_cache_bench.json").exists()


def test_ut2965_artifact_fields(artifact: dict) -> None:
    """UT-2965: mechanism evidence, no baseline claim, E1-E4 verdicts."""
    assert artifact["schema"] == bench.SCHEMA
    assert artifact["schema_version"] == 1
    assert artifact["evidence_level"] == "mechanism"
    assert artifact["baseline_claim"] is False
    assert set(artifact["verdicts"]) == {"E1", "E2", "E3", "E4"}
    assert artifact["overall"] == "supported"
    assert artifact["status"] == "ok"
    assert artifact["meta"]["device"] == "cpu"
