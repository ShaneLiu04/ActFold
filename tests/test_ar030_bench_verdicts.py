"""AR030 benchmark-verdict tests (UT-3069..3078, UT-3101)."""

from __future__ import annotations

import math

import pytest
import torch

import scripts.ar030_paged_agent_bench as bench


@pytest.fixture(scope="module")
def artifact() -> dict:
    """Run the quick mechanism face once for the artifact-level tests."""
    return bench.run(quick=True)


def test_ut3069_e1_supported_on_exact_references() -> None:
    """UT-3069: E1 passes only when both faces are bit-exact."""
    assert bench.verdict_e1([0.0, 0.0], [0.0, 0.0]) is True
    assert bench.verdict_e1([1e-9, 0.0], [0.0, 0.0]) is False
    assert bench.verdict_e1([0.0], [float("nan")]) is False
    assert bench.verdict_e1([], []) is True


def _e2_args(**overrides):
    """The frozen quick-face E2 inputs (512 geometry, at-boundary budget)."""
    args = {
        "physical_per_layer": {0: 16, 1: 16},
        "budget_pages": {0: 16, 1: 16},
        "paged_hot_bytes": 98304,
        "paged_total_bytes": 98304,
        "plan_expected_bytes": 98304,
        "independent_page_bytes": 98304,
        "nonshared_hot_bytes": 442368,
        "nonshared_pages": 144,
        "physical_pages": 32,
    }
    args.update(overrides)
    return args


def test_ut3070_e2_frozen_bounds() -> None:
    """UT-3070: budget, plan-byte, dense-byte, ledger and sharing bounds."""
    assert bench.verdict_e2(**_e2_args()) is True
    assert bench.verdict_e2(**_e2_args(physical_per_layer={0: 17, 1: 16})) is False
    assert bench.verdict_e2(**_e2_args(physical_pages=31)) is False
    assert bench.verdict_e2(**_e2_args(independent_page_bytes=98305)) is False
    assert bench.verdict_e2(**_e2_args(plan_expected_bytes=98303)) is False
    assert bench.verdict_e2(**_e2_args(nonshared_hot_bytes=98303)) is False
    assert bench.verdict_e2(**_e2_args(nonshared_pages=63)) is False
    # sharing boundary: physical == 0.5 * nonshared passes at equality.
    assert bench.verdict_e2(**_e2_args(nonshared_pages=64)) is True


def test_ut3077_e2_boundary_and_missing_layers() -> None:
    """UT-3077: absent layers read zero; diverging ledgers refute E2."""
    budget = {0: 16, 1: 16}
    assert bench.verdict_e2(**_e2_args(physical_per_layer={0: 16}, physical_pages=16)) is True
    assert (
        bench.verdict_e2(
            physical_per_layer={},
            budget_pages=budget,
            paged_hot_bytes=0,
            paged_total_bytes=0,
            plan_expected_bytes=0,
            independent_page_bytes=0,
            nonshared_hot_bytes=0,
            nonshared_pages=0,
            physical_pages=0,
        )
        is True
    )
    assert bench.verdict_e2(**_e2_args(physical_per_layer={}, physical_pages=32)) is False


def test_ut3071_e3_fork_and_diff_page_semantics() -> None:
    """UT-3071: fork is byte-free; new-page SET == content-diff SET."""
    coords = [{(0, 7), (1, 7)}]
    assert bench.verdict_e3([0, 0], coords, [{(0, 7), (1, 7)}], [True, True], [True, True]) is True
    assert bench.verdict_e3([1, 0], coords, coords, [True, True], [True, True]) is False
    assert bench.verdict_e3([0, 0], coords, [{(0, 7), (1, 6)}], [True, True], [True, True]) is False
    assert bench.verdict_e3([0, 0], [set(), set()], [set(), set()], [True], [True]) is False
    assert bench.verdict_e3([0, 0], coords, coords, [True, False], [True, True]) is False
    assert bench.verdict_e3([0, 0], coords, coords, [True, True], [True, False]) is False


def test_ut3072_e4_blocks_on_failed_checks_or_bad_stats() -> None:
    """UT-3072: honesty checks gate the verdict; stats must be finite."""
    checks = {"commit_fail_k0_atomic": True, "stale_plan_rejected": True}
    assert bench.verdict_e4(checks, [0.0, 1.0, 2.0]) is True
    assert bench.verdict_e4({"stale_plan_rejected": False}, [0.0]) is False
    assert bench.verdict_e4(checks, [math.nan]) is False
    assert bench.verdict_e4(checks, [-1.0]) is False
    assert bench.verdict_e4(checks, [math.inf]) is False


def test_ut3073_overall_precedence() -> None:
    """UT-3073: blocked (honesty) > supported > refuted."""
    assert bench.overall_verdict(True, True, True, False) == "blocked"
    assert bench.overall_verdict(False, True, True, True) == "refuted"
    assert bench.overall_verdict(True, False, True, True) == "refuted"
    assert bench.overall_verdict(True, True, False, True) == "refuted"
    assert bench.overall_verdict(True, True, True, True) == "supported"


def test_ut3076_geometry_is_frozen() -> None:
    """UT-3076: the SRS agent geometry is a module constant, not data-driven."""
    assert bench.SEQ == 128
    assert bench.PAGE == 16
    assert bench.DEPTH == 2
    assert bench.CANDIDATES == 8
    assert bench.FORCED_TOKENS == 8
    assert bench.TAU == 0.9
    assert bench.PRESSURE_TOTAL == 512
    assert bench.FULL_TOTAL == 8192
    assert bench.PRESSURE_TOTAL // bench.DEPTH // bench.PAGE == 16
    assert bench.MIN_SHARING_RATIO == 0.5
    assert bench.QUICK_TIME_BUDGET_S == 60.0


def test_ut3078_verdicts_and_honesty_keys_are_frozen() -> None:
    """UT-3078: verdicts live in the script; honesty checks have fixed keys."""
    assert bench.SCHEMA == "ar030_paged_agent_bench"
    assert bench.SCHEMA_VERSION == 1
    for name in ("verdict_e1", "verdict_e2", "verdict_e3", "verdict_e4"):
        assert getattr(bench, name).__module__ == bench.__name__
    assert len(set(bench.HONESTY_KEYS)) == len(bench.HONESTY_KEYS)


def test_ut3074_quick_path_and_formal_refusal(tmp_path, monkeypatch) -> None:
    """UT-3074: quick writes the quick artifact; formal is BLOCKED-DEFERRED."""
    target = tmp_path / "quick.json"
    monkeypatch.setattr(bench, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(bench, "QUICK_PATH", target)
    assert bench.main(["--quick"]) == 0
    assert target.is_file()
    assert bench.main([]) == 3
    assert not (tmp_path / "results/optimization/ar030_paged_agent_bench.json").exists()


def test_ut3075_artifact_fields(artifact: dict) -> None:
    """UT-3075: mechanism evidence, frozen artifact keys, E1-E4 verdicts."""
    assert artifact["schema"] == bench.SCHEMA
    assert artifact["schema_version"] == 1
    assert artifact["evidence_level"] == "mechanism"
    assert artifact["baseline_claim"] is False
    assert artifact["mode"] == "quick"
    assert set(artifact["verdicts"]) == {"E1", "E2", "E3", "E4"}
    assert artifact["overall"] == "supported"
    assert artifact["status"] == "ok"
    assert artifact["meta"]["device"] == "cpu"
    assert set(artifact["honesty_checks"]) == set(bench.HONESTY_KEYS)
    assert all(artifact["honesty_checks"].values())
    assert artifact["geometry"]["budget_pages_per_layer"] == 16
    assert artifact["geometry"]["pressure_total_tokens"] == 512
    assert 0.0 < artifact["duration_s"] < bench.QUICK_TIME_BUDGET_S
    pressure = artifact["metrics"]["pressure"]
    assert pressure["max_abs_diff_vs_dense"] == 0.0
    assert pressure["paged_hot_bytes"] <= pressure["plan_expected_bytes"]
    assert pressure["paged_total_bytes"] == pressure["independent_page_bytes"]
    assert pressure["physical_pages_per_layer"] == pressure["budget_pages_per_layer"]
    assert pressure["byte_sharing_ratio"] >= bench.MIN_SHARING_RATIO
    assert pressure["page_sharing_ratio"] >= bench.MIN_SHARING_RATIO
    assert pressure["evictions"] == 0
    assert artifact["metrics"]["dense_reference"]["page_equivalents"] == pressure["nonshared_pages"]


def test_ut3101_cow_off_contrast_refutes_e2_end_to_end() -> None:
    """UT-3101 (SRS FR6 acceptance): with COW dedup disabled every write is a
    fresh page, so physical == nonshared, sharing_ratio == 0 < 0.5, E2 is
    False and the frozen precedence yields overall refuted."""
    torch.manual_seed(1234)
    model = bench.TinyDecoder().eval()
    dense_out, dense_bytes, dense_pages = bench._run_dense_face(model)
    cache = bench._paged_cache(bench.FULL_TOTAL, enable_cow=False)
    outs, aux = bench._run_agent_face(model, cache)
    ps = cache.page_stats()
    physical = int(ps["physical_pages"])
    # No dedup: every candidate write allocated a fresh page, so the paged
    # resident set equals the nonshared (dense oracle) page count.
    assert physical == dense_pages
    assert 1 - physical / dense_pages == 0.0
    assert physical > bench.MIN_SHARING_RATIO * dense_pages
    physical_per_layer: dict[int, int] = {}
    independent_page_bytes = 0
    for page in cache._store.iter_pages():
        physical_per_layer[page.layer] = physical_per_layer.get(page.layer, 0) + 1
        independent_page_bytes += int(page.nbytes)
    budget_pages = {
        layer: bench.FULL_TOTAL // bench.DEPTH // bench.PAGE for layer in range(bench.DEPTH)
    }
    plan = cache.build_plan(["root"] + [f"cand{j}" for j in range(bench.CANDIDATES)])
    diffs = [
        float((a - b).abs().max().item()) if a.shape == b.shape else float("inf")
        for a, b in zip(outs, dense_out)
    ]
    e1 = bench.verdict_e1(diffs, diffs)
    e2 = bench.verdict_e2(
        physical_per_layer,
        budget_pages,
        int(ps["hot_bytes"]),
        int(ps["hot_bytes"] + ps["cold_bytes"]),
        int(plan.expected_bytes),
        independent_page_bytes,
        int(dense_bytes),
        int(dense_pages),
        physical,
    )
    e3 = bench.verdict_e3(
        aux["fork_byte_deltas"],
        aux["new_coords"],
        aux["diff_coords"],
        aux["shared_ok"],
        aux["meta_entries_ok"],
    )
    assert e1 is True  # fresh pages carry identical content
    assert e2 is False  # the sharing clause refutes the mechanism contrast
    assert e3 is False  # every page is new, but content differs on 2 per candidate
    assert bench.overall_verdict(e1, e2, e3, True) == "refuted"
