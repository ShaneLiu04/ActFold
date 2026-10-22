"""AR031 bench-verdict tests (UT-3153..3160, UT-3166)."""

from __future__ import annotations

import math
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _bench() -> ModuleType:
    """Import the bench (deferred: Red evidence is the ModuleNotFoundError)."""
    import scripts.ar031_adaptive_uncertainty_bench as bench

    return bench


def _counts(blocks: tuple[int, int] | None) -> dict | None:
    """Convert a (matched, compared) pair into the frozen counts shape."""
    return None if blocks is None else {"matched": blocks[0], "compared": blocks[1]}


def _entry(
    index: int,
    baseline_mean: float | None,
    off_mean: float | None,
    full_mean: float | None,
    baseline_len: int | None = 34,
    off_len: int | None = 34,
    full_len: int | None = 34,
    off_blocks: tuple[int, int] | None = (4, 4),
    full_blocks: tuple[int, int] | None = (4, 4),
) -> dict:
    """Build one frozen-schema per-prompt quality entry with passing defaults."""
    return {
        "prompt_index": index,
        "posthoc_overall_mean": {
            "baseline": baseline_mean,
            "adaptive-off": off_mean,
            "full": full_mean,
        },
        "decode_length": {
            "baseline": baseline_len,
            "adaptive-off": off_len,
            "full": full_len,
        },
        "block_match_counts": {
            "adaptive-off": _counts(off_blocks),
            "full": _counts(full_blocks),
        },
    }


def _supported_faces() -> dict:
    """A supported-side assembly input (exact prompts, clear speedups)."""
    return {
        "quality_faces": {"per_prompt": [_entry(0, 0.3, 0.3, 0.3), _entry(1, 0.57, 0.57, 0.57)]},
        "p50_base_ms": 1300.0,
        "p50_off_ms": 600.0,
        "p50_full_ms": 440.0,
        "std_merged_h2_ms": 50.0,
        "std_merged_h3_ms": 50.0,
        "equiv_rate": 1.0,
        "budget_violations": 0,
        "ledger_ok": True,
        "repro_ok": True,
    }


def test_ut3153_h1_boundary_none_and_sentinels() -> None:
    """UT-3153: H1 gap boundary, None refutation, and report-only sentinels."""
    bench = _bench()
    ok = bench.verdict_h1({"per_prompt": [_entry(0, 0.3, 0.3, 0.3), _entry(1, 0.57, 0.57, 0.57)]})
    assert ok["verdict"] == "supported"
    assert ok["faces"] == {
        "a_abs_gap_le_1e-3": True,
        "b_decode_length_recovery": True,
        "c_merged_block_match_ge_off": True,
    }
    assert ok["numbers"]["merged_gap_full"] == 0.0
    assert ok["numbers"]["rate_full"] == 1.0
    assert ok["numbers"]["rate_off"] == 1.0
    assert ok["numbers"]["attribution_gap_ratio_vs_off"] == 0.0
    boundary = bench.verdict_h1(
        {"per_prompt": [_entry(0, 0.3, 0.3, 0.301), _entry(1, 0.3, 0.3, 0.301)]}
    )
    assert boundary["verdict"] == "supported"
    assert boundary["faces"]["a_abs_gap_le_1e-3"] is True
    assert boundary["numbers"]["merged_gap_full"] == pytest.approx(1e-3)
    drifted = 0.3 + 1e-3 + 1e-6
    beyond = bench.verdict_h1(
        {"per_prompt": [_entry(0, 0.3, 0.3, drifted), _entry(1, 0.3, 0.3, drifted)]}
    )
    assert beyond["verdict"] == "refuted"
    assert beyond["faces"]["a_abs_gap_le_1e-3"] is False
    with_none = bench.verdict_h1(
        {"per_prompt": [_entry(0, 0.3, 0.3, 0.3), _entry(1, 0.3, 0.3, None)]}
    )
    assert with_none["verdict"] == "refuted"
    assert with_none["faces"]["a_abs_gap_le_1e-3"] is False
    sentinel = bench.verdict_h1(
        {"per_prompt": [_entry(0, 0.3, 0.3, 0.3005), _entry(1, 0.57, 0.57, 0.5705)]}
    )
    assert sentinel["numbers"]["attribution_gap_ratio_vs_off"] == math.inf
    assert sentinel["verdict"] == "supported"


def test_ut3154_h2_two_sigma_boundaries() -> None:
    """UT-3154: H2 needs ratio strictly above 1 and delta beyond two sigma."""
    bench = _bench()
    out = bench.verdict_h2(1300.0, 440.0, 50.0)
    assert out["verdict"] == "supported"
    assert out["numbers"]["p50_base_ms"] == 1300.0
    assert out["numbers"]["p50_full_ms"] == 440.0
    assert out["numbers"]["ratio"] == pytest.approx(1300.0 / 440.0)
    assert out["numbers"]["delta_ms"] == pytest.approx(860.0)
    assert out["numbers"]["std_merged_ms"] == 50.0
    assert out["numbers"]["threshold_2sigma_ms"] == pytest.approx(100.0)
    assert isinstance(out["numbers"]["rule"], str)
    assert bench.verdict_h2(500.0, 500.0, 10.0)["verdict"] == "refuted"
    assert bench.verdict_h2(500.0, 450.0, 100.0)["verdict"] == "refuted"
    assert bench.verdict_h2(500.0, 400.0, 50.0)["verdict"] == "refuted"
    assert bench.verdict_h2(None, 440.0, 50.0)["verdict"] == "refuted"
    assert bench.verdict_h2(1300.0, None, 50.0)["verdict"] == "refuted"
    assert bench.verdict_h2(1300.0, 440.0, None)["verdict"] == "refuted"
    assert bench.verdict_h2(500.0, 0.0, 10.0)["verdict"] == "refuted"
    assert bench.verdict_h2(math.inf, 440.0, 50.0)["verdict"] == "refuted"
    assert bench.verdict_h2(float("nan"), 440.0, 50.0)["verdict"] == "refuted"


def test_ut3155_h3_same_caliber_on_off_vs_full() -> None:
    """UT-3155: H3 reuses the H2 caliber with adaptive-off in the base slot."""
    bench = _bench()
    out = bench.verdict_h3(600.0, 400.0, 50.0)
    assert out["verdict"] == "supported"
    assert out["numbers"]["p50_base_ms"] == 600.0
    assert out["numbers"]["p50_full_ms"] == 400.0
    expected_keys = set(bench.verdict_h2(1300.0, 440.0, 50.0)["numbers"])
    assert set(out["numbers"]) == expected_keys
    assert bench.verdict_h3(600.0, 600.0, 50.0)["verdict"] == "refuted"
    assert bench.verdict_h3(600.0, 590.0, 20.0)["verdict"] == "refuted"
    assert bench.verdict_h3(None, 400.0, 50.0)["verdict"] == "refuted"


def test_ut3156_h4_equivalence_budget_repro_veto() -> None:
    """UT-3156: H4 gates and the reproduction one-vote veto."""
    bench = _bench()
    assert bench.verdict_h4(1.0, 0, True, True)["verdict"] == "supported"
    refuted = bench.verdict_h4(0.5, 0, True, True)
    assert refuted["verdict"] == "refuted"
    assert refuted["faces"]["winner_equivalence_100pct"] is False
    assert bench.verdict_h4(None, 0, True, True)["verdict"] == "refuted"
    violations = bench.verdict_h4(1.0, 1, True, True)
    assert violations["verdict"] == "refuted"
    assert violations["faces"]["budget_zero_violations"] is False
    ledger = bench.verdict_h4(1.0, 0, False, True)
    assert ledger["verdict"] == "refuted"
    assert ledger["faces"]["ledger_within_bound"] is False
    blocked = bench.verdict_h4(1.0, 0, True, False)
    assert blocked["verdict"] == "blocked"
    assert blocked["blocked_reason"]
    assert blocked["faces"]["reproduction_ok"] is False
    assert blocked["numbers"]["equiv_rate"] == 1.0
    assert blocked["numbers"]["budget_violations"] == 0
    faces = _supported_faces()
    faces["repro_ok"] = False
    vetoed = bench.assemble_verdicts(faces)
    assert set(vetoed) >= {"h1", "h2", "h3", "h4", "overall"}
    assert vetoed["overall"] == "blocked"
    for key in ("h1", "h2", "h3", "h4"):
        assert vetoed[key]["verdict"] == "blocked"
        assert vetoed[key]["blocked_reason"]
    assert vetoed["h1"]["faces"] == {
        "a_abs_gap_le_1e-3": True,
        "b_decode_length_recovery": True,
        "c_merged_block_match_ge_off": True,
    }
    assert vetoed["h1"]["numbers"]["merged_gap_full"] == 0.0
    assert vetoed["h2"]["numbers"]["p50_base_ms"] == 1300.0
    assert vetoed["h2"]["numbers"]["p50_full_ms"] == 440.0
    assert vetoed["h3"]["numbers"]["p50_base_ms"] == 600.0
    assert vetoed["h4"]["faces"] == {
        "winner_equivalence_100pct": True,
        "budget_zero_violations": True,
        "ledger_within_bound": True,
        "reproduction_ok": False,
    }
    assert vetoed["h4"]["numbers"]["equiv_rate"] == 1.0
    assert vetoed["h4"]["numbers"]["budget_violations"] == 0
    faces["repro_ok"] = True
    passed = bench.assemble_verdicts(faces)
    assert passed["overall"] == "supported"
    for key in ("h1", "h2", "h3", "h4"):
        assert passed[key]["verdict"] == "supported"


def test_ut3157_schema_constants_and_recomputability() -> None:
    """UT-3157: frozen constants, precedence, and artifact recomputability."""
    bench = _bench()
    assert bench.SCHEMA == "ar031_adaptive_uncertainty_bench"
    assert bench.SCHEMA_VERSION == 1
    assert bench.ARMS == ("baseline", "adaptive-off", "full")
    assert bench.DEFAULT_OUTPUT_PATH == Path(
        "results/optimization/ar031_adaptive_uncertainty_bench.json"
    )
    assert bench.QUICK_OUTPUT_PATH == Path(
        "results/optimization/ar031_adaptive_uncertainty_bench_quick.json"
    )
    assert bench.DEFAULT_OUTPUT_PATH != bench.QUICK_OUTPUT_PATH
    assert "_quick" not in bench.DEFAULT_OUTPUT_PATH.name
    assert bench.REPS_DEFAULT == 30
    assert bench.QUICK_REPS == 3
    assert bench.UNCERTAINTY_MARGIN == 0.9
    assert bench.UNCERTAINTY_MARGIN_GAP == 0.05
    assert bench.BUDGET_RATIO == 0.15
    assert bench.BUDGET_WARMUP == 5
    assert bench.BUDGET_WINDOW == 10
    assert bench._H1_GAP_MAX == 1e-3
    assert bench._H1_RATIO_EPS == 1e-9
    assert bench.REQUIRED_TRANSFORMERS == "4.53.1"
    assert isinstance(bench.ARTIFACT_LIMITATION_NOTES, tuple)
    assert len(bench.ARTIFACT_LIMITATION_NOTES) > 0
    assert all(isinstance(note, str) and note for note in bench.ARTIFACT_LIMITATION_NOTES)
    assert any("eager" in note.lower() for note in bench.ARTIFACT_LIMITATION_NOTES)
    assert any(
        "graph" in note.lower() and "defer" in note.lower()
        for note in bench.ARTIFACT_LIMITATION_NOTES
    )
    for name in ("verdict_h1", "verdict_h2", "verdict_h3", "verdict_h4"):
        assert callable(getattr(bench, name))
    assert bench.overall_verdict("supported", "supported", "supported", "supported") == "supported"
    assert bench.overall_verdict("supported", "refuted", "supported", "supported") == "refuted"
    assert bench.overall_verdict("refuted", "blocked", "supported", "supported") == "blocked"
    assert bench.overall_verdict("supported", "inconclusive", "supported", "supported") == "refuted"
    faces = _supported_faces()
    result = bench.assemble_verdicts(faces)
    assert bench.verdict_h1(faces["quality_faces"])["verdict"] == result["h1"]["verdict"]
    h2 = bench.verdict_h2(faces["p50_base_ms"], faces["p50_full_ms"], faces["std_merged_h2_ms"])
    assert h2["verdict"] == result["h2"]["verdict"]
    h3 = bench.verdict_h3(faces["p50_off_ms"], faces["p50_full_ms"], faces["std_merged_h3_ms"])
    assert h3["verdict"] == result["h3"]["verdict"]
    h4 = bench.verdict_h4(
        faces["equiv_rate"], faces["budget_violations"], faces["ledger_ok"], faces["repro_ok"]
    )
    assert h4["verdict"] == result["h4"]["verdict"]
    assert (
        bench.overall_verdict(
            result["h1"]["verdict"],
            result["h2"]["verdict"],
            result["h3"]["verdict"],
            result["h4"]["verdict"],
        )
        == result["overall"]
    )


def test_ut3158_quick_isolation() -> None:
    """UT-3158: quick and formal artifact paths stay isolated."""
    bench = _bench()
    explicit = Path("results/optimization/explicit_override.json")
    assert bench.resolve_output_path(None, True) == bench.QUICK_OUTPUT_PATH
    assert bench.resolve_output_path(None, False) == bench.DEFAULT_OUTPUT_PATH
    assert bench.resolve_output_path(explicit, True) == explicit
    assert bench.resolve_output_path(explicit, False) == explicit
    assert bench.QUICK_OUTPUT_PATH != bench.DEFAULT_OUTPUT_PATH
    assert "quick" in bench.QUICK_OUTPUT_PATH.name


def test_ut3159_environment_gate(monkeypatch, tmp_path) -> None:
    """UT-3159: the environment gate refuses bad inputs before any run."""
    bench = _bench()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    with pytest.raises(RuntimeError):
        bench.require_environment(None, model_dir)
    with pytest.raises(RuntimeError):
        bench.require_environment("4.54.0", model_dir)
    with pytest.raises(RuntimeError):
        bench.require_environment("4.53.1", None)
    with pytest.raises(RuntimeError):
        bench.require_environment("4.53.1", tmp_path / "missing")
    with pytest.raises(RuntimeError):
        bench.require_environment("4.53.1", bare_dir)
    assert bench.require_environment("4.53.1", model_dir) is None
    formal = tmp_path / "formal.json"
    quick = tmp_path / "quick.json"
    monkeypatch.setattr(bench, "DEFAULT_OUTPUT_PATH", formal)
    monkeypatch.setattr(bench, "QUICK_OUTPUT_PATH", quick)
    rc = bench.main(["--model-path", str(tmp_path / "missing")])
    assert rc != 0
    assert not formal.exists()
    assert not quick.exists()


def test_ut3166_repro_pairing_is_per_prompt() -> None:
    """UT-3166: the double-run comparison pairs strictly per (arm, prompt).

    The first formal run compared a loop-leaked last-prompt pass-A dict
    against per-prompt pass-B sequences and fabricated cross-prompt
    mismatches (pass_a_len 66 vs pass_b_len 34 on prompt 0); the pairing
    must key BOTH passes by (arm, prompt_index).
    """
    bench = _bench()
    arms = ("baseline", "adaptive-off", "full")
    pass_a: dict[tuple[str, int], torch.Tensor] = {}
    pass_b: dict[tuple[str, int], torch.Tensor] = {}
    for pi, length in enumerate((34, 66)):
        for arm in arms:
            seq = torch.full((1, length), pi, dtype=torch.long)
            pass_a[(arm, pi)] = seq
            pass_b[(arm, pi)] = seq.clone()
    rows = bench.repro_mismatch_rows(pass_a, pass_b, arms, 2)
    assert rows == []  # identical per-prompt pairs: zero mismatches

    # a genuine single-row divergence reports exactly that row
    pass_b[("full", 1)] = torch.cat(
        [pass_b[("full", 1)], torch.zeros(1, 1, dtype=torch.long)], dim=1
    )
    rows = bench.repro_mismatch_rows(pass_a, pass_b, arms, 2)
    assert rows == [{"arm": "full", "prompt_index": 1, "pass_a_len": 66, "pass_b_len": 67}]


def test_ut3160_honest_negative_recorded_verbatim() -> None:
    """UT-3160: a refuting run is recorded verbatim, never rewritten."""
    bench = _bench()
    faces = {
        "quality_faces": {
            "per_prompt": [
                _entry(0, 0.3, 0.3, 0.9, full_len=66, off_blocks=(4, 4), full_blocks=(1, 4)),
                _entry(1, 0.5, 0.5, 0.95, full_len=70, off_blocks=(4, 4), full_blocks=(2, 4)),
            ]
        },
        "p50_base_ms": 500.0,
        "p50_off_ms": 480.0,
        "p50_full_ms": 510.0,
        "std_merged_h2_ms": 10.0,
        "std_merged_h3_ms": 10.0,
        "equiv_rate": 0.5,
        "budget_violations": 2,
        "ledger_ok": True,
        "repro_ok": True,
    }
    result = bench.assemble_verdicts(faces)
    assert result["overall"] == "refuted"
    for key in ("h1", "h2", "h3", "h4"):
        assert result[key]["verdict"] == "refuted"
    assert result["h1"]["faces"] == {
        "a_abs_gap_le_1e-3": False,
        "b_decode_length_recovery": False,
        "c_merged_block_match_ge_off": False,
    }
    assert result["h1"]["numbers"]["merged_gap_full"] == pytest.approx((0.9 - 0.3 + 0.95 - 0.5) / 2)
    assert result["h1"]["numbers"]["rate_full"] == pytest.approx(3 / 8)
    assert result["h1"]["numbers"]["rate_off"] == pytest.approx(1.0)
    assert result["h1"]["numbers"]["attribution_gap_ratio_vs_off"] == math.inf
    assert result["h2"]["numbers"]["p50_base_ms"] == 500.0
    assert result["h2"]["numbers"]["p50_full_ms"] == 510.0
    assert result["h3"]["numbers"]["p50_base_ms"] == 480.0
    assert result["h4"]["numbers"]["equiv_rate"] == 0.5
    assert result["h4"]["numbers"]["budget_violations"] == 2
