"""AR032 bench-verdict tests (UT-3178, UT-3179).

The Red evidence was the deferred import inside ``_bench()`` failing with
``ModuleNotFoundError`` before ``scripts/ar032_speculative_advance_bench.py``
existed (T004 Green delivered the module). The frozen signatures pinned
here are the contract the implementation matches.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest


def _bench() -> ModuleType:
    """Import the bench (deferred: Red evidence is the ModuleNotFoundError)."""
    import scripts.ar032_speculative_advance_bench as bench

    return bench


_DEFAULT_LENGTHS: list[int] = [34, 66]


def _quality(
    on_mean: float | None = 0.3,
    baseline_mean: float | None = 0.3,
    on_lengths: list[int] | None = _DEFAULT_LENGTHS,
    off_lengths: list[int] | None = _DEFAULT_LENGTHS,
    on_exact_match: float | None = 1.0,
    off_exact_match: float | None = 1.0,
) -> dict:
    """Build one frozen-schema H1 quality-faces dict with passing defaults.

    The length defaults are a module-level constant (not an in-body
    ``None`` sentinel) so an explicit ``on_lengths=None`` / ``off_lengths=
    None`` PROPAGATES into the dict -- the None-refutes loops below depend
    on it (the in-body sentinel form silently rewrote None back to the
    passing default and made the None branch untestable, found in T004
    Green).
    """
    return {
        "on_posthoc_mean": on_mean,
        "baseline_posthoc_mean": baseline_mean,
        "on_decode_lengths": on_lengths,
        "off_decode_lengths": off_lengths,
        "on_exact_match": on_exact_match,
        "off_exact_match": off_exact_match,
    }


def _supported_faces() -> dict:
    """A supported-side assembly input (aligned quality, clear speedups)."""
    return {
        "quality_faces": _quality(),
        "baseline_ms": 100.0,
        "on_ms": 50.0,
        "sigma": 5.0,
        "off_ms": 100.0,
        "off_serial_forwards": 10,
        "on_serial_forwards": 8,
        "eligible_blocks": 2,
        "canary_pass": True,
        "ledger_within": True,
        "repro_rows": [],
    }


def test_ut3178_h1_quality_faces() -> None:
    """UT-3178: H1 three-face conjunction, None refutations, IEEE margins."""
    bench = _bench()
    ok = bench.verdict_h1(_quality())
    assert ok["verdict"] == "supported"
    assert ok["faces"] == {
        "a_posthoc_gap_le_1e-3": True,
        "b_decode_lengths_equal": True,
        "c_exact_match_ge_off": True,
    }
    assert ok["numbers"]["on_posthoc_mean"] == 0.3
    assert ok["numbers"]["baseline_posthoc_mean"] == 0.3
    assert ok["numbers"]["on_decode_lengths"] == [34, 66]
    assert ok["numbers"]["off_decode_lengths"] == [34, 66]
    assert ok["numbers"]["on_exact_match"] == 1.0
    assert ok["numbers"]["off_exact_match"] == 1.0
    assert ok["numbers"]["posthoc_gap"] == 0.0

    # the gap face is ABSOLUTE: an improving drift within 1e-3 also passes
    improved = bench.verdict_h1(_quality(on_mean=0.2995))
    assert improved["verdict"] == "supported"
    assert improved["faces"]["a_posthoc_gap_le_1e-3"] is True

    # boundary: gap exactly 1e-3 + float noise below the 1e-9 IEEE margin
    noisy = bench.verdict_h1(_quality(on_mean=0.3 + 1e-3 + 1e-12))
    assert noisy["verdict"] == "supported"
    assert noisy["faces"]["a_posthoc_gap_le_1e-3"] is True
    assert noisy["numbers"]["posthoc_gap"] == pytest.approx(1e-3)
    beyond = bench.verdict_h1(_quality(on_mean=0.3 + 1e-3 + 1e-6))
    assert beyond["verdict"] == "refuted"
    assert beyond["faces"]["a_posthoc_gap_le_1e-3"] is False

    # any None posthoc mean refutes face (a) alone
    for kwargs in ({"on_mean": None}, {"baseline_mean": None}):
        with_none = bench.verdict_h1(_quality(**kwargs))
        assert with_none["verdict"] == "refuted"
        assert with_none["faces"]["a_posthoc_gap_le_1e-3"] is False
        assert with_none["faces"]["b_decode_lengths_equal"] is True
        assert with_none["faces"]["c_exact_match_ge_off"] is True

    # decode-length face: elementwise on == off; None or unequal refutes
    mismatched = bench.verdict_h1(_quality(on_lengths=[34, 67]))
    assert mismatched["verdict"] == "refuted"
    assert mismatched["faces"]["a_posthoc_gap_le_1e-3"] is True
    assert mismatched["faces"]["b_decode_lengths_equal"] is False
    assert mismatched["faces"]["c_exact_match_ge_off"] is True
    for kwargs in ({"on_lengths": None}, {"off_lengths": None}):
        none_lens = bench.verdict_h1(_quality(**kwargs))
        assert none_lens["verdict"] == "refuted"
        assert none_lens["faces"]["b_decode_lengths_equal"] is False

    # exact-match face: on >= off - 1e-9 slack - 1e-9 IEEE margin
    slack_ok = bench.verdict_h1(_quality(on_exact_match=0.75 - 1e-9, off_exact_match=0.75))
    assert slack_ok["verdict"] == "supported"
    assert slack_ok["faces"]["c_exact_match_ge_off"] is True
    slack_beyond = bench.verdict_h1(_quality(on_exact_match=0.75 - 1e-6, off_exact_match=0.75))
    assert slack_beyond["verdict"] == "refuted"
    assert slack_beyond["faces"]["c_exact_match_ge_off"] is False
    for kwargs in ({"on_exact_match": None}, {"off_exact_match": None}):
        none_em = bench.verdict_h1(_quality(**kwargs))
        assert none_em["verdict"] == "refuted"
        assert none_em["faces"]["c_exact_match_ge_off"] is False


def test_ut3178_h2_strict_speedup() -> None:
    """UT-3178: H2 needs ratio strictly above 1.0; reported 4-decimal + raw."""
    bench = _bench()
    out = bench.verdict_h2(100.0, 50.0, 5.0)
    assert out["verdict"] == "supported"
    assert out["numbers"]["baseline_ms"] == 100.0
    assert out["numbers"]["on_ms"] == 50.0
    assert out["numbers"]["sigma"] == 5.0
    assert out["numbers"]["speedup"] == pytest.approx(2.0)
    assert out["numbers"]["speedup_raw"] == pytest.approx(2.0)

    # the reported speedup is rounded to 4 decimals; the raw ratio is kept
    rounded = bench.verdict_h2(100.0, 30.0, 5.0)
    assert rounded["verdict"] == "supported"
    assert rounded["numbers"]["speedup"] == 3.3333
    assert rounded["numbers"]["speedup_raw"] == pytest.approx(100.0 / 30.0)

    # equal p50s: speedup == 1.0 is NOT > 1.0 strictly
    assert bench.verdict_h2(100.0, 100.0, 5.0)["verdict"] == "refuted"
    assert bench.verdict_h2(50.0, 100.0, 5.0)["verdict"] == "refuted"
    # None / non-positive inputs refute (and never raise)
    assert bench.verdict_h2(None, 50.0, 5.0)["verdict"] == "refuted"
    assert bench.verdict_h2(100.0, None, 5.0)["verdict"] == "refuted"
    assert bench.verdict_h2(-100.0, 50.0, 5.0)["verdict"] == "refuted"
    assert bench.verdict_h2(100.0, -50.0, 5.0)["verdict"] == "refuted"
    assert bench.verdict_h2(100.0, 0.0, 5.0)["verdict"] == "refuted"
    assert bench.verdict_h2(0.0, 50.0, 5.0)["verdict"] == "refuted"


def test_ut3178_h3_speedup_and_mechanism_count() -> None:
    """UT-3178: H3 two faces -- strict speedup AND the serial-forward count."""
    bench = _bench()
    out = bench.verdict_h3(100.0, 80.0, 10, 8, 2)
    assert out["verdict"] == "supported"
    assert out["faces"] == {
        "a_speedup_gt_1": True,
        "b_mechanism_count": True,
    }
    assert out["numbers"]["off_ms"] == 100.0
    assert out["numbers"]["on_ms"] == 80.0
    assert out["numbers"]["off_serial_forwards"] == 10
    assert out["numbers"]["on_serial_forwards"] == 8
    assert out["numbers"]["eligible_blocks"] == 2
    assert out["numbers"]["speedup"] == pytest.approx(1.25)

    # speedup face: <= 1.0 refutes even with a passing count face
    equal = bench.verdict_h3(100.0, 100.0, 10, 8, 2)
    assert equal["verdict"] == "refuted"
    assert equal["faces"]["a_speedup_gt_1"] is False
    assert equal["faces"]["b_mechanism_count"] is True
    assert bench.verdict_h3(80.0, 100.0, 10, 8, 2)["verdict"] == "refuted"

    # count face: off_serial - on_serial != eligible_blocks refutes even
    # with a passing speedup face (the count face alone can refute)
    wrong_count = bench.verdict_h3(100.0, 80.0, 10, 8, 3)
    assert wrong_count["verdict"] == "refuted"
    assert wrong_count["faces"]["a_speedup_gt_1"] is True
    assert wrong_count["faces"]["b_mechanism_count"] is False
    # on_serial >= 0 is part of the count face: -2 satisfies 10 - (-2) == 12
    negative_on = bench.verdict_h3(100.0, 80.0, 10, -2, 12)
    assert negative_on["verdict"] == "refuted"
    assert negative_on["faces"]["b_mechanism_count"] is False

    # None / negative inputs refute
    assert bench.verdict_h3(None, 80.0, 10, 8, 2)["verdict"] == "refuted"
    assert bench.verdict_h3(100.0, None, 10, 8, 2)["verdict"] == "refuted"
    assert bench.verdict_h3(100.0, 80.0, None, 8, 2)["verdict"] == "refuted"
    assert bench.verdict_h3(100.0, 80.0, 10, None, 2)["verdict"] == "refuted"
    assert bench.verdict_h3(100.0, 80.0, 10, 8, None)["verdict"] == "refuted"
    assert bench.verdict_h3(-100.0, 80.0, 10, 8, 2)["verdict"] == "refuted"
    assert bench.verdict_h3(100.0, -80.0, 10, 8, 2)["verdict"] == "refuted"
    assert bench.verdict_h3(100.0, 0.0, 10, 8, 2)["verdict"] == "refuted"


def test_ut3178_h4_conjunction() -> None:
    """UT-3178: H4 supported iff canary AND ledger AND no repro rows."""
    bench = _bench()
    ok = bench.verdict_h4(True, True, [])
    assert ok["verdict"] == "supported"
    assert ok["numbers"]["canary_pass"] is True
    assert ok["numbers"]["ledger_within"] is True
    assert ok["numbers"]["repro_rows"] == []
    # each failure face alone refutes
    canary = bench.verdict_h4(False, True, [])
    assert canary["verdict"] == "refuted"
    assert canary["numbers"]["canary_pass"] is False
    ledger = bench.verdict_h4(True, False, [])
    assert ledger["verdict"] == "refuted"
    assert ledger["numbers"]["ledger_within"] is False
    sample_rows = [{"arm": "reuse-on", "prompt_index": 0, "first": {}, "second": {}}]
    repro = bench.verdict_h4(True, True, sample_rows)
    assert repro["verdict"] == "refuted"
    assert repro["numbers"]["repro_rows"] == sample_rows


def test_ut3178_overall_verdict_precedence() -> None:
    """UT-3178: supported iff all four; any refuted -> refuted; else inconclusive."""
    bench = _bench()
    assert bench.overall_verdict("supported", "supported", "supported", "supported") == "supported"
    assert bench.overall_verdict("refuted", "supported", "supported", "supported") == "refuted"
    assert bench.overall_verdict("supported", "refuted", "supported", "supported") == "refuted"
    assert bench.overall_verdict("supported", "supported", "refuted", "supported") == "refuted"
    assert bench.overall_verdict("supported", "supported", "supported", "refuted") == "refuted"
    # refuted outranks inconclusive
    assert bench.overall_verdict("refuted", "inconclusive", "supported", "supported") == "refuted"
    # mixed supported/inconclusive without a refuted -> inconclusive
    assert (
        bench.overall_verdict("supported", "inconclusive", "supported", "supported")
        == "inconclusive"
    )
    assert (
        bench.overall_verdict("inconclusive", "inconclusive", "inconclusive", "inconclusive")
        == "inconclusive"
    )


def test_ut3178_constants_and_recomputability() -> None:
    """UT-3178: frozen constants, paths, and artifact recomputability."""
    bench = _bench()
    assert bench.SCHEMA == "ar032_speculative_advance_bench"
    assert bench.SCHEMA_VERSION == 1
    assert bench.ARMS == ("baseline", "reuse-off", "reuse-on")
    assert bench.DEFAULT_OUTPUT_PATH == Path(
        "results/optimization/ar032_speculative_advance_bench.json"
    )
    assert bench.QUICK_OUTPUT_PATH == Path(
        "results/optimization/ar032_speculative_advance_bench_quick.json"
    )
    assert bench.DEFAULT_OUTPUT_PATH != bench.QUICK_OUTPUT_PATH
    assert "_quick" not in bench.DEFAULT_OUTPUT_PATH.name
    assert bench.REPS_DEFAULT == 30
    assert bench.QUICK_REPS == 3
    assert bench.REQUIRED_TRANSFORMERS == "4.53.1"
    assert bench._H1_GAP_MAX == 1e-3
    assert bench._H1_RATIO_EPS == 1e-9
    assert bench._H3_EXACT_MATCH_SLACK == 1e-9
    for name in (
        "verdict_h1",
        "verdict_h2",
        "verdict_h3",
        "verdict_h4",
        "repro_mismatch_rows",
        "overall_verdict",
        "assemble_verdicts",
    ):
        assert callable(getattr(bench, name))
    faces = _supported_faces()
    result = bench.assemble_verdicts(faces)
    assert set(result) >= {"h1", "h2", "h3", "h4", "overall"}
    assert result["overall"] == "supported"
    for key in ("h1", "h2", "h3", "h4"):
        assert result[key]["verdict"] == "supported"
    # every assembled verdict recomputes from the artifact's own inputs
    assert result["h1"]["verdict"] == bench.verdict_h1(faces["quality_faces"])["verdict"]
    assert (
        result["h2"]["verdict"]
        == bench.verdict_h2(faces["baseline_ms"], faces["on_ms"], faces["sigma"])["verdict"]
    )
    assert (
        result["h3"]["verdict"]
        == bench.verdict_h3(
            faces["off_ms"],
            faces["on_ms"],
            faces["off_serial_forwards"],
            faces["on_serial_forwards"],
            faces["eligible_blocks"],
        )["verdict"]
    )
    assert (
        result["h4"]["verdict"]
        == bench.verdict_h4(faces["canary_pass"], faces["ledger_within"], faces["repro_rows"])[
            "verdict"
        ]
    )
    assert (
        bench.overall_verdict(
            result["h1"]["verdict"],
            result["h2"]["verdict"],
            result["h3"]["verdict"],
            result["h4"]["verdict"],
        )
        == result["overall"]
    )


def test_ut3178_resolve_output_path() -> None:
    """UT-3178: quick and formal artifact paths stay isolated."""
    bench = _bench()
    explicit = Path("results/optimization/explicit_override.json")
    assert bench.resolve_output_path(None, False) == bench.DEFAULT_OUTPUT_PATH
    assert bench.resolve_output_path(None, True) == bench.QUICK_OUTPUT_PATH
    assert bench.resolve_output_path(explicit, False) == explicit
    assert bench.resolve_output_path(explicit, True) == explicit
    assert bench.QUICK_OUTPUT_PATH != bench.DEFAULT_OUTPUT_PATH
    assert "quick" in bench.QUICK_OUTPUT_PATH.name


def test_ut3178_require_environment(tmp_path) -> None:
    """UT-3178: the environment gate refuses bad inputs before any run."""
    bench = _bench()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    assert bench.require_environment("4.53.1", model_dir) is None
    with pytest.raises(RuntimeError):
        bench.require_environment(None, model_dir)
    with pytest.raises(RuntimeError):
        bench.require_environment("4.54.0", model_dir)
    with pytest.raises(RuntimeError):
        bench.require_environment("4.53.1", None)
    with pytest.raises(RuntimeError):
        bench.require_environment("4.53.1", tmp_path / "missing")


def test_ut3179_repro_pairing_is_per_prompt() -> None:
    """UT-3179: the double-run comparison pairs strictly per (arm, prompt).

    The AR031 first-formal-run incident (UT-3166): a loop-leaked
    last-prompt payload reused across the collection loop contaminated
    every earlier row's ``first`` with the LAST prompt's cargo and
    fabricated cross-prompt 66-vs-34 mismatches.  The pairing key is
    (arm, prompt_index) and a row mismatches iff its own first/second
    payloads differ.
    """
    bench = _bench()
    arms = ("baseline", "reuse-off", "reuse-on")

    # correct per-(arm, prompt) rows: first == second by value -> no rows
    rows: list[dict] = []
    for pi, length in enumerate((34, 66)):
        for arm in arms:
            payload = {"decode_length": length, "tokens_head": [pi, pi]}
            rows.append(
                {
                    "arm": arm,
                    "prompt_index": pi,
                    "first": dict(payload),
                    "second": dict(payload),
                }
            )
    assert bench.repro_mismatch_rows(rows) == []

    # a genuine single-row divergence reports exactly that row
    diverged = {
        "arm": "reuse-on",
        "prompt_index": 1,
        "first": {"decode_length": 66, "tokens_head": [1, 1]},
        "second": {"decode_length": 67, "tokens_head": [1, 1]},
    }
    rows[5] = diverged  # index 5 == ("reuse-on", 1)
    mismatches = bench.repro_mismatch_rows(rows)
    assert len(mismatches) == 1
    assert mismatches[0]["arm"] == "reuse-on"
    assert mismatches[0]["prompt_index"] == 1
    assert mismatches[0]["first"] == {"decode_length": 66, "tokens_head": [1, 1]}
    assert mismatches[0]["second"] == {"decode_length": 67, "tokens_head": [1, 1]}

    # the loop-leak regression shape: ONE "first" payload dict reused and
    # mutated across the collection loop, so after the loop every row's
    # first holds the LAST prompt's result -- each prompt-0 row compares
    # prompt-1 cargo (66) against its own second (34) and mismatches
    leaked_rows: list[dict] = []
    shared_first: dict = {}
    for pi, length in enumerate((34, 66)):
        for arm in arms:
            shared_first["decode_length"] = length
            shared_first["tokens_head"] = [pi, pi]
            leaked_rows.append(
                {
                    "arm": arm,
                    "prompt_index": pi,
                    "first": shared_first,
                    "second": {"decode_length": length, "tokens_head": [pi, pi]},
                }
            )
    leaked = bench.repro_mismatch_rows(leaked_rows)
    assert len(leaked) == 3
    assert {(row["arm"], row["prompt_index"]) for row in leaked} == {(arm, 0) for arm in arms}

    # the repro rows are preserved verbatim in verdict_h4's numbers
    h4 = bench.verdict_h4(True, True, leaked)
    assert h4["verdict"] == "refuted"
    assert h4["numbers"]["repro_rows"] == leaked

    # assembly: repro rows refute H4 (and thus the overall verdict) while
    # the other hypotheses keep their own recomputable verdicts
    faces = _supported_faces()
    faces["repro_rows"] = leaked
    result = bench.assemble_verdicts(faces)
    assert result["h4"]["verdict"] == "refuted"
    assert result["overall"] == "refuted"
    assert result["h1"]["verdict"] == bench.verdict_h1(faces["quality_faces"])["verdict"]
    assert result["h4"]["numbers"]["repro_rows"] == leaked
