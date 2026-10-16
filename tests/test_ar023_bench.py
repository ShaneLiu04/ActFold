"""Red-phase tests for AR023/T005: the cross-family four-mode speculative bench.

Target (design.md §4.1.8 + §6.2, srs FR5): ``scripts/ar023_cross_family_bench.py``
— ``--family llada|dream`` cross-family four-mode speculative benchmark
(baseline / folded-serial / folded-batched / folded-batched-greedy) with the
frozen hierarchical verdict gates (H1 quality, H2 speed, H4 full-stack) plus
the baseline double-run reproduction self-check and the degraded-family
blocked semantics inherited from the AR023 family probe conclusions.

Only the CI-safe pure-logic surface is tested here — no GPU, no network, no
real weights:

* protocol constants (``SEED`` / ``REPS_DEFAULT`` / ``QUICK_REPS`` /
  ``FULL_NUM_TOKENS`` / ``QUICK_NUM_TOKENS`` / ``BLOCK_GRID`` /
  ``NUM_CANDIDATES`` / ``CANDIDATE_TOP_K`` / ``SPECULATIVE_STEPS`` / ``MODES``
  / ``FOLDED_MODES`` / ``FAMILIES`` / ``DEFAULT_OUTPUT_PATHS``) and the
  lightweight-import contract (module top-level must stay stdlib-only);
* ``FAMILY_RECIPES`` frozen per-family sampling recipes;
* ``h1_quality_verdict``: the frozen hierarchical quality gate — gap shrink
  ratio (face a), length recovery (face b), strict block-exact improvement
  (face c), the no-fabricated-numbers None rule, the gap_bat==0 sentinels,
  the compared==0 rule, and the non-blocking STRONG layer;
* ``h2_speed_verdict``: strict ``> 1.0`` p50 speedup with the report-only
  aligned-vs-batched ratio;
* ``h4_fullstack_verdict``: canary/ledger conjunction with eager-neutral
  graph faces and report-only stable ratios;
* ``reproduction_verdict`` + ``mark_all_blocked``: strict baseline double-run
  equality and the non-mutating blocked overlay;
* ``probe_conclusion`` + ``build_blocked_artifact``: probe conclusion
  extraction with safe degrade and the blocked artifact schema (modes/quality
  None, top-level ``blocked_reason`` findable);
* ``parse_args`` CLI contract (family required, choices exactly the two
  families, exit code 2 on missing/invalid, ``--quick``/``--reps``/``--out``/
  ``--models-root``/``--probe-artifact`` + the ``_hf_env`` flags);
* output-path resolution (``quick_output_path`` / ``resolve_output_path``)
  and the ``build_artifact`` schema.

The target module is imported lazily inside each test so the Red state —
``scripts.ar023_cross_family_bench`` not existing yet — fails tests
individually with ``ModuleNotFoundError`` instead of aborting module
collection. A subprocess test additionally pins the lightweight-import
contract: importing the module must not pull torch / transformers /
bitsandbytes at module top (the heavy deps stay lazy inside the run layer).
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PROBE_ARTIFACT_DEFAULT = "results/optimization/ar023_family_probe.json"


# ---------------------------------------------------------------------------
# Fixture-style helpers
# ---------------------------------------------------------------------------


def _quality_fixture(
    baseline_means: list[float | None],
    batched_means: list[float | None],
    aligned_means: list[float | None],
    baseline_lens: list[int],
    batched_lens: list[int],
    aligned_lens: list[int],
    batched_exact: tuple[int, int],
    aligned_exact: tuple[int, int],
) -> dict:
    """Build a synthetic quality_faces input for h1_quality_verdict.

    ``*_means`` are per-prompt posthoc overall means keyed by the three
    scoreable modes; ``*_lens`` are per-prompt decode lengths; the exact
    tuples are (matched, compared) merged across prompts per folded mode.
    """
    per_prompt = []
    for i in range(len(baseline_means)):
        per_prompt.append(
            {
                "prompt_index": i,
                "posthoc_overall_mean": {
                    "baseline": baseline_means[i],
                    "folded-batched": batched_means[i],
                    "folded-batched-greedy": aligned_means[i],
                },
                "decode_length": {
                    "baseline": baseline_lens[i],
                    "folded-batched": batched_lens[i],
                    "folded-batched-greedy": aligned_lens[i],
                },
                "block_exact": {
                    "folded-batched": {
                        "matched": batched_exact[0],
                        "compared": batched_exact[1],
                    },
                    "folded-batched-greedy": {
                        "matched": aligned_exact[0],
                        "compared": aligned_exact[1],
                    },
                },
            }
        )
    return {"per_prompt": per_prompt}


_PASS_FIXTURE = _quality_fixture(
    baseline_means=[0.40, 0.40],
    batched_means=[0.20, 0.20],
    aligned_means=[0.35, 0.35],
    baseline_lens=[32, 64],
    batched_lens=[160, 77],
    aligned_lens=[33, 66],
    batched_exact=(0, 64),
    aligned_exact=(32, 64),
)


# ---------------------------------------------------------------------------
# (1) UT-2326 protocol constants + lightweight-import contract
# ---------------------------------------------------------------------------


def test_ut_2326_protocol_constants_and_import_contract() -> None:
    """Protocol constants are frozen verbatim and importing the module pulls
    no heavy deps (subprocess-isolated so a torch-importing pytest session
    cannot mask a violation)."""
    from scripts.ar023_cross_family_bench import (
        BLOCK_GRID,
        CANDIDATE_TOP_K,
        DEFAULT_OUTPUT_PATHS,
        FAMILIES,
        FOLDED_MODES,
        FULL_NUM_TOKENS,
        MODES,
        NUM_CANDIDATES,
        QUICK_NUM_TOKENS,
        QUICK_REPS,
        REPS_DEFAULT,
        SEED,
        SPECULATIVE_STEPS,
    )

    assert SEED == 2026
    assert REPS_DEFAULT == 30
    assert QUICK_REPS == 3
    assert FULL_NUM_TOKENS == 128
    assert QUICK_NUM_TOKENS == 64
    assert BLOCK_GRID == 32
    assert NUM_CANDIDATES == 4
    assert CANDIDATE_TOP_K == 10
    assert SPECULATIVE_STEPS == 8

    assert MODES == (
        "baseline",
        "folded-serial",
        "folded-batched",
        "folded-batched-greedy",
    )
    assert FOLDED_MODES == ("folded-serial", "folded-batched", "folded-batched-greedy")
    assert FAMILIES == ("llada", "dream")

    assert DEFAULT_OUTPUT_PATHS == {
        "llada": Path("results/optimization/ar023_llada_bench.json"),
        "dream": Path("results/optimization/ar023_dream_bench.json"),
    }

    code = (
        "import sys\n"
        "import scripts.ar023_cross_family_bench as m\n"
        "heavy = [n for n in ('torch', 'transformers', 'bitsandbytes')"
        " if n in sys.modules]\n"
        "assert not heavy, f'module top-level import pulled heavy deps: {heavy}'\n"
        "assert m.SEED == 2026\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"lightweight-import probe failed (exit {proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# (2) UT-2327 family recipes frozen
# ---------------------------------------------------------------------------


def test_ut_2327_family_recipes_frozen() -> None:
    """FAMILY_RECIPES pins the per-family sampling recipes verbatim."""
    from scripts.ar023_cross_family_bench import FAMILY_RECIPES

    assert FAMILY_RECIPES == {
        "llada": {"block_size": 32, "num_steps": 32, "remasking": "low_confidence"},
        "dream": {"alg": "maskgit_plus", "eps": 1e-3, "num_steps": "1:1 with num_tokens"},
    }
    assert set(FAMILY_RECIPES) == {"llada", "dream"}


# ---------------------------------------------------------------------------
# (3) UT-2328 h1_quality_verdict: the frozen hierarchical quality gate
# ---------------------------------------------------------------------------


def test_ut_2328_h1_quality_verdict_frozen_gate() -> None:
    """The passing fixture supports all three hard faces; each face fails
    alone; None means, gap_bat==0 sentinels, compared==0, the 0.5 boundary,
    and the STRONG layer all follow the frozen formulas."""
    from scripts.ar023_cross_family_bench import h1_quality_verdict

    # -- passing fixture: verdict supported, all faces True, strong all None
    ok = h1_quality_verdict(_PASS_FIXTURE)
    assert ok["verdict"] == "supported"
    assert ok["faces"] == {
        "a_gap_shrink_ratio_le_0_5": True,
        "b_length_recovery": True,
        "c_block_exact_strict_improvement": True,
    }
    assert ok["strong"] == {
        "quality_full_alignment": None,
        "exact_all_hit": None,
        "length_equals_baseline": None,
    }
    # merged gaps: batched -0.20, aligned -0.05 -> ratio 0.25
    numbers = ok["numbers"]
    assert numbers["gap"]["folded-batched"] == pytest.approx(-0.2, abs=1e-9)
    assert numbers["gap"]["folded-batched-greedy"] == pytest.approx(-0.05, abs=1e-9)
    assert numbers["ratio"] == pytest.approx(0.25, abs=1e-9)
    # merged exact rates: batched 0/64, aligned 32/64 = 0.5
    assert numbers["exact"]["folded-batched"] == pytest.approx(0.0, abs=1e-9)
    assert numbers["exact"]["folded-batched-greedy"] == pytest.approx(0.5, abs=1e-9)

    # -- face (a) alone: aligned gap -0.12 vs batched -0.20 -> ratio 0.6 > 0.5
    face_a = h1_quality_verdict(
        _quality_fixture(
            baseline_means=[0.40, 0.40],
            batched_means=[0.20, 0.20],
            aligned_means=[0.28, 0.28],
            baseline_lens=[32, 64],
            batched_lens=[160, 77],
            aligned_lens=[33, 66],
            batched_exact=(0, 64),
            aligned_exact=(32, 64),
        )
    )
    assert face_a["verdict"] == "refuted"
    assert face_a["faces"] == {
        "a_gap_shrink_ratio_le_0_5": False,
        "b_length_recovery": True,
        "c_block_exact_strict_improvement": True,
    }

    # -- face (a) boundary: ratio EXACTLY 0.5 -> face (a) True (<=)
    boundary = h1_quality_verdict(
        _quality_fixture(
            baseline_means=[0.40, 0.40],
            batched_means=[0.20, 0.20],
            aligned_means=[0.30, 0.30],
            baseline_lens=[32, 64],
            batched_lens=[160, 77],
            aligned_lens=[33, 66],
            batched_exact=(0, 64),
            aligned_exact=(32, 64),
        )
    )
    assert boundary["verdict"] == "supported"
    assert boundary["faces"]["a_gap_shrink_ratio_le_0_5"] is True
    assert boundary["numbers"]["ratio"] == pytest.approx(0.5, abs=1e-9)

    # -- face (b) alone: prompt 1 fails length recovery
    face_b = h1_quality_verdict(
        _quality_fixture(
            baseline_means=[0.40, 0.40],
            batched_means=[0.20, 0.20],
            aligned_means=[0.35, 0.35],
            baseline_lens=[32, 64],
            batched_lens=[160, 77],
            aligned_lens=[33, 200],
            batched_exact=(0, 64),
            aligned_exact=(32, 64),
        )
    )
    assert face_b["verdict"] == "refuted"
    assert face_b["faces"] == {
        "a_gap_shrink_ratio_le_0_5": True,
        "b_length_recovery": False,
        "c_block_exact_strict_improvement": True,
    }

    # -- face (c) alone: equal exact rates (0.5 vs 0.5) -> strict > fails
    face_c = h1_quality_verdict(
        _quality_fixture(
            baseline_means=[0.40, 0.40],
            batched_means=[0.20, 0.20],
            aligned_means=[0.35, 0.35],
            baseline_lens=[32, 64],
            batched_lens=[160, 77],
            aligned_lens=[33, 66],
            batched_exact=(32, 64),
            aligned_exact=(32, 64),
        )
    )
    assert face_c["verdict"] == "refuted"
    assert face_c["faces"] == {
        "a_gap_shrink_ratio_le_0_5": True,
        "b_length_recovery": True,
        "c_block_exact_strict_improvement": False,
    }

    # -- None posthoc mean anywhere -> hard refuted, no fabricated numbers
    none_mean = h1_quality_verdict(
        _quality_fixture(
            baseline_means=[0.40, None],
            batched_means=[0.20, 0.20],
            aligned_means=[0.35, 0.35],
            baseline_lens=[32, 64],
            batched_lens=[160, 77],
            aligned_lens=[33, 66],
            batched_exact=(0, 64),
            aligned_exact=(32, 64),
        )
    )
    assert none_mean["verdict"] == "refuted"

    # -- gap_bat == 0 & gap_aln == 0 -> ratio 0.0, face (a) True
    both_zero = h1_quality_verdict(
        _quality_fixture(
            baseline_means=[0.40, 0.40],
            batched_means=[0.40, 0.40],
            aligned_means=[0.40, 0.40],
            baseline_lens=[32, 64],
            batched_lens=[32, 64],
            aligned_lens=[32, 64],
            batched_exact=(32, 64),
            aligned_exact=(64, 64),
        )
    )
    assert both_zero["numbers"]["ratio"] == 0.0
    assert both_zero["faces"]["a_gap_shrink_ratio_le_0_5"] is True

    # -- gap_bat == 0 & gap_aln != 0 -> ratio inf, face (a) False
    aln_only = h1_quality_verdict(
        _quality_fixture(
            baseline_means=[0.40, 0.40],
            batched_means=[0.40, 0.40],
            aligned_means=[0.38, 0.38],
            baseline_lens=[32, 64],
            batched_lens=[32, 64],
            aligned_lens=[32, 64],
            batched_exact=(32, 64),
            aligned_exact=(64, 64),
        )
    )
    assert aln_only["numbers"]["ratio"] == math.inf
    assert aln_only["faces"]["a_gap_shrink_ratio_le_0_5"] is False

    # -- compared == 0 -> rate None -> face (c) fails
    no_compared = h1_quality_verdict(
        _quality_fixture(
            baseline_means=[0.40, 0.40],
            batched_means=[0.20, 0.20],
            aligned_means=[0.35, 0.35],
            baseline_lens=[32, 64],
            batched_lens=[160, 77],
            aligned_lens=[33, 66],
            batched_exact=(0, 0),
            aligned_exact=(0, 0),
        )
    )
    assert no_compared["verdict"] == "refuted"
    assert no_compared["faces"]["c_block_exact_strict_improvement"] is False

    # -- STRONG layer: aligned gaps all 0.0, exact rate 1.0, lengths equal
    strong = h1_quality_verdict(
        _quality_fixture(
            baseline_means=[0.40, 0.40],
            batched_means=[0.20, 0.20],
            aligned_means=[0.40, 0.40],
            baseline_lens=[32, 64],
            batched_lens=[160, 77],
            aligned_lens=[32, 64],
            batched_exact=(0, 64),
            aligned_exact=(64, 64),
        )
    )
    assert strong["verdict"] == "supported"
    assert strong["strong"] == {
        "quality_full_alignment": "supported",
        "exact_all_hit": "supported",
        "length_equals_baseline": "supported",
    }


# ---------------------------------------------------------------------------
# (4) UT-2329 h2_speed_verdict
# ---------------------------------------------------------------------------


def test_ut_2329_h2_speed_verdict_strict_threshold() -> None:
    """Speedup is baseline/aligned p50 with a STRICT > 1.0 threshold; the
    aligned-vs-batched ratio is report-only and never flips the verdict."""
    from scripts.ar023_cross_family_bench import h2_speed_verdict

    fast = h2_speed_verdict(1300.6, 440.9, 450.0)
    assert fast["verdict"] == "supported"
    assert fast["numbers"]["speedup_p50_vs_baseline"] == round(1300.6 / 440.9, 4)
    assert fast["numbers"]["speedup_p50_vs_baseline"] == pytest.approx(2.9498, abs=1e-4)
    assert fast["numbers"]["aligned_vs_batched_ratio"] == round(440.9 / 450.0, 4)

    equal = h2_speed_verdict(440.9, 440.9, 450.0)
    assert equal["verdict"] == "refuted"
    assert equal["numbers"]["speedup_p50_vs_baseline"] == 1.0

    barely = h2_speed_verdict(440.95, 440.9, 450.0)
    assert barely["verdict"] == "supported"

    # the report ratio is wildly unfavorable yet the verdict is unchanged
    wild = h2_speed_verdict(1300.6, 440.9, 4.4)
    assert wild["verdict"] == "supported"
    assert wild["numbers"]["aligned_vs_batched_ratio"] == round(440.9 / 4.4, 4)

    # Erratum (Green phase): ``equal`` was constructed as (440.9, 440.9,
    # 450.0), so the original ``for res in (fast, equal, wild)`` loop pinned
    # ``p50_baseline_ms == 1300.6`` on a result whose baseline IS 440.9 --
    # unsatisfiable by any non-fabricating implementation. The shared pins
    # apply to (fast, wild); ``equal`` pins its own recorded inputs.
    for res in (fast, wild):
        assert res["numbers"]["p50_baseline_ms"] == 1300.6
        assert res["numbers"]["p50_aligned_ms"] == 440.9
        assert res["numbers"]["rule"] == "supported iff p50(baseline)/p50(aligned) > 1.0"
    assert equal["numbers"]["p50_baseline_ms"] == 440.9
    assert equal["numbers"]["p50_aligned_ms"] == 440.9
    assert equal["numbers"]["rule"] == "supported iff p50(baseline)/p50(aligned) > 1.0"


# ---------------------------------------------------------------------------
# (5) UT-2330 h4_fullstack_verdict
# ---------------------------------------------------------------------------


def test_ut_2330_h4_fullstack_verdict_canary_ledger_graph() -> None:
    """All canaries+ledgers truthy -> supported; any single False refutes;
    graph_enabled=False is eager-neutral (note, never refutes); stable ratios
    are recorded with their mean."""
    from scripts.ar023_cross_family_bench import h4_fullstack_verdict

    canaries = {m: True for m in ("folded-serial", "folded-batched", "folded-batched-greedy")}
    ledgers = dict(canaries)
    ratios = {"folded-serial": 0.8, "folded-batched": 0.75, "folded-batched-greedy": 0.75}

    ok = h4_fullstack_verdict(canaries, ledgers, False, ratios)
    assert ok["verdict"] == "supported"
    assert ok["numbers"]["canary_degradation_zero"] == canaries
    assert ok["numbers"]["ledger_within_bound"] == ledgers
    assert ok["numbers"]["graph_enabled"] is False
    assert isinstance(ok["numbers"]["graph_note"], str) and ok["numbers"]["graph_note"]
    assert ok["numbers"]["stable_ratios"] == ratios
    assert ok["numbers"]["stable_ratio_mean"] == round((0.8 + 0.75 + 0.75) / 3, 6)

    one_canary = dict(canaries)
    one_canary["folded-batched"] = False
    bad_canary = h4_fullstack_verdict(one_canary, ledgers, False, ratios)
    assert bad_canary["verdict"] == "refuted"

    one_ledger = dict(ledgers)
    one_ledger["folded-batched-greedy"] = False
    bad_ledger = h4_fullstack_verdict(canaries, one_ledger, False, ratios)
    assert bad_ledger["verdict"] == "refuted"

    graph_on = h4_fullstack_verdict(canaries, ledgers, True, ratios)
    assert graph_on["verdict"] == "supported"
    assert graph_on["numbers"]["graph_enabled"] is True
    assert graph_on["numbers"]["graph_note"] is None


# ---------------------------------------------------------------------------
# (6) UT-2331 reproduction_verdict + mark_all_blocked
# ---------------------------------------------------------------------------


def test_ut_2331_reproduction_verdict_and_mark_all_blocked() -> None:
    """Matching double-run rows pass; content mismatch and row-count mismatch
    both fail with the first divergence index; mark_all_blocked returns a NEW
    dict with verdict blocked + reason, preserving numbers and never mutating
    the input."""
    from scripts.ar023_cross_family_bench import mark_all_blocked, reproduction_verdict

    rows_a = [[1, 2, 3], [4, 5], [7]]
    assert reproduction_verdict(rows_a, [list(r) for r in rows_a]) == {"ok": True}

    rows_b = [[1, 2, 3], [4, 6], [7]]
    mismatch = reproduction_verdict(rows_a, rows_b)
    assert mismatch["ok"] is False
    assert mismatch["reproduction_anomaly"] is True
    assert mismatch["mismatch"]["first_diff_prompt_index"] == 1
    assert mismatch["mismatch"]["a_len"] == 2
    assert mismatch["mismatch"]["b_len"] == 2

    # different row counts: common prefix equal -> divergence at index 2
    count_mismatch = reproduction_verdict(rows_a, rows_a[:2])
    assert count_mismatch["ok"] is False
    assert count_mismatch["reproduction_anomaly"] is True
    assert count_mismatch["mismatch"]["first_diff_prompt_index"] == 2

    verdicts = {
        "h1": {"verdict": "supported", "numbers": {"ratio": 0.25}},
        "h2": {"verdict": "refuted", "numbers": {"speedup": 1.0}},
        "h4": {"verdict": "supported", "numbers": {"stable_ratio_mean": 0.7}},
    }
    original = {k: dict(v) for k, v in verdicts.items()}
    blocked = mark_all_blocked(verdicts, "reproduction anomaly")
    assert set(blocked) == {"h1", "h2", "h4"}
    for key in ("h1", "h2", "h4"):
        assert blocked[key]["verdict"] == "blocked"
        assert blocked[key]["blocked_reason"] == "reproduction anomaly"
        assert blocked[key]["numbers"] == original[key]["numbers"]
    assert verdicts == original, "mark_all_blocked must not mutate its input"


# ---------------------------------------------------------------------------
# (7) UT-2332 probe_conclusion + build_blocked_artifact
# ---------------------------------------------------------------------------


def test_ut_2332_probe_conclusion_and_build_blocked_artifact() -> None:
    """probe_conclusion extracts the family conclusions with safe degrade for
    missing entries; build_blocked_artifact emits status blocked, None
    modes/quality, a top-level blocked_reason, and the skipped reproduction —
    never fabricated numbers."""
    from scripts.ar023_cross_family_bench import build_blocked_artifact, probe_conclusion

    probe = {
        "schema_version": 1,
        "meta": {},
        "families": {
            "llada": {
                "conclusions": {
                    "dtype_choice": "bf16",
                    "graph_enabled": False,
                    "degraded": False,
                    "degrade_reason": None,
                },
                "faces": {},
            },
            "dream": {
                "conclusions": {
                    "dtype_choice": "bf16",
                    "graph_enabled": False,
                    "degraded": True,
                    "degrade_reason": "smoke failed",
                },
                "faces": {},
            },
        },
    }

    healthy = probe_conclusion(probe, "llada")
    assert healthy == {
        "degraded": False,
        "degrade_reason": None,
        "dtype_choice": "bf16",
        "graph_enabled": False,
    }

    degraded = probe_conclusion(probe, "dream")
    assert degraded["degraded"] is True
    assert degraded["degrade_reason"] == "smoke failed"
    assert degraded["dtype_choice"] == "bf16"
    assert degraded["graph_enabled"] is False

    missing_family = probe_conclusion(probe, "zzz")
    assert missing_family["degraded"] is True
    assert isinstance(missing_family["degrade_reason"], str)
    assert missing_family["degrade_reason"] != ""

    no_conclusions = {
        "schema_version": 1,
        "meta": {},
        "families": {"dream": {"faces": {}}},
    }
    absent = probe_conclusion(no_conclusions, "dream")
    assert absent["degraded"] is True
    assert isinstance(absent["degrade_reason"], str)
    assert absent["degrade_reason"] != ""

    artifact = build_blocked_artifact("dream", degraded, "results/optimization/probe.json")
    assert artifact["schema"] == "ar023_cross_family_bench"
    assert artifact["schema_version"] == 1
    assert artifact["family"] == "dream"
    assert artifact["status"] == "blocked"
    assert artifact["modes"] is None
    assert artifact["quality"] is None
    assert artifact["blocked_reason"] == "smoke failed"
    assert artifact["meta"]["blocked_reason"] == "smoke failed"
    assert artifact["meta"]["probe_artifact_ref"] == "results/optimization/probe.json"
    assert artifact["meta"]["dtype_choice"] == "bf16"
    assert artifact["meta"]["graph_enabled"] is False
    assert artifact["reproduction"] == {
        "ok": False,
        "reproduction_anomaly": False,
        "skipped": True,
        "note": "degraded family per probe conclusion",
    }
    assert artifact["verdicts"] == {}


# ---------------------------------------------------------------------------
# (8) UT-2333 CLI parse_args
# ---------------------------------------------------------------------------


def test_ut_2333_parse_args_cli_contract() -> None:
    """--family is required with exactly the two choices; missing or invalid
    values exit code 2; --quick/--reps/--out and the _hf_env flags parse."""
    from scripts.ar023_cross_family_bench import parse_args

    basic = parse_args(["--family", "llada"])
    assert basic.family == "llada"
    assert basic.quick is False
    assert basic.reps == 30
    assert basic.probe_artifact == _PROBE_ARTIFACT_DEFAULT
    assert basic.out is None
    assert basic.models_root is None

    quick = parse_args(["--family", "dream", "--quick"])
    assert quick.family == "dream"
    assert quick.quick is True
    assert quick.reps == 30, "--quick must not change the --reps default in parse_args"

    reps = parse_args(["--family", "llada", "--reps", "7"])
    assert reps.reps == 7

    out = parse_args(["--family", "llada", "--out", "tmp/bench.json"])
    assert out.out == "tmp/bench.json"
    assert isinstance(out.out, str)

    hf = parse_args(["--family", "llada", "--hf-endpoint", "https://hf-mirror.example"])
    assert hf.hf_endpoint == "https://hf-mirror.example"

    hf_home = parse_args(["--family", "llada", "--hf-home", "D:/hf/cache"])
    assert hf_home.hf_home == "D:/hf/cache"

    with pytest.raises(SystemExit) as excinfo:
        parse_args([])
    assert excinfo.value.code == 2

    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--family", "bogus"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# (9) UT-2334 output paths + build_artifact schema
# ---------------------------------------------------------------------------


def test_ut_2334_output_paths_and_build_artifact_schema() -> None:
    """quick_output_path inserts _quick before .json and never collides with
    the formal path; resolve_output_path applies the override > quick > formal
    precedence; build_artifact assembles the exact top-level schema."""
    from scripts.ar023_cross_family_bench import (
        DEFAULT_OUTPUT_PATHS,
        build_artifact,
        quick_output_path,
        resolve_output_path,
    )

    assert quick_output_path("llada") == Path("results/optimization/ar023_llada_bench_quick.json")
    assert quick_output_path("dream") == Path("results/optimization/ar023_dream_bench_quick.json")
    for family in ("llada", "dream"):
        assert quick_output_path(family) != DEFAULT_OUTPUT_PATHS[family]

    override = Path("tmp/override.json")
    assert resolve_output_path("llada", True, override) == override
    assert resolve_output_path("llada", False, override) == override
    assert resolve_output_path("llada", True) == quick_output_path("llada")
    assert resolve_output_path("dream", False) == DEFAULT_OUTPUT_PATHS["dream"]

    with pytest.raises(KeyError):
        resolve_output_path("bogus", False)

    modes = {"baseline": {"wall_ms": {"p50": 1.0}}}
    quality = {"per_prompt": []}
    verdicts = {"h1": {"verdict": "supported"}}
    reproduction = {"ok": True}
    meta = {"seed": 2026}

    artifact = build_artifact("dream", "ok", modes, quality, verdicts, reproduction, meta)

    assert set(artifact) == {
        "schema",
        "schema_version",
        "family",
        "status",
        "modes",
        "quality",
        "verdicts",
        "reproduction",
        "meta",
    }
    assert artifact["schema"] == "ar023_cross_family_bench"
    assert artifact["schema_version"] == 1
    assert artifact["family"] == "dream"
    assert artifact["status"] == "ok"
    assert artifact["modes"] == modes
    assert artifact["quality"] == quality
    assert artifact["verdicts"] == verdicts
    assert artifact["reproduction"] == reproduction
    assert artifact["meta"] == meta
