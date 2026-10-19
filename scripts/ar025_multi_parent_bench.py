"""AR025/T005: synthetic multi-parent batch benchmark (design DD8, frozen verdicts).

Three-way deterministic face per geometry point, per construction
(non-split and split FFN layers):

* ``A`` — merged multi-parent batch: ``multi_parent_batch`` switch ON, ONE
  ``[sum(N_c), T]`` folded call with per-row ``parent_branch_ids`` and the
  ``put_batch`` eager publish (AR025 DD5/DD7).
* ``B`` — per-chain single-parent batches: switch OFF, one ``[N, T]``
  ``branch_ids`` call per chain against its own parent (the AR015 face,
  per-row ``put`` publish — the pre-AR025 state).
* ``C`` — serial per-candidate: switch OFF, ``[1, T]`` calls (the serial
  folding baseline; report-only context for the ledger).

Frozen verdicts (declared here, BEFORE any data; srs FR7 / design rev 3
DD8 — one-shot per point, no re-running until pass):

* **S1** — per point, A vs B logits parity: NON-split construction
  ``torch.equal`` (bitwise, the #41 CPU discipline); split construction
  ``torch.allclose(rtol=atol=1e-6)`` (the #41 split tiering — the split
  risk face is covered HERE because the real-face benchmark pins
  ``split_layers=False``). Any S1 FAIL blocks ALL synthetic verdicts.
* **S2** — launch-ledger proxy (``write_op_count``, #42 semantics):
  ``ops(A) < ops(B)`` STRICTLY, at every point and construction (equal
  means zero merged-batch benefit — blocked by the honest standard).
* **S3** — ledger reduction ratio ``1 - ops(A)/ops(B)`` per point:
  REPORTED, no hard gate (honest forecast: the gain is moderate).

This synthetic face carries NO timing statistics (deterministic
single-shot per point; the reps=30 wall-clock statistics are the real
face, ``--mode real``, delivered with AR025 T006 after the AR023 T006
artifacts). Default device is CPU — the #41 bitwise wording is
CPU-limited; a ``--device cuda`` run puts the non-split tier under the
known M-shape numerical boundary (``[sum(N),T]`` vs ``[N,T]`` GEMM
blocking) and an honest ``blocked`` is the correct outcome there
(direction strict, never lenient). The cache budget is sized to the
grid's worst-case layer population (see ``_CACHE_BUDGET_TOKENS``): an
undersized budget lets the A side's burst ``put_batch`` evict the
in-flight parents — a capacity confound, not a mechanism defect (DD8
erratum).

Run::

    python -m scripts.ar025_multi_parent_bench [--mode synth] [--device cpu]
        [--quick] [--out PATH]
    python -m scripts.ar025_multi_parent_bench --mode real --family llada|dream
        [--reps 30] [--quick] [--models-root PATH] [--probe-artifact PATH]

``--quick`` writes a SEPARATE ``_quick`` artifact, never the formal one
(the AR023 precedent). The real face requires the AR019 venv and the AR023
probe artifacts (``scripts/ar023_family_probe.py`` first; a degraded family
is blocked with zero fabricated numbers); the formal run is the reps=30
pre-committed single shot on a quiet high-compute host (see
``docs/RERUN_CHECKLIST.md`` — the WDDM workstation carries known
environment-conditional timing behavior, AGENTS #46).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward

DEFAULT_OUTPUT_PATH = Path("results/optimization/ar025_multi_parent_bench.json")
QUICK_OUTPUT_PATH = Path("results/optimization/ar025_multi_parent_bench_quick.json")

_NUM_LAYERS = 4
_HIDDEN = 64
_VOCAB = 128
_TAU = 0.95
_SPLIT_MIN_TOKENS = 512
_SEED = 2026
_ATOL = 1e-6
# Cache budget sized to the grid's worst-case layer population: the merged
# A side writes (chains + chains*N) groups per layer, peaking at
# (4 + 32) * 1024 = 36,864 token rows at layer 0 — a smaller budget lets the
# burst put_batch evict the LRU-oldest entries (the in-flight parents) and
# silently disables folding at the deeper layers: a capacity confound, not
# a mechanism defect (design DD8 erratum; evidence
# ar025_multi_parent_bench_eviction_blocked.json). 65536 covers the worst
# point with headroom.
_CACHE_BUDGET_TOKENS = 65536

# (chains, candidates_per_chain, seq_len): the design DD8 grid
# (chains {2,4}) x (N {4,8}) x (T {256,1024}), all 8 combinations.
_FULL_GRID: tuple[tuple[int, int, int], ...] = (
    (2, 4, 256),
    (2, 4, 1024),
    (2, 8, 256),
    (2, 8, 1024),
    (4, 4, 256),
    (4, 4, 1024),
    (4, 8, 256),
    (4, 8, 1024),
)
_QUICK_GRID: tuple[tuple[int, int, int], ...] = ((2, 4, 256),)

# Both constructions run at every point: non-split pins the S1 bitwise
# tier; split covers the 1-ULP gathered-GEMM risk face (the real bench
# pins split_layers=False, design G12).
_CONSTRUCTIONS: tuple[tuple[str, bool], ...] = (
    ("nonsplit", False),
    ("split", True),
)

# ---------------------------------------------------------------------------
# Real-face constants (DD8, delivered with AR025 T006): the AR023 aligned-mode
# anchor pinned explicitly (N=4 / top_k=10 / greedy anchoring on; LLaDA
# sub-block = the WHOLE block and EOS-aware OFF — the AR023 semantics against
# the AR024 defaults), the AR023 seed discipline, and the real-face cache
# budget. The budget is sized against the joint population peak
# (chains + chains*N) * T = 10 * T token rows per layer, where T is the CANVAS
# length (prompt_len + num_tokens), NOT num_tokens alone — e.g. the frozen
# quick/formal geometry at a 128-token prompt gives 10*256 = 2560 rows, so the
# 4096-row budget carries ~1.6x headroom and the DD8-erratum eviction
# predicate (population > budget) would need prompt_len > 281 at
# num_tokens=128 to trip. The erratum lesson: a burst put_batch under an
# undersized budget evicts the in-flight parents and silently disables
# folding (a capacity confound); at the 8B checkpoints (H=4096, bf16
# activations) 4096 rows cost ~32 MB per layer (~1 GB over 32 layers).
# ---------------------------------------------------------------------------
_REAL_SEED = 2026
_REAL_REPS = 30
_REAL_QUICK_REPS = 3
_REAL_NUM_TOKENS = 128
_REAL_QUICK_NUM_TOKENS = 64
_REAL_NUM_CANDIDATES = 4
_REAL_CANDIDATE_TOP_K = 10
_REAL_SPECULATIVE_STEPS = 8
_REAL_TAU = 0.95
_REAL_CACHE_BUDGET_TOKENS = 4096
_REAL_FAMILIES = ("llada", "dream")
_REAL_DEFAULT_OUTPUT = "results/optimization/ar025_real_bench_{family}.json"
_PROBE_ARTIFACT_DEFAULT = "results/optimization/ar023_family_probe.json"
_AR023_REAL_CROSS_REF = {
    "llada": "results/optimization/ar023_llada_bench.json",
    "dream": "results/optimization/ar023_dream_bench.json",
}


# ---------------------------------------------------------------------------
# Synthetic causal decoder (the AR025 test fixture family): the cumsum scan
# is prefix-stable, so a child sharing the parent's prefix reproduces the
# parent's prefix hidden states bit-for-bit — the mathematical basis for
# folding parity; the post_attention_layernorm -> mlp attribute chain is in
# detect_split_spec's candidate table, so split_layers=True engages the
# split FFN path. The embedding/layers/head layout is auto-discovered.
# ---------------------------------------------------------------------------
class _CausalBatchLayer(nn.Module):
    """Causal synthetic layer with a split-detectable token-wise FFN chain."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        cum = torch.cumsum(hidden_states, dim=1)
        denom = torch.arange(1, hidden_states.shape[1] + 1, device=hidden_states.device).view(
            1, -1, 1
        )
        h = hidden_states + 0.1 * cum / denom
        return cast(torch.Tensor, h + self.mlp(self.post_attention_layernorm(h)))


class _CausalBatchModel(nn.Module):
    """Causal synthetic decoder with a detectable embedding/layers/head layout."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(_CausalBatchLayer(hidden_dim) for _ in range(num_layers))
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return cast(torch.Tensor, self.head(x))


def _point_seed(chains: int, n: int, t: int, split: bool) -> int:
    """Frozen arithmetic seed per (point, construction) — never Python ``hash``."""
    return _SEED + 1000 * chains + 10 * n + (t // 256) + (5 if split else 0)


def _build_setup(
    device: str,
    split_layers: bool,
    seed: int,
    multi_parent_batch: bool,
) -> tuple[_CausalBatchModel, ManualFoldedForward, ActivationCache]:
    """Build a fresh causal model plus its manual folded stack on ``device``.

    Setups built with the same ``seed`` share identical weights but carry
    fully independent caches/gates/wrappers, so the A/B/C sides can never
    observe each other's cached activations.
    """
    torch.manual_seed(seed)
    raw = _CausalBatchModel(_VOCAB, _HIDDEN, _NUM_LAYERS).to(device)
    raw.eval()
    cache = ActivationCache(max_entries_per_layer=_CACHE_BUDGET_TOKENS, device=device)
    gate = SimilarityGate(tau=_TAU)
    folded = ManualFoldedForward(
        raw,
        cache=cache,
        gate=gate,
        split_layers=split_layers,
        split_min_tokens=_SPLIT_MIN_TOKENS,
        multi_parent_batch=multi_parent_batch,
    )
    return raw, folded, cache


def _make_world(
    device: str,
    chains: int,
    n: int,
    t: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
    """Build ``chains`` parents ``[1, T]`` and ``n`` candidates each ``[1, T]``.

    Each candidate clones its chain's parent and replaces ONE mid-sequence
    token with a per-candidate distinct value: the prefix stays bit-identical
    to the parent (stable), the cumsum scan propagates the mid divergence to
    every later position (divergent) — a deterministic mixed mask with no RNG
    in the benchmark loop.
    """
    torch.manual_seed(seed)
    parents = [torch.randint(1, _VOCAB, (1, t), device=device) for _ in range(chains)]
    mid = t // 2
    groups: list[list[torch.Tensor]] = []
    for parent in parents:
        base_token = int(parent[0, mid].item())
        group: list[torch.Tensor] = []
        for i in range(n):
            child = parent.clone()
            child[0, mid] = (base_token + 1 + i) % _VOCAB
            group.append(child)
        groups.append(group)
    return parents, groups


def _run_point(
    device: str,
    chains: int,
    n: int,
    t: int,
    split_layers: bool,
) -> dict[str, Any]:
    """Run the A/B/C three-way face at one geometry point + construction."""
    seed = _point_seed(chains, n, t, split_layers)
    parents, groups = _make_world(device, chains, n, t, seed + 1)

    _, folded_a, cache_a = _build_setup(device, split_layers, seed, True)
    _, folded_b, cache_b = _build_setup(device, split_layers, seed, False)
    _, folded_c, cache_c = _build_setup(device, split_layers, seed, False)

    # Populate each side's parents (serial no-parent forwards; identical
    # weights + tokens make the entries value-identical across sides).
    for c in range(chains):
        folded_a(parents[c], branch_id=f"pa{c}")
        folded_b(parents[c], branch_id=f"pa{c}")
        folded_c(parents[c], branch_id=f"pa{c}")

    # A: ONE merged multi-parent call [sum(N_c), T].
    stacked = torch.cat([torch.cat(group, dim=0) for group in groups], dim=0)
    a_ids = [f"a{c}-{i}" for c in range(chains) for i in range(n)]
    a_parents = [f"pa{c}" for c in range(chains) for i in range(n)]
    ops_a_before = cache_a.write_op_count
    logits_a = folded_a(
        stacked,
        branch_id=None,
        branch_ids=a_ids,
        parent_branch_ids=a_parents,
        step_idx=1,
    )
    ops_a = cache_a.write_op_count - ops_a_before

    # B: one [N, T] single-parent batched call per chain (switch off).
    ops_b_before = cache_b.write_op_count
    logits_b_parts: list[torch.Tensor] = []
    for c in range(chains):
        b_ids = [f"b{c}-{i}" for i in range(n)]
        logits_b_parts.append(
            folded_b(
                torch.cat(groups[c], dim=0),
                branch_id=None,
                branch_ids=b_ids,
                parent_branch_id=f"pa{c}",
                step_idx=1,
            )
        )
    ops_b = cache_b.write_op_count - ops_b_before
    logits_b = torch.cat(logits_b_parts, dim=0)

    # C: serial per-candidate (report-only ledger context).
    ops_c_before = cache_c.write_op_count
    logits_c_parts: list[torch.Tensor] = []
    for c in range(chains):
        for i in range(n):
            logits_c_parts.append(
                folded_c(
                    groups[c][i],
                    branch_id=f"c{c}-{i}",
                    parent_branch_id=f"pa{c}",
                    step_idx=1,
                )
            )
    ops_c = cache_c.write_op_count - ops_c_before

    # S1: A vs B parity, tiered by construction (#41 discipline).
    if split_layers:
        s1_pass = bool(torch.allclose(logits_a, logits_b, rtol=_ATOL, atol=_ATOL))
        parity_tier = "allclose(1e-6)"
    else:
        s1_pass = bool(torch.equal(logits_a, logits_b))
        parity_tier = "bitwise"

    # Report-only: B vs C serial parity on the non-split construction
    # (the serial split face is inactive at [1, T] shapes below
    # split_min_tokens; at T=1024 both sides split-activate, so the
    # bitwise reading only holds on non-split).
    serial_parity: bool | None = None
    if not split_layers:
        serial_parity = bool(torch.equal(logits_b, torch.cat(logits_c_parts, dim=0)))

    s2_pass = bool(ops_a < ops_b)
    s3_ratio = 1.0 - (ops_a / ops_b) if ops_b > 0 else float("nan")

    return {
        "chains": chains,
        "candidates_per_chain": n,
        "seq_len": t,
        "construction": "split" if split_layers else "nonsplit",
        "parity_tier": parity_tier,
        "s1_parity_pass": s1_pass,
        "s2_ledger_strictly_smaller": s2_pass,
        "ops_a_merged": ops_a,
        "ops_b_per_chain": ops_b,
        "ops_c_serial_report_only": ops_c,
        "s3_reduction_ratio_report_only": None if s3_ratio != s3_ratio else round(s3_ratio, 6),
        "serial_b_vs_c_bitwise_report_only": serial_parity,
    }


def run_synth_bench(device: str, quick: bool) -> dict[str, Any]:
    """Run the frozen synthetic grid and evaluate the S1-S3 verdicts."""
    grid = _QUICK_GRID if quick else _FULL_GRID
    points: list[dict[str, Any]] = []
    for chains, n, t in grid:
        for _name, split_layers in _CONSTRUCTIONS:
            record = _run_point(device, chains, n, t, split_layers)
            points.append(record)
            print(
                f"chains={chains} N={n} T={t} {record['construction']:8s}: "
                f"S1={record['s1_parity_pass']} ({record['parity_tier']}) "
                f"S2={record['s2_ledger_strictly_smaller']} "
                f"ops A/B/C={record['ops_a_merged']}/{record['ops_b_per_chain']}/"
                f"{record['ops_c_serial_report_only']}"
            )

    s1_all = all(p["s1_parity_pass"] for p in points)
    s2_all = all(p["s2_ledger_strictly_smaller"] for p in points)
    blocked = not (s1_all and s2_all)
    blocked_reasons: list[str] = []
    if not s1_all:
        failed = [
            f"(chains={p['chains']},N={p['candidates_per_chain']},T={p['seq_len']},"
            f"{p['construction']})"
            for p in points
            if not p["s1_parity_pass"]
        ]
        blocked_reasons.append(f"S1 parity failed at: {', '.join(failed)}")
    if not s2_all:
        failed = [
            f"(chains={p['chains']},N={p['candidates_per_chain']},T={p['seq_len']},"
            f"{p['construction']})"
            for p in points
            if not p["s2_ledger_strictly_smaller"]
        ]
        blocked_reasons.append(f"S2 strict ledger failed at: {', '.join(failed)}")

    verdicts = {
        "S1_parity": "blocked" if blocked else "supported",
        "S2_ledger_strictly_smaller": "blocked" if blocked else "supported",
        "S3_reduction_ratio": {
            "per_point": [p["s3_reduction_ratio_report_only"] for p in points],
            "note": "report-only, no hard gate (honest forecast: moderate gain)",
        },
        "overall": "blocked" if blocked else "supported",
        "blocked_reasons": blocked_reasons,
        "aggregation_rule": (
            "any S1 FAIL or S2 non-strict result blocks ALL synthetic "
            "verdicts (frozen, design DD8; equal ledger means zero merged-"
            "batch benefit — the honest standard)"
        ),
    }

    return {
        "meta": {
            "run_date_utc": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "torch_version": torch.__version__,
            "seed": _SEED,
            "seed_formula": (
                "SEED + 1000*chains + 10*N + T//256 + 5*split (frozen "
                "arithmetic, never Python hash)"
            ),
            "model": (
                f"synthetic causal cumsum decoder L={_NUM_LAYERS} "
                f"H={_HIDDEN} V={_VOCAB} (prefix-stable scan + detectable "
                "post_attention_layernorm->mlp chain)"
            ),
            "gate": f"SimilarityGate cosine tau={_TAU}",
            "grid": [list(p) for p in grid],
            "constructions": ["nonsplit", "split"],
            "switch_states": {"A_merged": "on", "B_per_chain": "off", "C_serial": "off"},
            "timing": None,
            "timing_note": (
                "deterministic single-shot per point (parity + ledger face "
                "only); the reps=30 wall-clock statistics are the real face "
                "(--mode real, AR025 T006)"
            ),
            "device_note": (
                "default CPU (the #41 bitwise wording is CPU-limited); a "
                "CUDA run puts the non-split tier under the known M-shape "
                "numerical boundary ([sum(N),T] vs [N,T] GEMM blocking) — "
                "an honest blocked is the correct outcome there"
            ),
            "cupti_note": (
                "launch-ledger proxy is write_op_count (#42 semantics); "
                "kernel counts are NOT claimed (this face carries no "
                "kernel-count measurement, #39)"
            ),
            "cache_budget_tokens_per_layer": _CACHE_BUDGET_TOKENS,
            "cache_budget_note": (
                "sized to the grid's worst-case layer population "
                "((chains + chains*N)*T = 36864 rows at the (4,8,1024) "
                "point) so no in-flight eviction occurs — the DD8 erratum: "
                "the original 8192 budget let the A side's burst put_batch "
                "evict the in-flight parents (LRU-oldest at store time) and "
                "silently disable folding at deeper layers, an asymmetric "
                "capacity confound vs the per-chain interleave (evidence: "
                "ar025_multi_parent_bench_eviction_blocked.json)"
            ),
            "command": "python -m scripts.ar025_multi_parent_bench --mode synth"
            + (" --quick" if quick else "")
            + (f" --device {device}" if device != "cpu" else ""),
        },
        "verdicts_declared": {
            "S1": (
                "per point, A (merged multi-parent [sum(N),T], switch on) vs "
                "B (per-chain [N,T], switch off) logits parity: non-split "
                "torch.equal (bitwise); split torch.allclose(rtol=atol=1e-6); "
                "any FAIL blocks ALL synthetic verdicts"
            ),
            "S2": (
                "write_op_count delta(A) < delta(B) STRICTLY at every point "
                "and construction (equal means zero merged-batch benefit — "
                "blocked by the honest standard)"
            ),
            "S3": ("ledger reduction ratio 1 - ops(A)/ops(B) per point: " "reported, no hard gate"),
        },
        "synthetic_verdicts": verdicts,
        "synthetic_points": points,
        "real": {
            "status": "pending_t006",
            "note": (
                "the real-face section (LLaDA/Dream two-chain speculative "
                "A/B, reps=30 pre-committed single shot, split_layers=False, "
                "H1/H2/delivery-line/H4 frozen verdicts) is delivered with "
                "AR025 T006 after the AR023 T006 artifacts; no numbers are "
                "fabricated here"
            ),
        },
    }


def write_results(data: dict[str, Any], path: str | Path) -> Path:
    """Write the benchmark record as JSON, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Real face (DD8, AR025 T006): two-chain lockstep A/B on the real checkpoints.
# The verdicts below are FROZEN here, BEFORE any data (srs FR7 / design DD8):
#
# * H1 (quality hard gate, split_layers=False construction): the per-chain
#   token sequences of A (joint, switch ON) and B (per-chain, switch pinned
#   OFF) must be IDENTICAL (token equality rate == 1.0; construction
#   rationale: non-split merged-vs-split batches are bit-equal and both
#   sides share one construction at temperature 0). ANY divergence blocks
#   ALL verdicts, with an ``h1_blocked_attribution`` face (first divergent
#   position per chain + an untimed logits allclose replay distinguishing
#   the known CUDA M-shape numerical boundary — a near-tie argmax flip —
#   from a mechanism error).
# * H2 (hypothesis gate): p50(B) / p50(A) > 1.0 (speedup above 1 passes).
# * Delivery line (independent of H2): relative speedup
#   p50(B)/p50(A) - 1 >= 0.10. H2 PASS below 10% triggers the srs FR7
#   frozen degraded delivery ("mechanism complete + synthetic face + real
#   face blocked note", zero fabricated numbers).
# * H4 (ledger, declared STRICT): write_op_count delta(A) <
#   write_op_count delta(B) (equality means zero merged-batch benefit —
#   blocked by the honest standard, the synthetic S2 discipline).
# * Canary: the real face runs eager only (graph off per the AR023 probe
#   conclusions), so the non-structural eager-fallback canary is recorded
#   as null with the eager-neutral note (the AR023 H4 precedent).
#
# Timing follows #39 (perf_counter + cuda synchronize, warmup discarded,
# per-rep A/B round-robin). The A wall-clock is ONE ``sample_joint`` call
# over both chains; the B wall-clock is the two sequential ``sample()``
# calls over the same prompt pair — the user-visible total. The
# reproduction self-check is a strict A-side double-run (seed discipline).
# The prompt pair is the AR023 probe set's first two texts, truncated to
# the common token length (the C2 equal-length joint constraint; a
# symmetric operation on both sides, recorded in the meta).
# ---------------------------------------------------------------------------


def _first_divergence(prompt_index: int, row_a: list[int], row_b: list[int]) -> dict[str, Any]:
    """First divergent position of two token rows (None when identical)."""
    for pos, (token_a, token_b) in enumerate(zip(row_a, row_b)):
        if token_a != token_b:
            return {
                "prompt_index": prompt_index,
                "position": pos,
                "token_a": token_a,
                "token_b": token_b,
            }
    if len(row_a) != len(row_b):
        return {
            "prompt_index": prompt_index,
            "position": min(len(row_a), len(row_b)),
            "token_a": None,
            "token_b": None,
            "note": "rows share a prefix but differ in length",
        }
    return {"prompt_index": prompt_index, "position": None, "token_a": None, "token_b": None}


def _h1_attribution_replay(
    llm: Any,
    family: str,
    row_a: list[int],
    row_b: list[int],
    divergence: dict[str, Any],
) -> dict[str, Any]:
    """Untimed logits replay at the first divergent position (attribution).

    One raw forward per side over the final committed canvas; the logits
    row at the divergent position is compared with ``allclose`` — a close
    pair with different argmaxes is the known CUDA M-shape near-tie
    boundary ([sum(N), T] vs [N, T] GEMM blocking outside the #41 CPU
    wording), a distant pair indicates a mechanism error. Numbers are
    recorded verbatim, never interpreted into a pass.
    """
    import torch

    from actfold.models.sampling_utils import right_shift_logits

    position = divergence.get("position")
    if position is None:
        return {"note": "no divergence to attribute"}
    device = next(llm.model.parameters()).device
    max_abs_diff = None
    close = None
    with torch.no_grad():
        per_side = []
        for row in (row_a, row_b):
            ids = torch.as_tensor(row, dtype=torch.long, device=device)[None, :]
            logits = llm.forward(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))
            if family == "dream":
                logits = right_shift_logits(logits)
            per_side.append(logits[0, position].float())
        max_abs_diff = float((per_side[0] - per_side[1]).abs().max().item())
        close = bool(torch.allclose(per_side[0], per_side[1], rtol=1e-2, atol=1e-2))
    return {
        "position": position,
        "logits_max_abs_diff": round(max_abs_diff, 6),
        "logits_allclose_1e-2": close,
        "attribution_note": (
            "close logits with different argmaxes = the known CUDA M-shape "
            "near-tie boundary (mechanism numerical boundary, not a "
            "correctness failure); distant logits = mechanism error"
        ),
    }


def _make_real_mff(raw_model: Any, multi_parent_batch: bool) -> Any:
    """One instrumented MFF for the real face (eager only, split pinned off).

    ``multi_parent_batch`` is the DD6 single behavior switch: True arms the
    joint merged path (per-row parents + the eager put_batch publish); the
    B side is PINNED OFF (the pre-AR025 per-chain state — the A/B net
    difference is the merged mode including the publish face).
    """
    from actfold.core import ActivationCache, SimilarityGate
    from actfold.models.architecture_utils import ManualFoldedForward

    class _RealBenchMFF(ManualFoldedForward):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.serial_calls = 0
            self.batched_calls = 0

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("branch_ids") is not None:
                self.batched_calls += 1
            else:
                self.serial_calls += 1
            return super().forward(*args, **kwargs)

    cache = ActivationCache(max_entries_per_layer=_REAL_CACHE_BUDGET_TOKENS, device="cuda")
    gate = SimilarityGate(tau=_REAL_TAU, metric="cosine")
    return _RealBenchMFF(
        raw_model,
        cache=cache,
        gate=gate,
        scheduler=None,
        split_layers=False,
        multi_parent_batch=multi_parent_batch,
    )


def _make_real_sampler(family: str, llm: Any, num_tokens: int) -> Any:
    """Family sampler in the AR023 aligned-mode configuration.

    The AR023 anchor is pinned EXPLICITLY against the AR024 defaults: the
    LLaDA speculative unit stays the WHOLE block
    (``speculative_sub_block_size=block_size`` — the ``s == block_size``
    degenerate is bit-identical to AR023, UT-2402) and candidate rows carry
    no EOS-aware continuation (``candidate_eos_aware=False`` — the AR023
    build semantics). Without these pins the AR024 defaults (window 8 /
    EOS-aware True) would silently shift the speculative unit against the
    DD8 "AR023 aligned-mode" wording and the AR023 cross-ref comparability.
    """
    from actfold.models.dream_sampler import DreamSampler, DreamSamplerConfig
    from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig
    from scripts.ar023_cross_family_bench import FAMILY_RECIPES

    recipe = FAMILY_RECIPES[family]
    if family == "llada":
        block_size = int(recipe["block_size"])
        config: Any = LLaDASamplerConfig(
            num_steps=int(recipe["num_steps"]),
            num_tokens=num_tokens,
            block_size=block_size,
            remasking=str(recipe["remasking"]),
            seed=_REAL_SEED,
            num_block_candidates=_REAL_NUM_CANDIDATES,
            candidate_top_k=_REAL_CANDIDATE_TOP_K,
            candidate_include_greedy=True,
            speculative_sub_block_size=block_size,
            candidate_eos_aware=False,
        )
        return LLaDASampler(llm, config=config)
    config = DreamSamplerConfig(
        num_steps=num_tokens,
        num_tokens=num_tokens,
        seed=_REAL_SEED,
        speculative_steps=_REAL_SPECULATIVE_STEPS,
        num_block_candidates=_REAL_NUM_CANDIDATES,
        candidate_top_k=_REAL_CANDIDATE_TOP_K,
        candidate_include_greedy=True,
    )
    return DreamSampler(llm, config=config)


def _real_blocked_artifact(
    family: str,
    reason: str,
    probe_ref: str,
    quick: bool,
    reps: int,
) -> dict[str, Any]:
    """Blocked real-face artifact: the reason verbatim, ZERO numbers."""
    return {
        "schema": "ar025_real_bench",
        "family": family,
        "status": "blocked",
        "blocked_reason": reason,
        "probe_artifact_ref": probe_ref,
        "quick": quick,
        "reps": reps,
        "verdicts_declared": "see scripts/ar025_multi_parent_bench.py docstring (frozen, DD8)",
        "meta": {
            "note": "no numbers are fabricated on a blocked face (srs FR5)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


def _load_real_model(
    family: str,
    models_root: Path,
    dtype_choice: str,
) -> tuple[Any, str, dict[str, Any]]:
    """Load the family checkpoint at the requested dtype.

    ``auto`` (the formal default) mirrors the AR023 loader verbatim (bf16
    first, int8 fallback); an explicit ``bf16``/``int8`` pins the dtype —
    the smoke-only escape hatch for hosts whose VRAM cannot hold the bf16
    weights (the formal run keeps one dtype discipline end to end).
    """
    import torch

    from scripts.ar023_cross_family_bench import _load_family_model
    from scripts.ar023_family_probe import _family_wrapper_cls, sha256_file

    if dtype_choice == "auto":
        return _load_family_model(family, models_root)
    local_dir = models_root / family
    if not local_dir.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory not found: {local_dir} (run the probe first; "
            "no synthetic-model fallback for a real-checkpoint benchmark)"
        )
    wrapper_cls = _family_wrapper_cls(family)
    if dtype_choice == "bf16":
        model = wrapper_cls(str(local_dir), torch_dtype=torch.bfloat16)
        model.model.cuda()
    else:
        model = wrapper_cls(str(local_dir), load_in_8bit=True, device_map={"": 0})
    model.model.eval()
    weights = local_dir / "model.safetensors"
    return (
        model,
        dtype_choice,
        {
            "checkpoint_dir": str(local_dir),
            "checkpoint_sha256": sha256_file(weights) if weights.exists() else None,
            "dtype_achieved": dtype_choice,
            "load_attempts": [],
        },
    )


def run_real_bench(
    family: str,
    models_root: Path,
    reps: int,
    quick: bool,
    probe_ref: str,
    dtype_choice: str = "auto",
    num_tokens_override: int | None = None,
) -> dict[str, Any]:
    """Run the two-chain lockstep A/B on the real checkpoint (CUDA required)."""
    import torch

    from scripts.algo_experiments import stats_from_samples
    from scripts.ar023_cross_family_bench import reproduction_verdict
    from scripts.ar023_family_probe import PROBE_PROMPTS

    if not torch.cuda.is_available():
        raise RuntimeError("run_real_bench requires a CUDA device")
    num_tokens = (
        num_tokens_override
        if num_tokens_override is not None
        else (_REAL_QUICK_NUM_TOKENS if quick else _REAL_NUM_TOKENS)
    )
    llm, dtype, load_meta = _load_real_model(family, models_root, dtype_choice)
    device = "cuda"

    if len(PROBE_PROMPTS) < 2:
        raise RuntimeError(
            "the real face requires two probe prompts (one per chain); the "
            "AR023 probe set has fewer"
        )
    prompts: list[dict[str, Any]] = []
    for text in PROBE_PROMPTS[:2]:
        ids = llm.tokenizer(text, return_tensors="pt").input_ids.to(device)
        prompts.append({"text": text, "ids": ids, "prompt_len": int(ids.shape[1])})
    # C2 equal-length construction: truncate to the common token length (a
    # symmetric operation on both sides — A and B consume the SAME pair).
    common_len = min(int(p["ids"].shape[1]) for p in prompts)
    truncated = any(int(p["ids"].shape[1]) != common_len for p in prompts)
    if truncated:
        for prompt in prompts:
            prompt["ids"] = prompt["ids"][:, :common_len]
            prompt["prompt_len"] = common_len

    # Independent stacks per side (the AR023 per-mode isolation pattern):
    # A = switch ON + sample_joint; B = switch PINNED OFF + per-chain
    # sample() — the only difference is the merged mode (publish included).
    mff_a = _make_real_mff(llm.model, multi_parent_batch=True)
    mff_b = _make_real_mff(llm.model, multi_parent_batch=False)
    sampler_a = _make_real_sampler(family, llm, num_tokens)
    sampler_b = _make_real_sampler(family, llm, num_tokens)
    prompt_ids = [prompt["ids"] for prompt in prompts]

    def _run_a() -> list[list[int]]:
        outs = sampler_a.sample_joint(prompt_ids, mff_a)
        return [out.sequences[0].tolist() for out in outs]

    def _run_b() -> list[list[int]]:
        rows: list[list[int]] = []
        for ids in prompt_ids:
            out = sampler_b.sample(ids, folded_model=mff_b)
            rows.append(out.sequences[0].tolist())
        return rows

    print(
        f"[ar025-real] family={family} dtype={dtype} num_tokens={num_tokens} "
        f"reps={reps} prompt_pair_truncated={truncated}"
    )

    # -- reproduction self-check: strict A-side double-run (untimed)
    repro_rows_a = _run_a()
    repro_rows_b = _run_a()
    reproduction = reproduction_verdict(repro_rows_a, repro_rows_b)
    print(f"[ar025-real] reproduction double-run ok={reproduction['ok']}")

    # -- warmup: one discarded run per side
    _run_a()
    _run_b()
    print("[ar025-real] warmup complete")

    # -- timed reps: per-rep A/B round-robin (rep r starts at r % 2)
    samples: dict[str, list[float]] = {"A": [], "B": []}
    first_rows: dict[str, list[list[int]]] = {}
    cross_rep_deterministic = True
    for rep in range(reps):
        order = ("A", "B") if rep % 2 == 0 else ("B", "A")
        for side in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            rows = _run_a() if side == "A" else _run_b()
            torch.cuda.synchronize()
            samples[side].append((time.perf_counter() - start) * 1000.0)
            if side not in first_rows:
                first_rows[side] = rows
            elif rows != first_rows[side]:
                cross_rep_deterministic = False
        print(f"[ar025-real] rep {rep + 1}/{reps} done", flush=True)

    # -- H1 (frozen): per-chain token equality A vs B, rate == 1.0
    divergences = [
        _first_divergence(chain, first_rows["A"][chain], first_rows["B"][chain])
        for chain in range(len(prompts))
    ]
    h1_pass = all(d["position"] is None for d in divergences)
    h1_attribution: dict[str, Any] = {"note": "H1 passed — no attribution needed"}
    if not h1_pass:
        h1_attribution = {
            "per_chain_divergences": divergences,
            "logits_replay": [
                _h1_attribution_replay(
                    llm,
                    family,
                    first_rows["A"][d["prompt_index"]],
                    first_rows["B"][d["prompt_index"]],
                    d,
                )
                for d in divergences
                if d["position"] is not None
            ],
        }

    # -- H2 / delivery line (frozen)
    stats_a = stats_from_samples(samples["A"])
    stats_b = stats_from_samples(samples["B"])
    p50_a = stats_a.p50
    p50_b = stats_b.p50
    p50_ratio = (p50_b / p50_a) if p50_a > 0 else float("nan")
    h2_pass = bool(p50_ratio > 1.0)
    delivery_pass = bool((p50_ratio - 1.0) >= 0.10)

    # -- H4 sidecar (untimed ledger window, strict)
    ops_a0 = mff_a.cache.write_op_count
    _run_a()
    ops_a = mff_a.cache.write_op_count - ops_a0
    ops_b0 = mff_b.cache.write_op_count
    _run_b()
    ops_b = mff_b.cache.write_op_count - ops_b0
    h4_pass = bool(ops_a < ops_b)
    print(
        f"[ar025-real] sidecar ledger: ops_A={ops_a} ops_B={ops_b} " f"(strict smaller: {h4_pass})"
    )

    # -- verdicts (frozen aggregation: repro fail blocks ALL; H1/H4 are hard
    # gates; the delivery line degrades per the srs FR7 trigger)
    blocked_reasons: list[str] = []
    if not h1_pass:
        blocked_reasons.append("H1 token parity failed (see h1_blocked_attribution)")
    if not h4_pass:
        blocked_reasons.append(
            "H4 strict ledger failed (equal or larger write-op delta means "
            "zero merged-batch benefit — the honest standard)"
        )
    if not delivery_pass:
        blocked_reasons.append(
            "delivery line < 10% relative speedup — srs FR7 frozen degraded "
            "delivery: 'mechanism complete + synthetic face + real-face "
            "blocked note', zero fabricated numbers"
        )
    overall_blocked = bool(blocked_reasons) or not reproduction["ok"]
    if not reproduction["ok"]:
        blocked_reasons.insert(
            0,
            "reproduction anomaly: A-side double-run mismatch "
            f"(first diff chain {reproduction.get('mismatch', {}).get('first_diff_prompt_index')})",
        )

    verdicts = {
        "H1_token_parity": {
            "pass": h1_pass,
            "rule": (
                "per-chain token sequences A (joint, switch on) vs B "
                "(per-chain, switch pinned off) identical (rate == 1.0); "
                "any divergence blocks ALL verdicts"
            ),
            "h1_blocked_attribution": h1_attribution,
        },
        "H2_speed": {
            "pass": h2_pass,
            "rule": "p50(B)/p50(A) > 1.0",
            "p50_ratio_b_over_a": round(p50_ratio, 4),
        },
        "delivery_line": {
            "pass": delivery_pass,
            "rule": "p50(B)/p50(A) - 1 >= 0.10 (independent of H2; below the "
            "line triggers the srs FR7 frozen degraded delivery)",
            "relative_speedup": round(p50_ratio - 1.0, 4),
        },
        "H4_ledger_strictly_smaller": {
            "pass": h4_pass,
            "rule": "write_op_count delta(A) < delta(B) STRICTLY",
            "ops_a_merged": ops_a,
            "ops_b_per_chain": ops_b,
        },
        "canary_structural_eager_fallback": {
            "timed_count": None,
            "note": (
                "graph path off per the AR023 probe conclusions (eager "
                "folding by construction) — the non-structural eager-fallback "
                "canary is null with this eager-neutral note (AR023 H4 "
                "precedent)"
            ),
        },
        "overall": "blocked" if overall_blocked else "supported",
        "blocked_reasons": blocked_reasons,
        "reproduction": reproduction,
    }

    import transformers

    meta = {
        "checkpoint_dir": load_meta["checkpoint_dir"],
        "checkpoint_sha256": load_meta["checkpoint_sha256"],
        "dtype": dtype,
        "dtype_choice_requested": dtype_choice,
        "dtype_note": (
            "auto = the AR023 loader discipline (bf16 first, int8 fallback); "
            "an explicit choice is the smoke-only escape hatch — the formal "
            "run keeps one dtype discipline end to end"
        ),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "seed": _REAL_SEED,
        "reps": reps,
        "quick": quick,
        "num_tokens": num_tokens,
        "num_tokens_override": num_tokens_override,
        "num_tokens_note": (
            "None = the frozen quick/formal value (64/128); an override is "
            "the smoke-only escape hatch — the formal run never passes it"
        ),
        "num_candidates": _REAL_NUM_CANDIDATES,
        "candidate_top_k": _REAL_CANDIDATE_TOP_K,
        "speculative_steps": _REAL_SPECULATIVE_STEPS,
        "split_layers": False,
        "split_layers_note": (
            "split layers pinned OFF (design G12): the split gathered GEMM "
            "crosses the micro-kernel boundary with a 1-ULP difference — the "
            "split risk face is covered by the synthetic S1 tier "
            "(allclose 1e-6), keeping this face constructively bit-equal"
        ),
        "switch_states": {"A_merged": "on", "B_per_chain": "off (pinned)"},
        "cache_budget_tokens_per_layer": _REAL_CACHE_BUDGET_TOKENS,
        "cache_budget_note": (
            "sized against the joint population peak "
            f"(chains + chains*N)*T = 10*{common_len + num_tokens} = "
            f"{10 * (common_len + num_tokens)} token rows per layer at the "
            f"actual canvas length T = prompt_len({common_len}) + "
            f"num_tokens({num_tokens}) — "
            f"{_REAL_CACHE_BUDGET_TOKENS / (10 * (common_len + num_tokens)):.2f}x "
            "headroom; the DD8 erratum lesson (a burst put_batch under an "
            "undersized budget evicts the in-flight parents)"
        ),
        "prompt_pair": [prompt["text"] for prompt in prompts],
        "prompt_pair_truncated_to_equal_len": truncated,
        "prompt_truncation_note": (
            "the joint C2 constraint requires equal prompt token lengths; "
            "the pair is truncated to the common length — a symmetric "
            "operation, both sides consume the SAME pair"
        ),
        "cross_rep_deterministic": cross_rep_deterministic,
        "cupti_note": (
            "#39 wall-clock discipline: host perf_counter + cuda "
            "synchronize; no kernel-launch claims (CUPTI not probed)"
        ),
        "probe_artifact_ref": probe_ref,
        "cross_ref": {
            "ar023_same_table_artifact": _AR023_REAL_CROSS_REF[family],
            "note": (
                "the real face positions its result against the AR023 "
                "family baseline on the same checkpoint/prompt/seed "
                "discipline (the T008 README same-table execution; the "
                "batched rows of the referenced artifact are the "
                "pre-AR025 comparison line)"
            ),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "schema": "ar025_real_bench",
        "family": family,
        "status": "blocked" if overall_blocked else "ok",
        "verdicts_declared": (
            "H1/H2/delivery-line/H4/canary frozen in "
            "scripts/ar025_multi_parent_bench.py (design DD8), declared "
            "BEFORE any data"
        ),
        "verdicts": verdicts,
        "wall_ms": {
            "A_joint": {
                "mean": stats_a.mean,
                "std": stats_a.std,
                "p50": stats_a.p50,
                "n": stats_a.n,
            },
            "B_per_chain": {
                "mean": stats_b.mean,
                "std": stats_b.std,
                "p50": stats_b.p50,
                "n": stats_b.n,
            },
            "comparability_note": (
                "A = ONE sample_joint call over both chains; B = the two "
                "sequential sample() calls over the same pair — the "
                "user-visible total on both sides"
            ),
        },
        "first_rep_rows": first_rows,
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("synth", "real"),
        default="synth",
        help="synth = the frozen S1-S3 deterministic face; real = the "
        "two-chain lockstep A/B on the real checkpoints (AR025 T006, "
        "requires --family and the AR023 probe artifacts)",
    )
    parser.add_argument(
        "--family",
        choices=_REAL_FAMILIES,
        default=None,
        help="real mode only: the checkpoint family (llada | dream)",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="real mode only: checkpoint root (default: the probe " "artifact's models_root)",
    )
    parser.add_argument(
        "--probe-artifact",
        type=str,
        default=_PROBE_ARTIFACT_DEFAULT,
        help="real mode only: the AR023 family-probe artifact path "
        "(degraded families are blocked with zero fabricated numbers)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=_REAL_REPS,
        help="real mode only: timed reps (formal = 30, the pre-committed " "single shot)",
    )
    parser.add_argument(
        "--real-num-tokens",
        type=int,
        default=None,
        help="real mode only: generation-length override (default: the "
        "frozen quick/formal values 64/128). The SMOKE-ONLY escape hatch "
        "for a minutes-scale pipeline check on slow hosts — the formal run "
        "never passes it",
    )
    parser.add_argument(
        "--real-dtype",
        choices=("auto", "bf16", "int8"),
        default="auto",
        help="real mode only: checkpoint dtype. auto (the formal default) "
        "mirrors the AR023 loader (bf16 first, int8 fallback); an explicit "
        "choice is the smoke-only escape hatch for hosts whose VRAM cannot "
        "hold the bf16 weights — the formal run keeps one dtype discipline "
        "end to end",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="synth mode only: device (default cpu — the #41 bitwise "
        "discipline; CUDA carries the known M-shape numerical boundary)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke subset writing a SEPARATE _quick artifact (never the " "formal path)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: the mode's formal artifact; "
        "--quick defaults to the separate _quick artifact)",
    )
    args = parser.parse_args()

    if args.mode == "real":
        from scripts._hf_env import apply_hf_env

        apply_hf_env()

        if args.family is None:
            parser.error("--mode real requires --family {llada,dream}")
        family = args.family
        reps = _REAL_QUICK_REPS if args.quick else args.reps
        out_path = args.out
        if out_path is None:
            base = _REAL_DEFAULT_OUTPUT.format(family=family)
            out_path = Path(base.replace(".json", "_quick.json") if args.quick else base)

        probe_path = Path(args.probe_artifact)
        probe_artifact: dict[str, Any] = {}
        if probe_path.exists():
            probe_artifact = json.loads(probe_path.read_text(encoding="utf-8"))
        from scripts.ar023_cross_family_bench import probe_conclusion

        conclusion = probe_conclusion(probe_artifact, family)
        if conclusion["degraded"]:
            artifact = _real_blocked_artifact(
                family,
                f"family degraded per probe: {conclusion.get('degrade_reason')}",
                str(probe_path),
                args.quick,
                reps,
            )
            out = write_results(artifact, out_path)
            print(f"[ar025-real] family '{family}' degraded per probe: benchmark blocked")
            print(f"results written to {out}")
            return

        models_root = (
            args.models_root
            if args.models_root is not None
            else Path(probe_artifact.get("meta", {}).get("models_root", "ar023_models"))
        )
        try:
            artifact = run_real_bench(
                family,
                models_root,
                reps,
                args.quick,
                str(probe_path),
                args.real_dtype,
                args.real_num_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - blocked artifact, never fabricated numbers
            artifact = _real_blocked_artifact(
                family,
                f"real-face run failed: {type(exc).__name__}: {exc}",
                str(probe_path),
                args.quick,
                reps,
            )
        out = write_results(artifact, out_path)
        verdicts = artifact.get("verdicts", {})
        print(f"H1 (token parity): {verdicts.get('H1_token_parity', {}).get('pass')}")
        print(f"H2 (p50 speedup): {verdicts.get('H2_speed', {}).get('pass')}")
        print(
            f"H4 (ledger strictly smaller): {verdicts.get('H4_ledger_strictly_smaller', {}).get('pass')}"
        )
        print(f"Overall (real): {verdicts.get('overall')}")
        print(f"results written to {out}")
        return

    out_path = args.out
    if out_path is None:
        out_path = QUICK_OUTPUT_PATH if args.quick else DEFAULT_OUTPUT_PATH

    data = run_synth_bench(args.device, args.quick)
    out = write_results(data, out_path)
    print(f"\nS1 (A vs B parity): {data['synthetic_verdicts']['S1_parity']}")
    print(
        "S2 (ledger strictly smaller): "
        f"{data['synthetic_verdicts']['S2_ledger_strictly_smaller']}"
    )
    print(f"Overall (synthetic): {data['synthetic_verdicts']['overall']}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
