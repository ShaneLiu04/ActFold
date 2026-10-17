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

``--quick`` writes a SEPARATE ``_quick`` artifact, never the formal one
(the AR023 precedent).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        return h + self.mlp(self.post_attention_layernorm(h))


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
        return self.head(x)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("synth", "real"),
        default="synth",
        help="synth = the frozen S1-S3 deterministic face (this delivery); "
        "real = the reps=30 checkpoint face (AR025 T006)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device for the synthetic face (default cpu — the #41 bitwise "
        "discipline; CUDA carries the known M-shape numerical boundary)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="single-point smoke subset writing a SEPARATE _quick artifact",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: the formal artifact; --quick "
        "defaults to the separate _quick artifact)",
    )
    args = parser.parse_args()

    if args.mode == "real":
        parser.error(
            "--mode real is the AR025 T006 delivery (requires the AR023 T006 "
            "artifacts; reps=30 pre-committed single shot); it is not "
            "implemented in this script version — no numbers are fabricated"
        )

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
