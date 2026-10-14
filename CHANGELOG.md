# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### AR018: Publish-Path Host-Cost Compression (P1-7, Fourth Rung)

Spec: `specs/changes/AR018-publish-cost/` (srs.md / design.md / tasks.md; req gate two rounds — Major-1: the mutation-③ red-test mapping was wrong, split into ③ non-fresh snapshot aliasing -> UT-1705 and ⑤ cross-step sums reuse/overwrite -> UT-1701; Major-2: the revalidate_host share claim contradicted the probe artifact (measured 0.33-1.40%), corrected; design gate two rounds — Major-1: the `phase_sum >= full_step` directional assertion was refuted by the probe's own data (3 of 4 configs have phase_sum < full step — host work between phases outweighs sync serialization), all wording switched to a neutral two-way deviation note with both values recorded in the artifact). Full regression zero-regression (855 passed = 846 + 9 new; existing tests untouched); demo baseline unchanged; production delta is two files (`stability_profiler.py` + `architecture_utils.py`) — `cuda_graph.py` is untouched.

#### Added

- **Precomputed-sum profiler entry (FR1)**: `StabilityProfiler.record_precomputed` — the batched-publish counterpart of `record`: the caller owns the reduction (one `mask_buf.sum(dim=2)` per step serves every (layer, row) pair) and passes the 0-dim device tensor plus the token count explicitly; the stored `_RawLayerStat` is field-for-field identical to what `record` produces (enabled early-return, `_raw`/`_history` append, `_max_history_len` trim mirrored line-for-line), so every downstream consumer (`_ratios`/`get_profile`/history/aggregations) behaves identically. An explicit `ndim == 0` validation rejects a whole [L, N] table mis-passed as one layer's sum (host-side metadata check, zero sync); `divergence_positions` is never produced here — the debug path falls back to `record`. The original `record` method is byte-identical (all four eager/serial call sites untouched).
- **Snapshot publish (FR2)**: `_publish_graph_children` clones the whole `[L, N, T, H]` child buffer ONCE into a fresh snapshot (`cache.put` copies into its own storage, so the per-row clones were pure overhead) and publishes row views; all (layer, row) stability sums come from ONE batched reduction recorded through `record_precomputed` (0-dim views, zero extra launches, zero host readback). Three-state dispatch: profiler off -> zero reductions and zero records; `debug_enabled` -> per-row `record` fallback (the only divergence_positions producer, correctness over launch count); normal -> one batched reduction. Launch ledger 3·L·N -> L·N+2 per publish (96 -> 34 at (256,8,4), 384 -> 130 at (1024,16,8)). The snapshot's freshness is the anti-aliasing argument for by-reference caches (private storage, zero writes after the publish — UT-1705 pins it with a RefCache + in-place child_buf mutation).
- Tests: UT-1701~1708 (`tests/test_publish_cost.py`, 1008 lines; UT-1709 is the mutation phase) — record vs record_precomputed field-for-field equivalence (CPU 240-record block with history-trim engagement + CUDA [L,N,T] 0-dim-view block; aggregation faces all compared; ValueError validation), a publish-window spy (`torch.Tensor.sum/clone/item/tolist/cpu` counting wrappers, window = one publish call: exactly one bool sum, zero host syncs, zero reductions when disabled), bit-exact snapshot-vs-per-row-reference publish, exactly one child_buf-identity clone, by-reference anti-aliasing, UT-1502/1509 completeness/isolation re-assertions, debug fallback vs normal-batch dispatch, and serial-graph + eager-batched zero interference with the `record` signature pinned. Double-sided invariant UTs (1703/1705/1706/1708, green in the Red run by design) are declared in the file docstring with their Red value carried by 1702/1704. Five mutations (batched reduction removed / snapshot deleted back to per-row clones / snapshot aliasing the static buffer / debug fallback removed / record_precomputed storing into a cross-call overwritten scratch) all turned their pinned tests red and were restored byte-identically (SHA256).
- Benchmark: `scripts.ar018_publish_bench` times the AR017 three-path matrix (baseline / fixed-ratio graph / adaptive graph, geometries (256,8) and (1024,16) × N ∈ {4,8} × T=512), round-robin interleaved, warmup=5, reps=30 pre-committed single run; a six-phase sync-bounded decomposition (revalidate_host/prefill/replay/readback/publish/logits_clone, the probe method fixed as a script-level measurement outside the timed window) records `phase_sum_ms` and `full_step_p50_ms` side by side with a two-way deviation note; the AR017 artifact is loaded at runtime as the cross-artifact anchor (availability + anchor run_date + comparison nature declared in meta; a missing anchor degrades H1/H2 to `no_anchor` without blocking); peak-memory observation (`max_memory_allocated` around one untimed adaptive step) quantifies the snapshot transient. Artifact: `results/optimization/ar018_publish_bench.json`.

#### Experiment Results (synthetic LLaMA-layout, Quadro RTX 5000, driver 556.18, p50 ms per verification step, n=30)

- **H1 AR018 adaptive p50 below the AR017 anchor: inconclusive overall (3/4 supported)** — supported at (256,8,8) and both deep geometries; (256,8,4) is direction-favorable (4.708 vs 7.120 ms) but below the 2-sigma bar, the same underpowered small shape as in the AR017 run.
- **H2 speedup no-regression: supported (4/4)** — adaptive-vs-baseline 2.40× / 2.67× / 2.66× / 2.95× against the AR017 anchor 1.55× / 1.68× / 2.30× / 2.46×; the small geometry lifts from 1.55/1.68× to 2.40/2.67× (the srs P1-7 fourth-rung acceptance). The fixed-capacity path benefits too (8.150 -> 5.664 ms at (256,8,4)) — both graph paths share the publish loop.
- **H3 publish-phase share below the probe reference: supported (4/4)** — 45.3% / 50.9% / 15.7% / 17.6% of the full adaptive step against the probe's 60.7% / 71.2% / 29.7% / 31.5% (publish p50 2.132 / 4.225 / 4.144 / 8.545 ms).
- **H4 consistency + profiler: supported (4/4)** — exact argmax agreement across the three paths, adaptive-vs-fixed allclose 1e-4, and every branch's last-step stable ratio equal to an independent per-row mask_buf re-sum (<=1e-6, sampled outside the timed window).

### AR017: Adaptive Divergent-Capacity Re-Capture (P1-7, Decisive Re-Answer)

Spec: `specs/changes/AR017-adaptive-capacity/` (srs.md / design.md / tasks.md; req gate two rounds — Major-1: the growth evaluation's call site is mutually exclusive with the exceeded branch's `return None`, so the design splits into two hooks, growth inside the exceeded branch before the eager fallback and shrink after the publish; Major-2: observation-window ownership — append only when the replay returns non-None, clear after a re-capture, freeze after retirement; Major-3: the retirement flag must be independent and must never set `_graph_batched_capture_failed`, which would discard the healthy old runner along with the adaptive path; design gate two rounds — the float identity `ceil(ratio * n) == target` holds only for power-of-two n (counterexample n=25/target=7), so the caller decides by the capacity the new ratio will actually produce, with a power-of-two fixture constraint pinned in §6; review gate PASS with six non-blocking Minors, all addressed: the test-file line count corrected (1195), the hypothesis-label mapping vs srs FR5 documented, the adaptation table extended with `capacity_trajectory` + `observed_max_layer_divergence` p50/min/max and the CUPTI probe added to meta (both requiring an artifact re-run — see the benchmark bullet for the n=30 re-run rationale), the `AdaptiveCapacityConfig` import moved to module top in `benchmark_runner.py`, and leftover cross-annotations between guide Parts 23/24). Full regression zero-regression (846 passed = 838 + 8 new; the only existing-test edit is UT-1608 gaining one throttle-warm-up step with an unchanged assertion face — a Red-phase test-side inference bug, fixed on the test side because the design is the approved spec); demo baseline unchanged; production delta is five files (new policy module + readback-free staging in `cuda_graph.py` + hooks/lifecycle in `architecture_utils.py` + six config fields + runner wiring).

#### Added

- **Adaptive capacity policy (`actfold/core/adaptive_capacity.py`)**: `AdaptiveCapacityConfig` (frozen; grow_factor >= 1.0, shrink_threshold in (0,1), shrink_patience / min_recapture_interval / max_recaptures >= 1, boundary `ValueError`s), `AdaptiveAction` (noop / recapture / retired), and `AdaptiveCapacityPolicy` — a pure state machine: a budget violation grows immediately to `ceil(grow_factor * maxD)` clamped to N*T; `shrink_patience` consecutive violation-free observations with `maxD <= shrink_threshold * C` shrink to `clamp(ceil(grow_factor * maxD), 1, C-1)`; re-captures are throttled by an increment-then-check step counter (initialized to the interval, so the first observe is never throttled; `note_recaptured` resets it) and capped at `max_recaptures` *successful* re-captures, after which the policy retires (the retiring observe appends first, returns `kind="retired"`, and later observes are permanent no-ops with a frozen window). `events` records only outcomes: one recapture entry per successful `note_recaptured` (trigger / target / from), never a failed or throttled attempt; the manual `retire()` (the re-capture-failure path) records no event.
- **Readback-free divergence staging (FR1)**: `FoldedGraphRunner.last_layer_divergences` — the per-layer divergent counts staged by `validate_budgets`, reusing its single `count_buf.tolist()` (no extra host sync); `None` until the first validation, refreshed on every one, never stale.
- **Hooks and re-capture lifecycle (`architecture_utils.py`)**: `ManualFoldedForward(graph_adaptive_capacity=True, graph_adaptive_config=...)` opts in (config type-checked always; a default config is built when `None`; default-off builds no policy state — the public probe `graph_adaptive_policy` reads `None`, UT-1606 zero-state contract). The growth hook runs inside the budget-exceeded branch (after the one-time warning, before the eager fallback — the violation *is* the growth signal); the shrink hook runs after `_publish_graph_children` (the capacity swap never precedes the publish, which consumes the old runner's static buffers). `_recapture_batched_runner` captures-new-first, swaps-on-success, drops-old-last; a failed re-capture warns exactly once ("adaptive..."), retires only the adaptive path (an independent flag — `_graph_batched_capture_failed` is deliberately untouched), and keeps the old runner in service with zero retries. A prospective guard decides by `ceil(new_ratio * N * T)` — the actual capacity the new ratio produces — so a shrink whose target rounds back to the current capacity is skipped instead of paying for a pointless capture.
- **Config flow**: `ActFoldConfig` gains `graph_adaptive_capacity` plus the five mirrored policy knobs (bounds validated regardless of the switch; the config object is assembled at the consumption site so utils stays decoupled from core); `BenchmarkRunner` wires them into the folded stack.
- Tests: UT-1601~1608 (`tests/test_adaptive_capacity.py`, 1195 lines) — the CPU table-driven policy contract (growth clamping, patience windows, beta blocking, C<=1 no-op, window trimming, interval throttling, max_recaptures retirement, config boundary errors, two-instance determinism), shrink and growth end-to-end (the violated step's semantics unchanged; a runner swap must not reset the budget-warning flag), the recapture cap with exactly one non-budget retirement warning, the failure safety net (injected capture `RuntimeError`: one adaptive warning, the old runner `is`-alive, `_graph_batched_capture_failed` stays False, zero retries, strictly increasing replay-spy counts), default-off zero behavior, the staging seam, and serial-path zero interference across two batched runner swaps. Four mutations (both hooks removed / growth branch removed / retirement check removed / old-first drop ordering) all turned their pinned tests red and were restored byte-identically (SHA256).
- Benchmark: `scripts.ar017_adaptive_bench` times three paths — raw baseline, fixed-capacity graph (ratio 0.5), adaptive graph — over geometries (H=256,L=8) and (H=1024,L=16) × N ∈ {4, 8} at T=512 (N*T stays a power of two, the capacity-exactness precondition), round-robin interleaved per repetition, warmup=5 covering the adaptation schedule, TimingStats with n=30 (the pre-run at n=10 flipped its verdict on the borderline (256,8,4) config between two same-code runs — underpowered evidence; the artifact of record is a single pre-committed n=30 run), per-config adaptation tables (`recapture_events`, `capacity_trajectory`, `observed_max_layer_divergence` p50/min/max — trajectories are collected outside the timed windows, zero timing pollution), and the GPU driver version plus the CUPTI probe in meta (AR016 Minor-2 closure). Hypothesis-label mapping vs srs FR5: the implemented H2 (adaptive-vs-baseline 2σ test) carries the P1-7 claim; the srs H2 (adaptive speedup >= fixed speedup) and H3 (geometry extrapolation) have no independent verdict field and both hold in the recorded data (1.55>1.35, 1.68>1.52, 2.30>1.39, 2.46>1.44; deep-geometry speedups exceed small-geometry ones); the implemented H3 is the srs H4 (consistency); the implemented H4 (capacity tracking) is new. Artifact: `results/optimization/ar017_adaptive_bench.json`.

#### Experiment Results (synthetic LLaMA-layout, Quadro RTX 5000, driver 556.18, p50 ms per verification step, n=30)

- **H1 adaptive vs fixed capacity: inconclusive overall** — supported at deep geometry (H=1024/L=16: 1.66× and 1.71×); direction favorable but below 2σ at H=256/L=8 (1.14×, 1.11×), consistent with the FFN[C] share being only ~25% of the small-geometry step (the same root cause behind AR016's P1-7 inconclusive).
- **H2 adaptive vs raw baseline: overall inconclusive under the strict all-config aggregation; 3/4 configs supported — the decisive-geometry re-answer of P1-7 is unambiguous.** At the srs §1.1 target geometry (H=1024/L=16) adaptive beats the raw baseline 2.30×/2.46× with deltas 40.5/86.4 ms against 2σ thresholds of 4.4/7.4 ms (9.3×/11.6× over — decisively supported). The small geometry is direction-positive everywhere (1.55×, 1.68×; (256,8,8) supported) with (256,8,4) below the spread-based 2σ bar — its delta is stable at 3.7–3.9 ms across three independent runs, so the effect is real and the bar is a WDDM-spread limitation of that tiny shape, honestly recorded. Guide Part 24 carries the full table.
- **H3 three-path consistency: supported (4/4 exact argmax agreement; adaptive vs fixed graph allclose 1e-4)** — re-captures do not change numerical semantics.
- **H4 capacity tracks demand: supported (4/4)** — capacity trajectories [1024, 1024, 340, …] / [2048, 2048, 680, …] settle after exactly one shrink recapture (observe 3); final capacities 340 / 680 / 339 against measured demand p50 272 / 544 / 271 (D_frac ≈13.3%), zero budget violations after adaptation; the final capacity is ≈3.01–3.02× smaller than the fixed C=0.5, quantitatively matching the D_frac evidence chain.

### AR016: Batched CUDA-Graph Capture + First P1-7 Wall-Clock Evidence

Spec: `specs/changes/AR016-batched-graph-verification/` (srs.md / design.md / tasks.md; req gate three rounds — Major-1 replay-time parent re-validation missing, whose fix introduced Major-R1: "silent successful eager fallback" is unimplementable for a parent cached at batch != 1 because the eager path itself raises the #41 RuntimeError, so the end state is split per sub-case — graph layer silently refuses, the caller receives eager's unmodified error; design gate three rounds — M-1~M-4 plus Major-N1 where the gate subagent empirically showed `ActivationCache.put` raises on same-key batch growth, so UT-1508(b)'s overwrite scenario uses `ChunkedActivationCache`). Full regression zero-regression (838 passed = 829 + 9 new; existing tests untouched); demo baseline unchanged; production delta is one file (`architecture_utils.py`) — `cuda_graph.py` is untouched.

#### Added

- **Batched CUDA-graph dispatch (P1-7 second tranche)**: batched-branch calls (`branch_ids`) no longer silently bypass the CUDA graph. `ManualFoldedForward` gains a second, independent runner (`graph_runner_batched`) capturing the `[N, T]` shape, plus `_graph_forward_batched_or_none` (a mirror of the serial degradation matrix with batched differences), `_parent_cache_complete_batched` (expected parent shape `[1, seq]` — the graph-side image of the #41 expand contract), and `_publish_graph_children` (per-branch `[1, T, H]` row-sliced **cloned** publishes plus profiler records, shape-identical to the AR015 eager batched stores). `FoldedGraphRunner` itself is unchanged — the AR002 static forward was already batch-agnostic.
- **Replay-time parent re-validation (the AR016 correctness core)**: the parent cache is re-checked before *every* replay (host-side shape metadata, no device sync). Without it, a parent re-cached at `[N, T, H]` under the same id after capture (legal with `ChunkedActivationCache`'s whole-chunk replacement) copies *cleanly* into the `[N, T, H]` static parent buffers — same-shape copy, no broadcast error — and every branch folds against a different parent row with the budget check none the wiser. The graph layer refuses; the eager body then raises the #41 RuntimeError before any store (zero cache/profiler pollution). A parent cached at batch != 1 keeps its eager error — the graph layer never swallows or alters the eager contract; only sequence mismatches and incomplete caches fall back to successful eager recompute.
- Degradation matrix per side: shape mismatch warns once (batched flag) and never re-captures; scheduler / non-cosine gate / non-CUDA share the single family flag (cross-reason suppression, serial precedent); an attention-mask object other than the captured one silently degrades; a failed capture permanently disables the batched path (EX-002) without touching the serial one; an exceeded divergent budget discards the replay output (one-time warning) and recomputes eagerly.
- Tests: UT-1501~1509 (`tests/test_batched_graph_forward.py`) — capture/replay parity vs batched eager (1e-4 + per-row argmax) with a capture spy pinning single capture, per-branch publish completeness (embedding rows 1e-6), serial-graph coexistence zero-interference plus a direct row-wise batched-vs-serial graph comparison (FR3 literal), shape-mismatch one-time warning, budget discard, the precondition matrix, EX-002 batched (one-shot capture failure via single-element `side_effect`, serial side unaffected after patch restore), the parent-batch != 1 double check (capture-time + replay-time; UT-1508(b) overwrites the parent under the same id on `ChunkedActivationCache` and pins refuse + eager RuntimeError + zero pollution + successful replay after re-populating at [1, T]), and publish isolation from the static buffers. Five mutations (dispatch removal / skipped clone / skipped branch / removed budget fallback / removed replay-time re-check) all turned their tests red and were restored byte-identically (SHA256).
- Benchmark: `scripts/ar016_batched_bench` (module syntax, `--quick` CI smoke subset) times four paths — raw baseline (N full forwards), folded serial, folded batched eager (AR015), folded batched graph — over N ∈ {2, 4, 8} × T ∈ {128, 512}, round-robin interleaved per repetition (RERUN_CHECKLIST §4.2), `perf_counter` around synchronize boundaries (CUDA-event timing would exclude the host-side prefill/readback/publish that are part of the graph path's real cost), TimingStats with n=10. Artifact: `results/optimization/ar016_batched_bench.json` (run date, CUPTI probe null + note, interleaving declaration, hypotheses declared separately from verdicts).

#### Experiment Results (synthetic LLaMA-layout, Quadro RTX 5000, p50 ms per verification step)

- **H1 batched eager ≫ folded serial: supported (6/6, 2.0–5.6×)** — the AR015 wall-clock dividend, first measured.
- **H2 batched graph < batched eager: inconclusive (supported in 4/6)** — the graph advantage narrows at large N×T as host-side prefill/budget-readback/publish costs grow (e.g. N=8/T=512: 17.4 vs 18.0 ms).
- **H3 four-path consistency: supported (6/6 exact argmax agreement; folded paths allclose 1e-4)**.
- **P1-7 (graph vs raw baseline): inconclusive — direction uniformly favorable (1.37–1.77×), 3/6 configs beyond the 2σ threshold.** Batched graph is the only folded path faster than the raw baseline; the honest verdict is recorded rather than a victory claim. Guide Part 23 carries the full table; remaining follow-ups: variable-length / multi-parent batches, Triton batched kernels, real-checkpoint end-to-end wall clock, and a host-side cost breakdown explaining the H2 narrowing.

### AR015: Batched-Branch Verification Forward in `folded_generate` (P1-7, First Tranche)

Spec: `specs/changes/AR015-batched-branch-verification/` (srs.md / design.md / tasks.md; req gate FAIL→fixed→re-check PASS, design gate FAIL→fixed→re-check FAIL→fixed→PASS, review gate FAIL→4 UTs added→re-check PASS). Full regression zero-regression (829 passed = 817 + 12 new; existing tests untouched); demo baseline unchanged (the demo does not route through `folded_generate` — it anchors the untouched serial layer path); production delta is three files.

#### Added

- **Batched-branch contract (`branch_ids`)**: `FoldedTransformerLayer.forward` accepts a keyword-only `branch_ids: Sequence[str]` — row `i` of the input belongs to `branch_ids[i]`, all branches share one `parent_branch_id` cached at batch 1 (the generate-loop shape), and the new `_forward_batched` path mirrors the serial per-child semantics branch-for-branch: cache-miss / no-parent / scheduler-disabled recompute the whole batch, the gate runs on the plain (non-fused) path against the stride-0 expanded parent, the three-way split is evaluated at batch granularity with a single readback, and stores plus profiler records happen per branch on row slices. Mutually exclusive with `branch_id` (`ValueError`); a parent cached at batch != 1 is a hard `RuntimeError` (expand contract).
- `ManualFoldedForward.forward` accepts `branch_ids` (batched calls run **eager** — the CUDA graph silently bypasses them, like the varlen fallback) and threads the kwarg through the layer loop unconditionally (serial passes `None`; serial behavior is bit-for-bit unchanged).
- `folded_generate` now dispatches same-parent, equal-length, parent-batch-1 candidate sets (>= 2 candidates, `ManualFoldedForward` only) to one batched folded forward via `_batched_forward_supported` / `_run_folded_forward_batched`; every other shape takes the exact serial loop. Layer-level launch count drops from `N * L` to `L` (the observable proxy for the guide's "~1/N wall-clock" claim — AGENTS #39 forbids citing unmeasured wall-clock numbers).
- Tests: UT-1401~1412 (`tests/test_batched_branch_forward.py`) — CPU varlen bitwise parity (logits / per-branch cache / profiler statistics), CUDA `allclose(1e-6)` + argmax parity, equal-length + split-layer variant, the launch-count proxy, an end-to-end `folded_generate` run (batched vs forced-serial: identical tokens/stable_ratio/acceptance_rate, zero rejected-branch cache residue), the serial-fallback matrix, layer-front-door contracts (including the first-generate-step cache-miss form where the root branch has no cached activations), graph-bypass non-interference, plus four review-gate additions: scheduler-disabled layers (whole-batch recompute per branch + dynamic-tau reuse on the folding layer), the all-stable fast path (a per-layer `_recompute_all` spy pins "no recompute" — a true-prefix child's recomputed values are value-identical to the reused parent FFN, so the call count is the only observable), the AR007 truncated-prefix all-stable donation, and the AR007 truncated parent FFN inside the mixed merge. Five additional mutations (scheduler branch / all-stable expand / fast-path truncation / merge truncation / dynamic tau) all turned the new tests red and were restored byte-identically (SHA256).

#### Design

- **Three-stage parent-FFN contract (design gate Major-1, two rounds)**: fetch-time check against the parent's stored grid `[1, prefix_len]` → explicit `expand(N, -1, -1)` → post-expand output check `[N, prefix_len]` (before the varlen suffix concat; degenerates to `stable_mask.shape` on the equal-length path, matching the serial `:476-482` template verbatim). The strict double check is what makes a "removed expand" mutation observable — PyTorch broadcasting would otherwise pass silently.
- **The first generate-loop step is a cache miss by construction** (req gate Major-2): the root branch has no cached activations and draft mode never forwards the parent, so the batched path must define (and pin) "parent id given but cache empty → whole-batch recompute, stored per branch" rather than treating it as an error.
- **T002 measured erratum of the design's split-generalization assumption**: the split FFN gathers the batch's divergent rows into one GEMM (M = D_total) while serial candidates gather per-candidate rows (M = D_i); when a serial candidate has exactly one divergent row, M=1 dispatches the BLAS **gemv** micro-kernel and the batched M>1 GEMM crosses the gemv/gemm boundary with a 1-ULP difference even on CPU (measured 2.4e-07). UT-1403 therefore pins `allclose(1e-6)` + argmax equality (masks and profiler statistics stay exactly equal); non-split paths (UT-1401) remain bit-exact on CPU.
- **Guide errata for the P1-7 row**: the serial loop lives at `folded_generation.py:139-143` (the cited `:114-117` was stale); `verification_engine.py` has no multi-branch loop (per-branch `verify_branch` API — list rot); `spiffy_baseline.py:77-86` is a deliberate non-folded baseline loop, out of scope. Later tranches (variable-length candidates, multi-parent batches, Triton batched kernels, CUDA-graph batch capture) remain open.

### AR014: Triton Fallback Granularity (P1-11) + Priority-Table Errata Sweep

Spec: `specs/changes/AR014-triton-fallback-granularity/` (srs.md / design.md / tasks.md; req gate FAIL→fixed→re-check PASS, design gate FAIL→fixed→re-check new Major→fixed→PASS). Full regression zero-regression (817 passed = 807 + 10 new); demo baseline unchanged; production delta is exactly one file (`fused_ops.py`, +58/−31).

#### Changed

- **Triton failure fallback granularity (guide P1-11)**: the three Triton kernels (`merge_stable_divergent_triton`, `gather_select`, `fused_gate_mask_count`) previously each had a single process-global boolean — one compile/launch failure permanently disabled that kernel for **all** dtypes and shapes. Fallback is now keyed by `(kernel_name, dtype, hidden_pow2_bucket)` in the new module-level `_TRITON_FAILED_KEYS` registry: a failed class silently falls back on subsequent calls while other dtypes / shape classes / kernels keep attempting Triton. `_hidden_bucket` is the pow2-ceiling shape class (4096→4096, 5000→8192); the shared `_record_triton_failure` helper emits **one** `RuntimeWarning` per failed key, now including kernel name, dtype, hidden class, and the original exception (the historical "…kernel unavailable" match substrings and exception-text passthrough are preserved verbatim; `stacklevel=3` keeps the warning pointing at the caller despite the helper frame).
- **Warning-contract change (explicit)**: the "once per kernel" guarantee becomes "once per (kernel, dtype, hidden-bucket) key"; the `_TRITON_*_DISABLED` booleans are no longer set by failures — they are narrowed to **manual kill-switches** (True forces the whole kernel onto the PyTorch path, checked before the registry; the existing monkeypatch-based debug/test usage is unchanged).

#### Added

- Tests: UT-1301~1306 (`tests/test_triton_fallback_granularity.py`) — dtype isolation (a dtype-conditional exploding kernel that delegates non-matching dtypes to the real kernel, a forbidden-fallback window proving the surviving dtype truly takes Triton, and a second failure key emitting a second independent warning), same-key silence (a launch-count spy kernel pins "the kernel is not re-attempted" directly via `count == 0` — warning-absence alone cannot distinguish the two dedup guards), hidden-bucket isolation (H=512 explodes while H=2048 takes real Triton), manual kill-switch compatibility (forced fallback: correct result, zero warnings, zero registry growth), warning diagnosability (kernel/dtype/hidden-class substrings), and the pure `_hidden_bucket` mapping on CPU.
- Three legacy failure-injection pins migrated to the key contract with measured key literals (`("gather_select", "torch.float32", 32)` / `("merge", "torch.float32", 128)` / `("gate", "torch.float32", 64)`) plus registry isolation via `monkeypatch.setattr`; the two existence pins gained `isinstance(..., set)`. Four zero-change consumers verified green.
- Mutation spot-checks (byte-exact restored): registry-not-written (3 red), dispatch-not-checking (1 red — the spy count is the designed observation surface), key-without-dtype-dimension (4 red). The "warning-unconditionally" mutation is black-box unreachable (the dispatch check keeps the helper from ever seeing an already-registered key) — removed from the set with the rationale recorded; the helper guard is retained as defense in depth.

#### Design

- **Priority-table errata sweep (third consecutive AR finding list rot, now extended from 6.1/6.2 to the P1/M tables)**: P1-1 (only `ffn_out` is cached; the gate reads layer L−1), P1-2 (the `fetch()`/`fetch_masked()` split exists; `cache.get(token_mask)` has zero production callers), P1-5 (`_evict_old_branch_steps` ring + prune `clear_branch`), P1-8 (batched `torch.topk(dim=-1)`, vectorized transfer-token schedule, LLaDA early stop), P1-10 (AR002 final-norm discovery + regression tests) — all checked off with evidence. **P1-9 declared invalid**: its premise (a separate parent-norm re-read) does not exist — cosine/l2/pearson are single-pass fused reductions and the fused gate kernel computes norms in-register. P1-3/P1-6/M-4 marked ◐ (the named sub-deliverables exist; the batching main body remains P1-7). M-3 checked off. **M-9 deliberately not checked off** (req-gate Major): only 2 of the original 7 global-seed sites were actually fixed — the guide's named site (`draft_generator`) uses local `torch.Generator`s and `ablation_study` is snapshot-wrapped, but 5 sampler/adapter top-level explicit seeds remain; they are characterized as deliberate reproducibility seeding rather than silent pollution, and a full localization would be its own item.
- Gate findings worth recording: the observable projection of the dispatch-side registry check is the **launch count**, not warning absence (the dual guard makes the two indistinguishable); a "forbidden fallback" monkeypatch must be scoped to the success-path call only — failure paths legitimately use the fallback, so an always-on forbidden patch makes the first failure-path assertion unreachable.

### AR013: Guide-6.1 Test-Debt List Final Audit & Closure (test-first + dead-code removal)

Spec: `specs/changes/AR013-test-debt-closure/` (srs.md / design.md / tasks.md; req gate first-round PASS, design gate FAIL→fixed→PASS). Full regression zero-regression (807 passed = 799 + 8 new); demo baseline unchanged; production delta is exactly `-6` lines in `branch_tree.py`.

#### Removed

- **`BranchTree.best_accepted_leaf` (branch_tree.py:88-93)**: zero-caller, zero-test dead code (repo-wide scan: only the definition and the guide's own debt line referenced it; the guide itself flagged it "死代码"). The correct disposition for dead code is removal, not tests. A removal pin (UT-1208: `not hasattr(BranchTree, "best_accepted_leaf")` plus a live-API-surface `hasattr` check on add/get/get_ancestors/prune) prevents resurrection.

#### Added

- Tests: UT-1201~1204 (`tests/test_algo_experiment_helpers.py`) pinning the two genuinely-uncovered experiment helpers named by guide :232 — `make_child` (same-seed independent recompute of the Generator→randperm→randint stream, dual-call determinism, cross-seed distinguishability, zero/negative-flip early-return clone semantics, `min` clamping, vocab-1, parent immutability, float32 dtype passthrough) and `fidelity` (identity metrics on a hand-picked exactly-representable-norm tensor, the 1e-12 variance floor, exact four-key schema, fp16→fp32 upcast, and a first-principles hand-computed case with literal constants — mse=0.25 / top1=0.5 exact, rel_mse / cosine via approx; the variance expectation mirrors the explicit unbiased formula `((base-mean)**2).sum()/(n-1)` instead of calling `base.var()`, so a torch default change cannot silently drag the test along).
- Tests: UT-1205~1207 (`tests/test_branch_tree.py`) — first direct API pins for the live `BranchTree` surface consumed by `folded_generate` (add/get round-trip, parent-child linking, the duplicate-add guard at :57-58, the orphan-parent condition at :55, `BranchNode.seq_len`, `get_ancestors` ordering / include_self / unregistered both arities, and `prune` recursion + parent unlink + missing-id no-op + the dangling `root_id` after pruning the root). Measured-behavior finding: `prune(root)` recursively clears the entire tree including siblings — pinned as current behavior with a docstring note.
- Mutation spot-checks (byte-exact restored): dropping the `make_child` clamp line reds UT-1202 with `RuntimeError` (shape mismatch 15 vs 12 — `randperm` slicing silently caps but `randint` uses the unclamped count); replacing the `fidelity` variance floor with the raw variance reds UT-1203 with `ZeroDivisionError` (Python-float division, not inf); deleting the `prune` recursion reds UT-1207 with a surviving leaf.

#### Design

- **Guide 6.1 (:228-:233) audited item-by-item and closed 6/6 with errata**: :228's premise was already closed by AR004/AR006 (31 folded_generate tests, 7 direct TargetMatchAcceptancePolicy cases, 59 acceptance.py hits) — the real remainder (direct BranchTree pins + dead code) is what this AR delivered; :230 was closed by T018's frozen pre-T018 reference parity; :231 was closed by AR002's `test_manual_folded_forward_applies_final_norm`; :232 was mostly closed by `test_experiment_stats.py` (16 cases) with `make_child`/`fidelity` as the true gap; :233's wording was inverted — the function-scope autouse seed **resets** per test (a decoupling mechanism, not a coupling source); the residual (implicit reliance on seed 42) is handled by the AR012 explicit-seed discipline for all new tests, and a wholesale rewrite of 800 tests was rejected (huge churn, no flaky-fix payoff).
- Gate findings worth recording: `seq_len` belongs to `BranchNode`, not `BranchTree` (a literal hasattr on the wrong class would have been a guaranteed first-run red); identity-input `cosine_similarity` returns 0.99999994 rather than 1.0 for ~60% of random tensors (norm sqrt→square round-trip) — exact identity assertions require a hand-picked exactly-representable norm.

### AR012: Sampler Stochastic-Branch Numeric Coverage (guide 6.1 :229, test-only)

Spec: `specs/changes/AR012-sampler-stochastic-coverage/` (srs.md / design.md / tasks.md; req and design gates each FAIL→fixed→PASS). Full regression zero-regression (799 passed = 785 + 14 new); demo baseline unchanged; `actfold/` untouched.

#### Added

- Tests: UT-1101~1106 (`tests/test_sampler_stochastic_branches.py`), pinning the numeric semantics of every RNG-consuming sampler branch — the branches a P1-8 vectorization rewrite would most easily break:
  - **Pure-function exact-recompute pins** (same-seed, operator-by-operator hand rewrite, no production imports in the recompute path): UT-1101 `add_gumbel_noise` with T∈{0.5, 1.0, 2.0} (bit-exact `exp(logits)/(-log u)**T` in float64, plus the first value-level no-op pin for T=0.0 — the pre-existing assertion was shape-only), UT-1105 `sample_tokens` temperature>0 across six filter/confidence combinations including the joint top_p+top_k case mirroring the production order (top_p first), with `-inf`-filtered positions pinned to exactly-zero probability.
  - **End-to-end stochastic-branch pins** (seed determinism + RNG-call spy as the branch-taken proof + structural invariants): UT-1102 LLaDA `remasking="random"` (recorded `torch.rand` shape, full reveal, suppress-token interaction), UT-1103 `stochastic_transfer=True` with T=0.5 (Binomial proxy spy, in-situ gumbel, history monotonicity), UT-1104 Dream `alg_temp>0` per-row multinomial soft selection (spy + a weak convergence property: at `alg_temp=1e-8` the softmax is exactly one-hot and each row's first sampled index equals the confidence argmax).
  - **Stochastic-schedule numeric pin**: UT-1106 `get_num_transfer_tokens(stochastic=True)` — same-seed determinism, stochastic≠deterministic distinguishability (a pinned seed), and a 100-seed statistical-mean pin against the analytically recomputed `p₀ = 0.12500083446502686` (production's four f32 roundings replicated by hand; 6σ tolerance).
- Mutation spot-checks (run at Red, then byte-exact restored): negating the remasking branch, hardcoding `stochastic=False`, and flipping the `alg_temp` condition each turn the corresponding spy-empty assertion red — the tests are not vacuously green.

#### Design

- **Frozen-reference parity deliberately not extended to stochastic branches** (req-gate Major): the T018 frozen reference classes share every RNG call site with production (bit-equal under the same seed), so parity there is a circular pin (proves "code unchanged", not the math) and over-pins the current RNG layout — a legitimate P1-8 RNG reorder would false-alarm. The pinning strategy is seed-determinism + spy + independent recompute instead.
- Guide 6.1 :229 checked with an erratum: CFG / suppress_tokens / greedy / low_confidence were already covered by T018 reference parity; the real gap was the stochastic branches (closed by this AR).

### AR011: `gather_select` Unbanned on the Truncated (Parent-Longer) Path (AR007 review follow-up)

Spec: `specs/changes/AR011-gather-select-truncated-path/` (srs.md / design.md / tasks.md, T001–T004 all passing). Full regression zero-regression (785 passed = 782 + 3 new); demo baseline unchanged; `vectorized_cache.py` and `fused_ops.py` untouched — the unban relies on (and pins) the existing `fetch_flat` contract.

#### Changed

- **`FoldedTransformerLayer._merge_parent_child` no longer skips the fused `gather_select` merge when the parent was truncated (AR007 `parent_longer`)**: the D3 dispatch condition drops `not parent_longer`, so a parent-longer reuse at large shapes (T≥2048, H≥4096, CUDA) merges in a single fused pass like the equal-length path. The AR007-era rationale ("the flat-row layout contract covers exactly `seq_len` stored tokens") was factually wrong — `fetch_flat`'s guard requires the stored tokens to **cover** the request (allowing `total > seq_len`) and its row formula `flat_rows[b*seq_len+t] = t*batch+b` is computed for the requested child length, reading exactly the truncated parent prefix. Safety nets are unchanged and pinned: an eviction-wrapped parent (`start > 0`), a shorter varlen parent (`total < seq_len`, plus the explicit `prefix_len` skip), and non-vectorized caches (no `fetch_flat`) all still fall back — bit-identical. The `parent_longer` parameter is retained (no signature change; no longer gates the dispatch).
- AGENTS #20 updated (truncated-path merge now fused) and #30's historical threshold wording errata'd to the current D3 values (T≥2048 with H≥4096 on CUDA, pinned by T016).

#### Added

- Tests: UT-1001~1003 (`tests/test_gather_select_truncated.py`, T014/T016 patterns): parent-longer fused equivalence (exactly one recorded `gather_select` call, dual-instance `torch.equal`, output-composition sanity, pre-forward mixed-stability sanity to avoid tau-flakiness), eviction-wrap fallback (independent capacity=6 cache; the child built from the **surviving window** prefix — a naive token-0..3 prefix would compare against the wrong window, go all-divergent, and skip the merge entirely; output-composition sanity proves the merge was reached), and the varlen not-fusable behavior pin.

### AR010: Model-Family Inference Unification & Fallback Visibility (guide 6.2 :241)

Spec: `specs/changes/AR010-model-family-inference-unification/` (srs.md / design.md / tasks.md, T001–T004 all passing). Full regression zero-regression (782 passed = 773 + 9 new); demo baseline unchanged; the inference rules are preserved verbatim (the existing `test_registry_resolve_family` assertions pass unchanged as the behavioral-pinning gate).

#### Changed

- **`ModelRegistry._resolve_family` no longer duplicates the keyword rules**: the `auto` branch now delegates to `actfold.models.utils.infer_model_family` (previously a line-for-line duplicate; `infer_model_family` had zero production callers). The rules themselves — lowercase folding, the `"llada"` / `"dream"` / `"fast" and "dllm"` substring checks, the `causal_lm` fallback — are preserved verbatim in `utils`, which becomes the single source of truth; the explicit `model_family != "auto"` path returns unchanged.
- **Substring matching kept (guide erratum)**: the :241 complaint about `'fast' in name` was analyzed and retained — the fast-dllm rule requires both substrings to match (a very narrow false-positive surface), while token-based matching would break the concatenated "FastDLLM" naming (lowercased `fastdllm` is a single alphanumeric token). Config-based (`AutoConfig.model_type`) revalidation stays out of scope.

#### Added

- **`UserWarning` on the silent auto fallback**: when `model_family="auto"` and inference falls back to `causal_lm` (no known keyword matched — the production default via `ActFoldConfig.model_family="auto"`), `_resolve_family` warns with the model name, an explicit-`model_family` pointer for diffusion models, and the dynamic list of registered families (`stacklevel=3`, attributed to the caller through `load`). Explicit family selections and known-keyword matches never warn; the fallback return value is unchanged (genuine causal models keep working — this is visibility, per AGENTS #10, not a behavior change).
- Tests: UT-901~904 (`tests/test_model_family_inference.py`): six-case equivalence matrix (mixed case / concatenated names / explicit passthrough), `pytest.warns(match="auto-detect fell back to causal_lm")` plus model-name assertion with a recording dummy registry (no `from_pretrained`), and `simplefilter("error")`-based zero-warning pins for the explicit-family and known-keyword paths.

### AR009: Silent-Misconfiguration Guards for Folding (guide 6.2 :239/:240 visibility fix)

Spec: `specs/changes/AR009-silent-misconfig-guards/` (srs.md / design.md / tasks.md, T001–T004 all passing). Full regression zero-regression (773 passed = 768 + 5 new); demo baseline unchanged; the fallback behavior itself is preserved line-for-line (AGENTS #16/#17 legitimate degradation path — only visibility is added).

#### Added

- **One-time `UserWarning` on the embedding-proxy fallback**: `ActFoldVerificationEngine._estimate_stable_ratio` warns once per engine instance (`_proxy_ratio_warned` flag, the AR002 `_graph_degraded_warned` precedent) when the stable ratio had to be estimated from layer-0 embeddings because no per-layer profile was recorded — i.e., folding did not run through a folded forward pass. The message points at the two common causes (adapter without a folded model, AGENTS #17; base model not propagating branch kwargs, AGENTS #18) and is attributed to the caller via `stacklevel=3`. A fresh engine instance warns again; the layered-profile path never warns.
- **Construction-time rejection of `AdaptiveQuantileGate` × `FoldingScheduler`** (AGENTS #29 upgraded from a documentation convention to a code contract): `FoldedTransformerLayer.__init__` and `ActFoldVerificationEngine.__init__` raise `RuntimeError` with the shared `_ADAPTIVE_GATE_SCHEDULER_MSG` (folded_transformer.py) when given that combination — the scheduler's dynamic `gate.tau` overwrites would silently have no effect on the quantile gate's top-k semantics. `SplitFoldedTransformerLayer` inherits the guard via `super().__init__`, `ManualFoldedForward` fails at construction through `_wrap_layer`, and `isinstance` covers AdaptiveQuantileGate subclasses. Legal combos (`SimilarityGate`+scheduler, `AdaptiveQuantileGate`+`scheduler=None`) are pinned by tests; all 15 existing `AdaptiveQuantileGate(` constructions carry no scheduler (zero regression surface, verified).
- Tests: UT-801~805 (`tests/test_silent_misconfig_guards.py`): exactly-once warning per instance / two-instance semantics (2 warnings) / layered-profile zero-warning with profiler mean / layer+split guard with legal-combo controls / engine guard with legal-combo controls; UT-801's returned ratio is pinned bit-exact against an independently computed gate mean (anti-tautology).

#### Changed

- **Guide 6.2 :240 erratum**: `detect_architecture` already raises `RuntimeError` on layer-stack discovery failure (architecture_utils.py:281-285, AGENTS #10-aligned) — the "silent degradation" premise was stale; the remaining silent spot was the engine's proxy fallback, now visible via the warning above. `nn.Sequential`/custom containers remain an explicit-wiring limitation (AGENTS #23) and the `folding_status` enum has no consumer (not built).
- AGENTS #29 now documents the enforced construction-time `RuntimeError`.

### AR008: On-Demand `output_hidden_states` in `GenericDiffusionLLM.forward` (guide 6.2 memory-debt fix)

Spec: `specs/changes/AR008-generic-hidden-states-on-demand/` (srs.md / design.md / tasks.md, T001–T004 all passing). Three output shapes (logits / last_hidden_state / hidden_states-only) behave identically; the RuntimeError text is preserved verbatim; full regression zero-regression; demo baseline unchanged. Real 7B checkpoint measurement remains deferred to the RERUN_CHECKLIST.

#### Changed

- **`GenericDiffusionLLM.forward` no longer unconditionally collects all-layer hidden states**: `output_hidden_states` defaults to `False` (~256 MB fp32 peak saved at 7B/T=512 for the common `logits`/`last_hidden_state` shapes). Only when the model output exposes neither field does the wrapper set the per-instance `_needs_hidden_states` flag and re-run the same forward once with collection enabled — at most one double forward per instance (the flag is sticky and safe: a logits-first model still takes the early return when the flag is on). The probing order and the verbatim "Model output exposes neither ..." RuntimeError are preserved line-for-line. `LLaDAModel`/`DreamModel` inherit `generic.forward` and benefit automatically; the sampler folded path and `_native_sampler` already bypass this wrapper (zero behavior change).
- New private attribute `_needs_hidden_states: bool` (no public API change).

#### Added

- Tests: UT-701~705 (`tests/test_generic_forward.py`) pin the three output shapes via **flag-conditional stubs** (False → `hidden_states=None`, mirroring real HF semantics; a constant-tuple stub would make the double-forward assertion a false Red) with call-log spies, the verbatim error message, exactly-two-calls termination, and branch-kwarg stripping under the rerun; EX-805 (CUDA) measures the `max_memory_allocated` peak delta between flag-False and flag-True instances over a 24-layer parameter-free stub — ≥150 MiB observed against the theoretical 192 MiB.

### AR007: Truncated-Prefix Reuse for Longer Parents (P3-2 first tier)

Spec: `specs/changes/AR007-truncated-parent-reuse/` (srs.md / design.md / tasks.md, T001–T004 all passing). Equal-length and append-only paths are bit-identical (full regression 761 passed, zero regressions; demo baseline unchanged). The paged multi-ancestor structure remains the P3-2 follow-up.

#### Changed

- **`FoldedTransformerLayer` no longer treats a longer parent as absent**: a cached parent with `T_p > T_c` is truncated to `h_parent[:, :T_c]` and folds through the existing equal-length branches (the symmetric completion of AR003: AR003 folded the longer child, AR007 folds the longer parent). A true prefix is certified all-stable by the gate and the parent FFN output is reused after the same truncation (`torch.equal` vs. full recompute, zero `_recompute_all` calls by spy); a non-prefix child diverges at the gate and recomputes — the similarity gate is the semantic safety net, so truncation never requires a *true* prefix for correctness. The fast path truncates by the **fetched shape** (anomalously length-mismatched per-layer cache entries resolve deterministically). Batch/hidden mismatches and zero-length entries still mean no-parent.
- `_merge_parent_child` gains `parent_longer: bool = False`, which conservatively skips the `gather_select` fast path and reads `fetch` + truncate + merge instead (bit-identical; one less fused kernel). The equal-length path is byte-for-byte unchanged.
- `_get_parent_ffn_output` accepts a parent FFN **longer** than the mask and truncates it, **only on the `prefix_len is None` path**; the variable-length (AR003) exact-shape `RuntimeError` contract is unchanged (EX-505 pinned).
- CUDA graph is untouched: `_parent_cache_complete`'s shape check routes a longer-parent child to a silent eager fallback (same path as AR003 variable-length steps; no capture, no warning, no disable — a later complete-parent child still captures).
- AGENTS #20 and README limitation #7 updated; stale docstrings in `test_varlen_parent_longer_full_recompute` (UT-203) and `test_bs001_folded_generate_graph_end_to_end` corrected to the current semantics.

#### Added

- Tests: UT-601~605 (`tests/test_folded_transformer_varlen.py`, `TestAR007TruncatedParentReuse`) and IT-611/IT-613 (`tests/test_cuda_graph.py`) — true-prefix full-reuse with per-layer `stable_ratio == 1.0`, independently-sampled non-prefix full recompute, batch-mismatch no-parent pin, fast-path reuse spy, cross-layer length-misalignment `RuntimeError` pin, end-to-end `ManualFoldedForward` truncation reuse, and graph-mode silent-eager no-capture.

### AR006: Real Draft Model, Tier 1 — `model_draft` Mode (P3-3 first tier / M-2 tier 3 / B10 final fix)

Spec: `specs/changes/AR006-model-draft/` (srs.md / design.md / tasks.md, T001–T005 all passing). The five pre-existing draft modes are bit-identical (full regression zero-regression; demo baseline unchanged — demo uses the default mode). Medusa/Eagle checkpoint loading remains tier 2 (requires real weights).

#### Added

- **`DraftGenerator` mode `"model_draft"` + `draft_model` constructor parameter** (default `None`; a callable mapping tokens `[B, T]` to logits `[B, T, V]` — any `nn.Module` qualifies). The draft logits come from one independent `draft_model(parent.tokens)` forward — the real speculative-decoding semantics — unlike `logits_draft`, whose logits come from a *target* parent forward (the "cheating upper bound"). The `parent_logits` argument is ignored in this mode.
- Validation at `generate` time (mirroring the `parent_logits` precedent): missing `draft_model` → `RuntimeError`; wrong shape, non-floating dtype, or device mismatch vs. `parent.tokens` → `ValueError`; draft-forward exceptions propagate unchanged.
- `logits_draft` / `model_draft` share the sampling implementation via `_sample_tokens_from_logits` (the two paths cannot drift apart); append positions sample from the draft logits at the parent's last position (first-tier approximation of autoregressive drafting).
- `ActFoldConfig.draft_mode` whitelist gains `"model_draft"` (value-domain extension, no new field).
- `BenchmarkRunner(config, model=..., draft_model=...)`: optional injection; `draft_mode == "model_draft"` without a `draft_model` → construction-time `RuntimeError` (AGENTS #10: no silent fallback). `AblationStudy` / `folded_generate` accept a `model_draft` generator through the existing `draft_generator` parameter — zero signature changes.
- Acceptance-rate gradient fixtures (tests only): a 4-layer causal token-wise target plus a single-layer correlated draft (sharing the target's first layer) prove `model_draft` acceptance sits strictly between the random floor and the `logits_draft` upper bound (measured ≈0.45 vs. random ≈0.001 vs. upper bound 0.5 at `top_k=2`) — the paper-grade acceptance semantics AR004 anticipated.

#### Changed

- README limitations #5/#8 and AGENTS #40 updated: the mechanism-level acceptance-rate caveat now points at `model_draft` (tier 1); Medusa/Eagle checkpoints remain tier 2.

### AR005: FFN/MoE FLOPs Geometry Correction (P3-4)

Spec: `specs/changes/AR005-ffn-flops-geometry/` (srs.md / design.md / tasks.md, T001–T005 all passing). Default-geometry paths are bit-identical (synthetic models keep the 4h-MLP estimate; demo baseline 85.5% / 2.35e-03 / 93.75% unchanged); full regression 735+ passed.

#### Added

- **MoE accounting in `count_diffusion_llm_flops`**: `moe_num_experts` / `moe_top_k` (the trigger — per-token FLOPs depend on top-k alone) / `moe_intermediate_dim` (falls back to `ffn_intermediate_dim`, then `4h`) / `moe_shared_expert` (adds one always-on expert-equivalent) / `moe_num_layers` (defaults to all layers; DeepSeek-style dense prefixes derive from `first_k_dense_replace` so mixed stacks are not overestimated). Per-token expert FLOPs: `2 · n_matmul · moe_inter · h · (top_k + shared)`; router gating (~1%) is ignored by design.
- **Automatic FFN geometry extraction (AR001 leftover wiring)**: `DiffusionLLM.config` (concrete property forwarding to `self.model.config` — real subclasses keep the HF config on the wrapped module) plus seven geometry properties (`ffn_intermediate_dim` / `ffn_type` / five MoE keys), all backed by a single `_extract_ffn_geometry(config)` implementation (attribute-name union across Qwen2/DeepSeek MoE families; SwiGLU family {silu, swish, swiglu} → 3 matmuls; unknown names degrade safely to the dense default).
- **`model_ffn_flops_kwargs` chain resolution**: drills `underlying_model` wrappers (adapter → model), prefers direct attributes/properties, then resolves the config via `_resolve_model_config` (`target.config` → `target.model.config`). The three call sites (verification engine, ablation study, base eval adapter) required **zero changes** — real-checkpoint TFLOPs now reflect true FFN geometry instead of silently falling back to 4h-MLP (the 10–25% systematic bias noted in the guide).
- Keys of the helper's returned dict are additive-only (5 new MoE keys default to None/False).

#### Changed

- README demo-output sample updated to the corrected 85.5% reduction (was a stale pre-AR001 78.5%).

#### Known follow-ups

- Real MoE checkpoint validation (formula + config extraction are synthetic-proven; no local MoE weights) — per `docs/RERUN_CHECKLIST.md`.
- Shared-expert intermediate dim is approximated by the routed `moe_intermediate_dim` (sub-5% effect).
- Attention T² term and KV-cache bandwidth remain opt-in/separate (M-4).

### AR004: True Speculative-Decoding Acceptance Semantics (P2-5)

Spec: `specs/changes/AR004-logit-acceptance-semantics/` (srs.md / design.md / tasks.md, T001–T006 all passing). Default-parameter behavior is unchanged (threshold 0.0 accepts every branch; demo baseline 85.5% / 2.35e-03 / 93.75% unchanged); full regression 699 passed.

#### Added

- **`actfold/speculative/acceptance.py`**: four pure functions implementing true acceptance semantics — `target_argmax_accept_mask` (per-position mask: draft token == target argmax at the same position), `draft_region_mask` (the positions a branch newly claims: prefix diffs vs the parent plus appended suffix; `T_p < T_c` truncates to the common prefix, a batch mismatch raises `ValueError`), `acceptance_rate` (masked mean over the draft region; empty region → 1.0), and `mean_log_prob` (fp32 `log_softmax` gather mean; empty region → 0.0 sentinel). Shared `_validate_token_tensor` gives all three a single shape-validation path.
- **`TargetMatchAcceptancePolicy`** (`acceptance_policy.py`): selects the candidate whose appended token matches the target argmax of its own forward logits (batch-mean rate; ties → first in list order; `logits=None` candidates are skipped; all-None → `candidates[0]`).
- `folded_generate` **per-step acceptance reporting**: after policy selection (decoupled from the policy by design), each step's accepted branch is scored against the pre-advance parent snapshot via the shared helpers; `accepted.metadata["acceptance_rate"]` is written per step and aggregated into the new `FoldedGenerationResult.acceptance_rate` field (mean over steps, 0.0 when no steps).
- `ActFoldVerificationEngine`: `ema_alpha: float = 0.3` constructor parameter (domain `(0, 1]`, else `ValueError`), a public `ema_acceptance_rate` attribute (first call initializes to that call's rate — no zero-prior dilution), and `VerificationResult` fields `acceptance_rate` / `mean_log_prob` / `ema_acceptance_rate`.

#### Changed

- **Acceptance decision switch**: `verify_branch` now decides `accepted = acceptance_rate >= acceptance_threshold` (was `stable_ratio >= threshold`). With the default threshold 0.0 and `acceptance_rate ∈ [0, 1]` every branch still passes — a zero-change default, but the knob now controls true acceptance instead of activation similarity; `stable_ratio` is still measured and reported.
- **`actfold_score` semantic upgrade** (key name unchanged by design): the meaningless `logits.float().mean().item()` placeholder (verification_engine.py:122) is replaced by `mean_log_prob` over the draft region. The only consumer compares ordinally (test_integration.py:169), so the change is consumer-safe.
- **`SpiffyBaseline.verify` `baseline_score`** placeholder replaced by `mean_log_prob` over all positions (key name unchanged).

#### Known follow-ups

- A real draft model (P3-3) is what the acceptance rate should measure against; current drafts are random/perturb modes, so the reported rate is a mechanism metric until then.
- Multi-ancestor reuse (P3-2), `Branch.hidden_states` lazy referencing, and MoE FLOPs correction remain open.

### AR003: Variable-Length Prefix Folding (P3-1) + BenchmarkRunner Graph Wiring (AR002 follow-up)

Spec: `specs/changes/AR003-var-len-prefix-folding/` (srs.md / design.md / tasks.md, T001–T007 all passing). Equal-length folding is bit-identical to the previous behavior (demo baseline 85.5% / 2.35e-03 / 93.75% unchanged).

#### Added

- **Append-only prefix folding in `FoldedTransformerLayer`**: a parent cached at `T_p ≤ T_c` (same batch/hidden) now folds — the gate compares only the child prefix `child[:, :T_p]` against the parent, the appended suffix is always divergent via an explicit all-False mask tail (robust for any tau, unlike a zero-padded parent), and the mixed-path merge aligns the parent FFN with the recomputed suffix (`cat([parent_ffn, child_out[:, T_p:]])`) before the existing `merge_stable_divergent`. The all-stable fast path remains reachable only on the equal-length path (var-len `stable_count ≤ B*T_p < B*T_c` by construction); `gather_select` is skipped on var-len steps (its `fetch_flat` contract indexes parent rows by the child length); the fused gate kernel is equal-length-only by its same-shape contract and the `B*T ≥ 1024` threshold (design D2).
- `folded_generate` **actually folds across steps**: each AR step's child is one token longer than its parent; a causal model's prefix is bit-identical to the parent's, so the prefix is all-stable and only the new token recomputes. Generated tokens are `torch.equal` to the eager full recompute; mean stable ratio ≈ 0.84 on the causal synthetic (4 growth steps, T 4→8). This supersedes the old "stable ratio 0.0, never folds" semantics.
- `BranchManager.align_tokens`: prefix alignment implemented (was `NotImplementedError`) — returns `(parent.hidden_states[0], child.hidden_states[0][:, :T_p])`; `T_c < T_p` or a batch mismatch raises `ValueError`; equal-length returns are bit-identical to before. API-complete (no production consumer yet).
- `BenchmarkRunner` now consumes `ActFoldConfig.use_cuda_graph` / `graph_capacity_ratio` (AR002 follow-up): `_build_folded_model` builds `ManualFoldedForward` (the supported non-mutating path) with both flags threaded through; architecture-detection failures map to `None` (the legacy `folding_applied` semantics).

#### Changed

- **`BenchmarkRunner` production path migrated from deprecated `FoldedModel` to `ManualFoldedForward`**: no in-place mutation, zero state_dict drift (behavioral equivalence was established bit-exact in AR002). When `use_cuda_graph=True` the runner attaches **no** `FoldingScheduler` (design D3): the graph contract (#37) forbids a scheduler, so attaching one would silently degrade every replay to eager — dynamic tau and graph replay are mutually exclusive.
- Var-len graph behavior (contract, verified by test): variable-length steps never capture — `_parent_cache_complete`'s shape check routes them to a **silent** eager fallback (no warning, no disable) where prefix folding applies. The fixed-shape (diffusion verification) graph workload is unchanged.

#### Known follow-ups

- Multi-ancestor reuse (P3-2), a parent longer than the child (truncation reuse), and `Branch.hidden_states` lazy referencing remain open.
- End-to-end wall-clock gains of prefix folding on a real speculative-decoding workload are to be measured on the target machine per `docs/RERUN_CHECKLIST.md`; this AR ships the mechanism plus synthetic-proxy evidence (stable ratio, token equality, graph non-interference).

### AR002: M4 Kernel Fusion & CUDA-Graph Verification Loop (P2-2/P2-3 sync elimination, B12 normalization)

Spec: `specs/changes/AR002-m4-graph-capture/` (srs.md / design.md / tasks.md, T001–T010 all passing). 624 passed, mypy --strict clean; demo baseline 85.5% / 2.35e-03 / 93.75% unchanged.

#### Added

- `actfold/core/split_layer.py`: `_exact_divergent_index(mask)` / `_padded_divergent_index(mask, C)` — device-side exact and fixed-capacity divergent index computation (P2-2: the `nonzero()` host sync is gone); `_recompute_merged` gains a `stable_count` kwarg so callers forward the count instead of a second readback.
- `actfold/core/fused_ops.py`: `fused_gate_mask_count` (M4a) — a single Triton kernel computing cosine similarity, the stable mask, **and** the stable count in one pass (fp32 block accumulation, `denom=max(sqrt(nc*np), eps)`, clamp, NaN→divergent; count via atomic accumulation). Dispatches for `SimilarityGate` (exact type) + cosine metric at `B*T >= 1024` (`_FUSED_GATE_MIN_TOKENS`); 6 ValueError input checks; `_TRITON_GATE_DISABLED` escape hatch; PyTorch fallback bit-exact (fp32/fp16/bf16).
- `actfold/core/cuda_graph.py`: `FoldedGraphRunner` (M4b) — static-buffer CUDA graph capture/replay for the fixed-shape folded child forward. Static buffer group: `tokens` / `parent_static[L+1]` (slot 0 = embedding) / `mask_buf[L]` / `count_buf[L]` (int32) / `child_buf[L]` / `logits`; side-stream warmup ×2 + `torch.cuda.graph` capture; replay prefetches the parent from cache on the host, zeroes counts, replays, and validates divergent budgets with the **single** allowed readback (`D == C` passes, `D == C+1` rejects). Capture preconditions raise `RuntimeError` (non-CUDA / scheduler / non-cosine gate / missing parent).
- `ManualFoldedForward`: `split_layers` / `split_min_tokens` / `use_cuda_graph` / `graph_capacity_ratio` (0 < r ≤ 1 else ValueError) — the graph path is **opt-in** and lazily captured on the first eligible child step. Full degradation matrix: shape mismatch (one-time warning, no re-capture, checked before the parent), scheduler / non-cosine gate / non-CUDA (one-time warning), non-pinned mask object or incomplete parent cache (silent eager), exceeded budget (one-time warning + eager recompute). The validated replay publishes `child_buf` clones into the cache and returns `logits.clone()`.
- `ActFoldConfig`: `use_cuda_graph=False` / `graph_capacity_ratio=0.5` (validated).
- `scripts/ar002_graph_bench.py`: BS-007 evidence benchmark (`run_bench` / `write_results` / CLI; artifact `results/optimization/ar002_graph_bench.json`). Measured on Quadro RTX 5000 (B=2, T=512, 4 layers): eager 5.083 ms/step vs graph replay 2.687 ms/step (**-47.1%**), 20/20 budget-validated steps. Kernel-launch counts are recorded only on CUPTI-capable builds; this host's `LIBKINETO_NOCUPTI` torch records zero CUDA profiler events, so counts are `null` with an explicit note (never fabricated) and the launch-reduction test self-skips there.
- `DraftGenerator`: dual local generators (cpu/dev RNG) — `generate(seed=...)` no longer touches the global RNG; identical seeds reproduce identical drafts.
- `BaseEvalAdapter._resolve_max_new_tokens`: per-task generation-length table (override > explicit > table > 256); `EvalPlusAdapter` maps `humaneval_plus`/`mbpp_plus` → 512.

#### Changed (breaking / legacy guidance)

- **`ManualFoldedForward` is the supported folded-forward path** (B12 complete): it now carries `split_layers` support, zero state_dict drift, and is bit-exact against `FoldedModel`. `AblationStudy` builds its internal stack with `ManualFoldedForward` (no in-place mutation of your model, no restore needed). `folded_generate` / `FastDLLMAdapter` accept `FoldedModel | ManualFoldedForward | None`.
- **`FoldedModel` is deprecated** (kept as legacy, behavior unchanged): it mutates the wrapped model in place (AGENTS #19) and depends on the deprecated contextvars branch context. Migrate by constructing `ManualFoldedForward(model, cache, gate, split_layers=True, ...)` and calling it with explicit `branch_id` / `parent_branch_id` kwargs — the kwargs path is bit-exact even under a poisoned contextvar, and is the only path the CUDA graph supports.
- `FoldingScheduler`-managed layers and `AdaptiveQuantileGate` are **not** graph-capturable (data-dependent host decisions); they degrade to eager with a one-time warning.

#### Known follow-ups

- Kernel-launch counts for the bench artifact require a CUPTI-capable torch build; the formal folded-vs-no-folding baseline comparison per `docs/RERUN_CHECKLIST.md` (locked clocks) remains open.
- Variable-length folding (P3) is out of scope: graph capacity is fixed at capture time (`graph_capacity_ratio`); longer divergent sets fall back to eager.

### AR001: Deep Optimization (M1 correctness / M2 sync / M3 memory / M5 methodology & portability)

Spec: `specs/changes/AR001-deep-optimization/` (srs.md / design.md / tasks.md, T001–T026 all passing).

#### Fixed (M1 — correctness, T001–T008/T027)

- Cross-step `stable_ratio` now averages over **all** diffusion steps (was last-step-only), with `final_norm` applied before the similarity comparison.
- Profiling a model that exposes **random head selection** now raises `RuntimeError` instead of silently profiling random layers; the stability profiler resets branches in `try/finally`.
- `NaN/Inf` guard moved **before** the hot path (fail fast instead of after expensive work); cosine similarity computed in **fp32**; CPU timing uses `time.perf_counter` (was `time.time`).
- Activation-cache growth is bounded: ring eviction keyed by `(branch, step)` with a `max_branch_steps` cap, so `branch_id` chains cannot grow memory without bound.
- `FoldedModel` restores the base model's original layers on exceptions (context-manager safety); draft distribution supports both `suffix_append` and `logits_draft` modes.

#### Improved (M2 — host-sync elimination, T009–T013)

- Stability profiler readback is lazy (no per-layer host sync); the folded layer's three-way stable/divergent split performs a **single** host sync via one `sum` readback.
- `AdaptiveQuantileGate` bottom-k selection is sync-free on the adaptive path.
- Legacy `ActivationCache` rewritten with contiguous per-`(branch, step)` group buffers: group-level LRU eviction under a per-layer token budget, ring-layout wrap for over-budget groups, and vectorized put/get (12–183x faster puts, 30–420x faster gets than the per-token version).
- Triton merge kernel: single-sided reads, correct strides, and tail masking; PyTorch fallback retained and bit-exact.

#### Added / Changed (M3 — memory-pass compression, T014–T018)

- `actfold/core/cache_protocol.py`: `ActivationCache` protocol (`fetch` / `get_all` / `contains` / `fetch_masked`) with three conforming implementations (legacy, chunked, vectorized); contract tests enforce the read-only zero-filled-view semantics.
- Redundant per-layer `hidden_states` cache entries removed (gate reads parent `ffn_out` at `L-1`, layer 0 reads the cached `embedding`).
- `fetch_flat` + fused `gather_select` path with a D3 threshold (T≥2048 & H≥4096, or T≥8192); PyTorch fallback for small shapes.
- `SplitFoldedTransformerLayer` uses resident module hooks + preallocated empty baseboard (no per-call hook registration).
- `get_num_transfer_tokens` fully vectorized — **bit-exact** against the official LLaDA implementation (87/87 cases); `LLaDASampler` / `DreamSampler` vectorized (per-prompt lengths, batched top-k, early stop), fixing a step-loop variable-shadowing bug.

#### Added (M5 — methodology & portability, T019–T025)

- `scripts/_hf_env.py`: `apply_hf_env` / `add_hf_env_arguments` — `--hf-endpoint` / `--hf-home` CLI flags with environment-variable fallback; **no hardcoded mirror or platform paths** (7 scripts migrated). All scripts run via `python -m scripts.<name>` (9 `sys.path` hacks removed).
- `TimingStats` (frozen dataclass: `mean/std/p50/n/as_dict`) + `stats_from_samples`; `repeat_with_seed` (per-repeat `manual_seed` with global-RNG snapshot/restore); `exp_sampling(repeats=3, seed=0)` with a `tokens_reproducible` check.
- `HardwareProfile.from_device(device, calibrate=False)`: device-name lookup table (H100/A100/RTX PRO 6000/4090/…, conservative fallback) plus optional CUDA microbenchmark calibration; cost model gains the attention quadratic term and KV-read bandwidth.
- `count_diffusion_llm_flops`: `ffn_intermediate_dim` / `ffn_type` (`mlp`|`swiglu`) / `include_attention_t2` parameters; `model_ffn_flops_kwargs` duck-typed extraction wired into all call sites.
- `ActFoldConfig.max_new_tokens` (default **256**, validated > 0), threaded through `BenchmarkRunner` into all eval adapters.
- `FoldedMeasurement` + `AblationStudy.measure_folding(tau, disabled_layers, max_entries_per_layer, seed)`: real per-layer measurements via a context-managed `FoldedModel` and the stability profiler.
- `docs/RERUN_CHECKLIST.md` + `results/**/INVALIDATED.md`: historical artifacts marked invalid with rerun instructions (environment, VRAM, commands, methodology requirements, and the list of numbers that changed).
- 82 new tests across `test_scripts_portability.py` (12), `test_experiment_stats.py` (16), `test_cost_model_device.py` (14), `test_flops_counter_ffn.py` (11), `test_eval_fixes.py` (20), `test_ablation_measured.py` (9).

#### Breaking changes

- `time_forward` returns a `TimingStats` object (callers use `.mean`); `exp_sampling` results use `baseline_ms_mean/std` + `folded_ms_mean/std` + `repeats` + `seed` + `tokens_reproducible` (old single-scalar keys removed).
- `LayerCost` fields renamed: `gate_flops`/`merge_flops` → `gate_bytes`/`merge_bytes` (bandwidth-bound, not compute-bound).
- FLOPs accounting: the **input** embedding is a table lookup and no longer counted; only the LM-head projection is. All FLOPs-reduction percentages shift up (synthetic demo baseline 78.5% → **85.5%**).
- Eval adapters: default `max_new_tokens` 16 → **256**; `BaseEvalAdapter._generate_one` signature is now `(prompt_tokens, use_actfold, seed) -> (text, ratio, latency_ms)`; `_generate_predictions` accepts pre-tokenized tensors; `_evaluate` adds `baseline_latency_ms` / `actfold_latency_ms` and canonical metric keys (`exact_match` for gsm8k/math, `prompt_level_acc` for ifeval via `LMEvalAdapter._TASK_METRIC_KEYS`); `LMEvalJudge.score` additionally returns per-doc `"metrics"` aggregates (`judges._aggregate_metric_means`).
- `AblationStudy`: layerwise results are **measured** via `FoldingScheduler.disabled_layers` (the linear extrapolation is kept only as the `linear_estimate_pct` comparison column); the cache sweep defaults to `[seq_len, 2*seq_len, 4*seq_len]` so the smallest budget forces real eviction; `draft_generator` is optional and defaults to `suffix_append`; `estimated_reduction_pct` is now a measured value. The study manages its own folded stack (the adapter must NOT carry a `folded_model` and must expose `underlying_model`).

#### Known follow-ups (AR002)

- M4 (single-kernel fusion / CUDA-graph capture) and the full B12 refactor are deferred to AR002.
- M-5 (controlled before/after timing) requires the rerun on the target GPU per `docs/RERUN_CHECKLIST.md`; `DraftGenerator.generate(seed=...)` still seeds the global RNG (measurement helpers snapshot/restore it).

#### Transport note (Gitee mirror)

- The oversized experiment figure PNGs under `figures/experiments/` and `results/*/figures/` (all pre-AR001, `INVALIDATED.md`-marked artifacts due for regeneration per `docs/RERUN_CHECKLIST.md`) were re-encoded with 256-color palette quantization (dimensions and visual content unchanged; 104–222 KB → 28–54 KB each). Reason: the corporate egress proxy blocks HTTPS request bodies above ~98 KB, so the original files cannot transit to the Gitee mirror from this network. All code, text, and data files are byte-exact.

### Added (pre-AR001)

- `constraints-bench.txt`: pins the `evalplus -> google-generativeai` gRPC/protobuf transitive stack (`grpcio`, `grpcio-status`, `googleapis-common-protos`, `proto-plus`, `protobuf`) so `pip install -r requirements-bench.txt` resolves in seconds instead of backtracking for minutes. `requirements-bench.txt` includes it via `-c`.
- CI is split into a fast `quality` job (runtime + dev dependencies only, Python 3.10/3.11 matrix, formatting/lint/mypy/fast tests/demo) and a constrained `bench-backends` job (bench backends + `-m slow` tests); both jobs have timeouts and pip caching.

- `actfold/profiler/stability_profiler.py`: Layer-Aware Stability Profiler (LASP) that records real per-layer, per-step stable ratios from `FoldedTransformerLayer`.
- `actfold/core/chunked_cache.py`: `ChunkedActivationCache`, a drop-in memory-efficient replacement for `ActivationCache` that stores activations in contiguous tensor chunks.
- `actfold/core/cache_factory.py`: `make_activation_cache` factory to switch between legacy and chunked caches via config.
- `actfold/utils/cost_model.py`: Compute-Bandwidth-Aware FLOPs Model (CBAF) that estimates wall-clock latency from compute throughput and memory bandwidth.
- `actfold/speculative/folded_generation.py`: True End-to-End Folded Generation (TEFG) engine where each new token is produced through a folded child forward pass.
- `actfold/speculative/branch_tree.py` and `acceptance_policy.py`: tree and policy helpers for folded generation.
- `actfold/speculative/adaptive_draft_controller.py`: Adaptive Draft-Growth Controller (ADGC) that varies the number of draft branches based on runtime stability and acceptance history.
- `actfold/models/diffusion_sampler.py`: abstract base for diffusion-native samplers with cross-timestep Branch Folding.
- `actfold/models/llada_sampler.py`, `dream_sampler.py`, `fast_dllm_sampler.py`: reference diffusion samplers for LLaDA, Dream, and Fast-dLLM.
- `ActFoldConfig` advanced switches: `use_stability_profiler`, `use_chunked_cache`, `cache_chunk_size`, `use_cost_model`, `use_folded_generation`, `max_active_branches`, `min_active_branches`, `use_adaptive_draft_growth`, `min_stable_ratio_to_expand`, `diffusion_sampler`.
- `ActFoldVerificationEngine` now reports `estimated_latency_ms` and a full `StabilityProfile` when a folded model is used.
- `BaseEvalAdapter` uses `folded_generate` automatically when the wrapped adapter carries a `FoldedModel`, so benchmark predictions are produced through the real folded path.
- `DiffusionLLM.generate()` now supports an optional `folded_model` argument and dispatches to native samplers when `num_steps > 1`.
- `actfold/models/architecture_utils.py`: architecture-agnostic detection of embedding modules, Transformer layer stacks, and language modeling heads for GPT/LLaMA/Qwen/Mistral/Gemma/OPT/BERT/RoBERTa/T5/BART/Falcon/Phi-style models.
- `ManualFoldedForward`: fallback folded path used when `FoldedModel` cannot auto-discover a layer stack.
- Extended `FoldedModel` default layer paths to cover `model.layers`, `transformer.h`, `transformer.layers`, `gpt_neox.layers`, `transformer.blocks`, `model.decoder.layers`, `decoder.layers`, `encoder.layer`, `model.encoder.layer`, `bert.encoder.layer`, `decoder.block`, `model.decoder.block`, `encoder.block`, and `model.encoder.block`.
- `demo.py` real-model path is now architecture-agnostic and supports `--dtype`, `--seq-len`, `--num-branches`, `--tau`, `--prompt`, `--num-steps`, and `--max-new-tokens`.
- Tests for architecture detection (`tests/test_architecture_utils.py`).

### Improved

- **Diffusion samplers aligned with official recipes**:
  - Added `actfold/models/sampling_utils.py` with shared masking schedulers (`LinearMaskingScheduler`, `CosineMaskingScheduler`), `get_num_transfer_tokens`, Gumbel-Max noise, top-p/top-k filtering, canvas builders, and AR logit shifting.
  - Rewrote `LLaDASampler` to follow the official LLaDA/MDLM recipe: right-padded canvas, block-wise decoding, masking schedule, `low_confidence`/`random` remasking, CFG, temperature/top-p/top-k, and Gumbel-Max noise.
  - Rewrote `DreamSampler` to follow the official Dream recipe: left-padded canvas, MaskGIT-style iterative decoding with `maskgit_plus`/`topk_margin`/`entropy` confidence rules, optional CFG, and `alg_temp` soft selection.
  - Rewrote `FastDLLMSampler` to follow the Fast-dLLM v2 recipe: block-wise masked decoding, small-block threshold unmasking, top-p/temperature sampling, stop-token early termination, and autoregressive block extension.
  - `DiffusionSampler` base class now uses config dataclasses (`SamplerConfig`), returns `SamplerOutput`, supports `attention_mask`/`position_ids`, and forwards `folded_model` through every denoising step.
  - `LLaDAModel`, `DreamModel`, and `FastDLLMModel` accept a `sampler_config` kwarg and forward sampler kwargs to their native configs.
  - README, `docs/EXPERIMENTS.md`, and `AGENTS.md` updated to describe the new sampler capabilities and hyperparameters.

### Changed

- `ActFoldVerificationEngine._estimate_stable_ratio` now prefers the mean per-layer stable ratio from LASP over the embedding-level proxy.
- `BenchmarkRunner` constructs caches via `make_activation_cache`, respecting `use_chunked_cache` and `cache_chunk_size`.
- `BenchmarkRunner` passes a `ComputeBandwidthCostModel` to the verification engine when `use_cost_model` is enabled.
- `FoldedTransformerLayer` and `FoldedModel` now accept `ActivationCacheType` (legacy or chunked).
- `ActFoldVerificationEngine` now accepts `ActivationCacheType`.
- `CausalLMDiffusionLLM.generate()` delegates to the base class when `num_steps != 1` or a `folded_model` is supplied.
- `GenericDiffusionLLM.generate()` delegates to the base class implementation.
- `LLaDAModel`, `DreamModel`, and `FastDLLMModel` implement `get_native_sampler()` and delegate to `DiffusionLLM.generate()` for diffusion sampling.
- README, AGENTS.md, and this changelog updated to document the new components and configuration switches.

### Fixed

- `ActFoldVerificationEngine` no longer references `profile` before it is defined.
- `FoldedModel` type annotations now accept the union cache type.
- `actfold/core/fused_ops.py`: Triton 3.x compatibility. `typing.Any` annotations inside `@triton.jit` kernels are rejected by Triton 3.4 (bundled with PyTorch 2.8); pointer parameters are now unannotated and any kernel compile/launch failure permanently falls back to the PyTorch merge.
- `gather_cached_activations`: partial reuse now works when the first token's cache entry was evicted; the sample entry is taken from the first available key instead of requiring token 0.
- `CausalLMDiffusionLLM` / `GenericDiffusionLLM`: quantization kwargs are only passed to `from_pretrained` when enabled (transformers 5.x forwards `load_in_8bit=False` to remote-code model constructors); `GenericDiffusionLLM.forward` now accepts `logits`, `last_hidden_state`, or `hidden_states` outputs; both wrappers strip ActFold branch kwargs before calling raw models.
- `LLaDAModel` / `DreamModel` / `FastDLLMModel`: constructors now accept and forward `torch_dtype`, `use_fast_tokenizer`, `device_map`, and quantization flags, matching the base wrappers and `BenchmarkRunner`.
- `FoldedModel`: unwraps Hugging Face `ModelOutput` dataclasses and tuples to a tensor (logits / last hidden state / first element); detects whether the wrapped model accepts ActFold kwargs instead of relying on a caught `TypeError`.
- `FoldedTransformerLayer`: preserves the original layer's output arity. Layers such as LLaDA blocks that return `(hidden_states, cache)` now receive `(folded_hidden, None)` from the folded path so the base model keeps unpacking correctly.
- Architecture detection: added the LLaDA paths `model.transformer.blocks` (layers) and `model.transformer.wte` (embedding) to `FoldedModel` and `actfold/models/architecture_utils.py`.
- `SimilarityGate`: cosine similarity is clamped to `[-1, 1]` so numerical noise cannot make `sim > tau` true at `tau=1.0`.
- Diffusion samplers (`LLaDASampler`, `DreamSampler`, `FastDLLMSampler`): forward passes are chained via `parent_branch_id` so cross-step Branch Folding actually activates instead of always passing `None`; unified mask-token resolution across `tokenizer.mask_token_id`, the model config, and the common checkpoint spellings (`<|mdm_mask|>`, `|<MASK>|`, `<|mask|>`).
- `DreamSampler`: the left-padded canvas attention mask is normalized to `bool`, fixing a torch SDPA dtype error (`attn_mask long` vs `query bfloat16`) for Dream checkpoints.
- `StabilityProfiler.all_profiles()` exposes all recorded branch profiles.
- Added `scripts/algo_experiments.py` (nine deep experiments), `scripts/overhead_bench.py` (per-layer overhead decomposition), and `scripts/make_experiment_figures.py` (explanatory figures).
- Added `actfold/core/vectorized_cache.py`: `VectorizedActivationCache`, a contiguous-buffer cache whose `put`/`get` are 12–183x / 30–420x faster than the per-token legacy cache; wired via `make_activation_cache(use_vectorized=...)`, `ActFoldConfig.use_vectorized_cache`, and `BenchmarkRunner`. The all-stable folded fast path drops from ~52 ms to ~6.5 ms (2.3–2.7x faster than the no-folding baseline).
- Added `actfold/core/split_layer.py`: `SplitFoldedTransformerLayer` (optimization #2). Attention stays on the full sequence while the token-wise FFN chain receives only divergent rows through temporary module hooks. Forced exactness tests and shape sweeps show 1.1–1.9x per-layer speedups once ``batch*seq >= 512``; controlled by `ActFoldConfig.use_split_layers` / `split_min_tokens`.
- Added `actfold/core/adaptive_gate.py`: `AdaptiveQuantileGate` (optimization #3) which top-k selects an exact target stable ratio per call and dominates the fixed-threshold fidelity/reuse frontier.
- Added `fused_ops.gather_select` (optimization #4): a Triton kernel fusing cached-parent gather and stable/divergent select (bit-exact, up to 3.4x at T=8192) with automatic PyTorch fallback.
- Added optimization experiment scripts: `scripts/opt1_cache_bench.py`, `opt2_split_bench.py`, `opt2_shape_bench.py`, `opt2_long_seq.py`, `opt3_adaptive_bench.py`, `opt4_fused_bench.py`, `make_optimization_figures.py`; tests: `tests/test_vectorized_cache.py`, `test_split_layer.py`, `test_adaptive_gate.py`.
- Fixed `ChunkedActivationCache.get`: it now returns full-length zero-filled tensors (was returning only the selected positions) and raises `KeyError` on empty caches, matching the `ActivationCache` contract.

## [Previous Releases]

### Added

- `actfold/core/model_wrapper.py`: high-level `FoldedModel` for wrapping existing models with Branch Folding.
- `actfold/configs/__init__.py` and `actfold/configs/per_model/__init__.py` so YAML configs ship with the package.
- `requirements-dev.txt` and `requirements-bench.txt` for clearer dependency separation.
- Optional `bench` extras in `pyproject.toml` for `lm-eval` and `evalplus`.
- CI workflow at `.github/workflows/ci.yml` running format, import, lint, type, and test checks.
- New unit tests for `config_manager`, `flops_counter`, `gpu_profiler`, `logger`, `fast_dllm_adapter`, `draft_generator`, `folding_scheduler`, `FoldedModel`, and `BaseEvalAdapter`.
- Added `@pytest.mark.slow` for tests that exercise real `lm-eval` / `evalplus` backends so the default test suite finishes quickly on CI and local development machines.
- `AGENTS.md`, `CHANGELOG.md`, and `CONTRIBUTING.md` documentation.
- `actfold/eval/judges.py`: unified `Judge` abstraction with real `lm-eval` / `evalplus` backends. Mock judges have been removed; evaluation always uses real backends.
- `actfold/eval/generation_utils.py`: shared prompt encoding / token decoding helpers for benchmark adapters.
- `tests/test_judges.py`: unit tests for the real judge factory and real judges.
- New `ActFoldConfig` fields: `torch_dtype`, `device_map`, `use_real_eval`, `eval_backend`, `eval_batch_size`, `eval_num_fewshot`, `eval_limit`, `eval_base_only`.
- `actfold/core/fused_ops.py`: optional Triton kernel for stable/divergent token fusion, with automatic PyTorch fallback on CPU or when Triton is absent.
- `tests/test_fused_ops.py`: unit tests for the fused merge, cache gather, and Triton/PyTorch fallback equivalence.
- `DiffusionLLMAdapter.embed()` and `FastDLLMAdapter.embed()`: real embedding lookup for verification engine cache population.
- `DiffusionLLM.embed()`: added as an abstract method on the base class; implemented in `CausalLMDiffusionLLM` and `GenericDiffusionLLM` via Hugging Face `get_input_embeddings()`.
- Tests for tuple-output layers, CPU-mask/CUDA-tensor merge, `DiffusionLLM.embed`, raw-model embedding lookup, verification-engine threshold validation, parent-cache embedding storage, and folded-model verification path.
- `FoldingScheduler.disabled_layers`: per-layer folding enable/disable support.
- `FoldedTransformerLayer` and `FoldedModel` now accept an optional `scheduler` and `step_idx` to respect folding decisions per layer/step.
- `actfold/core/folding_context.py`: thread-local `contextvars.ContextVar` for propagating branch identifiers through base models that do not forward kwargs.
- `BaseEvalAdapter` and adapters now accept `max_new_tokens` to generate completions of configurable length.
- Tests for `BranchManager` partial pruning, `ActivationCache` validation, `FoldedTransformerLayer` divergent-only/scheduler paths, `FoldedModel` context propagation, and `FastDLLMAdapter` wrapping a `DiffusionLLM`.

### Changed

- `FoldedTransformerLayer` now recomputes divergent tokens using full child hidden states to preserve self-attention context.
- `FoldingScheduler.should_fold` now correctly disables folding at the last layer and last diffusion step, matching its docstring.
- `CausalLMDiffusionLLM` and `GenericDiffusionLLM` now default to `torch.float32` and accept `torch_dtype` / `device_map` / `load_in_8bit` / `load_in_4bit` arguments.
- `BenchmarkRunner` no longer calls `.to(device)` when quantized loading (`load_in_8bit` / `load_in_4bit`) is enabled.
- `BenchmarkRunner` now loads prompts from the judge, generates text completions, and scores them through real backends.
- `LMEvalAdapter` and `EvalPlusAdapter` refactored to share common generation, scoring, and TFLOPs estimation logic through `BaseEvalAdapter`.
- `ActFoldVerificationEngine` now accepts an `acceptance_threshold`; branches below the threshold are rejected and evicted from cache.
- `load_config()` now emits a `UserWarning` for unknown YAML keys instead of silently dropping them.
- `demo.py` clearly labels the real-model path as a structural demonstration.
- README and `docs/EXPERIMENTS.md` updated to reflect that only real evaluation backends are supported and how to run slow backend tests.
- `AGENTS.md` updated to document the no-fallback tokenizer policy, the `slow` test marker, and the `BaseEvalAdapter` refactor.
- `ActivationCache.get` now uses a vectorized gather path for dense caches while preserving the legacy loop-based fallback for sparse caches.
- `ActivationCache.num_entries` and `core.branch_manager` now use `int | None` instead of `typing.Optional` for consistency with the rest of the codebase.
- `FoldedTransformerLayer.forward` now delegates the stable/divergent merge to `merge_stable_divergent`, replacing `nonzero` scatter with a fused select.
- `FoldedTransformerLayer._recompute_all` now handles tuple outputs from Hugging Face-style layers.
- `FoldedTransformerLayer` fast path (all tokens stable) now recomputes the whole layer when the cached parent FFN output is missing, avoiding an inconsistent slow-path fallback.
- `BenchmarkRunner` no longer silently builds a mock model when `model_name_or_path` is missing; it raises `ValueError`.
- `encode_prompt` no longer falls back to random tokens; it raises `RuntimeError` when no tokenizer is available.
- `LMEvalAdapter` and `EvalPlusAdapter` now estimate ActFold TFLOPs from the measured per-sample `stable_ratio` and the actual tokenized prompt length.
- `AblationStudy` replaces the hardcoded 0.7 stability assumption with a real full-model measurement.
- `ActFoldVerificationEngine` uses the model's real embedding layer, removes the synthetic depth-decay factor, and estimates TFLOPs from the actual sequence length, vocabulary size, and real head count (`model.num_heads`).
- `DraftGenerator.generate` now supports `max_new_tokens` and resets its internal counter when a seed is supplied for deterministic branch IDs.
- `SpiffyBaseline.generate` now respects `max_new_tokens` and forwards a `seed` to the draft generator.
- `demo.py` reports measured stable ratios and clearly labels the default run as a synthetic demonstration model; `--model` enables real-model experiments.
- `scripts/generate_figures.py` reads real benchmark/ablation artifacts; `--demo` generates example figures from a synthetic run.
- `scripts/run_ablation.sh` is now config-driven with a `--synthetic` debug mode.
- Default configs (`default.yaml`, `ablation_threshold.yaml`) now point to GPT-2 instead of `model_name_or_path: null`.

### Removed

- All mock evaluation logic (`MockLMEvalJudge`, `MockEvalPlusJudge`, mock fallbacks in `JudgeFactory`, and the ``"mock"`` `eval_backend` option).

### Fixed

- Removed unused imports and variables across `actfold/`, `tests/`, `demo.py`, and `scripts/`.
- Fixed `mypy` strict-mode errors in core, models, eval, and speculative modules.
- Fixed `DraftGenerator` "copy_flip" mode flipping at least one token even when `flip_ratio=0`.
- Fixed duplicate embedding allocation in `ActFoldVerificationEngine._token_to_hidden`.
- Fixed `LMEvalAdapter` and `EvalPlusAdapter` type annotations to accept `DiffusionLLMAdapter`.
- Fixed `LMEvalJudge.score` to extract the canonical primary metric instead of summing all numeric values in the lm-eval result dict.
- Fixed `SimilarityAnalyzer` and `SimilarityGate` L2 metric to avoid allocating a new `torch.tensor` on every forward call.
- Fixed `fused_ops._merge_stable_divergent_triton` to check `stable_mask.device`, defer `.contiguous()` until after the hidden-dim divisibility check, and added an `ActivationCacheDict` type alias.
- Fixed `FastDLLMAdapter.forward` to filter kwargs for raw `nn.Module` models based on their forward signature, preventing ActFold-specific arguments from breaking Hugging Face models.
- Fixed `FastDLLMAdapter.embed` to use `DiffusionLLM.embed()` directly and to raise a clearer error when no embedding layer is found.
- Fixed `ActFoldVerificationEngine._ensure_parent_cache` to store only `hidden_states` (not `ffn_out`) and clarified that layer-wise caches must be populated by the folded forward path.
- Fixed `FoldedModel.forward` to fall back to a normal forward if the base model rejects ActFold-specific kwargs and to only pass `attention_mask` when the wrapped model accepts it.
- Fixed `FastDLLMAdapter.forward` to filter ActFold-specific kwargs (`branch_id`, `parent_branch_id`, `step_idx`) when no `FoldedModel` is attached, keeping the adapter safe to call from the verification engine.
- Added optional `FastDLLMAdapter(..., folded_model=...)` support; when supplied, the verification engine runs the parent through the folded model to populate layer caches and passes branch identifiers during child verification.
- Lazy-imported evalplus inside `actfold/eval/judges.py` so that importing the module no longer requires evalplus to be installed when only `lm-eval` tasks are used.
- Removed dead `fallback_encode` from `actfold/eval/generation_utils.py`.
- Fixed `BenchmarkRunner` passing `load_in_8bit` / `load_in_4bit` to `load_model` constructors that previously did not accept them.
- Fixed `get_model_device()` to gracefully handle models whose `get_device()` raises `RuntimeError` because weights are not loaded.
- Fixed `FoldedModel.forward` fallback path silently dropping user kwargs; it now only strips the three ActFold-specific keys.
- Fixed `FoldedTransformerLayer` to read branch context from the thread-local folding context when the base model does not pass kwargs.
- Fixed `FoldedTransformerLayer` to align cached parent hidden states to the child's device/dtype before gating.
- Fixed `FoldedTransformerLayer` slow path to recompute the full layer when no tokens are stable, avoiding a missing-parent-FFN error.
- Fixed `FoldedTransformerLayer._recompute_all` to filter kwargs to the original layer's forward signature and drop unsupported `attention_mask`.
- Fixed `SimilarityGate` to validate 3-D inputs and align parent/child device/dtype.
- Fixed `ActivationCache.put` to reject empty activation dicts and inconsistent leading shapes.
- Fixed `merge_stable_divergent` to validate 3-D inputs and cast the mask to bool/device.
- Fixed `BranchManager.prune_rejected(include_subtree=False)` to reparent children to the deleted branch's parent instead of leaving dangling children.
- Fixed `ActFoldVerificationEngine._ensure_parent_cache` to create the probe mask on the embedding device.
- Fixed `BaseEvalAdapter` FLOPs estimation to use `self.model.num_heads` and to account for `max_new_tokens`.
- Fixed `AblationStudy` FLOPs estimation to use `self.model.num_heads`.
- Fixed `EvalPlusJudge` CLI fallback to invoke `python -m evalplus.evaluate`.
- Lazy-imported `lm_eval.tasks.TaskManager` inside `LMEvalJudge` methods so importing `actfold.eval.judges` does not require `lm-eval` unless it is used.
- Fixed `BenchmarkRunner` to wrap loaded models with `FoldedModel` when possible and to share the cache/gate/scheduler with the verification engine.
- Fixed `BenchmarkRunner` to skip `.to(device)` when `device_map` is configured and to validate the presence of a tokenizer early.
- Fixed CI and local test scripts to use `-m "not slow"` by default.
- Fixed `README.md` and `docs/EXPERIMENTS.md` command examples and synchronized them with the current code.

## [0.1.0] - 2025-01-01

### Added

- Initial release of ActFold with activation cache, similarity gate, folded Transformer layer, branch manager, folding scheduler, model registry, speculative verification engine, and mock benchmark adapters.
