# ActFold Algorithm

This document formalizes the Branch Folding mechanism at the heart of ActFold.

---

## 1. Notation

Consider a Diffusion LLM with `L` Transformer layers. At diffusion step `s`, let:

- `h_parent[l, t, s]` = hidden state of the parent branch at layer `l`, token `t`, step `s`.
- `h_child[l, t, s]` = hidden state of the child branch at the same position.
- `τ` = similarity threshold (default 0.95).

All hidden states are real-valued vectors of dimension `d`:

```
h_parent[l, t, s], h_child[l, t, s] ∈ ℝ^d
```

---

## 2. Similarity Score

For every `(l, t, s)` triplet, compute the cosine similarity between parent and child hidden states:

```
                    h_parent[l, t, s] · h_child[l, t, s]
sim(l, t, s) = ─────────────────────────────────────────────
               ||h_parent[l, t, s]|| · ||h_child[l, t, s]||
```

In batched form over the sequence dimension:

```python
sim = F.cosine_similarity(h_child, h_parent, dim=-1)  # [batch, seq_len]
```

---

## 3. Token-Level Gating

Partition tokens based on the configurable threshold `τ`:

```
stable(l, t, s) = 1  if sim(l, t, s) > τ
                = 0  otherwise
```

- **Stable tokens** (`stable = 1`): reuse parent's cached Attention and FFN outputs.
- **Divergent tokens** (`stable = 0`): perform full recomputation.

---

## 4. Folded Layer Forward Pass

For a single Transformer layer `l` at step `s`:

```
for each token t in {0, ..., T-1}:
    if stable(l, t, s):
        h_out[l, t, s] = cache.ffn_out[parent][l, t, s]
    else:
        h_out[l, t, s] = FullAttentionFFN(h_child[l, :, s])[l, t, s]

h_next[l, t, s] = LayerNorm(h_out[l, t, s] + residual[l, t, s])
```

Divergent tokens are recomputed using the **full** child sequence so that
self-attention context matches the baseline.  The final merge selects the
cached parent FFN output for stable tokens and the recomputed child output for
divergent tokens.

### Vectorized Implementation

```python
stable_mask = gate(h_child, h_parent)       # [batch, seq_len] bool

# Fast path: copy cached parent activations.
parent_ffn = cache.get(parent_id, layer_idx, stable_mask)["ffn_out"]

# Slow path: recompute divergent tokens with full child attention context,
# then fuse the two paths token-wise.
divergent_mask = ~stable_mask
if divergent_mask.any():
    child_out = full_attention_ffn(h_child)
    h_out = merge_stable_divergent(parent_ffn, child_out, stable_mask)
else:
    h_out = parent_ffn

return layer_norm(h_out + residual)
```

### Fused CUDA Kernel

When Triton is available and the tensors reside on CUDA, `merge_stable_divergent` launches a 1-D element-wise kernel over ``(batch, seq_len, hidden_dim)``. Each program instance loads the boolean stability mask for its token and selects either the cached parent FFN value or the freshly computed child value. This avoids:

1. The per-token Python loop in `ActivationCache.get`.
2. The `nonzero` + advanced-indexing scatter used to overwrite divergent positions.

On CPU or when Triton is not installed, a vectorized PyTorch fallback (`torch.where`) is used transparently.

---

## 5. Activation Cache

The cache stores per-token activations for each branch and layer:

- **Key**: `(branch_id, layer_idx, token_idx, step_idx)`
- **Value**: dictionary `{"hidden_states": Tensor, "ffn_out": Tensor}`. The
  `hidden_states` key stores the layer input (used for similarity gating), and
  `ffn_out` stores the layer output (reused for stable tokens).

The cache is populated from **real** parent forward passes.  The speculative
verification engine stores input embeddings at layer 0 for stable-ratio
estimation; layer-wise FFN outputs must be populated by a folded model forward
path (e.g. `FoldedModel`) before child branches are verified.  Synthetic random
embedding matrices are never inserted as real activations.

Eviction policy:

- Each layer maintains its own LRU cache.
- Maximum entries per layer is configurable (`max_entries_per_layer`).
- When a branch is rejected, all its entries are evicted immediately.

---

## 6. Dynamic Threshold Scheduling

The threshold `τ` can be adjusted per `(layer, step, task)`:

```
τ(l, s, task) = τ_base + bias_layer(l) + bias_step(s) + bias_task(task)
```

Heuristics:

- **Early layers**: higher `τ` (more reuse), because hidden states change more slowly.
- **Early diffusion steps**: lower `τ` (more conservative), because the model is less certain.
- **Math tasks**: lower `τ` (higher accuracy requirement).
- **Code tasks**: higher `τ` (more tolerance for reuse).

The final threshold is clamped to `[0.80, 0.99]`.

In the current implementation `FoldingScheduler.get_tau(layer, step, task)` is
available for per-layer/step/task thresholds, and `FoldedTransformerLayer`
applies the scheduler's tau before calling the similarity gate. The scheduler
also exposes `should_fold(layer, step)` to disable folding for specific layers
or steps.

---

## 7. Complexity Analysis

Let `T` be the sequence length and `R` the stable-token ratio.

### Baseline Verification

```
FLOPs_baseline = L · T · (F_attention + F_ffn)
```

### ActFold Verification

```
FLOPs_actfold = L · T · (1 - R) · (F_attention + F_ffn) + L · T · R · O(1)
```

The `O(1)` term corresponds to cache lookups and similarity computation, negligible compared to Attention/FFN.

### Expected Reduction

```
                FLOPs_baseline - FLOPs_actfold
reduction = ───────────────────────────────────── ≈ R
                   FLOPs_baseline
```

Empirically, `R` ranges from 0.21 to 0.62, matching the target TFLOPs reduction.

---

## 8. Correctness Guarantees

ActFold does **not** guarantee bit-exact equivalence with the baseline. However:

1. The similarity gate ensures that only sufficiently similar tokens reuse activations.
2. The threshold `τ` trades off speed vs. accuracy; higher `τ` is more conservative.
3. Output equivalence is measured by MSE between baseline and folded outputs.

For tokens below `τ`, full recomputation is performed, preserving correctness for the most divergent positions.


## 9. Branch Context Propagation

For standard Transformer stacks, `FoldedModel` passes `branch_id`,
`parent_branch_id`, and `step_idx` to the base model's `forward`. Because most
base models do not forward arbitrary kwargs to their layers, `FoldedModel` also
pushes these identifiers into a thread-local `contextvars.ContextVar` for the
duration of the forward pass. Each `FoldedTransformerLayer` reads the context
when the identifiers are not supplied as explicit keyword arguments. If the base
model rejects the ActFold kwargs, the wrapper falls back to a normal forward
while the context remains active, so folding still occurs for supported
architectures.

> **Deprecation (AR002)**: the contextvar path is deprecated together with
> `FoldedModel`. The supported `ManualFoldedForward` path threads `branch_id` /
> `parent_branch_id` / `step_idx` as explicit keyword arguments and is
> bit-exact even when a stale contextvar is active — explicit kwargs are also
> the form the CUDA-graph capture below requires.

## 10. Fixed-Shape CUDA-Graph Verification Loop (AR002)

The diffusion verification phase repeats the *same-shape* folded child forward
every step, which is the canonical CUDA-graph workload. `ManualFoldedForward(
use_cuda_graph=True)` (opt-in; `ActFoldConfig.use_cuda_graph /
graph_capacity_ratio`) captures the folded child forward on the first eligible
step via `actfold.core.cuda_graph.FoldedGraphRunner`:

1. **Static buffers** — tokens, per-layer parent activations (slot 0 =
   embedding), attention masks, counts, and child outputs are preallocated
   once; replay copies the new inputs in and replays the captured graph.
2. **Sync-free capture** — the captured region contains no cache writes, no
   profiler hooks, and no host readbacks; the exact divergent set is replaced
   by a fixed-capacity padded gather (§4, P2-2) so no `nonzero()` sync exists
   inside the graph.
3. **Budget validation** — after each replay, per-layer divergent counts are
   read back exactly once; `D == C` (capacity) validates and `D == C + 1`
   rejects, discarding the replay output and falling back to the eager folded
   forward for that step (one-time warning).
4. **Degradation matrix** — scheduler-managed layers, non-cosine gates, CPU
   tensors, shape changes, a different attention-mask object, or an incomplete
   parent cache each degrade that step to the eager path (one-time warnings
   where user-visible); capture preconditions raise `RuntimeError` instead.

Measured on a Quadro RTX 5000 (batch=2, seq=512, 4 layers), graph replay cuts
the folded child step from 5.083 ms to 2.687 ms (**-47.1%**) with all budget
validations passing — see `scripts/ar002_graph_bench.py` and the artifact
`results/optimization/ar002_graph_bench.json`.

## 11. Variable-Length Prefix Folding (AR003)

The equal-length contract above is the `T_p == T_c` special case of a more
general **append-only** rule: a parent cached at `T_p ≤ T_c` (same batch and
hidden size) can donate activations for the child's *prefix*.

1. **Prefix classification** — after fetching the parent gate activation, a
   batch/hidden mismatch or a zero-length entry treats the parent as absent
   (full recompute, cache-miss semantics, never an error). A parent cached at
   `T_p > T_c` is **truncated** to the child length (AR007, §12).
2. **Prefix gate + explicit False suffix** — the similarity gate compares
   only `child[:, :T_p]` against the parent; the full mask is
   `cat([prefix_mask, all-False suffix])`. The suffix has no parent activation
   to reuse, so it is divergent for every τ — an explicit False tail stays
   robust even for negative thresholds (a zero-padded parent would instead
   compare cosine 0 > τ as "stable" and copy zeros). The fused gate kernel and
   `gather_select` remain equal-length-only (their same-shape contracts).
3. **Three-way split** — all-stable is unreachable on var-len steps by
   construction (`stable_count ≤ B·T_p < B·T_c`), so the fast path stays an
   equal-length property; zero-stable recomputes; the mixed path recomputes
   the full child (attention context intact, §4) and merges with the parent
   FFN aligned to the child length: `cat([parent_ffn, child_out[:, T_p:]])`
   (the suffix fill never surfaces — the merge takes the child value at every
   suffix position).
4. **Chain recursion** — the child stores its activations at `T_c`, so the
   next step (`T_c → T_c+1`) prefix-folds against it.

For causal models the prefix is bit-identical to the parent's hidden states,
so `folded_generate` folds at every step (prefix all-stable, only the new
token recomputes) while remaining `torch.equal` to the eager full recompute
(measured mean stable ratio ≈ 0.84 over 4 growth steps, T 4→8). Variable-
length steps never enter the CUDA graph (§10): the parent-cache shape check
routes them to a silent eager fallback where prefix folding applies.

## 12. Truncated-Prefix Reuse for Longer Parents (AR007)

The symmetric completion of §11: a parent cached at `T_p > T_c` (same batch
and hidden size) donates its **first `T_c` positions**.

1. **Truncation** — the gate activation `h_parent` is sliced to
   `h_parent[:, :T_c]` and folds through the existing equal-length branches
   (`T_p == T_c` semantics after the slice); only batch/hidden mismatches and
   zero-length entries still mean no-parent.
2. **Reuse vs. safety net** — a *true* prefix of a causal model is
   bit-identical position-wise, so the gate certifies every position stable
   and the parent FFN output is reused after the same truncation
   (`torch.equal` to the eager recompute, zero `_recompute_all` calls). A
   non-prefix child diverges at the gate and recomputes — the similarity
   gate is the semantic safety net, so truncation never requires a true
   prefix for correctness. The fast path truncates by the fetched shape, so
   per-layer cache entries that disagree in length resolve deterministically.
3. **Merge path** — `gather_select` is conservatively skipped (its flat-row
   layout contract covers exactly `seq_len` stored tokens; the skip is
   bit-identical, one less fused kernel) and the parent FFN is fetched,
   truncated, and merged. `_get_parent_ffn_output` accepts a longer-than-mask
   FFN **only on the equal-length contract**; the §11 varlen exact-shape
   contract is unchanged.
4. **Graph path** — a longer parent fails the `_parent_cache_complete` shape
   check and silently runs eager (the same route as §11 variable-length
   steps): no capture, no warning, no disable; a later same-shape child with
   a complete parent cache still captures.

## 13. Speculative Block Commit and Cross-Family Generalization (AR021–AR023)

Diffusion samplers commit decode progress in *units* — a Fast-dLLM small
block, a LLaDA block, a Dream super-step window. Instead of running the
reference loop unit by unit, the sampler may build an `[N, T]` **speculative
canvas**: N candidate fills of the unit's still-masked positions, verified by
**one batched folded forward** (the batched-branch contract: all rows share
one parent cached at batch 1, §9), then committed by a single selection rule.

1. **Deterministic candidate construction** — each random row `j` is filled
   by a uniform `torch.randint` over the draft top-k index set (`k_eff`
   entries, gathered from `draft_set`) on a **dedicated CPU generator**
   seeded with the frozen formula
   `(S0·1_000_003 + unit_idx·8_191 + sub_idx·131 + j) & 0x7FFF_FFFF`
   (host integers, reproducible — never Python `hash`); construction reads
   device logits but never reads back device state. **Greedy anchoring**
   (AR022) pins row 0 to the greedy hypothesis (the draft top-k ordered-set
   first entry, the frozen argmax definition) built by pure device indexing —
   deterministic, zero generator consumption; `num_block_candidates` stays
   the TOTAL row count (greedy takes slot 0, random rows become `j = 1..N-1`
   and keep their row number in the seed formula).
2. **Selection = min-prob, no special cases** — each row is scored by the
   **minimum token probability over the fill positions only** (the worst
   position is the row's quality bottleneck) and the argmax row wins. The
   greedy row competes under the unchanged rule and CAN LOSE — a posterior
   trap fixture (zero-mixing layer + orthogonal embeddings + seed replay)
   constructs an all-C canvas where a random row beats it, so selection never
   special-cases row 0.
3. **Commit and chain continuation** — the winning row is committed; the
   unit's remaining remasking iterations are skipped entirely; the folding
   chain switches `_folding_parent` to the winner's branch id (its
   activations were published at batch 1), so subsequent units fold against
   the winner. Stop-token semantics mirror the family reference (a winning
   row that hits EOS finishes the sequence exactly as the reference would).
4. **The shared core (AR023)** — `SpeculativeCore`
   (`actfold/models/speculative_core.py`) is a mixin with **no `__init__`**
   (the host sampler owns config and counters) carrying `SpecCtx(mask,
   position)`, `_speculative_commit`, the `(T, prompt_len)`-keyed fold-ctx
   LRU (cap 32, host metadata, zero device readback), and the read-only
   `greedy_win_stats` probe. **Family adapters own only canvas/payload
   semantics**: LLaDA (`LLaDASampler`) uses one speculative unit per block on
   the right-padded canvas (ctx mask `(arange(T) < valid_end)` tiled
   `[T, T]`, `position=None` — LLaDA carries no rotary cargo); Dream
   (`DreamSampler`) uses super-step windows of `speculative_steps` (default
   8) over the transfer schedule with budget merge `sum[step:step+s]`,
   `k_eff = min(budget, masked_count)` (≤0 skips the window), and
   confidence-topk position selection. The logits shift convention follows
   the family reference defaults (Dream shifts / LLaDA does not / the core
   is shift-agnostic).
5. **`position_ids` threading (AR019 template)** — `DiffusionSampler._forward`
   attaches `position_ids` to the folded branch **only when non-`None`** (no
   kwarg otherwise — existing folded callers byte-for-byte unchanged), and
   `ManualFoldedForward.forward` threads it through all four construction
   faces: eager layer kwargs (signature-filtered), serial/batched graph
   capture (`position_ids_static`), pool replay (`is` identity validation),
   and the AR017 adaptive re-capture. On a batch-1 no-pad canvas
   `compute_position_ids(mask)` is value-identical to the model's internal
   default, so the N=0 reference loop is byte-untouched.
6. **Tuple-returning remote blocks** — the real LLaDA remote blocks return
   `(hidden, ...)` tuples; `FoldedTransformerLayer._pack_output` re-wraps
   `(hidden, None)` for the in-place-FoldedModel control flow, and MFF's
   eager layer loop consumes the first element (the `_recompute_all`
   convention). Fast-dLLM's remote layers return bare tensors, which is why
   the tuple face surfaced only at AR023's cross-family smoke. The graph
   static body (`FoldedGraphRunner`'s own layer loop) **does** unwrap —
   it calls `_recompute_all`, which inherits the `_pack_output` first-element
   convention (AR026 correction; pinned by UT-2603 on CPU and UT-2604 on
   CUDA).

Default-off (`num_block_candidates=0`) is byte-identical to the reference
samplers. Empirical face: on Fast-dLLM-v2-1.5B the greedy-anchored mode
restored baseline quality (posthoc gap 0.0, decode lengths and merged block
exact-match fully recovered) at p50 **7.02×** vs baseline (AR022 artifact
`results/optimization/ar022_speculative_quality_bench.json`, H1–H4 all
supported); cross-family LLaDA-8B/Dream-7B verdicts are pre-committed
reps=30 single runs in `results/optimization/ar023_{llada,dream}_bench.json`
(AR023). The greedy-anchored quality recovery is **family-dependent** —
see the AR023 artifacts for the per-family hierarchical quality faces
(Dream's gradual super-step denoising preserves the quality face far better
than LLaDA's whole-block one-shot commit).


## 14. Bounded-Memory Sentinel Cache (AR029)

A sentinel cache keeps a *subset* of a branch's activations and represents every
missing token with `NaN` plus an explicit availability bitmap:

```
availability(parent, L, T, step=0) -> Bool[T]
fetch(parent, L) -> {name: Tensor[B, T, H]}   # NaN where unavailable
```

`PyramidActivationCache` realizes this with block pages and **per-layer**
budgets: a plan-wide `total_tokens` is normalized into layer weights
(`linear` / `geometric`), floored to `block_size`, and reclaimed from the
deepest layer until the sum fits. Eviction walks a per-layer global write
order, protecting the deterministic anchors (`retained_blocks`) and the
just-written entry's recent blocks; only when the hard bound still requires it
does the oldest protected block get reclaimed. An entry with no blocks left is
deleted.

The folding forward is safe under this cache because of one ordering rule.
For layer `L`, the gate reads the parent's layer `L-1` `ffn_out` (layer 0 reads
the parent `embedding`), while the merge consumes the parent's layer `L`
`ffn_out`. With per-layer budgets, `L-1` can be resident while `L` is missing:

```
stable_mask = gate(h_child, h_parent[L-1])        # gate-side view
stable_mask &= ~forced_divergent                  # AR021 ordering
stable_mask &= availability(parent, L)            # AR029 target-layer guard
profiler.record(stable_mask)                      # then counts/fast paths
```

Applying the target-layer availability **after** the forced mask and **before**
the profiler/count/fast paths guarantees every missing target row recomputes;
a 1-D per-row mask stays 1-D (never expanded to `[T, T]`), and the fused gate's
cached count is discarded so `stable_mask.sum()` re-derives it. Legacy caches
without `availability` are a byte-for-byte no-op.

CUDA graph and sentinel caches are mutually exclusive: a captured static parent
buffer cannot observe runtime evictions. `ManualFoldedForward` rejects the
combination at construction and in the `cache` setter (validation precedes
propagation). The mechanism benchmark checks two invariant faces on CPU — the
full-budget pyramid against `ActivationCache` (bit-exact) and a genuine pyramid
against a per-layer NaN oracle (bit-exact) — and reports `E1`-`E4` with
`evidence_level="mechanism"` and `baseline_claim=false`; a real-checkpoint A/B
is deferred and requires a new `run_formal_queue` FaceSpec.

## 15. Paged Shared Fold Memory (AR030)

For Agent/RSI multi-candidate rollouts, N candidates share one prompt prefix
and each edits only the last few tokens. A dense cache stores whole-branch
tensors — bytes grow as O(N·T·L). PSFM stores activations as **pages** behind
a page table with per-page refcounts, so bytes grow with the *diff pages*
(O(N·diff_pages)); the mechanism bench measured page/byte sharing 0.778
(32 physical pages vs 144 naive) at 8 candidates × T=128.

Data structure (`actfold/core/page_table.py`):

```
Page        { tensors, layer, tier: hot|cold, refcount, write_seq, page_id }
PageStore   { pages: id -> Page, hot set, cold set }   # refcount ledger
PageTable   { branch -> (layer -> [page_id, ...]) }    # logical -> physical
```

`fork(parent, child)` is an **O(1) metadata copy**: the child's layer tables
point at the same physical pages and each page's refcount increments. Release
decrements; a refcount of 0 reclaims the page immediately.

COW write path (`put` of a branch's layer, one logical page at a time):

```
old = page_table[branch][layer][slot]          # incumbent physical page
if torch.equal(new_tensors, old.tensors):
    reuse old            # shared page, refcount+1, zero copy
elif old.refcount == 1:
    old.replace_tensors(new_tensors)           # exclusive page: write in place
else:
    new = store.alloc(new_tensors)             # shared + different: fresh page
    release(old)                               # old reference dropped
page_table[branch][layer][slot] = chosen
enforce_budget(layer)                          # physical-page hard bound
epoch += 1
```

The whole `put` runs under a write transaction journal: any failure restores
the captured state atomically (the `_fail_after_staged` hook injects failures
for tests).

**Eviction selection** (budget pressure, per layer) sorts candidates by the
frozen key `(protected_any, refcount, write_seq, page_id)`: pages under rollout
hard protection are unmovable (exhausting them raises `RolloutBudgetError`),
shared pages (higher refcount) are evicted last, then oldest write, then page
id for determinism. Victims are **demoted to cold**, not dropped: cold pages
have their tensors offloaded, `availability=False`, and read back as NaN;
only an explicit `promote` restores them (demote/promote round-trips
bit-exactly). A shared cold page (refcount >= 2) is **never overwritten in
place** — the exclusive guard routes divergent writes to a fresh page, since an
in-place write would corrupt every co-holding branch. Missing pages force
recomputation through the same target-layer availability guard as AR029, so
outputs stay bit-exact against the dense oracle (E1: `max_abs_diff == 0.0`).

**Rollout transactions** wrap the page table in an Agent-facing API:
`fork_rollouts(base, candidates)` freezes the base branch; a candidate failure
rolls back **atomically** to the fork baseline; `close_*` unfreezes the base.
`enter_plan` opens a read-only window over a frozen `FoldPagePlan` — writes
inside the window are staged and atomically committed on exit; a stale plan
raises `PlanStaleError`. `snapshot`/`restore` is sha256-checked, works across
instances (empty target only), and sets `epoch = max+1`.

Complexity: `fork` O(pages) (metadata copy, no tensor copies — measured fork
byte deltas 0); `put` O(pages × compare) (one `torch.equal` per logical page);
`fetch` O(pages) (page-table lookup + concat).

The mechanism benchmark reports `E1`-`E4` with `evidence_level="mechanism"`
and `baseline_claim=false`; the real-checkpoint A/B is deferred and requires a
new `run_formal_queue` FaceSpec.

## 16. Layer-Adaptive Recompute Budget and Uncertainty-Ordered Validation (AR031)

The gating of Sections 3-4 decides per token, per layer, whether the parent FFN
output is reused — but nothing bounds the total recompute. AR031 adds an
explicit per-layer budget:

- **Windowed policy (A2)**: during warmup the global budget is split uniformly
  across layers; at window boundaries the split is re-estimated from
  accumulated per-layer stability utilities (floor + largest-remainder,
  surplus redistributed, zero-budget layers lifted). The accumulation is
  device-pure (`observe_layer` adds a device-side scalar; a cross-device
  observation raises instead of silently moving).
- **Exact-count cap (A1)**: exempt rows (already stable, forced-divergent, or
  NaN-scored) always recompute and never consume the budget; among the
  non-exempt divergent rows the most similar flip back to stable until the
  non-exempt recompute count fits `B_l`. The threshold is selected by a
  capture-safe gather on the graph face (no host readback).
- **Forced broadcast**: the production forced tensor is position-level `[T]`
  (shared by every speculative-canvas row); both cap choke points (the eager
  layer and the graph static body) broadcast it to the mask shape while the
  core cap keeps its strict same-shape validation.
- **Uncertainty-ordered validation**: candidate rows participate in
  descending confidence order with a margin / margin-gap early-stop; the
  winner rule (min-prob) is unchanged, so ordering affects only the
  early-stop face.

Mask order (frozen): gate scores/mask -> AND ~forced -> budget cap ->
availability guard -> profiler record -> count consumption.

Empirical face (Fast-dLLM-v2-1.5B, three arms, reps=30, pre-committed single
shot): quality and safety faces all supported (posthoc gap 0.0, decode lengths
restored, winner equivalence 100%, budget violations 0, reproduction 6/6
strict-equal rows) while the mechanism attribution is an honest refuted — full
vs adaptive-off 1.0004x (delta 1.15ms << 2-sigma 1464.68ms): at tau=0.95 with
forced active blocks the folding demand never reaches the 15% budget, so the
cap is non-binding. The mechanism is a safety/regularization guarantee
(bounded recompute at zero quality loss), not a speedup at this workload.

## 17. Verification-Logit Advance Reuse (AR032)

The speculative block loop of Sections 13-14 (AR021/AR022) spends three folded
forwards per block: the draft serial forward, the `[N, T]` batched
verification forward, and the block-advance serial forward whose only product
is a single token (`argmax(logits[:, -1])`). The verification logits are
write-only in AR-domain speculative decoding implementations (SpecInfer,
EAGLE) and, before AR032, here as well. AR032 makes them a first-class
inference source for the advance step (opt-in, default off):

- **Frozen next-token definition**:
  `next_token = argmax(right_shift_logits(batched_logits)[j_star, -1, :])`
  where `batched_logits` is strictly the **last small block's** verification
  forward. Only the last small block's winner row equals the post-commit
  canvas bit-for-bit (S>1: earlier small blocks' winner rows still carry the
  mask token in later positions, so a wrong-source stash necessarily flips
  the advance token — pinned by a content-conditioned `(token, position)`
  fixture with a save-first defect injection).
- **Constructive input equivalence**: after commit `x == cands[j_star]`
  (candidate fills are overwritten by the commit; every other region is
  same-source), so the winner-row tail distribution and the independent
  advance forward's `logits[:, -1]` are the same distribution — same input,
  same weights, same `(T, prompt_len)` fold-context objects. Under full
  recompute (tau=1.0) the on/off A/B is bit-exact and a divergence is a
  mechanism bug; under real folding, two ULP drift faces are documented
  properties, not errors: the numerical source switch (verification vs
  serial forward, different fold parents) near argmax ties, and downstream
  chain-shape changes (the on arm folds later forwards directly against the
  winner branch, the off arm through the AR branch).
- **Tail-row clone (D8)**: `_speculative_small_block` returns
  `(j_star, shifted[j_star, -1:, :].clone())` — a `[1, V]` clone, never a
  full-row view, because the caller-side stash would otherwise pin the whole
  `[N, T, V]` cat backing (~250-500MB transient VRAM on the real face).
- **Count contract**: on-mode eligible blocks make exactly
  `2·num_small_blocks` forwards (off: `+1`); the omitted count is exactly 1
  per block-advance-eligible block (`remaining > 0`; the final block never
  advances on either side). The on mode never mints a `fastdllm_ar_*`
  branch, so off-arm ar-branch counts are the frozen eligible-block probe
  and the identity `off_serial − on_serial == eligible_blocks` is a hard,
  independently-refutable gate.

Empirical face (Fast-dLLM-v2-1.5B, three arms — official baseline /
reuse-off (AR022-aligned N=4+greedy, eager) / reuse-on (= ② plus the single
switch) — reps=30, pre-committed single shot, artifact
`results/optimization/ar032_speculative_advance_bench.json`): overall
**supported** — H1 gap 0.0 with decode lengths [34, 66] on==off and
exact-match 1.0/1.0; H2 **7.9698×** vs baseline (26305.87ms vs 3300.68ms
p50); H3 **1.2875×** vs reuse-off (4249.76ms vs 3300.68ms) with the count
identity 4−2==2; H4 canary 3/3, per-arm ledger bounds, reproduction 6/6
strictly equal. Every eligible block converts three forwards into two at
identical quality — the advance forward's entire cost is reclaimed because
its single-token product was already computed by the verification forward.

