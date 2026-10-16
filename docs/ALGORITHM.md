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
   static body (`FoldedGraphRunner`'s own layer loop) does NOT unwrap —
   unwrap it before any tuple-layer graph capture.

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

