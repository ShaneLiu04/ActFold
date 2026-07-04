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
