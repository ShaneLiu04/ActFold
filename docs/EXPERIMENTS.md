# Experiments Guide

This document describes how to reproduce ActFold experiments, including hyperparameters, expected results, and command-line workflows.

---

## 1. Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install runtime dependencies
pip install -r requirements.txt

# Install real benchmark backends (required for evaluation)
pip install -r requirements-bench.txt

# For development (tests, formatting, type checking)
pip install -r requirements-dev.txt

# Optional: Triton kernel acceleration for CUDA (Linux/WSL)
pip install triton>=2.0
```

> **Important:** ActFold does not use mock evaluation backends or silently
> fall back to synthetic models.  Benchmarking requires a real
> `model_name_or_path` and a real tokenizer.  The default configs use GPT-2 as
> a lightweight stand-in; set `model_name_or_path` to your target checkpoint
> for real Diffusion LLM results.

Set deterministic behavior in every script:

```python
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 2. Configuration Files

All experiments are driven by YAML configs in `actfold/configs/`:

| Config | Purpose |
|---|---|
| `default.yaml` | Default ActFold settings |
| `real_model_example.yaml` | Example for running with a real HF model |
| `ablation_threshold.yaml` | Threshold sensitivity study |
| `per_model/fast_dllm_v2_1.5b.yaml` | Fast-dLLM-v2 1.5B settings |
| `per_model/llada_8b.yaml` | LLaDA-8B settings |
| `per_model/dream_7b.yaml` | Dream-7B settings |

### Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `tau` | 0.95 | Similarity threshold for token gating |
| `metric` | "cosine" | Similarity metric |
| `max_entries_per_layer` | 1024 | Cache budget per layer |
| `enable_dynamic_tau` | false | Use FoldingScheduler |
| `model_name_or_path` | null | HF model identifier or local path |
| `model_family` | "auto" | Model family (llada, dream, fast_dllm, causal_lm, generic) |
| `load_in_8bit` | false | Load model in 8-bit mode |
| `load_in_4bit` | false | Load model in 4-bit mode |
| `torch_dtype` | null | Torch dtype for model weights: "float32", "float16", "bfloat16" |
| `device_map` | null | Hugging Face device map: "auto", "cuda:0", etc. |

### Evaluation Backend Parameters

| Parameter | Default | Description |
|---|---|---|
| `use_real_eval` | true | Must be ``True``; only real backends are supported |
| `eval_backend` | "auto" | "auto", "lm-eval", or "evalplus" |
| `eval_batch_size` | 1 | Batch size forwarded to lm-eval |
| `eval_num_fewshot` | null | Few-shot count; null uses task default |
| `eval_limit` | null | Max examples per task (int or float fraction) |
| `eval_base_only` | false | Use base-only tests for evalplus (ignore extra tests) |

---

## 3. Main Results

### 3.1 Baselines

| Baseline | Description |
|---|---|
| Standard Diffusion | No speculation, full diffusion steps |
| Spiffy | Multi-branch verification without activation reuse |
| **ActFold** | Full branch folding with cross-branch activation reuse |

### 3.2 Expected Results

| Model | TFLOPs Reduction | Accuracy Drop | Speedup |
|---|---|---|---|
| Fast-dLLM-v2-1.5B | 35-50% | ≤1% | 1.2-1.5x |
| Fast-dLLM-v2-7B | 40-55% | ≤1.5% | 1.3-1.6x |
| LLaDA-8B | 45-62% | ≤2% | 1.4-1.8x |
| Dream-7B | 38-52% | ≤1.5% | 1.3-1.6x |

---

## 4. Running Experiments

### 4.1 Demo (Synthetic Demonstration)

```bash
python demo.py
```

The default demo runs a small synthetic Transformer.  Stable ratios and FLOPs
reduction are measured from the actual parent/child hidden states of that
model.

### 4.2 Demo with a Real Model

```bash
# Use any causal LM as a stand-in (e.g., gpt2)
python demo.py --model gpt2 --model-family causal_lm

# Use a Diffusion LLM checkpoint
python demo.py --model "your-org/llada-8b" --model-family llada
```

> **Note:** The real-model folded path is currently wired for GPT2-like
> architectures.  Other architectures require equivalent model-specific
> integration.

### 4.3 Unit Tests

```bash
python -m pytest tests/ -v -m "not slow"
```

Tests that exercise the real `lm-eval` / `evalplus` backends are marked
`@pytest.mark.slow`. They are skipped by the default command above and can be
started separately when the benchmark dependencies are installed:

```bash
python -m pytest tests/ -v -m slow
```

### 4.4 Code Quality Checks

```bash
python -m black --check actfold tests demo.py scripts
python -m isort --check-only actfold tests demo.py scripts
python -m pyflakes actfold tests demo.py scripts
python -m mypy actfold --ignore-missing-imports
```

### 4.5 Real Model Benchmarks

```bash
# Use the provided GPT-2 example config
bash scripts/run_real_model_benchmark.sh

# Use a custom config (set model_name_or_path first, or use real_model_example.yaml)
bash scripts/run_benchmarks.sh actfold/configs/real_model_example.yaml
```

The first run will download model weights from the Hugging Face Hub if they are not already cached locally.

> **Note on folding path:** `BenchmarkRunner` automatically wraps the loaded
> model with `FoldedModel` when a recognizable Transformer layer stack is found,
> so the ActFold path reuses real parent activations during inference. The
> verification engine shares the same cache, gate, and scheduler with the folded
> model.

### 4.6 Ablation Studies

```bash
# With a real model config
bash scripts/run_ablation.sh actfold/configs/real_model_example.yaml

# Synthetic demonstration
bash scripts/run_ablation.sh --synthetic
```

Results are written to `results/threshold_sensitivity.csv`,
`results/layerwise_folding.csv`, and `results/cache_size_impact.csv`.

Outputs three tables:

1. Threshold sensitivity (τ ∈ {0.90, 0.95, 0.99})
2. Layer-wise folding (early / late / all)
3. Cache size impact (256, 512, 1024, 2048)

### 4.6a CUDA-Graph Verification-Loop Benchmark (AR002)

The fixed-shape folded verification loop can be captured as a CUDA graph
(`ManualFoldedForward(use_cuda_graph=True)` / `ActFoldConfig.use_cuda_graph`).
The BS-007 evidence benchmark measures eager vs graph replay per-step
wall-clock on a synthetic LLaMA-layout model (batch=2, seq=512, 4 layers) and
writes the artifact to `results/optimization/ar002_graph_bench.json`:

```bash
python -m scripts.ar002_graph_bench            # default path + 20 measured steps
python -m scripts.ar002_graph_bench --steps 50 --warmup 10
```

Reference measurement (Quadro RTX 5000): eager 5.083 ms/step vs graph
2.687 ms/step (**-47.1%**), 20/20 budget-validated steps. On
`LIBKINETO_NOCUPTI` torch builds the kernel-launch counts are recorded as
`null` with an explicit note (they cannot be measured there); rerun on a
CUPTI-capable host to record them.


### 4.7 Generate Figures

```bash
# Generate figures from real artifacts
python scripts/generate_figures.py --results-dir results/

# Generate example figures without running benchmarks
python scripts/generate_figures.py --demo
```

Produces:

- `figures/fig1_similarity_heatmap.png`
- `figures/fig2_pareto_frontier.png`
- `figures/fig3_tflops_reduction.png`
- `figures/fig4_ablation_table.png`

---

## 5. Real Model Configuration Examples

### GPT-2 (Lightweight Stand-In)

```yaml
model_name_or_path: "gpt2"
model_family: "causal_lm"
seq_len: 128
vocab_size: 50257
torch_dtype: "float32"

use_real_eval: true
eval_backend: "auto"
```

### LLaDA-8B

```yaml
model_name_or_path: "your-org/llada-8b"
model_family: "llada"
seq_len: 256
vocab_size: 32000
max_entries_per_layer: 2048
enable_dynamic_tau: true
torch_dtype: "float16"

use_real_eval: true
eval_backend: "auto"
eval_limit: 100
```

### Fast-dLLM-v2-1.5B

```yaml
model_name_or_path: "your-org/fast-dllm-v2-1.5b"
model_family: "fast_dllm"
seq_len: 256
vocab_size: 32000
enable_dynamic_tau: true

use_real_eval: true
eval_backend: "auto"
eval_limit: 100
```

---

## 6. Diffusion-Native Sampling

When ``model_family`` is ``llada``, ``dream``, or ``fast_dllm`` and
``num_steps > 1``, :meth:`~actfold.models.base.DiffusionLLM.generate` dispatches
to a model-family-specific reference sampler.  These samplers are aligned with
the official recipes and accept a ``sampler_config`` argument or individual
kwargs forwarded to the config dataclass.

### LLaDA (`LLaDASamplerConfig`)

```python
from actfold.models import load_model
from actfold.models.llada_sampler import LLaDASamplerConfig

model = load_model("your-org/llada-8b", model_family="llada")
config = LLaDASamplerConfig(
    num_steps=128,
    num_tokens=128,
    block_size=128,
    remasking="low_confidence",  # or "random"
    scheduler=None,              # None -> Linear; pass CosineMaskingScheduler()
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    cfg_scale=0.0,
)
output = model.generate(
    prompt_tokens,
    max_new_tokens=128,
    num_steps=128,
    sampler_config=config,
)
```

Key hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| ``num_steps`` | 128 | Total reverse-diffusion steps |
| ``block_size`` | ``num_tokens`` | Decode generation region in blocks |
| ``remasking`` | ``"low_confidence"`` | Rule for choosing which masks to keep |
| ``temperature`` | 0.0 | 0 = greedy + Gumbel-Max; >0 samples |
| ``cfg_scale`` | 0.0 | Classifier-free guidance scale (0 = disabled) |

### Dream (`DreamSamplerConfig`)

```python
from actfold.models.dream_sampler import DreamSamplerConfig

config = DreamSamplerConfig(
    num_steps=512,
    num_tokens=256,
    alg="maskgit_plus",       # "topk_margin" | "entropy"
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    right_shift_logits=True,  # AR alignment used by Dream
)
```

### Fast-dLLM (`FastDLLMSamplerConfig`)

```python
from actfold.models.fast_dllm_sampler import FastDLLMSamplerConfig

config = FastDLLMSamplerConfig(
    num_steps=128,
    num_tokens=256,
    block_size=32,
    small_block_size=32,
    threshold=0.95,
    temperature=0.0,
    top_p=0.95,
)
```

> **Note:** The samplers require the tokenizer to expose ``mask_token_id`` and
> ``eos_token_id`` (and ``pad_token_id`` for Fast-dLLM).  Final published
> results should be validated against the official implementation for the
> target checkpoint.

### Speculative Block Decoding (all diffusion families)

`LLaDASampler` and `DreamSampler` share one speculative mechanism core
(`actfold.models.speculative_core.SpeculativeCore`, AR023);
`FastDLLMSampler` keeps its original inline implementation as the
constant-compatible reference face (identical frozen constants, pinned by
UT parity — it does not inherit the core). With
`num_block_candidates > 0` the sampler builds an `[N, T]` candidate
canvas per decode unit (Fast-dLLM small block / LLaDA block / Dream
super-step window), verifies it with **one batched folded forward**, and
commits the winner by the min-prob rule (worst fill position maximized).
`candidate_include_greedy=True` (AR022) pins row 0 to the greedy hypothesis —
the anchor that restored baseline quality on Fast-dLLM-v2-1.5B.

| Parameter | Default | Description |
|---|---|---|
| ``num_block_candidates`` | 0 | Total candidate rows N (0 = off, byte-identical to the reference sampler) |
| ``candidate_top_k`` | 10 | Draft top-k pool the random rows sample from |
| ``candidate_include_greedy`` | False | Pin row 0 to the greedy (argmax) hypothesis |
| ``speculative_steps`` | 8 | Dream only: super-step window size over the transfer schedule |

```python
from actfold.models.dream_sampler import DreamSamplerConfig

config = DreamSamplerConfig(
    num_steps=512,
    num_tokens=256,
    num_block_candidates=4,       # speculative canvas [4, T]
    candidate_top_k=10,
    candidate_include_greedy=True,  # greedy anchor row 0
    speculative_steps=8,
)
```

Requirements and caveats: the speculative path requires B==1, a
``branch_ids``-capable ``folded_model`` (`ManualFoldedForward`), and
``cfg_scale <= 0`` (the code face rejects `cfg_scale > 0`;
construction- or call-time `ValueError` otherwise — never a silent
downgrade). The same keys are consumed from YAML
(``num_block_candidates`` / ``candidate_top_k`` / ``candidate_include_greedy``)
without the unknown-key warning; see the commented speculative sections in
``configs/per_model/llada_8b.yaml`` / ``dream_7b.yaml``. The per-candidate
seed formula is frozen and host-reproducible; see `docs/ALGORITHM.md` §13 for
the full mechanism and AGENTS.md #43–#45 for the operational contracts.

---

## 7. Cross-Family Probe and Benchmark (AR023)

Two scripts exercise the speculative stack against **real remote-code
checkpoints** (LLaDA-8B-Base, Dream-v0-Base-7B). Run both from a
transformers-4.53.1 environment (see `docs/RERUN_CHECKLIST.md` §1 and step
13 for the full rerun recipe, VRAM expectations, and pre-commitment
discipline).

### Family probe — `scripts/ar023_family_probe.py`

Per-family viability gate run **before** any formal benchmark: curl download
manifest + SHA256 verification, transformers 4.53.1 load, bf16 VRAM
measurement, bitsandbytes int8 Windows feasibility, short-generation smoke,
CUDA-graph capture headroom, and prompt adaptation (Dream's custom attention
rejects int64 masks — the probe uses bool masks throughout).

```bash
python -m scripts.ar023_family_probe --models-root <weights dir> --hf-home <cache dir>
```

Artifact: ``results/optimization/ar023_family_probe.json``. CUDA faces run
subprocess-isolated: a graph-face allocator assert can corrupt process-level
CUDA state, so a degrade in one face never poisons the others. Recorded
conclusions (Quadro RTX 5000 16GB): both families load in bf16 with
``degraded=False``; **graph capture is disabled** for both (headroom), so
the benchmark below records the eager-neutral note.

### Four-mode benchmark — `scripts/ar023_cross_family_bench.py`

Four modes over one shared model instance, prompt set, and seed 2026:
**baseline** (the family's official recipe, no folded model),
**folded-serial** (N=0 + one ManualFoldedForward chain), **folded-batched**
(N=4, top_k=10), and **folded-batched-greedy** (an independently constructed
identical MFF + `candidate_include_greedy=True` — same shape/seed/prompt A/B
that constructively zeroes the speed variable).

```bash
python -m scripts.ar023_cross_family_bench --family llada --models-root <weights dir> --reps 30
python -m scripts.ar023_cross_family_bench --family dream --models-root <weights dir> --reps 30
```

Artifacts: ``results/optimization/ar023_llada_bench.json`` /
``ar023_dream_bench.json`` (formal) and ``..._bench_quick.json``
(``--quick`` smoke = 1 prompt / reps=3 / 64 tokens — never overwrites the
formal artifact). Frozen verdict gates: **H1** hierarchical quality
(aligned-vs-baseline posthoc gap ≤ 0.5 of the batched gap, decode-length
recovery per prompt, merged block-grid exact-match strictly improving),
**H2** strict speedup `p50(baseline)/p50(aligned) > 1.0`, **H4**
canary/ledger full-stack conjunction. The baseline mode is sampled twice per
prompt as a reproduction self-check — a strict mismatch blocks **all**
verdicts; a degraded family or any single-mode failure (e.g. OOM) writes an
honest ``blocked`` artifact with the probe cross-reference and zero
fabricated numbers. Timing follows the pre-committed single-run discipline
(#39: host `perf_counter` + `cuda.synchronize`, reps=30, no rerun fishing).

### Multi-parent batch folding (synthetic face) — `scripts/ar025_multi_parent_bench.py`

The committed synthetic face of AR025 (multi-parent batch folding): 16 shape
points (chains × candidates-per-chain × seq_len) comparing the merged
multi-parent forward `[sum(N), T]` (A, fold switch on) against per-chain
forwards `[N, T]` (B, switch off), plus a report-only serial reference (C).

```bash
python -m scripts.ar025_multi_parent_bench   # CPU, minutes; --quick smoke
```

Artifact: ``results/optimization/ar025_multi_parent_bench.json`` (frozen S1/S2/S3
verdict formulas in meta; `--quick` writes a separate ``_quick`` artifact).
Frozen gates: **S1** parity (nonsplit bitwise `torch.equal`, split
`allclose(1e-6)`; any FAIL blocks ALL verdicts), **S2** strict
`write_op_count` reduction (ledger 4 vs 40–160), **S3** merged/serial
engagement ratio (report-only, 0.9–0.975). Formal run: overall **supported**
(S1/S2 supported, empty blocked list). DD8 erratum: the meta records
`cache_budget_tokens_per_layer=65536` — at the historical default 8192, a
`(parents+ΣN)·T` overflow evicts in-flight parents via the token-row LRU (a
capacity confound, not a mechanism defect); the first-run blocked evidence is
preserved verbatim in ``ar025_multi_parent_bench_eviction_blocked.json``.
`--mode real` is the pending AR025 T006 delivery (depends on AR023 T006) and
not yet available.

---

## 8. Programmatic Real Model Benchmark

```python
from actfold.eval.benchmark_runner import BenchmarkRunner
from actfold.utils.config_manager import load_config

config = load_config("actfold/configs/real_model_example.yaml")
runner = BenchmarkRunner(config)
results = runner.run(
    tasks=["gsm8k", "math", "ifeval", "humaneval_plus", "mbpp_plus"],
    num_samples=100,
    output_dir="results",
)

for task, metrics in results.items():
    if "accuracy" in metrics:
        print(f"{task}: baseline_acc={metrics['baseline_accuracy']:.3f}, "
              f"actfold_acc={metrics['actfold_accuracy']:.3f}, "
              f"tflops_reduction={1 - metrics['actfold_tflops'] / metrics['baseline_tflops']:.1%}")
    else:
        print(f"{task}: baseline_pass@1={metrics['baseline_pass_at_1']:.3f}, "
              f"actfold_pass@1={metrics['actfold_pass_at_1']:.3f}, "
              f"tflops_reduction={1 - metrics['actfold_tflops'] / metrics['baseline_tflops']:.1%}")
```

---

## 9. Reproducibility Checklist

- [ ] `requirements.txt` and `requirements-bench.txt` installed with pinned versions
- [ ] `torch.manual_seed(42)` in every entry point
- [ ] CUDA deterministic mode enabled
- [ ] Config file saved alongside results (manual, or use `output_dir`)
- [ ] Logs written to `logs/` directory (configure your logging handler)
- [ ] Model weights cached or checkpoint path documented

---

## 10. Extending to New Diffusion LLMs

To evaluate a new Diffusion LLM family:

1. Create a subclass of `actfold.models.base.DiffusionLLM`:

```python
from actfold.models.base import DiffusionLLM

class MyDiffusionLLM(DiffusionLLM):
    def embed(self, tokens):
        """Return input embeddings [B, T, H]."""
        return self.model.get_input_embeddings()(tokens)

    def forward(self, tokens, attention_mask=None, **kwargs):
        ...

    def generate(self, prompt_tokens, **kwargs):
        ...

    @property
    def num_layers(self): ...

    @property
    def hidden_dim(self): ...

    @property
    def num_heads(self): ...

    @property
    def vocab_size(self): ...
```

2. Register it:

```python
from actfold.models import ModelRegistry
ModelRegistry.register("my_family", MyDiffusionLLM)
```

3. Use it in a config:

```yaml
model_name_or_path: "organization/my-model"
model_family: "my_family"
```

4. To give the family's reference sampler the shared speculative mechanism
   (AR023), subclass both `DiffusionSampler` and `SpeculativeCore` and
   implement only the canvas/payload semantics — one speculative unit's
   `_build_unit_forced` region, its `SpecCtx(mask, position)` (rotary cargo
   only if the family's remote attention requires it), the host
   `_spec_ctx_for(x, prompt_len)` canvas adapter (the LRU skeleton lives in
   the core), the counter initialization (`_greedy_wins` /
   `_greedy_spec_calls`) in the sampler's own `__init__` (the core defines
   **no** `__init__`), and the entry validations. The commit rule, seed
   formula, ctx LRU, and `greedy_win_stats` probe come from the core (see
   `docs/ALGORITHM.md` §13 and `LLaDASampler` / `DreamSampler` for the two
   worked examples).

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| High accuracy drop | Threshold too aggressive | Lower `tau` or enable dynamic scheduling |
| Low FLOPs reduction | Threshold too conservative | Raise `tau` or check cache hit rate |
| Out of memory | Cache too large or model too big | Reduce `max_entries_per_layer`, enable `load_in_8bit` |
| CPU-GPU sync warning | `.item()` / `.cpu()` in hot path | Keep masks as CUDA tensors |
| Model download fails | No internet or gated model | Use local path or authenticate with `huggingface-cli login` |
| `trust_remote_code` error | Custom model architecture | Set `trust_remote_code: true` in config |
| `bitsandbytes` device conflict | `.to(device)` after quantized load | Quantized models manage their own device mapping; do not call `.to(device)` |
| "lm-eval not installed" error | Missing benchmark dependencies | Install `requirements-bench.txt` |
| "evalplus sandbox requires Unix" error | Running evalplus on Windows | Use WSL or a Unix-like environment; evalplus code execution relies on the `resource` module |
| Empty completions from tokenizer | Tokenizer decode returned only special tokens | Check `skip_special_tokens=True` and generation length |
| Triton kernel not used | Triton not installed or tensor on CPU | Install `triton>=2.0` on Linux/WSL; CPU tensors automatically use the PyTorch fallback |
| Slower merge than expected | Very small hidden dim or non-standard shape | The Triton kernel requires `hidden_dim % 128 == 0`; otherwise PyTorch fallback is used |
| "model_name_or_path is required" error | BenchmarkRunner config lacks a model path | Set a real model identifier in the config or pass `--model` |
| "A real tokenizer is required" error | Evaluation model has no tokenizer | Use a model family that loads a tokenizer (e.g., `causal_lm`) |
| Synthetic-looking ablation results | Using `--synthetic` or a template config without a checkpoint | Provide a real `model_name_or_path` for measured results |

---

## AR029 — Bounded-Memory Pyramid Cache (mechanism face)

AR029 is a mechanism/infrastructure experiment: it validates the cache contract
and the sentinel guard, not a baseline speedup. The only runnable face on the
current scope decision (`baseline_claim=false`) is `--quick`:

```bash
python -m scripts.ar029_pyramid_cache_bench --quick
```

It writes `results/optimization/ar029_pyramid_cache_bench_quick.json` with:

- `full`: no eviction, compared bit-for-bit against `ActivationCache`
  (`total_tokens = 4 * T * depth`, uniform ratios 1.0);
- `pressure`: a genuine linear pyramid `[4*block, 1*block]` (parent layer 0
  resident, target layer 1 tail evicted), compared bit-for-bit against
  `_HoleMaskedDense` — a real `ActivationCache` with the pyramid's per-layer
  holes injected as NaN and the same `availability`;
- `reference`: the dense oracle's own accounting.

Frozen verdicts: **E1** both diffs `== 0.0`; **E2** per-layer token bound and
`stats["bytes"] == sum(block tensor bytes)` on both faces; **E3**
`reuse_tokens > 0` and `reuse/valid >= 0.01`; **E4** every output and stat
finite and non-negative. Precedence is `blocked > supported > refuted`.

Measured (CPU, seed 1234, single shot): E1 `0.0/0.0`, E2 PASS, E3 `40/64`,
E4 PASS → `supported`. The formal real-checkpoint A/B is **BLOCKED-DEFERRED**;
`main([])` without `--quick` returns exit 3 and writes nothing. Adding it
requires a new `run_formal_queue` FaceSpec plus the standard dual-metric,
pre-committed, `reps>=30` protocol (see `docs/RERUN_CHECKLIST.md` step 17).

## AR030 — Paged Shared Fold Memory (mechanism face)

AR030 is a mechanism/infrastructure experiment for Agent/RSI multi-candidate
rollouts: it validates the page table / refcount / COW page-sharing contract
under a hard physical-page budget, not a baseline speedup. The only runnable
face under the current scope decision (`baseline_claim=false`) is `--quick`:

```bash
python -m scripts.ar030_paged_agent_bench --quick
```

It writes `results/optimization/ar030_paged_agent_bench_quick.json` with
schema `ar030_paged_agent_bench` (version 1).

**Frozen geometry** (single shot, CPU, deterministic):

| Parameter | Value |
|---|---|
| Model | real 2-layer `TinyDecoder` (hidden 32, vocab 64, real attention + FFN) |
| `SEQ` | 128 |
| `PAGE` (page tokens) | 16 |
| `CANDIDATES` | 8 (each changes the last 8 tokens) |
| Gate | `SimilarityGate` cosine, `TAU = 0.9` + `forced_divergent` |
| Pressure face budget | 16 pages/layer (`PRESSURE_TOTAL = 512` tokens, exactly at the boundary: 8 root + 8 candidate pages, zero evictions) |
| Full face budget | 256 pages/layer (`FULL_TOTAL = 8192` tokens) |
| `MIN_SHARING_RATIO` | 0.5 |

**Frozen verdicts** (precedence `blocked > supported > refuted`):

- **E1 exactness**: both faces' `max_abs_diff_vs_dense == 0.0` — the paged
  cache must be bit-exact against the dense `ActivationCache` oracle.
- **E2 memory bound + sharing**: physical pages within the per-layer budget;
  measured physical pages 32 (16/layer, exactly at the frozen 16-page budget) vs naive 144 pages, page
  sharing ratio and byte sharing ratio 0.778 (98304 vs 442368 bytes),
  `evictions == 0`.
- **E3 fork/COW**: fork byte deltas all 0; new physical pages per candidate ==
  content-diff pages == 2 (one per layer); shared pages bit-equal; sharing
  ratio >= `MIN_SHARING_RATIO` (0.778 >= 0.5).
- **E4 honesty** (14 checks): commit-failure injection (k=0/k=1) rolls back
  atomically; a stale plan is rejected; snapshot/restore round-trips
  bit-exactly with `epoch+1` including tier/availability equality;
  demote/promote round-trips bit-exactly; plan availability == executor
  availability; window reads match the frozen plan view; COW-off skips
  comparison and in-place mutation; refcount corruption is detected; a
  rollout rollback restores the fork baseline; a rollout window validation
  failure discards only that candidate's staged writes; `close` unfreezes
  the base; the refcount ledger has zero leaks; every output and stat is
  finite and non-negative.

Measured (CPU, single shot): E1 `0.0/0.0`, E2 `32` vs `144` / `0.778` /
`evictions=0`, E3 all-0 fork deltas / new==diff==2, E4 14/14 honesty checks →
**E1-E4 supported**, `overall="supported"`.

**Honest negatives**: verdict logic is frozen inside the script before data
collection (one-shot, never rerun-until-pass); a strict failure blocks ALL
verdicts; `inconclusive` / `refuted` / `blocked` outcomes are delivered
verbatim in the artifact.

**BLOCKED-DEFERRED boundary**: the artifact carries
`evidence_level="mechanism"` and `baseline_claim=false` — no baseline-speedup
claim may be inferred from it. `main([])` without `--quick` returns exit 3 and
writes nothing. The formal real-checkpoint A/B (LLaDA/Dream multi-candidate
rollout, dual-metric, pre-committed, `reps>=30`) first requires a new
`run_formal_queue` FaceSpec plus a >=24 GiB quiet GPU server; see
`docs/RERUN_CHECKLIST.md` step 18 for the unblocking conditions and the
protocol draft.
