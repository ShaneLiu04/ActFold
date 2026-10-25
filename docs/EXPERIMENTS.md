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

### 4.5a Adaptive-Uncertainty Three-Arm Benchmark (AR031)

```bash
# AR019 venv (transformers==4.53.1) + local Fast-dLLM-v2-1.5B weights
<venv>\Scripts\python.exe -X utf8 -m scripts.ar031_adaptive_uncertainty_bench `
    --hf-home <hf-cache> --model-path <weights-dir> --reps 30
```

Three arms (official baseline / adaptive-off / full), H1-H4 verdict contracts
frozen before data collection, per-(arm, prompt) double-run reproduction veto.
Expected (pre-committed single run,
`results/optimization/ar031_adaptive_uncertainty_bench.json`): H1/H2/H4
supported (quality restored — posthoc gap 0.0, decode lengths 34/66 —
6.2690x vs baseline, zero per-layer budget violations, reproduction 6/6),
H3 honest-refuted (no additional speedup over adaptive-off: the budget cap is
non-binding at this workload; delivered as a safety/regularization rung). See
`docs/RERUN_CHECKLIST.md` step 19 for the full protocol.

### 4.5b Speculative Advance-Reuse Three-Arm Benchmark (AR032)

```bash
# AR019 venv (transformers==4.53.1) + local Fast-dLLM-v2-1.5B weights
<venv>\Scripts\python.exe -X utf8 -m scripts.ar032_speculative_advance_bench `
    --hf-home <hf-cache> --model-path <weights-dir> --reps 30
```

Three arms (official baseline / reuse-off = the AR022-aligned N=4+greedy
eager config / reuse-on = ② plus the single `speculative_advance_reuse`
switch), H1-H4 verdict contracts frozen before data collection, per-arm
event-count-parameterized ledger bounds, repro double-run keyed by
`(arm, prompt_index)` (a mismatch refutes H4 and overall; H1-H3 keep their
recomputable verdicts). Expected (pre-committed single run,
`results/optimization/ar032_speculative_advance_bench.json`, ~46min at
~4.6GB peak on the 16GB WDDM host): overall **supported** — H1 gap 0.0 /
decode [34,66] on==off / exact-match 1.0/1.0; H2 **7.9698×** vs baseline;
H3 **1.2875×** vs reuse-off with the hard count face off_serial(4) −
on_serial(2) == eligible(2); H4 canary 3/3 + ledger within + repro 6/6.
Same-run A/B ratios are internally valid; absolute p50s are NOT comparable
across artifacts (WDDM host-state drift — see RERUN step 20). See
`docs/RERUN_CHECKLIST.md` step 20 for the full protocol.

### 4.5c Block-Causal History-Cache Four-Arm Benchmark (AR033)

```bash
# AR019 venv (transformers==4.53.1) + local Fast-dLLM-v2-1.5B weights
<venv>\Scripts\python.exe -X utf8 -m scripts.ar033_history_cache_bench `
    --model-path <weights-dir> --reps 30        # --quick smoke first
```

Four arms on the **official v2 geometry block=32/sb=8** (A non-cached
baseline = the pre-AR033 ladder's reference arm, restated by the strawman
erratum / B1 `history_cache` cross-block face / B2 + intra-block
small-block cache with the official boundary invalidation rule / C the
official native `batch_sample`, report-only timing reference fed RAW
unpadded prompts). H1-H4 frozen before data collection; the raw-call
recorder rebuilds the exact token ledger via class-geometry assertions;
repro double-run keyed by `(arm, prompt_index)` with a one-vote block
(mismatch ⇒ all verdicts blocked). Expected (pre-committed single run,
`results/optimization/ar033_history_cache_bench.json`, ~3.5h on the 16GB
WDDM host): overall **supported** — H1 gap 0.0 / decode [34,66] identical
across arms / token rate 1.0; H2 **B1 1.2989× / B2 1.5431×** vs the
non-cached baseline (B2 vs B1 1.188×); H3 exact ledger identity A=5280 /
B1=2208 / B2=714 tokens (B2/A=0.1352); H4 canary green + repro 8/8. C-arm
report-only: ActFold B1/B2 run 1.2821×/1.5232× faster than the official
native recipe; official vs non-cached baseline 1.0131×. Geometry
isolation: sb=8 arms are NOT cross-artifact comparable with the pre-AR033
lineage (sb=32). See `docs/RERUN_CHECKLIST.md` step 21 for the full
protocol.

### 4.5d KV-Cached Speculative Block-Decoding Four-Arm Benchmark (AR034)

```bash
# AR019 venv (transformers==4.53.1) + local Fast-dLLM-v2-1.5B weights
<venv>\Scripts\python.exe -X utf8 -m scripts.ar034_kv_speculative_bench `
    --model-path <weights-dir> --reps 30        # --quick smoke first
```

Four arms on the AR022-lineage geometry (block=32, small_block=32): A
non-cached baseline / B2 AR033 cache / D composition (`history_cache` +
`num_block_candidates=4` + greedy anchoring) / C native `batch_sample`
(report-only, RAW prompt). H1-H4 are frozen before data collection; the class
ledger plus a forward-count comparison (D < B2) is H3; a reproduction mismatch
blocks all verdicts. Expected (pre-committed single run, ~2.5h on the 16GB WDDM
host): overall **supported** -- H1 gap 0.0 / decode [34, 66] / token rate 1.0;
H2 **8.7308x vs A** and **6.9743x vs B2**; H3 D 8 forwards vs B2 62; H4
canary green + repro 8/8. Geometry law: `small_block_size` must equal
`block_size` (at sb=8 random rows can beat the greedy row, T007). See
`docs/RERUN_CHECKLIST.md` step 22.

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

### 4.5e AR035 gated speculative commit (cost/quality knob)

`python -m scripts.ar035_gated_spec_bench --quick` (three arms A/D/G, block=32/sb=32, 64 tokens,
reps=2, tau=0.95). Quick artifact: `results/optimization/ar035_gated_spec_bench_quick.json`.
Frozen gates: H1 quality (token-exact vs A, mask-free), H2 cost (A/G > 1.0, G/D published),
H3 ledger (`spec_gated_iterations == spec_verify_forwards > 0`, no degraded calls), H4 repro
double-run equality blocking all verdicts on mismatch. Quick outcome: H1/H3/H4 supported,
**H2 refuted** (A/G 0.7385, G/D 9.5832). Never cite the quick artifact as a speedup claim.

### 4.5f AR036 commit-draft fusion (A/D/F)

```powershell
$env:FASTDLLM_MODEL_PATH="<checkpoint>"
<venv>\Scripts\python.exe -m scripts.ar036_commit_draft_fusion_bench --quick `
  --num-tokens 160 --no-eos --prompt-text "Write a long detailed technical explanation of ..."
```

Three arms A (non-cached baseline) / D (AR034 composition) / F (D + `commit_draft_fusion`), frozen
H1-H4 (H2 threshold `p50(D)/p50(F) >= 1.15`). **Geometry requirement**: ordinary prompts early-stop
at ~34 tokens (one block), so fusion never triggers; `--num-tokens 160 --no-eos` gives 5 blocks and
`stop_token_id=-1` disables early exit. Quick artifact:
`results/optimization/ar036_commit_draft_fusion_bench_quick.json`; formal:
`ar036_commit_draft_fusion_bench.json` (reps=30, quiet server). `--arms` subsets write a `smoke`
payload without verdicts.

### 4.5g AR037 long-horizon measurement + adaptive candidate budget

```powershell
$env:FASTDLLM_MODEL_PATH="<checkpoint>"
<venv>\Scripts\python.exe -m scripts.ar037_long_quality_probe            # A vs D, 192 tokens
<venv>\Scripts\python.exe -m scripts.ar037_acb_probe --num-tokens 192 --threshold 0.5
<venv>\Scripts\python.exe -m scripts.ar037_acb_probe --num-tokens 192 --threshold 0.5 --fusion
```

EOS is disabled inside the probes (`stop_token_id=-1`). Report **mean log p under the same model**
as the quality metric — never token agreement with the baseline at long horizons (whole-block
greedy commit legitimately diverges from thresholded remasking). Expected reference host numbers:
A 172.4s / D 18.3s / D+CDF 16.0s / D+CDF+ACB(tau=0.5) 11.8s, all composed arms at log p -2.0870.
Sweep tau in {0.9, 0.7, 0.5}: 0% / 33% / 50% skip rates on the reference prompt.

### 4.5h AR038 confidence-scheduled cascade (escalation ablation)

```powershell
$env:FASTDLLM_MODEL_PATH="<checkpoint>"
<venv>\Scripts\python.exe -m scripts.ar037_acb_probe --num-tokens 192 --threshold 0.5 --fusion
<venv>\Scripts\python.exe -m scripts.ar037_acb_probe --num-tokens 192 --threshold 0.5 --fusion `
  --escalate --escalate-threshold 0.9 --escalate-candidates 8
```

Expected reference numbers (real checkpoint, Quadro RTX 5000): ACB+CDF 11.84 s / 3 skips;
D+CDF 16.02 s; escalation N=8 23.35 s — **all at mean log p -2.0870**. Escalation is therefore an
ablation, not a feature: do not enable it in production configs.

### 4.5i AR039 free verification + verification-free greedy commit

```powershell
$env:FASTDLLM_MODEL_PATH="<checkpoint>"
<venv>\Scripts\python.exe -m scripts.ar037_acb_probe --num-tokens 192 --threshold 0.5 --fusion --vfg
```

Runs three arms in one pass (D = CDF, S = ACB+CDF, V = VFG+CDF). Expected reference numbers:
D 14.50s / S 10.16s / V 8.84s, all at mean log p -2.0870; V shows
`vfg=4 fv=4/2 verify=2` (four verification-free blocks, two free-verify re-arms).

### 4.5j AR040 matrix benchmark (prompt × length, four arms)

```powershell
$env:FASTDLLM_MODEL_PATH="<checkpoint>"
<venv>\Scripts\python.exe -m scripts.ar040_matrix_bench --quick     # 3 prompts x 160, ~8 min
<venv>\Scripts\python.exe -m scripts.ar040_matrix_bench             # 6 prompts x {160,320}, ~2 h
```

Frozen gates: H1 quality cells (logp(V) >= logp(A) - 0.05 on >=90% cells), H2 median speedup >=10x
and V<S<D ordering on >=80% cells, H3 V-arm ledger identities, H4 per-cell V reproduction.
Quick result on the reference host: **H1 1/3, H2 ordering 1/3, H3/H4 pass -> refuted**, with median
speedup 16.17x. The lesson is in the artifact, not just the verdict: speed generalizes, quality does
not — always report both, per prompt.

### 4.5k AR041 remask fallback threshold sweep

```powershell
$env:FASTDLLM_MODEL_PATH="<checkpoint>"
<venv>\Scripts\python.exe -m scripts.ar041_remask_fallback_probe --thresholds 0.5,0.3,0.2,0.1
```

Runs A and D once per prompt, then the fallback arm at each threshold. Expected reference numbers
(3 prompts x 160 tokens): A 107-192 s with logp -1.1528 / -0.8482 / -0.8086; D 10-15 s with
-1.3118 / -0.9904 / -1.5370; R at tau >= 0.2 falls back on all blocks and reproduces A's logp
exactly at 77-134 s. At tau = 0.1 prompt 2 releases one block and degrades to -0.8914.
