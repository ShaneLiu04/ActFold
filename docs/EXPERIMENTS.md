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
