<div align="center">

# ActFold

**Cross-Branch Activation Reuse for Diffusion LLM Speculative Decoding**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![CI](https://github.com/ShaneLiu04/ActFold/actions/workflows/ci.yml/badge.svg)

[Overview](#overview) •
[Features](#features) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Usage](#usage) •
[Benchmarks](#benchmarks) •
[Docs](#docs) •
[Citation](#citation)

</div>

---

## Overview

ActFold is a research framework that reduces **verification-phase FLOPs** in Diffusion LLM speculative decoding by reusing activations across candidate branches.

In speculative decoding, multiple child branches are drafted from a parent sequence and verified independently. Despite children typically diverging by only a few tokens, standard implementations trigger a **full forward recomputation** across all Transformer layers. ActFold attacks this redundancy with **Branch Folding**: at every layer, each token is classified as either **stable** (reuse the parent's activation) or **divergent** (recompute), and the two groups are merged into a single output.

> **Target impact**: 21%–62% verification TFLOPs reduction with minimal accuracy loss.

### The Core Idea

For each layer `l`, token position `t`, and diffusion step `s`:

```
sim(l, t, s) = cosine_similarity(h_parent[l, t, s], h_child[l, t, s])
```

- **Stable tokens** (`sim > τ`): copy cached parent FFN outputs.
- **Divergent tokens** (`sim ≤ τ`): run the full layer on the child hidden states to preserve self-attention context.

The decision is made independently per layer and per token, so even branches that differ in many positions still benefit from folding wherever the hidden states agree.

---

## Features

| Feature | What it does |
|--------|--------------|
| **Branch Folding Engine** | Wrap any Transformer stack with `FoldedModel` to enable per-token activation reuse. |
| **True End-to-End Folded Generation** | `folded_generate()` produces each new token through a folded child forward pass, so benchmarks actually exercise the accelerated path. |
| **Layer-Aware Stability Profiler** | Records real per-layer stable ratios instead of estimating from input embeddings. |
| **Chunked Activation Cache** | Optional contiguous tensor-block cache that lowers memory fragmentation vs. per-token dict storage. |
| **Compute-Bandwidth Cost Model** | Estimates wall-clock latency from both compute FLOPs and memory bandwidth, not just FLOPs. |
| **Diffusion-Native Samplers** | High-quality reference samplers for LLaDA, Dream, and Fast-dLLM aligned with official recipes (masking schedules, block decoding, confidence-based unmasking). |
| **Real Evaluation Backends** | Integrated `lm-eval` and `evalplus` judges; no mock fallbacks. |
| **Optional Triton Kernel** | Fused stable/divergent merge on CUDA with a verified PyTorch fallback on CPU. |

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Hugging Face `transformers`
- CUDA-capable GPU (optional; CPU fallback supported)
- `lm-eval` and `evalplus` for benchmark evaluation

### From source

```bash
# Runtime dependencies
pip install -r requirements.txt

# Benchmark backends (required for evaluation)
pip install -r requirements-bench.txt

# Development tools (formatting, type checking, tests)
pip install -r requirements-dev.txt

# Editable install
pip install -e .
```

---

## Quick Start

### 1. Run the demo

```bash
# Synthetic demonstration model — no downloads, runs everywhere
python demo.py

# Standard causal LM (GPT-2, LLaMA, Qwen, Mistral, ...)
python demo.py --model gpt2 --model-family causal_lm

# Architecture-agnostic AutoModel wrapper (works with LLaDA/Dream/Fast-dLLM checkpoints)
python demo.py --model <llada-checkpoint> --model-family generic

# Diffusion-native generation with Branch Folding
python demo.py \
    --model <llada-checkpoint> \
    --model-family llada \
    --prompt "The future of artificial intelligence is" \
    --num-steps 128 \
    --max-new-tokens 32
```

The demo auto-detects the embedding module, Transformer layer stack, and
language modeling head for most Hugging Face architectures. It first tries
:class:`~actfold.core.model_wrapper.FoldedModel` (which recognizes layer paths
such as `model.layers`, `transformer.h`, `bert.encoder.layer`, and
`decoder.block`) and falls back to explicit extraction via
:class:`~actfold.models.architecture_utils.ManualFoldedForward` when needed.

Expected output (synthetic model):

```text
=================================================================
 ActFold Demo
=================================================================
 Device: cuda
 Model: synthetic demonstration Transformer
 Architecture: 4 layers, 128 hidden dim, 8 heads
 Vocab size: 1000
 Parent branch: [seq_len=16]
 Child branches: 2

 Verification Results:
+-------+----------+---------+------------+
| Layer | Baseline | ActFold | Similarity |
+-------+----------+---------+------------+
| 0     | 100%     | 6%      | 0.968      |
| 1     | 100%     | 5%      | 0.970      |
| 2     | 100%     | 6%      | 0.969      |
| 3     | 100%     | 6%      | 0.968      |
+-------+----------+---------+------------+
 Total FLOPs reduction: 78.5%
 Output equivalence (MSE): 2.35e-03  [HIGH]
 Estimated stable token ratio: 93.75%
=================================================================
```

### 2. Wrap a model and verify a child branch

```python
import torch
from actfold.core import ActivationCache, FoldedModel, SimilarityGate

# Load or build any Transformer model
raw_model = ...

cache = ActivationCache(max_entries_per_layer=1024, device="cuda")
gate = SimilarityGate(tau=0.95)
folded = FoldedModel(raw_model, cache=cache, gate=gate)

# Run parent to populate the cache
parent_logits = folded(parent_tokens, branch_id="parent")

# Verify child while reusing parent activations where stable
child_logits = folded(
    child_tokens,
    branch_id="child",
    parent_branch_id="parent",
)
```

### 3. End-to-end folded generation

```python
from actfold.speculative.folded_generation import folded_generate

result = folded_generate(
    adapter,
    prompt_ids,
    max_new_tokens=32,
    folded_model=adapter.folded_model,
)
print(result.tokens)         # [batch, prompt_len + max_new_tokens]
print(result.stable_ratio)   # measured mean per-layer stable ratio
```

### 4. Run benchmarks

```bash
# Real-model benchmarks with the provided GPT-2 example
bash scripts/run_real_model_benchmark.sh

# Custom config
bash scripts/run_benchmarks.sh actfold/configs/real_model_example.yaml
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Diffusion LLM                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │      Draft Generator        │
        └──────────────┬──────────────┘
                       │
            ┌──────────▼──────────┐
            │   Parent Branch     │
            └──────────┬──────────┘
                       │
         ┌─────────────▼─────────────┐
         │   ActFold Verification    │
         │  ┌─────────────────────┐  │
         │  │   Similarity Gate   │  │
         │  └──────────┬──────────┘  │
         │             │             │
         │    ┌────────┴────────┐    │
         │    ▼                 ▼    │
         │ Stable Tokens   Divergent │
         │    │               Tokens  │
         │    ▼                 ▼    │
         │ Activation     Full Layer │
         │   Cache        Recompute  │
         │    │                 │    │
         │    └────────┬────────┘    │
         │             ▼             │
         │      Merged Output        │
         └─────────────┬─────────────┘
                       │
            ┌──────────▼──────────┐
            │   Accepted Branch   │
            └─────────────────────┘
```

### Module map

```
actfold/
├── models/         # Diffusion LLM wrappers and native samplers
├── core/           # Branch Folding engine (cache, gate, folded layers, scheduler)
├── profiler/       # Stability profiler, similarity analysis, GPU metrics
├── speculative/    # Draft generator, verification engine, folded generation
├── eval/           # Benchmark runner, lm-eval / EvalPlus adapters, judges
├── utils/          # Config, FLOPs counter, cost model, logging
└── configs/        # YAML experiment configs
```

---

## Usage

### Wrapping a model with `FoldedModel`

`FoldedModel` discovers common Transformer layer stacks (`layers`, `model.layers`, `transformer.h`, `encoder.layer`, `gpt_neox.layers`, `model.decoder.layers`) and replaces each layer with a `FoldedTransformerLayer`.

```python
from actfold.core import ActivationCache, FoldedModel, SimilarityGate
from actfold.core.folding_scheduler import FoldingScheduler

cache = ActivationCache(max_entries_per_layer=1024, device="cuda")
gate = SimilarityGate(tau=0.95, metric="cosine")
scheduler = FoldingScheduler(
    base_tau=0.95,
    num_layers=model.num_layers,
    num_steps=1,
)

folded = FoldedModel(
    raw_model,
    cache=cache,
    gate=gate,
    scheduler=scheduler,
)

# Restore the original model at any time
base_model = folded.restore()
```

For unsupported architectures, pass `layer_names=("your.path",)` or implement a custom `DiffusionLLM.forward()` that routes hidden states through folded layers.

### Using the verification engine

```python
from actfold.speculative import ActFoldVerificationEngine, DraftGenerator, SpiffyBaseline
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter

adapter = FastDLLMAdapter(model, folded_model=folded)
draft_generator = DraftGenerator(vocab_size=adapter.vocab_size, mode="copy_flip")
baseline = SpiffyBaseline(adapter, draft_generator)

engine = ActFoldVerificationEngine(adapter, cache, gate)
result = engine.verify_branch(parent_branch, child_branch, step_idx=0)
print(result.accepted)
print(result.stable_ratio)          # real per-layer mean when folded_model is used
print(result.tflops)
print(result.estimated_latency_ms)  # compute-bandwidth-aware estimate
```

### Diffusion-native sampling

When `num_steps > 1`, `DiffusionLLM.generate()` dispatches to a model-family-specific sampler. The samplers are now high-quality reference implementations aligned with the official recipes:

- **LLaDA** (`LLaDASampler`): right-padded canvas, block-wise decoding, masking schedule (`linear`/`cosine`), `low_confidence` / `random` remasking, optional CFG, temperature/top-p/top-k, and Gumbel-Max noise.
- **Dream** (`DreamSampler`): left-padded canvas, MaskGIT-style iterative decoding with `maskgit_plus` / `topk_margin` / `entropy` confidence rules, optional CFG, and `alg_temp` soft selection.
- **Fast-dLLM** (`FastDLLMSampler`): block-wise masked decoding with small-block threshold unmasking, top-p/temperature sampling, stop-token early termination, and autoregressive block extension.

```python
from actfold.models import load_model
from actfold.models.llada_sampler import LLaDASamplerConfig

model = load_model("path/to/llada", model_family="llada")
config = LLaDASamplerConfig(
    num_steps=128,
    num_tokens=128,
    block_size=128,
    remasking="low_confidence",
    temperature=0.0,
)
output = model.generate(
    prompt_tokens,
    max_new_tokens=128,
    num_steps=128,
    folded_model=folded,
    sampler_config=config,
)
```

> These samplers closely follow the official LLaDA/MDLM, Dream, and Fast-dLLM v2 recipes. Always validate final published numbers against the official implementation for the exact checkpoint you are using.

### Configuration-driven benchmarks

```yaml
# actfold/configs/real_model_example.yaml
model_name_or_path: "gpt2"
model_family: "causal_lm"
use_real_eval: true
eval_backend: "auto"
eval_limit: 10
eval_batch_size: 1

# Advanced ActFold switches
use_stability_profiler: true
use_chunked_cache: false
use_cost_model: true
use_folded_generation: true
```

```python
from actfold.eval.benchmark_runner import BenchmarkRunner
from actfold.utils.config_manager import load_config

config = load_config("actfold/configs/real_model_example.yaml")
runner = BenchmarkRunner(config)
results = runner.run(tasks=["gsm8k", "math"], num_samples=10)
```

---

## Benchmarks

ActFold uses real evaluation backends:

| Task | Dataset | Metric | Backend |
|------|---------|--------|---------|
| Mathematical Reasoning | GSM8K, MATH | Accuracy | `lm-eval` |
| Code Generation | HumanEval+, MBPP+ | pass@1 | `evalplus` (Unix-like platforms) |
| Instruction Following | IFEval | Prompt-level accuracy | `lm-eval` |

### Expected targets

| Model | TFLOPs Reduction | Accuracy Drop | Speedup |
|-------|------------------|---------------|---------|
| Fast-dLLM-v2-1.5B | 35–50% | ≤1% | 1.2–1.5x |
| Fast-dLLM-v2-7B | 40–55% | ≤1.5% | 1.3–1.6x |
| LLaDA-8B | 45–62% | ≤2% | 1.4–1.8x |
| Dream-7B | 38–52% | ≤1.5% | 1.3–1.6x |

> These are project targets. Reproducing them requires the corresponding model weights and real `lm-eval` / `evalplus` backends.

### Platform note

`evalplus` executes generated code in a sandbox that requires Unix-like platform support (the `resource` module). On Windows, use `lm-eval` tasks or run inside WSL.

---

## Results & Figures

The figures below are example outputs generated from a small synthetic model for documentation illustration. Replace them with figures from real model benchmarks for publication.

### Layer-Token Similarity Heatmap

![Similarity heatmap](figures/fig1_similarity_heatmap.png)

*Figure 1: Cosine similarity between parent and child hidden states across layers and token positions. Brighter regions indicate stable tokens that can reuse parent activations.*

### Speedup vs. Accuracy Pareto Frontier

![Pareto frontier](figures/fig2_pareto_frontier.png)

*Figure 2: Pareto frontier of wall-clock speedup versus accuracy drop as the similarity threshold τ varies. Higher and further to the right is better.*

### TFLOPs Reduction by Model

![TFLOPs reduction](figures/fig3_tflops_reduction.png)

*Figure 3: Verification-phase TFLOPs reduction across model families. The reduction comes from reusing stable-token activations at each Transformer layer.*

### Ablation Study Summary

![Ablation table](figures/fig4_ablation_table.png)

*Figure 4: Ablation study over threshold τ, layer-wise folding strategy, and cache budget. Use these to choose a configuration for your target accuracy/latency budget.*

### Regenerating figures

```bash
# Run real benchmarks and ablations
bash scripts/run_benchmarks.sh actfold/configs/real_model_example.yaml
bash scripts/run_ablation.sh actfold/configs/real_model_example.yaml

# Generate figures from the artifacts
python scripts/generate_figures.py --results-dir results/
```

---

## Ablations

```bash
# Config-driven ablations with a real model
bash scripts/run_ablation.sh actfold/configs/real_model_example.yaml

# Quick synthetic demonstration
bash scripts/run_ablation.sh --synthetic
```

Supported studies:

1. **Threshold sensitivity**: τ ∈ {0.90, 0.95, 0.99}
2. **Layer-wise folding**: early-only, late-only, all layers
3. **Cache budget**: 256, 512, 1024, 2048 entries per layer

---

## Quality Assurance

All code is checked with:

- **black** (`line-length = 100`)
- **isort** (`profile = "black"`)
- **pyflakes**
- **mypy --strict**
- **pytest**

Run locally:

```bash
python -m black --check actfold tests demo.py scripts
python -m isort --check-only actfold tests demo.py scripts
python -m pyflakes actfold tests demo.py scripts
python -m mypy actfold --ignore-missing-imports
python -m pytest tests/ -q -m "not slow"
python demo.py
```

Tests that exercise real `lm-eval` / `evalplus` backends are marked `@pytest.mark.slow`:

```bash
python -m pytest tests/ -q -m slow
```

---

## Docs

- [`docs/ALGORITHM.md`](docs/ALGORITHM.md) — formal description of Branch Folding.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — detailed reproduction workflows.
- [`AGENTS.md`](AGENTS.md) — conventions and pitfalls for contributors and AI agents.
- [`CHANGELOG.md`](CHANGELOG.md) — release history.

---

## Current Limitations & Roadmap

1. **Diffusion samplers are high-quality reference implementations**. They closely follow the official LLaDA/MDLM, Dream, and Fast-dLLM v2 recipes, but final published numbers should still be validated against the official implementation for the exact checkpoint.
2. **No trained draft model**. `DraftGenerator` supports random/perturb/copy_flip modes, and `AdaptiveDraftGrowthController` varies branch count based on runtime stability. A dedicated draft model (e.g. Medusa/Eagle) is on the roadmap.
3. **Per-model YAML configs are templates**. You must supply the actual Hugging Face identifier or local checkpoint path.
4. **Variable-length folding is not yet supported**. Parent and child sequences must currently have the same length.
6. **Multi-ancestor reuse is not yet supported**. Folding is currently limited to a single parent branch.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: A real tokenizer is required...` | Benchmark/eval path loaded without a tokenizer. | Pass a model with a tokenizer or use `--synthetic` for debug runs. |
| `TypeError: Can't instantiate abstract class ... with abstract method embed` | A custom `DiffusionLLM` subclass is missing `embed()`. | Implement `embed(tokens)` returning `[B, T, H]`. |
| Triton kernel not used on CUDA | `triton` not installed or hidden dim not divisible by 128. | Install `triton>=2.0` on Linux/WSL; the PyTorch fallback is always correct. |
| `evalplus` fails on Windows | EvalPlus sandbox requires the Unix `resource` module. | Run evalplus tasks in WSL or use `lm-eval` tasks on native Windows. |
| Slow tests time out | Real backends load datasets and models. | Run fast tests with `pytest -m "not slow"`; run slow tests separately. |
| CI badge does not display | The workflow may not have run yet or the repo path is wrong. | Ensure `.github/workflows/ci.yml` exists and the badge URL matches `ShaneLiu04/ActFold`. |

---

## Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on code style, testing, and submitting pull requests.

---

## Citation

If you use ActFold in your research, please cite:

```bibtex
@software{actfold2025,
  title = {ActFold: Cross-Branch Activation Reuse and Branch Folding},
  author = {ShaneLiu04},
  year = {2025},
  url = {https://github.com/ShaneLiu04/ActFold},
}
```

---

## License

ActFold is released under the [MIT License](LICENSE).

---

> **Disclaimer:** This is a research codebase. Production deployment requires additional optimization, testing, and integration with the target serving framework.
