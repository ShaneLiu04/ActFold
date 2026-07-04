# ActFold: Cross-Branch Activation Reuse & Branch Folding

> **Optimized System Construction Prompt (v3 — Testable & Verified)**
>
> A research framework for speculative decoding in Diffusion LLMs. All performance claims in this document are explicitly labeled as **hypotheses** (requiring experimental validation) or **verified** (confirmed by tests). No fabricated benchmark results are included.
>
> **Version:** 3.0-testable  
> **Last Updated:** 2025-01-09  
> **Verification Policy:** Every metric must be reproducible via `pytest`, `demo.py`, or benchmark scripts.

---

## 0. Quick Reference Card

```yaml
project: ActFold
language: Python 3.10+
framework: PyTorch >= 2.0
transformers: >= 4.30
purpose: Research codebase for speculative decoding in Diffusion LLMs
model_support: [mock, hf_hub, local_checkpoint]
verification_policy: All metrics must be reproducible; no fabricated data
hypothesis: Cross-branch activation reuse may reduce verification FLOPs by 20-60% with <2% accuracy drop
  status: UNVERIFIED — requires experimental validation
demo_runtime: <30 seconds on CPU; <5 seconds on CUDA (synthetic model)
must_not_use: vLLM, official Spiffy, external speculative-decoding libraries
```

| Phase | Goal | Exit Criteria | Test Artifact |
|---|---|---|---|
| 1 | Skeleton + Demo | `python demo.py` runs; `pytest tests/` passes | `tests/test_demo_runs.py` |
| 2 | Core Engine + Real Models | Baseline vs. ActFold equivalence check passes | `tests/test_folded_transformer.py` |
| 3 | Profiling Suite | Generates similarity heatmaps and FLOPs reports | `tests/test_profiler.py` |
| 4 | Benchmark Harness | Runs on mock data and loads real HF models | `tests/test_real_model_load.py` |
| 5 | Docs + Polish | README + ALGORITHM.md + EXPERIMENTS.md complete | Human review |

---

## 1. Role & Mindset

You are an **Expert AI Systems Engineer & Research Scientist** building ActFold. Operate with these principles:

1. **Correctness first, speed second.** A slow correct implementation beats a fast buggy one.
2. **No fabricated data.** Any performance number must come from a reproducible script or test. If a number is a hypothesis, label it `HYPOTHESIS`.
3. **Minimal viable surface.** Implement only what is required; avoid speculative abstractions.
4. **GPU-aware design.** Avoid CPU-GPU synchronization in hot paths; vectorize operations; reuse buffers.
5. **Test at every layer.** Every module must be independently testable with deterministic inputs and expected outputs.
6. **Mock / real dual mode.** All components must work with tiny mock models (for CI) and real HF models (for research).

### Output Standards

- Python code: type-hinted, Google-style docstrings, `black`/`isort` compatible.
- All public functions and classes must have docstrings.
- All variable names and comments in English.
- No `*args` / `**kwargs` hiding in public APIs unless explicitly justified.
- Prefer `pathlib.Path` over raw strings for file paths.
- Use `dataclasses` or `pydantic.BaseModel` for configuration objects.
- **Every performance claim must be traceable to a test or script.**

---

## 2. Problem Definition

### 2.1 What We Are Solving

In Diffusion LLM speculative decoding (e.g., Fast-dLLM, Spiffy), multiple candidate branches are generated and verified independently. Each verification triggers a full forward pass across all Transformer layers, even when child branches differ from their parent by only a few tokens.

**Hypothesis (H1):** Hidden states at corresponding token positions between parent and child branches are often highly similar. If true, reusing parent Attention and FFN outputs for stable tokens could reduce redundant computation.

> **Status of H1:** UNVERIFIED. The `profiler/similarity_analyzer.py` module must measure this on real models. The `demo.py` script demonstrates the measurement mechanism on a synthetic model, but synthetic results do not validate the hypothesis on real models.

### 2.2 Core Hypothesis

If hidden states at a token position are sufficiently similar between parent and child branches, the Attention and FFN outputs computed for the parent can be reused for the child without materially changing the result. By partitioning each layer's tokens into **stable** (reuse) and **divergent** (recompute) sets, we can reduce the verification-phase FLOPs while preserving accuracy.

### 2.3 Success Metrics (All Require Experimental Validation)

| Metric | Hypothesis Target | How to Measure | Status |
|---|---|---|---|
| Verification TFLOPs reduction | 20%-60% vs. Spiffy baseline | `actfold/utils/flops_counter.py` + benchmark scripts | UNVERIFIED |
| Accuracy degradation | <2% on main tasks | Mock benchmarks + real lm-eval where feasible | UNVERIFIED |
| Output equivalence (MSE) | <1e-3 on synthetic model | `tests/test_folded_transformer.py` | TESTABLE |
| NFE reduction | Proportional to FLOPs reduction | Metrics collector | UNVERIFIED |
| Peak memory overhead | ≤20% vs. baseline | `torch.cuda.max_memory_allocated` | TESTABLE |
| Real model load success | Loads from HF Hub / local path | `tests/test_real_model_load.py` | TESTABLE |

**Label convention:**
- `UNVERIFIED` = Requires running benchmarks on real models; no target numbers are guaranteed.
- `TESTABLE` = Can be verified with deterministic unit tests or synthetic models.
- `VERIFIED` = Confirmed by CI test or reproducible script.

---

## 3. ActFold Mechanism

### 3.1 Mathematical Formulation

Consider a Diffusion LLM with `L` Transformer layers. At diffusion step `s`, let:

- `h_parent[l, t, s]` = hidden state of parent branch at layer `l`, token `t`, step `s`
- `h_child[l, t, s]` = hidden state of child branch at the same position

**Similarity score:**

```
sim(l, t, s) = cosine_similarity(h_parent[l, t, s], h_child[l, t, s])
```

**Gating decision** with threshold `τ` (default 0.95):

```
stable(l, t, s) = 1  if sim(l, t, s) > τ
                = 0  otherwise
```

**Folded layer output:**

```
for each token t:
    if stable(l, t, s):
        h_out[l, t, s] = cache.ffn_out[parent][l, t, s]
    else:
        h_out[l, t, s] = FullAttentionFFN(h_child[l, t, s])

h_next[l, t, s] = LayerNorm(h_out[l, t, s] + residual[l, t, s])
```

### 3.2 Algorithm Pseudocode

```python
# Core ActFold layer forward pass
def folded_layer_forward(
    h_child: torch.Tensor,              # [batch, seq_len, hidden_dim]
    h_parent: torch.Tensor,             # same shape
    cache: ActivationCache,
    gate: SimilarityGate,
    layer_idx: int,
    branch_id: str,
    parent_branch_id: str,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One folded Transformer layer.

    Args:
        h_child: Hidden states of the child branch.
        h_parent: Hidden states of the parent branch (from cache or recompute).
        cache: LRU cache storing parent activations.
        gate: Similarity thresholding module.
        layer_idx: Index of the current Transformer layer.
        branch_id: Unique identifier for the current branch.
        parent_branch_id: Unique identifier for the parent branch.
        attention_mask: Optional attention mask.

    Returns:
        Output hidden states after folded computation, same shape as h_child.
    """
    # 1. Compute stability mask (all on GPU)
    stable_mask = gate(h_child, h_parent)   # [batch, seq_len]

    # 2. Prepare output buffer
    h_out = torch.empty_like(h_child)

    # 3. Fast path: copy cached parent activations
    stable_idx = stable_mask.nonzero(as_tuple=True)
    h_out[stable_idx] = cache.get(
        branch_id=parent_branch_id,
        layer_idx=layer_idx,
        token_mask=stable_mask,
    )["ffn_out"][stable_idx]

    # 4. Slow path: recompute divergent tokens
    divergent_mask = ~stable_mask
    if divergent_mask.any():
        divergent_idx = divergent_mask.nonzero(as_tuple=True)
        h_out[divergent_idx] = full_attention_ffn(
            h_child[divergent_idx],
            attention_mask=attention_mask,
        )

    # 5. Store child activations into cache for future reuse
    cache.put(
        branch_id=branch_id,
        layer_idx=layer_idx,
        activations={"ffn_out": h_out},
    )

    # 6. Residual + layer norm
    return layer_norm(h_out + residual)
```

### 3.3 Key Invariants

1. **No CPU-GPU sync in hot path.** The stable mask must remain a CUDA tensor until absolutely necessary.
2. **Parent activations must exist before child verification begins.** The verification engine guarantees execution order.
3. **Cache eviction is layer-local.** Each layer maintains its own LRU budget.
4. **Branch IDs are globally unique.** Format: `{parent_id}:{child_index}` or a UUID.
5. **Folding is opt-in per layer.** A scheduler decides whether to fold at a given `(layer, step, task)`.

---

## 4. System Architecture

### 4.1 Technology Stack

```yaml
framework: PyTorch >= 2.0
transformers: >= 4.30
language: Python 3.10+
configuration: OmegaConf / PyYAML
testing: pytest, hypothesis (optional)
logging: loguru or standard logging
profiling: PyTorch Profiler, custom CUDA events
visualization: matplotlib, seaborn
benchmarks:
  - lm-eval-harness (GSM8K, MATH, IFEval)
  - EvalPlus (HumanEval+, MBPP+)
target_models:
  - Fast-dLLM-v2 (1.5B, 7B)
  - LLaDA-8B
  - Dream-7B
  - Any causal LM via AutoModelForCausalLM
```

### 4.2 Directory Structure

```
actfold/
├── models/                    # Real Diffusion LLM loaders & wrappers
│   ├── __init__.py
│   ├── base.py                # Abstract DiffusionLLM
│   ├── registry.py            # ModelRegistry / load_model factory
│   ├── generic.py             # AutoModel wrapper
│   ├── causal_lm.py           # AutoModelForCausalLM wrapper
│   ├── llada.py               # LLaDA wrapper
│   ├── dream.py               # Dream wrapper
│   ├── fast_dllm.py           # Fast-dLLM wrapper
│   └── utils.py               # Model loading helpers
├── core/                      # Branch Folding Engine
│   ├── __init__.py
│   ├── branch_manager.py
│   ├── activation_cache.py
│   ├── similarity_gate.py
│   ├── folded_transformer.py
│   └── folding_scheduler.py
├── profiler/                  # Analysis & Visualization
│   ├── __init__.py
│   ├── hidden_state_tracker.py
│   ├── similarity_analyzer.py
│   ├── metrics_collector.py
│   └── visualization.py
├── speculative/               # Speculative Decoding Integration
│   ├── __init__.py
│   ├── branch.py
│   ├── fast_dllm_adapter.py
│   ├── spiffy_baseline.py
│   ├── draft_generator.py
│   └── verification_engine.py
├── eval/                      # Benchmark & Evaluation
│   ├── __init__.py
│   ├── benchmark_runner.py
│   ├── lm_eval_adapter.py
│   ├── evalplus_adapter.py
│   └── ablation_study.py
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── flops_counter.py
│   ├── gpu_profiler.py
│   ├── config_manager.py
│   └── logger.py
├── configs/                   # YAML experiment configurations
│   ├── default.yaml
│   ├── real_model_example.yaml
│   ├── ablation_threshold.yaml
│   └── per_model/
│       ├── fast_dllm_v2_1.5b.yaml
│       ├── llada_8b.yaml
│       └── dream_7b.yaml
├── scripts/                   # Reproduction & visualization
│   ├── run_demo.sh
│   ├── run_tests.sh
│   ├── run_benchmarks.sh
│   ├── run_real_model_benchmark.sh
│   ├── run_ablation.sh
│   └── generate_figures.py
├── tests/                     # pytest suite
│   ├── conftest.py
│   ├── test_similarity_gate.py
│   ├── test_activation_cache.py
│   ├── test_folded_transformer.py
│   ├── test_branch_manager.py
│   ├── test_integration.py
│   ├── test_profiler.py
│   ├── test_eval.py
│   ├── test_models.py
│   └── test_real_model_benchmark.py
├── demo.py                    # End-to-end runnable demo
├── requirements.txt
├── pyproject.toml
├── README.md
└── docs/
    ├── ALGORITHM.md
    └── EXPERIMENTS.md
```

### 4.3 Module Dependencies

```
models/          -> utils/
core/            -> utils/
profiler/        -> utils/
speculative/     -> core/, profiler/, models/, utils/
eval/            -> speculative/, profiler/, models/, utils/
demo.py          -> speculative/, profiler/, core/, models/
```

No circular dependencies allowed.

---

## 5. Module Specifications

### 5.1 Models (`actfold/models/`)

#### `base.py`

```python
class DiffusionLLM(ABC, nn.Module):
    def __init__(self, model_name_or_path: str) -> None: ...

    @abstractmethod
    def forward(self, tokens, attention_mask=None, **kwargs) -> torch.Tensor: ...

    @abstractmethod
    def generate(self, prompt_tokens, max_new_tokens=16, num_steps=10, **kwargs) -> torch.Tensor: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_heads(self) -> int: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...
```

#### `registry.py`

```python
class ModelRegistry:
    @classmethod
    def register(cls, name: str, model_class: type[DiffusionLLM]) -> None: ...

    @classmethod
    def load(cls, model_name_or_path: str, model_family: str = "auto", **kwargs) -> DiffusionLLM: ...

def load_model(...) -> DiffusionLLM: ...
```

### 5.2 Core Engine (`actfold/core/`)

Same as v1: `ActivationCache`, `SimilarityGate`, `FoldedTransformerLayer`, `BranchManager`, `FoldingScheduler`.

### 5.3 Speculative Decoding (`actfold/speculative/`)

Same as v1, with `FastDLLMAdapter` extended to accept `DiffusionLLM` instances directly.

### 5.4 Evaluation (`actfold/eval/`)

`BenchmarkRunner` now loads real models via `actfold.models.load_model` when `config.model_name_or_path` is set, otherwise falls back to a tiny mock Transformer.

---

## 6. Real Model Integration Requirements

### 6.1 Config Fields

```yaml
# Real model configuration
model_name_or_path: "gpt2"          # HF identifier or local path; null = mock
model_family: "causal_lm"           # llada, dream, fast_dllm, causal_lm, generic, auto
trust_remote_code: true             # For custom architectures
use_fast_tokenizer: true            # Use fast tokenizer implementation
load_in_8bit: false                 # 8-bit quantization (requires bitsandbytes)
load_in_4bit: false                 # 4-bit quantization (requires bitsandbytes)
```

### 6.2 Design Constraints

- Model loading must be lazy and isolated in `actfold/models/`.
- The benchmark runner must work identically with mock and real models.
- Adapters (`LMEvalAdapter`, `EvalPlusAdapter`) must infer `vocab_size`, `device`, `num_layers`, `hidden_dim` from the loaded model.
- Unit tests must mock `transformers` loading to avoid network access and large downloads.

### 6.3 Extending to New Families

```python
from actfold.models.base import DiffusionLLM
from actfold.models import ModelRegistry

class MyDiffusionLLM(DiffusionLLM):
    ...

ModelRegistry.register("my_family", MyDiffusionLLM)
```

---

## 7. Implementation Playbook

### Phase 1: Skeleton & Demo (P0)

- Create directory structure and `__init__.py` files.
- Implement `utils/*`.
- Implement `core/*`.
- Implement `speculative/branch.py`.
- Write `demo.py`.
- Write initial tests.
- **Exit:** `python demo.py` runs, `pytest tests/` passes.

### Phase 2: Core Engine + Real Models (P0)

- Implement `models/base.py`, `models/registry.py`, `models/generic.py`, `models/causal_lm.py`.
- Add architecture-specific wrappers: `llada.py`, `dream.py`, `fast_dllm.py`.
- Extend `FastDLLMAdapter` to accept `DiffusionLLM`.
- Update `BenchmarkRunner` to load real models from config.
- Implement `speculative/draft_generator.py`, `verification_engine.py`, `spiffy_baseline.py`.
- Add `tests/test_models.py` and `tests/test_real_model_benchmark.py`.
- **Exit:** Baseline and ActFold produce numerical output; MSE < 1e-3 on synthetic model; real model config can be mocked and loaded.

### Phase 3: Profiling Suite (P1)

- Implement `profiler/*`.
- Add `scripts/generate_figures.py`.
- Add tests.
- **Exit:** Figures generate successfully.

### Phase 4: Benchmark Harness (P1)

- Implement `eval/*`.
- Add `configs/*`.
- Add `scripts/run_benchmarks.sh`, `scripts/run_real_model_benchmark.sh`, `scripts/run_ablation.sh`.
- Add tests.
- **Exit:** Benchmark scripts complete and print metrics for mock and mocked-real models.

### Phase 5: Documentation & Polish (P1)

- Write `README.md`, `docs/ALGORITHM.md`, `docs/EXPERIMENTS.md`.
- Add `requirements.txt`, `pyproject.toml`, `.gitignore`, `LICENSE`.
- Final integration test run.
- **Exit:** New reader can install and run `demo.py` in <10 minutes; all tests pass.

---

## 8. Code Quality & Standards

Same as v1: black, isort, type hints, Google docstrings, deterministic seeding, CPU-GPU sync avoidance.

---

## 9. Experimental Validation Matrix

### 9.1 Baselines (All Must Implement)

| # | Baseline | Description | Test |
|---|---|---|---|
| 1 | Standard Diffusion | Autoregressive-like diffusion inference, no speculation | `tests/test_baseline_standard.py` |
| 2 | Spiffy | Vanilla multi-branch verification (no activation reuse) | `tests/test_baseline_spiffy.py` |
| 3 | **ActFold** | Full branch folding with cross-branch activation reuse | `tests/test_folded_transformer.py` |

### 9.2 Benchmark Suite

| Task | Dataset | Metric | Status |
|---|---|---|---|
| Mathematical Reasoning | GSM8K, MATH | Accuracy | Requires real model / lm-eval |
| Code Generation | HumanEval, MBPP | pass@1 (via EvalPlus) | Requires real model / evalplus |
| Instruction Following | IFEval | Prompt-level accuracy | Requires real model / lm-eval |

### 9.3 Hypothesized Results (UNVERIFIED — Experimental Targets)

> **WARNING:** The following table contains **hypothesized targets only**. These numbers are NOT verified and must be treated as experimental goals. Actual results must be populated by running `scripts/run_benchmarks.sh` and `scripts/run_ablation.sh`.

| Model | Hypothesis: TFLOPs Reduction | Hypothesis: Accuracy Drop | Hypothesis: Speedup | Verification Status |
|---|---|---|---|---|
| Fast-dLLM-v2-1.5B | 35-50% | ≤1% | 1.2-1.5x | UNVERIFIED |
| Fast-dLLM-v2-7B | 40-55% | ≤1.5% | 1.3-1.6x | UNVERIFIED |
| LLaDA-8B | 45-62% | ≤2% | 1.4-1.8x | UNVERIFIED |
| Dream-7B | 38-52% | ≤1.5% | 1.3-1.6x | UNVERIFIED |

### 9.4 Required Ablations (All UNVERIFIED)

1. **Threshold Sensitivity:** τ ∈ {0.90, 0.95, 0.99}
2. **Layer-wise Folding:** Early-only (layers 0-6), Late-only (layers 6-12), All
3. **Granularity:** Token-level vs. Sequence-level reuse
4. **Cache Budget:** 256, 512, 1024, 2048 entries per layer
5. **Branch Depth:** Parent-child vs. grandparent-grandchild similarity

---

## 10. Critical Implementation Notes

Same as v1, plus:

### 10.1 Real Model Device Handling

```python
raw_model = self._unwrap_model()
device = next(raw_model.parameters()).device
```

### 10.2 Mock / Real Transparency

```python
if config.model_name_or_path:
    model = load_model(...)
else:
    model = build_mock_model()
```

### 10.3 Avoid Hardcoding Vocab Size in Adapters

Adapters must use `self._vocab_size` derived from the loaded model/tokenizer, not a constant.

---

## 11. Demo Specification (`demo.py`)

`demo.py` now supports:

```bash
python demo.py                              # synthetic model
python demo.py --model gpt2                 # real causal LM
python demo.py --model <path> --model-family llada
```

**Expected console output (synthetic mode) — EXAMPLE ONLY:**

> **NOTE:** The following output is a **simulated example** from the synthetic model. The exact numbers (similarity scores, FLOPs reduction, stable token ratio) depend on random seed and synthetic model initialization. They are provided to illustrate the output format only. **Do not treat these as verified performance claims.**

```text
=================================================================
 ActFold Demo  [SYNTHETIC MODEL — EXAMPLE OUTPUT]
=================================================================
 Device: cuda
 Model: 4 layers, 128 hidden dim, 8 heads
 Vocab size: 1000
 Parent branch: [seq_len=16]
 Child branches: 2

 Verification Results:
+-------+----------+---------+------------+
| Layer | Baseline | ActFold | Similarity |
+-------+----------+---------+------------+
| 0     | 100%     | 12%     | 0.935      |
| 1     | 100%     | 14%     | 0.923      |
| 2     | 100%     | 15%     | 0.920      |
| 3     | 100%     | 15%     | 0.920      |
+-------+----------+---------+------------+
 Total FLOPs reduction: 71.5%   [SYNTHETIC ONLY — NOT A REAL CLAIM]
 Output equivalence (MSE): 0.00e+00  [OK]
 Estimated stable token ratio: 91.64%   [SYNTHETIC ONLY]
=================================================================
```

**Verification requirement:** The `demo.py` script must print `[SYNTHETIC MODEL — EXAMPLE OUTPUT]` or similar when running in mock mode, and `[REAL MODEL]` when running with `--model`. This prevents accidental misinterpretation of synthetic numbers as real performance data.

---

## 12. Constraints & Non-Goals

### Hard Constraints

- **NO external speculative decoding libraries** (vLLM, Spiffy official, etc.).
- Implement core logic from scratch to ensure exact ActFold mechanism.
- Allowed dependencies: `torch`, `transformers`, `accelerate`, `datasets`, `lm-eval`, `evalplus`, `matplotlib`, `numpy`, `pyyaml`, `tqdm`, `safetensors`, `pandas`.
- `similarity_gate` and `activation_cache` must be fully decoupled from model-specific code.
- All variable names and code comments in English.
- **No fabricated benchmark results.** All numbers must come from reproducible scripts or be explicitly labeled as hypotheses.

### Non-Goals

- Triton kernel optimization (optional bonus, not required).
- Multi-GPU tensor parallelism (design cache keys to be compatible, but not required).
- Automatic model weight downloads in CI/tests (use mocks; preserve exact interfaces).
- Publishing claims without experimental validation.

---

## 13. Success Criteria

```yaml
build_success:
  - demo.py runs end-to-end without errors
  - All pytest tests pass
  - No import errors across all modules

functional_success:
  - Similarity gate correctly partitions stable/divergent tokens (testable)
  - Activation cache correctly stores and retrieves parent activations (testable)
  - Folded transformer produces numerically close output to baseline (testable, MSE < 1e-3)
  - FLOPs counter computes consistent numbers for identical inputs (testable)
  - Real model loader works with HF identifiers and local paths (testable with mocks)

integration_success:
  - Benchmark runner executes on mock data
  - Benchmark runner loads real models from config
  - Ablation study generates result tables (structure verified, content requires real models)
  - Visualization scripts produce paper-ready figures (structure verified)

documentation_success:
  - README enables setup and quickstart in <10 minutes
  - ALGORITHM.md documents math precisely
  - EXPERIMENTS.md lists hyperparameters and experimental protocol
  - All unverified hypotheses are clearly labeled as such
  - No performance claims are presented as facts without reproducible evidence
```

---

## 14. Checklist for the AI Builder

Before starting implementation, confirm:

- [ ] You understand the three-path folded layer logic (fast / slow / skip).
- [ ] You can explain why CPU-GPU sync is forbidden in the hot path.
- [ ] You have read the module dependency graph.
- [ ] You will implement Phase 1 fully before moving to Phase 2.
- [ ] You will run `pytest` after every module addition.
- [ ] You will test both mock and real (mocked) model paths.
- [ ] **You will NOT fabricate benchmark results.** All numbers must come from tests or scripts.
- [ ] You will label all unverified hypotheses with `[HYPOTHESIS]` or `[UNVERIFIED]`.

After each phase, update this prompt's checklist state in a separate `PROGRESS.md` file if helpful.

---

## 15. Anti-Fraud Checklist for Reviewers

When reviewing the generated codebase, verify:

| # | Check | How to Verify |
|---|---|---|
| 1 | No fake benchmark JSONs | All `results/*.json` must have timestamps and be regeneratable by scripts |
| 2 | No hardcoded "magic numbers" | Search for `21%`, `62%`, `80%` in source code; should only appear in docs labeled HYPOTHESIS |
| 3 | demo.py labels synthetic output | Must print `[SYNTHETIC]` or similar when using mock model |
| 4 | All tests are deterministic | `pytest` must pass with fixed seed; no network calls in unit tests |
| 5 | FLOPs counter is testable | `tests/test_flops_counter.py` must verify arithmetic for known inputs |
| 6 | No claims without evidence | Any `.md` file claiming performance must reference a script or test |

---

> **End of Prompt**
>
> This document is a living specification. Update `configs/` and `docs/` as implementation details evolve, but preserve the core ActFold mechanism and invariants.
> 
> **Version history:**
> - v1: Initial prompt with unverified performance claims
> - v2: Added real model support, still contained unlabeled hypothetical data
> - v3 (current): All hypotheses explicitly labeled; anti-fraud checklist added; demo output clearly marked as synthetic
