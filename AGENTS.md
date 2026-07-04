# Agent Guidance for ActFold

This file contains conventions and pointers for AI agents working on the ActFold codebase.

## Project Overview

ActFold is a research framework for reducing verification-phase FLOPs in Diffusion LLM speculative decoding. The core mechanism is **Branch Folding**: reusing parent-branch activations for "stable" tokens while recomputing "divergent" tokens.

## Key Architectural Decisions

- **Modular design**: `actfold/core/` contains the folding engine and is independent of model loading (`actfold/models/`) and evaluation (`actfold/eval/`).
- **Abstract interfaces**: `DiffusionLLM`, `DiffusionLLMAdapter`, and `ModelRegistry` make it easy to add new model families.
- **Config-driven**: `ActFoldConfig` centralizes hyperparameters. Prefer adding validated fields there rather than scattering magic values.
- **Research-first**: Synthetic models are first-class citizens for CI and rapid iteration, but evaluation always uses real backends.
- **Pluggable evaluation**: `actfold/eval/judges.py` unifies real `lm-eval` / `evalplus` backends via the `Judge` interface. There is no mock judge.

## Coding Conventions

- Python 3.10+ with `from __future__ import annotations`.
- Type hints are required for public functions and methods.
- Formatting: `black` with `line-length = 100`.
- Imports: `isort` with `profile = "black"`.
- Static analysis: `mypy --strict` with `ignore_missing_imports = true`.
- Linting: `pyflakes` for unused imports/variables.
- Docstrings: Google/NumPy style for all public classes and functions.

## Running Checks

```bash
python -m black actfold tests demo.py scripts
python -m isort actfold tests demo.py scripts
python -m pyflakes actfold tests demo.py scripts
python -m mypy actfold --ignore-missing-imports
python -m pytest tests/ -q
python demo.py
```

`lm-eval` / `evalplus` backend tests are marked `slow` and skipped by default in
CI. Run them explicitly with `python -m pytest tests/ -q -m slow` when the
benchmark dependencies are installed.

## Common Pitfalls

1. **`FoldedTransformerLayer` attention context**: Divergent tokens must be recomputed with full child hidden states to preserve self-attention context. Do **not** pass token subsets to the original layer unless the layer is token-wise (e.g., an MLP).
2. **Model loading dtype**: `CausalLMDiffusionLLM` and `GenericDiffusionLLM` default to `torch.float32` for broad compatibility. Use `torch_dtype=torch.float16` explicitly for speed on supported GPUs.
3. **Quantization device mapping**: When `load_in_8bit` or `load_in_4bit` is enabled, do not call `.to(device)` on the loaded model; bitsandbytes manages device placement.
4. **Branch dataclass**: `actfold.speculative.branch.Branch` is a lighter abstraction than `actfold.core.branch_manager.Branch`. Do not confuse the two.
5. **Config unknown keys**: `load_config` emits a `UserWarning` for unknown YAML keys but still filters them. Add new fields to `ActFoldConfig` if they need to be consumed programmatically.
6. **Evaluation backends**: ActFold only supports real `lm-eval` / `evalplus` backends. `use_real_eval` must be `True` and `eval_backend` must be one of `"auto"`, `"lm-eval"`, or `"evalplus"`. Install `requirements-bench.txt` before running benchmarks or tests. Note that `evalplus` code execution requires Unix-like platform support and does not run on native Windows.
7. **Tokenizer in evaluation**: Real evaluation adapters require a tokenizer to encode string prompts. `encode_prompt` raises `RuntimeError` if no tokenizer is provided; there is no production fallback. `decode_tokens` warns and uses a deterministic character fallback only when decoding without a tokenizer.
8. **Generation interface**: Benchmark adapters call `generate()` on the underlying `DiffusionLLM`. Tests and ablations that use raw `nn.Module` adapters pass pre-tokenized tensors directly and do not rely on string encoding fallbacks.
9. **Optional Triton kernel**: `actfold/core/fused_ops.py` provides optional CUDA acceleration for the stable/divergent merge. Code paths must remain correct on CPU and when Triton is not installed; always test the PyTorch fallback. Triton kernels require type annotations (use `Any` for Triton-specific types) and explicit `# type: ignore[untyped-decorator]` on `@triton.jit` to satisfy `mypy --strict`.
10. **No mock data as real results**: Production code must never silently fall back to mock models, random-token encoding, or synthetic metrics.  If a real component is missing (tokenizer, model checkpoint, per-layer hidden states), raise an explicit error or gate the behavior behind a clearly-named `--synthetic` / `--demo` flag.
11. **`DiffusionLLM.embed`**: All `DiffusionLLM` subclasses and `DiffusionLLMAdapter` implementations must implement `embed(tokens)` so the verification engine can populate the parent cache with real embeddings instead of random matrices.
12. **`FoldingScheduler` layer disabling**: Use `disabled_layers` to implement layer-wise ablations rather than hardcoding stability assumptions.
13. **`FoldedTransformerLayer` output tuples**: Many Hugging Face layers return `(hidden_states, ...)`. `_recompute_all` takes the first tuple element; do not assume a single tensor return value.
14. **`merge_stable_divergent` device safety**: The public function moves `stable_mask` to the activation device before dispatching to the Triton or PyTorch path, so callers do not need to ensure device agreement.
15. **`FastDLLMAdapter` raw-model kwargs**: Raw `nn.Module` models have their forward signatures inspected; only accepted kwargs are forwarded. This prevents ActFold-specific arguments such as `step_idx` from breaking Hugging Face models.
16. **Verification engine parent cache scope**: `ActFoldVerificationEngine._ensure_parent_cache` stores only input embeddings at layer 0. Layer-wise FFN outputs must be populated by the model's folded forward path (e.g. `FoldedModel`) before children are verified; do not store embeddings as `ffn_out`.
17. **`FastDLLMAdapter.folded_model`**: To exercise real layer-wise folding through the verification engine, pass a `FoldedModel` via `FastDLLMAdapter(..., folded_model=folded_model)`. Without it, ActFold-specific kwargs are filtered and only the embedding-based stable ratio is measured.
18. **`FoldedModel` kwargs propagation**: `FoldedModel` forwards `branch_id` / `parent_branch_id` / `step_idx` to the base model. If the base model's `forward` does not propagate these kwargs to the wrapped layers (common for standard Hugging Face models), folding will not activate and a custom folded forward path is required.

## Adding a New Model Family

1. Subclass `DiffusionLLM` in `actfold/models/<family>.py`.
2. Register it in `actfold/models/registry.py` via `ModelRegistry.register("family_name", FamilyModel)`.
3. Add a per-model YAML config in `actfold/configs/per_model/` if applicable.
4. Add unit tests in `tests/test_models.py` or a new `tests/test_<family>.py`.

## Adding Evaluation Tasks

1. If the task is supported by `lm-eval` or `evalplus`, add its canonical name to `LMEvalAdapter.TASKS` or `EvalPlusAdapter.TASKS` and to the corresponding `Judge` implementation.
2. Update `README.md` and `docs/EXPERIMENTS.md` with the new task and backend requirements.
3. Do not add mock fallbacks; tests must run with real backends installed.

## Adding Tests

- Use the `device` fixture from `tests/conftest.py` for GPU/CPU portability.
- Use the `seed` fixture for deterministic tests.
- Keep tests small and focused; integration tests belong in `tests/test_integration.py`.
- Benchmark tests must use real backends. Install `requirements-bench.txt` in the test environment.

## Documentation

When changing user-facing behavior, update:

- `README.md`
- `docs/ALGORITHM.md` (if the algorithm changes)
- `docs/EXPERIMENTS.md` (if workflows or configs change)
- `CHANGELOG.md` (see existing format)
- `AGENTS.md` (if agent-facing conventions or pitfalls change)

## Contact

For questions, open an issue on the project repository.
