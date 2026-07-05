# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

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
- Tests for LASP, chunked cache, cost model, folded generation, adaptive draft growth, and diffusion samplers.

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
