# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### AR002: M4 Kernel Fusion & CUDA-Graph Verification Loop (P2-2/P2-3 sync elimination, B12 normalization)

Spec: `specs/changes/AR002-m4-graph-capture/` (srs.md / design.md / tasks.md, T001–T010 all passing). 624 passed, mypy --strict clean; demo baseline 85.5% / 2.35e-03 / 93.75% unchanged.

#### Added

- `actfold/core/split_layer.py`: `_exact_divergent_index(mask)` / `_padded_divergent_index(mask, C)` — device-side exact and fixed-capacity divergent index computation (P2-2: the `nonzero()` host sync is gone); `_recompute_merged` gains a `stable_count` kwarg so callers forward the count instead of a second readback.
- `actfold/core/fused_ops.py`: `fused_gate_mask_count` (M4a) — a single Triton kernel computing cosine similarity, the stable mask, **and** the stable count in one pass (fp32 block accumulation, `denom=max(sqrt(nc*np), eps)`, clamp, NaN→divergent; count via atomic accumulation). Dispatches for `SimilarityGate` (exact type) + cosine metric at `B*T >= 1024` (`_FUSED_GATE_MIN_TOKENS`); 6 ValueError input checks; `_TRITON_GATE_DISABLED` escape hatch; PyTorch fallback bit-exact (fp32/fp16/bf16).
- `actfold/core/cuda_graph.py`: `FoldedGraphRunner` (M4b) — static-buffer CUDA graph capture/replay for the fixed-shape folded child forward. Static buffer group: `tokens` / `parent_static[L+1]` (slot 0 = embedding) / `mask_buf[L]` / `count_buf[L]` (int32) / `child_buf[L]` / `logits`; side-stream warmup ×2 + `torch.cuda.graph` capture; replay prefetches the parent from cache on the host, zeroes counts, replays, and validates divergent budgets with the **single** allowed readback (`D == C` passes, `D == C+1` rejects). Capture preconditions raise `RuntimeError` (non-CUDA / scheduler / non-cosine gate / missing parent).
- `ManualFoldedForward`: `split_layers` / `split_min_tokens` / `use_cuda_graph` / `graph_capacity_ratio` (0 < r ≤ 1 else ValueError) — the graph path is **opt-in** and lazily captured on the first eligible child step. Full degradation matrix: shape mismatch (one-time warning, no re-capture, checked before the parent), scheduler / non-cosine gate / non-CUDA (one-time warning), non-pinned mask object or incomplete parent cache (silent eager), exceeded budget (one-time warning + eager recompute). The validated replay publishes `child_buf` clones into the cache and returns `logits.clone()`.
- `ActFoldConfig`: `use_cuda_graph=False` / `graph_capacity_ratio=0.5` (validated).
- `scripts/ar002_graph_bench.py`: BS-007 evidence benchmark (`run_bench` / `write_results` / CLI; artifact `results/optimization/ar002_graph_bench.json`). Measured on Quadro RTX 5000 (B=2, T=512, 4 layers): eager 5.083 ms/step vs graph replay 2.687 ms/step (**-47.1%**), 20/20 budget-validated steps. Kernel-launch counts are recorded only on CUPTI-capable builds; this host's `LIBKINETO_NOCUPTI` torch records zero CUDA profiler events, so counts are `null` with an explicit note (never fabricated) and the launch-reduction test self-skips there.
- `DraftGenerator`: dual local generators (cpu/dev RNG) — `generate(seed=...)` no longer touches the global RNG; identical seeds reproduce identical drafts.
- `BaseEvalAdapter._resolve_max_new_tokens`: per-task generation-length table (override > explicit > table > 256); `EvalPlusAdapter` maps `humaneval_plus`/`mbpp_plus` → 512.

#### Changed (breaking / legacy guidance)

- **`ManualFoldedForward` is the supported folded-forward path** (B12 complete): it now carries `split_layers` support, zero state_dict drift, and is bit-exact against `FoldedModel`. `AblationStudy` builds its internal stack with `ManualFoldedForward` (no in-place mutation of your model, no restore needed). `folded_generate` / `FastDLLMAdapter` accept `FoldedModel | ManualFoldedForward | None`.
- **`FoldedModel` is deprecated** (kept as legacy, behavior unchanged): it mutates the wrapped model in place (AGENTS #19) and depends on the deprecated contextvars branch context. Migrate by constructing `ManualFoldedForward(model, cache, gate, split_layers=True, ...)` and calling it with explicit `branch_id` / `parent_branch_id` kwargs — the kwargs path is bit-exact even under a poisoned contextvar, and is the only path the CUDA graph supports.
- `FoldingScheduler`-managed layers and `AdaptiveQuantileGate` are **not** graph-capturable (data-dependent host decisions); they degrade to eager with a one-time warning.

#### Known follow-ups

- Kernel-launch counts for the bench artifact require a CUPTI-capable torch build; the formal folded-vs-no-folding baseline comparison per `docs/RERUN_CHECKLIST.md` (locked clocks) remains open.
- Variable-length folding (P3) is out of scope: graph capacity is fixed at capture time (`graph_capacity_ratio`); longer divergent sets fall back to eager.

### AR001: Deep Optimization (M1 correctness / M2 sync / M3 memory / M5 methodology & portability)

Spec: `specs/changes/AR001-deep-optimization/` (srs.md / design.md / tasks.md, T001–T026 all passing).

#### Fixed (M1 — correctness, T001–T008/T027)

- Cross-step `stable_ratio` now averages over **all** diffusion steps (was last-step-only), with `final_norm` applied before the similarity comparison.
- Profiling a model that exposes **random head selection** now raises `RuntimeError` instead of silently profiling random layers; the stability profiler resets branches in `try/finally`.
- `NaN/Inf` guard moved **before** the hot path (fail fast instead of after expensive work); cosine similarity computed in **fp32**; CPU timing uses `time.perf_counter` (was `time.time`).
- Activation-cache growth is bounded: ring eviction keyed by `(branch, step)` with a `max_branch_steps` cap, so `branch_id` chains cannot grow memory without bound.
- `FoldedModel` restores the base model's original layers on exceptions (context-manager safety); draft distribution supports both `suffix_append` and `logits_draft` modes.

#### Improved (M2 — host-sync elimination, T009–T013)

- Stability profiler readback is lazy (no per-layer host sync); the folded layer's three-way stable/divergent split performs a **single** host sync via one `sum` readback.
- `AdaptiveQuantileGate` bottom-k selection is sync-free on the adaptive path.
- Legacy `ActivationCache` rewritten with contiguous per-`(branch, step)` group buffers: group-level LRU eviction under a per-layer token budget, ring-layout wrap for over-budget groups, and vectorized put/get (12–183x faster puts, 30–420x faster gets than the per-token version).
- Triton merge kernel: single-sided reads, correct strides, and tail masking; PyTorch fallback retained and bit-exact.

#### Added / Changed (M3 — memory-pass compression, T014–T018)

- `actfold/core/cache_protocol.py`: `ActivationCache` protocol (`fetch` / `get_all` / `contains` / `fetch_masked`) with three conforming implementations (legacy, chunked, vectorized); contract tests enforce the read-only zero-filled-view semantics.
- Redundant per-layer `hidden_states` cache entries removed (gate reads parent `ffn_out` at `L-1`, layer 0 reads the cached `embedding`).
- `fetch_flat` + fused `gather_select` path with a D3 threshold (T≥2048 & H≥4096, or T≥8192); PyTorch fallback for small shapes.
- `SplitFoldedTransformerLayer` uses resident module hooks + preallocated empty baseboard (no per-call hook registration).
- `get_num_transfer_tokens` fully vectorized — **bit-exact** against the official LLaDA implementation (87/87 cases); `LLaDASampler` / `DreamSampler` vectorized (per-prompt lengths, batched top-k, early stop), fixing a step-loop variable-shadowing bug.

#### Added (M5 — methodology & portability, T019–T025)

- `scripts/_hf_env.py`: `apply_hf_env` / `add_hf_env_arguments` — `--hf-endpoint` / `--hf-home` CLI flags with environment-variable fallback; **no hardcoded mirror or platform paths** (7 scripts migrated). All scripts run via `python -m scripts.<name>` (9 `sys.path` hacks removed).
- `TimingStats` (frozen dataclass: `mean/std/p50/n/as_dict`) + `stats_from_samples`; `repeat_with_seed` (per-repeat `manual_seed` with global-RNG snapshot/restore); `exp_sampling(repeats=3, seed=0)` with a `tokens_reproducible` check.
- `HardwareProfile.from_device(device, calibrate=False)`: device-name lookup table (H100/A100/RTX PRO 6000/4090/…, conservative fallback) plus optional CUDA microbenchmark calibration; cost model gains the attention quadratic term and KV-read bandwidth.
- `count_diffusion_llm_flops`: `ffn_intermediate_dim` / `ffn_type` (`mlp`|`swiglu`) / `include_attention_t2` parameters; `model_ffn_flops_kwargs` duck-typed extraction wired into all call sites.
- `ActFoldConfig.max_new_tokens` (default **256**, validated > 0), threaded through `BenchmarkRunner` into all eval adapters.
- `FoldedMeasurement` + `AblationStudy.measure_folding(tau, disabled_layers, max_entries_per_layer, seed)`: real per-layer measurements via a context-managed `FoldedModel` and the stability profiler.
- `docs/RERUN_CHECKLIST.md` + `results/**/INVALIDATED.md`: historical artifacts marked invalid with rerun instructions (environment, VRAM, commands, methodology requirements, and the list of numbers that changed).
- 82 new tests across `test_scripts_portability.py` (12), `test_experiment_stats.py` (16), `test_cost_model_device.py` (14), `test_flops_counter_ffn.py` (11), `test_eval_fixes.py` (20), `test_ablation_measured.py` (9).

#### Breaking changes

- `time_forward` returns a `TimingStats` object (callers use `.mean`); `exp_sampling` results use `baseline_ms_mean/std` + `folded_ms_mean/std` + `repeats` + `seed` + `tokens_reproducible` (old single-scalar keys removed).
- `LayerCost` fields renamed: `gate_flops`/`merge_flops` → `gate_bytes`/`merge_bytes` (bandwidth-bound, not compute-bound).
- FLOPs accounting: the **input** embedding is a table lookup and no longer counted; only the LM-head projection is. All FLOPs-reduction percentages shift up (synthetic demo baseline 78.5% → **85.5%**).
- Eval adapters: default `max_new_tokens` 16 → **256**; `BaseEvalAdapter._generate_one` signature is now `(prompt_tokens, use_actfold, seed) -> (text, ratio, latency_ms)`; `_generate_predictions` accepts pre-tokenized tensors; `_evaluate` adds `baseline_latency_ms` / `actfold_latency_ms` and canonical metric keys (`exact_match` for gsm8k/math, `prompt_level_acc` for ifeval via `LMEvalAdapter._TASK_METRIC_KEYS`); `LMEvalJudge.score` additionally returns per-doc `"metrics"` aggregates (`judges._aggregate_metric_means`).
- `AblationStudy`: layerwise results are **measured** via `FoldingScheduler.disabled_layers` (the linear extrapolation is kept only as the `linear_estimate_pct` comparison column); the cache sweep defaults to `[seq_len, 2*seq_len, 4*seq_len]` so the smallest budget forces real eviction; `draft_generator` is optional and defaults to `suffix_append`; `estimated_reduction_pct` is now a measured value. The study manages its own folded stack (the adapter must NOT carry a `folded_model` and must expose `underlying_model`).

#### Known follow-ups (AR002)

- M4 (single-kernel fusion / CUDA-graph capture) and the full B12 refactor are deferred to AR002.
- M-5 (controlled before/after timing) requires the rerun on the target GPU per `docs/RERUN_CHECKLIST.md`; `DraftGenerator.generate(seed=...)` still seeds the global RNG (measurement helpers snapshot/restore it).

#### Transport note (Gitee mirror)

- The oversized experiment figure PNGs under `figures/experiments/` and `results/*/figures/` (all pre-AR001, `INVALIDATED.md`-marked artifacts due for regeneration per `docs/RERUN_CHECKLIST.md`) were re-encoded with 256-color palette quantization (dimensions and visual content unchanged; 104–222 KB → 28–54 KB each). Reason: the corporate egress proxy blocks HTTPS request bodies above ~98 KB, so the original files cannot transit to the Gitee mirror from this network. All code, text, and data files are byte-exact.

### Added (pre-AR001)

- `constraints-bench.txt`: pins the `evalplus -> google-generativeai` gRPC/protobuf transitive stack (`grpcio`, `grpcio-status`, `googleapis-common-protos`, `proto-plus`, `protobuf`) so `pip install -r requirements-bench.txt` resolves in seconds instead of backtracking for minutes. `requirements-bench.txt` includes it via `-c`.
- CI is split into a fast `quality` job (runtime + dev dependencies only, Python 3.10/3.11 matrix, formatting/lint/mypy/fast tests/demo) and a constrained `bench-backends` job (bench backends + `-m slow` tests); both jobs have timeouts and pip caching.

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
- `actfold/models/architecture_utils.py`: architecture-agnostic detection of embedding modules, Transformer layer stacks, and language modeling heads for GPT/LLaMA/Qwen/Mistral/Gemma/OPT/BERT/RoBERTa/T5/BART/Falcon/Phi-style models.
- `ManualFoldedForward`: fallback folded path used when `FoldedModel` cannot auto-discover a layer stack.
- Extended `FoldedModel` default layer paths to cover `model.layers`, `transformer.h`, `transformer.layers`, `gpt_neox.layers`, `transformer.blocks`, `model.decoder.layers`, `decoder.layers`, `encoder.layer`, `model.encoder.layer`, `bert.encoder.layer`, `decoder.block`, `model.decoder.block`, `encoder.block`, and `model.encoder.block`.
- `demo.py` real-model path is now architecture-agnostic and supports `--dtype`, `--seq-len`, `--num-branches`, `--tau`, `--prompt`, `--num-steps`, and `--max-new-tokens`.
- Tests for architecture detection (`tests/test_architecture_utils.py`).

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
- `actfold/core/fused_ops.py`: Triton 3.x compatibility. `typing.Any` annotations inside `@triton.jit` kernels are rejected by Triton 3.4 (bundled with PyTorch 2.8); pointer parameters are now unannotated and any kernel compile/launch failure permanently falls back to the PyTorch merge.
- `gather_cached_activations`: partial reuse now works when the first token's cache entry was evicted; the sample entry is taken from the first available key instead of requiring token 0.
- `CausalLMDiffusionLLM` / `GenericDiffusionLLM`: quantization kwargs are only passed to `from_pretrained` when enabled (transformers 5.x forwards `load_in_8bit=False` to remote-code model constructors); `GenericDiffusionLLM.forward` now accepts `logits`, `last_hidden_state`, or `hidden_states` outputs; both wrappers strip ActFold branch kwargs before calling raw models.
- `LLaDAModel` / `DreamModel` / `FastDLLMModel`: constructors now accept and forward `torch_dtype`, `use_fast_tokenizer`, `device_map`, and quantization flags, matching the base wrappers and `BenchmarkRunner`.
- `FoldedModel`: unwraps Hugging Face `ModelOutput` dataclasses and tuples to a tensor (logits / last hidden state / first element); detects whether the wrapped model accepts ActFold kwargs instead of relying on a caught `TypeError`.
- `FoldedTransformerLayer`: preserves the original layer's output arity. Layers such as LLaDA blocks that return `(hidden_states, cache)` now receive `(folded_hidden, None)` from the folded path so the base model keeps unpacking correctly.
- Architecture detection: added the LLaDA paths `model.transformer.blocks` (layers) and `model.transformer.wte` (embedding) to `FoldedModel` and `actfold/models/architecture_utils.py`.
- `SimilarityGate`: cosine similarity is clamped to `[-1, 1]` so numerical noise cannot make `sim > tau` true at `tau=1.0`.
- Diffusion samplers (`LLaDASampler`, `DreamSampler`, `FastDLLMSampler`): forward passes are chained via `parent_branch_id` so cross-step Branch Folding actually activates instead of always passing `None`; unified mask-token resolution across `tokenizer.mask_token_id`, the model config, and the common checkpoint spellings (`<|mdm_mask|>`, `|<MASK>|`, `<|mask|>`).
- `DreamSampler`: the left-padded canvas attention mask is normalized to `bool`, fixing a torch SDPA dtype error (`attn_mask long` vs `query bfloat16`) for Dream checkpoints.
- `StabilityProfiler.all_profiles()` exposes all recorded branch profiles.
- Added `scripts/algo_experiments.py` (nine deep experiments), `scripts/overhead_bench.py` (per-layer overhead decomposition), and `scripts/make_experiment_figures.py` (explanatory figures).
- Added `actfold/core/vectorized_cache.py`: `VectorizedActivationCache`, a contiguous-buffer cache whose `put`/`get` are 12–183x / 30–420x faster than the per-token legacy cache; wired via `make_activation_cache(use_vectorized=...)`, `ActFoldConfig.use_vectorized_cache`, and `BenchmarkRunner`. The all-stable folded fast path drops from ~52 ms to ~6.5 ms (2.3–2.7x faster than the no-folding baseline).
- Added `actfold/core/split_layer.py`: `SplitFoldedTransformerLayer` (optimization #2). Attention stays on the full sequence while the token-wise FFN chain receives only divergent rows through temporary module hooks. Forced exactness tests and shape sweeps show 1.1–1.9x per-layer speedups once ``batch*seq >= 512``; controlled by `ActFoldConfig.use_split_layers` / `split_min_tokens`.
- Added `actfold/core/adaptive_gate.py`: `AdaptiveQuantileGate` (optimization #3) which top-k selects an exact target stable ratio per call and dominates the fixed-threshold fidelity/reuse frontier.
- Added `fused_ops.gather_select` (optimization #4): a Triton kernel fusing cached-parent gather and stable/divergent select (bit-exact, up to 3.4x at T=8192) with automatic PyTorch fallback.
- Added optimization experiment scripts: `scripts/opt1_cache_bench.py`, `opt2_split_bench.py`, `opt2_shape_bench.py`, `opt2_long_seq.py`, `opt3_adaptive_bench.py`, `opt4_fused_bench.py`, `make_optimization_figures.py`; tests: `tests/test_vectorized_cache.py`, `test_split_layer.py`, `test_adaptive_gate.py`.
- Fixed `ChunkedActivationCache.get`: it now returns full-length zero-filled tensors (was returning only the selected positions) and raises `KeyError` on empty caches, matching the `ActivationCache` contract.

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
