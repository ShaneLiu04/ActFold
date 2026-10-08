# ActFold 技术报告：跨分支激活复用与 Branch Folding 机制

> **版本**：v2.0（2026-10-03）。在 v1.0（2026-07-05）基础上，补充两阶段真实模型实验（诊断与优化）、四项机制优化、十类工程修复，并更新全部受影响的实现细节、限制与文件索引。

---

## 摘要

ActFold 是一个面向 **Diffusion LLM 推测解码（Speculative Decoding）** 的研究框架，核心目标是 **降低验证阶段（verification phase）的 FLOPs**。在传统的推测解码中，系统会从一个父序列（parent branch）出发，生成多个候选子序列（child branches），再对每个子序列独立执行一次完整的 Transformer 前向传播进行验证。由于子序列通常只在少数 token 上与父序列不同，这种“全部重算”的方式存在大量冗余。

ActFold 提出 **Branch Folding**：在每一层、每一个 token 上，根据父序列与子序列隐藏状态的相似度，将 token 划分为 **stable tokens**（稳定 token，可复用父分支激活）和 **divergent tokens**（发散 token，需要完整重算），然后将两部分结果融合成最终输出。通过这一机制，ActFold 目标在 21%–62% 的验证 FLOPs 降低的同时保持极小的精度损失。

本报告从系统架构、核心算法、关键模块实现、优化组件、实验配置与限制等角度，对 ActFold 进行完整的技术解读，供学习与复现参考。

v2.0 的关键更新：在 Fast-dLLM-v2-1.5B、LLaDA-8B-Instruct 与 Dream-7B-Instruct 上完成两阶段受控实验。诊断阶段确认机制在算法层精确（自折叠与全发散前向与基线逐位一致，跨步稳定率 0.94–0.99），但原实现受缓存数据路径与"整层重算"语义限制而无墙钟收益（单层缓存读取 0.65–0.70 ms，超过原始层计算 0.53–0.61 ms）。优化阶段实现四项机制：向量化激活缓存（全稳定折叠前向 51–53 ms → 6.4–6.8 ms，较不折叠基线加速 2.3–2.7 倍）、层内 FFN 拆分（batch×seq≥512 时逐层 1.1–1.93 倍）、自适应分位数门控（精确命中目标稳定率并改善保真度前沿）、融合 gather-select 核（T=8192 时 3.43 倍，数值逐位一致）。为在三个真实模型族上端到端运行，共修复十类工程缺陷，回归测试 204 项通过。

---

## 1. 研究背景与动机

### 1.1 Diffusion LLM 与推测解码

近年来，Diffusion LLM（如 LLaDA、Dream、Fast-dLLM 等）逐渐成为自回归语言模型之外的重要生成范式。这类模型通常通过多步去噪（denoising）过程生成文本：从噪声或掩码序列出发，迭代调用模型预测更干净的版本，最终得到完整序列。

为了加速生成，**推测解码（Speculative Decoding）** 被广泛应用：
- 用一个轻量的 draft model 或多个候选分支快速生成候选 token；
- 再用目标大模型（target model）并行验证这些候选；
- 接受与目标模型一致的 token，拒绝不一致的部分并重新生成。

验证阶段通常是大模型推理的主要开销。ActFold 关注的核心问题就是：**当候选子序列与父序列高度相似时，能否避免完整重算？**

### 1.2 激活复用的直觉

Transformer 的前向传播可以逐层表示为：

```
h^(l+1) = Layer(l)(h^(l))
```

其中 `h^(l)` 是第 `l` 层的输入隐藏状态。若父分支与子分支在第 `l` 层的某些 token 上隐藏状态非常接近，那么这些 token 经过相同 Transformer 层后的输出也几乎相同。因此，可以直接复用父分支缓存下来的 FFN/Attention 输出，而不必重新计算。

这种复用的前提是：
1. 父分支的逐层激活已被缓存；
2. 子分支与父分支的相似度可以在运行时高效度量；
3. 对于不相似的 token，必须完整重算以保证注意力上下文一致。

ActFold 围绕这三个前提设计了完整的系统。

---

## 2. 核心概念与形式化定义

### 2.1 符号约定

参考 `docs/ALGORITHM.md` 中的形式化描述，设模型共有 `L` 层 Transformer，在扩散步骤 `s` 下：

- `h_parent[l, t, s]`：父分支在第 `l` 层、第 `t` 个 token 位置的隐藏状态；
- `h_child[l, t, s]`：子分支在对应位置的隐藏状态；
- `τ`：相似度阈值（默认 0.95）；
- `d`：隐藏维度。

所有隐藏状态均为 `ℝ^d` 中的实向量。

### 2.2 相似度分数

对每一个 `(l, t, s)` 三元组，计算父、子隐藏状态的余弦相似度：

```
                    h_parent · h_child
sim(l,t,s) = ───────────────────────────────
             ||h_parent|| · ||h_child||
```

在实现中，使用 PyTorch 的批量形式：

```python
sim = F.cosine_similarity(h_child, h_parent, dim=-1)  # [batch, seq_len]
```

`SimilarityGate`（`actfold/core/similarity_gate.py`）支持三种度量方式：
- `cosine`：余弦相似度；
- `l2`：负的归一化 L2 距离，阈值兼容性调整；
- `pearson`：Pearson 相关系数。

### 2.3 Token 级门控

基于阈值 `τ`，定义稳定 mask：

```
stable(l,t,s) = 1  if sim(l,t,s) > τ
                0  otherwise
```

- **Stable token**：复用父分支缓存的 Attention + FFN 输出；
- **Divergent token**：使用完整的子分支隐藏状态重新计算该层。

这种门控是 **逐层、逐 token、逐步** 的，因此即使两个分支在多处不同，仍然可以在相似的位置获得加速。

---

## 3. 系统架构总览

ActFold 的代码组织清晰，各模块职责分离：

```
actfold/
├── core/           # Branch Folding 引擎：cache、gate、folded layer、scheduler
│                   # 新增：vectorized_cache.py、split_layer.py、adaptive_gate.py
├── models/         # Diffusion LLM 包装器与原生采样器
├── speculative/    # 推测解码：draft generator、verification engine、folded generation
├── eval/           # 基准评测：lm-eval / evalplus 适配器、judge
├── profiler/       # 稳定性分析器、相似度分析、GPU 指标
├── utils/          # 配置、FLOPs 计数器、成本模型、日志
├── configs/        # YAML 实验配置
└── scripts/        # 实验与图表脚本（algo_experiments、overhead_bench、opt1–opt4、图表生成）
```

整体数据流如下图所示（来自 README）：

```
Diffusion LLM
     │
     ▼
Draft Generator  ──▶  Parent Branch
                         │
                         ▼
              ActFold Verification
              ┌─────────────────────┐
              │   Similarity Gate   │
              └──────────┬──────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
      Stable Tokens   Divergent Tokens
           │             │
           ▼             ▼
    Activation Cache   Full Layer Recompute
           │             │
           └─────────────┬─────────────┘
                         ▼
                   Merged Output
                         │
                         ▼
                   Accepted Branch
```

关键抽象接口：
- `DiffusionLLM`（`actfold/models/base.py`）：所有模型族的抽象基类；
- `DiffusionLLMAdapter` / `FastDLLMAdapter`（`actfold/speculative/fast_dllm_adapter.py`）：统一的模型适配接口；
- `ActivationCacheType`（`actfold/core/cache_factory.py`）：缓存抽象（逐 token / 分块 / 向量化三种实现）；
- `VectorizedActivationCache`（`actfold/core/vectorized_cache.py`）：连续缓冲缓存，v2.0 默认推荐；
- `SplitFoldedTransformerLayer` / `SplitSpec`（`actfold/core/split_layer.py`）：层内 FFN 拆分；
- `AdaptiveQuantileGate`（`actfold/core/adaptive_gate.py`）：按相似度秩分配复用预算的门控；
- `Judge`（`actfold/eval/judges.py`）：评测后端统一接口。

---

## 4. 核心模块详解

### 4.1 Branch Folding 引擎（`actfold/core/`）

#### 4.1.1 `SimilarityGate`—— token 级相似门控

`SimilarityGate` 是一个轻量的 `nn.Module`，核心接口为：

```python
stable_mask = gate(h_child, h_parent)  # [batch, seq_len], bool
```

它会在前向传播时：
1. 验证输入为 3-D `[B, T, H]` 且形状一致；
2. 将父分支隐藏状态对齐到子分支的设备与数据类型；
3. 根据指定 metric 计算相似度；
4. 返回 `sim > τ` 的布尔 mask。

支持运行时调整阈值：`gate.set_tau(tau)`。v2.0 起余弦相似度在返回前被裁剪到 `[-1, 1]`：bf16 下相同向量的余弦可略大于 1，若不裁剪，τ=1.0 时会出现假稳定（实测约 3.7% token 被误判）。裁剪后 τ=1.0 严格等价于全量重算，实验中的"全发散"不变式才成立。

#### 4.1.2 `ActivationCache` 与 `ChunkedActivationCache`——激活缓存

`ActivationCache`（`actfold/core/activation_cache.py`）采用 **逐 token 的 LRU 缓存**：

- 键为 `(branch_id, layer_idx, token_idx, step_idx)`；
- 值为 `{"hidden_states": Tensor, "ffn_out": Tensor}`；
- 每层独立维护一个 `OrderedDict`，超过 `max_entries_per_layer` 时淘汰最旧的条目；
- 分支被拒绝时，可调用 `clear_branch(branch_id)` 立即释放其所有缓存。

`ChunkedActivationCache`（`actfold/core/chunked_cache.py`）是其 **内存优化版本**：
- 将激活按 chunk（默认 64 个 token）连续存储；
- 减少 Python dict 的逐 token 开销，更适合长序列；
- 与 `ActivationCache` 拥有完全相同的公共 API，可通过 `cache_factory.make_activation_cache(..., use_chunked=True)` 切换。
- v2.0 修复了其 `get` 的合约缺陷：原实现返回**仅含被选中位置**的 `[B, K, H]` 张量（调用方约定为 `[B, T, H]`），且空缓存返回 `{}`。现在与 `ActivationCache` 一致：返回全长张量、未请求位置零填充、空缓存抛出 `KeyError`。

`VectorizedActivationCache`（`actfold/core/vectorized_cache.py`）是 v2.0 引入的高性能实现：
- 每个 `(branch, step, layer, 激活名)` 维护连续缓冲 `[cap, B, H]` 与写入计数，环形布局支持 `T > cap` 的"保留最新 token"语义；
- 写入为一次切片拷贝（`T > cap` 时按行号取模的 `index_copy_`）；读取在全命中时返回转置**视图**（零拷贝），部分命中时用"克隆 + 逐 batch 掩码置零"路径；
- 微基准（序列长度 34→512）显示 put 比逐 token 缓存快 12–183 倍、get 快 30–429 倍；T=512/H=4096 的部分命中读取由 0.174 ms 降至 0.058 ms；
- 通过 `make_activation_cache(..., use_vectorized=True)` 或 `ActFoldConfig.use_vectorized_cache` 启用。

`fused_ops.gather_cached_activations` 提供了向量化 gather：当缓存密集时通过 `torch.stack` 快速重建张量；稀疏时回退到循环实现。v2.0 还修复了首 token 被驱逐后无法复用的问题：现在从任意可用条目取样、缺失位置视为发散，缓存预算小于序列长度时可获得渐进的部分复用。

#### 4.1.3 `FoldedTransformerLayer`——折叠 Transformer 层

这是 Branch Folding 的 **核心实现**（`actfold/core/folded_transformer.py`）。它包装一个原始 Transformer 层，前向逻辑如下：

1. **解析分支上下文**：从显式 `branch_id`/`parent_branch_id`/`step_idx` 或线程本地 `FOLDING_CONTEXT` 读取；
2. **无父分支 / 调度器禁用 / 缓存缺失**：直接走完整前向 `_recompute_all`，并缓存当前层输出；
3. **计算稳定 mask**：使用 `SimilarityGate` 比较父、子隐藏状态；
4. **记录稳定性**：调用 `GLOBAL_STABILITY_PROFILER.record(...)`；
5. **全稳定**：直接复用父分支 `ffn_out`；
6. **全不稳定**：完整重算；
7. **混合情况**：
   - 对稳定 token 从缓存读取父 `ffn_out`；
   - 对完整子序列调用 `_recompute_all`（**必须全序列重算以保证 self-attention 上下文**）；
   - 使用 `merge_stable_divergent` 融合两者。

关键实现细节（v2.0 更新）：
- `_recompute_all` 会过滤掉 ActFold 专用 kwarg，并只传递原始层 `forward` 接受的参数；参数签名在首次调用时缓存（`inspect.signature` 每层每次调用的开销被消除）；
- 原始层返回 tuple 时，`_recompute_all` 取第一个元素，同时记录"该层返回元组"；折叠输出通过 `_pack_output` 重新包装为 `(hidden, None)`，保持与原层相同的元数。LLaDA 的 block 返回 `(hidden, cache)` 并由基座解包，若不保留元数会在真实模型上直接失败；
- 慢路径通过扩展点 `_recompute_merged(hidden_states, attention_mask, stable_mask, **kwargs)` 调用重算，默认实现忽略掩码并整层重算；`SplitFoldedTransformerLayer` 覆写该方法以只对发散行计算 FFN（见 6.7 节）；
- 子分支输出会被缓存，供后续作为父分支复用。

#### 4.1.4 `FoldedModel`——模型级包装

`FoldedModel`（`actfold/core/model_wrapper.py`）把现有 Hugging Face 风格的 Transformer 模型包装成支持 Branch Folding 的模型：

1. 自动探测常见的层列表路径：`layers`、`model.layers`、`transformer.h`、`transformer.blocks`、`model.transformer.blocks`（LLaDA）、`encoder.layer`、`gpt_neox.layers`、`model.decoder.layers` 等；
2. 将每一层替换为 `FoldedTransformerLayer`；当 `split_layers=True` 时改用 `SplitFoldedTransformerLayer`，并按 `split_min_tokens`（默认 512）自动决定是否实际启用拆分；
3. 在 `forward` 中通过 `folding_scope` 把 `branch_id`/`parent_branch_id`/`step_idx` 写入线程本地上下文，解决大多数 base model 不会转发任意 kwargs 的问题；同时预先检查 base model 的签名，只在接受这些参数时才通过 kwargs 传递，避免以异常回退作为正常路径；
4. 若 base model 拒绝 ActFold kwargs，则回退到普通 forward，但线程上下文仍然生效，折叠层仍可读取分支信息；
5. 返回值统一经 `_unwrap_output` 处理：兼容 `ModelOutput` 数据类（取 `logits`）、`last_hidden_state` 与元组首元素，保证上层（验证引擎、`folded_generate`）始终拿到张量。

`restore()` 方法可恢复原始模型层。

#### 4.1.5 `FoldingScheduler`——动态阈值调度

`FoldingScheduler`（`actfold/core/folding_scheduler.py`）实现按层、按扩散步、按任务类型动态调整 `τ`：

```
τ(l, s, task) = τ_base + bias_layer(l) + bias_step(s) + bias_task(task)
```

启发式规则：
- 早期层更稳定 → 提高 `τ`（更多复用）；
- 早期扩散步不确定性高 → 降低 `τ`（更保守）；
- 数学任务 → 降低 `τ`（精度优先）；
- 代码任务 → 提高 `τ`（容忍更多复用）。

最终 `τ` 被钳制在 `[0.80, 0.99]`。`should_fold(layer, step)` 还会默认禁用最后一层和最后一步的折叠，以避免边界误差。

#### 4.1.6 `fused_ops.py`——融合 CUDA 核函数

`merge_stable_divergent` 是稳定/发散融合的公共接口：

```python
h_out = merge_stable_divergent(parent_ffn, child_out, stable_mask)
```

当 Triton 可用且张量位于 CUDA、hidden_dim % 128 == 0、dtype 为 fp32/fp16/bf16 时，会启动一维 element-wise Triton kernel；否则自动回退到 `torch.where` 的 PyTorch 实现。两者数值结果一致。

v2.0 的兼容性修复：Triton 3.x（随 PyTorch 2.8 提供）拒绝在 `@triton.jit` kernel 内使用 `typing.Any` 注解，会以 `NameError` 直接编译失败；原实现因此在新版环境完全不可用。现在 kernel 指针参数不加注解、仅保留 `tl.constexpr` 块大小，并在编译或启动失败时**永久回退**到 PyTorch 路径。

v2.0 新增 `gather_select(parent_buffer, rows, child_out, stable_mask)`：单 kernel 内按 token 解析"稳定→按行号读取父缓冲；发散→拷贝子张量"，消除 `index_select` 的中间张量与独立的 `where` 遍历。数值与 PyTorch 路径逐位一致（最大绝对误差 0）；T=8192/H=4096 时加速 3.43 倍，T=128 时因启动开销为 0.64 倍，因此仅在大形状启用（见 6.9 节）。

### 4.2 推测解码模块（`actfold/speculative/`）

#### 4.2.1 `Branch` 与 `BranchTree`

- `Branch`（`actfold/speculative/branch.py`）：轻量候选轨迹，只记录 `branch_id`、`parent_id`、`tokens`、`scores`、`accepted`、`metadata`。
- `BranchNode` / `BranchTree`（`actfold/speculative/branch_tree.py`）：端到端折叠生成使用的树形结构，支持 add、prune、ancestor 查询。

> 注意：`actfold.core.branch_manager.Branch` 是更重的内部结构，保存完整 `hidden_states`，不要与轻量的 `speculative.branch.Branch` 混淆。

#### 4.2.2 `DraftGenerator`——候选分支生成器

`DraftGenerator`（`actfold/speculative/draft_generator.py`）在没有训练 draft model 时提供三种研究模式：
- `random`：随机采样 token；
- `perturb`：随机扰动（当前实现为随机 token，因为输入是离散的 token ID）；
- `copy_flip`：复制父分支并随机翻转少量 token。

`AdaptiveDraftGrowthController`（`actfold/speculative/adaptive_draft_controller.py`）在此基础上根据运行时稳定性与近期接受率动态决定每步生成的候选分支数：

```python
if stable_ratio >= threshold and avg_acceptance >= threshold:
    num_branches = scale_with_acceptance(...)
else:
    num_branches = 1
```

#### 4.2.3 `ActFoldVerificationEngine`——验证引擎

`ActFoldVerificationEngine`（`actfold/speculative/verification_engine.py`）负责验证子分支：

1. 用 `_ensure_parent_cache` 将父分支的输入嵌入缓存到 layer 0；
2. 若 adapter 带有 `FoldedModel`，先让父分支走一次折叠 forward，填充各层 FFN 缓存；
3. 子分支调用 `model.forward(..., branch_id=..., parent_branch_id=..., step_idx=...)`；
4. 估计 stable ratio：优先使用 `StabilityProfiler` 记录的逐层真实稳定率，否则回退到 layer 0 嵌入相似度；
5. 用 `flops_counter` 估算 TFLOPs；
6. 用 `ComputeBandwidthCostModel` 估算 wall-clock latency；
7. 若 stable ratio 低于 `acceptance_threshold`，拒绝该分支并清空其缓存。

返回的 `VerificationResult` 包含 `accepted`、`stable_ratio`、`tflops`、`latency_ms`、`estimated_latency_ms`、`stability_profile`。

#### 4.2.4 `folded_generate`——真正的端到端折叠生成

`folded_generate`（`actfold/speculative/folded_generation.py`）是 ActFold 的关键创新之一。传统的 `greedy_generate` 在每一步只是普通前向传播，而 `folded_generate` 在生成 **每一个新 token** 时都会构造一个子分支，并通过 `FoldedModel` 执行折叠 forward：

```python
for token_idx in range(max_new_tokens):
    candidates = _make_candidates(parent, ...)
    for node in candidates:
        _run_folded_forward(model, node, folded_model, step_idx)
    accepted = policy.select(candidates)
    prune rejected siblings
```

这样，benchmark 中的每个生成都实际经过了折叠路径，测得的 stable ratio 是真实的。`_run_folded_forward` 会重置该分支的 stability profile，因此 profile 只反映当前 forward。

返回 `FoldedGenerationResult`，包括最终 token 序列、平均 stable ratio、折叠步数、最终分支 ID。

#### 4.2.5 `SpiffyBaseline`

`SpiffyBaseline`（`actfold/speculative/spiffy_baseline.py`）实现了一个无激活复用的多分支推测解码基线，用于对比实验。

### 4.3 模型抽象与扩散采样器（`actfold/models/`）

#### 4.3.1 `DiffusionLLM` 抽象基类

`DiffusionLLM`（`actfold/models/base.py`）是所有模型族的抽象接口，必须实现：
- `forward(tokens, attention_mask=None, **kwargs)`；
- `embed(tokens)`：返回 `[B, T, H]` 输入嵌入；
- `num_layers`、`hidden_dim`、`num_heads`、`vocab_size` 属性。

`generate()` 方法：
- 当 `num_steps == 1` 时走默认贪心自回归生成；
- 当 `num_steps > 1` 时调用 `get_native_sampler()` 返回的扩散采样器。

#### 4.3.2 模型族实现

项目包含以下模型包装器与采样基础设施（位于 `actfold/models/`）：
- `causal_lm.py`：通用因果语言模型（如 GPT-2），可作为轻量占位；
- `llada.py` / `llada_sampler.py`：LLaDA 掩码扩散模型；
- `dream.py` / `dream_sampler.py`：Dream 连续扩散模型；
- `fast_dllm.py` / `fast_dllm_sampler.py`：Fast-dLLM 离散扩散模型；
- `generic.py`：通用 Diffusion LLM 包装器；
- `sampling_utils.py`：采样公共工具（masking schedule、`get_num_transfer_tokens`、Gumbel-Max、top-p/top-k、canvas 构建、logit 右移等）。

`ModelRegistry`（`actfold/models/registry.py`）负责按 `model_family` 字符串分派到对应类。

#### 4.3.3 `DiffusionSampler` 框架

`DiffusionSampler`（`actfold/models/diffusion_sampler.py`）定义了扩散采样的抽象接口：
- `initialize(prompt_ids)`：构造初始噪声/掩码状态 `x_T`；
- `denoise_step(x_t, t, branch_id, parent_branch_id, folded_model)`：单步去噪；
- `sample(prompt_ids, folded_model)`：完整循环；
- `decode_final(x)`：将最终状态解码为离散 token。

各具体采样器（均已对齐官方 recipe）：
- **LLaDASampler**：遵循 LLaDA/MDLM 官方实现。构建右填充画布，将生成区分块（`block_size`），通过 masking schedule（`LinearMaskingScheduler` / `CosineMaskingScheduler`）计算每步揭示 token 数，使用 `low_confidence` 或 `random` remasking，支持 CFG、temperature/top-p/top-k 与 Gumbel-Max 噪声。
- **DreamSampler**：遵循 Dream 官方实现。构建左填充画布，计算 1-D `position_ids`，采用 MaskGIT 风格的迭代解码，支持 `maskgit_plus`、`topk_margin`、`entropy` 三种置信度规则，可选 CFG 与 `alg_temp` 软选择。
- **FastDLLMSampler**：遵循 Fast-dLLM v2 官方实现。按 block 与 small-block 进行掩码扩散，基于 `threshold` 置信度阈值逐步 unmask，支持 top-p/temperature 采样、stop token 提前终止，以及自回归方式扩展新 block。

> 这些采样器已紧密对齐官方 recipe，但在发表最终结果前，仍建议用目标 checkpoint 的官方实现进行校验。

v2.0 修复了采样器中三个影响真实运行的问题：

1. **跨步折叠链缺失**：三个采样器此前在每次前向时都传 `parent_branch_id=None`，因此"跨去噪步复用激活"从未真正发生，只写缓存。现在通过 `_next_folding_branch()` 让每次前向以紧邻的前一次前向为父分支，链式传递分支标识；Fast-dLLM 的折叠采样因此与基线逐 token 完全一致（LLaDA/Dream 的 token 匹配率为 0.71–0.73，见 8.6 节）。
2. **mask token 解析**：真实 checkpoint 对掩码 token 的命名不同（LLaDA `<|mdm_mask|>`、Fast-dLLM `|<MASK>|`、Dream `<|mask|>`），而 tokenizer 未必暴露 `mask_token_id`。`_resolve_mask_token_id` 统一按 tokenizer 属性、模型配置、常见 token 拼写依次解析。
3. **Dream 掩码类型**：Dream 的注意力要求 bool/float 掩码，左填充画布生成的 long 掩码会触发 SDPA 类型错误；现在在采样器内规范为 bool。

此外，`DiffusionLLM.forward` 与 `CausalLMDiffusionLLM.forward` 会在调用原始模型前剥离 ActFold 专用 kwarg，避免分支标识泄漏到不接受 `**kwargs` 的 remote code 模型（LLaDA 即属此类）。

### 4.4 评测与基准（`actfold/eval/`）

#### 4.4.1 `Judge` 统一接口

`Judge`（`actfold/eval/judges.py`）抽象两个操作：
- `get_prompts(task, limit)`：返回提示与参考答案；
- `score(task, predictions, references)`：评分。

实现：
- `LMEvalJudge`：基于 `lm-eval`，支持 `gsm8k`、`math`、`ifeval`；
- `EvalPlusJudge`：基于 `evalplus`，支持 `humaneval_plus`、`mbpp_plus`，在临时目录写入 `samples.jsonl` 并调用 evaluate。

`JudgeFactory.create(task, backend, ...)` 自动根据任务选择 backend。

#### 4.4.2 `BaseEvalAdapter`、`LMEvalAdapter`、`EvalPlusAdapter`

`BaseEvalAdapter`（`actfold/eval/base_adapter.py`）封装了基准评测的通用逻辑：
- 用 `encode_prompt` 将文本 prompt 编码为 token；
- 分别生成 baseline（无 ActFold）和 ActFold 路径的预测；
- ActFold 路径优先调用 `folded_generate`，否则回退到 `greedy_generate` + 验证引擎估计 stable ratio；
- 用 `flops_counter.count_diffusion_llm_flops` 估算 TFLOPs；
- 返回 baseline/actfold 的指标与 TFLOPs。

`LMEvalAdapter` 与 `EvalPlusAdapter` 继承 `BaseEvalAdapter`，仅指定各自支持的 `TASKS` 与主指标 `_METRIC_KEY`。

#### 4.4.3 `BenchmarkRunner`

`BenchmarkRunner`（`actfold/eval/benchmark_runner.py`）是配置驱动的基准执行器：

1. 根据 `ActFoldConfig` 加载真实模型；
2. 自动构建 `FoldedModel`（若能识别层栈）；
3. 构建 `FastDLLMAdapter`、`DraftGenerator`、`SpiffyBaseline`、`ActFoldVerificationEngine`；
4. 对 `tasks` 列表中的每个任务，分派到 `LMEvalAdapter` 或 `EvalPlusAdapter`；
5. 保存结果到 `output_dir/benchmark_results.json`。

### 4.5 配置与工具（`actfold/utils/`、`actfold/configs/`）

#### 4.5.1 `ActFoldConfig`

`ActFoldConfig`（`actfold/utils/config_manager.py`）是集中式配置 dataclass，关键字段：
- 折叠参数：`tau`、`metric`、`max_entries_per_layer`、`enable_dynamic_tau`；
- 模型参数：`model_name_or_path`、`model_family`、`torch_dtype`、`device_map`、`load_in_8bit`/`load_in_4bit`；
- 评测参数：`use_real_eval`、`eval_backend`、`eval_limit` 等；
- 高级开关：`use_stability_profiler`、`use_chunked_cache`、`use_vectorized_cache`（v2.0，推荐）、`use_split_layers` 与 `split_min_tokens`（v2.0）、`cache_chunk_size`、`use_cost_model`、`use_folded_generation`、`use_adaptive_draft_growth`、`diffusion_sampler`。

`load_config(path)` 从 YAML 加载并校验。

#### 4.5.2 `flops_counter`

`count_diffusion_llm_flops`（`actfold/utils/flops_counter.py`）估算标准 Transformer 的 TFLOPs：
- Attention：`4 * L * H^2 * T`；
- FFN：`16 * L * H^2 * T`；
- Embedding：`2 * V * H * T`；
- 乘以 `(1 - reuse_ratio)` 体现 ActFold 节省。

#### 4.5.3 `cost_model`

`ComputeBandwidthCostModel`（`actfold/utils/cost_model.py`）同时考虑 **计算吞吐** 与 **内存带宽**：

```
time = compute_time + memory_time + gate_time + merge_time
```

其中：
- `compute_time`：仅 divergent token 的 Attention + FFN；
- `memory_time`：读取缓存的稳定激活、写入融合结果；
- `gate_time`：余弦相似度计算；
- `merge_time`：稳定/发散融合。

`HardwareProfile.from_device(...)` 提供保守默认值，CUDA 下为 100 TFLOPs/s 计算、600 GB/s 内存带宽。v2.0 的实测校准显示 RTX 6000D 上 FP16 矩阵乘约 136–138 TFLOPS、可达带宽约 1280 GB/s，约为默认假设的 1.4 倍与 2.1 倍；即便如此，成本模型对折叠前向的预测（约 0.02 ms/层）与实测（74–83 ms）仍相差约四个数量级，差距来自"整层重算"语义与数据路径开销，说明该模型只能作为理论上界而非延迟预测器（见 7.1 与 8.6 节）。

---

## 5. 关键算法流程

### 5.1 折叠层前向传播算法

```python
def folded_layer_forward(child_hidden, branch_id, parent_branch_id, layer_idx, step_idx):
    if parent_branch_id is None or scheduler disables folding:
        out = original_layer(child_hidden)
        cache.put(branch_id, layer_idx, {"ffn_out": out, "hidden_states": child_hidden})
        return out

    h_parent = cache.get(parent_branch_id, layer_idx, all_token_mask)["hidden_states"]
    stable_mask = gate(child_hidden, h_parent)  # [B, T]
    profiler.record(...)

    if stable_mask.all():
        out = cache.get(parent_branch_id, layer_idx, stable_mask)["ffn_out"]
    elif not stable_mask.any():
        out = original_layer(child_hidden)
    else:
        parent_ffn = cache.get(parent_branch_id, layer_idx, stable_mask)["ffn_out"]
        child_out = original_layer(child_hidden)  # 全序列重算，保证 attention 上下文
        out = merge_stable_divergent(parent_ffn, child_out, stable_mask)

    cache.put(branch_id, layer_idx, {"ffn_out": out, "hidden_states": child_hidden})
    return out
```

核心要点：
- divergent token 必须基于 **完整子序列** 重算，不能只取 token 子集；
- stable mask 全程在 GPU 上计算，避免 CPU-GPU 同步；
- 缓存的 `ffn_out` 与 `hidden_states` 分别用于结果复用和相似度比较。

### 5.2 验证引擎工作流程

```python
engine = ActFoldVerificationEngine(adapter, cache, gate, scheduler, cost_model)
result = engine.verify_branch(parent_branch, child_branch, step_idx=0)
```

内部步骤：
1. `_ensure_parent_cache(parent)`：将父分支输入嵌入写入 layer 0；
2. `_ensure_parent_layers(parent)`：若 adapter 有 `FoldedModel`，运行父 forward 填充各层缓存；
3. `adapter.forward(child.tokens, branch_id=..., parent_branch_id=..., step_idx=...)`：子分支走折叠路径；
4. `_estimate_stable_ratio`：优先使用 profiler 的逐层均值，否则回退 embedding 相似度；
5. `_estimate_tflops`：基于 `stable_ratio` 与 `flops_counter`；
6. `_estimate_latency`：基于 `cost_model`；
7. 若未通过 `acceptance_threshold`，清空子分支缓存。

### 5.3 `folded_generate` 端到端生成流程

```python
result = folded_generate(
    adapter,
    prompt_ids,
    max_new_tokens=32,
    folded_model=adapter.folded_model,
)
```

流程：
1. 以 prompt 为 root 创建 `BranchTree`；
2. 每轮生成候选 child：
   - 若有 `draft_generator` 且 `num_branches_per_step > 1`，使用 draft generator；
   - 否则基于父分支 logits 贪心取下一个 token；
3. 对每个 candidate 调用 `_run_folded_forward`，通过 `FoldedModel` 执行折叠 forward；
4. 用 `AcceptancePolicy`（默认 greedy）选择一个接受；
5. 剪枝未被接受的兄弟节点，释放缓存；
6. 返回最终序列与平均 stable ratio。

---

## 6. 优化组件

ActFold 除了核心 Branch Folding，还集成了若干研究型优化组件：

### 6.1 LASP：Layer-Aware Stability Profiler

`StabilityProfiler`（`actfold/profiler/stability_profiler.py`）全局单例 `GLOBAL_STABILITY_PROFILER`：
- 每个 `FoldedTransformerLayer` 在计算出 `stable_mask` 后调用 `record(...)`；
- 记录每层的 stable ratio、所用 `τ`、metric、发散位置；
- 验证引擎优先使用这些真实逐层数据，而不是仅比较 layer 0 嵌入；
- 支持 `set_enabled(False)` 关闭以消除开销。

### 6.2 CTAC：Chunked Tensor Activation Cache

`ChunkedActivationCache` 通过连续 tensor chunk 降低长序列下的内存碎片。配置中通过 `use_chunked_cache=True` 开启，由 `make_activation_cache` 统一构造。

### 6.3 CBAF：Compute-Bandwidth-Aware FLOPs Model

`ComputeBandwidthCostModel` 将 FLOPs 估算扩展为更接近 wall-clock latency 的估算，特别适用于内存受限场景。它还能通过 `calibrate(...)` 用实测 latency 调整计算吞吐假设。

### 6.4 TEFG：True End-to-End Folded Generation

`folded_generate` 保证每个新 token 都经过折叠路径，而非仅在验证时使用。这样 benchmark 真正测量了加速路径的性能与精度。

### 6.5 ADGC：Adaptive Draft-Growth Controller

`AdaptiveDraftGrowthController` 根据近期稳定率和接受率动态调整候选分支数量，高稳定场景下增加 speculation 深度，低稳定场景下保守回退到单分支贪心。

### 6.6 VCAC：Vectorized Activation Cache（向量化激活缓存，v2.0）

诊断实验显示原缓存是零加速的直接原因：逐 token 字典的读取（0.65–0.70 ms/层）超过原始层计算（0.53–0.61 ms/层）。VCAC 以连续缓冲替代字典：写入一次完成（`T > cap` 时环形复用以保留最新 token），读取在全命中时零拷贝返回转置视图、部分命中时走"克隆 + 掩码置零"。微基准上 put 快 12–183 倍、get 快 30–429 倍；端到端全稳定折叠前向由 51–53 ms 降至 6.4–6.8 ms，**较不折叠基线快 2.3–2.7 倍**，部分稳定路径由约 80 ms 降至 39–43 ms。这是论文宣称的 FLOPs 缩减第一次在真实模型上转化为墙钟加速。启用方式：`use_vectorized_cache: true`。

### 6.7 SFL：Split FFN Layer（层内 FFN 拆分，v2.0）

慢路径此前"只要一个 token 发散就整层重算"，计算节省无法兑现。SFL 利用注意力的上下文耦合性与 FFN 的逐 token 独立性：在 FFN 链两端模块（`post_attention_layernorm`→`mlp`，或 LLaDA 的 `ff_norm`→`ff_out`）上安装临时前/后向钩子，把链的输入切到发散行、输出散射回全长；注意力、残差与归一化仍由原层执行。由于行列独立，发散位置的输出与完整重算一致（三个模型的 MSE 与 top-1 完全相同，bf16 logits 最大差 0.16–0.28，属分块矩阵乘的舍入差异）。逐层加速随形状变化：batch×seq 为 64 时 0.7–0.8 倍，512 时 1.0–1.15 倍，2048 时 1.24–1.93 倍（1% 发散、B=4/T=2048 达 1.93 倍）。切分需要一次数据相关 `nonzero` 同步，因此以 `split_min_tokens`（默认 512）自动门控；seq=512 端到端把折叠路径在最优条件下缩短 17–18%，但仍未反超基线，剩余差距即每层固定的门控、缓存与调度开销。启用方式：`use_split_layers: true`。

### 6.8 AQG：Adaptive Quantile Gate（自适应分位数门控，v2.0）

固定阈值在测量中呈"平台 + 断崖"（τ∈[0.5,0.99] 稳定率不变），且同一 τ 在不同模型上含义不同。AQG 不设阈值，而是每次调用按相似度排名取 top-k 作为稳定集，`k = round(r × B·T)`，r 为目标稳定率。它精确命中任意目标（0.5/0.8/0.9/0.97/1.0 均验证通过），并把复用预算集中在最相似的位置。在保真度-复用前沿上全面优于固定阈值：单 token 扰动下，Fast-dLLM 相对误差 6.07e-2→4.33e-2，LLaDA 的 top-1 由 0.882 升至 0.971，Dream 相对误差由 2.20e-1 降至 8.35e-2。目标设为 1.0 会触发全稳定快速路径（约 6.5 ms），此时翻转 token 的保真度损失由任务约束决定。

### 6.9 FGS：Fused Gather-Select（融合 gather-select 核，v2.0）

稳定 token 的合并需要按行号读取父缓冲、发散 token 直接采用子张量。FGS 把 `index_select + where` 的两个算子融合为单次内存遍历：每个 token 行一个 program，读取稳定标志、解析源行指针、单遍拷贝 H 维。数值与 PyTorch 路径逐位一致（最大误差 0）。收益区间：T=128 时 0.64 倍（启动开销主导），T=2048/H=8192 时 1.50 倍，T=8192/H=4096 时 3.43 倍，T=8192/H=8192 时 1.84 倍。默认仅在大形状启用，小形状保留 PyTorch 路径。

### 6.10 工程修复清单（十类，v2.0）

| 编号 | 问题 | 修复 |
|---|---|---|
| 1 | Triton 3.x 拒绝 kernel 内类型注解，编译崩溃 | 移除注解，编译失败永久回退 PyTorch |
| 2 | transformers 5.x 透传 `load_in_8bit=False` 给 remote code | 仅在启用量化时传参 |
| 3 | 三个模型族封装不接受 `torch_dtype` 等加载参数 | 构造函数对齐基类并转发 |
| 4 | LLaDA 仅支持 `last_hidden_state` 输出 | 兼容 `logits`/`last_hidden_state`/`hidden_states` |
| 5 | 折叠包装器返回 `ModelOutput` 对象而非张量 | `_unwrap_output` 统一解包 |
| 6 | LLaDA 元组输出被折叠层压平导致基座解包失败 | `_pack_output` 保持输出元数 |
| 7 | LLaDA 层堆栈探测失败 | 补充 `model.transformer.blocks` 与 `wte` 路径 |
| 8 | bf16 余弦 >1.0 造成 τ=1.0 假稳定 | 余弦裁剪到 `[-1,1]` |
| 9 | 采样器不折叠父链、mask token 解析失败、Dream 掩码类型错误 | 跨步父链、统一 mask 解析、bool 化掩码 |
| 10 | 分块缓存 `get` 形状与空缓存语义不符合 API 合约 | 全长零填充并抛出 `KeyError` |

修复的回归覆盖：新增 `tests/test_vectorized_cache.py`、`tests/test_split_layer.py`、`tests/test_adaptive_gate.py`，并更新分块缓存、融合算子、折叠层与门控测试；快速套件 204 项通过。

---

## 7. 正确性与复杂度分析

### 7.1 复杂度

设序列长度为 `T`，稳定 token 比例为 `R`。

基线验证 FLOPs：

```
FLOPs_baseline = L · T · (F_attention + F_ffn)
```

ActFold 验证 FLOPs：

```
FLOPs_actfold = L · T · (1 - R) · (F_attention + F_ffn) + L · T · R · O(1)
```

其中 `O(1)` 为缓存读取与相似度计算，远小于 Attention/FFN。因此理论上：

```
reduction ≈ R
```

但 v2.0 的实测揭示了这一上界与墙钟之间的**可实现性差距**：原实现只要存在一个发散 token 就整层重算，且缓存读取本身超过层计算，因此 R<1 时墙钟收益为零甚至为负；只有 R=1（整层全稳定）的快速路径才真正跳过计算，而它在原缓存下仍比基线慢约三倍。测量得到的稳定率（单 token 扰动 0.94–0.98、跨步 0.98–0.99）远高于早期经验区间 0.21–0.62，说明瓶颈不在相似度而在此前的数据路径与合并语义。四项优化正是为消除这一差距而设计（见 6.6–6.9 节）。

### 7.2 正确性保证

ActFold **不保证与基线 bit-exact 等价**，原因：
- 稳定 token 直接复用父分支输出，可能引入微小数值差异；
- 阈值 `τ` 控制速度与精度的权衡。

但它提供以下保障：
- 只有相似度超过 `τ` 的 token 才会复用；
- 发散 token 完整重算，保留 self-attention 上下文；
- 输出等价性通过 baseline 与 ActFold 输出的 MSE 等指标衡量。

v2.0 在两个方向上把正确性从"设计保证"提升为"实测不变式"：自折叠（子分支等于父分支、整层走快速路径）与强制全发散（τ=1.0、余弦裁剪后）在三个模型上都与基线**逐位一致**（MSE=0、top-1=1.0）。这两组结果同时说明分歧只来自"稳定 token 复用父输出"这一设计选择，而非实现误差。低精度下的阈值行为也已被固定：余弦裁剪消除了 bf16 造成的假稳定。

### 7.3 关键不变式

- 父分支缓存来自真实前向传播，不会混入随机/合成激活；
- `FoldedTransformerLayer` 的 divergent 路径始终使用完整子序列；
- `merge_stable_divergent` 与 `gather_select` 的 Triton 与 PyTorch 路径数值一致（后者逐位一致）；
- 自折叠与强制全发散输出与基线逐位一致（三模型实测）；
- bf16 余弦被裁剪到 `[-1,1]`，τ=1.0 严格等价于全量重算；
- 分块缓存与向量化缓存均通过"全长零填充 + `KeyError`"的合约一致性测试；
- 被拒绝分支的缓存会被立即清理。

---

## 8. 实验配置与使用

### 8.1 安装

```bash
pip install -r requirements.txt
pip install -r requirements-bench.txt   # 真实评测后端
pip install -r requirements-dev.txt     # 开发工具
```

可选 Triton 加速（Linux/WSL）：

```bash
pip install triton>=2.0
```

### 8.2 快速 demo

```bash
python demo.py                              # 合成模型
python demo.py --model gpt2 --model-family causal_lm  # 真实模型结构演示
```

### 8.3 配置驱动 benchmark

```yaml
# actfold/configs/real_model_example.yaml
model_name_or_path: "gpt2"
model_family: "causal_lm"
use_real_eval: true
eval_backend: "auto"
eval_limit: 10

use_stability_profiler: true
use_chunked_cache: false
use_vectorized_cache: true   # v2.0：向量化缓存（推荐）
use_split_layers: false      # v2.0：注意力全量 + FFN 仅发散行
split_min_tokens: 512        # v2.0：batch×seq 达到该值时启用拆分
use_cost_model: true
use_folded_generation: true
```

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
```

### 8.4 消融实验

```bash
bash scripts/run_ablation.sh actfold/configs/real_model_example.yaml
bash scripts/run_ablation.sh --synthetic
```

输出包括：
- 阈值敏感性（τ ∈ {0.90, 0.95, 0.99}）；
- 层-wise folding（early / late / all）；
- 缓存大小影响（256 / 512 / 1024 / 2048）。

### 8.5 两阶段实测结果（v2.0）

**环境**：诊断阶段 NVIDIA RTX 6000D（84 GB），优化阶段 NVIDIA RTX PRO 6000 Blackwell（96 GB）；PyTorch 2.8.0+cu128、Triton 3.4、Transformers 4.53.1；模型为 Fast-dLLM-v2-1.5B（28 层/H=1536）、LLaDA-8B-Instruct（32 层/H=4096）、Dream-7B-Instruct（28 层/H=3584），bfloat16。子分支通过随机翻转 `n∈{0,1,2,4,8,16,32,128}` 个 token 构造，延迟用 CUDA Events 在 `no_grad` 下测量。

**算法不变量与快速路径（毫秒）**

| 指标 | Fast-dLLM | LLaDA-8B | Dream-7B |
|---|---|---|---|
| 自折叠 / 全发散 MSE | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| 基线全模型前向 | 16.5 | 17.6 | 16.6 |
| 原缓存全稳定折叠 | 52.1 | 51.2 | 53.7 |
| 向量化缓存全稳定折叠 | **6.45** | **6.37** | **6.81** |

**稳定性与相似度**：单 token 扰动下稳定率 0.94–0.98（逐层均值，τ=0.95），跨去噪步 0.98–0.99；相似度阈值在 [0.5,0.99] 呈平台，0.999 附近被 bf16 噪声支配。LLaDA/Dream 对单 token 扰动的 5% 分位相似度约 0.60，Fast-dLLM 为 0.96。

**根因（单层耗时，毫秒）**：原始层计算 0.538/0.533/0.606，缓存读取 0.699/0.652/0.697，缓存写入 0.228/0.190/0.230，门控约 0.066。缓存读取超过层计算；实测硬件为 136–138 TFLOPS、约 1280 GB/s，成本模型预测约 0.02 ms/层而折叠前向实测 74–83 ms。

**四项优化**

| 优化 | 关键测量 |
|---|---|
| VCAC 向量化缓存 | put 12–183×、get 30–429×；全稳定路径 2.3–2.7× 于基线；部分稳定 80→39–43 ms |
| SFL FFN 拆分 | batch×seq=2048 时 1.24–1.93×；MSE/top-1 与完整重算一致；seq=512 折叠路径最优缩短 17–18% |
| AQG 自适应门控 | 精确命中目标稳定率；前沿全面占优（Dream rel-MSE 0.220→0.083） |
| FGS 融合核 | 逐位一致；T=8192/H=4096 时 3.43× |

**跨步扩散采样（32 token/32 步）**：Fast-dLLM 折叠生成与基线 token 匹配率 1.000；LLaDA/Dream 分别为 0.727 与 0.712（稳定率均为 0.99 左右），差异来自两族对 logit 扰动的敏感度。

**边界**：以上结论成立于单卡、batch=1、bfloat16 与固定提示族；发布级结论需要多提示、多随机种子、任务级指标（GSM8K/HumanEval+）与 batch>1、seq>2048 的复测。

### 8.6 实测图表与数据索引（v2.0）

诊段图表（`figures/experiments/`）：`fig_invariants.png`、`fig_similarity_heatmap.png`、`fig_similarity_hist.png`、`fig_stable_by_layer.png`、`fig_tau_quality.png`、`fig_layer_ablation.png`、`fig_cache_budget.png`、`fig_overhead_breakdown.png`、`fig_latency_vs_prediction.png`、`fig_sampling.png`。

优化图表（`results/optimization/figures/`）：`fig_opt1_cache.png`、`fig_opt_summary.png`、`fig_opt2_shape.png`、`fig_opt2_longseq.png`、`fig_opt3_frontier.png`、`fig_opt4_fused.png`。

原始数据：诊断 `results/experiments/<model>/{results.json, similarity.npz, overhead.json}`；优化 `results/optimization/{baseline,opt1,opt2,opt3,opt4}/`。

复现命令：

```bash
python scripts/algo_experiments.py --model fastdllm --out results/experiments/fastdllm
python scripts/overhead_bench.py --model fastdllm --out results/experiments/fastdllm
python scripts/opt1_cache_bench.py --model fastdllm --out results/optimization/opt1/fastdllm
python scripts/opt2_shape_bench.py --out results/optimization/opt2/shape
python scripts/opt2_long_seq.py --model llada --out results/optimization/opt2/longseq/llada
python scripts/opt3_adaptive_bench.py --model dream --out results/optimization/opt3/dream
python scripts/opt4_fused_bench.py --out results/optimization/opt4
python scripts/make_experiment_figures.py --root results/experiments --out results/experiments/figures
python scripts/make_optimization_figures.py --root results/optimization --out results/optimization/figures
```

详细报告见仓库内 `docs/DEEP_EXPERIMENT_REPORT.md`（两阶段完整版）、`docs/OPTIMIZATION_REPORT.md`（优化专项）与 `docs/ACADEMIC_REPORT.md`（学术版）。

### 8.7 质量检查

```bash
python -m black --check actfold tests demo.py scripts
python -m isort --check-only actfold tests demo.py scripts
python -m pyflakes actfold tests demo.py scripts
python -m mypy actfold --ignore-missing-imports
python -m pytest tests/ -q -m "not slow"
python demo.py
```

v2.0 的快速套件包含 **204 项测试**，其中新增向量化缓存与 legacy/分块缓存的逐元素一致性测试、拆分层的精确性与自动阈值测试、自适应门控的目标命中测试，以及融合 gather-select 的逐位一致性测试。依赖真实 `lm-eval` / `evalplus` 后端的测试仍以 `@pytest.mark.slow` 标记并默认跳过。

---

## 9. 当前限制与未来方向

### 9.1 当前限制

1. **扩散采样器为高质量参考实现**：LLaDA、Dream、Fast-dLLM 的采样器已对齐官方 recipe，但最终发表前仍需用目标 checkpoint 的官方实现校验；
2. **部分稳定场景在本文规模下仍慢于不折叠基线**：batch=1、seq≤512 时，折叠路径的剩余开销来自每层固定的门控、缓存与调度（0.1–0.2 ms/层）。把门控、读取与合并融合为单个算子，并以静态预算替代数据相关选择，是使该场景反超基线的主要方向；
3. **优化收益与形状相关**：FFN 拆分默认仅在 `batch×seq ≥ 512` 启用，融合核仅在大 T 下优于 PyTorch 路径，两者都有自动回退；
4. **采样质量对模型敏感**：跨步折叠在 Fast-dLLM 上与基线逐 token 一致，但 LLaDA/Dream 在稳定率约 0.99 时 token 匹配率仅 0.71–0.73；这两族需要更高的 τ 或自适应目标；
5. **无训练好的 draft model**：`DraftGenerator` 只是研究用途的简单策略；
6. **模板配置**：per-model YAML 中的 `model_name_or_path` 需要用户替换为真实 checkpoint；
7. **不支持变长折叠与多祖先复用**：父/子序列长度必须相同，且只能复用直接父分支；
8. **`evalplus` 平台限制**：需要 Unix-like 环境，Windows 下需使用 WSL 或切换 `lm-eval` 任务。

已经解决、不再列为限制的历史条目：真实模型 demo 的架构适配已扩展为架构无关探测（含 LLaDA 路径）；Triton 在新版环境不可用的问题已修复并带编译失败回退；分块缓存的 API 合约问题已修复；`hidden_dim % 128 != 0` 时仍自动回退 PyTorch。

### 9.2 未来方向

- **全融合慢路径 kernel**：把门控、缓存读取与合并合并为单 kernel，消除部分稳定场景的逐层同步与多次内存往返，目标是在 1%–30% 发散区间也反超基线；
- **静态预算 top-k**：以固定数量的发散行替代数据相关 `nonzero`，扩大 FFN 拆分的适用形状；
- **闭环阈值调度**：以采样 token 一致率或任务指标为反馈调整目标稳定率；
- 实现 MABT（Multi-Ancestor Branch Tree），支持跨多祖先的激活复用；
- 支持变长序列对齐与前缀复用；
- 接入训练好的 draft model（如 Medusa/Eagle）；
- 扩展更多 Diffusion LLM 模型族与官方采样器对齐；
- 在 batch>1、seq>2048 与多任务基准上完成发布级验证。

---

## 10. 结论

ActFold 通过 **Branch Folding** 将 Diffusion LLM 推测解码中的验证阶段从“全序列重算”转变为“稳定 token 复用 + 发散 token 重算”的混合计算模式。v2.0 的实测进一步明确了这一机制的成立条件：复用本身是精确的（自折叠与全发散与基线逐位一致），稳定率也很高（0.94–0.99），但把稳定性转化为墙钟加速取决于数据路径与计算可分离性。向量化缓存消除了复用路径的固定开销，使全稳定折叠前向较不折叠基线加速 2.3–2.7 倍；层内 FFN 拆分在 batch×seq≥512 时提供 1.1–1.93 倍的逐层收益；自适应分位数门控以精确的复用预算改善了保真度前沿；融合 gather-select 核在长序列达 3.43 倍且数值一致。部分稳定场景在小规模下仍慢于基线，剩余瓶颈（每层 0.1–0.2 ms 的门控、缓存与调度开销）指向全融合慢路径内核这一明确的下一步。

设计亮点包括：

- **逐层、逐 token、逐步** 的细粒度门控，以及按秩分配预算的自适应门控；
- **真实激活缓存**，拒绝合成/随机 fallback；三种缓存实现共享同一 API 并通过一致性测试；
- **计算可分离性对齐的复用**：注意力保持完整上下文，FFN 仅计算发散行；
- **端到端折叠生成**，让 benchmark 真正测量加速路径；扩散采样器具备跨步折叠链与统一 mask 解析；
- **模块化架构**，核心引擎与模型加载、评测后端解耦；
- **严格的工程质量**：black、isort、pyflakes、mypy --strict 与 204 项 pytest 快速测试。

对于希望学习、复现或扩展 Diffusion LLM 推测解码加速的研究者与工程师，ActFold 提供了机制、证据与工程细节都完整记录的基线框架。

---

## 附录 A：术语表

| 术语 | 含义 |
|------|------|
| Branch Folding | 跨分支激活复用机制 |
| Stable token | 父/子隐藏状态相似度超过阈值、可复用父激活的 token |
| Divergent token | 需要完整重算的 token |
| Similarity Gate | 基于余弦/L2/Pearson 的 token 级门控 |
| Activation Cache | 父分支逐层激活缓存 |
| LASP | Layer-Aware Stability Profiler，逐层稳定率分析器 |
| CTAC | Chunked Tensor Activation Cache，分块激活缓存 |
| CBAF | Compute-Bandwidth-Aware FLOPs Model，计算带宽感知成本模型 |
| TEFG | True End-to-End Folded Generation，端到端折叠生成 |
| ADGC | Adaptive Draft-Growth Controller，自适应 draft 增长控制器 |
| VCAC | Vectorized Activation Cache，向量化（连续缓冲）激活缓存（v2.0） |
| SFL | Split FFN Layer，层内 FFN 拆分：注意力全量、FFN 仅发散行（v2.0） |
| AQG | Adaptive Quantile Gate，自适应分位数门控：按秩精确选取稳定集（v2.0） |
| FGS | Fused Gather-Select，融合缓存读取与稳定/发散选择的内核（v2.0） |
| Spiffy Baseline | 无激活复用的多分支推测解码基线 |

## 附录 B：核心文件索引

| 文件 | 职责 |
|------|------|
| `actfold/core/folded_transformer.py` | `FoldedTransformerLayer` 核心折叠层 |
| `actfold/core/model_wrapper.py` | `FoldedModel` 模型包装 |
| `actfold/core/similarity_gate.py` | `SimilarityGate` 门控 |
| `actfold/core/activation_cache.py` | 逐 token LRU 缓存 |
| `actfold/core/chunked_cache.py` | 分块缓存（v2.0 修复 API 合约） |
| `actfold/core/vectorized_cache.py` | 向量化连续缓冲缓存（v2.0） |
| `actfold/core/split_layer.py` | 层内 FFN 拆分层与拆分规格探测（v2.0） |
| `actfold/core/adaptive_gate.py` | 自适应分位数门控（v2.0） |
| `actfold/core/cache_factory.py` | 缓存工厂（逐 token / 分块 / 向量化） |
| `actfold/core/fused_ops.py` | Triton/PyTorch 融合 merge 与 gather-select |
| `actfold/core/folding_scheduler.py` | 动态阈值调度 |
| `actfold/speculative/verification_engine.py` | 验证引擎 |
| `actfold/speculative/folded_generation.py` | 端到端折叠生成 |
| `actfold/speculative/draft_generator.py` | 候选分支生成 |
| `actfold/speculative/adaptive_draft_controller.py` | 自适应 draft 增长 |
| `actfold/speculative/spiffy_baseline.py` | 无复用基线 |
| `actfold/models/base.py` | `DiffusionLLM` 抽象基类 |
| `actfold/models/diffusion_sampler.py` | 扩散采样器框架 |
| `actfold/models/*_sampler.py` | 各模型族参考采样器 |
| `actfold/eval/benchmark_runner.py` | 配置驱动 benchmark |
| `actfold/eval/base_adapter.py` | 评测适配器基类 |
| `actfold/eval/judges.py` | `lm-eval` / `evalplus` judge |
| `actfold/utils/config_manager.py` | `ActFoldConfig` 与 YAML 加载 |
| `actfold/utils/flops_counter.py` | FLOPs 估算 |
| `actfold/utils/cost_model.py` | 计算带宽成本模型 |
| `actfold/profiler/stability_profiler.py` | 逐层稳定性分析器 |
| `docs/ALGORITHM.md` | 算法形式化描述 |
| `docs/EXPERIMENTS.md` | 实验复现指南 |
| `docs/DEEP_EXPERIMENT_REPORT.md` | 两阶段深度实验报告（诊断 + 优化） |
| `docs/OPTIMIZATION_REPORT.md` | 优化阶段专项报告 |
| `docs/ACADEMIC_REPORT.md` | 学术论文版报告 |
| `scripts/algo_experiments.py` | 诊断实验（九组） |
| `scripts/overhead_bench.py` | 单层开销分解 |
| `scripts/opt1_cache_bench.py` … `scripts/opt4_fused_bench.py` | 四项优化的验证实验 |
| `scripts/make_experiment_figures.py` / `scripts/make_optimization_figures.py` | 图表生成 |
| `AGENTS.md` | 开发者/Agent 约定 |

---

**报告生成日期**：2026-10-03（v2.0；初版 2026-07-05）  
**基于代码版本**：main 分支（两阶段实验与四项优化完成后）  
**作者**：https://github.com/ShaneLiu04
