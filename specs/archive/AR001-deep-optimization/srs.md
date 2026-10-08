# [AR001] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR001 |
| AR 主题 | deep-optimization（正确性清零 + 同步清零 + 内存 pass 压缩 + 实验方法学与可移植性） |
| 关联 SR | 无（来源：`docs/OPTIMIZATION_GUIDE.md` 深度分析报告，2026-10-08） |
| 日期 | 2026-10-08 |
| 状态 | Draft |

## 1. 背景与目标

ActFold 是 Diffusion LLM 投机解码验证阶段的 Branch Folding 研究框架。深度分析（见 `docs/OPTIMIZATION_GUIDE.md`）发现三类系统性问题：

1. **正确性缺陷**：13 个 bug 直接污染实验数字（stable_ratio 只取末步值使 benchmark FLOPs 估算错误、draft 分布与真实投机解码脱节使 τ 敏感性实验结构性平凡、cache 跨 step 无限泄漏等）。
2. **性能瓶颈**：每层 folded 前向存在 5–7 次 host-device 同步、10–14 次全张量内存 pass（理论下限 1 次同步 + 3–4 pass），导致部分稳定 folded 前向仍慢于不折叠的 baseline（项目核心矛盾）。
3. **实验方法学缺陷**：脚本硬编码 AutoDL 路径与镜像、无方差/置信区间、cost model 硬件常数一刀切、flops_counter 不支持 SwiGLU（LLaDA/Dream 均为 SwiGLU，FFN FLOPs 低估 ~26%）。

**当前机器约束**：Quadro RTX 5000 16GB（Turing，被锁 P8 降频：fp16 实测 4.4 TFLOPS / 带宽 11 GB/s），无法运行 Dream-7B / LLaDA-8B 全精度实验，墙钟绝对值不可作为验收依据。

**本 AR 目标**：在不依赖大显存的前提下完成代码层深度优化，使项目满足：(a) 正确性 bug B1–B11 清零并有回归测试保护（B12 做轻量防护，完整重构留 AR002）；(b) 每层阻塞同步降至 ≤1 次（可异步化到 0）；(c) cache 显存与内存 pass 显著下降；(d) 实验脚本在任何高算力机器上开箱即完整复现。**M4（单 kernel gate+gather+merge 融合、CUDA graph 捕获）明确排除**，留待下一 AR。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- `actfold/core/`：folded_transformer.py、similarity_gate.py、adaptive_gate.py、activation_cache.py、vectorized_cache.py、chunked_cache.py、cache_protocol.py（新增）、fused_ops.py、split_layer.py、model_wrapper.py
  （注：branch_manager.py 本轮零改动——无任何 §3.x 需求触及，其重构属 P3 留后续 AR，见 design.md 决策 D5）
- `actfold/speculative/`：folded_generation.py、verification_engine.py、draft_generator.py
- `actfold/models/`：architecture_utils.py、generic.py、llada_sampler.py、dream_sampler.py、fast_dllm_sampler.py、sampling_utils.py
- `actfold/profiler/stability_profiler.py`
- `actfold/eval/`：base_adapter.py、benchmark_runner.py、ablation_study.py、judges.py
- `actfold/utils/`：cost_model.py、flops_counter.py、gpu_profiler.py
- `scripts/*.py`：可移植性参数化、统计方法
- `tests/*`：全部新增回归测试
- `results/`、`docs/`：作废标注与重跑清单

**Out of Scope（本 AR 不做的）：**

- M4 战略级优化：单 Triton kernel gate+gather+merge 融合、CUDA graph / torch.compile 捕获、验证循环 graph 化（留待 AR002）
- B12 完整重构：`FoldedModel` 不替换式的 `ManualFoldedForward` 常态化（state_dict key 兼容层、与 CUDA graph 的兼容设计）——与 M4 强耦合，留待 AR002；本轮仅做 B12-lite 轻量防护（见 3.1）
- 需要真实 7B/8B 模型权重的实验重跑（本机显存不足，仅准备重跑清单）
- 变量长度折叠、多祖先复用、真实 draft 模型接入、MoE 支持（P3 算法扩展）
- KV cache 与 folded 生成共存设计
- lm-eval `simple_evaluate` LM adapter 完整重构（仅做 metric key 显式映射）
- pyproject.toml 打包 bug 修复回馈上游 GitHub 仓库（本机已修复）

## 3. 功能需求

### 3.1 F1 正确性修复包（8 项 bug）

**描述：** 修复 OPTIMIZATION_GUIDE 第三部分中不影响 API 结构的 8 个正确性 bug，每项新增回归测试。

**触发条件：** 对应代码路径被执行时。

**期望行为与明细：**

| Bug ID | 位置 | 修复行为 |
|---|---|---|
| B1 | speculative/folded_generation.py:131-138 | `stable_ratio` 改为全步均值（累积 ratios 列表），`_estimate_actfold_tflops` 使用均值而非末步值 |
| B2 | models/architecture_utils.py:384-395 | `ManualFoldedForward` 补 final norm（`detect_architecture` 发现 `model.norm`/`transformer.ln_f`/`encoder.norm` 等），输出与基线 top-1 一致 |
| B3 | models/generic.py:79-84 | 无 head 时随机初始化 lm_head 改为显式 raise（符合 AGENTS.md #10），提供 `allow_random_head` 显式旗标 |
| B4 | speculative/folded_generation.py:83-84,128-129 | `record_stability=False` 的全局副作用改 try/finally（或 contextmanager） |
| B5 | models/fast_dllm_sampler.py:160-212 | 传递 attention_mask；docstring 修正（删除 KV-cache friendly 失实声明） |
| B6 | models/sampling_utils.py:284-291 | `sample_tokens` 的 `except Exception` 窄化为预期异常类型并 warn，不再静默降级 greedy |
| B7 | core/similarity_gate.py:31,86 | gate 的 eps 按 dtype 缩放（fp16≥1e-4）；dot/norm 用 fp32 累加；NaN 不进入 mask 判定 |
| B8 | profiler/stability_profiler.py:130-136；utils/gpu_profiler.py:40-42 | profiler 同步开销移出热路径（见 3.4）；CPU 路径延迟改 `time.perf_counter`，不再落盘 0 |
| B12-lite | core/model_wrapper.py:144,277-283 | B12 轻量防护：`FoldedModel` 增加 context manager 协议与 `__del__` 兜底 restore；wrap 时显式 WARNING 提示 state_dict key 漂移与"裸调用基模型将报错"（完整的不替换式重构留 AR002，与 M4 一并设计） |

**异常处理：** B3 场景必须显式报错；其余修复保持现有正常路径行为不变。

**验收标准：**
- Given 3 步 folded 生成且各步 stable_ratio 不同，When 读取 result.stable_ratio，Then 得到三步均值而非末步值
- Given LLaMA 结构模型，When 通过 ManualFoldedForward 前向，Then logits 与原模型 forward 的 top-1 一致率 100%
- Given 无 lm_head 的 AutoModel，When 构造 GenericDiffusionLLM，Then 抛出显式 RuntimeError（除非 allow_random_head=True）
- Given fp16 零向量 token，When gate 计算，Then 不产生 NaN mask
- 原 202 项测试全部保持通过，新增回归测试全部通过

### 3.2 F2 cache 生命周期修复

**描述：** VectorizedActivationCache 的 (branch_id, step_idx) buffer 无自动淘汰，folded_generation 长生成显存线性泄漏。

**触发条件：** `folded_generate` 多步生成、diffusion sampler 跨步链。

**期望行为：** cache 新增保留最近 N 个 step 的 ring 淘汰（N 可配置，默认覆盖跨步折叠链深度）；`folded_generation` 的 prune 逻辑真正调用 `clear_branch` 释放未被延续分支的 buffer。

**异常处理：** 跨步折叠链读取已被淘汰的 buffer 时，按 cache miss 处理（走 divergent 重算），不崩溃。

**验收标准：**
- Given 512 步 folded 生成、单层 cache 容量限制，When 生成结束，Then cache 持有的 buffer 数量有界（不随步数线性增长）
- Given 被淘汰分支的 token 被 gate 请求，When folded forward，Then 该 token 判为 divergent 并正确重算

### 3.3 F3 draft 分布修复

**描述：** copy_flip 全词表均匀随机翻转使任何 τ∈[0.5,1] 都判 divergent，τ 敏感性实验结构性平凡（历史 threshold_sensitivity.csv 三个 τ 结果完全相同即症状）。

**触发条件：** 生成 draft child 分支。

**期望行为：** DraftGenerator 新增两种模式：`suffix_append`（prompt 前缀不变，仅对后缀区域追加/重采样 token）与 `logits_draft`（用 parent logits top-k 采样模拟低质 draft 模型的语义近邻分歧）。benchmark_runner 与 ablation 默认 draft 模式切换为 `suffix_append`；`copy_flip` 保留用于对照。

**异常处理：** `flip_region` 参数支持限定翻转区域（排除 prompt/pad/eos）；flip_ratio=0 仍表达零分歧对照。

**验收标准：**
- Given suffix_append draft 与 τ∈{0.9,0.95,0.99}，When 阈值敏感性实验，Then 不同 τ 产生**可区分**的 stable_ratio（不再全部相同）
- Given 相同 seed，When 生成 draft，Then 结果完全可复现
- 全部现有 draft 相关测试保持通过

### 3.4 F4 同步清零包（9 项）

**描述：** 消除 folded 前向热路径的 host-device 阻塞同步，每层从 5–7 次降至 ≤1 次。

**触发条件：** 任何 folded forward。

**期望行为与明细：**

| 项 | 位置 | 行为 |
|---|---|---|
| profiler 异步化 | stability_profiler.py:130-136 | record 只在 GPU 上累加 sum/count tensor，不 `.item()`；`divergence_positions` 仅 debug 模式收集；profile 读取时一次性回读 |
| 三分支判定合并 | folded_transformer.py:165,187 | `stable_mask.all()`+`any()` 两次同步合并为一次 `sum()` 判定 |
| get_all 快路径 | vectorized_cache.py:166 + folded_transformer.py:128 | 新增无 mask 的 `get_all()`；调用方不再构造全 1 mask |
| ones mask 缓存 | folded_transformer.py:131-135 | 按 shape 缓存复用，不再每层每前向新分配 |
| signature 缓存 | folded_transformer.py:275；model_wrapper.py:209,233 | `__init__` 缓存 `inspect.signature` 结果 |
| adaptive gate 优化 | adaptive_gate.py:59-67 | topk(k≈N) 改 bottom-k(1−ratio)；`last_tau` 记录异步化，不再 `.item()` 写共享状态 |
| 两次 get 合一 | folded_transformer.py:128,230 | 单次调用同时取 hidden+ffn（配合 F6/F7） |
| hook 常驻注册 | split_layer.py:175-191 | 构造时注册，`_split_state` 守护，不再每前向注册/移除 |
| Triton 降级粒度 | fused_ops.py:143,304-311 | merge 与 gather_select 独立降级标志，按失败类别记录 |

**验收标准：**
- Given profiler 开启的 4 层 folded 前向（CUDA），When 用 torch profiler / nsys 统计 cudaStreamSynchronize/cudaMemcpy D2H 次数，Then 每层阻塞同步 ≤1 次（原为 5–7 次）
- Given profiler record 被调用 1000 次，When 期间无任何 profile 读取，Then 期间 D2H 同步次数为 0
- 原 202 测试全绿 + 新增同步语义回归测试通过

### 3.5 F5 Triton merge kernel 就地改进

**描述：** merge kernel 三处就地改进，不改 kernel 结构。

**触发条件：** CUDA 且 Triton 可用时的 merge_stable_divergent 调用。

**期望行为：** (a) 每 program 按 stable/divergent 单边读（3 pass→2 pass，对齐 `_gather_select_kernel` 已有行为）；(b) 用已传入的 `hidden_stride` 寻址，删除调用前 `.contiguous()` 双拷贝；(c) 删除 `hidden_dim % 128 != 0` 的过度保守禁用（kernel 已有 masking），非 2 的幂 hidden 也可走 Triton。

**验收标准：**
- Given 任意 [B,T,H]（含 H=3584、H 非整除 128）非连续输入，When merge，Then 与 PyTorch 参考路径 bit-exact（max abs err = 0）
- Given 非连续 transpose view 输入，When merge，Then 不发生调用方侧 contiguous 拷贝（张量分配计数不增加）

### 3.6 F6 冗余 hidden_states 缓存删除

**描述：** cache 同时存 layer 输入（"hidden_states"）与输出（"ffn_out"），残差流上二者跨层恒等（layer L 的输入 = layer L−1 的输出），约 2× 冗余。

**触发条件：** folded forward 的 `_store_activations`。

**期望行为：** 只存 `ffn_out`，另在 layer 0 单独存 embedding；gate 在 layer L 读取父分支 `ffn_out[L-1]` 作为 parent hidden；跨层读取偏移与 fast-path 边界（layer 0 用 embedding）正确处理。

**验收标准：**
- Given 相同输入与折叠配置，When F6 前后分别跑 folded forward，Then 输出 logits 数值等价（MSE ≤ 原 fast-path 容差）
- Given 单分支 cache，When 统计缓冲区字节，Then 较修改前减少 ~50%
- Given T=512、H=4096、28 层、fp16，When 单分支缓存，Then 缓存显存从 ~2×(T·H·L·2B) 降为 ~1×

### 3.7 F7 cache API 语义拆分 + legacy 重写

**描述：** `get(token_mask)` 同时承担"选择范围"与"零填充输出"两职，merge 路径的零填充随后被覆盖，纯冗余。

**触发条件：** 所有 cache 读写路径。

**期望行为：** 三个 cache 实现统一新增 `fetch()`（返回原始数据，不做零填充）与 `fetch_masked()`（兼容旧语义）；folded forward 的 merge 路径改走 `fetch()`；`verification_engine` 等依赖零填充语义的调用方显式使用 `fetch_masked()`。Legacy `ActivationCache` 内部重写为连续 buffer + 向量化 gather（保留公开 API），删除 per-token Python 循环（put O(T) 循环、get 的 LRU touch per-token `.any()` 同步）。

**验收标准：**
- Given legacy cache 的 put/get 基准（T=512），When 重写前后对比，Then put/get 调用中 host-device 同步次数为 0（原 get 为 T 次）
- Given merge 慢路径 folded forward，When 张量分配计数，Then 较修改前每层减少 ≥2 个 [B,T,H] 分配
- 三个 cache 实现的现有契约测试（zero-filled 语义、view 只读契约等）保持通过

### 3.8 F8 gather_select 接入主路径

**描述：** 单 pass `gather_select` kernel（已存在但未接入）替代「cache.get + merge」两步。

**触发条件：** vectorized cache + CUDA + Triton 的 folded forward 慢路径。

**期望行为：** 以 vectorized buffer 为源直接 gather_select，替代 get（含零填充）+ merge 两步；PyTorch fallback 路径行为不变；shape 阈值以下仍走原路径（kernel 只在大 T 获益）。

**验收标准：**
- Given 任意折叠配置，When 走 gather_select 主路径，Then 输出与原 get+merge 路径 bit-exact
- Given 小 shape（T<2048），When folded forward，Then 仍走 PyTorch 路径（阈值保护）

### 3.9 F9 split 层 scatter 合并

**描述：** `_post_hook` 先 zeros 全量再 index_copy divergent 行，随后 merge 又覆盖 stable 行——4–5 pass 可降为 2。

**触发条件：** `SplitFoldedTransformerLayer` 激活（batch×seq ≥ split_min_tokens）。

**期望行为：** `out = parent_ffn.clone()` 后仅对 divergent 行 index_copy，删除 zeros 分配与后续 merge 的 stable 覆盖；hook 常驻注册（并入 F4）。

**验收标准：**
- Given split 层激活的 folded forward，When 张量分配计数，Then 较修改前每层减少 ≥2 个 [B,T,H] 分配
- Given 相同输入，When F9 前后对比，Then 输出 bit-exact

### 3.10 F10 samplers 向量化 + 早停

**描述：** 三个 diffusion sampler 的每步每行 `.item()` 同步、逐行 topk、循环内 `full_like` 与 LLaDA 缺失的早停。

**触发条件：** `num_steps > 1` 的 diffusion 生成。

**期望行为：** (a) `get_num_transfer_tokens` 全向量化（一次 `.tolist()`、批量概率计算、`torch.binomial` 批量采样、去零列向量化）；(b) transfer 选择批量化（k_max + 行掩码的批量 topk）；(c) LLaDA 补 `mask_index.any()` 早停（对齐 dream 已有行为）；(d) dream 的 `x_` 构造提出行循环；(e) suppress tokens 改 `index_fill_` 批量。

**异常处理：** 向量化路径与官方 recipe 数值对拍：`get_num_transfer_tokens` 与逐元素参考实现在多种 (B, steps, schedule) 组合下逐值一致。

**验收标准：**
- Given B=8、steps=32 的 LLaDA 采样，When torch profiler 统计单次 generate 的 D2H 同步，Then 从 ≥256 次降至 ≤32 次
- Given 向量化 `get_num_transfer_tokens` 与逐元素参考实现，When 对拍（B∈{1,8}, steps∈{8,128}, schedule∈{linear,cosine}, stochastic∈{T,F}），Then 逐值一致
- Given 已解完的 block，When LLaDA 采样循环，Then 提前 break（forward 次数减少）
- 现有 sampler smoke 测试保持通过

### 3.11 F11 实验方法学与可移植性

**描述：** 使实验脚本在任何高算力机器开箱即完整复现。

**触发条件：** 运行 scripts/ 下实验脚本与 eval 路径。

**期望行为与明细：**

| 项 | 位置 | 行为 |
|---|---|---|
| 去硬编码 | 7 处脚本的 `HF_ENDPOINT`/`HF_HOME` | 改 `--hf-home`/`--repo-override` CLI 参数 + 环境变量 fallback，默认不再写 AutoDL 路径 |
| 统计方法 | `time_forward`（algo_experiments.py:193-211 等） | 返回 `{mean,std,p50,n}`；`exp_sampling` 等单次运行改 ≥3 重复；图题（make_*_figures.py）由数据格式化，删除硬编码结论 |
| cost model | utils/cost_model.py:31-35,63-106 | `from_device` 按设备名查表（覆盖 A100/H100/4090/RTX6000 系/Quadro 系）+ micro-bench 校准入口；补 attention O(T²) 项与 KV 访存；gate/merge 归入带宽项 |
| flops_counter | utils/flops_counter.py:58-66 | 新增 `ffn_intermediate_dim`/`ffn_type` 参数，从 `config.intermediate_size` 自动读取；embedding 双计数修正；attention T² 项（可选开关） |
| max_new_tokens | eval/base_adapter.py:49 等 + ActFoldConfig | 配置化并按任务默认（gsm8k:256, humaneval:512），runner 透传 |
| 墙钟测量 | eval/base_adapter.py:197-235 | `_generate_one` 包 `gpu_profile`，按路径聚合 latency_ms |
| metric key | eval/judges.py:119-134 | 按 task 显式声明 metric 映射，删除顺序取第一个命中 |
| RNG 隔离 | 7 处 `torch.manual_seed` | 改 `torch.Generator` 显式传递 |
| 消融实测化 | eval/ablation_study.py:150-163,216-264 | layerwise 用 `FoldingScheduler.disabled_layers` 实测；cache 扫描先 put ≥budget+1 分支制造驱逐 |

**验收标准：**
- Given 未设置任何 HF 环境变量的干净环境，When 运行任一实验脚本 `--help` 与干跑（synthetic），Then 不读写 `/root/autodl-tmp`、不依赖 hf-mirror
- Given SwiGLU 模型（intermediate=3.375h），When flops_counter 计算，Then FFN 项 = 2·3·3.375·h²·T（相对修改前的 16h²·T 修正 ~26%）
- Given `HardwareProfile.from_device`，When 在 Quadro RTX 5000 上调用，Then 返回该卡查表常数（而非通用 100/600）
- Given 3 次 `time_forward` 重复，When 读取结果 JSON，Then 含 mean/std/p50/n 字段
- Given config `max_new_tokens`，When benchmark run，Then adapter 生成长度使用该值

### 3.12 F12 历史数据处置与重跑清单

**描述：** 修复后部分历史 results/ 数字失效，需显式标注并生成高算力机器重跑清单。

**触发条件：** 本 AR 全部代码合入后。

**期望行为：** (a) `results/` 下受 B1/B10/M-3 影响的数据文件加 `INVALIDATED.md` 标注（列明失效原因与对应 bug ID）；(b) 生成 `docs/RERUN_CHECKLIST.md`：按实验列出重跑命令、预期显存、环境要求（transformers 版本、bf16、硬件下限）、验收的数字修复点。

**验收标准：**
- Given `results/` 目录，When 检索，Then 每个受影响数据文件（或其父目录）有作废标注
- Given `docs/RERUN_CHECKLIST.md`，When 按清单在某台 ≥24GB bf16 机器执行 fastdllm 命令，Then 可直接运行（路径/镜像无硬编码阻塞）

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | 回归测试 | 原 202 项测试 100% 保持通过；每个 bug 修复点有对应新增回归测试 |
| 性能（代理指标） | 每层阻塞同步 | folded 前向（profiler 开启）从 5–7 次/层降至 ≤1 次/层（torch profiler/nsys 计数） |
| 性能（代理指标） | 内存 pass | 慢路径每层 [B,T,H] 全张量分配/拷贝次数从 10–14 降至 ≤6（张量分配计数） |
| 内存 | cache 显存 | 单分支 cache 缓冲区字节较基线减少 ~50%（F6） |
| 数值 | Triton 对拍 | 所有 kernel 改动与 PyTorch 参考路径 bit-exact（max abs err = 0） |
| 可移植性 | 脚本 | 干净环境下无硬编码路径依赖；Windows 原生与 Linux 均可跑非 evalplus 实验 |
| 代码规范 | lint | black/isort/pyflakes/mypy --strict 全部通过（与 AGENTS.md 一致） |
| 兼容性 | API | 允许 breaking change，但必须同步更新全部内部调用方与测试，并在 CHANGELOG.md 登记 |

## 5. 约束与假设

**约束：**

- 开发与验收全部在本机（Quadro RTX 5000 16GB，Windows，P8 降频）完成；墙钟绝对值不作为验收依据，用同步次数/分配计数/合成微基准相对比较替代
- Triton 在本机 Windows 实测可编译运行（已验证），但必须保持 PyTorch fallback 在 CPU 与 Triton 缺失环境的正确性（AGENTS.md #9）
- 允许 breaking change，但三个 cache 实现的公开 API 行为契约（zero-filled get、view 只读）需通过 `fetch`/`fetch_masked` 明确化而非删除
- 不得引入 mock 数据作为真实结果（AGENTS.md #10）；synthetic 路径必须显式旗标
- M4（单 kernel 融合、CUDA graph）不在本轮，设计时不得做出与 M4 冲突的不可逆决定（如：保留 stride 寻址而非硬编码连续布局、同步消除方案兼容未来 graph 捕获）

**假设：**

- fp16（本机）与 bf16（高算力机）双精度路径都需要正确（B7 的 dtype 感知 eps 同时覆盖）
- 高算力目标机器为 ≥24GB 显存（fastdllm 全系 + 7B/8B 诊断）、复现论文全量需 84GB
- transformers 5.15.1 环境下合成模型测试可跑（已验证）；真实 diffusion checkpoint 需 4.53.1 环境（重跑清单中注明）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| Branch Folding | 逐层逐 token 分类 stable（复用父激活）/ divergent（重算）并合并输出的验证加速机制 |
| 阻塞同步 | 强制 GPU→CPU 等待的调用（`.item()`、`nonzero`、`bool(tensor)` 等） |
| 内存 pass | 对 [B,T,H] 全张量的一次完整读/写（以张量分配/拷贝计数衡量） |
| 稳定比（stable_ratio） | 被判为 stable 的 token 占比（逐层均值） |
| suffix_append | draft 模式：prompt 前缀不变、仅后缀区域追加/重采样，模拟真实投机解码的分歧分布 |
| fetch / fetch_masked | cache 新 API：前者返回原始缓冲数据（无零填充），后者保持旧零填充兼容语义 |
