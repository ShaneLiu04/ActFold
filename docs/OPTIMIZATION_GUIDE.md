# ActFold 深度分析与优化指导指南

> 生成日期：2026-10-08
> 分析基线：commit（main 分支克隆）、202 个快速测试全部通过（Python 3.11.9 / torch 2.5.1+cu121 / transformers 5.15.1 / Windows）
> 本文档用途：① 判定当前机器能否支撑完整实验；② 系统梳理项目全貌与可优化空间；③ 给出分级实施路线图，作为后续优化的指导指南。

---

## 目录

- [第一部分 本机环境能力评估](#第一部分-本机环境能力评估)
- [第二部分 项目深度剖析：一次 folded 前向的真实成本](#第二部分-项目深度剖析一次-folded-前向的真实成本)
- [第三部分 正确性 Bug 清单（优先于一切性能优化）](#第三部分-正确性-bug-清单优先于一切性能优化)
- [第四部分 性能优化路线图](#第四部分-性能优化路线图)
- [第五部分 实验方法学修复](#第五部分-实验方法学修复)
- [第六部分 工程质量与测试盲区](#第六部分-工程质量与测试盲区)
- [第七部分 实施优先级总表与里程碑](#第七部分-实施优先级总表与里程碑)
- [第八部分 AR001 完成回链（2026-10-09）](#第八部分-ar001-完成回链2026-10-09)

---

## 第一部分 本机环境能力评估

### 1.1 本机环境实测清单

| 项目 | 本机 | 论文实验机（对照） | 差距 |
|---|---|---|---|
| GPU | NVIDIA Quadro RTX 5000（Turing，SM 7.5，48 SM） | RTX 6000D 84GB（诊断）/ RTX PRO 6000 Blackwell 96GB（优化） | 架构落后两代，显存 16GB vs 84–96GB |
| 显存 | 16 GB（WDDM 显示模式，日常被桌面占用 ~0.7GB） | 84–96 GB | **5–6×** |
| GPU 实测算力 | fp16 matmul **4.4 TFLOPS**（4096³）；bf16 **1.0 TFLOPS**；fp32 0.5 | 136–138 TFLOPS | **~30×（正常时钟下也应差 ~25–30×）** |
| GPU 实测带宽 | 显存拷贝 **11 GB/s** | ~1280 GB/s | **>100×** |
| CPU | Xeon Gold 6234（8 核 16 线程） | — | 够用 |
| 内存 | 128 GB | — | 充足 |
| 磁盘 | D: 剩余 412 GB | — | 充足（三个模型权重共 ~33GB） |
| Triton | **可在原生 Windows 编译运行**（实测 `merge_stable_divergent` bit-exact 通过，`_TRITON_MERGE_DISABLED=False`） | Linux | 意外可用，但时序数字不可比 |
| WSL | 有 `hypos` 发行版（WSL1，已停止） | — | 需升级到 WSL2 才能跑 evalplus/CUDA |
| transformers | 5.15.1 | 4.53.1（Fast-dLLM v2 锁定；LLaDA/Dream remote code 在 5.x 损坏，AGENTS.md #24） | **必须建独立 venv 降级** |

### 1.2 关键诊断：GPU 被锁死在 P8 空闲状态

持续负载下实测（99% 利用率时）：

```
SM 时钟:    645 MHz（上限 2100 MHz，仅 31%）
显存时钟:   405 MHz（空闲档）
功耗:       38.6 W / 230 W 上限
P-state:    P8（99% 利用率下仍未升档）
```

尝试 `nvidia-smi -lgc` 锁频失败：**当前用户无管理员权限**。可能原因：WDDM 显示模式 + 驱动策略 + 无时钟管理权限。这意味着本机当前**任何 GPU 时序基准都不可信**——测出的 4.4 TFLOPS / 11 GB/s 并非硬件真实能力（Quadro RTX 5000 规格约 60 TFLOPS fp16 / 448 GB/s），但解锁需要管理员权限或切换 TCC 模式（该卡同时承担显示输出，TCC 需谨慎）。

### 1.3 逐实验可行性矩阵

| 实验 / 任务 | 显存需求 | 本机可行性 | 说明 |
|---|---|---|---|
| 快速测试套件（`pytest -m "not slow"`，204 项） | <2 GB | ✅ **已验证 202 passed** | |
| `demo.py` 合成模型 | <2 GB | ✅ **已验证通过** | |
| opt4 fused kernel 基准（合成张量，T≤8192） | ~2 GB | ⚠️ 能跑（Triton 实测可用），但**降频下数字无意义** | |
| Fast-dLLM-v2-1.5B 全系实验（algo/overhead/opt1/opt3） | ~5–6 GB | ⚠️ **降级可行**：显存够；但需 ① 独立 venv 装 transformers==4.53.1；② fp16 替代 bf16（Turing 无原生 bf16，实测 bf16 比 fp16 慢 4.4×）；③ 时序数字只做本机内相对比较 |
| Dream-7B（bf16 权重 ~14GB + 激活 + cache + logits） | ~17–20 GB | ❌ **不可行**（16GB 显存） | int4/int8 量化会改变测量对象本身，失去对照意义 |
| LLaDA-8B（bf16 权重 ~16GB） | ~17–26 GB | ❌ **不可行** | 同上；opt2_long_seq 单实验即需 22–26GB |
| lm-eval 任务（GSM8K/MATH/IFEval） | 视模型 | ⚠️ Windows 可行（仅小模型） | 但见 §E1：`max_new_tokens=16` 使 accuracy 数字本身无效 |
| evalplus（HumanEval+/MBPP+） | — | ❌ 原生 Windows 不支持 | 需 WSL2（当前 hypos 为 WSL1，须 `wsl --set-version hypos 2` 或新装） |
| 与论文 bf16 数字对齐复现 | — | ❌ | 硬件差 25–30×（正常时钟）/ >100×（当前降频状态），且无原生 bf16 |
| pyproject 打包 | — | ⚠️ 已发现并修复 bug | `dependencies`/`keywords` 误置于 `[project.urls]` 段落下，`pip install -e .` 失败（本机已修复，建议回馈上游） |

### 1.4 结论与建议

**结论：本机不能支撑"所有完整的实验"。** 可行分层如下：

1. **完全可行**：全部单元/集成测试、合成 demo、kernel 正确性对拍（Triton 可用）、算法不变式验证（self-fold MSE=0 等）——这些是 CPU/小 GPU 工作负载。
2. **降级可行**：Fast-dLLM-v2-1.5B 的全部诊断与优化实验，条件是：
   - 建独立 venv（`transformers==4.53.1`，torch 不动）；
   - 全部改用 **fp16**（Turing 无原生 bf16；注意 fp16 下 gate 的 `eps=1e-8` 会下溢，见 §B7）；
   - 时序结论**只做本机内相对比较**（folding vs baseline 的比值仍有参考价值，绝对毫秒数无意义）；
   - 先解决 P8 降频（申请管理员锁频 `nvidia-smi -lgc 2100` 或改 TCC），否则连相对比较都被同步噪声淹没。
3. **不可行**：Dream-7B / LLaDA-8B 全精度实验、与论文对齐的绝对性能数字、原生 Windows 的 evalplus。这些需要 ≥24GB（7B/8B bf16 裕量）至 84GB（复现论文全量）的显存。

**建议的机器使用策略**：本机做正确性、方法学修复、kernel 开发与对拍、小模型相对比较；绝对性能与 7B/8B 实验租用大显存 GPU（论文用的 AutoDL 类平台即可；历史脚本曾硬编码镜像与缓存路径，AR001 T019 已改为 `--hf-endpoint`/`--hf-home` CLI + env fallback，见 §R1 与 `scripts/_hf_env.py`）。

---

## 第二部分 项目深度剖析：一次 folded 前向的真实成本

### 2.1 项目定位与当前核心矛盾

ActFold 的主张：Diffusion LLM 投机解码验证阶段，多 child 分支与 parent 只差少数 token，逐层逐 token 分类 **stable（复用父激活）/ divergent（重算）**，减少验证 FLOPs 21–62%。

项目已完成四轮优化（vectorized cache、intra-layer FFN split、adaptive quantile gate、fused gather-select），当前**核心矛盾**（README 局限性 #2 与 DEEP_EXPERIMENT_REPORT 共同确认）：**部分稳定的 folded 前向仍慢于不做 folding 的 baseline**（batch=1、seq≤512 时），剩余开销来自 gating、cache 流量与 dispatch（0.1–0.2 ms/layer）。全部优化的终极目标就是把这层 overhead 压到可忽略。

### 2.2 一次 folded 慢路径前向的真实执行路径（逐 layer）

对 `FoldedTransformerLayer.forward`（`actfold/core/folded_transformer.py:63-210`）+ `VectorizedActivationCache` + 可选 split 层做静态解剖，每个 layer 每次前向实际发生：

| # | 操作 | 位置 | 性质 |
|---|---|---|---|
| 1 | `inspect.signature` 反射 | folded_transformer.py:275 | CPU，10–50µs |
| 2 | `cache.get`（父 hidden，全 1 mask，含 `bool(mask.all())` 同步） | vectorized_cache.py:166 | **同步** |
| 3 | cosine gate 多 pass 计算（child/parent 各一遍 norm + dot） | similarity_gate.py:86 | 3–4 次 [B,T,H] 读 |
| 4 | profiler `record`：`mean().item()` + `torch.nonzero` | stability_profiler.py:131,136 | **2 次同步** |
| 5 | `stable_mask.all()`；`stable_mask.any()` | folded_transformer.py:165,187 | **2 次同步** |
| 6 | 第二次 `cache.get`（父 ffn，stable mask，clone+masked_fill 或 zeros+index_copy+nonzero） | vectorized_cache.py:183-221 | 2–3 次 pass + **可能 1 次同步** |
| 7 | split 层：`nonzero` 求 divergent 索引 + zeros 全量 + index_copy | split_layer.py:169,134-139 | **1 次同步** + 冗余 pass |
| 8 | merge kernel 前 `.contiguous()` 双拷贝（cache 返回非连续 view） | fused_ops.py:189-191 | 2 次 pass |
| 9 | `cache.put`：两个张量 transpose 拷贝 | vectorized_cache.py:117-124 | 2 次 pass |

**合计：每 layer 约 5–7 次 host-device 同步、10–14 次 [B,T,H] 全张量内存 pass、若干 Python 反射。理论下限约为 1 次同步（可异步化到 0）+ 3–4 次 pass。** 这是整个优化空间的来源。

### 2.3 各组件实现现状速览

| 组件 | 位置 | 现状 | 关键问题 |
|---|---|---|---|
| `ActivationCache`（legacy） | core/activation_cache.py | per-token dict，key 为 4 元组 | put/get 均 O(T) Python 循环；get 内 LRU touch 每 token 一次 D2H 同步（:148-151）——**全库最差的一行** |
| `ChunkedActivationCache` | core/chunked_cache.py | chunk 级连续块 | get 每名字 3 次全张量分配；每次 get 都 `sorted()`（:191） |
| `VectorizedActivationCache` | core/vectorized_cache.py | 连续 ring buffer，put 单次 slice copy，full-mask get 返回 0 拷贝 view | **无 (branch, step) 级淘汰**（:224-247 仅有显式 clear_*），跨 step 缓存随生成步数线性泄漏；mask get 仍有零填充冗余 |
| `SimilarityGate` | core/similarity_gate.py | `F.cosine_similarity` 多 kernel | parent norm 每层重算；fp16 eps 下溢风险 |
| `AdaptiveQuantileGate` | core/adaptive_gate.py | `topk(flat, k≈0.97N)` + scatter | k 接近 N 应改 bottom-k；2 次 `.item()` 同步；`last_tau` 写共享状态有竞态 |
| `FoldedTransformerLayer` | core/folded_transformer.py | 三路分支（全稳/全散/慢路径） | 每层同时缓存 layer 输入与输出（输入=上层输出，**~2× 冗余显存**，:298-311）；`(hidden, None)` 硬编码契约 |
| `SplitFoldedTransformerLayer` | core/split_layer.py | 临时 hook 拦截 FFN 链，divergent-only FFN | 每次前向注册/移除 hook；nonzero 同步；zeros 全量冗余 |
| `fused_ops.py` | core/fused_ops.py | Triton merge（3 pass）+ 未接入主路径的单 pass gather_select | merge 无条件双读；强制 contiguous；`hidden_dim%128` 检查过度保守；单全局禁用标志管两条 kernel |
| `FoldedModel` | core/model_wrapper.py | 自动发现层栈并**原地替换** | state_dict key 漂移；restore 靠自觉；丢弃 past_key_values |
| 三个 diffusion sampler | models/*_sampler.py | 官方 recipe 对齐的参考实现 | 每步每行 `.item()` 同步 + 逐行 topk；LLaDA 缺早停；Fast-dLLM 不传 attention_mask |
| verification engine | speculative/verification_engine.py | 相似度门控验证 | 每次验证重跑 parent 前向（无命中判断）；多分支串行；无真正接受率语义 |

---

## 第三部分 正确性 Bug 清单（优先于一切性能优化）

> 这些 bug 直接污染实验数字或静默产出无效结果，必须在任何性能优化前修复并补回归测试。

| # | 位置 | 问题 | 影响 | 难度 |
|---|---|---|---|---|
| B1 | speculative/folded_generation.py:131-138 | `stable_ratio` 只保留**最后一步**的值，docstring 声称是全步均值；`base_adapter._estimate_actfold_tflops`（eval/base_adapter.py:170-195）用它估算总 FLOPs | **benchmark 的 `actfold_tflops` 数字错误** | 极低 |
| B2 | models/architecture_utils.py:384-395 | `ManualFoldedForward` 跳过 final RMSNorm（`model.norm`）直接进 lm_head | Manual 路径 logits 系统性偏差，fidelity/top-1 指标失真 | 低 |
| B3 | models/generic.py:79-84 | 无 head 模型被**随机初始化** lm_head 并静默用于推理 | 走到该分支的 benchmark 输出完全无效且无报错（违反 AGENTS.md #10 自身规定） | 极低 |
| B4 | speculative/folded_generation.py:83-84,128-129 | `record_stability=False` 全局副作用非 try/finally：循环抛异常则全局 profiler 永久关闭 | 跨实验污染 | 极低 |
| B5 | models/fast_dllm_sampler.py:160-212 | 从不传 attention_mask，pad token 无掩码参与注意力；docstring 声称 KV-cache friendly 但实现为全序列重算 | Fast-dLLM 采样正确性与官方 recipe 偏差；文档失实 | 低 |
| B6 | models/sampling_utils.py:284-291 | `sample_tokens` 用 `except Exception` 静默把采样降级为 greedy | 掩盖 NaN logits / device 错误，采样语义漂移 | 极低 |
| B7 | core/similarity_gate.py:31,86 | fp16 下 `eps=1e-8` 下溢为 0：零向量 token 产生 NaN 并可污染 AdaptiveQuantileGate 的 topk 排序；bf16 7-bit 尾数在 τ≈0.95 附近引入分类抖动 | 本机被迫用 fp16 时直接命中；复现性受损 | 低 |
| B8 | profiler/stability_profiler.py:130-136（叠加 utils/gpu_profiler.py:40-42） | profiler 的 2 次同步被计入所有 folded 延迟测量（实验脚本都没关它）；CPU 上延迟落盘为 0 | **当前全部 folded 延迟数字被污染**；0 被当真值 | 低-中 |
| B9 | speculative/folded_generation.py:123 | 注释称 "Prune siblings ... to free cache" 但**无任何 `clear_branch` 调用**；VectorizedActivationCache 的 (branch,step) buffer 无自动淘汰 | 长生成显存随步数线性泄漏（实测确认） | 低 |
| B10 | speculative/draft_generator.py:94-116 + eval/ablation_study.py | **方法学核心缺陷**：copy_flip 从全词表均匀采样，翻转 token 与原 token 相似度必为 0，任何 τ∈[0.5,1] 都判 divergent——`threshold_sensitivity.csv` 三个 τ 的 stable_ratio 完全相同（0.9375）即为此症状；真实 draft 分歧是语义近邻+集中于后缀 | **所有基于 stable_ratio 的结论（τ sweep、TFLOPs 节省、消融）目前是结构性平凡的** | 低（suffix 模式）/中（logits draft） |
| B11 | eval/base_adapter.py:49 + 全部 adapter 默认 `max_new_tokens=16`，runner 不透传 | 数学/代码题 16 token 必然答不出 | accuracy 数字无解释力 | 低 |
| B12 | core/model_wrapper.py:144 | `FoldedModel` 原地替换层 → state_dict key 从 `layers.0.self_attn.*` 变 `layers.0.original_layer.self_attn.*`，任何后续 `load_state_dict`/`save_pretrained`/量化权重映射静默失效 | 正确性 + 可维护性 | 中 |
| B13 | pyproject.toml | `dependencies`/`keywords` 误置于 `[project.urls]` 段落下，`pip install -e .` 直接失败 | 可安装性（**本机已修复，建议回馈上游**） | 已修 |

---

## 第四部分 性能优化路线图

> 总纲：把每 layer 的 folding 辅助成本从「5–7 次同步 + 10–14 次 pass」推向「0–1 次同步 + 2 读 1 写」。这是让部分稳定 folded 前向反超 no-folding baseline（项目当前核心矛盾）的唯一路径。

### P0 低垂果实（局部修改，合计 1–2 天，预期消除 90% host 同步 + 每层 4–6 个冗余 pass）

| # | 优化点 | 位置 | 方案 | 预期收益 |
|---|---|---|---|---|
| P0-1 | legacy cache per-token `.any()` 同步 | activation_cache.py:148-151 | 删除 LRU touch 循环（vectorized 路径下无意义）或批量 mask 判断 | legacy get 从 T 次同步 → 0–1 次 |
| P0-2 | 三分支判定两次同步 | folded_transformer.py:165,187 | 合并为一次 `stable_mask.sum().item()`，按 `count==0/==total/else` 分派 | 每 layer −1 同步 |
| P0-3 | profiler 热路径同步 | stability_profiler.py:130-136 | record 只存 GPU tensor（求和不 item）；`divergence_positions` 仅 debug 模式收集；读取时一次性回读 | 每 layer −2 同步；**folded 延迟测量去污染（同时修复 B8）** |
| P0-4 | full-mask get 也同步 | vectorized_cache.py:166 + folded_transformer.py:128 | 增设无 mask 的 `get_all()`；调用方永远传全 1 mask | 每 layer −1 同步 |
| P0-5 | merge kernel 无条件双读 | fused_ops.py:132-140 | stable/divergent 在 program 内 uniform，按行 `if` 只读选中一侧（`_gather_select_kernel`:252-256 已示范） | merge 带宽 3→2 pass（上限 +33%） |
| P0-6 | merge 前双 contiguous 拷贝 | fused_ops.py:189-191 | kernel 用已传入但未使用的 `hidden_stride` 寻址（:112），删 `.contiguous()` | 每 layer 省 2 个全张量拷贝（32 层 T=1024 H=4096 时 ~512MB 流量/前向） |
| P0-7 | `hidden_dim % 128` 过度保守 | fused_ops.py:183-186 | kernel 已有 masking（:134），删检查或自适应 BLOCK_SIZE | 覆盖非 2 的幂 hidden（如部分 MoE） |
| P0-8 | adaptive gate k≈N 的 topk | adaptive_gate.py:63-67 | 改 bottom-k(1−ratio) 再取反；`last_tau` 异步化 | 选择成本降数十倍；−2 同步 |
| P0-9 | 每次前向 `inspect.signature` | folded_transformer.py:275；model_wrapper.py:209,233 | `__init__` 缓存（original_layer 终身不变） | 每 layer 省 10–50µs CPU |
| P0-10 | 每层重新分配全 1 mask + 两次独立 get | folded_transformer.py:131-135,128,230 | 按 shape 缓存 mask；单次 get 同时取 hidden+ffn | 每 layer 省一次 get 全套开销 |
| P0-11 | split 层每次注册/移除 hook | split_layer.py:175-176,190-191 | 构造时永久注册，靠 `_split_state is None` 守护 | Python 开销 + 消除 graph break |
| P0-12 | gate 的 dtype 感知 eps / fp32 累加 | similarity_gate.py:31,86 | eps 按 dtype 缩放；dot/norm fp32 累加（修复 B7） | 数值正确性 + 复现性 |

### P1 结构性优化（1–2 周）

| # | 优化点 | 位置 | 方案 | 预期收益 | 难度 |
|---|---|---|---|---|---|
| P1-1 | **cache 同时存 layer 输入与输出，输入=上层输出** | folded_transformer.py:298-311 | 残差流上 layer L−1 的 ffn_out 恒等于 layer L 的输入；只存 `ffn_out`（+ layer 0 embedding），gate 跨层读 `ffn_out[L-1]` | **cache 显存 −50%、put 带宽 −50%**（单项最大内存收益） | 中 |
| P1-2 | cache.get 零填充被 merge 立即覆盖 | vectorized_cache.py:184-221；chunked_cache.py:221；fused_ops.py:469 | 拆分 `fetch()`（原始数据）/`fetch_masked()`（兼容层）；merge 路径走前者（需审计 verification_engine.py:185 的零填充语义依赖） | 每 layer 省 2–3 个 [B,T,H] pass | 低-中 |
| P1-3 | 单 pass `gather_select` 未接入主路径 | fused_ops.py:230-342 | 以 vectorized buffer 为源，主流程用 gather_select 替代「get+merge」两步 | folding 辅助 pass 从 ~5 → 2 | 中 |
| P1-4 | split 的 zeros 全量 + merge 再覆盖 | split_layer.py:134-139 | `out = parent_ffn.clone()` + 仅 scatter divergent 行 | split+merge 4–5 pass → 2 pass | 中 |
| P1-5 | (branch,step) 缓存无限增长 | vectorized_cache.py:224-247 + folded_generation.py | 保留 N 个最近 step 的 ring；prune 时真正 `clear_branch`（修复 B9） | 长生成显存 O(N)→O(1) | 低 |
| P1-6 | verification engine 重复 parent 前向 | verification_engine.py:146-168 | cache 增 `contains()`；多分支共享一次 parent forward（SpiffyBaseline 式 N 分支场景） | 多分支验证成本 −50% 以上 | 低 |
| P1-7 | 多分支验证串行 | verification_engine.py + folded_generation.py:114-117 | N 个 child 拼成 batch 一次 forward（分支 batch 维对齐，core 的 cache/gate 需支持 batch 分支语义） | 多分支墙钟 ~1/N（GPU 未饱和时）；**让 folded 验证真正快于 baseline 的关键** | 中-高 |
| P1-8 | samplers 每步每行同步 + 逐行 topk | llada_sampler.py:131-227；dream_sampler.py:166-178；sampling_utils.py:160-210 | 批量 topk（k_max + 行掩码）；`get_num_transfer_tokens` 全向量化（一次 `.tolist()`）；LLaDA 补早停（dream 已有，行为不一致）；`x_` 提出行循环 | 每步 B 次同步 → 0–1 次；block 提前解完省整段 forward；steps=128 时省 10–100ms/请求 | 中 |
| P1-9 | gate 读带宽：parent norm 每层重算 | similarity_gate.py:86 | `cache.put` 时顺带存 per-token L2 norm（[T] 向量，可忽略） | gate 读带宽 −1/3 | 低 |
| P1-10 | `ManualFoldedForward` 缺 final norm | architecture_utils.py:384-395 | `detect_architecture` 增加 final_norm 发现（`model.norm`/`transformer.ln_f`/`encoder.norm`）+ 对拍回归测试（修复 B2） | 正确性 | 低 |
| P1-11 | Triton 降级粒度过粗 | fused_ops.py:143,304-311 | 按 kernel、按 (dtype, shape 类别) 独立降级；失败一次性告警 | 可诊断性 | 低 |
| P1-12 | tuple 契约 / past_key_values 丢弃 | folded_transformer.py:220-221；model_wrapper.py:253-275 | 记录原始输出结构模板（元素数+类型）而非 bool；unwrap 丢弃 KV 时告警 | 健壮性 | 低 |

### P2 战略级优化（把 folded 前向推向理论极限）

| # | 优化点 | 方案 | 预期收益 | 难度 |
|---|---|---|---|---|
| P2-1 | **单 kernel gate+gather+merge** | 每 program（token 行）：load parent/child hidden 行 → 寄存器 fp32 cosine → stable 判定 → 按 stable 单边读 ffn 行写出 + stable 位写 device-side mask buffer；host 端用 D2H copy + cudaEvent 异步读取统计，替代全部阻塞同步 | 整层 folding 辅助开销 2 读 1 写 + 0–1 次同步，对比现状 ~14 pass；**辅助开销 3–5×** | 中-高 **◐ AR002 M4a 完成 gate+mask+count 单 kernel（`fused_gate_mask_count`）；gate↔merge 间存在重计算依赖不可合并（design D3），merge 沿用既有单 kernel** |
| P2-2 | nonzero 同步的根除（split 层） | 固定容量 padded gather（索引 clamp + mask 过滤），接受固定 divergent 预算的少量多余 FFN 行计算 | 消除每 layer 1 次同步；**CUDA graph 前置条件** | 中 **✅ AR002 T001（`_exact_divergent_index`/`_padded_divergent_index` + stable_count 转发）** |
| P2-3 | **CUDA graph / torch.compile 捕获验证循环** | 前置：P0 全部 + P2-2 + branch 上下文改由编译期 kwargs 传递（弃用 contextvars/thread-local）；diffusion 验证阶段是"同 shape 反复前向"的典型可捕获负载 | 固定 shape 场景潜在 2–5×；launch 开销归零 | 高（战略价值最大）**✅ AR002 T006–T008（kwargs 化 + `FoldedGraphRunner` + `ManualFoldedForward(use_cuda_graph=True)`；本机实测 per-step -47.1%，见第九部分）** |
| P2-4 | 生成循环 O(T²) 消除 | baseline 侧接 KV cache（HF `DynamicCache`）使对照公平；folded 路径明确其适用域是 diffusion 多分支验证（本无 KV cache），并在论文口径中区分 | baseline 5–20× 墙钟（长序列）；**方法学对照公平性** | 低（baseline）/高（共存设计） |
| P2-5 | 真正的投机解码接受语义 | 当前 engine 只测激活相似度，无基于 logits 的接受率验证（draft 分布 vs target 分布）；升级为 EMA[r] 式接受率 + log-prob 分数 | 论文叙事成立的前提；`logits.float().mean().item()`（verification_engine.py:122）这类无意义分数一并替换 | 高 **✅ AR004（`acceptance.py` 纯函数库 + 引擎 EMA/判定切换 + `TargetMatchAcceptancePolicy` + folded_generate 报告；见第十一部分）** |

### P3 算法/功能扩展（解锁真实场景）

1. **变量长度折叠** ~~只支持等长、直接 `NotImplementedError`~~ **✅ AR003（append-only 前缀折叠）**：`FoldedTransformerLayer` 前缀对齐（gate 前缀比较、后缀恒 divergent、merge 前缀对齐），`BranchManager.align_tokens` 已实现（前缀对齐对），`folded_generate` 真正跨步折叠（因果合成模型 stable_ratio ≈0.84、tokens 与全重算逐位一致）。**遗留子项**：`Branch` 强制持有全量 `hidden_states [L,B,T,H]`（~512MB/branch @ 8B/T=1024）的惰性引用化仍未做；parent 更长的截断复用与多祖先见 P3-2。
2. **多祖先复用**：README 局限性 #7 自认只支持单 parent。多 parent 激活树（类似 KV cache 的 paged 结构）可进一步提升复用率。
3. **真实 draft 模型**：`DraftGenerator` 只有 random/perturb/copy_flip；接入 Medusa/Eagle 类小 draft model 是 roadmap 承诺项（与 B10 联动，suffix/logits-draft 是其前奏）。
4. **MoE 支持**：~~`flops_counter` 硬编码 4h 中间维（utils/flops_counter.py:60-63），SwiGLU（LLaDA/Dream 均是，3 矩阵×3.375h）低估 FFN FLOPs ~26%；MoE 完全未覆盖。加 `ffn_type`/`intermediate_dim` 参数并从 `config.intermediate_size` 自动读取——**当前所有 TFLOPs 数字含 10–25% 系统偏差**~~ **✅ AR005（FFN/MoE FLOPs 几何修正收口）**：`DiffusionLLM` 几何属性 + `_extract_ffn_geometry` 属性名并集（含 SwiGLU 族判定）+ `model_ffn_flops_kwargs` `underlying_model` 链解析，三调用点零改动自动供参；MoE per-token expert 计量（top-k ± shared、`first_k_dense_replace` 混合层数）；无属性路径逐位零回归。见第十二部分。

---

## 第五部分 实验方法学修复

> 项目已有两次真实模型实验（诊断 + 优化），数据扎实（不变式 bit-exact、CUDA events 计时），但存在以下方法学缺口，影响结论的可信度与可发表性。

| # | 问题 | 位置 | 修复方案 | 难度 |
|---|---|---|---|---|
| M-1 | 无方差/置信区间：所有 `results/*.json` 只有标量；`time_forward` 返回均值不存 std；accuracy 在 num_samples=10 时标准误 ~±15pp | scripts/algo_experiments.py:193-211 等 | `time_forward` 返回 `{mean,std,p50,n}`；每条件 ≥3 seed；accuracy 报 Wilson 区间；图题由数据格式化（make_optimization_figures.py:316-317 目前把 "2.3-2.7x" 硬编码进标题） | 低（脚本）/中（补跑） |
| M-2 | draft 分布与真实场景脱节（=B10） | draft_generator.py | 三层递进：suffix_append → logits_draft（parent top-k 采样模拟低质 draft）→ 真 draft model | 低→中→高 |
| M-3 | "消融"是算术推演非实测：layerwise folding 用线性比例缩放（ablation_study.py:150-163）而非 `FoldingScheduler.disabled_layers` 真跑；cache 预算扫描根本触发不了 LRU 驱逐（四行 stable_ratio 全同） | eval/ablation_study.py:150-163,216-264 | disabled_layers 实测；cache 扫描先 put ≥budget+1 分支制造驱逐 | 低 |
| M-4 | cost model 硬件常数一刀切：所有 CUDA 设备按 100 TFLOPS/600 GB/s 计（utils/cost_model.py:31-35），实验机实际 400+/1280；且忽略 attention O(T²) 项与 KV 访存、gate/merge 误用算力计时（实为带宽主导） | cost_model.py:31-35,63-106 | `from_device` 按设备名查表 + 启动时一次 micro-bench 自动校准（复用 `exp_cost_model` 逻辑）；补 T² 项与 KV 带宽；`calibrate()` 分 compute/memory 两参数独立校准 | 低 |
| M-5 | 跨版本对比（opt 前 vs 后）依赖不同时间两次运行，GPU 状态未受控 | results/optimization/baseline/ | 同进程加载两版实现或同日交错重跑（now-内的三实现对比已正确，保持） | 中 |
| M-6 | lm-eval 内部 API 依赖（`fewshot_context`/`process_results`）随版本漂移；metric key 按 `("exact_match","acc",...)` 顺序取第一个命中，gsm8k 变体 key 可能拿错 | eval/judges.py:119-134,187-223 | 按 task 显式声明 metric 映射；改走 lm-eval 官方 `simple_evaluate` 的 `LM` adapter 接口 | 中 |
| M-7 | 硬编码 AutoDL 镜像与缓存路径（`HF_ENDPOINT=hf-mirror`、`HF_HOME=/root/autodl-tmp`，7 处脚本）；Windows 上静默在 `D:\root\...` 建目录 | scripts/*.py:25-36 | 改 `--hf-home`/`--repo-override` CLI 参数 + 环境变量 fallback **✅ 已完成（T019，`scripts/_hf_env.py`）** | 极低 |
| M-8 | 评测路径不测量墙钟：`_evaluate` 只产 accuracy/tflops，无 latency/throughput；论文需要"同质量下快多少"的墙钟证据 | eval/base_adapter.py:197-235 | `_generate_one` 内包 `gpu_profile` 按路径聚合 | 低 |
| M-9 | 全局 RNG 污染：7 处 `torch.manual_seed` 重置进程级随机态 | draft_generator.py:63 等 | 统一改 `torch.Generator` 显式传递 | 低 |

---

## 第六部分 工程质量与测试盲区

### 6.1 测试覆盖盲区（先补测试再动手优化）

- `folded_generate` 多分支路径**零测试**（现有 3 例全是单分支 greedy）；`AcceptancePolicy`、`BranchTree.prune/best_accepted_leaf`（后者是死代码）无测试。
- samplers 只做形状 smoke test：`stochastic_transfer`、`alg_temp>0` 软选择、`remasking="random"`、CFG、suppress_tokens 均未覆盖——**恰是 P1-8 向量化改造最易破坏的分支**。
- `get_num_transfer_tokens` 与官方参考实现无数值对拍（AGENTS.md #21 自我要求但无落地）。
- `ManualFoldedForward` 无 final-norm 回归测试（B2 有测试就会被抓到）。
- scripts 的公共 helper（`make_child`/`time_forward`/`fidelity`）无纯函数单测。
- `conftest.py:22-27` autouse seed 使测试共享随机态序列，测试间隐式耦合。

### 6.2 API/架构设计债

- `FoldedModel` 原地替换层（B12）：建议 context manager 协议 + `__del__` 兜底，或全面转向不替换式的 `ManualFoldedForward`（hook 方案已证明不替换也可注入逻辑，两种手段应统一）。
- cache 的 `get(token_mask)` 同时承担"选择范围"与"零填充输出"两职（三个实现都要维持该契约）——正是 P1-2 冗余的根源，应拆 API。
- `AdaptiveQuantileGate` 与 `FoldingScheduler` 互斥但无检测（AGENTS #29）：`FoldedTransformerLayer` 应显式拒绝组合。
- `architecture_utils.find_layer_list` 只认 `nn.ModuleList`（:147），`nn.Sequential`/自定义容器不支持，且 auto-discovery 失败时**静默降级为 embedding 代理测量，用户不知道 folding 没生效**——应返回 `folding_status` 枚举并 WARNING。
- `registry._resolve_family` 子串匹配（"fast" in name）与 `models/utils.infer_model_family` 双份维护；未知 diffusion 家族静默落 `causal_lm`。
- `generic.forward` 恒置 `output_hidden_states=True`（generic.py:112）：7B 模型 T=512 时数百 MB 峰值浪费。
- Windows 兼容：evalplus 明确 raise（合理）；bash 脚本全家桶需 WSL；`run_ablation.sh` heredoc 未加引号的变量插值存在注入面。

---

## 第七部分 实施优先级总表与里程碑

### 7.1 里程碑建议

| 里程碑 | 内容 | 完成标志 |
|---|---|---|
| **M1 正确性清零**（~3 天） | B1–B13 全部修复 + 每项补回归测试；6.1 测试盲区中与将被优化路径相关的先补齐 | 204+ 测试全绿；threshold_sensitivity.csv 不同 τ 出现差异（B10 生效的可观测信号） |
| **M2 同步清零**（~1 周） | P0-1 至 P0-12 | 每 layer 阻塞同步 ≤1（profiler 默认异步）；fastdllm 折叠前向同步计数可用 nsys 佐证 |
| **M3 内存 pass 压缩**（~2 周） | P1-1/2/3/4/5（重点 P1-1 冗余 hidden_states 与 P1-3 gather_select 接入） | cache 显存 −50%；每 layer folding 辅助 pass ≤4；部分稳定 folded 前向与 baseline 的差距收窄到 <20% |
| **M4 反超 baseline**（战略） | P2-1/2/3 | batch=1、seq≤512 下 folded 前向 **快于** no-folding baseline（项目核心矛盾的解决即 README 局限性 #2 的关闭） **◐ AR002 完成 M4a/M4b 机制与本机代理验证（graph vs eager per-step -47.1%）；与 no-folding baseline 的正式对拍需按 RERUN_CHECKLIST 在锁频机器上补跑** |
| **M5 实验可信化**（与 M2-M4 并行） | M-1 至 M-9 + B10/M-2 真实 draft 分布 | 所有 published 数字带 CI；消融全部实测化；cost model 校准后预测/实测误差 <1.3× |
| **M6 场景扩展** | P3 变量长折叠 ✅ AR003（append-only 前缀折叠）/ 多祖先 / 真 draft model / MoE | 解锁真实投机解码工作负载 |

### 7.2 与本机环境的配合

1. **本机可立即做**：M1 全部、M2/M3 的开发与正确性对拍（Triton 可用、202 测试套件是安全网）、Fast-dLLM-1.5B 的 fp16 相对比较（建议先申请管理员权限解决 P8 降频，否则相对比较的同步噪声也很大——0.1–0.2ms/layer 的 overhead 量级与降频后的 kernel 时间同数量级）。
2. **需另借算力**：M4/M5 的 7B/8B 绝对数字、与论文 bf16 数字的对齐复现（≥24GB 显存起步，84GB 复现全量）。
3. **环境准备清单**（本机）：独立 venv `transformers==4.53.1`；WSL2 升级（`wsl --set-version hypos 2`）以备 evalplus；HF 缓存路径经 `--hf-home`/`HF_HOME` 传入（M-7 已由 T019 完成，脚本无硬编码路径）。

### 7.3 一句话总结

ActFold 的工程质量（类型标注、AGENTS 约定、不变式测试）在研究代码中属上乘，四轮优化已把 cache 层做对；剩余的核心矛盾——**部分稳定 folded 前向仍慢于不折叠**——的根因不在算法而在工程：每层 5–7 次 host 同步与 ~14 次内存 pass。按 M1（正确性）→ M2（同步）→ M3（pass）→ M4（单 kernel + CUDA graph）的顺序推进，配合实验方法学修复（尤其 draft 分布），项目完全有机会把"21–62% FLOPs 减少"兑现为可测量的端到端墙钟收益。

---

## 第八部分 AR001 完成回链（2026-10-09）

AR001（`specs/changes/AR001-deep-optimization/`，T001–T026 全部 passing）已覆盖本指南的以下条目；全量 532 passed + mypy --strict 全绿。

| 指南条目 | AR001 任务 | 状态 |
|---|---|---|
| B1–B13 正确性清单（第三部分） | T001–T008、T027（stable_ratio 全步均值、final_norm、随机 head 报错、profiler try/finally、NaN/Inf 前置、fp32 相似度、CPU perf_counter、ring 有界、draft 双模式、FoldedModel 异常安全） | ✅ 完成；B12 完整重构留 AR002（本轮仅 B12-lite） |
| P0-1..P0-12 host 同步清零 | T009–T013（profiler 惰性回读、单 sync 三分支、bottom-k 自适应门、legacy cache 连续 buffer、Triton merge 单边读/stride/tail-mask） | ✅ 完成 |
| P1-1..P1-5 内存 pass 压缩 | T014–T018（删冗余 hidden_states、cache_protocol 契约、fetch_flat+gather_select D3 阈值、split 常驻 hook、get_num_transfer_tokens 全向量化 bit-exact + LLaDA/Dream sampler 向量化） | ✅ 完成 |
| M-1 无方差 | T020（TimingStats mean/std/p50/n、stats_from_samples、repeat_with_seed、exp_sampling repeats/seed/tokens_reproducible） | ✅ 代码完成；历史数据补跑见 RERUN_CHECKLIST |
| M-2 draft 分布 | draft 双模式（M1 内）+ suffix_append 作为消融缺省（T024） | ✅ 完成（真 draft model 属 P3/AR002+） |
| M-3 消融实测化 | T024（measure_folding 逐层实测、disabled_layers、真实驱逐的 cache 扫描） | ✅ 完成 |
| M-4 cost model 一刀切 | T021（_DEVICE_TABLE + from_device + calibrate、T²/KV 项、gate/merge 改带宽计） | ✅ 完成 |
| M-5 跨版本对比不受控 | 标注失效（results/optimization/INVALIDATED.md）+ RERUN_CHECKLIST 要求同日同机交错 + 锁频 | ⏳ 待目标机器重跑 |
| M-6 metric key 漂移 | T023（LMEvalAdapter._TASK_METRIC_KEYS 规范映射、judges metrics 聚合）；simple_evaluate adapter 接口未改 | ✅ 主体完成 |
| M-7 镜像/路径硬编码 | T019（scripts/_hf_env.py、7 脚本迁移、sys.path hack 清零） | ✅ 完成 |
| M-8 评测无墙钟 | T023（_generate_one 计时、baseline/actfold_latency_ms） | ✅ 完成 |
| M-9 全局 RNG 污染 | exp 路径与 measure_folding 均快照/恢复全局 RNG；DraftGenerator.generate 仍用全局 manual_seed | ◐ 部分完成（见 CHANGELOG Known follow-ups） |
| 6.1 测试盲区 | 新增 6 个测试文件共 82 用例（portability 12 / stats 16 / cost 14 / flops 11 / eval 20 / ablation 9） | ✅ 完成 |
| 里程碑 M4（反超 baseline） | 单 kernel 融合、CUDA graph | ⏸ 留 AR002 |

**数据处置**：`results/` 全部历史产物已加 INVALIDATED.md 标注（不可引用）；重跑按 `docs/RERUN_CHECKLIST.md` 执行。合成 demo 基线修正为 FLOPs reduction **85.5%**（原 78.5% 为 embedding 双计数偏差，见 T022）、MSE 2.35e-03、stable ratio 93.75%。

**变更登记**：全部变更与 breaking changes 见 `CHANGELOG.md` 的 AR001 章节；agent 约定新增见 `AGENTS.md` #31–#36。

---

## 第九部分 AR002 完成回链（2026-10-09）

AR002（`specs/changes/AR002-m4-graph-capture/`，T001–T010 全部 passing）覆盖本指南 M4 里程碑与 P2-2/P2-3 战略条目；全量 624 passed + mypy --strict 全绿。

| 指南条目 | AR002 任务 | 状态 |
|---|---|---|
| P2-2 nonzero 同步根除 | T001（`_exact_divergent_index`/`_padded_divergent_index`、`_recompute_merged` stable_count kwarg 转发免二次读回） | ✅ 完成 |
| P2-3 前置：branch 上下文 kwargs 化 | T004（`FOLDING_CONTEXT` 毒化下 Manual 路径 bit-exact；contextvars 依赖仅剩 deprecated `FoldedModel`） | ✅ 完成 |
| M4a 单 kernel gate+count | T006（`fused_gate_mask_count`：cosine+阈值+mask+count 单 pass；`_FUSED_GATE_MIN_TOKENS=1024`；fp32/fp16/bf16 bit-exact；gate↔merge 因重计算依赖保持两 kernel，design D3） | ✅ 完成 |
| M4b CUDA graph 捕获验证循环 | T007/T008（`FoldedGraphRunner` 静态 buffer 组 + side-stream warmup 捕获；`ManualFoldedForward(use_cuda_graph=True)` 惰性捕获 + 全降级矩阵 + 每步 ≤1 readback 预算校验；`ActFoldConfig.use_cuda_graph/graph_capacity_ratio` opt-in） | ✅ 完成 |
| B12 `ManualFoldedForward` 常态化 | T005（split 支持补齐、state_dict 零漂移、与 `FoldedModel` bit-exact 对拍；`FoldedModel` 标 deprecated 保留 legacy；`AblationStudy` 内部栈切 Manual） | ✅ 完成 |
| M4 性能代理实测 | T009（`scripts/ar002_graph_bench.py`；本机 Quadro RTX 5000，B=2/T=512/4 层：eager 5.083 ms/step vs graph 2.687 ms/step，**-47.1%**，20/20 validated steps；产物 `results/optimization/ar002_graph_bench.json`） | ✅ 本机代理完成；CUPTI 机器补 launch 计数、锁频机器补正式 baseline 对拍 |

**诚实性说明**：本机 torch 为 LIBKINETO_NOCUPTI 构建，无法记录 CUDA profiler 事件，kernel launch 计数以 `null` + 显式 note 记录（UT-006b launch 断言在无 CUPTI 主机自动 skip），不伪造数据；wall-clock 计时不受影响。

---

## 第十部分 AR003 完成回链（2026-10-10）

AR003（`specs/changes/AR003-var-len-prefix-folding/`，T001–T007 全部 passing）覆盖 P3-1 变量长度折叠与 AR002 遗留接线；等长折叠路径零回归（demo 基线 85.5% / 2.35e-03 / 93.75% 精确一致）。

| 指南条目 | AR003 任务 | 状态 |
|---|---|---|
| P3-1 变量长度折叠（前缀对齐） | T001（`align_tokens` 实现）/ T002–T003（`FoldedTransformerLayer` 前缀分类 + 前缀 gate + 全 False 后缀 mask + merge 前缀对齐；全 stable 快路径仅等长可达；split 层在完整 mask 上含全部后缀行；`_store_activations` 链式递归） | ✅ 完成（append-only；parent 更长/多祖先不支持） |
| P3-1 验收（folded_generate 真折叠） | T004（因果合成模型 4 步：tokens 与 eager 全重算 `torch.equal`、stable_ratio ≈0.84、链式 cache 递增；graph 模式零干扰——不捕获/零告警/不禁用） | ✅ 机制 + 合成代理完成；真实投机解码负载墙钟收益待目标机按 RERUN_CHECKLIST 测量 |
| AR002 遗留：benchmark_runner 接线 | T005（`_build_folded_model` 迁 `ManualFoldedForward` + 消费 `use_cuda_graph`/`graph_capacity_ratio`；D3：graph 开启时不挂 `FoldingScheduler`；架构检测失败 → None） | ✅ 完成 |
| P3-1 遗留子项 | `Branch.hidden_states [L,B,T,H]` 全量持有的惰性引用化 | ⏸ 未做（独立条目） |
| 文档收口 | T006（README 局限性 #7 状态更新、AGENTS #19/#20 修订、CHANGELOG、本回链、ALGORITHM.md §11） | ✅ 完成 |

**语义要点**：后缀恒 divergent 用显式 all-False mask tail 实现（对任意 tau 稳健，否决零填充方案——`tau<0` 时零向量 cosine=0 会被误判 stable）；var-len 不走 fused gate（同形连续契约 + 阈值不可达）与 `gather_select`（`fetch_flat` 按 child 长度索引 parent 行）；等长路径（`prefix_len == T_c`）逐位不变为最高不变量。

---

## 第十一部分 AR004 完成回链（2026-10-10）

AR004（`specs/changes/AR004-logit-acceptance-semantics/`，T001–T006 全部 passing）覆盖 P2-5 真投机解码接受语义；默认参数行为零变化（判定阈值 0.0 下全接受，demo 基线 85.5% / 2.35e-03 / 93.75% 精确一致），全量回归 699 passed。

| 指南条目 | AR004 任务 | 状态 |
|---|---|---|
| P2-5 接受率核心 | T001（`acceptance.py` 四纯函数：`target_argmax_accept_mask` / `draft_region_mask` / `acceptance_rate` / `mean_log_prob`，共享形状校验；engine `:122` 占位替换为 mean_log_prob） | ✅ 完成 |
| P2-5 EMA[r] + 判定切换 | T002（`ema_alpha=0.3` 域 (0,1]、首调初始化无零先验稀释；`accepted = acceptance_rate >= threshold`，默认 0.0 零行为变化；stable_ratio 照常上报） | ✅ 完成 |
| P2-5 policy + 报告 | T003（`TargetMatchAcceptancePolicy`：appended token 同位置 argmax 匹配率选优、并列取首、None 跳过、全 None→首候选；`folded_generate` 每步 metadata 接受率 + 结果字段跨步均值，与 policy 解耦） | ✅ 完成 |
| P2-5 baseline 分数 | T004（`SpiffyBaseline` `baseline_score` → 全位置 mean_log_prob，键名不变） | ✅ 完成 |
| 文档收口 | T005（AGENTS #40、CHANGELOG、README 局限 #8、本回链、指南 P2-5 勾选） | ✅ 完成 |

**语义要点**：同位置预测约定（target `logits[:, i]` 预测位置 i 的 token）是全链唯一契约；draft 区域 = 分支新主张位（前缀差异位 + 变长后缀），`T_p < T_c` 截到公共前缀；空 draft 区域 rate=1.0（无新主张=全接受）、mlp=0.0 哨兵；`actfold_score`/`baseline_score` 键名不变仅语义升级（唯一消费面为序数比较）；EMA 首调直初始化（否决 0 先验——首步接受率会被稀释）。**遗留**：接受率当前对着 random/perturb draft 测量，是机制指标；真 draft 模型（P3-3）接入后才成为论文口径的接受率。

---

## 第十二部分 AR005 完成回链（2026-10-10）

AR005（`specs/changes/AR005-ffn-flops-geometry/`，T001–T005 全部 passing）覆盖 P3-4 FFN/MoE FLOPs 几何修正收口；无属性路径逐位零回归（demo 基线 85.5% / 2.35e-03 / 93.75% 精确一致——demo 合成模型走默认几何），全量回归 735+ passed。

| 指南条目 | AR005 任务 | 状态 |
|---|---|---|
| P3-4 MoE 计量 | T001（`count_diffusion_llm_flops` 5 个 MoE 参数：top-k 触发键、±shared、`moe_num_layers`/`first_k_dense_replace` 混合层数、moe_inter 回退链；ValueError 域校验） | ✅ 完成 |
| P3-4 config 自动读取 | T002（`DiffusionLLM.config` 转发 property + 7 几何 property；`_extract_ffn_geometry` 属性名并集单一实现；`model_ffn_flops_kwargs` `underlying_model` 链解析 + `_resolve_model_config`） | ✅ 完成 |
| P3-4 三调用点接线 | T003（IT-411 engine / IT-412a ablation / IT-412b base_adapter：tflops == 真实几何手算且 ≠ 4h 默认；调用点零改动实证） | ✅ 完成 |
| 文档收口 | T004（AGENTS #33 修订、CHANGELOG AR005 节、README demo 样例 78.5%→85.5% 修正 + 本回链 + P3-4 勾选） | ✅ 完成 |

**语义要点**：AR001 只做了参数与 helper，全链无人供参——本 AR 补齐"属性暴露 + 链解析 + 调用点自动供参"三环；`moe_top_k` 是计量触发键（per-token FLOPs 只依赖 top-k，`num_experts` 仅容量校验）；`first_k_dense_replace` 消除"全层 MoE"假设的高估（DeepSeek 系前 k 层 dense）；SwiGLU 族 = HF `hidden_act` ∈ {silu/swish/swiglu}，未知名保守回退 `"mlp"`（不高于现状）；shared expert 按 routed 中间维近似（<5%）；router 门控忽略（<1%）。**遗留**：真实 MoE checkpoint 实测（本机无权重，公式 + 提取为合成实证，RERUN_CHECKLIST 流程）；历史数字（含 demo 合成模型口径）仍基于 4h-MLP 默认几何，真实 checkpoint 的 TFLOPs 自本 AR 起反映真实几何。

---

## 附录：本机实测数据存档（2026-10-08）

```
GPU: Quadro RTX 5000 (SM 7.5, 48 SM, 16GB), driver 556.18, WDDM
降频状态（99% util 时）: SM 645/2100 MHz, mem 405 MHz, 38.6W/230W, P8
matmul 1024^3: fp16 2.9 / bf16 1.0 / fp32 0.5 TFLOPS
matmul 4096^3: fp16 4.4 TFLOPS
显存拷贝带宽: 11 GB/s
torch.cuda.is_bf16_supported(): True（但 Turing 无原生 bf16，走仿真路径）
Triton: 原生 Windows 可编译运行（merge_stable_divergent bit-exact 通过）
nvidia-smi -lgc: 无管理员权限，失败
WSL: hypos 发行版存在，WSL1
测试: pytest -m "not slow" → 202 passed, 2 skipped, 3 deselected (9.89s, cuda)
demo.py: FLOPs reduction 78.5%, MSE 2.35e-03 (HIGH)，与 README 预期一致
```
