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
| Triton | **可在原生 Windows 编译运行**（实测 `merge_stable_divergent` bit-exact 通过，`_TRITON_MERGE_DISABLED=False`——该布尔现为手动总开关，失败降级自 AR014 起按 (kernel, dtype, hidden bucket) 键记入 `_TRITON_FAILED_KEYS`，不再自动置位） | Linux | 意外可用，但时序数字不可比 |
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
| P1-1 | **cache 同时存 layer 输入与输出，输入=上层输出** | folded_transformer.py:298-311 | 残差流上 layer L−1 的 ffn_out 恒等于 layer L 的输入；只存 `ffn_out`（+ layer 0 embedding），gate 跨层读 `ffn_out[L-1]` | **cache 显存 −50%、put 带宽 −50%**（单项最大内存收益） | 中 **✅ 已落地（AR014 勘误勾选）**：现行 `folded_transformer.py:572-574` 每层只 put `{"ffn_out"}`（layer 0 加 `embedding`），gate 跨层读 L−1（:163-167 `gate_layer = layer_idx - 1`/`gate_name = "ffn_out"`）正是本行处方——历次 AR 已完成未勾选 |
| P1-2 | cache.get 零填充被 merge 立即覆盖 | vectorized_cache.py:184-221；chunked_cache.py:221；fused_ops.py:469 | 拆分 `fetch()`（原始数据）/`fetch_masked()`（兼容层）；merge 路径走前者（需审计 verification_engine.py:185 的零填充语义依赖） | 每 layer 省 2–3 个 [B,T,H] pass | 低-中 **✅ 已落地（AR014 勘误勾选）**：三 cache 均有 `fetch()`（原始数据视图，activation_cache.py:257）+ `fetch_masked()`（get 兼容别名 :309-317）；merge/热路径全走 fetch（folded_transformer.py:176/:293/:466、architecture_utils.py:525/:533、cuda_graph.py:333/:340）；`cache.get(token_mask)` 生产零调用方（全仓 grep 仅别名转发）——见第二十一部分 |
| P1-3 | 单 pass `gather_select` 未接入主路径 | fused_ops.py:230-342 | 以 vectorized buffer 为源，主流程用 gather_select 替代「get+merge」两步 | folding 辅助 pass 从 ~5 → 2 | 中 **✅ AR002 M4b/AR011（D3 阈值 + truncated 路径接入）** |
| P1-4 | split 的 zeros 全量 + merge 再覆盖 | split_layer.py:134-139 | `out = parent_ffn.clone()` + 仅 scatter divergent 行 | split+merge 4–5 pass → 2 pass | 中 |
| P1-5 | (branch,step) 缓存无限增长 | vectorized_cache.py:224-247 + folded_generation.py | 保留 N 个最近 step 的 ring；prune 时真正 `clear_branch`（修复 B9） | 长生成显存 O(N)→O(1) | 低 **✅ 已落地（AR014 勘误勾选）**：vectorized_cache.py:63/:66 `_evict_old_branch_steps` ring 驱逐（`max_entries_per_layer`，:33 legacy 无界可关）+ folded_generation.py:170-172 prune 时 `clear_branch`——见第二十一部分 |
| P1-6 | verification engine 重复 parent 前向 | verification_engine.py:146-168 | cache 增 `contains()`；多分支共享一次 parent forward（SpiffyBaseline 式 N 分支场景） | 多分支验证成本 −50% 以上 | 低 **◐ `contains()` 已落地（AR014 勘误注）**：verification_engine.py:238 `_ensure_parent_cache` 消费 `cache.contains()`；"多分支共享一次 parent forward" 批量化主体 = P1-7 仍开放——见第二十一部分 |
| P1-7 | 多分支验证串行 | verification_engine.py + folded_generation.py:114-117 | N 个 child 拼成 batch 一次 forward（分支 batch 维对齐，core 的 cache/gate 需支持 batch 分支语义） | 多分支墙钟 ~1/N（GPU 未饱和时）；**让 folded 验证真正快于 baseline 的关键** | 中-高 **✅ AR015（第一档）+ AR016（第二档：graph 批量捕获）**：勘误——①行号陈旧：串行循环现为 folded_generation.py:139-143（:114-117 是 BranchTree 初始化）；②engine 侧无多分支循环（`verify_branch` 是 per-branch API，调用方均逐分支，点名它属清单腐化；`spiffy_baseline.py:77-86` 的多分支循环是故意的非折叠对照）；③落地：`branch_ids` 复数 kwarg 贯穿三层（`FoldedTransformerLayer._forward_batched` 批量分支语义——父 [1,T,H] expand 契约、批粒度三向拆分、逐分支 store/profiler；`ManualFoldedForward.forward` 批量入口 + graph 静默旁路；`folded_generate` `_batched_forward_supported` 谓词分发），同 parent 等长 parent-batch==1 候选单次批量前向，其余形态原样串行回退；层级调用次数 L vs N·L 为 launch-count 代理（#39，墙钟数字未实测不引用）；守护条件（同 parent/等长/Bp==1/ManualFoldedForward/plain gate+PyTorch merge/跳 gather_select）与后续档（变长/多 parent/Triton 批量/graph 批量捕获）见第二十二部分；**AR016 第二档**：批量调用不再静默旁路 graph——独立 `graph_runner_batched` 捕获 [N,T] 形状 + 重放前 parent 批量再校验（防覆写静默错值）+ 逐分支行切片 clone 发布，`cuda_graph.py` 零改动（AR002 静态前向 batch 无关红利）；**P1-7 论点首次墙钟数字**（graph vs baseline 1.37-1.77× 方向全正、3/6 配置过 2σ）见第二十三部分；**AR017 第三档**：固定容量 C=0.5 对实测 D_frac 13-16% 过配 3.0×——观测驱动重捕获（`AdaptiveCapacityPolicy` 增长/收缩/节流/退役状态机 + `last_layer_divergences` 零额外同步暂存），深几何 adaptive vs baseline **2.30×/2.46×（2/2 配置 decisively supported，delta 超 2σ 阈值 9-12×）**、P1-7 在决定性几何上的重答完成；小几何方向正 1.55×/1.68×（其中 (256,8,4) 展布受限 inconclusive——三次独立运行 delta 稳定 3.7-3.9ms），见第二十四部分；**AR018 第四档（发布成本）✅**：批量发布循环 3·L·N launches（逐行 clone/put 拷贝/sum 归约）压缩为 L·N+2（单次整块 `child_buf.clone()` 快照 + 单次 `mask_buf.sum(dim=2)` 批量归约经 `record_precomputed` 入账），发布相位占自适应全步 60.7%/71.2%/29.7%/31.5% → **45.3%/50.9%/15.7%/17.6%**，adaptive vs baseline 1.55/1.68/2.30/2.46× → **2.40/2.67/2.66/2.95×**（H2 不回退 4/4 supported，小几何抬升成立），见第二十五部分 |
| P1-8 | samplers 每步每行同步 + 逐行 topk | llada_sampler.py:131-227；dream_sampler.py:166-178；sampling_utils.py:160-210 | 批量 topk（k_max + 行掩码）；`get_num_transfer_tokens` 全向量化（一次 `.tolist()`）；LLaDA 补早停（dream 已有，行为不一致）；`x_` 提出行循环 | 每步 B 次同步 → 0–1 次；block 提前解完省整段 forward；steps=128 时省 10–100ms/请求 | 中 **✅ 已落地（AR014 勘误勾选）**：sampling_utils.py:255 批量 `torch.topk(..., dim=-1)` + :162-166 `get_num_transfer_tokens` 全向量化（docstring 自证 "no per-row or per-step host readback"）+ llada_sampler.py:194-197 早停（F10）——T018/F10 系已关闭，见第二十一部分 |
| P1-9 | gate 读带宽：parent norm 每层重算 | similarity_gate.py:86 | `cache.put` 时顺带存 per-token L2 norm（[T] 向量，可忽略） | gate 读带宽 −1/3 | 低 **✗ 失效勘误（AR014）**：前提不成立——cosine 路径单 pass 融合 `F.cosine_similarity`（similarity_gate.py:114，norm 在同一次归约内，无独立 norm 读）；l2/pearson 同为单 pass（:119/:126-133）；fused gate kernel 寄存器内计算（AGENTS #38）。无"第二次读 parent"可省——方案对象不存在，见第二十一部分 |
| P1-10 | `ManualFoldedForward` 缺 final norm | architecture_utils.py:384-395 | `detect_architecture` 增加 final_norm 发现（`model.norm`/`transformer.ln_f`/`encoder.norm`）+ 对拍回归测试（修复 B2） | 正确性 | 低 **✅ 已落地（AR014 勘误勾选）**：AR002 `detect_architecture` final_norm 发现 + `test_manual_folded_forward_applies_final_norm`（test_architecture_utils.py:216-235）+ `test_detect_architecture_finds_final_norm`（:202）——AR013 :231 勘误已证 |
| P1-11 | Triton 降级粒度过粗 | fused_ops.py:143,304-311 | 按 kernel、按 (dtype, shape 类别) 独立降级；失败一次性告警 | 可诊断性 | 低 **✅ AR014**：失败注册表 `_TRITON_FAILED_KEYS`（键 = (kernel, dtype, hidden pow2 bucket)）+ 共享 `_record_triton_failure`（每键一条 RuntimeWarning，含 dtype/hidden-class/异常）；三布尔收窄为手动总开关（失败不再自动置位）；UT-1301~1306（dtype/bucket 隔离 + launch-count spy 同键静默钉）+ T013/T006 三断言迁移——见第二十一部分 |
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

1. **变量长度折叠** ~~只支持等长、直接 `NotImplementedError`~~ **✅ AR003（append-only 前缀折叠）**：`FoldedTransformerLayer` 前缀对齐（gate 前缀比较、后缀恒 divergent、merge 前缀对齐），`BranchManager.align_tokens` 已实现（前缀对齐对），`folded_generate` 真正跨步折叠（因果合成模型 stable_ratio ≈0.84、tokens 与全重算逐位一致）。~~**遗留子项**：`Branch` 强制持有全量 `hidden_states [L,B,T,H]`（~512MB/branch @ 8B/T=1024）的惰性引用化仍未做；parent 更长的截断复用与多祖先见 P3-2。~~ **✅ AR007（parent 更长截断复用，P3-2 第一档）**：`T_p > T_c` 不再视为无 parent——gate 输入截断 `[:, :T_c]` 走既有等长分支，真前缀全 stable 时 parent FFN 截断复用（spy 实证零重算），非前缀经 gate 语义安全网全重算；`gather_select` 保守跳过（fetch+截断+merge bit-identical）；CUDA graph 零干扰（`_parent_cache_complete` 形状检查 silent eager 回退）。见第十四部分。遗留：`Branch` 全量 `hidden_states` 惰性引用化（生产零调用）；多祖先主体见 P3-2。
2. **多祖先复用**：README 局限性 #7 自认只支持单 parent。多 parent 激活树（类似 KV cache 的 paged 结构）可进一步提升复用率。**✅ 第一档（AR007）**：单 parent 更长时的截断前缀复用已收口（见上）；paged 多祖先结构仍为遗留主体。
3. **真实 draft 模型**：`DraftGenerator` 只有 random/perturb/copy_flip；接入 Medusa/Eagle 类小 draft model 是 roadmap 承诺项（与 B10 联动，suffix/logits-draft 是其前奏）。
4. **MoE 支持**：~~`flops_counter` 硬编码 4h 中间维（utils/flops_counter.py:60-63），SwiGLU（LLaDA/Dream 均是，3 矩阵×3.375h）低估 FFN FLOPs ~26%；MoE 完全未覆盖。加 `ffn_type`/`intermediate_dim` 参数并从 `config.intermediate_size` 自动读取——**当前所有 TFLOPs 数字含 10–25% 系统偏差**~~ **✅ AR005（FFN/MoE FLOPs 几何修正收口）**：`DiffusionLLM` 几何属性 + `_extract_ffn_geometry` 属性名并集（含 SwiGLU 族判定）+ `model_ffn_flops_kwargs` `underlying_model` 链解析，三调用点零改动自动供参；MoE per-token expert 计量（top-k ± shared、`first_k_dense_replace` 混合层数）；无属性路径逐位零回归。见第十二部分。

---

## 第五部分 实验方法学修复

> 项目已有两次真实模型实验（诊断 + 优化），数据扎实（不变式 bit-exact、CUDA events 计时），但存在以下方法学缺口，影响结论的可信度与可发表性。

| # | 问题 | 位置 | 修复方案 | 难度 |
|---|---|---|---|---|
| M-1 | 无方差/置信区间：所有 `results/*.json` 只有标量；`time_forward` 返回均值不存 std；accuracy 在 num_samples=10 时标准误 ~±15pp | scripts/algo_experiments.py:193-211 等 | `time_forward` 返回 `{mean,std,p50,n}`；每条件 ≥3 seed；accuracy 报 Wilson 区间；图题由数据格式化（make_optimization_figures.py:316-317 目前把 "2.3-2.7x" 硬编码进标题） | 低（脚本）/中（补跑） |
| M-2 | draft 分布与真实场景脱节（=B10） | draft_generator.py | 三层递进：suffix_append → logits_draft（parent top-k 采样模拟低质 draft）→ 真 draft model | 低→中→高 **✅ 三层全达（AR006 第一档：`model_draft` 模式 + `draft_model` callable 注入；见第十三部分）** |
| M-3 | "消融"是算术推演非实测：layerwise folding 用线性比例缩放（ablation_study.py:150-163）而非 `FoldingScheduler.disabled_layers` 真跑；cache 预算扫描根本触发不了 LRU 驱逐（四行 stable_ratio 全同） | eval/ablation_study.py:150-163,216-264 | disabled_layers 实测；cache 扫描先 put ≥budget+1 分支制造驱逐 | 低 **✅ 已落地（AR014 勘误勾选）**：`disabled_layers` 实测 + cache 扫描强制驱逐（ablation_study.py:153/:168）；AGENTS #35 明示 "sweeps are real measurements" 与 "Do not reintroduce linear extrapolation as the reported value"（linear_estimate 仅作对照列） |
| M-4 | cost model 硬件常数一刀切：所有 CUDA 设备按 100 TFLOPS/600 GB/s 计（utils/cost_model.py:31-35），实验机实际 400+/1280；且忽略 attention O(T²) 项与 KV 访存、gate/merge 误用算力计时（实为带宽主导） | cost_model.py:31-35,63-106 | `from_device` 按设备名查表 + 启动时一次 micro-bench 自动校准（复用 `exp_cost_model` 逻辑）；补 T² 项与 KV 带宽；`calibrate()` 分 compute/memory 两参数独立校准 | 低 **◐ 主体已落地（AR014 勘误注）**：设备名查表（cost_model.py:23-38）+ `HardwareProfile.from_device(calibrate=True)` micro-bench（AGENTS #33）已在；O(T²)/KV 带宽项未逐项核实——残余待独立评估 |
| M-5 | 跨版本对比（opt 前 vs 后）依赖不同时间两次运行，GPU 状态未受控 | results/optimization/baseline/ | 同进程加载两版实现或同日交错重跑（now-内的三实现对比已正确，保持） | 中 |
| M-6 | lm-eval 内部 API 依赖（`fewshot_context`/`process_results`）随版本漂移；metric key 按 `("exact_match","acc",...)` 顺序取第一个命中，gsm8k 变体 key 可能拿错 | eval/judges.py:119-134,187-223 | 按 task 显式声明 metric 映射；改走 lm-eval 官方 `simple_evaluate` 的 `LM` adapter 接口 | 中 |
| M-7 | 硬编码 AutoDL 镜像与缓存路径（`HF_ENDPOINT=hf-mirror`、`HF_HOME=/root/autodl-tmp`，7 处脚本）；Windows 上静默在 `D:\root\...` 建目录 | scripts/*.py:25-36 | 改 `--hf-home`/`--repo-override` CLI 参数 + 环境变量 fallback **✅ 已完成（T019，`scripts/_hf_env.py`）** | 极低 |
| M-8 | 评测路径不测量墙钟：`_evaluate` 只产 accuracy/tflops，无 latency/throughput；论文需要"同质量下快多少"的墙钟证据 | eval/base_adapter.py:197-235 | `_generate_one` 内包 `gpu_profile` 按路径聚合 | 低 |
| M-9 | 全局 RNG 污染：7 处 `torch.manual_seed` 重置进程级随机态 | draft_generator.py:63 等 | 统一改 `torch.Generator` 显式传递 | 低 **◐ 部分完成（AR014 勘误注）**：guide 点名位置 draft_generator.py（原 :64）已改全程局部 `torch.Generator`（AR002）；ablation_study:43 被 `measure_folding` 快照/恢复包裹（:181/:244）。**残余 5 处仍裸全局播种**：dream_sampler:106、llada_sampler:120、fast_dllm_sampler:123、diffusion_sampler:108、base_adapter:266——定性为 sampler/adapter 顶层 config.seed **显式**播种（eval 复现性语义、官方 recipe 对齐），非静默污染；若需彻底局部化另立独立小项（改语义须评估复现性契约），本 AR 不勾选整条 |

---

## 第六部分 工程质量与测试盲区

### 6.1 测试覆盖盲区（先补测试再动手优化）

- ~~`folded_generate` 多分支路径**零测试**（现有 3 例全是单分支 greedy）；`AcceptancePolicy`、`BranchTree.prune/best_accepted_leaf`（后者是死代码）无测试。~~ **✅ AR013（终审勘误 + 残余收口）**：勘误——本条前提被 AR004/AR006 大面积关闭：`test_folded_generation.py` 现 31 例（多候选 IT-512、model_draft、prune 集成、causal 等价、graph 零干扰、acceptance rate 报告全套）、`TargetMatchAcceptancePolicy` 7 个直接用例（:718-805）、`acceptance.py` 纯函数 59 hits（test_acceptance.py）。**残余由 AR013 收口**：`BranchTree` 活 API（add/get/get_ancestors/prune、`BranchNode.seq_len`）直接钉 UT-1205~1207（prune 原仅集成级间接覆盖；**实测新知**：`prune(root)` 递归清空全树含 sibling）；死代码 `best_accepted_leaf`（零调用零测试）**移除**（guide 自标死代码的正确处置是删而非补测试）+ UT-1208 移除钉防复活。见第二十部分。
- ~~samplers 只做形状 smoke test：`stochastic_transfer`、`alg_temp>0` 软选择、`remasking="random"`、CFG、suppress_tokens 均未覆盖——**恰是 P1-8 向量化改造最易破坏的分支**。~~ **✅ AR012（随机分支数值覆盖 + 勘误）**：勘误——CFG/suppress_tokens/begin_suppress/greedy/low_confidence 已由 T018（参考对拍）覆盖，本条立项时已部分过时；**真缺口是随机分支**：`add_gumbel_noise` T>0 公式、LLaDA `remasking="random"`、`stochastic_transfer=True` 采样级、Dream `alg_temp>0` 逐行 multinomial 软选择、`sample_tokens` T>0 Categorical 路径、`get_num_transfer_tokens` stochastic 数值语义——UT-1101~1106 钉死（纯函数同种子逐算子重算 bit-exact；端到端种子确定性 + RNG spy 分支命中 + 全揭示/单调结构不变量；schedule 统计均值 6σ 钉）。**钉法认知**：冻结参考对拍对随机分支是循环性钉（参考=生产逐字拷贝、共享 RNG 调用点，只证"代码未变"）且过度钉死 RNG 布局——P1-8 合法 RNG 重排会误报。见第十九部分。
- ~~`get_num_transfer_tokens` 与官方参考实现无数值对拍（AGENTS.md #21 自我要求但无落地）。~~ **✅ AR013（终审勘误）**：`test_t018_num_transfer_tokens_reference_parity`（test_sampling_utils.py:225，冻结 pre-T018 逐行参考 :80-154；官方对齐链自证见 sampling_utils.py:158/:166 "bit-identical to the official per-row loop"）+ 非随机/随机双模式钉（T018 :225 + AR012 UT-1106）已在位——本条立项时已被 T018 关闭。见第二十部分。
- ~~`ManualFoldedForward` 无 final-norm 回归测试（B2 有测试就会被抓到）。~~ **✅ AR013（终审勘误）**：`test_manual_folded_forward_applies_final_norm`（test_architecture_utils.py:216-235，folded vs raw allclose+argmax 双钉，docstring 自标 "srs 3.1 B2"）+ `test_detect_architecture_finds_final_norm`（:202）AR002 时代已钉——本条立项时已被关闭。见第二十部分。
- ~~scripts 的公共 helper（`make_child`/`time_forward`/`fidelity`）无纯函数单测。~~ **✅ AR013（勘误 + 真缺口补齐）**：勘误——`test_experiment_stats.py` 16 用例已覆盖 `TimingStats`/`stats_from_samples`/`time_forward`/`repeat_with_seed`/`exp_sampling`/`fig_sampling`（AGENTS #32 时代）；**真缺口** `make_child`/`fidelity` 由 UT-1201~1204 钉死（同种子独立重算 + 第一性原理字面常量手算 + 钳位/方差地板/dtype 边角；实测新知：恒等输入的 `cosine_similarity` 经 norm√→² 往返 ~60% 随机张量不 bit-exact 等于 1.0——恒等断言须用 norm 精确可表示张量）。见第二十部分。
- ~~`conftest.py:22-27` autouse seed 使测试共享随机态序列，测试间隐式耦合。~~ **✅ AR013（终审勘误 + 决策）**：表述失准——function-scope autouse **每测试函数重置** seed=42，是**解耦**机制（测试间不共享随机态）而非耦合源；真实残余是"测试隐式依赖 seed=42 值本身"。**决策**：新测试显式种子纪律已由 AR012 确立为模式（全部新 AR 用例显式 `manual_seed`/`Generator`）；全量重写 800 用例**否决**——churn 巨大、无 flaky 修复收益、旧测试在 autouse 下确定可复现。见第二十部分。

### 6.2 API/架构设计债

- `FoldedModel` 原地替换层（B12）：建议 context manager 协议 + `__del__` 兜底，或全面转向不替换式的 `ManualFoldedForward`（hook 方案已证明不替换也可注入逻辑，两种手段应统一）。
- ~~cache 的 `get(token_mask)` 同时承担"选择范围"与"零填充输出"两职（三个实现都要维持该契约）——正是 P1-2 冗余的根源，应拆 API。~~ **✅ AR014（终审勘误）**：拆分已落地——三 cache 均有 `fetch()`（原始数据视图）/`fetch_masked()`（get 兼容别名）；merge/热路径全走 fetch（folded_transformer/architecture_utils/cuda_graph 六处）；`cache.get(token_mask)` 生产零调用方（全仓 grep 仅别名转发）。本条与 :175 P1-2 表行同源于 AR014 勘误（两处 checkbox 均关）。见第二十一部分。
- ~~`AdaptiveQuantileGate` 与 `FoldingScheduler` 互斥但无检测（AGENTS #29）：`FoldedTransformerLayer` 应显式拒绝组合。~~ **✅ AR009（构造期守卫）**：`FoldedTransformerLayer.__init__`（`SplitFoldedTransformerLayer` 经 super 自动覆盖，`ManualFoldedForward` 构造期传递失败）与 `ActFoldVerificationEngine.__init__` 对该组合抛 `RuntimeError`（共享消息常量，含两条修复路径）；AGENTS #29 由文档约定升级为代码契约。见第十六部分。
- ~~`architecture_utils.find_layer_list` 只认 `nn.ModuleList`（:147），`nn.Sequential`/自定义容器不支持，且 auto-discovery 失败时**静默降级为 embedding 代理测量，用户不知道 folding 没生效**——应返回 `folding_status` 枚举并 WARNING。~~ **✅ AR009（勘误 + 可见性收口）**：勘误——`detect_architecture` 对层栈发现失败已显式 `RuntimeError`（architecture_utils.py:281-285，AGENTS #10 对齐），并非静默降级；`nn.Sequential`/自定义容器不支持属布局限制（AGENTS #23：显式 wiring 是既定出路），`folding_status` 枚举无消费方暂不做。真正的静默点在 `ActFoldVerificationEngine._estimate_stable_ratio` 的 embedding 代理回退——AR009 为其加实例级一次性 `UserWarning`（AGENTS #17/#18 成因指引），回退行为本身不变。见第十六部分。
- ~~`registry._resolve_family` 子串匹配（"fast" in name）与 `models/utils.infer_model_family` 双份维护；未知 diffusion 家族静默落 `causal_lm`。~~ **✅ AR010（去重 + 回退可见性）**：`_resolve_family` auto 分支委托 `infer_model_family`（utils 成为唯一规则实现源，规则逐字不变）；auto 回退 causal_lm 时发 UserWarning（含模型名 + 显式传参指引 + 可用家族列表，AGENTS #10 落实；显式 family 与已知子串命中不告警）。勘误——子串规则本身保留：fast+dllm 双子串 AND 误报面极窄，token 化反而破坏 "FastDLLM" 连写命名（lower 后单 alnum token）。见第十七部分。
- ~~`generic.forward` 恒置 `output_hidden_states=True`（generic.py:112）：7B 模型 T=512 时数百 MB 峰值浪费。~~ **✅ AR008（按需化）**：默认 `False`；输出既无 `logits` 也无 `last_hidden_state` 时置实例标志并带 `True` 递归重跑一次（每实例至多一次双跑，标志粘性安全）；LLaDA/Dream 继承链自动受益；EX-805 CUDA 实证峰值差 ≥150MB（理论 192MB）。见第十五部分。
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
| M-2 draft 分布 | draft 双模式（M1 内）+ suffix_append 作为消融缺省（T024）；真 draft model 第一档（AR006 `model_draft`） | ✅ 三层全达（Medusa/Eagle checkpoint 属第二档，需真实权重） |
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

## 第十三部分 AR006 完成回链（2026-10-10）

AR006（`specs/changes/AR006-model-draft/`，T001–T005 全部 passing）覆盖 P3-3 真 draft 模型**第一档**（M-2 三层递进第三层 / B10 最终修复 / AR004 遗留前奏）：`DraftGenerator` 新模式 `model_draft`——draft logits 来自**独立小模型**的一次 forward（`draft_model(parent.tokens)`），而非 target parent logits（`logits_draft` 作弊上界）或全词表均匀采样（`copy_flip` 平凡下界）。既有五模式逐位零回归（demo 基线不变）。

| 指南条目 | AR006 任务 | 状态 |
|---|---|---|
| M-2 第三层第一档：`model_draft` 模式 | T001（`draft_model` callable 构造参数 + generate 期 RuntimeError/三项 ValueError 校验（形状/int dtype/跨 device）+ `_sample_tokens_from_logits` 共用采样 helper + seed 复现/RNG 不污染；UT-501~505 + EX-801~803） | ✅ 完成 |
| B10 非平凡性验证 | T002（合成梯度 fixture：4 层因果 target + 1 层共享首层 draft，top_k=2；实测 model_draft 接受率 ≈0.45 严格介于 random ≈0.001 与 logits_draft 上界 0.5 之间——论文口径接受率落地） | ✅ 完成 |
| 端到端接线 | T003（config 白名单 +`model_draft`；BenchmarkRunner 可选 `draft_model` 注入 + 构造期 RuntimeError（AGENTS #10）；AblationStudy/folded_generate 经既有 `draft_generator` 参数零签名改动接入；IT-511/512 + EX-804） | ✅ 完成 |
| 文档收口 | T004（AGENTS #40 更新、CHANGELOG AR006 节、README 局限性 #5/#8 更新 + 本回链 + M-2 勾选） | ✅ 完成 |

**语义要点**：接受率数学口径——`_sample_from_topk` 是 top-k **均匀**采样，接受率 = P(target argmax ∈ draft top-k)/k（≠ argmax 命中率）；fixture 统一 top_k=2（k=1 使 logits_draft 比较器恒 1.0，k=10 使 model_draft−random gap 上界 1/k−1/V=0.099<0.1 不可达）。**fixture 结构教训**：one-hot-cumsum + 单位阵残差的共享结构使"随机权重"draft 也 0.4 相关——无关 draft 必须去掉 I 残差项（`x @ U.T`）才能落在 random 下界。append 位置从 parent 末位 draft logits 采样是第一档近似（真实 draft 逐 token 自回归属第二档）。**遗留**：Medusa/Eagle checkpoint 加载与 head 结构（第二档，需真实权重）；draft KV cache 逐 token 增量生成。

---

## 第十四部分 AR007 完成回链（2026-10-11）

AR007（`specs/changes/AR007-truncated-parent-reuse/`，T001–T004 全部 passing）覆盖 P3-2 多祖先复用**第一档**：**parent 更长时的截断前缀复用**——`T_p > T_c` 不再"视为无 parent 全重算"，而是截断 `[:, :T_c]` 走既有等长折叠分支（AR003 的对称补全：AR003 做 child 更长，AR007 做 parent 更长）。

| 指南条目 | AR007 任务 | 状态 |
|---|---|---|
| 截断复用核心 | T001（folded_transformer 三改动点：①守卫 `T_p > T_c` 截断 + `parent_longer` 标记（batch/hidden 不匹配、零长度仍 no-parent）；②全 stable 快路径 fetch parent FFN 后按实际形状截断再校验复用；③`_merge_parent_child(parent_longer=)` 禁 gather_select + `_get_parent_ffn_output` 放宽**仅限 `prefix_len is None`**（varlen 精确 shape 契约保留，EX-505 零回归）；UT-601~605：真前缀全 stable bit-exact + profiler stable_ratio==1.0 / 非前缀独立采样全重算 / batch 不匹配仍 no-parent / spy `_recompute_all` 计数==0（复用实证）/ 跨层错位 RuntimeError 钉死） | ✅ 完成 |
| 端到端与回归 | T002（IT-611 ManualFoldedForward 8-token parent → 6-token 真前缀 child `torch.equal` + 全层 stable_ratio==1.0 + cache 发布；IT-612 既有 append-only/等长套件 + 全量 759 零回归；IT-613 graph 更长 parent silent eager 回退零告警不禁用（后续 complete-parent child 仍捕获）） | ✅ 完成 |
| 文档收口 | T003（AGENTS #20 修订、README 局限性 #7 更新 + 本回链 + :199 勾选、CHANGELOG AR007 节、UT-203/bs001 陈旧 docstring 清理） | ✅ 完成 |
| 全量质量门 | T004（761 passed 零回归、demo 基线不变、mypy/pyflakes/100 列） | ✅ 完成 |

**语义要点**：截断复用**不要求**真前缀——gate 相似度是语义安全网（真前缀是收益场景：因果模型逐位一致 → 全 stable 完全复用；非前缀 divergent → 全重算 bit-exact）。快路径截断判定基于 fetch 后**实际形状**而非布尔标记（异常 cache 层间错位状态确定性兜底）。`gather_select` 在 parent 更长时保守跳过：fetch_flat 行公式（vectorized_cache.py:376-378）在非 None 返回时实际确定覆盖前 `seq_len` token，gather_select 对 parent 更长**实际正确**——留作后续性能 AR 线索。CUDA graph 零改动：`_parent_cache_complete` 形状检查天然把变长 parent 路由到 silent eager（与 AR003 变长步骤同路径）。**遗留**：paged 多祖先结构（P3-2 主体）；`Branch.hidden_states` 惰性引用化（生产零调用）。

---

## 第十五部分 AR008 完成回链（2026-10-11）

AR008（`specs/changes/AR008-generic-hidden-states-on-demand/`，T001–T004 全部 passing）收口 6.2 API 债中"generic.forward 恒置 `output_hidden_states=True`"：**真实模型前向内存峰值优化**——默认不收集全层 hidden states（7B/T=512 约 256MB fp32），仅对"既无 `logits` 也无 `last_hidden_state`"的第三类包装模型按需收集。

| 指南条目 | AR008 任务 | 状态 |
|---|---|---|
| 按需化核心 | T001（`__init__` 实例标志 `_needs_hidden_states: bool = False` + forward 默认 False/两字段缺失时置位并带 True 递归重跑一次/RuntimeError 消息逐字保留/平铺探测顺序逐行保留；UT-701~705：logits-only spy False+逐位 / last_hidden-only 同 / hidden_states-only 旗标条件式 stub 首跑双调（False→True）+次跑单调 True / 三态全无+空元组变体 RuntimeError 逐字+恰好 2 调 / branch 键剥离重跑下保持） | ✅ 完成 |
| 内存实证与回归 | T002（EX-805 CUDA：24 层无参数 stub False/True 实例对照 `max_memory_allocated` 峰值差 ≥150MB 实测通过；test_models.py 10 passed 零回归；LLaDA/Dream 继承链零回归） | ✅ 完成 |
| 文档收口 | T003（指南 6.2 勾选 + 本回链、CHANGELOG AR008 节） | ✅ 完成 |
| 全量质量门 | T004（全量零回归、demo 基线不变、mypy/pyflakes/100 列） | ✅ 完成 |

**语义要点**：重跑仅在"输出本就不可用"的路径发生（AGENTS #10 论证：修复而非静默切换）；标志单向 False→True 且粘性安全（True 态下 logits-first 模型仍走早退分支，仅付内存代价不损正确性）；train 模式双跑可接受（首跑输出不可用，重跑是唯一可用结果）。调用链覆盖（req 门控逐路径核实）：基类 AR fallback 与 LLaDA/Dream sampler 非折叠路径经继承的 generic.forward ✓；sampler 折叠路径与 `_native_sampler` 裸 model 今日即绕过 → 行为零变化。**测试教训（design 门控 WARN-1）**：hidden_states-only stub 必须旗标条件式（False → 字段 None，模拟真实 HF 语义），恒返回元组的 stub 使"首跑双调"断言成为假 Red。**遗留**：真实 7B checkpoint 实测（RERUN_CHECKLIST）。

---

## 第十六部分 AR009 完成回链（2026-10-12）

AR009（`specs/changes/AR009-silent-misconfig-guards/`，T001–T004 全部 passing）收口 6.2 API 债中两条"折叠静默失效"项：**折叠未生效/被错误配置时的可见性防护**——AGENTS #10 的直接代码落实。

| 指南条目 | AR009 任务 | 状态 |
|---|---|---|
| :239 互斥检测 | T001（`_ADAPTIVE_GATE_SCHEDULER_MSG` 模块级常量 + `FoldedTransformerLayer.__init__` 守卫（super 后、属性赋值前 fail-fast，`isinstance` 覆盖子类，Split 经 super 自动覆盖，`ManualFoldedForward` 构造期传递失败）+ `ActFoldVerificationEngine.__init__` 同守卫；UT-804/805：守卫触发 + 消息关键词 `gate.tau` + 四种合法组合对照钉死） | ✅ 完成 |
| :240 静默降级 | T001（`_estimate_stable_ratio` embedding 代理回退分支入口加实例级一次性 `UserWarning`（`_proxy_ratio_warned` 标志，AR002 `_graph_degraded_warned` 先例；消息含 AGENTS #17/#18 成因指引；`stacklevel=3` 归因用户调用点）；UT-801/802/803：同实例恰好 1 条 + 双实例 2 条（实例级语义）+ 层级 profile 零告警且返回 profiler 均值；UT-801 返回值逐位等于独立计算的 gate 均值（防恒真）） | ✅ 完成 |
| 勘误 | :240 前提过时——`detect_architecture` 对层栈失败已显式 `RuntimeError`（architecture_utils.py:281-285）；`nn.Sequential` 不支持属布局限制（#23 显式 wiring）；`folding_status` 枚举无消费方不做（本 AR 在 :240 勾选注中勘误） | ✅ 完成 |
| 文档收口 | T003（指南 6.2 双条目勾选 + 本回链、AGENTS #29 修订为代码契约、CHANGELOG AR009 节） | ✅ 完成 |
| 全量质量门 | T004（全量 773 passed = 768+5 零回归、demo 85.5%/2.35e-03/93.75% 精确一致、mypy clean/pyflakes 零/100 列） | ✅ 完成 |

**语义要点**：守卫是纯构造期校验——生产消费方（AblationStudy/BenchmarkRunner/demo）均以 `SimilarityGate` 构造折叠栈，零回归面（全库 15 处 `AdaptiveQuantileGate(` 构造均无 scheduler 组合已核实）；告警是**有意新增**的可观测行为（非泄漏），既有 engine 测试（UT-306/307 等走代理路径）开始收到一次性告警但 pyproject 无 `-W error` 不受影响；回退行为本身逐行不变（AGENTS #16/#17 写明的合法降级路径，只加可见性）。**设计要点**：`isinstance` 黑名单语义（子类同样忽略固定 tau）与 #38 fused dispatch 的精确类型白名单意图相反需区分；消息常量两构造器逐字一致防漂移（speculative→core 单向导入无环）。**遗留**：无（两守卫与告警即终态）。

---

## 第十七部分 AR010 完成回链（2026-10-12）

AR010（`specs/changes/AR010-model-family-inference-unification/`，T001–T004 全部 passing）收口 6.2 API 债 :241：**模型家族推断统一与回退可见性**——消除 `_resolve_family`/`infer_model_family` 的逐行双份维护，让 auto 回退的静默默认可见（AGENTS #10 的又一落实，与 AR009 同主题连续）。

| 指南条目 | AR010 任务 | 状态 |
|---|---|---|
| 双份维护消除 | T001（`_resolve_family` auto 分支改为 `return infer_model_family(...)`，registry 侧重复子串规则整体删除；utils 成为唯一实现源；既有 `test_registry_resolve_family` 5 断言零改动全过 = 行为不变硬门；UT-901 六组等价矩阵含混合大小写/连写/显式透传） | ✅ 完成 |
| 回退可见性 | T001（auto 且推断结果为 causal_lm → UserWarning，`stacklevel=3` 经 load 归因用户调用点；消息含 "auto-detect fell back to causal_lm" + 模型名 + 显式传参指引 + `list_models()` 动态家族列表；判据依赖 `infer_model_family` 不变量"返回 causal_lm 当且仅当无已知子串命中"，注记在 registry 侧告警块；UT-902 `pytest.warns(match=...)` + 消息含模型名双断言；UT-903/904 以 `simplefilter("error")` 钉死显式 family 与已知子串命中零告警） | ✅ 完成 |
| 子串规则勘误 | :241 点名的 "'fast' in name" 子串匹配经分析**保留**——fast+dllm 双子串 AND 误报面极窄（需同时含两子串且非 Fast-dLLM）；token 化破坏 "FastDLLM" 连写命名（lower→`fastdllm` 单 alnum token，子串命中 ✓/token 等值 ✗）；基于 `AutoConfig.model_type` 的二次校验因需权重/config 可得而 Out of Scope | ✅ 完成 |
| 文档收口 | T003（指南 :241 勾选 + 本回链、CHANGELOG AR010 节；README 经审查不需更——现有文档只示例显式 family，无 auto 行为描述） | ✅ 完成 |
| 全量质量门 | T004（全量 782 passed = 773+9 零回归、demo 85.5%/2.35e-03/93.75% 精确一致、mypy clean/pyflakes 零/100 列） | ✅ 完成 |

**语义要点**：`infer_model_family` 保持纯函数（告警在 registry 层——唯一同时握有 `model_family` 原值与推断结果的点）；回退**返回值不变**（真因果模型继续正确落 causal_lm，告警是提示而非强制）；registry 导入 utils 无环（utils 零 actfold 内部依赖，req/design 门控双核实）；scripts 全部显式传 `spec["family"]`（grep 核实），批量加载无告警噪音。**测试形态**：UT-901/903/904 为回归钉子（Red 期即过——规则未变/显式与命中路径今日就不告警），UT-902 是唯一真 Red（`DID NOT WARN`）——与 AR009 UT-803 钉子模式互为印证。**遗留**：无。

---

## 第十八部分 AR011 完成回链（2026-10-12）

AR011（`specs/changes/AR011-gather-select-truncated-path/`，T001–T004 全部 passing）收口 AR007 的 review 线索：**截断路径 gather_select 解禁**——`_merge_parent_child` 的 D3 分发条件删除 `not parent_longer`，截断复用在大形状下恢复单遍融合合并。

| 条目 | AR011 任务 | 状态 |
|---|---|---|
| 解禁核心 | T001（分发条件删除 `and not parent_longer` 一行，其余逐字保留；docstring 勘误——`fetch_flat` 守卫要求**覆盖**而非**恰好** seq_len；`vectorized_cache.py`/`fused_ops.py` 零改动——`fetch_flat` 行公式 `flat_rows[b*seq_len+t]=t*batch+b` 按请求子长计算，天然读父缓存前 T_c token = AR007 截断前缀，req 门控逐行核实） | ✅ 完成 |
| 安全网钉死 | T001（UT-1002 环回驱逐：独立 capacity=6 构造 + **幸存窗口前缀 child**（环回 fetch 返回 token 2..7 而非 0..3——直觉构造会全 divergent 空转，req 门控 Major）+ 输出组成 sanity → `fetch_flat` start>0 返回 None → gather_select 零调用 + torch.equal；UT-1003 varlen 不可融合行为钉子） | ✅ 完成 |
| 等价证据 | T001（UT-1001：parent-longer 融合恰一次调用 + 前向前混合稳定 sanity（防 tau=0.95 小扰动 flaky）+ 双实例 torch.equal + 组成 sanity；T014~T016/UT-601~605/IT-611/613 既有全套零改动全过） | ✅ 完成 |
| 文档收口 | T003（AGENTS #20 修订"conservatively skipped"→解禁表述、**#30 阈值勘误**（历史"H≥8192, or T≥8192"→现行 D3 值 T≥2048/H≥4096/CUDA，T016 钉死）、本回链、CHANGELOG AR011 节） | ✅ 完成 |
| 全量质量门 | T004（全量 785 passed = 782+3 零回归、demo 三指标精确一致、mypy clean/pyflakes 零/100 列） | ✅ 完成 |

**语义要点**：与等长路径**同一**融合调用（行公式已保证语义，不分叉实现）；环回（start>0）/varlen（total<seq_len）/非向量化缓存（无 fetch_flat）三类安全网全部由既有守卫拦截，行为零变化；`parent_longer` 参数保留（不再门控分发，语义/诊断价值，签名零改动）。**P3-2 系列注**：截断复用（AR007）现已完整覆盖融合路径（AR011），paged 多祖先结构仍是后续大项。**遗留**：无。

---

## 第十九部分 AR012 随机分支数值覆盖（2026-10-12）

AR012（`specs/changes/AR012-sampler-stochastic-coverage/`，纯测试 AR，`actfold/` 零改动）：补齐 6.1 :229 勘误后的真缺口——samplers **随机分支**（RNG 消耗路径）的数值级测试，UT-1101~1106（新文件 `tests/test_sampler_stochastic_branches.py`，14 参数化用例）。

| 项目 | AR012 交付 | 状态 |
|---|---|---|
| 纯函数公式钉 | UT-1101（`add_gumbel_noise` T∈{0.5,1,2} 同种子独立重算逐位相等 + float64 + T=0.0 `is` 同对象 no-op——:206-207 原仅有形状断言）；UT-1105（`sample_tokens` T>0 六组合含 top_p+top_k 联合（镜像生产顺序 top_p 先）：双返回值逐位重算 + margin/neg_entropy 覆写 + -inf 位概率精确 0） | ✅ 完成 |
| 端到端随机分支 | UT-1102（LLaDA `remasking="random"`：种子确定性 + `torch.rand` spy shape 钉 + 全揭示 + suppress 组合）；UT-1103（`stochastic_transfer=True`+T=0.5：Binomial 代理 spy + gumbel 在位 + history 单调）；UT-1104（Dream `alg_temp=1.0`：multinomial spy + 全揭示 + **弱收敛性质**（alg_temp=1e-8 时 softmax 精确 one-hot、每行首采样==置信度 argmax）） | ✅ 完成 |
| schedule 数值钉 | UT-1106（同种子 `torch.equal` + stochastic≠deterministic 分支可区分（S=0 钉死）+ 100 种子统计均值 |2.70−2.5000167|=0.20 ≤ 6σ=±0.89，p₀=0.12500083446502686 独立重算逐位复刻生产四次 f32 舍入） | ✅ 完成 |
| 测试自证 | mutation spot-check 三位点全翻红后逐字节还原：remasking 判断反义→UT-1102 spy 空；stochastic 硬编码→UT-1103 spy 空；alg_temp 条件改 `!=`→UT-1104 spy 空（spy 空断言是"空转通过"防护——AR011 教训落地） | ✅ 完成 |
| 门控 | req 门控 Major-1（冻结参考与生产共享 RNG 调用点、同种子 bit-equal——对拍是循环性钉而非"不可行"，依据更正）；design 门控 Major-1（统计钉构造 Σn 与常数矛盾，首跑必红陷阱——单行全 mask 修正）；两门控 FAIL→修复→PASS | ✅ 完成 |
| 质量门 | 全量 799 passed（785+14）零回归、demo 三指标精确一致、mypy clean/pyflakes 零/100 列 | ✅ 完成 |

**语义要点**：随机分支钉"当前平台确定契约"（同种子 bit-equal）而非跨平台分布正确性（AGENTS #21 免责仍立）；独立重算逐算子手写不 import 生产函数（防循环验证）；`(4,20)` 统计构造陷阱（Σn≈48 vs 硬编码 n=20 期望——首跑必红且伪装生产缺陷）由 design 门控拦截。**遗留**：:228 多分支路径、:230 官方对拍、:231 final-norm、:232 scripts helper、:233 conftest seed 耦合仍在债清单。

---

## 第二十部分 AR013 测试债清单终审收口（2026-10-12）

AR013（`specs/changes/AR013-test-debt-closure/`，混合 AR：测试为主 + branch_tree.py 删 6 行死代码）：6.1 清单 **6/6 终态**——AR012 关闭 :229 时发现清单大面积过时（AR008-012 连续五次发现债条目前提腐化），本 AR 对整个 6.1 做系统性终审：:228/:230/:231/:233 四条**勘误勾选**（各自证据与实际关闭者写入条目）、:232 勘误 + 真缺口补齐。

| 项目 | AR013 交付 | 状态 |
|---|---|---|
| make_child/fidelity 纯函数钉 | UT-1201~1204（`tests/test_algo_experiment_helpers.py`，168 行）：同种子独立重算（Generator→randperm→randint 序列逐位）+ 双调用确定 + seed 可区分；边角矩阵（零/负翻转 clone 非同对象、钳位、vocab=1、父不可变、float32 dtype 直通）；fidelity 恒等/方差地板/键集/fp16 上转 + 手算例（mse=0.25/top1=0.5 精确，rel_mse/cosine approx——第一性原理字面常量，var 用显式公式镜像而非 `base.var()` 防 torch 默认漂移同步） | ✅ 完成 |
| BranchTree 活 API 直接钉 | UT-1205~1207（`tests/test_branch_tree.py`，149 行）：add/get 往返、父子链接、重复 add 防护（:57-58）、孤儿条件（:55）、`BranchNode.seq_len` 值钉、get_ancestors 顺序/include_self/未注册双态、prune 递归+解链+no-op+dangling root_id；**实测新知**：`prune(root)` 递归清空全树（sibling 一并移除）——按现行行为钉并 docstring 标注非应然 | ✅ 完成 |
| 死代码移除 | `branch_tree.py:88-93` `best_accepted_leaf` 删除（-6 行，本 AR 唯一生产改动；仓级扫描零调用零测试）；UT-1208 移除钉（hasattr False + 活 API 面全真）防复活；删除前红/删除后绿为移除生效的天然证据 | ✅ 完成 |
| mutation 自证 | 三位点均翻红后逐字节还原：make_child 删钳位行 → RuntimeError shape mismatch 15 vs 12；fidelity 地板→var → ZeroDivisionError（Python float 除零非 inf）；prune 删递归 → leaf 残留 | ✅ 完成 |
| 门控 | req 首轮 PASS（5 Minor 计数/措辞修正：TargetMatch 7 例非 8、experiment_stats 16 例非 17、:230 出处链精确化）；design 首轮 FAIL→修复（**Major-1**：`seq_len` 归属 BranchNode 非 BranchTree，活 API 面字面实现必红；**Major-2**：恒等 cosine 精确断言 ~60% 随机张量不成立——改手选 norm 精确可表示张量） | ✅ 完成 |
| 质量门 | 全量 **807 passed（799+8）**零回归、demo 三指标精确一致、mypy clean/pyflakes 零/100 列、`git diff actfold` 恰 -6 行 | ✅ 完成 |

**语义要点**：死代码的正确处置是**移除而非补测试**（给无人使用的路径上保险是反向激励）；6.1 清单恢复可信度后，剩余债集中在 6.2（:237/:238/:243）与大项（P3-2 paged 多祖先、P2-1 超融合）。**遗留**：无（6.1 全清）。

---

## 第二十一部分 AR014 Triton 降级粒度细化 + 优先级表勘误（2026-10-13）

| 项 | 内容 | 状态 |
|---|---|---|
| 主题 | P1-11 落地（fused_ops 三 Triton kernel 失败降级从"每 kernel 全局布尔一次失败永久全域禁用"细化为按 (kernel, dtype, hidden pow2 bucket) 键独立降级 + 每键一次性告警）+ P1/M 优先级表系统性勘误（连续第三个 AR 发现清单腐化，本轮扩展到优先级表） | ✅ 完成 |
| 生产改动 | `fused_ops.py` 唯一（58+/31-）：`_TRITON_FAILED_KEYS` 注册表 + `_hidden_bucket`（pow2 上取整类别）+ `_record_triton_failure` 共享 helper（每键一条 RuntimeWarning，含 kernel/dtype/hidden-class/异常，stacklevel=3 对齐落点）；merge（:193 解包后插入）/gather_select/gate（use_triton 表达式）三入口；三布尔收窄为**手动总开关**（失败不再自动置位，`global` 声明移除）；docstring "permanently disables" 两处更正 | ✅ 完成 |
| 新测试 | UT-1301~1306（`tests/test_triton_fallback_granularity.py`，284 行 10 项）：dtype 隔离三步序列（失败路径合法回退→forbidden 区间真 Triton 自证→恢复后第二失败键第二告警）、同键静默（**launch-count spy 计数==0** 直接钉"不尝试 kernel"）、bucket 隔离（512 爆/2048 真）、总开关兼容（monkeypatch 强制回退零告警零增长）、告警三要素可诊断、bucket 纯函数 CPU 钉 | ✅ 完成 |
| 既有钉迁移 | T013/T006 三处布尔断言→注册表键断言（实测键值 gather_select/32、merge/128、gate/64）+ 布尔 `is False` + setattr 注册表隔离；两处存在性钉补 `isinstance(..., set)`；**零改动消费点**四处实测通过（test_ar002_perf_proxy:365、test_cuda_graph:596、test_fused_gate:217、test_fused_ops:440-461——后者向进程级注册表泄漏一键，与现行布尔泄漏同构、粒度更细，风险降不升） | ✅ 完成 |
| mutation 自证 | 三变异均翻红后逐字节还原：①注册表不写入→3 例红（注册表空集断言）；②分发不查→1 例红（UT-1302 spy 计数==1，精确命中设计观测面）；③键缺 dtype 维→4 例红（dtype 隔离面+消息面）。"告警去条件"变异黑盒不可达（dispatch 使 helper 对已注册键不可达）——从集合移除并记录，guard 保留为纵深防御 | ✅ 完成 |
| 优先级表勘误 | P1-1/P1-2/P1-5/P1-8/P1-10 勾选（历次 AR 已落地未勾选）；P1-9 **✗ 失效勘误**（单 pass 融合算子下无独立 norm 读可省——方案对象不存在）；P1-6/P1-3/M-4 ◐（contains/D3 接入/设备表已落地，批量化主体归 P1-7 等仍开放）；M-3 勾选；M-9 ◐（点名位置已修，残余 5 处为顶层显式 seed 定性，不勾选整条——req 门控 Major-2 教训：七处仅证两处即宣称"已落地"即复制清单腐化）；:238 与 :175 P1-2 两处 checkbox 同关 | ✅ 完成 |
| 门控 | req 首轮 FAIL→修复→复核 PASS（Major-1 FR6 漏 :175 表行/Major-2 M-9 过度声明；12 项全 RESOLVED）；design 首轮 FAIL→修复→复核新 Major-3（"全程 forbidden"阻断失败路径合法回退——**失败注入测试的 forbidden 区间必须精确圈定在成功路径调用上**）→免重读放行 | ✅ 完成 |
| 质量门 | 全量 **817 passed（807+10）**/3 skipped 零回归、demo 三指标精确一致（85.5%/2.35e-03/93.75%）、mypy clean（65 文件）/pyflakes 零/100 列、`git diff actfold` 仅 fused_ops.py | ✅ 完成 |

**语义要点**：(1) 失败降级的可观测投影是 **launch 计数**而非告警缺席——同键静默有 dispatch/helper 双守卫，仅观察告警无法区分"移除其一"的变异（design 门控 Major-1 教训）。(2) 债清单腐化是**系统性**的：6.1（AR013）、6.2+P1/M 表（AR014）连续三轮大面积过时，选题勘察逐行核实不可省；整条勾选前必须逐子项取证（M-9 教训）。(3) P1-9 类"优化点"可能**前提失效**——勘误注明失效比照单实现更有价值。**遗留**：P1-7 多分支批量验证（真遗留主体，中-高）；P1-12 tuple 契约/KV 告警（部分由 AGENTS #13 覆盖）；P2-4 baseline KV cache；P3-2 paged 多祖先；P3-3 真 draft checkpoint；:237/:243（低值）。

---

## 第二十二部分 AR015 folded_generate 多候选批量分支前向（P1-7 第一档）（2026-10-13）

| 项 | 内容 | 状态 |
|---|---|---|
| 主题 | P1-7 落地（第一档）：`folded_generate` 验证阶段多候选串行前向（:139-143 逐候选 `_run_folded_forward`，N-1 个被拒候选的完整前向纯浪费）批量化——同 parent 等长候选拼 [N,T] batch 一次 folded forward，core 层新增批量分支语义，不满足前置条件时原样串行回退 | ✅ 完成 |
| 勘误（P1-7 行） | ①行号 :114-117 → :139-143（陈旧）；②engine 侧无多分支循环（per-branch `verify_branch` API 属清单腐化；spiffy_baseline 多分支循环是故意非折叠对照）；③"~1/N 墙钟"以 launch-count 代理钉住（层级调用 L vs N·L，#39 纪律——本机 CUPTI 受限，未实测墙钟不引用数字） | ✅ 完成 |
| 生产改动 | 三文件：`folded_transformer.py`（`branch_ids` keyword-only 入口 + `_forward_batched` 批量分支语义：父 [1,T,H] **expand 契约**（Bp==1 违者 RuntimeError）、gate 固定 plain 路径（stride-0 expand 不进 fused kernel）、批粒度三向拆分、`_merge_parent_child_batched` **三段式契约**（fetch 期校验父存储 [1,·] → expand → post-expand 校验 [N, prefix_len]）、逐分支 store/profiler；串行 forward 逐位不动）、`architecture_utils.py`（`branch_ids` 形参 + 单复数互斥 ValueError + graph 门 `and branch_id is not None` 静默旁路 + 层循环无条件传参）、`folded_generation.py`（`_batched_forward_supported` 谓词：ManualFoldedForward/N>1/同 parent/等长/Bp==1 + `_run_folded_forward_batched` 助手 + 分发；串行循环逐字保留）；`_ACTFOLD_KWARGS` 增 `branch_ids` | ✅ 完成 |
| 新测试 | UT-1401~1412（`tests/test_batched_branch_forward.py`，960 行；UT-1409~1412 为 review 门控增补）：CPU varlen 逐位对拍（logits/cache/profiler 统计）、CUDA allclose(1e-6)+argmax、等长+split 变体、launch-count 代理（L vs N·L）、folded_generate 端到端（批量 vs 强制串行 tokens/ratio/acceptance 全等 + cache 被拒分支零残留）、回退矩阵（谓词五情形 False + greedy spy）、层级前置契约（ValueError×2/RuntimeError/无 parent 与 cache-miss 双对拍——生成循环第一步 root 缓存空形态）、graph 旁路（批量后 graph_runner 零变化 + 串行仍可 capture）、scheduler 禁用层批量同构（动态 tau 复用 + profile 只含折叠层）、全稳快路径（`_recompute_all` spy 钉零重算——真前缀子代重算值与父复用值逐位相同，调用计数是唯一可观测面）、AR007 截断全稳、AR007 截断混合 merge（删截断 → RuntimeError） | ✅ 完成 |
| mutation 自证 | Red 期四变异+review 期五变异均翻红后逐字节还原（SHA256 identical）：①删 expand→UT-1401/1403 红（gate shape ValueError——**三段式契约即可观测性**，PyTorch 广播本会静默通过）；②整批存单一 branch_id→cache 对拍红；③跳逐分支 profiler→profile None 红；④删分发→UT-1405 探针红（UT-1404 直调不经谓词不红——设计表勘误：④ 观测面是 UT-1405）；review 增补：⑤禁用批量 scheduler 分支→UT-1409 红；⑥删全稳 expand→UT-1410/1411 红；⑦删快路径 FFN 截断→UT-1411 红（回退重算被 spy 捕获）；⑧删 merge 内 AR007 截断→UT-1412 红；⑨禁用批量动态 tau→UT-1409 红 | ✅ 完成 |
| T002 实测勘误 | **split 聚集 GEMM 的 gemv/gemm 微内核边界**：串行候选恰 1 个 divergent 行时 M=1 走 BLAS gemv、批量 M=D_total>1 走 gemm → 1 ULP（L0 实测 2.4e-07）——"Split 层 CPU 逐位"假设数值性证伪，UT-1403 收紧为 allclose(1e-6)+argmax（mask/profiler 统计仍精确相等；非 split 路径 UT-1401 CPU 逐位不受影响） | ✅ 完成 |
| 门控 | req 首轮 FAIL→修复→复核 PASS（**Major-1**：demo 锚点定性错误——demo 不经 folded_generate，NFR 理由改写为锚定 FR1/FR2 串行层路径；**Major-2**：FR1 缺"parent id 非空但缓存 miss"分支——生成循环第一步必然触发，补验收项+UT-1407 用例）；design 首轮 FAIL（**Major-1**：批量父 FFN 契约两种字面规格互斥→三段式唯一化）→复核二轮 FAIL（③ varlen 半边 [N,T] vs [N,T_p] 互斥→[N, prefix_len] 统一）→三轮 PASS；review 首轮 FAIL→修复→复核 PASS（**Major-1** FR1 scheduler 验收项零覆盖→UT-1409；**Major-2** FR1 全稳三分支含 AR007 截断零覆盖→UT-1410/1411/1412——生产代码经 git diff 0a66dd9 独立验证全合规；Minor-1~5 措辞/簿记全修） | ✅ 完成 |
| 质量门 | 全量 **829 passed（817+12）**/3 skipped 零回归（既有测试零改动）、demo 三指标精确一致（85.5%/2.35e-03/93.75%——demo 不经 folded_generate，锚定串行层路径逐位不动）、mypy clean（65 文件）/pyflakes 零/100 列 | ✅ 完成 |

**语义要点**：(1) **第一档守护条件**：同 parent、等长、parent batch==1、`ManualFoldedForward`、plain gate + PyTorch merge（直调 `_merge_stable_divergent_torch`，公开面不扩）、跳 gather_select（`fetch_flat` 按父存储批量计行对 [N,T] 恒 None）、graph 静默 eager；变长/多 parent/Triton 批量为后续档，**graph 批量捕获 ✅ AR016（见第二十三部分）**。(2) **显式 expand + 严格双校验是可测性来源**——PyTorch 广播会让"删 expand"静默通过，三段式契约（fetch 期 [1,·] + post-expand [N, prefix_len]）把变异变成硬错误。(3) **cache-miss 分支是旗舰路径而非边角**：生成循环第一步 root 缓存为空（draft 模式不对 parent 前向），req 门控 Major-2 教训。(4) **CPU 逐位 ≠ 无条件**：split 聚集路径的 GEMM M 维跨微内核边界即 1 ULP——对拍容差按路径分档（UT-1401 逐位 / UT-1403 allclose），与 CUDA 容差同类。**遗留**：P1-12 tuple 契约/KV 告警；P2-4 baseline KV cache；P3-2 paged 多祖先；P3-3 真 draft checkpoint；:237/:243（低值）；变长/多 parent 批量与 Triton 批量 kernel（P1-7 后续档）。

---

## 第二十三部分 AR016 批量 CUDA graph 捕获 + P1-7 首次墙钟实证（2026-10-13）

| 项 | 内容 | 状态 |
|---|---|---|
| 主题 | P1-7 落地（第二档）：AR015 的批量分支前向**静默旁路 CUDA graph**（`_graph_forward_or_none` 只认单数 `branch_id`）——本 AR 为 [N,T] 批量形状建立独立 `graph_runner_batched` 捕获/重放/发布路径，`cuda_graph.py` **零改动**（AR002 静态前向设计 batch 无关红利：capture 体 :206-209 按 tokens.shape 取 batch、静态缓冲 [L(+1),batch,seq,H]、`_prefill_parent` copy_ 广播 [1,T,H]→[N,T,H]），并以四路径墙钟基准给 P1-7 论点（"让 folded 验证真正快于 baseline"）**首次实测数字** | ✅ 完成 |
| 生产改动 | `architecture_utils.py` 唯一（+150 行级）：`forward` 批量分支插入 graph 分发（拒绝自然落入 eager 体——parent 批量 != 1 的终点仍是 #41 RuntimeError，graph 层不吞不改）；`_graph_forward_batched_or_none`（镜像串行退化矩阵 + **重放前 parent 批量再校验**——捕获后 parent 被同 id 覆写为 [N,T,H] 时 `copy_` 同形拷贝会静默成功使每分支对不同父行折叠、预算校验拦不住，宿主侧 shape 检查每次重放前拒绝，ChunkedActivationCache 下覆写合法可达即真实生产面）；`_parent_cache_complete_batched`（期望形状 [1,seq]——expand 契约 #41 的 graph 侧镜像）；`_publish_graph_children`（逐分支 [1,T,H] 行切片 **clone** 发布 + profiler record mask 行切片——形状与 AR015 eager 批量 store 逐位一致）；5 新属性（独立 runner/mask/形状/预算/捕获失败标志；`_graph_degraded_warned` 家族标志共享——单一布尔跨原因抑制系串行先例） | ✅ 完成 |
| 新测试 | UT-1501~1509（`tests/test_batched_graph_forward.py`，925 行）：捕获-重放 vs 批量 eager 对拍（1e-4+argmax）+ capture spy 单次、发布完备性（embedding 行 1e-6）、串行 graph 共存零干扰 + **批量 graph 各行 vs 串行 graph 逐行直接对拍**（FR3 字面）、形状不匹配一次性告警、预算超出丢弃重算、前置矩阵（scheduler/gate/CPU/parent 空/mask 对象不匹配/parent 批量 2→eager #41 RuntimeError）、EX-002 批量版（一次性 capture 失败 + 串行侧不受扰）、parent 批量 != 1 捕获时+重放前双查（**UT-1508(b) 用 ChunkedActivationCache**——ActivationCache 的 put 对同 key 批量增大抛 RuntimeError（:154）是异常面，chunked 整 chunk 替换（:133-135）覆写静默成功才是 Major-1 洞的真实面；raise 先于任何 store → 零污染 + parent 恢复后重放成功）、发布隔离（mutate child_buf 不影响已发布 cache） | ✅ 完成 |
| mutation 自证 | ①-⑤ 均翻红后 SHA256 逐字节还原（42906736…99993 五次一致）：①删分发→UT-1501 红（capture spy 0）；②发布跳 clone→UT-1509 红；③发布跳 row 1→UT-1502 红（完备性）；④删预算回退→UT-1505 红；⑤删重放前再校验→UT-1508(b) 红（不抛错+cache 污染——Major-1 直接观测面） | ✅ 完成 |
| 墙钟基准 | `scripts/ar016_batched_bench.py`（`python -m scripts.ar016_batched_bench`，--quick 冒烟开关）：四路径 × N∈{2,4,8} × T∈{128,512}，每重复四路径 round-robin 交错（RERUN_CHECKLIST §4.2），perf_counter+sync 计时（**不用 CUDA-event `time_forward`**——后者排除宿主侧 prefill/读回/发布，而批量 graph 净收益恰含这些项），TimingStats n=10，全量 19s（预算 <10min）；工件 `results/optimization/ar016_batched_bench.json`（运行日期/CUPTI null+注记/交错声明/假设声明与判定分离） | ✅ 完成 |
| **实测墙钟表**（p50 ms，Quadro RTX 5000，seed 2026，完整数据见工件） | 见下 | ✅ 完成 |
| 门控 | req 三轮（**Major-1** 重放期 parent 批量校验缺失→修复引入**Major-R1** 契约矛盾——"静默 eager 成功回退"对 parent 批量 != 1 不可实现（eager 本身 #41 RuntimeError），终点拆子例：graph 层静默拒绝+调用方收 eager 原样报错→三轮 PASS）；design 三轮（M-1~4 UT 场景/编号/容差/假设标签→复核暴露 **Major-N1**：UT-1508(b) 覆写对 ActivationCache put 不可达（子代理实测），换 ChunkedActivationCache 夹具→PASS） | ✅ 完成 |
| 质量门 | 全量 **838 passed（829+9）**/3 skipped 零回归（既有测试零改动）、demo 三指标精确一致（85.5%/2.35e-03/93.75%）、mypy clean（65 文件）/pyflakes 零/100 列、`cuda_graph.py` 零改动（git status 证明） | ✅ 完成 |

**实测墙钟表（每步 = N 候选完整验证，p50 ms，本机 WDDM 无锁频，n=10 round-robin 交错）**：

| 配置 | baseline eager | folded 串行 | folded 批量 eager | folded 批量 graph | graph vs baseline |
|------|---------------|-------------|-------------------|-------------------|-------------------|
| N=2, T=128 | 7.00 | 25.37 | 12.18 | 4.67 | 1.50× |
| N=2, T=512 | 7.02 | 24.42 | 12.29 | 5.14 | 1.37× |
| N=4, T=128 | 12.51 | 50.34 | 14.55 | 7.49 | 1.67× |
| N=4, T=512 | 12.22 | 47.06 | 14.35 | 8.35 | 1.46× |
| N=8, T=128 | 23.57 | 100.64 | 18.24 | 13.28 | 1.77× |
| N=8, T=512 | 25.42 | 101.10 | 17.98 | 17.37 | 1.46× |

**假设判定（基于实测，不预设结论）**：
- **H1 批量 eager ≫ 串行（~N× 方向）：supported（6/6 配置）**——AR015 墙钟红利首次实证（串行 25-101ms → 批量 12-18ms，2.0-5.6×）；
- **H2 批量 graph 进一步降低每步墙钟：inconclusive（4/6 supported，N=8 两配置未过 2σ）**——graph 对批量 eager 的优势随 N×T 增大收窄（小形状 2.6× → 大形状 ~1.04×；重放本体恒快，但 [N,T] 大形状下宿主侧 prefill copy + 预算读回 + N×L 次发布占比上升）；
- **H3 四路径 logits 一致：supported（6/6 argmax 精确一致 + folded 三路径 allclose 1e-4）**——质量不因加速路径漂移；
- **P1-7 专项（graph vs baseline）：inconclusive（方向全正 1.37-1.77×，3/6 配置过 2σ）**——**首次以数字回应"让 folded 验证真正快于 baseline"**：批量 graph 是四路径中唯一快于裸 baseline 的 folded 路径，但统计显著性与配置相关；诚实判定 inconclusive 而非宣称胜利。

**语义要点**：(1) **重放前再校验是 graph 批量路径的正确性核心**——串行 runner 靠形状不匹配天然炸错，批量 runner 形状恰好匹配 [N,T,H] 覆写 → copy_ 静默同形拷贝 → 每分支对不同父行折叠且预算自洽；防线在分发层（每次重放前宿主侧 shape 检查，零 D2H 同步），不在 runner 内。(2) **终点行为拆子例**：graph 层是优化层——拒绝即可，不吞不改 eager 契约（parent 批量 != 1 的 #41 RuntimeError 原样透传给调用方；"静默 eager 成功"只对 seq 不匹配/缓存不完整成立）。(3) **计时口径决定结论**：CUDA-event 计时排除宿主侧 prefill/`tolist` 读回/逐分支发布——H2 在大 N×T 下的收窄恰是这些宿主侧成本占比上升的体现，用 `time_forward`（CUDA-event）会系统性高估 graph 优势。(4) **夹具 cache 语义影响可达性**：同 id 批量增大覆写在 ChunkedActivationCache 合法（整 chunk 替换）、在 ActivationCache 抛错（copy_ 广播不匹配）——测试场景构造必须核对具体 cache 的 put 覆盖语义（design 门控 Major-N1，子代理实测发现）。(5) **ActivationCache 反方向（先 [N,T] 再 [1,T] 同 key）是静默广播覆写**——既有隐患面，本 AR 的重放前再校验恰好覆盖，后续可在 cache 层收口。**遗留**：变长/多 parent 批量、Triton 批量 kernel（P1-7 其余后续档）；真实 checkpoint 端到端墙钟（本基准为合成 LLaMA-layout 模型）；H2 大 N×T 收窄的宿主侧成本分解（prefill/读回/发布占比定量；固定容量过配项 ✅ AR017 收口——见第二十四部分）。

---

## 第二十四部分 AR017 自适应发散容量重捕获 + P1-7 决定性重答（2026-10-13）

| 项 | 内容 | 状态 |
|---|---|---|
| 主题 | AR016 遗留收口（P1-7 第三档）：批量 graph 的发散容量固定于捕获时刻（`graph_capacity_ratio` 默认 0.5），而 AR016 工件实测 D_frac 仅 13-16%（stable_ratio 0.8438-0.8674）——**FFN 重算过配 3.0-3.8× 纯浪费**；反向固定低容量则每突发一次预算违例（eager 回退，graph 收益归零）。本 AR 落地**单图 + 观测驱动重捕获**：`AdaptiveCapacityPolicy` 状态机观测每步最大层发散量，持续过配→收缩重捕获、违例→立即增长重捕获，节流 + 上限退役保证稳定性，任意时刻至多双图、失败退化回旧 runner 永不劣于现状；并以三路径墙钟基准给 P1-7 **决定性重答** | ✅ 完成 |
| 选题证据链（三探针，本机实测） | ①AR016 工件 stable_ratio ∈[0.8438,0.8674] → D_frac 13-16% vs 固定 C=50% → FFN 重算浪费 3.2-3.8×；②容量因果探针：ratio 0.5→0.1875 在 H=256/L=8 降墙钟 8-12%（FFN[C] 仅占 ~25%、开销主导），H=1024/L=16 降 **41.7-42%**；③几何扫描：H=1024/L=16 下 graph-vs-baseline 从 1.36-1.41×（=AR016 inconclusive 区间）升到 2.32-2.43×（N=4/8, T=512）——机制与基准几何的双重依据 | ✅ 完成 |
| 生产改动 | ①新模块 `actfold/core/adaptive_capacity.py`（`AdaptiveCapacityConfig` frozen dataclass 五旋钮 + `AdaptiveAction` + `AdaptiveCapacityPolicy` 状态机：增长=违例即 `ceil(1.25·maxD)` 夹到 N·T、收缩=连续 patience 个无违例观测且 maxD≤β·C → `clamp(ceil(1.25·maxD),1,C-1)`、increment-then-check 节流（counter 初始化=interval 首次不节流）、`max_recaptures` 成功计数上限→退役、`events` 只记成功 `note_recaptured`）；②`cuda_graph.py`：`last_layer_divergences` 暂存——`validate_budgets` 的唯一 `tolist` 复用逐层发散计数，**零额外宿主同步**（FR1）；③`architecture_utils.py`：构造参数 `graph_adaptive_capacity`/`graph_adaptive_config`（isinstance 校验）+ 公开探针 `graph_adaptive_policy`（默认关=不建任何状态，UT-1606 零态契约）+ **双钩子**（增长=预算超限分支内、一次性告警后 return 前——违例即信号；收缩=发布后——换图永不先于发布，发布消费旧 runner 静态缓冲）+ `_recapture_batched_runner`（**先捕新、成功换、最后弃旧**；失败=一次性告警 + 只退役自适应路——`_graph_batched_capture_failed` 刻意不动（其语义是废弃整条批量 graph 路，会连健全旧 runner 一起弃）；prospective 防御行按 `ceil(new_ratio·N·T)` 实际容量判跳过——浮点恒等式 `ceil(ratio·n)==target` 仅 2 幂 n 精确，收缩 target=C-1 可能绕回 C，跳过免无谓捕获）；④`config_manager.py` 6 字段（graph_adaptive_capacity + 五旋钮镜像，边界校验独立于开关）+ `benchmark_runner.py` 接线 | ✅ 完成 |
| 新测试 | UT-1601~1608（`tests/test_adaptive_capacity.py`，1195 行）：UT-1601 策略纯逻辑表驱动（增长夹取/耐心窗/β 阻断/C≤1 noop/窗口裁剪/节流/退役/配置边界 ValueError/双实例确定性，CPU 无 CUDA 标记）；UT-1602 收缩端到端；UT-1603 增长端到端（C=2 违例→grow 事件 target=15 + 违例步语义不变 + 换图不重置预算告警标志）；UT-1604 上限退役（6 步负载交替，恰一次退役告警且非 budget 字样）；UT-1605 失败安全网（capture 注入 RuntimeError：恰一次 adaptive 告警、旧 runner `is` 存活、`_graph_batched_capture_failed` 保持 False、零重试、replay spy 递增）；UT-1606 默认关零行为（探针 None + 5 步同 runner 零告警）；UT-1607 观测接缝（暂存 None→刷新→随负载变化）；UT-1608 串行零干扰（批量双换图后串行 runner 独立捕获/重放）。**UT-1608 测试侧时序 bug 修复**（Red 期子代理推演漏算 increment-then-check 节流，shrink 后下一步的 grow 会被节流——插入低负载步让计数器重新累积，断言面不变；design 是 approved spec 不让步） | ✅ 完成 |
| mutation 自证 | ①-④ 均翻红后 SHA256 逐字节还原（architecture_utils 196E87…F3024 / adaptive_capacity 10A9CB…BCA15 四次一致）：①删双钩子→UT-1602 红（runner 未换）；②删增长分支→UT-1603 红（capacity 停在 2）；③删上限退役→UT-1604 红（`_retired` 未锁存）；④先弃旧后捕新→UT-1605 红（失败注入后 `graph_runner_batched` 为 None，`is` 断言炸） | ✅ 完成 |
| 墙钟基准 | `scripts/ar017_adaptive_bench.py`（`python -m scripts.ar017_adaptive_bench`，--quick 冒烟）：三路径（baseline=裸模型 N 前向 / graph_fixed=C 0.5 / graph_adaptive）× 几何 (256,8)/(1024,16) × N∈{4,8} × T=512（N·T∈{2048,4096} 全 2 幂——容量算术精确性前提），round-robin 交错，**warmup=5 覆盖适配期**（捕获 + 收缩耐心 3 + 节流 2，计时稳态反映已适配容量），**reps=30 高功效运行**（n=10 预运行在边界配置 (256,8,4) 判定翻转——两次同码运行 supported/inconclusive 各一，即欠功效证据；正式工件取 n=30 单次预承诺运行，判定照单全收），p50/std，adaptation 状态表（`adaptive_state.recapture_events`/`capacity_trajectory`/`observed_max_layer_divergence` p50-min-max，轨迹在计时窗口外采集零污染），H1-H4 声明与判定分离，meta 补 **GPU 驱动版本 556.18**（AR016 Minor-2 豁免收口）+ CUPTI 探测 null+注记（沿 AR016 字段）；**假设标签映射**：实现 H2（adaptive vs baseline 2σ 检验）对应 srs FR5 的 P1-7 论点；srs H2（自适应加速比 ≥ 固定加速比）与 H3（几何外推）无独立判定字段、由数据直接验证成立（1.55>1.35 / 1.68>1.52 / 2.30>1.39 / 2.46>1.44；深几何加速比 > 小几何），实现 H3=srs H4（一致性），实现 H4 为新增容量追踪判定；工件 `results/optimization/ar017_adaptive_bench.json` | ✅ 完成 |
| **实测墙钟表**（p50 ms，Quadro RTX 5000，驱动 556.18，seed 2026，完整数据见工件） | 见下 | ✅ 完成 |
| 门控 | req 两轮（**Major-1** 增长分支调用点与"超限即 return None"互斥→拆两处钩子：增长评估在超限分支内丢弃前/收缩评估在发布后；**Major-2** 观测窗口所有权→append 条件=replay 返回非 None、重捕获后清空、退役后冻结；**Major-3** 退役标志须独立、禁止置位 `_graph_batched_capture_failed`）+ 10 Minor 全修；design 两轮（**Major** 浮点恒等式 `ceil(ratio·n)==target` 仅 2 幂 N·T 精确（n=25/target=7 反例）→§4.4 改"按实际容量决策"防御行 + §6 钉 2 幂夹具）+ 11 Minor 全修 | ✅ 完成 |
| 质量门 | 全量 **846 passed（838+8）**/3 skipped 零回归（既有测试仅 UT-1608 插一步，断言面不变）、demo 三指标精确一致（85.5%/2.35e-03/93.75%）、mypy clean（66 文件）/pyflakes 零/100 列 | ✅ 完成 |

**实测墙钟表（每步 = N 候选完整验证，p50 ms，本机 WDDM 无锁频，n=30 round-robin 交错，reps=30 高功效运行）**：

| 配置 | baseline eager | graph 固定 C=0.5 | graph 自适应 | adaptive vs baseline | adaptive vs fixed |
|------|---------------|-----------------|--------------|---------------------|-------------------|
| H=256, L=8, N=4 | 11.037 | 8.150 | 7.120 | 1.55× | 1.14× |
| H=256, L=8, N=8 | 23.430 | 15.388 | 13.915 | 1.68× | 1.11× |
| H=1024, L=16, N=4 | 71.757 | 51.650 | 31.209 | 2.30× | 1.66× |
| H=1024, L=16, N=8 | 145.487 | 101.040 | 59.119 | 2.46× | 1.71× |

**假设判定（基于实测，不预设结论；2σ=合并 std 的 2 倍，沿用 AR016 `_verdict`）**：
- **H1 自适应 vs 固定容量：inconclusive（深几何 2/2 supported 1.66×/1.71×；小几何方向正 1.14×/1.11× 未过 2σ）**——机理一致：收益上界 = FFN 重算占比 × 容量缩减比，H=256 下 FFN[C] 仅占 ~25%（AR016 勘误根因）且 WDDM std 大，方向全正但统计显著性不足；
- **H2 自适应 vs baseline：整体 inconclusive（严格全配置聚合；per-config 3/4 supported）**——**决定性几何（H=1024/L=16，srs §1.1 靶场）2/2 decisively supported：2.30×/2.46×，delta 40.5/86.4ms 超 2σ 阈值（4.4/7.4ms）9.3/11.6×**；小几何 (256,8,8) supported 1.68×，(256,8,4) 方向正 1.55× 但未过 2σ（delta 跨三次独立运行稳定 3.7-3.9ms——效应真实，2σ 展布检验在该小形状受 WDDM 展布限制；baseline=4 次微小前向 launch 延迟敏感）。**P1-7 决定性重答**：AR016 的 inconclusive（1.37-1.77×、3/6 过 2σ）源于固定容量过配 + 小几何开销主导双重稀释；容量右移后深几何 2.30-2.46× decisively、全部 4 配置方向正；
- **H3 三路径 logits 一致：supported（4/4 argmax 精确一致 + 自适应 vs 固定 graph allclose 1e-4）**——重捕获不改变数值语义；
- **H4 容量追踪需求：supported（4/4）**——容量轨迹 [1024,1024,340,…]/[2048,2048,680,…]（observe 3 触发收缩后即稳定），最终容量 340（需求 p50=max=272）/680（544）/339（271），每次恰 1 次收缩重捕获，适配后零预算违例；最终容量 ≈16.6% N·T vs 需求 ≈13.3%——FFN 重算较固定 C=0.5 缩减 3.01-3.02×，与 D_frac 证据链定量吻合（需求样本 n=34=35 次调用减去换图步——换图后新 runner 尚未 validate，暂存 None 不采样，诚实缺口）。

**语义要点**：(1) **两层 noop 判定无缝衔接**——策略按 target 判（grow target≤C、shrink C≤1），调用侧按 prospective 判（`ceil(new_ratio·N·T)` 浮点绕回 ∈{target,target+1}），2 幂几何下防御行零触发但非 2 幂时防无谓捕获。(2) **钩子次序是正确性而非风格**：增长评估必须在超限分支内（违例即信号、丢弃回退前），收缩评估必须在发布后（发布读旧 runner 静态缓冲）——req 门控 Major-1 的互斥拆分。(3) **节流语义 increment-then-check**（counter 初始化=interval 首次不节流、`note_recaptured` 归零）是 UT-1601 钉死的契约——Red 期 UT-1608 曾因漏算它推演出错误时序，教训是**测试侧推演必须逐步跟踪策略全状态**，测试修而不动 approved design。(4) **失败退化粒度**：重捕获失败只退役自适应路（独立标志 + 独立一次性告警），绝不动 `_graph_batched_capture_failed`（会连健全旧 runner 一起废弃整条批量 graph 路）——失败后零重试、旧 runner 继续服役（UT-1605 replay spy 钉住）。(5) **事件只记成功**：`events` 在 `note_recaptured` 时 append（含 trigger/target/from），失败/节流零事件——工件表的 adaptation_events 因此是可审计的重捕获账本。**遗留**：真实 checkpoint 端到端墙钟（本基准为合成 LLaMA-layout 模型）；负载突变流的在线调参（grow_factor/β 恒定假设）；与变长/多 parent 批量（AR016 遗留）正交可叠加；~~宿主侧固定开销分解（重放前 prefill/预算读回/逐分支发布占比的定量拆分，承第二十三部分遗留——AR017 未触及该成本面）~~ **✅ AR018 收口——分解已做（六相探针：发布占自适应全步 60.7%/71.2%/29.7%/31.5% 为最大单项）+ 主项已压（发布压缩后 45.3%/50.9%/15.7%/17.6%）；prefill（0.31-1.55ms）与读回（0.06-0.13ms）实测合计 <5% 记录在案，见第二十五部分**；(256,8) 级小形状的 2σ 展布检验功效问题（效应 delta 稳定但 WDDM 展布大，若需该几何的严格判定需锁频主机或更高重复数）。

---

## 第二十五部分 AR018 发布路径宿主侧成本压缩：批量稳定度归约 + 快照发布（2026-10-14）

| 项 | 内容 | 状态 |
|---|---|---|
| 主题 | AR017 遗留收口（P1-7 第四档）：批量 graph 发布循环 `_publish_graph_children` 的 L×N 双循环每步付出 3·L·N 次 launch（逐行 `child_buf[l][r].clone()` + put 内拷贝 + `stable_mask.sum()` 归约）——探针实测发布占自适应全步 **60.7%/71.2%/29.7%/31.5%**（四配置，全步 p50 7.927/13.823/30.688/57.743ms），其中 profiler 归约占发布内 37-44%、`ActivationCache.put` 自带 `buffer[:seq_len].copy_` 拷贝使逐行 clone 成为纯冗余中间拷贝。本 AR 落地两项正交机制：**①批量稳定度归约**——`mask_buf` 是 [L,N,T] 单张量，单次 `sum(dim=2)` 产出 [L,N] 全部 (layer,row) 稳定计数，0-dim 视图经新入口 `record_precomputed` 入账（镜像 `record` 的 enabled 早退/`_raw`/`_history`/裁剪，`record` 原方法零改动）；**②快照发布**——单次整块 `child_buf.clone()` fresh 快照 + 行视图 put（put 自带拷贝，快照私有且发布后本侧零写——默认拷贝型 cache 与 by-reference cache 双安全） | ✅ 完成 |
| 选题证据链（探针 `ar018_hostcost_probe`，本机实测 n=30，AR017 同夹具同种子稳态容量） | ①发布占比 60.7-71.2%（小几何）/29.7-31.5%（深几何）为六相最大单项；②发布内 profiler 归约 37-44%（(256,8,8) 仅归约 ~3.7ms）；③读回 0.06-0.13ms、prefill 0.31-1.55ms 合计 <5%（其余项不值的定量记录）；④全步 p50 与 AR017 工件锚点吻合（7.927/13.823/30.688/57.743 vs 7.120/13.915/31.209/59.119ms）——探针在 AR017 后代码上测量的前提成立 | ✅ 完成 |
| 生产改动 | ①`stability_profiler.py`：新增 `record_precomputed`（branch/parent/layer/step/stable_sum/num_tokens/tau/metric；`ndim==0` 显式校验防误传 [L,N] 整表——纯宿主元数据检查零同步；不产 divergence_positions——debug 回退由调用侧走 `record`）；②`architecture_utils.py`：`_publish_graph_children` 重构——**三态分派**（profiler 关→零归约零记录；`debug_enabled`→逐行 `record` 回退（divergence_positions 唯一来源，正确性优先）；常态→1 批量归约+`record_precomputed`）+ 单次整块快照 + 行视图 put；`num_tokens=seq_len` 与现状 `stable_mask.numel()` 逐值相等；③**`cuda_graph.py` 零改动**；launch 账本 3·L·N → L·N+2（含 layer-0 embedding put 拷贝精确值 3·L·N+N → L·N+N+2；(256,8,4) 96→34、(1024,16,8) 384→130） | ✅ 完成 |
| 新测试 | UT-1701~1708（`tests/test_publish_cost.py`，1008 行；UT-1709 为 T002 mutation 阶段预留）：UT-1701 record vs record_precomputed 全字段等价（CPU 子块 240 条含 `_max_history_len` 裁剪 + CUDA 子块 [L,N,T] 批量归约 0-dim 视图；聚合面 mean_stable_ratio/all_profiles/get_mean_stable_ratio 全对拍 + ValueError 校验 + enabled=False 双入口零记录）；UT-1702 发布窗口 spy：bool-sum 恰 1、`.item()/.tolist()/.cpu()` 恰 0、enabled=False 时 bool-sum 恰 0；UT-1703 快照发布 vs 本地复刻旧逐行参考实现逐位 `torch.equal`（参考实现不调 record 防 GLOBAL 污染）；UT-1704 child_buf identity clone 恰 1 且窗口总 clone 恰 1；UT-1705 by-reference 防别名（RefCache + 直接调 `_publish_graph_children` 绕过 revalidate fetch；mutate child_buf 后条目逐位不变 + 二次重放新值正确）；UT-1706 UT-1502 完备性 + UT-1509 隔离本地重断言；UT-1707 debug 回退（逐行 bool-sum=L·N + positions）vs 常态（bool-sum=1 + positions 全 None）；UT-1708 串行 graph + 批量 eager 零干扰 + `record` 签名钉住。spy 窗口=单次发布调用（monkeypatch `torch.Tensor.sum/clone/item/tolist/cpu`，sum 过滤 bool dtype、clone 过滤 child_buf identity），GLOBAL 单例逐用例 save/restore | ✅ 完成 |
| mutation 自证 | ①-⑤ 均翻红后 SHA256 逐字节还原（architecture_utils 0876A68…04CA1 / stability_profiler 0FCCA1F…35B87，五次还原比对一致）：①删批量归约→UT-1702 红；②删整块快照回退逐行 clone→UT-1704 红；③快照别名 child_buf→UT-1705 红；④删 debug 回退→UT-1707 红；⑤`record_precomputed` stable_sum 跨调用 scratch 复用+覆写→UT-1701 红 | ✅ 完成 |
| 墙钟基准 | `scripts/ar018_publish_bench.py`（`python -m scripts.ar018_publish_bench`，--quick 冒烟）：三路径（baseline/graph_fixed C=0.5/graph_adaptive）× AR017 同配置矩阵 (256,8)/(1024,16) × N∈{4,8} × T=512，round-robin 交错，warmup=5，**reps=30 预承诺单次运行**；六相相位分解随行（探针方法固化，计时窗口外私有 API 驱动）；**跨工件锚点**：运行时加载 AR017 工件（可用性/AR017 run_date/对照性质声明入 meta；缺文件判 no_anchor 不阻断）；H1=AR018 adaptive p50 vs AR017（`_verdict` 两样本 2σ）、H2=加速比不回退（≥ 边界即 supported）、H3=发布占比 < 探针参照常量（provenance 注记）、H4=三路径一致 + profiler 与逐行重算逐值对拍；峰值显存 `max_memory_allocated` 观测字段（快照瞬态量化，承 srs §6 风险表承诺）；CUPTI 探测 null+注记（#39）；工件 `results/optimization/ar018_publish_bench.json` | ✅ 完成 |
| 门控 | req 两轮（**Major-1** mutation ③ 红测映射错误→拆 ③快照非 fresh→UT-1705 + ⑤sums 跨步复用覆写→UT-1701；**Major-2** revalidate_host 占比 <0.3% 与探针工件矛盾→0.3-1.4%）+ 8 Minor 全修 + 复核带 2 新 Minor 当场修入；design 两轮（**Major-1** `phase_sum ≥ 全步` 方向性断言被探针自家数据推翻（4 配置中 3 个 phase_sum < 全步——相位间宿主遗漏 > sync 串行化）→ 全文改中性双向偏差注记 + tasks 补 erratum）+ 9 Minor 全修 + 复核带 2 新 Minor（design §2 docstring 挤行/tasks 勘误 13.676→13.997）+ 2 Nit 当场修入 | ✅ 完成 |
| 质量门 | 全量 **855 passed（846+9）**/3 skipped 零回归（既有测试零改动）、demo 三指标精确一致（85.5%/2.35e-03/93.75%）、mypy clean（66 文件）/pyflakes 零/100 列 | ✅ 完成 |

**实测墙钟表 + 跨工件对照（每步 = N 候选完整验证，p50 ms，Quadro RTX 5000，驱动 556.18，seed 2026，n=30 round-robin 交错，完整数据见工件）**：

| 配置 | baseline eager | graph 固定 C=0.5 | graph 自适应（AR018） | 自适应（AR017 锚点） | adaptive vs baseline | AR017 加速比 | H1 | H2 |
|------|---------------|-----------------|----------------------|---------------------|---------------------|--------------|----|----|
| H=256, L=8, N=4 | 11.315 | 5.664 | 4.708 | 7.120 | **2.40×** | 1.55× | inconclusive | supported |
| H=256, L=8, N=8 | 22.197 | 9.910 | 8.300 | 13.915 | **2.67×** | 1.68× | supported | supported |
| H=1024, L=16, N=4 | 70.026 | 46.425 | 26.318 | 31.209 | **2.66×** | 2.30× | supported | supported |
| H=1024, L=16, N=8 | 143.186 | 89.444 | 48.594 | 59.119 | **2.95×** | 2.46× | supported | supported |

**相位分解表（graph_adaptive 稳态每步，p50 ms，n=30，探针方法，计时窗口外）**：

| 配置 | revalidate_host | prefill | replay | readback | publish | logits_clone | 相位和 | 全步 p50 | 发布占比 | 探针占比 |
|------|----------------|---------|--------|----------|---------|--------------|--------|---------|---------|---------|
| H=256, L=8, N=4 | 0.110 | 0.318 | 1.833 | 0.052 | **2.132** | 0.081 | 4.525 | 4.708 | 45.3% | 60.7% |
| H=256, L=8, N=8 | 0.111 | 0.314 | 3.237 | 0.057 | **4.225** | 0.125 | 8.070 | 8.300 | 50.9% | 71.2% |
| H=1024, L=16, N=4 | 0.184 | 0.826 | 20.664 | 0.101 | **4.144** | 0.083 | 26.003 | 26.318 | 15.7% | 29.7% |
| H=1024, L=16, N=8 | 0.195 | 1.556 | 36.190 | 0.128 | **8.545** | 0.132 | 46.746 | 48.594 | 17.6% | 31.5% |

**假设判定（基于实测，不预设结论）**：
- **H1 AR018 自适应 p50 < AR017 锚点：整体 inconclusive（3/4 supported）**——(256,8,8)/深几何 2 配置全过 2σ（delta 5.6/4.9/10.5ms）；(256,8,4) 方向正（4.708 vs 7.120ms）但未过 2σ——与 AR017 同配置欠功效先例一致（小形状 WDDM 展布限制，该配置在 AR017 基准中同样 inconclusive）；
- **H2 加速比不回退：supported（4/4）**——2.40/2.67/2.66/2.95× vs 锚点 1.55/1.68/2.30/2.46×，四配置全部抬升；**小几何从 1.55/1.68× 抬到 2.40/2.67×**（srs P1-7 第四档"小几何方向抬升"验收达成）；发布压缩同时受益 fixed 路径（8.150→5.664ms @(256,8,4)——两 graph 路径共用发布循环）；
- **H3 发布占比 < 探针参照：supported（4/4）**——45.3%/50.9%/15.7%/17.6% vs 60.7%/71.2%/29.7%/31.5%（机制直接验证）；发布绝对值 2.132/4.225/4.144/8.545ms（探针 4.81/9.84/9.12/18.17ms 口径的压缩后延续）；
- **H4 一致性 + profiler：supported（4/4）**——三路径 argmax 精确一致 + 自适应 vs 固定 allclose 1e-4 + 每分支 stable_ratio 与独立逐行 `mask_buf` 重算逐值相等（≤1e-6，计时窗口外采样）。

**语义要点**：(1) **三态分派是发布循环的控制骨架**——profiler 关→零归约零记录（与现状 `record` 早退等价）；`debug_enabled`→逐行 `record` 回退（divergence_positions 唯一来源，正确性优先于 launch 数）；常态→1 批量归约 + `record_precomputed`。三态由 `profiler.enabled and profiler.debug_enabled` 两个布尔决定，无新开关无新语义。(2) **快照 fresh 契约是 by-reference 安全的依据**——每步单次整块 `child_buf.clone()` 产出私有快照，发布后本侧零写、函数返回后零保留；默认 cache 的 put 拷贝与 by-reference cache 的视图存储都安全（后者持有私有快照行视图，不与静态缓冲/后续重放共享存储——UT-1705 RefCache mutate 钉住；若改回 child_buf 视图直传即红）。(3) **0-dim 消费面等价免改一切下游**——`record` 现路径 `stable_sum = stable_mask.sum()` 本就是 0-dim；`_ratios` 的 `torch.stack`/`.float()`、`get_profile` 的 `.item()` 对 0-dim 视图语义等价；bool→int64 sum 是精确整数累加（结合律），整块 dim=2 归约与逐行 sum 逐位相等——UT-1701 的 bit-exact 有硬保证。(4) **launch 账本 3·L·N → L·N+2**：clone L·N→1、sum L·N→1、put 拷贝 L·N 不变（put 的逐分支拷贝是 cache 协议事实）；`ndim==0` 校验是纯宿主元数据（纳秒级）相对消除的 launch（~10-20µs/个）可忽略。(5) **相位和与全步的双向偏差**——相位分解的 sync 边界串行化引入正向偏差、相位间宿主工作遗漏引入负向偏差，探针实测 4 配置中 3 个相位和 < 全步（(256,8,8) 13.997>13.823 为唯一正向）——两值同入工件、不预设方向（design 门控 Major-1 勘误钉住）；H3 判定用占比而非绝对值。(6) **峰值显存**——快照瞬态 ≈1×child_buf（(1024,16,8) ≈268MB），caching allocator 固定几何即时复用；工件峰值观测 332.8/528.7/2472.5/3521.7MB（16GB 上限远未触及）。**遗留**：真实 checkpoint 端到端墙钟（本基准为合成 LLaMA-layout 模型）；小几何发布仍占 45-51%——剩余主项是 L·N 次 put 拷贝带宽与 Python 循环开销（cache 批量 put API 已在 srs §1.3 排除：cache 协议 duck-typed、逐分支 put 语义是验收面）；(256,8,4) 的 2σ 功效问题沿 AR017 遗留；变长/多 parent 批量（AR016 遗留）正交可叠加。

---

## 第二十六部分 AR019 真实 checkpoint 端到端墙钟：position_embeddings 线程化 + Fast-dLLM-v2-1.5B 三路径基准（P1-7 第五档收官）（2026-10-14）

| 项 | 内容 | 状态 |
|---|---|---|
| 主题 | P1-7 最终档（AR016/17/18 三代遗留项）：综合框架此前只在合成 LLaMA-layout 模型上验证——真实 remote-code 模型（Fast-dLLM-v2-1.5B）的 Qwen 层**必须收 `position_embeddings`**（rotary），`ManualFoldedForward` 不透传则 parent prefill 即 `TypeError`（探针 P1 决定性证据）。本 AR：①生产代码 `position_embeddings` 线程化**四个构建面**（串行 eager / 批量 eager / graph 捕获静态体 / AR017 自适应重捕获），镜像 `attention_mask` 全套契约（入口 ValueError 校验 + by-reference 烘焙 + 重放期 `is` 对象恒一 + 钩子/重捕获透传）；②Fast-dLLM-v2-1.5B 真实块验证工作负载三路径墙钟基准，给 P1-7 在真实 checkpoint 上的最终论点 | ✅ 完成 |
| 选题证据链（探针 P1-P7，本机实测） | P1 缺口确证（remote attention `cos, sin = position_embeddings` 解包即炸）；P2 注入 pe+块掩码后 parent prefill 逐位一致（maxdiff=0）；P3 **真实块验证工作负载**（parent=224 prompt+32×MASK(151665)，children=top-10 采样填充）批量折叠 vs baseline **逐位一致**（maxdiff=0, argmax=1.0），门控完美二分（0.6667=理论值）；P4 块因果语义（`block_q >= block_kv`，32-token 块内双向、跨块因果）⇒ prompt 块位级稳定；P5 负面证据（少量 token 翻转负载 parity 仅 92.9%——块内双向传播，排除）；P6 环境（HF requests 栈与代理不兼容→`curl.exe -x` 逐文件下载 3.09GB 权重；venv `--system-site-packages` + `transformers==4.53.1` 实测可行）；P7 Triton bf16 merge/gate kernel 本机无 C 编译器→AR014 键记录+PyTorch 回退（capture warmup 侧流先回退再捕获，不进捕获体） | ✅ 完成 |
| 生产改动 | ①`architecture_utils.py`：`forward` 末位新参 `position_embeddings=None`（向后兼容）+ 入口校验（二元组/张量/seq 维=`tokens.shape[-1]`/cos-sin head_dim 一致，纯宿主元数据 `ValueError`；head_dim 与层内期望失配归 capture-failed 可见档——MFF 无从得知期望值，宽捕获转静默违反"不吞其他错误"）+ eager 循环**仅非 None 时附加 kwarg**（None 路径逐字节不变）+ `_graph_pe`/`_graph_batched_pe` 钉对象 + 重放恒一检查（`is` 失配静默回退 eager）+ `_observe_adaptive_capacity`/`_recapture_batched_runner` 透传（两钩子都只在恒一已过的重放路径触发⇒重捕获必收钉定对象）；②`cuda_graph.py`：`FoldedGraphRunner.__init__` 新参 `position_embeddings_static`（by-reference 烘焙，镜像 mask 不 clone）+ `_static_forward` 仅非 None 时附加 kwarg（捕获体零新增 host sync）；③`folded_transformer.py`：**零代码改动**（层 `**kwargs` 通道 → `_recompute_all` accepted/varkw 过滤天然承载，docstring 补一句） | ✅ 完成 |
| 新测试 | UT-1901~1908（`tests/test_position_embeddings.py`，955 行，9 测试函数）：1901/1902 串行/批量 eager 透传（层收到供给 pe `is` 同对象 + 输出=手工焊 pe 对照）；1903 graph 捕获烘焙（重放 3 次稳定 + 静态体以 `is` 同对象消费）；1904 自适应重捕获烘焙（触发 shrink 后新 runner 收正确 pe）；1905 重放恒一（pe_a 捕获/pe_b 重放→静默回退 eager 无告警 + CPU eager 伴随面）；1906 None 默认三子断言（a 不传=现状绿钉 / b 显式 None / c 非 None+`_PlainLayer` 过滤——Red 价值归属分开标注）；1907 结构校验 ValueError（五类畸形）；1908 捕获期 CUDA 拒绝（CPU pe 静默拒绝捕获） | ✅ 完成 |
| mutation 自证 | ①-④ 全翻红后 SHA256 逐字节还原（architecture_utils F46B776B…A74F / cuda_graph 3239A050…4810，两次还原比对一致）：①删 eager 循环附加→1901/1902 红；②删重放恒一检查→1905 红（pe_b 走图输出烘焙值语义错）；③重捕获不透传→1904 红；④`_static_forward` 不附加→1903 红；1907/1908 无独立 mutation（守卫缺失=Red 态重现，等价验证） | ✅ 完成 |
| 墙钟基准 | `scripts/ar019_real_e2e_bench.py`（816 行，`python -m scripts.ar019_real_e2e_bench --model-path …`，`--quick` 冒烟）：三路径（baseline_eager=raw 模型 N 次前向 / graph_fixed C=0.5 / graph_adaptive）× 真实块验证工作负载 × 配置 (256,4)/(256,8)/(512,4)，round-robin 交错，warmup=8（覆盖捕获+shrink 耐心 3+节流 2），**reps=30 预承诺单次运行**；侧车（计时窗外）：parity（argmax+maxdiff+allclose(1e-2,1e-1)，逐行 float cast 控显存瞬态）、stable ratio、容量轨迹、预算违例、**pe 恒一回退计数**（脚本层私有探针 `mff._graph_batched_pe is pe`，预期 0）、峰值显存、Triton `_TRITON_FAILED_KEYS` 快照；H1=adaptive vs baseline 2σ、H2=parity、H3=块二分（±0.02）、H4=容量右移（`final ≥ maxD 且 < fixed`）；meta 含权重 sha256/venv 标记/transformers 版本/CUPTI null+注记；`--model-path` 缺失→`FileNotFoundError`（#10 无合成回退）；工件 `results/optimization/ar019_real_e2e_bench.json` | ✅ 完成 |
| 门控 | req 两轮（**Major-1** graph 重放期 pe 对象恒一契约缺失→对象恒一 `is` 检查+静默回退档+`arange(T)` 常量位置方案前提；**Major-2** FR1 遗漏自适应重捕获面→补第四构建面+UT）+ 2 Minor + 2 Nit 全修；design 两轮（**Major-1** srs"维度不匹配→ValueError"子句无落点→入口 seq 维校验+head_dim 归属论证+UT-1907 补用例；**Major-3** UT-1906 Red 预期错误（现状 forward 无参传即 TypeError 不可能绿）→拆 a/b/c 三子断言）+ 复核 N1-N3（流程图同步/head_dim 内部一致检查/srs 对齐句）当场修入 | ✅ 完成 |
| 质量门 | 全量 **864 passed（855+9）**/3 skipped 零回归、demo 三指标精确一致（85.5%/2.35e-03/93.75%）、mypy clean（66 文件）/pyflakes 零/100 列 | ✅ 完成 |

**实测墙钟表（Fast-dLLM-v2-1.5B bf16，H=1536/L=28/V=151936，真实块验证工作负载，每步 = N 候选完整验证，p50 ms，Quadro RTX 5000，AR019 venv transformers 4.53.1，seed 2026，n=30 round-robin 交错）**：

| 配置 | baseline eager | graph 固定 C=0.5 | graph 自适应 | adaptive vs baseline | fixed vs baseline | 容量轨迹（固定→右移） | 峰值 VRAM | H1（2σ） |
|------|---------------|-----------------|--------------|---------------------|-------------------|----------------------|-----------|----------|
| T=256, N=4 | 796.148 | 450.645 | 326.890 | **2.44×** | 1.77× | 512→160（需求 128） | 5624 MB | inconclusive |
| T=256, N=8 | 1393.528 | 815.931 | 532.513 | **2.62×** | 1.71× | 1024→320（需求 256） | 7640 MB | supported |
| T=512, N=4 | 1349.446 | 868.789 | 520.735 | **2.59×** | 1.55× | 1024→160（需求 128） | 8949 MB | supported |

**假设判定（基于实测，不预设结论）**：
- **H1 自适应显著快于 baseline（P1-7 真实 checkpoint 最终论点）：整体 inconclusive（2/3 supported）**——三配置 p50 加速比全部 2.44/2.62/2.59×，方向压倒性；(256,4) 为**首个配置**，WDDM 热身窗方差（baseline std 497ms ≈ p50 的 62%）淹没 2σ 判定，后两配置 std 5-40ms 干净且 supported——与 AR017/18 同类"首配置欠功效"先例一致，按预承诺单次运行纪律如实记录；
- **H2 三路径一致：supported（3/3）**——三配置两 graph 路径 vs baseline **maxdiff=0.0、argmax 一致率 1.0**（逐位一致，强于 allclose(1e-2,1e-1) 档）；
- **H3 门控块二分：supported（3/3）**——每 (layer, branch) stable ratio **精确等于** K_PROMPT/T（0.875/0.875/0.9375，n=112/224/112 全部相等，零偏差——块因果语义完美二分）；
- **H4 容量右移：supported（3/3）**——捕获形状 [N,T] ✓、轨迹第 3 步收敛后零新事件、稳态违例 0、末容量 160/320/160 ≥ 需求 128/256/128 且 < 固定 512/1024/1024、各 1 次重捕获。

**语义要点**：(1) **恒一契约是 pe 线程化的安全骨架**——mask 先例全镜像：by-reference 烘焙（不 clone，内存归调用方持活）+ 重放期 `is` 恒一（换对象静默回退 eager，eager 侧用本次 pe 语义安全）+ 重捕获必收钉定对象（两钩子都在恒一已过的路径上）。(2) **入口校验档位三路分明**——seq 维失配=入口 `ValueError`（首捕获 seq 失配不可能到达 capture）；换对象=恒一静默档；head_dim 与层内期望失配=capture-failed 可见档（一次性告警+禁用；MFF 无从得知期望值，宽捕获转静默会吞其他错误）。(3) **None 零语义变化是强保证**——仅非 None 时附加 kwarg，既有调用方逐字节不变（UT-1906a 现状绿钉）。(4) **真实 checkpoint 图捕获首证**——bf16 1.5B 权重 + remote 28 层 + 块因果掩码 + rotary 全链路 CUDA graph 捕获/重放成功；Triton bf16 gate/merge 双 kernel 因本机无 C 编译器走 PyTorch 回退（AR014 键记录，warmup 侧流先回退再捕获），即便如此三配置仍 2.44-2.62×。(5) **pe 恒一回退计数 0**——脚本层 canary 全程 `is` 匹配，图路径无一次静默回退。**遗留**：LLaDA-8B/Dream-7B（16GB 显存不可行）；`FastDLLMAdapter` 接线（生成回路集成）；变长/多 parent 批量（AR016 遗留）；小几何 (256,4) 首配置 WDDM 方差（锁时钟需管理员权限）；cache 批量 put API（AR018 §1.3 延续）。

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
