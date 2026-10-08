# ActFold 深度实验报告（两阶段完整版）

**平台**：AutoDL
- 阶段一（诊断）：NVIDIA RTX 6000D（84 GB，Blackwell 级）
- 阶段二（优化）：NVIDIA RTX PRO 6000 Blackwell Server Edition（96 GB）

**作者**：https://github.com/ShaneLiu04
**日期**：2026-10-03
**代码版本**：ActFold 工作区（阶段一 9 类修复 + 阶段二 4 项优化）
**实验脚本**：
- 阶段一：`scripts/algo_experiments.py`、`scripts/overhead_bench.py`、`scripts/make_experiment_figures.py`
- 阶段二：`scripts/opt1_cache_bench.py`、`opt2_split_bench.py`、`opt2_shape_bench.py`、`opt2_long_seq.py`、`opt3_adaptive_bench.py`、`opt4_fused_bench.py`、`make_optimization_figures.py`

**配套文档**：`docs/OPTIMIZATION_REPORT.md`（优化阶段专项报告）

---

## 0. 摘要（TL;DR）

### 阶段一：诊断——算法正确，但实现零收益

1. **算法正确性成立**。自折叠（child == parent）稳定率恒为 1.0、输出 MSE 精确为 0；强制全发散（τ=1.0）时输出与基线**逐位一致**（MSE=0，top-1 一致率 1.0）。修复余弦裁剪后，τ=1.0 的边界行为严格正确。
2. **稳定率极高**。跨扩散步骤的逐层平均稳定率达 **0.98–0.99**；单 token 扰动下 Fast-dLLM 0.976、LLaDA 0.969、Dream 0.976（τ=0.95）。
3. **但原实现没有墙钟收益**：折叠前向延迟在 τ∈[0.5,0.99] 区间几乎恒定（约 80 ms，高于基线 16–18 ms）；"全稳定快速路径"反而比基线慢 **约 3 倍**；扩散采样折叠后慢 **5–8 倍**。
4. **根因定位**：单层开销分解显示 **缓存读取（cache get）0.65–0.70 ms 已超过原始层计算（0.53–0.61 ms）**；且实现语义是"只要有一个发散 token 就整层全量重算"，因此 FLOPs 缩减只在整层全稳定时兑换成时间。**原实现是 Python 开销受限（overhead-bound），不是算力受限**。
5. **FLOPs 成本模型与实测严重脱节**：模型预测 0.02 ms，实测 76 ms（差约 3800 倍）。
6. **模型相关性显著**：Dream 与 LLaDA 对激活复用最敏感（1 token 扰动即引入 15%/25% 相对 logit 误差，采样 token 匹配率 0.712/0.727），Fast-dLLM 最稳（采样匹配率 1.0）。

### 阶段二：优化——四项改进完成，收益区间清晰

7. **优化 #1（缓存向量化）成功兑现真实加速**：put 快 12–183x、get 快 30–420x；全稳定快速路径 **52.9 ms → 6.45 ms（8.2x）**，比基线全量重算快 **2.3–2.7x**（三个模型一致）。这是 ActFold 首次在真实模型上超越"不折叠"。
8. **优化 #2（层内拆分）收益与形状相关**：Attention 全量 + FFN 仅发散行，数值与原路径一致；**B×T ≥ 512 后逐层 1.1–1.93x**，小形状亏损，已用 `min_split_tokens=512` 自动门控；seq=512 端到端把折叠路径缩短 17–18%，但仍未反超基线（剩余每层固定开销）。
9. **优化 #3（自适应分位数门控）全面占优**：精确命中任意目标稳定率；在保真度-复用前沿上优于固定 τ（Dream k=1：rel-MSE 0.220→0.083 @ 稳定率 0.81）。
10. **优化 #4（融合 gather+select）用于大形状**：数值逐位一致；T=8192/H=4096 达 **3.43x**，T≥2048/H≥8192 有净收益，小形状因启动开销不划算。
11. 工程侧共修复 **10 类缺陷**（阶段一 9 类 + `ChunkedActivationCache` 合约缺陷），新增 4 个核心模块与 5 个测试文件，远程回归 **204 passed**。

---

## 1. 实验环境与方法

### 1.1 环境

| 项 | 阶段一 | 阶段二 |
|---|---|---|
| GPU | RTX 6000D，84 GB，600 W | RTX PRO 6000 Blackwell，96 GB，600 W |
| CPU / RAM / 磁盘 | 208 vCPU / 1 TB / 50 GB 数据盘 | 同左 |
| PyTorch | 2.8.0+cu128，Triton 3.4.0 | 同左 |
| transformers | **4.53.1**（venv） | 同左（`huggingface_hub==0.36.2`） |
| CUDA | 12.8 / 驱动 595 | 13.0 / 驱动 580 |

> `transformers==4.53.1` 是 Fast-dLLM v2 官方 pin；LLaDA/Dream remote code 在 transformers 5.x 下无法导入。阶段二实例的阿里云 PyPI 镜像返回 403，改用清华源安装。

### 1.2 模型

| 模型 | 层数 | hidden | 参数量 | 备注 |
|---|---|---|---|---|
| Fast-dLLM-v2-1.5B | 28 | 1536 | 1.5B | Qwen 系自定义架构，`model.layers` |
| LLaDA-8B-Instruct | 32 | 4096 | 8B | `model.transformer.blocks`，块返回 `(hidden, cache)` |
| Dream-7B-Instruct | 28 | 3584 | 7B | MaskGIT 系，`model.layers` |

### 1.3 方法

- **子分支构造**：在父序列上随机翻转 `n∈{0,1,8,…,128}` 个 token（copy-flip）。
- **指标**：LASP 逐层稳定率（稳定 token 占比）；相对 logit MSE（`MSE/var`）；top-1 一致率；平坦余弦；墙钟（CUDA Events，warmup≥2，reps≥8，全部 `no_grad`）。
- **受控 τ 扫描**：`SimilarityGate` 使用 `<` 严格阈值；余弦裁剪到 `[-1,1]` 保证 τ=1.0 严格全发散。
- **形状控制**：`FoldedModel(split_layers, split_min_tokens)` 与 `AdaptiveQuantileGate(target_stable_ratio)` 提供独立开关。

### 1.4 实验矩阵

**阶段一（诊断，18 个数据文件）**

| 实验 | 内容 | 产物 |
|---|---|---|
| `similarity` | 逐层×逐 token 父子余弦图谱（τ=1.0 无近似） | `results/experiments/<m>/similarity.npz` |
| `tau_sweep` | τ∈{0.5,…,1.0} 稳定率/保真度/延迟 | `results.json` |
| `invariants` | 自折叠、全发散、快速路径精确性 | `results.json` |
| `layer_ablation` | all / early-only / late-only / none | `results.json` |
| `cache_budget` | 每层缓存 {1,4,16,64,256,65536} | `results.json` |
| `dynamic_tau` | 固定 τ vs FoldingScheduler | `results.json` |
| `merge_bench` / `sampling` / `cost_model` | 融合核 / 跨步采样 / 成本模型 | `results.json` |
| `overhead` | 单层开销分解 | `overhead.json` |

**阶段二（优化，29 个数据文件 + 6 张图）**

| 优化 | 实验 | 产物 |
|---|---|---|
| #1 缓存向量化 | put/get 微基准（T=34…512）+ 端到端全稳定/部分稳定 A/B + 正确性 | `results/optimization/opt1/<m>/opt1_cache.json` |
| #2 层内拆分 | 三模型等价性与 τ/k 扫描；LLaDA 层 108 组 (B,T,发散比) 形状扫描；seq=512 端到端 | `opt2/<m>/opt2_split.json`、`opt2/shape/`、`opt2/longseq/<m>/` |
| #3 自适应门控 | 固定 τ vs 自适应目标 × 2 分支 × 3 模型 | `opt3/<m>/opt3_adaptive.json` |
| #4 融合 kernel | 12 组 (T,H) 形状扫描 + 逐位一致性 | `opt4/opt4_fused.json` |

---

## 2. 算法不变量（正确性）

| 不变量 | Fast-dLLM-v2-1.5B | LLaDA-8B | Dream-7B | 说明 |
|---|---|---|---|---|
| 自折叠稳定率 | 1.0000 | 1.0000 | 1.0000 | child == parent |
| 自折叠 MSE | **0.0** | **0.0** | **0.0** | 与父基线逐位一致 |
| 自折叠 top-1 | 1.0 | 1.0 | 1.0 | — |
| 全发散（τ=1.0）稳定率 | **0.0** | **0.0** | **0.0** | 余弦裁剪修复后严格无假稳定 |
| 全发散 MSE | **0.0** | **0.0** | **0.0** | 与基线逐位一致 |
| 全模型前向（ms） | 16.5 | 17.6 | 16.6 | batch=1, seq≈34–41 |
| 原实现自折叠快速路径（ms） | 52.1 | 51.2 | 53.7 | 见 §8 根因 |
| 优化后快速路径（ms）* | **6.45** | **6.37** | **6.81** | 见 §12，优化 #1 |

\* 阶段二硬件（RTX PRO 6000 Blackwell）；对比基线 15.4/17.2/16.6 ms，即 **2.4–2.7x 真实加速**。

> 修复前：bf16 余弦可达 1.00007，导致 τ=1.0 时仍有 3.7% token 被判"稳定"。`SimilarityGate` 现将余弦裁剪到 `[-1,1]`，边界语义严格。

图：`figures/experiments/fig_invariants.png`

---

## 3. 相似度结构

| 模型 | flip=0 均值 / p05 | flip=1 均值 / p05 | flip=8 均值 / p05 |
|---|---|---|---|
| Fast-dLLM-v2-1.5B | 0.9987 / 0.9961 | **0.9797** / 0.9584 | 0.7957 / 0.2494 |
| LLaDA-8B-Instruct | 0.9986 / 0.9936 | **0.9387** / 0.6029 | 0.8195 / 0.3945 |
| Dream-7B-Instruct | 0.9987 / 0.9961 | **0.9408** / 0.5939 | 0.6932 / 0.1654 |

结论：

- 即使逐位相同的张量，bf16 余弦也会偏离 1.0 约 1e-3 量级——**门控阈值在 0.999+ 区间被数值噪声支配**。这解释了 τ 扫描中 0.99→0.995 的断崖，也是阶段二改用"逐位稳定集"分析的原因。
- LLaDA/Dream 的隐藏状态对单 token 变化远比 Fast-dLLM 敏感（p05 = 0.60 vs 0.96），也解释了二者更大的复用误差与采样退化。

图：`fig_similarity_heatmap.png`（层×token 热力图）、`fig_similarity_hist.png`（分布）、`fig_stable_by_layer.png`（逐层曲线）

---

## 4. τ 阈值敏感性（固定门控）

**Fast-dLLM-v2-1.5B**（prompt 41 token，翻转 1 个）：

| τ | 稳定率 | top-1 一致 | 相对 MSE | 延迟 (ms) |
|---|---|---|---|---|
| 0.50 | 0.9782 | 0.9268 | 2.24e-2 | 78.4 |
| 0.80 | 0.9756 | 0.9268 | 2.20e-2 | 80.6 |
| 0.90 | 0.9756 | 0.9268 | 2.20e-2 | 80.4 |
| 0.95 | 0.9756 | 0.9268 | 2.20e-2 | 80.8 |
| 0.99 | 0.9756 | 0.9268 | 2.20e-2 | 80.5 |
| 0.995 | 0.6411 | 0.9756 | 1.17e-2 | 80.5 |
| 0.999 | 0.0 | 1.0000 | **0.0** | 61.6 |
| 1.0 | 0.0 | 1.0000 | **0.0** | 60.6 |

**LLaDA-8B-Instruct**（prompt 34 token，翻转 1 个）：

| τ | 稳定率 | top-1 一致 | 相对 MSE | 延迟 (ms) |
|---|---|---|---|---|
| 0.50 | 0.9991 | 0.9118 | 2.74e-1 | 52.0 |
| 0.80 | 0.9954 | 0.9118 | 2.74e-1 | 55.7 |
| 0.90 | 0.9706 | 0.9118 | 2.54e-1 | 82.8 |
| 0.95 | 0.9706 | 0.9118 | 2.54e-1 | 82.3 |
| 0.99 | 0.9688 | 0.9118 | 2.54e-1 | 83.4 |
| 0.995 | 0.5827 | 0.9706 | 1.95e-1 | 82.8 |
| 0.999 | 0.0 | 1.0000 | **0.0** | 64.1 |
| 1.0 | 0.0 | 1.0000 | **0.0** | 63.8 |

**Dream-7B-Instruct**（prompt 41 token，翻转 1 个）：

| τ | 稳定率 | top-1 一致 | 相对 MSE | 延迟 (ms) |
|---|---|---|---|---|
| 0.50–0.99 | 0.9756 | 0.9268 | 1.46e-1 | 79–81 |
| 0.995 | 0.5061 | 0.9512 | 9.05e-2 | 80.5 |
| 0.999 | 0.0 | 1.0000 | **0.0** | 60.4 |
| 1.0 | 0.0 | 1.0000 | **0.0** | 59.8 |

要点：

1. τ∈[0.5,0.99] 内稳定率是**平台**而不是斜坡：未变 token 相似度≈1.0，翻转 token 相似度≈0.9x——阈值在平台内移动不改变分组。三模型同形态（0.976 / 0.969 / 0.976）。
2. 只有把 τ 提到接近 1.0 才牺牲稳定率换取保真；τ=0.999 即退化为全量重算（此时反而快约 25%，整层全发散路径跳过缓存读取与合并）。
3. **延迟对 τ 完全不敏感**（80±2 ms），直接印证"一点发散即全层重算 + Python 开销恒定"。
4. 同一阈值含义因模型而异：platform 稳定率下相对误差 Fast 2.2%、Dream 14.6%、LLaDA 25.4%。**这直接催生了优化 #3 的自适应门控。**

图：`fig_tau_quality.png`

---

## 5. 层选择消融（τ=0.99）

| 折叠范围 | Fast: 稳定率 / rel-MSE / top-1 | LLaDA: 稳定率 / rel-MSE / top-1 | Dream: 稳定率 / rel-MSE / top-1 |
|---|---|---|---|
| all | 0.976 / 2.17e-2 / 0.927 | 0.971 / 2.50e-1 / 0.912 | 0.976 / 1.48e-1 / 0.927 |
| early-only | 0.976 / **8.13e-3** / 0.927 | 0.971 / 1.37e-1 / 0.971 | 0.976 / 1.27e-1 / 0.951 |
| late-only | 0.951 / 1.00e-2 / 0.976 | 0.753 / 5.71e-2 / 0.971 | 0.840 / **2.56e-2** / 0.976 |
| none | 0.0 / **0.0** / 1.0 | 0.0 / **0.0** / 1.0 | 0.0 / **0.0** / 1.0 |

- `none` 精确复现基线，再次验证路径正确性。
- Fast：折叠早期层误差最小（8.1e-3），折叠后期层更大（1.0e-2）。
- LLaDA/Dream：**恰好相反**——折叠后期层最安全（5.7e-2 / 2.6e-2）。Dream"只折叠后 1/3 层"把误差从 14.8% 降到 2.6%，同时保持 0.84 稳定率。
- **最优层集合因模型而异**；`FoldingScheduler.disabled_layers` 已支持按模型校准。

图：`fig_layer_ablation.png`

---

## 6. 缓存预算

| 每层上限 | Fast: 稳定率 / top-1 | LLaDA: 稳定率 / top-1 | Dream: 稳定率 / top-1 |
|---|---|---|---|
| 1 | 0.024 / 1.000 | 0.029 / 0.971 | 0.024 / 1.000 |
| 4 | 0.098 / 1.000 | 0.088 / 1.000 | 0.098 / 0.976 |
| 16 | 0.390 / 0.976 | 0.440 / 0.912 | 0.390 / 0.976 |
| 64 | 0.976 / 0.927 | 0.969 / 0.912 | 0.976 / 0.927 |
| 256 / 65536 | 0.976 / 0.927 | 0.969 / 0.912 | 0.976 / 0.927 |

- 稳定率随预算近似线性上升，直到预算 ≥ 序列长度后饱和；**缓存预算至少要覆盖序列长度**。
- 原实现在 token 0 被 LRU 驱逐后整体放弃复用；本轮修复为"从任意可用条目取样、缺失位置视为发散"，才观察到渐进曲线。
- 预算=4 时 top-1 反而完美（只复用最尾部少数 token）——**小规模保守复用可能比大范围近似更安全**。

图：`fig_cache_budget.png`

---

## 7. 动态阈值调度

`FoldingScheduler(base_tau=0.95)` 给出的逐层 τ 从 0.94 线性降到 0.882（clamp 下界 0.80）。在测试 prompt 上固定 τ=0.95 与调度器结果几乎相同（稳定率 0.9756 vs 0.9756；rel-MSE 2.198e-2 vs 2.166e-2）。原因是稳定率平台效应淹没了 ±0.03 的阈值偏置。**要让逐层调度产生实质影响，需要把 τ 推到 0.995+ 敏感区，或改用优化 #3 的逐层目标控制。**

---

## 8. 性能根因分析（阶段一核心发现）

### 8.1 单层开销分解（batch=1）

| 项 (ms) | Fast-dLLM (seq41,H1536) | LLaDA (seq34,H4096) | Dream (seq41,H3584) |
|---|---|---|---|
| 原始层重算 | 0.538 | 0.533 | 0.606 |
| 折叠层·全稳定（快速路径） | **1.889** | **1.544** | **1.825** |
| 折叠层·全发散 | 2.146 | 1.808 | 2.081 |
| 相似度门控 | 0.067 | 0.065 | 0.066 |
| **缓存读取 cache get** | **0.699** | **0.652** | **0.697** |
| 缓存写入 cache put | 0.228 | 0.190 | 0.230 |
| 融合·Triton | 0.026 | 0.038 | 0.024 |
| 融合·PyTorch | 0.007 | 0.011 | 0.007 |

**关键观察：缓存读取单层 0.65–0.70 ms ≥ 原始层计算 0.53–0.61 ms。** 即使全部 token 稳定、完全跳过 Transformer 计算，仅"查表拼回激活"就比重算更贵。快速路径因此实测慢约 3 倍（52.1/51.2/53.7 ms vs 16.5/17.6/16.6 ms）。三模型模式完全一致——**这是实现层（缓存 API）问题而非算法问题**，直接催生优化 #1。

T=512/H=4096 的补充微基准（阶段二）：

| 组件 | 时间 |
|---|---|
| gate（余弦） | 0.064 ms |
| cache get（全命中） | 0.020 ms |
| **cache get（部分命中，原实现）** | **0.174 ms** |
| cache put | 0.017 ms |
| Triton merge / torch merge | 0.026 / 0.008 ms |
| `inspect.signature`（每层） | 0.010 ms |
| 原始层（H=4096, T=512） | 1.122 ms |

### 8.2 成本模型 vs 实测

实测硬件（阶段一）：**FP16 矩阵乘约 136–138 TFLOPS，带宽约 1280 GB/s**。`ComputeBandwidthCostModel` 预测稳定率 0.97 时约 **0.02 ms**；实测折叠前向 **76–83 ms**（差约 3800 倍）。预测曲线随稳定率下降，实测曲线在 τ≤0.99 完全水平。

差距来源：
- 模型假设"仅发散 token 参与注意力+FFN"，而实现必须用**完整子序列**重算整层（自注意力上下文要求）；
- 模型忽略 Python/缓存/门控/合并开销；
- 文档中的 "reduction ≈ R" 是**理论 FLOPs 上界**，不是原实现的墙钟收益。

图：`fig_overhead_breakdown.png`、`fig_latency_vs_prediction.png`

---

## 9. 扩散采样中的跨步折叠

**修复前的严重缺口**：三个原生采样器在每一步都传 `parent_branch_id=None`，即"跨时间步折叠"从未真正发生（只写缓存、零复用）。本轮实现 `_next_folding_branch()` 链式父子关系。

| 模型 | token 匹配率 | 折叠步数 | 平均逐步稳定率 | 基线 (ms) | 折叠 (ms) | 慢倍数 |
|---|---|---|---|---|---|---|
| Fast-dLLM-v2-1.5B | **1.000** | 31 | 0.980 | 471 | 3791 | 8.0x |
| LLaDA-8B-Instruct | **0.727** | 31 | 0.989 | 726 | 3730 | 5.1x |
| Dream-7B-Instruct | **0.712** | 31 | 0.987 | 768 | 3867 | 5.0x |

- Fast-dLLM：31 步折叠生成与基线**逐 token 完全一致**，说明近似在块式掩码解码中可忽略。
- LLaDA/Dream：稳定率同样 0.987–0.989，但 token 匹配率仅 0.71–0.73，生成文本出现退化（重复词）。**稳定率 ≠ 可用性**：早期步骤的微小 logit 扰动被贪婪解码沿步骤放大。
- 结论：需要以"下游 token 一致率/任务指标"为约束选择 τ 与折叠范围。

图：`fig_sampling.png`

---

## 10. 融合核（阶段一基线）

| 形状 | Triton (ms) | PyTorch (ms) | 最大误差 |
|---|---|---|---|
| Fast: [1,41,1536] | 0.0242 | **0.0074** | 0.0 |
| LLaDA: [1,34,4096] | 0.0377 | **0.0115** | 0.0 |
| Dream: [1,41,3584] | 0.0237 | **0.0075** | 0.0 |

小形状下 Triton 启动开销主导，慢 3.3 倍；数值完全一致。该结论直接指导了优化 #4 的"大形状才启用"策略。同时修复了 Triton 3.4 拒绝 `typing.Any` 注解导致的编译崩溃，并加入编译失败自动回退。

---

## 11. 阶段一修复清单（9 类，全部有回归测试）

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 1 | Triton 核在 torch 2.8/Triton 3.4 直接崩溃 | kernel 参数使用 `typing.Any` 注解 | 去掉指针注解 + 运行时编译失败回退；更新 AGENTS #9 |
| 2 | transformers 5.x 下 `load_in_8bit=False` 透传给 remote code | v5 行为变化 | 仅在启用量化时传参 |
| 3 | 三个模型族封装不接受 `torch_dtype` 等 | 构造函数未与基类对齐 | 接受并转发全部加载参数 |
| 4 | LLaDA 前向/返回类型不匹配 | 模型直接返回 `logits` | 兼容 `logits`/`last_hidden_state`/`hidden_states` |
| 5 | `FoldedModel` 返回 HF `ModelOutput` 对象 | 未解包 | `_unwrap_output()` |
| 6 | LLaDA block 返回 `(hidden, cache)` 解包失败 | 折叠层只返回张量 | `_pack_output()` 保持元组元数 |
| 7 | LLaDA 层堆栈探测失败 | 路径不在默认列表 | 补充 `model.transformer.blocks` / `wte` |
| 8 | τ=1.0 时 3.7% 假稳定 | bf16 余弦 >1.0 | 余弦裁剪 `[-1,1]` |
| 9 | 采样器从不折叠 + mask token 解析失败 + Dream 崩溃 | `parent=None`；tokenizer 无 mask id；long 型 attention mask | 跨步父链、统一 mask 解析、Dream bool 化 |

---

## 12. 优化 #1：向量化激活缓存

### 12.1 设计

`actfold/core/vectorized_cache.py`：`VectorizedActivationCache` 用每分支连续缓冲区 `[capacity, batch, hidden]` 取代逐 token 字典：

- `put`：一次 slice/`index_copy_`（环形布局支持 `T > capacity` 的 LRU 语义）；
- `get`：全命中返回转置**视图（零拷贝）**；部分命中用"克隆 + 逐 batch `masked_fill_`"；
- 与 `ActivationCache` 完全同 API；接入 `make_activation_cache(use_vectorized=True)`、`ActFoldConfig.use_vectorized_cache`、`BenchmarkRunner`。

### 12.2 微基准（Fast-dLLM，H=1536）

| 序列长度 | put 加速（vs legacy） | get 加速（vs legacy） |
|---|---|---|
| 34 | 12x | 30x |
| 96 | 33x | 81x |
| 256 | 87x | 211x |
| 512 | **176x** | **429x** |

三模型趋势一致。T=512/H=4096 部分命中 `get` 从 0.174ms → **0.058ms**。

### 12.3 端到端（全稳定快速路径）

| 模型 | 基线前向 | legacy | chunked | **vectorized** | 相对基线 |
|---|---|---|---|---|---|
| Fast-dLLM-v2-1.5B | 15.4 ms | 52.1 | 8.5 | **6.45** | **2.4x 更快** |
| LLaDA-8B-Instruct | 17.2 ms | 51.2 | 8.6 | **6.37** | **2.7x 更快** |
| Dream-7B-Instruct | 16.6 ms | 53.0 | 8.8 | **6.81** | **2.4x 更快** |

部分稳定路径（1 个 token 发散）：约 80ms → 39–43ms。**这是 ActFold 首次在真实模型上实现墙钟加速。**

### 12.4 附带修复：ChunkedActivationCache

原 `get` 返回**只含选中位置**的 `[B,K,H]`（破坏调用方约定的 `[B,T,H]`），空缓存返回 `{}` 而非 `KeyError`。已修正为全长零填充 + `KeyError`，并新增与 legacy 的逐元素一致性测试。

图：`fig_opt1_cache.png`、`fig_opt_summary.png`

---

## 13. 优化 #2：层内拆分（Attention 全量 + FFN 仅发散）

### 13.1 设计

`actfold/core/split_layer.py`：`SplitFoldedTransformerLayer` 不修改模型代码，在 FFN 链两端安装**临时前/后钩子**：

- 前钩子（`post_attention_layernorm` / LLaDA `ff_norm`）把输入切到发散行；
- 后钩子（`mlp` / `ff_out`）把输出散射回全长；
- 注意力、残差、归一化仍由原层执行，发散位置数值与完整重算一致。

接入 `FoldedModel(split_layers=True, split_min_tokens=512)`、`ActFoldConfig.use_split_layers / split_min_tokens`、`BenchmarkRunner`。

### 13.2 正确性

split 与 no-split 的 **MSE、top-1 完全一致**；bf16 logits 最大绝对差异 0.16–0.28（仅 GEMM 分块的 ulp 级差异）。

### 13.3 形状扫描（LLaDA 层，H=4096；108 组）

| 形状 | 1% 发散 | 10% 发散 | 50% 发散 | 100% 发散 |
|---|---|---|---|---|
| B=1, T=64 | 0.8x | 0.8x | 0.7x | 0.8x |
| B=1, T=512 | 1.15x | 1.14x | 1.01x | 0.92x |
| B=1, T=2048 | **1.78x** | 1.69x | 1.24x | 0.98x |
| B=4, T=2048 | **1.93x** | 1.78x | 1.32x | 0.99x |

机理：切分/散射每层需要一次数据相关 `nonzero` 同步（~0.028ms）与定长开销；收益 ∝ 发散 token 数 × H²。**B×T ≥ 512 开始转正，≥2048 显著**；默认 `min_split_tokens=512` 自动避免小形状回退。

### 13.4 端到端（seq=512，向量化缓存 + split）

| 模型 | 基线 | no-split 折叠 | split 折叠 | split vs no-split |
|---|---|---|---|---|
| Fast-dLLM-v2-1.5B | 30.7 ms | 36.1–42.1 | 37.9–42.4 | ≈ |
| LLaDA-8B-Instruct | 35.4 ms | 49.8 | **40.9** | **−18%** |
| Dream-7B-Instruct | 29.8 ms | 42.6 | **35.3** | **−17%** |

batch=1、seq=512 时 split 把折叠路径缩短 17–18%，但仍未反超基线——剩余开销是每层门控+缓存+Python 包装（~0.1–0.2ms/层），只有全融合 kernel 才能进一步消除。

图：`fig_opt2_shape.png`、`fig_opt2_longseq.png`

---

## 14. 优化 #3：自适应分位数门控

`actfold/core/adaptive_gate.py`：`AdaptiveQuantileGate` 每次调用用 `topk` 精确选取 `target_stable_ratio` 比例的 token 作为稳定集，阈值由数据分位数决定并记录到 `last_tau`。

**效果**：

1. **精确可控**：目标 0.5/0.8/0.9/0.97/1.0 全部命中；固定 τ 是平台+断崖，无法细调。
2. **前沿占优**（k=1 翻转）：

| 模型 | 固定 τ=0.99（稳定率 → rel-MSE / top-1） | 自适应（稳定率 → rel-MSE / top-1） |
|---|---|---|
| Fast-dLLM | 0.976 → 6.07e-2 / 0.927 | 0.805 → **4.33e-2** / 0.927（target 0.8） |
| LLaDA-8B | 0.971 → 2.07e-1 / 0.882 | 0.794 → 1.81e-1 / **0.971**（target 0.8） |
| Dream-7B | 0.976 → 2.20e-1 / 0.902 | 0.805 → **8.35e-2** / **0.976**（target 0.8） |

3. **k=8 翻转**时固定 τ 的 top-1 崩到 0.46–0.68；自适应可把稳定率降到 0.5 换取 top-1 0.61–0.85，明显更可控。
4. 自适应 target=1.0 会触发全稳定快速路径（6.5ms），但对 k=8 的保真度损失巨大（top-1 0.51–0.78）——**"快"与"准"的权衡必须由任务决定**。

图：`fig_opt3_frontier.png`

---

## 15. 优化 #4：融合 Triton gather+select

`fused_ops.gather_select`：单 kernel 内按 token 解析"稳定→从父缓冲按行号取；发散→拷贝子张量"，消除 `index_select` 中间张量与 `torch.where` 二次往返；CPU/不支持时自动回退，数值与 PyTorch 路径**逐位一致（max err=0）**。

| 形状 | PyTorch | fused | 加速 |
|---|---|---|---|
| T=128, H=4096 | 0.0174 ms | 0.0271 ms | 0.64x |
| T=2048, H=8192 | 0.0413 ms | 0.0276 ms | **1.50x** |
| T=8192, H=4096 | 0.1716 ms | 0.0500 ms | **3.43x** |
| T=8192, H=8192 | 0.3724 ms | 0.2027 ms | 1.84x |

小形状受启动开销限制；**T≥2048 且 H≥8192、或 T≥8192** 时启用有净收益。默认不强制启用以避免回归。

图：`fig_opt4_fused.png`

---

## 16. 集成视角：优化前后与推荐配置

### 16.1 两阶段综合对比

| 视角 | 阶段一（原实现） | 阶段二（优化后） |
|---|---|---|
| 全稳定快速路径 | 52–54 ms（0.31x，比基线慢 3 倍） | **6.4–6.8 ms（2.4–2.7x，超越基线）** |
| 部分稳定（1 发散 token） | ~80 ms | 39–43 ms（仍不及基线） |
| 长序列 split 逐层 | —（无此功能） | 1.1–1.93x（B×T≥512） |
| 复用控制 | 固定 τ（平台+断崖，模型相关） | 自适应 top-k（精确、模型无关） |
| 大形状合并 | torch（0.17ms @T8192,H4096） | fused（0.050ms，3.43x） |
| 采样折叠 | 8.0x/5.1x/5.0x 慢，且 chain 缺失 | chain 修复 + 缓存加速；质量风险仍在 |
| 代码缺陷 | 10 类 | 已修复并有 204 项测试 |

### 16.2 按场景的推荐配置

| 场景 | 推荐配置 |
|---|---|
| 短序列（≤256 token-lane）、高稳定率 | `use_vectorized_cache=True`；触发全稳定快速路径时获得 **2.3–2.7x 真实加速** |
| 长序列 / 大 batch（B×T≥2048） | 加上 `use_split_layers=True`（`split_min_tokens=512`） |
| 保真度敏感 / 需要精确复用预算 | `AdaptiveQuantileGate(target_stable_ratio=…)`；必要时配合按层关闭（Dream 只折叠后期层） |
| 主干特殊形状（T≥8192） | 融合 gather+select |
| 扩散采样质量优先 | τ 提高到 0.99+ 或用目标 0.9–0.95 的自适应门控；Fast-dLLM 可安全使用跨步折叠 |

---

## 17. 更新后的路线图

| 优先级 | 事项 | 状态 |
|---|---|---|
| — | 缓存 API 向量化 | ✅ 完成（#1，2.3–2.7x） |
| — | 层内拆分（Attention 全量 + FFN 发散） | ✅ 完成（#2，大形状 1.1–1.9x） |
| — | 自适应门限 | ✅ 完成（#3，前沿占优） |
| — | 融合 kernel | ✅ 完成（#4，大形状 1.5–3.4x） |
| P0 | **全融合 slow-path kernel（gate+gather+merge 单 kernel）**：消除每层 `nonzero` 同步与多次内存往返，让部分稳定场景（1%–30% 发散）也反超基线 | 待做 |
| P1 | 把 split 的同步改为"固定预算 top-k"：用静态 k（非数据相关形状）避免 `nonzero`，扩大拆分收益区间 | 待做 |
| P1 | 起草模型驱动 τ 调度：以采样 token 一致率为反馈的闭环控制 | 待做 |
| P2 | batch>1/seq≥2048 的全链路验证与调优；任务级基准（GSM8K/HumanEval+）复测 | 待做 |
| P2 | 变长折叠、多祖先复用 | 原路线图，未开始 |

---

## 18. 产物与复现

### 18.1 产物清单

| 类别 | 路径 |
|---|---|
| 阶段一原始结果 | `results/experiments/{fastdllm,llada,dream}/{results.json, similarity.npz, overhead.json}` |
| 阶段二原始结果 | `results/optimization/{baseline,opt1,opt2,opt3,opt4}/**`（29 个文件） |
| 阶段一图表（10 张） | `figures/experiments/fig_*.png` |
| 阶段二图表（6 张） | `results/optimization/figures/fig_opt*.png` |
| 诊断脚本 | `scripts/algo_experiments.py`、`overhead_bench.py`、`make_experiment_figures.py` |
| 优化脚本 | `scripts/opt1_cache_bench.py`、`opt2_split_bench.py`、`opt2_shape_bench.py`、`opt2_long_seq.py`、`opt3_adaptive_bench.py`、`opt4_fused_bench.py`、`make_optimization_figures.py` |
| 优化代码 | `actfold/core/vectorized_cache.py`、`split_layer.py`、`adaptive_gate.py`、`fused_ops.gather_select`、`chunked_cache.py`（修复） |
| 测试 | `tests/test_vectorized_cache.py`、`test_split_layer.py`、`test_adaptive_gate.py`、`test_chunked_cache.py`、`test_fused_ops.py`（远程 204 passed） |
| 专项报告 | `docs/OPTIMIZATION_REPORT.md` |

### 18.2 复现命令

```bash
# 环境（RTX PRO 6000，system-site-packages venv）
python -m venv --system-site-packages /root/autodl-tmp/venv-tf4
/root/autodl-tmp/venv-tf4/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "transformers==4.53.1" "huggingface_hub==0.36.2" "tokenizers>=0.20,<0.22" einops

# 阶段一：诊断
python scripts/algo_experiments.py --model fastdllm --out results/experiments/fastdllm
python scripts/algo_experiments.py --model llada   --out results/experiments/llada
DREAM_MODEL_PATH=/root/autodl-tmp/models/dream \
    python scripts/algo_experiments.py --model dream --out results/experiments/dream
python scripts/overhead_bench.py --model <key> --out results/experiments/<key>
python scripts/make_experiment_figures.py --root results/experiments --out results/experiments/figures

# 阶段二：优化
python scripts/opt1_cache_bench.py --model fastdllm --out results/optimization/opt1/fastdllm
python scripts/opt2_split_bench.py --model llada --out results/optimization/opt2/llada
python scripts/opt2_shape_bench.py --out results/optimization/opt2/shape
python scripts/opt2_long_seq.py --model dream --out results/optimization/opt2/longseq/dream
python scripts/opt3_adaptive_bench.py --model dream --out results/optimization/opt3/dream
python scripts/opt4_fused_bench.py --out results/optimization/opt4
python scripts/make_optimization_figures.py --root results/optimization --out results/optimization/figures

# 回归测试
python -m pytest tests/ -q -m "not slow"    # 204 passed
```

---

## 19. 图表索引

### 阶段一（诊断，`figures/experiments/`）

| 文件 | 内容 |
|---|---|
| `fig_invariants.png` | 自折叠/全发散精确性 + 快速路径速度 |
| `fig_similarity_heatmap.png` | 层×token 父子余弦热力图（flip=1/8） |
| `fig_similarity_hist.png` | 相似度分布直方图 |
| `fig_stable_by_layer.png` | 逐层平均相似度与稳定率（τ∈{0.95,0.99,0.995}） |
| `fig_tau_quality.png` | τ 扫描：稳定率、top-1、相对 MSE |
| `fig_cache_budget.png` | 缓存预算 vs 稳定率/保真度 |
| `fig_layer_ablation.png` | all / early-only / late-only / none |
| `fig_overhead_breakdown.png` | 单层开销分解（cache get > 原层计算） |
| `fig_latency_vs_prediction.png` | FLOPs 成本模型预测 vs 实测（3800 倍差距） |
| `fig_sampling.png` | 跨步采样稳定率与折叠慢倍数 |

### 阶段二（优化，`results/optimization/figures/`）

| 文件 | 内容 |
|---|---|
| `fig_opt1_cache.png` | 缓存 put/get 加速曲线 + 全稳定端到端对比 |
| `fig_opt_summary.png` | 优化 #1 汇总：2.3–2.7x 超越基线 |
| `fig_opt2_shape.png` | 拆分逐层加速 vs (B,T,发散比) + 延迟曲线 |
| `fig_opt2_longseq.png` | seq=512 端到端：基线/no-split/split |
| `fig_opt3_frontier.png` | 固定 τ vs 自适应门控的保真度-复用前沿 |
| `fig_opt4_fused.png` | 融合 kernel vs PyTorch 的形状扫描与加速 |

---

> **免责声明**：以上为单卡、单 prompt、batch=1 的受控实验结果，用于机制解释与性能归因；发布级结论仍需多 prompt、多 seed、任务级指标（GSM8K/HumanEval+）以及更长序列的验证。FLOPs 缩减为理论上界，"墙钟加速"仅指本报告实测路径。
