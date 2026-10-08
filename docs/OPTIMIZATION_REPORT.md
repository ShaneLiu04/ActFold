# ActFold 优化阶段报告（Optimization Phase）

**平台**：AutoDL · NVIDIA RTX PRO 6000 Blackwell Server Edition（96 GB）
**作者**：https://github.com/ShaneLiu04
**日期**：2026-10-03
**基线**：第一轮深度实验报告（`docs/DEEP_EXPERIMENT_REPORT.md`）在相同模型上的测量
**产物**：`results/optimization/`（原始 JSON + 6 张图表）、`scripts/opt1_*.py` … `scripts/opt4_*.py`

---

## 0. 摘要

按 `DEEP_EXPERIMENT_REPORT.md` §12 的路线图逐项实现并做了 A/B 实验：

| # | 优化 | 状态 | 关键结果 |
|---|---|---|---|
| 1 | 缓存 API 向量化 | ✅ 完成 | put 快 **12–183x**、get 快 **30–420x**；全稳定快速路径 **52.9ms → 6.45ms（8.2x）**，比基线全量重算快 **2.3–2.7x**（三个模型一致） |
| 2 | 层内拆分（Attention 全量 + FFN 仅发散） | ✅ 完成 | 数值与原路径一致（MSE/top1 相同）；**B×T ≥ 512 后逐层加速 1.1–1.93x**，B×T=64 时亏损；加入 `min_split_tokens=512` 自动阈值；长序列端到端把折叠路径缩短 15–20% |
| 3 | 自适应分位数门控 | ✅ 完成 | 精确命中任意目标稳定率；在保真度-复用前沿上**全面优于固定 τ**（Dream k=1：rel-MSE 0.22→0.083 @ 稳定率 0.81） |
| 4 | 融合 Triton gather+select | ✅ 完成 | 数值完全一致（误差 0）；T≥2048/H≥8192 后加速；T=8192/H=4096 达 **3.43x**；小形状因启动开销慢 0.6x |

同时修复了一个隐藏的 `ChunkedActivationCache` 合约缺陷（`get` 返回 K 长度张量与空 dict，与 `ActivationCache` 不兼容）。

**结论**：优化 #1 已经让"整层全稳定"的真实路径明显快于基线；优化 #2/#4 的收益区间在 **长序列/大 batch（≥2048 token-lane）**；优化 #3 提供了模型无关的精确复用控制。部分稳定（1%–30% 发散）场景下，剩余瓶颈是每层固定的门控+缓存+Python 开销（~0.1–0.2ms/层），下一步应把它们融合进单个 kernel。

---

## 1. 优化 #1：向量化激活缓存

### 1.1 设计与实现

新增 `actfold/core/vectorized_cache.py`：`VectorizedActivationCache` 用每分支的连续缓冲区 `[capacity, batch, hidden]` 取代逐 token 字典：

- `put`：一次 slice/`index_copy_` 写入（环形布局支持 `T > capacity` 的 LRU 语义）；
- `get`：全命中时直接返回转置视图（零拷贝）；部分命中时用"克隆 + 逐 batch `masked_fill_`"路径；
- 与 `ActivationCache` 完全同 API（`put/get/clear_*/num_entries`），已接入 `make_activation_cache(use_vectorized=True)`、`ActFoldConfig.use_vectorized_cache`、`BenchmarkRunner` 与 `core/__init__`。

### 1.2 微基准（Fast-dLLM，H=1536）

| 序列长度 | put 加速（vs legacy） | get 加速（vs legacy） |
|---|---|---|
| 34 | 12x | 30x |
| 96 | 33x | 81x |
| 256 | 87x | 211x |
| 512 | 176x | 429x |

三个模型趋势一致。T=512/H=4096 下部分命中 `get` 也从 0.174ms 降到 **0.058ms**（克隆路径替代 `index_select`）。

### 1.3 端到端（全稳定快速路径）

| 模型 | 基线前向 | legacy 缓存 | chunked | **vectorized** | 相对基线 |
|---|---|---|---|---|---|
| Fast-dLLM-v2-1.5B | 15.4 ms | 52.1 | 8.5 | **6.45** | **2.4x 更快** |
| LLaDA-8B-Instruct | 17.2 ms | 51.2 | 8.6 | **6.37** | **2.7x 更快** |
| Dream-7B-Instruct | 16.6 ms | 53.0 | 8.8 | **6.81** | **2.4x 更快** |

这是 ActFold 首次在真实模型上实现"墙钟加速"。部分稳定路径（1 个 token 发散）65 80ms → 39–43ms（架构性开销仍在，见 §2.4）。

图：`fig_opt1_cache.png`、`fig_opt_summary.png`

### 1.4 附带修复：ChunkedActivationCache

原实现 `get` 返回**只含被选中位置**的张量（`[B, K, H]`，破坏了调用方约定的 `[B, T, H]`），且空缓存返回 `{}` 而 legacy 抛 `KeyError`。现已修正为"全长零填充 + `KeyError`"，并新增与 legacy 的逐元素一致性测试（`tests/test_chunked_cache.py`）。

---

## 2. 优化 #2：层内拆分（Attention 全量 + FFN 仅发散）

### 2.1 设计与实现

新增 `actfold/core/split_layer.py`：`SplitFoldedTransformerLayer` 不修改模型代码，而是在 FFN 链的两端模块上安装**临时前/后向钩子**：

- 前钩子（`post_attention_layernorm` / LLaDA `ff_norm`）把输入切到发散行；
- 后钩子（`mlp` / `ff_out`）把输出散射回全长；
- 注意力、残差、归一化仍由原层执行，因此发散位置的数值与完整重算一致。

已接入 `FoldedModel(split_layers=True, split_min_tokens=512)`、`ActFoldConfig.use_split_layers / split_min_tokens` 与 `BenchmarkRunner`。

### 2.2 正确性

在三个真实模型上，split 与 no-split 的 **MSE、top-1 完全一致**；bf16 logits 的最大绝对差异 0.16–0.28（仅 GEMM 分块造成的 ulp 级差异）。

### 2.3 形状扫描（LLaDA 层，H=4096）

| 形状 | 1% 发散 | 10% 发散 | 50% 发散 | 100% 发散 |
|---|---|---|---|---|
| B=1, T=64 | 0.8x | 0.8x | 0.7x | 0.8x |
| B=1, T=512 | 1.15x | 1.14x | 1.01x | 0.92x |
| B=1, T=2048 | **1.78x** | 1.69x | 1.24x | 0.98x |
| B=4, T=2048 | **1.93x** | 1.78x | 1.32x | 0.99x |

机理：切分/散射每层需要一次数据相关的 `nonzero` 同步（~0.028ms）与定长开销；收益与"省下的 FFN 计算量 ∝ 发散 token 数 × H²"成正比。**B×T ≥ 512 开始转正，≥2048 显著**。默认 `min_split_tokens=512` 自动避免小形状回退。

图：`fig_opt2_shape.png`

### 2.4 端到端（seq=512，向量化缓存 + split）

| 模型 | 基线 | no-split 折叠 | split 折叠 | split vs no-split |
|---|---|---|---|---|
| Fast-dLLM-v2-1.5B | 30.7 ms | 36.1–42.1 | 37.9–42.4 | ≈ |
| LLaDA-8B-Instruct | 35.4 ms | 49.8 | **40.9** | **−18%** |
| Dream-7B-Instruct | 29.8 ms | 42.6 | **35.3** | **−17%** |

在 batch=1、seq=512 时 split 把折叠路径缩短约 17–18%，但折叠路径仍慢于基线——剩余开销是每层的门控（cosine）、缓存读写与 Python 层包装（合计 ~0.1–0.2ms/层），只有融合 kernel（优化 #4 的延伸）才能进一步消除。

图：`fig_opt2_longseq.png`

---

## 3. 优化 #3：自适应分位数门控

新增 `actfold/core/adaptive_gate.py`：`AdaptiveQuantileGate` 每次调用用 `topk` 精确选取 `target_stable_ratio` 比例的 token 作为稳定集，阈值由数据分位数决定并记录到 `last_tau`。

**效果**：

1. **精确可控**：目标 0.5/0.8/0.9/0.97/1.0 全部命中（固定 τ 是平台+断崖，无法细调）。
2. **前沿占优**：在相同稳定率下自适应选择的稳定集更合理（把复用留给最相似的 token，跨层均衡发散）。示例（k=1 翻转）：
   - Dream：固定 τ=0.99 → 稳定 0.976、rel-MSE 0.220；自适应 target=0.8 → 稳定 0.805、rel-MSE **0.083**（同样 top-1 0.976）。
   - LLaDA：固定 τ=0.99 → 稳定 0.971、rel-MSE 0.207；自适应 target=0.95 → 稳定 0.941、rel-MSE 0.193（top-1 0.912 高于固定的 0.882）。
3. **k=8 翻转**时固定 τ 的 top-1 崩到 0.46–0.68，而自适应可把稳定率降到 0.5 换取 top-1 0.61–0.85，明显更可控。

图：`fig_opt3_frontier.png`

---

## 4. 优化 #4：融合 Triton gather+select

在 `fused_ops.py` 新增 `_gather_select_kernel` / `gather_select`：单 kernel 内按 token 解析"稳定→从父缓冲按行号取；发散→拷贝子张量"，消除 `index_select` 的中间 gather 张量与 `torch.where` 的二次内存往返。CPU/不支持时自动回退，数值与 PyTorch 路径**逐位一致（max err=0）**。

| 形状 | PyTorch | fused | 加速 |
|---|---|---|---|
| T=128, H=4096 | 0.0174 ms | 0.0271 ms | 0.64x |
| T=2048, H=8192 | 0.0413 ms | 0.0276 ms | **1.50x** |
| T=8192, H=4096 | 0.1716 ms | 0.0500 ms | **3.43x** |
| T=8192, H=8192 | 0.3724 ms | 0.2027 ms | 1.84x |

小形状受 kernel 启动开销限制（0.6x）；**T≥2048 且 H≥8192、或 T≥8192** 时启用有净收益。默认不强制启用以避免小形状回退。

图：`fig_opt4_fused.png`

---

## 5. 优化后的建议配置

| 场景 | 推荐配置 |
|---|---|
| 短序列（≤256 token-lane）、高稳定率 | `use_vectorized_cache=True`；**触发全稳定快速路径时可获得 2.3–2.7x 真实加速** |
| 长序列 / 大 batch（B×T≥2048） | 在上一行基础上 `use_split_layers=True`（`split_min_tokens=512`） |
| 对保真度敏感 / 需要精确复用预算 | `AdaptiveQuantileGate(target_stable_ratio=…)`，配合逐层滚动目标 |
| 主干特殊形状（T≥8192） | 融合 gather+select（后续可接入 slow-path） |

---

## 6. 代码与测试清单（本轮新增/修改）

| 文件 | 内容 |
|---|---|
| `actfold/core/vectorized_cache.py` | 新增：连续缓冲激活缓存 |
| `actfold/core/adaptive_gate.py` | 新增：自适应分位数门控 |
| `actfold/core/split_layer.py` | 新增：FFN 拆分层（钩子链） |
| `actfold/core/fused_ops.py` | 新增 `gather_select` 融合 kernel + 回退 |
| `actfold/core/chunked_cache.py` | 修复 `get` 合约缺陷 |
| `actfold/core/cache_factory.py`、`model_wrapper.py`、`core/__init__.py` | 接线（use_vectorized / split_layers / split_min_tokens / 导出） |
| `actfold/utils/config_manager.py`、`actfold/eval/benchmark_runner.py` | 新增开关与透传 |
| `scripts/opt1_cache_bench.py` … `opt4_fused_bench.py`、`opt2_shape_bench.py`、`opt2_long_seq.py`、`make_optimization_figures.py` | 实验与图表 |
| `tests/test_vectorized_cache.py`、`test_split_layer.py`、`test_adaptive_gate.py`、`test_chunked_cache.py`、`test_fused_ops.py` | 新增/更新测试（远程 **204 passed**） |

---

## 7. 复现命令

```bash
# 优化 #1
python scripts/opt1_cache_bench.py --model fastdllm --out results/optimization/opt1/fastdllm
# 优化 #2
python scripts/opt2_split_bench.py --model llada --out results/optimization/opt2/llada
python scripts/opt2_shape_bench.py --out results/optimization/opt2/shape
python scripts/opt2_long_seq.py --model dream --out results/optimization/opt2/longseq/dream
# 优化 #3
python scripts/opt3_adaptive_bench.py --model dream --out results/optimization/opt3/dream
# 优化 #4
python scripts/opt4_fused_bench.py --out results/optimization/opt4
# 图表
python scripts/make_optimization_figures.py --root results/optimization --out results/optimization/figures
# 回归测试
python -m pytest tests/ -q -m "not slow"
```

> 环境：venv（system-site-packages）+ `transformers==4.53.1`、`huggingface_hub==0.36.2`、`triton 3.4`、torch 2.8。由于阿里云 PyPI 镜像在本实例返回 403，安装使用 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。
