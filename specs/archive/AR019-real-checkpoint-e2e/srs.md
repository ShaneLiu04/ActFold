# [AR019] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR019 |
| AR 主题 | 真实 checkpoint 端到端墙钟：position_embeddings 上下文线程化 + Fast-dLLM v2 三路径基准（P1-7 第五档） |
| 关联 SR | P1-7 多分支验证墙钟（AR015-18 四档延续）；AR016/AR017/AR018 三代连续遗留项"真实 checkpoint 端到端墙钟" |
| 日期 | 2026-10-14 |
| 状态 | Draft |

## 1. 背景与目标

P1-7 的四档落地（AR015 批量前向 / AR016 批量图捕获 / AR017 自适应容量 / AR018 发布压缩）已把
合成 LLaMA-layout 基准上的 adaptive vs baseline 加速比推进到 **2.40/2.67/2.66/2.95×**，但全部
证据来自合成模型——"综合框架比 baseline 强"的最终证明需要在**真实 diffusion LLM checkpoint** 上
端到端复现。三代 AR 的遗留清单均点名此项。

本机探针（2026-10-14，见 §1.1）发现一个**可用性缺口**：综合框架当前**根本无法在真实
checkpoint 上运行**——`ManualFoldedForward` 的层调用面不传 `position_embeddings`，而
Fast-dLLM v2 remote attention 第一行就解包该参数（`cos, sin = position_embeddings`），直接
`TypeError`。因此本 AR 包含两个不可分割的部分：**① 生产代码适配**（position_embeddings
上下文线程化，贯穿 eager/批量/graph 静态体三路）；**② 真实 checkpoint 端到端基准**
（Fast-dLLM-v2-1.5B 三路径墙钟 + parity + 门控二分质量），给 P1-7 第五档证据。

### 1.1 选题证据链（本机实测，探针存档于 `Temp\opencode\ar019_probe_*.py`）

| # | 证据 | 探针 |
|---|------|------|
| P1 | **缺口确证**：`ManualFoldedForward` 在 Fast-dLLM-v2-1.5B 上 parent prefill 即 `TypeError: cannot unpack non-iterable NoneType`（remote attention `cos, sin = position_embeddings`）；`detect_architecture` 本身成功（`model.embed_tokens`/28 层/`model.norm`/`lm_head` 全部发现） | `ar019_probe_folded.py` |
| P2 | monkey-patch 注入 `position_embeddings` + block-causal mask 后，**parent prefill 逐位一致**（maxdiff=0.00000, argmax=1.000000）——适配方向正确 | `ar019_probe_parity.py` / `ar019_probe_block.py` |
| P3 | **真实块验证工作负载**（parent = prompt 块 + 全 mask 末块；children = 末块不同填充，模型 top-10 采样）：批量折叠 vs baseline **逐位一致**（maxdiff=0.00000, argmax_agree=1.000000），门控**完美二分**（stable_mean = stable_min = 0.6667，恰等于理论值 K_PROMPT/T=64/96，28 层 × 4 分支零误判） | `ar019_probe_block.py` |
| P4 | **块因果语义论证**（精确性的机理依据）：`eval_block_diff_mask` 为 `block_q >= block_kv`（32-token 块内双向、跨块因果）⇒ prompt 块（0..k-1）的注意力**不包含**末块 ⇒ 与末块内容无关 ⇒ 位级稳定；mask 嵌入 vs 真实 token 嵌入强发散 ⇒ tau=0.95 门控鲁棒二分 | remote `modeling.py:89-94` + 探针 P3 |
| P5 | **负面证据（工作负载选型）**："少量 token 翻转"工作负载下 parity 仅 92.9%（块内双向注意力使翻转传播到同块未翻位置，tau=0.95 判其为 stable——折叠成为近似）；块验证工作负载才是 Fast-dLLM 自身的验证语义（`use_block_cache`/`replace_position` 块解码流程），且折叠精确 | `ar019_probe_parity.py` |
| P6 | **环境**：本机 HF hub 的 requests 栈与公司代理不兼容（407/500 隧道错），`curl.exe` 可用（权重 3.09GB 已下载）；transformers 5.15.1 与 remote code 不兼容（AGENTS #24），venv 隔离 `transformers==4.53.1` + `--system-site-packages`（继承 torch 2.5.1+cu121）实测可行 | `ar019_probe_model.py` |
| P7 | **数值面**：Triton bf16 merge kernel 本机编译失败（无 C 编译器）→ AR014 粒度回退 PyTorch path（实测工作，一次性 RuntimeWarning）；graph 静态体的 fused gate kernel bf16 行为需在设计中确认回退安全性 | `ar019_probe_parity.py` 警告面 |

### 1.2 模型与工作负载选型依据

- **Fast-dLLM-v2-1.5B**（`Efficient-Large-Model/Fast_dLLM_v2_1.5B`）：唯一在本机 16GB 显存下
  可行的主要家族 checkpoint（bf16 权重 3.1GB，探针峰值 ~4GB @ T=96/N=4）；LLaDA-8B/Dream-7B
  bf16 权重 14-16GB，叠加图缓冲不可行（Out of Scope，更大 GPU 后按同法扩展）。
- **验证工作负载**：块解码验证——parent 为 prompt（前 k-1 块）+ 末块全 mask token
  （`mask_id=151665`）；N 个 children 为末块的不同模型采样填充。这是 Fast-dLLM 自身
  `replace_position` 块重写流程的投机验证抽象，与 AR016-18 的"同形多候选验证"口径一致。
- **baseline 口径**：逐 child 裸模型前向（N 次 `[1, T]`）——与 AR016/17/18 基准同口径。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- FR1 `position_embeddings` 上下文线程化（生产代码，四构建面全覆盖）
- FR2 真实 checkpoint 端到端基准脚本 + 工件
- FR3 假设-判定分离的工件 schema（H1-H4）
- FR4 环境与复现文档（venv 规格 / 模型获取 / RERUN_CHECKLIST 方法论）

**Out of Scope（本 AR 不做的）：**

- LLaDA-8B / Dream-7B（显存不可行；机制对它们同样适用，留路线图）
- `FastDLLMAdapter.folded_model` 生产接线（基准直接构造 `ManualFoldedForward`，沿 AR016-18
  先例；adapter 集成为后续独立条目）
- 变长 / 多 parent 批量（AR016 遗留，正交可叠加）
- remote modeling 的 training 分支（`bd_size` 噪声重塑路径）
- cache 批量 put API（AR018 srs §1.3 排除决策延续）
- 训练/微调路径、lm-eval 评测（本机无 lm_eval）

## 3. 功能需求

### 3.1 FR1 position_embeddings 上下文线程化

**描述**：让 `ManualFoldedForward` 支持调用方供给 rotary 位置上下文
`position_embeddings: tuple[torch.Tensor, torch.Tensor]`（`(cos, sin)`），并贯穿全部三条
执行路径传到原始层。

**触发条件**：真实 HF/remote 模型层需要 rotary（层签名含 `position_embeddings`）；合成模型
（层签名不含该参数）继续零变化。

**期望行为**：
- `ManualFoldedForward.forward(..., position_embeddings=None)` 新增可选参数；非 None 时
  随 `attention_mask` 同路传入，覆盖**四个构建面**：
  1. 串行 eager 路径（wrapped 层循环）；
  2. 批量 eager 路径（`branch_ids` 分支）；
  3. graph 静态体（`FoldedGraphRunner` 捕获时烘焙进静态缓冲，重放零变化——`cos/sin` 由
     形状决定，对固定形状图是常量；不得在 capture body 内新增任何 host sync）；
  4. **AR017 自适应重捕获面**（`_recapture_batched_runner` 再构造新 runner 时同样烘焙
     `position_embeddings`）——shrink/growth 重捕获后的新 runner 不得丢失 rotary 上下文，
     否则 `graph_adaptive` 路径静默错算。
- 层签名过滤机制（`_recompute_all` 的 accepted/varkw 面向）复用：`position_embeddings`
  仅透传给接受它的原始层，合成层自动忽略，行为逐位不变。
- `_publish_graph_children` / prefill / 预算读回路径的零 host-sync 契约（AGENTS #37）不得破坏。

**graph 重放期契约（对偶 `attention_mask` 对象恒等检查）**：重放时调用方传入的
`position_embeddings` 必须与捕获时为**同一对象**（`is` 恒等，镜像
`attention_mask is not self._graph_mask` 的既有检查）；不同对象 → **静默**回退 eager
（与 AR002 mask 对象不一致同款语义，不告警不吞错）。适用条件（"固定形状 ⇒ cos/sin
常量"的成立前提）明文化：position_ids 恒为 `arange(T)` 的常量位置方案——本 AR 工作负载
（无 KV cache、无偏移位置）满足；非常量位置方案超出本 AR 范围（路线图）。

**异常处理**：`position_embeddings` 形状非法（非二元组 / 非张量 / seq 维与 tokens 不一致 /
cos 与 sin 内部 head_dim 不一致）→ 调用期显式 `ValueError`；graph 路径捕获时
`position_embeddings` 与 `attention_mask` 的形状与捕获形状不一致 → 拒绝捕获并回退 eager
（沿既有 degradation 语义：形状类前置失败沿 `_parent_cache_complete` 的静默回退档，不吞
其他错误；**head_dim 与层内期望失配的例外**——该值在层内部而非签名，调用期无从校验，
wide-capture 转静默会违反"不吞其他错误"，故在 warmup 内以层内广播错误暴露 →
capture-failed 可见档（一次性告警 + 禁用），见 design 4.2.2 第 1 条）；重放期对象不一致
→ 上段静默回退档。

**验收标准**：
- Given 一个层签名含 `position_embeddings` 的 stub 模型，When 串行/批量/graph 三路折叠前向
  with `position_embeddings=(cos, sin)`，Then 原始层收到的 `cos/sin` 与供给值逐位一致，且
  输出与"手工把 `cos/sin` 焊进层"的对照路径一致。
- Given 自适应容量触发 shrink/growth 重捕获，When 新 runner 落成后重放，Then 原始层收到的
  `cos/sin` 仍与捕获时供给值逐位一致（重捕获面烘焙用例）。
- Given 合成模型（无该参数），When 任何现有测试面运行，Then 全部逐位不变（855 passed
  基线零回归）。
- Given graph 捕获，When 重放 N 次，Then `position_embeddings` 不引入任何新增 host sync
  （捕获体静态缓冲断言，UT 层面以 spy 验证层调用参数来源为静态缓冲）。
- Given graph 已捕获 with `pe_a`，When 以同形不同对象 `pe_b` 重放，Then 静默回退 eager
  且输出正确（对象恒等契约用例）。

### 3.2 FR2 真实 checkpoint 端到端基准

**描述**：`scripts/ar019_real_e2e_bench.py`（沿 AR017/18 脚本谱系，运行于 AR019 venv），
在 Fast-dLLM-v2-1.5B 上度量同形多候选块验证工作负载的三路径墙钟。

**触发条件**：`python -m scripts.ar019_real_e2e_bench`（venv 内；`--quick` 冒烟；
`--model-path` 默认探针下载的本地目录，可覆写）。

**期望行为**：
- 三路径：`baseline_eager`（裸模型 N 次 `[1,T]` 前向）/ `graph_fixed`（AR016 批量图，
  固定容量比 0.5）/ `graph_adaptive`（AR017 自适应容量 + AR018 压缩发布）。
- 工作负载：真实 prompt（前 k-1 块）+ 末块全 mask 的 parent；N 个末块填充候选
  （模型 top-10 采样，种子固定）；`cos/sin` 由 `model.model.rotary_emb` 计算、block-causal
  mask 按 `eval_block_diff_mask` 语义构造（脚本层复现，作为 `position_embeddings` /
  `attention_mask` 供给 FR1 接口）。**baseline 侧同享完全相同的 mask/cos-sin 供给语义**
  （raw 模型 eval 前向自身构建同款 mask + rotary，两侧语义同一）——parity 才有可比性。
- 计时：`perf_counter` + `torch.cuda.synchronize()` 边界，逐 rep round-robin 交错，
  warmup 覆盖捕获 + 适配期，reps=30（沿 AR018 修正后的预承诺单次高功效口径；AR016/17
  的 n=10 欠功效教训在案），p50/std。
- 侧车（不计时）：parity（argmax 一致率 + bf16 档 allclose）、stable ratio 观测
  （profiler）、容量轨迹（`AdaptiveCapacityPolicy.events`）、峰值显存、Triton 回退键记录。
- 工件：`results/optimization/ar019_real_e2e_bench.json`，meta 含 venv/transformers/torch/
  驱动版本、模型路径与 SHA256、CUPTI null 注记（#39）。
- 配置矩阵：T=256（8 块）× N∈{4, 8}；几何外推配置（如 T=512×N=4）在显存预算内可选
  （设计阶段定案）。

**异常处理**：模型路径缺失/权重不全 → 显式 `FileNotFoundError`（不回退合成模型，#10）；
graph 捕获失败 → 记录并使该路径基准失败可见（不静默替换）；预算违例回退 eager 的次数与
pe 恒等静默回退次数（预期为零）均入工件。

**验收标准**：
- Given venv + 本地权重，When 全量运行（reps=30），Then 工件含三路径 timings/speedups/
  parity/容量轨迹/显存，H1-H4 判定字段齐全，且单次运行时长 < 15 分钟。
- Given `--quick`，When 冒烟运行，Then 全路径走通并在 < 3 分钟内完成。

### 3.3 FR3 假设-判定分离

**描述**：假设在脚本 docstring 与 srs 预先声明，判定仅来自工件数据，取值
`supported / refuted / inconclusive`。

**期望行为**（判定字段）：
- **H1 综合框架 vs baseline**：`graph_adaptive` p50 < `baseline_eager` p50（方向）且过
  2σ 检验（AR017 `_verdict` 口径）——P1-7 在真实 checkpoint 上的最终论点。
- **H2 parity**：`graph_adaptive` vs 逐 child baseline 的 argmax 一致率 == 1.0 且 allclose
  过 bf16 档容差（探针 P3 已示逐位一致，断言档放宽防 BLAS 微内核边界，AGENTS #41 先例）。
- **H3 门控二分质量**：各层 stable ratio 的 min/mean 与理论值 `K_PROMPT/T` 一致
  （±2σ 或固定容差，设计定案）——块因果语义（P4）在 28 层全深度成立。
- **H4 图捕获与容量**：批量图捕获成功（`graph_runner_batched` 非 None 且形状 `[N,T]`）、
  自适应容量轨迹收敛（事件数有限、稳态容量 ≈ 末块 token 数）。

**验收标准**：工件四字段齐全；判定与数据可分离复核（判定函数纯数据输入）。

### 3.4 FR4 环境与复现文档

**描述**：把本机环境适配（venv 规格、权重获取、代理怪癖）固化为可复现文档。

**期望行为**：
- `docs/RERUN_CHECKLIST.md` 增补 AR019 节：venv 创建命令
  （`python -m venv --system-site-packages` + `transformers==4.53.1` + `huggingface_hub<1.0`
  + `einops`）、权重 curl 下载清单（11 个小文件 + `model.safetensors`）、运行命令。
- `docs/OPTIMIZATION_GUIDE.md` 第二十六部分：AR019 证据表 + 语义要点 + 遗留。
- `CHANGELOG.md` AR019 节；`AGENTS.md` 增补 position_embeddings 契约注记（#37 延伸）。
- **README 全方位中文更新**（AGENTS.md 新规：每创新 AR 后强制）。

**验收标准**：按文档从零可在同型机器复现 venv + 权重 + 基准运行。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 回归 | 全量测试 | 主 env `pytest tests\ -q -m "not slow"` 855 passed 零扰动（新 UT 计入后基线相应上移） |
| 静态检查 | mypy / pyflakes / black | mypy `--ignore-missing-imports` clean；pyflakes 零；100 列零违规 |
| 语义不变 | 现有路径 | `position_embeddings=None`（默认）时全部现有行为逐位不变 |
| 图契约 | 零 host sync | 捕获体内无 `.item()/.tolist()/.cpu()`（#37） |
| 墙钟纪律 | 工件 | CUPTI null + 注记；单次预承诺运行；工件不手改（#39） |
| 时间预算 | 基准 | 全量 < 15 分钟；冒烟 < 3 分钟 |

## 5. 约束与假设

**约束：**
- 主 env 不降级 transformers（5.15.1 服务于 855 passed 基线）；AR019 venv 与主 env 隔离，
  生产 UT 不依赖 transformers remote code（stub 层覆盖 FR1）。
- 图路径的全部前置条件（#37：CUDA / 精确类型 `SimilarityGate` cosine / 无 scheduler /
  同形 / 同 attention-mask 对象 / **同 position_embeddings 对象（本 AR 新增）** / parent
  缓存完备）继续成立；`position_embeddings` 的静态烘焙不得引入新的**硬**失败面——pe
  对象违例沿 mask 同款静默回退档，不新增任何抛错路径。
- bf16 数值：parity 断言用 argmax + bf16 档 allclose；探针实测逐位一致为实证记录，
  断言口径仍放宽（BLAS gemv/gemm 微内核边界 1-ULP 先例，#41）。
- 生成/评测路径零改动：`folded_generate` / samplers / eval 不感知 `position_embeddings`
  （本 AR 的消费面是基准脚本与未来的 adapter 接线）。

**风险表（AR018 先例延续）：**

| 风险 | 量级/依据 | 处置 |
|------|-----------|------|
| 显存：logits 静态缓冲 `[N,T,V]` bf16（V=151936） | N=8/T=256 ≈ 622MB + 侧车 clone 瞬态同量；N=4/T=512 ≈ 622MB 同量级 | 工件记录峰值显存；T=512×N=8 组合（≈1.24GB logits ×2）不进默认矩阵，仅显存预算复核后作外推配置 |
| 显存：child/parent 静态缓冲 `[L,N,T,H]` | N=8/T=256：28×8×256×1536×2 ≈ 176MB ×2（child+parent）+ eager 侧 cache | 同上 |
| 显存：重捕获瞬态新旧双 runner 并存 | ≈ 单 runner 静态面 ×2，事件数有限（AR017 上限退役保证） | 工件容量轨迹佐证；16GB 卡峰值预算 <10GB（模型 3.1GB + 上述合计） |
| 真实 bf16 图捕获未探针实证（P3 走的是批量 eager；fused gate kernel bf16 编译失败时的捕获体内回退行为未知） | P7 仅 eager 面证据；H4 可能 inconclusive | T003 `--quick` 冒烟先验证捕获；失败可见（FR2 异常处理），eager 回退计数入工件，H4 判定字段如实记录 refuted/inconclusive 亦为合法产出 |
| WDDM 计时展布（小几何欠功效先例） | AR017 (256,8,4) 2σ 展布受限 | 真实模型单前向 ~10ms 量级（合成小几何 ~1ms），效应量更大；仍以 p50 + 2σ 口径判定，inconclusive 如实记录 |

**假设：**
- Fast-dLLM-v2-1.5B 本地权重目录持续可用（探针已下载 3.09GB；SHA256 入工件）。
- `eval_block_diff_mask`（`block_q >= block_kv`，block_size=32，cache_seq_len=0）是 eval
  验证工作负载的正确语义基准（remote `modeling.py` 静态读取——
  `mask = (q_idx[:, None] // 32) >= (kv_idx[None, :] // 32)` 语义——+ 探针 P3 parity 实证）。
- 本机（Quadro RTX 5000 16GB / torch 2.5.1+cu121 / WDDM）为基准执行环境；WDDM 计时展布
  沿 AR017/18 先例以 p50 + 2σ 口径处理。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 块验证工作负载 | parent = 前 k-1 块真实 prompt + 末块全 mask；children = 末块不同填充的多候选；Fast-dLLM 块解码 `replace_position` 流程的投机验证抽象 |
| block-causal mask | `block_q >= block_kv`：32-token 块内双向、跨块因果（`eval_block_diff_mask`） |
| position_embeddings | `(cos, sin)` rotary 张量，模型级 `rotary_emb(hidden, position_ids)` 产出、逐层共享；形状由 `[B, T]` 决定，在 position_ids 恒为 `arange(T)` 的常量位置方案下固定形状为常量 |
| AR019 venv | `Temp\opencode\ar019_venv`：`--system-site-packages` 继承 torch 2.5.1+cu121，局部安装 `transformers==4.53.1` / `huggingface_hub<1.0` / `einops` |
