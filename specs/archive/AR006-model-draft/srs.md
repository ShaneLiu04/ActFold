# AR006 SRS — 真 draft 模型第一档：`model_draft` 模式与合成验证

> 覆盖指南 M-2 第三层第一档 / P3-3 / B10 最终修复 / AR004 遗留前奏。
> AR 编号：AR006；创建：2026-10-10；状态：draft

## 1. 背景与问题

- **现状**：`DraftGenerator` 五模式中 `logits_draft` 的 logits 来自 **target parent
  forward**（`parent_logits` 参数）——这是"作弊上界"（draft 直接用 target 分布），
  不是真实投机解码语义（draft 模型独立于 target 计算）。`copy_flip`/`random`/
  `perturb` 则与 target 分布零相关（B10：结构性平凡——任何 τ 判 divergent）。
- **缺口**：真实投机解码的 draft 是一个**独立的小模型**：输入 parent tokens，
  输出 logits，成本远低于 target。AR004 遗留注记明确："接受率当前对着
  random/perturb draft 测量，是机制指标；真 draft 模型（P3-3）接入后才成为
  论文口径的接受率"。
- **本 AR 范围（第一档）**：draft model 的**接口与模式**——`DraftGenerator` 新模式
  `model_draft`（`draft_model` 可调用对象：tokens `[B,T]` → logits `[B,T,V]`），
  采样语义复用 `_sample_from_topk`；**不包含** Medusa/Eagle checkpoint 加载与
  head 结构（第二档，需真实权重）。

## 2. 范围

**做：**
- `DraftGenerator` 新模式 `model_draft` + 构造参数 `draft_model`；
- draft model forward 的输入/输出契约（形状校验、dtype/device 对齐）；
- 合成验证：tiny causal target + tiny draft（结构对齐）的接受率非平凡性
  （model_draft 显著优于 random draft，接近但不超过 target-logits 上界）；
- `AblationStudy`/`BenchmarkRunner`/`folded_generate` 接线确认（draft_generator
  参数路径已通，benchmark_runner 构造处支持可选 draft_model 注入）；
- 文档：指南 M-2 第三层第一档 ✅、README 局限性、AGENTS #40 遗留注记更新。

**不做：**
- Medusa/Eagle 真实 checkpoint 加载（第二档）；
- draft 模型训练/蒸馏；
- draft KV cache 逐 token 增量生成（本档 append 位置统一从 parent 末位
  draft logits 采样，与 `logits_draft` 同语义——记录设计决策）；
- 多祖先复用（P3-2）、Branch 惰性引用（P3-1 遗留，生产零调用点价值低）。

## 3. 功能需求

### 3.1 `model_draft` 模式（DraftGenerator）

**描述**：`DraftGenerator` 支持用独立 draft 模型产生候选分支。

**触发条件**：`mode="model_draft"` 构造 + `draft_model` 可调用对象传入。

**期望行为：**
- 新构造参数 `draft_model: Callable[[torch.Tensor], torch.Tensor] | None = None`
  （tokens `[B,T]` int → logits `[B,T,V]` float；接受 nn.Module）；
- `generate` 时：`mode="model_draft"` 且 `draft_model is None` → RuntimeError；
- draft logits 获取：`draft_model(parent.tokens)` 一次 forward（**不消费
  `parent_logits` 参数**——draft 独立性是本模式语义核心）；
- 输出校验：draft logits 形状非 `[B,T,vocab_size]` → ValueError；
- flip 位置采样：复用 `_sample_from_topk`（draft logits 该位置的 top-k 均匀
  采样）；append 位置：parent 末位 draft logits top-k 采样（与 logits_draft
  同语义，设计决策记录）；
- `seed` 复现与全局 RNG 不污染语义与既有模式一致（AR002 §3.6 契约）；
- `parent_logits` 传入 `model_draft` 时：忽略（不报错，文档注记——上界对照
  实验可能同时携带）。

**异常处理：** draft_model 缺失 → RuntimeError；draft logits 形状错、
**非浮点 dtype**（int logits 会静默通过 topk）、**device 与 parent.tokens
不一致**（跨设备采样触发 torch 内部难解错误）→ ValueError；draft_model
forward 抛错 → 原样传播（不吞）。

**验收标准：**
- Given tiny draft model（返回固定 logits 的 stub）+ seed，When `generate`
  mode="model_draft"，Then flip 位置的 token ∈ 该位置 draft top-k 集合，
  且同 seed 两次调用逐位一致；
- Given mode="model_draft" 无 draft_model，Then RuntimeError；
- Given draft stub 返回错误形状，Then ValueError。

### 3.2 合成非平凡性验证（接受率梯度）

**描述**：证明 `model_draft` 的接受率介于 random draft（平凡下界）与
target-logits（作弊上界）之间——真实投机解码的论文口径。

**触发条件**：合成 tiny causal target + 结构对齐 tiny draft。

**期望行为：**
- 合成 target：因果 token-wise 模型（seed 决定确定性映射，`logits[:, i]`
  预测位置 i，AGENTS #40 同位置约定）；
- tiny draft：1 层 vs target 4 层，权重构造使 draft logits 与 target logits
  正相关但不相等（如 draft = target 前 1 层 + 随机投影 head）；
- 接受率测量用 AR004 `acceptance_rate`（同位置 argmax 契约）。

**验收标准（接受率数学口径，design 门控第 1 轮修正）：**
- `_sample_from_topk` 是 top-k **均匀**采样：接受率 = P(target argmax ∈
  draft top-k) / k。接受率测量 fixture 统一 `top_k=2`（k=1 会使
  logits_draft 比较器恒 1.0，k=10 使 gap 上界 1/k−1/V = 0.099 < 0.1 不可达；
  k=2 联合可满足：logits_draft 理论上界 1/2 = 0.5）；
- Given 上述 target/draft（固定 seed，flip 覆盖 ≥16 个位置，top_k=2），
  Then `model_draft` 接受率 − `random` 模式接受率 ≥ 0.1（即 P(hit) ≥
  0.202，强相关 fixture 首层共享可达），且 ≤ 同 fixtures 下
  `logits_draft` 模式的实测接受率（= 1/2 附近，非恒 1.0）；
- Given 无关 draft（随机权重，top_k=2），Then 其接受率与 `random` 模式
  接受率之差 < 0.05（绝对值；理论 ≈ k/V ÷ k = 1/V，梯度下端可复现）。

### 3.3 端到端接线

**描述**：`model_draft` 经既有 `draft_generator` 参数路径进入消融与生成。

**期望行为：**
- `AblationStudy(draft_generator=DraftGenerator(..., mode="model_draft",
  draft_model=...))` 的 `measure_folding` 正常测量（draft 分歧集中后缀）；
- `folded_generate(draft_generator=...)` 多候选路径跑通（候选 tokens 来自
  draft model）；
- `BenchmarkRunner` 构造接受可选 `draft_model` 注入（None → 现行行为零变化）；
- `config.draft_mode="model_draft"` 可配置：`ActFoldConfig.draft_mode` 白名单
  （config_manager.py:110）加入 `"model_draft"`（一行改动，属既有字段的值域
  扩展，无新 config 字段）。

**异常处理（两层校验时序）：** ① config 构造期——白名单校验 draft_mode 值
  合法性（ValueError，既有行为）；② BenchmarkRunner 构造期——config
  `draft_mode="model_draft"` 但未注入 draft_model → RuntimeError（显式，
  AGENTS #10 不静默回退）。

**验收标准：**
- Given model_draft generator + 合成 adapter，When `AblationStudy.measure_folding`
  / `folded_generate`，Then 正常返回且候选 tokens 非全随机（flip 位置 ∈
  draft top-k）；
- Given BenchmarkRunner 无 draft_model + config mode="model_draft"，Then
  构造期 RuntimeError；
- Given现行任何模式（五模式默认路径），Then 全量 738 基线逐位零回归。

### 3.4 文档收口

**期望行为：** 指南 M-2 三层递进第三层第一档 ✅（第二档 checkpoint 待真实
权重）；README draft 模式列表更新；AGENTS #40 遗留注记更新（"真 draft 模型
第一档已接入，论文口径接受率可测；Medusa/Eagle checkpoint 属第二档"）；
CHANGELOG AR006 节。

**验收标准：** Given 文档评审，Then 条目齐全且与实现一致（grep 可验证）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | 零回归 | 全量 738 passed 基线不回归；demo 基线 85.5% / 2.35e-03 / 93.75% 不变（demo 用 suffix_append） |
| 兼容性 | API 向后兼容 | `DraftGenerator` 既有五模式行为逐位不变；新参数 `draft_model` 默认 None |
| 可移植性 | 无 checkpoint 依赖 | 全部行为用合成 stub 可验证（不下载权重） |
| 质量门 | 静态检查 | mypy（AGENTS 标准命令）零错误；pyflakes clean；100 列；Google docstring |

## 5. 约束与假设

**约束：**
- draft model 接口为最小鸭子类型（callable tokens→logits），不引入
  Medusa/Eagle 专属结构（第二档再抽象）；
- AGENTS #10：无 draft_model 时显式 RuntimeError，不静默回退其他模式；
- AGENTS #32/#40：seed 语义与同位置预测约定沿用。

**假设：**
- 第一档 append 位置从 parent 末位 draft logits 采样是可接受的近似
  （真实 draft 逐 token 自回归；对 top-k 均匀采样语义影响有限）；
- 合成 target/draft 的相关性构造（前 1 层 + 投影 head）足以代表
  "draft 分布与 target 正相关但不相等"的真实属性。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 真 draft 模型 | 独立于 target 计算的 draft 分布来源（本档：callable tokens→logits） |
| 作弊上界 | draft 直接用 target parent forward logits（`logits_draft` 模式） |
| 平凡下界 | draft 与 target 分布零相关（`random`/`copy_flip`——B10） |
| 第一档/第二档 | 接口+合成验证 / Medusa/Eagle 真实 checkpoint 加载 |
