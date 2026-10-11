# AR007 SRS — P3-2 第一档：parent 更长时的截断前缀复用

> 覆盖指南 :199 P3-2 子项"parent 更长的截断复用"（AR003 遗留对称补全）。
> AR 编号：AR007；创建：2026-10-11；状态：draft

## 1. 背景与问题

- **现状（AR003 语义）**：`FoldedTransformerLayer` 前向中 parent 激活比 child 长
  （`T_p > T_c`）时**视为无 parent**（全重算，`folded_transformer.py:175-182`）。
  AGENTS #20 与 README #7 均记录该保守设计。
- **缺口**：tree/beam 式投机解码存在 child 比 parent 短的负载（回退到更短
  序列、兄弟分支收缩）。当 child tokens 是 parent tokens 的真前缀时，child
  的前 `T_c` 个位置激活与 parent 前 `T_c` 个位置逐位一致（因果模型），当前
  却全重算——白白丢弃可用复用。`append-only`（child 更长）已在 AR003 支持，
  parent 更长是对称的另一半。
- **正确性依据**：gate 是语义安全网——child 前 `T_c` token 与 parent 前
  `T_c` **不同**时，相似度低于 τ → divergent → 重算（与 append-only 中
  非 causal 模型的既有论证一致）；**相同**（真前缀）时全 stable 复用。

## 2. 范围

**做：**
- `FoldedTransformerLayer`：parent 更长 → 截断到 `T_c` 复用（gate 输入与
  parent FFN 输出**双截断**，merge 走等长语义）；
- `fetch_flat`（gather_select）快路径在 parent 更长时**跳过**（其按 child
  `seq_len` 索引 parent 行的契约不保证成立；走 `fetch` + 截断 +
  `merge_stable_divergent` 安全路径）；
- 合成验证：真前缀 child 全 stable 且输出与全重算 bit-exact；非前缀 child
  divergent 重算一致；既有等长/append-only 路径逐位零回归；
- 文档：指南 P3-2 截断子项勾选（第十四部分回链）、AGENTS #20 修订、
  README #7 修订、CHANGELOG AR007 节。

**不做：**
- `BranchManager.align_tokens` 的 child 更短 ValueError 放宽（该类生产零
  调用点，文档语义明确，无改动价值——srs §5 约束记录）；
- 多祖先复用（多 parent paged 结构，P3-2 主体，独立 AR）；
- CUDA graph 路径改动（`_parent_cache_complete` 形状检查天然把变长 parent
  路由到 eager 回退，零干扰——测试实证即可）；
- `Branch.hidden_states` 惰性引用化（P3-1 遗留独立条目）。

## 3. 功能需求

### 3.1 parent 更长的截断复用（`FoldedTransformerLayer`）

**描述**：parent 激活 `[B, T_p, H]`（`T_p > T_c`）不再视为无 parent，而是
截断到 `[B, T_c, H]` 参与正常折叠流程。

**触发条件**：child 前向时 cache 中 parent 激活长度 > child 序列长度，
且 batch/hidden 维一致、parent 长度非零。

**期望行为：**
- gate 输入截断：`h_parent = h_parent[:, :T_c]`，后续与等长路径一致
  （`prefix_len == seq_len` → fused gate 或普通 gate 全长比对）；
- parent FFN 输出截断：快路径（全 stable）fetch parent `ffn_out`
  `[B, T_p, H]` → 截断到 `[B, T_c, H]` 再比对形状、复用；慢路径
  `_get_parent_ffn_output` 同样截断（merge 按等长语义，`prefix_len=None`）；
- `fetch_flat` 快路径跳过：parent 更长时 `flat_rows` 按缓存布局索引，
  与 child 长度的对应关系不保证 → 不走 gather_select，直接 `fetch` +
  截断 + `merge_stable_divergent`（结果 bit-identical，仅少一次融合）；
- 既有失效条件不变：batch/hidden 不匹配、parent 长度为 0 → 仍视为无
  parent 全重算；
- CUDA graph 路径零干扰：变长 parent 由 `_parent_cache_complete` 形状
  检查路由到 eager（既有行为），不捕获、不告警、不禁用。

**异常处理：** 无新增异常（截断是形状自适应，不引入错误路径）；既有
`RuntimeError`（parent FFN 缺失、或短于所需形状）语义不变；varlen 路径
（`prefix_len` 非 None）的精确 shape 契约不变；等长/截断后路径上 ffn
长于 mask 的情况按形状截断（异常 cache 状态的确定性行为）。

**验收标准：**
- Given 因果合成模型 + parent 8 token + child 6 token（真前缀，cache 有
  parent 激活），When child 走 folded 前向，Then 每层输出与全重算
  `torch.equal` 且 `stable_ratio == 1.0`（全复用，AR003 等长路径同款
  验证强度）；
- Given 同上但 child 6 token 与 parent 前 6 token 不同（非前缀），Then
  输出与全重算 `torch.equal`（gate 判 divergent → 重算，语义安全网）；
- Given parent 更长且 batch 不匹配，Then 仍视为无 parent 全重算
  （既有失效条件零回归）。

### 3.2 端到端与零回归

**描述**：截断复用经 `ManualFoldedForward` 端到端可达，既有路径零回归。

**期望行为：**
- `ManualFoldedForward` 包裹的模型在 parent 更长场景正常折叠（无需改
  `ManualFoldedForward` 自身——截断在 `FoldedTransformerLayer` 层完成）；
- 既有等长折叠、append-only 前缀折叠路径**逐位零回归**；
- CUDA graph 模式下变长 parent 步骤走 silent eager 回退（AR003 既有
  契约），不因截断复用改变。

**验收标准：**
- Given `ManualFoldedForward` + parent 更长 child，When folded 前向，Then
  正常返回且激活 cache 存有 child 条目；
- Given 全量测试套（754 基线），Then 零回归；
- Given demo 基线，Then 85.5% / 2.35e-03 / 93.75% 不变（demo 走
  append-only，不触发截断路径）。

### 3.3 文档收口

**期望行为：** 指南 :199 P3-2"parent 更长的截断复用"勾选 + 第十四部分
回链；AGENTS #20 修订（"parent 更长视为 no-parent"→"截断复用"）；
README 局限性 #7 修订；CHANGELOG AR007 节。

**验收标准：** Given 文档评审，Then 条目齐全且与实现一致（grep 可验证，
含旧表述清理：folded_transformer.py:179-181 注释、branch_manager
docstring 的 "truncation reuse is unsupported" 措辞核对——后者属
`align_tokens` 未改动路径，保留其原文但 srs 记录决策）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | 零回归 | 全量 754 passed 基线不回归；demo 基线不变 |
| 兼容性 | 行为兼容 | 既有等长/append-only/CUDA graph 路径逐位不变；"视为无 parent"仅收窄为"仅 batch/hidden 不匹配或零长度" |
| 可移植性 | 无 checkpoint 依赖 | 合成模型可验证 |
| 质量门 | 静态检查 | mypy（AGENTS 标准命令）零错误；pyflakes clean；100 列；Google docstring |

## 5. 约束与假设

**约束：**
- 截断**不要求** child tokens 与 parent tokens 前缀相同——gate 相似度是
  语义安全网（与 AR003 非 causal 模型论证同源）；真前缀是**收益**场景
  而非**正确性**前提；
- `fetch_flat` 快路径跳过是保守决策（bit-identical，仅性能），不尝试
  推导 vectorized cache 的行索引在 parent 更长时的语义；
- `BranchManager.align_tokens` 不放宽（生产零调用；其 ValueError 文档
  语义保留，与 folded 层截断复用不矛盾——两个独立入口）；
- AGENTS #10：无静默行为切换——截断复用改变的是"parent 更长"这一原先
  定义为 no-parent 的分支语义，按 AGENTS "Documentation" 节契约在
  #20/README/CHANGELOG 显式记录（编号注：行为文档化要求源出
  Documentation 节而非 #10 本条，本条引意不引号）。

**假设：**
- 因果模型中 child 真前缀时 parent 前 `T_c` 激活与 child 逐位一致
  （与 AR003 append-only 对称论证）；
- 变长 parent 步骤不进入 CUDA graph（`_parent_cache_complete` 既有形状
  检查保证）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 截断复用 | parent 更长时取前 `T_c` 参与折叠（gate 输入 + parent FFN 双截断） |
| 真前缀 | child tokens[:, :T_c] == parent tokens[:, :T_c] 逐位相同 |
| append-only | child ≥ parent 的既有变长语义（AR003） |
