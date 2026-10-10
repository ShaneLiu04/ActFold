# [AR003] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR003 |
| AR 主题 | 变量长度前缀折叠（P3-1）+ AR002 遗留接线（benchmark_runner 迁移 Manual + graph config 消费） |
| 关联 SR | 无（源自 `docs/OPTIMIZATION_GUIDE.md` P3-1 / M6 场景扩展，与 AR002 遗留待办） |
| 日期 | 2026-10-10 |
| 状态 | Draft |

## 1. 背景与目标

README 局限性 #7 与 OPTIMIZATION_GUIDE P3-1 自认：**变量长度折叠不支持**——
`BranchManager.align_tokens` 直接 `NotImplementedError`，而 `FoldedTransformerLayer`
的 gate/merge 契约要求 parent 与 child 序列等长。

AR002 review 阶段实证了该限制的实际代价：`folded_generate`（AR 每步 append
1 token）中 child 恒比 parent 长 1，等长约束下 parent 激活被视为不可复用 →
**folded_generate 从未真正折叠**（stable ratio 恒 0.0，AGENTS #20 记录的语义）。
真实投机解码负载（draft 接受不同数量 token）天然变长，不解锁变长折叠，
branch folding 无法进入真实工作负载（M6 第一项）。

本 AR 目标：实现 **append-only 前缀折叠**——child 为 parent 的前缀扩展
（`T_c ≥ T_p`、batch 相同）时，前缀位置走既有相似度门控复用 parent 激活，
后缀位置恒 divergent 全量重算；使 `folded_generate` 真正折叠（因果模型下
前缀 stable ≈ 1，每步仅重算新增 token）；并完成 AR002 遗留接线
（benchmark_runner 迁移 `ManualFoldedForward` + 消费
`ActFoldConfig.use_cuda_graph/graph_capacity_ratio`）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `actfold/core/folded_transformer.py`：前缀折叠语义（gate 前缀比较、后缀恒
  divergent、merge 前缀对齐合并、三路判定适配、`_store_activations` 链式递归）
- `actfold/core/branch_manager.py`：`align_tokens` 前缀对齐实现（替换
  `NotImplementedError`）
- `actfold/speculative/folded_generation.py` 相关行为（无需改代码，但
  folded_generate 折叠激活后的端到端验证）
- `actfold/eval/benchmark_runner.py`：`_build_folded_model` 迁移
  `ManualFoldedForward`（消费 `use_cuda_graph`/`graph_capacity_ratio`，
  deprecated `FoldedModel` 退出生产路径）
- CUDA graph 路径零干扰验证（变长步骤不捕获、不禁用、静默 eager）
- 配套测试、README #7 状态更新、AGENTS #20 修订、CHANGELOG、
  OPTIMIZATION_GUIDE P3-1 勾选

**Out of Scope（本 AR 不做的）：**
- 多祖先复用 / paged 激活树（P3-2）
- parent 更长时的截断复用（child 为 parent 前缀的场景；投机解码 child 恒更长）
- `Branch.hidden_states` 全量持有的惰性引用化（P3 独立条目）
- KV-cache baseline 接入（P2-4，方法学对照公平性，独立 AR）
- 变长 CUDA graph 捕获（graph 保持固定 shape 契约；变长走 eager 前缀折叠）

## 3. 功能需求

### 3.1 前缀折叠语义（`FoldedTransformerLayer`，append-only）

**描述：** parent 缓存激活长度 `T_p < T_c`（batch 相同）时，child 前缀位置
与 parent 对齐折叠，后缀位置恒 divergent。

**触发条件：** folded child 前向且 parent 缓存存在且 `0 < T_p < T_c` 且
batch 相同；token 信任 branch 链语义（`parent_branch_id` 链蕴含前缀关系，
不做 token 内容校验——cache 不存 token）。

**期望行为：**
- gate 比较仅前缀：`h_child[:, :T_p]` vs `h_parent` → 前缀 stable mask
  `[B, T_p]`；完整 mask 为 `[B, T_c]` = 前缀 mask 拼接全 False 后缀；
- 三路判定适配：`T_p < T_c` 时后缀恒 divergent → 不可能全 stable（全 stable
  快路径仅 `T_p == T_c` 可达，等长语义不变）；全 divergent 路径不变；
  mixed 路径 merge：parent ffn `[B, T_p, H]` 前缀对齐（后缀位置以 child
  重算值填充，merge 后缀位置本就取 child 值）后走既有
  `merge_stable_divergent`/`gather_select` 链；
- `T_p > T_c` → 视为无可复用 parent（全量重算，与 cache miss 同语义，
  不做截断复用）；
- split 层（`SplitFoldedTransformerLayer`）在完整 mask `[B, T_c]` 上工作
  （divergent gather 含后缀行），既有 `stable_count` 转发语义不变；
- child 激活按 `T_c` 存储 → 链式递归（下一步 parent = 本 child）天然成立。

**异常处理：** batch 不同 / parent 条目缺失 → 现行 cache miss 语义（全量
重算）；`attention_mask` 非 None 时按 child 长度正常传入 original layer
（既有行为），gate/merge 不受影响。

**验收标准：**
- Given `T_p < T_c` 的 mixed 前缀场景，When 前缀折叠 child 前向，Then 输出
  == 参考合成语义（前缀 stable ← parent ffn、前缀 divergent + 后缀 ← child
  重算）逐位一致；
- Given `T_p == T_c`（等长），When 任意既有测试，Then 行为逐位不变（零回归）；
- Given `T_p > T_c`，When child 前向，Then 全量重算（无异常、无复用）；
- Given 后缀长度 ≥1 的任意分布（后缀全 divergent、前缀全 stable、前缀 mixed），
  Then mask 组装与 merge 结果正确。

### 3.2 `BranchManager.align_tokens` 前缀对齐

**描述：** 实现变长分支的 token 对齐（P3-1 的 API 层面交付）。

**触发条件：** `align_tokens(parent, child)` 且 `T_c ≥ T_p` 且 batch 相同。

**期望行为：**
- 返回前缀对齐的 `(h_parent, h_child)`：均取 layer 0 输入 hidden 的
  **前缀对齐形态**——`h_parent` 为 parent layer 0 hidden `[B, T_p, H]`，
  `h_child` 为 child layer 0 hidden 的前缀切片 `[B, T_p, H]`；调用方以
  `child.tokens.shape[1] - parent.tokens.shape[1]` 获知未对齐后缀长度；
- `T_p == T_c` 行为与现行完全一致（返回双方完整 layer 0 hidden）。

**异常处理：** `T_c < T_p` → `ValueError`（child 非前缀扩展，语义非法，
不做截断）；batch 不同 → `ValueError`。

**验收标准：**
- Given `T_c > T_p`，When `align_tokens`，Then 返回形态正确的对齐对，
  无 `NotImplementedError`；
- Given `T_p == T_c`，Then 与现行返回逐位一致；
- Given `T_c < T_p` 或 batch 不同，Then `ValueError`。

### 3.3 `folded_generate` 真正折叠（端到端）

**描述：** 前缀折叠落地后，AR 生成循环逐步真正复用 parent 激活。

**触发条件：** `folded_generate` + folded model（FoldedModel 或 Manual）。

**期望行为：**
- 每步 child = parent + 1 token → 前缀折叠激活：因果模型下前缀 token 与
  parent 相同 → 前缀 hidden 逐位一致 → cosine ≈ 1 → 前缀全 stable；
  非因果/玩具模型下按实际相似度门控（机制正确性优先于比例数值）；
- `result.stable_ratio` 反映真实折叠（因果合成模型上 > 0.5）；
- 生成 tokens 与不折叠路径完全一致（折叠是复用决策，不改变 greedy argmax
  语义——前缀 stable 复用 parent ffn，因果下逐位等价）。

**异常处理：** CUDA graph 模式下变长步骤：`_parent_cache_complete` shape
校验判 incomplete → 静默 eager（不捕获、不告警、不禁用）；固定 shape 场景
（diffusion 验证）graph 契约不变。

**验收标准：**
- Given 因果合成模型 + `folded_generate`（≥4 步），When 对比 eager 全重算
  路径，Then tokens 完全一致且 `stable_ratio > 0.5`；
- Given `use_cuda_graph=True` 的 folded_generate，Then graph 不捕获变长步骤
  （`graph_runner is None`）、零 UserWarning、tokens 一致；
- Given AR002 的等长固定 shape 验证循环测试，Then 全部不回归。

### 3.4 benchmark_runner 迁移 Manual + graph config 接线（AR002 遗留）

**描述：** 生产路径切换到 `ManualFoldedForward` 并消费 graph 配置字段。

**触发条件：** `BenchmarkRunner` 构造（config 驱动路径）。

**期望行为：**
- `_build_folded_model` 构造 `ManualFoldedForward`（cache/gate/scheduler/
  split 参数同现行；新增消费 `config.use_cuda_graph`、
  `config.graph_capacity_ratio`）；架构检测失败（无 layer 栈/head）→
  返回 `None`（对齐现行 `folding_applied` 语义）；
- `_build_engine` 共享 Manual 的 cache/gate/scheduler（属性名兼容）；
- `FastDLLMAdapter.folded_model` / `folded_generate` 接受 Manual（AR002 已
  放宽注解，此处消费）；
- deprecated `FoldedModel` 退出生产构造路径（保留 API 本身）。

**异常处理：** 架构检测失败 → folded_model 为 None（现行行为，不抛出）。

**验收标准：**
- Given 可检测架构的合成模型 config，When 构造 `BenchmarkRunner`，Then
  `folded_model` 为 `ManualFoldedForward` 实例且 `use_cuda_graph`/
  `graph_capacity_ratio` 从 config 传入；
- Given 不可检测架构，Then `folded_model is None`；
- Given `config.use_cuda_graph=True` + CUDA，Then Manual 携带该 flag（行为
  由 §3.3 graph 契约约束）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | 等长零回归 | 既有全部等长折叠测试（AR001/AR002 基线 628 passed）不回归；demo 基线 85.5% / 2.35e-03 / 93.75% 精确匹配 |
| 正确性 | 变长参考语义 | 前缀折叠输出与参考合成语义（mask 组装 + merge 组合）逐位一致 |
| 性能（代理） | AR 步折叠率 | 因果合成模型 folded_generate stable_ratio > 0.5（机制生效证据） |
| 可移植性 | CPU / 非 CUDA | 前缀折叠全功能可用（fold 不依赖 CUDA；graph 字段仅 opt-in） |
| 质量门 | 测试/静态检查 | 全量测试绿；`mypy --strict` 零错误；black/isort 风格（100 列）；Google docstring |
| 资源 | 本机可验证 | 全部验收在 16GB 本机完成（合成模型）；无需真实 checkpoint |

## 5. 约束与假设

**约束：**
- AGENTS.md 全部约定适用（#1 注意力上下文重算、#7→本 AR 修订、#10 无 mock
  结果、#19/#20/#23 随行为演进同步修订）；
- 前缀关系信任 branch 链语义（cache 无 token，不校验内容）——文档显式声明
  该假设；
- append-only：`T_c ≥ T_p`；parent 更长不做截断复用（Out of Scope）；
- CUDA graph 固定 shape 契约不变（变长 eager，不捕获不禁用）；
- 等长路径（`T_p == T_c`）为最高优先级不变量：任何实现不得改变等长行为。

**假设：**
- 因果模型前缀 hidden 逐位一致（causal attention 下数学事实）；非因果模型
  前缀相似度由 gate 实际判定，折叠决策保守正确；
- 真实投机解码 draft 接受长度分布下前缀折叠收益为正（绝对墙钟收益留目标机
  按 RERUN_CHECKLIST 测量，本 AR 交付机制 + 合成代理证据）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 前缀折叠 | parent `[B, T_p]` 与 child `[B, T_c]`（`T_c ≥ T_p`）共享前缀时，前缀位置走相似度门控复用、后缀位置全量重算的折叠语义 |
| append-only | 仅支持 child 为 parent 的前缀扩展；不支持截断/多祖先 |
| 后缀恒 divergent | `T_c - T_p` 个后缀位置的 stable mask 恒为 False（无 parent 激活可复用） |
| 参考合成语义 | 测试参考实现：`mask = concat(gate(child_prefix, parent), False_suffix)`；`out = merge(parent_ffn_prefix_aligned, child_recompute, mask)` |
