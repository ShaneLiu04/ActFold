# [AR029] 需求设计说明书 — 端侧有界内存金字塔激活缓存

| 字段 | 内容 |
|------|------|
| AR 编号 | AR029 |
| AR 主题 | edge-pyramid-cache（端侧长程 Agent 的有界内存金字塔激活缓存） |
| 关联 SR | 无（延续 AR002/AR015/AR020/AR025 折叠缓存线） |
| 日期 | 2026-10-19 |
| 状态 | Draft |

## 1. 背景与目标

ActFold 的 Branch Folding 依赖 parent FFN 激活缓存来复用稳定 token。当前缓存
（`ActivationCache` / `ChunkedActivationCache` / `VectorizedActivationCache`）的
容量以「每层 token 数」为单位，逐层相同，且按 `(branch, step)` 整组 LRU 淘汰。
端侧 Agent 的 RSI（recursive self-improvement）循环会在同一长上下文上反复提出
候选修改：分支数与步数随会话增长，缓存占用随层数×组数线性增长，而端侧显存/
内存是固定预算。现有实现缺少：

1. **逐层预算**：浅层与深层对 parent 激活的复用价值不同，却共享同一预算；
2. **块级分页与确定性保留集**：整组淘汰粒度粗，无法在固定预算内保留「锚点 +
   近期窗口」这类对长上下文最关键的激活；
3. **计划/执行分离（plan/run）**：保留策略在每次写入时临时决策，无法在模型加载
   前给出可审计的内存上界；
4. **缺失 token 的安全语义**：若缓存返回零值，余弦门在负阈值/零向量边界可能误判
   稳定并静默复用错误激活。

**目标：** 为端侧长程 Agent 提供一种**有界内存、逐层金字塔预算、块级分页、
plan/run 分离**的激活缓存；缺失 token 以 NaN 哨兵强制走重算路径，保证「宁可
多算，不可错用」。机制与 ActFold 现有折叠栈零侵入集成（实现缓存协议即可），
真实端侧模型实验由服务器按 RERUN 预承诺执行。

**参考思想（只吸收机制，不复制代码/不引用未复现数字）：**

- **PyramidInfer**：逐层 KV 保留呈金字塔（浅层多、深层少），以少量关键 token
  维持质量——本 AR 将其迁移到折叠的 *parent 激活缓存*，并给出确定性预算公式；
- **FlashInfer**：plan/run 两阶段元数据、分页 KV 与可组合掩码——本 AR 采用
  `PyramidPlan`（加载前规划）+ 块页存储（block-paged）+ 显式缺失语义；
- **RWKV 类线性/循环结构**：推理状态具有常数级内存上界——本 AR 以此为端侧
  动机，用「固定预算 + 锚点/近期窗口」实现与上下文长度无关的缓存上界。

## 2. 需求范围

**In Scope：**
- 新增 `actfold/core/pyramid_plan.py`：`PyramidPlan` 与 `build_pyramid_plan`；
- 新增 `actfold/core/pyramid_cache.py`：`PyramidActivationCache`（块级分页、
  逐层预算、确定性保留集、NaN 缺失语义、真实内存统计）；
- `cache_factory` 与 `ActFoldConfig` 接入（默认关闭，零行为变化）；
- 单元/集成测试（CPU，真实张量运算，不启动真实 checkpoint）；
- 机制基准脚本 `scripts/ar029_pyramid_cache_bench.py`（真实小模型前向 + 折叠，
  冻结判定，quick 面）；
- 文档：README 复现/创新段、RERUN 服务器配置与命令、AGENTS 运维条目。

**Out of Scope：**
- 不修改任何现有 bench 判定函数与工件 schema；
- 不实现新的注意力 kernel / CUDA 扩展（块页与保留策略均为 PyTorch 张量操作）；
- 不实现 RWKV 模型本体，仅借鉴常数内存思想；
- 不在本机执行正式真实 checkpoint 大实验（由服务器按 RERUN 复现）；
- 不改变折叠判定语义（tau、gate、acceptance 均不变）。

## 3. 功能需求

### 3.1 FR1 — 金字塔预算计划（plan）

**描述：** 模型加载前，根据深度、总预算与策略生成逐层预算，作为运行期唯一
预算真源。

**触发条件：** 构造 `PyramidActivationCache` 或 `build_pyramid_plan(...)`。

**期望行为：**
- `build_pyramid_plan(depth, total_tokens, block_size=16, max_ratio=1.0,
  min_ratio=0.25, schedule="linear"|"geometric", recent_blocks=2,
  anchor_blocks=1, anchor_stride=2)` 返回不可变 `PyramidPlan`；
- **`total_tokens` 是计划级总预算（所有层之和）**。先算未归一权重
  `w_l`（linear：`max_ratio → min_ratio` 等距；geometric：等比），再按
  `w_l / Σw` 归一；逐层预算 = `floor(total_tokens × w_l / Σw / block_size)
  × block_size`，最低 1 个 block；若归一后总和仍超过 `total_tokens`，
  从最深层起逐块回收直到满足（硬上界优先）；
- `depth==1` 时唯一层预算 = `floor(total_tokens / block_size) × block_size`
  （至少 1 block），不做向上取整；
- 校验 `0 < min_ratio <= max_ratio <= 1.0`；`depth<=0`、
  `depth × block_size > total_tokens`（无法给每层至少 1 块）、
  `total_tokens<block_size`、`block_size<=0`、recent/anchor 负数、stride<=0
  → `ValueError`；
- `PyramidPlan` 为 frozen dataclass，提供 `budget_tokens(layer)`、
  `budget_blocks(layer)`、`total_budget_tokens`、
  `retained_blocks(layer, seq_len)`（返回该层应长期保留的块索引集合）。

**异常处理：** 非法参数立即报错，不构造半有效计划。

**验收标准：**
- Given depth=8、total=4096、block=16，When build，Then 每层预算为 16 的整数倍，
  浅层预算 ≥ 深层预算，**所有层总和 ≤ 4096**。
- Given depth=1，Then 唯一层预算 = `floor(total/block)*block`。
- Given min_ratio > max_ratio 或 max_ratio > 1.0，Then `ValueError`。
- Given `depth=8,block=16,total=127`（< 8×16），Then `ValueError`（无法满足硬上界）。
- Given schedule 不在集合内，Then `ValueError`。
- Given 同一参数两次构建，Then 结果完全相等且不可变。

### 3.2 FR2 — 块级分页与逐层预算缓存

**描述：** `PyramidActivationCache` 实现现有缓存协议，以 block 为存储与淘汰单元。

**触发条件：** 折叠栈调用 `put` / `put_batch` / `fetch` / `get_all` / `get` /
`fetch_masked` / `contains` / `clear_*` / `num_entries`。

**期望行为：**
- 写入按 `block_size` 切块，块内保留 `[tokens, batch, ...]` 的连续张量；
- 每层最多保留 `plan.budget_tokens(layer)` 个 token（块边界向下取整，
  最后一块按实际长度计）；
- 保留集（deterministic）：刚写入组的最后 `recent_blocks` 块 + 每层全局锚点
  （前 `anchor_blocks` 块 + 每 `anchor_stride` 块取 1 块）；
- 超预算时淘汰「非保留集中最旧」的块；若仅剩保留块仍超预算，继续按最旧淘汰
  （硬上界优先，保留策略是启发式而非正确性依赖）；
- `stats()` 返回 `{"tokens": {layer: int}, "blocks": {layer: int},
  "bytes": int, "evictions": int, "missing_tokens": int, "requests": int}`；
- `bytes_used` 为真实张量 `numel*element_size` 汇总（不含 Python 开销）。

**异常处理：** 空 `activations`、`ndim < 2`、leading shape 不一致 → `ValueError`；
未写入的查询 → `KeyError`（与 `ActivationCache` 一致）。

**验收标准：**
- Given 预算 64 tokens/层、block=16，写入 2 组各 128 tokens，Then 每层
  `tokens ≤ 64` 且 `bytes` 与保留块实际张量一致。
- Given 只写 1 组 ≤ 预算，Then 全部 token 可读回。
- Given 超预算，Then 淘汰计数单调增加，且保留的近期块可读回。
- Given 两次相同写入序列，Then `stats()` 完全相同（确定性）。
- Given block_size 不整除序列长度，Then 最后一块按实际长度处理，无越界。

### 3.3 FR3 — 缺失 token 的安全语义（可用性守卫）

**描述：** 被淘汰/未写入的 token 绝不以零值、旧值或 NaN 被当作可复用激活。
仅靠 NaN 不够：门读取 layer `L-1`，而合并复用 layer `L`，逐层预算不同会造成
「门看到可用、合并层缺失」的错配。因此缓存暴露**目标层可用性**，折叠栈在
门之后、计数与快速路径之前把它 AND 进 `stable_mask`。

**触发条件：** `fetch` / `get_all` / `get` / `fetch_masked` 访问缺失 token；
折叠层计算 `stable_mask` 后读取 parent 的 `ffn_out`。

**期望行为：**
- `fetch` / `get_all`：缺失 token 的行填充 `float('nan')`（同 dtype/device），
  存在的 token 返回真实值；返回独立可写张量（哨兵缓存的文档化例外）；
- `get` / `fetch_masked`：未被 `token_mask` 请求的位置为 0；被请求但缺失的位置
  为 NaN；
- `PyramidActivationCache.availability(branch_id, layer_idx, seq_len, step_idx=0,
  name="ffn_out") -> bool[seq_len]`：**`branch_id` 必须是即将被复用的 parent
  branch**（不是 child），`layer_idx` 是合并目标层；**折叠栈固定使用
  `step_idx=0`**（`put`/`fetch` 均省略 step，必须用同一 key）；返回该 token
  是否真实存在（含最后一块的部分长度判断）；
- `FoldedTransformerLayer` 在 `stable_mask` 最终成形后、**`forced_divergent`
  AND 之后、`GLOBAL_STABILITY_PROFILER.record` / 计数 / all-stable 快速路径
  之前**调用守卫：`stable_mask &= availability(parent_branch_id, self.layer_idx,
  seq_len, 0)`，并丢弃门融合核的计数缓冲（`stable_count_buf = None`），
  使 profiler 与后续计数都看到最终 mask；1-D（AR028 逐行）mask 不扩维；
- 守卫覆盖串行、同父批量、多父逐行三种路径；split 路径因共享 `stable_mask`
  自动继承；layer 0 的门与合并同层，同样按目标层可用性判定；
- `ManualFoldedForward(use_cuda_graph=True, cache=<金字塔>)` 构造期直接
  `ValueError`（图静态父缓冲不刷新，禁止组合）；`PyramidActivationCache`
  暴露 `graph_incompatible = True`；
- 缓存不修改调用者张量；本语义在 `SimilarityGate` 的 cosine/l2/pearson 下均测试。

**异常处理：** 无（缺失是正常路径，不抛错）；非法组合在构造期 `ValueError`。

**验收标准：**
- Given 淘汰一个块，When `fetch`，Then 对应位置 `isnan` 为真、其余位置等于写入值。
- Given 全量预算，When 任意读，Then 与 `ActivationCache` 的返回值逐位相等。
- Given 逐层预算不同且仅目标层 L 缺失某块，When 折叠，Then 该块在 `stable_mask`
  中为 False，输出等于「同层 forced_divergent」参考（逐位）。
- Given 统一预算/统一保留集（各层缺失集相同），When 淘汰一块，Then 输出与
  `forced_divergent=该缺失掩码` 的参考运行逐位一致。
- Given `tau=0.9` 与 `tau=0.0`（合法下界），Then 缺失 token 始终不稳定。
- Given 可用性守卫，When 查询 profiler 统计，Then 记录的 stable 比例等于
  守卫后的 mask（不存在未守卫的过度计数）。
- Given split 路径 + 金字塔缓存，Then 守卫同样生效且稳定行不读取缺失父值。
- Given `use_cuda_graph=True` + 金字塔缓存，Then 构造期 `ValueError`。

### 3.4 FR4 — 协议兼容与零回归

**描述：** 新缓存是现有缓存的 drop-in 替代，不改变折叠数学。

**触发条件：** 作为 `ManualFoldedForward(cache=...)` / `FoldedTransformerLayer`
的 cache 传入。

**期望行为：**
- 实现 `ActivationCacheProtocol` 的全部方法；另实现
  `MissingSentinelCapable`（`cache_protocol.py` 新增）：`missing_value`
  与 `availability(...)`。**协议文档同步修订**：零填充语义只约束不实现
  `MissingSentinelCapable` 的缓存；实现该能力的缓存，缺失位置返回
  `missing_value`，调用方必须用 `availability` 判定可复用性；
- 激活维度校验与 `ActivationCache` 一致：接受 `ndim >= 2`（2-D 也有测试）；
- 提供 `put_batch(entries)`（先整体校验、后写入；与逐条 `put` 状态等价；
  重复目标按逐条 `put` 的 flush 顺序执行，不做 foreach 融合）；
- `write_op_count` 口径与 `ActivationCache` 相同：每次 `put` 增加
  `len(activations)`；`put_batch` 按逐条 `put` 展开累计；
  `num_entries(layer)` 返回该层 **token 数**；`stats()["blocks"]` 才是块数；
- 全量预算（`per_layer_tokens=None` 或总预算足够）时，串行/批量折叠输出与
  `ActivationCache` 逐位相等（CPU 非 split 面）；
- 批量分支、单父/多父路径可用；被淘汰行自动降级为重算（不报错）；
- `write_op_count` 与 `num_entries` 语义与现有缓存一致口径（文档注明块级计数）。

**异常处理：** 协议形状/类型错误与 `ActivationCache` 一致（`ValueError`/`KeyError`）。

**验收标准：**
- Given 全量预算，When 串行折叠 3 步，Then 与 `ActivationCache` 输出逐位相等。
- Given 全量预算，When `put_batch` 与逐条 `put`，Then 状态等价（值/计数/键序）。
- Given 小预算，When 批量折叠，Then 无异常、输出有限值、缺失行走重算。
- Given `contains`/`clear_branch`/`clear_layer`/`clear_all`，Then 行为与协议一致。

### 3.5 FR5 — 可观测性与内存上界

**描述：** 端侧部署需要可审计的内存与实际复用情况。

**触发条件：** 任意写入/读取后调用 `stats()`；部署前读取 `plan`。

**期望行为：**
- `stats()` 提供逐层 tokens/blocks、总 bytes、淘汰次数、缺失请求计数；
- `PyramidPlan.total_budget_tokens` 给出与序列长度无关的硬上界；
- 基准脚本记录 `cache_bytes`、`tokens_stored`、`reuse_ratio`（稳定且命中）、
  `missing_ratio`、`wall_ms`；
- 不允许出现负值/NaN 统计。

**验收标准：**
- Given 运行 100 次写入，Then 每层 tokens 从不超过计划预算。
- Given 统计前后，Then `bytes` 等于实际保留块张量字节和。
- Given 两次相同运行，Then 统计确定性一致。

### 3.6 FR6 — 工厂与配置接入（默认关闭）

**描述：** 通过统一入口启用，不改变默认行为。

**触发条件：** `make_activation_cache(use_pyramid=True, ...)` 或
`ActFoldConfig.use_pyramid_cache=True`。

**期望行为：**
- `cache_factory` 新增 `use_pyramid`、`pyramid_plan` / `pyramid_total_tokens`、
  `pyramid_block_size`、`pyramid_max_ratio`、`pyramid_min_ratio`、
  `pyramid_schedule`、`pyramid_recent_blocks`、`pyramid_anchor_blocks`、
  `pyramid_anchor_stride`；**优先级固定为 `use_pyramid` > `use_vectorized` >
  `use_chunked`**（金字塔是预算上界承诺，不允许被其他开关静默降级）；
- `cache_factory` 的 `ActivationCacheType` 与 `actfold.core.__init__` 导出
  `PyramidActivationCache`、`PyramidPlan`、`build_pyramid_plan`；
- `ActFoldConfig` 暴露 `use_pyramid_cache` 与同名 `pyramid_*` 字段（YAML 可消费，
  未知键不再告警）；
- **`use_pyramid_cache=True` 与 `use_cuda_graph=True` 互斥**：`ManualFoldedForward`
  构造期 `ValueError`（图静态父缓冲不刷新）；文档给出替代（eager 折叠）；
- 默认 `use_pyramid_cache=False`，现有路径逐字节不变。

**异常处理：** 计划参数非法 → 构造时 `ValueError`。

**验收标准：**
- Given 默认配置，When 构建缓存，Then 返回 `ActivationCache`（类型不变）。
- Given `use_pyramid=True`，Then 返回 `PyramidActivationCache` 且预算按计划。
- Given YAML 含金字塔键，Then `load_config` 无 unknown-key 警告。
- Given `use_pyramid` 与其他缓存开关同时为真，Then 按文档优先级选择金字塔。

### 3.7 FR7 — 机制基准与真实复现协议

**描述：** 用真实张量运算验证「有界内存 + 安全重算」，并给出服务器复现协议。

**范围裁定（用户授权）：** 用户明确本阶段不做大型实验。因此本 AR 被明确分类为
**机制/基础设施 AR（非创新性能 AR）**，按 AGENTS Experimental Validation Rule
的例外处理：不声称相对官方 baseline 的质量/速度优越性，**不进入 superiority
ladder**；README 的创新表只记录机制与 quick 证据并标注 `baseline_claim=false`，
ladder 行标记 `BLOCKED-DEFERRED`（不填任何未测量数字）。该分类由用户在
AR 启动指令中显式授权（“不做所有大型实验…大的模型实验内容和配置写清楚，
方便服务器复现”）。真实 checkpoint 的正式 A/B 由服务器按 RERUN 预承诺执行；
**未来若执行正式面，必须先在 `run_formal_queue` 增加 AR029 FaceSpec/校验器**，
不得手工拼命令。

**触发条件：** `python -m scripts.ar029_pyramid_cache_bench [--quick]`。

**期望行为：**
- 构建真实的小型 Transformer 栈（实际 nn.Linear/attention/FFN 前向）与
  `ManualFoldedForward`，在同一分支序列上比较 full 与 pyramid 缓存；
- 冻结判定（脚本内预声明，单次运行）：
  - **E1 精确性**：pyramid 输出与「相同缺失行的 forced_divergent 参考」逐位一致；
    全量预算下与 full 缓存逐位一致；
  - **E2 上界**：每层缓存 tokens ≤ 计划预算；`bytes` 与统计一致；
  - **E3 有效性**：在预算压力下 `reuse_ratio > 0`（缓存确实被复用）；
  - **E4 诚实性**：任何 NaN/Inf 或负统计 → `blocked`，零伪造数字；
- `--quick` 写 `*_quick.json`，永不覆写正式工件路径；
- 工件包含真实测量值、判定、环境与配置；明确标注 `evidence_level="mechanism"`
  与 `baseline_claim=false`；
- RERUN 提供**未来**服务器真实 checkpoint 协议（命令形状、模型目录、资源表、
  判定语义、工件路径），并标注 `BLOCKED-DEFERRED`；**该协议不是已承诺的
  formal face**——未来把它变成正式面前，必须新增 `run_formal_queue` FaceSpec
  与 CPU 合同测试（见 AGENTS Formal-Run Executor Rule）。

**异常处理：** 依赖缺失/形状错误 → 明确失败并写 `blocked` 工件（不伪造）。

**验收标准：**
- Given `--quick`，When 运行，Then 生成 quick 工件，E1/E2/E3/E4 全部 supported。
- Given 篡改预算/关闭重算守卫，Then 判定 refuted（测试内验证判定函数）。
- Given RERUN 协议，Then 含环境/资源/命令/工件路径与 `BLOCKED-DEFERRED` 标记；
  工件内 `baseline_claim=false`。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 可测试性 | CPU 测试 | 新增测试无 GPU/网络/真实 checkpoint 依赖，单文件 < 60s |
| 内存 | 硬上界 | 每层 tokens ≤ 该层预算；所有层总和 ≤ `PyramidPlan.total_budget_tokens` |
| 诚实性 | 零伪造 | 缺失 token 必须不可用 + 重算，不做静默近似；工件不写未测量值、不声称 baseline 优越性 |
| 兼容性 | 零回归 | 默认关闭；全量预算下与现有缓存输出逐位一致 |
| 质量门 | lint/types | black(100)/isort/pyflakes 零；mypy actfold + scripts 零 |
| 性能 | 块级开销 | 计划构建 O(depth)；写入按块向量化，无逐 token Python 循环 |
| 可移植性 | 平台 | CPU/CUDA 均可运行；NaN 语义不依赖 Triton |

## 5. 约束与假设

**约束：**
- 不修改 folded_transformer 的折叠判定数学与现有缓存类的行为；**唯一核心改动**
  是可用性守卫（串行/同父批量/多父逐行三处）+ graph×pyramid 互斥校验。
  具体锚点：AR028 重写后的每条路径在 **pad/forced AND 之后、
  `GLOBAL_STABILITY_PROFILER.record` 与 stable 计数之前**调用命名 helper
  `_apply_cache_availability(layer, parent_branch_id, stable_mask)`（内部固定
  step 0；1-D mask 不扩维；有效长度切片统一 `[..., :v_r]`）；
  AR028 未合并时以冻结面测试钉住 AR015/AR021/AR025 原有路径零变化；合并后
  由该方法调用点与冻结面共同保证；
- 不引入新的第三方依赖；
- 新增缓存必须在 `torch.no_grad()` 推理路径下工作；
- 大实验由服务器按 RERUN 执行；本 AR 的 quick 工件仅作机制证据，明确
  `baseline_claim=false`。

**假设：**
- 端侧 Agent 的上下文/分支数会超过固定缓存预算，需要硬上界；
- 折叠栈通过 `ActivationCacheProtocol` 访问缓存；
- 缺失 token 触发重算的额外算力可接受（以内存换精确性，报告真实权衡）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 金字塔预算 | 逐层递减的 token 保留预算（浅层多、深层少） |
| 块页（block page） | 固定 `block_size` 个 token 的连续激活存储单元 |
| 保留集 | 锚点块 + 近期窗口块；预算内优先保留 |
| NaN 哨兵 | 缺失 token 的返回值，使相似度门必判不稳定 → 重算 |
| plan/run | 预计算逐层预算与保留布局（plan）与运行期写入/淘汰（run）分离 |
| 可用性守卫 | 目标层 `ffn_out` 是否真实存在；折叠层将其 AND 进 stable_mask |
| MissingSentinelCapable | 显式能力协议：缺失值语义 + `availability` 查询 |
| RSI | recursive self-improvement，端侧 Agent 反复生成/验证候选修改的循环 |

## 7. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| draft-1 | 2026-10-19 | 初稿（FR1-FR7） |
