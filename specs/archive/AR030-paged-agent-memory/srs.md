# [AR030] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR030 |
| AR 主题 | 面向端侧 Agent/RSI 的页共享折叠内存（Page-Shared Fold Memory, PSFM） |
| 关联 SR | 无（延续 AR029 端侧有界内存主线） |
| 日期 | 2026-10-20 |
| 状态 | Draft（req 门控第 6 轮 PASS；design 阶段 draft-12 统一 restore epoch 与边界口径） |

## 1. 背景与目标

AR029 已把「父激活留多少」变成逐层可规划的块页预算，并用 `availability` 守卫保证
缺失行必然重算。但端侧 Agent / RSI（递归自我改进）工作负载的真实形态不是「一条链」，
而是**同一份长上下文上反复分叉出 N 个候选计划/自我编辑**：共享前缀占激活体积的
绝大多数，候选之间只差少量 token。现有缓存（含 AR029 金字塔）在每个分支上都**复制**
父激活的页，分叉成本与分支数线性增长；上下文切回旧会话时只能整段重算。这使端侧
Agent 的「多候选探索 + 会话恢复」在内存与延迟上都不可持续。

**目标**：在 AR029 的逐层预算与缺失哨兵之上，引入**页表寻址 + 写时复制（COW）共享**
的折叠内存，使分支分叉 O(1)（只复制页表元数据）、共享页只存一份、逐层预算与淘汰
在共享语义下仍然正确；再加**host 侧 plan / device 侧 run 拆分**与**精确冷热分层**，
让端侧 Agent 能在固定内存上做多分支 rollout、跨会话恢复，并保持与全量缓存**逐位
一致**。所有机制真实可运行、无 mock；大型真实 checkpoint 实验按用户范围裁定不做。

### 1.1 参考项目深度分析（只吸收机制，不复制代码/不引用未复现数字）

| 来源 | 核心思想 | 对本项目的可用价值 | 取舍 |
|------|----------|--------------------|------|
| **FlashInfer**（paged KV / plan-run / 可组合 attention） | 逻辑位置经**页表**映射到物理页；CPU 先做 plan，GPU 只执行不可变计划；不同后端共享同一页表语义；块稀疏通过页索引表达 | ①页表天然支持**多序列共享物理页**；②plan/run 分离可把分叉、预算、淘汰、可用性全部在 host 侧冻结，device 侧零回读；③块稀疏页索引与折叠的 stable/divergent 行天然同构 | 采用页表 + plan/run + 共享引用计数；**不**引入 JIT/Triton 内核（本 AR 为架构/内存机制，CPU 可验证） |
| **PyramidInfer**（pyramid KV / pivot + recent window） | 浅层保留更多 token、深层更少；pivot（注意力汇）与最近窗口优先保留；固定预算下长上下文可推理 | AR029 已有逐层预算与锚点；本 AR 把预算落到**共享页**上：pivot/anchor 页跨分支钉住，淘汰按物理页回收 | 采用逐层预算 + pivot/recent 保护；**不**采用「按注意力分数动态丢 token」的近似（会破坏折叠逐位正确性） |
| **RWKV**（线性注意力/状态空间、常量内存、状态可复用） | 推理状态大小与序列长度无关；跨 token/跨会话可携带紧凑状态；避免 KV 线性增长 | 提供「端侧会话应有常量内存、可快速恢复」的目标函数与评估口径（hot bytes 上界、恢复成本） | **明确不采用**近似 recurrent state 替代精确激活：折叠复用要求逐位一致，近似状态只能做提示不能做复用。对应实现为**精确冷热分层 + 锚点快照**，并在文档中说明差异 |
| **端侧 Agent / RSI 架构** | 同一 base context 上并行评估多个计划/自改候选；会话频繁切换；内存/能耗是硬约束；失败必须可回退 | 分支树（`BranchTree`）与候选 rollout 是天然的多分支场景；COW 使「候选只写差异页」；plan/run 与可用性守卫使候选隔离可验证 | 本 AR 提供缓存与执行计划机制，不实现上层 agent 策略/工具调用 |

**结论**：把三者交集收敛为一个可验证的架构创新——**页表寻址的 COW 折叠内存 +
逐层共享预算 + plan/run 冻结执行 + 精确冷热分层**。它直接服务端侧 Agent/RSI 的
多候选与会话恢复，同时保持 ActFold 的「零语义变化」底线。

### 1.2 与 AR029 / AR028 的关系（边界冻结）

- **复用 AR029**：`PyramidPlan` 的逐层 token 预算与确定性锚点直接作为 paged 预算来源；
  参数映射：`paged_page_tokens ↔ AR029 block_size`、`paged_total_tokens ↔ total_tokens`、
  `paged_recent_pages ↔ recent_blocks`、`paged_anchor_pages ↔ anchor_blocks`、
  `paged_anchor_stride ↔ anchor_stride`。`MissingSentinelCapable` 与
  `_apply_cache_availability` 守卫保持不变（paged `availability` 只反映 hot 页）。
- **物理层替换**：AR029 块页每分支独占；AR030 用页表把逻辑页映射到可共享物理页，
  写入时 COW。折叠数学、门控、forced 顺序不变。
- **优先级**：`use_paged > use_pyramid > use_vectorized > use_chunked`
  （paged 内部使用 pyramid 预算；同开时 paged 胜出）。
- **互斥**：paged × CUDA graph 禁止；`PagedFoldCache.graph_incompatible=True`；
  工厂、runner、MFF setter 三处显式拒绝。
- **AR028（并行会话）**：本 AR 不修改 `folded_transformer.py` 折叠逻辑；如需 fork
  钩子只加独立 helper。AR028 合入后 rebase 并跑其冻结面（UT-2804/2805/2814/2816）
  + AR015/AR021/AR025 代表用例（T014）。

## 2. 需求范围

**In Scope：**
- `actfold/core/page_table.py`、`actfold/core/paged_fold_cache.py`、
  `actfold/core/fold_plan.py`。
- 工厂/配置/导出/runner/MFF setter 接线（见 FR7）；`architecture_utils.py` 仅泛化
  graph 拒绝文案（保留 `CUDA-graph` 子串，行为不变）。
- `scripts/ar030_paged_agent_bench.py`（quick-only 机制基准）。
- 测试与文档（README/RERUN/AGENTS/GUIDE/CHANGELOG/ALGORITHM/EXPERIMENTS）。

**Out of Scope：**
- 真实 checkpoint formal（`BLOCKED-DEFERRED`；未来须先加 `run_formal_queue` FaceSpec）。
- CUDA/Triton 专用内核、多 GPU/分布式、磁盘持久化。
- 近似/有损复用（量化页、RWKV 式 recurrent 状态替代精确激活）。
- 上层 Agent 策略、工具调用、自我改进训练算法。
- CUDA graph × paged。
- **cold 隐式提升**：cold 页在 plan/run 执行期一律不可用；仅 host 侧显式 `promote`
  （执行窗外）可恢复，且必须重新 plan。

## 3. 功能需求

### 3.1 FR1 — 页表与写时复制分支分叉

**描述：** 激活按固定 token 页存储；分支通过复制父页表元数据实现 O(1) 分叉，写入
divergent 页时才分配新物理页（COW）。

**触发条件：** `fork_branch(parent_id, child_id, layers=None)`；向已 fork 子分支 `put`。

**期望行为：**
- `layers=None` 复制父分支所有已存在层；显式列表只复制这些层。
- **step 冻结**：`PagedFoldCache` 只支持 `step_idx=0`（折叠固定 step 0）；任何
  `put/put_batch/fetch/get/availability` 的非零 step → `ValueError`。
- fork 不复制页数据；共享页 refcount +1。
- **写路径 COW 粒度冻结**：`put`/`rollout_put` 接收完整 `[B,T,...]` 激活，但按逻辑页
  处理：逐页与当前映射页做**逐位内容比较**（`torch.equal`）——相同则保留共享页
  （不复制、不分配），不同才分配新物理页（旧页 refcount -1）。独占页原地写。
  因此「只改 8 个 token」只会在受影响页上分配新页。父/兄弟已见内容不变。
- 每页记录 `page_len`（末页有效 token 数）；分支/层记录 `seq_len`，`availability`
  由 hot 页 + `page_len` 精确到 token。**seq_len 变化**：同一分支/层重写时先解除
  旧映射（释放引用），缩短时丢弃超出范围的逻辑页，增长时新页走 COW；batch 由
  激活张量 `shape[0]` 决定且必须与已登记 batch 一致，不一致 → `ValueError`。
- `drop_branch`/`clear_branch` 释放引用；**reclaim 仅允许 refcount==0**，且改为
  **立即释放**（refcount 归零即删除页实体，不在 cold 保留零引用副本）。
- `demote(page_id: int)` 释放 hot 驻留、保留页 id/refcount 与 cold 精确副本；
  **demote 不是 reclaim**，可在任意 refcount 下发生。
- 不变量（无豁免）：①页表引用的页在 hot 或 cold 中存在；②refcount == 实际引用
  条目数；③refcount=0 的页不在任何页表；④每层 hot 物理页数 ≤ 预算页数；
  ⑤`hot_bytes == Σ(hot 页张量字节)`、`total_bytes(hot+cold) == 物理页字节和`。
- `page_stats()` 键：`branches/physical_pages/shared_pages/hot_pages/cold_pages/
  hot_bytes/cold_bytes/metadata_entries/refcount_leaks`。

**异常处理：** 重复 id、未知父、层越界、页大小不一致、同一次写入各 name 的
`shape[1]` 不一致 → `ValueError`；非法 fork/put 原子失败（页表/refcount/bytes 与失败前全等）。

**验收标准：**
- Given 父分支有完整页表，When fork N 次，Then 激活页字节数不变；每个子分支
  元数据条目 == Σ_layers(该层页数)，共享页 refcount == N+1。
- Given 子分支只改一页，Then 仅新增 1 页、父页 `torch.equal` 不变、其余仍共享。
- Given 非法 fork/put（含非零 step），When 抛 `ValueError`，Then 父/兄弟页表、
  refcount、bytes 全等。
- Given 删除全部 fork 子分支，Then 共享页 refcount 回到 1，`refcount_leaks==0`。

### 3.2 FR2 — 共享页上的逐层预算与淘汰

**描述：** 预算按**物理页**计（共享页只计一份）；锚点/recent 优先，但硬上界无例外。

**触发条件：** `put`/`put_batch` 后；显式 `evict(page_id: int) -> bool`。

**期望行为：**
- `budget_pages(layer) = max(1, plan.budget_tokens(layer) // page_tokens)`。
- 每层 hot 物理页数必须 ≤ `budget_pages(layer)`（无豁免）。
- `plan_expected_bytes` 精确定义：对每层、每个**已登记** name，按
  `page_tokens × batch × prod(shape[2:]) × element_size(name)` 计整页字节，再乘
  `budget_pages(layer)`；未写入层/name 计 0；末页按整页计（上界）。
- 淘汰键冻结：`(protected_any, refcount, write_seq, page_id)` 升序。
  `protected_any` = 页在**任一**引用分支中为锚点或 recent；`write_seq` 为首次写入
  全局序号（demote/promote 不改变）；page_id 升序 tie-break。
- 每淘汰一页立即 demote，重复直到 hot 页数 ≤ 预算；demote 后该逻辑页在**所有**
  引用分支 `availability=False`。
- `evict(page_id)` 是显式入口：该页若处于保护集（锚点/recent）、不存在或**非 hot**
  → 返回 `False` 且不改变状态；否则 demote 并返回 `True`。显式 evict 不计入
  `evictions`（只计入 `demotions`）；预算触发的 demote 计入两者。活跃 rollout 期
  被 base/候选引用的页同样不可显式 evict/demote（`False`/`ValueError`）。
- **rollout 隔离**：活跃 rollout 期内对任意分支的写，base 与全部活跃候选引用的页
  为硬不可淘汰（`unmovable`）；可淘汰页不足以回到预算时，本次写以
  `RolloutBudgetError(ValueError)` 原子失败（其他分支/base availability 不变），
  成功写仍满足 hot ≤ 预算（hard bound 永不突破）。
- 淘汰序列可由 `eviction_trace()` 确定性重放：同一状态重建两次 → 受害者序列逐项同。

**异常处理：** 预算 < 1 页、深度/页大小非法 → `ValueError`；demote 失败回滚并抛错。

**验收标准：**
- Given 两分支共享前缀页且共享页未被选中，Then 共享页只占一份字节，两分支
  availability 同时 True。
- Given 共享页使 hot 页超预算，Then 逐页 demote 至 ≤ 预算，所有引用分支 tail
  同时不可用。
- Given 同一状态重建两次，When `eviction_trace()`，Then 序列逐项相同。
- Given 任一淘汰，Then hot 页数 ≤ 预算、hot bytes 降、cold bytes 等量升、总字节不变、
  不变量全 PASS。

### 3.3 FR3 — Host 侧计划 / Device 侧执行窗口（冻结契约）

**描述：** host 生成不可变 `FoldPagePlan`；执行通过 `enter_plan(plan, ...)` 进入
**只读窗口 + 暂存写**，窗口内父读取完全由计划决定（零控制流回读），子激活写入先
暂存，退出窗口时原子提交并只递增一次 epoch。

**触发条件：**
`plan = cache.build_plan(branch_ids, layers=None, seq_len=...)`；
`with cache.enter_plan(plan, allow_plan_fallback=False): ...`。

**期望行为：**
- `layers=None` = 全层；计划 frozen，含 `plan_id`/`epoch`/`to_dict()`/`from_dict()`。
- **epoch 穷举表**（仅成功操作递增；失败操作不递增；`snapshot/build_plan` 不递增）：

  | 操作 | 增量 |
  |------|------|
  | `put` / `rollout_put` | +1 |
  | `put_batch` | +1（不是 +N） |
  | `evict` / `demote` / `promote` | +1 |
  | `fork_branch` / `drop_branch` / `clear_branch` / `clear_all` | +1 |
  | `fork_rollouts` | +N（N=候选数，等价 N 次 fork） |
  | `rollback_candidate` / `close_candidate` | +1（每次调用） |
  | `close_rollout` | +1（原子关闭全部候选） |
  | `enter_plan` 退出提交（有暂存写时） | +1 |
  | `restore`（无分支/无活跃 plan/rollout） | `max(当前 epoch, 快照 epoch) + 1` |
  | `snapshot` / `build_plan` / `sync_branch_tree`（只校验） | 0 |
  | 失败操作 + 自动回滚 | 0（回滚同时恢复操作前 epoch） |

  `put` 内部的预算淘汰/demote 属于该 `put` 的一部分，只计 `put` 的 +1，不额外叠加；
  `put_batch` 同理只 +1。**窗口注记**：上表的 `put/put_batch/rollout_put` 增量仅
  适用于窗口外调用；窗口内暂存写不单独计 epoch，退出提交时合并为 `enter_plan`
  的 +1（无论暂存条数），提交失败为 0。

- `enter_plan` 入口检查 `cache.epoch == plan.epoch`，不等则默认抛 `PlanStaleError`；
  仅 `allow_plan_fallback=True` 时重新 `build_plan`，记录 `plan_fallback_count` 与 reason。
- **窗口两阶段（冻结）**：
  1. **只读期**（进入 `with` 到 forward 结束）：`fetch/get_all/get/fetch_masked/
     availability` 只读计划视图；`put/put_batch/rollout_put` 进入暂存缓冲，不改变
     页表/驻留/epoch；任何其它 mutator（`fork_*/drop_*/evict/demote/promote/
     clear_*/restore/rollback_candidate/close_candidate/close_rollout/snapshot`）
     → `RuntimeError`；嵌套 `enter_plan` → `RuntimeError`。只读内省
     （`stats/page_stats/eviction_trace/sync_branch_tree/build_plan/assert_invariants/rollout_stats`）
     允许；`record_rollout_masks` 窗口内 → `RuntimeError`（窗口退出后补记）。
  2. **提交期**（退出 `with`）：原子提交暂存写。允许对暂存页做内容比较（COW 去重）
     与页分配；有暂存写时 `epoch += 1`；提交失败回滚全部暂存、epoch 不变。
- 计划构建只读元数据（页表、名称/形状/字节登记），不读页内容、不做张量归约；
  位图用整数列表/字节串。
- **只读期零控制流回读**：`.item()`/`.tolist()`/`.cpu()`/`__int__`/`__float__`
  调用数 == 0。提交期允许
  `torch.equal`（COW 去重）与页分配，但禁止把比较结果用于打分/门控等 semantic 决策。
- 失败自动回滚（FR5）只允许在窗口外执行；只读期内候选失败仅丢弃暂存写。

**异常处理：** 计划过期 → `PlanStaleError`（默认）；窗口内非法 mutator →
`RuntimeError`；形状/层缺失 → `ValueError`；提交失败回滚且 epoch 不变。

**验收标准：**
- Given 同状态，When 计划 + 窗口执行，Then 计划 availability 与 `cache.availability`
  全等，输出与无计划路径逐位一致；只读期 `.item()/.tolist()/.cpu()/__int__/__float__` 计数均为 0。
- Given 计划后缓存被成功修改，When `enter_plan`，Then 默认 `PlanStaleError`；
  `allow_plan_fallback=True` 时重 plan、计数 +1、输出正确。
- Given 窗口内调用 mutator，Then `RuntimeError` 且状态不变。
- Given 暂存写提交失败，Then 全部回滚、epoch 不变；页表/refcount/tier/内容/
  availability 与窗口前全等（事件计数保留已发生事件，失败 trace 条目不参与等值）。
- Given 只读期调用 `record_rollout_masks`，Then `RuntimeError`（需窗口退出后补记）。
- Given `to_dict()` → `from_dict()`，Then 映射/位图/字节账本/epoch 全等。

### 3.4 FR4 — 精确冷热分层与会话恢复

**描述：** hot 放驻留页，cold 放精确副本；cold 在折叠执行期不可用，仅显式 promote
后可用。

**期望行为：**
- `demote(page_id)/promote(page_id)`：精确复制、释放/恢复 hot；往返 `torch.equal`；
  保留页 id/refcount/write_seq；成功操作递增 epoch（失败不递增）。
- `promote` 先将目标页计入 hot，再以目标页硬钉住执行逐层预算：成功后
  `promotions += 1` 且 `hot_pages <= budget_pages`；非 rollout 期必成功；活跃
  rollout 期不可淘汰页不足 → 恢复原 cold/tier/data、不计数、抛
  `RolloutBudgetError`。cold 页 `availability=False`、`fetch` 返回 NaN；
  折叠强制重算；**禁止隐式 promote**。
- **snapshot schema（冻结）**：
  `{version, epoch, plan{depth,page_tokens,total_tokens,max_ratio,min_ratio,schedule,
  recent_pages,anchor_pages,anchor_stride,layer_budgets}, next_page_id, next_write_seq,
  pages{id:{tier,refcount,write_seq,page_len,
  names{name:{shape,dtype,bytes,sha256}},data}},
  entries{branch:{layer:{seq_len, logical_page:page_len}}},
  tables{branch:{layer:{logical_page:page_id}}}, checksum_alg:"sha256"}`。
  `data` 为张量副本；`sha256` 对每页所有 name 的连续字节按 name 排序拼接后计算。
- `snapshot()` 不修改缓存、不递增 epoch；`restore()` 要求**无任何分支/无活跃
  plan/rollout 的目标实例**（允许 epoch>0），先全量校验（schema/长度/sha256/
  page_id 唯一/refcount/不变量）后一次性提交，`epoch = max(当前, 快照) + 1`；
  tier 与 availability 原样恢复，保留目标实例事件计数。
- `stats()` 冻结 **11 键**：
  `hot_pages/cold_pages/shared_pages/hot_bytes/cold_bytes/evictions/demotions/
  promotions/refcount_leaks/requests/missing_tokens`。
  定义：`evictions` = 预算触发的 demote 次数；`demotions` = 全部 demote（含显式与预算）；
  `promotions` = **成功** promote 次数（失败恢复 tier/data 且不计数）；
  三者单调递增；`restore` 保留目标实例计数。
  `requests` = 成功 `fetch`/`fetch_masked` 调用次数；`fetch` 的 `missing_tokens`
  = `(seq_len - present_tokens) × batch`；`fetch_masked` 在**只读计划窗口外**
  按 `(mask & ~availability).sum()` 统计，窗口内只 `requests += 1`、`missing_tokens`
  不累计（零 device→host 同步豁免，窗口结束后不追溯）。
- 页丢失/损坏必须抛错，绝不静默返回零/NaN 当成功。

**异常处理：** 页不存在/重复 demote/已 hot 页 promote/快照损坏 → `ValueError`；restore 失败不部分应用。

**验收标准：**
- Given demote 的页，When promote 后 fetch，Then 与 demote 前逐位一致。
- Given cold 页，When fetch/折叠，Then availability False、fetch NaN、折叠输出等于
  重算参考；显式 promote 后与未分层路径逐位一致。
- Given 快照（hot+cold），When 新实例 restore，Then tier/availability（含末页
  page_len）/页内容逐位相同；promote 全部 cold 后输出与未快照路径逐位一致。
- Given 损坏快照（sha256/长度不符），Then `ValueError` 且目标实例状态不变。
- Given quick 几何（计时小缓存每层物理页 ≤ 预算），warmup 后 5 次中位数
  `restore + promote 全部 cold 页` < 50 ms。

### 3.5 FR5 — 端侧 Agent/RSI rollout 集成与可靠性

**描述：** 同一 base context 上 fork N 个候选，逐候选折叠，验证隔离/共享/回滚。

**API（冻结）：**
- `fork_rollouts(base_id: str, candidate_ids: Sequence[str]) -> RolloutSet`；
  `RolloutSet` 字段：`base_id`、`candidate_ids`（只读元组）、`closed`（bool）；
  缓存注册时保存 canonical 副本作为所有权判定依据（外部改写句柄字段不影响判定）。
- `rollout_put(rollout: RolloutSet, candidate_id: str, layer_idx: int,
  activations: dict[str, torch.Tensor], step_idx: int = 0) -> None`。
- `rollback_candidate(rollout, candidate_id) -> None`：恢复到该候选 fork 时状态。
- `close_candidate(rollout, candidate_id) -> None`：回滚并释放该候选引用（其余候
  选不受影响）。
- `close_rollout(rollout) -> None`：原子关闭全部候选（等价逐个 `close_candidate`，
  但 epoch 只 +1），释放 base 冻结；之后对 `rollout` 的任何写 → `ValueError`。
- `rollout_stats(rollout) -> dict`（不另设 `RolloutSet.stats()`）；字段：
  `hot_bytes/cold_bytes/shared_bytes/reused_tokens/valid_tokens/reuse_ratio/
  recompute_ratio/failures`。
- `record_rollout_masks(rollout, candidate_id, valid_mask, reused_mask) -> None`：
  显式统计入口（design 阶段补充，draft-7）；`valid_mask/reused_mask` 为真实
  guarded 路径的 bool 掩码（`[batch,T]` 或 `[T]`），`reused_mask ⊆ valid_mask`；
  只累加 `valid_tokens/reused_tokens`，不改页表/可用性/epoch（0）。
- `sync_branch_tree(tree) -> None`：校验树节点与缓存分支一致；树有节点/缓存无分支
  或反之 → `ValueError`（只校验，不改状态，不递增 epoch）。

**期望行为：**
- rollout 存续期间 base 冻结：对 base 的任何写（`put`/`put_batch`/`rollout_put`/
  `clear`/`drop`/`fork`）→ `ValueError`；`clear_all`/`restore` 等全局替换同样
  `ValueError`；只允许经 `rollout_put` 或指向候选的 `put`/`put_batch` 写候选自身
  （窗口内暂存，提交时统一生效）。
- 候选写入互不可见；base 保持原值。
- 候选事务：fork 时记录候选页表与 refcount 基线。**失败语义冻结为三类**：
  1) 窗口外单次 `rollout_put` 失败 → 只回滚该次操作（epoch 不变），此前成功的
  候选写保留，`failures += 1`；2) **显式** `rollback_candidate` → 恢复到该候选
  fork 时状态（epoch +1）；3) 只读窗口退出提交时任一条暂存写失败 → 整窗回滚
  （所有候选回到窗口前状态，epoch 不变）；窗口内暂存只读校验失败 → 仅丢弃该候选
  暂存。候选级隔离只在窗口外单次写与显式回滚层面保证。窗口内 `rollout_put`
  的只读校验失败计入 `failures`（不改变页表/epoch）。
- 与 `availability` 守卫协同：cold/缺失页行必然重算。
- `rollout_stats(rollout)`：每候选 `hot_bytes/cold_bytes/shared_bytes/
  reused_tokens/valid_tokens/recompute_ratio/failures`；
  `reuse_ratio = reused_tokens / max(valid_tokens,1)`（来自 guarded stable_mask 统计），
  `recompute_ratio = 1 - reuse_ratio`。

**异常处理：** 候选 id 冲突、父缺失、写 base/跨候选 → `ValueError`；预算不足
（共享页不可淘汰）→ `RolloutBudgetError(ValueError)`；失败按三类语义原子回滚；不得全局清空。

**验收标准：**
- Given base + 4 候选各改 8 token，When 逐候选折叠，Then 每候选输出与独立全量
  缓存逐位一致，共享前缀页只占一份 hot 字节。
- Given 用 `close_candidate` 删除 2 候选，Then 剩余候选 availability/输出不变、
  refcount 正确、`sync_branch_tree` 通过；再 `close_rollout` 原子释放其余候选。
- Given 窗口外一次候选写失败，Then 仅该次操作回滚（epoch 不变），此前成功写保留，
  其余候选/base availability/输出逐位不变。
- Given **显式** `rollback_candidate`，Then 该候选页表/availability 回到 fork 时
  状态（epoch +1），其余候选/base 不变。
- Given 只读窗口提交时任一条暂存写失败，Then 整窗回滚（所有候选回窗口前，epoch
  不变）；窗口内只读校验失败仅丢该候选暂存。
- Given rollout 存续时写 base 或跨候选写，Then `ValueError` 且状态不变。
- Given 树与缓存不一致，When `sync_branch_tree`，Then `ValueError`。
- Given `rollout_stats`，Then 所有比率 ∈[0,1]、计数器非负，且
  `reuse_ratio == reused_tokens/max(valid_tokens,1)`（guarded mask 统计，经
  窗口退出后的 `record_rollout_masks` 计入）可直接复算。

### 3.6 FR6 — 机制基准与复现协议

**几何冻结**：`T_base=128`、`page_tokens=16`、`depth=2`、`hidden=32`、`candidates=8`、
每候选改最后 8 token；`paged_total_tokens=512`、`max_ratio=min_ratio=1.0`
（depth=2 → 每层 16 页）；CPU、seed 固定、float32。预算足以容纳 base 8 页 + 8 个
差异页/层，故无预算淘汰，共享收益可直接测量。

**触发：** `python -m scripts.ar030_paged_agent_bench --quick`。

**冻结判定：**
- **E1 exactness**：每个候选输出 vs 独立 `ActivationCache` 全量路径 `torch.equal`。
- **E2 sharing（按物理页数判定，避免预算压缩混淆）**：
  - `paged_physical_pages = Σ_layer 物理页数`（hot+cold，去重）；
  - `nonshared_physical_pages = branches × Σ_layer ceil(seq_len/page_tokens)`，
    由独立 `ActivationCache` 每分支实际条目复算；
  - `sharing_ratio = 1 - paged_physical_pages / nonshared_physical_pages`；
  - `paged_hot_bytes = stats()["hot_bytes"]`；
  - `nonshared_hot_bytes` = 独立 `ActivationCache` 实际张量字节和；
  - supported：`sharing_ratio ≥ 0.5` 且 `paged_hot_bytes ≤ plan_expected_bytes`
    且 `paged_hot_bytes ≤ nonshared_hot_bytes`，且独立逐页字节和 ==
    `stats()["hot_bytes"] + stats()["cold_bytes"]`。
- **E3 fork cost**：fork 激活页复制字节 == 0；每子分支元数据条目 ==
  Σ_layers(该层页数)；写差异页后新增物理页数 == 差异页数。
- **E4 reliability**（全 PASS）：refcount 不变量、snapshot/restore tier+availability
  全等、demote/promote 往返、plan/executor availability 全等、**stale 检测生效**、
  失败原子回滚。任何非有限值/不变量破坏 → `blocked`。优先级 blocked > supported > refuted。
- `--quick` 写 `results/optimization/ar030_paged_agent_bench_quick.json`；无 `--quick`
  打印 BLOCKED-DEFERRED、exit 3、不写正式路径。
- mutation：关闭 COW 去重（每页强制分配）→ `paged_physical_pages ==
  nonshared_physical_pages`、`sharing_ratio == 0` → E2 refuted；破坏 refcount →
  E4 blocked；跳过 stale 校验 → E4 blocked。

**验收标准：**
- Given quick，Then E1–E4 全 supported，数字可由工件重算。
- Given 关闭 COW 对照，Then `sharing_ratio < 0.5`、E2 false、overall refuted。
- Given 无 `--quick`，Then exit 3 且正式工件不存在。

### 3.7 FR7 — 工厂/配置/接线（默认关闭零回归）

**期望行为：**
- 优先级 `use_paged > use_pyramid > use_vectorized > use_chunked`；paged 内部用
  `PyramidPlan`。
- `paged_depth >0` 显式；`0` 自动解析 `num_layers`/`config.num_hidden_layers`；
  无法解析 → `ValueError`（绝不猜 32）。
- `use_paged × use_cuda_graph` → 工厂/runner/MFF setter 均 `ValueError`（先校验后
  传播；setter 失败旧缓存不变）。
- YAML 冻结键（10）：`use_paged_cache/paged_depth/paged_page_tokens/
  paged_total_tokens/paged_max_ratio/paged_min_ratio/paged_schedule/
  paged_recent_pages/paged_anchor_pages/paged_anchor_stride`；`load_config` 无
  unknown-key 警告。
- `actfold.core` 导出 `PagedFoldCache/FoldPagePlan/PageTable/RolloutSet/
  PlanStaleError/RolloutBudgetError`。
- 构建失败不返回半初始化对象；`use_paged=False` 时既有类型/行为不变。

**验收标准：**
- Given 默认参数，Then `ActivationCache`；paged 符号全部可导入。
- Given `use_paged=True,paged_depth=2`，Then `PagedFoldCache` 且 depth=2；与
  pyramid/vectorized/chunked 同开 → paged 胜出。
- Given `paged_depth=0` 且无层数，Then runner `ValueError`；模型 `num_layers=4`
  → 深度 4。
- Given paged+graph，Then 工厂/runner/setter 均 `ValueError`；setter 后旧缓存不变。
- Given 10 个 YAML 键，Then 字段正确且无 unknown-key 警告。
- Given 非法深度/页大小，Then 工厂抛错且不返回半初始化对象。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | 输出一致性 | CPU 非 split 路径逐位一致（`torch.equal`） |
| 内存 | hot 上界 | 每层 hot 页 ≤ `budget_pages`；`hot_bytes ≤ plan_expected_bytes`；共享页只计一份 |
| 冷层 | cold 上界 | `cold_bytes == Σ(cold 页实际字节)`；`cold_bytes ≤ U × max_page_bytes`（U=任一 live 页表引用的唯一物理页数；零引用页立即 reclaim，故 hot+cold 去重即 U；`max_page_bytes = max_{layer,name} page_tokens×batch×prod(shape[2:])×element_size`，取自已登记元数据）；`total_bytes=hot+cold` |
| 分叉 | fork 成本 | 激活页复制字节 == 0；元数据条目 ≤ 分支×层×页数 |
| 恢复 | 快照恢复 | quick 几何、warmup 后 5 次中位数 `restore+promote 全部 cold 页` < 50 ms（计时小缓存每层物理页 ≤ 预算） |
| 可靠性 | 不变量 | `refcount_leaks==0`；计划/执行 availability 全等；失败原子回滚；stale 检测生效 |
| 性能门 | 测试时长 | AR030 单测套件 < 30 s；quick 基准 < 60 s |
| 兼容性 | 默认行为 | `use_paged=False` 时既有路径逐字节不变 |

## 5. 约束与假设

**约束：**
- 不改变折叠数学/门控；页层只替换存储/寻址。
- cold/缺失页行必须由 `availability` 守卫强制重算；执行窗口内禁止驻留状态改变。
- 不引入 mock；基准为真实张量运算。
- 与 AR028 同文件风险：优先不改 `folded_transformer.py`；AR028 合入后 rebase（T014）。
- CUDA graph × paged 三处拒绝。
- 本机 16GB WDDM，不做真实 checkpoint；正式面 BLOCKED-DEFERRED。

**假设：**
- 页 token 数、层数、隐藏维度在计划构建时已知；绝不猜。
- CPU cold tier 足够演示；真实端侧可扩展 pinned/磁盘（接口预留）。
- 页内 token 固定；末页允许部分有效；`availability` 为**页粒度**不可用映射到 token
  位图（整页 False）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| Page / Physical page | 固定 token 激活块 / 可位于 hot 或 cold 的数据实体（id/refcount 跨 tier 不变） |
| Page table | 分支/层 → 逻辑页 → 物理页 id 映射（不可变快照） |
| COW | 写时复制：共享页被写时分配新页，其他引用保持原值 |
| reclaim/drop | 仅 refcount==0 时释放页实体与 id |
| demote/promote | hot/cold 之间搬运精确副本；不改变页 id/refcount/write_seq |
| hot/cold | 驻留页 / 精确副本页；cold 不计入 availability |
| FoldPagePlan | host 冻结计划（映射、availability、预算决策、epoch） |
| epoch | 全局写版本；成功 mutator 按 FR3 表递增 |
| rollout | 同一 base 上并行候选集合 |
| PSFM | Page-Shared Fold Memory |

## 7. 变更记录

| 版本 | 日期 | 变更 |
|------|------|------|
| draft-1 | 2026-10-20 | 初稿（FR1–FR6） |
| draft-2 | 2026-10-20 | req r1 修复：hard-bound/cold/FR7/原子性/stale/量化/API/边界 |
| draft-3 | 2026-10-20 | req r2 修复：删除 hot 豁免与隐式 promote；快照全量；reclaim/demote；排序；schema；FR7；T010b/T011b |
| draft-4 | 2026-10-20 | req r3 修复：epoch 穷举表；`enter_plan` 只读窗口+暂存写（解决 publish/驻留冲突）；11 键定义；cold 可推导上界；E4 加 stale；完整基准几何与 `nonshared_hot_bytes`；API 签名；snapshot schema+sha256；plan_expected_bytes 逐 name；UT 索引与 rollout_stats 验收；零回读可观测口径；命名/映射 |
| draft-5 | 2026-10-20 | req r4 修复：COW 写粒度冻结（逐页内容去重）；E2 改物理页数判据 + COW-off mutation；窗口两阶段（只读期禁 mutator/提交期允许 COW 比较）；epoch 补 close_*/enter_plan 提交与失败回滚；snapshot 加 page_len/entries；close_candidate/close_rollout 语义；显式 evict 保护规则；cold 上界显式化；plan bytes batch；rollout_stats 可复算 |
| draft-6 | 2026-10-20 | req r5 修复：窗口暂存写合并 epoch +1；step 固定 0；零引用页立即 reclaim；只读内省白名单；seq_len/batch 变化语义；rollout_stats 字段补 reuse_ratio；max_page_bytes 与 U 定义；非法 step 验收 |
| draft-7 | 2026-10-20 | design 门控 r1 修复的 req 侧补充：FR5 增加显式 `record_rollout_masks` 统计入口（guarded mask 的真实生产者）；状态同步为 req 第 6 轮 PASS |
| draft-8 | 2026-10-20 | design 门控 r3 修复：FR2 增加活跃 rollout 硬保护 unmovable 与 RolloutBudgetError；FR4 snapshot plan 增加 layer_budgets（restore 权威预算）；FR5 明确预算不足失败语义 |

| draft-9 | 2026-10-20 | design 门控 r4 修复：rollout 期显式 evict/demote 不可动共享页；窗口提交全或无与候选级隔离边界；窗口内 missing_tokens 豁免；restore 保留计数 |

| draft-10 | 2026-10-20 | design 门控 r5 修复：FR5 三类失败语义与验收拆分；record 仅窗口退出后记账；promote 硬钉住且成功才计数；窗口失败等值口径；restore 仅空实例；evict cold=False；导出/In Scope 对齐 |


| draft-11 | 2026-10-20 | design 门控 r6 修复：promote 先计入 hot 再硬钉住预算、成功才计数；restore 为 max(当前,快照)+1；promotions 定义改成功次数；NFR promote 计时加预算条件；record_rollout_masks 窗口内拒绝并补进 FR3 |



| draft-12 | 2026-10-20 | design 门控 r7 修复：restore 表统一为 max(当前,快照)+1；NFR cold 页口径；补 name 形状/句柄 canonical/零回读 __int__/__float__ |


