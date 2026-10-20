# [AR029] 分页金字塔折叠内存 — 需求规格说明书

**AR029：Paged Pyramid Fold Memory（PPFM）**

| 字段 | 内容 |
|------|------|
| AR | AR029 |
| 前置 | AR015/AR016/AR018/AR020/AR025（批量折叠/图/零同步发布/批量缓存/多 parent）；**AR028（变长多 parent，未归档则本 AR 的变长子集阻塞）**；AR026（图静态体冻结面） |
| 外部参考 | FlashInfer（paged KV / block-sparse / cascade / plan-run）<br>PyramidInfer（pyramid KV / pivot + recent window）<br>PagedAttention（分页 KV 管理） |
| 后续 AR | AR030（循环折叠状态 + 端侧 RSI agent runtime，本 AR 的 Out-of-Scope） |
| 状态 | 需求草稿 v2（待 req 门控第 2 轮） |
| 真实面 | 按用户总纲延后；合成面 + UT 交付，真实配置写入 RERUN |

## 1. 背景与目标

### 1.1 现状与瓶颈

ActFold 的核心是 **Branch Folding**：用相似度门控复用父分支的逐层
FFN/embedding 激活，只重算发散 token。AR025 已把复用扩展到多 parent
批量面；AR018/AR020 把发布成本压到一次 `put_batch`；AR016/AR017 把批量
前向接入 CUDA Graph。

但现有折叠内存仍是**整序列、逐分支、全层常驻**的形态：

1. **无分页/共享**：每个 branch 的每层激活独立分配/复制；同一稳定前缀
   在链与候选之间重复存储（多链 joint 面尤甚），内存碎片与拷贝成本随
   批行数线性增长；
2. **无层级预算**：浅层与深层使用同一保留策略；而深层激活的复用收益
   更高、可保留的"枢轴"更少（PyramidInfer 的核心观察），当前没有按层
   分层保留；
3. **无逐 token 有效位**：现有 `fetch/get_all` 对未写位置零填充，无法
   表达"部分保留 + 其余重算"，因此任何部分保留都必须配一个显式的
   有效性通道，否则会静默用 0 值当证据。

### 1.2 外部思想吸收（深度分析）

| 来源 | 核心机制 | 对 ActFold 的映射 | 采纳/拒绝 |
|------|---------|------------------|-----------|
| **FlashInfer** | Paged KV：逻辑块→物理页的 block table；block-sparse 组合掩码；cascade attention 共享公共前缀；plan/run 分离（host 规划一次、device 零同步执行） | 折叠激活按 **token block 分页**存储；分支/链共享稳定前缀页（copy-on-write）；**cascade fold**：同一稳定前缀页在多分支间只物化/读取一次；**plan/run**：每步 host 生成 page plan，device 侧按 plan 读取，run 零同步 | 采纳（FR1） |
| **PyramidInfer** | 深层注意力更稀疏、集中在少量 pivot token；按层设置 pyramid 预算；保留 pivot + recent window；query-agnostic 重要 token 选择与周期性淘汰 | 逐层 **activation 保留预算**：浅层保留更多 token，深层只保留"稳定枢轴" + 最近窗口；未保留 token 走 **重算（miss）**，绝不静默近似 | 采纳（FR2，lossless-by-recompute） |
| **PagedAttention** | 分页消除 KV 碎片、支持前缀共享与连续批处理 | 与 FR1 同源；折叠缓存的页表/引用计数/COW 语义 | 采纳（FR1） |

**关键判断（不虚构结论）：**

- FlashInfer/PyramidInfer 的收益对象是 **KV cache**；ActFold 复用对象是
  **FFN/embedding 激活**，两者语义不同但**内存/调度问题同构**——分页、
  前缀共享、分层保留、plan/run 都可迁移；效果由合成面与后续真实面
  测量，需求阶段不宣称数字。
- PyramidInfer 的 pivot 选择依赖注意力分数；ActFold 的等价信号是
  **逐 token gate 稳定计数**。现有 `StabilityProfiler` 只保存每
  `(branch, layer)` 的标量 `stable_sum`（`stability_profiler.py:161-179`），
  **没有逐 token 计数**——本 AR 必须新增该计数（FR2），不能假设已有。
- FlashInfer 的 page table 以 `(request, block)` 为键；本 AR 的键为
  `(branch, layer, token_block)`，共享前缀由调用方声明的
  `shared_prefix_len` 驱动（宿主元数据，零设备回读）。

### 1.3 目标

1. **FR1**：无损分页折叠存储（页表/COW/前缀共享/cascade/plan-run/页级 LRU）；
2. **FR2**：无损分层保留（pyramid 预算 + 逐 token pivot + recent window +
   显式有效性通道 + miss 重算）；
3. **FR3**：可观测性（页命中/淘汰/保留/写账本；run 零同步）；
4. **FR4**：预注册合成基准（H1 内存、H2 写放大）+ 真实面配置延后。

**明确不在本 AR：** 循环折叠状态与端侧 RSI agent runtime（AR030）。
本 AR 交付其内存/观测基础，不承诺流式近似或代理循环。

## 2. 需求范围

**In Scope：**

- 新缓存后端 `PagedFoldStore`（实现 `ActivationCache` 的公共方法：
  `put`/`put_batch`/`get`/`get_all`/`fetch`/`clear`/`clear_branch`/
  `num_entries`/`write_op_count`；**不实现/不改变** `validate_entries`
  等自由函数）；
- 前缀共享元数据：`CachePutEntry` 可选 `shared_prefix_len: int = 0`
  （默认 0，旧构造逐字节兼容）；`put_batch`/`put` 接受该字段；
- 分层保留策略 `PyramidRetentionPolicy` + 逐 token 稳定计数扩展
  （`StabilityProfiler` 新增可选逐 token 计数张量，默认关闭）；
- 有效性通道 `fetch_with_validity`（或等价 `fetch(..., return_valid=True)`）：
  返回 `(values, valid_mask)`；未保留/未命中 token 的 `valid_mask=False`；
- 折叠层接入：`_fetch_parent_gate_row` 在 pyramid 模式下使用有效性通道，
  对无效 token 强制 `stable=False`（重算）；
- cascade 接入 `_forward_batched_multi`：同一父共享前缀页在一次前向内
  只聚合读取一次（按 page plan 去重）；
- 合成基准 `ar029_hierarchical_fold_bench.py`（CPU，H1/H2，判定式冻结）；
- 文档与真实面配置（RERUN）。

**Out of Scope：**

- **不实现 KV cache 分页**：本 AR 只分页 ActFold 的激活缓存；KV 仍由
  底模/采样器持有；
- **不修改底模结构/权重**；
- **不实现循环状态（recurrent）与 RSI agent runtime**：AR030；
- **不做近似保留**：所有未保留 token 必须重算（lossless-by-recompute）；
- **不引入新依赖**（不依赖 FlashInfer/PyramidInfer 包；机制自实现）；
- **不做大型真实实验**（用户总纲）：真实面配置/命令写清，延后高速机；
- **不实现 Triton 新核**：分页读写用 PyTorch 实现，CPU/CUDA 均可跑；
  性能只以写/读账本与内存峰值衡量。

## 3. 功能需求

### 3.1 FR1 无损分页折叠存储（Paged Fold Store）

**描述：** 把折叠激活按 token block 分页存储，支持分支/链间前缀共享、
copy-on-write 与 cascade 聚合；读取按 host 端生成 page plan 执行。

**触发条件：** 配置 `cache_backend="paged"`（或显式构造 `PagedFoldStore`）。

**期望行为：**
- 页大小 `page_tokens`（默认 64，可配）把序列维切块；页表键为
  `(branch_id, layer_idx, token_block_idx)` → 物理 page id；
- **前缀共享**：新 branch 写入时以 `shared_prefix_len: int = 0` 声明与
  `parent_branch_id`（`shared_prefix_len>0` 时必填）的逐位相同前缀长度；
  共享前缀的页直接引用（引用计数 +1），不复制；**写入时对声明前缀做
  逐页 `torch.equal` 校验，不一致 ⇒ `ValueError`（含页/偏移）**；
- **COW**：对共享页的首次分歧写入分配新 page，旧页内容与引用不变；
- **cascade fold**：多 parent 批量前向中，相同共享页在一次前向内只被
  物化/读取一次（按 page plan 去重）；cascade 是可观测的（plan 去重
  计数），不是隐含优化；
- **plan/run 分离**：每个生成步在 host 生成 `PagePlan`（读页列表、COW
  写页列表、去重计数）；run 路径只按 plan 执行，零 host sync；
- **图 plan 跨步策略（req 冻结）**：**固定最大 plan**（按
  `max_pages_per_layer`/最大批行数预留静态容量）；当某步 pagination 需求
  超出固定 plan（行数/页数增长）⇒ **显式 eager 回退并计数**
  （`plan_overflow_fallbacks += 1`），不重捕获；固定 plan 内跨步零
  重捕获、零同步；该策略由 UT 钉死（不得由 design 重定义）；
- **API 契约**：实现 §4 列出的 `ActivationCache` 公共方法；`get/fetch`
  返回逻辑值（分页不改变数值）；未写位置沿用零填充（**但 pyramid 模式
  必须走有效性通道**，见 FR2）；
- **容量与淘汰**：按页 LRU，受 `max_pages_per_layer` 约束；被淘汰页在
  **缓存层**读取抛 `KeyError`（与 `ActivationCache` 一致），折叠层
  `_fetch_parent_gate_row` 捕获后转 `None`（旧语义）；
- **图兼容**：page plan 在 capture 外生成；run 路径无动态分配；跨步按
  上条固定最大 plan + 溢出 eager 回退策略；
- **正确性**：同一输入下，`PagedFoldStore` 与 `ActivationCache` 的
  `get/fetch/put_batch` **逻辑状态等价**（值、`_counts`、逐层键序、
  `num_entries`）。

**异常处理：** 页大小非法 / `shared_prefix_len` 越界或声明前缀与 parent
不一致（逐页 `torch.equal` 失败） ⇒ `ValueError`；`shared_prefix_len>0`
但缺 `parent_branch_id` ⇒ `ValueError`；COW 引用计数不一致 ⇒
`RuntimeError`。

**验收标准：**
- Given 两条共享前 `P` 个 token 的 branch（`shared_prefix_len=P`），When
  分别写入，Then 前缀页物理 page id 相同、引用计数=2，`get` 值逐位一致。
- Given 声明前缀与实际值不一致，When 写入，Then `ValueError`（含页/
  偏移），旧页未被修改。
- Given 分歧块写入，When COW，Then 新 page 分配、旧页不变、页数按预期
  增加（ledger 可复算）。
- Given 同一输入，When 分别用 `ActivationCache` 与 `PagedFoldStore`，
  Then `get/fetch/put_batch` 逻辑状态等价。
- Given `PagePlan` 已生成，When run，Then 0 次 host sync（探针断言
  `.item()/.tolist()/.cpu()` 调用为 0）。
- Given 两条 branch 共享前缀页，When 单次 batch 前向，Then plan 去重
  计数=1（cascade 生效），未共享时不发生去重。
- Given `max_pages_per_layer` 达上限，When 写入新页，Then 最旧页被淘汰、
  其读取在缓存层抛 `KeyError`（折叠层转 `None`），`num_pages` 不超限。
- Given 非法 `page_tokens`（≤0/非整数）或 `shared_prefix_len` 越界，
  When 构造/写入，Then `ValueError`。
- Given 某步需求超出固定 plan，When run，Then 显式 eager 回退、计数 +1、
  不重捕获、数值与 plan 内路径逐位一致。

### 3.2 FR2 无损分层保留（Pyramid Retention）

**描述：** 按层设置保留预算，保留"稳定枢轴 + 最近窗口"，未保留 token
通过**显式有效性通道**标记无效并重算；不做近似。

**触发条件：** 配置 `retention="pyramid"`（默认关闭=全保留）。

**期望行为：**
- **逐层预算表（默认 pin）**：层数 `L`；浅层 `floor(L/3)` 层保留
  `T` token，中层 `floor(L/3)` 层保留 `T/2`，深层其余保留 `T/4`；
  另可显式传 `layer_budget: list[int]`；
- **pivot/window 预算自适应（默认 pin）**：`budget = layer_budget[l]`；
  `k_eff = min(64, max(0, budget - 1))`；`W_eff = min(64, budget - k_eff)`；
  `budget < 2` ⇒ 构造期 `ValueError`（默认表在 `T>=512` 时不触发）；
- **pivot**：新增**逐 token 稳定计数**（`StabilityProfiler` 可选张量
  `token_stable_counts[branch, layer, token]`，设备侧累加，默认关闭）；
  按每层/每 branch 的计数选 top-`k_eff` token；pivot 之外保留最近
  `W_eff` token；
- **保留布局**：`[0, T)` 上任一 token 属于 pivot 集合或 recent window
  才落盘；其余不落盘（允许出现"有效-无效-有效"空洞）；
- **有效性通道**：`fetch_with_validity(branch, layer, step_idx=0)` 返回
  `dict[name -> (values [B,T,...], valid_mask [B,T] bool)]`（与缓存
  activation 字典同构）；未落盘/被淘汰 token 的 `valid_mask=False`；
- **折叠层接入（逐段 gate）**：`_fetch_parent_gate_row` 在 pyramid 模式下
  把 `valid_mask` 的**每个连续 True 段**分别与 gate 比较（逐段 cosine/
  quantile），`valid_mask=False` 的 token 强制 `stable=False`（重算）；
  **禁止用零填充值作为证据**；与 AR028 的联合有效区 =
  `valid_mask & (token < v_r) & row_attention_mask[r]`（三者交集，
  交集外一律构造性发散）；
- **正确性（R0/R1/R2 三层）**：
  - R0 = 无父全重算；R1 = 全保留折叠；R2 = pyramid 折叠；
  - R2 在 **无效位置** == R0 对应位置（逐位）；
  - R2 在 **有效位置** == R1 对应位置（逐位；逐段 gate 与全画布 gate
    的逐 token 结果一致，因为门控是逐 token 的）；
  - 差异集合 `{i | R2[i] != R1[i]}` ⊆ `valid_mask=False` 的位置
    （子集断言；不要求"恰为"，避免 R0==R1 的偶然相等误杀）；
  - **不承诺 R2 ≡ R1**（父/子 FFN 不同，见 §1）；
- **内存上界**：每层落盘 token 数 ≤ 预算（含 pivot + window）；
- **统计零同步**：pivot 选择在 plan 阶段完成（允许一次 host 读取，
  计入 plan）；run 路径零同步。

**异常处理：** 预算 ≤0 / `budget<2` / window <0 / 层数不匹配 ⇒
`ValueError`；`fetch_with_validity` 未启用时被 pyramid 调用 ⇒
`RuntimeError`（不允许退回零填充）。

**验收标准：**
- Given R1/R2 同一输入，When 折叠，Then `R2[valid]==R1[valid]`、
  `R2[~valid]==R0[~valid]`（逐位），且差异位置 ⊆ `~valid`。
- Given 有效位空洞（有效-无效-有效），When gate，Then 每个有效段独立
  比较、空洞 token 重算；有效段的稳定判定与 R1 对应 token 一致。
- Given 层预算表，When 发布，Then 每层落盘 token 数 ≤ 预算。
- Given 未保留 token，When 折叠层读取，Then `valid_mask=False` 且该
  token 被强制重算；变异实现返回零填充证据 ⇒ 测试失败。
- Given 默认预算表，When 构造策略，Then 浅/中/深各层预算分别为
  `T/T/2/T/4`（层数分界可复算）；`k_eff/W_eff` 按预算截断，
  `budget<2` ⇒ `ValueError`。
- Given AR028 的 `valid_lengths`/`row_attention_mask` 与 pyramid
  `valid_mask` 同时存在，When gate，Then 门控区间 = 三者交集，pad 列
  不参与，整画布发布行序不变（与 AR028 验收一致）。

### 3.3 FR3 可观测性与零同步账本

**描述：** 暴露分页/保留的内存与命中指标，run 路径零 host sync。

**期望行为：**
- `stats` 只读探针：`page_hits/page_misses/page_evictions/page_copies/
  page_write_count/reuse_bytes`、`pivot_kept/pivot_total`、
  `write_op_count`；
- **账本口径**：`write_op_count` 维持 #42 语义（`put` 计 activation 数、
  `put_batch` 计 foreach 组 + 立即执行）；分页写/COW/cascade 去重另记
  `page_write_count`/`page_copies`，**不混入** `write_op_count`；
- 指标在 plan 阶段聚合（允许一次 host 读取）；run 路径只累加 device
  计数器或 Python int（不读 device）；
- 与 AR018/AR020 发布路径兼容：`_store_batched` 可把 `put_batch` 直接
  投向 `PagedFoldStore`，行序与键序不变。

**验收标准：**
- Given 运行 K 步，When 读取 stats，Then 各指标与工件/账本可复算。
- Given run 路径，When 探针监视，Then 0 次 host sync。
- Given `put_batch` 到 `PagedFoldStore`，Then `write_op_count` 与
  `ActivationCache` 同输入逐值相等（#42 口径不变）。

### 3.4 FR4 基准与验收（预注册）

**描述：** 合成面冻结 H1/H2 + 真实面配置延后。

**期望行为：**
- `scripts/ar029_hierarchical_fold_bench.py`（stdlib-only 顶层，CPU 可跑）：
  - 几何：T∈{1024, 4096}、branches∈{4,16}、`multi_parent_batch=True`、
    `page_tokens=64`；
  - **H1（内存）**：`peak_bytes(paged+pyramid) ≤ 0.6 × peak_bytes(AR025
    全保留)`；字节口径见下；无损模式（R2 语义）必须逐位满足 FR2 验收；
  - **H2（写放大）**：`page_write_count` 严格小于全保留的页写次数
    （按同一页大小折算），且 `write_op_count` 不高于全保留；
  - 判定式先于数据冻结；strict-fail（FR2 逐位判据失败/确定性失败）⇒
    全 blocked；
  - **字节口径（必须可复算）**：`peak_bytes = max over plan steps of
    (Σ 活页 × page_tokens × H × dtype_itemsize + metadata_bytes)`；
    包含页内 slack（末页按 `page_tokens` 计）；`metadata_bytes =
    16 × num_page_entries + 8 × num_refcounts`（页表/引用计数字段的
    固定宽度、不计 Python 对象开销/plan 缓存）；基线为 AR025
    `ActivationCache` 的
    `Σ (branch, layer) × T × H × itemsize + 16 × num_count_entries`
    （`num_count_entries = Σ (branch, step, layer)`，同口径计元数据）；
    峰值窗口=每个 plan step 结束时采样；工件必须记录分子/分母及
    `dtype_itemsize/H/page_tokens/num_page_entries/num_count_entries`，
    使 H1 可由工件重算；
- 工件：`results/optimization/ar029_hierarchical_fold_bench.json`
  （`--quick` 独立，永不覆写正式面）；
- 真实面：LLaDA/Dream 配置、命令、资源、判定式写入
  `docs/RERUN_CHECKLIST.md`，**延后**执行；回填前文档只声明机制与
  合成面结论（AGENTS Experimental Validation Rule 的授权裁剪）。

**验收标准：**
- Given 合成网格，When 正式面，Then 工件含判定式与逐点结果，README
  数字可由工件重算。
- Given 真实面未执行，When 文档发布，Then 明确标注延后且零伪数字。

## 4. 接口契约（需求级）

`PagedFoldStore` 必须实现（与 `ActivationCache` 同名同义）：

| 方法 | 要求 |
|------|------|
| `put(branch_id, layer_idx, activations, step_idx=0)` | 旧签名不变；前缀元数据经 `put_batch`/新 `put_with_prefix` 提供 |
| `put_batch(entries)` | `entries` 为 `CachePutEntry`（新增可选 `shared_prefix_len: int=0`、`parent_branch_id: str|None=None`，默认旧行为） |
| `get(...)` / `get_all(...)` / `fetch(...)` / `fetch_masked(...)` | 旧契约不变；未写位置零填充；**缓存层未命中抛 `KeyError`**（与 `ActivationCache` 一致） |
| `contains(...)` | 与旧缓存同名同义 |
| `fetch_with_validity(branch_id, layer_idx, step_idx=0)` | **新增**：`dict[name -> (values [B,T,...], valid_mask [B,T] bool)]`；未启用 pyramid ⇒ `RuntimeError` |
| `clear_branch(...)` / `clear_layer(...)` / `clear_all(...)` | 页引用计数正确回收（与现有方法名一致） |
| `num_entries` / `write_op_count` | 与旧缓存语义一致 |
| `stats` | 只读探针（FR3） |

**不实现**：`validate_entries`（`batched_write.py` 的自由函数，不是
cache 方法）。

## 5. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 兼容性 | 默认路径 | 未启用新后端/策略时与 AR025 逐字节一致 |
| 正确性 | R2 | 有效位置==R1、无效位置==R0；差异位置 ⊆ 无效位置（逐位） |
| 内存 | H1 | `peak_bytes ≤ 0.6 × AR025 全保留`（口径见 FR4，合成面） |
| 写放大 | H2 | `page_write_count` 严格更低；`write_op_count` 不更高 |
| 零同步 | run | 0 次 host sync；plan 每步 ≤1 次 |
| 可复现 | 合成面 | CPU 正式面 ≤ 15 分钟；seed 固定；工件溯源 |
| 质量门 | 全量 pytest | **AR029 启动时实测起点**，只增不减；四 lint 门干净 |
| 兼容 | AR028 | 变长面（`valid_lengths`/`row_attention_mask`）与页/保留语义不冲突 |

## 6. 约束与假设

**约束：**
- 不引入 FlashInfer/PyramidInfer/PagedAttention 依赖，机制自实现；
- 不改变底模权重/结构；不实现 KV 分页；
- 图路径：page plan 在 capture 外；**req 冻结策略 = 固定最大 plan +
  溢出 eager 回退计数**（不重捕获）；
- **真实面延后的授权与义务（用户总纲，2026-10-19）**：不做大型实验；
  真实 checkpoint 正式面延后，T009 必须显式记录 `BLOCKED-DEFERRED`；
  回填前文档**只可**声明机制与合成面结论，**不得**宣称端到端收益；
  回填义务写入 RERUN 步骤；
- AR028 未归档时，依赖变长的子集阻塞（分页/保留的等长子集可先行）。

**假设：**
- 调用方声明的 `shared_prefix_len` 与实际 token 前缀一致，且写入时经
  逐页 `torch.equal` 校验（不一致即 `ValueError`，不存在静默错误共享）；
- 逐 token 稳定计数（新增）可作为 pivot 的 query-agnostic 代理；
- 合成模型是 first-class（AGENTS），真实 checkpoint 不在本 AR 强制面；
- 页内 slack 按整页计费（保守口径）；
- `PagedFoldStore` 首版只分页 `ffn_out`/`embedding` 两类 activation
  （与折叠父读取所需一致），其他 name 走同一页机制。

## 7. 术语说明

| 术语 | 定义 |
|------|------|
| page | 序列维一个 token block 的激活存储单元 |
| page table | `(branch, layer, block) → page id` 映射 |
| COW | copy-on-write：共享页首次分歧写入时复制 |
| cascade fold | 同一共享前缀页在一次前向内只物化/读取一次 |
| pyramid budget | 每层允许落盘的 token 数（默认 T/T/2/T/4 分段） |
| pivot | 逐 token 稳定计数 top-k、优先保留的 token |
| recent window | 最近 W 个 token 的保留区 |
| valid_mask | 逐 token 有效性（False=未保留/淘汰→重算） |
| R0/R1/R2 | 无父全重算 / 全保留折叠 / pyramid 折叠 |
