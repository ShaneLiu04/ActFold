# [AR020] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR020 |
| AR 主题 | 发布路径 cache put 批量化：`put_batch` 能力协议 + 三 cache 实现 + 真实 checkpoint 发布窗口 A/B 基准（P1-7 第六档） |
| 关联 SR | P1-7 多分支验证墙钟（AR015-19 五档延续）；AR017 发现项 / AR018 srs §1.3 明确排除项 / AR019 遗留清单三度点名"cache 批量 put API" |
| 日期 | 2026-10-14 |
| 状态 | Draft |

## 1. 背景与目标

P1-7 五档落地后（AR019），真实 Fast-dLLM-v2-1.5B checkpoint 上 adaptive vs baseline p50
加速比为 **2.44/2.62/2.59×**（三配置）。发布窗口（`_publish_graph_children`）是 adaptive
路径中 graph 侧最后一段未压缩的宿主侧成本，其证据链跨两代工件、三度遗留点名：

- **AR018 六相探针**（AR017 稳态容量下的发布占比首测，GUIDE §24 遗留行明记"AR017 未触及
  该成本面…AR018 收口——分解已做"）：发布占 adaptive 窗口 **60.7/71.2/29.7/31.5%**；
- **AR018 发布压缩弹头**（GUIDE §25）：clone+sum 两项被压至 1+1 次调用后发布占比降至
  **45.3/50.9/15.7/17.6%**，launch 账本 3·L·N → L·N+2，**put 前向 L·N 未变**——剩余主体
  即 L·N 次 Python `cache.put` 循环，**cache 批量 put API 被 srs §1.3 明确排除**（留独立
  AR，即本 AR）；
- **AR019**：真实 checkpoint 基准环境就绪（L=28 层，AR019 工件矩阵内最大 N=8 时发布循环
  为 **L·N+N=232 次 copy 主机调用 + 224 次 Python 调用/步**；外推 N=16 为 464+448，仅作
  量级示意、非工件配置），并把"cache 批量 put API"列入遗留清单。

"综合框架比 baseline 强"的下一档增益点即在此：把发布循环的**逐条 Python 调用与逐条
copy 主机调用**批量化，同时**零语义变化**（缓存状态与逐条 put 逐位等价）。

### 1.1 选题证据链（代码事实 + 既有工件数字）

| # | 证据 | 来源 |
|---|------|------|
| P1 | 发布循环结构：`for layer × for row: cache.put(...)`，每条 put 内含激活校验循环、`ensure_layer`/LRU `move_to_end`、缓冲 ensure/grow、`transpose(0,1)` 视图、`buffer[:seq].copy_`（每激活名 1 次 copy 主机调用；layer 0 含 `ffn_out`+`embedding` 两名）、`_counts` 写、`_evict_over_budget`→`_layer_tokens` 全层组遍历 | `architecture_utils.py:1093-1104`；`activation_cache.py:83-161` |
| P2 | 每 put 的 copy 目的地为**各分支独立缓冲**（`[capacity, 1, H]`，按 `(branch, step)` 分组），故逐条 put 的 copy 无法在既有布局下自然合并；`torch._foreach_copy_` 可将 N 条同设备 copy 合并为 O(1) 级主机调用与极少 kernel launch | `activation_cache.py:135-158`；torch 2.5.1 实测待设计期确认（CPU/CUDA/非连续源） |
| P3 | 三 cache 的 put 结构同源（校验→ensure→slice copy / 分块 contiguous 存储），但 `ChunkedActivationCache.put` 为**按引用 contiguous 存储**（`tensor[:, s:e].contiguous()`），批量面存在一次性整层 contiguous 化的额外合并机会 | `chunked_cache.py:119-138`；`vectorized_cache.py:83-139` |
| P4 | 发布路径消费面已具备 duck-typed 前例：AR018 的 `record_precomputed`/`record` 双入口按能力分流；AR017 注记"cache 协议 duck-typed（不支持 put 的走旁路）" | AGENTS #37；`stability_profiler.py` |
| P5 | 基准环境：AR019 venv（transformers 4.53.1）+ 本地 Fast-dLLM-v2-1.5B 权重 + 三配置口径可直接复用；CUPTI null（#39）⇒ launch 计数不可得，主机调用计数与 CUDA-event 计时为可测代理 | `results/optimization/ar019_real_e2e_bench.json`；AGENTS #39 |
| P6 | 负面边界：串行图发布（`_publish_graph_child`）单条 put 无批量化收益；eager 路径的 put 与层计算交错，不可批（均 Out of Scope） | `architecture_utils.py:822-860` |

### 1.2 口径延续

- **基准模型/负载/baseline 口径**：与 AR019 完全一致（Fast-dLLM-v2-1.5B 块验证工作负载，
  baseline = 逐 child 裸前向；三配置矩阵沿用 AR019 工件的三配置）。
- **A/B 面**：同一 adaptive 路径下发布变体 `loop`（现状）vs `batch`（本 AR），发布窗口
  独立计时 + 端到端 p50 同报。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- FR1 `put_batch` 批量写能力：三 cache 实现（`ActivationCache` / `ChunkedActivationCache` /
  `VectorizedActivationCache`）+ 可选能力协议（runtime-checkable Protocol，非强制继承）
- FR2 发布路径消费：`_publish_graph_children` 能力探测分流（有 `put_batch` 走批量，无则
  原循环逐位不变）；零 host-sync 契约与 AR018 快照新鲜度论证保持
- FR3 真实 checkpoint 发布窗口 A/B 基准脚本 + 工件（H1-H4 假设-判定分离 schema，reps=30
  预承诺单次定案）
- FR4 文档收口（含 README 全量中文重写，按 AGENTS.md 既定规则）

**Out of Scope（本 AR 不做的）：**

- 串行图发布（单条 put，无收益）
- eager 路径逐层 put（与层计算交错，不可批）
- cache 存储布局重构（分支持久缓冲布局不变——本 AR 只批量化调用与 copy，不引入
  分支间共享存储块）
- 变长 / 多 parent 批量（AR016 遗留，正交）
- `FastDLLMAdapter` 生产接线（延续 AR019 排除决策）
- LLaDA-8B / Dream-7B（16GB 不可行）

## 3. 功能需求

### 3.1 FR1 `put_batch` 批量写能力（三 cache）

**描述**：为三种 activation cache 提供 `put_batch(entries)` —— 一次调用写入 M 条
`(branch_id, layer_idx, activations, step_idx)` 条目，语义与逐条 `put` **完全等价**。

**触发条件**：发布路径（FR2）或其他调用方需要一次写入多条同构条目。

**期望行为**：
- **等价性（H1，硬性）**：对任意**合法**条目序列，`put_batch` 后的缓存状态与逐条 `put`
  后**逐位一致**——每个 `(branch, layer, step)` 的各激活名缓冲 bit-equal、`_counts` 一致、
  LRU 顺序一致（`OrderedDict` 键序）、eviction 结果一致。三种 cache × CPU/CUDA 全覆盖。
  （非法序列不在等价性范围内——见下条原子性，那是比逐条 put **更强**的独立保证：逐条
  put 遇到中段非法条目是部分写入后抛错，`put_batch` 是零写入抛错，两者终态有意不同。）
- **原子性**：全部条目先校验后写入；任一条目非法（空激活 / 维度不足 / 前导形状不一致）
  → `ValueError` 且**缓存状态零改动**（不允许多条目部分写入）。
- **copy 批量化**：同一 **(device, dtype)** 分组内的 slice copy 经 `torch._foreach_copy_`
  （或设计期确认的等价机制）合并——`_foreach_copy_` 对组内异构布局的支持限制由设计期
  实测定案；dtype 混批按 `(device, dtype)` 分组执行，组内布局超出 foreach 支持面的条目
  安全降级为逐条路径（结果仍等价）。
- **容量语义保持**：`seq_len > capacity` 的环形布局条目走既有逐条等价路径（正确性优先，
  批量化不覆盖环形写）；grow 语义与逐条 put 一致。
- **能力协议**：新增 runtime-checkable `BatchedPutCapable` 协议（结构化，无继承要求）；
  三 cache 实现之。既有 `ActivationCacheProtocol` 不变（`put_batch` 是可选能力，非契约收紧）。
- **兼容性**：`torch._foreach_copy_` 不可用（更旧 torch）→ `put_batch` 整体降级为内部
  逐条 put 循环（结果等价、收益归零、不告警不报错——能力探测在 FR2 消费面完成）。

**异常处理**：条目结构非法 → 批量前置校验 `ValueError`（零部分写入）；设备混批 → 按
`(device, dtype)` 分组执行（合法）。

### 3.2 FR2 发布路径消费

**描述**：`_publish_graph_children` 在 cache 具备 `put_batch` 能力时，把 L×N 条目一次性
批量写入；否则走既有双循环，**逐位不变**。

**触发条件**：批量图路径发布（AR018 快照 + `record_precomputed` 路径）。

**期望行为**：
- 能力探测：发布时对 cache 做一次能力判定（`BatchedPutCapable` runtime check 或
  等价 duck 探测；探测结果可缓存于实例，避免每步重判——设计定案）。**探测本身不得
  引入 device sync**。
- 批量变体：条目负载与逐条版**完全相同**（`ffn_out` = 快照行视图 `[1,T,H]`；layer 0
  追加 `embedding` 行视图）；profiler 侧（`record_precomputed` 和 / debug `record`）
  **零变化**（AR018 语义原样）。
- 回退变体：无能力 cache（含测试用 duck-typed 假 cache）走原双循环——既有行为与
  既有测试零扰动。
- **零 host-sync 契约保持**（AGENTS #37）：发布入口到返回之间不得出现 `.item()`/
  `.tolist()`/`.cpu()`/隐式同步；`num_tokens` 仍取自 `tokens.shape[1]`。
- **快照新鲜度论证保持**（AR018）：批量条目的源仍是私有快照的行视图；发布返回后
  本侧对快照零写入——按引用消费的 cache（`ChunkedActivationCache` 分块 contiguous）
  依然安全。
- `debug_enabled` 路径（逐行 `record`）不变——批量化的对象只是 cache 写，profiler 面
  独立。

**异常处理**：`put_batch` 抛出的任何异常原样上抛（不吞不改档）；发布路径不因批量变体
新增任何静默回退。

### 3.3 FR3 基准脚本 + 工件

**描述**：真实 checkpoint 上测量发布窗口 A/B（loop vs batch）与端到端三路径墙钟，
产出预承诺单次定案工件。

**触发条件**：`python -m scripts.ar020_publish_put_bench`（AR019 venv + 本地权重）。

**期望行为**：
- 配置矩阵与 AR019 工件一致（三配置，同 T/N/块结构）；每配置：
  - **发布窗口微基准**：同一捕获 runner 上，`loop` 与 `batch` 两种发布变体各 reps 次，
    **宿主 `perf_counter` + 前后 `torch.cuda.synchronize()` 夹逼计时**（与 AR019 端到端
    同族口径——AR019 工件 meta 已明注 CUDA-event 计时**排除宿主侧发布成本**，GUIDE §23
    语义要点(3) 同记"会系统性高估"，而本 AR 测量对象恰以宿主侧成本为主体，故 CUDA-event
    不作判定口径，至多作 GPU 侧佐证）；warmup + round-robin + p50/std；
  - **端到端宏基准**：adaptive 路径分别在两种发布变体下的逐步墙钟 p50/std，以及
    baseline/fixed 参照（沿用 AR019 口径，宿主计时同族）；
  - **主机调用账本**：发布窗口内 copy 主机调用计数（spy 级，结构证据；CUPTI null 下
    不做 kernel-launch 计数声明，工件内注记 #39）；
  - **正确性门**：batch 变体发布后逐分支 fetch 与 loop 变体 bit-equal（发布等价性
    在真实模型上的端到端确认）。
- 工件 schema 假设-判定分离（三态映射：supported / refuted / inconclusive，判定式在
  设计期冻结后不可再改）：

| 假设 | 内容 | 判定（逐配置，记 σ = max(σ_loop, σ_batch)） |
|------|------|------|
| H1 | `put_batch` 与逐条 put 缓存状态逐位等价（三 cache × CPU/CUDA，UT 面 + 真实模型发布面） | supported ⇔ 全部 bit-equal；否则 refuted（无 inconclusive） |
| H2 | 发布窗口 p50：batch 优于 loop | supported ⇔ p50(loop) − p50(batch) > 2σ；refuted ⇔ p50(batch) − p50(loop) > 2σ；inconclusive ⇔ \|Δp50\| ≤ 2σ（WDDM 展布淹没先例：AR019 H1） |
| H3 | 端到端 adaptive：batch 无回退，且最小配置严格改善 | supported ⇔ 全配置 p50(batch) ≤ p50(loop) + 2σ **且** 最小配置 p50(loop) − p50(batch) > 2σ；refuted ⇔ 任一配置 p50(batch) > p50(loop) + 2σ；inconclusive ⇔ 其余（未回退但严格改善未达 2σ） |
| H4 | 主机调用账本：put 侧 copy 主机调用 L·N+N → O(L+N)（spy 计数；**测量对象 = 基准所用 cache，即 ActivationCache**——ChunkedActivationCache 无逐条 copy 账面，见 design 4.2.2(1)） | supported ⇔ batch 计数 ≤ c·(L+N)；否则 refuted。**c 于 design 冻结并写入工件 meta，基准运行前不可再改** |

  其中"最小配置" = 三配置中 `N·T` 最小者（design 期从 AR019 工件冻结指认并写入脚本
  常量）；H3 的整体判定 = 逐配置判定的最保守聚合（任一 refuted → 整体 refuted；全部
  supported → 整体 supported；其余 → 整体 inconclusive），聚合规则同适用于 H2。

- **预承诺单次纪律**（#39 延续）：reps=30 单次运行定案，不重跑钓鱼；WDDM 方差淹没
  判定时如实记 inconclusive。

**异常处理**：基准失败（捕获失败/显存不足）→ 工件记录失败配置与原因，不伪造数字。

### 3.4 FR4 文档收口

五件套：`docs/OPTIMIZATION_GUIDE.md`（第二十七部分）、`CHANGELOG.md`、`AGENTS.md`
（新 pitfall 条目：put_batch 能力契约）、`docs/RERUN_CHECKLIST.md`（复现行）、
`README.md` **全量中文重写**（AGENTS.md 既定规则：创新表新行 + 头条数字 + 质量基线 +
路线图重写；数字全部从工件重算）。

## 4. 非功能需求（质量基线）

| 类型 | 指标 | 要求 |
|------|------|------|
| 回归 | 全量 pytest（`-m "not slow"`） | 864 passed / 3 skipped + 新增 UT 数，零既有失败 |
| 静态 | mypy / pyflakes / 100 列 | clean / 零 / 新增行 ≤100 列 |
| 语义 | demo 三指标 | 85.5% / 2.35e-03 / 93.75% 锚点不变 |
| 语义 | 既有发布路径 | 无能力 cache 走原循环，逐位不变（既有 UT 零改动通过） |
| 性能 | 无回退面 | batch 不可用/不适用时降级路径的额外开销可忽略（探测 O(1)） |

## 5. 约束与假设

**约束：**
- torch 2.5.1+cu121 / WDDM / CUPTI null / 无 C 编译器（沿用 #39、AR014 记录）
- 16GB 显存：模型 + 图缓冲 + 基准峰值须留余量（AR019 峰值 8.9GB，本 AR 增量可忽略）
- `torch._foreach_copy_` 为下划线私有 API：使用点须隔离（单点封装），并在 AGENTS 条目
  注记版本依赖与降级路径

**假设：**
- 发布窗口的剩余主体确为 put 循环（AR018 工件 + P1 代码结构；若基准显示占比已低于
  测量噪声，H2/H3 判定会如实反映——本 AR 不预设收益幅度）
- 三配置矩阵下快照行视图均为连续切片 `snapshot[l][row:row+1]`（AR018 结构事实）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| put 循环 | `_publish_graph_children` 中 L×N 次逐条 `cache.put` 的双循环（AR018 账本记"put 前向 L·N"未变项） |
| 发布窗口 | 一次批量图发布（快照 clone → put 写入 → profiler 记录）的墙钟区间（AR017/18 计时口径） |
| 主机调用账本 | 发布窗口内 copy 类 torch 调用的宿主侧计数（CUPTI null 下 kernel launch 计数的可测代理） |
| put_batch | 一次调用写入 M 条 `(branch, layer, activations, step)` 的批量写能力；语义与逐条 put 逐位等价 |
