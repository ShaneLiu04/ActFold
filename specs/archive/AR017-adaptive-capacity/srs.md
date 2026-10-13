# [AR017] srs — 自适应发散容量重捕获 + 几何扫描基准（P1-7 决定性重答）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR017 |
| 主题 | P1-7 第三档「自适应容量」：批量 graph 路径的固定发散容量 `C = ceil(ratio·N·T)` 与真实逐层发散数 `D_l` 之间的 3.2-3.8× 重计算浪费，由**观测驱动的能力重捕获策略**（预算超限增长 + 滞回收缩 + 重捕获次数上限）消除；并以**模型几何扫描基准**（H=256/L=8 vs H=1024/L=16）重答 P1-7——探针已证实在计算受限几何下自适应容量把 graph-vs-baseline 从 1.36-1.41×（inconclusive 区间）推到 **2.32-2.43×** |
| 日期 | 2026-10-13 |
| 状态 | draft（待 req 门控） |
| 前置 | AR016（批量 graph 分发 `graph_runner_batched`、重放前 parent 再校验、`_graph_forward_batched_or_none` 退化矩阵）；AR002（`FoldedGraphRunner`、#37 graph 契约、固定容量语义 `C = ceil(ratio·N·T)`） |

## 1. 背景与动机

### 1.1 证据链（本 AR 的深度依据，全部本机实测）

1. **AR016 工件事实**（`results/optimization/ar016_batched_bench.json`）：
   六配置 `mean_stable_ratio ∈ [0.8438, 0.8674]`，即真实发散占比
   `D/(N·T) ∈ [0.1326, 0.1562]`；而 capture 固定容量
   `C = ceil(0.5·N·T)`（cuda_graph.py:208-209）——`_static_forward`
   每层对 `[C, hidden]` 行**全量**重算 FFN（cuda_graph.py:387，
   `_padded_divergent_index` 固定容量填充），与实际 D 无关。
   **FFN 重计算浪费因子 = C/D ≈ 3.2-3.8×**，六配置全部成立。
2. **容量因果探针**（同负载同模型，仅扫 `capacity_ratio`，配对交错）：
   H=256/L=8 下 ratio 0.5→0.1875 降低 graph 墙钟 8-12%（开销主导区，
   FFN[C] 项仅占 graph 墙钟 ~25%——按斜率反推）；H=1024/L=16 下降低
   **41.7-42.0%**。容量是因果主因之一，且收益随几何放大。
3. **几何扫描探针**（H=1024/L=16，FFN:head≈8:1 接近真实模型比例，
   N∈{4,8}×T=512，配对交错，n=10）：

   | 几何 | graph@0.5 vs baseline | graph@0.1875(≈D) vs baseline |
   |---|---|---|
   | H=256/L=8 | 1.37-1.46× | 1.49-1.66× |
   | H=1024/L=16 | 1.36-1.41× | **2.32-2.43×** |

   **解读**：AR016 的 P1-7 inconclusive（1.37-1.77×，3/6 过 2σ）有
   两个根因——容量浪费（机制层，本 AR 修复）与小几何下开销主导
   （测量层，本 AR 以几何扫描显形）。固定 ratio=0.5 恰好把 graph
   钉在 1.4× 附近；容量贴近真实 D 后，计算受限几何下加速比翻倍
   且决定性。

### 1.2 机会

`graph_capacity_ratio` 是 capture 时定死的静态超参；而每次重放的
预算读回（`count_buf.tolist()`，每步恰一次同步）**已经**免费产出
逐层 `D_l`——观测在手，却被丢弃。把"观测 → 策略 → 重捕获"闭环起来：

- **预算超限**（现有语义：丢弃重放 + eager 重算）之后，用观测到的
  max-layer D **增长**容量重捕获——当前该场景是永久性反复丢弃
  （ratio 太小则步步 eager，graph 形同虚设）；
- **持续低占用**（连续 K 步 max-D ≤ β·C）时**收缩**容量重捕获
  ——消除 3.2-3.8× 的 FFN 浪费；
- 滞回（hysteresis）+ 最小重捕获间隔 + 总次数上限约束抖动负载下的
  重捕获风暴；重捕获成本（capture ~几十 ms）由后续步摊销。

这是 AR016 遗留清单中「H2 收窄的宿主侧成本分解」的机制层答案之一
（容量项），并首次给出几何外推证据（框架优势随模型规模放大）。

### 1.3 明确不在范围

- 串行 graph 路径（`graph_runner`，[1,T]）的自适应——机制共享但
  状态/接缝独立，验证负载（P1-7）在批量侧，srs 记录为后续档；
- 变长/多 parent 批量、Triton 批量 kernel（AR016 遗留其余项）；
- prefill/发布/读回等**宿主侧**固定开销的进一步压缩——与容量项
  正交（容量是设备侧计算浪费，本 AR 收口；宿主侧分解仍开放，
  几何扫描证据只给出其占比随几何缩小趋势）；
- 动态 shape graph 池（每形状一图）——远超本 AR。

## 2. 术语

| 术语 | 定义 |
|------|------|
| 发散容量 C | capture 时固定：`C = ceil(capacity_ratio·N·T)`（cuda_graph.py:208-209）；每层预算约束 `D_l = N·T - count_l ≤ C`（:311） |
| 逐层发散数 D_l | 重放时 fused gate 统计的真实发散 token 数（layer 粒度）；`max_l D_l` 是容量需求的充分统计量 |
| 能力重捕获 | 以新 `capacity_ratio` 构造**新** `FoldedGraphRunner` 并 capture，成功后 swap `graph_runner_batched` 并释放旧 runner 的操作；**新图 capture 失败不弃旧图**（先捕新后弃旧，峰值双图驻留） |
| 自适应策略 | 纯函数决策：输入（当前 C、观测 D 序列、距上次重捕获步数、预算超限标志、已重捕获次数），输出（no-op / 重捕获至 C'）；配置驱动，默认关闭 |

## 3. 功能需求

### 3.1 FR1 观测接缝：单次读回内暴露逐层 D

**描述：** `FoldedGraphRunner.validate_budgets()`（cuda_graph.py:298-313）
已在每步重放后执行且恰含一次 `count_buf.tolist()` 读回（`replay()`
:295 无条件调用——**预算超限步同样执行**，暂存同样有真值）；将逐层
`D_l = N·T - count_l` **随预算校验一并暂存**（如
`runner.last_layer_divergences: list[int] | None`），供策略消费。

**验收标准：**
- 每步重放的 D2H 同步次数**不变**（仍恰一次——策略读暂存值，
  绝不二次 `tolist()`/`.item()`）；
- capture 体（`_static_forward`）零改动、零新增 host 同步（#37 契约）；
- `cuda_graph.py` 改动限此暂存（+ 初始化 None），既有 `test_cuda_graph`
  与 AR016 全部 UT 零改动全绿；
- 暂存语义：每次 `validate_budgets()` 覆写；未校验时为 None
  （prefill 失败步 `replay()` 于 `graph.replay()` 前返回 None，
  `validate_budgets` 未执行——暂存为**陈旧值**，调用侧不得消费，
  见 FR3 窗口契约）。

### 3.2 FR2 自适应策略（纯函数 + 配置驱动，默认关闭）

**描述：** 新增策略模块/函数：依据观测序列决定 no-op 或重捕获目标
容量 `C'`。`ActFoldConfig` 新增字段（命名沿 `graph_*` 族）：
`graph_adaptive_capacity: bool = False`、
`graph_adaptive_grow_factor: float`（重定容量的余量系数，建议默认
1.25）、`graph_adaptive_shrink_threshold: float`（收缩触发比 β，
建议默认 0.5）、`graph_adaptive_shrink_patience: int`（连续 K 步，
建议默认 3）、`graph_adaptive_min_recapture_interval: int`（建议默认
2）、`graph_adaptive_max_recaptures: int`（建议默认 8）。数值边界
（如 grow_factor ≥ 1、patience ≥ 1）构造时校验。

**验收标准：**
- **增长分支**：本步 `budget_exceeded` → 目标
  `C' = ceil(grow_factor · max_l D_l)`，clamp 到 `[1, N·T]`；**本步
  结果语义不变**（丢弃 + eager 重算，AR016 语义逐位不动）——自适应
  只影响**未来**容量；
- **收缩分支**：连续 `patience` 步 `max_l D_l ≤ β·C` → 目标
  `C' = ceil(grow_factor · max(最近 patience 步的 max_l D_l))`，clamp
  `[1, C)`（收缩分支恒不增容）；`C ≤ 1` 时收缩 no-op（clamp 区间
  为空）；
- **节流**：距上次重捕获 < `min_recapture_interval` 步 → no-op
  （观测仍记录）；累计重捕获次数达 `max_recaptures` → 永久 no-op
  （策略退役，容量定死于当前 C）；
- **方向单调**：增长分支目标 ≥ 当前预算需求，收缩分支目标 < 当前 C
  ——不出现"预算超限反而缩容"；
- 纯函数可测：给定观测序列与配置，动作序列确定；**策略无副作用**——
  退役/重捕获的一次性 UserWarning 由**调用侧**依据策略返回的动作发出
  （保持表驱动可测，FR3 落地）；
- **默认关闭零行为**：`graph_adaptive_capacity=False`（默认）时策略
  代码路径不执行，AR016 行为逐位一致（无新状态、无新告警、无重捕获）。

### 3.3 FR3 重捕获生命周期（批量侧）

**描述：** `_graph_forward_batched_or_none` 在自适应开启时运行策略。
**调用点（两处，均以"预算校验已执行"为前提——`runner.replay()`
返回非 None 时成立；prefill 失败步 `replay` 返回 None，策略不运行，
陈旧暂存不消费）：**
- **增长评估（超限步）**：`runner.budget_exceeded` 分支内、丢弃
  return None **之前**——消费本步 `last_layer_divergences`（超限步
  有真值，FR1），策略给出增长目标则执行重捕获（见下），随后**照常
  丢弃 + eager，本步语义逐位不变**（req 门控 Major-1：策略必须在
  超限分支内可见，否则 ratio 过小场景步步超限、增长分支不可达）；
- **收缩评估（通过步）**：发布完成**之后**（发布消费旧 runner 的
  `child_buf`/`mask_buf` 静态缓冲——swap 不得先于发布，否则发布
  读到新 runner 的未初始化缓冲）、返回 logits.clone() 之前。

**观测窗口契约（调用侧维护，req 门控 Major-2）：**
- 每次 `replay()` 返回非 None（即 `validate_budgets` 已执行）后
  append `(max_l D_l, exceeded)`；prefill 失败步**不入窗**；
- 重捕获成功后：窗口清空、距上次重捕获步数归零（新容量下的观测
  重新积累，避免旧观测驱动新图的误判）；
- 策略退役后：停止累积（窗口冻结不再消费）；
- 自适应关闭：不累积任何状态。

**重捕获操作（先捕新后弃旧）：** 以
`C'/ceil(N·T)` 换算新 ratio 构造**新** `FoldedGraphRunner`（构造
参数其余与 AR016 capture 完全一致）并 capture 同一 (tokens,
branch_ids[0], parent_branch_id)。

**验收标准：**
- **顺序**：先捕新、成功后 swap（`graph_runner_batched` = 新 runner、
  `_graph_batched_mask` 同步维护）、再弃旧（旧 runner 引用交 GC）——
  新 capture 失败（RuntimeError）时旧 runner 保持在场继续服务，**不
  swap**（req 门控 Major-3：**禁止置位
  `_graph_batched_capture_failed`**——该标志的语义是禁用整条批量
  graph 路径，会连废旧 runner；自适应退役走**独立标志**
  `graph_adaptive_retired`）；失败后一次性 UserWarning（**独立一次性
  标志**，不走 `_warn_graph_degraded_once` 家族布尔——避免与
  scheduler/gate/CUDA/capture-fail 的跨原因抑制语义互相干扰）+
  **自适应退役**（此后任何步不再尝试任何重捕获），本步结果照常
  返回——正确性零影响；
- 重捕获当步：本步结果由**旧图**重放产出（增长分支：本步随后丢弃
  走 eager；收缩分支：本步发布已完成）——重捕获耗时属适配事件，
  基准侧单独记录（FR5），不污染稳态计时；
- 预算超限一次性告警标志（`_graph_batched_budget_warned`）**不**因
  换 runner 复位（告警预算语义：同族只告警一次）；退役告警与重捕获
  失败告警各用独立一次性标志；
- `graph_runner_batched` 为 None（尚未捕获/永不捕获）时策略不运行
  ——零观测、零状态、静默无自适应；
- 串行 runner（`graph_runner`）与串行路径零干扰；
- 自适应关闭时本 FR 全部代码路径不触发。

### 3.4 FR4 对拍与观测钉（UT-1601~，`tests/test_adaptive_capacity.py`）

**验收标准（CUDA 用例 skipif 保护；容差沿 AR016 先例
allclose(1e-4, rtol=1e-4)）：**
- **UT-1601 策略纯函数表驱动**：增长（超限→目标含 grow_factor 余量、
  clamp 生效）、收缩（patience 计数、β 触发、恒小于当前 C、C≤1
  no-op）、节流（interval 内 no-op、达上限退役）、方向单调、边界配置
  校验（非法值构造时抛错）——CPU 可测（无 CUDA 依赖）；退役动作的
  一次性告警由调用侧发出（FR2），表驱动只断言动作序列；
- **UT-1602 收缩端到端**：负载 D/(N·T)≈0.15（沿 AR016 场景族），
  初始 ratio=0.5 + 自适应开启 → patience 步后 `graph_runner_batched`
  换新对象且 `capacity` 收缩到 ~`ceil(grow_factor·D_max)` 邻域
  （断言 `capacity < 原容量` 且 `capacity ≥ max_l D_l`）；收缩后
  logits 与 eager 批量对拍 allclose(1e-4) + 逐行 argmax 相等；
  预算持续通过；
- **UT-1603 增长端到端**：初始 ratio 极小（如 0.02）→ 首步预算超限：
  本步结果 == eager（语义不变）+ 一次性告警；随后自适应重捕获增长；
  第二步起重放预算通过且 logits 对拍一致；**swap 后构造高发散候选
  再次超限 → 零新预算告警（`_graph_batched_budget_warned` 未因换
  runner 复位）**；
- **UT-1604 抖动上限**：构造交替高低 D 的负载序列逼近上限 →
  重捕获次数 ≤ `max_recaptures`，达上限后容量不再变化 + 一次性
  退役告警；结果全部与 eager 对拍；
- **UT-1605 重捕获失败安全网**：monkeypatch `FoldedGraphRunner.capture`
  为"新构造调用失败"（首次成功捕获后置入一次性失败——patch 时序
  沿 UT-1507 先例，失败后立即还原）→ 一次性告警 + **独立退役标志
  置位且 `_graph_batched_capture_failed` 保持 False** + 旧 runner
  继续服务：**可观测断言（req 门控 Major-3："结果一致"无法区分
  eager 回退，必须钉对象与路径）**——后续合格步
  `graph_runner_batched` 恒为旧实例对象（`is` 断言）+ 重放 spy 计数
  递增（或等价探针）+ 结果与 eager 对拍一致；
- **UT-1606 默认关闭零行为**：`graph_adaptive_capacity=False`：
  多步重放后 runner 对象恒不变、容量不变、零告警、零策略状态——
  AR016 行为逐位一致；
- **UT-1607 观测接缝**：重放后 `last_layer_divergences` 为长度 L 的
  list 且 `max(...) ≤ N·T`、与 `capacity`/`count_buf` 自洽
  （`D_l = N·T - count_l`）；未校验时 None；连续两次重放值随负载
  变化（非陈旧值）；
- **UT-1608 串行零干扰**：同实例批量自适应多次重捕获后，串行
  graph 捕获/重放行为与 AR016 UT 一致（`graph_runner` 对象与容量
  不受批量侧影响）；
- **UT-1609 mutation spot-check（纪律沿用）**：①删策略调用点 →
  UT-1602 容量收缩断言红；②删增长分支（或增长调用点）→ UT-1603 红；
  ③删上限退役 → UT-1604 红；④删"先捕新后弃旧"顺序（改先弃旧）+
  capture 失败注入 → UT-1605 红（旧 runner 丢失，`is` 断言炸）。
  全验证后 SHA256 逐字节还原。

### 3.5 FR5 几何扫描基准（`scripts/ar017_adaptive_bench.py`）

**描述：** 独立基准脚本（沿 ar002/ar016 先例），配对比较三路径并
扫描模型几何，产出 `results/optimization/ar017_adaptive_bench.json`。

**验收标准：**
- **三路径**：(a) baseline eager（N 次全量前向）；(b) folded 批量
  graph 固定容量 ratio=0.5（AR016 现状）；(c) folded 批量 graph
  **自适应容量**（初始 ratio=0.5，FR2/FR3 生效）；
- **几何扫描**：(H=256, L=8) 与 (H=1024, L=16) × N ∈ {4, 8} ×
  T=512——探针证实的对照矩阵（小几何开销主导 vs 大几何计算受限）；
- **配对交错**：每重复内三路径 round-robin（RERUN_CHECKLIST §4.2）；
  warmup ≥ 3 + 重复 ≥ 10，TimingStats（mean/std/p50/n）复用
  `scripts/algo_experiments.py:196/:216`；计时 `time.per_counter` +
  `torch.cuda.synchronize()` 边界（沿 AR016 口径：墙钟含 prefill/
  读回/发布）；
- **适配事件独立记录**：路径 (c) 的重捕获事件表（步号、旧/新容量、
  触发原因）单列于工件（`adaptation_events`）——适配开销与稳态
  墙钟分离呈现，p50 取适配稳定后的重复（warmup 覆盖适配期：warmup
  步数 ≥ patience+1，或显式等待容量稳定——design 精确化）；
- **正确性随行**：三路径 tokens argmax 一致 + folded 两路径
  allclose(1e-4)；自适应路径的最终容量与观测 D 分布
  （`observed_max_layer_divergence` p50/min/max）入工件；
- **假设与结果分离**（docstring 声明，JSON 只存实测与判定）：
  - H1：自适应 graph 墙钟 < 固定 0.5 graph 墙钟（逐几何配对，
    2σ 判据沿 AR016 `_verdict`）；
  - H2：自适应 graph-vs-baseline 加速比 ≥ 固定容量的加速比
    （P1-7 以自适应容量重答）；
  - H3：几何外推——H=1024/L=16 的自适应加速比 > H=256/L=8 的
    加速比（计算受限区优势放大）；
  - H4：三路径 logits 一致（argmax 精确）；
- **环境与限制**（#39 纪律，沿 ar016 字段）：GPU/日期/CUPTI 探测
  （预期 null+注记）/交错声明/复现命令/种子；**补 GPU 驱动版本**
  （AR016 review Minor-2 豁免项，本次补记）；
- **运行预算**：全量 < 10 分钟；`--quick` 单配置子集。

### 3.6 FR6 文档收口

**验收标准：**
- 指南：P1-7 行（:180）追注第三档「自适应容量 ✅ AR017」+ 新增
  **第二十四部分**（AR 记录 + 几何扫描墙钟表 + P1-7 重答判定 +
  容量/观测/策略语义要点）；
- CHANGELOG AR017 节置 Unreleased 最前；
- AGENTS #37 扩注（自适应容量契约：默认关闭、单次读回不变、先捕新
  后弃旧、上限退役、增长分支不改变本步丢弃语义）；
- AR016 遗留清单收口注记：容量项收口（本 AR），宿主侧固定开销分解
  仍开放（几何证据已给出其占比趋势）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 静态检查 | mypy clean（65 文件基线）/ pyflakes 零 / 100 列 |
| 质量 | 全量回归 | 838 passed 基线 + 新增零失败；既有测试零改动 |
| 稳定 | demo 锚点 | 三指标精确一致（demo 不经批量/自适应路径） |
| 稳定 | 测试确定性 | CUDA 对拍仅 allclose+argmax；无墙钟断言进 pytest；UT 运行时 <30s |
| 兼容 | 默认行为 | `graph_adaptive_capacity=False` 默认：AR016 行为逐位一致（UT-1606 钉住） |
| 兼容 | 公开 API | `cuda_graph.py` 改动限 FR1 暂存；串行 graph 零语义变化；AR016 批量路径在自适应关闭时逐位不动 |
| 性能 | 同步纪律 | 每步重放 D2H 同步恰一次不变；capture 体零 host 同步 |
| 性能 | 基准诚实性 | 墙钟含 warmup/重复/std；适配事件与稳态分离；假设判定基于实测 |

## 5. 约束与假设

**约束：**
- 生产改动限：`cuda_graph.py`（FR1 暂存，最小）、
  `architecture_utils.py`（策略调用点 + 观测窗口 + 重捕获生命周期）、
  `actfold/utils/config_manager.py`（新字段 + 边界校验——仓库无
  `config.py`，现有 `use_cuda_graph`/`graph_capacity_ratio` 字段在
  config_manager.py:83-84 区域）、（可选）策略纯函数独立模块、
  新脚本 + 文档；自适应配置流入 `ManualFoldedForward` 经构造参数
  （沿 `graph_capacity_ratio` 先例 architecture_utils.py:396-402，
  `BenchmarkRunner`/脚本侧接线沿 AR016 基准同款）；
- 每步重放同步次数、capture 体纪律、预算超限丢弃语义、#41
  expand 契约、重放前 parent 再校验（AR016 Major-1）全部不变；
- 工件数字可复现（种子/命令/环境字段齐全）；文档统计数字从工件
  逐值复算（AR016 review 教训固化）。

**假设：**
- 同形验证负载的 `max_l D_l` 跨步近似平稳——AR016 工件为**跨配置**
  均值证据（0.8438↔0.8674，D 相对散布 ~17%），非跨步直接实测；
  抖动由 patience/interval/上限兜底（UT-1604 覆盖极端抖动）；
- 重捕获成本（capture ~几十 ms 量级）在 S 步验证循环内可摊销；
  适配事件表使其可审计（若探针实测摊销不成立，基准如实记录）；
- WDDM + 无锁频下配对交错 + p50 + ≥10 重复足以分辨三路径
  （探针 v3 已在该环境下分辨 42% 级差异）；
- 双图峰值驻留（swap 瞬间）可承受：H=1024/L=16 单 runner 静态
  缓冲 ≈570MB（parent_static 17 面 [8,512,1024]×4B ≈ 285MB +
  child_buf 16 面 ≈ 268MB + logits_buf ≈ 16MB），swap 瞬间双
  runner ≈ 1.1GB；基准三路径实例并存时全负载 < 3GB——仍远低于
  16GB（真实模型场景为后续档评估，风险表记录）。

## 6. 风险

| 风险 | 缓解 |
|------|------|
| 抖动负载下重捕获风暴（每步换图，capture 成本吞掉收益） | patience 连续计数 + 最小间隔 + 总次数上限（UT-1604 钉住）；上限后容量定死退化为固定容量，永不劣于现状 |
| 重捕获失败导致失去可用图 | 先捕新后弃旧（失败时旧图在场）；失败 → 独立标志自适应退役（**禁止置位** `_graph_batched_capture_failed`——其语义是禁用整条批量 graph 路径）+ 独立一次性告警；UT-1605 以 `is` 对象断言 + 重放 spy 钉住"旧 runner 继续服务"（结果对拍无法区分 eager 回退） |
| 双图峰值内存（swap 瞬间 2× 静态缓冲 + 2× graph 池，≈1.1GB@H=1024 几何；基准三路径并存 <3GB） | 远低于 16GB；真实模型场景为后续档评估，风险表记录 |
| 适配期墙钟被 capture 成本污染稳态统计 | 适配事件独立记录；p50 取稳定后重复（FR5）；warmup 覆盖适配期 |
| 策略观测与预算校验的耦合引入额外同步 | FR1 暂存语义（单次读回不变）+ UT-1607 自洽断言 |
| WDDM 墙钟噪声 | 配对交错 + p50 + std + 2σ 判据 + 环境字段；inconclusive 诚实记录 |
| 增长分支改变超限步语义（正确性风险） | FR2 明文：本步丢弃+eager 逐位不动，自适应只影响未来容量；UT-1603 断言超限步结果 == eager |
