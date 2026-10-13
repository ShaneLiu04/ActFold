# [AR014] 需求设计说明书 — Triton 降级粒度细化（P1-11）+ 优先级表过时条目勘误

| 字段 | 内容 |
|------|------|
| AR 编号 | AR014 |
| AR 主题 | fused_ops 三 Triton kernel（merge/gather_select/fused gate）的失败降级从"每 kernel 全局布尔一次失败永久全域禁用"细化为按 (kernel, dtype, hidden shape 类别) 独立降级 + 每键一次性告警；同步勘误指南 P1/M 优先级表 9 条已完成未勾选条目 + M-9 ◐ 勘误注 |
| 关联 SR | 指南 P1-11（:184，本轮勘察确认的真缺口）；指南 6.2 :238（P1-2，勘察证实已落地→勘误）；AGENTS #9（运行时编译失败回退 PyTorch）/ #38（fused gate 降级）；T013 已钉跨 kernel 独立性（test_fused_ops.py:411-490 三例） |
| 日期 | 2026-10-13 |
| 状态 | Final（req 门控首轮 FAIL→修复→复核 PASS（12 项全 RESOLVED）；design 门控首轮 FAIL：Major-1/2 mutation 观测面→修复（UT-1302 spy+集合重定义，srs FR1/FR2/3.4 同步勘误）→复核新 Major-3"全程 forbidden"scoping→修复（区间=①后②前 setattr、③前恢复）→免重读放行；T001 起跑） |

## 1. 背景与目标

AR014 选题勘察（:238 起查）发现：指南债清单的过时不仅限 6.1/6.2
（AR012/AR013 连续两次证实），**P1/P2/M 优先级表同样大面积腐化**——
大量处方已被历次 AR 实质完成但从未勾选。逐项代码取证（本轮勘察，
证据在案）：

| 条目 | 声称 | 现实（勘察证据） | 处置 |
|---|---|---|---|
| :238 / P1-2 | cache.get(token_mask) 双职责，拆 fetch()/fetch_masked() | **已落地**：三 cache 均有 `fetch()`（原始数据视图，activation_cache.py:257）+ `fetch_masked()`（get 兼容别名 ：309-317）；merge/热路径（folded_transformer.py:176/:293/:466、architecture_utils.py:525/:533、cuda_graph.py:333/:340）全走 fetch；`cache.get(token_mask)` **生产零调用方**（全仓 grep 仅 fetch_masked 别名内部转发） | **勘误勾选** |
| P1-1 | cache 双存 layer 输入+输出 → 只存 ffn_out | **已落地**：folded_transformer.py:572-574 每层只 put `{"ffn_out"}`（layer 0 加 `embedding`），gate 跨层读 L-1（:163-167）正是 P1-1 处方 | **勘误勾选** |
| P1-5 | (branch,step) 缓存无限增长 | **已落地**：vectorized_cache.py:63/:66 `_evict_old_branch_steps` ring 驱逐 + folded_generation.py:170-172 prune 时 `clear_branch` | **勘误勾选** |
| P1-6 | verification 重复 parent 前向 | **部分完成（◐）**：处方两半——`cache.contains()` 已落地（verification_engine.py:238 `_ensure_parent_cache` 消费）；"多分支共享一次 parent forward" 批量化主体 = P1-7 仍开放（T004 落笔时按两半定性降为 ◐，比整条勾选诚实） | **◐ 勘误注** |
| P1-8 | samplers 逐行同步 + 逐行 topk | **已落地**：sampling_utils.py:255 `torch.topk(logits, ..., dim=-1)` 批量化 + :162-166 `get_num_transfer_tokens` 全向量化（docstring 自证 "no per-row or per-step host readback"）+ llada_sampler.py:194-197 早停（F10）（T018/F10 系） | **勘误勾选** |
| P1-9 | gate parent norm 每层重算，存 [T] norm 可省 1/3 读带宽 | **前提不成立**：cosine 路径单 pass 融合 `F.cosine_similarity`（similarity_gate.py:114，norm 在同一次归约内，无独立 norm 读）；l2/pearson 同为单 pass（:119/:126-133）；fused gate kernel 寄存器内计算（AGENTS #38）。无"第二次读 parent"可省 | **勘误勾选（失效）** |
| P1-10 | ManualFoldedForward 缺 final norm | **已落地**：AR002 `detect_architecture` final_norm 发现 + 回归测试（AR013 :231 勘误已证） | **勘误勾选** |
| P1-11 | Triton 降级粒度过粗：按 kernel、按 (dtype, shape 类别) 独立降级；失败一次性告警 | **真缺口**：三 kernel 各一个进程级全局布尔（fused_ops.py:154/:155/:492），except 分支翻布尔（:226/:348/:682）→ 一次失败永久禁用该 kernel 的**所有** dtype/shape；粒度仅到 kernel 级（T013 钉），无 dtype/shape 级 | **本 AR 实现（FR1-FR3）** |
| M-3 | 消融是算术推演非实测 | **已落地**：ablation_study.py `disabled_layers` 实测 + cache 扫描强制驱逐（:153/:168，AGENTS #35） | **勘误勾选** |
| M-4 | cost model 硬件常数一刀切 | **部分完成（◐）**：设备名查表（cost_model.py:23-38）+ `HardwareProfile.from_device(calibrate=True)` micro-bench（AGENTS #33）已落地；**O(T²) 项与 KV 带宽项未逐项核实**——残余待独立评估（T004 落笔时按证据面定性降为 ◐，review 门控 Minor-1 同步） | **◐ 勘误注** |
| M-9 | 7 处 torch.manual_seed 全局 RNG 污染 | **部分完成（不整条勾选）**：guide 立项时 actfold 恰 7 处（req 门控 git 考古）——draft_generator.py:64（**guide 点名位置**）已改局部 `torch.Generator`（AR002）；ablation_study.py:43 被 `measure_folding` 的 `get_rng_state`/`set_rng_state` 快照恢复包裹（:181/:244）；**残余 5 处仍裸全局播种**：dream_sampler:106、llada_sampler:120、fast_dllm_sampler:123、diffusion_sampler:108、base_adapter:266 | **勘误注（◐ 部分）**：点名位置已修；残余 5 处为 sampler/adapter 顶层 config.seed 显式播种（eval 复现性语义、官方 recipe 对齐），定性为"显式行为非静默污染"但不勾选整条——整条勾选即复制清单腐化（req 门控 M-2） |

**目标**：(1) P1-11 落地——失败降级粒度细化到 (kernel, dtype,
shape 类别)，告警按键一次性且内容可诊断；(2) 勘误勾选恢复债清单
可信度：**P1 优先级表 7 行**（P1-1/P1-2/P1-5/P1-6/P1-8/P1-9/
P1-10）+ P1-11 本 AR ✅ + **6.2 :238 bullet**（P1-2 的 6.2 侧
表述，与 :175 表行是两处 checkbox）+ **M 表 3 行**（M-3/M-4，
M-9 仅 ◐ 勘误注不勾选）——连续第三个 AR 发现清单腐化，本轮对
P1/M 表做系统性纠偏。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `actfold/core/fused_ops.py`：新增失败注册表 `_TRITON_FAILED_KEYS`
  （模块级 set）与 shape 类别函数；三 kernel 分发/失败路径改造
  （FR1-FR3）；三布尔保留为手动总开关（语义变更：不再被失败自动置位）；
- 新测试文件 `tests/test_triton_fallback_granularity.py`：
  UT-1301~UT-1306（FR4）；
- 既有失败注入钉迁移到新契约（test_fused_ops.py T013 两处、
  test_fused_gate.py T006 一处——断言从"翻布尔"改为"记键"）；
- 文档：指南 **P1 表 7 行**（:175 P1-2、P1-1/P1-5/P1-6/P1-8/
  P1-9/P1-10）勾选+勘误注、P1-11 ✅ 本 AR、**6.2 :238 勘误勾选**、
  **M 表 M-3/M-4 勾选 + M-9 ◐ 勘误注**；第二十一部分回链；
  CHANGELOG AR014 节（Changed + Tests）、AGENTS #9/#38 粒度注；
  fused_ops docstring "permanently disables" 过时表述更正
  （:169-170/:591-592，Green 期顺手，w-4）。

**Out of Scope（本 AR 不做的）：**
- P1-7 多分支批量验证（真遗留主体，中-高难度，独立 AR）；
- P1-12 tuple 契约模板/KV 丢弃告警（部分由 AGENTS #13 re-wrap
  覆盖，残余留待独立评估）；
- P2-4 baseline KV cache、P3-2 paged 多祖先、P3-3 真 draft
  checkpoint（大项池）；
- :237 FoldedModel context manager（废弃路径低值）、:243 Windows
  脚本（低值）；
- shape 类别**不**做逐 shape 精确键（注册表无界增长风险；处方
  明示"类别"）；不做多 GPU/device 维度键（单进程单卡场景，
  现行布尔同样无 device 维度）；
- 不改变任何 kernel 的数学语义/调度阈值（D3/B*T 阈值、dtype 白名单
  原样）。

## 3. 功能需求

### 3.1 失败注册表与分发改造（FR1）

**描述：** fused_ops.py 三个 Triton 入口（`_merge_stable_divergent_triton`
:158、`gather_select` :277、`fused_gate_mask_count` :571）的分发与
失败路径统一改造。

**验收标准：**
- 模块级 `_TRITON_FAILED_KEYS: set[tuple[str, str, int]]`，键 =
  `(kernel_name, dtype_name, hidden_bucket)`：
  - merge：`("merge", str(parent_ffn.dtype), bucket(hidden_dim))`；
  - gather_select：`("gather_select", str(child_out.dtype), bucket(hidden_dim))`；
  - gate：`("gate", str(h_child.dtype), bucket(hidden_dim))`；
- `bucket(h) = 1 << (max(h,1)-1).bit_length()`（pow2 上取整：
  4096→4096、5000→8192、1→1）；以模块级纯函数暴露供直接钉；
- 分发序：`_HAS_TRITON` → 手动总开关布尔 → 既有 device/dtype
  白名单/阈值检查原样保留；注册表检查**在键可得处插入**（design
  门控 m-3：merge 的 hidden_dim 于 shape 解包后才可得，检查点在
  解包后；gather_select/gate 在 use_triton 评估前——两序无
  observable 差异，注册键只能来自已通过白名单的失败）；
- 失败路径：`except Exception` → 若键未注册则
  `_TRITON_FAILED_KEYS.add(key)` 并告警（FR2），**不再置位任何
  布尔** → 返回 PyTorch 回退（回退调用与现行完全一致）；
- 三布尔（`_TRITON_MERGE_DISABLED`/`_TRITON_GATHER_SELECT_DISABLED`/
  `_TRITON_GATE_DISABLED`）保留且语义收窄为**手动总开关**（True =
  整 kernel 强制回退，优先于注册表；monkeypatch 调试用法不变）。

### 3.2 每键一次性告警与可诊断内容（FR2）

**验收标准：**
- 同键首次失败：一条 `RuntimeWarning`（`stacklevel=3`——共享
  helper 多一层帧，落点对齐现行 stacklevel=2 的调用方位置；
  design D3，design 门控 m-2 勘误），消息
  含 kernel 名、dtype、hidden bucket、原始异常（如
  `Triton merge kernel unavailable (dtype=torch.float32,
  hidden-class=4096): ...; falling back ...`）；
- **既有告警 match 子串必须保留**（既有钉契约，w-1）：merge
  "merge kernel unavailable"、gather_select "gather/select kernel
  unavailable"、gate "fused gate kernel unavailable"，以及异常
  文本透传（T013 match="boom" 依赖）；
- 同键后续调用：静默回退（不告警、不尝试 kernel）；
- 不同键首次失败：各自一条告警（互不抑制）。

### 3.3 隔离语义（FR3）

**验收标准：**
- (kernel, dtype, bucket) 键失败**不**影响：
  同 kernel 不同 dtype、同 kernel 不同 bucket、不同 kernel 任意键
  ——后者由 T013 既有钉覆盖（更新断言后必须仍然成立）；
- 既有跨 kernel 独立性钉（test_fused_ops.py **:411-490 三例**，
  含 flags_independent 与两例互向独立性，req 门控 m-1）迁移到
  新契约后语义不变：merge 失败后 gather_select 仍尝试 Triton，
  反之亦然；
- `test_cuda_graph.py:596`（graph 路径后 `_TRITON_GATE_DISABLED
  is False`）零改动通过——失败不置位布尔是新语义的自然推论。

### 3.4 新测试钉（FR4，UT-1301~1306）

**验收标准（全部 CUDA+Triton 门控，参照既有 T013 skipif 先例；
新文件 autouse fixture 隔离注册表——每测试用 fresh set）：**
- **UT-1301 dtype 隔离**：dtype 条件爆炸 kernel（仅 float32 raise）
  → float32 首调告警+回退正确；float16 同形状调用仍走 Triton
  （真实 kernel，结果与 torch 参考逐位一致）且注册表仅含
  float32 键；**禁止回退手段**（monkeypatch torch 回退函数为
  forbidden，T013 :273/:379 先例）证明 float16 路径确走 Triton
  而非静默回退（w-2）；**追加第二失败键**（如 float64 dtype
  白名单外的合法回退不适用，改用第二爆炸 dtype 或同 dtype 不同
  bucket 失败）→ 第二条告警独立出现（互不抑制等价类，req 门控
  m-4）；
- **UT-1302 同键静默**：爆炸 kernel 首调 `pytest.warns` 捕到
  RuntimeWarning；次调前换 **launch-count spy kernel**（计数+委托
  真 kernel）→ 次调断言 spy 计数==0（"不尝试 kernel"直接钉）+
  `warnings.catch_warnings(record=True)` 零告警；两次回退结果均
  与 torch 参考逐位一致；
- **UT-1303 shape 类别隔离**：同 dtype，H=512 爆炸 → H=2048
  （不同 bucket）仍走真实 Triton（forbidden 回退手段同 UT-1301）
  逐位一致；注册表仅含 512 键；
- **UT-1304 总开关回退兼容**：手动置 `_TRITON_MERGE_DISABLED=True`
  → 未注册键也强制回退（结果正确、零告警、注册表零增长）；
  monkeypatch 退出自动还原（既有先例 test_ar002_perf_proxy.py:365）；
- **UT-1305 告警可诊断性**：首失败告警文本含 kernel 名、
  `torch.float32`、bucket 值三要素（子串断言）；
- **UT-1306 bucket 纯函数钉**（CPU 可跑）：4096→4096、5000→8192、
  512→512、1→1、2→2。
- **行为钉子型 AR 纪律**：Red 期对 FR1-3 契约做 mutation
  spot-check 自证，集合（design 门控 M-1/M-2 重定义）：
  ①注册表不写入 / ②分发不查注册表（两者观测面=UT-1302 次调
  launch-count spy，计数==1 翻红）/ ③键缺 dtype 维度（观测面
  =UT-1301 ② forbidden 回退 AssertionError）；"告警去条件"
  变异黑盒不可达（dispatch 使 helper 对已注册键不可达），
  从集合移除并记录（guard 保留为纵深防御）。

### 3.5 既有钉迁移（FR5）

**验收标准：**
- test_fused_ops.py T013 失败注入两处：`assert _TRITON_*_DISABLED
  is True` → `assert ("...", dtype, bucket) in fused_ops.
  _TRITON_FAILED_KEYS`（且布尔保持 False）；第二次静默断言保留；
- test_fused_gate.py T006 失败注入一处同型迁移，**out_count 累加
  断言原样保留**（gate 失败回退路径正确性的唯一钉，w-3）；
- **迁移用例补注册表隔离**：三迁移用例的键断言要求先快照/恢复
  `_TRITON_FAILED_KEYS`（或共用新文件的同型 fixture）——否则键
  永久残留进程级 set，与既有 CUDA 用例（如 test_cuda_graph.py:576
  hidden=64 fp32 同键）字母序侥幸隔离，后续同键用例的 Triton
  覆盖会被静默吞噬（req 门控 m-3 实证风险）；
- 旗标存在性钉（test_fused_ops.py:312-315、test_fused_gate.py:203）
  保留并补 `_TRITON_FAILED_KEYS` 存在性；
- 其余消费点（test_ar002_perf_proxy.py:365、test_cuda_graph.py:596、
  test_fused_gate.py:217、**test_fused_ops.py:440-461**——boom
  注入 gather + forbidden-fallback 钉 merge 仍走 Triton，新语义
  零改动自然通过，req 门控 m-2）**零改动**通过。

### 3.6 文档收口（FR6）

**验收标准：**
- 指南 **P1 优先级表**：P1-1/P1-5/P1-8/P1-9（失效勘误）/
  P1-10/**P1-2（:175 表行——与 6.2 :238 是两处 checkbox，均须
  勾选，req 门控 M-1）** 勾选+证据注；P1-11 ✅ 本 AR；
  P1-6 **◐**（contains 已落地，批量化主体归 P1-7 仍开放）；
- 指南 **6.2 :238** 勘误勾选（P1-2 已落地）；指南 :35 过时旗标
  表述一并更正（w-4）；
- 指南 **M 表**：M-3 勾选+证据注；M-4 **◐**（设备表/校准已落地，
  O(T²)/KV 带宽项未逐项核实——review 门控 Minor-1 同步）；
  M-9 **不勾选**，仅 ◐ 勘误注（点名位置已修 + 残余 5 处定性，
  req 门控 M-2）；
- **落笔期偏离披露（review 门控 Minor-2 补记）**：P1-3（gather_select
  主路径接入）为 T004 顺手勘误勾选——不在本 srs 立项清单内，证据
  AGENTS #30/T016/AR011 在案；P1-6 从"勾选"降为 ◐（处方两半，
  批量化主体归 P1-7）、M-4 从"勾选"降为 ◐——三处均以代码证据
  为准回写 §1 表；
- 指南第二十一部分 AR014 回链；CHANGELOG AR014 节（Changed +
  Tests，Unreleased 最前）；
- AGENTS.md #9/#38 补粒度注（失败降级按 (kernel, dtype, hidden
  bucket) 键独立，布尔仅为手动总开关——不再被失败自动置位）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 静态检查 | mypy 标准命令 clean（65 文件基线）/ pyflakes 零 / 100 列 |
| 质量 | 全量回归 | `pytest tests\ -q -m "not slow"`：807 passed 基线 + 新增/迁移零失败 |
| 稳定 | 测试确定性 | 新测试无统计容差断言（逐位一致）；注册表隔离 fixture 防跨测试污染 |
| 兼容 | 公开 API | 三布尔保留（monkeypatch 调试面不变）；新增注册表/桶函数为模块级可钉属性；分发顺序后段（device/dtype/阈值）不变 |
| 性能 | 分发开销 | 热路径新增一次 set 哈希查找（O(1)，键为 str/str/int）；无额外同步/拷贝 |
| 性能 | 用例耗时 | 新文件 CUDA 用例合计 <20s（真实 kernel 编译缓存后）；CPU 用例（UT-1306）<0.1s |

## 5. 约束与假设

**约束：**
- `actfold/` 改动仅 fused_ops.py 一个生产文件；kernel 数学语义、
  dtype 白名单、B*T/D3 阈值、回退实现逐行不变；
- 告警行为变更（每 kernel 一条 → 每键一条）是**有意契约变更**，
  必须在 CHANGELOG Changed 节明示；
- 既有测试除 FR5 列举三处外零改动。

**假设：**
- hidden bucket 维度足以刻画"shape 类别"：三 kernel constexpr
  均为常量（BLOCK 128/128/1024），编译失败主要由 (dtype, 尾掩码
  路径) 驱动、hidden 决定尾掩码形态；seq/batch 仅影响 grid 大小
  （launch 失败极罕见，接受其归入同 bucket 命运——类别本就是粗
  粒度防护网而非精确分类器）；
- 单卡场景无 device 维度键必要（与现行布尔一致）；多卡部署若
  出现 per-device 失败差异，属后续扩展（注册表键结构预留
  tuple，扩展为 4 元素不破坏现有 API）；
- Triton import 失败（`_HAS_TRITON=False`）路径不受影响（分发首
  检不变）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 降级粒度 | 一次 kernel 失败后回退决策的作用域——现行 kernel 级（全局布尔），本 AR 细化为 (kernel, dtype, hidden bucket) 键级 |
| hidden bucket | hidden_dim 的 pow2 上取整类别（`1 << (max(h,1)-1).bit_length()`）——粗粒度 shape 类别，防注册表无界增长 |
| 手动总开关 | 保留的三个模块布尔（`_TRITON_*_DISABLED`）——仅人工置位生效（调试/测试强制回退），失败路径不再自动置位 |
| 键一次性告警 | 每个失败键仅在首次失败时告警一次；同键后续静默回退 |
| 债条目腐化 | 指南条目前提被后续 AR 实质关闭（或失效）但未勾选——AR008-013 连续六次发现，本 AR 扩展到 P1/M 优先级表 |
