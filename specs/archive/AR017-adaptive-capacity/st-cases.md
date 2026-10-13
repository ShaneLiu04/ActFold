# [AR017] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR017 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-14 |

## 测试用例列表

### ST-001：观测接缝暂存语义（FR1）

**关联需求：** srs.md §3.1 FR1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- CUDA 可用；`FoldedGraphRunner` 已捕获；自适应开启与否均可

**测试步骤：**
1. 多步重放后读取 `runner.last_layer_divergences`
2. 断言其为长度 L 的 list、`max(...) ≤ N·T`、与 `count_buf` 自洽（`D_l = N·T - count_l`）
3. 未校验步（prefill 失败）暂存为陈旧值/None 语义不被消费

**期望结果：**
- UT-1607 全绿（暂存值自洽、随负载变化非陈旧）

**实际结果：** UT-1607 通过（本日新跑 `pytest tests\test_adaptive_capacity.py`：8 passed in 8.54s）

**状态：** PASS

---

### ST-002：单次读回与 capture 体纪律（FR1）

**关联需求：** srs.md §3.1 FR1（验收：D2H 同步次数不变；capture 体零改动）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- `cuda_graph.py` 改动仅限 FR1 暂存

**测试步骤：**
1. 复核 `validate_budgets` 内单次 `count_buf.tolist()` 复用（策略读暂存，无二次同步）
2. 复核 `_static_forward` capture 体零改动、零新增 host 同步
3. 全量回归确认既有 `test_cuda_graph` 与 AR016 UT 零改动全绿

**期望结果：**
- 单 tolist 复用；capture 体未触碰；既有测试零失败

**实际结果：** review 门控（ses_eca032b0…）逐项确认单 tolist 复用与 capture 体未触碰；全量回归 846 passed（T006）

**状态：** PASS

---

### ST-003：策略纯函数表驱动（FR2）

**关联需求：** srs.md §3.2 FR2
**测试类型：** 正常路径 + 边界条件
**优先级：** High

**前置条件：**
- 策略模块无 CUDA 依赖（CPU 可测）

**测试步骤：**
1. 表驱动断言：增长（超限→目标含 grow_factor 余量、clamp 生效）、收缩（patience 计数、β 触发、恒小于当前 C、C≤1 no-op）、节流（interval 内 no-op、达上限退役）、方向单调、非法配置构造抛错

**期望结果：**
- 给定观测序列与配置，动作序列确定且符合全部验收标准

**实际结果：** UT-1601 通过（本日新跑 8 passed）

**状态：** PASS

---

### ST-004：配置字段流入与边界校验（FR2）

**关联需求：** srs.md §3.2 FR2（`ActFoldConfig` 新字段 + 构造时校验）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- `actfold/utils/config_manager.py` 含 6 个新字段

**测试步骤：**
1. 默认构造读取 6 字段默认值
2. 构造 `graph_adaptive_grow_factor=0.9`（<1）→ 期望 ValueError
3. 构造 `graph_adaptive_shrink_threshold=0.0`（非法）→ 期望 ValueError
4. 构造 `graph_adaptive_max_recaptures=0`（<1）→ 期望 ValueError

**期望结果：**
- 默认值 `False/1.25/0.5/3/2/8`；三个非法值均抛 ValueError

**实际结果：** ST 探针执行（本日）：defaults `False 1.25 0.5 3 2 8`；三非法构造均 ValueError → config-boundary: PASS

**状态：** PASS

---

### ST-005：默认关闭零行为（FR2/FR3/NFR 兼容）

**关联需求：** srs.md §3.2 FR2 + §3.3 FR3（默认关闭）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- `graph_adaptive_capacity=False`（默认）

**测试步骤：**
1. 多步重放后断言：runner 对象恒不变、容量不变、零告警、零策略状态

**期望结果：**
- AR016 行为逐位一致

**实际结果：** UT-1606 通过（本日新跑）；全量回归 846 passed 零回归（T006）

**状态：** PASS

---

### ST-006：增长端到端（FR3）

**关联需求：** srs.md §3.3 FR3（增长评估调用点）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 初始 ratio 极小（0.02），自适应开启

**测试步骤：**
1. 首步预算超限：断言本步结果 == eager（语义不变）+ 一次性告警
2. 自适应重捕获增长；第二步起重放预算通过且 logits 对拍一致
3. swap 后构造高发散候选再次超限 → 断言零新预算告警（`_graph_batched_budget_warned` 未因换 runner 复位）

**期望结果：**
- 本步丢弃语义逐位不变；增长只影响未来容量；告警标志不复位

**实际结果：** UT-1603 通过（本日新跑）

**状态：** PASS

---

### ST-007：收缩端到端与观测窗口契约（FR3）

**关联需求：** srs.md §3.3 FR3（收缩评估调用点 + 窗口契约）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 负载 D/(N·T)≈0.15，初始 ratio=0.5 + 自适应开启

**测试步骤：**
1. patience 步后断言 `graph_runner_batched` 换新对象且容量收缩
2. 断言 `capacity < 原容量` 且 `capacity ≥ max_l D_l`
3. 收缩后 logits 与 eager 批量对拍 allclose(1e-4) + 逐行 argmax 相等；预算持续通过

**期望结果：**
- 收缩重捕获后行为与 eager 一致

**实际结果：** UT-1602 通过（本日新跑）

**状态：** PASS

---

### ST-008：重捕获失败安全网（FR3）

**关联需求：** srs.md §3.3 FR3（先捕新后弃旧 + 独立退役标志）
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- monkeypatch `FoldedGraphRunner.capture` 注入一次性失败（沿 UT-1507 时序）

**测试步骤：**
1. 失败后断言：一次性告警 + 独立退役标志置位 + `_graph_batched_capture_failed` 保持 False
2. 后续合格步 `graph_runner_batched` 恒为旧实例（`is` 断言）+ 重放 spy 计数递增 + 结果与 eager 对拍一致

**期望结果：**
- 旧 runner 继续服务；批量 graph 路径未被连禁；正确性零影响

**实际结果：** UT-1605 通过（本日新跑）

**状态：** PASS

---

### ST-009：抖动负载上限退役（FR3）

**关联需求：** srs.md §3.3 FR3（节流 + max_recaptures）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- 交替高低 D 负载序列逼近上限

**测试步骤：**
1. 断言重捕获次数 ≤ `max_recaptures`
2. 达上限后容量不再变化 + 一次性退役告警
3. 结果全部与 eager 对拍

**期望结果：**
- 上限后容量定死，永不劣于现状

**实际结果：** UT-1604 通过（本日新跑）

**状态：** PASS

---

### ST-010：串行路径零干扰（FR3）

**关联需求：** srs.md §3.3 FR3（串行 runner 零干扰）
**测试类型：** 回归测试
**优先级：** Medium

**前置条件：**
- 同实例批量自适应多次重捕获

**测试步骤：**
1. 断言串行 `graph_runner` 对象与容量不受批量侧影响，行为与 AR016 UT 一致

**期望结果：**
- 串行路径语义零变化

**实际结果：** UT-1608 通过（本日新跑）

**状态：** PASS

---

### ST-011：mutation spot-check（FR4）

**关联需求：** srs.md §3.4 FR4（UT-1609）
**测试类型：** 回归测试
**优先级：** Medium

**前置条件：**
- 全部 UT 绿后执行变异

**测试步骤：**
1. ①删策略调用点 → UT-1602 红；②删增长分支 → UT-1603 红；③删上限退役 → UT-1604 红；④先弃旧后捕新 + capture 失败注入 → UT-1605 红
2. 全验证后 SHA256 逐字节还原

**期望结果：**
- 四变异全部击杀对应 UT；还原后哈希与基线一致

**实际结果：** T002 执行记录：四变异全红（UT-1602/1603/1604/1605 各自击杀）；SHA256 还原（architecture_utils `196E87…F3024` / adaptive_capacity `10A9CB…BCA15`）；还原后 8/8 绿

**状态：** PASS

---

### ST-012：基准工件完整性（FR5）

**关联需求：** srs.md §3.5 FR5
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- `results/optimization/ar017_adaptive_bench.json` 存在（run D）

**测试步骤：**
1. 校验 meta：复现命令（`python -m scripts.ar017_adaptive_bench --reps 30`）、种子 2026、驱动 556.18、CUPTI 探测 None + 注记、交错声明（round-robin per repetition）、warmup=5（≥ patience+1=4）、reps=30（≥10）
2. 校验 configs：4 配置 × 三路径（baseline_eager/graph_fixed/graph_adaptive），timings 含 mean/std/p50/n
3. 校验 adaptation 记录：recapture_events + capacity_trajectory + observed_max_layer_divergence 在场

**期望结果：**
- 全部环境/方法/工件字段齐全，n=30 单次高功效运行

**实际结果：** ST 探针执行（本日）：4 配置 reps=30/warmup=5、三路径 timings 齐、`adaptive_state` 含 recapture_events(1)/capacity_trajectory/final_capacity/observed 分布、meta 全字段在场 → PASS

**状态：** PASS

---

### ST-013：基准正确性随行（FR5）

**关联需求：** srs.md §3.5 FR5（正确性随行）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 同上工件

**测试步骤：**
1. 校验 4 配置 `logits_parity_argmax_exact` 全 True（三路径 argmax 一致）
2. 校验 `adaptive_vs_fixed_allclose_1e-4` 全 True
3. 校验容量贴合需求：`capacity_tracks_demand` 全 supported/supported 判定、最终容量 vs 观测 D

**期望结果：**
- 三路径 logits 一致；自适应容量贴合观测需求

**实际结果：** ST 探针执行（本日）：parity=True×4、allclose=True×4、tracks_demand=True×4（H4 supported×4）；容量 1024→340（需求 272）等与文档一致

**状态：** PASS

---

### ST-014：假设与结果分离（FR5）

**关联需求：** srs.md §3.5 FR5（假设判定基于实测）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- 同上工件

**测试步骤：**
1. 校验 `hypotheses_declared`（H1-H4 docstring 声明）与 `hypotheses_verdicts` 分离在场
2. 校验逐配置 H1/H2 判定与 p50 加速比自洽：(256,8,4) 1.5502/1.1446 双 inconclusive；(1024,16,4) 2.2993/1.655 双 supported；(1024,16,8) 2.4609/1.7091 双 supported
3. 校验 P1-7 整体判定诚实记录（inconclusive，非结果选购）

**期望结果：**
- 假设声明与实测判定分离；整体判定严格聚合不选购

**实际结果：** ST 探针执行（本日）：declared H1-H4 齐；per-config H1/H2 与速度比自洽；global `H1/H2 inconclusive、H3/H4 supported`；p1_7 overall=inconclusive（per-config 3/4 supported 已注记）→ PASS

**状态：** PASS

---

### ST-015：文档收口（FR6）

**关联需求：** srs.md §3.6 FR6
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- T005 文档改动完成

**测试步骤：**
1. 指南 P1-7 行（:180）追注第三档「自适应容量 ✅ AR017」+ 第二十四部分在场 + Part 23/24 遗留交叉注记
2. CHANGELOG AR017 节置 Unreleased 最前
3. AGENTS #37 扩注（AR017 自适应容量契约）

**期望结果：**
- 三文档全部在场且数字从工件逐值重算

**实际结果：** ST 探针执行（本日）：guide-P1-7-AR017=True、guide-part24=True、交叉注记=True、changelog-AR017 且先于 AR016=True、agents-37-AR017=True；数字经 review 门控独立复核 → PASS

**状态：** PASS

---

### ST-016：质量门与回归（NFR）

**关联需求：** srs.md §4 非功能需求
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- 全部代码与文档改动就绪

**测试步骤：**
1. `python -X utf8 -m mypy actfold --ignore-missing-imports` → clean（66 文件）
2. pyflakes 零 + 100 列
3. 全量 `pytest tests\ -q -m "not slow"` → 846 passed（838 基线 + 8 新增）
4. demo 三指标 85.5%/2.35e-03/93.75% 精确一致

**期望结果：**
- 全部质量门通过，零回归

**实际结果：** T006 review 修复后重跑：mypy clean 66 文件、pyflakes 零、100 列、846 passed、demo 三指标精确一致 → PASS

**状态：** PASS

---

## 执行摘要

| 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|
| 16 | 16 | 0 | 0 |

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-14 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR1 | 观测接缝：单次读回内暴露逐层 D | ST-001, ST-002 | PASS |
| §3.2 FR2 | 自适应策略（纯函数 + 配置驱动，默认关闭） | ST-003, ST-004, ST-005 | PASS |
| §3.3 FR3 | 重捕获生命周期（批量侧） | ST-006, ST-007, ST-008, ST-009, ST-010, ST-005 | PASS |
| §3.4 FR4 | 对拍与观测钉（UT-1601~1609） | ST-011（+ST-001~010 复用 UT） | PASS |
| §3.5 FR5 | 几何扫描基准 | ST-012, ST-013, ST-014 | PASS |
| §3.6 FR6 | 文档收口 | ST-015 | PASS |
| §4 NFR | 质量/兼容/性能非功能需求 | ST-002, ST-005, ST-016 | PASS |

**需求覆盖率：** 6 / 6（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 8 | 8 | 0 | 0 |
| 边界条件 | 5 | 5 | 0 | 0 |
| 异常处理 | 1 | 1 | 0 | 0 |
| 回归测试 | 2 | 2 | 0 | 0 |
| **合计** | **16** | **16** | **0** | **0** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | (256,8,4) 小几何 H2 判定 inconclusive（WDDM 展布致 2σ 检验不过，效应方向真实且跨三次运行稳定） | 已诚实记录于工件/文档（Part 24 注记），非缺陷，不阻塞 |
| Minor | 串行 graph 路径自适应为后续档（srs §1.3 明确不在范围） | 延后处理（后续档评估） |
| Minor | 需求样本 n=34=35−1（换图步新 runner 未 validate 不采样） | 已注记于工件/文档，诚实缺口 |

### 结论

> **Go** — 16/16 用例 PASS，需求覆盖 6/6（100%），无 Critical/Major 缺陷，846 passed 零回归，质量门全绿，NFR 全部达标；3 项 Minor 均为已记录的诚实边界/后续档项，不阻塞归档。（用户预授权：全流程无需逐步确认，ST Go 自动执行）
