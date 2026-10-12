# [AR009] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR009 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-12 |

## 测试用例列表

### ST-001：同一 engine 实例的代理回退恰好告警一次

**关联需求：** srs.md §3.1 验收①
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- stub adapter 无 folded 前向（走 embedding 代理回退）

**测试步骤：**
1. 构造 engine，`recwarn` 下连续调用 `verify_branch` 两次

**期望结果：**
- 恰好 1 条含 "embedding-level proxy" 的 UserWarning；两次 stable_ratio 均逐位等于独立计算的 gate 均值

**实际结果：** UT-801 passed（`assert len(...) == 1` 成立；返回值 exact-equal True）

**状态：** PASS

---

### ST-002：实例级 once 语义（新实例重新告警）

**关联需求：** srs.md §3.1 验收②
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- 两个独立同参 engine 实例

**测试步骤：**
1. 各调用 `verify_branch` 一次，统计含 "embedding-level proxy" 的 UserWarning 总数

**期望结果：**
- 共 2 条（实例级而非全局级 once）

**实际结果：** UT-802 passed（`assert len(...) == 2` 成立）

**状态：** PASS

---

### ST-003：层级 profile 路径零告警

**关联需求：** srs.md §3.1 验收③
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- monkeypatch `GLOBAL_STABILITY_PROFILER.get_profile` 返回含 1 条 `LayerStabilityStats(stable_ratio=0.5)` 的真实 `StabilityProfile`

**测试步骤：**
1. 调用 `verify_branch` 一次

**期望结果：**
- 零 UserWarning；stable_ratio == 0.5（profiler 均值路径）

**实际结果：** UT-803 passed（零告警 + 0.5 精确返回）

**状态：** PASS

---

### ST-004：既有套件零回归（告警不升级为错误）

**关联需求：** srs.md §3.1 验收④ / §4 NFR 全量回归
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- pyproject 无 `-W error`（已核实）

**测试步骤：**
1. `python -X utf8 -m pytest tests\ -q -m "not slow"`

**期望结果：**
- 773 passed（768 基线 + 5 新增）零失败；既有 engine 测试（UT-306/307 等走代理路径）不受新告警影响

**实际结果：** 773 passed, 3 skipped, 3 deselected（44s）——零回归

**状态：** PASS

---

### ST-005：`FoldedTransformerLayer` 守卫触发

**关联需求：** srs.md §3.2 验收①
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- `(AdaptiveQuantileGate, FoldingScheduler())` 组合

**测试步骤：**
1. 构造 `FoldedTransformerLayer`

**期望结果：**
- RuntimeError；消息含 `AdaptiveQuantileGate`、`FoldingScheduler`、`gate.tau`

**实际结果：** UT-804 passed（`match="gate.tau"` + 两类名断言全过）

**状态：** PASS

---

### ST-006：`SplitFoldedTransformerLayer` 继承守卫

**关联需求：** srs.md §3.2 验收②
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- 同 ST-005 组合

**测试步骤：**
1. 构造 `SplitFoldedTransformerLayer`

**期望结果：**
- 同样 RuntimeError（super 传递覆盖证据）

**实际结果：** UT-804 passed（两 layer 类在同一循环内断言）

**状态：** PASS

---

### ST-007：`ActFoldVerificationEngine` 守卫触发

**关联需求：** srs.md §3.2 验收③
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- 同组合 + UT-306 stub adapter 模式

**测试步骤：**
1. 构造 `ActFoldVerificationEngine`

**期望结果：**
- 同样 RuntimeError

**实际结果：** UT-805 passed

**状态：** PASS

---

### ST-008：合法组合零影响钉死

**关联需求：** srs.md §3.2 验收④⑤
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- `(SimilarityGate, FoldingScheduler())` 与 `(AdaptiveQuantileGate, None)` 两组合

**测试步骤：**
1. 分别构造 FoldedTransformerLayer / SplitFoldedTransformerLayer / ActFoldVerificationEngine

**期望结果：**
- 全部构造成功（五种合法构造）

**实际结果：** UT-804/805 passed（合法对照循环全过）

**状态：** PASS

---

### ST-009：文档收口三件套

**关联需求：** srs.md §3.3
**测试类型：** 正常路径
**优先级：** Medium

**测试步骤：**
1. 核对 docs/OPTIMIZATION_GUIDE.md :239/:240 勾选 + :240 勘误注 + 第十六部分
2. 核对 AGENTS.md #29 修订（代码契约）
3. 核对 CHANGELOG.md AR009 节

**期望结果：**
- 三处均按 srs 3.3 验收落地

**实际结果：** 三处均已落地（:239 ✅ AR009 构造期守卫 / :240 ✅ 勘误 + 可见性收口 / 第十六部分回链 / AGENTS #29 "Enforced in code (AR009)" / CHANGELOG Added+Changed 两节）

**状态：** PASS

---

### ST-010：质量门与 demo 基线

**关联需求：** srs.md §4 非功能需求
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `mypy actfold --ignore-missing-imports` / `pyflakes actfold tests demo.py` / 100 列检查
2. `python -X utf8 demo.py`

**期望结果：**
- mypy clean（65 文件）/ pyflakes 零 / 100 列 clean；demo 85.5% / 2.35e-03 / 93.75% 精确一致

**实际结果：** 全部达标（Success: no issues found in 65 source files；pyflakes exit 0；改动文件 ≤100 列；demo 三指标精确一致）

**状态：** PASS

---

## 执行摘要

| 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|
| 10 | 10 | 0 | 0 |

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-12 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 | embedding 代理回退一次性告警 | ST-001, ST-002, ST-003, ST-004 | PASS |
| §3.2 | AdaptiveQuantileGate×FoldingScheduler 构造期拒绝 | ST-005, ST-006, ST-007, ST-008 | PASS |
| §3.3 | 文档收口 | ST-009 | PASS |
| §4 | 非功能需求（静态检查/回归/demo） | ST-004, ST-010 | PASS |

**需求覆盖率：** 4 / 4（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 4 | 4 | 0 | 0 |
| 边界条件 | 3 | 3 | 0 | 0 |
| 异常处理 | 3 | 3 | 0 | 0 |
| 回归测试 | 2 | 2 | 0 | 0 |
| **合计** | **10** | **10** | **0** | **0** |

覆盖率说明：design.md 未声明覆盖率目标（沿用全量回归 + 静态门）；全量 773 passed 覆盖守卫与告警全部新分支（UT-801~805 + 合法对照）。

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| （无） | — | — |

### 结论

> **Go**：10/10 用例 PASS，需求覆盖 4/4（100%），无 Critical/Major/Minor 缺陷，全量 773 passed 零回归，demo 与静态门全绿。按既定全自动指令执行归档与推送。
