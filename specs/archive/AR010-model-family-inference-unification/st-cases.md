# [AR010] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR010 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-12 |

## 测试用例列表

### ST-001：推断规则等价（去重不改行为）

**关联需求：** srs.md §3.1 验收②
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 生产代码已委托化（registry 侧规则删除）

**测试步骤：**
1. 直调 `_resolve_family` 过六组等价矩阵（GPT-LLaDA-8B / FastDLLM-v2 / some\/dream-7b / gpt2-auto / gpt2-explicit / anything+dream 透传）

**期望结果：**
- 返回值与改动前逐位一致（混合大小写/连写覆盖）

**实际结果：** UT-901 六参数化 passed

**状态：** PASS

---

### ST-002：既有 resolve_family 断言零改动全过

**关联需求：** srs.md §3.1 验收①
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. 运行既有 `test_registry_resolve_family`（test_models.py:25-30，5 断言零改动）

**期望结果：**
- 全过（去重的行为不变硬门）

**实际结果：** 5 断言零改动全过（19 passed 中含 10 项 test_models）

**状态：** PASS

---

### ST-003：auto 回退告警触发

**关联需求：** srs.md §3.2 验收①
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- `"gpt2"` + `model_family="auto"`，dummy 注册表注入

**测试步骤：**
1. `ModelRegistry.load`，`pytest.warns` 捕获

**期望结果：**
- 恰好 1 条 UserWarning，消息含 "auto-detect fell back to causal_lm" 与 "gpt2"；dummy 以 causal_lm 家族类构造（回退行为不变）

**实际结果：** UT-902 passed（match 短语 + 模型名双断言 + `len(record)==1`）

**状态：** PASS

---

### ST-004：显式 family 零告警钉死

**关联需求：** srs.md §3.2 验收②
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. `"gpt2"` + 显式 `"causal_lm"`，`simplefilter("error")` 下 load

**期望结果：**
- load 成功，零 UserWarning（任何告警即翻红）

**实际结果：** UT-903 passed

**状态：** PASS

---

### ST-005：已知子串命中零告警钉死

**关联需求：** srs.md §3.2 验收③
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. `"some/llada-8b"` + auto，`simplefilter("error")` 下 load

**期望结果：**
- load 成功，零 UserWarning；解析为 llada dummy

**实际结果：** UT-904 passed

**状态：** PASS

---

### ST-006：utils.py 零改动声明

**关联需求：** srs.md §3.1（唯一实现源 + 纯函数约束）
**测试类型：** 回归测试
**优先级：** Medium

**测试步骤：**
1. `git diff -- actfold/models/utils.py` 应为空；规则逐字核对

**期望结果：**
- utils.py 零改动（规则逐字保留；告警在 registry 层）

**实际结果：** diff 为空（review 核实：规则逐字一致、零 actfold 内部导入无环）

**状态：** PASS

---

### ST-007：全量回归零失败

**关联需求：** srs.md §3.2 验收④ / §4 NFR
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\ -q -m "not slow"`

**期望结果：**
- 782 passed（773 基线 + 9 新增）零失败；既有 gpt2-auto 直调用例（test_models.py:29）在告警下仍绿

**实际结果：** 782 passed, 3 skipped, 3 deselected（44s）——零回归

**状态：** PASS

---

### ST-008：文档收口三件套

**关联需求：** srs.md §3.3
**测试类型：** 正常路径
**优先级：** Medium

**测试步骤：**
1. 核对指南 :241 勾选 + 子串勘误注 + 第十七部分
2. 核对 CHANGELOG AR010 节
3. 核对 AGENTS "Adding a New Model Family" 指针（含 per-load 措辞勘误）

**期望结果：**
- 三处按 srs 3.3 验收落地

**实际结果：** 三处落地（:241 ✅ 去重+可见性+勘误 / 第十七部分回链 / CHANGELOG Changed+Added / AGENTS 唯一实现源 + causal_lm 不变量）

**状态：** PASS

---

### ST-009：质量门与 demo 基线

**关联需求：** srs.md §4 非功能需求
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. mypy / pyflakes / 100 列检查
2. `python -X utf8 demo.py`

**期望结果：**
- mypy clean（65 文件）/ pyflakes 零 / 100 列 clean；demo 85.5% / 2.35e-03 / 93.75% 精确一致

**实际结果：** 全部达标（Success: no issues found in 65 source files；pyflakes exit 0；改动文件 ≤100 列；demo 三指标精确一致）

**状态：** PASS

---

## 执行摘要

| 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|
| 9 | 9 | 0 | 0 |

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-12 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 | 家族推断单一实现源（去重） | ST-001, ST-002, ST-006 | PASS |
| §3.2 | auto 回退 causal_lm 一次性告警 | ST-003, ST-004, ST-005, ST-007 | PASS |
| §3.3 | 文档收口 | ST-008 | PASS |
| §4 | 非功能需求（静态检查/回归/demo） | ST-007, ST-009 | PASS |

**需求覆盖率：** 4 / 4（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 4 | 4 | 0 | 0 |
| 边界条件 | 2 | 2 | 0 | 0 |
| 回归测试 | 3 | 3 | 0 | 0 |
| **合计** | **9** | **9** | **0** | **0** |

覆盖率说明：design.md 未声明覆盖率目标（沿用全量回归 + 静态门）；全量 782 passed 覆盖委托路径与告警全部分支（UT-901~904 六参数化 + 三钉子 + 既有 5 断言）。

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| （无） | — | — |

### 结论

> **Go**：9/9 用例 PASS，需求覆盖 4/4（100%），无 Critical/Major/Minor 缺陷，全量 782 passed 零回归，demo 与静态门全绿。按既定全自动指令执行归档与推送。
