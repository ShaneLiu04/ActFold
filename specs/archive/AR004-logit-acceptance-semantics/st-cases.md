# [AR004] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR004 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-10 |

## 测试用例列表

### ST-001：draft 区域接受率精确比例（含 0.0/1.0 边界）

**关联需求：** srs.md §3.1-1（acceptance_rate == 匹配比例）
**测试类型：** 正常路径 + 边界条件
**优先级：** High

**前置条件：**
- 受控 logits（per-position argmax 可指定）的 stub adapter；child 与 parent 在指定位置差异

**测试步骤：**
1. 构造 argmax 使 draft 区域部分/全部/零匹配
2. 调用 `verify_branch` / 纯函数 `acceptance_rate`

**期望结果：**
- `acceptance_rate` 精确等于匹配比例（0.0、0.5、1.0 实测）

**实际结果：** `tests/test_acceptance.py::test_ut301_*`（4 用例）+ `test_ut303_*`（含 0.0/1.0 边界）+ `tests/test_verification_engine.py::test_ut306`（draft 区域 2 位中 1 位匹配 → 0.5）全部通过；UT-306 实测 rate == 0.5 精确。

**状态：** PASS

---

### ST-002：child == parent（等长逐位相同）→ rate 1.0

**关联需求：** srs.md §3.1-2（空 draft 区域哨兵）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- child tokens 与 parent 逐位相同、等长

**测试步骤：**
1. 调用 `verify_branch`（默认 threshold）

**期望结果：**
- `acceptance_rate == 1.0`、`mean_log_prob == 0.0`（哨兵）、`accepted is True`、无异常

**实际结果：** `tests/test_verification_engine.py::test_ex604_empty_draft_region_is_full_acceptance` 通过（rate=1.0 / mlp=0.0 / accepted / EMA=1.0）。

**状态：** PASS

---

### ST-003：变长 child（append-only）仅后缀计入 draft 区域

**关联需求：** srs.md §3.1-3（AR003 场景泛化）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- parent 短于 child（T_p < T_c），前缀逐位相同

**测试步骤：**
1. `draft_region_mask(parent, child)`
2. 核对 mask 覆盖范围

**期望结果：**
- 仅后缀位（T_p..T_c）为 True；前缀复用位不计入；无 ValueError

**实际结果：** `tests/test_acceptance.py::test_ut302_append_only_child_draft_region_is_suffix_only` 通过（前缀全 False、后缀全 True）；`test_ut302_parent_longer_than_child_*` 同轮通过（T_c<T_p 截公共前缀）。

**状态：** PASS

---

### ST-004：logits/tokens 形状不匹配 → ValueError（纯函数层）

**关联需求：** srs.md §3.1-4（不静默跳过验证）
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- T_c 不一致或 vocab 维缺失（rank-2）或 mask 形状不匹配

**测试步骤：**
1. 以畸形输入调用 4 个纯函数

**期望结果：**
- 全部 `ValueError`

**实际结果：** `tests/test_acceptance.py::test_ut305_*` 8 个 ValueError 用例全部通过（T 不匹配、rank-2、batch 不匹配、mask 不匹配）。

**状态：** PASS

---

### ST-005：形状校验经 engine 路径传播

**关联需求：** srs.md §3.1-4（engine 接线无守卫直调，传播需回归防护）
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- FixedLogitsModel 返回畸形 logits（rank-2）；或 parent/child batch 不匹配

**测试步骤：**
1. `engine.verify_branch`（畸形 logits）→ 期望 ValueError
2. `engine.verify_branch`（parent batch=2 / child batch=1）→ 期望 ValueError

**期望结果：**
- 两路径均 `ValueError`（不静默、不崩溃为其他异常）

**实际结果：** `tests/test_verification_engine.py::test_ex601_engine_propagates_malformed_logits_value_error` + `test_ex602_engine_propagates_batch_mismatch_value_error` 通过（review 第 1 轮 S5 缺陷修复后补齐，第 2 轮审查复核 YES）。

**状态：** PASS

---

### ST-006：actfold_score == 手算 log_softmax gather（draft 区域）

**关联需求：** srs.md §3.2-1（占位替换，数值对拍）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 受控 logits；draft 区域已知

**测试步骤：**
1. `verify_branch` 后取 `child.metadata["actfold_score"]`
2. 与 `torch.log_softmax` + gather 手算参考比对

**期望结果：**
- 逐位一致（fp32）；键名仍为 `actfold_score`

**实际结果：** `tests/test_verification_engine.py::test_ut306`（手算对拍 abs=1e-6）+ `tests/test_acceptance.py::test_ut304_*`（含 1e4 大 logits 有限性）通过。

**状态：** PASS

---

### ST-007：baseline_score == 全位置 mean log-prob

**关联需求：** srs.md §3.2-2（SpiffyBaseline 占位替换）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- stub 模型返回受控 logits；两个候选分支（高/低 log-prob）

**测试步骤：**
1. `SpiffyBaseline.verify(branches)`

**期望结果：**
- `baseline_score` == `mean_log_prob`（全部位置）；高分分支胜出；键名不变

**实际结果：** `tests/test_acceptance.py::test_spiffy_baseline_score_is_mean_log_prob`（精确断言）+ `test_spiffy_baseline_prefers_higher_log_prob_branch`（选择正确）通过。

**状态：** PASS

---

### ST-008：actfold_score 序数消费面零回归

**关联需求：** srs.md §3.2-3（依赖方不回归）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- 既有 `test_integration.py` demo 管线（`max(results, key=actfold_score)` 序数比较）

**测试步骤：**
1. 运行 integration 全量

**期望结果：**
- 序数比较语义下新分数不破坏既有断言

**实际结果：** `tests/test_integration.py` 4 passed（本轮 29 passed 联跑实测）；全量 703 passed 无回归。

**状态：** PASS

---

### ST-009：EMA 首调初始化 == 当次 rate

**关联需求：** srs.md §3.3-1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 新引擎实例；首调 rate=r1

**测试步骤：**
1. 首次 `verify_branch` 后读 `result.ema_acceptance_rate`

**期望结果：**
- EMA == r1（无 0 先验稀释）

**实际结果：** `tests/test_verification_engine.py::test_ut307_ema_initializes_to_first_rate` 通过（r1=0.5 → EMA=0.5）。

**状态：** PASS

---

### ST-010：EMA 链式更新手算一致

**关联需求：** srs.md §3.3-2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 连续 verify（r1, r2, ...），已知 α

**测试步骤：**
1. 多次 verify 后比对 EMA 与 `alpha*r_n + (1-alpha)*ema_{n-1}` 手算值

**期望结果：**
- 链式精确一致（含 α=1 瞬时退化）

**实际结果：** `test_ut307_*` 链式用例通过（α=0.25：0.5→0.625→0.46875 手算一致；α=1 瞬时值）。

**状态：** PASS

---

### ST-011：ema_alpha 越界 → ValueError

**关联需求：** srs.md §3.3-3
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 以 α ∈ {0, -0.5, 1.5} 构造引擎

**期望结果：**
- 全部 `ValueError`；α=1 合法

**实际结果：** `test_ex607` 系（EX-603）`test_ut307_ema_alpha_out_of_domain_raises_value_error`（0/-0.5/1.5 三例）通过；α=1 在链式用例中合法构造。

**状态：** PASS

---

### ST-012：默认 threshold=0.0 判定零变化

**关联需求：** srs.md §3.4-1
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. 默认参数 `verify_branch`（含 rate=0.0 情形）

**期望结果：**
- 全部接受（rate ∈ [0,1] 恒 ≥ 0.0）；与基线行为一致

**实际结果：** `test_ut308_*` 默认阈值用例通过（rate=0.0 仍 accepted）；全量 703 passed 零回归。

**状态：** PASS

---

### ST-013：threshold=0.8 + rate=0.5 → 拒绝 + cache 清理不变

**关联需求：** srs.md §3.4-2
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. threshold=0.8、受控 rate=0.5 的 `verify_branch`
2. 核对 rejected 分支 cache 状态与 stable_ratio 上报

**期望结果：**
- `accepted is False`；cache 中该分支被清理；stable_ratio 照常报告

**实际结果：** `test_ut308_*`（threshold 0.8 + rate 0.5）通过，含 cache contains 翻转断言与 stable_ratio=0.9 照常上报。

**状态：** PASS

---

### ST-014：threshold=0.8 + rate=1.0 → 接受

**关联需求：** srs.md §3.4-3
**测试类型：** 正常路径
**优先级：** Medium

**实际结果：** `test_ut308_*`（threshold 0.8 + rate 1.0）通过（`accepted is True`）。

**状态：** PASS

---

### ST-015：TargetMatchAcceptancePolicy 选接受率最高候选

**关联需求：** srs.md §3.5-1
**测试类型：** 正常路径 + 边界条件
**优先级：** High

**测试步骤：**
1. 构造多候选（接受率可分辨 / 并列 / None logits 混入 / 全 None / 单候选 / B=2 半接受）
2. `policy.select(candidates)`

**期望结果：**
- 选 rate 最高；并列取首个；None 跳过；全 None → candidates[0] 不抛错；单候选直返；0.5 精确分辨

**实际结果：** `tests/test_folded_generation.py::test_target_match_policy_*` 7 用例（UT-309a–g）通过（含 EX-605 全 None 场景与 UT-309f batch 半接受 0.5）。

**状态：** PASS

---

### ST-016：默认参数 folded_generate 零回归 + 跨步均值报告

**关联需求：** srs.md §3.5-2
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. 默认参数 `folded_generate`（因果合成模型）
2. tokens 与 eager `greedy_generate` 比对；读 `result.acceptance_rate`

**期望结果：**
- tokens 逐位相等（零回归）；`acceptance_rate` 为有限 float ∈ [0,1]（跨步均值）

**实际结果：** `test_folded_generate_default_policy_zero_regression_reports_rate`（token bit 相等）+ `test_folded_generate_acceptance_rate_reported_across_steps`（4 步因果 setup，有限 ∈[0,1]）+ 既有 causal 端到端（IT-301）通过。

**状态：** PASS

---

### ST-017：单步 acceptance_rate 0.0/1.0 精确可控

**关联需求：** srs.md §3.5-3
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. FixedArgmaxModel 受控 appended token 是否匹配 argmax
2. 单步/两步（混合均值 0.5）

**期望结果：**
- 单步精确 1.0 / 0.0；两步混合精确 0.5

**实际结果：** `test_folded_generate_acceptance_rate_exact_controlled`（参数化 5 例：单步 1.0、单步 0.0、两步全 1.0、两步全 0.0、混合 0.5）+ `test_folded_generate_zero_new_tokens_acceptance_rate_is_zero`（max_new_tokens=0 → 0.0，EC-12）通过。

**状态：** PASS

---

### ST-018：文档收口齐全且与实现一致

**关联需求：** srs.md §3.6
**测试类型：** 手工检查（MANUAL）
**优先级：** Medium

**测试步骤：**
1. 核对 AGENTS.md 新条目、CHANGELOG AR004 节、README 局限 #8、OPTIMIZATION_GUIDE P2-5 勾选 + 第十一部分回链

**期望结果：**
- 五处齐全；语义描述与实现一致（同位置约定/draft 区域/score 新语义/threshold 切换/EMA）

**实际结果：** 实测——AGENTS.md #40（同位置约定、draft 区域、键名不变语义升级、判定切换、EMA、P3-3 遗留说明）；CHANGELOG.md AR004 节（Added/Changed/Known follow-ups）；README.md 局限 #8（含默认行为零变化说明）；OPTIMIZATION_GUIDE.md :195 P2-5 ✅ + 第十一部分回链表（含语义要点与遗留）。review 第 1 轮 S1 §3.6 行 YES（四文档 + 指南全实证）。

**状态：** PASS

---

### ST-019：NFR 零回归（全量基线 + demo）

**关联需求：** srs.md §4（正确性/兼容性）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\ -q -m "not slow"`
2. `python -X utf8 demo.py`

**期望结果：**
- 652 基线不回归（新增测试使总数上升，既有测试零失败）；demo 85.5% / 2.35e-03 / 93.75% 精确不变

**实际结果：** 全量 **703 passed / 3 skipped / 3 deselected**（652 基线 + 51 新测试，既有零失败）；demo 实测 FLOPs reduction 85.5% / MSE 2.35e-03 [HIGH] / stable ratio 93.75% **精确一致**。

**状态：** PASS

---

### ST-020：NFR API 向后兼容

**关联需求：** srs.md §4（兼容性）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. 核对 `VerificationResult`/`FoldedGenerationResult` 新字段带默认值、metadata 键名不变、`__init__.py` 导出只增不减、engine 新参数带默认值

**期望结果：**
- 既有构造点/消费面全部无需修改即可通过既有测试

**实际结果：** 全量 703 passed（含全部既有构造点测试）；review C1 YES（git diff 实证导出零删减、新字段默认值追加末尾、键名不变）。

**状态：** PASS

---

### ST-021：NFR 数值稳定性（大 logits 无 inf/nan）

**关联需求：** srs.md §4（数值）
**测试类型：** 边界条件
**优先级：** Medium

**实际结果：** `tests/test_acceptance.py::test_ut304_*`（1e4 量级 logits → log_softmax 路径有限值）通过。

**状态：** PASS

---

### ST-022：NFR CPU 可验证 + 向量化实现

**关联需求：** srs.md §4（可移植性/性能）
**测试类型：** 回归测试
**优先级：** Medium

**实际结果：** 全部新增测试在 CPU 运行通过（本机实测）；`acceptance.py` 实现全为向量化 torch op（无 per-position Python 循环、无额外前向），review C2 YES。

**状态：** PASS

---

### ST-023：NFR 质量门（mypy/pyflakes/100 列）

**关联需求：** srs.md §4（质量门）
**测试类型：** 回归测试
**优先级：** High

**实际结果：** 实测——mypy（AGENTS 标准命令 `python -X utf8 -m mypy actfold --ignore-missing-imports`）`Success: no issues found in 65 source files`；pyflakes（actfold + tests）零输出；9 个改动文件 100 列逐行检查合规；Google docstring + 类型注解齐全（review C3 YES）。

**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-10 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 | target-argmax 接受率 | ST-001, ST-002, ST-003, ST-004, ST-005 | PASS |
| §3.2 | mean log-prob 替换占位 | ST-006, ST-007, ST-008 | PASS |
| §3.3 | EMA[r] 追踪 | ST-009, ST-010, ST-011 | PASS |
| §3.4 | 接受判定消费真语义 | ST-012, ST-013, ST-014 | PASS |
| §3.5 | target-match 策略 + folded 报告 | ST-015, ST-016, ST-017 | PASS |
| §3.6 | 文档收口 | ST-018 | PASS |
| §4 NFR | 零回归/兼容/数值/性能/可移植/质量门 | ST-019, ST-020, ST-021, ST-022, ST-023 | PASS |

**需求覆盖率：** 7 / 7（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 10 | 10 | 0 | 0 |
| 边界条件 | 8 | 8 | 0 | 0 |
| 异常处理 | 4 | 4 | 0 | 0 |
| 回归测试 | 5 | 5 | 0 | 0 |
| 手工检查 | 1 | 1 | 0 | 0 |
| **合计** | **23**（覆盖 51 个新增自动化测试 + 全量 703） | **23** | **0** | **0** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | 接受率当前对着 random/perturb draft 测量，是机制指标；真 draft 模型（P3-3）接入后才成为论文口径接受率（已文档化 AGENTS #40 / CHANGELOG Known follow-ups / README #8） | 延后处理（P3-3 独立 AR） |

### 结论

> **Go**——23/23 用例 PASS、需求覆盖 100%、无 Critical/Major 缺陷、全量 703 passed 零回归、demo 基线精确一致、质量门全过；1 个 Minor 遗留（真 draft 模型依赖，已文档化，属 P3-3 范围）。
