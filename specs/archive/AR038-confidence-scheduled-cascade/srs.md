# [AR038] 需求设计说明书 — 置信度分级验证调度（CSVC）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR038 |
| AR 主题 | 置信度分级验证调度（Skip / Standard / Escalate 三档） |
| 关联 SR | SR-038（自适应算力主线，承接 AR037 ACB） |
| 日期 | 2026-10-25 |
| 状态 | Draft |

## 1. 背景与目标

### 1.1 现状（实测事实）

- AR036 CDF：提交与下一块草稿融合，非跳过块 2 次前向。
- AR037 ACB：草稿置信度达标即跳过 `[N,B]` 验证（τ=0.5 时跳过率 50%，输出逐位等价）。
- 长几何（192 token，真实 checkpoint）：A 172.4 s / D+CDF 16.0 s / D+CDF+ACB 11.8 s（14.6×）。

### 1.2 缺口

ACB 只做**向下**一档：高置信 → 跳过。而**低置信**块仍固定用 N=4 候选，
既不升档（浪费了「这里最需要多样性」的机会），也没有把「算力跟随不确定性」补完整。

### 1.3 目标

实现三档级联 `CSVC`：`skip`（沿用 ACB）/ `standard`（N=4）/ `escalate`（N=`escalate_candidates`，默认 8），
由同一个草稿置信度 `g_min`（仍掩码位贪心概率的最小值）划分。冻结判定：

- **H1 质量**：在低置信 workload 上，`escalate` 档的 mean log p ≥ `standard` 档（预注册方向，容差 0.0 严格不劣）；
- **H2 成本**：`escalate` 档触发率如实发布；高置信块成本不发生回归（skip 档仍为 0 次额外前向）；
- **H3 账本**：`spec_budget_escalations == 实际升档块数`，且 `draft + fused == verify + skips` 恒等；
- **H4 复现**：双跑逐位一致。

若 H1 不成立（升档不提升似然），**如实交付为「无效配置」**（AR035 先例），不包装成收益。

## 2. 需求范围

**In Scope**：`candidate_budget_escalate` / `escalate_candidates` / `escalate_threshold` 三个配置面；
`g_min` 统一计算与三档分流；`spec_budget_escalations` 账本；UT；探针测量；文档与复现指南。

**Out of Scope**：改 skip 档语义；>2 档升档（不做 16/32）；batch>1；LLaDA/Dream；折叠路径；SCR。

## 3. 功能需求

### F1 统一置信度与三档分流

**描述**：每块计算 `g_min = min(softmax(draft_logits)[masked, greedy_token])`，然后：
`g_min >= candidate_budget_threshold` → skip；`g_min < escalate_threshold` → escalate（N=`escalate_candidates`）；
其余 → standard（N=`num_block_candidates`）。

**触发条件**：`adaptive_candidate_budget=True` 与/或 `candidate_budget_escalate=True`，组合臂生效，`num_masked > 0`。

**异常处理**：`escalate_candidates <= num_block_candidates` → 构造期 `ValueError`（无意义配置）；
与 `speculative_commit_gate` 同开 → `ValueError`。

**验收标准**：
- Given `g_min < escalate_threshold`，Then 该块候选行数 == `escalate_candidates` 且 `spec_budget_escalations += 1`；
- Given `escalate_threshold <= g_min < skip_threshold`，Then 行数 == `num_block_candidates`，账本不变；
- Given `candidate_budget_escalate=False`，Then 行为与 AR037 末态**逐字节一致**。

### F2 账本恒等

**描述**：`history_cache_stats` 新增 `spec_budget_escalations`；默认关为 0。

**验收标准**：Given 任意配置，Then `spec_draft_forwards + spec_fused_forwards == spec_verify_forwards + spec_budget_skips`
（跳过块无验证、升档块只多算行不多算次）。

### F3 质量门

**描述**：升档必须不劣化质量。

**验收标准**：Given 低置信 workload，Then `logp(escalate) >= logp(standard) - 0.0`（严格不劣即可）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | mean log p | escalate ≥ standard |
| 成本 | 升档率 | 如实发布（不设目标值） |
| 兼容 | 默认关 | 与 AR037 末态逐字节一致 |
| 可靠性 | 回退 | 阈值非法/未启用 → 完全不改变既有路径 |

## 5. 约束与假设

- 真实 checkpoint、无 mock；判定冻结；单次运行；
- 长几何必须 `stop_token_id=-1`；
- 假设：更多候选行会提升 min-prob 胜者的似然（待 H1 检验，可能被证伪）。

## 6. 术语

| 术语 | 定义 |
|------|------|
| `g_min` | 草稿在仍掩码位上贪心 token 概率的最小值（块级置信度） |
| skip 档 | `g_min` 达标，跳过验证（AR037 ACB） |
| escalate 档 | `g_min` 低于 `escalate_threshold`，候选行数升到 `escalate_candidates` |
