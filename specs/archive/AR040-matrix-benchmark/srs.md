# [AR040] 需求设计说明书 — 多 prompt × 多长度矩阵基准（统计鲁棒性）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR040 |
| AR 主题 | 矩阵基准：把单点结论升级为分布结论，并提供阈值标定 |
| 关联 SR | SR-040（可靠性与可复现性） |
| 日期 | 2026-10-25 |
| 状态 | Draft |

## 1. 背景与目标

AR033→AR039 的全部证据都是**单 prompt、单几何、确定性单 rep**。结论（长几何 ~19.5×、质量不降）
可能对 prompt/长度敏感而未被发现。SOTA 工程实践要求多 workload 矩阵与分布结论。

**目标**：提供一个可复用矩阵 runner，对 `{prompts} × {lengths}` 的每个 cell 运行四臂
（A 基线 / D CDF / S ACB+CDF / V VFG+CDF），产出单工件，并冻结聚合判定：

- **H1 质量分布**：V 的 mean log p ≥ A 的 mean log p − 0.05 的 cell 比例 ≥ 90%；
- **H2 加速分布**：`median(A/V) ≥ 10×`，且 V < S < D 的 cell 比例 ≥ 80%；
- **H3 账本一致性**：每个 V cell 满足 `free_verify_blocks == spec_fused_forwards`、
  `spec_verify_forwards ≤ 块数`、`degraded == 0`；
- **H4 复现**：对每个 V cell 复跑一次，token 逐位一致（不一致则全体 blocked）。

任一 H 失败 → 如实交付（不重跑凑数）。

## 2. 需求范围

**In Scope**：矩阵 runner（prompt 集/Lengths/arms/reps/quick）；四臂；logp 质量代理；
逐 cell 与聚合指标；冻结判定；`--quick` 小矩阵；工件 schema；文档与复现指南。

**Out of Scope**：新机制（本 AR 不改采样语义）；batch>1；自动调参；新模型族。

## 3. 功能需求

### F1 矩阵执行

**描述**：对每个 `(prompt, length)` cell，按 round-robin 顺序跑 A/D/S/V，记录 ms、生成 token 数、logp、账本。

**验收标准**：
- Given 3 prompts × 1 length × quick，When 运行，Then 产出 12 个 cell 记录且工件写入 `*_quick.json`；
- Given 同一输入重复运行，Then V 臂在每个 cell 逐位一致（H4）。

### F2 聚合判定

**描述**：按 H1-H4 计算聚合结论并写入工件 `verdicts` 字段。

**验收标准**：Given 任一 H 失败，Then `overall != supported`，且不修改原始数字。

### F3 阈值标定信息

**描述**：输出每个 cell 的 V 臂「免验证块比例」与再武装次数，供阈值标定。

**验收标准**：Given 任意运行，Then 工件含 `duty_cycle` 与 `rearms` 字段。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 复现 | 单命令 | 一条 `python -m scripts.ar040_matrix_bench` 复跑整矩阵 |
| 诚实 | quick/formal 分离 | quick 写 `_quick` 工件，永不覆盖 formal 路径 |
| 成本 | quick 预算 | 3 prompts × 160 token × 1 rep ≤ 15 min（Quadro RTX 5000） |
| 兼容 | 只读 | 不修改任何采样机制（零回归） |

## 5. 约束与假设

- 真实 checkpoint、真实文本 prompt（内置 6 条，无 mock）；`stop_token_id=-1`；
- 质量代理 = 同模型 mean log p（AR037 已确立：不得用与基线的一致率）；
- 假设：矩阵可在 16GB WDDM 工作站上以 quick 配置跑完。

## 6. 术语

| 术语 | 定义 |
|------|------|
| cell | 一次 `(prompt, length)` 组合下的四臂运行 |
| 免验证块比例 | V 臂 `spec_verification_free_blocks / 总块数` |
| 再武装 | 免费验证低置信后下一块强制完整验证 |
