# [AR038] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR038 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-25 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 统一 `g_min` 计算 + 三档分流（skip/standard/escalate）+ `spec_budget_escalations` 账本 | - | passing | 默认关零回归 |
| T002 | UT-3801~3805（默认关零回归 / 升档触发 / 账本恒等 / 阈值非法 / 与门控互斥） | T001 | passing | |
| T003 | 探针实测（192 token 长几何）：三档分布、logp(escalate) vs logp(standard)、墙钟 | T001 | passing | H1 可能被证伪，如实交付 |
| T004 | 文档收口（README/AGENTS/CHANGELOG/ALGORITHM/EXPERIMENTS/RERUN） | T003 | passing | |
| T005 | 质量门 + 归档 + commit + gitee 推送树核验 | T004 | passing | |

## 进度记录

- T001/T002：三档级联 + 账本 + UT-3801~3805。
- T003：192 token 实测升档零质量收益（logp 全 −2.0870）、成本 +46% → 判定无效档位（默认关）。

## 阶段门控记录
