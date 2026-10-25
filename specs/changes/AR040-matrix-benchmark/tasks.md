# [AR040] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR040 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-25 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | `scripts/ar040_matrix_bench.py`：prompt 集 × lengths × 四臂 A/D/S/V + logp + 账本 + duty cycle + 冻结 H1-H4 + quick/formal 分离 | - | passing | 只读，不改采样 |
| T002 | quick 实机运行（3 prompts × 160 token）并检出分布结论 | T001 | passing | A 臂为主成本 |
| T003 | 文档收口（README/AGENTS/CHANGELOG/EXPERIMENTS/RERUN） | T002 | passing | |
| T004 | 质量门 + 归档 + commit + gitee 推送树核验 | T003 | passing | |

## 进度记录

- T001/T002：矩阵 runner + quick 实机（3 prompts × 160）；结果 overall refuted（速度 16.2× 鲁棒、质量 1/3 cell 达标）。
- 结论：单 prompt 的质量平价不可推广；默认推荐 D/S，VFG 需矩阵复验。

## 阶段门控记录
