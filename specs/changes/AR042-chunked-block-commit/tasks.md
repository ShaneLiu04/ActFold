# [AR042] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR042 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-25 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 实现 `commit_chunk_size`：块内渐进解掩（每轮填最高置信 chunk 位 + 重新前向）+ stash 复用 + `spec_chunk_forwards` 账本 | - | passing | 默认 0 = 逐字节零回归 |
| T002 | UT-4201~4205（默认关 / chunk=1 前向数 == 掩码数 / chunk≥掩码数 等价一次性 / mask-free 与确定性 / 非法值） | T001 | passing | |
| T003 | 3 prompts × 160 token × chunk∈{4,8,16,32} 实测：logp/墙钟/前向数，对照 A/D（AR041 已测） | T001 | passing | 检验 H1/H2 |
| T004 | 文档收口 + 归档 + commit + gitee 推送树核验 | T003 | passing | |

## 进度记录

- T001/T002：分块提交 + 账本；UT-4201~4205。
- T003：3 prompts × 4 chunk 实测，H1 单调性 refuted（质量与 chunk 无单调关系，最优值逐 prompt 漂移）；C32 ≈ VFG 自洽。

## 阶段门控记录
