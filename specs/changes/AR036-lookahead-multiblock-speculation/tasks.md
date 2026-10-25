# [AR036] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR036 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-25 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 探针：P1 融合前向第 b 块 logits == 单独提交；P2 融合第 b+1 块决策行 vs 独立草稿 maxdiff/argmax；P3 KV 裁剪后逐层 byte-equal；P4 缓存裁剪 API 可用 | - | passing | 任一硬失败即判不可行并如实交付 |
| T002 | 实现 `commit_draft_fusion`（默认 False）+ 融合前向 + KV 裁剪 + 草稿括号暂存/消费 + 回退 + `spec_fused_forwards` 账本 | T001 | passing | 默认关零回归 |
| T003 | UT-3601~3606：默认关零回归、开关字节一致、融合账本与每块 2 前向、裁剪不可用回退、块号不匹配防御、非法组合 | T002 | passing | 复用 AR033 fixture |
| T004 | 三臂 bench `scripts/ar036_commit_draft_fusion_bench.py`（A/D/F）+ H1-H4 冻结 + repro | T002 | passing | A=基线 D=AR034 F=D+融合 |
| T005 | 快速面实机运行 → `ar036_commit_draft_fusion_bench_quick.json` | T004 | pending | quick 不冒充 formal |
| T006 | 文档收口：README 行21/§20/阶梯17/测试基线、AGENTS #55、CHANGELOG、ALGORITHM §21、EXPERIMENTS §4.5f、RERUN 第24步 | T005 | passing | UTF-8 Python 补丁 |
| T007 | 质量门全绿 + 归档 + commit + gitee 段推送树核验 | T006 | passing | |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

- 2026-10-25：需求阶段发现「深度 2 前瞻」被 advance-token 依赖 + 块内双向注意力锁死（结构不可行），
  同分析导出可行方案 CDF（提交-草稿融合），AR036 据此定稿。

## 阶段门控记录
