# [AR044] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR044 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-25 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 实现 commit_refine_passes / commit_refine_threshold：填后精修 + spec_refine_forwards 账本 | - | passing | 默认 0 零回归 |
| T002 | UT-4401~4404（默认关 / passes=1 账本 / mask-free 与确定性 / 非法值） | T001 | passing | |
| T003 | 3 prompts × 160 token × passes∈{1,2,3} 实测，对照 kv-only 与 latency | T001 | blocked | 外部 GPU 负载 99%，两次运行 30–60 min 无 cell；服务器待跑 |
| T004 | 文档收口 + 归档 + commit + gitee 推送树核验 | T003 | passing | 平价未验证 → 不新增 balanced 预设 |

## 进度记录

- T001/T002：填后精修 + 账本 + UT-4401~4404（1534 passed）。
- T003：**blocked** —— 两次运行被外部进程占满 GPU（99%），无任何 cell；已清理本仓库残留进程。
- T004：按诚实协议交付（无数据不固化预设）；服务器复现命令写入 README §28 / EXPERIMENTS 4.5n / RERUN 32。

## 阶段门控记录
