# [AR039] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR039 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-25 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 实现 `verification_free_commit` + `free_verify_floor`：VFG 跳过验证、FV 复用融合 logits、再武装、三计数账本 | - | passing | 默认关零回归；须 CDF |
| T002 | UT-3901~3905（默认关 / 跳过与账本 / 再武装 / 非法组合 / 逐位等价于关闭态开关矩阵） | T001 | passing | |
| T003 | 长几何三臂实测（A / ACB+CDF / VFG）：logp、墙钟、账本、repro | T001 | passing | 默认关 ACB；VFG 侧再武装率如实发布 |
| T004 | 文档收口（README/AGENTS/CHANGELOG/ALGORITHM/EXPERIMENTS/RERUN） | T003 | passing | |
| T005 | 质量门 + 归档 + commit + gitee 推送树核验 | T004 | passing | |

## 进度记录

- T001/T002：VFG + 免费验证 + 再武装 + 三计数账本；UT-3901~3905；修复跳过路径不交接 advance 的真实缺陷。
- T003：192 token 实测 D 14.50 s / ACB 10.16 s / VFG 8.84 s，logp 全 −2.0870；对基线 19.5×。

## 阶段门控记录
