# [AR043] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR043 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-25 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 预设模块 `actfold/models/sampler_presets.py`（baseline/quality/latency + 剔除清单） | - | passing | |
| T002 | UT-4301~4306 | T001 | passing | |
| T003 | 五臂组合收敛基准（A/KV/spec/quality/latency）+ 冻结 V1-V4 | T001 | passing | 干净运行决策；复跑标注争用 |
| T004 | 回退 `remask_fallback`（1.4542× vs 1.5129×，logp 相同）并重跑 | T003 | passing | |
| T005 | 文档收口 + 归档 + commit + gitee 推送树核验 | T004 | passing | |

## 进度记录

- T003 干净运行：kv-only 1.5129×（logp 平价）、spec 12.82×、quality(AR041) 1.4542×、latency 14.28×。
- T004 按回退规则把 `remask_fallback` 从 quality 预设剔除；quality 现等价于 kv-only。
- 复跑受宿主争用（431 s / 807 s 离群），仅记录 logp 一致性，不作计时结论。

## 阶段门控记录
