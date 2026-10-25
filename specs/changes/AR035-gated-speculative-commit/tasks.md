# [AR035] 任务清单

| 编号 | 任务 | 类型 | 依赖 | 验收 | 状态 |
|------|------|------|------|------|------|
| T001 | design.md 冻结：门控接受集合/强制推进/迭代语义、账本、判定式、三臂配置 | 设计 | — | design.md 落锤，判定先于数据收集 | pending |
| T002 | 实现：`speculative_commit_gate` 配置 + `_history_speculative_small_block` 门控返回 + 调用方继续迭代 + 账本 `spec_gated_iterations` | 开发 | T001 | FR1 全绿；默认关逐字节不变 | pending |
| T003 | UT（fp32 fixture）：门控语义/强制推进/默认关零回归/账本/确定性 | 测试 | T002 | 新增 UT 全绿；全量基线不降 | pending |
| T004 | `scripts/ar035_gated_spec_bench.py` 三臂（A/D/G）多块几何 + 冻结 H1-H4 | 开发 | T002 | lint/mypy 过；quick 结构自检 | pending |
| T005 | quick 实机 + 修复循环 | 执行 | T003,T004 | quick 工件；质量门绿 | pending |
| T006 | 文档：README 行/简节/阶梯、AGENTS、CHANGELOG、RERUN（含服务器 formal 协议）、design 收口 | 文档 | T005 | 数字由工件重算 | pending |
| T007 | 全量质量门 + 归档 + gitee 段推送 + 终树核验 | 质量/归档 | T006 | 零告警；树核验 PASS | pending |
