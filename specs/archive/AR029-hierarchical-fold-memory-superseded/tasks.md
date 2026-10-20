# [AR029] 任务清单

**AR029：分页金字塔折叠内存（PPFM）**（srs.md 同目录；本文件为唯一任务真源）

> 后续 AR030（循环折叠状态 + 端侧 RSI agent runtime）不在本文件。

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001a | Red UT（分页面）：`PagedFoldStore` API 等价、页表、前缀共享（`shared_prefix_len`）、COW、cascade plan 去重、`PagePlan` 零同步、页级 LRU/miss、图 plan 策略 | - | pending | UT-2901~2912；冻结面 AR015/016/018/020/025/026/028 零破坏 |
| T001b | Red UT（保留/观测/基准面）：逐 token 稳定计数、默认预算 T/T/2/T/4、pivot/window、`fetch_with_validity`、R0/R1/R2 逐位判据、stats/账本口径、H1/H2 判定式规格 | - | pending | UT-2913~2928；与 T001a 可并行 |
| T002a | Green 页表 + 引用计数 + 前缀共享 + COW（`CachePutEntry` 可选字段，默认旧行为） | T001a | pending | 旧 `put`/`put_batch` 逐字节兼容 |
| T002b | Green `PagePlan`（host）+ run 零同步 + cascade 去重计数 + 图 plan 跨步策略（design 三选一） | T002a | pending | 每步 plan ≤1 次 host 读取 |
| T002c | Green 页级 LRU/淘汰 + miss 语义 + 与 `ActivationCache` 逻辑状态等价 | T002a | pending | #42 账本不混入 page 指标 |
| T003a | Green 逐 token 稳定计数（`StabilityProfiler` 可选张量）+ `PyramidRetentionPolicy`（默认预算/pivot/window） | T001b | pending | 默认关闭；设备侧累加 |
| T003b | Green `fetch_with_validity` + `_fetch_parent_gate_row` 接入（无效 token 强制 `stable=False`）+ R0/R1/R2 判据实现 | T003a | pending | 禁止零填充证据 |
| T003c | Green cascade 在 `_forward_batched_multi` 的接入（共享前缀页聚合读取）+ AR028 整画布/变长语义兼容 | T002b, T003b | pending | 与 AR028 行序/mask 不冲突 |
| T004 | Green 观测面：`stats`（page_hits/misses/evictions/copies/write_count/reuse_bytes/pivot_kept/total）、`page_write_count` 独立账本、零同步探针 | T002c, T003c | pending | 对应 srs FR3 |
| T005 | 合成基准 `scripts/ar029_hierarchical_fold_bench.py`：H1/H2 冻结判定式 + 字节口径 + 几何网格 + `--quick` + 工件 schema | T004 | pending | CPU ≤15 分钟 |
| T006 | 质量门：全量 pytest（启动时实测起点）+ 四 lint + demo 三锚点 + 冻结面零破坏（含 AR028/AR026） | T002c-T005 | pending | canonical 口径 |
| T007 | 文档五件：README 六节全量重写（数字工件重算）+ AGENTS 新条目（分页/保留/有效位契约）+ GUIDE + CHANGELOG + RERUN（真实面配置/命令/资源/判定式） | T005, T006 | pending | 真实面延后注记 |
| T008 | review 门控（只读子代理：spec/design 合规 + 代码审查），全部问题修复至 PASS | T007 | pending | |
| T009 | ST 验收：st-cases.md + 执行（正常/边界/异常/回归） | T008 | pending | 真实面 BLOCKED-DEFERRED |
| T010 | 归档 `specs/archive/AR029-hierarchical-fold-memory/` + commit + gitee 分段推送 + 双端树验证 | T009 | pending | 沿 Gitee Segment-Push Protocol |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

- 2026-10-19：需求 v2（按第 1 轮门控拆分：仅分页 + 分层保留 + 观测 +
  H1/H2；循环状态/agent runtime 移至 AR030）；待第 2 轮门控。

## 阶段门控记录

### 2026-10-19 req 门控记录

- 门控结果：FAIL（第 1 轮）
- 失败项：G3/G6/G8/G9 —— 2 项 Critical：P1 FR2"与全保留逐位一致"判据
  不可达（父/子 FFN 不同）；P2 未保留 token 的 miss 语义无法由现有
  零填充 API 表达（静默近似风险）；Important：P3 前缀/cascade 无入口、
  P4 单 AR 范围过大、P5 缺 AR028 前置/冻结面、P6 H1/H2 口径不可复算、
  P7 阈值未 pin、P8 任务过粗、P9 FR5 无任务、P10 `validate_entries`
  误列为 cache API 且账本口径矛盾；Minor：P11-P13
- 严重性：Critical
- 修复：v2 重构——范围切分为分页+分层保留+观测+H1/H2（循环状态/agent
  runtime 移至 AR030）；FR2 改 R0/R1/R2 三层判据 + `valid_mask` 显式
  有效位；新增逐 token 稳定计数；`CachePutEntry` 可选前缀元数据；
  cascade 接入 `_forward_batched_multi`；H1/H2 字节/页写口径 pin；
  默认预算 T/T/2/T/4、pivot/window=64 pin；AR028/AR026 前置与冻结面；
  删除 `validate_entries` API 表述、page 指标独立账本；任务拆为
  T001a/T001b、T002a-c、T003a-c
- 审查代理：sdd-gate-reviewer
