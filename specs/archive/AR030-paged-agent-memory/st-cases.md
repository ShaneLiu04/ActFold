# AR030 ST 验收用例与执行记录

**AR**：AR030 面向端侧 Agent/RSI 的页共享折叠内存 PSFM（机制/基础设施交付）
**日期**：2026-10-20
**范围裁定**：真实 checkpoint A/B = BLOCKED-DEFERRED（`baseline_claim=false`）
**执行环境**：Windows 11 / Python 3.11.9 / CPU（quick 机制面）；无 GPU 实验
**review 门控**：2 轮（第 1 轮 FAIL 15 项已修复；第 2 轮 PASS 12/12 维度，代理
`ses_ea683ed2dffe45RZxW2Vz6L4Ze`，4 项非阻塞遗留已即时闭合 UT-3099~3101 + 记录）

## 需求覆盖

| FR | 需求 | 覆盖用例 | 结果 |
|----|------|----------|------|
| FR1 | 页表与 COW 分支分叉（O(1) fork/逐页去重/step=0/不变量） | ST-01, ST-02, UT-3001–3012, 3079–3084 | PASS |
| FR2 | 共享页逐层物理页预算与冻结淘汰键（hard bound/显式 evict/rollout 硬保护） | ST-03, ST-04, UT-3013–3028 | PASS |
| FR3 | Host 计划 / `enter_plan` 只读窗口 + 暂存原子提交（epoch 表/零回读/stale） | ST-05, ST-06, UT-3029–3038, 3086, 3089–3093 | PASS |
| FR4 | 冷热分层与 snapshot/restore（demote/promote 往返/冻结 schema/恢复计时） | ST-07–ST-09, UT-3039–3048, 3085, 3088, 3100 | PASS |
| FR5 | Agent rollout 事务（base 冻结/三类失败语义/close/stats/sync） | ST-10, ST-11, UT-3049–3062, 3094–3096, 3099 | PASS |
| FR6 | 机制基准与复现协议（E1–E4 冻结判定/quick-only/COW-off 对照 refuted） | ST-12, ST-13, UT-3069–3078, 3101 | PASS |
| FR7 | 工厂/配置/接线（优先级/深度解析/graph 三处拒绝/YAML 10 键/零回归） | ST-14, UT-3063–3068, 3097, 3098 | PASS |

## 用例

### ST-01（正常）O(1) fork 与逐页 COW 去重
- 命令：`python -m pytest tests/test_ar030_page_table.py -q`
- 期望：fork 不复制页数据、共享页 refcount+1；写差异页仅分配新页、独占页原地写；
  零引用立即 reclaim；不变量全 PASS。
- 实测：15 passed。

### ST-02（边界）非零 step 与非法写原子失败
- 命令：`python -m pytest tests/test_ar030_plan.py::test_ut3092_nonzero_step_rejected_everywhere -q`
- 期望：全部 8 个入口（put/put_batch/rollout_put/fetch/get_all/get/fetch_masked/
  availability）非零 step → `ValueError`；非法 fork/put 后页表/refcount/bytes 全等。
- 实测：PASS（UT-3092；原子性由页表套件 UT-3001–3012 覆盖）。

### ST-03（边界）逐层物理页预算恰好触界
- 命令：`python -m pytest tests/test_ar030_cache_budget.py -q`
- 期望：每层 hot 物理页 ≤ `budget_pages`；SRS 冻结几何（512 token = 16 页/层）
  下 8 根页 + 8 候选新页恰好触界、零驱逐；共享页只计一份。
- 实测：22 passed（含 UT-3079 触界零驱逐）。

### ST-04（异常）淘汰确定性与显式 evict 保护
- 命令：`python -m pytest tests/test_ar030_cache_budget.py tests/test_ar030_rollout.py -q`
  （淘汰序列/`evict` 返回 False 面/`RolloutBudgetError`）
- 期望：冻结淘汰键 `(protected_any, refcount, write_seq, page_id)` 序列确定；
  保护集/非 hot/rollout 引用页显式 evict → False；hard bound 永不突破。
- 实测：40 passed（两文件合计；含 UT-3096 rollout 保护页）。

### ST-05（正常）计划 availability 全等与窗口内零回读
- 命令：`python -m pytest tests/test_ar030_plan.py tests/test_ar030_tiering.py::test_ut3086_zero_readback_inside_plan_window -q`
- 期望：`enter_plan` 只读期 fetch/availability 与实时表全等；窗口内
  `.item()/.tolist()/.cpu()/__int__/__float__` 计数 == 0；输出与无计划路径逐位一致。
- 实测：plan 15 passed + UT-3086 PASS。

### ST-06（异常）stale plan / 窗口 mutator / 提交失败回滚
- 命令：`python -m pytest tests/test_ar030_plan.py::test_ut3089_build_plan_missing_face_raises tests/test_ar030_plan.py -q -k "3091 or 3093"`
- 期望：epoch 不匹配默认 `PlanStaleError`；窗口内任何 mutator/嵌套/restore/
  record_rollout_masks → `RuntimeError`；暂存提交失败整窗回滚、epoch 不变。
- 实测：PASS（UT-3089/3091/3093；提交失败 k=0/k=1 原子另由基准 E4 覆盖）。

### ST-07（正常）demote/promote 往返与 snapshot/restore 全等
- 命令：`python -m pytest tests/test_ar030_tiering.py -q`
- 期望：cold 页 `availability=False`、fetch NaN；promote 后与 demote 前 `torch.equal`；
  snapshot/restore 跨实例 tier/availability/内容逐位全等（含末页 page_len）。
- 实测：14 passed（含 UT-3088 冻结 entries schema）。

### ST-08（边界）恢复计时 < 50 ms
- 命令：`python -m pytest tests/test_ar030_tiering.py::test_ut3047_restore_then_promote_timing -q`
- 期望：quick 几何、warmup 后 5 次中位数 `restore + promote 全部 cold 页` < 50 ms
  （计时含 restore，SRS 验收单元；每 rep 新建空实例）。
- 实测：PASS（UT-3047 重写版）。

### ST-09（异常）损坏/legacy 快照与活跃 rollout 拒绝 restore
- 命令：`python -m pytest tests/test_ar030_tiering.py::test_ut3088_snapshot_entries_schema_is_frozen tests/test_ar030_tiering.py::test_ut3100_restore_rejects_cache_with_active_rollout -q`
- 期望：sha256/schema 损坏与旧 `__seq_len__` 载荷 → `ValueError` 且目标实例不变；
  活跃 rollout/非空缓存 restore → `ValueError`（同快照可恢复进空实例）。
- 实测：PASS。

### ST-10（正常）rollout 事务与候选隔离
- 命令：`python -m pytest tests/test_ar030_rollout.py -q`
- 期望：base 冻结（任何写 → `ValueError`）；候选写互不可见；close 解冻；
  `rollout_stats` 比率 ∈ [0,1]、可直接复算。
- 实测：18 passed。

### ST-11（异常）三类失败语义与 sync 守卫
- 命令：`python -m pytest tests/test_ar030_rollout.py -q -k "3094 or 3095 or 3096 or 3099 or 3061"`
- 期望：窗口外单候选写失败仅回滚该次（epoch 不变、failures+1）；窗口内校验失败
  仅丢弃该候选暂存；外部改写 `RolloutSet` 句柄不影响判定；显式 `rollback_candidate`
  恢复 fork 基线；sync 树 ghost/环/重复 id → `ValueError`、干净树通过。
- 实测：5 passed（UT-3094/3095/3096/3099/3061）。

### ST-12（正常）机制基准 quick 面
- 命令：`python -m scripts.ar030_paged_agent_bench --quick`
- 期望：`overall=supported`，E1–E4 全 true，写 `_quick.json`，exit 0。
- 实测：`supported`，exit 0。工件数字：E1 两面 `max_abs_diff=0.0`；E2 物理页
  32（16/层恰好触界）vs nonshared 144（dense oracle 实际条目复算）、
  `paged_hot_bytes 98304 == plan_expected_bytes 98304`（边界相等）、sharing 0.778、
  evictions 0；E3 fork 字节增量 0、每候选新增页==差异页==2、集合级相等；
  E4 诚实检查 14/14 全真；duration 0.184 s < 60 s 预算。

### ST-13（异常/mutation）COW-off 对照 refuted 与正式面拒绝
- 命令：`python -m pytest tests/test_ar030_bench_verdicts.py::test_ut3101_cow_off_contrast_refutes_e2_end_to_end -q`；
  `python -m scripts.ar030_paged_agent_bench`（无 `--quick`）
- 期望：COW-off 集成对照 → physical == nonshared == 144、sharing == 0 < 0.5、
  E2 False、overall **refuted**（SRS s369 验收链端到端）；无 `--quick` → exit 3、
  正式工件不存在。
- 实测：UT-3101 PASS；formal exit 3（BLOCKED-DEFERRED）、
  `ar030_paged_agent_bench.json` 不存在。

### ST-14（正常/异常）工厂优先级、深度解析与 graph 三处拒绝
- 命令：`python -m pytest tests/test_ar030_integration.py -q`
- 期望：`use_paged` 优先级最高；`paged_depth=0` 不可解析 → runner `ValueError`
  （绝不猜）、`num_layers=4` → 4；paged×graph 在 runner kwargs/MFF init/cache
  setter 三处 `ValueError`（setter 失败旧缓存不变）；YAML 10 键无 unknown-key
  警告；默认关闭零回归。
- 实测：8 passed（含 UT-3097/3098）。

### ST-15（回归）全量质量门
- 命令：`python -m pytest tests/ -q`；`black --check`；`isort --check-only`；
  `pyflakes`；`mypy actfold --ignore-missing-imports`；
  `mypy scripts --ignore-missing-imports --explicit-package-bases`
- 期望：全量通过；格式/静态检查零告警；mypy 73+28 零问题；AR030 套件 < 30 s、
  quick < 60 s。
- 实测：**1376 passed / 4 skipped / 3 deselected**（79.15 s；AR030 套件 103
  passed 7.13 s）；black 186 files unchanged；isort/pyflakes 零告警；mypy 73+28
  零问题。

## 结论

- FR1–FR7 全部覆盖并通过；AR030 套件 103 passed，全量 1376 passed。
- 无伪造数字；机制工件 `baseline_claim=false`、`evidence_level="mechanism"`；
  真实 checkpoint 面按范围裁定记录为 BLOCKED-DEFERRED（解除条件：run_formal_queue
  FaceSpec + ≥24GiB 安静 GPU/Linux-TCC）。
- review 门控 2 轮：FAIL（15 项：未记录偏离 + 验收缺口）→ 修复（6 代码修复 +
  bench 重写 + 19 新 UT + 4 裁决）→ **PASS**（12/12 维度；4 项非阻塞遗留已
  闭合：UT-3099/3100/3101 + UT-3087 编号空缺记录）。
- **Go**：需求覆盖 7/7（100%）、无 Critical/Major 缺陷、无回归、NFR 全达标
  （恢复 < 50 ms、套件 7.13 s < 30 s、quick 0.184 s < 60 s、默认关闭零回归）。
