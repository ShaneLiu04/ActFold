# [AR018] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR018 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-14 |

> 执行环境：Quadro RTX 5000 16GB，torch 2.5.1+cu121，WDDM，驱动 556.18，种子 2026。
> 测试命令统一 `python -X utf8`；全量口径 `-m "not slow"`（本机未装 bench 依赖，slow 用例为既有环境事实，见 review 门控 M-4 注记）。

## 测试用例列表

### ST-001：发布路径设备归约 launch 数 L×N → 恰 1

**关联需求：** srs.md §3.1 FR1（验收第 1 条）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- CUDA 可用；AR018 实现已合入；`GLOBAL_STABILITY_PROFILER.enabled=True`、`debug_enabled=False`

**测试步骤：**
1. 执行 `python -X utf8 -m pytest tests\test_publish_cost.py::test_ut1702_single_mask_reduction_and_zero_host_sync -q`

**期望结果：**
- 发布窗口 bool `Tensor.sum` 恰 1 次（单次 `mask_buf.sum(dim=2)`）；`enabled=False` 时恰 0 次且零记录

**实际结果：** 1 passed（spy 计数断言全过；同窗口 item/tolist/cpu 恰 0——见 ST-004）

**状态：** PASS

---

### ST-002：debug_enabled 回退与 divergence_positions 保留

**关联需求：** srs.md §3.1 FR1（验收第 2 条）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- CUDA 可用；混合 mask 负载（发布各层既有 stable 又有 divergent 位置）

**测试步骤：**
1. 执行 `tests\test_publish_cost.py::test_ut1707_debug_fallback_and_normal_batch`

**期望结果：**
- `debug_enabled=True`：逐行 `record` 回退（bool-sum=L·N=8）、混合层产出 divergence_positions（与现状逐位一致）；关闭后：bool-sum 恰 1、全部 positions 为 None

**实际结果：** 1 passed（两分支断言全过）

**状态：** PASS

---

### ST-003：record_precomputed 与 record 产出逐位等价 + 显式 ndim 校验

**关联需求：** srs.md §3.1 FR1（验收第 3、4 条）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 无需 CUDA 的 CPU 子块（240 条记录含 history 裁剪触发）+ CUDA 子块（[L,N,T] 批量归约 0-dim 视图）

**测试步骤：**
1. 执行 `test_ut1701_record_precomputed_equivalence_cpu` 与 `test_ut1701_batched_sum_views_equivalence_cuda`

**期望结果：**
- `_raw` 逐条全字段相等（stable_sum 经 `.item()`）；`_history` 键集/长度/裁剪一致；`get_profile().to_dict()`、`mean_stable_ratio()`/`all_profiles()`/`get_mean_stable_ratio()` 聚合面一致；非 0-dim/非 Tensor 输入 ValueError；`enabled=False` 双入口零记录

**实际结果：** 2 passed（240 条对拍 + [L,N,T] 视图对拍 + 2×ValueError + 裁剪断言全过）

**状态：** PASS

---

### ST-004：发布路径零新增 host 同步

**关联需求：** srs.md §3.1 FR1（验收第 5 条）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- CUDA 可用；spy 窗口=单次发布调用

**测试步骤：**
1. 执行 UT-1702（item/tolist/cpu 计数断言已含于同一用例）

**期望结果：**
- 窗口内 `.item()`/`.tolist()`/`.cpu()` 恰 0 次调用（#39 纪律；0-dim 视图构造是宿主元数据操作）

**实际结果：** 1 passed（三个 host-sync spy 计数断言全过，与 ST-001 同用例执行）

**状态：** PASS

---

### ST-005：快照发布内容 bit-exact

**关联需求：** srs.md §3.2 FR2（验收第 1 条）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 同一重放静态缓冲状态；测试本地复刻旧逐行 clone 参考实现（不调 record）

**测试步骤：**
1. 执行 `test_ut1703_snapshot_publish_bit_exact`

**期望结果：**
- 快照发布（cache A）与逐行参考发布（cache B）逐 (branch,layer) 键 `torch.equal`（ffn_out + layer 0 embedding）

**实际结果：** 1 passed（8 组键 × 2 张量逐位相等）

**状态：** PASS

---

### ST-006：防别名契约（默认 cache + by-reference cache 双面）

**关联需求：** srs.md §3.2 FR2（验收第 2 条）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- 真实捕获后的 runner；by-reference RefCache（put 存引用）替换 `mff.cache` 后直接调 `_publish_graph_children`

**测试步骤：**
1. 执行 `test_ut1705_by_reference_cache_anti_alias`（RefCache 面）
2. 执行 `test_ut1706_publish_completeness_and_isolation` 的隔离子块（ChunkedActivationCache 面）

**期望结果：**
- 发布后 `child_buf.add_(123)`：RefCache 全部条目逐位不变；chunked cache 条目逐位不变；二次重放发布新值正确写入且不覆写首发

**实际结果：** 2 passed（两用例含隔离断言全过）

**状态：** PASS

---

### ST-007：发布路径 clone 调用数 L×N → 恰 1

**关联需求：** srs.md §3.2 FR2（验收第 3 条）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- spy 窗口=单次发布调用（发布后的 logits.clone 在窗口外）

**测试步骤：**
1. 执行 `test_ut1704_single_child_buf_clone`

**期望结果：**
- child_buf identity 的 clone 恰 1；窗口内总 clone 恰 1（无逐行中间 clone）

**实际结果：** 1 passed

**状态：** PASS

---

### ST-008：cache.put 协议零变化（次数/形状/layer 0 双键）

**关联需求：** srs.md §3.2 FR2（验收第 4 条）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 默认 ActivationCache；两次批量调用（捕获步 + 重放步）

**测试步骤：**
1. 执行 `test_ut1706_publish_completeness_and_isolation` 完备性子块

**期望结果：**
- 每分支全 L 层 `ffn_out` [1,T,H] + layer 0 `embedding` [1,T,H]（与 embed_module 行 allclose 1e-6）——UT-1502 面继续成立

**实际结果：** 1 passed（捕获步+重放步两轮完备性断言全过）

**状态：** PASS

---

### ST-009：UT-1701~1708 一次全绿

**关联需求：** srs.md §3.3 FR3（UT-1701~1708 逐条）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. 执行 `python -X utf8 -m pytest tests\test_publish_cost.py -q`

**期望结果：**
- 9/9 passed（1701 CPU/CUDA 子块 + 1702~1708）

**实际结果：** 9 passed in 7.21s（<30s，NFR「测试确定性」达标）

**状态：** PASS

---

### ST-010：既有契约套件零改动全绿（UT-1502/1509/1602/1606）

**关联需求：** srs.md §3.3 FR3（验收第 6 条）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. 执行 `python -X utf8 -m pytest tests\test_batched_graph_forward.py tests\test_adaptive_capacity.py -q`

**期望结果：**
- 既有测试文件零改动全绿（发布路径行为兼容的直接证据）

**实际结果：** 与 ST-009 合并执行 26 passed（test_batched_graph_forward 9 + test_adaptive_capacity 8 + test_publish_cost 9）；`git status` 证实两既有文件零改动

**状态：** PASS

---

### ST-011：mutation spot-check ①-⑤ 全红后 SHA256 还原

**关联需求：** srs.md §3.3 FR3（UT-1709）
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 逐变异修改生产代码 → 运行映射 UT → 确认红 → 备份还原 → SHA256 比对（T002 执行）

**期望结果：**
- ①删批量归约→UT-1702 红；②删整块快照回退逐行 clone→UT-1704 红；③快照别名 child_buf→UT-1705 红；④删 debug 回退→UT-1707 红；⑤stable_sum 跨调用 scratch 复用+覆写→UT-1701 红；五次还原后 SHA256 与基线逐字节一致

**实际结果：** 5/5 红确认（红因均=生产缺失）；还原后 architecture_utils 0876A68B…04CA1 / stability_profiler 0FCCA1F0…35B87 与基线一致；还原后全量 855 passed 零回归

**状态：** PASS

---

### ST-012：基准工件字段完备性（FR4 结构验收）

**关联需求：** srs.md §3.4 FR4（三路径/几何/相位分解/交错与功效/环境与限制）
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 脚本化检查 `results/optimization/ar018_publish_bench.json`：三路径×4 配置、几何矩阵 (256,8)/(1024,16)×N∈{4,8}×T=512、全配置 n=30、六相分解、phase_sum+full_step 同入+双向偏差注记、锚点可用性+AR017 run_date、CUPTI null+注记、峰值显存字段、seed=2026+reps=30+command、H1-H4 声明与判定分离

**期望结果：**
- 13 项结构检查全部 PASS

**实际结果：** 13/13 PASS（`st_artifact_checks.py`，2026-10-14 执行）

**状态：** PASS

---

### ST-013：H1-H4 判定与数据自洽（复算）

**关联需求：** srs.md §3.4 FR4（假设与结果分离）
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 从工件 timings 逐配置重算 AR018 加速比与锚点加速比比较（H2）；重算发布占比与探针对照比较（H3）；核对 H1 逐配置判定与 2σ 判据

**期望结果：**
- H2 4/4 supported（2.4034/2.6743/2.6607/2.9466 ≥ 1.5502/1.6839/2.2993/2.4609）；H3 4/4 supported（占比 < 探针参照）；H1 整体 inconclusive（(256,8,4) delta 2.41ms < 2σ 3.37ms，其余 supported）；H4 supported（argmax 精确 + allclose 1e-4 + profiler 一致）

**实际结果：** H2/H3 重算 4/4 PASS；工件 verdicts {H1: inconclusive, H2/H3/H4: supported} 与重算一致（review 门控亦已全配置复算）

**状态：** PASS

---

### ST-014：基准运行预算与 --quick 冒烟

**关联需求：** srs.md §3.4 FR4（运行预算）
**测试类型：** 边界条件
**优先级：** Medium

**测试步骤：**
1. `python -X utf8 -m scripts.ar018_publish_bench --quick --out <temp>` 冒烟
2. 全量 `--reps 30` 运行（T003 执行）

**期望结果：**
- --quick 单配置 (256,8,4) 通过；全量 < 10 分钟

**实际结果：** --quick PASS（单配置 H3/H4 supported、H2 supported）；全量运行约 2 分钟（4 配置 × 35 交错迭代 + 相位分解 30×4）——远低于预算

**状态：** PASS

---

### ST-015：文档收口（FR5）

**关联需求：** srs.md §3.5 FR5
**测试类型：** 正常路径
**优先级：** Medium

**测试步骤：**
1. 脚本化检查：指南 P1-7 行第四档追注、第二十五部分（含三表+语义要点）、Part 24 遗留收口注记存在；CHANGELOG AR018 节位于 AR017 前；AGENTS #37 含 AR018 扩注（record_precomputed/快照 fresh）；review M-1/M-2 修正值在文（4.81/9.84/9.12/18.17 与 0.31-1.55/0.06-0.13）

**期望结果：**
- 全部标记检查 True

**实际结果：** 6/6 True（2026-10-14 执行）；数字与工件逐值对拍已由 review 门控完成（指南/CHANGELOG 全部数字逐位一致）

**状态：** PASS

---

### ST-016：非功能需求（质量门 + demo 锚点 + 兼容）

**关联需求：** srs.md §4（NFR 全表）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m mypy actfold --ignore-missing-imports`
2. `python -X utf8 -m pyflakes actfold tests demo.py scripts`
3. 100 列逐行复核（7 触碰文件）
4. `python -X utf8 -m pytest tests\ -q -m "not slow"` 全量
5. `python -X utf8 demo.py` 锚点
6. `git status` 佐证 `cuda_graph.py` 零改动、既有测试零改动

**期望结果：**
- mypy clean（66 文件）；pyflakes 零；100 列零；846+新增 passed 零回归；demo 三指标精确一致（85.5%/2.35e-03/93.75%）；`record` 签名零改动（UT-1708 钉住）；无新开关（无 config 改动）

**实际结果：** mypy "Success: no issues found in 66 source files"；pyflakes 零输出；100 列零违例；全量 **855 passed（846+9）/3 skipped** 零回归（ST 当日新鲜复跑一致）；demo 85.5%/2.35e-03/93.75%（T004 与 review 门控两次独立执行均精确一致）；git status 仅两生产文件+新测试/脚本/工件/文档——`cuda_graph.py` 与既有测试零改动

**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-14 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮（一次成 Go） |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR1 | 批量稳定度归约（record_precomputed） | ST-001, ST-002, ST-003, ST-004 | PASS |
| §3.2 FR2 | 快照发布（cache 写入路径） | ST-005, ST-006, ST-007, ST-008 | PASS |
| §3.3 FR3 | 对拍与观测钉（UT-1701~1709） | ST-009, ST-010, ST-011 | PASS |
| §3.4 FR4 | 墙钟基准与工件 | ST-012, ST-013, ST-014 | PASS |
| §3.5 FR5 | 文档收口 | ST-015 | PASS |
| §4 NFR | 质量/稳定/兼容非功能需求 | ST-016 | PASS |

**需求覆盖率：** 6 / 6（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 8 | 8 | 0 | 0 |
| 边界条件 | 3 | 3 | 0 | 0 |
| 异常处理 | 1 | 1 | 0 | 0 |
| 回归测试 | 4 | 4 | 0 | 0 |
| **合计** | **16** | **16** | **0** | **0** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | 工件 `meta.command` 未逐字含 `--reps 30`（脚本默认值抑制逻辑；实际运行即 `--reps 30`，`meta.reps: 30` 在案） | 脚本已修（未来运行恒含）；工件按墙钟纪律不手改（review M-3 注记） |
| Minor | 本机无 bench 依赖，无 marker 的 pytest 有 2 个 slow 用例 `ModuleNotFoundError: lm_eval`（既有环境事实，与本 AR 无关） | 注记（review M-4）；全量口径恒为 `-m "not slow"` |
| Minor | (256,8,4) 的 H1 2σ 功效限制（AR017 同配置先例；delta 稳定方向正） | 记录在指南 Part 25 遗留 |

### 结论

> **Go**——16/16 用例 PASS，需求覆盖 6/6（100%），无 Critical/Major 缺陷，全量回归零回归（855 passed），demo 锚点精确一致，质量门全过；3 条 Minor 均为注记级（工件不手改纪律/既有环境事实/统计功效记录），已记录延后。按用户既定全自动指令归档并推送。
