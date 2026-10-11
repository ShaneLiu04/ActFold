# AR007 系统测试验收 — parent 更长时的截断前缀复用（P3-2 第一档）

> 执行环境：Windows / Quadro RTX 5000 16GB / torch 2.5.1+cu121 / Python 3.10+
> 全部用合成 stub/fixture；真实 checkpoint 属遗留（paged 多祖先）。UT/IT 已在开发阶段回归全过；
> ST 阶段按验收标准实测复验 + 端到端回归（review 子代理独立复测同值）。

## 追溯矩阵

| ST 用例 | 源标准 | 类型 | 对应用例/测试 | 结果 |
|---------|--------|------|---------------|------|
| ST-001 | 3.1-AC1 真前缀（parent 8 → child 6）全 stable 复用，`torch.equal` 全重算 + profiler `layer_stats[*].stable_ratio == 1.0`（W1 口径） | 单元 | UT-601 `test_folded_transformer_varlen.py::TestAR007TruncatedParentReuse::test_ut_601_true_prefix_all_stable_reuse` | PASS |
| ST-002 | 3.1-AC2 非前缀独立采样 child（W2 构造）→ gate 全 divergent → 全重算 bit-exact | 单元 | UT-602 `test_ut_602_non_prefix_child_full_recompute`（含 gate sanity 前置断言） | PASS |
| ST-003 | 3.1-AC3 batch 不匹配（parent 更长）→ 仍视为无 parent 全重算，无异常 | 单元 | UT-603 `test_ut_603_batch_mismatch_still_no_parent`（`get_profile("child") is None` 证明 no-parent 分支） | PASS |
| ST-004 | 3.1 复用真实性：全 stable 时快路径复用截断 parent FFN 而非全重算 | 单元 | UT-604 `test_ut_604_fast_path_reuse_without_recompute`（warm-up 后 spy `_recompute_all` 计数 == 0 + 输出 == parent_out[:, :6]） | PASS |
| ST-005 | 3.1 异常 cache 状态：跨层长度错位（gate == child + merge < child）→ 既有 RuntimeError 机制 | 单元 | UT-605 `test_ut_605_cross_layer_misaligned_parent_ffn_raises`（混合 mask 前置断言钉死慢路径必经） | PASS |
| ST-006 | 3.2 端到端：`ManualFoldedForward` 8-token parent 填 cache → 6-token 真前缀 child 折叠 | 集成 | IT-611 `test_cuda_graph.py::test_it_611_long_parent_true_prefix_end_to_end`（`torch.equal` + 全层 stable_ratio==1.0 + cache 发布 [1,6,H]） | PASS |
| ST-007 | 3.2 既有路径零回归：append-only 全套 + 等长全套逐位不变；EX-505 varlen RuntimeError 保留 | 回归 | IT-612：`test_folded_transformer_varlen.py` 20/20（15 既有 + 5 新增）；varlen 分支与 `git show HEAD` 逐字节对照（review 代理） | PASS |
| ST-008 | 3.2 CUDA graph 零干扰：更长 parent 步骤 silent eager 回退，不捕获/零告警/不禁用 | 集成 | IT-613 `test_it_613_long_parent_silent_eager_no_capture`（四重断言：`graph_runner is None` + `_graph_capture_failed is False` + 零 UserWarning + 后续 complete-parent child 仍捕获） | PASS |
| ST-009 | 3.2 全量回归 | 回归 | `pytest tests\ -q -m "not slow"` = **761 passed / 3 skipped / 3 deselected**（754 基线 + 7 新增，纯增量无删除；review 代理独立复测同值 50.96s） | PASS |
| ST-010 | 3.3/NFR demo 基线 | 回归 | `python -X utf8 demo.py` = 85.5% / 2.35e-03 / 93.75% 精确一致（demo 默认模式不受影响） | PASS |
| ST-011 | 3.3 文档收口 | 审查 | grep 实证：AGENTS #20 截断复用段、README 局限性 #7、指南 :199 勾选 + 第十四部分、CHANGELOG AR007 节、ALGORITHM.md §11 修订 + §12 新增、UT-203/bs001/AR007 类三处 docstring 清理 | PASS |
| ST-012 | NFR 质量门 | 静态 | mypy 标准命令 65 files clean / pyflakes 零输出 / 改动文件 100 列 0 违例（review 代理独立复测） | PASS |

## 执行记录

- 2026-10-11（T001–T004 实测 + review 子代理独立复测同值）
  - 全量：`761 passed, 3 skipped, 3 deselected, 25 warnings in 46.07s`（review 复测 50.96s）
  - demo：`FLOPs reduction 85.5% / MSE 2.35e-03 / stable 93.75%`
  - mypy：`Success: no issues found in 65 source files`；pyflakes 零输出；改动文件 100 列 clean
  - 定向：test_folded_transformer_varlen.py 20 passed（15 既有 + UT-601~605）；test_cuda_graph.py IT-611/613 passed（CUDA 实跑）
- 结论：**12/12 PASS**

## 遗留说明

- 全量 3 skipped 为 slow 标记（bench 依赖未装）；无关回归。
- paged 多祖先结构（P3-2 主体）与 `Branch.hidden_states` 惰性引用化为后续 AR 候选；
  `gather_select` 对 parent 更长实际正确（vectorized_cache.py:376-378 行公式）留作性能 AR 线索。
