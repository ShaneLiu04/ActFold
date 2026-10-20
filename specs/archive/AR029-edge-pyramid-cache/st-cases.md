# AR029 ST 验收用例与执行记录

**AR**：AR029 端侧有界内存金字塔激活缓存（机制/基础设施交付）
**日期**：2026-10-20
**范围裁定**：真实 checkpoint A/B = BLOCKED-DEFERRED（`baseline_claim=false`）
**执行环境**：Windows 11 / Python 3.11.9 / CPU（quick 机制面）；无 GPU 实验

## 需求覆盖

| FR | 需求 | 覆盖用例 | 结果 |
|----|------|----------|------|
| FR1 | 逐层预算计划（归一化/块对齐/回收/边界） | ST-01, ST-02, UT-2901–2910 | PASS |
| FR2 | 块页 + 逐层硬预算缓存（淘汰/锚点/条目删除/统计） | ST-03, ST-04, UT-2911–2925, 2969–2972 | PASS |
| FR3 | 缺失 token 安全语义（NaN + availability + 目标层守卫） | ST-05–ST-09, UT-2926–2944, 2963b, 2974–2976, 2978 | PASS |
| FR4 | 协议与回归（全量预算逐位、put_batch、旧缓存 no-op） | ST-10, ST-11, UT-2934, 2945–2952 | PASS |
| FR5 | 可观察内存上界（stats 六键 + bytes 账本） | ST-04, UT-2918, 2970 | PASS |
| FR6 | 工厂/配置/接线（默认关闭、优先级、真实深度） | ST-12, UT-2953–2958, 2973 | PASS |
| FR7 | 机制基准与真实面协议（E1–E4、quick-only、BLOCKED-DEFERRED） | ST-13–ST-15, UT-2959–2966, 2961b, 2967b, 2957b | PASS |

## 用例

### ST-01（正常）默认计划构建
- 命令：`python -m pytest tests/test_ar029_pyramid_plan.py -q`
- 期望：预算块对齐、单调、总和 ≤ total；`depth*block>total` 拒绝。
- 实测：20 passed。

### ST-02（边界）最小可行预算
- 命令：`python -m pytest tests/test_ar029_pyramid_plan.py::test_ut2971_minimum_feasible_budget -q`
- 期望：`depth=2,total=32,block=16` → `(16,16)`。
- 实测：PASS（UT-2971）。

### ST-03（正常）块页读写与部分尾块
- 命令：`python -m pytest tests/test_ar029_pyramid_cache.py -q`
- 期望：块切分/部分尾块 round-trip、NaN 缺失、availability 精确。
- 实测：35 passed。

### ST-04（边界）逐层硬上界与字节账本
- 命令：`python -m pytest tests/test_ar029_pyramid_cache.py::test_ut2912_layer_budget tests/test_ar029_pyramid_cache.py::test_ut2918_bytes_used -q`
- 期望：每层 tokens ≤ budget；`bytes_used == Σ block bytes`。
- 实测：PASS。

### ST-05（正常）目标层缺失 → 重算（串行）
- 命令：`python -m pytest tests/test_ar029_pyramid_integration.py::test_ut2935_target_layer_missing_matches_forced_reference -q`
- 期望：与 forced 参考逐位一致；关闭守卫后输出非有限/不等。
- 实测：PASS（含 mutation 断言）。

### ST-06（正常）同父批量守卫
- 命令：`...::test_ut2941_same_parent_batch_guard`
- 期望：共享父 availability 广播；关闭守卫后不等。
- 实测：PASS。

### ST-07（正常）多父逐行守卫
- 命令：`...::test_ut2942_multi_parent_row_guard`
- 期望：每行使用自己的 parent 洞（r1 全缺失、r2 仅尾部缺失），逐行 forced 参考逐位一致。
- 实测：PASS。

### ST-08（边界）layer 0（门读 embedding、守卫查目标 ffn）
- 命令：`...::test_ut2940b_layer_zero_guard_uses_target_ffn ...::test_ut2940_serial_mff_guard_single_layer`
- 期望：layer 0 与串行 MFF 入口均按目标层可用性重算。
- 实测：PASS。

### ST-09（异常）关闭守卫必须失败
- 命令：`python -m pytest tests/test_ar029_bench_verdicts.py::test_ut2963b_guard_off_output_is_not_honest tests/test_ar029_bench_verdicts.py::test_ut2978_guard_off_face_is_refuted -q`
- 期望：守卫恒等 → NaN/非有限 → E4 false（blocked）、E1 false。
- 实测：PASS。

### ST-10（正常）全量预算逐位一致
- 命令：`...::test_ut2934_serial_full_budget_matches_dense ...::test_ut2946_batched_full_budget_matches_dense`
- 期望：串行/批量输出与 `ActivationCache` `torch.equal`。
- 实测：PASS。

### ST-11（回归）旧缓存 no-op
- 命令：`...::test_ut2944_legacy_cache_guard_is_noop ...::test_ut2952_legacy_cache_guard_is_identity`
- 期望：无 availability 的缓存守卫恒等，既有路径逐字节不变。
- 实测：PASS。

### ST-12（异常）graph × pyramid 与深度缺失
- 命令：`...::test_ut2943_graph_pyramid_init_rejected ...::test_ut2967_setter_swap_rejected_without_mutation ...::test_ut2958_graph_pyramid_rejected_at_both_layers ...::test_ut2957_runner_resolves_only_real_layer_counts`
- 期望：构造期/setter/runner 均 `ValueError`；setter 失败不半传播；深度不猜。
- 实测：PASS。

### ST-13（正常）机制基准 quick
- 命令：`python -m scripts.ar029_pyramid_cache_bench --quick`
- 期望：`overall=supported`，E1–E4 全 true，写 `_quick.json`。
- 实测：`supported`，E1–E4 true；full/pressure diff 均 0.0；pressure reuse 0.625。

### ST-14（异常）正式面拒绝执行
- 命令：`python -m scripts.ar029_pyramid_cache_bench`（无 `--quick`）
- 期望：exit 3，不写正式工件。
- 实测：exit 3（BLOCKED-DEFERRED）。

### ST-15（回归）全量质量门
- 命令：`python -m pytest tests/ -q`；`black --check`；`isort --check-only`；`pyflakes`；`mypy actfold/scripts`
- 期望：1273 passed / 4 skipped / 3 deselected；格式/静态检查零告警；mypy 70+27 零问题。
- 实测：全部 PASS。

## 结论

- FR1–FR7 全部覆盖并通过；AR029 套件 94 passed，全量 1273 passed。
- 无伪造数字；机制工件 `baseline_claim=false`；真实 checkpoint 面按范围裁定
  记录为 BLOCKED-DEFERRED。
- review 门控 3 轮：FAIL（isort/测试同义反复/工件归因/零长写入/锚点公式/UT 覆盖）
  → FAIL（UT 映射与计数）→ **PASS**（代理 `ses_eaa9e949effeguI7O6bTJ641Nx`）。
