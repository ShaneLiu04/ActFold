# AR008 系统测试验收 — generic.forward `output_hidden_states` 按需化

> 执行环境：Windows / Quadro RTX 5000 16GB / torch 2.5.1+cu121 / Python 3.10+
> 全部用合成 stub/fixture；真实 7B checkpoint 属遗留（RERUN_CHECKLIST）。UT/EX 已在开发阶段回归全过；
> ST 阶段按验收标准实测复验 + 端到端回归（review 子代理独立复测同值）。

## 追溯矩阵

| ST 用例 | 源标准 | 类型 | 对应用例/测试 | 结果 |
|---------|--------|------|---------------|------|
| ST-001 | 3.1-① logits-only 模型：每次底层调用 `output_hidden_states is False` + 返回逐位一致 | 单元 | UT-701 `test_generic_forward.py::test_ut_701_logits_only_passes_output_hidden_states_false`（call_log spy × 2 次调用） | PASS |
| ST-002 | 3.1-② last_hidden_state-only：False 传递 + `lm_head(last_hidden)` 逐位 | 单元 | UT-702 `test_ut_702_last_hidden_only_passes_false_and_projects` | PASS |
| ST-003 | 3.1-③ hidden_states-only：首跑恰好双调（False→True）+ 次跑单调 1 调 True（标志缓存） | 单元 | UT-703 `test_ut_703_hidden_states_only_reruns_once_then_caches_flag`（旗标条件式 stub：False → hidden_states=None，模拟真实 HF 语义） | PASS |
| ST-004 | 3.1-④ 三态全无 + 空元组变体：RuntimeError 逐字一致 + 恰好 2 次调用（不无限重跑） | 单元 | UT-704 参数化两场景（no-output-fields / empty-hidden-tuple），`str(excinfo.value) ==` 逐字比对 | PASS |
| ST-005 | 3.1-⑤ branch 键剥离在重跑下保持（两调 kwargs 均无 branch 键） | 单元 | UT-705 `test_ut_705_branch_kwargs_stripped_from_both_calls` | PASS |
| ST-006 | 3.3 内存峰值：False/True 实例对照 CUDA 峰值差 ≥ 150MB | 实证 | EX-805 `test_ex_805_on_demand_flag_cuts_peak_memory`（24 层无参数 stub，理论 192MiB；实测通过） | PASS |
| ST-007 | 3.2 既有 generic 消费链零回归（含 LLaDA/Dream 继承链） | 回归 | IT-806：test_models.py 10 passed（零 diff）；llada.py/dream.py/base.py/diffusion_sampler.py 零 diff（review 代理核验） | PASS |
| ST-008 | 3.2 全量回归 | 回归 | `pytest tests\ -q -m "not slow"` = **768 passed / 3 skipped / 3 deselected**（761 基线 + 7 新增纯增量；review 代理独立复测同值） | PASS |
| ST-009 | 3.2/NFR demo 基线 | 回归 | `python -X utf8 demo.py` = 85.5% / 2.35e-03 / 93.75% 精确一致 | PASS |
| ST-010 | 3.4 文档收口 | 审查 | grep 实证：指南 6.2 勾选 + 第十五部分、CHANGELOG AR008 节；口径统一（256MB/192MiB/150MB） | PASS |
| ST-011 | NFR 质量门 | 静态 | mypy 标准命令 65 files clean / pyflakes 零 / 改动文件 100 列 0 违例（review 代理独立复测） | PASS |

## 执行记录

- 2026-10-11（T001–T004 实测 + review 子代理独立复测同值）
  - 全量：`768 passed, 3 skipped, 3 deselected, 25 warnings in 44.99s`
  - demo：`FLOPs reduction 85.5% / MSE 2.35e-03 / stable 93.75%`
  - mypy：`Success: no issues found in 65 source files`；pyflakes 零输出；改动文件 100 列 clean
  - 定向：test_generic_forward.py 7 passed（UT-701~705 含 UT-704 两参数化场景 + EX-805 CUDA 实跑）；test_models.py 10 passed
  - EX-805 实测：peak_on − peak_off ≥ 150MiB（24×512×4096×4B 理论 192MiB，阈值留 ~22% 余量）
- 结论：**11/11 PASS**

## 遗留说明

- 全量 3 skipped 为 slow 标记（bench 依赖未装）；无关回归。
- 真实 7B checkpoint 内存峰值实测按 RERUN_CHECKLIST 流程执行（本机无权重；EX-805 合成 stub 已代理实证）。
- black 26.10.1 对仓库 46 个既有文件存在版本漂移重排（非 AR008 引入，未触碰）。
