# AR005 系统测试用例（FFN/MoE FLOPs 几何修正收口）

> 测试执行环境：Windows / Quadro RTX 5000 16GB / torch 2.5.1+cu121 / Python 3.10+
> 全部用 config-like stub（无 checkpoint 依赖）；UT/IT 已在开发阶段落地并全绿，
> ST 阶段复用已实测证据 + 端到端回归重跑。

## 追溯矩阵

| ST 编号 | 源需求 | 测试类型 | 对应测试/命令 | 结果 |
|---------|--------|----------|---------------|------|
| ST-001 | §3.1-1 几何提取（Dense SwiGLU） | 单元 | UT-407 `test_flops_counter.py`（swiglu/moe_num_layers=29 推导含 UT-408） | PASS |
| ST-002 | §3.1-2 裸模块零回归 | 单元 | UT-409b/UT-405 无 MoE 分支（逐位 dense 一致） | PASS |
| ST-003 | §3.2-1 MoE per-token 公式 | 单元 | UT-401/UT-402（手算 `2·n_matmul·I·h·(top_k+shared)·L_moe·T_eff`）/UT-403 混合层数 | PASS |
| ST-004 | §3.2-2 缺参零回归 | 单元 | UT-405 + EX-705（`moe_num_experts` 单独给不激活，逐位 dense） | PASS |
| ST-005 | §3.2-3 域校验 | 单元 | EX-701/702/703/704（top_k≤0、top_k>num_experts、num_experts≤0、moe_inter≤0 → ValueError） | PASS |
| ST-006 | §3.3-1 engine tflops 真实几何 | 集成 | IT-411 `test_verification_engine.py`（== 真实几何手算 rel=1e-12 且 ≠ 4h 默认） | PASS |
| ST-007 | §3.3-1 ablation/_flops_budget | 集成 | IT-412a `test_ablation_measured.py` | PASS |
| ST-008 | §3.3-1 base_adapter TFLOPs 上报 | 集成 | IT-412b `test_base_adapter.py` | PASS |
| ST-009 | §3.3-2 合成模型逐位一致 | 回归 | 全量 `pytest tests\ -q -m "not slow"` = 738 passed / 3 skipped / 3 deselected（703 基线 + 35 新增零回归） | PASS |
| ST-010 | §3.4 文档收口 | 评审 | grep 实证：AGENTS #33 修订 / CHANGELOG AR005 节 / 指南 P3-4 ✅ + 第十二部分 / README 78.5→85.5 | PASS |
| ST-011 | NFR 零回归 demo | 回归 | `python -X utf8 demo.py` = 85.5% / 2.35e-03 / 93.75% 精确一致 | PASS |
| ST-012 | NFR API 兼容 | 单元 | UT-410（`_GEOMETRY_KEYS` 集合断言：7 键，只增不改）+ test_flops_counter_ffn.py 按键断言 | PASS |
| ST-013 | NFR 质量门 | 静态 | mypy 标准命令 65 files clean / pyflakes clean / 改动文件 100 列 0 违例 | PASS |

## 执行记录

- 2026-10-10（T005 实测 + review 子代理独立复核）：
  - 全量：`738 passed, 3 skipped, 3 deselected, 24 warnings in 40.97s`
  - demo：`FLOPs reduction 85.5% / MSE 2.35e-03 / stable 93.75%`
  - mypy：`Success: no issues found in 65 source files`
  - pyflakes：零输出；100 列：改动文件零违例
  - 目标测试集（flops_counter/models/engine/ablation/base_adapter）：88 passed
  - coverage：flops_counter.py 99%；base.py 新增 property 全覆盖
- 结论：**13/13 PASS**

## 遗留说明

- BS-601 全量中 3 skipped 为 slow 标记（bench 依赖未装）；无过滤时 2 个
  `lm_eval` ModuleNotFoundError 为环境缺依赖（AR003 后零改动），非回归。
- base.py 文件级 coverage 75%：未覆盖行为既有 `get_device`/`estimate_memory`
  等需真实模型权重的行，与本 AR 新增代码无关（新增 property 全覆盖）。
