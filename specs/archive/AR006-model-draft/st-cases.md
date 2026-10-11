# AR006 系统测试用例（真 draft 模型第一档：`model_draft` 模式与合成验证）

> 测试执行环境：Windows / Quadro RTX 5000 16GB / torch 2.5.1+cu121 / Python 3.10+
> 全部用合成 stub/fixture（无 checkpoint 依赖）；UT/IT 已在开发阶段落地并全绿，
> ST 阶段复用已实测证据 + 端到端回归重跑（review 子代理独立复跑同值）。

## 追溯矩阵

| ST 编号 | 源需求 | 测试类型 | 对应测试/命令 | 结果 |
|---------|--------|----------|---------------|------|
| ST-001 | §3.1-1 flip 采样 ∈ draft top-k + seed 复现 | 单元 | UT-501 `test_draft_generator.py::test_model_draft_flip_tokens_in_stub_topk` | PASS |
| ST-002 | §3.1-2 draft_model 缺失 → generate 期 RuntimeError（构造期不报错） | 单元 | UT-502/EX-801 `test_model_draft_requires_draft_model_at_generate_time` | PASS |
| ST-003 | §3.1-3 三项校验（形状/int dtype/跨 device）→ ValueError；forward 抛错传播 | 单元 | UT-503a/b/c + EX-803 `test_model_draft_rejects_*` / `test_model_draft_forward_error_propagates` | PASS |
| ST-004 | §3.1-1 parent_logits 忽略（结果 == 不传）；append ∈ 末位 top-k；RNG 不污染 | 单元 | UT-504/505 + `test_model_draft_seed_does_not_pollute_global_rng` | PASS |
| ST-005 | §3.2 model_draft − random ≥ 0.1（top_k=2，16 flip 位） | 单元 | UT-506 `test_model_draft_acceptance_beats_random`（实测 gap 0.452） | PASS |
| ST-006 | §3.2 model_draft ≤ logits_draft 实测上界（非恒 1.0） | 单元 | UT-507 `test_model_draft_acceptance_below_logits_draft_upper_bound`（实测 0.45 ≤ 0.5） | PASS |
| ST-007 | §3.2 无关 draft 与 random 之差 < 0.05 | 单元 | UT-508 `test_uncorrelated_draft_acceptance_near_random_floor`（实测 ≈0.001） | PASS |
| ST-008 | §3.3-1 AblationStudy model_draft 端到端 | 集成 | IT-511 `test_ablation_measured.py::test_it511_ablation_with_model_draft_generator` | PASS |
| ST-009 | §3.3-1 folded_generate 多候选路径 | 集成 | IT-512 `test_folded_generation.py::test_it512_folded_generate_model_draft_candidates` | PASS |
| ST-010 | §3.3-1 BenchmarkRunner 注入 + EX-804 构造期 RuntimeError + passthrough 正例 | 集成 | `test_eval.py::test_benchmark_runner_model_draft_requires_draft_model` / `test_benchmark_runner_model_draft_passthrough` | PASS |
| ST-011 | §3.3-2 全量零回归 | 回归 | `pytest tests\ -q -m "not slow"` = **754 passed / 3 skipped / 3 deselected**（738 基线 + 16 新增；review 代理独立复跑同值） | PASS |
| ST-012 | §3.4 文档收口 | 评审 | grep 实证：指南 M-2 :213/:282 三层全达 + 第十三部分、README 局限性 #5/#8、AGENTS #40 tier-1 段、CHANGELOG AR006 节 | PASS |
| ST-013 | NFR demo 基线 | 回归 | `python -X utf8 demo.py` = 85.5% / 2.35e-03 / 93.75% 精确一致（demo 走默认模式） | PASS |
| ST-014 | NFR API 兼容 | 单元 | draft_model 默认 None + 既有 19 旧测试全过 + diff 级复核（五模式逐位不变：logits_draft 等值替换、其余分支逐字节未动） | PASS |
| ST-015 | NFR 质量门 | 静态 | mypy 标准命令 65 files clean / pyflakes clean / 改动文件 100 列 0 违例 | PASS |

## 执行记录

- 2026-10-10（T005 实测 + review 子代理独立复跑同值）：
  - 全量：`754 passed, 3 skipped, 3 deselected, 25 warnings in 41.85s`
  - demo：`FLOPs reduction 85.5% / MSE 2.35e-03 / stable 93.75%`
  - mypy：`Success: no issues found in 65 source files`；pyflakes 零输出；100 列零违例
  - 目标测试集：test_draft_generator.py 31 passed（19 旧 + 12 新）、IT-511/512、EX-804 + passthrough 4 passed
  - 接受率梯度实测：model_draft 0.453 / random 0.001 / logits_draft 0.5（top_k=2）
- 结论：**15/15 PASS**

## 遗留说明

- 全量中 3 skipped 为 slow 标记（bench 依赖未装）；无过滤时 2 个 `lm_eval`
  ModuleNotFoundError 为环境缺依赖（AR003 后零改动），非回归。
- Medusa/Eagle checkpoint 加载属第二档（需真实权重）；append 位置从 parent
  末位 draft logits 采样是第一档近似（逐 token 自回归属第二档）。
- fixture 结构教训（指南第十三部分已记录）：one-hot-cumsum + 单位阵残差的
  共享结构会使"随机权重"draft 也 ~0.4 相关——无关 draft 必须去 I 残差。
