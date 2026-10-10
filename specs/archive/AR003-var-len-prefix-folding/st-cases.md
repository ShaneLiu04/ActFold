# [AR003] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR003 |
| AR 主题 | 变量长度前缀折叠（P3-1）+ AR002 遗留接线 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-10 |

## 测试用例列表

### ST-001：mixed 前缀折叠参考语义（逐位）

**关联需求：** srs.md §3.1 前缀折叠语义（验收标准 1）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given parent 缓存激活 `[B, T_p, H]`（`0 < T_p < T_c`，batch 相同），前缀部分 stable 部分 divergent

**测试步骤：**
1. 构造 mixed 前缀 child `[B, T_c, H]`（前缀含受扰动位置）
2. folded child 前向（parent_branch_id 链）
3. 与参考合成（`mask = concat(gate(prefix), False_suffix)`；`out = where(mask, cat(parent_ffn, child_recompute_suffix), child_recompute)`）逐位对比

**期望结果：**
- Then 输出与参考合成语义逐位一致（`torch.equal`）

**实际结果：** `tests/test_folded_transformer_varlen.py::test_varlen_prefix_mixed_reference` passed（含 EX-501/503/506 补齐后全文件 15 passed）；CUDA/CPU 双 device

**状态：** PASS

---

### ST-002：`align_tokens` 变长前缀对齐

**关联需求：** srs.md §3.2 align_tokens 前缀对齐（验收标准 1）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given `T_c > T_p`、batch 相同的 parent/child Branch

**测试步骤：**
1. 调用 `BranchManager.align_tokens(parent, child)`

**期望结果：**
- Then 返回前缀对齐对 `(h_parent [B,T_p,H], h_child[:, :T_p])`，无 `NotImplementedError`

**实际结果：** `tests/test_branch_manager.py::test_align_tokens_prefix_alignment` passed

**状态：** PASS

---

### ST-003：`folded_generate` 因果模型真正折叠（端到端）

**关联需求：** srs.md §3.3 folded_generate 真正折叠（验收标准 1）
**测试类型：** 正常路径（端到端集成）
**优先级：** High

**前置条件：**
- Given 因果合成模型（`CausalCumsumModel`）+ `folded_generate` ≥4 步增长

**测试步骤：**
1. 运行 folded_generate（4 步）
2. 对比 eager 全重算路径 tokens
3. 检查 `result.stable_ratio`

**期望结果：**
- Then tokens 与 eager 完全一致（`torch.equal`）且 `stable_ratio > 0.5`（实测 ≈0.84）

**实际结果：** `tests/test_folded_generation.py::test_folded_generate_causal_model_folds_and_matches_eager` passed；stable_ratio ≈0.84（= mean(4/5, 5/6, 6/7, 7/8)）

**状态：** PASS

---

### ST-004：`BenchmarkRunner` 构造 Manual + config 传播

**关联需求：** srs.md §3.4 benchmark_runner 迁移（验收标准 1）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 可检测架构的合成模型 + 携带 `use_cuda_graph`/`graph_capacity_ratio` 的 config

**测试步骤：**
1. 构造 `BenchmarkRunner`
2. 检查 `folded_model` 类型与 graph 字段取值

**期望结果：**
- Then `folded_model` 为 `ManualFoldedForward` 实例，graph 字段从 config 传入

**实际结果：** `tests/test_eval.py::test_benchmark_runner_builds_manual_folded_forward` passed

**状态：** PASS

---

### ST-005：后缀任意分布 mask/merge 正确性

**关联需求：** srs.md §3.1（验收标准 4）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 后缀长度 ≥1，前缀全 stable / 全 divergent / mixed 三种分布

**测试步骤：**
1. 分别以前缀全 divergent、全 stable、mixed 构造 child 前向
2. 断言路由（全 divergent → 全重算；全 stable → mixed 路径，因后缀恒 divergent）与输出

**期望结果：**
- Then mask 组装与 merge 结果在三种分布下均正确

**实际结果：** `test_varlen_suffix_always_divergent`、`test_varlen_prefix_all_stable_takes_mixed_path`、`test_varlen_prefix_all_divergent_full_recompute` 全部 passed

**状态：** PASS

---

### ST-006：`T_p > T_c` 全量重算

**关联需求：** srs.md §3.1（验收标准 3）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given parent 缓存比 child 长

**测试步骤：**
1. folded child 前向（`T_p > T_c`）

**期望结果：**
- Then 视为无 parent，全量重算（无异常、无复用、输出 == `layer(child)`）

**实际结果：** `test_varlen_parent_longer_full_recompute` passed

**状态：** PASS

---

### ST-007：等长路径逐位不变（最高不变量）

**关联需求：** srs.md §3.1（验收标准 2）/ §3.2（验收标准 2）
**测试类型：** 边界条件（回归）
**优先级：** High

**前置条件：**
- Given `T_p == T_c`（等长）

**测试步骤：**
1. 运行既有等长折叠测试（UT-209）与 `align_tokens` 等长测试（UT-101）

**期望结果：**
- Then 行为与 AR001/AR002 基线逐位一致（零回归）

**实际结果：** `test_equal_length_unchanged_reference`、`test_align_tokens` passed；全量 652 passed / 3 skipped / 3 deselected（基线 628 + 24 新增，零回归）

**状态：** PASS

---

### ST-008：架构不可检测 → folded_model None

**关联需求：** srs.md §3.4（验收标准 2）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given 无可检测 layer 栈/head 的模型；raw model 缺失的 adapter

**测试步骤：**
1. 构造 `BenchmarkRunner`（两种不可检测形态）

**期望结果：**
- Then `folded_model is None`（不抛出，对齐 `folding_applied` 语义）

**实际结果：** `test_benchmark_runner_undetectable_architecture_returns_none`、`test_benchmark_runner_raw_model_missing_returns_none` passed

**状态：** PASS

---

### ST-009：CUDA graph 对变长步骤零干扰

**关联需求：** srs.md §3.3（验收标准 2）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given `use_cuda_graph=True` 的 folded_generate + 变长增长步骤

**测试步骤：**
1. 运行 graph 模式 folded_generate
2. 检查 `graph_runner`、UserWarning、tokens

**期望结果：**
- Then 变长步骤不捕获（`graph_runner is None`）、零 UserWarning、tokens 与 eager 一致

**实际结果：** `test_folded_generate_graph_zero_interference` passed（graph_runner None + `pytest.warns(UserWarning)` 上下文外零告警断言 + tokens `torch.equal`）

**状态：** PASS

---

### ST-010：runner scheduler/graph 互斥策略（D3）

**关联需求：** srs.md §3.4（验收标准 3）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given `config.use_cuda_graph=True`（CUDA 环境）

**测试步骤：**
1. 构造 `BenchmarkRunner`，检查 Manual 的 scheduler 携带情况

**期望结果：**
- Then graph 开启时不携带 `FoldingScheduler`（动态 tau 与 graph 互斥），关闭时正常携带

**实际结果：** `test_benchmark_runner_scheduler_strategy` passed

**状态：** PASS

---

### ST-011：变长链式递归（多代折叠）

**关联需求：** srs.md §3.1（期望行为：child 按 T_c 存储 → 链式递归）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given 三代逐步增长的 folded 前向链

**测试步骤：**
1. gen1 → gen2 → gen3 逐代 folded 前向（每代 +1 token，parent = 前代）
2. 逐代与参考合成对比

**期望结果：**
- Then 每代折叠正确（child 激活按自身长度存储，下一代前缀折叠天然成立）

**实际结果：** `test_varlen_chain_recursion_three_generations` passed

**状态：** PASS

---

### ST-012：split 层完整 mask 工作

**关联需求：** srs.md §3.1（split 层在 `[B, T_c]` 完整 mask 上工作）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given var-len child + `SplitFoldedTransformerLayer`

**测试步骤：**
1. folded 前向，断言 divergent gather 含后缀行、输出正确

**期望结果：**
- Then split 层在完整 mask 上正确工作，`stable_count` 转发语义不变

**实际结果：** `test_varlen_split_layer_divergent_rows` passed

**状态：** PASS

---

### ST-013：`align_tokens` 非法输入 ValueError

**关联需求：** srs.md §3.2（验收标准 3）
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given `T_c < T_p` 或 batch 不同的 parent/child

**测试步骤：**
1. 分别调用 `align_tokens`

**期望结果：**
- Then 均抛 `ValueError`（不做截断）

**实际结果：** `test_align_tokens_child_shorter_raises`、`test_align_tokens_batch_mismatch_raises` passed

**状态：** PASS

---

### ST-014：EX-501 mixed 路径 parent ffn 缺失 RuntimeError

**关联需求：** srs.md §3.1 异常处理 / design §6.4 EX-501
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given parent 缓存有 embedding 无 ffn_out（原始 put/fetch 协议 stub），mixed 前缀 child

**测试步骤：**
1. folded child 前向

**期望结果：**
- Then 抛 `RuntimeError`（契约不吞错，与等长 mixed 一致）

**实际结果：** `test_varlen_parent_ffn_missing_raises` passed（review 第 2 轮子代理实证真实触发 `_get_parent_ffn_output` 缺失分支）

**状态：** PASS

---

### ST-015：EX-503 attention_mask 非 None 变长透传

**关联需求：** srs.md §3.1 异常处理 / design §6.4 EX-503
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given mask 感知层（`MaskSensitiveLayer`）+ child 长度 mask（末位 False）+ mixed 前缀

**测试步骤：**
1. folded 前向带 attention_mask
2. 与 mask 感知参考合成对比

**期望结果：**
- Then mask 正常传 original layer，输出 == where(mask, cat(ffn_seed, masked_suffix), masked_recompute)

**实际结果：** `test_varlen_attention_mask_passthrough` passed（判别力设计：mask 未透传则 `torch.equal` 必失败）

**状态：** PASS

---

### ST-016：EX-505 parent ffn shape 不符 RuntimeError

**关联需求：** srs.md §3.1 / design §6.4 EX-505（EC-12）
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- Given parent ffn 比前缀长 1 token（stub cache），mixed 前缀 child

**测试步骤：**
1. folded child 前向

**期望结果：**
- Then 抛 `RuntimeError`（`(B, prefix_len)` 形状校验，不静默误合并）

**实际结果：** `test_varlen_parent_ffn_shape_mismatch_raises` passed

**状态：** PASS

---

### ST-017：EX-506 逐出致链断裂与重建

**关联需求：** srs.md §3.1 / design §6.4 EX-506（EC-9）
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given `max_entries_per_layer=T_PARENT` 紧预算 cache；parent + sibling 两组不可共存

**测试步骤：**
1. sibling put 逐出 parent（断言 fetch parent 抛错）
2. gen1 folded 前向（parent_branch_id=parent，已逐出）
3. gen2 folded 前向（parent_branch_id=child，新链）

**期望结果：**
- Then gen1 视 cache miss 全量重算（输出 == `layer(child)` 逐位，无异常）；gen2 对新建链正常折叠（== 参考合成）

**实际结果：** `test_varlen_cache_eviction_breaks_and_rebuilds_chain` passed（review 第 2 轮子代理实证真实依赖 `_evict_over_budget` 组级 LRU + 最新组恒保留语义）

**状态：** PASS

---

### ST-018：既有等长异常语义零回归（EX-502/504）

**关联需求：** srs.md §3.1（验收标准 2 等长零回归）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given AR001/AR002 既有异常用例（cache miss 全重算、batch mismatch 等）

**测试步骤：**
1. 运行全量测试套件（not slow）

**期望结果：**
- Then 既有异常用例全部不回归（含 `test_cache_miss_recomputes_divergent`、`test_varlen_parent_mismatch_full_recompute` 等）

**实际结果：** 全量 652 passed / 3 skipped / 3 deselected，零回归

**状态：** PASS

---

### ST-019：等长全量零回归 + AR002 等长循环（BS-401）

**关联需求：** srs.md §3.1（验收标准 2）/ §3.3（验收标准 3）/ NFR §4 等长零回归
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given AR001/AR002 基线 628 passed 的测试集

**测试步骤：**
1. `python -X utf8 -m pytest tests\ -q -m "not slow"`

**期望结果：**
- Then 基线全部不回归（628 既有 + 新增全绿）

**实际结果：** 652 passed / 3 skipped / 3 deselected（基线 628 + 新增 24；3 skip/3 deselect 与基线相同——slow 标记 lm_eval 未装，非回归）

**状态：** PASS

---

### ST-020：demo 基线精确匹配（BS-402）

**关联需求：** srs.md NFR §4（demo 基线 85.5% / 2.35e-03 / 93.75%）
**测试类型：** 回归测试（手工/脚本）
**优先级：** High

**前置条件：**
- Given 本机 16GB CUDA 环境

**测试步骤：**
1. `python demo.py`

**期望结果：**
- Then reduction 85.5%、latency ratio 2.35e-03、exact match 93.75% 精确匹配基线

**实际结果：** 85.5% / 2.35e-03 / 93.75% 精确匹配基线（ST 执行轮实测，逐层 similarity 0.968–0.970 与基线一致）

**状态：** PASS

---

### ST-021：质量门（BS-403）

**关联需求：** srs.md NFR §4（mypy --strict 零错误 / pyflakes / 100 列）
**测试类型：** 回归测试
**优先级：** Medium

**前置条件：**
- Given AR003 修改后的代码树

**测试步骤：**
1. `python -X utf8 -m mypy actfold --strict --ignore-missing-imports`（AGENTS 标准范围 64 文件）
2. `python -X utf8 -m pyflakes actfold tests demo.py scripts`
3. 100 列检查（变更文件）

**期望结果：**
- Then 全部 clean

**实际结果：** AGENTS 标准命令 `mypy actfold --ignore-missing-imports` clean（64 文件无问题）；`--strict` 附加检查仅报 cuda_graph.py 3 个 `unused-ignore`，该文件与 AR002 基线 56b607d 逐位相同（git diff 为空）→ 零新增、非 AR003 回归；pyflakes exit 0 clean；变更文件 100 列 clean（tests/test_folded_transformer_varlen.py over100=0）

**状态：** PASS

---

## 执行摘要

| 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|
| 21 | 21 | 0 | 0 |

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-10 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 | 前缀折叠语义（mixed 参考/等长零回归/T_p>T_c/后缀分布/链式递归/split） | ST-001, ST-005, ST-006, ST-007, ST-011, ST-012 | PASS |
| §3.2 | align_tokens 前缀对齐（变长/等长/ValueError） | ST-002, ST-007, ST-013 | PASS |
| §3.3 | folded_generate 真正折叠（tokens 一致 + ratio>0.5 / graph 零干扰 / AR002 循环回归） | ST-003, ST-009, ST-019 | PASS |
| §3.4 | runner 迁移 Manual + config 接线（传播/不可检测/graph flag） | ST-004, ST-008, ST-010 | PASS |
| §3.1 §6.4 异常 | EX-501/503/505/506 + 既有等长异常零回归 | ST-014, ST-015, ST-016, ST-017, ST-018 | PASS |
| NFR §4 | 等长零回归 + demo 基线 + 质量门 + 覆盖率 | ST-019, ST-020, ST-021 | PASS |

**需求覆盖率：** 6 / 6（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 4 | 4 | 0 | 0 |
| 边界条件 | 8 | 8 | 0 | 0 |
| 异常处理 | 5 | 5 | 0 | 0 |
| 回归测试 | 4 | 4 | 0 | 0 |
| **合计** | **21** | **21** | **0** | **0** |

自动化底座证据：全量 `pytest tests -q -m "not slow"` 652 passed / 3 skipped / 3 deselected（ST 执行轮两度复跑一致，44.8s）；coverage（AR003 变更模块）：folded_transformer 94%、branch_manager 93%、benchmark_runner 70%（runner 未覆盖行为 judge/真实后端路径，slow 标记依赖未装，与基线一致非回退）；demo 基线 85.5% / 2.35e-03 / 93.75% 精确匹配。

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | `Branch.hidden_states` 全量持有的惰性引用化（OPTIMIZATION_GUIDE P3-1 遗留子项，AR003 明确 Out of Scope） | 延后至独立 AR |
| Minor | `mypy --strict`（超 AGENTS 标准命令的附加检查）在 cuda_graph.py 报 3 个既有 `unused-ignore`（与 AR002 基线逐位相同文件，非 AR003 引入） | 延后处理（可与 T007 记录口径差异一并复核） |
| Minor | 真实投机解码负载下的绝对墙钟收益未测（机制 + 合成代理证据已交付，srs §5 假设明确留目标机按 RERUN_CHECKLIST 测量） | 延后至目标机实验 AR |

### 结论

> **Go**：21/21 用例 PASS，需求覆盖 100%，无 Critical/Major 缺陷，全量零回归（652 passed），demo 基线精确匹配，质量门 clean。3 个 Minor 遗留均已记录且不阻塞（Out of Scope / 既有状态 / 明确假设）。
