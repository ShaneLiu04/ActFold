# AR006 任务分解 — 真 draft 模型第一档：`model_draft` 模式与合成验证

> srs：`specs/changes/AR006-model-draft/srs.md`
> 拆分原则：TDD（Red 子代理/Green 主代理分离，微型任务主代理自完成）；
> 每任务完成即在下方表格更新状态（pending / in_progress / passing）。

| ID | 任务描述 | 前置 | 状态 | 备注 |
|----|----------|------|------|------|
| T001 | `DraftGenerator` 新模式 `model_draft` + 构造参数 `draft_model`（callable tokens→logits，接受 nn.Module）：构造校验（mode 合法）、generate 期 draft_model 缺失 RuntimeError、draft forward 一次得 logits（不消费 parent_logits）、三项校验 ValueError（形状 / int dtype / 跨 device）、flip/append 采样复用 `_sample_from_topk`（logits_draft/model_draft 共用 `_resample_from_logits` helper）、seed 复现 + 全局 RNG 不污染、parent_logits 传入时忽略（文档注记） | — | passing | 2026-10-10 完成：Red 9 failed（8 TypeError + 1 Unsupported mode）/Green 28 passed；helper 命名 `_sample_tokens_from_logits`；`logits_source` 统一采样源（logits_draft=model_draft 共用路径，既有五模式逐位不变） |
| T002 | 合成非平凡性验证 fixture（top_k=2 统一，D9）：tiny causal target（4 层 token-wise，seed 确定性映射）+ tiny draft（target 前 1 层 + 小扰动投影，正相关不相等）；AR004 `acceptance_rate` 测量梯度（seq_len=64/flip_ratio=0.25 → ≥16 flip 位）：model_draft − random ≥0.1（P(hit)≥0.202）、≤ logits_draft 实测（理论 0.5）；无关 draft 与 random 之差 <0.05 | T001 | passing | 2026-10-10 完成：31 passed；实测 P(hit)=0.906（gap 0.452）；**fixture 结构缺陷修复**：无关 draft 去掉单位阵残差项（`x @ U.T`）——cumsum+I 共享结构会使"随机权重"draft 也 0.4 相关（probe 实证）；相关 draft bias 裸 tensor → Parameter（`.to(device)` 随行）。**Red 态说明**：T001 已 Green 后落地，无独立 Red；首跑 3 failed（device 迁移 + 结构相关 0.4），即等效 Red——fixture 缺陷由实测失败暴露并修复 |
| T003 | 端到端接线：`AblationStudy(draft_generator=model_draft generator)` measure_folding 正常、`folded_generate` 多候选路径跑通、`BenchmarkRunner` 可选 draft_model 注入（None 零变化；config mode="model_draft" 无注入 → 构造期 RuntimeError）；五模式默认路径逐位零回归 | T001 | passing | 2026-10-10 完成：config_manager 白名单 +model_draft（值域扩展）；BenchmarkRunner `draft_model` 参数 + 构造期 RuntimeError（先于 DraftGenerator 构造）；IT-511（ablation 1 层 tiny draft 端到端）+ IT-512（folded_generate append ∈ 末位 top-k）+ EX-804 + passthrough 正例 4 passed；AblationStudy/folded_generate 签名零改动。**Red 态说明**：生产接线在 T001 后落地，无独立 Red；接线前 EX-804 将因 config 白名单 ValueError（非 RuntimeError）失败、IT-511/512 因 "Unsupported draft mode" 失败——依赖的生产能力均由 T001 Red 覆盖 |
| T004 | 文档收口：指南 M-2 第三层第一档 ✅（第十三部分回链）、README draft 模式列表、AGENTS #40 遗留注记更新、CHANGELOG AR006 节 | T001–T003 | passing | 2026-10-10 完成：指南 M-2 :213/:282 三层全达 + 第十三部分回链（含接受率数学口径与 fixture 结构教训）；README 局限性 #5/#8 更新（model_draft tier 1）；AGENTS #40 扩展（model_draft 语义 + 两层校验 + tier 2 遗留）；CHANGELOG AR006 节 |
| T005 | 全量回归 + 质量门：738 基线零回归、demo 基线（85.5% / 2.35e-03 / 93.75%）、mypy（AGENTS 标准命令）/pyflakes/100 列 | T001–T004 | passing | 2026-10-10 完成：全量 754 passed / 3 skipped / 3 deselected（738 基线 + 16 新增零回归，即 IT-513）；demo 基线精确一致（demo 走默认模式不受影响，即 BS-702）；mypy 65 files clean；pyflakes clean；改动文件 100 列 0 违例（BS-703） |

## 任务详细说明

### T001 `model_draft` 模式
- `draft_generator.py`：`_MODES` 加 `"model_draft"`；`__init__` 加
  `draft_model: Callable[[torch.Tensor], torch.Tensor] | None = None`；
  `generate` 分支：`mode == "model_draft"` → `draft_model is None` 时
  RuntimeError；`draft_logits = self.draft_model(parent.tokens)`，形状校验
  `(batch, seq_len, vocab_size)` → ValueError；flip 位置
  `_sample_from_topk(draft_logits[b, pos])`；append 位置
  `_sample_from_topk(draft_logits[b, seq_len-1])`；`parent_logits` 忽略。
- 既有 `suffix_append`/`logits_draft` 分支保持不动（mode 检查顺序注意：
  现有代码 `else: # suffix_append / logits_draft` 分支需改为三分支或保持
  兼容——`model_draft` 与 logits_draft 共享采样逻辑但 logits 来源不同）。

### T002 合成相关性 fixture
- tests 内 `_CausalTokenTarget`（4 层，`logits[:, i]` 只依赖
  `tokens[:, :i+1]`，seed 固定权重）+ `_CorrelatedDraft`（target 第 1 层
  权重 + 小扰动投影）；断言相关性存在（draft argmax 命中率 ∈ (0,1) 开区间）。
- 接受率测量 DraftGenerator 统一 top_k=2（D9：k=1 比较器恒 1.0、
  k=10 gap 上界 0.099 不可达）；推导见 design §2.3。
- fixture 规格显式：vocab_size=1000（理论值示例 1/V=0.001）、
  prompt_length=0（全序列 region）、seq_len=64、flip_ratio=0.25、
  batch=1；跨 device UT-503 leg 用 meta device tensor（CPU-only 可测）。

### T003 接线
- `actfold/utils/config_manager.py:110`：draft_mode 白名单加入
  `"model_draft"`（值域扩展，无新字段）。
- `benchmark_runner.py:80`：构造函数加可选 `draft_model` 参数（默认 None），
  注入 DraftGenerator；config mode="model_draft" 且 draft_model None →
  构造期 RuntimeError（AGENTS #10；两层校验时序见 srs 3.3）。
- `AblationStudy`/`folded_generate`：无签名改动，IT 只验证传参路径。

## 门控记录

### Requirements 门控（sdd-phase-gate）
- 第 1 轮：**FAIL**（1 必修 Important + 2 WARN）——D3：srs 3.3"无 config schema 改动"与 config_manager.py:110-114 draft_mode 白名单硬编码矛盾（config 构造期 ValueError 抢先 BenchmarkRunner RuntimeError，时序契约歧义）；W1：3.2 上界比较器歧义（argmax 直取恒 1.0 平凡）；W2：3.2"接近下界"未量化
- 修复：srs 3.3 白名单值域扩展 + 两层校验时序；3.2 比较器 = logits_draft 模式实测接受率 + 差值 ≥0.1/<0.05/≥16 位置量化；tasks.md T003 补 config_manager 子项
- 第 2 轮：**PASS**（3 项修复验证通过 + 7 维度复核无回归；1 Minor：T002 摘要措辞与 srs 比较器不一致——已顺手对齐）
- 审查代理：general（只读）

### Design 门控（sdd-phase-gate）
- 第 1 轮：**FAIL**（1 Critical + 1 Important + 4 Minor）——Critical：数值预期数学错误（top-k 均匀采样接受率 = P(argmax∈top-k)/k ≠ argmax 命中率；k=10 gap 上界 0.099<0.1 不可达、k=1 比较器恒 1.0，srs 两阈值联合不可满足且 fixture top_k 未指定）；Important：dtype/device 校验未落地（int logits 静默通过 topk、跨设备采样报难解错误）；Minor：≥16 flip 位未映射、§5 文件名错（test_eval.py 实）、UT/IT-513 编号、738 硬编码
- 修复：srs 3.2 重写验收标准（top_k=2 统一 + 完整推导）；design §2.2 三项校验（形状/is_floating_point/device）+ §2.3 理论值表 + D9 决策 + 风险旋钮修正；UT-503/506/507 扩展；§5 更正 test_eval.py；BS-701 相对基线
- 第 2 轮：**PASS**（数学复核通过：P(hit)/2−1/1000≥0.1 ⟺ P(hit)≥0.202；全维度无回归；2 nit：fixture V/prompt_length 未显式——已补 T002 规格）
- 审查代理：general（只读）

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-10 开发会话（T001–T005 全部完成）

- T001：`model_draft` 模式 + `draft_model` 参数 + `logits_source` 统一采样源 + `_sample_tokens_from_logits` 共用 helper + 三项校验（形状/int dtype/跨 device）；Red 9 failed / Green 28 passed；既有五模式逐位不变（`logits_draft` 仅把 `parent_logits` 引用换成同值 `logits_source`）。
- T002：接受率梯度 fixture（top_k=2，D9）；probe 先行锚定 P(hit)=0.906（gap 0.452 ≥0.1、≤0.5 上界）；**fixture 结构缺陷两处修复**：①无关 draft 去掉单位阵残差（cumsum+I 共享使随机权重也 0.4 相关，probe 实证）；②相关 draft bias 裸 tensor → Parameter（`.to(device)` 随行，修复 cuda 设备下 RuntimeError）；31 passed。
- T003：config_manager 白名单 +`model_draft`；BenchmarkRunner `draft_model` 注入 + 构造期 RuntimeError；IT-511（ablation 1 层 tiny draft 端到端 measure_folding）+ IT-512（folded_generate append ∈ 末位 top-k）+ EX-804 + passthrough 正例 4 passed；AblationStudy/folded_generate 签名零改动。
- T004：指南第十三部分 + M-2 三层全达勾选；README 局限性 #5/#8；AGENTS #40 扩展；CHANGELOG AR006 节。
- T005：全量 754 passed / 3 skipped / 3 deselected（738+16 零回归）；demo 85.5% / 2.35e-03 / 93.75% 精确一致；mypy/pyflakes/100 列全过。
- **review 修复（第 1 轮 Important）**：IT-512 插入点误切既有 `test_branch_ids_bounded_length` 尾部 3 行 `_next_branch_id` 断言——已还原至原函数尾部并从 IT-512 移除（test_folded_generation.py）；T002/T003 Red 态说明补记（Minor）。

### 开发后合规审查（sdd-task-review）

- **第 1 轮：FAIL**（1 Important + 1 Minor；S 5/5、C 3/3、D 1/4——功能/测试/质量门/零回归/零改动声明全部实证通过）
- Important（D4）：IT-512 插入点位于 `test_branch_ids_bounded_length` 函数体中间，尾部 `_next_branch_id` 断言被并入无关测试体——已修复还原（见进度记录）
- Minor（D4）：T002/T003 无独立 Red 态记录——已补记（等效 Red：首跑 3 failed 暴露 fixture 缺陷 / 接线前 EX-804+IT-511/512 必失败由 T001 Red 覆盖）
- 实证亮点：审查代理 diff 级复核确认 `logits_draft` 分支 `parent_logits`→`logits_source` 为等值替换、random/perturb/copy_flip 分支逐字节未动、RNG 消费顺序不变
- 审查代理：general（只读）
