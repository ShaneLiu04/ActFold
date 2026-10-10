# [AR004] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR004 |
| AR 主题 | 真投机解码接受语义（P2-5）：logit-based 接受率 + log-prob 分数 + EMA[r] |
| 关联 SR | 无（源自 `docs/OPTIMIZATION_GUIDE.md` P2-5，战略级优先级：高） |
| 日期 | 2026-10-10 |
| 状态 | Draft |

## 1. 背景与目标

OPTIMIZATION_GUIDE P2-5 自认：当前验证引擎**只测激活相似度，无基于 logits 的
接受率验证**（draft 分布 vs target 分布）——论文叙事（ActFold 加速投机解码的
验证阶段）成立的前提是真接受语义。具体缺陷：

- `verification_engine.py:122` 的 `score = logits.float().mean().item()` 是
  无意义分数（原始 logits 均值，非概率语义），作为 `actfold_score` 写入
  branch metadata；
- `spiffy_baseline.py:77` 的 `baseline_score` 是同类占位；
- 接受判定 `accepted = stable_ratio >= acceptance_threshold` 只反映折叠
  复用率，不反映 draft token 是否被 target 模型认可；
- 无接受率（EMA[r] 式）与 log-prob 分数的报告能力。

本 AR 目标：为验证引擎实现**target-argmax 接受语义**（diffusion LM 并行验证
约定：target 前向的 logits[:, i] 预测位置 i 的 token；draft token 被接受当且
仅当它等于该位置 target argmax），报告 per-call 接受率与跨调用 EMA 接受率，
以 mean log-prob 分数替换全部占位分数，并使接受判定消费真语义。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `actfold/speculative/verification_engine.py`：per-position 接受 mask、
  draft 区域接受率、mean log-prob 分数、EMA[r] 追踪、`VerificationResult`
  新字段、接受判定切换、`:122` 占位替换
- `actfold/speculative/spiffy_baseline.py`：`baseline_score` 占位替换为
  mean log-prob
- `actfold/speculative/acceptance_policy.py`：新增 target-match 接受策略
  （按接受率选候选）
- `actfold/speculative/folded_generation.py`：`FoldedGenerationResult`
  接受率报告（跨步均值）
- 配套测试与文档（AGENTS/README/CHANGELOG/OPTIMIZATION_GUIDE P2-5 勾选）

**Out of Scope（本 AR 不做的）：**
- 真实 draft 模型接入（Medusa/Eagle 类，P3-3）
- 随机拒绝采样接受（temperature/stochastic acceptance；本 AR 只交付
  greedy/argmax 确定性接受，与 Fast-dLLM/SPIFFY 并行解码契约一致）
- 多祖先复用（P3-2）、`Branch.hidden_states` 惰性引用（AR003 遗留子项）
- 真实 checkpoint 的接受率实验（RERUN_CHECKLIST 目标机流程，本 AR 交付
  机制 + 合成模型证据）

## 3. 功能需求

### 3.1 target-argmax 接受率（验证引擎核心语义）

**描述：** `verify_branch` 用 child 前向的 logits 计算 per-position 接受
mask，报告 draft 区域上的接受率。

**触发条件：** `verify_branch(parent_branch, child_branch)` 被调用（child
前向产出 logits `[B, T_c, V]`）。

**期望行为：**
- 接受 mask（同位置预测约定，文档显式声明）：
  `accept_mask[:, i] = (child_tokens[:, i] == logits[:, i].argmax(-1))`；
- draft 区域定义：child 与 parent 前缀（`[:, :T_p]`）的差异位
  （`child != parent`）+ 变长后缀位（`T_p..T_c`，含 AR003 append-only 场景）；
  child == parent 的纯复用位不计入（此前缀已被先前轮次验证）；
- `acceptance_rate = accept_mask 在 draft 区域上的均值`；draft 区域为空
  （child 与 parent 逐位相同且等长）时定义为 `1.0`（无新 draft，语义上
  全部接受）；
- logits 与 tokens 形状不匹配（`T_c` 不一致或 vocab 维缺失）→ `ValueError`
  （不静默跳过验证）。

**异常处理：** 形状不匹配 → `ValueError`；draft 区域为空 → rate = 1.0。

**验收标准：**
- Given 受控 adapter（可指定 argmax 的 logits），When draft token 部分等于
  target argmax，Then `acceptance_rate` 精确等于匹配比例（含 0.0 与 1.0
  边界）；
- Given child 与 parent 逐位相同（等长），Then `acceptance_rate == 1.0`；
- Given 变长 child（append-only，AR003 场景），Then 仅后缀位计入 draft
  区域，前缀复用位不影响接受率；
- Given logits 形状与 tokens 不匹配，Then `ValueError`。

### 3.2 mean log-prob 分数替换占位

**描述：** 以 draft 区域上的 mean log-prob（target 分布对 draft token 的
对数概率）替换两处 `logits.float().mean().item()` 占位。

**触发条件：** `verify_branch`（`actfold_score`）与 `SpiffyBaseline.verify`
（`baseline_score`）。

**期望行为：**
- `mean_log_prob = log_softmax(logits).gather(-1, child_tokens) 在 draft
  区域上的均值`（数值稳定实现，fp32 累计）；
- `child_branch.metadata["actfold_score"]` 语义替换为 mean_log_prob（不再
  是 raw-logit 均值；文档化语义变更）；
- `SpiffyBaseline.verify` 的 `baseline_score` 同样替换为 mean log-prob
  （全部位置，无 parent 对比的独立基线）。

**异常处理：** 无（复用 3.1 的形状校验）。

**验收标准：**
- Given 受控 logits，When 计算 `actfold_score`，Then 等于手算
  `log_softmax` gather 值（与 `torch.log_softmax` 参考逐位一致）；
- Given `SpiffyBaseline.verify`，Then `baseline_score` 为 mean log-prob
  （受控 logits 下与手算一致）；
- Given 全量测试，Then 依赖 `actfold_score` 既有断言（序数比较）不回归。

### 3.3 EMA[r] 接受率追踪

**描述：** 引擎跨 `verify_branch` 调用维护接受率的指数移动平均。

**触发条件：** 引擎构造（`ema_alpha` 参数）与每次 `verify_branch`。

**期望行为：**
- `ema = ema_alpha * acceptance_rate + (1 - ema_alpha) * ema`（首次调用
  直接初始化为当次 rate）；
- `ema_alpha ∈ (0, 1]`，默认值 design 阶段定（非法值 `ValueError`）；
- `VerificationResult` 新增字段：`acceptance_rate: float = 0.0`、
  `mean_log_prob: float = 0.0`、`ema_acceptance_rate: float = 0.0`
  （全部带默认值，dataclass 向后兼容既有构造点）；
- engine 暴露当前 EMA（属性或方法，benchmark/ablation 可读）。

**异常处理：** `ema_alpha` 非法 → `ValueError`。

**验收标准：**
- Given 首次 verify（rate=r1），Then EMA == r1；
- Given 连续 verify（r1, r2），Then EMA == alpha*r2 + (1-alpha)*r1（手算
  一致，含多次迭代）；
- Given `ema_alpha` 越界，Then `ValueError`。

### 3.4 接受判定消费真语义

**描述：** `accepted` 判定从 stable_ratio 切换到 acceptance_rate。

**触发条件：** `verify_branch` 的判定行（现行 `stable_ratio >=
acceptance_threshold`）。

**期望行为：**
- `accepted = acceptance_rate >= acceptance_threshold`；
- `acceptance_threshold` 默认 `0.0`（全接受）不变——默认行为零变化；
- `stable_ratio` 照常计算与报告（折叠复用指标，职责不变）；
- 语义变更文档化（AGENTS/CHANGELOG）：threshold>0 的调用方语义从"复用率
  门槛"变为"接受率门槛"。

**异常处理：** 无（threshold 校验既有）。

**验收标准：**
- Given 默认 threshold=0.0，Then 判定行为与基线一致（全部接受路径零回归）；
- Given threshold=0.8 + 受控 rate=0.5，Then `accepted is False`（且
  rejected 分支 cache 清理行为不变）；
- Given threshold=0.8 + 受控 rate=1.0，Then `accepted is True`。

### 3.5 target-match 接受策略与 folded_generate 接受率报告

**描述：** 新增按接受率选候选的策略；`folded_generate` 报告跨步接受率。

**触发条件：** `folded_generate(..., acceptance_policy=...)` 与结果聚合。

**期望行为：**
- 新增 `TargetMatchAcceptancePolicy`：为每个候选计算 appended token 的
  接受（child 最后位 token == child logits 同位置 argmax），按接受率选
  最佳（并列时取首个）；
- 既有 `GreedyAcceptancePolicy`/`ThresholdAcceptancePolicy` 保留不变
  （默认策略仍为 Greedy，零回归）；
- 每步将 `acceptance_rate` 写入 accepted 节点 metadata；
  `FoldedGenerationResult` 新增 `acceptance_rate: float = 0.0`（跨步
  均值，向后兼容默认）。

**异常处理：** 候选无 logits（policy 内部）→ 跳过该候选（既有行为模式）。

**验收标准：**
- Given 受控模型 + `TargetMatchAcceptancePolicy` + 多候选（接受率可分辨），
  Then 选出接受率最高的候选；
- Given `folded_generate`（默认参数），Then 既有行为/token 输出零回归，
  `FoldedGenerationResult.acceptance_rate` 报告跨步均值；
- Given 受控模型（appended token 可控是否匹配 argmax），Then 单步
  acceptance_rate 为 0.0 或 1.0 精确可控。

### 3.6 文档与配置收口

**描述：** 语义变更文档化 + P2-5 勾选。

**期望行为：**
- AGENTS.md：新增条目（接受语义约定：同位置预测、draft 区域定义、
  `actfold_score`/`baseline_score` 新语义、threshold 语义切换、EMA）；
- CHANGELOG：AR004 章节；README（如涉及用户可见行为）；OPTIMIZATION_GUIDE
  P2-5 勾选 ✅ + 回链。

**验收标准：**
- Given 文档评审，Then 上列表目齐全且与实现一致。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | 零回归 | 全量 652 passed 基线不回归（含 3 skip/3 deselect 口径）；demo 基线 85.5% / 2.35e-03 / 93.75% 不变 |
| 兼容性 | API 向后兼容 | `VerificationResult`/`FoldedGenerationResult` 新字段带默认值；`actfold_score`/`baseline_score` metadata 键名不变（仅语义升级并文档化）；默认 threshold/默认 policy 行为不变 |
| 数值 | log-prob 稳定性 | fp32 计算，大 vocab 下无 inf/nan（log_softmax 数值稳定路径） |
| 性能 | 验证开销 | 接受 mask/log-prob 计算为向量化 GPU op，无 per-position Python 循环、无额外前向 |
| 可移植性 | CPU/CUDA | 全功能 CPU 可验证（合成模型） |
| 质量门 | 静态检查 | mypy（AGENTS 标准命令）零错误；pyflakes clean；100 列；Google docstring |

## 5. 约束与假设

**约束：**
- 同位置预测约定（logits[:, i] 预测位置 i 的 token）为本 AR 的显式契约
  （与 Fast-dLLM/SPIFFY 并行验证、合成 TinyModel 的 token-wise head 一致）；
  文档显式声明，不做按模型类型分发（分发留给真实 checkpoint 接入时）；
- 接受判定语义切换必须保持默认行为零变化（threshold=0.0 全接受）；
- `actfold_score` 键名不变（消费方 test_integration.py:169 为序数比较，
  语义升级安全；仍需文档化）；
- AGENTS.md 全部既有约定适用（#10 无 mock 结果——接受率必须来自真实
  前向 logits，非合成数字；#34 eval 契约）。

**假设：**
- `DiffusionLLMAdapter.forward` 返回 `[B, T, V]` logits 且遵循同位置约定
  （合成模型实证；真实 diffusion LM 并行验证契约同理）；
- EMA 跨 verify 调用的会话粒度足够（per-engine 实例；benchmark 逐 prompt
  新建 engine 或复用均可解释，语义上不依赖具体粒度）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| target-argmax 接受 | draft token 被接受 ⟺ token == target 前向 logits 同位置 argmax（确定性 greedy 接受，Fast-dLLM 并行验证契约） |
| draft 区域 | child 与 parent 的差异位 + append-only 后缀位；纯复用位（前缀逐位相同）不计入 |
| EMA[r] | 跨 verify 调用的接受率指数移动平均（论文报告口径） |
| mean log-prob | log_softmax(target_logits) 在 draft token 上的 gather 均值（target 分布对 draft 的对数概率分数） |
