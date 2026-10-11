# [AR008] 需求设计说明书 — generic.forward `output_hidden_states` 按需化

| 字段 | 内容 |
|------|------|
| AR 编号 | AR008 |
| AR 主题 | 真实模型前向内存峰值优化：`GenericDiffusionLLM.forward` 不再恒置 `output_hidden_states=True` |
| 关联 SR | 指南 6.2 API/架构设计债（"generic.forward 恒置 output_hidden_states=True：7B 模型 T=512 时数百 MB 峰值浪费"） |
| 日期 | 2026-10-11 |
| 状态 | Reviewed（req/design 门控 PASS，T001–T004 passing，review PASS） |

## 1. 背景与目标

`GenericDiffusionLLM.forward`（generic.py:129-135）对被包装的 HF/AutoModel
**每次前向**都传 `output_hidden_states=True`。该标志使底层模型收集**每一层**
的 hidden states 到一个元组（7B 模型 T=512 时约 32×1×512×4096×4B ≈ 256MB
fp32 峰值），而 ActFold 只在**极少数**包装模型（既不暴露 `logits` 也不暴露
`last_hidden_state`，只给 `hidden_states` 元组；generic.py:140-152 的第三分支）
才消费它。绝大多数 HF 模型（`logits` 或 `last_hidden_state` 必有其一）白白
承担这笔分配与保留。

目标：默认不收集；仅在输出确实缺失前两者时按需重跑一次并缓存决策，
使三类模型行为完全兼容、真实模型内存峰值显著下降。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `GenericDiffusionLLM.forward`（generic.py）：默认 `output_hidden_states=False`；
  当且仅当输出既无 `logits` 也无 `last_hidden_state` 且 `hidden_states` 元组
  非空时消费元组；两字段缺失且元组不可用时带 `True` 重跑一次并置实例级
  缓存标志（后续前向直接传 `True`，不重复双跑）；
- 三态输出模型（logits-only / last_hidden_state-only / hidden_states-only
  stub）行为逐位兼容的测试钉死；
- 内存峰值的 CUDA 代理实证（大 hidden 合成 stub，峰值差可观测）。

**Out of Scope（本 AR 不做的）：**
- 专用 adapter 的独立 forward——Fast-dLLM 等不经 generic 路径（全库 grep
  证实 `output_hidden_states` 恒置仅 generic.py:132 一处）。**注意**：
  `LLaDAModel` / `DreamModel` 继承 `GenericDiffusionLLM` 且未覆写 forward
  （llada.py:14 / dream.py:14），随本改动自动受益（二者均产 `logits`，
  属第一态），其 sampler 非折叠路径（diffusion_sampler.py:171 调
  `model.forward`）同样被实例标志方案覆盖——3.2 回归需点名该继承链；
- 真实 7B checkpoint 实测（本机无权重；按 RERUN_CHECKLIST 属遗留）；
- `DiffusionLLM` 基类接口变更（零签名改动）。

## 3. 功能需求

### 3.1 `output_hidden_states` 按需化

**描述：** `GenericDiffusionLLM.forward` 不再无条件要求底层模型收集全层
hidden states。

**触发条件：** 每次 `forward` 调用。

**期望行为：**
- 默认传 `output_hidden_states=False`（实例缓存标志为假时）；
- 输出有 `logits` → 直接返回（不变）；
- 输出有 `last_hidden_state` → `lm_head(last_hidden)`（不变）；
- 两者均缺 → 若缓存标志为假：置实例标志 `True` 后带
  `output_hidden_states=True` **重跑一次**同一 forward，从元组取
  `hidden_states[-1]`；若重跑后仍无可提取张量 → 沿用现行 RuntimeError
  消息；
- 实例标志置位后，后续 forward 直接传 `True`（每实例至多一次双跑）。

**异常处理：**
- 三态之外的输出（重跑后仍无 logits/last_hidden/hidden_states）→
  RuntimeError 消息与现行逐字一致（"Model output exposes neither
  'logits', 'last_hidden_state', nor 'hidden_states'..."）；
- 重跑沿用同一入参（tokens/attention_mask/kwargs 剥离 branch 键后的
  剩余部分）；eval 确定性下结果一致；train 模式下 dropout 双跑差异
  可接受（第一次输出本就不可用，重跑是唯一可用结果）。

**验收标准：**
- Given logits-only stub 模型，When forward 被调用 N 次，Then 每次底层
  调用的 `output_hidden_states` 均为 False（spy 断言）且返回值与改动前
  逐位一致；
- Given last_hidden_state-only stub，When forward，Then 返回
  `lm_head(last_hidden)` 逐位一致且 `output_hidden_states=False`；
- Given hidden_states-only stub，When 首次 forward，Then 触发恰好一次
  重跑（底层共 2 次调用：第一次 False、第二次 True）且返回值正确；
  When 再次 forward，Then 底层仅 1 次调用且直接传 True；
- Given 三态全无的 stub，When forward，Then RuntimeError 消息与现行
  逐字一致（且不无限重跑）；
- Given 带 `branch_id`/`parent_branch_id`/`step_idx` kwargs 的
  hidden_states-only stub，When 首次 forward，Then 两次底层调用的
  kwargs 均不含 branch 键且结果正确（剥离逻辑在重跑下保持）。

### 3.2 端到端零回归

**描述：** 既有 generic 模型路径（test_models.py 相关 + demo/eval 消费链）
行为不变。

**验收标准：**
- Given 既有全量测试套（761 passed 基线），When 全量回归，Then 零失败；
- Given demo.py，When 运行，Then 85.5% / 2.35e-03 / 93.75% 精确一致
  （demo 合成模型不经 generic 路径，作无干扰证据）。

### 3.3 内存峰值收益实证（代理）

**描述：** 用大 hidden 合成 stub 在 CUDA 上证明峰值下降。

**验收标准：**
- Given 24 层 × T=512 × hidden=4096 fp32 的 last_hidden_state-only stub
  （模型权重本身 + 每层输出已有分配），When 分别以改动前后逻辑跑一次
  forward（或以 False/True 对照），Then `torch.cuda.max_memory_allocated`
  峰值差 ≥ 150MB（理论全层元组 ≈ 24×512×4096×4B ≈ 192MiB；阈值留余量）；
  无 CUDA 环境时该用例 skip。

### 3.4 文档收口

**验收标准：**
- 指南 6.2 对应条目勾选 + 第十五部分回链；CHANGELOG AR008 节；
  AGENTS 无需新增条目（若 review 判定需要则补）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 内存 | 真实模型前向峰值 | hidden states 元组不再分配（logits/last_hidden 模型）；见 3.3 代理阈值 |
| 性能 | logits/last_hidden 模型 | 每前向省一次全层收集（峰值与带宽双赢，不设墙钟指标——CUPTI 限制） |
| 质量 | 静态检查 | mypy 标准命令 clean / pyflakes 零 / black 100 列 |

## 5. 约束与假设

**约束：**
- `DiffusionLLM` 基类与 `GenericDiffusionLLM` 公开签名零改动；
- 三态模型行为完全兼容（第三类模型仍能运行，仅首次多一次前向）；
- RuntimeError 消息逐字保留（防依赖消息文本的调用方/测试破坏）；
- AGENTS #10：不静默改变可观测行为——重跑发生在"输出本就不可用"的
  路径，属修复而非静默切换；实例标志是实现细节，不暴露为公共 API。

**假设：**
- 评测/生成均在 eval 模式（demo/eval 链路 `.eval()` 为既有约定）；
  train 模式双跑差异仅影响本就不可用的首跑输出；
- `return_dict=True` 下 HF `ModelOutput` 的字段访问语义（getattr None
  探测）与现行一致。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 三态输出 | HF/AutoModel 包装返回的三类形状：`logits` / `last_hidden_state` / 仅 `hidden_states` 元组（generic.py:136-152 注释所列） |
| 实例级缓存标志 | `GenericDiffusionLLM` 实例上的私有布尔，记录"该模型需要 `output_hidden_states=True`"，避免重复双跑 |
