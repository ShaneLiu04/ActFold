# [AR002] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR002 |
| AR 主题 | M4 单 kernel 融合 + CUDA graph 验证循环 + B12 完整重构 |
| 关联 SR | 无（源自 `docs/OPTIMIZATION_GUIDE.md` 里程碑 M4/P2-2/P2-3 与 AR001 遗留 follow-ups） |
| 日期 | 2026-10-09 |
| 状态 | Draft |

## 1. 背景与目标

AR001（M1+M2+M3+M5）已清零正确性 bug、将每层阻塞同步降至 ≤1 次（三路判定）、
内存 pass 降至 ≤6、cache 显存 −50%。但项目核心矛盾仍在（README 局限性 #2）：
**部分稳定 folded 前向仍慢于 no-folding baseline**。OPTIMIZATION_GUIDE 判定根因
不在算法而在工程：

1. split 层的 `nonzero` 数据依赖 gather 每层追加 1 次 host 同步（mixed path 实际 2 次），
   且其动态 shape 是 CUDA graph 捕获的硬阻断（P2-2）；
2. branch 上下文经 `contextvars` thread-local 传递（`folding_context.py`），属 host 侧
   动态状态，捕获/replay 语义不可控（P2-3 前置）；
3. 慢路径 kernel 序列碎片化：gate cosine ≈5–6 个 kernel + mask sum + merge，
   launch 开销在小 batch/短序列场景占比高（M4a）；
4. diffusion 验证阶段是"同 shape 反复前向"的典型可捕获负载，却完全跑 eager（M4b，
   潜在 2–5×，P2-3）；
5. `FoldedModel` 原地替换层（B12）：state_dict key 漂移、raw model 不可直接调用，
   且替换式包装与 graph 捕获互相牵制（D6：B12 完整重构与 M4 绑定，故留本 AR）。

本 AR 目标：完成 M4（gate+gather+merge 单 kernel 融合、CUDA graph 捕获验证循环）
与 B12 完整重构（不替换式 `ManualFoldedForward` 常态化），并清掉 AR002 计划内的
2 项 Minor 跟进（DraftGenerator 全局 seed 污染、LMEvalAdapter 生成长度按任务一刀切）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `actfold/core/split_layer.py`：预算化 padded gather，根除 `nonzero` 同步（P2-2）
- `actfold/core/folding_context.py` / `folded_transformer.py` / `model_wrapper.py` /
  `models/architecture_utils.py`：branch 上下文显式 kwargs 化，thread-local 降级为
  legacy 兜底（P2-3 前置）
- `actfold/core/fused_ops.py`（或新模块）：gate+mask+merge 融合 Triton kernel + PyTorch
  fallback（M4a）
- `actfold/models/architecture_utils.py`：`ManualFoldedForward` 升级为完整不替换式
  folded 前向并常态化（B12）；适配 `FastDLLMAdapter` / `folded_generation` /
  `AblationStudy` 的接入路径
- 新增 CUDA graph 捕获/回放器（M4b，CUDA-only，opt-in），集成进 diffusion 验证循环
- `actfold/speculative/draft_generator.py`：`generate(seed=)` RNG 隔离
- `actfold/eval/lm_eval_adapter.py`：per-task `max_new_tokens` 默认表
- 配套测试、CHANGELOG、AGENTS.md、OPTIMIZATION_GUIDE、README（局限性 #2 状态声明）

**Out of Scope（本 AR 不做的）：**
- `BranchManager` 变量长度折叠（P3，后续 AR）
- 全模型 `torch.compile`（本 AR 以手动 graph 捕获为主；compile 不作交付项）
- 7B/8B 真实 checkpoint 的绝对性能数字与论文对齐复现（需 ≥24GB 目标机，
  按 `docs/RERUN_CHECKLIST.md` 在目标机执行）
- `FoldedModel` 的删除（保留为 legacy API，文档标注 deprecated 指向
  `ManualFoldedForward`；行为不回归）

## 3. 功能需求

### 3.1 预算化 padded split（P2-2，根除 nonzero 同步）

**描述：** `SplitFoldedTransformerLayer` 的 divergent 行 gather 改为固定容量
padded 形式，消除每层数据依赖 shape 的 host 同步。

**触发条件：** folded 慢路径（mixed stable/divergent）且 split 启用、token 数 ≥
`min_split_tokens`。

**期望行为：**
- 容量预算 `C = ceil(budget_ratio × N)`（`budget_ratio` 可配置，默认值设计阶段定）；
- padded gather（索引 clamp/填充构造，无 `nonzero`）取 divergent 行进固定 `C` 行
  buffer，FFN 计算 `C` 行，scatter 回写时按 mask 过滤，只写真实 divergent 行；
- divergent 数（由既有 `stable_count` 同步免费导出，不新增同步）超过 `C` 时，
  该层走全量 recompute（正确性优先）；
- eager 路径与现行实现 bit-exact 对拍。

**异常处理：** 超预算 → 全量 recompute；split 检测失败/异常 → 现行 fallback
（`RuntimeWarning` + `_recompute_all`）保留。

**验收标准：**
- Given mixed path 且 split 生效，When 以 monkeypatch 统计 host 同步
  （`cudaStreamSynchronize`/`.item()`/`.cpu()` 等代理），Then 每层同步次数 ≤1
  （现行 2）；
- Given 任意 stable/divergent 分布（含全 divergent、恰好等于预算、超预算 1 行），
  Then 输出与重构前 bit-exact。

### 3.2 branch 上下文显式 kwargs 化（P2-3 前置）

**描述：** 可捕获折叠路径的 branch 标识全部由显式参数传递，热路径不再依赖
`contextvars` thread-local。

**触发条件：** `ManualFoldedForward`（3.4）逐层调用 `FoldedTransformerLayer`。

**期望行为：**
- `ManualFoldedForward` 显式逐层传 `branch_id` / `parent_branch_id` / `step_idx`；
- `FoldedTransformerLayer.forward` 显式 kwargs 优先（现状保留），thread-local
  读取仅作为 legacy `FoldedModel` 路径兜底；
- 折叠语义与现行一致（gate、缓存、profiler 行为不变）。

**异常处理：** 两处均无 branch 标识 → 现行 `ValueError` 保留。

**验收标准：**
- Given 通过 `ManualFoldedForward` 的前向，When monkeypatch `FOLDING_CONTEXT` 使其
  `get()` 恒抛 `LookupError`/返回哨兵，Then 折叠仍正确激活且输出 bit-exact；
- Given legacy `FoldedModel` 路径（thread-local），Then 行为不回归。

### 3.3 gate+mask+merge 单 kernel 融合（M4a）

**描述：** 慢路径的相似度计算、阈值判定与 parent/child 合并融合为单个
Triton kernel，配套 PyTorch fallback。

**触发条件：** folded 慢路径（mixed），shape/设备满足融合阈值常量（沿用 D3
风格暴露可调常量），CUDA 且 Triton 可用。

**期望行为：**
- 融合 kernel 一次读 `hidden`/`h_parent`/parent 缓存/child 输出，产出合并后
  输出与 stable 计数（供三路判定与 profiler），数学定义与现行
  `SimilarityGate` + `merge_stable_divergent` 一致（cosine、dtype 感知 eps、fp32
  中间累加）；
- 不满足阈值 / CPU / Triton 缺失或编译失败 → 现行非融合路径（`UserWarning`
  仅编译失败时发一次）；
- profiler 统计（GPU 累加）在融合路径继续可用（kernel 写计数 buffer 或等价机制）。

**异常处理：** kernel 编译失败 → 运行时回退 PyTorch 路径（AGENTS #9）。

**验收标准：**
- Given 满足阈值的 CUDA 输入，When 经融合路径执行 folded 前向，Then 输出与现行
  gate+merge 链 bit-exact；
- Given 同输入，When 以 torch profiler 统计慢路径 CUDA kernel launch 计数，
  Then 较现行链（gate cosine ≈5–6 kernel + mask sum + merge）减少 ≥3 次；
- Given CPU / 无 Triton / 小 shape，When 执行同一前向，Then 走 fallback 且结果
  bit-exact。

### 3.4 `ManualFoldedForward` 常态化（B12 完整重构）

**描述：** 不替换式 folded 前从成为一等接入方式：base model 零改动、
state_dict key 零漂移、raw model 始终可直接调用。

**触发条件：** 任何经 `ManualFoldedForward` 的前向。

**期望行为：**
- 显式执行 embedding → 逐层 folded forward（显式 branch kwargs）→ LM head，
  支持 `FastDLLMAdapter` / `folded_generation` / `AblationStudy` 现有调用面
  （不破坏公开 API）；
- base model 的模块树、参数、`state_dict()` 键在包装前后完全一致；
- 与 `FoldedModel` 路径输出 bit-exact（同配置对拍）；
- `FoldedModel` 保留为 legacy（deprecation 指引写入 docstring 与 CHANGELOG）。

**异常处理：** 未发现 layer 栈/head → 现行显式报错保留（AGENTS #23）。

**验收标准：**
- Given 任一已支持架构，When 构造 `ManualFoldedForward` 并前向，Then base
  model `state_dict()` 键集合与包装前完全相等，且 raw model 直接前向始终可用；
- Given 同 cache/gate/scheduler 配置，Then `ManualFoldedForward` 与 `FoldedModel`
  输出 bit-exact；
- Given `FastDLLMAdapter` / `folded_generate` / `AblationStudy` 集成测试，Then
  全部通过（不回归）。

### 3.5 CUDA graph 捕获验证循环（M4b）

**描述：** 新增捕获/回放器：对固定 shape 的 diffusion 验证 folded 前向做
CUDA graph capture，replay 复用；opt-in，非 CUDA 优雅降级。

**触发条件：** `ActFoldConfig.use_cuda_graph=True` 且 CUDA 设备且走
`ManualFoldedForward`（可捕获路径）。

**期望行为：**
- 静态输入/输出 buffer，首步 warmup + capture，后续步 replay（copy-in/copy-out）；
- 捕获路径内无数据依赖 host 分支（依赖 3.1/3.2/3.3/3.4 的静态化成果）；
- replay 后一次性校验各层 divergent 预算（异步计数 readback，每步 ≤1 次）；
  超预算 → 丢弃该次 replay 结果、eager 重算并 `UserWarning`（正确性优先）；
- 捕获失败 / 非 CUDA / shape 变化 → `UserWarning`（一次性）+ eager fallback。

**异常处理：** 任何 capture/replay 异常 → eager 重算，不向上抛出破坏生成循环。

**验收标准：**
- Given 本机 CUDA 固定 shape 验证循环，When 执行 warmup capture 并连续 N 步
  replay，Then capture/replay 成功，且 replay 输出与 eager 数值一致
  （`allclose` atol=1e-4；token 级/argmax 级完全一致。修订说明：graph 静态
  前向使用固定容量 padded gather（D 行 + clamp 尾部填充），GEMM 归约顺序与
  eager 精确 gather 不同，浮点结果存在 ~1e-6 量级差异，逐位一致结构性不可达
  ——review 阶段修订，数值容差 + 语义一致即满足正确性目标）；
- Given 预算超限注入场景，When replay 校验触发，Then 走 eager 重算路径且最终
  输出正确；
- Given CPU-only 环境，When 启用 `use_cuda_graph`，Then 自动降级 eager 且测试
  通过；
- Given 验证循环 N 步，When 以 monkeypatch 统计 host 同步，Then replay 路径每步
  host 同步 ≤1 次（校验 readback，无逐层同步），而 eager 路径为每层 ≥1 次。

### 3.6 `DraftGenerator.generate(seed=)` RNG 隔离（AR001 Minor 跟进）

**描述：** `seed` 不再通过 `torch.manual_seed` 污染全局 RNG。

**期望行为：** 以局部 `torch.Generator`（或全局 RNG snapshot/restore）实现可复现
采样；`_counter` 重置语义保留（同 seed 同分支 ID）。

**验收标准：**
- Given `generate(seed=s)`，When 生成分支，Then 调用前后全局 RNG 状态
  （`get_rng_state`）不变，且两次同 seed 调用产出一致分支；
- Given 无 seed 调用，When 生成分支，Then 行为不回归。

### 3.7 LMEvalAdapter per-task 生成长度表（AR001 Minor 跟进）

**描述：** 以 `_TASK_MAX_NEW_TOKENS` per-task 默认表替代全局 256 一刀切。

**期望行为：** 代码类任务（humaneval 等）默认 512，其余沿用 256；
显式 `max_new_tokens` 参数优先级最高；`_TASK_METRIC_KEYS` 等现状不变。

**验收标准：**
- Given 未显式传长度的 humaneval 任务，When 构造 adapter，Then 生成长度 512；
- Given 显式 override，When 构造 adapter，Then 以 override 为准。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | bit-exact | 3.1/3.3/3.4 eager 路径与现行路径对拍逐位一致；3.5 replay vs eager 数值一致（allclose atol=1e-4 + token 级一致，见 §3.5 修订说明） |
| 性能（代理） | host 同步/层 | folded mixed path ≤1（split 生效时），CUDA graph 验证循环每步 ≤1 次校验 readback |
| 性能（代理） | kernel launch | 慢路径满足阈值时 launch 计数较现行链（gate cosine ≈5–6 kernel + mask sum + merge）减少 ≥3 次（torch profiler 断言；融合后预期 1 kernel + 计数 buffer 写） |
| 可移植性 | CPU / 无 Triton / 非 CUDA | 全部功能可用（自动 fallback），测试覆盖 fallback 路径 |
| 质量门 | 测试/静态检查 | 全量测试绿（AR001 基线 536 passed 不回归）；`mypy --strict` 零错误；black/isort CI |
| 回归 | demo 对拍基线 | FLOPs 85.5% / MSE 2.35e-03 / stable 93.75% 不回归 |
| 资源 | 本机可验证 | 全部验收在 16GB 本机完成（合成模型 + 本机 CUDA graph 实测）；无需大显存 |

## 5. 约束与假设

**约束：**
- AGENTS.md 全部约定适用（#9 Triton fallback、#10 无 mock 结果、#19/#23 包装语义
  演进需同步修订条目等）；
- CUDA graph 仅 CUDA；CPU/Triton 缺失路径必须保留并测试；
- breaking change 允许（公开 API 尽量兼容，破坏处在 CHANGELOG 登记）；
- 本机 Quadro RTX 5000（16GB）降频态：绝对墙钟不作为验收，用可测量代理
  （同步计数、kernel 计数、bit-exact、launch 计数）；
- 不引入 mock 数据作真实结果（AGENTS #10）。

**假设：**
- diffusion 验证阶段 stable 比例高（demo 93.75%），预算化 split（3.1）的容量预算
  在真实负载下极少触发全量 recompute fallback；
- 目标机（换机后）按 `docs/RERUN_CHECKLIST.md` 重跑时 graph 模式的绝对收益可
  单独测量（本 AR 只交付代理证据）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 预算化 padded gather | 固定容量 buffer + clamp/填充索引的行 gather，无数据依赖 shape，无 `nonzero` |
| 可捕获路径 | 无 host 数据分支、无 thread-local 依赖、shape 静态的前向路径（Manual + padded split + 融合 kernel） |
| replay 校验 | graph 回放后对固定计数 buffer 的一次性 readback 检查，超预算则丢弃结果 eager 重算 |
| M4 | OPTIMIZATION_GUIDE 里程碑：单 kernel 融合 + CUDA graph，战略目标为 folded 前向反超 baseline（绝对数字留目标机） |
