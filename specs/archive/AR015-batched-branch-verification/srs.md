# [AR015] 需求设计说明书 — folded_generate 多候选批量分支前向（P1-7 第一档）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR015 |
| AR 主题 | folded_generate 验证阶段的多候选串行前向（folded_generation.py:139-143 逐候选 `_run_folded_forward`）批量化：同 parent 等长候选拼成 [N, T] batch 一次 folded forward（P1-7 处方主体），core 层新增批量分支语义（`branch_ids` 复数 kwarg），逐行/逐分支与串行路径对拍 |
| 关联 SR | 指南 P1-7（:180，"让 folded 验证真正快于 baseline 的关键"，中-高）；指南 #20 append-only prefix folding（AR003，本 AR 的 varlen 前缀路径即其批量化）；AGENTS #19（ManualFoldedForward 为支持路径）/ #37（CUDA graph 契约——批量分支必须旁路）/ #30（gather_select D3 阈值——批量路径跳过）；AR004（acceptance 语义——`folded_generate` 逐候选 logits 消费方） |
| 日期 | 2026-10-13 |
| 状态 | Final（req 门控首轮 FAIL：Major-1 demo 锚点定性错误——demo 不经 folded_generate，NFR 理由改写为锚定 FR1/FR2 串行层路径；Major-2 FR1 缺"parent id 非空但缓存 miss"分支——生成循环第一步必然触发（root 缓存空），已补验收项 + UT-1407 用例；Minor-1"全仓唯一"限定为折叠验证循环（spiffy_baseline 为故意 baseline 对照）/Minor-2 不变量改为"同 parent+等长"（flip_ratio 翻转由 gate 兜底）/Minor-3 branch_id 保持必填类型放宽/Minor-4 logits 消费面三处补全/Minor-5 gather_select 跳过理由改为 fetch_flat 恒 None 优雅回退——同批修复，待复核）+ design 轮勘误回写：UT-1401 profiler 对拍面改统计量（stable_sum/num_tokens/mean_stable_ratio——profiler 不保留 mask，design 门控 m-3）、UT-1406 探针断言改 `kwargs.get("branch_ids") is None`（层循环无条件传参，design 门控 m-6） |

## 1. 背景与目标

P1-7（guide :180）指出：多分支验证串行——`verification_engine.py`
+ `folded_generation.py:114-117`——N 个 child 应拼成 batch 一次
forward，"多分支墙钟 ~1/N（GPU 未饱和时）；让 folded 验证真正
快于 baseline 的关键"。AR015 选题勘察（本轮，证据在案）逐点确认：

**串行点与消费方**：
- 全仓唯一的**折叠验证**多分支串行前向循环在
  `folded_generation.py:139-143`：
  `for node in candidates: tree.add(node); _run_folded_forward(...)`——
  N-1 个被拒候选的完整前向（embedding→L 层→head）纯浪费
  （`spiffy_baseline.py:77-86` 另有一个多分支循环，但那是**故意的
  非折叠 baseline 对照**，不属 P1-7 语义范围）；
- `verification_engine.py` 是 per-branch API（`verify_branch`），无
  多分支循环（guide :180 点名它的部分属清单腐化——engine 侧无需
  改动）；samplers 经 `_next_folding_branch` 链式单链（AGENTS #26），
  非多分支；**P1-7 的实质 = folded_generate 循环**（:114-117 行号
  陈旧，现 :139-143）；

**批量化前置不变量成立**（`_make_candidates` :198-257）：
- draft 模式（:210-234）：`draft_generator.generate(...,
  max_new_tokens=1)` → 每个 draft 长度恒为 parent 长度 + 1
  （suffix_append/logits_draft/model_draft 的 `flip_ratio` 可能
  重采样 suffix 区内位置；random/perturb 整列随机——仅长度不变）。
  批量真正依赖的不变量是**同 parent id + 等长**，两者在所有
  draft 模式下成立；前缀位置的稳定性由 gate 语义兜底（AR003
  varlen 前缀路径照常工作，翻转位在 gate 处判 divergent）；
- greedy 模式（:235-255）：单候选 N=1，无批量化空间（天然走串行）；
- 生成循环 parent tokens 批量恒为 1（`input_ids [1, prompt_len]` 起，
  逐 token append）→ 父缓存激活 [1, T, H] 沿 batch 维 expand 即可
  服务 [N, T] 子 batch；

**今日批量输入的行为**（勘察确认）：若直接把 [N, T] 子 batch 喂给
现行折叠层，:185-190 批量不匹配检查使 `h_parent=None` → 全重算
（正确但零折叠收益），且 `_store_activations`（:558-579）把整个
[N, T, H] 存进单一 branch_id——多分支语义需 core 层显式支持
（P1-7 原文"core 的 cache/gate 需支持 batch 分支语义"）。

**目标**：
1. core 层批量分支语义：`FoldedTransformerLayer` 接受
   `branch_ids: Sequence[str]`（N 行 ↔ N 分支、共享单 parent），
   批量路径完成 gate/merge/store/profiler 的逐分支语义；
2. 入口与分发：`ManualFoldedForward.forward` 接受 `branch_ids`
   （批量调用旁路 CUDA graph）；`folded_generate` 在前置条件满足时
   单次批量前向替代 N 次串行，否则**原样串行回退**；
3. 逐位对拍：批量输出第 i 行 == 串行第 i 个候选（CPU bit-exact；
   CUDA batched 归约序 allclose），cache 逐分支、profiler 逐分支
   同构对拍；层数调用次数 L（批量）vs N·L（串行）作 ~1/N 的
   可观测代理（AGENTS #39 纪律：不 assert 墙钟）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `actfold/core/folded_transformer.py`：`_ACTFOLD_KWARGS` 增
  `"branch_ids"`（:21，防泄漏进 original layer）；`forward` 增
  `branch_ids: Sequence[str] | None = None`，非 None 时委托专用
  `_forward_batched`（FR1）；**串行 forward 代码路径逐位不动**；
- `actfold/models/architecture_utils.py`：
  `ManualFoldedForward.forward` 增 `branch_ids` 形参，批量调用
  跳过 `_graph_forward_or_none`（FR2）；
- `actfold/speculative/folded_generation.py`：新增
  `_run_folded_forward_batched` 助手 + `folded_generate` 循环处
  批量/串行分发（FR3）；
- 新测试文件 `tests/test_batched_branch_forward.py`：
  UT-1401~UT-1408（FR4）；
- 文档：指南 :180 P1-7 勾选 + 勘误注（行号 :114-117→:139-143、
  engine 侧无多分支循环、批量路径守护条件）；第二十二部分回链；
  CHANGELOG AR015 节；AGENTS 新增批量分支契约 pitfall（FR5）。

**Out of Scope（本 AR 不做的）：**
- 变长候选批量（不同 T 的候选 pad/pack）；多 parent 混合批量
  （候选来自不同 parent）；parent batch > 1 的批量扩展——三者均
  回退串行，留待后续档；
- 批量路径的 Triton kernel 化（fused gate / Triton merge 对
  stride-0 expand parent 的适配）——批量路径固定用 plain gate +
  PyTorch merge（对拍优先，FR1 记录该取舍）；
- `FoldedModel`（废弃路径，AGENTS #19）批量支持；适配器直调
  `model.forward` 路径（:281-289）批量支持；
- `verification_engine.py` 改动（无多分支循环，见 §1 勘误）；
- CUDA graph 批量分支捕获（graph 契约 #37 明确同形静态 workload，
  批量分支静默 eager，与 varlen 先例 #20 一致）；
- 墙钟/加速比基准测试（AGENTS #39：本机 CUPTI 受限，launch-count
  代理已足够钉住机制；性能数字留待 GPU 基准 AR）。

## 3. 功能需求

### 3.1 折叠层批量分支语义（FR1）

**描述：** `FoldedTransformerLayer.forward` 新增
`branch_ids: Sequence[str] | None = None`；非 None 时不进入现行
串行路径，委托 `_forward_batched(hidden_states, attention_mask,
branch_ids, parent_branch_id, step_idx, **kwargs)` 实现 N 行 ↔
N 分支语义。

**验收标准：**
- **前置契约（编程错误 → 硬错误）**：`len(branch_ids) ==
  hidden_states.shape[0]` 否则 `ValueError`；
  `parent_branch_id is not None` 且其 gate 层缓存激活**存在但**
  批量 != 1 → `RuntimeError`（expand 契约 violation；
  `folded_generate` 层面已过滤，此处为防线）；
- **有 parent id 但缓存 miss**（生成循环第一步必然触发——root
  分支缓存为空，draft 模式不对 parent 做前向）：整批
  `_recompute_all` + 逐分支存（与串行 cache-miss 语义
  :181-182/:206-209 同构），不 expand、不 RuntimeError——批量
  路径必须覆盖此分支；
- **无 parent（`parent_branch_id=None`）**：整批 `_recompute_all`
  + 逐分支 `_store_activations`（行切片），不 expand——与串行
  no-parent 语义（:139-143）逐分支同构；
- **scheduler**：`should_fold` 为 False 的层整批重算 + 逐分支存
  （与 :146-149 同构）；动态 tau 逻辑复用；
- **gate**：父 gate 激活 [1, T_p, H] →
  `expand(N, -1, -1)`（stride-0 视图，零拷贝）后走 **plain
  `self.gate(...)` 路径**（不用 fused gate kernel——stride-0
  parent 未在 kernel 内验证，对拍优先；串行路径的 fused 分发
  :225-241 逐位不动）；等长（`prefix_len == seq_len`）与 varlen
  前缀（:244-266，前缀 gate + 全 False 尾）两分支语义与串行
  同构，掩码为 [N, T]；
- **三向拆分（批粒度）**：单次 `stable_mask.sum()` 读回（批粒度
  一次同步，与串行 F4a 同型）——全稳（count==numel）：父 FFN
  expand 后逐分支存（含 AR007 截断同构），不重算；全不稳：
  整批 `_recompute_all` 逐分支存；混合：整批走 slow path
  （`_recompute_merged` + merge）。**逐行快路径不复制**（生成
  循环后缀恒 divergent，串行同样走 slow path；混合批内全稳行经
  merge 掩码全 True 取父值 == 串行快路径结果，逐位等价）；
- **merge**：跳过 `gather_select`（:412-430——`fetch_flat` 按
  parent 存储批量（1）计算行数，对 [N, T] 子批量恒返回 `None`、
  必走回退（vectorized_cache 优雅降级而非错误行），故批量路径
  直接跳过；等长与 varlen 均跳过），父 FFN expand 后调
  `merge_stable_divergent` 的 **PyTorch 实现路径**（不进 Triton
  分发；varlen 时前缀 cat 后缀对齐 :434-439 同构）；
- **store**：`h_out[i:i+1]`（dim0 切片保持 contiguity）逐分支
  `cache.put`，layer 0 附 `embedding` 行切片（:571-574 同构）；
- **profiler**：逐分支 `GLOBAL_STABILITY_PROFILER.record`，
  `stable_mask[i:i+1]` 行切片、`parent_branch_id` 单值
  （:269-277 同构）——串行对拍面（profile per branch 相等）由此
  保证；
- **输出**：`_pack_output` 同串行（tuple 层 re-wrap 契约 #13 不变）；
- `_recompute_all` 的 kwargs 过滤（:539-541）不得把 `branch_ids`
  转发进 original layer（`_ACTFOLD_KWARGS` 增补）。

### 3.2 ManualFoldedForward 批量入口（FR2）

**验收标准：**
- `forward(tokens, branch_id, parent_branch_id=None,
  attention_mask=None, step_idx=0, branch_ids=None)`——
  `branch_id` 保持必填、类型放宽为 `str | None`；`branch_ids`
  非 None 时要求 `branch_id is None`（否则 `ValueError`——单复数
  互斥，批量调用显式传 None）；
- 批量调用**跳过 `_graph_forward_or_none`**（静默 eager，与 #20
  varlen 静默回退先例一致；graph 契约 #37 的同形静态 workload
  不含批量分支）；embedding 一次、`_wrapped_layers` 逐层传
  `branch_ids`、final_norm + head 对 [N, T, H] 整体作用，返回
  [N, T, V]；
- 串行调用（`branch_ids=None`）签名与行为**逐位不变**（含
  graph 路径）；
- `SplitFoldedTransformerLayer`（`_recompute_merged` 覆盖，#28）
  对 [N, T] 掩码天然泛化（逐 token gather），批量路径直接复用，
  parity 测试须含 `split_layers=True` 变体。

### 3.3 folded_generate 批量分发与串行回退（FR3）

**验收标准：**
- 新助手 `_run_folded_forward_batched(model, nodes, folded_model,
  step_idx)`：逐 node `profiler.reset_branch`（串行 :270 同构）→
  `torch.cat([n.tokens ...], dim=0)` 拼 [N, T] →
  `folded_model(stacked, branch_id=None, branch_ids=[...],
  parent_branch_id=pid, step_idx=...)` → `node.logits = logits[i:i+1]`（dim0 视图切片，
  不 clone——消费方 `:153` assert、`:155` 全位置 argmax
  （`target_argmax_accept_mask`）、`:245` 末位 argmax 均只读，
  dim0 视图切片安全）→ 逐 node 读 profile 写
  `metadata["stable_ratio"]`（:293-295 同构）；
- 分发条件（`folded_generate` :139 处，全部满足才批量，否则
  **现行串行循环原样**）：
  ① `folded_model` 为 `ManualFoldedForward` 实例（`FoldedModel`/
  None/adapter 直调 → 串行）；
  ② `len(candidates) > 1`；
  ③ 全部候选 `parent_id` 相同且非 None；
  ④ 全部候选 tokens 形状相同、且 parent tokens 批量为 1；
- 任何路径下 `tree.add` / prune / `clear_branch`（:139-172）与
  acceptance 语义（AR004，:145-159）**零改动**——批量只是
  `_run_folded_forward` 的批量化替代，候选求值语义不变。

### 3.4 对拍与观测钉（FR4，UT-1401~1412；1409~1412 为 review 门控增补）

**验收标准（新文件 `tests/test_batched_branch_forward.py`；
合成因果模型 + ManualFoldedForward，参照既有测试基建；CPU 用例
bit-exact，CUDA 用例 allclose(rtol=1e-6, atol=1e-6) + argmax
token 相等——batched GEMM 归约序差异的既定容差）：**
- **UT-1401 CPU varlen 逐位对拍**（生成循环形态：父 [1, T_p]
  缓存齐、N=4 候选 [1, T_p+1]）：批量 vs 逐候选串行（fresh
  cache 各跑一遍）——逐行 logits `torch.equal`、逐分支 cache
  `ffn_out`（含 layer 0 `embedding`）`torch.equal`、逐分支
  profiler 统计量（stable_sum/num_tokens/mean_stable_ratio——
  profiler 不保留 mask，统计量相等即逐位同构的完整可物化对拍面，
  design 门控 m-3 勘误）相等；
- **UT-1402 CUDA 对拍**：同 UT-1401 场景（`device` fixture），
  allclose + token 相等 + cache allclose；
- **UT-1403 等长 + split 变体**：等长候选（与 parent 同 T、中段
  分歧）+ `split_layers=True` 的 `ManualFoldedForward`——覆盖
  `_forward_batched` 等长分支与 Split 层 [N,T] 掩码泛化，CPU
  **allclose(1e-6) + 逐行 argmax 相等**（T002 勘误：split 聚集
  GEMM 的 M 维 = 批量 D_total vs 串行 D_i，串行单 divergent 行时
  M=1 走 BLAS gemv 微内核而批量 M>1 走 gemm——1 ULP；mask/profiler
  统计仍精确相等）；
- **UT-1404 launch-count 代理**：monkeypatch 计数 wrapped layer
  调用——批量 = L（层数），串行 = N·L；**不 assert 墙钟**
  （AGENTS #39）；
- **UT-1405 folded_generate 端到端**：draft 模式
  `num_branches_per_step=3` 多步生成——批量路径 tokens /
  stable_ratio / acceptance_rate 与强制串行（monkeypatch 分发
  条件为恒 False）完全一致；cache 无被拒分支残留（prune +
  clear_branch 生效）；
- **UT-1406 串行回退矩阵**：N=1 / 候选不等长（构造变长 draft）/
  混合 parent / `folded_model=None`——四情形均走串行（层 spy 断言
  `kwargs.get("branch_ids") is None`——层循环无条件传参，串行侧
  断言其值为 None 而非缺席，design 门控 m-6 同款）且结果正确；
- **UT-1407 层级前置契约**：`len(branch_ids) != batch` →
  `ValueError`；`branch_id` 与 `branch_ids` 同时给 → `ValueError`；
  parent 缓存批量 2 → `RuntimeError`；`branch_ids` + 无 parent →
  整批重算逐分支存（对拍正确）；**`branch_ids` + parent id 给定
  但缓存空**（生成循环第一步形态：root 无激活）→ 整批重算 +
  逐分支存，与串行 cache-miss 逐位对拍（req 门控 Major-2）；
- **UT-1408 graph 旁路**：`use_cuda_graph=True` 的
  `ManualFoldedForward`，批量调用不触发 capture/replay（graph
  runner 状态零变化）且 eager 批量结果正确；同实例串行调用
  graph 路径不受扰；
- **UT-1409 scheduler 禁用层批量同构**（review 门控 Major-1
  增补）：`FoldingScheduler`（layer 1 入 `disabled_layers`，
  layer 2 为调度器默认禁用的末层）挂在批量路径——禁用层整批
  `_recompute_all` + 逐分支存（profile 只含 layer 0；禁用层
  cache 行跨分支互异证非父视图共享）、折叠层动态 tau 复用
  （`gate.tau == scheduler.get_tau(0, step)`）、与同 scheduler
  串行运行 CPU 逐位对拍；
- **UT-1410 全稳快路径**（review 门控 Major-2 增补）：4 个与
  parent 逐 token 相同的候选——全稳三分支触发：父 FFN expand 后
  逐分支存**不重算**（`_recompute_all` 逐层计数 spy 钉 0——真前缀
  子代重算值与父复用值逐位相同，值级不可区分，调用计数是唯一
  可观测面）、逐分支 profile ratio == 1.0、逐分支 cache FFN
  `torch.equal` 父存储、与串行快路径逐位对拍；
- **UT-1411 AR007 截断全稳**（review 门控 Major-2 增补）：父
  [1,8] → 4 个真前缀子代 [1,6]——gate 激活与父 FFN 均先截断再
  expand 存储（截断回退被 spy 钉死不触发）、cache FFN 等于父
  截断值、逐位对拍；
- **UT-1412 AR007 截断混合 merge**（review 门控 Major-2 增补）：
  父 [1,8] → 4 个 [1,6] 子代末 2 位互异替换——每层混合慢路径，
  `_merge_parent_child_batched` 对更长父 FFN 先截断再 merge
  （删截断 → shape 校验 RuntimeError）、profile ratio 严格落于
  (0,1)、逐位对拍；
- **行为钉子型 AR 纪律（Red 期 mutation spot-check）**：
  ①删父 expand（gate 输入 shape 错/掩码全 False）→ UT-1401 红；
  ②整批存单一 branch_id → cache 逐分支对拍红（UT-1401）；
  ③跳过逐分支 profiler record → profiler 对拍红（UT-1401）；
  ④删批量分发（永远串行）→ launch-count 红（UT-1404）；
  **review 门控增补 mutation（五变异全验证后 SHA256 逐字节
  还原）**：⑤禁用批量 scheduler 分支 → UT-1409 红；⑥删全稳
  expand → UT-1410/1411 红（layer 1 len(branch_ids) != batch
  ValueError）；⑦删快路径 FFN 截断 → UT-1411 红（回退重算，
  spy 计数非零）；⑧删 merge 内 AR007 截断 → UT-1412 红
  （RuntimeError）；⑨禁用批量动态 tau → UT-1409 红。

### 3.5 文档收口（FR5）

**验收标准：**
- 指南 :180 P1-7 勾选 + 勘误注：①行号 :114-117 → :139-143
  （陈旧）；②`verification_engine.py` 侧无多分支循环（清单腐化
  部分，engine 为 per-branch API）；③第一档守护条件（同 parent
  等长、parent batch==1、ManualFoldedForward、PyTorch merge）与
  后续档（变长/多 parent/Triton 批量/CUDA graph 批量）；
- 指南第二十二部分 AR015 回链；CHANGELOG AR015 节
  （Added/Performance 语义，Unreleased 最前）；
- AGENTS.md 新 pitfall（#41）：批量分支契约——`branch_ids` 复数
  kwarg、单复数互斥、expand 契约（parent batch==1）、批量路径
  plain gate + PyTorch merge + 跳 gather_select、graph 静默
  eager、串行路径逐位不动；
- 指南 :180 涉及的"多分支墙钟 ~1/N"表述加 launch-count 代理注
  （本机 CUPTI 受限，#39），不引用未实测墙钟数字。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 静态检查 | mypy 标准命令 clean（65 文件基线）/ pyflakes 零 / 100 列 |
| 质量 | 全量回归 | `pytest tests\ -q -m "not slow"`：817 passed 基线 + 新增零失败；**既有测试零改动** |
| 稳定 | demo 锚点 | `python -X utf8 demo.py` 三指标精确一致——demo 不经 `folded_generate`（suffix_append draft + 直调 `folded_model`/`verify_branch`），锚定的是 FR1/FR2 **串行折叠层路径逐位不动**；FR3 串行回退与批量分发由 UT-1405/UT-1406 钉住 |
| 稳定 | 测试确定性 | CPU 对拍逐位（torch.equal）；CUDA 容差仅 allclose(1e-6) + token 相等；无统计/墙钟断言 |
| 兼容 | 公开 API | 串行路径逐位不动；新 kwarg 全部默认 None；`FoldedModel`/engine/adapter 直调零改动 |
| 性能 | 批量开销 | 父激活 expand 为 stride-0 零拷贝；单次读回同步（批粒度，与串行 F4a 次数同阶）；批量路径无额外 kernel |
| 性能 | 用例耗时 | 新文件合计 <30s（CUDA 含合成模型前向） |

## 5. 约束与假设

**约束：**
- 生产改动限三个文件：folded_transformer.py、
  architecture_utils.py、folded_generation.py（外加文档）；
- `FoldedTransformerLayer.forward` 串行路径（branch_ids=None）
  与 `ManualFoldedForward.forward` 串行路径**逐位不动**——817
  既有测试是回归网；
- 生成循环的候选求值/acceptance/prune 语义（AR004）零改动；
- 批量路径不引入新的进程级全局状态。

**假设：**
- 生成循环不变量由 `_make_candidates` 保证（各 draft 模式恒
  **同 parent + 等长**（长度 = parent+1；suffix 区可能翻转，由
  gate 兜底）、parent 批量恒 1）；分发条件④作防线而非依赖；
- CUDA batched vs serial 的数值差仅来自 GEMM 归约序（fp32
  allclose 1e-6 足够；若实测超限，按证据收紧为 CPU-only 逐位
  + CUDA token 相等，design 期记录实测值）；**T002 扩展**：CPU
  逐位仅在非 split 路径成立——split 聚集 GEMM 的 M 维在串行
  D_i=1（gemv）与批量 D_total>1（gemm）间跨微内核边界产生 1 ULP
  （实测 2.4e-07），UT-1403 据此用 allclose(1e-6)+argmax；
- stride-0 expand 在 plain gate（F.cosine_similarity 广播）与
  PyTorch merge（torch.where 广播）下数值与 materialize 等价
  （PyTorch 语义保证，不另行验证）；
- profiler 逐分支 record 的额外开销（N 次小 record vs 1 次）与
  串行 N 次调用完全同阶——不构成回归。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 批量分支 | 一次 folded forward 携带 N 个分支语义：`branch_ids`（N 个 id）+ 共享单 `parent_branch_id`，输入 [N, T]、行 i ↔ branch_ids[i] |
| expand 契约 | 批量分支要求父分支缓存激活批量 == 1：[1, T, H] 沿 batch 维 stride-0 expand 服务 [N, T, H] 子 batch |
| 第一档 | P1-7 的最小完整切片：同 parent、等长、parent batch==1 的候选批量前向 + 串行回退；变长/多 parent/Triton 批量化为后续档 |
| launch-count 代理 | 以 wrapped layer 调用次数（L vs N·L）作为"~1/N 墙钟"的可观测机制钉——替代本机不可测的 CUPTI kernel 计数（AGENTS #39） |
| 逐位对拍 | 批量输出第 i 行与串行第 i 候选输出 `torch.equal`（CPU）；cache/profiler 逐分支同构比较 |
