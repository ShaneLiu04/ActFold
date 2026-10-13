# [AR016] srs — 多候选验证的批量 CUDA graph 捕获 + P1-7 墙钟实证基准

| 字段 | 内容 |
|------|------|
| AR 编号 | AR016 |
| 主题 | P1-7 后续档「graph 批量捕获」：`ManualFoldedForward` 批量分支调用（`branch_ids`）在 `use_cuda_graph=True` 下捕获/重放 [N, T] 静态批量前向；并以基准实验首次给出 P1-7 论点（"批量验证让 folded 真正快于 baseline"）的墙钟证据 |
| 日期 | 2026-10-13 |
| 状态 | draft（待 req 门控） |
| 前置 | AR015（批量分支契约 #41、`branch_ids` 三层贯穿、launch-count 代理）；AR002（`FoldedGraphRunner`、#37 graph 契约、退化矩阵先例、-47.1%/step 串行实测） |

## 1. 背景与动机

### 1.1 现状

- AR015 落地批量分支前向后，`ManualFoldedForward.forward` 的 graph 门为
  `elif self.use_cuda_graph and branch_id is not None`
 （architecture_utils.py:486）——**批量调用（branch_id=None）静默旁路
  CUDA graph**，每步仍逐层发射 kernel；指南第二十二部分明列
  「graph 批量捕获」为 P1-7 后续档。
- `FoldedGraphRunner`（cuda_graph.py）的 capture 体**本身 batch 无关**：
  `capture()` 按 `tokens.shape` 取 `batch, seq_len`（:206-209），静态缓冲
  按 `[L(+1), batch, seq, H]` 分配（:218-227——child_buf 为 [L,…]、
  parent_static 为 [L+1,…]，index 0 = embedding，见术语表），
  `_prefill_parent` 的
  `parent_static[l].copy_(ffn_out)` 在 torch 语义下**广播**
  [1,T,H]→[N,T,H]（:338/:345），`_static_forward` 全程 shape 泛型
  （:350-408）。AR002 的静态前向设计（fused gate+mask+count 单 kernel +
  `_padded_divergent_index` 固定容量 + 分层缓冲）天然支持 batch=N。
- 串行 graph 的重放后发布模板 `_publish_graph_child`
  （architecture_utils.py:671-709）：逐层 `child_buf[l].clone()` 入缓存 +
  layer 0 embedding + `mask_buf[l]` 记 profiler——**行切片化即得批量版**。
- P1-7 的核心论点「多分支墙钟 ~1/N；**让 folded 验证真正快于 baseline
  的关键**」（guide :180）至今**没有任何墙钟数字**：AR015 按 #39 纪律
  只交付了 launch-count 代理（L vs N·L），未引用未实测的墙钟。本机
  CUPTI 受限（#39）只阻塞 kernel 计数，**墙钟可用 perf_counter + 重复
  统计严格测量**——AR002 基准（`scripts/ar002_graph_bench.py`、唯一有效
  工件 `results/optimization/ar002_graph_bench.json`，per-step -47.1%）
  就是先例。

### 1.2 机会

1. **批量 graph 捕获**：同形（same-shape [N,T]）多候选验证负载
   （diffusion 验证形态：固定 T、固定 N、跨步同形）在批量 eager 路径上
   每步仍付 L 层 × 每层多 kernel 的发射开销；graph 重放把整栈折叠为
   1 次 replay + 1 次 prefill 拷贝 + 1 次预算读回。这是 AR002 串行
   graph（-47.1%/step）在批量维度的自然补全。
2. **P1-7 墙钟实证**：基准脚本测量四路径 × (N,T) 扫描——
   (a) baseline eager（N 次全量前向，spiffy 式非折叠对照）、
   (b) folded 串行、(c) folded 批量 eager（AR015）、
   (d) folded 批量 graph（本 AR）——首次以 TimingStats（mean/std/p50/n）
   + tokens 一致性校验给出墙钟对比工件，验证或证伪指南论点，
   并量化 graph 增量。

### 1.3 明确不在范围

- 变长候选批量、多 parent 批量、Triton 批量 kernel（其余后续档）；
- engine `verify_branches` 复数 API（另行立项）；
- `folded_generate` 生长循环接入批量 graph（T 每步 +1，形状恒变，
  重捕获无收益——#20 变长静默回退同款结论，srs 仅记录定性）；
- baseline KV cache（P2-4，方法学对照另行评估）。

## 2. 术语

| 术语 | 定义 |
|------|------|
| 批量 graph 捕获 | 以 [N, T] 候选堆叠为 capture 形状的一次 `FoldedGraphRunner.capture`；N 个分支共享单 parent（expand 契约 #41），parent 静态缓冲 [N,T,H]（`parent_static` 为 **[L+1, N, T, H]**，index 0 = embedding）由 [1,T,H] 广播预填 |
| 批量重放 | **重放前 parent 批量再校验**（宿主侧 shape 元数据检查，无 D2H 同步，req 门控 Major-1）→ copy 填充 tokens_static + parent 广播预填 → `graph.replay()` → 单次 `count_buf.tolist()` 预算校验 → 逐分支行切片发布（cache/profiler） |
| 同形多候选验证负载 | 固定 (N, T) 跨步不变的多候选验证（diffusion 验证形态）；区别于 folded_generate 每步 T+1 的生长负载 |

## 3. 功能需求

### 3.1 FR1 ManualFoldedForward 批量 graph 分发

**描述：** `forward(tokens, branch_id=None, branch_ids=[...], parent_branch_id=pid, ...)`
在 `use_cuda_graph=True` 且批量前置条件满足时，路由到新私有方法
`_graph_forward_batched_or_none`；返回 `[N, T, V]` logits。任何前置
不满足按 FR2 回退批量 eager 路径（AR015 实现逐位不动）。

**验收标准：**
- 批量 graph 与串行 graph **各自独立捕获**：新属性
  `graph_runner_batched`（初值 None，与 `graph_runner` 互不干扰——
  串行 [1,T] 与批量 [N,T] 形状不同，共用 runner 必然形状冲突）；
- 批量前置（全部满足才尝试捕获/重放）：
  ① `use_cuda_graph=True` 且 `branch_ids` 非 None 且 `branch_id is None`
  （互斥已在 :476-481 保证）；
  ② `parent_branch_id is not None`（无 parent 批量调用恒 eager——
  parent 前向 populate 缓存，语义同串行 :603-605）；
  ③ 无 scheduler、exact-type `SimilarityGate(metric="cosine")`、
  tokens 在 CUDA（同串行 :606-616 判定面）；
  ④ `len(branch_ids) == tokens.shape[0]`（与 eager 批量契约一致）；
  ⑤ `attention_mask is None` 或与捕获时同一对象（mask 烘焙进 graph，
  同对象判定同串行 :648-651）；
- 首次合格调用捕获（EX 序列同串行 :618-647）：parent 缓存完整且
  **seq 匹配 + parent 批量 == 1**（私有 `_parent_cache_complete_batched`：
  逐层 shape[:2] 检查为 `[1, tokens.shape[1]]`——expand 契约 #41 的
  graph 侧镜像；该检查**捕获时与每次重放前各执行一次**，见重放条目）；
  `FoldedGraphRunner` 构造参数与串行完全一致；
- 重放（每次重放前全查，宿主侧零 D2H 同步——req 门控 Major-1）：
  ① **parent 批量再校验**：`_parent_cache_complete_batched` 在**每次重放前**
  重新执行（不只捕获时）——捕获后 parent 项被覆写为 [N,T,H]（串行路径
  `mff(tokens[N,T], branch_id="p")` 合法可达）时，`copy_` 会静默广播
  成功使每分支对不同父行折叠、预算校验（计数自洽）拦不住——**必须**
  显式拒绝。**终点行为分子例**（req 复核 Major-R1）：**seq 不匹配**
  → 静默批量 eager 回退（eager 截断/前缀折叠，成功返回）；
  **parent 批量 != 1** → graph 层**静默拒绝**（零告警、不重捕获、
  `graph_runner_batched`/降级标志状态不受扰、graph 侧零 cache/profiler
  写入），控制权交回调用方后由 eager 路径按 #41 契约抛
  `RuntimeError`（folded_transformer.py:476-484；graph 层透明不吞
  不改——与"AR015 eager 逐位不动"自洽）；
  ② `runner.replay(tokens, parent_branch_id, branch_ids[0])`
  （branch_id 形参 informational，runner 不消费）；返回 logits_buf
  **必须 `.clone()` 后返回**（静态缓冲复用契约，同串行 :669）；
- 预算校验失败（`runner.budget_exceeded`）：丢弃重放输出、一次性
  UserWarning、批量 eager 重算（同串行 :657-667 语义，逐分支正确性
  优先）；
- **重放后逐分支发布 `_publish_graph_children`**：镜像
  `_publish_graph_child`（:671-709）——`embedding = embed_module(tokens)`
  一次计算，逐分支 `embedding[i:i+1]`；逐层
  `ffn_out = child_buf[l][i:i+1]`（**clone 后 put**——静态缓冲跨重放
  复用，行视图必须物化，同串行整缓冲 clone 的行粒度版）；
  `cache.put(branch_id=branch_ids[i], ...)`；逐分支逐层
  `GLOBAL_STABILITY_PROFILER.record(stable_mask=mask_buf[l][i:i+1], ...)`
  （profiler 存 GPU tensor 无同步，#P0-3；下游
  `_run_folded_forward_batched` 经 `get_profile` 读 mean_stable_ratio
  的既有链路零改动）；
- 逐分支发布的开销面 = L 层 × N 分支的行切片 put/record（宿主侧
  GPU 张量操作，无 D2H 同步）——与 eager 批量路径的逐分支
  store/profiler 完全同阶，不构成回归。

### 3.2 FR2 退化矩阵（批量 graph 专用）

**验收标准（全部沿串行先例 :572-580 的既定语义）：**
- **形状不匹配**（已捕获 [N,T]，来电 [N',T']≠[N,T]）：一次性
  UserWarning（含两个形状），不重捕获，批量 eager——串行 :590-602 同款
  （区别于 #20 变长静默：graph 路径先例是一次性告警，保持机制类内
  一致）；
- **scheduler / 非 cosine gate / 非 CUDA（含 CPU 输入）**：一次性告警
  （`_warn_graph_degraded_once` 复用；**语义精确化**：其背后是**单一
  布尔** `_graph_degraded_warned`（:416/:522-532），覆盖整个
  scheduler/gate/CUDA/capture-fail 家族——任何原因的首个降级消耗掉
  该家族的唯一一次告警（如串行侧 scheduler 告警后，批量侧非 CUDA
  不再重复告警），跨原因抑制是串行先例既定行为，本 AR 接受不改），
  批量 eager；
- **attention_mask 对象不匹配**（捕获时非 None，重放传入不同对象）：
  静默批量 eager，不告警不重捕获（串行 :648-651 同款语义——mask
  烘焙进 graph，不同对象不得静默复用）；
- **parent 缓存不完整 / seq 不匹配（捕获时或任意重放前）**：静默 eager
  回退（eager 整批重算或截断折叠，成功返回），不禁用未来捕获（瞬态，
  同串行 :620-623）；**parent 批量 != 1（捕获时或任意重放前）**：
  graph 层静默拒绝（状态零扰动），终点由 eager 路径按 #41 契约抛
  `RuntimeError`——该子例**不是**"静默 eager 成功"家族成员（req 复核
  Major-R1）；重放前的 parent 批量再校验是正确性防线（req 门控
  Major-1），不是可选优化；
- **capture 失败**：`_graph_batched_capture_failed = True`（独立于串行
  `_graph_capture_failed`——一侧失败不得禁用另一侧；capture 失败
  消耗共享降级告警如上），一次性告警，永不重试（EX-002 语义）；
- **预算超出**：一次性告警 + 丢弃重算（FR1）；
- 串行 graph 路径（`branch_id` 单数）与批量 graph 路径互不干扰：
  各自 runner、各自形状/预算/捕获失败标志；
  `_warn_graph_degraded_once` 家族标志按上述单一布尔语义共享。

### 3.3 FR3 FoldedGraphRunner 复用契约

**验收标准：**
- `cuda_graph.py` 生产代码**目标零改动**；若实现中发现必须修改
  （如 `_prefill_parent` 广播语义在特定 cache 实现下不成立），修改
  必须：a) 不改变串行路径行为（既有 test_cuda_graph 全绿零改动）；
  b) 在 design/tasks 中记录改动点与理由；
- 广播预填契约钉住：`parent_static[l].copy_(ffn_out)` 中
  ffn_out.shape == [1, T, H]、parent_static[l].shape == [N, T, H] 时
  torch `copy_` 语义为广播复制（值沿 batch 维复制 N 份）——UT 以
  逐元素对拍钉住（重放后 cache 内各分支 ffn_out 与串行单分支
  graph 的对应值 allclose）；
- 容量语义：`C = ceil(ratio * N * T)`（:208-209 按 capture 形状计算，
  天然批量）；`D_l = N*T - count_l <= C` 校验（:311）不变；
- capture 体内 `fused_gate_mask_count` 在 `N*T < _FUSED_GATE_MIN_TOKENS
  (1024)` 时的内部回退路径必须 capture 安全（AR002 既有行为，本 AR
  以批量捕获在小形状 UT 通过为实证）。

### 3.4 FR4 对拍与观测钉（UT-1501~，`tests/test_batched_graph_forward.py`）

**验收标准（CUDA 用例 skipif 保护；容差沿 test_cuda_graph.py:396 先例
allclose(1e-4, rtol=1e-4)——padded 容量 gather 与 eager 精确 gather 的
既有机制差，另有 1e-6 级 for embedding 行）：**
- **UT-1501 批量捕获-重放 vs 批量 eager 对拍**：同 parent 等长 N=4
  候选 [1,T]，`use_cuda_graph=True`：首次批量调用捕获，第二次同形
  批量调用重放；重放输出 vs AR015 eager 批量（fresh 无 graph 实例）
  logits allclose(1e-4)+逐行 argmax 相等；逐分支 cache ffn_out
  allclose(1e-4)；逐分支 profiler 统计量（mean_stable_ratio）
  allclose(1e-6)（fused gate 与 plain gate 的浮点路径差异在阈值邻域
  可能翻转个别 token 的 stable 判定——统计量容差 1e-6 覆盖非邻域
  场景，若实测翻转则在 design 勘误并放宽该子项）；
- **UT-1502 逐分支发布完备性**：重放后逐分支 cache 含全部 L 层
  ffn_out + layer 0 embedding；embedding 行 == `embed_module(tokens)`
  对应行（allclose 1e-6）；被拒分支语义不在本 UT（graph 层不做
  prune，调用方职责同 AR015）；
- **UT-1503 串行 graph 共存零干扰**：同实例先批量捕获重放，再串行
  捕获重放（各自 `graph_runner`/`graph_runner_batched` 非同一对象、
  形状各异）；随后再次批量重放仍正确——两 runner 独立性钉住；
- **UT-1504 形状不匹配回退**：捕获 [4,T] 后来电 [2,T]（合法批量但
  形状不同）→ 一次性 UserWarning（断言形状信息在消息中）+ 结果 ==
  批量 eager；第二次不匹配零新告警；
- **UT-1505 预算超出回退**：`graph_capacity_ratio` 极小（如 0.02）
  + 中段大差异候选（divergent 超 C）→ 一次性 UserWarning + 丢弃重算，
  结果 == 批量 eager（正确性优先）；
- **UT-1506 前置条件矩阵**：scheduler 挂载 / 非 cosine gate / CPU
  tokens / parent 缓存空 / **attention_mask 对象不匹配**（捕获时非
  None，重放传不同对象 → 静默 eager 零告警且 `graph_runner_batched`
  状态不受扰）→ 均不捕获或不重放且结果 == 批量 eager（CPU 情形一次性
  告警，与降级家族同路）；**parent 批量 2** → 未捕获 +
  `pytest.raises(RuntimeError)`（#41 eager 契约，graph 层不吞）；
- **UT-1507 capture 失败永久禁用（EX-002 批量版）**：monkeypatch
  `FoldedGraphRunner.capture` 为**一次性**失败（`side_effect` 单元素
  或失败后立即还原 patch——类级持续 patch 会连带炸掉后续串行捕获）→
  一次性告警 + `_graph_batched_capture_failed` 置位 + **patch 还原后**
  串行 `graph_runner` 路径不受影响（后续串行合格调用仍可捕获）；
- **UT-1508 parent 批量 != 1 静默拒绝（捕获时 + 重放前双查）**：
  (a) parent 以 batch 2 缓存后首次批量调用 → 零告警 + 未捕获 +
  `pytest.raises(RuntimeError)`（graph 层静默拒绝，eager 按 #41 契约
  报错——expand 契约 #41 的 graph 侧镜像：graph 层是优化层不吞不改
  eager 契约）；
  (b) **捕获成功后** parent 被串行路径覆写为 [N,T,H]（req 门控
  Major-1 场景），再次同形批量调用 → 零告警 + 重放被拒 +
  `pytest.raises(RuntimeError)` + **cache/profiler 未被污染**（raise
  先于任何 store——folded_transformer.py:476-484）+ runner 状态不受
  扰（可再次合格调用成功重放）；**观测面（方向反转）**：mutation
  删除重放前再校验 → 调用**不再抛错**且 cache 被污染（每分支对不同
  父行折叠，cache 对拍必红）；
- **UT-1509 mutation spot-check（纪律沿用 AR015）**：①删
  `graph_runner_batched` 分发（批量恒 eager）→ UT-1501 重放断言
  （第二次调用仍捕获路径）翻红——观测面为"第二次同形调用走重放"
  的探针（如 `runner.replay` spy 或 `_publish_graph_children` 调用
  计数）；②发布跳过 clone → 跨重放污染（第二次重放后第一次的
  cache 值变化）红；③发布跳过某分支 → cache 完备性红；④预算
  校验删除 → UT-1505 红。全验证后 SHA256 逐字节还原。

### 3.5 FR5 墙钟基准实验（`scripts/ar016_batched_bench.py`）

**描述：** 独立基准脚本（沿 `scripts/ar002_graph_bench.py` 先例：命令行
参数、种子、环境记录、JSON 工件），测量同形多候选验证负载的四路径
墙钟，产出 `results/optimization/ar016_batched_bench.json`。

**验收标准：**
- **负载**：合成因果模型（可 detect 架构、CUDA）；parent [1,T] 缓存
  一次；每步 N 个同 parent 等长候选（中段分歧 + 独立 suffix，同
  AR015 场景族）验证 S 步（跨步同形——graph 可重放）；
- **四路径**：(a) baseline eager：`raw_model(tokens)` N 次全量前向
  （非折叠对照，spiffy 式）；(b) folded 串行 eager；(c) folded 批量
  eager（AR015）；(d) folded 批量 graph（本 AR）；
- **扫描**：N ∈ {2, 4, 8} × T ∈ {128, 512}（批量 graph 路径
  N*T ≥ 256，fused gate 在 512 时 N=2 恰过 1024 阈值——覆盖两侧）；
  每配置 warmup ≥3 + 重复 ≥10 次取 TimingStats（mean/std/p50/n）；
  **每重复内四路径轮转交错**（round-robin：rep1: a→b→c→d, rep2:
  a→b→c→d, ...）——RERUN_CHECKLIST §4.2 同日同机交错纪律，GPU 状态
  漂移均摊到四路径；`torch.cuda.synchronize()` 包裹计时边界（墙钟
  纪律）；计时用 `time.perf_counter`（**不是** `time_forward` 的
  CUDA-event 计时——后者排除宿主侧 prefill/`tolist` 读回/逐分支
  发布成本，而批量 graph 的净收益恰含这些项）；统计量复用
  `TimingStats`/`stats_from_samples`（scripts/algo_experiments.py:196/
  :216）；
- **正确性随行记录**：每配置记录四路径 tokens argmax 一致性
  （allclose 容差内标记 `logits_parity: true/false`）与 mean
  stable_ratio——**质量不因加速路径漂移**是工件的一等公民字段；
- **环境与限制记录**（#39 纪律）：GPU 名称/驱动/WDDM、无锁频声明、
  **运行日期**（RERUN_CHECKLIST §6）、CUPTI 探测结果（预期 null，
  记录 `"kernel_count": null` + 注记——墙钟为 perf_counter 实测，
  kernel 计数不可测不编造）、交错执行方式声明（round-robin 满足
  §4.2，无偏离需记录）；
- **假设与结果分离**：脚本头部 docstring 声明假设 H1（批量 eager
  显著快于 folded 串行，方向 ~N×）/ H2（批量 graph 进一步降低每步
  墙钟）/ H3（四路径 logits 一致）；JSON 只存实测值与假设判定
  （`supported/refuted/inconclusive`），**不预设结论**；
- **工件合规**：`results/optimization/ar016_batched_bench.json` 含
  RERUN_CHECKLIST 要求的环境/命令/方法学字段（沿 ar002_graph_bench
  .json 结构扩展），`INVALIDATED.md` 不适用于新工件；
- **运行预算**：全扫描单机 < 10 分钟（脚本 `--quick` 子集开关供
  CI 冒烟，默认全量）。

### 3.6 FR6 文档收口

**验收标准：**
- 指南 :180 P1-7 勘误注追加「graph 批量捕获 ✅ AR016」+ 第二十三部分
  （表格式 AR 记录 + **实测墙钟表**——首次以数字回应"让 folded 验证
  真正快于 baseline"，无论方向）；
- CHANGELOG AR016 节置 Unreleased 最前（Added/Design/实验结果小节）；
- AGENTS #41 扩注批量 graph 契约（独立 runner、逐分支行切片发布
  clone 语义、契约外输入回退矩阵、与串行 graph 的隔离性）；
- AR015 遗留清单中的「graph 批量捕获」标记收口。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 静态检查 | mypy clean（65 文件基线）/ pyflakes 零 / 100 列 |
| 质量 | 全量回归 | 829 passed 基线 + 新增零失败；**既有测试零改动** |
| 稳定 | demo 锚点 | 三指标精确一致（demo 不经批量路径） |
| 稳定 | 测试确定性 | CUDA 对拍仅 allclose+argmax；无墙钟断言进 pytest（基准是脚本非测试）；UT 运行时 <30s |
| 兼容 | 公开 API | `cuda_graph.py` 目标零改动；串行 graph 行为零变化（test_cuda_graph 零改动全绿）；AR015 批量 eager 逐位不动 |
| 性能 | 发布开销 | 逐分支发布 = L×N 行切片 put/record，无 D2H 同步，与 eager 批量同阶 |
| 性能 | 基准诚实性 | 墙钟含 warmup/重复/std；环境限制明示；假设判定基于实测而非预设 |

## 5. 约束与假设

**约束：**
- 生产改动限：`architecture_utils.py`（分发+发布）、`cuda_graph.py`
  （目标零改动，允许有据最小修改）、新脚本
  `scripts/ar016_batched_bench.py`（+ 文档）；
- 串行 graph 路径（#37 契约）零语义变化；capture 体零 host 同步/
  零缓存写纪律不破坏；
- 基准工件数字必须可复现（种子 + 命令行 + 环境字段齐全）。

**假设：**
- torch `Tensor.copy_` 广播语义（src [1,T,H] → dst [N,T,H]）跨
  dtype/device 稳定——UT-1501 隐式验证；
- `fused_gate_mask_count` 的 <1024 tokens 内部回退为 capture 安全的
  纯 GPU 路径（AR002 串行小形状捕获可行为既有实证；批量同构）；
- 本机 WDDM + 无锁频下 perf_counter 墙钟经 warmup + ≥10 重复 + p50
  足以分辨四路径差异（AR002 -47.1% 先例同环境可分辨）；
- fused gate（graph 内）与 plain gate（eager 批量）的 mask 在非阈值
  邻域一致；邻域翻转若实测发生，profiler 统计量容差按实测记录勘误
  （沿 AR015 UT-1403 gemv/gemm 勘误先例）。

## 6. 风险

| 风险 | 缓解 |
|------|------|
| **重放期 parent 覆写污染**（捕获后 parent 被覆写为 [N,T,H]：copy_ 静默广播成功 → 每分支对不同父行折叠，预算校验拦不住——req 门控 Major-1） | 每次重放前宿主侧 parent 批量/seq 再校验（零 D2H 同步）：批量 != 1 → graph 层静默拒绝，终点为 eager #41 契约 RuntimeError（req 复核 Major-R1）；UT-1508(b) 观测面：mutation 删除再校验 → 不抛错且 cache 污染，对拍必红 |
| graph 池内存压力：**双 runner 共存**（串行 [1,T] 与批量 [N,T] 图同时驻留）+ [N,T,H]×(L+1) 静态缓冲随 N 线性 | 基准最大 N=8、T=512、H=32（合成）远低于 16GB；真实模型场景属后续档评估，srs 记录不阻塞 |
| 批量 capture 在 fused gate 回退路径下的可捕获性 | UT-1501 小形状（N*T<1024）+ 基准大形状（≥1024）双侧实证 |
| 墙钟噪声（WDDM 调度、P8 降频） | ≥10 重复 + p50 + std 记录 + 每重复四路径轮转交错（RERUN_CHECKLIST §4.2）+ 环境字段；若四路径差异小于 std 则假设判定 `inconclusive`——诚实记录优于编造 |
| publish 行切片 clone 遗漏导致跨重放污染 | UT-1509 mutation ② 显式钉住 |
