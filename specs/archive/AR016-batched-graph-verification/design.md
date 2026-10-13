# AR016 设计文档：批量 CUDA graph 捕获与 P1-7 墙钟实证

状态：design 阶段（srs req 门控 PASS，三轮）
前置：AR015（批量分支 eager 契约，#41）、AR002（串行 CUDA graph，#37）、AR011（fused gate/mask/count）
srs：`specs/changes/AR016-batched-graph-verification/srs.md`

## 1. 设计总览

不改动 `actfold/core/cuda_graph.py`（runner 天然 batch 无关——AR002 静态前向
设计红利），只改 `actfold/models/architecture_utils.py`：在
`ManualFoldedForward` 上新增批量 graph 分发（镜像串行
`_graph_forward_or_none` 的退化矩阵结构）+ 批量发布（行切片化
`_publish_graph_child`）+ 基准脚本。eager 批量路径（AR015）逐位不动。

```
forward() 分发（:476-491 改造）
  ├─ branch_ids is not None（批量）
  │    ├─ use_cuda_graph → _graph_forward_batched_or_none(...)   [新]
  │    │      ├─ 捕获（首合格调用）或重放（含 parent 再校验）
  │    │      ├─ 任何拒绝 → None → 落入下方 eager 体（#41 契约原样生效）
  │    │      └─ 成功 → _publish_graph_children + logits.clone() [新]
  │    └─ 否则 → eager 体（AR015 逐位不动）
  └─ branch_id is not None（串行，:486-491 逐位不动）
```

## 2. 模块改动：`architecture_utils.py`

### 2.1 新属性（`__init__`，:414-419 之后镜像添加）

```python
self.graph_runner_batched: Any = None   # 批量专用 runner（串行 [1,T] 与批量 [N,T] 形状不同，共用必冲突）
self._graph_batched_mask: Any = None    # 捕获时钉住的 attention_mask 对象
self._graph_batched_shape_warned = False    # 形状不匹配一次性告警（批量独立）
self._graph_batched_budget_warned = False   # 预算超出一次性告警（批量独立）
self._graph_batched_capture_failed = False  # capture 失败永久禁用（EX-002，批量独立）
```

共享不改：`_graph_degraded_warned`（单一布尔覆盖 scheduler/gate/CUDA/
capture-fail 家族，跨原因抑制——串行先例既定，srs FR2 已精确化）。
串行侧四标志（:414-419）逐位不动。

### 2.2 `forward` 分发点改造（:476-491）

现 `elif self.use_cuda_graph and branch_id is not None:`（:486）只覆盖
串行。改为批量分支先行插入（互斥检查 :477-481 之后）：

```python
if branch_ids is not None:
    if branch_id is not None:
        raise ValueError(...)            # AR015 逐位不动
    if self.use_cuda_graph:
        graph_out = self._graph_forward_batched_or_none(
            tokens, branch_ids, parent_branch_id, attention_mask, step_idx
        )
        if graph_out is not None:
            return graph_out
elif self.use_cuda_graph and branch_id is not None:
    ...                                  # 串行块 :486-491 逐位不动
```

graph 拒绝时控制流自然落入下方 eager 体（:492-503）——**parent 批量 != 1
的终点 = eager 按 #41 抛 RuntimeError，graph 层零吞改**（req 复核
Major-R1 的实现机制：不在 graph 层 raise，也不在 graph 层"修复"）。

docstring：`branch_ids` 参数说明的"Batched calls always run eager — the
CUDA graph captures only the same-shape serial workload (AGENTS #37)"
句更新为批量 graph 分发描述（引用 #41 扩注——T005 文档任务一并改
AGENTS #37/#41 表述）。

### 2.3 `_graph_forward_batched_or_none`（新私有方法，镜像 :562-669）

签名：
```python
def _graph_forward_batched_or_none(
    self,
    tokens: Any,
    branch_ids: Sequence[str],
    parent_branch_id: str | None,
    attention_mask: Any,
    step_idx: int,
) -> Any:
```

控制流（顺序镜像串行，差异点显式标注）：

1. `if self._graph_batched_capture_failed: return None`（EX-002，无重试）；
2. `runner = self.graph_runner_batched`；`runner is not None` 且
   `tuple(tokens.shape) != tuple(runner.tokens_static.shape)` →
   `_graph_batched_shape_warned` 一次性 UserWarning（含两个形状）→
   `None`（不重捕获，串行 :590-602 同款）；
3. `parent_branch_id is None` → `None`（parent 前向 populate 缓存，恒
   eager——批量调用无 parent 属 `_batched_forward_supported` 范畴之外，
   eager 层已有语义）；
4. `self.scheduler is not None` / gate 非 exact-type cosine /
   `not tokens.is_cuda` → `_warn_graph_degraded_once(...)`（共享家族
   标志）→ `None`（串行 :606-616 同款同 reason 文案）；
5. `len(branch_ids) != tokens.shape[0]` → `None`（eager 层按 AR015 契约
   处理/报错，graph 层透明）；
6. `runner is None`（首次合格调用，捕获）：
   - `if not self._parent_cache_complete_batched(parent_branch_id, tokens):
     return None`（静默——捕获时 parent 批量 != 1 → 静默拒绝 → eager
     #41 RuntimeError；不完整/seq 不匹配 → eager 成功）；
   - `attention_mask is not None and not attention_mask.is_cuda` →
     `None`（串行 :624-625 同款）；
   - 构造 `FoldedGraphRunner`，**构造参数与串行 :628-638 逐字相同**
     （wrapped_layers/embed_fn/final_norm_fn/head_fn/cache/gate_tau/
     gate_eps/capacity_ratio/attention_mask_static）；
   - `try: new_runner.capture(tokens, branch_ids[0], parent_branch_id)
     except RuntimeError` → `_warn_graph_degraded_once(f"CUDA graph
     capture failed ({exc})")` + `_graph_batched_capture_failed = True` →
     `None`（消耗共享降级告警，串行 :641-643 同款）；
   - `self.graph_runner_batched = new_runner`；
     `self._graph_batched_mask = attention_mask`；
   - 捕获用 `branch_ids[0]`（capture 的 branch_id 形参
     informational——runner 只消费形状与 parent 缓存，不写该分支缓存；
     串行实现同语义）；
7. `elif attention_mask is not self._graph_batched_mask` → `None`
   （静默，mask 烘焙进 graph，串行 :648-651 同款）；
8. **重放前 parent 再校验（req 门控 Major-1，区别于串行的核心防线）**：
   `if not self._parent_cache_complete_batched(parent_branch_id, tokens):
   return None`——每次重放前重新执行（宿主侧 shape 元数据检查，零 D2H
   同步）。捕获后 parent 被覆写为 [N,T,H] 时此处拒绝：graph 侧零
   cache/profiler 写入，eager 抛 RuntimeError 先于任何 store
   （folded_transformer.py:476-484）→ 零污染。**覆写场景的 cache 变体
   事实（design 门控 Major-N1）**：`ChunkedActivationCache` 整 chunk
   替换语义（chunked_cache.py:133-135）下同 id 批量增大覆写静默成功
   ——Major-1 洞的真实生产面；`ActivationCache` 的 `put` 对该操作抛
   RuntimeError（:154，异常面而非错值面）——两种 cache 下步骤 8 均必要；
9. `logits = runner.replay(tokens, parent_branch_id, branch_ids[0])`；
   `logits is None` → `None`（cache miss 瞬态，静默）；
   `runner.budget_exceeded` → `_graph_batched_budget_warned` 一次性
   UserWarning → `None`（丢弃重放输出，eager 重算——正确性优先，
   串行 :657-667 同款）；
10. `self._publish_graph_children(runner, tokens, branch_ids,
    parent_branch_id, step_idx)`；`return logits.clone()`（静态缓冲复用
    契约）。

注意：串行 `replay` 内部会再次 prefill parent（`_prefill_parent`），批量
runner 的 prefill 是 `copy_` 广播 [1,T,H]→[N,T,H]（:338/:345 已验证）；
步骤 8 的再校验保证进入 prefill 的 parent 恒为 batch 1，广播路径唯一。

### 2.4 `_parent_cache_complete_batched`（新私有方法，镜像 :534-560）

与串行 `_parent_cache_complete` 唯一差异：期望形状从
`tuple(tokens.shape)` 改为 `(1, tokens.shape[1])`：

```python
expected = (1, tokens.shape[1])
if tuple(embedding.shape[:2]) != expected: return False   # batch != 1 或 seq 不匹配
...逐层 ffn_out 同检...
except (KeyError, RuntimeError): return False
```

一次检查同时覆盖 srs FR2 的三个静默/拒绝子例（不完整 / seq 不匹配 /
批量 != 1）——三者的**分流发生在下游 eager 层**（前两者成功返回，后者
RuntimeError），graph 层不区分。`.shape` 为宿主元数据，`cache.fetch` 在
`VectorizedActivationCache` 全掩码 get 下返回视图（#27，只读安全），零
D2H 同步。

### 2.5 `_publish_graph_children`（新私有方法，行切片化 :671-709）

```python
def _publish_graph_children(
    self, runner: Any, tokens: Any, branch_ids: Sequence[str],
    parent_branch_id: str, step_idx: int,
) -> None:
    embedding = self.profile.embed_module(tokens)   # [N, T, H] 一次计算
    num_layers = len(self._wrapped_layers)
    for layer_idx in range(num_layers):
        for row, branch_id in enumerate(branch_ids):
            activations: dict[str, Any] = {
                "ffn_out": runner.child_buf[layer_idx][row : row + 1].clone()
            }   # [1,T,H] 行切片 + clone——与 eager 批量 store 形状逐位一致（AR015）
            if layer_idx == 0:
                activations["embedding"] = embedding[row : row + 1]  # [1,T,H]
            self.cache.put(branch_id=branch_id, layer_idx=layer_idx,
                           activations=activations)
            GLOBAL_STABILITY_PROFILER.record(
                branch_id=branch_id, parent_branch_id=parent_branch_id,
                layer_idx=layer_idx, step_idx=step_idx,
                stable_mask=runner.mask_buf[layer_idx][row : row + 1],  # [1,T] 与串行形状一致
                tau=self.gate.tau, metric="cosine",
            )
```

要点：
- **行切片必须 `clone`**：`child_buf` 静态缓冲跨重放复用，不 clone 的
  发布会在下次重放被覆写（串行 :692 同款契约的批量强化）；
- 切片形状精确镜像 eager 批量 store（[1,T,H]）与串行 profiler record
  （mask [1,T]）——下游 `get_profile`/`mean_stable_ratio` 链路零改动；
- `mask_buf[l][row:row+1]` 是 device tensor 视图，`record` 内部做
  device-side 求和，无额外宿主读回（#P0-3）；
- 循环序（外层 layer 内层 row）与串行模板结构对齐，便于 diff 审查。

### 2.6 `cuda_graph.py`：零改动验证

设计主张：runner 捕获体 batch 无关（:206-209/:218-227/:338/:345/:350-408
均已在探索期核验）。UT-1501/1502 的对拍即运行时证据；若实现期发现
必须改动（例如 `replay` 中存在隐藏的 batch 假设），**回到 design 修订**
（srs FR3 允许"有据最小改"，须记录改动点与理由），不得静默顺手改。

## 3. 测试设计：`tests/test_batched_graph_forward.py`

夹具（沿 `test_batched_branch_forward.py`/`test_cuda_graph.py` 先例）：
- 合成 `ManualFoldedForward` 栈（小形状：L=2、H=32、T=16、V=64、
  **N=4 统一**——与 srs UT-1501/UT-1504 的 N=4 一致，design 门控
  Minor-N1 收口；CUDA 可用时 device="cuda"，否则整文件
  `pytest.mark.skipif` 保护——graph 捕获是 CUDA-only 语义，CPU 无可测
  对象）；cache 默认 `ActivationCache`（UT-1508 例外，见该行；**UT-1509
  亦例外——review 门控 Minor-1 勘误**：发布隔离断言需在重放后 mutate
  `runner.child_buf` 并观测已发布 cache 项不变，而 `ActivationCache.put`
  复制入私有缓冲使"跳 clone"的污染面不可观测，故该 UT 用
  `ChunkedActivationCache`（put 存引用）——测试 docstring 已注明该偏离）；
- parent 前向 populate：`mff(tokens_parent[1,T], branch_id="p",
  parent_branch_id=None)` 后 `mff(child, branch_id=None,
  branch_ids=[...], parent_branch_id="p")`；
- 种子固定（`seed` fixture）；gate `SimilarityGate(tau=0.95, metric="cosine")`；
  无 scheduler（#37 互斥）。

容差（按 srs FR4 逐项）：graph vs 批量 eager logits `torch.allclose(atol=1e-4,
rtol=1e-4)`（串行先例 test_cuda_graph.py:396——padded 容量 gather 与
eager 路径的机制性差异）+ **逐行 argmax 相等**；逐分支逐层 cache
`ffn_out` allclose 1e-4；layer 0 `embedding` 行 == `embed_module(tokens)`
对应行 **allclose 1e-6**；逐分支 profiler 统计量 `mean_stable_ratio`
**allclose 1e-6**（srs 勘误条款：fused gate 与 plain gate 的浮点路径
差异在阈值邻域可能翻转个别 token 的 stable 判定，统计量容差 1e-6 覆盖
非邻域场景，若实测翻转则在实现期勘误并放宽该子项）；mask 逐位相等
**不作断言**（ULP 差经层间传导可在深层翻转个别判定——仅作信息性
记录）。

FR3 广播预填契约的等价性论证（m-6）：UT-1501/1502 对拍对象为批量
eager；批量 eager 的 expand（[1,T,H]→[N,T,H]）与批量 graph 的
`_prefill_parent` copy_ 广播（:338/:345）逐行复制**同一 parent 值**，
故逐行 allclose 即同时钉住"与串行单分支 graph 对应值 allclose"——
eager 与 graph 两条路径对同一 parent 行的折叠输入逐位同源。

| UT | 断言核心（编号与 srs FR4 逐条对齐） | 观测面 |
|----|----------|--------|
| UT-1501 | 同 parent 等长 N=4 候选：首次批量调用捕获，第二次同形批量调用重放；重放输出 vs AR015 eager 批量（fresh 无 graph 实例）logits allclose(1e-4) + 逐行 argmax 相等；逐分支 cache ffn_out allclose(1e-4)；逐分支 mean_stable_ratio allclose(1e-6)（勘误条款见上）；"第二次走重放"探针 = `FoldedGraphRunner.capture` spy 计数恒 1（或 `_publish_graph_children` 调用计数 2） | 分发主路径 + 单次捕获（兼为 mutation ① 观测面） |
| UT-1502 | 逐分支发布完备性：重放后逐分支 cache 含全部 L 层 ffn_out + layer 0 embedding；embedding 行 == `embed_module(tokens)` 对应行（allclose 1e-6）；被拒分支语义不在本 UT（graph 层不做 prune，调用方职责同 AR015） | 发布正确性 |
| UT-1503 | 串行 graph 共存零干扰：同实例先批量捕获重放，再串行捕获重放（`graph_runner`/`graph_runner_batched` 非同一对象、形状各异）；随后再次批量重放仍正确——两 runner 独立性钉住；**顺带 FR3 字面直接对拍（design 门控 Minor-N2）**：批量 graph 重放输出各行 vs 串行 graph 重放输出（同 parent 同 tokens 行）逐行 allclose(1e-4)——不经 eager 中转，闭合 m-6 论证的传递性松弛 | 串行/批量隔离 + FR3 字面 |
| UT-1504 | 形状不匹配回退：捕获 [4,T] 后来电 [2,T]（合法批量但形状不同）→ 一次性 UserWarning（断言两个形状在消息中）+ 结果 == 批量 eager；第二次不匹配零新告警 | 形状退化 |
| UT-1505 | 预算超出回退：`graph_capacity_ratio` 极小（0.02）+ 中段大差异候选（divergent 超 C）→ 一次性 UserWarning + 丢弃重算，结果 == 批量 eager（正确性优先）；`runner.budget_exceeded` 置位 | 预算分支（mutation ④ 观测面） |
| UT-1506 | 前置矩阵：scheduler / 非 cosine gate / CPU / parent 缓存空 / mask 对象不匹配 → 不捕获或不重放 + 结果 == 批量 eager（CPU 一次性告警与降级家族同路）；parent 批量 2 → 未捕获 + `pytest.raises(RuntimeError)` | 退化矩阵 |
| UT-1507 | capture 一次性失败（`side_effect=[RuntimeError]` 或失败即还原 patch——类级持续 patch 会连带炸掉后续串行捕获）→ 一次性告警 + `_graph_batched_capture_failed` 置位 + **patch 还原后**串行合格调用仍可捕获 | EX-002 批量版 |
| UT-1508 | (a) parent 以 batch 2 缓存后首次批量调用 → 零告警 + 未捕获 + `pytest.raises(RuntimeError)`；(b) **捕获成功后** parent 被串行路径**同 id 覆写**：`mff(tokens[N,T], branch_id="p")`（[N,T,H] 覆盖原 [1,T,H]），再次同形批量调用 → 零告警 + 重放被拒 + `pytest.raises(RuntimeError)` + cache/profiler 零污染（raise 先于任何 store——folded_transformer.py:476-484）+ runner 状态不受扰（parent 以 [1,T] 重新 populate "p" 后重放成功）。**夹具（design 门控 Major-N1）：本 UT 用 `ChunkedActivationCache`**（可插拔公共 API #22）——其整 chunk 替换语义（chunked_cache.py:133-135）使同 id 批量增大覆写**合法可达**（srs FR1 场景前提成立）；`ActivationCache` 的 `put` 对同 key 批量增大在 `buffer[:seq].copy_()` 处抛 RuntimeError（activation_cache.py:154，activation_cache.py 实证）——两种 cache 下步骤 8 防线均必要（ActivationCache 下该场景以 put 异常面呈现，chunked 下以静默错值面呈现，后者是 Major-1 洞） | Major-1 观测面（mutation ⑤ 红面：删再校验 → 不抛错 + cache 污染 → 对拍必红） |
| UT-1509 | mutation spot-check（纪律沿用 AR015，见下方 mutation 计划）——发布隔离断言：重放后 mutate `runner.child_buf`，已发布 cache 项不变（clone 生效，mutation ② 观测面） | mutation 纪律 |

mutation 计划（T002 Green 后逐个验证红、SHA256 逐字节还原，沿 AR015
纪律；编号与 srs UT-1509 对齐 + Major-1 专属 mutation ⑤）：
- ① 删 `forward` 批量 graph 分发块（恢复静默旁路）→ UT-1501 红
  （"第二次同形调用走重放"探针翻红：capture spy 计数变 0/`graph_runner_batched` 恒 None）；
- ② `_publish_graph_children` 发布跳过 `.clone()` → UT-1509 红
  （第二次重放覆写 `child_buf` 后第一次发布的 cache 值变化）；
- ③ `_publish_graph_children` 发布跳过某分支（如 row 1）→ UT-1502 红
  （cache 完备性：该分支 L 层 ffn_out 缺失）；
- ④ 删 `budget_exceeded` 回退块 → UT-1505 红（预算超出仍返回重放
  输出 ≠ eager 结果）；
- ⑤ 删重放前 `_parent_cache_complete_batched` 再校验（§2.3 步骤 8）
  → UT-1508(b) 红（调用不再抛错 + cache 被污染——每分支对不同父行
  折叠，对拍必红；req 门控 Major-1 的直接观测面）。

## 4. 基准脚本：`scripts/ar016_batched_bench.py`

先例：`scripts/ar002_graph_bench.py`（唯一有效工件
`results/optimization/ar002_graph_bench.json` 的生产者，#39 纪律出处）。

模型与负载（FR5 负载构造逐项）：合成因果模型（可 detect 架构、CUDA），
`ManualFoldedForward` 栈 L=8、H=256、T 按 config、V=1000；parent [1,T]
缓存一次；每步 N 个同 parent 等长候选（**中段分歧 + 独立 suffix**，
同 AR015 场景族）验证 S 步（跨步同形——graph 可重放）。
**对 srs §6 风险表 "H=32" 的偏离声明（m-1）**：H 提至 256 的理由——
H=32 时每层 GEMM 仅 32×~86 FLOPs，墙钟由 kernel 启动开销主导且差异
落入噪声带，无法检验 H1/H2；显存复算：静态缓冲 (L+1)×N×T×H×4B
≤ 9×8×512×256×4B ≈ 38MB，加 child/mask/logits 缓冲与双 runner 驻留
< 200MB，**srs "远低于 16GB" 结论仅被加强**；该偏离不触碰任何 FR
验收语义（负载形状本就属 design 细节）。

四路径（每配置，与 srs FR5 逐字对齐）：
1. `baseline_eager`：`raw_model(tokens)` N 次全量前向（非折叠对照）；
2. `folded_serial_eager`：AR015 前的串行折叠（逐分支 `branch_id` 单调
   用，eager）；
3. `folded_batched_eager`：AR015 批量 eager（`branch_ids`）；
4. `folded_batched_graph`：本 AR 批量 graph 分发。

扫描：N ∈ {2, 4, 8} × T ∈ {128, 512}（批量 graph 路径 N*T ≥ 256，
fused gate 在 512 时 N=2 恰过 1024 阈值——覆盖两侧）；每配置：warmup
≥3 + 重复 ≥10，**每重复内四路径轮转交错**（rep r：a→b→c→d），
`torch.cuda.synchronize()` 包裹边界，`time.perf_counter` 计时（非
CUDA-event `time_forward`——后者排除宿主侧 prefill/`tolist` 读回/发布，
而批量 graph 净收益恰含这些项，srs FR5 已精确化）；统计量复用
`TimingStats`/`stats_from_samples`（scripts/algo_experiments.py:196/:216）。

正确性随行：每配置记录四路径 tokens argmax 一致性
（allclose 容差内标记 `logits_parity: true/false`）与 mean
stable_ratio——质量不因加速路径漂移是工件的一等公民字段。

假设判定（标签与 srs FR5 逐字对齐——design 首版曾互换标签，系违反
验收文本，已回正；与结果分离，如实记录）：
- **H1**：批量 eager 显著快于 folded 串行（方向 ~N×）——
  supported iff Δt 中位 > 2×合并 std；
- **H2**：批量 graph 进一步降低每步墙钟（vs 批量 eager）——同阈值；
- **H3**：四路径 logits 一致（`logits_parity` 全 true → supported）；
- 差异 < 噪声 → `inconclusive`（诚实优先）；
- **P1-7 专项判定**（独立字段 `p1_7_verdict`，不占用 H 编号）：
  folded 批量 graph vs baseline eager 的墙钟对比——"让 folded 验证
  真正快于 baseline" 论点的首次墙钟数字，无论方向如实记录。

工件 `results/optimization/ar016_batched_bench.json`：
`meta`（GPU 名称/驱动/WDDM 声明/**运行日期**（RERUN_CHECKLIST §6）/
torch 版本/种子/CUPTI 探测结果——预期 null 记 null + 注记（#39：墙钟
为 perf_counter 实测，kernel 计数不可测不编造）/交错方式声明
round-robin/复现命令）
+ `configs[]`（每配置四路径 TimingStats + parity + stable_ratio）+
`hypotheses`（H1/H2/H3 各自 supported/refuted/inconclusive + 支撑
数字）+ `p1_7_verdict`。

运行方式：`python -m scripts.ar016_batched_bench`（#31 端口纪律，无
`.py` 后缀）；`--quick` 子集开关（单配置 N=4/T=512）供 CI 冒烟，默认
全量；**运行预算：全扫描单机 < 10 分钟**（6 配置 × 4 路径 × (3+10)
次 × 毫秒级前向 + 捕获开销，远低于预算；超时如实记录不裁剪）。

## 5. 文档改动（T005）

- `docs/OPTIMIZATION_GUIDE.md`：P1-7 增补第二十三部分——批量 graph 机制
  摘要 + **实测墙钟表**（六配置 × 四路径 p50）+ H1/H2/H3 判定 + 后续档
  （变长/多 parent/Triton 批量）；
- `AGENTS.md`：#37 批量旁路表述更新（批量调用现可走 graph，串行契约
  不变）+ #41 扩注（批量 graph 契约：独立 runner/重放前 parent 再校验/
  发布行切片 clone）；
- `CHANGELOG.md`：AR016 节（Unreleased 最前）；
- **AR015 遗留清单中的「graph 批量捕获」标记收口**（FR6 验收项——
  指南第二十二部分遗留档列表 + P1-7 后续档列表处加 ✅ AR016 注记）；
- `srs.md`/`design.md`/`tasks.md` 归档至 `specs/archive/AR016-batched-graph-verification/`。

## 6. 明确不做（srs 约束镜像）

- 变长候选 / 多 parent / `attention_mask` 非 None 批量（静默 eager）；
- `cuda_graph.py` 任何"顺手"改动（有据最小改须回 design 记录）；
- 串行 graph / AR015 eager / scheduler / gate 任何行为改动；
- folded_generate 生长循环接入（T 每步 +1，永不在捕获形状——#20）；
- Triton 批量 kernel（`_fused_gate_mask_count_torch` 回退在
  N*T < 1024 时服务 capture，fused 路径 ≥1024 由基准大形状侧覆盖）。

## 7. 风险与缓解（设计层补充）

| 风险 | 缓解 |
|------|------|
| 捕获体内隐藏 batch 假设（设计主张错） | UT-1501/1502 对拍即证据；发现即回 design 修订（§2.6） |
| `_publish_graph_children` 外 layer 内 row 循环的 N×L 次 `cache.put` 开销挤占 graph 收益 | 发布成本计入批量 graph 路径计时（perf_counter 全程）；基准如实反映，不为美化路径剔除发布 |
| WDDM 墙钟噪声压过 H1/H2 差异 | ≥10 重复 + p50 + 交错 + `inconclusive` 诚实判定；判定阈值 2×合并 std；H 提至 256 提升信噪比（§4 偏离声明，srs "远低于 16GB" 结论仅被加强） |
| 双 runner 驻留显存（[1,T] 串行图 + [N,T] 批量图 + 静态缓冲） | 基准夹具 L=8/H=256/T=512/N=8 下静态缓冲 <200MB；真实模型场景属后续档 |
| **既有 cache 隐患面（信息记录，非本 AR 职责——design 门控 Minor-N3）**：`ActivationCache` 反方向再 put（先 [N,T] 再 [1,T] 同 key）静默广播覆写（每行同值写入 [T,N,H] 缓冲） | §2.3 步骤 8 的重放前 parent 批量再校验恰好也覆盖该态（shape[0]!=1 → 拒绝）；后续 AR 可在 cache 层收口 |
