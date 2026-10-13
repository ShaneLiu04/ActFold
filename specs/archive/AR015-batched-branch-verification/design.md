# AR015 详细设计 — folded_generate 多候选批量分支前向（P1-7 第一档）

> srs：`specs/changes/AR015-batched-branch-verification/srs.md`（权威契约，req 门控 PASS）
> 创建：2026-10-13；状态：final（design 门控首轮 FAIL：Major-1 D4 与 §3.3① 对批量版父 FFN 契约形状互斥——fetch 期 [1,·] vs 期望 [N,·] 两种字面实现一者永不 Green、一者变异不可观测 → 唯一化为三段式（fetch 期校验父存储 [1,prefix_len] → expand → post-expand 输出校验 [N, prefix_len]）；Minor-1 签名草图仅追加 keyword-only 新参不重排位序 / Minor-2 全稳路径补 `_returns_tuple` 守卫 / Minor-3 profiler 对拍面改统计量（mask 不保留）并回写 srs / Minor-4 私有 import 显式标注对 srs 字面偏离 / Minor-5 UT-1403 split 阈值降 ≤T 使双侧均走 split / Minor-6 探针断言改 `kwargs.get("branch_ids") is None`（层循环无条件传参）/ Minor-7 ChunkedCache 视图存储依赖注明（串行同构先例）/ Minor-8 profiler 单例互异 id + 即时捕获——同批修复；复核二轮 FAIL：D4③ varlen 半边期望形状误写 [N,T]（cat 前实际 [N,T_p]）→ 统一为 [N, prefix_len]（等长退化 stable_mask.shape，与串行范本 :476-482 逐字一致）+ srs UT-1406 措辞/状态行勘误——三轮 PASS）

## 1. 设计总览

方案：**A（`branch_ids` 复数 kwarg 贯穿三层 + 专用 `_forward_batched`
+ 严格父契约 + 逐分支 store/profiler + 串行原样回退）**——三个生产
文件各一处局部改动，一个新测试文件，指南勘误。

**否决 B（branch_id 接受 str | Sequence[str] 单参数多态）**：类型
分支魔法；`branch_id=None + branch_ids=[...]` 显式单复数互斥，
mypy 可精确标注，调用点一眼可辨。

**否决 C（批量逻辑内联进现行 forward）**：串行路径是 817 测试的
回归网（srs NFR"逐位不动"），内联必然搅动三向拆分/快路径代码；
独立 `_forward_batched` 使串行 forward 除入口 3 行外零改动。

**否决 D（靠 PyTorch 广播隐式处理 [1,..] 父张量）**：gate
（`F.cosine_similarity`）与 merge（`torch.where`）对 [1,T,H] 父都
能广播——但隐式广播使父批量契约不可观测（mutation ① 不可测），
且未来 Triton 批量 kernel 需要良构 stride。显式 expand（stride-0
零拷贝）+ 严格 shape 契约是可测性与前瞻性的来源。

## 2. 生产设计

### D1 层级入口（folded_transformer.py）

```python
def forward(self, hidden_states, branch_id=None, parent_branch_id=None,
            attention_mask=None, step_idx=0, *, branch_ids=None, **kwargs):
    # 签名仅追加 keyword-only 新参，位序与现行完全一致（design 门控 m-1）
    if branch_ids is not None:
        if branch_id is not None:
            raise ValueError(...)          # 单复数互斥（FR2 同款）
        return self._forward_batched(
            hidden_states, attention_mask, list(branch_ids),
            parent_branch_id, step_idx, **kwargs)
    <现行串行体逐位不动>
```

- `_ACTFOLD_KWARGS`（:21）增 `"branch_ids"`——`_recompute_all`
  过滤面（:539-541）不泄漏进 original layer；
- `branch_ids` 为 keyword-only——位置调用零影响；
- 串行体唯一触碰是入口前置 3 行，其余逐位不动。

### D2 `_forward_batched` 流程（与串行逐分支同构）

```
N = len(branch_ids)；N != hidden_states.shape[0] → ValueError
parent_branch_id is None → 整批 _recompute_all + 逐分支存 + pack   (no-parent)
scheduler should_fold False → 同上                                  (同 :146-149)
scheduler tau 覆盖（同 :151-158）
gate 父激活 fetch（gate_layer/gate_name 同 :166-179，miss → None）
h_parent is None → 整批 _recompute_all + 逐分支存                   (cache-miss，req Major-2)
h_parent 校验：dim!=3 / hidden 不匹配 / len==0 → 视同 miss（重算）
            ：shape[0] != 1 → RuntimeError                          (expand 契约，FR1)
h_parent 截断（长于 child → [:, :seq]，parent_longer=True，同 :196-204）
h_parent = h_parent.to(dtype/device).expand(N, -1, -1)              (D4：显式 expand)
prefix_len = h_parent.shape[1]
```

- **cache-miss 分支的必然性**：生成循环第一步 root 缓存为空
  （req 门控 Major-2），此分支是旗舰路径而非边角。

### D3 gate 与三向拆分（批粒度）

- **gate 固定 plain 路径**（`self.gate(...)`）：等长
  `stable_mask = gate(hidden, h_parent)` [N,T]；varlen
  `gate(hidden[:, :prefix_len], h_parent)` + cat 全 False 尾
  （:244-266 同构）。**不用 fused gate kernel**（stride-0 parent
  未在 kernel 内验证，srs FR1 取舍；串行 fused 分发 :225-241
  逐位不动）；
- profiler 逐分支 record：`stable_mask[i:i+1]` 行切片、
  `parent_branch_id` 单值（:269-277 同构）——串行对拍面；
- `stable_count = int(stable_mask.sum())`——批粒度单次读回
  （F4a 同型）；
- 三向拆分（批粒度）：
  - **全稳**（count==numel）：fetch 父 FFN（`layer_idx=self.layer_idx`
    ，:293-296 同位）→ AR007 截断（:299-309 同型）→ expand(N) →
    shape 校验 [N,T,H] + **`self._returns_tuple is not None` 守卫**
    （与串行 :310-313 同款——首调输出结构未知时重算，共享 cache
    多实例场景不发散，design 门控 m-2）→ 逐分支存行切片（不重算）；
    missing/形状不符/结构未知 → 整批重算存（:317-323 同型）；
  - **全不稳**：整批 `_recompute_all` + 逐分支存（:327-330 同型）；
  - **混合**：整批 `_recompute_merged(hidden, mask,
    stable_count=...)`（:338-344 同型，Split 子类 [N,T] 掩码天然
    泛化）→ D4 批量 merge → 逐分支存；
- **逐行快路径不复制**（srs FR1 记录的取舍）：混合批内全稳行经
  merge 掩码全 True 取父值 == 串行快路径结果（逐位等价）；生成
  循环后缀恒 divergent（AR003 全 False 尾），串行同走 slow path
  ——旗舰场景零损失。

### D4 批量 merge（固定 PyTorch 实现）

```python
from actfold.core.fused_ops import _merge_stable_divergent_torch
```

- 跳过 gather_select（`fetch_flat` 按 parent 存储批量计行，对
  [N,T] 恒 None、必走回退——直接跳过，srs Minor-5）；
- 父 FFN 批量版取用契约为**三段式**（design 门控 Major-1 唯一化）：
  ① fetch 期校验**父存储形状** `[1, prefix_len]`（varlen 契约
  :476-482 同型、批量维期望保持 1）/ 等长 AR007 截断（:484-499
  同型）；② `expand(N, -1, -1)`；③ **显式 post-expand 输出形状
  校验**：前两维 == `[N, prefix_len]`（校验先于 varlen cat；
  等长路径下 prefix_len==seq_len 自然退化为 `stable_mask.shape`
  ——与串行范本 :476-482 `expected=[stable_mask.shape[0],
  prefix_len]` 逐字一致），违者 `RuntimeError`——与 D3 全稳路径
  的 shape 校验同款。①③ 双校验使"删 expand"变异在 varlen 与
  等长两条路径上都以契约 RuntimeError 显式翻红（§3.3①），不
  依赖 gate/cat 的隐式触发；
- varlen：`cat([parent_ffn, child_out[:, prefix_len:]], dim=1)`
  （:434-439 同构，expand 后 [N,T_p,H]+[N,T-T_p,H]）；
- **直调 `_merge_stable_divergent_torch`**（fused_ops.py:38，签名
  兼容）——**对 srs FR1 字面"调 merge_stable_divergent 的
  PyTorch 实现路径"的有意偏离**（design 门控 m-4 显式标注）：
  私有跨模块 import 同包 core 内部、单一调用点，比给公开函数加
  `force_torch=True` 绕过分发的参数更小地扩大公开面；UT-1401/
  1403 逐位对拍即其正确性钉。Triton 批量化档期改公开参数时同步
  收口；
- expand 后无广播依赖——确定性 + mutation ① 可观测（§3.3）。

### D5 store 与输出

- `_store_activations` 逐分支复用：`h_out[i:i+1]` /
  `hidden_states[i:i+1]`（dim0 切片保持 contiguity；layer 0 附
  embedding 行切片，:571-574 同构）——**不改 `_store_activations`
  本体**，批量侧新增 3 行循环 helper；
- **存储安全依赖注明**（design 门控 m-7）：安全主要来自 put 侧
  拷贝（`ActivationCache.put` activation_cache.py:154 /
  `VectorizedActivationCache.put` vectorized_cache.py:144 均
  `copy_`）；`ChunkedActivationCache.put` 对 contiguous 切片
  `.contiguous()` **原样存视图**（chunked_cache.py:123）。串行
  全稳快路径同样存 fetch 视图（:293-316 无 clone），批量的
  expand 行切片与之同构——**非本 AR 新引入的暴露面**；本 AR 批量
  仅经 folded_generate（分支 id 唯一、单次 put、被拒分支即刻
  clear），实际安全；ChunkedCache 视图语义的既有暴露面留待独立
  收口（风险表记录），本 AR 不加 clone 以保持与串行存储形态
  逐位同构；
- `_pack_output` 原样复用（tuple re-wrap #13 不变；批量返回
  [N,T,V] 或 ([N,T,V], None)，ManualFoldedForward 层循环消费
  方式与串行完全一致）。

### D6 ManualFoldedForward 入口（architecture_utils.py）

```python
def forward(self, tokens, branch_id, parent_branch_id=None,
            attention_mask=None, step_idx=0, branch_ids=None):
    # branch_id: str | None（保持必填，类型放宽——req Minor-3）
    if branch_ids is not None:
        if branch_id is not None:
            raise ValueError(...)      # 单复数互斥
        # 批量：跳过 graph 块（静默 eager，#20 varlen 先例；#37 契约）
    <graph 门（仅 branch_ids is None 时评估）>
    <embedding → 层循环（追加 branch_ids=branch_ids 实参）→
     final_norm → head 原样>
```

- 层循环**无条件**传 `branch_ids=branch_ids`（串行传 None →
  层入口 3 行判断后走原路径，行为逐位不变；避免复制整个循环体）；
- graph 门条件由 `if self.use_cuda_graph:` 收紧为
  `if self.use_cuda_graph and branch_ids is None:`——串行 graph
  路径零变化，批量静默 eager（UT-1408 钉）；
- `SplitFoldedTransformerLayer` 继承 forward（只覆写
  `_recompute_merged`，:513）——批量路径自动泛化，UT-1403 钉。

### D7 folded_generate 分发（folded_generation.py）

```python
def _batched_forward_supported(folded_model, candidates) -> bool:
    # ① isinstance ManualFoldedForward  ② len>1
    # ③ 同 parent 且非 None  ④ tokens 形状相同且 shape[0]==1
```

- `folded_generate` :139 处：

```python
if _batched_forward_supported(folded_model, candidates):
    for node in candidates:
        tree.add(node)
    _run_folded_forward_batched(model, candidates, folded_model, step_idx)
    evaluated = list(candidates)
else:
    <现行串行循环逐字保留>
```

- `_run_folded_forward_batched`：逐 node
  `profiler.reset_branch`（:270 同构）→ `torch.cat([...], dim=0)`
  → `folded_model(stacked, branch_id=None, branch_ids=[...],
  parent_branch_id=pid, step_idx=...)` → `node.logits =
  logits[i:i+1]`（dim0 视图，消费面 :153/:155/:245 均只读）→
  逐 node profile 写 `metadata["stable_ratio"]`（:293-295 同构）；
- `model` 形参保持（与 `_run_folded_forward` 对称；批量路径不
  用——`del model` 于函数体）；
- 独立谓词函数 = **可 monkeypatch 面**（UT-1405 强制串行对照、
  UT-1406 矩阵的单元级钉点）；
- 条件④取 `candidates[0].tokens.shape[0] == 1`（draft 恒
  `parent.tokens.clone()` 派生，候选批量即父批量——无需触达
  parent 节点）；
- `tree.add`/prune/`clear_branch`/acceptance（:139-172）零改动。

## 3. 测试设计（tests/test_batched_branch_forward.py）

### 3.1 基础设施

| 组件 | 设计 |
|---|---|
| 合成模型 | 文件内自建 `_CausalBatchModel`（逐层因果 cumsum/线性，参照 test_folded_generation.py `CausalCumsumModel` :476-556 先例但不跨测试文件 import——各测试文件自载小模型是仓内惯例）；`_build_setup(device, split_layers=False, use_cuda_graph=False)` → (folded, cache, gate)，固定种子 |
| launch-count spy | `monkeypatch.setattr(FoldedTransformerLayer, "forward", counting_wrapper)`（类级、委托原实现、捕获 kwargs）——批量 L vs 串行 N·L；同一 spy 兼作"branch_ids 是否被传入"探针（UT-1406） |
| 强制串行开关 | `monkeypatch.setattr(folded_generation, "_batched_forward_supported", lambda *a, **k: False)`（UT-1405 对照） |
| 对拍纪律 | 两套 fresh setup（cache/gate/folded 各自独立实例、同种子同权重模型）分别跑批量与串行——杜绝缓存串扰；**两套使用互异 branch id 前缀**（批量侧 `b0..b3`、串行侧 `s0..s3`）且每侧 run 后**立即捕获** `get_profile`——`GLOBAL_STABILITY_PROFILER` 为全局单例，`get_profile` 有缓存、`reset_branch` 才清（stability_profiler.py:212/:257-264），互异 id + 即时捕获杜绝双侧串扰（design 门控 m-8）；CPU `torch.equal`，CUDA `allclose(1e-6, 1e-6)` + argmax token 相等（Green 期把实测最大差记入测试注释，srs §5 假设二） |
| profiler 对拍 | profiler 仅存 `stable_sum`/`num_tokens` 统计、**不保留 mask**（stability_profiler.py:161-179，design 门控 m-3）——对拍面为逐层 `stable_sum`+`num_tokens` 与 `mean_stable_ratio`（sum+num_tokens 相等 ⇒ ratio 相等；srs UT-1401 的"stable_mask 相等"表述据此勘误为统计量相等，mask 级残差由 logits/cache 逐位对拍兜底） |

### 3.2 用例矩阵

| ID | 构造 | 断言 |
|---|---|---|
| UT-1401 CPU varlen 逐位对拍 | 父 [1,T_p] 串行前向填充缓存（pid）；4 候选 [1,T_p+1] 末 token 互异；批量（branch_ids）vs 逐候选串行（两套 fresh setup、互异 id 前缀） | 逐行 logits `torch.equal`；逐分支逐层 cache `ffn_out`+layer 0 `embedding` `torch.equal`；逐分支 profiler 统计量（stable_sum/num_tokens/mean_stable_ratio）相等 |
| UT-1402 CUDA 对拍 | 同 1401，`device` fixture | logits/cache `allclose(1e-6)`；逐行 argmax token 相等；ratio `allclose(1e-6)` |
| UT-1403 等长 + split 变体 | 父 [1,T] 缓存齐；4 候选 [1,T] 中段位置互异（gate 判 divergent）；`split_layers=True` 且 **`split_min_tokens` 降到 ≤T**——split 判定用 `shape[0]*shape[1]`（split_layer.py:272-274），批量 N*T 与串行 T 阈值判定不对称，降阈值使**双侧均走 split 路径**（design 门控 m-5） | **CPU allclose(1e-6) + 逐行 argmax 相等**（T002 勘误：split 聚集 GEMM 的 M 维 = 批量 D_total vs 串行 D_i，串行单 divergent 行时 M=1 走 BLAS gemv 微内核、批量 M>1 走 gemm——跨微内核边界 1 ULP，语义逐位同构由 mask/profiler 统计精确相等 + argmax 相等钉住；原"CPU 逐位"假设数值性证伪） |
| UT-1404 launch-count 代理 | 层级 spy；同 1401 场景批量 1 次 vs 串行 N 次 | 批量层调用计数==L；串行==N·L；**不 assert 墙钟**（#39） |
| UT-1405 folded_generate 端到端 | 因果 adapter + ManualFoldedForward + `DraftGenerator` 多候选（`num_branches_per_step=3`，`max_new_tokens=3`）；跑两遍：默认（批量）vs 谓词强制 False（串行） | tokens/`stable_ratio`/`acceptance_rate` 完全相等（CPU 逐位同序计算）；层 spy 证第一遍各层收到 `branch_ids` 非 None、第二遍 `kwargs.get("branch_ids") is None`（D6 层循环无条件传参——串行侧断言其值为 None 而非缺席，design 门控 m-6）；结束后 cache 无被拒分支残留（prune+clear_branch 生效，:168-172） |
| UT-1406 回退矩阵 | 谓词单元钉：N=1 / 形状互异 / 混合 parent / `folded_model=None` / FoldedModel 实例 → False；greedy 单候选端到端 spy | 谓词五情形 False；greedy 运行层 spy 断言 `kwargs.get("branch_ids") is None`（m-6 同款）且生成正确 |
| UT-1407 层级前置契约 | 手工构造：`len(branch_ids)!=batch` → ValueError；层/入口两处单复数同给 → ValueError；父缓存批量 2（[2,T] 父前向播种）→ RuntimeError；`branch_ids`+无 parent → 与串行 no-parent 对拍；**parent id 给定但缓存空**（root 第一步形态）→ 与串行 cache-miss 对拍（req Major-2） | 各前置异常类型精确；两个对拍分支逐位相等 |
| UT-1408 graph 旁路 | CUDA + `use_cuda_graph=True` 实例：批量调用 → 结果正确；随后同实例串行调用（父缓存齐、同形）→ graph 正常 capture/replay | 批量后 `graph_runner` 状态零变化（仍 None）；串行调用后 graph 生效且结果正确——批量未毒化 graph 路径（参照 :593 graph 干扰测试先例） |

### 3.3 mutation spot-check（Red 期，四变异）

| # | 变异 | 观测面 |
|---|---|---|
| ① | 删父 expand | D4 三段式契约 ③ post-expand 输出校验：fetched [1,T_p,H] 前两维 != [N,T_p] → RuntimeError → UT-1401/1403 红（varlen 与等长两路径均显式触发；gate ValueError（similarity_gate.py:78-81）与 cat 形状错为次要兜底）。**严格双校验即可观测性**（PyTorch 广播本会静默通过——§1 否决 D 的实证） |
| ② | 整批存 `branch_ids[0]` 单一 id | 其余分支 cache 缺失 → UT-1401 cache 对拍红 |
| ③ | 跳过逐分支 profiler record | `get_profile(bid)` None/ratio 缺 → UT-1401 profiler 对拍红 |
| ④ | 删分发（谓词恒 False） | UT-1405 "branch_ids 被传"探针红（实测验证）。**T002 勘误**：原表把 UT-1404 也列为 ④ 观测面——实测 UT-1404 直调 folded 模型不经谓词、不翻红；UT-1404 钉的是 L vs N·L 机制本身，分发删除的观测面是 UT-1405 探针 |

- 行↔id 错排（行 i 存进 id j）由 ② 同观测面覆盖（候选末 token
  互异，cache 对拍即错排探针），不单列。

## 4. 文档设计（FR5，T004）

- 指南 :180 P1-7 勾选 + 勘误注（行号 :114-117→:139-143；engine
  无多分支循环（清单腐化）；第一档守护条件四条；后续档：变长/
  多 parent/Triton 批量/graph 批量）；"~1/N 墙钟"表述挂
  launch-count 代理注（#39，不引用未实测数字）；
- 第二十二部分回链；CHANGELOG AR015 节（Added：`branch_ids`
  批量分支契约；Performance：多候选验证层数调用 L vs N·L）；
- AGENTS #41：批量分支契约 pitfall（单复数互斥、expand 契约
  parent batch==1、cache-miss 分支、plain gate + PyTorch merge +
  跳 gather_select、graph 静默 eager、串行路径逐位不动）。

## 5. 风险与回退

| 风险 | 缓解 |
|---|---|
| CUDA 批量 vs 串行数值差超 1e-6 | srs §5 假设二预案：收紧为 CPU 逐位 + CUDA token 相等；Green 期记录实测值于 UT-1402 注释 |
| `_merge_stable_divergent_torch` 私有 import 面漂移 | 同包 core 内部、单一调用点、UT-1401/1403 逐位对拍即其正确性钉；Triton 批量档期改公开参数时同步收口 |
| 层循环追加 `branch_ids` 实参扰串行 | 传 None → 层入口 3 行后走原路径；817 全量 + demo 锚点（FR1/FR2 串行层路径）双兜底 |
| graph 实例批量/串行混用毒化 | UT-1408 钉 graph 状态零变化 + 串行后仍可 capture；graph 门条件收紧为 `and branch_ids is None` |
| profiler 逐分支 record 开销回归 | 与串行 N 次调用完全同阶（srs §5 假设四）；无新增断言 |
| 谓词条件④依赖 draft 派生不变量 | 候选批量==父批量由 `tokens.clone()` 派生保证；层级 RuntimeError 双防线（FR1） |
| ChunkedCache 视图存储暴露面（m-7） | 串行全稳快路径同存 fetch 视图（:293-316 先例），批量同构非新引入；本 AR 分支 id 唯一 + 被拒分支即刻 clear；独立收口留待后续 |
| profiler 全局单例双侧串扰（m-8） | §3.1 纪律：互异 branch id 前缀 + 每侧 run 后立即捕获 `get_profile` |
| split 聚集 GEMM 的 gemv/gemm 微内核边界（T002 实测验伪 §5 假设二 CPU 半边） | 串行单 divergent 行 M=1 走 gemv、批量 M=D_total 走 gemm → 1 ULP（L0 实测 2.4e-07）；UT-1403 收紧为 allclose(1e-6)+argmax——语义同构不受影响（mask/统计精确相等），非 split 路径（UT-1401）CPU 逐位不受影响 |
