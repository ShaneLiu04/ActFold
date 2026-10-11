# AR007 详细设计 — P3-2 第一档：parent 更长时的截断前缀复用

> srs：`specs/changes/AR007-truncated-parent-reuse/srs.md`（权威契约）
> 创建：2026-10-11；状态：draft

## 1. 设计总览

方案：**A（ folded 层内三点截断）**——截断完全收敛在
`FoldedTransformerLayer` 内（gate 输入 / 全 stable 快路径 / 慢路径
parent FFN），对外接口（`ManualFoldedForward`/cache/graph）零改动。
**否决 B（cache 层提供截断视图）**：`fetch`/`get_all` 返回的 parent 激活
被多个消费方共享（gate 比对、merge、profiler），在 cache 层截断会改变
所有调用方的可见形状，影响面不可控；folded 层是形状语义的唯一拥有者
（AR003 的 prefix 对齐同样落在这一层，先例一致）。

```
forward(child [B, T_c, H], parent_branch_id)
  ├─ h_parent = cache.get_all(parent, gate_layer)[gate_name]   # [B, T_p, H]
  ├─ 形状守卫（修订）:
  │    失效 → None:   dim≠3 / batch≠ / hidden≠ / T_p == 0        （不变）
  │    新:  T_p > T_c → h_parent = h_parent[:, :T_c]            （截断复用）
  │    并记录 parent_longer = (原始 T_p > T_c)
  ├─ 截断后 prefix_len == T_c == seq_len → 既有等长分支
  │    （fused gate 可用：fused_ops.py:667-672 显式支持非连续输入，
  │     但注意截断后 h_parent 切片连续性——见 D1）
  ├─ 全 stable 快路径: fetch parent ffn [B, T_p, H]
  │    → parent_longer 时截断 [:, :T_c] → 比对 shape → 复用
  └─ 慢路径 merge: parent_longer 时跳过 fetch_flat/gather_select
       → fetch parent ffn → 截断 → merge_stable_divergent（等长语义）
```

## 2. 关键算法与数据流

### 2.1 形状守卫与截断（folded_transformer.py:170-182 修订）

```python
parent_longer = False
if h_parent is not None:
    if (h_parent.dim() != 3
            or h_parent.shape[0] != hidden_states.shape[0]
            or h_parent.shape[2] != hidden_states.shape[2]
            or h_parent.shape[1] == 0):
        h_parent = None                      # 既有失效条件不变
    elif h_parent.shape[1] > seq_len:        # AR007 截断复用
        h_parent = h_parent[:, :seq_len]
        parent_longer = True
```

`parent_longer` 是**本层本次前向**的局部事实（每层独立判定：不同层
parent 长度可能不同——cache 逐层存储，正常路径一致，但设计上不假设跨层
一致，局部变量天然安全）。

### 2.2 三个消费点

| 消费点 | 现行 | 截断后 |
|---|---|---|
| gate 比对（:191-244） | `prefix_len == seq_len` 走等长分支 | 不变——截断后天然等长；fused gate 按等长契约调用（D1） |
| 全 stable 快路径（:269-290） | fetch ffn → `ffn_out.shape == hidden_states.shape` 校验 | `parent_longer` 时 fetch 后 `ffn_out = ffn_out[:, :seq_len]` 再校验；复用值即截断 parent FFN |
| 慢路径 merge（:342-397） | `prefix_len is None` → fetch_flat/gather_select 或 fetch+merge | `parent_longer` 时**禁用 gather_select**（fetch_flat 的 flat_rows 按缓存布局索引 child 长度行，parent 更长时对应关系不保证）→ fetch → 截断 → `merge_stable_divergent`；`prefix_len` 仍传 None（截断后等长） |

**实现结构（D2，design 门控第 1 轮修订）**：forward 在守卫段记录局部
`parent_longer: bool`（本层本次前向），**显式传入** `_merge_parent_child`
作为 gather_select 禁用条件——等长路径保持现行 fetch_flat 优先顺序零
额外 fetch（AR002 F8 免物化优化不回归）；parent 更长路径走 fetch +
截断 + `merge_stable_divergent`。跨层长度错位的异常 cache 状态以
fetch 后实际形状兜底截断（与局部布尔可能矛盾时以形状为准，行为仍
确定）。快路径（全 stable）同样在 fetch 后按实际形状截断再校验。

### 2.3 正确性论证

- **真前缀（收益场景）**：因果模型，child 前 `T_c` token 与 parent 前
  `T_c` 逐位相同 → 各层激活逐位一致 → cosine == 1.0 ≥ τ → 全 stable →
  快路径复用截断后的 parent FFN == child 全重算结果（bit-exact，与
  AR003 append-only 等长前缀同款论证）。
- **非前缀（安全网场景）**：child 前 `T_c` token 与 parent 不同 → 激活
  不相似 → gate 判 divergent → 重算。构造为**独立采样** child（req W2）：
  token 完全独立 → 激活相关性为零 → cosine 远低于 τ=0.95 → 全 divergent
  → `stable_count == 0` → 全重算 → bit-exact。避免"翻转单 token"构造
  （部分 stable 的断言意图歧义）。
- **变长不进入 graph**：`ManualFoldedForward._parent_cache_complete`
  （architecture_utils.py:516）精确形状检查，parent 更长 → 回退 eager，
  与 AR003 变长步骤同路径，零干扰（IT-613 实证）。

## 3. 接口设计

**公开接口零改动**：`FoldedTransformerLayer.forward` 签名、
`ManualFoldedForward`、cache 协议、`BranchManager.align_tokens` 均不变。
改动全部为 `folded_transformer.py` 私有实现（形状守卫 + 两个私有方法的
内部截断逻辑）。

| 内部函数 | 变更 |
|---|---|
| `forward`（:170-182 守卫段） | `T_p > T_c` 截断替代置 None；注释改写 |
| `forward`（:269-290 快路径） | fetch 后按形状截断再校验 |
| `_merge_parent_child`（:342-397） | 新增 `parent_longer: bool = False` 参数（**唯一**禁用机制，替代/覆盖 gather_select 判定）：True 时跳过 gather_select 走 fetch+截断+merge；False（等长）路径行为逐字节不变 |
| `_get_parent_ffn_output`（:399-440） | 契约放宽**仅限 `prefix_len is None`**（等长/截断后路径）：`ffn_out.shape[1] >= mask_len` 时截断返回；`< mask_len` 仍 RuntimeError。varlen 路径（`prefix_len` 非 None）**保持精确 shape 契约不变**（EX-505 基线行为，design 门控 Critical 修复） |

**公开接口零改动**（重申）：`forward` 签名不变；`_merge_parent_child`
的新参数带默认值且为私有方法。

## 4. 测试设计

### 4.1 单元测试（UT-601~604，tests/test_folded_transformer_varlen.py 追加）

| ID | 断言 | 对应 srs |
|---|---|---|
| UT-601 | 因果 token-wise 模型：parent 8 token 填 cache，child 6 token **真前缀** → folded 输出 `torch.equal(layer(child))` + profiler `layer_stats[*].stable_ratio == 1.0`（W1 口径：`GLOBAL_STABILITY_PROFILER`，各 folded 层；参照 tests/test_stability_profiler.py:32-34 accessor）。**同 UT-604 warm-up 约束**（防全重算空转通过：无 warm-up 时快路径不可达，bit-exact 断言对全重算恒真——warm-up 后复用路径才被实际覆盖） | 3.1-AC1 |
| UT-602 | 同 fixture，child 6 token **独立采样**（非前缀，token 无关）→ 输出 `torch.equal(layer(child))`（gate 全 divergent → 全重算） | 3.1-AC2 |
| UT-603 | parent 更长 + batch 不匹配（或 hidden 不匹配）→ 仍视为无 parent 全重算（既有失效条件零回归） | 3.1-AC3 |
| UT-604 | parent 更长且全 stable 时快路径复用值 == 截断 parent FFN（spy `_recompute_all` 调用次数 == 0）。**fixture 约束（design 门控 Major 修复）**：`_returns_tuple` 仅由 `_recompute_all` 置位（冷启动 :280 强制重算回退）——须先用**同一 folded 层实例**跑一次 parent 前向（warm-up，先例 test_folded_transformer.py:331-332），spy 在 warm-up 后挂载 | 3.1（复用真实性） |
| UT-605 | 跨层长度错位（构造钉死：gate 条目 == child 长度 + merge 条目 **短于** child → 慢路径 `_get_parent_ffn_output` RuntimeError 既有机制；该组合为既有可达路径的钉死测试） | 3.1（异常 cache 状态） |

### 4.2 集成测试（IT-611~613）

| ID | 场景 | 断言 |
|---|---|---|
| IT-611 | `ManualFoldedForward` + 因果合成 4 层模型：parent 8 token 前向填 cache → child 6 token 真前缀 folded 前向 | 输出与 eager 全重算 `torch.equal`；profiler 各 folded 层 stable_ratio == 1.0；cache 含 child 条目 |
| IT-612 | 既有 varlen append-only 全套 + 等长全套 + 全量 754 | 零回归 |
| IT-613 | graph 模式（`use_cuda_graph=True`）+ parent 更长 child 步骤（**child tokens shape == 捕获 shape**，变长仅在 parent cache 侧——否则 :572-584 一次性 shape 告警破坏零告警断言） | 不捕获/零告警/不禁用（silent eager 回退，AR003 既有断言风格，参照 test_cuda_graph.py） |

### 4.3 边界/系统测试（BS-701~703）

- BS-701：全量 `pytest tests\ -q -m "not slow"` = 754 + 新增，零回归；
- BS-702：demo 基线 85.5% / 2.35e-03 / 93.75% 不变；
- BS-703：mypy 标准命令 / pyflakes / 100 列。

## 5. 影响范围

| 文件 | 改动 |
|---|---|
| actfold/core/folded_transformer.py | 形状守卫截断 + 快路径截断 + merge 截断 + gather_select 长度前置条件 + 注释/docstring |
| tests/test_folded_transformer_varlen.py | UT-601~604 + 因果 fixture + UT-203 docstring 改写（req W4） |
| tests/test_folded_generation.py 或新文件 | IT-611（ManualFoldedForward 端到端） |
| tests/test_cuda_graph.py | IT-613 + :1017 docstring 核对（req W4） |
| docs/AGENTS/CHANGELOG/README/指南 | T003 |

**零改动声明**：`architecture_utils.py`（含 `ManualFoldedForward` 与
`_parent_cache_complete`）、`cuda_graph.py`、`vectorized_cache.py`、
`fused_ops.py`、`branch_manager.py`、`folded_generation.py`、
verification engine。

## 6. 决策记录

| # | 决策 | 理由 |
|---|---|---|
| D1 | 截断收敛在 folded 层（方案 A），cache 层不动 | cache 视图被多消费方共享；AR003 prefix 对齐同层先例 |
| D2 | 布尔 `parent_longer` 显式传 `_merge_parent_child` 禁 gather_select；形状作异常兜底 | 等长热路径保持 fetch_flat 优先零额外 fetch（AR002 F8 不回归）；异常 cache 状态（层间长度错位）以 fetch 后形状兜底，行为确定（design 门控 Major 修复） |
| D3 | `_get_parent_ffn_output` 放宽**仅限 `prefix_len is None`** | srs §3.1 权威契约仅要求等长/截断后路径截断；varlen 路径的精确 shape RuntimeError 是 EX-505 钉死的基线行为，统一放宽会吞掉它（design 门控 Critical 修复） |
| D4 | gather_select 在 parent 更长时保守跳过 | 保守不依赖 vectorized cache 内部布局契约；注：fetch_flat 行公式（vectorized_cache.py:376-378）在非 None 返回时实际确定覆盖前 seq_len token，gather_select 对 parent 更长**实际正确**——留作后续性能 AR 线索 |
| D5 | fused gate 在截断后走等长分支（可用） | 截断切片作为 gate 输入，fused_ops.py:667-672 显式支持非连续输入；等长分支的 fused 调用契约（同形状）在截断后成立 |
| D6 | 非前缀 IT 构造用独立采样 child（req W2） | token 无关 → 激活相关为零 → 全 divergent 确定性；避免部分 stable 断言歧义 |
| D7 | UT-601/IT-611 的 stable_ratio 口径 = `GLOBAL_STABILITY_PROFILER.layer_stats[*].stable_ratio`（req W1） | tests/test_stability_profiler.py:32-34 先例；各 folded 层逐层断言 |

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 截断切片非连续 → gate/fused/merge 数值异常 | PyTorch 切片语义安全（视图，数值不变）；D5 fused 契约实证；UT-601 bit-exact 兜底 |
| 快路径复用截断 parent FFN 与 child 真前缀全重算不一致 | 因果逐位一致论证（§2.3）+ UT-604 spy `_recompute_all` 计数 == 0 实证复用发生 |
| 层间 parent 长度不一致（异常 cache 状态） | D2 形状兜底自洽；UT-603 覆盖失效条件；UT-605 钉死跨层错位的既有 RuntimeError 行为（design 门控 Minor 修复） |
| gather_select 禁用遗漏导致错误行索引 | D4 按 ffn 实际形状判定 + 既有 gather_select 测试（等长路径）零回归 |
| 旧行为存量断言（UT-203 等）docstring 假陈述 | T003 清理清单（req W4）；测试本体在新语义下仍通过（review 预判），全量零回归佐证 |
