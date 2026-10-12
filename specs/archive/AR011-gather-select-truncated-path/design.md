# AR011 详细设计 — 截断路径 gather_select 解禁

> srs：`specs/archive/AR011-gather-select-truncated-path/srs.md`（权威契约）
> 创建：2026-10-12；状态：final（design 门控第 1 轮 PASS，review PASS，ST Go）

## 1. 设计总览

方案：**A（最小解禁——删一个条件项）**——`_merge_parent_child` D3 分发
条件删除 `and not parent_longer`，其余逐字保留。`fetch_flat` 的行公式与
守卫**天然正确**（req 门控逐行核实）：行按请求 `seq_len`=T_c 计算 →
读父缓存前 T_c token（=截断前缀）；`total<seq_len` / 环回（`start>0`）/
batch 不符自动 None → 回退。

**否决 B（`fetch_flat` 增加显式 `allow_longer` 参数）**：现守卫已正确，
加参数是自证性 API 膨胀；`vectorized_cache.py` 零改动是本 AR 的约束。

**否决 C（parent-longer 专用融合变体）**：与等长路径**同一**调用即可
——行公式已保证语义，分叉实现反而引入两份待测代码。

## 2. 关键算法与数据流

### 2.1 分发条件修改（folded_transformer.py:410-417）

```python
        batch, seq_len, hidden_dim = child_out.shape
        if (
            prefix_len is None
            and seq_len >= _GATHER_SELECT_MIN_TOKENS
            and hidden_dim >= _GATHER_SELECT_MIN_HIDDEN
            and (not _GATHER_SELECT_REQUIRE_CUDA or child_out.is_cuda)
        ):
```

仅删除 `and not parent_longer` 一行。融合分支体（:418-430）与回退分支
体（:431-440）**零改动**。

### 2.2 等价论证（req 门控核实结论，测试钉死）

| 项 | 回退路径（现行） | 融合路径（解禁后） | 等价性 |
|---|---|---|---|
| 父 FFN 读取 | `fetch` → 视图/幸存 gather → `[:, :T_c]` 截断 | `fetch_flat` flat 视图 + rows（按 T_c 计算） | 同一内部 buffer 的相同元素（flat 是 buffer 的 reshape 视图；无环回时行 t*batch+b ↔ buffer[t,b] = 父 token t < T_c） |
| 合并 | `merge_stable_divergent` = `where(mask, parent, child)` | `gather_select` = `index_select` + `where` | AGENTS #30 既有契约；T016 等长先例已钉 |
| dtype/device | 同源（同一缓存 buffer） | 同源 | 无转换 |

环回时（`start>0`）：`fetch` 返回幸存窗口（ring-gather，行 0..c-1
实存错位 token），`fetch_flat` 拒绝——**这正是守卫存在的意义**，
UT-1002 钉死。

### 2.3 docstring 更新（_merge_parent_child）

`parent_longer` 参数描述改写：保守跳过 → "the fused path is enabled:
``fetch_flat`` rows are computed for the requested child length and read
exactly the truncated prefix; an eviction-wrapped parent (``start > 0``)
still falls back because ``fetch_flat`` returns ``None``"。**:402-408**
（design 门控 WARN-4 行号勘误）的 "flat-row layout contract covers
exactly seq_len stored tokens" 陈述勘误（守卫要求的是**覆盖**而非
**恰好**）。

## 3. 接口设计

**公开接口零改动**。变更：`_merge_parent_child` 分发条件 -1 行 +
docstring；`_merge_parent_child` 签名不变（`parent_longer` 参数保留——
它仍用于快路径语义与文档，只是不再门控融合分发）。

> **注**：`parent_longer` 参数在解禁后是否仍有消费者？核查：:349-354
> 调用点传入；`_merge_parent_child` 内删除分发项后参数仅剩 docstring
> 语义——**保留**（调用点语义明确、未来回退诊断有用；删除参数会改
> 签名违反零改动约束。pyflakes 对未使用参数不报）。

## 4. 测试设计

### 4.1 单元测试（UT-1001~1003，新文件 tests/test_gather_select_truncated.py）

沿用 T016 模式：`DummyLayer`、`SimilarityGate(tau=0.95)`、
`VectorizedActivationCache`、三常量 monkeypatch
（`MIN_TOKENS=0`/`MIN_HIDDEN=0`/`REQUIRE_CUDA=False`）、
`fused_ops_module.gather_select` 记录 wrapper（同时 patch
`folded_transformer_module.gather_select` 引用，T016 :803-806 先例）。
**缓存 schema 按 T014**：layer 0 存 `{"ffn_out": p0}`（gate 输入）、
layer 1 存 `{"ffn_out": p1}`（父 FFN）。

| ID | 构造 | 断言 | 对应 srs |
|---|---|---|---|
| UT-1001 | T_p=8 无环回（capacity=8，T016 helper 直接可扩）；child T_c=4 = p0[:, :4] 前缀 + 位置 1、3 **定值扰动**（`+100*randn`，`_t016_mixed_child` 先例——小扰动可能击不穿 tau=0.95 导致全 stable 走快路径）；**前向前显式 sanity**：`0 < gate(child, p0[:, :4]).sum() < numel`（T016 :749-751 先例，防 flaky）；**双实例模式**（沿用 T016 :753-767：默认阈值实例 vs 三常量 patch 实例，各自独立缓存） | patch 态：`gather_select` 恰一次调用；`torch.equal(out_default, out_fused)`；组成 sanity：稳定位输出 == p1[:, :4] 对应值、扰动位 == DummyLayer(child) 对应值 | 3.1 |
| UT-1002 | **独立构造** `VectorizedActivationCache(max_entries_per_layer=6)`，put T_p=8 的 p0/p1（环回：start=2，`fetch` 幸存窗口 = token 2..7）；child T_c=4 = **幸存窗口前缀**（原 token 2..5）+ 位置 1、3 扰动；三常量 patch | (a) 组成 sanity：稳定位输出 == 幸存窗口 p1 的前 4 个对应值（证明抵达 `_merge_parent_child` 父复用合并，防全 divergent 空转）；(b) `gather_select` **零调用**；(c) `torch.equal` 默认阈值运行 | 3.2 |
| UT-1003 | varlen：父 T_p=3 短于子 T_c=5（`VectorizedActivationCache`，capacity≥5）；child = 父 3 token 前缀 + 2 新 token；三常量 patch | `gather_select` **零调用**（varlen 不可融合行为钉子——显式跳过与 `total<seq_len` 守卫双重防护，钉最终行为）；输出正确（前缀稳定 + 后缀 divergent 语义，既有 AR003 行为） | 3.2 |

**Red 预期**：UT-1001 失败（`gather_select` 零调用 vs 期望 1 次——
`not parent_longer` 仍挡）；UT-1002/1003 为**行为钉子**（Red 期即过：
现行代码本就不调用）。假 Green 防护：UT-1001 的记录 wrapper 若因
patch 顺序错误而零调用，断言 `len(calls) == 1` 翻红（非空转）。

### 4.2 集成/回归

| ID | 场景 | 断言 |
|---|---|---|
| IT-1004 | T014~T016（等长融合全套）+ UT-601~605 + IT-611/613 | 零改动全过（等长/截断回退路径不受解禁波及） |
| BS-1001 | 全量 `pytest tests\ -q -m "not slow"` | 785 passed（782 基线 + 3 新增）零失败 |
| BS-1002 | demo.py | FLOPs reduction 85.5% / MSE 2.35e-03 / stable token ratio 93.75% 精确一致 |
| BS-1003 | 静态 | mypy clean / pyflakes 零 / 100 列 |

## 5. 影响范围

| 文件 | 改动 |
|---|---|
| actfold/core/folded_transformer.py | 分发条件 -1 行 + docstring（~10 行净变化） |
| tests/test_gather_select_truncated.py（新） | UT-1001~1003 |
| AGENTS.md（#20 修订 + #30 阈值勘误）、docs/OPTIMIZATION_GUIDE.md（第十八部分）、CHANGELOG.md | T003 |

**零改动声明**：`vectorized_cache.py`、`fused_ops.py`、
`split_layer.py`、cuda graph 路径全部不动。

## 6. 决策记录

| # | 决策 | 理由 |
|---|---|---|
| D1 | 最小解禁（删一个条件项），不加参数不分叉 | `fetch_flat` 守卫与行公式已正确（req 门控逐行核实）；B/C 是自证性膨胀 |
| D2 | `parent_longer` 参数保留（虽不再门控分发） | 零签名改动约束；调用点语义/回退诊断价值；pyflakes 不报未用参数 |
| D3 | UT-1002 child 按幸存窗口前缀构造 + 组成 sanity | req 门控 Major：环回 `fetch` 返回 token 2..7，直觉构造（原始 0..3 前缀）会全 divergent → 全量重计算分支 → "零调用"空转通过 |
| D4 | UT-1003 表述为行为钉子 | 父短于子时显式跳过与 `total<seq_len` 守卫行为不可区分——钉最终行为而非实现路径 |
| D5 | 三常量 monkeypatch（含 REQUIRE_CUDA=False） | CPU 环境融合分支门控；T016 :764-766 先例；漏 patch 则 UT-1001 假失败 |

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 环回边界（start>0 但恰覆盖请求） | `fetch_flat` 现行守卫拒绝一切 start>0（:373）——UT-1002 钉死；`vectorized_cache.py` 零改动使守卫不被本 AR 意外触碰 |
| `gather_select` 与 `merge_stable_divergent` 语义漂移 | AGENTS #30 既有契约 + T016 等长先例 + UT-1001 `torch.equal` 三重钉 |
| 生产大形状路径（≥2048 token）无墙钟证据 | 与等长路径同一 kernel 同一收益（F8 既有论证）；CUPTI 限制不设墙钟（AGENTS #39）；bit-identical 是正确性硬门 |
| varlen 路径被误波及 | `prefix_len is None` 条件保留 + UT-1003 钉子 + AR003 既有全套回归 |
| `parent_longer` 参数成"死参数"误导后续维护 | docstring 明示其语义角色（快路径语义/诊断），§3 注记在案 |
