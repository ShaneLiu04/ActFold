# [AR011] 需求设计说明书 — 截断路径 gather_select 解禁

| 字段 | 内容 |
|------|------|
| AR 编号 | AR011 |
| AR 主题 | `_merge_parent_child` 的 parent-longer（AR007 截断）路径解除 `gather_select` 保守跳过——单遍融合合并恢复覆盖截断复用 |
| 关联 SR | AR007 review 线索（"`fetch_flat` 行公式对 parent-longer 实际正确 → gather_select 解禁是后续小 AR 候选"）；指南 P2-4/F8 融合路径覆盖面；AGENTS #20/#30 |
| 日期 | 2026-10-12 |
| 状态 | Final（req/design 门控 PASS，T001–T004 passing，review PASS，ST 7/7 Go，已归档） |

## 1. 背景与目标

AR007 在 `_merge_parent_child` 的 D3 分发条件中**保守**加入
`not parent_longer`（folded_transformer.py:413）：截断复用路径跳过
`gather_select` 单遍融合，走 `fetch` + 截断 + `merge_stable_divergent`
（两遍内存），理由是"flat-row 布局契约只覆盖恰好 seq_len 的存
token"。

**该理由经核实不成立**：`fetch_flat`（vectorized_cache.py:332-384）的
行公式 `flat_rows[b*seq_len+t] = t*batch + b`（:376-378）按**请求的**
`seq_len`（=子长 T_c）计算——指向父缓存前 T_c 个 token 行，恰为 AR007
截断前缀 `parent[:, :T_c]`；守卫 `total < seq_len → None`（:373）只要求
存储**覆盖**请求（允许 `total > seq_len`），环回驱逐（`start > 0`）自动
返回 None 走回退。即 `fetch_flat` 对 parent-longer **天然正确且自带安全
网**，跳过条件是多余的保守。

目标：解除跳过，让截断复用路径在大形状（D3 阈值）下恢复单遍融合合并
（省一次父 FFN 的物化遍历）；安全网（环回/varlen/非向量化缓存）行为
钉死不变。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `_merge_parent_child` D3 分发条件删除 `and not parent_longer` 一项
  （其余条件 `prefix_len is None` + 阈值 + CUDA + `fetch_flat` 存在
  逐字保留）；
- docstring 更新（`parent_longer` 参数描述与类级注释：保守跳过 →
  解禁理由 + 环回回退说明）；
- 测试：parent-longer 融合等价（bit-identical + 恰一次调用）、环回
  驱逐回退（不调用）、varlen（prefix_len）路径跳过钉子；
- 文档：AGENTS #20 修订（"conservatively skipped" → 解禁 + 理由）、
  指南第十八部分回链、CHANGELOG AR011 节。

**Out of Scope（本 AR 不做的）：**
- `vectorized_cache.py` **零改动**（`fetch_flat` 守卫已正确：覆盖检查
  + 环回拒绝——本 AR 依赖并钉死它，不改它）；
- D3 阈值本身（2048/4096/CUDA 不动）；
- varlen（AR003 prefix_len）路径的 gather_select——父短于子时
  `total < seq_len` 必然 None，显式跳过保留（防御性，语义不变）；
- CUDA graph 路径（截断步骤本就走 eager fallback，AGENTS #20，无交集）。

## 3. 功能需求

### 3.1 截断路径恢复单遍融合合并

**描述：** parent-longer 且 D3 阈值满足时，`_merge_parent_child` 走
`gather_select` 而非 fetch+截断+merge。

**触发条件：** `prefix_len is None` + `parent_longer=True` + 阈值满足 +
cache 有 `fetch_flat` 且返回非 None（父 token 覆盖请求且未环回）。

**期望行为：**
- 删除分发条件中的 `not parent_longer`——parent-longer 与等长路径
  **同一**融合分支：`gather_select(flat_buffer, flat_rows.view(batch,
  seq_len), child_out, stable_mask)`；
- 行公式语义：`flat_rows` 按 `seq_len`=T_c 计算 → 读父缓存前 T_c
  token = AR007 截断前缀——与现行回退路径 `merge_stable_divergent(
  parent_ffn[:, :T_c], child_out, stable_mask)` **逐位一致**；
- 阈值不满足 / `fetch_flat` 缺失（非向量化缓存）/ 返回 None 时照旧
  回退（行为零变化）。

**异常处理：** 无新异常；环回驱逐由 `fetch_flat` 现行守卫
（`start > 0 → None`）拦截 → 回退路径。

**验收标准：**
- Given 向量化缓存存 T_p=8 的父（hidden+FFN，capacity=8 无环回），
  child T_c=4 为父真前缀加扰动（混合稳定 mask），**三常量 monkeypatch**
  （`_GATHER_SELECT_MIN_TOKENS=0` / `_GATHER_SELECT_MIN_HIDDEN=0` /
  `_GATHER_SELECT_REQUIRE_CUDA=False`，T016 先例 :764-766——缺一
  则 CPU 上融合分支静默走回退），When 前向，Then `gather_select` 被
  调用**恰好一次**（记录 wrapper 证据），且输出与默认阈值（回退路径）
  的输出 `torch.equal`（**UT-1001**）；
- Given 等长混合稳定前向（T016 既有用例），When 全量回归，Then 零
  改动全过（等长融合路径不受影响的钉死）。

### 3.2 安全网行为钉死

**描述：** 三类不可融合情形继续走回退，且不被解禁波及。

**验收标准：**
- Given **独立构造**的环回驱逐缓存（`VectorizedActivationCache(
  max_entries_per_layer=6)` + put T_p=8 张量——T016 helper 硬编码
  capacity=T_p 无法触发环回，**不可复用**，req 门控 Minor-3），child
  T_c=4 **按幸存窗口前缀构造**（环回 `fetch` 返回幸存 token 2..7，
  gate 比对的是 token 2..5 而非原始 0..3——child 取 token 2..5 为基底
  加部分扰动，req 门控 Major），三常量 monkeypatch 至融合启用，When
  前向，Then：(a) **输出组成 sanity**——未扰动位置输出逐位等于父
  幸存窗口对应值、扰动位置等于子重算值（证明前向确实抵达
  `_merge_parent_child` 的父复用合并，而非全 divergent 空转走全量
  重计算分支 ：327-330）；(b) `gather_select` **不被调用**
  （`fetch_flat` 因 `start=2>0` 返回 None）；(c) 输出与默认阈值运行
  `torch.equal`（**UT-1002**）；
- Given varlen（父 T_p=3 短于子 T_c=5，prefix_len 路径，
  **`VectorizedActivationCache`**——指定缓存类型，req 门控 Minor-4），
  三常量 monkeypatch 至融合启用，When 前向，Then `gather_select`
  **不被调用**（varlen 路径不可融合的**行为钉子**——显式跳过与
  `fetch_flat` 的 `total<seq_len` 守卫双重防护，行为上不可区分，钉
  最终行为即可，**UT-1003**）；
- Given 全量基线（782 passed），When 回归，Then 零失败（含
  UT-601~605/IT-611/613 与 T014~T016 全部既有融合测试）。

### 3.3 文档收口

**验收标准：**
- AGENTS #20 修订：截断复用处 "gather_select is conservatively
  skipped" 改为解禁表述（fetch_flat 行按请求子长计算恰读截断前缀；
  环回驱逐自动回退）；**顺带勘误**（req 门控 WARN-6）：#30 的阈值
  描述 "T≥2048 with H≥8192, or T≥8192" 与代码/T016 钉死的
  2048/4096 不一致——修正为现行值；
- 指南第十八部分 AR011 回链（P3-2 系列收尾注：截断复用现已完整覆盖
  融合路径）；CHANGELOG AR011 节（Changed）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 截断路径大形状 | 跳过中间物化/内核合并，与等长路径同收益（F8 既有理由；回退路径 fetch/截断本是视图无拷贝——"省一遍物化"指融合省去中间 gather 物化，req 门控 WARN-5 措辞软化；不设墙钟指标——CUPTI 限制，AGENTS #39） |
| 质量 | 静态检查 | mypy 标准命令 clean / pyflakes 零 / black 100 列 |
| 质量 | 全量回归 | `pytest tests\ -q -m "not slow"`：782 passed 基线 + 新增零失败 |
| 兼容 | 公开 API | 零签名改动（纯分发条件与 docstring） |

## 5. 约束与假设

**约束：**
- `vectorized_cache.py` / `fused_ops.py` 零改动；
- bit-identical 是硬门：融合与回退两路径输出 `torch.equal`
  （UT-1001 是行为等价的核心证据，非仅"输出接近"）；
- 等长路径字节不动（`parent_longer=False` 时代码路径与今日逐字一致）；
- AGENTS #20/#30/#37 语义对齐（#20 修订属本 AR 范围）。

**假设：**
- `fetch_flat` 的守卫语义（覆盖检查 + 环回拒绝）是稳定契约（AR008
  时代码未动它；本 AR 以 UT-1002 钉死环回行为）；
- `gather_select` 的 `where(mask, gathered, child)` 语义与
  `merge_stable_divergent` 一致（AGENTS #30 既有契约，T016 已钉）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 截断路径 | AR007 parent-longer 复用：父激活 `[:, :T_c]` 截断后走等长分支（本 AR 的解禁对象） |
| 行公式 | `fetch_flat` 的 `flat_rows[b*seq_len+t] = t*batch+b`——按请求子长计算，天然读父缓存前 T_c token |
| 环回驱逐 | 父 token 数超缓存容量（`start>0`），`fetch_flat` 返回 None 的安全网 |
