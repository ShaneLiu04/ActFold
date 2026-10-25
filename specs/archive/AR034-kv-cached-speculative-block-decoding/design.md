# AR034 详细设计 — KV-cached speculative block decoding

| 组件名称 | actfold/models/fast_dllm_sampler.py（`history_cache` × 投机组合分支）+ tests fixture + scripts/ar034_kv_speculative_bench.py |
| --- | --- |
| AR系统流水号 | AR034 |
| AR描述 | 解除 AR033 的「历史 KV 缓存 × 投机候选」互斥：组合臂用 `[N, 活动块]` 批量验证（batch-1 历史 KV 展开 N 行、只读）× 贪心锚定候选，winner 一次性提交 small block；token 与缓存臂贪心语义等价，launch 数从 ~`2·S·sb+1` 压到 `2·S+1`。 |

# 1 概述

AR033 把重掩码迭代前向从全画布 `[T]` 收缩到活动切片（历史 KV 只读），B2 臂
1.5431× vs 非缓存基线；AR021/AR022 用 `[N, T]` 批量验证把一个小块的重掩码迭代
序列压成「1 草稿 + 1 验证 + 1 提交」。二者此前互斥（AR033 构造期
`ValueError`）。AR034 定义并实现其组合：

- 历史 KV 按 batch 维展开 N 行（只读副本，`update_past_key_values=False`）；
- 草稿与验证都前向**整个活动块**（块内双向注意力要求整块——T001 P5：
  切片前向 vs 整块前向 maxdiff 1.5–8.9，不可用切片）；
- 验证 `[N, block]` 的 logits 按 R1 对齐取活动 small block 的决策行，
  min-prob 选 winner，一次性提交该 small block；
- 随后既有 `[1, block]` commit 前向提交 KV 并产出推进 token（AR032 的
  advance reuse 在缓存路径上天然成立）。

# 2 动态行为

```
prefill(prompt+align) --update=True--> kv, boundary
for block:
  for small_block s:
    if 无 mask: continue
    draft  = raw(x[-block:], past=kv, update=False)          # [1, block, V]
    aligned= cat([boundary, draft[:, :-1]], 1)               # R1 决策行
    cands  = block_view.repeat(N,1) + 填 top-k（row0=argmax）
    verify = raw(cands, past=expand(kv,N), update=False)     # [N, block, V]
    vrows  = cat([boundary.expand(N), verify[:, :-1]], 1)[:, s_lo:s_hi]
    j*     = argmax_j min_pos prob(vrows, cands)
    x[:, s_lo:s_hi] = cands[j*, s_lo:s_hi]; finished |= stop
  commit = raw(x[-block:], past=kv, update=True)             # 提交+推进
  x = cat([x, next_token]); append fresh masks
```

# 3 功能点分解

| 编号 | 功能点 | 说明 |
|---|---|---|
| F1 | 互斥解除 | `history_cache & num_block_candidates>0` 允许；`history_small_block_cache` 组合、`folded_model` 组合、`uncertainty_validation_order` 组合均显式 `ValueError` |
| F2 | 草稿前向 | 整活动块 `[1, block]`，`past=kv, update=False`；R1 对齐；计数 `spec_draft` |
| F3 | 候选构造 | AR022 冻结语义：`top_k` 有序集、row0=argmax 贪心、冻结种子公式、`_cand{j}` 命名；全程 device 端零回读 |
| F4 | 批量验证 | `[N, block]` + `_expand_history_kv(kv, N)`（只读副本），`update=False`；计数 `spec_verify` |
| F5 | winner 提交 | min-prob（活动 small block 掩码位）→ `j*`；一次性提交 whole small block；`finished` 语义同 AR022 |
| F6 | 提交+推进 | 既有 `[1, block]` commit 前向（`update=True`）产出 KV 与推进 token |
| F7 | 账本 | `history_cache_stats` 增 `spec_draft`/`spec_verify`（默认 0） |
| F8 | 短尾块 | `active_len < block_size` 时整块切片语义同 AR033 |

# 4 实现设计

## 4.1 功能实现思路

- 组合臂是 `_history_cached_sample` 内的一个小分支：仅当 `num_candidates>0`
  时，把「`max_iters` 逐个解掩」替换为「草稿 → 验证 → 整块提交 `break`」。
  非组合路径逐字节不变。
- winner 的 KV **不取验证前向**（`update=False` 不写回，行选择依赖 KV 布局）；
  由既有 commit 前向在 `[1, block]` 上重算并提交。
- KV 展开用**显式容器重建 + detach 复制**（不用 `deepcopy`：KV 张量在
  autograd 跟踪下可能是非叶子，torch 拒绝 deepcopy；真实
  `batch_repeat_interleave` 原地修改且返回 None——T001 P1/P4 实测）。

## 4.2 接口描述

```python
@dataclass FastDLLMSamplerConfig:
    history_cache: bool = False
    history_small_block_cache: bool = False
    num_block_candidates: int = 0            # >0 + history_cache => 组合臂
    candidate_top_k: int = 10
    candidate_include_greedy: bool = False

sampler.history_cache_stats
# {prefill_forwards, slice_forwards, commit_forwards, block_rebuilds,
#  forward_tokens, degraded_calls, spec_draft_forwards, spec_verify_forwards}
```

## 4.3 代码设计（变更点）

| 位置 | 变更 |
|---|---|
| `__post_init__` | 删除 `history_cache & N>0` 硬 ValueError；新增 sb-cache/uncertainty 组合 ValueError |
| `sample()` 守卫 | `history_cache & N>0 & folded_model` → ValueError；原生批量投机不再要求 folded_model |
| `_expand_history_kv` | 新模块函数：key_cache/value_cache 或 layers 两种容器，detach + repeat_interleave |
| `_history_speculative_small_block` | 新方法：草稿/候选/验证/提交（F2–F5） |
| `_history_cached_sample` | 小分块循环内 `num_candidates>0` 分支 + `break` |
| 计数器 | `_hc_spec_draft_forwards` / `_hc_spec_verify_forwards` + 账本两键 |

# 5 重构设计

无。组合分支与既有分支共享坐标/`finished`/`history` 语义，不抽公共辅助以
避免改动热路径（零回归优先）。

# 6 测试设计

## 6.1 单元测试（UT-3401~3406，11 例）

| 编号 | 覆盖 |
|---|---|
| UT-3401 | 组合臂多几何确定性 + mask-free（4 形状点） |
| UT-3402 | 前向账本恒等（`spec_draft==spec_verify>0`、prefill=1、degraded 0）+ 组合臂前向总数 < 缓存臂；九键模式 |
| UT-3403/3403b | `_expand_history_kv` 复制语义；真实组合运行中只读前向不修改历史 KV |
| UT-3404 | 默认关零回归（`num_block_candidates=0`：spec 计数 0、输出==AR033 路径） |
| UT-3405 | 四种非法组合 fail-fast |
| UT-3406 | 贪心行参赛（`speculative_calls == spec_verify`、`greedy_wins` 为子集） |

## 6.2 四臂基准（预承诺单次，判定冻结）

臂：**A** 非缓存基线 / **B2** AR033 缓存臂 / **D** 组合臂
（`num_block_candidates=4, candidate_top_k=10, candidate_include_greedy=True`）
/ **C** 官方原生 `batch_sample`（report-only，RAW 未填充 prompt）。

几何：**AR022 血缘 `num_tokens=128, block=32, small_block=32`**（T007 落锤：
投机提交单元必须匹配 AR022 已验证的贪心语义；sb=8 下冻结 min-prob 规则让
随机 top-k 行击败贪心行——实测 2/8 胜、96 vs 34 tokens；sb=32 贪心 100% 胜且
token 级==A），2 prompts，reps=30。

**冻结判定式：**

| 判定 | 条件 |
|---|---|
| **H1 质量** | D/B2 vs A：`|posthoc gap| <= 1e-3`；逐 prompt decode 长度全臂一致；贪心 token 一致率 `D vs A >= 0.98`；任一 None → refuted |
| **H2 加速** | `p50(A)/p50(D) > 1.0` 严格 **且** `p50(B2)/p50(D) > 1.0` 严格；2σ 报告；C 比值 report-only |
| **H3 机制** | D 类账本精确重建（prefill=1、`spec_draft==spec_verify>0`、commit、degraded 0、spec len==block、verify batch==N）+ **D 总前向数 < B2**；A 全量画布结构面；token 行 report-only |
| **H4 诚实** | canary 合取（D armed/degraded 0、B2 armed、拓扑冻结）+ repro 按 `(arm, prompt_index)` 严格相等；失配 → 全部 blocked |

`overall = supported` 当且仅当 H1-H4 全 supported；优先级 blocked > refuted >
supported。工件 `results/optimization/ar034_kv_speculative_bench.json`，quick
面独立路径。

# 7 设计期探针结论（T001，2026-10-24 实测）

真实 Fast_dLLM_v2_1.5B bf16 @Quadro RTX 5000（mask_id=151665，DynamicCache
28 层）：

| 探针 | 结果 | 落锤 |
|---|---|---|
| P1 KV 只读 | 批量验证前后历史 KV `torch.equal` 全真 | 只读合同成立 |
| P2 N=2 行等价 | 两行均与单行前向逐位相等 | 无行间耦合 |
| P3 N=4 行等价 | maxdiff **0.0** / argmax 1.0 | 每行位级等价于独立前向 |
| P4 展开显存 | expand 4.36MB / peak delta 9.63MB | 16GB 主机无压力 |
| P5 切片 vs 整块 | maxdiff 1.5/7.22/8.88，argmax 0.875 | **必须整块前向**；设计据此定稿 |
| KV 展开 API | `batch_repeat_interleave` 原地改且返回 None | 实现用显式容器重建 + detach |

# 7.1 T007 quick 面实测（几何修正依据）

| 变体 | 结果 |
|---|---|
| D（sb=8, N=4 贪心） | decode 96 vs A 34、token rate 0.0、gap 0.398（贪心行 2/8 落败 → 画布污染） |
| D（sb=8, N=1 贪心） | token 级==A |
| D（sb=32, N=4 贪心） | token 级==A（贪心 1/1 胜） |

结论：发散源于随机行击败贪心行（非并行提交本身）；采用 sb=32。quick 四臂：
**overall supported**，p50 A≈23s / B2≈22.5s / **D≈3.1s**、decode 34/34/34。

# 8 交付收口

formal（2026-10-24T12:21:08Z，reps=30，Quadro RTX 5000）：**overall supported**
——p50 A=28101.0 / B2=22447.4 / D=3218.6 / C=25546.9ms；H2 D vs A **8.7308×**、
D vs B2 **6.9743×**（D 比官方快 7.94×，report-only）；H1 gap 0.0 / decode
[34,66] / token rate 1.0；H3 D 前向 8 次 vs B2 62 次、token 行 288 vs 1952 vs A
4576、账本恒等；H4 canary 全绿 + repro 8/8。质量门：black/isort/pyflakes 零
告警、mypy actfold 75 + scripts 32 零 error、pytest **1491 passed / 4 skipped /
3 deselected**。UT-3401~3406（11 例）。
