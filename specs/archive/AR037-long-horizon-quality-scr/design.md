# [AR037] 详细设计 — 自适应候选预算（ACB）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR037 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-25 |
| 状态 | Delivered |

## 1. T001 实测（先测量后设计，已完成）

| 指标（192 token，真实 checkpoint） | A 非缓存基线 | D 缓存×投机 |
|---|---|---|
| mean log p | -2.1162 | **-2.0870** |
| 低置信位 (<0.5) | 48/223 | 43/223 |
| 墙钟 | 172.4 s | **18.3 s（9.4×）** |
| 逐位一致率 | — | 0.4260（首次分歧 @66） |

- 塌缩假设**不成立**：整块贪心提交与阈值重掩码是不同但等价的轨迹；同模型似然 D 略优。
- 长几何加速 9.4×（192）/10.0×（320）> 短几何 8.73×。
- 由此得到 ACB 的动机：`[N,B]` 验证多数只是确认贪心行（AR034 T007 胜率 100%）。

## 2. 机制

配置：`adaptive_candidate_budget: bool = False`、`candidate_budget_threshold: float = 0.9`。
校验：需要 `history_cache=True` + `num_block_candidates>0`；与 `speculative_commit_gate` 互斥（均 `ValueError`）。

判定点在 `_history_speculative_small_block` 的**草稿决策行之后、候选画布构建之前**：

```
g_probs = softmax(small_logits[0])            # 温度 0 语义下即贪心分布
g_tokens = topk(masked, k=1)                  # 仍掩码位的贪心 token
if min(g_probs[masked, g_tokens]) >= threshold:
    commit greedy fill; spec_budget_skips += 1; return True   # 无 [N,B] 验证
```

- 跳过时**零额外前向**：草稿来自 AR036 的 CDF stash 时，该块前向数 = 0；
- 未跳过时行为与关闭态**逐字节一致**（UT-3703 钉死）；
- 账本新增 `spec_budget_skips`（默认关为 0）。

## 3. 实测（阈值扫描 + 全栈组合）

| 配置（192 token） | 墙钟 | verify | draft | fused | skips | mean log p |
|---|---|---|---|---|---|---|
| D（AR034） | 18.80 s | 6 | 6 | 0 | 0 | -2.0870 |
| D + CDF | 16.02 s | 6 | 2 | 4 | 0 | -2.0870 |
| **D + CDF + ACB(τ=0.5)** | **11.84 s** | 3 | 3 | 3 | 3 | **-2.0870** |

- τ=0.9 跳过率 0%（阈值高于块内最弱位置信度）；τ=0.7 → 33%，τ=0.5 → 50%；
- 所有跳过阈值下 **mean log p 与非跳过完全一致**（输出等价，因为贪心行本就是胜者）；
- 对基线 A：**172.36 s → 11.84 s = 14.6×**。

## 4. 回退与边界

| 条件 | 行为 |
|---|---|
| `num_masked == 0` | 不判定（无掩码位） |
| 阈值未达标 | 走原验证路径，逐字节同关闭态 |
| 与 CDF 同时开启 | 跳过块复用 stash，0 额外前向 |
| 与 AR035 门控同时开启 | 构造期 `ValueError` |

## 5. 测试

| ID | 面 |
|---|---|
| 3701 | 默认关：`spec_budget_skips == 0` |
| 3702 | τ=0.0：全部跳过（`verify == 0`、`draft == skips`）、mask-free、确定性 |
| 3703 | τ=1.1：从不跳过，输出与关闭态逐字节一致 |
| 3704 | 非法组合（无 history/无候选、与门控同开）→ `ValueError` |

## 6. 结论与限制

ACB 是**真实有效**的计算面优化：跳过率 50% 时 D+CDF 再快 1.35×，质量不变。
限制：跳过率依赖数据（本 prompt 50%）；`candidate_budget_threshold` 需要按 workload 标定，
默认 0.9 在本文 workload 下过于保守（0 跳过），建议 0.5–0.7。
