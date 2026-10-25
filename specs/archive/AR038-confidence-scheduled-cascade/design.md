# [AR038] 详细设计 — 置信度分级验证调度（CSVC）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR038 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-25 |
| 状态 | Delivered（升档档位为经实测的负面结论） |

## 1. 机制

配置面（全部默认关，与 AR037 末态逐字节一致）：

| 字段 | 默认 | 校验 |
|---|---|---|
| `candidate_budget_escalate` | `False` | 需 history_cache + N>0；`escalate_candidates > num_block_candidates`；与门控互斥 |
| `escalate_candidates` | `8` | 见上 |
| `escalate_threshold` | `0.3` | — |

块级置信度 `g_min = min(softmax(draft_logits)[masked, greedy_token])` 一次计算、三档分流：

```
g_min >= candidate_budget_threshold      -> skip      (0 次额外前向, spec_budget_skips += 1)
escalate_threshold <= g_min < skip_thr   -> standard  (N = num_block_candidates)
g_min <  escalate_threshold              -> escalate  (N = escalate_candidates, spec_budget_escalations += 1)
```

`n_rows` 只影响候选画布与验证行数，不改变提交语义（仍是 min-prob 胜者，`sb=block` 下为贪心行）。

## 2. 实测（192 token 长几何，真实 checkpoint，`stop_token_id=-1`）

| 配置 | 墙钟 | skips | escalations | verify | mean log p |
|---|---|---|---|---|---|
| D + CDF（N=4） | 16.02 s | 0 | 0 | 6 | −2.0870 |
| D + CDF + escalate(N=8) | **23.35 s** | 0 | 6 | 6 | **−2.0870** |
| ACB + CDF（τ=0.5） | **11.84 s** | 3 | 0 | 3 | −2.0870 |
| ACB + CDF + escalate | 15.23 s | 3 | 3 | 3 | −2.0870 |

**判读**：

1. **升档零质量收益**：N=8 与 N=4 的 mean log p 完全相同（−2.0870，逐位一致），
   确认「贪心锚定行总是胜者」在本 workload 下成立，增加候选行只增加验证计算。
2. **升档明确增加成本**：相对 N=4 多 +46%（23.35 vs 16.02 s），相对 ACB 多 +29%。
3. 因此 H1（"escalate ≥ standard"）**形式上成立（相等），但机制目的（用算力换质量）不成立** →
   按 AR035 先例交付为**默认关的消融性负面**，并记录「test-time 候选扩展在本架构中的价值上界 = 0」。

## 3. 结论

- 有价值的是 **skip 档**（AR037 ACB，实测 −29% 墙钟、输出等价）；
- **escalate 档不推荐启用**，默认关保留仅作为已测量的消融证据与将来换 workload 的旋钮；
- 若未来要在低置信块真正换取质量，需要改变**提交语义**（如自校正重掩码）而非增加候选行。

## 4. 测试

| ID | 面 |
|---|---|
| 3801 | 默认关：`spec_budget_escalations == 0` |
| 3802 | 恒升档：`escalations == verify` 且 `draft + fused == verify + skips` |
| 3803 | 标准档（`escalate_threshold=0.0`）输出与关闭态逐字节一致 |
| 3804 | `escalate_candidates <= num_block_candidates` → `ValueError` |
| 3805 | 非法组合（无 history/N、与门控同开）→ `ValueError` |

## 5. 回退矩阵

| 条件 | 行为 |
|---|---|
| `num_masked == 0` | `g_min = None`，不分流（保持原路径） |
| 阈值区间外 | 三档均不触发 → 与关闭态逐字节一致 |
| 未启用 | 完全不计算 `g_min`（零开销、零随机数变化） |
