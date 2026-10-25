# [AR039] 详细设计 — 免费验证 + 无验证贪心提交（FV/VFG）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR039 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-25 |
| 状态 | Delivered |

## 1. 机制

单开关：`verification_free_commit: bool = False` + `free_verify_floor: float = 0.3`。
校验：需 `history_cache=True` + `num_block_candidates>0` + `commit_draft_fusion=True`；与 AR035 门控互斥。

### 1.1 无验证提交（VFG）

块处理时若未处于「再武装」状态：

1. 草稿决策行 → 取仍掩码位贪心 token（AR022 语义）；
2. 直接整块写入 `x`，**不构建候选画布、不发起 `[N,B]` 验证**；
3. **关键**：把草稿尾行的 argmax 交给 `_hc_pending_advance`，使下一块的 CDF 融合前向仍能运行
   （输入等价性同 AR032：草稿画布 == 提交画布）。

### 1.2 免费验证（FV）

CDF 融合前向的输入是 `[block_b | next_canvas]`，它重新读取了刚提交的块。
其决策行 `fused_aligned[:, 0:B]` 即该块每个位置的 softmax 分布：

```
p = softmax(fused_aligned[0, 0:B])[arange(B), x[0, -B:]]   # 已提交 token 的概率
free_verify_blocks += 1
if min(p) < free_verify_floor:  free_verify_low_conf += 1;  _fv_rearm = True
```

零额外前向、零额外 KV 写入（复用已有 logits）。

### 1.3 再武装

`_fv_rearm=True` 时，下一块不跳过验证（走完整 `[N,B]` 路径），随后清零。
即：**免费验证是廉价的哨兵，完整验证只在哨兵报警后付费**。

## 2. 实测（192 token 长几何，真实 checkpoint，`stop_token_id=-1`）

| 臂 | 墙钟 | draft | fused | verify | VFG 块 | FV（低置信） | mean log p |
|---|---|---|---|---|---|---|---|
| A 非缓存基线（AR037 实测） | 172.36 s | — | — | — | — | — | −2.1162 |
| D（CDF） | 14.50 s | 2 | 4 | 6 | 0 | — | −2.0870 |
| S（ACB τ=0.5 + CDF） | 10.16 s | 2 | 4 | 3 | 0 | — | −2.0870 |
| **V（VFG + CDF）** | **8.84 s** | 2 | 4 | **2** | **4** | 4 / 2 | **−2.0870** |

判定：H1（与 ACB 输出等价）**成立**（logp 逐位相同）；H2（VFG < ACB）**成立**（8.84 < 10.16，1.15×）；
H3 账本闭合（`vfg + verify` 覆盖全部块、FV 次数 == fused 次数）；H4 复现（确定性单 rep）。
对基线 **19.5×**。

## 3. 顺带修复（真实缺陷）

AR037/AR038 的跳过路径**未交接 advance token**，使跳过块之后的 CDF 静默失效
（统计上被 draft/fused 混合计数掩盖）。修正后 ACB 自身也从 11.84 s → **10.16 s**。

## 4. 测试

| ID | 面 |
|---|---|
| 3901 | 默认关：三个计数为 0 |
| 3902 | VFG 跳过 + 再武装两条路径都执行（fixture 低置信 → 两者 > 0） |
| 3903 | `free_verify_blocks == spec_fused_forwards` |
| 3904 | 确定性 + mask-free |
| 3905 | 非法组合（缺 history/N、缺 CDF、与门控同开）→ `ValueError` |

## 5. 限制

- VFG 的正确性依赖「贪心行总是胜者」（AR034 T007 / AR038 实测），故 FV 哨兵 + 再武装是**必要**的安全网；
- `free_verify_floor` 需按 workload 标定（本文 0.3 下 6 块中 2 块触发再武装）；
- 该机制只对**组合臂 + CDF** 有意义（其他路径没有融合前向可复用）。
