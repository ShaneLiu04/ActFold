# [AR041] 详细设计 — 置信度触发的重掩码回退（CTR-F）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR041 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-25 |
| 状态 | Delivered（机制成立；阈值标定给出两种模式） |

## 1. 机制

| 字段 | 默认 | 校验 |
|---|---|---|
| `remask_fallback` | `False` | 需 `history_cache` + `num_block_candidates>0` |
| `remask_fallback_threshold` | `0.5` | 与 ACB/VFG/CDF 可组合 |

块级判定（在草稿决策行上）：`g_min = min(softmax(draft)[masked, greedy])`；
若 `g_min < remask_fallback_threshold`：

1. 把 `block_idx` 加入 `_hc_remask_fallback_blocks`（**块级闩锁**，本轮不再重判）；
2. `spec_remask_fallbacks += 1`；
3. **返回 `False`**，调用方在同一迭代内落入仓库既有的官方重掩码分支
   （`sb_available=False` 切片前向 + `threshold` 逐位置解掩），后续迭代因闩锁不再走投机路径。

回退块不设 advance 交接，其提交走非融合路径（少一次 CDF 机会，不影响正确性）。
`sample()` 入口重置 `_hc_remask_fallback_blocks` / `_hc_pending_advance` / `_hc_draft_stash`。

## 2. 实测（真实 checkpoint，160 token，AR040 三条 prompt）

| prompt | A 官方迭代 | D 整块贪心 | R(τ≥0.2) | 回退块 | logp A / D / R |
|---|---|---|---|---|---|
| p1 | 166.7 s | 10.0 s | 109.1 s | 5/5 | −1.1528 / −1.3118 / **−1.1528** |
| p2 | 106.6 s | 10.4 s | 77.5 s | 5/5 | −0.8482 / −0.9904 / **−0.8482** |
| p3 | 191.7 s | 15.0 s | 131.7 s | 5/5 | −0.8086 / −1.5370 / **−0.8086** |

τ 扫描（0.5/0.3/0.2/0.1）：前三个阈值在 15/15 个块上全部回退；τ=0.1 时 p2 有 1 块放行，
logp 从 −0.8482 掉到 −0.8914 —— **放行即掉质量**，与 AR040 一致。

## 3. 判读（核心产出）

1. **回退机制有效且无损**：R 的 logp 与 A **逐位相同**（三条 prompt 全部），证明官方迭代语义
   在组合臂内被完整复用（`sb_available=False` 路径精确）。
2. **质量安全模式下仍有 1.5× 收益**：R 保留历史 KV 缓存（每次迭代只前向当前块），
   比 A 快 1.53×/1.38×/1.46×，且质量零变化。
3. **`g_min` 过于悲观**：块内 31 个掩码位取最小值，几乎总有位置概率极低（<0.1），
   因此「整块贪心提交」在本 checkpoint 上**没有可安全放行的块**。这不是阈值选得不好，
   而是统计量本身的性质 —— 修正方向是分位数/均值统计量，或提交后修订（见 §5）。
4. 阈值扫描给出可操作结论：`remask_fallback_threshold >= 0.2` ⇒ 质量安全模式（等同官方解码 + KV 缓存）。

## 4. 测试

| ID | 面 |
|---|---|
| 4101 | 默认关：`spec_remask_fallbacks == 0` |
| 4102 | 恒回退：`fallbacks > 0`、`verify == 0`、`draft == fallbacks`（一次判定成本）、mask-free |
| 4103 | 阈值 0（从不回退）：输出与原组合臂**逐字节一致** |
| 4104 | 非法组合（无 history/候选）→ `ValueError` |

## 5. 明确的后续方向

- 更换置信统计量（块内分位数 / 均值 / 高于官方 0.95 的比例）以区分「真不确定」与「单点拖尾」；
- 提交后修订（自校正重掩码）：用 AR039 的免费验证定位低置信已提交位，回滚该块 KV 后重解；
- 两者都必须用 AR040 矩阵做多 prompt 验证，禁止单 prompt 结论。

## 6. 复现

```powershell
$env:FASTDLLM_MODEL_PATH="<checkpoint>"
<venv>\Scripts\python.exe -m scripts.ar041_remask_fallback_probe --thresholds 0.5,0.3,0.2,0.1
```

参考主机耗时：A 臂约 107–192 s/prompt（CPU 争用下波动），D 约 10–15 s，R 约 77–134 s。
