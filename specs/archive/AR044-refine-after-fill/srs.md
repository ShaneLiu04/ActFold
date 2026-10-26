# [AR044] 需求设计说明书 — 填后精修（Refine-after-Fill, RaF）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR044 |
| 关联 SR | SR-044（质量-延迟前沿，承接 AR043 收敛结论） |
| 日期 | 2026-10-25 |
| 状态 | Draft |

## 1. 背景与缺口

AR043 收敛后只有两个点：`quality` = kv-only（1.51×，logp 逐位等于基线）、`latency` = 全栈（14.28×，
logp 在 2/3 prompt 上下降 −0.17~−0.73）。**没有 ≥5× 且质量平价的组合。**

已证伪的中间路线：AR035 门控（成本 9.58×）、AR038 升档（零收益）、AR041 回退（同质量但更慢）、
AR042 分块渐进（混沌）。共同缺陷：**一次写入后没有复核机会**（AR042 是「填之前排序」，错误一旦写入不再回看）。

## 2. 机制（RaF）

`commit_refine_passes: int = 0`（默认 0 = 关闭，与 AR043 末态逐字节一致）、`commit_refine_threshold: float = 0.5`。

```
1) 一次性贪心填满块的仍掩码位（现有 AR022 语义）
2) for pass in range(passes):
       前向整块（历史 KV 前缀，块内双向注意力 -> 模型看到完整块）
       取本轮仍掩码位（最初被掩码的位置集合）上的 token 概率 p
       把 p < refine_threshold 的位置重新填为该轮 argmax
3) 提交最终画布
```

与 AR042 的关键差别：AR042 每轮只填最高置信的 chunk 且**不回看已填位**；RaF 每轮**复核全部已填位**
并允许改写低置信位，这正是官方迭代解掩的核心行为。

## 3. 目标（冻结判定）

在 AR040 的 3 条 prompt × 160 token 上（参照 AR043 的 kv-only 与 latency 数字）：

- **H1 质量平价**：存在 `passes ∈ {1,2,3}` 使 `logp >= logp(kv-only) - 0.05` 在 **3/3** cell 成立；
- **H2 速度**：该 `passes` 的中位加速 `>= 5×`（相对基线；用 AR043 同 session 基线换算）；
- **H3 账本**：`spec_refine_forwards == passes × 块数`（回退/末块除外，逐 cell 可核）；
- **H4 复现**：该 `passes` 双跑逐位一致。

若 H1 或 H2 不成立 → 如实记录（不重跑）；若成立则给出**新的第三预设 `balanced`**（≥5× 且平价），
否则维持 AR043 的两点结论。

## 4. 功能需求

### F1 精修

**验收标准**：
- Given `passes=0`，Then 与 AR043 末态逐字节一致；
- Given `passes=k>0`，Then 每块恰好 k 次精修前向，`spec_refine_forwards` 精确 +k；
- Given 精修后无低置信位，Then 仍执行完 k 次（不改语义，保证账本可预测）；
- Given 任一位置被精修改写，Then 最终提交内容包含改写结果。

### F2 与 CDF/VFG 组合

**验收标准**：Given 精修开启，Then 该块不设 CDF advance 交接（提交走非融合路径），输出正确、`degraded==0`。

### F3 账本

**描述**：`history_cache_stats` 新增 `spec_refine_forwards`；默认 0。

## 5. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 平价 | 3/3 cell 在 0.05 容差内 |
| 速度 | 加速 | ≥5× |
| 兼容 | 默认关 | 逐字节零回归 |

## 6. 约束与假设

真实 checkpoint、无 mock、`stop_token_id=-1`；假设 A1：块内完整上下文能纠正一次性填写的错误
（由 H1 检验，可能被证伪）。
