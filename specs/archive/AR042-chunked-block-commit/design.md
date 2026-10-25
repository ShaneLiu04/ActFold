# [AR042] 详细设计 — 分块提交（CBC）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR042 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-25 |
| 状态 | Delivered（H1 被证伪：块内渐进提交是混沌的） |

## 1. 机制

commit_chunk_size: int = 0（默认 0 = 关闭，与 AR041 末态逐字节一致；需 history_cache + N>0；负值 ValueError）。

开启后在组合臂内走**渐进解掩**路径（在候选/验证之前返回）：

`
fill = 当前块画布
remaining = 仍掩码位
loop:
    probs = softmax(当前决策行)
    p = probs[remaining, argmax]
    取 p 最大的 chunk 位填入 argmax token
    remaining -= chosen
    if remaining 空: break
    重新前向该块（历史 KV 前缀）-> 新决策行；spec_chunk_forwards += 1
`

每块前向数 ≈ ceil(掩码数 / chunk)；chunk >= 掩码数 退化为一次性提交（≈VFG，跳过验证），
chunk=1 近似逐位置解掩。渐进块不设 CDF advance 交接（其提交走非融合路径，少一次融合机会）。

## 2. 实测（真实 checkpoint，160 token × 3 prompts，chunk ∈ {4,8,16,32}）

| prompt | A | D | C4 | C8 | C16 | C32 |
|---|---|---|---|---|---|---|
| p1 logp | −1.1528 | −1.3118 | −2.1557 | **−0.6534** | **−0.6534** | −0.7859 |
| p2 logp | −0.8482 | −0.9904 | −0.5807 | −1.8783 | **−0.3138** | −2.1406 |
| p3 logp | −0.8086 | −1.5370 | **−0.8320** | −1.2467 | −2.8313 | −2.3622 |
| p1 墙钟 | 134.8 s | 8.1 s | 23.5 s | 13.1 s | 7.7 s | 5.2 s |
| p2 墙钟 | 87.2 s | 8.3 s | 23.4 s | 12.9 s | 7.9 s | 5.2 s |
| p3 墙钟 | 104.0 s | 8.1 s | 23.1 s | 12.8 s | 7.9 s | 5.3 s |

## 3. 判读

- **H1 单调性：refuted（3/3）**。chunk 与质量无单调关系；同一 chunk 跨 prompt 可从 −0.31 跳到 −2.14。
- **H2 存在平价点：per-prompt 成立但不可迁移**（p1 需 C8/C16、p2 需 C16、p3 需 C4），且相邻 chunk 可能灾难性变差。
- **H3 成本可核**：spec_chunk_forwards 精确等于「多轮重新前向」次数（C4=35、C8=15、C16=5、
  C32=0 与 ceil(掩码/chunk) 语义一致）。
- **交叉验证**：C32（一次性、跳过验证）的 logp 与 AR039/AR040 的 V 臂一致（p1 −0.786 vs −0.796），
  说明实现与既有测量自洽。

## 4. 结论

块内渐进**贪心**提交不是可靠的质量旋钮：它把「填哪些位」变成轨迹分叉点，质量对 chunk 高度敏感。
唯一稳健的质量安全模式仍是 AR041 的回退档（整块交还官方阈值迭代，质量逐位等于基线）。
保留 commit_chunk_size 为**默认关的消融能力**，并在文档中明确「不要用它调质量」。

## 5. 测试

| ID | 面 |
|---|---|
| 4201 | 默认 0：spec_chunk_forwards == 0 |
| 4202 | chunk=1：有 chunk 前向、无 verify、mask-free、degraded 0 |
| 4203 | chunk ≥ 掩码数：与一次性提交逐位一致 |
| 4204 | 确定性（双跑逐位一致） |
| 4205 | 非法值（负值、无 history/N）→ ValueError |
