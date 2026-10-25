# [AR036] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR036 |
| AR 主题 | 提交-草稿融合（Commit-Draft Fusion，CDF） |
| 关联 SR | SR-036（延续 AR021→AR035 投机解码主线） |
| 日期 | 2026-10-25 |
| 状态 | Draft |

## 1. 背景与目标

### 1.1 现状（实测基线）

- AR033 交付 block-causal 精确历史 KV；AR034 组合它做批量投机验证（`history_cache=True` + `num_block_candidates>0`，`sb=block=32`），
  formal 实测 **8.7308× vs 非缓存基线**、D 臂 p50 3218.6ms。
- 该组合臂每个 block 固定 **3 次前向**：① `[1,B]` 草稿 → ② `[N,B]` 批量验证 → ③ `[1,B]` 提交（写 KV + 产出 advance token）。
  60 个 block 即 180 次 launch，launch 密度仍是主要成本项。

### 1.2 结构性机会（本次可行性推演的核心结论）

1. **块推进 token 依赖**：第 b+1 块画布的第 0 位是第 b 块提交前向尾行的 argmax（advance token）。
2. **块内双向注意力**：块内任一位置依赖块内第 0 位，因此「把第 b+2 块也塞进一次前向」的深度 2 一体化前瞻**结构上不可行**
   （第 0 位未知 → 整块验证失效）。该结论记录为设计负面发现，避免后续重复尝试。
3. **可融合的是「提交(b) + 草稿(b+1)」**：verify 的 row-0 尾部 logits 与提交前向尾行输入等价（AR032 已证），
   故第 b+1 块画布在**提交前**即可构造；而块间因果注意力允许一次 `[1, 2B]` 前向同时：写第 b 块的 KV、
   并计算第 b+1 块全部决策行。把融合前向写出的第 b+1 块（仍为掩码画布）KV **裁掉**即可让历史保持精确。

### 1.3 目标

在**输出质量不变**前提下，把组合臂每块前向数由 3 降到 **2**（1.5× launch 降幅），
与 AR034 相比 p50 提速 **≥1.15×**（H2 预注册），并保持与非缓存基线 token 级一致。

### 1.4 非目标

- 不做深度 2 前瞻（结构不可行，见 1.2-2）；
- 不改 LLaDA/Dream（双向全序列掩码，机制域不同）；
- 不引入外部草稿模型；不做逐位门控提交（AR035 已证伪）；
- 不改官方 block 语义（advance token 约定、块边界、掩码调度）。

## 2. 需求范围

**In Scope：**
- `FastDLLMSamplerConfig.commit_draft_fusion`（默认 `False` = AR035 末态字节级不变）；
- 组合臂内：融合前向、KV 裁剪、草稿行暂存与消费、末块/剩余不足时自动退化；
- 账本：`spec_fused_forwards`（仅融合路径非零）；
- UT-3601~3606、三臂真实基准、README/AGENTS/CHANGELOG/ALGORITHM/EXPERIMENTS/RERUN 文档。

**Out of Scope：**
- batch>1、跨样本；
- 融合深度 >1（即一次写多块 KV）；
- 折叠路径（`folded_model`）上的融合（组合臂本就禁用折叠）。

## 3. 功能需求

### F1 融合前向（提交 + 下一块草稿）

**描述**：把第 b 块的提交前向扩展为 `[1, 2B]`，一次完成「第 b 块 KV 写入」与「第 b+1 块决策行计算」。

**触发条件**：`commit_draft_fusion=True`、组合臂生效、当前块为**完整块**（`active_len == block_size`）、
且提交后剩余长度 ≥ `block_size + 1`（够一个完整下一块 + advance）。

**期望行为**：以 `past=kv` 前向 `cat([block_b, next_canvas])`（`next_canvas = [advance_token, mask×(B-1)]`，
advance 取自本次 verify 的 row-0 尾行 argmax），`update_past_key_values=True`；
取前 B 段写回 KV（裁剪），第 B..2B 段作为第 b+1 块草稿行暂存。

**异常处理**：
- 缓存对象不支持裁剪 → 退回非融合路径并**只记一次** `RuntimeWarning`（不静默错误、不破坏正确性）；
- 剩余不足 → 走原提交路径，不计入融合账本。

**验收标准**：
- Given 融合开启且剩余足够，When 提交第 b 块，Then 仅 1 次前向，`spec_fused_forwards += 1`，`spec_draft_forwards` 不增；
- Given 缓存不支持裁剪，When 融合开启，Then 输出与融合关闭逐字节一致，且仅告警一次。

### F2 KV 裁剪精确性

**描述**：融合前向会额外写入下一块掩码画布的 KV，必须裁掉，历史 KV 必须与「单独提交前向」逐层一致。

**验收标准**：Given 同一画布，When 分别走融合前向（裁剪后）与单独提交前向，Then 每层 K/V 的 maxdiff = 0（探针 P3 定义）。

### F3 草稿消费

**描述**：第 b+1 块的小块循环直接使用暂存草稿行，不再发起草稿前向。

**验收标准**：
- Given 存在暂存草稿，When 进入第 b+1 块，Then 不发生草稿前向，且候选构造（AR022 冻结语义）与单独草稿逐字节一致（探针 P2 的 argmax 全等是前提）；
- Given 暂存草稿所属块号不匹配，When 消费，Then 丢弃并走正常草稿（防御性校验）。

### F4 质量不变

**描述**：融合不改变任何提交语义，输出必须与非缓存基线 token 级一致（AR034 口径）。

**验收标准**：Given 融合开启，When 生成，Then token 级 == 非缓存基线，decode 长度相同，mask-free。

### F5 账本

**描述**：`history_cache_stats` 新增 `spec_fused_forwards`；默认关时为 0。

**验收标准**：Given 默认关，When 任意运行，Then `spec_fused_forwards == 0`（UT-3601）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 每块前向数 | 融合生效时 = 2（对照 AR034 = 3） |
| 性能 | p50 | vs AR034 组合臂 ≥ 1.15×（H2） |
| 质量 | 输出对齐 | token 级 vs 非缓存基线 = 1.0 |
| 可靠性 | 回退 | 裁剪不可用 / 剩余不足 / 块号不匹配 → 与关闭态逐字节一致 |
| 兼容 | 默认关 | 与 AR035 末态字节级一致 |

## 5. 约束与假设

**约束：**
- 仅 `history_cache=True` + `num_block_candidates>0` + `small_block_size == block_size` + `folded_model is None`；
- 与 `speculative_commit_gate` 互斥（门控改变提交集，融合假定整块提交）；
- 真实 checkpoint、无 mock；判定函数冻结、单次运行。

**假设：**
- A1：融合前向的第 b+1 块决策行与独立草稿前向数值一致（bf16 sdpa 可能有 ULP 级漂移，探针 P2 记录 maxdiff 与 argmax 一致率）；
- A2：`DynamicCache` 支持裁剪（P4 探测）；不支持则按 F1 回退；
- A3：融合前向后裁掉尾部，历史 KV 与单独提交逐层 byte-equal（P3）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 提交前向 | 每块结束时写 KV 并产出 advance token 的 `[1,B]` 前向 |
| 融合前向 | `[1,2B]`：提交第 b 块 KV + 计算第 b+1 块决策行 |
| KV 裁剪 | 把融合前向写出的 `[hist+B+B]` 截回 `[hist+B]` |
| advance token | 提交前向尾行 argmax，作为下一块第 0 位 |
