# [AR039] 需求设计说明书 — 免费验证 + 无验证贪心提交（FV/VFG）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR039 |
| AR 主题 | 免费验证（Free Verification）+ 无验证贪心提交（Verification-Free Greedy Commit） |
| 关联 SR | SR-039（自适应算力主线，承接 AR037 ACB / AR038 级联负面界） |
| 日期 | 2026-10-25 |
| 状态 | Draft |

## 1. 背景与目标

### 1.1 实测事实（全部来自真实 checkpoint）

| 事实 | 来源 | 数值 |
|---|---|---|
| 长几何加速 | AR037 | A 172.4 s → ACB+CDF 11.84 s（14.6×） |
| 候选数扩展无质量收益 | AR038 | N=4 / N=8 的 mean log p 均为 −2.0870（逐位一致） |
| 验证的实际职责 | AR034 T007 + AR038 | 贪心行在 `sb=block` 下总是胜者；`[N,B]` 验证从未改变结果 |

### 1.2 结构性机会

CDF 的融合前向输入是 `[block_b | next_canvas]`，它**重新读取了刚提交的块 b**。
其决策行 `aligned[:, 0:B]` 给出的 softmax 概率，就是对已提交 token 的一次**免费复评**
（不需要任何额外前向、不增加 KV 写入）。由此得到两个机制：

1. **Free Verification（FV）**：每块提交后用融合 logits 复查，落账本 `free_verify_low_conf`；
2. **VFG（无验证贪心提交）**：既然验证从未改变结果，置信达标时直接整块提交贪心草稿、
   **跳过 `[N,B]` 验证**（CDF 下 = 1 次前向/块）；用 FV 的结果做**安全再武装**
   （上一块 FV 发现低置信 → 当前块强制走完整验证）。

### 1.3 目标（冻结判定）

- **H1 质量**：VFG 输出与「ACB+CDF」**逐位一致**（token-identical）；
- **H2 速度**：`p50(VFG) < p50(ACB+CDF)`，并发布相对非缓存基线的加速；
- **H3 成本/账本**：VFG 下 `spec_verify_forwards == spec_verification_free_blocks 中的再武装次数`
  且 `free_verify` 每块恰好 1 次；FV 零额外前向（`forward_tokens` 不因 FV 增加）；
- **H4 复现**：双跑逐位一致。
- 任一 H 失败 → 如实交付（不改判定、不重跑凑数）。

### 1.4 非目标

- 不改提交内容（仍是贪心行 / min-prob 胜者）；
- 不做修订/重掩码（SCR 是另一条线，见 AR038 结论）；
- 不改 `[N,B]` 验证实现本身；不做 batch>1；不涉 LLaDA/Dream。

## 2. 需求范围

**In Scope**：`verification_free_commit`（默认关）+ `free_verify_floor`；FV 账本；安全再武装；
UT；长几何探针（A / ACB+CDF / VFG 三臂）；文档与复现指南。

**Out of Scope**：OpenAI 外部服务、批处理、跨模型族、修订语义。

## 3. 功能需求

### F1 无验证提交

**描述**：VFG 开启且未再武装时，草稿决策行直接整块写入，不构建候选画布、不发起 `[N,B]` 验证。

**触发条件**：`verification_free_commit=True`、组合臂 + CDF 生效、`num_masked > 0`、本块未再武装。

**异常处理**：未启用 CDF → 构造期 `ValueError`（1 次前向/块的承诺依赖融合）；与门控互斥。

**验收标准**：
- Given 未再武装，When 该块处理，Then 无验证前向，`spec_verification_free_blocks += 1`；
- Given 已再武装，Then 走完整验证路径，再武装标志清零，`spec_verify_forwards += 1`。

### F2 免费验证

**描述**：CDF 融合前向的 `aligned[:, 0:B]` 复查刚提交块，记录最小概率是否低于 `free_verify_floor`。

**验收标准**：
- Given 融合前向发生，Then `free_verify_blocks += 1`；若该块最小已提交概率 < floor，则 `free_verify_low_conf += 1` 并置再武装；
- Given 融合未发生（末块/剩余不足），Then 不计数、不置位（不得凭空计算）。

### F3 账本

**描述**：`history_cache_stats` 新增 `spec_verification_free_blocks`、`free_verify_blocks`、`free_verify_low_conf`；
默认关时三者均为 0。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 输出一致性 | VFG 与 ACB+CDF 逐位一致 |
| 性能 | 每块前向数 | 未再武装 = 1（融合），再武装 = 3 |
| 开销 | FV | 零额外前向（仅读已有 logits） |
| 兼容 | 默认关 | 与 AR038 末态逐字节一致 |

## 5. 约束与假设

- 真实 checkpoint、无 mock；判定冻结；单次运行；长几何必须 `stop_token_id=-1`；
- 假设 A1：融合前向对已提交块给出与提交前向一致的决策行（AR036 P1 已证 maxdiff 0.0）；
- 假设 A2：贪心行在本 workload 总是胜者（AR034 T007 / AR038 实测），故跳过验证不改输出。

## 6. 术语

| 术语 | 定义 |
|------|------|
| 免费验证（FV） | 复用融合前向的 logits 复查已提交块，不额外前向 |
| 再武装（re-arm） | FV 发现低置信后，下一块强制走完整验证 |
| VFG | 未再武装时跳过 `[N,B]` 验证、直接整块提交贪心草稿 |
