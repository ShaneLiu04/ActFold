# [AR035] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR035 |
| AR 主题 | 置信度门控投机提交（quality-preserving multi-block speculative commit） |
| 关联 SR | README 当前限制第 14 条（AR022/031/032/033/034 共同边界：投机整块并行提交在 D4 多块长生成上塌缩，准确率 0.335→0.005–0.02）；AR034（KV-cached speculative block decoding，整块一次性提交，H1-H4 supported 但仅首块早停几何） |
| 日期 | 2026-10-24 |
| 状态 | Draft |

## 1. 背景与目标

AR034 的组合臂（历史 KV × 投机）把每个 small block 的 ~31 次串行前向压成
「整块草稿 + `[N, block]` 验证 + 1 次 commit」，首块早停几何下 8.73× vs 非缓存
基线且质量逐位不损。但该机制**把整块按 winner 的 draft 候选一次性落盘**，
D4 战役（GSM8K 200 题 × 512 token）已实测这一族语义在多块长生成上塌缩
（并行解码重复塌缩，准确率 0.335→0.005–0.02）。

**目标**：引入**置信度门控提交**——验证后仅提交验证概率 ≥ `threshold` 的位置，
其余位置保留 mask、进入下一轮验证；若整块无位置达标则强制提交 argmax 位置
保证推进。期望：长生成质量回到非投机路径附近，同时保留「每轮批量验证、
launch 数远低于逐迭代」的加速结构。

## 2. 需求范围

**In Scope：**
- `FastDLLMSamplerConfig.speculative_commit_gate`（默认 `False` = AR034 逐字节
  不变；仅 `history_cache=True` 且 `num_block_candidates>0` 合法）。
- 组合臂提交语义扩展：门控接受集合 + 至少一位强制推进 + 未完成时继续迭代
  （不 `break`）。
- 账本：`history_cache_stats` 增 `spec_gated_iterations`（门控轮数）。
- fp32 tiny fixture UT：门控语义、强制推进、默认关零回归、账本。
- 基准：`scripts/ar035_gated_spec_bench.py`（三臂 A / 未门控 D / 门控 G，
  多块几何），quick 面本机执行；formal 面按 RERUN 新步骤在服务器复现。
- 文档：README 新行/简节/阶梯、AGENTS 新条目、CHANGELOG、RERUN 步骤、design 收口。

**Out of Scope：** 多块 GSM8K 正式战役本机执行（按用户授权写入 RERUN 服务器节）；
跨族（LLaDA/Dream）门控。

## 3. 功能需求

### 3.1 FR1 门控提交

**描述：** 组合臂验证后，对 winner 行在活动 small block 每个仍为 mask 的位置
计算其提交 token 的验证概率 `p = softmax(vlogits)[token]`；仅 `p >= threshold`
的位置写入画布；若一个都没达标，强制提交 `p` 最大的位置（保证每轮 ≥1 位推进）。
small block 仍有 mask 时继续下一轮（调用方不 `break`）。

**验收标准：** Given fp32 tiny fixture + 门控开, When 多块贪心采样,
Then 输出 mask-free、确定性、每次验证至少推进 1 位；默认关时与 AR034 逐字节一致。

### 3.2 FR2 质量与加速（基准）

**描述：** 三臂基准（A 非缓存基线 / D 未门控 / G 门控），多块几何（长
`num_tokens`、prompt 不早停），判定：
- H1 质量：G vs A 的 decode 长度一致 + token 一致率 ≥0.98（D 作为对照，预期
  在长生成上低于 G）；
- H2 加速：p50(A)/p50(G) > 1 严格；
- H3 机制：G 的 `spec_gated_iterations > spec_verify`（每轮至少 1 位）且
  `spec_verify < A 的前向数`；账本恒等；
- H4 诚实：canary + repro 双跑严格相等（失配全 blocked）。

**验收标准：** quick 面本机产出工件；formal 面命令/资源/判定写入 RERUN，
服务器一次复跑即可回填。

## 4. 非功能需求

- NFR1 零回归：`speculative_commit_gate=False` 时 AR034 路径逐字节不变。
- NFR2 可观测：门控轮数只读计数。
- NFR3 可复现：判定冻结、预承诺单次、quick/formal 分离。

## 5. 约束与假设

- 门控使用 winner 行的验证概率（与 Fast-dLLM 阈值解掩同源语义）。
- 首块早停几何下门控可能与未门控等价（记录）；多块质量证据以服务器为准。
- 与 AR034 相同：sb=32（贪心语义已验证），不跨 AR033 工件比较。

## 6. 术语说明

| 术语 | 说明 |
|------|------|
| 门控提交 | 只提交验证概率 ≥ threshold 的位置，其余保留 mask |
| 强制推进 | 无位置达标时提交 p 最大的位置 |
| D / G | 未门控 / 门控组合臂 |

## 7. 任务清单

见 `tasks.md`。
