# AR035 详细设计 — 置信度门控投机提交

| 组件名称 | actfold/models/fast_dllm_sampler.py（组合臂门控提交）+ tests + scripts/ar035_gated_spec_bench.py |
| --- | --- |
| AR系统流水号 | AR035 |
| AR描述 | 组合臂验证后仅提交验证概率 ≥ threshold 的掩码位（空集强制 argmax 位），其余保留 mask 下轮再验；把 Fast-dLLM 的阈值解掩规则与投机验证统一，抑制多块长生成的并行提交塌缩。 |

# 1 机制

- 配置：`speculative_commit_gate: bool = False`（默认关 = AR034 逐字节不变；
  仅 `history_cache=True` 且 `num_block_candidates>0` 合法，否则构造期
  `ValueError`）。
- `_history_speculative_small_block` 门控分支：
  1. winner `j*` 选定后，取 `p = softmax(vrows[j*])` 在候选 token 处的概率；
  2. `accept = (p >= threshold) & mask_positions`；若 `accept` 为空，取
     `p.argmax()` 位置（保证每轮 ≥1 位推进）；
  3. 仅写入 `accept` 位置；返回 `complete = (该 small block 无剩余 mask)`。
- 调用方：门控时 `if complete: break`，否则继续 `max_iters` 循环；未门控保持
  「整块提交 + break」。
- 账本：`spec_gated_iterations` 每次门控验证 +1（默认 0）。
- 语义等价性：`threshold` 越高越接近逐迭代解掩；门控下每轮仍需 1 草稿 + 1
  验证（launch 数 = 轮数 × 2 + 1 commit），轮数 ≤ small_block_size。

# 2 判定（三臂，冻结先于数据收集）

臂：**A** 非缓存基线 / **D** 未门控组合臂（AR034）/ **G** 门控组合臂。
几何：`block=32, small_block=32`，多块（`num_tokens` 取 256，prompt 选不早停
文本），reps=30。

| 判定 | 条件 |
|---|---|
| H1 质量 | `G vs A`：decode 长度一致 且 token 一致率 ≥0.98；D 同面 report-only（预期长生成更低） |
| H2 加速 | `p50(A)/p50(G) > 1.0` 严格；D/G 比值 report-only |
| H3 机制 | `G.spec_gated_iterations == G.spec_verify_forwards`、`> spec_draft` 的块数 ≥1；`G` 总前向数 < `A` 总前向数；账本（`spec_draft==spec_verify`、prefill=1、degraded 0）恒等 |
| H4 诚实 | canary（G armed、degraded 0、拓扑冻结）+ repro `(arm,prompt)` 严格相等；失配 → 全部 blocked |

优先级 blocked > refuted > supported。quick 面本机；formal 面命令/资源/判定
写入 `docs/RERUN_CHECKLIST.md` 新步骤，服务器复跑回填（用户授权）。

# 3 UT（fp32 tiny fixture）

| 编号 | 覆盖 |
|---|---|
| UT-3501 | 门控默认关零回归（输出/计数==AR034 路径） |
| UT-3502 | 门控语义：输出 mask-free、确定性、每轮至少推进 1 位（`spec_gated_iterations >= 1`） |
| UT-3503 | 强制推进：`threshold=1.1`（无位置达标）时仍完成且轮数 == 掩码位数 |
| UT-3504 | 账本：`spec_draft==spec_verify`、`spec_gated_iterations==spec_verify`、degraded 0 |
| UT-3505 | 非法组合：门控无 `history_cache`/无 `num_block_candidates` → `ValueError` |

# 4 交付收口

（T005/T007 后补）


## 交付结果（诚实负面记录，冻结后不回改）

- **真实 checkpoint 冒烟**（`scripts/ar035_gated_spec_smoke.py`，64 token，τ=0.95）：
  D 臂 4.77 s / 1 轮验证；G 臂 54.93 s / **12 轮验证**。绝对阈值把「1 轮整块提交」打散成 ~12 轮，
  退化为近逐 token 解掩。
- **三臂快速基准**（`scripts/ar035_gated_spec_bench.py --quick`，reps=2，`ar035_gated_spec_bench_quick.json`）：
  H1 supported（token 级 == A、mask-free、decode [34] 三臂一致）；H3 supported
  （`gated == verify == 36`，A/D 为 0，degraded 0）；H4 supported（repro 0 mismatch）；
  **H2 refuted**（p50 A 28269.4 / D 3994.7 / G 38282.0 ms → `A/G 0.7385`、`G/D 9.5832`）。
- **结论**：AR035 按「安全/保守度旋钮」交付，不计入加速阶梯。绝对阈值不是正确形态；
  若要走加速档，需相对门控（`p >= τ·max` 或 top-m 位）——那是新设计而非阈值调参。
- 机制默认关（`speculative_commit_gate=False`）时与 AR034 字节级一致，UT-3501 钉死该回归面。
