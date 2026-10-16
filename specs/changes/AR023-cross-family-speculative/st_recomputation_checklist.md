# ST 验收重算面清单（T010 输入预备）

> ST 的验收方式 = **重算核对**，不是重跑预承诺基准（reps=30 单次定案，
> 重跑钓鱼禁令）。本清单列出每个验收数字的重算来源与算术核对式。
> 工件真值：`results/optimization/ar023_{family_probe,llada_bench,
> dream_bench}.json` + quick 工件三件。

## A. 工件 schema 有效性（两族正式工件）

- [ ] 顶层键全集：`schema=="ar023_cross_family_bench"` / `schema_version` /
      `family` / `status` / `modes` / `quality` / `verdicts{h1,h2,h4}` /
      `reproduction` / `meta`（blocked 工件另含 `blocked_reason`）
- [ ] 模式键连字符四值：`baseline` / `folded-serial` / `folded-batched` /
      `folded-batched-greedy`，各含 `wall_ms.prompt<i>.{p50_ms,mean_ms,
      std_ms,n}` 与 `decode_lengths`
- [ ] `status == "ok"`（否则走 blocked 事实链：`blocked_reason` 非空 +
      探针交叉引用存在 + 无伪造数字）

## B. H1 算术重算（每族）

- [ ] `numbers.gap` 键 = 连字符两值；`ratio` = gap["folded-batched-greedy"]
      / gap["folded-batched"]（gap_bat==0 时哨兵 0.0/inf 特例——核对
      faces.a 与哨兵语义一致）
- [ ] `faces.a_gap_shrink_ratio_le_0_5` == (ratio <= 0.5 + 1e-9) 重算一致
- [ ] `faces.b_length_recovery` == 逐 prompt `|len_aln−len_base| <=
      |len_bat−len_base|`（`per_prompt_decode_lengths` 重算一致）
- [ ] `faces.c_block_exact_strict_improvement` == `exact["folded-batched-greedy"]
      > exact["folded-batched"]`
- [ ] `verdict` == 三硬面合取；`strong` 三键非阻塞
- [ ] `none_posthoc_mean == false`（True ⇒ refuted 语义核对）

## C. H2 / H4 算术重算（每族）

- [ ] H2：`numbers.speedup_p50_vs_baseline` == p50_baseline_ms /
      p50_aligned_ms 重算（4 位舍入容差）；`verdict` == (>1.0)；
      p50 来源 = `.modes.*.wall_ms.prompt*.p50_ms` 跨 prompt 平均，抽验
      一模式与 h2.numbers.p50_baseline_ms 一致
- [ ] H4：ledger 界 `numbers.ledger_within_bound`——`write_op_count` 差分
      ≤ `4·((L+1)·serial+(L+N)·batched)`（L=族层数，serial/batched 为该
      模式 timed 段 ops；从 quality/mode 侧键重算）；`canary_degradation_zero
      == true`；`stable_ratios` 每模式一值 + `stable_ratio_mean` = 其平均

## D. 复现自检与跨工件一致性

- [ ] `.reproduction.ok == true`；成功工件**无** `reproduction_anomaly` 键
- [ ] bench `meta.dtype` == 探针 `dtype_choice`（bf16）；
      `meta.graph_enabled == false` == 探针结论；
      `meta.probe_artifact_ref` 指向探针工件存在
- [ ] `meta.seed == 2026` / `meta.reps == 30` / `len(meta.prompts) == 2` /
      `meta.family_constants.num_tokens == 128` / `meta.quick == false`
- [ ] checkpoint SHA256：bench meta vs 探针清单逐族一致
- [ ] quick 工件（三件）：`_quick` 后缀、`meta.quick == true`、reps=3/
      单 prompt/64 token——GUIDE §30 冒烟行数字（5.41×/1.35×/len 3 vs 64/
      ratios 0.491/0.391/0.391/0.397/0.340/0.569）与 quick 工件逐位核对

## E. 门记录与文档数字溯源

- [ ] T007 门记录（tasks.md）：全量 pytest ≥ **1022 passed / 0 failed /
      4 skipped**（deselect 两例 eval 后端既定面）；demo 三锚点
      85.5% / 2.35e-03 / 93.75%（T007 实测回填后核对）；lint/mypy 记录
- [ ] README 表数字逐槽对工件（readme_rewrite_skeleton.md §4 映射表——
      回填后骨架删除前完成核对）
- [ ] CHANGELOG AR023 Experiment Results 节 / GUIDE §30 实测块 /
      RERUN 第 13 步对照数字：全部与正式工件逐位一致，无凭空数字
- [ ] AGENTS #45 empirical face 数字句与工件一致

## F. ST 可执行命令面（GPU 空闲后）

- 全量 pytest canonical（含 deselect 两例）+ lint 三件 + demo 三锚点
- **禁止**：bench/probe 全量重跑（预承诺单次）；仅当工件自身标记
  `reproduction_anomaly` 或 blocked 时按其根因面另行处置（另立决策，
  不在 ST 自动重跑）
