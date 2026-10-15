# [AR022] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR022 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-15 |

> 执行环境：Quadro RTX 5000 16GB，torch 2.5.1+cu121（CUDA True），WDDM，驱动 556.18，
> python 3.11.9，pytest 9.1.1；测试命令统一 `python -X utf8`；lm_eval 两例环境缺失
> 先在（T002 stash 基线裁定，deselect 排除留档）。工件级用例（N1-N4）直接断言
> `results/optimization/ar022_speculative_quality_bench.json`（reps=30 预承诺单次定案
> 工件，T004）；复算脚本独立于工件内嵌判定（从 mode_comparison/quality/per_shape_window/
> batch_canary/ledger 原始字段重算，防自证），写入 temp 目录
> `C:\Users\L30086~1\AppData\Local\Temp\opencode\`（仓库外）。真实 checkpoint 面用
> AR019 venv（transformers==4.53.1 + 本地 Fast-dLLM-v2-1.5B 权重，#24 版本纪律）；
> `--quick` 冒烟输出至 `results/optimization/ar022_speculative_quality_bench_quick.json`
> （脚本默认 quick 路径，untracked 不入库——T007 Nit-3 裁定延续）。

## 测试用例列表

### ST-N1：工件级 H1-H4 复算（独立重算 + 逐位比对）

**关联需求：** srs.md §3.4 FR4 验收标准（H1-H4 判定式冻结 / 字段完备 / 单次纪律 / meta 声明）
**测试类型：** 正常路径（工件级断言）
**优先级：** High

**前置条件：**
- `results/optimization/ar022_speculative_quality_bench.json` 为 reps=30 预承诺单次定案工件

**测试步骤：**
1. 独立 python 脚本 `ar022_st_n1_h_recompute.py`（temp 目录）从原始字段重算四 H 判定，
   与工件 verdicts 逐位比对（不信任工件内嵌判定，防自证）：
   - H1 三面：(a) 位置级合并一致率 = Σmatched/Σcompared（quality 明细）> 0.5；(b) 块级合并
     率（对齐 vs batched 同口径）严格为正；(c) 逐 prompt 对齐 posthoc overall_mean > batched
     同 prompt 实测值
   - H2：p50(batched) − p50(greedy) vs −2σ（σ=max(σ_greedy, σ_batched)，mode_comparison
     原始 p50/std）
   - H3：逐 prompt len_aligned == len_baseline
   - H4 四子判据：(i) 对齐模式批量窗捕获>0 且 timed 重放>0；(ii) 三折叠模式 canary
     timed==0；(iii) 账本 bound==4((L+1)·ser+(L+N)·bat) 逐字段复算且 ops≤bound；(iv) AR017
     仅触发时核验（MFF_C 0=中性）
   - speedups_p50 五值复算；hypotheses_declared == verdicts hypothesis 全文（判定式冻结）；
     meta（reps=30/seed=2026/quick=false/greedy 声明）；VRAM 峰值 <16000；whole_gen n==60

**期望结果：**
- 全部复算断言通过；四 H 重算判定与工件记录逐位一致（supported × 4）

**实际结果：** 51 项断言全 PASS（首跑 50/1：跨工件 batched_vs_baseline 对照初设容差 5e-4——
AR022 记录 2.381 由本工件自身 p50 1300.6/3096.8 复算**精确成立**，与 AR021 的 2.3795 系两次
独立物理运行（AR021 p50 1193.9/2840.7），相对差 0.063% 属墙钟噪声带，容差修正 <1% 后全绿——
ST 容差错误，工件自洽，见执行注记 #1）。关键复算值：H1(a) 4/4=1.0、H1(b) 4/4 vs 0/4、H1(c)
0.369661>0.289692 / 0.570341>0.106418；H2 Δ=+859.68ms ≥ −1616.74ms（σ=808.3707=max(61.79,
808.37)）；H3 34==34 / 66==66；H4 账本六组 bound 复算 1324/360/3596/360/360/3132 逐字段一致；
speedups 7.0236/2.9498/2.381/2.4646/0.9661 全复算相等；declared==recorded 四全文相等；
VRAM 5413.0/5413.1/5840.1/5672.0 MB
**状态：** PASS

---

### ST-N2：双粒度质量面复算（位置级/块级/posthoc gap）

**关联需求：** srs.md §3.4 FR4 H1 判定式（双粒度 exact-match + posthoc 口径）+ §3.2 FR2 观测出口
**测试类型：** 正常路径（工件级断言）
**优先级：** High

**前置条件：**
- T004 工件 quality 明细定案

**测试步骤：**
1. 独立脚本 `ar022_st_n2n3n4_recompute.py` 从 quality 逐 prompt 明细重算合并值：
   - 位置级：窗口 `[32, 34)` / `[64, 66)` 冻结口径、各 2/2、合并 4/4=1.0；per_prompt_compared_counts
     == [2, 2]（34−32 / 66−64）；len_aligned == len_baseline == 34/66
   - 块级：对齐 2/2+2/2 → 合并 4/4 rate 1.0 vs batched 0/2+0/2 → 0/4 rate 0.0；
     overall_exact_match 对齐 1.0/1.0 vs batched 0.0/0.0
   - posthoc：逐 prompt 对齐 overall_mean − baseline overall_mean **逐位 == 0.0**
     （overall_min 亦逐位相等）；batched 面 0.289692/0.106418（AR021 先验值在场）

**期望结果：**
- 双粒度合并值从明细逐 prompt 重算与工件聚合值一致；posthoc gap 逐位 0.0

**实际结果：** 15 项断言全 PASS（位置级 4/4、块级 4/4 vs 0/4、gap 0.0+0.0 逐位、min
0.150954/0.275356 对齐侧逐位==baseline）
**状态：** PASS

---

### ST-N3：greedy_win 自洽（总量 vs per-prompt vs 跨面交叉）

**关联需求：** srs.md §3.2 FR2 验收标准（greedy_win 计数与批量调用次数自洽）
**测试类型：** 正常路径（工件级断言）
**优先级：** Medium

**测试步骤：**
1. 同脚本断言：`greedy_win_total` calls=2/wins=2/rate=1.0 == Σ per-prompt（各 1 call/1 win）；
   rate == wins/calls；per-prompt calls 与 per_shape_window greedy 模式批量 sidecar 前向
   （p32_block0_T64/p64_block0_T96 各 1）交叉自洽；账本 greedy batched_events == 1/1；
   H1.numbers.greedy_win == quality.greedy_win_total（单一数据源）

**期望结果：**
- 观测面三层（total/per-prompt/池窗/账本）交叉自洽

**实际结果：** 7 项断言全 PASS（2/2/1.0；sidecar 批量前向 1+1；账本 batched_events 1/1；
H1 numbers 与 quality 单源一致）
**状态：** PASS

---

### ST-N4：复现自检面复算（八字段逐位 + 先验独立溯源）

**关联需求：** srs.md §3.4 FR4 验收标准（复现性自检面：同 seed 应复现 AR021 先验，失配阻断）
**测试类型：** 正常路径（工件级断言）
**优先级：** High

**测试步骤：**
1. 同脚本断言：`reproduction_selfcheck` expected vs actual 八字段（2 prompt ×
   block_exact_match/len_other/posthoc_mean/posthoc_min）**逐位相等** + field_matches 全真 +
   strict/pass/gating 标志；顶层 `reproduction_anomaly` 为 null
2. 防自证加强：expected 先验值**独立溯源至 AR021 工件**
   （`ar021_generation_loop_bench.json` 的 quality.h3_batched_vs_baseline 原始字段：0.0/160/
   0.289692/5e-06 与 0.0/77/0.106418/0.0）；actual == 本次运行 batched 原始质量字段

**期望结果：**
- 八字段逐位相等；先验溯源与本次实测双侧闭环；无 reproduction_anomaly

**实际结果：** 26 项断言全 PASS（八字段逐位相等 5e-06 浮点亦精确；expected 八值逐位==AR021
工件原始字段；actual 八值逐位==本次 raw batched 面；anomaly null）
**状态：** PASS

---

### ST-N5：生产功能面（UT-2201~2207 全套 + 冻结面 UT-2108~2110）

**关联需求：** srs.md §3.1 FR1 + §3.2 FR2 + §3.3 FR3 验收标准（合成面全链）
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\test_speculative_greedy_alignment.py -v`
2. `python -X utf8 -m pytest tests\test_generation_loop_folding.py -k "ut2108 or ut2109 or ut2110" -v`
   （AR021 冻结面零改动）

**期望结果：**
- UT-2201（默认零扰动四面 + mask-token argmax 边界 + 观测面不触达）、UT-2202（贪心行构建/
  行 1..3 parity/generator 计数 N−1/零回读 spy）、UT-2203 双函数（贪心可落败后验陷阱 + 胜出
  提交）、UT-2204（观测面自洽/只读/不可赋值）、UT-2205（停止/链延续两分支）、UT-2206（True+N=0
  ValueError + YAML 零 UserWarning）、UT-2207（N=1 纯贪心）全绿；UT-2108/2109/2109c/2110
  冻结面零改动通过

**实际结果：** 8 passed（7.18s）+ 6 passed, 17 deselected（7.06s）——CUDA 实跑非 skip
**状态：** PASS

---

### ST-N6：`--quick` 真实 checkpoint 冒烟（四模式 #10 真实无 mock）

**关联需求：** srs.md §3.4 FR4 验收标准（`--quick` ≤5min 且产出合法工件）
**测试类型：** 正常路径（真实 checkpoint 面）
**优先级：** High

**前置条件：**
- AR019 venv + 本地 Fast-dLLM-v2-1.5B 权重可用

**测试步骤：**
1. `ar019_venv\Scripts\python.exe -X utf8 -m scripts.ar022_speculative_quality_bench --hf-home <ar019_hf> --model-path <Fast_dLLM_v2_1.5B> --quick`
2. 断言：四模式跑通 + 对齐面质量四列非空（block exact / position rate / len / posthoc）+
   退出码 0

**期望结果：**
- 四模式全部跑通 ≤5min；对齐面质量四列非空；退出码 0（H 判定 quick 期非定案）

**实际结果：** 完成（warmup ×4 + 3 reps round-robin，全程 ≪5min）：四模式 p50 量级 2896.9/
2997.8/768.8/371.7ms（rep 终值）；对齐面 block exact 1.0 / position rate 1.0 / len 34 /
posthoc 0.369661、greedy_win 1/1、batched exact 0.0（len 96=32+64，quick num_tokens=64
结构性不可比如实注记）；H1-H4 quick 全 supported（非定案）；复现自检 quick 降级结构性验证；
退出码 0（独立复跑捕获）；输出至默认 quick 路径（untracked）；Triton merge kernel 走
PyTorch 回退（#10 生产支持路径，环境既有事实：无 C 编译器）
**状态：** PASS

---

### ST-B1：N=1 纯贪心退化（合法边界）

**关联需求：** srs.md §3.1 FR1 验收标准（greedy on + N=1 单行纯贪心合法且链延续正确）
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. 引用 N5 执行结果中的 `test_ut2207_n1_pure_greedy`（已钉：`[1,T]` 单行批量调用、行 0 ==
   argmax 填充、无随机行（generator spy 计数 0）、平凡胜出 j*==0、链延续 `_folding_parent ==
   _cand0`、greedy_win_stats calls==wins==小块数）

**期望结果：**
- N=1 退化显式合法非报错

**实际结果：** UT-2207 1 passed（N5 执行记录）
**状态：** PASS

---

### ST-B2：mask-token argmax 边界（无特殊处理面）

**关联需求：** srs.md §3.1 FR1 期望行为（draft argmax 位为 mask token 时与 baseline argmax 家族一致，无特殊处理）
**测试类型：** 边界条件
**优先级：** Medium

**测试步骤：**
1. 引用 N5 执行结果中的 `test_ut2201_default_off_zero_disturbance`（含 (f)：constant-mask
   head（argmax ≡ mask_id）下 greedy on 与 baseline 双路无异常完成）

**期望结果：**
- mask-token argmax 双路（greedy on / baseline temperature=0 argmax）行为家族一致，无异常

**实际结果：** UT-2201 1 passed（含 (f) 边界断言面）
**状态：** PASS

---

### ST-B3：画布一致区前提（行 1..N−1 parity 的适用范围）

**关联需求：** srs.md §3.1 FR1 N 语义定案 + design §4.2.2(2)/Minor-1（行 1..N−1 逐位相同仅在画布一致区 = 首个投机小块内构造成立，分叉后失效——A/B 是构建算子对照面非全程轨迹等价）
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. 引用 N5 执行结果中的 `test_ut2202_greedy_row_construction`（fixture 钉死
   `num_tokens ≤ block_size` 单小块——多小块下提交分叉使后续小块行 1..N−1 失去逐位等价，
   断言只对首块/画布一致区有效；UT 对拍 greedy-off 同 seed 行 1..3 逐位相同如实落在该前提内）

**期望结果：**
- parity 断言在画布一致区内成立；多小块分叉行为如实（不夸大为全程等价）

**实际结果：** UT-2202 1 passed（单小块 fixture 内行 1..3 对拍逐位相同 + 前提钉死）
**状态：** PASS

---

### ST-B4：True+N=0 构造期 ValueError（不静默降级）

**关联需求：** srs.md §3.3 FR3 验收标准（`candidate_include_greedy=True, num_block_candidates=0` → ValueError）
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. 引用 N5 执行结果中的 `test_ut2206_config_validation_and_yaml`（冻结文案断言）
2. E1 现场脚本构造期面独立复验（layer-1）

**期望结果：**
- 构造期 `ValueError`（"candidate_include_greedy=True requires num_block_candidates"），不静默

**实际结果：** UT-2206 1 passed + E1 live：`ValueError('candidate_include_greedy=True requires
num_block_candidates ...')` 命中冻结文案；True+N=4 构造合法
**状态：** PASS

---

### ST-B5：greedy_win_rate 1.0 口径边界（观测面非普适断言，三处文档一致）

**关联需求：** srs.md §3.2 FR2（贪心可落败，A2 假设：胜出率非构造保证）+ §3.4（greedy_win_rate 为机理报告面）
**测试类型：** 边界条件
**优先级：** Medium

**测试步骤：**
1. UT-2203 引用（N5 执行记录：后验陷阱 fixture 构造级证明贪心行可输——j*==1、greedy_wins==0）
2. 文档三处口径一致核：README / docs/OPTIMIZATION_GUIDE.md / AGENTS.md #44 对
   greedy_win_rate 1.0 的表述必须一致限定为"本负载观测值"且"机制可落败（UT-2203）"，不得表述为普适断言

**期望结果：**
- UT-2203 落败面绿；三处文档口径一致（观测面 vs 断言面区分明确）

**实际结果：** UT-2203 双函数 passed；三处一致——README（L299/L355-356："greedy_win_rate
1.0（2/2）是本负载的观测值""贪心可落败是机制属性（UT-2203），win_rate 是观测面"）、GUIDE
第二十九部分（L784："贪心全胜不构成普适结论，机制上贪心可落败（UT-2203），win_rate 是观测面
不是断言面"）、AGENTS #44（"CAN LOSE (UT-2203's posterior-trap fixture ... never
special-case row 0 in selection)" + "greedy_win_rate is derived by the benchmark from
before/after snapshots"）；附加现场佐证：E1 control 运行 mini fixture 上 greedy_wins=0/
speculative_calls=2（贪心行实输两局）——可落败非纸面
**状态：** PASS

---

### ST-E1：folded_model 缺失/不支持 branch_ids + greedy on → 既有 ValueError 叠加

**关联需求：** srs.md §3.3 FR3 期望行为（True 的前置校验叠加：N>0 既有 folded_model/branch_ids 前置原样生效）+ §3.4 设计 §6.4
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 现场脚本 `ar022_st_e1_preconditions.py`（temp 目录，复用 UT fixture）两层验证：
   - 构造层：True+N=0 → ValueError（冻结文案）；True+N=4 构造合法
   - sample 层：greedy on + N=4 + 无 folded_model → ValueError；greedy on + N=4 +
     folded_model 不支持 branch_ids（object()）→ ValueError；greedy off 同面错误文案
     逐字相同（既有路径未改）；对照：branch_ids-capable MFF + greedy on 正常运行且观测面递增

**期望结果：**
- 两层 ValueError 全命中且消息指明期望；greedy on 不改既有前置行为；capable 路径正常运行

**实际结果：** 7 项断言全 PASS（构造层 ValueError 冻结文案；sample 层两面 ValueError 命中
"requires a folded_model for the..."/"requires a folded_model that supports the
branch_ids..."；greedy off 同文案；control：speculative_calls {0,0}→{0,2} 且 sample 无异常
完成）。（脚本自身两处修正：sample 返回 SamplerOutput 非 tensor——脚本断言面修正，产品无涉）
**状态：** PASS

---

### ST-E2：YAML 未知键对照（candidate_include_greedy 零 UserWarning）

**关联需求：** srs.md §3.3 FR3 验收标准（YAML 含键 → 程序化消费，无未知键告警路径）+ AGENTS #5
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 引用 UT-2206（N5 执行记录：`load_config` 载入含键 YAML 零 UserWarning + 字段 True + 缺省 False）
2. 现场对照脚本 `ar022_st_e2_yaml_face.py`：含 `candidate_include_greedy` 的 YAML 零
   UserWarning 且字段被消费；**真未知键对照面**（`totally_unknown_key_xyz` → 未知键
   UserWarning 如常触发）；无键缺省 False

**期望结果：**
- 新键走消费路径零告警；未知键告警路径未被破坏（对照面仍在）

**实际结果：** UT-2206 1 passed + 现场三面全 PASS（零告警/字段 True；真未知键
'Ignoring unknown config keys...' 如常告警；缺省 False）
**状态：** PASS

---

### ST-E3：复现自检失配路径（blocked 语义为设计定案，无实跑面）

**关联需求：** srs.md §3.4 FR4 异常处理（复现不符 → reproduction_anomaly + H 判定 blocked，不静默取先验）
**测试类型：** 异常处理
**优先级：** Medium

**测试步骤：**
1. 核工件字段存在性与文档一致性（不伪造失配——定案工件为 strict PASS 路径，失配分支为
   设计定案无实跑面）：
   - 顶层 `reproduction_anomaly` 键存在（值 null）
   - `reproduction_selfcheck` 含 `gating: true`（失配即阻断的载体标志）
   - `meta.reproduction_selfcheck_declaration` 含 blocked 语义全文（"ALL FOUR H verdicts
     are set to blocked"+ quick 降级结构性验证注记）
   - srs §3.4 异常处理 / design §6.4 措辞与上述一致

**期望结果：**
- 失配路径的工件载体（字段/标志/声明）与文档三面一致存在

**实际结果：** 全在场：`reproduction_anomaly` 键存在且 null（strict PASS 下正确）；gating=true；
declaration 尾部 "...recorded blocked honestly; under --quick the face degrades to a
structural validation ... and does not gate"——与 srs/design 措辞一致；本次运行 strict PASS
（N4 八字段逐位）故无 anomaly 记录（如实路径）
**状态：** PASS

---

### ST-R1：全量回归（951 基线 + 8 新增）

**关联需求：** srs.md §4 非功能需求（回归）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\ -q --deselect tests/test_eval.py::test_benchmark_runner_real_judges --deselect tests/test_judges.py::test_lm_eval_judge_prompts_and_score`

**期望结果：**
- 951 passed / 0 failed（943 AR021 基线 + 8 AR022 新增）；环境性 skip（CUPTI null #39 +
  evalplus Unix-only #6）

**实际结果：** **951 passed, 4 skipped, 2 deselected, 40 warnings, 54.48s**——passed/failed
与冻结基线（951/0）精确一致；4 skip 全环境性（1 CUPTI LIBKINETO_NOCUPTI + 3 evalplus
Unix 'resource' module，`-rs` 逐条确认）；T005 记录注记"3 skipped（1 CUPTI+2 evalplus）"
系**记录侧计数笔误**（test_judges.py 的 evalplus 条件 skip 实为 3 处 :47/:72/:86，文件
AR022 零改动——git status 证实），passed/failed 面零影响，见执行注记 #3
**状态：** PASS

---

### ST-R2：AR021 零改动面（脚本 + 工件）

**关联需求：** srs.md §5 约束（AR021 复用为导入式，ar021 脚本本体零改动）+ §2 口径延续
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `git diff scripts/ar021_generation_loop_bench.py`（预期空）
2. `git status`（AR021 正式工件 tracked 未动；quick 工件 untracked 不入库）

**期望结果：**
- AR021 脚本零改动；AR021 正式工件未被触碰

**实际结果：** diff 空（exit 0）；`results/optimization/ar021_generation_loop_bench.json`
tracked 无修改（status 无输出）；工作树与 ST 会话起点一致（仅 AR022 交付面 8 项变更 +
untracked 新文件；ar022 quick 工件为本次 N6 复跑覆写，untracked 不入库——允许面）
**状态：** PASS

---

### ST-R3：demo 三锚点

**关联需求：** srs.md §4 非功能需求（语义：demo 三指标锚点不变）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -X utf8 demo.py`

**期望结果：**
- 85.5% / 2.35e-03 / 93.75% 三锚点精确不变

**实际结果：** "Total FLOPs reduction: 85.5% / Output equivalence (MSE): 2.35e-03 [HIGH] /
Estimated stable token ratio: 93.75%"
**状态：** PASS

---

### ST-R4：静态质量门（AR022 改动文件面）

**关联需求：** srs.md §4 非功能需求（静态：mypy / pyflakes / 100 列）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -m pyflakes actfold/models/fast_dllm_sampler.py actfold/utils/config_manager.py scripts/ar022_speculative_quality_bench.py tests/test_speculative_greedy_alignment.py`
2. `python -m mypy actfold --ignore-missing-imports`
3. `python -m black --check --line-length 100`（AR022 四个代码改动/新增文件）

**期望结果：**
- pyflakes 零输出；mypy clean；四文件 black 100 列全过

**实际结果：** pyflakes exit 0（零输出）；mypy "Success: no issues found in 67 source
files"；black "4 files would be left unchanged"
**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-15 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮（一次成 Go） |

### 执行注记（ST 用例侧修正/记录侧勘误，非产品缺陷）

| # | 现象 | 裁定 |
|---|------|------|
| 1 | ST-N1 初版以 5e-4 容差对拍跨工件 batched_vs_baseline（AR022 2.381 vs AR021 2.3795）——两值各自由**各自工件自身** p50 复算精确成立（AR022 内部 speedups 51 项中逐值复算相等），两者差异源于两次独立物理运行的 p50 噪声（1300.6/3096.8 vs 1193.9/2840.7，相对差 0.063%） | ST 容差错误：跨工件对照应为噪声带（<1%）而非冻结值逐位；修正后全绿。工件自洽非缺陷（AR021 ST 注记 #1 同型先例） |
| 2 | ST-E1 脚本两处断言面修正（`sample()` 返回 `SamplerOutput` dataclass 非 tensor——输出形态断言改类型名；另移除冗余形状断言） | 脚本自身问题，产品行为无涉；修正后 7 项断言全过 |
| 3 | R1 实测 4 skipped（1 CUPTI + 3 evalplus-Unix，`-rs` 逐条：test_ar002_perf_proxy.py:351 / test_judges.py:47/:72/:86）vs T005 记录注记"3 skipped（1 CUPTI+2 evalplus）"——test_judges.py AR022 零改动（git status 证实），evalplus 条件 skip 实为 3 处 | T005 **记录侧计数笔误**；passed(951)/failed(0) 与冻结基线精确一致，skip 全环境性，零产品影响 |

三处均为 ST/记录侧修正，未发现产品缺陷；全部断言通过且多面交叉自洽。

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR1 | 贪心锚定候选（行 0 argmax/其余候选不变/N 语义定案/N=1 退化/mask-token 边界） | ST-N1, ST-N2, ST-N5, ST-B1, ST-B2, ST-B3 | PASS |
| §3.2 FR2 | 选定与质量观测（min-prob 不变可落败/greedy_win 观测面/双粒度出口/链停止语义） | ST-N2, ST-N3, ST-N5, ST-B5 | PASS |
| §3.3 FR3 | 零扰动与配置（默认逐字节/True+N=0 ValueError/YAML 消费） | ST-N5, ST-B4, ST-E1, ST-E2 | PASS |
| §3.4 FR4 | 四模式基准与工件（H1-H4/字段完备/单次纪律/复现自检/--quick） | ST-N1, ST-N2, ST-N3, ST-N4, ST-N6, ST-E3 | PASS |
| §3.5 FR5 | 文档收口五件 | （T006 交付 + T007 review E/F 维度逐位 VERIFIED；ST 层抽查：B5 三处口径一致核 + N1 README 头条数字工件溯源复算——speedups/质量面/VRAM 全部逐值复算相等） | PASS |
| §4 NFR | 回归/静态/语义基线/资源 | ST-R1, ST-R2, ST-R3, ST-R4, ST-N1（VRAM 四模式 <16000）, ST-N6（quick 冒烟资源面） | PASS |

**需求覆盖率：** 6 / 6（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 6 | 6 | 0 | 0 |
| 边界条件 | 5 | 5 | 0 | 0 |
| 异常处理 | 3 | 3 | 0 | 0 |
| 回归测试 | 4 | 4 | 0 | 0 |
| **合计** | **18** | **18** | **0** | **0** |

（执行计数：UT 粒度 N5 8+6=14 项次全绿（含 B1-B4 引用面与 B5 落败面）；工件复算脚本
N1 51 + N2/N3/N4 48 + E1 7 + E2 3 = **109 项独立断言全 PASS**；N6 真实 checkpoint quick
四模式 + 退出码 0；R1 全量 951 passed/0 failed；R2 git 零改动；R3 三锚点精确；R4
pyflakes 0/mypy 67 文件 clean/black 4 文件过。）

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| （无） | — | — |

### 结论

> **Go**。18/18 ST 用例一次全 PASS（三处 ST/记录侧修正——跨工件噪声带容差、脚本断言面、
> T005 skip 计数笔误——零产品缺陷），零 Critical/Major/Minor 缺陷；需求覆盖 6/6（100%）。
> 工件 109 项独立断言防自证复算全过：H1-H4 重算判定与工件逐位一致（H1 三面 4/4=1.0>0.5、
> 4/4>0/4、posthoc 双 prompt 严格占优；H2 Δ=+859.7ms 加速侧；H3 34/66 早停完全恢复；H4
> 捕获/重放>0、canary 0、账本 bound 六组逐字段复算、AR017 中性）；双粒度质量面 gap=0.0
> 逐位、greedy_win 三层自洽、复现自检八字段逐位且先验独立溯源至 AR021 工件闭环。全量
> 951 passed/0 failed 零回归；静态门（pyflakes 零/mypy clean/black 过）与 demo 三锚点
> （85.5%/2.35e-03/93.75%）精确达标；真实 checkpoint `--quick` 四模式跑通（质量四列非空、
> 退出码 0）；AR021 零改动面（脚本 diff 空 + 工件未动）确认。满足全部 Go 条件，进入 T009
> 归档。
