# [AR021] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR021 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-15 |

> 执行环境：Quadro RTX 5000 16GB，torch 2.5.1+cu121（CUDA True），WDDM，驱动 556.18，
> python 3.11.9，pytest 9.1.1；测试命令统一 `python -X utf8`，全量口径 `-m "not slow"`
> （3 deselected 为既有环境事实）。真实 checkpoint 面用 AR019 venv
> （`transformers==4.53.1` + 本地 Fast-dLLM-v2-1.5B 权重，#24 版本纪律）。工件级用例
> （ST-N2/B1/B2/B3）直接断言 `results/optimization/ar021_generation_loop_bench.json`
> （reps=30 预承诺单次定案工件，T004）；`--quick` 冒烟输出至独立临时路径，不触碰
> 正式/quick 工件。

## 测试用例列表

### ST-N1：生产回路折叠端到端（合成面全链 UT）

**关联需求：** srs.md §3.1 FR1 + §3.2 FR2 验收标准（掩码/pe/forced 供给、链延续、AR 步强制散度、ctx 供给纪律）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- AR021 实现已合入；折叠套件 23 测试（T007 后）全绿

**测试步骤：**
1. `python -X utf8 -m pytest tests\test_generation_loop_folding.py -k "ut2101 or ut2102 or ut2104 or ut2105 or ut2111 or ut2112" -q`

**期望结果：**
- UT-2101（块因果∧非 pad `[T,T]` 掩码语义 + `(T,prompt_len)` 键控对象钉定 + forced 基张量恰盖当前块）、UT-2102（rotary pe 钉定供给/朴素模型 None/dtype 跟随 embedding weight）、UT-2104（eager 四路径 forced AND 语义 + fused 计数再推导 + UT-2104c 判别）、UT-2105（folded-serial 双形态 parity bit-exact）、UT-2111（AR 步前向强制散度 + 跨块 runner 捕获）、UT-2112（ctx 供给纪律）全绿——掩码/pe/forced 供给与链延续端到端成立

**实际结果：** 10 passed（6 个 UT 函数含设备参数化；CUDA 实跑非 skip）
**状态：** PASS

---

### ST-N2：基准工件验收（字段完备 + H1-H4 判定复算）

**关联需求：** srs.md §3.4 FR4 验收标准（工件字段完备 / 判定式冻结 / 单次纪律 / meta 声明）
**测试类型：** 正常路径（工件级断言）
**优先级：** High

**前置条件：**
- `results/optimization/ar021_generation_loop_bench.json` 为 reps=30 预承诺单次定案工件

**测试步骤：**
1. python 断言脚本读取工件：九顶层字段齐备（hypotheses_declared/verdicts、mode_comparison、per_shape_window、quality、ledger、batch_canary、meta、speedups_p50）；meta 冻结面（reps=30、seed、venv、c=4、σ 口径绑模式、判定式 declared==recorded、#39 CUPTI null 注记、强制散度+形状键控池声明、warmup 覆盖每形状捕获与 AR017 过渡、三模式配置：baseline raw / serial N=0 / batched N=4+AR017 批量侧）
2. H1-H4 判定复算：H2/H3 的 σ=max(σ_a,σ_b)、delta、三态映射从 mode_comparison 原始统计重算；speedups_p50 三值复算；H1 双 prompt argmax 逐位一致（硬门）+ allclose 仲裁 + bit-parity 报告子字段；H4 五子判据（串行捕获/重放>0、批量捕获>0、canary、账本 c=4 界内、AR017 timed 0=中性）；账本逐 prompt 复算 bound==4((L+1)·ser+(L+N)·bat) 且 ops≤bound；VRAM 峰值 <16000

**期望结果：**
- 全部字段/复算断言通过；判定与工件记录一致（H1 supported / H2 inconclusive / H3 supported / H4 supported）

**实际结果：** 87 项断言全 PASS（首跑 2 项容差修正后全绿：H2 判定数值以 4 位小数存储，ST 容差 1e-9→1e-6 修正——工件自洽非缺陷，见执行注记）；H2 复算 delta=−107.5ms 未越 2σ=265.3 → inconclusive ✓；H3 复算 delta=1754.4 > 2σ=1483.8 → supported ✓；账本四组 bound 复算 1324/3596/360/3132 逐字段一致；VRAM 峰值 5374.5MB
**状态：** PASS

---

### ST-N3：基准可复现面（`--quick` 真实 checkpoint 冒烟）

**关联需求：** srs.md §3.4 FR4 验收标准（`--quick` ≤5min 且产出合法工件）
**测试类型：** 正常路径（真实 checkpoint 面）
**优先级：** High

**前置条件：**
- AR019 venv + 本地 Fast-dLLM-v2-1.5B 权重可用

**测试步骤：**
1. `ar019_venv\Scripts\python.exe -X utf8 -m scripts.ar021_generation_loop_bench --quick --hf-home <ar019_hf> --model-path <Fast_dLLM_v2_1.5B> --out <临时路径>`

**期望结果：**
- 三模式全部跑通，≤5min，产出合法工件（九顶层字段齐备）；H 判定允许 quick 期非定案（正式判定以 T004 工件为准）

**实际结果：** 完成（warmup 3 + 3 reps，全程 ≪5min）；三模式 H1 supported（bit-exact）/H3 supported（≈3.8×）/H4 supported；H2 quick 规模 refuted（单窗小形状折叠开销>收益，与 T003 冒烟记录同量级预期；正式定案 H2=inconclusive 以 T004 工件为准）；工件 schema 合法（九字段、reps=3、canary pass）；输出至临时路径未触碰仓库工件；Triton merge kernel 走 PyTorch 回退（#10 生产支持路径，环境既有事实）
**状态：** PASS

---

### ST-B1：单 prompt 早停 + 末窗无 AR 步

**关联需求：** srs.md §3.3 FR3 结束语义 + §3.4 FR4（早停/末窗边界如实入工件）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- T004 工件定案（两 prompt 实测均命中 stop token：prompt0 截断于 gen 位 2，prompt1 截断于 gen 位 2/13）

**测试步骤：**
1. 工件断言：prompt1 全池窗口止于 `p64_block0_T96`（folded-serial serial_events 27 / folded-batched batched_events 1、serial_events 2——stop 落入已提交候选 → finished 跳过后续块）；prompt0 folded-batched 投机候选不命中 stop → 全 4 块生成（batched_events 4，len 160 vs baseline 34，质量面如实报告分叉）
2. 末窗无 AR 步：`p32_block3_T160` sidecar serial forwards == 1（仅 draft，remaining=0 无 AR；非末窗 == 2）
3. `python -X utf8 -m pytest tests\test_generation_loop_folding.py::test_ut2109_selection_and_stop_semantics -q`（停止语义 UT 面）

**期望结果：**
- 早停截断/后续块跳过/末窗无 AR 全部与工件数值一致；UT 停止语义绿

**实际结果：** 工件 8 项断言全 PASS（prompt1 窗口集恰为 {p64_block0_T96}、66/77 < 全画布、末窗 1/非末窗 2）；UT-2109 停止语义 1 passed
**状态：** PASS

---

### ST-B2：池条目逐模式核算上限（≤8 零驱逐 + ctx 预算）

**关联需求：** srs.md §3.2 FR2 形状键控池 + §4 资源（VRAM 守恒）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- T004 工件 per_shape_window 定案（pool_cap=8）

**测试步骤：**
1. 工件断言：三池（folded-serial serial=2、folded-batched serial=4、batched=5）pool_size ≤8 且零 LRU 驱逐（cap 为防御界）；逐 rep 前向核算：folded-serial 单窗 31/27 sidecar 前向 == 30/26 replays/rep + 1 结构性 populate，timed replays 900/780 == replays/rep × 30 reps；timed 结构性 eager == 60（每 rep 每 prompt 恰 1 次 parent-None populate，canary 二分口径 by-design）；folded-batched 批量窗逐窗 replays_timed == 30、AR017 warmup 重捕获恰在两中段批量窗（captures_cumulative==2）；ctx 预算 8192 tokens/layer ≥ 2×max canvas 160

**期望结果：**
- 池核算逐位对账全过，零驱逐，预算充裕

**实际结果：** 21 项断言全 PASS（首跑 2 项预期修正后全绿：ST 初版误设"每窗 replays_timed==30/serial 池 4 条"——实测两 prompt 均于 block0 命中 stop（设计内 finished 语义），folded-serial 每 prompt 仅一个形状窗、窗内 30 replays/rep；修正后逐 rep 核算 sidecar==replays/rep+1 交叉自洽，见执行注记）
**状态：** PASS

---

### ST-B3：AR017 自适应边界（重捕获全落 warmup、timed 0=中性）

**关联需求：** srs.md §3.4 FR4 H4 判定式（自适应事件条件式核验：未触发=中性）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- T004 工件（AR017 仅 folded-batched 批量路径接通，#37）

**测试步骤：**
1. 工件断言：folded-batched ar017_recapture_events cumulative==2 且 timed==0（全落 warmup）；H4 numbers 一致且 status 含 "neutral"（timed 0 = 中性，不判 refuted）；folded-serial AR017 事件 0（串行路径自适应关）；meta warmup_strategy 覆盖 AR017 过渡期

**期望结果：**
- 重捕获全落 warmup 段；timed 0 中性判定式如实入工件

**实际结果：** 5 项断言全 PASS（cumulative 2 / timed 0 / neutral / serial 0 / warmup 注记）
**状态：** PASS

---

### ST-B4：默认零扰动边界（不传新参/新参默认值全既有行为）

**关联需求：** srs.md §3.2 FR2 + §3.3 FR3 默认关闭逐字节不变 + §4 语义基线（含 UT-1504 面）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- 无 folded_model / N=0 / forced None / 池开关关（默认）四重默认态

**测试步骤：**
1. `python -X utf8 -m pytest tests\test_generation_loop_folding.py -k "ut2103 or ut2108" -q`（forced None 零扰动；N=0 默认关 raw/folded eager/graph 三面 bit-exact + 停止 token 场景）
2. `python -X utf8 -m pytest tests\test_batched_graph_forward.py::test_ut1504_shape_mismatch_falls_back_eager -q`（池开关关默认 = 既有 warn-once-never-recapture 逐字节保持，srs FR2 验收第三条）
3. demo 三锚点不漂移见 ST-R2（ST 层抽验）

**期望结果：**
- 四重默认态全既有行为逐字节不变；UT-1504 面零扰动

**实际结果：** 3 passed（UT-2103 设备参数化 + UT-2108）+ UT-1504 1 passed；demo 锚点见 ST-R2
**状态：** PASS

---

### ST-E1：B>1 折叠路径 ValueError

**关联需求：** srs.md §3.1 FR1 + AGENTS #43（`[T,T]` 掩码无法表达逐行 padding → 构造期报错）
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\test_generation_loop_folding.py::test_ut2102_pe_supply_and_batch_guard -q`

**期望结果：**
- 折叠路径 `sample` 以 batch>1 调用 → `ValueError`（srs 明文；不静默降级）

**实际结果：** 1 passed（含 B>1 guard 断言）
**状态：** PASS

---

### ST-E2：FR3 构造期非法配置四面 ValueError

**关联需求：** srs.md §3.3 FR3 触发条件（无 MFF 时配置报错，不静默降级）+ UT-2110 冻结面
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\test_generation_loop_folding.py::test_ut2110_fr3_constructor_validation -q`

**期望结果：**
- 四面（N<0 / candidate_top_k<1 / 无 folded_model / folded_model 不支持 branch_ids）全部构造期 `ValueError`

**实际结果：** 1 passed（4×pytest.raises ValueError）
**状态：** PASS

---

### ST-E3：forced 形状/dtype 失配 ValueError（host 元数据无 sync）

**关联需求：** srs.md §3.1 FR1 异常处理（entry ValueError）+ AGENTS #43（host 元数据校验，零设备回读）
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 临时 python 脚本（temp 目录）双入口八面：`DiffusionSampler._forward`（float dtype / 末维失配）+ `ManualFoldedForward`（非张量 / float dtype / `[T-1]` / `[1,T-1]` / `[2,T]` 错批 / 批量 `[2,T]` vs `[3,T]` tokens）→ 全部 `ValueError` 不静默；100 次连续失配拒断墙钟 <5ms（排除每次调用设备回读的 sync 下界）

**期望结果：**
- 全部 ValueError 且消息指明期望形态；校验仅读 dtype/dim/shape host 元数据

**实际结果：** 10 项断言全 PASS（八面 ValueError 逐一命中；100 次拒断 0.54ms——host 元数据校验无设备 sync）
**状态：** PASS

---

### ST-E4：池 LRU 驱逐后同形状重现（再捕获正确性）

**关联需求：** srs.md §3.2 FR2 验收标准（池容量上限=1：LRU 驱逐旧 runner、新形状捕获成功、VRAM 探针不增长）
**测试类型：** 异常处理（资源边界）
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\test_generation_loop_folding.py::test_ut2106_lru_eviction_vram -q`

**期望结果：**
- cap=1 跨形状增长：LRU 驱逐（含 cuBLAS workspace 回收）、同形状重现再捕获正确、VRAM 守恒

**实际结果：** 1 passed
**状态：** PASS

---

### ST-R1：全量回归（920 基线 + 23 新增）

**关联需求：** srs.md §4 非功能需求（回归）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\ -q -m "not slow"`

**期望结果：**
- 943 passed / 3 skipped（= 920 既有基线 + 21 T002/T003 新增 + 2 T007 判别 UT），零既有失败

**实际结果：** 943 passed, 3 skipped, 3 deselected（slow）, 40 warnings（全部既有预期警告）, 50.55s
**状态：** PASS

---

### ST-R2：demo 三锚点

**关联需求：** srs.md §4 非功能需求（语义：demo 三指标锚点不变）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -X utf8 demo.py`

**期望结果：**
- 85.5% / 2.35e-03 / 93.75% 三锚点精确不变

**实际结果：** "Total FLOPs reduction: 85.5% / Output equivalence (MSE): 2.35e-03 [HIGH] / Estimated stable token ratio: 93.75%"
**状态：** PASS

---

### ST-R3：静态质量门（AR021 文件面）

**关联需求：** srs.md §4 非功能需求（静态：mypy / pyflakes / 100 列）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -m pyflakes actfold tests demo.py scripts`
2. `python -m mypy actfold --ignore-missing-imports`
3. `python -m black --check --line-length 100`（AR021 全部 9 改动/新增文件：cuda_graph / folded_transformer / benchmark_runner / architecture_utils / diffusion_sampler / fast_dllm_sampler / config_manager / ar021_generation_loop_bench.py / test_generation_loop_folding.py）

**期望结果：**
- pyflakes 零输出；mypy clean；AR021 九文件 black 100 列全过

**实际结果：** pyflakes exit 0（零输出）；mypy "Success: no issues found in 67 source files"；black "9 files would be left unchanged"
**状态：** PASS

---

### ST-R4：AR020 既有面回归（put_batch / batched-cache）

**关联需求：** srs.md §2 口径延续（AR020 批量 put 为折叠栈组成；#42 契约零扰动）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\test_batched_put.py tests\test_cache_protocol.py -q`
2. 对应测试名清单：UT-2001/2002/2003（三 cache put_batch 状态等价）、UT-2004（原子性）、UT-2005（环形/grow/重叠）、UT-2006（协议/发布分流/探测/异常）、UT-2007（账本）、UT-2009（foreach 降级）、随机化等价 + cache_protocol 既有面

**期望结果：**
- AR020 既有面全绿（全量 R1 已覆盖，此处独立列出确认）

**实际结果：** 80 passed（7.94s；全量 943 中零既有失败双重确认）
**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-15 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮（一次成 Go） |

### 执行注记（ST 用例侧修正，非产品缺陷）

| # | 现象 | 裁定 |
|---|------|------|
| 1 | ST-N2 初版以 rel_tol=1e-9 对拍 H2 判定数值 vs mode_comparison 原始统计——工件判定数值按 4 位小数存储（相对偏差 ~1.8e-8），容差修正为 1e-6 后全绿 | ST 容差错误；工件自洽（判定与原始统计同源，存储精度差异） |
| 2 | ST-B2 初版误设"folded-serial serial 池 4 条/每窗 replays_timed==30"——实测两 prompt 均于 block0 命中 stop token（finished 跳过后续块，设计内语义；prompt0 baseline len 34、prompt1 66），folded-serial 每 prompt 仅一个形状窗（池 2 条），窗内 30/26 replays/rep → timed 900/780 | ST 预期模型错误；修正后逐 rep 核算（sidecar 前向 == replays/rep + 1 结构性 populate、timed 结构性 eager == 60）交叉自洽，工件逐位对账全过 |

两处均为 ST 用例自身修正，未发现产品缺陷；修正后全部断言通过且多面交叉自洽。

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR1 | sampler 折叠前置条件线程化（pe/掩码钉定/B>1 卫/异常） | ST-N1, ST-E1, ST-E3 | PASS |
| §3.2 FR2 | 生产 generate() 折叠接线（强制散度/形状键控池/默认零扰动/驱逐） | ST-N1, ST-B2, ST-B4, ST-E3, ST-E4 | PASS |
| §3.3 FR3 | 批量投机块验证（起草/验证/选定/停止/构造校验/默认关） | ST-B1, ST-B4, ST-E2 | PASS |
| §3.4 FR4 | 基准脚本 + 工件（三模式/H1-H4/字段完备/单次纪律/--quick） | ST-N2, ST-N3, ST-B1, ST-B2, ST-B3 | PASS |
| §3.5 FR5 | 文档收口五件 | （T006 交付 + T007 review E 维度逐位核对 VERIFIED；ST 层不重复执行，README 头表数字经 ST-N2 工件断言溯源） | PASS |
| §4 NFR | 回归/静态/语义基线/资源 | ST-R1, ST-R2, ST-R3, ST-R4, ST-B2（ctx/VRAM 面） | PASS |

**需求覆盖率：** 6 / 6（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 3 | 3 | 0 | 0 |
| 边界条件 | 4 | 4 | 0 | 0 |
| 异常处理 | 4 | 4 | 0 | 0 |
| 回归测试 | 4 | 4 | 0 | 0 |
| **合计** | **15** | **15** | **0** | **0** |

（UT 粒度执行计数：折叠套件 23/23 在全量中全绿；ST 分组运行另计 N1 10 + B4 4 + B1/E1/E2/E4 各 1 + R4 80 = 98 项次；工件断言脚本 87+21+5+8+10 = 131 项断言全 PASS。）

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| （无） | — | — |

### 结论

> **Go**。15/15 ST 用例一次全 PASS（两处 ST 用例侧预期修正，零产品缺陷），零
> Critical/Major/Minor 缺陷；需求覆盖 6/6（100%）。全量 943 passed/3 skipped 零回归
> （920 基线 + 23 新增）；静态门（pyflakes 零 / mypy 67 文件 clean / black 九文件过）
> 与 demo 三锚点（85.5%/2.35e-03/93.75%）精确达标；真实 checkpoint `--quick` 冒烟
> 三模式跑通（H1/H3/H4 supported，H2 quick 规模 refuted 为 T003 既记预期量级，正式
> 定案以工件为准）；工件 131 项断言逐项核实（H1 supported bit-exact、H2 inconclusive
> 复算一致、H3 supported 2.47×、H4 supported 含 canary/账本/AR017 中性全字段、池核算
> 逐位对账、VRAM 5.4GB 留余量）。满足全部 Go 条件，进入 T009 归档。
