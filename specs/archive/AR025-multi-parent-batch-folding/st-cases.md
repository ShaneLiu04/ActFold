# [AR025] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR025（多 parent 批量折叠 + 双链 lockstep 并行投机） |
| 关联 srs.md | ./srs.md（FR1-FR7 + 验收标准 1-5） |
| 生成日期 | 2026-10-18 |

执行环境：ar019 venv（canonical pytest 口径）+ 系统 python（canonical
lint 口径）；Quadro RTX 5000 16GB WDDM；自动化证据全部为本轮实测。

## 测试用例列表

### ST-001：机制对拍——路径分档容差（srs 验收 1 / FR1-FR3）

**关联需求：** srs.md §2 FR1/FR2/FR3 + §3 验收 1
**测试类型：** 正常路径　**优先级：** High

**前置条件：** 等长多 parent 批量调用（A）与逐链分批调用（B）在同一
合成模型/同一缓存预算上构造。

**测试步骤：** ①运行 `scripts/ar025_multi_parent_bench --mode synth`
（8 形状点 × 双构造，CPU）；②复核 `tests/test_ar025_sample_joint.py`
UT-2430/UT-2440 parity 用例。

**期望结果：** 非 split 路径 `torch.equal` 逐位一致；split 路径
`allclose(1e-6)`；对拍覆盖同父混合/全异父/父缺失行/胜者延续两轮。

**实际结果：** 合成面正式工件 16/16 S1 supported（nonsplit 逐位 / split
allclose(1e-6)，含同父混合、全异父、父缺失行、两轮延续构造）；UT-2430
（LLaDA）/UT-2440（Dream）joint-vs-solo 逐位相等（torch.equal）通过；
ST 轮 `--quick` 冒烟重跑 S1 supported（split allclose）复现。

**状态：** PASS

---

### ST-002：零回归——冻结面不变（srs 验收 2）

**关联需求：** srs.md §3 验收 2
**测试类型：** 回归测试　**优先级：** High

**前置条件：** AR015 批量契约 UT、AR023 族投机面、AR024 粒度面在位。

**测试步骤：** 运行 sampler 冻结面回归
（`test_dream_speculative.py`/`test_llada_speculative.py`/
`test_llada_granularity.py`/`test_fast_dllm_sampler.py`/
`test_diffusion_samplers.py`）。

**期望结果：** 单 parent 批量路径与串行路径逐位一致不变；默认 off
逐字节兼容。

**实际结果：** 60 + 17 = **77 passed**（含 UT-2444 组合路径零漂移守卫、
AR024 s==block_size AR023 位级锚）；全量套件内 AR015/AR023 冻结面
全绿。

**状态：** PASS

---

### ST-003：合批开关默认 off + YAML 消费（FR6）

**关联需求：** srs.md §2 FR6
**测试类型：** 正常路径　**优先级：** Medium

**前置条件：** `ActFoldConfig.multi_parent_batch` 默认 False。

**测试步骤：** 复核 UT-2425/UT-2426（T004 交付）。

**期望结果：** 默认 off 逐字节兼容；YAML key 消费无 unknown-key 警告。

**实际结果：** UT-2425/2426 绿（T004 轮交付，全量套件内持续通过）。

**状态：** PASS

---

### ST-004：生成循环接线——W2 lockstep joint（FR5）

**关联需求：** srs.md §2 FR5
**测试类型：** 正常路径　**优先级：** High

**前置条件：** 双链等长 prompt、joint 开关全成立
（N>0+folded_model+branch_ids 支持+multi_parent_batch）。

**测试步骤：** ①`sample_joint([p1,p2])` vs 背靠背 `sample(p1)+sample(p2)`
双族 parity（UT-2430/UT-2440）；②UT-2438 joint+progressive_refill
parity（T009 补例：refill 链限定 `_r` tag + F1 实例槽换入换出）；
③UT-2433/UT-2441 off-switch 分发。

**期望结果：** joint 与逐链 solo 逐位相等（含 history 快照、refill 路径）；
off-switch 时逐链分发语义等价。

**实际结果：** UT-2430/2433/2438/2440/2441 全绿——**15/15 passed**
（本轮实测 7.49s）。

**状态：** PASS

---

### ST-005：发布面账本（FR4 / srs 验收 3）

**关联需求：** srs.md §2 FR4 + §3 验收 3
**测试类型：** 正常路径　**优先级：** High

**前置条件：** joint `[2N,T]` put_batch 发布 vs 逐链 `[N,T]` per-chain
发布，#42 `write_op_count` 账本。

**测试步骤：** 复核 UT-2431（LLaDA）/UT-2443（Dream）与合成面 S2。

**期望结果：** `write_op_count(A) < write_op_count(B)` 严格（每形状点
每构造）；#39 CUPTI null 注记入工件 meta。

**实际结果：** UT-2431/2443 绿；合成面 S2 supported——A=4 vs B=40–160
（16 点严格）；ST 轮 quick 重跑 ops A/B/C=4/40/40 复现；工件 meta 含
cupti_note（launch-ledger proxy 声明，不主张 kernel 计数）。

**状态：** PASS

---

### ST-006：C5 早结束链摘除（FR5 边界）

**关联需求：** srs.md §2 FR5（design C5）
**测试类型：** 边界条件　**优先级：** High

**前置条件：** 链 1 首窗提交 EOS（prompt-keyed 表头 fixture）。

**测试步骤：** UT-2432（早结束混合）+ UT-2436（spy rows 2N→N）。

**期望结果：** 早结束链摘除后 joint 批量行 2N→N 无崩溃；余链 token 与
solo 参考逐位相等。

**实际结果：** UT-2432/2436 绿（rows 2N→N、链 id 不相交、余链逐位相等）。

**状态：** PASS

---

### ST-007：互斥与入口守卫（FR1 / FR5 F3+armed）

**关联需求：** srs.md §2 FR1（互斥）+ FR5（joint 入口）
**测试类型：** 异常处理　**优先级：** High

**前置条件：** armed joint 路径（N>0 + switch-on MFF）。

**测试步骤：** ①`parent_branch_id` 与 `parent_branch_ids` 双传 →
ValueError（机制面 UT，T002 交付 UT-2401..2415 契约集）；②joint 入口：
空列表/batch>1/prompt 不等长/cfg_scale>0/temperature>0/stochastic
transfer 双族（UT-2434/2435/UT-2442/2445）。

**期望结果：** 全部 ValueError，措辞含 [T,T] mask 说明/cfg_scale 等。

**实际结果：** 全部绿（15/15 内）；`_MULTI_PARENT_MSG` 常量单一 raise
站点（T009 修复）；段A/段B 空 masked_local 防御对称（T009 修复）。

**状态：** PASS

---

### ST-008：DD8 erratum 容量边界（FR7 合成面边界）

**关联需求：** srs.md §2 FR7（合成面）
**测试类型：** 边界条件　**优先级：** High

**前置条件：** 缓存预算按网格最坏人口取 65536 行/层。

**测试步骤：** 复核合成面正式工件（两轮记录：8192 预算 5 点 FAIL →
65536 预算冻结判定式零改动重跑）。

**期望结果：** 高人口点无在飞父逐出；首跑 blocked 工件独立保全。

**实际结果：** 工件 `ar025_multi_parent_bench.json` 16/16 supported
（meta 含预算 65536 + erratum 注记）；blocked 首跑工件
`ar025_multi_parent_bench_eviction_blocked.json` 独立保全；失败谓词
`(chains+chains·N)·T>8192` 与 5 点完全吻合（诊断记录于 tasks.md T005）。

**状态：** PASS

---

### ST-009：合成面 multi-geometry 基准（FR7）

**关联需求：** srs.md §2 FR7 合成面
**测试类型：** 正常路径　**优先级：** High

**前置条件：** ≥4 形状点（实际 8：chains{2,4}×N{4,8}×T{256,1024}）。

**测试步骤：** 复核正式工件（预承诺单次 + erratum 重跑）。

**期望结果：** S1/S2 supported、S3 report-only；判定式冻结先于数据。

**实际结果：** 16/16（8 点×双构造）：S1 supported / S2 supported
（严格）/ S3 0.9–0.975；判定式全文冻结于工件 meta；ST 轮 quick 重跑
S1/S2 supported 复现。

**状态：** PASS

---

### ST-010：真实面基准 reps=30（FR7 真实面）

**关联需求：** srs.md §2 FR7 真实 checkpoint 面
**测试类型：** 正常路径（延后执行）　**优先级：** High

**前置条件：** ≥24GiB 高速算力 GPU + 本地权重 + probe 工件。

**测试步骤：** 按 `docs/RERUN_CHECKLIST.md` 步骤 16 执行
`--mode real --family llada|dream --reps 30`（预承诺单次）。

**期望结果：** H1 逐位硬门+归因 / H2 严格 p50(B)/p50(A)>1.0 / 交付线
≥10% / H4 账本严格小；判定式冻结于脚本先于数据。

**实际结果：** **BLOCKED-DEFERRED（诚实延后）**：接线与判定式已交付
（T006，AR023 模板零走样——T009 审查锚点核对）；本机 16GB WDDM 四次
冒烟尝试均环境限制止损（bf16 15.8GB 触顶硬崩无 traceback；int8 逃生门
bitsandbytes 量化驱动坏态停滞，fresh 重启复现），零伪数字；正式跑按
用户指令延后至高速算力机执行后回填（srs FR7 冻结降级路径 + T006 tasks
诚实留档）。**非缺陷**——延后执行属用户指令与环境约束，判定式/命令/
资源表已全量交付（RERUN 16）。

**状态：** BLOCKED-DEFERRED

---

### ST-011：降级交付触发器冻结（FR7 诚实负结果路径）

**关联需求：** srs.md §2 FR7（H2 <10% 且 H1 PASS ⇒ blocked 注记交付）
**测试类型：** 异常处理（声明面）　**优先级：** Medium

**前置条件：** 判定式文本冻结于脚本。

**测试步骤：** 复核脚本 DD8 判定式注释块（`_VERDICT_*` 常量区）与
`_real_blocked_artifact` 结构。

**期望结果：** 降级触发器先于数据冻结；blocked 工件零伪数字。

**实际结果：** T009 审查锚点核对：H1/H2/交付线/H4/canary 判定式与
DD8 原文逐字一致（:500-531 注释区）；`_real_blocked_artifact` 只含
reason + probe cross_ref，零数字字段；probe degraded → blocked 路径
接线完整。

**状态：** PASS

---

### ST-012：全量质量门（srs 验收 5）

**关联需求：** srs.md §3 验收 5
**测试类型：** 回归测试　**优先级：** High

**前置条件：** ar019 venv canonical deselect 口径。

**测试步骤：** `python -m pytest tests/ -q`（ST 轮实测）+ actfold 四
lint 门（black/isort/pyflakes/mypy）。

**期望结果：** 只增不减（AR023 基线 1022 起算）；改动面 lint clean。

**实际结果：** **1108 passed / 5 failed / 4 skipped / 3 deselected**
（90.05s，ST 轮实测）；1022→1108 = +86（15 AR025 + 71 并行 AR/轮次）；
5 failed 全 ⊆ 既有环境集 triton×5（本机无 CUDA toolkit）；lm-eval×2
bench-backend 已随 `pytest addopts` 默认 deselect（9a6ded3）；
black/isort/pyflakes exit 0 + mypy 68 文件 0 错。

**状态：** PASS

---

### ST-013：demo 三锚点不动（srs 验收 5）

**关联需求：** srs.md §3 验收 5
**测试类型：** 回归测试　**优先级：** Medium

**测试步骤：** `python demo.py`（ST 轮实测）。

**期望结果：** FLOPs 削减 85.5% / MSE 2.35e-03 / stable ratio 93.75%。

**实际结果：** 85.5% / 2.35e-03 / 93.75% —— 与 RERUN §5 一致，零漂移。

**状态：** PASS

---

### ST-014：T009 修复面回归（审查修复循环）

**关联需求：** tasks.md T009（12 项修复）
**测试类型：** 回归测试　**优先级：** Medium

**测试步骤：** UT-2438/UT-2445 新例 + sampler 回归面 + 重审子代理
git diff 逐字核对。

**期望结果：** 12/12 FIXED-OK、零新缺陷、修复面测试全绿。

**实际结果：** 15/15（含 2 新例）+ 60 回归绿；第二轮重审 12/12
FIXED-OK；唯一新发现（文档计数失同步）当轮修正并全量复跑
（1108/5/4/3 实测）。

**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-18 |
| 执行结果 | PASS（真实面 1 项 BLOCKED-DEFERRED 诚实延后，非缺陷） |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| FR1 | per-row parent 批量 API（互斥/None-attach） | ST-001, ST-007 | PASS |
| FR2 | per-row 父装配（行级降级） | ST-001, ST-008 | PASS |
| FR3 | per-row gate/merge | ST-001 | PASS |
| FR4 | put_batch 发布面 | ST-005 | PASS |
| FR5 | 生成循环接线（sample_joint W2） | ST-004, ST-006, ST-007 | PASS |
| FR6 | 配置面（默认 off 兼容） | ST-003 | PASS |
| FR7 | 基准（合成面/真实面/判定式冻结） | ST-008, ST-009, ST-010, ST-011 | PASS + BLOCKED-DEFERRED（真实面） |
| 验收 1 | 机制对拍（路径分档容差） | ST-001 | PASS |
| 验收 2 | 零回归 | ST-002 | PASS |
| 验收 3 | 账本 | ST-005 | PASS |
| 验收 4 | 基准工件 | ST-009, ST-010 | PASS + BLOCKED-DEFERRED（真实面工件待实机） |
| 验收 5 | 质量门 | ST-012, ST-013 | PASS |

**需求覆盖率：** 7/7 FR + 5/5 验收标准 = 100%（真实面 reps=30 为
BLOCKED-DEFERRED：接线/判定式/复现命令 100% 交付，数据面延后高速算力
机——srs FR7 冻结降级路径 + 用户指令，非覆盖缺口）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 6 | 5 | 0 | 1（BLOCKED-DEFERRED） |
| 边界条件 | 3 | 3 | 0 | 0 |
| 异常处理 | 2 | 2 | 0 | 0 |
| 回归测试 | 3 | 3 | 0 | 0 |
| **合计** | **14** | **13** | **0** | **1（延后，非缺陷）** |

自动化证据：UT 15/15（7.49s）+ sampler 回归 77（60+17）+ 全量
1108/5/4/3（90.05s）+ demo 三锚点 + quick 冒烟重跑 + 四 lint 门；
合成面正式工件 16/16 supported（引用）。

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor（延后执行项） | 真实面 reps=30 正式数据（ST-010）——本机 16GB WDDM 四次冒烟均环境限制止损（bf16 15.8GB 触顶硬崩/int8 量化驱动坏态停滞），零伪数字 | 按用户指令延后至高速算力机执行（RERUN 步骤 16 全量命令/资源/限制注记已交付），工件回填后 README 优越性总表第 8 档与最新结果节同步更新 |

### 结论

> **建议 Go**：需求覆盖 100%（7/7 FR + 5/5 验收）；零 Critical/Major；
> 零回归（1108 只增不减）；唯一延后项为真实面正式数据（环境约束 +
> 用户指令的诚实延后，机制/接线/判定式/复现路径 100% 交付且 T009 两轮
> 审查 PASS）。
