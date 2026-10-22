# [AR032] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR032 |
| 关联 srs.md | ./srs.md（final，2026-10-22） |
| 关联 design.md | ./design.md（final，2026-10-22） |
| 生成日期 | 2026-10-22 |
| ST 执行环境 | Windows / Quadro RTX 5000 16GB (WDDM) / AR019 venv（真实面）/ C:\Python311（合成面） |

## 测试用例列表

### ST-001：默认关零回归（FR1 AC1）

**关联需求：** srs §3.1 FR1 验收标准 1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 默认配置（`speculative_advance_reuse=False`），投机路径与非投机路径

**测试步骤：**
1. 运行 UT-3171（投机配置 + reuse False：decode_final + history 快照序列与 AR031 末态逐位一致）
2. 运行全量 pytest 套件比对基线（1463 passed / 4 skipped / 3 deselected）

**期望结果：**
- Then 新代码默认面与 AR031 末态逐位一致（含 history 快照序列）；全量零回归

**实际结果：** UT-3171 PASS（`tests/test_ar032_advance_reuse.py`）；全量 1463 passed / 4 skipped / 3 deselected（109.27s）——与 T004 基线完全一致，零回归。

**状态：** PASS

---

### ST-002：省略计数探针（FR1 AC2）

**关联需求：** srs §3.1 FR1 验收标准 2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 开启且投机路径激活（N=4 + greedy + folded_model）

**测试步骤：**
1. 运行 UT-3172（wrap `_forward` 计数：on 臂每 eligible block == 2 次、off == 3 次；末块两侧 advance 段均 0 增量；差值 == eligible block 数）

**期望结果：**
- Then 每 block `_forward` 调用数 == `2·num_small_blocks`；省略恰 1 次/block-advance-eligible block

**实际结果：** UT-3172 PASS（S=1 几何）；真实面佐证：formal 工件 mechanism_probe off_serial(4)−on_serial(2)==eligible(2)，逐 prompt delta==1。

**状态：** PASS

---

### ST-003：next_token 来源与产物同构（FR1 AC3）

**关联需求：** srs §3.1 FR1 验收标准 3
**测试类型：** 正常路径 / 边界条件
**优先级：** High

**前置条件：**
- Given 开启，S=1 与 S>1 双几何

**测试步骤：**
1. 运行 UT-3173（`_WinnerRowRecorder` 探针：commit 后 `x == cands[j_star]` 逐位，winner 行精确钉住，T007 强化版）
2. 运行 UT-3174（S>1 内容条件化 fixture：错拿非最后 small block 的 logits 必被捕获；monkeypatch 缺陷注入必失败）
3. 运行 UT-3177（`pad_id` 覆写 / canvas 增长序列 / history 快照数 on/off 同构）

**期望结果：**
- Then 来源为最后一个 small block 的验证 logits；shape/pad 覆写/append 顺序与关闭时完全同构

**实际结果：** UT-3173/3174/3177 全 PASS（强化后的 UT-3173 以 `sampler.winner_rows[k]` 精确断言 `cands[j_star] == committed`）。

**状态：** PASS

---

### ST-004：True + N=0 构造期 ValueError（FR1 AC4）

**关联需求：** srs §3.1 FR1 验收标准 4
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given `speculative_advance_reuse=True` 且 `num_block_candidates == 0`

**测试步骤：**
1. 运行 UT-3170（构造期 `ValueError`，消息含两键名）

**期望结果：**
- Then 构造期 fail-fast `ValueError`

**实际结果：** UT-3170 PASS。

**状态：** PASS

---

### ST-005：eager 全重算硬门（FR2 AC1）

**关联需求：** srs §3.2 FR2 验收标准 1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 合成模型 + `SimilarityGate(tau=1.0)`（cosine，`sim > tau` 判 stable ⇒ 全量重算）+ CPU 非 split 路径 + 投机开启

**测试步骤：**
1. 运行 UT-3175（on/off 完整轨迹逐位一致硬门）

**期望结果：**
- Then decode_final + history 逐位一致（分叉 ⇒ 机制语义错误）

**实际结果：** UT-3175 PASS（T003 Green 期实测逐位一致）。

**状态：** PASS

---

### ST-006：S>1 几何复用来源与省略计数（FR2 AC2）

**关联需求：** srs §3.2 FR2 验收标准 2
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 合成模型 + `small_block_size < block_size` + 投机开启

**测试步骤：**
1. 运行 UT-3174（错源捕获 + 省略计数 == 1 次/block-advance-eligible block）

**期望结果：**
- Then 复用来源为最后一个 small block；省略计数契约成立（末块两侧均为 0）

**实际结果：** UT-3174 PASS（内容条件化 `(token, position)` fixture + 缺陷注入验证捕获力）。

**状态：** PASS

---

### ST-007：折叠路径一致率记录面（FR2 AC3）

**关联需求：** srs §3.2 FR2 验收标准 3
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given 合成模型 + 折叠开启（tau<1）+ 投机开启

**测试步骤：**
1. 运行 UT-3176（on/off 逐块 next_token 一致率以断言形式记录；一致时 decode_final 逐位一致硬断言；on 输出为合法 canvas）

**期望结果：**
- Then 一致率如实记录；decode_final 分叉时如实断言分叉（不伪造一致）

**实际结果：** UT-3176 PASS（tau=0.95 实测 on/off 逐位一致，如实断言）。

**状态：** PASS

---

### ST-008：非投机路径不触达（FR2 AC4）

**关联需求：** srs §3.2 FR2 验收标准 4
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 非投机路径（N=0 folded-serial 与 baseline `folded_model=None`）

**测试步骤：**
1. 运行 UT-3171 ② 非投机子句（FR2 AC4 实现面：on 臂在非投机路径结构空洞——True+N=0 构造期 `ValueError`）

**期望结果：**
- Then 非投机路径输出与 AR031 末态逐位一致

**实际结果：** UT-3171 ② PASS。

**状态：** PASS

---

### ST-009：bench quick 冒烟 + 独立工件（FR3 AC1，design §6.3）

**关联需求：** srs §3.3 FR3 验收标准 1
**测试类型：** 集成（真实 checkpoint）
**优先级：** High

**前置条件：**
- Given Fast-dLLM v2-1.5B 真实 checkpoint，AR019 venv（transformers==4.53.1），`--quick`

**测试步骤：**
1. `python -m scripts.ar032_speculative_advance_bench --quick --model-path <ckpt>`（T005 执行）
2. 核验 `_quick.json` 存在、正式工件未被触碰

**期望结果：**
- Then 冒烟通过并写独立 `_quick.json`；三臂/判定与正式同一套

**实际结果：** T005 首跑零修正：`ar032_speculative_advance_bench_quick.json` overall=supported（H1-H4 全 supported）；正式工件路径未被触碰；ST 复核工件在盘、三臂 p50 = 23747.79 / 3747.76 / 3072.15 ms。

**状态：** PASS

---

### ST-010：formal 工件完整性与可重算性（FR3 AC2，design §6.3）

**关联需求：** srs §3.3 FR3 验收标准 2
**测试类型：** 集成（真实 checkpoint）
**优先级：** High

**前置条件：**
- Given reps=30 预承诺单次完成（T006）

**测试步骤：**
1. 从工件自身记录的 faces（quality faces / 各臂 pooled p50 与 sigma / mechanism totals / canary / ledger / repro rows）调用冻结的 `assemble_verdicts` 重导全部判定
2. 核验 h3 计数输入 == 工件 mechanism totals；`speedups_p50` 与 numbers 逐位一致（4 位舍入）
3. 对照 T007 审查子代理的独立逐位重算记录

**期望结果：**
- Then 工件含三臂全指标、H1-H4 判定、逐样本记录、forward 调用计数；每个数字可由工件重算

**实际结果：** ST 重算 PASS——h1/h2/h3/h4/overall 五判定全部 match（supported）；h3 计数输入与 totals 一致（4−2==2）；ratio 一致性 exact（7.9698 / 1.2875）；T007 审查子代理（`ses_e9e0bc23dffeR1htgBsxW1iO8X`）独立重算结论一致。

**状态：** PASS

---

### ST-011：honest negative 语义（FR3 AC3）

**关联需求：** srs §3.3 FR3 验收标准 3
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given 任一硬标准 refuted 的判定路径（本 AR 交付结果为全 supported——语义面由 UT 全分支钉死）

**测试步骤：**
1. 运行 UT-3178/UT-3179（判定函数合成输入全分支：None → refuted；计数面单独可 refute；repro 错配 → H4/overall refuted 且 H1-H3 保留）

**期望结果：**
- Then 判定语义与冻结契约一致；honest negative 原样交付面存在

**实际结果：** UT-3178/UT-3179 全 PASS（9 测试）；formal 实测结果 overall=supported 已原样入工件，无 rerun。

**状态：** PASS

---

### ST-012：YAML 消费面（FR4 AC1）

**关联需求：** srs §3.4 FR4 验收标准 1
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given 新 YAML 键 `speculative_advance_reuse`

**测试步骤：**
1. 运行 UT-3170（`ActFoldConfig` 消费无 unknown-key 警告；默认 False）

**期望结果：**
- Then 字段正确消费、无警告

**实际结果：** UT-3170 PASS。

**状态：** PASS

---

### ST-013：全量零回归（FR4 AC2）

**关联需求：** srs §3.4 FR4 验收标准 2
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given 默认配置全仓

**测试步骤：**
1. 全量 `pytest tests/ -q`

**期望结果：**
- Then 与 T004 基线（1463 passed / 4 skipped / 3 deselected）完全一致

**实际结果：** 1463 passed / 4 skipped / 3 deselected（109.27s）——逐项一致，零回归、零新增 skip。

**状态：** PASS

---

### ST-014：环境门 rc=2 不写工件（design §6.4）

**关联需求：** srs §3.3 异常处理；design §6.4
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- Given 环境不满足（权重路径缺失）

**测试步骤：**
1. ST 实跑（负向）：`python -m scripts.ar032_speculative_advance_bench --model-path D:\nonexistent_ckpt_dir`（AR019 venv）
2. 核验退出码与工件状态

**期望结果：**
- Then 构造期显式报错 rc=2，不写部分工件

**实际结果：** rc=2，stderr `environment gate failed: model directory not found`，正式工件原样未动。正向面（环境满足）由 T005/T006 实跑覆盖。

**状态：** PASS

---

### ST-015：正式工件覆盖保护 rc=4（design §6.3）

**关联需求：** srs §3.3 协议行；design §6.3
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- Given 默认正式路径工件已存在，非 quick、无显式 `--out`

**测试步骤：**
1. ST 实跑：`python -m scripts.ar032_speculative_advance_bench --model-path <real ckpt>`（AR019 venv；环境门通过后触发覆盖保护，模型不加载）
2. 核验退出码与提示

**期望结果：**
- Then 拒绝覆写 rc=4（预承诺单次纪律）；显式 `--out` 为操作员自觉重跑通道

**实际结果：** rc=4，stderr `refusing to overwrite the existing formal artifact ... (pre-committed single-shot discipline; pass an explicit --out for a deliberate rerun)`——数秒内返回（保护在 `run_bench` 之前）。

**状态：** PASS

---

### ST-016：on 模式 advance 无来源 RuntimeError（design §6.4 / D3）

**关联需求：** srs §3.1 边界；design §6.4
**测试类型：** 异常处理
**优先级：** Low

**前置条件：**
- Given on 模式 advance 段 `advance_tail_logits is None`（不可达防御）

**测试步骤：**
1. 代码面核验（fast_dllm_sampler.py advance 段 D3 fail-fast 分支）
2. UT-3174 错源捕获侧证

**期望结果：**
- Then `RuntimeError`，零静默降级

**实际结果：** 代码面核验：分支存在于 on/N>0 路径（fail-fast 优先于任何降级）；该分支结构性不可达（投机路径每 fresh block 必有投机调用），UT-3174 钉住相邻可达面（错源分叉必被捕获）。MANUAL-inspection。

**状态：** PASS

---

### ST-017：单臂失败不写工件（design §6.4）

**关联需求：** srs §3.3 异常处理；design §6.4
**测试类型：** 异常处理
**优先级：** Low

**前置条件：**
- Given 单臂失败（OOM 等）

**测试步骤：**
1. 设计面核验：bench 无 OOM blocked-工件路径（沿 AR031 实际口径）；工件写出仅在 `run_bench` 完整返回后（`write_results(data, out)`）

**期望结果：**
- Then 进程崩溃、不写任何工件（零伪造数字）

**实际结果：** 代码面核验：`main` 中工件写出严格后置于 `run_bench` 返回；三臂任一崩溃即进程退出、无部分工件面。不做真实 OOM 注入（会伪造故障条件），以写出位置的结构性核验为证据。MANUAL-inspection。

**状态：** PASS

---

### ST-018：文档七件套一致性（FR4 AC3）

**关联需求：** srs §3.4 FR4 验收标准 3
**测试类型：** 文档核验
**优先级：** High

**前置条件：**
- Given 文档七件套（README 六节全量重写 / RERUN 步骤 20 / AGENTS #51 / CHANGELOG / ALGORITHM §17 / EXPERIMENTS §4.5b / GUIDE 第三十七部分）

**测试步骤：**
1. T008 文档 pass 完成后逐件核验：计数/编号/措辞与工件一致（H2 7.9698× / H3 1.2875× / 计数 4−2==2 / repro 6/6 / ledger 界 / 全量基线 1463）
2. 跨 rung 计时漂移按 T007 记录口径书写（同 run 比值内部有效；跨工件绝对值不可直接比）

**期望结果：**
- Then 七件套数字全部可由工件重算；ladder 措辞符合漂移口径

**实际结果：** 七件套逐件核验 PASS——AGENTS #51（计数探针/尾行克隆/冻结契约语义/跨 rung 漂移口径全量入条）；RERUN 步骤 20（命令/资源实测/~92s quick 注记/漂移口径）；CHANGELOG（Added/Fixed/Results 三段，数字与工件一致）；ALGORITHM §17（冻结定义/等价论证/计数契约/实测面）；EXPERIMENTS §4.5b（沿 §4.5a 先例编号）；GUIDE 第三十七部分（问题/机制/结果/三件过程纪律）；README 六节全量重写（创新总览第 17 行 + 机理详解第 16 节 + 最新实验结果头条 + ladder 第 13 档 + 快速上手 + 质量基线 1463 + 路线图收口）；RERUN §0 质量门基线同步 1463（AR032 后基线，ST 实测 109.27s）。跨文档数字一致性扫描：7.9698/1.2875/26305.87/3300.68/4249.76/4−2==2/repro 6/6/1463 七件套全一致。

**状态：** PASS

---

### ST-019：NFR 性能门（srs §4）

**关联需求：** srs §4 非功能需求（性能门行）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given AR032 单测套件 < 30 s；bench quick < 90 s

**测试步骤：**
1. 计时 `pytest tests/test_ar032_advance_reuse.py tests/test_ar032_bench_verdicts.py`
2. 由 quick 工件计三臂 timed face 总耗时（3 臂 × 3 reps）

**期望结果：**
- Then 两项性能门满足

**实际结果：** UT 套件 18 passed in 6.87s ✓（< 30 s）。bench quick timed face = **91.7 s**（baseline 23747.79 / reuse-off 3747.76 / reuse-on 3072.15 ms p50，各 ×3 reps；不含模型加载与 warmup，全墙钟约 3 min）——**超出 90 s 门**。根因：baseline 臂在 WDDM 主机状态噪声下的墙钟主导（#46 环境条件性；本 AR 机制臂自身仅 3.07 s/p50，与机制无关）；该门为 srs draft 期预估，先于任何 quick 测量写入。如实记录，不调参、不重跑、不改门。

**状态：** FAIL（Minor，env-conditional；处置见报告遗留问题表）

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-22 |
| 执行结果 | **Conditional-Go** |
| 执行轮次 | 第 1 轮（ST-019 唯一 FAIL 为 NFR 计时门，非功能面） |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR1 | 投机块推进复用机制 | ST-001, ST-002, ST-003, ST-004, ST-016 | PASS |
| §3.2 FR2 | 机制等价硬验收（合成面） | ST-005, ST-006, ST-007, ST-008 | PASS |
| §3.3 FR3 | Fast-dLLM 真实面三臂基准 | ST-009, ST-010, ST-011, ST-014, ST-015, ST-017 | PASS |
| §3.4 FR4 | 配置/接线/导出/文档 | ST-012, ST-013, ST-018 | PASS |
| §4 NFR | 非功能需求（性能门） | ST-019（及其余用例承载的正确性/开销/协议面） | PASS 除 quick 计时门 |

**需求覆盖率：** 4/4 功能需求（100%）；NFR 8 行中 7 行 PASS、1 行 FAIL（Minor）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 8 | 8 | 0 | 0 |
| 边界条件 | 3 | 2 | 1 | 0 |
| 异常处理 | 5 | 5 | 0 | 0 |
| 回归测试 | 1 | 1 | 0 | 0 |
| 集成（真实 checkpoint） | 2 | 2 | 0 | 0 |
| **合计** | **19** | **18** | **1** | **0** |

**执行记录：**
- 自动化：18/18 AR032 UT 全绿（6.87s）；全量 1463 passed / 4 skipped / 3 deselected（109.27s，与基线逐项一致）
- 实跑负向面：rc=2（环境门，工件未动）、rc=4（覆盖保护，`run_bench` 前返回）
- 工件重算：`assemble_verdicts` 从工件自身 faces 重导 h1/h2/h3/h4/overall 全部 match（supported）；h3 计数输入==totals（4−2==2）；`speedups_p50` ratio 一致性 exact（7.9698 / 1.2875）；与 T007 审查子代理独立重算结论一致
- 真实面：T005 quick（首跑零修正，`_quick.json` overall=supported）+ T006 formal（reps=30 预承诺单次，overall=**supported**：H1 gap 0.0 / decode [34,66] on==off / exact-match 1.0/1.0；H2 **7.9698×**；H3 **1.2875×** + 计数恒等式；H4 canary 3/3 + ledger within + repro 6/6）

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | ST-019：bench quick 计时门 91.7 s > 90 s（timed face；全墙钟约 3 min）——WDDM 主机状态下 baseline 臂墙钟主导，机制无关（#46 环境条件性）；门值为 srs draft 期预估 | 按用户授权 auto 模式裁定：**记录延后处理**——不改工件、不重跑、不移动门柱；后续 AR 若重订 quick 几何，按主机条件性（#46）在 srs 期以 AR031/AR032 实测校准计时门 |
| Cosmetic | （T007 记录承接）跨 rung baseline p50 漂移（3096.76→17279.30→26305.87ms）| 已在 README ladder 措辞中按「同 run A/B 比值内部有效」口径处理（ST-018 核验） |

### 结论

> **Conditional-Go**：四项功能需求 100% 覆盖、18/19 用例 PASS、全量零回归、formal 工件整体 supported 且逐位可重算；唯一 FAIL 为 NFR quick 计时门（Minor、env-conditional、机制无关），按用户授权的 auto 模式确认记录延后处理。核心路径全部验证通过，AR032 进入归档。
