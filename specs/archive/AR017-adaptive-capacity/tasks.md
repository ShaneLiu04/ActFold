# [AR017] 任务清单 — 自适应发散容量重捕获 + 几何扫描基准

> 状态图例：☐ 待办 / ◐ 进行中 / ✅ 完成 / ✗ 取消

| # | 任务 | 状态 | 产出与验收 |
|---|------|------|-----------|
| T001 | Red：新测试 `tests/test_adaptive_capacity.py` UT-1601~1609 先于生产失败 | ✅ | 1184 行（后经 T002 的 UT-1608 修复增至 1195 行）；UT-1601~1608 全红（8 failed，红因全部=生产缺失：ImportError/TypeError/AttributeError，无测试自身 bug）；UT-1609 预留 mutation 阶段；回归 33 passed 零扰动；pyflakes 零/100 列零 |
| T002 | Green：FR1 暂存 + FR2 策略 + FR3 生命周期（观测窗口 + 先捕新后弃旧 swap）；mutation ①-④ 全红后 SHA256 还原 | ✅ | 五处生产改动全落地（新模块/暂存/钩子+生命周期/6 字段+校验/接线）；UT-1608 测试侧时序 bug 修复（插入节流 warm-up 步，断言面不变，design 不让步）；UT-1601~1608 一次全绿；mutation ①删双钩子→UT-1602 红/②删增长分支→UT-1603 红/③删上限退役→UT-1604 红/④先弃旧后捕新→UT-1605 红，SHA256 逐字节还原（architecture_utils 196E87…F3024 / adaptive_capacity 10A9CB…BCA15）；还原后全量 846 passed（838+8）零回归 |
| T003 | 基准脚本 `scripts/ar017_adaptive_bench.py` + 执行 + 工件 | ✅ | 三路径 × {(256,8),(1024,16)} × N{4,8} × T=512（N·T 全 2 幂），round-robin，warmup=5 覆盖适配期；**n=10 预运行在边界配置 (256,8,4) 判定翻转（两次同码运行 supported/inconclusive 各一）→ 正式工件取 n=30 单次预承诺运行（--reps 30，判定照单全收）**；adaptation 状态表（recapture_events/capacity_trajectory/observed_max_layer_divergence p50-min-max，轨迹窗口外采集零污染；需求样本 n=34=35 调用−换图步，新 runner 未 validate 暂存 None 不采样）；H1-H4 声明与判定分离；meta 补驱动版本 556.18 + CUPTI 探测 null+注记；全量 ~10 分钟，--quick 冒烟通过；工件 `results/optimization/ar017_adaptive_bench.json`（meta.command 含 --reps 30 可复现） |
| T004 | 全量回归 + demo 锚点 | ✅ | 846 passed / 3 skipped 零回归（既有测试仅 UT-1608 插步）；demo 三指标精确一致（85.5%/2.35e-03/93.75%） |
| T005 | 文档收口（FR6） | ✅ | 指南 P1-7 行第三档追注 + 第二十四部分（墙钟表 n=30 + H1-H4 诚实判定：H2 整体 inconclusive、深几何 2/2 decisively supported 2.30×/2.46×）+ Part 23/24 遗留交叉注记；CHANGELOG AR017 节置 Unreleased 最前（含 review 门控与 n=30 重跑理由）；AGENTS #37 扩注；数字全部从工件逐值复算（review 子代理独立复核吻合） |
| T006 | 质量门 | ✅ | mypy clean（**66 文件**，adaptive_capacity.py 新增）/ pyflakes 零 / 100 列零（python len 复核）/ 全量 846 passed / demo 锚点 |
| T007 | review 门控：只读子代理审查 srs/design/实现/测试/工件合规 | ✅ | 门控记录见下 |
| T008 | ST 验收：st-cases.md 场景用例 | ✅ | 16 用例一次成 Go（16 PASS / 0 FAIL / 0 BLOCKED，需求覆盖 6/6=100%）；执行证据：本日新跑 `pytest tests\test_adaptive_capacity.py` 8 passed、config 边界探针（6 默认值 + 三非法构造 ValueError）、工件全字段探针（4 配置 reps=30/warmup=5/parity=True×4/allclose=True×4/tracks_demand=True×4/verdicts 与文档一致/p1_7 overall=inconclusive 诚实）、文档在场探针（guide :180+Part24+交叉注记/CHANGELOG 最前/AGENTS #37）、T006 质量门与 mutation 记录复用；3 项 Minor 遗留均为已记录诚实边界（(256,8,4) WDDM 展布 inconclusive/串行路径后续档/n=34 采样缺口），不阻塞 |
| T009 | 归档 + git commit + gitee 推送 + 树验证 | ✅ | 两步交付先例：主体 commit `f11cae5`（15 文件 +4304/-9，含 specs/archive/AR017-adaptive-capacity 归档）→ T009 标记 commit → 各自 commit-tree 推 gitee → 双端树验证 |

## 门控记录

### review 门控
- 审查子代理：general（ses_eca032b0dffeOWDW2Xzslk9PTU）；审查面：srs FR1-FR6 / design §2-§7 / 五处生产改动 / UT-1601~1608（含 UT-1608 warm-up 修复的最小性与断言面保持）/ 基准脚本与工件 / 三处文档数字（工件逐值独立复算）
- 首轮结论：**PASS**（无 Major），8 项审查问题逐项确认（策略语义与 UT-1601 表逐例吻合含 grow_factor=1.0 合法边界；暂存单 tolist 零额外同步；钩子次序/一次性告警/`_graph_batched_capture_failed` 不动/失败保旧；prospective 防御行；配置流校验独立于开关；UT-1608 修复最小正确；基准几何/交错/轨迹表/驱动版本全确认；默认关与 AR016 逐位等价）
- Minor-1~6 全修：①测试文件行数 1184→1195（CHANGELOG/指南）②假设标签映射注记（实现 H2 承载 P1-7；srs H2/H3 由数据直接验证成立：1.55>1.35/1.68>1.52/2.30>1.39/2.46>1.44 且深几何>小几何；实现 H3=srs H4；实现 H4 新增）③adaptation 表补 `capacity_trajectory` + `observed_max_layer_divergence` p50/min/max（对齐 design §7 结构；键名保持 per-config `adaptive_state`，与 srs 顶层键的差异在指南/CHANGELOG 注记）④meta 补 CUPTI 探测字段（沿 AR016 kernel_count null+注记）⑤`benchmark_runner.py` import 上提模块顶（design §5.2 字面）⑥Part 23 遗留"固定容量过配项 ✅ AR017"交叉注记 + Part 24 遗留补宿主侧分解句
- ③④⑤ 触发工件重跑：n=10 两跑在 (256,8,4) 判定翻转（supported/inconclusive）证欠功效 → 预承诺 n=30 单次高功效运行为正式工件（期间发现 meta.command 漏记 `--reps 30`，修构造后再跑一次同参数运行锁定；最终判定照单全收：H1 inconclusive（深几何 2/2 supported）、H2 整体 inconclusive（3/4 supported，深几何 decisively）、H3/H4 supported）
- **最终结论：PASS（Minor 全修，可进入 ST）**

### ST 门控
- 执行轮次：第 1 轮（2026-10-14），一次成 **Go**
- 用例分布：正常路径 8 / 边界 5 / 异常 1 / 回归 2 = 16 全 PASS；需求覆盖 FR1-FR6 + NFR = 6/6（100%）
- 执行方式：UT 复用（本日新跑 8 passed）+ 独立 ST 探针（config 边界 / 工件字段 / 文档在场）+ T006 质量门与 T002 mutation 记录复用
- 遗留：3 Minor（诚实边界/后续档/采样缺口，见 st-cases.md 报告表），无 Critical/Major
- 用户预授权：全流程无需逐步确认，ST Go 自动执行归档

### design 门控
- 审查子代理：general（ses_eca6666c9ffeqJu3ZgWIFy5X0d，首轮 + 复核同代理）
- 首轮结论：**FAIL**——6 组事实核查全部属实（含深查裁定：钩子 A 处 parent cache 一致性/divs None 不可达论证/steps_since_recapture 初始化属 srs 未定义处的合理补全/方案取舍成立），1 Major：**Major-1** §4.4/§10 浮点恒等式 `ceil(ratio·n)==target` 作为"恒成立"命题不真（审查代理实测反例 n=25/target=7→ceil(0.28·25)=8；违例只上溢 ≤1 故正确性安全，但本 AR 全部 2 幂几何下"碰巧精确、论证错"）→ 修复：§4.4 改"按实际容量决策"防御行（prospective==C 跳过，置于构造前不付 capture 成本）+ 2 幂精确性限定 + §6 夹具 2 幂约束明文（UT N·T=64、基准 N·T∈{2048,4096}）+ §10 风险表对应改界
- Minor-1~11 全修：①§3 保留 `_require_captured()`+type-ignore②§5.2 声明 import③工件键名 `adaptation_events`（对齐 srs）④公开探针 property `graph_adaptive_policy`（基准/UT 不触下划线属性）⑤退役标志命名映射说明（srs"graph_adaptive_retired"落地为策略内 _retired 状态 + mff 侧告警布尔两半）⑥窗口 append 后 trim（内存有界，与"上限"措辞一致）⑦§6/§7 时序推演修正（UT-1602 重捕获实际第 2 次调用断言放第 3 次；§7 稳态自第 4 步 warmup=5 保守）+ 残缺字符串修复⑧pytestmark 改逐函数 `@_cuda` 标记（模块级会连带 skip CPU 可跑的 UT-1601）⑨UT-1601 方向单调显式断言⑩tasks.md T002 config.py→config_manager.py⑪`_observe_adaptive_capacity` 删 step_idx 死参
- 复核（第二轮，单点）：**PASS**——Major-1 三处收口且与浮点事实吻合（两层 noop 判定职责划分验证无缝：增长路径防御行永不触发、收缩仅 target==C-1 且 +1 上溢时触发、跳过时策略状态自洽零伪 events）；11 Minor 逐项落实；带入 2 个新 Minor（§4.4 片段 `runner` 未绑定 → 改传 `current_capacity` 形参；`import math` 补注）**当场修入 design**；
- **最终结论：PASS（可进入 T001 Red）**

### req 门控
- 审查子代理：general（ses_eca849cbaffex34TunrCeZAZgM，首轮 + 复核同代理）
- 首轮结论：**FAIL**——14 项事实核查 13 属实（1 项部分属实：仓库无 `config.py`，字段在 `actfold/utils/config_manager.py`），3 Major：**Major-1** 增长分支调用点与 FR3"成功重放后才运行策略"互斥（超限步 :845-855 丢弃后直接 return None，策略永远看不到超限事件——ratio 极小场景步步超限→增长分支不可达，UT-1603 字面不可实施）→ 修复：调用点拆两处（增长评估在超限分支内丢弃前 / 收缩评估在发布后返回前）；**Major-2** 观测序列维护责任未定义（策略纯函数但窗口谁 append、重捕获后是否清空、退役后是否冻结全未写）→ 修复：FR3 观测窗口契约小节（append 条件=replay 返回非 None、重捕获后清空+间隔归零、退役后冻结、关闭不累积）；**Major-3** 退役标志与 `_graph_batched_capture_failed` 关系未钉死（复用后者会连废旧 runner）且 UT-1605"结果一致"无法区分 eager 回退 → 修复：独立标志 `graph_adaptive_retired` + 明文禁止置位后者 + UT-1605 补 `is` 对象断言 + 重放 spy 探针 + mutation ④ 同步更新
- Minor-1~10 全修：①config.py→config_manager.py + 构造参数流入（:396-402 先例）②术语表"新图 capture 失败不弃旧图"③FR3 统一"发布后"（发布消费旧 runner 静态缓冲）④收缩公式括号 + C≤1 no-op⑤退役告警调用侧发出（策略保持纯函数）⑥独立一次性告警标志（不走家族布尔）⑦内存算术补 child_buf/logits（单 runner ≈570MB、swap ≈1.1GB、三路径 <3GB）⑧平稳性降格为跨配置均值证据（D 散布 ~17%）⑨无 runner 场景 + UT-1603 告警不复位断言⑩行号 :298-313
- 复核（第二轮）：**PASS**——3 Major 逐项收口且与代码事实逐行吻合（:845-855/:288-295/:760-764），10 Minor 全修，四个风险面（调用点↔触发条件、陈旧值↔append 条件、UT-1603 断言↔调用点、术语表↔正文）无新矛盾；两微瑕 N-1（UT-1603 括注措辞）/N-2（570MB 加总）当场修复；tasks.md 无需改动（任务粒度未变）
- **最终结论：PASS（可进入 design）**
