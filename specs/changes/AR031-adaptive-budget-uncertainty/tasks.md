# [AR031] 任务清单

**AR031：层自适应重算预算 + 不确定性感知候选验证**（srs.md 同目录；本文件为唯一任务真源）

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 设计：`design.md`（预算公式与分配/排名冻结、强制 stable 语义、窗口重估时序、AR017 钩子接线点、早停双条件判定规则、胜者等价验收设计、三臂基准协议、#29/#37/#44/#45/AR030 边界表、**可用性守卫优先于预算的顺序冻结（#48/#49 行）**、UT 矩阵）+ 设计门控 PASS | srs | done | 先于任何编码 |
| T002 | Red UT（预算策略面）：warmup 均匀预算逐位、窗口重估按 (1-s_l) 权重、Σ B_l == B_total、低稳定层预算更高、D_l > B_l 强制 stable、D_l ≤ B_l 上界语义不凑配额、round 余数补齐确定性、profiler 关闭退化均匀、tokens_l == 0、budget_ratio/W 非法参数 ValueError、深度不匹配 ValueError、**窗口内预算表不变（可复现）**、**可用性守卫优先于预算：不可用行翻回重算且豁免 B_l（#48/#49 × eager）**、**forced/NaN 豁免（UT-3162/3163）**、**接口边界 layer_idx 越界/形状/N==0 早退（UT-3161）** | T001 | done | 2026-10-21 完成（Red 子代理 ses_ea35c55a4ffeX022X9C0qwJIVK；接口用法修正见勘误 E7） |
| T003 | Green `actfold/core/layer_budget.py`：`LayerRecomputeBudgetPolicy`（窗口重估、权重向量冻结、`adaptive` 构造参数——False=永久均匀供 adaptive-off 臂、逐调用绝对预算派成、余数补齐）+ `apply_layer_budget`（sort+gather 精确计数 cap，eager/graph 统一）；模块顶层 stdlib+torch | T002 | done | 2026-10-21 完成（含 budget_table 显式面，勘误 E7） |
| T004 | Red UT（图联动面）：捕获体内零主机同步探针、Σ B_l 突破 C 回退 eager + canary 计数、预算漂移触发 AR017 observe/重捕获、不携带 policy 图路径与 AR030 末态逐位一致（零回归）、forced 掩码捕获体外计算 | T003 | done | UT-3114~3120 |
| T005 | Green `ManualFoldedForward` graph×budget 联动：预算掩码重放期数据（AR021 forced 同型）、AR017 growth/shrink 钩子消费预算需求、opt-in 零侵入 | T004 | done | #37 捕获契约不动 |
| T006 | Red UT（验证器面）：全同分数 tie-break id 序全量验证、margin+gap 双满足早停 validated==1/skipped==N-1、胜者与全量验证逐位一致、被跳过候选页表/availability 不变、单条件不满足全量验证、stats 计数守恒 validated+skipped==N、N==0/margin 非法/候选 id 冲突 ValueError、预算不足 RolloutBudgetError 原子上抛、单候选验证失败原子回滚（AR030 三类失败语义） | T001 | done | UT-3121~3133, UT-3164；可与 T003 并行（只依赖 PSFM 既有面） |
| T007 | Green `actfold/speculative/uncertainty_validator.py`：`UncertaintyRolloutValidator`（min-prob 降序、margin 早停双条件、PSFM fork/rollout_put/close_rollout 集成、stats 探针） | T006 | done | PSFM 语义消费者，不改页预算/淘汰 |
| T008 | Red UT（samplers 接入面）：默认 False 三 samplers 与 AR023 末态逐位一致（sentinels）、开启不触发早停胜者/提交逐位一致、开启触发早停胜者等价 UT 构造场景、种子公式/贪心行/greedy_win_stats 不变、True+N=0 构造期 ValueError、score/commit 段跳过计数 | T007 | done | UT-3134~3144 |
| T009 | Green `speculative_core.py` opt-in `uncertainty_validation_order`（共享纯函数 `plan_uncertainty_order`/`should_early_stop` + segment B 评分段早停；#44/#45 冻结公式与提交语义不动）+ `fast_dllm_sampler.py` inline 路径接入同一 helper（design 勘误 E4：否则 FR5 真实面测不到该机制）+ 三 samplers 配置字段与 `__post_init__` 校验 + `uncertainty_stats` 探针 | T008 | done | 默认关逐字节不变 |
| T010 | Green 接线/配置/导出/零回归：`ActFoldConfig` 7 新键 YAML 无告警（#5）、`actfold.core`/`actfold.speculative` 导出、默认配置全仓零回归、BenchmarkRunner 构建面 | T003, T005, T009 | done | UT-3145~3152 |
| T011 | 机制基准 `scripts/ar031_adaptive_uncertainty_bench.py`：三臂（官方 baseline / adaptive-off / AR031 全开）、H1-H4 冻结判定先于数据收集、reps=30 预承诺单次、round-robin 交错、#39 计时、AR019 venv、`--quick` 独立工件；判定函数 UT | T010 | pending | UT-3153~3160；eager 主声明面 |
| T012 | 真实面运行：quick 冒烟先行（quick < 60 s 时长门）→ Fast-dLLM v2-1.5B 正式 reps=30 单次运行（本机 16GB，AR022 先例可行）；honest negative 原样交付 | T011 | pending | 预承诺协议；绝不 rerun-until-pass |
| T013 | 文档草稿：README 六节全量重写（创新表新行/ladder 新 rung）、RERUN 步骤 19、AGENTS #50 契约、GUIDE、CHANGELOG、ALGORITHM/EXPERIMENTS | T011, T012 | pending | 数字全部来自工件 |
| T014 | 质量门：black/isort/pyflakes/mypy(actfold 73 + scripts 28 `--explicit-package-bases`)/全量 pytest/demo 三锚点；**AR031 套件 < 30 s 时长门** | T010, T011 | pending | 只增不减；产出最终测试计数 |
| T015 | 文档终稿：用 T014 最终计数刷新 README 质量段与 RERUN 计数，复核六节一致 | T013, T014 | pending | 避免 README 过期 |
| T016 | review 门控（只读子代理：spec/design 合规 + 代码审查 + 安全路径复核）循环至 PASS | T015 | pending | 记录轮次/代理 id |
| T017 | ST 验收：st-cases.md（FR1-FR6 全覆盖）+ 执行（正常/边界/异常/回归）；归档 + gitee 分段推送与树校验 | T016 | pending | 真实 graph 联动面记录为限制 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 约束（进度记录）

- 不改折叠数学、#29 门控契约、#37 图捕获契约、#44/#45 冻结种子公式与 min-prob
  胜者规则、AR030 PSFM 页预算/淘汰语义。
- 预算机制是门控之后的预算分配（相似度排名 + 强制 stable），不是新门控；
  与 FoldingScheduler 组合合法（#29 互斥不触发）。
- margin/margin_gap/budget_ratio 为冻结常数（YAML 可配、预承诺运行不得调参）。
- 胜者等价是预承诺集上的经验硬门（理论边界诚实记录），不一致 → honest refuted。
- 全程零 mock；基准为真实张量运算与真实 checkpoint（Fast-dLLM v2-1.5B）。
- 真实面 eager 主声明；真实 graph 联动面记录为限制（WDDM，#46 环境法则）。
- 用户已授权自动化 harnessX（每步无需确认、每步输出反馈文字）。

## 阶段门控记录

### 2026-10-21 需求门控第 1 轮 — PASS（代理 ses_ea3e73673ffeoOfdLK8R4M0v0W）- 门控结果：PASS（第 1 轮）；G1-G9 无 Critical/Important NO。
- 契约交叉验证：#29/#37/#44/#45/AR017/AR030/#5/#39/#46 引用全部无误；
  H2 2σ 口径经 `scripts/ar017_adaptive_bench.py:262-264` 实码验证；
  `specs/component-detail-design/` 不存在已实证；AR022 reps=30 先例支撑可行性。
- Minor WARN×2 已即时修复：
  - G4：冻结「可用性守卫优先于预算」子句（不可用行恒重算且豁免 `B_l` 上界，
    #48/#49 顺序不变；srs FR1 期望行为 + 新增验收 Given/When/Then + FR2 注记；
    T001 边界表补 #48/#49 行；T002 补 UT 项）。
  - G8：NFR 时长门落任务（T014 备注套件 < 30 s；T012 备注注 quick < 60 s；
    T002 补「窗口内预算表不变」断言）。

### 2026-10-21 设计门控第 1 轮 — FAIL（代理 ses_ea3c262e6ffe1GSPHzwaBA2JYR）

- 失败项：G9 NO（改动清单遗漏 `cuda_graph.py`——graph 捕获体掩码序在
  `FoldedGraphRunner._static_forward`，folded_transformer 967 行实为多父 eager）；
  G8/G12/G13/G15 WARN。技术专项：A1 算法 forced×budget 折叠漏洞（Critical P1）。
- 问题清单：P1 Critical（forced 位高分最先被折叠，破坏 #43 构造性重算语义，
  FR5 主线必触发）；P2 Important（cuda_graph.py 落点遗漏）；P3 Important
  （round 余数算术不成立：Σ round 可超 B_total、B_total<L' 时 Σ==B_total 不可达）；
  P4 Important（NaN×binding 破 ≤B_l 不变量；N==0 clamp 保护说法错误）；
  P5-P10 Minor（全 NaN best 初始化、私有字典直探、CUDA softmax 1-ULP 断言过强、
  A3 措辞矛盾、覆盖率目标缺失、接口测试软点、三臂联合归因措辞）。
- 修复（全部 10 项即时闭合）：
  - P1：A1 重写——`exempt = stable | forced | isnan(scores)` 构造性豁免（恒重算、
    豁免 B_l 记账，与守卫行同类）；不变量改为「非豁免重算 ≤ B_l；总重算 ≤
    B_l + 豁免数」；UT-3163（forced 永不折叠）。
  - P2：§1.1/§4.4.1/A4 补入 `cuda_graph.py`（runner 构造面 budget_static +
    _static_forward cap + count 重导出对 cap 后掩码）；三路径表述修正
    （串行/批量同父/批量多父 eager + graph 捕获体）；tasks T005 更新。
  - P3：A2 重写为 floor+最大余数法 + 上界 clamp 确定性削减 + B_total<L' 冻结
    条款；最终不变量 Σ B_l == max(B_total, L')；srs FR1 分配规则与 AC2 同步修订
    （勘误 E5）；UT-3104 判据更新。
  - P4：NaN 豁免并入 exempt（UT-3162）；N==0 显式早退、删除 clamp 保护说法
    （UT-3161）。
  - P5：A5 伪码 cid_best 初始化 order[0]（全 NaN 语义与 UT-3130 一致）；归属
    校验改公共探针 rollout_stats。
  - P6：softmax 一致性改为「CPU 构造 + CUDA UT 双面钉住」（UT-3135）；prior
    伪码清理为冻结公式。
  - P7：A3 统一为「捕获体内零观测；eager 掩码序 + graph 发布循环（体外）观测」。
  - P8：§6.1 头部声明覆盖率目标（F×UT 全行 + srs AC 全追溯 + <30s 门，仓库
    惯例）；UT-3102 判据修正。
  - P9：§6.2 映射修正 + 显式 UT-3161/3164。
  - P10：srs §1 改「联合归因」（勘误 E6）。
- 验证成立面（复审对照）：勘误 E1-E4 成立；A5 API 匹配（close_candidate 对零写
  跳过者合法、rollback 语义一致）；A6 与实码匹配（argmax 首现一致、inline 必要、
  探针放宿主可行）；A7 三臂仅差两机制开关、H4 全测可行。
- UT 总数：59 → 63（3102~3164）。

### 2026-10-21 设计门控第 2 轮 — FAIL（代理 ses_ea3a82cc2ffexJ1B4rygKfEelU）

- 结果：10 项修复中 7 项完全闭合（P4-P10）；P1/P2/P3/P9 部分闭合，残留
  Important×3 + Minor×3。
- 残留问题与修复（全部即时闭合）：
  - N1 Important（§4.2.3 流程图未随 A1 修订，按图实现会重现 P1 漏洞）：
    流程图重写为 r2 A1 冻结形态（exempt 构造 / pool / binding / flip 恒不含
    forced/NaN/stable 位注记 + N==0 早退分支）。
  - N2 Important（修复记录与执行不符）：§4.4.1 清单补入 `cuda_graph.py`
    （budget_static + _static_forward cap + mask_buf 写回 + fused 旁路）；
    §1.1 头部计数修正（新增 2 模块 + 1 bench 脚本 / 修改 10 文件）。
  - N3 Important（A2 clamp 调和算术两个反例下不变量不可达）：
    A2 重写为**分域规则**——域 1 `B_total < L'` → tokens_l>0 层全 1（Σ==L'）；
    域 2 floor+最大余数 → 上界盈余按 w' 降序 round-robin **再分配**（非「扣还」）
    → 下限抬升（0 层 +1、B_l 最大层扣 1、下限 1，鸽笼保证可达）；两域统一不变量
    Σ B_l == max(B_total, L')；srs FR1/E5/UT-3104/3105 判据同步。
  - N4 Minor（UT-3141 缺 ValueError 断言）：判据补
    `plan_uncertainty_order` 空/非 1-D 输入 `ValueError` 显式断言。
  - N5 Minor（A4 mask_buf 写回缺失）：A4 补「cap 后掩码写回 mask_buf（count/
    merge/发布循环同源）」；UT-3118 判据补 graph 面观测 == cap 后计数。
  - N6 Minor（UT-3114 探针窗口口径）：判据限定捕获体与 _static_forward 执行窗，
    重放路径允许既有 validate_budgets 单次读回（#37 契约），与 srs FR2 AC1 对齐。
- 验证成立面（复审对照）：A1 重写被确认修复 P1（构造性豁免 + thr 并列安全方向）；
  A5 API 面、A6 双侧接入、A7 三臂协议均维持闭合。

### 2026-10-21 设计门控第 3 轮 — FAIL（代理 ses_ea38146a2ffe4SQhChj7ai2nE5）

- 结果：N1-N4 闭合；**N5/N6 未闭合**——根因定位：r2 修复用 python `str.replace`
  无断言，UT-3114/3118 两处替换**静默失败**（修复记录与执行不符，与 N2 同型
  失败模式第二次出现）。G9/G10/G11 NO；G1-G8/G12-G14 PASS；G15 WARN。
- 新问题 P11-P17 与修复（全部闭合，**本轮起每处替换带落盘断言**，
  14 项校验全过）：
  - P11 Important（`actfold/eval/benchmark_runner.py` 缺席 §1.1/§4.4.1 改动
    清单）：两处补行（构建面实例化 policy + 深度 resolver，AR029 同型）；
    头部计数 10 → 11。
  - P12 Important（=N6 根因）：UT-3114 判据重写——探针窗口限定捕获体与
    `_static_forward` 执行窗；重放期允许既有 `validate_budgets` 单次读回
    （#37 契约）；与 srs FR2 AC1 对齐。
  - P13 Minor（=N5 根因）：UT-3118 判据补「发布循环 `observe_layer` 收到
    cap 后计数（== cap 后掩码规约值）」，A4 交叉引用闭合。
  - P14 Minor：F3 措辞改「三处 eager（串行/批量同父/批量多父 AR025）+
    graph 捕获体（A4）」。
  - P15 Minor：§4.4.1 补双 `__init__.py` 导出行。
  - P16 Minor：`apply_layer_budget` 返回注改「正常路径新张量；N==0 早退
    返回原张量」（与 UT-3161 一致）。
  - P17 Minor：UT-3102 判据限定「等 tokens_l 构造下逐层差异 ≤ 1」。
  - 修复过程自纠：UT-3114 重写时误引入重复 3115 行，即时发现删除。
- 过程教训入库：**文档批量修订一律 `assert old in src` + 修订后关键字断言**，
  杜绝「记录说改了、文件没改」。

### 2026-10-21 设计门控第 4 轮 — PASS（代理 ses_ea36be17cffeRUu24oIxsqMn4X）

- 结果：P11-P17 全部 CLOSED；N5/N6 终态 CLOSED；G1-G15 全 PASS；零新增问题。
- 复审自验亮点：G7 不变量反例（盈余再分配 headroom ≥ C、下限抬升鸽笼净零、
  域 1 无 clamp 违例）与 G4/G5/G8 实码行号级核实（folded_transformer.py:363-399/
  650-657/967-969；cuda_graph.py:365/382；paged_fold_cache.py:997-1140）。
- G10 正则核验：UT-3102~3164 共 63 行，无重复无跳号。
- **确认：design.md（r4）达到「可直接指导 T002 Red UT 编写」完备度。**
- T001 关闭（design.md + 设计门控 4 轮 PASS）。

## 进度记录

### 2026-10-21 T010 完成（配置/导出/runner 接线 Red→Green）

- T010 Red（子代理 `ses_ea1a64593ffeK6NdPz8MMd3elQ`）：`tests/test_ar031_wiring.py`
  （8 测试，UT-3145~3152）；Red 确认 = UT-3145 unknown-key 告警 + UT-3146
  ImportError（无导出）+ UT-3147/3148/3149 AttributeError（无 7 键/无
  `_resolve_budget_depth`/`_layer_budget_policy`）；UT-3150/3151/3152 三面 T003/T005
  已交付故 Red 期即绿（5 fail / 3 pass）。
- T010 Green 四文件：
  - `config_manager.py`：`ActFoldConfig` 7 新键（consumption-only，#44
    `candidate_include_greedy` 先例）：`layer_budget_enabled=False`/
    `layer_budget_ratio=0.15`/`layer_budget_warmup=5`/`layer_budget_window=10`/
    `uncertainty_validation_order=False`/`uncertainty_margin=0.9`/
    `uncertainty_margin_gap=0.05`；`load_config` 无 unknown-key 告警（#5）。
  - `actfold/core/__init__.py`：导出 `LayerRecomputeBudgetPolicy` +
    `apply_layer_budget`（design 文件图双符号）。
  - `actfold/speculative/__init__.py`：导出 `UncertaintyRolloutValidator`。
  - `benchmark_runner.py`：`_resolve_budget_depth()`（AR029 `_resolve_pyramid_depth`
    同型：adapter/raw 二级探测，显式 `num_layers` > `config.num_hidden_layers`，
    不可解析 ValueError「real layer count」不猜）+ `_layer_budget_policy()`
    （enabled=False 先于深度解析返回 None——默认关零状态；True 时按 config 四键
    构造 policy）接入 `_build_folded_model` 的 `ManualFoldedForward(...,
    budget_policy=...)`；既有 `except RuntimeError` 不吞 policy 的 ValueError
    （MFF 深度校验面经 runner 原样上抛，UT-3148(d)）。
- 判据面：UT-3145 YAML 7 键非默认值无告警 + 默认值精确断言；UT-3146 双包导出 +
  `__all__`；UT-3147 默认关 = AR030 末态（7 键默认/policy None/MFF `_budget_policy`
  None）；UT-3148 显式 4>3 深度优先、不可解析 raise、`_build_folded_model` 接出
  policy num_layers==2、深度失配 3≠2 ValueError；UT-3149 enabled=False →
  `_layer_budget_policy()` None + MFF 探针 None、enabled=True 不可解析 raise；
  UT-3150 三度量（cosine/l2/pearson）`forward == scores > tau` 逐位 + NaN 位恒
  divergent；UT-3151 `budget_policy=None` 与无 kwarg 逐位一致 + ratio=1.0 不绑定
  面逐位一致；UT-3152 meta 设备观测 ValueError + CPU 累加器未污染。
- 质量门：black/isort/pyflakes 0、mypy actfold 75 文件 + scripts 28 文件 0 error、
  AR031 套件 **56 测试 9.63s**（<30s 门）、全量 **1432 passed / 4 skipped /
  3 deselected**（1424 + 8 新增，零回归）。
- 实现注记：runner 面只接 budget（uncertainty 三键的运行字段在家族 sampler
  config 上，T009 已交付；bench 侧直接构造 sampler config，AR022/AR023 先例）；
  提交 `53cb10e`。
### 2026-10-21 T008/T009 完成（samplers 不确定性早停面 Red→Green）

- T008 Red：`tests/test_ar031_sampler_uncertainty.py`（11 测试，UT-3134~3144；
  LLaDA/Dream/FastDLLM 三家族，Hadamard 后验 trap fixture 族），Red 确认 =
  三家族配置 `TypeError`（无 uncertainty 字段）+ `ImportError`
  （speculative_core 无 helper）+ `AttributeError`（无 uncertainty_stats 探针）。
- T009 Green 四文件接入：
  - `speculative_core.py`：模块级共享纯函数 `plan_uncertainty_order`（降序、
    tie → 较小行号、空/非 1-D ValueError）+ `should_early_stop`（margin+gap
    精确合取，NaN 恒 False）；segment B opt-in 分支 `_uncertainty_order_select`
    （冻结先验 = fill 域 softmax(draft_logits_region)@cand 的 min；逐行 softmax
    评分、跳过行不评分；胜者 tie 语义 = 向量化 argmax 首现）；mixin
    `uncertainty_stats` 探针 + 三计数器注解（宿主 `__init__` 初始化，无
    `__init__` 契约不破）；
  - `fast_dllm_sampler.py`：inline 面 `_uncertainty_select_inline` 消费同一共享
    helper（E4；函数内 import 规避 `_MAX_FOLD_CTX_ENTRIES` 循环依赖），先验同
    公式，逐行后验保持本面冻结 whole-small-block min-prob 域；自有
    `uncertainty_stats` 探针 + 计数器；
  - `llada_sampler.py` / `dream_sampler.py`：三配置字段
    （`uncertainty_validation_order=False`/`uncertainty_margin=0.9`/
    `uncertainty_margin_gap=0.05`）+ `__post_init__` 校验（True+N=0
    ValueError；margin/gap ∉ [0,1] ValueError，闭边界 0.0/1.0 合法、无条件
    路径无关）+ 计数器初始化。
- 判据面：UT-3134 默认关三家族逐位一致（raw/N=0/N=2 六面）；UT-3135 margin=1.0
  不早停 = off 逐位（含 history/_folding_parent）；UT-3136 trap 早停 {1,1,1} 且
  胜者 == 全量 argmax、链切换 `_cand0`；UT-3137 segment A 构造不变
  （canvas/branch_ids 逐位）；UT-3138 greedy_win_stats 不变；UT-3139 三家族
  对称校验；UT-3140 直接 commit 跳过 EOS 行不触 finished；UT-3141 helper 面 +
  core/inline parity（inline 8×{1,1,1}={8,8,8}、8 批调用逐块胜者 == 全量
  argmax、on/off 序列逐位）；UT-3142 joint 逐链独立 {3,1,1} 且与 solo 逐位
  一致；UT-3143 gap 孪生钉 MIN-over-fill 先验与停机谓词（0.4261 过 0.40 不过
  0.45，mean/max-over-fill 与升序 mutant 均败）；UT-3144 探针只读/跨调用存活/
  赋值 AttributeError/返回 dict 突变不泄漏。
- 质量门：black/isort/pyflakes 0、mypy actfold 75 文件 0 error、AR031 套件
  6.1s（<30s 门）、全量 **1424 passed / 4 skipped / 3 deselected**（1413 + 11
  新增，零回归）。
- 实现注记：早停检查仅在剩余未验证行非空时评估（remaining 空 = 自然全量，
  early_stops 不计）；早停只重排评分段，`[N,T]` 前向批形状/种子公式/贪心锚/
  min-prob 胜者规则/提交段全部不动（#44/#45 冻结面保持）。

### 2026-10-21 T006/T007 完成（验证器面 Red→Green）

- T006 Red：`tests/test_ar031_uncertainty_validator.py`（14 测试，UT-3121~3133 +
  UT-3164；零 mock，全部真实 `PagedFoldCache` + `fork_rollouts`，延迟 import 使
  Red = ModuleNotFoundError）。子代理 `ses_ea2625baefferXsJ78xbxyC9yO`。
- T007 Green：`actfold/speculative/uncertainty_validator.py`
  `UncertaintyRolloutValidator`：
  - 先验降序 + id 升序 tie-break 定序；严格 `>` 更新（tie 保持首个验证者 = 最高
    先验/最小 id；NaN 恒 False 永不胜；全 NaN 保持 order[0]）；
  - 早停双条件：`q_best >= margin` 且 `q_best - max(remaining priors) >=
    margin_gap`（remaining 空自然全量，不计 early_stop）；
  - 归属探针走公共面 `cache.rollout_stats(rollout)`（外族/已关 rollout 经
    `_rollout_state` ValueError 转译；活跃候选集 = stats 键集）；
  - E2 收口：全部落败者 `close_candidate`（含跳过者），胜者保留、rollout 打开、
    base 冻结，`close_rollout` 归调用方；
  - 失败面：validate_fn 异常 → `rollback_candidate`(失败候选) + 原样 re-raise，
    stats 零计数、零关闭（AR030 原子语义）；
  - stats：validated/skipped/early_stops 累计 + winner_scores FIFO 100。
- Green 阶段发现（1 项，Red 期算术笔误）：UT-3125 终值 `validated_count == 110`
  与其自身循环内守恒断言（`5 + (i+1)`，109+1=110）矛盾 → 修正为 109（判据
  语义零变化，修复注记写入测试）。
- 质量门：black/isort/pyflakes 0、mypy **75 文件** 0 error（新模块 +1）、
  AR031 套件（3 文件 37 测试）~15s（<30s 门）、全量 **1413 passed / 4 skipped /
  3 deselected**（1399 + 14 新增，零回归）。

### 2026-10-21 T004/T005 完成（图联动面 Red→Green）

- T004 Red：`tests/test_ar031_graph_budget.py`（7 测试，UT-3114~3120；CPU 收集面 +
  CUDA 门控面），Red 确认 = `FoldedGraphRunner.__init__` 拒绝 `budget_static`
  （TypeError）。
- T005 Green 五文件接线：
  - `similarity_gate.py`：`scores()` 只读面，`forward ≡ scores() > tau`（UT-3150
    三度量等价由同源构造保证）；
  - `layer_budget.py`：`num_layers` property / `begin_step` 重估清 budget_for 缓存 /
    `observe_layer` 跨设备 ValueError（UT-3152 预置）；
  - `cuda_graph.py`：runner `budget_static` 构造面 + property；捕获体 scores 旁路
    （#38）→ forced AND → A1 cap → 单次 count 规约（UT-3114：count==mask.sum()）；
  - `folded_transformer.py`：serial/batched/multi 三处 eager 插入（cap 在 forced AND
    后、守卫前；observe 在守卫后、profiler 前；层总量面、device sum 复用给 count）；
  - `architecture_utils.py`：MFF `budget_policy` 构造面（深度校验）+ forward 入口
    begin_step + budget_table 暖缓存 + 规则 (a) eager kwargs 门控 + 三 runner 构造点
    budget_static + replay 前体外回填 + 两侧 publish 层总量 observe。
- **勘误 E8**：A3 层总量澄清（per-row 字面弃用，冻结 UT-3118 优先）；A4 规则 (a)
  （图降级/结构回退 eager 一律不加预算，UT-3115 钉住）；F3 变长步与多父混合/无效行
  分支豁免 cap（安全方向）。
- Green 阶段发现（三项，均记入 E8）：① `sorted_asc[idx]`（0-dim 张量索引）内部
  host 回读、捕获期非法 → `torch.gather` 等价替换（CUDA_LAUNCH_BLOCKING 定位，
  UT-3110/3111 判据不变）；② runner `mask_buf`/`count_buf` property 放宽为「须已
  分配」（UT-2603/3114 直驱面，生产语义不变）；③ UT-3120 CUDA 面 fixture 索引越界
  （Red 期被更早 TypeError 遮蔽、从未执行）→ 6 个改动位置修正入 [0,1024)（判据零
  变化）。
- 质量门：black/isort/pyflakes 0、mypy 74 文件 0 error（scripts 28 文件 0 error）、
  AR031 套件 8.1s（<30s 门）、全量 **1399 passed / 4 skipped / 3 deselected**
  （1392 + 7 新增，零回归）。

### 2026-10-21 T002/T003 完成（预算策略面 Red→Green）

- T002 Red：`tests/test_ar031_budget_policy.py`（16 测试函数覆盖 UT-3102~3113 +
  3161~3163），Red 确认 = ModuleNotFoundError。
- T003 Green：`actfold/core/layer_budget.py`（A1 sort+gather cap / A2 分域分配 /
  A3 零同步观测），16/16 通过。
- **Green 阶段发现 + 勘误 E7**：单层 `budget_for(l, tokens_l)` 信息不足，A2 需完整
  tokens 向量 → 显式新增 `budget_table(tokens)` 公共面（design §4.3.1 已补行）；
  Red 测试 `_budget_table` helper 改经 budget_table 面（测试判据零变化，接口用法
  修正）；UT-3102 补 budget_for 缓存查表钉住断言。
- Green 调试记录：首跑 15/16——`_reestimate` 漏除 `acc_tokens`（用原始计数而非
  稳定率 → 负效用 → 权重反转），单点修复后全绿（1 次尝试内闭合）。
- 质量门：black/isort 重排通过、pyflakes 0、mypy 74 文件 0 error、全量
  **1392 passed / 4 skipped / 3 deselected**（1376 + 16 新增，零回归）。

### 2026-10-21 设计完成（design.md draft-1）

- 领域知识重建：`SimilarityGate` 只产掩码（→ 新增 `scores()` 只读面）；topk 动态 k
  不可捕获（→ sort+gather 阈值算法，eager/graph 统一）；`fast_dllm_sampler` 为
  inline 实现（→ 共享 helper 双侧接入，勘误 E4）；`close_rollout` 释放全部候选页
  含胜者（→ FR3 收口改为「关闭落败者、保留胜者」，勘误 E2）；变长 T → 权重向量
  窗口冻结 + 逐调用绝对派生（勘误 E3）；graph 预算面 = `budget_static` 重放期
  体外 `copy_` + 捕获体内纯设备 cap（勘误 E1）。
- 方案裁定（3 方案对比，自动化授权）：阈值近似面（精确计数不可保证）与分批分段
  验证（未早停时双批形破坏逐位等价，#41 1-ULP 先例）被否决；采用统一 sort+gather
  精确计数 + samplers 评分段早停（前向批不变）+ PSFM 面前向级节省。
- Step 7 组件详设：`specs/component-detail-design/` 不存在 → 按规则跳过（边界真源
  为 AGENTS.md 契约）。
- T003/T009 任务描述已按设计对齐（adaptive 参数 / fast_dllm inline 接入）。
