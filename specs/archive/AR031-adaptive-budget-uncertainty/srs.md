# [AR031] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR031 |
| AR 主题 | 层自适应重算预算 + 不确定性感知候选验证（baseline 超越型，基于 AR030 PSFM 基础设施） |
| 关联 SR | 无独立 SR（延续 AR017/AR021-AR023/AR030 机制线） |
| 日期 | 2026-10-21 |
| 状态 | Draft |
| 前置 | AR030 已归档（PSFM 基础设施可用：`PagedFoldCache`/`fork_rollouts`/`StabilityProfiler` 逐层数据/`AdaptiveCapacityPolicy` observe 钩子） |
| 组件详设 | `specs/component-detail-design/` 不存在（组件详设缺失；以 AGENTS.md #29/#37/#41-#45/#48/#49 契约为边界真源） |

## 1. 背景与目标

两条已冻结机制线在「固定分配」上留有共同的收益缺口：

1. **折叠门控按全局 tau / 全局容量分配**：AR016 工件实测逐层 stable_ratio 分化明显
   （FFN 重算占比 D_frac ≈13-16% 为全局平均，掩盖层间差异），固定 tau 对高稳定层
   过保守（浪费折叠机会）、对低稳定层过激进（语义安全网承压）；AR017 自适应容量
   只调整**图容量**（divergent 容量 C），不改变**层间**分配。
2. **候选验证顺序固定、全部验证**：AR021-AR023 的 speculative block verification
   与 Agent/RSI rollout（AR030 PSFM）中，N 个候选**全部**前向验证后按冻结规则选
   胜者——无论候选间不确定性差异多大，验证成本恒为 O(N)。AR022 已证明贪心锚定
   候选（greedy row）常胜（`greedy_win_stats`），但验证开销未利用该先验。

**目标**（baseline 超越型，非机制 AR）：在 Fast-dLLM v2-1.5B 真实 checkpoint 上，
对官方家族 baseline 声明**质量对齐 + p50 加速 > 1.0** 双指标；同时通过
「自适应关闭同配置」对照臂**联合归因**两个自适应机制的贡献（H3；逐机制分解由
report-only 探针部分弥补，勘误 E6）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- `actfold/core/layer_budget.py`（新）：`LayerRecomputeBudgetPolicy` —— 基于逐层
  实测稳定率的全局重算 token 预算层间分配；窗口式重估。
- `actfold/speculative/uncertainty_validator.py`（新）：基于 PSFM
  `fork_rollouts` 的不确定性感知候选验证器（min-prob 降序 + margin 早停）。
- `actfold/models/speculative_core.py`（改）：opt-in 不确定性验证顺序接入
  （默认关 = AR023 末态逐字节不变）。
- 图路径联动：预算掩码为重放期数据；预算需求变化经 AR017
  `AdaptiveCapacityPolicy` observe/重捕获钩子联动（UT + 合成面钉住）。
- `scripts/ar031_adaptive_uncertainty_bench.py`（新）：Fast-dLLM v2-1.5B 真实面
  双 A/B 基准（官方 baseline / adaptive-off 同配置 / AR031 全开）。
- 配置/YAML/导出/文档（README/RERUN/AGENTS #50/CHANGELOG/GUIDE）。

**Out of Scope（本 AR 不做的）：**

- 不修改折叠数学、`SimilarityGate`/`AdaptiveQuantileGate` 本体与 #29 互斥契约
  （预算机制是**门控之后的预算分配**，不改变门控判定本身）。
- 不修改 #44/#45 冻结种子公式与 min-prob 胜者选择规则（早停只在「可证剩余候选
  不胜」时触发，胜者与全量验证等价为硬验收）。
- 不修改 AR030 PSFM 页预算/淘汰语义（验证器是 PSFM 的**消费者**）。
- LLaDA/Dream 真实面（仅 Fast-dLLM；三 samplers 共享核的接入由 UT 钉住语义，
  真实面外推为后续 AR）。
- 真实 checkpoint 的 CUDA graph 联动面（本机 WDDM 限制；UT + 合成面钉住，
  真实 graph 面记录为限制，参照 #46 环境法则）。

## 3. 功能需求

### 3.1 FR1 — 层自适应重算预算（`LayerRecomputeBudgetPolicy`）

**描述：** 将全局重算 token 预算 `B_total` 按各层实测不稳定度在层间分配；层内按
相似度排名取预算内 token 折叠、其余重算；预算窗口式重估，预热期用均匀分配。

**触发条件：** 折叠前向（`ManualFoldedForward` 层循环）携带 policy 且 profiler
数据就绪；每个重估窗口边界触发重估。

**期望行为：**
- **数据源**：`StabilityProfiler` 逐层 `stable_sum`/`num_tokens` 累计值（AR018
  `record_precomputed` 同源）；窗口内层稳定率 `s_l = stable_l / tokens_l`。
- **分配规则（冻结；勘误 E5 修订）**：`B_total = round(budget_ratio × Σ_l tokens_l)`
  （`budget_ratio` YAML 可配，预承诺运行不得调参）；层权重
  `w_l = (1 - s_l + ε) / Σ_j (1 - s_j + ε)`（`ε = 1e-6` 防全稳定除零）；
  分配为**分域规则**（design A2 r3）：`B_total < L'`（L' = 有 token 层数）时
  tokens_l > 0 层 `B_l = 1`（安全下限域，Σ == L'）；否则 **floor + 最大余数法**
  （`B_l = floor(B_total × w'_l)`，w' = L' 层归一；余数按 w' 降序逐层 +1）→
  上界 clamp（截到 tokens_l，盈余按 w' 降序 round-robin 再分配给有 headroom 层）→
  下限抬升（0 层 +1、从 B_l 最大层扣 1，下限 1）；最终不变量
  **`Σ B_l == max(B_total, L')`**（两域统一，鸽笼/floor 性质保证可达）。
- **层内选择（冻结）**：重算预算语义为**重算上限**。门控判 divergent 的 token
  默认全部重算（#29 语义安全网不动）；若某层**非豁免** divergent 数超过层预算
  `B_l`，则按门控相似度**升序**保留最不相似的前 `B_l` 个 token 重算、其余
  divergent token **强制 stable**（折叠，复用 parent 激活）——强制 stable 是质量
  风险面，由 FR5 质量验收钉住；若非豁免 divergent 数 `≤ B_l`，**不得**为凑预算把
  stable token 强制 divergent（预算是上界不是配额）。**豁免位恒重算且豁免
  `B_l` 记账**：`forced_divergent` 位（#43 构造性重算语义，分数高也必须重算）、
  NaN 相似度分数位、可用性守卫翻回位——三类豁免与预算正交，语义安全网永不因
  预算失效。
- **可用性守卫优先于预算（冻结）**：预算的强制 stable 掩码应用后，
  `_apply_cache_availability`（#48/#49 既有守卫，顺序不变：forced AND 之后、
  profiler 之前）仍可将父页缺失的不可用行翻回 divergent（重算）——守卫优先于
  预算，守卫翻回的重算行**豁免** `B_l` 上界（`B_l` 是可用性满足前提下的软上限；
  语义安全网永不因预算失效）。graph 路径不受影响（pyramid/paged × graph 均
  构造期拒绝，#48/#49）。
- **窗口式重估（冻结）**：预热窗口 `W_warm`（YAML 可配，默认 5）步内用均匀预算
  `B_l = B_total / L`；此后每 `W_est`（默认 10）步用最近窗口的层稳定率重估一次；
  重估原子替换预算表；窗口内预算冻结不变（可复现）。
- **零数据退化**：profiler 未启用或某层无数据 → 该层用均匀预算，policy 不抛错、
  不猜测（与 #48「绝不猜」一致）；`tokens_l == 0` 的层 `B_l = 0`。
- **开销**：重估 O(L)；每步每层 O(1) 查表 + 既有门控排名开销（排名本就存在）。

**异常处理：** `budget_ratio` ∉ (0, 1] → 构造期 `ValueError`；`W_warm`/`W_est` < 1
→ `ValueError`；深度与模型层数不一致 → `ValueError`（绝不静默截断）。

**验收标准：**
- Given 预热窗口内任意步，When 前向，Then 每层预算 == 均匀预算（逐位）。
- Given 窗口边界且各层稳定率分化（构造 s = {0.9, 0.5}），When 重估，Then
  `B_l` 严格按 `(1-s_l)` 权重分配（floor+最大余数法）、
  `Σ B_l == max(B_total, L')`、低稳定层预算更高。
- Given 门控 divergent 数超过层预算 `D_l > B_l`，When 前向，Then 该层**非豁免**
  重算 token 数 ≤ `B_l`（无并列时 == `B_l`；强制 stable 生效）、forced/NaN/守卫
  豁免位恒重算、输出与全量重算参考在强制 stable 位上的偏差由 FR5 质量面验收。
- Given `D_l < B_l`，When 前向，Then 重算数 == `D_l`（上界语义，不凑配额）。
- Given profiler 关闭，When 前向，Then 均匀预算、无异常、输出与固定预算路径
  逐位一致。
- Given 父页缺失（pyramid/paged × eager 合法组合）且预算已满触发强制 stable，
  When 前向，Then 不可用行经 `_apply_cache_availability` 翻回重算、豁免 `B_l`
  上界、输出与无预算路径逐位一致（守卫优先于预算）。

### 3.2 FR2 — 预算 × 图路径联动（AR017 钩子）

**描述：** 预算机制在 CUDA graph 路径下作为**重放期数据**生效（AR021
`forced_divergent_static` 模式），总重算需求经 `AdaptiveCapacityPolicy` 的
observe 钩子联动。

**触发条件：** `ManualFoldedForward(use_cuda_graph=True)` 且携带 budget policy。

**期望行为：**
- 强制 stable / divergent 决策在**捕获体外**计算、以重放期张量传入（AR021
  forced 掩码同型）；捕获体内部零主机同步（#37 契约不动）。
- 每层重算需求 `B_l` 之和受图 divergent 容量 `C` 约束：重放时 `Σ last_layer_divergences`
  突破 `C` → 按既有预算超限语义回退 eager（AR017 validate_budgets 语义不变）。
- 预算窗口重估导致需求漂移 → AR017 growth/shrink 钩子按既有节流/上限语义重捕获
  （`max_recaptures`/`min_recapture_interval` 不改）。
- policy 与 graph 的组合是 **opt-in**：不携带 policy 的 graph 路径逐字节不变。
- 可用性守卫优先序与 FR1 冻结子句一致（graph×pyramid/paged 均构造期拒绝，
  本条仅约束 eager 面；重放期预算掩码不削弱任何既有守卫）。

**异常处理：** 容量冲突回退 eager（非错误）；重捕获失败仅退役自适应路径
（AR017 语义）。

**验收标准：**
- Given 携带 policy 的图路径重放，Then 捕获体内 `.item()/.tolist()/.cpu()` 调用
  数 == 0（UT 探针）。
- Given 预算重估使 `Σ B_l > C`，When 重放，Then 回退 eager、输出正确、
  `graph_pool_stats` 的 degradation-fallback 计数 +1（canary 语义不变）。
- Given 不携带 policy，When 图捕获+重放，Then 与 AR030 末态逐位一致（零回归）。

### 3.3 FR3 — 不确定性感知候选验证器（PSFM 独立面）

**描述：** 基于 `PagedFoldCache.fork_rollouts` 的验证器：按 min-prob 不确定性
降序验证候选，margin 早停；被跳过候选的页保持 PSFM 共享（无状态丢失）。

**触发条件：** `UncertaintyRolloutValidator(cache, margin, ...)`；对 N 个候选
执行验证。

**期望行为：**
- **不确定性分数（冻结）**：复用 #45 min-prob 语义——候选画布 still-masked/
  待验证位置的最小 token 概率（由首遍 forward 或 parent logits 计算，度量与
  #45 完全同源）；分数越高=越可信=**后验**胜者概率越大 → 验证顺序为分数
  **降序**。
- **margin 早停（冻结）**：每验证完一个候选得其后验质量分 `q_j`（冻结为该候选
  全量验证后的 min-prob）；若 `q_j ≥ margin` 且 `q_j` 严格大于所有**未验证**
  候选的**不确定性分数上界**——注意：未验证候选分数是先验，不能构成严格上界，
  因此早停判定规则冻结为**双条件**：`q_j ≥ margin` **且** `q_j - max(未验证
  分数) ≥ margin_gap`（`margin_gap` YAML 可配，预承诺不得调参）；两阈值均为
  冻结常数。触发则跳过剩余验证。
- **胜者等价（硬验收）**：早停时选中的胜者必须与全量验证胜者一致；该等价在
  预承诺运行集上 100%（不一致 → honest refuted，绝不静默放行）。理论边界诚实
  记录：先验分数不构成严格占优证明，等价是**预承诺集上的经验硬门**。
- **PSFM 集成**：候选经 `fork_rollouts` 隔离；验证写经 `rollout_put`；被跳过
  候选不做任何写（其页与 base/胜者共享不变）；验证器结束 `close_rollout`；
  单候选验证失败按 AR030 三类失败语义原子回滚。
- **统计探针**：`validator.stats()` 只读：`validated_count/skipped_count/
  early_stops/winner_scores`——跨调用存活（#44 `greedy_win_stats` 先例）。

**异常处理：** N == 0 → `ValueError`；margin/margin_gap ∉ [0, 1] → 构造期
`ValueError`；候选 id 冲突/未知 → `ValueError`（PSFM 语义）；预算不足 →
`RolloutBudgetError` 原子上抛（不部分验证）。

**验收标准：**
- Given 全部候选不确定性相同，When 验证，Then 顺序为候选 id 序（确定性 tie-break）、
  全量验证、无早停。
- Given 构造「首候选大幅领先」场景（margin + margin_gap 双满足），When 验证，
  Then `validated_count == 1`、`skipped_count == N-1`、胜者与全量验证逐位一致、
  被跳过候选页表/availability 不变。
- Given 早停条件不满足（margin 或 gap 任一不满足），Then 全量验证。
- Given 验证器 stats，Then 全部计数非负且 `validated_count + skipped_count == N`。

### 3.4 FR4 — speculative_core opt-in 接入（三 samplers 共享核）

**描述：** `speculative_core` 的 speculative block verification 增加 opt-in
不确定性验证顺序；默认关闭 = AR023 末态逐字节不变。

**触发条件：** `FastDLLMSamplerConfig`/`LLaDASamplerConfig`/`DreamSamplerConfig`
新增 `uncertainty_validation_order: bool = False`；`ActFoldConfig` 消费同名
YAML 键（#5）。

**期望行为：**
- 默认 `False`：三 samplers 全路径逐字节不变（UT 钉住与 AR023 冻结面 sentinels）。
- 开启后：候选构建（#44/#45 冻结种子公式**不动**）→ 计算不确定性分数 → 按分数
  降序逐候选 `_speculative_score_commit`，margin 早停（FR3 同一双条件规则、同参
  语义）→ 提交语义不变（`min-prob` 胜者、chain continuation、commit skip 全部
  #45 原语义）。
- 早停跳过的候选**不进入** `_speculative_score_commit`（省一次区域切片与评分），
  但其画布行已构建（构建成本不可省——种子公式决定行内容）；节省的是**评分与
  提交段**的成本。收益来源诚实记录：验证顺序 + 早停减少的评分开销，而非候选
  构建开销。
- `greedy_win_stats`/`record_rollout_masks` 等既有探针语义不变。

**异常处理：** `uncertainty_validation_order=True` 且 `num_block_candidates == 0`
→ 构造期 `ValueError`（顺序仅在 speculative 路径有定义，#44 同型）。

**验收标准：**
- Given 默认配置（False），Then 三 samplers 输出与 AR023 末态逐位一致。
- Given 开启且不触发早停，Then 胜者/提交状态与关闭时逐位一致（仅顺序变化，
  全量验证下顺序不影响结果）。
- Given 开启且触发早停，Then 胜者与全量验证一致（UT 构造场景钉住）、被跳过
  候选不计入 `_greedy_spec_calls` 之外的任何状态突变。
- Given True + N=0，Then 构造期 `ValueError`。

### 3.5 FR5 — Fast-dLLM 真实面双 A/B 基准

**描述：** Fast-dLLM v2-1.5B 真实 checkpoint 上，官方 baseline / adaptive-off
同配置 / AR031 全开三臂对照，双指标硬标准，reps ≥ 30 预承诺单次运行。

**触发条件：** `python -m scripts.ar031_adaptive_uncertainty_bench [--quick]`。

**期望行为：**
- **三臂**：① official baseline（家族官方 recipe，`folded_model=None`）；
  ② adaptive-off（ActFold 同配置：均匀预算 + 全量验证 + 其余与 ③ 完全一致）；
  ③ AR031 全开（层自适应预算 + 不确定性验证顺序）。② 与 ③ 的差异**仅**为两
  机制开关（机制归因臂）。
- **硬标准（冻结）**：
  - **H1 质量对齐（③ vs ①）**：posthoc overall confidence gap ≤ 1e-3（沿
    AR023 H1 边界比较口径与 `_H1_RATIO_EPS` IEEE 噪声余量）、decode length
    每个 prompt 恢复、merged block exact-match 不低于 ②；任一 `None` → refuted。
  - **H2 p50 加速（③ vs ①）**：`p50(①)/p50(③) > 1.0` 严格成立（2σ 阈值沿
    AR017 两样本检验口径）。
  - **H3 机制归因（③ vs ②）**：`p50(②)/p50(③) > 1.0`（自适应机制本身带来
    加速，非配置差异）；report-only 记录逐臂 stable ratios / 验证跳过率。
  - **H4 诚实账本**：胜者等价率 == 100%（早停触发的样本中胜者与全量验证一致）；
    预算上界不突破；canary/ledger 沿 #45 `LEDGER_C` 口径；reproduction 双跑
    行严格相等（不一致 ⇒ 全部 blocked）。
- **协议**：冻结判定先于数据收集（脚本内冻结）；reps=30 预承诺单次；
  round-robin 交错；warmup 丢弃；#39 计时口径；AR019 venv（transformers==4.53.1）；
  `--quick` 写独立 `_quick.json`，绝不写正式路径；真实 graph 面记录为限制
  （eager 主声明面）。
- **仲裁**：质量或加速任一硬标准不满足 → honest negative 原样交付（refuted/
  inconclusive），绝不调参重跑。

**异常处理：** 环境不满足（transformers 版本/显存/权重缺失）→ 构造期显式报错
退出，不写部分工件；单臂失败 → blocked 工件 + 探针交叉引用 + 零伪造数字。

**验收标准：**
- Given quick，Then 冒烟通过并写 `_quick.json`（几何缩小版，同一三臂/判定）。
- Given 正式运行完成，Then 工件含三臂全指标、H1-H4 判定、逐样本记录；每个数字
  可由工件重算。
- Given 任一硬标准 refuted，Then 工件原样记录（honest negative 交付）。

### 3.6 FR6 — 配置/接线/导出/文档

**描述：** 配置面、公共导出与文档收口。

**期望行为：**
- `ActFoldConfig` 新键（默认关闭 = 零回归）：`layer_budget_enabled: bool = False`、
  `layer_budget_ratio: float = 0.15`（初始默认沿 AR016 实测 D_frac 区间上沿，
  预承诺运行显式给定）、`layer_budget_warmup: int = 5`、`layer_budget_window: int = 10`、
  `uncertainty_validation_order: bool = False`、`uncertainty_margin: float = 0.9`、
  `uncertainty_margin_gap: float = 0.05`；`load_config` 无 unknown-key 警告（#5）。
- `actfold.core` 导出 `LayerRecomputeBudgetPolicy`；`actfold.speculative` 导出
  `UncertaintyRolloutValidator`；构建失败不返回半初始化对象。
- AGENTS.md 新增 #50 契约条目；README/RERUN（步骤 19）/CHANGELOG/GUIDE 全量
  更新；README 六节全量重写规则（AR 收口时）。

**验收标准：**
- Given 7 个新 YAML 键，Then 字段正确消费、无 unknown-key 警告。
- Given 默认配置，Then 全仓行为与 AR030 末态逐位一致（零回归 UT）。
- Given 符号导入，Then `LayerRecomputeBudgetPolicy`/`UncertaintyRolloutValidator`
  可从公共路径导入。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | 默认零回归 | 默认配置下与 AR030 末态逐位一致（全量 pytest + 定向 UT） |
| 开销 | policy 重估 | O(L)/窗口；每步每层 O(1) 查表；图路径零新增主机同步 |
| 速度 | 验证跳过收益 | 早停触发时 `validated_count < N`；收益诚实记录（省评分/提交段，不省构建） |
| 可复现 | 窗口冻结 | 窗口内预算/顺序不变；同输入同种子 → 同输出 |
| 性能门 | 测试时长 | AR031 单测套件 < 30 s；bench quick < 60 s |
| 协议 | 预承诺 | 判定冻结于数据收集前；单次运行；honest negative 原样交付 |
| 环境 | 真实面 | AR019 venv / transformers==4.53.1 / Fast-dLLM v2-1.5B 权重 / 16GB 显存上限内 |

## 5. 约束与假设

**约束：**
- 不改折叠数学、#29 门控契约、#37 图捕获契约、#44/#45 冻结公式与胜者规则、
  AR030 PSFM 页预算/淘汰语义。
- 全程零 mock；基准为真实张量运算与真实 checkpoint。
- 本机 16GB WDDM：真实面 eager 主声明；真实 graph 联动面为记录限制。
- 与 `folded_transformer.py` 同文件协作沿用 #49 谨慎原则（预算机制经
  `ManualFoldedForward` 参数面接入，最小化侵入）。

**假设：**
- 逐层稳定率在窗口尺度上平稳（窗口式重估的前提；分布漂移由下一窗口自然吸收）。
- min-prob 先验分数与后验胜者质量正相关（margin_gap 双条件即为该假设的诚实
  风险控制；等价由预承诺集硬门验收）。
- Fast-dLLM v2-1.5B 在 16GB 上 reps=30 可行（AR022 先例实测可行）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 层预算 `B_l` | 第 l 层单步重算 token 上界（门控 divergent 恒重算；预算只约束「强制 stable」的折叠面） |
| 强制 stable | 门控判 divergent 但层预算已满 → 该 token 折叠（复用 parent 激活）；质量风险面，FR5 验收 |
| 不确定性分数 | 候选待验证位置 min-prob（#45 同源度量）；降序 = 最可信优先验证 |
| margin 早停 | 双条件：`q_j ≥ margin` 且 `q_j - max(未验证分数) ≥ margin_gap`；两阈值冻结常数 |
| 胜者等价 | 早停胜者 == 全量验证胜者；预承诺集上 100% 硬门（理论边界诚实记录） |
| adaptive-off 臂 | ActFold 同配置、均匀预算 + 全量验证（机制归因对照） |

## 7. 变更记录

| 版本 | 日期 | 变更 |
|------|------|------|
| draft-1 | 2026-10-21 | 初稿（需求澄清 3 轮：实验面/机制边界/早停语义/图联动/度量/阈值冻结均按用户授权自动裁定） |
