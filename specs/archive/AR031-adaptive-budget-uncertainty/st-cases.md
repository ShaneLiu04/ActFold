# [AR031] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR031 层级自适应重算预算 + 不确定性排序验证 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-22 |

**执行环境**：Windows 11 / Quadro RTX 5000 16GB WDDM / Python 3.11.9 / torch 2.5.1+cu121；真实面基准使用 AR019 venv（transformers==4.53.1）+ 本地 Fast-dLLM-v2-1.5B 权重。所有自动化证据均为 2026-10-21/22 本轮实测（review 修复轮后的最终态：全量 **1445 passed / 4 skipped / 3 deselected**，AR031 套件 **69 测试**，demo 三锚点精确命中，mypy actfold 75 文件 + scripts 29 文件 0 error，black/isort/pyflakes 全过）。

## 测试用例列表

### ST-001：FR1 层自适应重算预算策略全链（AC1-AC5）

**关联需求：** srs.md §3.1 FR1（AC1 预热均匀 / AC2 守恒重估 / AC3 cap 上界 / AC4 非配额 / AC5 无数据退化）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given `LayerRecomputeBudgetPolicy`（ratio∈(0,1]、深度 L、warmup/window 参数）构造合法

**测试步骤：**
1. warmup 窗口内多步前向，查逐层预算表（UT-3102）
2. 注入分化稳定度观测 s={0.9,0.5}，跨窗口边界重估（UT-3103）；多随机种子验证 `Σ B_l == max(B_total, L')` 全路径（UT-3104）
3. 构造 D_l=8 > B_l=5 无并列分数场景跑 cap（UT-3110）；D_l=3 ≤ B_l=5（UT-3111）
4. profiler 数据全缺失下前向（UT-3106）

**期望结果：**
- Then 预热逐位均匀；重估权重 ∝ (1-s+ε) 且低稳定层更高；cap 后重算数 == min(D_l, B_l) 且被折叠者恰为相似度最高者；D<B 时零强制翻转；无数据退化全均匀无异常。

**实际结果：** UT-3102/3103/3104/3105/3106/3110/3111 全部 passed（`tests/test_ar031_budget_policy.py`，本轮 27 passed 套件内）。

**状态：** PASS

---

### ST-002：FR1 AC6 可用性守卫优先于预算（端到端）

**关联需求：** srs.md §3.1 FR1 AC6（父页缺失 × 预算已满）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 真实 `PagedFoldCache` + `ManualFoldedForward` + 绑定预算；父分支已缓存后其层 1 页被 demote 为冷页（availability 全 False）

**测试步骤：**
1. 绑定预算前向 child（单 token 散度构造，隔离守卫面）
2. 断言 observe 记账 `[(0,7,8),(1,0,8)]`（UT-3169）
3. 与无预算同洞参考路径逐位比对 + NaN 扫描

**期望结果：**
- Then 不可用行经 `_apply_cache_availability` 翻回重算、豁免 `B_l`（层 1 观测 0，错序 cap 则为 7）；输出与无预算路径逐位一致；零 NaN（错序折叠会读冷页 NaN sentinel）。

**实际结果：** UT-3169 passed（review r1 Important 修复新增，端到端真实缓存/MFF/冷页；UT-3113 纯张量单元面同过）。

**状态：** PASS

---

### ST-003：FR2 预算 × 图路径联动

**关联需求：** srs.md §3.2 FR2（AC1 捕获体零同步 / AC2 容量冲突回退 / AC3 无 policy 零回归）
**测试类型：** 正常路径 + 边界条件
**优先级：** High

**前置条件：**
- Given CUDA 可用时捕获 + 重放；不可用时合成面 + graph 逻辑单测（design §6.1.2 环境注记）

**测试步骤：**
1. 探针窗口断言捕获体与 `_static_forward` 执行窗内 `.item()/.tolist()/.cpu()` == 0（UT-3114）
2. 构造 `Σ B_l > C` 重放（UT-3115）
3. 不携带 policy 捕获+重放与 AR030 末态 sentinel 比对（UT-3117）；graph 静态体消费位置级 `[T]` forced（UT-3165，seed 4335）
4. fused gate 旁路（UT-3120）与 AR017 shrink/growth 联动方向性（UT-3116/3119）

**期望结果：**
- Then 捕获体零新增主机同步；容量冲突回退 eager 且 degradation-fallback +1、输出正确；无 policy 逐位零回归；1-D forced 与显式 `[1,T]` 位相同。

**实际结果：** UT-3114~3120 + UT-3165 全部 passed（`tests/test_ar031_graph_budget.py`，1027 行套件本轮全绿）。

**状态：** PASS

---

### ST-004：FR3 不确定性感知候选验证器（PSFM 面）

**关联需求：** srs.md §3.3 FR3（AC1 确定性 tie-break / AC2 早停 / AC3 单条件不足 / AC4 stats 守恒）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 真实 `PagedFoldCache` + `fork_rollouts` + `UncertaintyRolloutValidator`（margin=0.9, margin_gap=0.05）

**测试步骤：**
1. 全同先验验证，记录 validate_fn 调用序（UT-3121）
2. 首候选大幅领先场景（UT-3122）；margin 满足 gap 不满足（UT-3124）
3. 跳过者 rollout_stats/availability/页表与 fork 后一致（UT-3123）
4. 落败者关闭、胜者保留、base 冻结、stats 跨调用守恒（UT-3125/3131/3132）

**期望结果：**
- Then tie-break 为 id 升序；早停 validated==1/skipped==N-1 且胜者与全量一致；单条件不足全量验证；跳过者零写；`validated + skipped == N` 每调用成立；落败者 `close_candidate`、胜者页保留、rollout 保持打开。

**实际结果：** UT-3121~3133 全部 passed（`tests/test_ar031_uncertainty_validator.py`，真实 COW 缓存回滚断言含）。

**状态：** PASS

---

### ST-005：FR4 speculative_core opt-in 接入（三 samplers）

**关联需求：** srs.md §3.4 FR4（AC1 默认零回归 / AC2 开启未早停 / AC3 开启早停等价 / AC4 True+N=0）
**测试类型：** 正常路径 + 异常处理
**优先级：** High

**前置条件：**
- Given LLaDA/Dream/Fast-dLLM 三 samplers 共享 `plan_uncertainty_order`/`should_early_stop` 核（inline parity E4）

**测试步骤：**
1. 三 samplers 默认配置输出与 AR023 冻结 sentinels 逐位比对（UT-3134）
2. 开启未早停：胜者/提交/`_folding_parent` 与关闭态逐位一致（UT-3135）
3. 先验分化场景触发早停（UT-3136）；跳过行零突变（UT-3140）
4. inline parity + 种子公式不动 + greedy 探针不变（UT-3141/3137/3138）
5. True+N=0 / margin 越界构造期 `ValueError` 三家族对称（UT-3139）

**期望结果：**
- Then 默认逐位零回归；仅顺序变化；早停胜者 == 全量 argmax；配置校验对称拒绝。

**实际结果：** UT-3134~3144 全部 passed（`tests/test_ar031_sampler_uncertainty.py`，1571 行套件本轮全绿）。

**状态：** PASS

---

### ST-006：FR5 Fast-dLLM 真实面三臂基准（正式工件判定复核）

**关联需求：** srs.md §3.5 FR5（AC1 quick 冒烟 / AC2 工件完备可复算 / AC3 honest negative 原样）
**测试类型：** 正常路径 + 回归测试
**优先级：** High

**前置条件：**
- Given AR019 venv + 本地权重；判定函数冻结先于数据收集；预承诺单次运行

**测试步骤：**
1. quick 冒烟（单 prompt/reps=3/64 token，~3min，`_quick.json` 独立工件不入库）
2. 正式面（reps=30、num_tokens=128、2 prompts、seed 2026、~12min、峰值 ~5.8GB）
3. 从工件自身记录的输入复算 h1/h2/h3/h4（判定函数纯函数可复算）
4. 核对 honest negative：H3 refuted 与 overall refuted 原样保留

**期望结果：**
- Then 工件含三臂全指标 + H1-H4 + reproduction + meta；reproduction False ⇒ 全 blocked 一票否决；数字不四舍五入；refuted 原样交付零改写。

**实际结果：** formal 工件 `results/optimization/ar031_adaptive_uncertainty_bench.json`（`bd2eaa5` 入库）：**H1 supported**（posthoc gap 0.0/0.0、长度 34/66 恢复、块级 exact ≥ off）；**H2 supported**（6.2690×，17279.30 vs 2756.30ms，Δ=14522.99 > 2σ=3535.60）；**H3 REFUTED 诚实负结果**（1.0004×，Δ=1.15ms << 2σ=1464.68ms——cap 恒不绑定）；**H4 supported**（等价 1.0、违规 0、repro 6/6）；**overall refuted** 纯由 H3 触发，原样入库。第一次 formal 全 blocked 为 bench 仪器 bug（forced 形状 + repro 配对），修复后单次重跑，两次运行均留档 tasks.md。判定面 UT-3153~3160 + UT-3166 全部 passed。

**状态：** PASS

---

### ST-007：FR5 环境门与 quick 隔离

**关联需求：** srs.md §3.5 FR5 AC1（quick 隔离）+ 环境门
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- Given transformers 版本失配或权重缺失环境

**测试步骤：**
1. 环境门失败运行（`require_environment`：transformers==4.53.1 + 权重目录 + config.json）
2. `--quick` 运行核对工件路径
3. `_resolve_model_path` OSError 路径

**期望结果：**
- Then rc=2、零工件落盘；quick 只写 `_quick.json` 绝不触正式路径。

**实际结果：** UT-3158/3159 passed（quick 隔离断言 + 环境门 rc=2 不落盘断言；T012 真实 quick 冒烟 ~3min 实跑通过留档）。

**状态：** PASS

---

### ST-008：FR6 配置/接线/导出/文档

**关联需求：** srs.md §3.6 FR6（AC1 YAML 消费 / AC2 默认零回归 / AC3 公共导出）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given `ActFoldConfig` 七键（layer_budget_* 四键 + uncertainty_* 三键）consumption-only

**测试步骤：**
1. 7 新 YAML 键经 `load_config` 消费（UT-3145）
2. 默认配置 runner/MFF 零回归探针（UT-3147/3149/3151）
3. 符号导入 + `scores` 公共面三度量等价（UT-3146/3150）
4. runner 深度 resolver：显式 > config，不可解析 raise（UT-3148）；跨设备观测拒绝（UT-3152）
5. 位置级 forced `[T]` eager 阻塞点广播（UT-3167）；exact-type gate 拒绝（UT-3168，design E9）

**期望结果：**
- Then 无 unknown-key 告警；默认全关零回归；公共符号可导入；深度不猜。

**实际结果：** UT-3145~3152 + UT-3167/3168 全部 passed（`tests/test_ar031_wiring.py` 本轮 14 passed）。

**状态：** PASS

---

### ST-009：异常场景面（design §6.4 全 11 行）

**关联需求：** srs.md §3.3/§3.4 异常约束 + design §6.4
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given design §6.4 异常场景表 11 行

**测试步骤：**
1. 逐行核对覆盖 UT：构造非法 3108/3127/3139；深度失配 3148；validate_fn 三类失败 3128/3129；预算不足/容量冲突 3128/3115；NaN 3162/3130；forced×binding 3163；空/零长 3107/3161/3126；B_total<L' 域 3104；设备/环境 3152/3159；窗口原子性 3112；closed-rollout 前置 3164
2. 全部异常断言零状态突变（`before`/`_assert_stats_unchanged`/缓存 `assert_invariants` 模式）

**期望结果：**
- Then 每类异常显式拒绝且零副作用。

**实际结果：** 11 行全覆盖全部 passed（design §6.4 表逐行映射，本轮 AR031 套件 69 测试全绿）。

**状态：** PASS

---

### ST-010：全量回归 + demo 不变量 + 静态质量门

**关联需求：** srs.md §4 NFR（零回归、可复算、质量门）+ design §6.3 回归面
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given AR030 末态基线（1443 passed）与 demo 三锚点（85.5% / 2.35e-03 / 93.75%）

**测试步骤：**
1. 全量 pytest
2. `python -X utf8 demo.py`
3. lint 三件套 + mypy 双面

**期望结果：**
- Then 全量零既有失败；demo 三锚点逐字命中；mypy 0 error。

**实际结果：** 全量 **1445 passed / 4 skipped / 3 deselected（76.60s，1443 基线 + UT-3168/3169 两新增，零既有失败）**；demo **85.5% / MSE 2.35e-03 [HIGH] / stable ratio 93.75%** 逐字命中；black 195 files unchanged / isort rc=0 / pyflakes 零输出 / mypy actfold 75 文件 + scripts 29 文件 0 error；AR031 套件 69 测试 ~9s < 30s 门。

**状态：** PASS

---

### ST-011：真实 checkpoint graph 联动面（WDDM 环境限制）

**关联需求：** srs.md §3.2 FR2（真实面 graph 联动）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given 16GB WDDM 工作站（#46 环境法则：真实面 eager 主声明；真实 graph 联动面为 BLOCKED-DEFERRED）

**测试步骤：**
1. graph 联动面以 UT + 合成面覆盖（UT-3114~3120/3165）
2. 工件 meta 核对 BLOCKED-DEFERRED 注记

**期望结果：**
- Then 真实 graph 联动面在 ≥24GiB 安静 GPU/Linux-TCC 环境补做；当前以合成面 + 逻辑单测覆盖，工件 meta 诚实记录。

**实际结果：** BLOCKED-DEFERRED（环境约束非缺陷，AR025/AR029/AR030 同先例）；合成面与逻辑单测全绿，工件 meta 含限制注记。

**状态：** BLOCKED-DEFERRED

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-22 |
| 执行结果 | PASS（10 PASS + 1 BLOCKED-DEFERRED，环境约束如实记录非缺陷） |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR1 | 层自适应重算预算 | ST-001, ST-002 | PASS |
| §3.2 FR2 | 预算 × 图路径联动 | ST-003, ST-011 | PASS（真实 graph 面 BLOCKED-DEFERRED） |
| §3.3 FR3 | 不确定性感知验证器 | ST-004, ST-009 | PASS |
| §3.4 FR4 | speculative_core opt-in | ST-005, ST-009 | PASS |
| §3.5 FR5 | Fast-dLLM 真实面三臂基准 | ST-006, ST-007 | PASS |
| §3.6 FR6 | 配置/接线/导出 | ST-008 | PASS |
| §4 NFR | 零回归/可复算/质量门 | ST-010 | PASS |

**需求覆盖率：** 6/6 FR + NFR（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 5 | 5 | 0 | 0 |
| 边界条件 | 3 | 2 | 0 | 1（BLOCKED-DEFERRED） |
| 异常处理 | 2 | 2 | 0 | 0 |
| 回归测试 | 1 | 1 | 0 | 0 |
| **合计** | **11** | **10** | **0** | **1** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| - | 真实 checkpoint graph 联动面需 ≥24GiB 安静 GPU（WDDM #46 限制） | BLOCKED-DEFERRED：UT + 合成面覆盖，工件 meta 记录，高速算力机补做 |
| - | H3 机制归因诚实 refuted（cap 在 tau=0.95 + forced 活跃块下恒不绑定） | 非缺陷：honest negative 原样交付，AR031 定位安全/正则化面（重算硬上界 + 质量零损），绝不作为加速 rung 引用 |

### 结论

> **Go**。FR1-FR6 + NFR 100% 覆盖，10/11 用例 PASS、0 FAIL，唯一 BLOCKED-DEFERRED 为环境约束（高速算力机补做，先例一致）。全量 1445 passed 零回归、demo 三锚点零漂移、mypy 双面 0 error、formal 工件三臂判定可复算且 honest negative 原样入库。review 门 r2 PASS（r1 两 Important + 三 Minor 全部修复闭环）。AR031 作为安全/正则化档交付（ladder 第 12 档）。
