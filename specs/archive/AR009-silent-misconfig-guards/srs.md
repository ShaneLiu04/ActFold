# [AR009] 需求设计说明书 — 折叠静默失效防护（silent-misconfiguration guards）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR009 |
| AR 主题 | 折叠未生效/被错误配置时的可见性：embedding 代理降级一次性告警 + AdaptiveQuantileGate×FoldingScheduler 构造期显式拒绝 |
| 关联 SR | 指南 6.2 :239（"`AdaptiveQuantileGate` 与 `FoldingScheduler` 互斥但无检测：`FoldedTransformerLayer` 应显式拒绝组合"）与 :240（"auto-discovery 失败时静默降级为 embedding 代理测量，用户不知道 folding 没生效"）；AGENTS #10（不静默回退）/#29（gate×scheduler 互斥现仅文档约定） |
| 日期 | 2026-10-12 |
| 状态 | Final（req/design 门控 PASS，T001–T004 passing，review PASS，ST 10/10 Go，已归档） |

## 1. 背景与目标

两处"用户以为 folding 在工作、实际没有"的静默失效：

1. **代理降级无告警**：`ActFoldVerificationEngine._estimate_stable_ratio`
   （verification_engine.py:275-295）在 profiler 无层级数据时回退到
   layer-0 embedding 相似度代理。该回退本身是既有设计（AGENTS #16/#17：
   adapter 无 `folded_model`、或 base model 不传播 branch kwargs（#18）时
   folding 不激活），但**零告警**——调用方拿到 `stable_ratio` 数字却不知道
   它不反映真实折叠行为，误导 ablation/验收结论。
2. **互斥组合无防护**：`AdaptiveQuantileGate` 按分位数选 top-k，不读固定
   `tau`（AGENTS #29 "Do not combine it with `FoldingScheduler`, which
   overwrites `gate.tau`"）。该约定目前**只有文档**：`FoldedTransformerLayer`
   构造器与 `ActFoldVerificationEngine` 构造器都接受
   `gate=AdaptiveQuantileGate, scheduler=FoldingScheduler(...)` 组合并静默
   运行——scheduler 的动态 tau 全部无效，分位数语义被无声破坏。

指南 :240 的另一半（`find_layer_list` 静默降级）经核实**已过时**：
`detect_architecture`（architecture_utils.py:281-285）对层栈发现失败已
显式 `RuntimeError`，符合 #10；`nn.Sequential`/自定义容器不支持属布局
限制（#23 建议显式 wiring），不在本 AR 范围。

目标：让折叠失效**可见**（一次性 UserWarning）、让错误配置**早失败**
（构造期 RuntimeError），两处防护与 AGENTS #10/#29 对齐并落为代码契约。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `ActFoldVerificationEngine._estimate_stable_ratio` 的 embedding 代理回退
  路径：进入回退时发一次性 `UserWarning`（engine 实例级 once 标志，参照
  AR002 `_warn_graph_degraded_once` 先例），消息含成因提示（无 folded
  前向/kwargs 未传播 → AGENTS #17/#18 指针）；
- `FoldedTransformerLayer.__init__`：`scheduler is not None` 且
  `isinstance(gate, AdaptiveQuantileGate)` → `RuntimeError`（子类
  `SplitFoldedTransformerLayer` 经 super 自动覆盖）；
- `ActFoldVerificationEngine.__init__`：同一组合同样 `RuntimeError`
  （engine 回退路径 :285-291 同样调 `gate.set_tau(tau)` 动态覆写）；
- 上述行为的单元测试（告警恰好一次 / 实例级语义 / 层级路径零告警 /
  三类守卫触发 + 合法组合对照）；每条守卫用例内含合法组合对照断言；
- 文档：指南 6.2 :239 与 :240 **两项均勾选**（:240 附前提过时的勘误
  说明——detect_architecture 已显式 raise，本 AR 收口的是 engine 代理
  回退可见性）+ 第十六部分回链、AGENTS #29 修订为代码强制、CHANGELOG
  AR009 节。

**Out of Scope（本 AR 不做的）：**
- `find_layer_list` 对 `nn.Sequential`/自定义容器的支持（#23：显式
  wiring 是既定出路，检测失败已 raise 非 silent）；
- `folding_status` 枚举 API（指南建议的形态；现 raise + warning 已满足
  #10 可见性，枚举属新公共 API，无消费方，暂不做）；
- 其他 gate/scheduler 参数校验（`disabled_layers`、tau 范围等既有校验
  不动）；
- engine 代理回退本身的语义（保留 fallback 行为，只加可见性——#16/#17
  写明的合法降级路径）。

## 3. 功能需求

### 3.1 embedding 代理回退的一次性告警

**描述：** verification engine 在 stable ratio 只能由 embedding 代理估出
时明确告知调用方。

**触发条件：** `_estimate_stable_ratio` 中 `profile is None or
not profile.layer_stats`（即进入 embedding 回退分支）。

**期望行为：**
- 首次进入回退分支：发 `UserWarning`，消息包含 (a) 本次 stable_ratio 为
  layer-0 embedding 代理、不反映真实折叠；(b) 常见成因提示（adapter 未
  挂 `folded_model`（AGENTS #17）或 base model 未传播 branch kwargs
  （AGENTS #18））；
- 同一 engine 实例后续回退：**不再重复告警**（实例级 once 布尔，参照
  AR002 `_graph_degraded_warned` 先例）；
- 新 engine 实例：重新告警一次（once 是实例级非全局级）；
- 层级 profile 可用（`profile.layer_stats` 非空）时：**永不告警**。

**异常处理：** 无新异常；回退行为（形状校验 ValueError、scheduler tau
覆写、gate 调用）逐字不变。

**验收标准：**
- Given 无层级 profile 的 engine（stub adapter 无 folded 前向），When
  连续调用 `verify_branch` 两次，Then 恰好 1 条该 UserWarning（recwarn
  计数）且两次返回值与改动前一致；
- Given 两个**独立** engine 实例（同参数），When 各调用 `verify_branch`
  一次，Then 共计 2 条该 UserWarning（实例级 once 语义钉死）；
- Given 有层级 profile 的 engine（folded 前向产生 `layer_stats`），When
  调用 `verify_branch`，Then 零 UserWarning 且返回 profiler 均值；
- Given pyproject 无 `-W error`（现状），When 全量回归，Then 既有 engine
  测试（UT-306/307 等走代理路径）零失败（告警不升级为错误）。

### 3.2 AdaptiveQuantileGate×FoldingScheduler 构造期拒绝

**描述：** 互斥组合在构造期显式失败，而非运行期静默失效。

**触发条件：** `FoldedTransformerLayer.__init__` 或
`ActFoldVerificationEngine.__init__` 收到
`scheduler is not None and isinstance(gate, AdaptiveQuantileGate)`。

**期望行为：**
- `FoldedTransformerLayer`：`RuntimeError`，消息指名
  `AdaptiveQuantileGate` 与 `FoldingScheduler` 互斥、scheduler 会覆写
  `gate.tau` 而分位数 gate 忽略固定 tau（AGENTS #29 语义），并提示改用
  `SimilarityGate` 或去掉 scheduler；
- `SplitFoldedTransformerLayer`：经 `super().__init__` 自动获得同一守卫；
- `ActFoldVerificationEngine.__init__`：同一组合同样 `RuntimeError`
  （engine 回退路径的 `set_tau(tau)` 同样静默无效）；
- 守卫用 `isinstance`（覆盖 AdaptiveQuantileGate 的子类），非精确类型。

**异常处理：** 合法组合零影响：`SimilarityGate + scheduler`（既有全量
测试主力组合）、`AdaptiveQuantileGate + scheduler=None`（gate 单测/脚本
组合，engine 构造同样合法——回退路径直接 `gate(h_child, h_parent)`）均
照常构造。全库 15 处 `AdaptiveQuantileGate(` 构造（test_adaptive_gate.py
13 处、test_fused_gate.py:390 1 处、scripts/opt3_adaptive_bench.py:142
1 处——脚本无 scheduler 匹配）均无 scheduler 组合，守卫零回归面。

**验收标准：**
- Given `AdaptiveQuantileGate + FoldingScheduler`，When 构造
  `FoldedTransformerLayer`，Then `RuntimeError` 且消息含两类名与
  `gate.tau`；
- Given 同组合，When 构造 `SplitFoldedTransformerLayer`，Then 同样
  `RuntimeError`（继承覆盖证据）；
- Given 同组合，When 构造 `ActFoldVerificationEngine`，Then 同样
  `RuntimeError`；
- Given `SimilarityGate + FoldingScheduler`，When 构造上述三者，Then 全部
  成功（合法组合钉死）；
- Given `AdaptiveQuantileGate + scheduler=None`，When 构造上述三者
  （含 engine），Then 全部成功（合法组合钉死）。

### 3.3 文档收口

**验收标准：**
- 指南 6.2 **:239 与 :240 两项均勾选**；:240 附勘误注
  （detect_architecture 已显式 raise，本 AR 收口的是 engine 代理回退
  可见性 + 互斥防护）+ 第十六部分回链；
- AGENTS #29 修订：互斥已由构造期 RuntimeError 强制（两处构造器），
  文档约定升级为代码契约；
- CHANGELOG AR009 节（Added：告警 + 守卫）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 静态检查 | mypy 标准命令 clean / pyflakes 零 / black 100 列 |
| 质量 | 全量回归 | `pytest tests\ -q -m "not slow"`：768 passed 基线零失败（slow 标记的 bench 依赖用例被 deselected——预存 ModuleNotFoundError 与本 AR 无关；告警不升级为错误；守卫零回归面已核实） |
| 兼容 | 公开 API | 零签名改动（纯构造器内校验 + 私有 once 标志） |
| 性能 | 告警开销 | once 布尔判断，稳态零开销 |

## 5. 约束与假设

**约束：**
- AGENTS #10：本 AR 是其直接落实——降级路径加可见性、错误配置早失败，
  不改变任何既有回退/拒继行为本身；
- 回退语义与守卫消息均不改变数值行为：3.1 的返回值、3.2 的合法组合路径
  逐位不变；
- `FoldedTransformerLayer` 与 `verification_engine` 对
  `actfold.core.adaptive_gate` 的 isinstance 导入不得成环（两者与
  adaptive_gate 同在 core/speculative 邻域，设计阶段验证导入方向）；
- once 标志为私有属性，不进入公共 API / state_dict / config。

**假设：**
- 既有 engine 测试的 stub adapter 无 folded 前向（走代理路径）——它们
  将开始收到一次性告警，因无 `-W error` 不破坏（pyproject 现状核实）；
- pytest `recwarn` / `pytest.warns` 足以断言告警次数与类别；
- 生产消费方（AblationStudy/BenchmarkRunner）均以 `SimilarityGate` 构造
  折叠栈（AR002 #37/#19 契约），不被守卫影响。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| embedding 代理 | `_estimate_stable_ratio` 在无层级 profile 时用 layer-0 embedding 相似度估计 stable ratio 的回退路径（AGENTS #16/#17） |
| 互斥组合 | `AdaptiveQuantileGate`（分位数 top-k，忽略固定 tau）× `FoldingScheduler`（动态覆写 `gate.tau`）——覆写无效导致调度静默失效（AGENTS #29） |
| once 标志 | engine/layer 实例级私有布尔，保证同类告警每实例至多一条（AR002 `_graph_degraded_warned` 先例模式） |
