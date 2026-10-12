# AR009 详细设计 — 折叠静默失效防护（silent-misconfiguration guards）

> srs：`specs/archive/AR009-silent-misconfig-guards/srs.md`（权威契约）
> 创建：2026-10-12；状态：final（design 门控第 1 轮 PASS，review PASS，ST Go）

## 1. 设计总览

方案：**A（双构造器守卫 + 一次性 UserWarning）**——
FR2：`FoldedTransformerLayer.__init__` 与 `ActFoldVerificationEngine.__init__`
在构造期对 `scheduler is not None and isinstance(gate, AdaptiveQuantileGate)`
组合抛 `RuntimeError`；FR1：`_estimate_stable_ratio` 进入 embedding 代理
回退分支时，经实例级 once 标志发 `UserWarning`。

**否决 B（仅 layer 守卫）**：engine 回退路径（verification_engine.py:285-291）
同样以 `gate.set_tau(tau)` 动态覆写——只守 layer 则 engine 组合仍静默失效。

**否决 C（`AdaptiveQuantileGate.set_tau` 覆写为 raise）**：把组合错误延迟到
运行期第一次回退调用才爆（违反 fail-fast），且改变继承方法行为波及所有
`set_tau` 调用点（含合法无 scheduler 场景的误伤风险）。

**否决 FR1-B（logging）**：项目用户可见提示的既定惯例是 `UserWarning`
（AGENTS #5 config unknown keys、AR002 graph degrade），不引入新日志约定。
**否决 FR1-C（每次回退告警）**：`verify_branch` 每步一回退，逐条告警成噪音。

## 2. 关键算法与数据流

### 2.1 FR2：构造器守卫（folded_transformer.py + verification_engine.py）

```python
# folded_transformer.py（新增导入，无环：adaptive_gate 仅依赖 similarity_gate）
from actfold.core.adaptive_gate import AdaptiveQuantileGate

class FoldedTransformerLayer(nn.Module):
    def __init__(self, original_layer, cache, gate, layer_idx, scheduler=None):
        super().__init__()
        if scheduler is not None and isinstance(gate, AdaptiveQuantileGate):
            raise RuntimeError(
                "AdaptiveQuantileGate cannot be combined with FoldingScheduler: "
                "the scheduler overwrites gate.tau, but the quantile gate "
                "ignores fixed thresholds (AGENTS #29). Use SimilarityGate "
                "with the scheduler, or pass scheduler=None with "
                "AdaptiveQuantileGate."
            )
        ...  # 既有赋值不变
```

```python
# verification_engine.py（同一守卫，消息复用同一构造辅助或逐字一致文案）
if scheduler is not None and isinstance(gate, AdaptiveQuantileGate):
    raise RuntimeError(<同上消息>)
```

要点：
- **插入点**：layer 在 `super().__init__()` 之后、属性赋值之前（fail-fast，
  构造失败不留半初始化对象）；engine 在既有 `ema_alpha` 校验之后、
  `self.model = model` 之前；
- **`isinstance` 而非精确类型**：`AdaptiveQuantileGate` 的子类同样不读固定
  `tau`，须一并拒绝（与 AGENTS #38 fused dispatch 的"精确类型"意图相反——
  那是性能分发白名单，这是语义黑名单）；
- **消息契约**：含 `AdaptiveQuantileGate`、`FoldingScheduler`、`gate.tau`
  三个关键词 + 两条修复路径（换 `SimilarityGate` / 去 scheduler）+ AGENTS #29
  指针；两处构造器消息**逐字一致**（抽模块级常量防漂移：在
  folded_transformer.py 定义 `_ADAPTIVE_GATE_SCHEDULER_MSG`，
  engine 侧导入复用——engine 已导入 core 模块，方向无环）；
- **Split 子类覆盖**：`SplitFoldedTransformerLayer.__init__` 经
  `super().__init__` 透传 `gate`/`scheduler`，守卫自动生效，零额外代码；
- **`ManualFoldedForward` 传递性**：`_wrap_layer` 构造 layer 时组合即触发
  守卫——`ManualFoldedForward.__init__` 内包层（architecture_utils.py:419-421），
  故仍是构造期失败。

### 2.2 FR1：代理回退一次性告警（verification_engine.py）

```python
import warnings  # 新增

# __init__ 新增（ema 校验后）：
self._proxy_ratio_warned: bool = False

# _estimate_stable_ratio 回退分支入口（现行 "# Fallback" 注释后）：
if not self._proxy_ratio_warned:
    self._proxy_ratio_warned = True
    warnings.warn(
        "The stable_ratio for this verification is an embedding-level "
        "proxy: no per-layer stability profile was recorded, so folding "
        "did not run through a folded forward pass. Attach a folded model "
        "(e.g. ManualFoldedForward) via the adapter and make sure the base "
        "model propagates branch kwargs (AGENTS #17/#18).",
        UserWarning,
        stacklevel=2,
    )
```

要点：
- **once 判定先行、置位紧随**：首条告警后标志置位，后续回退零开销短路；
- **层级路径零接触**：告警块位于 `profile is not None and profile.layer_stats`
  早退分支**之后**，层级路径不进入该代码区；
- **实例级语义**：标志挂在 engine 实例上——新实例重新告警（UT-802 钉死）；
- **行为零变化**：回退分支的形状校验 / scheduler tau 覆写 / gate 调用 /
  返回值逐行不动，仅插入告警块；
- **`stacklevel=3`（design 门控 Minor-1 修正）**：告警在
  `_estimate_stable_ratio` 内发出、由 `verify_branch` 调用——`stacklevel=2`
  仍归因到框架内部（`verify_branch` 行号），`stacklevel=3` 才归因到
  `verify_branch` 的用户调用点；

## 3. 接口设计

**公开接口零改动**：两处构造器签名、`_estimate_stable_ratio` 签名均不变。
新增私有成员：

| 位置 | 成员 | 说明 |
|---|---|---|
| `ActFoldVerificationEngine.__init__` | `self._proxy_ratio_warned: bool = False` | 代理告警 once 标志 |
| `folded_transformer.py` 模块级 | `_ADAPTIVE_GATE_SCHEDULER_MSG: str` | 守卫消息常量（engine 导入复用） |

## 4. 测试设计

### 4.1 单元测试（UT-801~805，新文件 tests/test_silent_misconfig_guards.py）

Engine stub 沿用 test_verification_engine.py 既有最小 adapter 模式
（`embed` 返回真实 embedding、`num_layers`/`hidden_dim`/`num_heads`/
`vocab_size` 属性、`forward` 返回 logits）；profile 注入用**真实
`StabilityProfile` + `LayerStabilityStats(layer_idx=0, step_idx=0,
stable_ratio=0.5, tau_used=0.5, metric="cosine", num_tokens=4)`** +
monkeypatch `GLOBAL_STABILITY_PROFILER.get_profile`（不 mock
`_estimate_stable_ratio` 本身——被测函数必须真实执行）。

| ID | 构造 | 断言 | 对应 srs |
|---|---|---|---|
| UT-801 | 同一 engine（stub adapter 无 folded 前向），`recwarn` 下连续 `verify_branch` 两次 | 恰好 1 条 UserWarning 且消息含 "embedding-level proxy"；两次 `stable_ratio` 均逐位等于测试内独立计算的 `gate(h_child, h_parent).mean()`（回退公式本身，防恒真断言） | 3.1-① |
| UT-802 | 两个**独立**同参 engine 实例，各 `verify_branch` 一次 | 共 2 条该 UserWarning（实例级 once 语义） | 3.1-② |
| UT-803 | engine + monkeypatch `get_profile` 返回含 1 条 `LayerStabilityStats(stable_ratio=0.5)` 的 `StabilityProfile`，`verify_branch` 一次 | 零 UserWarning；`stable_ratio == 0.5`（profiler 均值路径不触发告警） | 3.1-③ |
| UT-804 | `FoldedTransformerLayer` 与 `SplitFoldedTransformerLayer`：各以 `(AdaptiveQuantileGate, FoldingScheduler)` 构造 → `pytest.raises(RuntimeError, match="gate.tau")`；合法对照：`(SimilarityGate, FoldingScheduler)` 与 `(AdaptiveQuantileGate, None)` 构造成功（四种组合 × 两个类） | 守卫触发 + 消息关键词 + 继承覆盖 + 合法组合零影响 | 3.2 |
| UT-805 | `ActFoldVerificationEngine`：`(AdaptiveQuantileGate, FoldingScheduler)` → RuntimeError 同消息；合法对照：`(SimilarityGate, scheduler)` 与 `(AdaptiveQuantileGate, scheduler=None)` 构造成功 | engine 守卫 + 合法组合钉死 | 3.2 |

Layer 守卫测试的最小 layer 用 `nn.Module` 空 stub（守卫在属性赋值前触发，
不触及 layer 内部）；`FoldingScheduler` 用**无参构造 `FoldingScheduler()`**
（design 门控 Minor-2 修正：签名实为 `base_tau/num_layers/num_steps/
disabled_layers`，无 `tau_schedule` 参数）；合法对照的 cache 用 CPU
`ActivationCache`（engine 合法对照需真实 `cache.device`，design 门控
WARN-4）。

**UT-801/802 返回值钉死方式（design 门控 Minor-3）**：不写"与预期一致"
的模糊断言——测试内以同一 gate 独立计算
`expected = gate(h_child, h_parent).float().mean().item()`（embedding 代理
公式本身），断言 engine 返回值 **逐位等于** 该独立计算值（钉死"回退行为
不变"= pre-change 语义），且两次调用彼此相等；`recwarn` 计数独立断言。

### 4.2 集成/回归

| ID | 场景 | 断言 |
|---|---|---|
| IT-806 | 既有全量（`pytest tests\ -q -m "not slow"`） | 768 passed 基线 + 新增 5 例全过；既有 engine 测试（UT-306/307 等走代理路径）因无 `-W error` 零失败 |
| BS-801 | demo.py | 85.5% / 2.35e-03 / 93.75% 精确一致（改动不触及 demo 路径） |
| BS-802 | 静态 | mypy 标准命令 clean / pyflakes 零 / 100 列 |

## 5. 影响范围

| 文件 | 改动 |
|---|---|
| actfold/core/folded_transformer.py | 新增导入 + 模块级消息常量 + `__init__` 守卫（~10 行净变化） |
| actfold/speculative/verification_engine.py | 新增 `warnings`/`AdaptiveQuantileGate` 导入 + `__init__` 守卫与 `_proxy_ratio_warned` + `_estimate_stable_ratio` 告警块（~20 行净变化） |
| tests/test_silent_misconfig_guards.py（新） | UT-801~805 |
| docs/OPTIMIZATION_GUIDE.md、AGENTS.md（#29 修订）、CHANGELOG.md | T003 |

**零改动声明**：`adaptive_gate.py`、`similarity_gate.py`、
`split_layer.py`、`architecture_utils.py`、ablation/benchmark/eval 消费链
全部不动（守卫是纯构造期校验，消费方均以 `SimilarityGate` 构造折叠栈）。

## 6. 决策记录

| # | 决策 | 理由 |
|---|---|---|
| D1 | 双构造器守卫（方案 A） | 两条 `set_tau` 动态覆写路径（layer :138 / engine :291）全覆盖；构造期 fail-fast（AGENTS #10） |
| D2 | 消息抽模块级常量、两构造器逐字一致 | 防两处文案漂移；`pytest.raises(match=)` 断言可跨构造器复用 |
| D3 | `isinstance` 黑名单语义 | 子类同样忽略固定 tau；与 #38 精确类型白名单（性能分发）意图相反需区分 |
| D4 | once 标志实例级（非模块级全局） | 新 engine 实例应重新告知（AR002 `_graph_degraded_warned` 同语义）；模块级 once 会吞掉后续实例的首告警 |
| D5 | UT-803 用真实 `StabilityProfile` 数据类 + monkeypatch `get_profile` | 被测函数真实执行；伪造 `layer_stats` 走 dataclass 真实 `mean_stable_ratio` 属性逻辑 |
| D6 | 守卫置于属性赋值前 | fail-fast：构造失败不留半初始化 `nn.Module`（属性已挂一半） |

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 既有 engine 测试开始收到代理告警，未来若有人加 `-W error` 会翻绿为红 | pyproject 现状无 filterwarnings（已核实）；本 AR 在 srs/CHANGELOG 注明告警是**有意新增**的可观测行为（AGENTS #10 落实），非泄漏 |
| 消费方（AblationStudy/BenchmarkRunner）未来想用 adaptive gate 做实验 | 守卫消息给出两条修复路径；实验可传 `scheduler=None`（分位数 gate 本就不需要 tau 调度） |
| `_ADAPTIVE_GATE_SCHEDULER_MSG` 常量成为 core→speculative 的文本耦合 | 单向导入（speculative→core），无环；消息变更由 UT-804/805 `match=` 双端钉死 |
| monkeypatch `get_profile` 泄漏到其他测试 | `pytest.MonkeyPatch` 作用域自动还原；`verify_branch` 开头自身的 `reset_branch` 清真实 profiler，UT-803 无需显式清理（review WARN 勘误：原文误称 UT-803 内显式调用） |
| 告警在 CUDA graph capture 等禁同步路径内触发 | 回退分支只在**无** folded 前向时进入，graph capture 需完整 parent cache（capture 前提），路径不相交 |
