# AR006 详细设计 — 真 draft 模型第一档：`model_draft` 模式与合成验证

> srs：`specs/changes/AR006-model-draft/srs.md`（权威契约）
> 创建：2026-10-10；状态：draft

## 1. 设计总览

方案：**A（DraftGenerator 模式扩展）**——在既有 `DraftGenerator` 加第 6 模式
`model_draft` + `draft_model` 构造参数，采样骨架复用 `_sample_from_topk`；
config 白名单值域扩展一行；BenchmarkRunner 可选注入。**否决 B（独立
`ModelDraftGenerator` 类）**：五处调用点（spiffy_baseline/ablation/
base_adapter/folded_generation/adaptive_draft_controller）类型注解均为
`DraftGenerator`，新类需改全部签名或引入 Union 类型，违背"接线成本极低"
的选型动机；模式扩展是既有 `logits_draft` 先例的自然延续。

```
DraftGenerator(vocab_size, mode="model_draft", draft_model=callable)
  └─ generate(parent, ...)
       ├─ draft_logits = draft_model(parent.tokens)      # 独立 forward
       ├─ 校验 (B, T, vocab_size) → ValueError
       ├─ flip 位置: _sample_from_topk(draft_logits[b, pos])
       └─ append 位置: _sample_from_topk(draft_logits[b, T-1])
config_manager 白名单 + "model_draft"          # 值域扩展
BenchmarkRunner(..., draft_model=None)          # 可选注入 + 构造期 RuntimeError
```

## 2. 关键算法与数据流

### 2.1 draft logits 获取（generate 内，mode == "model_draft"）

```
generate(parent, ...)
  ├─ seed → cpu_rng/dev_rng（AR002 §3.6 局部 Generator，复用既有逻辑）
  ├─ region 解析（_region，复用）
  ├─ if mode == "model_draft":
  │    if self.draft_model is None: raise RuntimeError(...)     # EX-801
  │    draft_logits = self.draft_model(parent.tokens)
  │    if shape != (B, T, V): raise ValueError(...)             # EX-802
  │    logits_source = draft_logits        # 统一后续采样
  ├─ elif mode == "logits_draft": logits_source = parent_logits（校验复用）
  └─ else: 现行分支不动
  采样：flip → _sample_from_topk(logits_source[b, pos], dev_rng)
        append → _sample_from_topk(logits_source[b, T-1], dev_rng)
```

**实现结构决策（D1）**：现有 `else: # suffix_append / logits_draft` 大分支
改为显式三分支（suffix_append / logits_draft / model_draft），flip/append
的 top-k 采样段抽小 helper `_resample_from_logits(...)` 供
logits_draft 与 model_draft 共用（避免两份近似重复的采样代码漂移）；
suffix_append（均匀采样）与 random/perturb/copy_flip 分支**逐字节不动**。

### 2.2 draft model 契约（鸭子类型）

```python
Callable[[torch.Tensor], torch.Tensor]
# 输入: tokens [B, T] int（parent.tokens 原样）
# 输出: logits [B, T, vocab_size] float（torch.is_floating_point 校验；
#        采样内部 torch.topk 不要求归一化——logits 语义，非 probs）
# nn.Module 天然满足 callable；forward 抛错原样传播（不吞，AGENTS #10）
```

校验三项（EX-802 扩展，srs 3.1 权威契约）：① 形状 == (B, T, V)；
② `torch.is_floating_point(draft_logits)`（int logits 会静默通过 topk）；
③ `draft_logits.device == parent.tokens.device`（跨设备时
`_sample_from_topk` 的 randint 用 `logits_pos.device`（:96）而 dev_rng
建于 parent.tokens.device（:141-145），torch 内部报难解 RuntimeError——
前置显式 ValueError）。

不抽象基类、不引入 Medusa/Eagle 结构（第二档再抽象，srs §5 约束）。

### 2.3 合成非平凡性 fixture（tests 内，生产不引入）

- `_CausalTokenTarget`（4 层 token-wise）：每层 `W_l ∈ R^{V×V}`（seed 固定），
  `logits[:, i] = MLP(tokens[:, :i+1])` 逐层叠加——同位置预测约定（#40），
  因果（位置 i 只依赖前缀）；
- `_CorrelatedDraft`（1 层）：`logits = W_1 @ tokens + P`（W_1 = target 第 1 层
  权重，P = seed 固定随机投影 `[V]` 扰动）——与 target 正相关（共享首层
  主导项）但不相等（缺后 3 层 + 扰动）；
- 无关 draft：全随机权重；
- 接受率测量：AR004 `acceptance_rate(child, target_logits)` 对
  draft token（flip 区域 = draft region，`draft_region_mask` 复用）。

**接受率数值推导（design 门控第 1 轮修正）**：`_sample_from_topk` 是
top-k **均匀**采样，故接受率 = P(target argmax ∈ draft top-k) / k，
**不是** P(draft argmax == target argmax)（后者仅 k=1 时相等）。接受率
测量 fixture 统一 **top_k=2**：

| 模式 | 接受率理论值 |
|---|---|
| random | 1/V = 0.001（V=1000） |
| model_draft（相关 draft） | P(hit)/2，P(hit) = P(target argmax ∈ draft top-2) |
| logits_draft（比较器上界） | 1/2 = 0.5（target argmax 恒在自己的 top-2 内） |
| 无关 draft | ≈ (k/V)/k = 1/V（k=2 时 ≈ 0.001） |

- srs 断言 gap ≥ 0.1 ⟺ P(hit) ≥ 0.202：fixture 首层共享 + 小扰动使
  draft top-1 ≈ target top-1（P(hit) 实测量级 0.3–0.9，阈值保守）；
- UT-507 的 ≤ 断言恒成立（P(hit) ≤ 1 → P(hit)/2 ≤ 0.5），比较器非恒 1.0；
- k 的选择不可挪动：k=1 → 比较器恒 1.0（srs 明确排除）；k=10 → gap
  上界 1/k − 1/V = 0.099 < 0.1 不可达。
- flip 位置覆盖：seq_len=64、flip_ratio=0.25 → 16 个 flip 位（srs ≥16）。

## 3. 接口设计

### 3.1 `DraftGenerator.__init__` 变更

```python
def __init__(
    self,
    vocab_size: int,
    mode: str = "suffix_append",
    perturbation_std: float = 0.1,
    flip_ratio: float = 0.1,
    top_k: int = 10,
    prompt_length: int | None = None,
    draft_model: Callable[[torch.Tensor], torch.Tensor] | None = None,  # 新
) -> None:
```

`_MODES` 加 `"model_draft"`。构造期**不**校验 draft_model（mode 与参数的
组合校验放 generate 期——与既有 logits_draft 的 parent_logits 校验时序
一致：先例是"参数在用时校验"）。

### 3.2 `generate` 行为（model_draft 分支）

| 输入条件 | 行为 |
|---|---|
| draft_model is None | RuntimeError（EX-801） |
| draft_model 输出形状 ≠ (B,T,V) | ValueError（EX-802） |
| draft_model forward 抛错 | 原样传播（EX-803 不吞） |
| parent_logits 非 None | 忽略（docstring 注记：上界对照实验可同时携带） |
| seed 非 None | 局部 Generator + counter 重置（复用既有，seed 复现） |
| flip_region 覆盖 | 与其他模式同语义（region 内 resample） |

### 3.3 config_manager / benchmark_runner 变更

- `config_manager.py:110`：白名单加 `"model_draft"`（值域扩展，无新字段，
  load_config 未知键告警路径不受影响）；
- `BenchmarkRunner.__init__` 加 `draft_model: Callable[...] | None = None`：
  `draft_mode == "model_draft" and draft_model is None` → RuntimeError
  （构造期，先于 DraftGenerator 构造）；非 None 且 mode 兼容 →
  `DraftGenerator(..., draft_model=draft_model)`；其余组合 draft_model
  忽略（非 model_draft 模式携带无害，docstring 注记）。

## 4. 测试设计

### 4.1 单元测试（UT-501~508，tests/test_draft_generator.py 既有文件追加）

| ID | 断言 | 对应 srs |
|---|---|---|
| UT-501 | model_draft + 固定 logits stub + seed：flip 位置 token ∈ 该位置 stub top-k 集合；同 seed 两次逐位一致 | 3.1 |
| UT-502 | model_draft 无 draft_model → RuntimeError | 3.1 |
| UT-503 | stub 返回错误形状 / int dtype / 跨 device → ValueError | 3.1 |
| UT-504 | parent_logits 非 None 时忽略（结果 == 不传） | 3.1 |
| UT-505 | max_new_tokens>0：append token ∈ 末位 stub top-k；seed 复现 | 3.1 |
| UT-506 | 合成 fixtures（top_k=2，seq_len=64/flip_ratio=0.25 → ≥16 flip 位）：model_draft 接受率 − random 接受率 ≥0.1 | 3.2 |
| UT-507 | model_draft 接受率 ≤ logits_draft 同 fixtures 实测接受率（理论 0.5，非恒 1.0） | 3.2 |
| UT-508 | 无关 draft 接受率 − random 接受率 <0.05（绝对值） | 3.2 |

### 4.2 异常测试（EX-801~804）

EX-801/802/803 见 3.2 表；EX-804：BenchmarkRunner config
mode="model_draft" 无注入 → 构造期 RuntimeError。

### 4.3 集成测试（IT-511~513）

| ID | 场景 | 断言 |
|---|---|---|
| IT-511 | AblationStudy(model_draft generator + 合成 adapter) measure_folding | 正常返回；候选 flip 位置 ∈ draft top-k（非全随机） |
| IT-512 | folded_generate(draft_generator=model_draft) 多候选 | 正常返回；候选 tokens 与 draft stub 采样一致 |
| IT-513 | 五模式默认路径 + 全量 738 | 逐位零回归（BS-701） |

### 4.4 边界/系统测试（BS-701~703）

- BS-701：全量 `pytest tests\ -q -m "not slow"` = 738 + 新增，零回归；
- BS-702：demo 基线 85.5% / 2.35e-03 / 93.75% 不变（demo 走默认模式）；
- BS-703：mypy 标准命令 / pyflakes / 100 列。

## 5. 影响范围

| 文件 | 改动 |
|---|---|
| actfold/speculative/draft_generator.py | `_MODES` + `draft_model` 参数 + 三分支重构 + `_resample_from_logits` helper + 形状/dtype/device 三项校验 |
| actfold/utils/config_manager.py | 白名单 +1 值（一行） |
| actfold/eval/benchmark_runner.py | `draft_model` 可选参数 + 构造期 RuntimeError |
| tests/test_draft_generator.py | UT-501~508 + EX-801~803 + fixtures |
| tests/test_ablation_measured.py / test_folded_generation.py / test_eval.py | IT-511/512/513 + EX-804（test_eval.py:118 有 draft_mode passthrough 先例可仿） |
| docs/AGENTS/CHANGELOG/README/指南 | T004 |

**零改动声明**：spiffy_baseline.py / ablation_study.py / base_adapter.py /
folded_generation.py / adaptive_draft_controller.py（draft_generator 参数
路径已通，无签名变更）。

## 6. 决策记录

| # | 决策 | 理由 |
|---|---|---|
| D1 | 模式扩展而非新类（方案 A） | 五调用点签名零改动；logits_draft 先例延续 |
| D2 | draft_model 组合校验放 generate 期（非构造期） | 与 parent_logits 先例时序一致（"用时校验"） |
| D3 | logits_draft/model_draft 采样段抽 `_resample_from_logits` 共用 helper | 消除两份近似重复代码的漂移风险 |
| D4 | draft logits 不要求 softmax/归一化，但校验 float dtype + device 一致 | `_sample_from_topk` 只用 topk 索引；int logits 静默通过、跨设备采样报难解错误——前置显式 ValueError（srs 3.1 权威） |
| D5 | append 从 parent 末位 draft logits 采样（非逐 token 自回归） | 第一档近似（srs §5 假设）；与 logits_draft 同语义 |
| D6 | BenchmarkRunner 非 model_draft 模式携带 draft_model → 忽略 | 无害宽松（构造校验只锁逆向组合缺失） |
| D7 | fixture 放 tests（生产不引入合成模型类） | AGENTS #10：生产无 mock；合成仅测试 |
| D8 | 接受率断言用 srs 阈值（≥0.1/<0.05），不锚定实测值 | 实测值随权重 seed 漂移；阈值是契约 |
| D9 | 接受率测量 fixture top_k=2 | k=1 → logits_draft 比较器恒 1.0（srs 排除）；k=10 → gap 上界 0.099 < 0.1 不可达；k=2 联合可满足（上界 0.5、gap 需 P(hit)≥0.202） |

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 三分支重构动到既有 else 分支 | IT-513 逐位零回归 + 既有 test_draft_generator.py 全套不动先行验证 |
| 合成 fixture 相关性 P(hit) < 0.202 | UT-506 先跑锚定；正确旋钮是 fixture 的相关性构造（扰动幅度 P、共享层数）与 top_k 配置（D9 已锁 k=2）；不放松 srs 阈值 |
| draft_model 跨设备返回 / int logits | 前置显式 ValueError（§2.2 三项校验），不依赖 torch 内部错误信息 |
