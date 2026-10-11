# AR008 详细设计 — generic.forward `output_hidden_states` 按需化

> srs：`specs/changes/AR008-generic-hidden-states-on-demand/srs.md`（权威契约）
> 创建：2026-10-11；状态：draft

## 1. 设计总览

方案：**A（惰性重跑 + 实例标志）**——`GenericDiffusionLLM.forward` 默认传
`output_hidden_states=False`；仅当输出既无 `logits` 也无 `last_hidden_state`
时，置实例标志并带 `True` 重跑同一前向，从 `hidden_states` 元组取末层。
标志置位后后续前向直接传 `True`（每实例至多一次双跑）。

**否决 B（构造期探测前向）**：`__init__` 阶段跑一次 `True` 前向探测输出
形状——改变构造副作用（加载即跑前向，大模型不可接受），且
`GenericDiffusionLLM` 构造时模型可能尚未完成设备/dtype 迁移。

**否决 C（恒 False + 报错指导）**：第三类模型直接 raise 指导用户改用
其它路径——破坏行为兼容（srs 3.1 要求三态完全兼容）。

## 2. 关键算法与数据流

### 2.1 forward 改造（generic.py:115-153）

```python
def __init__(...):
    ...
    # AR008: set once when a forward output turns out to expose neither
    # ``logits`` nor ``last_hidden_state``; later forwards then request
    # hidden states up front instead of re-running twice.
    self._needs_hidden_states = False

def forward(self, tokens, attention_mask=None, **kwargs):
    model = self.model
    if model is None:
        raise RuntimeError("Model has not been loaded.")
    for key in ("branch_id", "parent_branch_id", "step_idx"):
        kwargs.pop(key, None)
    outputs = model(
        input_ids=tokens,
        attention_mask=attention_mask,
        output_hidden_states=self._needs_hidden_states,
        return_dict=True,
        **kwargs,
    )
    if getattr(outputs, "logits", None) is not None:
        return cast(torch.Tensor, outputs.logits)
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            last_hidden = hidden_states[-1]
    if last_hidden is None:
        if not self._needs_hidden_states:
            # AR008: this model exposes neither logits nor
            # last_hidden_state in its default output — re-run once
            # with hidden-state collection enabled and cache the
            # decision (at most one double forward per instance).
            self._needs_hidden_states = True
            return self.forward(tokens, attention_mask=attention_mask, **kwargs)
        raise RuntimeError(<现行消息逐字>)
    logits = cast(torch.Tensor, self.lm_head(last_hidden))
    return logits
```

要点：
- **平铺结构（design 门控 WARN-2 修复）**：`last_hidden` 的 hidden_states
  回退与"最终不可用"判定保持现行两段式——真值元组但末元素 None 时仍
  raise RuntimeError（伪码不得把 None 喂给 `lm_head`）；
- **递归重跑恰好一层**：置位后递归调用自身，第二次进入时
  `self._needs_hidden_states` 已为 True → 不会再递归（srs"不无限重跑"）。
  递归前 `kwargs` 已剥离 branch 键 → 第二次剥离是幂等 no-op；
  `attention_mask` 是具名参数（Python 语法层面不可能同时出现在
  kwargs 中，无重复键）；
- **train 模式双跑**：首跑输出不可用（两字段缺失），重跑是唯一可用
  结果；eval 链路（demo/eval 既有 `.eval()` 约定）确定性一致；
- **ModelOutput 探测语义**：`getattr(outputs, "logits", None)` 对
  `ModelOutput` 缺失字段返回 None（跨 transformers 版本稳定，
  AGENTS #24 域）；stub 测试用 `SimpleNamespace` 模拟同语义。

### 2.2 实例标志的调用链覆盖（req 门控核实结论）

| 调用路径 | 是否经 `GenericDiffusionLLM.forward` | 标志覆盖 |
|---|---|---|
| 基类 `DiffusionLLM.generate` AR fallback（base.py:125） | 是（`self.forward`） | ✓ |
| LLaDA/Dream sampler 非折叠路径（diffusion_sampler.py:171 `model.forward`） | 是（继承未覆写） | ✓ |
| sampler 折叠路径（diffusion_sampler.py:156 `folded_model` 直调） | 否（今日即绕过，不传该标志） | 行为零变化 |
| `_native_sampler`（llada.py:79-87 传裸 `self.model`） | 否（同上） | 行为零变化 |

## 3. 接口设计

**公开接口零改动**：`GenericDiffusionLLM.__init__` / `forward` / `embed` /
`generate` 签名不变。新增私有实例属性 `_needs_hidden_states: bool`
（mypy 可见、非公共 API）。

| 函数 | 变更 |
|---|---|
| `__init__` | 新增 `self._needs_hidden_states: bool = False`（一行，内联注解满足 `--strict`） |
| `forward` | `output_hidden_states=self._needs_hidden_states`；两字段缺失且元组不可用时置位 + 递归重跑；`True` 态下仍不可用 → 现行 RuntimeError 逐字 |

## 4. 测试设计

### 4.1 单元测试（UT-701~705，tests/test_models.py 追加或新文件 tests/test_generic_forward.py）

Stub 构造原则（req 门控）：**真实 `SimpleNamespace` 输出对象**（非
MagicMock 返回值——`getattr` 探测语义必须真实）；底层模型用
`nn.Module` stub 记录每次调用的 kwargs（`call_log: list[dict]`）。
**旗标条件式（design 门控 WARN-1 修复）**：hidden_states-only stub 的
`forward` 必须检查 `kwargs["output_hidden_states"]`——`False` 时返回
`SimpleNamespace(hidden_states=None)`（真实 HF 语义：不收集则字段为
None），`True` 时返回含 `(h0, h1)` 的 SimpleNamespace。恒返回元组的
stub 会在首跑（False）即发现非空元组直接返回，"首次 2 调"断言成为
假 Red（实现正确也过不了）。

| ID | 构造 | 断言 | 对应 srs |
|---|---|---|---|
| UT-701 | logits-only stub（SimpleNamespace(logits=tensor)），调用 2 次 | 每次底层 `output_hidden_states is False`（call_log spy）；返回值 == stub logits 逐位 | 3.1-① |
| UT-702 | last_hidden_state-only stub | `False` 传递 + 返回 == `lm_head(last_hidden)` 逐位 | 3.1-② |
| UT-703 | hidden_states-only **旗标条件式** stub（False → hidden_states=None；True → (h0, h1)），首次调用后再调一次 | 首次：底层 2 次调用（kwargs 依次 False→True），返回 == `lm_head(h1)`；第二次：底层仅 1 次调用且 `True`（标志缓存生效） | 3.1-③ |
| UT-704 | 三态全无 stub（SimpleNamespace() 或全 None 字段）**及 `hidden_states=()` 空元组变体**（design 门控 WARN-4） | RuntimeError 消息与现行逐字一致；底层恰好 2 次调用（False→True，不无限重跑）；空元组变体同断言 | 3.1-④ |
| UT-705 | hidden_states-only 旗标条件式 stub + forward 传 `branch_id`/`parent_branch_id`/`step_idx` | 首次两调的 kwargs 均不含 branch 键；结果正确 | 3.1-⑤ |

`GenericDiffusionLLM` 实例构造沿用 test_models.py 既有 MagicMock +
patch `from_pretrained` 模式（:62-99 先例），`self.model` 换成 stub
nn.Module。

### 4.2 端到端/实证（EX-805 + 回归）

| ID | 场景 | 断言 |
|---|---|---|
| EX-805（CUDA skipif） | 24 层 stub（每层输出 [1,512,4096] fp32）last_hidden-only；同一模型同一输入分别以 `_needs_hidden_states=False` 与 `True` 实例跑 forward，`torch.cuda.reset_peak_memory_stats()` + `max_memory_allocated` 对照 | 峰值差 ≥ 150MB（理论 192MB）；无 CUDA skip。**构造（design 门控 WARN-3）**：① True 实例在构造后手动置私有属性 `inst._needs_hidden_states = True`（last_hidden-only stub 永不自然触发；测试内注入私有属性可接受）；② stub 无参数层——持 1 个 [1,512,4096] buffer，`output_hidden_states=True` 时 clone 24 次拼元组（192MiB 是运行期分配，权重近零，无需 4096×4096 Linear）；③ patch `from_pretrained` **返回真实 stub 模块**（MagicMock 返回值不产生 CUDA 分配），stub 带 `.config`（hidden_size=4096 等）供 generic `__init__` :70 读取 |
| IT-806 | 既有 generic 消费链 | test_models.py 全过；LLaDA/Dream 继承链零回归（既有测试） |
| BS-701 | 全量 `pytest tests\ -q -m "not slow"` | 761 passed 基线 + 新增全过（req 门控实测 767 collected 与 761+3 skipped+3 deselected 口径吻合，T004 备注写明） |
| BS-702 | demo.py | 85.5% / 2.35e-03 / 93.75% |
| BS-703 | 静态 | mypy 标准命令 / pyflakes / 100 列 |

## 5. 影响范围

| 文件 | 改动 |
|---|---|
| actfold/models/generic.py | `__init__` 一行 + `forward` 按需化（~15 行净变化） |
| tests/test_generic_forward.py（新）或 test_models.py 追加 | UT-701~705 + EX-805 |
| docs/OPTIMIZATION_GUIDE.md、CHANGELOG.md | T003 |

**零改动声明**：`base.py`、`diffusion_sampler.py`、`llada.py`、`dream.py`、
`fast_dllm_adapter.py`、folded/cuda 路径全部不动（继承链自动受益）。

## 6. 决策记录

| # | 决策 | 理由 |
|---|---|---|
| D1 | 惰性重跑 + 实例标志（方案 A） | 构造期探测有副作用（否决 B）；恒 False 破坏兼容（否决 C）；重跑仅在输出本就不可用的路径发生，每实例至多一次 |
| D2 | 递归实现重跑（恰好一层） | 置位后第二次进入不会递归；kwargs 剥离幂等；代码路径复用（无重复的 model 调用块） |
| D3 | RuntimeError 消息逐字保留 | 防依赖消息文本的调用方/测试破坏（AGENTS 稳定性偏好） |
| D4 | stub 用 SimpleNamespace + nn.Module call_log，非 MagicMock 返回值 | `getattr(x, field, None)` 探测语义必须真实（req 门控：既有套件对三分支零覆盖，UT 是唯一钉子） |
| D5 | EX-805 用 False/True 实例对照而非前后版本对照 | 绕开重跑混杂与分支对照工程量；`max_memory_allocated` 字节精确，16GB 卡无干扰 |

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 某模型 `logits`/`last_hidden_state` 字段存在但值为 None 之外的 falsy | 现行代码同语义（`getattr(...) is not None` + truthy 元组检查）——探测顺序逐行保留，无新判断 |
| 重跑在 train 模式产生不同 dropout 结果 | 首跑输出本就不可用，重跑是唯一可用结果；eval 链路确定性（demo/eval `.eval()` 约定） |
| 实例标志跨 generate/eval 复用导致陈旧 | 标志只从 False→True 单向置位；模型实例不变则输出形状不变（无失效场景） |
| transformers 版本差异（ModelOutput 字段语义） | `getattr` None 探测跨版本稳定（AGENTS #24 域内）；三态 stub 测试钉死探测行为 |
| EX-805 分配器缓存掩盖峰值差 | `reset_peak_memory_stats` 后单次 forward 对照；stub 权重固定；阈值 150MB 留 42MB 余量 |
