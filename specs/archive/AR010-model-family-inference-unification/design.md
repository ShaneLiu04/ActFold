# AR010 详细设计 — 模型家族推断统一与回退可见性

> srs：`specs/archive/AR010-model-family-inference-unification/srs.md`（权威契约）
> 创建：2026-10-12；状态：final（design 门控第 1 轮 PASS，review PASS，ST Go）

## 1. 设计总览

方案：**A（委托去重 + registry 层告警）**——`_resolve_family` 的 auto
分支改为 `return infer_model_family(model_name_or_path)`；当
`model_family == "auto"` 且委托结果为 `"causal_lm"` 时在
`_resolve_family` 内发 `UserWarning`。

**否决 B（utils 层告警）**：`infer_model_family` 保持纯函数（srs 约束）；
utils 层告警会波及未来所有调用方且丢失 registry 语境（可用家族列表、
显式传参指引都在 registry 域内）。

**否决 C（load 层告警）**：`load` 的 family 已解析完毕，区分"auto 回退"
与"显式 causal_lm"需再传上下文；`_resolve_family` 是唯一同时握有
`model_family` 原值与推断结果的点，落点最短。

**否决 D（token 化匹配改造）**：srs §1 勘误——"FastDLLM" 连写名 lower
后是单 alnum token，token 等值匹配失效；双子串 AND 误报面极窄。规则
逐字保留。

## 2. 关键算法与数据流

### 2.1 委托 + 告警（registry.py）

```python
import warnings
from actfold.models.utils import infer_model_family

    @classmethod
    def _resolve_family(cls, model_name_or_path: str, model_family: str) -> str:
        """Resolve model family from config or identifier."""
        if model_family != "auto":
            return model_family
        family = infer_model_family(model_name_or_path)
        if family == "causal_lm":
            warnings.warn(
                f"auto-detect fell back to causal_lm for "
                f"'{model_name_or_path}': no known family keyword matched. "
                f"If this is a diffusion model, pass model_family "
                f"explicitly (available: {', '.join(cls.list_models())}).",
                UserWarning,
                stacklevel=3,
            )
        return family
```

要点：
- **告警判据 `family == "causal_lm"`**：`infer_model_family` 的子串规则
  不含 causal_lm 匹配分支——返回 causal_lm **当且仅当**走了回退
  （req 门控核实的事实，无需额外返回标志）；
- **`stacklevel=3`（design 门控 Minor-1 修正）**：告警在
  `_resolve_family` 内发出，生产路径为 `load` → `_resolve_family` →
  warn——2 层归因到 registry.py:68（库内部），3 层才归因到用户的
  `load` 调用点（AR009"归因到公开 API 调用方"同惯例；测试直调
  `_resolve_family` 时落点多一层，可接受）；
- **registry 侧重复规则整体删除**（:82-90），子串逻辑只剩 utils 一份；
- **显式路径零接触**：`model_family != "auto"` 提前返回，告警块不可达；
- **消息契约**：含 "auto-detect fell back to causal_lm"（match 短语）、
  模型名、显式传参指引、`list_models()` 动态家族列表——UT-902 用
  `pytest.warns(UserWarning, match="auto-detect fell back to causal_lm")`
  + 消息含模型名双断言。

### 2.2 utils.infer_model_family（零改动）

规则逐字保留（utils.py:12-28）：lower 折叠 → `"llada"` / `"dream"` /
`"fast" and "dllm"` 子串 → `"causal_lm"` 回退。**唯一实现源**自此成立。
D3 判据依赖的不变量（"返回 causal_lm 当且仅当走回退"）以 registry 侧
告警块注释注明（design 门控 Minor-2：不动 utils.py——§5 零改动声明
保持为真；不变量注记放消费侧）。

### 2.3 导入方向（req 门控核实）

`actfold.models.utils` 仅依赖 pathlib/typing/torch/transformers（零
actfold 内部导入）→ registry 导入 utils 无环；`actfold\models\__init__.py`
的模块加载顺序不受影响。

## 3. 接口设计

**公开接口零改动**：`load` / `load_model` / `infer_model_family` /
`_resolve_family` 签名均不变。行为变化仅两点（均为 srs 规格）：auto
路径的实现从本地规则变为委托（结果逐位一致）；auto 回退多一条
UserWarning。

| 位置 | 变更 |
|---|---|
| registry.py 顶部 | `import warnings` + `from actfold.models.utils import infer_model_family` |
| registry.py `_resolve_family` | 删除重复规则，改为委托 + fallback 告警（净 -7/+11 行） |

## 4. 测试设计

### 4.1 单元测试（UT-901~904，新文件 tests/test_model_family_inference.py）

`ModelRegistry.load` 的 mock 模式沿用 test_models.py:105-108 先例
（`patch("actfold.models.registry.ModelRegistry.load")` 风格反向——
本处需 patch `_REGISTRY` 中被解析家族的构造器，或 patch
`ModelRegistry._REGISTRY` 临时注入 dummy 类，避免真实 from_pretrained）。
**告警计数**：UT-902 用 `pytest.warns`（恰好一条断言由 with 块内单次
load 保证）；UT-903/904 用 `warnings.catch_warnings()` + `simplefilter`
（`error` 模式——任何 UserWarning 即抛，钉死零告警）。

| ID | 构造 | 断言 | 对应 srs |
|---|---|---|---|
| UT-901 | 等价矩阵：`("GPT-LLaDA-8B","auto")→"llada"`、`("FastDLLM-v2","auto")→"fast_dllm"`、`("some/dream-7b","auto")→"dream"`、`("gpt2","auto")→"causal_lm"`、`("gpt2","causal_lm")→"causal_lm"`（显式）、`("anything","dream")→"dream"`（显式透传） | `_resolve_family` 返回值逐一相等（混合大小写/连写覆盖）；子串规则保留的等价证据。**gpt2-auto 用例以 `catch_warnings` + `simplefilter("ignore")` 包裹**（design 门控 WARN-1：该用例触发新告警，不包裹则泄漏到报告且未来 `-W error` 即断——等价矩阵只验返回值，告警已由 UT-902 专测） | 3.1 |
| UT-902 | `"gpt2"` + auto，patch `_REGISTRY` 注入 dummy 构造器（记录调用） | `pytest.warns(UserWarning, match="auto-detect fell back to causal_lm")`；`str(w.message)` 含 `"gpt2"`；dummy 被以 causal_lm 家族类调用（回退行为不变） | 3.2-① |
| UT-903 | `"gpt2"` + 显式 `"causal_lm"`，同 patch | `catch_warnings` + `simplefilter("error")` 下 load 成功——零 UserWarning（显式选择钉死） | 3.2-② |
| UT-904 | `"some/llada-8b"` + auto，同 patch | 同 UT-903 手法——零 UserWarning（已知子串命中钉死）；解析为 llada dummy | 3.2-③ |

**既有钉子（零改动）**：test_models.py `test_registry_resolve_family`
5 断言全过 = 去重不改行为的硬门；`test_load_model_convenience`
（:105-108）的 mock 路径不受影响。

### 4.2 集成/回归

| ID | 场景 | 断言 |
|---|---|---|
| IT-905 | 既有 test_models.py 全过（含 :29 gpt2-auto 在告警下仍绿） | 5 断言 + load ValueError 路径零改动 |
| BS-901 | 全量 `pytest tests\ -q -m "not slow"` | 782 passed（773 基线 + 9 新增——UT-901 参数化 6 例 + UT-902/903/904，review Minor-2 勘误）零失败 |
| BS-902 | demo.py | 85.5% / 2.35e-03 / 93.75% 精确一致（demo 不经 registry.load——合成模型直构造，作无干扰证据） |
| BS-903 | 静态 | mypy 标准命令 clean / pyflakes 零 / 100 列 |

## 5. 影响范围

| 文件 | 改动 |
|---|---|
| actfold/models/registry.py | 导入 +2、`_resolve_family` 委托化与告警（净变化 ~15 行） |
| tests/test_model_family_inference.py（新） | UT-901~904 |
| docs/OPTIMIZATION_GUIDE.md、CHANGELOG.md | T003 |
| AGENTS.md | T003 可选项（srs 3.3 授权）："Adding a New Model Family" 补唯一实现源 + causal_lm 不变量注（review WARN-4 补录） |

**零改动声明**：`actfold/models/utils.py`、config_manager.py、
benchmark_runner.py、scripts 全部不动（"auto" 语义不变，告警是提示性
可观测行为）。**README.md 经审查不需更**（design 门控 WARN-2：README
:319 与 EXPERIMENTS.md:293 均只示例显式 family，无 auto 回退行为描述，
无文档会因本 AR 变陈旧——AGENTS Documentation 检查项闭合）。

## 6. 决策记录

| # | 决策 | 理由 |
|---|---|---|
| D1 | 委托方向 registry→utils（utils 为唯一实现源） | utils 是叶子模块（零 actfold 依赖），反向会让 utils 依赖 registry 成环 |
| D2 | 告警在 `_resolve_family` 层 | 唯一同时握有 `model_family` 原值与推断结果的点（否决 C）；utils 保持纯函数（否决 B） |
| D3 | 告警判据 `family == "causal_lm"` | 子串规则无 causal_lm 匹配分支——返回值即回退标志，无需改 `infer_model_family` 返回协议 |
| D4 | 每次 `_resolve_family` 调用一条（无 once 标志） | registry 是无实例类；load 非热路径（每模型一次）；重复告警场景不存在 |
| D5 | UT-903/904 用 `simplefilter("error")` 钉零告警 | 比 recwarn 空列表断言更强（任何类别 UserWarning 都翻红） |
| D6 | UT-902 patch `_REGISTRY` 注入 dummy 构造器 | 避免 from_pretrained 网络加载；同时验证"回退仍构造 causal_lm 家族类"（行为不变） |

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 既有测试/脚本在 auto+回退路径收到新告警 | pyproject 无 `-W error`（核实）；生产 auto+未知名场景本就该提示（AGENTS #10） |
| 告警每 load 一次在批量加载脚本中成噪音 | scripts 均显式传 `spec["family"]`（grep 核实），不触发告警；benchmark 默认 auto 但每模型仅 load 一次 |
| 消息中 `list_models()` 输出随注册表变化 | 动态生成是有意设计（注册新家族后指引自动更新）；match 断言只钉 "auto-detect fell back" 短语与模型名，不钉家族列表 |
| `infer_model_family` 未来被赋予更多规则时 registry 告警语义漂移 | D3 判据依赖"causal_lm 仅由回退产生"——不变量以 registry 侧告警块注释注明（design 门控 Minor-2：消费侧注记，utils.py 保持零改动） |
| `_resolve_family` 是私有方法但被测试直接调用 | 既有先例（test_models.py:26-30）已如此；签名零改动保持兼容 |
