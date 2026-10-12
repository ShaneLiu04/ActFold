# [AR010] 需求设计说明书 — 模型家族推断统一与回退可见性

| 字段 | 内容 |
|------|------|
| AR 编号 | AR010 |
| AR 主题 | `ModelRegistry._resolve_family` 与 `infer_model_family` 去重（单一实现源）+ auto 回退 causal_lm 的一次性告警 |
| 关联 SR | 指南 6.2 :241（"`registry._resolve_family` 子串匹配（"fast" in name）与 `models/utils.infer_model_family` 双份维护；未知 diffusion 家族静默落 `causal_lm`"）；AGENTS #10（不静默回退） |
| 日期 | 2026-10-12 |
| 状态 | Final（req/design 门控 PASS，T001–T004 passing，review PASS，ST 9/9 Go，已归档） |

## 1. 背景与目标

两处家族推断缺陷：

1. **双份维护**：`ModelRegistry._resolve_family`（registry.py:77-90）的
   auto 分支与 `models/utils.infer_model_family`（utils.py:12-28）是**逐行
   重复**的实现（同子串规则：`"llada"` / `"dream"` / `"fast" and "dllm"`
   → 同 causal_lm 回退）。`infer_model_family` 当前**零生产调用**（全库
   grep：仅 registry 侧被 `load` 使用，utils 侧无任何调用方）——两条
   实现可独立腐化（改一处漏一处）。
2. **静默回退**：`ActFoldConfig.model_family` 默认 `"auto"`
   （actfold\utils\config_manager.py:57），benchmark_runner
   （actfold\eval\benchmark_runner.py:127/:143）生产路径默认走
   auto 推断；当名字不含已知子串时**静默**落 `causal_lm`（registry.py:89-90
   注释自认 "safest fallback"）。对真因果模型（llama/qwen）这是正确默认，
   但对**未知 diffusion 家族**（如 MDLM/SEDD 系）则是静默错误包装——
   AGENTS #10 视角下用户不知道自己的 diffusion 模型被当成了 causal LM。

**子串规则本身保留（勘误）**：指南 :241 点名 "'fast' in name" 的子串
匹配，经分析该规则要求 `"fast" and "dllm"` **双子串同时命中**，误报面
极窄（需名字同时含两子串且非 Fast-dLLM）；而 token 化匹配会破坏
"FastDLLM" 连写命名（lower→`fastdllm` 是单个 alnum token，子串命中 ✓ /
token 等值 ✗）。故匹配规则不动，本 AR 只收口重复与静默。

目标：家族推断有**唯一实现源**（utils.infer_model_family），auto 回退
**可见**（UserWarning + 显式传参指引）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `ModelRegistry._resolve_family` 的 auto 分支**委托**
  `actfold.models.utils.infer_model_family`（registry 侧删除重复规则；
  显式 `model_family != "auto"` 路径原样保留在 registry）；
- `_resolve_family` 在"auto 推断且结果为 causal_lm 回退"时发
  `UserWarning`（每次 load 一条——load 非热路径；消息含模型名、
  "auto-detect fell back to causal_lm"、显式传 `model_family` 指引、
  可用家族列表）；
- 行为等价与告警触发的单元测试（既有 `test_registry_resolve_family`
  5 断言零改动全过 = 去重不改行为的钉死）；
- 文档：指南 6.2 :241 勾选（含子串规则勘误注）+ 第十七部分、CHANGELOG
  AR010 节；AGENTS 若 review 判定需要则补一句"唯一实现源"指针。

**Out of Scope（本 AR 不做的）：**
- 子串匹配规则本身的改造（token 化/正则/前缀树）——勘误论证见 §1，
  双子串 AND 已足够窄，改造反而破坏连写命名；
- 基于加载后 `AutoConfig.model_type` 的二次校验（需网络/本地权重，
  把推断时点从纯字符串推迟到 config 可得，工程量与收益不匹配）；
- `ModelRegistry` 注册表项增删（`_REGISTRY` 不动）；
- benchmark_runner/scripts 的 `model_family` 传参方式（它们可继续用
  `"auto"`，告警是提示而非强制）。

## 3. 功能需求

### 3.1 家族推断单一实现源（去重）

**描述：** registry 不再维护自己的子串规则，委托 utils。

**触发条件：** `ModelRegistry.load(..., model_family="auto")`。

**期望行为：**
- `_resolve_family`：`model_family != "auto"` → 原样返回（不变）；
  `"auto"` → `return infer_model_family(model_name_or_path)`；
- `infer_model_family`（utils）成为唯一规则实现，规则**逐字不变**
  （llada/dream/fast+dllm 子串、causal_lm 回退、大小写折叠）；
- registry 导入 `actfold.models.utils` 不成环（utils 仅依赖
  torch/transformers/标准库，无模型类依赖——设计阶段验证）。

**异常处理：** 无新异常；`load` 的 unknown-family ValueError 路径
（registry.py:70-73）逐字不变。

**验收标准：**
- Given 既有 `test_registry_resolve_family` 全部 5 断言（llada/dream/
  fast_dllm/gpt2-auto/causal_lm-explicit），When 全量回归，Then 零改动
  全过（去重不改行为）；
- Given `"GPT-LLaDA-8B"`、`"FastDLLM-v2"`、`"some/dream-7b"` 等混合
  大小写/连写名，When `_resolve_family(name, "auto")`，Then 结果与
  改动前逐位一致（**UT-901** 等价矩阵）。

### 3.2 auto 回退 causal_lm 的一次性告警

**描述：** auto 推断落入 causal_lm 默认时明确告知调用方。

**触发条件：** `model_family == "auto"` 且 `infer_model_family` 返回
`"causal_lm"`（即无已知子串命中）。**落点在 `_resolve_family` 层**：
直接调用 `_resolve_family(name, "auto")`（如既有 test_models.py:29）
同样触发告警——"每次 load 至多一条"按经 load 的调用链理解，非 load
独有（design 阶段落点歧义消除，req 门控 WARN-4）。

**期望行为：**
- 发 `UserWarning`：含 (a) "auto-detect fell back to causal_lm" 与
  模型名；(b) 指引——若该模型是 diffusion 家族应显式传
  `model_family`；(c) 可用家族列表（`ModelRegistry.list_models()`）；
- 每次 `load` 调用至多一条（load 非热路径，无实例级 once 标志需求；
  registry 为类无实例）；
- **不告警的情形**：显式 `model_family != "auto"`（用户已选，含显式
  `"causal_lm"`）；auto 命中已知子串（llada/dream/fast_dllm）。

**异常处理：** 告警不改变返回值——回退仍为 causal_lm（真因果模型的
正确默认，AGENTS #10 的可见性补齐而非行为变更）。

**验收标准：**
- Given `"gpt2"` + `model_family="auto"`，When `ModelRegistry.load`
  （mock 构造器），Then 恰好 1 条该 UserWarning（**UT-902**，
  `pytest.warns(UserWarning, match=...)` 断言消息含 "auto-detect fell
  back to causal_lm" 关键短语与模型名，防消息退化不可测）且 family
  解析为 causal_lm；
- Given `"gpt2"` + `model_family="causal_lm"`（显式），When load，Then
  **零**该 UserWarning（显式选择不告警钉死，**UT-903**）；
- Given `"some/llada-8b"` + `model_family="auto"`，When load，Then
  **零**告警（已知子串命中不告警钉死，**UT-904**）；
- Given pyproject 无 `-W error`，When 全量回归，Then 既有测试零失败。

### 3.3 文档收口

**验收标准：**
- 指南 6.2 :241 勾选，附勘误注（子串规则经分析保留——双子串 AND 误报
  面极窄，token 化破坏 FastDLLM 连写命名；本 AR 收口的是去重 + 回退
  可见性）+ 第十七部分回链；
- CHANGELOG AR010 节（Changed：去重；Added：告警）；
- AGENTS 视 review 判定补"唯一实现源"指针（可选）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 静态检查 | mypy 标准命令 clean / pyflakes 零 / black 100 列 |
| 质量 | 全量回归 | `pytest tests\ -q -m "not slow"`：773 passed 基线零失败 |
| 兼容 | 公开 API | `load`/`load_model`/`infer_model_family` 签名零改动 |
| 性能 | 推断开销 | 纯字符串操作不变（告警仅 fallback 路径，每 load 一次） |

## 5. 约束与假设

**约束：**
- 推断规则逐字不变（去重是搬移不是改写）——既有 5 断言零改动全过是
  硬门；
- 告警是可见性补齐，不改变任何返回值/包装行为（真因果模型继续正确
  落 causal_lm）；
- `infer_model_family` 保持纯函数（告警在 registry 层，utils 无副作用）；
- AGENTS #10/#23 对齐。

**假设：**
- `load` 非热路径（每模型加载一次），每次 load 一条告警不构成噪音；
- benchmark/scripts 显式传 family 的路径不受影响（spec["family"] 多为
  显式；即便 "auto" 也是提示性告警）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 双份维护 | `_resolve_family` 与 `infer_model_family` 各自实现同一子串规则，可独立腐化（本 AR 消除对象） |
| auto 回退 | auto 推断无已知子串命中时落 `causal_lm` 的默认行为（保留，加可见性） |
| 双子串 AND | fast_dllm 的匹配需 `"fast" in name and "dllm" in name` 同时成立——误报面极窄的结构性原因 |
