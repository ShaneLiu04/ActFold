# [AR032] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR032 |
| AR 主题 | 投机验证前向的块推进复用（verification-logit advance reuse；baseline 超越型，基于 AR021-AR022 投机回路） |
| 关联 SR | 无独立 SR（延续 AR021/AR022/AR031 投机机制线） |
| 日期 | 2026-10-22 |
| 状态 | Draft |
| 前置 | AR031 已归档（投机回路三档质量面冻结：aligned N=4+greedy 对齐 baseline，p50 7.02×；`greedy_win_stats`/`posthoc_confidence` 探针面可用） |
| 组件详设 | `specs/component-detail-design/` 不存在（组件详设缺失；以 AGENTS.md #39/#41-#46/#50 契约为边界真源） |

## 1. 背景与目标

### 1.1 已定位的架构级冗余

Fast-dLLM 块解码的投机路径（AR021/AR022）在每个 block 上花费**三次**折叠前向：

1. **draft serial forward**（构建候选的 draft logits）；
2. **`[N, T]` batched verification forward**（验证 N 行候选、min-prob 选胜者、
   commit、publish winner 行激活）；
3. **block-advance serial forward**（块推进：`next_token =
   argmax(logits[:, -1])`，append 到 canvas，再 append 下一 block 的 mask）。

其中第 3 次前向的**唯一产出**是一个 token：`argmax` 于「commit 后 canvas 的
下一位置分布」。而第 2 次前向已经对 **winner 行**（= commit 后的 canvas，
逐位相同）计算了全 `[T, V]` logits——其最后位置的 right-shifted logits
`shifted[j_star, -1, :]` 在数学上就是同一个分布（同输入、同模型、同权重）。
〔叙事简化注记（req r3 Minor，T008 处理）：本节按 S=1 几何叙述——每 block
一个 small block，winner 行 == commit 后 canvas；S>1（`small_block_size <
block_size`）时该等价仅对**最后一个 small block**成立（更早 small block
的 winner 行在后续位置仍携带 `mask_id`），冻结的广义契约见 FR1/FR2 的
「last small block」子句与 UT-3174 错源 fixture。〕
**验证 logits 在当前实现（以及 SpecInfer/EAGLE 等业界投机解码实现）中是
write-only 的——验证完即被丢弃**；本 AR 把它升级为块推进的一等推理来源，
整块省去第 3 次 serial forward。

### 1.2 最先进参考的核心思想与移植分析

| 参考 | 核心思想 | 对本项目的移植价值 |
|------|---------|-------------------|
| SpecInfer / EAGLE 系（token-tree speculative decoding） | 验证前向用 tree attention 一次接纳多 token，但验证 logits 本身用后即弃 | **反向启示**：块解码的块推进恰好需要验证 logits 的尾部——本 AR 让验证前向「一鱼两吃」；该复用面在 AR 域投机解码里不存在（无块推进概念），是 diffusion LLM 块解码特有的结构性冗余 |
| vLLM prefix caching / TensorRT-LLM kernel fusion | 推理加速的永恒主题是消除冗余前向/访存 | 本 AR 消除的是**前向级**冗余（整次 serial forward），比 kernel 级优化收益上限更高、机制更简洁 |
| AR031 honest negative（H3 refuted） | 不确定性排序不改变验证成本（cap 恒不绑定） | AR031 交付安全档；本 AR 回到**加速主线**：不改验证结构，而是消除验证邻接路径上的冗余前向 |

### 1.3 与「候选行数调度」方向的对比裁定

考虑过 EAGLE-2 context-aware dynamic draft tree 的直接移植（按在线置信度
EMA 调度 `num_block_candidates` ∈ [1, 4]）。**裁定不做**（记录为后续可选）：
AR031 工件实测每 prompt 仅触发 1-2 个投机块（decode 34/66 tokens，
posthoc block1-3 为 null），调度器预热期（首块必须 N_max）后几乎无块可调，
headroom 不足以支撑一个 AR；且 batched 折叠下 N 行共享 parent，行边际成本
远低于一次独立 serial forward。本 AR 的收益面（每次投机块省一整次
serial forward）不依赖块数分布，**每个投机块都全额收益**。〔叙事简化注记
（req r3 Minor，T008 处理）：本节的「省一整次」按 S=1 几何叙述；S>1 时
省略的是每个 block-advance-eligible block 的 1 次 serial forward（复用源
为块内最后一个 small block），计数契约以 FR1/FR2 冻结面为准。〕

### 1.4 目标

**baseline 超越型**：在 Fast-dLLM v2-1.5B 真实 checkpoint 上，对官方家族
baseline 声明**质量对齐 + p50 加速 > 1.0** 双指标；通过「同配置仅开关复用」
对照臂（reuse-off）**联合归因**加速来源为被省略的 block-advance 前向（H3）。
在 ladder 上，本 AR 把投机加速 rung（AR022 的 7.02×）继续推高，而不是新增
安全/正则化档。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- `actfold/models/fast_dllm_sampler.py`（改）：`sample` 循环的投机路径增加
  opt-in 块推进复用——省略 block-advance serial forward，`next_token` 取
  winner 行验证 logits 的最后位置 argmax（默认关 = AR031 末态逐字节不变）。
- `FastDLLMSamplerConfig` + `ActFoldConfig`：`speculative_advance_reuse`
  YAML 面（#5）。
- 机制等价验收面：合成模型 bit-exact A/B（eager 全重算路径）+ 折叠路径
  argmax 一致性记录面。
- `scripts/ar032_speculative_advance_bench.py`（新）：Fast-dLLM v2-1.5B
  真实面三臂基准（官方 baseline / reuse-off 同配置 / reuse-on）。
- 配置/导出/文档（README 六节全量重写 / RERUN / AGENTS #51 / CHANGELOG /
  GUIDE / ALGORITHM / EXPERIMENTS）。

**Out of Scope（本 AR 不做的）：**

- 不修改 #44/#45 冻结种子公式、min-prob 胜者规则、commit 语义、
  `greedy_win_stats`/`posthoc_confidence` 探针（复用机制是 commit 之后的
  **读取面**，不触碰投机决策本身）。
- 不修改非投机路径的 block-advance（无 `folded_model` 的官方 baseline、
  N=0 的 folded-serial 模式逐字节不变）。
- LLaDA/Dream 真实面（仅 Fast-dLLM；三 samplers 的推进结构不同——LLaDA
  无块推进 AR 步、Dream 为 super-step 窗口推进，复用面需按家族单独定义，
  记录为后续 AR；先例：AR031 仅 Fast-dLLM）。
- 候选行数 N 的在线调度（§1.3 已裁定 defer，headroom 不足）。
- 真实 checkpoint 的 CUDA graph 联动面差异分析（本机 WDDM #46 限制；
  折叠臂按设计 draft-3 模板运行 **eager 主声明面**（AR031 口径，
  `use_cuda_graph=False`），机制在 eager 与 graph 两路径下同构——复用读的
  是 `_forward` 的返回 logits，与其路径无关；UT + 合成面钉住语义。
  〔勘误 T007 补遗（T008）：draft-2 原文「bench 沿 AR022 配置
  `use_cuda_graph=True` 原样运行」与 §3.3 协议行勘误同源，一并修正〕

## 3. 功能需求

### 3.1 FR1 — 投机块推进复用机制（`fast_dllm_sampler`）

**描述：** 投机路径（`num_block_candidates > 0` 且携带 branch_ids-capable
`folded_model`）的 block-advance 步骤改为读取 winner 行验证 logits 的尾部
argmax，省略独立 serial forward；folding chain 直连 winner candidate branch。

**触发条件：** `FastDLLMSamplerConfig.speculative_advance_reuse = True` 且
投机路径激活（N > 0 + folded_model）。

**期望行为：**
- **`next_token` 定义（冻结）**：`right_shift_logits(batched_logits)
  [j_star, -1, :].argmax(dim=-1)`，其中 `batched_logits` 特指本 block
  **最后一个 small block** 的验证前向返回值（S>1 时只有最后一个 small
  block 的 winner 行 == commit 后 canvas；更早 small block 的 winner 行在
 后续位置仍携带 `mask_id`，其尾部 logits 不构成推进分布——实现必须钉住
  「last small block」来源，S=1 几何下退化为唯一来源）；输出形状
  `[1, 1]`（B=1 硬约束沿 #43）；`next_token[finished] = pad_id` 语义逐字
  保留。
- **输入等价前提（构造性）**：投机 commit 后 `x` 与**最后一个 small
  block** 的 `cands[j_star]` 逐位相同（fill positions 被 commit 覆写、其余
  区域本就同源）——该 winner 行 forward 的输入 == commit 后 canvas，故其
  尾部 logits 与独立 block-advance forward 的 `logits[:, -1]` 是同一分布。
- **省略范围**：仅投机块循环内的 block-advance forward 调用；canvas
  append（`next_token` cat + fresh masks cat）逐字保留；
  `return_history` 的 snapshot append 逐字保留（x 更新后的快照语义不变，
  history 面零回归）。
- **folding chain**：投机 commit 后 `_folding_parent` 已指向 winner
  candidate branch id（#43 既有语义）；省略 AR forward 后下一 block 的
  draft serial forward 以 winner 为 parent 直连——链条不断、零新增状态、
  零额外 publish。
- **边界**：`remaining <= 0` 时不产生 next_token（沿既有 `continue` 语义，
  省略的 forward 本就不跑）；`finished.all()` 提前 break 语义不变。
- **默认关**：`speculative_advance_reuse = False` 时全路径（含投机路径）
  逐字节不变。

**异常处理：** `speculative_advance_reuse=True` 且 `num_block_candidates == 0`
→ 构造期 `ValueError`（复用仅在投机路径有定义，#44 同型边界）。

**验收标准：**
- Given 默认配置（False），Then 投机路径与非投机路径输出与 AR031 末态
  逐位一致（含 history 快照序列）。
- Given 开启且投机路径激活，Then 每 block 的 `_forward` 调用数 ==
  `2·num_small_blocks`（draft + batched verification，每 small block 一对；
  省略前为 `2·num_small_blocks + 1`）；**省略数恰为 1 次/每个
  block-advance-eligible block**（`remaining > 0` 的块；末块沿既有
  `continue` 语义本就不推进），UT 用计数探针钉住（S=1 与 S>1 双几何）。
- Given 开启，Then `next_token` 的来源为最后一个 small block 的验证
  logits（S>1 fixture 钉住：错拿非最后 small block 的 logits 必须被 UT
  捕获——其 winner 行含 `mask_id` 尾部，推进 token 必然不同）；shape/
  `pad_id` 覆写/append 顺序与关闭时的 block-advance 产物完全同构。
- Given True + N=0，Then 构造期 `ValueError`。

### 3.2 FR2 — 机制等价硬验收（合成面）

**描述：** 机制语义（next_token 来源正确性）由合成模型 bit-exact A/B 钉住；
折叠数值边界诚实记录。

**触发条件：** 合成模型（tests 既有 fixture 家族）+ 投机路径 A/B。

**期望行为：**
- **eager 全重算硬门**：合成模型、`folded_model` 携带但 gate `tau=1.0`
  （cosine 相似度 `sim > tau` 判 stable，`tau=1.0` ⇒ 全量重算、零复用；
  **fixture 约束：CPU 非 split 路径**——#41 的 batched-vs-serial bit-exact
  契约不覆盖 split FFN 路径的 gemv/gemm 1-ULP 边界）路径下，reuse-on 与
  reuse-off 的**完整轨迹**（`decode_final` 序列 + history 快照序列）逐位
  一致——两个 forward 在全重算下是同一计算图、同输入，argmax 必然一致；
  分叉 ⇒ 机制语义错误（非数值噪声），硬 FAIL。
- **折叠路径记录面**：折叠开启（真实复用语义）时，on/off 两模式的
  `next_token` 逐块一致率为**记录面**（不同 parent 的复用集差异可产生
  ULP 级 logits 差，argmax 平局邻域内可能分叉——这是 #41 已知的折叠数值
  属性，不是机制错误）；UT 在 CPU 合成面断言一致率并记录，真实面由 FR3
  的质量门验收。
- **S>1 几何 fixture**：`small_block_size < block_size` 的合成面钉住
  复用来源（FR1「最后一个 small block」子句）与省略计数
  （`2·num_small_blocks` vs `+1`）。
- **ULP 边界诚实记录**：文档（design/AGENTS #51）明确声明**两类**数值
  漂移面：① next_token 的数值来源切换（batched-verification 折叠 forward
  vs 独立 serial 折叠 forward，parent 不同），argmax 平局邻域内可能翻转；
  ② **下游链形变化**——on 臂省略 AR forward 后，后续所有 forward 的
  folding parent 直连 winner branch（off 臂经 AR branch 中转），折叠
  draft logits 同样可能 ULP 漂移。两类面均由 FR3 双指标硬门 + next-token
  一致率记录面联合控制（不构成机制错误）。

**异常处理：** eager 全重算硬门分叉 → 机制 bug，修复后重验（允许仪器修复
性质的单次重跑，留档）。

**验收标准：**
- Given 合成模型 + tau=1.0 全重算（CPU 非 split 路径）+ 投机开启，When
  on/off A/B，Then 完整轨迹逐位一致（硬门）。
- Given 合成模型 + S>1 几何（`small_block_size < block_size`）+ 投机
  开启，When on/off A/B，Then 复用来源为最后一个 small block 的验证
  logits（错源必须被捕获）、省略计数 == 1 次/block-advance-eligible
  block（末块不推进，两侧均为 0）。
- Given 合成模型 + 折叠开启 + 投机开启，When on/off A/B，Then next_token
  一致率记录于测试输出；decode_final 在分叉时如实断言分叉（不伪造一致）。
- Given 非投机路径，When on/off A/B，Then 逐位一致（复用不触达非投机路径）。

### 3.3 FR3 — Fast-dLLM 真实面三臂基准

**描述：** Fast-dLLM v2-1.5B 真实 checkpoint 上，官方 baseline /
reuse-off 同配置 / reuse-on 三臂对照，双指标硬标准，reps ≥ 30 预承诺单次
运行。

**触发条件：** `python -m scripts.ar032_speculative_advance_bench [--quick]`。

**期望行为：**
- **三臂**：① official baseline（家族官方 recipe，`folded_model=None`）；
  ② reuse-off（ActFold 投机配置：N=4 + greedy anchoring，
  **即 AR022 aligned 模式**，复用关）；③ reuse-on（与 ② **完全同配置**，
  仅 `speculative_advance_reuse=True`）。② 与 ③ 的差异**仅为复用开关**
  （机制归因臂）。〔勘误 T007：srs draft-2 原文称 ②「+ graph pool」；
  交付基准的折叠臂运行 **eager 主声明面**（`use_cuda_graph=False`，
  AR031 模板口径，见下方协议行勘误）——设计 draft-3 的 bench 模板即
  `ar031_adaptive_uncertainty_bench.py`，机制本身路径无关（读 `_forward`
  返回值，D7）〕
- **硬标准（冻结，先于数据收集）**：
  - **H1 质量对齐（③ vs ①，含 ③ vs ② 一致性子句）**：posthoc overall
    confidence gap ≤ 1e-3（沿 AR023/AR031 H1 边界比较口径与 IEEE 噪声
    余量）、decode length ③ 与 ② 相等（next_token 一致性的 stop-token
    面）、merged block exact-match ③ 不低于 ② - 1e-9；任一 `None` →
    refuted。
  - **H2 p50 加速（③ vs ①）**：`p50(①)/p50(③) > 1.0` 严格成立。
  - **H3 机制归因（③ vs ②）**：`p50(②)/p50(③) > 1.0`（省略一整次
    serial forward 必须在墙钟上显形）**且** 计数面
    `off_serial − on_serial == eligible_blocks` 为硬门（单独可 refute；
    逐臂 `_forward` 调用数按臂记录——on 臂每 block 恰好少
    1 次/block-advance-eligible block，末块不推进两侧均为 0——机制级
    计数探针）。〔勘误 T007：draft-2 原文计数面为 report-only；冻结
    UT-3178 契约将其升格为硬门——强化方向，安全〕
  - **H4 诚实账本**：③ 的 next-token 一致率（③ vs ② 逐块 sidecar 对比
    ——两次独立采样，轨迹在无 ULP 分叉时应逐位一致；分叉块数如实记录，
    **不作为硬门**，作为 ULP 边界的经验证据）；canary/ledger 沿 #45
    `LEDGER_C` 口径（on 臂的 serial 事件数按省略后的实际调用计）；repro
    双跑行严格相等（不一致 ⇒ H4 refuted 且 overall refuted，H1-H3 保留
    各自可重算判定——冻结 UT 契约 UT-3178/3179 无 blocked 档）。〔勘误
    T007：draft-2 原文为「不一致 ⇒ 全部 blocked」（AR031 口径）；T004
    Red 冻结的判定函数/测试契约为 refuted 语义并先于 formal 运行钉死，
    formal 工件按该语义交付——以冻结契约 + 勘误记录为准，不回改已冻结
    判定〕
- **协议**：冻结判定先于数据收集（脚本内冻结）；reps=30 预承诺单次；
  round-robin 交错；warmup 丢弃；#39 计时口径；AR019 venv
  （transformers==4.53.1）；`--quick` 写独立 `_quick.json`，绝不写正式
  路径；折叠臂运行 **eager 主声明面**（`use_cuda_graph=False`，AR031
  模板口径；正式工件覆盖保护：默认正式路径已存在则拒绝覆写）。〔勘误
  T007：draft-2 原文为「`use_cuda_graph=True` 沿 AR022 配置原样」；设计
  draft-3 的 bench 模板钉死为 `ar031_adaptive_uncertainty_bench.py`
  （eager 主面 + 图联动面 BLOCKED-DEFERRED），交付按设计执行；机制本身
  路径无关（D7），eager 面为完整机制验证面〕
- **仲裁**：质量或加速任一硬标准不满足 → honest negative 原样交付
  （refuted/inconclusive），绝不调参重跑。

**异常处理：** 环境不满足（transformers 版本/显存/权重缺失）→ 构造期显式
报错退出（rc=2），不写部分工件；单臂失败（OOM 等）→ 进程崩溃、不写任何
工件（零伪造数字，AR031 实际口径）。〔勘误 T007：draft-2 原文为
「blocked 工件 + 探针交叉引用」——AR031 bench 实际并无 OOM blocked-工件
路径（其 blocked 是 repro 判定档），该引用不准确；交付按实际口径如实
修正〕

**验收标准：**
- Given quick，Then 冒烟通过并写 `_quick.json`（几何缩小版，同一三臂/判定）。
- Given 正式运行完成，Then 工件含三臂全指标、H1-H4 判定、逐样本记录、
  forward 调用计数；每个数字可由工件重算。
- Given 任一硬标准 refuted，Then 工件原样记录（honest negative 交付）。

### 3.4 FR4 — 配置/接线/导出/文档

**描述：** 配置面、公共导出与文档收口。

**期望行为：**
- `FastDLLMSamplerConfig` 新键 `speculative_advance_reuse: bool = False`；
  `ActFoldConfig` 消费同名 YAML 键（#5，零 unknown-key 警告）；True + N=0
  的 `ValueError` 验证落在 sampler config `__post_init__`（#44 同型）。
- AGENTS.md 新增 #51 契约条目（复用语义/等价前提/ULP 边界/计数探针）；
  README 六节全量重写（AR 收口规则）/ RERUN 步骤 20 / CHANGELOG /
  ALGORITHM §17 / EXPERIMENTS §4.5b（沿 AR031 §4.5a 编号先例；设计文
  原「§4.6」与既有 Ablation 编号冲突，T008 修正）/ GUIDE 第三十七部分。
- 复现指南：RERUN 记录精确命令、资源表（GPU 类/VRAM/墙钟实测）、
  verdict 语义、honest negative 语义（大型实验复现规则全文适用）。

**验收标准：**
- Given 新 YAML 键，Then 字段正确消费、无 unknown-key 警告。
- Given 默认配置，Then 全仓行为与 AR031 末态逐位一致（零回归 UT）。
- Given 文档七件套，Then 计数/编号/措辞与工件一致（一致性终检）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | 默认零回归 | 默认配置下与 AR031 末态逐位一致（全量 pytest + 定向 UT） |
| 开销 | 机制开销 | 零新增前向、零新增设备同步（复用是既有 logits 的切片读取） |
| 速度 | 收益面 | 每个投机 block 省 1 次 serial forward（UT 计数探针钉住 == `2·num_small_blocks`/块；省略恰 1 次/block-advance-eligible block） |
| 可复现 | 冻结语义 | next_token 定义冻结；同输入同种子 → 同输出（合成面钉住） |
| 性能门 | 测试时长 | AR032 单测套件 < 30 s；bench quick < 90 s |
| 协议 | 预承诺 | 判定冻结于数据收集前；单次运行；honest negative 原样交付 |
| 环境 | 真实面 | AR019 venv / transformers==4.53.1 / Fast-dLLM v2-1.5B 权重 / 16GB 显存上限内 |

## 5. 约束与假设

**约束：**
- 不改 #44/#45 冻结公式与胜者规则、#43 fold-context 契约、#37 图捕获
  契约、#46 环境法则。
- 全程零 mock；基准为真实张量运算与真实 checkpoint。
- 本机 16GB WDDM：真实面折叠臂运行 eager 主声明面（AR031 模板口径，
  `use_cuda_graph=False`；设计 draft-3 钉死；机制本身路径无关——读
  `_forward` 返回值）。〔勘误 T007 补遗（T008）：draft-2 原文「graph pool
  开启 + 摊销语义诚实注记（AR022 先例）」与 §3.3 勘误同源，一并修正〕
- B=1 硬约束沿 #43（投机路径既有约束，复用不放宽）。

**假设：**
- **输入等价前提（构造性，非假设）**：commit 后 `x == cands[j_star]`
  （最后一个 small block）逐位成立（FR1 论证；UT 钉住）。
- winner 行尾部 logits 的 argmax 与独立 serial forward 的 argmax 在真实
  checkpoint 上实际一致（ULP 分叉概率低）——**经验假设**，由 FR2 折叠
  记录面 + FR3 质量门联合验收；若真实面大规模分叉 → honest negative。
- Fast-dLLM v2-1.5B 在 16GB 上 reps=30 三臂可行（AR031 三臂先例实测可行，
  本 AR ③ 臂比 ② 更快）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 块推进（block advance） | Fast-dLLM 每 block commit 后的 AR 步：`next_token = argmax(logits[:, -1])`，append 后展开下一 block 的 mask |
| 验证前向复用 | 省略块推进 serial forward，`next_token` 取 winner 行验证 logits 尾部 argmax（本 AR 机制） |
| 输入等价前提 | commit 后 canvas 与**最后一个 small block** 的 winner 候选行逐位相同 → 两处 logits 同分布（构造性） |
| ULP 分叉边界 | 折叠路径不同 parent 的复用集差异产生的浮点差，在 argmax 平局邻域可能翻转选择（#41 已知数值属性）；含下游链形变化面（on 臂后续 forward parent 直连 winner branch） |
| reuse-off 臂 | ActFold 投机配置（N=4 + greedy，eager MFF 主声明面）、复用关（= AR022 aligned 配置，机制归因对照；T007 勘误：原「+ graph pool」措辞随协议行勘误一并修正） |

## 7. 考虑过并裁定的替代方向

| 方向 | 裁定 | 理由 |
|------|------|------|
| 候选行数在线调度（EAGLE-2 dynamic tree 移植） | defer | AR031 工件实测投机块数 1-2/prompt，预热期后无可调块，headroom 不足 |
| LLaDA/Dream 推进复用 | 后续 AR | 两家族推进结构与 Fast-dLLM 不同（无块推进 AR 步 / super-step 窗口），复用面需单独定义 |
| 验证 logits 全量保留供多 block 复用 | 拒绝 | canvas 在下一 block 会增长（append mask），旧 logits 形状失效；复用面仅存在于「同 canvas 的紧邻推进步」 |

## 8. 变更记录

| 版本 | 日期 | 变更 |
|------|------|------|
| draft-1 | 2026-10-22 | 初稿（需求澄清按用户授权自动裁定：方向分析含 EAGLE-2/SpecInfer 移植评估与 N 调度 headroom 否决；机制边界/等价前提/ULP 边界/三臂协议按先例 AR022/AR031 冻结） |
| draft-2 | 2026-10-22 | 门控 r1 FAIL 修复（子代理 `ses_e9edfbb51ffeh97MjuyOSFJGWI`）：① Critical——FR2 全重算硬门 tau=0 反向，改 tau=1.0（`sim > tau` 判 stable）+ CPU 非 split fixture 约束；② Important——复用来源冻结为「块最后一个 small block 的验证前向」+ 省略计数推广 `2·num_small_blocks` vs `+1` + S>1 fixture（FR1/FR2/T003）；③ #45→#43 引用修正；④ ULP 账本补下游链形变化面；⑤ H1 标签改「③ vs ①，含 ③ vs ② 一致性子句」 |
| draft-3 | 2026-10-22 | T007 review 收口勘误（审查子代理 `ses_e9e0bc23dffeR1htgBsxW1iO8X`，PASS + 4 Important）：① repro 不一致语义「全部 blocked」→ 冻结 UT 契约的 refuted 语义（H4/overall refuted，H1-H3 保留）——不回改已冻结判定；② H3 计数面 report-only → 硬门（强化方向）；③ 三臂「graph pool」与协议「use_cuda_graph=True」→ 交付 eager 主声明面（设计 draft-3 模板即 AR031）；④ 异常行「blocked 工件」→ 实际口径（崩溃不写工件，零伪造）；⑤ 附跨 rung 计时漂移记录（baseline p50 同几何 3096.76ms[AR022,graph]→17279.30ms[AR031]→26305.87ms[AR032,eager]，跨会话 WDDM 主机状态噪声；同 run A/B 比值内部有效，跨工件绝对值不可直接比——T008 README ladder 措辞依据） |
| final | 2026-10-22 | T008 ST 收口：① req r3 两个非阻塞 Minor 落地（§1.1/§1.3 叙事加 S=1 简化注记，广义契约指向 FR1/FR2 冻结面）；② T007 勘误补遗两处（§2 Out of Scope 与 §5 约束行的 graph-mode 残留，与 §3.3 勘误同源）；ST 验收见 st-cases.md |
