# [AR021] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR021 |
| AR 主题 | 生产生成回路折叠接线：sampler 前置条件线程化 + remasking 链折叠（强制散度活跃块）+ 批量投机块验证 + 真实 generate() 级基准（P1-7 第七档） |
| 关联 SR | P1-7 多分支验证墙钟（AR015-20 六档延续）；AR019 srs §Out-of-Scope "FastDLLMAdapter 生产接线"、AR020 srs §Out-of-Scope 同项（两代连续点名） |
| 日期 | 2026-10-14 |
| 状态 | Draft（rev 3，req 门控第 2 轮修复） |

## 1. 背景与目标

AR015-AR020 六档把批量验证栈建成：批量分支前向（AR015）→ 批量 CUDA 图（AR016）→
自适应容量（AR017，批量路径专属）→ 发布压缩（AR018）→ 真实 checkpoint 端到端
（AR019）→ 批量 cache put（AR020，发布窗口 1.77-1.97×、账本 L·N+N→O(L+N)、
e2e ab vs baseline 2.40-2.67×）。**但该栈至今只有基准脚本在调用——生产
`generate()` 路径零调用方**："综合框架 vs baseline"从未在真实生成面测过。

### 1.1 生产回路现状与缺口（代码事实）

生产生成回路 = `FastDLLMSampler.sample`（`fast_dllm_sampler.py:111-259`，fast-dLLM v2
块解码配方：块→小块→remasking 迭代，每迭代一次 `[1, cur_len]` 全画布前向，
`_next_folding_branch` 跨迭代链父（#26））。缺口四项：

| # | 缺口 | 代码事实 | 后果 |
|---|------|---------|------|
| 缺口-1 | sampler 不传 `position_embeddings` | `diffusion_sampler.py:139-178` `_forward` 无 pe 参数 | 真实 Fast-dLLM-v2 remote 模型上 MFF 父 prefill 直接 `TypeError`（AR019 探针 P1 已证）——折叠路径在真实 checkpoint 上**必崩** |
| 缺口-2 | 掩码语义错 + 对象不钉 | `fast_dllm_sampler.py:187,231` 每迭代 `attention_mask = x != pad_id`（`[B,T]` padding 掩码，新张量对象/次） | (i) 经 MFF 直调 remote 层时绕过模型内部 `eval_block_diff_mask` 块因果掩码构建，语义错；(ii) 图前置条件 `attention_mask is not self._graph_mask`（#37）每次失配——现状代码**捕获一次后永不重放**（首个合格调用捕获、后续静默 eager），CUDA 图收益归零 |
| 缺口-3 | `[N,T]` 同父批量验证形态不存在 | remasking 迭代为串行 `[1,T]` 前向；eval 侧 `folded_generate` 被 `BaseEvalAdapter` 以 `num_branches_per_step=1` 调用（`base_adapter.py:193`）且变长步永不入图（#20，`test_folded_generation.py:592-628` 钉死） | AR016-020 优化的批量图/自适应容量/批量发布**在生产中零行使**；AR019 基准的"块验证工作负载"是真实生产中不存在的抽象 |
| 缺口-4 | eval ActFold 路径测非所优 | eval 的 actfold_latency = `folded_generate` 串行 AR 逐 token 循环（非块解码） | 现有"生产"数字与被优化的栈无关 |

**机理依据与最大风险（本 AR 的核心论证，含负面证据）**：

- **前缀半边（有正面证据）**：块因果语义（block i 只 attend block ≤ i，AR019 P3/P4）
  保证**已解码前缀块**跨迭代位稳定 ⇒ 父 FFN 缓存逐位有效——这部分折叠是精确的。
- **活跃块半边（有负面证据，本 AR 最大机理风险）**：remasking 链的父=上一迭代，
  活跃小块内存在**部分** mask→token 翻转。AR019 P5（翻转 token 工作负载）实测：
  同块未翻位置经块内双向注意力被翻转传播，tau=0.95 gate 判其为 stable → 折叠成为
  **近似**（parity 92.9% 先例）。即"未翻位置复用父缓存"不精确。
- **结论（FR2 的设计含义）**：链折叠要精确必须**由构造保证**——引入**强制散度活跃
  块区**（新语义）：折叠路径下当前活跃小块整区强制判散度（全量重算，不依赖 gate 对
  同块未翻位置的判断），前缀块照常走 gate（其精确性有 P3/P4 证据）。如此
  folded-serial = "只重算活跃小块的生成"，parity 由构造精确；P5 风险被显式绕开而非
  假装不存在。
- **形状增长半边（缺口-2 的第二层）**：画布每块增长（AR 步追加 token + 新 mask 块）
  ⇒ 即便掩码/pe 对象钉定，既有 MFF runner 也是**单形状**（runner 存在后形状失配
  warn-once 永不重捕获，`architecture_utils.py:761-773` 串行 / `:924-929` 批量）——
  生产整代生成需要**形状键控 runner 池**（每画布形状一个捕获 runner，新语义），
  否则图收益只覆盖第一个形状窗口。
- 小块迭代窗口内画布形状恒定 ⇒ 对象钉定 + 形状键控池后，串行图可在生产回路**真正
  捕获并重放**（现状：捕获一次后永不重放）。批量投机块验证（N 候选一次 `[N,T]` MFF
  调用）则把 AR019 工作负载变成真实生产路径，AR016/017/018/020 批量面首次被生产行使。

### 1.2 口径延续

- 基准模型/负载：Fast-dLLM-v2-1.5B 本地权重 + AR019 venv（transformers 4.53.1），
  宿主 `perf_counter` + sync 夹逼主报（AR019/20 同族口径，#39 CUPTI null 注记）。
- baseline = 无折叠 sampler（raw 前向，现状逐字节）；综合框架 = 折叠栈全开
  （串行图（FR2 形状键控池）/批量图+自适应容量+批量发布（FR3 批量路径，AR017 自
  适应为批量路径专属 #37）/批量 put（AR020），配置由 `ActFoldConfig` 既有旋钮驱动）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- FR1 sampler 折叠前置条件线程化：`DiffusionSampler._forward` 增 `position_embeddings`
  透传 + 块因果且 **pad 排除**的 `[T,T]` 掩码按画布形状一次构建、对象钉定（仅折叠
  路径生效；raw 路径逐字节不变）
- FR2 生产 generate() 折叠接线：`FastDLLMModel.generate(folded_model=MFF)` 全链路在
  真实 checkpoint 可跑——**强制散度活跃块区**（链折叠精确由构造保证，新语义）+
  **形状键控 graph runner 池**（跨块增长的图收益，新语义）+ 默认无 folded_model 时
  行为逐字节不变
- FR3 批量投机块验证（opt-in）：`num_block_candidates=N` 模式——父 logits top-k 起草
  N 个候选小块填充 → 单次 `[N,T]` MFF 批量调用验证（branch_ids + 共享父分支）→
  置信度准则选定提交；默认关闭时 sampler 输出逐字节不变
- FR4 真实 generate() 级基准脚本 + 工件（baseline / folded-serial / folded-batched
  三模式：墙钟 + 输出质量证据 + 假设-判定分离，reps=30 预承诺单次定案）
- FR5 文档收口五件（含 README 全量中文重写，AGENTS.md 既定规则）

**FR3 与 FR1/FR2 同 AR 的耦合依据**：FR3 复用 FR1 的掩码/pe 钉定与 FR2 的形状键控
池/发布路径（同一接线下游）；拆分会造成两次接线上下文与两个基准工件重复。若设计
期发现 FR3 风险面过大（起草/选定语义不可单 AR 定案），拆分为后续 AR 的退路在
tasks.md T002 备注中显式记录（基准届时以 folded-serial 两模式先行）。

**Out of Scope（本 AR 不做的）：**

- LLaDA/Dream sampler 同类接线（remote 代码各异；Fast-dLLM v2 为主基准对象）
- 变长/多 parent 批量（AR016 遗留，正交）
- eval 适配器改造（`BaseEvalAdapter`/LM-eval 任务级质量跑分——后续独立 AR）
- 批量投机块的接受策略族谱（本 AR 单一准则定案，族谱留后续）
- LLaDA-8B/Dream-7B（16GB 不可行）

## 3. 功能需求

### 3.1 FR1 sampler 折叠前置条件线程化

**描述**：`DiffusionSampler._forward` 及 FastDLLM 掩码构建支持折叠路径所需的
`position_embeddings` 与块因果掩码。

**触发条件**：sampler 以 `folded_model`（MFF）运行（remote 模型需 pe；图路径需对象
钉定掩码）。

**期望行为**：
- **pe 透传**：`_forward(..., position_embeddings=...)` 可选参数；非 None 时原样传给
  folded_model（AR019 四构建面契约沿用：entry 校验 host 元数据、`is` 钉定、mismatch
  静默 eager）。raw 路径不传（现状不变）。
- **块因果 + pad 排除掩码**：折叠路径下按画布形状（block_size 感知）构建
  `block_causal[q,kv] = (q//bs >= kv//bs) AND kv 非 pad` 的 `[T,T]` 布尔掩码（AR019
  `_build_context` 块因果式 **∪ raw 路径 `x != pad_id` 的 pad 排除语义**——sampler
  恒以 pad 对齐画布，纯块因果式会 attend pad 位，与 raw 路径语义不等价）；**同形状
  只构建一次、对象钉定复用**；raw 路径维持 `x != pad_id` padding 掩码逐字节不变。
- **语义匹配义务**：折叠路径掩码必须与底层模型 raw 路径的实际语义匹配（Fast-dLLM
  remote = 块因果+pad 排除；**合成 parity 面的合成模型必须与其 raw 路径语义自洽**
  ——设计期定案合成面构造，禁止"合成模型 raw 语义与折叠掩码不一致"的假 parity）。
- **掩码/pe 供给面**：sampler 在画布形状变化时（跨块增长）重建并重钉；同小块迭代
  窗口内恒同对象（图捕获前置条件）。

**异常处理**：pe 形状/头维不一致按 AR019 契约（entry `ValueError`；replay `is` 失配
静默 eager）；块掩码构建失败 → 显式异常，不静默回退 padding 掩码。

**验收标准**：
- Given 合成模型 + MFF，When remasking 链折叠运行，Then 同画布形状窗口内所有
  folded_model 调用收到 `is` 同对象的 mask 与 pe（spy 断言），raw 路径调用不收到 pe
  且掩码仍为 padding 式（既有 UT 面零扰动）
- Given 含 pad 对齐的画布，When 折叠路径掩码构建，Then 掩码在 pad 列恒 False
  （与 raw `x != pad_id` 语义等价面 UT）
- Given 真实 checkpoint，When folded 路径运行，Then 不再出现 AR019 P1 类 `TypeError`
  （基准面验证）

### 3.2 FR2 生产 generate() 折叠接线（强制散度活跃块 + 形状键控 runner 池）

**描述**：`FastDLLMModel.generate(..., folded_model=MFF)` 在真实 checkpoint 上全链路
可运行：remasking 链折叠（父=上一迭代，活跃小块强制散度）+ 跨块增长的形状键控
graph runner 池。

**触发条件**：调用方显式传 `folded_model`（demo/BenchmarkRunner/基准脚本）。

**期望行为**：
- **链折叠 + 强制散度活跃块区（新语义，P5 风险的构造级绕开）**：折叠路径下当前
  活跃小块的 token 区间在 gate 判定前**强制判散度**（整区全量重算，不依赖 gate 对
  同块 mask→token 部分翻转的判断——P5 实证该判断在 tau=0.95 下会误判 stable）；
  已解码前缀块照常走 gate（块因果位稳定 ⇒ similarity≈1 ⇒ 复用，P3/P4 正面证据）。
  传递面为显式 token 级散度强制标记（设计期定案 MFF 接口形态），默认 None 时 MFF
  行为逐字节不变（既有 UT 面零扰动）。
- **形状键控 graph runner 池（新语义，跨块增长的图收益）**：**激活面=显式 opt-in**
  （MFF 构造参数 / `ActFoldConfig` 旋钮，**默认关**——关闭时既有单 runner
  warn-once-never-recapture 行为逐字节保持，`test_ut1504_shape_mismatch_falls_back_eager`
  面零扰动；AR016-020 基准脚本可复现性面不受扰）。开启时 runner 池按画布形状
  （tokens shape + 掩码/pe 对象组）键控——同形状窗口首个合格调用捕获、后续重放；
  画布增长 → 新形状键 → 新 runner 捕获；池容量上限可配（LRU 驱逐，VRAM 守恒），
  驱逐后同形状重现再捕获。串行/批量两池独立（AR016 矩阵语义保持）。图前置条件
  契约（#37）本身零放宽：同形、mask/pe `is` 钉定、精确型 SimilarityGate cosine、
  无 scheduler——池只是让"每形状一次捕获"在增长的画布上成立。
- **AR 步前向散度策略（显式定案面，P5 同式风险）**：块推进的 AR 步前向
  （`fast_dllm_sampler.py:230-243`）与最后去噪迭代同画布形状、链父=末迭代、最后
  块位置部分翻转——**恰为 P5 场景且在"当前活跃小块"字面之外**。FR2 意向定案：
  AR 步前向对新提交块区**强制散度**（与活跃小块同式构造级绕开；设计期定案传递
  形态并 UT 钉死），不依赖 gate 对该区的判断。
- **parity（硬性，由构造保证）**：合成模型上 folded-serial 生成 token 序列与 raw
  基线 **bit-exact**（前缀块精确复用 + 活跃块强制全量重算 ⇒ 无近似面；合成面构造
  按 FR1 语义匹配义务）；真实 checkpoint 上同式 parity 为基准 H1 判定面（若真实面
  出现非 bit-exact，如假删定案，如实 refuted——不预设）。
- **无 folded_model 时逐字节不变**：默认路径（含 eval/既有测试）零扰动。

**异常处理**：图前置失配按 #37 既有静默回退（单形状窗口内）；池捕获失败按既有
capture-failed 语义（家族 flag，不重试）；MFF 异常原样上抛。

**验收标准**：
- Given 合成模型（raw 语义与折叠掩码自洽），When 同 prompt 分别 raw/folded-serial
  生成，Then token 序列 bit-exact 且前缀复用探针>0、活跃块重算探针=活跃块 token 数
- Given 合成模型 + 强制散度标记 None（默认），When MFF 既有前向运行，Then 行为与
  AR020 末态 bit-exact（零扰动面）
- Given 合成模型 + 池开关**关**（默认），When 形状失配的图路径前向运行，Then 既有
  warn-once-never-recapture 行为逐字节保持（UT-1504 面零扰动）
- Given 合成模型 + 跨 ≥2 个块增长窗口，When folded-serial 生成（池开），Then 每个
  形状窗口各捕获一个 runner（池大小≥2 探针）且同窗口内重放（重放计数>0、零
  UserWarning）
- Given 池容量上限=1，When 画布跨形状增长，Then LRU 驱逐旧行 runner、新形状捕获
  成功且 VRAM 探针不增长（守恒面）
- Given 合成模型 + AR 步前向，When 块推进发生，Then 新提交块区重算探针=块 token
  数（强制散度覆盖，P5 同式绕开 UT）

### 3.3 FR3 批量投机块验证（opt-in）

**描述**：`FastDLLMSamplerConfig.num_block_candidates`（默认 0=关闭）：>0 时当前小块
以"N 候选一次批量验证"替代逐迭代 remasking。

**触发条件**：`num_block_candidates=N>0` 且 folded_model 为 MFF（批量面需要
AR015 `branch_ids` 契约；无 MFF 时配置报错，不静默降级）。

**期望行为**：
- **起草**：父前向（画布现状态）logits 在掩码位 top-k 采样（种子可控）构造 N 个
  候选小块填充（AR019 `_make_workload` 同配方：per-position top-k、种子化）。
- **批量验证**：N 候选堆叠 `[N,T]` 单次 MFF 批量调用（`branch_ids`、共享
  `parent_branch_id`=当前画布分支、钉定 mask/pe；候选差异全部落在活跃小块——
  强制散度区面复用 FR2）——AR016 批量图（FR2 形状键控池的批量侧）+ AR017 自适应
  容量 + AR018/020 发布路径首次被生产回路行使。
- **选定提交**：单一定案准则（设计期冻结：候选各掩码位 logits 置信度聚合，取最优
  候选整体提交）提交后跳过该小块剩余 remasking 迭代；后续 AR 步与块推进照旧。
- **结束语义**：提交的候选块内含停止 token 时按既有逐迭代规则判
  `finished`（`fast_dllm_sampler.py:219` 同式）；`finished.all()` 后续块跳过、
  停止位后 pad 填充行为保持。
- **默认关闭逐字节不变**：`num_block_candidates=0`（默认）时 sampler 全行为与既有
  bit-exact（既有 UT 面零扰动）。
- **质量证据义务**：真实 checkpoint 上提交块与默认配方输出的 exact-match 率 +
  置信度分布入工件（质量-速度权衡如实报，不预设 quality-parity）。

**异常处理**：批量调用异常原样上抛；N 超批量图容量按 AR016/AR017 既有矩阵回退
（eager 批量），不新增静默路径。

**验收标准**：
- Given 合成模型 + N=4，When 批量模式生成，Then 每小块恰好 1 次 `[N,T]` MFF 调用 +
  1 次批量发布（spy/账本断言），选定候选 ∈ 起草集合
- Given 合成模型 + N=0（默认），When 生成，Then 输出与既有 sampler bit-exact
  （回归面）
- Given 候选块含停止 token，When 提交，Then `finished` 语义与逐迭代规则一致（停止
  /pad 填充 UT）
- Given 真实 checkpoint，When N>0 运行，Then 工件记录 exact-match 率与墙钟
  （基准面，H3 判定）

### 3.4 FR4 真实 generate() 级基准脚本 + 工件

**描述**：真实 checkpoint 上三模式 `generate()` 墙钟与质量证据，预承诺单次定案。

**触发条件**：`python -m scripts.ar021_generation_loop_bench`（AR019 venv + 本地权重）。

**期望行为**：
- 三模式（同一 prompt 集与种子）：**baseline**（raw sampler）/ **folded-serial**
  （FR2 链折叠+串行图形状键控池）/ **folded-batched**（FR3，N 取 AR019 工件矩阵内
  值）；**warmup 覆盖每形状窗口捕获与自适应过渡期**（首 rep 捕获成本不入 p50，
  AR019 FR2 先例）+ **逐 rep 三模式 round-robin 交错**（WDDM 方差公平均衡）；
  逐块墙钟 + 整代墙钟（宿主 perf_counter+sync 夹逼，#39 注记 CUPTI null）；
  每模式输出 token 序列、稳定比、图捕获/重放计数（按形状窗分组）、发布账本
  （AR020 write_op_count）、no-fallback canary（AR019/20 同款）。
- **质量面**：folded-serial vs baseline 的 argmax parity（合成面已由构造保证；真实
  面为 H1 仲裁）；folded-batched vs baseline 的块级 exact-match 率 + 置信度
  （H3 质量分面）。
- **工件 schema 假设-判定分离**（判定式设计期冻结）：

| 假设 | 内容 | 判定（σ = max(σ_a, σ_b)，a/b 为参与对比的两模式；三态映射沿用 AR020 先例） |
|------|------|------|
| H1 | folded-serial 生成与 baseline 等价（合成 UT 面 bit-exact；真实生成面按 AR019 H2 口径仲裁：argmax 逐位一致 + bf16 allclose，bit-parity 作为报告子字段） | supported ⇔ 合成面全 bit-exact **且** 真实面 argmax 逐位一致 + allclose；否则 refuted（无 inconclusive） |
| H2 | folded-serial 整代墙钟 < baseline（前缀复用只重算活跃小块） | σ=max(σ_base, σ_serial)；supported ⇔ p50(base) − p50(serial) > 2σ；refuted ⇔ 反向 > 2σ；inconclusive ⇔ 其余 |
| H3 | folded-batched 整代墙钟 < folded-serial（批量验证摊薄）**且** exact-match 率如实报告（质量-速度权衡，不设质量门槛、不预设） | 速度分面：σ=max(σ_serial, σ_batched)，三态同 H2；质量分面纯报告（exact-match 率/置信度分布），不做 supported/refuted 判定 |
| H4 | 全栈生产行使：每形状窗串行图捕获>0 且重放>0（folded-serial）、批量图捕获>0（folded-batched）、发布账本 ≤ c·(L+N)（**c 设计期冻结入工件 meta**，AR020 同式）、no-fallback canary 全真 | supported ⇔ 所列计数器全部符合声明且 canary 全真；**自适应容量事件仅在其触发条件实际达成时核验**（未触发=中性，不判 refuted）；否则 refuted |

- **预承诺单次纪律**（#39 延续）：reps=30 单次运行定案，不重跑钓鱼。

**异常处理**：基准失败（捕获失败/显存不足）→ 工件记录失败配置与原因，不伪造数字。

**验收标准**：
- Given 全量运行（reps=30，warmup 覆盖每形状捕获），When 三模式 round-robin 交错
  执行，Then 完成时长 ≤ 60min；`--quick` 冒烟 ≤ 5min 且产出合法工件（H 判定允许
  quick 期 inconclusive）
- Given 运行完成，Then 工件字段完备：H1-H4 判定 + 三模式逐模式计数器（捕获/重放/
  账本/稳定比/canary）+ meta（判定式全文冻结、σ 口径与模式绑定、c 常数、reps、
  seed、venv、#39 注记、强制散度与形状键控池声明）
- Given 工件定案，Then 单次定案纪律注记在工件 meta（无重跑钓鱼）

### 3.5 FR5 文档收口

**描述**：五件套：`docs/OPTIMIZATION_GUIDE.md`（第二十八部分）、`CHANGELOG.md`、
`AGENTS.md`（新 pitfall 条目：生产回路折叠接线契约——强制散度活跃块/形状键控池/
sampler 前置条件）、`docs/RERUN_CHECKLIST.md`（复现行）、`README.md` **全量中文
重写**（AGENTS.md 既定规则：创新表新行 + 头条数字 + 质量基线 + 路线图重写；数字
全部从工件重算）。

**触发条件**：T005 质量门通过后。

**验收标准**：
- Given T004 工件定案，Then 五件齐备且 README 头条数字逐项可溯源至工件 JSON
  （抽查核验）；AGENTS 新条目与实现代码逐条一致

## 4. 非功能需求（质量基线）

| 类型 | 指标 | 要求 |
|------|------|------|
| 回归 | 全量 pytest（`-m "not slow"`） | 920 passed / 3 skipped + 新增 UT 数，零既有失败 |
| 静态 | mypy / pyflakes / 100 列 | clean / 零 / 新增行 ≤100 列 |
| 语义 | demo 三指标 | 85.5% / 2.35e-03 / 93.75% 锚点不变 |
| 语义 | 默认 sampler/MFF 行为 | 无 folded_model + N=0 + 强制散度 None + **池开关关（默认）** 时逐字节不变（既有 sampler/eval/MFF UT 零改动通过，含 UT-1504 面） |
| 资源 | 16GB 显存 | 基准峰值留余量（AR019/20 峰值 8.9/10.0GB 先例；[N,T] 批量缓冲与形状键控池的多 runner 静态缓冲计入，池上限可配） |

## 5. 约束与假设

**约束：**
- torch 2.5.1+cu121 / WDDM / CUPTI null / 无 C 编译器（#39 沿用）
- AR019 venv（transformers==4.53.1）+ Fast-dLLM-v2-1.5B 本地权重（#24 版本纪律）
- 块解码常数：block_size=32/small_block_size=32/threshold=0.95（官方配方对齐，#21）
- MFF 图前置条件契约不可放宽（#37：同形、mask/pe `is` 钉定、精确型 SimilarityGate
  cosine、无 scheduler）——形状键控池不放宽前置条件，只让"每形状一次捕获"成立

**假设（含风险定级，H 面为仲裁）：**
- **A1（最大机理风险，已由构造级设计绕开）**：同块未翻位置被 gate 误判 stable
  （AR019 P5 实证，parity 92.9% 先例）⇒ FR2 强制散度活跃块区使其不进入 gate 判定
  面；前缀块 gate 精确性有 P3/P4 证据。残余风险=强制散度传递面实现缺陷（UT 钉死）
- A2：小块迭代窗口内画布形状恒定 ⇒ 对象钉定+形状键控池后图可捕获重放（H4 仲裁）
- A3：[N,T] 候选批量前向的掩码位 logits 可支撑置信度选定准则（AR019 工件
  argmax parity=1.0 的同构证据；准则设计期单一定案冻结）
- A4：形状键控池的多 runner 静态缓冲 VRAM 可控（池上限 LRU；H4/工件 VRAM 面核验）
- A5：**批量池捕获摊销风险（设计期注记）**：生产整代生成中每画布形状的 `[N,T]`
  批量 runner 仅服务一次验证调用——若 warmup 不预捕获，摊销为负。设计期必须定案
  预捕获策略（每形状窗 warmup 捕获）或如实退回 eager 批量（H3 速度分面不受保真
  影响）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 生产生成回路 | `FastDLLMSampler.sample` 块解码循环（官方 fast-dLLM v2 配方对齐，#21） |
| remasking 链折叠 | 逐迭代以父=上一迭代的串行折叠（#26 链），前缀复用+活跃小块强制散度全量重算 |
| 强制散度活跃块区 | 折叠路径下当前活跃小块整区在 gate 判定前强制判散度的显式标记（FR2 新语义，P5 风险构造级绕开） |
| 形状键控 runner 池 | 按画布形状（tokens shape+mask/pe 对象组）键控的 graph runner 池：每形状一次捕获、跨块增长持续生效（FR2 新语义） |
| 批量投机块验证 | N 个候选小块填充一次 `[N,T]` MFF 批量调用验证后择优提交（AR019 工作负载的生产化） |
| 置信度准则 | 候选各掩码位 logits 的置信度聚合（最小值/均值，设计期冻结单一定案） |
| 三模式 | baseline（raw）/ folded-serial（FR2）/ folded-batched（FR3） |
