# [AR022] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR022 |
| AR 主题 | 批量投机质量对齐（贪心锚定候选机制）：folded-batched 批量投机模式的提交区质量面缺口修复——候选集新增贪心假设锚定官方 confidence-greedy 语义，四模式真实 generate() 级基准（质量硬门 + 速度不回退） |
| 关联 SR | AR021 srs §2 Out-of-Scope "批量投机块的接受策略族谱（本 AR 单一准则定案，族谱留后续）"+ GUIDE 第二十八部分遗留清单同项；AR021 工件 H3 质量分面缺口（exact-match 0.0 / 置信度落差 / 解码长度失控——§1.1 全引） |
| 日期 | 2026-10-15 |
| 状态 | rev 2（req 门控第 1 轮 PASS + Minor-1/Nit-1~3 当轮修入） |

## 1. 背景与目标

用户总纲（已定案，本 AR 最高依据）："最终看的是综合框架比 baseline 要强，创新框架的
生产要有深度依据"——综合框架须在**速度与质量两面同时**强于 baseline。

AR021 把折叠栈接入了真实生产生成回路并交付三模式基准：速度面 folded-batched 达成
**2.38×**（`speedups_p50.batched_vs_baseline` = 2.3795，p50 1193.9ms vs baseline
2840.7ms，`mode_comparison.*.whole_gen_ms.p50`）；等价面 folded-serial 与 baseline
**bit-exact**（`hypotheses_verdicts.H1.verdict` = supported）。**但质量面留下最大
短板**：folded-batched 批量投机模式的提交区构造性偏离 baseline 语义——这是"综合框架
vs baseline"叙事中唯一未闭合的面，本 AR 选题即此。

### 1.1 AR021 质量分面缺口（工件数字全引，`results/optimization/ar021_generation_loop_bench.json`）

| 缺口面 | AR021 工件数字（字段路径） | baseline 对照 | 机理后果 |
|--------|--------------------------|---------------|----------|
| 块级 exact-match | 两 prompt 均 **0.0**（`quality.h3_batched_vs_baseline[0].block_exact_match.overall_exact_match` = 0.0；`[1]` 同） | 参照系（34/66 token 轨迹） | 投机提交与 baseline 逐位一致率为零——质量面无任何对齐证据 |
| posthoc 置信度 | prompt0：**0.289692**（`[0].posthoc_confidence.folded-batched.overall_mean`，`overall_min` 5e-06） | **0.369661**（`[0].posthoc_confidence.baseline.overall_mean`，min 0.150954） | 低置信 token 被一次性提交（min 5e-06 vs baseline 0.15） |
| posthoc 置信度 | prompt1：**0.106418**（`[1].posthoc_confidence.folded-batched.overall_mean`，`overall_min` 0.0） | **0.570341**（`[1].posthoc_confidence.baseline.overall_mean`，min 0.275356） | 同上，落差 0.46 |
| 解码长度（早停） | prompt0 **160**（`[0].block_exact_match.len_other`——全画布 32+128，全程不早停）；prompt1 **77**（`[1].block_exact_match.len_other`——伪位置早停） | **34**（`[0].block_exact_match.len_baseline`）/ **66**（`[1].block_exact_match.len_baseline`） | prompt0 走满全程多做无用功（4 块 vs baseline 2 块——速度分面也受损）；prompt1 在错误位置早停 |

**根因（代码事实）**：AR021 FR3 的候选 = **种子化 top-k 采样**——
`fast_dllm_sampler.py:346-360`（`_speculative_small_block`）：每个候选 j 以冻结种子
公式（`:354-356`）从 draft logits 的 top-k（`candidate_top_k=10`）中**随机**抽取仍
掩码位填充。该构造与 baseline 的提交语义结构性偏离：

- baseline（`fast_dllm_sampler.py:544-563`，官方 fast-dLLM v2 配方对齐 #21）是
  **confidence-greedy**：temperature=0.0（工件 `meta.block_constants.temperature`）
  下 `sample_tokens` 即逐位 argmax，迭代式提交（prob > threshold=0.95 位 + 全局
  argmax 位，`meta.block_constants.threshold`）。NVlabs Fast-dLLM v2 官方
  `generation_functions.py` 同为置信度贪心（模块 docstring `fast_dllm_sampler.py:1-13`
  配方描述、`:33-34` NVlabs URL 引用）。
- 种子化 top-k 候选不含 stop token 的命中面：stop token 常不在 draft top-10 内或
  随机选择不命中 → prompt0 全程无早停（len 160）、prompt1 伪位早停（77 ≠ 66）。
- min-prob 选定（`:373-379`，AR021 冻结准则）只能在**已构造性偏离的候选集**内择优
  ——选定准则无错，错在候选集缺官方语义锚定。

### 1.2 机制草案与深度依据（本 AR 核心论证）

**贪心锚定候选**：候选集新增 candidate 0 = **贪心假设**——仍掩码位全填 draft logits
argmax（确定性、无种子）；经 `[N,T]` 批量验证前向后以**真实后验 logits** 参与
min-prob 选定竞争。

- **官方语义对齐依据**（#21 义务）：baseline 即 confidence-greedy（§1.1）；贪心假设
  是官方语义在投机路径上的**锚定表达**——不是新解码配方，是把"baseline 会怎么填"
  作为候选之一送入批量验证。
- **构造级对齐范围（精确陈述，不夸大）**：贪心候选胜出 ⇒ 提交区 = 贪心假设（构造
  级、确定可复现）。该假设与 baseline 的逐位一致范围：**对齐归纳前提成立时**
  （对齐轨迹画布与 baseline 在该小块起点逐位一致），draft 前向 = baseline 迭代 1
  前向（AR021 H1 bit-exact 证据链：folded-serial 全程 bit-exact ⇒ 同链前向同
  logits），则贪心假设在 baseline **迭代 1 提交位**（prob > 0.95 位 + 全局 argmax
  位）构造性逐位一致；**低置信位**（baseline 于后续迭代提交）为经验重合面，若任一
  位偏离则归纳链断裂、轨迹分叉——**全轨迹 parity 不预设**，H1 仲裁。
- **非平凡性（不退化投机多样性）**：批量验证后提交区置信度可变——贪心假设以真实
  后验 logits 参与竞争，**可能输给其他候选**（须 UT 构造落败场景钉死，FR2）。
- **早停恢复（效果非新机制）**：贪心假设含 stop token（stop 位 argmax = stop 时）
  且胜出 ⇒ `finished` 语义自然对齐 baseline（`fast_dllm_sampler.py:381` 同式，链
  延续/停止语义零改动）——额外速度收益（prompt0 类场景少做 2 块无用功）。
- **速度面由构造保持**：贪心候选行仍是 `[N,T]` 批量行（#37 契约），N/top-k/批量形状
  全不变 ⇒ 批量池/AR017/账本/canary 全部 AR021 契约面零改动。

### 1.3 口径延续

- 基准模型/负载/纪律全沿 AR021：Fast-dLLM-v2-1.5B 本地权重 + AR019 venv
  （transformers==4.53.1，#24），宿主 `perf_counter`+sync（#39 CUPTI null 注记），
  seed 2026、P=2 prompt（padded 32/64）、num_tokens=128（W=4 形状窗）、warmup 覆盖
  每形状窗捕获、逐 rep round-robin 交错、reps=30 预承诺单次定案。
- 四模式：**baseline**（raw）/ **folded-serial**（AR021 FR2）/ **folded-batched**
  （AR021 FR3，N=4）/ **folded-batched-greedy（对齐模式，本 AR 新增）** = 
  folded-batched 全部配置 + `candidate_include_greedy=True`。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- FR1 贪心锚定候选：`candidate_include_greedy=True` 时候选集 0 号位 = 贪心假设
  （argmax 填充、确定性无种子、不改其余候选构建）；N 语义定案（§3.1）
- FR2 选定与质量观测：min-prob 选定规则不变（贪心公平竞争、可能落败）；新增
  greedy_win 观测面与双粒度 exact-match 观测出口；链延续/停止语义不变
- FR3 零扰动与配置：`candidate_include_greedy: bool = False` 默认——与 AR021 末态
  逐字节；True 前置校验（循 UT-2110 构造期先例）；YAML/config 载入
- FR4 四模式基准与工件：真实 checkpoint 四模式（含对齐模式）墙钟 + 质量硬门 H1 +
  速度不回退 H2 + 早停恢复 H3 + 全栈行使 H4（判定式冻结，reps=30 单次定案）
- FR5 文档收口五件（含 README 全量中文重写，AGENTS 既定规则）

**Out of Scope（本 AR 不做的）：**

- 串行模式墙钟提升（AR021 H2 inconclusive：Δ=−107.5181ms 未越 2σ=265.3231ms
  （`hypotheses_verdicts.H2.numbers`）——单窗小形状折叠开销>收益，另立 AR）
- ctx LRU-池条目生命周期（AR021 design §7 Nit-1 注记：ctx 驱逐后同窗重现重建池
  条目，VRAM 浪费非正确性）
- 生产单发摊销（AR021 A5 负注记原样延续：单次 generate 每形状窗恰 1 次 `[N,T]`
  调用，捕获摊销为负——本 AR 不解决，工件 meta 注记延续）
- 质量校准/温度/threshold/top-k 调参族谱（配方常数冻结 #21：block 32 / small 32 /
  threshold 0.95 / temperature 0.0，工件 `meta.block_constants`）
- 候选集扩展族谱（贪心锚定为单一定案；其他锚定/接受策略留后续 AR）
- LLaDA/Dream sampler 投机接线（remote 代码各异）；eval 适配器 / LM-eval 任务级
  质量跑分（后续独立 AR）；LLaDA-8B/Dream-7B（16GB 不可行）

**FR 间耦合依据（AR021 srs:80-83 先例式）**：FR1/FR2 是同一"候选-选定"机制的两分
（FR1 管候选集构建、FR2 管选定与观测——拆分以便独立验收：构建确定性/零回读面 vs
选定公平竞争/非退化面）；FR3 是 FR1/FR2 的默认零扰动外壳（新旋钮不开启则两机制
均不触达，943/3 基线由 FR3 守护）；FR4 是 FR1-FR3 的生产级仲裁（四模式构造直接
复用 AR021 基准——对齐模式 = folded-batched 全配置 + 单旋钮差，同 `[N,T]` 形状
A/B 的可比性由 FR1 的 N 语义定案 (i) 支撑）；FR5 为交付收口（README 全量重写义务，
AGENTS 既定规则）。

## 3. 功能需求

### 3.1 FR1 贪心锚定候选

**描述**：`candidate_include_greedy=True` 时，`_speculative_small_block` 的候选集
0 号位为贪心假设（仍掩码位全填 draft logits argmax），其余候选构建不变。

**触发条件**：`num_block_candidates=N>0` 且 `candidate_include_greedy=True` 且
folded_model 支持 `branch_ids`（既有前置原样）。

**期望行为**：
- **贪心假设构建**：候选行 0 的每个仍掩码位 = draft logits（起草前向 = baseline
  迭代 1 前向，§1.2）该位 argmax（即 top-k 有序集首位——topk 有序性保证）；不使用
  generator/种子，跨运行逐字节可复现；**确定性构建、零设备回读**（无 `.item()`/
  `.tolist()`/`.cpu()`）、无 generator 消费。若 draft argmax 位为 mask token，行为与 baseline 逐位 argmax
  家族一致（无特殊处理——baseline `sample_tokens` temperature=0 同样可提交 argmax，
  语义自洽）。
- **其余候选不变**：行 j≥1 沿用 AR021 冻结种子公式
  `(S0·1_000_003 + block_idx·8_191 + small_block_idx·131 + j) & 0x7FFF_FFFF`
  （`fast_dllm_sampler.py:354-356`）原样，**j 保持行号**（j=1..N−1）。
- **N 语义定案（二选一冻结）**：**候选总数 = 配置 N 不变；贪心占据 0 号位；随机
  候选数 = N−1。** 理由：
  (i) `num_block_candidates` 语义保持"批量验证行数"⇒ `[N,T]` 形状不变 ⇒ 批量池键
  `(N,T,prompt_len)`、AR017 容量 `⌈ratio·N·T⌉`、VRAM 估算与 AR021 folded-batched
  在 N=4 完全一致——对齐模式与 folded-batched 同形状 A/B，**速度面差异被构造性
  归零，质量变量被隔离**（FR4 可比性的根基）；
  (ii) 附加为 N+1 行会静默放大批量算力/显存（配置 N=4 实际执行 `[5,T]`），违背配置
  语义最小惊讶，且破坏 (i) 的同形状对比；
  (iii) 与 AR021 冻结面兼容：`branch_ids` 命名 `[f"{p}_cand{j}"]`（行号语义）、链
  延续、发布路径全部沿用——贪心只是行 0 的**内容**变化，非结构变化。
  备选（附加为 N+1 行）因 (i)(ii) 否决。
  **UT-2109 兼容性分析**：默认 `candidate_include_greedy=False`（FR3）时候选集、
  种子公式、输出逐字节不变 ⇒ UT-2108/2109/2109c/2110 原断言零扰动；开启时行 0 =
  argmax、行 1..N−1 与关闭态行 1..N−1 逐位相同（种子公式 j 保持行号 ⇒ 唯一差异
  = 行 0，A/B 可观测性由构造成立），新 UT（UT-2202 系列）钉死。
- **N=1 退化（合法）**：greedy on + N=1 ⇒ 单行 = 纯贪心假设（`[1,T]` 批量调用），
  选定平凡——纯贪心锚定模式，显式合法非报错。

**异常处理**：贪心构建不新增异常路径（argmax 为纯设备算子）；既有构造期校验
（N<0 / top_k<1 / 无 folded_model / 不支持 branch_ids）原样。

**验收标准**：
- Given 合成模型 + greedy on + N=4，When 批量投机运行，Then 批量调用形状 `[4,T]`
  不变、行 0 = 仍掩码位 argmax 填充（确定性、跨运行 bit-exact、无 generator 消费）、
  行 1..3 与 greedy off 同 seed 运行的行 1..3 逐位相同
- Given 合成模型 + greedy on，When 候选构建，Then 零设备回读（spy 断言无
  `.item`/`.tolist`/`.cpu` 调用）
- Given greedy on + N=1，When 运行，Then `[1,T]` 单行纯贪心合法且链延续正确
- Given greedy off（默认），When 任意运行，Then 与 AR021 末态逐字节（FR3 验收
  交叉引用）

### 3.2 FR2 选定与质量观测

**描述**：min-prob 选定规则不变（AR021 冻结准则原样），贪心候选以批量验证后真实
logits 公平竞争；新增 greedy_win 观测面与双粒度 exact-match 观测出口。

**触发条件**：同 FR1（greedy on 的投机路径）；观测面在 greedy on 时激活。

**期望行为**：
- **选定规则不变**：候选得分 = 提交位（仍掩码位）token 概率（批量 logits
  right-shift 后切片、softmax、gather）之最小值，取最大候选（
  `fast_dllm_sampler.py:373-379` 冻结式原样）；贪心候选无豁免/加权——**公平竞争，
  可能落败**（投机多样性不退化：落败场景须 UT 构造钉死，防止实现退化为恒选贪心）。
- **胜出语义**：贪心候选胜出 ⇒ 提交区 = 贪心假设（构造级）；其与 baseline 的对齐
  范围按 §1.2 精确陈述（迭代 1 提交位构造性一致 / 低置信位经验面——H1 仲裁，
  不预设全轨迹 parity）。
- **新增观测面（供基准与 UT 探针）**：采样器暴露 per-small-block **greedy_win**
  判定（胜者是否贪心行）与累计计数（greedy 胜出数 / 投机调用数；接口形态设计期
  定案，如实例级只读探针）；基准脚本据此计算 greedy_win_rate 入工件。
- **exact-match 观测出口（基准侧）**：对齐模式 vs baseline 的块级 exact-match
  （AR021 既有口径）**扩展位置级一致率**（提交位 token 逐位比对比例——双粒度，
  H1(a) 主门的测量面）。
- **链延续/停止语义不变**：胜出分支 id 切换 `_folding_parent`
  （`fast_dllm_sampler.py:382` 同式，含贪心胜出分支）；stop ∈ 提交区 ⇒ `finished`
  同基线式（`:381`）——**早停恢复是效果非新机制**（贪心假设含 stop 且胜出时自然
  对齐 baseline 停止位），后续块跳过/pad 填充行为保持。

**异常处理**：无新增异常路径（选定/链/停止全部既有面）。

**验收标准**：
- Given 构造已知后验概率的 fixture（贪心行 min-prob 低于某随机候选），When 选定，
  Then 非贪心候选胜出（落败场景实证，防退化）
- Given 贪心候选胜出的场景，When 提交，Then 提交区 == 贪心假设逐位且 greedy_win
  判定为真、计数 +1
- Given 贪心假设含 stop token 且胜出，When 提交，Then `finished`/pad 填充与基线
  同式（停止语义 UT 面）且 `_folding_parent` = 胜出候选分支 id（贪心/随机胜出两
  分支均断言）
- Given greedy on 运行完成，When 读取观测面，Then greedy_win 计数与批量调用次数
  自洽（探针断言）

### 3.3 FR3 零扰动与配置

**描述**：`FastDLLMSamplerConfig.candidate_include_greedy: bool = False`（默认）——
默认与 AR021 末态逐字节；True 的前置校验与载入面。

**触发条件**：配置载入 / 构造期校验。

**期望行为**：
- **默认零扰动（AR021 契约 #43 延续）**：`candidate_include_greedy=False`（默认）
  时 sampler 全行为与 AR021 末态 bit-exact——不传新参全既有面逐字节不变（含
  raw 路径 / N=0 / 投机模式 N=4 greedy-off 三面；UT-2108/2109/2109c/2110 冻结面
  零改动通过）。
- **True 前置（构造期校验，循 UT-2110 `__post_init__` 先例）**：
  `candidate_include_greedy=True` 且 `num_block_candidates == 0` → 构造期
  `ValueError`（贪心锚定仅定义于投机路径，不静默忽略）；`N>0` 的既有前置
  （folded_model / branch_ids 支持，`sample()` 入口）原样叠加生效。
- **config 载入**：新字段入 `FastDLLMSamplerConfig` dataclass（AGENTS #5：新字段
  须入 dataclass 方可经 `load_config` 程序化消费，YAML 键未知时仅 UserWarning 过滤
  ——本 AR 字段可消费性 UT 钉死）。

**异常处理**：`True + N=0` 构造期 `ValueError`，不静默降级（UT-2110 风格）。

**验收标准**：
- Given 默认配置（不传新参），When 全部既有面运行，Then 与 AR021 末态 bit-exact
  （UT-2108/2109/2109c/2110 零改动通过 + 新增默认零扰动 UT 双面钉死）
- Given `candidate_include_greedy=True, num_block_candidates=0`，When 构造
  sampler config，Then `ValueError`（不静默）
- Given YAML 含 `candidate_include_greedy` 键，When `load_config` 载入，Then 字段
  被程序化消费（无未知键告警路径）

### 3.4 FR4 四模式基准与工件

**描述**：真实 checkpoint 四模式 `generate()` 墙钟与质量证据，质量硬门 + 速度不
回退，预承诺单次定案。

**触发条件**：`python -m scripts.ar022_speculative_quality_bench`（AR019 venv +
本地权重；新脚本复用 `scripts/ar021_generation_loop_bench` 助手与三模式构造——
AR021 复用 ar019 先例）。

**期望行为**：
- **四模式构造（冻结）**：`baseline` / `folded-serial` / `folded-batched`
  （三模式构造 = AR021 逐字沿用：MFF_A 串行池 / MFF_B 批量池+AR017，N=4、
  top_k=10、split_layers=False）+ **`folded-batched-greedy`（对齐模式）** =
  MFF_C（与 MFF_B 同构：use_cuda_graph=True、graph_runner_pool_size=8、
  graph_adaptive_capacity=True、split_layers=False，独立实例——池计数/canary 按
  模式隔离）+ sampler(N=4, candidate_top_k=10, **candidate_include_greedy=True**)。
  **同形状 A/B（构造可比性根基，FR1 定案 (i)）**：对齐模式与 folded-batched 共享
  同 N/同 top-k/同 prompt 集/同 seed/同 `[N,T]` 批量形状——速度面差异被构造性
  归零。
- **协议（AR021 全沿）**：seed 2026、P=2 prompt（padded 32/64）、num_tokens=128
  （W=4）；warmup 每模式每 prompt 完整一轮丢弃（每形状窗池预捕获 + Triton/JIT +
  AR017 过渡期，捕获成本不入 p50）；逐 rep **四模式** round-robin 交错（rep r 起
  始位 r%4）；宿主 perf_counter+sync（#39 CUPTI null 注记）；逐块 + 整代墙钟；
  canary 二分口径（降级类 timed=0 / 结构性 eager 单独计数）、账本 c=4 每发布事件、
  `graph_pool_stats` per-shape-window 出口全沿 AR021 冻结口径。
- **质量面（对齐模式 vs baseline，sidecar 同构对比）**：块级 exact-match +
  **位置级一致率（双粒度）** + posthoc 置信度分布 + 解码长度 + greedy_win_rate；
  folded-batched 质量面同口径重测（同 seed 应复现 AR021 先验：块级 0.0、长度
  160/77、置信度 0.289692/0.106418——复现性自检面）。
- **工件 schema**（`results/optimization/ar022_speculative_quality_bench.json`）：
  `hypotheses_declared`/`hypotheses_verdicts`（H1-H4 判定式全文 + 三态 + numbers）/
  `mode_comparison`（四模式墙钟 mean/std/p50 per block + 整代 + per prompt）/
  `per_shape_window`（三折叠模式池计数 + 两类 eager + AR017 事件，AR021 口径）/
  `quality`（对齐模式 vs baseline 双粒度 exact-match + posthoc + 解码长度 +
  greedy_win_rate；folded-batched vs baseline 同口径；folded-serial vs baseline
  argmax parity 面 AR021 沿用）/ `ledger`（c=4）/ `batch_canary`（三折叠模式）/
  `speedups_p50` / `meta`（reps=30、seeds、四模式配置全文含
  `candidate_include_greedy: true` 声明与贪心行 0 语义声明、种子公式延续声明、
  warmup/round-robin/单次定案纪律、A5 单发摊销负注记延续、CUPTI null、VRAM 峰值
  per 模式、驱动/venv 环境面）。
- **H 假设（判定式冻结，σ 纪律：σ = max(σ_a, σ_b) 绑参与对比的模式对，AR021
  先例；质量面为确定性单次测量，无 σ——σ 纪律仅墙钟面）**：

| 假设 | 内容 | 判定 |
|------|------|------|
| H1 质量对齐（硬门） | 对齐模式提交区质量对齐 baseline（vs folded-batched 显著改善 + 绝对门） | supported ⇔ 三面全过：(a) **位置级一致率**（对齐模式 vs baseline，**口径冻结**：逐 prompt 按画布绝对位置取 `min(len_aligned, len_baseline)` 范围内两模式均生成 token 的位集合（小块划分以对齐模式自身经历为准），逐位 token 相等比对，两 prompt 全部比对位**合并聚合**为单一均值）> **0.5**；(b) 块级 exact-match 聚合 > folded-batched 同口径实测值（AR021 先验 0.0——严格为正即过）；(c) 每 prompt posthoc overall_mean > folded-batched 同 prompt 实测值（先验 0.289692 / 0.106418）。任一不过 → refuted（确定性输出无 inconclusive）；greedy_win_rate 与 vs baseline 置信度差距为机理报告面 |
| H2 速度保持（硬门） | 对齐模式整代墙钟 vs folded-batched 不回退（同 `[N,T]` 形状 A/B） | σ=max(σ_aligned, σ_batched)；supported ⇔ p50(batched) − p50(aligned) ≥ −2σ（即 p50(aligned) − p50(batched) ≤ 2σ——含加速与持平）；refuted ⇔回退 > 2σ；无 inconclusive（方向性不回退门）。vs baseline 加速比为报告面（README 头条：早停恢复下预期 ≥ AR021 的 2.38×） |
| H3 早停恢复 | 对齐模式解码长度对齐 baseline（stop token 命中面恢复） | supported ⇔ 两 prompt 解码长度均 == baseline 同 prompt（先验 34 / 66）；否则 refuted；vs folded-batched（先验 160 / 77）的改善幅度为 numbers 报告子字段（残余改善如实记录） |
| H4 全栈行使不回退 | 对齐模式全栈契约面与 AR021 同口径不回退 | supported ⇔ (i) 对齐模式每个含批量调用的形状窗批量捕获>0 且 timed 重放>0；(ii) 降级类 canary timed=0（AR021 二分口径原样，三折叠模式全真）；(iii) 发布账本 ≤ 4·(L+N)（c=4 冻结，AR020/021 同式）；(iv) AR017 事件仅触发时核验（未触发=中性）。否则 refuted |

- **H1(a) 门 0.5 的量化依据（机理推导，如实陈述风险）**：构造级下界 = baseline
  迭代 1 提交位集合（对齐归纳前提成立时贪心假设在该集合逐位一致，§1.2）+ 低置信
  位经验重合；AR021 侧车证据（folded-serial 前向数 31/27 ⇒ 每小块 ~14 迭代）表明
  低置信位占多数、经验重合占比未知——0.5 是"过半提交位对齐"的最低可辩护门槛
  （用户总纲要求质量面对齐叙事可成立）；若实测 refuted，改善幅度如实入工件
  numbers（预承诺单次定案，不重跑钓鱼），作为接受策略族谱后续 AR 的选题依据。
  主门取**位置级** 0.5：严于"任意改善"、宽于块级 0.5（块级全对要求低置信位全部
  重合，无构造级支撑）。
- **预承诺单次纪律（#39 延续）**：reps=30 单次运行定案，不重跑钓鱼。

**异常处理**：基准失败（捕获失败/显存不足）→ 工件记录失败配置与原因，不伪造数字；
对齐模式与 folded-batched 质量复现面不符（同 seed 不同果）→ 如实记录为复现性
异常并阻断 H 判定（不静默取先验值）。

**验收标准**：
- Given 全量运行（reps=30，warmup 覆盖每形状捕获），When 四模式 round-robin 交错
  执行，Then 完成时长 ≤ 60min；`--quick` 冒烟（reps=3/单 prompt/2 块）≤ 5min 且
  产出合法工件（H 判定允许 quick 期非定案）
- Given 运行完成，Then 工件字段完备：H1-H4 判定 + 四模式逐模式计数器（池/账本/
  canary/质量双粒度/greedy_win_rate/解码长度）+ meta（判定式全文冻结、σ 口径绑
  模式、c=4、reps、seed、贪心锚定声明、A5 注记延续、#39 注记）
- Given 工件定案，Then 单次定案纪律注记在工件 meta（无重跑钓鱼）
- Given 四模式运行，Then VRAM 峰值 < 16GB 留余量（AR021 峰值 5374.5MB 先验 +
  第三 MFF 实例同 `[N,T]` 形状同量级；`--quick` 实测校准后入 meta）

### 3.5 FR5 文档收口

**描述**：五件套：`docs/OPTIMIZATION_GUIDE.md`（第二十九部分）、`CHANGELOG.md`、
`AGENTS.md`（新 pitfall 条目：贪心锚定候选契约——行 0 语义/N 语义定案/公平竞争
不退化/默认零扰动/质量观测面）、`docs/RERUN_CHECKLIST.md`（复现行）、`README.md`
**全量中文重写**（AGENTS 既定规则：创新表新行 + 头条数字（质量对齐 + 速度双面）+
质量基线 + 路线图重写；数字全部从工件重算）。

**触发条件**：T005 质量门通过后。

**验收标准**：
- Given T004 工件定案，Then 五件齐备且 README 头条数字逐项可溯源至工件 JSON
  （抽查核验）；AGENTS 新条目与实现代码逐条一致

## 4. 非功能需求（质量基线）

| 类型 | 指标 | 要求 |
|------|------|------|
| 回归 | 全量 pytest（`-m "not slow"`） | **943 passed / 3 skipped** + 新增 UT 数（AR021 末态基线），零既有失败 |
| 静态 | mypy / pyflakes / 100 列 | clean / 零 / 新增与改动文件 ≤100 列（black） |
| 语义 | demo 三指标 | 85.5% / 2.35e-03 / 93.75% 锚点不变 |
| 语义 | 默认 sampler 行为 | `candidate_include_greedy=False`（默认）时与 AR021 末态 bit-exact（含 UT-2108/2109/2109c/2110 冻结面零改动） |
| 资源 | 16GB 显存 | 四模式基准峰值留余量（AR021 峰值 5374.5MB 先验；三 MFF 实例池并存 + `[N,T]` 批量缓冲计入，quick 实测校准入 meta） |

## 5. 约束与假设

**约束：**
- AR021 冻结测试面零破坏：UT-2108/2109/2109c/2110 原断言不动（新机制默认关 ⇒
  逐字节）；`_speculative_small_block` 既有种子公式/min-prob 选定/`branch_ids`
  命名不改——贪心为纯增量行
- 贪心候选不得破坏批量图路径（#37 契约）：候选行仍是 `[N,T]` 批量行——同
  mask/pe/forced 钉定、同父 `parent_branch_id`、同发布路径（AR018 快照 +
  AR020 put_batch / `_store_batched`），不新增图前置分支/静默路径
- 官方配方对齐义务（#21）：baseline 配方常数不动（block_size=32 /
  small_block_size=32 / threshold=0.95 / temperature=0.0，工件
  `meta.block_constants`）；贪心锚定是官方 confidence-greedy 语义的投机路径锚定
  表达，非新解码配方
- torch 2.5.1+cu121 / WDDM / CUPTI null / 无 C 编译器（#39 沿用）；无 mock 数据
  （#10）
- AR019 venv（transformers==4.53.1）+ Fast-dLLM-v2-1.5B 本地权重（#24 版本纪律）

**假设（含风险定级，H 面为仲裁）：**
- **A1（对齐归纳论证，本 AR 核心机理）**：若对齐轨迹画布与 baseline 在某小块起点
  逐位一致，则该块 draft 前向 logits 与 baseline 迭代 1 逐位一致（AR021 H1
  bit-exact 构造：前缀复用精确 + 强制散度活跃块全量重算 ⇒ 同画布同 logits），
  贪心假设在 baseline 迭代 1 提交位（prob>0.95 + 全局 argmax 位）构造性一致。
  **残余风险**：低置信位（baseline 后续迭代提交）与一次性贪心填充为经验重合面，
  任一位偏离则归纳链断裂、后续块进入分叉轨迹——全轨迹 parity 不预设，H1/H3 仲裁
- A2：贪心候选胜出率理论上高（argmax 位概率 ≥ 同位任何 top-k 替代 ⇒ min-prob
  竞争倾向贪心），但非构造保证（后验 logits 随候选内容可变，贪心可落败——
  FR2 落败 UT 钉死）；greedy_win_rate 如实入工件（机理报告面）
- A3：早停恢复依赖 stop 位 draft argmax == stop 且贪心胜出（baseline 于迭代 1
  提交 stop 位时构造成立；否则经验面）——H3 仲裁；AR021 实证种子化候选 stop
  命中面缺失（prompt0 len 160 全程不早停 / prompt1 伪位早停 77 ≠ 66）
- A4：`[N,T]` 形状不变（FR1 定案）⇒ 批量池键/AR017 容量/账本/canary 全部 AR021
  契约面零改动；对齐模式 vs folded-batched 速度 A/B 构造可比（H2 判定有效性根基）
- A5（负注记延续）：生产单发摊销为负（AR021 A5 注记原样：单次 generate 每形状窗
  恰 1 次 `[N,T]` 批量调用，捕获 ≈2-3 次前向成本服务 1 次重放——单发建议
  `use_cuda_graph=False`；本 AR 不解决，工件 meta 注记延续）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 贪心锚定候选 | `candidate_include_greedy=True` 时候选集 0 号位 = 贪心假设的机制（本 AR 新语义） |
| 贪心假设 | candidate 0：仍掩码位全填 draft logits argmax 的确定性候选（无种子，确定性构建零设备回读） |
| 对齐模式 | `folded-batched-greedy`：批量投机 + 贪心锚定（`[N,T]` 形状与 folded-batched 相同） |
| greedy_win | per-small-block 判定：min-prob 选定胜者是否贪心行（0 号位） |
| min-prob 选定 | AR021 冻结准则：候选提交位 token 概率最小值最大化（`fast_dllm_sampler.py:373-379`） |
| 位置级一致率 | 提交位 token 与 baseline 对应位一致的比例（exact-match 双粒度之细粒度，H1(a) 主门测量面；比对窗口口径见 §3.4 H1(a) 冻结式：画布绝对位置、`min(len_aligned, len_baseline)` 范围、两 prompt 合并聚合） |
| 块级 exact-match | AR021 既有口径：小块全部提交位与 baseline 一致的比率（全对率） |
| 对齐归纳 | A1 论证：画布一致 ⇒ draft 前向一致 ⇒ 迭代 1 提交位构造一致；低置信位偏离即断裂 |
| 四模式 | baseline（raw）/ folded-serial（AR021 FR2）/ folded-batched（AR021 FR3）/ folded-batched-greedy（本 AR 对齐模式） |
