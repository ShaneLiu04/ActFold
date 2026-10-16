# [AR023] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR023 |
| AR 主题 | 贪心锚定投机解码跨族泛化（LLaDA-8B + Dream-v0-7B）：AR021 生成循环折叠 + AR022 贪心锚定候选完整生产链向 LLaDA/Dream 双族移植，四模式真实 checkpoint 基准对齐 AR022 口径（质量分级门 rev 3 + 加速比>1） |
| 关联 SR | AR022 srs §2 Out-of-Scope 遗留项："LLaDA/Dream sampler 投机接线（remote 代码各异）；LLaDA-8B/Dream-7B（16GB 不可行）"——本 AR 以量化加载路径重启该面；AR021 GUIDE 第二十八部分遗留清单"变长/多 parent 批量"同源谱系 |
| 日期 | 2026-10-15 |
| 状态 | rev 3（req 门控第 1 轮 PASS + Minor G2/G8/G3 三项当轮修入；design 阶段 OQ1 定案：质量门改分级口径——AR022 gap==0 系经验结果非构造保证，跨族严格硬门存在正确实现仍 FAIL 的风险） |

## 1. 背景与目标

用户总纲（已定案，全 AR 系列最高依据）："最终看的是综合框架比 baseline 要强，创新
框架的生产要有深度依据"——综合框架须在**速度与质量两面同时**强于 baseline。

AR022 末态（`results/optimization/ar022_speculative_quality_bench.json`，reps=30）：
贪心锚定对齐模式恢复 baseline 质量（posthoc gap 0.0、长度 ==、块级 exact 4/4）同时
p50 **7.0236×** vs baseline / 2.9498× vs batched——速度质量双面闭合。**但整条生产链
（AR021 生成循环折叠 + 投机块候选 + AR022 贪心锚定）仅存在于
`FastDLLMSampler`**（`actfold/models/fast_dllm_sampler.py`，AGENTS #43/#44）。

### 1.1 泛化缺口（代码事实）

- `LLaDASampler`（`actfold/models/llada_sampler.py`，296 行）与 `DreamSampler`
  （`actfold/models/dream_sampler.py`，246 行）是**参照实现**：块解码/全画布迭代去噪
  面向官方配方对齐（AGENTS #21），folding 仅有老式 cross-step 链（`_next_folding_branch`
  ，#26）——**无投机块候选、无 `[N,T]` 批量分支验证（#41）、无贪心锚定、无
  `greedy_win_stats`、无生成循环折叠 ctx**。
- 机制核心（贪心行 = draft argmax 冻结定义、min-prob 公平竞争、冻结种子公式、
  `branch_ids` 批量验证、胜者链延续）理论上**模型族无关**；但三族画布/规则差异需要
  适配层：
  - LLaDA：右填充画布、块解码（block_size 切分）、`low_confidence` remasking、
    long attention_mask；
  - Dream：左填充画布、全画布迭代、`maskgit_plus`/`topk_margin`/`entropy` 置信
    规则、bool attention_mask + `position_ids`（AR019 位置语义先例）。
- 泛化验证的深度依据：若机制仅在 Fast-dLLM remote code 路径成立，"综合框架强于
  baseline"的叙事就绑定单一 checkpoint；跨族复现 AR022 口径（同 checkpoint 内 A/B
  对齐 gap=0 + 加速比>1）是机制普遍性的直接证据。

### 1.2 环境约束与探针先决（本 AR 特有）

- **VRAM**：Quadro RTX 5000 16GB（实测空闲 15.3GB）。LLaDA-8B bf16 ≈16.1GB **不可
  行** → 8bit 量化（bitsandbytes Windows 兼容性未验证，探针定）；Dream-v0-7B
  bf16 ≈14.2GB **边缘**（WDDM 开销 + 激活），探针实测定 bf16 或 8bit。
- **权重获取**：HF hub python requests 栈与公司代理不兼容（407/500，AR019 P6 教训）
  → `curl.exe` 手动下载（两 checkpoint 合计约 30GB）；探针产物含下载清单 + SHA256。
- **transformers pin**：LLaDA/Dream remote code 与 transformers 5.x 不兼容
  （AGENTS #24）→ 复用 AR019 venv（transformers==4.53.1 + torch 2.5.1+cu121，
  `--system-site-packages`）。
- **对齐口径不受量化破坏**：gap=0 是**同 checkpoint 同 dtype 内**对齐模式 vs baseline
  的 A/B 对比——量化影响绝对质量，不破坏对齐口径；工件 meta 记录 dtype/量化面。

### 1.3 口径延续

- 四模式与 AR022 同构：**baseline**（各族官方配方可信解码）/ **folded-serial** /
  **folded-batched**（N=4）/ **folded-batched-greedy（对齐模式）**；各族 baseline =
  该族参照采样器官方配方路径（不是 Fast-dLLM baseline）。
- 墙钟纪律全沿 AR021/AR022：宿主 `perf_counter`+sync（#39 CUPTI null 注记）、
  warmup 覆盖、逐 rep round-robin 交错、reps=30 预承诺单次定案不重跑。
- 种子/负载：seed 2026 沿用；prompt 集与各族 token 化适配由探针/冒烟定案（各族
  tokenizer mask token 命名不同，#25：LLaDA `<|mdm_mask|>` / Dream `<|mask|>`）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- FR1 **模型探针（T001 先决）**：探针脚本 + 工件入库——两 checkpoint 下载
  （curl 清单 + SHA256）、transformers 4.53.1 加载、bf16 VRAM 实测、bitsandbytes
  8bit Windows 可行性、短生成冒烟、graph 捕获余量探测；结论字段**定案**量化口径
  （每族 bf16 或 8bit）与 graph 口径（每族开/关 + 注记）
- FR2 **LLaDA 族全链**：投机块候选 + `[N,T]` 批量分支验证 + 贪心锚定 + 生成循环
  折叠 ctx 移植 `LLaDASampler`（右填充块解码画布适配）；机制语义跨族一致
  （贪心行 = argmax 冻结定义、min-prob 公平竞争可落败、种子公式冻结、N 语义总数
  不变、`greedy_win_stats` 只读探针）；默认关 byte-identical
- FR3 **Dream 族全链**：同 FR2 语义（左填充画布 + `position_ids` + bool mask +
  `maskgit_plus` 置信规则适配）
- FR4 **配置面**：`ActFoldConfig` 透传新字段（与 Fast-dLLM 面同构：投机开关 +
  贪心锚定开关；具体字段形态设计期定案）+ per_model YAML 载入可消费
- FR5 **双族四模式基准与工件**：单脚本 `--family llada|dream`（复用 AR022 四模式
  框架与导入面先例）；reps=30 预承诺；**每族一工件**；硬门 = 对齐模式 posthoc
  gap=0 + 解码长度恢复 + 块级 exact + 加速比>1；复现自检（各族 baseline 复跑
  deterministic 一致）
- FR6 **质量门**：951/4/0 基线 + 两族新 UT（默认零扰动冻结面 + 机制面）+
  pyflakes/mypy/black；demo 三锚点不动（合成面与族无关）
- FR7 **文档收口五件**：README 全量中文重写（AGENTS 既定规则）+ AGENTS 新条目
  （跨族投机契约）+ GUIDE 新部分 + CHANGELOG + RERUN 新步骤（下载/量化运维）

**Out of Scope（本 AR 不做的）：**

- eval 任务级质量跑分（lm_eval/evalplus 适配——后续独立 AR）
- 生产单发摊销（AR021 A5 负注记原样延续；本 AR 基准为 warmup+reps 协议）
- 调参族谱（各族官方配方常数冻结：LLaDA low_confidence/block、Dream
  maskgit_plus/eps；投机 N=4/top-k/min-prob 沿 AR021 冻结值）
- CUDA graph 池的族特定性能深挖（可行则开 + 注记，不作硬门；AR017 自适应容量等
  附件按既有契约随 graph 开关联动，不新做优化）
- ctx LRU-池条目生命周期（AR021 design §7 Nit-1 注记原样延续）
- Fast-dLLM 侧任何改动（AR021/AR022 机制冻结面零改动——跨族移植是新增路径非
  改造既有路径）
- 变长/多 parent 批量（AR021 遗留，另立 AR）

**降级规则（预定义，探针结论触发）**：若某族 bf16 与 8bit 均不可行（VRAM/兼容性），
该族降级为 **UT 合成验证 + 工件注记**（机制面 UT 仍全交付，真实 checkpoint 基准面
记 blocked + 原因），**另一族不阻塞**、正常收口。

**FR 间耦合依据**：FR1 是 FR2/FR3/FR5 的先决（量化口径/graph 口径/prompt 适配由探针
定案）；FR2/FR3 是同一机制的两族实例（拆分以便独立验收与降级独立触发）；FR4 是
FR2/FR3 的配置外壳（默认关零扰动，951/4 基线由 FR4 守护）；FR5 是 FR1-FR4 的生产级
仲裁（四模式 A/B 可比性由 FR2/FR3 的同形状 N 语义支撑）；FR6/FR7 为质量与交付收口。

## 3. 功能需求

### 3.1 FR1 模型探针（T001 先决）

**描述**：`scripts/ar023_family_probe.py` 对两 checkpoint 逐项实测并把结论写入
`results/optimization/ar023_family_probe.json`——后续任务的量化口径/graph 口径/
prompt 适配以工件结论字段为准。

**触发条件**：AR 执行 T001（探针先于一切生产改动）。

**期望行为**：
- **下载面**：`curl.exe` 经代理下载两 checkpoint 全量文件（清单 + 每文件 SHA256 入
  工件）；下载目录入工件 meta（复用 `--hf-home`/本地目录惯例）。
- **加载面**：AR019 venv（transformers==4.53.1）逐族加载（`LLaDAModel`/
  `DreamModel` 既有 wrapper + trust_remote_code）；记录加载成败与 remote code 兼容
  性注记。
- **VRAM 面**：bf16 加载实测 `torch.cuda.mem_get_info` 前后差；失败（OOM）则 8bit
  （bitsandbytes Windows wheel 可用性探针：import + `load_in_8bit` 实测）；每族
  结论字段 `dtype_choice: "bf16" | "int8"` + 依据数字。
- **冒烟面**：每族短生成（小 num_tokens）成功 + 输出可 decode；tokenizer mask
  token 解析正确（#25 `_resolve_mask_token_id` 路径）。
- **graph 余量面**：量化口径下 MFF 批量捕获一次的 VRAM 增量实测（可行/不可行 + 余量
  数字）；结论字段 `graph_enabled: bool`（每族独立）。
- **prompt 适配面**：各族 tokenizer 对基准 prompt 集的编码/padding 实测（LLaDA 右
  填充 / Dream 左填充语义走通）。

**异常处理**：单项探针失败不中断其余项——逐项记录 `ok: false` + 原因；族级结论按
降级规则（§2）判定并显式入工件（`degraded: bool` + 原因）。

**验收标准**：
- Given AR019 venv 环境与下载目录就绪，When 探针脚本以默认参数运行，Then 两族
  逐项结论 + SHA256 清单 + 结论字段（dtype_choice/graph_enabled/degraded）完整
  入工件
- Given 某族 bf16 加载 OOM，When 探针接续执行，Then 8bit 探针自动接续且结论记录
  实测依据
- Given 某族 bf16 与 8bit 均不可行，When 探针收敛，Then 该族 `degraded: true` +
  原因入工件，另一族不受影响

### 3.2 FR2 LLaDA 族全链

**描述**：AR021 投机块候选 + 批量分支验证 + AR022 贪心锚定 + 生成循环折叠 ctx
移植 `LLaDASampler`——右填充块解码画布上的投机路径。

**触发条件**：`num_block_candidates=N>0`（族内新开关，默认 0 = 参照路径逐字节
不变）。

**期望行为**：
- **画布适配**：LLaDA 右填充画布（prompt 左对齐 + eos pad）上的投机小块构造——
  块边界/pad 语义/attention_mask 与既有 `build_right_padded_canvas` 一致；投机
  画布一致区前提（AR022 M1 先例：仅投机块起点画布与 baseline 一致时对齐归纳
  成立）同式注记。
- **候选集**：随机行 j 种子公式沿用 AR021 冻结式；`candidate_include_greedy=True`
  时行 0 = draft argmax（冻结定义 `topk sorted` 首位）、确定性零回读、N 语义总数
  不变 j=1..N-1；`True + N=0` 构造期 `ValueError`（AR022 先例）。
- **选定与链延续**：min-prob 公平竞争（仍掩码位 token 概率最小值最大化）、贪心可
  落败（UT 钉死）；胜者分支链延续 `_folding_parent` 切换；eos 早停语义与 LLaDA
  参照路径同式。
- **批量验证**：`[N,T]` `branch_ids` 批量前向（#41 契约：共享单 parent、batch-1
  父缓存展开契约）；MFF eager 链必做；graph 池按 FR1 结论可行则开 + 注记。
- **观测面**：`greedy_win_stats` 只读探针（`{"greedy_wins", "speculative_calls"}`
  跨 sample 存活，AR022 口径同构）；默认关保持 `{0, 0}`。
- **默认零扰动**：`num_block_candidates=0`（默认）时 `LLaDASampler` 全行为与现
  末态逐字节——T018 参照对拍（`test_t018_llada_*`）与基础面
  （`test_llada_sampler_*`）零改动通过。

**异常处理**：`True + N=0` 构造期 `ValueError`；N>0 无 folded_model / 不支持
branch_ids 的既有前置式（AR021 `sample()` 入口先例）族内同构实现。

**验收标准**：
- Given 合成模型 + LLaDA 采样器 + greedy on + N=4，When 投机运行，Then `[4,T]`
  批量形状、行 0 = argmax 填充确定性、行 1..3 与 greedy off 同 seed 在首个投机
  小块（画布一致区）内逐位相同（#44 先例：胜者分叉后两模式画布 fork，非全轨迹
  等价）
- Given 已知后验 fixture，When 选定，Then 贪心可落败（防退化钉死）
- Given 默认配置，When 全部既有面运行，Then 与现末态逐字节（T018 + 基础 UT 零
  改动通过）
- Given `True + N=0`，When 构造，Then `ValueError`

### 3.3 FR3 Dream 族全链

**描述**：同 FR2 语义的 Dream 实例——左填充画布 + `position_ids` + bool
attention_mask + `maskgit_plus` 置信规则适配。

**触发条件**：同 FR2（族内投机开关，默认 0）。

**期望行为**：
- **画布适配**：Dream 左填充画布（prompt 右对齐）+ `compute_position_ids` 忽略左
  pad；投机小块构造在生成区（画布左段）上展开；bool mask（SDPA 契约，
  `dream_sampler.py:80-82` 注记）保持。
- **置信规则适配**：baseline 参照路径 `maskgit_plus`（默认 alg）不动；投机路径的
  min-prob 选定与 draft argmax 锚定语义与 FR2 一致（跨族机制一致性——draft
  argmax 与 min-prob 是投机机制面，与各族 confidence 规则正交）。
- **候选/选定/链延续/批量验证/观测面/默认零扰动**：同 FR2 条目逐项（Dream 实例）。

**异常处理**：同 FR2。

**验收标准**：
- Given 合成模型 + Dream 采样器 + greedy on + N=4，When 投机运行，Then 同 FR2
  第一条（Dream 实例，画布一致区限定同式）
- Given 已知后验 fixture，When 选定，Then 贪心可落败
- Given 默认配置，When 全部既有面运行，Then 与现末态逐字节（
  `test_t018_dream_*` + `test_dream_sampler_*` 零改动通过）
- Given `True + N=0`，When 构造，Then `ValueError`

### 3.4 FR4 配置面

**描述**：族级投机开关 + 贪心锚定开关入族采样器 config dataclass 与
`ActFoldConfig` 透传（YAML 可消费）。

**触发条件**：配置载入 / 构造期校验。

**期望行为**：
- 族采样器 config 新字段（`LLaDASamplerConfig`/`DreamSamplerConfig`）：投机开关
  （`num_block_candidates` 语义同构，默认 0）+ `candidate_include_greedy`（默认
  False）；`__post_init__` 校验（True+N=0 ValueError，AR022 先例）。
- `ActFoldConfig` 透传字段（与 Fast-dLLM 面同构；字段形态设计期定案，如族前缀或
  共享字段——**设计期二选一定案并记录依据**）；YAML 键无未知告警（AGENTS #5）。
- per_model YAML 配置新增（`actfold/configs/per_model/` 惯例）。

**异常处理**：构造期校验同 AR022 式（不静默降级）。

**验收标准**：
- Given YAML 含族投机键，When `load_config`，Then 程序化消费无未知键告警
- Given 默认配置，When 构造各族 sampler，Then 与现末态逐字节（FR2/FR3 零扰动
  交叉引用）

### 3.5 FR5 双族四模式基准与工件

**描述**：`scripts/ar023_cross_family_bench.py --family llada|dream` 四模式
`generate()` 级墙钟 + 质量证据，reps=30 预承诺单次定案，每族一工件
（`results/optimization/ar023_llada_bench.json` / `ar023_dream_bench.json`）。

**触发条件**：探针（FR1）结论非 degraded 的族执行；degraded 族跳过基准面（工件
注记或 tasks.md 记录）。

**期望行为**：
- **四模式构造**：baseline = 该族参照采样器官方配方路径；folded-serial /
  folded-batched / folded-batched-greedy = FR2/FR3 投机路径三档（复用 AR022 四
  模式框架与 AR021 助手先例——导入式复用，禁止复制粘贴重复）。
- **质量硬门（H1，分级口径，OQ1 定案 rev 3）**：对齐模式 vs baseline——
  **硬门层**：质量显著优于 folded-batched（posthoc gap 大幅收缩 + 解码长度恢复 +
  块级 exact > 0）——机制有效性门；**强结论层**：gap == 0 / exact 全中若经验达
  成则 verdict=supported 入工件，**不作收口阻塞**（AR022 的 gap 0.0 是 Fast-dLLM
  经验结果，跨族 baseline 提交语义与 one-shot 贪心填充结构差异更大，无构造保
  证）；判定式冻结入脚本（gap 对比、长度恢复语义与 AR022 判定式同构，阈值分
  级）。
- **速度硬门（H2）**：对齐模式 vs baseline 加速比 p50 > 1（**不预设 7× 量级**——
  量化加载下带宽/算力比变化，AR022 数字是 Fast-dLLM bf16 特定）；速度不回退
  （对齐 vs folded-batched 同形状 A/B 差异归零由构造）。
- **全栈行使（H4 同构）**：MFF_C 池捕获/重放计数、canary（退化 fallback 计数
  ==0 期望）、账本界内、AR017 中性面（graph 开启族）。
- **复现自检**：族内 baseline 复跑 deterministic 一致（贪心 confidence 路径
  temperature 语义逐族注记——LLaDA gumbel off / Dream sample_tokens greedy 面）；
  失配 ⇒ 工件 `reproduction_anomaly` + H1/H2 verdict blocked（AR022 阻断语义
  同构）。
- **工件 meta**：checkpoint SHA256、dtype/量化面（FR1 结论）、transformers/torch
  版本、prompt 集、配方常数、CUPTI null 注记（#39）、graph 开关（FR1 结论）。

**异常处理**：单模式 OOM/失败 ⇒ 该族工件记 blocked + 原因，不伪造数字（AGENTS
#10）；不因单族失败阻塞另一族。

**验收标准**：
- Given 非 degraded 族四模式 reps=30 运行，Then 工件含四模式墙钟分布 + H1/H2/H4
  判定 + 复现自检 + meta 全要素
- Given 对齐模式运行，Then posthoc gap / 长度恢复 / 块级 exact 判定式自动输出
  verdict（supported/refuted/blocked）
- Given 降级族，Then 基准面记 blocked + 降级原因（探针工件交叉引用）

### 3.6 FR6 质量门

**描述**：全量 pytest 基线（951 passed/4 skipped/0 failed 之上只增不减）+ 两族新
UT + 静态检查。

**触发条件**：T007 质量门任务。

**期望行为**：新增 `tests/test_llada_speculative.py` + 
`tests/test_dream_speculative.py`（机制面：候选构建确定性/零回读、贪心锚定、
落败场景、链延续、早停、默认零扰动、构造期校验、YAML 面）；既有冻结面
（`test_diffusion_samplers.py` 全部、`test_fast_dllm_sampler.py`、AR021/AR022 套
件、951/4/0 基线）零破坏；pyflakes 零/mypy clean（`--ignore-missing-imports`）/
black ≤100 列（仅新增行）。

**验收标准**：
- Given 两族新 UT 与全部既有面就绪，When 全量 pytest 在主 python 环境执行，Then
  ≥951 passed/4 skipped/0 failed（新增 UT 计入增长）
- Given lint 三件配置，When pyflakes/mypy/black 对新增与改动文件执行，Then
  clean（pyflakes 零/mypy clean/black ≤100 列仅新增行）

### 3.7 FR7 文档收口五件

**描述**：README 全量中文重写（AGENTS 既定规则：每创新 AR 后全方位重写，数字从
工件重算）+ AGENTS 新条目（跨族投机契约：默认零扰动/机制语义跨族一致/降级注记
面）+ GUIDE 新部分（双族机制与运维）+ CHANGELOG + RERUN 新步骤（curl 下载/量化/
探针/基准复跑指引）。

**验收标准**：
- Given T008 文档五件产出，When 交付审阅（数字溯源逐项核查），Then 全部数字工件
  溯源（无凭空数字）、README 结构覆盖核心创新表新行 + 最新结果表 + 快速上手 +
  质量 + 限制与路线图

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 对齐模式 vs baseline 加速比 p50 | > 1（每族工件硬门 H2） |
| 质量 | 对齐模式质量分级门 | 硬门层：显著优于 folded-batched（gap 大幅收缩+长度恢复+exact>0）；强结论层 gap==0 非阻塞（rev 3） |
| 资源 | 量化口径 VRAM | 探针实测 + 批量池/graph 余量入工件 |
| 可复现 | 权重/种子/环境 | SHA256 + seed 2026 + 版本 pin 全入工件 meta |
| 墙钟 | 测量纪律 | perf_counter+sync、CUPTI null 注记、reps=30 预承诺 |

## 5. 约束与假设

**约束**：
- transformers==4.53.1 pin（remote code，#24）；AR019 venv 复用。
- HF python 栈与代理不兼容 → curl.exe 手动下载（~30GB 磁盘 + 下载时长预算）。
- bitsandbytes Windows 兼容性未验证（探针定；不可行触发降级规则）。
- LLaDA-8B bf16 不可行（16.1GB > 15.3GB 空闲）为既定事实，仅 8bit 路径。
- Fast-dLLM 侧零改动（AR021/AR022 冻结面）。

**假设**：
- LLaDA/Dream remote code 在 transformers 4.53.1 + trust_remote_code 下可加载
  （AGENTS #24 表述隐含 4.x 兼容；探针验证，失败触发降级规则）。
- 8bit 量化下投机机制语义不受破坏（量化影响 logits 数值面但机制面是构造/索引
  逻辑；A/B 同 dtype 对比不受影响）。
- 族 wrapper（`LLaDAModel`/`DreamModel`）既有 forward 面可承载 MFF 折叠栈
  （老 cross-step folding 链已在用 `_forward` 通道；生成循环折叠 ctx 面设计期
  验证）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 族（family） | 模型注册族（`ModelRegistry`）：llada / dream / fast_dllm |
| 对齐模式 | folded-batched-greedy：批量投机 + 贪心锚定候选（AR022 口径） |
| 降级（degraded） | 探针结论：该族真实 checkpoint 基准面 blocked，仅 UT 合成验证 |
| 探针（probe） | T001 先决任务：下载/加载/VRAM/量化/graph/冒烟逐项实测 |
