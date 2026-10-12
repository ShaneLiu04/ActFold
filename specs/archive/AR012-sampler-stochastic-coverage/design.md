# AR012 详细设计 — sampler 随机分支数值覆盖

> srs：`specs/changes/AR012-sampler-stochastic-coverage/srs.md`（权威契约）
> 创建：2026-10-12；状态：final（design 门控第 1 轮 FAIL→修复 Major-1 + Minor-1~4 + WARN-1~4 采纳；复验见 tasks.md）

## 1. 设计总览

方案：**A（独立重算 + spy 钉法，弃用冻结参考对拍）**——六个缺口
（G1-G6）全部落在**一个新测试文件** `tests/test_sampler_stochastic_
branches.py`，UT-1101~1106；生产代码零改动。

**否决 B（沿用 T018 冻结参考对拍扩展到随机分支）**：req 门控
Major-1 实测冻结参考与生产共享全部 RNG 调用点、同种子 bit-equal
——对拍是循环性钉（参考=生产逐字拷贝，只证明代码未变，不钉数学
语义），且会把当前 RNG 布局（Dream 逐行 multinomial、step-major
Binomial）过度钉死，P1-8 合法 RNG 重排即误报。

**否决 C（分布级统计测试全覆盖，KS 检验等）**：跨平台/样本量敏
感、慢；srs Out-of-Scope 已明确钉"当前平台确定契约"，仅 G6 保留
一个 6σ 宽容差的统计均值钉（钉 Binomial 参数，非分布形状）。

## 2. 测试架构

### 2.1 复用与新建

| 组件 | 来源 | 说明 |
|---|---|---|
| 确定性 fake model | `T018DeterministicModel`（tests/test_diffusion_samplers.py:213-266，design 门控 Minor-3 勘误）**import 复用**（`from tests.test_diffusion_samplers import ...`，借 pyproject `pythonpath=["."]`；模块顶层仅类定义无副作用） | 画布依赖（req 门控 WARN-2 已核实）；避免第三份 fake model 拷贝 |
| 采样器配置与调用 | **散参形态**（design 门控 Minor-1 勘误）：`LLaDASamplerConfig`/`DreamSamplerConfig`（T018 :588-596/:677-695 形态，`Config(**kwargs, seed=…, return_history=True)`）+ `sampler.sample(prompt_ids, folded_model=None)`；"全 mask 新 token 区"由 `sampler.initialize` 内部构建，测试只传 prompt 张量 | `SamplerInput` 不存在，勿引用 |
| spy 机制 | monkeypatch 记录 wrapper（T016 :803-806 先例：包装 + 记录 + 还原原函数） | 钉点清单见 §2.2 |

### 2.2 spy 钉点（srs §5，req 门控 WARN-1）

| spy 目标 | monkeypatch 位置 | 记录内容 | 波及面 |
|---|---|---|---|
| `add_gumbel_noise` | `actfold.models.llada_sampler.add_gumbel_noise`（from-import 绑定；钉 sampling_utils 属性**无效**） | (tensor shape, temperature) | 仅 LLaDA 采样路径 |
| `torch.rand` | `torch.rand`（llada_sampler.py:225 模块级直呼） | shape | 测试内全局（fixture 窗口内无其他消费者） |
| `torch.distributions.Binomial` | 类替换 + 代理 `.sample()`（sampling_utils.py:208 构造即调） | (total_count, probs) + sample 返回 shape | 补丁窗口内仅 get_num_transfer_tokens |
| `torch.multinomial` | `torch.multinomial`（dream_sampler.py:191 模块级直呼） | (input shape, num_samples) | 同 torch.rand |

实现要点：wrapper 先记录再委托原函数（**不改变被测行为**）；
`Binomial` 代理类须接受 `(total_count, probs)` 构造并转发 `.sample()`
到原类实例（保持 RNG 消耗不变——spy 只观察不重排）。

### 2.3 显式种子纪律（srs NFR）

每个用例开头显式 `torch.manual_seed(S)`（S 固定常量），不依赖
conftest autouse seed=42（:22-27）——正用例自带种子即自证确定性；
G6 统计钉用 `torch.Generator` 局部生成器循环 100 个种子。

## 3. 用例设计（UT-1101~1106）

### 3.1 纯函数精确重算（G1/G5）

| ID | 构造 | 独立重算（测试内逐算子手写） | 断言 |
|---|---|---|---|
| UT-1101 | `logits = torch.randn(2,3,5)`（固定种子）；T ∈ {0.5, 1.0, 2.0} 参数化 | `manual_seed(S)` → `noise = torch.rand_like(logits.double())` → `out = logits.double().exp() / (-torch.log(noise)) ** T` | `torch.equal(prod, recompute)`；`prod.dtype == torch.float64`；同种子两次 `torch.equal`；T=0.0 时 `add_gumbel_noise(logits, 0.0) is logits` |
| UT-1105 | `logits = torch.randn(4,6)`（固定种子）；矩阵参数化 {T>0 单独, +top_k, +top_p, **+top_p+top_k 联合**, +margin, +neg_entropy}；生产过滤顺序 **top_p 先、top_k 后**（sampling_utils.py:306-309），联合用例镜像该顺序 | `manual_seed(S)` → `lg = logits/T` → **逐算子镜像**：top_p（sort(descending)→softmax→cumsum→严格 `>`→`[...,0]=False` 保首→scatter→masked_fill，WARN-1：并列值跨 nucleus 边界时 removal 集合才 bit 相等，禁用"保留 cumsum≤top_p 前缀"等替代表述）；top_k（`torch.topk`→`values[...,-1]`→`where(lg<min, -inf, lg)`，平 tie 全保留）→ `probs = softmax(lg)` → `tok = Categorical(probs).sample()` → `conf = gather(probs, tok)`；margin/neg_entropy 按生产公式覆写 | `(conf, tok)` 双返回值逐位相等；`gathered prob > 0` 恒成立（-inf 过滤语义） |

### 3.2 端到端随机分支（G2/G3/G4）

公共骨架：fake model + `manual_seed(S)` → 构造 `SamplerInput`
（全 mask 新 token 区，T018 形态）→ spy 安装 → `sampler.sample()`
两次（各自重新 `manual_seed(S)`）。

| ID | 配置 | 硬断言 | 分支命中证据 |
|---|---|---|---|
| UT-1102 | LLaDA `remasking="random", temperature=0.0, suppress_tokens=[3]` | (a) 两运行序列+history `torch.equal`；(c) 新 token 区无 mask 残留；(d) 新 token ∉ suppress 且 ∈ [0, vocab) | (b) `torch.rand` spy 记录 ≥1 次且 shape == (B, T)（空则实现静默回落 low_confidence，翻红） |
| UT-1103 | LLaDA `stochastic_transfer=True, temperature=0.5, remasking="low_confidence"` | (a) 同种子确定性；(c) 全揭示；history 单调（每步新揭示 ≥0，累计不减，终态全揭示） | (b) `Binomial` 代理 spy 非空；(c′) `add_gumbel_noise` spy 以 temperature=0.5 被调 |
| UT-1104 | Dream `alg_temp=1.0, alg="maskgit_plus", cfg_scale=0.0` | (a) 同种子确定性；(c) 全揭示 | (b) `torch.multinomial` spy 非空（静默走批量 topk 则空，翻红）；(d 弱性质) `alg_temp=1e-8` 运行时每行首个 multinomial 采样索引 == 该行置信度 argmax（one-hot 首样本必 argmax；第 2+ 样本零概率回填不钉，req 门控 Minor-2） |

**空转通过防护**（AR011 教训）：三个用例的 spy 非空断言是分支命中的
正向组成证据；配合 (a) 确定性——若 spy 记录到调用但输出不确定，
或 spy 空但其余断言过，均立即翻红。

**G2 对比证据**（画布依赖已核实）：同种子下 random 与
low_confidence 两运行的**历史画布**揭示次序不同（`x0_p` 来源不同
→ 提交集合不同 → 中间画布不同）；终态序列因画布依赖亦应不同——
作为软证据记录（构造性差异由 spy 硬证据承担，避免偶合翻转 flaky）。

### 3.3 schedule 数值钉（G6，UT-1106）

构造（design 门控 Major-1 修正——统计钉用单行使 n=20 常数有效）：
- (a)/(b) 可区分钉：`gen = torch.Generator().manual_seed(7)`；
  `mask_index = torch.rand((4, 20), generator=gen) < 0.6`；steps=8；
- (c) 统计钉：`mask_index = torch.ones((1, 20), dtype=torch.bool)`
  （单行全 mask，n=20 恒定）；种子 0..99 循环。

- (a) `manual_seed(S)` 两次调用输出 `torch.equal`；
- (b) 分支可区分：固定 S 使 `stochastic=True` 输出 ≠
  `stochastic=False`（Red 期实测选定 S 并钉死——若实现静默忽略
  stochastic 参数，两输出恒等翻红）；
- (c) 统计钉：种子 0..99 循环，第 0 步传输量均值 vs 解析期望
  `n·p₀ = 2.5000167`（n=20 单行），断言容差 ±0.89（6σ = ±0.887，
  σ=√(n·p₀·(1−p₀))=1.4790）；p₀ 独立重算公式（design 门控
  Minor-2 修正，复刻生产 :85-89 四次舍入 + :200 double 减法）：
  `p₀ = 1 − _f32r(_f32r(1−α(s)) / _f32r(_f32r(1−α(t)) + 1e-6))`
  = 0.12500083446502686（steps=8/j=0: s=7/8, t=1）；解析重算
  不 import 生产 scheduler（防循环验证）。末步 p=1（j=7:
  s=0→α=1→num=0→p=1）保证 Binomial 全量传输——"终态全揭示"
  断言的机制依据（design 门控核实）。

### 3.4 Red 预期

本 AR 是**行为钉子型**（production 本就正确，非修 bug）：六个用例
对当前实现**首跑应全绿**（design 门控对 UT-1106(c) 构造修正后
成立——Major-1 修复消除了首跑必红的自相矛盾）。Red 期价值 =
验证测试自身可翻红：
- 用例编写完成后主代理运行，若任一翻红 → 生产缺陷或测试构造错误，
  停顿裁决（srs 3.2 升级条款）；
- 自证翻红（mutation spot-check，Red 子代理执行）：临时把
  `remasking == "low_confidence"` 判断反义（llada_sampler.py:221，
  random 走 else 分支——design 门控 WARN-4 措辞对齐）/
  `stochastic` 参数硬编码 False / `alg_temp` 分支条件改 `!=`
  各跑一次对应用例，确认 spy 断言捕捉（design 门控逐位点推演：
  三个 mutation 均由对应 spy 空断言先红——(i) UT-1102(b)、
  (ii) UT-1103(b) 且 UT-1106(b) 双保险、(iii) UT-1104(b)；
  **Red 期实测勘误（review Minor-2）**：(ii) 的"双保险"仅在
  mutation 落在 sampling_utils 层时成立——实际位点在 llada 调用
  层，UT-1106 不红，由 UT-1103(b) 单点捕捉），
  还原后全绿。

## 4. 影响范围

| 文件 | 改动 |
|---|---|
| tests/test_sampler_stochastic_branches.py（新） | UT-1101~1106（~230 行） |
| docs/OPTIMIZATION_GUIDE.md | :229 勾选 + 勘误注；第十九部分回链 |
| CHANGELOG.md | AR012 节（Tests） |

**零改动声明**：`actfold/` 全目录、AGENTS.md（预计零改动）、
README、既有 tests/* 文件（`T018DeterministicModel` 以 import
复用，不拷贝不修改）。

## 5. 决策记录

| # | 决策 | 理由 |
|---|---|---|
| D1 | 弃用冻结参考对拍，改独立重算+spy+确定性 | req 门控 Major-1：对拍=循环性钉 + 过度钉死 RNG 布局 |
| D2 | 独立重算逐算子手写、不 import 生产函数 | 防生产代码改了测试跟着绿的循环验证（srs §5 约束） |
| D3 | `T018DeterministicModel` import 复用 | 第三份 fake model 拷贝是维护债；画布依赖已核实满足证据形态 |
| D4 | UT-1104(d) 降级为"首采样索引==argmax"弱性质 | req 门控 Minor-2：one-hot 下 multinomial 第 2+ 样本零概率回填 ≠ top-k，bit-equal 预期必然失败 |
| D5 | G6 分支可区分钉用"实测选定的固定种子" | 随机断言会 flaky；固定 S 钉死确定性差异，实现忽略参数时恒等翻红 |
| D6 | mutation spot-check 进 Red 流程 | 行为钉子型 AR 的测试自证：全绿首跑无法证明测试有效，临时反转分支证捕捉力 |
| D7 | 单文件收纳六用例 | 同主题（随机分支）、共享 spy 骨架与种子纪律；避免六个微文件 |

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| spy 改变 RNG 消耗（wrapper 内多次/少次消费） | wrapper 严格"记录后单次委托"；UT-1101/1105 纯函数重算钉不依赖 spy，交叉验证 |
| `torch.rand` 全局 patch 波及用例内其他消费者 | spy 安装紧贴被测调用；断言记录 shape 匹配 x0.shape 而非仅非空 |
| Binomial 代理类破坏 `.sample()` RNG 契约 | 代理仅转发构造与 sample 调用到原类实例；构造不消费 RNG、仅 `.sample()` 消费（sampling_utils.py:208-211 先构造后立即 sample，转发保序成立——design 门控 WARN-3：UT-1103(a) 确定性断言只能捕捉非确定性扰动，确定性扰动的真实保障是代理转发本身） |
| G6 统计钉跨平台 flaky | 6σ 容差（±0.89 vs σ/√100=0.148）；torch 平台内 RNG 确定性已由 T018/req 门控实证背书 |
| T018DeterministicModel import 造成测试间耦合 | module-level import（`from tests.test_diffusion_samplers import ...`，pyproject `pythonpath=["."]`）——import 失败直接 collect ERROR（fail loud，design 门控 WARN-2 措辞修正：不留 skip 出口）；该类自 T018 冻结后未动，漂移面极小 |
| 用例耗时超标 | 小形状（B≤4, T≤24, vocab≤16）+ G6 100 小种子循环，预估 <5s |
