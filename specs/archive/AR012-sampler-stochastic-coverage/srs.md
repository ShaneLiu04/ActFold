# [AR012] 需求设计说明书 — sampler 随机分支数值覆盖

| 字段 | 内容 |
|------|------|
| AR 编号 | AR012 |
| AR 主题 | 补齐 samplers 随机分支的数值级测试：`add_gumbel_noise` T>0、LLaDA `remasking="random"`、`stochastic_transfer=True`、Dream `alg_temp>0` 软选择、`sample_tokens` T>0 Categorical 路径、`get_num_transfer_tokens` stochastic 数值钉 |
| 关联 SR | 指南 6.2 :229（测试债清单："samplers 只做形状 smoke test……恰是 P1-8 向量化改造最易破坏的分支"）；AGENTS #21（参考采样器定位）；T018 既有覆盖的边界 |
| 日期 | 2026-10-12 |
| 状态 | Final（req 门控 FAIL→修复 PASS；design 门控 FAIL→修复 PASS；T001–T007 passing（14/14 首跑全绿 + mutation 三位点自证）；review PASS（0 Major）；ST 10/10 一次成 Go；已归档） |

## 1. 背景与目标

指南 :229 声称 samplers 仅有形状级 smoke test，且
`stochastic_transfer`、`alg_temp>0`、`remasking="random"`、CFG、
suppress_tokens 均未覆盖。**逐行核实（勘误）**：该条目部分过时——
T018（tests/test_diffusion_samplers.py:545-699）已用冻结参考循环
bit-exact 钉死 greedy（temperature=0）/low_confidence remasking/
suppress_tokens/begin_suppress_tokens/CFG 参考对拍与早停；
tests/test_sampling_utils.py 已钉 NaN/Inf 拒采、greedy 路径、
**非随机** schedule 的官方对拍、host-sync 预算与 stochastic 模式的
dtype/形状/边界不变量（:261-280，无数值语义）。

**真正的缺口是随机分支**（P1-8 向量化改造最易破坏处，本 AR 的对象）：

| # | 分支 | 位置 | 现状 |
|---|------|------|------|
| G1 | `add_gumbel_noise` temperature>0 公式 | sampling_utils.py:236-248 | 仅 :206-207 形状断言，无值级 no-op 钉（req 门控 Minor-1） |
| G2 | LLaDA `remasking="random"` | llada_sampler.py:224-225 | 零覆盖 |
| G3 | `stochastic_transfer=True` 采样级（Binomial schedule 在位） | llada_sampler.py:185 / sampling_utils.py:204-211 | util 级仅不变量（:261），采样级零 |
| G4 | Dream `alg_temp>0` 逐行 multinomial 软选择 | dream_sampler.py:185-192 | 零覆盖（T018 全部 alg_temp=0.0） |
| G5 | `sample_tokens` T>0 Categorical happy path | sampling_utils.py:313-315 | 仅错误传播钉（:43-55） |
| G6 | `get_num_transfer_tokens` stochastic 数值语义 | sampling_utils.py:207-211 | 仅 dtype/边界 |

**钉法的关键认知（req 门控 Major-1 修正）**：T018 式"冻结参考循环
bit-exact 对拍"对随机分支**不采用**，但理由不是"RNG 次序不同必然
不等"——实测冻结参考类与生产实现共享全部 RNG 调用点（random
remasking 同为批量 `rand`、schedule 同调向量化 `get_num_transfer_
tokens`、Dream 同为逐行 `multinomial`），同种子下逐位相等、对拍
今天会通过。问题在对拍因此是**循环性钉**：参考是生产的逐字拷贝，
只证明"代码未变"，不钉数学语义；且会把当前 RNG 布局（Dream 逐行
multinomial、step-major Binomial）过度钉死，P1-8 合法的 RNG 重排
即误报。真正逐行的冻结 schedule 参考（row-major，
test_sampling_utils.py:109-135）与向量化 step-major 实测同种子
不同（[9,4,0,4] vs [9,3,2,1]）——schedule 级对拍同样不可用。
替代钉法：
- **纯函数**（G1/G5）：同种子独立重算公式，逐位相等（对拍对象是
  数学公式而非另一份实现）；
- **端到端采样**（G2/G3/G4）：种子确定性（同种子两次运行
  bit-equal）+ **RNG 调用 spy 证明分支命中**（记录 wrapper，AR011
  "空转通过"教训：必须正向证明路径抵达）+ 结构不变量（全揭示、
  合法词表、单调揭示）；
- **随机 schedule**（G6）：种子确定性 + 与确定性模式可区分 +
  固定种子集合上的统计均值钉（Binomial 参数正确性）。

目标：六个缺口各至少一个数值级用例，使 P1-8（向量化）或任何
未来重构若改变这些分支的数值语义/RNG 契约，测试立即翻红。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- 新测试文件 `tests/test_sampler_stochastic_branches.py`
  （UT-1101~1106，编号续 AR011 的 1001 档）；
- 纯函数精确重算钉（G1/G5）、端到端确定性+spy+结构钉（G2/G3/G4）、
  schedule 数值钉（G6）；
- 文档：指南 6.2 :229 勾选 + **勘误**（CFG/suppress 已由 T018
  覆盖，真缺口是随机分支）、指南第十九部分回链、CHANGELOG
  AR012 节（Tests）。

**Out of Scope（本 AR 不做的）：**
- `actfold/` **零改动**（纯测试 AR；若 Red 期发现实现缺陷，升级为
  显式修复并单独记录——srs 状态不隐藏范围变化）；
- fast_dllm 采样器专属分支（其 T>0 采样全部经 `sample_tokens`，
  由 G5 钉死其数值核心）；
- `folded_model` 交互路径（全部新测试 `folded_model=None`，折叠
  交互已有 T014-T018/AR003 系列覆盖）；
- :230 非 stochastic 官方对拍（T018 :225 已钉）、:231 final-norm、
  :232 scripts helper、:233 conftest autouse seed 耦合（相邻债，
  另立 AR）；
- 随机分支的**分布正确性证明**（如 KS 检验）——钉的是"给定种子
  的确定契约 + 分支命中 + 结构不变量"，分布级验证属官方实现对拍
  范畴（AGENTS #21 免责声明仍然成立）。

## 3. 功能需求

### 3.1 纯函数公式精确重算钉（G1/G5）

**描述：** `add_gumbel_noise` 与 `sample_tokens` 是无状态的纯函数
（单次 RNG 调用），可对公式本身做同种子 bit-exact 重算钉。

**验收标准：**
- **UT-1101（G1）**：Given float32 logits `[2,3,5]` 有限值与
  T∈{0.5, 1.0, 2.0}，`torch.manual_seed(S)` 后调用，Then 输出与
  测试内独立重算
  `logits.double().exp() / (-torch.log(torch.rand_like(...,dtype=float64)))**T`
  逐位相等（同种子、同调用序），且 dtype==float64；同种子两次
  调用输出 `torch.equal`（确定性）；T=0.0 返回原张量（`is` 同对象，
  补 :206-207 形状断言之外缺失的值级 no-op 钉）；
- **UT-1105（G5）**：Given 有限 logits `[4,6]` 与
  T>0（±top_p/top_k 组合），`torch.manual_seed(S)` 后调用
  `sample_tokens`，Then `(confidence, tokens)` 与测试内独立重算
  （`logits/T → top_p/top_k → softmax → Categorical(probs).sample()
  → gather(probs, tokens)`）双返回值逐位相等；margin_confidence/
  neg_entropy 置信度覆写公式在 T>0 下同样逐位重算钉；
  过滤语义钉：被 top_k 置 -inf 的 token 在采样中概率为 0
  （`gather(probs, sampled) > 0` 恒成立——-inf 经 softmax 为 0，
  采到即违约）。

### 3.2 端到端随机分支：确定性 + spy 命中 + 结构不变量（G2/G3/G4）

**描述：** 采样器端到端用确定性 fake model（T018DeterministicModel
先例）驱动；随机分支不可对拍，钉"同种子确定契约 + 分支确实执行 +
输出结构合法"。

**验收标准：**
- **UT-1102（G2）**：Given LLaDA 配置
  `remasking="random", temperature=0.0`，torch.rand 记录 spy，
  When `sample()` 同种子两次，Then：(a) 两输出 `torch.equal`
  （种子确定性）；(b) spy 记录到 shape==x0.shape 的调用
  （随机 remasking 分支命中——**空转通过防护**：若实现静默回落
  low_confidence，spy 为空立即翻红）；(c) 全揭示：新 token 区无
  mask token 残留；(d) 新 token ∈ 合法词表且
    `suppress_tokens` 未出现（suppress 与 random remasking 的
    组合分支）；
- **UT-1103（G3）**：Given LLaDA 配置
  `stochastic_transfer=True, temperature=0.5, remasking="low_confidence"`，
  `torch.distributions.Binomial` 记录 spy，When `sample()` 同种子
  两次，Then：(a) 种子确定性；(b) Binomial spy 非空
  （stochastic schedule 分支命中）；(c) `add_gumbel_noise` 以
  T=0.5 被调用（T>0 在位路径——gumbel 公式本体已由 UT-1101 钉，
  此处钉组合）；(d) 全揭示 + return_history 下画布单调
  （每步揭示数≥0、总量不减、终止全揭示）；
- **UT-1104（G4）**：Given Dream 配置 `alg_temp=1.0, alg=
  "maskgit_plus", cfg_scale=0.0`，`torch.multinomial` 记录 spy，
  When `sample()` 同种子两次，Then：(a) 种子确定性；(b) multinomial
  spy 非空  （逐行软选择分支命中——若实现静默走 alg_temp=0 批量
  topk，spy 为空翻红）；(c) 全揭示；(d) **软选择收敛性质**
  （req 门控 Minor-2 修正机制描述）：`alg_temp=1e-8` 时
  `softmax(full_confidence/alg_temp)` 数值上精确 one-hot
  （torch 2.5.1 实测），`multinomial(one-hot, k>1)` 第 2+ 样本
  取自零概率索引、与置信度 top-k **不等**——故 bit-equal 预期
  直接降级，钉弱性质"每行 multinomial 首个采样索引 == 该行
  置信度 argmax"（one-hot 下首样本必为 argmax）；(a)-(c) 为
  硬门，(d) 以弱性质形式保留。

### 3.3 随机 schedule 数值钉（G6）

**验收标准：**
- **UT-1106**：Given steps=8——(a)/(b) 用随机密度
  `mask_index`（固定生成器），(c) 统计钉用**单行全 mask**
  `torch.ones((1,20), bool)`（n=20 恒定，design 门控 Major-1：
  统计常数与构造的 Σn 必须一致），Then：(a) 同种子两次调用输出
  `torch.equal`；(b) 与
  `stochastic=False`（期望值取整模式）输出可区分——固定一个实测
  不同的种子钉死（分支命中证明：若 stochastic 参数被静默忽略，
  两输出恒等翻红）；(c) **统计均值钉**：固定 100 个种子上，第 0
  步传输量均值 ≈ 解析期望 `n·p₀`，容差 ≥6σ。p₀ 解析式 =
  `1 − α(s)/(α(t)+1e-6)`（`_round_f32` 三次舍入后，
  steps=8/j=0 实测 **0.12500083446502686**；n=20 期望
  2.5000167、σ=1.4790、100 种子 6σ 容差 **±0.887**，req 门控
  WARN-3 数值）；解析期望在测试中独立重算（不调用生产
  代码——钉参数正确性，非循环验证）。

### 3.4 文档收口

**验收标准：**
- 指南 6.2 :229 勾选 + 勘误注（原文列 CFG/suppress 未覆盖与
  T018 现状不符——勘误记录比照单实现更重要，AR008-011 先例）；
- 指南第十九部分 AR012 回链；CHANGELOG AR012 节（Tests，
  Unreleased 下最前）；
- AGENTS.md **预计零改动**（纯测试 AR，无新契约/坑位；若 Red 期
  出现实现修复或新认知，再评估）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 静态检查 | mypy 标准命令 clean / pyflakes 零 / 100 列（black 未装逐行查） |
| 质量 | 全量回归 | `pytest tests\ -q -m "not slow"`：785 passed 基线 + 新增零失败 |
| 稳定 | 测试确定性 | 所有钉用显式 `torch.manual_seed`/`Generator`，不依赖 conftest autouse seed（:233 债的反向示范）；统计钉容差 ≥6σ 防平台 RNG 差异 |
| 性能 | 用例耗时 | 单文件 <10s（fake model 小形状；统计钉 100 种子 × 小张量） |
| 兼容 | 生产代码 | `actfold/` 零改动（3.2 Out of Scope 的升级条款除外） |

## 5. 约束与假设

**约束：**
- 不引入与生产实现共享代码路径的"参考重算"（G1/G5/G6 的独立重算
  必须逐算子手写，防止生产代码改了测试跟着绿的循环验证）；
- spy 用 monkeypatch 记录 wrapper（还原原函数、记录入参 shape），
  不改变被测行为；**钉点清单**（req 门控 WARN-1）：`add_gumbel_
  noise` 须钉 `actfold.models.llada_sampler.add_gumbel_noise`
  （from-import 绑定，钉 sampling_utils 属性不影响该调用点）；
  `torch.rand`（llada_sampler.py:225）/`torch.multinomial`
  （dream_sampler.py:191）模块级直呼可全局钉；
  `torch.distributions.Binomial` 钉类并代理 `.sample()`（补丁
  窗口内仅 get_num_transfer_tokens 使用，波及面可控）；
- 端到端测试 `folded_model=None`、`return_history=True`（画布
  不变量需要历史）。

**假设（req 门控已核实闭环）：**
- T018DeterministicModel.forward 依赖画布 x（:243-245 读 tokens：
  `seq_bias=cos(0.29·tokens.sum)`、`+0.037·tokens`）→ G2 的
  low_confidence 对比证据按主案可行（揭示次序不同 → 中间画布
  不同 → 终态序列不同）；置信度并列风险实证极低（sin/cos 连续
  构造、各位置参数不同）；
- `torch.distributions.Binomial`/`Categorical`/`multinomial` 的
  同种子输出在 torch 2.5.1 固定平台上是稳定契约（钉的是当前
  平台确定契约，跨平台/跨版本不承诺 bit-exact——与 T018 冻结
  参考同样的局限）；生产采样器同种子两次运行 bit-equal 已经
  req 门控实证（钉法前提成立）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 随机分支 | 采样器中消耗 RNG 的路径：gumbel 噪声、random remasking、Binomial schedule、multinomial 软选择、Categorical 采样 |
| RNG spy | monkeypatch 记录 wrapper：包装 `torch.rand`/`torch.multinomial`/`Binomial.sample` 等，断言分支命中的正向证据 |
| 空转通过 | 负断言（spy 为空/输出不变）在实现静默走错分支时仍然通过的假绿（AR011 req 门控教训） |
| 全揭示 | 采样结束时新生成区无 mask token（`x == mask_token_id` 在新 token 区为 False） |
