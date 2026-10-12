# [AR012] 任务分解 — sampler 随机分支数值覆盖

基线：main `bba5b20`（树 `1d3f0ce`），全量 785 passed / 3 skipped，mypy clean（65 文件）。
约束：`actfold/` 零改动（纯测试 AR）；UT 编号 1101~；新文件 `tests/test_sampler_stochastic_branches.py`。

## 任务

- [x] T001 Red：子代理实现 UT-1101~1106（14/14 首跑全绿——行为钉子型符合 design 预言；mutation 三位点均翻红后逐字节还原；实测裁决：UT-1104(d) 弱性质可靠保留、import 用 `from tests.test_diffusion_samplers` 形态、(ii) 位点 UT-1106 不红——双保险仅对 sampling_utils 层位成立，design §3.4 已勘误（review Minor-2）、UT-1106(c) 测量口径偏置已补注 docstring（review Minor-1：首个正传输列的零步替代使均值偏高，6σ 带吸收））
- [x] T002 Green：主代理亲验（14 passed 复跑一致；独立重算零生产函数 import 审计通过——仅 import 三个被测函数本体；pyflakes 零/100 列）
- [x] T003 全量回归：**799 passed（785+14）/ 3 skipped / 3 deselected** 零失败；demo 三指标精确一致（85.5%/2.35e-03/93.75%）
- [x] T004 文档：指南 :229 勾选+勘误、第十九部分回链、CHANGELOG AR012 节
- [x] T005 质量门：mypy clean（65 文件）/ pyflakes 零 / 100 列
- [x] T006 review 子代理：**PASS**（0 Major + 2 Minor + 3 WARN；A-E 五维核查；Minor-1/2 已修复：UT-1106 docstring 测量口径注 + design §3.4 双保险实测勘误；WARN-1/2/3 不处置理由在案）
- [x] T007 ST：st-cases.md **一次成 Go（10/10 PASS）**——ST-001~006 映射 srs §3 验收、ST-007 mutation 自证、ST-008 文档、ST-009 质量门（799 passed/demo 精确/mypy/pyflakes/100 列）、ST-010 生产零改动审计
- [ ] T008 归档 `specs/archive/AR012-sampler-stochastic-coverage/` + git commit + commit-tree 推 gitee + 树验证

## 门控记录

### req 门控
- **首轮 FAIL**（1 Major + 2 Minor + 3 WARN，只读子代理逐行核实）：
  - Major-1：核心认知"冻结参考 RNG 次序不同→对拍必然不等"**不成立**——冻结参考与生产共享全部 RNG 调用点（同批量 rand/同向量化 schedule/同逐行 multinomial），同种子实测 bit-equal；真实问题是**循环性钉**（参考=生产逐字拷贝，只证明代码未变）+ 过度钉死 RNG 布局（P1-8 合法重排即误报）。→ srs §1 已改写，结论（弃对拍）不变、依据更正
  - Minor-1：:206-207 仅形状断言非 equality 钉（现状比原描述更弱）→ G1 表格与 UT-1101 已修正
  - Minor-2：alg_temp=1e-8 收敛性质真实失效模式 = softmax 精确 one-hot 后 multinomial 第 2+ 样本取零概率索引 ≠ top-k（非平局次序）→ UT-1104(d) 降级为"每行首个采样索引==置信度 argmax"弱性质
  - WARN-1 采纳：spy 钉点清单入 §5（add_gumbel_noise 须钉 llada_sampler 命名空间的 from-import 绑定）
  - WARN-2 采纳：T018DeterministicModel 画布依赖已核实，假设条款闭环
  - WARN-3 采纳：p₀=0.12500083446502686 / 期望 2.5000167 / σ=1.4790 / 6σ=±0.887 数值入 UT-1106
  - 附核实：fast_dllm T>0 确全经 sample_tokens（Out-of-Scope 声明成立）；生产采样器同种子两次运行 bit-equal 实证（钉法前提成立）
- **修复后 PASS**（Major/Minor 全修，WARN 全采纳）

### design 门控
- **首轮 FAIL**（1 Major + 4 Minor + 4 WARN，只读子代理逐行核实）：
  - Major-1：G6 统计钉构造 (4,20)@0.6 的 Σn≈48 与硬编码 n=20 常数自相矛盾——按字面实现首跑必红（|6.0−2.5|=3.5 > 0.89）且会误触 srs 升级条款误报生产缺陷。→ 采用修复方案 (i)：统计钉改单行全 mask `torch.ones((1,20), bool)`（n=20 常数全部有效），(4,20)@0.6 保留给 (a)/(b)；srs 3.3 同步修正
  - Minor-1：`SamplerInput` 不存在——实际散参形态 `sample(prompt_ids, folded_model=None)` + Config kwargs → design §2.1 已勘误
  - Minor-2：p₀ 复刻公式缺外层舍入 → 修正为 `p₀ = 1 − _f32r(_f32r(1−α(s)) / _f32r(_f32r(1−α(t)) + 1e-6))`（常数 0.12500083446502686 独立验算通过）
  - Minor-3：T018DeterministicModel 实际位置 :213-266 → 已勘误
  - Minor-4：UT-1105 参数矩阵缺 top_p+top_k 联合组合；生产顺序 top_p 先 top_k 后 → 已补并明示顺序；"top_k/top_k" 笔误已修
  - WARN-1 采纳：top_p 手写逐算子镜像（sort desc→softmax→cumsum→严格>→保首→scatter→masked_fill），禁替代表述
  - WARN-2 采纳：import 形态 `from tests.test_diffusion_samplers import ...`（pythonpath=["."]）；module-level import 失败即 collect ERROR（无 skip 出口）
  - WARN-3 采纳：UT-1103(a) 兜底措辞修正——确定性扰动的真实保障是代理转发本身（构造不消费 RNG）
  - WARN-4 采纳：mutation 位点措辞对齐（生产是 `remasking == "low_confidence"`，random 走 else）
  - 附核实：三个 mutation 位点均被 spy 空断言先捕捉（(ii) 有 UT-1106(b) 双保险）；末步 p=1 保证 Binomial 全量传输（全揭示断言的机制依据）；四 spy 钉点全部可行
- **修复后 PASS**（Major/Minor 全修，WARN 全采纳；Major-1 修复为机械性构造变更，常数不变）

### review 门控
- **PASS**（0 Major + 2 Minor + 3 WARN，只读子代理 A-E 五维核查）：
  - A spec 符合性：srs §3.1-3.3 验收标准逐条在测试中落实（UT-1101~1106 全部断言核对）
  - B design 符合性：spy 钉点/种子纪律/D1-D7 落实；D2 import 审计通过（仅三被测函数本体）；UT-1106(c) 单行全 mask（Major-1 修复）落实
  - C git diff 逐字节对照：actfold/ 零改动；改动面仅新测试文件+指南+CHANGELOG+specs 工作文档，无夹带
  - D 运行验证：14 passed / 全量 799 passed 复算一致；pyflakes/列宽/docstring 完整；文档事实核对逐项一致（p₀ 常数、均值 2.70、容差）
  - E 回归风险：低——全文件 CPU device-agnostic、固定种子确定性断言无 flaky、import 副作用零
  - Minor-1 修复：UT-1106 docstring 补测量口径注（零步替代偏置、6σ 带吸收）+ tasks.md 陈述修正
  - Minor-2 修复：design §3.4 补 Red 期实测勘误（双保险仅在 sampling_utils 层位成立）
  - WARN-1/2/3 不处置：fail-loud 方向可接受、平台契约已在 srs §5 声明、断言强于 spec 无风险

### ST 门控
- **一次成 Go（10/10 PASS）**：ST-001~010 全过（详见 st-cases.md）；结论——srs §3 全部验收标准落实、测试经 mutation 自证非空转绿、全量回归与 demo 零漂移、生产零改动审计干净，AR012 可归档
