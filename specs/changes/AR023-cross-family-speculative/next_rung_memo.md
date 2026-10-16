# P1-7 第十档候选方向备忘（AR023 结档后启动）

> 基于 AR023 的机制发现与实机证据起草；优先级以 T006 预承诺工件的正式
> verdict 定案（决策树见文末）。本文档为方向备忘，非 AR 立项——立项走
> harnessX req 阶段完整流程。

## 候选 A：LLaDA 整块提交质量面机理缓解

**证据**：quick 冒烟 LLaDA H1 refuted——batched 区 offset-2 EOS 早停
（len 3 vs 64），aligned==batched 逐位同（贪心锚定未恢复质量面）；
EOS 审计确认非代码缺陷（suppress 对称），是"空白画布单发整块提交 vs
8 轮渐进去噪"的固有质量面差。对照组：Dream 超步窗口质量面保持好得多
（长度恢复 64/64/64）。

**机理假设**：LLaDA 的投机单元 = 整块 one-shot 填充，把全部去噪压力压在
一次前向；Dream 的超步分批提交天然渐进。质量面差是**单元粒度**的函数，
不是族的函数。

**方向**：①子块提交——LLaDA 单元从整块改为 `s` 子块序列（复用 Dream
超步的预算合并/置信度选位语义，机制核不动，只改族适配器的单元定义）；
②渐进填充——单元内多轮 draft-commit（每轮提交 top-k 置信位）；③
EOS 感知候选——候选行构建时保证至少一行含 stop-token 假设（贪心锚定
的自然扩展）。验收协议直接复用 AR023 四模式基准（H1 三硬面）。

**风险**：LLaDA 官方配方是块解码——子块化会偏离参照配方，N=0 参照
路径必须保持官方语义不动（投机路径内部再分轮），否则 A/B 失去参照锚。

## 候选 B：eval 适配器改造——投机模式任务级质量跑分

**证据**：AR021/AR022/AR023 三档的质量面全部是内生指标（posthoc 置信 /
exact-match / 解码长度），从未在任务级（gsm8k/math/ifeval）上对比投机
模式 vs baseline。AR022 遗留注记已三次提及"eval 适配器改造，后续独立
AR"。内生质量面对齐（gap=0.0）不必然推出任务分数对齐。

**方向**：`BaseEvalAdapter` + `folded_generate` 既有面接入
`DiffusionSampler` 投机路径（当前 eval 适配器走 `generate()` 的
`folded_model` 面，投机开关的传导需显式化）；LM-eval 任务上跑
baseline / folded-batched-greedy 双模式对比；预承诺任务集 + 样本数。

**价值**：所有投机创新的证据从内生面对齐升级到任务级对齐——这是
"最终看的是综合框架比 baseline 要强"总纲的直接证据面。族覆盖：
Fast-dLLM（AR022 机制已定案）优先，LLaDA/Dream 视 T006 verdict。

**风险**：evalplus 面 Unix-only（Windows 不可跑，CI deselect 先例）；
LM-eval 真后端依赖（#6）；任务级跑分 GPU 时长远超四模式基准——
预承诺样本数要量力。

## 候选 C：graph 静态体元组解包（backlog 工程项）

**证据**：UT-2335 + AGENTS #45 caveat——`FoldedGraphRunner` 自有层循环
不解包元组层返回，而真实 LLaDA 远程 block 返回元组。两族探针结论
graph-off（16GB 卡捕获余量不足），故未演练。

**定位**：前置条件型工程项，不独立立项——任何未来 graph-on 族面
（换更大显存硬件 / int8 量化口径）开工前必做。挂 backlog，跟随下一个
需要图捕获的 AR 顺带完成。

## 决策树（T006 定案后）

| T006 正式 verdict | 第十档建议 |
|---|---|
| LLaDA H1 refuted（quick 已示，概率高） | **A 优先**——质量面缺口是当前主矛盾，且 A 的验收协议（四模式 H1）现成；B 次之 |
| LLaDA H1 supported（正式尺度翻案） | **B 优先**——跨族机制故事已闭环，任务级证据是把创新从机制面推向发表面的关键台阶；A 降为观察项 |
| Dream H1 refuted（超出 quick 预期） | 重审机制核假设（Dream quick 三面里 a 面过界但 b/c 过——正式 refuted 意味着 a 面系统性），先归因分析再谈 A/B |
| 任一族 blocked（OOM/复现失配） | 优先处理 blocked 根因（复现失配 = 确定性假设破裂，影响所有后续测量的可信度），A/B 暂缓 |
