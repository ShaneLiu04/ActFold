# [AR022] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR022 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（design 门控 PASS）|
| 创建日期 | 2026-10-15 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | Red：`tests/test_speculative_greedy_alignment.py`（UT-2201~2207，映射 FR1-FR3），子代理编写，红因全=生产缺失（`candidate_include_greedy` 字段/贪心行构建/greedy_win 观测面不存在） | - | passing | 2026-10-15 完成：8 测试函数/1289 行全红（红因逐条=TypeError 未知字段，2201(e)/2204/2207 AttributeError 面被前置 TypeError 遮蔽=复合红因结构预期内）；冻结面 23 passed 零破坏；fixture：`_PosteriorTrapFixture`（种子重放 all-C seed=584 已对 greedy-off 实跑验证 row 1 全抽 C）/_ReadbackSpyMFF/_recompute_min_scores；子代理 ses_ec283a1ecffeiVktVyA5N8YXxN |
| T002 | Green：`fast_dllm_sampler.py`（config `candidate_include_greedy` 字段 + `__post_init__` 校验 + 贪心行构建（行 0 = top-k 有序集首位、随机行 j=1..N-1 种子公式原样）+ greedy_win 观测计数）+ `config_manager.py`（`ActFoldConfig.candidate_include_greedy` YAML 消费面，设计 §4.1.3 双面定案）；mutation 全红还原；全量回归 | T001 | passing | 2026-10-15 完成：8/8 一次 Green；全量 951 passed/4 skipped/0 failed（AR021 基线 943 + 新增 8；2 例 lm_eval 环境缺失失败先在——stash 基线复跑证实与 AR022 无关，deselect 排除留档；skip=1 CUPTI+3 evalplus-unix）；black/pyflakes/mypy 67 文件 clean；mutation 5/5 全杀全还原（M1 贪心行取末位→5 failed/M2 种子 j 偏移→3 failed 含冻结面 UT-2109c/M3 恒选贪心→2 failed/M4 计数缺失→4 failed/M5 校验条件弱化→1 failed；SHA256 逐次还原恒 57A24F00…）；UT-2108/2109/2109c/2110 冻结面零改动通过 |
| T003 | 基准脚本 `scripts/ar022_speculative_quality_bench.py`（复用 `scripts/ar021_generation_loop_bench` 助手与三模式构造——**导入式**复用（设计 §4.1.4 定案，AR021 导入 ar019 共享助手先例的推进），ar021 脚本本体零改动；新增第四模式 folded-batched-greedy（MFF_C 独立实例）+ greedy_win_rate / 块级+位置级双粒度 exact-match / 解码长度出口）+ `--quick` 冒烟 | T002 | passing | 2026-10-15 完成：1343 行，导入面 30 符号 + 模块别名 `_ar021` + 本地新增 12 函数 + `_AR021_PRIORS` 逐位核对；三处耦合注记全落地；black/pyflakes/mypy clean；quick 冒烟（真实 checkpoint）：四模式 p50 2904.3/2989.0/765.3/371.2ms（greedy_vs_batched 2.06×、vs baseline 7.83×）、对齐面 block exact 1.0/position 1.0/len 34==34/posthoc==baseline 逐位/greedy_win 1/1、复现自检 quick 降级结构性验证（num_tokens=64 不可比，如实注记）、VRAM 峰值 4112MB（16GB 大裕量）、H1-H4 quick 全 supported（非定案）；子代理 ses_ec253e97fffexNk4LHtYUM1Cmc |
| T004 | 工件单次定案：reps=30 预承诺运行，`results/optimization/ar022_speculative_quality_bench.json`（H1-H4，判定式全文冻结入 meta） | T003 | passing | 2026-10-15 完成（单次无重跑）：**H1-H4 全 supported**——H1 三面全过（位置级合并 4/4=1.0>0.5；块级 1.0>batched 0.0；posthoc 0.369661/0.570341 逐位==baseline gap=0.0 > batched 0.289692/0.106418）；H2 supported（greedy p50 440.9ms vs batched 1300.6ms，Δ=+859.7ms 加速侧通过，σ=808.4）；H3 supported（长度 34/66 ==baseline 34/66，batched 160/77 早停完全恢复）；H4 supported（MFF_C 捕获/重放>0、三模式 canary 0、账本界内、AR017 中性）；复现自检 strict PASS（batched 重测逐位==AR021 先验 0.0/160/0.289692/5e-06 + 0.0/77/0.106418/0.0，无 reproduction_anomaly）；greedy_win_rate 1.0（2/2）；speedups：greedy_vs_baseline **7.02×**/greedy_vs_batched 2.95×/batched_vs_baseline 2.381（与 AR021 2.3795 跨工件一致）；VRAM 峰值 baseline 5413/batched 5840/greedy 界内（16GB 安全）；工件 53131 字节（≈51.9 KiB，< 96KB gitee 上传限——T009 单段可推） |
| T005 | 质量门：全量 pytest / mypy / pyflakes / 100 列（black）/ demo 锚点 | T004 | passing | 2026-10-15 完成：951 passed/4 skipped/0 failed（2 lm_eval 环境缺失先在——stash 基线证实，deselect 排除留档；skip=1 CUPTI+3 evalplus-unix）；pyflakes 零；mypy 67 文件 clean；black 改动 4 文件 clean（全仓 60 文件遗留态基线同现——纪律仅新增行 ≤100）；demo 三锚点 85.5%/2.35e-03/93.75% 全中 |
| T006 | 文档收口五件：GUIDE 第二十九部分 / CHANGELOG / AGENTS.md 新条目（贪心锚定候选契约）/ RERUN_CHECKLIST / README 全量中文重写 | T005 | passing | 2026-10-15 完成：AGENTS **#44**（贪心锚定候选契约 8 面：行 0=topk 首位/N 语义/可落败/ValueError/探针/画布一致区前提/YAML 面/实证面）；GUIDE 第二十九部分 +44 行；CHANGELOG AR021 上方插条；RERUN 资源表+第 12 步；README 1039→1182 行全量重写（核心创新表第 12 行+机理深潜第 11 节/四模式头条表/951 计数/限制路线图重写）；42 项数字溯源逐项工件核验（含 p50 3096.8/3205.4/1300.6/440.9、逐 prompt greedy 379.4/501.7、VRAM 5413.0/5413.1/5840.1/5672.0、稳定比三模式、H4 池窗重放 30/30）；未经工件支撑的时长表述已清除；子代理 ses_ec23011a3ffecs6FORkg61Y5UH |
| T007 | review 门控（只读子代理：spec/design 合规 + 代码审查），全部问题（含 Minor）修复 | T006 | passing | 2026-10-15 完成：**PASS**（0 Major/5 Minor/3 Nit，审查代理 ses_ec21b32f6ffectbcOLWW0t2JjF）；A-F 六维度全 VERIFIED（srs FR1-FR5/design §4.2.2 五插入点/代码 #10#37#41-43/UT 断言面/39 项数字逐位核对/运维一致性）；mutation 纸面复核 5/5 成立；**全部 8 项当轮修入**：Minor-1 GUIDE UT 编号映射修正（2205=停止链/2206=校验+YAML/2207=N=1，补 mask-token/N=1 缺失面）；Minor-2 测试基线口径统一 951/4/0（README/CHANGELOG/GUIDE/tasks 四处，环境注记保留；skip=1 CUPTI+3 evalplus-unix——ST 复跑裁定）；Minor-3 工件字节数 53131（≈51.9 KiB<96KB 单段可推）；Minor-4 导入 30 符号+`_ar021` 别名；Minor-5 测试 1289 行；Nit-1 design §7 追记导入清单差异（指示性枚举非逐字契约）；Nit-2 UT-2204 补 property 不可赋值断言（AttributeError，8 passed 复验）；Nit-3 quick 工件不入库（git status 确认 untracked，T009 排除） |
| T008 | ST 验收：st-cases.md 生成 + 执行（正常/边界/异常/回归四类），一次成 Go | T007 | passing | 2026-10-15 完成：**Go 一次成**（18/18 全 PASS——N1-N6 正常/B1-B5 边界/E1-E3 异常/R1-R4 回归；工件复算 109 项独立断言防自证全 PASS；需求覆盖 6/6；零产品缺陷）；st-cases.md 409 行；子代理 ses_ec2034a9fffejLveu7BBJ1OL1z；主代理后置裁定：skip 口径定案 951/4/0（"3 skipped"为误记，4 skip=1 CUPTI+3 evalplus-unix，四处文档已修正） |
| T009 | 归档 `specs/archive/AR022-speculative-quality-alignment/`，本地 commit + gitee 推送 + 双端树验证 | T008 | passing | 2026-10-15 完成：本地主体 commit `b474363`（14 文件 +6204/−47，quick 工件排除）+ T009 标记 commit；gitee **七段快进传输链**（`72edfa011`→`5e3185272`→`44e5baf73`→`37828ae25`→`f6a702fe3`→`633b7b17a`→`7de5afe2a`，每段 thin 包 4.3-31.8KB 全部 <96KB 代理限，一次性 read-tree 基树 4c3a665d 后逐批累加——每段重读基树会使累加失效，AR021 记录补强）+ T009 标记段；双端树验证 PASS（最终树 3e301f53 与本地 b474363 一致——永久 commit id 分歧预期内）；诊断注记：双端链完全不相交（增量对象 1120=整仓 3.93MB），直接推不可行，分段链为唯一通路 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-15 会话（需求阶段 + req 门控）

- 需求阶段完成（子代理 ses_ec2de1cd1ffew7qtVOoHyG6aoy）：领域知识（AR021 归档四件+工件质量分面+`_speculative_small_block` 现实现+UT-2108~2110 冻结面）→ srs.md（385 行 rev 2）+ tasks.md（T001-T009）；量化依据 17 项逐项核对工件字段路径
- 关键定案：N 语义=候选总数 N 不变/贪心占 0 号位/随机行 j=1..N-1（同形状 A/B 构造性隔离速度变量）；H1(a) 主门=位置级 0.5（构造级下界+经验面诚实陈述）；四模式基准（新增 folded-batched-greedy 对齐模式）

## 阶段门控记录

### 2026-10-15 req 门控记录（第 1 轮）

- 门控结果：**PASS**（0 Major / 1 Minor / 3 Nit）
- 审查项数：G1-G8 八维度（完整性/准确性/可满足性/冻结面/H 假设/范围耦合/NFR/tasks）——全 PASS
- 数字复核：srs 引用 vs AR021 工件 17 项逐项相等（2.3795/p50 三模式/exact-match 0.0×2/置信度四值/长度 160/77 vs 34/66/Δ/σ/threshold/temperature/VRAM/前向数 31/27/seed/N/top_k/943）
- 问题与修复（全部当轮修入 srs rev 2，纯文档级，免复审）：
  - Minor-1：H1(a) 位置级一致率比对窗口未定案 → 冻结口径入 H1(a) 判定式与术语表（画布绝对位置、逐 prompt `min(len_aligned, len_baseline)` 范围内两模式均生成位集合、小块划分以对齐模式为准、逐位比对、两 prompt 合并聚合）
  - Nit-1：docstring 行号引用补 `:33-34`（NVlabs URL）
  - Nit-2：'host 侧确定性构建' → '确定性构建、零设备回读、无 generator 消费'
  - Nit-3：§2 补 FR 间耦合依据段（AR021 srs:80-83 先例式）
- 审查代理：ses_ec2c94f7affe2wAHCwfWzfYuX0

### 2026-10-15 design 阶段 + design 门控记录（第 1 轮）

- design 阶段完成（子代理 ses_ec2bc1228ffe7tPQb9f3u6nM0）：`design.md`（712 行，七节结构对齐 AR021 基准）；5 组方案对比定案（贪心行算子=topk 首位/观测面=实例级计数器+property/配置双面=FastDLLMSamplerConfig 消费+ActFoldConfig YAML 面/基准脚本=导入式复用/MFF_C 同构独立）；UT-2203 落败机理 `_PosteriorTrapFixture`（零混合层+正交嵌入+右移表头）；tasks.md T001/T002/T003 备注同步细化
- 门控结果：**PASS**（0 Major / 1 Minor / 4 Nit）；审查代理 ses_ec2981b3dffe4ge2Rr72aR7Yaj；A-H 八维度全 VERIFIED（FR 追溯/锚点逐行核验/机理诚实/接口完备/UT 可构造/边界风险/AGENTS 纪律/tasks 一致性）；工件先验数字逐位复核相等
- 问题与修复（全部当轮修入 design.md，纯文档级，免复审）：
  - Minor-1：'行 1..N−1 两模式逐位相同'无条件陈述过强 → 三处（§2.1 ④/§4.2.2(2)/§6.1 UT-2202）补画布一致前提（首个投机小块内构造成立；分叉后失效；A/B 指构建算子对照面非全程轨迹等价；UT-2202 fixture 钉死 num_tokens ≤ block_size 单小块）
  - Nit-1：导入耦合注记补第三条（ar021 模块级 `apply_hf_env()` :90 + 重型导入，幂等良性）
  - Nit-2：`_ctx_mask_map` 措辞修正（本地整函数副本 ~8 行，实质差异仅模式名元组一行）
  - Nit-3：`_position_match` '两行均真实生成 token' 补构造依据（baseline 每迭代至少 unmask 1 位 ⇒ max_iters=2·S 内 mask 必清空；候选全填仍掩码位；stop 尾被 decode_final 修剪；承 AR021 同族假设）
  - Nit-4：blocked 作用域明示（复现不符 ⇒ H1-H4 全部四 verdict 置 blocked + 理由：同环境确定性假设整体失效）+ 可满足性论证补同池状态前提（warmup 后 replay 态）
- 已知遗留裁定：ActFoldConfig 键无生产线程化消费点=可接受（srs FR3 验收冻结唯一途径+诚实注记）；blocked 载体=与 srs 一致

### 2026-10-15 T007 review 门控记录

- 门控结果：**PASS**（0 Major / 5 Minor / 3 Nit）；审查代理 ses_ec21b32f6ffectbcOLWW0t2JjF；A-F 六维度（srs 合规/design 合规/代码质量/测试质量/工件文档数字 39 项逐位/运维一致性）全 VERIFIED；mutation 5 个纸面复核全部成立（M2 波及冻结面 UT-2109c=真跨面 mutant）；唯一覆盖缺口=property 不可赋值（当轮 UT-2204 硬化补齐）
- 全部 8 项当轮修入（Minor-1 GUIDE UT 映射/Minor-2 基线口径 951/4/0 四处/Minor-3 工件 53131 字节/Minor-4 导入 30 符号/Minor-5 测试 1289 行/Nit-1 design §7 导入清单差异追记/Nit-2 UT-2204 AttributeError 断言/Nit-3 quick 不入库确认），修后 UT-2204 套件 8 passed 复验

### 2026-10-15 T008 ST 验收记录

- 总判定：**Go 一次成**（18/18 用例全 PASS：正常 6/边界 5/异常 3/回归 4；st-cases.md 409 行）；工件级 H1-H4/质量面/greedy_win/复现自检共 109 项独立断言（防自证复算）全 PASS；需求覆盖 6/6
- 主代理后置裁定：测试基线 skip 口径定案 **951 passed/4 skipped/0 failed**（review 报告"3 skipped"为计数笔误，本人 canonical 命令复跑两次 + `-rs` 枚举：4 skip = 1 CUPTI（LIBKINETO_NOCUPTI）+ 3 evalplus-unix（resource 模块）；README/CHANGELOG/GUIDE/tasks 四处已统一修正）
- 执行注记（ST 侧留档）：跨工件噪声带容差 0.063%（2.381 vs 2.3795）；E1 live 脚本 7 断言；零产品缺陷

### 2026-10-15 T009 归档与推送记录（gitee 运维）

- 本地：主体 commit `b474363`（AR022 全交付 14 文件 +6204/−47）+ 本 T009 标记 commit
- gitee：七段快进传输链 `72edfa011`（生产+清单+tasks）→`5e3185272`（srs+st-cases+工件）→`44e5baf73`（测试+基准脚本）→`37828ae25`（design+AGENTS）→`f6a702fe3`（README）→`633b7b17a`（GUIDE）→`7de5afe2a`（CHANGELOG，终树 3e301f53==b474363 树）+ T009 标记段（树==本地标记树）
- 每段 thin 包实测 4.3/31.8/30.6/22.7/6.0/5.6/4.3 KB——全部远低于 ≈96KB 华为 HIS 代理 403 限；段链构造：临时 index `GIT_INDEX_FILE` + **一次性** `read-tree 4c3a665d`（gitee 基树）+ 逐段 `update-index --cacheinfo` 累加 + `write-tree` + `commit-tree -p 前段`（**教训：每段重读基树会使累加失效**——首次构造终树不等，改为单次 read-tree 后逐批累加即闭合）
- 双端树验证：本地 tip 树 == gitee tip 树（PASS）；永久 commit id 分歧预期内（AR021 先例）
- 环境注记：双端 git 链完全不相交（`rev-list b474363 --not b72f0e0` = 1120 对象=整仓，thin 直推 3.93MB）——分段链是唯一可行通路；PS 管道 CR 污染 revs 输入，pack 估测走 python subprocess（二进制安全）
