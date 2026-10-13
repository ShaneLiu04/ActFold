# [AR016] 任务清单 — 多候选验证批量 CUDA graph 捕获 + P1-7 墙钟实证

> 状态图例：☐ 待办 / ◐ 进行中 / ✅ 完成 / ✗ 取消

| # | 任务 | 状态 | 产出与验收 |
|---|------|------|-----------|
| T001 | Red：新测试 `tests/test_batched_graph_forward.py` UT-1501~1509 先于生产失败 | ✅ | 925 行 9 测试函数；8 红（capture spy 0/AttributeError graph_runner_batched/告警未现）+ 1 绿（UT-1502 eager 发布契约本就完备，预期分类）；回归 36 passed 零改动；pyflakes 零/100 列零 |
| T002 | Green：`architecture_utils.py` 批量 graph 分发 + 逐分支发布；`cuda_graph.py` 目标零改动 | ✅ | 5 新属性 + 分发块 + `_graph_forward_batched_or_none`/`_parent_cache_complete_batched`/`_publish_graph_children`；**UT-1501~1509 一次全绿（9/9）**；`cuda_graph.py` 零改动（git status 证明）；mutation ①-⑤ 全红（①UT-1501/②UT-1509/③UT-1502/④UT-1505/⑤UT-1508）后 SHA256 逐字节还原（42906736…99993 五次一致）；还原后 45 passed 净 |
| T003 | 基准脚本 `scripts/ar016_batched_bench.py` + 执行 + 工件 | ✅ | 四路径×6 配置 round-robin 交错；TimingStats n=10；工件字段全（运行日期/CUPTI null+注记/交错声明/复现命令/假设声明与判定分离）；**H1 supported（批量 eager 比串行快 2.0-5.6×）/ H2 inconclusive（4/6 supported，N=8 时优势收窄）/ H3 supported（6/6 argmax 精确一致 + folded 三路径 allclose 1e-4）/ P1-7 inconclusive（方向全正 1.37-1.77×，3/6 过 2σ）**；graph 6/6 捕获+预算通过；全量运行 19s（预算 <10min）；两处脚本 bug 修复（branch_id=None 必传 / serial 返回 list 未 cat / runner 探针时机）后重跑 |
| T004 | 全量回归 + demo 锚点 | ✅ | **838 passed（829+9）**/3 skipped/3 deselected 零回归（既有测试零改动）；demo 三指标精确一致（85.5%/2.35e-03/93.75%） |
| T005 | 文档收口（FR6） | ✅ | 指南 P1-7 行 ✅ AR016 第二档追注 + **第二十三部分**（AR 记录表 + 实测墙钟表 6 配置×4 路径 + H1/H2/H3/P1-7 判定 + 语义要点 5 条）；CHANGELOG AR016 节置 Unreleased 最前（Added/实验结果小节）；AGENTS #37 扩注（batch-agnostic runner + 重放前再校验缘由）/#41 扩注（批量 graph 分发 + graph 层不吞 eager 契约 + eager/graph 门路径区分） |
| T006 | 质量门 | ✅ | mypy clean（65 文件）/ pyflakes 零（actfold+tests+demo+脚本）/ 100 列零 / 全量 838 passed / demo 锚点精确一致 |
| T007 | review 门控：只读子代理审查 srs/design/实现/测试/基准工件合规 | ✅ | 首轮 FAIL（3 Major 全为文档数字转录/遗留收口——生产/测试/工件本体全合规，838/mypy/demo 子代理复验通过）→ 修复 → 复核 PASS（逐值复算验证；1 Minor 豁免记录、1 Minor 撤回为审查方编码误报）；门控记录见下 |
| T008 | ST 验收：st-cases.md 场景用例一次成 Go | ✅ | **7/7 Go（一次成）**：ST-01 端到端分发对拍/ST-02 退化-恢复/ST-03 串行+AR015 零干扰（36 passed）/ST-04 UT 9 passed/ST-05 工件逐字段（graph 6/6，verdicts 与文档一致）/ST-06 文档一致性（执行中修正 1 格墙钟表舍入 12.52→12.51——工件精确值 vs 控制台转录，review 教训再应用）/ST-07 838 passed+demo 锚点+静态检查 |
| T009 | 归档 + git commit + gitee 推送 + 树验证 | ✅ | `specs/archive/AR016-batched-graph-verification/`（4 文件整目录）；本地 commit `5ee8342`；commit-tree `3fa29a3` -p `43f81d8` 推 gitee；双端树一致 `9c1b444ab730ea5811f9963374682936dfe5d0ec` |

## 门控记录

### review 门控
- 审查子代理：general（ses_ecb131321ffeGsl4h9qHPzL7hS，首轮 + 复核同代理）
- 首轮结论：**FAIL**——**生产代码/测试/基准脚本/工件本体全部合规**（A1-A7/B1-B9/C1-C2/C5/D1-D3/E1-E3/F1 全属实，838 passed/mypy/pyflakes/demo 锚点子代理本机复验通过），3 Major 全为文档收口：**Major-1** P1-7 统计数字转录错误（"4/6 过 2σ"实为 **3/6**——N=4 两配置+N=8/T=128；"1.93×"实为 **1.77×**，与同部分墙钟表自相矛盾——系与 H2 supported 计数 4/6 混淆）→ 指南 :180/第二十三部分/CHANGELOG 三处改 "1.37-1.77×、3/6"；**Major-2** H1 下界转录错误（"2.1-5.6×"实为 **2.0-5.6×**——N=2/T=512 实测 1.99×）→ 两处改；**Major-3** FR6"AR015 遗留清单收口"未落实（第二十二部分 :520 "graph 批量捕获为后续档"过时且与第二十三部分矛盾）→ 加 ✅ AR016 注记
- Minor-1~3：①UT-1509 夹具偏离（ChunkedActivationCache——ActivationCache.put 复制入私有缓冲使跳 clone 不可观测）未回写 design → design §3 勘误注已补；②工件 meta 缺 GPU 驱动版本（沿 ar002 先例，**记录豁免**：与先例一致，下次重跑补记）；③测试文件 :196 超 100 列——**审查误报**（实测 73 字符，全文 python len 零超限；其工具编码差异所致）
- 数字转录自检教训：写文档数字前先用工件逐配置复算（本次 3 处错均为主代理心算/记忆转录，review 逐值核对拦下）
- 复核（第二轮）：**PASS**——3 Major 逐值复算验证（P1-7 范围 1.3651-1.7747→"1.37-1.77×"、supported 3/6；H1 范围 1.988-5.623→"2.0-5.6×"）；指南/CHANGELOG/工件三方数字一致；Minor-3 撤回（PS Get-Content 编码字符虚增，Python len 全文最大 91 字符零超限）；信息性瑕疵（第二十三部分要点 (5) `**` 后多余空格）已顺手修
- **最终结论：PASS（可进入 T008 ST）**

### design 门控
- 审查子代理：general（ses_ecb75eac9ffeuU7PzrtUZNRQV8，首轮 + 复核同代理）
- 首轮结论：**FAIL**——事实核查 20 条中 16 条属实（架构方案本体获认可），4 Major：**M-1** UT-1508(b) 覆写场景不可达（串行调用存 "p2" 名下不覆写 "p"，须同 id 覆写）→ 改 `mff(tokens[N,T], branch_id="p")`；**M-2** UT 集合与 srs FR4 编号错位 + 预算路径零覆盖（srs UT-1503 共存/UT-1505 预算被 design 挪作他用）→ UT 表按 srs 编号逐条重写 + 补 argmax 断言与 embedding 1e-6 容差；**M-3** "mask 逐位相等"断言违反 srs（FR4 明文 mean_stable_ratio allclose 1e-6 + 勘误条款，ULP 层间传导使深层逐位相等不可绿）→ 改统计量容差、mask 相等降为信息记录；**M-4** 基准假设标签与 FR5 互换（H1/H2/H3 回正为 srs 定义，P1-7 graph-vs-baseline 改独立 `p1_7_verdict` 字段）
- Minor m-1~6 全修：H=256 对 srs "H=32" 的偏离声明（信噪比理由 + 显存复算 <200MB，结论仅被加强）/补 `--quick` + <10 分钟运行预算/补 AR015 遗留清单收口项/`python -m scripts.ar016_batched_bench` 去 .py 后缀/补 FR5 负载构造（中段分歧+独立 suffix、S 步跨步同形）/FR3 广播预填契约等价性论证（expand 与 copy_ 广播逐行同源）
- mutation 计划对齐：srs UT-1509 ①-④（删分发/跳 clone/跳某分支/删预算校验）+ Major-1 专属 ⑤（删重放前再校验 → UT-1508(b) 红）
- 复核（第二轮）：**FAIL**——**Major-N1**（M-1 修复暴露的更深代码事实，门控子代理实测）：`ActivationCache.put` 对同 key 批量增大在 `buffer[:seq].copy_()`（activation_cache.py:154）抛 RuntimeError，UT-1508(b) 同 id 覆写场景不可达（ChunkedActivationCache 整 chunk 替换语义 chunked_cache.py:133-135 下合法）→ 修复：UT-1508(b) 夹具改用 `ChunkedActivationCache`（#22 可插拔公共 API）+ §2.3 步骤 8 补 cache 变体事实（chunked 静默错值面=Major-1 洞真实生产面 / ActivationCache 异常面，步骤 8 两种 cache 下均必要）；Minor-N1~N3 顺手修：夹具 N 统一 4 / UT-1503 补批量 graph 各行 vs 串行 graph 逐行直接对拍（闭合 m-6 传递性松弛，FR3 字面）/ §7 风险表加 ActivationCache 反方向静默广播覆写隐患注记（步骤 8 恰好覆盖）
- 复核（第三轮）：**PASS**——Major-N1 修复以"换可达夹具（ChunkedActivationCache）+ 两种 cache 覆盖语义事实显式注明 + 步骤 8 防线两变体均必要论证"收口，门控子代理本机实证全链路可达（覆写 [4,T,H] 成功 → 拒绝 → eager RuntimeError → 零污染 → 恢复 [1,T,H] 后重放成功 → mutation ⑤ 同形拷贝静默错值红面成立）；Minor-N1~N3 闭环无回退；两条信息性观察（UT-1503 跨 runner 1e-4 容差按 FR4 勘误条款兜底 / UT-1508(a) 随 chunked 夹具可达性无影响）不立项
- **最终结论：PASS（可进入 T001 Red）**

### req 门控
- 审查子代理：general（ses_ecba1a712ffepZubytVwMvDrxY，首轮 + 复核同代理）
- 首轮结论：**FAIL**——**Major-1** 重放期 parent 批量再校验缺失：捕获后 parent 项被串行路径覆写为 [N,T,H] 时 `copy_` 静默广播成功（串行 runner 因形状不匹配必炸而批量 runner 形状恰好匹配），每分支对不同父行折叠、预算校验（计数自洽）拦不住——FR2"parent 批量 != 1 → 静默 eager"按原规格不可实施 → 修复：每次重放前宿主侧 parent 批量/seq 再校验（零 D2H 同步）+ UT-1508(b) 覆写场景对拍观测面
- Minor-1~6 全修：①FR2 降级标志语义精确化（单一布尔 `_graph_degraded_warned` 覆盖全家族、跨原因抑制系串行先例）②CPU 回退钉死"一次性告警"消歧 ③attention_mask 对象不匹配补入退化矩阵 + UT-1506 ④FR5 补运行日期/每重复四路径轮转交错（RERUN_CHECKLIST §4.2/§6）/计时源精确化（perf_counter 而非 CUDA-event `time_forward`——后者排除宿主侧 prefill/读回/发布成本，净收益恰含这些项）/统计量复用点（scripts/algo_experiments.py:196/:216）⑤UT-1507 一次性失败机制（side_effect 单元素或失败即还原，持续类级 patch 会连带炸串行捕获）⑥术语表 parent_static [L+1,...] 修正 + 风险表双 runner graph 池共存显式命名
- 10 项事实核查全部属实（:486 批量旁路 / copy_ 广播 / capture 体 batch 无关 / 容量语义 / 1e-4 容差先例 :396 / 生长循环不适格 / profiler 下游链路 :337-370 / ar002 先例与本 srs 纪律超标 / UT-1501+ 无冲突 / 829 基线活体复验）
- 复核（第二轮）：**FAIL**——**Major-R1**（首轮修复引入，审查代理如实披露）：parent 批量 != 1 的"静默批量 eager 回退（成功）"与不可修改的 AR015 eager 契约矛盾（folded_transformer.py:476-484 对 parent 批量 != 1 是硬 RuntimeError，raise 先于任何 store）→ 修复：终点行为拆子例——seq 不匹配 → 静默 eager 成功；parent 批量 != 1 → graph 层**静默拒绝**（状态零扰动、graph 侧零 cache/profiler 写入）+ 调用方收到 eager #41 RuntimeError（graph 层透明不吞不改）；UT-1506/1508 断言改 pytest.raises；UT-1508(b) 观测面方向反转（mutation 删再校验 → 不抛错 + cache 污染 → 对拍必红）；另闭非阻塞 srs:22 背景简写 [L(+1),…]
- 复核（第三轮）：**PASS**——6 处修复逐项 OK，捕获时/重放前双路径终点一致，全文无"parent 批量 != 1 → 静默成功"残留，章节交叉引用闭环，无新矛盾
- **最终结论：PASS（可进入 design）**
