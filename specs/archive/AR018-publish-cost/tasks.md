# [AR018] 任务清单 — 发布路径宿主侧成本压缩：批量稳定度归约 + 快照发布

> 状态图例：☐ 待办 / ◐ 进行中 / ✅ 完成 / ✗ 取消

| # | 任务 | 状态 | 产出与验收 |
|---|------|------|-----------|
| T001 | Red：新测试 `tests/test_publish_cost.py` UT-1701~1709 先于生产失败 | ✅ | 2026-10-14 完成：9 函数（1701 拆 CPU/CUDA 子块）实测 5 红（1701×2 AttributeError / 1702 bool-sum 8 / 1704 clone 0·总 8 / 1707 常态分支 8）+ 4 绿（1703/1705/1706/1708 双侧不变量，与 design §5 预期一致）；pyflakes 零/100 列零/收集无错；回归零扰动（全量 846 passed 3 skipped） |
| T002 | Green：FR1 预求和入口 + FR2 快照发布；mutation ①-⑤ 全红后 SHA256 还原 | ✅ | 2026-10-14 完成：两文件按 design §2/§3 草案照抄实现；UT-1701~1708 一次全绿（9/9）；mutation ①删批量归约→UT-1702 红/②删整块快照回退逐行 clone→UT-1704 红/③快照别名 child_buf→UT-1705 红/④删 debug 回退→UT-1707 红/⑤record_precomputed stable_sum 跨调用 scratch 复用+覆写→UT-1701 红，五变异逐个确认红后 SHA256 逐字节还原（0FCCA1F0…/0876A68B… 双文件比对一致）；cuda_graph.py 零改动（git status 证明）；全量 855 passed（846+新增 9）零回归 |
| T003 | 基准脚本 `scripts/ar018_publish_bench.py` + 执行 + 工件 | ✅ | 2026-10-14 完成：三路径×4 几何 reps=30 预承诺单次运行（<10 分钟，--quick 冒烟先过）；加速比 adaptive_vs_baseline 2.4034/2.6743/2.6607/2.9466×（AR017 锚点 1.5502/1.6839/2.2993/2.4609×）；发布占比 45.3%/50.9%/15.7%/17.6%（探针 60.7%/71.2%/29.7%/31.5%）；H1 整体 inconclusive（(256,8,4) 欠功效 3/4 supported）、H2/H3/H4 supported；锚点可用（AR017 run_date 2026-10-13 嵌入）；CUPTI null+注记；峰值显存 332.8/528.7/2472.5/3521.7MB；工件 `results/optimization/ar018_publish_bench.json` |
| T004 | 全量回归 + demo 锚点 | ✅ | 2026-10-14 完成：855 passed（846+新增 9）/3 skipped 零回归（既有测试零改动）；demo 三指标精确一致（85.5%/2.35e-03/93.75%） |
| T005 | 文档收口（FR5） | ✅ | 2026-10-14 完成：指南 P1-7 行第四档追注 + 第二十五部分（AR 记录表 + 实测墙钟/跨工件对照表 + 相位分解表 + 语义要点六条 + 遗留）+ Part 24 遗留收口注记（宿主侧分解 ✅ AR018 + prefill/读回 <5% 记录）；CHANGELOG AR018 节置 Unreleased 最前；AGENTS #37 扩注（快照 fresh 契约/put 自带拷贝下逐行 clone 消除/双入口 debug 回退/发布零 host 同步）；数字全部从工件逐值复算 |
| T006 | 质量门 | ✅ | 2026-10-14 完成：mypy clean（66 文件）/ pyflakes 零 / 100 列零（7 触碰文件逐行复核）/ 全量 855 passed / demo 锚点一致 |
| T007 | review 门控：只读子代理审查 srs/design/实现/测试/工件合规 | ✅ | 2026-10-14 完成：PASS（六清单属实）；4 非阻断 Minor——M-1/M-2 文档数字转写已修、M-3 脚本命令串恒含 --reps（工件不手改注记）、M-4 slow 用例环境注记；记录见下 |
| T008 | ST 验收：st-cases.md 场景用例 | ✅ | 2026-10-14 完成：16 用例 16 PASS 一次成 Go（正常 8/边界 3/异常 1/回归 4）；需求覆盖 6/6（FR1-FR5+NFR）；工件结构 13/13 检查过、H2/H3 复算 4/4、全量新鲜复跑 855 passed；3 条 Minor 注记级已记录；st-cases.md 已写 |
| T009 | 归档 + git commit + gitee 推送 + 树验证 | ◐ | 两步交付先例：主体 commit → T009 标记 commit → 各自 commit-tree 推 gitee → 双端树验证 |

## 门控记录

> 各阶段门控审查后追加（req/design/review/ST）。

### req 门控
- 审查子代理：general（ses_ec812b2daffeHi3mXUhNyPHjr7，首轮 + 复核同代理）
- 首轮结论：**FAIL**——事实核查 1a-1f/2a/2e 全部属实（发布双循环/静态缓冲形状/record 消费面 0-dim 等价/put 自带拷贝/UT-1502·1509 验收面继续成立/AR017 工件对照前提字段齐全、探针数字逐值复算吻合），2 Major：**Major-1** UT-1709 mutation ③ 红测映射错误（mutate stable_sum 只影响 profiler 统计应由 UT-1701 捕获，UT-1705 的 by-reference cache 断言与其不相交不会红）→ 修复：mutation 拆为 ③快照非 fresh（别名 child_buf）→UT-1705 红 + ⑤sums 跨步复用覆写→UT-1701 红（srs/tasks 同步五变异）；**Major-2** revalidate_host 占比写 <0.3% 与探针工件矛盾（实测 0.33-1.40%）→ 改 0.3-1.4%
- Minor-1~8 全修：①~40µs/对→~85µs/对（32 对口径）②"归约 128→1"→"L×N→1（32/64/64/128）"③FR4 相位清单补 revalidate_host（六相）④H3 措辞区分本 AR 探针与 AR017 工件⑤FR1 显式 profiler.enabled 门控批量归约⑥风险表补瞬态显存峰值（+268MB）与 by-reference 钉住粒度（caching allocator 复用缓解）⑦UT-1702 补 .item/.tolist/.cpu spy 恰 0 次⑧UT-1706 措辞精确化（UT-1602=收缩后输出 vs eager 对拍）
- 复核（第二轮，单点）：**PASS**——两 Major 落实且未引入互斥（③/⑤ 红测映射链验证：③别名→mock 存储污染→UT-1705 torch.equal 红；⑤覆写→stable_sum 漂移→UT-1701 逐位对拍红），8 Minor 全收口，无新漂移；带入 2 个新 Minor（FR4 缺峰值显存观测字段/"swap 瞬间"笔误与 ChunkedActivationCache 过度概括）**当场修入**（FR4 验收补 `torch.cuda.max_memory_allocated` 峰值字段；风险行改"快照瞬间"+限定"默认参数下"）
- **最终结论：PASS（可进入 design）**
- **erratum（design 门控 Major-1 触发，回溯修 srs）**：srs §1.1 锚点校验与 FR4/§6 中"phase_sum ≥ 全步（串行化）"的方向性断言被探针工件自身推翻（4 配置中 3 个 phase_sum < 全步——相位间宿主工作遗漏的负向偏差大于 sync 串行化正向偏差）→ 已改中性措辞：两值同入工件、偏差双向注记、不预设方向

### 探针记录（选题依据，req 前置）
- `ar018_hostcost_probe`（temp，2026-10-14）：AR017 同夹具同种子稳态容量，n=30，四配置相位分解——发布占比 60.7%/71.2%/29.7%/31.5%（全步 p50 7.927/13.823/30.688/57.743ms，与 AR017 工件锚点吻合）；发布内 profiler 归约 37-44%；读回 0.06-0.13ms、prefill 0.31-1.55ms 合计 <5%；工件 `Temp\opencode\ar018_hostcost_probe.json`
- **勘误（design 门控 Major-1 发现）**：探针工件相位和与全步的偏差以负向为主（phase_sum 7.390/13.997/30.462/56.064 vs 全步 7.927/13.823/30.688/57.743——(256,8,8) 为唯一正向偏差配置，双向并存的实证）——相位间宿主工作遗漏 > sync 串行化；锚点校验结论不受影响（基于全步值）

### design 门控
- 审查子代理：general（ses_ec7f97d9affe6Es65YmTlkGfhP，首轮 + 复核同代理）
- 首轮结论：**FAIL**——核心机制逐行核验全部成立（record_precomputed 镜像/0-dim 视图等价/三态分派真值表/spy 先例 test_cuda_graph.py:614-628/AR017 工件取数/命名空间包导入），1 Major：**Major-1** `phase_sum_ge_full: true` 固定注记被探针自家数据推翻（4 配置中 3 个 phase_sum < 全步）→ 修复：design §6/§9 与 srs §1.1/FR4/§6 改中性双向偏差注记（两值同入工件，不预设方向）+ req 记录补 erratum
- Minor-1~9 全修：①H2 算符 > → ≥（对齐 srs 不回退语义）②UT-1705 RefCache 驱动方式明确为直接调 `_publish_graph_children`（绕过 revalidate fetch）③UT-1701 补 mean_stable_ratio/all_profiles/get_mean_stable_ratio 聚合面对拍④UT-1706 补 UT-1602/1606 套件承载映射⑤UT-1705 表格行清理（双侧绿结论，删草稿自问自答）⑥launch 账本补 layer-0 embedding put 拷贝 +N 脚注⑦`:meth:`record`` 双反引号笔误 + 时序图 embedding/clone 次序对齐代码⑧§6 复用清单补 `_make_mff`⑨UT-1703 参考实现改为不调 record（cache 对拍与 profiler 无关，避免 GLOBAL 污染）
- 复核（第二轮，单点）：**PASS**——Major-1 与 Minor-1~9 全部落实、未引入互斥；带入 2 个新 Minor（design §2 docstring 收尾引号与首语句挤行；tasks 勘误 phase_sum (256,8,8) 值 13.676→13.997——恰为唯一正向偏差配置）+ 2 Nit（§9 串行化措辞/双侧绿清单漏 1703）**当场修入**（docstring 拆行；13.997 修正；§9 双向偏差措辞；清单补 1703）
- **最终结论：PASS（可进入 T001 Red）**

### review 门控
- 审查子代理：general（ses_ec7a51501ffefxPf1LZxIWa1Wz，只读）
- 结论：**PASS**——六大清单全部属实：实现与 design §2/§3 逐行一致（record 零改动、4 既有调用方零改动、cuda_graph.py 零改动）；测试逐 UT 断言面吻合 design §5（双侧不变量声明/GLOBAL 卫生/逐函数 _cuda）；工件数字全配置重算逐位自洽（加速比/发布占比/相位和/H1-H4 判定复算一致）；文档三表与工件逐值对拍一致；质量门实跑复核相符（855 passed/mypy 66 文件/demo 锚点）；FR1-FR5 验收闭环无遗漏
- 4 条非阻断 Minor 处置：**M-1** 指南 Part 25 H3 探针发布绝对值转写滑差（9.85/9.11/18.19 → 实值舍入 9.84/9.12/18.17）**已修**；**M-2** Part 24 收口注记 prefill/读回区间（0.31-1.56/0.05-0.13 → srs 口径 0.31-1.55/0.06-0.13）**已修**；**M-3** 工件 `meta.command` 未逐字含 `--reps 30`（脚本默认值抑制逻辑所致；实际运行命令即 `--reps 30`，`meta.reps: 30` 在案可复现）——**脚本命令串生成已改为恒含 `--reps`**（未来运行逐字满足 srs FR4）；工件按墙钟纪律不手改，特此注记；**M-4** 环境观察：本机未装 bench 依赖，无 marker 的 `pytest tests/` 有 2 个 slow 用例 `ModuleNotFoundError: lm_eval`（AR017 起既有环境事实，与本 AR 改动无关）；全量数字口径均为 `-m "not slow"`（855 passed/3 skipped 与之精确吻合）——特此注记
- **最终结论：PASS（可进入 ST）**
