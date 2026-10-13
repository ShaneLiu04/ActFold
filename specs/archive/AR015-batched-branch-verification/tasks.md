# [AR015] 任务清单 — folded_generate 多候选批量分支前向（P1-7 第一档）

> 状态图例：☐ 待办 / ◐ 进行中 / ✅ 完成 / ✗ 取消

| # | 任务 | 状态 | 产出与验收 |
|---|------|------|-----------|
| T001 | Red：新测试文件 `tests/test_batched_branch_forward.py` UT-1401~UT-1408（对拍/回退矩阵/前置契约/launch-count/graph 旁路）先于生产代码失败 | ✅ | 717 行（review 门控增补 UT-1409~1412 后现为 960 行）；8 failed 全归因生产未实现（TypeError branch_ids / AttributeError 谓词 / DID NOT RAISE 契约）；串行基准先行验证可用；3 处合理偏离记录在案（UT-1407 len≠batch Red 期伪通过、模型带 split 可发现 FFN 链、cache 分支列举用 put spy+contains） |
| T002 | Green：三生产文件最小实现（FR1 `_forward_batched` / FR2 入口 / FR3 分发+回退）使 UT-1401~1408 全绿 | ✅ | folded_transformer.py +327-2（入口 3 行+`_forward_batched`/`_store_batched`/`_merge_parent_child_batched`，串行体逐位不动）；architecture_utils.py（branch_ids 形参+互斥+graph 门 `and branch_id is not None`+层循环传参）；folded_generation.py（`_batched_forward_supported`+`_run_folded_forward_batched`+分发）；Green 期 2 bug 修复（varlen post-expand 校验误用 stable_mask.shape→按 D4③ 改 [N, prefix_len]；mypy 窄化×2）；**mutation 四变异全验证后逐字节还原（SHA256 三文件 identical）**：①删 expand→UT-1401/1403 红（gate ValueError）②单一 branch_id 存→UT-1401 cache 红③跳 profiler→UT-1401 profile None 红④删分发→UT-1405 探针红（UT-1404 直调不经谓词不红——design §3.3④ 观测面勘误）；mypy 65 文件 clean/pyflakes 零/100 列零违例 |
| T003 | 全量回归 + demo 锚点 | ✅ | 全量 **825 passed（817+8）/ 3 skipped / 3 deselected**（41.6s，既有测试零改动零回归）；demo 三指标精确一致（FLOPs reduction 85.5% / MSE 2.35e-03 / acceptance 93.75%——demo 不经 folded_generate，锚定 FR1/FR2 串行层路径逐位不动） |
| T004 | 文档收口（FR5） | ✅ | 指南 P1-7 行勾选 + 勘误注（行号 :139-143 / engine 无多分支循环 / launch-count 代理 #39）；指南第二十二部分（表格式 AR 记录 + 语义要点 + 后续档遗留）；CHANGELOG AR015 节置 Unreleased 最前（Added/Design 两小节）；AGENTS #41 批量分支契约 pitfall（单复数互斥 / expand 契约 Bp==1 / cache-miss 合法路径 / plain gate+直调 `_merge_stable_divergent_torch`+跳 gather_select / graph 静默旁路 / 路径分档容差 gemv-gemm 1 ULP） |
| T005 | 质量门 | ✅ | 首跑（8 UT）：mypy clean/pyflakes 零/100 列零/全量 825 绿/demo 精确一致；review 修复后复跑（12 UT）：mypy 65 文件 clean、pyflakes 零、100 列零、全量 **829 passed（817+12）/ 3 skipped / 3 deselected**（39.0s）、demo 三指标精确一致（85.5% / 2.35e-03 / 93.75%） |
| T006 | review 门控：只读子代理审查 srs/design/实现/测试合规性 | ✅ | 首轮 **FAIL**（2 Major）：Major-1 FR1 scheduler 验收项零 UT 覆盖→补 **UT-1409**（scheduler 禁用层整批重算+逐分支存、动态 tau 复用、profile 只含折叠层、禁用层 cache 行跨分支互异、CPU 逐位对拍）；Major-2 FR1 全稳三分支（含 AR007 截断）零覆盖→补 **UT-1410/1411/1412**（全稳快路径 `_recompute_all` spy 钉零重算——真前缀子代重算值与父复用值逐位相同，调用计数是唯一可观测面；截断全稳；截断混合 merge 删截断→RuntimeError）；生产代码 git diff 0a66dd9 独立验证全合规（串行体零 diff hunk）。Minor-1~5 全修（测试头 UT-1403 陈旧表述/CHANGELOG gemv 笔误/AGENTS #41 签名精确化/两处 docstring/行数簿记）。**review 增补 mutation 五变异全验证后 SHA256 逐字节还原**（⑤scheduler 分支→UT-1409 红 ⑥全稳 expand→UT-1410/1411 双红 ⑦快路径截断→UT-1411 红（spy）⑧merge 截断→UT-1412 红 ⑨动态 tau→UT-1409 红）；srs §3.4 增补 UT-1409~1412 验收项+mutation ⑤-⑨；**复核 PASS**（同代理 ses_ecc17d2b3ffe：两 Major 确认 mutation 敏感非装饰——tau 钉非空穴 0.9457≠0.95、spy 经 after_parent 钩子规避父前向计数；复核新 Minor-1 helper return 后重复死代码块→已删（975→960 行，12 passed 复跑）；Minor-2 门控记录时序注→PASS 后即准确） |
| T007 | ST 验收：st-cases.md 场景用例一次成 Go | ✅ | 7/7 Go（一次成）：ST-1 层级契约 / ST-2 三向拆分+scheduler / ST-3 入口+graph / ST-4 分发回退 / ST-5 容差分档 / ST-6 文档 / ST-7 质量门；12 UT verbose 复跑全 PASSED；FR1-FR5+NFR 全映射；详见 st-cases.md |
| T008 | 归档 + git commit + gitee 推送 + 树验证 | ◐ | `specs/changes/AR015-*` → `specs/archive/`（含 tasks.md，沿 AR001-014 惯例）；本地 commit；commit-tree -p f481973 推 gitee；双端树一致 |

## 门控记录

### req 门控（已完成）
- 审查子代理：general（ses_ecc81f620ffemlAF77b3jkMY1m，首轮 + 复核同代理）
- 首轮结论：**FAIL**——Major-1 demo 锚点定性错误（demo.py 不调用 folded_generate：suffix_append draft + 直调 folded_model :481/:490 + verify_branch :574，NFR"恒串行"理由系编造；改写为锚定 FR1/FR2 串行层路径）；Major-2 FR1 缺"parent id 非空但缓存 miss"分支（生成循环第一步必然触发——root 缓存空、draft 模式不对 parent 前向；补验收项 + UT-1407 用例）
- Minor-1~5：①"全仓唯一"→限定折叠验证循环（spiffy_baseline.py:77-86 为故意 baseline 对照）②不变量改"同 parent+等长"（flip_ratio=0.1 重采样 suffix 区、random/perturb 全随机——仅长度不变，翻转由 gate 兜底）③branch_id 保持必填、类型放宽 str | None ④logits 消费面补全（:153 assert/:155 全位置 argmax/:245 末位）⑤gather_select 跳过理由改"fetch_flat 按 parent 存储批量计行、恒 None 优雅回退"
- 复核结论：**PASS**（7 项全 RESOLVED；1 处非阻断新瑕疵：FR3 调用示意漏 branch_id=None——已顺手修正）
- 10 项证据核实全部属实（首轮记录）：串行点 :139-143 ✓ / guide :180 引 :114-117 确已陈旧 ✓ / verify_branch 无多分支循环 ✓ / 层级行号 :185-190/:21/:558-579/:269-277/:412-430 ✓ / UT-14xx 无冲突 ✓ / 817 基线一致 ✓

### design 门控（已完成）
- 审查子代理：general（ses_ecc6d69d4ffeHQIOMExw9olu5H，三轮同代理）
- 首轮结论：**FAIL**——Major-1 D4 与 §3.3① 对批量版父 FFN 契约形状互斥（fetch 期 [1,·] vs 期望 [N,·]：一者 UT-1401 永不 Green、一者 mutation ① 不可观测）→ 唯一化三段式契约（fetch 期校验父存储 [1,prefix_len] → expand → post-expand 输出校验）；Minor-1~8：签名不重排位序/全稳补 `_returns_tuple` 守卫/profiler 对拍面改统计量（profiler 不保留 mask，stability_profiler.py:161-179）并回写 srs/私有 import 标注 srs 字面偏离/UT-1403 split 阈值降 ≤T（split_layer.py:272-274 批量 N*T 与串行 T 判定不对称）/探针改 `kwargs.get("branch_ids") is None`（层循环无条件传参）/ChunkedCache 视图存储注明（串行 :293-316 同构先例，chunked_cache.py:123）/profiler 单例互异 id+即时捕获——同批修复
- 复核二轮结论：**FAIL**——新 Major：D4③ varlen 半边期望形状误写 [N,T]（校验先于 cat，实际 [N,T_p]）→ 统一 [N, prefix_len]（等长退化 stable_mask.shape，与串行范本 :476-482 逐字一致）；srs UT-1406 措辞勘误 + srs 状态行补记 design 轮
- 三轮结论：**PASS**（三项全 RESOLVED；2 项非阻断簿记——design 状态行补二轮记录、§3.3① 指针 D1→§1——已顺手修正）
- 技术断言核实记录（首轮 a-h 全成立）：keyword-only 兼容（调用全关键字传参）/全稳 fetch 同位/varlen cat 同构/put 侧 copy_ 安全/graph 门收紧零变化/谓词④由 tokens.clone() 派生保证/logits 视图满足 acceptance.py:73-85 形状契约/Split 层 flat-index row-major 展平天然泛化（split_layer.py:94-103/:222/:249）
- 结论：设计可进入 develop（T001 Red）
