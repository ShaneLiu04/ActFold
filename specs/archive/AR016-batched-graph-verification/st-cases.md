# [AR016] ST 验收用例（T008）

> 执行日期：2026-10-13；执行环境：Quadro RTX 5000 16GB / torch 2.5.1+cu121 / Windows WDDM
> 结论：**7/7 Go（一次成）**

| # | 用例 | 验证命令/方法 | 期望（srs 映射） | 结果 |
|---|------|--------------|------------------|------|
| ST-01 | 端到端批量 graph 分发：公开 API 捕获→重放→对拍→发布 | 临时脚本（`_build_model`/`_make_mff` 夹具复用）：parent populate → 批量调用×2（首次捕获、二次重放）vs fresh eager 实例 | 首次+二次 logits 均 allclose(1e-4)+argmax 相等；`graph_runner_batched` 非 None 且形状 [4,16]；逐分支 cache 含 [1,16,H] ffn_out（FR1） | **Go**：capture+replay parity（1e-4+argmax）、逐分支 cache 发布 |
| ST-02 | 形状不匹配退化→恢复重放 | 捕获 [4,16] 后来电 [2,16]：`warnings.catch_warnings` 捕告警；再回 [4,16] | 恰一次 UserWarning 且消息含两个形状；退化结果 == eager；再次同形调用零新告警且重放 logits == eager（FR2） | **Go**：一次性形状告警（两形状在消息中）、eager 对拍、重放完好 |
| ST-03 | `cuda_graph.py` 零改动 + 串行 graph 零干扰 | `pytest tests\test_cuda_graph.py tests\test_batched_branch_forward.py -q`（既有测试零改动全绿） | 串行 graph 契约逐位不动；AR015 批量 eager 契约逐位不动（FR3/NFR 兼容） | **Go**：36 passed 零改动（与 UT 套件合并跑 45 passed） |
| ST-04 | UT 套件 | `pytest tests\test_batched_graph_forward.py -q` | UT-1501~1509 全绿（FR4） | **Go**：9 passed |
| ST-05 | 基准工件核验 | python 断言脚本逐字段读 `results/optimization/ar016_batched_bench.json` | meta 11 字段全（运行日期/CUPTI null+注记/交错声明/复现命令/计时口径）；6 配置×4 路径；graph 6/6 捕获+预算通过；verdicts H1=supported/H2=inconclusive/H3=supported/p1_7=inconclusive（3/6 supported，speedup 1.3651-1.7747）；假设声明与判定分离（FR5） | **Go**：全部断言通过，工件与文档数字一致 |
| ST-06 | 文档一致性（FR6） | python 断言脚本：指南第二十三部分墙钟表 24 个 p50 值 vs 工件逐值核对；P1-7 行 ✅ AR016 注记；第二十二部分遗留收口；CHANGELOG AR016 节置 Unreleased 最前（数字 1.37–1.77×/3/6、2.0–5.6×）；AGENTS #37/#41 扩注在场 | 墙钟表与工件逐值一致（%.2f）；五处文档收口全在（FR6 四验收项） | **Go**：24/24 值一致（执行中修正一格舍入：N=4/T=128 baseline 12.52→12.51，系照控制台 %.3f 转录而非工件精确值——review 教训的再次应用）；五处收口全在 |
| ST-07 | 质量门 NFR | `pytest tests\ -q -m "not slow"` + `demo.py` + mypy/pyflakes（T006 已跑） | 838 passed（829+9）零回归；demo 三指标精确一致（85.5%/2.35e-03/93.75%）；mypy clean/pyflakes 零/100 列 | **Go**：838 passed/3 skipped/3 deselected；demo 三锚点精确一致；静态检查全绿（T006 复验） |

**ST 备注**：ST-06 执行中发现的墙钟表舍入偏差（1 格）已当场修正——这验证了"工件是唯一事实源"纪律的必要性（同 review 门控 Major-1/2 教训：文档数字必须从工件精确重算，不得照控制台打印转录）。
