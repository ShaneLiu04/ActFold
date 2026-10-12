# [AR013] 任务分解 — 指南 6.1 测试债清单终审收口

基线：main `edb190d`（gitee `624d971`，树 `78ec477`），全量 799 passed / 3 skipped，mypy clean（65 文件）。
约束：`actfold/` 唯一改动 = branch_tree.py 删 :88-93 死方法；UT 编号 1201~；新文件 `tests/test_algo_experiment_helpers.py` + `tests/test_branch_tree.py`。

## 任务

- [x] T001 Red：子代理完成两测试文件 + :88-93 删除（UT-1208 删除前红/后绿=移除生效天然证据；UT-1207 首跑红 1 处为测试构造错误——**实测新知：`prune(root)` 递归清空全树含 sibling**，按现行行为钉+docstring 标注；mutation 三位点翻红机制与 design 勘误后预期一致：RuntimeError/ZeroDivisionError/leaf 残留）
- [x] T002 Green：主代理亲验（8 passed 复跑一致、pyflakes 零、`git diff actfold` 恰 -6 行、import 审计通过——algo_experiments 仅 import 被测函数本体）
- [x] T003 全量回归：**807 passed（799+8）/ 3 skipped / 3 deselected** 零失败；demo 三指标精确一致（85.5%/2.35e-03/93.75%）
- [x] T004 文档：指南 :228/:230/:231/:232/:233 五条勾选+勘误（6.1 清单 6/6 终态）、第二十部分回链、CHANGELOG AR013 节（Removed + Added + Design）
- [x] T005 质量门：mypy clean（65 文件）/ pyflakes 零 / 100 列 / `git diff actfold` 恰 branch_tree.py -6 行
- [x] T006 review 子代理：**PASS**（0 Major + 0 Minor + 3 WARN 不需修复；A-E 五维核查；勘误数字逐项实测全对）
- [x] T007 ST：st-cases.md **一次成 Go（7/7 PASS）**——ST-001~004 映射 FR1~FR4、ST-005 文档、ST-006 质量门（807 passed/demo 精确/mypy/pyflakes/列宽）、ST-007 生产边界审计
- [ ] T008 归档 `specs/archive/AR013-test-debt-closure/` + git commit + commit-tree 推 gitee + 树验证

## 门控记录

### req 门控
- **首轮 PASS**（0 Major + 5 Minor + 3 WARN，只读子代理 12 项事实核查）：
  - Minor-1/2 采纳：TargetMatch 直接用例 8→**7**（:718-805）；test_experiment_stats 17→**16** 用例（逐函数计数无误）
  - Minor-3 采纳：:230 出处链精确化——冻结 **pre-T018** 逐行参考（:80-154），官方对齐是 sampling_utils :158/:166 二级自证（避免与 AR012 "参考=生产逐字拷贝"批判抵触）
  - Minor-4 采纳：get_ancestors 未注册 id 仅 include_self=False 时为空（True 时返回 `[id]`）——照原字面写必失败
  - Minor-5 采纳：profile_summary 排除理由改为 ":232 未点名"（实为纯函数）
  - WARN-1/2/3 采纳：恒等 cosine 用非零张量；dangling root_id 显式钉；:347-379 行号
  - 附核实：fidelity [2,2] 手算例实算全有限（mse=0.25/rel=2.5e11/cos≈0.9449/top1=0.5）；var unbiased 默认实证；best_accepted_leaf 仓级扫描确认零调用；基线 799 passed 本机精确复现（40.27s）

### design 门控
- **首轮 FAIL**（2 Major + 3 Minor + 1 WARN，只读子代理实证级核查）：
  - Major-1：`seq_len` 是 `BranchNode` property 非 `BranchTree` 方法——UT-1208 活 API 面按原字面 `hasattr(BranchTree, "seq_len")` 首跑必红；且 srs §3.3 的 seq_len 值钉在 design 用例表中无承载 → §2.3 面删 seq_len + UT-1205 补 `BranchNode.seq_len` 值钉
  - Major-2：恒等例 `cosine_similarity` 经 norm√→² 往返，**随机张量 ~60%（120/200 实测）得 0.99999994 ≠ 1.0**——"四键精确"断言是抽签 → 恒等例改**手选 norm 精确可表示张量**（norm²=16、√=4、积=dot=16 → bit-exact 1.0），保住"精确"措辞
  - Minor-1：mutation (i) 翻红机制勘误——randperm 切片静默封顶无越界，真正红点是 `randint(15,)` vs 12 索引赋值 **RuntimeError shape mismatch**（mutation 有效，机制描述已更正）
  - Minor-2：mutation (ii) 机制勘误——`.item()` 后 Python float `mse/0.0` 抛 **ZeroDivisionError** 非 inf（mutation 有效）
  - Minor-3：var "独立书写"名义化（与生产同原语同默认）→ UT-1204 改第一性原理**字面常量** + 显式公式镜像 `((base−mean)**2).sum()/(n−1)`（torch 改默认时测试不同步漂移）
  - WARN-1 采纳：rel_mse==1.0 bit-exact 依赖 Welford 顺序运气 → approx 1e-6
  - 附核实：UT-1204 四指标手算全对（cos=5/(2√7)=0.9449112 实证）；-6 行恰为 :88-93；仓内无 BranchTree 子类；mutation (iii) 推演一致；seed 7 vs 123 可区分实证
- **修复后 PASS**（Major/Minor 全修，WARN 采纳）

### review 门控
- **PASS**（0 Major + 0 Minor + 3 WARN，只读子代理 A-E 五维核查）：
  - A spec 符合性：FR1~FR5 验收标准逐条落实（make_child 八项/fidelity 五项/BranchTree 九项/-6 行/五条勘误数字全对——31 例、7 例、59 hits、16 例、807、-6 行、168/149 行全实测命中）
  - B design 符合性：Major-1（活 API 面无 seq_len）/Major-2（手选张量）/Minor-3（显式 var 公式）修复全部落实；mutation 机制与生产 :123/:143/:85-86 逐行推演一致
  - C git diff 逐字节对照：改动面恰为声明集合，无夹带
  - D 运行验证：8 passed / 全量 807 passed / pyflakes 零 / max 列 91 / mypy clean
  - E 回归风险：低——零 flaky（全固定 seed + 确定性张量）、删除非破坏、UT-1208 约束面可接受（复活应被质疑是 D4 设计意图）
  - WARN-1/2/3 不处置：私有 `_nodes` 快照是测试常见做法；记录性引用合法；hasattr 钉约束面即设计意图

### ST 门控
- **一次成 Go（7/7 PASS）**：ST-001~007 全过（详见 st-cases.md）；结论——srs §3 五项验收落实、mutation 自证非空转绿、全量回归与 demo 零漂移、生产边界恰 -6 行、指南 6.1 清单 6/6 终态，AR013 可归档
