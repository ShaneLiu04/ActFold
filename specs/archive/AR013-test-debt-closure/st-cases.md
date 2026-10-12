# [AR013] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR013 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-12 |

## 测试用例列表

### ST-001：`make_child` 纯函数钉

**关联需求：** srs.md §3.1（FR1）
**测试类型：** 正常路径 + 边角
**优先级：** High

**期望结果：** 同种子独立重算逐位相等（Generator→randperm→randint 流镜像）+ 双调用确定 + seed 可区分；零/负翻转早退（clone 非同对象+空 positions）；钳位（15>12 → len==12 互异）；vocab=1；父不可变；float32 dtype 直通

**实际结果：** UT-1201/1202 passed（重算路径零生产 import——review A 项核实）；mutation (i) 删钳位行翻红（RuntimeError shape mismatch 15 vs 12）

**状态：** PASS

---

### ST-002：`fidelity` 纯函数钉

**关联需求：** srs.md §3.2（FR2）
**测试类型：** 正常路径 + 手算
**优先级：** High

**期望结果：** 恒等四键精确（手选 norm 精确可表示张量）；方差地板（isfinite + approx(mse/1e-12)）；键集恰四键；fp16 上转；手算例 mse=0.25/top1=0.5 精确 + rel_mse/cosine approx（第一性原理字面常量 + 显式 unbiased 公式镜像）

**实际结果：** UT-1203/1204 passed（无 `base.var()` 调用——torch 默认漂移不会拖测试同步漂移）；mutation (ii) 地板→var 翻红（ZeroDivisionError——Python float 除零非 inf）

**状态：** PASS

---

### ST-003：`BranchTree` 活 API 直接钉

**关联需求：** srs.md §3.3（FR3）
**测试类型：** 正常路径 + 行为钉
**优先级：** High

**期望结果：** add/get 往返、父子链接、重复防护（:57-58）、孤儿（:55）、`BranchNode.seq_len` 值钉、get_ancestors 双态（未注册 False→[]/True→[id]）、prune 递归+解链+no-op+dangling root_id

**实际结果：** UT-1205~1207 passed；**实测新知**：`prune(root)` 递归清空全树含 sibling（T001 首跑红 1 处为测试构造错误，按现行行为钉 + docstring 标注非应然）；mutation (iii) 删递归行翻红（leaf 残留）

**状态：** PASS

---

### ST-004：死代码移除与移除钉

**关联需求：** srs.md §3.4（FR4）
**测试类型：** 变更审计
**优先级：** High

**期望结果：** branch_tree.py :88-93 恰 -6 行；`not hasattr(BranchTree, "best_accepted_leaf")` + 活 API 面（add/get/get_ancestors/prune）hasattr 全真；删除前 UT-1208 红/删除后绿

**实际结果：** `git diff actfold` 恰 -6 行（review C 项逐字节核对）；UT-1208 删除前红（hasattr True）/删除后绿——移除生效的天然证据；仓级引用仅剩记录性文本（review WARN-2）

**状态：** PASS

---

### ST-005：指南 6.1 五条勘误勾选

**关联需求：** srs.md §3.5（FR5）
**测试类型：** 文档审查
**优先级：** Medium

**期望结果：** :228/:230/:231/:232/:233 全部删除线+✅+勘误+证据；第二十部分回链；CHANGELOG AR013 节；AGENTS 零改动

**实际结果：** 6.1 清单 6/6 终态；勘误数字全对（review A 项逐项实测：31 例/7 例/59 hits/16 例）；行号抽查 3+ 处全中

**状态：** PASS

---

### ST-006：全量质量门

**关联需求：** srs.md §4 NFR
**测试类型：** 回归测试
**优先级：** High

**实际结果：**
- 全量：**807 passed / 3 skipped / 3 deselected**（799 基线 + 8 新增，零失败，39.31s）
- demo：FLOPs reduction 85.5% / MSE 2.35e-03 / stable ratio 93.75% **精确一致**
- mypy clean（65 文件）/ pyflakes 零 / max 列 91 ≤ 100
- 两新文件 8 passed in 5.5s

**状态：** PASS

---

### ST-007：生产改动边界审计

**关联需求：** srs.md §2 Out of Scope（`actfold/` 唯一改动 = branch_tree.py 删 :88-93）
**测试类型：** 变更审计
**优先级：** High

**实际结果：** `git status --porcelain -uall` 恰为：M CHANGELOG.md / M branch_tree.py / M 指南 + untracked specs/changes/AR013（3 文件）+ 两新测试文件；`git diff actfold` 净 -6 行；mutation 还原后两次确认干净（T001 + review C）

**状态：** PASS

---

## 验收结论

**7/7 Go。** srs §3 五项验收全部落实并经 review 五维核查（A spec/B design/C diff/D 运行/E 风险，0 Major 0 Minor）；两道门控（req 首轮 PASS / design FAIL→修复 PASS）的修正全部落文档；mutation 三位点自证非空转绿；指南 6.1 测试债清单 6/6 终态。AR013 可归档。
