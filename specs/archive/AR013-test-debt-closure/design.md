# AR013 详细设计 — 指南 6.1 测试债清单终审收口

> srs：`specs/changes/AR013-test-debt-closure/srs.md`（权威契约）
> 创建：2026-10-12；状态：final（design 门控第 1 轮 FAIL→修复 Major-1/2 + Minor-1/2/3 + WARN-1 采纳；复验见 tasks.md）

## 1. 设计总览

方案：**A（三块钉 + 死代码删除 + 清单勘误）**——两个新测试文件
（scripts helper 层 / speculative 核心层各一），branch_tree.py 删
6 行死方法，指南五条勘误勾选。

**否决 B（把 make_child/fidelity 塞进 test_experiment_stats.py）**：
该文件主题是"统计契约"（AGENTS #32），make_child（张量构造）与
fidelity（度量）是实验 helper，混入稀释主题；新文件
`tests/test_algo_experiment_helpers.py` 命名即文档。

**否决 C（给 best_accepted_leaf 补测试保留死代码）**：guide :228
自标"死代码"——零调用零测试的公共面，正确处置是移除；补测试
等于给无人使用的路径上保险，反向激励。

**否决 D（BranchTree 全 API 移除只留 add/prune）**：get 被
get_ancestors 内部消费；工具类的导航面（get/get_ancestors/
seq_len）是合理公共契约——移除属过度修剪（srs §2 已定）。

## 2. 测试架构

### 2.1 文件与用例布局

| 文件 | 用例 | 被测对象 |
|---|---|---|
| tests/test_algo_experiment_helpers.py（新） | UT-1201/1202（make_child）、UT-1203/1204（fidelity） | scripts/algo_experiments.py（仅 import 被测函数本体——AR012 D2 纪律） |
| tests/test_branch_tree.py（新） | UT-1205~1207（活 API）、UT-1208（移除钉） | actfold/speculative/branch_tree.py |

### 2.2 种子纪律与独立重算

- 全部显式 `torch.manual_seed(S)` / `torch.Generator().manual_seed(S)`
  （AR012 纪律，:233 勘误的正向示范）；
- UT-1201 独立重算：手写 `Generator(seed)` → `randperm(seq_len)[:n]`
  → `randint(0, vocab, (n,), dtype)` 序列（与生产 :121-127 同序同流），
  positions 与翻转值逐位对照——**不 import 生产函数进重算路径**；
- UT-1204 手算：固定例 `out=[[1.,1.],[1.,1.]]` vs `base=[[1.,2.],[1.,1.]]`
  （形状 [2,2]=batch 2/vocab 2）——**第一性原理字面常量**（design
  门控 Minor-3/WARN-1：`base.var()` 与生产同原语同默认是名义独立，
  改用显式公式镜像 `((base−mean)**2).sum()/(n−1)` + 字面常量断言）：
  `mse = (0+1+0+0)/4 = **0.25**`；
  `var` unbiased：mean=1.25、平方差和=0.75、/3 = **0.25**（字面）；
  `rel_mse = 0.25/0.25 = **1.0**`（approx 1e-6——Welford 顺序的
  bit-exact 是运气非契约，WARN-1）；
  `cosine = (1+2+1+1)/(2·√7) = 5/5.29150 ≈ **0.944911**`（approx）；
  `top1`：argmax(out)=[0,0] vs argmax(base)=[1,0] → **0.5**（精确）。
  mse/top1 精确断言（2 的幂精确可表示）；var 路径显式公式镜像而非
  调 `base.var()`——torch 改默认（biased）时测试不同步漂移。

### 2.3 UT-1208 移除钉

`not hasattr(BranchTree, "best_accepted_leaf")` + 模块可导入 +
活 API 面（**add/get/get_ancestors/prune**——`seq_len` 是
`BranchNode` 的 property（:30-33）不在 BranchTree 上，design
门控 Major-1 勘误）`hasattr` 全真。移除决策入钉防复活（无调用方
的方法重新出现应被 review 质疑）。

## 3. 用例设计

### 3.1 make_child（UT-1201/1202）

| ID | 构造 | 断言 |
|---|---|---|
| UT-1201 | `tokens=torch.arange(12).reshape(1,12)`（long），vocab=10，num_flips=5，seed∈{7, 123} 参数化；同 seed 双调用；独立重算序列 | (child, positions) 双返回逐位==重算；双调用 `torch.equal`；seed 7 vs 123 输出可区分（分支命中——防 seed 被静默忽略） |
| UT-1202 | 边角矩阵：num_flips=0 / num_flips=-3 / num_flips=15>seq_len=12 / num_flips=12==seq_len / vocab=1（翻转值必为 0）/ float32 tokens | 零/负翻转 → clone 逐位等且**非同对象** + positions==[]；钳位后 len(positions)==12 且位置互异∈[0,12)；vocab=1 → child 翻转位全 0；父张量逐位不变（前后快照 `torch.equal`）；child.dtype==tokens.dtype（float32 路径钉 `dtype=tokens.dtype`） |

### 3.2 fidelity（UT-1203/1204）

| ID | 构造 | 断言 |
|---|---|---|
| UT-1203 | 恒等（**手选 norm 精确可表示张量** `torch.ones(2,2)*2` 类：norm²=16、√=4、积=16、dot=16 → cosine bit-exact 1.0——design 门控 Major-2：随机张量经 norm√→² 往返 ~60% 得 0.99999994，"精确"断言是抽签）/ 常量 base（var=0）/ float16 输入 | 恒等：mse=0/rel_mse=0/cosine=1/top1=1 四键精确（手选张量下成立）；地板：`math.isfinite(rel_mse)` 且 rel_mse == pytest.approx(mse/1e-12)（有限）；fp16：== float32 等价输入的结果（`.float()` 上转语义）；键集恰四键 |
| UT-1204 | §2.2 手算例 | 四指标 vs 手算（mse/rel/top1 精确，cosine approx 1e-6） |

### 3.3 BranchTree（UT-1205~1208）

节点构造镜像 folded_generation 生产形态（branch_id/parent_id/
tokens 必填）。三层链 root→mid→leaf + 旁支 sibling。

| ID | 断言 |
|---|---|
| UT-1205 | add/get 往返（同对象、未注册→None）；父子链接（parent.children 含子 id）；重复 add 同节点 children 不重复（:57-58）；孤儿 add（parent_id 未注册）不报错不入任何 children（:55）；**`BranchNode.seq_len == tokens.shape[1]` 值钉**（design 门控 Major-1 补——srs §3.3 验收项承载） |
| UT-1206 | get_ancestors：root→node 有序（[root, mid]）；include_self=True 追加自身；未注册 id：False→[]、True→[id]（req 门控 Minor-4：现行行为钉） |
| UT-1207 | prune(mid)：mid+leaf 递归移除（get 均 None）、root.children 解链不含 mid、sibling 不受波及；prune 缺失 id no-op；prune(root)：get(root)→None 且 `root_id` 属性残留 dangling（WARN-2 行为钉，docstring 记录"现行行为非应然"） |
| UT-1208 | 移除钉（§2.3） |

### 3.4 Red 预期与 mutation spot-check

行为钉子型：UT-1201~1207 对当前实现首跑全绿；UT-1208 在
**删除执行后**绿（T001 顺序：先写测试 → 删 :88-93 → 全绿；
删除前 UT-1208 红是天然的"移除确已生效"证据）。

mutation spot-check（Red 子代理执行，逐字节还原 + `git diff
actfold` 干净确认；design 门控 Minor-1/2 机制勘误——两者均经
**异常**而非值断言翻红，仍然有效）：
- (i) make_child 删 `num_flips = min(num_flips, seq_len)` 行 →
  `randperm` 切片静默封顶 12 个位置，但 `randint(0, vocab,
  (num_flips,))` 用未钳位的 15 生成 `(15,)` 值张量 →
  `child[0, <12 索引>] = rand` 抛 **RuntimeError（shape
  mismatch 15 vs 12）** → UT-1202 钳位例红；
- (ii) fidelity 地板 `max(var, 1e-12)` → `var` → 常量 base 时
  `.item()` 后的 Python float `mse / 0.0` 抛 **ZeroDivisionError**
  （非 inf）→ 异常传播 UT-1203 常量例红；
- (iii) branch_tree prune 删递归行 `for child_id ...: self.prune(
  child_id)` → UT-1207 红（leaf 残留 get 非 None——值断言翻红）。

## 4. 影响范围

| 文件 | 改动 |
|---|---|
| tests/test_algo_experiment_helpers.py（新，~150 行） | UT-1201~1204 |
| tests/test_branch_tree.py（新，~140 行） | UT-1205~1208 |
| actfold/speculative/branch_tree.py | **-6 行**（:88-93 best_accepted_leaf）——本 AR 唯一生产改动 |
| docs/OPTIMIZATION_GUIDE.md | :228/:230/:231/:232/:233 勾选+勘误、第二十部分 |
| CHANGELOG.md | AR013 节（Tests + Removed） |

**零改动声明**：actfold/ 其余全部、scripts/、AGENTS.md、既有
tests/*。

## 5. 决策记录

| # | 决策 | 理由 |
|---|---|---|
| D1 | 两个新文件而非塞入既有文件 | 主题分层（scripts helper vs speculative 核心）；test_experiment_stats 主题是统计契约 |
| D2 | best_accepted_leaf 移除而非补测试 | guide 自标死代码；零调用零测试（仓级 grep 证据）；补测试反向激励 |
| D3 | get/get_ancestors/seq_len 保留+钉死 | get 有内部消费者；工具类导航面是合理契约；只移除确证死代码 |
| D4 | UT-1208 移除钉（hasattr False） | 防复活：无调用方方法重现应被质疑；记录决策而非依赖 git 历史 |
| D5 | :233 勾选+勘误而非全量重写 799 用例 | autouse seed 本是解耦机制（每测试重置）；残余是隐式 seed 依赖——AR012 显式纪律已是新模式；全量重写 churn 巨大无 flaky 收益 |
| D6 | UT-1207 钉 dangling root_id | 行为钉子型纪律：钉现行行为并在 docstring 标注非应然，留给未来 AR 决策 |
| D7 | mutation 三位点对应三块钉 | clamp/地板/递归各代表一条被钉语义；spy 空断言型（AR012）在此为直接断言型 |

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| make_child 独立重算与生产 RNG 序漂移（randperm/randint 顺序） | req 门控已逐行核实 :121-127 调用序；UT-1201 双保险（重算 + 双调用确定） |
| fidelity 手算例 var 语义误判（biased vs unbiased） | req 门控 torch 2.5.1 实证 unbiased=0.25；UT-1204 手算从第一性原理推导（平方差和/3），Red 期若红即触发 srs 勘误条款 |
| UT-1207 prune(root) 钉过度锁定 quirk | docstring 显式"现行行为非应然"；未来改语义只需同步改钉（测试即文档） |
| 删除 best_accepted_leaf 影响序列化/子类 | 仓级扫描零引用（req 门控 #9）；方法体仅用 stdlib/self，无类型连带 |
| [1,T] 契约假设掩盖批>1 quirk | srs §5 假设明示不钉批>1；测试全部用 [1,T] 形状 |
