# [AR013] 需求设计说明书 — 指南 6.1 测试债清单终审收口

| 字段 | 内容 |
|------|------|
| AR 编号 | AR013 |
| AR 主题 | 6.1 测试债清单（:228-:233）逐条终审：四条勘误勾选 + 真缺口补齐（`make_child`/`fidelity` 纯函数钉、`BranchTree` 活 API 直接钉）+ 死代码 `best_accepted_leaf` 移除 |
| 关联 SR | 指南 6.1 :228/:230/:231/:232/:233（测试覆盖盲区清单）；AGENTS #32（scripts 统计契约）；AR012 确立（显式种子纪律） |
| 日期 | 2026-10-12 |
| 状态 | Final（req 门控首轮 PASS：5 Minor 全采纳；design 门控 FAIL→修复 Major-1/2 + Minor/WARN 全采纳→PASS；T001–T007 passing（UT-1208 删除前红/后绿 + mutation 三位点自证）；review PASS（0 Major 0 Minor）；ST 7/7 一次成 Go；已归档） |

## 1. 背景与目标

指南 6.1 "测试覆盖盲区（先补测试再动手优化）" 共 6 条，AR012 关闭
:229 时已发现清单**大面积过时**——历次 AR（AR002/AR004/AR006/
AR008/T018）已把多数条目实质关闭但从未勾选。逐条终审（本轮勘察
已完成，证据在案）：

| 条目 | 声称 | 现实（勘察证据） | 处置 |
|---|---|---|---|
| :228 | folded_generate 多分支零测试；AcceptancePolicy/BranchTree 无测试 | `tests/test_folded_generation.py` 现 31 例（多候选 IT-512、model_draft、prune 集成、causal 等价、graph 零干扰、acceptance rate 报告全套）；`TargetMatchAcceptancePolicy` 7 个直接用例（:718-805）；`acceptance.py` 纯函数 59 hits 于 test_acceptance.py | **勘误勾选**；残余：BranchTree **直接** API 钉缺（prune 仅经集成测试间接覆盖 :347-379）、`best_accepted_leaf` 死代码 → 本 AR FR3/FR4 |
| :229 | samplers 形状 smoke | AR012 已关闭（随机分支数值钉） | 已勾选 |
| :230 | get_num_transfer_tokens 无官方对拍 | `test_t018_num_transfer_tokens_reference_parity`（test_sampling_utils.py:225，冻结 pre-T018 逐行参考 :80-154；官方对齐链自证见 sampling_utils.py:158/:166 "bit-identical to the official per-row loop"）已钉 | **勘误勾选** |
| :231 | ManualFoldedForward 无 final-norm 回归 | `test_manual_folded_forward_applies_final_norm`（test_architecture_utils.py:216-235，folded vs raw allclose+argmax 双钉，docstring 自标 "srs 3.1 B2"）+ `test_detect_architecture_finds_final_norm`（:202）AR002 时代已钉 | **勘误勾选** |
| :232 | make_child/time_forward/fidelity 无纯函数单测 | `tests/test_experiment_stats.py` 16 用例已覆盖 `TimingStats`(2)/`stats_from_samples`(4)/`time_forward`(2)/`repeat_with_seed`(5)/`exp_sampling`(1)/`fig_sampling`(2)；**真缺口**：`make_child`（algo_experiments.py:115-130）与 `fidelity`（:133-146）零覆盖 | **勘误 + FR1/FR2 补真缺口** |
| :233 | conftest autouse seed 使测试共享随机态 | 表述失准：autouse（conftest.py:22-27）每测试函数**重置** seed=42——它是**解耦**机制而非耦合源；真实残余是"测试隐式依赖 seed=42 值本身" | **勘误勾选**（决策记录：显式种子纪律 AR012 已立为新模式；全量重写 799 用例否决——churn 巨大且无 flaky 修复收益） |

**目标**：6.1 清单 6/6 终态（勾选+勘误），两条真缺口数值钉死，
死代码清除，使指南债清单恢复可信度（AR008-012 连续五次发现债
条目前提腐化——本 AR 是对整个清单的系统性纠偏）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- 新测试文件 `tests/test_algo_experiment_helpers.py`：UT-1201
  （`make_child` 纯函数钉）、UT-1202（`fidelity` 纯函数钉）；
- 新测试文件 `tests/test_branch_tree.py`：UT-1203（`BranchTree`
  活 API 直接钉：add/get/get_ancestors/prune/seq_len）；
- **死代码移除**：`branch_tree.py` 删除 `best_accepted_leaf`
  （:88-93，全仓仅定义+债行两处引用，零调用零测试——guide 自标
  "死代码"，移除即对 :228 残余的正确处置，而非给死代码补测试）；
- 文档：:228/:230/:231/:233 勾选+勘误、:232 勾选+勘误（真缺口
  由本 AR 关闭）、指南第二十部分回链、CHANGELOG AR013 节
  （Tests + Removed）。

**Out of Scope（本 AR 不做的）：**
- `actfold/` 其余零改动（branch_tree.py 仅删 6 行死方法）；
- BranchTree 的 `get`/`get_ancestors`/`seq_len` **不移除**（get 被
  get_ancestors 内部消费、prune 经 `_nodes` 直取；工具类公共面，
  移除属过度修剪）——改为 UT-1203 直接钉；
- :233 全量显式种子重写（见 §1 决策记录）；
- algo_experiments.py 其余实验入口（exp_similarity/exp_tau_sweep
  等——是脚本 main 性质，非纯函数 helper，guide :232 未点名）；
- `encode_prompt`/`build_folded`/`run_folded_child`（依赖
  tokenizer/GPU/profiler 环境，非纯函数）、`profile_summary`
  （实为纯函数，排除理由是 :232 未点名——req 门控 Minor-5 勘误）。

## 3. 功能需求

### 3.1 `make_child` 纯函数钉（UT-1201）

**描述：** scripts/algo_experiments.py:115-130——parent 扰动构造器，
被 exp_* 系列实验用作折叠子分支生成；其 RNG 契约与钳位语义无钉。

**验收标准：**
- **确定性**：同 seed 两次调用 → (child, positions) 逐位相等；
  不同 seed → 输出可区分（分支命中证明，防实现静默忽略 seed）；
- **零翻转早退**：`num_flips<=0` → 返回 `tokens.clone()` 与空
  positions（且 clone 非同对象——调用方改 child 不污染 parent）；
- **位置语义**：`len(positions) == num_flips`、位置互异、
  ∈ [0, seq_len)；
- **值域语义**：翻转后 `child[0, positions]` ∈ [0, vocab_size)；
- **父不可变性**：调用后 parent 逐位不变（clone 防护钉）；
- **钳位**：`num_flips > seq_len` → 实际翻转数 == seq_len（:123
  `min` 钳位）；
- **dtype/设备**：child dtype == parent dtype（long 输入经
  `dtype=tokens.dtype` 生成器路径）；CPU 张量测试（`.to(device)`
  分支以同设备直通，不钉 CUDA 特有行为）；
- 独立重算钉：同 seed 下 positions 与翻转值与测试内手写
  `Generator.manual_seed(seed)` → `randperm` → `randint` 序列
  逐位一致（纯函数单 RNG 流，重算可行——AR012 G1 同型）。

### 3.2 `fidelity` 纯函数钉（UT-1202）

**描述：** :133-146——folded vs baseline logits 的四指标字典
（mse/rel_mse/cosine/top1_agreement），实验结论的度量基础。

**验收标准：**
- **恒等输入**：out == base（**非零张量**——cosine 的零向量 clamp
  路径返回 0，req 门控 WARN-1）→ mse=0.0、rel_mse=0.0、
  cosine=1.0、top1_agreement=1.0（四键全在，值精确）；
- **手算小例**：固定小张量（如 [2,2] 含单元素差异），四指标与
  测试内**独立手算**逐位一致——含 `Tensor.var()` 的 unbiased
  语义（除 n-1）与 `rel_mse = mse / max(var, 1e-12)` 公式
  （Red 期以手算为准校验生产语义，防止照抄生产实现的循环验证：
  测试重算用 `torch` 原语但公式逐算子镜像推导，不 import 生产
  函数）；
- **方差地板**：base 为常量张量（var=0）→ rel_mse == mse/1e-12
  （分母地板生效、值有限非 inf）；
- **dtype 上转**：输入 float16 → 内部 `.float()` 上转后计算
  （以 float32 精度结果断言，钉 :135-136 上转语义）；
- **键集完备**：返回 dict 恰为四键（无多无少）。

### 3.3 `BranchTree` 活 API 直接钉（UT-1203）

**描述：** branch_tree.py——folded_generation.py:115 构造、:141
add、:170 prune 生产在用；直接 API 钉缺（prune 现仅集成级间接
覆盖）。

**验收标准：**
- **add/get 往返**：add 后 get 返回同对象；未注册 id → None；
- **父子链接**：add 子节点后 `parent.children` 含子 id；**重复
  add 同一节点不重复入列**（:57-58 防护钉）；孤儿（parent_id
  未注册）add 不报错不入任何 children（:55 条件钉）；
- **get_ancestors**：root→node 顺序、include_self 开关、
  未注册 id → 空列表（**仅 include_self=False**；True 时 while 经
  get 取 None 终止后仍 append，返回 `[branch_id]`——req 门控
  Minor-4：照"恒空"字面写必失败，钉现行行为）；
- **prune 递归语义**：剪中间节点 → 其全部后代递归移除（get 均
  None）、父 children 解链、**缺失 id no-op**（`pop(None 默认)`
  幂等）；剪 root 合法性以现行行为为钉（root 被移除后 get(root)
  → None，**且 `root_id` 属性残留指向已移除 id**（:50 赋值后
  prune 不清理——dangling 行为钉，req 门控 WARN-2））；
- **seq_len 属性**：`BranchNode.seq_len == tokens.shape[1]`；
- **与生产形态一致**：节点结构镜像 folded_generation 的构造形态
  （branch_id/parent_id/tokens 必填，其余默认）。

### 3.4 死代码移除（`best_accepted_leaf`）

**验收标准：**
- `branch_tree.py:88-93` 整方法删除；全仓引用扫描仅剩指南 :228
  债行（本 AR 勾选勘误时一并清除表述）；
- 移除后 mypy/pyflakes clean、全量回归零失败（零测试引用它——
  勘察已证）；`git diff actfold` 净变化恰为该删除。

### 3.5 文档收口

**验收标准：**
- :228/:230/:231/:232/:233 五条全部勾选 + 勘误注（各自证据与
  关闭者见 §1 表格；:233 含否决全量重写的决策记录）；
- 指南第二十部分 AR013 回链；CHANGELOG AR013 节（Tests +
  Removed，Unreleased 最前）；
- AGENTS.md 预计零改动（best_accepted_leaf 未被 AGENTS 引用；
  勘察已证）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 静态检查 | mypy 标准命令 clean / pyflakes 零 / 100 列 |
| 质量 | 全量回归 | `pytest tests\ -q -m "not slow"`：799 passed 基线 + 新增零失败 |
| 稳定 | 测试确定性 | 全部显式 `manual_seed`/`Generator`（AR012 纪律）；无统计容差断言 |
| 兼容 | 公开 API | `best_accepted_leaf` 为零调用死方法，删除非破坏（grep 证据在案） |
| 性能 | 用例耗时 | 两新文件合计 <5s（纯 CPU 小张量） |

## 5. 约束与假设

**约束：**
- `actfold/` 唯一改动 = branch_tree.py 删 6 行（:88-93）；任何
  其他生产文件零改动；
- UT-1201/1202 独立重算不 import 生产函数（algo_experiments 仅
  import 被测函数本体；公式逐算子手写——AR012 D2 先例）；
- UT-1203 钉现行行为（含 prune root 的边角行为），不做应然设计
  ——行为钉子型 AR 纪律。

**假设：**
- `Tensor.var()` 默认 unbiased=True（torch 2.5.1）是 fidelity
  rel_mse 的现行语义——Red 期手算校验确认，若实测为 biased 则
  以实测勘误 srs（勘误比照单实现重要的先例）；
- `make_child` 的契约形状是 `[1, T]` 单行（:129 `child[0, ...]`
  硬编码行 0）——批>1 行为是未定义边角，不钉不测（钉了反而
  过度锁定实现 quirk）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 债条目腐化 | 指南债清单条目的前提被后续 AR 实质关闭但未勾选——AR008-012 连续五次发现的系统性问题 |
| 活 API | 有生产调用方的公共方法（BranchTree: add/prune；get 经 get_ancestors 内部消费） |
| 死代码 | 零调用零测试的公共方法（best_accepted_leaf）——正确处置是移除而非补测试 |
| 方差地板 | fidelity 的 `max(var, 1e-12)` 分母防护——base 常量时 rel_mse 仍有限 |
