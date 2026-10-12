# [AR012] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR012 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-12 |

## 测试用例列表

### ST-001：`add_gumbel_noise` T>0 公式精确重算钉

**关联需求：** srs.md §3.1 验收①（G1）
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 固定种子下 `add_gumbel_noise(logits, T)` 与测试内逐算子手写重算
   `exp(logits.double()) / (-log(rand))**T` 对照，T∈{0.5, 1.0, 2.0}
2. dtype/同种子两次调用/T=0.0 `is` 同对象

**期望结果：** 逐位相等；float64；`torch.equal`；`is` 同对象

**实际结果：** UT-1101 4 参数化全 passed（T=0.0 no-op 是 :206-207 形状断言之外的**首个值级** no-op 钉）

**状态：** PASS

---

### ST-002：`sample_tokens` T>0 Categorical 路径重算钉

**关联需求：** srs.md §3.1 验收②（G5）
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 六参数组（T 单独/+top_k/+top_p/**top_p+top_k 联合（top_p 先）**/+margin/+neg_entropy）
   固定种子调用，与镜像生产算子序的手写重算对照双返回值
2. 断言 sampled token 的 gathered prob > 0；-inf 过滤位概率精确 0

**实际结果：** UT-1105 6 参数化全 passed（双返回值逐位相等 + -inf 位概率和精确 0；margin/neg_entropy 覆写镜像 :319-326）

**状态：** PASS

---

### ST-003：LLaDA `remasking="random"` 端到端

**关联需求：** srs.md §3.2 验收①（G2）
**测试类型：** 正常路径 + 分支命中证明
**优先级：** High

**测试步骤：**
1. 同种子两次 `sample()`（序列 + 全部 history `torch.equal`）
2. `torch.rand` spy 记录 shape==x0.shape；全揭示；suppress_tokens[3] 不出现且 ∈ 合法词表

**实际结果：** UT-1102 passed（spy shape==(2,9) 命中——空转通过防护在位；mutation (i) 反义分支后 spy 空先红实证）

**状态：** PASS

---

### ST-004：LLaDA `stochastic_transfer=True` + T=0.5 端到端

**关联需求：** srs.md §3.2 验收②（G3）
**测试类型：** 正常路径 + 分支命中证明
**优先级：** High

**测试步骤：**
1. 同种子确定性；Binomial 代理 spy 非空；`add_gumbel_noise` 以 T=0.5 被调
2. history 单调（mask 计数不增、终态全揭示）

**实际结果：** UT-1103 passed（末步 p=1 保证 Binomial 全量传输的机制成立；mutation (ii) 硬编码 False 后 Binomial spy 空先红实证）

**状态：** PASS

---

### ST-005：Dream `alg_temp>0` 逐行软选择端到端

**关联需求：** srs.md §3.2 验收③（G4）
**测试类型：** 正常路径 + 分支命中证明 + 弱收敛性质
**优先级：** High

**测试步骤：**
1. 同种子确定性；`torch.multinomial` spy 非空；全揭示
2. 弱收敛：`alg_temp=1e-8` 时每行首个 multinomial 采样索引 == 该行置信度 argmax

**实际结果：** UT-1104 passed（`-inf/1e-8 = -inf` 非 NaN、softmax 精确 one-hot——Red 期实测可靠保留弱性质，未降级；mutation (iii) 条件改 `!=` 后 spy 空先红实证）

**状态：** PASS

---

### ST-006：`get_num_transfer_tokens` stochastic 数值钉

**关联需求：** srs.md §3.3（G6）
**测试类型：** 数值/统计
**优先级：** High

**测试步骤：**
1. 同种子两次调用 `torch.equal`；(4,20)@0.6 密度下钉死种子 S=0 使 stochastic ≠ deterministic
2. 单行全 mask（n=20）种子 0..99：第 0 步（首个正传输列，口径偏置已注 docstring）均值 vs 解析期望 `n·p₀`，容差 ±0.89（6σ）；p₀ 手写复刻生产四次 f32 舍入

**实际结果：** UT-1106 passed（实测均值 2.70，|Δ|=0.20 ≤ 0.89；手算 p₀ == 0.12500083446502686 逐位钉；未 import 生产 scheduler——非循环验证）

**状态：** PASS

---

### ST-007：测试自证（mutation spot-check）

**关联需求：** design.md §3.4（行为钉子型的测试有效性）
**测试类型：** 变异测试
**优先级：** High

**测试步骤：**
1. 临时反转 llada:221 remasking 判断 → UT-1102 应红
2. 临时硬编码 llada:185 stochastic=False → UT-1103 应红
3. 临时改 dream:173 `== 0.0` 为 `!= 0.0` → UT-1104 应红
4. 每次逐字节还原，`git diff actfold` 确认干净

**实际结果：** 三位点全部按预期翻红（spy 空断言先红），还原后 14/14 全绿、生产零残留（T001 报告 + review C 项 git diff 复核）

**状态：** PASS

---

### ST-008：文档收口

**关联需求：** srs.md §3.4
**测试类型：** 文档审查
**优先级：** Medium

**期望结果：** 指南 :229 勾选+勘误（CFG/suppress 已由 T018 覆盖，真缺口为随机分支）；第十九部分回链；CHANGELOG AR012 节置 Unreleased 最前；AGENTS 零改动（纯测试 AR）

**实际结果：** 全部落实（review D 项文档事实核对逐项一致：799 passed/14 用例/p₀ 常数/mutation 结果）

**状态：** PASS

---

### ST-009：全量质量门

**关联需求：** srs.md §4 NFR
**测试类型：** 回归测试
**优先级：** High

**实际结果：**
- 全量 `pytest tests\ -q -m "not slow"`：**799 passed / 3 skipped / 3 deselected**（785 基线 + 14 新增，零失败，39.76s）
- demo：FLOPs reduction 85.5% / MSE 2.35e-03 / stable ratio 93.75% **精确一致**
- mypy clean（65 文件）/ pyflakes 零 / 100 列
- 新文件耗时 6.66s（<10s NFR）

**状态：** PASS

---

### ST-010：生产代码零改动

**关联需求：** srs.md §2 Out of Scope（`actfold/` 零改动约束）
**测试类型：** 变更审计
**优先级：** High

**实际结果：** `git status --porcelain` 仅 `M CHANGELOG.md`、`M docs/OPTIMIZATION_GUIDE.md`、`?? specs/changes/`、`?? tests/test_sampler_stochastic_branches.py`——**actfold/ 全目录零改动**（review C 项逐字节对照 + mutation 还原后两次确认）

**状态：** PASS

---

## 验收结论

**10/10 Go。** srs §3 全部验收标准由 UT-1101~1106 落实且经独立 review 五维核查（A spec/B design/C diff/D 运行/E 风险）；两门控（req/design）各 FAIL→修复→PASS 的修正全部落文档；测试经 mutation 自证非空转绿；全量回归与 demo 基线零漂移。AR012 可归档。
