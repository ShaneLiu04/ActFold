# README 全量重写骨架（T008 工作文档）

> 依据 `AGENTS.md` "README Full-Update Rule"（每创新 AR 结档后全方位中文重写）。
> 本文档在 T006 正式工件落盘**前**起草：结构与散文已定稿，数字槽位用
> `⟨SLOT-n⟩` 标记，映射表给出槽位 ← 工件 JSON 键路径。回填时按 §4 顺序
> 逐槽替换，替换完成后**删除本文档中已回填章节的占位形式**（本文档本身
> 不入 README，结档后可归档或删除）。

---

## 1. 五节结构规划

| 节 | 现状锚点 | AR023 增量 | 操作 |
|---|---|---|---|
| 核心创新 | 创新总览表（AR022 行止）+ 各机制深潜节（按创新演进序） | 表加 AR023 行（§2.1 草稿）；深潜加"跨族投机泛化"节（§2.2 草稿，接在贪心锚定节后） | 追加 |
| 最新实验结果 | AR022 四模式表领头 + 历史档表 | AR023 两族四模式表领头（每族一张：p50 四模式 + H1/H2/H4 verdict 行）；AR022 表降位保留（演进叙事） | 重排 + 追加 |
| 快速上手 | 推荐路径（AR022 末态） | 补投机跨族配置最小例（`EXPERIMENTS.md` §6 已有 Python 例，README 给 YAML 面）；命令清单加 probe/bench 两行 | 修订 |
| 质量保障 | 951→1011 计数 | **1022→T007 终值**（全量 pytest + demo 三锚点后冻结；当前 1022/0/4 + demo 待跑） | 更新 |
| 限制与路线图 | AR022 末态版 | 按 AR023 结案重写：graph-off 两族注记 / MFF 元组图静态体未解包 / LLaDA 整块提交质量面（以 T006 verdict 定优先级）/ eval 适配器遗留 | 重写 |

README 既有五节外不动（系统架构/配置用法/文档/引用等节仅在过时处点修）。

---

## 2. 机制深潜散文（无数字定稿稿）

### 2.1 创新总览表 AR023 行

| 创新 | 动机（实测依据） | 机制 | 结果 |
|---|---|---|---|
| 跨族投机泛化（AR023） | AR022 在 Fast-dLLM 上以贪心锚定恢复基准质量（⟨SLOT-8⟩），但机制是否跨族泛化未经验证——LLaDA 块解码与 Dream maskgit 转移调度的画布语义截然不同 | 机制核/族适配器分离：`SpeculativeCore` 单源承载提交算子（冻结种子公式/min-prob 选位/贪心锚定/ctx LRU），族适配器只定义画布与载荷（LLaDA 整块右填充画布；Dream 超步窗口+预算合并+置信度选位） | LLaDA：⟨SLOT-1⟩；Dream：⟨SLOT-2⟩ |

### 2.2 深潜节正文草稿（标题：跨族投机泛化——机制核与族适配器分离）

> 叙事线遵循总纲：先测量 → 定位瓶颈 → 设计机制。

AR021 的测量暴露了投机路径的质量分叉（批量投机快 2.38× 但解码长度
160/77 vs 基准早停 34/66）；AR022 用贪心锚定在 Fast-dLLM 上修复了它。
但这两个答案都绑定在 Fast-dLLM 的画布语义上——小步块、裸张量层返回、
无 rotary 载荷。扩散 LM 的真实部署面跨三个族：LLaDA 每块解码、Dream
按转移调度渐进去噪，它们的"一个解码单元"含义完全不同。机制若不能跨族，
就只是单模型的补丁。

AR023 的设计答案是把"提交什么"与"画布是什么"拆开：

- **`SpeculativeCore`**（`actfold/models/speculative_core.py`）是无状态
  mixin（无 `__init__`，宿主 sampler 持有配置与计数器），承载全部与族
  无关的面：候选行构建的冻结种子公式（专用 CPU generator，零设备回读，
  跨族可复现）、min-prob 提交规则（最差填充位置最大化——行质量由瓶颈
  位置决定）、贪心锚定行 0（draft top-k 有序集首位，纯设备索引构建，
  零随机性消耗）、`(T, prompt_len)` 键控的 fold-ctx LRU、`greedy_win_stats`
  只读观测面。提交算子在三族间逐字节相同。
- **族适配器**只回答两个问题：一个投机单元是什么、画布怎么铺。LLaDA：
  每块一个单元，右填充画布，ctx mask 由 `(arange(T) < valid_end)` 平铺，
  `position=None`（LLaDA 无 rotary 载荷）。Dream：`speculative_steps=8`
  的超步窗口扫转移调度，预算按窗口合并（`k_eff=min(budget, masked_count)`），
  置信度 top-k 选位，`position_ids` 显式线程化（batch-1 无填充时与模型
  内部默认逐值相同——N=0 参照回路逐字节不动，这是跨族 A/B 可信的前提）。
- **位移从参照默认**：Dream 右移 logits、LLaDA 不移、Core 位移无关——
  族差异不泄漏进共享核。

跨族泛化的真正考验在**质量面**。真实 checkpoint 冒烟暴露了三处只在真实
族表面显现的生产缺口：远程 LLaDA block 返回元组（Fast-dLLM 返回裸张量，
故 AR019–AR022 从未触发 MFF 层循环的元组解包）；posthoc 质量面的行数据
是 Python list；sidecar 稳定比窗口必须按模式重置（sampler 每次 `sample()`
重置折叠链，branch id 跨调用复用）。这三处全部以 TDD 修复并钉死
（UT-2335 位精确对拍）。EOS 审计确认投机路径无抑制缺陷——长度分歧是
"空白画布单发整块提交 vs 多轮渐进去噪"的固有质量面差：Dream 的超步
渐进提交天然保持质量面，而 LLaDA 的整块 one-shot 提交把全部去噪压力
压在一次前向上。这一机制洞见直接指向下一档的方向（分块提交/渐进填充）。

（结果数字段接最新实验结果表的引导句，⟨SLOT-1⟩/⟨SLOT-2⟩。）

---

## 3. 最新实验结果节结构

> 表中 p50 列 = 各 prompt p50 的平均（与 H2 判定同口径；工件统计为逐
> prompt 嵌套，见 §4）。

### 3.1 LLaDA-8B-Base 表（⟨SLOT-1⟩）

| 模式 | p50 | vs baseline | H1 面 | 解码长度 | posthoc gap |
|---|---|---|---|---|---|
| baseline | ⟨SLOT-1a⟩ | — | — | ⟨SLOT-1d⟩ | — |
| folded-serial | ⟨SLOT-1b⟩ | | | | |
| folded-batched | ⟨SLOT-1c⟩ | | | | |
| folded-batched-greedy（对齐） | ⟨SLOT-1e⟩ | | | | |

verdict 行：H1 ⟨SLOT-1f⟩ / H2 ⟨SLOT-1g⟩ / H4 ⟨SLOT-1h⟩；复现自检
⟨SLOT-1i⟩；ratios ⟨SLOT-1j⟩。

### 3.2 Dream-v0-Base-7B 表（⟨SLOT-2⟩）

结构同上（槽位 2a–2j）。

### 3.3 前档表处理

AR022 表（四模式 3096.8/3205.4/1300.6/440.9ms，7.0236×）降位为
"上一档（Fast-dLLM 族）"，演进叙事保留；更早档表不动。

---

## 4. 数字占位映射表（槽位 ← 工件键路径）

> 工件：`results/optimization/ar023_llada_bench.json` /
> `ar023_dream_bench.json`（schema `ar023_cross_family_bench`；顶层
> `{schema, schema_version, family, status, modes, quality,
> verdicts{h1,h2,h4}, reproduction, meta}`，blocked 工件另有顶层
> `blocked_reason`）。
> **回填规则**：任何槽位对应 `status != "ok"` 或键缺失 ⇒ 该族表整体改为
> blocked 事实陈述（`blocked_reason` + 探针交叉引用原文），**不造数**。
> 模式键为**连字符**：`baseline` / `folded-serial` / `folded-batched` /
> `folded-batched-greedy`。逐模式统计是**逐 prompt 嵌套**的：
> `.modes.<mode>.wall_ms.prompt<i>.{p50_ms, mean_ms, std_ms, n}`——
> README 表的每模式聚合值 = 各 prompt p50 的平均（与 H2 判定式同口径；
> H2 侧的跨 prompt 平均值可直接取 `.verdicts.h2.numbers.*`）。

| 槽位 | 键路径（fam = llada|dream；`<mode>` 用连字符原值） | 说明 |
|---|---|---|
| 1a/2a | `.modes.baseline.wall_ms.prompt<i>.p50_ms` | 逐 prompt p50（i=0..P-1）；表值取平均 |
| 1b/2b | `.modes.folded-serial.wall_ms.prompt<i>.p50_ms` | 同上 |
| 1c/2c | `.modes.folded-batched.wall_ms.prompt<i>.p50_ms` | 同上 |
| 1e/2e | `.modes.folded-batched-greedy.wall_ms.prompt<i>.p50_ms` | 对齐模式 |
| vs baseline 列 | baseline/aligned 直接取 `.verdicts.h2.numbers.speedup_p50_vs_baseline`；其余模式由 1a-1e 平均值计算 | H2 侧已有冻结比值，勿自算口径漂移 |
| 1d/2d | `.verdicts.h1.numbers.per_prompt_decode_lengths[*].len_baseline` | 基准早停长度（每 prompt） |
| 解码长度列 | 同上 `len_folded_batched` / `len_aligned` | 长度恢复面 b |
| posthoc gap 列 | `.verdicts.h1.numbers.gap.{"folded-batched","folded-batched-greedy"}` | 键为连字符 |
| 1f/2f | `.verdicts.h1.verdict` + `.verdicts.h1.faces.{a_gap_shrink_ratio_le_0_5, b_length_recovery, c_block_exact_strict_improvement}` + `.verdicts.h1.strong.{quality_full_alignment, exact_all_hit, length_equals_baseline}` | supported/refuted + 三硬面 + 强结论层 |
| 1g/2g | `.verdicts.h2.verdict` + `.verdicts.h2.numbers.{p50_baseline_ms, p50_aligned_ms, speedup_p50_vs_baseline}` | 严格 >1.0 门；对齐 vs batched 仅报告面 |
| 1h/2h | `.verdicts.h4.verdict` + `.verdicts.h4.numbers.{stable_ratio_mean, canary_degradation_zero, ledger_within_bound}` | canary/ledger 合取 |
| 1i/2i | `.reproduction.ok`（**注**：`reproduction_anomaly` 键仅失配时存在——成功工件无此键） | 基准双跑 strict |
| 1j/2j | `.verdicts.h4.numbers.stable_ratios` | 每模式稳定比 |
| ⟨SLOT-8⟩ | AR022 工件既有数字（7.0236×）不变 | 动机句引用 |

**质量保障节数字**（非 T006 依赖）：T007 终值——全量 pytest 计数
（当前 1022/0/4，T007 后冻结）+ demo 三锚点（85.5% / 2.35e-03 / 93.75%
待 T007 复核后回填）。

**环境句数字**（表后注脚）：`meta.dtype` / `meta.graph_enabled` +
`meta.graph_note` / `meta.seed`（=2026）/ `meta.reps`（=30）/ P =
`len(meta.prompts)`（=2）/ `meta.family_constants.num_tokens`（=128）/
`meta.cupti_note`——均可直读；**工件 meta 无 GPU 型号键**，GPU 型号
（Quadro RTX 5000 16GB）从探针工件/运行环境另行注明，勿写成从本工件
直读。`meta.cross_rep_deterministic` 为复现自检的布尔汇总。

---

## 5. 回填操作顺序（T006 落盘后）

1. 两族工件 `status` 判定（§4 回填规则）→ 决定 §3 是数字表还是 blocked 陈述
2. §3.1/3.2 逐槽回填（§4 映射表）→ §2.1 结果列、§2.2 引导句
3. §1 质量保障行：T007 终值
4. 质量保障/快速上手/限制与路线图三节按 §1 操作列执行
5. 全文通读：数字与工件一致、技术词保留英文、演进叙事完整
6. 回填完成后删除本骨架文档（或移入 AR 归档目录）
