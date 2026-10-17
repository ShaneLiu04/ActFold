# [AR025] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR025 |
| AR 主题 | 多 parent 批量折叠（per-row parent 寻址）：批量分支验证的单一 parent 限定（AR015 #41 契约）泛化为逐行独立 parent——批量读/逐行门控/逐行发布的生产多链负载路径；变长 pad 面明示 Out-of-Scope（下一 AR 切分） |
| 关联 SR | AR023 srs §2 Out-of-Scope 与 GUIDE 遗留清单"变长/多 parent 批量"（Part 25 谱系，AR016/AR020/AR021/AR023 四处同引）；AR020 srs §2 Out-of-Scope 列"变长 / 多 parent 批量（AR016 遗留，正交）"——批量写（#42）已就绪、批量读缺失；并行 AR：AR024（LLaDA 提交粒度，用户裁决"都要做"，编号冲突已按先占者让号解决——本 AR 原 AR024 改号） |
| 日期 | 2026-10-16 |
| 状态 | rev 2（req 门控第 1 轮 PASS + Minor×8 当轮修入：G1 引用措辞去伪引文/G4+G8 multi-geometry 补承诺与质量门口径钉死/G6 降级触发器冻结式/G7 容差分档与基线口径标注/G9 模板压缩注记+编号冲突解决；模板为四段压缩版——逐 FR Given/When/Then 与非功能表按仓库六章式惯例在 design 阶段展开，表头注记依据） |

## 1. 背景与目标

用户总纲（全 AR 系列最高依据）："最终看的是综合框架比 baseline 要强，创新框架
的生产要有深度依据"——机制创新必须以实测证据推进，且生产链每一环的批量收益
不可在中途静默丢失。

### 1.1 现状与缺口（代码事实）

- **批量分支验证（AR015 #41）**：`ManualFoldedForward.forward` 接受
  `branch_ids: Sequence[str]`——行 i 属于 `branch_ids[i]`，但**全部行共享
  单一 `parent_branch_id`**（父激活必须以 batch 1 存储，expand 到
  `[N, T]` 供全行复用）。`folded_generation._batched_forward_supported`
  （`actfold/speculative/folded_generation.py:310-334`）三条件：ManualFoldedForward
  路径 / ≥2 候选同 parent / 等长且 parent-batch-1——**多 parent 或变长
  候选集直接回退逐条串行**。
- **批量写已就绪（AR020 #42）**：`put_batch` 结构协议 +
  `torch._foreach_copy_` 融合——发布侧天然支持多 branch entry 批量落盘
  （`_publish_graph_children` 每次发布重探能力）。**批量读缺失**：
  父激活的批量装配只有"单 parent expand"一种形态
  （`folded_transformer.py` `_forward_batched` 单键 fetch →
  `[N, T, H]` 广播，:498/:514-522/:536），无 per-row gather。
- **生产负载形态**：投机解码服务场景 = 多请求并行，每请求一条折叠链
  （各自的 parent 链、各自的胜者延续）。当前两条链 = 两次独立
  `[N, T]` 批量调用（launch 2 次）或逐条串行（launch N 次）——
  **跨链合批（`[2N, T]`，行 0..N-1 parent A / 行 N..2N-1 parent B）
  无机制支持**。
- **机制数学基础**：行 i 的折叠 = 该行 child 前缀 vs **该行自己的
  parent** 的逐行 gate 比较 + 逐行 merge——行间本无耦合（注意力在
  batch 维独立）；单 parent 限定是实现约束（expand 装配），非数学
  必要。这使 per-row parent 泛化成为"读路径的泛化"而非新算法——
  深度依据充分（AR015 批量对拍先例：批量 vs 串行 bit-exact）。

### 1.2 目标口径

- **主口径**：等长候选集（T 相同）下多 parent 批量的**机制正确性**（对拍
  等长串行/分批基线逐位一致）+ **生产收益**（跨链合批 vs 现状分批的
  launch 账本与 p50，合成 + 真实 checkpoint 双面）。
- **加速预期（诚实预告）**：合批收益 = launch 次数减半级的温和改进
  （1.x×），非 AR022 级 7×——叙事是**服务吞吐与机制完备性**（批量
  路径覆盖全部合法负载形态），不是单请求延迟革命。加速比>1 硬门；
  质量面**显式继承 AR023 rev 3 分级口径**（硬层/强层）并注记简化依据：
  合批 vs 分批由验收 1 构造性逐位一致，质量面直接继承 AR023 对齐
  模式工件结果（同 checkpoint 同 dtype 的 A/B 不因合批改变）。
- **生产链位次**：P1-7 第十一档（AR023 跨族 + AR024 提交粒度之后的
  指南遗留主项）。

### 1.3 环境约束

- llada/dream reps=30 正式基准（AR023 T006）占用本机 GPU/CPU——
  本 AR 需求/设计/UT 开发零实验先行；真实 checkpoint 基准待其完成
  （ar019 venv，复用已验证的 curl 权重 + bf16 加载路径，探针结论
  degraded=False 直接沿用，无需新探针）。
- Windows torch 2.5.1 cu121（图路径 graph-off 沿 AR023 探针结论；
  合批 eager 路径为主面，图静态体多 parent 面明示 Out-of-Scope）。

## 2. 功能需求（FR）与范围

- **FR1 per-row parent 批量 API**：`ManualFoldedForward.forward` 增
  `parent_branch_ids: Sequence[str]`（与 `branch_ids` 对齐；与
  `parent_branch_id` 互斥，`ValueError`）；单 parent 调用 =
  现路径逐字节不变（默认 None 附加零扰动，AR021 None-attach 先例）。
- **FR2 per-row 父装配（批量读）**：每行父键各自查缓存（各父须以
  batch 1 存储——#41 expand 契约逐行版）；装配 = per-row gather
  到 `[N, T, H]`（等长约束下纯索引操作，无跨行数据依赖）；任一行
  父缺失/形状不匹配 = **该行** 全重算（单行降级不整批回退——对拍
  现契约"缺失=整批重算"的语义泛化，粒度由 design 定案，OQ1）。
- **FR3 per-row gate/merge**：行 i 的稳定掩码 = child 行 i vs parent
  行 i 的逐行比较（数学上与分批调用恒等——对拍 UT 钉死）；merge
  逐行对齐各自父 FFN。
- **FR4 发布面**：`put_batch` 批量发布逐行胜者（#42 已支持多 branch
  entry——预期零改动或仅装配层，design 验证）。
- **FR5 生成循环接线**：`folded_generate` 多链变体或批量投机服务
  入口（≥2 条独立链的候选合批；`_batched_forward_supported` 条件
  泛化：等长保留、同 parent 放宽为"全部行有各自 batch-1 父"；
  形态由 design 定案，OQ2）。
- **FR6 配置面**：合批开关（默认 off 逐字节兼容）+ YAML 消费
  （#5 unknown-key 规则）。
- **FR7 基准（预注册，丰富性按 AGENTS Experimental Validation Rule）**：
  - 合成面：多 parent 批量 vs 分批 vs 串行三方对拍（逐位一致 +
    launch 账本）；**multi-geometry ≥4 形状点**（链数 {2,4} ×
    N {4,8} × T 两档的网格取 ≥4 组合——合批收益对 (链数×N×T)
    形状敏感，无豁免）；
  - 真实 checkpoint 面：双链并行投机（LLaDA-8B-Base / Dream-v0-7B
    各一链，N=4 候选）合批 vs 分批 A/B，reps=30 预承诺单次定案，
    跨工件对照 AR023 同表；
  - H 假设预注册（H1 质量对齐——继承 AR023 rev 3 分级口径，构造性
    逐位一致故简化为硬层 / H2 加速>1 / H4 账本），判定式冻结先于
    数据；**降级交付触发器（冻结式）**：H2 delta_p50 相对提升
    <10% 且 H1 硬层 PASS ⇒ 交付"机制完备 + 合成面证据 + 真实面
    blocked 注记"（诚实负结果路径，先于数据冻结）；负结果如实交付。

### Out-of-Scope（明示）

- **变长 pad 面**（同批异长候选的 pad-to-max + per-row 有效长度
  mask）——与多 parent 正交，本 AR 等长约束保留；变长切分留下
  一 AR（Part 25 谱系第二项）。
- **图静态体多 parent**（graph runner 的 per-row 父缓冲）——
  AR023 探针 graph_enabled=False，图面整体冻结直至捕获问题另案。
- 多 parent 下的 LRU/淘汰交互深化（ctx 键仍 `(T, prompt_len)`
  逐链独立，预期零改动，design 注记即可）。

## 3. 验收标准

1. **机制对拍**：等长多 parent 批量 vs 逐链分批，hidden/logits
   **路径分档容差**（沿 #41 先例）：非 split 路径逐位一致；split
   聚集路径 allclose(1e-6)（gathered GEMM 跨 BLAS 微内核边界）。
   对拍用例覆盖：同父混合调用（退化=现路径）、全异父、含父缺失行、
   含胜者延续两轮。
2. **零回归**：单 parent 批量路径（AR015 冻结面 UT）与串行路径
   逐位一致不变；默认 off 时 `folded_generate` 逐字节不变。
3. **账本**：合批 launch 账本 ≤ 分批（`write_op_count` 沿 #42 语义
   作为 launch 代理，排除循环验证；#39 CUPTI null 注记入工件 meta）。
4. **基准工件**：合成 + 真实双面 reps=30 预承诺工件
   （`results/optimization/ar025_multi_parent_bench.json`），
   H 判定式冻结入脚本、数字全部工件溯源（README 全量重写按
   AGENTS 强制规则随后执行）。
5. **质量门**：全量 pytest 只增不减（AR023 基线 1022/0/4 起算，
   **ar019 venv canonical deselect 口径**——系统 python 为
   1021/3/4 环境漂移，AR023 tasks 实录）、black/pyflakes/mypy
   改动面 clean、demo 三锚点不动。

## 4. 风险与开放问题

- **OQ1**：父缺失行的降级粒度——"该行全重算"需在批量 eager 体内
  表达 per-row 重算（掩码式选择 vs 拆批）；拆批破坏 launch 收益，
  掩码式增加机制面。design 阶段定案（倾向掩码式，#20 整批重算
  契约的行级泛化）。
- **OQ2**：`folded_generate` 接线形态——多链变体函数 vs 独立批量
  服务入口；影响 FR5 验收面（合成基准走哪个入口）。
- **OQ3**：合批收益的真实放大器——launch 减半之外，WDDM 下
  30-40μs/launch 的实际占比待合成基准量化（AR018 §1.3 先例：
  发布占比 45-50% 的谱系延续）；降级交付路径已在 FR7 冻结式
  预注册（见上），不再以模糊触发。
