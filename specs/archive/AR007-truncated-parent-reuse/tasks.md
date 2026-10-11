# AR007 任务分解 — P3-2 第一档：parent 更长时的截断前缀复用

> srs：`specs/changes/AR007-truncated-parent-reuse/srs.md`
> 拆分原则：TDD（Red 子代理/Green 主代理分离，微型任务主代理自完成）；
> 每任务完成即在下方表格更新状态（pending / in_progress / passing）。

| ID | 任务描述 | 前置 | 状态 | 备注 |
|----|----------|------|------|------|
 | T001 | `FoldedTransformerLayer` 截断复用：`h_parent` 形状守卫放宽（`T_p > T_c` 不再置 None，截断 `[:, :T_c]`，记录 `parent_longer`）；快路径 parent `ffn_out` fetch 后截断再比对形状；`_get_parent_ffn_output` 截断**仅限 `prefix_len is None`**（varlen 精确 shape RuntimeError 保留，EX-505 零回归）；`_merge_parent_child` 新增 `parent_longer=False` 参数禁 gather_select（等长路径逐字节不变）；既有失效条件（batch/hidden 不匹配、零长度）不变 | — | passing | UT-601~605：Red 3 failed（UT-601/602/604——parent 更长置 None 全重算）+ 2 钉死通过；Green 后 20/20，全量 759 passed 零回归（2026-10-11） |
| T002 | 端到端与回归：`ManualFoldedForward` parent 更长场景正常折叠（IT-611，parent 先行前向天然 warm-up）；append-only/等长路径逐位零回归（IT-612）；CUDA graph 变长 parent 零干扰（IT-613，silent eager 回退契约，**child tokens shape == 捕获 shape**——变长仅在 parent cache 侧） | T001 | passing | IT-611/613 落点 test_cuda_graph.py（复用 LlamaLikeModel/_make_mff harness，偏离 design §5"test_folded_generation.py 或新文件"）；IT-612 于 T001 全量 759 确认、T004 全量 761 复确认（2026-10-11） |
| T003 | 文档收口：指南 :199 截断子项勾选 + 第十四部分回链、AGENTS #20 修订、README 局限性 #7 修订、CHANGELOG AR007 节、folded_transformer.py:179-181 注释更新、存量测试 docstring 清理（test_folded_transformer_varlen.py:157 UT-203 "treated as no parent" → 截断复用口径；test_cuda_graph.py:1017 核对措辞） | T001–T002 | passing | 全部 grep 可验证；UT-203/bs001/AR007 类 docstring 三处清理；:179-181 注释于 T001 Green 内同步改写；**review 增补**：ALGORITHM.md §11 陈旧陈述修订 + 新增 §12 截断复用小节（srs 3.3/design §5 清单原漏列，规格遗漏已补记）（2026-10-11） |
| T004 | 全量回归 + 质量门：754 基线零回归、demo 基线（85.5% / 2.35e-03 / 93.75%）、mypy（AGENTS 标准命令）/pyflakes/100 列 | T001–T003 | passing | 761 passed（754+7 新增）/ 3 skipped / 3 deselected；demo 三指标精确一致；mypy 65 文件 clean；pyflakes 零；改动文件 100 列 clean（2026-10-11） |

## 任务详细说明

### T001 截断复用实现要点
- 改动集中在 `folded_transformer.py:170-182` 形状守卫与三个消费点：
  ① gate 输入（:170-182 + :191-244）：`T_p > T_c` → `h_parent = h_parent[:, :seq_len]`
  （截断后 `prefix_len == seq_len` 走既有等长分支，fused gate 可用——形状
  已相同）；② 全 stable 快路径（:269-290）：fetch parent `ffn_out` 后若
  `T_p > T_c` 截断再比对 `hidden_states.shape`；③ 慢路径
  `_get_parent_ffn_output`（:399-440）+ `_merge_parent_child`（:342-397）：
  截断后走 `prefix_len=None` 等长语义。
- `fetch_flat` 跳过条件：forward 早期记录 `parent_longer: bool`（D2 定案），
  传给 `_merge_parent_child` 作为 gather_select 禁用条件；等长路径保持
  现行 fetch_flat 优先顺序零额外 fetch（AR002 F8 不回归）；跨层错位异常
  状态以 fetch 后实际形状兜底截断。
- 注释与 docstring 同步：:177-181 的 "parent LONGER than the child ...
  Treat the parent as absent" 改写为截断复用语义（失效条件仅剩
  batch/hidden 不匹配与零长度）。

### T002 端到端
- IT-611：`ManualFoldedForward` + 因果合成模型，先跑 parent（8 token）
  填 cache，再跑 child（6 token，真前缀）——输出与 eager 全重算
  `torch.equal`、profiler 记录 stable_ratio==1.0、cache 含 child 条目。
- IT-612：既有 AR003 append-only 测试套（test_folded_transformer_varlen.py）
  全过零回归 + 全量 754。
- IT-613：graph 模式下变长 parent 步骤不捕获（`_parent_cache_complete`
  形状检查路由 eager），参照 test_cuda_graph.py 既有断言风格。

## 门控记录

### Requirements 门控（sdd-phase-gate）
- 第 1 轮：**PASS**（0 NO；4 WARN 均 Minor 移交 design/开发处理）——W1：`stable_ratio == 1.0` 测量口径未钉死（design 阶段定为 profiler `layer_stats[*].stable_ratio`，tests/test_stability_profiler.py:32-34 先例）；W2：AC2"非前缀→divergent"构造依赖需明确为独立采样 child（design IT 落地）；W3：AGENTS #10 编号引用不准（行为文档化源出 Documentation 节，srs 已修）；W4：存量测试 docstring（UT-203 varlen:157 / cuda_graph:1017）假陈述未入清理清单（已并入 T003）
- 审查代理亮点：fused kernel stride 契约实证（fused_ops.py:667-672 显式支持非连续输入，截断切片安全）；grep 确认无生产调用方依赖旧 no-parent 行为；`_parent_cache_complete` 实际位置勘误（architecture_utils.py:516 非 cuda_graph.py）
- 审查代理：general（只读）

### Design 门控（sdd-phase-gate）
- 第 1 轮：**FAIL**（1 NO + 2 Major + 3 Minor）——NO（Critical）：D3 将 `_get_parent_ffn_output` 统一放宽为 `>= mask_len 截断`会吞掉 varlen 路径（prefix_len=T_p）既有 RuntimeError → EX-505（varlen:586-618）基线回归 + 违反 srs §3.1 → 修复：放宽限定 `prefix_len is None`；Major-1：D2 trade-off 失实（fetch-first 的额外 fetch 实际发生在等长大形状 gather_select 路径，回归 AR002 F8）且 tasks 与 design 矛盾 → 修复：采纳布尔穿透（parent_longer 参数），等长路径零额外 fetch；Major-2：UT-601/604 冷启动 `_returns_tuple is None` 使快路径首调用不可达（UT-601 空转通过、UT-604 spy==0 必败）→ 修复：warm-up 前向约定（先例 test_folded_transformer.py:331-332）；Minor：D4 理由与 vectorized_cache.py:376-378 实际代码不符（fetch_flat 行公式在非 None 时确定性覆盖前 seq_len token——gather_select 对 parent 更长实际正确，留作后续性能 AR 线索）；跨层错位场景无测试 → 增补 UT-605 钉死 RuntimeError；IT-613 补 child shape==捕获 shape 约束
- 审查代理亮点：五问全数核验（快路径形状竞态安全/慢路径双 fetch 可行但开销定位颠倒/gate-ffn 层间正常同长且 D2-D7 表述自洽/spy 先例三处/graph silent 路由时序实证 architecture_utils.py:600-605 + cuda_graph.py:338-347）；UT-203 测试本体兼容性预判（随机 child vs 截断 parent 全 divergent 仍全重算 == layer(child)，本体无需改只改 docstring）
- 第 2 轮：**PASS**（第 1 轮 6 项修复全部实证落地；4 Minor 措辞级当场修复：命名统一 `parent_longer`、§3 表合并 _merge_parent_child 两行、UT-605 构造钉死 gate==child+merge<child、srs §3.1 异常措辞收窄）
- 审查代理：general（只读，两轮）

### 开发审查（sdd-task-review）
- 第 1 轮：**PASS**（0 NO；2 Minor 文档 WARN，当场修复）——①ALGORITHM.md §11 残留 "truncation reuse is unsupported" 与 AR007 矛盾（srs 3.3/design §5 清单漏列该文件，属规格遗漏）→ 新增 §12 截断复用小节 + §11 第 1 条修订 + T003 台账补记；②指南第十四部分/CHANGELOG "759" 与终态 761 不符（中间检查点数字）→ 两处修订
- 审查代理亮点：全量 761/demo/mypy/pyflakes/100 列本机实测复现；varlen 分支与 `git show HEAD` 逐字节对照等值；UT-605 错位构造路径推演（混合 mask 前置断言钉死慢路径必经）；IT-613 四重断言充分性确认；761=754+7 纯增量核对（无删除/跳过）
- 审查代理：general（只读）

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-11（T001–T004 完成）
- **T001**：Red 子代理（general）写 UT-601~605 → 3 failed（UT-601/602/604，失败原因均为"parent 更长置 None 全重算"）+ 2 钉死通过（UT-603/605），既有 15 测试全过；Green 主代理四处改动：①守卫 `T_p > T_c` 截断 + `parent_longer`（失效条件保序）；②快路径 fetch 后按**实际形状**截断（异常 cache 兜底，等长路径逐字节不变）；③`_merge_parent_child(parent_longer=)` 禁 gather_select；④`_get_parent_ffn_output` 放宽仅限 `prefix_len is None`（varlen 契约逐字节保留）→ 20/20，全量 759。
- **T002**：IT-611（CPU：真前缀 `torch.equal` + 全层 stable_ratio==1.0 + cache 发布）+ IT-613（CUDA：silent eager 零告警不禁用 + 后续 complete-parent child 仍捕获）落 test_cuda_graph.py 复用 harness；IT-612 回归确认。2 passed。
- **T003**：AGENTS #20 / README #7 / 指南 :199 勾选 + 第十四部分 / CHANGELOG AR007 节 / UT-203 + bs001 + AR007 类三处陈旧 docstring 清理。
- **T004**：全量 761 passed / 3 skipped / 3 deselected；demo 85.5% / 2.35e-03 / 93.75% 精确一致；mypy clean（65 文件）；pyflakes 零；改动文件 100 列 clean。
