# [AR008] 任务跟踪 — generic.forward `output_hidden_states` 按需化

| 字段 | 内容 |
|------|------|
| AR 编号 | AR008 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成）|
| 创建日期 | 2026-10-11 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | `GenericDiffusionLLM.forward` 按需化改造：默认 `output_hidden_states=False`；两字段缺失时置实例标志并带 `True` 重跑一次；标志置位后直传 `True`；RuntimeError 消息逐字保留。UT-701~705（Red 子代理）：logits-only spy False+逐位一致 / last_hidden-only 同 / hidden_states-only 首跑双调（False→True）+次跑单调 True / 三态全无 RuntimeError 逐字一致+不无限重跑 / branch 键剥离不受重跑影响 | — | passing | Red 6 failed（恒 True/缺重跑，无构造错误）；Green 后 7/7；test_models.py 10 passed 零回归（2026-10-11） |
| T002 | 内存峰值代理实证（EX-805，CUDA skipif）：24 层 × T=512 × hidden=4096 fp32 last_hidden-only stub，False vs True 对照 `max_memory_allocated` 峰值差 ≥ 150MB；既有 generic 消费链回归（test_models.py 全过） | T001 | passing | EX-805 实测通过（修 lm_head 设备迁移）；UT+EX 7 passed；消费链 10 passed（2026-10-11） |
| T003 | 文档收口：指南 6.2 条目勾选 + 第十五部分回链、CHANGELOG AR008 节 | T001–T002 | passing | grep 可验证（2026-10-11） |
| T004 | 全量回归 + 质量门：761 基线零回归、demo 基线（85.5% / 2.35e-03 / 93.75%）、mypy / pyflakes / 100 列 | T001–T003 | passing | 768 passed（761+7 新增，req 门控 767 collected 口径吻合：768+3 skipped+3 deselected）/ demo 三指标精确一致 / mypy 65 文件 clean / pyflakes 零 / 100 列 clean（2026-10-11） |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-11（T001–T004 完成）
- **T001**：Red 子代理（general）写 UT-701~705 → 6 failed（UT-701/702 恒 True 断言、UT-703/704×2/705 缺重跑 call_log 1≠2；UT-704 消息逐字断言 Red 期即通过）；Green 主代理：`__init__` 实例标志一行 + forward 按需化（平铺结构保留、递归重跑恰好一层、消息逐字）→ 7/7，test_models.py 10 passed。
- **T002**：EX-805 CUDA 峰值实证（24 层无参数 stub False/True 实例对照，修 lm_head CPU→CUDA 迁移）实测通过 ≥150MB。
- **T003**：指南 6.2 勾选 + 第十五部分回链、CHANGELOG AR008 节。
- **T004**：全量 768 passed / 3 skipped / 3 deselected；demo 85.5% / 2.35e-03 / 93.75% 精确一致；mypy clean（65 文件）；pyflakes 零；改动文件 100 列 clean。

## 门控记录

### Requirements 门控（sdd-phase-gate）
- 第 1 轮：**PASS**（0 NO；3 WARN 当场修复）——①§2 Out of Scope 论据错误：LLaDA/Dream 继承 generic.forward 未覆写（llada.py:14/dream.py:14），属"随动受益"而非"不经该路径"（Fast-dLLM 才是不经），3.2 回归点名继承链；②§1 算式 256MB 与"0.5-1GB"自相矛盾 → 修正为指南"数百 MB"口径；③UT-705（branch 键剥离重跑下保持）仅 tasks 有、srs 无验收条目 → §3.1 补第五条 Given/When/Then
- 审查代理亮点：调用链逐路径核实（base.py:125 AR fallback ✓ / diffusion_sampler.py:171 sampler 非折叠 ✓ / folded_model 直调与 `_native_sampler` 裸 model 今日即绕过 → 行为零变化，实例标志方案完备）；`pytest --collect-only` 实测 767 collected 与基线吻合；指出既有套件对 generic.forward 三分支**零覆盖**（generic 测试仅构造）——UT-701~705 是唯一钉子，须用真实 ModelOutput/SimpleNamespace stub 而非 MagicMock 返回值
- 审查代理：general（只读）

### Design 门控（sdd-phase-gate）
- 第 1 轮：**PASS**（0 NO / 4 WARN，当场修复）——WARN-1（高，防假 Red）：UT-703/705 stub 必须旗标条件式（False → hidden_states=None；True → 元组），恒返回元组的 stub 使"首次 2 调"断言成为假 Red → §4.1 补构造原则；WARN-2：伪码 raise 嵌入 else 使"真值元组但末元素 None"从 RuntimeError 变 TypeError → 改回现行平铺结构；WARN-3：EX-805 三条构造说明（手动置位 True 实例/无参数层运行期 clone/patch from_pretrained 返回真 stub 含 .config）；WARN-4：`hidden_states=()` 空元组边界 → UT-704 增变体；附注：`_needs_hidden_states: bool` 内联注解 + BS-701 口径备注
- 审查代理亮点：入参匹配语法级核实（attention_mask 具名参数不可能进 kwargs）；__init__ 单一构造路径核实（model=None 防御死码，LLaDA/Dream super().__init__ 全覆盖）；标志"粘性安全"补充论证（True 态下 logits 输出仍走第一分支，仅内存代价）；Red 可写性三条独立验证（UT-701/703/704 现行行为下必败）
- 审查代理：general（只读）

### 开发审查（sdd-task-review）
- 第 1 轮：**PASS**（0 NO；2 WARN 文档口径，当场修复）——①srs.md:15 仍为 "≈ 0.5–1GB"（req 门控修复轮的 edit 因 oldString 不匹配**静默失败未落地**，被 review 逐字比对抓到）→ 修正为 ≈256MB 与 CHANGELOG/guide 对齐；②srs §3.3 "≈200MB"（十进制）vs design "192MB"（MiB）→ 统一为 192MiB；srs 状态 Draft → Reviewed
- 审查代理亮点：全量 768/demo/mypy/pyflakes/100 列本机实测复现；RuntimeError 与 `git show HEAD` 逐字符对照；UT-703 第二次调用断言 call_log==3 钉死标志缓存（无缓存必为 4 调）；EX-805 时序核验（model_on lm_head 测前迁移，192MiB 元组是唯一大额新增）；768=761+7 纯增量（test_models.py 零 diff）
- 审查代理：general（只读）
