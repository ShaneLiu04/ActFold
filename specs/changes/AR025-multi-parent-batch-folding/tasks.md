# [AR025] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR025 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成）|
| 创建日期 | 2026-10-16 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | Red UT：多 parent 批量机制面测试全红（生产缺失：`parent_branch_ids` 参数/per-row 父装配/逐行 gate 对拍）；冻结面（AR015 批量契约 UT + AR023 族投机面）零破坏确认 | - | pending | 覆盖点矩阵待 design §6.1 |
| T002 | Green 机制核心：`ManualFoldedForward.parent_branch_ids` 尾参（互斥校验/None-attach 零扰动）+ per-row 父装配（gather + 行级降级，OQ1 定案形态）+ 逐行 gate/merge；mutation 全杀还原 | T001 | pending | 冻结常数/语义对拍 UT 钉死 |
| T003 | Green 发布与循环接线：`put_batch` 逐行胜者发布验证（#42 零改动预期验证）+ `_batched_forward_supported` 条件泛化 + `folded_generate` 多链入口（OQ2 定案形态）；mutation 全杀还原 | T002 | pending | |
| T004 | 配置面：合批开关（默认 off）+ `ActFoldConfig` 消费 + YAML 注释段；默认 off 逐字节兼容 UT | T002 | pending | #5 unknown-key 规则 |
| T005 | 合成基准脚本：多 parent 批量 vs 分批 vs 串行三方（路径分档容差对拍 + launch 账本）+ multi-geometry ≥4 形状点 + `--quick` 冒烟 | T003, T004 | pending | 判定式冻结入脚本 |
| T006 | 真实 checkpoint 基准：LLaDA/Dream 双链并行投机合批 vs 分批 A/B（reps=30 预承诺单次定案，ar019 venv；AR023 T006 基准完成后执行）；工件 `ar025_multi_parent_bench.json` | T005 | pending | 待 GPU 空闲 |
| T007 | 质量门：全量 pytest（1022/0/4 起只增不减，ar019 venv canonical 口径）+ pyflakes/mypy/black + demo 三锚点不动确认 | T003, T004 | pending | |
| T008 | 文档五件：README 全量中文重写（AGENTS 强制规则）+ AGENTS 新条目（多 parent 契约）+ GUIDE 新部分 + CHANGELOG + RERUN 新步骤（数字工件溯源） | T006, T007 | pending | |
| T009 | review 门控（只读子代理：spec/design 合规 + 代码审查），全部问题（含 Minor/Nit）修复循环至 PASS | T008 | pending | |
| T010 | ST 验收：st-cases.md 生成 + 执行（正常/边界/异常/回归四类） | T009 | pending | |
| T011 | 归档 `specs/archive/AR025-multi-parent-batch-folding/`，本地 commit + gitee 分段推送（AGENTS Gitee Segment-Push Protocol）+ 双端树验证 | T010 | pending | 永久 commit id 分歧预期 |

## 状态说明

- `pending`：待开启
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-16 需求起草 + req 门控第 1 轮 PASS（llada 正式基准运行期间，零实验先行）

- 选题定案（用户裁决）：指南 Part 25 遗留主项"变长/多 parent 批量"——
  本 AR 切分**多 parent**（等长约束保留），变长 pad 面明示 Out-of-Scope
  留下一 AR；并行 AR024（LLaDA 提交粒度）同为用户裁决"都要做"。
- 调研事实锚定：`_batched_forward_supported`（folded_generation.py:310-334）
  三条件边界；#41 单 parent expand 契约；#42 put_batch 批量写就绪/
  批量读缺失（folded_transformer.py `_forward_batched` :498/:514-522/:536
  单键 fetch/expand 广播，无 per-row gather）；机制数学基础（行间无
  耦合——泛化是读路径泛化非新算法）。
- srs.md rev 1 起草 → req 门控第 1 轮（审查代理 ses_eb8979327ffeDKNbfXYBzcD1b9）
  **PASS**（N1-N4 证据核对全 YES：三条件代码事实/#41 原文/#42 put_batch/
  单键广播无 gather 逐行核实）+ Minor×8 当轮修入：
  - G1 引用措辞：AR020 尾注"下一步"系伪引文 → 改为 srs §2 Out-of-Scope
    转述（"变长/多 parent 批量（AR016 遗留，正交）"）；
  - G4/G8 FR7 补 multi-geometry ≥4 形状点承诺（链数{2,4}×N{4,8}×T 两档
    网格）；质量门口径显式继承 AR023 rev 3 分级（构造性逐位一致故 H1
    简化为硬层+注记）；
  - G6 降级触发器冻结式（H2 delta_p50 <10% 且 H1 硬层 PASS ⇒ 真实面
    blocked 注记交付）入 FR7 判定式，先于数据；删"质量门风险高"模糊触发；
  - G7 对拍容差路径分档（非 split 逐位/split allclose 1e-6，#41 先例）；
    验收 5 基线标注 ar019 venv canonical 口径；launch 账本=write_op_count
    代理语义统一+#39 CUPTI null 注记入 meta；
  - G9 模板压缩注记（四段压缩版，design 阶段展开）；**AR 编号冲突解决**：
    specs/changes 同日双 AR024（本 AR 与 commit-granularity）→ 先占者
    （AR024-commit-granularity，用户裁决"都要做"的候选 A）保留编号，
    本 AR 改号 **AR025**（目录/srs/tasks 全部同步）。
- 教训留档：PowerShell `-replace`+`Set-Content -Encoding utf8` 对中文
  md 造成双重编码损坏（读回乱码）——srs/tasks 已用 write 工具整体重写
  修复；此后中文 md 一律 write/edit 工具（与 D:\RSI 会话纪律一致）。

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。

### 2026-10-16 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：13 项（G1-G9 + 证据核对 N1-N4）
- 修复的问题：Minor×8 当轮修入（G1 伪引文/G4 multi-geometry 缺承诺/G6 降级触发器未预注册/G7 容差分档+基线口径+账本代理语义/G8 质量门口径未钉死/G9 模板偏离+AR 编号冲突）
- 审查代理：sdd-gate-reviewer（ses_eb8979327ffeDKNbfXYBzcD1b9）；证据核对 N1-N4 全 YES（`_batched_forward_supported` :310-334 精确命中五原子检查/#41 原文逐点吻合/#42 put_batch 三点核实/folded_transformer.py :498-522-536 单键 expand 广播无 per-row gather+AR020 design §4.3 方案 C 否决理由旁证读路径从未批量化）
- 结论：需求阶段门控通过，可进入 design 阶段。
