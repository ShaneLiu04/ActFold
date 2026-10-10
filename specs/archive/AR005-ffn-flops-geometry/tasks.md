# [AR005] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR005 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-10 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | `count_diffusion_llm_flops` MoE 扩展：`moe_num_experts`/`moe_top_k`/`moe_intermediate_dim`/`moe_shared_expert`/`moe_num_layers`（默认全层，`[0, num_layers]` 域校验）参数 + per-token expert FLOPs 公式（routed top-k ± shared，dense/MoE 混合层数）+ ValueError 校验（top_k 域、num_experts 正性、维度正性）；缺省逐位零回归 | - | passing | 2026-10-10 完成：UT-401~406 + EX-701~705 全绿（12 新测试）；test_flops_counter 21 passed |
| T002 | `DiffusionLLM` base `config` 转发 property（`self.model.config`）+ 7 个具体 FFN/MoE 几何属性 + `_extract_ffn_geometry(config)`/`_resolve_model_config(target)` 单一实现 + `model_ffn_flops_kwargs` adapter `underlying_model` 链解析 + 新 MoE 键 | T001 | passing | 2026-10-10 完成：UT-407~410 全绿（Red：16 failed 实证）；含 SwiGLU 族判定；真实路径 = 几何仅在 `.model.config`；实现偏离见 design §4.3.2 补记（链尾属性并查 + 有界钻探深度） |
| T003 | 端到端接线实证：config-like stub（SwiGLU ± MoE）经 `FastDLLMAdapter` → `ActFoldVerificationEngine._estimate_tflops` 供参；合成模型零回归断言 | T002 | passing | 2026-10-10 完成：IT-411（engine，tflops==真实几何手算且 ≠ 4h 默认）+ IT-412a（ablation `_flops_budget`）+ IT-412b（base_adapter `_estimate_baseline_tflops`）全绿；三调用点零改动（设计核心收益实证）；因生产接线在 T002 已成，T003 以验证测试直接落地（无 Red 态） |
| T004 | 文档收口：AGENTS #33 修订、CHANGELOG AR005、指南 P3-4 ✅ + 第十二部分回链、README/docs TFLOPs 口径检查 | T001–T003 | passing | 2026-10-10 完成：AGENTS #33 全面修订（几何自动提取/MoE 公式/历史数字口径注记）、CHANGELOG AR005 节、指南 P3-4 ✅ + 第十二部分回链、README demo 样例 78.5%→85.5% 陈旧数字修正 |
| T005 | 全量回归 + 质量门：703 基线零回归、demo 基线、mypy（AGENTS 标准命令）/pyflakes/100 列 | T001–T004 | passing | 2026-10-10 完成：全量 738 passed / 3 skipped / 3 deselected（703 基线 + 35 新测试零回归）；mypy 标准命令 clean（65 文件，修复 7 处 Any 返回 via cast）；pyflakes clean；改动文件 100 列 clean；demo 基线 85.5% / 2.35e-03 / 93.75% 精确一致；coverage：flops_counter 99%、base.py 新增 property 全覆盖（文件 75% 为既有 get_device/estimate_memory 等行，与本 AR 无关） |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-10 开发会话（T001–T005 全部完成）

- T001：`count_diffusion_llm_flops` 5 个 MoE 参数 + per-token expert 公式 + 混合层数 + 域校验；UT-401~406 + EX-701~705 全绿（Red：12 failed TypeError 实证）。
- T002：`_extract_ffn_geometry`/`_resolve_model_config` 单一实现 + `model_ffn_flops_kwargs` 链解析重写 + `DiffusionLLM.config` 转发 property + 7 几何 property；UT-407~410 全绿（Red：16 failed 实证）；Red 子代理同步修正 test_flops_counter_ffn.py 旧"精确字典"断言为按键断言（键只增不改的必然结果）。
- T003：IT-411（engine tflops == 真实几何手算 ≠ 4h 默认）+ IT-412a（ablation `_flops_budget`）+ IT-412b（base_adapter `_estimate_baseline_tflops`）；三调用点零改动（设计核心收益）；生产接线在 T002 已成，T003 以验证测试直接落地（无 Red 态）。
- T004：AGENTS #33 修订、CHANGELOG AR005 节、指南 P3-4 ✅ + 第十二部分回链、README demo 样例 78.5%→85.5%；design 4.2.1 helper 流程图补 `_resolve_model_config` 下钻行（design 门控备忘）。
- T005：全量 738 passed / 3 skipped / 3 deselected；mypy clean（修复 base.py 7 处 Any 返回）；pyflakes/100 列 clean；demo 基线精确一致；flops_counter coverage 99%、base 新增 property 全覆盖。

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### 2026-10-10 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（G1–G9：7 YES + 2 WARN）
- 修复的问题：2 个 WARN 移交 design 阶段吸收——①§3.2 公式隐含全层 MoE（DeepSeek 系前 k 层 dense 会高估），须在 design 显式声明假设或加 `moe_layer_fraction` 注记；②`moe_top_k` 给了而 `moe_num_experts` 缺失时的校验行为未定义（ValueError 或全激活语义，design 定夺）
- 交叉验证：srs 现状描述 7/7 实证一致（helper 恒回退、三调用点行号、base 无 FFN 属性、underlying_model、指南 P3-4、demo 直调、703/demo 基线实测吻合）
- 审查代理：sdd-gate-reviewer

### 2026-10-10 design 门控记录

- 门控结果：PASS（第 2 轮）
- 第 1 轮：FAIL——1 Critical（真实子类 config 在 `self.model.config`，4.3.3 原设计 `getattr(self, "config")` 恒 None，主目标静默落空且 stub 测试会掩盖）+ 1 Major（`moe_num_experts<=0` 校验缺失）+ 3 Minor（流程图缺校验分支、T001 漏 moe_num_layers、两路径 config 前提未写明）
- 修复的问题：①4.3.3 新增 `config` 转发 property + `_resolve_model_config` 同链复用 + 优先级顺序（D9）；UT-408/409、IT-411 改为"几何仅在 `.model.config`"真实路径证明；②4.3.1 Raises + EX-702 补 `moe_num_experts<=0`；③流程图补 moe_num_layers/num_experts 校验分支；④T001/T002 描述同步；⑤tasks.md 头部更新
- 第 2 轮：全部修复项 YES（代码事实复核：causal_lm.py:63/generic.py:70、base.py:33/:175-179）；1 条非阻塞备忘（4.2.1 helper 流程图 `.model.config` 下钻一行，开发阶段顺手补）
- 审查代理：sdd-gate-reviewer

### 开发后合规审查（sdd-task-review）

- **第 1 轮：PASS**（12/13 YES；2 Minor D4 记录性问题，当场修复）
- S 维度 5/5 YES：§3.1–3.4 逐条测试映射核对（UT-401~410/EX-701~705/IT-411/412a/412b）；全量 738/3/3 + demo 85.5%/2.35e-03/93.75% + mypy 65 files clean + pyflakes clean + 100 列 0 违例（BS-601~603 实测吻合）；私有直测有据（design §4.4/§6.2 声明 + 公开接口等价断言）
- D 维度 3/4 YES：D1 签名逐参数一致；D2 算法逻辑一致（top_k 触发/回退链/混合层数/SwiGLU/shared+1/router 忽略）；D3 影响范围 git status 实证（源文件仅 flops_counter.py+base.py，四个零改动声明文件不在改动列表）
- C 维度 3/3 YES：新参数全默认零回归、键只增不改、config 无遮蔽（grep 实证）；AGENTS #10/#33 与实现逐点一致；风格全过
- 修复的 Minor：①tasks.md T002 状态 pending→passing（台账不一致）；②design §4.3.2 补记实现偏离（先钻透链 + 有界 range(8) 深度 + 仅链尾属性并查，当前无受影响类）
- 审查代理：general（只读审查）
