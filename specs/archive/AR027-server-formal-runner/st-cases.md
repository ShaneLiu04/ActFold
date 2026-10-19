# [AR027] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR027 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-19 |
| 执行环境 | Windows 11 / Python 3.11.9 / CPU 语义面（真实 GPU 正式跑由操作者按 RERUN 回填） |

> 执行方式说明：ST 以 srs.md 的黑盒验收标准为准。自动化用例引用
> `tests/test_ar027_formal_queue_contract.py` 中的合同测试（注入式 Probes /
> FakeRunner，不启动真实 bench）；CLI 与文档用例直接运行执行器命令。

## 测试用例列表

### ST-001：队列定义冻结（七面顺序/argv/工件）

**关联需求：** srs.md §3.1 FR1
**测试类型：** 正常路径
**优先级：** High

**前置条件：** 仓库可导入 `scripts.run_formal_queue`。

**测试步骤：**
1. 读取 `FACE_QUEUE`，核对 7 个面 ID 与顺序。
2. 核对每面 formal argv 与工件路径。
3. 核对 AR026 两面正式 argv 均含显式 `--out`。

**期望结果：** 7 面顺序/argv/工件与 srs §3.1 表一致；ar026 llada/dream 分文件。

**实际结果：** 通过（UT-2701，合同测试 66/66 全绿）。

**状态：** PASS

---

### ST-002：`--list` 只打印队列

**关联需求：** srs.md §3.1（`--list`）
**测试类型：** 正常路径
**优先级：** High

**前置条件：** 无 GPU/模型要求。

**测试步骤：**
1. 运行 `python -m scripts.run_formal_queue --list`。
2. 检查输出行数与内容；确认无子进程启动。

**期望结果：** 7 行队列预览，退出码 0，零子进程。

**实际结果：** 通过（UT-2702 / UT-2747；CLI 实测 7 行、exit 0、耗时 121ms、零子进程）。

**状态：** PASS

---

### ST-003：`--only` 子集与未知 ID

**关联需求：** srs.md §3.1（`--only`）
**测试类型：** 边界条件 / 异常处理
**优先级：** High

**前置条件：** 合同测试注入 FakeRunner。

**测试步骤：**
1. `--only ar026-dream` 只跑该面且保持队列顺序。
2. `--only unknown` → 退出码 3 且列出合法 ID。
3. `--only ""` → 退出码 3。

**期望结果：** 子集正确；未知/空 ID 参数错误。

**实际结果：** 通过（UT-2703 / UT-2704 / UT-2741；CLI 未知 ID 实测 exit 3）。注：PowerShell 会丢弃空字符串实参，`--only ""` 的验收由 UT-2741（直接调用 main）覆盖。

**状态：** PASS

---

### ST-004：`--models-root` 可选透传

**关联需求：** srs.md §3.1（`--models-root`）
**测试类型：** 正常路径
**优先级：** Medium

**测试步骤：**
1. 不传 `--models-root`，检查子进程 argv 不含该 flag。
2. 传入后检查每面 argv 追加 `--models-root <path>`（quick 与正式一致）。

**期望结果：** 仅在显式传入时透传。

**实际结果：** 通过（UT-2705 / UT-2706）。

**状态：** PASS

---

### ST-005：quick 模式 argv 与正式工件保护

**关联需求：** srs.md §3.1（quick）
**测试类型：** 正常路径 / 边界条件
**优先级：** High

**测试步骤：**
1. `--quick` 时检查每面 argv 含 `--quick`。
2. 检查 ar026 llada quick 省略正式 `--out`；dream quick 仅含 `_dream_quick.json`。
3. 预置正式工件后执行 quick，比较 sha256。

**期望结果：** quick 不触碰正式工件；ar026 双面 quick 路径正确。

**实际结果：** 通过（UT-2707 / UT-2708 / UT-2709 / UT-2738）。

**状态：** PASS

---

### ST-006：preflight — transformers 版本

**关联需求：** srs.md §3.2 FR2
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 注入 `transformers_version() == "5.0.0"` 执行。
2. 检查退出码与日志项。

**期望结果：** 退出码 3，日志含版本 FAIL 项，runner 0 次。

**实际结果：** 通过（UT-2710）。

**状态：** PASS

---

### ST-007：preflight — CUDA 不可用

**关联需求：** srs.md §3.2 FR2
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 注入 `cuda() == (False, 0)` 执行。

**期望结果：** 退出码 3，runner 0 次，无异常逃逸。

**实际结果：** 通过（UT-2711 / UT-2757）。

**状态：** PASS

---

### ST-008：preflight — 显存阈值（24 / 16 GiB）

**关联需求：** srs.md §3.2 FR2
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. 正式模式：23.9 GiB → 拒绝；24.0 GiB → 通过。
2. quick 模式：15.9 GiB → 拒绝；16.0 GiB → 通过。

**期望结果：** 阈值按 `>=` 判定；不足即 exit 3。

**实际结果：** 通过（UT-2712）。

**状态：** PASS

---

### ST-009：preflight — 权重文件缺失

**关联需求：** srs.md §3.2 FR2
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 注入任一 family 缺 `config.json` 或 `*.safetensors`。

**期望结果：** 退出码 3，detail 列出缺失 family。

**实际结果：** 通过（UT-2713 / UT-2756）。

**状态：** PASS

---

### ST-010：preflight — probe 工件缺失

**关联需求：** srs.md §3.2 FR2
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 注入 `probe_artifact_ok() == False` 执行。

**期望结果：** 退出码 3（不静默降级），runner 0 次。

**实际结果：** 通过（UT-2714）。

**状态：** PASS

---

### ST-011：preflight — GPU 静默与 `--allow-busy`

**关联需求：** srs.md §3.2 FR2
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. util=10% / used=1024MiB → 通过；util=10.1% 或 1025MiB → 拒绝。
2. GPU 忙 + `--allow-busy` → 继续且覆盖事实入日志/queue.json。
3. `nvidia-smi` 不可测 + `--allow-busy` → 仍拒绝。

**期望结果：** 仅 GPU 静默项可被覆盖；不可测不可豁免。

**实际结果：** 通过（UT-2715 / UT-2716 / UT-2745 / UT-2754）。

**状态：** PASS

---

### ST-012：models-root 解析顺序

**关联需求：** srs.md §3.2 FR2
**测试类型：** 正常路径 / 异常处理
**优先级：** Medium

**测试步骤：**
1. 缺省时从 probe `meta.models_root` 读取；probe 形态畸形时回退默认根。
2. 显式 `--models-root` 优先。

**期望结果：** `--models-root` > probe meta > 仓库 `ar023_models`；畸形不崩溃。

**实际结果：** 通过（UT-2706 / UT-2759）。

**状态：** PASS

---

### ST-013：子进程 argv、逐面日志与退出码

**关联需求：** srs.md §3.3 FR3
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 注入 FakeRunner 执行一面，核对命令为 `[sys.executable, "-m", script, ...]`。
2. 检查 `face-<id>.log` 首行 `argv: ...`、末行 `exit_code=0`。
3. 检查 queue.log 行带 UTC 时间戳。

**期望结果：** argv 完全一致；日志结构固定；queue.log 带时间戳。

**实际结果：** 通过（UT-2717 / UT-2746）。

**状态：** PASS

---

### ST-014：超时与启动异常

**关联需求：** srs.md §3.3 FR3
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 注入 runner 抛 `TimeoutExpired`（`--timeout-hours` 生效路径）。
2. 注入 runner 抛 `OSError`。

**期望结果：** 该面失败、exit 4、保留 stdout/stderr、后续 not_run。

**实际结果：** 通过（UT-2733 / UT-2734 / UT-2742）。

**状态：** PASS

---

### ST-015：status 值域校验

**关联需求：** srs.md §3.4 FR4
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 工件 `status` 缺失、`"failed"`、空串分别校验。

**期望结果：** 一律 mismatch（exit 2），不得静默放行。

**实际结果：** 通过（UT-2748）。

**状态：** PASS

---

### ST-016：逐面×逐状态 schema 表

**关联需求：** srs.md §3.4 FR4（schema 表）
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 7 面 ok 夹具逐面校验。
2. 降级 blocked（ar023 schema）与异常 blocked（各自 schema）逐面校验。
3. ar024 降级使用 ar023 schema；ar025 无 schema_version。

**期望结果：** 各状态期望 schema/version 全部 PASS。

**实际结果：** 通过（UT-2722 / UT-2723 / UT-2724）。

**状态：** PASS

---

### ST-017：h1/h2 重算与篡改检测

**关联需求：** srs.md §3.4 FR4（判定重算）
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 正常 ar023 夹具校验 → PASS。
2. 篡改记录 h1 verdict → mismatch，exit 2，日志含字段差异。
3. 已提交 `results/optimization/ar023_llada_bench.json` → PASS。
4. h1 映射层（`posthoc_confidence → posthoc_overall_mean`）重算一致。

**期望结果：** 重算使用冻结函数；篡改即失配；真实工件可通过。

**实际结果：** 通过（UT-2725 / UT-2726 / UT-2727）。

**状态：** PASS

---

### ST-018：ar026 参与度输入（禁止自证）

**关联需求：** srs.md §3.4 FR4（ar026 重算输入）
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 正常 ar026 夹具（顶层 `modes[*].engagement`）→ PASS。
2. 篡改顶层 engagement captures → mismatch。
3. 删除顶层 modes → mismatch（零捕获重算 inconclusive）。

**期望结果：** 重算只读顶层 engagement；verdict numbers 仅作比对面。

**实际结果：** 通过（UT-2764）。

**状态：** PASS

---

### ST-019：h4/reproduction 与 ar025 八键结构自洽

**关联需求：** srs.md §3.4 FR4（结构自洽）
**测试类型：** 正常路径 / 异常处理
**优先级：** High

**测试步骤：**
1. 校验 h4 非 dict、ar025 多余/缺失键、reproduction 非 bool。
2. 校验 ar025 精确八键正常面。
3. 降级 blocked 缺 boolean reproduction.ok → mismatch。

**期望结果：** 结构问题全部 mismatch；不扩展 schema。

**实际结果：** 通过（UT-2731 / UT-2743 / UT-2750 / UT-2751 / UT-2760）。

**状态：** PASS

---

### ST-020：工件缺失/坏 JSON/非对象

**关联需求：** srs.md §3.4 FR4（异常处理）
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 目标工件不存在。
2. 工件为非法 JSON。
3. 工件为 `[]`。

**期望结果：** exit 2，validation ∈ {missing, mismatch}，不写工件。

**实际结果：** 通过（UT-2735 / UT-2749 / UT-2736）。

**状态：** PASS

---

### ST-021：校验不写工件（sha256）

**关联需求：** srs.md §3.4 / NFR 诚实性
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 记录工件 sha256。
2. 执行校验（含 mismatch 面）。
3. 再次计算 sha256。

**期望结果：** 完全一致；执行器只读。

**实际结果：** 通过（UT-2736）。

**状态：** PASS

---

### ST-022：失败即停与 not_run

**关联需求：** srs.md §3.5 FR5
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 第 1 面返回非 0。
2. 检查第 2-7 面是否启动与 queue.json outcome。

**期望结果：** 立即停止；后续标记 not_run；不重试。

**实际结果：** 通过（UT-2718）。

**状态：** PASS

---

### ST-023：覆盖保护与 `--force` 审计

**关联需求：** srs.md §3.5 FR5
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. 目标已存在且无 `--force` → exit 4。
2. `--force` 无理由/空串 → exit 3。
3. `--force` + 理由 → 覆盖，queue.json 记 `overwrite=true` 与 `force_reason`。
4. 目标不存在时带 force → `force_reason=null`、`overwrite=false`。
5. 仅有 reason 无 force → exit 4。

**期望结果：** 覆盖需双条件；审计字段准确。

**实际结果：** 通过（UT-2719 / UT-2720 / UT-2721 / UT-2758 / UT-2739）。

**状态：** PASS

---

### ST-024：blocked 来源分类

**关联需求：** srs.md §3.5 FR5（blocked 分类）
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 合法降级（note / 前缀 / ar025 前缀）→ outcome=blocked，队列继续。
2. 异常 blocked（`real-face run failed:` / `RuntimeError:`）→ failure，exit 4。
3. ar025 诚实负结果（verdicts overall=blocked、reproduction.ok=true）→ blocked。
4. 缺 reproduction.ok / overall=supported / 降级理由+verdicts → failure。

**期望结果：** 合法 blocked 继续；无法归类一律失败。

**实际结果：** 通过（UT-2728 / UT-2729 / UT-2730 / UT-2740 / UT-2761 / UT-2762 / UT-2763）。

**状态：** PASS

---

### ST-025：status ok 但复现失败

**关联需求：** srs.md §3.5 FR5
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. `status="ok"` 且 `reproduction.ok == False` 执行。

**期望结果：** exit 4（不是 2），停止队列。

**实际结果：** 通过（UT-2732）。

**状态：** PASS

---

### ST-026：运行目录隔离与 queue.json 字段

**关联需求：** srs.md §3.6 FR6
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 连续两次运行。
2. 检查两个 `<UTC>` 目录与 queue.json/queue.log 字段。

**期望结果：** 目录互不覆盖；queue.json 含 run_id/argv/quick/allow_busy/force_reason/preflight/faces/overall。

**实际结果：** 通过（UT-2737 / UT-2717）。

**状态：** PASS

---

### ST-027：全绿与混合 blocked 业务场景

**关联需求：** srs.md §3.6 / design §6.3
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 7 面全绿 → exit 0、overall=ok、7 条 face。
2. 一面降级 blocked + 后续正常 → exit 0、overall=blocked、队列继续。

**期望结果：** 汇总语义正确。

**实际结果：** 通过（UT-2752 / UT-2753）。

**状态：** PASS

---

### ST-028：防御式输入（探针/嵌套/标量）

**关联需求：** srs.md §3.2 / §3.4 / design §6.4
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. 探针抛异常（model_files / vram）。
2. verdicts/quality/per_prompt/modes 嵌套畸形与标量类型错误。

**期望结果：** 探针异常 → preflight FAIL/exit 3；工件畸形 → mismatch/exit 2，无 traceback。

**实际结果：** 通过（UT-2755 / UT-2756 / UT-2757）。

**状态：** PASS

---

### ST-029：CPU-only 合同测试与运行时长

**关联需求：** srs.md §3.7 FR7 + NFR
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. `pytest tests/test_ar027_formal_queue_contract.py -q`。
2. 记录耗时；确认无 GPU/模型/网络依赖。

**期望结果：** 全绿且单文件 <30s。

**实际结果：** 通过：66 passed；pytest 内部 0.94s，进程墙钟 4.9s（含导入），无 GPU/模型/网络。

**状态：** PASS

---

### ST-030：FR→用例映射表存在且准确

**关联需求：** srs.md §3.7 FR7
**测试类型：** 正常路径
**优先级：** Medium

**测试步骤：**
1. 检查测试文件头部注释的 FR1..FR7 映射。
2. 抽查映射函数名在文件中存在。

**期望结果：** 映射完整、无悬空名。

**实际结果：** 通过（人工核对 66 个函数名与映射一致）。

**状态：** PASS

---

### ST-031：质量门 NFR

**关联需求：** srs.md §4 NFR
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. black --check / isort --check-only / pyflakes。
2. mypy actfold + mypy scripts。
3. pytest tests/ 全量。

**期望结果：** lint/type 零问题；全量测试全绿。

**实际结果：** 通过：black 168 文件零改动；isort/pyflakes 零；mypy actfold 68 + scripts 26 文件零；pytest 1179 passed / 4 skipped / 3 deselected（88.21s）。

**状态：** PASS

---

### ST-032：文档可执行性（RERUN §0/§7 与 CLI 一致性）

**关联需求：** srs.md 约束 + AGENTS Reproduction Guide Rule
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 按 RERUN §0 执行器的 `--list` 命令实跑。
2. 核对 §7 第 7 条/AR026 两条正式命令的 flag 与脚本 `--help`/`argparse` 定义一致。
3. 核对 AR026 正式/quick 工件命名与执行器 FACE_QUEUE 一致。

**期望结果：** 文档命令可直接执行；无 flag/路径漂移。

**实际结果：** 通过（CLI `--list` exit 0、7 行；`--list --force` 亦 exit 0；AR026 双面 `--out` 与 FACE_QUEUE/RERUN 一致）。

**状态：** PASS

---

### ST-033：执行器自身开销 <30s

**关联需求：** srs.md §4 NFR（开销）
**测试类型：** 边界条件
**优先级：** Medium

**测试步骤：**
1. 计时 `python -m scripts.run_formal_queue --list`。
2. 合同测试全量耗时作为 preflight+校验语义面参考。

**期望结果：** 远低于 30s（实验本身除外）。

**实际结果：** 通过：`--list` 实测 121ms；合同测试 66 例 pytest 内部 0.94s。

**状态：** PASS

---

## 执行摘要

| 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|
| 33 | 33 | 0 | 0 |

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-19 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR1 | 队列定义、quick、--only/--list、models-root | ST-001..ST-005, ST-012 | PASS |
| §3.2 FR2 | preflight 六项硬阻塞与 allow-busy | ST-006..ST-012 | PASS |
| §3.3 FR3 | 子进程执行、日志、超时、启动异常 | ST-013, ST-014 | PASS |
| §3.4 FR4 | schema 表、判定重算、结构自洽、只读校验 | ST-015..ST-021 | PASS |
| §3.5 FR5 | blocked 分类、失败即停、覆盖保护 | ST-022..ST-025 | PASS |
| §3.6 FR6 | 日志/退出码/目录隔离 | ST-026, ST-027 | PASS |
| §3.7 FR7 | CPU 合同测试与 FR→用例映射 | ST-028..ST-030 | PASS |
| §4 NFR | 质量门、开销、可移植性、诚实性 | ST-029, ST-031, ST-033 | PASS |

**需求覆盖率：** 8 / 8（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 13 | 13 | 0 | 0 |
| 边界条件 | 5 | 5 | 0 | 0 |
| 异常处理 | 13 | 13 | 0 | 0 |
| 回归测试 | 2 | 2 | 0 | 0 |
| **合计** | **33** | **33** | **0** | **0** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | 真实服务器七面正式执行（reps=30）不在本机 ST 范围 | 由操作者按 RERUN §7 执行并回填工件；执行器语义面已由 66 例合同测试覆盖 |
| Cosmetic | design 辅助函数名（recompute_verdicts/validate_structure）与实现名（_recompute_and_compare/_validate_structure）不同 | 记录，延后 |
| Cosmetic | `args_quick` 变量同时承载 quick/formal argv | 记录，延后 |

### 结论

> Go：需求覆盖 100%，无 Critical/Major 缺陷；真实服务器执行按预承诺由操作者回填。
