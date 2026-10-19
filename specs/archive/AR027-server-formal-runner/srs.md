# [AR027] 服务器正式跑执行器（server formal-run runner）需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR027 |
| AR 主题 | server-formal-runner |
| 关联 SR | 无（运维/复现基础设施，承接 AGENTS「Reproduction Guide Rule」） |
| 日期 | 2026-10-18（rev2，req 门控修复） |
| 状态 | Draft |

## 1. 背景与目标

AR023/AR024/AR025/AR026 的正式面（reps=30 预承诺单次）均需在 ≥24GiB 的高算力
服务器上执行，且当前每项都靠操作者手工拼命令、手工判读工件、手工记录日志。
风险有三：命令漂移（同一实验不同机器参数不一致）、环境不满足时白跑数十小时、
工件判定靠肉眼（可能漏读 blocked/inconclusive）。

本 AR 提供一个**服务器正式跑执行器** `scripts/run_formal_queue.py`：

- 一条命令按固定顺序执行全部正式面（可 `--only` 选子集、`--quick` 先冒烟）；
- preflight 硬阻塞不满足条件的机器（环境/权重/显存/GPU 静默度）；
- 子进程隔离执行，结构化日志（queue.log + queue.json）与区分退出码；
- 落盘工件按「逐面 × 逐状态」期望 schema 校验，并对可重算的冻结判定
  （AR023/AR024/AR026 的 h1/h2）做**重算比对**；h4/reproduction 做结构自洽校验
  （其输入未落盘，见 3.4）；
- 绝不自动重试、绝不 rerun-until-pass（预承诺单次纪律）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- 新增 `scripts/run_formal_queue.py`（模块方式运行）与其 CPU 合同测试
- 七个正式面：AR023-llada、AR023-dream、AR024、AR025-real-llada、
  AR025-real-dream、AR026-llada、AR026-dream
- 工件校验：逐面 × 逐状态 schema 校验 + 判定重算/结构自洽 + blocked 来源分类
- 文档：`docs/RERUN_CHECKLIST.md` §0/§7 执行器入口与 AR026 双面工件命名统一；
  AGENTS 记录使用约束

**Out of Scope（本 AR 不做的）：**
- 不修改任何 bench 脚本（判定函数与工件合同保持冻结；执行器只 import 复用）
- 不扩展 artifact schema（因此 h4/reproduction 不重算，只做结构自洽）
- 不实现自动重试、自动续跑、失败后继续队列
- 不做分布式/多机调度、不做 GPU 锁频（由操作者按 RERUN §1 执行）
- 不在本机执行任何正式跑（本 AR 只交付执行器与 CPU 测试）

## 3. 功能需求

### 3.1 队列定义与选择（FR1）

**描述：** 固定七个正式面及其命令、工件路径。

**触发条件：** 执行 `python -m scripts.run_formal_queue`。

**期望行为：**
- 默认按固定顺序执行全部七面（`<root>` 仅在操作者传入 `--models-root` 时
  追加该 flag；未传入时**不追加**，交还各 bench 的默认解析
  `probe meta.models_root → "ar023_models"`，禁止硬编码平台路径）：

| 序 | 面 ID | 命令（子进程，正式模式） | 正式工件 |
|---|---|---|---|
| 1 | ar023-llada | `-m scripts.ar023_cross_family_bench --family llada --reps 30` | `results/optimization/ar023_llada_bench.json` |
| 2 | ar023-dream | 同上 `--family dream` | `results/optimization/ar023_dream_bench.json` |
| 3 | ar024 | `-m scripts.ar024_granularity_bench --family llada --reps 30` | `results/optimization/ar024_llada_granularity_bench.json` |
| 4 | ar025-real-llada | `-m scripts.ar025_multi_parent_bench --mode real --family llada --reps 30` | `results/optimization/ar025_real_bench_llada.json` |
| 5 | ar025-real-dream | 同上 `--family dream` | `results/optimization/ar025_real_bench_dream.json` |
| 6 | ar026-llada | `-m scripts.ar026_real_graph_bench --family llada --reps 30 --out results/optimization/ar026_real_graph_bench.json` | 同左 |
| 7 | ar026-dream | 同上 `--family dream --out results/optimization/ar026_real_graph_bench_dream.json` | 同左 |

> AR026 的 `--out` 在正式模式必须显式分文件：bench 默认正式路径不按 family
> 分键（`ar026_real_graph_bench.py` DEFAULT_OUTPUT_PATH），后跑的 dream 会覆盖
> llada。该命名同时写入 RERUN（T008）。

- **quick 模式（`--quick`）**：每面追加 `--quick`，且**不得**传正式 `--out`
  （AR026 的 `--out` 优先级高于 quick，会写正式路径）。quick 工件固定为：

| 面 ID | quick 工件 |
|---|---|
| ar023-llada / ar023-dream | `ar023_{family}_bench_quick.json`（bench 默认） |
| ar024 | `ar024_llada_granularity_bench_quick.json`（bench 默认） |
| ar025-real-* | `ar025_real_bench_{family}_quick.json`（bench 默认） |
| ar026-llada | `ar026_real_graph_bench_quick.json`（bench 默认，省略 `--out`） |
| ar026-dream | 显式 `--out results/optimization/ar026_real_graph_bench_dream_quick.json` |

- `--only <逗号列表>`：按给定面 ID 子集执行（保持队列内相对顺序）；
  `--list`：只打印队列（ID/命令/工件）不执行，退出码 0。
- `--models-root PATH`（可选）：仅当传入时给每面追加 `--models-root PATH`。

**异常处理：** `--only` 含未知面 ID → 参数错误（退出码 3），错误信息列出合法 ID。

**验收标准：**
- Given 无 `--models-root`, When 执行任一正式面, Then argv 中**不含**
  `--models-root`。
- Given `--quick`, When 执行 AR026 两面, Then argv 不含正式 `--out`；
  dream 面 argv 仅含 `--out ..._dream_quick.json`；正式工件的 sha256
  与执行前一致（未被新建或修改）。
- Given 仅传 `--list`, When 执行, Then 输出 7 行且不执行任何子进程。
- Given `--only ar026-dream`, When 执行, Then 只跑该面。

### 3.2 preflight 硬阻塞（FR2）

**描述：** 模型加载前的机器条件检查。

**触发条件：** 队列执行开始前。

**期望行为（全部硬阻塞）：**

| 检查 | 条件 | 适用面 |
|---|---|---|
| transformers | `transformers.__version__ == "4.53.1"` | 全部 |
| CUDA | `torch.cuda.is_available()` 且设备数 ≥1 | 全部 |
| 显存 | `torch.cuda.get_device_properties(0).total_memory >= 24 GiB`（`--quick` 时 ≥16 GiB） | 全部 |
| 权重 | `<root>/<family>/config.json` 且目录下 ≥1 个 `*.safetensors` | 该 family 的面 |
| probe 工件 | `results/optimization/ar023_family_probe.json` 存在 | **全部 7 面**（四个 bench 都读 probe 并据此降级） |
| GPU 静默 | `nvidia-smi --query-gpu=utilization.gpu,memory.used`：util ≤10% 且 used ≤1024MiB | 全部 |

- `--allow-busy`：仅覆盖「GPU 静默」一项；覆盖事实写入 queue.log 与 queue.json。
- `<root>` 解析顺序：`--models-root` > probe 工件 `meta.models_root` >
  `ar023_models`（相对仓库根），与 bench 的默认解析一致。
- 所有探测通过可注入函数（`probe_*` 参数）实现，便于 CPU 合同测试。

**异常处理：** 阻塞时打印逐项 PASS/FAIL 表与修复指引（引 RERUN §0/§7），
退出码 3，不写工件、不启动任何子进程。`nvidia-smi` 不可用视为阻塞。

**验收标准：**
- Given transformers=5.0.0（任意 ≠4.53.1）, When 执行, Then 退出码 3 且日志含版本项。
- Given GPU util=99%, When 执行, Then 退出码 3；加 `--allow-busy` 则继续且覆盖事实入日志。
- Given AR023 面且 probe 工件缺失, Then 退出码 3（不再静默降级）。

### 3.3 子进程执行与冒烟模式（FR3）

**描述：** 每面用当前解释器子进程执行，输出落盘。

**期望行为：**
- 子进程：`[sys.executable, "-m", "scripts.<bench>", ...]`，cwd=仓库根，
  环境变量原样继承（HF 端点/缓存由操作者环境变量控制）。
- 每面日志文件 `face-<id>.log` 结构固定：首行 `argv: <完整命令>`，随后为
  子进程原始输出，末尾行 `exit_code=<n>`（queue.log 同步逐行带 UTC 时间戳）。
- `--timeout-hours <float>`（可选）：单面超时 → 该面失败（退出码 4），不重试；
  默认无超时。
- 子进程退出码非 0 → 该面失败（退出码 4）。

**异常处理：** 子进程启动异常（模块不存在/解释器错误）→ 该面失败并记录异常。

**验收标准：**
- Given 注入假 runner, When 执行一面, Then 命令 argv 与预期完全一致，且
  `face-<id>.log` 首行为该面完整 argv、末段含 `exit_code=<n>` 行。
- Given `--quick`, When 执行, Then 所有 argv 含 `--quick` 且正式工件未被修改。

### 3.4 工件校验（FR4）

**描述：** 每面成功后校验工件，防止「跑完但结果不可信」。

**期望行为：**
1. **status 值域**：工件 `status` 只接受 `ok` 与 `blocked`；缺失或其它值
   一律 mismatch（不得静默放行）。
2. **期望 schema 表（逐面 × 逐状态）**：

| 面 ID | status=ok 的 schema(+version) | probe 降级 blocked 的 schema(+version) | 运行异常 blocked 的 schema(+version) |
|---|---|---|---|
| ar023-* | `ar023_cross_family_bench` v1 | `ar023_cross_family_bench` v1（`build_blocked_artifact`） | `ar023_cross_family_bench` v1（exit 1，异常被捕获写工件） |
| ar024 | `ar024_granularity_bench` v1 | `ar023_cross_family_bench` v1（复用 ar023 builder） | `ar024_granularity_bench` v1（exit 1） |
| ar025-real-* | `ar025_real_bench`（**无** schema_version） | `ar025_real_bench`（无 version；无 verdicts，顶层 `blocked_reason`） | `ar025_real_bench`（无 version，exit 0；无 verdicts，`real-face run failed:`） |
| ar025-real-*（诚实负结果） | — | — | — |
| ar026-* | `ar026_real_graph_bench` v1 | `ar023_cross_family_bench` v1（复用） | `ar026_real_graph_bench` v1（exit 1） |

> 「诚实负结果」行：`ar025_real_bench`（无 version）、`status="blocked"`、
> **含完整 verdicts**（overall=blocked）、无顶层 `blocked_reason`——
> 这是正常跑完的冻结判定结果，不是失败（FR5 归类为合法 blocked）。

3. **判定重算比对（单一真源）**：
   - AR023/AR024：import `scripts.ar023_cross_family_bench` 的
     `h1_quality_verdict` / `h2_speed_verdict`，用工件记录的 face 数据重算，
     与记录值比对。
   - AR026：import `scripts.ar026_real_graph_bench.h1_graph_parity_verdict`
     与 ar023 的 `h2_speed_verdict` 做同样重算比对。
   - **重算输入口径**：
     - h2 输入取 `verdicts.h2.numbers.p50_baseline_ms` / `p50_aligned_ms`；
       `aligned_vs_batched_ratio` 的输入 `p50_batched_ms` 未落盘 →
       该字段列为 **report-only，不重算**（不扩展 schema）。
     - h1（ar023/ar024）输入取 `quality.per_prompt[*]` 的
       `posthoc_confidence[mode].overall_mean`、`block_exact`、`decode_length`
       与 `quality` 聚合面（与 bench 组装 h1 时的取值一一对应）。
     - ar026 h1_graph_parity 输入取工件的两类独立记录：`quality` 面与
       顶层 `modes[*].engagement`（`captures/replays/evictions`；缺失或非
       dict 时该 mode 记 `{}`，重算因零捕获判 `inconclusive`）。参与度门由
       冻结函数从 engagement 的 `captures` 推导；
       `verdicts.h1_graph_parity.modes[*].numbers` 只作为**被比对面**，
       不参与重算输入（禁止自证）。
   - **比对口径**：
     - `verdict`/布尔/计数类字段**严格相等**；
     - 未取整的输入数值字段 `math.isclose(rel_tol=1e-9, abs_tol=1e-6)`；
     - **bench 以固定小数位落盘的派生字段**（如 h2 的
       `speedup_p50_vs_baseline` 为 4 位），重算后按**同一小数位 round**
       再严格相等（这不是语义放宽：记录精度即 bench 的判定精度）。
   - **判定结构校验只对含 verdicts 的工件执行**（status=ok，或 ar025 的
     overall=blocked 负结果变体）；无 verdicts 的 blocked 工件只做 schema +
     blocked 来源分类。
   - **h4 / reproduction 不重算**（canary/ledger 与 token 行未落盘，扩展
     schema 属 Out of Scope）：改为**按真实字段的结构自洽校验**——
     - ar023/ar024/ar026：`reproduction` 必须含布尔 `ok`；降级 blocked 时
       必须含 `skipped: true` 与非空 `note`。**结构校验不强制成功工件的
       `ok: true`**——`status ok` 且 `reproduction.ok == False` 由 FR5 判为
       失败（退出码 4）。
     - 判定值域：`verdict` ∈ {supported, refuted, inconclusive, blocked}
       （ar023/ar026）；`pass` 为布尔（ar025 real 的 H1/H2/H4）。
     - 与 `status` 一致：blocked 工件不得含 supported 判定。
   - AR025 real 面：校验 H1/H2/H4/delivery/canary 判定结构完整、
     `blocked_reasons` 与 verdict 值自洽（blocked 必有非空原因；非 blocked
     必有数值面）；canary 允许 `timed_count: null` + 非空 `note`（结构化
     降级面，不要求 `pass` 字段）；S1-S3 属 synth 面，不在本执行器范围。
4. **blocked 语义**：
   - **降级/异常 blocked**（无 verdicts）：必须携带非空顶层 `blocked_reason`，
     且不得包含伪造数值面（`modes`/`quality` 为 null）。
   - **诚实负结果 blocked**（含完整 verdicts）：豁免顶层 `blocked_reason`，
     自洽条件为 `verdicts.blocked_reasons` 非空且判定值不出现 supported。
5. 任何不一致 → 退出码 2，日志打印差异字段（不修正工件）。唯一例外：
   `status=="ok"` 且 `reproduction.ok == False` 由 FR5 判为退出码 4，
   满足“单一失败语义、无 2/4 歧义”。

**异常处理：** 工件缺失/JSON 解析失败 → 退出码 2。

**验收标准：**
- Given 假工件 h1 记录与重算不一致, When 校验, Then 退出码 2 且日志含字段差异。
- Given 测试夹具构造的 7 面正常工件（ok/supported 或合法判定）, When 逐面校验, Then PASS。
- Given results/optimization 中实际存在的已提交工件（如 ar023_llada_bench.json）, When 校验, Then PASS；不存在的面不在本用例内。
- Given ar024 的 probe 降级工件（ar023 schema + reproduction.note）, Then schema 校验 PASS（按表）。
- Given 校验前后对比工件文件 sha256, Then 完全一致（执行器不写工件）。

### 3.5 失败语义、blocked 分类与覆盖保护（FR5）

**描述：** 预承诺单次纪律的执行层落实。

**期望行为：**
- **blocked 来源分类**（决定是否算失败）：
  - 合法 blocked（probe 降级）：`status=="blocked"` 且
    （ar023/ar024/ar026：`reproduction.get("note") == "degraded family per probe conclusion"`；
    ar025：`blocked_reason` 以 `family degraded per probe` 开头）
    → 队列继续，queue.json 记 `outcome="blocked"`。
  - 合法 blocked（**诚实负结果**）：ar025 `status=="blocked"` 且含完整
    verdicts、无顶层 `blocked_reason`（冻结判定 overall=blocked）→ 队列继续，
    `outcome="blocked"`。
  - 异常 blocked（运行失败）：
    `blocked_reason` 以 `real-face run failed:` 开头（ar025）、或
    `blocked_reason` 形如 `<ExceptionType>: ...`（ar023/ar024/ar026）、或
    `reproduction.note` 缺失、或无法归类 → **失败**（退出码 4，停止队列）。
- 子进程退出码非 0 → 失败（退出码 4）。
- **status ok 但 `reproduction.ok == False`**（双跑复现失败）→ 失败
  （退出码 4），停止队列；该分支优先于结构校验（结构校验不重复判定，
  见 FR4 第 4 条）。
- 某面失败 → **立即停止后续队列**，queue.json 记录
  completed/failed/not_run，非零退出。不自动重试。
- 目标工件（--quick 时为 quick 工件，否则为正式工件；quick 永不检查/覆盖正式工件）已存在 → 执行前拒绝该面（退出码 4），除非 `--force`
  且必须同时提供非空 `--force-reason "<理由>"`；理由写入 queue.json
  （「覆盖」事实 + 理由，供事后审计）。

**验收标准：**
- Given 第 1 面失败, Then 第 2-7 面未启动且 queue.json 标记 not_run。
- Given 目标工件已存在且无 `--force`, Then 拒绝并退出码 4。
- Given 目标工件已存在、`--force` 但 `--force-reason` 缺失或为空串, Then 参数错误退出码 3。
- Given ar025 异常 blocked（reason 以 `real-face run failed:` 开头）,
  Then 判失败并停止队列。

### 3.6 结构化日志与退出码（FR6）

**描述：** 服务器无人值守场景的可审计记录。

**期望行为：**
- 运行目录：`results/optimization/formal_runs/<UTC 时间戳 YYYYmmddTHHMMSSZ>/`，
  内含 `queue.log`（人类可读，逐行 `[UTC ISO8601] 内容`）与 `queue.json`
  （结构化摘要，字段固定：`run_id`、`argv`、`quick`、`allow_busy`、`force_reason`、
  `preflight`（逐项 name/ok/detail）、`faces[]`（id/argv/start/end/duration_s/
  exit_code/artifact/validation（pass|mismatch|missing|null）/outcome）、`overall`）。
- 每次运行新建目录，**不跨运行追加**；同一运行内 queue.log 只追加。
- 退出码：`0` 全通过（`overall` ∈ {ok, blocked}）；`2` 工件校验不一致；
  `3` preflight 阻塞或参数错误；`4` 实验失败或覆盖保护拒绝。
- **任何非 0 退出（2/3/4）均立即终止后续面**（3 发生在任何子进程启动前）。

**验收标准：**
- Given 全部面通过, Then 退出码 0 且 queue.json 含 7 条 face 记录。
- Given 校验不一致, Then 退出码 2 且 queue.json 记 validation mismatch。
- Given 两次运行, Then 生成两个不同 `<UTC>` 目录且互不覆盖。

### 3.7 CPU 合同测试（FR7）

**期望行为：** 新增 `tests/test_ar027_formal_queue_contract.py`，覆盖：
- 队列定义冻结（7 面顺序/argv/工件路径；models-root 可选透传；AR026 正式
  双面显式 `--out`；quick 模式不产生正式文件）；
- `--only`/`--list`/未知 ID 语义；
- preflight 各阻塞项（注入探针）与 `--allow-busy` 覆盖；
- 子进程 argv 构造（注入 runner，不真正执行 bench）；
- 判定重算：正常/篡改/结构不完整；
- 期望 schema 表（逐面 × 逐状态）；
- blocked 分类（合法降级 vs 异常）；
- 失败即停、覆盖保护（`--force` + `--force-reason`）；
- 退出码 0/2/3/4 与 queue.json 结构。
- 测试文件内提供 **FR→用例映射表**（注释形式：FR1..FR7 各列出对应测试名）。

**验收标准：**
- Given 完整测试集, When `pytest tests/test_ar027_formal_queue_contract.py`, Then 全绿且全程无 GPU/模型/网络依赖。
- Given 执行器测试注入缺失 CUDA, Then 无异常逃逸且 preflight 按预期拒绝。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 可测试性 | CPU 合同测试 | 新增测试全绿，单文件本地执行 < 30s，无 GPU/模型/网络 |
| 覆盖 | 关键分支 | 新增测试覆盖 FR1-FR6 的每条验收标准（测试文件内附 FR→用例映射表） |
| 质量门 | lint+types | black(100)/isort/pyflakes 零问题；`mypy actfold --ignore-missing-imports` 与 `mypy scripts --ignore-missing-imports --explicit-package-bases` 零问题；`pytest tests/` 全绿 |
| 可移植性 | 平台 | Windows 与 Linux 均可运行；CI（ubuntu-latest）必须跑新增测试 |
| 开销 | 执行器自身 | preflight 与校验合计 < 30s（不含实验本身） |
| 诚实性 | 零伪造 | 执行器不写任何工件：校验前后目标工件文件 sha256 不变（测试断言） |

## 5. 约束与假设

**约束：**
- 不得修改任何 bench 脚本的判定函数或工件 schema（冻结合同）。
- 不得引入自动重试/续跑；不得在测试中真正启动 GPU 实验。
- 正式跑必须由操作者显式执行；CI 只跑 CPU 合同测试。
- `--force` 仅表示「操作者的显式新承诺」，不是重试机制；理由必填。

**假设：**
- 服务器环境为 RERUN §0 的 AR019 venv（transformers 4.53.1 + 系统 torch）。
- 模型目录布局 `<models-root>/<family>/{config.json,*.safetensors}`。
- 已提交的 AR023 probe 工件随仓库分发（tracked）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 正式面（formal face） | reps=30 预承诺单次的实验执行单元 |
| 冻结判定（frozen verdict） | 实验脚本内预先声明、不可事后调整的 H/S 判定函数 |
| 合法 blocked | probe 判定家族降级导致的类型化结果（零伪造，队列可继续） |
| 异常 blocked | 运行异常被 bench 捕获后写出的 blocked 工件（判失败） |
| preflight | 模型加载前的机器条件硬检查 |
