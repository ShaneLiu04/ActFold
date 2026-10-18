# [AR026] 任务清单

**AR026：真实 checkpoint 端到端图路径**（srs.md 同目录；本文件为唯一任务真源）

## 任务清单

| 编号 | 任务 | 状态 | 依赖 | 门 |
|------|------|------|------|-----|
| T001 | design.md 冻结：裸前向断言根因定位计划（静态体 vs 裸前向算子序列差异审计；分配器/WDDM/remote sdpa 交互假设树）；Dream 形状断裂「探针构造 vs 生产」判定设计；FR1-FR5 映射到实现点 | passing | — | design 门 |
| T002 | FR1 Red→Green：`safe_capture` + `CaptureDiagnostics`（cuda_graph.py，primary `__context__` 链保真 + best-effort 复位 + 失败重抛原异常）；三捕获点（serial/batched/adaptive）统一路由，降级语义字节不变；`graph_pool_stats.capture_diagnostics` 只读探针键。UT：tests/test_ar026_capture_safety.py（UT-2601/2602 + OOM/复位失败变体，CPU mock） | passing | T001 | Red→Green |
| T003 | FR3：tuple-unwrap Caveat 重审落地——tests/test_ar026_graph_unwrap.py（UT-2603 静态体直接调用 CPU 等价 + UT-2604 CUDA 门重放 CPU skip）；AGENTS #45 文本修正（钉 UT-2603/2604 引用使其可溯源） | passing | T001 | Red→Green |
| T004 | FR4：探针 graph face 生产化——`ar023_family_probe` graph face 改走 MFF+FoldedGraphRunner 生产捕获；`graph_enabled` 语义对齐；脚本 lint 三清 | passing | T002, T003 | lint+合同 |
| T005 | FR5：四模式实机基准脚本 `scripts/ar026_real_graph_bench.py`（导入式复用 ar023 合同件先例）——baseline 官方配方 / folded-batched eager（AR023 锚）/ graph 固定 C / graph 自适应（模式间仅图旋钮差异）；预注册判定式冻结；VRAM 余量面；schema `ar026_real_graph_bench`；UT：tests/test_ar026_bench_contract.py（UT-2606 + UT-2605 探针合同） | passing | T002 | lint+合同 |
| T006 | GPU quick smoke：真实 LLaDA-8B 生产捕获首验（FR1 实机面）——失败签名/降级面/VRAM 余量记录；**Dream 面显式交付（FR2）**：生产 Dream 批量捕获验证工件，或 blocked 工件携带最小复现签名（零伪造）；**FR1 失败分支预案**：生产捕获若同样触发分配器断言，修复迭代回路归属本任务 Red→Green 循环或落类型化降级结论（design 期定案） | passing | T002-T005, GPU 空闲 | smoke |
| T007 | GPU formal：reps=30 预承诺单次四模式基准（AR023 H 面全承）→ `results/optimization/ar026_real_graph_bench.json` | pending | T006 | formal |
| T008 | 质量门：图重放 vs eager allclose 容差面 + 质量双指标 + 账本/canary + VRAM 面判定；诚实负结果义务（inconclusive/blocked 原样交付） | pending | T007 | 质量门 |
| T009 | 文档全量更新：README Full-Update Rule（优越性阶梯表新增真实端到端档）+ AGENTS #45 修正入库 + GUIDE 新部分 + CHANGELOG | pending | T008 | docs |
| T010 | 审查：spec/design 合规 + 代码审查（含本 AR 全 UT pytest 形式复跑，CUDA 可见） | pending | T009 | review |
| T011 | ST 验收 + 归档（**串行约束：AR024/AR025 先归档**） | pending | T010 | ST+归档 |

## 排队约束（进度记录）

- **GPU 串行队列**（共享单卡 + T006 采样期禁 CUDA）：
  `AR023-T006（✅ 10-17 17:49 完成，工件落盘）→ AR023-T007（▶ 运行中，dream formal，17:49 起）→ AR024-T005/T006 → AR025 实机面 → AR026-T006/T007`
- 本 AR CPU-first 排序：T001-T005 全部 CPU 可完成（T002/T003 UT 以 mock/静态体直接调用承载）；T006+ 为实机面。
- **归档次序约束**：T011 必须晚于 AR023 T007 结档、AR024 T011 与 AR025 归档（AR 编号串行结档纪律）。

### 2026-10-17 夜间考证：队列状态 + venvlauncher 机器陷阱（T006 前置必读）

- **AR023-T006（llada formal）完成**：`results/optimization/ar023_llada_bench.json`
  （未跟踪新工件）——诚实混合判定：**h1 refuted**（llada num_tokens=128 下
  folded 投机质量差距真实存在：baseline 早停 3/18 vs folded 全 128，posthoc
  gap -0.23/-0.29）、h2 supported（p50 加速 1.49×）、h4 supported；复现双跑
  严格 PASS、cross-rep deterministic。**对本 AR 的含义**：AR026 锚
  （folded-batched）在 llada formal 上与 baseline 存在真实质量差距——我的
  h1_graph_parity（graph vs 锚）设计不受影响，但 T008 阶梯/对账面必须对
  **此**工件对账（不得沿用 quick 的 4.53.1 环境小几何印象；quick 64 tokens
  时 folded 早停/baseline 全长，formal 128 tokens 时角色反转——canvas 长度
  敏感是扩散模型本性，非不一致）。
- **AR023-T007（dream formal）运行中且协议有效**：`.cmd` 行 6 经
  `ar019_venv\Scripts\python.exe` 启动。**机器陷阱（已实证）**：该 venv
  python.exe（274KB）是 **venvlauncher**——启动基础 `C:\Python311\python.exe`
  作为子进程并注入 venv 环境；故 WMI 的 ExecutablePath/CommandLine 显示
  基础路径（误导），真实环境 = venv transformers 4.53.1（父进程链验证：
  5832 基础 python ← parent 21892 venv python = .cmd 调用）。**判据：进程
  父链（ParentProcessId），永不凭 ExecutablePath 判定环境**。此前笔记中
  「21892 孤儿无害」**勘误**：21892 = dream formal 的 launcher 父进程，
  活跃关键进程，绝不可清；11580/17832/21904 仍为 multiprocessing-fork
  孤儿（parent 22584 已亡），无害。
- **AR023 未归档**（specs/archive/ 无 AR023 目录；T006/T007 formal 为其结档
  前置）——AGENTS #45 的工件引用先于 formal 完成，h1 诚实负结果（refuted）
  归 AR023 会话处置，本 AR 不越权。
- **时长核算（T006/T007 排程输入）**：llada formal 单样本 p50 ≈ 330s
  （128 tokens，llama-server 竞争 + WDDM）；30 reps 总耗时 ≈ 29.2h
  （10/16 12:35 → 10/17 17:49）。AR026 quick smoke（reps=3/1 prompt/
  64 tokens ≈ 18 样本）预估 ~1-2h；**T007 formal（reps=30/2 prompts/
  128 tokens ≈ 240 样本）预估 30h+**——须在预承诺单次跑纪律下显式排程
  （接受多日窗口），不得因时长改协议。

### T006 Runbook（GPU 窗口执行清单，冻结于 T005 完成夜）

**前置检查**（全部满足才开工）：
1. dream formal 完成：`ar023_t006_formal.log` 尾部出现
   `T006 formal runs complete`；进程 21892（venv launcher 父）/5832（子）
   均已退出——**凭父链验证，永不凭 ExecutablePath**（venvlauncher 陷阱）。
2. 无其他 CUDA 消费者（`Get-CimInstance Win32_Process -Filter "Name='python.exe'"`
   逐个查 CommandLine）。
3. 运行环境：`C:\Users\l30086046\AppData\Local\Temp\opencode\ar019_venv`
   （transformers 4.53.1，RERUN_CHECKLIST 钉版）。

**Step 1 — 探针生产 graph face 首验（子进程隔离，~30min）**：
```
& <venv>\Scripts\python.exe -m scripts.ar023_family_probe `
    --models-root D:\ar023_models `
    --output results/optimization/ar026_family_probe_rerun.json
```
读数：`graph` 面（T004 生产化驱动器：captures>0 且零失败诊断 → ok）；
`graph_raw`（诊断面，对照旧失败签名）；headroom（>1024MB 门）。
**禁止写默认输出路径**（tracked AR023 结论工件，覆盖风险）。
注：tracked probe 工件仍是 T004 前旧格式（graph={ok,error} 裸捕获失败）——
生产面从未实机跑过，本步即 FR1 首验。

**Step 2 — AR026 bench quick smoke llada（~1-2h）**：
```
& <venv>\Scripts\python.exe -m scripts.ar026_real_graph_bench `
    --family llada --quick --models-root D:\ar023_models
```
→ `results/optimization/ar026_real_graph_bench_quick.json`。
读数：engagement（graph-fixed/graph-adaptive captures>0）；
`h1_graph_parity`（engagement 门 + 三面）；canaries（timed 窗
degradation_fallback_count 增量==0）；meta 的 VRAM 余量面 +
capture_diagnostics（safe_capture 失败记录，首验分配器断言是否在
生产路径复现）。

**Step 3 — Dream 面显式交付（FR2）**：
```
& <venv>\Scripts\python.exe -m scripts.ar026_real_graph_bench `
    --family dream --quick --models-root D:\ar023_models `
    --out results/optimization/ar026_dream_graph_quick.json
```
交付物：生产 Dream 批量捕获验证工件，或 blocked 工件携带最小复现
签名（零伪造，srs FR2 原文）。

**决策树**：
- 双族捕获 OK → T007 formal 排程（llada 先行；30h+ 预承诺单次窗口；
  启动记录沿用 ar023 .cmd+log 先例；期间禁一切 CUDA 工作）。
- 生产路径复现分配器断言 → **T006 拥有修复回路**：capture_diagnostics
  primary 链 → 根因定位 → Red→Green 迭代；不可修 → 类型化降级结论
  （诚实负结果，AR021 H2 先例）。
- OOM → 类型化降级 + VRAM 记录（16GB 卡 bf16-8B ≈15GB + 图池捕获侧
  VRAM 是已知紧约束，GEOMETRY_INHERITANCE_NOTE 已预注册）。
- 纪律：预承诺单次跑——失败按决策树处置，**绝不 rerun-until-pass**。

### 2026-10-18 午后：main() 接线冒烟（T005 补强证据）+ 队列 ETA

- **main() CPU 冒烟 ×2（Runbook Step 2 的精确接线预验证）**：
  1. **CUDA 错误路径**：`--quick --out <tmp>` CUDA 隐藏 →
     `run_graph_bench` raise → blocked 工件落盘（status blocked /
     modes=quality=None / verdicts={} / blocked_reason=
     "RuntimeError: run_graph_bench requires a CUDA device" /
     repro skipped 标记）——**零伪造合同端到端成立**。退出码 0xC0000005
     为已知机器级 teardown AV（工作全部完成后才发生，工件完好），
     非脚本缺陷；Runbook 注意：CUDA 隐藏时退出码不可作失败信号。
  2. **degraded 族路径**：合成 degraded probe（复用 tracked 工件改
     conclusions）→ exit 0 + blocked 工件携带 degrade 原因与 probe 引用，
     无 torch 导入、无 AV。
  3. **附带确认**：tracked 旧格式 probe 工件（T004 前版式）与本 bench
     `probe_conclusion` 兼容（llada degraded=False 正确放行至 CUDA 检查）
     ——Runbook Step 2 默认 probe 引用无需变更。
  4. **llada formal 短解码政权兼容性审计**（CPU，读 formal 工件）：
     baseline 早停 len 3/18 时 posthoc 窗（≥1 token）无零除、
     block_exact common=1 起合法——本 bench 质量面在 formal 政权下
     无边界缺陷，零代码改动。
- **AR023-T007 ETA 精化**：dream formal rep 7/30 @10-18 12:36，rep 间隔
  ≈2h+ → 余 23 reps ≈ 46h+ → **预计 10/20 午后完成**；期间 GPU 持续
  占用，AR024-T005/T006 与本 AR T006 顺延。冒烟工件在
  `C:\Users\l30086046\AppData\Local\Temp\opencode\ar026_main_smoke.json` /
  `ar026_degraded_smoke.json`（临时，不入库）。
- **文档诚实性审计（T009 前置考证）**：GUIDE Part 30（L808）明确
  「后续 T006 完成后写入」正式工件数字——docs 未预支 formal 结论；
  README 无任何 AR023 引用（其全量更新待 AR023 结档）。llada formal
  h1-refuted 与现有文档**零矛盾**，处置归属 AR023 会话；本 AR T009
  重写时以 AR023 结档后的最新文档为基线。

## 进度记录

### 2026-10-17（需求期，会话：AR024 收尾 + 全自动下一创新指令）

- **选题依据**：Part 25 遗留三选一——变长（未做）、多 parent（AR025 在途）、真实
  checkpoint 端到端（本 AR）；与「框架级优越性」总纲最契合（AR018 2.40-2.95× 全
  合成证据，真实 8B 图路径从未打通）。
- **需求期代码审计（CPU-only，符合「先不做实验」指令）**：
  1. probe 工件 graph face 签名提取：llada
     `captures_underway.empty() INTERNAL ASSERT (CUDACachingAllocator.cpp:2967)`；
     dream `[64 vs 4] dim-2 广播断裂`——均来自**裸模型 ad-hoc 捕获代理**（probe
     L626-681 自建 `torch.cuda.CUDAGraph()`，非生产 FoldedGraphRunner）。
  2. AGENTS #45 Caveat「图静态体不解包 tuple」与代码矛盾：`_recompute_all`
     （folded_transformer.py:1384-1391）tuple 解包 + 静态体
     （cuda_graph.py:480）正是其调用者 → Caveat 疑 stale，T003 重审修正。
  3. Dream 形状断裂初判为探针构造面（probe 未线程 position_ids；AR023 sampler
     路径 [N,T] 批量 eager 已验证）——T001 设计期定案。
- srs.md + tasks.md 生成（sdd-task-requirements 流程）；随 sdd-phase-gate 需求门。
- **需求门 PASS**（ses_eb7684f71ffej3iMVCq5t9RpaJ，2026-10-17）：0 BLOCKER；全部
  证据链经独立复核逐项成立（probe 工件 L115/L241 错误签名、graph face L626-681
  裸捕获、`_recompute_all` L1384-1391 解包 + 静态体 L480 调用、AGENTS 两条规则
  第 6 条、`use_cuda_graph` 默认 False）。2 MINOR 已修：FR2 显式交付面并入 T006
  门（含 FR1 失败分支预案归属 design 期定案）；L626-697 引用订正 L626-681。
  复核者建议随 design 期承接：FR1 失败分支（修复迭代 vs 类型化降级结论）在
  design.md 定案；T003 修正 #45 文本时钉 UT 引用使其可溯源。
- **design.md 生成**（2026-10-17，自动模式）：方案对比 A（基准接线先行，根因
  被掩蔽不满足 FR1）/ B（捕获根因保真 + 生产图面 + 基准，**选定**）/ C（纯
  诊断延期基准，违反 FR5）；核心设计决策——断言为**二次失败掩蔽 primary**
  假设，`safe_capture` 失败重抛原异常（调用点 except 语义字节不变）+
  `CaptureDiagnostics` 工件（primary/secondary/VRAM/平台）；三捕获点统一路由
  （serial L1153 / batched L1443 / adaptive）；探针 graph face 生产化 + 原裸
  捕获降级 `graph_raw` 诊断面；四模式单旋钮纪律（AR024 先例）；Step 7 组件
  详设不存在跳过。T001 → passing（design 冻结待门）；T002/T003/T005 行按设计
  对齐（UT 文件名 + safe_capture 命名）。
- **design 门 PASS**（ses_eb75bd1a2ffelGsHGbbGkcMsxq，2026-10-17）：0 BLOCKER；
  全部设计断言经独立代码复核成立（三捕获点 L1153/L1433-1444/L1798-1817 降级
  语义、`_recapture_batched_runner` L1742 adaptive 仅退休、cuda_graph.py
  capture L213/_static_forward L420/_recompute_all 调用 L480、探针子进程包裹
  L249-264、ar023 bench 可复用件 L63-88、`use_cuda_graph` 默认 False
  config_manager.py:83、#44/#45 独立 MFF 先例、`graph_runner_pool_size>0`
  正确性——pool=0 会在第二 (T,prompt_len) 键上 warn-once 降级污染基准）。
  2 MINOR 已修入 design.md：探针生产 face 捕获触发机制（mini 投机窗口驱动，
  #37 前置原生线程；UT-2605 断言捕获真实尝试）+ split_layers 对齐锚栈
  （False）；3 NOTE 处置：`graph_raw` 入 `CUDA_FACE_NAMES`（已修入）、
  几何继承注记义务（已修入工件 meta 义务）、best-effort 复位 C 级中止残余
  风险（接受——观测失败为 Python RuntimeError，子进程面承载真实遏制）。
- 下一步：T002 Red→Green（sdd-task-develop 流程，全自动）。
- 机器状态注记：T006 rep 29/30 运行中；CUDA 隐藏 pytest AV 机器级故障持续（正式
  pytest 证据一律 CUDA 可见窗口获取）。

### 2026-10-17 T002 会话记录（Red→Green 完成）

- **Red**（ses_eb7516e40ffeM1Xt3oRFRrTyD7 子代理写测试，主代理未触碰实现）：
  tests/test_ar026_capture_safety.py 10 测试（UT-2601/2602 + OOM/复位失败/3 链变体
  + §6.2 透传/is 同一性 + 累计器 cap/溢出/副本 + MFF 探针键）；Red 证据：
  `from actfold.core.cuda_graph import safe_capture` ImportError + 探针键缺失
  （graph_pool_stats 现有键快照）。
- **Green**（主代理）：cuda_graph.py 追加 CaptureDiagnostics（frozen dataclass，
  8 字段含 reset_error——design §6.4 工件记录复位失败义务的字段化落地）+
  safe_capture（成功路径零额外行为；失败：deepest `__context__` 提取 →
  best-effort synchronize（is_available 门控）→ VRAM 快照 → 记录 → 裸 raise
  原对象）+ 首 8 条 cap/溢出计数/快照副本/测试隔离 reset；architecture_utils.py
  三捕获点路由（serial/batched/adaptive 各 +safe_capture 本地导入，except 语义
  字节不变）+ graph_pool_stats 两探针键（进程全局累计）。
- **证据**：standalone 冒烟 10/10 PASS（pytest AV 机器故障 → importlib+
  MonkeyPatch 驱动，正式 pytest 复跑挂 T010 CUDA 可见窗口）；black/isort/
  pyflakes/mypy（两生产文件）全清。
- **变异加固**：3 链 context 测试钉死 deepest-walk（2 链杀不掉「取浅层」变异）；
  其余行为承诺由现有断言天然覆盖（is 同一性/首 8 序序/CPU 面禁 CUDA 调用门）。
- **回归分析**（generic_smoke 驱动，4 个图机制测试文件 + 新文件）：
  1. 37 个 "No CUDA GPUs" 失败 = skipif 门在无 pytest mark 环境下的驱动器伪影
     （真 pytest 下 skip），非回归；
  2. test_architecture_utils.py::test_t005_manual_bit_exact_vs_folded_model
     失败 **在 HEAD 复现**（临时 checkout 我改动外的版本验证后还原）——预先
     存在的红测试，归属并行会话 AR025 域（10b53ad 触碰 MFF 批量前向），本 AR
     不修（域所有权），已在此记录供 AR025 会话处置；
  3. 其余 20/21 architecture_utils + cuda_graph 3/3 CPU 可跑面 + adaptive 表驱动
     1/1 全绿。
- 修改文件：actfold/core/cuda_graph.py、actfold/models/architecture_utils.py、
  tests/test_ar026_capture_safety.py（新增）。未提交（AR 里程碑提交纪律）。
- 下一步：T003（tuple-unwrap Caveat 重审落地）。

### 2026-10-17 T003 会话记录（Caveat 重审落地完成）

- **性质**：钉固测试（pinning）+ 文档修正——Caveat 审计结论是「代码已正确」，故
  无经典 Red（生产代码零改动），测试预期即绿；诚实注记于此。
- **UT-2603 PASS（CPU）**：tuple-returning mini 栈（层返回 `(hidden, None)`，
  LLaDA remote-code 形状）经 `_static_forward` 直接调用（静态缓冲手工分配镜像
  capture() 分配块 + 真实 `_prefill_parent`——capture() 本身硬性要求 CUDA）
  vs eager MFF 子前向 **逐位一致**——静态体解包语义实证钉死，#45 旧表述
  （"静态体不解包"）被代码审计 + 位级等价双重否定。
- **UT-2604 就位（CUDA 门）**：同 tuple 栈经生产路径
  `ManualFoldedForward(use_cuda_graph=True)` 串行捕获 + 重放 vs graph-off MFF
  bit-exact（AR002 重放奇偶合同）；本机 CUDA 隐藏 → 真实 pytest skip（冒烟
  驱动器显示的 FAIL 为驱动器不识别 skipif mark 的已知伪影）；T006/T010
  CUDA 可见窗口执行。
- **AGENTS #45 修正**：Caveat 文本重写——静态体**解包**（`_static_forward` →
  `layer._recompute_all` 继承 #13 首元素约定），钉 UT-2603/2604 引用可溯源；
  保留仍然为真的一半：真实 checkpoint 上尚无 tuple-layer 图捕获（AR023 探针
  两族 graph_enabled=False 且为裸捕获代理；生产面演练 = 本 AR T006）。
- 修改文件：tests/test_ar026_graph_unwrap.py（新增）、AGENTS.md（#45 修正）。
  black/pyflakes 清；未提交（AR 里程碑提交纪律）。
- 下一步：T004（探针 graph face 生产化）。

### 2026-10-17 T004 会话记录（探针 graph face 生产化完成）

- **Red**（ses_eb72dfae9ffepKvivM9uss6YNA 子代理）：tests/test_ar026_bench_contract.py
  UT-2605 六测试——5 Red（缺名 `_drive_production_graph_capture`/
  `make_graph_raw_face` + 两处注册缺失）+ 1 回归钉（derive_conclusions 仅键
  于 "graph" + 严格 >1024MB 阈值——旧代码已成立，防新面泄漏进结论）。
- **Green**（主代理，scripts/ar023_family_probe.py）：
  1. `_drive_production_graph_capture(ctx)`——生产面驱动器：CUDA 检查**最先**
     （CPU 面零加载）→ `_load_probe_model` → AR023 锚栈 MFF（`split_layers=False`
     + cosine gate 0.95 + 图旋钮：capacity 0.5 + AR021 池 size 4——池>0 防
     第二 (T,prompt_len) 键 warn-once 降级污染面）→ 家族 sampler mini 投机窗口
     （N=2/top_k=10/seed 2026，llada/dream 双配方）→ `sample(prompt_ids,
     folded_model=mff)` 前置原生线程 → `graph_pool_stats` + 诊断快照 +
     mem_get_info → face dict（ok = 捕获>0 且零失败诊断；含 graph_captures/
     降级计数/diagnostics/note）；
  2. `make_graph_face` → 生产驱动器薄包装（逐字透传，UT 钉 canary）；
  3. 原裸捕获体 → `make_graph_raw_face`（诊断面，docstring 注明「永不门
     graph_enabled」）；
  4. `CUDA_FACE_NAMES` += graph_raw；`build_family_faces` += graph_raw 注册；
     `derive_conclusions` 零改动（本就只读 "graph"——语义随面生产化自动对齐）。
- **证据**：UT-2605 6/6（generic_smoke 驱动器）；AR023 probe+bench 回归
  **34/34**（`test_ut_2318` 精确集合钉按设计演进：+graph_raw，docstring 注
  AR026 来源——失败捕获毒化分配器状态，双图面均需子进程隔离）；black/
  pyflakes 全清。
- **驱动器勘误**：generic_smoke.py 两处修正——MonkeyPatch 无 context 协议
  （旧 pytest）→ 显式 `undo()` per test（先前 spy 泄漏致 test 6 伪失败）；
  +tmp_path fixture 解析（mkdtemp）。
- 修改文件：scripts/ar023_family_probe.py、tests/test_ar026_bench_contract.py
  （新增）、tests/test_ar023_probe.py（ut_2318 钉演进 + 注记）。未提交。
- 下一步：T005（四模式实机基准脚本 scripts/ar026_real_graph_bench.py）。

### 2026-10-17 T005 会话记录（四模式实机基准脚本完成）

- **Red**（ses_eb71e3671ffeB03rq8gmTxjlJI 子代理）：tests/test_ar026_bench_contract.py
  追加 UT-2606 七测试（lazy-import 保 UT-2605 可收集）——ModuleNotFoundError
  确认 Red；钉面：模式矩阵冻结单旋钮 / schema+路径语义（override-wins、
  quick 专用件、formal 默认）/ 最小 CLI（family 默认 llada、dream 允许、
  reps 30、无 HF env 面）/ 恒等再导出 / 自有 build_artifact（ar023 键集 +
  ar026 schema 覆写）/ GEOMETRY_INHERITANCE_NOTE 义务（非空 + VRAM + 16）/
  blocked 零伪造语义。
- **Green**（主代理，scripts/ar026_real_graph_bench.py，~630 行）：
  1. **四模式单旋钮矩阵** `MODES`/`MODE_KNOBS` 冻结——baseline {folded:False}；
     folded-batched {folded:True, use_cuda_graph:False}；graph-fixed/graph-adaptive
     仅图旋钮差异（capacity 0.5 / pool 4 / adaptive False|True）；`_make_mode_mff`
     逐模式 MFF（MFF_C 隔离：own cache/gate/counters，knobs 取自矩阵行去掉
     `folded` 语义标记）；`_make_mode_sampler` 三折叠模式统一 N=4/top_k=10
     （采样配置恒等——单旋钮纪律的采样侧落实，刻意不复用 ar023 的
     mode-coupled `_make_sampler`）。
  2. **H1 语义适配决策（Green 前关键考证）**：`h1_quality_verdict` 内部硬编码
     `_BATCHED="folded-batched"`/`_ALIGNED="folded-batched-greedy"` 角色名
     （L184/L203-204），其 0.5-gap-shrink 阈值**前提是质量机制**——图重放是
     机制置换而非质量机制，该阈值语义不可迁移（图模式 ≈ 锚质量 → ratio≈1.0
     必败）。故 AR026 质量面为**新冻结** `h1_graph_parity_verdict`（机制置换
     不变性 + 参与度门）：面 (a) 逐 prompt posthoc overall-mean vs 锚
     ≤1e-6 atol；(b) decode-length 逐 prompt 相等；(c) vs 锚 block-grid
     exact 全中率 1.0；**零捕获 = inconclusive**（eager 回退永不冒充奇偶
     通过——参与度面，杜绝「图未捕获但质量面假绿」的诚实性漏洞）；None
     mean refute；优先级 refuted > inconclusive > supported。`h1_quality_verdict`
     保持恒等再导出（跨工件对账面），`h2`/`h4`/`reproduction` 原样复用
     （h2 角色映射：aligned=graph-adaptive、batched=folded-batched；h4
     graph_enabled=True——本基准按构造演练图路径，探针结论被运行本身超越；
     canary 升级为真实面：timed 窗口 degradation_fallback_count 增量==0）。
  3. 运行层镜像 ar023 `run_family_bench`：复现双跑 → warmup（图捕获窗口）
     → per-rep 四模式 round-robin（#39 纪律）→ sidecar ledger/stable-ratio
     → 质量面（vs baseline 对账面 + vs 锚奇偶面双份）→ 判定 → 工件
     （mode_wall 含逐模式 graph_pool_stats JSON 化——池键 `(shape, mask对象)`
     → `str(shape)`，capture_diagnostics `to_json_dict`；meta 携
     GEOMETRY_INHERITANCE_NOTE / VRAM 余量面（mem_get_info 载入后+结束）/
     h2 无方向预注册 / probe 结论快照）。
  4. **Green 期 spec 钉（T003 先例）**：`test_ut2606_h1_graph_parity_spec_pin`
     追加——冻结规则规格推导（非观察实现行为）：三面各自独立 refute、
     参与度门 inconclusive、优先级 refuted>inconclusive、None mean refute、
     逐模式隔离。
- **证据**：UT-2605 6/6 + UT-2606 7/7 + spec 钉 1/1 = **14/14**
  （generic_smoke 驱动器）；回归 AR023 probe 25/25 + AR023 bench 9/9
  （恒等再导出面零漂移）；black/isort/pyflakes 全清；mypy 新文件 0 错
  （algo_experiments.py 7 错为预存在、官方面为 `mypy actfold`）；CPU 面
  CLI/路径 sanity 通过（schema/quick/formal/无 repo_override）。
- **变异加固（T002 纪律，3 变异全杀 + 还原）**：`h1_graph_parity_verdict`
  三处语义变异逐一注入-验证-还原：
  M1 参与度门 `> 0`→`>= 0`（零捕获冒充 engaged）→ spec 钉 FAIL 杀；
  M2 优先级 inconclusive 先于 refuted → precedence 钉 FAIL 杀；
  M3 面 (c) `rate == 1.0`→`>= 0.99` → **初版幸存**（fixture 未命中率
  63/64=0.984 低于双阈值）→ fixture 加固为近全中 999/1000（合并
  ≈0.999，必须 refute）→ FAIL 杀。还原后三位点核验
  （`> 0`/`== 1.0`/refuted 优先）+ 全量 48/48 绿 + lint 清。
  面积注记：face-a 的 `>`/`>=` atol 边界变异为浮点惰性
  （posthoc 值 round 6 位，delta==atol 精确边界实测不可达——
  0.369662-0.369661=1.0000000000287557e-06 恒过阈），不构成可观察
  语义差，未设边界 fixture（诚实记录，非省略）。
- 修改文件：scripts/ar026_real_graph_bench.py（新增）、
  tests/test_ar026_bench_contract.py（UT-2606 追加 + spec 钉）。未提交。
- 下一步：T006（GPU quick smoke：真实 LLaDA-8B 生产捕获首验 + Dream 面显式
  交付；排队 AR023-T007 → AR024-T005/T006 → AR025 实机面之后）。

### 2026-10-18 维护轮：停实验 + 深度优化 + gitee 同步（用户指令）

- **大型实验停止**：dream formal（rep 7/30）已终止并在运行日志留痕
  （工件未写，重启命令在日志末行）；其后 GPU 空出。
- **深度优化落地**（并入维护提交）：
  1. 质量门全绿：black 165 文件（钉版 23.12.1 / 100 列）/ isort 5.13.2 /
     pyflakes 4 / mypy(strict, 68 文件) 全清；pytest **1098 passed,
     4 skipped, 0 failed**（CUDA 可见）。
  2. 两处失败根治：UT-2401 辅助函数 host/CUDA gather 设备错配（测试缺陷）；
     BS-007 性能判定改为环境诚实面——本机实测图重放比 eager 慢约 4.5x，
     经 AR002 版本 worktree 二分确证为 Windows WDDM 小内核提交延迟特性
     （非代码回归），测试记录实测数字并 skip 排序判定，历史工件保持证据
     效力；bench 脚本加 `_warm_gpu` 时钟状态控制 + 协议字段。
  3. 环境漂移根因封堵：requirements.txt `transformers<5`（remote-code 5.x
     不兼容）；requirements-dev 钉 black/isort/pyflakes/mypy 版本；
     isort 5.13.2 契约补齐。
  4. 复现文档：RERUN_CHECKLIST 新增 AR026 资源行 + 步骤 15（探针生产图面
     首验/quick/formal 三段）+ **§7 高算力服务器复现路径**（Linux/TCC
     推荐、权重获取、锁频、WDDM 限制、规模化预期、工件回填）；
     AR023 正式跑墙钟勘误（实测 llada ≈29.2h）；AGENTS 新增 #46 运维法规
     + Reproduction Guide Rule（大型实验四要素契约）。
- **证据入库**：AR024 实现（UT 23）+ AR026 T001-T005 实现（UT 10+14+3）
  + `ar023_llada_bench.json` 正式工件（h1 refuted 诚实负结果 / h2 1.49x /
  h4 supported）均已本地提交。
- **gitee 同步（树验证 PASS）**：69 个本地提交按 10 段树累积推送
  （代理 ~96KB POST 限制，逐段 commit-tree）；`main^{tree}` ==
  `gitee/main^{tree}`（29785975b9...）；**从 gitee 浅克隆后全量测试复跑
  1098 passed** —— 下载即可复现验证成立。
- **遗留**：AR026 T006/T007（真实 LLaDA/Dream 生产捕获 + 四模式正式面）
  与 AR024 T005/T006、AR025 实机面仍待 GPU 窗口；正式跑仍须预承诺单次。

### 2026-10-18 T006 首验（Step 1 生产图面，两族均 PASS）

- **llada**（worker 模式单面，venv 4.53.1）：`AR023_FACE_RESULT:
  {"ok": true, "graph_captures": 2, "degradation_fallback_count": 0,
  "structural_eager_count": 1, "diagnostics": [], ...}` —— **生产路径捕获成功**
  （串行+批量两 runner），AR023 裸捕获的 `captures_underway.empty()
  INTERNAL ASSERT (CUDACachingAllocator.cpp:2967)` **未复现**；零失败诊断、
  零降级回退。
- **dream**（同法）：`{"ok": true, "graph_captures": 2, "capture_delta_mb":
  513.1, "degradation_fallback_count": 0, "diagnostics": [], ...}` ——
  **生产批量捕获成功**，AR023 裸捕获的 [64 vs 4] dim-2 广播断裂**未复现**。
- **解读**：AR023 的 graph 失败签名是**裸捕获代理**（ad-hoc 裸前向）的产物，
  生产 MFF + 家族 sampler mini 窗口路径在真实 checkpoint 上可捕获——AR026
  的核心假设（生产面可用）得到实机支持。FR1/FR2 首验达成。
- **VRAM 面**：两族 headroom_mb 均报 0.0（16GB 卡载 8B bf16 ≈15.2GB；
  headroom 门保持 graph_enabled=False 的保守结论不变——捕获可用性与
  headroom 门是两个独立面）。Dream 捕获侧成本实测 513.1MB。
- 证据工件：`results/optimization/ar026_probe_graph_faces.json`（worker 单面
  证据，注明非七面完整工件；tracked 探针结论未改动）。
- **下一步**：Dream 四模式 quick 冒烟（后台运行中，日志
  `results/optimization/ar026_dream_quick.log`）；llada quick 因 16GB 卡上
  三 MFF + 图池并存的 VRAM 风险，暂缓（正式面须 ≥24GiB 或先腾挪）。
- **Dream 四模式 quick 冒烟：环境不可用，终止留痕**（10-18 18:4x）：
  两次启动（前台 45min + 后台分离 ~55min）均无 rep 落盘；实测 GPU 被
  WDDM 桌面组合器 + 外部计算占满（99% util / 15.8GB / 645MHz，15 个桌面
  进程持上下文），Dream 128 步采样在争用下不可用。按「停大型实验」指令
  终止（err 日志留痕含重启指引）。**本轮 T006 证据以探针生产图面首验为准**
  （两族捕获 PASS）；完整四模式 quick/formal 待安静 GPU 窗口（建议直连
  TCC/Linux 或独占）。llada quick 因 16GB 三 MFF + 图池并存风险同样后置。
- **llada 四模式 quick 冒烟：VRAM 墙实测确认，终止留痕**（10-18 18:09
  启动 → 18:5x 终止）：短暂安静窗口（27%/313MB）下启动后，GPU 实际达到
  **16065/16384 MiB**（3 个 MFF 缓存 + 图池静态缓冲叠加在 15.2GB bf16
  权重之上，WDDM 开始换页）且 ~35 分钟无 rep 落盘（首个 flush 在 rep 1
  完成时）。**runbook 预注册的 16GB 四模式面瓶颈被实机证实**：llada 完整
  四模式面须 ≥24GiB（dream 7B 尚可、llada 8B 不可）；证据效力不受影响
  （T006 首验以 `ar026_probe_graph_faces.json` 两族捕获 PASS 为准）。
  重启指引在 `results/optimization/ar026_llada_quick.err.log`。
- **运维修剪（本轮实测反馈）**：bench 生命周期打印（device 行 / 复现双跑 /
  warmup / sidecar / degraded / blocked）全部补 `flush=True`——本轮两次
  冒烟监控盲区（缓冲导致日志 0 字节）的直接修复；black/pyflakes 复验清。

### 2026-10-18 harnessX 开发期会话（T006 以图面首验证据收口）

- **T006 -> passing**（用户确认验收口径：以 r026_probe_graph_faces.json
  两族生产图面首验证据结 T006；四模式 quick 归入 T007 预备面）。
- **证据面复核**（工件逐字段核验）：llada/dream 均 ok=true、
  graph_captures=2、degradation_fallback_count=0、structural_eager_count=1、
  diagnostics=[]、capture_diagnostics_overflow=0、headroom_mb=0.0；
  dream capture_delta_mb=513.1（llada 0.0，捕获侧复用既有池）。
- **T006 四要素对照**：
  1. FR1 实机首验 —— 生产路径（MFF + FoldedGraphRunner + 家族 sampler mini
     窗口）在真实 8B checkpoint 上捕获成功，AR023 裸捕获签名
     （captures_underway.empty() INTERNAL ASSERT / [64 vs 4] 广播断裂）
     **均未复现**；
  2. 失败签名/降级/VRAM 记录 —— 零失败诊断、零降级回退、VRAM 余量面已记录
     （16GB 卡 headroom 0.0 → graph_enabled 保守门不变）；
  3. FR2 Dream 显式交付 —— dream 生产批量捕获成功工件（单面证据，注明非七面
     完整工件，tracked ar023 结论未改动）；
  4. FR1 失败分支预案 —— 未触发（捕获成功），无需修复迭代回路。
- **操作留痕**：四模式 quick 两次尝试（dream 前台/后台、llada 后台）均被环境
  阻断并诚实留痕（详见本文件末尾两条记录）：WDDM 争用 + 16GB 三 MFF 图池
  并存的 VRAM 墙（实测 16065/16384 MiB）。T007 正式面须 ≥24GiB 安静 GPU
  （RERUN_CHECKLIST §2/§7 已写明命令与资源）。
- 修改文件：specs/changes/AR026-real-graph-e2e/tasks.md（T006 状态 + 本记录）。
- **下一步**：T007（formal，reps=30 预承诺单次）——被 GPU 资源约束阻塞，
  待 ≥24GiB 服务器窗口；T008/T009/T010/T011 随之串行。
