# [AR024] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR024 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成） |
| 创建日期 | 2026-10-16 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 前置确认：AR023 探针工件结论复用交叉引用（LLaDA degraded=False/bf16/graph-off）——无新探针，结论注记入进度记录 | - | passing | 2026-10-16 完成：`ar023_family_probe.json` conclusions 实测复核（llada/dream 均 bf16/graph_off/degraded=False） |
| T002 | Red：`tests/test_llada_granularity.py` 失败测试批（UT-2401 起：FR1 窗口迭代/前向次数/链延续/s=block_size A/B 锚/默认零扰动/s 校验；FR2 两阶段/退化/确定性；FR3 假设对/退化同值/N=1/正交；FR4 YAML 面；FR5 基准合同面）——全红验证 | T001 | passing | 2026-10-17 完成：23 测试函数全红（CPU 隐藏 CUDA 运行 21.44s） |
| T003 | Green：生产实现（llada_sampler 三机制 + config 面 + speculative_core 仅在确属核面时最小扩展）+ 全绿 + mutation 自证（三机制面变异全杀还原） | T002 | in_progress | 代码完成+lint 三清；全绿验证被本机 AV 环境故障阻塞（见进度记录 2026-10-17） |
| T004 | 基准脚本 `scripts/ar024_granularity_bench.py`（导入式复用 ar023 助手，AR022 先例）+ 脚本 UT 全绿 | T003 | in_progress | 脚本完成+lint 三清+UT-2411 standalone 合同全过；pytest 正式证据待机器恢复 |
| T005 | `--quick` 实机冒烟（两模式组：推荐栈四模式 + refill 报告面）——机制验证非定案；blocked 语义实机验证 | T004 + **GPU 空闲**（AR023 T006 完成且 AR023 实机面结束） | pending | GPU |
| T006 | 正式 reps=30 预承诺单次（四模式 + 复现自检 + meta 全要素）→ `ar024_llada_granularity_bench.json` | T005 | pending | GPU |
| T007 | 质量门：全量 pytest（≥1022/0/4 只增不减）+ demo 三锚点 + pyflakes/mypy/black | T003（CPU 面先行）；完整门 GPU 空闲后 | pending | CPU+GPU |
| T008 | 文档收口五件：README 全量中文重写（数字从工件重算）+ AGENTS #46 + GUIDE 第三十一部分 + CHANGELOG + RERUN 第 14 步 | T006, T007 | pending | CPU |
| T009 | review 门（spec/design 合规审查，多轮修复循环至 PASS） | T008 | pending | CPU |
| T010 | ST 验收（重算面清单驱动，数字工件溯源逐项核对） | T009 | pending | CPU |
| T011 | 归档：`specs/archive/AR024-commit-granularity/` + 单交付提交 + gitee 分段链推送 + 双端树验证 | T010 + **AR023 已归档**（串行结档约束） | pending | CPU |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

- 2026-10-16：req 阶段完成。用户定案：机制范围 ①+②+③ 全做；开启投机时
  子块默认 s=8；验收协议复用 AR023 四模式基准。srs.md/tasks.md 生成。
  并行态注记：AR023 T006 正式跑进行中（GPU 独占），AR024 CPU 面
  （T001-T004、T007 CPU 部分、T008 起草）先行，实机面（T005/T006）与
  归档（T011）排队——AR023 先归档、AR024 后归档。
- 2026-10-16：design 阶段完成。方案 A 定案（族持窗口循环 + Core 最小三扩
  展：`eos_aware`/`refill_top_k` 尾部可选参默认字节不变）；5 设计冻结点定案
  （k=None→s//2、行 N-1 无条件优先、FR3 验收措辞精化、批量前向计数口径、
  refill history 每窗一条）；校验域冻结（s 上界 scoped N>0）。design.md 生
  成，任务结构与设计对齐确认（T002/T003 描述已含 Core 最小扩展授权面，无
  结构调整）。开发期按 G6 Minor 建议以 FR1→FR2→FR3→FR4 微任务链推进。
- 2026-10-17：T002 Red 完成（23 测试函数全红，CPU 隐藏 CUDA 运行）。
- 2026-10-17：T003 生产代码完成（4 文件）：`speculative_core.py`
  `_speculative_commit` 尾部 `eos_aware`/`refill_top_k` 可选参 + eos 延续行
  （`non_eos.to(torch.uint8).argmax(dim=1)`+gather，最后赋值覆盖行 N-1）+
  部分提交（`k_eff=min(k,num_masked)`，k≥num_masked 退化 FR1；置信 topk；
  refill 模式 Core 不 append history）；`llada_sampler.py` config 四字段
  （`speculative_sub_block_size: int|None=None`→`__post_init__` 归一 8 +
  `_sub_block_size_explicit` 旗标 / `progressive_refill` / `refill_top_k` /
  `candidate_eos_aware=True`）+ 五段校验 + 入口 resolved block 校验 + 块内
  while 窗口循环（w_idx 计全窗口、无掩码窗口零前向跳过、tag `llada_spec_b{b}`
  `_w{w}` 直赋链）+ `_refill_phase_two`（`_r` tag、父=胜者分支、argmax 填充、
  零前向提前 return）；`config_manager.py` ActFoldConfig 四消费键；
  `llada_8b.yaml` AR024 注释段。**冻结 tag 无 folding-counter 后缀**：窗口
  循环与 phase two 用直赋链替代 `_next_folding_branch`（该方法会追加
  `_{counter}` 与冻结 tag 冲突；AR023 旧测试不钉 sampler 级 tag，安全）。
  lint 三清：black（行折叠+`__post_init__` 局部窄化重构修 3 处 mypy
  `int|None` 比较错误）/pyflakes/mypy `--ignore-missing-imports` 全过。
  首次全量（误在 CUDA 可见下）14 passed/9 failed：6 个 tag 后缀失败已由直赋
  链修复；UT-2409 cuda 字面量 1 个 CPU 环境应自消；UT-2411 留给 T004；
  UT-2405/2406 失败同 tag 根因。
- 2026-10-17：**GPU 违规事故（诚实记录）**：首次测试运行因 PowerShell
  `$env:CUDA_VISIBLE_DEVICES=""` 实为删除变量（PS 5.1 无法持有空串环境变
  量，Windows 子进程 env 亦会丢弃），测试在 cuda:0 跑了 ~33s，正值 T006
  rep~23-27 采样期，可能污染该 rep 墙钟计时。正确隐藏方式确认为
  `$env:CUDA_VISIBLE_DEVICES='-1'`（'99' 触发 native AV）。之后所有运行严
  格 CPU。
- 2026-10-17：**本机 AV 环境故障（T003 验证阻塞根因）**：CPU 隐藏 CUDA 下
  pytest 进程 STATUS_ACCESS_VIOLATION（0xC0000005）非确定性死亡（崩溃点漂
  移：linecache/traceback.format_stack（`torch.manual_seed`→CUDA lazy 队
  列 `format_stack`）、`_pytest/assertion/rewrite.py mark_rewrite`、
  layer_norm 等）。已证非本 AR 代码所致：(a) 隔离 worktree（detach 于
  main，仅含我方 5 文件副本，无并行会话 in-flight 编辑）同样崩溃；(b)
  stash 我方编辑后旧 AR023 测试同样崩溃；(c) **最小复现：仅含
  `torch.manual_seed(42)` 的单测试 + pytest + CUDA 隐藏 = 测试通过但进程
  0xC0000005 退出**；(d) 独立 python 进程完全健康（全量 actfold 导入、
  200 次连续 manual_seed、black/mypy 重 lint 均正常）；(e) `-p
  no:faulthandler`/`-p no:cacheprovider`/`PYTHONDONTWRITEBYTECODE=1`/
  `OMP_NUM_THREADS=1` 均不缓解，worktree conftest 中和
  `torch.cuda._lazy_call`（CUDA 不可用时语义等价 no-op）亦不缓解。结论：
  机器级故障（pytest 启动/文件 IO 路径在高负载下被杀；疑企业安全软件注入
  DLL 与异常处理链冲突），与代码无关；独立 python 不受影响。验证策略改
  为：T006 完成（机器负载回落）后重试 pytest；若仍崩溃按小批次重试以
  summary 输出为准。lint 面已借此窗口全部完成。
- 2026-10-17（续）：**standalone 冒烟替代验证**（pytest 被环境故障阻塞期
  间的早期验证手段，非正式证据）：以 importlib 加载测试模块 + 手工递归
  解析 fixture 的冒烟脚本（`gran_smoke.py`/`spec_smoke.py`，temp 目录）
  跑通两套测试面。结果与修复：
  (a) AR024 面 23/23 全过、AR023 面 19/19 全过（修复后）；
  (b) **测试 fixture 别名 bug 修复**（非生产 bug）：`_SpyMFF` 按引用记录
  输入张量，而 `_refill_phase_two` 在前向之后原位 argmax 填充画布——
  UT-2405/2406 断言读到的是填充后状态。修复：spy 在调用时
  `detach().clone()` 快照输入张量（对全部 args 消费者更正确）；
  (c) **AR023 trap fixture 语义钉固**（设计 §4.1.6 注记 sanctioned）：
  `_LLaDATrapFixture` config 未显式设 `candidate_eos_aware` → AR024 默认
  on 使行 N-1 成为延续行（argmax 填充）而非钉定随机抽取，UT-2303
  greedy-loses 面失败。修复：fixture config 显式
  `candidate_eos_aware=False`（AR023 语义还原 = 显式 False，基线保护）；
  (d) **device 字面量修复**（3 测试 5 处）：UT-2409 sampler-default-eos-on
  / UT-2413 window-early-stop / UT-2413 unmasked-window-skip 中
  `torch.equal(cuda张量, cpu字面量)` 在 CUDA 可见时 raise——字面量补
  `device=device`（CPU 冒烟不可见该问题，T007 CUDA 门会踩中）；
  (e) T004 脚本 `scripts/ar024_granularity_bench.py` 完成（~630 行）：
  模块级 `_ar023` 导入 + 助手恒等复用（`mark_all_blocked`/
  `h1_quality_verdict`/`_make_bench_mff` 赋值再导出——from-import 与赋值
  二选一，pyflakes 重定义约束）；冻结 `GRANULARITY_STACKS`（batched=
  AR023 锚 s=32/eos off；greedy=AR024 推荐 s=8/eos on；构造性 A/B 隔离：
  两栈仅粒度三旋钮差异）+ `GRANULARITY_CONSTANTS`（五键 +
  anchor_semantics）；llada-only CLI（dream→argparse exit 2）；
  `--refill-face`（强制 quick 尺度 + 对齐栈 progressive_refill=True，
  独立 `_refill` 工件，不入正式 verdict）；工件 schema 名覆写为
  `ar024_granularity_bench`（装配逻辑恒等复用，schema 名自持——溯源）；
  meta 全承 ar023 + granularity_constants/granularity_stacks/refill 注记。
  lint 三清（black/isort/pyflakes/mypy——mypy 仅 pre-existing
  algo_experiments.py 9 错误，本文件零错误；scripts/ 不在 mypy 门内）。
  UT-2411 合同 standalone 全过（恒等/栈字典/路径三元组/parse_args 行为/
  常量键集）。
- 2026-10-17（续 2）：**T003 mutation 面完成（standalone 冒烟承载）**：
  5 个手工变异体全部被杀、双文件 SHA256 还原验证。M1（FR1 跳过窗口
  w_idx 不计数）→ UT-2413-unmasked-skip 1 失败；M2（FR2 k 钳制失效
  `k_eff=num_masked`）→ UT-2405/2406/2413-partial 3 失败；M3（FR2
  phase-two 填充 argmax→argmin）→ UT-2405 1 失败；M4（FR3 延续行去
  EOS 排除 `pool[:,0]`）→ UT-2407/2409×2 3 失败；M5（FR3 延续行
  N-1→N-2）→ UT-2407/2408/2409×2 4 失败。还原后 23/23 全绿复跑。
  speculative_core.py 还原 SHA256
  `DB3424564DB39EEFDBB2973E999D714C5E65B767A9DC30CF8D14AAFE4B66EE77`、
  llada_sampler.py 还原 SHA256
  `2FE3F8D049B9641C9BB010147A492CFBC11648B3648B5C43E0734765BE8D2FA1`
  （与变异前快照逐位一致）。**承载体注记（诚实记录）**：pytest 被
  本机 AV 环境故障阻塞期间，mutation 杀灭证据经 standalone 冒烟脚本
  （同一测试函数集）承载；T007 正式 pytest 门恢复后须复跑 pytest 形式
  确认（预期一致——同一测试函数集）。
- 2026-10-17（续 3）：**AR023 回归面三文件 standalone 冒烟全绿**：
  `test_llada_speculative.py` 19/19（trap fixture 钉固后）、
  `test_dream_speculative.py` 17/17、
  `test_speculative_greedy_alignment.py` 8/8——默认 eos-ON 零扰动论
  证实（唯一受扰动面 = trap fixture，已按设计注记显式钉固）。
  **并行会话共享树注记**：AR025 会话提交 `10b53ad`（T003+T004，
  12:10:14）staged 共享文件 `actfold/utils/config_manager.py` 时将本
  AR 未提交的四消费键一并带入 HEAD（共享工作树现实，归属注记在此；
  本 AR 提交面相应不含 config_manager.py）。本机 AV 环境故障经 venv
  python 复测仍复现（与 python 解释器无关）；并行会话同期全量
  pytest 195 passed 系 CUDA 可见运行（CUDA 隐藏 = AV、CUDA 可见 =
  完整完成的经验规则确认）——正式 pytest 证据在 T006 完成 GPU 空闲
  后以 CUDA 可见运行获取。

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### 2026-10-16 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（G1-G9）
- 修复的问题：无
- 遗留 Minor：G6——T002/T003 批次粒度偏粗属 TDD 惯例可接受；开发阶段按 FR1→FR2→FR3→FR4 微任务链逐个 Red/Green 缓解，不改 tasks.md 结构
- 审查代理：sdd-gate-reviewer（ses_ebb3b85edffe4HT3vhLPTmgbFQ）

### 2026-10-16 design 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：15 项（G1-G15，无 NO；G10 组件详设目录不存在确认跳过）
- 修复的问题（4 项当场修入）：G13 Important——§4.2.1 窗口级早停缺显式 UT，补 UT-2413（早停确定性 + refill finished-仅-top-k 位 + 无掩码位跳过），随 T002 Red 实现；G3——design §4.1.4 伪码签名补 `ctx`（与 §4.3 一致）；G8——design §6 头部补覆盖率目标显式声明（功能点/分支 100% 映射 + 基线只增不减）；G14——srs FR3 第二条验收句按 design §4.1.5 精化措辞回写
- 关键交叉验证：`_speculative_commit` 现签名逐参一致（扩展确为尾部两可选参）；Dream 调用点不传新参兼容论证成立；22 个 ar023 导入助手名防虚构全命中；branch tag 冻结与现行一致；A/B 锚种子公式逐项成立
- 审查代理：sdd-gate-reviewer（ses_eb88c4bf7ffeKBol8MhP6gWLwh）
