# [AR019] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR019 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成）|
| 创建日期 | 2026-10-14 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | Red：子代理编写 `tests/test_position_embeddings.py`（FR1 四个构建面 UT：串行/批量/graph 静态体/**自适应重捕获** + None 默认零变化双侧不变量 + 重放期对象恒等契约 + 异常面），实测确认红因=生产缺失 | - | passing | 951 行，9 红（1905 拆 CUDA/CPU 两函数），红因全部=forward 签名缺失 TypeError / 1907 异常档位错；pyflakes/black ✓ |
| T002 | Green：实现 `position_embeddings` 线程化（`ManualFoldedForward.forward` 新参 → 批量/串行 eager → `FoldedTransformerLayer`/`SplitFoldedTransformerLayer` → `_recompute_all` kwargs；`FoldedGraphRunner` 捕获烘焙静态缓冲 + `_recapture_batched_runner` 重捕获面同烘焙 + 重放期对象恒等检查），mutation ①-④ 全红后 SHA256 还原；全量回归零扰动 | T001 | passing | 9/9 绿；mutation ①(1901/02)②(1905)③(1904)④(1903) 全红后 SHA256 还原（F46B776B/3239A050）；回归 864=855+9 零扰动；mypy clean、pyflakes 零、新增行 100 列合规（基线两文件本就非 black-clean，不属本 AR 责任面）；demo 锚点 85.5%/2.35e-03/93.75% ✓ |
| T003 | `scripts/ar019_real_e2e_bench.py`（三路径 × 真实块验证工作负载 × H1-H4），`--quick` 冒烟于 AR019 venv + 本地权重走通（含真实 bf16 图捕获首证） | T002 | passing | 816 行（review N-3 修复后终值）；`--quick` 冒烟 PASS：T=256/N=4 baseline 1684ms / fixed 1002ms / adaptive 734ms，容量 512→160（需求 128）1 次重捕获，峰值 8891MB，H2/H3/H4 supported、H1 inconclusive（reps=3 下 2σ 判定预期）；真实 bf16 图捕获首证成功；Triton bf16 merge 回退按 P7 预期触发并记录 |
| T004 | 工件运行：reps=30 预承诺单次（<15 分钟），`results/optimization/ar019_real_e2e_bench.json` | T003 | passing | 单次定案：三配置加速比（adaptive vs baseline）2.4355/2.6169/2.5914×；H1 整体 inconclusive——(256,4) 首配置 WDDM 热身方差（baseline std 497ms>p50 62%）淹没 2σ，(256,8)/(512,4) std 5-40ms 干净且 H1 supported；H2/H3/H4 全 supported；容量右移 512→160/1024→320/1024→160（需求 128/256/128）各 1 次重捕获；峰值 VRAM 5624/7640/8949MB；pe 恒像失配 0、违例 0 |
| T005 | 回归 + 质量门：全量 pytest（新基线）、demo 三指标锚点、mypy clean、pyflakes 零、black 100 列零（含 T003 基准脚本，scripts 在检查范围内） | T003 | passing | 864 passed（=855+9 新基线）/ mypy 66 文件 clean / pyflakes 零 / 基准脚本 100 列合规 / demo 锚点 85.5%+2.35e-03+93.75% ✓ |
| T006 | 文档收口：OPTIMIZATION_GUIDE 第二十六部分、CHANGELOG、AGENTS.md（#37 延伸 position_embeddings 契约）、RERUN_CHECKLIST（venv + 权重获取）+ **README 全方位中文更新**（AGENTS.md 新规：创新总览表第 9 行 + 机理详解 + 最新结果置顶 + 质量基线 + 限制路线图） | T004, T005 | passing | 五文件全收口：GUIDE 第二十六部分（含实测墙钟表+H 判定+语义要点）；CHANGELOG AR019 英文条目（沿 AR018 谱系）；AGENTS #37 增 AR019 pe 契约句；RERUN_CHECKLIST 显存表行+第 9 步（权重获取/venv/基准命令+对照数字）；README 全量中文重写 637→762 行（9 行创新表/AR019 头条表置顶/864 基线/P1-7 五档收官叙事）——子代理执行，数字抽查与工件一致；portability 14 passed |
| T007 | review 门控（只读子代理：spec/design 合规 + 代码审查），全部问题（含 Minor）修复 | T006 | passing | **PASS**（审查代理 ses_ec6d6f802ffejRJ789QE2sSahC，六维度全过，本机复核 864/mypy/pyflakes/demo 锚点）：1 Minor（M-1 文档行数失准 786→实际 812）+ 4 Nit 全修——M-1/N-1 行数回写（bench 816 终值/README 762/UT 955）；N-2 生产修复：入口校验补 `dim()<2` 前置（1-D cos/sin 原会漏 IndexError，现落 ValueError 档）+ UT-1907 补第 6 用例；N-3 `fixed_capacity` 改读 `fixed_runner.capacity`（无 runner 时回退算术值）；N-4 AGENTS README 规则去 "8-row" 数字化措辞 |
| T008 | ST 验收：st-cases.md 生成 + 执行（正常/边界/异常/回归四类），一次成 Go | T007 | passing | **Go**——st-cases.md 11 用例（正常 4/边界 2/异常 2/回归 3）一次全 PASS：UT 文件 9 passed（8.40s）；工件校验 PASS（H2/H3/H4 supported、三配置 parity 逐位一致 maxdiff=0.0/argmax=1.0、pe 恒一失配 0、违例 0、容量三断言全过、H1=inconclusive 如实）；全量 864 passed/3 skipped；demo 三指标 85.5%/2.35e-03[HIGH]/93.75% 锚点精确；mypy 66 文件 clean/pyflakes 零/100 列零违规。H1 inconclusive 为预承诺单次纪律下如实记录，非缺陷非阻塞 |
| T009 | 归档 `specs/archive/AR019-real-checkpoint-e2e/`，本地 commit + gitee commit-tree 推送 + 双端树验证 | T008 | passing | 完成——本地 commit `a1e000e`（15 文件 +3446/-40）；gitee commit `456c084`（commit-tree 本地树+gitee 父 `49d53b3`）；双端树 `31cf27cb927a36e77508f53291de290ef46f8ffb` 一致；工作树 clean。AR019 全流程闭环 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### Requirements 门控
- **Round 1 — FAIL**（审查代理 ses_ec75107a4ffeB1MEbbRsCHaNWh）：2 Major（Major-1 graph 重放期 pe 对象恒等契约缺失；Major-2 FR1 遗漏自适应重捕获面）+ 2 Minor + 2 Nit。全部修入 srs.md。
- **Round 2 — PASS**（同代理复核）：余量 Minor/Nit 已当场修入（In Scope 枚举口径、#37 清单补 pe 对象、术语表前提、工件记 pe 回退计数）。

### Design 门控
- **Round 1 — FAIL**（审查代理 ses_ec73c6393ffe7cihKDpUvhrQ2r）：2 Major（Major-1 srs"维度不匹配→ValueError"子句无设计落点；Major-3 UT-1906 Red 预期标注错误——现状 forward 无该参数、传参即 TypeError 不可能为绿）+ 5 Minor + 4 Nit。
- **Round 2 — PASS**（同代理复核）：必修项核验通过（入口 seq 维校验 + head_dim 归属 capture-failed 可见档论证 + UT-1906 拆 a/b/c 三子断言）；余 N1/N2/N3 非阻塞项已当场修入（N1 4.2.1 流程图补 seq/head_dim 校验节点；N2 4.2.1 末行 growth/shrink 分叉表述同步；N3 入口补 cos/sin 内部 head_dim 一致性检查 + UT-1907 补用例 + srs.md 异常处理节补 head_dim 例外对齐句）。
