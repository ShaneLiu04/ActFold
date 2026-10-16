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
| T001 | 前置确认：AR023 探针工件结论复用交叉引用（LLaDA degraded=False/bf16/graph-off）——无新探针，结论注记入进度记录 | - | pending | CPU |
| T002 | Red：`tests/test_llada_granularity.py` 失败测试批（UT-2401 起：FR1 窗口迭代/前向次数/链延续/s=block_size A/B 锚/默认零扰动/s 校验；FR2 两阶段/退化/确定性；FR3 假设对/退化同值/N=1/正交；FR4 YAML 面；FR5 基准合同面）——全红验证 | T001 | pending | CPU |
| T003 | Green：生产实现（llada_sampler 三机制 + config 面 + speculative_core 仅在确属核面时最小扩展）+ 全绿 + mutation 自证（三机制面变异全杀还原） | T002 | pending | CPU |
| T004 | 基准脚本 `scripts/ar024_granularity_bench.py`（导入式复用 ar023 助手，AR022 先例）+ 脚本 UT 全绿 | T003 | pending | CPU |
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

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### 2026-10-16 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（G1-G9）
- 修复的问题：无
- 遗留 Minor：G6——T002/T003 批次粒度偏粗属 TDD 惯例可接受；开发阶段按 FR1→FR2→FR3→FR4 微任务链逐个 Red/Green 缓解，不改 tasks.md 结构
- 审查代理：sdd-gate-reviewer（ses_ebb3b85edffe4HT3vhLPTmgbFQ）
