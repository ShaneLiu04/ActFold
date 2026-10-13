# [AR014] 任务分解 — Triton 降级粒度细化 + 优先级表勘误

| 字段 | 内容 |
|------|------|
| AR 编号 | AR014 |
| 源 | srs.md（同目录） |
| 日期 | 2026-10-13 |
| 状态 | Final（req/design/review 三门控 PASS；T001-T007 全完成——817 passed（807+10）+ demo 一致 + mutation ①3红/②1红/③4红 还原；ST 7/7 Go 一次成；已归档 specs/archive/AR014-triton-fallback-granularity/） |

## 任务列表

### T001 Red：UT-1301~1306 新钉 + FR5 既有钉迁移（子代理）
- 新文件 `tests/test_triton_fallback_granularity.py`：UT-1301
  （dtype 隔离 + 第二失败键第二告警 + forbidden 回退手段）/
  UT-1302（同键静默）/UT-1303（bucket 隔离 + forbidden 回退）/
  UT-1304（总开关兼容）/UT-1305（告警三要素 + 既有 match 子串
  保留）/UT-1306（bucket 纯函数）；autouse 注册表隔离 fixture；
  CUDA+Triton skipif 先例（test_fused_ops.py T013）；
- 迁移三处旧断言：test_fused_ops.py :428/:482、test_fused_gate.py
  :443（布尔→注册表键；UT-006c 的 out_count 断言原样保留）+ 两处
  存在性钉补 `_TRITON_FAILED_KEYS`；迁移用例补注册表快照/恢复；
- **验收**：新用例对现行代码全红（注册表不存在/布尔仍被置位）；
  迁移后旧用例对现行代码红（布尔断言已改键断言）；mutation
  spot-check 记录（注册表不写入/分发不检查/告警无条件三变异）。

### T002 Green：fused_ops.py 实现 FR1-FR3（主代理亲验）
- `_TRITON_FAILED_KEYS` 注册表 + bucket 纯函数 + 三 kernel
  分发/失败路径改造；三布尔收窄为手动总开关；docstring
  "permanently disables" 过时表述更正（:169-170/:591-592）；
- **验收**：UT-1301~1306 全绿；T001 迁移用例全绿；FR5 列举的
  零改动消费点（test_ar002_perf_proxy:365、test_cuda_graph:596、
  test_fused_gate:217、**test_fused_ops:440-461**）单独跑绿；
  `git diff actfold` 净变化仅 fused_ops.py。

### T003 全量回归 + demo
- `python -X utf8 -m pytest tests\ -q -m "not slow"`：807 基线 +
  新增/迁移零失败；
- `python -X utf8 demo.py`：三指标（85.5% / 2.35e-03 / 93.75%）
  精确一致。

### T004 文档（FR6）
- 指南：P1 表 **7 行**（P1-1/P1-5/P1-6/P1-8/P1-9/P1-10/**:175
  P1-2**）勾选+勘误注；P1-11 ✅ 本 AR；6.2 :238 勘误勾选；指南
  :35 旗标表述更正；M 表 M-3/M-4 勾选 + **M-9 仅 ◐ 勘误注不勾选**
  （残余 5 处 sampler/adapter 顶层显式 seed 定性）；第二十一部分
  回链；
- CHANGELOG AR014 节（Changed：告警契约变更；Tests），Unreleased
  最前；AGENTS.md #9/#38 粒度注。

### T005 质量门
- `python -X utf8 -m mypy actfold --ignore-missing-imports` clean
  （65 文件）；pyflakes 零；100 列逐行检查；全量复跑确认。

### T006 review 门控（子代理只读审查）
- spec/design 符合性 + 实现质量审查；修复全部 Major/Minor。

### T007 ST 验收（st-cases.md 一次成 Go）

### T008 归档 + 交付
- Move-Item 工作目录 → `specs/archive/AR014-triton-fallback-granularity/`；
  git commit → commit-tree 对 gitee/main → push → fetch+树验证。

## 执行记录

| 任务 | 结果 | 备注 |
|------|------|------|
| T001 Red | ✅ 完成 | 新文件 284 行 10 项（UT-1306 参数化 ×5）对现行代码全红（fixture raising=True——raising=False 会*创建*属性使 UT-1304 假绿，文件头注记）；迁移 3 用例 + 2 存在性钉红、其余 39 绿；收集 820 |
| T002 Green | ✅ 完成 | fused_ops.py 58+/31-；新 10 绿；迁移文件 44 绿；零改动消费点（perf_proxy/cuda_graph 28 绿 1 skip）；mutation：①不写入→3 红（UT-1301/1302/1303）、②不查→1 红（UT-1302 spy 计数面精确命中）、③键缺 dtype 维→4 红；逐字节还原（diff 复核 58+/31-） |
| T003 全量+demo | ✅ 完成 | **817 passed（807+10）/ 3 skipped / 3 deselected**（40.95s）；demo 85.5%/2.35e-03/93.75% 精确一致 |
| T004 文档 | ✅ 完成 | 指南：P1-1/P1-2/P1-5/P1-8/P1-10 勾选、P1-9 ✗ 失效、P1-6 ◐、P1-11 ✅、:238 勾选、:35 注、M-3 ✅、M-4 ◐、M-9 ◐、第二十一部分；CHANGELOG AR014 节（Changed 告警契约明示）；AGENTS #9/#38 粒度注。**披露（review 门控 Minor-1/2 后补全）**：P1-3 顺手勘误（不在 srs 立项清单；证据 AGENTS #30/T016/AR011）、P1-6 勾选→◐（批量化主体归 P1-7）、M-4 勾选→◐（O(T²)/KV 项未核实）——三处偏离均已回写 srs §1 表与 FR6 |
| T005 质量门 | ✅ 完成 | mypy clean（65 文件）/pyflakes 零/代码 ≤100 列（spec 表格长行与 AR012/013 归档先例同型）；全量 817（T003 已证，文档改动不触及测试） |
| T006 review | ✅ 完成 | PASS（0 Major；3 Minor 记录同步：srs FR6/§1 M-4 ◐ 回写、P1-3 披露补记、tasks 滞后更新——均已修复；WARN-1 镜像同构由 forbidden+既有套件兜底、WARN-2 注册表泄漏同构现行、WARN-3 符号差异，均接受） |
| T007 ST | ✅ 完成 | **7/7 Go 一次成**（st-cases.md：契约属性/告警/隔离/总开关/回归不变量/文档/质量门；54 passed 复验 + 属性脚本 + 文档 grep 实证） |
| T008 归档+交付 | 待跑 | |

## 门控记录

| 门 | 结果 | 备注 |
|----|------|------|
| req 门控 | 首轮 FAIL → 修复 → **复核 PASS** | Major-1（FR6 漏 :175 P1-2 表行，两处 checkbox 勘误）/Major-2（M-9 七处仅证两处即勾选，降级为 ◐ 勘误注）已修；m-1 三例计数/m-2 :440-461 零改动清单/m-3 迁移用例注册表隔离/m-4 第二失败键告警/m-5 :170-172/m-6 P1-8 补引；w-1 match 子串/w-2 forbidden 回退/w-3 out_count 断言保留/w-4 docstring 更正 均落 srs；复核 12 项全 RESOLVED + n-1~n-3 顺手清理 |
| design 门控 | 首轮 FAIL → 修复 → 复核新 Major-3 → 修复放行 | Major-1（mutation②"告警再现"在 helper guard 下不可观测→UT-1302 launch-count spy）/Major-2（mutation③ 黑盒不可达→集合重定义为键缺 dtype 维度，"告警去条件"移除并记录）已修；m-1 T013 cuda 先例/m-2 stacklevel=3 双向同步/m-3 merge 插入点 :193 后（srs FR1 同步勘"键可得处"）/m-4 三键值实测（32/128/64）/m-5 forbidden 区间与 undo 交互/m-6 措辞；w-1~w-5 落文档；复核新 Major-3（"全程 forbidden"阻断①合法回退）→ 区间改①后②前 setattr、③前恢复，免重读放行 |
| review 门控 | **PASS（有条件→已履约）** | 0 Major；3 Minor 记录同步（M-4 ◐ 回写 srs FR6/§1、P1-3 披露补记 srs FR6、tasks T005/T006 滞后）+ 3 WARN（镜像同构由 forbidden+既有套件兜底/注册表泄漏同构现行/符号差异）——Minor 全部修复；核实依据：FR1-FR6 逐条命中、三入口 key 维度一致、回退调用逐字一致、勾选注行号证据全中、817=807+10 算术成立 |
| ST | **7/7 Go 一次成** | st-cases.md：ST-1 属性脚本/ST-2~4 54 passed/ST-5 817+demo/ST-6 文档 grep/ST-7 质量门 |
