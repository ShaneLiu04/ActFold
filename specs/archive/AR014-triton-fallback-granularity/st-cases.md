# [AR014] ST 验收用例 — Triton 降级粒度细化 + 优先级表勘误

> srs：`specs/changes/AR014-triton-fallback-granularity/srs.md`
> 执行：2026-10-13；结果：**7/7 Go（一次成）**

| # | 用例 | 验收标准（srs 映射） | 执行证据 | 结果 |
|---|---|---|---|---|
| ST-1 | 注册表/桶/总开关契约属性 | FR1：`_TRITON_FAILED_KEYS` 为 set；`_hidden_bucket` 4096→4096/5000→8192/1→1；三布尔初始 False（手动总开关） | `python -c` 断言脚本全部通过（"ST-1 attribute/bucket/kill-switch OK"） | **Go** |
| ST-2 | 每键一次性告警 + 可诊断内容 | FR2：首失败一条 RuntimeWarning（kernel/dtype/hidden-class 三要素 + 异常透传）；同键次调零告警零尝试 | UT-1302（launch-count spy 计数==0 + catch_warnings 零告警）+ UT-1305（三子串）通过；match 子串保留由迁移后 T013/UT-006c `pytest.warns(match=...)` 绿证 | **Go** |
| ST-3 | 隔离语义 | FR3：dtype 维（UT-1301 两键两告警）、bucket 维（UT-1303 512 爆/2048 真）、跨 kernel（迁移后 T013 三例 + test_fused_ops:440-461 零改动） | `pytest tests\test_triton_fallback_granularity.py tests\test_fused_ops.py tests\test_fused_gate.py -q` → **54 passed** | **Go** |
| ST-4 | 手动总开关兼容 | FR1：`_TRITON_MERGE_DISABLED=True` 强制回退优先于注册表；monkeypatch 调试面不变 | UT-1304（结果正确/零告警/注册表零增长）+ test_ar002_perf_proxy:365 与 test_fused_gate:217 零改动通过（T002 单跑 28 绿 1 skip） | **Go** |
| ST-5 | 全量回归 + demo 不变量 | NFR：807 基线 + 新增零失败；demo 85.5%/2.35e-03/93.75% 精确一致 | T003：**817 passed（807+10）/ 3 skipped / 3 deselected**（40.95s）；demo 三指标逐字一致 | **Go** |
| ST-6 | 文档收口 | FR6：P1 表勾选（P1-1/P1-2/P1-5/P1-8/P1-10 ✅、P1-9 ✗、P1-6 ◐、P1-11 ✅）+ :238 + M-3 ✅/M-4 ◐/M-9 ◐ + :35 注 + 第二十一部分 + CHANGELOG AR014（Changed 告警契约明示）+ AGENTS #9/#38 粒度注；偏离披露（P1-3/P1-6/M-4）回写 srs | grep 实证：指南五项标记在位、"第二十一部分" 8 处命中、CHANGELOG/AGENTS 各 2 处 AR014 命中；review 门控 Minor-1/2 修复后 srs FR6 与交付物一致 | **Go** |
| ST-7 | 质量门 | NFR：mypy clean（65 文件）/pyflakes 零/代码 ≤100 列 | T005：`mypy → Success: no issues found in 65 source files`；pyflakes 零输出；fused_ops.py 与三测试文件零超列（spec 表格长行与 AR012/013 归档先例同型） | **Go** |

**结论**：7/7 Go，一次成。AR014 验收通过，进入 T008 归档交付。
