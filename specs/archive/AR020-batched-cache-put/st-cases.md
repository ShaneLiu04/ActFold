# [AR020] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR020 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-14 |

> 执行环境：Quadro RTX 5000 16GB，torch 2.5.1+cu121，WDDM，驱动 556.18，python 3.11.9，
> pytest 9.1.1；测试命令统一 `python -X utf8`，全量口径 `-m "not slow"`（3 deselected 为
> 既有环境事实）。工件级用例（ST-009/010）直接断言
> `results/optimization/ar020_publish_put_bench.json`（reps=30 预承诺单次定案工件）。

## 测试用例列表

### ST-001：三 cache put_batch 状态等价性（H1 UT 面）

**关联需求：** srs.md §3.1 FR1 等价性（硬性）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- AR020 实现已合入；三 cache 均实现 `put_batch`

**测试步骤：**
1. `python -X utf8 -m pytest tests\test_batched_put.py -k "ut2001 or ut2002 or ut2003" -q`

**期望结果：**
- 三 cache（Activation/Vectorized/Chunked）× CPU/CUDA，固定场景+种子固定随机化性质测试全绿；比较面=buffer[:最终 _counts] torch.equal + _counts + OrderedDict 键序 + num_entries

**实际结果：** 12 passed（2001/2002/2003 各固定+随机化 × cpu/cuda）
**状态：** PASS

---

### ST-002：BatchedPutCapable 结构化协议 isinstance 面

**关联需求：** srs.md §3.1 FR1 能力协议
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- `cache_protocol.py` 定义 `@runtime_checkable` Protocol

**测试步骤：**
1. `python -X utf8 -m pytest "tests\test_batched_put.py::test_ut2006_protocol_and_isinstance" -q`

**期望结果：**
- 三 cache 均满足 isinstance；无 put_batch 的对象不满足；既有 `ActivationCacheProtocol` 不受影响

**实际结果：** 1 passed
**状态：** PASS

---

### ST-003：发布批量分流 + 条目负载逐字段对拍

**关联需求：** srs.md §3.2 FR2 批量变体
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- `_publish_graph_children` 具备能力分流

**测试步骤：**
1. `python -X utf8 -m pytest "tests\test_batched_put.py::test_ut2006_publish_dispatch_batched_payload" -q`

**期望结果：**
- 有能力 cache 走一次 `put_batch`；条目负载与逐条版逐字段相同（快照行视图、layer 0 embedding 行、step_idx 省略=0）

**实际结果：** 1 passed
**状态：** PASS

---

### ST-004：put_batch 直连账本 O(L+N)

**关联需求：** srs.md §3.1 FR1 copy 批量化（H4 组件面）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- `write_op_count` 账本已实现

**测试步骤：**
1. `python -X utf8 -m pytest "tests\test_batched_put.py::test_ut2007_put_batch_direct_ledger" -q`

**期望结果：**
- 直连 put_batch 账本按 (device,dtype) 分组数+立即执行数计数（分组混批面）

**实际结果：** 1 passed
**状态：** PASS

---

### ST-005：发布账本 loop vs batch

**关联需求：** srs.md §3.3 FR3 主机调用账本
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 发布路径两变体均可触发

**测试步骤：**
1. `python -X utf8 -m pytest "tests\test_batched_put.py::test_ut2007_publish_ledger_loop_and_batch" -q`

**期望结果：**
- loop 发布账本 L·N+N，batch 发布账本 O(L+N)（组数级）

**实际结果：** 1 passed
**状态：** PASS

---

### ST-006：每次发布探测 + cache 换装重探（property 传播）

**关联需求：** srs.md §3.2 FR2 能力探测
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- `ManualFoldedForward.cache` 为 property + setter 传播

**测试步骤：**
1. `python -X utf8 -m pytest "tests\test_batched_put.py::test_ut2006_probe_each_publish_and_cache_swap" -q`

**期望结果：**
- 每次发布重探（零实例缓存）；换 cache 后 wrapped layers 与两 runner 引用同步刷新，批量图不静默 eager 回退

**实际结果：** 1 passed
**状态：** PASS

---

### ST-007：profiler 面零变化

**关联需求：** srs.md §3.2 FR2（AR018 双入口语义保持）
**测试类型：** 正常路径
**优先级：** Medium

**测试步骤：**
1. `python -X utf8 -m pytest "tests\test_batched_put.py::test_ut2006_profiler_face_unchanged" -q`

**期望结果：**
- `record_precomputed`（sums）与 debug `record`（逐行）在批量分支下行为与 loop 分支一致

**实际结果：** 1 passed
**状态：** PASS

---

### ST-008：零 host-sync + 快照新鲜度 + debug 逐行 record

**关联需求：** srs.md §3.2 FR2 零 host-sync 契约 + 快照新鲜度论证
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest "tests\test_batched_put.py::test_ut2008_zero_host_sync_and_freshness" "tests\test_batched_put.py::test_ut2008_debug_path_per_row_record" -q`

**期望结果：**
- put_batch 内无 `.item()`/`.tolist()`/`.cpu()`/隐式同步；发布后本侧对快照零写入；debug 路径逐行 record

**实际结果：** 2 passed
**状态：** PASS

---

### ST-009：基准工件有效性（H1-H4 判定 + 正确性门 + meta 冻结）

**关联需求：** srs.md §3.3 FR3 工件 schema（假设-判定分离）
**测试类型：** 正常路径（工件级断言）
**优先级：** High

**前置条件：**
- `results/optimization/ar020_publish_put_bench.json` 为 reps=30 预承诺单次定案工件

**测试步骤：**
1. python 断言脚本读取工件：`hypotheses_verdicts` 全 supported；`failed_configs` 空；每配置 `correctness_gate.bit_equal=True` 且 checked_tensors>0；meta `c_constant==2`、`reps==30`、预承诺注记存在、`verdict_formulas`（H1-H4+aggregation）冻结

**期望结果：**
- H1/H2/H3/H4 全 supported；正确性门 116/232/116 张量 bit-equal；判定式在运行前冻结

**实际结果：** 三配置 gate bit_equal=True（tensors 116/232/116）；H1-H4 全 supported；failed_configs=[]；meta c=2、reps=30、verdict_formulas 冻结——ST-009 PASS
**状态：** PASS

---

### ST-010：端到端零回退 + pe canary + 预算 + VRAM

**关联需求：** srs.md §3.3 FR3 端到端宏基准（H3 组件）
**测试类型：** 正常路径（工件级断言）
**优先级：** High

**测试步骤：**
1. python 断言脚本读取工件 `end_to_end.batch_canary`：`batch_no_fallback=True`、`batch_publishes==expected_steps`（38/38）、`pe_identity_mismatches` 全 0、`steady_state_budget_violations==0`；`peak_vram_mb < 16000`

**期望结果：**
- batch 变体全路径零静默回退；pe 恒等 canary 零失配；稳态零预算违例；峰值 VRAM 留余量

**实际结果：** 三配置 batch_no_fallback=True、publishes 38=38、loop_put_calls 4284/8540/4284（与 28+38×(L·N+N) 精确一致）、pe/预算全 0、VRAM 5714/8901/10027MB——ST-010 PASS
**状态：** PASS

---

### ST-011：文档收口五件齐备 + 头条数字与工件一致

**关联需求：** srs.md §3.4 FR4
**测试类型：** 正常路径（文档级检查）
**优先级：** Medium

**测试步骤：**
1. 检索五件：GUIDE 第二十七部分标题、CHANGELOG AR020 条目、AGENTS.md #42、RERUN_CHECKLIST ar020 行、README 头条（发布窗口 p50 表 + 7.894/2.40×/920/put_batch 关键词）

**期望结果：**
- 五件全存在；README 头条数字可溯源至工件

**实际结果：** 9/9 检索全命中（GUIDE/CHANGELOG/AGENTS/RERUN/README×5）
**状态：** PASS

---

### ST-012：环形/grow/重复目的地重叠等价 + 冲刷写入序

**关联需求：** srs.md §3.1 FR1 容量语义保持（环形布局条目走既有逐条等价路径）
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\test_batched_put.py::test_ut2005_ring_grow_overlap_equivalence -q`

**期望结果：**
- seq_len > capacity 环形写、grow、重复逻辑目的地重叠场景全等价；ring index_copy 恒立即；冲刷写入序=条目序断言通过

**实际结果：** 4 passed（cache_kind × device 参数化）
**状态：** PASS

---

### ST-013：eviction 密集随机化等价（驱逐/重建/grow 交叉）

**关联需求：** srs.md §3.1 FR1 等价性（eviction 结果一致）
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\test_batched_put.py -k "randomized" -q`

**期望结果：**
- 种子固定随机化性质测试（小预算×多分支×重复 key×grow；branch-step 驱逐；逐插入驱逐）全绿——逐条驱逐等价性由构造保证

**实际结果：** 6 passed（三 cache 随机化 × cpu/cuda）
**状态：** PASS

---

### ST-014：旧 torch `_foreach_copy_` 不可用降级

**关联需求：** srs.md §3.1 FR1 兼容性（降级为内部逐条路径，结果等价、不告警）
**测试类型：** 边界条件
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest "tests\test_batched_put.py::test_ut2009_foreach_downgrade_equivalence_and_ledger" "tests\test_batched_put.py::test_ut2009_publish_downgrade_dispatch" -q`

**期望结果：**
- patch `_FOREACH_AVAILABLE` → 逐对 copy_，结果等价、账本按对数；发布降级分流正确；事后 patch torch 属性不生效属预期

**实际结果：** 2 passed
**状态：** PASS

---

### ST-015：批量前置校验原子性（非法条目三态 × 位置）

**关联需求：** srs.md §3.1 FR1 原子性（零部分写入）
**测试类型：** 异常处理
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\test_batched_put.py::test_ut2004_atomicity_zero_writes -q`

**期望结果：**
- 任一位置（首/中/尾）的非法条目（空激活 dict / ndim<2 / 前导形状不一致）→ `ValueError` 带条目序号，缓存状态零改动

**实际结果：** 27 passed（三态 × 位置参数化）
**状态：** PASS

---

### ST-016：发布异常原样上抛

**关联需求：** srs.md §3.2 FR2 异常处理（不吞不改档）
**测试类型：** 异常处理
**优先级：** Medium

**测试步骤：**
1. `python -X utf8 -m pytest "tests\test_batched_put.py::test_ut2006_publish_exception_propagates" -q`

**期望结果：**
- put_batch 抛出的异常类型/消息原样上抛，无静默回退

**实际结果：** 1 passed
**状态：** PASS

---

### ST-017：无能力 cache 回退原循环（逐位不变）

**关联需求：** srs.md §3.2 FR2 回退变体 + §4 语义基线（既有发布路径逐位不变）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest "tests\test_batched_put.py::test_ut2006_publish_incapable_fallback_loop" -q`
2. 全量套件（ST-018）中既有发布/图/cache 测试零改动通过

**期望结果：**
- duck-typed 无能力 cache 走原双循环（含 debug record 回退），行为与既有完全一致

**实际结果：** 1 passed（+全量零扰动见 ST-018）
**状态：** PASS

---

### ST-018：全量回归（864 基线 + 56 新增）

**关联需求：** srs.md §4 非功能需求（回归）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -X utf8 -m pytest tests\ -q -m "not slow"`

**期望结果：**
- 920 passed / 3 skipped（= 864 既有基线 + 56 新增），零既有失败

**实际结果：** 920 passed, 3 skipped, 3 deselected（slow）, 40 warnings（全部既有预期警告）, 46.60s
**状态：** PASS

---

### ST-019：静态质量门 + demo 语义锚点

**关联需求：** srs.md §4 非功能需求（静态/语义）
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. `python -m pyflakes actfold tests demo.py scripts`
2. `python -m mypy actfold --ignore-missing-imports`
3. `python -X utf8 demo.py`

**期望结果：**
- pyflakes 零输出；mypy clean；demo 三锚点 85.5% / 2.35e-03 / 93.75% 精确不变

**实际结果：** pyflakes exit 0（零输出）；mypy "Success: no issues found in 67 source files"；demo 输出 "Total FLOPs reduction: 85.5% / Output equivalence (MSE): 2.35e-03 [HIGH] / Estimated stable token ratio: 93.75%"
**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-14 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮（一次成 Go） |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR1 | put_batch 批量写能力（等价/原子/批量化/容量/协议/兼容） | ST-001, ST-002, ST-004, ST-012, ST-013, ST-014, ST-015 | PASS |
| §3.2 FR2 | 发布路径消费（分流/负载/回退/零 sync/新鲜度/异常） | ST-003, ST-005, ST-006, ST-007, ST-008, ST-016, ST-017 | PASS |
| §3.3 FR3 | 基准脚本 + 工件（H1-H4/正确性门/无回退） | ST-009, ST-010 | PASS |
| §3.4 FR4 | 文档收口五件 | ST-011 | PASS |
| §4 NFR | 回归/静态/语义基线 | ST-017, ST-018, ST-019 | PASS |

**需求覆盖率：** 5 / 5（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 11 | 11 | 0 | 0 |
| 边界条件 | 3 | 3 | 0 | 0 |
| 异常处理 | 2 | 2 | 0 | 0 |
| 回归测试 | 3 | 3 | 0 | 0 |
| **合计** | **19** | **19** | **0** | **0** |

（UT 粒度执行计数：56/56 用例在全量套件中全绿；ST 分组运行另计 12+4+6+27+8×1+2+2=65 项次。）

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| （无） | — | — |

### 结论

> **Go**。19/19 ST 用例一次全 PASS，零 Critical/Major/Minor 缺陷；需求覆盖 5/5（100%）；
> 全量 920 passed/3 skipped 零回归；静态质量门（pyflakes 零/mypy 67 文件 clean）与 demo
> 三锚点（85.5%/2.35e-03/93.75%）精确达标；工件 H1-H4 全 supported 且 ST-009/010 断言
> 逐项核实（正确性门 bit-equal、零回退、pe canary 零失配、VRAM 留余量）。满足全部 Go
> 条件，归档 AR020。（用户既定全自动指令：ST Go/归档/推送无需逐步确认。）
