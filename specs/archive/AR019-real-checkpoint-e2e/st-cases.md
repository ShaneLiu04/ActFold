# [AR019] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR019 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-14 |

> 执行环境：Quadro RTX 5000 16GB，torch 2.5.1+cu121，WDDM，驱动 556.18，种子 2026；
> 真实 checkpoint 基准另需 AR019 venv（transformers==4.53.1）+ Fast-dLLM-v2-1.5B 本地权重。
> 测试命令统一 `python -X utf8`；全量口径 `-m "not slow"`（本机未装 bench 依赖，slow 用例
> 为既有环境事实，沿 review 门控注记）。

## 测试用例列表

### ST-001：串行/批量 eager pe 透传

**关联需求：** srs.md FR1 验收标准 1（eager 面）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- AR019 实现已合入；`_ContextLayer` stub（签名含 pe、记录调用并消费 cos）

**测试步骤：**
1. 执行 `python -X utf8 -m pytest tests\test_position_embeddings.py::test_ut1901_serial_eager_pe_passthrough tests\test_position_embeddings.py::test_ut1902_batched_eager_pe_passthrough -q`

**期望结果：**
- 层收到的 pe 与供给值 `is` 同对象；输出与"手工焊 pe"对照路径一致

**实际结果：** 2 passed（对象恒等 + 对照一致断言全过）
**状态：** PASS

---

### ST-002：graph 捕获烘焙 + 重放稳定

**关联需求：** srs.md FR1 验收标准 1/4（graph 静态体）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- CUDA 可用；`use_cuda_graph=True` 批量路径

**测试步骤：**
1. 执行 `tests\test_position_embeddings.py::test_ut1903_graph_capture_bakes_pe`

**期望结果：**
- capture 后 replay 层收到的 pe 值与供给 bit-equal；重放 3 次稳定；静态体以 `is` 同对象消费 pe（spy 断言）

**实际结果：** 1 passed
**状态：** PASS

---

### ST-003：自适应重捕获烘焙

**关联需求：** srs.md FR1 验收标准 2（第四构建面）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- CUDA 可用；`graph_adaptive_capacity=True`；构造持续过配触发 shrink

**测试步骤：**
1. 执行 `tests\test_position_embeddings.py::test_ut1904_adaptive_recapture_bakes_pe`

**期望结果：**
- shrink 重捕获后新 runner 的 replay 层收到正确 pe（新 runner 烘焙面完整）

**实际结果：** 1 passed
**状态：** PASS

---

### ST-004：真实 checkpoint 端到端工件有效性

**关联需求：** srs.md FR2/FR3/FR4（基准 + 工件）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- `results/optimization/ar019_real_e2e_bench.json` 已由 reps=30 预承诺单次运行产出

**测试步骤：**
1. 载入工件并断言：H2/H3/H4 supported；三配置两 graph 路径 maxdiff=0.0 且 argmax=1.0；pe 恒一失配全 0；预算违例全 0；每配置 final_capacity ≥ 需求且 < fixed_capacity

**期望结果：**
- 全部断言通过；H1 如实记录（inconclusive，2/3 supported——首配置 WDDM 方差，不粉饰）

**实际结果：** artifact validation: PASS, 3 configs, H1 = inconclusive
**状态：** PASS

---

### ST-005：None 默认 + 合成层零变化（三子断言）

**关联需求：** srs.md FR1 验收标准 3（None 零语义变化双侧不变量）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- `_PlainLayer` stub（签名不含 pe、无 varkw）

**测试步骤：**
1. 执行 `tests\test_position_embeddings.py::test_ut1906_none_default_and_plain_layer_unchanged`

**期望结果：**
- a) 不传 kwarg 与现状逐位一致（钉）；b) 显式 `position_embeddings=None` 与 a) 逐位一致；c) pe 非 None 被 `_PlainLayer` 过滤忽略、输出与 a) 一致

**实际结果：** 1 passed（三子断言全过）
**状态：** PASS

---

### ST-006：重放期对象恒一静默回退

**关联需求：** srs.md FR1 验收标准 5（重放契约）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- CUDA 可用；pe_a 捕获 / pe_b（同形不同对象）重放

**测试步骤：**
1. 执行 `test_ut1905_replay_pe_object_identity`（CUDA graph 面）与 `test_ut1905e_eager_pe_semantics_companion`（CPU eager 伴随面）

**期望结果：**
- pe_b 重放静默回退 eager（无 UserWarning）且输出 = pe_b 语义（不用烘焙值）；eager 伴随面输出正确

**实际结果：** 2 passed
**状态：** PASS

---

### ST-007：结构校验 ValueError（六类畸形）

**关联需求：** srs.md FR1 异常处理（调用期显式档）
**测试类型：** 异常路径
**优先级：** High

**前置条件：**
- 无需 CUDA（纯宿主元数据校验）

**测试步骤：**
1. 执行 `tests\test_position_embeddings.py::test_ut1907_structural_validation_value_error`

**期望结果：**
- 非二元组 / 元素非张量 / 长度≠2 / seq 维失配 / head_dim 不一致 / 1-D cos（review N-2 补，不得漏 IndexError）全部 `ValueError`

**实际结果：** 1 passed（六子用例全过）
**状态：** PASS

---

### ST-008：捕获期 CUDA 拒绝

**关联需求：** srs.md FR1 异常处理（degradation 档）
**测试类型：** 异常路径
**优先级：** Medium

**前置条件：**
- CUDA 可用；pe 在 CPU

**测试步骤：**
1. 执行 `tests\test_position_embeddings.py::test_ut1908_capture_rejects_cpu_pe_silently`

**期望结果：**
- CPU pe 静默拒绝捕获（无告警无 disable）、eager 输出正确（CUDA gate 下跳图）

**实际结果：** 1 passed
**状态：** PASS

---

### ST-009：全量回归零扰动

**关联需求：** srs 质量基线（864 = 855 + 9 新）
**测试类型：** 回归
**优先级：** High

**测试步骤：**
1. 执行 `python -X utf8 -m pytest tests\ -q -m "not slow"`

**期望结果：**
- 864 passed / 3 skipped / 3 deselected；既有测试零改动（review 门控 git status 核验）

**实际结果：** 864 passed, 3 skipped, 3 deselected（review 修复 N-2/N-3 后复跑一致）
**状态：** PASS

---

### ST-010：demo 三指标锚点

**关联需求：** srs 质量基线（demo 语义零扰动）
**测试类型：** 回归
**优先级：** Medium

**测试步骤：**
1. 执行 `python demo.py`

**期望结果：**
- FLOPs reduction 85.5% / MSE 2.35e-03 (HIGH) / stable ratio 93.75% 与基线精确一致

**实际结果：** 85.5% / 2.35e-03 [HIGH] / 93.75% ✓
**状态：** PASS

---

### ST-011：静态质量门

**关联需求：** srs 质量基线
**测试类型：** 回归
**优先级：** Medium

**测试步骤：**
1. `python -m mypy actfold --ignore-missing-imports`
2. `python -m pyflakes actfold tests scripts demo.py`
3. 新增/改动文件 100 列检查

**期望结果：**
- mypy clean（66 文件）；pyflakes 零输出；100 列零违规

**实际结果：** Success: no issues found in 66 source files / pyflakes exit 0 / 三个 AR019 文件 over100 全空
**状态：** PASS

---

## ST 结论

**Go**——11/11 用例一次通过（正常 4 / 边界 2 / 异常 2 / 回归 3），无遗留缺陷。
H1 整体 inconclusive 为预承诺单次运行下的如实记录（2/3 配置 supported，三配置 p50
加速比 2.44/2.62/2.59× 方向一致），非缺陷非阻塞。
