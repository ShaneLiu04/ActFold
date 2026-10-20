# [AR029] 任务清单

**AR029：端侧有界内存金字塔激活缓存**（srs.md 同目录；本文件为唯一任务真源）

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 设计：`design.md`（预算公式、块页存储、可用性守卫、协议扩展、AR028 冲突边界、UT 矩阵）+ 设计门控 PASS | srs | passing | 先于任何编码 |
| T002 | Red UT（计划面）：不可变/归一预算/linear+geometric/总和不超/边界/非法参数/确定性 | T001 | passing | UT-2901~2910 |
| T003 | Green `actfold/core/pyramid_plan.py`：`PyramidPlan`、`build_pyramid_plan`、`retained_blocks(layer, seq_len)` | T002 | passing | 模块顶层 stdlib+dataclasses |
| T004 | Red UT（缓存面）：块切分/逐层硬上界/保留集/淘汰/最后不整块/确定性/非法输入/bytes 精确/条目删除/统计六键/最小预算边界/无 fetch_flat | T003 | passing | UT-2911~2925, UT-2969~2972 |
| T005 | Green `actfold/core/pyramid_cache.py`：块页存储、逐层预算淘汰、`stats()`、`put_batch` 校验 | T004 | passing | 无逐 token 循环 |
| T006 | Red UT（安全面）：fetch/get/fetch_masked 的 NaN 与请求语义、`availability`（parent id + step 0 + 末块偏移）、三 metric、全量预算逐位、FR3 #3/#4、tau=0.9/0.0、profiler 守卫后统计、split 继承、1-D/2-D rank | T005 | passing | UT-2926~2944, UT-2968, UT-2974~2976 |
| T007 | Green 读路径 + `MissingSentinelCapable` 协议 + MFF 串行可用性守卫（含 fused count 失效） | T006 | passing | `cache_protocol.py` 文档扩展 |
| T008 | Green 同父批量/多父逐行守卫（每行用其 parent id；1-D 不扩维）+ MFF graph×pyramid **init+setter** 双校验（先校验后传播，失败不半传播） | T007 | passing | UT-2939~2944, UT-2967；合并锚点见 srs §5 |
| T009 | Red/Green 协议与回归（若 AR028 已合入：加跑其 UT-2804/2805/2814/2816 + availability no-op 断言）：全量预算 vs `ActivationCache` 串行/批量逐位、`put_batch` 等价、`write_op_count`/`num_entries`、clear/contains | T008 | passing | UT-2945~2952；冻结面 AR015/AR021/AR025 |
| T010 | 工厂/配置/导出/接线：`use_pyramid` 最高优先级、`ActFoldConfig.pyramid_*`、YAML 无告警、`__init__` 导出、BenchmarkRunner 两处构建 + 统一真实深度 resolver + both-flags `ValueError` | T009 | passing | UT-2953~2958, UT-2973 |
| T011 | 机制基准 `scripts/ar029_pyramid_cache_bench.py`：真实小模型、E1-E4 冻结判定、quick 工件、判定函数 UT | T009 | passing | UT-2959~2966（含 UT-2963b 关闭守卫）、UT-2977~2978；aseline_claim=false` |
| T012 | 文档草稿：README 六节重写（创新表/结果/限制，ladder 行 BLOCKED-DEFERRED）、RERUN 服务器协议（BLOCKED-DEFERRED，未来须先加 FaceSpec）、AGENTS 新条目、GUIDE、CHANGELOG、ALGORITHM/EXPERIMENTS | T010, T011 | passing | 数字全部来自工件；aseline_claim=false |
| T013 | 质量门：black/isort/pyflakes/mypy(70+27)/全量 pytest/demo 三锚点 | T010, T011 | passing | 只增不减；产出最终测试计数 |
| T014 | 文档终稿：用 T013 的最终计数刷新 README 质量段与 RERUN 计数，并复核六节一致 | T012, T013 | passing | 避免 README 过期 |
| T015 | review 门控（只读子代理：spec/design 合规 + 代码审查 + 安全路径复核）循环至 PASS | T014 | passing | 记录轮次/代理 id |
| T016 | ST 验收（含 AR028 合入后的 rebase 回归证据）：st-cases.md + 执行（正常/边界/异常/回归）；归档 + gitee 分段推送与树校验；归档说明「机制交付、ladder BLOCKED-DEFERRED」 | T015 | passing | 真实 checkpoint 面 BLOCKED-DEFERRED |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 约束（进度记录）

- 折叠数学不变；唯一核心改动是三处可用性守卫 + graph×pyramid 互斥校验。
- 与并行会话 AR028 同文件（folded_transformer.py）：以独立 helper + 函数边界
  合并；本 AR 测试不得依赖 AR028 未合并的行为。
- 缺失 token 必须不可用（NaN + availability）→ 重算；任何静默零值/NaN 复用视为缺陷。
- 本机不跑真实 checkpoint；aseline_claim=false`。
- 用户已授权「不做大型实验」的范围裁定；正式真实面由服务器回填。

## 进度记录

### 2026-10-20 实现完成（T002-T011 passing）

- 计划/缓存/协议/守卫/接线/基准全部落地；AR029 套件 94 passed，全量
  1273 passed / 4 skipped / 3 deselected；black/isort/pyflakes/mypy(70+27) 全清。
- 实现期勘误（已写入 design.md §4.2.3）：原冻结的统一预算压力面不可执行
  （门层缺失使守卫冗余；子分支写 L0 先淘汰父 L0 导致整层重算，单 forced mask
  无法复现）。改为线性金字塔 `[4·block, 1·block]` + `_HoleMaskedDense` 逐层
  NaN-oracle；quick 工件 E1-E4 全 supported。
- 基准脚本修复：`_run_face` 必须对 child 折叠（初版误用 root 双写）；
  full 预算放大到 `4·T` 以容纳 root+child 双分支；新增 reference 测量块。
- T012 文档草稿开始：README 六节重写、RERUN（BLOCKED-DEFERRED）、AGENTS #48、
  GUIDE、CHANGELOG、ALGORITHM/EXPERIMENTS。


## 阶段门控记录

### 2026-10-19 需求门控第 1 轮 — FAIL（代理 ses_eac0cb868ffeo9hcJ1YEYaNZ3M）

- 主要问题：NaN 单独不足以守卫（门读 L-1、合并用 L）；CUDA graph 不兼容未声明；
  预算公式总和矛盾；forced_divergent 参考不可执行；协议零填充契约冲突；
  机制 AR 未做范围裁定；executor 扩展缺失；工厂优先级未定；任务缺设计阶段与 UT 映射；
  AR028 重叠低估。
- 修复：引入 vailability 目标层守卫 + graph×pyramid 互斥；预算归一化并固定
  depth=1/上界；用统一保留集的 forced 参考 + 合法 tau 下界；新增
  MissingSentinelCapable 与 ndim>=2 口径；aseline_claim=false）；固定工厂优先级/导出；任务改为 T001 设计先行并补
  UT-2901~2966 与五件文档；记录 AR028 同文件合并边界。

### 2026-10-19 需求门控第 2 轮 — FAIL（代理 ses_eabf1fb47ffepIcG9wK2pUW59H）

- 问题：total<depth*block 时硬上界不可满足；守卫顺序未精确到 profiler 之前且未指明
  parent id；FR2/FR4 的 ndim 表述矛盾；协议零填充裁免未落到需求级；机制 AR 与
  AGENTS 规则/README ladder 未显式调和；任务缺 FR3 专项 UT 与 README 计数时序；
  AR028 合并锚点不精确。
- 修复：新增 depth*block>total → ValueError 与验收；守卫固定在 forced AND 之后、
  profiler/计数/快速路径之前，并指定 parent branch + 目标层；FR2 统一 
dim<2；
  协议文档修订为「零填充仅约束非哨兵缓存」；明确机制/基础设施 AR 豁免 + ladder
  BLOCKED-DEFERRED + 未来正式面先加 FaceSpec；补 UT-2926~2944（含 FR3 #3/#4、
  profiler、split）；T012 草稿→T013 质量→T014 计数刷新→T015 review→T016 ST；
  合并锚点写死为命名 helper 调用点。

### 2026-10-19 设计门控第 1 轮 — FAIL（代理 ses_eabc0afcbffe0RfRW2N4V6hdPX）

- 问题：graph 仅 init 校验，cache setter 可绕过；availability 传 forward step_idx 与
  put/fetch 的 step 0 key 不一致；BenchmarkRunner 未接线且默认深度 32 不安全；
  FR7 无 E1-E4 具体设计；AR028 合并锚点缺失；统计/put_batch/条目删除语义未冻结。
- 修复：init+setter 双校验并先置 use_cuda_graph；availability 固定 step 0；
  BenchmarkRunner 两处接线 + 真实深度 + 强制关图；新增 4.2.3 基准冻结设计与
  4.2.4 AR028 锚点；新增 4.1.6/4.1.7/4.1.8（条目删除/统计/写失败/复用审计/回收）；
  UT 扩到 2973。

### 2026-10-19 设计门控第 2 轮 — FAIL（代理 ses_eabad6969ffeOGvLQNmdFiLNCt）

- 问题：helper 对 AR028 的 1-D mask 会错误扩成 [T,T]；E4=blocked 与 overall 公式
  矛盾；E2/E3 量化来源未定义；工件缺 E3/E4；SRS 仍传 step_idx；availability 末块
  部分长度会误报；UT 未分配/表格行损坏；rebase 未含 AR028 套件。
- 修复：helper rank 合同（1-D 永不扩维、切片 [..., :v_r]）；判定优先级
  blocked→supported→refuted；冻结 E2/E3 公式与工件四键；SRS/§5 统一 step 0；
  availability 加块内偏移；UT 扩至 2978 并分配到 T006/T010/T011；修表格行；
  rebase 加入 AR028 功能套件与 no-op 断言；runner both-flags 改为显式 ValueError。

### 2026-10-19 设计门控第 3 轮 — FAIL（代理 ses_eab978f12ffefsb9LR5j2z9iPN）

- 问题：UT-2967/2968/2973~2976 未分配到任务；E2 自比较且语法不可执行；E1 只定义
  一个比较面；E4 的 stat_values 未定义；_build_engine 深度来源不明确。
- 修复：T006/T008/T010 显式分配全部 UT 并加入 setter/接线职责；E2 改用
  ytes_used + iter_block_tensors() 独立字节和 + stats 三方核对；E1 定义
  full/pressure 两个 diff；E4 摊平数值叶子；新增共享 _resolve_pyramid_depth。

### 2026-10-19 设计门控第 4 轮 — FAIL（代理 ses_eab7de68fffeRi9KI66Jg3c1t4）

- 问题：UT-2963b 悬空；AR028 rebase 无任务归属；E2 伪代码 stats 未绑定/未用 .get；
  E4 outputs 未定义；深度 resolver 对 raw adapter 误失败；retained_blocks 可越界；
  工件缺 FR5 测量键。
- 修复：T011 注明 UT-2963b；T009/T016 加 AR028 rebase 与套件回归；E2 绑定 stats 并
  .get；E4 定义 outputs；resolver 遍历候选；锚点 clamp 到 n_blocks；schema 补
  full/pressure/reference 三面与 FR5 五键。

### 2026-10-19 设计门控第 5 轮 — FAIL（代理 ses_eab6641b1ffeC9IozzTuaZ9nYs）

- 问题：E1 压力参考写成 
orced_divergent=availability_mask，极性与语义相反。
- 修复：改为 missing_mask = ~availability_mask；并冻结两面计划配置与
  	okens_stored/missing_ratio/stable_ratio/wall_ms 公式。

### 2026-10-19 设计门控第 6 轮 — PASS（代理 ses_eab5227daffeLpQ7mwN60z9B5L）

- PASS；6 条非阻塞备注已并入设计：守卫关闭篡改用逐层缺失面、availability 快照时机、
  E2 两面求值、full/reference 指标来源、depth/block 校验 UT、layer-0 专项 UT、
  recent 窗口仅保护刚写入 entry。


### 2026-10-20 review 与 ST 收口

- review 3 轮（只读子代理）：r1 `ses_eaae5d6baffeuEBng5Gy53EKOQ` FAIL
  （isort、守卫测试同义反复、full 指标归因、零长写入、锚点公式、UT 覆盖）→
  r2 `ses_eaab3f88affejM4a572R6ZS0UQ` FAIL（UT-2934/2940 缺失与计数）→
  r3 `ses_eaa9e949effeguI7O6bTJ641Nx` **PASS**（零 findings）。
- 修复落点：守卫测试改用 gate-stable 子输入 + identity monkeypatch 证明
  负载性；补 UT-2934/2940/2946/2949/2952/2957b/2961b/2963b/2967b/2971/2972；
  full 面改真实 `_Trace` 计时与复用；`_validate_activations` 拒绝零长写入；
  `retained_blocks` 纳入 block 0（对齐冻结公式）；多父守卫统一走共享 helper。
- ST：st-cases.md 15 例（FR1–FR7 全覆盖）全部 PASS；AR029 套件 94 passed，
  全量 1273 passed / 4 skipped / 3 deselected；black/isort/pyflakes/mypy 全清。
- 归档：本目录 `specs/archive/AR029-edge-pyramid-cache/`；正式真实 checkpoint 面
  BLOCKED-DEFERRED（`baseline_claim=false`，未来须先加 `run_formal_queue` FaceSpec）。
- 重复编号草案：`specs/archive/AR029-hierarchical-fold-memory-superseded/`
  （superseded 说明，不参与验收）。
