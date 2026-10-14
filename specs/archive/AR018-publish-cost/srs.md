# [AR018] srs — 发布路径宿主侧成本压缩：批量稳定度归约 + 快照发布（P1-7 第四档）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR018 |
| 主题 | 批量 graph 路径的**逐分支发布循环**（`_publish_graph_children` 的 L×N 次 clone+put + L×N 次逐行 `mask.sum()` 归约）是自适应步墙钟的最大单一成本项——探针实测占全步 **60.7%/71.2%/29.7%/31.5%**（(256,8)N=4/8、(1024,16)N=4/8，AR017 稳态容量下）。本 AR 以两项正交机制压缩：①**批量稳定度归约**（单次 `mask_buf.sum(dim=2)` → [L,N] 替代 L×N 次逐行归约 launch）；②**快照发布**（单次整块 `child_buf.clone()` fresh 快照 + 行视图 put，替代 L×N 次逐行 clone——`ActivationCache.put` 自带 `copy_`，逐行 clone 是纯冗余中间拷贝）。发布内容逐位不变，无新开关、无语义变化 |
| 日期 | 2026-10-14 |
| 状态 | draft（待 req 门控） |
| 前置 | AR016（`_publish_graph_children` 逐分支行切片 clone 发布 + UT-1502 完备性/UT-1509 隔离）；AR017（自适应容量稳态——探针与基准的容量前提）；AR002（`FoldedGraphRunner` 静态缓冲契约 #37） |

## 1. 背景与动机

### 1.1 证据链（本 AR 的深度依据，全部本机实测）

1. **AR016 首次显形**（工件 + Part 23 语义要点 3）：批量 graph 对批量 eager 的优势随
   N×T 增大收窄（2.6×→~1.04×），归因假设"宿主侧 prefill/读回/发布占比上升"
   ——**未分解定量**，Part 23 遗留清单明记"宿主侧成本分解"待办；
2. **AR017 收口容量项后遗留**（Part 24 遗留清单）："宿主侧固定开销分解（重放前
   prefill/预算读回/逐分支发布占比的定量拆分）——AR017 未触及该成本面"；
3. **AR018 相位分解探针**（`ar018_hostcost_probe`，2026-10-14，AR017 同夹具
   同种子，稳态自适应容量，n=30，perf_counter+sync 相位边界，四配置全测）：

   | 配置（H,L,N,T=512） | 全步 p50 | 发布 p50 | **发布占比** | 发布内 profiler | 读回 | prefill |
   |---|---|---|---|---|---|---|
   | (256,8,4) | 7.927ms | 4.809ms | **60.7%** | 43.5% | 0.059ms | 0.310ms |
   | (256,8,8) | 13.823ms | 9.835ms | **71.2%** | 37.2% | 0.065ms | 0.324ms |
   | (1024,16,4) | 30.688ms | 9.117ms | **29.7%** | 43.1% | 0.102ms | 0.829ms |
   | (1024,16,8) | 57.743ms | 18.167ms | **31.5%** | 41.3% | 0.132ms | 1.554ms |

   锚点校验：全步 p50 与 AR017 工件 adaptive p50 吻合（7.93↔7.12 /
   13.82↔13.92 / 30.69↔31.21 / 57.74↔59.12——锚点比对基于全步实测值
   本身，不依赖相位和；相位和与全步的偏差双向并存：sync 串行化正向
   vs 相位间宿主工作遗漏负向，实测 4 配置中 3 个 phase_sum < 全步）。
   **结论：小几何下自适应步的瓶颈根本不是
   重放计算（2.03ms），而是发布循环；读回与 prefill 合计 <5%，可忽略**。
4. **发布循环的结构性浪费**（代码事实）：
   - `_publish_graph_children`（architecture_utils.py:961-1001）双循环
     L×N 次，每次 ①`child_buf[l][r:r+1].clone()`——而
     `ActivationCache.put` 内部 `buffer[:seq_len].copy_(transposed)`
     **自带拷贝**（activation_cache.py put 体），逐行 clone 是
     **纯冗余中间拷贝**（每步 2·L·N 次设备拷贝，一半可消除）；
     ②`GLOBAL_STABILITY_PROFILER.record(...stable_mask=mask_buf[l][r:r+1])`
     ——record 内 `stable_mask.sum()`（stability_profiler.py:161）每次一个
     归约 launch，L×N 次/步；WDDM 下每 launch ~10-20µs，(256,8,8) 仅
     profiler 归约即 ~3.7ms；
   - `mask_buf` 为 **[L,N,T] 单张量**、`child_buf` 为 **[L,N,T,H] 连续单张量**
     （cuda_graph.py:237-243）——批量归约（`sum(dim=2)` → [L,N]，1 次
     launch）与整块快照（1 次 clone）的形状前提天然成立。

### 1.2 机会

- **机制量化**：发布循环每步 = L×N×(clone + put 拷贝 + sum 归约) ≈
  3·L·N 次 launch（本 AR 探针配置下 L×N ∈ {32, 64, 64, 128}）。两项
  机制后 = 1×(整块 clone) + L×N×(put 拷贝) + 1×(批量归约) ≈ L·N+2 次
  ——launch 数 ~3× 缩减、设备拷贝量减半（clone 侧）、逐行归约
  L×N→1（32/64/64/128 → 1）；
- **P1-7 直接受益**：小几何 (256,8) 是 H2 判定 inconclusive 的配置
  （AR017 Part 24 遗留"2σ 展布检验功效问题"）——发布占比 60-71% 意味着
  压缩后该几何加速比有实质性抬升空间（方向性预期，不预设数字，基准实测）；
- **无语义变化**：发布到 cache 的内容逐位不变（同样的值经不同的拷贝路径），
  profiler 记录的 `_RawLayerStat` 内容逐位不变（同样的 stable_sum/num_tokens
  经预求和入口）——两项机制都是纯实现优化，不引入新开关。

### 1.3 明确不在范围

- **惰性/延迟发布**（验收判定后再发布被接受分支）——改变 cache 时序语义，
  发布完备性契约（UT-1502）与 profiler 记账时点被触碰，风险/收益另评估；
- **cache 协议批量 put API**（一次调用写 N 分支）——put 的逐分支拷贝保留
  （每分支独立存储是 cache 协议事实），Python 侧开销在快照发布后已非主导；
- **prefill N× 读放大消除**（[1,T,H] parent_static + expand 进 capture 体）
  ——触碰 #37 capture 语义（stride-0 广播进 graph），且实测占比仅 2.3-3.9%；
- **读回异步化**（pinned+event 重叠）——实测 0.06-0.13ms，占比 <1%；
- 串行路径 `_publish_graph_child` 与 eager 折叠路径的 record 调用——
  P1-7 验证负载在批量侧（AR016/017 先例保持串行零干扰）；
- `revalidate_host` 的 L+1 次 fetch（0.11-0.19ms，占比 0.3-1.4%）。

## 2. 术语

| 术语 | 定义 |
|------|------|
| 发布（publish） | 重放预算通过后，把 child_buf 行切片（ffn_out，layer 0 另含 embedding 行）写入 cache、并逐 (layer,row) 记录 profiler 稳定度统计的步骤 |
| 批量稳定度归约 | 单次 `mask_buf.sum(dim=2)` 产出 [L,N] 逐 (layer,row) 稳定计数，替代 L×N 次逐行 `stable_mask.sum()` 的设备归约 |
| 快照发布 | 单次整块 `child_buf.clone()`（每步 fresh 分配，发布后不再被本侧写）+ 行视图交给 `cache.put`；替代 L×N 次逐行 clone。防别名依据：快照是 fresh 私有张量，即便 by-reference cache 存视图也不与静态缓冲或后续重放共享存储 |
| 预求和记录入口 | `StabilityProfiler` 新增接受 0-dim `stable_sum` 张量 + 显式 `num_tokens` 的记录方法；`_RawLayerStat`/`_raw`/`_history`/`get_profile` 语义与既有 `record` 产出逐位一致 |

## 3. 功能需求

### 3.1 FR1 批量稳定度归约（profiler 记录路径）

**描述：** `StabilityProfiler` 新增预求和记录入口（如 `record_precomputed(
branch_id, parent_branch_id, layer_idx, step_idx, stable_sum, num_tokens,
tau, metric)`）：跳过内部 `stable_mask.sum()`，直接以调用方提供的 0-dim
设备张量为 `stable_sum` 构造 `_RawLayerStat` 并入 `_raw`/`_history`——
**与 `record` 对同一数据的产出逐位一致**（含 `_max_history_len` 裁剪、
`enabled=False` 早退）；既有 `record` 原入口**零改动**（eager/串行路径
调用方不受影响）。`_publish_graph_children` 改为：以
`profiler.enabled` 门控执行单次 `mask_buf.sum(dim=2)` → [L,N]
（**enabled=False 时零归约 launch**，1 次归约 launch），双循环内以
`sums[layer_idx][row]`（0-dim 视图，零 launch）+ `num_tokens=T` 调用
预求和入口。

**验收标准：**
- 每步发布路径的设备归约 launch 数从 L×N 降为 **恰 1**（UT 以归约计数
  spy 钉住——monkeypatch `Tensor.sum` 计数或等价探针）；
- `debug_enabled=True` 时的 `divergence_positions` 行为：预求和入口**不
  支持**（无 stable_mask 可 nonzero）——显式拒绝（`ValueError` 或文档化
  不支持 + 调用侧回退 `record` 原入口）；批量发布路径 `debug_enabled`
  下的行为**与现状逐位一致**（回退逐行 record，正确性优先于性能）；
- `_RawLayerStat` 字段内容（stable_sum 值/num_tokens/tau/metric/键序）
  与逐行 record 产出逐位一致（UT 对拍同一 replay 的两种记录路径）；
- profiler 聚合面不变：`get_profile`/`all_profiles`/`get_mean_stable_ratio`
  /`mean_stable_ratio` 对同一记录序列产出一致（UT 对拍）；
- 发布路径**零新增 host 同步**（`.item()`/`.tolist()`/`.cpu()` 禁入；
  #39 同源纪律——0-dim 视图构造是宿主侧元数据操作）。

### 3.2 FR2 快照发布（cache 写入路径）

**描述：** `_publish_graph_children` 以单次整块 `runner.child_buf.clone()`
（[L,N,T,H] fresh 快照，每步新分配）替代 L×N 次逐行
`child_buf[l][r:r+1].clone()`；双循环内 `cache.put` 消费快照行视图
`snap[l][r:r+1]`（[1,T,H]，put 内部 `copy_` 自带防别名——快照本身 fresh
且发布后本侧不再写，即便 by-reference cache 存视图也无静态缓冲别名）。
embedding 行发布维持现状（`embed_module(tokens)` 每步 fresh，行视图本就
安全）。

**验收标准：**
- 发布内容逐位一致：快照发布后 cache 内各 (branch,layer) 的 ffn_out/
  embedding 与逐行 clone 实现**bit-exact**（同一 replay 两种路径对拍）；
- 防别名契约（req 门控预期重点）：发布后 mutate `runner.child_buf`
  →cache 内容不变（UT-1509 既有断言继续钉住）+ **by-reference cache
  mock**（put 存引用不拷贝的最小实现）下，发布后 mutate child_buf 同样
  不影响其存储——快照 fresh 性是新增的正确性论证面；
- 每步发布路径的 clone 调用数从 L×N 降为 **恰 1**（UT 以 clone 计数
  spy 钉住）；
- `cache.put` 调用次数/参数形状不变（[1,T,H] 行、layer 0 双键——cache
  协议零变化，UT-1502 发布完备性继续钉住）；
- 快照张量生命周期：发布函数返回后快照无本侧保留引用（交 GC；by-reference
  cache 持视图则其生命周期由 cache 管理——不与静态缓冲共享存储的论证
  独立于 GC 时序）。

### 3.3 FR3 对拍与观测钉（UT-1701~，`tests/test_publish_cost.py`）

**验收标准（CUDA 用例 skipif 保护；容差沿先例 allclose(1e-4, rtol=1e-4)，
逐位场景用 bit-exact）：**
- **UT-1701 批量归约等价**：同一多步重放序列下，预求和入口记录的
  `_raw`/`_history` 与逐行 `record` 的产出逐位一致（stable_sum 值、
  num_tokens、分支归属、history 裁剪）；`get_profile().to_dict()` 与
  `mean_stable_ratio()` 一致；
- **UT-1702 归约计数钉**：发布路径执行一步后 `mask_buf.sum` 类设备归约
  恰 1 次（spy 计数；`enabled=False` 时恰 0 次）；同路径**零新增 host
  同步**钉住（`.item()`/`.tolist()`/`.cpu()` spy 恰 0 次调用）；
- **UT-1703 快照发布逐位一致**：同一重放（同一静态缓冲状态）先后经
  快照发布与逐行 clone 发布写入两个独立 cache，逐 (branch,layer) 键
  bit-exact 对拍（含 layer 0 embedding 行）；
- **UT-1704 clone 计数钉**：发布路径执行一步后 `Tensor.clone` 恰 1 次
  （child_buf 整块）+ logits.clone 不受影响（分路径断言或等价探针）；
- **UT-1705 by-reference 防别名**：最小 by-reference cache mock（put 存
  引用）下发布，随后 mutate `child_buf` →mock 存储不变；同场景 mutate
  静态缓冲后再次发布 → 新内容正确写入（快照每步 fresh）；
- **UT-1706 既有契约回归**：UT-1502 发布完备性 / UT-1509 发布隔离 /
  AR017 UT-1602 收缩端到端（收缩后输出 vs eager 对拍，allclose+argmax）/
  UT-1606 默认配置端到端——全部零改动全绿（发布路径行为兼容的直接
  证据）；
- **UT-1707 debug_enabled 回退**：`GLOBAL_STABILITY_PROFILER.debug_enabled
  =True` 下批量发布仍产出 divergence_positions（回退逐行 record），
  关闭后恢复预求和路径（spy 验证路径切换）；
- **UT-1708 eager/串行零干扰**：eager 折叠路径与串行 graph 路径的
  record/clone 行为不受本 AR 影响（既有 UT 全绿 + record 原入口签名
  零改动断言）；
- **UT-1709 mutation spot-check（纪律沿用）**：①删批量归约（回退逐行
  sum）→UT-1702 红；②删整块快照（回退逐行 clone）→UT-1704 红；
  ③快照非 fresh（复用/别名 child_buf 存储）→UT-1705 红（by-reference
  mock 存储被 mutate 污染）；④删 debug 回退→UT-1707 红；⑤sums 张量
  跨步复用后被覆写→UT-1701 红（stable_sum 逐位对拍捕获）。全验证后
  SHA256 逐字节还原。

### 3.4 FR4 墙钟基准与工件（`scripts/ar018_publish_bench.py`）

**描述：** 独立基准脚本（沿 ar017 先例，复用其夹具结构），三路径 × 四几何
× 相位分解入工件，产出 `results/optimization/ar018_publish_bench.json`。

**验收标准：**
- **三路径**：(a) baseline eager；(b) 批量 graph 固定容量 ratio=0.5；
  (c) 批量 graph 自适应容量（AR017 机制，稳态后计时）；
- **几何**：(H=256,L=8) 与 (H=1024,L=16) × N∈{4,8} × T=512（与 AR017
  工件同配置矩阵——**跨工件对照的前提**）；
- **相位分解随行**：路径 (c) 稳态每步的相位 p50（revalidate_host/
  prefill/replay/readback/publish/clone 六相）单列工件（探针方法固化：
  sync 相位边界；`phase_sum_ms` 与 `full_step_p50_ms` 实测值同时入工件，
  偏差方向注记——sync 串行化正向偏差与相位间宿主工作遗漏的负向偏差
  并存，不预设方向）；
- **配对交错 + 功效**：每重复内三路径 round-robin；warmup=5 覆盖
  capture+适配期；**reps=30**（AR017 已证 n=10 在 (256,8,4) 欠功效，
  沿用高功效单次预承诺运行）；TimingStats mean/std/p50/n；
- **假设与结果分离**（docstring 声明，JSON 只存实测与判定）：
  - H1：AR018 自适应步 p50 < AR017 工件同配置 adaptive p50（逐配置；
    跨工件对照——同机/同种子/同方法/同配置矩阵的先例对照，2σ 判据
    沿 `_verdict`，工件内明记"跨工件对照"性质）；
  - H2：AR018 自适应 vs baseline 加速比 ≥ AR017 工件同配置加速比
    （P1-7 第四档：发布压缩后综合框架对 baseline 的优势不回退且
    小几何方向抬升）；
  - H3：发布相位 p50 占比 < 本 AR 探针（`ar018_hostcost_probe`，AR017
    夹具/稳态容量前提）同配置占比（机制直接验证：发布占比
    60.7%/71.2%/29.7%/31.5% 显著下降）；
  - H4：三路径 logits 一致（argmax 精确 + folded 两路径 allclose
    1e-4）+ profiler 统计一致（stable_ratio 与逐行路径同值）；
- **环境与限制**（#39 纪律）：GPU/日期/驱动版本/CUPTI 探测（预期
  null+注记）/交错声明/复现命令（含 --reps 30）/种子；**峰值显存观测**
  字段（`torch.cuda.max_memory_allocated`，快照瞬态峰值的量化证据，
  承 §6 风险表承诺）；
- **运行预算**：全量 < 10 分钟；`--quick` 单配置子集。

### 3.5 FR5 文档收口

**验收标准：**
- 指南：P1-7 行（:180）追注第四档「发布成本 ✅ AR018」+ 新增
  **第二十五部分**（AR 记录 + 相位分解表 + 跨工件对照表 + 发布路径
  语义要点：批量归约/快照发布/防别名论证/debug 回退）；
- CHANGELOG AR018 节置 Unreleased 最前；
- AGENTS 扩注（发布路径契约：快照 fresh 性、put 自带拷贝下 clone
  冗余的消除、预求和入口与 record 的双入口并存、debug_enabled 回退）；
- AR017 Part 24 遗留清单收口注记（宿主侧分解项 ✅ AR018——分解
  已做（探针）+ 主项已压（发布）；prefill/读回实测 <5% 记录在案）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 质量 | 静态检查 | mypy clean（66 文件基线）/ pyflakes 零 / 100 列 |
| 质量 | 全量回归 | 846 passed 基线 + 新增零失败；既有测试零改动（UT-1706 引用既有 UT 零改动全绿为验收面） |
| 稳定 | demo 锚点 | 三指标精确一致（demo 不经批量 graph 路径） |
| 稳定 | 测试确定性 | CUDA 对拍仅 allclose+argmax/bit-exact；无墙钟断言进 pytest；UT 运行时 <30s |
| 兼容 | 默认行为 | **无新开关**：发布路径行为语义不变，全部配置下生效（与 AR016/17 的 opt-in 模式不同——本 AR 是纯实现优化，发布内容逐位一致由 UT-1703/1705 钉住） |
| 兼容 | 公开 API | `record` 原入口签名零改动；`cache.put` 协议零变化；`cuda_graph.py` **零改动**（静态缓冲与 capture 体不触碰） |
| 性能 | 同步纪律 | 发布路径零新增 host 同步；重放 D2H 恰一次不变 |
| 性能 | 基准诚实性 | 跨工件对照性质显式声明；相位分解的串行化偏差如实注记；假设判定基于实测 |

## 5. 约束与假设

**约束：**
- 生产改动限：`actfold/profiler/stability_profiler.py`（预求和入口）、
  `actfold/models/architecture_utils.py`（`_publish_graph_children` 重构）、
  新基准脚本 + 文档；**`cuda_graph.py` 零改动**（git status 证明，沿
  AR016 先例）；
- `record` 原入口、eager/串行路径调用方、cache 协议、#37/#41 契约、
  AR017 双钩子次序全部不变；
- 工件数字可复现（种子/命令/环境字段齐全）；文档数字从工件逐值复算
  （AR016 review 教训固化）。

**假设：**
- 快照整块 clone 的单次大拷贝（[L,N,T,H]）在 WDDM 下快于 L×N 次小
  clone+中间拷贝（探针已证 launch 开销主导：(256,8,4) 无 profiler
  发布 2.72ms 中 32 对拷贝 ~85µs/对，launch 占主导）；
- 预求和 0-dim 视图在 `_RawLayerStat.stable_sum` 消费面（
  `_ratios` 的 `torch.stack`、`get_profile` 的 `.float().item()`）与
  0-dim 张量语义一致（stack 兼容；视图指向每步 fresh 的 [L,N] 归约
  输出，无生命周期别名）；
- 跨工件对照（AR018 vs AR017 工件）有效性：同机/同驱动/同种子/同
  方法/同配置矩阵，且 AR017 工件的复现命令在当前代码上除本 AR 改动
  外逐位可重放（同仓单变量变更）；
- `debug_enabled` 默认 False 且生产/benchmark 路径不开启（现状事实，
  UT-1707 钉回退行为）。

## 6. 风险

| 风险 | 缓解 |
|------|------|
| by-reference cache（第三方实现）存快照视图导致跨步别名（快照被 GC 后悬挂或被复用） | 快照每步 fresh 分配、发布后本侧零保留；UT-1705 以 by-reference mock 钉住 mutate 隔离；文档明记快照生命周期契约 |
| 瞬态显存峰值上升（快照瞬间 child_buf+快照并存，(1024,16,8) ≈+268MB；by-reference cache 视图钉住整快照存储使释放粒度变粗——总量与今日逐行 clone 持平） | torch caching allocator 对固定几何固定尺寸的 freed block 即时复用（稳态代价≈常驻一个额外块，碎片风险低）；默认参数下的拷贝型 cache（ActivationCache；ChunkedActivationCache 在默认 chunk_size<T 切片非连续时亦拷贝）不受钉住影响；16GB 显存下 (1024,16,8) 基准峰值仍远低于上限；基准工件记录峰值显存观测字段（FR4） |
| 预求和入口漏记/错记（batch/layer/row 归属错位） | UT-1701 与逐行 record 逐位对拍（同一重放序列双路径记录、`_raw` 全字段比较）；归属错位必然造成 stable_sum 值错位被对拍捕获；bool sum 为 int64 精确累加，整块 dim=2 归约与逐行 sum 数学上逐位相等 |
| `debug_enabled` 路径行为漂移（divergence_positions 丢失） | 预求和入口显式不支持 + 调用侧回退 record；UT-1707 钉住回退与恢复 |
| 跨工件对照被质疑（非同批运行） | 工件明记对照性质 + 全部前提字段（种子/命令/环境）；H1/H2 判定 2σ 判据照用；AR017 工件复现命令保留可重放 |
| 相位分解 sync 边界的双向偏差（串行化正向 vs 相位间宿主遗漏负向，实测以后者为主） | `phase_sum_ms` 与 `full_step_p50_ms` 实测值同入工件、不预设方向；H3 判定用占比而非绝对值 |
| WDDM 墙钟噪声（沿 AR017 教训） | reps=30 + round-robin + p50/std + 2σ 判据；inconclusive 诚实记录 |
