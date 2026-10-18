# [AR026] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR026 |
| AR 主题 | 真实 checkpoint 端到端图路径（生产 FoldedGraphRunner 捕获在真实 remote-code 模型上的启用/修复 + 图上四模式实机基准） |
| 关联 SR | 指南第二十五部分遗留项「真实 checkpoint 端到端」；README 路线图条目 1/7（AR018 合成证据桥接 + 小几何 WDDM 残余） |
| 日期 | 2026-10-17 |
| 状态 | Draft |

## 1. 背景与目标

AR017/AR018/AR020 的图路径（自适应容量 + 批量稳定度归约 + `put_batch` 发布）在合成
LLaMA-layout 模型上交付 **2.40–2.95×** p50 加速与 **L·N+2** launch 账本；但该谱系的
全部证据均为合成模型。真实 checkpoint 侧：

- AR023 家族探针（`ar023_family_probe.json`）结论均为 `graph_enabled=False`，其
  graph face 是**裸模型 ad-hoc 捕获代理**（非生产 `FoldedGraphRunner` 路径），两个
  真实失败签名为：
  - **llada**：`RuntimeError: captures_underway.empty() INTERNAL ASSERT FAILED
    (CUDACachingAllocator.cpp:2967)`——裸 remote 前向在本栈（Windows WDDM +
    torch cu121 + bf16-8B）上捕获不安全；
  - **dream**：`The size of tensor a (64) must match the size of tensor b (4) at
    non-singleton dimension 2`——[4,64] 批量下裸前向内部广播断裂（缺
    position_ids/mask 形状面，与 AR023 sampler 路径已验证的批量前向行为不一致，
    疑探针构造面而非生产面缺陷）。
- AGENTS #45 记载 Caveat：「图静态体（`FoldedGraphRunner` 自身层循环）不解包
  tuple——未演练；tuple-layer 图捕获前需解包」。**本 AR 需求期代码审计发现该表述
  与现行代码矛盾**：`_recompute_all`（folded_transformer.py:1384-1391）对 tuple
  返回取首元素解包，而静态体（cuda_graph.py:480）正是调用
  `layer._recompute_all(...)`——解包实际存在且被静态体继承。该 Caveat 需重审并
  修正（含必要的 UT 钉固：tuple-returning 层的图捕获等价性）。

**目标**：把生产图机制（AR021 形状键 runner 池 + AR017 自适应容量 + AR018 发布
路径）在真实 remote-code checkpoint（LLaDA-8B 优先，Dream-7B 跟随）上启用/修复，
并以图上四模式实机基准（官方 baseline / eager 折叠（AR023 面）/ graph 固定容量 /
graph 自适应）把 AR018 谱系的加速证据桥接到真实 checkpoint——框架级优越性阶梯
（AGENTS「Experimental Validation Rule」第 6 条）的真实端到端档。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- 生产捕获路径在真实 LLaDA-8B 上的根因分析：裸前向 `captures_underway` 断言的
  触发链（WDDM/分配器/remote sdpa 交互）；`FoldedGraphRunner` 批量捕获与裸捕获
  的算子序列差异审计；修复或带类型化 capture-failed 层的规避（禁止原生崩溃穿越
  生产边界）。
- Dream [N,T] 捕获面诊断：区分探针构造缺陷（未线程 position_ids——AR023 sampler
  路径已批量验证）与生产缺陷；生产 Dream 批量捕获验证或精确 blocked 记录。
- AGENTS #45 tuple-unwrap Caveat 重审与修正；tuple-returning 层图捕获 UT（CUDA
  门，CPU skip）。
- 探针合同升级：`ar023_family_probe` graph face 从裸模型代理改为**生产路径捕获**
  （MFF + FoldedGraphRunner），`graph_enabled` 语义对齐生产现实。
- 图上四模式实机基准脚本（导入式复用 AR023 助手先例）+ reps=30 预承诺单次 +
  VRAM 余量面（bf16-8B 14–16GB @16GB 卡：图池 VRAM 成本是绑定约束）。
- README 全量重写 + AGENTS #45 修正 + GUIDE 新部分 + CHANGELOG（README
  Full-Update Rule；优越性阶梯表新档）。

**Out of Scope（本 AR 不做的）：**

- 投机提交粒度/质量机制（AR024 域）；多 parent 批量折叠（AR025 域）。
- Fast-dLLM 图路径（AR019-AR022 已交付且真实 checkpoint 已验证）。
- 跨家族探针重跑中的下载/加载/dtype 面变更（仅 graph face 合同变化）。
- CUDA graph 捕获的跨进程/多 GPU 语义。

## 3. 功能需求

### 3.1 FR1 生产图捕获根因与修复（LLaDA-8B）

**描述：** 生产 `FoldedGraphRunner` 批量捕获在真实 LLaDA-8B remote 层栈上启用；
裸前向的分配器断言根因定位并修复或隔离。

**触发条件：** MFF `use_cuda_graph=True` 构造 + LLaDA sampler 批量投机路径首窗
捕获。

**期望行为：** 捕获成功并重放正确（与 eager 批量前向 allclose 容差面）；或落入
既有类型化 capture-failed 层（warn-once + eager 降级），进程永不因原生 INTERNAL
ASSERT 崩溃。

**异常处理：** 捕获失败走 AR002 既有降级矩阵（capture_failed 旗标 + eager 回
退）；不可恢复错误 raise 带操作指引的 RuntimeError。

**验收标准：**
- Given 真实 LLaDA-8B bf16 + 生产 MFF 图构造， When 批量投机采样运行，
  Then 捕获/重放路径无原生崩溃且与 eager 输出 allclose（或类型化降级且基准记录
  降级面）。
- Given 裸前向断言复现脚本, When 根因定位, Then 修复或规避记录入 design/进度
  （含 PyTorch 上游 issue 引用如适用）。

### 3.2 FR2 Dream 批量捕获面诊断

**描述：** Dream-7B [N,T] 捕获的形状断裂定位（探针构造 vs 生产缺陷）与生产面
验证。

**触发条件：** Dream sampler 批量投机 + 图构造。

**期望行为：** 生产路径批量捕获验证通过；或精确记录 blocked 原因（零伪造）。

**验收标准：**
- Given AR023 Dream sampler 批量路径已验证 eager [N,T], When 生产图捕获运行,
  Then 捕获成功（position_ids/pe 静态烘焙正确）或 blocked 工件携带最小复现签名。

### 3.3 FR3 tuple-unwrap Caveat 重审（AGENTS #45 修正）

**描述：** 图静态体解包状态代码级重审（需求期已完成初判：解包存在于
`_recompute_all` 且被静态体继承）；AGENTS #45 文本修正；tuple 层图捕获等价
UT。

**验收标准：**
- Given tuple-returning mini 层栈（CPU 可构造）, When 图捕获+重放（CUDA 门；
  CPU 上以静态体直接调用等价 UT 钉固）, Then 解包语义与 eager 循环逐位一致。
- AGENTS #45 Caveat 文本按审计结论修正（含本 srs 的矛盾发现溯源）。

### 3.4 FR4 探针 graph face 生产化

**描述：** `ar023_family_probe` graph face 改为生产路径捕获（MFF+runner），反映
生产 `graph_enabled` 语义。

**验收标准：**
- Given 升级后的探针, When 真实 checkpoint 探针运行, Then graph face 记录生产
  捕获 ok/VRAM 余量/失败签名（与 FR1/FR2 结论一致）。

### 3.5 FR5 图上四模式实机基准

**描述：** LLaDA-8B 四模式（baseline 官方配方 / folded-batched eager（AR023 锚）/
folded-batched graph 固定 C / graph 自适应）reps=30 预承诺单次；H 面全承
AR023/AR018 判定式（质量双指标 + p50 + 账本/canary + VRAM 余量）。

**验收标准：**
- Given 预注册判定式冻结于脚本, When 单次 reps=30 运行, Then 工件全要素可溯源
  （`results/optimization/ar026_real_graph_bench.json`），框架优越性阶梯表新增
  真实端到端档（quality gap + p50 speedup vs 官方 baseline）。
- 诚实负结果义务：inconclusive/blocked 原样交付（AR021 H2 先例）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 图 vs eager p50 | 预注册 H 面（不预设方向——WDDM 小几何残余是已知风险） |
| 资源 | 图池 VRAM | bf16-8B @16GB：余量观测入工件；探针 >1024MB 阈值语义保留 |
| 可靠性 | 原生崩溃 | 生产边界零 INTERNAL ASSERT 穿越（类型化降级） |
| 兼容 | 默认关 | `use_cuda_graph` 默认 False 字节不变（AR002 合同） |

## 5. 约束与假设

**约束：**
- 实机面排队：AR023 T006 完成后 GPU 空闲窗口按 AR024 T005/T006 → AR025 实机面
  → 本 AR 顺序（串行归档约束：AR024/AR025 先归档）。
- Windows WDDM + torch 2.5.1+cu121 + 16GB Quadro RTX 5000（AR019 venv，
  transformers==4.53.1）——捕获行为平台特定，结论注记平台。
- #37 图捕获合同全承（同形窗口 / mask 对象 pin / 预算校验 / AR017 自适应 / AR021
  池）。

**假设：**
- 裸前向断言与生产捕获算子序列差异可定位（静态体走 fused gate + merge +
  `_recompute_all`，非裸 remote 前向）。
- Dream 形状断裂为探针构造面（sampler 已线程 position_ids/pe——AR023 批量 eager
  已验证）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 生产捕获 | `ManualFoldedForward(use_cuda_graph=True)` → `FoldedGraphRunner` 静态体捕获（非裸模型 ad-hoc 捕获） |
| 裸前向断言 | 探针 graph face 记录的 `captures_underway.empty()` INTERNAL ASSERT |
| 框架优越性阶梯 | AGENTS「Experimental Validation Rule」第 6 条的累积对照表（每档 vs 官方 baseline 双指标） |

## 7. 任务清单

见同目录 `tasks.md`（T001-T011；GPU 串行队列 + 串行归档约束载于其「排队约束」）。
