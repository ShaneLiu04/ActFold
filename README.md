<div align="center">

# ActFold

**面向 Diffusion LLM 投机解码的跨分支激活复用（Branch Folding）研究框架**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![CI](https://github.com/ShaneLiu04/ActFold/actions/workflows/ci.yml/badge.svg)

[项目概述](#项目概述) •
[核心创新](#核心创新) •
[最新实验结果](#最新实验结果) •
[安装](#安装) •
[快速上手](#快速上手) •
[进阶用法](#进阶用法) •
[文档](#文档) •
[引用](#引用)

</div>

<p align="center">
  <img src="docs/assets/cover-v7.png" alt="ActFold — 面向 Diffusion LLM 投机解码的跨分支激活复用" width="100%">
</p>

---

## 项目概述

ActFold 是一个降低 Diffusion LLM 投机解码**验证阶段算力**的研究框架。

在投机解码中，多个子分支（candidate branches）从同一条父序列出发被独立验证。
尽管子分支通常只与父序列相差少数几个 token，标准实现仍然对每个子分支触发
**全量前向重计算**——这是验证阶段的系统性冗余。

ActFold 用 **Branch Folding（分支折叠）** 攻击这一冗余：在每一层，每个 token
被二分为 **stable**（复用父分支激活）或 **divergent**（重计算），两组结果合并
为单次输出。判定逐层、逐 token 独立进行，即使分支在许多位置不同，只要隐状态
一致的位置就能获益。

> **目标影响**：验证阶段 TFLOPs 削减 21%–62%，精度损失极小。
>
> **最新交付（AR022）**：Fast-dLLM-v2-1.5B 真实生成回路，贪心锚定批量投机
> **7.02×** 于 baseline，且质量面与 baseline **逐位对齐**（post-hoc 置信
> gap = 0.0、解码长度/早停完全恢复）。

### 核心思想

对每层 `l`、每个 token 位置 `t`、每个扩散步 `s`：

```
sim(l, t, s) = cosine_similarity(h_parent[l, t, s], h_child[l, t, s])
```

- **Stable tokens**（`sim > τ`）：直接复用缓存的父分支 FFN 输出。
- **Divergent tokens**（`sim ≤ τ`）：在完整子分支隐状态上运行该层，
  保持 self-attention 上下文不被破坏。

相似度门控本身是**语义安全网**：任何被误判为 stable 的位置都会被门控拦下
重算，这为后续所有激进优化（前缀复用、固定容量 CUDA Graph、批量发布）提供了
正确性底线。

---

## 核心创新

ActFold 的创新遵循一条统一的工程方法论：**先测量、再定位瓶颈、后设计机制**——
每一项机制都以本机实测证据为选题依据，以零语义变化（输出逐位一致）为设计约束，
以 mutation 测试 + 跨工件基准为验收面。

### 创新总览

| # | 创新点 | 动机（实测证据） | 机理 | 结果 |
|---|--------|------------------|------|------|
| 1 | **Branch Folding 逐层逐 token 门控复用** | 投机验证中子分支仅差少量 token，却整分支全量重算 | 每层逐 token cosine 门控 stable/divergent 二分 + 合并 | demo 85.5% FLOPs 削减，MSE 2.35e-03 |
| 2 | **向量化激活缓存** | 旧 cache 单层读取 0.65–0.70ms **超过**层重算本身（0.53–0.61ms）——折叠零加速的根因 | 连续缓冲 + 批 `index_select` 替代逐 token dict | put/get 快 12–183× / 30–429×；全稳定路径 2.3–2.7× **快于重算** |
| 3 | **前缀折叠 + 截断前缀复用**（AR003/AR007） | 变长子分支原本直接报错；父比子长原本无法复用 | 门控前缀对齐（因果模型前缀逐位相同→全 stable）；截断复用以门控为安全网 | `folded_generate` 真正跨步折叠（stable≈0.84，token 与全重算逐位一致） |
| 4 | **真接受语义**（AR004/AR006） | 历史 logit 均值是无意义分数，不构成论文级指标 | 同位置约定 + 独立 draft 模型分布的接受率计量（`model_draft`） | 论文级投机解码接受率 + EMA 判定 |
| 5 | **批量分支前向**（AR015） | N 个子分支串行验证 = N·L 次 layer 调用，launch 开销主导 | `branch_ids` 复数 kwarg 贯穿三层（层/入口/生成循环），同父等长候选**单次批量前向** | 批量 eager vs 串行 **2.0–5.6×**（6/6 配置） |
| 6 | **批量 CUDA Graph 捕获**（AR002/AR016） | 热路径由 launch 与宿主侧开销主导，eager 无法再压 | 固定发散容量 padded 消除每层数据依赖同步（graph 前置条件）+ 独立 `[N,T]` runner 捕获 + **重放前父缓存再校验**（防同形覆写静默错值） | 折叠子步 **−47.1%**（AR002）；graph vs 裸基线 1.37–1.77× 首次墙钟证据（AR016） |
| 7 | **自适应发散容量重捕获**（AR017） | 固定容量 C=0.5 vs 实测发散需求 D_frac 仅 13–16% → **FFN 重算过配 3.0–3.8×**；反向低容量则每突发一次违例即丢失 graph 收益 | 观测驱动状态机：违例→增长重捕获、连续耐心观测→收缩重捕获、节流 + 上限退役保证稳定；发散计数由既有预算读回**零额外同步暂存** | 深几何 adaptive vs baseline **2.30×/2.46×**（2σ 阈值的 9–12× 裕量，decisively supported）；容量较固定值缩减 3.01–3.02× |
| 8 | **发布路径宿主成本压缩**（AR018） | 六相探针实测：发布相位占自适应全步 **60.7%/71.2%/29.7%/31.5%**（最大单项），其中逐行 clone 是纯冗余（put 自带拷贝） | 单次整块 `child_buf.clone()` fresh 快照（by-reference cache 安全的防别名论证）+ 单次 `mask_buf.sum(dim=2)` 批量归约经 `record_precomputed` 入账；launch 账本 **3·L·N → L·N+2** | 发布占比降至 **45.3%/50.9%/15.7%/17.6%**；最终加速比 **2.40/2.67/2.66/2.95×** |
| 9 | **position_embeddings 线程化 + 真实 checkpoint 端到端**（AR019） | 前三代墙钟证据全部来自合成 LLaMA-layout 模型；探针实测确证 remote 层必须收 rotary——`ManualFoldedForward` 不透传 `position_embeddings` 则 Fast-dLLM parent prefill 即 `TypeError` | `forward` 末位 `position_embeddings=(cos, sin)` 新参线程化**四构建面**（串行/批量 eager、graph 捕获静态体 by-reference 烘焙、AR017 自适应重捕获），镜像 attention_mask 契约（入口 ValueError 校验 + 重放期 `is` 对象恒一静默回退）；层接口零改动（`**kwargs` 通道）；`None` 不附加 kwarg，既有调用逐字节不变 | Fast-dLLM-v2-1.5B 真实块验证三路径 **2.44–2.62×**（p50，n=30）+ 逐位一致（maxdiff=0.0、argmax 一致率 1.0）+ 完美块二分（stable ratio 精确 = K_PROMPT/T） |
| 10 | **发布路径 cache put 批量化**（AR020） | AR018 压缩后小几何发布相位仍占自适应全步 45.3–50.9%，剩余主体 = L·N+N 次逐条 `cache.put` 的宿主 launch 开销（WDDM 实测 ~30–40µs/次）；foreach 探针实测 L·N=224 条 bf16 宿主入队 **27.8×** | `BatchedPutCapable` 结构化协议 + 三 cache `put_batch`（**冲刷语义**：写入序恒=条目序，消除对 foreach 组内顺序的依赖）+ `_foreach_copy_` 单点封装（(device,dtype) 分组，旧 torch 逐对降级）+ 发布路径**每次发布能力探测分流**（无能力 cache 逐字节走原循环）+ `mff.cache` property 传播 | 真实 checkpoint 发布窗口 p50 **1.77–1.97×**（−43.4%~−49.2%）；主机调用账本 **116/232/116 → 1**；正确性门全 bit-equal；端到端 adaptive(batch) vs baseline **2.40–2.67×**；H1–H4 全 supported |
| 11 | **生产生成回路折叠接线**（AR021） | AR015–020 批量验证栈**零生产调用方**（sampler 不传 pe 则真实 remote 必崩、掩码对象不钉则图捕获一次后永不重放、`[N,T]` 批量验证形态在生产回路不存在）；AR019 P5 实证 tau=0.95 gate 会把同块未翻位置误判 stable（parity 92.9% 先例）——链折叠直接上 gate 不精确 | sampler **(T,prompt_len) 键控折叠供给**（块因果∧pad 排除 `[T,T]` 掩码 + rotary pe + forced 三件套对象钉定，跨 sample 存续）+ **强制散度活跃块**（当前块整区不进 gate 判定面、全量重算——parity 由构造保证，P5 构造级绕开）+ **形状键控 runner 池**（opt-in `graph_runner_pool_size`，跨块增长每形状窗捕获/重放，LRU+cuBLAS workspace 回收）+ **批量投机块验证**（opt-in `num_block_candidates`：top-k 起草→单次 `[N,T]` 批量验证→min-prob 选定提交；AR016/017/018/020 批量面首次被生产回路行使） | 真实 generate() 三模式：**H1 supported（真实 checkpoint bit-exact）**；H2 inconclusive（Δ=−107.5ms 未越 2σ=265.3ms，小形状串行折叠开销>收益）；**H3 supported（batched vs serial 2.47× / vs baseline 2.38×，p50 1193.9 vs 2948.3/2840.7ms）**；H4 supported（池计数逐位对账、账本 c=4 界内、canary 真） |
| 12 | **贪心锚定候选：批量投机质量对齐**（AR022） | AR021 批量投机 2.38× 但**质量面与 baseline 分叉**：块级 exact-match 0.0、post-hoc 置信 0.3697→0.2897 / 0.5703→0.1064、解码长度 160/77 vs baseline 早停 34/66（非贪心提交不命中 stop token）——速度-质量权衡是 AR021 的如实遗留 | `candidate_include_greedy`（opt-in）：候选**行 0 = 贪心假设**（draft top-k 有序集首位，srs 冻结 argmax 定义；确定性构建、零设备回读、零 generator 消费）；N 总数不变、随机行 j=1..N-1 在冻结种子公式中保持行号；贪心行进入 **min-prob 公平竞争、可落败**（UT-2203 后验陷阱构造级钉死，选定零特判）；`greedy_win_stats` 只读观测面 + `ActFoldConfig` YAML 消费面；四模式基准 MFF_C 与 MFF_B 同构独立实例——速度变量构造性清零 | 四模式 H1-H4 **全 supported**：posthoc 0.369661/0.570341 逐位==baseline（gap=0.0）、长度 34/66 早停**完全恢复**、块级 exact 4/4=1.0 vs batched 0.0；p50 440.9ms——**7.0236× vs baseline / 2.9498× vs batched**；复现自检 strict PASS（batched 逐位==AR021 先验） |
| 13 | **多 parent 批量折叠 + 双链 lockstep 并行投机**（AR025） | AR015 批量契约限定**同父**：双链投机只能逐链各发一次 `[N,T]` 批量前向——写账本随链数线性放大（合成面实测 chains=2/4 时 40–160 ops vs 合并 4 ops），跨链候选永不共享一次前向；逐链交错路径还让缓存 LRU 行为不对称（DD8 erratum 实证） | ①机制面：`ManualFoldedForward.parent_branch_ids` 尾参——per-row 父装配（gather + 行级降级，OQ1）+ 逐行 gate/merge，`[sum(N_c), T]` 单次折叠前向服务多链候选；②采样面：`sample_joint` W2 lockstep（单实例双链、per-chain 画布/折叠链状态/链限定 cache tag，双实例禁令=branch id 碰撞）；`_speculative_commit` 按 forward 接缝拆分 `_speculative_build`/`_speculative_score_commit`，合并闭包重写单数→复数 parent 参数；**种子公式无链分量**（同 unit/sub/j per chain ⇒ A/B 逐位等价构造前提）；③基准面：`--mode real` AR023 模板接线 + DD8 冻结判定式（H1 逐位硬门+logits 归因回放、H2 严格 p50(B)/p50(A)>1.0、交付线 ≥10%、H4 账本严格小） | 合成面 **16/16 全过**（8 形状点×双构造：S1 supported——nonsplit 逐位/split allclose 1e-6；S2 supported——写账本 4 < 40–160 严格；S3 0.9–0.975 report-only）；UT 13 例（逐位相等+账本严格小+守卫面）+ 冻结面哨兵 96 绿；真实面 reps=30 接线交付、按 RERUN 步骤 16 待高速算力机执行回填 |

### 机理详解（按创新演进顺序）

#### 1. Branch Folding：基础机理

见上文"核心思想"。关键设计：**注意力永远在完整序列上运行**（divergent token
的重计算必须携带完整子分支上下文），只有 FFN 输出按 token 复用；`SplitFoldedTransformerLayer`
进一步将 FFN 拆为仅对 divergent 行计算（attention 全序列），在 B×T≥512 时
per-layer 加速 1.1–1.93× 且输出与全重算一致。

#### 2. 向量化激活缓存：修复"折叠反而更慢"

第一阶段诊断发现折叠路径比基线**更慢**的根因不在算法而在数据结构：逐 token
dict 缓存的读取成本（0.65–0.70ms/层）超过了层计算本身。`VectorizedActivationCache`
把激活存进连续缓冲、以批量 `index_select` 读写，使"全 stable 折叠前向"首次
**快于**全量重算（2.3–2.7×）。

#### 3. 前缀折叠：让跨步生成真正复用

生成循环中每步子序列比父序列长一个 token——形变负载原本无法折叠。AR003 引入
前缀对齐：门控只比较公共前缀（因果模型下逐位相同→全 stable），追加后缀恒
divergent。AR007 进一步允许**父比子长**：父激活截断到子长度，真前缀经门控
认证后复用，非前缀子分支在门控处发散、全量重算。

#### 4. 批量分支前向（AR015）：N 分支一次算完

同父、等长、parent-batch==1 的候选集合不再逐个串行验证：`branch_ids` 复数
关键字贯穿 `FoldedTransformerLayer` → `ManualFoldedForward` → `folded_generate`
三层，N 个分支拼成 batch 单次前向。其余形态（变长/多父）自动回退串行路径，
零行为变化。

#### 5. 批量 CUDA Graph（AR002/AR016）：消除 launch 与同步

CUDA Graph 要求捕获体无数据依赖同步。为此 AR002 先用**固定发散容量 padded
gather**替代 `nonzero`（消除每层一次的同步），再以 `FoldedGraphRunner` 捕获
固定形状子分支前向。AR016 把批量调用接进图路径：独立 `[N,T]` runner、
**每次重放前宿主侧再校验父缓存形状**（防止 ChunkedActivationCache 整块替换
场景下的同形静默覆写——这是正确性核心，不是优化）、逐分支行切片 clone 发布。

#### 6. 自适应发散容量（AR017）：容量跟随需求

图捕获把发散容量冻结在捕获时刻，而实测 D_frac 仅 13–16%——固定 C=0.5 意味着
FFN 重算量是需求的三倍以上。`AdaptiveCapacityPolicy` 状态机观测每步最大层
发散量（由既有预算读回顺带暂存，**零额外同步**）：持续过配→收缩重捕获、
违例→立即增长重捕获、increment-then-check 节流 + 成功次数上限退役保证稳定；
任意时刻至多双图，失败退化回旧 runner 永不劣于现状。增长钩子位于预算超限分支
内（违例即信号），收缩钩子位于发布之后（发布必须消费旧 runner 的静态缓冲）。

#### 7. 发布路径宿主成本压缩（AR018）：压掉最后一项宿主开销

容量右移后，六相探针（revalidate/prefill/replay/readback/publish/clone）
显示发布相位成为最大单项（小几何占 60.7–71.2%）。旧发布循环的 L×N 双循环每步
付出 3·L·N 次 launch：逐行 clone（put 自带拷贝，纯冗余）+ 逐行 sum 归约。
AR018 以两项正交机制压缩为 L·N+2：

- **快照发布**：单次整块 `child_buf.clone()` 产出私有 fresh 快照，行视图交给
  `cache.put`——快照私有可能导致 by-reference cache 存到静态缓冲视图被后续
  重放污染，fresh 快照从论证上排除该别名；
- **批量稳定度归约**：单次 `mask_buf.sum(dim=2)` 产出全部 (layer,row) 稳定
  计数，0-dim 视图经新入口 `record_precomputed` 入账（与 `record` 产出逐字段
  一致，下游聚合面零改动；`debug_enabled` 回退原入口保留 divergence_positions）。

发布内容逐位不变（对拍 UT 钉住），无新开关、无新语义——纯实现优化。

#### 8. position_embeddings 线程化（AR019）：真实 checkpoint 的最后一块拼图

P1-7 前四档把合成基准的 adaptive vs baseline 推进到 2.40–2.95×，但**全部证据
来自合成 LLaMA-layout 模型**——本机探针实测确证：Fast-dLLM v2 remote attention
第一行就解包 `cos, sin = position_embeddings`，`ManualFoldedForward` 不透传该
参数则 parent prefill 即 `TypeError`，综合框架此前**根本无法在真实 remote-code
checkpoint 上运行**。AR019 补上这块拼图：

- **四构建面线程化**：`ManualFoldedForward.forward` 末位新参
  `position_embeddings=(cos, sin)`（默认 `None`，向后兼容），贯穿串行 eager /
  批量 eager / graph 捕获静态体（`FoldedGraphRunner(position_embeddings_static=...)`
  by-reference 烘焙，镜像 mask 不 clone）/ AR017 自适应重捕获（新 runner 必收
  钉定对象，否则 shrink/growth 后静默错算）；
- **镜像 attention_mask 契约**：入口纯宿主元数据校验（二元组/张量/seq 维/
  cos-sin head_dim 一致，失配 `ValueError`）；重放期 `is` 对象恒一检查
  （换对象静默回退 eager，不告警不吞错）；成立前提明文化——position_ids 恒为
  `arange(T)` 的常量位置方案下 cos/sin 对固定形状图是常量；
- **层接口零改动**：`position_embeddings` 经层 `**kwargs` 通道与
  `_recompute_all` 的 accepted/varkw 过滤天然承载，仅透传给接受它的原始层，
  合成层自动忽略；`None` 不附加 kwarg，既有调用方**逐字节不变**（现状绿钉
  UT 钉住）。

以此为前提，Fast-dLLM-v2-1.5B 真实块验证工作负载（parent = K_PROMPT
prompt + 32×MASK(151665) 全遮块，children = top-10 采样填充；块因果掩码
`(q//32) ≥ (kv//32)`；pe = 模型自身 rotary_emb）给出三路径端到端墙钟：
**2.44–2.62×（p50）+ 逐位一致（maxdiff=0.0）+ 完美块二分（stable ratio
精确 = K_PROMPT/T，n=112/224/112 全等）**——P1-7 在真实 checkpoint 上的
收官证据（详见[最新实验结果](#最新实验结果)）。

#### 9. 发布路径 cache put 批量化（AR020）：压掉发布窗口的 launch 主体

AR018 把发布窗口压到 L·N+2 次 launch 后，小几何发布相位仍是自适应全步最大
单项（45.3–50.9%）——剩余主体是 L·N+N 次逐条 `cache.put` 的**宿主 launch
开销**（WDDM 实测每次 ~30–40µs，N=8 时 232 次 put ≈ 数 ms）。AR020 以
`torch._foreach_copy_`（单点封装于 `batched_write.copy_pairs`，按
(device,dtype) 分组）把整批 put 融合为每发布 **1 次**主机调用：

- **能力协议**：`BatchedPutCapable` 结构化 Protocol（`isinstance` 即方法
  存在性探测）；发布路径**每次发布重探**（零实例缓存——cache 可替换），
  无能力 cache 逐字节走原双循环（既有调用零变化）；
- **冲刷语义**（正确性骨架）：同一逻辑目的地的首现对进延迟列表，第二次
  出现先冲刷首现对再立即执行当前对，此后全立即——**写入序恒等于条目序**，
  精确复刻逐条路径（含重叠覆盖序），不依赖 foreach 组内顺序这一未文档化
  不变量；发布用例目的地恒唯一，冲刷零触发；
- **等价性**：三 cache × CPU/CUDA 随机化性质测试钉死（比较面 =
  `buffer[:最终 _counts]` + 计数 + 键序 + `num_entries`）；驱逐在与逐条
  put 完全相同的逐条修剪点触发（"每层一次终态修剪"被随机化测试实证否证
  ——驱逐+重建的名集分叉）；`validate_entries` 全量前置（非法批零变更，
  原子性）；
- **账本**：`write_op_count` 免 monkeypatch 宿主探针（AR017/18 谱系），
  put 侧 L·N+N → put_batch 侧 O(L+N)（实测 116/232/116 → 1）。

实测（Fast-dLLM-v2-1.5B 真实块验证，n=30）：发布窗口 p50
**1.77–1.97×**（−43.4%~−49.2%），端到端 adaptive(batch) vs baseline
**2.40–2.67×**，正确性门三配置 116/232/116 张量全 bit-equal，H1–H4
全 supported（预承诺单次运行，判定式冻结于工件 meta）。

#### 10. 生产生成回路折叠接线（AR021）：批量验证栈获得第一个生产调用方

AR015–020 把批量验证栈建成六档，但**该栈至今只有基准脚本在调用——生产
`generate()` 路径零调用方**：sampler 不传 pe（真实 remote checkpoint 上折叠
路径必崩）、每迭代新建掩码对象（图前置 `is` 钉定失配，捕获一次后永不重放，
图收益归零）、`[N,T]` 同父批量验证形态在生产回路不存在。AR021 把整栈接入
`FastDLLMSampler.sample` 的折叠分支，四个机制层：

- **sampler 折叠供给（FR1）**：`_fold_ctx_for` 以 **(T, prompt_len)** 为键
  （host 元数据，零 device readback——纯 shape 键会把同画布长、不同 pad 数
  的 prompt 撞进同条目导致 pad 列静默错误）构建并**对象钉定**三件套：
  块因果∧pad 排除 `[T,T]` 掩码（pad 区由 prompt 对齐填充的 host 元数据确定，
  正常输入下与 raw `x != pad_id` 逐位等价）、rotary pe（AR019 常量位置
  方案；**dtype 跟随 embedding weight**——fp32 dummy 对 bf16 remote sdpa
  崩，冒烟实证）、forced 基张量（当前块区 True）；ctx 跨 sample 调用存续，
  多 rep 命中同对象；折叠路径要求 B==1（`ValueError`）；
- **强制散度活跃块（FR2 核心，P5 的构造级绕开）**：AR019 P5 实证活跃小块内
  部分 mask→token 翻转经块内双向注意力传播、tau=0.95 gate 把同块未翻位置
  误判 stable（parity 92.9% 先例）——所以链折叠的精确性**不依赖 gate**：
  当前块整区（含 AR 步前向的新提交块区，恰为 P5 场景）经 `forced_divergent`
  尾参在 gate 判定前强制判散度、全量重算；前缀块照常走 gate（P3/P4 位稳定
  证据，复用精确）⇒ **folded-serial 与 raw bit-exact 由构造保证**。eager
  面四条 gate 路径（fused/plain/变长/批量）单一排序：stable_mask 物化 →
  AND `~forced` → profiler 记账 → 计数消费（fused 路径 kernel 计数弃用走
  fallback `sum()`，不改 Triton kernel）；图面由 runner 持 `forced_static`
  缓冲，捕获体内 AND + count 设备侧归约重导出，replay 前 `copy_` 零 sync；
  forced 捕获身份 = None/not-None + 形状（forced 张量每次调用新建，对象级
  `is` 永不命中）；
- **形状键控 runner 池（FR2，跨块增长的图收益）**：**opt-in**
  `graph_runner_pool_size`（默认 0 = legacy 单 runner warn-once 行为逐字节，
  UT-1504 面零扰动）；池键实现载体 `(tokens.shape, mask 对象)` 与语义键
  (T,prompt_len)/(N,T,prompt_len) 窗口存续期内双射；同形状窗口首捕获后续
  重放、画布增长新键新捕获、LRU 上限驱逐（驱逐时回收 cuBLAS workspace——
  capture 侧流泄漏 ~8.5MB/次）；#37 图前置条件**零放宽**——池只是让"每形状
  一次捕获"在增长的画布上成立；`cache` setter 传播至两池全部 runner
  （#42 扩展）；`graph_pool_stats` 只读观测面暴露按形状键计数与**两类
  eager 计数**（降级类 = 全部非结构性回退，canary 面；结构性 = 窗口首前向/
  切换首串行/末窗，by-design 单独计数）；
- **批量投机块验证（FR3，opt-in）**：`num_block_candidates=N`（默认 0 关，
  逐字节不变）把每个小块的逐迭代 remasking 换成：父前向 top-k 起草 N 候选
  （种子公式冻结、跨运行可复现）→ 单次 `[N,T]` 批量 MFF 验证（branch_ids +
  共享父分支 + 钉定 mask/pe）→ min-prob 准则选定提交、跳过剩余迭代、链父
  切换为胜者分支——AR016 批量图（批量池）+ AR017 自适应容量 + AR018 快照
  发布 + AR020 `put_batch` **首次被生产回路行使**。

实测（真实 generate() 三模式，reps=30 预承诺单次运行）：**H1 supported
（真实 checkpoint bit-exact——token 一致率 1.0、逐前向 max_abs_diff=0.0）**；
H2 inconclusive（Δ=−107.5ms 未越 2σ=265.3ms——num_tokens=128 小形状下串行
折叠供给/发布开销 > 前缀复用收益，两 prompt 方向一致，如实记录）；**H3
supported（batched vs serial 2.47× / vs baseline 2.38×）**；H4 supported
（池计数逐位对账、账本 c=4 界内、canary 降级类 timed=0、AR017 事件全落
warmup=中性）。**生产单发注记（A5）**：单次 generate 下每形状窗仅 1 次
`[N,T]` 批量调用，捕获摊销为负——单发场景建议 `use_cuda_graph=False`
（基准/批量场景 warmup+reps 摊销成立）。

#### 11. 贪心锚定候选（AR022）：批量投机的质量对齐

AR021 的如实遗留：批量投机把整代墙钟压到 2.38×，但投机提交是**非贪心**
的——min-prob 准则选出的候选行不携带 draft argmax 的停止/置信结构，
实测块级 exact-match 0.0、post-hoc 置信低于 baseline、且不命中 baseline
的 stop token（prompt0 上 baseline 34 token 早停而 batched 生成满 128）。
AR022 不换选择准则、不加特判，只用一个构建算子把贪心结构**送回竞争面**：

- **贪心行（FR1）**：`candidate_include_greedy=True` 时 `[N,T]` 画布的
  **行 0 = 贪心假设**——全部仍掩码位置填 draft top-k **有序集首位**
  （`sorted=True` topk 的首列，srs 冻结的 argmax 定义），纯设备索引构建：
  **确定性、零设备回读、零 generator 消费**（随机行用专用 CPU generator
  采样 top-k 内选项，贪心行不消耗任何随机性）；
- **N 语义**：`num_block_candidates` 仍是**候选总数**——贪心占 0 号位，
  随机行变 `j=1..N-1` 且在冻结种子公式
  `(S0·1_000_003 + block_idx·8_191 + small_block_idx·131 + j) & 0x7FFF_FFFF`
  中保持行号 ⇒ 行 1..N-1 与 greedy-off 构建**逐位相同**（仅画布一致区
  成立：首个投机小块内构造成立，胜者分叉后失效——A/B 是构建算子对照面，
  非全程轨迹等价）；`_cand{j}` 自 AR021 起即"行 j"语义，分支命名零改动；
- **公平竞争、可落败**：贪心行进入**原封不动的 min-prob 准则**，没有任何
  行 0 特判——UT-2203 的后验陷阱 fixture（零混合层 + 正交嵌入 + 种子重放
  all-C 候选行）构造级证明贪心行可输；选定时提交/停止/链延续与既有语义
  完全一致；
- **观测面与配置面（FR2/FR3）**：`greedy_win_stats` 只读 property
  `{"greedy_wins", "speculative_calls"}`——实例级计数器跨 sample 存续
  （graph_pool_stats 先例），仅贪心锚定投机路径递增（默认关恒 `{0, 0}`），
  胜出判定 `j_star == 0`，基准以 before/after 快照差分导出 rate；
  `ActFoldConfig.candidate_include_greedy` 提供 YAML 消费面（免
  unknown-key 警告），`True` 且 `num_block_candidates==0` 在 sampler
  config 构造期 `ValueError`（贪心锚定仅定义于投机路径）。

四模式基准（新增 folded-batched-greedy = MFF_C：与 MFF_B **同构独立实例**
——自有 cache/gate/pool/canary/AR017 policy，同 `[N,T]` 形状/N=4/top_k=10/
prompts/seed，仅 `candidate_include_greedy=True`——速度变量构造性清零；
folded-batched sidecar 同时充当 AR021 复现自检，strict 失配则四 verdict
全置 blocked）给出定案：**H1-H4 全 supported**——质量三面全过（位置级
4/4=1.0、块级 4/4=1.0 vs batched 0.0、posthoc 逐位==baseline gap=0.0），
早停完全恢复（34/66 == baseline），且贪心行命中 stop token 使对齐模式
反而**更快**（p50 440.9 vs 1300.6ms，7.02× vs baseline）——速度与质量
在本工作负载上不再是对立面。greedy_win_rate 1.0（2/2）是本负载的观测值
（机制允许落败），不是普适断言。

#### 12. 多 parent 批量折叠与双链 lockstep 并行投机（AR025）：跨链候选共享一次前向

AR015 的批量契约限定**同父**——多条投机链（如双 prompt 并行投机）只能
逐链各发一次 `[N,T]` 批量前向，写账本与 launch 随链数线性放大，跨链候选
从不共享一次前向。AR025 分三面拆除该限制：

- **机制面（`parent_branch_ids`）**：`ManualFoldedForward.forward` 新尾参
  接受 per-row 父 id 序列——`[sum(N_c), T]` 的每一行 gather 自己的父激活，
  行级降级（父缺失/形状失配的行整行重算，绝不炸批），逐行 gate/merge 与
  #41 语义逐字相同；与单数 `parent_branch_id` 互斥（双传 `ValueError`），
  `None` 不附加 kwarg——既有调用逐字节不变。写发布走 #42 `put_batch` 冲刷
  语义（能力探测每次发布重探，无能力 cache 逐字节走原循环）。
- **采样面（`sample_joint` W2 lockstep）**：单实例同时驱动双链——per-chain
  画布/折叠链状态/链限定 cache tag（`llada_spec_j{c}_b{block}` /
  `dream_spec_j{c}_u{unit}`），双实例禁令（branch id 碰撞）在入口 `ValueError`
  封死；共享投机 ctx（`(T, prompt_len)` 键），双窗循环：per-chain
  draft+build → **joint C4 单次 `[2N, T]` 折叠前向**（`branch_ids=链序×
  候选序`、`parent_branch_ids=[p1]*N+[p2]*N`、`forced_divergent`）→
  per-chain score/commit → refill 换入换出。`_speculative_commit` 按
  forward 接缝拆分为 `_speculative_build`（段A：batch-1 守卫、候选构造、
  冻结种子公式、EOS-aware 行）与 `_speculative_score_commit`（段B：region
  切片、min-prob、j_star、`_folding_parent` 写、commit），组合入口语义
  逐字节同——solo 路径零扰动由冻结面哨兵（96 例）钉死。
- **等价构造前提**：冻结种子公式**无链分量**——同 (unit, sub, j) 在两链上
  取同一种子 ⇒ joint 与逐链 solo 的候选画布逐位相同，A/B 差异被构造性
  归零到「一次前向 vs 两次前向」本身。入口守卫（仅 armed 路径）：空列表/
  batch>1/prompt 长度不等（截断到公共长度由基准层做）/cfg_scale>0/
  temperature>0/stochastic transfer 全部 `ValueError`。

合成面基准（8 形状点 = chains∈{2,4} × N∈{4,8} × T∈{256,1024}，双构造，
CPU 逐位纪律）定案：**16/16 全过**——S1 supported（nonsplit
`torch.equal` 逐位 / split `allclose(1e-6)`，#41 分档容差）；S2 supported
（写账本 A=4 **严格小于** B=40–160，每点每构造）；S3 账本缩减比
0.9–0.975 report-only。**DD8 erratum 留档**：首跑 8192 行/层预算下 A 侧
burst `put_batch` 按 token 行数 LRU 逐出在飞父条目（先于子条目写入故最旧）
→ 高人口点整批重算（值正确、折叠丢失）——容量混淆因子非机制缺陷，预算
按网格最坏人口取 65536 后冻结判定式零改动 16/16 全过；真实面几何
（~2560 行/层）不触发该边界。真实面基准（LLaDA/Dream 双链 A/B，reps=30
预承诺单次）与判定式（H1 逐位硬门+logits allclose 归因回放、H2 严格
p50(B)/p50(A)>1.0、交付线 ≥10%、H4 账本严格小、canary graph-off）已随
T006 接线交付（`--mode real`，AR023 模板），正式数字待高速算力机按
RERUN 步骤 16 执行回填——16GB WDDM 本机实测 bf16 权重 15.8GB+折叠缓存
即触顶硬崩、int8 逃生门在驱动坏态下量化停滞，均如实注记于 RERUN。

---

## 最新实验结果

### 多 parent 批量折叠合成基准（AR025）：跨链候选共享一次前向，16/16 全过（最新头条，机制面交付）

**工作负载**：合成因果 cumsum decoder（L=4/H=64/V=128，prefix-stable
scan + 可探测 FFN 链），`SimilarityGate` cosine τ=0.95，CPU 逐位纪律
（#41），seed 2026，确定性单发（无计时——墙钟统计属真实面专有）。
**三方对照**（每形状点×每构造）：**A** = 合并多 parent 批量
（`multi_parent_batch` 开，单次 `[sum(N_c), T]` 折叠调用 + per-row
`parent_branch_ids` + `put_batch` 发布）/ **B** = 逐链单 parent 批量
（开关关，每链一次 `[N,T]` `branch_ids` 调用——AR025 前的形态）/
**C** = 串行逐候选（report-only 上下文）。

**网格**：8 形状点（chains∈{2,4} × N∈{4,8} × T∈{256,1024}）× 双构造
（nonsplit / split FFN）= 16 面；缓存预算 65536 行/层（按网格最坏人口
(4+32)·1024=36864 取值——DD8 erratum，见下）。

**判定（预承诺，判定式全文冻结于工件 meta；S1/S2 supported、S3
report-only）**：

| 面 | 判定 | 结果 |
|----|------|------|
| S1 奇偶 | A vs B logits：nonsplit `torch.equal` 逐位 / split `allclose(1e-6)` | **supported（16/16）** |
| S2 写账本 | `ops(A) < ops(B)` **严格**（每点每构造） | **supported**：A=4 vs B=40（2 链）–160（4 链 N=8） |
| S3 缩减比 | `1 − ops(A)/ops(B)` report-only | **0.9–0.975**（写账本 10–40× 缩减） |

**DD8 erratum（诚实留档）**：首跑 8192 行/层预算下 5 点 S1 FAIL——
根因不是机制而是**容量混淆因子**：A 侧 L0 burst `put_batch` 按 token
行数 LRU 逐出在飞父条目（先于子条目写入故最旧）→ L1+ 取父 miss → 整批
重算（值正确、折叠丢失），与逐链交错路径（每层 fetch 刷新幸存）不对称；
FAIL ⟺ (chains+chains·N)·T>8192 与 5 点完全吻合。按网格最坏人口取
65536 后**冻结判定式零改动**重跑 16/16 全过；blocked 首跑工件独立保全
（`ar025_multi_parent_bench_eviction_blocked.json`）。机制级逐出边界入
design DD10（真实面几何 ~2560 行/层不触发）。

**真实面（T006 接线已交付，正式数字待实机）**：LLaDA-8B/Dream-7B 双链
lockstep A/B（A=`sample_joint`×1 双链总时 / B=`sample()`×2 串行，同
prompt 对同 seed，AR023 测量同构）——DD8 冻结判定式（H1 逐位硬门 +
logits allclose 归因回放、H2 严格 p50(B)/p50(A)>1.0、交付线 ≥10%、H4
账本严格小、canary graph-off）与 AR023 模板（probe 结论门、复现双跑、
#39 round-robin 计时）已冻结入 `scripts/ar025_multi_parent_bench.py
--mode real`；本机 16GB WDDM 实测 bf16 权重 15.8GB+折叠缓存即触顶硬崩
（int8 逃生门在驱动坏态下量化停滞）——正式 reps=30 按
[`docs/RERUN_CHECKLIST.md`](docs/RERUN_CHECKLIST.md) 第 16 步在高速算力
机执行后回填，勿以冒烟替代。

工件：`results/optimization/ar025_multi_parent_bench.json`（判定式全文 +
erratum 注记 + quick 冒烟独立工件
`ar025_multi_parent_bench_quick.json`）；复现
`python -m scripts.ar025_multi_parent_bench --mode synth`（CPU，分钟级；
`--quick` 单点）。UT：`tests/test_ar025_sample_joint.py` 13 例（sample_joint
vs 逐链逐位相等 + 账本严格小 + joint 守卫面）+ 冻结面哨兵 96 例。

### 四模式投机质量对齐基准（AR022）：贪心锚定 7.02× 且质量逐位对齐 baseline

**工作负载**：Fast-dLLM-v2-1.5B bf16（H=1536/L=28/V=151936，remote code）
真实 `sample` 生成回路——AR021 协议全承（seed 2026、P=2 prompt padded
32/64、num_tokens=128、warmup 每模式每 prompt 完整一轮丢弃、逐 rep
round-robin、宿主 `perf_counter`+sync），新增第四模式；四模式共享同一
模型实例/prompt 集/种子：**baseline** / **folded-serial**（AR021 FR2）/
**folded-batched**（AR021 FR3，MFF_B——其 sidecar 质量面同时充当 AR021
复现自检）/ **folded-batched-greedy**（AR022 对齐模式，MFF_C——与 MFF_B
同构独立实例，仅 `candidate_include_greedy=True`；同 `[N,T]` 形状/N=4/
top_k=10/seed ⇒ 速度变量**构造性清零**，同形状 A/B）。

**四模式整代墙钟 p50 ms**（Quadro RTX 5000 16GB，AR019 venv
transformers 4.53.1，torch 2.5.1+cu121，n=60/模式 = 2 prompt × 30 reps，
单次预承诺运行 2026-10-15T04:21 UTC）：

| 模式 | p50 | mean | std | vs baseline | vs folded-batched |
|------|-----|------|-----|-------------|-------------------|
| baseline（raw sampler） | 3096.8 | 3095.4 | 145.9 | — | — |
| folded-serial（链折叠+串行池） | 3205.4 | 3197.6 | 141.6 | 0.9661× | — |
| folded-batched（批量投机+批量池+AR017） | 1300.6 | 1315.1 | 808.4 | 2.381× | — |
| folded-batched-greedy（贪心锚定，MFF_C） | **440.9** | 440.9 | 61.8 | **7.0236×** | **2.9498×** |

逐 prompt p50（ms）：greedy **379.4/501.7** vs batched 2119.3/507.3——
贪心行命中 baseline 的 stop token、恢复早停路径，对齐模式在**加速侧**
通过速度保持门（H2 Δ=+859.682ms ≥ −2σ=−1616.7413ms，σ=808.3707）。

**质量面对齐三面（H1，硬门全过）——对齐模式恢复 baseline 质量**：

| 质量面 | baseline | folded-batched | folded-batched-greedy |
|--------|----------|----------------|-----------------------|
| 合并块级 exact-match（4 块） | —（参照） | 0/4 = 0.0 | **4/4 = 1.0** |
| 合并位置级一致率（窗口 [32,34)/[64,66)） | —（参照） | — | **4/4 = 1.0**（> 0.5 门） |
| post-hoc 置信 overall_mean（prompt0/1） | 0.369661 / 0.570341 | 0.289692 / 0.106418 | **0.369661 / 0.570341（gap = 0.0，逐位==baseline）** |
| 解码长度（prompt0/1） | 34 / 66（早停） | 160 / 77 | **34 / 66（早停完全恢复）** |

**假设判定（预承诺单次运行，无重跑钓鱼；判定式全文冻结于工件 meta）**：
**H1 质量对齐 supported（三面全过）**；**H2 速度保持 supported**（Δ 在
加速侧）；**H3 早停恢复 supported**（34/66 == baseline 实测）；
**H4 全栈行使 supported**——MFF_C 在域窗（p32_block0_T64/p64_block0_T96）
批量捕获 >0 且 timed 重放 30/30 >0；三 folded 模式降级类 canary
timed == 0（结构性 eager 60/150/60 单独计数，by-design）；发布账本
≤ 4·(L+N) 三模式逐字段（timed 段 ops serial 50460/batched 7980/greedy
3540）；AR017 事件仅触发时验证（batched cumulative 2 全落 warmup=中性；
greedy 0=未触发中性）。VRAM 峰值 5413.0/5413.1/5840.1/5672.0 MB。

**复现自检 strict PASS**：folded-batched sidecar 质量面对 AR021 工件先验
逐位相等（block_exact 0.0/0.0、len 160/77、posthoc_mean 0.289692/0.106418、
posthoc_min 5e-06/0.0——同环境确定性假设成立，无 reproduction_anomaly）。
**greedy_win 面**：greedy_win_rate 1.0（2 次投机调用 2 胜——本负载 P=2
恰各 1 个投机小块；贪心可落败是机制属性（UT-2203），win_rate 是观测面
不是断言面）。timed 段稳定比 serial 0.5774（n=47040）/batched 0.6790
（n=22680）/greedy 0.5833（n=8400）。

工件：`results/optimization/ar022_speculative_quality_bench.json`
（H1-H4 判定式全文 + 贪心锚定声明 + 复现自检先验 `_AR021_PRIORS` 冻结于
meta）；复现 `python -m scripts.ar022_speculative_quality_bench`
（AR019 venv + 本地权重，[`docs/RERUN_CHECKLIST.md`](docs/RERUN_CHECKLIST.md)
第 12 步；`--quick` 冒烟 ≤5min，其复现自检降级为结构性验证并如实注记）。

### 生产生成回路三模式基准（AR021）：真实 generate() 首测，batched 2.38×

**工作负载**：Fast-dLLM-v2-1.5B bf16（H=1536/L=28/V=151936，remote code）
真实 **generate()**——P=2 prompt（预 padding 至 32 倍数，padded 32/64），
num_tokens=128（W=4 形状窗），seed 2026；三模式共享同一 prompt 集/种子/模型
实例：**baseline**（raw sampler 配方不变）/ **folded-serial**（FR2 链折叠 +
串行形状键控池）/ **folded-batched**（FR3 批量投机 N=4 + 批量池 + AR017
自适应容量接通）；warmup 每模式每 prompt 完整一轮丢弃（每形状窗池预捕获 +
自适应过渡期，捕获成本不入 p50），逐 rep 三模式 round-robin 交错；宿主
`perf_counter` + `torch.cuda.synchronize()` 夹逼（#39 CUPTI null 注记）。

**三模式整代墙钟 p50 ms**（Quadro RTX 5000 16GB，AR019 venv
transformers 4.53.1，torch 2.5.1+cu121，n=60/模式 = 2 prompt × 30 reps）：

| 模式 | p50 | mean | std | vs baseline | vs folded-serial |
|------|-----|------|-----|-------------|------------------|
| baseline（raw sampler） | 2840.7 | 2850.9 | 132.7 | — | — |
| folded-serial（链折叠+串行池） | 2948.3 | 2959.9 | 124.7 | 0.9635× | — |
| folded-batched（批量投机+批量池+AR017） | **1193.9** | 1210.0 | 741.9 | **2.3795×** | **2.4695×** |

逐 prompt p50（ms）：baseline 2717.2/2975.0；folded-serial 2839.0/3082.3；
folded-batched **1956.4/468.4**（prompt1 两模式均早停——baseline 66 /
batched 77 token，未到满生成 128；468.4ms 的量级差主要来自"1 次批量验证 +
2 次串行前向替代 27 次串行前向"的负载压缩）。

**假设判定（预承诺单次运行 2026-10-14，无重跑钓鱼；判定式全文冻结于工件
meta）**：

- **H1 folded-serial 等价性：supported（真实 checkpoint bit-exact，超规格
  强度）**——两 prompt token 逐位一致率 1.0、前向数一致（31/27）、逐前向
  max_abs_diff=0.0、bit_equal_fraction_mean=1.0（声明硬门为 argmax 逐位
  一致 + bf16 allclose，bit-parity 为报告子字段——实测直接 bit-exact；
  合成面由 UT-2105 双形态 mini-model 构造钉死）；
- **H2 folded-serial 墙钟 < baseline：inconclusive**——Δp50=−107.5181ms
  （两 prompt 数值均慢，方向一致），|Δ| < 2σ=265.3231ms
  （σ=132.6615=max(σ_base,σ_serial)）；num_tokens=128 小形状下串行折叠
  供给/发布开销 > 前缀复用收益，未越预注册噪声界，如实记录；
- **H3 folded-batched 墙钟 < folded-serial：supported（速度分面）**——
  Δp50=1754.3966ms > 2σ=1483.8483ms（σ=741.9241）；质量分面 report-only
  （不设门槛、不预设）：块级 exact-match 率 0.0（两 prompt），post-hoc
  置信 baseline 0.3697/0.5703 vs batched 0.2897/0.1064——投机提交非贪心
  不命中 baseline 的 stop token（prompt0：baseline 34 token 早停 vs
  batched 生成满 128 token），属设计行为，如实报告；
- **H4 全栈生产行使：supported**——在域窗（≥2 前向）串行捕获 1/1、timed
  重放 900/780 >0（folded-serial）；批量捕获 5/5 窗 >0（含 AR017 自适应
  重捕获 2 次落 p32_block2_T128/p32_block3_T160，全在 warmup 段、timed=0
  =中性）；发布账本 c=4 界内逐字段（sidecar 代 serial 899≤3596/783≤3132、
  batched 207≤1324/59≤360 ops vs bounds）；no-fallback canary 降级类
  timed=0 双模式全真（结构性 eager 60/150 timed 单独计数，by-design）；
  VRAM 峰值 4947.1/4947.1/5374.5 MB（16GB 卡余量充足）。

**池计数侧车**：cap=8 零驱逐（LRU 为防御上限）；folded-serial 串行池
2 runner（timed 重放 900/780）；folded-batched 串行池 4 + 批量池 5 runner
（timed 重放各 30）；timed 段稳定比 serial mean 0.5774（n=47040）/batched
mean 0.6790（n=22680）。

工件：`results/optimization/ar021_generation_loop_bench.json`（判定式全文 +
c=4 + canary 二分口径 + A5 摊销注记冻结于 meta；三模式逐块/逐 prompt 墙钟 +
按形状窗池计数 + 账本 + 质量面）；复现
`python -m scripts.ar021_generation_loop_bench`（AR019 venv + 本地权重，
[`docs/RERUN_CHECKLIST.md`](docs/RERUN_CHECKLIST.md) 第 11 步；
`--quick` 冒烟 ≤5min）。

### 发布路径 cache put 批量化 A/B（AR020）：真实 checkpoint 发布窗口 1.77–1.97×

**工作负载**：AR019 同款真实块验证（Fast-dLLM-v2-1.5B bf16，H=1536/L=28/
V=151936，remote code）；A/B 变体 = **对称包装** cache（loop = 不暴露
`put_batch` 的委托包装 / batch = 直 delegation，两侧同增一跳 Python 委托），
经 `mff.cache` 交换实现（能力探测每次发布重探）。宿主 `perf_counter` +
`torch.cuda.synchronize()` 夹逼计时——发布成本恰以宿主侧为主体。

**发布窗口 p50 ms**（Quadro RTX 5000 16GB，AR019 venv transformers 4.53.1，
seed 2026，n=30 交错；σ = max(σ_loop, σ_batch)；账本 = write_op_count）：

| 配置 | loop 发布 | batch 发布 | 加速比 | Δp50 vs 2σ | 主机调用账本 | put 微基准 |
|------|----------|-----------|--------|------------|--------------|-----------|
| T=256, N=4 | 7.894  | **4.393** | **1.80×** | 3.500 > 2.380 ✓ | 116 → **1** | 3.870→1.936（2.00×） |
| T=256, N=8 | 16.809 | **9.510** | **1.77×** | 7.299 > 3.465 ✓ | 232 → **1** | 9.569→4.514（2.12×） |
| T=512, N=4 | 8.708  | **4.425** | **1.97×** | 4.282 > 2.315 ✓ | 116 → **1** | 3.757→2.477（1.52×） |

**端到端四路径 p50 ms**（同口径；批量 put 为生产默认）：

| 配置 | baseline | graph 固定 C=0.5 | adaptive（loop） | adaptive（batch） | ab vs baseline |
|------|----------|-----------------|-----------------|-------------------|----------------|
| T=256, N=4 | 672.242  | 403.516 | 284.046 | **280.146** | **2.40×** |
| T=256, N=8 | 1368.044 | 790.898 | 519.580 | **511.629** | **2.67×** |
| T=512, N=4 | 1267.166 | 814.486 | 490.426 | **485.967** | **2.61×** |

**假设判定（预承诺单次运行，H1–H4 全 supported，零失败配置）**：

- **H1 等价性：supported**——UT 面 56/56（三 cache × CPU/CUDA 随机化性质
  等价）+ 真实模型发布面三配置 116/232/116 张量全 **bit-equal**；
- **H2 发布窗口：supported（3/3）**——Δp50 = 3.500/7.299/4.282ms 全部
  超过 2σ（2.380/3.465/2.315ms），−43.4%~−49.2%；
- **H3 端到端：supported**——零回退（batch 发布计数 38/38 = 期望步数、
  pe canary 零失配、稳态零预算违例）+ 三配置 batch ≤ loop+2σ + 最小配置
  (256,4) 严格改善 3.900ms > 2σ = 3.712ms；
- **H4 主机调用账本：supported（3/3）**——发布窗口 write_op_count
  **116/232/116 → 1**（单 (device,dtype) 组一个 foreach），远低于冻结
  c=2 的 c·(L+N)=64/72/64；CUPTI null（#39）下账本为结构证据。

工件：`results/optimization/ar020_publish_put_bench.json`（判定式全文 +
c=2 + 最小配置冻结于 meta；三层测量 + 正确性门 + 四路径端到端 + no-fallback
canary）；复现 `python -m scripts.ar020_publish_put_bench`（AR019 venv +
本地权重，[`docs/RERUN_CHECKLIST.md`](docs/RERUN_CHECKLIST.md) 第 10 步；
`--quick` 冒烟）。

### 真实 checkpoint 端到端墙钟（AR019）：Fast-dLLM-v2-1.5B 三路径

**工作负载**：Fast-dLLM-v2-1.5B bf16（H=1536/L=28/V=151936，remote code）真实
块验证——parent = K_PROMPT prompt token + 32×MASK(151665) 全遮块，children =
top-10 模型采样填充；块因果掩码 `(q//32) ≥ (kv//32)`；position_embeddings 由
模型自身 rotary_emb 按常量位置方案计算（AR019 新接口供给）。

**三路径 p50 ms**（Quadro RTX 5000 16GB，AR019 venv transformers 4.53.1，
seed 2026，n=30 round-robin 交错；每步 = N 候选完整验证）：

| 配置 | baseline eager | graph 固定 C=0.5 | graph 自适应（AR019） | adaptive vs baseline | fixed vs baseline | 容量轨迹（固定→右移） | 峰值 VRAM | H1（2σ） |
|------|---------------|-----------------|----------------------|---------------------|-------------------|----------------------|-----------|----------|
| T=256, N=4 | 796.148  | 450.645  | **326.890**  | **2.44×** | 1.77× | 512→160（需求 128）  | 5624 MB | inconclusive |
| T=256, N=8 | 1393.528 | 815.931  | **532.513**  | **2.62×** | 1.71× | 1024→320（需求 256） | 7640 MB | supported |
| T=512, N=4 | 1349.446 | 868.789  | **520.735**  | **2.59×** | 1.55× | 1024→160（需求 128） | 8949 MB | supported |

**假设判定（声明与判定分离，如实呈现）**：

- **H1 自适应显著快于 baseline：整体 inconclusive（2/3 配置 supported）**——
  三配置 p50 加速比全部 2.44/2.62/2.59×，方向压倒性；(256,4) 为首个配置，
  WDDM 热身窗方差（baseline std 497ms ≈ p50 的 62%）淹没 2σ 判定，后两配置
  std 5–40ms 干净且 supported——与 AR017/18 同类"首配置欠功效"先例一致，
  按预承诺单次运行纪律如实记录；
- **H2 三路径一致：supported（3/3）**——三配置两 graph 路径 vs baseline
  **maxdiff=0.0、argmax 一致率 1.0**（逐位一致，强于 allclose(1e-2,1e-1) 档）；
- **H3 门控块二分：supported（3/3）**——每 (layer, branch) stable ratio
  **精确等于** K_PROMPT/T（0.875/0.875/0.9375，n=112/224/112 全部相等，零偏差
  ——块因果语义下 prompt 块位级稳定、mask 嵌入强发散，门控完美二分）；
- **H4 容量右移：supported（3/3）**——捕获形状 [N,T] ✓、轨迹恰一次收缩重捕获
  后收敛、稳态违例 0、末容量 160/320/160 ≥ 需求 128/256/128 且 < 固定
  512/1024/1024。

**侧车证据**：pe 恒一回退计数全程 **0**（脚本层 canary `is` 匹配，图路径无一次
静默回退）；Triton bf16 gate/merge 双 kernel 因本机无 C 编译器走 PyTorch 回退
（AR014 键记录，warmup 侧流先回退再捕获，不进捕获体）——即便如此三配置仍
2.44–2.62×。这是**真实 checkpoint 图捕获首证**：bf16 1.5B 权重 + remote
28 层 + 块因果掩码 + rotary 全链路 CUDA graph 捕获/重放成功。

工件：`results/optimization/ar019_real_e2e_bench.json`（含权重 SHA256、venv
标记、容量轨迹、Triton 回退键、CUPTI null 注记）；复现
`python -m scripts.ar019_real_e2e_bench`（AR019 venv + 本地权重，详参
[`docs/RERUN_CHECKLIST.md`](docs/RERUN_CHECKLIST.md) 第 9 步；`--quick` 冒烟）。

> 谱系：本表是 P1-7 端到端证据链的**第五档（真实 checkpoint）**——其四级
> 合成基准演进（AR015→AR016→AR017→AR018）见下表，两表同口径（三路径、
> round-robin、p50、同型 GPU）；AR020 在同一工作负载上进一步给出发布窗口
> A/B（loop vs batch 发布，见上表）；AR021 把整栈接入生产生成回路并在真实
> generate() 面三模式复测；AR022 以贪心锚定候选把批量投机的质量面对齐
> baseline（7.02×，见头条表）。

### 合成基准端到端墙钟（AR018）：折叠验证 vs 裸基线（P1-7 四级演进）

**最终结果**（合成 LLaMA-layout，V=1000，T=512，Quadro RTX 5000 16GB，
WDDM，驱动 556.18，seed 2026，n=30 round-robin 交错，p50 ms；
每步 = N 候选完整验证）：

| 配置 (H, L, N) | baseline eager | graph 固定容量 | graph 自适应（AR018） | vs baseline（AR017 → AR018） |
|----------------|---------------|---------------|----------------------|------------------------------|
| H=256, L=8, N=4  | 11.315  | 5.664  | **4.708**  | 1.55× → **2.40×** |
| H=256, L=8, N=8  | 22.197  | 9.910  | **8.300**  | 1.68× → **2.67×** |
| H=1024, L=16, N=4 | 70.026 | 46.425 | **26.318** | 2.30× → **2.66×** |
| H=1024, L=16, N=8 | 143.186 | 89.444 | **48.594** | 2.46× → **2.95×** |

> 演进链：AR015 批量前向（eager 串行 → 批量 2.0–5.6×）→ AR016 批量图捕获
> （vs baseline 1.37–1.77×，方向全正）→ AR017 自适应容量（深几何 2.30/2.46×
> decisively）→ AR018 发布压缩（四配置全面抬升，小几何从 ~1.6× 抬到 ~2.5×）
> → AR019 真实 checkpoint 复测（Fast-dLLM-v2-1.5B 2.44–2.62×，逐位一致）
> → AR020 发布 put 批量化（发布窗口 1.77–1.97×，端到端 2.40–2.67×）
> → AR021 生产生成回路折叠接线（真实 generate()：batched vs baseline
> 2.38×、vs serial 2.47×，folded-serial parity bit-exact）
> → AR022 贪心锚定候选（质量对齐：posthoc gap=0.0、早停恢复，
> 7.02× vs baseline）。
> 全部配置三路径 argmax 精确一致 + 自适应 vs 固定 allclose 1e-4；每分支
> stable_ratio 与独立逐行重算逐值相等。

**机理验证：六相相位分解**（AR018 自适应路径稳态，p50 ms）：

| 配置 | revalidate | prefill | replay | readback | publish | clone | 发布占比（AR017 探针 → AR018） |
|------|-----------|---------|--------|----------|---------|-------|--------------------------------|
| H=256, L=8, N=4  | 0.110 | 0.318 | 1.833  | 0.052 | **2.132** | 0.081 | 60.7% → **45.3%** |
| H=256, L=8, N=8  | 0.111 | 0.314 | 3.237  | 0.057 | **4.225** | 0.125 | 71.2% → **50.9%** |
| H=1024, L=16, N=4 | 0.184 | 0.826 | 20.664 | 0.101 | **4.144** | 0.083 | 29.7% → **15.7%** |
| H=1024, L=16, N=8 | 0.195 | 1.556 | 36.190 | 0.128 | **8.545** | 0.132 | 31.5% → **17.6%** |

**容量追踪证据**（AR017）：D_frac 实测 13–16%，自适应容量轨迹
[1024, 1024, 340, …] / [2048, 2048, 680, …] 在恰一次收缩后稳定，最终容量
340/680 对需求 p50 272/544——FFN 重算较固定 C=0.5 缩减 **3.01–3.02×**，
适配后零预算违例。

工件：`results/optimization/ar018_publish_bench.json`（含 AR017 跨工件锚点、
相位分解、峰值显存、CUPTI 探测注记）；复现 `python -m scripts.ar018_publish_bench`
（`--quick` 冒烟）。

### 真实模型两阶段实验（LLaDA-8B / Dream-7B / Fast-dLLM-v2-1.5B）

两阶段受控实验覆盖三个真实 Diffusion LLM 家族（bfloat16）：诊断阶段
（RTX 6000D 84GB）+ 优化阶段（RTX PRO 6000 Blackwell 96GB）。子分支由随机
翻转 `n ∈ {0,1,2,4,8,16,32,128}` 个 token 构造。原始数据：
`results/experiments/` 与 `results/optimization/`。

#### 1. 算法不变量

| 不变量 | Fast-dLLM | LLaDA-8B | Dream-7B |
|---|---|---|---|
| 自折叠 stable ratio / MSE | 1.000 / **0.0** | 1.000 / **0.0** | 1.000 / **0.0** |
| 强制全 divergent stable / MSE | 0.000 / **0.0** | 0.000 / **0.0** | 0.000 / **0.0** |
| 基线全量前向 (ms) | 16.5 | 17.6 | 16.6 |
| 旧缓存全稳定折叠 (ms) | 52.1 | 51.2 | 53.7 |
| **向量化缓存全稳定折叠 (ms)** | **6.45** | **6.37** | **6.81** |

自折叠与强制全 divergent 前向与基线**逐位一致**；向量化后全稳定折叠路径
**2.3–2.7× 快于重算**。

#### 2. 相似度结构与阈值

| 模型 | 翻转 1 token（mean / p05） | 翻转 8 token（mean / p05） |
|---|---|---|
| Fast-dLLM-v2-1.5B | 0.9797 / 0.9584 | 0.7957 / 0.2494 |
| LLaDA-8B-Instruct | 0.9387 / 0.6029 | 0.8194 / 0.3945 |
| Dream-7B-Instruct | 0.9408 / 0.5939 | 0.6932 / 0.1654 |

stable ratio 在 τ∈[0.5, 0.99] 形成平台、仅在 0.999 附近崩塌（bf16 噪声主导）；
LLaDA/Dream 对单 token 扰动的敏感度约为 Fast-dLLM 的六倍。逐层折叠最优区
是模型相关的（Fast-dLLM 早期层、LLaDA/Dream 晚期层最安全）。

#### 3. 零加速根因诊断（第一阶段）

| 单层成本 (ms) | Fast-dLLM | LLaDA-8B | Dream-7B |
|---|---|---|---|
| 层重算 | 0.538 | 0.533 | 0.606 |
| **缓存读取（旧实现）** | **0.699** | **0.652** | **0.697** |
| 缓存写入 | 0.228 | 0.190 | 0.230 |
| 相似度门控 | 0.067 | 0.065 | 0.066 |

缓存读取成本超过层计算本身，且旧实现只要层内出现一个 divergent token 就整层
重算——这是"算法省 FLOPs、实测零加速"的完整根因链，也是向量化缓存的选题依据。

#### 4. 跨步扩散采样

| 模型 | token 与基线一致率 | 平均逐步 stable ratio | baseline / folded (ms) |
|---|---|---|---|
| Fast-dLLM-v2-1.5B | **1.000** | 0.980 | 471 / 3791 |
| LLaDA-8B-Instruct | 0.727 | 0.989 | 726 / 3730 |
| Dream-7B-Instruct | 0.712 | 0.987 | 768 / 3867 |

Fast-dLLM 的折叠采样与基线**逐 token 一致**；LLaDA/Dream 达到同等 stable
ratio 但因 logits 对扰动更敏感、贪心解码放大早期差异而损失一致性（提高 τ
可缓解）。

#### 5. 优化 #2：层内 FFN 拆分

| 形状（LLaDA 层，H=4096） | 1% divergent | 10% divergent | 50% divergent |
|---|---|---|---|
| B=1, T=2048 | 1.78× | 1.69× | 1.24× |
| B=4, T=2048 | **1.93×** | 1.78× | 1.32× |

拆分与不拆分输出 MSE/top-1 完全一致；`batch × seq ≥ 512` 才自动启用。

#### 6. 优化 #3：自适应分位门控

| 模型（翻转 1 token） | 固定 τ=0.99（stable / rel-MSE / top-1） | 自适应目标 0.8（stable / rel-MSE / top-1） |
|---|---|---|
| Fast-dLLM | 0.976 / 6.07e-2 / 0.927 | 0.805 / **4.33e-2** / 0.927 |
| Dream-7B | 0.976 / 2.20e-1 / 0.902 | 0.805 / **8.35e-2** / **0.976** |

门控可精确命中任意目标 stable ratio，覆盖固定阈值无法触达的平台/悬崖区。

#### 7. 优化 #4：融合 gather-select kernel

T=8192, H=4096 时融合 Triton kernel 达 **3.43×**（逐位精确，仅大形状启用）。

#### 8. 推荐配置

| 场景 | 配置 |
|---|---|
| 生产块解码生成（Fast-dLLM v2） | `sampler.sample(prompt_ids, folded_model=mff)`（AR021）：前缀块 gate 复用 + 活跃块强制散度（parity 由构造 bit-exact）+ `graph_runner_pool_size=8` 形状键控池；批量投机 `num_block_candidates=4` + **`candidate_include_greedy=True`**（AR022 质量对齐：posthoc==baseline、早停恢复，**7.02× vs baseline**）；**单发场景建议 `use_cuda_graph=False`**（捕获摊销为负，A5 注记） |
| 多分支验证（同父等长候选） | `ManualFoldedForward(use_cuda_graph=True, graph_adaptive_capacity=True)`——批量图路径 + 批量发布（`put_batch` 能力自动探测，AR020），真实 checkpoint **2.40–2.67×**（AR020）/ 合成 2.40–2.95×（AR018）vs 裸基线 |
| remote-code 模型（Fast-dLLM v2 等） | 供给 `position_embeddings=(cos, sin)`（AR019 新参，模型自身 rotary_emb 产出）；transformers 4.53.1 venv（remote code 版本钉住） |
| 长序列 / 大 batch（B×T ≥ 2048） | 追加 `use_split_layers: true`；T ≥ 8192 时融合 kernel 自动启用 |
| 保真敏感复用 | `AdaptiveQuantileGate(target_stable_ratio=...)`；Dream 仅折叠后 1/3 层 |
| 采样质量优先 | τ ≥ 0.99 或自适应目标 0.9–0.95；Fast-dLLM 容忍跨步折叠 |

#### 9. 实测图

| | |
|---|---|
| ![Similarity heatmap](figures/experiments/fig_similarity_heatmap.png) | ![Similarity distribution](figures/experiments/fig_similarity_hist.png) |
| ![Per-layer stability](figures/experiments/fig_stable_by_layer.png) | ![Tau quality](figures/experiments/fig_tau_quality.png) |
| ![Vectorized cache](results/optimization/figures/fig_opt1_cache.png) | ![Adaptive frontier](results/optimization/figures/fig_opt3_frontier.png) |

#### 10. 复现

```bash
# 诊断（逐模型）
python scripts/algo_experiments.py --model fastdllm --out results/experiments/fastdllm
python scripts/overhead_bench.py --model fastdllm --out results/experiments/fastdllm

# 优化
python scripts/opt1_cache_bench.py --model fastdllm --out results/optimization/opt1/fastdllm
python scripts/opt2_shape_bench.py --out results/optimization/opt2/shape
python scripts/opt3_adaptive_bench.py --model dream --out results/optimization/opt3/dream
python scripts/opt4_fused_bench.py --out results/optimization/opt4

# 端到端墙钟（P1-7 五档演进）
python -m scripts.ar016_batched_bench          # 批量图首证
python -m scripts.ar017_adaptive_bench         # 自适应容量
python -m scripts.ar018_publish_bench          # 发布压缩（合成最终结果）
python -m scripts.ar018_publish_bench --quick  # 冒烟

# 真实 checkpoint 端到端（AR019，P1-7 第五档；AR019 venv + 本地权重）
<venv目录>\Scripts\python -m scripts.ar019_real_e2e_bench \
    --model-path <本地权重目录> --reps 30 --warmup 8

# 发布路径 cache put 批量化 A/B（AR020；同 venv/权重，三层测量 + 正确性门）
<venv目录>\Scripts\python -m scripts.ar020_publish_put_bench \
    --model-path <本地权重目录> --reps 30 --warmup 8   # --quick 冒烟

# 生产生成回路三模式基准（AR021，P1-7 第七档；同 venv/权重）
<venv目录>\Scripts\python -m scripts.ar021_generation_loop_bench \
    --model-path <本地权重目录> --reps 30               # --quick 冒烟

# 四模式投机质量对齐基准（AR022，P1-7 第八档；同 venv/权重）
<venv目录>\Scripts\python -m scripts.ar022_speculative_quality_bench \
    --model-path <本地权重目录> --reps 30               # --quick 冒烟

# 跨族投机四模式基准（AR023，P1-7 第九档；LLaDA-8B-Base + Dream-v0-Base-7B
# 真实 checkpoint，~30GB 权重，探针自带 curl 下载清单 + SHA256 校验）
<venv目录>\Scripts\python -m scripts.ar023_family_probe \
    --models-root <权重根目录> --hf-home <HF缓存目录>   # 前置探针（必跑）
<venv目录>\Scripts\python -m scripts.ar023_cross_family_bench \
    --family llada --models-root <权重根目录> --reps 30  # 全程 ~4-5h（8B 单卡）
<venv目录>\Scripts\python -m scripts.ar023_cross_family_bench \
    --family dream --models-root <权重根目录> --reps 30  # 全程 ~6-8h

# 多 parent 批量折叠合成面基准（AR025，合成模型 CPU 即可、分钟级；
# --mode real 已随 T006 接线交付——真实面命令见上文第 5 节与 RERUN 步骤 16）
python -m scripts.ar025_multi_parent_bench

# 服务器一键正式跑执行器（AR027）：七面固定队列 + 六项硬 preflight +
# 冻结判定重算校验 + 结构化日志（results/optimization/formal_runs/<UTC>/）
python -m scripts.run_formal_queue --list            # 队列预览
python -m scripts.run_formal_queue --quick           # 七面冒烟（不写正式工件）
python -m scripts.run_formal_queue --models-root <权重根目录>  # 正式（预承诺单次）
```

完整报告：[`docs/DEEP_EXPERIMENT_REPORT.md`](docs/DEEP_EXPERIMENT_REPORT.md)、
[`docs/OPTIMIZATION_REPORT.md`](docs/OPTIMIZATION_REPORT.md)、
[`docs/OPTIMIZATION_GUIDE.md`](docs/OPTIMIZATION_GUIDE.md)（全部 29 个 AR 的
机制与证据链）、[`docs/ACADEMIC_REPORT.md`](docs/ACADEMIC_REPORT.md)。

---

## 特性一览

| 特性 | 说明 |
|------|------|
| **Branch Folding 引擎** | `ManualFoldedForward`（推荐路径）零侵入包装任意 Transformer 栈，逐 token 激活复用；`FoldedModel` 仍可用（已弃用） |
| **生产生成回路折叠**（AR021） | `FastDLLMSampler.sample(prompt_ids, folded_model=mff)`：sampler (T,prompt_len) 键控供给（掩码/pe/forced 对象钉定）+ 强制散度活跃块（parity 由构造 bit-exact，真实 checkpoint 实测 bit-exact）+ 形状键控 runner 池（`graph_runner_pool_size`，opt-in 默认 legacy）+ 批量投机块验证（`num_block_candidates`，opt-in）——AR015-020 批量验证栈的生产接线 |
| **贪心锚定候选**（AR022） | `candidate_include_greedy=True`：候选行 0 = draft argmax（top-k 有序集首位，确定性构建/零回读/零 generator），进入 min-prob 公平竞争（可落败，UT-2203 后验陷阱钉死）；N 总数不变、随机行 j=1..N-1 保持行号（与 greedy-off 构建逐位相同，画布一致区内）；`greedy_win_stats` 只读观测面 + `ActFoldConfig` YAML 消费面；默认 False 逐字节 AR021 末态，True+N=0 构造期 ValueError。实测：质量逐位对齐 baseline（gap=0.0、早停恢复）+ **7.02× vs baseline** |
| **批量投机块验证**（AR021） | `num_block_candidates=N`：top-k 起草（种子公式冻结）→ 单次 `[N,T]` 批量验证 → min-prob 选定提交；AR016 批量图 + AR017 自适应 + AR018/020 发布路径首次被生产回路行使；默认 0 关逐字节不变 |
| **position_embeddings 线程化**（AR019） | `position_embeddings=(cos, sin)` 新参贯穿四构建面（含 graph 烘焙与自适应重捕获），remote-code 模型（Fast-dLLM v2）可跑；`None` 时既有调用逐字节不变 |
| **批量分支验证**（AR015） | `branch_ids` 复数 kwarg：同父等长候选单次批量前向，其余形态自动串行回退 |
| **CUDA Graph 验证**（AR002/AR016/AR021） | `use_cuda_graph=True` 固定形状捕获 + 独立批量 runner + 重放前父缓存再校验；AR021 形状键控池（opt-in）让跨画布增长的生成回路持续捕获重放；变长负载静默 eager 回退 |
| **自适应发散容量**（AR017） | `graph_adaptive_capacity=True`：观测驱动重捕获状态机，容量跟随实测需求，失败退化永不劣于现状 |
| **压缩发布路径**（AR018） | 单次整块快照 + 批量稳定度归约（launch 3·L·N → L·N+2），发布内容逐位不变，零新增同步 |
| **批量 cache 写**（AR020） | 三 cache `put_batch`（冲刷语义，写入序恒=条目序）+ `BatchedPutCapable` 结构化协议 + 发布路径能力探测分流（主机调用 L·N+N → 1）；`foreach` 单点封装，旧 torch 逐对降级；无能力 cache 逐字节走原循环 |
| **真实 checkpoint 端到端**（AR019/AR021/AR022） | AR019：真实块验证三路径 2.44–2.62×（p50），逐位一致 + 完美块二分；AR021：真实 generate() 三模式，folded-serial parity bit-exact，folded-batched 2.38× vs baseline；AR022：四模式质量对齐——贪心锚定 posthoc 逐位==baseline、早停恢复，7.02× vs baseline |
| **前缀折叠**（AR003/AR007） | 变长生成步前缀对齐折叠；父比子长时截断前缀复用，门控作语义安全网 |
| **真接受语义**（AR004/AR006） | 同位置约定的接受率/log-prob 计量；`model_draft` 模式接入独立草稿模型（论文级指标） |
| **向量化激活缓存** | 连续缓冲 + `index_select`；put/get 快 12–183× / 30–429× |
| **层内 FFN 拆分** | attention 全序列 + FFN 仅 divergent 行；B×T ≥ 512 自动启用 |
| **自适应分位门控** | 按相似度排名（top-k）精确命中目标 stable ratio |
| **融合 Triton kernel** | gate+mask+count 单 kernel、融合 gather-select、融合 merge（逐位精确 + PyTorch 回退） |
| **逐层稳定度 profiler** | 记录真实逐层 stable ratio（批量路径经 `record_precomputed` 零同步入账） |
| **Diffusion 原生采样器** | LLaDA / Dream / Fast-dLLM 参考实现（对齐官方 recipe），跨步折叠链 + 掩码 token 统一解析 |
| **真实评测后端** | `lm-eval` / `evalplus` 集成；无 mock 回退 |
| **计算-带宽成本模型** | 从 FLOPs 与显存带宽双口径估计墙钟 |

---

## 安装

### 依赖

- Python 3.10+，PyTorch 2.0+，Hugging Face `transformers`
- CUDA GPU（可选；CPU 回退可用）
- `lm-eval` / `evalplus`（评测后端）

### 从源码安装

```bash
pip install -r requirements.txt        # 运行时依赖
pip install -r requirements-bench.txt  # 评测后端
pip install -r requirements-dev.txt    # 开发工具
pip install -e .                       # 可编辑安装
```

> remote-code diffusion LM（Fast-dLLM v2）钉住 `transformers==4.53.1`——主环境
> 不可降级时按 [`docs/RERUN_CHECKLIST.md`](docs/RERUN_CHECKLIST.md) 第 9 步建
> AR019 venv（`--system-site-packages` 继承 torch）。

---

## 综合框架优越性总表（Framework-vs-Baseline Ladder）

> 每一档创新都以**官方家族 baseline**（同一模型 / prompt / seed / 环境）为固定参照，
> 从对应工件 JSON 回填（禁止手输）；「状态」列区分**已交付**与**待实机**——
> 待实机档的数字不得当作已交付结论引用。

| # | 创新档 | 交付配置 vs 官方 baseline | 质量面 | p50 加速 | 证据工件 | 状态 |
|---|---|---|---|---|---|---|
| 1 | AR018 发布成本压缩（批量稳定度归约 + 快照发布，P1-7 第四档） | 合成验证回路，4 形状点（T=512, N∈{4,8}） | H2/H3/H4 supported；H1 inconclusive（(256,8,4) 欠功效先例，诚实留档） | adaptive vs baseline **2.40/2.67/2.66/2.95×** | `ar018_publish_bench.json` | 已归档 |
| 2 | AR019 真实 checkpoint 端到端（Fast-dLLM-v2-1.5B） | 真实块验证三路径 | 位级对拍 + CUPTI 注记 | **1.35–1.97×** | `ar019_real_e2e_bench.json` | 已归档 |
| 3 | AR020 批量 cache put（发布路径写放大消除） | 真实 checkpoint A/B | state-equivalence 面逐项 PASS | **1.77–1.97×** | `ar020_publish_put_bench.json` | 已归档 |
| 4 | AR021 生产生成回路折叠（folded supply + forced-divergent + 池化） | 真实 `generate()` 底盘 | canary 零降级 + 账本 | batched **2.38×** | `ar021_generation_loop_bench.json` | 已归档 |
| 5 | AR022 贪心锚定投机候选 | 四模式质量对齐 | **质量差距归零**（posthoc/length/block-exact 全恢复）+ 复现自检 PASS | **7.02×** vs baseline | `ar022_speculative_quality_bench.json` | 已归档 |
| 6 | AR023 跨族投机（LLaDA-8B-Base / Dream-v0-Base-7B） | 真实 8B/7B 四模式 | **诚实负结果**：llada formal h1 refuted（投机质量差距真实）、h4 supported；quick 面与探针结论工件齐全 | llada formal h2 **1.49×** | `ar023_llada_bench.json`（formal）、`ar023_*_bench_quick.json`、`ar023_family_probe.json` | 正式面部分交付（dream formal 未跑）；归档进行中 |
| 7 | AR024 提交粒度子块窗口（子块分解 + 渐进回填 + EOS 感知延续） | LLaDA 8B 投机链 | UT 23 例全绿（位级锚定/后验陷阱/EOS 延续） | 待实机 | `specs/changes/AR024-commit-granularity/` | **待实机**（GPU 窗口） |
| 8 | AR025 多 parent 批量折叠 + 双链 lockstep 并行投机 | 合成面 16 形状点；真实面接线（LLaDA/Dream，判定式冻结） | S1 合并奇偶 16/16 / S2 写账本 4<40–160 严格 / S3 0.9–0.975（DD8 预算 erratum 留档）；UT 15 例 + 哨兵 96 绿 | 待真实面（H1 逐位+归因 / H2 p50 / 交付线 ≥10% / H4 账本） | `ar025_multi_parent_bench.json` | 已归档（`specs/archive/AR025-multi-parent-batch-folding/`，ST Go）；正式 reps=30 待高速算力机（RERUN 步骤 16，本机 16GB 限制如实注记） |
| 9 | AR026 真实 checkpoint 端到端图路径（生产捕获 + 四模式图基准） | LLaDA-8B / Dream-7B 生产捕获 | **T006 首验 PASS**：两族生产捕获 `graph_captures=2`、零失败诊断、零降级回退（AR023 裸捕获断言/广播断裂未复现）；四模式奇偶判定冻结 | 待正式面 | `ar026_probe_graph_faces.json` | 首验 PASS；正式面待跑 |

**读表口径**：第 1–5 档为已归档交付；第 6 档含诚实负结果（正式面 h1 refuted 与 quick 面并存，处置归 AR023 结档）；第 7–9 档的实现/合同/首验已交付，正式数字须待实机工件回填。所有数字可由所列工件重算（时间口径 #39：perf_counter + synchronize，warmup 丢弃，per-rep round-robin；跨机复现见 `docs/RERUN_CHECKLIST.md` §7）。

## 快速上手

### 1. 运行 demo

```bash
# 合成演示模型——无需下载，任意平台可跑
python demo.py

# 标准因果 LM（GPT-2 / LLaMA / Qwen / Mistral ...）
python demo.py --model gpt2 --model-family causal_lm

# Diffusion 原生生成 + Branch Folding
python demo.py \
    --model <llada-checkpoint> \
    --model-family llada \
    --prompt "The future of artificial intelligence is" \
    --num-steps 128 \
    --max-new-tokens 32
```

合成模型期望输出：

```text
 Total FLOPs reduction: 85.5%   # 修正：输入 embedding 是查表，只计 LM head
 Output equivalence (MSE): 2.35e-03  [HIGH]
 Estimated stable token ratio: 93.75%
```

### 2. 包装模型并验证子分支（推荐路径）

```python
import torch
from actfold.core import ActivationCache, SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward

cache = ActivationCache(max_entries_per_layer=1024, device="cuda")
gate = SimilarityGate(tau=0.95, metric="cosine")

# 零侵入包装：不修改原模型（区别于已弃用的 FoldedModel）
mff = ManualFoldedForward(raw_model, cache=cache, gate=gate, split_layers=True)

# 父分支前向填充缓存
parent_logits = mff(parent_tokens, branch_id="parent")

# 子分支验证：stable 位置复用父激活
child_logits = mff(child_tokens, branch_id="child", parent_branch_id="parent")
```

### 3. 多分支批量 + CUDA Graph + 自适应容量 + 形状键控池（加速主路径）

```python
from actfold.core.adaptive_capacity import AdaptiveCapacityConfig

mff = ManualFoldedForward(
    raw_model,
    cache=cache,
    gate=gate,
    split_layers=True,
    use_cuda_graph=True,              # 固定形状 [N, T] 图捕获
    graph_capacity_ratio=0.5,         # 初始发散容量
    graph_adaptive_capacity=True,     # AR017：容量跟随实测需求
    graph_adaptive_config=AdaptiveCapacityConfig(),
    graph_runner_pool_size=8,         # AR021：形状键控 runner 池（opt-in；
                                      # 0=legacy 单 runner 逐字节；>0=LRU 上限，
                                      # 跨画布增长每形状窗捕获/重放）
)

# 同父、等长、parent-batch==1 的候选集合：单次批量前向 + 图重放
logits = mff(
    stacked_candidates,                      # [N, T]
    branch_id=None,
    branch_ids=["c0", "c1", "c2", "c3"],
    parent_branch_id="parent",
)

# remote-code 模型（Fast-dLLM v2 / Qwen 系）：层需要 rotary（AR019）——
# 供给 position_embeddings=(cos, sin)（模型自身 rotary_emb 产出；
# arange(T) 常量位置方案；重放期须为捕获时同一对象）
logits = mff(
    stacked_candidates,
    branch_id=None,
    branch_ids=["c0", "c1", "c2", "c3"],
    parent_branch_id="parent",
    attention_mask=block_causal_mask,        # 块因果掩码 (q//32) >= (kv//32)
    position_embeddings=(cos, sin),          # AR019：None 则不附加，现状不变
    forced_divergent=forced_mask,            # AR021：True 位强制重算（None 不附加）
)
```

> 发布自动批量化（AR020）：`ActivationCache` / `VectorizedActivationCache` /
> `ChunkedActivationCache` 均实现 `put_batch`，批量图发布经每次发布的能力
> 探测自动走单次 `put_batch`（主机调用 L·N+N → 1）——用户零配置；传给
> `ManualFoldedForward` 的 cache 不暴露 `put_batch` 时逐字节走原循环。

### 4. 生产生成回路折叠 + 贪心锚定批量投机（AR021/AR022，Fast-dLLM v2）

```python
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig

# 折叠接线：sampler 侧自动供给钉定的 mask/pe/forced 三件套
# （(T, prompt_len) 键控、跨调用复用）；活跃块强制散度——parity 由构造保证
config = FastDLLMSamplerConfig(
    num_block_candidates=0,            # 0=关（默认，逐字节现状）
    candidate_top_k=10,                # 起草 top-k 池（num_block_candidates>0 时生效）
    candidate_include_greedy=False,    # AR022 贪心锚定（默认 False；True 需 N>0）
)
sampler = FastDLLMSampler(model, config)
output = sampler.sample(prompt_ids, folded_model=mff)   # 要求 batch=1

# 质量对齐模式（AR022）：行 0 = draft argmax 进入 min-prob 公平竞争
config = FastDLLMSamplerConfig(
    num_block_candidates=4,            # 批量投机验证（opt-in）
    candidate_top_k=10,
    candidate_include_greedy=True,     # 贪心锚定：posthoc==baseline、早停恢复
)
sampler = FastDLLMSampler(model, config)
output = sampler.sample(prompt_ids, folded_model=mff)
print(sampler.greedy_win_stats)        # {"greedy_wins": ..., "speculative_calls": ...}
```

> 单发场景（单次 `generate()`）建议 `mff` 构造时 `use_cuda_graph=False`——
> 每形状窗仅 1 次批量调用时图捕获摊销为负（AR021 工件 A5 注记）；批量/
> 基准场景（warmup + 多次调用）摊销成立。质量注记：`num_block_candidates>0`
> 且 `candidate_include_greedy=False` 时投机提交非贪心，不保证与逐迭代配方
> exact-match（AR021 实测 exact-match 0.0、置信低于 baseline——速度-质量
> 权衡如实报告）；**开启 `candidate_include_greedy=True`（AR022）后本负载
> 实测质量逐位对齐 baseline（posthoc gap=0.0、解码长度/早停恢复）且更快
> （7.02× vs baseline / 2.95× vs 无锚定 batched）**。

### 5. 真实 checkpoint 基准（AR019 venv + 本地权重）

```bash
# 详参 docs/RERUN_CHECKLIST.md 第 9/10/11/12 步（权重 curl 下载 + venv 创建）
<venv目录>\Scripts\python -m scripts.ar019_real_e2e_bench \
    --model-path <本地权重目录> --reps 30 --warmup 8   # --quick 冒烟

# 发布窗口 A/B + 端到端四路径（AR020；同 venv/权重）
<venv目录>\Scripts\python -m scripts.ar020_publish_put_bench \
    --model-path <本地权重目录> --reps 30 --warmup 8   # --quick 冒烟

# 生产生成回路三模式基准（AR021；同 venv/权重，峰值 ~5.4GB）
<venv目录>\Scripts\python -m scripts.ar021_generation_loop_bench \
    --model-path <本地权重目录> --reps 30               # --quick 冒烟 ≤5min

# 四模式投机质量对齐基准（AR022；同 venv/权重，峰值 ~5.8GB）
<venv目录>\Scripts\python -m scripts.ar022_speculative_quality_bench \
    --model-path <本地权重目录> --reps 30               # --quick 冒烟 ≤5min

# 多 parent 批量折叠基准（AR025）——合成面 CPU 分钟级、无需 GPU/权重：
python -m scripts.ar025_multi_parent_bench --mode synth  # 8 形状点×双构造
python -m scripts.ar025_multi_parent_bench --mode synth --quick  # 单点冒烟
# 真实面（LLaDA/Dream 双链 lockstep A/B，reps=30 预承诺单次）——16GB 级
# 主机不可行（bf16 权重 15.8GB 触顶），按 RERUN 第 16 步在 ≥24GiB 主机执行：
# python -m scripts.ar025_multi_parent_bench --mode real --family llada \
#     --models-root <权重根> --probe-artifact results/optimization/ar023_family_probe.json --reps 30
```

### 6. 端到端折叠生成

```python
from actfold.speculative.folded_generation import folded_generate

result = folded_generate(
    adapter,
    prompt_ids,
    max_new_tokens=32,
    folded_model=adapter.folded_model,
)
print(result.tokens)          # [batch, prompt_len + max_new_tokens]
print(result.stable_ratio)    # 实测逐层平均 stable ratio
print(result.acceptance_rate) # 真接受率（AR004）
```

### 7. 运行评测

```bash
bash scripts/run_real_model_benchmark.sh
bash scripts/run_benchmarks.sh actfold/configs/real_model_example.yaml
```

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Diffusion LLM                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │      Draft Generator        │
        └──────────────┬──────────────┘
                       │
            ┌──────────▼──────────┐
            │   Parent Branch     │
            └──────────┬──────────┘
                      │ │ │ │  N 个候选
          ┌─────────▼──▼─▼──▼──────────┐
          │   ActFold 批量验证          │
          │  （AR015 批量 + AR016 图）  │
          │  ┌─────────────────────┐  │
          │  │   Similarity Gate   │  │
          │  └──────────┬──────────┘  │
          │    ┌────────┴────────┐    │
          │    ▼                 ▼    │
          │ Stable Tokens   Divergent │
          │    │               Tokens  │
          │    ▼                 ▼    │
          │ Activation     Full Layer │
          │   Cache        Recompute  │
          │    │                 │    │
          │    └────────┬────────┘    │
          │             ▼             │
          │      Merged Output        │
          │   （AR017 自适应容量）     │
          │   （AR018 压缩发布）       │
          │   （AR020 批量 put）       │
          └─────────────┬─────────────┘
                        │
              ┌──────────▼──────────┐
              │   Accepted Branch   │
              │（AR019 真实 checkpoint│
              │  pe 线程化 + AR021   │
              │  生产生成回路折叠接线：│
              │  强制散度活跃块 +     │
              │  形状键控 runner 池 + │
              │  AR022 贪心锚定候选） │
              └─────────────────────┘
```

### 模块地图

```
actfold/
├── models/         # Diffusion LLM 包装器与原生采样器（含 ManualFoldedForward、折叠供给）
├── core/           # 折叠引擎：cache/gate/folded layers/CUDA graph/自适应容量/批量写
├── profiler/       # 稳定度 profiler（批量路径零同步入账）
├── speculative/    # Draft 生成、验证引擎、折叠生成
├── eval/           # 评测：BenchmarkRunner / lm-eval / evalplus 适配器
├── utils/          # 配置、FLOPs 计量、成本模型、日志
└── configs/        # YAML 实验配置
```

---

## 进阶用法

### Diffusion 原生采样

`num_steps > 1` 时 `DiffusionLLM.generate()` 按模型家族分发参考采样器
（对齐官方 recipe：掩码调度、块解码、置信度重掩码、CFG、温度/top-p/top-k）：
**LLaDA**（右 padding 画布 + 块解码）、**Dream**（MaskGIT 式迭代 +
`maskgit_plus`/`topk_margin`/`entropy` 置信规则）、**Fast-dLLM**
（小块阈值解掩码 + 自回归块扩展 + 停止 token 早停）。

> 采样器为高质量参考实现；发表数字请对具体 checkpoint 用官方实现复核。
> Fast-dLLM 折叠路径（AR021）：传 `folded_model=MFF` 启用链折叠（强制散度
> 活跃块 + 形状键控池），`num_block_candidates>0` 启用批量投机验证，
> `candidate_include_greedy=True`（AR022）启用贪心锚定质量对齐——三者
> 默认关（raw 配方逐字节不变）。

### 配置驱动评测

```yaml
# actfold/configs/real_model_example.yaml
model_name_or_path: "gpt2"
model_family: "causal_lm"
use_real_eval: true
eval_backend: "auto"

# ActFold 开关
use_stability_profiler: true
use_vectorized_cache: true
use_split_layers: false       # attention 全序列 + FFN 仅 divergent
split_min_tokens: 512
use_cuda_graph: true          # 固定形状图捕获
graph_capacity_ratio: 0.5
graph_adaptive_capacity: true # AR017 自适应容量
graph_runner_pool_size: 8     # AR021 形状键控 runner 池（0=legacy）
candidate_include_greedy: false  # AR022 贪心锚定（YAML 消费面；True 需批量投机开启）
use_cost_model: true
use_folded_generation: true
```

### 消融

```bash
bash scripts/run_ablation.sh actfold/configs/real_model_example.yaml
bash scripts/run_ablation.sh --synthetic
```

支持：阈值敏感性（τ ∈ {0.90, 0.95, 0.99}）、逐层折叠策略、缓存预算、
缓存实现对比（per-token / chunked / vectorized）、FFN 拆分形状扫描、
自适应门控前沿、融合 kernel 对拍。

---

## 质量保障

- **black**（line-length=100）/ **isort** / **pyflakes** 零告警
- **mypy**（`--ignore-missing-imports`）：`actfold` 68 文件 0 错 +
  `scripts` 26 文件 0 错（`--explicit-package-bases`，命名空间包口径；
  维护轮已清 29 项既有债务）
- **pytest** 快速套件 **1179 passed / 4 skipped / 3 deselected**（零失败）
  （AR027 轮次实测 ~85s；含 AR025 sample_joint 15 例、AR024/AR026 并行轮
  新增、AR027 合同测试 66 例（UT-2701~2764：队列冻结/preflight/执行/校验/
  覆盖审计/异常面）；lm-eval×2 bench-backend 用例已随 `pytest addopts`
  默认 deselect（3 deselected，与代码无关）；skip = CUPTI + evalplus-unix
  平台面等；BS-007 图重放性能判定在 Windows WDDM 小内核主机上按实测数字
  诚实 skip——同一份 AR002 代码在该类主机上图重放慢于 eager 已经版本二分
  确证为平台特性，历史工件保持证据效力）
- 工具链钉版：`requirements-dev.txt`（black==23.12.1 / isort==5.13.2 /
  pyflakes==4.0.0 / mypy==2.4.0），`requirements.txt` 限 `transformers<5`
  （remote-code 5.x 不兼容）；换机后按同版本安装即可复现质量门
- 每个机制的验收面：Red/Green 分离 TDD + **mutation 测试**（删机制必红，
  SHA256 逐字节还原）+ 跨工件墙钟基准（声明与判定分离，2σ 判据，
  inconclusive 诚实照收）+ 严格复现自检（AR022：batched 面对 AR021 工件
  先验逐位核对，失配全 verdict blocked）

```bash
python -m black --check actfold tests demo.py scripts
python -m pyflakes actfold tests demo.py scripts
python -m mypy actfold --ignore-missing-imports
python -m pytest tests/ -q -m "not slow"
python demo.py
```

真实后端用例标记 `@pytest.mark.slow`：`python -m pytest tests/ -q -m slow`。

---

## 文档

- [`docs/ALGORITHM.md`](docs/ALGORITHM.md) — Branch Folding 形式化描述
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — 复现工作流
- [`docs/DEEP_EXPERIMENT_REPORT.md`](docs/DEEP_EXPERIMENT_REPORT.md) — 两阶段深度实验报告（诊断 + 优化，真实模型实测）
- [`docs/OPTIMIZATION_GUIDE.md`](docs/OPTIMIZATION_GUIDE.md) — **29 个 AR 的完整机制与证据链**（每项创新动机 → 机理 → mutation → 基准）
- [`docs/OPTIMIZATION_REPORT.md`](docs/OPTIMIZATION_REPORT.md) — 优化阶段报告
- [`docs/ACADEMIC_REPORT.md`](docs/ACADEMIC_REPORT.md) — 学术版报告
- [`docs/RERUN_CHECKLIST.md`](docs/RERUN_CHECKLIST.md) — 复现清单（环境/venv/权重/命令；第 13–16 步 = AR023 探针与跨族基准 / AR025 合成面 / AR026 图路径 / AR025 真实面双链基准；§7 = 高算力服务器复现路径）
- [`AGENTS.md`](AGENTS.md) — 贡献者与 AI 代理的约定和陷阱清单（含 Reproduction Guide Rule 与 #46 真实 checkpoint 运维法规）
- [`CHANGELOG.md`](CHANGELOG.md) — 发布历史（AR001–AR026）

---

## 当前限制与路线图

1. **P1-7 端到端证据链：AR018–AR022 已归档，AR023–AR026 逐档推进中**。
   已归档档见[综合框架优越性总表](#综合框架优越性总表framework-vs-baseline-ladder)：
   批量前向（AR015）→ 批量图捕获（AR016）→ 自适应容量（AR017）→ 发布压缩
   （AR018）→ 真实 checkpoint 复测（AR019）→ 发布 put 批量化（AR020）→
   生产生成回路折叠接线（AR021）→ **贪心锚定质量对齐（AR022）**：真实
   checkpoint 端到端 **2.40–2.67×**（AR020 四路径）、发布窗口
   **1.77–1.97×**、主机调用账本 **L·N+N → 1**、合成基准 2.40–2.95×；
   AR021 把整栈接入生产 `generate()`（folded-serial parity bit-exact +
   folded-batched 2.38×）；AR022 把批量投机的质量面对齐 baseline——
   **posthoc 逐位==baseline（gap=0.0）+ 早停恢复（34/66）+ 7.02× vs
   baseline（H1-H4 全 supported，复现自检 strict PASS）**。AR023 跨族投机
   已交付实现与 quick/探针工件，llada 正式面为**诚实负结果**（h1 refuted：
   投机质量差距在 T=128 下真实存在；h2 1.49×、h4 supported；dream formal
    待跑）；AR024（提交粒度窗口）、AR026（真实 checkpoint 图路径，T006
    首验两族生产捕获均 PASS）的正式数字待实机工件
    回填；AR025（多 parent 批量折叠+双链 lockstep 并行投机）**已归档**
    （`specs/archive/AR025-multi-parent-batch-folding/`，ST Go：合成面
    16/16 supported + 真实面接线与判定式冻结交付 + 15 例 UT + 12 项审查
    修复闭环），正式 reps=30 按 RERUN 步骤 16 待高速算力机执行回填（本机
    16GB WDDM 实测 bf16 权重 15.8GB+折叠缓存触顶硬崩、int8 逃生门在驱动
    坏态下量化停滞，均如实留档）。保留判定：AR019 (256,4) 首配置 H1 因
    WDDM 热身方差欠功效
   （方向压倒性）、AR021 H2 inconclusive（见第 4 条）——均按预承诺单次
   运行纪律如实记录。
2. **真实模型覆盖面**：LLaDA-8B / Dream-7B bf16 权重 14–16GB，16GB 显存叠加
   图缓冲为紧约束（AR026 生产捕获首验在此跑通，但四模式正式面建议 ≥24GiB）；
   真实模型两阶段实验（TFLOPs 削减与逐层结论）已在
   [最新实验结果](#最新实验结果)覆盖。AR023 已交付 LLaDA/Dream 两族的
   speculative 接线（共享 `SpeculativeCore` + 族适配器）；AR026 已把探针
   graph 面生产化并在两族真实 checkpoint 上首验通过（裸捕获代理的失败签名
   不再出现在生产路径）。
 3. **批量路径要求等长、parent-batch==1**。变长步仍是串行回退（已在前缀
    折叠下获益，AR003/AR007）；**多父批量已由 AR025 交付**（`parent_branch_ids`
    per-row 父装配 + `sample_joint` 双链 lockstep，合成面 16/16 supported，
    真实面待实机）；多祖先激活树（>2 链、树状分支结构）是路线图主体
    （AR016 遗留）。
4. **串行折叠的小形状开销（AR021 H2 inconclusive，如实入限制）**。
   num_tokens=128（4 个形状窗）下 folded-serial 比 baseline 慢 107.5ms
   （Δ=−107.5181ms，未越 2σ=265.3231ms 预注册噪声界；两 prompt 方向
   一致）——串行折叠的供给/发布开销（sidecar 代 899 发布 ops）超过前缀
   复用收益；批量模式（folded-batched）不受此限（2.47× supported）。
   更长生成（更多形状窗、更大 T）下串行折叠的相对收益预期改善——待后续
   按同法实测，不预设。
5. **贪心锚定的实证边界（AR022 收口后的剩余口径）**。质量对齐结论
   （posthoc gap=0.0、早停恢复、exact 1.0）与 greedy_win_rate 1.0（2/2）
   均来自单次预承诺运行 × P=2 prompt 的同形状 A/B——**贪心行可落败是机制
   属性**（UT-2203 后验陷阱构造级钉死），win_rate 在更大 prompt 集/更长
   生成/其他 checkpoint 上的分布未测，不预设；未锚定的 folded-batched
   质量面（exact 0.0、置信低于 baseline）仍是如实报告的现状——接受策略
   族谱（多准则/温度化 min-prob/贪心变体）留后续 AR。
6. **生产单发场景图摊销为负（AR021 A5 注记）**。block_size==small_block_size
   下单次 generate 每形状窗仅 1 次 `[N,T]` 批量调用——捕获（≈2-3 次前向
   成本）服务 1 次重放，摊销为负；单发场景建议 `use_cuda_graph=False`。
   基准/批量场景（warmup 预捕获 + 多 rep 重放）摊销成立（AR021/AR022
   基准协议）。
7. **小几何首配置 WDDM 方差**。锁时钟需管理员权限；(256,4) 级 2σ 检验
   欠功效先例在案（AR017/18/19 三代同类，效应 delta 稳定方向正；AR020 的
   发布窗口/端到端判定在 (256,4) 均已 supported）。发布窗口的宿主 launch
   主体已由 AR020 消除；剩余相位为快照 clone 带宽与 profiler 记录。
8. **采样质量是模型敏感的**。Fast-dLLM 跨步折叠 token 一致率 1.000；
   LLaDA/Dream 在 stable≈0.99 时为 0.71–0.73（提高 τ 或用自适应目标缓解）。
9. **专用 draft 架构（Medusa/Eagle）为 tier 2**。`model_draft` 已支持任意
   独立草稿模型的论文级接受率计量（AR006）。
10. **逐模型 YAML 是模板**——请填入实际 Hugging Face 标识或本地 checkpoint 路径。

---

## 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `RuntimeError: A real tokenizer is required...` | 评测路径缺 tokenizer | 传入带 tokenizer 的模型或用 `--synthetic` |
| `TypeError: Can't instantiate abstract class ... with abstract method embed` | 自定义 `DiffusionLLM` 缺 `embed()` | 实现 `embed(tokens)` 返回 `[B, T, H]` |
| remote 模型前向 `TypeError: cannot unpack non-iterable NoneType` | remote attention 层需要 rotary（`cos, sin = position_embeddings`）而调用未供给 | 经 `ManualFoldedForward(..., position_embeddings=(cos, sin))` 供给（AR019；模型自身 rotary_emb 产出，dtype 跟随模型） |
| `ValueError: ... supports batch 1 only` | 折叠采样路径收到 B>1（`[T,T]` 折叠掩码不可表达逐行 pad） | 折叠路径逐 prompt 调用（AR021）；batch 需求走 raw 路径 |
| `ValueError: num_block_candidates > 0 requires a folded_model...` | 批量投机模式缺 `branch_ids` 能力的 folded_model | 传入 `ManualFoldedForward`（构造期显式报错，不静默降级） |
| `ValueError: candidate_include_greedy=True requires num_block_candidates > 0` | 贪心锚定仅定义于投机路径（AR022） | 同时设置 `num_block_candidates>0`（构造期显式报错，不静默降级） |
| CUDA 上未用 Triton kernel | 未装 `triton` / 无 C 编译器（bf16 键记录） / hidden dim 不被 128 整除 | Linux/WSL 安装 `triton>=2.0`；PyTorch 回退恒正确（AR014 键记录） |
| `evalplus` 在 Windows 失败 | 沙箱依赖 Unix `resource` 模块 | WSL 内运行 evalplus，或 Windows 用 lm-eval |
| 慢测试超时 | 真实后端需加载模型与数据集 | 快速套件 `-m "not slow"`，慢测试单独跑 |
| CUDA Graph 不生效 | 形状与捕获不一致 / attention mask 或 position_embeddings 对象不同 / forced 身份失配（None/not-None）/ 非 cosine `SimilarityGate` / 挂了 `FoldingScheduler` / 池未开（画布增长后单 runner warn-once 不再捕获） | 图路径前置条件见 AGENTS #37/#43（AR019 起 pe 亦须同对象；AR021 起 forced 须同 None/not-None 身份）；跨画布增长开 `graph_runner_pool_size>0`；变长步会静默 eager 回退（前缀折叠仍生效） |

---

## 贡献

欢迎贡献！请先阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)（代码风格、测试、PR 流程）。

---

## 引用

如在你的研究中使用 ActFold，请引用：

```bibtex
@software{actfold2025,
  title = {ActFold: Cross-Branch Activation Reuse and Branch Folding},
  author = {ShaneLiu04},
  year = {2025},
  url = {https://github.com/ShaneLiu04/ActFold},
}
```

---

## 许可证

ActFold 以 [MIT License](LICENSE) 发布。

---

> **声明**：本研究代码库。生产部署需要额外的优化、测试和目标推理框架集成。
