# ActFold：扩散语言模型推测解码中跨分支激活复用的可实现性研究

**作者**：https://github.com/ShaneLiu04
**日期**：2026 年 10 月
**关键词**：扩散语言模型；推测解码；激活复用；推理加速；缓存系统

---

## 摘要

扩散语言模型以迭代掩码去噪生成文本，推理成本随去噪步数线性增长；推测解码在验证阶段并行展开多个候选分支，而这些分支通常只与父序列相差少数 token。逐分支完整前向造成大量冗余，跨分支激活复用因此具有吸引力。我们系统研究了这一复用机制的可实现性：逐层、逐 token 比较父子隐藏状态，稳定 token 复用父分支缓存，发散 token 在完整自注意力上下文中重算后合并。在 Fast-dLLM-v2-1.5B、LLaDA-8B-Instruct 与 Dream-7B-Instruct 上的受控实验揭示了三个层面的问题。算法层面，复用是精确的：自折叠与强制全发散均与基线逐位一致；跨去噪步稳定率达 0.94–0.99。系统层面，原实现的复用路径受内存与同步开销支配，单层缓存读取（0.65–0.70 ms）超过原始层计算（0.53–0.61 ms），且"任一发散即整层重算"使算力节省仅在整层全稳定时兑现，成本模型与实测相差约四个数量级。机制层面，注意力的上下文耦合与 FFN 的逐 token 可分离性、缓存布局、复用预算的分配方式共同决定了稳定性何时转化为时间。据此我们提出四项机制：连续缓冲的向量化激活缓存、注意力全量而 FFN 仅计算发散行的层内拆分、按相似度秩精确分配稳定集的自适应门控、以及融合 gather 与 select 的 CUDA 核。全稳定折叠前向由 51–53 ms 降至 6.4–6.8 ms，较不折叠基线加速 2.3–2.7 倍；FFN 拆分在 batch×seq≥512 时取得 1.1–1.93 倍的逐层收益，并把 seq=512 的折叠路径缩短 8–18%；自适应门控在保真度与复用前沿上优于固定阈值；融合核在长序列达 3.43 倍且数值逐位一致。部分稳定场景的折叠路径在本文测试规模下仍慢于不折叠基线，其剩余开销来自逐层固定的门控、缓存与调度成本。三个模型族的端到端运行另需修复十类工程缺陷，回归测试 204 项通过。

**Abstract.** Diffusion language models generate text by iterative masked denoising, and speculative decoding verifies multiple candidate branches that differ from their parent in only a few tokens. Full per-branch recomputation is redundant, which motivates cross-branch activation reuse: token-wise parent/child hidden states are compared at every layer, stable tokens reuse cached parent activations, and divergent tokens are recomputed under full self-attention context and merged. We study when this reuse becomes real wall-clock acceleration. Controlled experiments on Fast-dLLM-v2-1.5B, LLaDA-8B-Instruct, and Dream-7B-Instruct expose three layers of the problem. Algorithmically, reuse is exact: self-fold and forced all-divergent passes match the baseline bit-for-bit, and cross-step stable ratios reach 0.94–0.99. Systematically, the original reuse path is dominated by memory and synchronization costs—per-layer cache reads (0.65–0.70 ms) exceed raw layer compute (0.53–0.61 ms), and an all-or-nothing recompute rule defers any compute saving until a layer is entirely stable. Mechanistically, the context-coupled attention, the token-wise FFN, the cache layout, and the reuse-budget allocation jointly determine whether stability converts into time. We contribute four mechanisms: a contiguous-buffer vectorized cache, an intra-layer split that keeps attention full while computing the FFN only on divergent rows, an adaptive gate that selects the stable set by similarity rank, and a fused gather-select CUDA kernel. The fully stable folded forward drops from 51–53 ms to 6.4–6.8 ms, 2.3–2.7× faster than recomputation; the FFN split yields 1.1–1.93× per-layer gains for batch×seq≥512 and shortens seq=512 folded paths by 8–18%; the adaptive gate dominates fixed thresholds on the fidelity–reuse frontier; the fused kernel reaches 3.43× on long sequences with bit-exact output. Partially stable folded paths remain slower than the no-folding baseline at the studied scale, with the residual cost originating in per-layer gating, cache, and dispatch overhead. Running three real model families end-to-end additionally required ten engineering fixes, validated by 204 regression tests.

---

## 1 引言

扩散语言模型（dLLM）把文本生成建模为迭代去噪：从掩码序列出发，在数十至上百步中逐步确定 token。LLaDA [1]、Dream [2] 与 Fast-dLLM v2 [3] 等模型在通用、数学与代码任务上展现出与同规模自回归模型可比的生成能力，代价是每步一次全序列前向。推测解码 [4,5] 用并行验证替代逐 token 采样，在自回归模型中已被广泛采用；在 dLLM 中，同一条父序列可以展开多个候选分支，每个分支只改动少数 token，但标准实现仍为每个分支重复完整的前向计算。

跨分支激活复用因此成为一个自然的选择：若某 token 在父子分支中的隐藏状态相似，就复用父分支的激活。然而"可复用"并不等于"更快"。我们的诊断实验显示，隐藏状态的稳定性极高（单 token 扰动下逐层稳定率 0.94–0.98，跨去噪步 0.98–0.99），但按原实现得到的墙钟延迟在阈值区间内恒定，全稳定路径反而比不折叠基线慢约三倍。进一步的开销分解把原因定位到系统层：逐 token 组织的缓存读取本身超过一层 Transformer 的计算，且"任一发散 token 即整层重算"的实现语义使计算节省只在整层全稳定时才发生。复用机制没有失败在算法上，而是失败在数据布局、调度语义与逐层固定开销上。

这些观察把研究问题收敛为一个可实现性问题：什么样的机制能让高隐藏状态稳定性转化为真实的墙钟加速，其适用边界在哪里。我们从三个方向回答。第一，注意力的计算与上下文耦合，但 FFN 逐 token 独立，因此复用设计应作用于可分离的计算单元。第二，复用路径的成本由数据移动与同步主导，缓存布局与算子融合决定稳定性能否变成时间。第三，跨层与跨模型的相似度尺度不同，复用预算应按相对秩分配而非绝对阈值。

围绕这一认识，我们实现了四项机制，并逐项验证其对延迟与保真度的影响：

- **向量化激活缓存**：以连续缓冲与批量索引替代逐 token 字典，缓存读写加速数十至数百倍，全稳定折叠前向较不折叠基线加速 2.3–2.7 倍。
- **层内 FFN 拆分**：注意力保持完整序列计算，FFN 链仅处理发散行。数值与完整重算一致，batch×seq≥512 时逐层加速 1.1–1.93 倍。
- **自适应秩门控**：每次调用按相似度排名精确选出目标比例的稳定 token，在保真度与复用前沿上优于固定阈值，且不需要按模型调参。
- **融合 gather-select 核**：把稳定行的父缓存读取与发散行的拷贝合并为单次内存遍历，长序列下达 3.43 倍，数值逐位一致。

实验覆盖三个真实 dLLM 族、两个 GPU 平台（RTX 6000D 与 RTX PRO 6000 Blackwell）、诊断与优化两阶段的 47 个数据文件与 15 张图。第 3 节形式化复用机制与可实现性差距；第 4 节给出四项机制的设计；第 6 节按证据链给出结果，并在结尾界定适用边界。

---

## 2 背景与相关工作

### 2.1 掩码扩散语言模型

掩码扩散语言模型在序列上定义前向掩码过程与反向去噪过程，推理时每步预测所有掩码位置的分布并确定其中一部分。LLaDA [1] 采用块级解码与低置信度重掩码，Dream [2] 采用 MaskGIT [6] 式迭代解码与置信度规则，Fast-dLLM v2 [3] 采用块级掩码解码与小阈值解掩码。三者共享同一成本结构：每步一次全序列前向，步数决定总成本；验证阶段的多分支进一步成倍放大这一成本。本文的工作负载正来自这一阶段。

### 2.2 推测解码与多分支验证

自回归推测解码以小型草案模型起草、大型模型并行验证 [4,5]；Medusa [7] 与 EAGLE [8] 通过多解码头或特征预测提高草案质量。dLLM 的去噪过程天然提供并行候选：不同置信度假设、不同掩码调度、不同草案 token 都可以构成分支。分支之间高度相似，这既是加速机会，也是本文研究的冗余来源。与上述工作不同，我们不改进草案质量，而是消除验证阶段各分支间的重复计算。

### 2.3 跨层与跨分支的激活复用

激活复用已有多种形式。DeepCache [9] 在扩散图像模型中复用深层特征跳过部分计算；PagedAttention [10] 通过分页管理 KV 缓存降低内存碎片与访问成本。本文复用的对象是跨分支、逐层的 FFN 输出，其正确性依赖两个条件：发散 token 必须获得完整的自注意力上下文，稳定 token 的输出必须与父分支一致到可接受误差。前一条件决定了注意力的不可分割性，后一条件由相似度门控与数值精度共同约束。

---

## 3 问题形式化与核心问题

### 3.1 Branch Folding

设模型有 L 层。在去噪步 s，父分支与子分支在第 l 层的隐藏状态分别为 h_p[l,t] 与 h_c[l,t]，t 为 token 位置。逐 token 相似度与稳定性为

```
sim(l,t) = cos(h_c[l,t], h_p[l,t]) ∈ [-1,1]        (1)
stable(l,t) = 1  当且仅当 sim(l,t) > τ              (2)
```

折叠后的层输出为

```
h_out[l,t] = cache_ffn_p[l,t]                    若 stable(l,t)      (3a)
             LayerFFN_c[l,t]（完整上下文）         否则                (3b)
```

式 (3b) 要求发散 token 使用完整子序列重算，以保持自注意力上下文与基线一致。稳定率 R 定义为 stable 的 token 比例，理论 FLOPs 缩减近似 R。

### 3.2 可实现性鸿沟

理论缩减不会自动变成墙钟收益，原因有二。其一是计算语义：原实现在存在任何发散 token 时执行完整层前向再按 token 合并，因此 R<1 时实际计算量仍接近整层；只有 R=1 的快速路径真正跳过计算。其二是数据路径：逐 token 组织的缓存以 Python 字典与逐 token 张量操作实现，其固定成本不随稳定率下降。两者叠加的结果是，稳定性最高的区间反而最慢，因为快速路径要支付全部缓存开销却不节省计算。

### 3.3 核心问题与优化目标

上述鸿沟可归结为一个问题：在何种机制与规模条件下，跨分支激活复用的稳定性能够转化为墙钟加速。给定质量约束（相对 logit MSE 或 top-1 一致率），目标为

```
min Latency(FoldedForward)  s.t. Fidelity ≥ δ       (4)
```

决策变量包括缓存实现、FFN 拆分、复用预算（由阈值或目标稳定率控制）、以及合并算子的实现。第 4 节按"计算可分离性、数据移动、预算分配"三条线索给出机制。

---

## 4 方法

### 4.1 折叠引擎总览

ActFold 分为模型适配、折叠引擎、验证与采样、评测与剖析四层。模型适配提供统一的 dLLM 接口与三个模型族封装；折叠引擎包含缓存、门控、调度与折叠层；验证与采样层提供验证引擎与跨步采样；评测层接入真实基准与逐层剖析。本文的改动集中在折叠引擎，对外接口保持不变。折叠包装器自动发现常见 Transformer 层堆栈并逐层替换，缓存与门控通过配置切换实现。

### 4.2 向量化激活缓存

原缓存以 (branch, layer, token, step) 为键，值为逐 token 张量。向量化缓存为每个 (branch, step, layer, 激活名) 维护连续缓冲 B ∈ R^{cap×B×H} 与写入计数。写入时一次切片拷贝完成 T 个 token 的存储；当序列超过容量时按行号取模形成环形布局，保留最近 cap 个 token。读取分两种路径：全命中时返回转置视图，零拷贝；部分命中时以"克隆加逐 batch 掩码置零"完成，避免数据相关的索引构造与多次内存往返。该实现与旧缓存 API 完全兼容，并在容量语义上保持"保留最新 token、缺失位置视为发散"的一致性。

### 4.3 层内 FFN 拆分

式 (3b) 的完整上下文要求注意力在全序列上计算，但 FFN 的每一行相互独立。我们在 FFN 链的两端模块上安装临时前向钩子：前钩子把输入按发散掩码切到 K 行，后钩子把输出散射回全长；注意力、残差与归一化仍由原层执行。由于切片与散射保持行独立，发散位置的数值与完整重算一致。切分与散射需要一次数据相关的 nonzero 同步，成本固定；当 batch×seq 较小时该成本超过 FFN 节省，因此拆分仅在 batch×seq 达到阈值时启用，检测不到可拆分链时自动回退整层重算。

### 4.4 自适应秩门控

固定阈值在实验中呈"平台加断崖"的形态，且同一阈值在不同模型上含义不同。自适应门控不设阈值，而是每次调用按相似度排名选取 k = round(r×B·T) 个最相似的 token 作为稳定集，其中 r 为目标稳定率。该规则精确命中目标比例，自动适应模型与层的相似度尺度，并把复用预算集中在最相似的 token 上。与调度器不同，秩门控不依赖逐层偏置参数。

### 4.5 融合 gather-select 核

稳定 token 的合并需要按行号读取父缓冲，发散 token 直接采用子张量。原路径由 index_select 与 where 两个算子组成，产生一次中间 gather 与一次独立遍历。融合核为每个 token 行启动一个程序：读取稳定标志，解析源行指针，单遍拷贝 H 维。核在 CUDA 上启用，其余情况回退 PyTorch 路径；两者逐位一致。

### 4.6 使三个模型族可运行的工程修复

在真实模型上端到端运行需要解决十类缺陷，其中影响正确性的修复包括：LLaDA 块返回 (hidden, cache) 元组而折叠层只返回张量导致解包失败；bf16 余弦相似度可超过 1.0，使 τ=1.0 时出现假稳定，裁剪到 [-1,1] 后边界严格；分块缓存的 get 返回形状与空缓存语义不符合 API 合约；扩散采样器从不传递父分支标识，使跨步折叠从未生效；Triton 3.4 拒绝 kernel 内的类型注解导致编译崩溃。完整清单见表 14。

---

## 5 实验设置

实验在两个平台完成：诊断阶段使用 NVIDIA RTX 6000D（84 GB），优化阶段使用 NVIDIA RTX PRO 6000 Blackwell（96 GB）；CPU 为 208 核、内存 1 TB。软件环境为 PyTorch 2.8.0+cu128、Triton 3.4、Transformers 4.53.1（Fast-dLLM v2 的官方锁定版本，LLaDA 与 Dream 的 remote code 在 5.x 下无法加载）。模型为 Fast-dLLM-v2-1.5B（28 层，隐藏维 1536）、LLaDA-8B-Instruct（32 层，4096）、Dream-7B-Instruct（28 层，3584），精度为 bfloat16。

输入为 chat template 提示（不超过 96 token）与 512 token 长提示。子分支通过在父序列上随机翻转 n∈{0,1,2,4,8,16,32,128} 个 token 构造，翻转位置固定随机种子。指标包括逐层稳定率、相对 logit MSE（MSE 除以基线方差）、top-1 一致率与墙钟延迟；延迟以 CUDA Events 测量，预热不少于两次、重复不少于八次，全部在 no_grad 下执行。计时在独占 GPU 上重复三轮取均值。所有原始数据保存在 results/experiments 与 results/optimization，共 47 个数据文件。

---

## 6 证据与结果

### 6.1 精确性：算法不变量

折叠机制的正确性由两组不变量界定（表 1）。自折叠迫使整层快速路径，三个模型的输出与各自基线逐位一致（MSE 为 0，top-1 为 1.0）；强制全发散（τ=1.0，余弦裁剪后）同样逐位一致。这两组结果说明，折叠路径本身不引入数值误差，后续所有保真度差异都来自"稳定 token 采用父输出"这一设计选择。图 8 给出不变量与快速路径的对照。

**表 1. 算法不变量（bfloat16）**

| 不变量 | Fast-dLLM-v2-1.5B | LLaDA-8B | Dream-7B |
|---|---|---|---|
| 自折叠稳定率 / MSE / top-1 | 1.000 / 0.0 / 1.0 | 1.000 / 0.0 / 1.0 | 1.000 / 0.0 / 1.0 |
| 全发散稳定率 / MSE / top-1 | 0.000 / 0.0 / 1.0 | 0.000 / 0.0 / 1.0 | 0.000 / 0.0 / 1.0 |
| 基线全模型前向（ms） | 16.5 | 17.6 | 16.6 |
| 原实现全稳定路径（ms） | 52.1 | 51.2 | 53.7 |
| 向量化缓存后全稳定路径（ms） | 6.45 | 6.37 | 6.81 |

### 6.2 稳定性现象与相似度结构

三个模型的相似度结构呈现一致的形态（表 2）。未改动 token 的余弦相似度接近 1，单 token 扰动对大多数位置的影响可以忽略；被翻转的 token 相似度降至 0.9 附近。bf16 下即使逐位相同的张量，余弦仍有约 1e-3 的噪声，因此阈值在 0.999 附近由数值误差主导。模型之间存在稳定差异：Fast-dLLM 的单 token 扰动后 5% 分位为 0.96，LLaDA 与 Dream 降至 0.60 附近，说明后两者对激活扰动的敏感度约为前者的六倍。这一差异在后续的保真度与采样结果中反复出现。图 1 与图 2 分别给出层与 token 的相似度图谱和逐层曲线。

**表 2. 父子逐层余弦相似度（均值 / 5% 分位）**

| 模型 | 0 翻转 | 1 翻转 | 8 翻转 |
|---|---|---|---|
| Fast-dLLM-v2-1.5B | 0.9987 / 0.9961 | 0.9797 / 0.9584 | 0.7957 / 0.2494 |
| LLaDA-8B-Instruct | 0.9986 / 0.9936 | 0.9387 / 0.6029 | 0.8195 / 0.3945 |
| Dream-7B-Instruct | 0.9987 / 0.9961 | 0.9408 / 0.5939 | 0.6932 / 0.1654 |

### 6.3 原实现的失败模式

**阈值与层、缓存的作用。** 固定阈值扫描（表 3）显示稳定率在 τ∈[0.5,0.99] 内保持平台，仅在 0.995 附近断崖下降，在 0.999 处归零；延迟在整个区间恒定，只有全发散时降至约 61 ms。层选择消融（表 4）给出相反的最优层集合：Fast-dLLM 折叠早期层误差最小，LLaDA 与 Dream 折叠后期层最安全，Dream 仅折叠后三分之一层即可把相对误差从 14.8% 降至 2.6%。缓存预算（表 5）显示稳定率随预算近似线性增长并在覆盖序列长度后饱和；预算极小时 top-1 反而完美，说明小范围保守复用比大范围近似更安全。图 3 至图 5 给出对应曲线。

**表 3. 固定阈值扫描（1 token 翻转；稳定率 / 相对 MSE / top-1）**

| τ | Fast-dLLM | LLaDA-8B | Dream-7B |
|---|---|---|---|
| 0.95 | 0.976 / 2.2e-2 / 0.927 | 0.971 / 2.5e-1 / 0.912 | 0.976 / 1.5e-1 / 0.927 |
| 0.99 | 0.976 / 2.2e-2 / 0.927 | 0.969 / 2.5e-1 / 0.912 | 0.976 / 1.5e-1 / 0.927 |
| 0.995 | 0.641 / 1.2e-2 / 0.976 | 0.583 / 1.9e-1 / 0.971 | 0.506 / 9.0e-2 / 0.951 |
| 0.999 | 0.000 / 0.0 / 1.000 | 0.000 / 0.0 / 1.000 | 0.000 / 0.0 / 1.000 |

**表 4. 层选择消融（τ=0.99；稳定率 / 相对 MSE / top-1）**

| 折叠范围 | Fast-dLLM | LLaDA-8B | Dream-7B |
|---|---|---|---|
| all | 0.976 / 2.17e-2 / 0.927 | 0.971 / 2.50e-1 / 0.912 | 0.976 / 1.48e-1 / 0.927 |
| early-only | 0.976 / 8.13e-3 / 0.927 | 0.971 / 1.37e-1 / 0.971 | 0.976 / 1.27e-1 / 0.951 |
| late-only | 0.951 / 1.00e-2 / 0.976 | 0.753 / 5.71e-2 / 0.971 | 0.840 / 2.56e-2 / 0.976 |
| none | 0.000 / 0.0 / 1.000 | 0.000 / 0.0 / 1.000 | 0.000 / 0.0 / 1.000 |

**表 5. 缓存预算（Fast-dLLM；稳定率 / top-1）**

| 每层上限 | 1 | 4 | 16 | 64 | 256 | 65536 |
|---|---|---|---|---|---|---|
| 稳定率 | 0.024 | 0.098 | 0.390 | 0.976 | 0.976 | 0.976 |
| top-1 | 1.000 | 1.000 | 0.976 | 0.927 | 0.927 | 0.927 |

**开销分解。** 表 6 把单层耗时拆成组件。缓存读取在三个模型上分别为 0.699、0.652、0.697 ms，均高于原始层计算的 0.538、0.533、0.606 ms；门控与融合仅占 0.03–0.07 ms。这解释了稳定率高而延迟不降的现象：问题的根源在数据路径，不在相似度计算或合并算子。硬件微基准测得 FP16 矩阵乘约 136–138 TFLOPS、带宽约 1280 GB/s；以此为参数的成本模型预测稳定率 0.97 时单层约 0.02 ms，实测折叠前向约 74–83 ms，相差约四个数量级，差距来自"整层重算"语义与被忽略的数据路径开销。图 6 与图 7 分别展示开销分解与预测对照。

**表 6. 单层开销分解（batch=1，ms）**

| 组件 | Fast-dLLM | LLaDA-8B | Dream-7B |
|---|---|---|---|
| 原始层重算 | 0.538 | 0.533 | 0.606 |
| 折叠层，全稳定 | 1.889 | 1.544 | 1.825 |
| 折叠层，全发散 | 2.146 | 1.808 | 2.081 |
| 相似度门控 | 0.067 | 0.065 | 0.066 |
| 缓存读取 | 0.699 | 0.652 | 0.697 |
| 缓存写入 | 0.228 | 0.190 | 0.230 |
| 融合（Triton / PyTorch） | 0.026 / 0.007 | 0.038 / 0.011 | 0.024 / 0.007 |

### 6.4 向量化缓存的效果

向量化缓存的微基准在三个模型上一致：put 比逐 token 缓存快 12–183 倍，get 快 30–429 倍（序列长度 34 至 512）。T=512、隐藏维 4096 下，部分命中的读取由 0.174 ms 降至 0.058 ms。端到端结果见表 7：全稳定折叠前向由原缓存的 51–53 ms 降至 6.4–6.8 ms，相对不折叠基线的加速为 2.3–2.7 倍；部分稳定路径（单个发散 token）由约 80 ms 降至 39–43 ms。图 10 与图 11 给出加速曲线与汇总。

**表 7. 向量化缓存的端到端效果（全稳定路径，ms）**

| 模型 | 基线 | 逐 token 缓存 | 分块缓存 | 向量化缓存 | 相对基线 |
|---|---|---|---|---|---|
| Fast-dLLM-v2-1.5B | 15.4 | 52.1 | 8.5 | 6.45 | 2.4× |
| LLaDA-8B-Instruct | 17.2 | 51.2 | 8.6 | 6.37 | 2.7× |
| Dream-7B-Instruct | 16.6 | 53.0 | 8.8 | 6.81 | 2.4× |

### 6.5 层内 FFN 拆分的效果

拆分与不拆分的输出在三个模型上 MSE 与 top-1 完全一致，bf16 logits 的最大绝对差为 0.16–0.28，属于分块矩阵乘的舍入差异。逐层增益随形状变化（表 8）：任意发散比例下，batch×seq 为 64 时拆分为 0.7–0.8，512 时转为 1.0–1.15，2048 时达到 1.24–1.93。收益来自"省下的 FFN 计算量随发散 token 数与隐藏维平方增长"，成本是每层一次数据相关同步；二者相交的位置决定了启用阈值，默认取 batch×seq≥512。端到端 seq=512 的结果见表 9：在单个翻转 token 的最有利条件下，拆分把 LLaDA 与 Dream 的折叠路径缩短约 17–18%，随翻转数增加收益收窄至 8–14%；Fast-dLLM 在少翻转时略慢 5–9%，多翻转时与不拆分持平。三条折叠路径仍慢于不折叠基线，剩余差距即 6.3 节定位的逐层固定开销。图 12 与图 13 展示形状扫描与长序列对照。

**表 8. 拆分逐层加速（LLaDA 层，隐藏维 4096；full/split）**

| 形状 | 1% 发散 | 10% 发散 | 50% 发散 | 100% 发散 |
|---|---|---|---|---|
| B=1, T=64 | 0.80× | 0.80× | 0.70× | 0.80× |
| B=1, T=512 | 1.15× | 1.14× | 1.01× | 0.92× |
| B=1, T=2048 | 1.78× | 1.69× | 1.24× | 0.98× |
| B=4, T=2048 | 1.93× | 1.78× | 1.32× | 0.99× |

**表 9. seq=512 端到端（ms；折叠列为随翻转数变化的区间）**

| 模型 | 基线 | 不拆分折叠 | 拆分折叠 |
|---|---|---|---|
| Fast-dLLM-v2-1.5B | 30.7 | 36.1–42.1 | 37.9–42.4 |
| LLaDA-8B-Instruct | 35.4 | 49.8 | 40.9–46.0 |
| Dream-7B-Instruct | 29.8 | 42.6 | 35.3–36.5 |

### 6.6 自适应秩门控的效果

自适应秩门控在目标 0.5、0.8、0.9、0.97 与 1.0 上精确命中稳定率，而固定阈值只能给出平台值。表 10 对比单 token 扰动下的前沿代表点：在稳定率更低（复用更少）的设定下，Fast-dLLM 的相对误差由 6.07e-2 降至 4.33e-2，LLaDA 的 top-1 由 0.882 升至 0.971，Dream 的相对误差由 2.20e-1 降至 8.35e-2。8 token 扰动时固定阈值的 top-1 降至 0.46–0.68，秩门控通过降低稳定率可换回 0.61–0.85。目标设在 1.0 会触发全稳定快速路径（约 6.5 ms），此时 8 token 扰动的保真度损失显著，适用性由任务约束决定。图 14 给出完整前沿。

**表 10. 自适应秩门控与固定阈值的前沿对比（1 token 翻转；稳定率 / 相对 MSE / top-1）**

| 模型 | 固定 τ=0.99 | 自适应秩门控，target=0.8 |
|---|---|---|
| Fast-dLLM | 0.976 / 6.07e-2 / 0.927 | 0.805 / 4.33e-2 / 0.927 |
| LLaDA-8B | 0.971 / 2.07e-1 / 0.882 | 0.794 / 1.81e-1 / 0.971 |
| Dream-7B | 0.976 / 2.20e-1 / 0.902 | 0.805 / 8.35e-2 / 0.976 |

### 6.7 融合核的效果

融合核在全部测试形状上与 PyTorch 路径逐位一致（最大绝对误差 0）。表 11 显示收益随形状增大而出现：T=128 时启动开销占优（0.64×），T=2048 且隐藏维 8192 时转为 1.50×，T=8192 时达 1.84–3.43×。默认不为小形状启用融合路径。图 15 给出形状扫描。

**表 11. 融合 gather-select（bfloat16，ms）**

| 形状 | PyTorch | 融合核 | 加速 |
|---|---|---|---|
| T=128, H=4096 | 0.0174 | 0.0271 | 0.64× |
| T=2048, H=8192 | 0.0413 | 0.0276 | 1.50× |
| T=8192, H=4096 | 0.1716 | 0.0500 | 3.43× |
| T=8192, H=8192 | 0.3724 | 0.2027 | 1.84× |

### 6.8 跨步扩散采样

跨步折叠在 31 步生成中的表现见表 12。Fast-dLLM 的折叠生成与基线逐 token 完全一致，说明近似在块式掩码解码中可忽略；LLaDA 与 Dream 的稳定率同为 0.99 左右，但 token 匹配率仅 0.71–0.73，文本出现重复词。原因与 6.2 节的敏感度一致：早期步骤的微小 logit 扰动被贪婪解码沿步骤放大，其中 LLaDA 与 Dream 的放大更明显。提高阈值或降低目标稳定率可以换取一致性，代价是复用率下降。图 9 给出逐步稳定率与延迟对照。

**表 12. 扩散采样跨步折叠（32 token、32 步）**

| 模型 | token 匹配率 | 平均逐步稳定率 | 基线 / 折叠（ms） |
|---|---|---|---|
| Fast-dLLM-v2-1.5B | 1.000 | 0.980 | 471 / 3791 |
| LLaDA-8B-Instruct | 0.727 | 0.989 | 726 / 3730 |
| Dream-7B-Instruct | 0.712 | 0.987 | 768 / 3867 |

### 6.9 适用边界与配置建议

证据链给出的边界可以概括为三点。全稳定或接近全稳定的场景，向量化缓存直接兑现 2.3–2.7 倍加速。部分稳定场景的折叠路径在本文测试规模下仍慢于不折叠基线，提升来自拆分与融合核，但尚不足以反超；其剩余开销是每层固定的门控、缓存与调度成本，量级为 0.1–0.2 ms 每层。长序列与大 batch 是拆分与融合核的收益区间，batch×seq≥2048 时逐层收益可达 1.3–1.9 倍。表 13 按场景给出配置。

**表 13. 场景化配置**

| 场景 | 配置 |
|---|---|
| 短序列、高稳定率 | 向量化缓存；全稳定快速路径获得 2.3–2.7 倍加速 |
| 长序列或大 batch（batch×seq≥2048） | 叠加层内拆分；T≥8192 时启用融合核 |
| 保真度敏感或需要精确复用预算 | 自适应秩门控；必要时按模型选择折叠层集合（Dream 折叠后三分之一层） |
| 扩散采样质量优先 | 提高阈值或降低目标稳定率；Fast-dLLM 可安全使用跨步折叠 |

---

## 7 讨论

### 7.1 设计启示

复用的收益由两层因素决定。算法层决定"可复用多少"，即稳定 token 的比例与误差；系统层决定"复用换回多少时间"，即数据路径与调度成本。本文的测量表明，当复用比例高但每层固定成本接近甚至超过整层计算时，系统层是瓶颈。这一结论对缓存驱动的推理系统具有一般性：KV 缓存、特征复用与激活复用都面临相同的成本结构，只有当读取路径的开销低于被跳过的计算时，复用才有意义。

由此得到两条设计原则。其一，复用单元应与计算可分离性对齐：注意力必须保留完整上下文，FFN 可以按 token 拆分，把两者分开处理才能在部分稳定场景获得按比例的计算节省。其二，复用预算应按相对秩分配：绝对阈值在不同模型、不同层之间含义漂移，而秩选择天然适应分布，并且能够把误差预算花在最相似的位置上。

### 7.2 有效性边界

实验在单卡、batch=1、bfloat16 与固定提示族下完成，这是结论成立的范围。三个模型覆盖 LLaDA、Dream 与 Fast-dLLM 三种解码配方，但不覆盖更大规模（14B 以上）与训练期协同设计。延迟测量使用 CUDA Events 均值，未锁定频率，跨实例比较以相对加速为准。采样结果来自单次 32 步生成，匹配率受随机种子影响。发布级结论还需要多提示、多随机种子与任务级指标（例如 GSM8K 与 HumanEval+）的复测，以及 batch 大于 1 与序列超过 2048 的全链路验证。

---

## 8 结论

跨分支激活复用在扩散语言模型的推测解码中是精确且高稳定的机制：自折叠与全发散路径与基线逐位一致，跨步稳定率达到 0.94–0.99。把这一稳定性转化为墙钟加速要求在系统层解决两件事：把逐 token 的数据路径改为连续缓冲与批量操作，把 FFN 从上下文中解耦到可按 token 拆分。完成这两项后，全稳定折叠前向较不折叠基线加速 2.3–2.7 倍；层内拆分在 batch×seq≥512 时取得 1.1–1.93 倍逐层收益，并把 seq=512 折叠路径缩短 8–18%；自适应秩门控在保真度与复用前沿上优于固定阈值；融合核在长序列达 3.43 倍。部分稳定场景的剩余瓶颈是每层固定的门控、缓存与调度开销（0.1–0.2 ms 每层），把门控、读取与合并融合为单个算子、并以静态预算替代数据相关选择，是使该场景反超基线的主要方向。

---

## 参考文献

[1] S. Nie, F. Zhu, Z. You, et al. Large Language Diffusion Models. arXiv:2502.09992, 2025.
[2] Dream-org. Dream-v0-Instruct-7B（模型卡与远程代码）. Hugging Face, 2025.
[3] NVlabs. Fast_dLLM_v2_1.5B（模型卡、代码与生成工具）. Hugging Face / GitHub, 2025.
[4] Y. Leviathan, M. Kalman, Y. Matias. Fast Inference from Transformers via Speculative Decoding. ICML 2023.
[5] C. Chen, S. Borgeaud, G. Irving, et al. Accelerating Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318, 2023.
[6] H. Chang, H. Zhang, L. Jiang, et al. MaskGIT: Masked Generative Image Transformer. CVPR 2022.
[7] T. Cai, Y. Li, Z. Geng, et al. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. ICML 2024.
[8] Y. Li, F. Wei, C. Zhang, H. Zhang. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. ICML 2024.
[9] X. Ma, G. Fang, X. Wang. DeepCache: Accelerating Diffusion Models for Free. CVPR 2024.
[10] W. Kwon, Z. Li, S. Zhuang, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023.
[11] A. Paszke, S. Gross, F. Massa, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.
[12] T. Wolf, L. Debut, V. Sanh, et al. Transformers: State-of-the-Art Natural Language Processing. EMNLP（系统演示）, 2020.
[13] P. Tillet, H.-T. Kung, D. Cox. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. MAPL 2019.

---

## 附录 A 图表索引

| 编号 | 文件 | 内容 |
|---|---|---|
| 图 1 | `figures/experiments/fig_similarity_heatmap.png` | 层与 token 的相似度热力图（flip=1/8） |
| 图 2 | `figures/experiments/fig_stable_by_layer.png` | 逐层相似度与稳定率 |
| 图 3 | `figures/experiments/fig_tau_quality.png` | 阈值扫描：稳定率、top-1、相对误差 |
| 图 4 | `figures/experiments/fig_layer_ablation.png` | 层选择消融 |
| 图 5 | `figures/experiments/fig_cache_budget.png` | 缓存预算 |
| 图 6 | `figures/experiments/fig_overhead_breakdown.png` | 单层开销分解 |
| 图 7 | `figures/experiments/fig_latency_vs_prediction.png` | 成本模型与实测对照 |
| 图 8 | `figures/experiments/fig_invariants.png` | 算法不变量与快速路径 |
| 图 9 | `figures/experiments/fig_sampling.png` | 跨步采样 |
| 图 10 | `results/optimization/figures/fig_opt1_cache.png` | 向量化缓存加速曲线 |
| 图 11 | `results/optimization/figures/fig_opt_summary.png` | 向量化缓存汇总（2.3–2.7 倍） |
| 图 12 | `results/optimization/figures/fig_opt2_shape.png` | 拆分形状扫描 |
| 图 13 | `results/optimization/figures/fig_opt2_longseq.png` | seq=512 端到端 |
| 图 14 | `results/optimization/figures/fig_opt3_frontier.png` | 自适应秩门控前沿 |
| 图 15 | `results/optimization/figures/fig_opt4_fused.png` | 融合核形状扫描 |

**表 14. 工程修复清单**

| 编号 | 问题 | 修复 |
|---|---|---|
| 1 | Triton 3.4 拒绝 kernel 内类型注解，编译崩溃 | 移除注解，编译失败回退 PyTorch |
| 2 | transformers 5.x 透传 load_in_8bit=False | 仅在启用量化时传参 |
| 3 | 三个模型族封装不接受 torch_dtype 等参数 | 对齐基类构造参数 |
| 4 | LLaDA 仅支持 last_hidden_state 输出 | 兼容 logits、last_hidden_state、hidden_states |
| 5 | 折叠包装器返回 ModelOutput 对象而非张量 | 输出解包 |
| 6 | LLaDA 块返回元组，折叠层只返回张量 | 保持输出元数 |
| 7 | LLaDA 层堆栈探测失败 | 补充 model.transformer.blocks 与 wte 路径 |
| 8 | bf16 余弦超过 1 造成假稳定 | 余弦裁剪到 [-1,1] |
| 9 | 采样器不传递父分支标识；mask token 解析失败；Dream 掩码类型错误 | 跨步父链、统一 mask 解析、bool 化掩码 |
| 10 | 分块缓存的 get 形状与空缓存语义不符合合约 | 全长零填充并抛出 KeyError |

## 附录 B 复现与数据可用性

诊断与优化的完整命令见 `docs/DEEP_EXPERIMENT_REPORT.md` 第 18.2 节与 `docs/OPTIMIZATION_REPORT.md` 第 7 节。原始数据位于 `results/experiments/` 与 `results/optimization/`；代码与测试位于 `actfold/` 与 `tests/`，`pytest -m "not slow"` 通过 204 项。环境依赖：PyTorch 2.8、Triton 3.4、Transformers 4.53.1、huggingface_hub 0.36.2。
