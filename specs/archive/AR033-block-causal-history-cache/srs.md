# [AR033] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR033 |
| AR 主题 | Block-causal 精确历史缓存（官方 Fast-dLLM v2 parity）+ 非缓存基线臂 strawman 勘误 + 三臂实机基准 |
| 关联 SR | AGENTS「Experimental Validation Rule」第 6 条（框架级优越性阶梯的基线口径）；D1（block-causal 编辑点左侧 cos≡1.0 构造级精确）；D4（多块生成质量门）；README「跨平台复核」节基线口径 |
| 日期 | 2026-10-24 |
| 状态 | Delivered |

## 1. 背景与目标

### 1.1 官方 v2 配方审计结论（本 AR 的需求期事实发现）

官方 NVlabs Fast-dLLM v2 参考实现（`v2/generation_functions.py`，main@a9b81e4，
本地浅克隆 `C:\Users\l30086046\AppData\Local\Temp\opencode\fastdllm_ref`）的
`batch_sample` **不是全量重算配方**，而是分层缓存配方：

- **跨块历史 KV 缓存**：prompt 预填充一次建立 `past_key_values`（:37，
  `use_cache=True, update_past_key_values=True`）；每完成一块，对该块做一次
  forward 将其 KV 提交进历史（:81-82）；
- **块内 sub-block KV 缓存**（`use_block_cache=True` 时）：以 small_block 为粒度
  缓存当前块内已定部分的 KV，失效规则为「当前 small block 的前界 token 仍为
  mask 则整块重建」（:102），命中时仅对 small-block 切片做部分前向（:108，
  `replace_position`）；
- **即使不开 block cache**，每个重掩码迭代也只前向当前块 `x_t[:, -block_size:]`
  对着冻结历史 KV（:111）——单迭代代价 O(block)，而非 O(T)；
- 官方 README 明示「Hierarchical Caching System: Block-level cache + Sub-block
  cache」，v2 声明对标准 AR 解码最高 2.5× 加速；
- 携带 `use_cache` / `past_key_values` / `update_past_key_values` /
  `use_block_cache` / `block_past_key_values` / `replace_position` 前向签名
  的模型代码随 HF checkpoint 分发（trust_remote_code，
  `Efficient-Large-Model/Fast_dLLM_v2_1.5B` / `_7B`）；`generation_functions.py`
  是挂接在该模型类上的 mixin。

### 1.2 ActFold 现状与 strawman 判定

- `FastDLLMSampler` 官方路径（`fast_dllm_sampler.py:726-766`）每个重掩码迭代对
  整个 [T] 画布全量重算（:740 / :756 `self._forward(x, ...)` 全序列）；
- `_forward`（`diffusion_sampler.py:139-229`）无任何 cache 尾参；全仓库
  `past_key_values|use_cache` 生产代码零命中——ActFold 没有任何历史 KV 缓存；
- **判定**：本仓库阶梯（AR016-AR032）所对照的「官方配方基线臂」实为**非缓存
  全量重算变体**，不是官方 v2 `batch_sample` 配方。既有加速比（vs 基线臂）
  不能解读为「vs 官方 SOTA 配方」的加速比。AGENTS #21 的「aligned with the
  official recipes」限定于掩码调度/块解码/置信度解掩码语义，不含计算策略——
  该口径差异必须勘误。

### 1.3 机会与目标

D1 已证 block-causal 掩码下编辑点左侧输出 cos≡1.0（构造级精确）——历史前缀
跨重掩码迭代、跨块逐位不变。官方 v2 正是用这一结构性精确性在 KV 层提取了
该冗余。本 AR 的目标：

1. **勘误（诚实义务，无论机制结果如何都必须交付）**：基线臂口径重述为
   「非缓存 block-diffusion 基线臂」；阶梯加速比全部重新圈定；README/AGENTS/
   CHANGELOG 叙事级修正（冻结工件与判定函数零回改）。
2. **机制（官方 parity）**：`FastDLLMSampler` 新增默认关闭的 `history_cache`
   模式——跨块历史 KV + 块内 small-block KV（含官方边界失效规则），重掩码
   迭代的前向从 [T] 收缩到活动切片。
3. **证据（三臂实机基准）**：A=非缓存基线臂 / B=history_cache 臂 /
   C=官方原生 `batch_sample` 参照臂（真实 checkpoint 上直接挂官方 mixin 跑），
   预承诺单次运行 + 冻结判定式 + 多块质量门（D4 教训）。
4. **复现**：RERUN 服务器节（权重重下配方 + AR019 venv 重建 + 命令 + 资源表）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- 官方 v2 `batch_sample` 配方的代码级审计结论落入 design.md（缓存数据结构、
  失效规则、token-shift logits 对齐、提交时序）。
- `FastDLLMSamplerConfig.history_cache`（默认 `False` 字节不变）+ 采样循环
  cache 分支：预填充建史 → 每迭代切片前向 → 块完成提交 → 块内 small-block
  缓存与边界重建。
- 模型能力探测与委托合同：wrapped 模型前向签名支持 cache kwargs 时委托原生
  缓存（#15 签名探测先例）；不支持时 armed 路径显式 `RuntimeError`，**绝不**
  静默退化为全量重算。
- fp32 tiny 真实 block-causal 模型（tests fixture，2 层/小 hidden，携带同构
  cache kwargs 接口）——位级精确性 UT 的载体（真实计算，非 mock，#10）。
- 三臂 bench 脚本（AR031/AR032 模板）：冻结 H1-H4、`--quick` 与 formal 分离、
  预承诺单次、覆盖保护。
- strawman 勘误 + README 全量重写（Full-Update Rule）+ AGENTS 新条目 +
  CHANGELOG + RERUN 步骤。
- 环境重建：Fast_dLLM_v2_1.5B 权重重下（RERUN curl 配方）+ AR019 venv 重建
  （transformers==4.53.1）+ remote-code cache 签名探针。

**Out of Scope（本 AR 不做的）：**

- LLaDA / Dream 家族的 history cache（Fast-dLLM 的 block-causal 结构是精确
  前提；LLaDA 全双向掩码下历史不精确冻结，属不同机制域）。
- 折叠谱系与缓存的组合机制（B 臂与折叠的正交性分析作为基准 sidecar 记录，
  不新增组合机制档）。
- CUDA graph 与 history cache 的组合（图捕获静态缓冲与动态 KV 生长的合同
  冲突需独立设计，本 AR 记录为路线图项）。
- 官方 `use_block_cache=True` 之外的新失效策略优化（本 AR 只做 parity 与
  证据，不做超越官方失效策略的调优）。

## 3. 功能需求

### 3.1 FR1 history_cache 采样模式（默认关）

**描述：** `FastDLLMSamplerConfig.history_cache`（默认 `False`，字节不变）开启
后，官方路径采样循环改为分层缓存形态：prompt 预填充一次建立每层历史 KV；
每个重掩码迭代仅对活动块（含块内缓存时为活动 small-block 切片）前向，历史
KV 作为只读上下文；块完成后将该块 KV 提交进历史；下一个 AR token 的推进
前向（block-advance）同样切片+提交。

**触发条件：** `history_cache=True` 的 `FastDLLMSampler.sample` 官方路径调用。

**期望行为：** 生成 token 序列与关闭时语义等价（fp32 位级一致见 FR3）；每迭代
前向 token 数从 [T] 收缩到活动切片；投机路径（`num_block_candidates>0`）与
history_cache 的组合在本 AR 为显式 `ValueError`（未定义面，禁止静默混跑）。

**异常处理：** wrapped 模型不支持所需 cache kwargs → 构造/首调用时
`RuntimeError`（带能力探测详情）；KV 形状/层数不匹配 → `RuntimeError`
（fail-fast，禁止截断式静默继续）。

**验收标准：**
- Given fp32 tiny block-causal 模型 + 多块生成配置, When `history_cache` 开/关
  两次贪心采样, Then 输出 token 序列逐位一致且开臂每迭代前向 token 数等于
  活动切片长度（计数器钉固）。
- Given 默认配置（`history_cache=False`）, When 任意既有 UT/基准运行,
  Then 行为与 AR032 终态字节一致（零回归合同）。

### 3.2 FR2 块内 small-block 缓存与官方边界失效规则

**描述：** `history_cache` 开启时，当前块内已完成 small-block 的 KV 以
官方 `use_block_cache` 语义缓存：活动 small block 的前界 token（上一 small
block 末 token）仍为 mask → 整块重建缓存；否则仅对活动 small-block 切片
前向（`replace_position` 语义对齐）。失败/重建路径不产生错误输出。

**验收标准：**
- Given 块内多 small-block 配置 + 人为构造「前界 token 迟迟未解掩」的调度
  序列, When 缓存臂运行, Then 重建恰在前界解掩前触发（UT 钉固失效规则与
  官方 :102 语义一致）。
- Given 块内缓存开/关（`history_cache` 的子开关）, When fp32 tiny 模型
  A/B, Then 输出逐位一致。

### 3.3 FR3 精确性证据（位级 + 真实面）

**描述：** 精确性分两面交付：
(a) **构造级证明面**：fp32 tiny 真实 block-causal 模型上，cache 臂 vs 全量
重算臂在多块、batch 1/2、pad 排除、forced 区域、token-shift logits 对齐下
逐位一致（`torch.equal`）；
(b) **真实面漂移记录**：真实 checkpoint（bf16）上贪心 token 一致率 +
解掩调度对照，bf16 kernel 归约序噪声导致的 ULP 级 logits 漂移与近并列
argmax 翻转按 #51 先例记录为漂移事实而非错误。

**验收标准：**
- Given fp32 tiny 模型, When cache 开/关全因子 A/B（≥4 形状点），Then
  `torch.equal` 全绿。
- Given 真实 checkpoint 贪心多块生成, When cache 开/关 A/B, Then token
  一致率入工件（冻结门限见 FR4 H1），漂移样本原样记录。

### 3.4 FR4 三臂实机基准（预承诺单次）

**描述：** `scripts/ar033_history_cache_bench.py`（AR031/AR032 模板）：
A=非缓存基线臂（现状官方路径）/ B=history_cache 臂 / C=官方原生
`batch_sample` 参照臂（真实 checkpoint 挂官方 mixin，`use_block_cache` 两态
均可，默认态如实记录）。判定式在数据收集前冻结于脚本：

- **H1 质量**：B vs A 的 posthoc 置信度 gap（AR022/031/032 同构面）+
  解码长度恢复 + 贪心 token 一致率（冻结门限，设计期定数）；
- **H2 速度**：p50(A)/p50(B) > 1（多块工作负载；WDDM 平台注记 #46）+
  B vs C 的 parity 差距如实报告（不作门限，作定位面）；
- **H3 计数恒等**：B 臂每迭代前向 token 数账本（活动切片 vs A 的全画布），
  恒等式预冻结；
- **H4 canary/repro**：复算双跑按 `(arm, prompt_index)` 键控严格行相等
  （AR031 教训）+ 覆盖保护 + `--quick` 独立工件。

**验收标准：**
- Given 冻结判定式, When 单次 reps=30 运行, Then 工件
  `results/optimization/ar033_history_cache_bench.json` 全要素可溯源，
  诚实负结果（inconclusive/refuted/blocked）原样交付。
- C 臂不可运行（remote code 缺 kwargs / mixin 不兼容）时 → C 臂记 blocked
  携带最小复现签名，零伪造数字，A/B 判定不受 C 阻塞影响。

### 3.5 FR5 strawman 勘误与阶梯口径重述

**描述：** 叙事级修正（冻结工件/判定函数零回改）：README 阶梯表与跨平台复核
节的基线口径圈定为「非缓存 block-diffusion 基线臂」，并新增本 AR 勘误注记
（官方 v2 配方含分层缓存，引用官方文件行号）；AGENTS 新增条目载明该口径
律（后续 AR 的 baseline 面必须声明计算策略口径）；CHANGELOG 记录。
**并入（design §6.5 设计期发现）**：基线臂 block-advance off-by-one 勘误——
旧臂 advance 消费 raw[T-2]（预测位置 T-1）而非官方语义的未位移尾行
raw[T-1]；T004 内修复（含 AR032 reuse stash 同步修复，保持其 on/off A/B
等价合同）；AR033 前多块工件的块推进 token 携带旧约定（A/B 内部对照仍
有效），README 勘误注记同节载明。

**验收标准：**
- Given README/AGENTS/CHANGELOG 更新, When 读者对照官方 v2 仓库, Then
  口径差异（掩码调度对齐 vs 计算策略未对齐）可独立复核。

### 3.6 FR6 复现文档（服务器节）

**描述：** `docs/RERUN_CHECKLIST.md` 新步骤 + README 复现块：权重重下
（repo id + 本地布局）、AR019 venv 重建、三臂命令、预期资源表（GPU/VRAM/
墙钟，引用工件实测值）、判定式语义。

**验收标准：**
- Given 全新高算力服务器 + 仓库, When 按 RERUN 步骤执行, Then 三臂基准
  可复现且工件 schema 与本 AR 交付一致。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 兼容 | 默认关 | `history_cache=False` 字节不变（零回归合同，FR1） |
| 质量 | 静态检查 | black/isort/pyflakes 零告警；mypy actfold 75 + scripts 30（`--explicit-package-bases`）零 error |
| 质量 | 测试基线 | 全量 1465 passed / 4 skipped / 3 deselected + 新增 UT 全绿 |
| 诚实 | 零 mock | #10 全承：tiny 模型为真实计算 fixture；C 臂不可运行记 blocked 不伪造 |
| 诚实 | 单次承诺 | formal 预承诺单次，不重跑钓鱼；WDDM 挂起先杀僵尸进程（#46 协议） |
| 平台 | 计时纪律 | #39 wall-clock + 同环境 A/B；跨工件绝对 p50 不可比（#51 先例注记） |

## 5. 约束与假设

**约束：**
- 本地权重（`ar019_model\Fast_dLLM_v2_1.5B`）与 AR019 venv 均已被 Temp 清理
  删除——T001 环境重建是实机面前置（RERUN curl 配方 + transformers==4.53.1）。
- Windows WDDM + torch 2.5.1+cu121 + 16GB Quadro RTX 5000；GPU 用前清僵尸
  进程（`Get-Process python` 全杀协议）。
- 企业代理：HF 下载走 RERUN 配方镜像；Web fetch 403（git 协议可用）。

**假设：**
- 1.5B checkpoint remote code 携带官方 mixin 所调用的 cache kwargs（官方
  `batch_sample` 即挂接该类；T001 探针实测落锤——若缺 `use_block_cache`，
  C 臂以无块内缓存态运行并如实记录）。
- bf16 真实面的 ULP 漂移不系统性破坏贪心调度（H1 冻结门限吸收；fp32 UT
  为构造级证明面）。

**风险：**
- token 一致率低于预期（近并列 argmax 翻转聚集）→ H1 按冻结门限如实判定，
  漂移样本入工件。
- WDDM 小几何下 B vs C 差距失真（#46 先例）→ 平台注记 + RERUN 服务器面。
- 历史缓存使折叠谱系的前缀 FFN 复用失去对象（前缀不再计算）→ 作为基准
  sidecar 分析记录，非本 AR 失败面。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 非缓存基线臂 | 本仓库既有官方路径采样（每重掩码迭代全量重算 [T]）——阶梯历史加速比的分母 |
| 官方 v2 配方 | NVlabs `batch_sample`：预填充建史 + 每迭代块切片前向 + 块完成提交 + 可选块内 small-block 缓存 |
| 历史冻结 | block-causal 掩码下编辑点左侧激活跨迭代/跨块逐位不变（D1 构造级精确） |
| C 参照臂 | 真实 checkpoint 上挂官方 mixin 直接运行 `batch_sample` 的真 SOTA 定位面 |
| token 一致率 | 同 prompt 同种子下 cache 臂 vs 基线臂贪心生成 token 序列的逐位相等比例 |

## 7. 任务清单

见同目录 `tasks.md`（T001-T011；环境重建为实机面前置）。
