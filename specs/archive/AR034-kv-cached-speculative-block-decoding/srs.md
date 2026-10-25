# [AR034] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR034 |
| AR 主题 | KV-cached speculative block decoding（历史 KV 缓存 × 投机候选验证的组合加速档） |
| 关联 SR | AGENTS「Experimental Validation Rule」第 6 条（框架级优越性阶梯）；AR033（分层 KV 缓存，官方 v2 parity，显式将「缓存 × 投机/折叠组合」列为 out of scope）；AR021/AR022（批量投机 + 贪心锚定质量对齐）；AR032（推进前向复用） |
| 日期 | 2026-10-24 |
| 状态 | Delivered |

## 1. 背景与目标

### 1.1 现状事实（AR033 收口后）

- AR033 交付了官方 Fast-dLLM v2 分层缓存的 ActFold parity：`history_cache`
  把每个重掩码迭代的前向从全画布 `[T]` 收缩到活动切片，B2 臂实测
  **1.5431×** vs 非缓存基线臂（token 账本 B2/A=0.1352），且跨块面（B1）在
  真实 checkpoint 上逐位精确（AR033 T001 P3：maxdiff=0.0）。
- AR021/AR022 交付了批量投机验证 + 贪心锚定：每个 small block 用一次
  `[N, T]` 批量前向验证 N 个候选画布，min-prob 选中 winner 后一次性提交
  整个 small block（跳过剩余重掩码迭代），贪心锚定臂质量逐位==baseline
  （AR022：posthoc gap 0.0、解码长度 34/66 恢复）。
- **两者互斥**：AR033 的 `FastDLLMSamplerConfig.__post_init__` 在
  `history_cache=True` 且 `num_block_candidates > 0` 时构造期
  `ValueError`（投机路径在缓存计算策略上「未定义」）。AR033 srs §2 显式把
  「折叠谱系与缓存的组合机制」列为 out of scope，仅做正交性 sidecar 记录。

### 1.2 机会

在缓存路径下，每个 small block 仍需 ~`small_block_size` 次切片前向（每
迭代解 ~1 token）才能提交。投机验证把这 ~8 次**串行前向**压成**1 次批量
验证 + 1 次草稿前向**：一次 `[N, sb]` 批量前向共享同一份历史 KV（按 batch
维展开），winner 一次性提交整块。其收益是双重的：

1. **launch 数**：每块 `2·S + 1` 次前向（S=4 → 9）vs 缓存臂 ~`2·S·sb + 1`
   （~33）——WDDM/小内核主机上 launch 与同步是主开销（AR016/AR020 的既有
   结论）；
2. **token 行数**：`N·(sb+1)` 行/验证 × S + 块提交 vs `(sb+1)` 行 × 迭代数
   ——历史 KV 只读上下文，行数也显著下降。

且历史 KV 在跨块面是**构造级精确**（AR033 P3），批量候选行共享同一份
历史 KV 不引入新的近似面——质量面应与贪心锚定 AR022 一致（gap 0.0、
解码长度恢复）。

### 1.3 目标

1. **机制**：`FastDLLMSampler` 新增组合模式——`history_cache=True` 且
   `num_block_candidates>0`（贪心锚定）时，small-block 验证走
   「历史 KV（batch 展开）× N 候选批量验证」，winner 一次性提交；默认
   `num_block_candidates=0` 时 AR033 路径逐字节不变。
2. **证据**：四臂实机基准（A 非缓存基线 / B2 AR033 缓存臂 / D 缓存×投机
   组合臂 / C 官方原生 `batch_sample` report-only），预承诺单次 + 冻结
   判定式 + 多块/早停质量门。
3. **可复现**：RERUN 服务器节（命令 + 资源表 + 判定语义），README 全量
   更新，阶梯新档。

## 2. 需求范围

**In Scope：**

- `history_cache=True` + `num_block_candidates>0` 组合的**定义与实现**：
  草稿前向（`[1, sb+1]` 或 `[1, sb]` 切片）→ N 候选构造（复用 AR022 的
  冻结种子公式与贪心行）→ `[N, sb(+1)]` 批量验证（历史 KV batch 展开，
  `update_past_key_values=False`）→ min-prob winner → 一次性提交 small
  block → 既有 commit+advance 前向。
- 配置合同：lifting 构造期互斥；组合模式要求 `folded_model is None`
  （原生批量路径，非折叠）；`history_small_block_cache` 在组合模式下
  显式 `ValueError`（v1 只做跨块精确面，块内缓存的 N 行 partial 留后续）。
- 计数器/账本：`history_cache_stats` 增补 `spec_draft` / `spec_verify`
  两类（默认 0，向后兼容）；H3 结构面用整数账本恒等式。
- fp32 tiny 真实 block-causal fixture（AR033 fixture 复用/扩展）上的
  位级与语义 UT：缓存关 vs 缓存开的贪心输出一致、组合臂的提交语义、
  forward 计数恒等、默认关零回归、非法组合 fail-fast。
- 四臂 bench 脚本（AR033 模板）：冻结 H1-H4、`--quick` 独立工件、
  覆盖保护、`(arm, prompt_index)` 复算、C 臂 blocked 降级零伪造。
- strawman 口径沿用 AR033：阶梯 baseline = 非缓存 block-diffusion 基线臂。

**Out of Scope：**

- 块内 small-block 缓存 × 投机（B2×投机，N 行 `replace_position` 的近似
  漂移面）——v1 只做跨块精确面；路线图记录。
- CUDA graph × 缓存/投机（动态 KV 增长与图静态缓冲的合同冲突）。
- LLaDA/Dream（双向掩码，无历史前缀精确性）。
- 不确定性排序验证（AR031 A6）在组合臂上的接线（保留 sidecar 可行性，
  不做正式档）。

## 3. 功能需求

### 3.1 FR1 组合模式（`history_cache` × `num_block_candidates`）

**描述：** `history_cache=True` 且 `num_block_candidates>0` 时，采样循环
进入「缓存 × 投机」组合分支：每个 small block 先做一次草稿切片前向取
`top_k` 候选源，构造 N 个候选画布（row 0 = 贪心假设，AR022 冻结语义），
再做一次 `[N, 活动切片]` 批量验证前向（历史 KV 按 batch 维度展开、只读），
按 min-prob 选中 winner 并一次性提交该 small block 的全部掩码位；随后执行
既有的 commit + advance 前向。

**触发条件：** `FastDLLMSampler.sample` 且 `history_cache=True`、
`num_block_candidates>0`、`folded_model is None`。

**期望行为：** 生成 token 序列与贪心锚定 AR022 语义等价（fp32 位级一致见
FR3）；每 small block 恰 2 次前向（草稿 + 验证）+ 每块 1 次 commit；
`history_cache_stats` 的 `spec_draft`/`spec_verify` 计数与几何恒等。

**异常处理：**
- `history_cache=True` + `num_block_candidates>0` + `folded_model is not None`
  → `ValueError`（组合臂走原生批量，不走折叠）；
- 组合臂 + `history_small_block_cache=True` → `ValueError`（v1 未定义）；
- 原生模型不接受 cache kwargs（AR033 既有探测）→ `RuntimeError`。

**验收标准：**
- Given fp32 tiny 缓存能力模型 + 多块生成, When 组合臂与缓存臂（N=0）在
  贪心锚定下运行, Then 输出 token 序列逐位一致（`torch.equal`）且组合臂
  每 small block 前向计数恒等。
- Given 默认配置（`num_block_candidates=0`）, When 任意既有 AR033 UT 运行,
  Then 行为逐字节一致（零回归合同）。

### 3.2 FR2 批量候选验证的 KV 展开合同

**描述：** 组合臂的验证前向把 batch-1 历史 KV 展开到 N 行（`expand`/
`repeat_interleave` + `contiguous`），`update_past_key_values=False`
（历史只读，不写回）；winner 的 KV 提交由既有 commit 前向在 `[1, block]`
上完成（不取验证前向的 KV——避免行选择的 KV 布局依赖）。

**验收标准：**
- Given 历史 KV 与 N 候候选, When 验证前向（N=1）与缓存臂同位置切片前向
  对比, Then logits 逐位一致（fp32 fixture）；N>1 时每行独立等价于各候选
  单独前向（逐位或 ≤1 ULP，按 fixture 记录）。
- Given 验证前向, When 结束, Then 输入历史 KV 未被修改（
  `torch.equal` 快照对比）。

### 3.3 FR3 精确性与语义证据

**描述：** 精确性分两面：
(a) fp32 tiny 真实 block-causal 模型（携带同构 cache kwargs 接口）上，
缓存臂（N=0）与组合臂（N>0，贪心锚定）在多块、batch 1、pad 排除、forced
区域下逐位一致；
(b) 真实 checkpoint（bf16）上贪心 token 一致率 + 解码长度恢复 + posthoc
gap（质量门）。

**验收标准：**
- Given fp32 tiny 模型, When 缓存 N=0 vs 组合 N=4 贪心, Then `torch.equal`
  全绿（≥4 形状点/几何）。
- Given 真实 checkpoint 四臂 reps=30, When 组合臂 vs 非缓存基线臂, Then
  H1 质量面全过（gap ≤1e-3、decode 长度全臂一致、token rate ≥0.98）。

### 3.4 FR4 四臂实机基准（预承诺单次）

**描述：** `scripts/ar034_kv_speculative_bench.py`：四臂（A 非缓存基线 /
B2 `history_cache=True, history_small_block_cache=True`（AR033 最快缓存臂）/
D 组合臂（`history_cache=True, num_block_candidates=4,
candidate_include_greedy=True`）/ C 官方原生 `batch_sample` report-only），
官方 v2 几何 block=32/sb=8，reps=30 预承诺单次；判定 H1-H4 冻结先于数据
收集；repro 双跑一票否决；覆盖保护；`--quick` 独立工件。

**验收标准：**
- 四臂工件全要素可溯源（token 账本、p50、判定输入、环境）。
- H2 同时给出 `p50(A)/p50(D)` 与 `p50(B2)/p50(D)`（组合臂须胜两个参照）。
- 时间口径沿用 #39 + AR033（同 run A/B 内部有效，跨工件绝对值不可比）。

### 3.5 FR5 文档与阶梯

**描述：** README 全量更新（创新表新行、机理节、最新结果头条、阶梯新档、
质量基线、快速上手、路线图），AGENTS 新条目（组合合同 + KV 展开律），
CHANGELOG、RERUN 新步骤、ALGORITHM/EXPERIMENTS 简节、design 收口。

**验收标准：**
- 所有数字由工件 JSON 重算；冻结工件/判定函数零回改；strawman 口径与
  AR033 一致（baseline = 非缓存 block-diffusion 基线臂）。

### 3.6 FR6 复现文档（服务器节）

**描述：** RERUN 新增步骤：AR019 venv + 权重重下 + AR033/AR034 脚本、
四臂命令、资源表（VRAM/墙钟/峰值）、判定语义、quick 先行。

**验收标准：**
- 仅凭仓库 + RERUN 步骤即可在 ≥24GiB Linux/TCC 服务器复跑四臂正式面。

## 4. 非功能需求

- NFR1 **零回归**：`num_block_candidates=0` 时 AR033 路径逐字节不变；
  全量 pytest 基线（1480 passed）不降。
- NFR2 **可观测**：组合臂的草稿/验证/提交前向计数可通过只读探针取得。
- NFR3 **显存**：历史 KV 的 N 行展开瞬态不得导致 16GB WDDM 主机 OOM
  （真实面 <1GB 增量目标）。
- NFR4 **可复现**：判定函数冻结、预承诺单次、repro 双跑。

## 5. 约束与假设

- 组合臂走**原生批量**前向（非折叠），因此 `folded_model` 必须为 None。
- 跨块历史 KV 精确性前提（AR033 P3，真实 checkpoint maxdiff=0.0）。
- 贪心锚定语义与 AR022 完全一致：row 0 = 草稿 top-k 首项（argmax），
  min-prob 公平竞争；候选可落败。
- 真实面质量证据覆盖首块早停几何（34/66 token，AR022/031/032/033 共同
  边界；D4 多块塌缩限制外推）。
- 本机 16GB WDDM 主机的计时受宿主状态噪声影响，比值同 run 内部有效。

## 6. 术语说明

| 术语 | 说明 |
|------|------|
| 历史 KV | prompt + 已提交块的 `past_key_values`（只读上下文） |
| 活动切片 | 当前 small block（含左边界行）或短尾块的 block 对齐切片 |
| 组合臂 D | `history_cache × num_block_candidates>0`（贪心锚定） |
| 贪心锚定 | AR022：候选 row 0 = 草稿 argmax，min-prob 公平竞争 |
| 非缓存基线臂 | AR016-AR033 阶梯的固定参照：全画布重算的官方路径 |

## 7. 任务清单

见 `tasks.md`。
