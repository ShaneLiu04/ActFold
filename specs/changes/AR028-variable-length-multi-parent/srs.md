# [AR028] 变长多 parent 批量折叠 — 需求规格说明书

**AR028：变长多 parent 批量折叠（variable-length multi-parent batch folding）**

| 字段 | 内容 |
|------|------|
| AR | AR028 |
| 关联 SR | 无（延续 AR025 Out-of-Scope 明文遗留项：同批异长候选的 pad-to-max + per-row 有效长度 mask） |
| 前置 | AR025（多 parent 批量折叠，已归档）；AR015（batched-branch 契约，#41）；AR020（批量缓存写，#42）；AR021（生产生成循环折叠，#43） |
| 状态 | 需求门控已 PASS（第 10 轮） |
| 真实面 | 按用户总纲延后（合成面 + UT 交付） |

## 1. 背景与目标

AR025 交付了多 parent 批量折叠（`parent_branch_ids` per-row 父装配 +
双链 lockstep `sample_joint`），但把**同批异长**显式列为 Out-of-Scope：
批量画布的每一行必须等长（`[sum(N_c), T]` 单一 T）；`sample_joint` 对不等长
prompt 直接抛 `ValueError`（C2：长度必须相等；截断只发生在基准层脚本，
不在生产入口）；生成循环 `folded_generate_batched` 亦只接受等长候选
（其合并路径为 `_run_folded_forward_multi`，合并谓词为
`_multi_chain_merge_supported`；逐链 AR015 路径为
`_run_folded_forward_batched`，两函数均无 attention_mask 通道）。
真实工作负载里 prompt 长度天然不等、投机候选的有效区间也可能不同——
强制等长会让大量调用退回串行或直接拒绝。

本 AR 的目标：在**不改变缓存格式、默认路径逐字节兼容**的前提下，让多
parent 批量折叠支持**右 pad 到 T_max + per-row 有效长度**，并把该能力接到
`sample_joint`（opt-in）与 `folded_generate_batched` 谓词上。

组件详设目录（`specs/component-detail-design/`）当前缺失——本 AR 以
`AGENTS.md`（#41/#42/#43/#47 契约）与已归档 AR025 spec 作为领域基准。

## 2. 需求范围

**In Scope：**

- 机制面：`ManualFoldedForward.forward` + `FoldedTransformerLayer.forward`
  新增 `valid_lengths: Sequence[int] | None` 与
  `parent_valid_lengths: Sequence[int] | None`（批量面专用）+
  **专用 per-row attention mask 通道 `row_attention_mask`**
  （`[N, T]` bool，**仅 plural 多 parent 面允许**；既有 `attention_mask`
  的 `[T,T]` 语义逐字节不变，`row_attention_mask=None` 维持旧语义）；
  pad 感知的 per-row gate / 构造性发散 / 父前缀捐赠 / 发布 / 统计排除。
- 采样面：`sample_joint(..., pad_to_max=False)`（**仅 LLaDA 实现**；
  Dream 同名签名接受该 kwarg，joint-armed 时抛 `ValueError`、
  off-switch 时 no-op——两族签名一致、行为按族定义）——opt-in 时不等长
  prompt 右 pad 到 T_max，不丢信息；默认仍抛 `ValueError`（AR025 逐字节
  兼容）。
- 生成循环：`folded_generate_batched` 的**合并路径谓词**
  `_multi_chain_merge_supported` 放宽为变长受支持（调用方提供
  per-candidate 真实长度；满足条件才合批，否则逐链回退）；
  `_batched_forward_supported` 逐字节不变。
- 验收：UT 矩阵 + 合成变长几何基准（CPU，冻结判定式 S1/S2/S3）。
- 文档五件（README/AGENTS/GUIDE/CHANGELOG/RERUN，含真实面延后配置）。

**Out of Scope（明示）：**

- **左 pad / 后缀有效区 / 位置编码重排**：与现有
  `compute_position_ids`/rotary pe 钉定面冲突面大；机制面唯一形态为
  前缀有效（右 pad）。**Dream 的 `pad_to_max` 因而明确不在本 AR**：
  Dream 原生画布是左 pad（prompt 右对齐、有效区为后缀），与机制面形态
  不一致；**joint-armed 时** Dream 上 `pad_to_max=True` 抛 `ValueError`
  （off-switch 逐链分发不校验），后缀有效面留后续 AR。
- **AR015 单父（singular）批量面的变长支持**：该面保留既有 AR007 截断
  语义，本 AR 零改动；变长机制只定义在 plural 多 parent 面。
- **非 causal 模型的变长生成循环面**：`_run_folded_forward_multi` 与
  `_run_folded_forward_batched` 均无 attention_mask 通道，右 pad 前缀
  安全仅对 causal 成立；该面留后续 AR（机制/采样面已由 `row_attention_mask`
  覆盖非 causal 等价）。
- **cache 内部长度元数据**：长度一律由调用方以宿主元数据传入
  （`valid_lengths`），零设备回读、缓存格式不变。
- **图静态体多 parent / 变长**（AR026 冻结面）：任何携带 `valid_lengths`、
  `parent_valid_lengths` 或 `row_attention_mask` 的 plural 调用一律 eager，
  图路径不扩展。
- **真实 checkpoint 正式数据**：本 AR 的**用户总纲明确授权**（"不做所有
  大型实验"）对 AGENTS Experimental Validation Rule 的真实 checkpoint
  强制项做范围裁剪；命令/配置/判定式全部写入 RERUN，回填前不得在任何
  文档宣称真实面有效性。
- 变长父的 **AR007 截断复用**（父比子**画布**长 `T_p > T`）：沿 AR025 规则，
  长父仍整行降级（不做截断捐赠）；`T_p > v_i` 但 `<= T` 的父**不降级**，
  gate 只取 `[0, v_i)`（见 FR2 验收 5）。

## 3. 功能需求

### 3.1 FR1 per-row 有效长度 API

**描述：** plural 多 parent 批量折叠面新增逐行有效长度、父有效长度与
逐行 attention mask 三个可选输入；默认路径（全 `None`）逐字节兼容。

**触发条件：** plural 多 parent 批量调用（`parent_branch_ids` 非 None）
且 `valid_lengths` / `parent_valid_lengths` / `row_attention_mask` 任一非 None。

**期望行为：**
- **接口**：`ManualFoldedForward.forward` / `FoldedTransformerLayer.forward`
  新增 keyword-only `valid_lengths: Sequence[int] | None = None`、
  `parent_valid_lengths: Sequence[int] | None = None`、
  `row_attention_mask: torch.Tensor | None = None`；长度参数按批行序
  （`[N]`），mask 为 `[N, T]` bool（1=有效 key、0=pad）；三者默认 None
  时**不向底层 layer 传递任何新 kwarg**；
- **行有效区**：行 i 的有效区为 `[0, v_i)`，`[v_i, T)` 为 pad
  （右 pad 唯一形态）；
- **父证据有效区**：行 i 的父有效区是 `[0, v_p,i)`（`v_p,i` 来自
  `parent_valid_lengths[i]`，缺省=父张量长度）。gate 比较区间 =
  `[0, min(v_i, v_p,i))`；`[v_p,i, v_i)` 构造性发散；`[v_i, T)` pad 区
  构造性发散。**没有该通道，整画布发布会让父 pad 列在下一步的
  `T_p >= v_i` 情形下被误当有效证据**（T_p 是画布长度而非真实长度）；
- **per-row attention mask 通道（非 causal 等价的必要条件）**：新增专用
  keyword `row_attention_mask`（bool `[N, T]`，1=有效 key、0=pad；
  **仅 plural 多 parent 批量面允许**，其他面传非 None ⇒ `ValueError`）；
  语义 = 行 i 的有效 token 不得 attend 到 key 位置 `>= v_i`（pad 列）；
  专用 kwarg 与既有 `attention_mask`（`[T,T]`，语义逐字节不变）并存：
  **两者同时给出时，行 i 的允许 key 集 = `attention_mask` 的允许集
  ∩ `row_attention_mask[i]`（交集合取，顺序无关；design 记录实现）**；
  `None` 维持旧语义。层必须把该 mask 传给底模（remote 模型接受度由
  design 验证并记录；若模型只接受 `[T,T]`，则由 design 定案等价实现：
  逐行计算 additive mask 或行级广播），**不得静默忽略**；
- **非 causal 等价的验收口径**：有效区的 per-token logits/hidden 与
  "该行未 pad 的独立运行"在 `allclose(1e-6)` 内（masked softmax 归约
  长度不同，跨画布不承诺逐位）；**同一 T_max 画布上的 1 行参考**则
  nonsplit 逐位 / split `allclose(1e-6)`（shape 相同的批不变性面）；
- `valid_lengths` / `parent_valid_lengths` 为 `None` ⇒ 全画布有效
  （等价于旧行为，附加零 kwarg，逐字节兼容）；
- 校验（入口，宿主元数据）：长度 == 批行数；每行 `1 <= v <= T`
  （父侧 `1 <= v_p <= T`，且层内以父条目张量长度 `T_p` 为上界
  `clamp(v_p, T_p)`——`v_p > T_p` 时取 `T_p`，不得越过父行）；仅
  **plural 多 parent 批量面**允许（串行面/singular 面传入 ⇒ `ValueError`）；
  与 `branch_ids`/`parent_branch_ids` 同长；`parent_valid_lengths` 无
  `parent_branch_ids` ⇒ `ValueError`；`row_attention_mask` 形状非
  `[N,T]` 或出现在非 plural 面 ⇒ `ValueError`；
- **图路径一律 eager 回退**：任何携带 `valid_lengths`、
  `parent_valid_lengths` 或 `row_attention_mask` 的 plural 批量调用
  不得进入图静态体——图静态体不感知 pad，必须走 eager 批量体；
  一次性 `RuntimeWarning`（**单一 warn-once 身份**：复用既有
  `_multi_parent_graph_warned` 机制，不新增第二个警告源；UT 计 `1` 次）。

**异常处理：** 任何校验失败 ⇒ `ValueError`（消息含具体行号/取值）。

**验收标准：**
- Given 批量调用未传 `valid_lengths`/`parent_valid_lengths`/
  `row_attention_mask`，When 执行 folded forward，Then 输出与 AR025 实现
  逐位相同（`torch.equal`），且底层 layer 调用 kwargs 中不出现该三参数。
- Given `valid_lengths`（或 `parent_valid_lengths`）长度与行数不符 /
  含 0 或 >T / `parent_valid_lengths` 无 `parent_branch_ids` /
  `row_attention_mask` 形状非 `[N,T]` 或出现在非 plural 面，When 调用，
  Then 抛 `ValueError` 且缓存与模型状态不被修改（入口校验先于任何前向）。
- Given 携带 `valid_lengths` 的 plural 批量调用，When 前向，Then 不触发
  图捕获/重放（走 eager）：**与显式 `use_cuda_graph=False` 的同输入输出
  逐位一致，且图 runner 调用计数为 0（Mock/探针断言未命中）**；CUDA
  可用时验证真实捕获面，无 CUDA 时以能力探测/Mock 验证分支选择并诚实
  跳过真实捕获；**警告契约：fresh 实例上恰好一次，同一实例生命周期内
  至多一次**（仅当 `use_cuda_graph=True` 且携带长度/mask 时触发，与现有
  `_multi_parent_graph_warned` 一致；`use_cuda_graph=False` 不产生该警告；
  "先旧后新"顺序下不重复警告，UT 钉死）。
- Given 同时给出 `attention_mask`（`[T,T]`）与 `row_attention_mask`，When
  前向，Then `row_attention_mask=None` 与 `[T,T]` 面逐字节一致；两者同时
  给出时行为等于允许集交集（用小模型/探针断言行 i 看不到
  `attention_mask=0` 或 `row_attention_mask_i=0` 的 key）。

### 3.2 FR2 pad 感知折叠（gate / 发散 / 父前缀）

**描述：** pad 位置不作为可复用证据，有效区严格按 per-row 长度判定
（**仅 plural 多 parent 面；AR015 单父批量面行为不变**）。

**触发条件：** 任一长度参数使子行或父行有效区小于画布
（`v_i < T` 或 `v_p,i < T`）。

**期望行为：**
- 行 i 的 gate 相似度比较区间 = `[0, min(v_i, v_p,i))`（pad 列与父 pad 列
  都不参与比较）；
- pad 列（`>= v_i`）与父 pad 列（`[v_p,i, v_i)`）**构造性**
  `stable_mask=False`（在 gate 之后、forced AND 之前强制置 False；与
  metric 无关的 no-leak，沿 AR025 DD2 模板），即这些位置恒走子分支重算；
- 父张量短于行 i 的有效长度（`T_p < v_i`）且未给 `parent_valid_lengths`
  时：父前缀 `[0, T_p)` 参与 gate，其余有效列 `[T_p, v_i)` 构造性发散
  （append-only 生成循环形状，沿 AR025 `_fetch_parent_gate_row`）；
  给了 `parent_valid_lengths` 时以 `v_p,i` 为准（见上条）；
- 父激活**长于整画布** T（`T_p > T`）时：该行整行降级（沿 AR025 多
  parent 规则，不做 AR007 截断；**singular 单父面仍走 AR015 的既有截断，
  本 AR 不改**）；
- 父缓存 batch != 1 仍为硬 `RuntimeError`（#41 逐行契约，不变）。

**异常处理：** 父条目缺失/形状不匹配 ⇒ 该行整行重算（不炸批，沿 AR025）。

**验收标准：**
- Given 两行变长（`v=[T, T/2]`）、同一父与**逐行 `row_attention_mask`**
  （`[N,T]`，行 i 的 `[v_i, T)` 置 0），When folded forward，Then **有效区**
  逐列等于"该行在同一 T 画布上的 1 行参考（`branch_ids=[br_i]`、
  `parent_branch_ids=[p]`、同一 mask 行、`valid_lengths=[v_i]`、
  **`parent_valid_lengths` 与目标同值**——参考父为全有效，或按 `v_i`
  重建/对齐的切片，由 design pin 具体构造）"的结果
  （nonsplit `torch.equal` / split `allclose(1e-6)`——shape 相同的批
  不变性面）；与"未 pad 的独立运行（画布 `T=v_i`）"比较按
  `allclose(1e-6)`（跨画布归约长度不同，见 FR1 验收口径）；**pad 区**
  逐列等于"同一行在同一 T 画布、**同一 `row_attention_mask` 行**、
  **无父（cache miss）或 `forced_divergent` 全 True** 的整画布重算"的
  对应列（该参考调用刻意不提供可复用父，确保是全量重算而非门控复用；
  参考仍携带同一 mask 行——非 causal 下缺 mask 会 attend 到 pad 列，
  不可比）；同时父缓存 pad 列（若存在）内容刻意与子重算不同，断言
  pad 区输出 **!=** 父缓存对应列（child pad 未被复用的反例判别）；
  长行同档。
- Given **父缓存画布 = T 但父真实有效长度 `v_p < v_i`**（`parent_valid_lengths`
  通道；**因果/前缀稳定合成构造**，且父缓存的 pad 列内容刻意与子重算
  不同——否则该用例不可判别），When 下一步子行以 `v_i` 折叠，Then：
  `[0, v_p)` 按 gate 复用；`[v_p, v_i)` **构造性发散**——同时满足
  ① 输出 == 无父整画布重算的对应列，② 输出 **!=** 父缓存 pad 列的
  对应值（两条断言共同证明未复用父 pad 列）。
- Given 目标行与参考行使用不同的 `row_attention_mask`，When 对比，
  Then 明确判为构造错误（fixture 断言）；契约要求目标与参考 mask 在
  公共有效区一致（层内不从 `valid_lengths` 自动生成 mask）。
- Given 父长度 `T_p < v_i` 且未给 `parent_valid_lengths`，When 执行，
  Then `[0, T_p)` 可按门控复用、`[T_p, v_i)` 恒重算，输出与全量重算
  逐位一致（**因果/前缀稳定合成构造**：父前缀与子前缀隐藏态逐位相同；
  双向扩散 fixture 不做该逐位断言，只做 stable 语义断言）。
- Given **子行满画布 `v_i = T` 但父真实有效长度 `v_p < T`**（仅给
  `parent_valid_lengths`），When 执行，Then gate 只比较 `[0, v_p)`，
  `[v_p, T)` 构造性发散（触发条件覆盖父 pad 面；输出与无父重算逐位
  一致）。
- Given 父长度 `T_p >= v_i`（含 `T_p == v_i` 与 `T_p > v_i` 但 `<= T`），
  When 执行，Then gate 比较区间严格为 `[0, min(v_i, v_p,i))`（无
  `parent_valid_lengths` 时 `v_p,i = T_p`），pad 列不参与比较。

### 3.3 FR3 pad 感知发布

**描述：** 合批发布保持整画布语义，下游按 `valid_lengths` 使用。

**触发条件：** 带 `valid_lengths` 的批量前向完成后发布子激活。

**期望行为：**
- 发布**整画布** `[N, T, H]`（不裁剪 pad 列）；缓存条目布局与逐行
  发布完全一致（`buffer[:final _counts]`、`_counts`、逐层键顺序、
  `num_entries` 均等价，#42 契约）；
- 发布路径不读取 `valid_lengths`/`row_attention_mask` 做**裁剪或重排**
  （长度只影响前向的 gate/发散语义，见 FR2；统计口径见 FR4，eager 面经
  `record` 生效，`record_precomputed` 仅图发布面使用）；
- eviction 逐条按 `put` 的裁剪点执行（#42），发布顺序为子行序；
- 发布路径保持零 host sync（不引入 `.item()`/`.tolist()`/`.cpu()`）；
- 自一致性：整画布发布 + 下游按 `valid_lengths` 使用，与"逐行发布 +
  逐行下游"语义等价（证明缓存契约未被 pad 面破坏）。

**异常处理：** 发布形状与父/缓存契约不匹配 ⇒ 沿用既有 `ValueError`/`RuntimeError`
（不新增吞错路径）。

**验收标准：**
- Given 变长批量前向（含 pad 行），When 发布，Then cache 状态与
  `chains·N` 次单行发布**逐位等价**：`buffer[:final _counts]`
  `torch.equal`、`_counts` 相等、逐层键顺序一致、`num_entries` 相等。
- Given 发布携带 `row_attention_mask`，When 检查缓存内容，Then 缓存值
  仅由前向 hidden 决定（mask 不进入缓存布局/键），且与不传 mask 的
  发布在相同 hidden 下逐位一致。
- Given 单步变长折叠，When 从发布的缓存回读父，Then 回读值与发布时的
  hidden 逐位一致（整画布无裁剪）。

### 3.4 FR4 统计口径（stability profiler）

**描述：** 稳定性统计只覆盖有效 token，pad 列不计入分子/分母。

**触发条件：** 带 `valid_lengths` 的批量发布/记录。

**期望行为：**
- **eager 面（本 AR 变长路径）**：`StabilityProfiler.record` 的每 branch
  `num_tokens` = 该行**有效 token 数** `v_i`（不是画布 `T`）；
- **`record_precomputed`（图发布面）**：保持现有整画布语义不变（变长调用
  被 FR1 强制 eager、不进入该路径）；若未来接入需同样按 `v_i` 口径扩展，
  本 AR 不实现；
- 稳定计数/比例只在 `[0, v_i)` 上计算；pad 列不计入 stable 或
  divergent 计数；
- `v_i == T` 时与旧行为逐位一致（全画布）；
- 记录路径保持零 host sync（沿用 AR018 `record_precomputed` 的
  snapshot 语义，不新增读回）。

**异常处理：** 统计输入行数与 `valid_lengths` 不一致 ⇒ `ValueError`
（入口已校验，统计层不吞错）。

**验收标准：**
- Given 变长批量发布（`v=[T, T/2]`），When 记录统计，Then 每 branch
  的 `num_tokens` 分别等于 `T` 与 `T/2`，稳定计数只覆盖有效区。
- Given `v_i == T` 的行，When 记录，Then 与不传 `valid_lengths` 的
  同输入记录逐位一致。
- Given `valid_lengths` 与记录行数不一致，When 调用，Then `ValueError`
  （错误消息含行号/取值）。

### 3.5 FR5 sample_joint 变长 prompt（opt-in，LLaDA 面）

**描述：** 双链 joint 投机为不等长 prompt 提供 pad-to-max 路径（不丢信息），
默认行为与 AR025 完全一致。

**触发条件：** `sample_joint(prompt_ids_list, folded_model, pad_to_max=True)`
且两 prompt token 长度不同（LLaDA 族）。

**期望行为（按 joint 开关分面，AR025 代码事实口径）：**
- **joint-armed 判定** = `num_block_candidates>0` **且** `folded_model` 非空
  **且** 模型支持 `branch_ids` **且** `multi_parent_batch` 开（四项全成立；
  沿 AR025 `sample_joint` 的实际分支条件）；
- **joint-armed 路径**：
  - `pad_to_max=False`（默认）：**不等长 prompt 抛 `ValueError`**（AR025
    原语义逐字节不变；AR025 C2 是"必须等长/否则拒绝"，截断只发生在基准
    层脚本，`sample_joint` 本身无截断路径）；等长路径不变；
  - `pad_to_max=True`（LLaDA）：画布 `T_max = max_c(prompt_len_c) +
    num_tokens`，每链**右 pad** 到 `T_max`；填充 id 沿用既有
    `build_right_padded_canvas` 的 `eos_token_id` 语义（valid 区输出不受
    影响，仅保证复现一致），**新增扩展列用同一 id**；每链的
    验证批量调用携带 `row_attention_mask`（`[N,T_max]`，行 c 的
    有效区 1 / `[v_c, T_max)` 置 0）、`valid_lengths` 与
    `parent_valid_lengths`；
  - **draft fold-context 必须按链重键**：每链的串行 draft 使用
    `_spec_ctx_for(xs[chain], prompt_len_chain)`（或等价 per-chain key
    `(T_max, prompt_len_c)`），**不得复用链 0 的 ctx/mask**——否则短链的
    pad 列会被当作有效 key，draft 激活与 solo 不等价；
- **长度参数的行序与取值（可复算定义）**：joint 批量按 **chain-major**
  展开——每个存活链贡献 `N` 行（`N = num_block_candidates`），行序 =
  `[chain0 的 N 行, chain1 的 N 行, ...]`；长度参数按同一行序展开：
  首窗口两链均存活时 `valid_lengths = [p1+L_new]*N + [p2+L_new]*N`，
  `parent_valid_lengths` 同构（`L_new = num_tokens`；joint face 的父草案
  pad 边界 = 自身 `valid_lengths`）；**C5 提前结束链被移除、或某链零预算
  窗被跳过时，行数与对应区间按实际存活窗口动态缩减**（不写死两链）；
  `touch` 语义（**语义级硬门 + 数值级观测**）：每条链的**输出 token**
  与"该链单独以自身 prompt 长度跑 solo `sample()`"**相等**（greedy
  确定性路径；design 须 pin 有明确 logit margin 的 fixture，避免近
  平局数值翻转）；有效区 logits 的 `allclose(1e-6)` 为 report-only
  （masked softmax 归约长度不同，跨画布不承诺逐位）；种子公式无链
  分量；Dream 族抛 `ValueError`（显式范围裁剪）；
- **off-switch 路径**（`joint_armed == False`）：走 AR025 的逐链
  `sample()` 分发，`pad_to_max` 是 no-op（逐链分发无需 pad）。其中：
  - `num_block_candidates == 0` 或 `folded_model` 合法支持 `branch_ids`
    ⇒ 不等长 prompt **合法**（solo `sample()` 本就不要求等长）；
  - `num_block_candidates > 0` 但 `folded_model` 缺失/不支持 `branch_ids`
    ⇒ 沿用既有的 solo 投机守卫（`ValueError`，与 AR023/AR025 逐字节一致，
    不因本 AR 改变）；
- 等长 prompt 时两路（pad_to_max 开/关）**输出 token 相同**；同画布
  logits nonsplit 逐位 / split `allclose(1e-6)`（构造性：无 pad）。

**异常处理：** 入口守卫其余不变（N>0/branch_ids 能力/cfg_scale/
temperature/stochastic transfer）；joint-armed 且 `pad_to_max=True` 且
非 LLaDA ⇒ `ValueError`（Dream 同名签名接受该 kwarg、armed 拒）；off-switch
不校验（逐链分发、no-op）。
**draft 串行面注记**：每链 draft 串行前向不携带长度参数（FR1 禁串行/
singular 面传参），但**必须携带该链自己的 per-chain ctx/mask**
（见上）；draft 面的 pad 列**不在本 AR 的发散保证范围内**
（`_build_unit_forced` 只覆盖单元窗口，不覆盖右 pad 区）——最终 no-leak
由 joint 验证批量的 `parent_valid_lengths` 保证（验证面的 pad 列恒
重算）；spec 不承诺 draft 面按有效长度裁 gate，但承诺**每链 ctx 独立**
（等价性由验收钉死）。

**验收标准：**
- Given joint-armed、两条不等长 LLaDA prompt、`pad_to_max=False`，When
  执行 sample_joint，Then 抛 `ValueError`（与 AR025 逐字节同消息面）。
- Given joint-armed、两条不等长 LLaDA prompt 与 pad_to_max=True，When 执行
  sample_joint，Then 每链**输出 token 等于逐链 solo `sample()`**（语义
  硬门；fixture 的 logit margin 由 design pin）；比较区严格为
  `joint_sequences[chain][: p_c + L_new]`（截去 `T_max` 画布的右 pad 尾，
  与 solo `[1, p_c+L_new]` 同形）；有效区 logits 对未 pad 独立运行的
  `allclose(1e-6)` 为 report-only；且 joint 批量调用携带
  按 chain-major 行序展开的
  `valid_lengths == [p1+L_new]*N + [p2+L_new]*N` 与同构
  `parent_valid_lengths`（`L_new = num_tokens`，N=num_candidates；
  存活链动态缩减）与**逐行 `row_attention_mask`**（行 c 的
  `[v_c, T_max)` 置 0）。
- Given joint-armed 不等长 LLaDA prompt，When 逐链 draft，Then 每链使用
  自己的 fold ctx（键含 `prompt_len_c`）；构造"复用链 0 ctx"的变异实现
  会使短链 draft 与 solo 不等价（**变异测试/UT 断言 per-chain ctx 调用
  计数与 key**），正确实现下 draft 分支激活等价。
- Given joint-armed 且某链在首窗提交 EOS 被移除（C5），When 执行，
  Then 余链的 `valid_lengths`/`parent_valid_lengths` 行数缩减为 `N`
  且值不变（动态行序契约）。
- Given **off-switch 且 `num_block_candidates == 0`**、两条不等长 prompt，
  When 以 `pad_to_max=True` 或 `False` 执行 sample_joint，Then 均逐链分发
  成功（不抛错），且两路输出逐位相同（pad_to_max 为 no-op）。
- Given off-switch 且 `num_block_candidates > 0` 但 folded_model 缺失
  **或不支持 `branch_ids`**，When 执行 sample_joint，Then 沿用既有 solo
  投机值守卫 `ValueError`（与 AR025 逐字节一致）。
- Given joint-armed 等长 prompt，When 分别以 pad_to_max True/False 执行，
  Then 两次**输出 token 相同**，同画布 logits nonsplit 逐位 / split
  `allclose(1e-6)`。
- Given joint-armed 不等长 prompt 的返回，When 检查 `SamplerOutput`，Then
  `sequences` 保持**整画布 `T_max`（不裁剪）**（与 `decode_final` 现状
  一致），验收在 `[:p_c+L_new]` 切片上比较（返回契约与比较区分别 pin）。
- Given joint-armed DreamSampler 与 `pad_to_max=True`，When 调用，Then 抛
  `ValueError`（显式不支持）。

### 3.6 FR6 生成循环变长谓词

**描述：** `folded_generate_batched` 的合批谓词接受变长候选，真实长度由
调用方显式提供。

**触发条件：** 候选集合带 per-candidate 真实长度（不同长度）、同画布 T
（右 pad 后）。

**期望行为：**
- **长度元数据契约**：`ChainInput` 新增逐链 `valid_length: int | None`
  （该链真实长度；None=旧语义），`BranchNode` 新增逐候选
  `valid_length`/`parent_valid_length: int | None`（None=旧语义；
  `folded_generate_batched` 以链长度填充候选行）；单链 `folded_generate`
  路径**显式忽略**这些字段（由内部 helper 层 UT 验证，不静默丢弃）；
  **右 pad 由调用方（上层生成循环）构造**后以 `[N,T]` 传入（按 `T_max`
  填充、不改变真实 token；**本面 pad id = 模型/分词器 `pad_token_id`，
  缺省回退 mask token id**——注意采样面（FR5）按
  `build_right_padded_canvas` 的 `eos_token_id` 语义填充，两面定源不同、
  各自 pin），批量入口只消费 `valid_lengths`/`parent_valid_lengths`，
  不自行 pad；
- **跨步长度推导（design 必须落成公式并 UT 断言）**：`L0` 为入口链的
  **真实有效长度**（不是右 pad 后的画布 `T_max`），第 `t` 步（`t=0` 起）
  候选的有效长度 = `valid_length = L0 + t + 1`、
  `parent_valid_length = L0 + t`（append-only 形态；实际取值以调用方
  传入的字段为准，框架不自行推导）；
- 谓词放宽为：**仅合并路径谓词 `_multi_chain_merge_supported` 泛化**
  （`multi_parent_batch=True`；`_batched_forward_supported` 保持原样、
  逐字节不变）；满足 ⇒ 单次 `[N,T]` 批量前向（携带两长度参数）；
- **不提供长度元数据且为等长集合** ⇒ 走 AR025 既有合并批量路径
  （逐字节不变，不是回退）；**变长/非右 pad 形态且不满足谓词** ⇒
  既有**逐链回退**（AR015 批量可用则批量，否则串行；`folded_generation.py`
  的 dispatch 语义），零行为变化；**父缓存 batch != 1 不是回退条件**——它是
  layer 侧硬 `RuntimeError`（#41/#47 契约不变，谓词不观测缓存内容）。

**异常处理：** 长度元数据非法（负值/大于画布/与候选数不一致）⇒
`ValueError`（入口）；缺失元数据走旧路径，不报错。

**验收标准：**
- Given 带 `valid_length` 的变长候选集合（`multi_parent_batch=True`、
  合并谓词命中、右 pad 同画布；**各行可有各自的父**，不要求同父；
  **前提：causal 模型右 pad 前缀安全**——`_run_folded_forward_multi`
  与 `_run_folded_forward_batched` 均无 attention_mask 通道，非 causal
  模型不在本 AR 范围；本面不传 `row_attention_mask`），When
  `folded_generate_batched`，Then 走批量路径，且**有效区**与"同一
  `T_max` 画布上的逐候选单行参考（causal：每行独立 `[1,T_max]` 调用 +
  相同长度参数）"逐位一致（nonsplit `torch.equal`；split
  `allclose(1e-6)`），**pad 区**等于无父/`forced_divergent` 全 True 的
  整画布重算对应列（与 FR2 同口径）。
- Given 未提供长度元数据的旧等长候选集合，When 执行，Then 谓词与行为与
  AR025 逐字节一致（`valid_lengths` kwarg 不出现；合并路径仍走批量）。
- Given 单链 `folded_generate` 的候选集合携带新长度字段，When 执行，
  Then 字段被**显式忽略**，行为与未携带逐字节一致（不静默走批量；
  在内部 helper 层注入字段做 UT）。
- Given 入口链长 `L0` 的候选，When 执行第 `t` 步，Then 传入的
  `valid_length == L0+t+1` 且 `parent_valid_length == L0+t`（公式在
  helper 层 UT 断言；调用方字段优先）。

### 3.7 FR7 基准与验收（预注册）

**描述：** 合成面冻结判定式 + 真实面配置延后交付。

**触发条件：** 新脚本 `scripts/ar028_variable_length_bench.py`。

**期望行为：**
- 合成面几何网格**精确 pin（12 点，含双构造）**，T = 256、候选数
  **N = 4**、`multi_parent_batch=True`、缓存预算 **16384 行/层**
  （网格最坏人口 `(chains + chains·N)·T` = chains=4 时 `20·256=5120`，
  3.2× 余量；#47 失效谓词 `人口 > 预算` 不触发，工件 meta 记录
  `cache_budget_tokens_per_layer`）：
  - chains=2：逐链有效长度 `[T,T]` / `[T,T/2]` / `[3T/4,T/4]`；
  - chains=4：`[T,T,T,T]` / `[T,T/2,T,T/2]` / `[3T/4,T/4,3T/4,T/4]`；
  - 每种 × 双构造（nonsplit/split）= 2×3×2 = **12 点**；
  - 每点的 `parent_valid_lengths` **pin 为同一组逐链有效长度**
    （单步机制对拍：父发布与子折叠用同一长度；`v_p < v_i` 的两步
    no-leak 面由 T001/T002 的 UT 覆盖，不在基准网格内）；
- 判定式**先于数据**冻结：
  - **S1**（采样面）：LLaDA `sample_joint(pad_to_max=True)`（N=4，
    chains=2 点）vs 逐链 solo `sample()`——**输出 token 相等为硬门**
    （greedy 确定性 fixture；比较区 pin 为
    `joint_sequences[c][:p_c+L_new]` vs solo 同形切片，右 pad 尾不参与
    比较）；有效区 logits `allclose(1e-6)` 为 report-only（跨画布归约
    长度不同）；任一 token 不等 ⇒ 全 blocked；
    chains=4 的 joint 变长面为 **report-only**（无冻结门，工件记录以防
    4 链路径无证据）；机制面另设 **S1m**：带 `valid_lengths`+
    `parent_valid_lengths`+`row_attention_mask` 的批量折叠（A 行 =
    `chains·N`）vs **同一 T 画布的单行参考**（nonsplit `torch.equal` /
    split `allclose(1e-6)`；参考调用的 `valid_lengths`/`parent_valid_lengths`
    与目标对应行同值，父为全有效或按 `v_i` 对齐的切片，design pin 构造）
    + 无父/forced 全 True 整画布重算（pad 区，同一 mask 行）对拍
    （12 点全覆盖）；
  - **S2**（机制面，12 点全覆盖）：A = 变长合批路径（开关
    `multi_parent_batch=True`，`chains·N` 行单次 `[N_total,T]` 折叠调用，
    **显式携带 `branch_ids`（chain-major 行序）与 `parent_branch_ids`
    （每行父）** 及两长度参数，发布走 #42 `put_batch`：`put` 计
    `+= len(activations)`、`put_batch` 计 foreach 组数 + 立即执行数）；
    B = 逐行参考路径——**`chains·N` 次单行 plural 折叠调用**
    （`branch_ids=[b]`、`parent_branch_ids=[p]`、`valid_lengths=[v]`、
    `parent_valid_lengths=[v_p]`，与 A 同语义）+ 逐条 `put`；**B 侧使用
    不具备 `put_batch` 能力的包装缓存（#42 显式 `AttributeError`），
    确保发布走逐条 `put`**（B 不得走串行/singular 面——FR1 在串行面传
    长度参数会 `ValueError`）；
    **测量窗口 pin**：A/B 前各自 `write_op_count` 归零，**归零点在各自
    父预填充之后**（prefill 不计入 delta，两侧同构），逐点共用同一
    cache 实例（生命周期一致）；
    `write_op_count(A) < write_op_count(B)` 逐点逐构造比较；缓存实现
    **pin 为 `ActivationCache` / `VectorizedActivationCache`**（#42 严格
    A<B 成立的两类；`ChunkedActivationCache` 按 materialization 计数，
    不在本判定式范围）；
  - **S3**：账本缩减比 report-only；逐点 p50 记录 report-only；
- 工件：`results/optimization/ar028_variable_length_bench.json`（`--quick`
  独立 `_quick` 工件，永不覆写正式面）；
- 真实面：LLaDA 不等长 prompt 对配置、命令、资源与判定式写入
  `docs/RERUN_CHECKLIST.md`，**延后**高速算力机执行回填（零伪数字；
  Dream 面因 FR5 范围裁剪不设）。

**验收标准：**
- Given 合成网格，When 执行正式面，Then 工件含冻结判定式全文与逐点结果，
  README/文档数字可全部由工件重算。
- Given 本机 16GB 环境，When 不跑真实面，Then 文档明确标注延后且无任何
  编造数字。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 兼容性 | 默认路径 | 所有新参默认 `None`/`False` ⇒ 旧调用逐字节一致（冻结面 UT 钉死） |
| 正确性 | 变长 parity | nonsplit 逐位 / split `allclose(1e-6)`；pad 区恒重算 |
| 性能 | 合批收益（硬门） | 写账本 `write_op_count(A) < write_op_count(B)` 严格（S2，逐点逐构造）；谓词是否命中由 FR6 验收/UT 证明（不由 S2 推断） |
| 性能 | 计时（report-only） | 工件记录逐点 p50（perf_counter + synchronize、warmup 弃），不作硬门（合批收益由 launch 账本代理，#39 纪律） |
| 可复现 | 合成面 | CPU 单次正式面 ≤ 10 分钟（**12 点（含双构造）**）；seed 固定；工件溯源 |
| 质量门 | 全量 pytest | 只增不减（AR028 起点 1108/5/4/3）；四 lint 门干净；demo 三锚点不动 |

## 5. 约束与假设

**约束：**
- 不改 `ActivationCache`/`VectorizedActivationCache`/`ChunkedActivationCache`
  的存储格式与公开 API；
- 不引入新依赖；CPU 可完整验证（**图回退验收在无 CUDA 主机上以
  能力探测/Mock 验证分支选择，并诚实跳过真实捕获面**）；
- 右 pad 的唯一形态（`[0, v)` 有效、`[v, T)` pad）；pad id **按面定源**：
  采样面（FR5）沿用 `build_right_padded_canvas` 的 `eos_token_id` 语义，
  生成循环面（FR6）用 `pad_token_id`（缺省回退 mask id）；
- **图路径（CUDA Graph 静态体）不感知 pad**：任何携带 `valid_lengths`、
  `parent_valid_lengths` 或 `row_attention_mask` 的 plural 批量调用一律
  eager 回退，一次性 warn（单一 warn-once 身份）；多 parent 面保持既有
  warn-once + eager 回退。
- **真实面延后的授权与义务**：真实 checkpoint 正式面按用户总纲延后；
  RERUN 步骤必须给出完整命令/配置/资源/判定式；回填前文档只可声明
  机制与合成面结论。

**假设：**
- 调用方提供的 `valid_lengths`/`parent_valid_lengths` 与实际 token 有效区
  一致（宿主元数据可信，与 `branch_ids` 同等契约级输入）；
- 右 pad 不改变注意力语义：`row_attention_mask` 由采样侧提供（**仅
  plural 多 parent 面**），层内 pad 列不参与 gate；**非 causal 模型
  （LLaDA/Dream）的等价性依赖该逐行 mask 通道**（`[T,T]` 共享 mask 无法
  表达逐行 padding，AGENTS #43；LLaDA/Dream 的 B==1 同构限制另见
  `llada_sampler.py:649-651`、`dream_sampler.py:492-494`）——不传 mask 的
  非 causal 变长调用不在等价承诺内；
- AR025 的种子公式无链分量前提继续成立（变长不引入链分量）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| valid_lengths | 每行真实有效 token 数（`[0, v)` 有效，`[v, T)` pad） |
| parent_valid_lengths | 每行父证据的真实有效 token 数（缺省=父张量长度） |
| row_attention_mask | 逐行 key mask（`[N,T]`，1=有效、0=pad；仅 plural 多 parent 面） |
| pad-to-max | 变长行右 pad 到同批最大画布 T_max |
| 构造性发散 | 不经 gate 判定、直接置 divergent 的强制重算语义 |
| 父前缀捐赠 | 短父（`T_p < v`）的前 `T_p` 列参与门控复用，其余有效列重算 |
| 冻结判定式 | 基准脚本内数据采集前写死的验收规则（S1/S2/S3） |
