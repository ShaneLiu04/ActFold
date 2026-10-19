# [AR028] 任务清单

**AR028：变长多 parent 批量折叠**（srs.md 同目录；本文件为唯一任务真源）

> 编号说明：本 AR 原拟编号 AR027，实施前发现与既有活动 AR
> `AR027-server-formal-runner` 冲突，故重编号为 **AR028**（目录与文档已同步）。

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001a | Red UT（机制面）：`valid_lengths`/`parent_valid_lengths`/`row_attention_mask` 参数缺失、per-row pad gate、**父 pad no-leak（父画布 T 但 v_p<v_i，plural 面）**、**child pad 反例判别**、统计排除（`num_tokens==v_i`）、入口校验（形状/长度/非 plural 面传 mask）、**plural 面长父 T_p>T 整行降级**、singular/串行面传参 ValueError、图回退 warn-once（runner 计数 0）、**`attention_mask`∩`row_attention_mask` 组合矩阵（`[N,T,T]`/`[N,1,T]`）**、**FR3 发布自一致性** + 冻结面（AR015/AR023/AR025/**AR026 图静态体**）零破坏确认 | - | pending | UT-2801~2815/2825/2827/2831/2832/2834；**AR015 singular 面零改动由冻结面钉死** |
| T001b | Red UT（采样/生成循环/基准面）：LLaDA `pad_to_max` armed/off-switch/Dream ValueError 守卫/chain-major 动态行序/**每链块窗偏移（region/w 用 p_c）**/**每链 draft ctx 重键与输出切片比较区**/**两面对 pad id 契约断言**（FR5 eos spy；FR6 为调用方契约、无代码面）+ 生成循环（谓词分层/链级+候选级长度字段/单链忽略）+ 基准合同（S1/S1m/S2/S3 判定式规格，含 S2 B=单行 plural 调用） | - | pending | UT-2816~2824/2826/2828~2830/2833；与 T001a 可并行 |
| T002 | Green 层内机制（**仅 plural `_forward_batched_multi`**）：`FoldedTransformerLayer` per-row 有效长度（gate 区间 `[0,min(v_i,v_p,i))` + pad/父 pad 构造性发散 + 短父前缀捐赠 + profiler 有效区切片）+ **`row_attention_mask` 传递/形状校验（`[N,T]`，底模接受度验证）** + **`_compose_attention_mask` 组合（DD10）** | T001a | pending | 核心风险面；singular 面不动 |
| T003 | Green 入口与发布：`ManualFoldedForward` 三参数（`valid_lengths`/`parent_valid_lengths`/`row_attention_mask`）贯穿/校验（singular/串行面 ValueError）、**`attention_mask`∩`row_attention_mask` 交集传递**、批量发布整画布、eager 路径 kwargs 附加（None 零扰动）、**任何长度参数的 plural 调用绕过图路径 + 单一 warn-once** | T002 | pending | #41/#42 契约交叉；UT 归 T001a |
| T004 | Green 采样面：`sample_joint(pad_to_max=)`（**LLaDA armed 实现；Dream 签名接受但 armed 拒/off-switch no-op**）+ **每链块窗几何（region/w 用 p_c 独立推导）** + `row_attention_mask`（`[N,T_max]`）+ 验证批共享 mask=最长链 ctx + **chain-major 逐行展开的 `valid_lengths`/`parent_valid_lengths`（每步常量，C5 缩行）** + **每链 draft 用各自 `_spec_ctx_for(xs[c], prompt_len_c)`** + 输出比较区 `[:p_c+L_new]` + 等长两路**输出 token 相同**（同画布 logits nonsplit 逐位/split 1e-6） | T003, T001b | pending | LLaDA 右 pad（eos 填充语义）；Dream 后缀有效面留后续 AR |
| T005 | Green 生成循环：链级/候选级长度元数据契约（`ChainInput.valid_length`、`BranchNode.valid_length/parent_valid_length`，单链显式忽略）+ **调用方 pad 契约（`pad_token_id`，缺省回退 mask id；框架不新增 pad 构造 API，T001b 只验证契约与两面 id 区分）** + **合并谓词分层：`_multi_chain_merge_supported` 结构判定不变 + `_length_metadata_supported`（all/none/partial）**（`_batched_forward_supported` 逐字节不变；legacy 等长无元数据仍走 AR025 合并路径）+ 逐链回退不变 + 跨步长度公式 helper 断言 | T003, T001b | pending | 谓词泛化；无长度元数据⇒旧路径；causal 前提 |
| T006 | 合成基准脚本 `scripts/ar028_variable_length_bench.py`：冻结判定式 S1/S1m/S2/S3 + **12 点**（chains{2,4}×三种逐链长度映射×双构造，T=256/N=4/预算 16384 行/层）+ 逐点 p50（report-only）+ `--quick` 冒烟 | T004, T005 | pending | CPU ≤10 分钟 |
| T007 | 质量门：全量 pytest（1108/5/4/3 起只增不减）+ 四 lint 门 + demo 三锚点 + **AR026 图静态体冻结面零破坏** | T002-T006 | pending | canonical 口径 |
| T008 | 文档五件：README **六节全量重写**（AGENTS README Full-Update Rule；数字全部由工件重算+真实面延后注记）+ AGENTS 新条目（变长有效长度契约，含图路径 eager 回退）+ GUIDE 新部分 + CHANGELOG + RERUN 新步骤（LLaDA 真实面配置/命令/资源/判定式） | T006, T007 | pending | 数字工件溯源 |
| T009 | review 门控（只读子代理：spec/design 合规 + 代码审查），全部问题（含 Minor/Nit）修复循环至 PASS | T008 | pending | |
| T010 | ST 验收：st-cases.md 生成 + 执行（正常/边界/异常/回归四类） | T009 | pending | 真实面 BLOCKED-DEFERRED 记录 |
| T011 | 归档 `specs/archive/AR028-variable-length-multi-parent/` + commit + gitee 分段推送 + 双端树验证 | T010 | pending | 沿 Gitee Segment-Push Protocol |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

- 2026-10-19：需求阶段完成（srs.md + tasks.md），req 门控第 10 轮 PASS；
  发现编号冲突后由 AR027 重编号为 AR028。

## 阶段门控记录

### 2026-10-18 req 门控记录

- 门控结果：FAIL（第 1 轮）
- 失败项：G3/G4 —— FR5 对 AR025 截断事实的描述错误（把基准层截断写成
  `sample_joint` 路径），另有 6 项 Important
- 严重性：Important
- 修复：当轮全部修入（FR5 的 armed/off-switch 两面 + 5 条 Given；
  FR2/FR6 pad 参考改无父/forced 全 True；T003 增图回退 + warn-once 并
  把 UT 归 T001；T008 改六节全量重写；S2 定义 A=变长合批 / B=逐行参考 +
  #42 账本口径）
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req 门控记录

- 门控结果：FAIL（第 2 轮）
- 失败项：G3/G4 —— FR5 未区分 armed 与 off-switch 两条路径，验收混写
- 严重性：Important
- 修复：FR5 拆为 joint-armed / off-switch 两面（各自触发条件与 Given），
  并明确 armed 默认拒绝、off-switch `pad_to_max` no-op
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req 门控记录

- 门控结果：FAIL（第 3 轮）
- 失败项：G3/G4 —— NO-1 FR5 off-switch 定义窄化为"`multi_parent_batch`
  未开"，与代码 `not joint_armed` 事实不符（N>0 但无 folded_model 时逐链
  `sample()` 仍抛既有守卫）；NO-2 Out-of-Scope「父比子有效长度长降级」
  与 FR2 验收「T_p > v_i 但 ≤ T 不降级」矛盾
- 严重性：Important
- 修复：当轮全部修入（off-switch 定义 = `joint_armed == False` 并分
  N==0 合法 / N>0 无 model 守卫两子例；长父范围改为"父比子画布长
  T_p>T 才降级"；FR2/FR5 mask 契约显式；T001 扩到全流程 Red + T005
  双谓词；S2 pin ActivationCache/VectorizedActivationCache）
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req 门控记录

- 门控结果：FAIL（第 4 轮）
- 失败项：G3/G4/G9 —— NO-1 **父 pad 区证据通道缺失**：FR3 整画布发布 +
  FR2 仅凭父张量长度判定，导致父画布 T 而真实 `v_p < v_i` 时下一步会复用
  父 pad 列（no-leak 承诺不可实现）；NO-2 FR7 12 点未 pin 逐链长度映射/
  N/预算与 #47 谓词，S2 的 A/B 行数不唯一
- 严重性：Important
- 修复：当轮全部修入（新增 **`parent_valid_lengths` 通道**贯穿
  FR1/FR2/FR3/FR5/FR6 + 父 pad no-leak 核心验收用例；FR7 pin 逐链长度
  映射/N=4/预算 16384/#47 谓词/12 点（含双构造）；§4 措辞统一；
  Dream 表述改 joint-armed；补不支持 branch_ids 子例）
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req 门控记录

- 门控结果：FAIL（第 5 轮）
- 失败项：G3/G4/G8 —— NO-1 srs 未列 `_multi_chain_merge_supported` 且
  FR6 验收缺 `multi_parent_batch=True` 前置（与 tasks T005 不一致）；
  NO-2 FR5 的 `parent_valid_lengths` 取值点（已提交数）与代码
  `valid_end`/solo parity 矛盾；NO-3 FR2 父 pad no-leak 用例缺因果前缀
  稳定前提且「父=重算」构造下不可判别；WARN-1..9（v_p<=T_p clamp/网格
  parent 长度 pin/FR3 措辞/右 pad 构造方与 pad id/父 batch≠1 应为硬错/
  §4 证据指针/单链路径忽略字段/S1 chains=4 report-only/N 记号冲突）
- 严重性：Important
- 修复：当轮全部修入（srs §2/FR6 增双谓词 + `multi_parent_batch=True`
  前置；FR5 父长度改 pad 边界=prompt_len_c+num_tokens 且验收给可复算
  期望值；no-leak 用例加因果前缀稳定前提 + 「== 无父重算 且 != 父 pad 列」
  双断言；入口 `clamp(v_p,T_p)`；网格 pin parent 长度；FR3 措辞统一；
  右 pad 构造方=调用方且 pad id 定源；父 batch≠1 改 layer 硬错；§4 删
  S2 推断句；单链显式忽略字段 + 验收；S1 chains=4 report-only；FR5
  记号改 L_new）
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req 门控记录

- 门控结果：FAIL（第 6 轮）
- 失败项：G3/G4 —— NO-1 FR5 的长度期望按链（2 值）而非按 joint 批量的
  chain-major 行（`sum_live*N` 行），与 FR1 的"长度==批行数"契约及代码
  事实冲突（C5 还需动态缩减）；NO-2 FR5 joint-vs-solo 写死 `torch.equal`，
  与 §4/FR7 及 #41 split 1-ULP 事实冲突；WARN-1..6（FR6 谓词措辞与父
  形态、FR2 参考父长度未 pin、draft 串行 pad 语义未声明、§2/§5 漏列
  parent_valid_lengths、S2 测量窗口与 S1m 参考父未 pin、pad id 与既有
  eos 填充 helper 对齐）
- 严重性：Important
- 修复：当轮全部修入（FR5 改 chain-major 逐行期望
  `[p+L_new]*N + ...` + C5 动态缩减验收；parity 分档 nonsplit
  equal/split allclose；FR6 改"三处谓词面各自泛化"+"各行非 None 父"+
  父 batch≠1 为 layer 硬错；FR2 参考父按 v_i 重建/切片；FR5 增 draft
  串行 pad 注记；§2/§5 补 parent_valid_lengths；S2 增测量窗口 pin、
  S1m 参考父 pin；pad 填充 id 对齐 build_right_padded_canvas 的 eos 语义）
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req 门控记录

- 门控结果：FAIL（第 7 轮）
- 失败项：G3/G4/G8 —— NO-1 FR6 写"三处谓词面"但只列两处（与 T005
  "两个谓词"冲突）；NO-2 单父/逐链批量面无法承载 `parent_valid_lengths`
  （FR1 plural-only 契约，no-leak 不可实现/不可测）；NO-3 FR2 长父降级
  与 AR015 单父面 AR007 截断矛盾且未限定作用面；WARN-1..8
  （`_batched_forward_batched` 符号不存在/FR3 等价面缺
  `buffer[:final _counts]`/draft pad 被 forced 覆盖的说法不实/FR6 缺
  causal 前提/FR6 参考画布未 pin/FR5 跨画布逐位无契约支撑/CPU 与 CUDA
  图验收环境张力/T005 未点名 AR015 面）
- 严重性：Important
- 修复：当轮全部修入——**机制面收敛为仅 plural 多 parent 面**（FR1
  校验 singular/串行面传参 ValueError；FR2/§2/§5 明示 singular 面零改动
  保留 AR007 截断）；FR6 改为**仅合并谓词 `_multi_chain_merge_supported`
  泛化**（`_batched_forward_supported` 逐字节不变）；§1 符号改
  `_run_folded_forward_batched`；FR3 等价面补 `buffer[:final _counts]`；
  draft pad 注记改为"不在本 AR 发散保证范围"；FR6 增 causal 前提 +
  参考画布=同一 T_max/同 mask（跨画布 allclose report-only）；FR5 参考
  构造 pin 同 T_max；§5 CPU 验证注记（Mock 分支选择+诚实 skip 捕获面）；
  S2 归零点 pin 在父预填充后；T001/T002/T003/T005 同步作用面
- 审查代理：sdd-gate-reviewer

### 2026-10-19 req 门控记录

- 门控结果：FAIL（第 8 轮）
- 失败项：G3/G4 —— NO-1 **非 causal 等价的 mask 通道缺失**：LLaDA/Dream
  右 pad 的有效 token 会在双向注意力中看到 pad 列，仅 `valid_lengths`
  不足以保证与独立运行等价（AGENTS #43：`[T,T]` 共享 mask 无法表达逐行
  padding；folded_transformer 单 mask 广播全行）；NO-2 FR5 的 parity 面
  按 `torch.equal`/`allclose` 比较 token 序列（int）无意义，跨画布
  masked-softmax 归约长度不同也不支持逐位；NO-3 Dream `pad_to_max` 的
  "签名不接受"与"同签名 ValueError"两处措辞自相矛盾；NO-4 pad id 定源
  §5 全局写 eos 与 FR6 `pad_token_id` 冲突；NO-5 FR6"无长度元数据 ⇒
  串行回退"与 AR025 等长合并路径事实冲突
- 严重性：Important
- 修复：当轮全部修入——用户定案**扩展 per-row attention mask 通道**
  （FR1 三参数/`[N,T]` key mask/非 causal 等价 `allclose(1e-6)`、FR2 参考
  改同 T_max 单行 + 同 mask 行、FR5 输出 token 语义硬门 + 有效区 logits
  report-only、FR6 明示 causal-only 且参考为同画布单行、§5 图回退触发含
  mask、pad id 按面定源、FR6 无元数据等长走 AR025 合并批量、Dream 措辞
  统一为"签名接受/armed 拒/off-switch no-op"、S1 改语义硬门 + report-only
  logits、T001-T004 同步）
- 审查代理：sdd-gate-reviewer

### 2026-10-19 req 门控记录

- 门控结果：FAIL（第 9 轮）
- 失败项：G8 —— NO-1 T005 仍写"eos 填充"，与 FR6 `pad_token_id`（缺省
  回退 mask id）定源冲突；Important WARN-1 FR5 输出 token 比较区未 pin
  （joint `T_max` vs solo `p_c+L_new` 形状不一致）；WARN-2 joint draft 复用
  链 0 ctx，短链 pad 列被当有效 key；WARN-3 FR2 pad 区参考缺"同一 mask
  行"；WARN-4 S2 的 B="串行折叠调用"与 FR1"串行面传长度参数 ValueError"
  冲突；Minor WARN-5 §1 路径名与"串行回退"措辞；WARN-6 `ChainInput`
  （逐链）误写为 per-candidate、单链忽略验收不可注入；WARN-7 mask 作用
  面/`N==T` 判别歧义；WARN-8 T001 过粗；WARN-9 图回退验收自证
- 严重性：Important
- 修复：当轮全部修入——T005 改 `pad_token_id` + T001b 两面 pad id 区分
  断言；FR5 比较区 pin `[:p_c+L_new]` + draft 按链 `_spec_ctx_for` + 每链
  ctx UT；FR2 pad 区参考补同一 mask 行；S2 的 B 改单行 plural 调用；§1
  分列两路径名与"逐链回退"；FR6 字段改 ChainInput 逐链/BranchNode 逐
  候选，单链忽略改 helper UT；per-row mask 改专用 `row_attention_mask`
  （仅 plural 面；其他面 ValueError；消除 N==T 歧义）；T001 拆
  T001a/T001b 并更新依赖；图回退验收改显式 `use_cuda_graph=False` 对比 +
  runner 计数 0
- 审查代理：sdd-gate-reviewer

### 2026-10-19 req 门控记录

- 门控结果：PASS（第 10 轮）
- 审查项数：9 项（G1-G9 全 YES）
- 修复的问题：第 9 轮 10 项全部 FIXED；非阻塞 WARN 处理：
  W1 mask 组合语义已在 FR1 冻结为允许集交集（并补验收）；W2 跨步长度
  公式已在 FR6 pin（helper UT）；W3 warn-once 措辞改"fresh 实例恰好
  一次/生命周期至多一次"；W4 **编号冲突 → 由 AR027 重编号为 AR028**
  （与 `AR027-server-formal-runner` 消歧）；W5 T005 边界明确（框架不新增
  pad 构造 API，仅验证契约与两面 id 区分）
- 审查代理：sdd-gate-reviewer

### 2026-10-19 req 门控记录

- 门控结果：PASS（第 11 轮，重建后全维度重审）
- 审查项数：9 项（G1-G9 全 YES/PASS）
- 修复的问题：重建完整性确认（章节/FR/记录 1-10 连续）；非阻塞 WARN 当轮清理：W-01 FR2 触发面扩到父 pad（补 v_i=T,v_p<T 用例）；W-02 S2 的 B 固定为无 put_batch 能力的包装缓存；W-03 FR5 返回整画布契约与比较切片分列；W-04 交叉引用修正（FR2 验收 5、T001a/T002、#43）；W-05 术语/假设补 parent_valid_lengths；W-06 T001a/T007 补 AR026 图冻结面；W-07 FR2/S1m 参考父长度 pin；W-08 child pad 反例判别补全；W-09 T001a/T003 补 mask 交集；W-10 警告仅在 use_cuda_graph=True 触发；W-11 等长两路统一为输出 token/logits 分档；W-12 FR4 区分 eager record 与图 record_precomputed；W-13 S2 的 A 显式 branch_ids/parent_branch_ids；W-14 FR6 的 L0=真实长度
- 审查代理：sdd-gate-reviewer

### 2026-10-19 design 门控记录

- 门控结果：PASS（第 4 轮）
- 审查项数：15 项（G1-G15 全 PASS）
- 第 3 轮问题：NO-1（1-D mask expand）、NO-2（零预算窗缩行）均 FIXED；
  WARN-1..7 全部 FIXED
- 当轮清理的 WARN：W-08 显式传 orced_divergent=joint_forced（+UT-2816 spy）；
  W-09 pad_to_max=True 跳过既有等长守卫 pin；W-10 定义
  window_chains=[entry.chain for entry in window]；W-11 refill 用链自身
  ctx_c；W-12 共享 mask 每窗按存活链最长者重算（+UT-2829 子例）；
  W-13 chains=4 三组映射与 S1m 参考构造展开；W-14 _max_length=T_max
  确定性收口
- 审查代理：sdd-gate-reviewer
