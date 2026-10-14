# 1 AR概述

| 组件名称 | actfold 生产生成回路折叠接线（FastDLLMSampler ↔ ManualFoldedForward） |
| --- | --- |
| AR系统流水号 | AR021 |
| AR描述 | 把 AR015-020 批量验证栈接入真实生产生成回路：sampler 折叠前置条件线程化（pe + 块因果∧pad 掩码对象钉定）、remasking 链折叠（强制散度活跃块，构造级绕开 AR019 P5 近似风险）、形状键控 graph runner 池（跨块增长图收益）、批量投机块验证（opt-in，AR019 工作负载生产化）、真实 generate() 级三模式基准 |

# 2 动态行为

## 2.1 折叠路径时序图（denoise 迭代，一个形状窗口内）

```plantuml
@startuml
title FR1/FR2 folded-serial：remasking 链折叠（同形状窗口）
participant Sampler as FastDLLMSampler
participant Ctx as _fold_ctx (shape 键控)
participant MFF as ManualFoldedForward
participant Pool as runner 池 (opt-in)
participant Cache as ActivationCache

Sampler -> Ctx : 形状/pad 模式变化？→ 按 (T, prompt_len) 构建\nmask[T,T](块因果∧非pad，pad 区=host 元数据)\n+ pe(rotary,AR019 同式) + forced[T](当前块 True)\n同键 → 缓存命中（对象钉定）
Sampler -> MFF : forward(x, branch_id, parent_branch_id,\nstep_idx, attention_mask=mask,\nposition_embeddings=pe,\nforced_divergent=forced)
alt 图路径（池开 + 前置满足）
  MFF -> Pool : 形状键查找 runner
  alt 命中（mask/pe is 钉定 + forced None/not-None 身份）
    MFF -> Pool : forced_static.copy_(forced)\ntokens_static.copy_(x)\nreplay()
  else 未命中
    MFF -> Pool : 捕获新 runner（LRU 守恒）\nforced_static 随捕获分配
  end
else eager 路径
  MFF -> MFF : 层循环：gate(child,parent) **AND ~forced**\n→ merge(父FFN[前缀复用], 重算[当前块])
end
MFF -> Cache : 发布（串行图：逐层 cache.put，_publish_graph_child\narchitecture_utils.py:850-888；AR018 快照+AR020 put_batch\n仅在批量图发布 _publish_graph_children——FR3 侧）
MFF --> Sampler : logits
Sampler -> Sampler : 阈值 unmask + argmax（基线配方不变）
note right
  前缀块：gate≈1（P3/P4 位稳定）→ 复用父缓存
  当前块：forced 整区强制散度 → 全量重算
  （P5 的"同块未翻位置误判 stable"被构造级绕开）
end note
@enduml
```

## 2.2 批量投机块验证时序图（FR3，opt-in）

```plantuml
@startuml
title FR3 folded-batched：N 候选一次 [N,T] 批量验证
participant Sampler as FastDLLMSampler
participant MFF as ManualFoldedForward
participant Pool as 批量 runner 池（键 (N,T,prompt_len)）

Sampler -> MFF : ① 起草前向（= 基线迭代1 前向，链父分支 p）
MFF --> Sampler : logits@当前块掩码位
Sampler -> Sampler : ② top-k 起草 N 候选\n(每候选独立 CPU generator 种子，公式冻结见 §4.2.2(4))
Sampler -> Sampler : ③ 堆叠 [N,T]（画布克隆+候选填充）\nforced=[N,T] 当前块 True
Sampler -> MFF : ④ 单次批量调用：branch_ids=[f"{p}_cand{j}"],\nparent_branch_id=p, mask/pe 钉定, forced
MFF -> Pool : 批量池查找（键 (N,T,prompt_len)）
alt 命中（warmup 已预捕获）
  MFF -> Pool : forced_static.copy_(forced)\ntokens_static.copy_(cands)\nreplay()
else 未命中（warmup 首轮/驱逐后）
  MFF -> Pool : 既有前置满足 → 捕获入池\n（warmup 预捕获，A5 方案 a）
else 前置失配/预算超限
  MFF -> MFF : AR016 矩阵 eager 批量回退\n（既有矩阵，不新增静默路径）
end
MFF -> Cache : 发布（图路径：AR018 快照 + AR020 put_batch；\neager 回退：_store_batched 逐分支 [1,T]）
MFF --> Sampler : logits [N,T,V]
Sampler -> Sampler : ⑤ 选定：min-prob 准则\n(候选各掩码位 token 概率最小值，取最大候选)
Sampler -> Sampler : ⑥ 提交选定块 → 链父切换为选定候选分支\n→ 跳过剩余迭代\nfinished/停止语义同基线 → AR 步照旧
note right
  A5 定案（srs §5，方案 a）：folded-batched 保留
  use_cuda_graph=True + 批量池——warmup 轮每形状
  窗首个合格批量调用自然预捕获（零新机制），
  timed reps=30 全程重放（摊销成立）；srs H4
  "批量图捕获>0"由此成立，AR016/018/020 批量面
  + AR017 自适应容量（MFF_B 接通，事件条件式
  核验：未触发=中性）首次被生产回路行使。
  生产单发（单次 sample）摊销为负的
  注记入工件 meta。
end note
@enduml
```

# 3 功能点分解

| 序号 | 功能点名称 | 功能点描述 | 追溯 |
| --- | --- | --- | --- |
| 1 | sampler 折叠上下文供给 | shape 键控缓存：块因果∧非 pad `[T,T]` 掩码 + rotary pe（AR019 同式）+ forced 基张量；同形状对象钉定；raw 路径零扰动 | FR1 |
| 2 | 强制散度标记面 | `ManualFoldedForward.forward(+forced_divergent)` → 层 gate 结果 AND ~forced（eager）+ runner `forced_static` 缓冲（图捕获体）；默认 None 零扰动 | FR2 |
| 3 | 形状键控 runner 池 | opt-in（`graph_runner_pool_size`，默认 0=legacy 单 runner 行为）；形状键查找/LRU 驱逐/VRAM 守恒；串行/批量独立；AR017 自适应对"当前活跃 runner"照常 | FR2 |
| 4 | 批量投机块验证 | `FastDLLMSamplerConfig.num_block_candidates`（默认 0）：top-k 起草 → 单次 `[N,T]` 批量验证 → min-prob 选定提交；停止语义同基线；默认关逐字节不变 | FR3 |
| 5 | 基准脚本 + 工件 | `scripts/ar021_generation_loop_bench.py`：三模式真实 generate() 墙钟 + 质量面 + 账本/canary + H1-H4（判定式冻结、reps=30 单次定案；三模式构造/warmup/round-robin/工件 schema 见 §4.2.2(6)） | FR4 |
| 6 | 文档收口五件 | GUIDE 第二十八部分 / CHANGELOG / AGENTS 新条目 / RERUN_CHECKLIST / README 全量中文重写 | FR5 |

# 4 实现设计

## 4.1 功能实现思路（方案对比）

### 4.1.1 强制散度传递面（FR2 核心）

- **方案 A（选定）：MFF forward 新增 `forced_divergent` kwarg → 层内 gate 结果 AND**。
  与 AR015 `branch_ids`/AR019 `position_embeddings` 的尾参线程化模式同构。
  **eager 面 AND 锚点（四条 gate 路径全覆盖，单一排序规则）**：`stable_mask`
  物化之后立即 AND（`stable_mask &= ~forced`，forced 非 None 时），先于
  `GLOBAL_STABILITY_PROFILER.record`（folded_transformer.py:300 串行/:528 批量
  ——profiler 消费 AND 后掩码，H4 稳定比观测反映强制散度后的真实面），再先于
  count 消费（:315-317/:540——三路快分 :322/:358 与批量 :543/:572 按新鲜计数
  分派，杜绝过期计数整层复用含 forced 位）。各路径物化点：串行等长 **fused**
  （:256-272，`fused_gate_mask_count` 分派 :265，掩码物化于 :272）/ 串行等长
  plain（:274）/ 串行变长（:285-297，:297 物化完成）/ 批量（:510/:514-526）。
  **fused 路径计数弃用**：kernel 计数（`stable_count_buf` :260-264）在 forced
  非 None 时置 None → 既有 :316 fallback `stable_mask.sum()`（一次设备归约），
  **不修改 Triton kernel**。图面由 runner 持 `forced_divergent_static` 缓冲，
  replay 前 `copy_`（纯设备拷贝零 sync，发生在捕获体外）；捕获体内 fused gate
  （cuda_graph.py:408-415）产出 mask_buf 之后 AND（:415-419 之间）并自 AND 后
  掩码**设备侧归约重导出 count_buf**（固定形状归约算子可捕获，无 host sync，
  保持 AR016/AR017 预算校验/自适应观测语义一致——forced 位计入 divergent
  预算）。默认 None 不附加 kwarg → 全既有面逐字节不变。
- 方案 B（否决）：ForcedRegionGate 包装 gate。缺点：fused gate kernel（#38）与图捕获
  体内的 gate 由 runner/层持有，包装不进捕获体；且 gate 类型契约（#37 精确型
  SimilarityGate）会被破坏。
- 方案 C（否决）：发布后失效父缓存的活跃块条目。缺点：违背 cache 逐条 put 语义面
  （#42 等价性面），粒度粗（整层失效），且不解决"前缀块 gate 误判"以外的任何面。

### 4.1.2 形状键控 runner 池（FR2 核心）

- **方案 A（选定）：MFF 内 `OrderedDict[形状键, runner]` 池 + LRU，opt-in 旋钮**。
  键=串行 `(T, prompt_len)`、批量 `(N, T, prompt_len)`（与 ctx 键对齐——对象组
  信息显式含于键，§4.2.2(3)）；命中→`is` 钉定校验（**mask/pe 面**）+ **forced
  None/not-None 身份校验**（forced 内容经 replay 前 `copy_` 传入，形状随捕获钉定
  ——与 mask/pe 的对象级 `is` 契约不同：forced 的 [N,T] 张量每次调用新建，对象
  级钉定永不命中，故身份= None/not-None + 形状，见 §4.2.2(2)）→replay；未命中→
  按既有前置条件捕获新 runner；超上限→`popitem(last=False)` 驱逐（runner 引用
  释放→静态缓冲回收，VRAM 守恒由 UT 探针钉死）。**池关（默认）= 现状单 runner
  逐字节**（含 UT-1504 warn-once-never-recapture 面零扰动）。AR017 自适应机械照常：
  池模式下 `self.graph_runner_batched` 维持为"当前形状活跃 runner"别名（每次批量调用
  刷新指向），自适应重捕获原地替换池条目，事件/计数语义不变。**#42 cache setter
  传播面扩展**：`cache` property setter（architecture_utils.py:484-505，现状仅遍历
  `("graph_runner","graph_runner_batched")`）池化后遍历**串行池与批量池的全部
  runner**（legacy 别名属性即池活跃成员，驱逐后剩余成员与再捕获成员同理），
  `mff.cache = other` 对池内每个 runner 的 `_cache` 生效——否则池内非活跃 runner
  持过期 cache，replay 前置父缓存校验/父 fetch 走旧 cache。  **观测面（#42 外部契约 + 基准出口）**：MFF 新增只读属性 `graph_pool_stats`
  （dict：按形状键分组的 captures/replays/evictions + 池当前大小 + **两类
  eager 计数器**（`degradation_fallback_count`/`structural_eager_count`，实例
  累计，口径见 §4.2.2(6)），串行/批量分列；池关时池部分为空 dict、计数器
  照常）。
  **批量池（键 (N,T,prompt_len)）为 FR3 批量验证的生产路径**（A5 方案 a：
  warmup 预捕获 + timed reps 重放，§4.2.2(4)④——srs H4"批量图捕获>0"由此
  成立；AR016/017/018/020 批量面首次被生产回路行使）。
- 方案 B（否决）：形状失配时 reset 单 runner 重捕获。每块一次捕获开销（~一次全前向
  量级）+ VRAM 抖动；且与 UT-1504 钉死的 legacy 行为冲突需改既有面。
- 方案 C（否决）：生产关图只用 eager。放弃图收益，FR2 的 H4 面不成立。

### 4.1.3 批量投机块验证落点（FR3）

- **方案 A（选定）：FastDLLMSampler.sample 内 opt-in 分支**。复用 FR1 上下文供给、
  #26 链、AR 步与停止语义；`num_block_candidates=0`（默认）时零新代码路径执行。
- 方案 B（否决）：子类 SpeculativeFastDLLMSampler。需复制整个 sample 循环（300 行），
  漂移风险（基线配方修正不同步）。
- 方案 C（否决）：扩展 folded_generate。其为 AR 逐 token 循环（#20/#34），非块解码
  语义，与 fast-dLLM v2 配方不符。

### 4.1.4 折叠上下文供给（FR1）

- **方案 A（选定）：sampler 实例级 shape 键控 `_fold_ctx` 缓存**（跨 sample 调用存续
  ——基准多 rep 同形状必须命中同对象，否则 runner `is` 钉定失配→eager）。**键 =
  `(T, prompt_len)`**（均为 host 侧元数据，零 device readback）：mask 的 pad 列由
  host 元数据构建——pad 唯一合法来源是 prompt 对齐填充（fast_dllm_sampler.py:142-149，
  `first_block_padding` 由 prompt_len 与 block_size 唯一确定），pad 区 =
  `[prompt_len, prompt_len+first_block_padding)`，故 (T, prompt_len) → mask 内容
  **确定性映射**（同键必同内容，跨 sample 调用复用安全；同 T 不同 prompt_len 的
  不同 prompt 必然分键——纯 shape 键会把同画布长、不同 pad 数的 prompt 撞进同一
  条目，pad 列错误 → 语义静默错误，本设计以此显式排除）。条目 = (mask `[T,T]`,
  pe|None, forced `[T]`)；上限 32 LRU（键含 prompt_len 后条目数不再坍缩）。pe 供给：
  drill `model.model.rotary_emb`（AR019 `_build_context` 同式：`zeros[1,T,H] bf16 +
  arange(T)` 常位方案）；无 rotary（合成模型）→ None（丢弃机制为 `_recompute_all`
  的 `**kwargs` 层签名过滤——folded_transformer.py:131-136 / architecture_utils.py
  :644-654，非 #15 的 raw-adapter 面）。**已知边界（如实记录）**：prompt 内部本身
  含 pad_id 的病态输入下，折叠掩码（只排除对齐填充区）与 raw `x != pad_id`（排除
  一切 pad 位）存在差异——生产 prompt 不含 pad token，UT-2101 于正常输入断言两种
  掩码逐位等价，该边界入 GUIDE 限制清单。
- 方案 B（否决）：每调用重建。破坏对象钉定（缺口-2 复活）；且无法解决 pad 键碰撞
  （每调用以 `x[0]!=pad_id` 设备比较构建掩码需跨调用比对内容，引入 sync）。
- 方案 C（否决）：调用方（bench/demo）供给。生产调用方负担重、易漏，且 ctx 与画布
  形状/pad 元数据的耦合知识在 sampler 内。

## 4.2 功能实现设计

### 4.2.1 流程图（sampler 折叠路径总控）

```plantuml
@startuml
start
:sample(prompt, folded_model);
if (folded_model?) then (yes)
  if (B > 1?) then (yes)
    :raise ValueError("折叠路径要求 B==1\n（[T,T] 掩码不可表达逐行 pad）");
    stop
  endif
  :folded = true;
else (no)
  :folded = false（raw 路径逐字节现状）;
endif
while (还有块?)
  :ctx = _fold_ctx_for(T, prompt_len)\n（未命中→构建 mask/pe/forced 并钉定）;\n  while (小块去噪迭代)
    if (folded) then (yes)
      if (N = num_block_candidates > 0) then (yes)
        :FR3 批量投机：起草前向→top-k N 候选\n→单次 [N,T] 批量验证（批量池：warmup 预捕获\n+replay；A5 方案 a）\n→min-prob 选定→提交→链父=选定候选→跳出;
      else (no)
        :FR2 链折叠前向：mask/pe/forced 传入\n（forced=当前块整区）;
        :基线阈值 unmask + argmax（不变）;
      endif
    else (no)
      :raw 前向：attention_mask = x != pad_id（不变）;
      :基线阈值 unmask + argmax（不变）;
    endif
  endwhile
  :AR 步前向（folded 时 forced=当前块——\n刚提交块 vs 父的部分翻转，P5 同式绕开）;
  :追加 token + 新 mask 块（画布增长→新形状键）;
endwhile
:decode_final（不变）;
stop
@enduml
```

### 4.2.2 流程说明

**（1）FR1 上下文构建**（`FastDLLMSampler._fold_ctx_for`）：
- 键：`(T, prompt_len)`（host 元数据，零 device readback；见 §4.1.4 方案 A）。
- mask：`arange[:,None]//bs >= arange[None,:]//bs` 块因果（AR019 P4 同式）∧
  pad 列排除——pad 区 `[prompt_len, prompt_len+first_block_padding)` 由 host
  元数据置 False（`torch.ones(T,T,bool)` 上 host 整数切片赋值，与 raw 路径
  `x != pad_id` 在"prompt 无 pad_id"前提下逐位等价，UT-2101 断言）；`[T,T]` bool。
- pe：`model.model.rotary_emb` 存在 → `(cos, sin) = rotary(zeros(1,T,H,bf16),
  arange(T))`（AR019 `_build_context` 逐字同式）；不存在 → None。
- forced 基张量：`zeros(T) bool`，当前块区 `[T-bs:T]` 置 True（去噪与 AR 步共用——
  同形状窗口内当前块恒为活跃区）。
- 供给纪律：同形状窗口内全部调用 `is` 同对象（图前置 mask/pe 钉定）；跨形状增长
  重建重钉；ctx 缓存跨 sample 调用存续（多 rep 命中；同 (T,prompt_len) 必同内容，
  不同 prompt_len 必分键）；构建全程零 device readback（mask/forced 为 host 元数据
  驱动的设备张量构建，无 `.item()`/`.cpu()`）。

**（2）FR2 强制散度语义**：
- 定义：`forced_divergent` 为 bool 张量 `[T]`（串行，广播到 [1,T]）或 `[N,T]`（批量）；
  True 位置的 token 在**每层**的稳定性判定中被强制判散度（`gate_result & ~forced`），
  即整区全量重算——不依赖 gate 对该区的判断。
- 覆盖区（sampler 侧）：当前正在去噪的**整个块**（含同块已提交的前部小块——块内
  双向注意力会把当前小块的翻转传播到同块全部位置，P5 实证）；AR 步前向同区（刚提交
  块 vs 父=末去噪迭代，部分翻转场景）。前缀块（当前块之前）不受 forced，走 gate
  （P3/P4 位稳定证据 → 复用精确）。**与 srs 术语映射**：srs §3.2 定义覆盖区为
  "当前活跃小块"；本设计实现为"当前块整区"——保守超集（默认冻结常数
  block_size==small_block_size==32 下二者**重合**；仅当 small_block_size<
  block_size 的非默认配置下设计多复算同块前部小块，parity 仍由构造成立，仅
  前缀复用率略降）。srs"当前活跃小块" ⊆ 本设计覆盖区。
- parity 由构造保证：前缀复用精确 + 活跃块全量重算 ⇒ folded-serial 输出与 raw
  bit-exact（合成面 UT 钉死；真实面 H1 仲裁）。
- 图面：runner 捕获时若调用携带 forced → 分配 `forced_divergent_static`（形状随捕获
  钉定）；replay 前 `copy_`（内容可变、形状恒定）；捕获体内 gate 之后 AND 并重导出
  count_buf（设备侧归约，捕获安全，§4.1.1）。**forced 面参与捕获身份（None/
  not-None + 形状）**：无 forced 捕获的 runner 不服务 forced 调用（replay 时
  None/not-None 失配 → 静默 eager；与 mask/pe 的对象级 `is` 契约同为捕获身份面，
  但粒度不同——mask/pe 由 ctx 供给恒同对象故 `is` 可行，forced 的 [N,T] 张量
  每次调用新建故只校验 None/not-None + 形状，内容经 `copy_` 传入）；反之亦然
  （有 forced 捕获的 runner 不服务 None 调用）。
- 零扰动：`forced_divergent=None`（默认）不附加 kwarg（AR019 pe 同式），全既有
  MFF/层/runner 面逐字节不变。

**（3）FR2 形状键控池**：
- 键（**与 srs"tokens shape + 掩码/pe 对象组"键控语义对齐**）：串行
  `(T, prompt_len)`、批量 `(N, T, prompt_len)`；值：`FoldedGraphRunner`。对象组
  信息显式含于键——mask/pe 对象由 ctx（键 (T,prompt_len)，§4.1.4）唯一供给，
  **同池键 ⇒ 同 ctx 条目 ⇒ 同 mask/pe 对象**，故 replay 前置的 mask/pe `is`
  校验在同键下恒成立（校验保留为恒真防御面；纯 tokens-shape 键在多 prompt
  基准下同 T 异 prompt_len 会撞键——池命中但对象组不同 → `is` 失配，行为未定，
  本设计以此显式排除）。
- 查找：命中 → 既有 replay 前置校验（shape/mask `is`/pe `is`/**forced None-
  not-None 身份**/父缓存完整/budget）→通过则 replay；未命中 → 既有捕获前置
  （#37 全条）满足则捕获入池。**失配行为定案**：键控后同键 `is` 恒真，失配仅
  可能来自异常态（ctx 被外部篡改等）——届时**静默 eager 不重捕获**（镜像
  现状 architecture_utils.py:823-830 语义，不新增替换/驱逐路径）。
- LRU：命中 `move_to_end`；`len > cap` → `popitem(last=False)`（最旧 runner 引用
  释放）。驱逐后同键重现 → 再捕获（成本=一次捕获，正确性无虞）。
- **cap 联动定案（多 prompt 相容）**：池条目数 = 形状窗数 W × prompt 数 P
  （每 prompt 的 prompt_len 不同 → 各自成键）。基准冻结 W=4（num_tokens=128，
  4 块）+ P=2（§4.2.2(6)）。**条目数逐模式核算（结构性 eager 计入）**：
  `folded-batched` 模式——起草前向恒结构性 eager（每窗首串行前向 parent
  旧形状，不入图），可捕获串行前向 = 窗 1..W-1 的 AR 步（末窗 remaining=0
  无 AR 步）⇒ 串行条目 (W-1)·P = **6**；批量条目 W·P = **8**。
  `folded-serial` 模式——串行条目 ≤ W·P = **8**（数据依赖上界：每窗首个
  去噪前向结构性 eager，≥2 次前向的窗方可捕获；单迭代完成的窗 0 捕获，
  Nit-11 数据依赖边界见 §4.2.2(6)）。**cap=8 容纳全部模式条目（≤8）零驱逐**
  （LRU 仅为防御上限）；ctx 条目 = W·P = 8 ≤ 32 同裕。cap 不足 P 增配时跨
  prompt LRU 抖动（每 rep 全量重捕获）——基准规模与 cap 联动冻结，工件
  meta 记录。
- 激活面：`ManualFoldedForward.__init__(graph_runner_pool_size: int = 0)` +
  `ActFoldConfig.graph_runner_pool_size: int = 0`（≥0 校验，`__post_init__`
  ValueError 模式循 config_manager.py:146-177 先例；BenchmarkRunner 透传循
  benchmark_runner.py:191-225 先例）。0=legacy（单 runner 属性与 warn-once 行为
  逐字节，UT-1504 面零扰动）。
- #42 传播：`cache` setter 遍历串行池+批量池全部 runner（§4.1.2；UT-2106 断言
  池模式下 `mff.cache = other` 后池内每个 runner 的 `_cache` 均已切换）。
- AR017 自适应兼容：池模式下 `graph_runner_batched` 为"当前活跃 runner"别名（每次
  批量调用刷新指向）；自适应重捕获原地替换对应池条目；`graph_adaptive_policy`/
  事件计数为 MFF 实例级语义不变。**注记**：adaptive 的实例级观测状态（过/欠配给
  streak、throttle 计数）跨池条目共享——各条目 (C=⌈ratio·N·T⌉, T) 不同，混合
  观测可能跨形状误判过/欠配给；行为被"原地替换 + prospective guard + 事件如实
  计数 + 墙钟不隐藏"（§4.2.2(6)）约束，无正确性/可满足性风险（srs H4"未触发
  =中性"已容）。
- 观测面：`graph_pool_stats` 只读属性（§4.1.2；按形状键分组 captures/replays/
  evictions + 池大小，串行/批量分列）——UT-2106 探针与基准 per-shape-window
  计数出口共用（§4.2.2(6)）。
- VRAM 量级（A4 设计侧估算）：每串行 runner 静态缓冲 ≈ parent [L+1,1,T,H] +
  child [L,1,T,H]（bf16）+ **logits_buf [1,T,V]** + mask/count 小项 ≈
  `(2·(L+1)·T·H + T·V)·2 bytes`；批量 [N,T] runner 的 parent/child/logits 按
  N 倍计 + 图私有内存池开销。以 Fast_dLLM v2 1.5B（L≈28、V≈32k）与 T≤
  prompt_pad+128 量级：单串行 runner ≈ 0.1GB 级、批量 [N=4,T] ≈ 0.3-0.4GB 级。
  基准 cap=8 下：folded-serial 模式串行池（≤8 条目，数据依赖）≈0.8GB；
  folded-batched 模式串行池（6 条目，§4.2.2(3) 核算）≈0.6GB + 批量池
  （8 条目）≈2.6GB（两 MFF 实例池并存峰值 ≈3.6GB 量级，§4.2.2(6)
  `split_layers=False`）；与模型本体+缓存/workspace 合计峰值预计 ≈9-10GB，
  16GB 卡相容（AR019/20 实测峰值 8.9/10.0GB 先例）；**--quick 实测校准钉死**
  后入工件 meta（池上限 + 实测 VRAM 峰值；若实测超界，如实记录并以降 P 或
  W 的预案重定，不伪造）。

**（4）FR3 批量投机块验证**（逐小块，`num_block_candidates=N>0` 且 folded）：
- 前置：`num_block_candidates>0` 且 `folded_model` 为 None 或不支持 `branch_ids` →
  构造期 `ValueError`（不静默降级）；`N<0` → `ValueError`；`candidate_top_k<1` →
  `ValueError`（构造期）；`candidate_top_k` 默认 10。
- ① 起草前向 = 基线迭代 1 前向（链分支 p，#26 链照常）——该前向本身即基线行为，
  其 logits 供起草。
- ② 起草：当前小块**仍掩码位**的 draft logits `top-k`（k=`candidate_top_k`）；
  候选 j 的每位取值 = `torch.randint(0,k,(S,), generator=g_{b,s,j})` 索引该位
  top-k（AR019 `_make_workload` 的起草**风格**：CPU generator、逐位独立选取；
  种子推导式为本设计冻结——AR019 为单 generator 顺序式，此处 per-candidate
  独立种子）。**种子公式（冻结）**：
  `seed(b,s,j) = (S0·1_000_003 + block_idx·8_191 + small_block_idx·131 + j) &
  0x7FFF_FFFF`，其中 `S0 = config.seed if config.seed is not None else 0`；
  `g_{b,s,j} = torch.Generator(device="cpu").manual_seed(seed(b,s,j))`——全
  host 整数运算，跨运行可复现（不依赖 Python `hash` 盐化），UT-2109 断言。
- ③ 堆叠：`cands[N,T]` = 画布克隆 + 候选填充（仅仍掩码位）；`forced[N,T]` 当前块
  True（ctx.forced 广播展开——每调用新建张量，故图侧 forced 身份为 None/
  not-None+形状而非对象 `is`，§4.2.2(2)）。
- ④ 验证（**A5 定案 = 方案 a：warmup 预捕获，srs §5 义务**）：单次批量调用
  `mff(cands, branch_id=None, branch_ids=[f"{p}_cand{j}" for j in range(N)],
  parent_branch_id=p, step_idx=block_idx, attention_mask=ctx.mask,
  position_embeddings=ctx.pe, forced_divergent=forced)`——照常走 MFF 批量图
  分派（FR3 模式 MFF `use_cuda_graph=True`）：批量池（键 (N,T,prompt_len)）
  命中 → replay（`forced_static.copy_(forced)` + `tokens_static.copy_(cands)`）；
  未命中 → 既有捕获前置（#37 全条 + 父缓存完整 + forced 非 None →
  `forced_static` 分配）满足则**捕获入池**；前置失配/预算超限 → **AR016 矩阵
  eager 批量回退**（#41 契约：plain gate + torch merge；`_store_batched` 逐
  分支行切片发布——每分支 [1,T] 条目，folded_transformer.py:599-615；既有
  矩阵，不新增静默路径）。图路径发布走 `_publish_graph_children`（**AR018
  快照 + AR020 put_batch——批量发布面首次被生产回路行使**）。
  **A5 摊销论证（修正定案）**：block_size==small_block_size==32（srs §5 冻结）
  ⇒ 单次 sample 内每形状窗恰好 1 次 [N,T] 批量调用——**单发（生产单次
  generate）场景捕获摊销为负**（捕获 ≈ 2-3 次前向成本服务 1 次重放），此
  注记如实入工件 meta（单发场景建议关图）；但 srs H4 明文要求"批量图捕获>0
  （folded-batched）"且 FR3 期望行为明列 AR016 批量图+AR017+AR018/020 生产
  行使，且基准协议（warmup 一轮预捕获 + reps=30 + 池跨 sample 存续）下每窗
  批量 runner 服务 30 次重放——**摊销成立**，warmup 捕获成本不入 p50（srs
  FR4 明文）。故定案方案 a：保留 `use_cuda_graph=True` + 批量池，warmup 轮
  每窗首个合格批量调用自然预捕获（池未命中→捕获入池，零新机制）。
- ⑤ 选定（**单一定案准则，冻结**）：候选 j 的得分 = 其提交位（仍掩码位）上所选
  token 的概率（批量 logits right-shift 后切片当前小块、`gather`）之**最小值**；
  取得分最大的候选（min-prob 最大化——与阈值 0.95 的逐位置信语义同族、保守端）。
- ⑥ 提交：选定候选 j* 的 token 写入 `x` 仍掩码位；`finished |= (stop ∈ 提交位)
  .any(1)`（`fast_dllm_sampler.py:219` 同式）；跳出该小块剩余迭代。
  **链延续定案**：提交后折叠链切换为选定候选分支——`self._folding_parent =
  f"{p}_cand{j*}"`（链父追踪属性，diffusion_sampler.py:216-217；其激活已经
  图发布/`_store_batched` 发布为逐分支 [1,T,H] 条目，满足 #41 expand 契约），
  后续小块前向与 AR 步照 #26 以其为父链式新 id（父=选定候选给出精确前缀复用；
  forced 覆盖当前块整区的保守面仍叠加生效）。UT-2109 断言链父指向。
  AR 步与块推进照旧（AR 步 forced=当前块，见 (2)）。
- 退路（srs §2）：若 Green 期发现 FR3 语义风险不可单 AR 定案，T002 拆出为后续 AR，
  基准以 baseline/folded-serial 两模式先行（工件 schema 预留 folded-batched 空位）。

**（5）默认零扰动汇总**：无 folded_model（raw 全路径）+ `num_block_candidates=0` +
`forced_divergent=None` + 池关（默认）→ sampler/MFF/层/runner 全部既有面逐字节
不变（含 UT-1504、既有 sampler/eval/图 UT 零改动）。

**（6）FR4 基准设计**（`scripts/ar021_generation_loop_bench.py`，复用 AR019 助手
`_hf_env`/模型加载/detach；墙钟纪律 #39：CUPTI null 注记、宿主 `perf_counter`+
`torch.cuda.synchronize`、reps=30 预承诺单次定案不重跑）：
- **三模式构造（冻结）**：`baseline` = raw sampler（无 folded_model，现状路径）；
  `folded-serial` = sampler(folded_model=MFF_A(use_cuda_graph=True,
  graph_runner_pool_size=8))；`folded-batched` = sampler(folded_model=MFF_B(
  use_cuda_graph=True, graph_runner_pool_size=8, **graph_adaptive_capacity=
  True**), num_block_candidates=4, candidate_top_k=10)（N=4 取 AR019 工件矩阵值；
  A5 方案 a §4.2.2(4)④；**AR017 自适应容量在 MFF_B 接通**——srs §1.2"折叠栈
  全开"与 FR3"AR016+AR017+AR018/020 生产行使"由此兑现；AR017 为批量路径专属
  （#37），MFF_A（纯串行模式）无该面）。三模式共享同一 prompt 集、同一 seed、
  同一模型实例与 tokenizer；**两 MFF 实例均显式 `split_layers=False`**（Split
  层 resident hooks 挂在原始模型子模块上（split_layer.py:181-187），两实例共享
  模型且 split_layers=True 会叠加 hooks）。
- **基准规模（冻结，与池 cap 联动 §4.2.2(3)）**：prompt 集 **P=2**、
  `num_tokens=128`（**W=4** 形状窗）——池条目逐模式 ≤8（§4.2.2(3) 核算：
  folded-batched 串行 6/批量 8、folded-serial 串行 ≤8 数据依赖）容 cap=8
  零驱逐；ctx 8 条目 ≤ 32。
- **warmup（覆盖每形状窗捕获与自适应过渡期）**：每模式每 prompt 计时前完整
  sample 一轮（丢弃）——折叠两模式该轮自然完成全部形状窗的**串行+批量池
  捕获**（A5 方案 a 预捕获：每窗首个合格调用入池，零新机制；后续 rep 全程
  重放）+ Triton/JIT 预热 + **AR017 自适应过渡期**（MFF_B：若容量持续过/欠
  配置触发重捕获，warmup 轮内收敛——throttle/max_recaptures 既有机械；timed
  段若仍有重捕获事件，如实计数入工件并计入墙钟，不隐藏）。warmup 捕获成本
  不入 p50（srs FR4 明文）。
- **round-robin 交错（防漂移）**：逐 rep 三模式轮转（rep r 的模式顺序起始位
  = r%3 轮转），rep 间无额外间隔。
- **计时面**：逐块（sampler 侧 per-block `perf_counter` 锚点）+ 整代 generate
  墙钟双层记录；σ 报告绑模式名（AR019 先例）。
- **计数器出口（H4）**：MFF `graph_pool_stats`（§4.1.2：按形状键分组 captures/
  replays/evictions + **两类 eager 计数器**——`degradation_fallback_count` 与
  `structural_eager_count`（实例累计；timed 段 = warmup 后快照差分，快照语义
  在此声明）——脚本按 T→block_idx 映射直接序列化为 per_shape_window 字段；
  **captures 为实例累计（含 warmup，H4"批量图捕获>0"按此口径），replays
  取 timed 段差分**）；**no-fallback canary 口径（冻结，枚举闭包）**：MFF
  实例区分两类 eager——**降级类回退 := 全部非结构性 eager 回退**（已知矩阵
  成员列举：mask/pe `is` 失配、forced 身份失配、budget 超限 D>C 丢弃、
  capture-failed、scheduler/gate 族、**父缓存校验失败的非结构性成因**（串行
  捕获/重放时 parent 条目异常缺失（architecture_utils.py:793-794/:832-835）、
  批量侧 AR016 父再校验失配→eager）——设计回路内均为异常态，恰是 canary
  应捕获的静默降级面，实现按"非结构性"闭包判定而非清单成员匹配，杜绝枚举
  遗漏逃出 canary）与**结构性 eager**（窗口首前向 parent-None populate
  （:774-776）、窗口切换后首串行前向 parent 旧形状（:832-835）、末窗
  无 AR 步）；**canary 只查降级类（timed 段 = 0 全真）**，结构性 eager 如实
  单独计数入工件不计入 canary（否则每次生成的结构性 eager 恒 >0 → canary
  恒假 → H4 构造性自动 refuted）；账本 `write_op_count`（既有）；
  `GLOBAL_STABILITY_PROFILER` 既有面。
  **账本判定 c 冻结 = 4（归一化单位 = 每发布事件，设计期钉死）**：每串行
  前向发布 1 事件计 L+1（该事件 N=1 ⇒ L+1 ≤ 4·(L+1) 平凡成立——含起草
  前向（窗口切换后首串行前向=结构性 eager，发布账本同计 L+1，逐层 put，
  计数与图发布相同）与 AR 步前向）；每批量发布 1 事件计 O(L+N)（put_batch：
  foreach 组 + 立即执行，≤ 4·(L+N)）——**AR020 账本面首次被生产回路行使**。
  全代合计按事件求和保持各界。c=4 预承诺入工件 meta（AR020 同式）。
  **H4 串行子判据的数据依赖边界（声明冻结）**：H4"每形状窗串行图捕获>0 且
  重放>0（folded-serial）"的核验范围为**窗内存在 ≥2 次前向的窗**（窗内首
  前向恒结构性 eager，第 2 次起方可捕获/重放）；末窗（remaining=0 无 AR 步）
  或单迭代即完成的窗计数如实入工件、不计入该子判据（口径入 meta）——
  若核验范围内出现 0 捕获/0 重放窗，H4 如实 refuted，不静默。
- **质量面（H1，口径收窄至 srs 原文——两模式，非三模式）**：合成面 =
  mini-model bit-exact（UT-2105 钉死，不入工件）；真实面 = **AR019 H2 口径：
  folded-serial vs baseline** 输出 argmax 逐位一致（硬门）+ bf16 logits
  allclose（仲裁）；bit-parity 为**报告子字段**（如实记录，不作硬门）。
  **folded-batched 不入 H1 硬门**——其 min-prob 投机提交构造上即与 baseline
  分叉，块级 exact-match 率与置信度分布归入 H3 质量分面（纯报告，不预设
  quality-parity，srs 原文口径）。H2/H3 墙钟判定 σ=max（AR019 先例）；
  H3 = folded-batched vs folded-serial 三态（加速/持平/减速）+ 质量纯报告。
- **工件 schema**（`results/optimization/ar021_generation_loop_bench.json`）：
  `hypotheses_verdicts`（H1-H4 逐条：判定式全文 + supported/refuted/inconclusive
  + 数值）/ `mode_comparison`（三模式墙钟 mean/std/p50 per block + 整代）/
  `per_shape_window`（池捕获/重放/驱逐计数 + 两类 eager 计数（降级/结构性）
  per 窗 + AR017 重捕获事件计数 per 窗）/
  `quality`（argmax 一致率、allclose、bit-parity 子字段）/ `ledger`（账本计数
  与 ≤4·(L+N) 校验）/ `batch_canary`（AR020 同式）/ `meta`（reps=30、seeds、prompt 集 P=2 与
  W=4、warmup 策略、三模式配置全文（**强制散度活跃块与形状键控池声明**——
  srs FR4 验收明列）、`use_cuda_graph: true`（两折叠模式）+ A5 方案 a 注记
  （warmup 预捕获摊销成立；生产单发场景摊销为负、建议关图）、池 cap=8、
  VRAM 峰值 per 模式、驱动/transformers 环境面、CUPTI null 注记）。
- **`--quick` 冒烟**：reps=3、单 prompt、`num_tokens` 截断（2 块）、≤5min；
  全量 ≤60min（AR019 先例界）。

## 4.3 接口描述

```python
# actfold/models/diffusion_sampler.py（DiffusionSampler._forward 尾参扩展）
def _forward(
    self, x, branch_id, parent_branch_id, folded_model, step_idx,
    attention_mask=None, position_ids=None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    branch_ids: Sequence[str] | None = None,
    forced_divergent: torch.Tensor | None = None,
) -> torch.Tensor
# 仅 folded_model 路径透传三个新参（raw 路径行为逐字节不变——新尾参均带
# 默认值，raw 分支不读不传）；校验：forced bool 且 shape[-1]==x.shape[-1]
# （host 元数据，无 sync）→ ValueError

# actfold/models/fast_dllm_sampler.py
@dataclass
class FastDLLMSamplerConfig(SamplerConfig):
    ...  # 既有字段不变
    num_block_candidates: int = 0     # 0=关（默认，逐字节现状）；>0=FR3 批量投机
    candidate_top_k: int = 10         # 起草 top-k 池；<1 → 构造期 ValueError
class FastDLLMSampler(DiffusionSampler):
    def _fold_ctx_for(self, x: torch.Tensor, prompt_len: int) -> _FoldCtx: ...
    # shape+prompt_len 键控供给（§4.1.4/§4.2.2(1)）
    # sample() 内 folded 分支接线（4.2.2 (1)/(2)/(4)）

# actfold/models/architecture_utils.py（ManualFoldedForward）
def __init__(..., graph_runner_pool_size: int = 0) -> None: ...
@property
def graph_pool_stats(self) -> dict[str, Any]: ...
# 只读观测面：按形状键分组 captures/replays/evictions + 池大小 + 两类 eager
# 计数器（degradation_fallback_count / structural_eager_count，实例累计；
# 串行/批量分列）；池关时池部分为空 dict、计数器照常（§4.1.2；UT-2106 探针
# + 基准 per_shape_window 出口 + canary 口径 §4.2.2(6)）
def forward(
    self, tokens, branch_id, parent_branch_id=None, attention_mask=None,
    step_idx=0, branch_ids=None, position_embeddings=None,
    forced_divergent: torch.Tensor | None = None,   # 新尾参（AR019 pe 同式）
) -> Any
# 校验：bool dtype、shape 末维==tokens 末维、(串行 [T]/[1,T]；批量 [N,T])；
# None 不附加层 kwarg（零扰动）

# actfold/core/folded_transformer.py（FoldedTransformerLayer.forward 尾参）
def forward(..., forced_divergent: torch.Tensor | None = None) -> ...
# 四条 gate 路径 stable_mask 物化点（fused :272 / plain :274 / 变长 :297 /
# 批量 :526）之后、profiler.record（:300/:528）与 count 消费（:315/:540）之前：
# stable_mask &= ~forced（None 时跳过）；fused 计数弃用 → :316 fallback sum()

# actfold/core/cuda_graph.py（FoldedGraphRunner）
def __init__(..., forced_divergent_static: torch.Tensor | None = None) -> None: ...
# 捕获体：fused gate（:408-415）mask_buf 之后 AND（:415-419 间）；count_buf
# 自 AND 后掩码设备侧归约重导出（固定形状归约可捕获）
# replay：forced_static.copy_(forced)（仅当捕获时分配；捕获体外，零 sync）

# actfold/utils/config_manager.py（ActFoldConfig；校验循 __post_init__
# :146-177 ValueError 先例；BenchmarkRunner 透传点 benchmark_runner.py:191-225）
graph_runner_pool_size: int = 0   # 0=legacy 单 runner；>0=形状键控池上限（LRU）
# ≥0 校验，<0 → ValueError
```

外部接口（服务器/第三方）：无新增；对 HF transformers 的依赖面不变（remote 层
`position_embeddings` 消费为 AR019 既有契约）。

## 4.4 代码设计

```
actfold/models/diffusion_sampler.py        _forward +3 尾参（folded 透传；raw 行为
                                           零改动）
actfold/models/fast_dllm_sampler.py        +_FoldCtx/_fold_ctx_for（(T,prompt_len)
                                           键控供给，host 元数据 pad 区）
                                           +config 2 字段（含 candidate_top_k<1
                                           校验）；sample() folded 分支接线
                                           （FR2 forced 传递 + FR3 批量投机分支
                                           + 链延续切换）
actfold/models/architecture_utils.py       MFF：+graph_runner_pool_size、forward
                                           +forced_divergent 尾参、层循环透传、
                                           池查找/LRU/活跃 runner 别名（批量侧）、
                                           捕获身份含 forced 面（None/not-None）、
                                           cache setter 池传播扩展（#42）、
                                           +graph_pool_stats 只读观测面
actfold/core/folded_transformer.py         层 forward +forced_divergent 尾参（四路径
                                           AND：fused/plain/变长/批量；fused 计数
                                           弃用走 :316 fallback）
actfold/core/cuda_graph.py                 runner +forced_divergent_static（捕获体
                                           :415-419 AND + count 设备侧重导出；
                                           replay copy_）
actfold/utils/config_manager.py            +graph_runner_pool_size（≥0 校验循
（ActFoldConfig）                          __post_init__ 先例）；BenchmarkRunner
                                           透传（benchmark_runner.py:191-225）
scripts/ar021_generation_loop_bench.py     新基准脚本（复用 ar019 助手，导入顺序
                                           纪律；三模式/warmup/round-robin/工件
                                           schema 见 §4.2.2(6)）
tests/test_generation_loop_folding.py      UT-2101~2112（新文件）
results/optimization/ar021_generation_loop_bench.json   工件（T004 产出）
```

模块化要点：sampler（配方与上下文供给）/ MFF（折叠编排与图池）/ 层（gate 语义）/
runner（静态捕获体）四层解耦；forced 语义的最小知识分布——sampler 只知"哪块活跃"，
层只知"AND 进稳定掩码"，runner 只知"静态缓冲拷贝"；池对 #37 前置零放宽。

# 5 重构设计

无既有代码整理点（`_forward` 尾参扩展为纯增量；`FastDLLMSampler.sample` 的 raw 分支
零改动——folded 分支为新增 else 面）。

# 6 测试设计

## 6.1 单元测试（UT，tests/test_generation_loop_folding.py，UT-2101~2112）

- **UT-2101（FR1）**：mask 构建语义——块因果∧非 pad（pad 列恒 False、块内双向、跨
  块因果；pad 区由 host 元数据 (prompt_len, first_block_padding) 构建）；同
  (T,prompt_len) 对象钉定（`is` spy）+ 同 T 不同 prompt_len **必然分键**（Major
  对应面：不同 pad 数不得命中同条目）；正常输入下与 raw `x!=pad_id` 掩码**逐位
  等价**断言（病态 prompt 含 pad_id 的差异为文档化边界，不入硬门）；raw 路径
  零扰动（仍 `x!=pad_id`、不传 pe）。
- **UT-2102（FR1）**：pe 供给——rotary 模型 → (cos,sin) 线程化且钉定；无 rotary →
  None（`_recompute_all` **kwargs 层签名过滤，§4.1.4）；折叠路径 B>1 →
  `ValueError`。
- **UT-2103（FR2 零扰动）**：`forced_divergent=None` 默认——MFF forward/层/runner
  全既有面 bit-exact（与 AR020 末态对拍）；UT-1504 面池关时逐字节。
- **UT-2104（FR2 eager 语义，四 gate 路径全覆盖）**：forced 位恒重算（父 FFN 不
  复用探针：cache fetch 该位不命中/merge 重算路径计数）、非 forced 前缀复用；
  forced 广播 [T]/[1,T]/[N,T]；**fused 路径专项**（`SimilarityGate`+cosine+B·T≥1024
  触发 `fused_gate_mask_count`）：AND 后计数走 `stable_mask.sum()` 重导出（无过期
  计数；三路快分不误走全稳快径）；profiler record 消费 AND 后掩码（stable 比例
  断言）。
- **UT-2105（FR2 parity）**：**合成块因果 mini-model（双形态掩码契约）**——层
  attention 同时接受 raw 调用形态（`[B,T]` padding 掩码，内部构建块因果∧pad）
  与折叠调用形态（`[T,T]` 预构建块因果∧pad），两种形态下**同一 attention 实现**
  语义等价（与 remote 行为同构，满足 srs 语义匹配义务）：同 seed 下 folded-serial
  生成 token 序列 == raw **bit-exact**；前缀复用探针>0、活跃块重算探针=块 token
  数。
- **UT-2106（FR2 池）**：池关=legacy 逐字节（形状失配 warn-once）；池开=跨 ≥2 形状窗
  各捕获 1 runner（`graph_pool_stats` 探针）、同窗重放>0、零 UserWarning；LRU=
  上限驱逐+VRAM 探针（`torch.cuda.memory_allocated` 驱逐后不增长）；批量池键
  (N,T,prompt_len)（同 (N,T) 异 prompt_len 分键）；**cache setter 池传播**（#42：
  `mff.cache = other` 后池内全部 runner `_cache` 均切换——含非活跃池成员）。
- **UT-2107（FR2 图 forced 缓冲）**：捕获后改 forced 内容重放 → 重算区随内容变化
  （设备侧 stable 计数探针）；None/not-None 捕获身份失配 → 静默 eager；捕获体内
  count_buf 重导出与预算校验一致（forced 位计入 divergent 预算）。
- **UT-2108（FR3 零扰动）**：`num_block_candidates=0`（默认）——与既有 sampler 输出
  bit-exact（含图/无图、含停止 token 场景）。
- **UT-2109（FR3 批量流）**：N=4——每小块恰好 1 次 `[N,T]` MFF 调用 + 1 次批量
  发布（CUDA 设备图路径 = put_batch（`_publish_graph_children` spy/账本，
  AR020 面生产行使）；CPU/回退路径 = `_store_batched` 逐分支 [1,T]）；候选 ⊆
  top-k 起草集合（**种子公式可复现**：
  §4.2.2(4)② 冻结式，跨运行同 seed 同候选）；选定 = min-prob argmax（构造已知
  概率的对手测试）；提交后跳过剩余迭代（前向计数）；**链延续**（提交后折叠链父
  = 选定候选分支 id，其缓存条目存在于 batch 1，后续前向以其为父——探针断言）；
  停止 token → `finished`/pad 填充与基线同式。
- **UT-2110（FR3 异常）**：N>0 且无 folded_model / folded_model 不支持 branch_ids /
  N<0 / candidate_top_k<1 → 构造期 `ValueError`，不静默降级。
- **UT-2111（FR2 AR 步）**：块推进 AR 前向 forced=当前块（提交块重算探针=块 token
  数；跨块形状窗新 runner 捕获）。
- **UT-2112（供给纪律）**：`_fold_ctx_for` 跨 sample 调用同 (T,prompt_len) 同对象
  （多 rep 命中）；同 T 不同 prompt_len 分键（不同 pad 模式不共用条目）；ctx
  构建零 device readback（spy `.item/.tolist/.cpu` 零调用）。

## 6.2 接口测试

新尾参签名兼容面：既有调用方（`_forward` 的全部既有调用点、MFF forward 的全部既有
调用点、层 forward 既有调用点）零改动通过（全量回归覆盖）；`ActFoldConfig` 新字段
YAML 载入/校验（config 既有 UT 面扩展）。

## 6.3 业务场景测试

真实 checkpoint 三模式生成（FR4 基准 = 业务场景级，设计全文见 §4.2.2(6)）：同一
prompt 集与 seed，baseline / folded-serial / folded-batched（三模式构造冻结）；
warmup 覆盖每形状窗捕获与自适应过渡期（MFF_B）、逐 rep round-robin 交错、
逐块+整代墙钟；质量面（H1 硬门=folded-serial vs baseline：AR019 H2 口径
argmax 逐位一致 + bf16 allclose 仲裁，bit-parity 报告子字段；folded-batched
块级 exact-match/置信度归入 H3 质量分面纯报告）
+ 按形状窗分组计数器 + 账本（c=4 冻结）+ canary 入工件（H1-H4 判定式全文）；
`--quick` ≤5min / 全量 ≤60min。

## 6.4 异常场景测试

- 折叠路径 B>1（显式 `ValueError`）；pe 形状/头维失配（AR019 entry `ValueError` 契约）
  ；池驱逐后同形状重现（再捕获，正确性无虞）；捕获失败（家族 flag 语义不变）；
  FR3 构造期非法配置（UT-2110）；批量调用异常原样上抛；显存不足（基准工件如实记录
  失败配置）。

# 7 实现澄清（post-gate，T002 落地追记，文档级——语义与 rev 5 门控定案一致）

> T002 Green 落地时由冻结 UT 面收敛的实现载体/细化点，追记于此供 T007 review 与
> ST 对照；不改变 §4.2.2 门控定案的任何判定式。

- **池键实现载体**：§4.2.2(3) 的语义键 (T,prompt_len)/(N,T,prompt_len) 在 MFF 侧
  的实现载体为 `(tuple(tokens.shape), attention_mask 对象)`——MFF 不直接知晓
  prompt_len，而 `_fold_ctx_for` 的对象钉定使 (T,prompt_len) ↔ mask 对象在窗口
  存续期内双射：同窗同对象必同键命中（`is` 恒真由构造保证）、同 T 异 prompt_len
  必产生不同 mask 对象而分键；池以 mask 对象为键持引用，杜绝 gc/id 复用。stats
  dict 同键持久（驱逐不删 stats，`evictions` 可观测）。
- **全稳快路径父条目防逐出回写**：全稳子条目与父条目冗余，池预算冲突时快路径以
  `contains` 探测 + `put` 回收保留父条目（UT-2104 在 T=预算上限下依赖此语义；
  兼容性由既有 pin 测试 `test_manual_path_never_reads_folding_context` 覆盖）。
- **驱逐时 cuBLAS workspace 回收**：LRU 驱逐 runner 后调用
  `torch._C._cuda_clearCublasWorkspaces()`——capture 侧流泄漏的 ~8.5MB/次
  workspace 是 UT-2106 VRAM 守恒断言（驱逐后 `memory_allocated` 不增长）的唯一
  可满足机制；活图烘焙的 workspace 位于各自私有池不受影响。
- **UT-2101 测试设备卫修**：T001 冻结测试原文 `causal`（CPU）∧ `raw_kv`（CUDA）
  在 `.to(device)` 之前执行，任何实现在 CUDA 主机必抛 RuntimeError；T002 经主代理
  裁定应用单行卫修（`causal` 先 `.to(device)`），断言语义零变化（临时副本预验证
  20/20 后落盘）。
- **ctx LRU 与池条目生命周期（T007 Nit-1 注记）**：ctx 缓存上限 32 条、池上限
  独立（基准 cap=8）——ctx 驱逐后同 (T,prompt_len) 窗重现会构建新 mask 对象，
  池 `is` 钉定校验失配 → 静默 eager → 重捕获新池条目（旧条目待 LRU 驱逐）。
  该路径仅浪费一次捕获的 VRAM/时间（P=2/W=4 规模下 ctx 8 条 < 32 不触发），
  无正确性风险；生产大 prompt 集场景为已知非最优面，留待后续 AR。
