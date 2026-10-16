# [AR023] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR023 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成）|
| 创建日期 | 2026-10-15 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 模型探针（先决）：`scripts/ar023_family_probe.py` + 工件 `results/optimization/ar023_family_probe.json`——两 checkpoint curl 下载（清单+SHA256）、transformers 4.53.1 加载、bf16 VRAM 实测、bitsandbytes 8bit Windows 可行性、短生成冒烟、graph 捕获余量、prompt 适配；结论字段（dtype_choice/graph_enabled/degraded）定案后续任务参数 | - | passing | 降级规则见 srs §2；~30GB 下载时长预算；两轮重跑定案：**两族均 bf16 / degraded=False / graph_enabled=False**（llada bf16 15265MB/int8 10966MB；dream bf16 14836.7MB/int8 12093MB；[T001-3] 探针 bool mask 修复后） |
| T002 | Red UT：`tests/test_llada_speculative.py`（UT-2301~2308）+ `tests/test_dream_speculative.py`（UT-2309~2315）机制面测试全红（红因=生产缺失：SpeculativeCore/族投机路径/config 字段）；冻结面（`test_diffusion_samplers.py` T018 参照对拍 + 基础用例）零破坏确认 | T001 | passing | 覆盖点矩阵见 design §6.1；探针结论影响 fixture dtype 面 |
| T003 | Green LLaDA 全链：`speculative_core.py` 共享机制核心（候选构建/min-prob 选定与提交/链延续/greedy_win_stats/_supports_branch_ids/ctx LRU 骨架——冻结常数单一来源，与 fast_dllm 实现常量对拍 UT 钉死）+ `llada_sampler.py` block 单元适配（右填充 ctx 2-D pad 列 mask/block 超集 forced/B==1 限制）+ **FR4 配置面**（`LLaDASamplerConfig` 新字段 + `ActFoldConfig` 消费面 + per_model YAML）；mutation 全杀还原 | T002 | passing | 默认关 byte-identical；design §4.1.2/4.1.3；mutation 14/14 杀尽（含 2 处 UT 强化钉死） |
| T004 | Green Dream 全链：`dream_sampler.py` super-step 适配（s=8 预算合并/confidence 选位/提交位 forced）+ `_forward` 折叠分支 `position_ids` 尾参扩展（AR019 None-attach 模式）+ MFF `architecture_utils` position_ids 透传（探针 Dream 面验证后启用，不可行则折叠 degraded 注记）+ **FR4 配置面**（`DreamSamplerConfig` + `speculative_steps` + YAML）；mutation 全杀还原 | T002 | passing | 与 T003 并行面，机制共享（SpeculativeCore 已在 T003 交付）；design §4.1.4；mutation 13/14 KILLED + D10 justified survivor（batch-1 键双射，无行为面） |
| T005 | 基准脚本：`scripts/ar023_cross_family_bench.py --family llada\|dream`（单脚本双族，复用 AR022 四模式框架+导入面）+ `--quick` 冒烟 | T003, T004 | passing | 判定式冻结入脚本；Green 9/9 + 两族 --quick 实机冒烟通过（详见进度记录）；期生产缺口 3 修（MFF 元组解包 UT-2335/posthoc 列张量/sidecar profiler reset） |
| T006 | 工件定案：非 degraded 族四模式 reps=30 预承诺单次定案（每族一工件：`ar023_llada_bench.json`/`ar023_dream_bench.json`）；degraded 族基准面 blocked+注记 | T005 | pending | 墙钟纪律 #39；复现自检 |
| T007 | 质量门：全量 pytest（≥951/4/0 只增不减）+ pyflakes/mypy/black + demo 三锚点不动确认 | T003, T004 | pending | |
| T008 | 文档五件：README 全量中文重写（强制规则）+ AGENTS 新条目（跨族投机契约）+ GUIDE 新部分 + CHANGELOG + RERUN 新步骤（curl 下载/量化/探针/基准复跑） | T006, T007 | pending | 数字工件溯源 |
| T009 | review 门控（只读子代理：spec/design 合规 + 代码审查），全部问题（含 Minor/Nit）修复循环至 PASS | T008 | pending | |
| T010 | ST 验收：st-cases.md 生成 + 执行（正常/边界/异常/回归四类，含降级路径用例） | T009 | pending | |
| T011 | 归档 `specs/archive/AR023-cross-family-speculative/`，本地 commit + gitee 分段推送（>96KB 代理限，commit-tree 部分树快进链，一次性 read-tree 后逐批累加——AR022 T009 教训）+ 双端树验证 | T010 | pending | 永久 commit id 分歧预期内 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-15 T001（探针交付 + 实机执行）

- **T001 Green 交付**：`scripts/ar023_family_probe.py`（纯逻辑面 + 七面真实探针惰性导入 + CLI）+ `tests/test_ar023_probe.py` 17 测（UT-2301~2317）一次全绿；lint 三件 clean；基线 968 passed。
- **[T001-1] repo id 修正**：`GSA-SAI/LLaDA-8B` 全 org 401（错误 id）→ 官方 `GSAI-ML/LLaDA-8B-Base`；design.md §4.1.7 + §7 追记 + PROBE_REPOS + UT 期望同步修正。
- **bitsandbytes 0.50.2 装入 ar019 venv**（pip 经代理成功）→ int8 面有实测条件。
- **首轮实机执行**（ar019 venv + 代理，两 checkpoint ~30GB 下载 + SHA256 全录）：**LLaDA 全面可用**——bf16 OK（15265MB VRAM）、int8 OK（10950MB）、smoke 生成连贯文本（mask_token_id=126336）；**graph 捕获不可行**（Windows torch 2.5.1 cu121 + remote-code LLaDA：`captures_underway.empty()` INTERNAL ASSERT → graph_enabled=False，与 AR022 单发不启用图建议一致）。
- **[T001-2] 隔离缺陷修复**：graph 面的分配器断言损坏进程级 CUDA 状态 → LLaDA prompt 面与 Dream 全部 GPU 面同错误失效（首轮 Dream degraded=True 为进程顺序伪影，非族不可行）。修复：CUDA 五面（bf16/int8/smoke/graph/prompt）改由子进程执行（`--_face-worker` 哨兵协议 + `wrap_cuda_faces_with_subprocess`，download/load 保留原 callable）；Red UT-2318~2325（8 测）→ Green 一次全绿（25/25）；lint 三件 clean。附：curl `-C -` 对已完成文件返回 416/rc=0 → 重复执行时 download 面自动跳过重下、仅重验 SHA256。
- **重跑进行中**（子进程隔离版，工件覆写 `results/optimization/ar023_family_probe.json`）。

### 2026-10-15 T002（Red UT）

- `tests/test_llada_speculative.py`（1492 行，UT-2301~2308）+ `tests/test_dream_speculative.py`（1006 行，UT-2309~2315）交付：35 测函数全红（红因=生产缺失：13× ModuleNotFoundError speculative_core、16× TypeError 缺 config 字段、4× AttributeError、2× 意向断言）；冻结面 `test_diffusion_samplers.py` 17 passed 零破坏。Red 子代理 ses_ec02f460effesdumD3fq3zcrBK。

### 2026-10-16 T003（Green LLaDA 全链）

- **Green 交付（19/19 绿）**：
  - `actfold/models/speculative_core.py` 新建：`SpecCtx(mask, position)` dataclass；`SpeculativeCore` mixin（无 `__init__`，宿主注解 config/_greedy_wins/_greedy_spec_calls/_folding_parent）；`greedy_win_stats` 只读 property；`_supports_branch_ids` 静态（照抄 fast_dllm:325-349）；`_ctx_cache_get/_ctx_cache_put`（LRU cap=`_MAX_FOLD_CTX_ENTRIES`=32 从 fast_dllm 单源导入）；`_speculative_commit`（B!=1→ValueError；空 masked_local 早退；topk→draft_set[0][masked_local]；冻结种子公式+专用 CPU generator；forward_fn 调用不传 attention_mask；min-prob 评分域=fill positions；commit/finished/链延续/history）。
  - `llada_sampler.py`：LLaDASamplerConfig + num_block_candidates/candidate_top_k/candidate_include_greedy + `__post_init__` 校验（N<0/k<1/include_greedy+N=0→ValueError）；`LLaDASampler(DiffusionSampler, SpeculativeCore)`；`_spec_ctx_for`（键 (T,prompt_len)，mask=[T,T]=(arange<valid_end).repeat，position=None）；`_build_unit_forced`（block 超集 [T] bool）；sample() 入口校验（N>0：无 folded_model/不支持 branch_ids/B>1/cfg_scale>0 均 ValueError，cfg 校验为超出 design 的防御性新增——投机路径不实现 CFG，拒绝静默丢弃）；block 循环内 spec 分支（draft=`_next_folding_branch`+`_forward` 带 ctx.mask/position_embeddings/forced_divergent；suppress→right_shift→begin_suppress 同参照序；`_batched_forward` 闭包）；N=0 参照路径逐字节未动。
  - FR4 配置面：`config_manager.py` ActFoldConfig + `num_block_candidates=0`/`candidate_top_k=10`；`llada_8b.yaml` 注释投机段（默认关）。
  - lint 三件 clean；mypy 全 actfold 68 文件 clean；冻结面+probe 42 passed。
- **Green 期修 4 处 test-own fixture bug**（Red 合同语义不变）：①UT-2301 `_replay_random_rows` 调用 off-by-one（`n-1`→`n`、比较 `[j]`）②UT-2304 `host(num_block_candidates=2)`（CraftedRecorder 2 行 vs N=4）③`_PosTableHead` 广播 bug（`logits[...,idx]=vals` 覆盖全行致 argmax 恒 1）→ 改行索引数组 `logits[:, rows, ...]`（llada+dream 两文件同步修）④`_CraftedRecorder.__call__` 未记录 calls → 补 append。UT-2303 trap 实证无 bug（首匹配 base 1/2 两 draw 恰相等）。
- **mutation 14/14 杀尽**（M1-M14：种子常数×4/greedy 行/min→max/fill 域/eos/链/S0 回退/greedy 门/LRU cap/forced 超集）：首轮 12/14 → 2 SURVIVED 均为合法未钉死语义 → UT 强化：UT-2301 补 seed=4000 面钉 31-bit mask（(4000·1_000_003)&0x7FFF_FFFF=1_852_527_552≥2^30，30-bit 掩码突变体产生不同 draw）；UT-2304 补 min-vs-max 判别器（双高/双低行 vs 均匀中排行，min 选均匀行、max 突变体选高低行）→ 复跑 14/14 KILLED，还原后 19/19 复绿。
- **全量基线**：998 passed / 13 failed / 4 skipped（canonical deselect 口径）——951 基线只增不减成立；13 失败全部为 test_dream_speculative.py 红（T004 范围），另 3 个 Dream 测因 SpeculativeCore 存在而自然转绿（预期内）。

### 2026-10-16 T001（重跑 1 结论 + [T001-3] mask dtype 修复）

- **重跑 1（子进程隔离版）完成**：LLaDA 定案——`dtype_choice=bf16`（15265MB）、`graph_enabled=False`（Windows torch 2.5.1 cu121 + remote-code：`captures_underway.empty()` INTERNAL ASSERT，独立复现非伪影）、`degraded=False`；prompt 面 2-D mask 接受（T=21/26 两形状）。
- **Dream 仍 degraded=True——但为探针 fixture bug 非族缺陷**：bf16/int8 面报 `Expected attn_mask dtype to be bool or float or to match query dtype, but got attn_mask.dtype: __int64`；smoke 面 OK（generate() 内部自建 mask）→ **[T001-3] `_short_forward_ok` mask dtype 修复**：`torch.ones_like(ids)`（int64）→ `torch.ones_like(ids, dtype=torch.bool)`（scripts/ar023_family_probe.py:508；bf16/int8 面共用；graph/prompt 面本已 bool）；独立 GPU 实证（ar019 venv 手测）：Dream bf16 raw forward bool mask OK finite=True shape=(1,16,152064)、float mask 同 OK——**族本身 bf16 可行**。Dream load 面 transformers==4.53.1 / model_type=Dream / vocab=151667；prompt 面 position_ids 接受（T=21/27 两形状，左填充）→ T004 折叠透传无阻塞。
- **重跑 2（bool mask 修复版）后台进行中**（PID 34476，日志 `D:\ar023_models\probe_rerun2.log`，工件覆写 `results/optimization/ar023_family_probe.json`）：预期 Dream `dtype_choice=bf16`、`degraded=False`（手测已证 bf16 可行；int8 待重跑确认）。

### 2026-10-16 T001（重跑 2 定案）

- **重跑 2 完成定案**（工件 `results/optimization/ar023_family_probe.json` 覆写，日志 `D:\ar023_models\probe_rerun2.log`）：**两族均 degraded=False、dtype_choice=bf16、graph_enabled=False**——llada bf16 15265MB / int8 10966MB；dream bf16 14836.7MB / int8 12093MB。Dream prompt 面 position_ids 接受（T=21/27 左填充）→ T004 折叠透传无阻塞。T001 定案 passing：无 degraded 族，T006 两族均需 reps=30 定案基准。

### 2026-10-16 T004（Green Dream 全链）

- **Green 交付（16/16 绿）**：
  - `dream_sampler.py`：`DreamSamplerConfig` + `num_block_candidates=0`/`candidate_top_k=10`/`candidate_include_greedy=False`/`speculative_steps=8` + `__post_init__` 四校验（N<0/k<1/include_greedy+N=0/s<1 → ValueError，消息与 LLaDA 对称）；`DreamSampler(DiffusionSampler, SpeculativeCore)` + 计数器；`_spec_ctx_for(x, prompt_len)`（键 (T,prompt_len)，mask=`_attention_mask` bool，position=`compute_position_ids(mask)` 恒非 None——batch-1 无填充时=arange，与模型内部默认等价）；`_build_unit_forced`（[T] bool 精确集）；sample() spec 超步路径：入口校验（N>0：无 folded_model/不支持 branch_ids/B>1/cfg_scale>0 → ValueError）；while step<effective_steps；掩码数=0 早退；预算合并 `int(sum[0,step:step+s])`、`k_eff=min(budget,masked_count)`、≤0 skip step+=s；draft=`_next_folding_branch(dream_spec_u{unit})` 串行 folded 调用带 ctx.mask/position；right_shift 后 `sample_tokens` confidence topk 选位（full_confidence -inf 模板）；`_batched_forward` 闭包绑 ctx；`_speculative_commit(region_start=0, masked_local=selected, sub_idx=0)`；finished break；step+=s；**N=0 参照 for 循环逐字节未动**。
  - `cuda_graph.py`：`FoldedGraphRunner` `position_ids_static` 尾参（by-ref 存储）+ `_static_forward` extra_kwargs dict（pe/position_ids attach-only-when-non-None）。
  - `diffusion_sampler.py`：`_forward` 折叠分支 `fold_kwargs["position_ids"]`（None 无 kwarg，docstring 注记 AR023）。
  - `architecture_utils.py` MFF：`forward(..., position_ids=None)` 尾参；host 元数据校验（int dtype、dim1 或 dim2 尾维=seq、首维∈{1,batch}）；eager `pos_ids_kwargs` 进层循环；**四构建面全线程**（`_graph_position_ids`/`_graph_batched_position_ids` init pin、serial/batched installers、capture cuda 检查 + `position_ids_static=`、pool/非 pool replay `is` 校验、post-replay 别名刷新、`_observe_adaptive_capacity`/`_recapture_batched_runner` 透传）。
  - `dream_7b.yaml`：注释投机段（默认关）。
- **Green 期修 2 处 test-own fixture bug**（Red 合同语义不变）：①`_SpyMFF` args 按引用记录 + sampler/Core 原位改画布 → 记录被追溯污染 → 改 **tensor args clone 快照**（shift_convention 失败签名实证：candidates 在 {3..6} 持 MASK 而记录画布被改为已填值）②budget_merge 期望张量 `torch.tensor([7,8,9,10])`/`[3,4,5,6]` 补 `device=device`（跨设备 torch.equal RuntimeError）。
- **UT 强化（mutation 预钉）**：UT-2311 补 same-T-different-prompt_len 面（prompt [1,2,3,4]+num_tokens=7 → T=11 key (11,4)≠(11,3)，`ctx_p4 is not ctx_a`）；UT-2310 补 kwargs 钉（drafts/batched `attention_mask is family_mask`、`position_ids is ctx_used.position`）+ `_KwargsRecordingLayer`（记录层 kwargs 后委托 super）钉 MFF eager 层级 position_ids 到达与值=`compute_position_ids(family_mask)`。
- **mutation 13/14 KILLED + 1 justified survivor**（脚本 `C:\Users\l30086046\AppData\Local\Temp\opencode\ar023_t004_mutation.py`，14 突变体 D1/D3/D5/D6/D7/D8/D9/D10/D11/D13/D14/D15/D17/D19）：
  - 首轮 12/14 → **D3（step += s → += 1）SURVIVED**：均匀调度（1/步）下重叠再切窗口的预算序列与正确划分巧合收敛（min 钳位吸收）→ UT-2310 补 **crafted 非均匀调度面**：monkeypatch `dream_sampler.get_num_transfer_tokens` 返回 `[[1,1,1,1,4]]`（8 masked、s=2）——正确划分 [0:2]/[2:4]/[4:6] 预算 2/2/4 = 3 超步（6 前向），突变体重叠再切需 4 超步（8 前向）→ 前向计数断言杀；`get_num_transfer_tokens` 零列剔除使自定义调度器无法构造零头调度，monkeypatch 注入为唯一可行面 → 复跑 **13/14**。
  - **D10（ctx 键 prompt_len→0）justified survivor**：Dream batch-1 画布 `T = prompt_len + num_tokens` 恒成立（build_left_padded_canvas 仅 batch 内补齐）⇒ 键第二分量在单实例上与 T 双射（纯 T 键行为等价；跨实例 `_spec_ctx_cache` 不共享）；`(T, prompt_len)` 为设计冻结宿主元数据键形式（与 AR021 `_fold_ctx_for`/LLaDA 面对称，面向未来 batch/padding 支持），无行为面可杀。
  - 还原后 16/16 复绿 + 全量基线复绿。
- **lint 三件 clean**（两 spec 测试文件经 black 26.10.1 归一——black 版本更新后旧行换行风格非规范）+ mypy 全 actfold 68 文件 clean。
- **全量基线**：1011 passed / 0 failed / 4 skipped（canonical deselect 口径）——951→968→998→1011 只增不减成立；13 Dream 红全转绿。

### 2026-10-16 T005（Green 基准脚本 + 两族实机冒烟）

- **Red 交付**（ses_ebf50e0ddffeLOlRzPqOw73B4D）：`tests/test_ar023_bench.py` 9 测（UT-2326~2334）全红（ModuleNotFoundError）——协议常数/轻量导入合同（subprocess 钉 torch/transformers/bitsandbytes 不入模块顶层）、FAMILY_RECIPES 冻结、h1 分级质量门（三硬面+None 规则+gap_bat==0 哨兵+compared==0+非阻塞 strong 层）、h2 严格 >1.0、h4 canary/ledger 合取+graph 中性注记+stable ratio 报告面、reproduction 严格行等值+mark_all_blocked 不变异、probe_conclusion 安全降级+blocked 工件 schema、parse_args CLI 合同、输出路径解析+build_artifact schema。
- **Green 交付**：`scripts/ar023_cross_family_bench.py`（~1030 行）——纯逻辑层（冻结判定式/CLI/schema/blocked，CI 可测）+ 懒加载运行层（探针式 bf16→int8 加载、_BenchMFF 串行/批量计数器、四模式构造（baseline=官方配方 folded=None；folded-serial=N=0+MFF；folded-batched=N=4/top_k=10；folded-batched-greedy=+include_greedy，每模式独立 MFF 实例 AR022 MFF_C 模式）、baseline 双跑复现自检、warmup+每 rep 四模式轮转（rep r 起点 r%4）宿主 perf_counter+sync（#39）、sidecar ledger（write_op_count delta vs LEDGER_C=4·((L+1)·serial+(L+N)·batched)）+stable ratio（每模式窗口 GLOBAL_STABILITY_PROFILER.reset 后全窗口 profile 均值）、质量面（decode 长度至首 EOS 含/posthoc 置信度单次 raw forward（Dream right_shift/LLaDA 直取）/block 32 网格 exact 计数/位置匹配）、工件 meta（checkpoint SHA256/dtype/transformers/torch 版本/族常数/cross_rep_deterministic/CUPTI 注记）。
- **Green 期 2 处 Red 测试自身缺陷修正**（合同语义不变，注记入测试文件）：①UT-2328 边界 0.5：IEEE-754 表示噪声（0.30−0.40 与 0.20−0.40 的比值=0.5000000000000002）→ 冻结 `<= 0.5` 比较加 `_H1_RATIO_EPS=1e-9` epsilon（module 常数+docstring 注记）；②UT-2329 `equal` fixture 自相矛盾（构造 (440.9, 440.9, 450.0) 却循环断言 p50_baseline_ms==1300.6，任何不伪造实现均不可满足）→ 共享钉移至 (fast, wild)、equal 钉自身记录值。
- **Green 期生产缺口 3 修（TDD）**：
  - **[T005-1] MFF eager 循环元组解包**（实机首跑实证）：LLaDA 远程 block 返回元组 → `FoldedTransformerLayer._pack_output` 重包 `(hidden, None)`（AGENTS #13 为原地 FoldedModel 控制流设计）→ MFF 自有 eager 循环把元组直喂下一层 → 真实 LLaDA-8B 折叠首前向 `'tuple' object has no attribute 'device'`（Fast-dLLM 远程层返回裸张量故 AR019~AR022 未触发）→ Red UT-2335（`LlamaLikeTupleModel` 元组块 MFF 串行折叠 vs 原模型 forward 位精确，CI 复现同失败模式）→ Green：MFF 层循环 `isinstance(x, tuple): x = x[0]`（`_recompute_all` 同约定）；**graph 静态体（cuda_graph.py 自有层循环）未修**——两族探针结论 graph_enabled=False 不入本 AR 面，已知注记。
  - **[T005-2] posthoc 列→张量**：质量面行来自 `.tolist()`（Python list），`seq_row[None, :]` 张量式索引崩溃 → `torch.as_tensor(seq_row, dtype=long, device=...)`（实机二跑实证）。
  - **[T005-3] sidecar stable ratio 窗口化**：采样器每 `sample()` 重置折叠链（branch id 跨调用复用），"新增 branch"探测恒空 → 每模式 sidecar 窗口前 `GLOBAL_STABILITY_PROFILER.reset()`，窗口内全 profile 归属该模式（构造性）。
  - 两次崩溃均产出诚实 blocked 工件（无伪造数字）——FR5 blocked 语义实机验证 ✓。
- **两族 `--quick` 实机冒烟通过**（ar019 venv + Quadro RTX 5000，工件 `ar023_{llada,dream}_bench_quick.json`，reps=3/单 prompt/num_tokens=64）：
  - **LLaDA**：bf16、repro ok=True、cross_rep_deterministic=True、ledger 三模式全 within bound、stable ratios 0.491/0.391/0.391；p50：baseline 59.3s / folded-serial 51.7s / batched 11.0s / aligned 11.0s——**H2 5.41×**；H1 refuted（batched 区域 offset 2 EOS 早停 vs baseline 64 全长，greedy 锚定 quick 尺度未恢复，ratio 1.0/exact 0.0）；H4 supported。
  - **Dream**：bf16、repro ok=True、ledger 全 bound、stable ratios 0.397/0.340/0.569、sidecar 8 窗口（s=8，1:1/64 预算合并 ✓）；p50：baseline 82.5s / serial 68.2s / batched 58.3s / aligned 61.2s——**H2 1.35×**；H1 refuted（面 a ratio 0.861>0.5；面 b 长度恢复 ✓（64/64/64 无早停分歧）；面 c exact 0.047→0.0625 严格改善 ✓）；H4 supported。
  - 冒烟尺度 verdicts 仅为机制验证，**非预承诺定案**——T006 正式单次 reps=30（P=2/128 tokens）为预承诺判定面。
- **lint**：black/pyflakes 改动文件 clean；bench 脚本 mypy clean（algo_experiments.py 7 处预存错误不在改动范围）；测试文件 black 26.10.1 归一。
- **全量基线**：**1022 passed / 0 failed / 4 skipped**（canonical deselect 口径）——951→968→998→1011→1022 只增不减成立（+9 bench UT-2326~2334、+UT-2311 Dream N=0 position_ids 缺口修复新增、+UT-2335 MFF 元组解包架构面）。
- **T005 定案 passing**：T006 两族均非 degraded → 均需正式 reps=30 定案基准（预承诺单次定案，墙钟纪律 #39）。

### 2026-10-16 T007 初轮（系统 python 环境定性 + llada 正式基准进行中）

- **环境背景**：AR023 llada 正式基准（reps=30，T006）正在本机 ar019 venv 运行（12:35 起，bf16 LLaDA-8B）——T007 完整执行（全量 pytest/四件套/demo 锚点）须待基准完成并在 ar019 venv 复跑，避免负载噪声污染性能类用例（BS007 教训）+bench 依赖完整口径（前会话 canonical 基线 1022/0/4 即 venv 口径）。
- **系统 python（C:\Python311）初轮**：1021 passed / 3 failed / 4 skipped（729s）。3 failed 定性：2×`ModuleNotFoundError: lm_eval`（`test_benchmark_runner_real_judges`/`test_lm_eval_judge_prompts_and_score`——系统环境未装 bench 依赖，前会话 1022/0/4 在装全 lm_eval 的 ar019 venv 取得，环境漂移非代码回归）+ 1×性能类单测在 llada 基准 CPU/GPU 竞争负载下失败（同类 BS007 单独重跑 10min 超时不可判——负载噪声，机器空闲+venv 环境后复验）。`.pytest_cache/lastfailed` 另含 2 条陈旧缓存项（UT-1403 已不存在于当前测试集）非本次失败。
- **AGENTS.md 双规则增补**（用户指令落实）：①Experimental Validation Rule（每个创新 AR 的有效性主张必须经预注册假设/reps≥30/多几何/同环境 A/B/跨工件对照/双指标/诚实负结果/工件溯源的丰富实验协议——AR017-AR023 线标准化为最低标准）；②Gitee Segment-Push Protocol（commit-tree 部分树快进链+双端树验证——T011 推送协议提前固化，永久 commit 分歧为预期不变量）。
- T007 保持 pending：llada 基准完成 → ar019 venv 全量（期望 1022+/0/4 只增不减）+ pyflakes/mypy/black + demo 三锚点。


## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### 2026-10-15 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：13 项（G1-G9 + 证据核对 N1-N4）
- 修复的问题：Minor×3 当轮修入——G2（FR1/FR6/FR7 验收标准补显式 When 子句）、G8（T003/T004 描述显式追加 FR4 配置面）、G3（FR2/FR3 验收句补画布一致区限定词，#44 先例）
- 审查代理：sdd-gate-reviewer（ses_ec181e667ffeNIMjRynVU3kbIW）；证据核对 N1-N4 全 YES（代码事实行数/机制存在性、AR022 工件数字 7.0236×/2.9498×/gap 0.0/长度 34/66/exact 4/4 复算吻合、冻结面 UT 名命中、Out-of-Scope 衔接自洽）

### 2026-10-15 design 门控记录

- 门控结果：FAIL（第 1 轮）
- 失败项：G9（模块影响与实际不符：architecture_utils.py 实位于 actfold/models/ 非 core/；per_model 应扩展既有 llada_8b.yaml/dream_7b.yaml 而非新建）、G10（Dream logits 移位约定内部矛盾：`DreamSamplerConfig.right_shift_logits` 默认 True，design"LLaDA/Dream 不做移位"表述与现状/自身不变量冲突）
- 严重性：Important×2（Minor×7 顺带：G3 宿主回调签名/G7 替代方案对照/G12-G14 测试落点缺口/G15 Dream ctx 用例/N5 硬门口径交叉注记/G11 括号格式）
- 审查代理：sdd-gate-reviewer（ses_ec0a5b619ffeThT2lMDWdFwv3X）；N1-N5 证据核对全 YES（机制常数对拍/_forward 现状/族现状引用/配置面现状/分级门落实均属实）
- 修复：全部 9 项当轮修入 design.md（G9 两处路径/文件计划、G10 移位约定统一为"族适配层按参照默认约定处理：Dream 移位/LLaDA 无移位、Core 零感知"+UT-2304/2310 断言、G3 宿主回调与 LRU 骨架签名补全、G7 方案 B/C 权衡、G12/G14 §6.3 文档数字溯源审阅型用例+lint 三件、G13 UT-2301 补 _supports_branch_ids 对拍/UT-2309 补 Dream 入口校验/单元无掩码跳过用例、G15 UT-2311 补 Dream ctx 对象契约、N5 §4.1.8 口径交叉注记、G11 括号闭合）→ 重审第 2 轮
- 门控结果（第 2 轮）：**PASS**——F1-F10 修复定向复核全部确认（路径/文件计划与实际逐项吻合、移位约定三处一致且引用行号与源码逐条命中、N5 蕴含论证成立）；G1-G15 全维度重审全 YES；新发现 3 项 Minor（W1 Dream 批量验证移位机制落点表述缺口/W2 `_build_unit_forced` 返回值内容断言缺/W3 Core ctx LRU 淘汰行为无专门用例）当轮修入 design.md（§4.1.2 forward_fn 注记 Dream=right_shift_logits∘_forward、UT-2307/2310 forced 张量内容断言、UT-2308 LRU 淘汰用例与 `_MAX_FOLD_CTX_ENTRIES`=32 同值对拍）
- 审查代理（第 2 轮）：sdd-gate-reviewer（ses_ec08f0e43ffeAjI7ESnKS4jN1F）；代码事实交叉验证面全部通过（种子公式 :411/探测 :325-349/LRU :52/折叠分支缺 position_ids :186-202/config 两新字段缺失/per_model 两文件存在无投机段）
- 结论：design 阶段门控通过，T001-T011 可进入开发（T001 探针先决）
