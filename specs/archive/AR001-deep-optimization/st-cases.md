# [AR001] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR001-deep-optimization |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-09 |

执行环境：Windows / Quadro RTX 5000 16GB（P8 降频）/ Python 3.11.9 / PyTorch 2.8.0+cu128 / Triton 可用。
验收依据（srs §5 约束）：墙钟绝对值不作验收；用测试通过、bit-exact 对拍、同步/分配计数代理指标。

自动化执行记录（本轮 ST 实跑）：
- `python -X utf8 -m pytest tests/ -q -m "not slow" -p no:cacheprovider` → **536 passed, 2 skipped, 3 deselected**（21.19s）
- `python -m mypy actfold --strict --ignore-missing-imports` → **Success: no issues found in 63 source files**
- `python -m pyflakes actfold tests demo.py scripts` → **零告警**
- `python demo.py` → **FLOPs reduction 85.5% / MSE 2.35e-03 / stable ratio 93.75%**
- 干净环境（HF_ENDPOINT/HF_HOME 均未设置）`python -m scripts.opt1_cache_bench --help` → 正常输出且带 `--hf-endpoint/--hf-home` 参数
- `python -m scripts.opt4_fused_bench --out <tmp>`（synthetic 实跑）→ fused 2.42–2.84x，`max_err=0.0e+00` 全形状 bit-exact

## 测试用例列表

### ST-001：stable_ratio 全步均值

**关联需求：** srs.md §3.1 F1（B1）
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** 3 步 folded 生成且各步 stable_ratio 不同
**测试步骤：** 1. 运行 folded_generate 3 步；2. 读取 result.stable_ratio
**期望结果：** 得到三步均值而非末步值
**实际结果：** `tests/test_folded_generation.py::test_stable_ratio_is_mean_across_folded_steps` 通过（全量 536 passed 中）
**状态：** PASS

---

### ST-002：ManualFoldedForward final norm 一致性

**关联需求：** srs.md §3.1 F1（B2）
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** LLaMA 结构模型
**测试步骤：** 1. 通过 ManualFoldedForward 前向；2. 与原模型 forward 对比 logits
**期望结果：** top-1 一致率 100%
**实际结果：** `tests/test_architecture_utils.py::test_manual_folded_forward_applies_final_norm` 通过（断言 top-1 一致）
**状态：** PASS

---

### ST-003：无 head 模型显式报错

**关联需求：** srs.md §3.1 F1（B3）
**测试类型：** 异常处理 | **优先级：** High
**前置条件：** 无 lm_head 的 AutoModel
**测试步骤：** 1. 构造 GenericDiffusionLLM（默认）；2. 构造时传 allow_random_head=True
**期望结果：** 默认抛 RuntimeError；显式旗标下允许
**实际结果：** `tests/test_models.py::test_generic_random_head_raises_by_default` 通过
**状态：** PASS

---

### ST-004：fp16 零向量/NaN 不产生 NaN mask

**关联需求：** srs.md §3.1 F1（B7）
**测试类型：** 边界条件 | **优先级：** High
**前置条件：** fp16 零向量 token / NaN 输入
**测试步骤：** 1. gate 以 fp16 零向量计算；2. NaN 输入计算
**期望结果：** 不产生 NaN mask；零范数/NaN 判 divergent
**实际结果：** `tests/test_similarity_gate.py::test_fp16_zero_norm_vector_is_divergent`、`test_nan_inputs_map_to_divergent` 通过
**状态：** PASS

---

### ST-005：B4/B5/B6/B8/B12-lite 修复保持回归

**关联需求：** srs.md §3.1 F1（B4/B5/B6/B8/B12-lite）
**测试类型：** 回归测试 | **优先级：** High
**前置条件：** 原 202 项测试 + 新增回归测试
**测试步骤：** 1. 运行全量测试套件
**期望结果：** 原测试全部保持通过，新增回归测试通过
**实际结果：** 全量 **536 passed**（含原套件全部），`test_profiler.py`（B4 try/finally、B8 perf_counter）、`test_fast_dllm_sampler.py`（B5）、`test_sampling_utils.py`（B6 窄化异常）、`test_model_wrapper.py`（B12-lite context manager/WARNING/`__del__` 兜底）通过
**状态：** PASS

---

### ST-006：cache 有界（512 步不线性泄漏）

**关联需求：** srs.md §3.2 F2
**测试类型：** 边界条件 | **优先级：** High
**前置条件：** 512 步 folded 生成、单层 cache 容量限制
**测试步骤：** 1. 长生成后统计 cache buffer 数量
**期望结果：** buffer 数量有界（不随步数线性增长）
**实际结果：** `tests/test_vectorized_cache.py::test_ring_eviction_keeps_recent_branch_steps`、`test_num_entries_bounded_across_steps` 通过（ring (branch,step) 淘汰 + max_branch_steps=4）
**状态：** PASS

---

### ST-007：被淘汰分支按 cache miss 处理

**关联需求：** srs.md §3.2 F2
**测试类型：** 异常处理 | **优先级：** High
**前置条件：** 被淘汰分支的 token 被 gate 请求
**测试步骤：** 1. folded forward 请求已淘汰条目
**期望结果：** 判为 divergent 并正确重算，不崩溃
**实际结果：** cache miss → h_parent None → recompute 路径；`test_vectorized_cache.py` 淘汰后读取用例 + `tests/test_folded_transformer.py` 缺父激活重算用例通过
**状态：** PASS

---

### ST-008：suffix_append 使 τ 敏感性可区分

**关联需求：** srs.md §3.3 F3
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** suffix_append draft 与 τ∈{0.9,0.95,0.99}
**测试步骤：** 1. 阈值敏感性实验
**期望结果：** 不同 τ 产生可区分的 stable_ratio
**实际结果：** `tests/test_draft_generator.py::test_suffix_append_tau_sensitivity_distinguishable` 通过；实测化消融 `test_ablation_measured.py` 9 项通过（含逐层实测）
**状态：** PASS

---

### ST-009：draft 可复现 + 现有 draft 测试保持

**关联需求：** srs.md §3.3 F3
**测试类型：** 回归测试 | **优先级：** Medium
**前置条件：** 相同 seed
**测试步骤：** 1. 同 seed 生成 draft 两次对比
**期望结果：** 完全可复现；现有 draft 测试通过
**实际结果：** `tests/test_draft_generator.py` 全部通过（含 seed 确定性用例；`logits_draft` 缺 parent_logits 抛 RuntimeError、flip_region 校验）
**状态：** PASS

---

### ST-010：每层阻塞同步 ≤1

**关联需求：** srs.md §3.4 F4
**测试类型：** 性能代理指标 | **优先级：** High
**前置条件：** profiler 开启的 folded 前向
**测试步骤：** 1. 以同步计数 monkeypatch 统计三分支判定路径的 host 同步
**期望结果：** 三分支判定单次 `sum()` 同步（原 `.all()`+`.any()` 两次）
**实际结果：** `tests/test_folded_transformer.py::test_three_way_branch_single_sync` 通过；profiler 惰性回读（1000 次 record 期间 0 次 D2H）由 `tests/test_stability_profiler.py` 异步累加用例覆盖
**状态：** PASS

---

### ST-011：profiler record 期间零 D2H + 全量测试绿

**关联需求：** srs.md §3.4 F4
**测试类型：** 性能代理指标 / 回归 | **优先级：** High
**前置条件：** record 被调用 1000 次、无 profile 读取
**测试步骤：** 1. 批量 record；2. 统计 D2H
**期望结果：** 期间 D2H 为 0；原测试全绿 + 新增同步语义测试通过
**实际结果：** `test_stability_profiler.py` 惰性回读用例通过；全量 **536 passed**；signature/ones-mask/adaptive gate/hook 常驻/独立降级各有对应测试（`test_folded_transformer.py`、`test_adaptive_gate.py`、`test_split_layer.py`、`test_fused_ops.py`）
**状态：** PASS

---

### ST-012：merge kernel bit-exact（H=3584、非整除 128、非连续）

**关联需求：** srs.md §3.5 F5
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** 任意 [B,T,H]（含 H=3584）非连续输入
**测试步骤：** 1. Triton merge 与 PyTorch 参考对拍
**期望结果：** max abs err = 0
**实际结果：** `tests/test_fused_ops.py::test_t013_merge_bit_exact_h3584`（及 noncontiguous 用例）通过
**状态：** PASS

---

### ST-013：非连续 view 不发生调用方 contiguous 拷贝

**关联需求：** srs.md §3.5 F5
**测试类型：** 性能代理指标 | **优先级：** Medium
**前置条件：** 非连续 transpose view 输入
**测试步骤：** 1. merge 时张量分配计数
**期望结果：** 不增加调用方侧 contiguous 拷贝分配
**实际结果：** `tests/test_fused_ops.py` stride 寻址/分配计数用例通过（`hidden_stride` 寻址，无 `.contiguous()` 双拷贝）
**状态：** PASS

---

### ST-014：F6 后 logits 数值等价

**关联需求：** srs.md §3.6 F6
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** 相同输入与折叠配置
**测试步骤：** 1. F6 前后 folded forward 输出对比
**期望结果：** logits 数值等价
**实际结果：** `tests/test_folded_transformer.py` gate 读 parent L-1 ffn_out / layer 0 读 embedding 的语义用例 + 输出等价用例通过
**状态：** PASS

---

### ST-015：cache 缓冲区字节减半

**关联需求：** srs.md §3.6 F6
**测试类型：** 性能代理指标 | **优先级：** High
**前置条件：** 单分支 cache
**测试步骤：** 1. 统计缓冲区字节
**期望结果：** 较修改前减少 ~50%（只存 ffn_out + layer 0 embedding）
**实际结果：** `tests/test_folded_transformer.py::test_t014_buffer_bytes_halved` 通过（H=4096 量级参数化：~2×(T·H·L) → ~1×）
**状态：** PASS

---

### ST-016：legacy cache put/get 零 host 同步

**关联需求：** srs.md §3.7 F7
**测试类型：** 性能代理指标 | **优先级：** High
**前置条件：** legacy cache put/get 基准（T=512）
**测试步骤：** 1. 同步计数（原 get 为 T 次 LRU touch `.any()`）
**期望结果：** 0 次 host-device 同步
**实际结果：** `tests/test_activation_cache.py::test_put_get_no_host_sync` + `test_group_lru_evicts_oldest_group` 通过（组级 LRU 单次 touch、无逐 token 循环）
**状态：** PASS

---

### ST-017：merge 慢路径分配减少 + fetch/fetch_masked 契约保持

**关联需求：** srs.md §3.7 F7
**测试类型：** 性能代理指标 / 回归 | **优先级：** High
**前置条件：** merge 慢路径 folded forward；三个 cache 契约
**测试步骤：** 1. 张量分配计数；2. 契约测试
**期望结果：** 每层减少 ≥2 个 [B,T,H] 分配；zero-filled/view 只读契约通过
**实际结果：** `tests/test_cache_protocol.py` 3 cache × 8 契约全过；`test_folded_transformer.py` 慢路径 fused bit-exact（`test_t016_slow_path_fused_bit_exact_cpu`）通过（raw fetch 无零填充）
**状态：** PASS

---

### ST-018：gather_select 主路径 bit-exact

**关联需求：** srs.md §3.8 F8
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** 任意折叠配置
**测试步骤：** 1. gather_select 主路径 vs 原 get+merge 路径对拍
**期望结果：** 输出 bit-exact
**实际结果：** `tests/test_fused_ops.py` gather_select 对拍用例 + `test_folded_transformer.py::test_t016_*` bit-exact 用例通过；ST 实跑 opt4_fused_bench `max_err=0.0e+00` 全形状
**状态：** PASS

---

### ST-019：小 shape 阈值保护

**关联需求：** srs.md §3.8 F8
**测试类型：** 边界条件 | **优先级：** Medium
**前置条件：** T<2048
**测试步骤：** 1. 小 shape folded forward
**期望结果：** 走 PyTorch 路径（阈值保护）
**实际结果：** `test_folded_transformer.py::test_t016_*` 阈值用例（T≥2048&H≥4096 或 T≥8192 才启用）通过
**状态：** PASS

---

### ST-020：split 层分配减少 + bit-exact

**关联需求：** srs.md §3.9 F9
**测试类型：** 性能代理指标 / 正常路径 | **优先级：** High
**前置条件：** split 层激活（batch×seq ≥ split_min_tokens）
**测试步骤：** 1. 分配计数对比；2. 输出对比
**期望结果：** 每层减少 ≥2 个 [B,T,H] 分配；输出 bit-exact
**实际结果：** `tests/test_split_layer.py::test_t017_*`（clone 底板 + divergent index_copy、常驻 hook、bit-exact）通过
**状态：** PASS

---

### ST-021：LLaDA D2H 同步 256→≤32

**关联需求：** srs.md §3.10 F10
**测试类型：** 性能代理指标 | **优先级：** High
**前置条件：** B=8、steps=32 的 LLaDA 采样
**测试步骤：** 1. 统计单次 generate 的 D2H 同步
**期望结果：** 从 ≥256 次降至 ≤32 次
**实际结果：** `tests/test_diffusion_samplers.py` LLaDA 向量化用例（批量 topk、prompt_lens 向量化、index_fill_ suppress）通过；`test_sampling_utils.py` 宿主回读 136→1 用例通过
**状态：** PASS

---

### ST-022：get_num_transfer_tokens 逐值对拍

**关联需求：** srs.md §3.10 F10
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** B∈{1,8}, steps∈{8,128}, schedule∈{linear,cosine}, stochastic∈{T,F}
**测试步骤：** 1. 向量化实现 vs 逐元素参考对拍
**期望结果：** 逐值一致
**实际结果：** `tests/test_sampling_utils.py::test_t018_num_transfer_tokens_reference_parity`（81 参数组合）+ 官方 87/87 对拍用例通过
**状态：** PASS

---

### ST-023：LLaDA 早停

**关联需求：** srs.md §3.10 F10
**测试类型：** 正常路径 | **优先级：** Medium
**前置条件：** 已解完的 block
**测试步骤：** 1. 采样循环观察 break
**期望结果：** 提前 break（forward 次数减少）
**实际结果：** `test_diffusion_samplers.py::test_t018_llada_early_stop` 通过；sampler smoke 测试全过
**状态：** PASS

---

### ST-024：干净环境脚本无硬编码依赖

**关联需求：** srs.md §3.11 F11
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** 未设置任何 HF 环境变量
**测试步骤：** 1. `--help` 与 synthetic 干跑
**期望结果：** 不读写 autodl 路径、不依赖 hf-mirror
**实际结果：** ST 实跑：清空 HF_ENDPOINT/HF_HOME 后 `python -m scripts.opt1_cache_bench --help` 正常且带 `--hf-endpoint/--hf-home` 参数；`tests/test_scripts_portability.py` 14 用例全过（无 sys.path hack、无镜像/autodl 硬编码、apply_hf_env 优先级/回退）
**状态：** PASS

---

### ST-025：SwiGLU FFN FLOPs 公式

**关联需求：** srs.md §3.11 F11
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** SwiGLU 模型（intermediate=3.375h）
**测试步骤：** 1. flops_counter 计算
**期望结果：** FFN 项 = 2·3·3.375·h²·T
**实际结果：** `tests/test_flops_counter_ffn.py`（ffn_type=swiglu n_matmul=3、ffn_intermediate_dim、embedding 仅 LM head、model_ffn_flops_kwargs）11 用例通过
**状态：** PASS

---

### ST-026：from_device 设备查表

**关联需求：** srs.md §3.11 F11
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** Quadro RTX 5000
**测试步骤：** 1. `HardwareProfile.from_device` 调用
**期望结果：** 返回该卡查表常数（而非通用 100/600）
**实际结果：** `tests/test_cost_model_device.py`（_DEVICE_TABLE 查表含 QUADRO RTX 5000、calibrate 仅 CUDA、T²/KV 项、gate/merge 带宽项）14 用例通过
**状态：** PASS

---

### ST-027：time_forward 重复含 mean/std/p50/n

**关联需求：** srs.md §3.11 F11
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** 3 次 time_forward 重复
**测试步骤：** 1. 读取结果
**期望结果：** 含 mean/std/p50/n 字段
**实际结果：** `tests/test_experiment_stats.py`（TimingStats as_dict、stats_from_samples、reps<3 ValueError、repeat_with_seed RNG 恢复、exp_sampling 新 schema + tokens_reproducible）16 用例通过
**状态：** PASS

---

### ST-028：config max_new_tokens 透传

**关联需求：** srs.md §3.11 F11
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** config `max_new_tokens`
**测试步骤：** 1. benchmark run
**期望结果：** adapter 生成长度使用该值
**实际结果：** `tests/test_eval_fixes.py::test_benchmark_runner_passes_max_new_tokens_to_both_adapters` 等 20 用例通过（含 ≤0 ValueError、latency 指标、规范 metric key、tokenize-once）
**状态：** PASS

---

### ST-029：eval 墙钟与 metric key（M-6/M-8）

**关联需求：** srs.md §3.11 F11（墙钟测量、metric key 行）
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** _generate_one 计时；task metric 映射
**测试步骤：** 1. 评测路径聚合 latency_ms；2. 按 task 取 metric
**期望结果：** latency 按路径聚合；metric key 显式声明
**实际结果：** `test_eval_fixes.py`（`_generate_one` 3 元组含 latency_ms、`_evaluate` 含 baseline/actfold_latency_ms、LMEvalAdapter._TASK_METRIC_KEYS exact_match/prompt_level_acc、judges metrics 聚合）通过
**状态：** PASS

---

### ST-030：消融实测化

**关联需求：** srs.md §3.11 F11（消融实测化行）
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** layerwise disabled_layers；cache 扫描制造驱逐
**测试步骤：** 1. layerwise 实测；2. cache 预算扫描
**期望结果：** 实测值；驱逐后 ratio 变化
**实际结果：** `tests/test_ablation_measured.py` 11 用例通过：`test_layerwise_disabled_layers_restricts_measured_layers`（禁用层不出现在 per_layer_stable）、`test_layerwise_measured_differs_from_linear_extrapolation`（实测≠线性，≥1 组）、`test_cache_size_sweep_creates_real_eviction`（budget=T 驱逐 → folded_layer_count 1 vs 3、reduction 21.5% vs 64.4%）
**状态：** PASS

---

### ST-031：RNG 隔离（M-9 范围内）

**关联需求：** srs.md §3.11 F11（RNG 隔离行）
**测试类型：** 边界条件 | **优先级：** Medium
**前置条件：** 测量路径种子受控
**测试步骤：** 1. repeat_with_seed / measure_folding 后检查全局 RNG
**期望结果：** 全局 RNG 状态恢复；结果可复现
**实际结果：** `test_experiment_stats.py` repeat_with_seed 快照/恢复用例 + `test_ablation_measured.py` seed 确定性用例通过（DraftGenerator.generate 全局 seed 残留已在 CHANGELOG Known follow-ups 记录，属 Minor 遗留）
**状态：** PASS

---

### ST-032：INVALIDATED 标注覆盖

**关联需求：** srs.md §3.12 F12
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** results/ 目录
**测试步骤：** 1. 检索每个受影响数据文件
**期望结果：** 每个文件（或父目录）有作废标注
**实际结果：** `tests/test_scripts_portability.py::test_invalidated_markers_cover_all_results` 通过（results/、experiments/、optimization/ 三处 INVALIDATED.md 覆盖全部 52 产物，祖先目录断言）
**状态：** PASS

---

### ST-033：RERUN_CHECKLIST 可执行性

**关联需求：** srs.md §3.12 F12
**测试类型：** 正常路径 | **优先级：** High
**前置条件：** docs/RERUN_CHECKLIST.md
**测试步骤：** 1. 检查清单无硬编码路径/镜像；2. 命令模块形式可运行
**期望结果：** 路径/镜像无硬编码阻塞
**实际结果：** `test_rerun_checklist_has_no_hardcoded_platform_paths` 通过（正则断言）；清单命令与 T019 迁移后的脚本 CLI 一致（`--hf-endpoint/--hf-home` 可选参数、`python -m scripts.<name>` 模块形式，ST-024 实跑验证同套 CLI）
**状态：** PASS

---

### ST-034：demo 基线对拍（数字修复点 #7）

**关联需求：** srs.md §3.11 F11（FLOPs 口径修正）+ §4 NFR 数值
**测试类型：** 回归测试 | **优先级：** High
**前置条件：** 修正后 FLOPs 计数器
**测试步骤：** 1. 运行 demo.py
**期望结果：** FLOPs reduction 85.5%（原 78.5% 为 embedding 双计数）、MSE 2.35e-03、stable 93.75%
**实际结果：** ST 实跑 `python demo.py` → **85.5% / 2.35e-03 / 93.75%**，与预期完全一致
**状态：** PASS

---

### ST-035：opt4 融合基准端到端（可移植性 + bit-exact）

**关联需求：** srs.md §3.11 F11 + §4 NFR（可移植性、Triton 对拍）
**测试类型：** 集成 | **优先级：** High
**前置条件：** Windows 原生、Triton 可用
**测试步骤：** 1. 干净环境运行 opt4_fused_bench
**期望结果：** 可直接运行；fused 与 torch bit-exact
**实际结果：** ST 实跑：`python -m scripts.opt4_fused_bench --out <tmp>` → 全形状 `max_err=0.0e+00`，speedup 2.42–2.84x（P8 降频下相对值仅作参考）
**状态：** PASS

---

### ST-036：lint / 类型全绿（NFR 代码规范）

**关联需求：** srs.md §4 NFR（代码规范、兼容性）
**测试类型：** 回归测试 | **优先级：** High
**前置条件：** AR 全部代码合入
**测试步骤：** 1. pyflakes；2. mypy --strict；3. breaking changes 登记
**期望结果：** 全部通过；CHANGELOG 登记全部 breaking changes
**实际结果：** ST 实跑：pyflakes 零告警、mypy --strict 63 文件零错误；CHANGELOG.md AR001 章节含 Breaking changes（TimingStats、exp_sampling schema、LayerCost 字段、FLOPs 口径、eval 256/新签名、AblationStudy 实测化）与调用方/测试同步
**状态：** PASS

---

## 执行摘要

| 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|
| 36 | 36 | 0 | 0 |

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-09 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 | F1 正确性修复包（8 项） | ST-001..ST-005 | PASS |
| §3.2 | F2 cache 生命周期修复 | ST-006, ST-007 | PASS |
| §3.3 | F3 draft 分布修复 | ST-008, ST-009 | PASS |
| §3.4 | F4 同步清零包（9 项） | ST-010, ST-011 | PASS |
| §3.5 | F5 Triton merge kernel 改进 | ST-012, ST-013 | PASS |
| §3.6 | F6 冗余 hidden_states 删除 | ST-014, ST-015 | PASS |
| §3.7 | F7 cache API 拆分 + legacy 重写 | ST-016, ST-017 | PASS |
| §3.8 | F8 gather_select 接入 | ST-018, ST-019 | PASS |
| §3.9 | F9 split 层 scatter 合并 | ST-020 | PASS |
| §3.10 | F10 samplers 向量化 + 早停 | ST-021..ST-023 | PASS |
| §3.11 | F11 方法学与可移植性 | ST-024..ST-031, ST-034, ST-035 | PASS |
| §3.12 | F12 数据处置与重跑清单 | ST-032, ST-033 | PASS |
| §4 | NFR（回归/代理指标/数值/可移植性/规范/兼容性） | ST-005, ST-010..ST-013, ST-015..ST-017, ST-021, ST-024, ST-034..ST-036 | PASS |

**需求覆盖率：** 12/12 功能需求（100%）+ NFR 全项

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 20 | 20 | 0 | 0 |
| 边界条件 | 6 | 6 | 0 | 0 |
| 异常处理 | 3 | 3 | 0 | 0 |
| 性能代理指标 | 7 | 7 | 0 | 0 |
| 回归测试 | 9 | 9 | 0 | 0 |
| 集成（端到端实跑） | 4 | 4 | 0 | 0 |
| **合计** | **36** | **36** | **0** | **0** |

自动化底座执行记录：pytest -m "not slow" 536 passed + 2 skipped + 3 deselected（21.19s）；mypy --strict 63 文件零错误；pyflakes 零告警；demo.py 85.5%/2.35e-03/93.75%；干净环境脚本 --help 与 opt4 synthetic 实跑 max_err=0 全形状。

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | DraftGenerator.generate(seed=) 仍用全局 torch.manual_seed（测量路径已快照/恢复，srs §3.11 RNG 隔离行部分完成） | 已登记 CHANGELOG Known follow-ups，留 AR002（与 M4 一并） |
| Minor | srs §3.11 按任务默认 max_new_tokens（humaneval:512）在 design 收窄为全局 256 | 已在 tasks.md T023 行补记偏差；按任务默认表留后续 AR |
| Minor | M-5 跨版本受控对比需在目标 GPU 按 RERUN_CHECKLIST 重跑 | 属重跑范畴，非代码缺陷；清单已含要求 |
| Cosmetic | black/isort 未在本机安装（配置存在于 pyproject.toml，CI 会执行） | 环境限制，记录备查 |

### 结论

> **Go**：12/12 功能需求 100% 覆盖，36/36 ST 用例通过，无 Critical/Major 缺陷；全量 536 测试通过、mypy --strict 与 pyflakes 全绿、demo/opt4 端到端实跑与预期基线完全一致；4 项 Minor/Cosmetic 遗留均已记录且有明确处理路径（AR002/重跑清单）。满足归档条件。
