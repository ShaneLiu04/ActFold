# [AR005] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR005 |
| AR 主题 | FFN/MoE FLOPs 几何修正收口（P3-4）：真实 config 自动提取 + MoE 计量 |
| 关联 SR | 无（源自 `docs/OPTIMIZATION_GUIDE.md` P3-4 + M-4 关联，AR001 遗留半成品） |
| 日期 | 2026-10-10 |
| 状态 | Draft |

## 1. 背景与目标

AR001 给 `count_diffusion_llm_flops` 加了 `ffn_intermediate_dim`/`ffn_type` 参数与
`model_ffn_flops_kwargs` 提取 helper，但**全链没有任何模型/适配器暴露这些属性**
（grep 实证：`self.ffn_intermediate_dim` 零命中）——helper 恒回退
`4 * hidden_dim` + `"mlp"` 默认值。后果：

- 指南 P3-4 自认的 **10–25% FFN FLOPs 系统偏差仍然存在于所有上报的 TFLOPs
  数字**（LLaDA/Dream 均为 SwiGLU 3 矩阵，实际中间维 ≠ 4h）；
- **MoE 完全未覆盖**（`ffn_type` 仅接受 `"mlp"`/`"swiglu"`；HF MoE config 属性
  如 `num_experts`/`num_experts_per_tok`/`moe_intermediate_size` 无处消费）；
- "从 `config.intermediate_size` 自动读取"（AR001 design 承诺）从未接线。

本 AR 目标：让 FFN 几何从真实 HF config **自动**流到全部 3 个
`model_ffn_flops_kwargs` 调用点（verification_engine / ablation_study /
base_adapter），补齐 MoE 计量（routed top-k ± shared expert），并保持无属性
路径的逐位零回归。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `actfold/models/base.py`：`DiffusionLLM` 新增具体（非抽象）FFN 几何属性，
  从 `self.config` 鸭子类型读取（`intermediate_size`、`hidden_act` SwiGLU 族、
  MoE 属性族），缺失回退 None/默认
- `actfold/utils/flops_counter.py`：MoE 计量参数 + 校验；
  `model_ffn_flops_kwargs` 解析 adapter `underlying_model` 链
- `actfold/speculative/fast_dllm_adapter.py`：FFN/MoE 几何转发（属性或
  property，走 `underlying_model`）
- 合成 HF-config-like fixture 测试（不下载 checkpoint）
- 文档（AGENTS #33 修订 / CHANGELOG / 指南 P3-4 ✅ + 回链）

**Out of Scope（本 AR 不做的）：**
- 真实 MoE checkpoint 的实测验证（无本地 MoE 权重；交付公式 + config 提取 +
  合成实证；RERUN_CHECKLIST 目标机流程）
- attention T² 项默认开启（既有 `include_attention_t2` 开关不变）
- KV cache 访存/cost_model 带宽项（M-4 独立条目）
- router 门控 FLOPs（数量级 ~h·E，可忽略；文档注记）

## 3. 功能需求

### 3.1 FFN 几何自动提取（dense 路径）

**描述：** 真实 checkpoint 的 FFN 中间维与拓扑从 HF config 自动流入 FLOPs
估计，替换恒回退的 4h-MLP 默认。

**触发条件：** `model_ffn_flops_kwargs(adapter)` 被调用（engine/ablation/
base_adapter 三调用点），且 underlying model 暴露 HF config。

**期望行为：**
- `DiffusionLLM` 新增具体属性：`ffn_intermediate_dim`（读
  `config.intermediate_size`，缺失 → None）、`ffn_type`（`config.hidden_act`
  ∈ SwiGLU 族 {`silu`, `swish`, `swiglu` 等 HF 惯用值及带 `-swiglu` 后缀变体}
  → `"swiglu"`，否则 `"mlp"`）；
- `model_ffn_flops_kwargs` 经 adapter 的 `underlying_model` 属性解析到被包装
  模型再取属性（adapter 自身或裸 nn.Module 无属性 → 现行默认，零变化）；
- SwiGLU 判定注意：HF `hidden_act="silu"` 且 FFN 结构为 gate/up/down 的模型
  （LLaDA/Dream 用的 LLaMA 系块）即 3 矩阵拓扑。

**异常处理：** 属性缺失 → 回退默认（不抛错）；config 值非法（≤0）→ 沿
`count_diffusion_llm_flops` 既有 ValueError。

**验收标准：**
- Given 带 `intermediate_size=I` 与 SwiGLU `hidden_act` 的 config-like stub，
  When 经 adapter 调 `model_ffn_flops_kwargs`，Then 返回
  `{ffn_intermediate_dim: I, ffn_type: "swiglu"}`（含 MoE 键，见 3.2）；
- Given 无 config 的裸 nn.Module（合成模型），Then 返回现行默认（4h/None +
  `"mlp"`），全链逐位零回归。

### 3.2 MoE FLOPs 计量

**描述：** `count_diffusion_llm_flops` 支持 MoE 层的 per-token expert FLOPs
（routed top-k ± shared expert）。

**触发条件：** 传入 MoE 几何参数（或 config 暴露 MoE 属性）。

**期望行为：**
- 新参数：`moe_num_experts`、`moe_top_k`、`moe_intermediate_dim`（专家中间
  维；缺省回退 `ffn_intermediate_dim` 或 4h）、`moe_shared_expert`（bool，
  有 shared expert 时 +1 专家等价 FLOPs）；
- per-token FFN FLOPs（MoE 层）= `2 * n_matmul * moe_inter * h *
  (top_k + shared)`（n_matmul 由专家激活拓扑决定，沿用 `ffn_type`；
  reuse_ratio 语义同 dense——复用 token 不算）；
- `DiffusionLLM` 属性从 config 读取：专家数（`num_experts` /
  `n_routed_experts`）、top-k（`num_experts_per_tok` / `num_selected_experts`）、
  专家中间维（`moe_intermediate_size` / `expert_intermediate_size`）、shared
  （`shared_expert_intermediate_size` 存在且 >0）；
- `model_ffn_flops_kwargs` 返回全部 MoE 键（无 MoE → None/False 默认）。

**异常处理：** `moe_top_k <= 0` 或 `> moe_num_experts`、`moe_num_experts <= 0`
（给了 top_k 时）、`moe_intermediate_dim <= 0` → ValueError（复用既有校验
风格）。

**验收标准：**
- Given `moe_num_experts=8, moe_top_k=2, moe_intermediate_dim=I,
  ffn_type="swiglu"`，When 计算 FFN FLOPs，Then 等于手算
  `2*3*I*h*(2+0)*L*T_eff`（+shared 时系数 3）；
- Given MoE 参数缺失，Then 与现行 dense 公式逐位一致（零回归）；
- Given `moe_top_k=0` / `moe_top_k > moe_num_experts`，Then ValueError。

### 3.3 调用点自动供参（端到端接线）

**描述：** 三个 `model_ffn_flops_kwargs` 调用点（verification_engine:307、
ablation_study:144、base_adapter:233/254）无需修改签名即获得真实几何。

**触发条件：** adapter 包装的模型暴露几何（真实 checkpoint 或 config-like
stub）；合成模型继续走默认。

**期望行为：**
- adapter → underlying model → config 属性链打通；
- engine `_estimate_tflops` / ablation `measure_folding` / base_adapter
  TFLOPs 上报的数字反映真实 FFN 几何。

**验收标准：**
- Given config-like stub（SwiGLU + MoE）经 `FastDLLMAdapter` 进
  `ActFoldVerificationEngine`，When `verify_branch` 产 TFLOPs，Then 数值
  等于用真实几何手算的 `count_diffusion_llm_flops`（而非 4h 默认）；
- Given 合成模型（TinyModel 系），Then TFLOPs 与基线逐位一致（703 全量零
  回归）。

### 3.4 文档与偏差说明收口

**描述：** P3-4 勾选 + 偏差口径修订。

**期望行为：**
- AGENTS #33 修订（几何自动提取路径 + MoE 公式 + "历史数字基于 4h-MLP 默认"
  注记）；CHANGELOG AR005 节；指南 P3-4 ✅ + 回链（第十二部分）；
  README/experiment 文档中 TFLOPs 数字口径检查（如有硬编码偏差声明则更新）。

**验收标准：**
- Given 文档评审，Then 条目齐全且与实现一致（grep 可验证）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 正确性 | 零回归 | 全量 703 passed 基线不回归；demo 基线 85.5% / 2.35e-03 / 93.75% 不变（demo 用合成模型走默认几何） |
| 兼容性 | API 向后兼容 | `count_diffusion_llm_flops` 新参数全带默认值；`model_ffn_flops_kwargs` 返回字典键只增不改（新 MoE 键默认 None/False） |
| 可移植性 | 无 checkpoint 依赖 | 全部行为用 config-like stub 可验证（不下载权重） |
| 质量门 | 静态检查 | mypy（AGENTS 标准命令）零错误；pyflakes clean；100 列；Google docstring |

## 5. 约束与假设

**约束：**
- HF config 属性名按主流 MoE 家族取并集（Qwen2/DeepSeek 系），
  **不做家族分发**（鸭子类型 getattr 并集，未知家族 → dense 回退，安全侧）；
- SwiGLU 判定依据 `hidden_act` 字符串族 + 已知 LLaMA 系块结构，无则
  `"mlp"`（保守：2 矩阵，低估不高于现状）；
- AGENTS #10 适用：不引入假 config 值；属性读取失败一律显式回退默认，
  不静默编造。

**假设：**
- HF config 的 `intermediate_size` 即 FFN 中间维（LLaDA/Dream/Fast-dLLM
  均如此）；
- router 门控 FLOPs（`h * num_experts` 量级）相对 expert FFN 可忽略
  （<1%，文档注记）；
- `FastDLLMAdapter.underlying_model` 是唯一包装路径（AblationStudy 另有
  `underlying_model` 契约，AGENTS #35）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| FFN 几何 | 中间维 `I` + 拓扑（mlp 2 矩阵 / swiglu 3 矩阵） |
| MoE 计量 | per-token expert FLOPs = top-k routed ± 1 shared expert 的 FFN FLOPs（复用 token 不计） |
| config-like stub | 暴露 HF config 惯用属性的合成对象（测试用，不加载权重） |
| SwiGLU 族 | HF `hidden_act` ∈ {silu, swish, swiglu, *-swiglu 变体} 且块结构为 gate/up/down |
