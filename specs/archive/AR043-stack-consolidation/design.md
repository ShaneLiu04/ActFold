# [AR043] 详细设计 — 组合收敛与预设固化

| 字段 | 内容 |
|------|------|
| AR 编号 | AR043 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-25 |
| 状态 | Delivered |

## 1. 组合兼容性矩阵（来自代码硬约束）

| 组合 | 结果 |
|---|---|
| `history_cache` + `num_block_candidates>0` | ✅ 组合臂（AR034 起） |
| `history_cache` + `history_small_block_cache` | ✅（AR033 面，无投机） |
| `history_cache` + N>0 + `history_small_block_cache` | ❌ `ValueError`（v1 未定义） |
| `history_cache` + `uncertainty_validation_order` | ❌ `ValueError`（AR031 仅折叠面） |
| `commit_draft_fusion` | 需 `history_cache` + N>0 |
| `verification_free_commit` | 需 `history_cache` + N>0 + `commit_draft_fusion` |
| `adaptive_candidate_budget` / `candidate_budget_escalate` / `remask_fallback` / `commit_chunk_size` | 需 `history_cache` + N>0 |
| `speculative_commit_gate` / ACB / escalate / VFG | 与门控互斥（`ValueError`） |

## 2. 五臂实测（160 token，3 prompts；干净运行 = 决策依据）

| 臂 | p1 ms / logp | p2 | p3 | 中位加速 | logp 平价 |
|---|---|---|---|---|---|
| baseline | 198.0 / −1.1528 | 123.7 / −0.8482 | 150.5 / −0.8086 | 1.00× | — |
| **kv-only** | 122.8 / −1.1528 | 87.2 / −0.8482 | 99.5 / −0.8086 | **1.5129×** | ✅ 3/3 |
| spec (D) | 12.4 / −1.3118 | 11.8 / −0.9904 | 11.7 / −1.5370 | 12.82× | ❌ 0/3 |
| quality（AR041 回退档，旧版） | 126.7 / −1.1528 | 91.1 / −0.8482 | 103.5 / −0.8086 | 1.4542× | ✅ 3/3 |
| **latency** | 8.3 / −0.7963 | 9.6 / −1.3032 | 10.5 / −1.8494 | **14.28×** | ❌ 0/3 |

复跑（宿主争用：baseline p2=431 s、quality p3=807 s 离群）仅作稳健性记录，logp 与上表逐位一致；
**不用噪声下结论**。

## 3. 回退决策

| 创新 | 进入最佳组合 | 依据 |
|---|---|---|
| AR033 历史 KV | ✅ quality + latency | +51% 且 logp 逐位不变 |
| AR034/036/037/039 | ✅ latency | 14.28×（D 12.82× → 叠加 +11%） |
| AR041 `remask_fallback` | ❌ 从 quality 剔除 | 1.4542× vs kv-only 1.5129×，logp 相同 → 纯开销 |
| AR035 / AR038 / AR042 | ❌ | 成本 9.58× / 零收益+46% / 质量混沌 |
| AR032 `speculative_advance_reuse` | ❌ 不显式开启 | 组合臂提交前向已产出 advance |
| AR031 `uncertainty_validation_order` | ❌ | 与 history_cache 互斥 |

## 4. 预设（`actfold/models/sampler_presets.py`）

- `baseline`：官方配方（无 KV、无投机）
- `quality`：`history_cache=True`（官方重掩码 + 跨块 KV）→ 1.51×，logp 逐位等于 baseline
- `latency`：`history_cache` + N=4 + greedy + CDF + ACB(0.5) + VFG → 14–20×，质量按 workload

`PRESET_REJECTED_KNOBS` 记录剔除原因；`fastdllm_preset` 未知名 → `KeyError`，非法覆盖 → `ValueError`。

## 5. 验证

UT-4301~4306（预设可构造/形状/剔除开关全关/未知名与非法覆盖/原因非空）；
`scripts/ar043_stack_consolidation_bench.py` 冻结四项判定（最佳质量臂、最佳延迟臂、叠加增益、预设复现）
→ `overall=supported`。

## 6. 复现

```powershell
$env:FASTDLLM_MODEL_PATH="<checkpoint>"
<venv>\Scripts\python.exe -m scripts.ar043_stack_consolidation_bench --quick
```
