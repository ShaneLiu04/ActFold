# [AR044] 详细设计 — 填后精修（RaF）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR044 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-25 |
| 状态 | Delivered（机制 + UT 完成；**对照测量被外部 GPU 负载阻塞**） |

## 1. 机制

| 字段 | 默认 | 校验 |
|---|---|---|
| `commit_refine_passes` | `0` | 需 `history_cache` + N>0；负值 `ValueError` |
| `commit_refine_threshold` | `0.5` | — |

流程（在候选画布/验证之前返回，与 AR042 同一插入点）：

1. 一次性贪心填满该块仍掩码位（AR022 语义，行 0）；
2. 重复 `passes` 次：写入当前画布 → **前向整块**（历史 KV 前缀，块内双向注意力）
   → 取最初被掩码位置上的 token 概率 → `p < threshold` 的位置改填为该轮 argmax；
3. 提交最终画布，`spec_refine_forwards += passes`。

与 AR042 的本质差别：AR042 每轮只填「最高置信的 chunk」且**不回看已写位**；RaF 每轮**复核全部已写位**
并允许改写，对应官方迭代解掩的核心行为。精修块不设 CDF advance 交接（提交走非融合路径）。

## 2. 账本

`spec_refine_forwards`（默认 0）；每块精确 +`passes`，与块数线性可核。

## 3. 测量状态（诚实记录）

- 计划：3 prompts × 160 token × passes∈{1,2,3}，对照 `quality`（kv-only）与 `latency`。
- 实际：两次运行均被**外部进程**占满 GPU（`nvidia-smi` 99%、他人 `kev.serve` / `jev matrix`），
  探针在 30–60 min 内连首个 cell 都无法完成；已清理本仓库残留进程（GPU 回落到 82% / 903 MiB，
  仍被他人占用）。**未获得任何可用对照数字，不编造。**
- 因此 H1/H2 未判定；**不新增 `balanced` 预设**（无证据不得固化），AR043 的两点结论保持不变。

## 4. 服务器复现命令

```powershell
$env:FASTDLLM_MODEL_PATH="<checkpoint root>\Fast_dLLM_v2_1.5B"
# 先确认 GPU 静默（util<=10%, used<=1024MiB），再跑：
<venv>\Scripts\python.exe -m scripts.ar044_refine_probe --num-tokens 160 --prompts 3 --passes 1,2,3
```

判读：把 `R1/R2/R3` 的 `logp` 与 `kv` 臂比较（容差 0.05）、`ms` 与 `latency` 臂比较；
若某 `passes` 同时满足「logp ≥ kv − 0.05（3/3）」且「中位加速 ≥ 5×」，再新增 `balanced` 预设并回填文档。

## 5. 测试

| ID | 面 |
|---|---|
| 4401 | 默认 0：`spec_refine_forwards == 0` |
| 4402 | `passes=2`：账本为 2 的倍数、mask-free、degraded 0 |
| 4403 | 确定性（双跑逐位一致） |
| 4404 | 非法值（负值、无 history/N）→ `ValueError` |
