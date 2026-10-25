# [AR040] 详细设计 — 矩阵基准

| 字段 | 内容 |
|------|------|
| AR 编号 | AR040 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-25 |
| 状态 | Delivered（overall refuted，结论有价值） |

## 1. 矩阵设计

| 维度 | quick | formal |
|---|---|---|
| prompts | 前 3 条真实长文 prompt | 全部 6 条 |
| lengths | 160 | 160 / 320 |
| arms | A / D / S / V | 同 |
| seed | 2026（temperature 0，确定性） | 同 |
| EOS | `stop_token_id=-1` | 同 |

每 cell：四臂各跑一次（round-robin），V 臂再复跑一次做逐位复现；每次运行后用**同模型 posthoc 前向**
计算生成区间的 mean log p（AR037 已确立的质量代理，禁止用「与基线一致率」）。

## 2. 冻结判定（先于取数）

- **H1**：`logp(V) >= logp(A) - 0.05` 的 cell 比例 ≥ 90%
- **H2**：`median(ms(A)/ms(V)) >= 10` 且 `ms(V) < ms(S) < ms(D)` 的 cell 比例 ≥ 80%
- **H3**：每个 V cell 满足 `free_verify_blocks == spec_fused_forwards`、`verify <= 块数`、`degraded == 0`
- **H4**：V 臂逐 cell 复跑逐位一致；不通过 → overall = blocked

## 3. 实测（quick：3 prompts × 160 token）

| cell | A (s) | D (s) | S (s) | V (s) | logp A | logp D=S | logp V | V 免验证/再武装 |
|---|---|---|---|---|---|---|---|---|
| p1 (diffusion 技术长文) | 190.2 | 12.04 | 11.98 | **7.76** | −1.1528 | −1.3118 | **−0.7963** | 3 / 2 |
| p2 (向量数据库) | 125.3 | 11.63 | 11.73 | **8.56** | **−0.8482** | −0.9904 | −1.3032 | 2 / 3 |
| p3 (编译器) | 150.1 | 12.02 | 12.59 | **9.28** | **−0.8086** | −1.5370 | −1.8494 | 2 / 3 |

判定：`h1_pass=False (1/3)`、`h2_pass=False (序关系 1/3，中位加速 16.17× 达标)`、`h3=True`、`h4=True`
→ **overall = refuted**。

## 4. 判读（本 AR 的核心产出）

1. **速度是鲁棒的**：中位 16.2×，各 cell 16.1–24.5×，无一例外。
2. **质量不鲁棒**：p2/p3 上组合臂（含带验证的 D/S）明显劣于基线；单 prompt 的「logp 不降」是
   workload 特例，不能推广。
3. **VFG 放大分歧**：跳过验证会把「贪心行」提交，即便随机候选行本会胜出；p1 上 V 反而最好
   （−0.796 vs −1.312），p2/p3 上最差 → VFG 是**方差换取速度**，不是无条件改进。
4. D 与 S 的 logp 恒等（ACB 只跳过验证、不改变提交内容），符合设计。

## 5. 结论与建议

- 本仓库的速度结论（对非缓存基线 ~16–19×，长几何）**成立且鲁棒**；
- 质量结论必须**限定 workload**：默认推荐 **D 或 S（保留验证）**；
  `verification_free_commit` 仅在「可接受似然损失、追求极限延迟」时启用，并必须用本矩阵复验；
- 后续若要恢复质量鲁棒性，应从**提交语义**入手（自校正重掩码 / 对低置信块回退到阈值重掩码路径），
  而不是调整候选数或阈值。

## 6. 复现

```powershell
$env:FASTDLLM_MODEL_PATH="<checkpoint>"
<venv>\Scripts\python.exe -m scripts.ar040_matrix_bench --quick          # 3×160, ~8 min
<venv>\Scripts\python.exe -m scripts.ar040_matrix_bench                  # 6×{160,320}, ~2 h
```

工件：`results/optimization/ar040_matrix_bench_quick.json`（已入库）/ `ar040_matrix_bench.json`（服务器）。
`--quick` 永不写正式路径；`--force` 才允许覆盖。
