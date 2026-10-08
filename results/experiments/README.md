> ⚠️ **本目录产物已标注失效**（见本目录 INVALIDATED.md）：AR001 修复了相似度口径、
> FLOPs 计数与统计方法学，并重写了被测实现路径。重跑步骤见 docs/RERUN_CHECKLIST.md。

# ActFold 深度实验结果索引

本目录保存 2026-10-03 在 AutoDL RTX 6000D（84 GB）上对三个真实扩散语言模型运行
`scripts/algo_experiments.py` 与 `scripts/overhead_bench.py` 的全部原始产物。

## 目录结构

```
experiments/
├── fastdllm/                 # Fast-dLLM-v2-1.5B（28 层, H=1536）
│   ├── results.json          # 9 组实验：相似度/tau/不变量/层选择/缓存/动态τ/合并/采样/成本模型
│   ├── similarity.npz        # 逐层×逐token 余弦相似度矩阵（flip0/1/8）+ prompt tokens
│   └── overhead.json         # 单层开销分解
├── llada/                    # LLaDA-8B-Instruct（32 层, H=4096）
│   └── ...（同上）
├── dream/                    # Dream-7B-Instruct（28 层, H=3584）
│   └── ...（同上）
└── figures/                  # 10 张解释性图表（与 figures/experiments/ 相同）
```

## 图表说明

| 文件 | 内容 |
|---|---|
| `fig_overhead_breakdown.png` | 单层开销分解：cache get > 原始层计算（核心发现） |
| `fig_latency_vs_prediction.png` | FLOPs 成本模型预测 vs 实测延迟（约 3800 倍差距） |
| `fig_similarity_heatmap.png` | 层×token 父子余弦相似度热力图（flip=1/8） |
| `fig_similarity_hist.png` | 相似度分布直方图（flip=1 vs 8） |
| `fig_stable_by_layer.png` | 逐层平均相似度与稳定率（τ∈{0.95,0.99,0.995}） |
| `fig_tau_quality.png` | τ 扫描：稳定率、top-1 一致率、相对 MSE |
| `fig_cache_budget.png` | 缓存预算 vs 稳定率/保真度 |
| `fig_layer_ablation.png` | all / early-only / late-only / none 层选择消融 |
| `fig_sampling.png` | 扩散采样跨步稳定率与折叠慢倍数 |
| `fig_invariants.png` | 算法不变量：自折叠/全发散精确性 + 快速路径速度 |

## 关键结论（详见 `docs/DEEP_EXPERIMENT_REPORT.md`）

- 算法正确性严格成立：自折叠与全发散 MSE=0；Fast-dLLM 折叠采样 token 匹配 1.0。
- 稳定率极高（跨步 0.98–0.99），但当前实现零墙钟收益：缓存读取开销超过单层计算，
  且"任一发散即整层重算"。
- LLaDA/Dream 对复用敏感（采样匹配 0.71–0.73），Fast-dLLM 最稳（1.0）。

## 复现

```bash
python scripts/algo_experiments.py --model fastdllm --out results/experiments/fastdllm
python scripts/algo_experiments.py --model llada   --out results/experiments/llada
DREAM_MODEL_PATH=/path/to/dream python scripts/algo_experiments.py --model dream --out results/experiments/dream
python scripts/overhead_bench.py --model <key> --out results/experiments/<key>
python scripts/make_experiment_figures.py --root results/experiments --out results/experiments/figures
```
