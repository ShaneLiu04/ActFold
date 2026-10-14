# 实验重跑清单（AR001 RERUN CHECKLIST）

> AR001 深度优化完成后，`results/` 下全部历史产物已标注失效（见
> `results/INVALIDATED.md` 及其子目录标注）。本清单给出在换机（高算力 GPU）
> 场景下完整复现实验的步骤、环境要求、预期资源与数字修复点。
> 本清单不含任何特定平台路径或镜像硬编码；HF 端点与缓存目录一律通过
> `--hf-endpoint` / `--hf-home` CLI 参数或对应环境变量传入。

## 1. 环境要求

| 项 | 要求 |
|---|---|
| Python | 3.10+ |
| PyTorch | ≥2.1（含 CUDA；Triton 随近期 PyTorch 附带） |
| transformers | `==4.53.1`（Fast-dLLM v2 remote code 所需；LLaDA/Dream remote code 不兼容 5.x） |
| 依赖 | `pip install -r requirements.txt -r requirements-bench.txt`（评测需 bench 依赖） |
| 平台 | evalplus 代码执行需 Unix-like 平台（Linux/WSL2），原生 Windows 不可用 |
| GPU 时钟 | 时延对比类实验建议锁定时钟（`nvidia-smi -lgc <clock>` 或等价方式），避免 M-5 类状态漂移 |
| HF 访问 | `--hf-endpoint <endpoint>`（或 `HF_ENDPOINT` 环境变量）、`--hf-home <缓存目录>`（或 `HF_HOME`） |

## 2. 预期资源（显存）

| 实验 | 模型/形状 | 建议显存 |
|---|---|---|
| `algo_experiments` / `overhead_bench`（fastdllm） | Fast-dLLM-v2-1.5B bf16（28 层, H=1536） | ≥8 GiB |
| `algo_experiments` / `overhead_bench`（llada） | LLaDA-8B-Instruct bf16（32 层, H=4096） | ≥24 GiB（权重约 16 GiB + 激活/KV；40 GiB 更舒适） |
| `algo_experiments` / `overhead_bench`（dream） | Dream-7B-Instruct bf16（28 层, H=3584） | ≥24 GiB |
| `opt1`–`opt3` 基准 | 真实模型单层/单路径测量 | 与所测模型同量级（1.5B ≥8 GiB；7B/8B ≥24 GiB） |
| `opt2_shape_bench` / `opt4_fused_bench` | 合成形状（H≤8192, T≤8192, bf16） | <8 GiB |
| 消融（`run_ablation.sh --synthetic`） | 合成小模型 | CPU 即可 |
| 评测（`run_benchmarks.sh`，`max_new_tokens=256`） | lm-eval gsm8k/math/ifeval 等 | 与模型权重同量级；7B/8B ≥24 GiB |
| `demo.py` 基线对拍 | 合成小模型 | CPU 即可 |
| `ar019_real_e2e_bench`（真实块验证三路径） | Fast-dLLM-v2-1.5B bf16（28 层, H=1536，本地权重目录） | 实测峰值 5624/7640/8949 MB（三配置），16 GB 卡可跑 |

## 3. 重跑步骤

所有脚本以模块形式运行（自 T019 起不再依赖 `sys.path` hack）。

```bash
# 0) 环境准备
pip install -r requirements.txt -r requirements-bench.txt
pip install transformers==4.53.1

# 1) 真实模型相似度/稳定率/采样实验（逐模型；以 llada 为例）
python -m scripts.algo_experiments --model llada --out results/experiments/llada \
    --hf-home <你的HF缓存目录>
python -m scripts.overhead_bench --model llada --out results/experiments/llada
# dream/fastdllm 同理（--model dream / --model fastdllm）

# 2) 实验图表
python -m scripts.make_experiment_figures --root results/experiments --out results/experiments/figures

# 3) 优化基准（opt1–opt4）
python -m scripts.opt1_cache_bench --model llada --out results/optimization/opt1/llada
python -m scripts.opt2_split_bench --model llada --out results/optimization/opt2/llada
python -m scripts.opt2_long_seq --model llada --out results/optimization/opt2/longseq/llada
python -m scripts.opt2_shape_bench --out results/optimization/opt2/shape --repo <模型repo或本地路径>
python -m scripts.opt3_adaptive_bench --model llada --out results/optimization/opt3/llada
python -m scripts.opt4_fused_bench --out results/optimization/opt4

# 4) 优化图表
python -m scripts.make_optimization_figures --root results/optimization --out results/optimization/figures

# 5) 消融研究（合成模型；实测化后的产物，含逐层实测与真实驱逐曲线）
bash scripts/run_ablation.sh --synthetic

# 6) 评测（配置驱动；max_new_tokens 默认 256，结果含 baseline/actfold 时延）
bash scripts/run_benchmarks.sh actfold/configs/real_model_example.yaml

# 7) 论文图表
python -m scripts.generate_figures --results-dir results/

# 8) 基线对拍（合成基线，正确性守恒）
python demo.py

# 9) AR019 真实 checkpoint 端到端墙钟（Fast-dLLM-v2-1.5B，P1-7 第五档）
#    a) 权重获取：HF API 枚举 siblings 后逐文件下载（requests 栈在企业代理下
#       407/500 时用 curl 走代理；11 个小文件 + model.safetensors 3,087,467,144 字节）：
#       curl.exe -x "<代理>" -sL -o <本地目录>\<文件> \
#           https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B/resolve/main/<文件>
#    b) AR019 venv（主环境 transformers 5.x 不可降级时；--system-site-packages 继承 torch）：
#       python -m venv --system-site-packages <venv目录>
#       <venv目录>\Scripts\pip install "transformers==4.53.1" "huggingface_hub<1.0" einops
#    c) 基准（--model-path 必填或设 FASTDLLM_MODEL_PATH；无合成回退）：
<venv目录>\Scripts\python -m scripts.ar019_real_e2e_bench \
    --model-path <本地权重目录> --hf-home <HF缓存目录> --reps 30 --warmup 8
#    工件：results/optimization/ar019_real_e2e_bench.json（单次预承诺运行，
#    对照数字：三配置 adaptive vs baseline 2.44/2.62/2.59×，maxdiff=0.0，
#    stable ratio 精确 0.875/0.875/0.9375，容量 512→160/1024→320/1024→160）
#
# 10) AR020 发布路径 cache put 批量化 A/B（Fast-dLLM-v2-1.5B，同 venv/权重）
#    a) 前置：第 9 步的 venv 与本地权重（transformers==4.53.1，AR019 venv）
#    b) 基准（复用 ar019 助手；--quick 冒烟 = (256,4) reps=3）：
<venv目录>\Scripts\python -m scripts.ar020_publish_put_bench \
    --model-path <本地权重目录> --hf-home <HF缓存目录> --reps 30 --warmup 8
#    工件：results/optimization/ar020_publish_put_bench.json（单次预承诺运行；
#    判定式/c=2/最小配置 (256,4) 冻结于工件 meta）
#    对照数字：H1-H4 全 supported——发布窗口 p50 三配置 7.894→4.393 /
#    16.809→9.510 / 8.708→4.425 ms（1.80/1.77/1.97×），账本 116/232/116→1，
#    正确性门 116/232/116 张量全 bit-equal，e2e 最小配置严格改善
#    3.900ms > 2σ=3.712ms，no_fallback 38/38，峰值 VRAM 10027 MB
```

## 4. 方法学要求（重跑必须满足）

1. **方差与种子**：`time_forward` 每条件 repeats≥3（CUDA 事件逐次计时），
   `exp_sampling` 以 `repeats`/`seed` 参数运行并校验 `tokens_reproducible`；
   报告 mean±std，不要只报单次值。
2. **跨版本对比**：优化前后对比必须**同日同机交错**运行（M-5），并锁定 GPU
   时钟；不同时间不同状态的对比不得作为结论。
3. **准确率区间**：小样本 accuracy 报 Wilson 置信区间，不得只报点估计。
4. **消融口径**：layerwise 结果来自 `disabled_layers` 实测（对照列
   `linear_estimate_pct` 仅供说明），cache 扫描预算随 `seq_len` 缩放并保证
   出现真实驱逐（`folded_layer_count` 随预算变化）。
5. **FLOPs 口径**：使用修正后计数器——embedding 仅计 LM head；FFN 形状经
   `model_ffn_flops_kwargs`（`ffn_intermediate_dim`/`ffn_type`）按模型提取；
   成本模型经 `HardwareProfile.from_device`（必要时 `calibrate=True`）。
6. **评测口径**：`max_new_tokens=256`（ActFoldConfig 字段），结果含
   `baseline_latency_ms`/`actfold_latency_ms`，metric key 为规范名
   （gsm8k/math→`exact_match`，ifeval→`prompt_level_acc`）。
7. **对拍校验**：opt4 融合路径 `max_abs_err==0`（与 PyTorch 路径 bit-exact）；
   `demo.py` 输出应复现第 5 节的合成基线数字。

## 5. 数字修复点（旧 → 新口径）

| # | 数字 | 旧口径 | 新口径 |
|---|---|---|---|
| 1 | FLOPs 节省百分比 | embedding 双计数（输入 embedding 计 V·h·T）压低约 7pp | 仅计 LM head；合成基线 78.5% → **85.5%**；真实模型数字须整体重算 |
| 2 | 消融三张 CSV | 合成外推（layerwise 线性缩放；cache 扫描无驱逐平坦曲线） | 逐层实测 + 真实驱逐曲线（预算 < 2·seq_len 时 `folded_layer_count` 下降） |
| 3 | 相似度/跨步稳定率 | 部分步均值、无 final_norm、非 fp32 余弦 | 全步均值 + final_norm + fp32 余弦，旧值不可比 |
| 4 | 时延（overhead/opt benches） | 单次测量，无 std/种子 | repeats≥3 + 种子受控，报 mean±std；且实现路径已重写，旧开销分解失效 |
| 5 | 成本模型预测 | 无注意力二次项/KV 读，固定硬件常数 | 含 T²/KV 项 + 设备表（`from_device`，可 `calibrate`）；"预测 vs 实测"结论需重做 |
| 6 | 评测数字 | `max_new_tokens=16`，无时延，metric key 不规范 | `max_new_tokens=256` + 时延指标 + 规范 metric key |
| 7 | `demo.py` 基线 | — | FLOPs reduction ≈85.5%、MSE ≈2.35e-03、stable ratio ≈93.75%（重跑应复现） |

## 6. 验收

- 重跑产物落盘后，`results/` 各目录的 `INVALIDATED.md` 应删除（或移至归档），
  避免新旧混用。
- 每组对比实验附：运行日期、GPU 型号与时钟状态、种子列表、repeats 数。
- 与本清单命令不一致的自定义运行（不同 prompt 长度、dtype、时钟策略）须在
  产物 `meta` 中注明。
