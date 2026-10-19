# 实验重跑清单（AR001 RERUN CHECKLIST）

> AR001 深度优化完成后，`results/` 下全部历史产物已标注失效（见
> `results/INVALIDATED.md` 及其子目录标注）。本清单给出在换机（高算力 GPU）
> 场景下完整复现实验的步骤、环境要求、预期资源与数字修复点。
> 本清单不含任何特定平台路径或镜像硬编码；HF 端点与缓存目录一律通过
> `--hf-endpoint` / `--hf-home` CLI 参数或对应环境变量传入。

## 0. 服务器一键快速开始（复制即用）

```bash
# 1) 取代码（gitee 与本地主仓内容一致；commit id 天然不同属正常）
git clone https://gitee.com/liu-xingyan04/act-fold.git && cd act-fold

# 2) 基础环境（CPU 可跑 demo 与整仓测试）
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt

# 3) 30 秒自检：期望 85.5% FLOPs 降幅 / MSE 2.35e-03 / 稳定率 93.75%
python demo.py

# 4) 整仓质量门：期望 1179 passed / 4 skipped / 3 deselected（slow 默认跳过）
python -m pytest tests/ -q

# 5) 真实 checkpoint 实验（LLaDA/Dream/Fast-dLLM remote code 需 transformers 4.53.1）
#    新建实验 venv（--system-site-packages 继承系统 torch），详见 §1/§9b：
python -m venv --system-site-packages .venv-hf
. .venv-hf/bin/activate && pip install "transformers==4.53.1" "huggingface_hub<1.0" einops

# 6) 冒烟 → 正式：任何真实实验先跑 --quick，正式跑为预承诺单次（§7 有逐实验命令）
python -m scripts.<实验名> --quick ... # 分钟到小时级

# 7) 一键正式跑执行器（AR027，推荐在服务器上使用）：
python -m scripts.run_formal_queue --list          # 先看 7 面队列与工件路径
python -m scripts.run_formal_queue --quick         # 全部七面 quick 冒烟（不写正式工件）
python -m scripts.run_formal_queue --models-root /data/ar023_models
#    执行器会做六项硬 preflight（transformers 4.53.1 / CUDA / 24GiB 或 quick 16GiB /
#    权重文件 / AR023 probe 工件 / GPU 静默），失败即拒绝启动；失败即停、不重试；
#    日志与结构化摘要写入 results/optimization/formal_runs/<UTC>/（queue.log + queue.json）；
#    退出码 0=全通过 2=工件校验不一致 3=preflight/参数错误 4=实验失败。
#    存在正式工件时须 --force --force-reason "<理由>" 才允许覆盖（理由入审计日志）。
```

- 评测类实验另装 `requirements-bench.txt`（lm-eval / evalplus；evalplus 需 Unix-like）。
- 显存与耗时的逐实验预期见 §2；高算力服务器复现要点（Linux/TCC、锁频、权重获取、
  WDDM 限制、工件回填）见 **§7**。
- 判定语义（H1-H4 冻结门 / 诚实负结果 / 绝不 rerun-until-pass）见各实验脚本头注释与
  `AGENTS.md`「Experimental Validation Rule」。

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
| `ar021_generation_loop_bench`（生产生成回路三模式） | Fast-dLLM-v2-1.5B bf16（同上权重；AR019 venv） | 实测峰值 ≤5374.5 MB（~5.4 GB），16 GB 卡可跑 |
| `ar022_speculative_quality_bench`（四模式质量对齐基准） | Fast-dLLM-v2-1.5B bf16（同上权重；AR019 venv） | 实测峰值 ≤5840.1 MB（~5.8 GB），16 GB 卡可跑 |
| `ar023_family_probe` / `ar023_cross_family_bench` | LLaDA-8B-Base / Dream-v0-Base-7B bf16（~30GB 权重下载） | ≥24 GiB（8B bf16 权重 ~16 GiB + 激活；单卡 40 GiB 更舒适，全程 LLaDA ~4-5h / Dream ~6-8h） |
| `ar025_multi_parent_bench`（多 parent 批量折叠合成面） | 合成小模型（16 形状点：多链×N 候选×seq_len） | CPU 即可（分钟级） |
| `ar026_real_graph_bench`（真实 checkpoint 四模式图基准） | LLaDA-8B-Base / Dream-v0-Base-7B bf16（同 AR023 权重；AR019 venv） | ≥24 GiB（8B bf16 ~16 GiB + 图池捕获侧 VRAM；16 GB 卡为紧约束，正式跑建议 ≥24 GiB 或按 quick 冒烟实测余量） |

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
#
# 11) AR021 生产生成回路三模式基准（Fast-dLLM-v2-1.5B，同 venv/权重）
#    a) 前置：第 9 步的 AR019 venv（transformers==4.53.1）与本地权重目录
#    b) 基准（真实 generate() 三模式 baseline/folded-serial/folded-batched；
#       预承诺单次纪律，不重跑钓鱼；全量 reps=30 全程 ~10min（timed 段
#       ~7min），峰值 VRAM ~5.4GB；--quick 冒烟 = 单 prompt/reps=3/2 块 ≤5min）：
<venv目录>\Scripts\python -m scripts.ar021_generation_loop_bench \
    --model-path <本地权重目录> --hf-home <HF缓存目录> --reps 30
#    工件：results/optimization/ar021_generation_loop_bench.json
#    （判定式全文/c=4/canary 二分口径/A5 摊销注记冻结于 meta）
#    对照数字：p50 baseline 2840.7 / folded-serial 2948.3 / folded-batched
#    1193.9 ms（batched vs serial 2.4695×、vs baseline 2.3795×）；
#    H1 supported（真实 checkpoint bit-exact：token 一致率 1.0、逐前向
#    max_abs_diff=0.0）；H2 inconclusive（Δ=−107.5181ms 未越 2σ=265.3231ms，
#    小形状串行折叠开销>收益，如实记录）；H3 supported（速度分面；质量分面
#    report-only：exact-match 0.0、置信低于 baseline）；H4 supported（在域窗
#    串行捕获/重放>0、批量捕获 5/5 窗、账本 c=4 界内、canary 降级类 timed=0、
#    AR017 事件全落 warmup=中性）
#
# 12) AR022 四模式投机质量对齐基准（Fast-dLLM-v2-1.5B，同 venv/权重）
#    a) 前置：第 9 步的 AR019 venv（transformers==4.53.1）与本地权重目录；
#       导入式复用 ar021 助手（脚本以 `python -m scripts.ar022_...` 运行，
#       ar021 脚本本体零改动）
#    b) 基准（四模式 baseline/folded-serial/folded-batched/folded-batched-greedy
#       ——MFF_C 与 MFF_B 同构独立实例 + candidate_include_greedy=True；
#       预承诺单次纪律，不重跑钓鱼；全量 reps=30，峰值 VRAM
#       ~5.8GB；--quick 冒烟 = 单 prompt/reps=3/64 token ≤5min，其复现自检
#       降级为结构性验证（num_tokens=64 不可比全量先验，如实注记））：
<venv目录>\Scripts\python -m scripts.ar022_speculative_quality_bench \
    --model-path <本地权重目录> --hf-home <HF缓存目录> --reps 30
#    工件：results/optimization/ar022_speculative_quality_bench.json
#    （H1-H4 判定式全文/贪心锚定声明/复现自检先验 _AR021_PRIORS 冻结于 meta；
#    复现自检 strict 失配 ⇒ 四 verdict 全置 blocked，不静默替换）
#    对照数字：H1-H4 全 supported——H1 三面全过（合并位置级 4/4=1.0>0.5、
#    合并块级 4/4=1.0>batched 0.0、posthoc 0.369661/0.570341 逐位==baseline
#    gap=0.0）；H2 速度保持（Δ=+859.682ms 加速侧，σ=808.3707）；H3 早停恢复
#    （长度 34/66 ==baseline，batched 160/77 完全恢复）；H4 全栈（MFF_C 在域窗
#    捕获/重放 30/30>0、三模式 canary 降级类 timed=0、账本界内、AR017 中性）；
#    p50 baseline 3096.8 / serial 3205.4 / batched 1300.6 / greedy 440.9 ms
#    （greedy vs baseline 7.0236×、vs batched 2.9498×、batched vs baseline
#    2.381× 与 AR021 2.3795× 跨工件一致）；greedy_win_rate 1.0（2/2 观测面）；
#    复现自检 strict PASS（batched sidecar 逐位==AR021 先验，无
#    reproduction_anomaly）
#
# 13) AR023 跨族投机（LLaDA-8B-Base + Dream-v0-Base-7B 真实 checkpoint）
#     a) 权重下载（探针脚本自带 curl 清单 + SHA256 校验；~30GB 总量，
#        断点续传 `-C -`，已完成文件自动跳过重下仅重验）：
# <venv目录>\Scripts\python -m scripts.ar023_family_probe \
#     --models-root <权重根目录> --hf-home <HF缓存目录>
#     工件：results/optimization/ar023_family_probe.json
#     （两族结论 dtype_choice/graph_enabled/degraded；CUDA 五面子进程隔离
#      ——图面分配器断言会损坏进程级 CUDA 状态，勿与其它 GPU 任务同进程；
#      Dream 自定义注意力拒 int64 mask，探针已统一 bool mask）
#     定案结论：两族均 bf16 / degraded=False / graph_enabled=False
#     （llada bf16 15265MB/int8 10966MB；dream bf16 14836.7MB/int8 12093MB）
#     b) 四模式跨族基准（预承诺单次纪律，不重跑钓鱼；每族一工件；
#        全量 reps=30/P=2/128 token。墙钟以实跑为准：本机（16GB WDDM
#        工作站 + CPU 竞争）llada 正式跑实测 ≈29.2 h（单样本 p50 ≈330 s）；
#        高算力服务器预期显著缩短（数量级估计，未实测——以新机计时为准）；
#        --quick 冒烟 = 单 prompt/reps=3/64 token
#        每族 ~10min，写独立 _quick 工件、永不覆写正式工件）：
# <venv目录>\Scripts\python -m scripts.ar023_cross_family_bench \
#     --family llada --models-root <权重根目录> --reps 30
# <venv目录>\Scripts\python -m scripts.ar023_cross_family_bench \
#     --family dream --models-root <权重根目录> --reps 30
#     工件：results/optimization/ar023_llada_bench.json /
#           results/optimization/ar023_dream_bench.json
#     （H1 分级质量门三硬面 + 0.5 比值（边界含 1e-9 IEEE 噪声 epsilon）+
#      None-mean 拒判 + gap_bat==0 哨兵；H2 严格 >1.0；H4 canary/ledger 合取
#      + graph-off 中性注记 + 每模式 profiler 窗口 stable ratio；
#      baseline 双跑复现自检 strict 失配 ⇒ 全 verdict 置 blocked；
#      degraded 族/单模式失败（如 OOM）⇒ blocked 工件 + 探针交叉引用，
#      零伪造数字）
#     对照数字：（T006 定案后回填——预承诺 reps=30 单次，两族四模式
#      p50/H1-H4 verdict 以工件为准，勿以 --quick 冒烟数字替代）
#
# 14) AR025 多 parent 批量折叠合成面基准（合成模型，CPU 即可，分钟级）
#     python -m scripts.ar025_multi_parent_bench
#     工件：results/optimization/ar025_multi_parent_bench.json
#     （S1/S2/S3 判定式全文冻结于 meta；16 个合成形状点（chains×N×seq_len）
#      ——S1 A 侧合并 [sum(N),T] vs B 侧逐链 [N,T] logits parity（nonsplit
#      bitwise / split allclose 1e-6，任一 FAIL 全 verdict blocked）、
#      S2 write_op_count 严格小于逐链（账本 4 vs 40..160）、S3 合并/串行
#      投入比 report-only 0.9–0.975；meta.cache_budget_tokens_per_layer=65536
#      ——DD8 erratum：默认 8192 预算下 (parents+ΣN)·T 超限会触发
#      ActivationCache LRU 逐出在飞 parent（容量混淆因子，非机制缺陷），
#      首跑 blocked 证据工件 ar025_multi_parent_bench_eviction_blocked.json
#      独立保全；--quick 冒烟写独立 _quick 工件、永不覆写正式工件；
#      --mode real 见步骤 16（AR025 T006，已开放））
#     对照数字：verdicts_declared.overall=supported（S1/S2 全 supported、
#      blocked_reasons 空）；真实面结果以 AR025 T006 工件为准
#
# 15) AR026 真实 checkpoint 端到端图路径（生产捕获 + 四模式图基准）
#     a) 探针生产图面首验（每族子进程隔离，~30min）：
# <venv目录>\Scripts\python -m scripts.ar023_family_probe \
#     --models-root <权重根目录> \
#     --output results/optimization/ar026_family_probe_rerun.json
#     （勿写默认输出路径——会覆写 AR023 结论工件；graph 面 =
#      MFF+FoldedGraphRunner 生产捕获（captures>0 且零失败诊断才 ok），
#      graph_raw = 旧裸捕获诊断面；graph_enabled 结论只依据生产面）
#     b) 四模式基准 quick 冒烟（llada/dream 各一，~1-2h/族）：
# <venv目录>\Scripts\python -m scripts.ar026_real_graph_bench \
#     --family llada --quick --models-root <权重根目录>
# <venv目录>\Scripts\python -m scripts.ar026_real_graph_bench \
#     --family dream --quick --models-root <权重根目录> \
#     --out results/optimization/ar026_real_graph_bench_dream_quick.json
#     工件（quick）：results/optimization/ar026_real_graph_bench_quick.json
#     （dream quick）：results/optimization/ar026_real_graph_bench_dream_quick.json
#     c) 正式面（预承诺单次；reps=30/P=2/128 token，与 AR023 同协议；
#        本机 16GB WDDM 上 AR023 llada 正式跑实测 ≈29.2h → 须排多日窗口）：
# <venv目录>\Scripts\python -m scripts.ar026_real_graph_bench \
#     --family llada --models-root <权重根目录> --reps 30 \
#     --out results/optimization/ar026_real_graph_bench.json
# <venv目录>\Scripts\python -m scripts.ar026_real_graph_bench \
#     --family dream --models-root <权重根目录> --reps 30 \
#     --out results/optimization/ar026_real_graph_bench_dream.json
#     工件（llada）：results/optimization/ar026_real_graph_bench.json
#     （dream）：results/optimization/ar026_real_graph_bench_dream.json
#     注意：--out 必须显式给出——脚本默认输出路径不按 family 分键，
#     省略 --out 会让 dream 正式面覆写 llada 正式面
#     （schema=ar026_real_graph_bench；四模式 baseline 官方配方 /
#      folded-batched eager 锚 / graph-fixed / graph-adaptive —— 模式间
#      仅图旋钮差异（capacity 0.5 / pool 4 / adaptive on|off）；
#      H1 图奇偶：参与度门（零捕获 = inconclusive，永不冒充奇偶通过）+
#      三面（vs 锚 posthoc |Δ|≤1e-6、decode-length 逐 prompt 相等、
#      block-grid 全中率 ==1.0）；H2 严格 baseline/图加速 >1.0，图-vs-锚
#      预注册无方向（report-only）；H4 canary 为真实面：timed 窗
#      degradation_fallback_count 增量 ==0；复现双跑 strict 失配 ⇒ 全 verdict
#      blocked；degraded 族 ⇒ blocked 工件零伪造；EVR≥4 形状面显式豁免
#      （理由在 meta.geometry_inheritance_note：16GB VRAM 为绑定约束））
#     对照数字：待 T006/T007 实机工件（预承诺单次；勿以 quick 冒烟替代）
#
# 16) AR025 真实面双链并行投机基准（W2 lockstep sample_joint vs 逐链串行）
#     a) quick 冒烟（独立 _quick 工件，永不覆写正式面）：
# <venv目录>\Scripts\python -m scripts.ar025_multi_parent_bench \
#     --mode real --family llada --quick \
#     --models-root <权重根目录> --probe-artifact results/optimization/ar023_family_probe.json
# <venv目录>\Scripts\python -m scripts.ar025_multi_parent_bench \
#     --mode real --family dream --quick \
#     --models-root <权重根目录> --probe-artifact results/optimization/ar023_family_probe.json
#     工件（quick）：results/optimization/ar025_real_bench_llada_quick.json /
#      ar025_real_bench_dream_quick.json
#     b) 正式面（预承诺单次，reps=30、num_tokens=128、P=2 prompt 对截断公共长度）：
# <venv目录>\Scripts\python -m scripts.ar025_multi_parent_bench \
#     --mode real --family llada \
#     --models-root <权重根目录> --probe-artifact results/optimization/ar023_family_probe.json \
#     --reps 30
#     工件：results/optimization/ar025_real_bench_llada.json（dream 同理）
#     c) 判定式（DD8 冻结，先于数据）：H1 逐位（A vs B 两链 token 逐位相等硬门 +
#      logits allclose 归因回放）；H2 严格 p50(B)/p50(A)>1.0；交付线 ≥10%（不过
#      ⇒ FR7 冻结降级为 blocked 注记，非错误）；H4 写账本严格小于；canary =
#      graph-off null note。复现双跑 strict 失配 ⇒ 全 verdict blocked；degraded
#      族（probe 结论）⇒ blocked 工件零伪造数字。
#     d) dtype 纪律：默认 auto = AR023 装载纪律（bf16 优先、int8 回退）——正式跑
#      一条 dtype 纪律贯穿到底。`--real-dtype int8` 仅为 16GB 级主机冒烟逃生门；
#      `--real-num-tokens` 仅为慢机冒烟规模逃生门（正式跑不传 = 冻结值 64/128）。
#     e) 本机 16GB WDDM 限制（实测）：bf16 权重 15.8GB + 折叠缓存 4096 行/层
#      满载显存 ⇒ 进程硬崩无 traceback（VRAM 上限，非代码回归）；int8 逃生门
#      可装载但 bitsandbytes 量化在 WDDM 坏态下停滞（fresh 重启复现）。正式
#      面与完整冒烟应在 Linux/TCC 高算力机执行后回填工件（步骤 7 流程）。
#     对照数字：待正式工件（预承诺单次；勿以 quick 冒烟替代）
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

## 7. 高算力服务器复现（推荐路径）

本清单命令在任意 Linux/TCC GPU 服务器上等价可用；换到高速算力机时的推荐流程与差异点：

1. **环境（一次性）**：

   ```bash
   git clone <repo> && cd ActFold
   python -m venv .venv && . .venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   # 真实 checkpoint 实验需要 remote-code 兼容版本（requirements.txt 已限 <5）：
   pip install "transformers==4.53.1" "huggingface_hub<1.0" einops
   python -m pytest tests/ -q -m "not slow"    # 期望全绿（CUDA 面自动启用）
   ```

2. **权重获取**：探针脚本内置 curl 清单 + SHA256 校验（见步骤 13a）；也可直接
   `huggingface-cli download` 后以 `--models-root <目录>` 指向本地布局
   （`<root>/llada`、`<root>/dream`）。
3. **时钟与干扰**：对比类实验锁频（如 `nvidia-smi -lgc <clock>`）并独占 GPU；
   正式跑为预承诺单次承诺，不与其它任务共享（本仓库的共享工作站教训：CPU 竞争
   会把单样本 p50 从秒级推到数百秒）。
4. **Windows WDDM 限制（重要）**：小内核 CUDA Graph 在 WDDM 下不消除提交延迟
   ——同一份 AR002 代码在本机实测出现图重放比 eager 慢约 4.5x（worktree 二分
   确证非代码回归），BS-007 性能判定因此在该类主机上诚实 skip。**图路径的正式
   时延结论应在 Linux/TCC 主机取得**；迁移后重跑步骤 15 的 graph 模式并以新
   工件为准。
5. **规模化预期**：AR023/AR026 的 reps=30 正式跑在本机（16GB WDDM + CPU 竞争）
   实测 llada 约 29.2h（单样本 p50 约 330s）；数据中心 8B bf16 吞吐远高于此
   （数量级估计、未实测——以新机首跑计时为准）。建议先跑 --quick 冒烟取得
   本机单样本实测，再按比例估算正式窗口并一次性排程。
6. **产物回填**：新机工件落到同一 `results/optimization/` 路径后，按 README
   优越性总表口径从工件回填（禁止手输数字；跨机数字须在工件 meta 注明
   GPU/时钟状态）。
7. **一键执行器（AR027，推荐）**：`scripts/run_formal_queue.py` 按固定顺序执行
   七个正式面（AR023-llada/dream → AR024 → AR025-real-llada/dream →
   AR026-llada/dream），自动完成 §1 的环境/显存/权重/probe/GPU 静默检查、逐面
   子进程隔离、工件 schema 与冻结判定重算校验、结构化日志。命令：

   ```bash
   # 队列预览（不执行）
   python -m scripts.run_formal_queue --list
   # 冒烟（全部七面 quick，工件 *_quick.json，永不碰正式路径）
   python -m scripts.run_formal_queue --quick --models-root /data/ar023_models
   # 正式（预承诺单次；建议先 --only 校验单面再全量）
   python -m scripts.run_formal_queue --models-root /data/ar023_models
   # 子集 / 覆盖审计
   python -m scripts.run_formal_queue --only ar026-llada,ar026-dream --models-root /data/ar023_models
   python -m scripts.run_formal_queue --force --force-reason "server rerun after env fix"
   ```

   退出码 `0` 全通过（含合法 blocked）/ `2` 工件校验不一致 / `3` preflight 或
   参数错误 / `4` 实验失败或覆盖保护拒绝；每次运行在
   `results/optimization/formal_runs/<UTC>/` 下生成 `queue.log` 与 `queue.json`
   （逐面 argv、耗时、退出码、工件路径、校验结论）。失败即停、不重试、不续跑。
   `--allow-busy` 仅覆盖 GPU 静默一项且写入日志（其它 preflight 项不可豁免）。
