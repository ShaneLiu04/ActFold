# [AR024] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR024 |
| AR 主题 | LLaDA 投机提交粒度质量面缓解：子块提交 + 渐进填充 + EOS 感知候选（P1-7 第十档） |
| 关联 SR | 无（P1-7 演进线；`next_rung_memo.md` 候选 A，用户定案 ①+②+③ 全做） |
| 日期 | 2026-10-16 |
| 状态 | Draft |

## 1. 背景与目标

**质量面缺口（AR023 quick 冒烟实证，非正式定案）**：LLaDA-8B 四模式 quick
基准 H1 refuted——投机区 offset-2 EOS 早停（解码长度 3 vs baseline 64），
对齐模式与 batched 逐位相同（贪心锚定未恢复质量面）；速度面 H2 5.41×
（59.3s→11.0s）。**EOS 审计已排除代码缺陷**（suppress 序两路径对称）——
长度分歧是"空白画布单发整块提交 vs 多轮渐进去噪"的固有质量面差。

**机制洞见（Dream 对照）**：Dream 超步窗口（s=8、预算合并、置信度选位、
渐进提交）在同等投机机制下质量面保持（长度恢复 64/64/64）——**质量面是
提交粒度的函数，不是族的函数**。LLaDA 的投机单元 = 整块 one-shot 填充，
把全部去噪压力压在一次前向；Dream 的单元 = 小窗口渐进提交。

**目标**：把 LLaDA 投机路径的提交单元从整块改为**子块序列**，并叠加两项
候选集/提交流精化（渐进填充、EOS 感知覆盖），使对齐模式恢复 baseline
质量面（H1 分级硬面）同时保持 H2 严格加速。AR023 四模式基准协议与判定
式全承（面向族官方配方 baseline 定案）。

> 正式依据现状：AR023 T006 正式 reps=30 运行中（预承诺单次）；本 AR 的
> quick 冒烟证据为机制动机，正式验收独立预承诺。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `actfold/models/llada_sampler.py`：投机路径提交单元子块化（FR1）+ 窗口
  内渐进填充（FR2）+ 候选集 EOS 感知覆盖（FR3）——**仅投机路径**（N>0），
  N=0 参照路径逐字节不动
- `actfold/models/speculative_core.py`：仅当窗口迭代/部分提交语义确属
  机制核面时最小扩展（FR1 依赖项；目标零改动——Dream 超步已证核支持窗口
  单元；设计期定案）
- `actfold/utils/config_manager.py` + `configs/per_model/llada_8b.yaml`：
  三旋钮消费面（FR4）
- `tests/test_llada_granularity.py`（新）+ 既有冻结面零破坏（FR6）
- `scripts/ar024_granularity_bench.py`（导入式复用 ar023 助手，AR022 先例）
  + 正式工件 `results/optimization/ar024_llada_granularity_bench.json`（FR5）
- 文档收口五件 + README 全量重写（FR7，AGENTS 既定规则）

**Out of Scope（本 AR 不做的）：**
- N=0 参照路径任何改动（官方块解码语义 = 逐字节铁律）
- Dream/Fast-dLLM 适配器改动（机制洞见已由 Dream 实证；子块化仅 LLaDA）
- 机制核 `_speculative_commit` 选定规则（min-prob）改动——单元变、选定
  规则不变
- eval 任务级跑分（备忘录候选 B，独立 AR）
- 图捕获面（两族探针结论 graph-off；graph 静态体元组解包挂 backlog）

## 3. 功能需求

### 3.1 FR1 子块投机提交（机制 ①）

**描述**：LLaDA 投机单元从整块改为 `speculative_sub_block_size`（s）位置的
子块窗口序列——块内窗口逐个投机：每窗口构建 `[N, T]` 候选画布（填充该
窗口的掩码位，条件于已提交前缀）、一次批量折叠前向验证、min-prob 选定、
窗口提交、折叠链延续。

**触发条件**：`num_block_candidates > 0`（投机开启）。

**期望行为**：
- 窗口迭代：块内按 s 切窗（末窗取余），逐窗投机轮；每窗候选画布仅填充
  该窗掩码位（`_build_unit_forced` 窗口超集同式），已提交前缀为条件
- 选定/提交/链延续/早停语义：`_speculative_commit` 原样（单元窗口化，
  选定规则不变）；胜者提交跳过该窗剩余 remasking 迭代
- **默认 s=8**（用户定案"开启投机时默认"；与 Dream `speculative_steps=8`
  跨族机制一致）；`1 ≤ s ≤ block_size` 校验，超界构造期 `ValueError`
  不静默截断（block_size < 8 配置需显式设置）
- N=0 参照路径逐字节不动（该字段在 N=0 下不被读取）

**异常处理**：s 越界构造期 `ValueError`；其余入口校验（B>1 / 无
folded_model / 无 branch_ids 能力 / cfg_scale>0）全承 AR023。

**验收标准**：
- Given 合成模型 + LLaDA 采样器 + N=4 + s=8 + block_size=32，When 投机
  运行，Then 每块 4 个投机窗、每窗一次 `[N,T]` 批量前向、逐窗链延续，
  总前向次数与窗口数一致
- Given s=block_size，When 投机运行，Then 与 AR023 整块语义逐位相同
  （粒度旋钮 A/B 锚）
- Given N=0（默认关），When 全部既有面运行，Then 与 AR023 末态逐字节
- Given s=0 或 s>block_size，When 构造，Then `ValueError`

### 3.2 FR2 窗口内渐进填充（机制 ②）

**描述**：opt-in 的窗口内两阶段提交精化——阶段一提交胜者行高置信 top-k
填充位，阶段二单次重前向（部分提交画布）后填充窗口余位。

**触发条件**：`progressive_refill=True`（默认 False）且投机开启。

**期望行为**：
- 阶段一：胜者行填充位按 draft 置信排序取 top-k 提交（k 参数设计期冻结，
  候选 s//2）
- 阶段二：单行重前向（非投机——无第二候选画布），余位按前向置信填充
- 确定性、零设备回读合同全承（专用 CPU generator 不消费——阶段二为
  确定性填充，无随机行）
- 每窗恰好一次额外前向（成本上界冻结）

**异常处理**：窗口余位为 0（k==s）时阶段二跳过（无额外前向）。

**验收标准**：
- Given N=4 + s=8 + refill + k=4，When 窗口投机，Then 每窗两次前向
  （一次 [N,T] 批量 + 一次单行）、top-4 置信位先提交、余位后填充
- Given refill=False（默认），When 投机运行，Then 与 FR1 单阶段逐位相同
- Given k==s，When 窗口投机，Then 阶段二跳过、行为退化为 FR1

### 3.3 FR3 EOS 感知候选覆盖（机制 ③）

**描述**：候选集保证 stop-token 假设的显式覆盖——行 N-1 无条件构建为
"排除 EOS 的 draft argmax"填充行（延续假设），与行 0（贪心锚定，可能
含 EOS）构成 stop-vs-continue 假设对，min-prob 公平仲裁。

**触发条件**：`candidate_eos_aware=True`（默认 True，投机开启时）。

**期望行为**：
- 行 N-1 = draft top-k 池排除 EOS 后的 argmax 填充（纯设备索引构建，
  零随机性消耗、零设备回读；无条件构建——EOS 不在池中时该行退化为
  与贪心行同值，min-prob 平局按行序裁定，确定性不变）
- **N 总数语义不变**：行 N-1 从随机行改造为延续假设行（j=1..N-2 保持
  随机行、种子公式行号不变）；N=1 时行 0 兼任两角色（构造恒同）
- 与 `candidate_include_greedy` 独立正交（无贪心锚定时行 N-1 仍可构建）
- 早停语义：胜者行含 EOS 时与族参照同式（#45 合同不变）

**异常处理**：无新增异常面（构建恒可，无检测分支）。

**验收标准**：
- Given EOS 在 draft 池中的 fixture，When eos_aware=True，Then 行 N-1
  不含 EOS 且 == 排除 EOS 的 argmax 填充、行 0 可能含 EOS、行 1..N-2
  与 eos_aware=False 逐位相同
- Given EOS 不在池中的 fixture，When eos_aware=True，Then 行 N-1 == 行 0
  （退化同值）且构建确定性（双跑 bit-exact）；N=1+greedy 退化 fixture 下
  选定结果与 eos_aware=False 一致——一般 N 时行集合构造性不同（随机行
  N-1 被替换为延续行），选定不预设一致（design §4.1.5 措辞精化，
  2026-10-16 design 门控 G14 回写）
- Given eos_aware=False，When 投机运行，Then 与 AR023 候选构建逐位相同
- Given N=1 + eos_aware=True，When 构建，Then 单行兼两角色、无异常

### 3.4 FR4 配置面

**描述**：三旋钮入 `LLaDASamplerConfig` + `ActFoldConfig` 透传（YAML
可消费）+ per_model YAML 注释段更新。

**期望行为**：
- `speculative_sub_block_size: int = 8`（FR1）/ `progressive_refill:
  bool = False`（FR2）/ `candidate_eos_aware: bool = True`（FR3）+
  `__post_init__` 对称校验（s<1 / s>block_size / refill_top_k<1 或
  >s → `ValueError`，AR022 先例不静默降级）
- `ActFoldConfig` 三字段消费 YAML 键免 unknown-key 告警（#5）；
  `llada_8b.yaml` 注释段更新为 AR024 推荐栈（s=8 + eos_aware on +
  refill off）与 A/B 还原写法（s=block_size + eos_aware off 复现
  AR023 语义）
- Dream/Fast-dLLM config 零改动

**验收标准**：
- Given YAML 含三键，When `load_config`，Then 程序化消费零 UserWarning
- Given 默认配置，When 构造各族 sampler，Then Dream/Fast-dLLM 与
  AR023 末态逐字节、LLaDA N=0 逐字节

### 3.5 FR5 验收基准与工件

**描述**：`scripts/ar024_granularity_bench.py --family llada`（AR023
四模式协议与判定式全承，导入式复用 ar023 助手——AR022 先例，禁止复制
粘贴重复）；对齐模式 = AR024 推荐栈。

**期望行为**：
- 四模式：baseline（官方配方）/ folded-serial（N=0+MFF）/ folded-batched
  （**s=block_size + eos_aware=False——AR023 语义 A/B 锚**）/
  folded-batched-greedy（**s=8 + eos_aware=True——AR024 推荐栈**）
- **粒度 A/B 构造性隔离**：batched 与 greedy 同 N/形状/seed/prompt/独立
  同构 MFF，唯一差异 = 提交粒度栈——H1 的 gap 收缩/长度恢复/exact 改善
  直接归因粒度机制
- 判定式全承：H1 分级质量硬面（gap ≤ 0.5 比值 + eps / 逐 prompt 长度
  恢复 / exact 严格改善 / None 拒判 / 哨兵 / 非阻塞强结论层）；H2 严格
  `p50(base)/p50(aligned) > 1.0`；H4 canary/ledger 合取 + profiler 窗口
  reset；复现双跑 strict 失配 ⇒ 全 verdict blocked；单模式失败 ⇒ blocked
  工件零伪造（#10）
- **机制 ② 的实机面**：`--refill-face` 报告面（quick 冒烟尺度验证 refill
  开启时的四模式行为；正式 verdict 不含 refill——② 为 opt-in 机制，其
  UT/mutation 面为交付验收，实机推荐栈不含②，设计期冻结该边界）
- seed 2026 / reps=30 预承诺单次 / #39 墙钟纪律 / `--quick` 独立工件
- 工件：`results/optimization/ar024_llada_granularity_bench.json`

**异常处理**：全承 AR023 blocked 语义。

**验收标准**：
- Given reps=30 运行，Then 工件含四模式墙钟 + H1/H2/H4 判定 + 复现自检
  + meta 全要素（含粒度栈常数）
- Given 对齐模式，Then 长度恢复/gap 收缩/exact 改善三硬面自动 verdict
- Given 降级/失败，Then blocked 工件 + 探针交叉引用零伪造

### 3.6 FR6 质量门

**描述**：全量 pytest 基线（1022/0/4 之上只增不减）+ 新 UT + 静态检查。

**期望行为**：新 `tests/test_llada_granularity.py`（UT-2401 起编）：
三机制各自面（窗口迭代/前向次数/链延续/默认零扰动/s=block_size A/B 锚/
校验；refill 两阶段/退化/确定性；eos_aware 假设对/退化同值/N=1 边界/
与贪心锚正交）+ config/YAML 面 + 基准脚本合同面；既有冻结面
（`test_llada_speculative.py` 19 / `test_diffusion_samplers.py` /
`test_ar023_bench.py` 9 等）零破坏；pyflakes 零 / mypy clean /
black ≤100 列（仅新增行）。

**验收标准**：
- Given 新 UT 与既有面就绪，When 全量 pytest，Then ≥1022 passed /
  4 skipped / 0 failed
- Given lint 三件，When 对新增改动文件执行，Then clean

### 3.7 FR7 文档收口五件

**描述**：README 全量中文重写（AGENTS 既定规则，数字从工件重算）+
AGENTS #46（提交粒度契约）+ GUIDE 第三十一部分 + CHANGELOG AR024 条目
+ RERUN 第 14 步。

**验收标准**：
- Given T008 文档五件产出，When 交付审阅（数字溯源逐项核查），Then 全部
  数字工件可溯、README 五节全覆盖（创新表新行/最新结果表/快速上手/
  质量/路线图）

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 对齐模式 vs baseline 加速比 p50 | > 1（H2 硬门，正式工件） |
| 质量 | 对齐模式 H1 分级硬面 | gap ≤ 0.5 比值 + 长度恢复 + exact 严格改善（正式工件） |
| 资源 | VRAM | 子块画布与整块同形（`[N,T]` 不变）——无新峰值面 |
| 可复现 | 种子/环境/粒度常数 | seed 2026 + s/k/eos 常数全入工件 meta |
| 墙钟 | 测量纪律 | perf_counter+sync、CUPTI null 注记、reps=30 预承诺单次 |

## 5. 约束与假设

**约束：**
- **GPU 排队**：AR023 T006 正式跑独占 GPU（预承诺采样期禁 CUDA 工作）——
  AR024 CPU 面（Red/Green/mutation/脚本 UT/文档起草）先行，实机面
  （quick 冒烟 T005 / 正式 reps=30 T006）排队至 GPU 空闲；**AR023 先归档、
  AR024 后归档**（并行开发、串行结档）
- N=0 参照逐字节铁律；`_speculative_commit` 选定规则不变；#39/#10/#45
  合同全承；mypy `--ignore-missing-imports` clean
- 组件详设缺失（`specs/component-detail-design/` 不存在——沿 AR023
  先例，领域知识以 AGENTS.md #43-#45 + 代码为准）

**假设：**
- AR023 探针工件结论可复用（LLaDA degraded=False / bf16 / graph-off）——
  无需新探针，FR5 直接交叉引用
- AR023 T006 完成后存在 GPU 可用窗口（AR024 实机面依赖）
- LLaDA 远程 block 元组返回面已被 UT-2335/MFF 解包钉死（AR024 子块
  路径经 MFF eager 链，不触图静态体）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 提交粒度（commit granularity） | 一个投机单元内一次提交的填充位数；质量面的假设自变量 |
| 子块窗口（sub-block window） | 块内 s 个连续掩码位；LLaDA 投机单元的粒度化形态（Dream 超步窗口的同构物） |
| 延续假设行（continuation-hypothesis row） | 行 N-1：排除 EOS 的 draft argmax 填充行；与贪心行构成 stop-vs-continue 假设对 |
| 渐进填充（progressive refill） | 窗口内两阶段提交：高置信 top-k 先提交 + 单次重前向 + 余位填充（机制 ②） |
| AR024 推荐栈 | s=8 + candidate_eos_aware=True + progressive_refill=False（对齐模式的机制组合） |
