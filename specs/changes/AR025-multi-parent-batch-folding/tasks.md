# [AR025] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR025 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md rev 3（design 门控 PASS）|
| 创建日期 | 2026-10-16 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | Red UT：多 parent 批量机制面测试全红（生产缺失：`parent_branch_ids` 参数/per-row 父装配/逐行 gate 对拍）；冻结面（AR015 批量契约 UT + AR023 族投机面）零破坏确认 | - | passing | 覆盖点矩阵=design §6.1（rev 3 定稿）；UT-2400..UT-2427 共 28 例：26 红+2 绿锚点 |
| T002 | Green 机制核心：`ManualFoldedForward.parent_branch_ids` 尾参（互斥校验/None-attach 零扰动）+ per-row 父装配（gather + 行级降级，OQ1 定案形态）+ 逐行 gate/merge；mutation 全杀还原 | T001 | passing | UT-2401..UT-2415 全绿（16/16）；spy 计数面即 mutation-kill 探针（UT-2407 counts==[1]*L/UT-2410 counts==[0]*L/UT-2413 顶层 drop） |
| T003 | Green 发布与循环接线：`put_batch` 逐行胜者发布验证（#42 零改动预期验证）+ `_batched_forward_supported` 条件泛化 + `folded_generate` 多链入口（OQ2 定案形态）；mutation 全杀还原 | T002 | passing | UT-2416..UT-2424 全绿（9/9）+ UT-2427 谓词泛化；`_forward_batched_multi` 前缀捐赠修正（见进度记录） |
| T004 | 配置面：合批开关（默认 off）+ `ActFoldConfig` 消费 + YAML 注释段；默认 off 逐字节兼容 UT | T002 | passing | UT-2425/UT-2426 绿；known-key 过滤随 dataclass 字段自动生效；注释段落点 dream/fast_dllm 两 yaml（llada_8b.yaml 为 AR024 并行会话在飞文件，注释段随其提交后补） |
| T005 | 合成基准脚本：多 parent 批量 vs 分批 vs 串行三方（路径分档容差对拍 + launch 账本）+ multi-geometry ≥4 形状点 + `--quick` 冒烟 | T003, T004 | passing | `scripts/ar025_multi_parent_bench.py` --mode synth：判定式 S1/S2/S3 冻结入脚本（DD8 原文）；8 点网格×双构造（nonsplit 逐位/split allclose 1e-6）；`--quick` 单点冒烟 CPU 过（S1/S2 supported，ops A/B/C=4/40/40，ratio 0.9，独立 _quick 工件）。**正式 8 点面已执行（两轮，均有档）**：首跑 5 点 S1 FAIL（T=1024 全档+(4,8,256)）→ 冻结判定式落 blocked（工件保全 `ar025_multi_parent_bench_eviction_blocked.json`）；根因诊断（gate 仪器化+逐层 fetch 追踪）=**容量混淆因子**——缓存预算 8192 下 A 侧 L0 burst put_batch 按 token 行数 LRU 逐出在飞 parents（先于子条目写入故最旧）→ L1+ 取父 miss → 整批重算（值正确、折叠丢失），与逐链交错路径（每层 fetch 刷新幸存）不对称；FAIL ⟺ (chains+chains·N)·T>8192 完全吻合；诊断验证预算 65536 恢复 parity。**DD8 erratum**（design rev 4）：预算按网格最坏人口 (4+32)·1024=36864 取 65536，冻结判定式零改动重跑——**16/16 全过：S1 supported（nonsplit 逐位/split allclose 1e-6）+S2 supported（严格 4/40..4/160）+S3 0.9-0.975**（工件 `ar025_multi_parent_bench.json`，meta 含预算字段+erratum 注记）；机制级逐出边界入 DD10（真实面几何 ~2560 行/层不触发）。--mode real 为 T006 交付（诚实脚手架，parser.error 零伪数字） |
| T006 | 真实 checkpoint 基准：LLaDA/Dream 双链并行投机合批 vs 分批 A/B（reps=30 预承诺单次定案，ar019 venv；AR023 T006 基准完成后执行）；工件 `ar025_multi_parent_bench.json` | T005 | pending | **设计定案（design rev 4 DD8a，门控 PASS WITH CONDITIONS ses_eb75aa130ffeOVCYAtYBaDeYxf——F1/F2/F3/N6/N7 全部当轮修入，T006 实现解锁）**：接线=W2 lockstep `sample_joint`（单实例+per-chain 画布/折叠链状态，双实例禁令=branch id 碰撞；`_speculative_commit` 按 forward 接缝两段拆分，合并闭包覆写 `parent_branch_id`→`parent_branch_ids`；temperature>0 入口 ValueError）；UT 前置守卫（合成族 sample_joint vs 逐链 sample 逐位相等+账本严格小）；计时对象=两 prompt 解码总墙钟（A=sample_joint×1 / B=sample()×2 背靠背，AR023 测量同构）。**实现待 AR024 sampler 在飞文件落地+AR023 基准完成（PID 8292 仍在跑）+GPU 空闲** |
| T007 | 质量门：全量 pytest（1022/0/4 起只增不减，ar019 venv canonical 口径）+ pyflakes/mypy/black + demo 三锚点不动确认 | T003, T004 | pending | |
| T008 | 文档五件：README 全量中文重写（AGENTS 强制规则）+ AGENTS 新条目（多 parent 契约）+ GUIDE 新部分 + CHANGELOG + RERUN 新步骤（数字工件溯源） | T006, T007 | pending | |
| T009 | review 门控（只读子代理：spec/design 合规 + 代码审查），全部问题（含 Minor/Nit）修复循环至 PASS | T008 | pending | |
| T010 | ST 验收：st-cases.md 生成 + 执行（正常/边界/异常/回归四类） | T009 | pending | |
| T011 | 归档 `specs/archive/AR025-multi-parent-batch-folding/`，本地 commit + gitee 分段推送（AGENTS Gitee Segment-Push Protocol）+ 双端树验证 | T010 | pending | 永久 commit id 分歧预期 |

## 状态说明

- `pending`：待开启
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-16 需求起草 + req 门控第 1 轮 PASS（llada 正式基准运行期间，零实验先行）

- 选题定案（用户裁决）：指南 Part 25 遗留主项"变长/多 parent 批量"——
  本 AR 切分**多 parent**（等长约束保留），变长 pad 面明示 Out-of-Scope
  留下一 AR；并行 AR024（LLaDA 提交粒度）同为用户裁决"都要做"。
- 调研事实锚定：`_batched_forward_supported`（folded_generation.py:310-334）
  三条件边界；#41 单 parent expand 契约；#42 put_batch 批量写就绪/
  批量读缺失（folded_transformer.py `_forward_batched` :498/:514-522/:536
  单键 fetch/expand 广播，无 per-row gather）；机制数学基础（行间无
  耦合——泛化是读路径泛化非新算法）。
- srs.md rev 1 起草 → req 门控第 1 轮（审查代理 ses_eb8979327ffeDKNbfXYBzcD1b9）
  **PASS**（N1-N4 证据核对全 YES：三条件代码事实/#41 原文/#42 put_batch/
  单键广播无 gather 逐行核实）+ Minor×8 当轮修入：
  - G1 引用措辞：AR020 尾注"下一步"系伪引文 → 改为 srs §2 Out-of-Scope
    转述（"变长/多 parent 批量（AR016 遗留，正交）"）；
  - G4/G8 FR7 补 multi-geometry ≥4 形状点承诺（链数{2,4}×N{4,8}×T 两档
    网格）；质量门口径显式继承 AR023 rev 3 分级（构造性逐位一致故 H1
    简化为硬层+注记）；
  - G6 降级触发器冻结式（H2 delta_p50 <10% 且 H1 硬层 PASS ⇒ 真实面
    blocked 注记交付）入 FR7 判定式，先于数据；删"质量门风险高"模糊触发；
  - G7 对拍容差路径分档（非 split 逐位/split allclose 1e-6，#41 先例）；
    验收 5 基线标注 ar019 venv canonical 口径；launch 账本=write_op_count
    代理语义统一+#39 CUPTI null 注记入 meta；
  - G9 模板压缩注记（四段压缩版，design 阶段展开）；**AR 编号冲突解决**：
    specs/changes 同日双 AR024（本 AR 与 commit-granularity）→ 先占者
    （AR024-commit-granularity，用户裁决"都要做"的候选 A）保留编号，
    本 AR 改号 **AR025**（目录/srs/tasks 全部同步）。
- 教训留档：PowerShell `-replace`+`Set-Content -Encoding utf8` 对中文
  md 造成双重编码损坏（读回乱码）——srs/tasks 已用 write 工具整体重写
  修复；此后中文 md 一律 write/edit 工具（与 D:\RSI 会话纪律一致）。

### 2026-10-17 design 起草 + 门控 2 轮 PASS（llada/dream 基准运行期间，零实验先行）

- design.md rev 1 起草：核心洞察经代码核实成立——批量 eager 体的
  gate/forced/profiler/merge 逐行机制面已 per-row，整批粒度仅父装配
  （单键 fetch→expand，:490-536）与三分支决策（:582-615）；DD1-DD10
  决策+UT 矩阵+PlantUML+复用映射；OQ1 掩码式/OQ2 folded_generate_batched
  双定案。
- design 门控第 1 轮（审查代理 ses_eb85d707bffeut2qVgEtMzVCIm）
  **FAIL**（G12/G13 两 Major）：G12 H1"构造性逐位"与 DD10 自认的
  split 1-ULP 风险自相矛盾（split 层 N=4/T≥128 必然激活→近平票
  argmax 翻转→H1 假阳性 block）；G13 DD9 虚构路径
  `actfold/config_manager.py`（实际 `utils/config_manager.py`）+
  遗漏 DD1 主落点 `architecture_utils.py`。另 Minor×5（G8 占位值
  流出对 l2/tau=0 不闭合→构造性强制/G10 B 侧开关未钉死/G11 单链
  提前结束语义缺/G15 章节压缩未注记/G9 措辞）。
- 修复 7 项 → rev 2 → 第 2 轮重审（同代理续审
  ses_eb84f414fffefXCz2O1eIE44k6）**PASS**：7 项修复全 VERIFIED
  （split_layers=False 构造定案经 `_wrap_layer` :562-574 结构性
  证实 fast/slow path 均无 split 依赖；G12 修复采用"真实基准显式
  split_layers=False 恢复构造性+split 面归合成 S1 分档"两面互补
  方案；H2/交付线拆分；S2/H4 严格"<"从严声明）；代码锚点实测
  全命中零回归；srs↔design FR 映射完整。
- 第 2 轮新发现 Minor×4+Nit×4 当轮修入（rev 3）：DD2 伪代码残句
  （else zeros else）；H1 CUDA 残余风险注记（#41 逐位契约 CPU
  限定措辞——CUDA/bf16 M-shape 差异→meta `h1_blocked_attribution`
  归因字段，方向过严不过松）；跨工件对照 AR023 同表显式映射
  （工件 cross_ref 节）；PlantUML 补构造性强制步骤；§6.1 补
  "含胜者延续两轮"；合成面确定性对拍注记（reps=30 专指真实面）；
  日期字段补 rev 3；本条 tasks 门控留档（补 design 第 1 轮 FAIL
  记录——Minor-4 流程项）。
- 关键定案汇总：OQ1 掩码式（`where(stable_mask,parent_ffn,child_out)`
  行级退化，:617-620 契约依据）；OQ2 `folded_generate_batched`
  多链薄包装（单链提前结束=退出合批余链继续）；FR4 放大器=eager
  put_batch 首接（L·N→L launch，单 parent 同受益）；真实基准
  split_layers=False+H1==1.0 硬门+H4 严格"<"。

### 2026-10-17 T001 Red UT 完成（零实验先行）

- `tests/test_multi_parent_batch.py` 新建：UT-2400..UT-2427 共 28 例，
  覆盖 design §6.1 全矩阵（API×4/装配×6/三分支×4/merge×3/发布×4/
  接线×5/配置×3+冻结锚点）。复用 AR015 测试基建（_build_setup 变体
  `_build_setup_mp` 支持 gate 定制+MFF 开关 kwargs；两 parent 世界
  构造；`_multi_call`/`_per_parent_reference` 对拍纪律——参考侧永远
  只用现行生产 API，Red 失败隔离在新调用侧）。
- Red 状态：26 failed（生产缺失面：`parent_branch_ids` 尾参
  TypeError/多父装配/folded_generate_batched ImportError/
  ActFoldConfig.multi_parent_batch AttributeError/YAML unknown-key
  warning/MFF multi_parent_batch 构造 kwargs TypeError）+2 passed
  （UT-2400 参考基线自洽锚点/UT-2415 `_merge_stable_divergent_torch`
  [N,T] 行级语义零改动回归）。
- 冻结面零破坏：test_batched_branch_forward + dream/llada/
  greedy_alignment 投机面 56 passed。
- 教训：UT-2415 初版 mask 构造笔误（`mask[:] = True` 全量断言后未
  复位即设混合模式——行 2 残留全 True 假失败）；最小复现定位后修
  为 `mask[:] = False` 复位再设混合。函数本身零问题。
- 测试专用基建：`_NoBatchPutCache`（#42 显式 AttributeError 包装
  纪律——无 blanket `__getattr__` 泄漏）/`_DropEntryCache`（隐藏
  单一 (branch, layer) fetch 项——快路径 FFN 前置条件破坏器，
  gate 层 fetch 不受扰）。

### 2026-10-17 T002 Green 机制核心完成

- `folded_transformer.py`：`FoldedTransformerLayer.forward` keyword-only
  尾参 `parent_branch_ids`；`_forward_batched` 多父分发（None=单 parent
  路径逐字节不变）；新 `_forward_batched_multi`（DD2 逐行 fetch
  `_fetch_parent_gate_row`：缺失/dim/H/零长/T!=T_c 降级、batch!=1
  逐行硬错 RuntimeError；DD2 构造性强制 `stable_mask[无效行]=False`
  （gate 后 forced 前，metric 无关不流出）；DD3 快路径条件=全行
  valid+全 stable+逐行父 FFN 可取（`_fetch_parent_ffn_row` 逐行
  [1,T,H]→stack），任一不满足→统一 slow path；DD4
  `_merge_parent_child_multi` 逐行父 FFN stack+缺失行占位
  child_out 行+mask 行 False）。
- `architecture_utils.py`：`ManualFoldedForward.forward` 尾参
  `parent_branch_ids`（互斥/无 branch_ids/长度三类 ValueError）+
  None-attach 零扰动（mp_kwargs 仅非 None 附着）+ DD10 防御（多父+
  use_cuda_graph→warn-once `_multi_parent_graph_warned`+eager 回退，
  graph 分支条件收紧 `parent_branch_ids is None`）。
- 机制面 UT-2400..UT-2415 全绿 16/16；冻结面（AR015/AR023/graph
  批量）64 passed+1 pre-existing fail（UT-2303 posterior-trap：经
  stash 验证与本 AR 改动无关——AR024 并行会话在飞改动
  llada_sampler/speculative_core 所致，非本会话责任）。
- 质量门：black/isort/pyflakes/mypy（68 文件 0 错）全过。
- 调试留档两则：①UT-2412 "全发散"前提在 CUDA RNG 下被 token 碰撞
  破坏（层 0 gate 比较纯 embedding，碰撞位 cosine==1.0 判 stable→
  merge 取父值=正确折叠语义但违反测试前提）——改保证逐位 token
  互异（固定偏移 13+2g+i mod 32）；②UT-2413 drop 层选顶层（内层
  drop 会扰动下行层 gate 稳定性——重算行与父分歧）。
- CUDA 对拍纪律发现：同形同分配的多次运行逐位确定（rows 0/2/3
  exact）；跨形/跨分配误差主要来自 gate 语义而非数值噪声。

### 2026-10-17 T003+T004 发布/循环接线+配置面完成

- **T003 发布面**：`FoldedTransformerLayer.__init__` 加
  `multi_parent_batch: bool = False`（DD5/DD6 单开关，layer 级）；
  `_store_batched` put_batch 化——开关 on+`BatchedPutCapable`
  （#42 结构探针，**每次重探不缓存**）→ 每层一次 put_batch N 行
  `CachePutEntry`（payload 与逐行面逐字节同构：ffn_out 行视图+
  layer0 加 embedding 行视图）；无能力→逐行 put 逐字节回退；
  单 parent 批同受益（FR4 放大器，UT-2419）。
  `SplitFoldedTransformerLayer` ctor 透传；`ManualFoldedForward.__init__`
  加 kwargs → `_wrap_layer` 双分支传参。
- **T003 接线面**：`_batched_forward_supported` 泛化（开关 on：同
  parent 条件放宽为逐行 parent 仅要求非 None；off 逐字节不变）；
  `ChainInput` dataclass + `folded_generate_batched` 多链入口（每链
  独立 tree/active/policy select/acceptance 记账/prune+clear_branch；
  每 step 各链候选收集→全部等长+开关 on→`_run_folded_forward_multi`
  一次 `parent_branch_ids` 调用；否则逐链 folded_generate 语义分发
  ——AR015 单父批 or 串行）；**多链 root id `root-{chain_idx}`
  唯一化**（单链硬编码 "root" 跨链 cache 碰撞）；单链提前结束=
  退出合批余链继续；`_run_folded_forward_multi` 镜像
  `_run_folded_forward_batched`（stack→MFF branch_ids+
  parent_branch_ids→逐行 logits 视图+profile 回填）。
- **关键修正（design FR2 措辞偏离，机制不变方向）**：T002 的
  `_fetch_parent_gate_row`/`_fetch_parent_ffn_row` 按 design rev 3
  "v1 等长语义 T_p==T_c" 实现全部 `T_p != T_c → 降级`——但生成循环
  正典形态是 child=parent+1 token（**父恒短于子一字**），多父路径
  在循环内永不折叠（UT-2420/2423 稳定比 0.0 vs 参考 0.563 暴露）。
  修正为镜像单父 `_forward_batched` :577-604 前缀语义：**父短于子
  →前缀捐赠**（gate 比较前缀+后缀构造性发散；FFN merge 补零对齐，
  `where` 永不读补零位）；**父长于子→仍降级**（UT-2409 钉死的
  无 AR007 截断面不变）；等长→全画布比较（机制 UT-2401..2415 原
  路径逐字节保留）。gate 组装分档：全 valid+统一长度→一次批量
  gate（统一前缀或全画布，单父面镜像）；混合长度/部分 invalid→
  逐行 gate+逐行 zeros 行（无效行不再对 zeros placeholder 过 gate，
  构造性强制后终态 mask 不变）。design.md DD2/FR2 已补 erratum。
- **测试基建修正两则**：①`_NoBatchPutCache` 删除误加的
  `def put_batch: raise AttributeError`——#42 `runtime_checkable`
  isinstance=hasattr 探针会把它判 True（`_IncapableCacheWrapper`
  先例=完全不定义该方法）；②`_track_cached_branches`
  （test_batched_branch_forward.py 共享基建）扩 put_batch spy——
  `CachePutEntry` 逐条解包 layer 0 branch_id，开关 off 调用方
  seen 列表不变（冻结面 68 passed 验证）。
- **T004 配置面**：`ActFoldConfig.multi_parent_batch: bool = False`
  （utils/config_manager.py，consumption+operational 双面——MFF ctor
  消费）；known-key 过滤随 `__dataclass_fields__` 自动生效（UT-2426
  unknown-key 零警告）。
- 验证：AR025 模块 28/28 passed（本机 cuda:0 device fixture）；
  冻结面回归：test_batched_branch_forward+test_batched_put 68
  passed / generation+config+arch+publish_cost 75 passed /
  graph+adaptive+pe+perf_proxy 53 passed+1 skipped+1 failed
  （BS007 类性能单测 `test_bs007_graph_replay_not_slower` 在
  AR023 llada 基准（PID 8292）GPU/CPU 竞争负载下失败——AR023
  tasks :109 已实录同型噪声，本 AR 未触碰 graph 路径，GPU 空闲后
  T007 全量门复验）/ split_layer 52 passed。
- 质量门：black/isort/pyflakes 过；mypy 系统 python 68 文件 0 错
  （ar019 venv 下 generic.py/causal_lm.py 各 1
  `no-untyped-call`——venv transformers 4.53.1 桩差异，两文件均非
  本会话改动，系统 python 0 错交叉证实环境性）。
- 提交纪律：只 stage 本会话 7 文件（folded_transformer/split_layer/
  architecture_utils/folded_generation/config_manager/
  test_multi_parent_batch/test_batched_branch_forward+tasks.md），
  AR024 并行会话在飞文件（llada_sampler/speculative_core/
  llada_8b.yaml/test_llada_granularity/ar024_granularity_bench/
  AR024 tasks.md）不碰。

### 2026-10-17 T005 合成基准脚本完成（零实验先行，冒烟除外）

- `scripts/ar025_multi_parent_bench.py`（DD8 名）：`--mode synth`
  三方 A/B/C——A=合批（开关 on，一次 `[ΣN,T]` per-row
  `parent_branch_ids`+put_batch 发布）/B=分批（开关 off，每链一次
  `[N,T]` 单父批量+逐行 put，pre-AR025 现状）/C=串行（逐候选，
  账本 report-only 上下文）；8 点网格（链 {2,4}×N {4,8}×T
  {256,1024}）×双构造（nonsplit+split——真实面 pin
  split_layers=False，split 1-ULP 风险面归本面 allclose(1e-6)
  分档完整覆盖，G12 两面互补定案）。
- 判定式冻结入脚本（先于数据）：S1 逐点 A vs B 对拍（nonsplit
  `torch.equal`/split `torch.allclose(rtol=atol=1e-6)`，任一 FAIL
  ⇒ 全 verdict blocked）；S2 `ops(A)<ops(B)` 严格（write_op_count
  #42 语义，相等=零收益 blocked 从严）；S3 缩减比 report-only。
  确定性单发无计时（计时统计=真实面 T006 专有）；默认 CPU（#41
  逐位措辞 CPU 限定；CUDA 载 M-shape 数值边界风险注记入 meta，
  blocked 即诚实结论方向从严不过松）；seed 冻结算式（1000·chains
  +10·N+T//256+5·split，永不 Python hash）。
- 合成模型：AR025 测试夹具族（cumsum 前缀稳定扫描+
  post_attention_layernorm→mlp 可探测链+embedding/layers/head
  自动发现布局）；中位单 token 替换构造（前缀逐位同父+cumsum
  下行传播分歧=确定性混合 mask，循环零 RNG）；三侧同 seed 同权重
  独立 cache/gate（互不可见）。
- `--quick` 冒烟（CPU 单点 (2,4,256) 双构造）：S1 双构造过
  （bitwise/allclose）、S2 过、ops A/B/C=4/40/40（A=每层一次
  put_batch 4 层=4；B=C=8 行×(层0 记 2+他层各 1)=40——账本
  区分发布粒度，B/C 同为逐行面）、S3 ratio 0.9；独立
  `_quick` 工件（AR023 先例，正式 8 点面随 GPU 空闲与 T006
  同窗执行）。
- `--mode real` 诚实脚手架：parser.error 指向 T006（需 AR023
  T006 工件；reps=30 预承诺单次），零伪数字。
- lint：black（重排）/isort/pyflakes 过。

**正式 8 点面执行+erratum（2026-10-17，AR023 llada 基准完成后
CPU 执行）**：

- 首跑：5/8 点 S1 FAIL（T=1024 全档+(4,8,256)，双构造同败）→
  冻结判定式落 blocked（S1/S2/overall 全 blocked，理由字段
  逐点列出；工件保全
  `results/optimization/ar025_multi_parent_bench_eviction_blocked.json`）。
- 根因诊断（临时诊断脚本，非判定运行）：①gate 仪器化两侧
  对拍——A 侧整次 forward 仅 **1 次** gate 调用（L0），B 侧
  逐层 8 次（mask 语义一致：每行仅中位 512 divergent）；②
  逐层 fetch 追踪——A 侧 L1 起 `_fetch_parent_gate_row` 全
  None 但缓存 dump 明明有 pa layer-0 ffn_out；③聚合模式：
  FAIL ⟺ A 侧 layer-0 token 行数 `(chains+chains·N)·T>8192`
  （原 fixture 预算）——`_evict_over_budget` 按 token 行数
  LRU 逐出，A 侧 L0 burst put_batch 写 ΣN 子条目时 parents
  虽刚被 fetch 触碰但先于子条目写入（LRU 较旧）→ put_batch
  进行中即被逐出 → L1+ 取父 miss → 整批重算（**值正确、
  深层折叠丢失**）；B 侧逐链交错调用每层 fetch 刷新 parents
  LRU 而幸存——**降级不对称**，A vs B 在 513+ 位结构性分歧
  （max 0.0039），512 位（替换位）两侧同重算一致。④中间假设
  逐一排除：跨形状 GEMM ULP（实测同输入 M=8T vs 4T 逐位同）、
  tau 边界翻转（cos[513+]≈1-2e-8 远离 0.95，无近-tau 位）。
- 诊断验证：三个失败点预算放大 65536 后 S1 双构造全恢复+账本
  面不变（4/40..4/160）。
- DD8 erratum（design rev 4 同批）：fixture 预算按网格最坏
  layer 人口 (4+32)·1024=36864 行取 65536
  （`_CACHE_BUDGET_TOKENS`，meta 新增
  `cache_budget_tokens_per_layer`+erratum 注记+docstring 注记）；
  **冻结判定式 S1/S2/S3 零改动**。
- 重跑（一次性）：**16/16 全过——S1 supported（nonsplit
  逐位/split allclose 1e-6）、S2 supported（严格）、S3
  0.9/0.95/0.975 报告值**；工件
  `results/optimization/ar025_multi_parent_bench.json`
  （`verdicts_declared`+`synthetic_verdicts` 双节，blocked
  证据工件独立保全供溯源）。
- 机制级逐出边界入 DD10（触发条件/真实面几何不触发结论/
  潜在缓解注记——延迟发布或 parent pinning，超本 AR 范围）。


## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。

### 2026-10-16 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：13 项（G1-G9 + 证据核对 N1-N4）
- 修复的问题：Minor×8 当轮修入（G1 伪引文/G4 multi-geometry 缺承诺/G6 降级触发器未预注册/G7 容差分档+基线口径+账本代理语义/G8 质量门口径未钉死/G9 模板偏离+AR 编号冲突）
- 审查代理：sdd-gate-reviewer（ses_eb8979327ffeDKNbfXYBzcD1b9）；证据核对 N1-N4 全 YES（`_batched_forward_supported` :310-334 精确命中五原子检查/#41 原文逐点吻合/#42 put_batch 三点核实/folded_transformer.py :498-514-522-536 单键 expand 广播无 per-row gather+AR020 design §4.3 方案 C 否决理由旁证读路径从未批量化）
- 结论：需求阶段门控通过，可进入 design 阶段。

### 2026-10-17 design 门控记录

- 门控结果：PASS（第 2 轮；第 1 轮 FAIL）
- 第 1 轮（ses_eb85d707bffeut2qVgEtMzVCIm）：FAIL——G12 Major
  （H1 构造性逐位 vs DD10 split 1-ULP 自相矛盾）+G13 Major
  （DD9 虚构路径 utils/config_manager+遗漏 architecture_utils.py
  主落点）+Minor×5（G8/G9/G10/G11/G15）；修复 7 项→rev 2。
- 第 2 轮（ses_eb84f414fffefXCz2O1eIE44k6）：PASS——7 项修复全
  VERIFIED（split_layers=False 构造经 `_wrap_layer` :562-574
  结构性证实；三个 architecture_utils 锚点 :585/:759-771/:1510-1575
  实测命中；构造性强制/开关钉死/提前结束语义/压缩注记全核实）；
  第 1 轮 OK 项零回归；新发现 Minor×4+Nit×4 当轮修入→rev 3。
- 结论：设计阶段门控通过，可进入开发阶段（T001 Red UT）。
