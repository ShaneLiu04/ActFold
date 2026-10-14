# [AR021] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR021 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-14 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | Red：`tests/test_generation_loop_folding.py`（UT-2101~210x：FR1 pe/块因果+pad 排除掩码钉定与 raw 零扰动、FR2 强制散度活跃块+形状键控 runner 池+链折叠 parity+默认零扰动、FR3 批量模式起草/验证/选定/停止语义/默认关零扰动、账本与 no-fallback 面），子代理编写，红因全=生产缺失 | - | passing | 1311 行/20 测试函数，20 failed/0 passed，红因全=生产缺失（_fold_ctx_for/forced_divergent/graph_runner_pool_size/num_block_candidates 等不存在）；pyflakes 零、black 100 列过；CUDA 4 条实跑非 skip；子代理 ses_ec4e50392ffeC6Z3eFejOpK3bd |
| T002 | Green：`DiffusionSampler._forward` pe 透传 + 块因果+pad 排除掩码钉定、sampler 折叠路径接线、MFF 强制散度标记面 + 形状键控 runner 池（串行/批量）、`num_block_candidates` 批量投机模式；mutation 全红还原；全量回归。**退路**：若 FR3（批量投机）设计期判风险面过大，本任务拆出为后续 AR，基准以 baseline/folded-serial 两模式先行（srs §2 耦合依据） | T001 | passing | 7 生产文件 +1007/−178；折叠套件 20/20（含 UT-2101 设备卫修，见偏差）；全量回归 **940 passed/3 skipped** 零既有失败；mutation 17 个 14 杀（3 存活=等价/防御纵深：M2 fused stale count 无 UT 探针、M13 串行 forced 身份线被 mask 检查前拦、M17 fixture 常数头同分）全还原；pyflakes 零/black 8 文件过/mypy 67 文件 clean；demo 三锚点 85.5%/2.35e-03/93.75% 命中；池键载体定案 `(tokens.shape, mask 对象)`（与语义键双射，见 design §5 实现澄清）；子代理 ses_ec4b9fb35ffeBGJbKBMgWW0XQM |
| T003 | 基准脚本 `scripts/ar021_generation_loop_bench.py`（三模式 generate() 墙钟 + 质量面 + 账本 + canary，复用 ar019 助手）+ `--quick` 冒烟 | T002 | passing | 脚本 ~1460 行自包含；--quick 冒烟通过（三模式/21 测试保持全绿/工件落 quick 路径）：H1 supported **bit-exact**（真实 checkpoint，超规格强度）、H3 supported 3.99×、H4 supported（canary 降级=0/账本 c=4 内/AR017 未触发=中性）、H2 quick 规模 refuted（单窗小形状折叠开销>收益，预期量级，T004 定案）；暴露修复 2 项：pe fp32 dummy 崩 bf16 sdpa→dtype 跟随 embedding weight（+新测试 UT-2102 扩展）、脚本 device bug；子代理 ses_ec434c83dffey6MVvxiYN0Yghc |
| T004 | 工件单次定案：reps=30 预承诺运行，`results/optimization/ar021_generation_loop_bench.json`（H1-H4） | T003 | passing | 单次运行无重跑（timed 段 ~7min，全程 ~10min，VRAM 峰值 ≤5.4GB 无 OOM）；**H1 supported（真实 checkpoint bit-exact：argmax 100%+allclose 全过+bit-parity 1.0）**；H2 inconclusive（Δ=−107.5ms 未越 2σ=265.3 预注册噪声界，两 prompt 数值均慢但如实记录）；**H3 supported（batched vs serial 2.47×/vs baseline 2.38×）**；**H4 supported**（在域窗捕获/重放>0、账本 c=4 界内逐字段、canary 双模式真、AR017 事件 warmup 段 2 次/timed 0=中性）；per_shape_window 池计数逐位对账（结构 eager/捕获/重放=前向数双模式全对）；工件 schema 完备（50,595B）；子代理 ses_ec407f81cffeyEJbKU1SWHSlO3 |
| T005 | 质量门：全量 pytest / mypy / pyflakes / 100 列 / demo 锚点 | T004 | passing | 全量 **941 passed/3 skipped/3 deselected**（AR020 基线 920/3 + 新增 21，零既有失败）；pyflakes actfold+tests+demo+scripts 零；mypy 67 文件 clean；black 100 列 AR021 全部 9 改动/新增文件过（测试文件一处后续卫修引入漂移已格式化，21 passed 保持）；demo 三锚点 85.5%/2.35e-03/93.75% 命中（全树 61 文件 black 漂移为 HEAD 既有，非本 AR 引入） |
| T006 | 文档收口五件：GUIDE 第二十八部分 / CHANGELOG / AGENTS.md 新条目 / RERUN_CHECKLIST / README 全量中文重写 | T005 | passing | 五件落盘：OPTIMIZATION_GUIDE 第二十八部分（+43）/CHANGELOG AR021 条目（+21）/AGENTS **#43** 生产回路折叠契约（B>1/ctx 键控/forced 四路径排序/池契约/批量投机/A5 单发摊销负注记）/RERUN_CHECKLIST AR021 段（+19）/README 全量重写 854→1039 行（创新表 11 行+AR021 深潜 #10+AR021 头表领起+演进叙事保留+941 基线+限制如实 H2 inconclusive/质量 report-only/单发摊销负）；README 头表 17 项数字经 verify 脚本从工件逐项复核全命中，零 AR020 时代残留；子代理 ses_ec3f326adffexxxsZF5hx9H33a |
| T007 | review 门控（只读子代理：spec/design 合规 + 代码审查），全部问题（含 Minor）修复 | T006 | passing | 第 1 轮 **PASS**（审查代理 ses_ec3d6c36effemCETpKMmkv8RFv，六维度 A-F 全 VERIFIED：spec FR1-FR5/H1-H4、design rev5+§7、代码、测试、文档 17+ 项数字逐位核对、偏差台账六项）；0 Major/2 Minor/2 Nit；两 Minor 当轮已修（判别 UT：UT-2104c fused+forced+gate 全 stable 杀 M2、UT-2109c 互异 min fixture 杀 M17，变异红→还原→绿实证，套件 21→23）；Nit-1 ctx LRU 驱逐后同窗可重建池条目（VRAM 浪费非正确性）已入 design §7 注记、Nit-2 untracked 归档提交解决（T009） |
| T008 | ST 验收：st-cases.md 生成 + 执行（正常/边界/异常/回归四类），一次成 Go | T007 | passing | st-cases.md 15 条（N3/B4/E4/R4）**15/15 全 PASS 一次成 Go**：R1 全量 **943 passed/3 skipped**（920+23 零既有失败）；R2 三锚点命中；R3 静态门全过；R4 AR020 面回归 80 passed；工件复算 H1-H4 与判定一致（87 断言）；真实 checkpoint --quick 冒烟过（H1/H3/H4 supported）；E3 拒断无 sync（100 次 0.54ms）；池核算逐位对账+交叉自洽；2 处 ST 用例侧修正（N2 容差 1e-6/B2 预期模型修正，非产品缺陷）；子代理 ses_ec3b39757ffeh80UqI14D9DmE7 |
| T009 | 归档 `specs/archive/AR021-generation-loop-folding/`，本地 commit + gitee commit-tree 推送 + 双端树验证 | T008 | passing | 见归档目录 tasks.md 进度记录 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-15 会话（design 门控 + T001）

- design.md 完成 4 轮门控（FAIL/FAIL/FAIL/PASS，审查代理 ses_ec535f6d3ffegtn4fwkbkiIefF 四轮同一会话）+ PASS 后 Minor-12/Nit-12/13 文档微修订闭合 → rev 5 定稿（详见阶段门控记录）
- T001 Red 完成：`tests/test_generation_loop_folding.py`（1311 行/20 测试函数/UT-2101~2112 全覆盖），20 failed/0 passed 全红、红因全=生产缺失；pyflakes 零、black 100 列过（子代理 ses_ec4e50392ffeC6Z3eFejOpK3bd）
- 测试冻结面注记（T002 须遵守）：`_fold_ctx_for` 命名、`graph_pool_stats` schema、UT-2110 Green 后红因应转为期望 ValueError

### 2026-10-15 会话（T002 Green）

- T002 完成（子代理 ses_ec4b9fb35ffeBGJbKBMgWW0XQM，两轮：首轮空产出复用会话重试后 19/20 并上报 UT-2101 测试设备 bug，主代理裁定方案 1 后续跑收口）：7 生产文件 +1007/−178，折叠套件 20/20，全量 940/3，静态门全过，demo 锚点命中
- 偏差记录：(1) UT-2101 测试设备卫修一行（`causal` CPU ∧ `raw_kv` CUDA 在 `.to(device)` 前，任何实现必炸；修复语义零变化，临时副本预验证 20/20 后落盘）；(2) black 整文件归一含 6 文件少量既有行纯格式（AST 等价）；(3) 两处设计细化反推自冻结 UT（快路径父条目防逐出回写、驱逐时 cuBLAS workspace 回收）——已作为实现澄清追记 design.md §5
- mutation 存活注记：M2（fused stale count 弃用无 UT 探针——record 消费 AND 后掩码使语义等价）、M13（串行 forced 身份检查为 mask 身份后的防御纵深线）、M17（fixture 常数头使候选同分，等价 mutant）——非测试缺口，T007 review 复核

### 2026-10-15 会话（T003 基准脚本）

- T003 完成（子代理 ses_ec434c83dffey6MVvxiYN0Yghc）：`scripts/ar021_generation_loop_bench.py` ~1460 行 + `--quick` 冒烟通过（真实 checkpoint，单 prompt/reps=3/2 块，工件 `results/optimization/ar021_generation_loop_bench_quick.json` 未触碰正式路径）
- quick 结果：H1 supported bit-exact（31/31 前向 max_abs_diff=0.0）；H3 supported 3.99×（batched 697.8ms vs serial 2783.2ms）；H4 supported（池计数逐位对账、canary 降级=0、账本 c=4 内）；H2 quick 规模 refuted（109.4ms > 2σ——单窗小形状折叠开销>收益，预期；T004 W=4 正式定案）
- 生产 bug 修复（冒烟暴露）：`_resolve_position_embeddings` fp32 dummy 对 bf16 远程模型 sdpa 崩溃 → dtype 跟随 embedding weight（新测试 `test_ut2102_pe_dtype_follows_model`，折叠套件 20→21）；脚本侧 device bug 修复
- T004 注意事项：VRAM 全量预估 ~8GB 偏紧（OOM 如实报告）；早停行为压缩可比窗（P=2 冻结不调）；folded-batched 质量面 report-only（exact_match=0.0 源于投机提交非贪心不命中 stop token，语义一致属设计行为）

### 2026-10-15 会话（T004 工件定案）

- T004 完成（子代理 ses_ec407f81cffeyEJbKU1SWHSlO3）：全量 reps=30 单次运行（无重跑），工件 `results/optimization/ar021_generation_loop_bench.json`（50,595B）schema 完备
- **判定汇总：H1 supported（bit-exact）/ H2 inconclusive / H3 supported（2.47×）/ H4 supported**
- 墙钟 p50：baseline 2840.7 / folded-serial 2948.3 / folded-batched 1193.9 ms（speedups_p50：serial vs base 0.9635、batched vs serial 2.4695、batched vs base 2.3795）
- 对账：per_shape_window 池计数双模式逐位对账全对；账本 c=4 界内逐字段；canary 双模式真；AR017 重捕获 2 次全落 warmup 段（timed 0 = 中性）
- 注记：Triton merge kernel 本机不可用走 PyTorch 回退（#10 生产支持路径）；CUPTI null 注记如实（#39）；baseline/串行 per-block 锚仅 block0 桶（串行 publish 不携带 step_idx，数值自洽非缺陷）——T006 文档收口须引用

### 2026-10-15 会话（T005 质量门 + T006 文档收口）

- T005 完成（主代理直执）：全量 **941 passed/3 skipped**（920 基线+21 新增零既有失败）；pyflakes 零/mypy 67 文件 clean/black AR021 九文件过（测试文件一处卫修引入漂移已格式化）/demo 三锚点 85.5%/2.35e-03/93.75% 命中
- T006 完成（子代理 ses_ec3f326adffexxxsZF5hx9H33a）：文档五件收口，README 全量中文重写（854→1039 行），头表 17 项数字从工件逐项复核全命中，AGENTS #43 生产回路折叠契约入册

### 2026-10-15 会话（T007 review 门控 + Minor 修复）

- T007 第 1 轮 **PASS**（审查代理 ses_ec3d6c36effemCETpKMmkv8RFv）：A spec 合规（FR1-FR4+验收）/B design 合规（rev5+§7）/C 代码质量/D 测试/E 工件文档一致性（README ≥15 项+GUIDE+CHANGELOG+RERUN+AGENTS #43 逐位核对）/F 偏差台账（六项）全 VERIFIED；0 Major/2 Minor/2 Nit
- Minor-1（M2 变异非严格等价：fused+forced+gate 全 stable 场景 stale 计数可错触整层复用且无判别探针）与 Minor-2（M17 fixture 平局巧合 min↔max 不可杀）当轮修复：新增 UT-2104c（`_recompute_merged` 间谍断言 `[960]`=T−forced 数）与 UT-2109c（`_GradedSpikeHead` 互异 min fixture，winner 全 top1 行 min=0.9113 非地板），M2/M17 变异红→还原→绿实证；套件 21→23 全绿；实现在 T002 会话（ses_ec4b9fb35ffeBGJbKBMgWW0XQM）复用完成
- Nit-1（ctx LRU 32 驱逐后同窗重现会重建池条目——VRAM 浪费非正确性）：注记入 design §7；Nit-2（交付 untracked）：T009 归档提交解决

### 2026-10-15 会话（T008 ST 验收）

- T008 完成（子代理 ses_ec3b39757ffeh80UqI14D9DmE7）：st-cases.md 15 条（N3/B4/E4/R4），**15/15 全 PASS 一次成 Go**——R1 全量 943/3、R2 三锚点、R3 静态门、R4 AR020 回归面 80 passed、工件 H1-H4 复算一致、真实 checkpoint 冒烟过、池核算逐位对账；2 处 ST 用例侧修正（非产品缺陷）
- **AR021 验收结论：Go**（6/6 需求覆盖、五面全绿、零产品缺陷）→ T009 归档

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。

### 2026-10-14 req 门控记录（第 1 轮）

- 门控结果：FAIL
- 审查项数：8 项（G1-G8）
- 失败项：G1（Minor-4 FR4 无验收标准；Nit-4 FR5 缺验收行）、G2（Major-2 图"再捕获"误述既有行为；Minor-8 发布窗口区间 1.80-1.97 过时；Minor-9 "永不捕获"应为"捕获一次后永不重放"）、G3（**Major-1 链折叠 bit-exact 断言被 AR019 P5 否证**（同块未翻位置 tau=0.95 误判 stable，parity 92.9% 先例）；Major-2 波及；Minor-3 掩码缺 pad 排除；Minor-3b 折叠掩码须与底层模型 raw 语义匹配/合成面自洽；Minor-7 FR3 停止 token 语义未定）、G5（Major-1 波及：最大机理风险未列入假设；Minor-6 FR3 捆绑未论证）
- 其余：G4 PASS（Minor-5 H4 自适应事件条件式未闭口；Nit-1 σ 未绑模式名）、G6 PASS、G7 PASS、G8 PASS（Nit-11 缺口表 G1-G4 撞名；Nit-12 自适应容量批量专属注记）
- 修复（全部修入 srs.md rev 2）：Major-1 → §1 机理依据改写为"前缀半边（P3/P4 正面）+ 活跃块半边（P5 负面）"，FR2 引入**强制散度活跃块区**（新语义+自有 UT，构造级绕开 P5），parity 硬门改为"由构造保证"；§5 假设 A1 风险定级入列。Major-2 → FR2 引入**形状键控 graph runner 池**（新语义+LRU 上限+VRAM 守恒面 UT），删除"既有行为零新语义"误述。Minor-3/3b → 掩码式改 `block_causal AND 非pad` + 语义匹配义务 + 合成面自洽条款。Minor-4 → FR4 验收标准补齐（时长界/工件字段完备/单次纪律）。Minor-5 → H4 判定式自适应事件条件式闭口（未触发=中性）。Minor-6 → §2 FR3 耦合依据 + T002 退路备注。Minor-7 → FR3 结束语义（停止 token/finished/pad 填充）。Minor-8 → 1.77-1.97×。Minor-9 → 措辞更正。Nit-1/11/12/13 → σ 绑模式名/缺口表改"缺口-N"/自适应批量专属注记/FR5 验收行
- 审查代理：ses_ec55a8823ffeiPmz2MRtLLRYIC

### 2026-10-14 req 门控记录（第 2 轮，复审）

- 门控结果：PASS
- 审查项数：第 1 轮 Major-1/2 + Minor-3~9 + Nit-10~14 逐项复核 + 新问题扫描
- 修复确认：9+5 项全部 VERIFIED（两 Major 以"强制散度活跃块 + 形状键控池"新语义化解，机理诚实：P5 构造级绕开、#37 零放宽）
- 新发现（4 Minor + 3 Nit，当轮全部修入 srs rev 3）：Minor-1 池激活面未定→opt-in 旋钮默认关（UT-1504 零扰动+NFR 扩展）；Minor-2 AR 步前向散度策略缺→新提交块区强制散度（P5 同式，设计期定案+UT）；Minor-3 FR4 缺 warmup/round-robin→补齐（AR019 先例）；Minor-4 H1 真实面口径过严→AR019 H2 口径（argmax+bf16 allclose 仲裁，bit-parity 报告子字段）；Nit-5 验收 Given/When 补全+时长界措辞；Nit-6 H4 账本 ≤c·(L+N) c 冻结（AR020 同式）；Nit-7 批量池捕获摊销→§5 假设 A5 设计期注记
- 审查代理：ses_ec55a8823ffeiPmz2MRtLLRYIC（复用第 1 轮会话）

### 2026-10-14 design 门控记录（第 1 轮）

- 门控结果：FAIL
- 审查项数：追溯矩阵 FR1-FR5/NFR/H1-H4 + 代码锚点核验 + AGENTS 纪律（#15/#20/#26/#37/#38/#41/#42/#24）
- 失败项：Major-1 eager AND 锚点枚举遗漏串行 fused-gate 路径（:256-272）且 count/profiler 排序未定案（过期计数可整层复用含 forced 位）；Major-2 `_fold_ctx` 纯 shape 键 pad 碰撞（同画布 T 不同 prompt_len → pad 列错误，语义静默错误）；Major-3 池化后 cache setter 未扩展池传播（违 #42）；Major-4 A5 批量池预捕获策略未定案（srs 显式设计期义务）；Major-5 FR4 基准落地 materially 不足（三模式接线/warmup/round-robin/按形状窗计数出口/c 冻结/H1 口径/--quick 全缺）。另 Minor-1~6（forced 捕获身份三处互斥、覆盖区措辞漂移、FR3 链延续未定案、种子公式未冻结、candidate_top_k<1 校验缺、VRAM 量级未估）+ Nit-1~5（config 路径、pe 丢弃归因、raw 措辞、UT-2105 双形态契约、池观测面）
- 修复（全部修入 design.md rev 2）：Major-1 → §4.1.1 四路径全覆盖单一排序规则（物化→AND→record→count；fused 计数弃用走 :316 fallback，不改 Triton kernel）；Major-2 → 键 (T,prompt_len) + pad 区 host 元数据构建 + 病态 prompt 文档化边界；Major-3 → setter 遍历串行+批量池全部 runner；Major-4 → 初判方案 b（FR3 全 eager）；Major-5 → 新增 §4.2.2(6) 基准设计。主代理另自查修正 FR3 eager 批量发布机制（_store_batched 非 put_batch）。Minor/Nit 全修
- 审查代理：ses_ec535f6d3ffegtn4fwkbkiIefF

### 2026-10-14 design 门控记录（第 2 轮，复审）

- 门控结果：FAIL
- 复核：第 1 轮 16 项中 15 项 VERIFIED、Major-4 PARTIAL
- 新发现：Major-7（A5 方案 b 与 srs H4"批量图捕获>0"构造性冲突 + §1.2 折叠栈全开落空 + 摊销论证与自身 warmup+reps 协议自相矛盾 + c=4 推导引用不运行路径）；Major-8（池键 (T,)/(N,T) 与 srs"shape+对象组"键控语义漂移、多 prompt 下 shape-命中-身份失配行为未定案、cap/ctx 联动未定）；Minor-9（§2.1 串行发布标注错误——put_batch 仅批量图发布）；Nit-6（`_folding_parent` 属性名）、Nit-7（meta 缺强制散度/池声明字样）、Nit-8（VRAM 缺 logits_buf）、Nit-9（bench MFF split_layers 未声明）
- 修复（design.md rev 3）：Major-7 → 改选**方案 a**（保留 use_cuda_graph=True + 批量池 + warmup 预捕获，摊销在基准协议下成立；生产单发摊销负注记入 meta；H4/全栈行使恢复）；Major-8 → 池键 (T,prompt_len)/(N,T,prompt_len) 与 ctx 键对齐 + 失配行为定案（异常态静默 eager）+ cap 联动（基准 P=2/W=4 冻结）；Minor-9/Nit-6/7/8/9 全修（c=4 推导按 put_batch 路径重写）
- 审查代理：ses_ec535f6d3ffegtn4fwkbkiIefF（复用会话）

### 2026-10-14 design 门控记录（第 3 轮，复审）

- 门控结果：FAIL
- 复核：Major-7(a)(c)(d) 已修、Major-8/Minor-9/Nit-6~9 全 VERIFIED
- 残留+新发现：Major-7(b)（§2.2"AR017 行使"与 §4.2.2(6)"adaptive 默认关"自相矛盾）；Major-10（no-fallback canary 口径未定义——结构性 eager 恒>0 → canary 恒假 → H4 构造性 refuted）；Major-11（H1 硬门越界至三模式——folded-batched 构造上与 baseline 分叉，结构性必败）；Minor-11（c=4 归一化单位未钉死，folded-serial 子句破界）；Nit-10（folded-batched 串行条目 6 非 8）；Nit-11（末窗/单迭代窗数据依赖未声明）
- 修复（design.md rev 4）：Major-7(b) → MFF_B 启用 graph_adaptive_capacity=True（AR017 接通，事件条件式核验）；Major-10 → canary 二分口径冻结（降级类=全部非结构性回退 vs 结构性 eager 三情形，canary 只查降级类 timed=0）；Major-11 → H1 硬门收窄 folded-serial vs baseline（srs 原文口径），folded-batched 归 H3 纯报告；Minor-11 → 归一化单位钉死每发布事件；Nit-10/11 → 逐模式条目核算 + H4 子判据核验范围声明（≥2 前向窗）
- 审查代理：ses_ec535f6d3ffegtn4fwkbkiIefF（复用会话）

### 2026-10-14 design 门控记录（第 4 轮，复审）

- 门控结果：**PASS**
- 复核：第 3 轮 Major-7(b)/Major-10/Major-11/Minor-11/Nit-10/Nit-11 全部 VERIFIED（代码锚点逐一复核：architecture_utils.py:409 kwarg 既有、:774-776/:832-835 结构性 eager、:793-794 父校验；H1/H4 与 srs 判定式逐字对齐；c=4 每事件口径与 AR020 write_op_count 先例相容）
- 遗留（PASS 后文档微修订，主代理当轮已闭合、审查代理判定免复审）：Minor-12（canary 降级类枚举闭包为"全部非结构性回退"+ 两类 eager 计数器入 graph_pool_stats 接口出口 + timed 段差分快照语义 + schema 补 adaptive 事件/两类计数字段）；Nit-12（条目"各 8"改"≤8"、MFF_B 串行池 6 条目 VRAM 修正、schema 反引号笔误）；Nit-13（§4.2.2(3) AR017 实例级观测状态跨池条目共享注记）
- 结论：design.md（rev 5）定稿——FR1 (T,prompt_len) 键控供给/FR2 四路径 AND+形状键控池/FR3 方案 a 批量投机/FR4 三模式基准（AR017 接通+canary 二分+H1 两模式硬门+c=4 每事件）/FR5 五件；T001 可启动
- 审查代理：ses_ec535f6d3ffegtn4fwkbkiIefF（四轮同一会话）

