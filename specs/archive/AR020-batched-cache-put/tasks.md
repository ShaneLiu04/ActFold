# [AR020] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR020 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-14 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | Red：`tests/test_batched_put.py`（UT-2001~2009：三 cache 等价性/原子性/环形降级/能力协议/发布消费分流/主机调用账本/零 host-sync/新鲜度/**旧 torch 无 `_foreach_copy_` 兼容性降级面**），子代理编写，红因全=生产缺失 | - | passing | 1890 行（Red 期；Green 修正 #1 后 1896 行）/21 函数/56 用例；Red 55 failed（协议 ImportError/cache AttributeError/发布分流缺失 spy 0 calls/账本探针缺失/batched_write 模块缺失）+1 passed（UT-2006② 无能力回退=现状 loop，双面不变量先例）；pyflakes 零/≤100 列 |
| T002 | Green：三 cache `put_batch` + `BatchedPutCapable` 协议 + 发布路径能力探测分流；mutation 全红还原；全量回归 | T001 | passing | 56/56 绿；全量 920 passed/3 skipped（864 基线零扰动）；pyflakes 零/mypy clean/新增行 ≤100 列；mutation 13/13 全红后 SHA256 字节级还原；3 项 Green 期修正见进度记录 |
| T003 | 基准脚本 `scripts/ar020_publish_put_bench.py`（发布窗口 A/B + 端到端三路径 + 主机调用账本 + 正确性门）+ `--quick` 冒烟 | T002 | passing | 998 行；pyflakes 零/≤100 列；--quick 冒烟一次通过（(256,4)：put-micro 3.76→2.52ms 账本 116→1；发布窗口 7.56→4.40ms H2=supported；e2e 四路径 no_fallback=True；H1/H2/H4 supported、H3 inconclusive（quick reps=3 σ 大，预期——正式判定在 T004 reps=30）） |
| T004 | 工件单次定案：reps=30 预承诺运行，`results/optimization/ar020_publish_put_bench.json`（H1-H4） | T003 | passing | 单次定案：H1/H2/H3/H4 全 supported，零失败配置；发布窗口 p50 三配置 7.894→4.393 / 16.809→9.510 / 8.708→4.425ms；账本 116/232/116→1；正确性门 116/232/116 张量全 bit-equal；e2e 最小配置严格改善 3.900ms > 2σ=3.712ms；no_fallback 38/38；峰值 VRAM 10.0GB |
| T005 | 质量门：全量 pytest / mypy / pyflakes / 100 列 / demo 锚点 | T004 | passing | 920 passed/3 skipped（零既有失败）；mypy 67 文件 clean；pyflakes 零（actfold/tests/demo/scripts）；新增行 ≤100 列（唯一超长为 activation_cache.py:54 既有行）；demo 三锚点 85.5%/2.35e-03/93.75% 精确不变 |
| T006 | 文档收口五件：GUIDE 第二十七部分 / CHANGELOG / AGENTS.md 新条目 / RERUN_CHECKLIST / README 全量中文重写 | T005 | passing | 五件全交付：GUIDE §27（表格+三层实测表+H1-H4 判定+语义要点六条+遗留）；CHANGELOG AR020 英文条目（三门控摘要+Green 三修正+13 mutation+四假设实测）；AGENTS #42（put_batch 能力契约）；RERUN 第 10 步（命令+工件+对照数字）；README 全量重写（创新表 10 行+机理详解 §9+头条 AR020 表+谱系链+推荐配置+特性行+架构图+质量 920/67+文档 27 AR+路线图六档收官重写，数字全部自工件重算） |
| T007 | review 门控（只读子代理：spec/design 合规 + 代码审查），全部问题（含 Minor）修复 | T006 | passing | G1-G8 全 PASS（无 Major）；4 Minor + 2 Nit 全部修复并经同会话复验 VERIFIED（gate CLOSED）：Minor-1 行数 1890→1896（GUIDE/CHANGELOG，tasks T001 行注记保留 Red 记录）；Minor-2 README:460 "26 个 AR"→27；Minor-3 GUIDE §27 UT 映射更正（2001=ActivationCache 等价/2002=Vectorized/2003=Chunked/2005 冲刷写入序/2007 分组账本/2008 新鲜度+debug）；Minor-4 design.md §4.2.2 顶部带日期勘误块 + 四处行内注记（每层一次终态修剪被 UT-2001 否证→逐条驱逐；step_idx 透传被 Green 修正 #1 否决）；Nit-5 Δp50/2σ 全精度化（3.500/7.299/4.282 与 2.380/3.465/2.315，GUIDE/README/CHANGELOG 六处，H2 判定不变）；Nit-6 design §4.4/§5 "put 体零改动"改"仅新增账本一行（语义零改动）"并限定范围 |
| T008 | ST 验收：st-cases.md 生成 + 执行（正常/边界/异常/回归四类），一次成 Go | T007 | passing | 19 用例（正常 11/边界 3/异常 2/回归 3）一次全 PASS 零缺陷；需求覆盖 5/5（FR1-FR4+NFR）100%；工件级断言 ST-009（H1-H4 全 supported/正确性门 116/232/116 bit-equal/meta 冻结）与 ST-010（no_fallback 38/38、pe canary 零失配、预算零违例、VRAM 5714/8901/10027MB）；全量 920 passed/3 skipped（46.60s）；pyflakes 零/mypy 67 文件 clean/demo 三锚点 85.5%/2.35e-03/93.75% 精确；结论 **Go** |
| T009 | 归档 `specs/archive/AR020-batched-cache-put/`，本地 commit + gitee commit-tree 推送 + 双端树验证 | T008 | passing | 归档四件（srs/design/st-cases/tasks，AR019 先例）；本地 commit + gitee commit-tree 推送 + fetch 双端树 SHA 一致 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-14 T002 Green 完成记录

**交付**：`cache_protocol.py` +`CachePutEntry`/`BatchedPutCapable`；新模块 `batched_write.py`（`validate_entries` 前置校验/`copy_pairs` (device,dtype) 分组 foreach/`_FOREACH_AVAILABLE` 模块 flag）；三 cache `+put_batch`+`write_op_count`（put 体仅加账本一行；chunked 抽 `_store_chunks` 共用）；`architecture_utils.py` `_publish_graph_children` 每次发布 isinstance 探测分流（零实例缓存）。

**验证**：`tests/test_batched_put.py` 56/56 绿；全量 pytest 920 passed/3 skipped（= 864 基线 + 56 新增，零既有失败）；pyflakes 零；mypy `--ignore-missing-imports` clean；新增行 ≤100 列（activation_cache.py:54 为既有 101 列行，非本 AR 新增）。

**mutation 测试**：13 个变异全部红后字节级还原（SHA256 校验 6 文件一致，还原后 56/56 复绿）——M1 foreach 降级恒走逐对/M2 账本返回 0/M3 校验序号 off-by-one/M4 空激活校验删除/M5 冲刷删除（写入序倒置）/6 put 账本 +=1/M7 put_batch 逐条驱逐删除/M8 vectorized _counts 清零/M9 chunked 账本虚计/M10 发布探测禁用/M11 发布条目改静态缓冲别名/M12 cache 换装传播删除/M13 协议字段序错乱。

**Green 期修正 3 项**（Red 测试冻结面内最小改动，理由链如下）：

1. **测试缺陷修正（test_batched_put.py:1388，`entry.step_idx == 1` → `== 0`）**：该断言与同函数自身文档串（"the branch/layer/step fields of the loop variant"）、文件头契约（"per-entry payload is field-identical to the loop variant"）及同文件三处 step-0 fetch 面（UT-2006② Red 即绿的 loop 回退、UT-2006③/UT-2008 的 `fetch` 默认 step）矛盾。代码事实：折叠栈全链路 step-0 存储（eager 层 `folded_transformer.py:907` put 不传 step、串行/批量两图发布同、`_parent_cache_complete*` 全部 step-0 fetch；跨步链接靠 #26 每 step 新 branch id）。若按 design §4.2.2(4) 草图把发布 step_idx 穿入条目，发布子代将落入 step≠0 键，所有 step-0 消费者（父校验/后续 serial 父 fetch）全部失配 → 批量图发布后静默 eager 回退（折叠静默失效）+ A/B 两变体缓存状态不等（违反 srs FR2/H1）。生产实现取 loop 镜像语义（条目省略 step_idx，NamedTuple 默认 0）。
2. **设计修正：ActivationCache put_batch 驱逐改逐条**（design §4.2.2(1)"每层一次终态修剪"论证被 UT-2001 随机化性质测试实证驳倒）：中途被驱逐又重建的 key，逐条路径的重建 store 仅含重建条目的激活名（名集重置），终态修剪路径保留旧名 → 名集发散（`{'a'} vs {'a','b'}` 实测红）。改为与 put 完全相同的逐条 `_evict_over_budget` 触发点（`_layer_tokens` 只读 count/capacity，不依赖缓冲内容，驱逐决策点两侧恒同 → 等价性由构造保证）；被驱逐组的 pending 对冲刷落死缓冲、不可见，与逐条路径的已覆写行同归。Vectorized（`_evict_old_branch_steps` 经 `_ensure_layer` 每条目触发）与 Chunked（逐 chunk `popitem`）本就逐条，无需改。
3. **新增生产面：`ManualFoldedForward.cache` 改 property + setter 传播**（design §4.2.2(4)/(5) 未列）：wrapped layers 与两 runner 各自钉住构造期 cache 引用，裸 `mff.cache = X` 换装后 eager 重发布的父条目仍写旧 cache → `_parent_cache_complete_batched`（读 `self.cache`）失配 → 批量图路径静默 eager 回退（UT-2006④ 实测红：spy_b 0 次 put_batch）。setter 传播至 `_wrapped_layers[i].cache` 与 `graph_runner(_batched)._cache`（runner 仅 replay 期父 fetch 用 cache，capture 体不触 cache——#37），换装语义完整闭环；既有 UT-1705 换装-恢复模式字节兼容（property 经 nn.Module.__setattr__ → object.__setattr__ 正确路由，实测通过）。


## 阶段门控记录

### 2026-10-14 req 门控记录（第 1 轮）

- 门控结果：FAIL
- 审查项数：8 项（G1-G8）
- 失败项：G1（Minor：M-1 代际溯源错标——发布占比首测实为 AR018 探针、账本表述归位 AR018；M-2 N=16 例证超出工件矩阵）、G3（Major：M-3 发布窗口计时口径与 AR019 工件 meta/GUIDE §23 "CUDA-event 排除宿主侧成本"未调和；M-4 等价性与原子性对非法序列措辞矛盾；N-4 dtype 混批分组未明示）、G4（Minor：M-5 H3 判定式三处不闭口+H2/H3 三态分界未定义；N-1 H4 常数 c 预承诺注记）
- 其余：G2/G8 PASS；G5/G6/G7 PASS（N-2 基线写全 3 skipped；N-3 T001 补兼容性降级面）
- 修复：全部 9 项（M-1~M-5、N-1~N-4）已修入 srs.md/tasks.md——计时口径改宿主 perf_counter+sync 夹逼主报、等价性限定合法序列且原子性单列为更强保证、H1-H4 判定式全部闭口（三态映射+聚合规则+c 预承诺冻结）、代际溯源与例证数字更正、dtype 按 (device,dtype) 分组、T001 增旧 torch 降级面
- 审查代理：ses_ec6b2b666ffepCCNhpeJ16qQlZ

### 2026-10-14 req 门控记录（第 2 轮，复审）

- 门控结果：PASS
- 审查项数：M-1~M-5 / N-1~N-4 逐项复核 + 新问题扫描
- 修复确认：全部 9 项正确修复，未引入新代际错标/判定矛盾/口径不一致
- 遗留：N-5（Nit，T001 测试面 9 个 vs UT 编号 8 个）——已当场修正为 UT-2001~2009
- 审查代理：ses_ec6b2b666ffepCCNhpeJ16qQlZ（复用第 1 轮会话）

### 2026-10-14 design 门控记录（第 1 轮）

- 门控结果：FAIL
- 审查项数：8 项（G1-G8）
- 失败项：G2（Major-2 去重键未定义+foreach 组内顺序未文档化依赖；Major-3 等价比较面不精确——驱逐重建场景 buffer 形状/未初始化行致 UT-2001 假红/漏测双向风险）、G4（Major-1 探测实例缓存与基准换 cache A/B 自相矛盾）、G5/G7（Major-1 波及 H2 有效性；Minor-2 A/B 包装不对称；Minor-3 AR019 导入顺序契约未述）、G8（条目构建开销/导入副作用/foreach 顺序依赖未识别）
- 其余：G1 PASS（Minor-1 H4 测量对象注记）、G3 PASS、G6 PASS（Minor-4 UT-2009 patch 对象为模块级 flag）
- 修复（全部 16 项当场修入 design.md/srs.md/tasks.md）：Major-1 探测改每次发布 isinstance（去实例缓存，基准换 cache 必然重探）；Major-2 重复目的地改"仅首现进 foreach 延迟列表、后续立即逐对按条目序执行"（消除 foreach 顺序依赖）；Major-3 等价比较面精确定义为 buffer[:最终 count]+_counts+LRU 键序+num_entries（fetch 伴生断言）；Minor-1 srs H4 注记测量对象=ActivationCache；Minor-2 A/B 双侧对称包装；Minor-3 导入顺序纪律；Minor-4 UT-2009 patch `_FOREACH_AVAILABLE`；Nit-1~6（流程图措辞/grow 时序写死/账本口径注记/srs 降级措辞机制定案/tasks 元数据/微基准计时区）
- 审查代理：ses_ec6918c26ffe7kccQ9f9AuJ1CA

### 2026-10-14 design 门控记录（第 2 轮，复审）

- 门控结果：FAIL
- 修复确认：Major-3/Minor-1~4/Nit-1~6 共 12 项全部正确修复
- 新发现：NP-1（Major，"首现延迟+重复立即"规范倒置写入序——首现对反而在 copy_pairs 阶段最后落地，与比较面矛盾且 UT-2001/2005 必然红）；Major-1 残留（§3 功能点 4 一处"实例级探测缓存"未同步）；NP-2（时序图 BW 框"/环形"措辞）；NP-3（UT-2004 比较面叠用冗余）
- 修复（4 项当场修入）：NP-1 改**冲刷语义**（重复目的地第二次出现时先冲刷已收集首现对、随后执行当前对、该目的地自此全部立即执行——写入序恒=条目序；跨目的地写入互不相交，冲刷不破坏其余顺序；发布用例目的地恒唯一零触发）；§3 功能点 4 改"每次发布能力探测（零实例缓存）"；NP-2/NP-3 措辞修正；UT-2005 补冲刷序断言
- 审查代理：ses_ec6918c26ffe7kccQ9f9AuJ1CA（复用第 1 轮会话）

### 2026-10-14 design 门控记录（第 3 轮，定向复核）

- 门控结果：PASS
- 复核确认：冲刷语义三处一致（4.2.2/流程图/UT-2005）+ 反例推演闭合（e1(D,10)/e2(D,5)：冲刷后写入序=条目序，终态=逐条路径）；§3 功能点 4 / 时序图 / 4.2.2(4) / 4.4 四处一致（每次发布探测、零实例缓存）；NP-2/NP-3 修正
- 遗留：Nit-O1（重复目的地中途 grow 时冲刷对 dst 指向被弃 buffer——等价性仍成立，超集覆写论证）——已当场落字 4.2.2 冲刷规则处
- 审查代理：ses_ec6918c26ffe7kccQ9f9AuJ1CA（复用第 1 轮会话）

### 2026-10-14 T003 基准脚本完成记录

- 交付 `scripts/ar020_publish_put_bench.py`（998 行）：三层测量（put 微基准条目构建计时区外/发布窗口微基准——单次 replay 不计时、`_publish_graph_children` 单独计时（脚本层私有面探测，AR018/19 先例）、A/B 经 `mff.cache` 交换对称包装实现（`_NoBatchCache` 显式对 `put_batch` 抛 AttributeError 阻断能力探测泄漏，`_BatchCache` 直 delegation）；端到端四路径 baseline/fixed/adaptive(loop)/adaptive(batch) round-robin）；正确性门（同一 replay 后两变体各发布到独立 check 分支，逐 (branch,layer,name) fetch `torch.equal`）；工件 schema 假设-判定分离 + 判定式全文冻结（含 c=2、最小配置 (256,4)、σ=max 口径、H2/H3 最保守聚合、UT 面溯源注记、#39 CUPTI null 注记、`_FOREACH_AVAILABLE` 探针）。
- 复用 `scripts.ar019_real_e2e_bench`（模型加载/工作负载/MFF 构建/计时/聚合，导入顺序纪律：ar019 先于一切 torch/transformers 导入——其模块顶 apply_hf_env）。
- 质量：pyflakes 零、新增行 ≤100 列（一处 pyflakes 4.x 行为：`del` 掉的闭包引用名会被误报 undefined——`del` 语句改为返回即释放 + 注记，非语义变更）。
- `--quick` 冒烟（AR019 venv，(256,4)，reps=3）一次通过：put-micro loop 3.761ms vs batch 2.524ms（账本 116→1，与 L·N+N=28×4+4 精确一致）；发布窗口 7.557→4.396ms（σ=0.459ms，H2=supported，预算零超限）；正确性门 bit_equal=True；e2e baseline 716.8 / fixed 423.8 / adaptive_loop 305.5 / adaptive_batch 299.6ms，no_fallback=True（put_batch 发布计数=期望步数、pe canary 零失配、稳态零预算违规）；H1/H2/H4 supported、H3 inconclusive（quick reps=3 σ 大导致严格改善未达 2σ——预期，正式判定在 T004 reps=30 预承诺单次运行）。

### 2026-10-14 T007 review 门控记录（第 1 轮 + 修复复验）

- 门控结果：PASS（G1-G8 全 PASS，无 Major；关门条件=4 Minor + 2 Nit 全修）
- 审查方式：只读子代理全量读 spec 三件/生产代码/测试/基准脚本/工件 JSON，工件引用数字全部自 `results/optimization/ar020_publish_put_bench.json` 全精度重算（含内部一致性交叉核：loop_put_calls 4508/8988 = 40×112+56 / 40×224+56 ✓、L·N+N=116/232/116 ✓、batch_put_batch_calls 40 = 1 setup+38 timed+1 gate ✓、e2e canary 4284/8540 ✓）；只读算术经 `python -X utf8 -c`
- 通过项要点：G1 FR1-FR3 全合规（协议/copy_pairs 单点/探测分流/基准三层）；G2 三项 Green 修正全部技术成立且在 Red 冻结面内；G3 spec 三件完整性（推荐 erratum→Minor-4）；G4 1896 行 56 用例覆盖 srs §6.1 矩阵 + 13 mutation 全红还原；G5 H1-H4 全 supported 零失败配置、发布数字全部可溯源；G6 920/3、mypy/pyflakes clean、demo 锚点精确（activation_cache.py:54 超 100 列为既有行）；G7 五件文档全交付且 AGENTS #42 逐条与代码核对一致；G8 变更面精确匹配 srs §6.1、无能力回退逐字节保持、#41 串行路径不受扰、账本排除等价面无循环论证
- 问题清单与修复（4 Minor + 2 Nit 全部主代理自修，保持门独立性）：
  - Minor-1 行数 1890 过时（Green 修正 #1 后实际 1896）→ GUIDE:671/CHANGELOG:15 改 1896；tasks.md T001 行改"1890 行（Red 期；Green 修正 #1 后 1896 行）"注记式（不静默改写 Red 记录）
  - Minor-2 README:460 "全部 26 个 AR" 与 :774 自相矛盾 → 改 27
  - Minor-3 GUIDE:671 UT 编号映射错位（原写 2002=冲刷语义/2003=foreach 分组账本；实际 2002=Vectorized 等价、2003=Chunked 等价）→ 全行重写为正确映射并补 2005 冲刷写入序/2007 (device,dtype) 分组/2008 新鲜度+debug record
  - Minor-4 design.md 含被自家 Red/Green 过程否证的断言无就地注记 → §4.2.2 顶部加带日期勘误块 + :31 流程图/:102 活动图/§4.2.2(1)/§4.2.2(4) 四处行内勘误注记（保留原文，注记式勘误实践）
  - Nit-5 Δp50/2σ 发表值由四舍五入表值派生（真值 3.5004/7.2992/4.2822 与 2.3799/3.4648/2.3154，最大偏差 0.001ms，判定不变）→ GUIDE 表列+H2 行/README 表+H2 行/CHANGELOG H2 行共六处按全精度重印
  - Nit-6 design.md "put 体零改动"过claim（实际各加账本一行）→ §4.4 两行改"put 体仅新增账本一行，语义零改动"；§5 改限定范围表述（loop 分支逐字节不变）
- 复验（同子代理会话第 2 轮，只读）：4 Minor + 2 Nit 逐项 VERIFIED；新不一致扫描零残留（stale-number sweep 唯一命中=tasks.md:14 有意保留的注记式 Red 记录）；GUIDE §27 表格结构完好；design.md 324→342 行、内部交叉引用全部可解析；判定 **gate CLOSED — PASS**
- 审查代理：ses_ec5c10f8effeyxTlSQyR0urG88（两轮，均只读）



