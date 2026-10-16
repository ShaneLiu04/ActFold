# T011 归档提交计划（AR023 结档执行单）

> 结档顺序依赖：T006 两族工件落盘 → T007 质量门（全量 pytest + demo）
> → T008 文档回填（README 全量重写等）→ T009 review → T010 ST →
> **本计划执行**。本文档为工作单，归档时随 AR 文件夹一并入
> `specs/archive/AR023-cross-family-speculative/`。

## 1. 提交范围（git status 2026-10-16 清点）

**修改（15 文件）**：AGENTS.md / CHANGELOG.md /
configs/per_model/{dream_7b,llada_8b}.yaml / core/cuda_graph.py /
models/{architecture_utils,diffusion_sampler,dream_sampler,llada_sampler}.py /
utils/config_manager.py / docs/{ALGORITHM,EXPERIMENTS,OPTIMIZATION_GUIDE,
RERUN_CHECKLIST}.md / tests/test_architecture_utils.py

**新增（7 文件）**：models/speculative_core.py /
scripts/{ar023_cross_family_bench,ar023_family_probe}.py /
tests/{test_llada_speculative,test_dream_speculative,test_ar023_probe,
test_ar023_bench}.py

**工件（新增，4 件）**：results/optimization/ar023_family_probe.json +
ar023_llada_bench.json + ar023_dream_bench.json（T006 落盘后）+
**AR023 三件 `_quick` 工件一并提交**——committed 文档（GUIDE §30 实机
冒烟行）引用了 quick 数字，可溯性规则（无无源数字）要求工件入库。
（对照：AR021/022 的 quick 工件无 committed 文档引用，维持不入库先例，
留本地。）

**README.md**：T008 全量重写后入提交。

**spec 归档移动**：`specs/changes/AR023-cross-family-speculative/` →
`specs/archive/AR023-cross-family-speculative/`（命名循
`AR0NN-slug` 惯例，git add -A 捕捉 rename）。随夹文件：srs.md /
design.md / tasks.md / st-cases（T010 产出）/ next_rung_memo.md /
st_recomputation_checklist.md / 本文档。
**删除**：readme_rewrite_skeleton.md（T008 §5 step 6 规定回填后删除）。

**不入库**：ar021/022 quick 工件（维持 untracked 先例）；
D:\ar023_models\ 下的日志/cmd（仓库外）；attempt1 中断日志（留本地）。

## 2. 提交信息

单交付提交（既定模式：每 AR 一次主交付），循仓库风格：

```
AR023: cross-family speculative decoding - shared SpeculativeCore +
LLaDA/Dream family adapters (P1-7 ninth rung)
```

注：本地与 gitee 提交哈希**恒不同**是既定事实（AGENTS README 规则：
identical trees / divergent hashes expected）——树一致性才是验收面。

## 3. gitee 分段推送预案（HIS 代理 POST ≤ ~96KB）

已证事实：docs/OPTIMIZATION_GUIDE.md = **179.5KB**，远超限制——AR022
推送已建立 commit-tree 部分树多段快进链程序（S1..S7 段），本次复用。
代理：`git config http.proxy http://:@proxyhk.huawei.com:8080`（先例）。

分段草案（每段 ≤ ~90KB，段间树单调快进）：
- S1 `actfold/`（models+core+utils+configs，8 文件 ~150KB 若超则拆 S1a models/S1b core+utils）
- S2 `scripts/` 两脚本 ~66KB
- S3 `tests/` 新 4 件 ~160KB → 拆 S3a（llada+probe ~90KB）/ S3b（dream+bench）
- S4 `specs/archive/AR023-.../`（按文件大小拆 2-3 段）
- S5 `docs/`（ALGORITHM/EXPERIMENTS/RERUN_CHECKLIST ~100KB 拆 2 段）
- S6 OPTIMIZATION_GUIDE.md 单段（179.5KB——**超大单文件**：用 git hash-object
  + commit-tree 直接构造 blob 段（AR022 已验证此路径），不走 index POST）
- S7 README.md + AGENTS.md + CHANGELOG.md + results/optimization/ 工件段

执行纪律：每段后 `git ls-tree` 远端核对；全部段完成后
`git diff --stat local-main gitee/main` 必须为空（树一致验收）。

## 4. 验收核对单

- [ ] 两族正式工件存在且 `status == "ok"`（或 blocked 事实链完整）
- [ ] T007 记录：全量 pytest ≥1022/0/4 + demo 三锚点
- [ ] T009 review PASS 记录 / T010 ST Go 记录入 tasks.md
- [ ] README 数字全部工件可溯（骨架 §4 映射表逐槽核对后删除骨架）
- [ ] quick 工件入库后 GUIDE §30 冒烟行数字与工件逐位一致
- [ ] gitee 树一致（diff --stat 空），哈希分歧为预期
