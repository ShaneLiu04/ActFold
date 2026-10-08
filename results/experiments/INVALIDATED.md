# ⚠️ 数据失效标注（INVALIDATED）

**标注日期**：2026-10-09
**失效范围**：本目录全部产物（`*/results.json`、`*/similarity.npz`、
`*/overhead.json`、`figures/*`）——2026-10-03 在原租赁机器（RTX 6000D）上由
AR001 之前的旧代码产生。

## 失效原因

1. **相似度/稳定率口径变更（M1）**：跨步稳定率改为全步均值、final_norm 修正、
   余弦相似度改 fp32 计算。`results.json` 中的相似度统计、τ 扫描、跨步稳定率
   与新实现不可比（旧口径存在系统性偏差）。
2. **FLOPs/成本模型口径变更（T021/T022）**：`results.json` 中的 FLOPs 与成本
   模型实验数字基于旧计数器（embedding 双计数、无 FFN 形状参数、无注意力
   二次项），全部需按新口径重算。
3. **开销分解不再描述当前实现（M2/M3）**：`overhead.json` 的 cache get/put、
   gate、merge 耗时测量于旧的逐 token 掩码路径；M2/M3 重写后（连续缓冲、
   fetch/get_all、Triton merge 单边读）这些数字已失真，"缓存读取开销超过
   单层计算"的核心结论对当前实现不成立，必须重测。
4. **无方差/无种子（T020）**：所有时延为单次测量，无 std、无种子记录；
   `exp_sampling` 采样实验缺 repeats/`tokens_reproducible` 校验。

## 处置

- 本目录产物**不得引用**；`docs/DEEP_EXPERIMENT_REPORT.md` 中基于这些产物的
  结论需在重跑后复核。
- 重跑命令（模块形式）与环境要求见 `docs/RERUN_CHECKLIST.md`。
