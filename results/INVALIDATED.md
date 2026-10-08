# ⚠️ 数据失效标注（INVALIDATED）

**标注日期**：2026-10-09
**失效范围**：本目录下 `threshold_sensitivity.csv`、`layerwise_folding.csv`、
`cache_size_impact.csv`，以及 `experiments/`、`optimization/` 两个子目录的全部产物
（子目录各有独立标注）。

## 失效原因

这些产物由 AR001 深度优化之前的旧实现与旧方法学生成，AR001 的以下变更使其
**不可再引用**（保留仅作历史参考）：

1. **消融方法学重写（T024）**：三张 CSV 由旧版"合成外推"产生——layerwise 为
   线性缩放推演（非实测）、cache 扫描预算恒大于序列长度（从未发生驱逐，曲线为
   无信息量的平坦线）。新实现为真实折叠测量（逐层实测 stable ratio、
   `disabled_layers` 层消融、随 `seq_len` 缩放并强制真实驱逐的缓存扫描），
   旧数字与新数字口径完全不同。
2. **FLOPs 口径修正（T021/T022）**：embedding 双计数移除（输入 embedding 是查表
   不计 matmul，仅保留 LM head）、FFN 形状参数化（`ffn_intermediate_dim`/
   `ffn_type`）、成本模型加入注意力二次项与 KV 读并设备表化。所有 FLOPs 节省
   百分比整体上移（合成基线 78.5% → 85.5%）。
3. **测量统计修复（T020）**：旧时延数据为单次测量、无 std/无种子记录；新
   `time_forward` 返回 `{mean, std, p50, n}`，`exp_sampling` 强制 repeats≥3 且
   种子受控、含 `tokens_reproducible` 校验。
4. **指标正确性修复（M1）**：跨步稳定率改全步均值、final_norm 修正、相似度计算
   改 fp32。旧相似度/稳定率统计与新口径不可比。
5. **实现路径变更（M2/M3）**：缓存读取路径（fetch/get_all）、合并 kernel、
   逐层 hook 等重写后，旧开销分解（overhead）数值不再描述当前实现。

## 处置

- 本目录产物**不得用于论文、报告或对比基线**。
- 重跑步骤、环境要求与预期资源见 `docs/RERUN_CHECKLIST.md`。
- 重跑产出前请先删除或移走旧产物，避免新旧混用。
