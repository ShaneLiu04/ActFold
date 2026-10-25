# [AR036] 详细设计 — 提交-草稿融合（Commit-Draft Fusion）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR036 |
| 关联 srs.md | ./srs.md |
| 日期 | 2026-10-25 |
| 状态 | Delivered（T001 实测：P1/P2/P3 全 0.0、P4 crop 可用） |

## 1. 架构位置

AR034 组合臂的调用链：

```
_history_cached_sample
  for block_idx:                     # 每块
    for small_block_idx:             # 组合臂下恒 1 次
      _history_speculative_small_block   # ① 草稿 [1,B]  ② 验证 [N,B]  ③ 写 x
    commit forward [1,B]             # ④ 写 KV + advance token + 追加掩码块
```

AR036 把 ④ 扩展为 `[1,2B]`（④' 融合前向），并把 ① 从下一块中移除：

```
block b:  ①draft [1,B] → ②verify [N,B] → ④'fused commit+draft [1,2B] (KV crop → hist+B)
block b+1: (草稿由 ④' 提供，跳过 ①) → ②verify [N,B] → ④'fused ...
```

每块前向数：3 → 2（末块或剩余不足时退回 3）。

## 2. 配置与契约

| 字段 | 类型 | 默认 | 校验 |
|---|---|---|---|
| `commit_draft_fusion` | bool | `False` | 需 `history_cache` + `num_block_candidates>0`；与 `speculative_commit_gate` 互斥；`ValueError` 否则 |

默认 `False` 时任何代码路径、账本、随机数消耗均与 AR035 末态**逐字节一致**（UT-3601 钉死）。

## 3. 数据流细节

### 3.1 advance token 与下一块画布（融合前）

- 验证前向给出 `vrows[N, B, V]`；row 0（贪心锁定行）的**未移位尾行** `raw[B-1]` 即下一块第 0 位
  （AR032 已验证：提交前向输入与 row 0 输入等价 → 尾行 argmax 相同）。
- `next_canvas = cat([advance, full((1, B-1), mask_id)])`，仅当 `remaining_after >= B + 1` 时构造。
- 与 AR034 的差异：AR034 提交前向的尾行 argmax 与 row-0 尾行 argmax 在 bf16 下可能 ULP 漂移；
  探针 P2 记录 `argmax_match_rate`，若不等于 1.0 则融合**在质量面上退化为「草稿源切换」**，
  由 H1（token 级 vs 基线）与 repro 面兜底，如实记录。

### 3.2 融合前向

```python
fused_in = torch.cat([x[:, -block_size:], next_canvas], dim=1)      # [1, 2B]
out = raw_model(input_ids=fused_in, use_cache=True,
                past_key_values=kv, update_past_key_values=True,
                block_size=block_size)
kv = _crop_history_kv(out.past_key_values, keep=hist_len + block_size)
aligned = cat([block_boundary_raw, out.logits[:, :-1, :]], dim=1)    # [1, 2B]
block_boundary_raw = aligned[:, block_size:block_size+1].clone()     # 下一块边界行
next_token = out.logits[:, block_size-1:block_size].argmax(-1)       # 与 AR034 语义一致
stash = {"block_idx": block_idx + 1, "aligned": aligned[:, block_size:]}
```

- 第 b 块的 KV 写入语义与 AR034 的 `[1,B]` 提交前向**完全相同**（块内因果/双向掩码不变），
  多出的第 b+1 块 KV 由 `_crop_history_kv` 裁掉；
- `next_token` 取 `out.logits[:, B-1]`（第 b 块尾行），与 AR034 的 `out.logits[:, -1]` 在语义上同源。

### 3.3 KV 裁剪

`DynamicCache` 面：`crop(keep)`；兼容 `layers` / `key_cache`+`value_cache` 两种容器面（AR034 `_expand_history_kv` 已踩过该坑）。
裁剪后必须校验 `get_seq_length() == keep`；任何异常 → 返回 `None` 并触发回退（不改变已写 `x`）。

### 3.4 草稿消费

`_history_speculative_small_block` 增参 `draft_aligned: Tensor | None`：

- 非 `None` 且 `stash.block_idx == block_idx` → 跳过草稿前向，直接 `aligned = draft_aligned`，**不增** `spec_draft_forwards`；
- 否则正常草稿（并清空 stash，防御块号不匹配）。

### 3.5 回退矩阵

| 条件 | 行为 |
|---|---|
| 裁剪 API 缺失 / 裁剪后长度不符 | 使用非融合提交路径；`RuntimeWarning` 一次；`commit_draft_fusion` 实例内降级为 False |
| `remaining_after < B + 1` | 非融合提交；不计 `spec_fused_forwards` |
| `active_len != block_size`（末块） | 非融合（融合仅定义于完整块） |
| 块号不匹配的 stash | 丢弃 + 正常草稿 |

## 4. 账本

| 键 | 含义 |
|---|---|
| `spec_fused_forwards` | 融合前向次数（默认关/回退时为 0） |
| `spec_draft_forwards` | 草稿前向次数（融合生效时按块数 −1） |
| `spec_verify_forwards` | 验证前向次数（不变） |

H3 面：`draft_forwards + fused_forwards == verify_forwards`（每块恰好一次「草稿来源」前向）。

## 5. 测试设计（UT-3601~3606）

| ID | 面 |
|---|---|
| 3601 | 默认关：两次运行 token 逐字节一致 + `spec_fused_forwards == 0` |
| 3602 | 开关打开后 mask-free、确定性、账本恒等式 |
| 3603 | 每块前向数：`fused_forwards == verify_forwards - 1`（末块退化） |
| 3604 | 裁剪不可用（monkeypatch `_crop_history_kv` 返回 None）→ 输出与关闭态逐字节一致 |
| 3605 | 块号不匹配 stash → 走正常草稿且输出不变 |
| 3606 | 非法组合：无 history / 无候选 / 与 `speculative_commit_gate` 同开 → `ValueError` |

## 6. 基准设计

三臂（`scripts/ar036_commit_draft_fusion_bench.py`）：A 非缓存基线 / D AR034 组合臂 / F D+融合。
冻结判定：
- **H1 质量**：F 的 token 级 == A、mask-free、decode 长度相同；
- **H2 速度**：`p50(D)/p50(F) >= 1.15`（预注册），并发布 `p50(A)/p50(F)`；
- **H3 账本**：`fused + draft == verify`（F 臂），D 臂 `fused == 0`，degraded 0；
- **H4 复现**：双跑逐 (arm,prompt) 全等；不一致 → 全体 blocked。

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| bf16 ULP 漂移导致 row-0 尾行 argmax ≠ 提交尾行 argmax | 探针 P2 量化；质量面 H1 + repro 兜底；若漂移导致不一致则如实记录并在文档中限定适用面 |
| KV 裁剪破坏历史（远端模型缓存实现差异） | P3/P4 探针 + UT-3604 回退 + 裁剪后长度校验 |
| 融合前向的 2B 画布触发远端模型内部布局假设 | P1 校验第 b 块 logits 与独立提交 bit-equal |
