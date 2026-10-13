# AR014 详细设计 — Triton 降级粒度细化（P1-11）+ 优先级表勘误

> srs：`specs/changes/AR014-triton-fallback-granularity/srs.md`（权威契约）
> 创建：2026-10-13；状态：final（design 门控首轮 FAIL：Major-1/2 mutation 观测面 → 修复；复核 FAIL：新 Major-3 "全程 forbidden" scoping → 修复 forbidden 区间（①后②前 setattr、③前恢复）+ n-1 D2 片段注 → 免重读放行；m-1~m-6/w-1~w-5 全落文档）

## 1. 设计总览

方案：**A（失败注册表 + 桶函数 + 共享记录 helper + 既有钉迁移）**
——fused_ops.py 一个生产文件改造，一个新测试文件，三个旧断言
迁移，指南勘误。

**否决 B（逐 shape 精确键）**：注册表随测试/模型 shape 无界增长；
处方明示"类别"。bucket 是防护网不是分类器（srs §5 假设一）。

**否决 C（dataclass 键对象）**：`tuple[str, str, int]` 哈希快、
可读、mypy 零成本；dataclass 需 frozen 才可哈希，纯增仪式。

**否决 D（失败路径继续置位布尔 + 注册表并存）**：双状态源必然
漂移；布尔收窄为纯手动开关后语义唯一（FR1），旧测试的"布尔被
置位"断言全部迁移为键断言（FR5），无残留消费方（req 门控全仓
grep 已证：消费点恰为 FR5 列举集合）。

## 2. 生产设计（fused_ops.py）

### D1 注册表与键布局

```python
_TRITON_FAILED_KEYS: set[tuple[str, str, int]] = set()

def _hidden_bucket(hidden_dim: int) -> int:
    """Pow2 ceiling of ``hidden_dim`` (4096→4096, 5000→8192, 1→1)."""
    return 1 << max(hidden_dim - 1, 0).bit_length()
```

- 键 = `(kernel_name, str(dtype), _hidden_bucket(hidden_dim))`：
  - merge（:174 附近）：`("merge", str(parent_ffn.dtype),
    _hidden_bucket(hidden_dim))`——`hidden_dim` 在 :193 解包后；
  - gather_select：`("gather_select", str(child_out.dtype),
    _hidden_bucket(hidden_dim))`——:304 解包后、:317 use_triton 前；
  - gate：`("gate", str(h_child.dtype), _hidden_bucket(hidden))`
    ——:637 解包后（`hidden = h_child.shape[2]`）；
- 公式用 `max(hidden_dim - 1, 0).bit_length()`（h=1 → 0 → 1<<0=1；
  srs 样例 4096→12→4096、5000→13→8192 逐一成立）；`hidden_dim`
  生产恒 ≥1，max 仅防 0 的病态输入；
- **键计算位置在 use_triton 评估之前**（三入口一致），保证注册表
  检查与既有 device/dtype/阈值检查的短路顺序稳定可钉。

### D2 分发序（三入口统一插入点）

现行：`_HAS_TRITON` → 手动布尔 → device/dtype 白名单 →（gate
另有 B*T 阈值）。改为在**手动布尔之后**插入注册表检查：

```
if not _HAS_TRITON or _TRITON_X_DISABLED or key in _TRITON_FAILED_KEYS:
    → PyTorch 回退
```

（示意：gather_select/gate 的字面序——三条件同 if。**merge 例外**
：`key` 在 shape 解包后才可得，注册表检查位于解包后、早退段之后，
见下"键可得处"子弹——Red 期勿照抄本片段到 merge，design 门控 n-1）

- **插入点以"键可得处"为准**（design 门控 m-3：merge 的
  `hidden_dim` 在 :193 才解包，:174 早退段不可得）：
  - gather_select：:304 解包后、:317 use_triton 前（srs FR1 字面
    序成立）；
  - gate：:637 解包后（`hidden = h_child.shape[2]` 为**新增读取**
    ——现行解包丢弃为 `_`，design 门控 w-5）、:638 use_triton 前；
  - merge：**:193 解包后、:195 block_size 前插入注册表检查**——
    位于 device/dtype 早退（:179-191）之后。两序无 observable
    差异：注册键只能来自"已通过 device/dtype 白名单"的失败，
    非白名单输入在两种顺序下都走同一回退调用（srs FR1 序表述
    同步勘误为"在键可得处插入"）；
- device/dtype 白名单/阈值检查顺序原样保留；
- 集合查找 O(1)，键为 (str, str, int)——无同步/拷贝（NFR）；
- 手动布尔语义收窄：True = 整 kernel 强制回退（优先于注册表），
  **失败路径不再赋值**——`global _TRITON_X_DISABLED` 声明随之删除
  （三入口从"读写"变"只读"，pyflakes/mypy 自然干净）。

### D3 共享失败记录 helper

```python
def _record_triton_failure(
    key: tuple[str, str, int], kernel_desc: str, exc: Exception
) -> None:
    """Record a (kernel, dtype, hidden-bucket) Triton failure once with a warning."""
    if key in _TRITON_FAILED_KEYS:
        return
    _TRITON_FAILED_KEYS.add(key)
    warnings.warn(
        f"Triton {kernel_desc} unavailable (dtype={key[1]}, "
        f"hidden-class={key[2]}): {exc}; falling back to the PyTorch path.",
        RuntimeWarning,
        stacklevel=3,
    )
```

- `kernel_desc` ∈ {"merge kernel", "gather/select kernel",
  "fused gate kernel"}——**既有 match 子串全部保留**（FR2/w-1：
  "…kernel unavailable" 是新模板连续子串；`{exc}` 透传保 T013
  match="boom"）；
- `stacklevel=3`（**有意偏离 srs FR2 字面值 2**——helper 多一层
  帧：1=helper、2=kernel 入口、3=调用方，落点对齐现行 stacklevel=2
  的语义；srs 同步勘误，design 门控 m-2）；
- 保留 helper 去重 guard（双守卫：dispatch 检查 + helper 幂等）。
  **代价明示**：同键静默因此有两条不可区分的守卫路径，"告警再现"
  型观测面对"仅移除其一"的变异不再可测——UT-1302 以 launch-count
  spy 直接钉"不尝试 kernel"（§3.2），mutation 集合相应重定义
  （§3.2 末，design 门控 M-1/M-2）；
- 直接读模块属性 `_TRITON_FAILED_KEYS`（读+方法调用，非重绑定）
  ——monkeypatch.setattr 替换绑定后 dispatch 与 helper 看到同一
  个 set（D5 隔离机制的根基；design 门控 w-3 复核一致）；
- 三处 except 改为：`_record_triton_failure(key, ..., exc)` + 原样
  回退调用（回退实现逐行不动；**三处回退形式互不相同**——merge/gate
  为模块函数、gather_select 为内联 index_select+where（:354-355），
  故 forbidden 回退手段仅对 merge/gate 可行，本设计仅用 merge，
  design 门控 w-2）。

### D4 docstring 更正（w-4）

- :169-170 "the PyTorch fallback is used for all subsequent calls"
  → "…for subsequent calls with the same (dtype, hidden-bucket)
  class; other classes still attempt Triton"；
- :591-592 同型；三布尔注释补"manual kill-switch: failures no
  longer set this flag (AR014)"。

### D5 测试隔离机制

- 新文件 autouse fixture：`monkeypatch.setattr(fused_ops,
  "_TRITON_FAILED_KEYS", set())`——**setattr 而非 .clear()**：
  monkeypatch 退出自动还原原绑定（进程级注册表零泄漏，FR5 m-3）；
- 三个迁移用例同法（各测试内快照替换，不用共享 conftest——
  改动面最小且不污染无关文件）。

## 3. 测试设计（tests/test_triton_fallback_granularity.py）

### 3.1 基础设施

| 组件 | 设计 |
|---|---|
| skipif | `@pytest.mark.skipif(not _HAS_TRITON or not torch.cuda.is_available(), ...)`——与 T013 双检写法相同（:368/:384/:409/:438/:464 先例，design 门控 m-1：cuda 检查非新增）；UT-1306 除外，CPU 直跑 |
| autouse fixture | D5 setattr 隔离 |
| 爆炸 kernel | 条件委托型（非 T013 无条件 boom）：`__getitem__(grid)` 返回闭包，命中条件（dtype）raise `RuntimeError("boom")` 否则 `real_kernel[grid](*args)` 透传——一个 monkeypatch 同时提供"爆炸键"与"真实键"两条路径 |
| 计数 spy kernel | `__getitem__` 计数 +1 后委托真 kernel——UT-1302 次调用以 `count == 0` 直接钉"dispatch 跳过、kernel 不再被尝试"（design 门控 M-1：告警缺席在双守卫下不可区分，launch 计数才是 dispatch 检查的可观测投影） |
| 禁止回退 | `monkeypatch.setattr(fused_ops, "_merge_stable_divergent_torch", _forbidden)`（T013 :273/:379 先例）——真实键调用若错误回退即 AssertionError；**仅对 merge/gate 可行**（gather_select 回退是内联的，w-2） |
| torch 参考 | merge：`torch.where(mask.unsqueeze(-1), parent, child)` 手写镜像（不 import 生产回退函数——AR012 D2 纪律：回退与参考同实现即循环验证）；与 forbidden 正交——镜像管期望值、forbidden 管路径 |

### 3.2 用例矩阵

| ID | 构造 | 断言 |
|---|---|---|
| UT-1301 dtype 隔离 | merge，fp32+fp16 CUDA 张量（B=2,T=8,H=128 同 bucket）；第一步 dtype 条件爆炸（仅 fp32 raise，fp16 委托真 kernel）——**①为失败路径，回退必须合法**；**①后②前** setattr forbidden 回退（②消费：真键路径若错误回退即 AssertionError）；**③前**保存原函数、setattr 恢复（③为失败路径，回退须合法），再换第二个 dtype 条件爆炸（仅 **fp16** raise——第二失败键 **dtype 维**不同，键必然相异） | ①fp32 首调 `pytest.warns(RuntimeWarning, match="merge kernel unavailable")` + 结果==参考（合法回退）；②fp16（forbidden 回退下）结果==参考——真 Triton 路径自证；③fp16 首调**第二条**告警（互不抑制，m-4）+ 结果==参考（回退已恢复合法）——注册表恰含 {("merge","torch.float32",128), ("merge","torch.float16",128)} 两键 |
| UT-1302 同键静默 | merge，无条件爆炸（T013 `_BoomKernel` 直用）首调；**次调前换计数 spy kernel**（计数+委托真 kernel） | 首调 `pytest.warns` + 结果正确 + 注册表恰一键；次调：**spy 计数==0**（dispatch 跳过、kernel 不再被尝试——FR1"不尝试 kernel"的直接钉）+ `catch_warnings(record=True)` 零 RuntimeWarning + 结果==参考（回退） |
| UT-1303 bucket 隔离 | merge，同 dtype fp32，H=512（爆）vs H=2048（真）；shape 条件爆炸 | H=512 告警+回退正确；H=2048（forbidden 回退下）==参考；注册表恰含 512 键 |
| UT-1304 总开关兼容 | `_TRITON_MERGE_DISABLED=True`（monkeypatch），未注册键，CUDA 张量 | 强制回退：结果正确、零告警、**注册表零增长**（monkeypatch 还原属 teardown 行为，不在测试体内断言——design 门控 m-5：与 autouse fixture 共享同一 monkeypatch 实例，`monkeypatch.undo()` 会连带撤销注册表隔离，禁用） |
| UT-1305 告警可诊断性 | 复用 UT-1301 ①的告警捕获 | 消息含 "merge kernel unavailable"、`torch.float32`、`hidden-class=128` 三子串 |
| UT-1306 bucket 纯函数 | CPU 直跑，不 skipif CUDA | `_hidden_bucket`：4096→4096、5000→8192、512→512、1→1、2→2（srs 3.4 全样例） |

- 显式种子纪律：数据张量用 `torch.manual_seed(S)` 生成（AR012
  纪律；断言逐位一致无统计容差）；
- **mutation spot-check（srs 3.4，集合经 design 门控 M-1/M-2
  重定义）**：
  - ①注册表不写入（helper 不 add）→ UT-1302 次调 spy 计数==1
    翻红（dispatch 未命中→重试）；
  - ②分发不查注册表（dispatch 检查移除）→ UT-1302 次调 spy
    计数==1 翻红（与 ① 同观测面，但根因路径不同——①空表、
    ②有键不查）；
  - ③键缺 dtype 维度（`key=(kernel, bucket)` 两元）→ UT-1301 ②
    fp16 在 forbidden 回退下静默回退 → AssertionError 翻红；
  - ~~告警去条件（helper guard 移除）~~ **不可测，从集合移除**：
    dispatch 正常时同键次调根本不进 except/helper，guard 对黑盒
    不可达（design 门控 M-2 全仓推演）；guard 保留为纵深防御，
    其正确性由 dispatch 单守卫的可观测性传递覆盖。

### 3.3 既有钉迁移（FR5）

| 文件:行 | 旧断言 | 新断言 |
|---|---|---|
| test_fused_ops.py:428 | `assert fused_ops._TRITON_GATHER_SELECT_DISABLED is True` | `assert ("gather_select", "torch.float32", 32) in fused_ops._TRITON_FAILED_KEYS`（输入 `_t013_small_gather_inputs` hidden=**32**，design 门控 m-4 实测）+ `..._DISABLED is False` + 注册表 setattr 隔离 |
| test_fused_ops.py:482 | `assert fused_ops._TRITON_MERGE_DISABLED is True` | `assert ("merge", "torch.float32", 128) in fused_ops._TRITON_FAILED_KEYS`（mp/mc=`torch.randn(2,4,128)` fp32）+ 布尔 False + 隔离 |
| test_fused_gate.py:443 | `assert fused_ops._TRITON_GATE_DISABLED is True` | `assert ("gate", "torch.float32", 64) in fused_ops._TRITON_FAILED_KEYS`（h=`torch.randn(2,512,64)` fp32）+ 布尔 False + 隔离；**:441/:453 out_count 断言原样保留**（w-3） |
| test_fused_ops.py:312-315 / test_fused_gate.py:203 | 存在性钉 | 保留 + 补 `isinstance(fused_ops._TRITON_FAILED_KEYS, set)` |
| test_fused_ops.py:440-461 / test_ar002_perf_proxy.py:365 / test_cuda_graph.py:596 / test_fused_gate.py:217 | — | **零改动**（Green 期单独跑绿清单）；:440-461 会向进程级注册表泄漏 `("gather_select","torch.float32",32)`——与现行布尔泄漏同构且粒度更细（w-1，风险降不升），"零改动"禁止其加隔离，Green 全量跑观察 |

- 迁移用例键的字面值已按实际张量实测（上表，design 门控 m-4）；
  二次调用静默断言（:432/:451 附近）保留——新语义下仍成立
  （注册表命中→回退，不再尝试 kernel）。

## 4. 文档设计（FR6，T004）

- 指南 P1 表 7 行 + :238 + M 表（M-9 ◐）逐条勾选注格式沿用
  AR013 先例（~~原文~~ **✅ 证据（AR/文件:行）**）；:35 旗标句、
  fused_ops 两处 docstring 同步；
- 第二十一部分回链 + CHANGELOG（Changed：**告警契约变更明示**
  ——"每 kernel 一条"→"每 (kernel, dtype, hidden bucket) 键一条"
  + 失败不再置位布尔；Tests）；
- AGENTS #9/#38 各补一句粒度注。

## 5. 风险与回退

| 风险 | 缓解 |
|---|---|
| 注册表键泄漏跨测试（m-3 实证：UT-006c 与 test_cuda_graph:576 同键） | D5 setattr 隔离（新文件 autouse + 迁移用例逐个）；Green 期全量跑验证无互噬 |
| 爆炸 kernel 委托真 kernel 的 monkeypatch 面过宽（改 `_merge_kernel` 属性） | T013 先例同法（:420/:478）；monkeypatch 退出还原；真 kernel 仅在 CUDA+Triton 用例内触达 |
| 告警模板重组破坏既有 match | FR2 三子串逐字保留（D3 模板推演已核）；UT-1305 + 迁移用例 + :440-461 零改动钉三重护栏 |
| hidden 维度键不足以隔离 grid 型 launch 失败 | srs §5 假设一明示接受（类别是防护网）；键 tuple 结构预留扩展 |
| 807 基线外的隐性消费方 | req 门控全仓 grep 已清点（FR5 集合封闭）；T003 全量兜底 |
