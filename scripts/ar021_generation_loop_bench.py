"""AR021/T003: production generation-loop folding benchmark (real checkpoint).

Runs the REAL ``FastDLLMSampler.sample`` generation loop (the AR021 production
wiring: pinned fold-context trio, forced-divergent active block, shape-keyed
graph runner pool, batched speculative block verification) on the local
Fast-dLLM-v2-1.5B checkpoint in THREE modes over one shared model instance,
prompt set, and seed:

* ``baseline`` -- the raw sampler (``folded_model=None``), the unchanged
  production recipe.
* ``folded-serial`` -- ``sampler(folded_model=MFF_A)`` with
  ``MFF_A = ManualFoldedForward(use_cuda_graph=True,
  graph_runner_pool_size=8, split_layers=False)``: FR2 chain folding with a
  serial shape-keyed runner pool.
* ``folded-batched`` -- ``sampler(folded_model=MFF_B,
  num_block_candidates=4, candidate_top_k=10)`` with ``MFF_B = MFF_A +
  graph_adaptive_capacity=True``: FR3 batched speculative verification with
  the AR017 adaptive capacity engaged on the batched path (batched-only by
  the #37 contract; N=4 from the AR019 artifact matrix; A5 scheme a).

Benchmark scale (frozen, design 4.2.2(6)): P=2 prompts, ``num_tokens=128``
(W=4 shape windows); pool cap 8 holds every mode's entries (<= 8 per pool)
with zero evictions. ``--quick``: reps=3, single prompt, ``num_tokens``
truncated to two blocks (W=2), <= 5 min.

Hypotheses (declared before measurement; verdicts from data only):

* **H1** -- folded-serial generation is equivalent to baseline: exact
  token-level agreement position-by-position (hard gate, temperature 0 makes
  every committed token an argmax) AND per-forward decision-slice logits
  ``allclose(rtol=1e-2, atol=1e-1)`` (AR019 H2 arbitration tolerance);
  bit-parity is a report-only subfield. The synthetic mini-model face is
  pinned bit-exact by UT-2105 (not re-measured here).
* **H2** -- folded-serial whole-generation wall clock < baseline:
  ``sigma = max(sigma_base, sigma_serial)``; supported iff
  ``p50(base) - p50(serial) > 2*sigma``; refuted iff the reverse exceeds
  ``2*sigma``; else inconclusive.
* **H3** -- folded-batched whole-generation wall clock < folded-serial
  (speed face, three-state as H2); quality face (per-block exact-match rate
  vs baseline + post-hoc confidence distribution) is REPORT-ONLY, no gate.
* **H4** -- full-stack production exercise: every in-scope shape window
  (>= 2 forwards within the window; window-first forwards are structural
  eager by design) has serial graph captures > 0 AND timed replays > 0
  (folded-serial); every window with a batched call has batched captures >
  0 (folded-batched); the publish ledger stays within ``c * (L+N)`` with
  ``c = 4`` frozen at design (per serial event ``L+1`` ops, per batched
  event ``O(L+N)`` ops, AR020-style); the no-fallback canary (degradation
  class only, timed segment == 0) is all-true. Adaptive-capacity events are
  verified only when their trigger condition actually fired (untriggered =
  neutral, never refutes).

Warmup: one full discarded sample per (mode, prompt) before any timing --
pool pre-capture for every shape window + Triton/JIT warmup + the AR017
transition; capture costs are excluded from p50 (srs FR4). Timed-segment
re-captures, if any, are counted AND timed honestly (never hidden).

Interleaving: per-rep round-robin over the three modes (rep r starts at
``r % 3``). Timing is host ``perf_counter`` + ``torch.cuda.synchronize``
(#39; CUPTI is probed and recorded null + note when unavailable). Per-block
anchors come from the forward wrappers (a ``step_idx`` change is the block
boundary; syncs happen only at those boundaries) on top of the
whole-generation wall clock.

Run (AR019 venv, transformers==4.53.1)::

    python -m scripts.ar021_generation_loop_bench [--quick] [--reps 30]
        [--model-path PATH] [--out PATH] [--hf-home PATH]

``--model-path`` defaults to the ``FASTDLLM_MODEL_PATH`` environment
variable; neither present raises ``FileNotFoundError`` (AGENTS #10: no
synthetic-model fallback for a real-checkpoint benchmark). ``--quick``
writes to a separate artifact path and never the formal one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from actfold.core import ActivationCache, SimilarityGate  # noqa: E402
from actfold.core.adaptive_capacity import AdaptiveCapacityConfig  # noqa: E402
from actfold.models.architecture_utils import ManualFoldedForward  # noqa: E402
from actfold.models.base import DiffusionLLM  # noqa: E402
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig  # noqa: E402
from actfold.models.sampling_utils import right_shift_logits  # noqa: E402
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER  # noqa: E402
from scripts.algo_experiments import stats_from_samples  # noqa: E402

DEFAULT_OUTPUT_PATH = Path("results/optimization/ar021_generation_loop_bench.json")
QUICK_OUTPUT_PATH = Path("results/optimization/ar021_generation_loop_bench_quick.json")
DEFAULT_MODEL_ENV = "FASTDLLM_MODEL_PATH"

BLOCK = 32
TAU = 0.95
EPS = 1e-8
CAPACITY_RATIO = 0.5
POOL_SIZE = 8
CACHE_BUDGET = 8192
SEED = 2026
NUM_CANDIDATES = 4
CANDIDATE_TOP_K = 10
LEDGER_C = 4
FULL_NUM_TOKENS = 128
QUICK_NUM_TOKENS = 64

_MODES = ("baseline", "folded-serial", "folded-batched")

# AR017 event-count snapshot per MFF (id -> len(policy.events)), taken right
# after the sidecar and consumed by ``_serialize_windows`` for timed diffs.
_snapshot_events: dict[int, int] = {}

_PROMPT_TEXTS = (
    "Explain the key ideas of the theory of relativity in a few short "
    "sentences for a general audience.",
    "The ocean appears blue because water absorbs red light more strongly "
    "than blue light. Describe how photosynthesis works in plants, why "
    "chlorophyll matters, and list the first five prime numbers.",
)


# ---------------------------------------------------------------------------
# Instrumented wrappers (block anchors, counters, recording)
# ---------------------------------------------------------------------------
class _BlockTimer:
    """Host-side per-block anchors with syncs only at block boundaries."""

    def __init__(self) -> None:
        self.blocks: list[tuple[int, float]] = []
        self._current: int | None = None
        self._anchor: float | None = None

    def reset(self) -> None:
        self.blocks = []
        self._current = None
        self._anchor = None

    def note(self, block_idx: int) -> None:
        """Open/close anchors on a ``step_idx`` (block) change."""
        if block_idx == self._current:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self._current is not None and self._anchor is not None:
            self.blocks.append((self._current, (now - self._anchor) * 1000.0))
        self._current = block_idx
        self._anchor = now

    def finalize(self) -> None:
        """Close the trailing block anchor (drains the last forwards)."""
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self._current is not None and self._anchor is not None:
            self.blocks.append((self._current, (now - self._anchor) * 1000.0))
        self._current = None
        self._anchor = None


class _BenchDiffusionLLM(DiffusionLLM):
    """Raw-path wrapper: strips branch kwargs, block anchors, recording.

    The remote Fast-dLLM forward overwrites the passed ``attention_mask``
    with its internal block-causal ``eval_mask`` (verified element-equal to
    the fold-context mask by the AR019 P4 probe), so passing the raw
    sampler's ``[B, T]`` padding mask through is inert and the two paths
    share one mask semantics.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        super().__init__("fast-dllm-v2-1.5B-ar021-bench")
        self.model = model
        self.tokenizer = tokenizer
        self.block_timer = _BlockTimer()
        self.recording = False
        self.records: list[torch.Tensor] = []

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.block_timer.note(int(kwargs.pop("step_idx", 0)))
        kwargs.pop("branch_id", None)
        kwargs.pop("parent_branch_id", None)
        outputs = self.model(input_ids=tokens, attention_mask=attention_mask)
        logits: torch.Tensor = outputs.logits
        if self.recording:
            self.records.append(logits[:, -BLOCK:].detach().to("cpu"))
        return logits

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        embed_fn = self.model.get_input_embeddings()
        return embed_fn(tokens)

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def hidden_dim(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def num_heads(self) -> int:
        return int(self.model.config.num_attention_heads)

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)


class _BenchMFF(ManualFoldedForward):
    """Instrumented ``ManualFoldedForward``: anchors, counters, records.

    The instrumentation is host-only bookkeeping around the production
    forward (a tuple append per call, a ``len()`` per batched call, and the
    block-boundary syncs of the shared ``_BlockTimer``); no graph-path or
    cache semantics are touched.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.block_timer = _BlockTimer()
        self.recording = False
        self.records: list[torch.Tensor] = []
        self.window_calls: list[tuple[int, int, str]] = []
        self.recapture_windows: list[tuple[int, int]] = []
        self.branches_seen: list[Any] = []
        self.serial_calls = 0
        self.batched_calls = 0

    def _adaptive_event_count(self) -> int:
        policy = self.graph_adaptive_policy
        return 0 if policy is None else len(policy.events)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        tokens = args[0] if args else kwargs["tokens"]
        step_idx = int(kwargs.get("step_idx", 0))
        batched = kwargs.get("branch_ids") is not None
        seq_len = int(tokens.shape[-1])
        events_before = 0
        if batched:
            self.batched_calls += 1
            events_before = self._adaptive_event_count()
        else:
            self.serial_calls += 1
        self.block_timer.note(step_idx)
        self.window_calls.append((step_idx, seq_len, "batched" if batched else "serial"))
        branch_id = kwargs.get("branch_id")
        if branch_id is not None:
            self.branches_seen.append(branch_id)
        for bid in kwargs.get("branch_ids") or ():
            self.branches_seen.append(bid)
        out = super().forward(*args, **kwargs)
        if self.recording and not batched:
            self.records.append(out[:, -BLOCK:].detach().to("cpu"))
        if batched:
            after = self._adaptive_event_count()
            for _ in range(after - events_before):
                self.recapture_windows.append((step_idx, seq_len))
        return out


# ---------------------------------------------------------------------------
# Model / prompt / config construction
# ---------------------------------------------------------------------------
def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block_bytes = handle.read(chunk)
            if not block_bytes:
                break
            digest.update(block_bytes)
    return digest.hexdigest()


def _load_model(model_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    weights = model_path / "model.safetensors"
    meta = {
        "model_path": str(model_path),
        "weights_sha256": _sha256_of(weights) if weights.exists() else None,
        "config": {
            "hidden_size": int(model.config.hidden_size),
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "vocab_size": int(model.config.vocab_size),
        },
    }
    return model, tokenizer, meta


def _prepare_prompts(tokenizer: Any, quick: bool) -> list[dict[str, Any]]:
    """Tokenize and pre-pad prompts to exact block multiples.

    Pre-padding makes the sampler's own first-block padding zero, so the
    pinned fold-context mask is the pure block-causal mask -- element-equal
    to the remote model's internal ``eval_mask`` (which overwrites whatever
    the raw path passes). Non-zero padding would diverge the two mask
    semantics and break cross-mode parity by construction.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    texts = _PROMPT_TEXTS[:1] if quick else _PROMPT_TEXTS
    prompts: list[dict[str, Any]] = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        raw_len = int(ids.shape[1])
        pad_len = (BLOCK - raw_len % BLOCK) % BLOCK
        if pad_len:
            ids = torch.cat(
                [
                    ids,
                    torch.full((1, pad_len), pad_id, dtype=ids.dtype, device=ids.device),
                ],
                dim=1,
            )
        prompts.append(
            {
                "text": text,
                "ids": ids,
                "raw_len": raw_len,
                "padded_len": int(ids.shape[1]),
                "pad_id": int(pad_id),
            }
        )
    padded = [p["padded_len"] for p in prompts]
    if len(set(padded)) != len(padded):
        raise RuntimeError(
            "prompts must pad to distinct block-multiple lengths (pool-key "
            f"separation, design 4.2.2(3)); got {padded}"
        )
    return prompts


def _sampler_config(num_tokens: int, num_block_candidates: int):
    return FastDLLMSamplerConfig(
        num_tokens=num_tokens,
        block_size=BLOCK,
        small_block_size=BLOCK,
        threshold=0.95,
        temperature=0.0,
        seed=SEED,
        num_block_candidates=num_block_candidates,
        candidate_top_k=CANDIDATE_TOP_K,
    )


def _make_mff(model: Any, adaptive: bool) -> _BenchMFF:
    cache = ActivationCache(max_entries_per_layer=CACHE_BUDGET, device="cuda")
    gate = SimilarityGate(tau=TAU, metric="cosine", eps=EPS)
    return _BenchMFF(
        model,
        cache=cache,
        gate=gate,
        scheduler=None,
        split_layers=False,
        use_cuda_graph=True,
        graph_capacity_ratio=CAPACITY_RATIO,
        graph_adaptive_capacity=adaptive,
        graph_adaptive_config=AdaptiveCapacityConfig() if adaptive else None,
        graph_runner_pool_size=POOL_SIZE,
    )


# ---------------------------------------------------------------------------
# Environment probes (AR019 style)
# ---------------------------------------------------------------------------
def _gpu_driver_version() -> str | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        first = out.stdout.strip().splitlines()[0].strip()
        return first or None
    except Exception:  # noqa: BLE001 - metadata probe must never fail the run
        return None


def _cuda_profiler_supported() -> bool:
    try:
        x = torch.ones(4, device="cuda")
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            x = x + 1
            torch.cuda.synchronize()
        return any(evt.device_type == torch.profiler.DeviceType.CUDA for evt in prof.key_averages())
    except Exception:  # noqa: BLE001 - probe must never raise
        return False


# ---------------------------------------------------------------------------
# Comparison / quality helpers
# ---------------------------------------------------------------------------
def _token_agreement(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float]:
    """Exact position-wise agreement of two decoded [1, L] token rows."""
    if a.shape != b.shape:
        n = min(a.shape[1], b.shape[1])
        if n == 0:
            return False, 0.0
        rate = float((a[:, :n] == b[:, :n]).float().mean().item())
        return False, rate
    equal = a == b
    return bool(equal.all().item()), float(equal.float().mean().item())


def _compare_logits_slices(
    ref_records: list[torch.Tensor], live_records: list[torch.Tensor]
) -> dict[str, Any]:
    """Pairwise per-forward comparison of the decision-slice logits.

    Both sides record the raw (pre-right-shift) ``logits[:, -32:]`` slice on
    CPU in bf16; temperature 0 makes every sampler decision an argmax over
    exactly this slice, so forward i is comparable iff the trajectories
    agree up to i.
    """
    n = min(len(ref_records), len(live_records))
    entries: list[dict[str, Any]] = []
    for i in range(n):
        a = ref_records[i]
        b = live_records[i]
        af = a.float()
        bf = b.float()
        entries.append(
            {
                "allclose_rtol1e-2_atol1e-1": bool(torch.allclose(af, bf, rtol=1e-2, atol=1e-1)),
                "max_abs_diff": round(float((af - bf).abs().max().item()), 6),
                "bit_equal_fraction": round(float((a == b).float().mean().item()), 6),
            }
        )
    return {
        "compared_forwards": n,
        "ref_forwards": len(ref_records),
        "live_forwards": len(live_records),
        "allclose_all": bool(entries) and all(e["allclose_rtol1e-2_atol1e-1"] for e in entries),
        "max_abs_diff_overall": round(max((e["max_abs_diff"] for e in entries), default=0.0), 6),
        "bit_equal_fraction_mean": round(
            sum(e["bit_equal_fraction"] for e in entries) / max(1, len(entries)),
            6,
        ),
        "per_forward": entries,
    }


def _block_exact_match(
    base_seq: torch.Tensor,
    other_seq: torch.Tensor,
    padded_len: int,
    num_tokens: int,
) -> dict[str, Any]:
    """Per-block token exact-match rate of a mode's output vs baseline."""
    b = base_seq[0].tolist()
    c = other_seq[0].tolist()
    n = min(len(b), len(c))
    per_block: dict[str, Any] = {}
    matched = 0
    total = 0
    for blk in range(num_tokens // BLOCK):
        s = padded_len + blk * BLOCK
        e = s + BLOCK
        seg = max(0, min(e, n) - s)
        if seg <= 0:
            per_block[f"block{blk}"] = None
            continue
        m = sum(1 for i in range(seg) if b[s + i] == c[s + i])
        per_block[f"block{blk}"] = round(m / seg, 6)
        matched += m
        total += seg
    return {
        "per_block_exact_match": per_block,
        "overall_exact_match": round(matched / total, 6) if total else None,
        "len_baseline": len(b),
        "len_other": len(c),
    }


def _posthoc_confidence(
    model: Any, seq_row: torch.Tensor, padded_len: int, num_tokens: int
) -> dict[str, Any]:
    """Post-hoc confidence of the committed generated tokens (report-only).

    One untimed raw-model forward over the final committed row; the token
    probability at generated position j is the softmax (after the sampler's
    right-shift) gathered at the committed token.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        ids = seq_row[None, :].to(device)
        logits = model(input_ids=ids).logits
    shifted = right_shift_logits(logits)[0]
    probs = torch.softmax(shifted.float(), dim=-1)
    length = int(seq_row.shape[0])
    row = seq_row.to(probs.device)
    conf = probs[torch.arange(length, device=probs.device), row]
    per_block: dict[str, Any] = {}
    for blk in range(num_tokens // BLOCK):
        s = padded_len + blk * BLOCK
        e = min(s + BLOCK, length)
        if s >= e:
            per_block[f"block{blk}"] = None
            continue
        seg = conf[s:e]
        per_block[f"block{blk}"] = {
            "mean": round(float(seg.mean().item()), 6),
            "min": round(float(seg.min().item()), 6),
        }
    gen = conf[padded_len : min(padded_len + num_tokens, length)]
    return {
        "overall_mean": round(float(gen.mean().item()), 6) if gen.numel() else None,
        "overall_min": round(float(gen.min().item()), 6) if gen.numel() else None,
        "per_block": per_block,
    }


# ---------------------------------------------------------------------------
# Pool-window serialization
# ---------------------------------------------------------------------------
def _ctx_mask_map(samplers: dict[str, FastDLLMSampler]) -> dict[int, tuple[int, int]]:
    """Map ``id(ctx.mask)`` -> ``(T, prompt_len)`` from the folded samplers."""
    out: dict[int, tuple[int, int]] = {}
    for mode in ("folded-serial", "folded-batched"):
        cache = getattr(samplers[mode], "_fold_ctx_cache", None) or {}
        for key, ctx in cache.items():
            out[id(ctx.mask)] = (int(key[0]), int(key[1]))
    return out


def _window_label(shape: tuple[int, ...], mask: Any, ctx_map: dict[int, Any]) -> str:
    """Serialize a pool key to a ``p{prompt_len}_block{b}_T{t}`` label."""
    hit = ctx_map.get(id(mask))
    if hit is None:
        return "shape_" + "x".join(str(d) for d in shape) + "_unresolved"
    seq_len, prompt_len = hit
    block = (seq_len - prompt_len) // BLOCK - 1
    return f"p{prompt_len}_block{block}_T{seq_len}"


def _snapshot_pool(mff: _BenchMFF) -> dict[str, Any]:
    """Raw-key snapshot of the pool counters for timed-segment diffs."""
    stats = mff.graph_pool_stats
    snap: dict[str, Any] = {}
    for section in ("serial", "batched"):
        snap[section] = {
            key: dict(entry)
            for key, entry in stats[section].items()
            if isinstance(entry, dict) and "captures" in entry
        }
    snap["degradation"] = stats["degradation_fallback_count"]
    snap["structural"] = stats["structural_eager_count"]
    return snap


def _window_forwards(window_calls: list[tuple[int, int, str]]) -> dict[str, dict]:
    """Per-window forward counts (serial/batched) from one generation."""
    counts: dict[str, dict] = {}
    for step_idx, seq_len, kind in window_calls:
        prompt_len = seq_len - BLOCK * (step_idx + 1)
        label = f"p{prompt_len}_block{step_idx}_T{seq_len}"
        entry = counts.setdefault(label, {"serial": 0, "batched": 0})
        entry[kind] += 1
    return counts


def _serialize_windows(
    mff: _BenchMFF,
    snap: dict[str, Any],
    ctx_map: dict[int, Any],
) -> dict[str, Any]:
    stats = mff.graph_pool_stats
    result: dict[str, Any] = {
        "semantics": (
            "captures_cumulative: instance lifetime (warmup + sidecar + "
            "timed, H4 capture criterion); replays_timed / *_timed: diff "
            "from the post-warmup snapshot (sidecar included in the "
            "snapshot base); evictions: cumulative"
        )
    }
    for section in ("serial", "batched"):
        windows: dict[str, Any] = {}
        for key, entry in stats[section].items():
            if not (isinstance(entry, dict) and "captures" in entry):
                continue
            label = _window_label(key[0], key[1], ctx_map)
            base = snap[section].get(key, {})
            windows[label] = {
                "captures_cumulative": int(entry["captures"]),
                "replays_cumulative": int(entry["replays"]),
                "replays_timed": int(entry["replays"]) - int(base.get("replays", 0)),
                "evictions": int(entry.get("evictions", 0)),
            }
        result[section] = {"pool_size": int(stats[section]["pool_size"])}
        result[section]["windows"] = windows
    result["degradation_fallback_count"] = {
        "cumulative": int(stats["degradation_fallback_count"]),
        "timed": int(stats["degradation_fallback_count"] - snap["degradation"]),
    }
    result["structural_eager_count"] = {
        "cumulative": int(stats["structural_eager_count"]),
        "timed": int(stats["structural_eager_count"] - snap["structural"]),
    }
    policy = mff.graph_adaptive_policy
    result["ar017_recapture_events"] = {
        "cumulative": 0 if policy is None else len(policy.events),
        "timed": (
            0
            if policy is None
            else len(policy.events) - _snapshot_events.get(id(mff), len(policy.events))
        ),
    }
    return result


# ---------------------------------------------------------------------------
# Generation runner
# ---------------------------------------------------------------------------
def _generate(
    mode_name: str,
    prompt: dict[str, Any],
    modes: dict[str, Any],
    llm: _BenchDiffusionLLM,
) -> dict[str, Any]:
    """Run one full generation of a mode over a prompt; return the record."""
    mff: _BenchMFF | None = modes[mode_name]["mff"]
    sampler: FastDLLMSampler = modes[mode_name]["sampler"]
    timer = llm.block_timer if mff is None else mff.block_timer
    timer.reset()
    if mff is not None:
        mff.window_calls = []
        mff.recapture_windows = []
        mff.branches_seen = []
    GLOBAL_STABILITY_PROFILER.reset()
    with torch.no_grad():
        out = sampler.sample(prompt["ids"], folded_model=mff)
    timer.finalize()
    record: dict[str, Any] = {
        "blocks": list(timer.blocks),
        "sequences": out.sequences.detach().to("cpu"),
    }
    if mff is not None:
        record["window_calls"] = list(mff.window_calls)
        record["recapture_windows"] = list(mff.recapture_windows)
        ratios: list[float] = []
        for bid in mff.branches_seen:
            profile = GLOBAL_STABILITY_PROFILER.get_profile(bid)
            if profile:
                ratios.extend(s.stable_ratio for s in profile.layer_stats)
        record["stable_ratios"] = ratios
    GLOBAL_STABILITY_PROFILER.reset()
    return record


def _verdict_p50(faster: Any, slower: Any) -> dict[str, Any]:
    """Three-state p50 verdict with sigma = max(sigma_a, sigma_b) (srs FR4)."""
    sigma = max(faster.std, slower.std)
    delta = slower.p50 - faster.p50
    if delta > 2 * sigma:
        verdict = "supported"
    elif -delta > 2 * sigma:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "p50_slower_ms": round(slower.p50, 4),
        "p50_faster_ms": round(faster.p50, 4),
        "delta_ms": round(delta, 4),
        "sigma_ms": round(sigma, 4),
        "sigma_rule": "sigma = max(sigma_slower, sigma_faster)",
    }


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run_bench(
    model_path: Path,
    reps: int,
    quick: bool,
) -> dict[str, Any]:
    """Run the three-mode generation-loop benchmark on the real checkpoint."""
    if not torch.cuda.is_available():
        raise RuntimeError("run_bench requires a CUDA device")

    num_tokens = QUICK_NUM_TOKENS if quick else FULL_NUM_TOKENS
    model, tokenizer, model_meta = _load_model(model_path)
    prompts = _prepare_prompts(tokenizer, quick)
    llm = _BenchDiffusionLLM(model, tokenizer)
    mff_serial = _make_mff(model, adaptive=False)
    mff_batched = _make_mff(model, adaptive=True)
    samplers = {
        "baseline": FastDLLMSampler(llm, config=_sampler_config(num_tokens, 0)),
        "folded-serial": FastDLLMSampler(llm, config=_sampler_config(num_tokens, 0)),
        "folded-batched": FastDLLMSampler(llm, config=_sampler_config(num_tokens, NUM_CANDIDATES)),
    }
    modes: dict[str, Any] = {
        "baseline": {"sampler": samplers["baseline"], "mff": None},
        "folded-serial": {"sampler": samplers["folded-serial"], "mff": mff_serial},
        "folded-batched": {
            "sampler": samplers["folded-batched"],
            "mff": mff_batched,
        },
    }
    num_layers = llm.num_layers

    # -- warmup: one full discarded sample per (mode, prompt) --------------
    for mode_name in _MODES:
        for prompt in prompts:
            _generate(mode_name, prompt, modes, llm)
        print(f"warmup done: {mode_name}")

    # -- sidecar (untimed, instrumented) ------------------------------------
    sidecar: dict[str, Any] = {"h1": [], "quality": [], "ledger": {}}
    base_sequences: list[torch.Tensor] = []
    window_counts: dict[str, dict[str, dict]] = {
        "folded-serial": {},
        "folded-batched": {},
    }
    for pi, prompt in enumerate(prompts):
        padded_len = prompt["padded_len"]
        llm.recording = True
        llm.records = []
        base_rec = _generate("baseline", prompt, modes, llm)
        llm.recording = False
        base_records, llm.records = llm.records, []
        base_sequences.append(base_rec["sequences"])

        ops0 = mff_serial.cache.write_op_count
        s0, b0 = mff_serial.serial_calls, mff_serial.batched_calls
        mff_serial.recording = True
        mff_serial.records = []
        ser_rec = _generate("folded-serial", prompt, modes, llm)
        mff_serial.recording = False
        ser_records, mff_serial.records = mff_serial.records, []
        ledger_serial = {
            "ops_delta": mff_serial.cache.write_op_count - ops0,
            "serial_events": mff_serial.serial_calls - s0,
            "batched_events": mff_serial.batched_calls - b0,
        }
        ledger_serial["bound"] = LEDGER_C * (
            (num_layers + 1) * ledger_serial["serial_events"]
            + (num_layers + NUM_CANDIDATES) * ledger_serial["batched_events"]
        )
        ledger_serial["within_bound"] = bool(ledger_serial["ops_delta"] <= ledger_serial["bound"])
        token_ok, token_rate = _token_agreement(base_rec["sequences"], ser_rec["sequences"])
        logits_cmp = _compare_logits_slices(base_records, ser_records)
        sidecar["h1"].append(
            {
                "prompt_index": pi,
                "token_match": token_ok,
                "token_match_rate": round(token_rate, 6),
                "forward_counts_match": bool(
                    logits_cmp["ref_forwards"] == logits_cmp["live_forwards"]
                ),
                "logits": logits_cmp,
            }
        )
        window_counts["folded-serial"][pi] = _window_forwards(ser_rec["window_calls"])

        ops0 = mff_batched.cache.write_op_count
        s0, b0 = mff_batched.serial_calls, mff_batched.batched_calls
        bat_rec = _generate("folded-batched", prompt, modes, llm)
        ledger_batched = {
            "ops_delta": mff_batched.cache.write_op_count - ops0,
            "serial_events": mff_batched.serial_calls - s0,
            "batched_events": mff_batched.batched_calls - b0,
        }
        ledger_batched["bound"] = LEDGER_C * (
            (num_layers + 1) * ledger_batched["serial_events"]
            + (num_layers + NUM_CANDIDATES) * ledger_batched["batched_events"]
        )
        ledger_batched["within_bound"] = bool(
            ledger_batched["ops_delta"] <= ledger_batched["bound"]
        )
        sidecar["ledger"][f"prompt{pi}"] = {
            "folded-serial": ledger_serial,
            "folded-batched": ledger_batched,
        }
        window_counts["folded-batched"][pi] = _window_forwards(bat_rec["window_calls"])
        exact = _block_exact_match(
            base_rec["sequences"], bat_rec["sequences"], padded_len, num_tokens
        )
        sidecar["quality"].append(
            {
                "prompt_index": pi,
                "block_exact_match": exact,
                "posthoc_confidence": {
                    "baseline": _posthoc_confidence(
                        model, base_rec["sequences"][0], padded_len, num_tokens
                    ),
                    "folded-batched": _posthoc_confidence(
                        model, bat_rec["sequences"][0], padded_len, num_tokens
                    ),
                },
            }
        )
        print(
            f"sidecar prompt {pi}: token_match={token_ok} "
            f"({logits_cmp['ref_forwards']} vs "
            f"{logits_cmp['live_forwards']} forwards), "
            f"logits allclose_all={logits_cmp['allclose_all']}, "
            f"exact_match={exact['overall_exact_match']}"
        )
    sidecar_ledger_all = {
        mode: bool(
            all(
                sidecar["ledger"][f"prompt{pi}"][mode]["within_bound"] for pi in range(len(prompts))
            )
        )
        for mode in ("folded-serial", "folded-batched")
    }

    # -- snapshots for timed-segment diffs ----------------------------------
    global _snapshot_events
    ctx_map = _ctx_mask_map(samplers)
    snap_serial = _snapshot_pool(mff_serial)
    snap_batched = _snapshot_pool(mff_batched)
    _snapshot_events = {
        id(mff_serial): (
            len(mff_serial.graph_adaptive_policy.events)
            if mff_serial.graph_adaptive_policy is not None
            else 0
        ),
        id(mff_batched): (
            len(mff_batched.graph_adaptive_policy.events)
            if mff_batched.graph_adaptive_policy is not None
            else 0
        ),
    }
    timed_ops_base = {
        "folded-serial": mff_serial.cache.write_op_count,
        "folded-batched": mff_batched.cache.write_op_count,
    }

    # -- timed reps: per-rep round-robin over the three modes ---------------
    whole_samples: dict[str, dict[int, list[float]]] = {
        m: {pi: [] for pi in range(len(prompts))} for m in _MODES
    }
    block_samples: dict[str, dict[int, dict[int, list[float]]]] = {
        m: {pi: {} for pi in range(len(prompts))} for m in _MODES
    }
    profiler_ratios: dict[str, list[float]] = {m: [] for m in _MODES}
    peaks: dict[str, list[float]] = {m: [] for m in _MODES}
    timed_window_calls: dict[str, list[tuple[int, int, str]]] = {
        "folded-serial": [],
        "folded-batched": [],
    }
    for rep in range(reps):
        order = _MODES[rep % 3 :] + _MODES[: rep % 3]
        for mode_name in order:
            mff = modes[mode_name]["mff"]
            torch.cuda.reset_peak_memory_stats()
            for pi, prompt in enumerate(prompts):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                rec = _generate(mode_name, prompt, modes, llm)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                whole_samples[mode_name][pi].append((t1 - t0) * 1000.0)
                for blk, ms in rec["blocks"]:
                    block_samples[mode_name][pi].setdefault(blk, []).append(ms)
                if mff is not None:
                    profiler_ratios[mode_name].extend(rec["stable_ratios"])
                    if mode_name in timed_window_calls:
                        timed_window_calls[mode_name].extend(rec["window_calls"])
            peaks[mode_name].append(round(torch.cuda.max_memory_allocated() / 1024**2, 1))
        summary = " | ".join(f"{m}: last={whole_samples[m][0][-1]:.1f}ms" for m in order)
        print(f"rep {rep + 1}/{reps}: {summary}")

    # -- statistics ----------------------------------------------------------
    def _pooled(mode: str) -> list[float]:
        return [x for pi in range(len(prompts)) for x in whole_samples[mode][pi]]

    whole_stats = {m: stats_from_samples(_pooled(m)) for m in _MODES}
    per_prompt_stats = {
        m: {
            f"prompt{pi}": stats_from_samples(whole_samples[m][pi]).as_dict()
            for pi in range(len(prompts))
        }
        for m in _MODES
    }
    block_stats: dict[str, Any] = {}
    for m in _MODES:
        merged: dict[int, list[float]] = {}
        for pi in range(len(prompts)):
            for blk, samples in block_samples[m][pi].items():
                merged.setdefault(blk, []).extend(samples)
        block_stats[m] = {
            f"block{blk}": stats_from_samples(samples).as_dict()
            for blk, samples in sorted(merged.items())
        }
    speedups = {
        "serial_vs_baseline": (
            whole_stats["baseline"].p50 / whole_stats["folded-serial"].p50
            if whole_stats["folded-serial"].p50 > 0
            else None
        ),
        "batched_vs_serial": (
            whole_stats["folded-serial"].p50 / whole_stats["folded-batched"].p50
            if whole_stats["folded-batched"].p50 > 0
            else None
        ),
        "batched_vs_baseline": (
            whole_stats["baseline"].p50 / whole_stats["folded-batched"].p50
            if whole_stats["folded-batched"].p50 > 0
            else None
        ),
    }

    # -- pool serialization + H4 faces ---------------------------------------
    windows_serial = _serialize_windows(mff_serial, snap_serial, ctx_map)
    windows_batched = _serialize_windows(mff_batched, snap_batched, ctx_map)
    for mode, windows in (
        ("folded-serial", windows_serial),
        ("folded-batched", windows_batched),
    ):
        sidecar_counts: dict[str, dict] = {}
        for pi in range(len(prompts)):
            for label, counts in window_counts[mode][pi].items():
                entry = sidecar_counts.setdefault(label, {"serial": 0, "batched": 0})
                entry["serial"] += counts["serial"]
                entry["batched"] += counts["batched"]
        windows["sidecar_forwards_per_window"] = sidecar_counts

    serial_scope = [
        label
        for label, counts in windows_serial["sidecar_forwards_per_window"].items()
        if counts["serial"] >= 2
    ]
    serial_pool = windows_serial["serial"]["windows"]
    serial_capture_ok = all(
        serial_pool.get(label, {}).get("captures_cumulative", 0) > 0 for label in serial_scope
    )
    serial_replay_ok = all(
        serial_pool.get(label, {}).get("replays_timed", 0) > 0 for label in serial_scope
    )
    batched_scope = [
        label
        for label, counts in windows_batched["sidecar_forwards_per_window"].items()
        if counts["batched"] >= 1
    ]
    batched_pool = windows_batched["batched"]["windows"]
    batched_capture_ok = all(
        batched_pool.get(label, {}).get("captures_cumulative", 0) > 0 for label in batched_scope
    )
    canary_serial = windows_serial["degradation_fallback_count"]["timed"] == 0
    canary_batched = windows_batched["degradation_fallback_count"]["timed"] == 0
    h4_checks = {
        "serial_scope_windows": serial_scope,
        "serial_capture_positive": serial_capture_ok,
        "serial_timed_replay_positive": serial_replay_ok,
        "batched_scope_windows": batched_scope,
        "batched_capture_positive": batched_capture_ok,
        "ledger_within_bound": sidecar_ledger_all,
        "canary_degradation_timed_zero": {
            "folded-serial": canary_serial,
            "folded-batched": canary_batched,
        },
    }
    ar017_timed = windows_batched["ar017_recapture_events"]["timed"]
    h4_checks["ar017_events"] = {
        "cumulative": windows_batched["ar017_recapture_events"]["cumulative"],
        "timed": ar017_timed,
        "status": "neutral (untriggered)" if ar017_timed == 0 else "exercised",
        "note": (
            "untriggered = neutral for H4 (never refutes); the policy records "
            "only successful re-captures after persistent misprovisioning, so "
            "a positive count means the trigger condition actually fired"
        ),
    }
    h4_pass = (
        bool(serial_scope)
        and serial_capture_ok
        and serial_replay_ok
        and bool(batched_scope)
        and batched_capture_ok
        and all(sidecar_ledger_all.values())
        and canary_serial
        and canary_batched
    )
    if not serial_scope or not batched_scope:
        h4_verdict = "inconclusive"
    else:
        h4_verdict = "supported" if h4_pass else "refuted"

    # -- H1 verdict -----------------------------------------------------------
    h1_real_ok = all(
        entry["token_match"] and entry["forward_counts_match"] and entry["logits"]["allclose_all"]
        for entry in sidecar["h1"]
    )
    h1_verdict = "supported" if h1_real_ok else "refuted"

    # -- H2 / H3 verdicts ------------------------------------------------------
    h2 = _verdict_p50(whole_stats["folded-serial"], whole_stats["baseline"])
    h3 = _verdict_p50(whole_stats["folded-batched"], whole_stats["folded-serial"])

    # -- verdict texts (frozen wording) ----------------------------------------
    hypotheses = {
        "H1": (
            "folded-serial generation is equivalent to baseline (synthetic "
            "UT face bit-exact, pinned by UT-2105; real face: token-level "
            "argmax agreement position-by-position (hard gate) + per-forward "
            "decision-slice logits allclose(rtol=1e-2, atol=1e-1) "
            "arbitration; bit-parity reported, not gating): supported iff "
            "the real face passes; else refuted (no inconclusive)"
        ),
        "H2": (
            "folded-serial whole-generation wall clock < baseline; "
            "sigma = max(sigma_base, sigma_serial); supported iff "
            "p50(base) - p50(serial) > 2*sigma; refuted iff the reverse "
            "exceeds 2*sigma; else inconclusive"
        ),
        "H3": (
            "folded-batched whole-generation wall clock < folded-serial "
            "(speed face, three-state as H2); quality face (per-block "
            "exact-match rate vs baseline + post-hoc confidence "
            "distribution) is report-only, no gate, no presupposition"
        ),
        "H4": (
            "full-stack production exercise: every in-scope shape window "
            "(>= 2 forwards within the window) has serial graph captures > 0 "
            "and timed replays > 0 (folded-serial); every window with a "
            "batched call has batched captures > 0 (folded-batched); publish "
            "ledger <= c*(L+N) with c=4 frozen (serial event L+1 ops, "
            "batched event O(L+N) ops); no-fallback canary (degradation "
            "class, timed segment == 0) all-true; AR017 events verified only "
            "when triggered (untriggered = neutral)"
        ),
    }

    stable_summary: dict[str, Any] = {}
    for m in _MODES:
        ratios = profiler_ratios[m]
        if ratios:
            stable_summary[m] = {
                "mean": round(sum(ratios) / len(ratios), 4),
                "std": round(
                    (sum((r - sum(ratios) / len(ratios)) ** 2 for r in ratios) / len(ratios))
                    ** 0.5,
                    4,
                ),
                "min": round(min(ratios), 4),
                "max": round(max(ratios), 4),
                "n": len(ratios),
            }
        else:
            stable_summary[m] = None

    vram_peaks = {m: {"max_over_reps_mb": max(peaks[m]), "per_rep_mb": peaks[m]} for m in _MODES}
    counts_supported = _cuda_profiler_supported()
    driver = _gpu_driver_version()
    import transformers

    timed_ops = {
        "folded-serial": mff_serial.cache.write_op_count - timed_ops_base["folded-serial"],
        "folded-batched": mff_batched.cache.write_op_count - timed_ops_base["folded-batched"],
    }

    artifact: dict[str, Any] = {
        "meta": {
            "quick": quick,
            "reps": reps,
            "seed": SEED,
            "prompts": {
                "count": len(prompts),
                "num_tokens": num_tokens,
                "shape_windows_W": num_tokens // BLOCK,
                "texts": [p["text"] for p in prompts],
                "raw_token_lens": [p["raw_len"] for p in prompts],
                "padded_token_lens": [p["padded_len"] for p in prompts],
                "prepad_note": (
                    "prompts pre-padded to exact 32-token multiples (pad=eos) "
                    "so the sampler's own first-block padding is zero and the "
                    "pinned fold-context mask is the pure block-causal mask, "
                    "element-equal to the remote model's internal eval_mask "
                    "(the raw path's [B,T] mask is overwritten internally by "
                    "the remote forward); non-zero padding would break "
                    "cross-mode parity by construction"
                ),
            },
            "block_constants": {
                "block_size": BLOCK,
                "small_block_size": BLOCK,
                "threshold": 0.95,
                "temperature": 0.0,
            },
            "warmup_strategy": (
                "one full discarded sample per (mode, prompt) before any "
                "timing: per-shape-window pool pre-capture (A5 scheme a: the "
                "warmup round naturally captures every window's serial and "
                "batched runners) + Triton/JIT warmup + the AR017 adaptive "
                "transition; capture costs are excluded from p50 (srs FR4); "
                "timed-segment re-captures, if any, are counted and timed "
                "honestly (never hidden)"
            ),
            "interleaving": (
                "per-rep round-robin over the three modes; rep r starts at "
                "position r%3 (anti-drift, WDDM fairness)"
            ),
            "timing_method": (
                "host perf_counter + torch.cuda.synchronize (#39); "
                "whole-generation wall clock plus per-block anchors (the "
                "forward wrappers observe the step_idx change as the block "
                "boundary; syncs only at those boundaries)"
            ),
            "mode_configs": {
                "baseline": {
                    "folded_model": None,
                    "num_block_candidates": 0,
                    "description": "raw FastDLLMSampler (unchanged recipe)",
                },
                "folded-serial": {
                    "mff_kwargs": {
                        "use_cuda_graph": True,
                        "graph_runner_pool_size": POOL_SIZE,
                        "split_layers": False,
                        "graph_capacity_ratio": CAPACITY_RATIO,
                        "graph_adaptive_capacity": False,
                        "scheduler": None,
                        "gate": f"SimilarityGate(tau={TAU}, cosine, eps={EPS})",
                        "cache": f"ActivationCache({CACHE_BUDGET} tokens/layer)",
                    },
                    "sampler": {"num_block_candidates": 0},
                    "description": (
                        "FR2 serial chain folding (#26) with the pinned "
                        "fold-context trio and the shape-keyed serial pool"
                    ),
                },
                "folded-batched": {
                    "mff_kwargs": {
                        "use_cuda_graph": True,
                        "graph_runner_pool_size": POOL_SIZE,
                        "split_layers": False,
                        "graph_capacity_ratio": CAPACITY_RATIO,
                        "graph_adaptive_capacity": True,
                        "graph_adaptive_config": "AdaptiveCapacityConfig()",
                        "scheduler": None,
                        "gate": f"SimilarityGate(tau={TAU}, cosine, eps={EPS})",
                        "cache": f"ActivationCache({CACHE_BUDGET} tokens/layer)",
                    },
                    "sampler": {
                        "num_block_candidates": NUM_CANDIDATES,
                        "candidate_top_k": CANDIDATE_TOP_K,
                    },
                    "description": (
                        "FR3 batched speculative block verification (N=4 "
                        "from the AR019 artifact matrix, candidate_top_k=10, "
                        "A5 scheme a); AR017 adaptive capacity engaged on "
                        "the batched path only (#37); split_layers=False on "
                        "both MFF instances (Split-layer resident hooks "
                        "would stack on the shared model)"
                    ),
                },
            },
            "forced_divergent_declaration": (
                "the sampler pins a per-(T, prompt_len) forced base tensor "
                "whose True positions cover exactly the active block; every "
                "folded forward (denoise iterations, the block-advance AR "
                "step, and the [N,T] batched verification) threads it "
                "through ManualFoldedForward(forced_divergent=...); the "
                "freshly committed / actively denoised region is always "
                "recomputed, never gate-certified stable against a stale "
                "parent"
            ),
            "shape_keyed_pool_declaration": (
                "graph_runner_pool_size=8: one runner per (tokens shape, "
                "pinned mask identity) window == (T, prompt_len) serial / "
                "(N, T, prompt_len) batched; expected entries <= 8 per pool "
                "(W*P upper bound; folded-batched serial (W-1)*P, batched "
                "W*P; folded-serial <= W*P data-dependent), zero evictions "
                "at cap=8 (LRU is a defensive bound only)"
            ),
            "use_cuda_graph": True,
            "a5_scheme_a_note": (
                "warmup pre-capture amortization holds under the bench "
                "protocol (every timed rep replays each window's runners); "
                "in single-shot production generation the capture amortizes "
                "NEGATIVE (one capture ~= 2-3 forwards serving a single "
                "replay) -- recommend use_cuda_graph=False for single-shot "
                "generate scenarios"
            ),
            "pool_cap": POOL_SIZE,
            "cache_budget_tokens_per_layer": CACHE_BUDGET,
            "ledger_c": LEDGER_C,
            "ledger_normalization": (
                "per publish event: a serial forward publishes 1 event "
                "counting L+1 ops (<= 4*(L+1), trivially in bound); a "
                "batched publish counts O(L+N) ops (put_batch foreach groups "
                "+ immediate executions, <= 4*(L+N)); the aggregate over a "
                "generation preserves the bounds (AR020-style, c frozen at "
                "design); verified on the untimed sidecar generation (same "
                "deterministic trajectory as the timed reps)"
            ),
            "sigma_convention": (
                "sigma = max(sigma_a, sigma_b) bound to the mode pair; "
                "three-state mapping per srs FR4; samples pooled over "
                "prompts x reps per mode"
            ),
            "h4_scope_note": (
                "the serial capture/replay sub-criterion is verified only on "
                "windows with >= 2 forwards in the sidecar generation (the "
                "window-first forward is structural eager by design: "
                "parent-None populate or parent-at-old-shape window switch); "
                "the last window (remaining=0, no AR step) and "
                "single-iteration windows are reported but out of scope"
            ),
            "canary_note": (
                "the no-fallback canary checks the DEGRADATION class only "
                "(timed segment == 0): every non-structural graph-path eager "
                "fallback; structural eager (window-populate, window-switch, "
                "last-window no-AR) is counted separately and excluded by "
                "design (else the canary is constructively false)"
            ),
            "vram_peak_mb_per_mode": vram_peaks,
            "device": torch.cuda.get_device_name(0),
            "gpu_driver_version": driver,
            "gpu_driver_version_note": (
                "queried via nvidia-smi; null means the query failed on "
                "this host and the value was omitted rather than fabricated"
            ),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "platform_note": (
                "Windows WDDM display-driver mode (no clock locking); noise "
                "mitigated by warmup + reps + p50 + per-rep round-robin"
            ),
            "kernel_count": None,
            "kernel_count_note": (
                "runtime CUPTI probe: this torch build cannot record CUDA "
                "profiler events (LIBKINETO_NOCUPTI); kernel-launch counts "
                "recorded null + note, never fabricated (#39) - the "
                "write_op_count ledger and graph_pool_stats counters are the "
                "structural evidence"
                if not counts_supported
                else "CUPTI available; wall-clock remains the primary metric"
            ),
            "single_run_discipline": (
                "the full (reps=30) run is a pre-committed single-shot "
                "verdict, no rerun fishing (#39 continuation); this "
                "artifact is the --quick smoke run (separate path, never "
                "the formal artifact)"
                if quick
                else "pre-committed single-shot verdict, no rerun fishing"
            ),
            "model": "Fast-dLLM-v2-1.5B (Efficient-Large-Model, remote code)",
            "model_details": model_meta,
            "venv": sys.prefix != sys.base_prefix,
            "run_date_utc": datetime.now(timezone.utc).isoformat(),
            "command": (
                "python -m scripts.ar021_generation_loop_bench"
                + (" --quick" if quick else f" --reps {reps}")
                + f" --model-path {model_path}"
            ),
        },
        "hypotheses_declared": hypotheses,
        "hypotheses_verdicts": {
            "H1": {
                "hypothesis": hypotheses["H1"],
                "verdict": h1_verdict,
                "numbers": {
                    "per_prompt": sidecar["h1"],
                    "synthetic_face": ("pinned bit-exact by UT-2105 (not re-measured here)"),
                },
            },
            "H2": {
                "hypothesis": hypotheses["H2"],
                "verdict": h2["verdict"],
                "numbers": {
                    "p50_baseline_ms": h2["p50_slower_ms"],
                    "p50_folded_serial_ms": h2["p50_faster_ms"],
                    "delta_ms": h2["delta_ms"],
                    "sigma_ms": h2["sigma_ms"],
                },
            },
            "H3": {
                "hypothesis": hypotheses["H3"],
                "verdict": h3["verdict"],
                "verdict_quality_face": "report-only (no gate)",
                "numbers": {
                    "p50_folded_serial_ms": h3["p50_slower_ms"],
                    "p50_folded_batched_ms": h3["p50_faster_ms"],
                    "delta_ms": h3["delta_ms"],
                    "sigma_ms": h3["sigma_ms"],
                    "quality": sidecar["quality"],
                },
            },
            "H4": {
                "hypothesis": hypotheses["H4"],
                "verdict": h4_verdict,
                "numbers": h4_checks,
            },
        },
        "mode_comparison": {
            m: {
                "whole_gen_ms": whole_stats[m].as_dict(),
                "per_prompt_ms": per_prompt_stats[m],
                "per_block_ms": block_stats[m],
                "stable_ratio_timed": stable_summary[m],
            }
            for m in _MODES
        },
        "speedups_p50": {k: (round(v, 4) if v is not None else None) for k, v in speedups.items()},
        "per_shape_window": {
            "folded-serial": windows_serial,
            "folded-batched": windows_batched,
        },
        "quality": {
            "h1_serial_vs_baseline": sidecar["h1"],
            "h3_batched_vs_baseline": sidecar["quality"],
        },
        "ledger": {
            "c": LEDGER_C,
            "normalization": (
                "ops <= c*((L+1)*serial_events + (L+N)*batched_events) per "
                "sidecar generation; L = num_hidden_layers, N = "
                f"{NUM_CANDIDATES}"
            ),
            "num_layers": num_layers,
            "per_prompt": sidecar["ledger"],
            "within_bound": sidecar_ledger_all,
            "timed_segment_ops_total": timed_ops,
        },
        "batch_canary": {
            "folded-serial": {
                "timed_degradation_fallbacks": (
                    windows_serial["degradation_fallback_count"]["timed"]
                ),
                "cumulative_degradation_fallbacks": (
                    windows_serial["degradation_fallback_count"]["cumulative"]
                ),
                "cumulative_structural_eager": (
                    windows_serial["structural_eager_count"]["cumulative"]
                ),
                "timed_structural_eager": (windows_serial["structural_eager_count"]["timed"]),
                "pass": canary_serial,
            },
            "folded-batched": {
                "timed_degradation_fallbacks": (
                    windows_batched["degradation_fallback_count"]["timed"]
                ),
                "cumulative_degradation_fallbacks": (
                    windows_batched["degradation_fallback_count"]["cumulative"]
                ),
                "cumulative_structural_eager": (
                    windows_batched["structural_eager_count"]["cumulative"]
                ),
                "timed_structural_eager": (windows_batched["structural_eager_count"]["timed"]),
                "pass": canary_batched,
            },
            "pass": bool(canary_serial and canary_batched),
            "note": (
                "canary = degradation class only, timed segment == 0; "
                "structural eager (by-design window populate/switch) is "
                "counted separately and never fails the canary (AR020-style)"
            ),
        },
    }
    return artifact


def write_results(data: dict[str, Any], path: str | Path) -> Path:
    """Write the benchmark record as JSON, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _resolve_model_path(cli_value: str | None) -> Path:
    """Resolve the checkpoint path from the CLI flag or the env fallback.

    Raises:
        FileNotFoundError: If neither the flag nor ``FASTDLLM_MODEL_PATH``
            yields a path (AGENTS #10: a real-checkpoint benchmark never
            silently falls back to a synthetic model).
    """
    value = cli_value or os.environ.get(DEFAULT_MODEL_ENV)
    if not value:
        raise FileNotFoundError(
            "no model path given: pass --model-path or set "
            f"{DEFAULT_MODEL_ENV} (no synthetic fallback, AGENTS #10)"
        )
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"model directory not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path (default: the formal artifact for full runs, "
            f"the separate quick path {QUICK_OUTPUT_PATH} for --quick)"
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Local checkpoint directory (env fallback: {DEFAULT_MODEL_ENV}).",
    )
    parser.add_argument("--reps", type=int, default=30, help="Timed reps")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke subset: reps=3, single prompt, num_tokens=64 (2 blocks)",
    )
    add_hf_env_arguments(parser)
    args = parser.parse_args()

    quick = args.quick
    reps = 3 if quick else args.reps
    out_path = args.out or (QUICK_OUTPUT_PATH if quick else DEFAULT_OUTPUT_PATH)
    model_path = _resolve_model_path(args.model_path)
    data = run_bench(model_path, reps=reps, quick=quick)
    out = write_results(data, out_path)
    print()
    for key in ("H1", "H2", "H3", "H4"):
        entry = data["hypotheses_verdicts"][key]
        verdict = entry.get("verdict", entry)
        print(f"{key}: {verdict}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
