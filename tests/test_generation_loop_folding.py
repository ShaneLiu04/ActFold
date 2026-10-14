"""Red-phase tests for AR021: production generation-loop folding (UT-2101..UT-2112).

Contract: ``specs/changes/AR021-generation-loop-folding/design.md`` §4.2.2/§4.3/§6.1
and ``srs.md`` §3.1/§3.2/§3.3 — wiring the AR015-020 batched-verification stack into
the real production generation loop (``FastDLLMSampler.sample``):

* FR1: a shape-keyed ``(T, prompt_len)`` fold context (``_fold_ctx_for``) supplying
  the pinned block-causal-and-non-pad ``[T, T]`` mask, the rotary
  ``position_embeddings`` (or ``None``), and the ``forced`` base tensor — zero
  device readback, object-pinned within a shape window and across ``sample`` calls;
* FR2: ``forced_divergent`` threading (``ManualFoldedForward.forward`` ->
  ``FoldedTransformerLayer`` gate paths -> runner ``forced_divergent_static``
  buffer) and the opt-in shape-keyed graph runner pool (``graph_runner_pool_size``
  / ``graph_pool_stats`` / cache-setter pool propagation / LRU + VRAM conservation);
* FR3: opt-in batched speculative block verification
  (``FastDLLMSamplerConfig.num_block_candidates`` / ``candidate_top_k``) with the
  frozen per-candidate seed formula, min-prob selection, chain continuation onto
  the selected candidate branch, and stop-token semantics identical to baseline.

UT -> test function mapping (one UT may map to several functions):

===== ========================================== ================================
UT    Function(s)                                Asserted points
===== ========================================== ================================
2101  test_ut2101_fold_ctx_mask_semantics        block-causal AND non-pad mask from
      _and_pinning                               host pad metadata; raw
                                                 ``x != pad_id`` equivalence;
                                                 (T, prompt_len) object pinning;
                                                 same-T different-prompt_len key
                                                 separation; forced base tensor;
                                                 raw-path zero disturbance.
2102  test_ut2102_pe_supply_and_batch_guard          rotary -> pinned (cos, sin); no
      test_ut2102_pe_dtype_follows_model            rotary -> None; ``_forward``
                                                  position_embeddings tail param;
                                                  folded B>1 -> ValueError; the
                                                  rotary dummy follows the wrapped
                                                  model's dtype (bf16 checkpoint
                                                  face, T003 smoke).
2103  test_ut2103_forced_none_zero_disturbance   forced_divergent=None explicit ==
                                                 omitted, both == raw model
                                                 (bit-exact, non-split path).
2104  test_ut2104_eager_forced_semantics         forced bits always recomputed
      test_ut2104_fused_forced_count             (profiler consumes the AND-ed
                                                 mask); [T]/[1,T]/[N,T]
                                                 broadcasts; fused-gate path
                                                 re-derives the count (no stale
                                                 all-stable fast path).
2105  test_ut2105_folded_serial_parity_dual_form dual-form mask contract (raw
                                                 [B,T] internal build vs folded
                                                 [T,T] prebuilt, ONE attention
                                                 implementation); folded-serial
                                                 generation bit-exact vs raw;
                                                 every folded call receives the
                                                 pinned [T,T] mask + forced;
                                                 prefix-reuse and active-block
                                                 recompute probes.
2106  test_ut2106_config_pool_size_validation    ActFoldConfig.graph_runner_pool_
      test_ut2106_pool_multi_shape_windows       size (>=0, ValueError below 0);
      test_ut2106_lru_eviction_vram              pool on: >=2 shape windows each
      test_ut2106_cache_setter_pool_propagation  capture one runner, replays > 0,
      test_ut2106_batched_pool_key               zero UserWarning; LRU eviction
                                                 with VRAM conservation; batched
                                                 pool keys (N, T, prompt_len);
                                                 cache setter reaches every pool
                                                 runner incl. non-active members.
2107  test_ut2107_graph_forced_buffer            capture with forced -> replay;
                                                 changed forced content changes
                                                 the recomputed region (device
                                                 count probe); None/not-None
                                                 identity mismatch -> silent
                                                 eager both directions; forced
                                                 bits count into the divergent
                                                 budget.
2108  test_ut2108_num_candidates_zero_zero_      num_block_candidates=0 (default)
      disturbance                                bit-exact vs the existing
      test_ut2108_graph_mode_zero_disturbance    sampler (raw, folded eager,
                                                 folded graph, stop-token
                                                 scenario).
2109  test_ut2109_batched_speculative_flow       exactly one [N,T] batched call
      test_ut2109_selection_and_stop_semantics   per small block; batched publish
                                                 (per-branch [1,T] cache entries
                                                 at batch 1); candidates within
                                                 the top-k draft set; same-seed
                                                 reproducibility; min-prob argmax
                                                 selection (adversarial
                                                 two-spike model); chain parent
                                                 switches to the selected
                                                 candidate; stop-token semantics
                                                 == baseline.
2110  test_ut2110_fr3_constructor_validation    N<0 / candidate_top_k<1 / N>0
                                                 without folded_model / a
                                                 folded_model without branch_ids
                                                 support -> ValueError (no
                                                 silent degradation).
2111  test_ut2111_ar_step_forced                 AR-step forwards carry forced =
      test_ut2111_cross_block_runner_capture     current block (recompute probe =
                                                 block token count); cross-block
                                                 growth captures a new pool
                                                 runner per shape window.
2112  test_ut2112_ctx_supply_discipline          ctx survives across sample calls
                                                 (same (T, prompt_len) -> same
                                                 objects); same-T different-
                                                 prompt_len key separation; zero
                                                 device readback (.item/.tolist/
                                                 .cpu) during construction.
===== ========================================== ================================

Red state (why every test fails today) — each failure is a production-missing
face, never a test-own bug:

* ``FastDLLMSampler._fold_ctx_for`` does not exist (AttributeError: UT-2101,
  UT-2102, UT-2112);
* ``DiffusionSampler._forward`` has no ``position_embeddings`` / ``branch_ids`` /
  ``forced_divergent`` tail params and ``FastDLLMSampler.sample`` never threads a
  ``[T, T]`` mask nor ``forced_divergent`` (TypeError / assertion failures:
  UT-2102, UT-2104, UT-2105, UT-2107, UT-2111);
* ``ManualFoldedForward.forward`` has no ``forced_divergent`` kwarg (TypeError:
  UT-2103, UT-2104, UT-2107);
* ``ManualFoldedForward.__init__`` has no ``graph_runner_pool_size`` kwarg and no
  ``graph_pool_stats`` attribute (TypeError / AttributeError: UT-2106, UT-2111);
* ``FastDLLMSamplerConfig`` has no ``num_block_candidates`` / ``candidate_top_k``
  fields and no construction-time validation (TypeError: UT-2108, UT-2109,
  UT-2110);
* ``ActFoldConfig`` has no ``graph_runner_pool_size`` field (TypeError: UT-2106).

Zero-disturbance sub-assertions that already hold today (raw-path mask form,
bit-exact eager baselines, the dual-form mask equivalence) run FIRST inside their
UTs and must stay green after the Green phase; each UT as a whole is red through
the missing-face assertion listed above.

``graph_pool_stats`` schema frozen by these UTs (design §4.1.2 observation face):
``{"serial": {<shape key>: {"captures", "replays", "evictions"}, "pool_size"},
"batched": {same layout}, "degradation_fallback_count", "structural_eager_count"}``
— pool sections are empty dicts when the pool is off, the two eager counters are
instance-cumulative ints.

Fixtures follow ``tests/test_batched_graph_forward.py`` (self-contained
LLaMA-layout decoder, L=2, H=32, V=64, SimilarityGate(tau=0.95, cosine), no
scheduler) with two production-shaped extensions required by design §6.1: the
attention accepts BOTH mask forms (raw ``[B, T]`` padding form building the
block-causal AND pad mask internally, and the folded ``[T, T]`` prebuilt form)
through ONE implementation, and an optional ``model.model.rotary_emb`` drills the
AR019 pe supply. The autouse ``seed`` fixture plus explicit builder/config seeds
pin every random face. CUDA-only semantics (graph capture, pools, VRAM) are
gated per-function via ``_cuda``; the autouse ``_profiler_state`` fixture resets
the global stability profiler around every case. Bit-exact faces use
``split_layers=False`` (pitfall #41); graph faces use ``split_layers=True`` with
``split_min_tokens=1`` (test_cuda_graph.py precedent).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.cuda_graph import FoldedGraphRunner
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.models.base import DiffusionLLM
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig
from actfold.models.sampling_utils import right_shift_logits
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from actfold.utils.config_manager import ActFoldConfig

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_VOCAB = 64
_HIDDEN = 32
_NUM_LAYERS = 2
_TAU = 0.95
_CACHE_BUDGET = 1024
_MASK_ID = 63
_STOP_ID = 62
_PAD_ID = 61
_SPIKE_TOP1_ID = 5
_SPIKE_TOP2_ID = 6
_BLOCK_SIZE = 8
_SMALL_BLOCK_SIZE = 4
_MODEL_SEED = 2026


@pytest.fixture(autouse=True)
def _profiler_state() -> Iterator[None]:
    """Snapshot-reset the global stability profiler around every case."""
    GLOBAL_STABILITY_PROFILER.reset()
    yield
    GLOBAL_STABILITY_PROFILER.reset()


# ---------------------------------------------------------------------------
# Self-contained mini block-causal decoder with the dual-form mask contract
# ---------------------------------------------------------------------------


class _DualFormAttention(nn.Module):
    """One attention implementation accepting both production mask forms.

    Raw form: ``[B, T]`` padding mask — the block-causal AND non-pad mask is
    built internally (the raw sampler path semantics). Folded form: ``[T, T]``
    prebuilt block-causal AND non-pad mask (the FR1 fold-context semantics).
    Both forms must be semantically equivalent for the same canvas (UT-2105).
    """

    def __init__(self, hidden: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, None]:
        del position_embeddings
        batch, seq_len, hidden = hidden_states.shape
        attn_mask: torch.Tensor | None = None
        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape[1] != seq_len:
                raise ValueError(
                    f"unsupported attention mask shape {tuple(attention_mask.shape)} "
                    f"for B={batch}, T={seq_len}"
                )
            am = attention_mask.to(torch.bool)
            if am.shape[0] == batch and batch != seq_len:
                q_idx = torch.arange(seq_len, device=hidden_states.device)
                q_idx = q_idx // self.block_size
                causal = q_idx[:, None] >= q_idx[None, :]
                attn_mask = causal.unsqueeze(0) & am[:, None, :]
            elif tuple(am.shape) == (seq_len, seq_len):
                attn_mask = am.unsqueeze(0)
            else:
                raise ValueError(
                    f"unsupported attention mask shape {tuple(am.shape)} "
                    f"for B={batch}, T={seq_len}"
                )
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch, seq_len, hidden)
        return self.out_proj(out), None


class _DualFormLayer(nn.Module):
    """Pre-norm decoder layer with a detectable split FFN chain."""

    def __init__(self, hidden: int, block_size: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = _DualFormAttention(hidden, num_heads=4, block_size=block_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class _MiniRotary(nn.Module):
    """Minimal HF-style rotary: ``(cos, sin)`` of shape ``[1, T, head_dim]``."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = position_ids.to(torch.float32)
        exponent = torch.arange(0, self.head_dim, 2, device=pos.device).float()
        inv_freq = 1.0 / (10000.0 ** (exponent / self.head_dim))
        freqs = pos[..., None] * inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).to(hidden_states.dtype)
        sin = emb.sin().unsqueeze(0).to(hidden_states.dtype)
        return cos, sin


class _SpikeHead(nn.Module):
    """Head with two clearly ranked spikes: top-1 and top-2 known probabilities."""

    def __init__(self, vocab: int, top1_id: int, top2_id: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.top1_id = top1_id
        self.top2_id = top2_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*x.shape[:-1], self.vocab, dtype=x.dtype, device=x.device)
        logits[..., self.top1_id] = 3.0
        logits[..., self.top2_id] = 1.0
        return logits


class _ConstantHead(nn.Module):
    """Head whose argmax is a fixed token at every position (stop-token scenarios)."""

    def __init__(self, vocab: int, token_id: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.token_id = token_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*x.shape[:-1], self.vocab, dtype=x.dtype, device=x.device)
        logits[..., self.token_id] = 1.0
        return logits


class _MiniBlockCausalModel(nn.Module):
    """Mock LLaMA-layout decoder: embed -> layers -> model.norm -> lm_head."""

    def __init__(
        self,
        block_size: int = _BLOCK_SIZE,
        with_rotary: bool = False,
        head: str = "linear",
    ) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": _VOCAB})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.model.layers = nn.ModuleList(
            [_DualFormLayer(_HIDDEN, block_size) for _ in range(_NUM_LAYERS)]
        )
        self.model.norm = nn.LayerNorm(_HIDDEN)
        if with_rotary:
            self.model.rotary_emb = _MiniRotary(_HIDDEN)
        if head == "spike":
            self.lm_head = _SpikeHead(_VOCAB, _SPIKE_TOP1_ID, _SPIKE_TOP2_ID)
        elif head == "constant":
            self.lm_head = _ConstantHead(_VOCAB, _STOP_ID)
        else:
            self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.lm_head(self.model.norm(x))


class _MiniTokenizer:
    """Minimal tokenizer exposing the special ids the samplers resolve."""

    def __init__(self, mask_id: int, stop_id: int, pad_id: int) -> None:
        self.mask_token_id = mask_id
        self.eos_token_id = stop_id
        self.bos_token_id = 1
        self.pad_token_id = pad_id


class _MiniDiffusionLLM(DiffusionLLM):
    """DiffusionLLM wrapper around the mini decoder, recording raw forwards."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__("mini-block-causal")
        self.model = inner
        self.tokenizer = _MiniTokenizer(_MASK_ID, _STOP_ID, _PAD_ID)
        self.raw_calls: list[dict[str, Any]] = []

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.raw_calls.append(
            {
                "tokens": tokens.detach().clone(),
                "mask": (attention_mask.detach().clone() if attention_mask is not None else None),
                "kwargs": set(kwargs),
            }
        )
        return self.model(tokens, attention_mask=attention_mask)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(tokens)

    @property
    def num_layers(self) -> int:
        return _NUM_LAYERS

    @property
    def hidden_dim(self) -> int:
        return _HIDDEN

    @property
    def num_heads(self) -> int:
        return 4

    @property
    def vocab_size(self) -> int:
        return _VOCAB


class _SpyMFF(ManualFoldedForward):
    """ManualFoldedForward that records every forward call and its output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict[str, Any]] = []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = super().forward(*args, **kwargs)
        self.calls.append({"args": args, "kwargs": kwargs, "out": out})
        return out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_mini(
    device: str,
    with_rotary: bool = False,
    head: str = "linear",
) -> _MiniBlockCausalModel:
    """Deterministically build the mini decoder on ``device``."""
    torch.manual_seed(_MODEL_SEED)
    model = _MiniBlockCausalModel(with_rotary=with_rotary, head=head)
    model.to(device)
    model.eval()
    return model


def _make_llm(inner: nn.Module) -> _MiniDiffusionLLM:
    return _MiniDiffusionLLM(inner)


def _sampler_config(**overrides: Any) -> FastDLLMSamplerConfig:
    """FastDLLMSamplerConfig with mini-model token ids and pinned defaults."""
    values: dict[str, Any] = {
        "num_tokens": 8,
        "temperature": 0.0,
        "block_size": _BLOCK_SIZE,
        "small_block_size": _SMALL_BLOCK_SIZE,
        "threshold": 0.95,
        "mask_token_id": _MASK_ID,
        "stop_token_id": _STOP_ID,
        "pad_token_id": _PAD_ID,
        "seed": 2100,
    }
    values.update(overrides)
    return FastDLLMSamplerConfig(**values)


def _make_mff(
    model: nn.Module,
    device: str,
    cls: type[ManualFoldedForward] = ManualFoldedForward,
    use_cuda_graph: bool = False,
    graph_capacity_ratio: float = 0.5,
    split_layers: bool = False,
    **extra: Any,
) -> ManualFoldedForward:
    """Fresh ManualFoldedForward (own cache + cosine gate) over the shared model."""
    cache = extra.pop("cache", None)
    if cache is None:
        cache = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    gate = extra.pop("gate", None)
    if gate is None:
        gate = SimilarityGate(tau=_TAU, metric="cosine")
    return cls(
        model,
        cache=cache,
        gate=gate,
        scheduler=None,
        split_layers=split_layers,
        split_min_tokens=1 if split_layers else 512,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=graph_capacity_ratio,
        **extra,
    )


def _make_canvas(prompt: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int, int]:
    """Pad the prompt to block alignment and append one mask block."""
    prompt_len = prompt.shape[1]
    fbp = (block_size - prompt_len % block_size) % block_size
    pad = torch.full((1, fbp), _PAD_ID, dtype=torch.long, device=prompt.device)
    masks = torch.full((1, block_size), _MASK_ID, dtype=torch.long, device=prompt.device)
    return torch.cat([prompt, pad, masks], dim=1), prompt_len, fbp


def _fold_mask(
    seq_len: int,
    prompt_len: int,
    first_block_padding: int,
    block_size: int,
    device: str,
) -> torch.Tensor:
    """Expected fold mask: block-causal AND non-pad from host pad metadata."""
    q_idx = torch.arange(seq_len) // block_size
    causal = q_idx[:, None] >= q_idx[None, :]
    non_pad = torch.ones(seq_len, dtype=torch.bool)
    non_pad[prompt_len : prompt_len + first_block_padding] = False
    return (causal & non_pad.unsqueeze(0)).to(device)


def _user_warning_messages(record: Any) -> list[str]:
    """Return the messages of every UserWarning in a catch_warnings log."""
    return [str(entry.message) for entry in record if issubclass(entry.category, UserWarning)]


def _prompt(device: str) -> torch.Tensor:
    return torch.tensor([[1, 2, 3]], device=device)


# ---------------------------------------------------------------------------
# UT-2101 (FR1): fold-context mask semantics, pinning, key separation
# ---------------------------------------------------------------------------


def test_ut2101_fold_ctx_mask_semantics_and_pinning(device: str) -> None:
    """UT-2101 (FR1): block-causal AND non-pad mask from host pad metadata.

    The raw path stays byte-identical (padding-form ``x != pad_id`` masks, no
    ``position_embeddings``); the folded context mask is ``[T, T]``, equals the
    block-causal AND non-pad expectation built from ``(prompt_len,
    first_block_padding)`` and the raw ``x != pad_id`` equivalence on normal
    inputs; the same ``(T, prompt_len)`` pins the same objects; the same ``T``
    with a different ``prompt_len`` lands in a different entry with different
    content; the ``forced`` base tensor marks exactly the current block.
    """
    inner = _build_mini(device)
    llm = _make_llm(inner)
    sampler = FastDLLMSampler(llm, config=_sampler_config(seed=2101))
    out_raw = sampler.sample(_prompt(device))
    assert out_raw.sequences.shape[0] == 1
    assert llm.raw_calls, "raw path must issue model forwards"
    for call in llm.raw_calls:
        mask = call["mask"]
        tokens = call["tokens"]
        assert mask is not None
        assert mask.shape == tokens.shape
        assert torch.equal(mask.to(torch.bool), tokens != _PAD_ID)
        assert "position_embeddings" not in call["kwargs"]

    x, prompt_len, fbp = _make_canvas(_prompt(device), _BLOCK_SIZE)
    seq_len = x.shape[1]
    ctx = sampler._fold_ctx_for(x, prompt_len=prompt_len)
    assert ctx.mask.shape == (seq_len, seq_len)
    assert ctx.mask.dtype == torch.bool
    expected = _fold_mask(seq_len, prompt_len, fbp, _BLOCK_SIZE, device)
    assert torch.equal(ctx.mask, expected)
    raw_kv = x[0] != _PAD_ID
    q_idx = torch.arange(seq_len) // _BLOCK_SIZE
    causal = q_idx[:, None] >= q_idx[None, :]
    assert torch.equal(ctx.mask, (causal.to(device) & raw_kv.unsqueeze(0)))

    ctx_again = sampler._fold_ctx_for(x, prompt_len=prompt_len)
    assert ctx_again.mask is ctx.mask
    assert ctx_again.pe is ctx.pe
    assert ctx_again.forced is ctx.forced

    assert ctx.forced.shape == (seq_len,)
    assert ctx.forced.dtype == torch.bool
    assert not bool(ctx.forced[: seq_len - _BLOCK_SIZE].any())
    assert bool(ctx.forced[seq_len - _BLOCK_SIZE :].all())

    prompt_b = torch.randint(1, 10, (1, 8), device=device)
    x_b = torch.cat(
        [prompt_b, torch.full((1, 8), _MASK_ID, dtype=torch.long, device=device)],
        dim=1,
    )
    assert x_b.shape[1] == seq_len
    ctx_b = sampler._fold_ctx_for(x_b, prompt_len=8)
    assert ctx_b.mask is not ctx.mask
    assert not torch.equal(ctx_b.mask, ctx.mask)


# ---------------------------------------------------------------------------
# UT-2102 (FR1): pe supply and the folded B>1 guard
# ---------------------------------------------------------------------------


def test_ut2102_pe_supply_and_batch_guard(device: str) -> None:
    """UT-2102 (FR1): rotary models get a pinned (cos, sin); plain models None.

    The pe threads through ``DiffusionSampler._forward``'s new tail param into
    the folded model; a folded ``sample`` with batch > 1 must raise ValueError
    (the ``[T, T]`` mask cannot express per-row padding).
    """
    inner = _build_mini(device, with_rotary=True)
    llm = _make_llm(inner)
    sampler = FastDLLMSampler(llm, config=_sampler_config(seed=2102))
    x, prompt_len, _fbp = _make_canvas(_prompt(device), _BLOCK_SIZE)
    ctx = sampler._fold_ctx_for(x, prompt_len=prompt_len)
    assert ctx.pe is not None
    cos, sin = ctx.pe
    assert isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor)
    assert cos.shape[-2] == x.shape[1]
    assert sin.shape[-2] == x.shape[1]
    ctx_again = sampler._fold_ctx_for(x, prompt_len=prompt_len)
    assert ctx_again.pe is ctx.pe

    mff = _make_mff(inner, device)
    with torch.no_grad():
        out = sampler._forward(
            x,
            branch_id="2102-b",
            parent_branch_id=None,
            folded_model=mff,
            step_idx=0,
            attention_mask=ctx.mask,
            position_embeddings=ctx.pe,
        )
    assert tuple(out.shape) == (1, x.shape[1], _VOCAB)

    plain = _build_mini(device)
    llm_plain = _make_llm(plain)
    sampler_plain = FastDLLMSampler(llm_plain, config=_sampler_config(seed=2102))
    ctx_plain = sampler_plain._fold_ctx_for(x, prompt_len=prompt_len)
    assert ctx_plain.pe is None

    mff_plain = _make_mff(plain, device)
    with pytest.raises(ValueError):
        sampler_plain.sample(
            torch.tensor([[1, 2, 3], [4, 5, 6]], device=device),
            folded_model=mff_plain,
        )


def test_ut2102_pe_dtype_follows_model() -> None:
    """UT-2102 (FR1, real-checkpoint face): the rotary dummy carries the model dtype.

    Remote rotary modules return ``cos.to(dtype=dummy.dtype)`` while the raw
    forward feeds them real bf16 hidden states, so an fp32 dummy on a bf16
    checkpoint yields fp32 cos/sin whose rotary application promotes q/k
    against a bf16 v and breaks the attention dtype contract (RuntimeError
    on Fast-dLLM-v2, AR021 T003 smoke). The supply must follow the wrapped
    embedding weight's dtype.
    """
    device = "cpu"
    inner_bf16 = _build_mini(device, with_rotary=True).to(torch.bfloat16)
    sampler_bf16 = FastDLLMSampler(_make_llm(inner_bf16), config=_sampler_config(seed=2102))
    pe = sampler_bf16._resolve_position_embeddings(16, torch.device(device))
    assert pe is not None
    assert pe[0].dtype == torch.bfloat16
    assert pe[1].dtype == torch.bfloat16

    inner_fp32 = _build_mini(device, with_rotary=True)
    sampler_fp32 = FastDLLMSampler(_make_llm(inner_fp32), config=_sampler_config(seed=2102))
    pe32 = sampler_fp32._resolve_position_embeddings(16, torch.device(device))
    assert pe32 is not None
    assert pe32[0].dtype == torch.float32
    assert pe32[1].dtype == torch.float32


# ---------------------------------------------------------------------------
# UT-2103 (FR2): forced_divergent=None zero disturbance
# ---------------------------------------------------------------------------


def test_ut2103_forced_none_zero_disturbance(device: str) -> None:
    """UT-2103 (FR2): explicit forced_divergent=None == omitted == raw model.

    The no-forced baselines (first call vs the raw model, chained all-stable
    reuse call) are bit-exact today and must stay so; the explicit ``None``
    kwarg must be accepted with byte-identical results (zero disturbance).
    """
    inner = _build_mini(device)
    x, _prompt_len, _fbp = _make_canvas(_prompt(device), _BLOCK_SIZE)
    mff = _make_mff(inner, device)
    with torch.no_grad():
        out_baseline = mff(x, branch_id="2103-a", step_idx=0)
        raw = inner(x)
    assert torch.equal(out_baseline, raw)
    with torch.no_grad():
        out_child = mff(x, branch_id="2103-c", parent_branch_id="2103-a", step_idx=1)
    assert torch.equal(out_child, out_baseline)
    with torch.no_grad():
        out_none = mff(x, branch_id="2103-n", step_idx=0, forced_divergent=None)
    assert torch.equal(out_none, out_baseline)
    with torch.no_grad():
        out_child_none = mff(
            x,
            branch_id="2103-cn",
            parent_branch_id="2103-a",
            step_idx=1,
            forced_divergent=None,
        )
    assert torch.equal(out_child_none, out_child)


# ---------------------------------------------------------------------------
# UT-2104 (FR2): eager forced semantics across the gate paths
# ---------------------------------------------------------------------------


def test_ut2104_eager_forced_semantics(device: str) -> None:
    """UT-2104 (FR2): forced positions are recomputed, prefix reused.

    With parent == child the gate certifies everything stable; AND-ing a
    ``forced`` region must flip exactly those positions to divergent in the
    profiler-consumed mask (mean ratio drops from 1.0 by the forced fraction)
    so the all-stable reuse fast path never fires on forced bits. The ``[T]``,
    ``[1, T]`` and batched ``[N, T]`` forms must behave identically.
    """
    inner = _build_mini(device)
    x, _prompt_len, _fbp = _make_canvas(_prompt(device), _BLOCK_SIZE)
    seq_len = x.shape[1]
    mff = _make_mff(inner, device)
    with torch.no_grad():
        mff(x, branch_id="2104-p", step_idx=0)
        mff(x, branch_id="2104-c0", parent_branch_id="2104-p", step_idx=1)
    profile_plain = GLOBAL_STABILITY_PROFILER.get_profile("2104-c0")
    assert profile_plain is not None
    assert math.isclose(profile_plain.mean_stable_ratio, 1.0, rel_tol=1e-9)

    forced = torch.zeros(seq_len, dtype=torch.bool, device=device)
    forced[seq_len - _BLOCK_SIZE :] = True
    expected_ratio = (seq_len - _BLOCK_SIZE) / seq_len

    with torch.no_grad():
        mff(
            x,
            branch_id="2104-c1",
            parent_branch_id="2104-p",
            step_idx=1,
            forced_divergent=forced,
        )
    profile_forced = GLOBAL_STABILITY_PROFILER.get_profile("2104-c1")
    assert profile_forced is not None
    assert math.isclose(profile_forced.mean_stable_ratio, expected_ratio, rel_tol=1e-9)

    with torch.no_grad():
        mff(
            x,
            branch_id="2104-c2",
            parent_branch_id="2104-p",
            step_idx=1,
            forced_divergent=forced.unsqueeze(0),
        )
    profile_2d = GLOBAL_STABILITY_PROFILER.get_profile("2104-c2")
    assert profile_2d is not None
    assert math.isclose(profile_2d.mean_stable_ratio, expected_ratio, rel_tol=1e-9)

    stacked = x.repeat(3, 1)
    with torch.no_grad():
        mff(
            stacked,
            branch_id=None,
            branch_ids=["2104-b0", "2104-b1", "2104-b2"],
            parent_branch_id="2104-p",
            step_idx=1,
            forced_divergent=forced.repeat(3, 1),
        )
    for bid in ("2104-b0", "2104-b1", "2104-b2"):
        profile_row = GLOBAL_STABILITY_PROFILER.get_profile(bid)
        assert profile_row is not None
        assert math.isclose(profile_row.mean_stable_ratio, expected_ratio, rel_tol=1e-9)


@_cuda
def test_ut2104_fused_forced_count() -> None:
    """UT-2104 (FR2, fused path): the AND-ed count must be re-derived.

    At ``B*T >= 1024`` the fused gate kernel produces the stable count; with a
    forced region the count must be re-derived from the AND-ed mask (no stale
    kernel count, no all-stable fast path on forced bits).
    """
    device = "cuda"
    inner = _build_mini(device)
    mff = _make_mff(inner, device)
    torch.manual_seed(2104)
    x = torch.randint(1, 10, (1, 1024), device=device)
    with torch.no_grad():
        mff(x, branch_id="2104f-p", step_idx=0)
        mff(x, branch_id="2104f-c0", parent_branch_id="2104f-p", step_idx=1)
    profile_plain = GLOBAL_STABILITY_PROFILER.get_profile("2104f-c0")
    assert profile_plain is not None
    assert math.isclose(profile_plain.mean_stable_ratio, 1.0, rel_tol=1e-9)

    forced = torch.zeros(1024, dtype=torch.bool, device=device)
    forced[-64:] = True
    with torch.no_grad():
        mff(
            x,
            branch_id="2104f-c1",
            parent_branch_id="2104f-p",
            step_idx=1,
            forced_divergent=forced,
        )
    profile_forced = GLOBAL_STABILITY_PROFILER.get_profile("2104f-c1")
    assert profile_forced is not None
    assert math.isclose(profile_forced.mean_stable_ratio, 960 / 1024, rel_tol=1e-9)


@_cuda
def test_ut2104c_fused_forced_recompute_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2104c (FR2, fused path): the AND-ed count drives the recompute dispatch.

    A stale fused-kernel count surviving the forced AND would reroute the
    forced child through the all-stable fast path, so the merged-recompute
    dispatch -- taken only by the mixed path, with the AND-ed mask -- would
    never fire.  A spy on ``FoldedTransformerLayer._recompute_merged`` pins
    both the invocation count and the stable-token count the recompute sees.
    """
    from actfold.core.folded_transformer import FoldedTransformerLayer

    device = "cuda"
    inner = _build_mini(device)
    mff = _make_mff(inner, device)
    torch.manual_seed(2104)
    x = torch.randint(1, 10, (1, 1024), device=device)
    with torch.no_grad():
        mff(x, branch_id="2104c-p", step_idx=0)
        mff(x, branch_id="2104c-c0", parent_branch_id="2104c-p", step_idx=1)

    forced = torch.zeros(1024, dtype=torch.bool, device=device)
    forced[-64:] = True
    recompute_stable_counts: list[int] = []
    original = FoldedTransformerLayer._recompute_merged

    def spy(
        self: FoldedTransformerLayer,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        stable_mask: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        recompute_stable_counts.append(int(stable_mask.sum()))
        return original(self, hidden_states, attention_mask, stable_mask, *args, **kwargs)

    monkeypatch.setattr(FoldedTransformerLayer, "_recompute_merged", spy)
    with torch.no_grad():
        mff(
            x,
            branch_id="2104c-c1",
            parent_branch_id="2104c-p",
            step_idx=1,
            forced_divergent=forced,
        )

    assert recompute_stable_counts == [1024 - int(forced.sum())]


# ---------------------------------------------------------------------------
# UT-2105 (FR2): parity on the dual-form mini model
# ---------------------------------------------------------------------------


def test_ut2105_folded_serial_parity_dual_form(device: str) -> None:
    """UT-2105 (FR2): folded-serial generation is bit-exact vs the raw path.

    The dual-form mask contract holds first (raw ``[B, T]`` internal build and
    the folded ``[T, T]`` prebuilt mask feed ONE attention implementation with
    equal outputs). Every folded-path call must receive the pinned ``[T, T]``
    mask plus ``forced_divergent``; the decoded token sequence must equal the
    raw generation bit-exactly; the prefix-reuse probe observes partial-stable
    branches and the active-block recompute probe observes exactly the block
    token count as divergent.
    """
    inner = _build_mini(device)
    llm = _make_llm(inner)
    prompt = _prompt(device)
    x, prompt_len, fbp = _make_canvas(prompt, _BLOCK_SIZE)
    seq_len = x.shape[1]
    pad_mask = x != _PAD_ID
    folded = _fold_mask(seq_len, prompt_len, fbp, _BLOCK_SIZE, device)
    with torch.no_grad():
        out_raw_form = inner(x, attention_mask=pad_mask)
        out_fold_form = inner(x, attention_mask=folded)
    assert torch.equal(out_raw_form, out_fold_form)

    out_raw = FastDLLMSampler(llm, config=_sampler_config(seed=2105)).sample(prompt)
    mff = _make_mff(inner, device, cls=_SpyMFF)
    sampler = FastDLLMSampler(llm, config=_sampler_config(seed=2105))
    out_folded = sampler.sample(prompt, folded_model=mff)
    assert mff.calls, "folded path must issue folded_model calls"
    mask_ids: set[int] = set()
    for call in mff.calls:
        tokens = call["args"][0]
        am = call["kwargs"].get("attention_mask")
        call_len = tokens.shape[1]
        assert am is not None, "folded calls must carry the fold-context mask"
        assert am.dim() == 2 and am.shape[0] == am.shape[1] == call_len
        assert "forced_divergent" in call["kwargs"]
        mask_ids.add(id(am))
    assert len(mask_ids) == 1, "the same shape window must pin one mask object"
    assert torch.equal(out_raw.sequences, out_folded.sequences)

    ratios: list[float] = []
    for call in mff.calls:
        bid = call["kwargs"].get("branch_id")
        profile = GLOBAL_STABILITY_PROFILER.get_profile(bid)
        if profile is not None:
            ratios.append(profile.mean_stable_ratio)
    assert any(0.0 < ratio < 1.0 for ratio in ratios), "prefix reuse probe must be > 0"
    expected_ratio = (seq_len - _BLOCK_SIZE) / seq_len
    assert any(
        math.isclose(ratio, expected_ratio, rel_tol=1e-6) for ratio in ratios
    ), "active-block recompute probe must equal the block token count"


# ---------------------------------------------------------------------------
# UT-2106 (FR2): shape-keyed runner pool
# ---------------------------------------------------------------------------


def test_ut2106_config_pool_size_validation() -> None:
    """UT-2106 (FR2): ActFoldConfig.graph_runner_pool_size with >=0 validation."""
    cfg = ActFoldConfig(graph_runner_pool_size=8)
    assert cfg.graph_runner_pool_size == 8
    cfg_default = ActFoldConfig()
    assert cfg_default.graph_runner_pool_size == 0
    with pytest.raises(ValueError):
        ActFoldConfig(graph_runner_pool_size=-1)


@_cuda
def test_ut2106_pool_multi_shape_windows() -> None:
    """UT-2106 (FR2): pool on — one runner per shape window, replays, no warnings.

    A two-block generation crosses two canvas shapes; the serial pool must hold
    one captured runner per window with replays > 0, emit zero UserWarning, and
    expose the two instance-cumulative eager counters with a zero
    degradation-fallback canary.
    """
    device = "cuda"
    inner = _build_mini(device)
    llm = _make_llm(inner)
    mff = _make_mff(
        inner,
        device,
        use_cuda_graph=True,
        split_layers=True,
        graph_runner_pool_size=2,
    )
    sampler = FastDLLMSampler(llm, config=_sampler_config(num_tokens=16, seed=2106))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sampler.sample(_prompt(device), folded_model=mff)
    assert _user_warning_messages(record) == []
    stats = mff.graph_pool_stats
    assert set(stats) >= {
        "serial",
        "batched",
        "degradation_fallback_count",
        "structural_eager_count",
    }
    serial = stats["serial"]
    assert serial["pool_size"] >= 2
    entries = {k: v for k, v in serial.items() if isinstance(v, dict) and "captures" in v}
    assert len(entries) >= 2
    for entry in entries.values():
        assert entry["captures"] >= 1
        assert entry["replays"] > 0
    assert stats["degradation_fallback_count"] == 0


@_cuda
def test_ut2106_lru_eviction_vram() -> None:
    """UT-2106 (FR2): capacity-1 LRU evicts and keeps VRAM bounded.

    Driving two shape windows under ``graph_runner_pool_size=4`` keeps both
    runners resident; under capacity 1 the older runner is evicted (its shape
    re-captures later at no correctness cost) and the resident allocation stays
    below the two-runner footprint.
    """
    device = "cuda"
    inner = _build_mini(device)
    x16 = torch.randint(1, 10, (1, 16), device=device)
    x24 = torch.randint(1, 10, (1, 24), device=device)
    mask16 = _fold_mask(16, 3, 5, _BLOCK_SIZE, device)
    mask24 = _fold_mask(24, 3, 5, _BLOCK_SIZE, device)

    def _drive(mff: ManualFoldedForward, tag: str) -> None:
        for x, mask in ((x16, mask16), (x24, mask24)):
            seq_len = x.shape[1]
            with torch.no_grad():
                mff(x, branch_id=f"{tag}-p{seq_len}", step_idx=0, attention_mask=mask)
                mff(
                    x,
                    branch_id=f"{tag}-c{seq_len}",
                    parent_branch_id=f"{tag}-p{seq_len}",
                    step_idx=1,
                    attention_mask=mask,
                )
                mff(
                    x,
                    branch_id=f"{tag}-r{seq_len}",
                    parent_branch_id=f"{tag}-p{seq_len}",
                    step_idx=2,
                    attention_mask=mask,
                )

    mff_big = _make_mff(
        inner,
        device,
        use_cuda_graph=True,
        split_layers=True,
        graph_runner_pool_size=4,
    )
    _drive(mff_big, "big")
    torch.cuda.synchronize()
    mem_big = torch.cuda.memory_allocated()

    mff_small = _make_mff(
        inner,
        device,
        use_cuda_graph=True,
        split_layers=True,
        graph_runner_pool_size=1,
    )
    _drive(mff_small, "small")
    torch.cuda.synchronize()
    mem_small = torch.cuda.memory_allocated()

    stats = mff_small.graph_pool_stats
    serial = stats["serial"]
    assert serial["pool_size"] == 1
    captures = [v["captures"] for v in serial.values() if isinstance(v, dict) and "captures" in v]
    assert sum(captures) >= 2, "both shape windows must have captured a runner"
    evictions = [v.get("evictions", 0) for v in serial.values() if isinstance(v, dict)]
    assert sum(evictions) >= 1, "capacity 1 must evict the older runner (LRU)"
    assert mem_small < mem_big, "eviction must conserve resident VRAM"


@_cuda
def test_ut2106_cache_setter_pool_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2106 (FR2, #42): ``mff.cache = other`` reaches every pool runner.

    Runner instances are observed through a recording ``FoldedGraphRunner.__init__``
    patch; after two shape windows capture two pool runners, the cache swap must
    propagate to every runner's pinned ``_cache`` — including non-active pool
    members.
    """
    device = "cuda"
    inner = _build_mini(device)
    created: list[Any] = []
    original_init = FoldedGraphRunner.__init__

    def _recording_init(self: Any, *args: Any, **kwargs: Any) -> None:
        created.append(self)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(FoldedGraphRunner, "__init__", _recording_init)

    mff = _make_mff(
        inner,
        device,
        use_cuda_graph=True,
        split_layers=True,
        graph_runner_pool_size=4,
    )
    x16 = torch.randint(1, 10, (1, 16), device=device)
    x24 = torch.randint(1, 10, (1, 24), device=device)
    mask16 = _fold_mask(16, 3, 5, _BLOCK_SIZE, device)
    mask24 = _fold_mask(24, 3, 5, _BLOCK_SIZE, device)
    for x, mask, tag in ((x16, mask16, "16"), (x24, mask24, "24")):
        with torch.no_grad():
            mff(x, branch_id=f"prop-p{tag}", step_idx=0, attention_mask=mask)
            mff(
                x,
                branch_id=f"prop-c{tag}",
                parent_branch_id=f"prop-p{tag}",
                step_idx=1,
                attention_mask=mask,
            )
    assert len(created) >= 2, "two shape windows must capture two pool runners"
    other = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    mff.cache = other
    for runner in created:
        assert runner._cache is other


@_cuda
def test_ut2106_batched_pool_key() -> None:
    """UT-2106 (FR2): batched pool keys are (N, T, prompt_len).

    Two batched calls with the same ``[N, T]`` shape but different prompt
    padding patterns must split pool keys (never share a runner); a same-key
    repeat call replays.
    """
    device = "cuda"
    inner = _build_mini(device)
    mff = _make_mff(
        inner,
        device,
        use_cuda_graph=True,
        split_layers=True,
        graph_runner_pool_size=8,
    )
    num_candidates = 4
    seq_len = 16
    mask_a = _fold_mask(seq_len, 3, 5, _BLOCK_SIZE, device)
    mask_b = _fold_mask(seq_len, 8, 0, _BLOCK_SIZE, device)
    x_a = torch.randint(1, 10, (1, seq_len), device=device)
    x_b = torch.randint(1, 10, (1, seq_len), device=device)
    forced_a = torch.zeros(num_candidates, seq_len, dtype=torch.bool, device=device)
    forced_a[:, seq_len - _BLOCK_SIZE :] = True
    forced_b = forced_a.clone()
    with torch.no_grad():
        mff(x_a, branch_id="bpA", step_idx=0, attention_mask=mask_a)
        mff(
            x_a.repeat(num_candidates, 1),
            branch_id=None,
            branch_ids=[f"bpA_cand{j}" for j in range(num_candidates)],
            parent_branch_id="bpA",
            step_idx=1,
            attention_mask=mask_a,
            forced_divergent=forced_a,
        )
        mff(x_b, branch_id="bpB", step_idx=0, attention_mask=mask_b)
        mff(
            x_b.repeat(num_candidates, 1),
            branch_id=None,
            branch_ids=[f"bpB_cand{j}" for j in range(num_candidates)],
            parent_branch_id="bpB",
            step_idx=1,
            attention_mask=mask_b,
            forced_divergent=forced_b,
        )
        mff(
            x_a.repeat(num_candidates, 1),
            branch_id=None,
            branch_ids=[f"bpA_c2_{j}" for j in range(num_candidates)],
            parent_branch_id="bpA",
            step_idx=2,
            attention_mask=mask_a,
            forced_divergent=forced_a,
        )
    stats = mff.graph_pool_stats
    batched = stats["batched"]
    entries = {k: v for k, v in batched.items() if isinstance(v, dict) and "captures" in v}
    assert len(entries) == 2, "same (N, T) with different prompt_len must split keys"
    assert sum(1 for v in entries.values() if v["replays"] >= 1) == 1


# ---------------------------------------------------------------------------
# UT-2107 (FR2): graph-side forced buffer
# ---------------------------------------------------------------------------


@_cuda
def test_ut2107_graph_forced_buffer() -> None:
    """UT-2107 (FR2): the runner's forced static buffer drives replay semantics.

    Capture with a forced region, replay identically, then change the forced
    content: the recomputed region must follow (device-side count probe via
    ``last_layer_divergences`` and the published stable ratio). A None call on
    a forced-captured runner (and the reverse) must silently take the eager
    path with correct results; forced bits must count into the divergent
    budget (capacity 8, forced 8 -> replay passes).
    """
    device = "cuda"
    inner = _build_mini(device)
    mff = _make_mff(inner, device, use_cuda_graph=True, split_layers=True)
    x, prompt_len, fbp = _make_canvas(_prompt(device), _BLOCK_SIZE)
    seq_len = x.shape[1]
    mask = _fold_mask(seq_len, prompt_len, fbp, _BLOCK_SIZE, device)
    forced = torch.zeros(seq_len, dtype=torch.bool, device=device)
    forced[seq_len - _BLOCK_SIZE :] = True

    with torch.no_grad():
        mff(x, branch_id="2107-p", step_idx=0, attention_mask=mask)
        out_capture = mff(
            x,
            branch_id="2107-c1",
            parent_branch_id="2107-p",
            step_idx=1,
            attention_mask=mask,
            forced_divergent=forced,
        )
        out_replay = mff(
            x,
            branch_id="2107-c2",
            parent_branch_id="2107-p",
            step_idx=2,
            attention_mask=mask,
            forced_divergent=forced,
        )
    assert torch.equal(out_replay, out_capture)
    runner = mff.graph_runner
    assert runner is not None
    assert runner.last_layer_divergences == [_BLOCK_SIZE] * _NUM_LAYERS

    forced_narrow = torch.zeros(seq_len, dtype=torch.bool, device=device)
    forced_narrow[seq_len - 4 :] = True
    with warnings.catch_warnings(record=True) as record_narrow:
        warnings.simplefilter("always")
        with torch.no_grad():
            mff(
                x,
                branch_id="2107-c3",
                parent_branch_id="2107-p",
                step_idx=3,
                attention_mask=mask,
                forced_divergent=forced_narrow,
            )
    assert _user_warning_messages(record_narrow) == []
    assert runner.last_layer_divergences == [4, 4]
    profile_narrow = GLOBAL_STABILITY_PROFILER.get_profile("2107-c3")
    assert profile_narrow is not None
    assert math.isclose(profile_narrow.mean_stable_ratio, (seq_len - 4) / seq_len)

    mff_eager = _make_mff(inner, device, split_layers=True)
    with torch.no_grad():
        mff_eager(x, branch_id="2107-ep", step_idx=0, attention_mask=mask)
        ref = mff_eager(
            x,
            branch_id="2107-ec",
            parent_branch_id="2107-ep",
            step_idx=1,
            attention_mask=mask,
        )
    with warnings.catch_warnings(record=True) as record_none:
        warnings.simplefilter("always")
        with torch.no_grad():
            out_none = mff(
                x,
                branch_id="2107-c4",
                parent_branch_id="2107-p",
                step_idx=4,
                attention_mask=mask,
            )
    assert _user_warning_messages(record_none) == []
    assert torch.equal(out_none, ref)

    mff_plain = _make_mff(inner, device, use_cuda_graph=True, split_layers=True)
    with torch.no_grad():
        mff_plain(x, branch_id="2107-pp", step_idx=0, attention_mask=mask)
        mff_plain(
            x,
            branch_id="2107-pc",
            parent_branch_id="2107-pp",
            step_idx=1,
            attention_mask=mask,
        )
        with warnings.catch_warnings(record=True) as record_forced:
            warnings.simplefilter("always")
            out_forced = mff_plain(
                x,
                branch_id="2107-pf",
                parent_branch_id="2107-pp",
                step_idx=2,
                attention_mask=mask,
                forced_divergent=forced,
            )
    assert _user_warning_messages(record_forced) == []
    assert torch.equal(out_forced, ref)


# ---------------------------------------------------------------------------
# UT-2108 (FR3): num_block_candidates=0 zero disturbance
# ---------------------------------------------------------------------------


def test_ut2108_num_candidates_zero_zero_disturbance(device: str) -> None:
    """UT-2108 (FR3): the default-off candidate mode is byte-identical.

    Explicit ``num_block_candidates=0`` / ``candidate_top_k=10`` must reproduce
    the default-config outputs bit-exactly on the raw path, on the folded eager
    path, and on a deterministic stop-token scenario.
    """
    inner = _build_mini(device)
    llm = _make_llm(inner)
    prompt = _prompt(device)
    out_default = FastDLLMSampler(llm, config=_sampler_config(seed=2108)).sample(prompt)
    cfg_zero = _sampler_config(seed=2108, num_block_candidates=0, candidate_top_k=10)
    out_zero = FastDLLMSampler(llm, config=cfg_zero).sample(prompt)
    assert torch.equal(out_zero.sequences, out_default.sequences)

    mff_default = _make_mff(inner, device)
    mff_zero = _make_mff(inner, device)
    out_default_f = FastDLLMSampler(llm, config=_sampler_config(seed=2108)).sample(
        prompt, folded_model=mff_default
    )
    out_zero_f = FastDLLMSampler(llm, config=cfg_zero).sample(prompt, folded_model=mff_zero)
    assert torch.equal(out_zero_f.sequences, out_default_f.sequences)

    inner_stop = _build_mini(device, head="constant")
    llm_stop = _make_llm(inner_stop)
    out_stop_default = FastDLLMSampler(llm_stop, config=_sampler_config(seed=2108)).sample(prompt)
    cfg_stop = _sampler_config(seed=2108, num_block_candidates=0)
    out_stop_zero = FastDLLMSampler(llm_stop, config=cfg_stop).sample(prompt)
    assert torch.equal(out_stop_zero.sequences, out_stop_default.sequences)


@_cuda
def test_ut2108_graph_mode_zero_disturbance() -> None:
    """UT-2108 (FR3): the default-off candidate mode under graph folding."""
    device = "cuda"
    inner = _build_mini(device)
    llm = _make_llm(inner)
    prompt = _prompt(device)
    cfg_zero = _sampler_config(seed=2108, num_block_candidates=0, candidate_top_k=10)
    mff_default = _make_mff(inner, device, use_cuda_graph=True, split_layers=True)
    mff_zero = _make_mff(inner, device, use_cuda_graph=True, split_layers=True)
    out_default_f = FastDLLMSampler(llm, config=_sampler_config(seed=2108)).sample(
        prompt, folded_model=mff_default
    )
    out_zero_f = FastDLLMSampler(llm, config=cfg_zero).sample(prompt, folded_model=mff_zero)
    assert torch.equal(out_zero_f.sequences, out_default_f.sequences)


# ---------------------------------------------------------------------------
# UT-2109 (FR3): batched speculative flow
# ---------------------------------------------------------------------------


def test_ut2109_batched_speculative_flow(device: str) -> None:
    """UT-2109 (FR3): N=4 — one [N,T] call per small block, chain continuation.

    Each small block performs exactly one draft serial forward plus one
    ``[N, T]`` batched call (``branch_ids=[f"{p}_cand{j}"]`` sharing the draft
    parent, pinned ``[T, T]`` mask, ``[N, T]`` forced) and skips the remaining
    remasking iterations; the batched publish lands per-branch ``[1, T]``
    entries; the folding chain parent switches to the selected candidate; the
    frozen per-candidate seed formula makes the whole flow reproducible across
    runs with the same seed.
    """
    inner = _build_mini(device)
    llm = _make_llm(inner)
    prompt = _prompt(device)
    cfg = _sampler_config(seed=2109, num_block_candidates=4, candidate_top_k=2)
    mff = _make_mff(inner, device, cls=_SpyMFF)
    sampler = FastDLLMSampler(llm, config=cfg)
    out = sampler.sample(prompt, folded_model=mff)

    batched_calls = [c for c in mff.calls if c["kwargs"].get("branch_ids") is not None]
    serial_calls = [c for c in mff.calls if c["kwargs"].get("branch_ids") is None]
    assert len(batched_calls) == 2, "each small block must make exactly one [N,T] call"
    assert len(serial_calls) == 2, "each small block must make exactly one draft forward"

    first = batched_calls[0]
    tokens = first["args"][0]
    seq_len = tokens.shape[1]
    assert tokens.shape[0] == 4
    forced = first["kwargs"]["forced_divergent"]
    assert forced.shape == (4, seq_len)
    assert bool(forced[:, seq_len - _BLOCK_SIZE :].all())
    am = first["kwargs"]["attention_mask"]
    assert am.dim() == 2 and am.shape[0] == am.shape[1] == seq_len
    draft_bid = serial_calls[0]["kwargs"]["branch_id"]
    assert first["kwargs"]["parent_branch_id"] == draft_bid
    assert first["kwargs"]["branch_ids"] == [f"{draft_bid}_cand{j}" for j in range(4)]

    for bid in first["kwargs"]["branch_ids"]:
        assert mff.cache.contains(bid, 0)
        entry = mff.cache.fetch(branch_id=bid, layer_idx=0)
        assert tuple(entry["ffn_out"].shape) == (1, seq_len, _HIDDEN)

    second_draft = serial_calls[1]
    assert "_cand" in str(second_draft["kwargs"]["parent_branch_id"])
    assert "_cand" in str(sampler._folding_parent)
    assert mff.cache.contains(sampler._folding_parent, 0)

    mff_rerun = _make_mff(inner, device, cls=_SpyMFF)
    out_rerun = FastDLLMSampler(
        llm, config=_sampler_config(seed=2109, num_block_candidates=4, candidate_top_k=2)
    ).sample(prompt, folded_model=mff_rerun)
    rerun_batched = [c for c in mff_rerun.calls if c["kwargs"].get("branch_ids") is not None]
    assert rerun_batched
    assert torch.equal(rerun_batched[0]["args"][0], tokens)
    assert torch.equal(out_rerun.sequences, out.sequences)


def test_ut2109_selection_and_stop_semantics(device: str) -> None:
    """UT-2109 (FR3): min-prob argmax selection and baseline stop semantics.

    On the two-spike adversarial model every candidate token lies in the top-k
    draft set; the committed small block must equal the candidate row whose
    per-position minimum token probability is maximal (computed from the spied
    batched logits). On the constant-stop model the committed stop token must
    reproduce the baseline finished/pad-trimming semantics bit-exactly.
    """
    inner = _build_mini(device, head="spike")
    llm = _make_llm(inner)
    prompt = _prompt(device)
    cfg = _sampler_config(seed=2109, num_block_candidates=4, candidate_top_k=2)
    mff = _make_mff(inner, device, cls=_SpyMFF)
    sampler = FastDLLMSampler(llm, config=cfg)
    out = sampler.sample(prompt, folded_model=mff)
    batched_calls = [c for c in mff.calls if c["kwargs"].get("branch_ids") is not None]
    assert batched_calls
    first = batched_calls[0]
    cands = first["args"][0]
    seq_len = cands.shape[1]
    lo = seq_len - _BLOCK_SIZE
    hi = lo + _SMALL_BLOCK_SIZE
    chosen = cands[:, lo:hi]
    drafted = {int(t) for t in chosen.reshape(-1).tolist()}
    assert drafted <= {_SPIKE_TOP1_ID, _SPIKE_TOP2_ID}

    shifted = right_shift_logits(first["out"])
    small_logits = shifted[:, lo:hi, :]
    probs = F.softmax(small_logits, dim=-1)
    scores = torch.gather(probs, -1, chosen.unsqueeze(-1)).squeeze(-1)
    min_scores = scores.min(dim=-1).values
    j_star = int(min_scores.argmax())
    assert torch.equal(out.sequences[0, lo:hi], cands[j_star, lo:hi])

    inner_stop = _build_mini(device, head="constant")
    llm_stop = _make_llm(inner_stop)
    out_base = FastDLLMSampler(llm_stop, config=_sampler_config(seed=2110)).sample(prompt)
    cfg_stop = _sampler_config(seed=2110, num_block_candidates=2, candidate_top_k=1)
    mff_stop = _make_mff(inner_stop, device, cls=_SpyMFF)
    out_spec = FastDLLMSampler(llm_stop, config=cfg_stop).sample(prompt, folded_model=mff_stop)
    assert torch.equal(out_spec.sequences, out_base.sequences)
    assert _STOP_ID in out_spec.sequences[0].tolist()


class _GradedSpikeHead(nn.Module):
    """Spike head whose top-1 logit grows with the position.

    Per-position top-1 probabilities are pairwise distinct, so per-candidate
    window min-probs can be pairwise distinct even when every candidate draws
    only top-1/top-2 tokens (the uniform ``_SpikeHead`` caps them at two
    values and ties the min-prob argmax).
    """

    def __init__(self, vocab: int, top1_id: int, top2_id: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.top1_id = top1_id
        self.top2_id = top2_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*x.shape[:-1], self.vocab, dtype=x.dtype, device=x.device)
        pos = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)
        logits[..., self.top1_id] = 3.0 + 0.5 * pos
        logits[..., self.top2_id] = 1.0
        return logits


def test_ut2109c_min_prob_selection_distinct_mins(device: str) -> None:
    """UT-2109c (FR3): min-prob selection discriminates distinct candidates.

    With the position-graded spike head and seed 2118 the four candidate
    window min-probs are pairwise distinct; the min-prob winner is an
    all-top-1 row whose min is a top-1 probability (non-floor), and it differs
    from the max-prob argmax pick, so a min->max selection regression commits
    the wrong candidate row into the output sequence.
    """
    inner = _build_mini(device, head="spike")
    inner.lm_head = _GradedSpikeHead(_VOCAB, _SPIKE_TOP1_ID, _SPIKE_TOP2_ID)
    llm = _make_llm(inner)
    prompt = _prompt(device)
    cfg = _sampler_config(seed=2118, num_block_candidates=4, candidate_top_k=2)
    mff = _make_mff(inner, device, cls=_SpyMFF)
    sampler = FastDLLMSampler(llm, config=cfg)
    out = sampler.sample(prompt, folded_model=mff)
    batched_calls = [c for c in mff.calls if c["kwargs"].get("branch_ids") is not None]
    assert batched_calls
    first = batched_calls[0]
    cands = first["args"][0]
    seq_len = cands.shape[1]
    lo = seq_len - _BLOCK_SIZE
    hi = lo + _SMALL_BLOCK_SIZE
    chosen = cands[:, lo:hi]

    shifted = right_shift_logits(first["out"])
    probs = F.softmax(shifted[:, lo:hi, :], dim=-1)
    scores = torch.gather(probs, -1, chosen.unsqueeze(-1)).squeeze(-1)
    min_scores = scores.min(dim=-1).values
    j_min = int(min_scores.argmax())
    j_max = int(scores.max(dim=-1).values.argmax())

    assert len(set(min_scores.tolist())) == cands.shape[0]
    assert j_min != j_max
    assert bool((cands[j_min, lo:hi] == _SPIKE_TOP1_ID).all())
    top2_floor = probs[:, :, _SPIKE_TOP2_ID].max()
    assert float(min_scores[j_min]) > float(top2_floor)
    assert torch.equal(out.sequences[0, lo:hi], cands[j_min, lo:hi])


# ---------------------------------------------------------------------------
# UT-2110 (FR3): constructor-time validation
# ---------------------------------------------------------------------------


def test_ut2110_fr3_constructor_validation(device: str) -> None:
    """UT-2110 (FR3): illegal candidate configurations raise ValueError.

    ``num_block_candidates < 0``, ``candidate_top_k < 1``, ``N > 0`` without a
    folded model, and a folded model without branch_ids support must all raise
    ValueError (never silently degrade to the baseline recipe).
    """
    inner = _build_mini(device)
    llm = _make_llm(inner)
    prompt = _prompt(device)
    with pytest.raises(ValueError):
        _sampler_config(num_block_candidates=-1)
    with pytest.raises(ValueError):
        _sampler_config(candidate_top_k=0)
    cfg = _sampler_config(num_block_candidates=4, candidate_top_k=2)
    sampler = FastDLLMSampler(llm, config=cfg)
    with pytest.raises(ValueError):
        sampler.sample(prompt)
    with pytest.raises(ValueError):
        sampler.sample(prompt, folded_model=object())


# ---------------------------------------------------------------------------
# UT-2111 (FR2): AR-step forced coverage and cross-block runner capture
# ---------------------------------------------------------------------------


def test_ut2111_ar_step_forced(device: str) -> None:
    """UT-2111 (FR2): AR-step forwards force-diverge the freshly committed block.

    Every ``fastdllm_ar_*`` forward must carry ``forced_divergent`` covering
    exactly the current block (the committed block vs the parent is the P5
    partial-flip scenario), and its published stable ratio must leave exactly
    the block token count recomputed.
    """
    inner = _build_mini(device)
    llm = _make_llm(inner)
    mff = _make_mff(inner, device, cls=_SpyMFF)
    sampler = FastDLLMSampler(llm, config=_sampler_config(num_tokens=16, seed=2111))
    sampler.sample(_prompt(device), folded_model=mff)
    ar_calls = [
        c for c in mff.calls if str(c["kwargs"].get("branch_id", "")).startswith("fastdllm_ar")
    ]
    assert ar_calls, "block-advance AR forwards must exist"
    for call in ar_calls:
        tokens = call["args"][0]
        seq_len = tokens.shape[1]
        forced = call["kwargs"].get("forced_divergent")
        assert forced is not None, "AR-step forwards must carry forced_divergent"
        assert forced.shape[-1] == seq_len
        assert bool(forced[..., -_BLOCK_SIZE:].all())
        assert not bool(forced[..., : seq_len - _BLOCK_SIZE].any())
        profile = GLOBAL_STABILITY_PROFILER.get_profile(call["kwargs"]["branch_id"])
        assert profile is not None
        expected_ratio = (seq_len - _BLOCK_SIZE) / seq_len
        assert math.isclose(profile.mean_stable_ratio, expected_ratio, rel_tol=1e-6)


@_cuda
def test_ut2111_cross_block_runner_capture() -> None:
    """UT-2111 (FR2): canvas growth captures a new pool runner per window.

    A two-block generation crosses the T=16 and T=24 windows; with the pool on,
    each window must end with its own captured serial runner (pool size 2, no
    UserWarning).
    """
    device = "cuda"
    inner = _build_mini(device)
    llm = _make_llm(inner)
    mff = _make_mff(
        inner,
        device,
        use_cuda_graph=True,
        split_layers=True,
        graph_runner_pool_size=4,
    )
    sampler = FastDLLMSampler(llm, config=_sampler_config(num_tokens=16, seed=2111))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sampler.sample(_prompt(device), folded_model=mff)
    assert _user_warning_messages(record) == []
    stats = mff.graph_pool_stats
    serial = stats["serial"]
    assert serial["pool_size"] == 2
    entries = {k: v for k, v in serial.items() if isinstance(v, dict) and "captures" in v}
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# UT-2112 (FR1): context supply discipline
# ---------------------------------------------------------------------------


def test_ut2112_ctx_supply_discipline(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2112 (FR1): pinned ctx across sample calls, keyed by (T, prompt_len).

    The same ``(T, prompt_len)`` must hit the same pinned objects across
    simulated sample-call boundaries; the same ``T`` with a different
    ``prompt_len`` must land in a different entry (different pad pattern, never
    shared); the ctx construction must perform zero device readback.
    """
    inner = _build_mini(device)
    llm = _make_llm(inner)
    sampler = FastDLLMSampler(llm, config=_sampler_config(seed=2112))
    x, prompt_len, _fbp = _make_canvas(_prompt(device), _BLOCK_SIZE)

    def _forbid_readback(name: str) -> Any:
        def _spy(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"device readback .{name}() during _fold_ctx_for")

        return _spy

    monkeypatch.setattr(torch.Tensor, "item", _forbid_readback("item"))
    monkeypatch.setattr(torch.Tensor, "tolist", _forbid_readback("tolist"))
    monkeypatch.setattr(torch.Tensor, "cpu", _forbid_readback("cpu"))
    ctx = sampler._fold_ctx_for(x, prompt_len=prompt_len)
    ctx_again = sampler._fold_ctx_for(x.clone(), prompt_len=prompt_len)
    monkeypatch.undo()

    assert ctx_again.mask is ctx.mask
    assert ctx_again.pe is ctx.pe
    assert ctx_again.forced is ctx.forced
    for _ in range(3):
        repeat = sampler._fold_ctx_for(x, prompt_len=prompt_len)
        assert repeat.mask is ctx.mask

    prompt_b = torch.randint(1, 10, (1, 8), device=device)
    x_b = torch.cat(
        [prompt_b, torch.full((1, 8), _MASK_ID, dtype=torch.long, device=device)],
        dim=1,
    )
    ctx_b = sampler._fold_ctx_for(x_b, prompt_len=8)
    assert ctx_b.mask is not ctx.mask
    assert not torch.equal(ctx_b.mask, ctx.mask)
