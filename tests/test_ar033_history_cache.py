"""AR033 tests: block-causal history KV cache (official Fast-dLLM v2 parity).

T003 fixture: an fp32 tiny REAL block-causal decoder
(``_TinyBlockCausal``) carrying the same native cache-kwarg contract as
the real Fast-dLLM v2 ``modeling.py`` —NAMED ``use_cache`` /
``past_key_values`` / ``update_past_key_values`` / ``block_size`` /
``use_block_cache`` / ``block_past_key_values`` / ``replace_position``
forward parameters, the ``eval_block_diff_mask`` block-causal attention
over absolute coordinates (the external ``attention_mask`` is IGNORED in
eval, mirroring the real remote model's unconditional overwrite), rotary
positions from ``cache_position`` (absolute), per-layer history KV
append/commit, block-cache first-build + in-place ``replace_position``
writes, and ``attention_mask=None`` on the block-cache replace path. The
fixture is a real ``nn.Module`` (real projections/attention/MLP, fp32) —zero mock outputs (#10).

UT faces (design §6.1):
- UT-3301: bit-level A/B —the cache-off and cross-block-only arms
  produce ``torch.equal`` token sequences over >= 4 shape points (block
  sizes 8/16, aligned AND short-final-block canvases, batch 1/2, with
  and without in-prompt pad). The intra-block sb arm joins the equality
  at S=1 (``small_block_size == block_size``: the block cache is fully
  rewritten every iteration, provably exact); at S>1 it is the
  documented official-parity APPROXIMATION face (stale layer>=1 KV of
  earlier small blocks as later siblings unmask —T001 P4; the official
  ``batch_sample`` shares the semantics), pinned by mechanism/counters
  in UT-3303 instead of token equality.
- UT-3302: the alignment contract —prefill/slice/commit cache forwards
  are row-wise bitwise equal to the full-canvas forward (the
  fixture-level P3), including the short-final-block tail slice; the
  frozen boundary row equals the full-canvas history tail row.
- UT-3302b: the R1 advance convention —the fixed legacy advance takes
  the UNSHIFTED tail row raw[T-1] (AR033 §6.5 erratum), pinned on a
  canvas whose raw tail rows have DIFFERENT argmaxes (the discriminating
  fixture the AR032 table model cannot provide —its winner-row last two
  tokens coincide).
- UT-3303: sb-cache mechanism faces —rebuild/partial accounting, the
  boundary-rule inertness in this flow structure (each small block
  completes before the next starts —the official loop shares the
  property), degraded == 0, double-run reproducibility.
- UT-3304: the integer forward ledger —``history_cache_stats`` counters
  equal exact geometry predictions (``threshold > 1`` forces exactly one
  argmax-guaranteed unmask per iteration), aligned and short-final-block
  geometries.
- UT-3305: default-off zero regression —all counters zero, the probe is
  never called, and the output is deterministic across runs.
- UT-3306: mutual exclusion and fail-fast —five illegal combinations
  with exception types and message key phrases.
- UT-3307: capability probe —named-parameter strictness (a ``**kwargs``
  catch-all is NOT accepted), result caching, stats schema / fresh-dict
  reads / cumulativeness across ``sample`` calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from actfold.models.base import DiffusionLLM
from actfold.models.fast_dllm_sampler import (
    _HC_PROBED_KWARGS,
    FastDLLMSampler,
    FastDLLMSamplerConfig,
)

_VOCAB = 48
_HIDDEN = 24
_NUM_LAYERS = 2
_NUM_HEADS = 2
_MASK_ID = 44
_STOP_ID = 45
_PAD_ID = 46
_MODEL_SEED = 3303


# ---------------------------------------------------------------------------
# T003 fixture: fp32 tiny real block-causal decoder with native cache kwargs
# ---------------------------------------------------------------------------


class _TinyRotary(nn.Module):
    """Minimal HF-style rotary: ``(cos, sin)`` of shape ``[1, L, head_dim]``."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids arrives as [1, L]; the output is [1, L, head_dim]
        # (the real rotary contract —_apply_rotary adds the head axis).
        pos = position_ids.to(torch.float32)
        exponent = torch.arange(0, self.head_dim, 2, device=pos.device).float()
        inv_freq = 1.0 / (10000.0 ** (exponent / self.head_dim))
        freqs = pos[..., None] * inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(hidden_states.dtype)
        sin = emb.sin().to(hidden_states.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class _TinyKVCache:
    """DynamicCache-like per-layer KV store (the fixture's own container).

    ``update`` appends (first store clones, later stores concatenate);
    ``__len__`` counts populated layers (the ``len(cache) <= layer_idx``
    first-build probe of the real modeling.py:212); ``__getitem__``
    returns the layer's ``(k, v)`` pair for the in-place
    ``replace_position`` write path (modeling.py:216-222).
    """

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0))
            self.value_cache.append(torch.empty(0))
        if len(self.key_cache[layer_idx].shape) < 4:
            self.key_cache[layer_idx] = key_states.clone()
            self.value_cache[layer_idx] = value_states.clone()
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        k = self.key_cache[layer_idx]
        if len(k.shape) < 4:
            return 0
        return int(k.shape[2])

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


@dataclass
class _TinyCausalOutput:
    """ModelingOutputs-like return: logits + the two KV handles."""

    logits: torch.Tensor
    past_key_values: _TinyKVCache | None = None
    block_past_key_values: _TinyKVCache | None = None


class _TinyAttentionLayer(nn.Module):
    """Real pre-norm transformer layer with the official KV mechanics.

    The forward mirrors modeling.py:176-254 exactly: block-cache
    first-build vs in-place ``replace_position`` write, then the history
    cache (``update`` on commit, plain ``cat`` on read), then sdpa-style
    attention under the caller's boolean mask.
    """

    def __init__(self, hidden: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.mlp_up = nn.Linear(hidden, 2 * hidden)
        self.mlp_down = nn.Linear(2 * hidden, hidden)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bool: torch.Tensor | None,
        past_kv: _TinyKVCache | None,
        update_past: bool,
        block_kv: _TinyKVCache | None,
        replace_position: int | None,
        layer_idx: int,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        bsz, seq_len, _ = h.shape
        q = self.q_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rotary(q, k, cos, sin)
        if block_kv is not None:
            if len(block_kv) <= layer_idx:
                k, v = block_kv.update(k, v, layer_idx)
            else:
                if replace_position is None:
                    raise RuntimeError(
                        "block cache hit requires replace_position (fixture contract)"
                    )
                bk, bv = block_kv[layer_idx]
                bk[:, :, replace_position : replace_position + k.shape[2]] = k
                bv[:, :, replace_position : replace_position + v.shape[2]] = v
                k, v = bk, bv
        if past_kv is not None:
            if update_past:
                k, v = past_kv.update(k, v, layer_idx)
            elif len(past_kv) > layer_idx:
                pk, pv = past_kv[layer_idx]
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attn_bool is not None:
            scores = scores.masked_fill(~attn_bool, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, -1)
        h = residual + self.o_proj(ctx)
        residual = h
        h = self.post_attention_layernorm(h)
        return residual + self.mlp_down(F.gelu(self.mlp_up(h)))


class _TinyBlockCausal(nn.Module):
    """fp32 tiny Fast-dLLM-style decoder (the T003 fixture).

    Real embeddings/projections/rotary/attention/MLP with the official
    ``eval_block_diff_mask`` block-causal semantics over ABSOLUTE
    coordinates; the external ``attention_mask`` is ignored in eval
    (the real modeling.py:490-496 unconditionally overwrites it). The
    cache kwargs are NAMED forward parameters —the face the AR033
    probe inspects.
    """

    def __init__(
        self, vocab: int = _VOCAB, hidden: int = _HIDDEN, default_block_size: int = 8
    ) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "fast_dllm_tiny", "vocab_size": vocab})()
        self.default_block_size = default_block_size
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.rotary_emb = _TinyRotary(hidden // _NUM_HEADS)
        self.layers = nn.ModuleList(
            [_TinyAttentionLayer(hidden, _NUM_HEADS) for _ in range(_NUM_LAYERS)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        # Keep greedy decodes stop-free so every UT exercises the full
        # multi-block geometry (the stop face is the legacy loop's own).
        with torch.no_grad():
            self.lm_head.weight[_STOP_ID] -= 30.0

    @staticmethod
    def _eval_mask(q_pos: torch.Tensor, kv_pos: torch.Tensor, block_size: int) -> torch.Tensor:
        block_q = q_idx_block = q_pos // block_size
        block_kv = kv_pos // block_size
        del q_idx_block
        return block_q[:, None] >= block_kv[None, :]

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: _TinyKVCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        update_past_key_values: bool = False,
        block_size: int | None = None,
        use_block_cache: bool = False,
        block_past_key_values: _TinyKVCache | None = None,
        replace_position: int | None = None,
    ) -> _TinyCausalOutput:
        # Eval contract: the external attention_mask is IGNORED (the real
        # remote model overwrites it unconditionally in eval).
        del attention_mask
        if block_size is None:
            block_size = self.default_block_size
        if input_ids is None:
            raise ValueError("input_ids required")
        if use_cache and past_key_values is None:
            past_key_values = _TinyKVCache()
        if use_block_cache and block_past_key_values is None:
            block_past_key_values = _TinyKVCache()
        seq_len = int(input_ids.shape[1])
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            if use_block_cache and replace_position is not None:
                block_start = past_seen + replace_position
            else:
                block_start = past_seen
            cache_position = torch.arange(block_start, block_start + seq_len)
        if use_block_cache and block_past_key_values.get_seq_length() != 0:
            attn_bool = None  # modeling.py:493-494 —replace path, full attention
        else:
            q_pos = cache_position
            kv_pos = torch.arange(past_seen + seq_len)
            attn_bool = self._eval_mask(q_pos, kv_pos, block_size)
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden, cache_position.unsqueeze(0))
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                cos,
                sin,
                attn_bool,
                past_key_values,
                update_past_key_values,
                block_past_key_values,
                replace_position,
                layer_idx,
            )
        logits = self.lm_head(self.norm(hidden))
        return _TinyCausalOutput(
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            block_past_key_values=block_past_key_values if use_block_cache else None,
        )


class _MiniTokenizer:
    """Minimal tokenizer exposing the special ids the samplers resolve."""

    def __init__(self) -> None:
        self.mask_token_id = _MASK_ID
        self.eos_token_id = _STOP_ID
        self.bos_token_id = 1
        self.pad_token_id = _PAD_ID


class _TinyDiffusionLLM(DiffusionLLM):
    """DiffusionLLM wrapper around the tiny decoder (records raw calls).

    ``canvas_block_size`` threads the test geometry into the legacy
    full-canvas forwards (the real remote model's forward default is the
    production block size; the fixture mirrors that with an explicit
    per-instance default).
    """

    def __init__(self, inner: nn.Module, canvas_block_size: int = 8) -> None:
        super().__init__("tiny-block-causal")
        self.model = inner
        self.canvas_block_size = canvas_block_size
        self.tokenizer = _MiniTokenizer()
        self.raw_calls: list[dict[str, Any]] = []

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.raw_calls.append({"tokens": tokens.detach().clone(), "kwargs": set(kwargs)})
        return self.model(input_ids=tokens, block_size=self.canvas_block_size).logits

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        raw = self.model
        return raw.embed_tokens(tokens)

    @property
    def num_layers(self) -> int:
        return _NUM_LAYERS

    @property
    def hidden_dim(self) -> int:
        return _HIDDEN

    @property
    def num_heads(self) -> int:
        return _NUM_HEADS

    @property
    def vocab_size(self) -> int:
        return _VOCAB


def _build_llm(block_size: int = 8) -> _TinyDiffusionLLM:
    torch.manual_seed(_MODEL_SEED)
    model = _TinyBlockCausal(default_block_size=block_size)
    model.eval()
    return _TinyDiffusionLLM(model, canvas_block_size=block_size)


def _prompt(batch: int = 1, with_pad: bool = False) -> torch.Tensor:
    rows = [[1, 2, 3]]
    if batch == 2:
        rows = [[1, 2, 3], [_PAD_ID if with_pad else 4, 2, 3]]
    return torch.tensor(rows, dtype=torch.long)


def _cfg(**overrides: Any) -> FastDLLMSamplerConfig:
    base: dict[str, Any] = {
        "num_tokens": 16,
        "block_size": 8,
        "small_block_size": 4,
        "threshold": 1.1,  # exactly one argmax-guaranteed unmask per iteration
        "temperature": 0.0,
        "stop_token_id": _STOP_ID,
        "pad_token_id": _PAD_ID,
        "mask_token_id": _MASK_ID,
        "seed": 3301,
    }
    base.update(overrides)
    return FastDLLMSamplerConfig(**base)


# ---------------------------------------------------------------------------
# UT-3301: bit-level A/B (cache-off vs cross-block; sb joins at S=1)
# ---------------------------------------------------------------------------

_SHAPE_POINTS = [
    pytest.param(8, 4, 16, 1, False, id="b8s4t16_batch1"),
    pytest.param(8, 4, 12, 1, False, id="b8s4t12_short_final_batch1"),
    pytest.param(8, 4, 16, 2, True, id="b8s4t16_batch2_pad"),
    pytest.param(16, 4, 24, 1, False, id="b16s4t24_short_final"),
]


@pytest.mark.parametrize("block_size,small,num_tokens,batch,with_pad", _SHAPE_POINTS)
def test_ut3301_bitwise_cache_off_vs_cross_block(
    block_size: int, small: int, num_tokens: int, batch: int, with_pad: bool
) -> None:
    """UT-3301: greedy multi-block decodes are bitwise equal, cache off vs on."""
    llm = _build_llm(block_size=block_size)
    prompt = _prompt(batch=batch, with_pad=with_pad)
    common = dict(block_size=block_size, small_block_size=small, num_tokens=num_tokens)
    out_off = FastDLLMSampler(llm, config=_cfg(**common)).sample(prompt)
    out_on = FastDLLMSampler(llm, config=_cfg(history_cache=True, **common)).sample(prompt)
    assert torch.equal(out_off.sequences, out_on.sequences)
    # Multi-block coverage sanity: the off-arm really ran the full-canvas
    # legacy path (raw wrapper calls with the whole canvas), and the on-arm
    # produced a non-degenerate decode.
    assert _MASK_ID not in out_on.sequences
    assert out_on.sequences.shape[1] > block_size


@pytest.mark.parametrize(
    "block_size,num_tokens",
    [(8, 16), (8, 12)],
    ids=["aligned", "short_final"],
)
def test_ut3301_s1_three_way_equality(block_size: int, num_tokens: int) -> None:
    """UT-3301 (S=1): the sb arm joins the bitwise equality.

    With ``small_block_size == block_size`` every intra-block partial
    forward rewrites the WHOLE block cache before attention (the in-place
    ``replace_position=0`` write covers all positions), so the block
    cache is never stale —the sb arm is provably exact and all three
    switch states must agree bitwise.
    """
    llm = _build_llm(block_size=block_size)
    prompt = _prompt()
    common = dict(block_size=block_size, small_block_size=block_size, num_tokens=num_tokens)
    out_off = FastDLLMSampler(llm, config=_cfg(**common)).sample(prompt)
    out_cross = FastDLLMSampler(llm, config=_cfg(history_cache=True, **common)).sample(prompt)
    out_sb = FastDLLMSampler(
        llm,
        config=_cfg(history_cache=True, history_small_block_cache=True, **common),
    ).sample(prompt)
    assert torch.equal(out_off.sequences, out_cross.sequences)
    assert torch.equal(out_off.sequences, out_sb.sequences)


# ---------------------------------------------------------------------------
# UT-3302: alignment contract (fixture-level P3) + UT-3302b: R1 advance
# ---------------------------------------------------------------------------


def _canvas_after_block0(
    sampler: FastDLLMSampler, prompt: torch.Tensor
) -> tuple[Any, torch.Tensor]:
    """Run the legacy sampler once; return ``(output, completed-block0 canvas)``.

    The canvas is the last history snapshot at the initial canvas length
    whose window carries no masks —the exact input the block-advance
    decision consumes.
    """
    out = sampler.sample(prompt)
    initial_len = int(out.history[0].shape[1])
    done = [
        h
        for h in out.history
        if h.shape[1] == initial_len
        and not bool((h[0, -sampler.config.block_size :] == _MASK_ID).any())
    ]
    assert done, "no completed-block-0 snapshot found"
    return out, done[-1]


def test_ut3302_alignment_contract() -> None:
    """UT-3302: cache forwards are row-wise equal to the full canvas.

    Row-wise equality is compared within allclose(1e-4): the history prefill
    and slice forwards run different GEMM shapes than the full canvas, so
    their last bits are not portable across torch builds.
    """
    llm = _build_llm()
    model = llm.model
    assert isinstance(model, _TinyBlockCausal)
    sampler = FastDLLMSampler(llm, config=_cfg(return_history=True))
    _out, canvas = _canvas_after_block0(sampler, _prompt())
    # canvas = [prompt(3) | align pad(5) | block0(8)], all block0 committed.
    assert canvas.shape[1] == 16
    full = model(input_ids=canvas).logits

    # Prefill over the history region: interior rows + the frozen boundary
    # (tail) row match the full-canvas rows within allclose(1e-4).
    out_p = model(
        input_ids=canvas[:, :8],
        use_cache=True,
        update_past_key_values=True,
        block_size=8,
    )
    assert torch.allclose(out_p.logits, full[:, :8, :], rtol=1e-4, atol=1e-4)
    assert torch.allclose(out_p.logits[:, -1:, :], full[:, 7:8, :], rtol=1e-4, atol=1e-4)
    kv = out_p.past_key_values
    assert kv is not None and kv.get_seq_length() == 8

    # Whole-block slice forward against the history KV.
    out_s = model(
        input_ids=canvas[:, 8:],
        use_cache=True,
        past_key_values=kv,
        update_past_key_values=False,
        block_size=8,
    )
    assert torch.allclose(out_s.logits, full[:, 8:, :], rtol=1e-4, atol=1e-4)

    # Commit forward: the tail row that produces the advance token.
    out_c = model(
        input_ids=canvas[:, 8:],
        use_cache=True,
        past_key_values=kv,
        update_past_key_values=True,
        block_size=8,
    )
    assert torch.allclose(out_c.logits[:, -1:, :], full[:, -1:, :], rtol=1e-4, atol=1e-4)
    kv2 = out_c.past_key_values
    assert kv2 is not None and kv2.get_seq_length() == 16

    # Short-final-block tail slice (the num_tokens=12 geometry): the
    # block-aligned tail [adv | masks] against the committed history —    # rows match the full canvas within allclose(1e-4), never overlapping history.
    adv = full[:, -1:, :].argmax(dim=-1)
    short = torch.cat(
        [
            canvas,
            adv,
            torch.full((1, 3), _MASK_ID, dtype=torch.long),
        ],
        dim=1,
    )
    full2 = model(input_ids=short).logits
    out_f = model(
        input_ids=short[:, 16:],
        use_cache=True,
        past_key_values=kv2,
        update_past_key_values=False,
        block_size=8,
    )
    assert torch.allclose(out_f.logits, full2[:, 16:, :], rtol=1e-4, atol=1e-4)


def test_ut3302b_advance_convention_unshifted_tail() -> None:
    """UT-3302b: the legacy advance takes the UNSHIFTED tail row raw[T-1].

    AR033 §6.5 erratum: raw[T-1] predicts position T (R1, T001 P2
    teacher-forcing hit_next=0.9028); the historical code consumed the
    right-shifted tail raw[T-2] (the decision row of the already-filled
    position T-1). Pinned on a canvas whose two tail rows have DIFFERENT
    argmaxes —the discriminating fixture.
    """
    llm = _build_llm()
    model = llm.model
    assert isinstance(model, _TinyBlockCausal)
    sampler = FastDLLMSampler(llm, config=_cfg(return_history=True))
    out, canvas = _canvas_after_block0(sampler, _prompt())
    full = model(input_ids=canvas).logits
    next_expected = int(full[0, -1, :].argmax())
    shifted_tail = int(full[0, -2, :].argmax())
    # Discriminator premise: the two candidate sources disagree here.
    assert next_expected != shifted_tail
    # The appended advance token equals the UNSHIFTED tail row's argmax.
    idx = next(i for i, h in enumerate(out.history) if h is canvas)
    post = out.history[idx + 1]
    assert int(post[0, canvas.shape[1]]) == next_expected
    assert int(post[0, canvas.shape[1]]) != shifted_tail


# ---------------------------------------------------------------------------
# UT-3303: sb-cache mechanism faces (S>1 approximation arm)
# ---------------------------------------------------------------------------


def test_ut3303_sb_cache_mechanism_faces() -> None:
    """UT-3303: rebuild/partial accounting, inert boundary rule, repro.

    Geometry: block=8, small=4 (S=2), num_tokens=16 -> block 0 (8 masked
    positions) + block 1 (window ``[adv | 7 masks]``); block 1 is final
    (no commit). ``threshold=1.1`` forces exactly one unmask per
    iteration, so the forward counts are geometry-exact:

    - block 0: sb0 4 iterations (1 rebuild + 3 four-row partials), sb1
      4 iterations (4 five-row partials —the S+1 boundary-refresh
      slice); block rebuilds == one first-build per block (the boundary
      rule is inert in this flow structure: each small block completes
      before the next starts, so the boundary token is never masked at
      the next small block's entry —the official ``batch_sample`` loop
      shares the property);
    - block 1: sb0 3 iterations (1 rebuild + 2 partials), sb1 4
      iterations (4 partials).
    """
    llm = _build_llm()
    sampler = FastDLLMSampler(
        llm,
        config=_cfg(
            history_cache=True,
            history_small_block_cache=True,
        ),
    )
    prompt = _prompt()
    out1 = sampler.sample(prompt)
    stats1 = dict(sampler.history_cache_stats)
    out2 = sampler.sample(prompt)
    stats2 = dict(sampler.history_cache_stats)

    # AR034 adds two additive ledger keys (spec_draft/spec_verify); the
    # AR033 cache path keeps them at zero.
    assert {
        k: v
        for k, v in stats1.items()
        if k
        not in (
            "spec_gated_iterations",
            "spec_fused_forwards",
            "spec_budget_skips",
            "spec_budget_escalations",
            "spec_verification_free_blocks",
            "free_verify_blocks",
            "free_verify_low_conf",
            "spec_remask_fallbacks",
            "spec_chunk_forwards",
        )
    } == {
        "prefill_forwards": 1,
        "slice_forwards": 15,  # 8 (block 0) + 7 (block 1), rebuilds included
        "commit_forwards": 1,  # block 0 only; block 1 is final
        "block_rebuilds": 2,  # one first-build per block; boundary rule inert
        "forward_tokens": 8 + (8 + 3 * 4 + 4 * 5 + 8) + (8 + 2 * 4 + 4 * 5),
        "degraded_calls": 0,
        "spec_draft_forwards": 0,
        "spec_verify_forwards": 0,
    }
    # Cumulative across sample calls; the second run adds the same ledger.
    assert stats2["prefill_forwards"] == 2 * stats1["prefill_forwards"]
    assert stats2["slice_forwards"] == 2 * stats1["slice_forwards"]
    assert stats2["forward_tokens"] == 2 * stats1["forward_tokens"]
    assert stats2["degraded_calls"] == 0
    # Double-run reproducibility (greedy, seeded).
    assert torch.equal(out1.sequences, out2.sequences)


# ---------------------------------------------------------------------------
# UT-3304: integer forward ledger (cross-block arm, exact geometry)
# ---------------------------------------------------------------------------


def test_ut3304_cross_block_ledger_aligned() -> None:
    """UT-3304: the H3 integer ledger on the aligned geometry.

    block=8, small=4, num_tokens=16: block 0 runs 8 whole-block slice
    forwards (one per unmask), block 1 runs 7 (window ``[adv | 7
    masks]``); block 0 commits once, block 1 is final. Every slice
    forward carries the full 8-token active block.
    """
    llm = _build_llm()
    sampler = FastDLLMSampler(llm, config=_cfg(history_cache=True))
    sampler.sample(_prompt())
    assert {
        k: v
        for k, v in sampler.history_cache_stats.items()
        if k
        not in (
            "spec_gated_iterations",
            "spec_fused_forwards",
            "spec_budget_skips",
            "spec_budget_escalations",
            "spec_verification_free_blocks",
            "free_verify_blocks",
            "free_verify_low_conf",
            "spec_remask_fallbacks",
            "spec_chunk_forwards",
        )
    } == {
        "prefill_forwards": 1,
        "slice_forwards": 8 + 7,
        "commit_forwards": 1,
        "block_rebuilds": 0,
        "forward_tokens": 8 + 8 * 8 + 8 + 7 * 8,
        "degraded_calls": 0,
        "spec_draft_forwards": 0,
        "spec_verify_forwards": 0,
    }


def test_ut3304_cross_block_ledger_short_final() -> None:
    """UT-3304: the ledger on the short-final-block geometry (num_tokens=12).

    Block 1's canvas tail is not block-aligned (``min(block_size-1,
    remaining-1)`` mask append): the active block is the 4-token
    block-aligned tail ``[adv | 3 masks]`` —slice forwards carry 4
    tokens (never overlapping committed history), and small block 0 of
    the final window (pure committed history) never fires a forward.
    """
    llm = _build_llm()
    sampler = FastDLLMSampler(llm, config=_cfg(history_cache=True, num_tokens=12))
    sampler.sample(_prompt())
    assert {
        k: v
        for k, v in sampler.history_cache_stats.items()
        if k
        not in (
            "spec_gated_iterations",
            "spec_fused_forwards",
            "spec_budget_skips",
            "spec_budget_escalations",
            "spec_verification_free_blocks",
            "free_verify_blocks",
            "free_verify_low_conf",
            "spec_remask_fallbacks",
            "spec_chunk_forwards",
        )
    } == {
        "prefill_forwards": 1,
        "slice_forwards": 8 + 3,  # block 0: 8; final block: sb1 only, 3 unmasks
        "commit_forwards": 1,
        "block_rebuilds": 0,
        "forward_tokens": 8 + 8 * 8 + 8 + 3 * 4,
        "degraded_calls": 0,
        "spec_draft_forwards": 0,
        "spec_verify_forwards": 0,
    }


# ---------------------------------------------------------------------------
# UT-3305: default-off zero regression
# ---------------------------------------------------------------------------


def test_ut3305_default_off_zero_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-3305: default-off keeps every counter at zero and never probes."""
    cfg = FastDLLMSamplerConfig(
        num_tokens=12,
        block_size=8,
        small_block_size=4,
        threshold=1.1,
        temperature=0.0,
        stop_token_id=_STOP_ID,
        pad_token_id=_PAD_ID,
        mask_token_id=_MASK_ID,
        seed=3301,
    )
    assert cfg.history_cache is False
    assert cfg.history_small_block_cache is False

    def _boom(self: FastDLLMSampler) -> Any:
        raise AssertionError("the capability probe must never run on the default-off path")

    monkeypatch.setattr(FastDLLMSampler, "_probe_history_cache_support", _boom)
    llm = _build_llm()
    sampler = FastDLLMSampler(llm, config=cfg)
    out1 = sampler.sample(_prompt())
    out2 = sampler.sample(_prompt())
    assert torch.equal(out1.sequences, out2.sequences)
    assert {
        k: v
        for k, v in sampler.history_cache_stats.items()
        if k
        not in (
            "spec_gated_iterations",
            "spec_fused_forwards",
            "spec_budget_skips",
            "spec_budget_escalations",
            "spec_verification_free_blocks",
            "free_verify_blocks",
            "free_verify_low_conf",
            "spec_remask_fallbacks",
            "spec_chunk_forwards",
        )
    } == {
        "prefill_forwards": 0,
        "slice_forwards": 0,
        "commit_forwards": 0,
        "block_rebuilds": 0,
        "forward_tokens": 0,
        "degraded_calls": 0,
        "spec_draft_forwards": 0,
        "spec_verify_forwards": 0,
    }


# ---------------------------------------------------------------------------
# UT-3306: mutual exclusion and fail-fast
# ---------------------------------------------------------------------------


class _BareModel(nn.Module):
    """Real module whose forward lacks every cache kwarg (probe must fail)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        return self.lm_head(self.embed_tokens(input_ids))


class _KwargsCatchAllModel(nn.Module):
    """Real module that only offers a ``**kwargs`` catch-all (NOT accepted)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def forward(self, input_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return self.lm_head(self.embed_tokens(input_ids))


class _PartialCacheModel(nn.Module):
    """Real module with the required cache kwargs but without the sb set."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
        update_past_key_values: bool = False,
        block_size: int = 32,
    ) -> torch.Tensor:
        del attention_mask, past_key_values, use_cache, update_past_key_values, block_size
        return self.lm_head(self.embed_tokens(input_ids))


def test_ut3306_mutual_exclusion_and_fail_fast() -> None:
    """UT-3306: five illegal combinations, exact exception semantics."""
    # (1) AR034 supersedes the plain cache x speculation exclusion: the
    #     cross-block composition is now legal; the intra-block sb cache is
    #     the v1-undefined face and must still fail fast.
    with pytest.raises(ValueError, match="history_small_block_cache"):
        FastDLLMSamplerConfig(
            history_cache=True,
            history_small_block_cache=True,
            num_block_candidates=4,
        )
    FastDLLMSamplerConfig(history_cache=True, num_block_candidates=4)  # legal (AR034)
    # (2) sb sub-switch without the cached strategy.
    with pytest.raises(ValueError, match="requires history_cache=True"):
        FastDLLMSamplerConfig(history_small_block_cache=True)
    # (3) folded_model x cache (sample-time —folded_model is a runtime arg).
    sampler = FastDLLMSampler(_build_llm(), config=_cfg(history_cache=True))
    with pytest.raises(ValueError, match="mutually exclusive with a"):
        sampler.sample(_prompt(), folded_model=object())
    # (4) missing required cache kwargs -> RuntimeError with probe details.
    bare = FastDLLMSampler(_TinyDiffusionLLM(_BareModel()), config=_cfg(history_cache=True))
    with pytest.raises(RuntimeError, match="use_cache"):
        bare.sample(_prompt())
    # (5) sb sub-switch on but the block-cache kwargs are missing.
    part = FastDLLMSampler(
        _TinyDiffusionLLM(_PartialCacheModel()),
        config=_cfg(history_cache=True, history_small_block_cache=True),
    )
    with pytest.raises(RuntimeError, match="use_block_cache"):
        part.sample(_prompt())


# ---------------------------------------------------------------------------
# UT-3307: capability probe + stats schema
# ---------------------------------------------------------------------------


def test_ut3307_capability_probe() -> None:
    """UT-3307: named-parameter strictness, result caching, full fixture face."""
    sampler = FastDLLMSampler(_build_llm(), config=_cfg(history_cache=True))
    support = sampler._probe_history_cache_support()
    assert support.has_raw_model is True
    assert support.required_ok is True
    assert support.small_block_ok is True
    assert all(support.accepted[k] for k in _HC_PROBED_KWARGS)
    # The probe result is cached on the instance (same object on re-read).
    assert sampler._probe_history_cache_support() is support

    # A **kwargs catch-all is NOT acceptance —fail fast at sample().
    catch_all = FastDLLMSampler(
        _TinyDiffusionLLM(_KwargsCatchAllModel()), config=_cfg(history_cache=True)
    )
    support2 = catch_all._probe_history_cache_support()
    assert support2.required_ok is False
    assert support2.small_block_ok is False
    with pytest.raises(RuntimeError, match=r"\*\*kwargs"):
        catch_all.sample(_prompt())


def test_ut3307_stats_schema_and_cumulation() -> None:
    """UT-3307: stats schema, fresh-dict reads, cumulation across calls."""
    sampler = FastDLLMSampler(_build_llm(), config=_cfg(history_cache=True))
    keys = {
        "prefill_forwards",
        "slice_forwards",
        "commit_forwards",
        "block_rebuilds",
        "forward_tokens",
        "degraded_calls",
        "spec_draft_forwards",
        "spec_verify_forwards",
        "spec_gated_iterations",
        "spec_fused_forwards",
        "spec_budget_skips",
        "spec_budget_escalations",
        "spec_verification_free_blocks",
        "free_verify_blocks",
        "free_verify_low_conf",
        "spec_remask_fallbacks",
        "spec_chunk_forwards",
    }
    assert set(sampler.history_cache_stats) == keys
    read = sampler.history_cache_stats
    read["prefill_forwards"] = 999  # mutating the return never leaks in
    assert sampler.history_cache_stats["prefill_forwards"] == 0
    sampler.sample(_prompt())
    assert sampler.history_cache_stats["prefill_forwards"] == 1
    assert sampler.history_cache_stats["degraded_calls"] == 0
    sampler.sample(_prompt())
    assert sampler.history_cache_stats["prefill_forwards"] == 2
