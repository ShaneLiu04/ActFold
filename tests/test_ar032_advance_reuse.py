"""AR032 tests: speculative verification-logit advance reuse.

Covers the opt-in mechanism (FR1), the equivalence hard gates (FR2), and
the default-off zero-regression faces. UT-3170..UT-3177.

UT numbering continues from UT-3170 (AR031 ended at UT-3169).

Contract: ``specs/changes/AR032-*/design.md`` §4.2 — the opt-in
``FastDLLMSamplerConfig.speculative_advance_reuse`` (T001 delivered the
config face, pinned by UT-3170) replaces, when ``True`` and the
speculative path is active (``num_block_candidates > 0`` +
``folded_model``), the per-block block-advance serial forward with a read
of the LAST small block's winning verification row's tail logits
(``next_token = stash.argmax(-1)``; ``next_token[finished] = pad_id``;
canvas cat + fresh-mask cat + history append preserved verbatim). The
folding chain continues directly onto the winner candidate branch — no
``fastdllm_ar_{block_idx}`` branch is created. Per block-advance-eligible
block the MFF forward count drops from ``2*num_small_blocks + 1`` to
``2*num_small_blocks``; the final non-advancing block (``remaining <= 0``)
has 0 advance forwards on BOTH arms; the on/off difference over a whole
run is exactly the number of eligible blocks.

Red evidence (kept for the record; T002 Green delivered the mechanism):
UT-3172/UT-3177 originally failed on the production-missing faces only —
before T002, ``sample`` never consumed ``speculative_advance_reuse``, so
the on-arm still ran the block-advance serial forward (3 forwards per
eligible block, a ``fastdllm_ar_`` branch was created). UT-3171
(default-off zero regression) passed in the Red state by design — it pins
that an explicit ``False`` is bitwise the default config on the
speculative, folded-serial, and raw paths.

Fixtures follow ``tests/test_ar031_sampler_uncertainty.py`` (self-contained
mini block-causal decoder, ``_MiniDiffusionLLM``, ``_SpyMFF``, ``_make_mff``,
pinned-seed builds) — copied minimal, CPU, real tensors, zero mocks.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from actfold.core import ActivationCache, SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.models.base import DiffusionLLM
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig
from actfold.models.sampling_utils import right_shift_logits
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from actfold.utils.config_manager import ActFoldConfig

_VOCAB = 64
_HIDDEN = 32
_NUM_LAYERS = 2
_TAU = 0.95
_CACHE_BUDGET = 1024
_MASK_ID = 63
_EOS_ID = 62
_PAD_ID = 61
_MODEL_SEED = 2026
_BLOCK_SIZE = 8
_SMALL_BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def _profiler_state() -> Iterator[None]:
    """Snapshot-reset the global stability profiler around every case."""
    GLOBAL_STABILITY_PROFILER.reset()
    yield
    GLOBAL_STABILITY_PROFILER.reset()


def test_ut_3170_advance_reuse_config_defaults_and_validation() -> None:
    """UT-3170: config face — defaults, True+N=0 ValueError, YAML consumption."""
    # Default is False (zero-regression contract).
    cfg = FastDLLMSamplerConfig()
    assert cfg.speculative_advance_reuse is False
    # True requires the speculative path (N > 0), the #44-style boundary.
    with pytest.raises(ValueError) as exc_info:
        FastDLLMSamplerConfig(speculative_advance_reuse=True)
    msg = str(exc_info.value)
    assert "speculative_advance_reuse" in msg
    assert "num_block_candidates" in msg
    # Legal construction with N > 0.
    ok = FastDLLMSamplerConfig(num_block_candidates=4, speculative_advance_reuse=True)
    assert ok.speculative_advance_reuse is True


def test_ut_3170_actfold_config_yaml_consumption(tmp_path) -> None:
    """UT-3170: ActFoldConfig consumes the YAML key without unknown-key warnings."""
    cfg = ActFoldConfig(speculative_advance_reuse=True)
    assert cfg.speculative_advance_reuse is True
    assert ActFoldConfig().speculative_advance_reuse is False
    # load_config face: the key must be a recognized field (no UserWarning).
    from actfold.utils.config_manager import load_config

    yaml_text = "model_name_or_path: dummy\n" "speculative_advance_reuse: true\n"
    path = tmp_path / "ar032_cfg.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = load_config(str(path))
    assert loaded.speculative_advance_reuse is True


# ---------------------------------------------------------------------------
# Self-contained mini block-causal decoder (FastDLLM fixture family,
# copied minimal from tests/test_ar031_sampler_uncertainty.py)
# ---------------------------------------------------------------------------


class _DualFormAttention(nn.Module):
    """One attention implementation accepting both production mask forms.

    Raw form: ``[B, T]`` padding mask (the block-causal AND non-pad mask is
    built internally). Folded form: ``[T, T]`` prebuilt block-causal AND
    non-pad mask.
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
    """Pre-norm block-causal decoder layer with a detectable split FFN chain."""

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


class _MiniBlockCausalModel(nn.Module):
    """Mock LLaMA-layout block-causal decoder: embed -> layers -> norm -> head."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": _VOCAB})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.model.layers = nn.ModuleList(
            [_DualFormLayer(_HIDDEN, _BLOCK_SIZE) for _ in range(_NUM_LAYERS)]
        )
        self.model.norm = nn.LayerNorm(_HIDDEN)
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


class _StopHead(nn.Module):
    """Constant head whose argmax is the stop token at EVERY position.

    ``logits[..., stop_id] = 5.0`` and ``0`` elsewhere: the draft argmax is
    the stop token everywhere, so the greedy candidate row commits the stop
    token in the first speculative small block (``finished`` flips True
    mid-generation), and the winner's tail-logit argmax is also the stop
    token — so the on-arm advance-reuse read MUST hit the
    ``next_token[finished] = pad_id`` face.
    """

    def __init__(self, vocab: int, stop_id: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.stop_id = stop_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*x.shape[:-1], self.vocab, dtype=x.dtype, device=x.device)
        logits[..., self.stop_id] = 5.0
        return logits


class _MiniTokenizer:
    """Minimal tokenizer exposing the special ids the samplers resolve."""

    def __init__(self, mask_id: int, eos_id: int, pad_id: int) -> None:
        self.mask_token_id = mask_id
        self.eos_token_id = eos_id
        self.bos_token_id = 1
        self.pad_token_id = pad_id


class _MiniDiffusionLLM(DiffusionLLM):
    """DiffusionLLM wrapper around a mini decoder, recording raw forwards."""

    def __init__(
        self,
        inner: nn.Module,
        vocab: int = _VOCAB,
        tokenizer: _MiniTokenizer | None = None,
    ) -> None:
        super().__init__("mini-bidirectional")
        self.model = inner
        self._vocab = vocab
        self.tokenizer = tokenizer or _MiniTokenizer(_MASK_ID, _EOS_ID, _PAD_ID)
        self.raw_calls: list[dict[str, Any]] = []

    def forward(self, tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.raw_calls.append({"tokens": tokens.detach().clone(), "kwargs": dict(kwargs)})
        return self.model(tokens, **kwargs)

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
        return self._vocab


class _SpyMFF(ManualFoldedForward):
    """ManualFoldedForward that records every forward call and its output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict[str, Any]] = []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = super().forward(*args, **kwargs)
        self.calls.append({"args": args, "kwargs": kwargs, "out": out})
        return out


def _build_block_causal_mini() -> _MiniBlockCausalModel:
    """Deterministically build the mini block-causal decoder (CPU)."""
    torch.manual_seed(_MODEL_SEED)
    model = _MiniBlockCausalModel()
    model.eval()
    return model


def _make_mff(model: nn.Module, **extra: Any) -> _SpyMFF:
    """Fresh ManualFoldedForward (own cache + cosine gate) over the model (CPU)."""
    cache = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device="cpu")
    gate = SimilarityGate(tau=_TAU, metric="cosine")
    return _SpyMFF(
        model,
        cache=cache,
        gate=gate,
        scheduler=None,
        split_layers=False,
        split_min_tokens=512,
        use_cuda_graph=False,
        graph_capacity_ratio=0.5,
        **extra,
    )


def _prompt() -> torch.Tensor:
    """The shared 3-token batch-1 prompt."""
    return torch.tensor([[1, 2, 3]])


def _fastdllm_config(**overrides: Any) -> FastDLLMSamplerConfig:
    """FastDLLMSamplerConfig with mini-model token ids and pinned defaults."""
    values: dict[str, Any] = {
        "num_tokens": 8,
        "temperature": 0.0,
        "block_size": _BLOCK_SIZE,
        "small_block_size": _SMALL_BLOCK_SIZE,
        "threshold": 0.95,
        "mask_token_id": _MASK_ID,
        "stop_token_id": _EOS_ID,
        "pad_token_id": _PAD_ID,
        "seed": 2100,
    }
    values.update(overrides)
    return FastDLLMSamplerConfig(**values)


def _batched_calls(mff: _SpyMFF) -> list[dict[str, Any]]:
    """MFF calls carrying ``branch_ids`` (the [N, T] batched verifications)."""
    return [c for c in mff.calls if c["kwargs"].get("branch_ids") is not None]


def _serial_calls(mff: _SpyMFF) -> list[dict[str, Any]]:
    """MFF calls without ``branch_ids`` (draft / block-advance forwards)."""
    return [c for c in mff.calls if c["kwargs"].get("branch_ids") is None]


def _ar_advance_calls(mff: _SpyMFF) -> list[dict[str, Any]]:
    """Serial MFF calls whose branch id is a block-advance branch."""
    return [
        c
        for c in mff.calls
        if c["kwargs"].get("branch_ids") is None
        and str(c["kwargs"].get("branch_id", "")).startswith("fastdllm_ar_")
    ]


# ---------------------------------------------------------------------------
# UT-3171: default-off zero regression (FR1 + FR2 AC4) — PASSES today
# ---------------------------------------------------------------------------


def test_ut3171_default_off_zero_regression() -> None:
    """UT-3171: explicit ``speculative_advance_reuse=False`` is bitwise the default.

    (a) Speculative path (N=2, candidate_top_k=2, greedy on, seed pinned,
    return_history=True): the explicit-False arm and the default-config arm
    (field absent) produce bitwise-equal decode_final sequences AND equal
    history snapshot lists.
    (b) Non-speculative clause: N=0 over the folded-serial path
    (folded_model given) and the raw baseline path (folded_model=None) —
    explicit False vs default, bitwise equal on decode_final.

    Scope note (T007 review): this default-vs-explicit-False A/B cannot
    detect a regression that affects BOTH arms equally; pinning against
    the literal AR031 end state relies on the pre-existing AR021-AR031
    sentinel suites plus the verbatim preservation of the off path (the
    advance segment's else-branch is the AR031 code untouched).
    """
    inner = _build_block_causal_mini()
    llm = _MiniDiffusionLLM(inner)

    # (a) speculative path.
    spec: dict[str, Any] = {
        "num_tokens": 16,
        "seed": 3171,
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "return_history": True,
    }
    out_default = FastDLLMSampler(llm, config=_fastdllm_config(**spec)).sample(
        _prompt(), folded_model=_make_mff(inner)
    )
    out_off = FastDLLMSampler(
        llm, config=_fastdllm_config(**spec, speculative_advance_reuse=False)
    ).sample(_prompt(), folded_model=_make_mff(inner))
    assert torch.equal(out_off.sequences, out_default.sequences)
    assert len(out_off.history) == len(out_default.history)
    for h_off, h_default in zip(out_off.history, out_default.history):
        assert torch.equal(h_off, h_default)

    # (b) N=0 folded-serial path (folded_model given, remasking recipe).
    serial: dict[str, Any] = {"num_tokens": 16, "seed": 3171}
    out_s_default = FastDLLMSampler(llm, config=_fastdllm_config(**serial)).sample(
        _prompt(), folded_model=_make_mff(inner)
    )
    out_s_off = FastDLLMSampler(
        llm, config=_fastdllm_config(**serial, speculative_advance_reuse=False)
    ).sample(_prompt(), folded_model=_make_mff(inner))
    assert torch.equal(out_s_off.sequences, out_s_default.sequences)

    # (b) raw baseline path (folded_model=None).
    out_b_default = FastDLLMSampler(llm, config=_fastdllm_config(**serial)).sample(_prompt())
    out_b_off = FastDLLMSampler(
        llm, config=_fastdllm_config(**serial, speculative_advance_reuse=False)
    ).sample(_prompt())
    assert torch.equal(out_b_off.sequences, out_b_default.sequences)


# ---------------------------------------------------------------------------
# UT-3172: forward-count probe, S=1 geometry — designed Red on the on-arm
# ---------------------------------------------------------------------------


def test_ut3172_forward_count_probe_s1() -> None:
    """UT-3172: on-arm omits one serial forward per block-advance-eligible block.

    Geometry: ``block_size == small_block_size`` (S=1) and ``num_tokens=24``
    over ``block_size=8`` -> 3 decode blocks; blocks 0 and 1 are
    block-advance-eligible (``remaining > 0`` after the block decode) and
    block 2 is the final non-advancing block (``remaining <= 0``, 0 advance
    forwards on BOTH arms). Same seed and identical model/mff construction
    per arm (fresh _SpyMFF per arm).

    Hard faces: on-arm total MFF forwards == 2*num_blocks (draft serial +
    batched verify per block, zero advance), off-arm == 2*num_blocks + 2;
    the on/off difference == the number of eligible blocks (2); both arms
    have the same number of batched calls (num_blocks); on-arm serial ==
    draft count only (num_blocks), off-arm serial == draft + advance
    (num_blocks + 2). The per-block attribution is derived from call
    ordering: every draft serial immediately precedes its batched call.

    Red: the mechanism is absent — the on-arm currently runs the
    block-advance serial forward per eligible block, so the on-arm total is
    8 (expected 6) and the on-arm serial count is 5 (expected 3).
    """
    num_blocks = 3
    num_eligible = 2

    def run_arm(reuse: bool) -> tuple[Any, _SpyMFF]:
        inner = _build_block_causal_mini()
        llm = _MiniDiffusionLLM(inner)
        mff = _make_mff(inner)
        cfg = _fastdllm_config(
            num_tokens=24,
            block_size=_BLOCK_SIZE,
            small_block_size=_BLOCK_SIZE,  # S=1: one small block per block
            seed=3172,
            num_block_candidates=2,
            candidate_top_k=2,
            candidate_include_greedy=True,
            speculative_advance_reuse=reuse,
        )
        out = FastDLLMSampler(llm, config=cfg).sample(_prompt(), folded_model=mff)
        return out, mff

    out_off, mff_off = run_arm(False)
    out_on, mff_on = run_arm(True)
    assert out_on.sequences.shape[0] == 1 and out_on.sequences.shape[1] > 0
    assert out_off.sequences.shape[0] == 1 and out_off.sequences.shape[1] > 0

    calls_on = mff_on.calls
    calls_off = mff_off.calls
    batched_on = _batched_calls(mff_on)
    batched_off = _batched_calls(mff_off)
    serial_on = _serial_calls(mff_on)
    serial_off = _serial_calls(mff_off)

    # Hard faces: total counts, on/off difference, batched equality.
    assert len(calls_on) == 2 * num_blocks  # Red today: 8 (advance still runs)
    assert len(calls_off) == 2 * num_blocks + num_eligible
    assert len(calls_off) - len(calls_on) == num_eligible
    assert len(batched_on) == len(batched_off) == num_blocks
    assert len(serial_on) == num_blocks  # draft only — Red today: 5
    assert len(serial_off) == num_blocks + num_eligible  # draft + advance

    # Derived per-block attribution from call ordering: with S=1 each block
    # contributes exactly one (draft serial -> batched verify) pair; the
    # off-arm's eligible blocks add one trailing advance serial each. The
    # on-arm must alternate serial/batched with no trailing serials, so
    # every eligible block totals exactly 2 forwards and the final block
    # totals 2 on BOTH arms (0 advance forwards on the final block).
    types_on = ["B" if c["kwargs"].get("branch_ids") is not None else "S" for c in calls_on]
    assert types_on == ["S", "B"] * num_blocks
    types_off = ["B" if c["kwargs"].get("branch_ids") is not None else "S" for c in calls_off]
    expected_off: list[str] = []
    for block_idx in range(num_blocks):
        expected_off += ["S", "B"] + (["S"] if block_idx < num_blocks - 1 else [])
    assert types_off == expected_off


# ---------------------------------------------------------------------------
# UT-3177: advance semantics — stop face, canvas growth, snapshot count,
# chain continuation — designed Red on the on-arm chain faces
# ---------------------------------------------------------------------------


def test_ut3177_advance_semantics_stop_and_history() -> None:
    """UT-3177: on-arm advance semantics with a mid-generation stop commit.

    Fixture: a constant ``_StopHead`` whose argmax is the stop token at
    every position — the greedy candidate row wins every small block
    (per-fill posterior ``e^5/(e^5+V-1)`` vs the uniform ``1/(e^5+V-1)`` for
    any non-stop fill) and commits the stop token, so ``finished`` flips
    True during the FIRST speculative block. Block 0 is
    block-advance-eligible (``remaining > 0``); block 1 is skipped by the
    ``finished.all()`` loop guard.

    Faces:
    * ``next_token[finished] = pad_id``: the position after the committed
      stop block in the final canvas snapshot is pad_id on BOTH arms (the
      on-arm's stash tail-logit argmax IS the stop token — only the
      finished override turns it into pad). Pinned face: on/off decode_final
      equality (the constant head makes the argmax deterministic, immune to
      any fold ULP), plus the on-arm alone: no mask_id in the decoded
      region, decode trimmed at the first stop token, prompt prefix intact.
    * Canvas-growth isomorphism: on/off history snapshot canvas-length
      sequences are equal; snapshot counts on == off (D4: the advance-tail
      snapshot append is preserved).
    * Chain: the on-arm folding chain continues directly onto the winner
      candidate branch — NO ``fastdllm_ar_{block_idx}`` branch is created
      and serial calls are draft-only (== small-block count). Red today:
      the mechanism is absent, so the on-arm still creates the
      ``fastdllm_ar_0`` advance branch (3 serial vs 2 batched calls).
    """
    num_small_blocks = _BLOCK_SIZE // _SMALL_BLOCK_SIZE  # 2

    def run_arm(reuse: bool) -> tuple[Any, _SpyMFF]:
        inner = _build_block_causal_mini()
        inner.lm_head = _StopHead(_VOCAB, _EOS_ID)
        llm = _MiniDiffusionLLM(inner)
        mff = _make_mff(inner)
        cfg = _fastdllm_config(
            num_tokens=16,
            block_size=_BLOCK_SIZE,
            small_block_size=_SMALL_BLOCK_SIZE,
            seed=3177,
            num_block_candidates=2,
            candidate_top_k=2,
            candidate_include_greedy=True,
            return_history=True,
            speculative_advance_reuse=reuse,
        )
        out = FastDLLMSampler(llm, config=cfg).sample(_prompt(), folded_model=mff)
        return out, mff

    out_off, mff_off = run_arm(False)
    out_on, mff_on = run_arm(True)

    # Stop face (pinned: on/off decode equality — deterministic argmax).
    assert torch.equal(out_on.sequences, out_off.sequences)
    row_on = out_on.sequences[0].tolist()
    assert _MASK_ID not in row_on
    assert row_on[:3] == [1, 2, 3]
    assert row_on[-1] == _EOS_ID
    assert _EOS_ID not in row_on[:-1]

    # ``next_token[finished] = pad_id``: the canvas position right after the
    # committed stop block is pad (not another stop) on both arms.
    snap_on = out_on.history[-1]
    stop_idx_on = int((snap_on[0] == _EOS_ID).nonzero(as_tuple=False)[0, 0])
    assert int(snap_on[0, stop_idx_on + _BLOCK_SIZE]) == _PAD_ID
    snap_off = out_off.history[-1]
    stop_idx_off = int((snap_off[0] == _EOS_ID).nonzero(as_tuple=False)[0, 0])
    assert int(snap_off[0, stop_idx_off + _BLOCK_SIZE]) == _PAD_ID

    # Canvas-growth isomorphism + history snapshot count (D4).
    lens_on = [int(h.shape[1]) for h in out_on.history]
    lens_off = [int(h.shape[1]) for h in out_off.history]
    assert lens_on == lens_off
    # 1 initial snapshot + 1 per speculative small block + 1 advance-tail.
    assert len(out_on.history) == len(out_off.history) == 1 + num_small_blocks + 1

    # Chain faces (designed Red): on-arm — no block-advance branch, serial
    # calls are draft-only (== batched == small-block count).
    assert _ar_advance_calls(mff_on) == []  # Red today: fastdllm_ar_0 exists
    assert len(_serial_calls(mff_on)) == num_small_blocks  # Red today: 3
    assert len(_batched_calls(mff_on)) == num_small_blocks
    # Off-arm sanity: the advance path IS exercised exactly once (block 0).
    assert len(_ar_advance_calls(mff_off)) == 1
    assert len(_serial_calls(mff_off)) == num_small_blocks + 1


# ---------------------------------------------------------------------------
# AR032 T003 (FR2): equivalence hard-acceptance faces — UT-3173..UT-3176
# (file order: appended after UT-3177; UT numbers are design.md §6.1 ids)
# ---------------------------------------------------------------------------


def _make_mff_with_tau(model: nn.Module, tau: float) -> _SpyMFF:
    """Fresh ManualFoldedForward with an explicit gate tau (CPU, cosine).

    Same construction as ``_make_mff`` but with the gate tau as a parameter —
    UT-3175 needs the tau=1.0 eager hard-gate variant (nothing is stable:
    cosine similarity clamps to <= 1.0 and stability requires sim > tau, so
    every folded forward is a full recompute).
    """
    cache = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device="cpu")
    gate = SimilarityGate(tau=tau, metric="cosine")
    return _SpyMFF(
        model,
        cache=cache,
        gate=gate,
        scheduler=None,
        split_layers=False,
        split_min_tokens=512,
        use_cuda_graph=False,
        graph_capacity_ratio=0.5,
    )


class _WinnerRowRecorder(FastDLLMSampler):
    """Records each speculative small block's winning row index (UT-3173).

    A bare ``super()`` delegation around ``_speculative_small_block``;
    batched verification calls and ``_speculative_small_block`` calls are
    1:1, so ``winner_rows[k]`` is the ``j_star`` of the k-th batched call.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.winner_rows: list[int] = []

    def _speculative_small_block(self, *args: Any, **kwargs: Any) -> Any:
        j_star, tail_logits = super()._speculative_small_block(*args, **kwargs)
        self.winner_rows.append(int(j_star))
        return j_star, tail_logits


def test_ut3173_advance_reuse_winner_row_carried_forward() -> None:
    """UT-3173: input-equivalence premise — the committed canvas after each
    speculative small block IS the WINNER row ``cands[j_star]`` of that
    block's ``[N, T]`` cands canvas (design row: ``x == cands[j_star]``
    bitwise; T007 review strengthened this from the earlier some-row form,
    which a wrong-row commit bug would have passed).

    S=1 geometry (``small_block_size == block_size``, ``num_tokens=24`` ->
    >= 2 decode blocks), N=2 greedy-anchored, reuse ON, return_history=True.
    For every batched verification call (``kwargs["branch_ids"]`` is not
    None) the recorded ``args[0]`` is the ``[N, T]`` cands canvas, and the
    recorded ``j_star`` (the ``_WinnerRowRecorder`` probe, 1:1 with the
    batched calls) selects the row that must equal the canvas the NEXT
    forward call sees (its input tokens row 0, truncated to the cands
    length) — the winning verification row is carried forward verbatim as
    the committed canvas. The comparison is exact post-hoc because the
    canvas is append-only beyond the cands length (later commits touch only
    positions >= the next block's start, which is at or beyond the current
    cands length). The final batched call has no successor forward: its
    committed canvas is the last history snapshot (appended post-commit
    inside the speculative small block).
    """
    inner = _build_block_causal_mini()
    llm = _MiniDiffusionLLM(inner)
    mff = _make_mff(inner)
    cfg = _fastdllm_config(
        num_tokens=24,
        block_size=_BLOCK_SIZE,
        small_block_size=_BLOCK_SIZE,  # S=1: one small block per block
        seed=3173,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        return_history=True,
        speculative_advance_reuse=True,
    )
    sampler = _WinnerRowRecorder(llm, config=cfg)
    out = sampler.sample(_prompt(), folded_model=mff)

    calls = mff.calls
    batched_positions = [
        i for i, c in enumerate(calls) if c["kwargs"].get("branch_ids") is not None
    ]
    # At least two blocks ran the speculative path (S=1 -> one batched call
    # per block), and the winner-row probe saw exactly one j_star per
    # batched call (1:1).
    assert len(batched_positions) >= 2
    assert len(sampler.winner_rows) == len(batched_positions)
    for k, i in enumerate(batched_positions):
        cands = calls[i]["args"][0]
        j_star = sampler.winner_rows[k]
        seq_len = int(cands.shape[1])
        if i + 1 < len(calls):
            committed = calls[i + 1]["args"][0][0, :seq_len]
        else:
            committed = out.history[-1][0, :seq_len]
        assert torch.equal(
            cands[j_star], committed
        ), f"batched call {i}: committed canvas != winner row cands[j_star={j_star}]"


class _PassThroughLayer(nn.Module):
    """Token-wise identity layer accepting the folded-call kwarg superset."""

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return hidden_states


class _TableHeadModel(nn.Module):
    """Content-conditioned table decoder: ``logits[b, t] = table[tokens[b, t]]``.

    Identity embedding (one-hot), pass-through layers, NO final norm, and an
    ``lm_head`` whose weight column ``i`` is the logits row for token ``i``.
    The mapping stays inside the REAL token range ``0.._TABLE_REAL-1``
    (``% _TABLE_REAL``): every token's top-2 is
    ``{(tok+1) % 60, (tok+2) % 60}`` (values 5.0 / 1.0, no ties), so
    candidate fills can never be pad/eos/mask — the right-shift draft
    convention (position p draws from ``W[:, x[p-1]]``) once filled EOS from
    the pad-predecessor row and ended the run early. A forward's logits are
    therefore an exact lookup of the input tokens: every path (serial /
    batched / folded / raw) is bit-identical for the same token row, and two
    rows that differ at the tail position have provably different tail-logit
    argmaxes — the wrong-source discriminator for UT-3174 (the FIRST small
    block's winner row still carries ``mask_id`` at the canvas tail while
    the LAST small block's winner row has a committed real token there).
    """

    _TABLE_REAL = 60  # fills stay in 0..59: never pad(61)/eos(62)/mask(63)

    def __init__(self, vocab: int = _VOCAB) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": vocab})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, vocab)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(torch.eye(vocab))
        self.model.layers = nn.ModuleList([_PassThroughLayer() for _ in range(_NUM_LAYERS)])
        table = torch.zeros(vocab, vocab)
        for tok in range(vocab):
            table[(tok + 1) % self._TABLE_REAL, tok] = 5.0
            table[(tok + 2) % self._TABLE_REAL, tok] = 1.0
        self.lm_head = nn.Linear(vocab, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(table)

    def forward(self, tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        hidden = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return self.lm_head(hidden)


def test_ut3174_s2_wrong_source_capture() -> None:
    """UT-3174: S>1 geometry — count faces + the wrong-source stash contract.

    Geometry: ``block_size=8``, ``small_block_size=4``, ``num_tokens=16`` ->
    2 blocks, S=2; only block 0 is block-advance-eligible (block 1 is the
    final non-advancing block, ``remaining <= 0``).

    (a) Count face: per eligible block the on-arm runs ``2*S`` MFF forwards
    (one draft serial + one ``[N, T]`` batched verification per small block,
    ZERO advance) and the off-arm ``2*S + 1``; the on/off total difference
    == the number of eligible blocks.

    (b) Wrong-source face on the content-conditioned table model: the FIRST
    small block's winner-row tail logits (canvas tail still ``mask_id``)
    have a DIFFERENT argmax than the LAST small block's winner-row tail
    logits (tail committed to a real token), so a stash captured from the
    wrong small block is detectable. The correct implementation's advance
    stash (captured by wrapping the sampler instance's
    ``_speculative_small_block`` after construction, before ``sample``) is
    bitwise the LAST small block's batched winner-row tail —
    ``right_shift_logits(out)[j_star, -1:, :]`` recomputed from the
    ``_SpyMFF``-recorded batched ``out``. The exact-lookup head also makes
    the on/off decodes bitwise equal. Defect injection (the capture-power
    face): a wrapper that returns the FIRST small block's tail for every
    subsequent call — the classic wrong-source bug — changes the decode,
    proving this test FAILS when the mechanism is broken.
    """
    num_blocks = 2
    small_per_block = _BLOCK_SIZE // _SMALL_BLOCK_SIZE  # S = 2
    num_eligible = 1  # block 0 only; block 1 is the final non-advancing block

    def run_arm(
        reuse: bool, inject_wrong_source: bool = False
    ) -> tuple[Any, _SpyMFF, list[dict[str, Any]]]:
        inner = _TableHeadModel()
        llm = _MiniDiffusionLLM(inner)
        mff = _make_mff(inner)
        cfg = _fastdllm_config(
            num_tokens=16,
            block_size=_BLOCK_SIZE,
            small_block_size=_SMALL_BLOCK_SIZE,
            seed=3174,
            num_block_candidates=2,
            candidate_top_k=2,
            candidate_include_greedy=True,
            return_history=True,
            speculative_advance_reuse=reuse,
        )
        sampler = FastDLLMSampler(llm, config=cfg)
        captured: list[dict[str, Any]] = []
        orig = sampler._speculative_small_block
        if inject_wrong_source:
            first_tail: dict[str, torch.Tensor | None] = {"tail": None}

            def broken(*args: Any, **kwargs: Any) -> tuple[int, torch.Tensor]:
                j_star, tail = orig(*args, **kwargs)
                if first_tail["tail"] is None:
                    first_tail["tail"] = tail
                return j_star, first_tail["tail"]

            sampler._speculative_small_block = broken
        else:

            def wrapped(*args: Any, **kwargs: Any) -> tuple[int, torch.Tensor]:
                j_star, tail = orig(*args, **kwargs)
                captured.append(
                    {
                        # positional call contract of _speculative_small_block
                        # (fast_dllm_sampler.py sample loop): args[9] ==
                        # block_idx, args[10] == small_block_idx.
                        "block_idx": args[9],
                        "small_block_idx": args[10],
                        "j_star": j_star,
                        "tail": tail.clone(),
                    }
                )
                return j_star, tail

            sampler._speculative_small_block = wrapped
        out = sampler.sample(_prompt(), folded_model=mff)
        return out, mff, captured

    out_on, mff_on, cap_on = run_arm(reuse=True)
    out_off, mff_off, _cap_off = run_arm(reuse=False)
    out_bad, _mff_bad, _cap_bad = run_arm(reuse=True, inject_wrong_source=True)

    # (a) Count faces: totals, on/off difference, batched equality.
    assert len(mff_on.calls) == 2 * small_per_block * num_blocks  # 8
    assert len(mff_off.calls) == 2 * small_per_block * num_blocks + num_eligible  # 9
    assert len(mff_off.calls) - len(mff_on.calls) == num_eligible
    assert (
        len(_batched_calls(mff_on)) == len(_batched_calls(mff_off)) == small_per_block * num_blocks
    )
    assert len(_serial_calls(mff_on)) == small_per_block * num_blocks  # draft only
    assert (
        len(_serial_calls(mff_off)) == small_per_block * num_blocks + num_eligible
    )  # draft + advance
    assert _ar_advance_calls(mff_on) == []
    assert len(_ar_advance_calls(mff_off)) == num_eligible
    types_on = ["B" if c["kwargs"].get("branch_ids") is not None else "S" for c in mff_on.calls]
    assert types_on == ["S", "B"] * (small_per_block * num_blocks)
    types_off = ["B" if c["kwargs"].get("branch_ids") is not None else "S" for c in mff_off.calls]
    assert types_off == ["S", "B", "S", "B", "S"] + ["S", "B", "S", "B"]

    # (b) Wrong-source faces. Capture structure: one speculative call per
    # small block, in block order — cap_on[1] is block 0's LAST small block.
    assert len(cap_on) == small_per_block * num_blocks
    assert (cap_on[0]["block_idx"], cap_on[0]["small_block_idx"]) == (0, 0)
    assert (cap_on[1]["block_idx"], cap_on[1]["small_block_idx"]) == (
        0,
        small_per_block - 1,
    )
    # The discriminator premise: the first small block's winner-row tail
    # (canvas tail still mask_id) has a different argmax than the last
    # small block's (tail committed) — otherwise a wrong-source stash
    # would be undetectable at this fixture.
    first_tail_argmax = int(cap_on[0]["tail"].argmax())
    last_tail_argmax = int(cap_on[1]["tail"].argmax())
    assert first_tail_argmax != last_tail_argmax
    # The stash used at the block-0 advance seam is bitwise the LAST small
    # block's batched winner-row tail, recomputed from the recorded MFF out.
    batched_on = _batched_calls(mff_on)
    assert len(batched_on) == len(cap_on)
    expected_tail = right_shift_logits(batched_on[1]["out"])[cap_on[1]["j_star"], -1:, :]
    assert torch.equal(cap_on[1]["tail"], expected_tail)
    # Exact-lookup head: the on-arm (stash advance) and off-arm (serial
    # advance) decodes are bitwise equal.
    assert torch.equal(out_on.sequences, out_off.sequences)
    row = out_on.sequences[0].tolist()
    assert _MASK_ID not in row
    assert row[:3] == [1, 2, 3]
    # Defect injection: the wrong-source stash (first small block's tail for
    # every subsequent call) changes the decode — this face FAILS against
    # the broken capture, proving the test's capture power.
    assert not torch.equal(out_bad.sequences, out_on.sequences)
    if out_bad.sequences.shape == out_on.sequences.shape:
        # The block-0 advance token sits right after prompt + first-block
        # padding + block 0 on the (untrimmed) decoded canvas.
        advance_pos = 3 + (_BLOCK_SIZE - 3) + _BLOCK_SIZE
        assert int(out_bad.sequences[0, advance_pos]) != int(out_on.sequences[0, advance_pos])


def test_ut3175_tau_one_eager_hard_gate() -> None:
    """UT-3175: tau=1.0 eager hard gate — on/off bitwise equality.

    With ``SimilarityGate(tau=1.0, metric="cosine")`` nothing is stable
    (cosine similarity clamps to <= 1.0 and stability requires ``sim >
    tau``), so every folded forward is a full recompute and the CPU
    non-split batched/serial forwards agree on the same input. The on-arm
    (advance reuse: the next token comes from the last small block's
    winning verification row's tail logits) and the off-arm (the serial
    block-advance forward) must then produce bitwise-equal decode_final
    sequences AND equal history snapshots — divergence here is a mechanism
    semantics error, not ULP noise (the hard gate of FR2).
    """
    spec: dict[str, Any] = {
        "num_tokens": 16,
        "block_size": _BLOCK_SIZE,
        "small_block_size": _SMALL_BLOCK_SIZE,
        "seed": 3175,
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "return_history": True,
    }

    def run_arm(reuse: bool) -> Any:
        inner = _build_block_causal_mini()
        llm = _MiniDiffusionLLM(inner)
        mff = _make_mff_with_tau(inner, tau=1.0)
        cfg = _fastdllm_config(**spec, speculative_advance_reuse=reuse)
        return FastDLLMSampler(llm, config=cfg).sample(_prompt(), folded_model=mff)

    out_off = run_arm(False)
    out_on = run_arm(True)
    assert torch.equal(out_on.sequences, out_off.sequences)
    assert len(out_on.history) == len(out_off.history)
    for snap_on, snap_off in zip(out_on.history, out_off.history):
        assert torch.equal(snap_on, snap_off)


def test_ut3176_folding_path_record_face() -> None:
    """UT-3176: folding-path record face (normal tau) — honest empirical pin.

    Same on/off A/B as UT-3175 but with the fixture's normal gate
    (``tau=_TAU``, folding ACTIVE). ULP-level divergence between the
    on-arm stash read (tail of the last small block's batched winner row)
    and the off-arm serial advance forward is LEGAL in principle (folded
    batched vs serial recomputation cross different GEMM shapes); the
    empirical record face pins what this CPU fixture actually observes.
    On this fixture the on/off decode_final IS bitwise equal (verified
    empirically at write time — the identity of the committed canvas makes
    the argmax source agree), so the strict equality is asserted here as
    the recorded behavior, alongside the unconditional legality faces:
    the on-arm decode is a legal canvas (no mask_id, prompt prefix intact)
    and the decode lengths match.
    """
    spec: dict[str, Any] = {
        "num_tokens": 16,
        "block_size": _BLOCK_SIZE,
        "small_block_size": _SMALL_BLOCK_SIZE,
        "seed": 3176,
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "return_history": True,
    }

    def run_arm(reuse: bool) -> Any:
        inner = _build_block_causal_mini()
        llm = _MiniDiffusionLLM(inner)
        mff = _make_mff(inner)
        cfg = _fastdllm_config(**spec, speculative_advance_reuse=reuse)
        return FastDLLMSampler(llm, config=cfg).sample(_prompt(), folded_model=mff)

    out_off = run_arm(False)
    out_on = run_arm(True)

    # Unconditional legality faces: the on-arm decode is a legal canvas.
    row_on = out_on.sequences[0].tolist()
    assert _MASK_ID not in row_on
    assert row_on[:3] == [1, 2, 3]
    assert out_on.sequences.shape == out_off.sequences.shape

    # Empirical record face: on this fixture the on/off bitwise equality
    # holds (pinned as observed behavior; ULP divergence would be legal but
    # is not observed here).
    assert torch.equal(out_on.sequences, out_off.sequences)
    assert len(out_on.history) == len(out_off.history)
    for snap_on, snap_off in zip(out_on.history, out_off.history):
        assert torch.equal(snap_on, snap_off)
