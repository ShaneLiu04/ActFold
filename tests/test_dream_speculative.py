"""Red-phase tests for AR023: Dream-family speculative decoding (UT-2309..UT-2314).

Contract: ``specs/changes/AR023-cross-family-speculative/design.md`` §4.1.1,
§4.1.4, §4.2.2, §6.1 and ``srs.md`` FR3/FR4 — porting the shared
``SpeculativeCore`` mechanism onto ``DreamSampler`` through the left-padded
super-step family adapter:

* UT-2309: ``num_block_candidates=0`` default == explicit off bit-exact (raw
  and N=0 folded faces) and the symmetric entry validation (no folded_model /
  branch_ids unsupported / B>1+N>0 → ValueError);
* UT-2310: the super-step unit — the s-step budget merge (one draft + one
  ``[N, T]`` batched call per super-step vs s reference iterations), the
  confidence top-k position selection, the one-shot tail degeneration when s
  covers the schedule, ``speculative_steps < 1`` → ValueError, eos → finished
  skip, the no-masked-position skip, the ``_build_unit_forced`` commit-set
  content, and the shift-convention equivalence (Dream feeds Core
  ``right_shift_logits``-corrected logits — the spec commits match the
  reference path's shifted-argmax commits bit-for-bit);
* UT-2311: the ``DiffusionSampler._forward`` folded-branch ``position_ids``
  tail parameter (None attaches no kwarg — existing callers byte-for-byte
  unchanged; non-None attaches; the raw branch is untouched) and the Dream
  ``_spec_ctx_for`` object contract (bool ``[1, T]`` mask +
  ``compute_position_ids`` payload, ``(T, prompt_len)`` object pinning);
* UT-2312: the symmetric config validation (True+N=0 / N<0 / top_k<1 /
  speculative_steps<1 ValueErrors; symmetric field defaults) on both
  ``LLaDASamplerConfig`` and ``DreamSamplerConfig``;
* UT-2313: the ``ActFoldConfig`` YAML consumption face
  (``num_block_candidates`` / ``candidate_top_k`` / reused
  ``candidate_include_greedy`` consumed with zero unknown-key UserWarning)
  and the extended per_model YAML section (default-off);
* UT-2314: the ``SpeculativeCore`` defensive faces (empty ``masked_local``
  is a no-op; batch > 1 raises ValueError).

Red state (why every test fails today) — each failure is a production-missing
face, never a test-own bug:

* ``DreamSamplerConfig`` has no ``num_block_candidates`` /
  ``candidate_top_k`` / ``candidate_include_greedy`` / ``speculative_steps``
  fields — every explicit kwarg pass fails with ``TypeError`` (UT-2309,
  UT-2310 sampler-driven faces, UT-2312);
* ``actfold.models.speculative_core`` does not exist — the direct-Core tests
  fail with ``ModuleNotFoundError`` (UT-2314);
* ``DreamSampler`` has no ``_spec_ctx_for`` / ``_build_unit_forced`` — the
  direct adapter faces fail with ``AttributeError`` (UT-2310 forced,
  UT-2311 ctx);
* ``DiffusionSampler._forward``'s folded branch drops ``position_ids``
  (only the raw branch consumes it, ``diffusion_sampler.py:186-217``) — the
  non-None attach face fails on the recorded-kwargs assertion (UT-2311);
* ``ActFoldConfig`` has no ``num_block_candidates`` / ``candidate_top_k`` —
  the direct constructor raises ``TypeError`` and the YAML face trips the
  ``load_config`` unknown-key ``UserWarning`` (UT-2313);
* the per_model YAMLs carry no speculative section — the text assertion
  fails (UT-2313).

Fixtures follow ``tests/test_speculative_greedy_alignment.py`` (self-contained
LLaMA-layout mini decoder, ``_MiniDiffusionLLM``, ``_SpyMFF``, ``_make_mff``)
with the AR023 change: fully bidirectional attention accepting every mask
form the Dream family may pass (``[B, T]`` / ``[1, T]`` bool padding masks,
``[T, T]`` prebuilt masks) and a ``position_ids``-tolerant forward signature.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from actfold.core import ActivationCache, SimilarityGate
from actfold.models import dream_sampler as dream_sampler_mod
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.models.base import DiffusionLLM
from actfold.models.dream_sampler import DreamSampler, DreamSamplerConfig
from actfold.models.llada_sampler import LLaDASamplerConfig
from actfold.models.sampling_utils import compute_position_ids, right_shift_logits
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from actfold.utils.config_manager import ActFoldConfig, load_config

_VOCAB = 64
_HIDDEN = 32
_NUM_LAYERS = 2
_TAU = 0.95
_CACHE_BUDGET = 1024
_MASK_ID = 63
_EOS_ID = 62
_PAD_ID = 61
_MODEL_SEED = 2026


@pytest.fixture(autouse=True)
def _profiler_state() -> Iterator[None]:
    """Snapshot-reset the global stability profiler around every case."""
    GLOBAL_STABILITY_PROFILER.reset()
    yield
    GLOBAL_STABILITY_PROFILER.reset()


# ---------------------------------------------------------------------------
# Self-contained mini bidirectional decoder (T018 fixture family)
# ---------------------------------------------------------------------------


class _BidirectionalAttention(nn.Module):
    """Fully bidirectional attention accepting every production mask form.

    Padding form: ``[B, T]`` or ``[1, T]`` (bool or long) — broadcast over
    query positions. Prebuilt form: ``[T, T]`` bool — used as-is. ``None``
    attends everywhere.
    """

    def __init__(self, hidden: int, num_heads: int) -> None:
        super().__init__()
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
            if tuple(am.shape) == (seq_len, seq_len):
                attn_mask = am
            elif am.shape[0] in (1, batch):
                attn_mask = am[:, None, :]
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


class _BidirectionalLayer(nn.Module):
    """Pre-norm decoder layer with a detectable split FFN chain (LLaMA layout)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = _BidirectionalAttention(hidden, num_heads=4)
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


class _ConstantHead(nn.Module):
    """Head whose argmax is a fixed token at every position."""

    def __init__(self, vocab: int, token_id: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.token_id = token_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*x.shape[:-1], self.vocab, dtype=x.dtype, device=x.device)
        logits[..., self.token_id] = 1.0
        return logits


class _PosTableHead(nn.Module):
    """Position-table head: top-1 at position p is ``(p+1) % V`` (graded).

    ``logits[:, p, (p+1) % V] = 3.0 + 0.01*p`` and
    ``logits[:, p, (p+2) % V] = 2.0 + 0.01*p`` — input-independent with
    confidence increasing in p, so the maskgit_plus top-k position selection
    is deterministic (the highest masked positions first) and the unshifted
    argmax at p (``(p+1) % V``) is distinguishable from the right-shifted
    argmax (``p % V``) — the shift-convention discriminator, UT-2310.
    """

    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        rows = torch.arange(seq_len, device=x.device)
        positions = rows
        logits = torch.zeros(*x.shape[:-1], self.vocab, dtype=x.dtype, device=x.device)
        logits[:, rows, (positions + 1) % self.vocab] = 3.0 + 0.01 * positions
        logits[:, rows, (positions + 2) % self.vocab] = 2.0 + 0.01 * positions
        return logits


class _MiniBidirectionalModel(nn.Module):
    """Mock LLaMA-layout bidirectional decoder: embed -> layers -> norm -> head."""

    def __init__(self, head: str = "linear", vocab: int = _VOCAB) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "dream", "vocab_size": vocab})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, _HIDDEN)
        self.model.layers = nn.ModuleList([_BidirectionalLayer(_HIDDEN) for _ in range(2)])
        self.model.norm = nn.LayerNorm(_HIDDEN)
        if head == "constant_eos":
            self.lm_head = _ConstantHead(vocab, _EOS_ID)
        elif head == "postable":
            self.lm_head = _PosTableHead(vocab)
        else:
            self.lm_head = nn.Linear(_HIDDEN, vocab, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs, position_ids
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.lm_head(self.model.norm(x))


class _KwargsRecordingLayer(_BidirectionalLayer):
    """``_BidirectionalLayer`` that records every original-layer kwarg set.

    Used to pin the MFF eager ``position_ids`` attach face: the wrapped
    ORIGINAL layer must receive the ctx position payload on every folded
    forward (AR023 end-to-end threading). Delegates to the parent forward
    unchanged, so model behavior is byte-identical.
    """

    def __init__(self, source: _BidirectionalLayer) -> None:
        super().__init__(_HIDDEN)
        self.load_state_dict(source.state_dict())
        self.to(next(source.parameters()).device)
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        record = {"attention_mask": attention_mask, **kwargs}
        self.calls.append(record)
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )


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
    """ManualFoldedForward that records every forward call and its output.

    Tensor args are CLONED at record time: the Dream sampler (and the
    shared SpeculativeCore) mutate the canvas in place across super-steps,
    so by-reference recording would retroactively show post-commit state
    (the call-time snapshot is the spy's contract).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict[str, Any]] = []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        snapshot = tuple(a.detach().clone() if isinstance(a, torch.Tensor) else a for a in args)
        out = super().forward(*args, **kwargs)
        self.calls.append({"args": snapshot, "kwargs": kwargs, "out": out})
        return out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_mini(device: str, head: str = "linear") -> _MiniBidirectionalModel:
    """Deterministically build the mini bidirectional decoder on ``device``."""
    torch.manual_seed(_MODEL_SEED)
    model = _MiniBidirectionalModel(head=head)
    model.to(device)
    model.eval()
    return model


def _make_mff(
    model: nn.Module,
    device: str,
    cls: type[ManualFoldedForward] = _SpyMFF,
) -> ManualFoldedForward:
    """Fresh ManualFoldedForward (own cache + cosine gate) over the model."""
    cache = ActivationCache(max_entries_per_layer=_CACHE_BUDGET, device=device)
    gate = SimilarityGate(tau=_TAU, metric="cosine")
    return cls(
        model,
        cache=cache,
        gate=gate,
        scheduler=None,
        split_layers=False,
        split_min_tokens=512,
        use_cuda_graph=False,
        graph_capacity_ratio=0.5,
    )


def _dream_config(**overrides: Any) -> DreamSamplerConfig:
    """DreamSamplerConfig with pinned defaults (new AR023 fields via overrides)."""
    values: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 6,
        "temperature": 0.0,
        "seed": 2300,
    }
    values.update(overrides)
    return DreamSamplerConfig(**values)


def _prompt(device: str) -> torch.Tensor:
    return torch.tensor([[1, 2, 3]], device=device)


def _user_warning_messages(record: Any) -> list[str]:
    """Return the messages of every UserWarning in a catch_warnings log."""
    return [str(entry.message) for entry in record if issubclass(entry.category, UserWarning)]


def _split_calls(mff: _SpyMFF) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split spied MFF calls into (serial draft, batched verification)."""
    drafts = [c for c in mff.calls if c["kwargs"].get("branch_ids") is None]
    batched = [c for c in mff.calls if c["kwargs"].get("branch_ids") is not None]
    return drafts, batched


class _ForwardRecorder:
    """Stub ``forward_fn`` recording every call (UT-2314 defensive faces)."""

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab
        self.calls: list[dict[str, Any]] = []

    def __call__(self, tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append({"tokens": tokens.detach().clone(), "kwargs": dict(kwargs)})
        n, seq_len = tokens.shape
        logits = torch.zeros(n, seq_len, self.vocab, dtype=torch.float32)
        for b in range(n):
            logits[b, torch.arange(seq_len), tokens[b]] = 3.0
        return logits


class _CoreHostFactory:
    """Build a minimal ``SpeculativeCore`` host (design §4.1.2 host contract)."""

    def host(self, **cfg: Any) -> Any:
        from actfold.models.speculative_core import SpeculativeCore  # noqa: PLC0415

        values: dict[str, Any] = {
            "num_block_candidates": 2,
            "candidate_top_k": 2,
            "candidate_include_greedy": True,
            "seed": 2300,
            "return_history": True,
        }
        values.update(cfg)

        class _Host(SpeculativeCore):
            def __init__(self, config: Any) -> None:
                self.config = config
                self._greedy_wins = 0
                self._greedy_spec_calls = 0
                self._folding_parent: str | None = None

        return _Host(SimpleNamespace(**values))

    def ctx(self, seq_len: int) -> Any:
        from actfold.models.speculative_core import SpecCtx  # noqa: PLC0415

        return SpecCtx(
            mask=torch.ones(1, seq_len, dtype=torch.bool),
            position=torch.arange(seq_len).unsqueeze(0),
        )

    def canvas(self) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, _MASK_ID, _MASK_ID, _MASK_ID, _MASK_ID, 7]])

    def draft_region(self) -> torch.Tensor:
        draft = torch.zeros(1, 4, _VOCAB)
        for m in range(4):
            draft[0, m, (m * 7 + 5) % _VOCAB] = 3.0
            draft[0, m, (m * 7 + 6) % _VOCAB] = 1.0
        return draft


# ---------------------------------------------------------------------------
# UT-2309: Dream default zero disturbance + entry validation
# ---------------------------------------------------------------------------


def test_ut2309_default_zero_disturbance(device: str) -> None:
    """UT-2309: N=0 explicit off == default bit-exact on raw and folded faces.

    ``num_block_candidates=0`` (the default) must keep the reference Dream
    recipe byte-identical: default vs explicit off on the raw path and the
    N=0 folded path; the default ``greedy_win_stats`` probe must read
    ``{0, 0}``.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)

    out_default = DreamSampler(llm, config=_dream_config(seed=2309)).sample(prompt)
    cfg_off = _dream_config(seed=2309, num_block_candidates=0, candidate_include_greedy=False)
    out_off = DreamSampler(llm, config=cfg_off).sample(prompt)
    assert torch.equal(out_off.sequences, out_default.sequences)

    out_folded_default = DreamSampler(llm, config=_dream_config(seed=2309)).sample(
        prompt, folded_model=_make_mff(inner, device)
    )
    out_folded_off = DreamSampler(llm, config=cfg_off).sample(
        prompt, folded_model=_make_mff(inner, device)
    )
    assert torch.equal(out_folded_off.sequences, out_folded_default.sequences)

    sampler_off = DreamSampler(llm, config=cfg_off)
    sampler_off.sample(prompt)
    assert sampler_off.greedy_win_stats == {"greedy_wins": 0, "speculative_calls": 0}
    sampler_folded_off = DreamSampler(llm, config=cfg_off)
    sampler_folded_off.sample(prompt, folded_model=_make_mff(inner, device))
    assert sampler_folded_off.greedy_win_stats == {
        "greedy_wins": 0,
        "speculative_calls": 0,
    }


def test_ut2309_entry_validation(device: str) -> None:
    """UT-2309: N>0 entry preconditions raise ValueError (never silent).

    Without a ``folded_model``, with a non-``branch_ids``-capable folded
    model, and with batch > 1 — the Dream faces symmetric to UT-2307.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _dream_config(
        seed=2309,
        num_steps=8,
        num_tokens=8,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        speculative_steps=4,
    )

    with pytest.raises(ValueError, match=r"folded_model"):
        DreamSampler(llm, config=cfg).sample(prompt)

    class _NoBranchIdsCallable:
        def __call__(self, tokens: torch.Tensor, attention_mask: Any = None) -> torch.Tensor:
            return torch.zeros(tokens.shape[0], tokens.shape[1], _VOCAB)

    with pytest.raises(ValueError, match=r"branch_ids"):
        DreamSampler(llm, config=cfg).sample(prompt, folded_model=_NoBranchIdsCallable())

    prompt_b2 = torch.cat([prompt, prompt], dim=0)
    with pytest.raises(ValueError):
        DreamSampler(llm, config=cfg).sample(prompt_b2, folded_model=_make_mff(inner, device))


# ---------------------------------------------------------------------------
# UT-2310: Dream super-step unit
# ---------------------------------------------------------------------------


def test_ut2310_super_step_budget_merge(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2310: one draft + one batched call per super-step (s-step budget merge).

    With 8 masked positions, ``num_steps=8`` (linear schedule: one transfer
    per step) and ``speculative_steps=4``, the schedule merges into two
    super-steps of budget 4: exactly 2 draft + 2 batched folded forwards vs
    the reference's 8 serial iterations. The first super-step must fill the
    top-4 confidence positions (the graded head makes confidence increase
    with p) with the shifted-argmax tokens.
    """
    inner = _build_mini(device, head="postable")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)

    # Reference forward count: 8 serial iterations (one transfer per step).
    cfg_ref = _dream_config(seed=2310, num_steps=8, num_tokens=8)
    llm.raw_calls.clear()
    DreamSampler(llm, config=cfg_ref).sample(prompt)
    reference_forwards = len(llm.raw_calls)
    assert reference_forwards == 8, "reference recipe: 8 per-step iterations"

    # Pin the MFF eager position_ids attach: the wrapped ORIGINAL layer
    # receives the ctx position payload on every folded forward.
    rec_layer = _KwargsRecordingLayer(inner.model.layers[0])
    inner.model.layers[0] = rec_layer
    rec_layer.calls.clear()

    cfg = _dream_config(
        seed=2310,
        num_steps=8,
        num_tokens=8,
        num_block_candidates=1,
        candidate_top_k=2,
        candidate_include_greedy=True,
        speculative_steps=4,
    )
    mff = _make_mff(inner, device)
    sampler = DreamSampler(llm, config=cfg)
    out = sampler.sample(prompt, folded_model=mff)

    drafts, batched = _split_calls(mff)
    assert len(drafts) == 2, "one draft per super-step"
    assert len(batched) == 2, "one [N, T] batched verification per super-step"
    assert (
        len(mff.calls) == 4 < reference_forwards
    ), "each super-step consumes s schedule steps (step += s - 1)"

    # First super-step: fill set == the top-4 confidence masked positions.
    seq_len = drafts[0]["out"].shape[1]
    assert seq_len == 11, "prompt 3 + generation 8"
    x_at_draft = drafts[0]["args"][0]
    cands0 = batched[0]["args"][0]
    assert tuple(cands0.shape) == (1, seq_len), "N=1 pure-greedy single-row batch"
    changed0 = (cands0[0] != x_at_draft[0]).nonzero(as_tuple=False).reshape(-1)
    assert torch.equal(
        changed0, torch.tensor([7, 8, 9, 10], device=device)
    ), "the fill set is the top-4 confidence positions"

    # Fill values == the SHIFTED draft argmax (Dream convention), computed
    # from the spied draft logits.
    shifted = right_shift_logits(drafts[0]["out"])
    expected_fill = shifted[0, 7:11, :].argmax(dim=-1)
    assert torch.equal(cands0[0, 7:11], expected_fill)
    assert torch.equal(out.sequences[0, 7:11], expected_fill), "commit == greedy fill"

    # Second super-step commits the remaining masked positions {3..6}.
    cands1 = batched[1]["args"][0]
    changed1 = (cands1[0] != drafts[1]["args"][0][0]).nonzero(as_tuple=False).reshape(-1)
    assert torch.equal(changed1, torch.tensor([3, 4, 5, 6], device=device))
    assert bool((out.sequences[0, 3:11] != _MASK_ID).all()), "canvas fully denoised"

    assert sampler.greedy_win_stats == {"greedy_wins": 2, "speculative_calls": 2}

    # The family forward_fn closure binds the ctx end to end: the draft and
    # batched folded calls carry the family bool mask (object identity) and
    # the ctx position payload; the wrapped original layer receives
    # position_ids on every folded forward.
    family_mask = sampler._attention_mask
    assert drafts[0]["kwargs"]["attention_mask"] is family_mask
    assert batched[0]["kwargs"]["attention_mask"] is family_mask
    ctx_used = sampler._spec_ctx_for(out.sequences, prompt_len=3)
    assert drafts[0]["kwargs"]["position_ids"] is ctx_used.position
    assert batched[0]["kwargs"]["position_ids"] is ctx_used.position
    assert rec_layer.calls, "the recording layer saw the folded forwards"
    assert all("position_ids" in call for call in rec_layer.calls)
    expected_pos = compute_position_ids(family_mask)
    for call in rec_layer.calls:
        assert torch.equal(call["position_ids"], expected_pos)

    # Window-partition pin: a crafted non-uniform schedule [[1, 1, 1, 1, 4]]
    # (8 masked, s=2) merges into exactly three super-step windows
    # [0:2]/[2:4]/[4:6] with budgets 2/2/4 — each super-step consumes s
    # schedule steps (step += s). A step += 1 advance would re-slice
    # overlapping windows and need four super-steps for the same canvas.
    inner2 = _build_mini(device, head="postable")
    llm2 = _MiniDiffusionLLM(inner2)
    crafted = torch.tensor([[1, 1, 1, 1, 4]], dtype=torch.int64, device=device)
    monkeypatch.setattr(dream_sampler_mod, "get_num_transfer_tokens", lambda **_kw: crafted)
    cfg2 = _dream_config(
        seed=2310,
        num_steps=6,
        num_tokens=8,
        num_block_candidates=1,
        candidate_top_k=2,
        candidate_include_greedy=True,
        speculative_steps=2,
    )
    mff2 = _make_mff(inner2, device)
    out2 = DreamSampler(llm2, config=cfg2).sample(prompt, folded_model=mff2)
    drafts2, batched2 = _split_calls(mff2)
    assert len(drafts2) == 3, "s-step windows partition the schedule (2/2/4)"
    assert len(batched2) == 3, "one batched verification per merged window"
    assert len(mff2.calls) == 6, "3 super-steps: 3 drafts + 3 batched forwards"
    assert bool((out2.sequences[0] != _MASK_ID).all()), "canvas fully denoised"


def test_ut2310_one_shot_tail(device: str) -> None:
    """UT-2310: s covering the whole schedule degenerates to a one-shot tail.

    ``speculative_steps=16 >= num_steps=8`` merges the entire schedule into
    ONE super-step whose budget (8) equals the masked count: exactly one
    draft + one batched call and every masked position committed.
    """
    inner = _build_mini(device, head="postable")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _dream_config(
        seed=2310,
        num_steps=8,
        num_tokens=8,
        num_block_candidates=1,
        candidate_top_k=2,
        candidate_include_greedy=True,
        speculative_steps=16,
    )
    mff = _make_mff(inner, device)
    out = DreamSampler(llm, config=cfg).sample(prompt, folded_model=mff)

    drafts, batched = _split_calls(mff)
    assert len(drafts) == 1 and len(batched) == 1, "one-shot tail: 1 draft + 1 batched"
    assert bool((out.sequences[0, 3:11] != _MASK_ID).all()), "k_eff_budget commits all 8"
    shifted = right_shift_logits(drafts[0]["out"])
    expected_fill = shifted[0, 3:11, :].argmax(dim=-1)
    assert torch.equal(out.sequences[0, 3:11], expected_fill)


def test_ut2310_shift_convention_equivalence(device: str) -> None:
    """UT-2310: Dream feeds Core right-shift-corrected logits (AR alignment).

    The spec path's committed tokens must equal the reference path's
    shifted-argmax commits position-by-position (and the full canvas
    bit-exactly — the input-independent graded head makes commit order
    irrelevant), while differing from the UNshifted argmax — the
    ``right_shift_logits`` convention discriminator (design §4.1.4).
    """
    inner = _build_mini(device, head="postable")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)

    out_ref = DreamSampler(llm, config=_dream_config(seed=2310, num_steps=8, num_tokens=8)).sample(
        prompt
    )

    cfg = _dream_config(
        seed=2310,
        num_steps=8,
        num_tokens=8,
        num_block_candidates=1,
        candidate_top_k=2,
        candidate_include_greedy=True,
        speculative_steps=4,
    )
    mff = _make_mff(inner, device)
    out_spec = DreamSampler(llm, config=cfg).sample(prompt, folded_model=mff)

    # Strong equivalence: the whole canvas is bit-exact with the reference.
    assert torch.equal(
        out_spec.sequences, out_ref.sequences
    ), "the shifted-argmax spec commits must reproduce the reference canvas"

    drafts, batched = _split_calls(mff)
    assert len(drafts) >= 1 and len(batched) >= 1
    raw_draft = drafts[0]["out"]
    shifted = right_shift_logits(raw_draft)
    cands0 = batched[0]["args"][0]
    fill_positions = (cands0[0] != drafts[0]["args"][0][0]).nonzero(as_tuple=False).reshape(-1)
    shifted_fill = shifted[0, fill_positions, :].argmax(dim=-1)
    unshifted_fill = raw_draft[0, fill_positions, :].argmax(dim=-1)
    assert torch.equal(
        cands0[0, fill_positions], shifted_fill
    ), "candidate fills follow the shifted argmax (reference-path convention)"
    assert not torch.equal(
        cands0[0, fill_positions], unshifted_fill
    ), "the fills must NOT follow the unshifted argmax (LLaDA convention)"
    assert torch.equal(out_spec.sequences[0, fill_positions], shifted_fill)


def test_ut2310_speculative_steps_validation() -> None:
    """UT-2310: ``speculative_steps < 1`` raises ValueError; 1 is legal.

    The super-step budget knob is Dream-only and validates at construction
    (no silent zero/negative step skipping).
    """
    with pytest.raises(ValueError, match=r"speculative_steps"):
        _dream_config(speculative_steps=0, num_block_candidates=2)
    with pytest.raises(ValueError, match=r"speculative_steps"):
        _dream_config(speculative_steps=-1, num_block_candidates=2)
    cfg_legal = _dream_config(speculative_steps=1, num_block_candidates=2)
    assert cfg_legal.speculative_steps == 1


def test_ut2310_eos_finished_skip(device: str) -> None:
    """UT-2310: eos in the commit set finishes and skips later super-steps.

    A constant-eos head commits eos for the first super-step's budget (4
    positions); ``finished`` must skip the remaining super-step — exactly 2
    folded calls (1 draft + 1 batched), 4 committed eos and 4 surviving
    masks.
    """
    inner = _build_mini(device, head="constant_eos")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _dream_config(
        seed=2310,
        num_steps=8,
        num_tokens=8,
        num_block_candidates=2,
        candidate_top_k=1,
        candidate_include_greedy=True,
        speculative_steps=4,
    )
    mff = _make_mff(inner, device)
    out = DreamSampler(llm, config=cfg).sample(prompt, folded_model=mff)

    drafts, batched = _split_calls(mff)
    assert len(drafts) == 1 and len(batched) == 1, "finished skips the later super-steps"
    assert len(mff.calls) == 2
    assert int((out.sequences[0] == _EOS_ID).sum()) == 4, "budget-sized eos commit"
    assert int((out.sequences[0] == _MASK_ID).sum()) == 4, "remaining masks stay masked"


def test_ut2310_no_masked_unit_skip(device: str) -> None:
    """UT-2310: a unit without masked positions is skipped without a forward.

    A zero-token Dream canvas has no masked position: the speculative path
    must make no forward at all (design §4.2.1 "单元内仍有掩码位?").
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _dream_config(
        seed=2310,
        num_steps=2,
        num_tokens=0,
        num_block_candidates=2,
        candidate_top_k=1,
        candidate_include_greedy=True,
        speculative_steps=4,
    )
    mff = _make_mff(inner, device)
    DreamSampler(llm, config=cfg).sample(prompt, folded_model=mff)
    assert len(mff.calls) == 0, "no masked positions: no forward at all"


def test_ut2310_forced_commit_set(device: str) -> None:
    """UT-2310: ``_build_unit_forced`` returns the exact commit-position set.

    The Dream forced tensor is the selected commit set (True at
    ``masked_local + region_start`` and False elsewhere) — the precise
    counterpart of LLaDA's block superset (design §4.1.4).
    """
    llm = _MiniDiffusionLLM(_build_mini(device))
    sampler = DreamSampler(llm, config=_dream_config())
    x = torch.zeros(1, 11, dtype=torch.long, device=device)
    forced = sampler._build_unit_forced(
        x, region_start=0, region_end=11, masked_local=torch.tensor([4, 5, 6, 7])
    )
    assert forced.dtype == torch.bool and tuple(forced.shape) == (11,)
    assert bool(forced[4:8].all()), "forced covers exactly the selected commit set"
    assert int(forced[:4].sum()) == 0 and int(forced[8:].sum()) == 0


# ---------------------------------------------------------------------------
# UT-2311: position_ids folding transparency + Dream ctx contract
# ---------------------------------------------------------------------------


class _RecordingFolded:
    """Callable folded-model stub recording every kwarg of ``_forward``."""

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab
        self.calls: list[dict[str, Any]] = []

    def __call__(self, tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append({"tokens": tokens.detach().clone(), "kwargs": dict(kwargs)})
        return torch.zeros(tokens.shape[0], tokens.shape[1], self.vocab)


def test_ut2311_forward_position_ids_none_attach(device: str) -> None:
    """UT-2311: the folded branch attaches ``position_ids`` only when non-None.

    Raw branch (unchanged): ``position_ids`` flows to ``model.forward`` when
    non-None and attaches no kwarg when None. Folded branch: None attaches
    no kwarg (existing callers byte-for-byte unchanged, the AR019
    ``position_embeddings`` precedent); non-None MUST attach it — today the
    folded branch drops the parameter (``diffusion_sampler.py:186-217``),
    which is the red face of this test.
    """
    llm = _MiniDiffusionLLM(_build_mini(device))
    sampler = DreamSampler(llm, config=_dream_config())
    x = torch.tensor([[1, 2, 3, _MASK_ID]], device=device)
    mask = torch.ones(1, 4, dtype=torch.bool, device=device)
    pos = torch.tensor([[0, 1, 2, 3]], device=device)

    # Raw branch, non-None: position_ids flows through (existing behavior).
    sampler._forward(
        x,
        branch_id="r1",
        parent_branch_id=None,
        folded_model=None,
        step_idx=0,
        attention_mask=mask,
        position_ids=pos,
    )
    assert llm.raw_calls[-1]["kwargs"].get("position_ids") is pos
    # Raw branch, None: no kwarg attached (existing behavior).
    sampler._forward(
        x,
        branch_id="r2",
        parent_branch_id=None,
        folded_model=None,
        step_idx=0,
        attention_mask=mask,
        position_ids=None,
    )
    assert "position_ids" not in llm.raw_calls[-1]["kwargs"]

    recorder = _RecordingFolded(_VOCAB)
    # Folded branch, None: no kwarg attached (zero perturbation face).
    sampler._forward(
        x,
        branch_id="f1",
        parent_branch_id=None,
        folded_model=recorder,
        step_idx=0,
        attention_mask=mask,
        position_ids=None,
    )
    assert "position_ids" not in recorder.calls[0]["kwargs"]
    # Folded branch, non-None: the kwarg MUST be attached (AR023 extension).
    sampler._forward(
        x,
        branch_id="f2",
        parent_branch_id=None,
        folded_model=recorder,
        step_idx=0,
        attention_mask=mask,
        position_ids=pos,
    )
    assert (
        recorder.calls[1]["kwargs"].get("position_ids") is pos
    ), "the folded branch must forward position_ids to the folded model"


def test_ut2311_dream_ctx_contract(device: str) -> None:
    """UT-2311: Dream ``_spec_ctx_for`` bool mask + position_ids, object-pinned.

    The ctx mask is the family's bool ``[1, T]`` attention mask (the
    ``initialize`` mask form), the position payload is
    ``compute_position_ids(mask)``, and the context is object-pinned per
    ``(T, prompt_len)`` key (the UT-2308 LLaDA-face counterpart).
    """
    llm = _MiniDiffusionLLM(_build_mini(device))
    sampler = DreamSampler(llm, config=_dream_config(num_tokens=8))
    prompt = _prompt(device)
    x = sampler.initialize(prompt)
    family_mask = sampler._attention_mask
    assert family_mask.dtype == torch.bool and tuple(family_mask.shape) == (1, 11)

    ctx_a = sampler._spec_ctx_for(x, prompt_len=3)
    ctx_b = sampler._spec_ctx_for(x, prompt_len=3)
    assert ctx_a is ctx_b, "same (T, prompt_len) key -> the same pinned ctx object"

    assert tuple(ctx_a.mask.shape) == (1, 11)
    assert ctx_a.mask.dtype == torch.bool
    assert torch.equal(ctx_a.mask, family_mask)
    expected_pos = compute_position_ids(family_mask)
    assert torch.equal(
        ctx_a.position, expected_pos
    ), "the Dream position payload is compute_position_ids(mask)"

    # A different canvas length is a different key with a different object.
    sampler_other = DreamSampler(llm, config=_dream_config(num_tokens=6))
    x_other = sampler_other.initialize(prompt)
    ctx_other = sampler_other._spec_ctx_for(x_other, prompt_len=3)
    assert ctx_other is not ctx_a
    assert ctx_other.mask.shape[1] == 9

    # Same T with a different prompt_len is a different key (the pad-aware
    # pinning — a pure shape key would collide the two contexts).
    prompt_p4 = torch.tensor([[1, 2, 3, 4]], device=device)
    sampler_p4 = DreamSampler(llm, config=_dream_config(num_tokens=7))
    x_p4 = sampler_p4.initialize(prompt_p4)
    assert int(x_p4.shape[1]) == 11, "same canvas length, different prompt_len"
    ctx_p4 = sampler_p4._spec_ctx_for(x_p4, prompt_len=4)
    assert ctx_p4 is not ctx_a, "the ctx key is (T, prompt_len), not T alone"


def test_ut2311_folded_reference_position_ids_threading(device: str) -> None:
    """UT-2311: the N=0 folded reference path threads ``position_ids``.

    The layer-driven folded path bypasses the raw model's internal
    position-id computation (remote Dream rotary reads ``position_ids``
    directly), so the reference loop must carry the family position payload
    on every folded serial call even for a batch-1 no-pad canvas — where
    the payload is the plain ``arange``, value-identical to what the raw
    model computes internally, so the raw reference recipe stays
    byte-identical (``folded_model=None`` attaches nothing).
    """
    inner = _build_mini(device, head="postable")
    llm = _MiniDiffusionLLM(inner)
    mff = _make_mff(inner, device)
    sampler = DreamSampler(llm, config=_dream_config(num_tokens=8))
    prompt = _prompt(device)

    out = sampler.sample(prompt, folded_model=mff)
    family_mask = sampler._attention_mask
    expected_pos = compute_position_ids(family_mask)
    serial_calls = [c for c in mff.calls if c["kwargs"].get("branch_ids") is None]
    assert serial_calls, "the reference loop makes folded serial calls"
    for call in serial_calls:
        assert (
            call["kwargs"].get("position_ids") is not None
        ), "the folded reference path threads the position payload"
        assert torch.equal(call["kwargs"]["position_ids"], expected_pos)
    assert bool((out.sequences[0] != _MASK_ID).all())


# ---------------------------------------------------------------------------
# UT-2312: symmetric configuration validation
# ---------------------------------------------------------------------------


def test_ut2312_config_validation_symmetric() -> None:
    """UT-2312: construction-time ValueErrors and symmetric defaults.

    ``candidate_include_greedy=True`` + ``num_block_candidates == 0`` raises
    the frozen AR022 message on BOTH family configs; ``N < 0`` and
    ``candidate_top_k < 1`` raise on both; Dream-only
    ``speculative_steps < 1`` raises; the field defaults are symmetric
    (0 / 10 / False, Dream adds ``speculative_steps=8``); legal combinations
    construct and survive ``dataclasses.replace``.
    """
    for config_cls in (LLaDASamplerConfig, DreamSamplerConfig):
        with pytest.raises(
            ValueError,
            match=r"candidate_include_greedy=True requires num_block_candidates > 0",
        ):
            config_cls(num_block_candidates=0, candidate_include_greedy=True)
        with pytest.raises(ValueError, match=r"num_block_candidates must be >= 0"):
            config_cls(num_block_candidates=-1)
        with pytest.raises(ValueError, match=r"candidate_top_k must be >= 1"):
            config_cls(candidate_top_k=0)
        cfg_on = config_cls(num_block_candidates=2, candidate_include_greedy=True)
        assert cfg_on.num_block_candidates == 2
        cfg_replaced = dataclasses.replace(cfg_on, seed=4321)
        assert cfg_replaced.candidate_include_greedy is True
        assert cfg_replaced.num_block_candidates == 2
        assert cfg_replaced.seed == 4321

    with pytest.raises(ValueError, match=r"speculative_steps"):
        DreamSamplerConfig(speculative_steps=0, num_block_candidates=2)

    # Symmetric defaults.
    llada_defaults = LLaDASamplerConfig()
    dream_defaults = DreamSamplerConfig()
    for cfg in (llada_defaults, dream_defaults):
        assert cfg.num_block_candidates == 0
        assert cfg.candidate_top_k == 10
        assert cfg.candidate_include_greedy is False
    assert dream_defaults.speculative_steps == 8, "Dream-only super-step default"


# ---------------------------------------------------------------------------
# UT-2313: YAML consumption faces
# ---------------------------------------------------------------------------


def test_ut2313_yaml_consumption(tmp_path: Any) -> None:
    """UT-2313: ``load_config`` consumes the new keys with zero UserWarning.

    ``ActFoldConfig`` gains ``num_block_candidates`` / ``candidate_top_k``
    (YAML consumption face, the AR022 ``candidate_include_greedy`` precedent)
    so the keys load without the unknown-key ``UserWarning`` (AGENTS #5);
    absent keys keep the defaults 0 / 10 / False.
    """
    cfg_direct = ActFoldConfig(num_block_candidates=2, candidate_top_k=4)
    assert cfg_direct.num_block_candidates == 2
    assert cfg_direct.candidate_top_k == 4

    cfg_file = tmp_path / "ar023_ut2313.yaml"
    cfg_file.write_text(
        "num_block_candidates: 2\n" "candidate_top_k: 4\n" "candidate_include_greedy: true\n",
        encoding="utf-8",
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        loaded = load_config(cfg_file)
    assert _user_warning_messages(record) == [], "keys must be consumed, not warned off"
    assert loaded.num_block_candidates == 2
    assert loaded.candidate_top_k == 4
    assert loaded.candidate_include_greedy is True

    cfg_file.write_text("# empty config\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as record_default:
        warnings.simplefilter("always")
        loaded_default = load_config(cfg_file)
    assert _user_warning_messages(record_default) == []
    assert loaded_default.num_block_candidates == 0
    assert loaded_default.candidate_top_k == 10
    assert loaded_default.candidate_include_greedy is False


def test_ut2313_per_model_yaml_section() -> None:
    """UT-2313: the per_model YAMLs carry the (default-off) speculative section.

    ``llada_8b.yaml`` and ``dream_7b.yaml`` must mention the speculative keys
    (commented or active — the design freezes a default-off section) and must
    load with zero unknown-key UserWarning with the fields at their defaults.
    """
    per_model_dir = Path(__file__).resolve().parents[1] / "actfold" / "configs" / "per_model"
    for name in ("llada_8b.yaml", "dream_7b.yaml"):
        path = per_model_dir / name
        text = path.read_text(encoding="utf-8")
        assert "num_block_candidates" in text, f"{name} must carry the speculative section"
        assert "candidate_top_k" in text
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            loaded = load_config(path)
        assert _user_warning_messages(record) == []
        assert loaded.num_block_candidates == 0, "the shipped section defaults to off"


# ---------------------------------------------------------------------------
# UT-2314: SpeculativeCore defensive faces
# ---------------------------------------------------------------------------


def test_ut2314_empty_masked_local_noop() -> None:
    """UT-2314: an empty ``masked_local`` is a silent no-op.

    No batched forward, no canvas change, no finished change, no history
    append, no chain mutation — the defensive sentinel return (design §4.3:
    the family adapter should not call into the core for an empty unit, and
    the core must not explode if it does).
    """
    factory = _CoreHostFactory()
    host = factory.host()
    x = factory.canvas()
    x_before = x.clone()
    finished = torch.zeros(1, dtype=torch.bool)
    history: list[torch.Tensor] = [x.clone()]
    recorder = _ForwardRecorder(_VOCAB)

    host._speculative_commit(
        x=x,
        finished=finished,
        draft_logits_region=factory.draft_region(),
        region_start=3,
        masked_local=torch.zeros(0, dtype=torch.long),
        draft_branch_id="spec_u0",
        ctx=factory.ctx(8),
        unit_forced=torch.zeros(8, dtype=torch.bool),
        folded_model=object(),
        unit_idx=0,
        sub_idx=0,
        history=history,
        forward_fn=recorder,
        stop_token_id=_EOS_ID,
    )

    assert recorder.calls == [], "empty fill set: no batched forward"
    assert torch.equal(x, x_before), "canvas untouched"
    assert not bool(finished[0]), "finished untouched"
    assert len(history) == 1, "history untouched"
    assert host._folding_parent is None, "chain untouched"


def test_ut2314_batch_gt1_valueerror() -> None:
    """UT-2314: a batch > 1 canvas raises ValueError (the #41 B==1 limit).

    The ``[T, T]`` / family-mask semantics cannot express per-row padding,
    so the core must reject a multi-row canvas up front (AR021-isomorphic).
    """
    factory = _CoreHostFactory()
    host = factory.host()
    x2 = factory.canvas().repeat(2, 1)
    with pytest.raises(ValueError):
        host._speculative_commit(
            x=x2,
            finished=torch.zeros(2, dtype=torch.bool),
            draft_logits_region=factory.draft_region(),
            region_start=3,
            masked_local=torch.tensor([0, 1, 2, 3]),
            draft_branch_id="spec_u0",
            ctx=factory.ctx(8),
            unit_forced=torch.zeros(8, dtype=torch.bool),
            folded_model=object(),
            unit_idx=0,
            sub_idx=0,
            history=[],
            forward_fn=_ForwardRecorder(_VOCAB),
            stop_token_id=_EOS_ID,
        )
