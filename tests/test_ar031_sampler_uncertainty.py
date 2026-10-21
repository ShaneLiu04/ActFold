"""Red-phase tests for AR031: sampler uncertainty-aware validation (UT-3134..UT-3144).

Contract: ``specs/changes/AR031-adaptive-budget-uncertainty/design.md`` 4.2.2 (A6),
4.3, 6.1.4 and ``srs.md`` FR4 — the opt-in uncertainty-aware candidate
validation order on the three diffusion samplers (LLaDA / Dream / Fast-dLLM):

* the ``[N, T]`` batched verification forward, the segment-A candidate
  construction (frozen seed formula, greedy anchor), and the min-prob winner
  rule stay untouched; the opt-in branch only reorders the ROW SCORING inside
  segment B (``_speculative_score_commit`` / the fast_dllm inline twin) and
  may early-stop the scoring loop after a validated row clears
  ``q_best >= margin and (q_best - remaining_max_prior) >= margin_gap``;
* the prior is the frozen draft-side formula: the minimum over the fill
  positions of ``softmax(draft_logits_region)[pos, cand_token]``;
* per unit: ``validated + skipped == N``; skipped rows never touch the
  commit / ``finished`` / history / ``_folding_parent``.

Frozen interface (design 4.3 / 4.2.2 A6) for the T009 implementation:

* config fields on ALL THREE family configs (``FastDLLMSamplerConfig`` /
  ``LLaDASamplerConfig`` / ``DreamSamplerConfig``):
  ``uncertainty_validation_order: bool = False`` /
  ``uncertainty_margin: float = 0.9`` / ``uncertainty_margin_gap: float =
  0.05``; ``__post_init__``: order=True with ``num_block_candidates == 0``
  -> ``ValueError``; margin / margin_gap outside ``[0, 1]`` -> ``ValueError``;
* module-level pure functions in ``actfold.models.speculative_core``:
  ``plan_uncertainty_order(prior: Tensor) -> list[int]`` (descending, tie ->
  smaller index; ``ValueError`` on empty / non-1-D input) and
  ``should_early_stop(q_best: float, remaining_max_prior: float, margin:
  float, margin_gap: float) -> bool`` — consumed by BOTH the core segment B
  and the fast_dllm inline commit (UT-3141 pins the parity);
* read-only instance probe ``uncertainty_stats`` ->
  ``{"validated_rows", "skipped_rows", "early_stops"}`` (counters
  ``_unc_validated_rows`` / ``_unc_skipped_rows`` / ``_unc_early_stops``
  initialized in the HOST samplers' ``__init__`` — the mixin defines no
  ``__init__``; the ``greedy_win_stats`` precedent); default-off stays
  ``{0, 0, 0}`` forever.

Red state (why every test fails today) — production-missing faces only:

* the three family configs have no ``uncertainty_validation_order`` /
  ``uncertainty_margin`` / ``uncertainty_margin_gap`` fields — every explicit
  kwarg pass fails with ``TypeError`` (UT-3134..3140, UT-3142, UT-3143);
* ``actfold.models.speculative_core`` exports neither
  ``plan_uncertainty_order`` nor ``should_early_stop`` — the function-local
  import in UT-3141 fails with ``ImportError``;
* no host sampler defines ``uncertainty_stats`` — the probe read in UT-3144
  fails with ``AttributeError``.

Fixtures follow ``tests/test_llada_speculative.py`` (self-contained mini
bidirectional decoder, ``_MiniDiffusionLLM``, ``_SpyMFF``, ``_make_mff``,
Hadamard-basis posterior trap, frozen-formula seed search),
``tests/test_dream_speculative.py`` (Dream config / spec-path faces), and
``tests/test_speculative_greedy_alignment.py`` (block-causal mini decoder,
``_ShiftedTableHead`` right-shift cancellation, FastDLLM small-block trap
geometry) with the AR031 change: the posterior table is the UNCERTAINTY
variant ``T[mask]={A:3, C:1}, T[A]={A:4}, T[C]={C:3}`` whose greedy row BOTH
tops the prior order AND clears the early-stop margin. All bit-exact
comparisons run on CPU (design A6: per-row vs batch softmax parity is
bit-exact on CPU only — no ``device`` fixture usage).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from actfold.core import ActivationCache, SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.models.base import DiffusionLLM
from actfold.models.dream_sampler import DreamSampler, DreamSamplerConfig
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig
from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig
from actfold.models.sampling_utils import right_shift_logits
from actfold.models.speculative_core import SpecCtx
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

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

# Uncertainty-trap constants: trap vocab 16 over hidden 32 so 16
# pairwise-orthogonal zero-mean basis vectors exist (Hadamard rows 1..16).
_TRAP_VOCAB = 16
_TRAP_MASK_ID = 15
_TRAP_EOS_ID = 14
_TRAP_PAD_ID = 13
_TRAP_GREEDY_ID = 5  # "A": draft argmax; T[A]={A:4} -> high posterior self-prob
_TRAP_ALT_ID = 6  # "C": draft runner-up; T[C]={C:3} -> mid posterior self-prob
_TRAP_BLOCK_SIZE = 8
_TRAP_NUM_SMALL_BLOCKS = 8  # FastDLLM face: small_block_size=1, one canvas block
_TRAP_BLOCK_START = 8  # canvas position of the block region (prompt 3 + pad 5)
_TRAP_PROMPT_LEN = 3
_TRAP_NUM_TOKENS = 2  # LLaDA face: single block of 2 masked positions (M=2)

# Joint-face constants (UT-3142): the first-token broadcast head keys each
# chain's table row by its first prompt token.
_JOINT_KEY0 = 7
_JOINT_KEY1 = 8
_JOINT_X0 = 3
_JOINT_X1 = 4
_JOINT_Y0 = 9
_JOINT_Y1 = 10


@pytest.fixture(autouse=True)
def _profiler_state() -> Iterator[None]:
    """Snapshot-reset the global stability profiler around every case."""
    GLOBAL_STABILITY_PROFILER.reset()
    yield
    GLOBAL_STABILITY_PROFILER.reset()


# ---------------------------------------------------------------------------
# Self-contained mini bidirectional decoder (LLaDA / Dream fixture family)
# ---------------------------------------------------------------------------


class _BidirectionalAttention(nn.Module):
    """Fully bidirectional attention accepting every production mask form.

    Padding form: ``[B, T]`` or ``[1, T]`` (long or bool) — broadcast over
    query positions. Prebuilt form: ``[T, T]`` bool — used as-is (the LLaDA
    spec-path fold-context mask). ``None`` attends everywhere.
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


class _SpikeHead(nn.Module):
    """Head with two clearly ranked spikes: top-1 and top-2 known ids."""

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


class _MiniBidirectionalModel(nn.Module):
    """Mock LLaMA-layout bidirectional decoder: embed -> layers -> norm -> head."""

    def __init__(self, head: str = "linear", vocab: int = _VOCAB) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llada", "vocab_size": vocab})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, _HIDDEN)
        self.model.layers = nn.ModuleList([_BidirectionalLayer(_HIDDEN) for _ in range(2)])
        self.model.norm = nn.LayerNorm(_HIDDEN)
        if head == "spike":
            self.lm_head = _SpikeHead(vocab, 5, 6)
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


# ---------------------------------------------------------------------------
# Self-contained mini block-causal decoder (FastDLLM fixture family)
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

    def __init__(self, head: str = "linear") -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": _VOCAB})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.model.layers = nn.ModuleList(
            [_DualFormLayer(_HIDDEN, _BLOCK_SIZE) for _ in range(_NUM_LAYERS)]
        )
        self.model.norm = nn.LayerNorm(_HIDDEN)
        if head == "spike":
            self.lm_head = _SpikeHead(_VOCAB, 5, 6)
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


# ---------------------------------------------------------------------------
# Shared wrappers and helpers
# ---------------------------------------------------------------------------


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


def _build_mini(head: str = "linear") -> _MiniBidirectionalModel:
    """Deterministically build the mini bidirectional decoder (CPU)."""
    torch.manual_seed(_MODEL_SEED)
    model = _MiniBidirectionalModel(head=head)
    model.eval()
    return model


def _build_block_causal_mini(head: str = "linear") -> _MiniBlockCausalModel:
    """Deterministically build the mini block-causal decoder (CPU)."""
    torch.manual_seed(_MODEL_SEED)
    model = _MiniBlockCausalModel(head=head)
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


def _llada_config(**overrides: Any) -> LLaDASamplerConfig:
    """LLaDASamplerConfig with pinned defaults (AR031 fields via overrides)."""
    values: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 8,
        "temperature": 0.0,
        "seed": 2300,
        "block_size": _BLOCK_SIZE,
    }
    values.update(overrides)
    return LLaDASamplerConfig(**values)


def _dream_config(**overrides: Any) -> DreamSamplerConfig:
    """DreamSamplerConfig with pinned defaults (test_dream_speculative face)."""
    values: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 6,
        "temperature": 0.0,
        "seed": 2300,
    }
    values.update(overrides)
    return DreamSamplerConfig(**values)


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


# ---------------------------------------------------------------------------
# Uncertainty posterior-trap fixtures (Hadamard basis + table heads)
# ---------------------------------------------------------------------------


def _hadamard_basis(vocab: int, hidden: int) -> torch.Tensor:
    """Zero-mean unit-variance pairwise-orthogonal rows (Sylvester Hadamard).

    Row v uses Hadamard row v+1 (row 0 is all-ones and not zero-mean); every
    row index >= 1 is balanced (+1/-1), so each basis vector has zero mean
    and unit variance, and distinct rows are exactly orthogonal (integer dot
    products, exact in float32). ``LayerNorm(3.0 * t_v + 0.5) == t_v`` for
    the gamma=1 / beta=0 norm that precedes the head.
    """
    rows = []
    for v in range(vocab):
        i = v + 1
        rows.append([-1.0 if bin(i & h).count("1") % 2 else 1.0 for h in range(hidden)])
    return torch.tensor(rows, dtype=torch.float32)


def _uncertainty_trap_table() -> torch.Tensor:
    """AR031 posterior table ``T`` (prior-split early-stop construction).

    ``T[mask] = {A: 3.0, C: 1.0}`` — the draft top-2 at a still-masked
    position is {A, C} with argmax A, so the greedy row fills A and the
    pinned random row fills C. ``T[A] = {A: 4.0}`` — the greedy row's
    posterior min-prob is ``e^4 / (e^4 + 15) = 0.7845``, high enough to
    clear the margin while its prior (``0.5458``) tops the random row's
    (``0.0739``): the early stop fires after row 0 and the commit equals
    the full-validation argmax. ``T[C] = {C: 3.0}`` — the random row's
    posterior (``0.5724``) loses to the greedy row either way.
    """
    table = torch.zeros(_TRAP_VOCAB, _TRAP_VOCAB)
    table[_TRAP_MASK_ID, _TRAP_GREEDY_ID] = 3.0
    table[_TRAP_MASK_ID, _TRAP_ALT_ID] = 1.0
    table[_TRAP_GREEDY_ID, _TRAP_GREEDY_ID] = 4.0
    table[_TRAP_ALT_ID, _TRAP_ALT_ID] = 3.0
    return table


def _joint_trap_table() -> torch.Tensor:
    """Per-chain table rows keyed by the chains' first prompt tokens (UT-3142).

    Chain 0 (key 7): ``softmax(T[7])[X0] = e^8 / (e^8 + e^0.5 + 14) =
    0.9948`` — the greedy row clears margin 0.9 after ONE validated row.
    Chain 1 (key 8): ``softmax(T[8])[Y0] = e^2 / (e^2 + e^1 + 14) = 0.3065``
    — below any legal margin, so the chain fully validates.
    """
    table = torch.zeros(_TRAP_VOCAB, _TRAP_VOCAB)
    table[_JOINT_KEY0, _JOINT_X0] = 8.0
    table[_JOINT_KEY0, _JOINT_X1] = 0.5
    table[_JOINT_KEY1, _JOINT_Y0] = 2.0
    table[_JOINT_KEY1, _JOINT_Y1] = 1.0
    return table


class _TableHead(nn.Module):
    """Direct table head: ``logits[:, p] = T[input[:, p]]`` (NO shift).

    ``W = sum_u outer(T[u], t_u) / H`` over the orthogonal basis makes
    ``W @ t_in == T[in]`` exactly (up to the LayerNorm eps scale), so the
    commit posterior of a filled token is ``softmax(T[fill])[fill]`` — fully
    controlled by the candidate content (LLaDA feeds the core the UNSHIFTED
    logits: position i predicts position i).
    """

    def __init__(self, table: torch.Tensor, basis: torch.Tensor) -> None:
        super().__init__()
        vocab, hidden = table.shape[0], basis.shape[1]
        weight = torch.zeros(vocab, hidden)
        for u in range(vocab):
            weight += torch.outer(table[u], basis[u]) / hidden
        self.proj = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _ShiftedTableHead(nn.Module):
    """Right-shift table head: ``raw[:, q] = T_table[input[q+1]]`` (wrap at end).

    The same orthogonal-basis weight construction as ``_TableHead`` but the
    projection consumes the NEXT position's hidden state, so the head's raw
    logits cancel the sampler's ``right_shift_logits``: ``shifted[:, p] =
    raw[:, p-1] = T_table[input[p]]`` — the commit posterior of a filled
    token is ``softmax(T_table[fill])[fill]`` (FastDLLM convention).
    """

    def __init__(self, table: torch.Tensor, basis: torch.Tensor) -> None:
        super().__init__()
        vocab, hidden = table.shape[0], basis.shape[1]
        weight = torch.zeros(vocab, hidden)
        for u in range(vocab):
            weight += torch.outer(table[u], basis[u]) / hidden
        self.proj = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_next = torch.cat([x[:, 1:, :], x[:, :1, :]], dim=1)
        return self.proj(x_next)


class _FirstTokenTableHead(nn.Module):
    """Broadcast first-token table head: ``logits[b, p, :] = T[input[b, 0]]``.

    The projection consumes ONLY position 0's hidden state and broadcasts
    over the sequence — every position's logits (draft AND verification) are
    the chain-key table row, so the posterior equals the prior and the two
    joint chains (keyed by their first prompt token) see different tables
    through one shared model.
    """

    def __init__(self, table: torch.Tensor, basis: torch.Tensor) -> None:
        super().__init__()
        vocab, hidden = table.shape[0], basis.shape[1]
        weight = torch.zeros(vocab, hidden)
        for u in range(vocab):
            weight += torch.outer(table[u], basis[u]) / hidden
        self.proj = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.proj(x[:, 0, :])
        return first.unsqueeze(1).expand(-1, x.shape[1], -1).contiguous()


class _TrapDecoder(nn.Module):
    """Generic zero-mixing Hadamard trap decoder (UT-2203/2303 fixture family).

    (i) Zero-mixing layers (attention and MLP contribute exactly zero — the
    residual chain carries the embedding through), (ii) orthogonal Hadamard
    embeddings (``LayerNorm(embed(v)) == t_v``), (iii) the injected table
    head. ``layer_factory`` decides the family attention form (fully
    bidirectional for the LLaDA / Dream / joint faces, block-causal
    dual-form for the FastDLLM face).
    """

    def __init__(
        self,
        head: nn.Module,
        layer_factory: Callable[[], nn.Module],
        model_type: str,
    ) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": model_type, "vocab_size": _TRAP_VOCAB})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(_TRAP_VOCAB, _HIDDEN)
        self.model.layers = nn.ModuleList([layer_factory() for _ in range(_NUM_LAYERS)])
        self.model.norm = nn.LayerNorm(_HIDDEN)
        self.lm_head = head
        basis = _hadamard_basis(_TRAP_VOCAB, _HIDDEN)
        with torch.no_grad():
            for layer in self.model.layers:
                for param in layer.self_attn.parameters():
                    param.zero_()
                for param in layer.mlp.parameters():
                    param.zero_()
            for v in range(_TRAP_VOCAB):
                self.model.embed_tokens.weight[v] = 3.0 * basis[v] + 0.5

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


def _llada_trap_decoder() -> _TrapDecoder:
    """LLaDA-face trap: direct table head (position i predicts i, no shift)."""
    return _TrapDecoder(
        _TableHead(_uncertainty_trap_table(), _hadamard_basis(_TRAP_VOCAB, _HIDDEN)),
        lambda: _BidirectionalLayer(_HIDDEN),
        "llada",
    )


def _fastdllm_trap_decoder() -> _TrapDecoder:
    """FastDLLM-face trap: right-shift head canceling ``right_shift_logits``."""
    return _TrapDecoder(
        _ShiftedTableHead(_uncertainty_trap_table(), _hadamard_basis(_TRAP_VOCAB, _HIDDEN)),
        lambda: _DualFormLayer(_HIDDEN, _TRAP_BLOCK_SIZE),
        "llama",
    )


# ---------------------------------------------------------------------------
# Frozen-seed-formula replay helpers (host side, no production code)
# ---------------------------------------------------------------------------


def _row_draws(
    base_seed: int, unit_idx: int, sub_idx: int, j: int, num_masked: int, k: int
) -> list[int]:
    """Replay the frozen per-candidate seed formula for row ``j``."""
    seed_j = (base_seed * 1_000_003 + unit_idx * 8_191 + sub_idx * 131 + j) & 0x7FFF_FFFF
    generator = torch.Generator(device="cpu").manual_seed(seed_j)
    return torch.randint(0, k, (num_masked,), generator=generator).tolist()


def _search_base_seed(
    target: list[int], unit_idx: int = 0, sub_idx: int = 0, j: int = 1, k: int = 2
) -> int:
    """Search a base seed whose row-``j`` draws equal ``target`` (first unit)."""
    for base in range(1, 100_000):
        if _row_draws(base, unit_idx, sub_idx, j, len(target), k) == target:
            return base
    raise AssertionError(
        f"no base seed found with row-{j} draws == {target} "
        "(fixture construction failure, not a production face)"
    )


def _search_fastdllm_base_seed() -> int:
    """Search a base seed whose row-1 draw is C at EVERY trap small block."""
    for base in range(1, 100_000):
        if all(
            _row_draws(base, 0, small_block, 1, 1, 2) == [1]
            for small_block in range(_TRAP_NUM_SMALL_BLOCKS)
        ):
            return base
    raise AssertionError(
        "no FastDLLM trap base seed found with row-1 draw == C at every "
        "small block (fixture construction failure, not a production face)"
    )


# ---------------------------------------------------------------------------
# Trap bundles
# ---------------------------------------------------------------------------


class _LLaDAUncertaintyTrap:
    """LLaDA uncertainty-trap bundle: prior-split early-stop fixture.

    ``num_tokens=2`` / ``block_size=2`` gives a single fully-masked block
    (M=2, one speculative unit); N=2 by default (the greedy row + one pinned
    random row — ``num_candidates`` raises N for the multi-row faces; row 1
    stays the pinned draw at any N), k=2, ``candidate_eos_aware=False`` (the AR023 frozen-face reason: the
    default EOS-aware continuation row would overwrite the pinned random
    draw), and the row-1 draw pinned to C via the frozen seed formula. The
    greedy row (prior 0.5458) validates first and its posterior 0.7845
    clears margin 0.4 with gap 0.7106 to spare — the unit early-stops with
    stats {1, 1, 1} and commits the same winner a full validation selects.
    """

    def __init__(
        self,
        uncertainty: bool,
        margin: float = 0.4,
        gap: float = 0.05,
        num_candidates: int = 2,
    ) -> None:
        self.inner = _llada_trap_decoder()
        self.llm = _MiniDiffusionLLM(
            self.inner,
            vocab=_TRAP_VOCAB,
            tokenizer=_MiniTokenizer(_TRAP_MASK_ID, _TRAP_EOS_ID, _TRAP_PAD_ID),
        )
        overrides: dict[str, Any] = {
            "seed": _search_base_seed([1, 1]),
            "num_steps": 2,
            "num_tokens": _TRAP_NUM_TOKENS,
            "block_size": _TRAP_NUM_TOKENS,
            "num_block_candidates": num_candidates,
            "candidate_top_k": 2,
            "candidate_include_greedy": True,
            "candidate_eos_aware": False,
        }
        if uncertainty:
            overrides.update(
                uncertainty_validation_order=True,
                uncertainty_margin=margin,
                uncertainty_margin_gap=gap,
            )
        self.config = _llada_config(**overrides)
        self.mff = _make_mff(self.inner)

    def sample(self) -> tuple[LLaDASampler, Any]:
        """Run one speculative generation over the trap prompt."""
        sampler = LLaDASampler(self.llm, config=self.config)
        out = sampler.sample(_prompt(), folded_model=self.mff)
        return sampler, out

    def batched_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.mff.calls if c["kwargs"].get("branch_ids") is not None]

    def draft_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.mff.calls if c["kwargs"].get("branch_ids") is None]


class _FastDLLMUncertaintyTrap:
    """FastDLLM inline uncertainty-trap bundle (right-shift convention).

    ``num_tokens=8`` / ``block_size=8`` / ``small_block_size=1`` gives one
    canvas block of eight single-position speculative units (the UT-2203
    geometry); N=2, k=2, greedy on, and the row-1 draw pinned to C at every
    small block via the frozen seed formula. Every unit repeats the LLaDA
    face's pattern — prior 0.5458 vs 0.0739, posterior 0.7845 vs 0.5724 —
    so the inline face accumulates exactly eight times the per-unit core
    stats {1, 1, 1} -> {8, 8, 8}.
    """

    def __init__(self, uncertainty: bool, margin: float = 0.4, gap: float = 0.05) -> None:
        self.inner = _fastdllm_trap_decoder()
        self.llm = _MiniDiffusionLLM(
            self.inner,
            vocab=_TRAP_VOCAB,
            tokenizer=_MiniTokenizer(_TRAP_MASK_ID, _TRAP_EOS_ID, _TRAP_PAD_ID),
        )
        overrides: dict[str, Any] = {
            "seed": _search_fastdllm_base_seed(),
            "mask_token_id": _TRAP_MASK_ID,
            "stop_token_id": _TRAP_EOS_ID,
            "pad_token_id": _TRAP_PAD_ID,
            "num_block_candidates": 2,
            "candidate_top_k": 2,
            "small_block_size": 1,
            "candidate_include_greedy": True,
        }
        if uncertainty:
            overrides.update(
                uncertainty_validation_order=True,
                uncertainty_margin=margin,
                uncertainty_margin_gap=gap,
            )
        self.config = _fastdllm_config(**overrides)
        self.mff = _make_mff(self.inner)

    def sample(self) -> tuple[FastDLLMSampler, Any]:
        """Run one greedy-on speculative generation over the trap prompt."""
        sampler = FastDLLMSampler(self.llm, config=self.config)
        out = sampler.sample(_prompt(), folded_model=self.mff)
        return sampler, out

    def batched_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.mff.calls if c["kwargs"].get("branch_ids") is not None]

    def draft_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.mff.calls if c["kwargs"].get("branch_ids") is None]


class _CraftedForward:
    """Stub ``forward_fn`` with a fixed winner row (direct-commit fixtures).

    ``logits[b, t, tokens[b, t]] = winner_logit`` for the winner row and
    ``0.0`` for every other row (whose per-fill posterior is the uniform
    ``1 / V``) — the winner row's per-fill posterior is
    ``softmax(winner_logit over V)``, fully controlled by the fixture.
    """

    def __init__(self, vocab: int, winner_row: int = 0, winner_logit: float = 3.0) -> None:
        self.vocab = vocab
        self.winner_row = winner_row
        self.winner_logit = winner_logit
        self.calls: list[dict[str, Any]] = []

    def __call__(self, tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append({"tokens": tokens.detach().clone(), "kwargs": dict(kwargs)})
        n, seq_len = tokens.shape
        logits = torch.zeros(n, seq_len, self.vocab, dtype=torch.float32)
        for b in range(n):
            logits[b, torch.arange(seq_len), tokens[b]] = (
                self.winner_logit if b == self.winner_row else 0.0
            )
        return logits


# ---------------------------------------------------------------------------
# UT-3134: default-off bit-exactness on all three families
# ---------------------------------------------------------------------------


def test_ut3134_default_off_bit_exact_all_families() -> None:
    """UT-3134: default vs explicit uncertainty-off is bit-exact everywhere.

    The feature flag defaults to off, so an explicit
    ``uncertainty_validation_order=False`` must be byte-identical to the
    default config on every family and every face: LLaDA / Dream / Fast-dLLM,
    each on the raw reference path AND the N=2 greedy speculative path.
    Red: the explicit ``uncertainty_validation_order=False`` kwarg fails with
    ``TypeError`` — no family config carries the field yet.
    """
    inner_bd = _build_mini()
    llm_bd = _MiniDiffusionLLM(inner_bd)

    out_default = LLaDASampler(llm_bd, config=_llada_config(seed=3134)).sample(_prompt())
    cfg_off = _llada_config(seed=3134, uncertainty_validation_order=False)
    out_off = LLaDASampler(llm_bd, config=cfg_off).sample(_prompt())
    assert torch.equal(out_off.sequences, out_default.sequences)

    spec = {
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "candidate_eos_aware": False,
    }
    out_spec_default = LLaDASampler(llm_bd, config=_llada_config(seed=3134, **spec)).sample(
        _prompt(), folded_model=_make_mff(inner_bd)
    )
    cfg_spec_off = _llada_config(seed=3134, **dict(spec, uncertainty_validation_order=False))
    out_spec_off = LLaDASampler(llm_bd, config=cfg_spec_off).sample(
        _prompt(), folded_model=_make_mff(inner_bd)
    )
    assert torch.equal(out_spec_off.sequences, out_spec_default.sequences)

    out_d_default = DreamSampler(llm_bd, config=_dream_config(seed=3134)).sample(_prompt())
    cfg_d_off = _dream_config(seed=3134, uncertainty_validation_order=False)
    out_d_off = DreamSampler(llm_bd, config=cfg_d_off).sample(_prompt())
    assert torch.equal(out_d_off.sequences, out_d_default.sequences)

    spec_d = {
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "speculative_steps": 4,
    }
    out_ds_default = DreamSampler(llm_bd, config=_dream_config(seed=3134, **spec_d)).sample(
        _prompt(), folded_model=_make_mff(inner_bd)
    )
    cfg_ds_off = _dream_config(seed=3134, **dict(spec_d, uncertainty_validation_order=False))
    out_ds_off = DreamSampler(llm_bd, config=cfg_ds_off).sample(
        _prompt(), folded_model=_make_mff(inner_bd)
    )
    assert torch.equal(out_ds_off.sequences, out_ds_default.sequences)

    inner_bc = _build_block_causal_mini()
    llm_bc = _MiniDiffusionLLM(inner_bc)

    out_f_default = FastDLLMSampler(llm_bc, config=_fastdllm_config(seed=3134)).sample(_prompt())
    cfg_f_off = _fastdllm_config(seed=3134, uncertainty_validation_order=False)
    out_f_off = FastDLLMSampler(llm_bc, config=cfg_f_off).sample(_prompt())
    assert torch.equal(out_f_off.sequences, out_f_default.sequences)

    spec_f = {
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
    }
    out_fs_default = FastDLLMSampler(llm_bc, config=_fastdllm_config(seed=3134, **spec_f)).sample(
        _prompt(), folded_model=_make_mff(inner_bc)
    )
    cfg_fs_off = _fastdllm_config(seed=3134, **dict(spec_f, uncertainty_validation_order=False))
    out_fs_off = FastDLLMSampler(llm_bc, config=cfg_fs_off).sample(
        _prompt(), folded_model=_make_mff(inner_bc)
    )
    assert torch.equal(out_fs_off.sequences, out_fs_default.sequences)


# ---------------------------------------------------------------------------
# UT-3135: off-state bit-equality (order without early stop)
# ---------------------------------------------------------------------------


def test_ut3135_off_state_bit_equality() -> None:
    """UT-3135: uncertainty-on that never early-stops is bit-identical to off.

    ``uncertainty_margin=1.0`` is unreachable (a softmax min-prob is strictly
    below 1 on a finite-vocab canvas), so the opt-in branch reorders the row
    scoring but never stops early: sequences, per-commit history snapshots,
    and the ``_folding_parent`` chain slot stay bit-identical to the
    default-off run, with stats ``{validated_rows: 4, skipped_rows: 0,
    early_stops: 0}`` for the single N=4 unit.
    Red: the ``uncertainty_validation_order=True`` config kwarg fails with
    ``TypeError`` — the field does not exist yet.
    """
    inner = _build_mini()
    llm = _MiniDiffusionLLM(inner)
    common = {
        "seed": 3135,
        "num_block_candidates": 4,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "candidate_eos_aware": False,
        "return_history": True,
    }

    sampler_off = LLaDASampler(llm, config=_llada_config(**common))
    out_off = sampler_off.sample(_prompt(), folded_model=_make_mff(inner))
    cfg_on = _llada_config(
        **dict(
            common,
            uncertainty_validation_order=True,
            uncertainty_margin=1.0,
            uncertainty_margin_gap=0.0,
        )
    )
    sampler_on = LLaDASampler(llm, config=cfg_on)
    out_on = sampler_on.sample(_prompt(), folded_model=_make_mff(inner))

    assert torch.equal(out_on.sequences, out_off.sequences)
    assert len(out_on.history) == len(out_off.history)
    for h_on, h_off in zip(out_on.history, out_off.history):
        assert torch.equal(h_on, h_off)
    assert sampler_on._folding_parent == sampler_off._folding_parent
    assert sampler_on.uncertainty_stats == {
        "validated_rows": 4,
        "skipped_rows": 0,
        "early_stops": 0,
    }
    assert sampler_off.uncertainty_stats == {
        "validated_rows": 0,
        "skipped_rows": 0,
        "early_stops": 0,
    }


# ---------------------------------------------------------------------------
# UT-3136: prior-descending order + margin early stop (LLaDA trap)
# ---------------------------------------------------------------------------


def test_ut3136_prior_order_early_stop_trap() -> None:
    """UT-3136: the greedy row validates first and stops the loop early.

    On the uncertainty trap the greedy row A tops the draft prior
    (``softmax(T[mask])[A] = 0.5458`` vs the pinned C row's ``0.0739``) AND
    clears the margin (posterior ``softmax(T[A])[A] = 0.7845 >= 0.4`` with
    gap ``0.7106 >= 0.05``), so the loop validates exactly one row, skips
    one, records one early stop, and commits the same winner a full
    validation selects (recomputed from the spied batched logits — LLaDA
    feeds the core the UNSHIFTED logits); the chain continuation switches to
    ``{draft}_cand0``.
    Red: the trap's uncertainty config kwargs fail with ``TypeError``.
    """
    fixture = _LLaDAUncertaintyTrap(uncertainty=True)
    sampler, out = fixture.sample()

    assert sampler.uncertainty_stats == {
        "validated_rows": 1,
        "skipped_rows": 1,
        "early_stops": 1,
    }

    drafts = fixture.draft_calls()
    batched = fixture.batched_calls()
    assert len(drafts) == 1
    assert len(batched) == 1
    cands = batched[0]["args"][0]
    assert tuple(cands.shape) == (2, 5)
    assert torch.equal(cands[0, 3:5], torch.tensor([_TRAP_GREEDY_ID, _TRAP_GREEDY_ID]))
    assert torch.equal(cands[1, 3:5], torch.tensor([_TRAP_ALT_ID, _TRAP_ALT_ID]))

    probs = F.softmax(batched[0]["out"][:, 3:5, :], dim=-1)
    scores = torch.gather(probs, -1, cands[:, 3:5].unsqueeze(-1)).squeeze(-1)
    full_winner = int(scores.min(dim=-1).values.argmax())
    assert full_winner == 0, "the early-stopped winner must equal the full-validation argmax"
    assert torch.equal(out.sequences[0, 3:5], cands[full_winner, 3:5])
    assert torch.equal(out.sequences[0, 3:5], torch.tensor([_TRAP_GREEDY_ID, _TRAP_GREEDY_ID]))

    draft_bid = drafts[0]["kwargs"]["branch_id"]
    assert sampler._folding_parent == f"{draft_bid}_cand0"


# ---------------------------------------------------------------------------
# UT-3137: segment-A construction untouched
# ---------------------------------------------------------------------------


def test_ut3137_construction_untouched() -> None:
    """UT-3137: the opt-in branch leaves segment-A construction untouched.

    N=4 trap, uncertainty on with an unreachable margin 1.0 vs off: every
    ``[N, T]`` batched verification call must carry a bit-identical
    candidate canvas (greedy row + frozen-seed random rows) and identical
    ``branch_ids`` — the reordering lives ONLY in segment B.
    Red: the on-fixture's uncertainty config kwargs fail with ``TypeError``.
    """
    fixture_off = _LLaDAUncertaintyTrap(uncertainty=False, num_candidates=4)
    sampler_off, out_off = fixture_off.sample()
    fixture_on = _LLaDAUncertaintyTrap(uncertainty=True, margin=1.0, gap=0.05, num_candidates=4)
    sampler_on, out_on = fixture_on.sample()

    batched_off = fixture_off.batched_calls()
    batched_on = fixture_on.batched_calls()
    assert len(batched_on) == len(batched_off)
    for call_on, call_off in zip(batched_on, batched_off):
        assert torch.equal(
            call_on["args"][0], call_off["args"][0]
        ), "the candidate canvas must be bit-identical with the branch on"
        assert call_on["kwargs"]["branch_ids"] == call_off["kwargs"]["branch_ids"]
    assert torch.equal(out_on.sequences, out_off.sequences)
    assert sampler_on.uncertainty_stats == {
        "validated_rows": 4,
        "skipped_rows": 0,
        "early_stops": 0,
    }


# ---------------------------------------------------------------------------
# UT-3138: greedy_win_stats untouched
# ---------------------------------------------------------------------------


def test_ut3138_greedy_win_stats_untouched() -> None:
    """UT-3138: the greedy_win_stats semantics are unchanged by the branch.

    On the trap the greedy row wins with AND without the uncertainty branch
    (its posterior 0.7845 tops the C row's 0.5724 under full validation, and
    the early stop fires right after it), so both probes read
    ``{"greedy_wins": 1, "speculative_calls": 1}``.
    Red: the on-fixture's uncertainty config kwargs fail with ``TypeError``.
    """
    sampler_off, _ = _LLaDAUncertaintyTrap(uncertainty=False).sample()
    assert sampler_off.greedy_win_stats == {"greedy_wins": 1, "speculative_calls": 1}

    sampler_on, _ = _LLaDAUncertaintyTrap(uncertainty=True).sample()
    assert sampler_on.greedy_win_stats == {"greedy_wins": 1, "speculative_calls": 1}


# ---------------------------------------------------------------------------
# UT-3139: config validation on all three family configs
# ---------------------------------------------------------------------------


def test_ut3139_config_validation() -> None:
    """UT-3139: ``__post_init__`` validation on all three family configs.

    ``uncertainty_validation_order=True`` with ``num_block_candidates == 0``
    raises ``ValueError`` (ordering is defined only on the speculative
    path); ``uncertainty_margin`` / ``uncertainty_margin_gap`` outside
    ``[0, 1]`` raise ``ValueError`` on every family config; the closed
    boundaries (margin 0.0, gap 1.0) and the bare default construct.
    Red: the first ``pytest.raises(ValueError)`` block fails with
    ``TypeError`` — the uncertainty fields do not exist on any config yet.
    """
    for make in (_llada_config, _dream_config, _fastdllm_config):
        with pytest.raises(ValueError, match=r"num_block_candidates"):
            make(num_block_candidates=0, uncertainty_validation_order=True)
        for bad in (
            {"uncertainty_margin": 1.5},
            {"uncertainty_margin": -0.01},
            {"uncertainty_margin_gap": -0.1},
            {"uncertainty_margin_gap": 1.5},
        ):
            with pytest.raises(ValueError, match=r"uncertainty_margin"):
                make(num_block_candidates=2, uncertainty_validation_order=True, **bad)
        make(
            num_block_candidates=2,
            uncertainty_validation_order=True,
            uncertainty_margin=0.0,
            uncertainty_margin_gap=1.0,
        )
        make()


# ---------------------------------------------------------------------------
# UT-3140: direct commit — early stop skips the EOS row
# ---------------------------------------------------------------------------


def test_ut3140_direct_commit_early_stop_skip() -> None:
    """UT-3140: a direct ``_speculative_commit`` pins the skip semantics.

    V=64 canvas ``[1, 2, 3, M, M, M, M, 7]`` with draft logits
    ``{11: 3.0, EOS: 1.0}`` (greedy row fills 11, the pinned row-1 draws
    ``[1, 1, 1, 1]`` fill EOS everywhere) and a crafted winner-row logit
    ``8.0`` (per-fill posterior ``e^8 / (e^8 + 63) = 0.9793``). With margin
    0.9 / gap 0.05 the loop stops after the greedy row: stats
    ``{1, 1, 1}``, the commit fills 11, ``finished`` stays False (the
    SKIPPED EOS row never touches it), one history snapshot, exactly one
    batched forward. A margin-1.0 twin (unreachable) fully validates: stats
    ``{2, 0, 0}`` with a bit-identical commit.
    Red: the uncertainty config kwargs fail with ``TypeError``.
    """
    llm = _MiniDiffusionLLM(_build_mini())
    common = {
        "seed": _search_base_seed([1, 1, 1, 1]),
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "candidate_eos_aware": False,
        "return_history": True,
    }

    def run(margin: float, gap: float) -> tuple[Any, torch.Tensor, torch.Tensor, list[Any]]:
        cfg = _llada_config(
            **dict(
                common,
                uncertainty_validation_order=True,
                uncertainty_margin=margin,
                uncertainty_margin_gap=gap,
            )
        )
        sampler = LLaDASampler(llm, config=cfg)
        x = torch.tensor([[1, 2, 3, _MASK_ID, _MASK_ID, _MASK_ID, _MASK_ID, 7]])
        finished = torch.zeros(1, dtype=torch.bool)
        history: list[Any] = [x.clone()]
        draft = torch.zeros(1, 4, _VOCAB)
        draft[0, :, 11] = 3.0
        draft[0, :, _EOS_ID] = 1.0
        recorder = _CraftedForward(_VOCAB, winner_row=0, winner_logit=8.0)
        sampler._speculative_commit(
            x=x,
            finished=finished,
            draft_logits_region=draft,
            region_start=3,
            masked_local=torch.tensor([0, 1, 2, 3]),
            draft_branch_id="spec_b0",
            ctx=SpecCtx(mask=torch.ones(8, 8, dtype=torch.bool), position=None),
            unit_forced=torch.zeros(8, dtype=torch.bool),
            folded_model=object(),
            unit_idx=0,
            sub_idx=0,
            history=history,
            forward_fn=recorder,
            stop_token_id=_EOS_ID,
        )
        return sampler, x, finished, history

    sampler_a, x_a, finished_a, history_a = run(0.9, 0.05)
    assert sampler_a.uncertainty_stats == {
        "validated_rows": 1,
        "skipped_rows": 1,
        "early_stops": 1,
    }
    assert torch.equal(x_a[0, 3:7], torch.tensor([11, 11, 11, 11]))
    assert not bool(finished_a[0]), "the skipped EOS row must not touch finished"
    assert len(history_a) == 2, "exactly one post-commit snapshot"

    sampler_b, x_b, finished_b, history_b = run(1.0, 0.05)
    assert sampler_b.uncertainty_stats == {
        "validated_rows": 2,
        "skipped_rows": 0,
        "early_stops": 0,
    }
    assert torch.equal(x_b, x_a)
    assert torch.equal(history_b[-1], history_a[-1])
    assert not bool(finished_b[0])


# ---------------------------------------------------------------------------
# UT-3141: shared pure helpers + core/inline parity
# ---------------------------------------------------------------------------


def test_ut3141_shared_helpers_and_parity() -> None:
    """UT-3141: the shared helpers and the core/inline consumption parity.

    (a) ``plan_uncertainty_order`` sorts descending with ties to the smaller
    index and raises ``ValueError`` on empty / non-1-D input;
    ``should_early_stop`` is the exact conjunction
    ``q_best >= margin and (q_best - remaining_max_prior) >= margin_gap``.
    (b) The LLaDA trap accumulates ``{1, 1, 1}``. (c) The Fast-dLLM inline
    twin repeats the same per-unit pattern at all eight small blocks:
    ``{8, 8, 8}`` — exactly eight times the core face per key — with eight
    batched calls whose winner equals the full-validation argmax (recomputed
    from the spied raw logits through ``right_shift_logits``) and bit-equal
    on/off sequences.
    Red: the function-local import fails with ``ImportError`` —
    ``speculative_core`` exports neither helper.
    """
    from actfold.models.speculative_core import (  # noqa: PLC0415
        plan_uncertainty_order,
        should_early_stop,
    )

    assert plan_uncertainty_order(torch.tensor([0.1, 0.9, 0.5, 0.9])) == [1, 3, 2, 0]
    assert plan_uncertainty_order(torch.tensor([0.5, 0.5, 0.5])) == [0, 1, 2]
    with pytest.raises(ValueError):
        plan_uncertainty_order(torch.zeros(0))
    with pytest.raises(ValueError):
        plan_uncertainty_order(torch.zeros(2, 2))
    assert should_early_stop(0.9, 0.85, 0.9, 0.05)
    assert not should_early_stop(0.9, 0.86, 0.9, 0.05)
    assert not should_early_stop(0.89, 0.84, 0.9, 0.05)

    llada_sampler, _ = _LLaDAUncertaintyTrap(uncertainty=True).sample()
    llada_stats = dict(llada_sampler.uncertainty_stats)
    assert llada_stats == {"validated_rows": 1, "skipped_rows": 1, "early_stops": 1}

    fixture = _FastDLLMUncertaintyTrap(uncertainty=True)
    fast_sampler, fast_out = fixture.sample()
    fast_stats = dict(fast_sampler.uncertainty_stats)
    assert fast_stats == {"validated_rows": 8, "skipped_rows": 8, "early_stops": 8}
    for key in llada_stats:
        assert (
            fast_stats[key] // 8 == llada_stats[key]
        ), "the inline face must accumulate exactly 8x the per-unit core stats"

    batched = fixture.batched_calls()
    assert len(batched) == 8, "one batched verification per small block"
    for small_block, call in enumerate(batched):
        lo = _TRAP_BLOCK_START + small_block
        cands = call["args"][0]
        probs = F.softmax(right_shift_logits(call["out"])[:, lo : lo + 1, :], dim=-1)
        scores = torch.gather(probs, -1, cands[:, lo : lo + 1].unsqueeze(-1)).squeeze(-1)
        assert int(scores.min(dim=-1).values.argmax()) == 0, (
            f"small block {small_block}: the early-stopped winner must equal "
            "the full-validation argmax under the right-shift convention"
        )
        assert torch.equal(fast_out.sequences[0, lo : lo + 1], cands[0, lo : lo + 1])

    fixture_off = _FastDLLMUncertaintyTrap(uncertainty=False)
    _, fast_out_off = fixture_off.sample()
    assert torch.equal(fast_out.sequences, fast_out_off.sequences)


# ---------------------------------------------------------------------------
# UT-3142: joint chains — per-chain decisions through one shared model
# ---------------------------------------------------------------------------


def test_ut3142_joint_chains_per_chain_decisions() -> None:
    """UT-3142: ``sample_joint`` makes per-chain uncertainty decisions.

    A first-token broadcast head keys each chain's posterior table by its
    first prompt token through ONE shared model: chain 0 (row
    ``T[7] = {X0: 8.0, X1: 0.5}``, posterior == prior) stops after the
    greedy row (``q = 0.9948 >= 0.9``); chain 1 (``T[8] = {Y0: 2.0,
    Y1: 1.0}``, best posterior 0.3065) fully validates. The joint run
    issues ONE merged ``[4, T]`` batched call, accumulates the summed stats
    ``{3, 1, 1}``, and each chain's output / stats match a back-to-back
    solo ``sample()`` on the same config (the frozen seed formula carries no
    chain component).
    Red: the uncertainty config kwargs fail with ``TypeError``.
    """
    basis = _hadamard_basis(_TRAP_VOCAB, _HIDDEN)
    inner = _TrapDecoder(
        _FirstTokenTableHead(_joint_trap_table(), basis),
        lambda: _BidirectionalLayer(_HIDDEN),
        "llada",
    )
    llm = _MiniDiffusionLLM(
        inner,
        vocab=_TRAP_VOCAB,
        tokenizer=_MiniTokenizer(_TRAP_MASK_ID, _TRAP_EOS_ID, _TRAP_PAD_ID),
    )
    cfg = _llada_config(
        seed=_search_base_seed([1, 1]),
        num_steps=2,
        num_tokens=2,
        block_size=2,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        candidate_eos_aware=False,
        uncertainty_validation_order=True,
        uncertainty_margin=0.9,
        uncertainty_margin_gap=0.05,
    )
    prompts = [torch.tensor([[_JOINT_KEY0, 1, 2]]), torch.tensor([[_JOINT_KEY1, 1, 2]])]

    joint_mff = _make_mff(inner, multi_parent_batch=True)
    joint_sampler = LLaDASampler(llm, config=cfg)
    outs = joint_sampler.sample_joint(prompts, folded_model=joint_mff)

    assert joint_sampler.uncertainty_stats == {
        "validated_rows": 3,
        "skipped_rows": 1,
        "early_stops": 1,
    }
    assert torch.equal(outs[0].sequences[0, 3:5], torch.tensor([_JOINT_X0, _JOINT_X0]))
    assert torch.equal(outs[1].sequences[0, 3:5], torch.tensor([_JOINT_Y0, _JOINT_Y0]))
    armed = any(
        len(c["kwargs"]["branch_ids"]) == 4
        for c in joint_mff.calls
        if c["kwargs"].get("branch_ids") is not None
    )
    assert armed, "the joint path must issue ONE merged [sum(N), T] batched call"

    expected_solo_stats = [
        {"validated_rows": 1, "skipped_rows": 1, "early_stops": 1},
        {"validated_rows": 2, "skipped_rows": 0, "early_stops": 0},
    ]
    totals = {"validated_rows": 0, "skipped_rows": 0, "early_stops": 0}
    for chain, prompt in enumerate(prompts):
        solo = LLaDASampler(llm, config=cfg)
        out_solo = solo.sample(prompt, folded_model=_make_mff(inner))
        assert torch.equal(
            out_solo.sequences, outs[chain].sequences
        ), f"chain {chain}: the joint output must match the back-to-back solo run"
        assert solo.uncertainty_stats == expected_solo_stats[chain]
        for key in totals:
            totals[key] += solo.uncertainty_stats[key]
    assert totals == {"validated_rows": 3, "skipped_rows": 1, "early_stops": 1}


# ---------------------------------------------------------------------------
# UT-3143: margin / margin_gap semantics (min-over-fill prior twin)
# ---------------------------------------------------------------------------


def test_ut3143_margin_gap_semantics() -> None:
    """UT-3143: the gap twin pins the exact stop predicate and prior form.

    V=16 canvas ``[1, 2, 3, M, M, 7]`` with draft logits ``{5: 3.0,
    6: 1.0}`` (greedy row ``[5, 5]``, pinned row-1 draws ``[0, 1]`` ->
    ``[5, 6]``) and a crafted winner logit ``ln(15)`` (per-fill posterior
    exactly 0.5). Margin 0.45 passes; the stop then hinges on the gap
    against the C row's MIN-over-fill prior ``0.0739``: ``0.5 - 0.0739 =
    0.4261`` clears gap 0.40 (twin A stops, ``{1, 1, 1}``) but not 0.45
    (twin B fully validates, ``{2, 0, 0}``) — a mean- or max-over-fill
    prior mutant (0.3099 / 0.5458) and an ascending-order mutant both fail
    twin A. Both twins commit the same greedy row.
    Red: the uncertainty config kwargs fail with ``TypeError``.
    """
    llm = _MiniDiffusionLLM(_build_mini())
    common = {
        "seed": _search_base_seed([0, 1]),
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "candidate_eos_aware": False,
        "return_history": True,
    }

    def run(gap: float) -> tuple[Any, torch.Tensor, torch.Tensor]:
        cfg = _llada_config(
            **dict(
                common,
                uncertainty_validation_order=True,
                uncertainty_margin=0.45,
                uncertainty_margin_gap=gap,
            )
        )
        sampler = LLaDASampler(llm, config=cfg)
        x = torch.tensor([[1, 2, 3, _TRAP_MASK_ID, _TRAP_MASK_ID, 7]])
        finished = torch.zeros(1, dtype=torch.bool)
        draft = torch.zeros(1, 2, _TRAP_VOCAB)
        draft[0, :, 5] = 3.0
        draft[0, :, 6] = 1.0
        recorder = _CraftedForward(_TRAP_VOCAB, winner_row=0, winner_logit=math.log(15.0))
        sampler._speculative_commit(
            x=x,
            finished=finished,
            draft_logits_region=draft,
            region_start=3,
            masked_local=torch.tensor([0, 1]),
            draft_branch_id="spec_b0",
            ctx=SpecCtx(mask=torch.ones(6, 6, dtype=torch.bool), position=None),
            unit_forced=torch.zeros(6, dtype=torch.bool),
            folded_model=object(),
            unit_idx=0,
            sub_idx=0,
            history=[],
            forward_fn=recorder,
            stop_token_id=_TRAP_EOS_ID,
        )
        return sampler, x, finished

    sampler_a, x_a, finished_a = run(0.40)
    assert sampler_a.uncertainty_stats == {
        "validated_rows": 1,
        "skipped_rows": 1,
        "early_stops": 1,
    }
    assert torch.equal(x_a[0, 3:5], torch.tensor([5, 5]))
    assert not bool(finished_a[0])

    sampler_b, x_b, finished_b = run(0.45)
    assert sampler_b.uncertainty_stats == {
        "validated_rows": 2,
        "skipped_rows": 0,
        "early_stops": 0,
    }
    assert torch.equal(x_b, x_a)
    assert not bool(finished_b[0])


# ---------------------------------------------------------------------------
# UT-3144: the read-only uncertainty_stats probe
# ---------------------------------------------------------------------------


def test_ut3144_uncertainty_stats_probe() -> None:
    """UT-3144: ``uncertainty_stats`` is a read-only, accumulating probe.

    The default read is ``{"validated_rows": 0, "skipped_rows": 0,
    "early_stops": 0}`` and stays there through raw, N=0-folded, and N=2
    speculative samples with the branch off; assignment raises
    ``AttributeError`` and mutating the returned dict never leaks in; on the
    trap the SAME sampler instance accumulates ``{1, 1, 1}`` -> ``{2, 2,
    2}`` across two ``sample`` calls with bit-equal sequences.
    Red: the probe read fails with ``AttributeError`` — no host sampler
    defines ``uncertainty_stats``.
    """
    inner = _build_mini()
    llm = _MiniDiffusionLLM(inner)
    sampler = LLaDASampler(llm, config=_llada_config(seed=3144))
    zero = {"validated_rows": 0, "skipped_rows": 0, "early_stops": 0}
    assert sampler.uncertainty_stats == zero

    sampler.sample(_prompt())
    sampler.sample(_prompt(), folded_model=_make_mff(inner))
    assert sampler.uncertainty_stats == zero

    cfg_spec = _llada_config(
        seed=3144,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
    )
    sampler_spec = LLaDASampler(llm, config=cfg_spec)
    sampler_spec.sample(_prompt(), folded_model=_make_mff(inner))
    assert sampler_spec.uncertainty_stats == zero

    with pytest.raises(AttributeError):
        sampler.uncertainty_stats = dict(zero, validated_rows=99)  # type: ignore[misc]
    probe = sampler.uncertainty_stats
    probe["validated_rows"] = 99
    assert sampler.uncertainty_stats == zero

    fixture = _LLaDAUncertaintyTrap(uncertainty=True)
    trap_sampler = LLaDASampler(fixture.llm, config=fixture.config)
    out_first = trap_sampler.sample(_prompt(), folded_model=fixture.mff)
    assert trap_sampler.uncertainty_stats == {
        "validated_rows": 1,
        "skipped_rows": 1,
        "early_stops": 1,
    }
    fixture.mff = _make_mff(fixture.inner)
    out_second = trap_sampler.sample(_prompt(), folded_model=fixture.mff)
    assert torch.equal(out_second.sequences, out_first.sequences)
    assert trap_sampler.uncertainty_stats == {
        "validated_rows": 2,
        "skipped_rows": 2,
        "early_stops": 2,
    }
