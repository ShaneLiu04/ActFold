"""Red-phase tests for AR023: LLaDA-family speculative decoding (UT-2301..UT-2308).

Contract: ``specs/changes/AR023-cross-family-speculative/design.md`` §4.1.1-4.1.3,
§4.2.2, §6.1 and ``srs.md`` FR2/FR4 — porting the AR021 generation-loop folding +
AR022 greedy-anchored speculative candidates onto ``LLaDASampler`` through the
shared mechanism core ``SpeculativeCore`` (right-padded block canvas family
adapter):

* UT-2301: candidate construction frozen constants — row 0 = the draft argmax
  (``sorted=True`` topk first entry, pure device indexing), random rows via the
  frozen seed formula ``(S0*1_000_003 + unit_idx*8_191 + sub_idx*131 + j) &
  0x7FFF_FFFF`` on a dedicated CPU generator, zero device readback and zero
  global-RNG consumption inside the construction window;
  ``_supports_branch_ids`` parity with ``fast_dllm_sampler.py:325-349`` on the
  True/False probe routes;
* UT-2302: N semantics — greedy on keeps random rows ``j=1..N-1`` bit-identical
  to a greedy-off run on the same canvas (j keeps its row number); ``[N, T]``
  batch shape and ``branch_ids`` naming ``[f"{draft}_cand{j}"]``;
* UT-2303: the greedy row competes under the unchanged min-prob rule and CAN
  LOSE (LLaDA-side posterior-trap fixture with the seed pinned via the frozen
  formula — the ``_PosteriorTrapFixture`` pattern without right shift);
* UT-2304: min-prob selection over the FILL positions, commit == winner row,
  ``_folding_parent == f"{draft}_cand{j_star}"``, eos → finished, and the LLaDA
  no-shift logits convention (position i predicts position i);
* UT-2305: the read-only ``greedy_win_stats`` probe (default ``{0, 0}``,
  assignment raises AttributeError, counts only on the greedy-anchored path);
* UT-2306: ``num_block_candidates=0`` default == explicit off bit-exact (raw
  and N=0 folded faces) — the reference path is undisturbed;
* UT-2307: block speculative unit — one draft + one ``[N, T]`` batched call per
  block (skipping the block's remaining remasking iterations), entry validation
  (no folded_model / branch_ids unsupported / B>1+N>0 → ValueError), the
  no-masked-position skip, and the ``_build_unit_forced`` block-superset
  content;
* UT-2308: the LLaDA fold context — ``_spec_ctx_for`` object pinning keyed by
  ``(T, prompt_len)``, the ``[T, T]`` pad-column mask (element-equal to the 1-D
  reference mask), and the LRU eviction cap 32 (parity with
  ``_MAX_FOLD_CTX_ENTRIES``).

Red state (why every test fails today) — each failure is a production-missing
face, never a test-own bug:

* ``actfold.models.speculative_core`` does not exist — the direct-Core tests
  (UT-2301/2302/2304/2305, the ``_build_unit_forced``/``_spec_ctx_for``/
  ``_ctx_cache_*`` faces) fail with ``ModuleNotFoundError`` (SpecCtx/Core
  import) or ``AttributeError`` (missing host method on a today-constructible
  sampler);
* ``LLaDASamplerConfig`` has no ``num_block_candidates`` / ``candidate_top_k``
  / ``candidate_include_greedy`` fields — every explicit kwarg pass fails with
  ``TypeError`` (UT-2303, UT-2304 shift face, UT-2306, UT-2307).

Fixtures follow ``tests/test_speculative_greedy_alignment.py`` (self-contained
LLaMA-layout mini decoder, ``_MiniDiffusionLLM``, ``_SpyMFF``, ``_make_mff``,
Hadamard-basis posterior trap) with the AR023 change: the attention is fully
bidirectional and accepts every mask form the LLaDA family may pass (``[B, T]``
/ ``[1, T]`` padding masks, ``[T, T]`` prebuilt bool masks).
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from actfold.core import ActivationCache, SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.models.base import DiffusionLLM
from actfold.models.fast_dllm_sampler import _MAX_FOLD_CTX_ENTRIES, FastDLLMSampler
from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig
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

# LLaDA posterior-trap constants (UT-2303): trap vocab 16 over hidden 32 so 16
# pairwise-orthogonal zero-mean basis vectors exist (Hadamard rows 1..16).
_TRAP_VOCAB = 16
_TRAP_MASK_ID = 15
_TRAP_EOS_ID = 14
_TRAP_PAD_ID = 13
_TRAP_GREEDY_ID = 5  # "A": draft argmax; T[A] uniform -> low posterior self-prob
_TRAP_ALT_ID = 6  # "C": draft runner-up; T[C][C]=3.0 -> high posterior self-prob
_TRAP_PROMPT_LEN = 3
_TRAP_NUM_TOKENS = 2  # single block of 2 masked positions (M=2)


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
    ``logits[:, p, (p+2) % V] = 2.0 + 0.01*p`` — input-independent, so the
    unshifted argmax at p is ``(p+1) % V`` while a right-shifted argmax at p
    would be ``p % V`` (the shift-convention discriminator, UT-2304).
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
        self.config = type("Config", (), {"model_type": "llada", "vocab_size": vocab})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, _HIDDEN)
        self.model.layers = nn.ModuleList([_BidirectionalLayer(_HIDDEN) for _ in range(2)])
        self.model.norm = nn.LayerNorm(_HIDDEN)
        if head == "spike":
            self.lm_head = _SpikeHead(vocab, 5, 6)
        elif head == "constant_eos":
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


# ---------------------------------------------------------------------------
# LLaDA posterior-trap fixture (UT-2303; no right shift — LLaDA convention)
# ---------------------------------------------------------------------------


def _hadamard_basis(vocab: int, hidden: int) -> torch.Tensor:
    """Zero-mean unit-variance pairwise-orthogonal rows (Sylvester Hadamard)."""
    rows = []
    for v in range(vocab):
        i = v + 1
        rows.append([-1.0 if bin(i & h).count("1") % 2 else 1.0 for h in range(hidden)])
    return torch.tensor(rows, dtype=torch.float32)


def _trap_table() -> torch.Tensor:
    """Posterior table ``T``: T[mask]={A:3, C:1}, T[A]=uniform, T[C][C]=3."""
    table = torch.zeros(_TRAP_VOCAB, _TRAP_VOCAB)
    table[_TRAP_MASK_ID, _TRAP_GREEDY_ID] = 3.0
    table[_TRAP_MASK_ID, _TRAP_ALT_ID] = 1.0
    table[_TRAP_ALT_ID, _TRAP_ALT_ID] = 3.0
    return table


class _TableHead(nn.Module):
    """Direct table head: ``logits[:, p] = T[input[:, p]]`` (NO shift).

    ``W = sum_u outer(T[u], t_u) / H`` over the orthogonal basis makes
    ``W @ t_in == T[in]`` exactly (up to the LayerNorm eps scale), so the
    commit posterior of a filled token is ``softmax(T[fill])[fill]`` — fully
    controlled by the candidate content (LLaDA feeds Core the UNshifted
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


class _TrapModel(nn.Module):
    """LLaMA-layout zero-mixing posterior-trap decoder (design §6.1 UT-2203).

    (i) Zero-mixing layers (attention and MLP contribute exactly zero — the
    residual chain carries the embedding through), (ii) orthogonal Hadamard
    embeddings (``LayerNorm(embed(v)) == t_v``), (iii) the direct table head.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llada", "vocab_size": _TRAP_VOCAB})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(_TRAP_VOCAB, _HIDDEN)
        self.model.layers = nn.ModuleList([_BidirectionalLayer(_HIDDEN) for _ in range(2)])
        self.model.norm = nn.LayerNorm(_HIDDEN)
        basis = _hadamard_basis(_TRAP_VOCAB, _HIDDEN)
        self.lm_head = _TableHead(_trap_table(), basis)
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


def _trap_row1_draw(base_seed: int) -> int:
    """Replay the frozen seed formula for the trap geometry (host side).

    First LLaDA speculative unit: unit_idx=0, sub_idx=0, j=1, k_eff=2,
    num_masked=2 — the first of the two per-position draws decides the
    all-A / all-C pattern (both draws equal by the pinned seed search).
    """
    seed_j = (base_seed * 1_000_003 + 0 * 8_191 + 0 * 131 + 1) & 0x7FFF_FFFF
    generator = torch.Generator(device="cpu").manual_seed(seed_j)
    return int(torch.randint(0, 2, (1,), generator=generator))


def _trap_base_seed(row1_draw: int) -> int:
    """Search a base seed whose first-unit row-1 draws are all-``row1_draw``."""
    for base in range(1, 100_000):
        if _trap_row1_draw(base) == row1_draw:
            return base
    raise AssertionError(
        f"no trap base seed found with row-1 draw == {row1_draw} "
        "(fixture construction failure, not a production face)"
    )


class _LLaDATrapFixture:
    """LLaDA posterior-trap bundle: the greedy row CAN lose the min-prob race.

    num_tokens=2 / block_size=2 gives a single fully-masked block (M=2, one
    speculative unit); N=2 (greedy row + one random row), k=2, with the row-1
    draw pinned via the frozen seed formula. On the greedy-loses pattern
    (draws all C) j*==1 and the commit is C at both positions; on the twin
    pattern (draws all A) the rows tie and the trivial argmax selects row 0.
    """

    def __init__(self, device: str, row1_draw: int) -> None:
        self.inner = _TrapModel()
        self.inner.to(device)
        self.inner.eval()
        self.llm = _MiniDiffusionLLM(
            self.inner,
            vocab=_TRAP_VOCAB,
            tokenizer=_MiniTokenizer(_TRAP_MASK_ID, _TRAP_EOS_ID, _TRAP_PAD_ID),
        )
        self.config = _llada_config(
            seed=_trap_base_seed(row1_draw),
            num_steps=2,
            num_tokens=_TRAP_NUM_TOKENS,
            block_size=_TRAP_NUM_TOKENS,
            num_block_candidates=2,
            candidate_top_k=2,
            candidate_include_greedy=True,
            # AR023 semantic restoration (AR024 design §4.1.6 note): the
            # default flipped to eos-aware in AR024, which makes row N-1 the
            # argmax continuation row; this frozen AR023 face pins the pinned
            # random-draw semantics explicitly.
            candidate_eos_aware=False,
        )
        self.mff = _make_mff(self.inner, device)

    def sample(self) -> tuple[LLaDASampler, Any]:
        sampler = LLaDASampler(self.llm, config=self.config)
        prompt = torch.tensor([[1, 2, 3]], device=self.inner.model.embed_tokens.weight.device)
        out = sampler.sample(prompt, folded_model=self.mff)
        return sampler, out

    def batched_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.mff.calls if c["kwargs"].get("branch_ids") is not None]

    def draft_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.mff.calls if c["kwargs"].get("branch_ids") is None]


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


def _llada_config(**overrides: Any) -> LLaDASamplerConfig:
    """LLaDASamplerConfig with pinned defaults (new AR023 fields via overrides)."""
    values: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 8,
        "temperature": 0.0,
        "seed": 2300,
        "block_size": _BLOCK_SIZE,
    }
    values.update(overrides)
    return LLaDASamplerConfig(**values)


def _prompt(device: str) -> torch.Tensor:
    return torch.tensor([[1, 2, 3]], device=device)


class _ForwardRecorder:
    """Stub ``forward_fn`` recording every call and crafting batched logits.

    The frozen ``SpeculativeCore`` contract calls
    ``forward_fn(cands, branch_id=None, parent_branch_id=draft, branch_ids=...,
    forced_divergent=...)``; this recorder accepts anything, stores the
    arguments, and returns ``[N, T, V]`` logits whose winning row is fixed
    (``logits[b, t, cands[b, t]] = 3.0 if b == winner_row else 0.0``) so the
    min-prob selection is draw-independent.
    """

    def __init__(self, vocab: int, winner_row: int = 0) -> None:
        self.vocab = vocab
        self.winner_row = winner_row
        self.calls: list[dict[str, Any]] = []

    def __call__(self, tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append({"tokens": tokens.detach().clone(), "kwargs": dict(kwargs)})
        n, seq_len = tokens.shape
        logits = torch.zeros(n, seq_len, self.vocab, dtype=torch.float32)
        for b in range(n):
            logits[b, torch.arange(seq_len), tokens[b]] = 3.0 if b == self.winner_row else 0.0
        return logits


class _CoreHostFactory:
    """Build a minimal ``SpeculativeCore`` host (design §4.1.2 host contract).

    The host carries the config namespace
    (num_block_candidates/candidate_top_k/candidate_include_greedy/seed/
    return_history), the ``_greedy_wins``/``_greedy_spec_calls`` counters,
    and the ``_folding_parent`` chain slot — nothing else, so the mechanism
    faces are tested in isolation from any family adapter.
    """

    def __init__(self, device: str) -> None:
        self.device = device

    def host(self, **cfg: Any) -> Any:
        from actfold.models.speculative_core import SpeculativeCore  # noqa: PLC0415

        values: dict[str, Any] = {
            "num_block_candidates": 4,
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
            mask=torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
            position=None,
        )

    def canvas(self) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, _MASK_ID, _MASK_ID, _MASK_ID, _MASK_ID, 7]])

    def draft_region(self) -> torch.Tensor:
        """[1, 4, V] draft logits with position-dependent top-2."""
        draft = torch.zeros(1, 4, _VOCAB)
        for m in range(4):
            draft[0, m, (m * 7 + 5) % _VOCAB] = 3.0
            draft[0, m, (m * 7 + 6) % _VOCAB] = 1.0
        return draft


def _replay_random_rows(
    draft_region: torch.Tensor, base_seed: int, unit_idx: int, sub_idx: int, n: int, k: int
) -> torch.Tensor:
    """Replay the frozen seed formula for rows j (host-side, no production code)."""
    num_masked = int(draft_region.shape[1])
    draft_set = torch.topk(draft_region, k, dim=-1).indices[0]  # [M, k]
    rows = []
    for j in range(n):
        seed_j = (base_seed * 1_000_003 + unit_idx * 8_191 + sub_idx * 131 + j) & 0x7FFF_FFFF
        generator = torch.Generator(device="cpu").manual_seed(seed_j)
        choice = torch.randint(0, k, (num_masked,), generator=generator)
        rows.append(torch.gather(draft_set, 1, choice.unsqueeze(1)).squeeze(1))
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# UT-2301: shared-core candidate construction (frozen constants)
# ---------------------------------------------------------------------------


def test_ut2301_seed_formula_constants_pinned() -> None:
    """UT-2301: random rows replay the frozen seed formula exactly.

    With unit_idx=2 and sub_idx=3 both index terms (8_191 and 131) are
    nonzero and distinct, so a production formula missing either term, using
    the wrong constants, or consuming the wrong row number produces different
    draws and fails the bit-exact comparison. The greedy row must equal the
    draft top-k ordered-set first entry (the frozen argmax definition).
    """
    factory = _CoreHostFactory("cpu")
    host = factory.host()
    x = factory.canvas()
    unit_idx, sub_idx, n, k = 2, 3, 4, 2
    recorder = _ForwardRecorder(_VOCAB, winner_row=0)

    host._speculative_commit(
        x=x.clone(),
        finished=torch.zeros(1, dtype=torch.bool),
        draft_logits_region=factory.draft_region(),
        region_start=3,
        masked_local=torch.tensor([0, 1, 2, 3]),
        draft_branch_id="spec_b0",
        ctx=factory.ctx(8),
        unit_forced=torch.tensor([False, False, False, True, True, True, True, True]),
        folded_model=object(),
        unit_idx=unit_idx,
        sub_idx=sub_idx,
        history=[],
        forward_fn=recorder,
        stop_token_id=_EOS_ID,
    )

    assert len(recorder.calls) == 1
    cands = recorder.calls[0]["tokens"]
    assert tuple(cands.shape) == (n, 8), "[N, T] batch shape"
    draft = factory.draft_region()
    expected_row0 = torch.topk(draft, k, dim=-1).indices[..., 0][0]
    assert torch.equal(cands[0, 3:7], expected_row0), "row 0 == draft argmax fill"
    expected_random = _replay_random_rows(draft, 2300, unit_idx, sub_idx, n, k)
    for j in range(1, n):
        assert torch.equal(cands[j, 3:7], expected_random[j]), (
            f"row {j} must replay the frozen seed formula "
            "(S0*1_000_003 + unit_idx*8_191 + sub_idx*131 + j) & 0x7FFF_FFFF"
        )

    # The 31-bit mask term is pinned by a base seed whose masked intermediate
    # has bit 30 set: (4000*1_000_003) & 0x7FFF_FFFF = 1_852_527_552 >= 2^30,
    # so a 30-bit mask mutant produces different draws (and a different
    # commit) on this face.
    host_big = factory.host(seed=4000)
    recorder_big = _ForwardRecorder(_VOCAB, winner_row=0)
    host_big._speculative_commit(
        x=factory.canvas(),
        finished=torch.zeros(1, dtype=torch.bool),
        draft_logits_region=draft,
        region_start=3,
        masked_local=torch.tensor([0, 1, 2, 3]),
        draft_branch_id="spec_b0",
        ctx=factory.ctx(8),
        unit_forced=torch.zeros(8, dtype=torch.bool),
        folded_model=object(),
        unit_idx=0,
        sub_idx=0,
        history=[],
        forward_fn=recorder_big,
        stop_token_id=_EOS_ID,
    )
    expected_big = _replay_random_rows(draft, 4000, 0, 0, n, k)
    cands_big = recorder_big.calls[0]["tokens"]
    for j in range(1, n):
        assert torch.equal(
            cands_big[j, 3:7], expected_big[j]
        ), f"row {j} must replay the frozen formula with the full 31-bit mask"


def test_ut2301_greedy_row_argmax_and_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2301: greedy on builds N-1 generators; the global RNG is untouched.

    The dedicated CPU generator consumption is N-1 with greedy on (row 0 is
    pure device indexing) and N with greedy off; with ``seed=None`` the
    global RNG state snapshot must be unchanged across the whole commit.
    """
    factory = _CoreHostFactory("cpu")
    real_generator = torch.Generator

    for include_greedy, expected_gens in ((True, 3), (False, 4)):
        host = factory.host(candidate_include_greedy=include_greedy, seed=None)
        recorder = _ForwardRecorder(_VOCAB)
        counter = {"n": 0}

        def _counting_generator(*args: Any, **kwargs: Any) -> Any:
            counter["n"] += 1
            return real_generator(*args, **kwargs)

        monkeypatch.setattr(torch, "Generator", _counting_generator)
        torch.manual_seed(97531)
        state_before = torch.get_rng_state()
        host._speculative_commit(
            x=factory.canvas(),
            finished=torch.zeros(1, dtype=torch.bool),
            draft_logits_region=factory.draft_region(),
            region_start=3,
            masked_local=torch.tensor([0, 1, 2, 3]),
            draft_branch_id="spec_b0",
            ctx=factory.ctx(8),
            unit_forced=torch.zeros(8, dtype=torch.bool),
            folded_model=object(),
            unit_idx=0,
            sub_idx=0,
            history=[],
            forward_fn=recorder,
            stop_token_id=_EOS_ID,
        )
        monkeypatch.undo()
        assert (
            counter["n"] == expected_gens
        ), f"greedy={include_greedy}: expected {expected_gens} generator constructions"
        assert torch.equal(
            torch.get_rng_state(), state_before
        ), "the construction must consume only dedicated CPU generators"
        # seed=None pins the frozen base-seed fallback S0=0 (fast_dllm
        # parity): the random rows replay the seed formula with base 0.
        expected_s0 = _replay_random_rows(factory.draft_region(), 0, 0, 0, 4, 2)
        cands = recorder.calls[0]["tokens"]
        for j in range(1 if include_greedy else 0, 4):
            assert torch.equal(cands[j, 3:7], expected_s0[j]), (
                f"greedy={include_greedy}: row {j} must replay the frozen "
                "formula with S0=0 when seed is None"
            )


def test_ut2301_zero_readback_construction_window() -> None:
    """UT-2301: zero .item/.tolist/.cpu inside the candidate-construction window.

    The window opens at commit entry and closes when the batched call arrives
    at the ``forward_fn`` (the AR022 ``_ReadbackSpyMFF`` pattern — the
    ``int(argmax)`` selection readback happens AFTER the batched call and
    outside the window).
    """
    factory = _CoreHostFactory("cpu")
    host = factory.host()
    originals: dict[str, Any] = {}
    armed = {"yes": True}

    def _make_forbidden(name: str) -> Any:
        def _forbidden(tensor_self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                f"device readback .{name}() inside the candidate construction window"
            )

        return _forbidden

    class _DisarmingRecorder(_ForwardRecorder):
        def __call__(self, tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            for name, original in originals.items():
                setattr(torch.Tensor, name, original)
            originals.clear()
            armed["yes"] = False
            return super().__call__(tokens, **kwargs)

    for name in ("item", "tolist", "cpu"):
        originals[name] = getattr(torch.Tensor, name)
        setattr(torch.Tensor, name, _make_forbidden(name))
    try:
        host._speculative_commit(
            x=factory.canvas(),
            finished=torch.zeros(1, dtype=torch.bool),
            draft_logits_region=factory.draft_region(),
            region_start=3,
            masked_local=torch.tensor([0, 1, 2, 3]),
            draft_branch_id="spec_b0",
            ctx=factory.ctx(8),
            unit_forced=torch.zeros(8, dtype=torch.bool),
            folded_model=object(),
            unit_idx=0,
            sub_idx=0,
            history=[],
            forward_fn=_DisarmingRecorder(_VOCAB),
            stop_token_id=_EOS_ID,
        )
    finally:
        for name, original in originals.items():
            setattr(torch.Tensor, name, original)
        originals.clear()
    assert not armed["yes"], "the batched call must have arrived and disarmed the spy"


def test_ut2301_supports_branch_ids_parity() -> None:
    """UT-2301: ``_supports_branch_ids`` matches the Fast-dLLM probe on all routes.

    Explicit ``branch_ids`` parameter → True; ``**kwargs`` passthrough → True;
    neither → False; nn.Module variants route through ``.forward``;
    non-callables → False. Every route must agree with
    ``FastDLLMSampler._supports_branch_ids`` (fast_dllm_sampler.py:325-349).
    """
    from actfold.models.speculative_core import SpeculativeCore  # noqa: PLC0415

    def _explicit(cands: torch.Tensor, branch_ids: Any = None) -> torch.Tensor:
        return cands

    def _kwargs_passthrough(cands: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return cands

    def _neither(cands: torch.Tensor, attention_mask: Any = None) -> torch.Tensor:
        return cands

    class _ExplicitModule(nn.Module):
        def forward(self, cands: torch.Tensor, branch_ids: Any = None) -> torch.Tensor:
            return cands

    class _KwargsModule(nn.Module):
        def forward(self, cands: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return cands

    class _NoForwardModule(nn.Module):
        pass

    targets = [
        (_explicit, True),
        (_kwargs_passthrough, True),
        (_neither, False),
        (_ExplicitModule(), True),
        (_KwargsModule(), True),
        (_NoForwardModule(), False),
        (object(), False),
        (42, False),
    ]
    for target, expected in targets:
        assert (
            SpeculativeCore._supports_branch_ids(target) is expected
        ), f"SpeculativeCore._supports_branch_ids({target!r}) must be {expected}"
        assert SpeculativeCore._supports_branch_ids(target) == (
            FastDLLMSampler._supports_branch_ids(target)
        ), f"parity with FastDLLMSampler._supports_branch_ids broken for {target!r}"


# ---------------------------------------------------------------------------
# UT-2302: N semantics + batched-call naming
# ---------------------------------------------------------------------------


def test_ut2302_n_semantics_branch_ids() -> None:
    """UT-2302: greedy on keeps rows 1..N-1 identical to greedy off (same seed).

    ``num_block_candidates`` stays the TOTAL row count — the greedy row takes
    slot 0 and the random rows keep ``j`` as their row number in the frozen
    seed formula. The batched call must carry ``branch_ids`` named
    ``[f"{draft}_cand{j}"]`` with ``parent_branch_id == draft_branch_id``,
    ``branch_id is None``, and ``forced_divergent == unit_forced.repeat(N, 1)``.
    """
    factory = _CoreHostFactory("cpu")
    n = 4
    recorders: dict[str, _ForwardRecorder] = {}
    for label, include_greedy in (("on", True), ("off", False)):
        host = factory.host(candidate_include_greedy=include_greedy)
        recorders[label] = _ForwardRecorder(_VOCAB)
        host._speculative_commit(
            x=factory.canvas(),
            finished=torch.zeros(1, dtype=torch.bool),
            draft_logits_region=factory.draft_region(),
            region_start=3,
            masked_local=torch.tensor([0, 1, 2, 3]),
            draft_branch_id="spec_b0",
            ctx=factory.ctx(8),
            unit_forced=torch.tensor([False, False, False, True, True, True, True, True]),
            folded_model=object(),
            unit_idx=1,
            sub_idx=2,
            history=[],
            forward_fn=recorders[label],
            stop_token_id=_EOS_ID,
        )

    cands_on = recorders["on"].calls[0]["tokens"]
    cands_off = recorders["off"].calls[0]["tokens"]
    assert tuple(cands_on.shape) == (n, 8)
    assert torch.equal(
        cands_on[1:], cands_off[1:]
    ), "rows 1..N-1 must be bit-identical to the greedy-off build on the same canvas"
    # Outside the fill region every row is the unchanged canvas.
    assert torch.equal(cands_on[:, :3], factory.canvas().repeat(n, 1)[:, :3])

    kwargs = recorders["on"].calls[0]["kwargs"]
    assert kwargs.get("branch_id") is None, "batched calls pass branch_id=None"
    assert kwargs.get("parent_branch_id") == "spec_b0"
    assert kwargs.get("branch_ids") == [f"spec_b0_cand{j}" for j in range(n)]
    forced = torch.tensor([False, False, False, True, True, True, True, True])
    assert torch.equal(kwargs.get("forced_divergent"), forced.unsqueeze(0).repeat(n, 1))


# ---------------------------------------------------------------------------
# UT-2303: the greedy row competes fairly (posterior trap)
# ---------------------------------------------------------------------------


def test_ut2303_greedy_loses_posterior_trap(device: str) -> None:
    """UT-2303: the greedy row loses the min-prob competition on the LLaDA trap.

    The greedy row fills the draft argmax A but the pinned random row fills C,
    whose posterior self-probability is far higher (``T[C][C]=3.0`` vs
    ``T[A]`` uniform): the unit must select j*==1, commit the winner row, and
    the recomputed min scores must satisfy ``min_scores[0] < min_scores[1]`` —
    constructive evidence that LLaDA-porting never degenerates into
    always-pick-greedy.
    """
    fixture = _LLaDATrapFixture(device, row1_draw=1)
    sampler, out = fixture.sample()

    batched = fixture.batched_calls()
    drafts = fixture.draft_calls()
    assert len(drafts) == 1, "one draft serial forward for the single block unit"
    assert len(batched) == 1, "one [N, T] batched verification call"
    cands = batched[0]["args"][0]
    seq_len = cands.shape[1]
    assert seq_len == _TRAP_PROMPT_LEN + _TRAP_NUM_TOKENS
    fill_lo = _TRAP_PROMPT_LEN
    fill_hi = seq_len
    assert tuple(cands.shape) == (2, seq_len)
    assert int(cands[0, fill_lo]) == _TRAP_GREEDY_ID, "greedy row fills draft argmax A"
    assert int(cands[1, fill_lo]) == _TRAP_ALT_ID, "pinned random row draws C"
    assert int(cands[0, fill_hi - 1]) == _TRAP_GREEDY_ID
    assert int(cands[1, fill_hi - 1]) == _TRAP_ALT_ID

    # Recompute the frozen min-prob scores from the spied batched logits
    # (NO right shift — the LLaDA family convention).
    probs = F.softmax(batched[0]["out"][:, fill_lo:fill_hi, :], dim=-1)
    committed = cands[:, fill_lo:fill_hi]
    scores = torch.gather(probs, -1, committed.unsqueeze(-1)).squeeze(-1)
    min_scores = scores.min(dim=-1).values
    assert float(min_scores[0]) < float(min_scores[1])
    assert int(min_scores.argmax()) == 1, "j* == 1: the greedy row loses"

    assert int(out.sequences[0, fill_lo]) == _TRAP_ALT_ID, "commit == winner row"
    assert int(out.sequences[0, fill_hi - 1]) == _TRAP_ALT_ID
    draft_bid = drafts[0]["kwargs"]["branch_id"]
    assert sampler._folding_parent == f"{draft_bid}_cand1", "chain continues on the winner"
    assert sampler.greedy_win_stats == {"greedy_wins": 0, "speculative_calls": 1}


def test_ut2303_greedy_wins_twin_commit(device: str) -> None:
    """UT-2303: a winning greedy row commits the greedy hypothesis verbatim.

    The same trap fixture with the row-1 draw pinned to A twins the random
    row with the greedy row: the trivial tie selects row 0, the committed
    region equals the greedy hypothesis bit-for-bit, and greedy_wins equals
    speculative_calls.
    """
    fixture = _LLaDATrapFixture(device, row1_draw=0)
    sampler, out = fixture.sample()

    batched = fixture.batched_calls()
    assert len(batched) == 1
    cands = batched[0]["args"][0]
    fill_lo = _TRAP_PROMPT_LEN
    fill_hi = cands.shape[1]
    assert torch.equal(cands[0], cands[1]), "pinned twin row ties the greedy row"
    probs = F.softmax(batched[0]["out"][:, fill_lo:fill_hi, :], dim=-1)
    committed = cands[:, fill_lo:fill_hi]
    scores = torch.gather(probs, -1, committed.unsqueeze(-1)).squeeze(-1)
    assert int(scores.min(dim=-1).values.argmax()) == 0
    assert int(out.sequences[0, fill_lo]) == _TRAP_GREEDY_ID
    assert int(out.sequences[0, fill_hi - 1]) == _TRAP_GREEDY_ID
    assert sampler.greedy_win_stats == {"greedy_wins": 1, "speculative_calls": 1}


# ---------------------------------------------------------------------------
# UT-2304: min-prob selection / commit / chain / finished / shift convention
# ---------------------------------------------------------------------------


def test_ut2304_commit_winner_chain_history() -> None:
    """UT-2304: commit == winner row; chain and history continue on the winner.

    With the batched logits fixed so row 1 wins, the canvas fill positions
    must equal the winner row's tokens, ``_folding_parent`` must switch to
    ``{draft}_cand1``, and the history must gain exactly one snapshot equal
    to the post-commit canvas.
    """
    factory = _CoreHostFactory("cpu")
    host = factory.host()
    x = factory.canvas()
    recorder = _ForwardRecorder(_VOCAB, winner_row=1)
    history: list[torch.Tensor] = [x.clone()]

    host._speculative_commit(
        x=x,
        finished=torch.zeros(1, dtype=torch.bool),
        draft_logits_region=factory.draft_region(),
        region_start=3,
        masked_local=torch.tensor([0, 1, 2, 3]),
        draft_branch_id="spec_b0",
        ctx=factory.ctx(8),
        unit_forced=torch.zeros(8, dtype=torch.bool),
        folded_model=object(),
        unit_idx=0,
        sub_idx=0,
        history=history,
        forward_fn=recorder,
        stop_token_id=_EOS_ID,
    )

    cands = recorder.calls[0]["tokens"]
    assert torch.equal(x[0, 3:7], cands[1, 3:7]), "commit == winner row at fill positions"
    assert torch.equal(x[0, :3], factory.canvas()[0, :3]), "prompt region untouched"
    assert torch.equal(x[0, 7], factory.canvas()[0, 7]), "non-region position untouched"
    assert host._folding_parent == "spec_b0_cand1", "chain continuation onto the winner"
    assert len(history) == 2, "one history snapshot per speculative commit"
    assert torch.equal(history[-1], x)


def test_ut2304_eos_finished_semantics() -> None:
    """UT-2304: eos entering the commit region flips ``finished`` (both faces).

    A pure-greedy N=1 commit whose fill is the eos token must set
    ``finished``; the same construction with a non-eos argmax must leave it
    untouched.
    """
    factory = _CoreHostFactory("cpu")

    def _run(fill_token: int) -> torch.Tensor:
        host = factory.host()
        x = factory.canvas()
        draft = torch.zeros(1, 4, _VOCAB)
        draft[0, :, fill_token] = 3.0
        finished = torch.zeros(1, dtype=torch.bool)
        host._speculative_commit(
            x=x,
            finished=finished,
            draft_logits_region=draft,
            region_start=3,
            masked_local=torch.tensor([0, 1, 2, 3]),
            draft_branch_id="spec_b0",
            ctx=factory.ctx(8),
            unit_forced=torch.zeros(8, dtype=torch.bool),
            folded_model=object(),
            unit_idx=0,
            sub_idx=0,
            history=[],
            forward_fn=_ForwardRecorder(_VOCAB),
            stop_token_id=_EOS_ID,
        )
        return finished

    assert bool(_run(_EOS_ID)[0]), "eos in the commit region must finish the sequence"
    assert not bool(_run(11)[0]), "a non-eos commit must leave finished untouched"


def test_ut2304_min_prob_fill_domain() -> None:
    """UT-2304: the min-prob scoring domain is the FILL position set.

    The region contains one masked (fill) and one already-committed position;
    the crafted batched logits make row 0 win on the fill domain while row 1
    would win a whole-region score — the commit must follow the fill-domain
    winner (design §4.1.1 generalization #1).
    """
    factory = _CoreHostFactory("cpu")
    host = factory.host(num_block_candidates=2)
    x = torch.tensor([[1, 2, 3, _MASK_ID, 9, 7, 7, 7]])
    draft = torch.zeros(1, 2, _VOCAB)
    draft[0, 0, 11] = 3.0  # fill position (local 0): greedy row fills token 11
    draft[0, 0, 12] = 1.0  # runner-up for the random draws

    class _CraftedRecorder(_ForwardRecorder):
        def __call__(self, tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            self.calls.append({"tokens": tokens.detach().clone(), "kwargs": dict(kwargs)})
            logits = torch.zeros(2, tokens.shape[1], _VOCAB)
            logits[0, 3, tokens[0, 3]] = 3.0
            logits[1, 3, tokens[1, 3]] = 1.0
            logits[0, 4, tokens[0, 4]] = -5.0
            logits[1, 4, tokens[1, 4]] = 3.0
            return logits

    recorder = _CraftedRecorder(_VOCAB)
    host._speculative_commit(
        x=x,
        finished=torch.zeros(1, dtype=torch.bool),
        draft_logits_region=draft,
        region_start=3,
        masked_local=torch.tensor([0]),
        draft_branch_id="spec_b0",
        ctx=factory.ctx(8),
        unit_forced=torch.zeros(8, dtype=torch.bool),
        folded_model=object(),
        unit_idx=0,
        sub_idx=0,
        history=[],
        forward_fn=recorder,
        stop_token_id=_EOS_ID,
    )

    cands = recorder.calls[0]["tokens"]
    assert int(cands[0, 3]) == 11, "greedy row fills the draft argmax"
    assert int(x[0, 3]) == 11, "fill-domain scoring selects row 0"
    assert int(x[0, 3]) == int(cands[0, 3]), "commit == fill-domain winner row"

    # The selection rule is MIN over the fill positions (the AR022 frozen
    # rule): a row with two high-prob and two low-prob fills must LOSE to a
    # uniform-medium row — a max-over-fill mutant picks the high/low row.
    class _MinMaxRecorder(_ForwardRecorder):
        def __call__(self, tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            self.calls.append({"tokens": tokens.detach().clone(), "kwargs": dict(kwargs)})
            logits = torch.zeros(2, tokens.shape[1], _VOCAB)
            logits[0, 3, tokens[0, 3]] = 4.0
            logits[0, 4, tokens[0, 4]] = 4.0
            logits[0, 5, tokens[0, 5]] = -4.0
            logits[0, 6, tokens[0, 6]] = -4.0
            for pos in (3, 4, 5, 6):
                logits[1, pos, tokens[1, pos]] = 1.0
            return logits

    host_min = factory.host(num_block_candidates=2)
    x_min = factory.canvas()
    recorder_min = _MinMaxRecorder(_VOCAB)
    host_min._speculative_commit(
        x=x_min,
        finished=torch.zeros(1, dtype=torch.bool),
        draft_logits_region=factory.draft_region(),
        region_start=3,
        masked_local=torch.tensor([0, 1, 2, 3]),
        draft_branch_id="spec_b0",
        ctx=factory.ctx(8),
        unit_forced=torch.zeros(8, dtype=torch.bool),
        folded_model=object(),
        unit_idx=0,
        sub_idx=0,
        history=[],
        forward_fn=recorder_min,
        stop_token_id=_EOS_ID,
    )
    cands_min = recorder_min.calls[0]["tokens"]
    assert torch.equal(
        x_min[0, 3:7], cands_min[1, 3:7]
    ), "min over the fill positions selects the uniform-medium row"


def test_ut2304_llada_no_shift_convention(device: str) -> None:
    """UT-2304: LLaDA feeds Core the UNshifted logits (position i predicts i).

    On the position-table head the unshifted argmax at position p is
    ``(p+1) % V`` while a right-shifted argmax would be ``p % V``; the
    committed block must equal the unshifted fill and differ from the
    shifted one (design §4.1.1 generalization #2, LLaDA face).
    """
    inner = _build_mini(device, head="postable")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _llada_config(
        seed=2304,
        num_steps=4,
        num_tokens=4,
        block_size=4,
        num_block_candidates=1,
        candidate_top_k=2,
        candidate_include_greedy=True,
    )
    mff = _make_mff(inner, device)
    sampler = LLaDASampler(llm, config=cfg)
    out = sampler.sample(prompt, folded_model=mff)

    drafts = [c for c in mff.calls if c["kwargs"].get("branch_ids") is None]
    batched = [c for c in mff.calls if c["kwargs"].get("branch_ids") is not None]
    assert len(drafts) == 1 and len(batched) == 1
    fill_lo, fill_hi = 3, 7
    unshifted = torch.topk(drafts[0]["out"][:, fill_lo:fill_hi, :], 2, dim=-1).indices[..., 0]
    shifted = torch.topk(
        torch.cat([drafts[0]["out"][:, :1, :], drafts[0]["out"][:, :-1, :]], dim=1)[
            :, fill_lo:fill_hi, :
        ],
        2,
        dim=-1,
    ).indices[..., 0]
    assert torch.equal(
        out.sequences[0, fill_lo:fill_hi], unshifted[0]
    ), "commit == the unshifted draft argmax (LLaDA: no right shift)"
    assert not torch.equal(
        out.sequences[0, fill_lo:fill_hi], shifted[0]
    ), "the commit must NOT follow a right-shifted argmax (Dream convention)"


# ---------------------------------------------------------------------------
# UT-2305: greedy_win_stats probe
# ---------------------------------------------------------------------------


def test_ut2305_greedy_win_stats_probe() -> None:
    """UT-2305: read-only, default {0,0}, cumulative, greedy-path-only counts.

    The probe dict is read-only (assignment raises AttributeError; external
    mutation cannot corrupt the counters), starts at ``{0, 0}``, increments
    ``speculative_calls`` on every greedy-anchored commit and
    ``greedy_wins`` only when j*==0, and stays at ``{0, 0}`` when greedy
    anchoring is off.
    """
    factory = _CoreHostFactory("cpu")
    host = factory.host()
    assert host.greedy_win_stats == {"greedy_wins": 0, "speculative_calls": 0}

    with pytest.raises(AttributeError):
        host.greedy_win_stats = {"greedy_wins": 99, "speculative_calls": 99}  # type: ignore[misc]

    def _commit(host: Any, recorder: _ForwardRecorder) -> None:
        host._speculative_commit(
            x=factory.canvas(),
            finished=torch.zeros(1, dtype=torch.bool),
            draft_logits_region=factory.draft_region(),
            region_start=3,
            masked_local=torch.tensor([0, 1, 2, 3]),
            draft_branch_id="spec_b0",
            ctx=factory.ctx(8),
            unit_forced=torch.zeros(8, dtype=torch.bool),
            folded_model=object(),
            unit_idx=0,
            sub_idx=0,
            history=[],
            forward_fn=recorder,
            stop_token_id=_EOS_ID,
        )

    # Row 0 wins (uniform crafted logits tie -> argmax picks row 0).
    _commit(host, _ForwardRecorder(_VOCAB, winner_row=0))
    assert host.greedy_win_stats == {"greedy_wins": 1, "speculative_calls": 1}
    # Row 1 wins: calls grow, wins do not.
    _commit(host, _ForwardRecorder(_VOCAB, winner_row=1))
    assert host.greedy_win_stats == {"greedy_wins": 1, "speculative_calls": 2}

    # External mutation of the returned dict must not leak inside.
    probe = host.greedy_win_stats
    probe["greedy_wins"] = -1
    probe["speculative_calls"] = -1
    assert host.greedy_win_stats == {"greedy_wins": 1, "speculative_calls": 2}

    # Greedy-off control: both counters stay 0.
    host_off = factory.host(candidate_include_greedy=False)
    _commit(host_off, _ForwardRecorder(_VOCAB, winner_row=0))
    assert host_off.greedy_win_stats == {"greedy_wins": 0, "speculative_calls": 0}


# ---------------------------------------------------------------------------
# UT-2306: default-off zero disturbance
# ---------------------------------------------------------------------------


def test_ut2306_default_zero_disturbance(device: str) -> None:
    """UT-2306: N=0 explicit off == default bit-exact on raw and folded faces.

    ``num_block_candidates=0`` (the default) must keep the reference LLaDA
    recipe byte-identical: default vs explicit ``num_block_candidates=0`` /
    ``candidate_include_greedy=False`` on the raw path and the N=0 folded
    path, plus the stop-token scenario; the default ``greedy_win_stats``
    probe must read ``{0, 0}`` on both faces.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)

    out_default = LLaDASampler(llm, config=_llada_config(seed=2306)).sample(prompt)
    cfg_off = _llada_config(seed=2306, num_block_candidates=0, candidate_include_greedy=False)
    out_off = LLaDASampler(llm, config=cfg_off).sample(prompt)
    assert torch.equal(out_off.sequences, out_default.sequences)

    mff_default = _make_mff(inner, device)
    out_folded_default = LLaDASampler(llm, config=_llada_config(seed=2306)).sample(
        prompt, folded_model=mff_default
    )
    mff_off = _make_mff(inner, device)
    out_folded_off = LLaDASampler(llm, config=cfg_off).sample(prompt, folded_model=mff_off)
    assert torch.equal(out_folded_off.sequences, out_folded_default.sequences)

    inner_stop = _build_mini(device, head="constant_eos")
    llm_stop = _MiniDiffusionLLM(inner_stop)
    out_stop_default = LLaDASampler(llm_stop, config=_llada_config(seed=2306)).sample(prompt)
    cfg_stop_off = _llada_config(seed=2306, num_block_candidates=0, candidate_include_greedy=False)
    out_stop_off = LLaDASampler(llm_stop, config=cfg_stop_off).sample(prompt)
    assert torch.equal(out_stop_off.sequences, out_stop_default.sequences)

    sampler_off = LLaDASampler(llm, config=cfg_off)
    sampler_off.sample(prompt)
    assert sampler_off.greedy_win_stats == {"greedy_wins": 0, "speculative_calls": 0}
    sampler_folded_off = LLaDASampler(llm, config=cfg_off)
    sampler_folded_off.sample(prompt, folded_model=_make_mff(inner, device))
    assert sampler_folded_off.greedy_win_stats == {
        "greedy_wins": 0,
        "speculative_calls": 0,
    }


# ---------------------------------------------------------------------------
# UT-2307: LLaDA block speculative unit
# ---------------------------------------------------------------------------


def test_ut2307_block_unit_flow(device: str) -> None:
    """UT-2307: one draft + one [N, T] batched call commits a whole block.

    The reference recipe runs ``steps_per_block`` remasking iterations; the
    speculative path must make exactly 2 folded forwards per block (1 serial
    draft + 1 ``[N, T]`` batched verification), carry the frozen branch
    naming, and pass the block-superset ``forced_divergent`` (True over the
    whole block window, False outside).
    """
    inner = _build_mini(device, head="spike")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)

    # Reference forward count: steps_per_block serial raw forwards.
    cfg_ref = _llada_config(seed=2307, num_steps=8, num_tokens=8, block_size=8)
    llm.raw_calls.clear()
    LLaDASampler(llm, config=cfg_ref).sample(prompt)
    reference_forwards = len(llm.raw_calls)
    assert reference_forwards == 8, "reference recipe: 8 remasking iterations"

    cfg = _llada_config(
        seed=2307,
        num_steps=8,
        num_tokens=8,
        block_size=8,
        num_block_candidates=4,
        candidate_top_k=2,
        candidate_include_greedy=True,
    )
    mff = _make_mff(inner, device)
    sampler = LLaDASampler(llm, config=cfg)
    out = sampler.sample(prompt, folded_model=mff)

    drafts = [c for c in mff.calls if c["kwargs"].get("branch_ids") is None]
    batched = [c for c in mff.calls if c["kwargs"].get("branch_ids") is not None]
    assert len(drafts) == 1, "one draft serial forward per block"
    assert len(batched) == 1, "one [N, T] batched verification per block"
    assert (
        len(mff.calls) == 2 < reference_forwards
    ), "the speculative commit must skip the block's remaining iterations"

    cands = batched[0]["args"][0]
    n, seq_len = cands.shape
    assert (n, seq_len) == (4, 11)
    draft_bid = drafts[0]["kwargs"]["branch_id"]
    assert batched[0]["kwargs"]["parent_branch_id"] == draft_bid
    assert batched[0]["kwargs"]["branch_ids"] == [f"{draft_bid}_cand{j}" for j in range(n)]

    # Row 0 == the still-masked draft argmax fill (spike head: token 5).
    expected_row0 = torch.topk(drafts[0]["out"][:, 3:11, :], 2, dim=-1).indices[..., 0]
    assert torch.equal(cands[0, 3:11], expected_row0[0])
    assert (out.sequences[0, 3:11] != _MASK_ID).all(), "the whole block is committed"

    # forced_divergent: the block-region superset ([start, end) all True).
    forced = batched[0]["kwargs"]["forced_divergent"]
    assert tuple(forced.shape) == (n, seq_len)
    assert bool(forced[:, 3:11].all()), "forced covers the whole block window"
    assert int(forced[:, :3].sum()) == 0, "forced is False outside the block window"

    assert sampler.greedy_win_stats["speculative_calls"] == 1


def test_ut2307_entry_validation(device: str) -> None:
    """UT-2307: N>0 entry preconditions raise ValueError (never silent).

    Without a ``folded_model``, with a non-``branch_ids``-capable folded
    model, and with batch > 1 — all construction/entry ValueErrors.
    """
    inner = _build_mini(device, head="spike")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _llada_config(
        seed=2307, num_block_candidates=2, candidate_top_k=2, candidate_include_greedy=True
    )

    with pytest.raises(ValueError, match=r"folded_model"):
        LLaDASampler(llm, config=cfg).sample(prompt)

    class _NoBranchIdsCallable:
        def __call__(self, tokens: torch.Tensor, attention_mask: Any = None) -> torch.Tensor:
            return torch.zeros(tokens.shape[0], tokens.shape[1], _VOCAB)

    with pytest.raises(ValueError, match=r"branch_ids"):
        LLaDASampler(llm, config=cfg).sample(prompt, folded_model=_NoBranchIdsCallable())

    prompt_b2 = torch.cat([prompt, prompt], dim=0)
    with pytest.raises(ValueError):
        LLaDASampler(llm, config=cfg).sample(prompt_b2, folded_model=_make_mff(inner, device))


def test_ut2307_no_masked_unit_skip(device: str) -> None:
    """UT-2307: units without masked positions are skipped without a forward.

    A zero-token canvas makes no forward at all; a constant-eos head finishes
    the first block so the second block is skipped (exactly 2 folded calls:
    1 draft + 1 batched), leaving the second block's positions masked.
    """
    inner = _build_mini(device, head="spike")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)

    cfg_empty = _llada_config(
        seed=2307,
        num_steps=2,
        num_tokens=0,
        block_size=4,
        num_block_candidates=2,
        candidate_top_k=1,
        candidate_include_greedy=True,
    )
    mff_empty = _make_mff(inner, device)
    LLaDASampler(llm, config=cfg_empty).sample(prompt, folded_model=mff_empty)
    assert len(mff_empty.calls) == 0, "no masked positions: no forward at all"

    inner_stop = _build_mini(device, head="constant_eos")
    llm_stop = _MiniDiffusionLLM(inner_stop)
    cfg_stop = _llada_config(
        seed=2307,
        num_steps=4,
        num_tokens=4,
        block_size=2,
        num_block_candidates=2,
        candidate_top_k=1,
        candidate_include_greedy=True,
    )
    mff_stop = _make_mff(inner_stop, device)
    out = LLaDASampler(llm_stop, config=cfg_stop).sample(prompt, folded_model=mff_stop)
    assert len(mff_stop.calls) == 2, "block 0 commits eos -> block 1 skipped"
    assert bool((out.sequences[0, 3:5] == _EOS_ID).all()), "block 0 committed eos"
    assert bool((out.sequences[0, 5:7] == _MASK_ID).all()), "block 1 stays masked"


def test_ut2307_forced_block_superset(device: str) -> None:
    """UT-2307: ``_build_unit_forced`` returns the block-region superset.

    The [T] bool forced tensor must be True over the whole block window
    ``[region_start, region_end)`` (the conservative superset, AR021 P5
    reasoning) and False outside it.
    """
    llm = _MiniDiffusionLLM(_build_mini(device))
    sampler = LLaDASampler(llm, config=_llada_config())
    x = torch.tensor([[1, 2, 3, _MASK_ID, _MASK_ID, _MASK_ID, _MASK_ID, 7]], device=device)
    forced = sampler._build_unit_forced(
        x, region_start=3, region_end=7, masked_local=torch.tensor([0, 1, 2, 3])
    )
    assert forced.dtype == torch.bool and tuple(forced.shape) == (8,)
    assert bool(forced[3:7].all()), "forced is the block-region superset"
    assert int(forced[:3].sum()) == 0 and int(forced[7:].sum()) == 0


# ---------------------------------------------------------------------------
# UT-2308: LLaDA fold context (ctx pinning + LRU cap)
# ---------------------------------------------------------------------------


def test_ut2308_ctx_mask_semantics_pinning(device: str) -> None:
    """UT-2308: ``_spec_ctx_for`` object pinning and pad-column mask semantics.

    The context is keyed by ``(T, prompt_len)``: the same key returns the
    SAME ``SpecCtx`` object; a different ``prompt_len`` at the same T is a
    different object with different pad content. The ``[T, T]`` bool mask
    must be element-equal to the 1-D reference mask
    (``arange(T) < prompt_len + num_tokens``) tiled over rows, and the
    LLaDA position payload is ``None`` (no rotary cargo on the mini model).
    """
    llm = _MiniDiffusionLLM(_build_mini(device))
    sampler = LLaDASampler(llm, config=_llada_config(num_tokens=4))
    # Manual canvas with a pad tail: prompt 3 + generation 4 + eos-pad 2.
    x = torch.tensor(
        [[1, 2, 3, _MASK_ID, _MASK_ID, _MASK_ID, _MASK_ID, _EOS_ID, _EOS_ID]], device=device
    )

    ctx_a = sampler._spec_ctx_for(x, prompt_len=3)
    ctx_b = sampler._spec_ctx_for(x, prompt_len=3)
    assert ctx_a is ctx_b, "same (T, prompt_len) key -> the same pinned ctx object"

    seq_len = x.shape[1]
    valid_end = min(3 + 4, seq_len)
    reference_1d = torch.arange(seq_len, device=device) < valid_end
    assert tuple(ctx_a.mask.shape) == (seq_len, seq_len)
    assert ctx_a.mask.dtype == torch.bool
    assert torch.equal(ctx_a.mask, reference_1d.unsqueeze(0).expand(seq_len, seq_len)), (
        "the [T, T] mask is the 1-D reference mask tiled over rows "
        "(pad columns excluded everywhere)"
    )
    assert ctx_a.position is None, "LLaDA carries no position payload on the mini model"

    # Same T, different prompt_len: a different key, object, and pad content.
    x_other = torch.tensor(
        [[1, 2, 3, 4, _MASK_ID, _MASK_ID, _MASK_ID, _MASK_ID, _EOS_ID]], device=device
    )
    ctx_other = sampler._spec_ctx_for(x_other, prompt_len=4)
    assert ctx_other is not ctx_a
    assert not torch.equal(ctx_other.mask, ctx_a.mask)


def test_ut2308_lru_eviction_cap32(device: str) -> None:
    """UT-2308: the ctx LRU cache holds at most 32 entries (put-refresh).

    Filling 32 keys, re-putting the oldest (refresh), then inserting a 33rd
    key must evict the SECOND-oldest entry (not the refreshed one); the cap
    parity-checks ``_MAX_FOLD_CTX_ENTRIES == 32``
    (fast_dllm_sampler.py:52).
    """
    from actfold.models.speculative_core import SpecCtx  # noqa: PLC0415

    assert _MAX_FOLD_CTX_ENTRIES == 32, "Core LRU cap parity with _MAX_FOLD_CTX_ENTRIES"

    llm = _MiniDiffusionLLM(_build_mini(device))
    sampler = LLaDASampler(llm, config=_llada_config())
    ctx_of = lambda key: SpecCtx(  # noqa: E731
        mask=torch.ones(4, 4, dtype=torch.bool, device=device), position=None
    )
    for i in range(32):
        sampler._ctx_cache_put((i, 0), ctx_of((i, 0)))
    sampler._ctx_cache_put((0, 0), ctx_of((0, 0)))  # refresh the oldest
    sampler._ctx_cache_put((32, 0), ctx_of((32, 0)))  # 33rd entry -> evict key 1

    assert sampler._ctx_cache_get((0, 0)) is not None, "refreshed key survives"
    assert sampler._ctx_cache_get((1, 0)) is None, "the stalest entry was evicted"
    assert sampler._ctx_cache_get((32, 0)) is not None, "the newest entry is cached"
    assert sampler._ctx_cache_get((31, 0)) is not None
