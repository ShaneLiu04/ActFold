"""Red→Green tests for AR025 T006: W2 lockstep joint speculative sampling (UT-2430..UT-2445).

Contract: ``specs/changes/AR025-multi-parent-batch-folding/design.md`` DD8a
(W2 lockstep joint loop ``sample_joint``) — ``LLaDASampler`` and
``DreamSampler`` each gain::

    def sample_joint(self, prompt_ids_list, folded_model=None) -> list[SamplerOutput]

A SINGLE-threaded lockstep joint loop with ONE sampler instance driving BOTH
chains (per-chain canvas/finished/history state + per-chain folding-chain
state). Per window: per-chain draft serial forward ([1, T] each) -> per-chain
candidate construction (the frozen seed formula, SAME unit_idx/sub_idx per
chain — no chain component in the seed) -> ONE joint [2N, T] batched
verification forward with per-row parents -> per-chain min-prob score/commit:

* UT-2430: LLaDA parity — ``sample_joint([p1, p2], mff_joint)`` (mff with
  ``multi_parent_batch=True``) vs B-side ``sample(p1, mff_solo)`` +
  ``sample(p2, mff_solo)`` back-to-back on a same-config sampler: per-chain
  output tokens BITWISE equal (torch.equal) with two different equal-length
  prompts, plus per-chain history parity with ``return_history=True``;
* UT-2431: ledger — the ``write_op_count`` delta on the joint cache is
  STRICTLY SMALLER than the solo cache delta (the joint [2N, T] publish is
  one put_batch group per layer vs two — #42 semantics);
* UT-2432: early-finish mixed (C5) — chain 1 commits EOS in the first window
  (prompt-keyed table head: the table row keyed on prompt token 0 emits EOS)
  and is removed from the joint batch; chain 2 completes normally and its
  tokens equal the solo-run reference bitwise; the batched-call rows shrink
  2N -> N with no crash;
* UT-2433: off-switch dispatch — mff WITHOUT ``multi_parent_batch`` (default
  False) makes ``sample_joint`` semantically equal to
  ``[sample(p1), sample(p2)]`` (it literally dispatches per chain, DD7-
  isomorphic);
* UT-2434: F3 entry ValueErrors — ``temperature > 0`` and
  ``stochastic_transfer=True`` raise ValueError on the joint path ONLY (the
  greedy-path determinism mirror of the ``cfg_scale`` rejection style; the
  off-switch dispatch of the same configs must not raise);
* UT-2435: C2/shape entry ValueErrors — empty ``prompt_ids_list``, a prompt
  with batch 2, prompts of unequal token length, and ``cfg_scale > 0`` all
  raise ValueError on the joint path;
* UT-2436: branch-structure spy — every batched joint call carries
  ``parent_branch_ids`` (plural) with ``branch_id=None``; NO call carries
  both singular and plural parents; joint rows are 2N (N after one chain
  finished) with ``parent_branch_ids == [p1]*N + [p2]*N`` where p1/p2 are
  the chains' OWN draft branch ids; branch ids are chain-qualified (the two
  chains' id sets never collide);
* UT-2437: ``greedy_win_stats`` AGGREGATES across both chains on the joint
  path (single instance drives both; 1 + 2 per-chain windows -> 3 counts on
  the early-finish fixture, > 1 and moved vs before);
* UT-2438: LLaDA joint + ``progressive_refill`` parity — the AR024 two-phase
  refill inside the joint loop (segment B partial commit + the chain-scoped
  ``_r`` refill tag + the F1 instance-slot swap) equals back-to-back solo
  bitwise, histories included (the T009 coverage-gap fix);
* UT-2440: Dream parity (DreamSampler, two equal-length prompts,
  ``speculative_steps`` default; the Dream ctx carries position_ids);
* UT-2441: Dream off-switch dispatch;
* UT-2442: Dream F3 ValueErrors (temperature>0 AND stochastic_transfer=True);
* UT-2443: Dream ledger strictly smaller;
* UT-2444: composed-path zero-drift guard — one solo LLaDA and one solo
  Dream speculative ``sample()`` smoke each (the existing composed
  single-chain ``_speculative_commit`` path must be unchanged by the
  refactor; outputs are tensors of the expected shape with the
  finished/decoded semantics intact);
* UT-2445: Dream C2/shape entry ValueErrors — an empty ``prompt_ids_list``,
  a prompt with batch 2, prompts of unequal token length, and
  ``cfg_scale > 0`` all raise on the Dream joint path (the UT-2435 mirror,
  the T009 coverage-gap fix).

Red state (why every test fails today) — each failure is a production-missing
face, never a test-own bug:

* ``LLaDASampler`` and ``DreamSampler`` have no ``sample_joint`` method —
  every test fails with ``AttributeError`` (or a DID-NOT-RAISE inside
  ``pytest.raises``) at the first ``sample_joint`` call; all fixtures (the
  mini bidirectional decoder, the MFF spy, the prompt-keyed EOS trap) run
  green through the existing ``sample()``/MFF faces.

Fixtures follow ``tests/test_llada_speculative.py`` and
``tests/test_dream_speculative.py`` (self-contained LLaMA-layout mini
bidirectional decoder ~2 layers H=32 V=64, mini tokenizer, the cloning
``_SpyMFF``, ``_make_mff`` — here extended with the AR025
``multi_parent_batch`` ctor switch — and the Hadamard-basis trap pattern for
the deterministic EOS face).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from actfold.core import ActivationCache, SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.models.base import DiffusionLLM
from actfold.models.diffusion_sampler import SamplerOutput
from actfold.models.dream_sampler import DreamSampler, DreamSamplerConfig
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

# Prompt-keyed EOS-trap constants (UT-2432/UT-2436/UT-2437): trap vocab 16
# over hidden 32 so 16 pairwise-orthogonal zero-mean basis rows exist.
_EOS_TRAP_VOCAB = 16
_EOS_TRAP_MASK_ID = 15
_EOS_TRAP_EOS_ID = 14
_EOS_TRAP_PAD_ID = 13
_EOS_TRAP_KEY = 1  # prompt[0] keying the EOS-emitting table row
_PLAIN_TRAP_KEY = 2  # prompt[0] keying the plain (non-EOS) table row
_EOS_TRAP_ALT = 5  # runner-up in the EOS row (the eos-aware candidate fill)
_PLAIN_TRAP_TOP = 6  # top-1 in the plain row (never EOS -> chain completes)
_PLAIN_TRAP_ALT = 7  # runner-up in the plain row
_TRAP_PROMPT_LEN = 3
_TRAP_NUM_TOKENS = 8  # two 4-token windows (block_size 4, sub-block 4)
_TRAP_SEQ_LEN = _TRAP_PROMPT_LEN + _TRAP_NUM_TOKENS


@pytest.fixture(autouse=True)
def _profiler_state() -> Iterator[None]:
    """Snapshot-reset the global stability profiler around every case."""
    GLOBAL_STABILITY_PROFILER.reset()
    yield
    GLOBAL_STABILITY_PROFILER.reset()


# ---------------------------------------------------------------------------
# Self-contained mini bidirectional decoder (T018 fixture family, copied from
# tests/test_llada_speculative.py / tests/test_dream_speculative.py)
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


class _PosTableHead(nn.Module):
    """Position-table head: top-1 at position p is ``(p+1) % V`` (graded).

    ``logits[:, p, (p+1) % V] = 3.0 + 0.01*p`` and
    ``logits[:, p, (p+2) % V] = 2.0 + 0.01*p`` — input-independent, so the
    unshifted argmax at p is ``(p+1) % V`` while a right-shifted argmax at p
    would be ``p % V`` (the shift-convention discriminator).
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
        if head == "postable":
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
    """ManualFoldedForward that records every forward call and its output.

    Tensor args are CLONED at record time: the samplers mutate the canvas in
    place across windows, so by-reference recording would retroactively show
    post-commit state (the call-time snapshot is the spy's contract).
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
# Prompt-keyed EOS trap (UT-2432/UT-2436/UT-2437; the _TrapModel pattern from
# tests/test_llada_speculative.py with the table keyed on prompt token 0)
# ---------------------------------------------------------------------------


def _hadamard_basis(vocab: int, hidden: int) -> torch.Tensor:
    """Zero-mean unit-variance pairwise-orthogonal rows (Sylvester Hadamard)."""
    rows = []
    for v in range(vocab):
        i = v + 1
        rows.append([-1.0 if bin(i & h).count("1") % 2 else 1.0 for h in range(hidden)])
    return torch.tensor(rows, dtype=torch.float32)


def _prompt_keyed_table() -> torch.Tensor:
    """Table ``T`` keyed on the FIRST prompt token.

    ``T[_EOS_TRAP_KEY]`` tops out at EOS (3.0 vs 1.0) so every candidate row
    keyed on it either fills EOS (the greedy row — the min-prob winner, the
    high self-probability fill) or a low-probability runner-up: the commit
    always contains EOS and the chain finishes in the first window.
    ``T[_PLAIN_TRAP_KEY]`` tops out at a plain token and its whole top-2 pool
    is EOS-free, so that chain can never commit EOS and completes normally.
    """
    table = torch.zeros(_EOS_TRAP_VOCAB, _EOS_TRAP_VOCAB)
    table[_EOS_TRAP_KEY, _EOS_TRAP_EOS_ID] = 3.0
    table[_EOS_TRAP_KEY, _EOS_TRAP_ALT] = 1.0
    table[_PLAIN_TRAP_KEY, _PLAIN_TRAP_TOP] = 3.0
    table[_PLAIN_TRAP_KEY, _PLAIN_TRAP_ALT] = 1.0
    return table


class _PromptKeyedTableHead(nn.Module):
    """Direct table head keyed on position 0: ``logits[:, p] = T[input[:, 0]]``.

    ``W = sum_u outer(T[u], basis[u]) / H`` over the orthogonal basis makes
    ``W @ t_in == T[in]`` exactly, so the logits at EVERY position are the
    table row of the FIRST prompt token — per-ROW in a batch, hence per-CHAIN
    in the joint [2N, T] verification batch (each row keeps its own chain's
    prompt at position 0).
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
        key = x[:, 0:1, :]
        return self.proj(key).expand(-1, x.shape[1], -1)


class _PromptKeyedTrapModel(nn.Module):
    """LLaMA-layout zero-mixing decoder keyed on the first prompt token.

    (i) Zero-mixing layers (attention and MLP contribute exactly zero — the
    residual chain carries the embedding through), (ii) orthogonal Hadamard
    embeddings (``LayerNorm(embed(v)) == t_v``), (iii) the prompt-keyed table
    head — so the logits are fully controlled by each row's own prompt[0]
    even inside a shared joint batch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llada", "vocab_size": _EOS_TRAP_VOCAB})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(_EOS_TRAP_VOCAB, _HIDDEN)
        self.model.layers = nn.ModuleList([_BidirectionalLayer(_HIDDEN) for _ in range(2)])
        self.model.norm = nn.LayerNorm(_HIDDEN)
        basis = _hadamard_basis(_EOS_TRAP_VOCAB, _HIDDEN)
        self.lm_head = _PromptKeyedTableHead(_prompt_keyed_table(), basis)
        with torch.no_grad():
            for layer in self.model.layers:
                for param in layer.self_attn.parameters():
                    param.zero_()
                for param in layer.mlp.parameters():
                    param.zero_()
            for v in range(_EOS_TRAP_VOCAB):
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
    multi_parent_batch: bool = False,
) -> ManualFoldedForward:
    """Fresh ManualFoldedForward (own cache + cosine gate) over the model.

    ``multi_parent_batch`` threads the AR025 T002 ctor switch through: True
    arms the joint path (per-row parents), False (the default) keeps the
    per-chain dispatch semantics.
    """
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
        multi_parent_batch=multi_parent_batch,
    )


def _llada_config(**overrides: Any) -> LLaDASamplerConfig:
    """LLaDASamplerConfig with pinned defaults (AR023/AR024 fields via overrides)."""
    values: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 8,
        "temperature": 0.0,
        "seed": 2430,
        "block_size": _BLOCK_SIZE,
    }
    values.update(overrides)
    return LLaDASamplerConfig(**values)


def _dream_config(**overrides: Any) -> DreamSamplerConfig:
    """DreamSamplerConfig with pinned defaults (AR023 fields via overrides)."""
    values: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 8,
        "temperature": 0.0,
        "seed": 2440,
    }
    values.update(overrides)
    return DreamSamplerConfig(**values)


def _prompts2(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Two DIFFERENT prompts of equal token length (the joint-path shape)."""
    return (
        torch.tensor([[1, 2, 3]], device=device),
        torch.tensor([[4, 5, 6]], device=device),
    )


def _split_calls(
    mff: _SpyMFF,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split spied MFF calls into (serial draft, batched verification)."""
    drafts = [c for c in mff.calls if c["kwargs"].get("branch_ids") is None]
    batched = [c for c in mff.calls if c["kwargs"].get("branch_ids") is not None]
    return drafts, batched


def _chain_of_canvas(
    tokens: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, prompt_len: int
) -> int:
    """Attribute a spied [1, T] canvas to a chain by its (never-masked) prompt region."""
    if torch.equal(tokens[0, :prompt_len], p1[0]):
        return 0
    if torch.equal(tokens[0, :prompt_len], p2[0]):
        return 1
    raise AssertionError("spied canvas matches neither fixture prompt (fixture bug)")


def _eos_trap_llada_config(seed: int) -> LLaDASamplerConfig:
    """Two-window EOS-trap config: N=2 (greedy row + eos-aware row, both
    deterministic — no seed-formula draws), 4-token windows over 8 tokens."""
    return _llada_config(
        seed=seed,
        num_steps=2,
        num_tokens=_TRAP_NUM_TOKENS,
        block_size=4,
        speculative_sub_block_size=4,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        candidate_eos_aware=True,
    )


def _eos_trap_prompts(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Chain 1 keys the EOS table row (prompt[0]==1), chain 2 the plain row (==2)."""
    return (
        torch.tensor([[_EOS_TRAP_KEY, 3, 4]], device=device),
        torch.tensor([[_PLAIN_TRAP_KEY, 3, 4]], device=device),
    )


# ---------------------------------------------------------------------------
# UT-2430: LLaDA joint parity (bitwise, both chains, history included)
# ---------------------------------------------------------------------------


def test_ut2430_llada_joint_parity_bitwise(device: str) -> None:
    """UT-2430: joint [p1, p2] equals back-to-back solo runs bit-for-bit.

    A-side: one ``sample_joint`` call on an mff with ``multi_parent_batch=True``
    driving both chains. B-side: ``sample(p1)`` + ``sample(p2)`` back-to-back
    on a separately-constructed identical (switch-on) mff and a same-config
    sampler. The frozen per-candidate seed formula has NO chain component
    (same seed/unit_idx/sub_idx/j per chain), so per-chain outputs must be
    BITWISE equal (torch.equal, never allclose) — including the history
    snapshots with ``return_history=True``.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    p1, p2 = _prompts2(device)
    cfg_kwargs: dict[str, Any] = {
        "num_steps": 4,
        "num_tokens": 8,
        "block_size": 4,
        "speculative_sub_block_size": 4,
        "num_block_candidates": 3,
        "candidate_top_k": 3,
        "candidate_include_greedy": True,
        "candidate_eos_aware": True,
        "seed": 2430,
        "return_history": True,
    }

    mff_joint = _make_mff(inner, device, multi_parent_batch=True)
    sampler_a = LLaDASampler(llm, config=_llada_config(**cfg_kwargs))
    outs = sampler_a.sample_joint([p1, p2], mff_joint)

    mff_solo = _make_mff(inner, device, multi_parent_batch=True)
    sampler_b = LLaDASampler(llm, config=_llada_config(**cfg_kwargs))
    out1 = sampler_b.sample(p1, folded_model=mff_solo)
    out2 = sampler_b.sample(p2, folded_model=mff_solo)

    assert isinstance(outs, list) and len(outs) == 2, "sample_joint returns per-chain outputs"
    assert isinstance(outs[0], SamplerOutput) and isinstance(outs[1], SamplerOutput)
    assert torch.equal(
        outs[0].sequences, out1.sequences
    ), "chain 1 tokens must be bitwise equal to the solo run"
    assert torch.equal(
        outs[1].sequences, out2.sequences
    ), "chain 2 tokens must be bitwise equal to the solo run"

    assert len(outs[0].history) == len(out1.history), "chain 1 history length parity"
    assert len(outs[1].history) == len(out2.history), "chain 2 history length parity"
    for joint_snap, solo_snap in zip(outs[0].history, out1.history):
        assert torch.equal(joint_snap, solo_snap), "chain 1 history snapshots bitwise equal"
    for joint_snap, solo_snap in zip(outs[1].history, out2.history):
        assert torch.equal(joint_snap, solo_snap), "chain 2 history snapshots bitwise equal"


# ---------------------------------------------------------------------------
# UT-2431: LLaDA joint write-op ledger strictly smaller
# ---------------------------------------------------------------------------


def test_ut2431_llada_joint_ledger_strictly_smaller(device: str) -> None:
    """UT-2431: the joint cache write ledger is STRICTLY smaller than solo.

    Both deltas are measured around the full sampling on separately
    constructed identical (switch-on) mffs. The joint [2N, T] verification
    publish is ONE put_batch group per layer covering both chains, while the
    solo side publishes two [N, T] batches — #42 semantics — so the joint
    delta must be strictly smaller (and both deltas positive: real writes).
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    p1, p2 = _prompts2(device)
    cfg_kwargs: dict[str, Any] = {
        "num_steps": 4,
        "num_tokens": 8,
        "block_size": 4,
        "speculative_sub_block_size": 4,
        "num_block_candidates": 3,
        "candidate_top_k": 3,
        "candidate_include_greedy": True,
        "candidate_eos_aware": True,
        "seed": 2431,
    }

    mff_joint = _make_mff(inner, device, multi_parent_batch=True)
    sampler_a = LLaDASampler(llm, config=_llada_config(**cfg_kwargs))
    before_joint = mff_joint.cache.write_op_count
    sampler_a.sample_joint([p1, p2], mff_joint)
    joint_delta = mff_joint.cache.write_op_count - before_joint

    mff_solo = _make_mff(inner, device, multi_parent_batch=True)
    sampler_b = LLaDASampler(llm, config=_llada_config(**cfg_kwargs))
    before_solo = mff_solo.cache.write_op_count
    sampler_b.sample(p1, folded_model=mff_solo)
    sampler_b.sample(p2, folded_model=mff_solo)
    solo_delta = mff_solo.cache.write_op_count - before_solo

    assert joint_delta > 0, "the joint path performs real cache writes"
    assert solo_delta > 0, "the solo path performs real cache writes"
    assert joint_delta < solo_delta, (
        "the joint [2N, T] publish must cost strictly fewer write ops than "
        "two solo [N, T] publishes (#42 ledger semantics)"
    )


# ---------------------------------------------------------------------------
# UT-2432: LLaDA early-finish mixed (C5) — EOS chain leaves, other continues
# ---------------------------------------------------------------------------


def test_ut2432_llada_early_finish_mixed(device: str) -> None:
    """UT-2432: an EOS-finishing chain is removed; the survivor continues alone.

    The prompt-keyed trap keys chain 1's logits on an EOS-emitting table row
    (its greedy row fills EOS everywhere — the min-prob winner) and chain 2's
    on an EOS-free row: chain 1 finishes in the first window and is REMOVED
    from the joint batch (C5), chain 2 completes both windows normally. Both
    outputs stay well-formed, chain 2's tokens equal the solo reference
    BITWISE, the joint batched rows shrink 2N -> N, and nothing crashes.
    """
    inner = _PromptKeyedTrapModel()
    inner.to(device)
    inner.eval()
    llm = _MiniDiffusionLLM(
        inner,
        vocab=_EOS_TRAP_VOCAB,
        tokenizer=_MiniTokenizer(_EOS_TRAP_MASK_ID, _EOS_TRAP_EOS_ID, _EOS_TRAP_PAD_ID),
    )
    p1, p2 = _eos_trap_prompts(device)
    cfg = _eos_trap_llada_config(seed=2432)

    mff_joint = _make_mff(inner, device, multi_parent_batch=True)
    sampler_a = LLaDASampler(llm, config=cfg)
    outs = sampler_a.sample_joint([p1, p2], mff_joint)

    mff_solo = _make_mff(inner, device, multi_parent_batch=True)
    sampler_b = LLaDASampler(llm, config=cfg)
    out1 = sampler_b.sample(p1, folded_model=mff_solo)
    out2 = sampler_b.sample(p2, folded_model=mff_solo)

    # Well-formed outputs: full-canvas shapes, long ids, prompts preserved.
    assert len(outs) == 2
    for out, prompt in zip(outs, (p1, p2)):
        assert tuple(out.sequences.shape) == (1, _TRAP_SEQ_LEN)
        assert out.sequences.dtype == torch.long
        assert torch.equal(out.sequences[0, :_TRAP_PROMPT_LEN], prompt[0])

    # Chain 1: early EOS — window-1 commit is all EOS, the tail stays masked.
    gen1 = outs[0].sequences[0, _TRAP_PROMPT_LEN:_TRAP_SEQ_LEN]
    assert bool((gen1[:4] == _EOS_TRAP_EOS_ID).all()), "chain 1 commits EOS in window 1"
    assert bool((gen1[4:] == _EOS_TRAP_MASK_ID).all()), "chain 1's tail stays masked (early exit)"

    # Chain 2: completes normally — fully denoised, EOS-free, plain fills.
    gen2 = outs[1].sequences[0, _TRAP_PROMPT_LEN:_TRAP_SEQ_LEN]
    assert bool((gen2 != _EOS_TRAP_MASK_ID).all()), "chain 2 fully denoises"
    assert bool((gen2 != _EOS_TRAP_EOS_ID).all()), "chain 2 never commits EOS"
    assert bool((gen2 == _PLAIN_TRAP_TOP).all()), "chain 2 commits the plain top-1 fills"

    # Bitwise parity per chain against the solo references.
    assert torch.equal(outs[0].sequences, out1.sequences), "chain 1 parity (early finish)"
    assert torch.equal(outs[1].sequences, out2.sequences), "chain 2 parity (survivor)"

    # C5 batch structure: the joint verification rows shrink 2N -> N.
    drafts, batched = _split_calls(mff_joint)
    assert len(batched) == 2, "two windows: joint [2N, T] then survivor [N, T]"
    assert tuple(batched[0]["args"][0].shape)[0] == 4, "window 1: 2N joint rows"
    assert tuple(batched[1]["args"][0].shape)[0] == 2, "window 2: N rows after chain 1 left"


# ---------------------------------------------------------------------------
# UT-2433: LLaDA off-switch dispatch (default mff == per-chain sample)
# ---------------------------------------------------------------------------


def test_ut2433_llada_off_switch_dispatch(device: str) -> None:
    """UT-2433: without the switch, sample_joint == [sample(p1), sample(p2)].

    A default-constructed mff (``multi_parent_batch=False``) fails the joint
    preconditions, so ``sample_joint`` must dispatch per chain (DD7-isomorphic
    fallback): the outputs equal back-to-back solo runs on an identical
    off-switch mff BITWISE per chain.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    p1, p2 = _prompts2(device)
    cfg_kwargs: dict[str, Any] = {
        "num_steps": 4,
        "num_tokens": 8,
        "block_size": 4,
        "speculative_sub_block_size": 4,
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "candidate_eos_aware": True,
        "seed": 2433,
    }

    mff_off_a = _make_mff(inner, device)
    outs = LLaDASampler(llm, config=_llada_config(**cfg_kwargs)).sample_joint([p1, p2], mff_off_a)

    mff_off_b = _make_mff(inner, device)
    sampler_b = LLaDASampler(llm, config=_llada_config(**cfg_kwargs))
    out1 = sampler_b.sample(p1, folded_model=mff_off_b)
    out2 = sampler_b.sample(p2, folded_model=mff_off_b)

    assert isinstance(outs, list) and len(outs) == 2
    assert torch.equal(
        outs[0].sequences, out1.sequences
    ), "off-switch chain 1 == solo sample(p1) bitwise"
    assert torch.equal(
        outs[1].sequences, out2.sequences
    ), "off-switch chain 2 == solo sample(p2) bitwise"


# ---------------------------------------------------------------------------
# UT-2434: LLaDA F3 entry ValueErrors (joint path only)
# ---------------------------------------------------------------------------


def test_ut2434_llada_joint_f3_valueerrors(device: str) -> None:
    """UT-2434: temperature>0 / stochastic_transfer=True raise on the joint path.

    F3: candidate-construction determinism holds only on the greedy path, so
    the joint entry rejects ``temperature > 0`` and truthy
    ``stochastic_transfer`` with ValueError (the ``cfg_scale`` rejection
    style). The rejections are JOINT-ONLY: the same configs on an off-switch
    mff dispatch per chain (``sample()`` stays as-is) and must NOT raise.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    p1, p2 = _prompts2(device)
    mff_joint = _make_mff(inner, device, multi_parent_batch=True)
    mff_off = _make_mff(inner, device)
    base: dict[str, Any] = {
        "num_steps": 2,
        "num_tokens": 4,
        "block_size": 4,
        "speculative_sub_block_size": 4,
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "seed": 2434,
    }

    with pytest.raises(ValueError, match=r"temperature"):
        LLaDASampler(llm, config=_llada_config(**base, temperature=0.5)).sample_joint(
            [p1, p2], mff_joint
        )
    with pytest.raises(ValueError, match=r"stochastic"):
        LLaDASampler(llm, config=_llada_config(**base, stochastic_transfer=True)).sample_joint(
            [p1, p2], mff_joint
        )

    # Joint-ONLY rejections: the off-switch dispatch of the same configs runs.
    outs_temp = LLaDASampler(llm, config=_llada_config(**base, temperature=0.5)).sample_joint(
        [p1, p2], mff_off
    )
    assert len(outs_temp) == 2, "off-switch dispatch does not reject temperature > 0"
    outs_stoch = LLaDASampler(
        llm, config=_llada_config(**base, stochastic_transfer=True)
    ).sample_joint([p1, p2], mff_off)
    assert len(outs_stoch) == 2, "off-switch dispatch does not reject stochastic_transfer"


# ---------------------------------------------------------------------------
# UT-2435: LLaDA C2/shape entry ValueErrors
# ---------------------------------------------------------------------------


def test_ut2435_llada_joint_shape_valueerrors(device: str) -> None:
    """UT-2435: joint entry shape/validation ValueErrors (C2 and friends).

    On the joint path (switch-on mff, N > 0): an empty ``prompt_ids_list``,
    a prompt with batch 2, prompts of UNEQUAL token length (C2: the fold ctx
    key (T, prompt_len) must be shared), and ``cfg_scale > 0`` (the frozen
    mirror of ``sample()``) all raise ValueError.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    p1, p2 = _prompts2(device)
    p_short = torch.tensor([[4, 5]], device=device)
    p_batch2 = torch.tensor([[1, 2, 3], [7, 8, 9]], device=device)
    mff_joint = _make_mff(inner, device, multi_parent_batch=True)
    cfg_kwargs: dict[str, Any] = {
        "num_steps": 2,
        "num_tokens": 4,
        "block_size": 4,
        "speculative_sub_block_size": 4,
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "seed": 2435,
    }
    sampler = LLaDASampler(llm, config=_llada_config(**cfg_kwargs))

    with pytest.raises(ValueError):
        sampler.sample_joint([], mff_joint)
    with pytest.raises(ValueError):
        sampler.sample_joint([p_batch2, p2], mff_joint)
    with pytest.raises(ValueError):
        sampler.sample_joint([p1, p_short], mff_joint)
    with pytest.raises(ValueError, match=r"cfg_scale"):
        LLaDASampler(llm, config=_llada_config(**cfg_kwargs, cfg_scale=1.5)).sample_joint(
            [p1, p2], mff_joint
        )


# ---------------------------------------------------------------------------
# UT-2436: LLaDA joint branch-structure spy (plural parents, chain-qualified ids)
# ---------------------------------------------------------------------------


def test_ut2436_llada_joint_branch_structure(device: str) -> None:
    """UT-2436: the joint forward form and chain-qualified branch namespace.

    (a) every batched joint call carries ``parent_branch_ids`` (plural) with
    ``branch_id=None``; (b) NO call carries both a non-None singular
    ``parent_branch_id`` and ``parent_branch_ids`` (mutually exclusive MFF
    face); (c) joint rows are 2N with ``parent_branch_ids == [p1]*N+[p2]*N``
    where p1/p2 are the chains' OWN draft branch ids (per-chain folding
    state — chain 2's candidates must NOT hang on chain 1's winner), and N
    with ``[p2']*N`` after chain 1 finished (the C5 survivor face), with the
    forced_divergent stack shaped [rows, T]; (d) the two chains' branch id
    sets are DISJOINT (chain-qualified ids never collide in the shared cache).
    """
    inner = _PromptKeyedTrapModel()
    inner.to(device)
    inner.eval()
    llm = _MiniDiffusionLLM(
        inner,
        vocab=_EOS_TRAP_VOCAB,
        tokenizer=_MiniTokenizer(_EOS_TRAP_MASK_ID, _EOS_TRAP_EOS_ID, _EOS_TRAP_PAD_ID),
    )
    p1, p2 = _eos_trap_prompts(device)
    mff = _make_mff(inner, device, multi_parent_batch=True)
    sampler = LLaDASampler(llm, config=_eos_trap_llada_config(seed=2436))
    sampler.sample_joint([p1, p2], mff)

    drafts, batched = _split_calls(mff)
    assert len(drafts) == 3, "chain 1 drafts once (finishes), chain 2 drafts twice"
    assert len(batched) == 2, "one joint verification per window"

    # (b) global mutual exclusion: never both singular and plural parents.
    for call in mff.calls:
        kw = call["kwargs"]
        assert not (
            kw.get("parent_branch_id") is not None and kw.get("parent_branch_ids") is not None
        ), "parent_branch_id (singular) and parent_branch_ids (plural) are mutually exclusive"

    # Attribute each serial draft to its chain via the spied canvas prompt.
    chain_drafts: list[list[str]] = [[], []]
    for call in drafts:
        toks = call["args"][0]
        assert tuple(toks.shape)[0] == 1, "per-chain drafts are [1, T] serial forwards"
        chain = _chain_of_canvas(toks, p1, p2, _TRAP_PROMPT_LEN)
        chain_drafts[chain].append(call["kwargs"]["branch_id"])
    assert len(chain_drafts[0]) == 1, "chain 1 stops drafting after its EOS window"
    assert len(chain_drafts[1]) == 2, "chain 2 drafts for both windows"

    n = 2
    # Window 1: the joint [2N, T] call.
    kw0 = batched[0]["kwargs"]
    assert kw0.get("branch_id") is None, "(a) batched joint calls pass branch_id=None"
    pids0 = kw0.get("parent_branch_ids")
    assert pids0 is not None, "(a) batched joint calls carry parent_branch_ids (plural)"
    assert tuple(batched[0]["args"][0].shape)[0] == 2 * n, "(c) joint rows == 2N"
    assert (
        pids0 == [chain_drafts[0][0]] * n + [chain_drafts[1][0]] * n
    ), "(c) per-row parents are the chains' OWN window-1 draft ids"
    bids0 = kw0["branch_ids"]
    assert len(bids0) == 2 * n
    forced0 = kw0.get("forced_divergent")
    assert forced0 is not None and tuple(forced0.shape) == (
        2 * n,
        _TRAP_SEQ_LEN,
    ), "the joint forced_divergent is the stack of both chains' unit_forced"

    # Window 2: the survivor's [N, T] call against chain 2's OWN new draft.
    kw1 = batched[1]["kwargs"]
    assert kw1.get("branch_id") is None, "(a) survivor calls also pass branch_id=None"
    pids1 = kw1.get("parent_branch_ids")
    assert pids1 is not None, "(a) survivor calls carry parent_branch_ids (plural)"
    assert tuple(batched[1]["args"][0].shape)[0] == n, "(c) survivor rows == N"
    assert (
        pids1 == [chain_drafts[1][1]] * n
    ), "C5: the surviving chain folds against its OWN window-2 draft"
    forced1 = kw1.get("forced_divergent")
    assert forced1 is not None and tuple(forced1.shape) == (n, _TRAP_SEQ_LEN)

    # (d) chain-qualified ids: the two chains' namespaces never collide.
    chain1_ids = set(chain_drafts[0]) | set(bids0[:n])
    chain2_ids = set(chain_drafts[1]) | set(bids0[n:]) | set(kw1["branch_ids"])
    assert chain1_ids.isdisjoint(
        chain2_ids
    ), "(d) branch ids are chain-qualified — no shared-cache id collision"
    assert pids1[0] not in chain1_ids, "chain 2 never hangs on a chain 1 branch"


# ---------------------------------------------------------------------------
# UT-2437: LLaDA greedy_win_stats aggregation across both chains
# ---------------------------------------------------------------------------


def test_ut2437_llada_greedy_win_stats_aggregation(device: str) -> None:
    """UT-2437: the instance-level probe aggregates across both chains.

    One sampler instance drives both chains, so ``greedy_win_stats`` counts
    BOTH chains' windows on the joint path: on the early-finish fixture
    chain 1 contributes 1 window and chain 2 contributes 2 —
    ``speculative_calls == 3`` (a per-joint-call counter would read 2; a
    single-chain counter 1), the counters move vs before, and the greedy
    wins stay within [0, speculative_calls].
    """
    inner = _PromptKeyedTrapModel()
    inner.to(device)
    inner.eval()
    llm = _MiniDiffusionLLM(
        inner,
        vocab=_EOS_TRAP_VOCAB,
        tokenizer=_MiniTokenizer(_EOS_TRAP_MASK_ID, _EOS_TRAP_EOS_ID, _EOS_TRAP_PAD_ID),
    )
    p1, p2 = _eos_trap_prompts(device)
    mff = _make_mff(inner, device, multi_parent_batch=True)
    sampler = LLaDASampler(llm, config=_eos_trap_llada_config(seed=2437))

    before = dict(sampler.greedy_win_stats)
    assert before == {"greedy_wins": 0, "speculative_calls": 0}, "fresh probe reads {0, 0}"

    sampler.sample_joint([p1, p2], mff)
    after = sampler.greedy_win_stats

    assert after["speculative_calls"] > before["speculative_calls"], "counters move"
    assert (
        after["speculative_calls"] == 3
    ), "aggregation across both chains: 1 (chain 1, early finish) + 2 (chain 2)"
    assert after["speculative_calls"] > 1, "both chains' windows are counted"
    assert 0 <= after["greedy_wins"] <= after["speculative_calls"]


# ---------------------------------------------------------------------------
# UT-2440: Dream joint parity (bitwise, both chains)
# ---------------------------------------------------------------------------


def test_ut2440_dream_joint_parity_bitwise(device: str) -> None:
    """UT-2440: Dream joint [p1, p2] equals back-to-back solo runs bit-for-bit.

    Same discipline as UT-2430 on the Dream family (left-padded super-step
    adapter, ``speculative_steps`` default): per-chain tokens and history
    snapshots bitwise equal, and the joint verification calls carry the Dream
    ctx ``position_ids`` payload (the AR023 position threading must survive
    the joint batching).
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    p1, p2 = _prompts2(device)
    cfg_kwargs: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 8,
        "num_block_candidates": 3,
        "candidate_top_k": 3,
        "candidate_include_greedy": True,
        "seed": 2440,
        "return_history": True,
    }

    mff_joint = _make_mff(inner, device, multi_parent_batch=True)
    sampler_a = DreamSampler(llm, config=_dream_config(**cfg_kwargs))
    outs = sampler_a.sample_joint([p1, p2], mff_joint)

    mff_solo = _make_mff(inner, device, multi_parent_batch=True)
    sampler_b = DreamSampler(llm, config=_dream_config(**cfg_kwargs))
    out1 = sampler_b.sample(p1, folded_model=mff_solo)
    out2 = sampler_b.sample(p2, folded_model=mff_solo)

    assert isinstance(outs, list) and len(outs) == 2, "sample_joint returns per-chain outputs"
    assert isinstance(outs[0], SamplerOutput) and isinstance(outs[1], SamplerOutput)
    assert torch.equal(
        outs[0].sequences, out1.sequences
    ), "Dream chain 1 tokens must be bitwise equal to the solo run"
    assert torch.equal(
        outs[1].sequences, out2.sequences
    ), "Dream chain 2 tokens must be bitwise equal to the solo run"

    assert len(outs[0].history) == len(out1.history), "Dream chain 1 history length parity"
    assert len(outs[1].history) == len(out2.history), "Dream chain 2 history length parity"
    for joint_snap, solo_snap in zip(outs[0].history, out1.history):
        assert torch.equal(joint_snap, solo_snap), "Dream chain 1 history snapshots bitwise equal"
    for joint_snap, solo_snap in zip(outs[1].history, out2.history):
        assert torch.equal(joint_snap, solo_snap), "Dream chain 2 history snapshots bitwise equal"

    # The Dream ctx position payload threads through the joint batched calls.
    drafts, batched = _split_calls(mff_joint)
    assert batched, "the joint path makes batched verification calls"
    for call in batched:
        assert (
            call["kwargs"].get("position_ids") is not None
        ), "joint batched calls carry the Dream ctx position_ids payload"


# ---------------------------------------------------------------------------
# UT-2441: Dream off-switch dispatch
# ---------------------------------------------------------------------------


def test_ut2441_dream_off_switch_dispatch(device: str) -> None:
    """UT-2441: without the switch, Dream sample_joint == per-chain sample().

    The default-constructed mff fails the joint preconditions, so the Dream
    ``sample_joint`` dispatches per chain and the outputs equal back-to-back
    solo runs on an identical off-switch mff BITWISE per chain.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    p1, p2 = _prompts2(device)
    cfg_kwargs: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 8,
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "seed": 2441,
    }

    mff_off_a = _make_mff(inner, device)
    outs = DreamSampler(llm, config=_dream_config(**cfg_kwargs)).sample_joint([p1, p2], mff_off_a)

    mff_off_b = _make_mff(inner, device)
    sampler_b = DreamSampler(llm, config=_dream_config(**cfg_kwargs))
    out1 = sampler_b.sample(p1, folded_model=mff_off_b)
    out2 = sampler_b.sample(p2, folded_model=mff_off_b)

    assert isinstance(outs, list) and len(outs) == 2
    assert torch.equal(
        outs[0].sequences, out1.sequences
    ), "off-switch Dream chain 1 == solo sample(p1) bitwise"
    assert torch.equal(
        outs[1].sequences, out2.sequences
    ), "off-switch Dream chain 2 == solo sample(p2) bitwise"


# ---------------------------------------------------------------------------
# UT-2442: Dream F3 entry ValueErrors (joint path only)
# ---------------------------------------------------------------------------


def test_ut2442_dream_joint_f3_valueerrors(device: str) -> None:
    """UT-2442: Dream temperature>0 / stochastic_transfer=True raise on the joint path.

    The F3 greedy-path rejections mirror UT-2434 on the Dream family — and
    are joint-ONLY: the same configs on an off-switch mff dispatch per chain
    without raising.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    p1, p2 = _prompts2(device)
    mff_joint = _make_mff(inner, device, multi_parent_batch=True)
    mff_off = _make_mff(inner, device)
    base: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 8,
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "seed": 2442,
    }

    with pytest.raises(ValueError, match=r"temperature"):
        DreamSampler(llm, config=_dream_config(**base, temperature=0.5)).sample_joint(
            [p1, p2], mff_joint
        )
    with pytest.raises(ValueError, match=r"stochastic"):
        DreamSampler(llm, config=_dream_config(**base, stochastic_transfer=True)).sample_joint(
            [p1, p2], mff_joint
        )

    outs_temp = DreamSampler(llm, config=_dream_config(**base, temperature=0.5)).sample_joint(
        [p1, p2], mff_off
    )
    assert len(outs_temp) == 2, "off-switch dispatch does not reject temperature > 0"
    outs_stoch = DreamSampler(
        llm, config=_dream_config(**base, stochastic_transfer=True)
    ).sample_joint([p1, p2], mff_off)
    assert len(outs_stoch) == 2, "off-switch dispatch does not reject stochastic_transfer"


# ---------------------------------------------------------------------------
# UT-2443: Dream joint write-op ledger strictly smaller
# ---------------------------------------------------------------------------


def test_ut2443_dream_joint_ledger_strictly_smaller(device: str) -> None:
    """UT-2443: the Dream joint cache write ledger is strictly smaller than solo.

    Same discipline as UT-2431 on the Dream family (one-shot super-step with
    the default ``speculative_steps``): the joint [2N, T] publish vs two solo
    [N, T] publishes, both deltas positive, joint strictly smaller.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    p1, p2 = _prompts2(device)
    cfg_kwargs: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 8,
        "num_block_candidates": 3,
        "candidate_top_k": 3,
        "candidate_include_greedy": True,
        "seed": 2443,
    }

    mff_joint = _make_mff(inner, device, multi_parent_batch=True)
    sampler_a = DreamSampler(llm, config=_dream_config(**cfg_kwargs))
    before_joint = mff_joint.cache.write_op_count
    sampler_a.sample_joint([p1, p2], mff_joint)
    joint_delta = mff_joint.cache.write_op_count - before_joint

    mff_solo = _make_mff(inner, device, multi_parent_batch=True)
    sampler_b = DreamSampler(llm, config=_dream_config(**cfg_kwargs))
    before_solo = mff_solo.cache.write_op_count
    sampler_b.sample(p1, folded_model=mff_solo)
    sampler_b.sample(p2, folded_model=mff_solo)
    solo_delta = mff_solo.cache.write_op_count - before_solo

    assert joint_delta > 0, "the Dream joint path performs real cache writes"
    assert solo_delta > 0, "the Dream solo path performs real cache writes"
    assert joint_delta < solo_delta, (
        "the joint [2N, T] publish must cost strictly fewer write ops than "
        "two solo Dream [N, T] publishes"
    )


# ---------------------------------------------------------------------------
# UT-2444: composed-path zero-drift guard (solo smoke, both families)
# ---------------------------------------------------------------------------


def test_ut2444_composed_solo_zero_drift_smoke(device: str) -> None:
    """UT-2444: the solo composed ``_speculative_commit`` path is undisturbed.

    One solo LLaDA and one solo Dream speculative ``sample()`` through the
    ordinary single-chain entry (the refactor must not change solo
    behavior): outputs are tensors of the expected full-canvas shape, the
    prompt region is preserved, and the finished/decoded semantics leave the
    generation region fully denoised (no mask tokens) with the deterministic
    position-table heads.
    """
    prompt = torch.tensor([[1, 2, 3]], device=device)

    inner_llada = _build_mini(device, head="postable")
    llm_llada = _MiniDiffusionLLM(inner_llada)
    cfg_llada = _llada_config(
        num_steps=4,
        num_tokens=8,
        block_size=4,
        speculative_sub_block_size=4,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        seed=2444,
    )
    out_llada = LLaDASampler(llm_llada, config=cfg_llada).sample(
        prompt, folded_model=_make_mff(inner_llada, device)
    )
    assert isinstance(out_llada, SamplerOutput)
    assert isinstance(out_llada.sequences, torch.Tensor)
    assert tuple(out_llada.sequences.shape) == (1, 11)
    assert out_llada.sequences.dtype == torch.long
    assert torch.equal(out_llada.sequences[0, :3], prompt[0]), "prompt region preserved"
    assert bool((out_llada.sequences[0, 3:11] != _MASK_ID).all()), "generation fully denoised"

    inner_dream = _build_mini(device, head="postable")
    llm_dream = _MiniDiffusionLLM(inner_dream)
    cfg_dream = _dream_config(
        num_steps=8,
        num_tokens=8,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        seed=2444,
    )
    out_dream = DreamSampler(llm_dream, config=cfg_dream).sample(
        prompt, folded_model=_make_mff(inner_dream, device)
    )
    assert isinstance(out_dream, SamplerOutput)
    assert isinstance(out_dream.sequences, torch.Tensor)
    assert tuple(out_dream.sequences.shape) == (1, 11)
    assert out_dream.sequences.dtype == torch.long
    assert torch.equal(out_dream.sequences[0, :3], prompt[0]), "prompt region preserved"
    assert bool((out_dream.sequences[0, 3:11] != _MASK_ID).all()), "generation fully denoised"


# ---------------------------------------------------------------------------
# UT-2438: LLaDA joint + progressive_refill parity (the T009 coverage gap)
# ---------------------------------------------------------------------------


def test_ut2438_llada_joint_refill_parity_bitwise(device: str) -> None:
    """UT-2438: joint + ``progressive_refill`` equals back-to-back solo bitwise.

    The AR024 two-phase refill runs INSIDE the joint window loop (segment B's
    partial commit + ``_refill_phase_two`` with the chain-scoped ``_r``
    branch tag and the F1 instance-slot swap). A wrong chain continuation
    (the refill writing through the WRONG chain's parent slot or an
    unqualified ``_r`` tag colliding across chains) would silently break the
    C6/H1 premise — this test pins the joint path against the per-chain solo
    reference up to refill tie-order. Exact sequence equality is not portable:
    the joint and solo logits can differ in the last bits on some torch
    builds, which flips the confidence-sort order of equal-scoring refill
    positions (tokens permute within the same block). A wrong chain
    continuation still changes token VALUES and fails the multiset check.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    p1, p2 = _prompts2(device)
    cfg_kwargs: dict[str, Any] = {
        "num_steps": 4,
        "num_tokens": 8,
        "block_size": 4,
        "speculative_sub_block_size": 4,
        "num_block_candidates": 3,
        "candidate_top_k": 3,
        "candidate_include_greedy": True,
        "candidate_eos_aware": True,
        "progressive_refill": True,
        "refill_top_k": 2,
        "seed": 2438,
        "return_history": True,
    }

    mff_joint = _make_mff(inner, device, multi_parent_batch=True)
    outs = LLaDASampler(llm, config=_llada_config(**cfg_kwargs)).sample_joint([p1, p2], mff_joint)
    mff_solo = _make_mff(inner, device, multi_parent_batch=True)
    sampler_b = LLaDASampler(llm, config=_llada_config(**cfg_kwargs))
    ref1 = sampler_b.sample(p1, folded_model=mff_solo)
    ref2 = sampler_b.sample(p2, folded_model=mff_solo)

    def _assert_tie_order_parity(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
        assert actual.shape == expected.shape, label
        assert torch.equal(
            torch.sort(actual, dim=-1).values, torch.sort(expected, dim=-1).values
        ), label

    _assert_tie_order_parity(outs[0].sequences, ref1.sequences, "chain 0 refill parity")
    _assert_tie_order_parity(outs[1].sequences, ref2.sequences, "chain 1 refill parity")
    assert len(outs[0].history) == len(ref1.history), "chain 0 history length parity"
    assert len(outs[1].history) == len(ref2.history), "chain 1 history length parity"
    for j, (h_joint, h_ref) in enumerate(zip(outs[0].history, ref1.history)):
        _assert_tie_order_parity(h_joint, h_ref, f"chain 0 history snapshot {j} parity")
    for j, (h_joint, h_ref) in enumerate(zip(outs[1].history, ref2.history)):
        _assert_tie_order_parity(h_joint, h_ref, f"chain 1 history snapshot {j} parity")


# ---------------------------------------------------------------------------
# UT-2445: Dream joint entry shape/validation ValueErrors (the T009 gap)
# ---------------------------------------------------------------------------


def test_ut2445_dream_joint_shape_valueerrors(device: str) -> None:
    """UT-2445: Dream joint entry shape/validation ValueErrors (C2 and friends).

    Mirrors UT-2435 on the Dream family: an empty ``prompt_ids_list``, a
    prompt with batch 2, prompts of UNEQUAL token length (C2: the joint
    fold ctx must be shared), and ``cfg_scale > 0`` all raise ValueError on
    the joint path.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    p1, p2 = _prompts2(device)
    p_short = torch.tensor([[4, 5]], device=device)
    p_batch2 = torch.tensor([[1, 2, 3], [7, 8, 9]], device=device)
    mff_joint = _make_mff(inner, device, multi_parent_batch=True)
    cfg_kwargs: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 8,
        "num_block_candidates": 2,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "seed": 2445,
    }
    sampler = DreamSampler(llm, config=_dream_config(**cfg_kwargs))

    with pytest.raises(ValueError):
        sampler.sample_joint([], mff_joint)
    with pytest.raises(ValueError):
        sampler.sample_joint([p_batch2, p2], mff_joint)
    with pytest.raises(ValueError):
        sampler.sample_joint([p1, p_short], mff_joint)
    with pytest.raises(ValueError, match=r"cfg_scale"):
        DreamSampler(llm, config=_dream_config(**cfg_kwargs, cfg_scale=1.5)).sample_joint(
            [p1, p2], mff_joint
        )
