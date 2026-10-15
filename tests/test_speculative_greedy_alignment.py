"""Red-phase tests for AR022: speculative quality alignment (UT-2201..UT-2207).

Contract: ``specs/changes/AR022-speculative-quality-alignment/design.md``
§4.2.2/§4.3/§6.1 and ``srs.md`` §3.1-§3.3 — anchoring candidate row 0 of the
batched speculative verification to the greedy hypothesis (the draft top-k
ordered-set first entry = the srs-frozen argmax definition), competing under
the unchanged min-prob selection with the real posterior logits:

* FR1: ``candidate_include_greedy=True`` fills candidate row 0 with the
  still-masked positions' draft argmax (deterministic across runs, zero
  device readback, zero generator consumption); random rows j=1..N-1 keep
  the frozen per-candidate seed formula with j keeping its row number, so
  rows 1..N-1 match a greedy-off run bit-for-bit on an identical canvas
  (asserted only inside the first speculative small block — the
  canvas-consistent region); ``[N, T]`` batch shape and ``branch_ids``
  naming unchanged; greedy on + N=1 degenerates to a single-row
  pure-greedy candidate (legal, trivially winning);
* FR2: the min-prob selection rule is unchanged — the greedy row competes
  fairly and CAN LOSE (the ``_PosteriorTrapFixture`` pins a posterior-table
  loss scenario, guarding against a degenerate always-pick-greedy
  implementation); the instance-level ``greedy_win_stats`` probe exposes
  ``{"greedy_wins", "speculative_calls"}`` (cumulative across ``sample``
  calls); chain continuation and stop semantics are identical on both the
  greedy-win and random-win branches;
* FR3: ``candidate_include_greedy`` defaults to False with byte-identical
  behavior (default off == explicit off == AR021 end state on the raw /
  N=0 / speculative N=4 / stop-token faces); True + N=0 raises ValueError
  at construction (no silent degradation); the YAML key is consumed by
  ``load_config`` with zero unknown-key UserWarning (AGENTS #5).

UT -> test function mapping (one UT may map to several functions):

===== ========================================== ================================
UT    Function(s)                                Asserted points
===== ========================================== ================================
2201  test_ut2201_default_off_zero_disturbance   explicit greedy-off == default ==
                                                  AR021 end state bit-exact on
                                                  raw / N=0 / speculative N=4 /
                                                  stop-token faces;
                                                  ``greedy_win_stats`` untouched
                                                  ({0, 0}) when off; the
                                                  constant-mask argmax boundary
                                                  (argmax == mask_id) completes
                                                  without exception on both the
                                                  greedy-on and baseline paths.
2202  test_ut2202_greedy_row_construction        [N,T] shape + ``branch_ids``
                                                  naming + draft parent
                                                  unchanged; row 0 == the
                                                  still-masked draft topk
                                                  first entry (recomputed from
                                                  the spied draft forward);
                                                  rows 1..3 == the greedy-off
                                                  same-seed rows (single small
                                                  block premise pinned);
                                                  cross-instance row-0
                                                  bit-exactness; generator
                                                  constructions N-1 (off: N);
                                                  global RNG state snapshot
                                                  unchanged; zero
                                                  .item/.tolist/.cpu inside
                                                  the construction window
                                                  (``_ReadbackSpyMFF``).
2203  test_ut2203_greedy_loses_to_random         posterior-trap fixture: j*==1
      test_ut2203_greedy_wins_commit             with recomputed
                                                  min_scores[0] <
                                                  min_scores[1]; committed
                                                  region == winner row (all
                                                  blocks); greedy_wins == 0;
                                                  the winning-seed twin
                                                  commits the greedy
                                                  hypothesis verbatim with
                                                  greedy_wins ==
                                                  speculative_calls.
2204  test_ut2204_greedy_win_stats_consistency   speculative_calls == spied
                                                  batched calls across two
                                                  ``sample`` calls;
                                                  greedy_wins == sum(j*==0)
                                                  recomputed per block from
                                                  the spied batched logits;
                                                  greedy-off control stays
                                                  {0, 0}; the probe dict is
                                                  read-only (external mutation
                                                  cannot corrupt the counts).
2205  test_ut2205_stop_and_chain_semantics       greedy-win + stop: baseline
                                                  bit-exact finished/pad
                                                  trimming, ``_folding_parent``
                                                  == ..._cand0 (trivial win at
                                                  candidate_top_k=1);
                                                  random-win (trap fixture):
                                                  ``_folding_parent`` ==
                                                  ..._cand1 and the next draft
                                                  forward chains onto the
                                                  winner branch (both winner
                                                  branches share one code
                                                  path).
2206  test_ut2206_config_validation_and_yaml     True + N=0 -> ValueError
                                                  (frozen message); True +
                                                  N>0 legal; default False;
                                                  ``dataclasses.replace``
                                                  face; ActFoldConfig direct
                                                  construction + YAML
                                                  ``load_config`` consume the
                                                  key with zero UserWarning
                                                  (True) and default False
                                                  (key absent).
2207  test_ut2207_n1_pure_greedy                 one [1,T] single-row batched
                                                  call per small block; row 0
                                                  == draft argmax (UT-2202
                                                  recompute); zero generator
                                                  constructions; trivial
                                                  j*==0; chain continuation
                                                  onto ..._cand0; stats:
                                                  speculative_calls == small
                                                  blocks, greedy_wins == calls.
===== ========================================== ================================

Red state (why every test fails today) — each failure is a production-missing
face, never a test-own bug:

* ``FastDLLMSamplerConfig`` has no ``candidate_include_greedy`` field —
  every explicit kwarg pass (greedy on, and the explicit-off zero-disturbance
  controls per the red-phase discipline) fails with ``TypeError`` (UT-2201,
  UT-2202, UT-2203, UT-2204, UT-2205, UT-2207, and the UT-2206 validation
  face);
* ``FastDLLMSampler`` has no ``greedy_win_stats`` probe — the observation
  faces fail with ``AttributeError`` (UT-2201(e), UT-2203, UT-2204,
  UT-2207);
* ``ActFoldConfig`` has no ``candidate_include_greedy`` field — the direct
  constructor raises ``TypeError``, and the YAML face trips the load_config
  unknown-key ``UserWarning`` plus ``AttributeError`` on the missing field
  (UT-2206).

``greedy_win_stats`` schema frozen by these UTs (design §4.1.2):
``{"greedy_wins": int, "speculative_calls": int}`` — instance-cumulative,
surviving across ``sample`` calls; both counters 0 while greedy anchoring is
off (the observation code never runs).

Fixtures follow ``tests/test_generation_loop_folding.py`` (self-contained
LLaMA-layout dual-form-mask mini decoder, ``_MiniDiffusionLLM``, ``_SpyMFF``,
``_sampler_config`` helper, autouse seed/profiler fixtures) with the AR022
additions required by design §6.1: a ``constant_mask`` head variant (the
mask-token argmax boundary), a ``_ReadbackSpyMFF`` (readback-forbidding
wrappers installed between the draft forward and the batched call — the FR1
construction window, UT-2112 precedent), and the ``_PosteriorTrapFixture``
five-component posterior-trap construction (zero-mixing layers, orthogonal
Hadamard embeddings, right-shift table head, small_block_size=1, pinned seed)
making the greedy row's min-prob loss construct-level reproducible.
"""

from __future__ import annotations

import dataclasses
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
from actfold.utils.config_manager import ActFoldConfig, load_config

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

# Posterior-trap fixture constants (design §6.1 UT-2203). The trap vocabulary
# is smaller than the hidden size so 16 pairwise-orthogonal zero-mean
# unit-variance basis vectors exist in 32 dimensions (Hadamard rows 1..16).
_TRAP_VOCAB = 16
_TRAP_MASK_ID = 15
_TRAP_STOP_ID = 14
_TRAP_PAD_ID = 13
_TRAP_GREEDY_ID = 5  # "A": draft argmax; T[A] uniform -> low posterior self-prob
_TRAP_ALT_ID = 6  # "C": draft runner-up; T[C][C]=3.0 -> high posterior self-prob
_TRAP_BLOCK_SIZE = 8
_TRAP_NUM_SMALL_BLOCKS = 8  # small_block_size=1, single canvas block
_TRAP_BLOCK_START = 8  # canvas position of the block region (prompt 3 + pad 5)


@pytest.fixture(autouse=True)
def _profiler_state() -> Iterator[None]:
    """Snapshot-reset the global stability profiler around every case."""
    GLOBAL_STABILITY_PROFILER.reset()
    yield
    GLOBAL_STABILITY_PROFILER.reset()


# ---------------------------------------------------------------------------
# Self-contained mini block-causal decoder (test_generation_loop_folding style)
# ---------------------------------------------------------------------------


class _DualFormAttention(nn.Module):
    """One attention implementation accepting both production mask forms.

    Raw form: ``[B, T]`` padding mask (the block-causal AND non-pad mask is
    built internally). Folded form: ``[T, T]`` prebuilt block-causal AND
    non-pad mask. Both forms must be semantically equivalent for the same
    canvas.
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
    """Head whose argmax is a fixed token at every position."""

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
        if head == "spike":
            self.lm_head = _SpikeHead(_VOCAB, _SPIKE_TOP1_ID, _SPIKE_TOP2_ID)
        elif head == "constant":
            self.lm_head = _ConstantHead(_VOCAB, _STOP_ID)
        elif head == "constant_mask":
            # UT-2201(f): the mask-token argmax boundary (argmax == mask_id).
            self.lm_head = _ConstantHead(_VOCAB, _MASK_ID)
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
    """DiffusionLLM wrapper around a mini decoder, recording raw forwards."""

    def __init__(
        self,
        inner: nn.Module,
        vocab: int = _VOCAB,
        hidden: int = _HIDDEN,
        num_layers: int = _NUM_LAYERS,
        tokenizer: _MiniTokenizer | None = None,
    ) -> None:
        super().__init__("mini-block-causal")
        self.model = inner
        self._vocab = vocab
        self._hidden = hidden
        self._num_layers = num_layers
        self.tokenizer = tokenizer or _MiniTokenizer(_MASK_ID, _STOP_ID, _PAD_ID)
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
        return self._num_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden

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


class _ReadbackSpyMFF(_SpyMFF):
    """_SpyMFF forbidding device readback inside the candidate-construction window.

    The window opens when a serial (draft) forward returns and closes when the
    next batched ``[N, T]`` call arrives — exactly the FR1 candidate-
    construction segment (design §4.1.1: the ``int(argmax)`` selection
    readback happens AFTER the batched call and outside the window). Any
    ``.item``/``.tolist``/``.cpu`` call inside the window raises
    AssertionError (UT-2112 raising-wrapper precedent). ``disarm`` restores
    the original tensor methods and must be reachable from a ``finally``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.armed = False
        self._originals: dict[str, Any] = {}

    def _arm(self) -> None:
        self.armed = True
        for name in ("item", "tolist", "cpu"):
            self._originals[name] = getattr(torch.Tensor, name)
            setattr(torch.Tensor, name, self._make_forbidden(name))

    def _make_forbidden(self, name: str) -> Any:
        def _forbidden(tensor_self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                f"device readback .{name}() inside the greedy candidate "
                "construction window (FR1 zero-readback face)"
            )

        return _forbidden

    def disarm(self) -> None:
        for name, original in self._originals.items():
            setattr(torch.Tensor, name, original)
        self._originals.clear()
        self.armed = False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("branch_ids") is not None and self.armed:
            self.disarm()
        out = super().forward(*args, **kwargs)
        if kwargs.get("branch_ids") is None and not self.armed:
            self._arm()
        return out


# ---------------------------------------------------------------------------
# Posterior-trap fixture (design §6.1 UT-2203 five-component construction)
# ---------------------------------------------------------------------------


def _hadamard_basis(vocab: int, hidden: int) -> torch.Tensor:
    """Zero-mean unit-variance pairwise-orthogonal rows (Sylvester Hadamard).

    Row v uses Hadamard row v+1 (row 0 is all-ones and not zero-mean); every
    row index >= 1 is balanced (+1/-1), so each basis vector has zero mean
    and unit variance, and distinct rows are exactly orthogonal (integer dot
    products, exact in float32). LayerNorm(s * t_v + c * 1) == t_v for the
    gamma=1 / beta=0 norm that precedes the head.
    """
    rows = []
    for v in range(vocab):
        i = v + 1
        rows.append([-1.0 if bin(i & h).count("1") % 2 else 1.0 for h in range(hidden)])
    return torch.tensor(rows, dtype=torch.float32)


def _trap_table() -> torch.Tensor:
    """Posterior table ``T`` (design §6.1 UT-2203 component (iv)).

    ``T[mask]``: A=3.0 / C=1.0 — the draft top-2 at a still-masked commit
    position is {A, C} with argmax A, so the greedy row fills A. ``T[A]``:
    all zero (uniform) — P(A) ~= 1/V, the greedy row's commit score.
    ``T[C]``: C=3.0 — P(C) ~= e^3/(e^3 + V-1), the winning score.
    """
    table = torch.zeros(_TRAP_VOCAB, _TRAP_VOCAB)
    table[_TRAP_MASK_ID, _TRAP_GREEDY_ID] = 3.0
    table[_TRAP_MASK_ID, _TRAP_ALT_ID] = 1.0
    table[_TRAP_ALT_ID, _TRAP_ALT_ID] = 3.0
    return table


class _ShiftedTableHead(nn.Module):
    """Right-shift table head: ``raw[:, q] = T_table[input[q+1]]`` (wrap at end).

    ``W = sum_u outer(T_table[u], t_u) / H`` over the orthogonal basis
    ``{t_u}`` makes ``W @ t_in == T_table[in]`` exactly (up to the LayerNorm
    eps scale), so the head's raw logits cancel the sampler's
    ``right_shift_logits``: ``shifted[:, p] = raw[:, p-1] =
    T_table[input[p]]`` — the posterior probability of a committed token is
    ``softmax(T_table[fill])[fill]``, fully controlled by the candidate
    content. This input-dependence is what a pure position-function head can
    never provide (design §4.2.2(3): the greedy row is unbeatable there).
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


class _TrapModel(nn.Module):
    """LLaMA-layout posterior-trap decoder (design §6.1 UT-2203 (i)+(ii)+(iii)).

    (i) Zero-mixing layers: the attention and MLP blocks contribute exactly
    zero, so the residual chain carries the embedding through unchanged.
    (ii) Orthogonal embeddings: ``embed.weight[v] = s * t_v + c * 1`` with
    the Hadamard basis, so ``LayerNorm(embed(v)) == t_v`` and the head input
    is the constructed basis vector of the input token. (iii) The
    ``_ShiftedTableHead`` supplies the input-dependent posterior table.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": _TRAP_VOCAB})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(_TRAP_VOCAB, _HIDDEN)
        self.model.layers = nn.ModuleList(
            [_DualFormLayer(_HIDDEN, _TRAP_BLOCK_SIZE) for _ in range(_NUM_LAYERS)]
        )
        self.model.norm = nn.LayerNorm(_HIDDEN)
        basis = _hadamard_basis(_TRAP_VOCAB, _HIDDEN)
        self.lm_head = _ShiftedTableHead(_trap_table(), basis)
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
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.lm_head(self.model.norm(x))


def _trap_row1_draw(base_seed: int, small_block_idx: int) -> int:
    """Predict row 1's top-k draw by replaying the frozen seed formula (host).

    Mirrors ``fast_dllm_sampler.py:354-358`` exactly for the trap geometry
    (block_idx=0, j=1, k_eff=2, num_masked=1) without touching production
    code — the draw index selects draft_set[0] (A) or draft_set[1] (C).
    """
    seed_j = (base_seed * 1_000_003 + 0 * 8_191 + small_block_idx * 131 + 1) & 0x7FFF_FFFF
    generator = torch.Generator(device="cpu").manual_seed(seed_j)
    return int(torch.randint(0, 2, (1,), generator=generator))


def _trap_base_seed(row1_draw: int) -> int:
    """Search a base seed whose row-1 draw equals ``row1_draw`` at every block.

    Design §6.1: "seed derived offline and pinned via the frozen formula" —
    replaying the frozen formula over candidate base seeds pins the
    deterministic draw pattern (expected search ~2^8 candidates for the
    all-C / all-A patterns).
    """
    for base in range(1, 100_000):
        if all(_trap_row1_draw(base, sb) == row1_draw for sb in range(_TRAP_NUM_SMALL_BLOCKS)):
            return base
    raise AssertionError(
        f"no trap base seed found with row-1 draw == {row1_draw} at every "
        "small block (fixture construction failure, not a production face)"
    )


class _PosteriorTrapFixture:
    """Posterior-trap bundle (design §6.1 UT-2203): greedy CAN lose min-prob.

    Components: (i) zero-mixing layers, (ii) orthogonal Hadamard embeddings
    (LayerNorm(embed(v)) == t_v), (iii) the right-shift table head canceling
    ``right_shift_logits`` so the commit posterior is softmax(T[fill])[fill],
    (iv) the table {T[mask]: A=3/C=1, T[A]: uniform, T[C]: C=3}, (v)
    small_block_size=1 (M=1 per speculative call — the all-C row is the only
    row that can beat the greedy row, and M=1 makes the construction robust
    to the seed) with N=2, k=2 and the row-1 draw pinned via the frozen seed
    formula. On the greedy-loses pattern every small block commits C via
    row 1 (j*==1); on the greedy-wins pattern row 1 twins the greedy row and
    the trivial tie selects row 0.
    """

    def __init__(self, device: str, row1_draw: int) -> None:
        self.inner = _TrapModel()
        self.inner.to(device)
        self.inner.eval()
        self.llm = _MiniDiffusionLLM(
            self.inner,
            vocab=_TRAP_VOCAB,
            tokenizer=_MiniTokenizer(_TRAP_MASK_ID, _TRAP_STOP_ID, _TRAP_PAD_ID),
        )
        self.config = _sampler_config(
            seed=_trap_base_seed(row1_draw),
            mask_token_id=_TRAP_MASK_ID,
            stop_token_id=_TRAP_STOP_ID,
            pad_token_id=_TRAP_PAD_ID,
            num_block_candidates=2,
            candidate_top_k=2,
            small_block_size=1,
            candidate_include_greedy=True,
        )
        self.mff = _make_mff(self.inner, device, cls=_SpyMFF)

    def sample(self) -> tuple[FastDLLMSampler, Any]:
        """Run one greedy-on speculative generation over the trap prompt."""
        sampler = FastDLLMSampler(self.llm, config=self.config)
        out = sampler.sample(
            _prompt(self.inner.model.embed_tokens.weight.device), folded_model=self.mff
        )
        return sampler, out

    def batched_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.mff.calls if c["kwargs"].get("branch_ids") is not None]

    def draft_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.mff.calls if c["kwargs"].get("branch_ids") is None]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_mini(
    device: str,
    head: str = "linear",
) -> _MiniBlockCausalModel:
    """Deterministically build the mini decoder on ``device``."""
    torch.manual_seed(_MODEL_SEED)
    model = _MiniBlockCausalModel(head=head)
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
        split_layers=False,
        split_min_tokens=512,
        use_cuda_graph=False,
        graph_capacity_ratio=0.5,
        **extra,
    )


def _recompute_min_scores(call: dict[str, Any], lo: int, hi: int) -> torch.Tensor:
    """Recompute per-candidate min commit probabilities from a spied batched call.

    UT-2109c precedent: right-shift the spied batched logits, softmax over
    the commit window, gather at the candidate tokens, take the per-row
    minimum — the frozen min-prob selection score.
    """
    cands = call["args"][0]
    shifted = right_shift_logits(call["out"])
    probs = F.softmax(shifted[:, lo:hi, :], dim=-1)
    committed = cands[:, lo:hi]
    scores = torch.gather(probs, -1, committed.unsqueeze(-1)).squeeze(-1)
    return scores.min(dim=-1).values


def _user_warning_messages(record: Any) -> list[str]:
    """Return the messages of every UserWarning in a catch_warnings log."""
    return [str(entry.message) for entry in record if issubclass(entry.category, UserWarning)]


def _prompt(device: str) -> torch.Tensor:
    return torch.tensor([[1, 2, 3]], device=device)


# ---------------------------------------------------------------------------
# UT-2201 (FR3): default-off zero disturbance
# ---------------------------------------------------------------------------


def test_ut2201_default_off_zero_disturbance(device: str) -> None:
    """UT-2201 (FR3): explicit greedy-off is bit-exact with the default config.

    ``candidate_include_greedy=False`` must reproduce the default (= AR021
    end state) outputs bit-exactly on the raw path, the N=0 face, the
    speculative N=4 folded face, and the stop-token scenario; the default-off
    ``greedy_win_stats`` probe must read {0, 0} (observation code untouched);
    the constant-mask head (argmax == mask_id) must complete without
    exception on both the greedy-on and baseline paths (family-consistent
    boundary, no special handling).
    """
    inner = _build_mini(device)
    llm = _make_llm(inner)
    prompt = _prompt(device)

    # (a) raw path: default vs explicit False — bit-exact.
    out_default = FastDLLMSampler(llm, config=_sampler_config(seed=2201)).sample(prompt)
    cfg_off = _sampler_config(seed=2201, candidate_include_greedy=False)
    out_off = FastDLLMSampler(llm, config=cfg_off).sample(prompt)
    assert torch.equal(out_off.sequences, out_default.sequences)

    # (b) N=0 + explicit False vs default — bit-exact.
    cfg_n0 = _sampler_config(seed=2201, num_block_candidates=0, candidate_include_greedy=False)
    out_n0 = FastDLLMSampler(llm, config=cfg_n0).sample(prompt)
    assert torch.equal(out_n0.sequences, out_default.sequences)

    # (c) speculative N=4, explicit False vs default N=4 (field default False).
    out_spec_default = FastDLLMSampler(
        llm, config=_sampler_config(seed=2201, num_block_candidates=4, candidate_top_k=2)
    ).sample(prompt, folded_model=_make_mff(inner, device, cls=_SpyMFF))
    cfg_spec_off = _sampler_config(
        seed=2201,
        num_block_candidates=4,
        candidate_top_k=2,
        candidate_include_greedy=False,
    )
    out_spec_off = FastDLLMSampler(llm, config=cfg_spec_off).sample(
        prompt, folded_model=_make_mff(inner, device, cls=_SpyMFF)
    )
    assert torch.equal(out_spec_off.sequences, out_spec_default.sequences)

    # (d) stop-token scenario (constant-stop head) — same face.
    inner_stop = _build_mini(device, head="constant")
    llm_stop = _make_llm(inner_stop)
    out_stop_default = FastDLLMSampler(llm_stop, config=_sampler_config(seed=2201)).sample(prompt)
    cfg_stop_off = _sampler_config(seed=2201, candidate_include_greedy=False)
    out_stop_off = FastDLLMSampler(llm_stop, config=cfg_stop_off).sample(prompt)
    assert torch.equal(out_stop_off.sequences, out_stop_default.sequences)

    # (e) default-off observation face: both counters stay 0 (raw and spec).
    sampler_off = FastDLLMSampler(llm, config=cfg_off)
    sampler_off.sample(prompt)
    assert sampler_off.greedy_win_stats == {"greedy_wins": 0, "speculative_calls": 0}
    sampler_spec_off = FastDLLMSampler(llm, config=cfg_spec_off)
    sampler_spec_off.sample(prompt, folded_model=_make_mff(inner, device))
    assert sampler_spec_off.greedy_win_stats == {
        "greedy_wins": 0,
        "speculative_calls": 0,
    }

    # (f) mask-token argmax boundary: greedy on and baseline both complete.
    inner_mask = _build_mini(device, head="constant_mask")
    llm_mask = _make_llm(inner_mask)
    out_mask_base = FastDLLMSampler(llm_mask, config=_sampler_config(seed=2201)).sample(prompt)
    assert out_mask_base.sequences.shape[0] == 1
    cfg_mask_on = _sampler_config(
        seed=2201,
        num_block_candidates=2,
        candidate_top_k=1,
        candidate_include_greedy=True,
    )
    out_mask_on = FastDLLMSampler(llm_mask, config=cfg_mask_on).sample(
        prompt, folded_model=_make_mff(inner_mask, device)
    )
    assert out_mask_on.sequences.shape[0] == 1


# ---------------------------------------------------------------------------
# UT-2202 (FR1): greedy row construction
# ---------------------------------------------------------------------------


def test_ut2202_greedy_row_construction(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2202 (FR1): row 0 is the deterministic draft argmax fill.

    Greedy on N=4 vs the same-seed greedy-off control: the ``[4, T]`` batch
    shape, draft parent, and ``branch_ids`` naming are unchanged; row 0
    equals the still-masked positions' ``topk(draft logits)[..., 0]``
    (recomputed from the spied draft forward); rows 1..3 equal the greedy-off
    run's rows 1..3 bit-for-bit (frozen seed formula with j keeping its row
    number — pinned to the canvas-consistent first speculative small block
    by the single-small-block fixture); row 0 is bit-exact across
    independent sampler instances; the candidate construction consumes N-1
    generators (off: N) and leaves the global RNG state untouched; and zero
    ``.item``/``.tolist``/``.cpu`` calls happen inside the construction
    window (``_ReadbackSpyMFF``).
    """
    inner = _build_mini(device, head="spike")
    llm = _make_llm(inner)
    prompt = _prompt(device)
    n_cands = 4
    k_eff = 2
    # Single-small-block premise: small_block_size == block_size and
    # num_tokens <= block_size => exactly one draft forward + one [N, T]
    # batched call (the canvas-consistent region, design §4.2.2(2)).
    cfg_on = _sampler_config(
        seed=2202,
        num_block_candidates=n_cands,
        candidate_top_k=k_eff,
        small_block_size=_BLOCK_SIZE,
        candidate_include_greedy=True,
    )

    mff_on = _make_mff(inner, device, cls=_SpyMFF)
    sampler_on = FastDLLMSampler(llm, config=cfg_on)
    real_generator = torch.Generator
    gen_on = {"n": 0}

    def _counting_generator(*args: Any, **kwargs: Any) -> Any:
        gen_on["n"] += 1
        return real_generator(*args, **kwargs)

    monkeypatch.setattr(torch, "Generator", _counting_generator)
    out_on = sampler_on.sample(prompt, folded_model=mff_on)
    monkeypatch.undo()
    assert gen_on["n"] == n_cands - 1, "greedy on must construct N-1 generators"

    assert len(mff_on.calls) == 2, "single small block: one draft + one batched call"
    serial_on = [c for c in mff_on.calls if c["kwargs"].get("branch_ids") is None]
    batched_on = [c for c in mff_on.calls if c["kwargs"].get("branch_ids") is not None]
    assert len(serial_on) == 1
    assert len(batched_on) == 1
    first = batched_on[0]
    cands_on = first["args"][0]
    seq_len = cands_on.shape[1]
    assert tuple(cands_on.shape) == (n_cands, seq_len), "[N, T] batch shape unchanged"
    draft_bid = serial_on[0]["kwargs"]["branch_id"]
    assert first["kwargs"]["parent_branch_id"] == draft_bid
    assert first["kwargs"]["branch_ids"] == [f"{draft_bid}_cand{j}" for j in range(n_cands)]

    # Row 0 == still-masked positions' topk(draft logits)[..., 0] (the whole
    # block is masked on the first small block, so the fill covers [lo, hi)).
    lo = seq_len - _BLOCK_SIZE
    hi = seq_len
    draft_small_logits = right_shift_logits(serial_on[0]["out"])[:, lo:hi, :]
    expected_row0 = torch.topk(draft_small_logits, k_eff, dim=-1).indices[..., 0]
    assert torch.equal(cands_on[0, lo:hi], expected_row0[0])

    # Rows 1..3 == the greedy-off same-seed run's rows 1..3 (A/B face).
    cfg_off = _sampler_config(
        seed=2202,
        num_block_candidates=n_cands,
        candidate_top_k=k_eff,
        small_block_size=_BLOCK_SIZE,
        candidate_include_greedy=False,
    )
    mff_off = _make_mff(inner, device, cls=_SpyMFF)
    gen_off = {"n": 0}

    def _counting_generator_off(*args: Any, **kwargs: Any) -> Any:
        gen_off["n"] += 1
        return real_generator(*args, **kwargs)

    monkeypatch.setattr(torch, "Generator", _counting_generator_off)
    FastDLLMSampler(llm, config=cfg_off).sample(prompt, folded_model=mff_off)
    monkeypatch.undo()
    assert gen_off["n"] == n_cands, "greedy off keeps N random rows"
    batched_off = [c for c in mff_off.calls if c["kwargs"].get("branch_ids") is not None]
    assert len(batched_off) == 1
    cands_off = batched_off[0]["args"][0]
    assert torch.equal(cands_on[1:], cands_off[1:])

    # Cross-instance determinism: an independent sampler reproduces row 0.
    mff_rerun = _make_mff(inner, device, cls=_SpyMFF)
    out_rerun = FastDLLMSampler(llm, config=cfg_on).sample(prompt, folded_model=mff_rerun)
    rerun_batched = [c for c in mff_rerun.calls if c["kwargs"].get("branch_ids") is not None]
    assert len(rerun_batched) == 1
    assert torch.equal(rerun_batched[0]["args"][0][0], cands_on[0])
    assert torch.equal(out_rerun.sequences, out_on.sequences)

    # Global RNG state snapshot: a greedy-on run with seed=None (no
    # manual_seed at sample entry) must leave the global RNG state unchanged
    # — the construction consumes only dedicated CPU generators.
    cfg_noglobal = _sampler_config(
        seed=None,
        num_block_candidates=n_cands,
        candidate_top_k=k_eff,
        small_block_size=_BLOCK_SIZE,
        candidate_include_greedy=True,
    )
    torch.manual_seed(97531)
    state_before = torch.get_rng_state()
    FastDLLMSampler(llm, config=cfg_noglobal).sample(prompt, folded_model=_make_mff(inner, device))
    assert torch.equal(torch.get_rng_state(), state_before)

    # Zero device readback inside the construction window.
    mff_rb = _make_mff(inner, device, cls=_ReadbackSpyMFF)
    try:
        FastDLLMSampler(llm, config=cfg_on).sample(prompt, folded_model=mff_rb)
    finally:
        mff_rb.disarm()
    assert not mff_rb.armed, "the batched call must have arrived and disarmed the spy"
    rb_batched = [c for c in mff_rb.calls if c["kwargs"].get("branch_ids") is not None]
    assert len(rb_batched) == 1


# ---------------------------------------------------------------------------
# UT-2203 (FR2): fair competition is non-degenerate
# ---------------------------------------------------------------------------


def test_ut2203_greedy_loses_to_random(device: str) -> None:
    """UT-2203 (FR2): the greedy row loses the min-prob competition.

    On the posterior-trap fixture the greedy row fills A (draft argmax) but
    the pinned random row fills C, whose posterior self-probability is far
    higher: every small block must select j*==1, commit the winner row (not
    the greedy row), and the recomputed min_scores must satisfy
    min_scores[0] < min_scores[1] — constructive evidence that the selection
    is a fair competition, never a degenerate always-pick-greedy.
    """
    fixture = _PosteriorTrapFixture(device, row1_draw=1)
    sampler, out = fixture.sample()

    batched = fixture.batched_calls()
    assert len(batched) == _TRAP_NUM_SMALL_BLOCKS
    for k, call in enumerate(batched):
        cands = call["args"][0]
        assert tuple(cands.shape) == (2, _TRAP_BLOCK_START + _TRAP_BLOCK_SIZE)
        pos = _TRAP_BLOCK_START + k
        assert int(cands[0, pos]) == _TRAP_GREEDY_ID, "greedy row fills draft argmax A"
        assert int(cands[1, pos]) == _TRAP_ALT_ID, "pinned random row draws C"
        min_scores = _recompute_min_scores(call, pos, pos + 1)
        assert float(min_scores[0]) < float(min_scores[1])
        assert int(min_scores.argmax()) == 1
        assert int(out.sequences[0, pos]) == _TRAP_ALT_ID, "commit == winner row"
        assert int(out.sequences[0, pos]) != int(cands[0, pos])

    stats = sampler.greedy_win_stats
    assert stats == {"greedy_wins": 0, "speculative_calls": _TRAP_NUM_SMALL_BLOCKS}


def test_ut2203_greedy_wins_commit(device: str) -> None:
    """UT-2203 (FR2): a winning greedy row commits the greedy hypothesis verbatim.

    The same trap fixture with the row-1 draw pinned to A twins the random
    row with the greedy row: the trivial tie selects row 0, the committed
    region equals the greedy hypothesis bit-for-bit, and greedy_wins equals
    speculative_calls.
    """
    fixture = _PosteriorTrapFixture(device, row1_draw=0)
    sampler, out = fixture.sample()

    batched = fixture.batched_calls()
    assert len(batched) == _TRAP_NUM_SMALL_BLOCKS
    for k, call in enumerate(batched):
        cands = call["args"][0]
        pos = _TRAP_BLOCK_START + k
        assert torch.equal(cands[0], cands[1]), "pinned twin row ties the greedy row"
        assert int(cands[0, pos]) == _TRAP_GREEDY_ID
        min_scores = _recompute_min_scores(call, pos, pos + 1)
        assert int(min_scores.argmax()) == 0
        assert int(out.sequences[0, pos]) == int(cands[0, pos])

    assert sampler.greedy_win_stats == {
        "greedy_wins": _TRAP_NUM_SMALL_BLOCKS,
        "speculative_calls": _TRAP_NUM_SMALL_BLOCKS,
    }


# ---------------------------------------------------------------------------
# UT-2204 (FR2): greedy_win_stats observation consistency
# ---------------------------------------------------------------------------


def test_ut2204_greedy_win_stats_consistency(device: str) -> None:
    """UT-2204 (FR2): the probe counts are self-consistent and cumulative.

    Across two ``sample`` calls (multi-small-block fixture):
    ``speculative_calls`` equals the total number of spied ``[N, T]`` batched
    calls; ``greedy_wins`` equals the sum of per-block j*==0 recomputed from
    the spied batched logits (UT-2109c style) and never exceeds
    speculative_calls; a greedy-off control run leaves both counters at 0;
    and the probe dict is read-only — external mutation cannot corrupt the
    internal counters, and assigning to the property raises AttributeError
    (no setter face).
    """
    inner = _build_mini(device, head="spike")
    llm = _make_llm(inner)
    prompt = _prompt(device)
    cfg_on = _sampler_config(
        seed=2204,
        num_block_candidates=4,
        candidate_top_k=2,
        candidate_include_greedy=True,
    )
    sampler = FastDLLMSampler(llm, config=cfg_on)
    mff = _make_mff(inner, device, cls=_SpyMFF)
    sampler.sample(prompt, folded_model=mff)
    sampler.sample(prompt, folded_model=mff)

    batched = [c for c in mff.calls if c["kwargs"].get("branch_ids") is not None]
    num_small_blocks = _BLOCK_SIZE // _SMALL_BLOCK_SIZE
    assert len(batched) == 2 * num_small_blocks, "two samples x two small blocks"

    wins = 0
    for idx, call in enumerate(batched):
        cands = call["args"][0]
        seq_len = cands.shape[1]
        sb = idx % num_small_blocks
        lo = seq_len - _BLOCK_SIZE + sb * _SMALL_BLOCK_SIZE
        hi = lo + _SMALL_BLOCK_SIZE
        min_scores = _recompute_min_scores(call, lo, hi)
        wins += int(int(min_scores.argmax()) == 0)

    stats = sampler.greedy_win_stats
    assert stats["speculative_calls"] == len(batched)
    assert stats["greedy_wins"] == wins
    assert stats["greedy_wins"] <= stats["speculative_calls"]

    # Read-only probe face: mutating the returned dict must not leak inside.
    original = dict(sampler.greedy_win_stats)
    probe = sampler.greedy_win_stats
    probe["greedy_wins"] = -1
    probe["speculative_calls"] = -1
    assert sampler.greedy_win_stats == original

    # Read-only probe face (assignment): the property has no setter — an
    # external write must raise AttributeError, never silently overwrite the
    # counters (a writable plain attribute would survive this probe).
    with pytest.raises(AttributeError):
        sampler.greedy_win_stats = {"greedy_wins": 99, "speculative_calls": 99}

    # Greedy-off control: both counters stay 0 across two sample calls.
    cfg_off = _sampler_config(
        seed=2204,
        num_block_candidates=4,
        candidate_top_k=2,
        candidate_include_greedy=False,
    )
    sampler_off = FastDLLMSampler(llm, config=cfg_off)
    mff_off = _make_mff(inner, device, cls=_SpyMFF)
    sampler_off.sample(prompt, folded_model=mff_off)
    sampler_off.sample(prompt, folded_model=mff_off)
    assert sampler_off.greedy_win_stats == {"greedy_wins": 0, "speculative_calls": 0}


# ---------------------------------------------------------------------------
# UT-2205 (FR2): stop semantics and chain continuation on both winner branches
# ---------------------------------------------------------------------------


def test_ut2205_stop_and_chain_semantics(device: str) -> None:
    """UT-2205 (FR2): stop semantics and chain continuation on both branches.

    Greedy-win + stop: the constant-stop head with ``candidate_top_k=1``
    makes every row identical (all stop), so j*==0 trivially; the
    finished/pad trimming must match the baseline bit-exactly and
    ``_folding_parent`` must be the greedy winner branch (``..._cand0``).
    Random-win: on the posterior-trap fixture (j*==1) the chain parent must
    switch to ``..._cand1`` and the next draft serial forward must chain
    onto that winner branch — both winner branches share one code path.
    """
    inner_stop = _build_mini(device, head="constant")
    llm_stop = _make_llm(inner_stop)
    prompt = _prompt(device)
    out_base = FastDLLMSampler(llm_stop, config=_sampler_config(seed=2205)).sample(prompt)
    cfg_stop_on = _sampler_config(
        seed=2205,
        num_block_candidates=2,
        candidate_top_k=1,
        candidate_include_greedy=True,
    )
    mff_stop = _make_mff(inner_stop, device, cls=_SpyMFF)
    sampler_stop = FastDLLMSampler(llm_stop, config=cfg_stop_on)
    out_spec = sampler_stop.sample(prompt, folded_model=mff_stop)
    assert torch.equal(out_spec.sequences, out_base.sequences)
    assert _STOP_ID in out_spec.sequences[0].tolist()
    stop_drafts = [c for c in mff_stop.calls if c["kwargs"].get("branch_ids") is None]
    assert len(stop_drafts) == _BLOCK_SIZE // _SMALL_BLOCK_SIZE
    assert sampler_stop._folding_parent == f"{stop_drafts[-1]['kwargs']['branch_id']}_cand0"
    assert sampler_stop.greedy_win_stats == {
        "greedy_wins": _BLOCK_SIZE // _SMALL_BLOCK_SIZE,
        "speculative_calls": _BLOCK_SIZE // _SMALL_BLOCK_SIZE,
    }

    # Random-win branch: the trap fixture selects row 1 at every small block.
    fixture = _PosteriorTrapFixture(device, row1_draw=1)
    sampler_trap, _out_trap = fixture.sample()
    trap_drafts = fixture.draft_calls()
    assert len(trap_drafts) == _TRAP_NUM_SMALL_BLOCKS
    first_winner = f"{trap_drafts[0]['kwargs']['branch_id']}_cand1"
    assert (
        trap_drafts[1]["kwargs"]["parent_branch_id"] == first_winner
    ), "the next draft forward must chain onto the random winner branch"
    assert sampler_trap._folding_parent == f"{trap_drafts[-1]['kwargs']['branch_id']}_cand1"


# ---------------------------------------------------------------------------
# UT-2206 (FR3): construction validation and configuration faces
# ---------------------------------------------------------------------------


def test_ut2206_config_validation_and_yaml(tmp_path: Any) -> None:
    """UT-2206 (FR3): True+N=0 ValueError, defaults, and YAML consumption.

    ``candidate_include_greedy=True, num_block_candidates=0`` must raise
    ValueError at construction with the frozen message (no silent
    degradation); True + N>0 is legal; the field defaults to False;
    ``dataclasses.replace`` preserves it (programmatic construction);
    ``ActFoldConfig`` consumes the key directly and via ``load_config`` with
    zero unknown-key UserWarning (AGENTS #5), defaulting to False when the
    YAML key is absent.
    """
    with pytest.raises(
        ValueError,
        match=r"candidate_include_greedy=True requires num_block_candidates > 0",
    ):
        _sampler_config(num_block_candidates=0, candidate_include_greedy=True)

    cfg_on = _sampler_config(num_block_candidates=2, candidate_include_greedy=True)
    assert cfg_on.candidate_include_greedy is True
    assert _sampler_config().candidate_include_greedy is False
    cfg_replaced = dataclasses.replace(cfg_on, seed=4321)
    assert cfg_replaced.candidate_include_greedy is True
    assert cfg_replaced.seed == 4321
    assert cfg_replaced.num_block_candidates == 2

    # ActFoldConfig direct construction face.
    cfg_direct = ActFoldConfig(candidate_include_greedy=True)
    assert cfg_direct.candidate_include_greedy is True

    # YAML load_config face: the key is consumed (zero UserWarning).
    cfg_file = tmp_path / "ar022_ut2206.yaml"
    cfg_file.write_text("candidate_include_greedy: true\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        loaded = load_config(cfg_file)
    assert _user_warning_messages(record) == []
    assert loaded.candidate_include_greedy is True

    # YAML default face: key absent -> False, still zero UserWarning.
    cfg_file.write_text("# empty config\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as record_default:
        warnings.simplefilter("always")
        loaded_default = load_config(cfg_file)
    assert _user_warning_messages(record_default) == []
    assert loaded_default.candidate_include_greedy is False


# ---------------------------------------------------------------------------
# UT-2207 (FR1): N=1 pure-greedy degeneration
# ---------------------------------------------------------------------------


def test_ut2207_n1_pure_greedy(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2207 (FR1): greedy on + N=1 is a legal single-row pure-greedy mode.

    Every small block makes exactly one ``[1, T]`` single-row batched call;
    the row equals the still-masked draft argmax fill (UT-2202 recompute);
    no random row is constructed (zero ``torch.Generator`` constructions);
    the selection is trivially j*==0; the chain continues onto
    ``..._cand0`` with the next draft forward parenting onto it; the run
    completes without exception; and ``greedy_win_stats`` reports
    speculative_calls == small-block count with greedy_wins == calls (the
    trivial-win annotation face).
    """
    inner = _build_mini(device, head="spike")
    llm = _make_llm(inner)
    prompt = _prompt(device)
    k_eff = 2
    cfg_on = _sampler_config(
        seed=2207,
        num_block_candidates=1,
        candidate_top_k=k_eff,
        candidate_include_greedy=True,
    )
    mff = _make_mff(inner, device, cls=_SpyMFF)
    sampler = FastDLLMSampler(llm, config=cfg_on)

    real_generator = torch.Generator
    gen_count = {"n": 0}

    def _counting_generator(*args: Any, **kwargs: Any) -> Any:
        gen_count["n"] += 1
        return real_generator(*args, **kwargs)

    monkeypatch.setattr(torch, "Generator", _counting_generator)
    out = sampler.sample(prompt, folded_model=mff)
    monkeypatch.undo()
    assert gen_count["n"] == 0, "range(1, 1) is empty: no random row, no generator"

    num_small_blocks = _BLOCK_SIZE // _SMALL_BLOCK_SIZE
    serial = [c for c in mff.calls if c["kwargs"].get("branch_ids") is None]
    batched = [c for c in mff.calls if c["kwargs"].get("branch_ids") is not None]
    assert len(serial) == num_small_blocks
    assert len(batched) == num_small_blocks

    for k, call in enumerate(batched):
        cands = call["args"][0]
        seq_len = cands.shape[1]
        assert tuple(cands.shape) == (1, seq_len), "single-row [1, T] batched call"
        lo = seq_len - _BLOCK_SIZE + k * _SMALL_BLOCK_SIZE
        hi = lo + _SMALL_BLOCK_SIZE
        draft_small_logits = right_shift_logits(serial[k]["out"])[:, lo:hi, :]
        expected_fill = torch.topk(draft_small_logits, k_eff, dim=-1).indices[..., 0]
        assert torch.equal(cands[0, lo:hi], expected_fill[0])
        min_scores = _recompute_min_scores(call, lo, hi)
        assert int(min_scores.argmax()) == 0, "single row: trivial selection"
        assert torch.equal(out.sequences[0, lo:hi], cands[0, lo:hi])

    first_draft_bid = serial[0]["kwargs"]["branch_id"]
    assert serial[1]["kwargs"]["parent_branch_id"] == f"{first_draft_bid}_cand0"
    assert sampler._folding_parent == f"{serial[-1]['kwargs']['branch_id']}_cand0"
    assert sampler.greedy_win_stats == {
        "greedy_wins": num_small_blocks,
        "speculative_calls": num_small_blocks,
    }
