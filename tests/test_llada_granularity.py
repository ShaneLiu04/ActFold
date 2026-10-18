"""Red-phase tests for AR024: LLaDA commit-granularity quality face (UT-2401..2413).

Contract: ``specs/changes/AR024-commit-granularity/design.md`` §4.1.2-4.1.7,
§6.1 and ``srs.md`` §3.1-3.4 — commit-granularity mitigation for the LLaDA
speculative path, ported on top of the AR023 shared-core state:

* UT-2401 (FR1 window iteration): blocks are split into ``s``-sized windows
  (remainder tail); per window one draft serial forward + one ``[N, T]``
  batched verification (counts == window count); per-window chain
  continuation (the next draft's parent is the previous window's winner
  branch); the window-region forced superset (True only over
  ``[w_start, w_end)`` — NOT the whole block); frozen branch tags
  (``w_idx == 0`` -> ``llada_spec_b{block}`` = the AR023 form; ``w_idx > 0``
  -> ``llada_spec_b{block}_w{w_idx}``); the frozen seed formula carries the
  window index in the ``sub_idx`` slot;
* UT-2402 (FR1 A/B anchor): ``s == block_size`` + eos off + refill off with
  all new fields explicitly off is BIT-IDENTICAL to the AR023-equivalent
  construction (outputs, branch naming, chain, history);
* UT-2403 (FR1 default zero disturbance): N=0 keeps the AR023 end state
  byte-identical (raw + folded faces) with the new fields at their defaults
  (s=8 / eos on); ``_refill_phase_two`` is never called and the window loop
  is never entered (no ``llada_spec_*`` branch tags on the folded N=0 face);
* UT-2404 (FR1/FR2 validation domain): ``s < 1`` always ValueError; N>0 +
  s > block_size (explicit) ValueError; N>0 + block_size=None + resolved
  block < s ValueError at the sample entry (before any forward);
  progressive_refill + N=0 ValueError; refill_top_k < 1 / > s ValueError
  (always — malformed values are path-independent); N=0 + small block_size
  + default s=8 triggers NO validation (the frozen reference-zero face);
* UT-2405 (FR2 two-phase refill): per window 3 forwards (draft + batched +
  phase-two ``_r`` serial); the top-k commit lands on the winner-row tokens
  at the highest draft-confidence positions; the remaining positions are
  filled with the phase-two argmax; phase two consumes zero generators;
  k == s degenerates to FR1 (no ``_r`` forward, bit-identical output);
  refill=False default is bit-identical to FR1;
* UT-2406 (FR2 k resolution): refill_top_k=None -> k = s // 2 (committed
  count at the phase-two entry); explicit k wins; k > window masked count
  -> runtime clamp to full commit + phase two skipped; exactly one extra
  forward per window (count upper bound);
* UT-2407 (FR3 hypothesis pair, EOS in the draft pool): row N-1 is the
  continuation row (first non-EOS pool entry, EOS-free) while row 0 keeps
  the raw greedy argmax (may contain EOS); rows 1..N-2 are bit-identical to
  the eos-off build (same seed, j keeps its row number); ``[N, T]`` shape
  unchanged; the continuation row competes under the unchanged min-prob
  rule and CAN LOSE;
* UT-2408 (FR3 degenerate + determinism): EOS not in the pool -> row N-1
  degenerates to the greedy value (argmax=0 path); N=1 + greedy -> the
  single row plays both roles and the selection matches the eos-off run
  exactly (the strict degenerate fixture); the build is deterministic
  (double-run bit-exact), zero device readback and zero generator
  consumption inside the construction window;
* UT-2409 (FR3 orthogonality + default): greedy off + eos on still builds
  the continuation row; ``eos_aware=None`` resolves through
  ``getattr(cfg, "candidate_eos_aware", False)`` (an attribute-less host =
  the Dream face -> eos off); the explicit param overrides the cfg; the
  default config (N>0, field unset) resolves eos ON (default-on face at the
  sampler level); a no-new-param call is byte-identical to the explicit
  ``eos_aware=False, refill_top_k=None`` call (Dream hosts unchanged);
* UT-2410 (FR4 config/YAML): the four new keys on ``LLaDASamplerConfig``
  and ``ActFoldConfig`` with symmetric defaults (8 / False / None / True);
  ``load_config`` consumes the keys with zero unknown-key UserWarning;
  ``llada_8b.yaml`` carries the AR024 comment section; Dream / Fast-dLLM
  configs gain NO new fields (the LLaDA N=0 bit-exact face lives in
  UT-2403, the Dream host byte face in UT-2409);
* UT-2411 (FR5 bench contract): ``scripts/ar024_granularity_bench`` exists
  with the module-level ``_ar023`` import-reuse face (helpers are the ar023
  objects, never re-implemented), the frozen four-mode granularity stacks
  (batched = the AR023 anchor s=block_size+eos off+greedy off; greedy = the
  AR024 stack s=8+eos on+greedy on) with constructive A/B isolation, the
  ``--refill-face`` separate ``_refill`` artifact (never the formal verdict
  path), the llada-only CLI family domain, and the meta
  ``granularity_constants`` keys;
* UT-2413 (FR1 window-level early stop, design gate G13): EOS committed in
  the second window of a block skips the remaining windows AND the later
  blocks (forward count == executed windows x 2), the unfinished positions
  stay masked and the decode face never trims; with refill on, the Core
  checks ``finished`` ONLY on the committed top-k positions (an EOS winner
  token at an uncommitted low-confidence position does NOT finish); a
  window without masked positions is skipped with zero forwards.

Red state (why every test fails today) — each failure is a production-
missing face, never a test-own bug:

* ``LLaDASamplerConfig`` has no ``speculative_sub_block_size`` /
  ``progressive_refill`` / ``refill_top_k`` / ``candidate_eos_aware``
  fields — every explicit kwarg pass fails with ``TypeError``, every
  default-value read with ``AttributeError`` (UT-2401..2406, UT-2409
  default face, UT-2410);
* ``SpeculativeCore._speculative_commit`` has no ``eos_aware`` /
  ``refill_top_k`` tail parameters — the Core-level faces fail with
  ``TypeError`` (UT-2407..2409, UT-2413 Core face);
* ``LLaDASampler._refill_phase_two`` does not exist and the window loop is
  not implemented (UT-2405/2406/2413 sampler faces — all reached only
  after the config TypeError);
* ``scripts/ar024_granularity_bench`` does not exist — UT-2411 fails with
  ``ModuleNotFoundError`` (imported lazily inside the test body so module
  collection always succeeds);
* ``llada_8b.yaml`` carries no AR024 section and ``load_config`` warns the
  four keys off as unknown (UT-2410).

Fixtures follow ``tests/test_llada_speculative.py`` (self-contained LLaMA-
layout mini bidirectional decoder, ``_MiniDiffusionLLM``, ``_SpyMFF``,
``_make_mff``, ``_ForwardRecorder``, the Core-host pattern) with the AR024
additions: the ``_EosRegionHead`` position-table head (EOS top-1 only
inside a position window) and the ``_AR024RefillSpySampler`` refill spy.
"""

from __future__ import annotations

import sys
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
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.models.base import DiffusionLLM
from actfold.models.dream_sampler import DreamSamplerConfig
from actfold.models.fast_dllm_sampler import FastDLLMSamplerConfig
from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from actfold.utils.config_manager import ActFoldConfig, load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


@pytest.fixture(autouse=True)
def _profiler_state() -> Iterator[None]:
    """Snapshot-reset the global stability profiler around every case."""
    GLOBAL_STABILITY_PROFILER.reset()
    yield
    GLOBAL_STABILITY_PROFILER.reset()


# ---------------------------------------------------------------------------
# Self-contained mini bidirectional decoder (AR023 fixture family)
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


class _PosTableHead(nn.Module):
    """Position-table head: top-1 at position p is ``(p+1) % V`` (graded).

    ``logits[:, p, (p+1) % V] = 3.0 + 0.01*p`` and
    ``logits[:, p, (p+2) % V] = 2.0 + 0.01*p`` — input-independent with a
    draft confidence that strictly increases with the position, so the
    refill top-k commit lands on the window's last positions.
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


class _EosRegionHead(nn.Module):
    """Position-table head whose top-1 is EOS only inside ``[lo, hi)``.

    Input-independent. Outside the region ``fill_id`` is the argmax (3.0)
    with token 6 runner-up (1.0); inside the region EOS is the argmax (3.0)
    with ``fill_id`` runner-up (1.0) — the forced-EOS-window (UT-2413) and
    EOS-in-pool (UT-2409 default face) fixtures.
    """

    def __init__(self, vocab: int, lo: int, hi: int, fill_id: int = 5) -> None:
        super().__init__()
        self.vocab = vocab
        self.lo = lo
        self.hi = hi
        self.fill_id = fill_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        rows = torch.arange(seq_len, device=x.device)
        logits = torch.zeros(*x.shape[:-1], self.vocab, dtype=x.dtype, device=x.device)
        inside = (rows >= self.lo) & (rows < self.hi)
        outside = ~inside
        logits[:, rows[outside], self.fill_id] = 3.0
        logits[:, rows[outside], 6] = 1.0
        logits[:, rows[inside], _EOS_ID] = 3.0
        logits[:, rows[inside], self.fill_id] = 1.0
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
        elif head == "postable":
            self.lm_head = _PosTableHead(vocab)
        elif head == "eos_tail":
            self.lm_head = _EosRegionHead(vocab, 6, 7)
        elif head == "eos_window":
            self.lm_head = _EosRegionHead(vocab, 5, 7)
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

    Input tensors are snapshotted (``detach().clone()``) at call time: the
    sampler mutates the canvas in place after serial forwards (the phase-two
    argmax fill), so a by-reference record would observe post-call state.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict[str, Any]] = []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = super().forward(*args, **kwargs)
        snap = tuple(a.detach().clone() if isinstance(a, torch.Tensor) else a for a in args)
        self.calls.append({"args": snap, "kwargs": kwargs, "out": out})
        return out


class _AR024RefillSpySampler(LLaDASampler):
    """LLaDASampler spy counting ``_refill_phase_two`` invocations (AR024)."""

    def __init__(self, model: DiffusionLLM, config: LLaDASamplerConfig | None = None) -> None:
        super().__init__(model, config)
        self.refill_calls = 0

    def _refill_phase_two(self, *args: Any, **kwargs: Any) -> None:
        self.refill_calls += 1
        return super()._refill_phase_two(*args, **kwargs)


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
    """LLaDASamplerConfig with pinned defaults (new AR024 fields via overrides)."""
    values: dict[str, Any] = {
        "num_steps": 8,
        "num_tokens": 8,
        "temperature": 0.0,
        "seed": 2400,
        "block_size": _BLOCK_SIZE,
    }
    values.update(overrides)
    return LLaDASamplerConfig(**values)


def _prompt(device: str) -> torch.Tensor:
    return torch.tensor([[1, 2, 3]], device=device)


def _split_calls(
    mff: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split spy calls into (drafts, batched, refill-serial) by their kwargs.

    Draft serial calls carry ``branch_id`` without the ``_r`` suffix and no
    ``branch_ids``; batched calls carry ``branch_ids``; the refill phase-two
    serial call carries a ``..._r`` branch tag (design §4.1.4).
    """
    drafts: list[dict[str, Any]] = []
    batched: list[dict[str, Any]] = []
    refills: list[dict[str, Any]] = []
    for call in mff.calls:
        kwargs = call["kwargs"]
        if kwargs.get("branch_ids") is not None:
            batched.append(call)
        elif isinstance(kwargs.get("branch_id"), str) and kwargs["branch_id"].endswith("_r"):
            refills.append(call)
        else:
            drafts.append(call)
    return drafts, batched, refills


class _ForwardRecorder:
    """Stub ``forward_fn`` recording every call and crafting batched logits.

    The frozen ``SpeculativeCore`` contract calls
    ``forward_fn(cands, branch_id=None, parent_branch_id=draft, branch_ids=...,
    forced_divergent=...)``; this recorder accepts anything, stores the
    arguments, and returns ``[N, T, V]`` logits whose winning row is fixed
    so the min-prob selection is draw-independent.
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


def _core_host(**cfg: Any) -> Any:
    """Build a minimal ``SpeculativeCore`` host (design §4.1.2 contract).

    The host carries the config namespace plus the counters and the chain
    slot. ``candidate_eos_aware`` is deliberately ABSENT unless explicitly
    requested — the attribute-less namespace is the Dream-host face whose
    ``getattr`` default must resolve to eos-off.
    """
    from actfold.models.speculative_core import SpeculativeCore  # noqa: PLC0415

    values: dict[str, Any] = {
        "num_block_candidates": 4,
        "candidate_top_k": 2,
        "candidate_include_greedy": True,
        "seed": 2400,
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


def _core_ctx(seq_len: int) -> Any:
    """An all-ones ``[T, T]`` SpecCtx (the neutral Core-level fold context)."""
    from actfold.models.speculative_core import SpecCtx  # noqa: PLC0415

    return SpecCtx(mask=torch.ones(seq_len, seq_len, dtype=torch.bool), position=None)


def _core_commit(
    host: Any,
    x: torch.Tensor,
    draft_region: torch.Tensor,
    masked_local: torch.Tensor,
    recorder: _ForwardRecorder,
    history: list[torch.Tensor],
    seq_len: int,
    **tail: Any,
) -> torch.Tensor:
    """Run one Core-level ``_speculative_commit`` with the frozen AR023 args.

    ``**tail`` forwards the AR024 parameters (``eos_aware`` /
    ``refill_top_k``) verbatim — an empty tail is the Dream-host no-new-param
    call. Returns the ``finished`` tensor for post-commit assertions.
    """
    finished = torch.zeros(1, dtype=torch.bool)
    host._speculative_commit(
        x=x,
        finished=finished,
        draft_logits_region=draft_region,
        region_start=3,
        masked_local=masked_local,
        draft_branch_id="spec_b0",
        ctx=_core_ctx(seq_len),
        unit_forced=torch.zeros(seq_len, dtype=torch.bool),
        folded_model=object(),
        unit_idx=0,
        sub_idx=0,
        history=history,
        forward_fn=recorder,
        stop_token_id=_EOS_ID,
        **tail,
    )
    return finished


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
        # The draw itself is host-side (frozen seed replay, deterministic);
        # gather on the region's device so the helper works on CPU and CUDA.
        choice = torch.randint(0, k, (num_masked,), generator=generator).to(draft_region.device)
        rows.append(torch.gather(draft_set, 1, choice.unsqueeze(1)).squeeze(1))
    return torch.stack(rows)


def _expected_continuation(draft_region: torch.Tensor, k: int) -> torch.Tensor:
    """The frozen continuation-row fill: the first non-EOS pool entry.

    ``pool != stop_token_id`` -> ``.to(torch.uint8).argmax(dim=1)`` -> gather
    — the pure-device-indexing formula of design §4.1.2 (an all-EOS pool
    degenerates to ``pool[:, 0]`` because argmax of zeros returns 0).
    """
    pool = torch.topk(draft_region, k, dim=-1).indices[0]  # [M, k]
    non_eos = pool != _EOS_ID
    first = non_eos.to(torch.uint8).argmax(dim=1)
    return pool.gather(1, first.unsqueeze(1)).squeeze(1)


def _user_warning_messages(record: Any) -> list[str]:
    """Return the messages of every UserWarning in a catch_warnings log."""
    return [str(entry.message) for entry in record if issubclass(entry.category, UserWarning)]


# ---------------------------------------------------------------------------
# UT-2401: sub-block window iteration (FR1)
# ---------------------------------------------------------------------------


def test_ut2401_window_iteration_counts(device: str) -> None:
    """UT-2401: blocks split into s-sized windows; one draft + one batched each.

    num_tokens=6 / block_size=3 gives two blocks; s=2 splits each block into
    windows of widths [2, 1] (the remainder tail), so the speculative path
    must make exactly 4 draft serial forwards and 4 ``[N, T]`` batched
    verifications — one pair per WINDOW, not per block — with no refill
    forwards (progressive_refill off) and every generated position committed.
    """
    inner = _build_mini(device, head="spike")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _llada_config(
        seed=2401,
        num_steps=2,
        num_tokens=6,
        block_size=3,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        candidate_eos_aware=False,
        speculative_sub_block_size=2,
    )
    mff = _make_mff(inner, device)
    out = LLaDASampler(llm, config=cfg).sample(prompt, folded_model=mff)

    drafts, batched, refills = _split_calls(mff)
    assert len(drafts) == 4, "one draft serial forward per WINDOW (2 per block)"
    assert len(batched) == 4, "one [N, T] batched verification per WINDOW"
    assert refills == [], "progressive_refill off: no phase-two forwards"
    assert len(mff.calls) == 8

    cands = batched[0]["args"][0]
    assert tuple(cands.shape) == (2, 9), "N=2 rows on the [N, T] window canvas"
    assert bool((out.sequences[0, 3:9] != _MASK_ID).all()), "the whole region is committed"


def test_ut2401_branch_tags_and_chain_continuation(device: str) -> None:
    """UT-2401: frozen window branch tags and the per-window winner chain.

    ``w_idx == 0`` keeps the AR023 tag ``llada_spec_b{block}``; ``w_idx > 0``
    is ``llada_spec_b{block}_w{w_idx}``. Each batched call's parent is the
    window's draft tag, the branch ids stay ``[f"{tag}_cand{j}"]``, every
    later draft's parent is the previous window's WINNER branch, and the
    final ``_folding_parent`` is the last window's winner.
    """
    inner = _build_mini(device, head="spike")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _llada_config(
        seed=2401,
        num_steps=2,
        num_tokens=8,
        block_size=4,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        candidate_eos_aware=False,
        speculative_sub_block_size=2,
    )
    mff = _make_mff(inner, device)
    sampler = LLaDASampler(llm, config=cfg)
    sampler.sample(prompt, folded_model=mff)

    drafts, batched, _ = _split_calls(mff)
    expected_tags = [
        "llada_spec_b0",
        "llada_spec_b0_w1",
        "llada_spec_b1",
        "llada_spec_b1_w1",
    ]
    tags = [c["kwargs"]["branch_id"] for c in drafts]
    assert tags == expected_tags, "frozen window tag scheme (w_idx=0 keeps the AR023 form)"

    for i, tag in enumerate(expected_tags):
        assert batched[i]["kwargs"]["parent_branch_id"] == tag
        assert batched[i]["kwargs"]["branch_ids"] == [f"{tag}_cand{j}" for j in range(2)]

    for i in (1, 2, 3):
        assert (
            drafts[i]["kwargs"]["parent_branch_id"] in batched[i - 1]["kwargs"]["branch_ids"]
        ), f"window {i} draft must chain onto window {i - 1}'s winner branch"

    assert (
        sampler._folding_parent in batched[3]["kwargs"]["branch_ids"]
    ), "the chain slot ends on the last window's winner"


def test_ut2401_window_forced_superset(device: str) -> None:
    """UT-2401: ``forced_divergent`` is the WINDOW-region superset.

    Each window's batched call carries an ``[N, T]`` forced tensor that is
    True exactly over ``[w_start, w_end)`` (the sub-block window, NOT the
    whole block) and False outside — the per-window conservative superset.
    """
    inner = _build_mini(device, head="spike")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _llada_config(
        seed=2401,
        num_steps=2,
        num_tokens=8,
        block_size=4,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        candidate_eos_aware=False,
        speculative_sub_block_size=2,
    )
    mff = _make_mff(inner, device)
    LLaDASampler(llm, config=cfg).sample(prompt, folded_model=mff)

    _, batched, _ = _split_calls(mff)
    spans = [(3, 5), (5, 7), (7, 9), (9, 11)]
    for i, (lo, hi) in enumerate(spans):
        forced = batched[i]["kwargs"]["forced_divergent"]
        assert tuple(forced.shape) == (2, 11)
        assert bool(forced[:, lo:hi].all()), f"window {i}: forced covers [{lo}, {hi})"
        assert int(forced[:, :lo].sum()) == 0, f"window {i}: forced False before the window"
        assert int(forced[:, hi:].sum()) == 0, f"window {i}: forced False after the window"


def test_ut2401_seed_subidx_is_window_index(device: str) -> None:
    """UT-2401: the frozen seed formula carries the window index in sub_idx.

    For window i the random rows must replay with ``unit_idx = block_idx``
    and ``sub_idx = w_idx`` (the AR023 constants otherwise unchanged), so a
    production loop that keeps ``sub_idx=0`` for every window, swaps the
    index slots, or re-seeds per block produces different draws and fails
    the bit-exact comparison.
    """
    inner = _build_mini(device, head="spike")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _llada_config(
        seed=2401,
        num_steps=2,
        num_tokens=8,
        block_size=4,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        candidate_eos_aware=False,
        speculative_sub_block_size=2,
    )
    mff = _make_mff(inner, device)
    LLaDASampler(llm, config=cfg).sample(prompt, folded_model=mff)

    drafts, batched, _ = _split_calls(mff)
    spans = [(3, 5), (5, 7), (7, 9), (9, 11)]
    for i, (lo, hi) in enumerate(spans):
        block_idx, w_idx = divmod(i, 2)
        region = drafts[i]["out"][:, lo:hi, :]
        expected = _replay_random_rows(region, 2401, block_idx, w_idx, 2, 2)
        cands = batched[i]["args"][0]
        assert torch.equal(
            cands[1, lo:hi], expected[1]
        ), f"window {i}: random row must replay with sub_idx={w_idx}"


# ---------------------------------------------------------------------------
# UT-2402: FR1 A/B anchor (bit-exact AR023 equivalence)
# ---------------------------------------------------------------------------


def test_ut2402_ab_anchor_bit_exact(device: str) -> None:
    """UT-2402: s == block_size + eos off + refill off == AR023 bit-exact.

    With every AR024 field explicitly neutralized
    (``speculative_sub_block_size=4 == block_size``,
    ``candidate_eos_aware=False``, ``progressive_refill=False``,
    ``refill_top_k=None``) the whole sample — sequences, history, branch
    naming, batched branch ids, and the final chain slot — must be
    bit-identical to the AR023-equivalent construction (one window per
    block, unchanged commit semantics).
    """
    inner = _build_mini(device, head="spike")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    base = dict(
        seed=2402,
        num_steps=2,
        num_tokens=8,
        block_size=4,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        return_history=True,
    )

    mff_a = _make_mff(inner, device)
    sampler_a = LLaDASampler(llm, config=_llada_config(**base))
    out_a = sampler_a.sample(prompt, folded_model=mff_a)

    mff_b = _make_mff(inner, device)
    sampler_b = LLaDASampler(
        llm,
        config=_llada_config(
            **base,
            speculative_sub_block_size=4,
            candidate_eos_aware=False,
            progressive_refill=False,
            refill_top_k=None,
        ),
    )
    out_b = sampler_b.sample(prompt, folded_model=mff_b)

    assert torch.equal(out_b.sequences, out_a.sequences), "anchor: sequences bit-exact"
    assert len(out_b.history) == len(out_a.history), "anchor: same history length"
    for step_b, step_a in zip(out_b.history, out_a.history):
        assert torch.equal(step_b, step_a), "anchor: history snapshots bit-exact"

    drafts_a, batched_a, _ = _split_calls(mff_a)
    drafts_b, batched_b, _ = _split_calls(mff_b)
    tags_a = [c["kwargs"]["branch_id"] for c in drafts_a]
    tags_b = [c["kwargs"]["branch_id"] for c in drafts_b]
    assert tags_a == ["llada_spec_b0", "llada_spec_b1"], "AR023 face: one window per block"
    assert tags_b == tags_a, "anchor: s == block_size keeps the AR023 branch naming"
    assert [c["kwargs"]["branch_ids"] for c in batched_b] == [
        c["kwargs"]["branch_ids"] for c in batched_a
    ], "anchor: batched branch ids bit-exact"
    assert sampler_b._folding_parent == sampler_a._folding_parent, "anchor: chain slot equal"


# ---------------------------------------------------------------------------
# UT-2403: default zero disturbance (N=0)
# ---------------------------------------------------------------------------


def test_ut2403_default_zero_disturbance_n0(device: str) -> None:
    """UT-2403: the AR024 defaults keep the N=0 end state byte-identical.

    ``num_block_candidates=0`` with the new fields at their defaults
    (``speculative_sub_block_size=8``, ``candidate_eos_aware=True``,
    ``progressive_refill=False``, ``refill_top_k=None``) must keep both the
    raw and the folded N=0 faces bit-exact against the AR023 construction:
    ``_refill_phase_two`` is never called and no window-loop ``llada_spec``
    branch tag appears on the folded face.
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg_ar024 = dict(
        seed=2403,
        num_block_candidates=0,
        speculative_sub_block_size=8,
        candidate_eos_aware=True,
        progressive_refill=False,
        refill_top_k=None,
    )

    out_ref = LLaDASampler(llm, config=_llada_config(seed=2403, num_block_candidates=0)).sample(
        prompt
    )
    sampler_raw = _AR024RefillSpySampler(llm, config=_llada_config(**cfg_ar024))
    out_new = sampler_raw.sample(prompt)
    assert torch.equal(out_new.sequences, out_ref.sequences), "raw N=0 face bit-exact"
    assert sampler_raw.refill_calls == 0, "refill phase two never fires on N=0"

    mff_ref = _make_mff(inner, device)
    out_ref_folded = LLaDASampler(
        llm, config=_llada_config(seed=2403, num_block_candidates=0)
    ).sample(prompt, folded_model=mff_ref)
    mff_new = _make_mff(inner, device)
    sampler_folded = _AR024RefillSpySampler(llm, config=_llada_config(**cfg_ar024))
    out_new_folded = sampler_folded.sample(prompt, folded_model=mff_new)
    assert torch.equal(
        out_new_folded.sequences, out_ref_folded.sequences
    ), "folded N=0 face bit-exact"
    assert sampler_folded.refill_calls == 0, "refill phase two never fires on N=0"

    for call in mff_new.calls:
        branch_id = call["kwargs"].get("branch_id")
        assert not (
            isinstance(branch_id, str) and "llada_spec" in branch_id
        ), "N=0 never enters the speculative window loop"


# ---------------------------------------------------------------------------
# UT-2404: validation domain (static + runtime-resolved)
# ---------------------------------------------------------------------------


def test_ut2404_static_validation_domain(device: str) -> None:
    """UT-2404: malformed granularity values raise ValueError at construction.

    The static domain (config-level, path-independent): ``s < 1``; N>0 with
    an explicit ``block_size < s``; ``progressive_refill`` with N=0;
    ``refill_top_k < 1`` or ``> s`` (always, even when no refill path could
    run). The frozen no-raise face — N=0 with a small block and the default
    ``s=8`` — must sample cleanly (the defaults never disturb N=0).
    """
    inner = _build_mini(device)
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)

    def _config(**overrides: Any) -> LLaDASamplerConfig:
        return _llada_config(**overrides)

    with pytest.raises(ValueError, match=r"speculative_sub_block_size"):
        _config(speculative_sub_block_size=0)
    with pytest.raises(ValueError, match=r"speculative_sub_block_size"):
        _config(
            num_block_candidates=2,
            block_size=4,
            speculative_sub_block_size=8,
        )
    with pytest.raises(ValueError, match=r"progressive_refill"):
        _config(num_block_candidates=0, progressive_refill=True)
    with pytest.raises(ValueError, match=r"refill_top_k"):
        _config(num_block_candidates=2, block_size=8, refill_top_k=0)
    with pytest.raises(ValueError, match=r"refill_top_k"):
        _config(
            num_block_candidates=2,
            block_size=8,
            speculative_sub_block_size=4,
            progressive_refill=True,
            refill_top_k=5,
        )
    with pytest.raises(ValueError, match=r"refill_top_k"):
        _config(num_block_candidates=0, refill_top_k=0)

    out = LLaDASampler(
        llm,
        config=_llada_config(
            seed=2404, num_steps=2, num_tokens=4, block_size=4, speculative_sub_block_size=8
        ),
    ).sample(prompt)
    assert tuple(out.sequences.shape) == (1, 7), "N=0 + small block + default s: no raise"


def test_ut2404_runtime_resolved_block_validation(device: str) -> None:
    """UT-2404: a runtime-resolved block smaller than s raises before forwards.

    With ``block_size=None`` the block is resolved at the sample entry from
    the generation geometry (``num_tokens``); N=2 with s=8 over a resolved
    block of 4 must raise ``ValueError`` at the entry — before ANY forward
    call reaches the model.
    """
    inner = _build_mini(device, head="spike")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _llada_config(
        seed=2404,
        num_steps=2,
        num_tokens=4,
        block_size=None,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        speculative_sub_block_size=8,
    )
    mff = _make_mff(inner, device)
    with pytest.raises(ValueError, match=r"speculative_sub_block_size"):
        LLaDASampler(llm, config=cfg).sample(prompt, folded_model=mff)
    assert len(mff.calls) == 0, "the runtime validation fires before any folded forward"
    assert len(llm.raw_calls) == 0, "the runtime validation fires before any raw forward"


# ---------------------------------------------------------------------------
# UT-2405: progressive refill — two-phase flow (FR2)
# ---------------------------------------------------------------------------


def test_ut2405_refill_two_phase_flow(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2405: per window draft + batched + phase-two ``_r`` serial refill.

    On the position-table head (confidence graded by position) with
    ``refill_top_k=2`` over a 4-token window: the top-2 commit lands on the
    two highest-confidence positions (the window tail) with the winner-row
    tokens; the phase-two ``..._r`` serial forward (parent = the winner
    branch) sees exactly those committed and the rest still masked, and its
    argmax fills the remaining positions; the phase two consumes zero
    generators; the history gains exactly one snapshot for the window and
    the chain slot switches to the refill branch.
    """
    generator_calls: list[Any] = []
    real_generator = torch.Generator

    def _counting_generator(*args: Any, **kwargs: Any) -> Any:
        generator_calls.append((args, kwargs))
        return real_generator(*args, **kwargs)

    monkeypatch.setattr(torch, "Generator", _counting_generator)

    inner = _build_mini(device, head="postable")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _llada_config(
        seed=2405,
        num_steps=2,
        num_tokens=4,
        block_size=4,
        num_block_candidates=1,
        candidate_top_k=2,
        candidate_include_greedy=True,
        candidate_eos_aware=False,
        return_history=True,
        speculative_sub_block_size=4,
        progressive_refill=True,
        refill_top_k=2,
    )
    mff = _make_mff(inner, device)
    sampler = _AR024RefillSpySampler(llm, config=cfg)
    out = sampler.sample(prompt, folded_model=mff)

    drafts, batched, refills = _split_calls(mff)
    assert len(drafts) == 1 and len(batched) == 1 and len(refills) == 1
    assert len(mff.calls) == 3, "exactly one extra forward per window (phase two)"
    assert sampler.refill_calls == 1

    draft_bid = drafts[0]["kwargs"]["branch_id"]
    assert draft_bid == "llada_spec_b0"
    refill_bid = refills[0]["kwargs"]["branch_id"]
    assert refill_bid == "llada_spec_b0_w0_r", "frozen phase-two tag"
    assert (
        refills[0]["kwargs"]["parent_branch_id"] == "llada_spec_b0_cand0"
    ), "phase two chains onto the winner branch"

    winner_tokens = torch.tensor([(p + 1) % _VOCAB for p in range(3, 7)], device=device)
    region = drafts[0]["out"][:, 3:7, :]
    winner_conf = F.softmax(region, dim=-1)[0, torch.arange(4, device=device), winner_tokens]
    top_conf_local = torch.topk(winner_conf, 2).indices.tolist()
    assert top_conf_local == [3, 2], "fixture check: confidence is graded by position"

    r_input = refills[0]["args"][0]
    assert tuple(r_input.shape) == (1, 7), "phase two is a batch-1 serial forward"
    assert bool((r_input[0, 5:7] != _MASK_ID).all()), "the top-2 positions are committed"
    assert torch.equal(r_input[0, 5:7], winner_tokens[2:4]), "committed == winner tokens"
    assert bool((r_input[0, 3:5] == _MASK_ID).all()), "the remaining positions stay masked"

    refill_logits = refills[0]["out"]
    remaining_fill = torch.topk(refill_logits[0, 3:5, :], 1, dim=-1).indices[..., 0]
    assert torch.equal(out.sequences[0, 3:5], remaining_fill), "remaining == phase-two argmax"
    assert torch.equal(out.sequences[0, 3:7], winner_tokens), "the full window is committed"

    assert len(out.history) == 2, "one history snapshot per window (Core never appends)"
    assert torch.equal(out.history[-1], out.sequences), "the last snapshot is the commit"
    assert sampler._folding_parent == "llada_spec_b0_w0_r", "chain slot switches to _r"

    assert generator_calls == [], "phase two is argmax-only: zero generator consumption"


def test_ut2405_refill_degenerate_k_eq_s(device: str) -> None:
    """UT-2405: ``refill_top_k == s`` degenerates to the FR1 full commit.

    With k == s every window position is committed in phase one, so the
    phase-two forward is skipped entirely (2 calls: draft + batched) and the
    output is bit-identical to the explicit refill-off run; the default
    config (no refill kwargs at all) is bit-identical to both.
    """
    inner = _build_mini(device, head="postable")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    base = dict(
        seed=2405,
        num_steps=2,
        num_tokens=4,
        block_size=4,
        num_block_candidates=1,
        candidate_top_k=2,
        candidate_include_greedy=True,
        candidate_eos_aware=False,
        speculative_sub_block_size=4,
    )

    mff_fr1 = _make_mff(inner, device)
    out_fr1 = LLaDASampler(
        llm,
        config=_llada_config(**base, progressive_refill=False, refill_top_k=None),
    ).sample(prompt, folded_model=mff_fr1)

    mff_k_eq_s = _make_mff(inner, device)
    out_k_eq_s = LLaDASampler(
        llm,
        config=_llada_config(**base, progressive_refill=True, refill_top_k=4),
    ).sample(prompt, folded_model=mff_k_eq_s)
    _, _, refills = _split_calls(mff_k_eq_s)
    assert len(mff_k_eq_s.calls) == 2, "k == s: full commit, no phase-two forward"
    assert refills == []
    assert torch.equal(out_k_eq_s.sequences, out_fr1.sequences), "k == s == FR1 bit-exact"

    mff_default = _make_mff(inner, device)
    out_default = LLaDASampler(llm, config=_llada_config(**base)).sample(
        prompt, folded_model=mff_default
    )
    assert torch.equal(out_default.sequences, out_fr1.sequences), "default == FR1 bit-exact"


# ---------------------------------------------------------------------------
# UT-2406: refill top-k resolution (FR2)
# ---------------------------------------------------------------------------


def test_ut2406_k_resolution(device: str) -> None:
    """UT-2406: k = s // 2 default, explicit k wins, runtime clamp to full.

    ``refill_top_k=None`` resolves to ``s // 2`` committed positions at the
    phase-two entry; an explicit ``refill_top_k=1`` commits exactly one; a
    k larger than the window's masked count clamps to a full commit and
    skips phase two (the remainder-tail window of width 1); and the
    per-window extra-forward bound holds (exactly one ``_r`` forward per
    refilled window).
    """
    inner = _build_mini(device, head="postable")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)

    def _run(
        sub_block: int, refill_top_k: int | None, num_tokens: int, block_size: int
    ) -> tuple[ManualFoldedForward, torch.Tensor]:
        mff = _make_mff(inner, device)
        cfg = _llada_config(
            seed=2406,
            num_steps=2,
            num_tokens=num_tokens,
            block_size=block_size,
            num_block_candidates=1,
            candidate_top_k=2,
            candidate_include_greedy=True,
            candidate_eos_aware=False,
            speculative_sub_block_size=sub_block,
            progressive_refill=True,
            refill_top_k=refill_top_k,
        )
        out = LLaDASampler(llm, config=cfg).sample(prompt, folded_model=mff)
        return mff, out

    # Default k = s // 2 = 2: the two highest-confidence positions committed.
    mff_a, _ = _run(4, None, 4, 4)
    _, _, refills_a = _split_calls(mff_a)
    assert len(refills_a) == 1
    r_input = refills_a[0]["args"][0]
    assert int((r_input[0, 3:7] != _MASK_ID).sum()) == 2, "k=None -> s//2 committed"

    # Explicit k = 1: exactly one position committed.
    mff_b, _ = _run(4, 1, 4, 4)
    _, _, refills_b = _split_calls(mff_b)
    assert len(refills_b) == 1
    r_input_b = refills_b[0]["args"][0]
    assert int((r_input_b[0, 3:7] != _MASK_ID).sum()) == 1, "explicit k=1 commits one"

    # k = 2 over remainder-tail windows of widths [2, 1]: every window
    # clamps to a full commit — no phase-two forward at all.
    mff_c, out_c = _run(2, 2, 6, 3)
    drafts_c, batched_c, refills_c = _split_calls(mff_c)
    assert len(drafts_c) == 4 and len(batched_c) == 4
    assert refills_c == [], "k >= masked count: clamp to full commit, phase two skipped"
    assert bool((out_c.sequences[0, 3:9] != _MASK_ID).all()), "clamped windows fully commit"

    # Two s-sized windows per block, k < s on both: exactly one _r per window.
    mff_d, _ = _run(4, 2, 8, 8)
    drafts_d, batched_d, refills_d = _split_calls(mff_d)
    assert len(drafts_d) == 2 and len(batched_d) == 2 and len(refills_d) == 2
    assert len(mff_d.calls) == 6, "one extra forward per refilled window"


# ---------------------------------------------------------------------------
# UT-2407: EOS-aware continuation row — hypothesis pair (FR3)
# ---------------------------------------------------------------------------


def test_ut2407_eos_aware_hypothesis_pair() -> None:
    """UT-2407: row N-1 becomes the EOS-free continuation; rows 1..N-2 unchanged.

    Hypothesis pair (EOS in the draft pool): with ``eos_aware=True`` the
    greedy row 0 keeps the raw argmax (EOS at the first position), row N-1
    is rebuilt as the first non-EOS pool entry per position (EOS-free),
    rows 1..N-2 are bit-identical to the eos-off build (same seed, ``j``
    keeps its row number), the ``[N, T]`` shape is unchanged, and the
    continuation row competes under the unchanged min-prob rule — here the
    random row 1 WINS and the commit takes its fill.
    """
    x = torch.tensor([[1, 2, 3, _MASK_ID, _MASK_ID, 7]])
    draft = torch.zeros(1, 2, _VOCAB)
    draft[0, 0, _EOS_ID] = 3.0
    draft[0, 0, 5] = 2.0
    draft[0, 1, 7] = 3.0
    draft[0, 1, _EOS_ID] = 2.0
    masked_local = torch.tensor([0, 1])

    host_on = _core_host()
    recorder_on = _ForwardRecorder(_VOCAB, winner_row=1)
    x_on = x.clone()
    _core_commit(host_on, x_on, draft, masked_local, recorder_on, [], 6, eos_aware=True)
    cands_on = recorder_on.calls[0]["tokens"]
    assert tuple(cands_on.shape) == (4, 6), "[N, T] canvas shape unchanged"
    assert torch.equal(
        cands_on[0, 3:5], torch.tensor([_EOS_ID, 7])
    ), "row 0 keeps the raw greedy argmax (EOS allowed)"
    expected = _expected_continuation(draft, 2)
    assert torch.equal(cands_on[3, 3:5], expected), "row N-1 == first non-EOS pool entry"
    assert not bool((cands_on[3, 3:5] == _EOS_ID).any()), "row N-1 is EOS-free"

    host_off = _core_host()
    recorder_off = _ForwardRecorder(_VOCAB, winner_row=1)
    x_off = x.clone()
    _core_commit(host_off, x_off, draft, masked_local, recorder_off, [], 6)
    cands_off = recorder_off.calls[0]["tokens"]
    for j in (1, 2):
        assert torch.equal(
            cands_on[j, 3:5], cands_off[j, 3:5]
        ), f"row {j} bit-identical to the eos-off build (j keeps its row number)"

    assert torch.equal(
        x_on[0, 3:5], cands_on[1, 3:5]
    ), "the continuation row LOSES: the commit takes winner row 1's fill"
    assert host_on._folding_parent == "spec_b0_cand1"


# ---------------------------------------------------------------------------
# UT-2408: EOS-aware degenerate faces + determinism (FR3)
# ---------------------------------------------------------------------------


def test_ut2408_eos_not_in_pool_degenerate_faces() -> None:
    """UT-2408: EOS absent from the pool -> the continuation degenerates.

    With no EOS in the draft pools the first-non-EOS index is 0 everywhere,
    so row N-1 equals the greedy fill (a): row N-1 == row 0; with N=1 the
    single row plays both roles and the eos-on run matches the eos-off run
    exactly on the canvas AND the commit (b); and the construction is
    deterministic — a double run is bit-exact (c).
    """
    x0 = torch.tensor([[1, 2, 3, _MASK_ID, _MASK_ID, 7]])
    draft = torch.zeros(1, 2, _VOCAB)
    draft[0, :, 5] = 3.0
    draft[0, :, 6] = 1.0
    masked_local = torch.tensor([0, 1])

    # (a) N=2, greedy on: row N-1 degenerates to the greedy value.
    host_a = _core_host(num_block_candidates=2)
    recorder_a = _ForwardRecorder(_VOCAB, winner_row=0)
    _core_commit(host_a, x0.clone(), draft, masked_local, recorder_a, [], 6, eos_aware=True)
    cands_a = recorder_a.calls[0]["tokens"]
    assert torch.equal(
        cands_a[1, 3:5], cands_a[0, 3:5]
    ), "row N-1 degenerates to the greedy fill when EOS is not in the pool"

    # (b) N=1: the single row plays both roles; eos on == eos off exactly.
    host_on = _core_host(num_block_candidates=1)
    recorder_on = _ForwardRecorder(_VOCAB, winner_row=0)
    x_on = x0.clone()
    _core_commit(host_on, x_on, draft, masked_local, recorder_on, [], 6, eos_aware=True)
    host_off = _core_host(num_block_candidates=1)
    recorder_off = _ForwardRecorder(_VOCAB, winner_row=0)
    x_off = x0.clone()
    _core_commit(host_off, x_off, draft, masked_local, recorder_off, [], 6)
    assert torch.equal(
        recorder_on.calls[0]["tokens"], recorder_off.calls[0]["tokens"]
    ), "N=1: eos on == eos off on the canvas"
    assert torch.equal(x_on[0, 3:5], x_off[0, 3:5]), "N=1: eos on == eos off on the commit"

    # (c) Determinism: a double run is bit-exact.
    host_c = _core_host(num_block_candidates=2)
    recorder_c1 = _ForwardRecorder(_VOCAB, winner_row=0)
    _core_commit(host_c, x0.clone(), draft, masked_local, recorder_c1, [], 6, eos_aware=True)
    host_d = _core_host(num_block_candidates=2)
    recorder_c2 = _ForwardRecorder(_VOCAB, winner_row=0)
    _core_commit(host_d, x0.clone(), draft, masked_local, recorder_c2, [], 6, eos_aware=True)
    assert torch.equal(
        recorder_c1.calls[0]["tokens"], recorder_c2.calls[0]["tokens"]
    ), "the eos-aware construction is deterministic"


def test_ut2408_construction_determinism_zero_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-2408: the continuation build is pure device indexing, zero cost.

    Building the continuation row replaces the ``j = N-1`` random draw, so
    the eos-aware N=2 greedy commit consumes ZERO generators while the
    eos-off twin consumes exactly one; and inside the eos-aware commit
    window no ``.item()`` / ``.tolist()`` / ``.cpu()`` device readback
    happens (pure device indexing, deterministic construction).
    """
    x0 = torch.tensor([[1, 2, 3, _MASK_ID, _MASK_ID, 7]])
    draft = torch.zeros(1, 2, _VOCAB)
    draft[0, 0, _EOS_ID] = 3.0
    draft[0, 0, 5] = 2.0
    draft[0, 1, 7] = 3.0
    draft[0, 1, _EOS_ID] = 2.0
    masked_local = torch.tensor([0, 1])

    generator_calls: list[Any] = []
    real_generator = torch.Generator

    def _counting_generator(*args: Any, **kwargs: Any) -> Any:
        generator_calls.append((args, kwargs))
        return real_generator(*args, **kwargs)

    monkeypatch.setattr(torch, "Generator", _counting_generator)

    host_on = _core_host(num_block_candidates=2)
    recorder_on = _ForwardRecorder(_VOCAB, winner_row=0)
    _core_commit(host_on, x0.clone(), draft, masked_local, recorder_on, [], 6, eos_aware=True)
    assert generator_calls == [], "eos-aware N=2 greedy: zero generator consumption"

    generator_calls.clear()
    host_off = _core_host(num_block_candidates=2)
    recorder_off = _ForwardRecorder(_VOCAB, winner_row=0)
    _core_commit(host_off, x0.clone(), draft, masked_local, recorder_off, [], 6)
    assert len(generator_calls) == 1, "eos-off N=2 greedy: exactly the j=1 draw"

    readback: list[str] = []
    readback_names = ("item", "tolist", "cpu")
    originals = {name: getattr(torch.Tensor, name) for name in readback_names}

    def _make_spy(name: str) -> Any:
        def _spy(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            readback.append(name)
            return originals[name](self, *args, **kwargs)

        return _spy

    for name in readback_names:
        monkeypatch.setattr(torch.Tensor, name, _make_spy(name))

    host_spy = _core_host(num_block_candidates=2)
    recorder_spy = _ForwardRecorder(_VOCAB, winner_row=0)
    _core_commit(host_spy, x0.clone(), draft, masked_local, recorder_spy, [], 6, eos_aware=True)
    assert readback == [], "the eos-aware build performs zero device readback"


# ---------------------------------------------------------------------------
# UT-2409: EOS-aware orthogonality + default resolution (FR3)
# ---------------------------------------------------------------------------


def test_ut2409_orthogonality_and_default_resolution() -> None:
    """UT-2409: eos-aware composes with greedy-off; None resolves via cfg.

    (a) greedy off + eos on still builds the continuation row (the row is
    defined by row index, not by greedy anchoring); (b) an omitted
    ``eos_aware`` resolves through ``getattr(cfg, "candidate_eos_aware",
    False)`` — a cfg carrying ``candidate_eos_aware=True`` turns the
    no-param call on; (c) the explicit parameter overrides the cfg value;
    (d) an attribute-less host (the Dream face) resolves eos OFF and the
    no-new-param call is byte-identical to the explicit
    ``eos_aware=False, refill_top_k=None`` call.
    """
    x0 = torch.tensor([[1, 2, 3, _MASK_ID, _MASK_ID, 7]])
    draft = torch.zeros(1, 2, _VOCAB)
    draft[0, 0, _EOS_ID] = 3.0
    draft[0, 0, 5] = 2.0
    draft[0, 1, 7] = 3.0
    draft[0, 1, _EOS_ID] = 2.0
    masked_local = torch.tensor([0, 1])
    expected = _expected_continuation(draft, 2)

    # (a) greedy off + eos on: row N-1 is still the continuation row.
    host_a = _core_host(num_block_candidates=2, candidate_include_greedy=False)
    recorder_a = _ForwardRecorder(_VOCAB, winner_row=0)
    _core_commit(host_a, x0.clone(), draft, masked_local, recorder_a, [], 6, eos_aware=True)
    cands_a = recorder_a.calls[0]["tokens"]
    assert torch.equal(cands_a[1, 3:5], expected), "greedy-off: row N-1 is the continuation"

    # (b) eos_aware omitted -> getattr(cfg, ...) resolution (cfg True -> on).
    host_b = _core_host(num_block_candidates=2, candidate_eos_aware=True)
    recorder_b = _ForwardRecorder(_VOCAB, winner_row=0)
    _core_commit(host_b, x0.clone(), draft, masked_local, recorder_b, [], 6)
    cands_b = recorder_b.calls[0]["tokens"]
    assert torch.equal(cands_b[1, 3:5], expected), "cfg candidate_eos_aware=True resolves on"

    # (c) the explicit parameter overrides the cfg (cfg False + param True).
    host_c = _core_host(num_block_candidates=2, candidate_eos_aware=False)
    recorder_c = _ForwardRecorder(_VOCAB, winner_row=0)
    _core_commit(host_c, x0.clone(), draft, masked_local, recorder_c, [], 6, eos_aware=True)
    cands_c = recorder_c.calls[0]["tokens"]
    assert torch.equal(cands_c[1, 3:5], expected), "the explicit param beats the cfg"

    # (d) attribute-less host (Dream face): no-param == explicit off,
    #     byte-identical on the canvas and the commit.
    host_d1 = _core_host(num_block_candidates=2)
    recorder_d1 = _ForwardRecorder(_VOCAB, winner_row=0)
    x_d1 = x0.clone()
    _core_commit(host_d1, x_d1, draft, masked_local, recorder_d1, [], 6)
    host_d2 = _core_host(num_block_candidates=2)
    recorder_d2 = _ForwardRecorder(_VOCAB, winner_row=0)
    x_d2 = x0.clone()
    _core_commit(
        host_d2,
        x_d2,
        draft,
        masked_local,
        recorder_d2,
        [],
        6,
        eos_aware=False,
        refill_top_k=None,
    )
    assert torch.equal(
        recorder_d1.calls[0]["tokens"], recorder_d2.calls[0]["tokens"]
    ), "attr-less host: no-param call == explicit off (canvas)"
    assert torch.equal(x_d1[0, 3:5], x_d2[0, 3:5]), "attr-less host: commits bit-identical"


def test_ut2409_sampler_default_eos_on(device: str) -> None:
    """UT-2409: the LLaDA default resolves EOS-aware ON (default-on face).

    A config with N>0 and no ``candidate_eos_aware`` kwarg must read
    ``candidate_eos_aware is True``; on the ``eos_tail`` head (EOS top-1 at
    the last region position only) the batched canvas keeps the raw greedy
    row 0 (EOS at the position) while row N-1 is the EOS-free continuation,
    and the greedy row WINS the min-prob race so the committed block ends
    with EOS.
    """
    inner = _build_mini(device, head="eos_tail")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _llada_config(
        seed=2409,
        num_steps=2,
        num_tokens=4,
        block_size=4,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        speculative_sub_block_size=4,
    )
    assert cfg.candidate_eos_aware is True, "the default resolves EOS-aware ON"

    mff = _make_mff(inner, device)
    out = LLaDASampler(llm, config=cfg).sample(prompt, folded_model=mff)

    _, batched, _ = _split_calls(mff)
    cands = batched[0]["args"][0]
    assert torch.equal(
        cands[0, 3:7], torch.tensor([5, 5, 5, _EOS_ID], device=device)
    ), "row 0 keeps the raw greedy argmax (EOS at the eos-tail position)"
    assert not bool((cands[1, 3:7] == _EOS_ID).any()), "row N-1 is EOS-free"
    assert torch.equal(
        cands[1, 3:7], torch.tensor([5, 5, 5, 5], device=device)
    ), "row N-1 == the first non-EOS pool entry per position"
    assert torch.equal(
        out.sequences[0, 3:6], torch.tensor([5, 5, 5], device=device)
    ), "the greedy row wins the min-prob race (EOS self-prob is high)"
    assert out.sequences[0, 6] == _EOS_ID, "the commit ends with EOS"


# ---------------------------------------------------------------------------
# UT-2410: config / YAML consumption faces (FR4)
# ---------------------------------------------------------------------------


def test_ut2410_config_defaults_symmetry() -> None:
    """UT-2410: the four new keys with symmetric defaults; Dream/Fast-dLLM none.

    ``LLaDASamplerConfig`` and ``ActFoldConfig`` both carry
    ``speculative_sub_block_size=8`` / ``progressive_refill=False`` /
    ``refill_top_k=None`` / ``candidate_eos_aware=True``; the Dream and
    Fast-dLLM configs gain NO new fields (their faces stay byte-identical —
    the Dream host byte face lives in UT-2409, the LLaDA N=0 face in
    UT-2403).
    """
    cfg = LLaDASamplerConfig()
    assert cfg.speculative_sub_block_size == 8
    assert cfg.progressive_refill is False
    assert cfg.refill_top_k is None
    assert cfg.candidate_eos_aware is True

    act = ActFoldConfig()
    assert act.speculative_sub_block_size == 8
    assert act.progressive_refill is False
    assert act.refill_top_k is None
    assert act.candidate_eos_aware is True

    new_fields = (
        "speculative_sub_block_size",
        "progressive_refill",
        "refill_top_k",
        "candidate_eos_aware",
    )
    for cls in (DreamSamplerConfig, FastDLLMSamplerConfig):
        instance = cls()
        for field_name in new_fields:
            assert not hasattr(
                instance, field_name
            ), f"{cls.__name__} must not gain LLaDA-only granularity fields"


def test_ut2410_yaml_consumption(tmp_path: Any) -> None:
    """UT-2410: ``load_config`` consumes the four keys with zero UserWarning.

    ``ActFoldConfig`` gains the four AR024 keys (the UT-2313 precedent) so
    the YAML loads without the unknown-key ``UserWarning`` (AGENTS #5);
    absent keys keep the defaults 8 / False / None / True.
    """
    cfg_direct = ActFoldConfig(
        speculative_sub_block_size=4,
        progressive_refill=True,
        refill_top_k=2,
        candidate_eos_aware=False,
    )
    assert cfg_direct.speculative_sub_block_size == 4
    assert cfg_direct.progressive_refill is True
    assert cfg_direct.refill_top_k == 2
    assert cfg_direct.candidate_eos_aware is False

    cfg_file = tmp_path / "ar024_ut2410.yaml"
    cfg_file.write_text(
        "speculative_sub_block_size: 4\n"
        "progressive_refill: true\n"
        "refill_top_k: 2\n"
        "candidate_eos_aware: false\n",
        encoding="utf-8",
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        loaded = load_config(cfg_file)
    assert _user_warning_messages(record) == [], "keys must be consumed, not warned off"
    assert loaded.speculative_sub_block_size == 4
    assert loaded.progressive_refill is True
    assert loaded.refill_top_k == 2
    assert loaded.candidate_eos_aware is False

    cfg_file.write_text("# empty config\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as record_default:
        warnings.simplefilter("always")
        loaded_default = load_config(cfg_file)
    assert _user_warning_messages(record_default) == []
    assert loaded_default.speculative_sub_block_size == 8
    assert loaded_default.progressive_refill is False
    assert loaded_default.refill_top_k is None
    assert loaded_default.candidate_eos_aware is True


def test_ut2410_per_model_yaml_section() -> None:
    """UT-2410: ``llada_8b.yaml`` carries the AR024 granularity section.

    The per-model YAML must mention the granularity keys (commented or
    active — the design freezes a default-off section) and must load with
    zero unknown-key ``UserWarning``.
    """
    path = _REPO_ROOT / "actfold" / "configs" / "per_model" / "llada_8b.yaml"
    text = path.read_text(encoding="utf-8")
    for key in ("speculative_sub_block_size", "candidate_eos_aware", "progressive_refill"):
        assert key in text, f"llada_8b.yaml must carry the AR024 section ({key})"
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        load_config(path)
    assert _user_warning_messages(record) == [], "the shipped section loads cleanly"


# ---------------------------------------------------------------------------
# UT-2411: bench contract (FR5)
# ---------------------------------------------------------------------------


def test_ut2411_bench_contract() -> None:
    """UT-2411: the AR024 granularity bench exists and reuses the ar023 face.

    ``scripts/ar024_granularity_bench`` imports
    ``scripts.ar023_cross_family_bench`` at module level as ``_ar023`` and
    reuses its helpers by IDENTITY (``mark_all_blocked`` /
    ``h1_quality_verdict`` / ``_make_bench_mff`` — never re-implemented);
    the frozen four-mode granularity stacks pin the A/B isolation (the
    greedy stack differs from the batched anchor ONLY in the three
    granularity knobs); the ``--refill-face`` artifact is a separate
    ``_refill`` path (never the formal verdict path); the CLI family domain
    is llada-only; and the meta ``granularity_constants`` carry the frozen
    keys. The import lives in the test body so module collection always
    succeeds.
    """
    import scripts.ar023_cross_family_bench as _ar023
    import scripts.ar024_granularity_bench as _ar024

    assert _ar024._ar023 is _ar023, "module-level import-reuse of the ar023 bench"
    assert _ar024.mark_all_blocked is _ar023.mark_all_blocked
    assert _ar024.h1_quality_verdict is _ar023.h1_quality_verdict
    assert _ar024._make_bench_mff is _ar023._make_bench_mff

    stacks = _ar024.GRANULARITY_STACKS
    assert stacks["folded-batched"] == {
        "num_block_candidates": 4,
        "candidate_top_k": 10,
        "candidate_include_greedy": False,
        "speculative_sub_block_size": _ar023.BLOCK_GRID,
        "candidate_eos_aware": False,
        "progressive_refill": False,
    }, "the batched stack is the AR023 anchor (s=block_size, eos off)"
    assert stacks["folded-batched-greedy"] == {
        "num_block_candidates": 4,
        "candidate_top_k": 10,
        "candidate_include_greedy": True,
        "speculative_sub_block_size": 8,
        "candidate_eos_aware": True,
        "progressive_refill": False,
    }, "the greedy stack is the AR024 reference granularity"
    diff_keys = {
        key
        for key in stacks["folded-batched"]
        if stacks["folded-batched"][key] != stacks["folded-batched-greedy"][key]
    }
    assert diff_keys == {
        "candidate_include_greedy",
        "speculative_sub_block_size",
        "candidate_eos_aware",
    }, "constructive A/B isolation: only the granularity knobs differ"

    assert _ar024.DEFAULT_OUTPUT_PATH == Path(
        "results/optimization/ar024_llada_granularity_bench.json"
    )
    assert _ar024.quick_output_path() == Path(
        "results/optimization/ar024_llada_granularity_bench_quick.json"
    )
    assert _ar024.refill_output_path() == Path(
        "results/optimization/ar024_llada_granularity_bench_refill.json"
    )
    assert (
        _ar024.refill_output_path() != _ar024.DEFAULT_OUTPUT_PATH
    ), "the refill face never writes the formal verdict artifact"

    refill_args = _ar024.parse_args(["--family", "llada", "--refill-face"])
    assert refill_args.refill_face is True
    plain_args = _ar024.parse_args(["--family", "llada"])
    assert plain_args.refill_face is False
    with pytest.raises(SystemExit) as excinfo:
        _ar024.parse_args(["--family", "dream"])
    assert excinfo.value.code == 2, "the AR024 bench is llada-only"

    constants = _ar024.GRANULARITY_CONSTANTS
    assert set(constants) == {
        "sub_block_size",
        "candidate_eos_aware",
        "progressive_refill",
        "refill_top_k",
        "anchor_semantics",
    }
    assert constants["anchor_semantics"] == "ar023 (s=block_size, eos off)"


# ---------------------------------------------------------------------------
# UT-2413: window-level early stop + partial-commit finished (design gate G13)
# ---------------------------------------------------------------------------


def test_ut2413_window_level_early_stop(device: str) -> None:
    """UT-2413: EOS committed in a window skips the remaining windows/blocks.

    On the ``eos_window`` head (EOS top-1 inside [5, 7)) with s=2 windows:
    window 0 commits plain fills; window 1 commits EOS at both positions,
    so the window loop breaks — the remaining windows of the block AND the
    whole second block are skipped (exactly 4 folded calls for the 2
    executed windows), the unfinished positions stay masked, and the decode
    face never trims them.
    """
    inner = _build_mini(device, head="eos_window")
    llm = _MiniDiffusionLLM(inner)
    prompt = _prompt(device)
    cfg = _llada_config(
        seed=2413,
        num_steps=2,
        num_tokens=8,
        block_size=4,
        num_block_candidates=1,
        candidate_top_k=2,
        candidate_include_greedy=True,
        candidate_eos_aware=False,
        speculative_sub_block_size=2,
    )
    mff = _make_mff(inner, device)
    out = LLaDASampler(llm, config=cfg).sample(prompt, folded_model=mff)

    drafts, batched, _ = _split_calls(mff)
    assert len(drafts) == 2 and len(batched) == 2, "only the executed windows forward"
    assert len(mff.calls) == 4, "2 windows x (1 draft + 1 batched); later windows skipped"
    tags = [c["kwargs"]["branch_id"] for c in drafts]
    assert tags == ["llada_spec_b0", "llada_spec_b0_w1"]
    assert not any("llada_spec_b1" in tag for tag in tags), "block 1 is skipped entirely"

    assert tuple(out.sequences.shape) == (1, 11), "the decode face never trims"
    assert torch.equal(
        out.sequences[0, 3:5], torch.tensor([5, 5], device=device)
    ), "window 0 commits fills"
    assert bool((out.sequences[0, 5:7] == _EOS_ID).all()), "window 1 commits EOS"
    assert bool((out.sequences[0, 7:11] == _MASK_ID).all()), "unfinished positions stay masked"


def test_ut2413_core_partial_commit_finished_topk_only() -> None:
    """UT-2413: the Core checks ``finished`` only on the committed top-k.

    With ``refill_top_k=3`` over a 4-position unit whose winner row carries
    EOS only at the LOWEST-confidence position (uncommitted): the top-3
    positions commit the winner tokens, the last position stays masked,
    ``finished`` stays False (an EOS at an uncommitted position must not
    finish), the Core does not append history, and the chain switches to
    the winner branch.
    """
    x = torch.tensor([[1, 2, 3, _MASK_ID, _MASK_ID, _MASK_ID, _MASK_ID, 7]])
    draft = torch.zeros(1, 4, _VOCAB)
    draft[0, 0, 11] = 5.0
    draft[0, 1, 11] = 5.0
    draft[0, 2, 11] = 5.0
    draft[0, 3, _EOS_ID] = 3.0
    draft[0, 3, 12] = 1.0
    masked_local = torch.tensor([0, 1, 2, 3])

    host = _core_host(num_block_candidates=1)
    recorder = _ForwardRecorder(_VOCAB, winner_row=0)
    history: list[torch.Tensor] = []
    finished = _core_commit(
        host,
        x,
        draft,
        masked_local,
        recorder,
        history,
        8,
        eos_aware=False,
        refill_top_k=3,
    )

    assert not bool(finished.any()), "EOS at an uncommitted position must not finish"
    assert torch.equal(x[0, 3:6], torch.tensor([11, 11, 11])), "top-3 commit the winner tokens"
    assert x[0, 6] == _MASK_ID, "the lowest-confidence position stays masked"
    assert history == [], "the Core never appends history (the family owns it)"
    assert host._folding_parent == "spec_b0_cand0", "the chain switches to the winner"


def test_ut2413_unmasked_window_skip(device: str) -> None:
    """UT-2413: a window without masked positions is skipped with zero forwards.

    A MASK token inside the prompt makes ``prompt_len`` the first-mask index
    (2), so the generation region is [2, 10) over a canvas of T=15: block 0
    splits into windows [2,4) / [4,6) / [6,8) of which the middle one is
    fully unmasked (prompt tokens 9, 10 — skipped with zero forwards), and
    block 1 is the single window [8, 10). Exactly 3 draft + 3 batched calls
    run; the window tag carries ``w_idx=2`` for the third block-0 window;
    prompt tokens are never overwritten and the pad tail stays masked.
    """
    inner = _build_mini(device, head="spike")
    llm = _MiniDiffusionLLM(inner)
    prompt = torch.tensor([[1, 2, _MASK_ID, 8, 9, 10, 11]], device=device)
    cfg = _llada_config(
        seed=2413,
        num_steps=2,
        num_tokens=8,
        block_size=6,
        num_block_candidates=2,
        candidate_top_k=2,
        candidate_include_greedy=True,
        candidate_eos_aware=False,
        speculative_sub_block_size=2,
    )
    mff = _make_mff(inner, device)
    out = LLaDASampler(llm, config=cfg).sample(prompt, folded_model=mff)

    drafts, batched, _ = _split_calls(mff)
    assert len(drafts) == 3 and len(batched) == 3, "the unmasked window is skipped"
    assert len(mff.calls) == 6
    tags = [c["kwargs"]["branch_id"] for c in drafts]
    assert tags == [
        "llada_spec_b0",
        "llada_spec_b0_w2",
        "llada_spec_b1",
    ], "w_idx counts EVERY window of the block, skipped ones included"

    assert tuple(out.sequences.shape) == (1, 15)
    assert out.sequences[0, 2] == 5, "the in-prompt mask position is generated"
    assert torch.equal(
        out.sequences[0, 3:7], torch.tensor([8, 9, 10, 11], device=device)
    ), "prompt tokens are never overwritten"
    assert bool((out.sequences[0, 7:10] == 5).all()), "the run windows commit the spike fill"
    assert bool((out.sequences[0, 10:15] == _MASK_ID).all()), "the pad tail stays masked"
