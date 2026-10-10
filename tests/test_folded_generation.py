"""Tests for true end-to-end folded generation."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, FoldedModel, SimilarityGate
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.vectorized_cache import VectorizedActivationCache
from actfold.eval.base_adapter import BaseEvalAdapter
from actfold.eval.generation_utils import greedy_generate
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from actfold.speculative.acceptance_policy import AcceptancePolicy
from actfold.speculative.branch_tree import BranchNode
from actfold.speculative.draft_generator import DraftGenerator
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
from actfold.speculative.folded_generation import _next_branch_id, folded_generate

_PROFILER_TARGET = "actfold.speculative.folded_generation.GLOBAL_STABILITY_PROFILER"


class TinyTransformer(nn.Module):
    """Tiny transformer for folded generation tests."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, hidden_dim // 64),
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _make_adapter(
    vocab_size: int = 16, hidden_dim: int = 64, num_layers: int = 2
) -> FastDLLMAdapter:
    raw = TinyTransformer(vocab_size, hidden_dim, num_layers)
    cache = ActivationCache(max_entries_per_layer=128, device="cpu")
    gate = SimilarityGate(tau=0.95)
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=num_layers, num_steps=1)
    folded = FoldedModel(raw, cache=cache, gate=gate, scheduler=scheduler)
    adapter = FastDLLMAdapter(
        raw, folded_model=folded, num_layers=num_layers, hidden_dim=hidden_dim
    )
    adapter.underlying_model.eval()
    return adapter


def _make_vectorized_adapter(
    vocab_size: int = 16, hidden_dim: int = 64, num_layers: int = 2
) -> tuple[FastDLLMAdapter, FoldedModel, VectorizedActivationCache]:
    """Build an adapter whose folded model uses a vectorized activation cache."""
    raw = TinyTransformer(vocab_size, hidden_dim, num_layers)
    cache = VectorizedActivationCache(max_entries_per_layer=128, device="cpu")
    gate = SimilarityGate(tau=0.95)
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=num_layers, num_steps=1)
    folded = FoldedModel(raw, cache=cache, gate=gate, scheduler=scheduler)
    adapter = FastDLLMAdapter(
        raw, folded_model=folded, num_layers=num_layers, hidden_dim=hidden_dim
    )
    adapter.underlying_model.eval()
    return adapter, folded, cache


def test_folded_generate_runs_with_folded_model() -> None:
    """folded_generate successfully uses the folded forward path."""
    folded_adapter = _make_adapter()
    prompt = torch.tensor([[1, 2, 3]])

    result = folded_generate(folded_adapter, prompt, max_new_tokens=4)

    assert result.tokens.shape == (1, 7)
    assert result.num_folded_steps == 4
    assert 0.0 <= result.stable_ratio <= 1.0


def test_folded_generate_reports_stable_ratio() -> None:
    """folded_generate returns a non-negative stable ratio."""
    adapter = _make_adapter()
    prompt = torch.tensor([[1, 2, 3]])

    result = folded_generate(adapter, prompt, max_new_tokens=3)

    assert result.tokens.shape[1] == 6
    assert 0.0 <= result.stable_ratio <= 1.0
    assert result.num_folded_steps == 3


def test_folded_generate_without_folded_model_falls_back() -> None:
    """Without a folded model, folded_generate still produces greedy output."""
    raw = TinyTransformer(vocab_size=16, hidden_dim=64, num_layers=2)
    raw.eval()
    adapter = FastDLLMAdapter(raw, num_layers=2, hidden_dim=64)
    prompt = torch.tensor([[1, 2, 3]])

    expected = greedy_generate(adapter, prompt, max_new_tokens=3)
    result = folded_generate(adapter, prompt, max_new_tokens=3)

    assert torch.equal(result.tokens, expected)


@dataclass
class _ScriptedProfile:
    """Minimal StabilityProfile stand-in with a fixed mean stable ratio."""

    mean_stable_ratio: float


class _ScriptedProfiler:
    """Minimal StabilityProfiler stand-in with scripted per-step stable ratios.

    Branch IDs are opaque bounded tokens (``session:counter``), so steps are
    identified by call order: the Nth ``get_profile`` call for a non-root
    branch maps to ``step_ratios[N-1]``. The root branch (``"root"``) returns
    no profile, mirroring branches that never recorded stability statistics.
    """

    def __init__(self, step_ratios: dict[int, float]) -> None:
        self._step_ratios = step_ratios
        self._call_count = 0

    def reset_branch(self, branch_id: Any) -> None:
        """No-op reset satisfying the folded-generation loop contract."""
        return None

    def set_enabled(self, enabled: bool) -> None:
        """No-op toggle satisfying the folded-generation loop contract."""
        return None

    @property
    def enabled(self) -> bool:
        """Report enabled so the loop's state save/restore finds an attribute."""
        return True

    def get_profile(self, branch_id: Any) -> Optional[_ScriptedProfile]:
        """Return the scripted profile for step branches, else ``None``."""
        if branch_id == "root":
            return None
        self._call_count += 1
        ratio = self._step_ratios.get(self._call_count - 1, 0.0)
        return _ScriptedProfile(mean_stable_ratio=ratio)


class _FakeTokenizer:
    """Minimal tokenizer encoding every prompt as the same three tokens."""

    def encode(
        self,
        prompt: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> torch.Tensor:
        """Return a fixed ``[1, 3]`` token tensor for any prompt."""
        return torch.tensor([[1, 2, 3]])


class _FakeEvalModel:
    """Minimal model exposing the shape attributes the FLOPs counter reads."""

    def __init__(self) -> None:
        self.num_layers = 2
        self.hidden_dim = 64
        self.num_heads = 4
        self.underlying_model = nn.Linear(8, 8)


def _make_minimal_eval_adapter() -> BaseEvalAdapter:
    """Build a ``BaseEvalAdapter`` carrying only what TFLOPs estimation reads.

    ``BaseEvalAdapter.__new__`` skips the heavy ``__init__`` dependencies
    (baseline, engine, judge) that ``_estimate_actfold_tflops`` never touches;
    this is a test-only minimal mock.
    """
    adapter = BaseEvalAdapter.__new__(BaseEvalAdapter)
    adapter.model = _FakeEvalModel()
    adapter.tokenizer = _FakeTokenizer()
    adapter.vocab_size = 16
    adapter.max_new_tokens = 5
    return adapter


def _hand_actfold_tflops(
    num_layers: int,
    hidden_dim: int,
    seq_len: int,
    vocab_size: int,
    reuse_ratio: float,
) -> float:
    """Independently compute total TFLOPs for one forward pass.

    This mirrors the FLOPs model documented on
    :func:`actfold.utils.flops_counter.count_diffusion_llm_flops` without
    calling it, so it serves as an oracle for the adapter's estimate.
    """
    effective_seq_len = seq_len * (1.0 - reuse_ratio)
    attention = 4 * num_layers * hidden_dim * hidden_dim * effective_seq_len
    ffn = 16 * num_layers * hidden_dim * hidden_dim * effective_seq_len
    # LM-head output projection only (the input embedding is a table lookup).
    embedding = vocab_size * hidden_dim * seq_len
    return (attention + ffn + embedding) / 1e12


@pytest.mark.parametrize(
    ("step_ratios", "expected_mean"),
    [
        ({0: 1.0, 1: 0.5, 2: 0.25}, (1.0 + 0.5 + 0.25) / 3),
        ({0: 1.0, 1: 1.0, 2: 0.25}, 0.75),
    ],
)
def test_stable_ratio_is_mean_across_folded_steps(
    monkeypatch: pytest.MonkeyPatch,
    step_ratios: dict[int, float],
    expected_mean: float,
) -> None:
    """stable_ratio is the mean over all folded steps, not the last step's value.

    The first case uses the srs §3.1 B1 per-step ratios 1.0/0.5/0.25, whose
    arithmetic mean is 1.75/3 (the srs quotes 0.75, which is the mean of the
    second case's ratios 1.0/1.0/0.25, also covered here with the same last
    step 0.25). Either way the result must not collapse to the last step's
    value.
    """
    adapter = _make_adapter()
    prompt = torch.tensor([[1, 2, 3]])
    monkeypatch.setattr(_PROFILER_TARGET, _ScriptedProfiler(step_ratios))

    result = folded_generate(adapter, prompt, max_new_tokens=3)

    assert result.num_folded_steps == 3
    assert result.stable_ratio == pytest.approx(expected_mean)


def test_stable_ratio_single_step_equals_step_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a single folded step, stable_ratio equals that step's value."""
    adapter = _make_adapter()
    prompt = torch.tensor([[1, 2, 3]])
    monkeypatch.setattr(_PROFILER_TARGET, _ScriptedProfiler({0: 0.5}))

    result = folded_generate(adapter, prompt, max_new_tokens=1)

    assert result.num_folded_steps == 1
    assert result.stable_ratio == pytest.approx(0.5)


def test_stable_ratio_zero_steps_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no folded steps recorded, stable_ratio is 0.0 without errors."""
    adapter = _make_adapter()
    prompt = torch.tensor([[1, 2, 3]])
    monkeypatch.setattr(_PROFILER_TARGET, _ScriptedProfiler({}))

    result = folded_generate(adapter, prompt, max_new_tokens=0)

    assert result.num_folded_steps == 0
    assert result.stable_ratio == 0.0
    assert torch.equal(result.tokens, prompt)


def test_stable_ratio_defaults_to_zero_when_steps_record_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Steps whose profiles default to 0.0 average to 0.0 without errors."""
    adapter = _make_adapter()
    prompt = torch.tensor([[1, 2, 3]])
    monkeypatch.setattr(_PROFILER_TARGET, _ScriptedProfiler({}))

    result = folded_generate(adapter, prompt, max_new_tokens=2)

    assert result.num_folded_steps == 2
    assert result.stable_ratio == 0.0


def test_estimate_actfold_tflops_matches_hand_formula_for_mean_ratio() -> None:
    """_estimate_actfold_tflops matches the hand FLOPs formula at mean ratios.

    The ratio ``0.75`` is the all-step mean reported by ``folded_generate``
    for per-step ratios such as ``1.0/1.0/0.25``; the estimator consumes
    ``result.stable_ratio`` as-is, so its output must equal the hand formula
    evaluated at that mean.
    """
    eval_adapter = _make_minimal_eval_adapter()
    # Pre-tokenized prompts (tokenize-once contract): two 3-token prompts.
    prompt_tokens = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])]
    ratios = [0.75, 0.25]

    got = eval_adapter._estimate_actfold_tflops(prompt_tokens, ratios)

    seq_len = 3 + eval_adapter.max_new_tokens
    expected = _hand_actfold_tflops(
        num_layers=2, hidden_dim=64, seq_len=seq_len, vocab_size=16, reuse_ratio=0.75
    ) + _hand_actfold_tflops(
        num_layers=2, hidden_dim=64, seq_len=seq_len, vocab_size=16, reuse_ratio=0.25
    )
    assert got == pytest.approx(expected)


def test_folded_generate_profiler_state_restored_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """folded_generate restores the global profiler state when the model errors.

    With ``record_stability=False`` the loop disables ``GLOBAL_STABILITY_PROFILER``
    up front.  A RuntimeError raised mid-loop must not leak that disabled state:
    after the exception propagates, the profiler must be re-enabled.
    """
    adapter = _make_adapter()

    def _explode(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(adapter, "forward", _explode)
    prior_enabled = GLOBAL_STABILITY_PROFILER.enabled
    try:
        prompt = torch.tensor([[1, 2, 3]])

        with pytest.raises(RuntimeError):
            folded_generate(adapter, prompt, max_new_tokens=2, record_stability=False)

        assert GLOBAL_STABILITY_PROFILER.enabled is True
    finally:
        GLOBAL_STABILITY_PROFILER.set_enabled(prior_enabled)


def test_prune_frees_activation_cache() -> None:
    """Pruning non-accepted siblings must free their cached activations."""
    adapter, folded, cache = _make_vectorized_adapter()
    calls: list[str] = []
    orig_clear = cache.clear_branch

    def _spy_clear_branch(branch_id: str) -> None:
        calls.append(branch_id)
        return orig_clear(branch_id)

    cache.clear_branch = _spy_clear_branch  # type: ignore[assignment]

    draft = DraftGenerator(vocab_size=16, mode="random")
    prompt = torch.tensor([[1, 2, 3]])

    result = folded_generate(
        adapter,
        prompt,
        max_new_tokens=2,
        folded_model=folded,
        draft_generator=draft,
        num_branches_per_step=2,
    )

    assert result.tokens.shape[1] == 5
    assert calls, "pruned sibling branches must free their activation cache"

    mask = torch.ones(1, 4, dtype=torch.bool)
    for pruned_id in calls:
        assert pruned_id != result.final_branch_id
        with pytest.raises(KeyError):
            cache.get(pruned_id, 0, mask)


def test_branch_ids_bounded_length() -> None:
    """Branch IDs must stay constant-length instead of growing with steps."""
    adapter = _make_adapter()
    prompt = torch.tensor([[1, 2, 3]])

    short = folded_generate(adapter, prompt, max_new_tokens=1)
    long = folded_generate(adapter, prompt, max_new_tokens=6)

    assert len(str(short.final_branch_id)) < 32
    assert len(str(long.final_branch_id)) < 32

    ids = [_next_branch_id("root", i, 0) for i in range(50)]
    assert all(len(branch_id) < 32 for branch_id in ids)
    assert len(set(ids)) == 50


class _DoublingLayer(nn.Module):
    """Minimal token-wise layer used as a FoldedTransformerLayer base."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hidden_states * 2


def test_cache_miss_recomputes_divergent() -> None:
    """Regression guard: a parent cache miss falls back to full recompute."""
    hidden_dim = 16
    layer = _DoublingLayer()
    cache = ActivationCache(max_entries_per_layer=16, device="cpu")
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0)

    x = torch.randn(2, 4, hidden_dim)
    out = folded(x, branch_id="child", parent_branch_id="never_cached")

    assert torch.allclose(out, layer(x))


# ---------------------------------------------------------------------------
# AR003/T004 (IT-301 / IT-302): causal-model end-to-end folding
# ---------------------------------------------------------------------------

_CAUSAL_VOCAB = 16
_CAUSAL_HIDDEN = 32
_CAUSAL_LAYERS = 3
_CAUSAL_SEED = 2026
_CAUSAL_PROMPT = [[1, 5, 9, 13]]


class CausalCumsumLayer(nn.Module):
    """Causal synthetic layer whose prefix outputs depend only on prefix inputs.

    ``torch.cumsum`` at position ``t`` accumulates only positions ``<= t``
    and is bit-exact prefix-stable (the prefix of a longer scan equals the
    shorter scan), so a child that extends a parent with suffix tokens
    reproduces the parent's prefix hidden states bit-for-bit.  This is the
    mathematical basis for prefix folding being fully stable on this model.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cum = torch.cumsum(hidden_states, dim=1)
        denom = torch.arange(1, hidden_states.shape[1] + 1, device=hidden_states.device).view(
            1, -1, 1
        )
        return hidden_states + 0.1 * cum / denom


class CausalCumsumModel(nn.Module):
    """Causal synthetic decoder with a detectable embedding/layers/head layout.

    The attribute names (``embedding`` / ``layers`` / ``head``) are all in the
    :func:`detect_architecture` default path tables, so
    :class:`ManualFoldedForward` auto-discovers the stack without explicit
    wiring.  Unlike ``TinyTransformer`` above, this model is causal: a
    non-causal full-attention layer would make the child prefix hidden states
    differ from the parent's, so folding could never activate.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(CausalCumsumLayer() for _ in range(num_layers))
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _make_causal_setup(
    device: str = "cpu",
    use_cuda_graph: bool = False,
    graph_capacity_ratio: float = 0.5,
) -> tuple[nn.Module, FastDLLMAdapter, ManualFoldedForward, ActivationCache]:
    """Build a causal model plus its Manual folded adapter on ``device``.

    The eager comparison adapter wraps the SAME raw model (shared weights).
    ``ManualFoldedForward`` is non-mutating (AR002), so the folded path can
    never change what the eager run observes.
    """
    torch.manual_seed(_CAUSAL_SEED)
    raw = CausalCumsumModel(_CAUSAL_VOCAB, _CAUSAL_HIDDEN, _CAUSAL_LAYERS).to(device)
    raw.eval()
    cache = ActivationCache(max_entries_per_layer=128, device=device)
    gate = SimilarityGate(tau=0.95)
    folded = ManualFoldedForward(
        raw,
        cache=cache,
        gate=gate,
        scheduler=None,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=graph_capacity_ratio,
    )
    adapter = FastDLLMAdapter(
        raw,
        folded_model=folded,
        num_layers=_CAUSAL_LAYERS,
        hidden_dim=_CAUSAL_HIDDEN,
    )
    adapter.underlying_model.eval()
    return raw, adapter, folded, cache


def test_folded_generate_causal_model_folds_and_matches_eager() -> None:
    """IT-301: causal-model folded_generate matches eager greedy and truly folds.

    On a causal model the child's prefix hidden states equal the parent's
    bit-for-bit, so every generation step's prefix is fully stable.  Over 4
    steps (T: 4 -> 8) the folded output must (1) equal the all-eager greedy
    run on the SAME raw weights bit-for-bit, (2) report a mean stable ratio
    above 0.5, and (3) leave the chained branch recursion's cache entries at
    the final token length — proof that each step really folded against its
    parent branch.
    """
    raw, adapter, folded, cache = _make_causal_setup("cpu")
    eager_adapter = FastDLLMAdapter(raw, num_layers=_CAUSAL_LAYERS, hidden_dim=_CAUSAL_HIDDEN)
    prompt = torch.tensor(_CAUSAL_PROMPT)
    max_new_tokens = 4

    expected = greedy_generate(eager_adapter, prompt, max_new_tokens=max_new_tokens)
    result = folded_generate(adapter, prompt, max_new_tokens=max_new_tokens, folded_model=folded)

    assert result.tokens.shape == (1, prompt.shape[1] + max_new_tokens)
    assert result.num_folded_steps == max_new_tokens
    assert torch.equal(result.tokens, expected)
    assert result.stable_ratio > 0.5

    final_len = result.tokens.shape[1]
    for layer_idx in range(_CAUSAL_LAYERS):
        entry = cache.fetch(branch_id=result.final_branch_id, layer_idx=layer_idx)
        assert tuple(entry["ffn_out"].shape) == (1, final_len, _CAUSAL_HIDDEN)
        if layer_idx == 0:
            assert tuple(entry["embedding"].shape) == (1, final_len, _CAUSAL_HIDDEN)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_folded_generate_graph_zero_interference() -> None:
    """IT-302: graph-enabled folded_generate is inert on variable-length steps.

    Every generation step appends one token, so the parent cache shape never
    matches the child tokens shape: ``_parent_cache_complete`` marks the
    parent as incomplete and the graph path must silently stay eager (no
    capture, zero UserWarning) while the folded eager path still produces
    tokens identical to the all-eager greedy run.
    """
    device = "cuda"
    raw, adapter, folded, _cache = _make_causal_setup(
        device, use_cuda_graph=True, graph_capacity_ratio=0.5
    )
    eager_adapter = FastDLLMAdapter(raw, num_layers=_CAUSAL_LAYERS, hidden_dim=_CAUSAL_HIDDEN)
    prompt = torch.tensor(_CAUSAL_PROMPT, device=device)
    max_new_tokens = 4

    expected = greedy_generate(eager_adapter, prompt, max_new_tokens=max_new_tokens)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = folded_generate(
            adapter, prompt, max_new_tokens=max_new_tokens, folded_model=folded
        )

    user_warnings = [w for w in record if issubclass(w.category, UserWarning)]
    assert not user_warnings, (
        f"graph-enabled folded_generate must emit zero UserWarning, got: "
        f"{[str(w.message) for w in user_warnings]}"
    )
    assert folded.graph_runner is None
    assert result.num_folded_steps == max_new_tokens
    assert result.tokens.shape == (1, prompt.shape[1] + max_new_tokens)
    assert torch.equal(result.tokens, expected)


# ---------------------------------------------------------------------------
# AR004/T003 (UT-309 / UT-310): target-match acceptance policy + rate report
# ---------------------------------------------------------------------------

_AR004_VOCAB = 16
_AR004_PROMPT = [[1, 5, 9]]


def _target_match_policy() -> AcceptancePolicy:
    """Instantiate the target-match acceptance policy (lazily imported).

    The lazy import keeps the module collectible before the implementation
    lands: in the TDD Red state only the UT-309 tests below fail with an
    ImportError, while every pre-existing test in this file keeps running.
    """
    from actfold.speculative.acceptance_policy import TargetMatchAcceptancePolicy

    return TargetMatchAcceptancePolicy()


def _argmax_logits(
    argmax_seq: list[int],
    batch: int,
    vocab_size: int = _AR004_VOCAB,
) -> torch.Tensor:
    """Build logits with a scripted per-position argmax, shared by all rows.

    Args:
        argmax_seq: Argmax token id for each position ``t``.
        batch: Batch size of the returned tensor.
        vocab_size: Vocabulary dimension.

    Returns:
        ``[batch, len(argmax_seq), vocab_size]`` logits with a +10 spike at
        ``argmax_seq[t]`` for every batch row at position ``t``.
    """
    logits = torch.zeros(batch, len(argmax_seq), vocab_size)
    for pos, tok in enumerate(argmax_seq):
        logits[:, pos, tok] = 10.0
    return logits


def _make_policy_candidate(
    branch_id: str,
    tokens: torch.Tensor,
    logits: torch.Tensor | None,
) -> BranchNode:
    """Build a BranchNode candidate with explicit tokens and logits."""
    return BranchNode(
        branch_id=branch_id,
        parent_id="root",
        tokens=tokens,
        logits=logits,
        depth=1,
    )


class FixedArgmaxModel(nn.Module):
    """Stub model whose logits have a scripted argmax per position.

    Position ``t`` of the returned logits always has argmax
    ``argmax_table[t]`` regardless of the input tokens.  The greedy path of
    ``folded_generate`` therefore appends ``argmax_table[T-1]`` for a parent
    of length ``T``, and whether that appended token matches the child's
    same-position argmax (``argmax_table[T]``) is exactly controllable —
    giving precise 0.0 / 1.0 single-step acceptance rates (srs §3.5-3).
    """

    def __init__(self, vocab_size: int, argmax_table: list[int]) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._argmax_table = list(argmax_table)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.shape[1]
        logits = torch.zeros(tokens.shape[0], seq_len, self._vocab_size)
        for pos in range(seq_len):
            logits[:, pos, self._argmax_table[pos]] = 10.0
        return logits


def _make_fixed_argmax_adapter(argmax_table: list[int]) -> FastDLLMAdapter:
    """Wrap a FixedArgmaxModel in an adapter (no folded model; direct path)."""
    raw = FixedArgmaxModel(_AR004_VOCAB, argmax_table)
    return FastDLLMAdapter(raw, num_layers=1, hidden_dim=8, vocab_size=_AR004_VOCAB)


def test_target_match_policy_prefers_matching_candidate() -> None:
    """UT-309a (srs §3.5-1): the candidate whose appended token matches its
    same-position argmax (rate 1.0) is selected over a mismatching one (0.0),
    regardless of list order."""
    policy = _target_match_policy()
    tokens = torch.tensor([[1, 5, 9, 7]])
    match = _make_policy_candidate("match", tokens, _argmax_logits([0, 0, 0, 7], 1))
    mismatch = _make_policy_candidate("mismatch", tokens, _argmax_logits([0, 0, 0, 0], 1))

    assert policy.select([mismatch, match]) is match
    assert policy.select([match, mismatch]) is match


def test_target_match_policy_tie_breaks_to_first() -> None:
    """UT-309b: candidates with equal rates tie-break to the first in list order."""
    policy = _target_match_policy()
    tokens = torch.tensor([[1, 5, 9, 7]])
    first = _make_policy_candidate("first", tokens, _argmax_logits([0, 0, 0, 7], 1))
    second = _make_policy_candidate("second", tokens, _argmax_logits([1, 1, 1, 7], 1))

    assert policy.select([first, second]) is first
    assert policy.select([second, first]) is second


def test_target_match_policy_skips_none_logits_candidates() -> None:
    """UT-309c (EX-605): candidates with ``logits=None`` are skipped; the
    selection is made among the candidates that do carry logits."""
    policy = _target_match_policy()
    tokens = torch.tensor([[1, 5, 9, 7]])
    no_logits = _make_policy_candidate("none", tokens, None)
    match = _make_policy_candidate("match", tokens, _argmax_logits([0, 0, 0, 7], 1))
    mismatch = _make_policy_candidate("mismatch", tokens, _argmax_logits([0, 0, 0, 0], 1))

    assert policy.select([no_logits, match, mismatch]) is match
    assert policy.select([mismatch, no_logits]) is mismatch


def test_target_match_policy_all_none_logits_returns_first() -> None:
    """UT-309d (EX-605/EC-9): when every candidate has ``logits=None`` the
    policy returns ``candidates[0]`` without raising."""
    policy = _target_match_policy()
    tokens = torch.tensor([[1, 5, 9, 7]])
    first = _make_policy_candidate("first", tokens, None)
    second = _make_policy_candidate("second", tokens, None)

    selected = policy.select([first, second])

    assert selected is first


def test_target_match_policy_single_candidate_returned_directly() -> None:
    """UT-309e (EC-11): a lone candidate is returned as-is."""
    policy = _target_match_policy()
    tokens = torch.tensor([[1, 5, 9, 7]])
    only = _make_policy_candidate("only", tokens, _argmax_logits([0, 0, 0, 7], 1))

    assert policy.select([only]) is only


def test_target_match_policy_batch_half_match_selects_between_extremes() -> None:
    """UT-309f (srs §3.5-3): with batch=2 and exactly one matching row the
    rate is 0.5 — it must beat a 0.0-rate candidate and lose to a 1.0-rate
    candidate, pinning the value strictly between the extremes."""
    policy = _target_match_policy()
    tokens = torch.tensor([[1, 5, 9, 7], [2, 6, 10, 7]])
    full = _make_policy_candidate("full", tokens, _argmax_logits([0, 0, 0, 7], 2))
    zero = _make_policy_candidate("zero", tokens, _argmax_logits([0, 0, 0, 0], 2))
    half_logits = torch.zeros(2, 4, _AR004_VOCAB)
    half_logits[:, :3, 0] = 10.0
    half_logits[0, 3, 7] = 10.0
    half_logits[1, 3, 0] = 10.0
    half = _make_policy_candidate("half", tokens, half_logits)

    assert policy.select([zero, half]) is half
    assert policy.select([half, zero]) is half
    assert policy.select([half, full]) is full


def test_target_match_policy_conforms_to_acceptance_policy_interface() -> None:
    """UT-309g: the policy subclasses AcceptancePolicy and its ``select``
    accepts the interface's optional ``logits`` argument."""
    policy = _target_match_policy()
    tokens = torch.tensor([[1, 5, 9, 7]])
    only = _make_policy_candidate("only", tokens, _argmax_logits([0, 0, 0, 7], 1))

    assert isinstance(policy, AcceptancePolicy)
    assert policy.select([only], None) is only


def test_folded_generate_acceptance_rate_reported_across_steps() -> None:
    """UT-310a (srs §3.5-2): multi-step folded_generate reports a finite
    ``acceptance_rate`` mean in [0, 1]."""
    _raw, adapter, folded, _cache = _make_causal_setup("cpu")
    prompt = torch.tensor(_CAUSAL_PROMPT)

    result = folded_generate(adapter, prompt, max_new_tokens=4, folded_model=folded)

    assert result.num_folded_steps == 4
    rate = result.acceptance_rate
    assert isinstance(rate, float)
    assert math.isfinite(rate)
    assert 0.0 <= rate <= 1.0


@pytest.mark.parametrize(
    ("argmax_table", "max_new_tokens", "expected_rate", "expected_tokens"),
    [
        ([3, 3, 7, 7], 1, 1.0, [1, 5, 9, 7]),
        ([3, 3, 7, 0], 1, 0.0, [1, 5, 9, 7]),
        ([3, 3, 7, 7, 7], 2, 1.0, [1, 5, 9, 7, 7]),
        ([3, 3, 7, 0, 7], 2, 0.0, [1, 5, 9, 7, 0]),
        ([3, 3, 7, 7, 0], 2, 0.5, [1, 5, 9, 7, 7]),
    ],
)
def test_folded_generate_acceptance_rate_exact_controlled(
    argmax_table: list[int],
    max_new_tokens: int,
    expected_rate: float,
    expected_tokens: list[int],
) -> None:
    """UT-310b (srs §3.5-3): a FixedArgmax stub pins each step's acceptance
    rate exactly, so the reported ``acceptance_rate`` is the exact cross-step
    mean (1.0, 0.0, or a 1.0/0.0 mix averaging 0.5).

    The prompt has length 3 and position 2 of the table (7) picks the first
    appended token.  A step appending token ``x`` at length ``T`` accepts iff
    ``argmax_table[T] == x``; the greedy path appends ``argmax_table[T-1]``,
    making every step's rate and the appended token sequence fully scripted.
    """
    adapter = _make_fixed_argmax_adapter(argmax_table)
    prompt = torch.tensor(_AR004_PROMPT)

    result = folded_generate(adapter, prompt, max_new_tokens=max_new_tokens)

    assert result.num_folded_steps == max_new_tokens
    assert torch.equal(result.tokens, torch.tensor([expected_tokens]))
    assert result.acceptance_rate == pytest.approx(expected_rate)


def test_folded_generate_default_policy_zero_regression_reports_rate() -> None:
    """UT-310c (srs §3.5-2): with all-default arguments (greedy policy, no
    folded model, no draft generator) the token output matches the eager
    greedy baseline bit-for-bit and ``acceptance_rate`` is still reported."""
    raw, _adapter, _folded, _cache = _make_causal_setup("cpu")
    plain_adapter = FastDLLMAdapter(raw, num_layers=_CAUSAL_LAYERS, hidden_dim=_CAUSAL_HIDDEN)
    prompt = torch.tensor(_CAUSAL_PROMPT)

    expected = greedy_generate(plain_adapter, prompt, max_new_tokens=4)
    result = folded_generate(plain_adapter, prompt, max_new_tokens=4)

    assert torch.equal(result.tokens, expected)
    assert math.isfinite(result.acceptance_rate)
    assert 0.0 <= result.acceptance_rate <= 1.0


def test_folded_generate_zero_new_tokens_acceptance_rate_is_zero() -> None:
    """UT-310d (EC-12): with no generation steps the acceptance-rate mean
    falls back to the 0.0 sentinel."""
    _raw, adapter, folded, _cache = _make_causal_setup("cpu")
    prompt = torch.tensor(_CAUSAL_PROMPT)

    result = folded_generate(adapter, prompt, max_new_tokens=0, folded_model=folded)

    assert result.num_folded_steps == 0
    assert torch.equal(result.tokens, prompt)
    assert result.acceptance_rate == 0.0
