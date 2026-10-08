"""Tests for true end-to-end folded generation."""

from __future__ import annotations


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
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
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
