"""Tests for actfold.speculative.draft_generator."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from actfold.core import SimilarityGate
from actfold.speculative.acceptance import (
    acceptance_rate,
    draft_region_mask,
    target_argmax_accept_mask,
)
from actfold.speculative.branch import Branch
from actfold.speculative.draft_generator import DraftGenerator


def test_random_mode(device: str) -> None:
    parent = Branch(
        branch_id="root",
        parent_id=None,
        tokens=torch.randint(0, 100, (2, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=100, mode="random")
    children = generator.generate(parent, num_branches=3)
    assert len(children) == 3
    for child in children:
        assert child.parent_id == "root"
        assert child.tokens.shape == (2, 8)


def test_copy_flip_mode(device: str) -> None:
    parent = Branch(
        branch_id="root",
        parent_id=None,
        tokens=torch.randint(0, 100, (1, 16), device=device),
    )
    generator = DraftGenerator(vocab_size=100, mode="copy_flip", flip_ratio=0.1)
    children = generator.generate(parent, num_branches=2, seed=42)
    assert len(children) == 2
    for child in children:
        assert child.tokens.shape == (1, 16)
        # Some tokens should differ because flip_ratio > 0 and seq_len is large enough.
        assert not torch.equal(child.tokens, parent.tokens)


def test_perturb_mode(device: str) -> None:
    parent = Branch(
        branch_id="root",
        parent_id=None,
        tokens=torch.randint(0, 100, (1, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=100, mode="perturb", perturbation_std=0.5)
    children = generator.generate(parent, num_branches=1, seed=42)
    assert len(children) == 1
    assert children[0].tokens.shape == (1, 8)


def test_invalid_mode() -> None:
    with pytest.raises(ValueError):
        DraftGenerator(vocab_size=100, mode="unknown")


def test_copy_flip_zero_flips(device: str) -> None:
    parent = Branch(
        branch_id="root",
        parent_id=None,
        tokens=torch.randint(0, 100, (1, 16), device=device),
    )
    generator = DraftGenerator(vocab_size=100, mode="copy_flip", flip_ratio=0.0)
    children = generator.generate(parent, num_branches=1, seed=42)
    assert torch.equal(children[0].tokens, parent.tokens)


def test_copy_flip_behavior_unchanged(device: str) -> None:
    """Regression: copy_flip with flip_ratio=1.0 still flips almost all positions."""
    parent = Branch(
        branch_id="root",
        parent_id=None,
        tokens=torch.randint(0, 100, (1, 16), device=device),
    )
    generator = DraftGenerator(vocab_size=100, mode="copy_flip", flip_ratio=1.0)
    children = generator.generate(parent, num_branches=1, seed=42)
    diff_count = int((children[0].tokens != parent.tokens).sum().item())
    assert diff_count >= 14  # almost all of the 16 positions differ


# ---------------------------------------------------------------------------
# T008: suffix_append / logits_draft modes, bounded child IDs, tau sensitivity
# ---------------------------------------------------------------------------


def test_suffix_append_default_mode(device: str) -> None:
    """suffix_append is the new default mode and protects the prompt prefix."""
    parent = Branch(
        branch_id="root",
        parent_id=None,
        tokens=torch.randint(0, 50, (2, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=50, prompt_length=4, flip_ratio=0.5)
    assert generator.mode == "suffix_append"

    children = generator.generate(parent, num_branches=2, seed=0)
    assert len(children) == 2
    for child in children:
        # max_new_tokens=0 keeps the child length equal to the parent length.
        assert child.tokens.shape == (2, 8)
        # Positions before prompt_length are never resampled.
        assert torch.equal(child.tokens[:, :4], parent.tokens[:, :4])

    # At least one position inside the default suffix region [4, 8) differs
    # across the two children or from the parent.
    region_parent = parent.tokens[:, 4:]
    region_children = [child.tokens[:, 4:] for child in children]
    differs_from_parent = any(not torch.equal(rc, region_parent) for rc in region_children)
    differs_between_children = not torch.equal(region_children[0], region_children[1])
    assert differs_from_parent or differs_between_children


def test_suffix_append_flip_ratio_zero_is_control(device: str) -> None:
    """flip_ratio=0 is the legal zero-divergence control: children equal parent."""
    parent = Branch(
        branch_id="root",
        parent_id=None,
        tokens=torch.randint(0, 50, (1, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=50, mode="suffix_append", prompt_length=4, flip_ratio=0.0)
    children = generator.generate(parent, num_branches=2, seed=0)
    for child in children:
        assert torch.equal(child.tokens, parent.tokens)


def test_suffix_append_flip_region_overrides_default(device: str) -> None:
    """An explicit flip_region overrides the prompt_length-derived default.

    With flip_region=(0, 2) and prompt_length=4, only positions [0, 2) may be
    resampled; positions [2, 8) never change.
    """
    parent = Branch(
        branch_id="root",
        parent_id=None,
        tokens=torch.randint(0, 50, (1, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=50, mode="suffix_append", prompt_length=4, flip_ratio=1.0)
    children = generator.generate(parent, num_branches=2, seed=0, flip_region=(0, 2))
    for child in children:
        assert child.tokens.shape == (1, 8)
        # Positions outside the flip region are never touched.
        assert torch.equal(child.tokens[:, 2:], parent.tokens[:, 2:])


def test_flip_region_validation(device: str) -> None:
    parent = Branch(
        branch_id="root",
        parent_id=None,
        tokens=torch.randint(0, 50, (1, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=50, mode="suffix_append")
    # start >= end
    with pytest.raises(ValueError):
        generator.generate(parent, num_branches=1, seed=0, flip_region=(3, 3))
    # start < 0
    with pytest.raises(ValueError):
        generator.generate(parent, num_branches=1, seed=0, flip_region=(-1, 2))
    # end > seq_len
    with pytest.raises(ValueError):
        generator.generate(parent, num_branches=1, seed=0, flip_region=(0, 99))
    # negative extension length
    with pytest.raises(ValueError):
        generator.generate(parent, num_branches=1, seed=0, max_new_tokens=-1)


def test_logits_draft_requires_parent_logits(device: str) -> None:
    parent = Branch(
        branch_id="root",
        parent_id=None,
        tokens=torch.randint(0, 50, (1, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=50, mode="logits_draft")
    with pytest.raises(RuntimeError):
        generator.generate(parent, num_branches=1, seed=0, parent_logits=None)


def test_logits_draft_samples_from_topk(device: str) -> None:
    """logits_draft resamples region positions from the top-k logits tokens."""
    parent = Branch(
        branch_id="root",
        parent_id=None,
        tokens=torch.randint(0, 20, (1, 6), device=device),
    )
    # Top-1 token at position i is token i.
    parent_logits = torch.zeros(1, 6, 20, device=device)
    for i in range(6):
        parent_logits[0, i, i] = 10.0

    generator = DraftGenerator(
        vocab_size=20, mode="logits_draft", top_k=1, flip_ratio=1.0, prompt_length=0
    )
    children = generator.generate(parent, num_branches=1, seed=0, parent_logits=parent_logits)
    expected = torch.arange(6, device=device).unsqueeze(0)
    assert torch.equal(children[0].tokens, expected)


def test_topk_validation() -> None:
    # The mode itself must be constructible before the invalid top_k is checked.
    DraftGenerator(vocab_size=10, mode="logits_draft")
    with pytest.raises(ValueError):
        DraftGenerator(vocab_size=10, mode="logits_draft", top_k=0)


def test_child_ids_bounded() -> None:
    """Child IDs use the bounded ``{session}:{counter}`` format (B/O(T^2) growth)."""
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.randint(0, 10, (1, 8)),
    )
    generator = DraftGenerator(vocab_size=10)

    all_ids: list[str] = []
    for _ in range(4):
        children = generator.generate(parent, num_branches=5)
        all_ids.extend(child.branch_id for child in children)

    assert len(all_ids) == 20
    # Unique across calls (no seed -> counter keeps increasing).
    assert len(set(all_ids)) == 20
    for branch_id in all_ids:
        assert len(branch_id) < 32
        assert "child_" not in branch_id


def test_seed_determinism_preserved(device: str) -> None:
    """Same seed -> identical child tokens AND identical bounded child IDs."""
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.randint(0, 10, (1, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=10, mode="suffix_append", flip_ratio=0.2)
    first = generator.generate(parent, num_branches=2, seed=123)
    second = generator.generate(parent, num_branches=2, seed=123)
    assert [child.branch_id for child in first] == [child.branch_id for child in second]
    for child_a, child_b in zip(first, second):
        assert torch.equal(child_a.tokens, child_b.tokens)


def test_suffix_append_tau_sensitivity_distinguishable(device: str) -> None:
    """B10: small-divergence drafts give a graded stable-ratio profile over tau.

    One-hot token proxies are compared with the "l2" similarity metric: an
    identical token scores exactly 1.0 while a resampled token scores
    ``1 - sqrt(2)/sqrt(vocab)`` (about 0.859 for vocab 100), so different taus
    separate identical vs resampled positions. (With the default cosine metric
    one-hot vectors are strictly orthogonal: the score is exactly 0.0 or 1.0
    and no graded profile is possible.)
    """
    tokens = torch.randint(0, 100, (1, 16), device=device)
    parent = Branch(branch_id="root", parent_id=None, tokens=tokens)
    generator = DraftGenerator(
        vocab_size=100, mode="suffix_append", prompt_length=4, flip_ratio=0.2
    )
    child = generator.generate(parent, num_branches=1, seed=1)[0]

    h_parent = F.one_hot(parent.tokens, 100).float()
    h_child = F.one_hot(child.tokens, 100).float()

    taus = [0.0, 0.3, 0.6, 0.9, 1.0]
    ratios = []
    for tau in taus:
        gate = SimilarityGate(tau=tau, metric="l2")
        mask = gate(h_child, h_parent)
        ratios.append(mask.float().mean().item())

    # Monotonically non-increasing in tau.
    for earlier, later in zip(ratios, ratios[1:]):
        assert earlier >= later
    # Graded profile: at least three distinguishable ratio levels.
    assert len(set(ratios)) >= 3
    # tau=0.0 accepts every position (resampled one-hot pairs still score ~0.859).
    assert ratios[0] == pytest.approx(1.0)
    # tau=1.0 rejects every position (identical pairs score exactly 1.0 and
    # the gate uses a strict >).
    assert ratios[-1] == pytest.approx(0.0)
    assert ratios[-1] < ratios[1]


def test_seed_does_not_pollute_global_rng(device: str) -> None:
    """AR002 srs 3.6: generate(seed=...) must leave the global RNG state untouched."""
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.randint(0, 10, (1, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=10, mode="suffix_append", flip_ratio=0.2)
    before = torch.get_rng_state()
    generator.generate(parent, num_branches=2, seed=7)
    after = torch.get_rng_state()
    assert torch.equal(before, after)

    # Also with appended tokens and the copy_flip mode (different sampling sites).
    gen_flip = DraftGenerator(vocab_size=10, mode="copy_flip", flip_ratio=0.2)
    before = torch.get_rng_state()
    gen_flip.generate(parent, num_branches=2, seed=7, max_new_tokens=3)
    assert torch.equal(before, torch.get_rng_state())


def test_seed_isolation_still_reproducible(device: str) -> None:
    """AR002 srs 3.6: local generator keeps same-seed reproducibility."""
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.randint(0, 10, (2, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=10, mode="suffix_append", flip_ratio=0.25)
    first = generator.generate(parent, num_branches=2, seed=99)
    second = generator.generate(parent, num_branches=2, seed=99)
    assert [child.branch_id for child in first] == [child.branch_id for child in second]
    for child_a, child_b in zip(first, second):
        assert torch.equal(child_a.tokens, child_b.tokens)


def test_seed_boundary_values(device: str) -> None:
    """AR002 design UT-002c: seed=0, negative, and extreme int64 seeds.

    Negative seeds pass through to ``manual_seed`` exactly like
    ``torch.manual_seed`` (which wraps them into the uint64 domain), so
    ``seed=-1`` must reproduce identically and equal ``seed=2**64 - 1``.
    """
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.randint(0, 10, (1, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=10, mode="suffix_append", flip_ratio=0.2)
    for seed in (0, -1, 2**63 - 1):
        first = generator.generate(parent, num_branches=2, seed=seed)
        second = generator.generate(parent, num_branches=2, seed=seed)
        for child_a, child_b in zip(first, second):
            assert torch.equal(child_a.tokens, child_b.tokens)

    # torch.manual_seed(-1) wraps to 2**64 - 1; the local generator must
    # share that semantics (same wrapped state -> identical children).
    wrapped = generator.generate(parent, num_branches=2, seed=2**64 - 1)
    negated = generator.generate(parent, num_branches=2, seed=-1)
    for child_a, child_b in zip(wrapped, negated):
        assert torch.equal(child_a.tokens, child_b.tokens)


# ---------------------------------------------------------------------------
# AR006: model_draft mode (UT-501~505, EX-801~803)
# ---------------------------------------------------------------------------


def _arange_stub(vocab_size: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Fixed-logits draft stub: arange logits broadcast to [batch, seq_len, V]."""

    def stub(tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        return (
            torch.arange(vocab_size, dtype=torch.float32, device=tokens.device)
            .view(1, 1, vocab_size)
            .expand(batch, seq_len, vocab_size)
        )

    return stub


def test_model_draft_flip_tokens_in_stub_topk(device: str) -> None:
    """UT-501: flip positions sample from the per-position stub top-k set."""
    vocab_size, seq_len = 20, 8
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.zeros(1, seq_len, dtype=torch.long, device=device),
    )
    generator = DraftGenerator(
        vocab_size=vocab_size,
        mode="model_draft",
        top_k=3,
        flip_ratio=1.0,
        prompt_length=0,
        draft_model=_arange_stub(vocab_size),
    )
    first = generator.generate(parent, num_branches=2, seed=0)

    # arange logits put the top-3 tokens at {17, 18, 19} for every position;
    # the parent tokens (all zeros) are outside that set, so every flipped
    # position must land inside the stub top-k.
    allowed = torch.tensor([17, 18, 19], device=device)
    for child in first:
        assert child.tokens.shape == (1, seq_len)
        assert torch.isin(child.tokens, allowed).all()

    second = generator.generate(parent, num_branches=2, seed=0)
    assert [child.branch_id for child in first] == [child.branch_id for child in second]
    for child_a, child_b in zip(first, second):
        assert torch.equal(child_a.tokens, child_b.tokens)


def test_model_draft_requires_draft_model_at_generate_time(device: str) -> None:
    """UT-502 / EX-801: missing draft_model raises RuntimeError at generate time.

    Construction must succeed: the mode/parameter combination is validated
    when the draft model is used (same timing as the logits_draft
    parent_logits precedent).
    """
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.randint(0, 50, (1, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=50, mode="model_draft")
    assert generator.mode == "model_draft"
    with pytest.raises(RuntimeError):
        generator.generate(parent, num_branches=1, seed=0)


def test_model_draft_rejects_wrong_shape_logits(device: str) -> None:
    """UT-503a: draft logits shape != (batch, seq_len, vocab_size) -> ValueError."""
    vocab_size, seq_len = 20, 8
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.zeros(1, seq_len, dtype=torch.long, device=device),
    )

    def bad_shape_stub(tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        return torch.zeros(batch, seq_len, vocab_size - 1, device=tokens.device)

    generator = DraftGenerator(
        vocab_size=vocab_size, mode="model_draft", draft_model=bad_shape_stub
    )
    with pytest.raises(ValueError):
        generator.generate(parent, num_branches=1, seed=0)


def test_model_draft_rejects_int_dtype_logits(device: str) -> None:
    """UT-503b: non-floating-point draft logits -> ValueError."""
    vocab_size, seq_len = 20, 8
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.zeros(1, seq_len, dtype=torch.long, device=device),
    )

    def int_dtype_stub(tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        return torch.zeros(batch, seq_len, vocab_size, dtype=torch.int64, device=tokens.device)

    generator = DraftGenerator(
        vocab_size=vocab_size, mode="model_draft", draft_model=int_dtype_stub
    )
    with pytest.raises(ValueError):
        generator.generate(parent, num_branches=1, seed=0)


def test_model_draft_rejects_cross_device_logits() -> None:
    """UT-503c: draft logits on a device != parent.tokens -> ValueError."""
    vocab_size, seq_len = 20, 8
    try:
        torch.zeros(1, device="meta")
    except RuntimeError:
        pytest.skip("meta device not available in this torch build")
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.zeros(1, seq_len, dtype=torch.long),
    )

    def meta_stub(tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        return torch.zeros(batch, seq_len, vocab_size, device="meta")

    generator = DraftGenerator(vocab_size=vocab_size, mode="model_draft", draft_model=meta_stub)
    with pytest.raises(ValueError):
        generator.generate(parent, num_branches=1, seed=0)


def test_model_draft_forward_error_propagates(device: str) -> None:
    """EX-803: a draft model forward exception propagates unchanged."""

    def boom_stub(tokens: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom")

    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.randint(0, 20, (1, 8), device=device),
    )
    generator = DraftGenerator(vocab_size=20, mode="model_draft", draft_model=boom_stub)
    with pytest.raises(RuntimeError, match="boom"):
        generator.generate(parent, num_branches=1, seed=0)


def test_model_draft_ignores_parent_logits(device: str) -> None:
    """UT-504: parent_logits is ignored in model_draft mode (draft independence)."""
    vocab_size, seq_len = 20, 8
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.randint(0, vocab_size, (1, seq_len), device=device),
    )
    generator = DraftGenerator(
        vocab_size=vocab_size,
        mode="model_draft",
        top_k=3,
        flip_ratio=0.5,
        prompt_length=2,
        draft_model=_arange_stub(vocab_size),
    )
    without = generator.generate(parent, num_branches=2, seed=7)
    parent_logits = torch.randn(1, seq_len, vocab_size, device=device)
    with_logits = generator.generate(parent, num_branches=2, seed=7, parent_logits=parent_logits)
    for child_a, child_b in zip(without, with_logits):
        assert torch.equal(child_a.tokens, child_b.tokens)


def test_model_draft_append_tokens_in_last_position_topk(device: str) -> None:
    """UT-505: appended tokens sample from the last position's stub top-k."""
    vocab_size, seq_len = 20, 6
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.zeros(1, seq_len, dtype=torch.long, device=device),
    )
    generator = DraftGenerator(
        vocab_size=vocab_size,
        mode="model_draft",
        top_k=3,
        flip_ratio=0.0,
        prompt_length=0,
        draft_model=_arange_stub(vocab_size),
    )
    first = generator.generate(parent, num_branches=1, seed=0, max_new_tokens=4)

    allowed = torch.tensor([17, 18, 19], device=device)
    for child in first:
        assert child.tokens.shape == (1, seq_len + 4)
        # flip_ratio=0 keeps the parent prefix untouched...
        assert torch.equal(child.tokens[:, :seq_len], parent.tokens)
        # ...and each appended token comes from the last position's top-k.
        assert torch.isin(child.tokens[:, seq_len:], allowed).all()

    second = generator.generate(parent, num_branches=1, seed=0, max_new_tokens=4)
    for child_a, child_b in zip(first, second):
        assert torch.equal(child_a.tokens, child_b.tokens)


def test_model_draft_seed_does_not_pollute_global_rng(device: str) -> None:
    """AR002 srs 3.6 (model_draft): seeded generate leaves the global RNG untouched."""
    parent = Branch(
        branch_id="p",
        parent_id=None,
        tokens=torch.randint(0, 20, (1, 8), device=device),
    )
    generator = DraftGenerator(
        vocab_size=20,
        mode="model_draft",
        top_k=3,
        flip_ratio=0.5,
        draft_model=_arange_stub(20),
    )
    before = torch.get_rng_state()
    generator.generate(parent, num_branches=2, seed=7, max_new_tokens=3)
    assert torch.equal(before, torch.get_rng_state())


# ---------------------------------------------------------------------------
# AR006: acceptance-rate gradient fixtures (UT-506~508)
#
# Acceptance math (design 2.3): _sample_from_topk is uniform top-k sampling,
# so accept rate = P(target argmax in draft top-k) / k. The acceptance-rate
# fixture pins top_k=2 (k=1 makes the logits_draft comparator trivially 1.0;
# k=10 caps the model_draft-vs-random gap at 1/k - 1/V < 0.1).
# ---------------------------------------------------------------------------

_GRADIENT_V = 1000
_GRADIENT_T = 64
_GRADIENT_FLIP_RATIO = 0.25  # -> max(1, int(64 * 0.25)) = 16 flip positions


class _CausalTokenTarget(torch.nn.Module):
    """4-layer token-wise causal target fixture (UT-506~508).

    ``logits[:, i]`` depends only on ``tokens[:, :i+1]`` (AGENTS #40
    same-position convention). The first layer dominates (scale 1.0) while
    the three later layers are weak (scale 0.05), so a single-layer draft
    sharing the first layer is strongly correlated with the target without
    being equal to it.
    """

    def __init__(self, vocab_size: int, num_layers: int = 4, seed: int = 7) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.first = torch.nn.Parameter(
            torch.randn(vocab_size, vocab_size, generator=g) * (1.0 / vocab_size**0.5)
        )
        self.rest = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.randn(vocab_size, vocab_size, generator=g) * (0.05 / vocab_size**0.5)
                )
                for _ in range(num_layers - 1)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(tokens, self.first.shape[0]).float()
        x = torch.cumsum(x, dim=1)
        x = x + x @ self.first.T
        for layer in self.rest:
            x = x + x @ layer.T
        return x


class _CorrelatedDraft(torch.nn.Module):
    """Single-layer draft sharing the target's first layer plus a small bias."""

    def __init__(self, target: _CausalTokenTarget, seed: int = 11) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.first = target.first
        self.bias = torch.nn.Parameter(torch.randn(target.first.shape[0], generator=g) * 0.1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(tokens, self.first.shape[0]).float()
        x = torch.cumsum(x, dim=1)
        x = x + x @ self.first.T
        return x + self.bias


class _UncorrelatedDraft(torch.nn.Module):
    """Single-layer draft with a fresh random transform (zero correlation).

    Unlike the correlated draft this one has **no identity residual**
    (``x @ U.T`` instead of ``x + x @ U.T``): the residual term would share
    the raw one-hot-cumsum structure with the target and produce a spurious
    ~0.4 acceptance rate even with random weights.
    """

    def __init__(self, vocab_size: int, seed: int = 23) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.first = torch.nn.Parameter(
            torch.randn(vocab_size, vocab_size, generator=g) * (1.0 / vocab_size**0.5)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(tokens, self.first.shape[0]).float()
        x = torch.cumsum(x, dim=1)
        return x @ self.first.T


def _measure_accept_rate(
    mode: str,
    parent_tokens: torch.Tensor,
    target_logits: torch.Tensor,
    draft_model: Callable[[torch.Tensor], torch.Tensor] | None,
    seed: int = 123,
) -> float:
    """Generate one child in ``mode`` and measure its acceptance rate."""
    parent = Branch(branch_id="p", parent_id=None, tokens=parent_tokens)
    generator = DraftGenerator(
        vocab_size=_GRADIENT_V,
        mode=mode,
        flip_ratio=_GRADIENT_FLIP_RATIO,
        top_k=2,
        prompt_length=0,
        draft_model=draft_model,
    )
    child = generator.generate(
        parent,
        num_branches=1,
        seed=seed,
        parent_logits=target_logits if mode == "logits_draft" else None,
    )[0]
    accept = target_argmax_accept_mask(child.tokens, target_logits)
    region = draft_region_mask(parent_tokens, child.tokens)
    return acceptance_rate(accept, region)


def test_model_draft_acceptance_beats_random(device: str) -> None:
    """UT-506: correlated model_draft acceptance exceeds random by >= 0.1."""
    target = _CausalTokenTarget(_GRADIENT_V).to(device)
    draft = _CorrelatedDraft(target).to(device)
    # Seed 7 keeps the parent stream disjoint from the draft stream (seed 123):
    # on CPU both would share one RNG stream and the random draft would
    # reproduce the parent tokens exactly (empty draft region -> rate 1.0).
    g = torch.Generator(device="cpu").manual_seed(7)
    parent_tokens = torch.randint(0, _GRADIENT_V, (1, _GRADIENT_T), generator=g).to(device)
    target_logits = target(parent_tokens)

    # Fixture sanity: the draft argmax hit rate is strictly inside (0, 1).
    hit = draft(parent_tokens).argmax(dim=-1) == target_logits.argmax(dim=-1)
    hit_rate = hit.float().mean().item()
    assert 0.0 < hit_rate < 1.0

    model_rate = _measure_accept_rate("model_draft", parent_tokens, target_logits, draft)
    random_rate = _measure_accept_rate("random", parent_tokens, target_logits, None)
    assert model_rate - random_rate >= 0.1


def test_model_draft_acceptance_below_logits_draft_upper_bound(device: str) -> None:
    """UT-507: model_draft acceptance <= logits_draft comparator (not 1.0)."""
    target = _CausalTokenTarget(_GRADIENT_V).to(device)
    draft = _CorrelatedDraft(target).to(device)
    g = torch.Generator(device="cpu").manual_seed(123)
    parent_tokens = torch.randint(0, _GRADIENT_V, (1, _GRADIENT_T), generator=g).to(device)
    target_logits = target(parent_tokens)

    model_rate = _measure_accept_rate("model_draft", parent_tokens, target_logits, draft)
    logits_rate = _measure_accept_rate("logits_draft", parent_tokens, target_logits, None)
    assert model_rate <= logits_rate
    assert logits_rate < 1.0  # comparator is measured, not the trivial argmax


def test_uncorrelated_draft_acceptance_near_random_floor(device: str) -> None:
    """UT-508: uncorrelated draft acceptance stays within 0.05 of random."""
    target = _CausalTokenTarget(_GRADIENT_V).to(device)
    draft = _UncorrelatedDraft(_GRADIENT_V).to(device)
    # Seed 7 keeps the parent stream disjoint from the draft stream (seed 123).
    g = torch.Generator(device="cpu").manual_seed(7)
    parent_tokens = torch.randint(0, _GRADIENT_V, (1, _GRADIENT_T), generator=g).to(device)
    target_logits = target(parent_tokens)

    uncorrelated_rate = _measure_accept_rate("model_draft", parent_tokens, target_logits, draft)
    random_rate = _measure_accept_rate("random", parent_tokens, target_logits, None)
    assert abs(uncorrelated_rate - random_rate) < 0.05
