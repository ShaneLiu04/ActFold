"""Tests for actfold.speculative.draft_generator."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from actfold.core import SimilarityGate
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
