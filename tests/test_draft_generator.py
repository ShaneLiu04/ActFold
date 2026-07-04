"""Tests for actfold.speculative.draft_generator."""

from __future__ import annotations

import pytest
import torch

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
