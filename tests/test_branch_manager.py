"""Tests for actfold.core.branch_manager."""

from __future__ import annotations

import pytest
import torch

from actfold.core.branch_manager import Branch, BranchManager


def test_create_root(device: str) -> None:
    manager = BranchManager()
    tokens = torch.randint(0, 100, (1, 8), device=device)
    hidden = torch.randn(2, 1, 8, 32, device=device)
    root = manager.create_root(tokens, hidden)
    assert root.branch_id.startswith("root")
    assert manager.root_id == root.branch_id
    assert len(manager.branches) == 1


def test_create_child(device: str) -> None:
    manager = BranchManager()
    root = manager.create_root(
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    child = manager.create_child(
        root.branch_id,
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    assert child.parent_id == root.branch_id
    assert child.branch_id in root.children


def test_get_parent(device: str) -> None:
    manager = BranchManager()
    root = manager.create_root(
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    child = manager.create_child(
        root.branch_id,
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    parent = manager.get_parent(child.branch_id)
    assert parent.branch_id == root.branch_id


def test_get_parent_root_raises(device: str) -> None:
    manager = BranchManager()
    root = manager.create_root(
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    with pytest.raises(ValueError, match="root"):
        manager.get_parent(root.branch_id)


def test_prune_rejected(device: str) -> None:
    manager = BranchManager()
    root = manager.create_root(
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    child = manager.create_child(
        root.branch_id,
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    removed = manager.prune_rejected(child.branch_id)
    assert child.branch_id in removed
    assert child.branch_id not in manager.branches
    assert child.branch_id not in root.children


def test_branch_validation(device: str) -> None:
    with pytest.raises(ValueError):
        Branch(
            branch_id="bad",
            parent_id=None,
            tokens=torch.randint(0, 100, (1, 8, 2), device=device),
            hidden_states=torch.randn(2, 1, 8, 32, device=device),
        )


def test_prune_rejected_reparents_children(device: str) -> None:
    manager = BranchManager()
    root = manager.create_root(
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    child = manager.create_child(
        root.branch_id,
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    grandchild = manager.create_child(
        child.branch_id,
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )

    removed = manager.prune_rejected(child.branch_id, include_subtree=False)
    assert removed == [child.branch_id]
    assert child.branch_id not in manager.branches
    assert grandchild.branch_id in manager.branches
    assert grandchild.parent_id == root.branch_id
    assert grandchild.branch_id in root.children


def test_prune_rejected_partial_root_raises(device: str) -> None:
    manager = BranchManager()
    root = manager.create_root(
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    with pytest.raises(ValueError, match="root"):
        manager.prune_rejected(root.branch_id, include_subtree=False)


def test_accept_branch(device: str) -> None:
    manager = BranchManager()
    root = manager.create_root(
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    manager.accept_branch(root.branch_id)
    assert manager.branches[root.branch_id].accepted is True


def test_clear(device: str) -> None:
    manager = BranchManager()
    manager.create_root(
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    manager.clear()
    assert not manager.branches
    assert manager.root_id is None


def test_align_tokens(device: str) -> None:
    manager = BranchManager()
    root = manager.create_root(
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    child = manager.create_child(
        root.branch_id,
        torch.randint(0, 100, (1, 8), device=device),
        torch.randn(2, 1, 8, 32, device=device),
    )
    h_parent, h_child = manager.align_tokens(root, child)
    assert h_parent.shape == (1, 8, 32)
    assert h_child.shape == (1, 8, 32)


def test_align_tokens_prefix_alignment(device: str) -> None:
    """UT-102: T_c > T_p returns the prefix-aligned pair (srs §3.2, design 4.3.3)."""
    manager = BranchManager()
    root = manager.create_root(
        torch.randint(0, 100, (1, 6), device=device),
        torch.randn(2, 1, 6, 32, device=device),
    )
    child = manager.create_child(
        root.branch_id,
        torch.randint(0, 100, (1, 9), device=device),
        torch.randn(2, 1, 9, 32, device=device),
    )
    h_parent, h_child = manager.align_tokens(root, child)
    assert h_parent.shape == (1, 6, 32)
    assert h_child.shape == (1, 6, 32)
    assert torch.equal(h_parent, root.hidden_states[0])
    assert torch.equal(h_child, child.hidden_states[0][:, :6, :])


def test_align_tokens_child_shorter_raises(device: str) -> None:
    """UT-103a: T_c < T_p is not a prefix extension and must raise ValueError."""
    manager = BranchManager()
    root = manager.create_root(
        torch.randint(0, 100, (1, 9), device=device),
        torch.randn(2, 1, 9, 32, device=device),
    )
    child = manager.create_child(
        root.branch_id,
        torch.randint(0, 100, (1, 4), device=device),
        torch.randn(2, 1, 4, 32, device=device),
    )
    with pytest.raises(ValueError, match="prefix"):
        manager.align_tokens(root, child)


def test_align_tokens_batch_mismatch_raises(device: str) -> None:
    """UT-103b: batch mismatch cannot be prefix-aligned and must raise ValueError."""
    manager = BranchManager()
    root = manager.create_root(
        torch.randint(0, 100, (1, 6), device=device),
        torch.randn(2, 1, 6, 32, device=device),
    )
    child = manager.create_child(
        root.branch_id,
        torch.randint(0, 100, (2, 8), device=device),
        torch.randn(2, 2, 8, 32, device=device),
    )
    with pytest.raises(ValueError, match="batch"):
        manager.align_tokens(root, child)
