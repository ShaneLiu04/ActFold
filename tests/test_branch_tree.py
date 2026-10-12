"""Direct API pins for ``actfold.speculative.branch_tree`` (AR013).

``BranchTree.add``/``prune`` are exercised by ``folded_generate`` in
production but previously had no direct unit tests (prune was only covered
indirectly through generation-level integration tests).  These are behavior
pins: they record the current implementation — including the prune-root
quirk explicitly flagged as "current behavior, not normative" — rather than
a desired redesign.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from actfold.speculative.branch_tree import BranchNode, BranchTree


def _node(branch_id: str, parent_id: str | None, seq_len: int) -> BranchNode:
    """Build a ``BranchNode`` mirroring the folded_generation construction shape."""
    from actfold.speculative.branch_tree import BranchNode

    tokens = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len)
    return BranchNode(branch_id=branch_id, parent_id=parent_id, tokens=tokens)


def _build_tree() -> tuple[BranchTree, BranchNode, BranchNode, BranchNode, BranchNode]:
    """Build root -> mid -> leaf plus a sibling side branch off root."""
    from actfold.speculative.branch_tree import BranchTree

    root = _node("root", None, 5)
    tree = BranchTree(root)
    mid = _node("mid", "root", 6)
    leaf = _node("leaf", "mid", 7)
    sibling = _node("sibling", "root", 6)
    tree.add(mid)
    tree.add(leaf)
    tree.add(sibling)
    return tree, root, mid, leaf, sibling


def test_branch_tree_add_get_roundtrip_and_links() -> None:
    """UT-1205: add/get roundtrip, parent links, duplicate guard, orphan add, seq_len.

    Duplicate ``add`` of the same node must not duplicate the entry in
    ``parent.children``; adding a node whose parent is unregistered must
    neither raise nor attach it to any children list.  ``BranchNode.seq_len``
    must equal ``tokens.shape[1]``.
    """
    tree, root, mid, leaf, sibling = _build_tree()

    assert tree.get("mid") is mid
    assert tree.get("leaf") is leaf
    assert tree.get("sibling") is sibling
    assert tree.get("nonexistent") is None

    assert "mid" in root.children
    assert "sibling" in root.children
    assert "leaf" in mid.children

    tree.add(mid)
    assert root.children.count("mid") == 1

    orphan = _node("orphan", "ghost", 4)
    tree.add(orphan)
    assert tree.get("orphan") is orphan
    assert "orphan" not in root.children
    assert "orphan" not in mid.children
    assert "orphan" not in sibling.children

    assert root.seq_len == 5
    assert mid.seq_len == 6
    assert leaf.seq_len == 7
    assert orphan.seq_len == 4


def test_branch_tree_get_ancestors() -> None:
    """UT-1206: root-to-node ordering, include_self toggle, unregistered-id behavior.

    For an unregistered id the current implementation returns ``[]`` when
    ``include_self=False`` but ``[branch_id]`` when ``include_self=True``
    (the while loop terminates on the missing node and the id is still
    appended) — pinned as-is.
    """
    tree, _root, _mid, leaf, _sibling = _build_tree()

    assert tree.get_ancestors("leaf") == ["root", "mid"]
    assert tree.get_ancestors("leaf", include_self=True) == ["root", "mid", "leaf"]
    assert tree.get_ancestors("mid") == ["root"]
    assert tree.get_ancestors("mid", include_self=True) == ["root", "mid"]
    assert tree.get_ancestors("root") == []
    assert tree.get_ancestors("root", include_self=True) == ["root"]

    assert tree.get_ancestors("nonexistent") == []
    assert tree.get_ancestors("nonexistent", include_self=True) == ["nonexistent"]


def test_branch_tree_prune() -> None:
    """UT-1207: recursive descendant removal, unlinking, missing-id no-op, prune-root.

    Pruning a middle node removes its whole subtree, unlinks it from the
    parent's children, and leaves siblings untouched.  Pruning the root is
    pinned as current behavior, not normative: the root and its remaining
    subtrees are removed recursively (``get`` returns None) yet
    ``tree.root_id`` keeps dangling onto the removed id because prune never
    clears it.
    """
    tree, root, mid, leaf, sibling = _build_tree()

    tree.prune("mid")
    assert tree.get("mid") is None
    assert tree.get("leaf") is None
    assert "mid" not in root.children
    assert tree.get("root") is root
    assert tree.get("sibling") is sibling
    assert "sibling" in root.children

    before: dict[Any, int] = {k: len(v.children) for k, v in tree._nodes.items()}
    tree.prune("nonexistent")
    after: dict[Any, int] = {k: len(v.children) for k, v in tree._nodes.items()}
    assert after == before

    tree.prune("root")
    assert tree.get("root") is None
    assert tree.get("sibling") is None
    assert tree.root_id == "root"


def test_branch_tree_removed_dead_api() -> None:
    """UT-1208: dead ``best_accepted_leaf`` stays removed and the live API stays.

    ``best_accepted_leaf`` had zero callers and zero tests (guide 6.1 :228
    dead code); its removal is pinned so a reintroduction without a
    production consumer gets flagged in review.  The live surface exercised
    by ``folded_generate`` (add/prune) plus the navigation face (get,
    get_ancestors) must remain, and the module must stay importable.
    """
    from actfold.speculative import branch_tree as bt
    from actfold.speculative.branch_tree import BranchNode, BranchTree

    assert not hasattr(BranchTree, "best_accepted_leaf")
    assert hasattr(BranchTree, "add")
    assert hasattr(BranchTree, "get")
    assert hasattr(BranchTree, "get_ancestors")
    assert hasattr(BranchTree, "prune")
    assert bt.BranchTree is BranchTree
    assert bt.BranchNode is BranchNode
