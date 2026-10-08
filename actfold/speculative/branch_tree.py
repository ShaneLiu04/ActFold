"""Light-weight branch tree for folded generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class BranchNode:
    """A node in the active branch tree used by folded generation.

    Each node stores the token sequence, the logits produced by the last
    forward pass, and tree metadata.  Parent-child relationships reflect the
    speculative decoding history and are independent of the speculative
    ``Branch`` class used elsewhere.
    """

    branch_id: Any
    parent_id: Optional[Any]
    tokens: torch.Tensor  # [batch, seq_len]
    logits: Optional[torch.Tensor] = None  # [batch, seq_len, vocab_size]
    accepted: bool = True
    depth: int = 0
    children: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        """Return the sequence length."""
        return int(self.tokens.shape[1])


class BranchTree:
    """Manages a set of active generation branches.

    The tree supports lookup by ``branch_id``, parent/child tracking, and
    pruning of rejected branches.  It is intentionally simple: each generation
    step typically keeps a single accepted branch and may transiently hold
    draft candidates.

    Args:
        root: Root branch node (usually the encoded prompt).
    """

    def __init__(self, root: BranchNode) -> None:
        self._nodes: dict[Any, BranchNode] = {root.branch_id: root}
        self.root_id = root.branch_id

    def add(self, node: BranchNode) -> None:
        """Register ``node`` and link it to its parent."""
        self._nodes[node.branch_id] = node
        if node.parent_id is not None and node.parent_id in self._nodes:
            parent = self._nodes[node.parent_id]
            if node.branch_id not in parent.children:
                parent.children.append(node.branch_id)

    def get(self, branch_id: Any) -> Optional[BranchNode]:
        """Return the node for ``branch_id`` if it exists."""
        return self._nodes.get(branch_id)

    def get_ancestors(self, branch_id: Any, include_self: bool = False) -> list[Any]:
        """Return branch IDs from root to ``branch_id`` (optionally inclusive)."""
        ancestors: list[Any] = []
        current = self.get(branch_id)
        while current is not None and current.parent_id is not None:
            ancestors.append(current.parent_id)
            current = self.get(current.parent_id)
        ancestors.reverse()
        if include_self:
            ancestors.append(branch_id)
        return ancestors

    def prune(self, branch_id: Any) -> None:
        """Remove ``branch_id`` and recursively remove its descendants."""
        node = self._nodes.pop(branch_id, None)
        if node is None:
            return
        if node.parent_id is not None:
            parent = self._nodes.get(node.parent_id)
            if parent is not None and branch_id in parent.children:
                parent.children.remove(branch_id)
        for child_id in list(node.children):
            self.prune(child_id)

    def best_accepted_leaf(self) -> BranchNode:
        """Return the deepest accepted leaf node."""
        candidates = [n for n in self._nodes.values() if n.accepted]
        if not candidates:
            return self._nodes[self.root_id]
        return max(candidates, key=lambda n: n.depth)
