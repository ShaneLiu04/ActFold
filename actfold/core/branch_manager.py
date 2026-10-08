"""Branch tree management and parent-child alignment."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Branch:
    """Represents one candidate generation trajectory.

    Attributes:
        branch_id: Globally unique branch identifier.
        parent_id: Identifier of the parent branch, or None for the root.
        tokens: Token tensor of shape ``[batch, seq_len]``.
        hidden_states: Cached hidden states of shape
            ``[num_layers, batch, seq_len, hidden_dim]``.
        accepted: Whether this branch has been accepted by verification.
        children: List of child branch IDs.
    """

    branch_id: str
    parent_id: str | None
    tokens: torch.Tensor
    hidden_states: torch.Tensor
    accepted: bool = False
    children: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.tokens.dim() != 2:
            raise ValueError(f"tokens must be 2D [batch, seq_len], got {self.tokens.shape}")
        if self.hidden_states.dim() != 4:
            raise ValueError(
                f"hidden_states must be 4D [layers, batch, seq_len, hidden_dim], "
                f"got {self.hidden_states.shape}"
            )


class BranchManager:
    """Maintains the branch tree and handles parent lookup / token alignment.

    The root branch has ``parent_id=None``. Every child branch stores a
    reference to its parent and is registered in the parent's ``children`` list.
    """

    def __init__(self) -> None:
        self.branches: dict[str, Branch] = {}
        self.root_id: str | None = None
        self._counter: int = 0

    def _next_id(self, prefix: str = "branch") -> str:
        """Generate a unique branch identifier."""
        branch_id = f"{prefix}_{self._counter}"
        self._counter += 1
        return branch_id

    def create_root(self, tokens: torch.Tensor, hidden_states: torch.Tensor) -> Branch:
        """Create the root branch.

        Args:
            tokens: Root token tensor ``[batch, seq_len]``.
            hidden_states: Root hidden states ``[layers, batch, seq_len, hidden_dim]``.

        Returns:
            The root Branch instance.
        """
        if self.root_id is not None:
            raise RuntimeError("Root branch already exists.")

        branch_id = self._next_id("root")
        branch = Branch(
            branch_id=branch_id,
            parent_id=None,
            tokens=tokens,
            hidden_states=hidden_states,
        )
        self.branches[branch_id] = branch
        self.root_id = branch_id
        return branch

    def create_child(
        self,
        parent_id: str,
        tokens: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Branch:
        """Create a child branch under ``parent_id``.

        Args:
            parent_id: Parent branch identifier.
            tokens: Child token tensor.
            hidden_states: Child hidden states.

        Returns:
            The child Branch instance.

        Raises:
            KeyError: If ``parent_id`` does not exist.
        """
        if parent_id not in self.branches:
            raise KeyError(f"Parent branch not found: {parent_id}")

        branch_id = self._next_id("child")
        branch = Branch(
            branch_id=branch_id,
            parent_id=parent_id,
            tokens=tokens,
            hidden_states=hidden_states,
        )
        self.branches[parent_id].children.append(branch_id)
        self.branches[branch_id] = branch
        return branch

    def get_parent(self, branch_id: str) -> Branch:
        """Return the immediate parent branch of ``branch_id``.

        Raises:
            KeyError: If the branch or its parent does not exist.
            ValueError: If ``branch_id`` is the root.
        """
        if branch_id not in self.branches:
            raise KeyError(f"Branch not found: {branch_id}")
        branch = self.branches[branch_id]
        if branch.parent_id is None:
            raise ValueError(f"Branch {branch_id} is the root and has no parent.")
        return self.branches[branch.parent_id]

    def align_tokens(
        self,
        parent: Branch,
        child: Branch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token-aligned parent and child hidden states.

        For the first implementation, both branches are assumed to share the
        same sequence length. Future versions may implement prefix matching for
        variable-length branches.

        Args:
            parent: Parent branch.
            child: Child branch.

        Returns:
            Tuple ``(h_parent, h_child)`` both of shape
            ``[batch, seq_len, hidden_dim]`` for a single layer (typically layer 0
            input hidden states, or the caller slices the desired layer).
        """
        if parent.tokens.shape[1] != child.tokens.shape[1]:
            raise NotImplementedError("Variable sequence length alignment is not yet implemented.")
        # Return the input hidden states at layer 0.
        return parent.hidden_states[0], child.hidden_states[0]

    def prune_rejected(self, branch_id: str, include_subtree: bool = True) -> list[str]:
        """Remove a rejected branch and optionally its subtree.

        Args:
            branch_id: Branch to remove.
            include_subtree: If True, also remove all descendants. If False,
                children of the removed branch are reparented to its parent so
                the tree remains connected. The root branch cannot be partially
                pruned.

        Returns:
            List of removed branch IDs.

        Raises:
            ValueError: If ``include_subtree`` is False and ``branch_id`` is the
                root, or if it has children and no parent exists.
        """
        if branch_id not in self.branches:
            return []

        target = self.branches[branch_id]
        if not include_subtree:
            if target.parent_id is None:
                raise ValueError("Cannot partially prune the root branch.")
            if target.children:
                # Reparent children to the target's parent.
                grandparent = self.branches[target.parent_id]
                for child_id in target.children:
                    child = self.branches[child_id]
                    child.parent_id = target.parent_id
                grandparent.children.extend(target.children)
                # Remove the target from its parent but keep its children.
                if branch_id in grandparent.children:
                    grandparent.children.remove(branch_id)
                del self.branches[branch_id]
                return [branch_id]
            # No children: fall through to normal deletion.

        removed: list[str] = []
        stack = [branch_id]
        while stack:
            current_id = stack.pop()
            current = self.branches[current_id]
            removed.append(current_id)
            stack.extend(current.children)

            # Remove this branch from its parent's children list.
            if current.parent_id is not None and current.parent_id in self.branches:
                parent = self.branches[current.parent_id]
                if current_id in parent.children:
                    parent.children.remove(current_id)

            del self.branches[current_id]

            if self.root_id == current_id:
                self.root_id = None

        return removed

    def accept_branch(self, branch_id: str) -> None:
        """Mark a branch as accepted."""
        if branch_id not in self.branches:
            raise KeyError(f"Branch not found: {branch_id}")
        self.branches[branch_id].accepted = True

    def clear(self) -> None:
        """Reset the manager state."""
        self.branches.clear()
        self.root_id = None
        self._counter = 0
