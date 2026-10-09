"""Thread-local context for propagating branch identifiers through base models.

.. deprecated:: AR002
    This module exists only for the legacy
    :class:`~actfold.core.model_wrapper.FoldedModel` code path, where standard
    Transformer implementations (e.g. Hugging Face) do not forward arbitrary
    keyword arguments from ``model.forward`` down to each individual layer.
    The recommended path is ``ManualFoldedForward``
    (:mod:`actfold.models.architecture_utils`), which threads branch context
    through explicit ``branch_id`` / ``parent_branch_id`` / ``step_idx``
    arguments and never touches this thread-local state.  New code must not
    read ``FOLDING_CONTEXT``; route inference through the Manual path instead.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

FoldingContext = dict[str, Any] | None

FOLDING_CONTEXT: ContextVar[FoldingContext] = ContextVar(
    "actfold_folding_context",
    default=None,
)


class folding_scope:
    """Context manager that sets the ActFold branch context for a forward pass.

    Args:
        branch_id: Identifier of the current branch.
        parent_branch_id: Optional parent branch identifier.
        step_idx: Current diffusion step index.
    """

    def __init__(
        self,
        branch_id: str,
        parent_branch_id: str | None,
        step_idx: int,
    ) -> None:
        self._branch_id = branch_id
        self._parent_branch_id = parent_branch_id
        self._step_idx = step_idx
        self._token: Any = None

    def __enter__(self) -> None:
        self._token = FOLDING_CONTEXT.set(
            {
                "branch_id": self._branch_id,
                "parent_branch_id": self._parent_branch_id,
                "step_idx": self._step_idx,
            }
        )

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            FOLDING_CONTEXT.reset(self._token)
