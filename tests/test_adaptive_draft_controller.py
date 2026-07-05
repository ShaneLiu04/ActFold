"""Tests for the adaptive draft-growth controller."""

from __future__ import annotations

import torch

from actfold.speculative.adaptive_draft_controller import AdaptiveDraftGrowthController
from actfold.speculative.branch import Branch
from actfold.speculative.draft_generator import DraftGenerator


def test_high_stability_expands() -> None:
    """High stable ratio and acceptance rate request more branches."""
    draft = DraftGenerator(vocab_size=16, mode="copy_flip")
    ctrl = AdaptiveDraftGrowthController(
        draft,
        max_branches=4,
        min_stable_ratio_to_expand=0.6,
        min_acceptance_to_expand=0.5,
    )
    parent = Branch(branch_id="p", parent_id=None, tokens=torch.tensor([[1, 2, 3]]))

    # Warm up acceptance history.
    for _ in range(5):
        ctrl.update_acceptance(True)

    children = ctrl.generate_children(parent, max_new_tokens=1, seed=0, recent_stable_ratio=0.9)
    assert len(children) > 1


def test_low_stability_stays_greedy() -> None:
    """Low stable ratio requests only a single branch."""
    draft = DraftGenerator(vocab_size=16, mode="copy_flip")
    ctrl = AdaptiveDraftGrowthController(
        draft,
        max_branches=4,
        min_stable_ratio_to_expand=0.6,
    )
    parent = Branch(branch_id="p", parent_id=None, tokens=torch.tensor([[1, 2, 3]]))

    children = ctrl.generate_children(parent, max_new_tokens=1, seed=0, recent_stable_ratio=0.1)
    assert len(children) == 1


def test_reset_clears_state() -> None:
    """reset clears the acceptance history."""
    draft = DraftGenerator(vocab_size=16, mode="copy_flip")
    ctrl = AdaptiveDraftGrowthController(draft, max_branches=4)
    ctrl.update_acceptance(True)
    ctrl.reset()
    assert ctrl._average_acceptance() == 0.5
