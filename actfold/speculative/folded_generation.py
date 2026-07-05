"""True end-to-end folded generation for ActFold.

This module implements :func:`folded_generate`, a generation loop where every
new token is produced by running the child sequence through the folded forward
path with the previous sequence as its parent.  Unlike the legacy
``greedy_generate``, this function passes ``branch_id`` / ``parent_branch_id`` /
``step_idx`` so that :class:`~actfold.core.folded_transformer.FoldedTransformerLayer`
can actually reuse parent activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from actfold.core.model_wrapper import FoldedModel
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from actfold.speculative.acceptance_policy import AcceptancePolicy, GreedyAcceptancePolicy
from actfold.speculative.branch_tree import BranchNode, BranchTree
from actfold.speculative.draft_generator import DraftGenerator
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
from actfold.utils.logger import get_logger

logger = get_logger("speculative.folded_generation")


@dataclass
class FoldedGenerationResult:
    """Result of a folded generation pass."""

    tokens: torch.Tensor
    stable_ratio: float
    num_folded_steps: int
    final_branch_id: Any


def _next_branch_id(parent_id: Any, token_idx: int, draft_idx: int = 0) -> str:
    """Generate a deterministic child branch ID."""
    return f"{parent_id}/t{token_idx}/d{draft_idx}"


def folded_generate(
    model: FastDLLMAdapter,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    folded_model: Optional[FoldedModel] = None,
    draft_generator: Optional[DraftGenerator] = None,
    acceptance_policy: Optional[AcceptancePolicy] = None,
    num_branches_per_step: int = 1,
    record_stability: bool = True,
    step_idx: int = 0,
) -> FoldedGenerationResult:
    """Generate ``max_new_tokens`` new tokens using the folded forward path.

    At each step a child branch is formed by appending a candidate token to the
    current accepted sequence.  The child forward pass is executed with
    ``parent_branch_id`` set to the current branch so that
    :class:`~actfold.core.folded_transformer.FoldedTransformerLayer` can reuse
    cached parent activations for stable tokens.

    Args:
        model: Model adapter that exposes ``forward()`` and ``embed()``.
        input_ids: Prompt token IDs ``[batch, prompt_len]``.
        max_new_tokens: Number of new tokens to generate.
        folded_model: Optional folded model used to resolve the folding context
            when the base model drops ActFold kwargs.
        draft_generator: Optional draft generator for producing multiple
            candidate continuations per step.  If ``None``, a single greedy
            candidate is produced.
        acceptance_policy: Policy for selecting among candidate branches.
            Defaults to greedy last-token selection.
        num_branches_per_step: Number of draft candidates to evaluate per step
            when ``draft_generator`` is provided.
        record_stability: Whether to collect per-layer stability profiles.
        step_idx: Current diffusion step index (1 for autoregressive models).

    Returns:
        FoldedGenerationResult containing the full generated sequence, the mean
        stable ratio across folded steps, and the final branch identifier.
    """
    if not record_stability:
        GLOBAL_STABILITY_PROFILER.set_enabled(False)

    policy = acceptance_policy or GreedyAcceptancePolicy()
    root = BranchNode(
        branch_id="root",
        parent_id=None,
        tokens=input_ids.clone(),
        depth=0,
    )
    tree = BranchTree(root)
    active = root

    for token_idx in range(max_new_tokens):
        candidates = _make_candidates(
            model=model,
            parent=active,
            token_idx=token_idx,
            folded_model=folded_model,
            draft_generator=draft_generator,
            num_branches=num_branches_per_step,
            step_idx=step_idx,
        )

        if not candidates:
            # No candidates produced; fall back to appending an EOS-like token.
            logger.warning("No candidates produced at step %d; stopping generation.", token_idx)
            break

        # Register candidates in the tree and run the folded forward pass.
        evaluated = []
        for node in candidates:
            tree.add(node)
            _run_folded_forward(model, node, folded_model, step_idx)
            evaluated.append(node)

        accepted = policy.select(evaluated)
        accepted.accepted = True
        active = accepted

        # Prune siblings that were not accepted to free cache.
        for node in evaluated:
            if node.branch_id != accepted.branch_id:
                tree.prune(node.branch_id)

    if not record_stability:
        GLOBAL_STABILITY_PROFILER.set_enabled(True)

    # Aggregate stability information across all folded steps.
    stable_ratio = active.metadata.get("stable_ratio", 0.0)
    return FoldedGenerationResult(
        tokens=active.tokens,
        stable_ratio=float(stable_ratio),
        num_folded_steps=active.depth,
        final_branch_id=active.branch_id,
    )


def _make_candidates(
    model: FastDLLMAdapter,
    parent: BranchNode,
    token_idx: int,
    folded_model: Optional[FoldedModel],
    draft_generator: Optional[DraftGenerator],
    num_branches: int,
    step_idx: int,
) -> list[BranchNode]:
    """Create candidate child branches by extending ``parent``."""
    candidates: list[BranchNode] = []

    if draft_generator is not None and num_branches > 1:
        # Use the draft generator to produce multiple candidate next tokens.
        from actfold.speculative.branch import Branch as SpecBranch

        spec_parent = SpecBranch(
            branch_id=parent.branch_id,
            parent_id=parent.parent_id,
            tokens=parent.tokens,
        )
        drafts = draft_generator.generate(
            spec_parent,
            num_branches=num_branches,
            max_new_tokens=1,
            seed=token_idx,
        )
        for draft_idx, draft in enumerate(drafts):
            child_id = _next_branch_id(parent.branch_id, token_idx, draft_idx)
            candidates.append(
                BranchNode(
                    branch_id=child_id,
                    parent_id=parent.branch_id,
                    tokens=draft.tokens.clone(),
                    depth=parent.depth + 1,
                )
            )
    else:
        # Single greedy candidate: we need the parent logits to pick the next
        # token.  If parent.logits is missing, run one non-folded forward pass
        # to obtain it.
        if parent.logits is None:
            _run_folded_forward(model, parent, folded_model=None, step_idx=step_idx)

        if parent.logits is None:
            raise RuntimeError("Failed to obtain logits for greedy candidate generation.")

        next_token = parent.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        child_tokens = torch.cat([parent.tokens, next_token], dim=-1)
        child_id = _next_branch_id(parent.branch_id, token_idx, 0)
        candidates.append(
            BranchNode(
                branch_id=child_id,
                parent_id=parent.branch_id,
                tokens=child_tokens,
                depth=parent.depth + 1,
            )
        )

    return candidates


def _run_folded_forward(
    model: FastDLLMAdapter,
    node: BranchNode,
    folded_model: Optional[FoldedModel],
    step_idx: int,
) -> None:
    """Execute the forward pass for ``node`` and store its logits."""
    parent_id = node.parent_id

    # Reset any stale profile so the profiler only records the current pass.
    GLOBAL_STABILITY_PROFILER.reset_branch(node.branch_id)

    if folded_model is not None:
        # Use the folded model wrapper; it will propagate branch context even
        # if the underlying base model drops kwargs.
        logits = folded_model(
            node.tokens,
            branch_id=node.branch_id,
            parent_branch_id=parent_id,
            step_idx=step_idx,
        )
    else:
        # Direct call: the adapter is responsible for forwarding ActFold kwargs
        # to the underlying model if it supports them.
        logits = model.forward(
            node.tokens,
            branch_id=node.branch_id,
            parent_branch_id=parent_id,
            step_idx=step_idx,
        )

    node.logits = logits

    profile = GLOBAL_STABILITY_PROFILER.get_profile(node.branch_id)
    if profile is not None:
        node.metadata["stable_ratio"] = profile.mean_stable_ratio
