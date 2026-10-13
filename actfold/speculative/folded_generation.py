"""True end-to-end folded generation for ActFold.

This module implements :func:`folded_generate`, a generation loop where every
new token is produced by running the child sequence through the folded forward
path with the previous sequence as its parent.  Unlike the legacy
``greedy_generate``, this function passes ``branch_id`` / ``parent_branch_id`` /
``step_idx`` so that :class:`~actfold.core.folded_transformer.FoldedTransformerLayer`
can actually reuse parent activations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import torch

from actfold.core.model_wrapper import FoldedModel
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER
from actfold.speculative.acceptance import (
    acceptance_rate,
    draft_region_mask,
    target_argmax_accept_mask,
)
from actfold.speculative.acceptance_policy import AcceptancePolicy, GreedyAcceptancePolicy
from actfold.speculative.branch_tree import BranchNode, BranchTree
from actfold.speculative.draft_generator import DraftGenerator
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
from actfold.utils.logger import get_logger

logger = get_logger("speculative.folded_generation")

# Session-scoped prefix + monotonic counter keep branch IDs at bounded length.
# Chaining parent IDs into child IDs (the old format) grew O(T^2) in total
# string memory over a generation run.
_BRANCH_SESSION = uuid4().hex[:8]
_BRANCH_COUNTER = itertools.count(1)


@dataclass
class FoldedGenerationResult:
    """Result of a folded generation pass."""

    tokens: torch.Tensor
    stable_ratio: float
    num_folded_steps: int
    final_branch_id: Any
    acceptance_rate: float = 0.0


def _next_branch_id(parent_id: Any, token_idx: int, draft_idx: int = 0) -> str:
    """Generate a bounded-length, unique child branch ID.

    The ID is ``f"{session}:{counter}"`` so every ID has constant length
    regardless of generation depth.
    """
    return f"{_BRANCH_SESSION}:{next(_BRANCH_COUNTER)}"


def folded_generate(
    model: FastDLLMAdapter,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    folded_model: FoldedModel | ManualFoldedForward | None = None,
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
        folded_model: Optional folded model (FoldedModel or ManualFoldedForward)
            used to resolve the folding context
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
    profiler_was_enabled = getattr(GLOBAL_STABILITY_PROFILER, "enabled", True)
    if not record_stability:
        GLOBAL_STABILITY_PROFILER.set_enabled(False)

    try:
        policy = acceptance_policy or GreedyAcceptancePolicy()
        root = BranchNode(
            branch_id="root",
            parent_id=None,
            tokens=input_ids.clone(),
            depth=0,
        )
        tree = BranchTree(root)
        active = root
        step_ratios: list[float] = []
        step_acceptances: list[float] = []

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
                logger.warning(
                    "No candidates produced at step %d; stopping generation.", token_idx
                )
                break

            # Register candidates in the tree and run the folded forward pass.
            evaluated: list[BranchNode] = []
            if isinstance(folded_model, ManualFoldedForward) and _batched_forward_supported(
                folded_model, candidates
            ):
                # AR015 (P1-7 first tranche): same-parent equal-length
                # candidates share one batched folded forward instead of N
                # serial passes; every other shape takes the exact serial
                # loop below.
                for node in candidates:
                    tree.add(node)
                _run_folded_forward_batched(model, candidates, folded_model, step_idx)
                evaluated = list(candidates)
            else:
                for node in candidates:
                    tree.add(node)
                    _run_folded_forward(model, node, folded_model, step_idx)
                    evaluated.append(node)

            accepted = policy.select(evaluated)
            accepted.accepted = True

            # True acceptance semantics (AR004), decoupled from the policy:
            # score the accepted candidate's new claims (diffs vs the parent
            # plus the appended token) against the target argmax of its own
            # forward logits. Computed against the pre-advance parent snapshot.
            # ``_run_folded_forward`` always stores logits on evaluated nodes.
            assert accepted.logits is not None
            parent_tokens = active.tokens
            accept_mask = target_argmax_accept_mask(accepted.tokens, accepted.logits)
            draft_mask = draft_region_mask(parent_tokens, accepted.tokens)
            step_rate = acceptance_rate(accept_mask, draft_mask)
            accepted.metadata["acceptance_rate"] = step_rate
            step_acceptances.append(float(step_rate))

            active = accepted
            step_ratio = active.metadata.get("stable_ratio", 0.0)
            step_ratios.append(float(step_ratio))

            # Prune siblings that were not accepted to free cache. Without the
            # explicit ``clear_branch`` the activation cache retained every
            # rejected draft's buffers until the end of generation.
            for node in evaluated:
                if node.branch_id != accepted.branch_id:
                    tree.prune(node.branch_id)
                    if folded_model is not None and folded_model.cache is not None:
                        folded_model.cache.clear_branch(node.branch_id)
    finally:
        # Always restore the global profiler state: an exception mid-loop must
        # not leak a disabled (or re-enabled) profiler into subsequent calls.
        if not record_stability:
            GLOBAL_STABILITY_PROFILER.set_enabled(profiler_was_enabled)

    # Aggregate stability information across all folded steps: the reported
    # stable ratio is the mean over every folded step, not the last step's
    # value (each step overwrites node metadata, so the last-step read was a
    # silent bug that skewed downstream TFLOPs estimates).
    stable_ratio = (
        sum(step_ratios) / len(step_ratios) if step_ratios else 0.0
    )
    mean_acceptance = (
        sum(step_acceptances) / len(step_acceptances) if step_acceptances else 0.0
    )
    return FoldedGenerationResult(
        tokens=active.tokens,
        stable_ratio=float(stable_ratio),
        num_folded_steps=active.depth,
        final_branch_id=active.branch_id,
        acceptance_rate=float(mean_acceptance),
    )


def _make_candidates(
    model: FastDLLMAdapter,
    parent: BranchNode,
    token_idx: int,
    folded_model: FoldedModel | ManualFoldedForward | None,
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
    folded_model: FoldedModel | ManualFoldedForward | None,
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


def _batched_forward_supported(
    folded_model: FoldedModel | ManualFoldedForward | None,
    candidates: list[BranchNode],
) -> bool:
    """Whether ``candidates`` can share one batched folded forward (AR015).

    Conditions (all required): the folded model is the supported
    :class:`~actfold.models.architecture_utils.ManualFoldedForward` path,
    there are at least two candidates, they share one non-None parent, and
    they are equal-length with parent-batch 1 (the generate-loop shape:
    candidates derive from ``parent.tokens`` by appending one token, so the
    candidate batch equals the parent batch).
    """
    if folded_model is None or not isinstance(folded_model, ManualFoldedForward):
        return False
    if len(candidates) <= 1:
        return False
    if len({node.parent_id for node in candidates}) != 1:
        return False
    if candidates[0].parent_id is None:
        return False
    first_shape = tuple(candidates[0].tokens.shape)
    if any(tuple(node.tokens.shape) != first_shape for node in candidates):
        return False
    return first_shape[0] == 1


def _run_folded_forward_batched(
    model: FastDLLMAdapter,
    nodes: list[BranchNode],
    folded_model: ManualFoldedForward,
    step_idx: int,
) -> None:
    """Execute one batched folded forward for all ``nodes`` (AR015).

    Mirrors :func:`_run_folded_forward` per node: profiles are reset per
    branch, row ``i`` of the stacked batch belongs to ``nodes[i]``, and
    logits are stored as read-only ``[1, seq_len, vocab]`` row views.
    """
    del model  # The batched path always routes through the folded model.
    parent_id = nodes[0].parent_id

    for node in nodes:
        # Reset any stale profile so the profiler only records the current pass.
        GLOBAL_STABILITY_PROFILER.reset_branch(node.branch_id)

    stacked = torch.cat([node.tokens for node in nodes], dim=0)
    logits = folded_model(
        stacked,
        branch_id=None,
        branch_ids=[node.branch_id for node in nodes],
        parent_branch_id=parent_id,
        step_idx=step_idx,
    )

    for row_idx, node in enumerate(nodes):
        node.logits = logits[row_idx : row_idx + 1]

        profile = GLOBAL_STABILITY_PROFILER.get_profile(node.branch_id)
        if profile is not None:
            node.metadata["stable_ratio"] = profile.mean_stable_ratio
