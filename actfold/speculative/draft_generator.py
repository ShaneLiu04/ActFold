"""Candidate branch generation for speculative decoding."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

import torch

from actfold.speculative.branch import Branch


class DraftGenerator:
    """Generate multiple candidate child branches from a parent branch.

    Supported modes:

    - ``suffix_append`` (default): clone the parent and resample a small
      fraction (``flip_ratio``) of positions inside the *suffix region*
      (``flip_region`` or everything after ``prompt_length``). The prompt
      prefix is never touched, so drafts stay close to the parent and the
      stable-ratio sweep over ``tau`` is graded and distinguishable (B10).
    - ``logits_draft``: like ``suffix_append`` but resampled positions are
      drawn from the top-k tokens of ``parent_logits`` at that position,
      approximating the model's own next-token distribution.
    - ``model_draft`` (AR006): like ``logits_draft`` but the logits come from
      an independent draft model (``draft_model(parent.tokens)``), not from a
      target forward pass — the real speculative-decoding semantics. The
      ``parent_logits`` argument is ignored in this mode (upper-bound
      comparison experiments may still carry it). Append positions are
      sampled from the draft logits at the parent's last position (first-tier
      approximation of autoregressive drafting).
    - ``copy_flip``: copy the parent and flip a fraction of positions anywhere
      (uniform over the full vocabulary). Kept for backward compatibility.
    - ``random`` / ``perturb``: uniform random tokens (research stand-ins).

    Args:
        vocab_size: Size of the token vocabulary.
        mode: Drafting mode ("suffix_append", "logits_draft", "model_draft",
            "random", "perturb", "copy_flip").
        perturbation_std: Deprecated; kept for API compatibility.
        flip_ratio: Fraction of region positions to resample.
        top_k: Candidate pool size for "logits_draft"/"model_draft".
        prompt_length: Protected prefix length used for the default suffix
            region (positions before it are never resampled).
        draft_model: Callable mapping tokens ``[batch, seq_len]`` to logits
            ``[batch, seq_len, vocab_size]`` (an ``nn.Module`` qualifies).
            Required at ``generate`` time for "model_draft"; ignored by every
            other mode. Draft models should run in eval mode so seeded
            generation is reproducible.
    """

    _MODES: tuple[str, ...] = (
        "suffix_append",
        "logits_draft",
        "model_draft",
        "random",
        "perturb",
        "copy_flip",
    )

    def __init__(
        self,
        vocab_size: int,
        mode: str = "suffix_append",
        perturbation_std: float = 0.1,
        flip_ratio: float = 0.1,
        top_k: int = 10,
        prompt_length: int | None = None,
        draft_model: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if mode not in self._MODES:
            raise ValueError(f"Unsupported draft mode: {mode}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        self.vocab_size = vocab_size
        self.mode = mode
        self.perturbation_std = perturbation_std
        self.flip_ratio = flip_ratio
        self.top_k = top_k
        self.prompt_length = prompt_length
        self.draft_model = draft_model
        self._counter = 0
        # Session-scoped prefix keeps child IDs at bounded length; chaining
        # parent IDs (the old format) grew O(T^2) in total string memory.
        self._session = uuid4().hex[:8]

    def _region(
        self,
        seq_len: int,
        flip_region: tuple[int, int] | None,
    ) -> tuple[int, int]:
        """Resolve the resampling region, validating an explicit override."""
        if flip_region is not None:
            start, end = flip_region
            if start < 0 or end > seq_len or start >= end:
                raise ValueError(
                    f"flip_region {flip_region} is invalid for seq_len={seq_len} "
                    "(require 0 <= start < end <= seq_len)"
                )
            return start, end
        return (self.prompt_length or 0), seq_len

    def _sample_from_topk(
        self,
        logits_pos: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> int:
        """Sample one token uniformly among the top-k of ``logits_pos``."""
        k = min(self.top_k, logits_pos.numel())
        _, top_idx = torch.topk(logits_pos, k)
        choice = torch.randint(0, k, (1,), generator=generator, device=logits_pos.device)
        return int(top_idx[choice])

    def _sample_tokens_from_logits(
        self,
        logits_source: torch.Tensor,
        batch_idx: int,
        positions: list[int],
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Sample one top-k token per position from ``logits_source``.

        Shared by the ``logits_draft`` and ``model_draft`` modes so the two
        sampling paths cannot drift apart.
        """
        return torch.tensor(
            [
                self._sample_from_topk(
                    logits_source[batch_idx, pos], generator=generator
                )
                for pos in positions
            ],
            dtype=dtype,
            device=device,
        )

    def generate(
        self,
        parent: Branch,
        num_branches: int = 2,
        seed: int | None = None,
        max_new_tokens: int = 0,
        parent_logits: torch.Tensor | None = None,
        flip_region: tuple[int, int] | None = None,
    ) -> list[Branch]:
        """Generate ``num_branches`` child branches.

        Args:
            parent: Parent branch.
            num_branches: Number of children to generate.
            seed: Optional random seed.
            max_new_tokens: Number of new tokens to append to each child. When
                zero (the default), children have the same sequence length as
                the parent.
            parent_logits: Required for "logits_draft": logits tensor of shape
                ``[batch, seq_len, vocab_size]`` from a parent forward pass.
            flip_region: Optional ``(start, end)`` override of the resampling
                region. Defaults to the suffix after ``prompt_length``.

        Returns:
            List of child Branch instances.

        Raises:
            ValueError: If ``max_new_tokens`` is negative, ``flip_region`` is
                invalid, or ``parent_logits`` has the wrong shape.
            RuntimeError: If "logits_draft" is used without ``parent_logits``.
        """
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

        # Seeded sampling runs on local torch.Generator objects so the global
        # RNG state is never polluted (AR002 srs 3.6). Without a seed, sampling
        # uses the global RNG exactly as before.  Two generators are seeded:
        # CPU ops (``randperm`` without an explicit device) read ``cpu_rng``
        # while device ops read ``dev_rng``; on CPU they are the same object.
        if seed is not None:
            cpu_rng = torch.Generator(device="cpu")
            cpu_rng.manual_seed(seed)
            if parent.tokens.device.type == "cpu":
                dev_rng = cpu_rng
            else:
                dev_rng = torch.Generator(device=parent.tokens.device.type)
                dev_rng.manual_seed(seed)
            # Reset the deterministic counter so repeated calls with the same
            # seed produce the same branch IDs.
            self._counter = 0
        else:
            cpu_rng = None
            dev_rng = None

        batch_size, seq_len = parent.tokens.shape
        region_start, region_end = self._region(seq_len, flip_region)
        region_len = region_end - region_start

        logits_source: torch.Tensor | None = None
        if self.mode == "logits_draft":
            if parent_logits is None:
                raise RuntimeError(
                    "DraftGenerator mode 'logits_draft' requires parent_logits "
                    "from a parent forward pass; got None."
                )
            if tuple(parent_logits.shape) != (batch_size, seq_len, self.vocab_size):
                raise ValueError(
                    f"parent_logits shape {tuple(parent_logits.shape)} does not "
                    f"match expected {(batch_size, seq_len, self.vocab_size)}"
                )
            logits_source = parent_logits
        elif self.mode == "model_draft":
            if self.draft_model is None:
                raise RuntimeError(
                    "DraftGenerator mode 'model_draft' requires a draft_model "
                    "(callable tokens -> logits); got None."
                )
            draft_logits = self.draft_model(parent.tokens)
            if tuple(draft_logits.shape) != (
                batch_size,
                seq_len,
                self.vocab_size,
            ):
                raise ValueError(
                    f"draft_model output shape {tuple(draft_logits.shape)} does "
                    f"not match expected "
                    f"{(batch_size, seq_len, self.vocab_size)}"
                )
            if not torch.is_floating_point(draft_logits):
                raise ValueError(
                    "draft_model output must be floating-point logits, got "
                    f"dtype {draft_logits.dtype}"
                )
            if draft_logits.device != parent.tokens.device:
                raise ValueError(
                    "draft_model output device "
                    f"{draft_logits.device} does not match parent tokens device "
                    f"{parent.tokens.device}"
                )
            logits_source = draft_logits

        children: list[Branch] = []

        for _ in range(num_branches):
            child_id = f"{self._session}:{self._counter}"
            self._counter += 1

            child_tokens: torch.Tensor
            if self.mode in ("random", "perturb"):
                # Research stand-in: sample random tokens; real perturbation in
                # embedding space requires access to the model's embeddings.
                child_tokens = torch.randint(
                    low=0,
                    high=self.vocab_size,
                    size=(batch_size, seq_len + max_new_tokens),
                    dtype=parent.tokens.dtype,
                    device=parent.tokens.device,
                    generator=dev_rng,
                )
            elif self.mode == "copy_flip":
                child_tokens = parent.tokens.clone()
                if self.flip_ratio > 0 and seq_len > 0:
                    num_flips = max(1, int(seq_len * self.flip_ratio))
                    for b in range(batch_size):
                        flip_positions = torch.randperm(seq_len, generator=cpu_rng)[:num_flips]
                        new_tokens = torch.randint(
                            0,
                            self.vocab_size,
                            (num_flips,),
                            dtype=child_tokens.dtype,
                            device=child_tokens.device,
                            generator=dev_rng,
                        )
                        child_tokens[b, flip_positions] = new_tokens
                if max_new_tokens > 0:
                    appended = torch.randint(
                        0,
                        self.vocab_size,
                        (batch_size, max_new_tokens),
                        dtype=child_tokens.dtype,
                        device=child_tokens.device,
                        generator=dev_rng,
                    )
                    child_tokens = torch.cat([child_tokens, appended], dim=1)
            else:  # suffix_append / logits_draft / model_draft
                child_tokens = parent.tokens.clone()
                if self.flip_ratio > 0 and region_len > 0:
                    num_flips = max(1, int(region_len * self.flip_ratio))
                    for b in range(batch_size):
                        flip_offsets = torch.randperm(region_len, generator=cpu_rng)[:num_flips]
                        flip_positions = flip_offsets + region_start
                        if self.mode == "suffix_append":
                            new_tokens = torch.randint(
                                0,
                                self.vocab_size,
                                (num_flips,),
                                dtype=child_tokens.dtype,
                                device=child_tokens.device,
                                generator=dev_rng,
                            )
                        elif logits_source is not None:  # logits_draft / model_draft
                            new_tokens = self._sample_tokens_from_logits(
                                logits_source,
                                b,
                                flip_positions.tolist(),
                                child_tokens.dtype,
                                child_tokens.device,
                                dev_rng,
                            )
                        else:
                            raise RuntimeError(
                                "DraftGenerator mode 'logits_draft' requires parent_logits "
                                "from a parent forward pass; got None."
                            )
                        child_tokens[b, flip_positions] = new_tokens
                if max_new_tokens > 0:
                    if self.mode == "suffix_append":
                        appended = torch.randint(
                            0,
                            self.vocab_size,
                            (batch_size, max_new_tokens),
                            dtype=child_tokens.dtype,
                            device=child_tokens.device,
                            generator=dev_rng,
                        )
                    elif logits_source is not None:
                        # logits_draft / model_draft: sample from the last
                        # position's top-k
                        appended = torch.stack(
                            [
                                self._sample_tokens_from_logits(
                                    logits_source,
                                    b,
                                    [seq_len - 1] * max_new_tokens,
                                    child_tokens.dtype,
                                    child_tokens.device,
                                    dev_rng,
                                )
                                for b in range(batch_size)
                            ],
                            dim=0,
                        )
                    else:
                        raise RuntimeError(
                            "DraftGenerator mode 'logits_draft' requires parent_logits "
                            "from a parent forward pass; got None."
                        )
                    child_tokens = torch.cat([child_tokens, appended], dim=1)

            child = Branch(
                branch_id=child_id,
                parent_id=parent.branch_id,
                tokens=child_tokens,
            )
            children.append(child)

        return children
