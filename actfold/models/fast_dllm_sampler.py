"""Fast-dLLM discrete diffusion sampler with Branch Folding support.

This sampler follows the Fast-dLLM v2 recipe:

- Generation is performed block-by-block.  Each block is initialized to the
  ``<mask>`` token.
- Within a block, the model predicts all positions.  Positions are unmasked
  in small sub-blocks when their predicted probability exceeds a confidence
  ``threshold``.  The highest-probability position in each small block is
  always unmasked.
- When a block is fully unmasked, an autoregressive step generates the next
  block's first token using KV-cache friendly block processing.
- Support top-p / temperature sampling and early stopping on a stop token.

AR021 production generation-loop folding:

- FR1: ``_fold_ctx_for`` pins a per-``(T, prompt_len)`` fold context (the
  block-causal AND non-pad ``[T, T]`` attention mask built from host pad
  metadata, the model's own rotary ``position_embeddings`` or ``None``, and
  the ``forced`` base tensor covering exactly the active block) — zero
  device readback, object-pinned across denoise iterations and ``sample``
  calls.
- FR2: every folded forward (denoise iterations and the block-advance AR
  step) threads the pinned mask / rotary / forced trio through
  ``DiffusionSampler._forward``'s new tail parameters.
- FR3: ``num_block_candidates > 0`` switches each small block to batched
  speculative verification — one draft serial forward, one ``[N, T]``
  batched-branch call over top-k candidate fills (per-candidate seeds from
  the frozen formula), min-prob argmax selection, chain continuation onto
  the selected candidate branch, and baseline-identical stop-token
  semantics. Default-off is byte-identical to the existing recipe.

Reference:
- https://github.com/NVlabs/Fast-dLLM/blob/main/v2/generation_functions.py
"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from actfold.core.model_wrapper import FoldedModel
from actfold.models.base import DiffusionLLM
from actfold.models.diffusion_sampler import DiffusionSampler, SamplerConfig, SamplerOutput
from actfold.models.sampling_utils import right_shift_logits, sample_tokens

_MAX_FOLD_CTX_ENTRIES = 32


@dataclass
class FastDLLMSamplerConfig(SamplerConfig):
    """Configuration for the Fast-dLLM discrete diffusion sampler.

    Args:
        num_block_candidates: Number of speculative candidate continuations
            per small block (AR021 FR3). ``0`` (the default) keeps the exact
            baseline remasking recipe; ``N > 0`` replaces the per-iteration
            confidence remasking with one draft forward plus one ``[N, T]``
            batched-branch verification call per small block (requires a
            ``folded_model`` that supports ``branch_ids``).
        candidate_top_k: Draft-set size per position for the speculative
            candidates (top-k tokens of the draft logits). Consumed only
            when ``num_block_candidates > 0``.
    """

    block_size: int = 32
    small_block_size: int = 32
    threshold: float = 0.95
    stop_token_id: int | None = None
    pad_token_id: int | None = None
    mask_token_id: int | None = None
    num_block_candidates: int = 0
    candidate_top_k: int = 10

    def __post_init__(self) -> None:
        if self.num_block_candidates < 0:
            raise ValueError(
                "num_block_candidates must be >= 0, got " f"{self.num_block_candidates}"
            )
        if self.candidate_top_k < 1:
            raise ValueError(f"candidate_top_k must be >= 1, got {self.candidate_top_k}")


@dataclass
class _FoldCtx:
    """Pinned per-``(T, prompt_len)`` fold context (AR021 FR1).

    Attributes:
        mask: Block-causal AND non-pad ``[T, T]`` boolean attention mask
            built from host pad metadata (never a device readback).
        pe: Rotary ``(cos, sin)`` tuple from the model's own
            ``rotary_emb`` under the constant-position scheme, or ``None``
            when the model has no detectable rotary module.
        forced: ``[T]`` boolean tensor whose True positions cover exactly
            the active (last) block — the freshly committed / actively
            denoised region is always recomputed, never gate-certified
            stable against a stale parent.
    """

    mask: torch.Tensor
    pe: Any | None
    forced: torch.Tensor


class FastDLLMSampler(DiffusionSampler):
    """Discrete diffusion sampler for Fast-dLLM-style models.

    Args:
        model: Diffusion model wrapper.
        config: Sampler configuration. If omitted, sensible defaults are used.
    """

    config: FastDLLMSamplerConfig

    def __init__(
        self,
        model: DiffusionLLM,
        config: FastDLLMSamplerConfig | None = None,
    ) -> None:
        super().__init__(model, config or FastDLLMSamplerConfig())
        if self.config.block_size < 1 or self.config.small_block_size < 1:
            raise ValueError("block_size and small_block_size must be >= 1")
        if self.config.block_size % self.config.small_block_size != 0:
            raise ValueError("block_size must be divisible by small_block_size")
        self._fold_ctx_cache: "OrderedDict[tuple[int, int], _FoldCtx]" = OrderedDict()

    def initialize(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Return prompt_ids; block-wise state is built inside :meth:`sample`."""
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("FastDLLMSampler requires a tokenizer to obtain special token ids.")

        cfg = self.config
        mask_token_id = cfg.mask_token_id
        stop_token_id = cfg.stop_token_id
        pad_token_id = cfg.pad_token_id

        if mask_token_id is None:
            mask_token_id = self._resolve_mask_token_id(self.model, tokenizer)
        if stop_token_id is None:
            stop_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None and stop_token_id is not None:
                pad_token_id = stop_token_id

        if mask_token_id is None:
            raise RuntimeError(
                "FastDLLMSampler requires a mask_token_id. Set it in the config or tokenizer."
            )
        if stop_token_id is None:
            raise RuntimeError(
                "FastDLLMSampler requires a stop_token_id. Set it in the config or tokenizer."
            )

        self._mask_token_id = int(mask_token_id)
        self._stop_token_id = int(stop_token_id)
        self._pad_token_id = int(pad_token_id) if pad_token_id is not None else int(stop_token_id)
        return prompt_ids.clone()

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        branch_id: Any,
        parent_branch_id: Any | None,
        folded_model: FoldedModel | None,
    ) -> torch.Tensor:
        """Not used directly; generation is block-wise inside :meth:`sample`."""
        return x_t

    def _fold_ctx_for(self, x: torch.Tensor, prompt_len: int) -> _FoldCtx:
        """Return the pinned fold context for ``(x.shape[1], prompt_len)``.

        The context (FR1 mask / rotary / forced trio) is constructed once per
        ``(T, prompt_len)`` window from HOST metadata only — the pad pattern
        comes from ``prompt_len`` and the block arithmetic, never from a
        device readback (``.item`` / ``.tolist`` / ``.cpu`` are forbidden on
        this path) — and is object-pinned: the same key returns the SAME
        mask / pe / forced objects across denoise iterations, canvas growth,
        and ``sample`` calls. Same ``T`` with a different ``prompt_len`` is a
        different key with different pad content (never shared). The cache is
        an instance-level LRU capped at ``_MAX_FOLD_CTX_ENTRIES``.
        """
        seq_len = int(x.shape[1])
        key = (seq_len, int(prompt_len))
        cache: "OrderedDict[tuple[int, int], _FoldCtx] | None" = getattr(
            self, "_fold_ctx_cache", None
        )
        if cache is None:
            cache = OrderedDict()
            self._fold_ctx_cache = cache
        ctx = cache.get(key)
        if ctx is not None:
            cache.move_to_end(key)
            return ctx

        block_size = self.config.block_size
        device = x.device
        q_idx = torch.arange(seq_len, device=device) // block_size
        mask = q_idx[:, None] >= q_idx[None, :]
        first_block_padding = (block_size - int(prompt_len) % block_size) % block_size
        if first_block_padding > 0:
            mask = mask.clone()
            mask[:, int(prompt_len) : int(prompt_len) + first_block_padding] = False
        pe = self._resolve_position_embeddings(seq_len, device)
        forced = torch.zeros(seq_len, dtype=torch.bool, device=device)
        forced[max(0, seq_len - block_size) :] = True
        ctx = _FoldCtx(mask=mask, pe=pe, forced=forced)
        cache[key] = ctx
        cache.move_to_end(key)
        while len(cache) > _MAX_FOLD_CTX_ENTRIES:
            cache.popitem(last=False)
        return ctx

    def _resolve_position_embeddings(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Build the rotary ``(cos, sin)`` for a constant-position window.

        Drills the model wrapper for a Hugging Face-style ``rotary_emb``
        (``model.model.rotary_emb`` then ``model.rotary_emb`` where
        ``model`` is the raw backbone). The hidden size for the dummy input
        resolves through ``config.hidden_size``, then the embedding table's
        ``embedding_dim``, then ``rotary.dim``; an undiscoverable rotary
        geometry is an explicit ``RuntimeError`` (never a silent wrong-shape
        call). Models without a rotary module return ``None`` (the folded
        path then attaches no ``position_embeddings`` kwarg at all).

        The dummy carries the wrapped model's activation dtype (resolved
        from the embedding weight, the same host chain as the hidden-size
        fallback): remote rotary modules return ``cos.to(dtype=dummy.dtype)
        `` and the raw forward feeds them real bf16 hidden states, so an
        fp32 dummy on a bf16 checkpoint yields fp32 cos/sin whose rotary
        application promotes q/k against a bf16 v and breaks the attention
        dtype contract (RuntimeError on the real Fast-dLLM checkpoint; the
        fp32 synthetic mini models never see it — AR021 T003 smoke).
        """
        raw = getattr(self.model, "model", None)
        rotary: Any = None
        if raw is not None:
            inner = getattr(raw, "model", None)
            if inner is not None:
                rotary = getattr(inner, "rotary_emb", None)
            if rotary is None:
                rotary = getattr(raw, "rotary_emb", None)
        if rotary is None:
            return None
        hidden: Any = None
        config = getattr(raw, "config", None)
        if config is not None:
            hidden = getattr(config, "hidden_size", None)
        if hidden is None:
            for host in (raw, getattr(raw, "model", None)):
                embed = getattr(host, "embed_tokens", None)
                if embed is not None:
                    hidden = getattr(embed, "embedding_dim", None)
                    if hidden is not None:
                        break
        if hidden is None:
            hidden = getattr(rotary, "dim", None)
        if hidden is None:
            raise RuntimeError(
                "FastDLLMSampler found a rotary module but could not resolve "
                "the hidden size for the position-embedding supply."
            )
        dtype = torch.float32
        for host in (raw, getattr(raw, "model", None)):
            embed = getattr(host, "embed_tokens", None)
            weight = getattr(embed, "weight", None) if embed is not None else None
            if weight is not None:
                dtype = weight.dtype
                break
        with torch.no_grad():
            dummy = torch.zeros(1, seq_len, int(hidden), device=device, dtype=dtype)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pe = rotary(dummy, position_ids)
        cos, sin = pe
        return cos, sin

    @staticmethod
    def _supports_branch_ids(folded_model: Any) -> bool:
        """Probe whether ``folded_model`` accepts the ``branch_ids`` kwarg.

        Non-callables and signatures that would reject ``branch_ids`` (an
        explicit parameter list without it and without a ``**kwargs``
        catch-all) fail the probe — the caller raises ``ValueError`` instead
        of letting the batched call degrade silently. A ``**kwargs``
        passthrough (e.g. a recording subclass of
        ``ManualFoldedForward``) is accepted: the kwarg is syntactically
        forwardable and the real folded forward enforces its own contract.
        """
        target = folded_model
        if isinstance(target, torch.nn.Module):
            target = getattr(target, "forward", None)
            if target is None:
                return False
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return False
        params = signature.parameters
        if "branch_ids" in params:
            return True
        return any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())

    def _speculative_small_block(
        self,
        x: torch.Tensor,
        finished: torch.Tensor,
        small_logits: torch.Tensor,
        small_mask_row: torch.Tensor,
        sb_start: int,
        sb_end: int,
        draft_branch_id: str,
        ctx: _FoldCtx,
        folded_model: Any,
        block_idx: int,
        small_block_idx: int,
        history: list[torch.Tensor],
    ) -> None:
        """Run one FR3 speculative small block (mutates ``x``/``finished``).

        Builds ``N`` candidate continuations of the current canvas by
        sampling each still-masked small-block position from the draft's
        top-k token set (per-candidate frozen seed formula
        ``(seed*1_000_003 + block*8_191 + small*131 + j) & 0x7FFF_FFFF`` on
        a dedicated CPU generator — deterministic across runs and unaffected
        by the global RNG), verifies all candidates in ONE ``[N, T]``
        batched-branch call sharing the draft parent, and commits the row
        whose per-position minimum token probability (softmax over the
        right-shifted batched logits, gathered at the candidate tokens) is
        maximal. The folding chain parent switches to the selected candidate
        branch; the stop-token check matches the baseline's
        ``finished`` semantics.
        """
        cfg = self.config
        num_candidates = int(cfg.num_block_candidates)
        vocab = int(small_logits.shape[-1])
        k_eff = max(1, min(int(cfg.candidate_top_k), vocab))
        topk_idx = torch.topk(small_logits, k_eff, dim=-1).indices  # [1, S, k]
        masked_local = small_mask_row.nonzero(as_tuple=False).reshape(-1)
        num_masked = int(masked_local.shape[0])
        cands = x.repeat(num_candidates, 1)
        if num_masked > 0:
            draft_set = topk_idx[0, masked_local, :]  # [M, k]
            base_seed = int(cfg.seed) if cfg.seed is not None else 0
            for j in range(num_candidates):
                seed_j = (
                    base_seed * 1_000_003 + int(block_idx) * 8_191 + int(small_block_idx) * 131 + j
                ) & 0x7FFF_FFFF
                generator = torch.Generator(device="cpu").manual_seed(seed_j)
                choice = torch.randint(0, k_eff, (num_masked,), generator=generator).to(x.device)
                cand_tokens = torch.gather(draft_set, 1, choice.unsqueeze(1)).squeeze(1)
                cands[j, masked_local + sb_start] = cand_tokens
        cand_branch_ids = [f"{draft_branch_id}_cand{j}" for j in range(num_candidates)]
        batched_logits = self._forward(
            cands,
            branch_id=None,
            parent_branch_id=draft_branch_id,
            folded_model=folded_model,
            step_idx=block_idx,
            attention_mask=ctx.mask,
            position_embeddings=ctx.pe,
            branch_ids=cand_branch_ids,
            forced_divergent=ctx.forced.unsqueeze(0).repeat(num_candidates, 1),
        )
        shifted = right_shift_logits(batched_logits)
        commit_logits = shifted[:, sb_start:sb_end, :]
        probs = F.softmax(commit_logits, dim=-1)
        committed = cands[:, sb_start:sb_end]
        scores = torch.gather(probs, -1, committed.unsqueeze(-1)).squeeze(-1)
        min_scores = scores.min(dim=-1).values
        j_star = int(min_scores.argmax())
        x[:, sb_start:sb_end] = cands[j_star, sb_start:sb_end].unsqueeze(0)
        finished |= (x[:, sb_start:sb_end] == self._stop_token_id).any(dim=1)
        self._folding_parent = cand_branch_ids[j_star]
        if self.config.return_history:
            history.append(x.clone())

    def sample(
        self,
        prompt_ids: torch.Tensor,
        folded_model: FoldedModel | None = None,
    ) -> SamplerOutput:
        """Run Fast-dLLM block-wise masked diffusion sampling.

        Prompt padding positions (``pad_id``) are excluded from attention via
        an explicit ``attention_mask`` recomputed before every raw forward
        pass, so padded tokens never influence the logits of real tokens.
        The folded path (AR021) instead threads the pinned fold-context trio
        (block-causal ``[T, T]`` mask, rotary ``position_embeddings``,
        ``forced_divergent`` over the active block) through every folded
        forward; the raw path is byte-for-byte unchanged.

        Raises:
            ValueError: If ``folded_model`` is given with a batch larger than
                1 (the ``[T, T]`` mask cannot express per-row padding); or
                if ``num_block_candidates > 0`` without a ``folded_model``
                that supports ``branch_ids`` (never a silent degradation).
        """
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        num_candidates = int(self.config.num_block_candidates)
        if folded_model is not None and prompt_ids.shape[0] > 1:
            raise ValueError(
                "FastDLLMSampler folded sampling supports batch 1 only; the "
                f"[T, T] fold-context mask cannot express per-row padding "
                f"(got batch {prompt_ids.shape[0]})."
            )
        if num_candidates > 0:
            if folded_model is None:
                raise ValueError(
                    "num_block_candidates > 0 requires a folded_model for the "
                    "batched speculative verification path."
                )
            if not self._supports_branch_ids(folded_model):
                raise ValueError(
                    "num_block_candidates > 0 requires a folded_model that "
                    "supports the branch_ids batched-branch kwarg."
                )

        self._reset_folding_chain()
        x = self.initialize(prompt_ids)
        B = x.shape[0]
        device = x.device
        block_size = self.config.block_size
        small_block_size = self.config.small_block_size
        num_small_blocks = block_size // small_block_size
        max_new_tokens = self.config.num_tokens
        mask_id = self._mask_token_id
        stop_id = self._stop_token_id
        pad_id = self._pad_token_id

        prompt_len = x.shape[1]
        total_len = prompt_len + max_new_tokens
        num_blocks = max(1, (max_new_tokens + block_size - 1) // block_size)

        # Pad prompt to a multiple of block_size so that subsequent blocks are aligned.
        first_block_padding = (
            (block_size - prompt_len % block_size) if prompt_len % block_size != 0 else 0
        )
        if first_block_padding > 0:
            pad = torch.full((B, first_block_padding), pad_id, dtype=torch.long, device=device)
            x = torch.cat([x, pad], dim=1)
        else:
            first_block_padding = 0

        # Append the first block of masks.
        x = torch.cat(
            [x, torch.full((B, block_size), mask_id, dtype=torch.long, device=device)],
            dim=1,
        )

        history: list[torch.Tensor] = [x.clone()] if self.config.return_history else []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for block_idx in range(num_blocks):
            if finished.all():
                break

            # Denoise the last block.
            for small_block_idx in range(num_small_blocks):
                start = -block_size + small_block_idx * small_block_size
                end = (
                    None
                    if small_block_idx == num_small_blocks - 1
                    else -block_size + (small_block_idx + 1) * small_block_size
                )

                # Keep denoising this small block until all positions are unmasked.
                max_iters = small_block_size * 2
                for _ in range(max_iters):
                    block_view = x[:, -block_size:]
                    mask_idx = block_view == mask_id
                    if mask_idx[:, start:end].sum() == 0:
                        break

                    branch_id, parent_id = self._next_folding_branch(
                        f"fastdllm_b{block_idx}_s{small_block_idx}"
                    )
                    if folded_model is not None:
                        # AR021 FR1/FR2: the pinned fold context supplies the
                        # block-causal [T, T] mask, the rotary pair, and the
                        # forced region over the active block.
                        ctx = self._fold_ctx_for(x, prompt_len=prompt_len)
                        logits = self._forward(
                            x,
                            branch_id=branch_id,
                            parent_branch_id=parent_id,
                            folded_model=folded_model,
                            step_idx=block_idx,
                            attention_mask=ctx.mask,
                            position_embeddings=ctx.pe,
                            forced_divergent=ctx.forced,
                        )
                    else:
                        # Pad tokens are placeholders, not content: exclude
                        # them from attention so they never influence real
                        # logits (B5). Recomputed per call because ``x``
                        # grows between blocks.
                        attention_mask = x != pad_id
                        logits = self._forward(
                            x,
                            branch_id=branch_id,
                            parent_branch_id=parent_id,
                            folded_model=folded_model,
                            step_idx=block_idx,
                            attention_mask=attention_mask,
                        )
                    logits = right_shift_logits(logits)
                    logits = logits[:, -block_size:]
                    logits = logits[:, start:end]

                    if num_candidates > 0:
                        # AR021 FR3: one draft serial forward (this call) plus
                        # one [N, T] batched verification replaces the whole
                        # remasking iteration sequence for this small block.
                        seq_len_now = int(x.shape[1])
                        sb_start = seq_len_now + start
                        sb_end = seq_len_now + end if end is not None else seq_len_now
                        self._speculative_small_block(
                            x,
                            finished,
                            logits,
                            mask_idx[0, start:end],
                            sb_start,
                            sb_end,
                            branch_id,
                            ctx,
                            folded_model,
                            block_idx,
                            small_block_idx,
                            history,
                        )
                        break

                    confidence, x1 = sample_tokens(
                        logits,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                    )
                    probs = F.softmax(logits, dim=-1)
                    x1_p = torch.gather(probs, dim=-1, index=x1.unsqueeze(-1)).squeeze(-1)

                    small_mask = mask_idx[:, start:end]
                    x1_p = torch.where(small_mask, x1_p, torch.full_like(x1_p, float("-inf")))

                    unmask = x1_p > self.config.threshold
                    max_idx = x1_p.argmax(dim=-1)
                    unmask[torch.arange(B), max_idx] = True
                    unmask = unmask & small_mask

                    x[:, start:end][unmask] = x1[unmask]

                    finished = finished | ((x1 == stop_id) & unmask).any(dim=1)

                    if self.config.return_history:
                        history.append(x.clone())

            # After the block is decoded, generate the next token autoregressively
            # and append a fresh mask block if more tokens remain.
            remaining = total_len + first_block_padding - x.shape[1]
            if remaining <= 0:
                continue

            branch_id, parent_id = self._next_folding_branch(f"fastdllm_ar_{block_idx}")
            if folded_model is not None:
                # AR021 FR2/UT-2111: the block-advance AR forward reuses the
                # same pinned fold context — its forced region covers exactly
                # the just-committed block (the P5 partial-flip scenario).
                ctx = self._fold_ctx_for(x, prompt_len=prompt_len)
                logits = self._forward(
                    x,
                    branch_id=branch_id,
                    parent_branch_id=parent_id,
                    folded_model=folded_model,
                    step_idx=block_idx,
                    attention_mask=ctx.mask,
                    position_embeddings=ctx.pe,
                    forced_divergent=ctx.forced,
                )
            else:
                attention_mask = x != pad_id
                logits = self._forward(
                    x,
                    branch_id=branch_id,
                    parent_branch_id=parent_id,
                    folded_model=folded_model,
                    step_idx=block_idx,
                    attention_mask=attention_mask,
                )
            logits = right_shift_logits(logits)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            next_token[finished] = pad_id
            x = torch.cat([x, next_token], dim=1)

            if remaining > 1:
                fresh_masks = torch.full(
                    (B, min(block_size - 1, remaining - 1)),
                    mask_id,
                    dtype=torch.long,
                    device=device,
                )
                x = torch.cat([x, fresh_masks], dim=1)

            if self.config.return_history:
                history.append(x.clone())

        # Trim padding and stop tokens for finished sequences.
        decoded = self.decode_final(x)
        return SamplerOutput(sequences=decoded, history=history)

    def decode_final(self, x: torch.Tensor) -> torch.Tensor:
        """Trim leading pad/mask positions and trailing stop tokens.

        Fast-dLLM uses left-padded canvas internally.  This decoder removes the
        initial padding and any trailing stop-token suffix.
        """
        B, T = x.shape
        pad_id = self._pad_token_id
        stop_id = self._stop_token_id
        mask_id = self._mask_token_id

        # Find the first non-pad/non-mask position in each row.
        valid = (x != pad_id) & (x != mask_id)
        out_rows: list[torch.Tensor] = []
        for i in range(B):
            pos = valid[i].nonzero(as_tuple=False)
            if pos.numel() == 0:
                out_rows.append(torch.tensor([stop_id], dtype=torch.long, device=x.device))
                continue
            start = int(pos[0].item())
            row = x[i, start:].clone()
            # Trim trailing stop tokens (but keep one).
            stop_positions = (row == stop_id).nonzero(as_tuple=False)
            if stop_positions.numel() > 0:
                end = int(stop_positions[0].item()) + 1
                row = row[:end]
            out_rows.append(row)

        max_len = max(r.shape[0] for r in out_rows)
        out = torch.full((B, max_len), pad_id, dtype=torch.long, device=x.device)
        for i, r in enumerate(out_rows):
            out[i, : r.shape[0]] = r
        return out
