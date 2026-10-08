"""Tests for diffusion-native samplers."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from actfold.core.model_wrapper import FoldedModel
from actfold.models.base import DiffusionLLM
from actfold.models.diffusion_sampler import SamplerOutput
from actfold.models.dream_sampler import DreamSampler, DreamSamplerConfig
from actfold.models.fast_dllm_sampler import FastDLLMSampler, FastDLLMSamplerConfig
from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig
from actfold.models.sampling_utils import (
    add_gumbel_noise,
    compute_position_ids,
    get_num_transfer_tokens,
    right_shift_logits,
    sample_tokens,
)


class DummyTokenizer:
    """Minimal tokenizer stand-in for sampler tests."""

    def __init__(self, vocab_size: int = 16) -> None:
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 2
        self.bos_token_id = vocab_size - 3
        self.pad_token_id = self.eos_token_id


class DummyDiffusionModel(DiffusionLLM):
    """Minimal DiffusionLLM for sampler tests."""

    def __init__(
        self,
        vocab_size: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_logits: bool = True,
    ) -> None:
        super().__init__("dummy")
        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.tokenizer = DummyTokenizer(vocab_size)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.output_logits = output_logits
        if output_logits:
            self.head = nn.Linear(hidden_dim, vocab_size)
        else:
            self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del attention_mask, kwargs
        if tokens.dtype in (torch.long, torch.int):
            x = self.embedding(tokens)
        else:
            x = tokens
        return self.head(x)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_heads(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


def test_llada_sampler_reduces_masks() -> None:
    """LLaDASampler progressively unmasks generated positions."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    config = LLaDASamplerConfig(num_steps=4, num_tokens=4)
    sampler = LLaDASampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert isinstance(output, SamplerOutput)
    assert output.sequences.shape == (1, 7)
    # Some generated positions should be unmasked (not equal to mask id 15).
    assert (output.sequences == model.tokenizer.mask_token_id).sum().item() < 4


def test_fast_dllm_sampler_changes_tokens() -> None:
    """FastDLLMSampler produces an output tensor of expected shape."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    config = FastDLLMSamplerConfig(num_steps=4, num_tokens=4, block_size=4, small_block_size=4)
    sampler = FastDLLMSampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert isinstance(output, SamplerOutput)
    assert output.sequences.shape[0] == 1
    assert output.sequences.shape[1] >= 1


def test_dream_sampler_outputs_tokens() -> None:
    """DreamSampler returns discrete tokens from masked diffusion."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    config = DreamSamplerConfig(num_steps=4, num_tokens=4)
    sampler = DreamSampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert isinstance(output, SamplerOutput)
    assert output.sequences.shape == (1, 7)
    assert output.sequences.dtype in (torch.long, torch.int)


def test_llada_sampler_with_history() -> None:
    """LLaDASampler can return intermediate canvases."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    config = LLaDASamplerConfig(num_steps=4, num_tokens=4, return_history=True)
    sampler = LLaDASampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert isinstance(output, SamplerOutput)
    assert len(output.history) > 1


def test_dream_sampler_cfg() -> None:
    """DreamSampler supports classifier-free guidance."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    config = DreamSamplerConfig(num_steps=4, num_tokens=4, cfg_scale=1.0)
    sampler = DreamSampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3]])

    output = sampler.sample(prompt)
    assert isinstance(output, SamplerOutput)
    assert output.sequences.shape == (1, 7)


def test_fast_dllm_sampler_trim_stop() -> None:
    """FastDLLMSampler trims trailing stop tokens in decode_final."""
    model = DummyDiffusionModel(vocab_size=16, hidden_dim=32)
    config = FastDLLMSamplerConfig(num_steps=4, num_tokens=4, block_size=4, small_block_size=4)
    sampler = FastDLLMSampler(model, config=config)
    sampler.initialize(torch.tensor([[1, 2, 3]]))

    x = torch.tensor([[1, 2, 3, 14, 14, 14]])  # 14 is eos/stop
    decoded = sampler.decode_final(x)
    # decode_final removes leading pads/masks and keeps a single trailing stop.
    assert decoded.shape[0] == 1
    assert decoded[0, -1].item() == model.tokenizer.eos_token_id


def test_sampling_utilities() -> None:
    """Smoke test for shared sampling helpers used by all samplers."""
    from actfold.models.sampling_utils import (
        LinearMaskingScheduler,
        add_gumbel_noise,
        get_num_transfer_tokens,
        right_shift_logits,
        sample_tokens,
        top_k_logits,
        top_p_logits,
    )

    scheduler = LinearMaskingScheduler()
    assert scheduler.alpha(0.0) == 1.0
    assert scheduler.alpha(1.0) == 0.0

    mask_index = torch.tensor([[True, True, True, True]])
    transfers = get_num_transfer_tokens(mask_index, steps=4, scheduler=scheduler)
    assert transfers.shape[0] == 1
    assert transfers.sum().item() == 4

    logits = torch.randn(2, 4, 16)
    shifted = right_shift_logits(logits)
    assert shifted.shape == logits.shape

    top_k = top_k_logits(logits, top_k=5)
    assert (top_k == float("-inf")).sum().item() > 0

    top_p = top_p_logits(logits, top_p=0.9)
    assert top_p.shape == logits.shape

    conf, tokens = sample_tokens(logits, temperature=0.0)
    assert tokens.shape == (2, 4)

    noisy = add_gumbel_noise(logits, temperature=0.0)
    assert noisy.shape == logits.shape


# ---------------------------------------------------------------------------
# T018: vectorized samplers (bit-exact deterministic numerics, bounded syncs)
# ---------------------------------------------------------------------------
class T018DeterministicModel(DiffusionLLM):
    """Deterministic fake DiffusionLLM for T018 sampler tests.

    Logits are a fixed position-based function of (position, vocab index, the
    row's token sum, and the local token id), so forward passes are
    bit-reproducible and involve no random weights.  All kwargs the samplers
    pass (attention_mask, position_ids, branch_id, parent_branch_id,
    step_idx, folded_model) are accepted.  The mask-token logit is pushed
    down so argmax predictions are always real tokens.
    """

    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__("t018-deterministic")
        self.tokenizer = DummyTokenizer(vocab_size)
        self._vocab_size = vocab_size
        self.forward_calls = 0

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del attention_mask, position_ids, kwargs
        self.forward_calls += 1
        T = tokens.shape[1]
        pos = torch.arange(T, dtype=torch.float32)
        vs = torch.arange(self._vocab_size, dtype=torch.float32)
        base = torch.sin(0.71 * pos[:, None] + 0.13 * vs[None, :])
        seq_bias = torch.cos(0.29 * tokens.to(torch.float32).sum(dim=1, keepdim=True))
        logits = base.unsqueeze(0) * seq_bias.unsqueeze(2)
        logits = logits + 0.037 * tokens.to(torch.float32)[:, :, None] + 0.011 * vs
        logits[..., self.tokenizer.mask_token_id] -= 10.0
        return logits

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.one_hot(tokens, self._vocab_size).float()

    @property
    def num_layers(self) -> int:
        return 2

    @property
    def hidden_dim(self) -> int:
        return 32

    @property
    def num_heads(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class _ReferenceLLaDASampler(LLaDASampler):
    """Test-local LLaDASampler that keeps the pre-T018 per-row sample loop."""

    def reference_sample(
        self,
        prompt_ids: torch.Tensor,
        folded_model: FoldedModel | None = None,
    ) -> SamplerOutput:
        """Frozen pre-T018 LLaDA block-wise sampling loop (verbatim copy).

        This is a verbatim copy of the current ``LLaDASampler.sample`` so the
        vectorized rewrite can be checked for bit-exact output parity.
        """
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        self._reset_folding_chain()
        x = self.initialize(prompt_ids)
        B, T = x.shape
        max_new_tokens = self.config.num_tokens
        block_size = self._block_size
        mask_token_id = self._mask_token_id
        eos_token_id = self._eos_token_id

        # Pre-compute per-sample prompt lengths and attention mask.
        prompt_lens = []
        for i in range(B):
            non_eos = (x[i] != eos_token_id).nonzero(as_tuple=False)
            if non_eos.numel() == 0:
                prompt_lens.append(0)
            else:
                # First masked position after prompt.
                mask_positions = (x[i] == mask_token_id).nonzero(as_tuple=False)
                if mask_positions.numel() == 0:
                    prompt_lens.append(T)
                else:
                    prompt_lens.append(int(mask_positions[0].item()))

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=x.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that are given at the start (non-mask, non-EOS, valid).
        unmasked_index = (x != mask_token_id) & attention_mask.bool()
        if self.config.cfg_keep_tokens:
            keep = torch.isin(
                x,
                torch.tensor(self.config.cfg_keep_tokens, device=x.device),
            )
            unmasked_index = unmasked_index & ~keep

        num_blocks = max(1, math.ceil(max_new_tokens / block_size))
        steps_per_block = max(1, math.ceil(self.config.num_steps / num_blocks))
        history: list[torch.Tensor] = [x.clone()] if self.config.return_history else []

        for block_idx in range(num_blocks):
            # Determine which positions in this block are still masked.
            block_mask_index = torch.zeros((B, block_size), dtype=torch.bool, device=x.device)
            for j in range(B):
                start = prompt_lens[j] + block_idx * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = x[j, start:end] == mask_token_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.config.scheduler,
                stochastic=self.config.stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            for _ in range(effective_steps):
                mask_index = x == mask_token_id

                logits = self._forward_with_cfg(
                    x=x,
                    attention_mask=attention_mask,
                    unmasked_index=unmasked_index,
                    folded_model=folded_model,
                    step_idx=block_idx,
                )

                if self.config.suppress_tokens:
                    for token_id in self.config.suppress_tokens:
                        logits[:, :, token_id] = float("-inf")

                if self.config.right_shift_logits:
                    logits = right_shift_logits(logits)

                if self.config.begin_suppress_tokens:
                    for token_id in self.config.begin_suppress_tokens:
                        logits[:, :, token_id] = float("-inf")

                # Greedy prediction with optional Gumbel noise.
                logits_noisy = add_gumbel_noise(logits, self.config.temperature)
                x0 = torch.argmax(logits_noisy, dim=-1)

                # Confidence for choosing which masks to commit.
                if self.config.remasking == "low_confidence":
                    probs = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                else:  # random
                    x0_p = torch.rand(x0.shape, device=x0.device)

                # Restrict selection window to the current block.
                for j in range(B):
                    block_end = prompt_lens[j] + (block_idx + 1) * block_size
                    x0_p[j, block_end:] = float("-inf")

                # Only allow updates at currently masked positions.
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, float("-inf"))

                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, _].item())
                    if k > 0:
                        # T018 note: the production loop unpacks
                        # ``_, select_idx = torch.topk(...)`` which shadows the
                        # outer step variable ``_`` and makes the NEXT row's
                        # ``num_transfer_tokens[j, _]`` index with a float
                        # tensor (IndexError for B >= 2).  The unpacking is
                        # renamed here so the reference can define the
                        # intended per-row semantics; for B == 1 the behavior
                        # is bit-identical to the production loop.
                        _topk_values, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                x[transfer_index] = x0[transfer_index]
                if self.config.return_history:
                    history.append(x.clone())

        decoded = self.decode_final(x)
        return SamplerOutput(sequences=decoded, history=history)


class _ReferenceDreamSampler(DreamSampler):
    """Test-local DreamSampler that keeps the pre-T018 per-row sample loop."""

    def reference_sample(
        self,
        prompt_ids: torch.Tensor,
        folded_model: FoldedModel | None = None,
    ) -> SamplerOutput:
        """Frozen pre-T018 Dream sampling loop (verbatim copy).

        This is a verbatim copy of the current ``DreamSampler.sample`` so the
        vectorized rewrite can be checked for bit-exact output parity.
        """
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        self._reset_folding_chain()
        x = self.initialize(prompt_ids)
        B, T = x.shape
        mask_token_id = self._mask_token_id
        attention_mask = self._attention_mask

        pos_id: torch.Tensor | None = None
        if torch.any(attention_mask == 0):
            pos_id = compute_position_ids(attention_mask)

        mask_index = x == mask_token_id
        num_transfer_tokens_list = get_num_transfer_tokens(
            mask_index=mask_index,
            steps=self.config.num_steps,
            scheduler=self.config.scheduler,
            stochastic=self.config.stochastic_transfer,
        )
        effective_steps = num_transfer_tokens_list.size(1)

        # For CFG, only the original prompt positions are masked in the
        # unconditional branch; step-wise revealed tokens are not masked again.
        prompt_index = attention_mask.bool() & (
            torch.arange(T, device=x.device).unsqueeze(0) < T - self.config.num_tokens
        )

        history: list[torch.Tensor] = [x.clone()] if self.config.return_history else []

        for step in range(effective_steps):
            mask_index = x == mask_token_id

            logits = self._forward_with_cfg(
                x=x,
                attention_mask=attention_mask,
                position_ids=pos_id,
                prompt_index=prompt_index,
                folded_model=folded_model,
                step_idx=step,
            )

            if self.config.right_shift_logits:
                logits = right_shift_logits(logits)

            mask_logits = logits[mask_index]
            if mask_logits.numel() == 0:
                break

            confidence, x0 = sample_tokens(
                mask_logits,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                margin_confidence=self.config.alg == "topk_margin",
                neg_entropy=self.config.alg == "entropy",
            )

            full_confidence = torch.full_like(x, float("-inf"), device=x.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence

            for j in range(B):
                k = int(num_transfer_tokens_list[j, step].item())
                if k > 0:
                    if self.config.alg_temp is None or self.config.alg_temp == 0.0:
                        _, transfer_index = torch.topk(full_confidence[j], k=k)
                    else:
                        fc = full_confidence[j] / self.config.alg_temp
                        fc = F.softmax(fc, dim=-1)
                        transfer_index = torch.multinomial(fc, num_samples=k)

                    x_ = torch.full_like(x, mask_token_id, device=x.device)
                    x_[mask_index] = x0
                    x[j, transfer_index] = x_[j, transfer_index]

            if self.config.return_history:
                history.append(x.clone())

        decoded = self.decode_final(x)
        return SamplerOutput(sequences=decoded, history=history)


def _install_sync_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch ``torch.Tensor.item``/``tolist``/``cpu`` to count host readbacks.

    The patched methods keep their original behavior; they only increment a
    per-method counter so tests can measure tensor-to-host synchronization
    counts without breaking the code under test.
    """
    counts: dict[str, int] = {"item": 0, "tolist": 0, "cpu": 0}

    def _make_wrapper(method_name: str, original: Any) -> Any:
        def _counting(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            counts[method_name] += 1
            return original(self, *args, **kwargs)

        return _counting

    for method_name in ("item", "tolist", "cpu"):
        original = getattr(torch.Tensor, method_name)
        monkeypatch.setattr(torch.Tensor, method_name, _make_wrapper(method_name, original))
    return counts


def _total_syncs(counts: dict[str, int]) -> int:
    """Return the total number of counted host readbacks."""
    return counts["item"] + counts["tolist"] + counts["cpu"]


def _assert_history_equal(
    new_out: SamplerOutput, ref_out: SamplerOutput, expected_min_entries: int
) -> None:
    """Assert two sampler outputs have identical intermediate canvases."""
    assert len(new_out.history) == len(ref_out.history)
    assert len(new_out.history) >= expected_min_entries
    for new_x, ref_x in zip(new_out.history, ref_out.history):
        assert torch.equal(new_x, ref_x)


# T018: LLaDASampler.sample must stay bit-identical to the frozen reference.
_T018_LLADA_PARITY_PROMPTS = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


@pytest.mark.parametrize(
    "config_kwargs",
    [
        pytest.param(
            dict(
                num_steps=6,
                num_tokens=4,
                temperature=0.0,
                remasking="low_confidence",
                suppress_tokens=[3],
                begin_suppress_tokens=[7],
                seed=1234,
                return_history=True,
            ),
            id="single_block",
        ),
        pytest.param(
            dict(
                num_steps=9,
                num_tokens=6,
                block_size=4,
                temperature=0.0,
                remasking="low_confidence",
                cfg_scale=1.0,
                suppress_tokens=[3],
                begin_suppress_tokens=[7],
                seed=1234,
                return_history=True,
            ),
            id="two_blocks_cfg",
        ),
    ],
)
@pytest.mark.parametrize("batch_rows", [pytest.param(1, id="b1"), pytest.param(3, id="b3")])
def test_t018_llada_output_matches_reference(
    config_kwargs: dict[str, Any], batch_rows: int
) -> None:
    """Vectorized LLaDASampler.sample reproduces the per-row loop bit-exactly.

    With a deterministic fake model, temperature=0 and low-confidence
    remasking, the rewritten sample() must return the identical sequence and
    identical history canvases as the frozen pre-T018 reference loop, for a
    single row and for a multi-row batch.
    """
    model = T018DeterministicModel()
    sampler = _ReferenceLLaDASampler(model, config=LLaDASamplerConfig(**config_kwargs))
    prompt = _T018_LLADA_PARITY_PROMPTS[:batch_rows]

    new_out = sampler.sample(prompt)
    ref_out = sampler.reference_sample(prompt)

    assert torch.equal(new_out.sequences, ref_out.sequences)
    _assert_history_equal(new_out, ref_out, expected_min_entries=2)


# T018: host-sync count must not grow with the batch size.
def test_t018_llada_sync_count_batch_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLaDASampler.sample syncs to host equally for B=1 and B=4 (<= 12).

    Replicating the same prompt across batch rows must not change the number
    of tensor-to-host readbacks (item/tolist/cpu) performed by a full
    sample() call, the count must stay within the sync budget, and every
    replicated row must decode identically to the B=1 run.
    """
    counts = _install_sync_counter(monkeypatch)
    model = T018DeterministicModel()
    prompt = torch.tensor([[2, 5, 7, 11]])
    config = LLaDASamplerConfig(
        num_steps=8, num_tokens=6, block_size=3, temperature=0.0, seed=7
    )
    sampler = LLaDASampler(model, config=config)

    out_b1 = sampler.sample(prompt)
    syncs_b1 = _total_syncs(counts)
    for key in counts:
        counts[key] = 0

    out_b4 = sampler.sample(prompt.repeat(4, 1))
    syncs_b4 = _total_syncs(counts)

    assert syncs_b4 <= 12
    assert syncs_b1 == syncs_b4
    for row in range(4):
        assert torch.equal(out_b4.sequences[row], out_b1.sequences[0])


# T018: LLaDASampler.sample must break before a forward when nothing is masked.
def test_t018_llada_early_stop() -> None:
    """LLaDASampler skips forward passes once no masked positions remain.

    With an empty generation region the schedule still contains one
    zero-transfer step ("more scheduled steps than needed"); the sampler must
    break before running that forward, and the final canvas must contain no
    mask tokens anywhere.
    """
    model = T018DeterministicModel()
    config = LLaDASamplerConfig(
        num_steps=8, num_tokens=0, block_size=1, temperature=0.0, seed=3
    )
    sampler = LLaDASampler(model, config=config)
    prompt = torch.tensor([[2, 5, 7, 11]])

    # An all-False block schedule still pads to one effective step.
    scheduled_steps = get_num_transfer_tokens(
        mask_index=torch.zeros((1, 1), dtype=torch.bool), steps=8
    ).size(1)
    assert scheduled_steps == 1

    before = model.forward_calls
    output = sampler.sample(prompt)
    forwards = model.forward_calls - before

    assert forwards < scheduled_steps
    assert not bool((output.sequences == model.tokenizer.mask_token_id).any())
    assert torch.equal(output.sequences, prompt)


# T018: DreamSampler.sample must stay bit-identical to the frozen reference.
@pytest.mark.parametrize(
    "alg,cfg_scale",
    [
        pytest.param("maskgit_plus", 0.0, id="maskgit_plus"),
        pytest.param("topk_margin", 0.0, id="topk_margin"),
        pytest.param("maskgit_plus", 1.0, id="maskgit_plus_cfg"),
    ],
)
def test_t018_dream_output_matches_reference(alg: str, cfg_scale: float) -> None:
    """Vectorized DreamSampler.sample reproduces the per-row loop bit-exactly.

    With a deterministic fake model, temperature=0 and alg_temp=0, both the
    maskgit_plus and topk_margin confidence rules must return the identical
    sequence and identical history canvases as the frozen reference loop.
    """
    model = T018DeterministicModel()
    config = DreamSamplerConfig(
        num_steps=8,
        num_tokens=6,
        temperature=0.0,
        alg_temp=0.0,
        alg=alg,
        cfg_scale=cfg_scale,
        seed=1234,
        return_history=True,
    )
    sampler = _ReferenceDreamSampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    new_out = sampler.sample(prompt)
    ref_out = sampler.reference_sample(prompt)

    assert torch.equal(new_out.sequences, ref_out.sequences)
    _assert_history_equal(new_out, ref_out, expected_min_entries=2)


# T018: host-sync count must not grow with the batch size.
def test_t018_dream_sync_count_batch_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """DreamSampler.sample syncs to host equally for B=1 and B=4 (<= 12).

    Replicating the same prompt across batch rows must not change the number
    of tensor-to-host readbacks (item/tolist/cpu) performed by a full
    sample() call, the count must stay within the sync budget, and every
    replicated row must decode identically to the B=1 run.
    """
    counts = _install_sync_counter(monkeypatch)
    model = T018DeterministicModel()
    prompt = torch.tensor([[2, 5, 7, 11]])
    config = DreamSamplerConfig(
        num_steps=8, num_tokens=6, temperature=0.0, alg="maskgit_plus", seed=7
    )
    sampler = DreamSampler(model, config=config)

    out_b1 = sampler.sample(prompt)
    syncs_b1 = _total_syncs(counts)
    for key in counts:
        counts[key] = 0

    out_b4 = sampler.sample(prompt.repeat(4, 1))
    syncs_b4 = _total_syncs(counts)

    assert syncs_b4 <= 12
    assert syncs_b1 == syncs_b4
    for row in range(4):
        assert torch.equal(out_b4.sequences[row], out_b1.sequences[0])
