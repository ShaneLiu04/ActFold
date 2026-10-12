"""AR012 (UT-1101..1106): numeric coverage for stochastic sampler branches.

Pure functions (G1/G5) are pinned by same-seed independent recompute of the
mathematical formula; end-to-end random branches (G2/G3/G4) are pinned by
same-seed determinism plus RNG-call spies (positive proof the branch was
hit) plus structural invariants; the stochastic schedule (G6) is pinned by
determinism, distinguishability from the deterministic mode, and a 6-sigma
statistical mean check of the step-0 Binomial parameter.  Every case seeds
torch explicitly and never relies on the conftest autouse seed.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from actfold.models.dream_sampler import DreamSampler, DreamSamplerConfig
from actfold.models.llada_sampler import LLaDASampler, LLaDASamplerConfig
from actfold.models.sampling_utils import (
    add_gumbel_noise,
    get_num_transfer_tokens,
    sample_tokens,
)
from actfold.models import llada_sampler as _llada_module
from tests.test_diffusion_samplers import T018DeterministicModel

_REAL_BINOMIAL = torch.distributions.Binomial


def _f32_round(x: float) -> float:
    """Round a Python float to float32 precision (test-local recompute)."""
    return float(torch.tensor(x, dtype=torch.float32))


def _sample_twice(
    sampler: Any,
    prompt: torch.Tensor,
    seed: int,
) -> tuple[Any, Any]:
    """Run ``sample()`` twice under the same explicit seed and assert equality.

    Returns both outputs after proving same-seed determinism of the sequences
    and of every history canvas.
    """
    torch.manual_seed(seed)
    out1 = sampler.sample(prompt, folded_model=None)
    torch.manual_seed(seed)
    out2 = sampler.sample(prompt, folded_model=None)
    assert torch.equal(out1.sequences, out2.sequences)
    assert len(out1.history) == len(out2.history)
    for h1, h2 in zip(out1.history, out2.history):
        assert torch.equal(h1, h2)
    return out1, out2


def _spy_torch_rand(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[int, ...]]) -> None:
    """Record every ``torch.rand`` shape, then delegate once to the original."""

    real = torch.rand

    def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
        size = args[0] if args else kwargs["size"]
        calls.append(tuple(size))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", wrapper)


def _spy_torch_multinomial(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[tuple[int, ...], int, torch.Tensor, torch.Tensor]],
) -> None:
    """Record (input shape, num_samples, input, result) for torch.multinomial."""

    real = torch.multinomial

    def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
        inp = args[0]
        num_samples = args[1] if len(args) > 1 else kwargs["num_samples"]
        result = real(*args, **kwargs)
        calls.append(
            (tuple(inp.shape), int(num_samples), inp.detach().clone(), result.detach().clone())
        )
        return result

    monkeypatch.setattr(torch, "multinomial", wrapper)


def _spy_binomial(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[torch.Tensor, float]]
) -> None:
    """Proxy torch.distributions.Binomial: record construction, forward sample().

    Construction does not consume RNG and ``.sample()`` is forwarded to the
    real instance, so the RNG consumption order is unchanged.
    """

    class _BinomialProxy:
        def __init__(self, total_count: Any, probs: Any, **kwargs: Any) -> None:
            calls.append((total_count.detach().clone(), float(probs)))
            self._inner = _REAL_BINOMIAL(total_count, probs, **kwargs)

        def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
            return self._inner.sample(sample_shape)

    monkeypatch.setattr(torch.distributions, "Binomial", _BinomialProxy)


def _spy_llada_gumbel(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[tuple[int, ...], float]]
) -> None:
    """Record (shape, temperature) at the llada_sampler from-import binding."""

    real = _llada_module.add_gumbel_noise

    def wrapper(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        calls.append((tuple(logits.shape), float(temperature)))
        return real(logits, temperature)

    monkeypatch.setattr(_llada_module, "add_gumbel_noise", wrapper)


# ---------------------------------------------------------------------------
# UT-1101 (G1): add_gumbel_noise exact formula recompute
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 2.0])
def test_ut1101_add_gumbel_noise_exact_recompute(temperature: float) -> None:
    """add_gumbel_noise matches a hand-recomputed Gumbel formula bit-exactly.

    For T>0 the output must equal ``exp(logits) / (-log(u))**T`` with ``u``
    drawn from the same seed, be float64, and be same-seed deterministic.
    For T=0.0 it must be a value-level no-op returning the original object.
    """
    seed = 1101
    torch.manual_seed(seed)
    logits = torch.randn(2, 3, 5)

    if temperature == 0.0:
        assert add_gumbel_noise(logits, 0.0) is logits
        return

    torch.manual_seed(seed)
    out1 = add_gumbel_noise(logits, temperature)
    torch.manual_seed(seed)
    out2 = add_gumbel_noise(logits, temperature)
    assert out1.dtype == torch.float64
    assert torch.equal(out1, out2)

    torch.manual_seed(seed)
    noise = torch.rand_like(logits.double(), dtype=torch.float64)
    ref = logits.double().exp() / (-torch.log(noise)) ** temperature
    assert torch.equal(out1, ref)


# ---------------------------------------------------------------------------
# UT-1102 (G2): LLaDA remasking="random" branch
# ---------------------------------------------------------------------------
def test_ut1102_llada_random_remasking_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """random remasking hits torch.rand on the canvas shape and stays deterministic.

    Same-seed runs are bit-equal (sequences and history); the torch.rand spy
    proves the random branch executed (empty spy means silent fallback to
    low_confidence); the new-token region is fully revealed, in-vocabulary,
    and free of the suppressed token.
    """
    seed = 1102
    model = T018DeterministicModel()
    config = LLaDASamplerConfig(
        num_steps=6,
        num_tokens=5,
        temperature=0.0,
        remasking="random",
        suppress_tokens=[3],
        seed=seed,
        return_history=True,
    )
    sampler = LLaDASampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    rand_calls: list[tuple[int, ...]] = []
    _spy_torch_rand(monkeypatch, rand_calls)

    out1, _ = _sample_twice(sampler, prompt, seed)

    assert any(shape == (2, 9) for shape in rand_calls)

    new_region = out1.sequences[:, 4:]
    assert not bool((new_region == model.tokenizer.mask_token_id).any())
    assert bool(((new_region >= 0) & (new_region < model.vocab_size)).all())
    assert not bool((new_region == 3).any())


# ---------------------------------------------------------------------------
# UT-1103 (G3): LLaDA stochastic_transfer=True branch
# ---------------------------------------------------------------------------
def test_ut1103_llada_stochastic_transfer_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stochastic_transfer hits the Binomial schedule and the T>0 gumbel path.

    Same-seed runs are bit-equal; the Binomial spy proves the stochastic
    schedule branch executed; the add_gumbel_noise spy proves the T=0.5
    in-place path ran; the canvas history is monotone (mask count never
    increases) and the final new-token region is fully revealed.
    """
    seed = 1103
    model = T018DeterministicModel()
    config = LLaDASamplerConfig(
        num_steps=6,
        num_tokens=5,
        temperature=0.5,
        remasking="low_confidence",
        stochastic_transfer=True,
        seed=seed,
        return_history=True,
    )
    sampler = LLaDASampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    binomial_calls: list[tuple[torch.Tensor, float]] = []
    _spy_binomial(monkeypatch, binomial_calls)
    gumbel_calls: list[tuple[tuple[int, ...], float]] = []
    _spy_llada_gumbel(monkeypatch, gumbel_calls)

    out1, _ = _sample_twice(sampler, prompt, seed)

    assert len(binomial_calls) > 0
    assert any(t == 0.5 for _, t in gumbel_calls)

    mask_token_id = model.tokenizer.mask_token_id
    mask_counts = [(h == mask_token_id).sum().item() for h in out1.history]
    assert all(c2 <= c1 for c1, c2 in zip(mask_counts, mask_counts[1:]))
    assert not bool((out1.sequences[:, 4:] == mask_token_id).any())


# ---------------------------------------------------------------------------
# UT-1104 (G4): Dream alg_temp>0 per-row multinomial soft selection
# ---------------------------------------------------------------------------
def test_ut1104_dream_alg_temp_soft_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """alg_temp>0 hits per-row torch.multinomial soft selection deterministically.

    Same-seed runs are bit-equal; the multinomial spy proves the soft-selection
    branch executed (empty spy means silent fallback to batched topk); the
    final canvas is fully revealed.  Weak property (d): with alg_temp=1e-8
    the softmaxed confidence is numerically one-hot, so each row's first
    multinomial draw must equal that row's confidence argmax.
    """
    seed = 1104
    model = T018DeterministicModel()
    config = DreamSamplerConfig(
        num_steps=8,
        num_tokens=6,
        temperature=0.0,
        alg="maskgit_plus",
        alg_temp=1.0,
        cfg_scale=0.0,
        seed=seed,
        return_history=True,
    )
    sampler = DreamSampler(model, config=config)
    prompt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    multinomial_calls: list[tuple[tuple[int, ...], int, torch.Tensor, torch.Tensor]] = []
    _spy_torch_multinomial(monkeypatch, multinomial_calls)

    out1, _ = _sample_twice(sampler, prompt, seed)

    assert len(multinomial_calls) > 0
    assert not bool((out1.sequences == model.tokenizer.mask_token_id).any())

    multinomial_calls.clear()
    weak_config = DreamSamplerConfig(
        num_steps=8,
        num_tokens=6,
        temperature=0.0,
        alg="maskgit_plus",
        alg_temp=1e-8,
        cfg_scale=0.0,
        seed=seed,
        return_history=False,
    )
    weak_sampler = DreamSampler(model, config=weak_config)
    torch.manual_seed(seed)
    weak_sampler.sample(prompt, folded_model=None)
    assert len(multinomial_calls) > 0
    for _shape, _k, fc, result in multinomial_calls:
        assert int(result[0]) == int(fc.argmax())


# ---------------------------------------------------------------------------
# UT-1105 (G5): sample_tokens T>0 Categorical happy path
# ---------------------------------------------------------------------------
_UT1105_CASES = [
    pytest.param(dict(temperature=0.7), id="temp_only"),
    pytest.param(dict(temperature=0.7, top_k=3), id="top_k"),
    pytest.param(dict(temperature=0.7, top_p=0.6), id="top_p"),
    pytest.param(dict(temperature=0.7, top_p=0.6, top_k=3), id="top_p_top_k"),
    pytest.param(dict(temperature=0.7, top_k=3, margin_confidence=True), id="margin"),
    pytest.param(dict(temperature=0.7, top_p=0.6, neg_entropy=True), id="neg_entropy"),
]


@pytest.mark.parametrize("kwargs", _UT1105_CASES)
def test_ut1105_sample_tokens_exact_recompute(kwargs: dict[str, Any]) -> None:
    """sample_tokens matches a hand-recomputed filter/sample chain bit-exactly.

    The recompute mirrors the production operator order (temperature divide,
    top_p nucleus filter, top_k cutoff, softmax, Categorical sample, gather)
    without importing the production helpers.  Tokens filtered to -inf carry
    exactly zero probability, so the gathered probability of every sampled
    token is strictly positive.
    """
    seed = 1105
    torch.manual_seed(seed)
    logits = torch.randn(4, 6)

    torch.manual_seed(seed)
    conf, tokens = sample_tokens(logits, **kwargs)

    temperature = kwargs["temperature"]
    top_p = kwargs.get("top_p")
    top_k = kwargs.get("top_k")

    torch.manual_seed(seed)
    lg = logits / temperature
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(lg, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 0] = False
        indices_remove = sorted_remove.scatter(-1, sorted_indices, sorted_remove)
        lg = lg.masked_fill(indices_remove, float("-inf"))
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(lg, min(top_k, lg.size(-1)), dim=-1)
        min_values = values[..., -1].unsqueeze(-1)
        lg = torch.where(lg < min_values, torch.full_like(lg, float("-inf")), lg)
    probs = F.softmax(lg, dim=-1)
    ref_tokens = torch.distributions.Categorical(probs=probs).sample()
    ref_conf = torch.gather(probs, -1, ref_tokens.unsqueeze(-1)).squeeze(-1)
    if kwargs.get("margin_confidence", False):
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        ref_conf = sorted_probs[..., 0] - sorted_probs[..., 1]
    if kwargs.get("neg_entropy", False):
        log_probs = torch.log(probs + 1e-10)
        ref_conf = -(probs * log_probs).sum(dim=-1)

    assert torch.equal(conf, ref_conf)
    assert torch.equal(tokens, ref_tokens)

    gathered = torch.gather(probs, -1, ref_tokens.unsqueeze(-1)).squeeze(-1)
    assert bool((gathered > 0).all())
    if top_k is not None and top_k > 0:
        assert bool((lg == float("-inf")).any())
        assert float(probs[lg == float("-inf")].sum()) == 0.0


# ---------------------------------------------------------------------------
# UT-1106 (G6): get_num_transfer_tokens stochastic numeric pin
# ---------------------------------------------------------------------------
def test_ut1106_stochastic_transfer_schedule_numeric() -> None:
    """Stochastic schedule is deterministic, distinguishable, and Binomial-correct.

    (a) same-seed outputs are bit-equal; (b) stochastic output differs from
    the deterministic expected-value mode at a pinned seed (equal outputs
    would mean the stochastic flag is silently ignored); (c) over seeds
    0..99 the step-0 transfer mean matches the analytically recomputed
    ``n * p0`` within 6 sigma.  p0 is hand-recomputed from the linear
    schedule alpha(t) = 1 - t with float32 rounding mirrored operator by
    operator, without importing the production scheduler.  Measurement
    caliber: ``out[0, 0]`` is the first positive entry (production drops
    all-zero steps), so seeds whose step-0 draw is zero substitute a
    later, larger-p step — a small upward bias absorbed by the 6-sigma
    band.
    """
    gen = torch.Generator().manual_seed(7)
    mask_index = torch.rand((4, 20), generator=gen) < 0.6

    torch.manual_seed(1106)
    sto1 = get_num_transfer_tokens(mask_index, steps=8, stochastic=True)
    torch.manual_seed(1106)
    sto2 = get_num_transfer_tokens(mask_index, steps=8, stochastic=True)
    assert torch.equal(sto1, sto2)

    torch.manual_seed(0)
    sto = get_num_transfer_tokens(mask_index, steps=8, stochastic=True)
    det = get_num_transfer_tokens(mask_index, steps=8, stochastic=False)
    assert not torch.equal(sto, det)

    steps = 8
    s_norm = (steps - 1) / steps
    t_norm = steps / steps
    a_s = _f32_round(1.0 - s_norm)
    a_t = _f32_round(1.0 - t_norm)
    numerator = _f32_round(1.0 - a_s)
    denominator = _f32_round(_f32_round(1.0 - a_t) + 1e-6)
    p0 = 1.0 - _f32_round(numerator / denominator)
    assert p0 == 0.12500083446502686

    full = torch.ones((1, 20), dtype=torch.bool)
    n = 20
    expected = n * p0
    values = []
    for sd in range(100):
        torch.manual_seed(sd)
        out = get_num_transfer_tokens(full, steps=steps, stochastic=True)
        values.append(out[0, 0].item())
    mean = sum(values) / len(values)
    assert abs(mean - expected) <= 0.89
