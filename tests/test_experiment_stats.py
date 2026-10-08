"""Tests for T020 "统计方法": timing statistics and repeat-based sampling experiments.

Red-phase tests encoding the Green contract for:
- ``scripts/algo_experiments.py``: ``TimingStats``, ``stats_from_samples``,
  ``time_forward``, ``repeat_with_seed``, ``exp_sampling``.
- ``scripts/make_experiment_figures.py``: ``fig_sampling`` with the new
  mean/std sampling schema plus backward compatibility with the old schema.

Imports of the modules under test happen inside each test function so that a
missing name fails only that test instead of erroring at collection time.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strings_from_calls(mock: Any) -> list[str]:
    """Collect every string argument (positional or keyword) from a MagicMock's calls."""
    found: list[str] = []
    for args, kwargs in mock.call_args_list:
        for item in (*args, *kwargs.values()):
            if isinstance(item, str):
                found.append(item)
    return found


def _patched_axes(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, list[MagicMock]]:
    """Monkeypatch ``plt.subplots`` in the figures module and return (fig, axes)."""
    from scripts import make_experiment_figures

    fig = MagicMock(name="fig")
    axes = [MagicMock(name="ax0"), MagicMock(name="ax1")]

    def fake_subplots(*args: Any, **kwargs: Any) -> tuple[MagicMock, list[MagicMock]]:
        return fig, axes

    monkeypatch.setattr(make_experiment_figures.plt, "subplots", fake_subplots)
    return fig, axes


def _sampling_model(sampling: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "fastdllm": {
            "results": {
                "meta": {"label": "Fast-dLLM"},
                "sampling": sampling,
            }
        }
    }


# ---------------------------------------------------------------------------
# TimingStats dataclass
# ---------------------------------------------------------------------------


def test_timing_stats_as_dict_roundtrip() -> None:
    from scripts.algo_experiments import TimingStats

    stats = TimingStats(mean=1.0, std=0.1, p50=1.0, n=3)

    as_dict = stats.as_dict()
    assert isinstance(as_dict, dict)
    assert set(as_dict) == {"mean", "std", "p50", "n"}
    assert as_dict["mean"] == pytest.approx(1.0)
    assert as_dict["std"] == pytest.approx(0.1)
    assert as_dict["p50"] == pytest.approx(1.0)
    assert as_dict["n"] == 3


def test_timing_stats_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    from scripts.algo_experiments import TimingStats

    stats = TimingStats(mean=1.0, std=0.1, p50=1.0, n=3)

    with pytest.raises(FrozenInstanceError):
        stats.mean = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# stats_from_samples
# ---------------------------------------------------------------------------


def test_stats_from_samples_even_sample() -> None:
    from scripts.algo_experiments import stats_from_samples

    stats = stats_from_samples([1.0, 2.0, 3.0, 4.0])

    assert stats.mean == pytest.approx(2.5)
    assert stats.std == pytest.approx(1.25**0.5)
    assert stats.p50 == pytest.approx(2.5)
    assert stats.n == 4


def test_stats_from_samples_odd_median() -> None:
    from scripts.algo_experiments import stats_from_samples

    stats = stats_from_samples([1.0, 2.0, 3.0])

    assert stats.p50 == pytest.approx(2.0)
    assert stats.mean == pytest.approx(2.0)
    assert stats.n == 3


def test_stats_from_samples_single_value() -> None:
    from scripts.algo_experiments import stats_from_samples

    stats = stats_from_samples([5.0])

    assert stats.mean == pytest.approx(5.0)
    assert stats.std == pytest.approx(0.0)
    assert stats.p50 == pytest.approx(5.0)
    assert stats.n == 1


def test_stats_from_samples_empty_raises() -> None:
    from scripts.algo_experiments import stats_from_samples

    with pytest.raises(ValueError):
        stats_from_samples([])


# ---------------------------------------------------------------------------
# time_forward
# ---------------------------------------------------------------------------


def test_time_forward_reps_below_floor_raises_before_work() -> None:
    from scripts.algo_experiments import time_forward

    called: list[int] = []

    def fn() -> None:
        called.append(1)

    with pytest.raises(ValueError):
        time_forward(fn, warmup=0, reps=2)

    assert called == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_time_forward_cuda_returns_timing_stats() -> None:
    from scripts.algo_experiments import TimingStats, time_forward

    x = torch.zeros(8, device="cuda")

    def fn() -> None:
        x.add_(1.0)

    stats = time_forward(fn, warmup=1, reps=5)

    assert isinstance(stats, TimingStats)
    assert stats.n == 5
    assert stats.mean > 0


# ---------------------------------------------------------------------------
# repeat_with_seed
# ---------------------------------------------------------------------------


def test_repeat_with_seed_deterministic_and_restores_global_rng() -> None:
    from scripts.algo_experiments import repeat_with_seed

    seed = 1234

    def fn() -> tuple[torch.Tensor, float]:
        t = torch.randn(4)
        return t, 0.5

    torch.manual_seed(999)
    pre_state = torch.get_rng_state()
    reference = torch.randn(4)
    torch.set_rng_state(pre_state)

    out = repeat_with_seed(fn, repeats=3, seed=seed)

    assert set(out) == {"ms", "tokens", "identical"}
    assert len(out["ms"]) == 3
    assert len(out["tokens"]) == 3
    assert out["identical"] is True
    assert all(isinstance(ms, float) for ms in out["ms"])

    # Caller-visible RNG state must be restored BEFORE we disturb it again.
    assert torch.equal(torch.get_rng_state(), pre_state)
    assert torch.equal(torch.randn(4), reference)

    torch.manual_seed(seed)
    expected = torch.randn(4)
    for token in out["tokens"]:
        assert torch.equal(token, expected)


def test_repeat_with_seed_repeats_one_is_legal() -> None:
    from scripts.algo_experiments import repeat_with_seed

    def fn() -> tuple[torch.Tensor, float]:
        return torch.randn(4), 1.0

    out = repeat_with_seed(fn, repeats=1, seed=7)

    assert len(out["ms"]) == 1
    assert len(out["tokens"]) == 1
    assert out["identical"] is True


def test_repeat_with_seed_repeats_zero_raises() -> None:
    from scripts.algo_experiments import repeat_with_seed

    def fn() -> tuple[torch.Tensor, float]:
        return torch.randn(4), 1.0

    with pytest.raises(ValueError):
        repeat_with_seed(fn, repeats=0, seed=7)


def test_repeat_with_seed_non_deterministic_fn_is_not_identical() -> None:
    from scripts.algo_experiments import repeat_with_seed

    counter = {"i": 0}

    def fn() -> tuple[torch.Tensor, float]:
        counter["i"] += 1
        return torch.full((2,), float(counter["i"])), 1.0

    out = repeat_with_seed(fn, repeats=3, seed=0)

    assert out["identical"] is False
    assert not torch.equal(out["tokens"][0], out["tokens"][1])


def test_repeat_with_seed_restores_rng_on_exception() -> None:
    from scripts.algo_experiments import repeat_with_seed

    def boom() -> tuple[torch.Tensor, float]:
        torch.randn(2)
        raise RuntimeError("boom")

    pre_state = torch.get_rng_state()
    with pytest.raises(RuntimeError):
        repeat_with_seed(boom, repeats=2, seed=0)
    assert torch.equal(torch.get_rng_state(), pre_state)


# ---------------------------------------------------------------------------
# exp_sampling
# ---------------------------------------------------------------------------


def test_exp_sampling_repeats_below_floor_raises_at_entry(tmp_path: Any) -> None:
    from scripts.algo_experiments import exp_sampling

    with pytest.raises(ValueError):
        exp_sampling(None, None, None, tmp_path, repeats=2)


# ---------------------------------------------------------------------------
# fig_sampling
# ---------------------------------------------------------------------------


def test_fig_sampling_new_schema_renders_mean_std(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from scripts.make_experiment_figures import fig_sampling

    _, axes = _patched_axes(monkeypatch)
    sampling = {
        "baseline_ms_mean": 100.0,
        "baseline_ms_std": 5.0,
        "folded_ms_mean": 120.0,
        "folded_ms_std": 6.0,
        "repeats": 3,
        "step_stats": [{"mean_stable": 0.9}, {"mean_stable": 0.8}],
        "token_match_rate": 1.0,
    }
    models = _sampling_model(sampling)

    fig_sampling(models, tmp_path)

    assert axes[0].plot.called

    title_strings = _strings_from_calls(axes[1].set_title)
    assert any("n=" in s for s in title_strings)

    annotation_strings = _strings_from_calls(axes[1].text) + _strings_from_calls(axes[1].bar)
    assert any("±" in s for s in annotation_strings)


def test_fig_sampling_old_schema_backward_compat(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from scripts.make_experiment_figures import fig_sampling

    _, axes = _patched_axes(monkeypatch)
    sampling = {
        "baseline_ms": 100.0,
        "folded_ms": 120.0,
        "step_stats": [{"mean_stable": 0.9}, {"mean_stable": 0.8}],
        "token_match_rate": 1.0,
    }
    models = _sampling_model(sampling)

    fig_sampling(models, tmp_path)

    assert axes[0].plot.called
