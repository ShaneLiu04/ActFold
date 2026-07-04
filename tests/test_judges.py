"""Tests for actfold.eval.judges with real backends."""

from __future__ import annotations

import platform

import pytest

from actfold.eval.judges import EvalPlusJudge, JudgeFactory, LMEvalJudge, _lm_eval_primary_metric


def test_lm_eval_primary_metric_extracts_first_metric() -> None:
    """_lm_eval_primary_metric returns the canonical metric from a result dict."""
    assert _lm_eval_primary_metric({"exact_match": 0.8}) == 0.8
    assert _lm_eval_primary_metric({"acc": 0.75, "other": 0.5}) == 0.75
    assert _lm_eval_primary_metric({"prompt_level_acc": 0.9}) == 0.9
    assert _lm_eval_primary_metric({}) == 0.0


def test_lm_eval_primary_metric_numeric_fallback() -> None:
    """_lm_eval_primary_metric falls back to the first numeric value."""
    assert _lm_eval_primary_metric({"foo": 0.5, "bar": 0.6}) == 0.5


@pytest.mark.slow
def test_lm_eval_judge_prompts_and_score() -> None:
    """LMEvalJudge loads real GSM8K prompts and scores predictions."""
    judge = LMEvalJudge(device="cpu", batch_size=1)
    prompts, references = judge.get_prompts("gsm8k", limit=3)
    assert len(prompts) == 3
    assert len(prompts) == len(references)
    assert isinstance(prompts[0], str)

    predictions = ["answer a", "answer b", "answer c"]
    result = judge.score("gsm8k", predictions, references)
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["num_samples"] == 3


def test_lm_eval_judge_unsupported_task() -> None:
    judge = LMEvalJudge()
    with pytest.raises(ValueError):
        judge.get_prompts("unknown_task")


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="evalplus sandbox requires the Unix 'resource' module",
)
@pytest.mark.slow
def test_evalplus_judge_prompts() -> None:
    """EvalPlusJudge loads real HumanEval+ prompts."""
    judge = EvalPlusJudge()
    prompts, references = judge.get_prompts("humaneval_plus", limit=2)
    assert len(prompts) == 2
    assert len(prompts) == len(references)
    assert "def " in prompts[0]


def test_evalplus_judge_unsupported_task() -> None:
    judge = EvalPlusJudge()
    with pytest.raises(ValueError):
        judge.get_prompts("unknown_task")


def test_judge_factory_auto_selects_backend() -> None:
    judge = JudgeFactory.create("gsm8k", backend="auto", device="cpu")
    assert isinstance(judge, LMEvalJudge)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="evalplus sandbox requires the Unix 'resource' module",
)
def test_judge_factory_auto_selects_evalplus() -> None:
    judge = JudgeFactory.create("humaneval_plus", backend="auto")
    assert isinstance(judge, EvalPlusJudge)


def test_judge_factory_explicit_backend() -> None:
    judge = JudgeFactory.create("gsm8k", backend="lm-eval", device="cpu")
    assert isinstance(judge, LMEvalJudge)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="evalplus sandbox requires the Unix 'resource' module",
)
def test_judge_factory_explicit_evalplus() -> None:
    judge = JudgeFactory.create("humaneval_plus", backend="evalplus")
    assert isinstance(judge, EvalPlusJudge)


def test_judge_factory_unknown_task() -> None:
    with pytest.raises(ValueError):
        JudgeFactory.create("unknown_task", backend="auto")


def test_judge_factory_backend_mismatch() -> None:
    with pytest.raises(ValueError):
        JudgeFactory.create("gsm8k", backend="evalplus")
    with pytest.raises(ValueError):
        JudgeFactory.create("humaneval_plus", backend="lm-eval")
