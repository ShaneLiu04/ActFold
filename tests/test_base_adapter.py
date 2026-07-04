"""Tests for BaseEvalAdapter shared behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from actfold.eval.base_adapter import BaseEvalAdapter
from actfold.speculative import FastDLLMAdapter
from actfold.speculative.branch import Branch


class TinyTransformer(nn.Module):
    """Tiny transformer for base adapter tests."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, hidden_dim // 64),
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class DummyAdapter(BaseEvalAdapter):
    """Concrete adapter for testing shared methods."""

    TASKS = ["dummy"]

    def evaluate(
        self,
        task: str,
        num_samples: int = 10,
        limit: int | float | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        self._validate_task(task)
        prompts, references = self.judge.get_prompts(task, limit=limit or num_samples)
        baseline_predictions, _ = self._generate_predictions(prompts, use_actfold=False, seed=seed)
        baseline_score = self.judge.score(task, baseline_predictions, references)
        actfold_predictions, actfold_ratios = self._generate_predictions(
            prompts, use_actfold=True, seed=seed
        )
        actfold_score = self.judge.score(task, actfold_predictions, references)
        return {
            "task": task,
            "baseline_accuracy": baseline_score.get("accuracy", 0.0),
            "actfold_accuracy": actfold_score.get("accuracy", 0.0),
            "baseline_tflops": self._estimate_baseline_tflops(prompts),
            "actfold_tflops": self._estimate_actfold_tflops(prompts, actfold_ratios),
        }


def _make_adapter(vocab_size: int = 16, hidden_dim: int = 32, num_layers: int = 2) -> DummyAdapter:
    model = FastDLLMAdapter(
        TinyTransformer(vocab_size, hidden_dim, num_layers),
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    tokenizer.decode = MagicMock(return_value="answer")

    judge = MagicMock()
    judge.get_prompts = MagicMock(return_value=(["p"], ["r"]))
    judge.score = MagicMock(return_value={"accuracy": 1.0})

    baseline = MagicMock()
    baseline.draft_generator.generate = MagicMock(
        return_value=[Branch(branch_id="c1", parent_id="root", tokens=torch.tensor([[1, 2, 3, 4]]))]
    )
    baseline.verify = MagicMock(
        return_value=Branch(branch_id="c1", parent_id="root", tokens=torch.tensor([[1, 2, 3, 4]]))
    )

    engine = MagicMock()
    result = MagicMock()
    result.child_branch.tokens = torch.tensor([[1, 2, 3, 4]])
    result.stable_ratio = 0.5
    engine.verify_branch = MagicMock(return_value=result)

    return DummyAdapter(
        model=model,
        baseline=baseline,
        engine=engine,
        judge=judge,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
    )


def test_generate_one_actfold_uses_stable_ratio() -> None:
    """_generate_one returns decoded text and a stable ratio when use_actfold=True."""
    adapter = _make_adapter()
    text, ratio = adapter._generate_one("hello", use_actfold=True, seed=0)
    assert isinstance(text, str)
    assert 0.0 <= ratio <= 1.0


def test_generate_one_baseline_zero_ratio() -> None:
    """Baseline path returns a stable ratio of 0.0."""
    adapter = _make_adapter()
    _, ratio = adapter._generate_one("hello", use_actfold=False, seed=0)
    assert ratio == 0.0


def test_estimate_actfold_tflops_uses_measured_ratio() -> None:
    """_estimate_actfold_tflops uses the measured stable ratios per prompt."""
    adapter = _make_adapter()
    baseline = adapter._estimate_baseline_tflops(["p1", "p2"])
    actfold = adapter._estimate_actfold_tflops(["p1", "p2"], [0.5, 0.0])
    assert actfold < baseline


def test_evaluate_runs_baseline_and_actfold() -> None:
    """evaluate generates predictions for both modes and returns metrics."""
    adapter = _make_adapter()
    result = adapter.evaluate("dummy", num_samples=1, seed=0)
    assert result["baseline_accuracy"] == 1.0
    assert result["actfold_accuracy"] == 1.0
    assert result["actfold_tflops"] < result["baseline_tflops"]
