"""Tests for BaseEvalAdapter shared behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
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
        encoded = self._encode_prompts(prompts, seed)
        baseline_predictions, _, baseline_latencies = self._generate_predictions(
            encoded, use_actfold=False, seed=seed
        )
        baseline_score = self.judge.score(task, baseline_predictions, references)
        actfold_predictions, actfold_ratios, actfold_latencies = self._generate_predictions(
            encoded, use_actfold=True, seed=seed
        )
        actfold_score = self.judge.score(task, actfold_predictions, references)
        return {
            "task": task,
            "baseline_accuracy": baseline_score.get("accuracy", 0.0),
            "actfold_accuracy": actfold_score.get("accuracy", 0.0),
            "baseline_tflops": self._estimate_baseline_tflops(encoded),
            "actfold_tflops": self._estimate_actfold_tflops(encoded, actfold_ratios),
            "baseline_latency_ms": sum(baseline_latencies) / len(baseline_latencies),
            "actfold_latency_ms": sum(actfold_latencies) / len(actfold_latencies),
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
    """_generate_one returns decoded text, ratio, and latency when use_actfold=True."""
    adapter = _make_adapter()
    text, ratio, latency_ms = adapter._generate_one(
        torch.tensor([[1, 2, 3]]), use_actfold=True, seed=0
    )
    assert isinstance(text, str)
    assert 0.0 <= ratio <= 1.0
    assert latency_ms > 0.0


def test_generate_one_baseline_zero_ratio() -> None:
    """Baseline path returns a stable ratio of 0.0."""
    adapter = _make_adapter()
    _, ratio, latency_ms = adapter._generate_one(
        torch.tensor([[1, 2, 3]]), use_actfold=False, seed=0
    )
    assert ratio == 0.0
    assert latency_ms > 0.0


def test_estimate_actfold_tflops_uses_measured_ratio() -> None:
    """_estimate_actfold_tflops uses the measured stable ratios per prompt."""
    adapter = _make_adapter()
    tokens = [torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])]
    baseline = adapter._estimate_baseline_tflops(tokens)
    actfold = adapter._estimate_actfold_tflops(tokens, [0.5, 0.0])
    assert actfold < baseline


def test_evaluate_runs_baseline_and_actfold() -> None:
    """evaluate generates predictions for both modes and returns metrics."""
    adapter = _make_adapter()
    result = adapter.evaluate("dummy", num_samples=1, seed=0)
    assert result["baseline_accuracy"] == 1.0
    assert result["actfold_accuracy"] == 1.0
    assert result["actfold_tflops"] < result["baseline_tflops"]


def test_it412b_base_adapter_tflops_use_config_geometry() -> None:
    """IT-412b (AR005 design 6.2): TFLOPs estimation consumes real geometry.

    ``_estimate_baseline_tflops`` picks the SwiGLU/MoE geometry from
    ``model_ffn_flops_kwargs`` (reachable only via ``DiffusionLLM.model.config``,
    the real checkpoint path) instead of the 4h-MLP default.
    """
    from types import SimpleNamespace

    from actfold.models.base import DiffusionLLM
    from actfold.utils.flops_counter import count_diffusion_llm_flops, model_ffn_flops_kwargs

    class _GeometryOnlyModel(DiffusionLLM):
        """DiffusionLLM stub exposing dims and geometry via ``model.config``."""

        def __init__(self, config: SimpleNamespace) -> None:
            super().__init__("geometry-stub")
            self.model = SimpleNamespace(config=config)

        def forward(self, tokens, attention_mask=None, **kwargs):
            raise NotImplementedError

        def embed(self, tokens):
            raise NotImplementedError

        @property
        def num_layers(self):
            return 2

        @property
        def hidden_dim(self):
            return 16

        @property
        def num_heads(self):
            return 2

        @property
        def vocab_size(self):
            return 100

    config = SimpleNamespace(
        intermediate_size=48,
        hidden_act="silu",
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=32,
        num_hidden_layers=2,
        first_k_dense_replace=1,
    )
    model_adapter = FastDLLMAdapter(_GeometryOnlyModel(config))
    eval_adapter = BaseEvalAdapter(
        model=model_adapter,
        baseline=None,
        engine=None,
        judge=None,
        tokenizer=None,
        vocab_size=100,
        max_new_tokens=4,
    )

    got = eval_adapter._estimate_baseline_tflops([torch.tensor([[1, 2, 3]])])

    kwargs = model_ffn_flops_kwargs(model_adapter)
    expected = count_diffusion_llm_flops(
        num_layers=2,
        hidden_dim=16,
        num_heads=2,
        seq_len=3 + 4,
        vocab_size=100,
        num_steps=1,
        reuse_ratio=0.0,
        **kwargs,
    ).total_tflops
    assert got == pytest.approx(expected, rel=1e-12)

    default_total = count_diffusion_llm_flops(
        num_layers=2,
        hidden_dim=16,
        num_heads=2,
        seq_len=3 + 4,
        vocab_size=100,
        num_steps=1,
        reuse_ratio=0.0,
    ).total_tflops
    assert got != pytest.approx(default_total)
