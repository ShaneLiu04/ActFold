"""Tests for actfold.eval modules."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from actfold.core.folding_scheduler import FoldingScheduler
from actfold.eval.ablation_study import AblationStudy
from actfold.eval.benchmark_runner import BenchmarkRunner
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.models.base import DiffusionLLM
from actfold.speculative import DraftGenerator, FastDLLMAdapter
from actfold.utils.config_manager import ActFoldConfig


class MockTransformer(nn.Module):
    """Tiny transformer for adapter tests."""

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


class _DummyDiffusionLLM(DiffusionLLM):
    """Minimal real DiffusionLLM for benchmark tests."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__("dummy/real-model")
        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._embedding = nn.Embedding(vocab_size, hidden_dim)
        self.tokenizer = MagicMock()
        self.tokenizer.encode = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        self.tokenizer.decode = MagicMock(return_value="answer")

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return torch.randn(tokens.size(0), tokens.size(1), self._vocab_size)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._embedding(tokens)

    def generate(self, prompt_tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return prompt_tokens

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_heads(self) -> int:
        return max(1, self._hidden_dim // 64)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


@pytest.mark.slow
def test_benchmark_runner_real_judges() -> None:
    """BenchmarkRunner uses real lm-eval / evalplus judges with a real model."""
    config = ActFoldConfig(
        model_name_or_path="dummy/real-model",
        model_family="causal_lm",
        num_layers=2,
        hidden_dim=64,
        num_heads=4,
        seq_len=8,
        vocab_size=100,
        device="cpu",
        use_real_eval=True,
        eval_backend="auto",
        eval_limit=2,
    )
    dummy_model = _DummyDiffusionLLM(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    with patch("actfold.eval.benchmark_runner.load_model", return_value=dummy_model):
        runner = BenchmarkRunner(config)
        results = runner.run(tasks=["gsm8k"], num_samples=2)
    assert "gsm8k" in results
    assert "baseline_accuracy" in results["gsm8k"]
    assert "actfold_accuracy" in results["gsm8k"]


def test_benchmark_runner_draft_mode_passthrough() -> None:
    """BenchmarkRunner builds its DraftGenerator from ``config.draft_mode``.

    Pins the T008 config passthrough instead of a hardcoded "copy_flip".
    """
    config = ActFoldConfig(
        model_name_or_path="dummy/real-model",
        num_layers=2,
        hidden_dim=64,
        num_heads=4,
        seq_len=8,
        vocab_size=100,
        device="cpu",
        use_real_eval=True,
        eval_backend="auto",
        draft_mode="copy_flip",
    )
    model = FastDLLMAdapter(
        MockTransformer(config.vocab_size, config.hidden_dim, config.num_layers),
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
    )
    runner = BenchmarkRunner(config, model=model)
    assert runner.draft_generator.mode == "copy_flip"


def test_ablation_study() -> None:
    vocab_size = 100
    hidden_dim = 64
    num_layers = 2

    model = FastDLLMAdapter(
        MockTransformer(vocab_size, hidden_dim, num_layers),
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )
    draft_generator = DraftGenerator(vocab_size=vocab_size, mode="copy_flip", flip_ratio=0.05)
    study = AblationStudy(
        model=model,
        baseline=None,
        draft_generator=draft_generator,
        vocab_size=vocab_size,
        seq_len=8,
        device="cpu",
    )

    df_threshold = study.run_threshold_sensitivity(taus=[0.90, 0.95])
    assert len(df_threshold) == 2
    assert "tau" in df_threshold.columns

    df_layer = study.run_layerwise_folding(layer_ranges=[(0, 0), (0, num_layers - 1)])
    assert len(df_layer) == 2
    assert "estimated_reduction_pct" in df_layer.columns

    df_cache = study.run_cache_size_impact(cache_sizes=[128, 256])
    assert len(df_cache) == 2
    assert "cache_size" in df_cache.columns


def test_ablation_study_run_all_saves_results(tmp_path) -> None:
    """run_all saves artifacts when output_dir is provided."""
    vocab_size = 32
    hidden_dim = 32
    num_layers = 2

    model = FastDLLMAdapter(
        MockTransformer(vocab_size, hidden_dim, num_layers),
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )
    draft_generator = DraftGenerator(vocab_size=vocab_size, mode="copy_flip", flip_ratio=0.05)
    study = AblationStudy(
        model=model,
        baseline=None,
        draft_generator=draft_generator,
        vocab_size=vocab_size,
        seq_len=8,
        device="cpu",
    )
    summary = study.run_all(output_dir=tmp_path)
    assert "threshold_sensitivity" in summary
    assert "layerwise_folding" in summary
    assert "cache_size_impact" in summary
    assert any(tmp_path.iterdir())


class _ModelBackedDummyDiffusionLLM(_DummyDiffusionLLM):
    """_DummyDiffusionLLM exposing a detectable raw ``.model`` Transformer stack."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__(vocab_size, hidden_dim, num_layers)
        self.model = MockTransformer(vocab_size, hidden_dim, num_layers)


class _BareModelDummyDiffusionLLM(_DummyDiffusionLLM):
    """_DummyDiffusionLLM whose ``.model`` has no detectable layer stack or embedding."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__(vocab_size, hidden_dim, num_layers)
        self.model = nn.Linear(hidden_dim, hidden_dim)


def _make_runner_config(**overrides: Any) -> ActFoldConfig:
    """Build a CPU BenchmarkRunner config, applying optional keyword overrides."""
    kwargs: dict[str, Any] = {
        "model_name_or_path": "dummy/real-model",
        "num_layers": 2,
        "hidden_dim": 64,
        "num_heads": 4,
        "seq_len": 8,
        "vocab_size": 100,
        "device": "cpu",
        "use_real_eval": True,
        "eval_backend": "auto",
    }
    kwargs.update(overrides)
    return ActFoldConfig(**kwargs)


def test_benchmark_runner_builds_manual_folded_forward() -> None:
    """IT-303 (srs §3.4 acceptance 1): detectable architecture builds a
    ``ManualFoldedForward`` whose graph flags are propagated from the config."""
    config = _make_runner_config(use_cuda_graph=False, graph_capacity_ratio=0.7)
    stub = _ModelBackedDummyDiffusionLLM(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    with patch("actfold.eval.benchmark_runner.load_model", return_value=stub):
        runner = BenchmarkRunner(config)
    folded = runner.model.folded_model
    assert isinstance(folded, ManualFoldedForward)
    assert folded.use_cuda_graph is False
    assert folded.graph_capacity_ratio == pytest.approx(0.7)

    graph_config = _make_runner_config(use_cuda_graph=True, graph_capacity_ratio=0.3)
    graph_stub = _ModelBackedDummyDiffusionLLM(
        vocab_size=graph_config.vocab_size,
        hidden_dim=graph_config.hidden_dim,
        num_layers=graph_config.num_layers,
    )
    with patch("actfold.eval.benchmark_runner.load_model", return_value=graph_stub):
        graph_runner = BenchmarkRunner(graph_config)
    graph_folded = graph_runner.model.folded_model
    assert isinstance(graph_folded, ManualFoldedForward)
    assert graph_folded.use_cuda_graph is True
    assert graph_folded.graph_capacity_ratio == pytest.approx(0.3)


def test_benchmark_runner_undetectable_architecture_returns_none() -> None:
    """IT-304 (srs §3.4 acceptance 2): undetectable architecture yields
    ``folded_model is None`` without raising during runner construction."""
    config = _make_runner_config()
    stub = _BareModelDummyDiffusionLLM(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    with patch("actfold.eval.benchmark_runner.load_model", return_value=stub):
        runner = BenchmarkRunner(config)
    assert runner.model.folded_model is None


def test_benchmark_runner_scheduler_strategy() -> None:
    """IT-305 (D3 graph contract): ``use_cuda_graph=True`` builds no scheduler,
    ``use_cuda_graph=False`` keeps a ``FoldingScheduler``."""
    graph_config = _make_runner_config(use_cuda_graph=True)
    graph_stub = _ModelBackedDummyDiffusionLLM(
        vocab_size=graph_config.vocab_size,
        hidden_dim=graph_config.hidden_dim,
        num_layers=graph_config.num_layers,
    )
    with patch("actfold.eval.benchmark_runner.load_model", return_value=graph_stub):
        graph_runner = BenchmarkRunner(graph_config)
    graph_folded = graph_runner.model.folded_model
    assert graph_folded is not None
    assert graph_folded.scheduler is None

    eager_config = _make_runner_config(use_cuda_graph=False)
    eager_stub = _ModelBackedDummyDiffusionLLM(
        vocab_size=eager_config.vocab_size,
        hidden_dim=eager_config.hidden_dim,
        num_layers=eager_config.num_layers,
    )
    with patch("actfold.eval.benchmark_runner.load_model", return_value=eager_stub):
        eager_runner = BenchmarkRunner(eager_config)
    eager_folded = eager_runner.model.folded_model
    assert eager_folded is not None
    assert isinstance(eager_folded.scheduler, FoldingScheduler)


def test_benchmark_runner_raw_model_missing_returns_none() -> None:
    """IT-306: a diffusion model without a ``.model`` attribute keeps the early
    exit — ``folded_model is None``."""
    config = _make_runner_config()
    dummy_model = _DummyDiffusionLLM(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    with patch("actfold.eval.benchmark_runner.load_model", return_value=dummy_model):
        runner = BenchmarkRunner(config)
    assert runner.model.folded_model is None
