"""Red tests for T023: eval fixes.

Covers the green contract for:
- ``ActFoldConfig.max_new_tokens`` (benchmark generation length).
- ``BaseEvalAdapter`` tokenize-once discipline, latency reporting, and
  canonical per-task metric keys.
- ``judges._aggregate_metric_means`` pure aggregation helper.
- ``BenchmarkRunner`` passing ``max_new_tokens`` to both eval adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from actfold.eval.base_adapter import BaseEvalAdapter
from actfold.eval.generation_utils import encode_prompt as real_encode_prompt
from actfold.eval.generation_utils import get_model_device
from actfold.speculative import FastDLLMAdapter
from actfold.speculative.branch import Branch
from actfold.utils.config_manager import ActFoldConfig


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


class EvalFixAdapter(BaseEvalAdapter):
    """Concrete adapter for testing shared eval behavior."""

    TASKS = ["dummy"]
    _METRIC_KEY = "accuracy"

    def evaluate(
        self,
        task: str,
        num_samples: int = 10,
        limit: int | float | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        raise NotImplementedError("Not needed for these tests; _evaluate is called directly.")


class MetricKeyAdapter(EvalFixAdapter):
    """Adapter with a task -> metric key mapping."""

    TASKS = ["gsm8k", "dummy"]
    _METRIC_KEY = "accuracy"
    _TASK_METRIC_KEYS = {"gsm8k": "exact_match"}


def _make_adapter(
    adapter_cls: type[BaseEvalAdapter] = EvalFixAdapter,
    score_value: dict[str, Any] | None = None,
) -> BaseEvalAdapter:
    """Build a minimal concrete adapter with mocked baseline/engine/judge."""
    model = FastDLLMAdapter(
        TinyTransformer(16, 32, 2),
        num_layers=2,
        hidden_dim=32,
    )
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    tokenizer.decode = MagicMock(return_value="answer")

    judge = MagicMock()
    judge.get_prompts = MagicMock(return_value=(["p1", "p2"], ["r1", "r2"]))
    judge.score = MagicMock(return_value=score_value or {"accuracy": 1.0})

    baseline = MagicMock()
    baseline.draft_generator.generate = MagicMock(
        return_value=[Branch(branch_id="c1", parent_id="root", tokens=torch.tensor([[1, 2, 3, 4]]))]
    )

    engine = MagicMock()
    result = MagicMock()
    result.child_branch.tokens = torch.tensor([[1, 2, 3, 4]])
    result.stable_ratio = 0.5
    engine.verify_branch = MagicMock(return_value=result)

    return adapter_cls(
        model=model,
        baseline=baseline,
        engine=engine,
        judge=judge,
        tokenizer=tokenizer,
        vocab_size=16,
    )


# ---------------------------------------------------------------------------
# A. ActFoldConfig.max_new_tokens
# ---------------------------------------------------------------------------


def test_config_max_new_tokens_default_is_256() -> None:
    config = ActFoldConfig()
    assert config.max_new_tokens == 256


def test_config_max_new_tokens_zero_raises_value_error() -> None:
    with pytest.raises(ValueError):
        ActFoldConfig(max_new_tokens=0)


def test_config_max_new_tokens_negative_raises_value_error() -> None:
    with pytest.raises(ValueError):
        ActFoldConfig(max_new_tokens=-5)


def test_config_max_new_tokens_positive_accepted() -> None:
    config = ActFoldConfig(max_new_tokens=64)
    assert config.max_new_tokens == 64


# ---------------------------------------------------------------------------
# B. BaseEvalAdapter
# ---------------------------------------------------------------------------


def test_adapter_default_max_new_tokens_is_256() -> None:
    adapter = _make_adapter()
    assert adapter.max_new_tokens == 256


def test_encode_prompts_encodes_each_prompt_once_with_per_index_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    calls: list[tuple[str, int, torch.device]] = []

    def counting_encode(
        prompt: str,
        tokenizer: Any,
        vocab_size: int,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        calls.append((prompt, seed, device))
        return real_encode_prompt(prompt, tokenizer, vocab_size, seed, device)

    monkeypatch.setattr("actfold.eval.base_adapter.encode_prompt", counting_encode)

    tokens = adapter._encode_prompts(["a", "b"], seed=5)

    assert isinstance(tokens, list)
    assert len(tokens) == 2
    assert all(isinstance(t, torch.Tensor) for t in tokens)
    assert len(calls) == 2
    assert calls[0][0] == "a"
    assert calls[1][0] == "b"
    assert [seed for _, seed, _ in calls] == [5, 6]
    expected_device = get_model_device(adapter.model)
    assert calls[0][2] == expected_device
    assert calls[1][2] == expected_device


def test_generate_one_actfold_returns_text_ratio_latency() -> None:
    adapter = _make_adapter()
    result = adapter._generate_one(torch.tensor([[1, 2, 3]]), use_actfold=True, seed=0)
    assert isinstance(result, tuple)
    assert len(result) == 3
    text, ratio, latency_ms = result
    assert isinstance(text, str)
    assert 0.0 <= ratio <= 1.0
    assert latency_ms > 0.0


def test_generate_one_baseline_returns_text_ratio_latency() -> None:
    adapter = _make_adapter()
    text, ratio, latency_ms = adapter._generate_one(
        torch.tensor([[1, 2, 3]]), use_actfold=False, seed=0
    )
    assert isinstance(text, str)
    assert ratio == 0.0
    assert latency_ms > 0.0


def test_generate_predictions_returns_three_equal_length_lists() -> None:
    adapter = _make_adapter()
    result = adapter._generate_predictions(["p1", "p2"], use_actfold=False, seed=0)
    assert isinstance(result, tuple)
    assert len(result) == 3
    predictions, ratios, latencies = result
    assert len(predictions) == 2
    assert len(ratios) == 2
    assert len(latencies) == 2
    assert all(isinstance(x, str) for x in predictions)
    assert all(isinstance(x, float) and x > 0.0 for x in latencies)


def test_estimate_tflops_accept_pretokenized_tensors_without_re_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    calls: list[Any] = []

    def counting_encode(
        prompt: str,
        tokenizer: Any,
        vocab_size: int,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        calls.append(prompt)
        return real_encode_prompt(prompt, tokenizer, vocab_size, seed, device)

    monkeypatch.setattr("actfold.eval.base_adapter.encode_prompt", counting_encode)

    prompt_tokens = [torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])]
    baseline = adapter._estimate_baseline_tflops(prompt_tokens)
    actfold = adapter._estimate_actfold_tflops(prompt_tokens, [0.5, 0.5])

    assert calls == []
    assert actfold < baseline


def test_evaluate_encodes_prompts_exactly_once_per_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    calls: list[str] = []

    def counting_encode(
        prompt: str,
        tokenizer: Any,
        vocab_size: int,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        calls.append(prompt)
        return real_encode_prompt(prompt, tokenizer, vocab_size, seed, device)

    monkeypatch.setattr("actfold.eval.base_adapter.encode_prompt", counting_encode)

    adapter._evaluate("dummy", limit=None, seed=0, item_key="num_samples")

    # Two prompts: encode_prompt must run exactly len(prompts) times total
    # (generation + TFLOPs estimates reuse the tokenized tensors).
    assert len(calls) == 2


def test_evaluate_reports_mean_latency_keys() -> None:
    adapter = _make_adapter()
    result = adapter._evaluate("dummy", limit=None, seed=0, item_key="num_samples")
    assert "baseline_latency_ms" in result
    assert "actfold_latency_ms" in result
    assert isinstance(result["baseline_latency_ms"], float)
    assert isinstance(result["actfold_latency_ms"], float)
    assert result["baseline_latency_ms"] > 0.0
    assert result["actfold_latency_ms"] > 0.0


def test_metric_key_prefers_task_mapping_over_default() -> None:
    adapter = _make_adapter(MetricKeyAdapter)
    assert BaseEvalAdapter._TASK_METRIC_KEYS == {}
    assert MetricKeyAdapter._TASK_METRIC_KEYS == {"gsm8k": "exact_match"}
    assert adapter._metric_key("gsm8k") == "exact_match"
    assert adapter._metric_key("dummy") == "accuracy"
    plain = _make_adapter(EvalFixAdapter)
    assert plain._metric_key("dummy") == "accuracy"


def test_evaluate_uses_canonical_metric_keys_for_mapped_task() -> None:
    adapter = _make_adapter(
        MetricKeyAdapter,
        score_value={"accuracy": 0.5, "metrics": {"exact_match": 0.7}},
    )
    result = adapter._evaluate("gsm8k", limit=None, seed=0, item_key="num_samples")
    assert result["baseline_exact_match"] == 0.7
    assert result["actfold_exact_match"] == 0.7
    assert "baseline_accuracy" not in result
    assert "actfold_accuracy" not in result


def test_evaluate_unmapped_task_uses_default_metric_key() -> None:
    adapter = _make_adapter(
        MetricKeyAdapter,
        score_value={"accuracy": 0.5, "metrics": {"exact_match": 0.7}},
    )
    assert adapter._metric_key("dummy") == "accuracy"
    result = adapter._evaluate("dummy", limit=None, seed=0, item_key="num_samples")
    assert result["baseline_accuracy"] == 0.5
    assert result["actfold_accuracy"] == 0.5
    assert "baseline_exact_match" not in result
    assert "actfold_exact_match" not in result


def test_lm_eval_adapter_task_metric_keys() -> None:
    from actfold.eval.lm_eval_adapter import LMEvalAdapter

    assert LMEvalAdapter._TASK_METRIC_KEYS == {
        "gsm8k": "exact_match",
        "math": "exact_match",
        "ifeval": "prompt_level_acc",
    }


# ---------------------------------------------------------------------------
# C. judges._aggregate_metric_means
# ---------------------------------------------------------------------------


def test_aggregate_metric_means_uniform_metric() -> None:
    from actfold.eval.judges import _aggregate_metric_means

    details = [{"metrics": {"exact_match": 1.0}}, {"metrics": {"exact_match": 0.0}}]
    assert _aggregate_metric_means(details) == {"exact_match": 0.5}


def test_aggregate_metric_means_mixed_metric_names() -> None:
    from actfold.eval.judges import _aggregate_metric_means

    details = [
        {"metrics": {"exact_match": 1.0}},
        {"metrics": {"exact_match": 0.0, "acc": 1.0}},
        {"metrics": {"acc": 0.0}},
    ]
    result = _aggregate_metric_means(details)
    assert result["exact_match"] == pytest.approx(0.5)
    assert result["acc"] == pytest.approx(0.5)


def test_aggregate_metric_means_empty_details() -> None:
    from actfold.eval.judges import _aggregate_metric_means

    assert _aggregate_metric_means([]) == {}


# ---------------------------------------------------------------------------
# D. BenchmarkRunner max_new_tokens passthrough (source scan)
# ---------------------------------------------------------------------------


def test_benchmark_runner_passes_max_new_tokens_to_both_adapters() -> None:
    import actfold.eval.benchmark_runner as benchmark_runner_mod

    source = Path(benchmark_runner_mod.__file__).read_text(encoding="utf-8")
    kwarg = "max_new_tokens=self.config.max_new_tokens"
    assert source.count(kwarg) >= 2

    for ctor in ("LMEvalAdapter(", "EvalPlusAdapter("):
        start = source.index(ctor)
        end = source.index(")", start)
        block = source[start:end]
        assert kwarg in block, f"{ctor} construction must pass {kwarg}"
