"""Adapter for lm-eval-harness style benchmarks."""

from __future__ import annotations

from typing import Any

from actfold.eval.base_adapter import BaseEvalAdapter
from actfold.eval.judges import Judge
from actfold.speculative import ActFoldVerificationEngine, SpiffyBaseline
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter


class LMEvalAdapter(BaseEvalAdapter):
    """Evaluate a model on GSM8K, MATH, and IFEval tasks.

    The adapter uses a pluggable :class:`~actfold.eval.judges.Judge` to load
    task prompts, generate answers with the model, and compute metrics through
    the real ``lm-eval`` backend.

    Args:
        model: Model adapter.
        baseline: Vanilla speculative decoding baseline.
        engine: ActFold verification engine.
        judge: Real lm-eval judge.
        tokenizer: Tokenizer for encoding prompts and decoding completions.
        vocab_size: Vocabulary size for TFLOPs estimation.
        max_new_tokens: Number of new tokens to generate for each prompt.
    """

    TASKS = ["gsm8k", "math", "ifeval"]
    _METRIC_KEY = "accuracy"

    def __init__(
        self,
        model: FastDLLMAdapter,
        baseline: SpiffyBaseline,
        engine: ActFoldVerificationEngine,
        judge: Judge,
        tokenizer: Any | None = None,
        vocab_size: int = 1000,
        max_new_tokens: int = 16,
    ) -> None:
        super().__init__(
            model=model,
            baseline=baseline,
            engine=engine,
            judge=judge,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
        )

    def evaluate(
        self,
        task: str,
        num_samples: int = 10,
        limit: int | float | None = None,
        seed: int = 42,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate on a task.

        Args:
            task: Task name ("gsm8k", "math", "ifeval").
            num_samples: Number of examples. Used as ``limit`` when no explicit
                limit is provided by the judge or config.
            limit: Maximum number of examples. Overrides ``num_samples``.
            seed: Random seed.
            max_new_tokens: If given, overrides the adapter's default number of
                tokens to generate.

        Returns:
            Dictionary with accuracy, latency, and TFLOPs metrics.
        """
        self._validate_task(task)
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        eval_limit = limit if limit is not None else num_samples
        return self._evaluate(task, limit=eval_limit, seed=seed, item_key="num_samples")
