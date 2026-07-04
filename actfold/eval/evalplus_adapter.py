"""Adapter for EvalPlus-style code generation benchmarks."""

from __future__ import annotations

from typing import Any

from actfold.eval.base_adapter import BaseEvalAdapter
from actfold.eval.judges import Judge
from actfold.speculative import ActFoldVerificationEngine, SpiffyBaseline
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter


class EvalPlusAdapter(BaseEvalAdapter):
    """Evaluate code generation on HumanEval+ and MBPP+ tasks.

    The adapter uses a pluggable :class:`~actfold.eval.judges.Judge` to load
    code-completion prompts, generate completions with the model, and compute
    ``pass@1`` through the real ``evalplus`` backend.

    Args:
        model: Model adapter.
        baseline: Vanilla speculative decoding baseline.
        engine: ActFold verification engine.
        judge: Real evalplus judge.
        tokenizer: Tokenizer for encoding prompts and decoding completions.
        vocab_size: Vocabulary size for TFLOPs estimation.
        max_new_tokens: Number of new tokens to generate for each prompt.
    """

    TASKS = ["humaneval_plus", "mbpp_plus"]
    _METRIC_KEY = "pass_at_1"

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
        num_problems: int = 10,
        limit: int | float | None = None,
        seed: int = 42,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate on a code generation task.

        Args:
            task: Task name ("humaneval_plus", "mbpp_plus").
            num_problems: Number of problems. Used as ``limit`` when no explicit
                limit is provided.
            limit: Maximum number of problems. Overrides ``num_problems``.
            seed: Random seed.
            max_new_tokens: If given, overrides the adapter's default number of
                tokens to generate.

        Returns:
            Dictionary with pass@1, latency, and TFLOPs metrics.
        """
        self._validate_task(task)
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        eval_limit = limit if limit is not None else num_problems
        return self._evaluate(task, limit=eval_limit, seed=seed, item_key="num_problems")
