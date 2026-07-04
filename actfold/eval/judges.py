"""Unified judge interface for lm-eval and evalplus benchmarks.

The module exposes a small abstraction layer over two popular evaluation
backends:

* `lm-eval <https://github.com/EleutherAI/lm-evaluation-harness>`_ for
  mathematical and instruction-following tasks (e.g. GSM8K, MATH, IFEval).
* `evalplus <https://github.com/evalplus/evalplus>`_ for code generation
  tasks (HumanEval+, MBPP+).

Both backends are required at runtime; there is no mock fallback.
Install them with ``pip install -r requirements-bench.txt``.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from actfold.utils.logger import get_logger


def _get_evalplus_problems(dataset: str) -> dict[str, Any]:
    """Lazily load EvalPlus problem dictionaries.

    EvalPlus is imported inside this function so that importing
    ``actfold.eval.judges`` does not require evalplus to be installed when only
    ``lm-eval`` tasks are used.
    """
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    if dataset == "humaneval":
        return dict(get_human_eval_plus())
    return dict(get_mbpp_plus())


logger = get_logger("judges")


class Judge(ABC):
    """Abstract correctness judge for a benchmark task.

    A judge is responsible for two things:

    1. Loading a task and returning its evaluation prompts / references.
    2. Scoring a list of model predictions against the references.

    This split lets the speculative-decoding pipeline generate answers with
    its own model/forward path and then hand the predictions to the judge for
    metric computation.
    """

    @abstractmethod
    def get_prompts(
        self,
        task: str,
        limit: int | float | None = None,
    ) -> tuple[list[str], list[Any]]:
        """Return (prompts, references) for ``task``.

        Args:
            task: Canonical task name, e.g. ``"gsm8k"``.
            limit: Maximum number of examples. An ``int`` limits to the first
                N examples; a ``float`` in ``[0, 1]`` limits to a fraction.

        Returns:
            Tuple of prompt strings and reference objects.
        """
        ...

    @abstractmethod
    def score(
        self,
        task: str,
        predictions: list[str],
        references: list[Any],
    ) -> dict[str, Any]:
        """Score ``predictions`` against ``references``.

        Args:
            task: Task name.
            predictions: Model-generated answers in the same order as
                ``get_prompts`` returned the prompts.
            references: Reference objects returned by ``get_prompts``.

        Returns:
            Dictionary of metrics. Must contain a primary metric key such as
            ``"accuracy"`` or ``"pass_at_1"``.
        """
        ...

    def evaluate(
        self,
        task: str,
        predictions: list[str],
        references: list[Any] | None = None,
        limit: int | float | None = None,
    ) -> dict[str, Any]:
        """Convenience wrapper around :meth:`score`.

        If ``references`` is not provided, the judge loads them from the task.
        """
        if references is None:
            _, references = self.get_prompts(task, limit=limit)
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references "
                f"({len(references)}) must have the same length"
            )
        return self.score(task, predictions, references)


def _lm_eval_primary_metric(result: dict[str, Any]) -> float:
    """Extract the primary scalar metric from an lm-eval per-doc result dict.

    Looks for the most common primary metrics used by the supported tasks and
    returns the first one found as a float.
    """
    for metric in ("exact_match", "acc", "prompt_level_acc"):
        if metric in result:
            value = result[metric]
            if isinstance(value, (int, float)):
                return float(value)
    # Fallback: return the first numeric value found.
    for value in result.values():
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _limit_iterator(items: list[Any], limit: int | float | None) -> list[Any]:
    """Apply an lm-eval-style ``limit`` to a list."""
    if limit is None:
        return items
    if isinstance(limit, float) and 0.0 <= limit <= 1.0:
        n = int(limit * len(items))
    else:
        n = int(limit)
    return items[: max(0, n)]


class LMEvalJudge(Judge):
    """Judge for lm-eval-harness tasks.

    Supports ``gsm8k``, ``math``, and ``ifeval`` out of the box. Additional
    generative tasks can be used as long as they expose the standard lm-eval
    ``Task`` interface.

    Args:
        device: Device passed to lm-eval when it loads a model. Not used when
            only prompts/references are requested.
        batch_size: Batch size for lm-eval model evaluation (unused when only
            scoring pre-generated predictions).
        num_fewshot: Number of few-shot examples. ``None`` uses task default.
    """

    SUPPORTED_TASKS = {"gsm8k", "math", "ifeval"}

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int | str = 1,
        num_fewshot: int | None = None,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.num_fewshot = num_fewshot

    def get_prompts(
        self,
        task: str,
        limit: int | float | None = None,
    ) -> tuple[list[str], list[Any]]:
        """Return prompts and references for ``task``."""
        from lm_eval.tasks import TaskManager

        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported lm-eval task: {task}")
        task_manager = TaskManager()
        loaded = task_manager.load([task])
        task_obj = loaded["tasks"][task]

        docs: list[Any] = list(task_obj.eval_docs)
        docs = _limit_iterator(docs, limit)

        num_fewshot = self.num_fewshot
        if num_fewshot is None:
            num_fewshot = int(task_obj.config.num_fewshot)

        prompts: list[str] = []
        references: list[Any] = []
        for doc in docs:
            ctx = task_obj.fewshot_context(doc=doc, num_fewshot=num_fewshot)
            prompts.append(ctx + task_obj.doc_to_text(doc))
            references.append(doc)
        return prompts, references

    def score(
        self,
        task: str,
        predictions: list[str],
        references: list[Any],
    ) -> dict[str, Any]:
        """Score pre-generated predictions using the task's own metrics."""
        from lm_eval.tasks import TaskManager

        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported lm-eval task: {task}")
        task_manager = TaskManager()
        loaded = task_manager.load([task])
        task_obj = loaded["tasks"][task]

        total = 0.0
        details: list[dict[str, Any]] = []
        for prediction, reference in zip(predictions, references):
            # lm-eval expects ``results`` as a list (one entry per completion).
            result = task_obj.process_results(reference, [prediction])
            score_value = _lm_eval_primary_metric(result)
            total += score_value
            details.append({"score": score_value, "metrics": result})

        accuracy = total / len(predictions) if predictions else 0.0
        return {
            "task": task,
            "num_samples": len(predictions),
            "accuracy": accuracy,
            "details": details,
        }


class EvalPlusJudge(Judge):
    """Judge for EvalPlus code-generation tasks.

    Supports ``humaneval_plus`` and ``mbpp_plus``. EvalPlus evaluates code
    samples by executing them against the official test suites. This judge
    therefore needs executable Python completions.

    Args:
        dataset: Either ``"humaneval"`` or ``"mbpp"``.
        base_only: If ``True``, use only the original tests (not the extra
            EvalPlus tests).
        parallel: Number of parallel workers for test execution.
        min_time_limit: Minimum per-test timeout in seconds.
    """

    TASK_TO_DATASET = {
        "humaneval_plus": "humaneval",
        "mbpp_plus": "mbpp",
    }

    def __init__(
        self,
        base_only: bool = False,
        parallel: int | None = None,
        min_time_limit: float = 1.0,
    ) -> None:
        self.base_only = base_only
        self.parallel = parallel or max(1, (os.cpu_count() or 1) // 2)
        self.min_time_limit = min_time_limit

    def get_prompts(
        self,
        task: str,
        limit: int | float | None = None,
    ) -> tuple[list[str], list[Any]]:
        """Return code-completion prompts and problem dicts."""
        dataset = self._dataset(task)
        problems = _get_evalplus_problems(dataset)

        items = list(problems.items())
        items = _limit_iterator(items, limit)
        prompts = [problem["prompt"] for _, problem in items]
        references = [problem for _, problem in items]
        return prompts, references

    def score(
        self,
        task: str,
        predictions: list[str],
        references: list[Any],
    ) -> dict[str, Any]:
        """Score code predictions with EvalPlus.

        Writes the predictions to a temporary ``samples.jsonl`` file and
        invokes EvalPlus. The primary returned metric is ``pass_at_1``.
        """
        if not predictions:
            return {"task": task, "num_samples": 0, "pass_at_1": 0.0}

        dataset = self._dataset(task)
        samples = [
            {"task_id": ref.get("task_id", f"{dataset}/{idx}"), "completion": pred}
            for idx, (pred, ref) in enumerate(zip(predictions, references))
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples.jsonl"
            with samples_path.open("w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            result = self._run_evalplus(dataset, str(samples_path))

        key = "base" if self.base_only else "base + extra"
        pass_at_1 = result.get(key, {}).get("pass@1", 0.0)
        return {
            "task": task,
            "num_samples": len(predictions),
            "pass_at_1": pass_at_1,
            "evalplus_result": result,
        }

    def _dataset(self, task: str) -> str:
        if task not in self.TASK_TO_DATASET:
            raise ValueError(f"Unsupported evalplus task: {task}")
        return self.TASK_TO_DATASET[task]

    def _run_evalplus(self, dataset: str, samples_path: str) -> dict[str, Any]:
        """Invoke evalplus via its Python module or CLI fallback."""
        import platform

        # Try the evalplus Python API first.
        try:
            from evalplus.evaluate import evaluate

            raw: Any = evaluate(
                dataset=dataset,
                samples=samples_path,
                base_only=self.base_only,
                parallel=self.parallel,
                min_time_limit=self.min_time_limit,
            )
            if not isinstance(raw, dict):
                raise RuntimeError(f"Unexpected evalplus result type: {type(raw)}")
            return raw
        except Exception as exc:
            logger.debug("evalplus Python API failed: %s", exc)
            # Fallback to CLI on Unix-like systems where the entry point is available.
            if platform.system() == "Windows":
                raise RuntimeError(
                    "evalplus evaluation failed on Windows. The evalplus sandbox "
                    "requires Unix-like platform support (the 'resource' module). "
                    f"Original error: {exc}"
                ) from exc

            cmd = [
                "python",
                "-m",
                "evalplus.evaluate",
                "--dataset",
                dataset,
                "--samples",
                samples_path,
                "--parallel",
                str(self.parallel),
            ]
            if self.base_only:
                cmd.append("--base-only")
            output = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if output.returncode != 0:
                raise RuntimeError(
                    f"evalplus CLI failed (code {output.returncode}): {output.stderr}"
                )
            return self._parse_evalplus_output(output.stdout)

    @staticmethod
    def _parse_evalplus_output(stdout: str) -> dict[str, Any]:
        """Parse the evalplus CLI text output into a dict."""
        result: dict[str, Any] = {}
        current_key: str | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if line in {"Base", "Base + Extra"}:
                current_key = "base" if line == "Base" else "base + extra"
            elif line.startswith("{'pass@1':") and current_key is not None:
                try:
                    parsed: dict[str, Any] = ast.literal_eval(line)
                    result[current_key] = parsed
                except (ValueError, SyntaxError):
                    result[current_key] = {"raw": line}
        return result


class JudgeFactory:
    """Create real judges based on task and backend configuration."""

    @staticmethod
    def create(
        task: str,
        backend: str = "auto",
        device: str = "cuda",
        batch_size: int | str = 1,
        num_fewshot: int | None = None,
        base_only: bool = False,
    ) -> Judge:
        """Return a judge for ``task``.

        Args:
            task: Task name, e.g. ``"gsm8k"`` or ``"humaneval_plus"``.
            backend: ``"auto"``, ``"lm-eval"``, or ``"evalplus"``.
            device: Device string passed to lm-eval.
            batch_size: Batch size passed to lm-eval.
            num_fewshot: Few-shot count for lm-eval tasks.
            base_only: Use base-only tests for evalplus.

        Returns:
            A :class:`Judge` instance.

        Raises:
            ValueError: If the task is unsupported or the backend mismatches.
        """
        is_lm_task = task in LMEvalJudge.SUPPORTED_TASKS
        is_code_task = task in EvalPlusJudge.TASK_TO_DATASET

        if not is_lm_task and not is_code_task:
            raise ValueError(f"Unsupported task: {task}")

        if backend == "auto":
            if is_lm_task:
                return LMEvalJudge(
                    device=device,
                    batch_size=batch_size,
                    num_fewshot=num_fewshot,
                )
            return EvalPlusJudge(base_only=base_only)

        if backend == "lm-eval":
            if not is_lm_task:
                raise ValueError(f"Task '{task}' is not an lm-eval task")
            return LMEvalJudge(
                device=device,
                batch_size=batch_size,
                num_fewshot=num_fewshot,
            )

        if backend == "evalplus":
            if not is_code_task:
                raise ValueError(f"Task '{task}' is not an evalplus task")
            return EvalPlusJudge(base_only=base_only)

        raise ValueError(f"Unknown backend: {backend}")
