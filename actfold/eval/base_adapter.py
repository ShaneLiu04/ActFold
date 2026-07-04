"""Shared base class for lm-eval and EvalPlus benchmark adapters."""

from __future__ import annotations

from typing import Any

import torch

from actfold.eval.generation_utils import (
    decode_tokens,
    encode_prompt,
    get_model_device,
    greedy_generate,
)
from actfold.eval.judges import Judge
from actfold.speculative import ActFoldVerificationEngine, SpiffyBaseline
from actfold.speculative.branch import Branch
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
from actfold.utils.flops_counter import count_diffusion_llm_flops


class BaseEvalAdapter:
    """Common adapter logic for text-generation benchmarks.

    Subclasses define ``TASKS`` and the metric keys returned by the judge.

    Args:
        model: Model adapter.
        baseline: Vanilla speculative decoding baseline.
        engine: ActFold verification engine.
        judge: Real evaluation judge.
        tokenizer: Tokenizer for encoding prompts and decoding completions.
        vocab_size: Vocabulary size for TFLOPs estimation.
        max_new_tokens: Number of new tokens to generate for each prompt.
    """

    TASKS: list[str] = []
    _METRIC_KEY: str = ""

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
        self.model = model
        self.baseline = baseline
        self.engine = engine
        self.judge = judge
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_new_tokens = max_new_tokens

    def _validate_task(self, task: str) -> None:
        """Raise if ``task`` is not supported by this adapter."""
        if task not in self.TASKS:
            raise ValueError(f"Unsupported task: {task}. Choose from {self.TASKS}")

    def _generate_predictions(
        self,
        prompts: list[str],
        use_actfold: bool,
        seed: int,
    ) -> tuple[list[str], list[float]]:
        """Generate a prediction for every prompt.

        Returns:
            Tuple of (predictions, per-sample stable ratios).
        """
        predictions: list[str] = []
        stable_ratios: list[float] = []
        for idx, prompt in enumerate(prompts):
            prediction, ratio = self._generate_one(
                prompt,
                use_actfold=use_actfold,
                seed=seed + idx,
            )
            predictions.append(prediction)
            stable_ratios.append(ratio)
        return predictions, stable_ratios

    def _generate_one(
        self,
        prompt: str,
        use_actfold: bool,
        seed: int,
    ) -> tuple[str, float]:
        """Generate a single completion using greedy decoding.

        For the ActFold path, a same-length child branch is also verified by
        the engine to obtain a measured stable ratio for TFLOPs estimation.
        The baseline path simply returns ``0.0`` for the stable ratio.

        Returns:
            Tuple of (decoded text, stable ratio).
        """
        device = get_model_device(self.model)
        prompt_tokens = encode_prompt(
            prompt,
            self.tokenizer,
            self.vocab_size,
            seed,
            device,
        )

        prediction_ids = greedy_generate(self.model, prompt_tokens, self.max_new_tokens)

        if use_actfold:
            parent = Branch(branch_id="root", parent_id=None, tokens=prompt_tokens)
            # Use a same-length child for the stable-ratio estimate. Variable-
            # length branch folding is not yet supported by the generic cache.
            children = self.baseline.draft_generator.generate(
                parent,
                num_branches=1,
                max_new_tokens=0,
                seed=seed,
            )
            result = self.engine.verify_branch(parent, children[0], step_idx=0)
            stable_ratio = result.stable_ratio
        else:
            stable_ratio = 0.0

        return decode_tokens(prediction_ids[0], self.tokenizer), stable_ratio

    def _estimate_baseline_tflops(self, prompts: list[str]) -> float:
        """Estimate total baseline TFLOPs from actual tokenized prompt lengths."""
        total = 0.0
        for prompt in prompts:
            prompt_tokens = encode_prompt(
                prompt,
                self.tokenizer,
                self.vocab_size,
                seed=0,
                device=get_model_device(self.model),
            )
            seq_len = prompt_tokens.shape[1] + self.max_new_tokens
            total += count_diffusion_llm_flops(
                num_layers=self.model.num_layers,
                hidden_dim=self.model.hidden_dim,
                num_heads=max(1, self.model.num_heads),
                seq_len=max(1, seq_len),
                vocab_size=self.vocab_size,
                num_steps=1,
                reuse_ratio=0.0,
            ).total_tflops
        return total

    def _estimate_actfold_tflops(
        self,
        prompts: list[str],
        stable_ratios: list[float],
    ) -> float:
        """Estimate total ActFold TFLOPs using measured stable ratios."""
        total = 0.0
        for idx, (prompt, ratio) in enumerate(zip(prompts, stable_ratios)):
            prompt_tokens = encode_prompt(
                prompt,
                self.tokenizer,
                self.vocab_size,
                seed=idx,
                device=get_model_device(self.model),
            )
            seq_len = prompt_tokens.shape[1] + self.max_new_tokens
            total += count_diffusion_llm_flops(
                num_layers=self.model.num_layers,
                hidden_dim=self.model.hidden_dim,
                num_heads=max(1, self.model.num_heads),
                seq_len=max(1, seq_len),
                vocab_size=self.vocab_size,
                num_steps=1,
                reuse_ratio=float(ratio),
            ).total_tflops
        return total

    def _evaluate(
        self,
        task: str,
        limit: int | float | None,
        seed: int,
        item_key: str,
    ) -> dict[str, Any]:
        """Run the common baseline/ActFold evaluation loop for a task."""
        torch.manual_seed(seed)
        prompts, references = self.judge.get_prompts(task, limit=limit)

        baseline_predictions, _ = self._generate_predictions(
            prompts,
            use_actfold=False,
            seed=seed,
        )
        baseline_score = self.judge.score(task, baseline_predictions, references)

        actfold_predictions, actfold_ratios = self._generate_predictions(
            prompts,
            use_actfold=True,
            seed=seed,
        )
        actfold_score = self.judge.score(task, actfold_predictions, references)

        baseline_tflops = self._estimate_baseline_tflops(prompts)
        actfold_tflops = self._estimate_actfold_tflops(prompts, actfold_ratios)
        mean_stable_ratio = sum(actfold_ratios) / len(actfold_ratios) if actfold_ratios else 0.0

        metric = self._METRIC_KEY
        return {
            "task": task,
            item_key: len(prompts),
            f"baseline_{metric}": baseline_score.get(metric, 0.0),
            f"actfold_{metric}": actfold_score.get(metric, 0.0),
            "baseline_tflops": baseline_tflops,
            "actfold_tflops": actfold_tflops,
            "mean_stable_ratio": mean_stable_ratio,
        }
