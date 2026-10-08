"""Shared base class for lm-eval and EvalPlus benchmark adapters."""

from __future__ import annotations

from typing import Any

import torch

from actfold.eval.generation_utils import (
    decode_tokens,
    encode_prompt,
    folded_generate,
    get_model_device,
    greedy_generate,
)
from actfold.eval.judges import Judge
from actfold.profiler.metrics_collector import MetricsCollector
from actfold.speculative import ActFoldVerificationEngine, SpiffyBaseline
from actfold.speculative.branch import Branch
from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
from actfold.utils.flops_counter import count_diffusion_llm_flops, model_ffn_flops_kwargs


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
            Defaults to the library default (256).
    """

    TASKS: list[str] = []
    _METRIC_KEY: str = ""
    _TASK_METRIC_KEYS: dict[str, str] = {}

    def __init__(
        self,
        model: FastDLLMAdapter,
        baseline: SpiffyBaseline,
        engine: ActFoldVerificationEngine,
        judge: Judge,
        tokenizer: Any | None = None,
        vocab_size: int = 1000,
        max_new_tokens: int = 256,
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

    def _metric_key(self, task: str) -> str:
        """Return the canonical judge metric key for ``task``.

        Prefers the per-task canonical metric name (e.g. ``exact_match`` for
        gsm8k) over the adapter-level fallback ``_METRIC_KEY``.
        """
        return self._TASK_METRIC_KEYS.get(task, self._METRIC_KEY)

    def _encode_prompts(self, prompts: list[str], seed: int) -> list[torch.Tensor]:
        """Tokenize every prompt exactly once.

        The returned token tensors are reused by generation and by the TFLOPs
        estimators, so a full evaluation encodes each prompt a single time.

        Args:
            prompts: Text prompts.
            seed: Base seed; prompt ``i`` is encoded with ``seed + i``.

        Returns:
            List of ``[1, seq_len]`` token tensors on the model device.
        """
        device = get_model_device(self.model)
        return [
            encode_prompt(prompt, self.tokenizer, self.vocab_size, seed + idx, device)
            for idx, prompt in enumerate(prompts)
        ]

    def _generate_predictions(
        self,
        prompts: list[str] | list[torch.Tensor],
        use_actfold: bool,
        seed: int,
    ) -> tuple[list[str], list[float], list[float]]:
        """Generate a prediction for every prompt.

        Args:
            prompts: Text prompts or pre-tokenized ``[1, seq_len]`` tensors
                (tokenize-once reuse: tensors are used as-is).
            use_actfold: Whether to run the folded/verified path.
            seed: Base seed for encoding and drafting.

        Returns:
            Tuple of (predictions, per-sample stable ratios, per-sample
            latency in ms).
        """
        if prompts and isinstance(prompts[0], torch.Tensor):
            prompt_tokens = [t for t in prompts if isinstance(t, torch.Tensor)]
        else:
            prompt_tokens = self._encode_prompts(
                [p for p in prompts if isinstance(p, str)], seed
            )
        predictions: list[str] = []
        stable_ratios: list[float] = []
        latencies_ms: list[float] = []
        for idx, tokens in enumerate(prompt_tokens):
            prediction, ratio, latency_ms = self._generate_one(
                tokens,
                use_actfold=use_actfold,
                seed=seed + idx,
            )
            predictions.append(prediction)
            stable_ratios.append(ratio)
            latencies_ms.append(latency_ms)
        return predictions, stable_ratios, latencies_ms

    def _generate_one(
        self,
        prompt_tokens: torch.Tensor,
        use_actfold: bool,
        seed: int,
    ) -> tuple[str, float, float]:
        """Generate a single completion using greedy decoding.

        For the ActFold path, a same-length child branch is also verified by
        the engine to obtain a measured stable ratio for TFLOPs estimation.
        The baseline path simply returns ``0.0`` for the stable ratio.

        Args:
            prompt_tokens: Pre-tokenized ``[1, seq_len]`` prompt tensor.
            use_actfold: Whether to run the folded/verified path.
            seed: Seed for the draft generator fallback path.

        Returns:
            Tuple of (decoded text, stable ratio, latency in ms). The latency
            is measured with :class:`MetricsCollector` around the generation
            call (CUDA events on GPU, ``perf_counter`` on CPU).
        """
        device = get_model_device(self.model)
        device_str = str(device) if isinstance(device, torch.device) else str(device)
        collector = MetricsCollector(device=device_str, seq_len=int(prompt_tokens.shape[1]))

        with collector:
            if use_actfold:
                folded_model = getattr(self.model, "folded_model", None)
                if folded_model is not None:
                    # True end-to-end folded generation: each new token is
                    # produced by a folded child forward pass.
                    gen_result = folded_generate(
                        self.model,
                        prompt_tokens,
                        self.max_new_tokens,
                        folded_model=folded_model,
                        draft_generator=self.baseline.draft_generator,
                        num_branches_per_step=1,
                        step_idx=0,
                    )
                    prediction_ids = gen_result.tokens
                    stable_ratio = gen_result.stable_ratio
                else:
                    # Fallback for adapters without a folded model: generate
                    # greedily and estimate the stable ratio from a
                    # same-length verification branch.  This preserves
                    # compatibility with existing callers.
                    prediction_ids = greedy_generate(self.model, prompt_tokens, self.max_new_tokens)
                    parent = Branch(branch_id="root", parent_id=None, tokens=prompt_tokens)
                    children = self.baseline.draft_generator.generate(
                        parent,
                        num_branches=1,
                        max_new_tokens=0,
                        seed=seed,
                    )
                    result = self.engine.verify_branch(parent, children[0], step_idx=0)
                    stable_ratio = result.stable_ratio
            else:
                prediction_ids = greedy_generate(self.model, prompt_tokens, self.max_new_tokens)
                stable_ratio = 0.0

        latency_ms = float(collector.metrics.latency_ms)
        return decode_tokens(prediction_ids[0], self.tokenizer), stable_ratio, latency_ms

    def _estimate_baseline_tflops(self, prompt_tokens: list[torch.Tensor]) -> float:
        """Estimate total baseline TFLOPs from pre-tokenized prompt lengths."""
        total = 0.0
        for tokens in prompt_tokens:
            seq_len = tokens.shape[1] + self.max_new_tokens
            total += count_diffusion_llm_flops(
                num_layers=self.model.num_layers,
                hidden_dim=self.model.hidden_dim,
                num_heads=max(1, self.model.num_heads),
                seq_len=max(1, seq_len),
                vocab_size=self.vocab_size,
                num_steps=1,
                reuse_ratio=0.0,
                **model_ffn_flops_kwargs(self.model),
            ).total_tflops
        return total

    def _estimate_actfold_tflops(
        self,
        prompt_tokens: list[torch.Tensor],
        stable_ratios: list[float],
    ) -> float:
        """Estimate total ActFold TFLOPs using measured stable ratios."""
        total = 0.0
        for tokens, ratio in zip(prompt_tokens, stable_ratios):
            seq_len = tokens.shape[1] + self.max_new_tokens
            total += count_diffusion_llm_flops(
                num_layers=self.model.num_layers,
                hidden_dim=self.model.hidden_dim,
                num_heads=max(1, self.model.num_heads),
                seq_len=max(1, seq_len),
                vocab_size=self.vocab_size,
                num_steps=1,
                reuse_ratio=float(ratio),
                **model_ffn_flops_kwargs(self.model),
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

        # Tokenize each prompt exactly once; the token tensors are shared by
        # generation and by the TFLOPs estimators.
        encoded_prompts = self._encode_prompts(prompts, seed)

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        baseline_predictions, _, baseline_latencies = self._generate_predictions(
            encoded_prompts,
            use_actfold=False,
            seed=seed,
        )
        baseline_score = self.judge.score(task, baseline_predictions, references)

        actfold_predictions, actfold_ratios, actfold_latencies = self._generate_predictions(
            encoded_prompts,
            use_actfold=True,
            seed=seed,
        )
        actfold_score = self.judge.score(task, actfold_predictions, references)

        baseline_tflops = self._estimate_baseline_tflops(encoded_prompts)
        actfold_tflops = self._estimate_actfold_tflops(encoded_prompts, actfold_ratios)
        mean_stable_ratio = _mean(actfold_ratios)

        metric = self._metric_key(task)

        def _score_value(score: dict[str, Any]) -> float:
            value: float = score.get("metrics", {}).get(metric, score.get(metric, 0.0))
            return float(value)

        return {
            "task": task,
            item_key: len(prompts),
            f"baseline_{metric}": _score_value(baseline_score),
            f"actfold_{metric}": _score_value(actfold_score),
            "baseline_tflops": baseline_tflops,
            "actfold_tflops": actfold_tflops,
            "mean_stable_ratio": mean_stable_ratio,
            "baseline_latency_ms": _mean(baseline_latencies),
            "actfold_latency_ms": _mean(actfold_latencies),
        }
