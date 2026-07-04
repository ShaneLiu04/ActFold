"""Config-driven benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from actfold.core import ActivationCache, FoldedModel, SimilarityGate
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.eval.evalplus_adapter import EvalPlusAdapter
from actfold.eval.judges import Judge, JudgeFactory
from actfold.eval.lm_eval_adapter import LMEvalAdapter
from actfold.models import load_model
from actfold.models.utils import get_model_device, resolve_torch_dtype
from actfold.speculative import (
    ActFoldVerificationEngine,
    DraftGenerator,
    FastDLLMAdapter,
    SpiffyBaseline,
)
from actfold.utils.config_manager import ActFoldConfig
from actfold.utils.logger import get_logger


class BenchmarkRunner:
    """Execute benchmarks based on an ActFoldConfig.

    The runner loads a real Diffusion LLM from the Hugging Face Hub or a local
    path according to ``config.model_name_or_path``.  Evaluation uses real
    ``lm-eval`` / ``evalplus`` backends. When the underlying model exposes a
    recognizable Transformer layer stack, it is wrapped with :class:`FoldedModel`
    so that the ActFold path reuses real parent activations.

    Args:
        config: Experiment configuration.  Must provide ``model_name_or_path``
            unless ``model`` is passed explicitly.
        model: Optional pre-built model adapter. If provided, it overrides the
            config-based model loading.
    """

    def __init__(
        self,
        config: ActFoldConfig,
        model: FastDLLMAdapter | None = None,
    ) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.logger = get_logger("BenchmarkRunner")

        if model is not None:
            self.model = model
            self._vocab_size = model.vocab_size
            self._tokenizer = None
            self._diffusion_model: Any | None = model.underlying_model
        else:
            diffusion_model = self._load_diffusion_model()
            self._diffusion_model = diffusion_model
            self._vocab_size = diffusion_model.vocab_size
            self._tokenizer = getattr(diffusion_model, "tokenizer", None)
            if self._tokenizer is None:
                raise RuntimeError(
                    "The loaded model does not expose a tokenizer. Benchmark "
                    "evaluation requires a real tokenizer to encode prompts. "
                    "Set use_fast_tokenizer=True in the config or provide a "
                    "model adapter with a tokenizer attribute."
                )
            folded_model = self._build_folded_model(diffusion_model)
            self.model = FastDLLMAdapter(
                diffusion_model,
                folded_model=folded_model,
            )

        self.draft_generator = DraftGenerator(
            vocab_size=self._vocab_size,
            mode="copy_flip",
            flip_ratio=0.05,
        )
        self.baseline = SpiffyBaseline(self.model, self.draft_generator)
        self.engine = self._build_engine()

    @staticmethod
    def _resolve_device(config_device: str) -> str:
        """Return a usable device string, falling back to CPU if needed."""
        try:
            torch.device(config_device)
        except RuntimeError:
            return "cpu"
        if config_device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return config_device

    def _load_diffusion_model(self) -> Any:
        """Load a real Diffusion LLM based on config.

        Raises:
            ValueError: If ``config.model_name_or_path`` is not set.
        """
        if not self.config.model_name_or_path:
            raise ValueError(
                "model_name_or_path is required for benchmarking. "
                "Pass a model identifier in the config or provide a pre-built model adapter."
            )

        self.logger.info(
            "Loading real model: %s (family=%s)",
            self.config.model_name_or_path,
            self.config.model_family,
        )
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "use_fast_tokenizer": self.config.use_fast_tokenizer,
            "torch_dtype": resolve_torch_dtype(self.config.torch_dtype) or torch.float32,
        }
        if self.config.device_map is not None:
            kwargs["device_map"] = self.config.device_map
        if self.config.load_in_8bit:
            kwargs["load_in_8bit"] = True
        if self.config.load_in_4bit:
            kwargs["load_in_4bit"] = True

        diffusion_model = load_model(
            self.config.model_name_or_path,
            model_family=self.config.model_family,
            **kwargs,
        )
        # Avoid overriding accelerate/bnb device placement when device_map or
        # quantization is configured.
        if (
            self.config.device_map is None
            and not self.config.load_in_8bit
            and not self.config.load_in_4bit
        ):
            diffusion_model.to(self.device)
        self.logger.info(
            "Model loaded on %s, memory=%.1fMB",
            get_model_device(diffusion_model),
            diffusion_model.estimate_memory_mb(),
        )
        return diffusion_model

    def _build_folded_model(self, diffusion_model: Any) -> FoldedModel | None:
        """Wrap the underlying ``nn.Module`` with :class:`FoldedModel` if possible."""
        raw_model = getattr(diffusion_model, "model", None)
        if raw_model is None:
            return None
        cache = ActivationCache(
            max_entries_per_layer=self.config.max_entries_per_layer,
            device=self.device,
        )
        gate = SimilarityGate(tau=self.config.tau, metric=self.config.metric)
        scheduler = FoldingScheduler(
            base_tau=self.config.tau,
            num_layers=getattr(diffusion_model, "num_layers", 1),
            num_steps=self.config.num_steps,
        )
        folded = FoldedModel(
            raw_model,
            cache=cache,
            gate=gate,
            scheduler=scheduler,
        )
        if not folded.folding_applied:
            return None
        return folded

    def _build_engine(self) -> ActFoldVerificationEngine:
        """Build the ActFold verification engine from config.

        The engine shares the cache, gate, and scheduler with the folded model
        so that parent activations populated by the model are reused during
        child verification.
        """
        folded = self.model.folded_model
        if folded is not None:
            cache = folded.cache
            gate = folded.gate
            scheduler = folded.scheduler
        else:
            cache = ActivationCache(
                max_entries_per_layer=self.config.max_entries_per_layer,
                device=self.device,
            )
            gate = SimilarityGate(tau=self.config.tau, metric=self.config.metric)
            scheduler = FoldingScheduler(
                base_tau=self.config.tau,
                num_layers=self.model.num_layers,
                num_steps=self.config.num_steps,
            )
        return ActFoldVerificationEngine(self.model, cache, gate, scheduler)

    def _build_judge(self, task: str) -> Judge:
        """Create a real judge for ``task``."""
        return JudgeFactory.create(
            task=task,
            backend=self.config.eval_backend,
            device=self.device,
            batch_size=self.config.eval_batch_size,
            num_fewshot=self.config.eval_num_fewshot,
            base_only=self.config.eval_base_only,
        )

    def run(
        self,
        tasks: list[str],
        num_samples: int = 10,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run the requested benchmarks.

        Args:
            tasks: List of task names such as "gsm8k", "math", "ifeval",
                "humaneval_plus", "mbpp_plus".
            num_samples: Number of examples per task. For real backends this
                is forwarded as ``eval_limit`` when ``config.eval_limit`` is
                not explicitly set.
            output_dir: Optional directory to save ``benchmark_results.json``.

        Returns:
            Dictionary mapping task name to result dictionary.
        """
        lm_tasks = set(LMEvalAdapter.TASKS)
        code_tasks = set(EvalPlusAdapter.TASKS)
        unknown_tasks = [t for t in tasks if t not in lm_tasks and t not in code_tasks]
        if unknown_tasks:
            raise ValueError(f"Unsupported benchmark tasks: {unknown_tasks}")

        eval_limit = self.config.eval_limit
        if eval_limit is None:
            eval_limit = num_samples

        results: dict[str, Any] = {}
        progress = tqdm(tasks, desc="Benchmark tasks")
        for task in progress:
            progress.set_postfix(task=task)
            if task in lm_tasks:
                judge = self._build_judge(task)
                lm_adapter = LMEvalAdapter(
                    model=self.model,
                    baseline=self.baseline,
                    engine=self.engine,
                    judge=judge,
                    tokenizer=self._tokenizer,
                    vocab_size=self._vocab_size,
                )
                results[task] = lm_adapter.evaluate(
                    task,
                    num_samples=num_samples,
                    limit=eval_limit,
                    seed=self.config.seed,
                )
            else:
                judge = self._build_judge(task)
                code_adapter = EvalPlusAdapter(
                    model=self.model,
                    baseline=self.baseline,
                    engine=self.engine,
                    judge=judge,
                    tokenizer=self._tokenizer,
                    vocab_size=self._vocab_size,
                )
                results[task] = code_adapter.evaluate(
                    task,
                    num_problems=num_samples,
                    limit=eval_limit,
                    seed=self.config.seed,
                )
            self.logger.info("Task %s finished: %s", task, results[task])

        if output_dir is not None:
            self._save_results(results, Path(output_dir))

        return results

    def _save_results(self, results: dict[str, Any], output_dir: Path) -> None:
        """Persist benchmark results to ``output_dir/benchmark_results.json``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "benchmark_results.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info("Saved benchmark results to %s", path)
