"""Configuration management for ActFold."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ActFoldConfig:
    """Top-level immutable configuration for ActFold.

    Attributes:
        tau: Default similarity threshold for token-level gating.
        metric: Similarity metric ("cosine", "l2", "pearson").
        max_entries_per_layer: Maximum cached entries per layer.
        enable_dynamic_tau: Whether to use FoldingScheduler.
        device: PyTorch device string.
        seed: Random seed for reproducibility.
        num_layers: Number of Transformer layers in the target model.
        hidden_dim: Hidden dimension of the target model.
        num_heads: Number of attention heads.
        seq_len: Maximum sequence length for FLOPs estimation.
        vocab_size: Vocabulary size.
        num_steps: Number of diffusion steps.
        torch_dtype: Model weight dtype ("float32", "float16", "bfloat16").
        device_map: Hugging Face device_map for model loading.
        use_real_eval: Must be ``True``; ActFold only supports real
            ``lm-eval`` / ``evalplus`` backends.
        eval_backend: Evaluation backend selector ("auto", "lm-eval",
            "evalplus").
        eval_batch_size: Batch size for lm-eval model evaluation.
        eval_num_fewshot: Number of few-shot examples for lm-eval tasks.
        eval_limit: Maximum number of evaluation examples.
        eval_base_only: Use base-only tests for evalplus (ignore extra tests).
    """

    tau: float = 0.95
    metric: str = "cosine"
    max_entries_per_layer: int = 1024
    enable_dynamic_tau: bool = False
    device: str = "cuda"
    seed: int = 42
    num_layers: int = 4
    hidden_dim: int = 128
    num_heads: int = 8
    seq_len: int = 16
    vocab_size: int = 1000
    num_steps: int = 10

    # Real model configuration.
    model_name_or_path: str | None = None
    model_family: str = "auto"
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str | None = None
    device_map: str | None = None

    # Evaluation configuration.
    use_real_eval: bool = True
    eval_backend: str = "auto"
    eval_batch_size: int | str = 1
    eval_num_fewshot: int | None = None
    eval_limit: int | float | None = None
    eval_base_only: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {self.tau}")
        if self.metric not in {"cosine", "l2", "pearson"}:
            raise ValueError(f"Unsupported metric: {self.metric}")
        if self.max_entries_per_layer <= 0:
            raise ValueError(
                f"max_entries_per_layer must be positive, got {self.max_entries_per_layer}"
            )
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Cannot set both load_in_8bit and load_in_4bit.")
        if self.torch_dtype is not None and self.torch_dtype not in {
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ValueError(f"Unsupported torch_dtype: {self.torch_dtype}")
        if not self.use_real_eval:
            raise ValueError(
                "ActFold only supports real evaluation backends. "
                "Set use_real_eval=True and install requirements-bench.txt."
            )
        if self.eval_backend not in {"auto", "lm-eval", "evalplus"}:
            raise ValueError(f"Unsupported eval_backend: {self.eval_backend}")


def load_config(path: Path | str) -> ActFoldConfig:
    """Load a YAML config file and return a validated ActFoldConfig.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated ActFoldConfig instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    known = {f.name for f in ActFoldConfig.__dataclass_fields__.values()}
    unknown = [k for k in raw if k not in known]
    if unknown:
        warnings.warn(
            f"Ignoring unknown config keys in {path}: {unknown}",
            stacklevel=2,
        )
    filtered = {k: v for k, v in raw.items() if k in known}
    return ActFoldConfig(**filtered)


def default_config() -> ActFoldConfig:
    """Return the default ActFold configuration."""
    return ActFoldConfig()
