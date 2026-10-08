"""Utilities for model loading, dtype resolution, and device placement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig


def infer_model_family(model_name_or_path: str) -> str:
    """Infer the model family from the model identifier.

    Args:
        model_name_or_path: Hugging Face model identifier or local path.

    Returns:
        Inferred family name.
    """
    name_lower = model_name_or_path.lower()
    if "llada" in name_lower:
        return "llada"
    if "dream" in name_lower:
        return "dream"
    if "fast" in name_lower and "dllm" in name_lower:
        return "fast_dllm"
    return "causal_lm"


def get_model_config(model_name_or_path: str, **kwargs: Any) -> AutoConfig:
    """Load a model config without loading weights.

    Args:
        model_name_or_path: Hugging Face model identifier or local path.
        **kwargs: Additional arguments for ``AutoConfig.from_pretrained``.

    Returns:
        The model configuration object.
    """
    config: AutoConfig = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
    return config


def get_available_device(preferred: str = "cuda") -> torch.device:
    """Return the best available torch device.

    Args:
        preferred: Preferred device string.

    Returns:
        Available torch.device.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_directory(path: Path | str) -> Path:
    """Ensure a directory exists and return its Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_torch_dtype(name: str | None) -> torch.dtype | None:
    """Resolve a dtype name to a ``torch.dtype``.

    Args:
        name: One of ``"float32"``, ``"float16"``, ``"bfloat16"``, or ``None``.

    Returns:
        The corresponding ``torch.dtype`` or ``None``.

    Raises:
        ValueError: If ``name`` is not a supported dtype string.
    """
    if name is None:
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype name: {name}")
    return mapping[name]


def get_model_device(model: Any) -> torch.device:
    """Return the device of ``model`` without assuming its type.

    Tries, in order:

    1. ``model.get_device()`` if defined and succeeds.
    2. The device of the first parameter.
    3. ``cpu`` as a safe fallback.

    Args:
        model: A PyTorch module or ActFold adapter.

    Returns:
        A ``torch.device`` instance.
    """
    if hasattr(model, "get_device") and callable(model.get_device):
        try:
            return torch.device(model.get_device())
        except RuntimeError:
            pass
    if hasattr(model, "parameters") and callable(model.parameters):
        try:
            return torch.device(next(model.parameters()).device)
        except StopIteration:
            pass
    return torch.device("cpu")


def model_has_attr(model: Any, name: str) -> bool:
    """Return whether ``model`` has a non-None attribute ``name``."""
    return hasattr(model, name) and getattr(model, name) is not None
