"""Model registry and factory for loading Diffusion LLMs."""

from __future__ import annotations

from typing import Any

from actfold.models.base import DiffusionLLM
from actfold.models.causal_lm import CausalLMDiffusionLLM
from actfold.models.dream import DreamModel
from actfold.models.fast_dllm import FastDLLMModel
from actfold.models.generic import GenericDiffusionLLM
from actfold.models.llada import LLaDAModel


class ModelRegistry:
    """Registry mapping model family names to wrapper classes."""

    _REGISTRY: dict[str, type[DiffusionLLM]] = {
        "llada": LLaDAModel,
        "dream": DreamModel,
        "fast_dllm": FastDLLMModel,
        "fast-dllm": FastDLLMModel,
        "causal_lm": CausalLMDiffusionLLM,
        "causal": CausalLMDiffusionLLM,
        "generic": GenericDiffusionLLM,
    }

    @classmethod
    def register(
        cls,
        name: str,
        model_class: type[DiffusionLLM],
    ) -> None:
        """Register a new model family.

        Args:
            name: Family name used in configs.
            model_class: Concrete DiffusionLLM subclass.
        """
        cls._REGISTRY[name.lower()] = model_class

    @classmethod
    def list_models(cls) -> list[str]:
        """Return all registered model family names."""
        return sorted(cls._REGISTRY.keys())

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        model_family: str = "auto",
        **kwargs: Any,
    ) -> DiffusionLLM:
        """Load a model by family name or auto-detect.

        Args:
            model_name_or_path: Hugging Face model identifier or local path.
            model_family: One of the registered family names, or "auto" to infer
                from the model identifier.
            **kwargs: Additional arguments forwarded to the wrapper constructor.

        Returns:
            Loaded DiffusionLLM instance.

        Raises:
            ValueError: If ``model_family`` is unknown.
        """
        family = cls._resolve_family(model_name_or_path, model_family)
        model_class = cls._REGISTRY.get(family.lower())
        if model_class is None:
            raise ValueError(
                f"Unknown model family: {family}. "
                f"Available: {', '.join(cls.list_models())}"
            )
        return model_class(model_name_or_path, **kwargs)

    @classmethod
    def _resolve_family(cls, model_name_or_path: str, model_family: str) -> str:
        """Resolve model family from config or identifier."""
        if model_family != "auto":
            return model_family

        name_lower = model_name_or_path.lower()
        if "llada" in name_lower:
            return "llada"
        if "dream" in name_lower:
            return "dream"
        if "fast" in name_lower and "dllm" in name_lower:
            return "fast_dllm"
        # Default to causal LM as the safest AutoModel-based fallback.
        return "causal_lm"


def load_model(
    model_name_or_path: str,
    model_family: str = "auto",
    **kwargs: Any,
) -> DiffusionLLM:
    """Convenience function to load a registered Diffusion LLM.

    Args:
        model_name_or_path: Hugging Face model identifier or local path.
        model_family: Model family name or "auto".
        **kwargs: Additional constructor arguments.

    Returns:
        Loaded DiffusionLLM instance.
    """
    return ModelRegistry.load(model_name_or_path, model_family=model_family, **kwargs)
