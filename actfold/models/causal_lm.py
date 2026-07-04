"""Wrapper for standard causal language models (e.g., LLaMA, GPT-2)."""

from __future__ import annotations

from typing import Any, Callable, cast

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from actfold.models.base import DiffusionLLM


class CausalLMDiffusionLLM(DiffusionLLM):
    """Adapter for causal LMs used as a stand-in for Diffusion LLMs.

    This wrapper allows ActFold to run on any causal LM from the Hugging Face
    Hub, which is useful for rapid prototyping when a native Diffusion LLM is
    not available.

    Args:
        model_name_or_path: Hugging Face model identifier or local path.
        trust_remote_code: Whether to trust remote code in the model repo.
        use_fast_tokenizer: Whether to use the fast tokenizer implementation.
        torch_dtype: Torch dtype for model weights. Defaults to float32 for
            broad CPU/GPU compatibility; use float16/bfloat16 for speed.
        device_map: Optional Hugging Face ``device_map`` for model loading.
        load_in_8bit: Load model in 8-bit via bitsandbytes.
        load_in_4bit: Load model in 4-bit via bitsandbytes.
    """

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = True,
        use_fast_tokenizer: bool = True,
        torch_dtype: torch.dtype = torch.float32,
        device_map: str | None = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        super().__init__(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = self.model.config
        self._vocab_size = getattr(config, "vocab_size", len(self.tokenizer))
        self._hidden_dim = getattr(config, "hidden_size", getattr(config, "hidden_dim", 768))
        self._num_layers = getattr(config, "num_hidden_layers", getattr(config, "num_layers", 12))
        self._num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", 12))

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return input embeddings for ``tokens``."""
        model = self.model
        if model is None:
            raise RuntimeError("Model has not been loaded.")
        getter = cast(Callable[[], nn.Module], model.get_input_embeddings)
        embeddings: torch.Tensor = getter()(tokens)
        return embeddings

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return logits from the causal LM."""
        model = self.model
        if model is None:
            raise RuntimeError("Model has not been loaded.")
        outputs = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        logits: torch.Tensor = outputs.logits
        return logits

    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 16,
        num_steps: int = 10,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using the model's built-in generation method."""
        model = self.model
        if model is None:
            raise RuntimeError("Model has not been loaded.")
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded.")
        generate_fn = cast(
            Callable[..., torch.Tensor],
            model.generate,
        )
        generated: torch.Tensor = generate_fn(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )
        return generated

    @property
    def num_layers(self) -> int:
        return int(self._num_layers)

    @property
    def hidden_dim(self) -> int:
        return int(self._hidden_dim)

    @property
    def num_heads(self) -> int:
        return int(self._num_heads)

    @property
    def vocab_size(self) -> int:
        return int(self._vocab_size)
