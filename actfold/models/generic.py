"""Generic Diffusion LLM wrapper using Hugging Face AutoModel."""

from __future__ import annotations

from typing import Any, Callable, cast

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from actfold.models.base import DiffusionLLM


class GenericDiffusionLLM(DiffusionLLM):
    """Generic wrapper for Diffusion LLMs loadable via ``AutoModel``.

    This adapter handles architectures that expose a ``forward`` method
    returning hidden states. A language modeling head is either taken from
    the model itself or constructed as a simple linear projection to the
    vocabulary size.

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
        self.model = AutoModel.from_pretrained(
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

        # Infer vocab size from model config or tokenizer.
        config = self.model.config
        self._vocab_size = getattr(config, "vocab_size", len(self.tokenizer))
        self._hidden_dim = getattr(config, "hidden_size", getattr(config, "hidden_dim", 768))
        self._num_layers = getattr(config, "num_hidden_layers", getattr(config, "num_layers", 12))
        self._num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", 12))

        # Attach a lm_head if the model does not have one.
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            self.lm_head = cast(nn.Module, self.model.lm_head)
        elif hasattr(self.model, "get_output_embeddings"):
            output_emb = cast(Callable[[], nn.Module | None], self.model.get_output_embeddings)()
            if output_emb is not None:
                self.lm_head = output_emb
            else:
                self.lm_head = nn.Linear(self._hidden_dim, self._vocab_size, bias=False)
                nn.init.normal_(self.lm_head.weight, std=0.02)
        else:
            self.lm_head = nn.Linear(self._hidden_dim, self._vocab_size, bias=False)
            # Initialize the head with small random weights.
            nn.init.normal_(self.lm_head.weight, std=0.02)

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
        """Run a forward pass and return logits."""
        model = self.model
        if model is None:
            raise RuntimeError("Model has not been loaded.")
        outputs = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        last_hidden = outputs.last_hidden_state
        logits: torch.Tensor = cast(torch.Tensor, self.lm_head(last_hidden))
        return logits

    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 16,
        num_steps: int = 10,
        folded_model: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens, delegating to the base class implementation.

        The base class handles autoregressive fallback and diffusion sampling
        via :meth:`DiffusionLLM.get_native_sampler`.
        """
        return super().generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            num_steps=num_steps,
            folded_model=folded_model,
            **kwargs,
        )

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
