"""Text generation helpers for benchmark adapters."""

from __future__ import annotations

from typing import Any

import torch

from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
from actfold.utils.logger import get_logger

logger = get_logger("eval.generation")


def fallback_decode(tokens: torch.Tensor) -> str:
    """Deterministically map token ids back to a string without a tokenizer."""
    chars = [chr(min(1114111, max(0, int(t)))) for t in tokens.flatten().tolist()]
    return "".join(chars)


def encode_prompt(
    prompt: str,
    tokenizer: Any | None,
    vocab_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode ``prompt`` to token ids using a real tokenizer.

    Raises:
        RuntimeError: If no tokenizer is provided.  Benchmark evaluation
            requires a real tokenizer to encode text prompts.
    """
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        ids: torch.Tensor = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        return ids.to(device)
    raise RuntimeError(
        "A real tokenizer is required to encode benchmark prompts. "
        "The model loaded for evaluation does not expose a tokenizer."
    )


def decode_tokens(tokens: torch.Tensor, tokenizer: Any | None) -> str:
    """Decode ``tokens`` to a string, using a tokenizer if available.

    If no tokenizer is provided, a deterministic character fallback is used.
    This fallback is intended only for debugging or visualization; production
    evaluation should always supply a real tokenizer.
    """
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        text: str = tokenizer.decode(tokens, skip_special_tokens=True)
        return text
    logger.warning("No tokenizer provided; using deterministic fallback decode.")
    return fallback_decode(tokens)


def greedy_generate(
    model: FastDLLMAdapter,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Greedy autoregressive generation using the adapter's forward pass."""
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model.forward(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
    return generated


def get_model_device(model: FastDLLMAdapter) -> torch.device:
    """Return the device of ``model``'s underlying module."""
    raw = model.underlying_model
    if hasattr(raw, "parameters"):
        try:
            return torch.device(next(raw.parameters()).device)
        except StopIteration:
            pass
    if hasattr(raw, "get_device"):
        device_getter = raw.get_device
        if callable(device_getter):
            return torch.device(device_getter())
    return torch.device("cpu")
