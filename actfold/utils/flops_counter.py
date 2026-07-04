"""TFLOPs estimation for Diffusion LLMs with optional activation reuse."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionLLMFLOPs:
    """FLOPs breakdown for a Diffusion LLM forward pass."""

    attention_tflops: float
    ffn_tflops: float
    embedding_tflops: float
    total_tflops: float


def count_diffusion_llm_flops(
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    seq_len: int,
    vocab_size: int,
    num_steps: int,
    reuse_ratio: float = 0.0,
) -> DiffusionLLMFLOPs:
    """Estimate TFLOPs for a Diffusion LLM forward pass.

    The estimation assumes a standard Transformer with:
    - Self-attention: 4 * hidden_dim^2 * seq_len per layer
    - FFN: 16 * hidden_dim^2 * seq_len per layer (expansion factor 4, two projections)
    - Embeddings: 2 * vocab_size * hidden_dim * seq_len

    Args:
        num_layers: Number of Transformer layers.
        hidden_dim: Hidden dimension size.
        num_heads: Number of attention heads (used for validation, not FLOPs).
        seq_len: Sequence length.
        vocab_size: Vocabulary size.
        num_steps: Number of diffusion steps.
        reuse_ratio: Fraction of tokens using cached activations (0 = baseline).

    Returns:
        DiffusionLLMFLOPs breakdown.

    Raises:
        ValueError: If any argument is invalid.
    """
    if num_layers <= 0 or hidden_dim <= 0 or seq_len <= 0 or vocab_size <= 0 or num_steps <= 0:
        raise ValueError("Model dimensions must be positive.")
    if not 0.0 <= reuse_ratio <= 1.0:
        raise ValueError(f"reuse_ratio must be in [0, 1], got {reuse_ratio}")
    if hidden_dim % num_heads != 0:
        raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

    effective_seq_len = seq_len * (1.0 - reuse_ratio)

    # Attention: QKV projection + output projection + softmax/score ops.
    # Simplified: 4 * hidden_dim^2 * seq_len per layer.
    attention_flops = 4 * num_layers * hidden_dim * hidden_dim * effective_seq_len

    # FFN: up-project + down-project with intermediate dim 4 * hidden_dim.
    ffn_flops = 16 * num_layers * hidden_dim * hidden_dim * effective_seq_len

    # Embeddings (input + output projection); assume full vocab projection even with reuse.
    embedding_flops = 2 * vocab_size * hidden_dim * seq_len

    # Total across diffusion steps.
    total_flops = num_steps * (attention_flops + ffn_flops + embedding_flops)

    # Convert to TFLOPs (1e12).
    return DiffusionLLMFLOPs(
        attention_tflops=attention_flops * num_steps / 1e12,
        ffn_tflops=ffn_flops * num_steps / 1e12,
        embedding_tflops=embedding_flops * num_steps / 1e12,
        total_tflops=total_flops / 1e12,
    )
