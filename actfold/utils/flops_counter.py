"""TFLOPs estimation for Diffusion LLMs with optional activation reuse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiffusionLLMFLOPs:
    """FLOPs breakdown for a Diffusion LLM forward pass."""

    attention_tflops: float
    ffn_tflops: float
    embedding_tflops: float
    total_tflops: float


def model_ffn_flops_kwargs(model: Any) -> dict[str, Any]:
    """Duck-typed extraction of FFN shape kwargs from a model, if exposed.

    Models (or adapters) that know their FFN geometry may expose
    ``ffn_intermediate_dim`` and ``ffn_type`` attributes; this helper turns
    them into keyword arguments for :func:`count_diffusion_llm_flops` so the
    FLOPs estimate reflects the real architecture instead of the 4h-MLP
    default.  Missing attributes fall back to the estimator defaults
    (``4 * hidden_dim`` intermediate, ``"mlp"``).

    Args:
        model: Any object; FFN attributes are optional.

    Returns:
        Dict with keys ``ffn_intermediate_dim`` and ``ffn_type``.
    """
    return {
        "ffn_intermediate_dim": getattr(model, "ffn_intermediate_dim", None),
        "ffn_type": getattr(model, "ffn_type", "mlp"),
    }


def count_diffusion_llm_flops(
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    seq_len: int,
    vocab_size: int,
    num_steps: int,
    reuse_ratio: float = 0.0,
    ffn_intermediate_dim: int | None = None,
    ffn_type: str = "mlp",
    include_attention_t2: bool = False,
) -> DiffusionLLMFLOPs:
    """Estimate TFLOPs for a Diffusion LLM forward pass.

    The estimation assumes a standard Transformer with:

    - Self-attention: ``4 * hidden_dim^2 * seq_len`` per layer, plus the
      quadratic score/value term ``2 * T_eff * seq_len * hidden_dim`` per layer
      when ``include_attention_t2=True`` (divergent queries attending over the
      full sequence).
    - FFN: ``2 * n_matmul * intermediate * hidden_dim * seq_len`` per layer,
      where ``n_matmul`` is 2 for ``"mlp"`` (up/down projection) and 3 for
      ``"swiglu"`` (gate/up/down), and ``intermediate`` is
      ``ffn_intermediate_dim`` or ``4 * hidden_dim`` when not given.
    - Embedding: ``vocab_size * hidden_dim * seq_len`` — only the LM-head
      output projection is a matmul; the input embedding is a table lookup and
      contributes no FLOPs.

    Args:
        num_layers: Number of Transformer layers.
        hidden_dim: Hidden dimension size.
        num_heads: Number of attention heads (used for validation, not FLOPs).
        seq_len: Sequence length.
        vocab_size: Vocabulary size.
        num_steps: Number of diffusion steps.
        reuse_ratio: Fraction of tokens using cached activations (0 = baseline).
        ffn_intermediate_dim: FFN intermediate dimension. ``None`` falls back
            to the classic ``4 * hidden_dim`` expansion.
        ffn_type: FFN topology, ``"mlp"`` (two matmuls) or ``"swiglu"``
            (three matmuls).
        include_attention_t2: Include the quadratic attention score/value
            FLOPs term. Off by default for comparability with the historical
            estimates.

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
    if ffn_type not in ("mlp", "swiglu"):
        raise ValueError(f"ffn_type must be 'mlp' or 'swiglu', got {ffn_type!r}")
    if ffn_intermediate_dim is not None and ffn_intermediate_dim <= 0:
        raise ValueError(f"ffn_intermediate_dim must be positive, got {ffn_intermediate_dim}")

    effective_seq_len = seq_len * (1.0 - reuse_ratio)

    # Attention: QKV projection + output projection + optional quadratic
    # score/value term (divergent queries x full sequence keys/values).
    attention_flops = 4 * num_layers * hidden_dim * hidden_dim * effective_seq_len
    if include_attention_t2:
        attention_flops += 2 * num_layers * effective_seq_len * seq_len * hidden_dim

    # FFN: n_matmul projections of hidden_dim <-> intermediate per token.
    n_matmul = 3 if ffn_type == "swiglu" else 2
    intermediate = hidden_dim * 4 if ffn_intermediate_dim is None else ffn_intermediate_dim
    ffn_flops = 2 * n_matmul * intermediate * hidden_dim * num_layers * effective_seq_len

    # Embedding: only the LM-head output projection is a matmul; the input
    # embedding is a table lookup.  The head runs on all positions regardless
    # of reuse.
    embedding_flops = vocab_size * hidden_dim * seq_len

    # Total across diffusion steps.
    total_flops = num_steps * (attention_flops + ffn_flops + embedding_flops)

    # Convert to TFLOPs (1e12).
    return DiffusionLLMFLOPs(
        attention_tflops=attention_flops * num_steps / 1e12,
        ffn_tflops=ffn_flops * num_steps / 1e12,
        embedding_tflops=embedding_flops * num_steps / 1e12,
        total_tflops=total_flops / 1e12,
    )
