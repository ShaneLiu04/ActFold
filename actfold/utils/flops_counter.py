"""TFLOPs estimation for Diffusion LLMs with optional activation reuse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# HF ``hidden_act`` values whose FFN uses the gated (gate/up/down, 3-matmul)
# topology. Unknown names fall back to the 2-matmul "mlp" estimate.
_SWIGLU_ACTS = frozenset({"silu", "swish", "swiglu"})

# Geometry keys shared by ``model_ffn_flops_kwargs`` and the
# ``DiffusionLLM`` properties (single extraction implementation, D4).
_GEOMETRY_KEYS = (
    "ffn_intermediate_dim",
    "ffn_type",
    "moe_num_experts",
    "moe_top_k",
    "moe_intermediate_dim",
    "moe_shared_expert",
    "moe_num_layers",
)


@dataclass(frozen=True)
class DiffusionLLMFLOPs:
    """FLOPs breakdown for a Diffusion LLM forward pass."""

    attention_tflops: float
    ffn_tflops: float
    embedding_tflops: float
    total_tflops: float


def _first_not_none(*values: Any) -> Any:
    """Return the first non-None argument (None if all are None)."""
    for value in values:
        if value is not None:
            return value
    return None


def _extract_ffn_geometry(config: Any) -> dict[str, Any]:
    """Duck-typed extraction of FFN/MoE geometry from an HF-like config.

    Reads the union of mainstream HF attribute names (Qwen2/DeepSeek MoE
    families included) and falls back to estimator defaults for anything
    missing. Unknown model families therefore degrade safely to the dense
    4h-MLP estimate instead of erroring.

    Args:
        config: Any object; all attributes are optional.

    Returns:
        Dict with the seven geometry keys of ``_GEOMETRY_KEYS``
        (``ffn_intermediate_dim``/``moe_*`` are ``None``, ``ffn_type`` is
        ``"mlp"`` and ``moe_shared_expert`` is ``False`` when unknown).
    """
    if config is None:
        geometry: dict[str, Any] = {key: None for key in _GEOMETRY_KEYS}
        geometry["ffn_type"] = "mlp"
        geometry["moe_shared_expert"] = False
        return geometry

    hidden_act = getattr(config, "hidden_act", "")
    ffn_type = "swiglu" if isinstance(hidden_act, str) and hidden_act in _SWIGLU_ACTS else "mlp"

    total_layers = _first_not_none(
        getattr(config, "num_hidden_layers", None),
        getattr(config, "num_layers", None),
    )
    moe_num_layers = (
        None if total_layers is None else total_layers - getattr(config, "first_k_dense_replace", 0)
    )

    shared_intermediate = getattr(config, "shared_expert_intermediate_size", 0)

    return {
        "ffn_intermediate_dim": getattr(config, "intermediate_size", None),
        "ffn_type": ffn_type,
        "moe_num_experts": _first_not_none(
            getattr(config, "num_experts", None),
            getattr(config, "n_routed_experts", None),
        ),
        "moe_top_k": _first_not_none(
            getattr(config, "num_experts_per_tok", None),
            getattr(config, "num_selected_experts", None),
        ),
        "moe_intermediate_dim": _first_not_none(
            getattr(config, "moe_intermediate_size", None),
            getattr(config, "expert_intermediate_size", None),
        ),
        "moe_shared_expert": bool(shared_intermediate and shared_intermediate > 0),
        "moe_num_layers": moe_num_layers,
    }


def _resolve_model_config(target: Any) -> Any:
    """Resolve the HF config of ``target`` through the standard chains.

    Search order: ``target.config`` (DiffusionLLM ``config`` property or a raw
    HF module), then ``target.model.config`` (a wrapper holding the HF module
    directly). Both chains feed the same :func:`_extract_ffn_geometry`, so the
    ``DiffusionLLM`` properties and ``model_ffn_flops_kwargs`` always see the
    same config object (design D9).

    Args:
        target: Any object; all attributes are optional.

    Returns:
        The resolved config object, or ``None`` when unavailable.
    """
    config = getattr(target, "config", None)
    if config is not None:
        return config
    inner = getattr(target, "model", None)
    if inner is not None:
        return getattr(inner, "config", None)
    return None


def model_ffn_flops_kwargs(model: Any) -> dict[str, Any]:
    """Extract FFN/MoE geometry kwargs from a model or adapter, if exposed.

    The helper first drills through ``underlying_model`` wrappers (e.g.
    :class:`~actfold.speculative.fast_dllm_adapter.FastDLLMAdapter`), then
    resolves the target's HF config through :func:`_resolve_model_config`.
    Geometry exposed as direct attributes or properties (the
    :class:`~actfold.models.base.DiffusionLLM` FFN properties) takes
    precedence; the config extraction fills the rest. Missing everything
    falls back to the estimator defaults (``4 * hidden_dim`` intermediate,
    ``"mlp"``, no MoE accounting).

    Args:
        model: Any object; all attributes are optional.

    Returns:
        Dict with the seven keys of ``_GEOMETRY_KEYS`` (keys are
        additive-only; the historical two-key contract is preserved).
    """
    target = model
    for _ in range(8):  # bounded wrapper-drill depth
        inner = getattr(target, "underlying_model", None)
        if inner is None or inner is target:
            break
        target = inner

    geometry = _extract_ffn_geometry(_resolve_model_config(target))
    for key in _GEOMETRY_KEYS:
        value = getattr(target, key, None)
        if value is not None:
            geometry[key] = value
    return geometry


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
    moe_num_experts: int | None = None,
    moe_top_k: int | None = None,
    moe_intermediate_dim: int | None = None,
    moe_shared_expert: bool = False,
    moe_num_layers: int | None = None,
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
    - MoE layers (``moe_top_k`` given, AR005): each token activates
      ``moe_top_k`` routed experts plus an optional always-on shared expert;
      per-token expert FLOPs are
      ``2 * n_matmul * moe_inter * hidden_dim * (top_k + shared)``.  The
      expert intermediate ``moe_inter`` falls back to ``ffn_intermediate_dim``
      and then ``4 * hidden_dim``.  ``moe_num_layers`` (default: all layers)
      splits the stack into MoE and dense prefixes; dense layers use the
      dense formula above.  Router gating FLOPs (~``hidden_dim *
      num_experts`` per token, <1% of expert FFN) are ignored.
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
        ffn_type: FFN / expert activation topology, ``"mlp"`` (two matmuls)
            or ``"swiglu"`` (three matmuls).
        include_attention_t2: Include the quadratic attention score/value
            FLOPs term. Off by default for comparability with the historical
            estimates.
        moe_num_experts: Total number of routed experts. Used only to
            validate ``moe_top_k`` capacity; per-token FLOPs depend on
            ``moe_top_k`` alone.
        moe_top_k: Experts activated per token. This is the MoE accounting
            trigger; ``None`` keeps the dense formula bit-identical.
        moe_intermediate_dim: Expert FFN intermediate dimension. ``None``
            falls back to ``ffn_intermediate_dim`` then ``4 * hidden_dim``.
        moe_shared_expert: Whether an always-on shared expert runs on every
            token (adds one expert-equivalent to the per-token count).
        moe_num_layers: Number of MoE layers. ``None`` treats all layers as
            MoE; the remaining ``num_layers - moe_num_layers`` layers use the
            dense formula (DeepSeek-style dense prefixes).

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
    if moe_top_k is not None:
        if moe_top_k <= 0:
            raise ValueError(f"moe_top_k must be positive, got {moe_top_k}")
        if moe_num_experts is not None and moe_num_experts <= 0:
            raise ValueError(f"moe_num_experts must be positive, got {moe_num_experts}")
        if moe_num_experts is not None and moe_top_k > moe_num_experts:
            raise ValueError(
                f"moe_top_k ({moe_top_k}) must not exceed moe_num_experts "
                f"({moe_num_experts})"
            )
        if moe_intermediate_dim is not None and moe_intermediate_dim <= 0:
            raise ValueError(
                f"moe_intermediate_dim must be positive, got {moe_intermediate_dim}"
            )
        if moe_num_layers is not None and not 0 <= moe_num_layers <= num_layers:
            raise ValueError(
                f"moe_num_layers ({moe_num_layers}) must be in [0, {num_layers}]"
            )

    effective_seq_len = seq_len * (1.0 - reuse_ratio)

    # Attention: QKV projection + output projection + optional quadratic
    # score/value term (divergent queries x full sequence keys/values).
    attention_flops = 4 * num_layers * hidden_dim * hidden_dim * effective_seq_len
    if include_attention_t2:
        attention_flops += 2 * num_layers * effective_seq_len * seq_len * hidden_dim

    # FFN: n_matmul projections of hidden_dim <-> intermediate per token.
    n_matmul = 3 if ffn_type == "swiglu" else 2
    dense_intermediate = hidden_dim * 4 if ffn_intermediate_dim is None else ffn_intermediate_dim
    if moe_top_k is None:
        ffn_flops = (
            2 * n_matmul * dense_intermediate * hidden_dim * num_layers * effective_seq_len
        )
    else:
        # MoE accounting (AR005): top_k routed experts +/- one shared expert
        # per token, applied to the MoE layer subset; the remaining layers
        # stay dense.  Router gating is negligible (<1% of expert FFN).
        moe_intermediate = (
            dense_intermediate if moe_intermediate_dim is None else moe_intermediate_dim
        )
        experts_per_token = moe_top_k + (1 if moe_shared_expert else 0)
        moe_layer_count = num_layers if moe_num_layers is None else moe_num_layers
        moe_flops = (
            2
            * n_matmul
            * moe_intermediate
            * hidden_dim
            * experts_per_token
            * moe_layer_count
            * effective_seq_len
        )
        dense_flops = (
            2
            * n_matmul
            * dense_intermediate
            * hidden_dim
            * (num_layers - moe_layer_count)
            * effective_seq_len
        )
        ffn_flops = moe_flops + dense_flops

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
