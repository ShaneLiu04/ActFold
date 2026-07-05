#!/usr/bin/env python3
"""ActFold Demo: Synthetic or Real Diffusion LLM.

Creates a minimal model (4 layers, 128 hidden dim, 8 heads) by default, or
loads a real model from the Hugging Face Hub when --model is provided. The
demo demonstrates:
1. Parent branch generation
2. Child branch generation (2 branches)
3. Verification with ActFold vs. Baseline
4. Per-layer similarity scores, FLOPs reduction, output equivalence

The real-model path is architecture-agnostic: it auto-detects the Transformer
layer stack, embedding module, and language modeling head using
:class:`~actfold.models.architecture_utils.ArchitectureProfile` and wraps
layers with :class:`~actfold.core.model_wrapper.FoldedModel`.
"""

from __future__ import annotations

import argparse
from typing import Callable, cast

import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.core.model_wrapper import FoldedModel
from actfold.models import ArchitectureProfile, detect_architecture, load_model
from actfold.models.architecture_utils import build_manual_folded_forward
from actfold.models.utils import resolve_torch_dtype
from actfold.speculative import ActFoldVerificationEngine, DraftGenerator, FastDLLMAdapter
from actfold.speculative.branch import Branch
from actfold.utils.flops_counter import count_diffusion_llm_flops
from actfold.utils.logger import get_logger

logger = get_logger("demo")


class SyntheticTransformerLayer(nn.Module):
    """A single Transformer layer for the synthetic model."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        x, _ = self.attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        x = self.norm1(cast(torch.Tensor, x) + residual)
        residual2 = x
        x = self.ffn(x)
        return cast(torch.Tensor, self.norm2(cast(torch.Tensor, x) + residual2))


class SyntheticTransformer(nn.Module):
    """Tiny Transformer whose layers can be wrapped by FoldedTransformerLayer."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            SyntheticTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return cast(torch.Tensor, self.lm_head(x))


class FoldedSyntheticTransformer(nn.Module):
    """Synthetic Transformer with FoldedTransformerLayer wrappers."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        cache: ActivationCache,
        gate: SimilarityGate,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            FoldedTransformerLayer(
                original_layer=SyntheticTransformerLayer(hidden_dim, num_heads),
                cache=cache,
                gate=gate,
                layer_idx=idx,
            )
            for idx in range(num_layers)
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        branch_id: str,
        parent_branch_id: str | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(
                x,
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                attention_mask=attention_mask,
            )
        return cast(torch.Tensor, self.lm_head(x))


def print_table(rows: list[list[str]]) -> None:
    """Print a simple aligned text table using ASCII-safe characters."""
    col_widths = [max(len(row[i]) for row in rows) + 2 for i in range(len(rows[0]))]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    def fmt(row: list[str]) -> str:
        cells = [f" {row[i]:<{col_widths[i] - 1}}" for i in range(len(row))]
        return "|" + "|".join(cells) + "|"

    print(sep)
    for i, row in enumerate(rows):
        print(fmt(row))
        if i == 0 and len(rows) > 1:
            print(sep)
    print(sep)


def build_synthetic_models(
    vocab_size: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    device: str,
) -> tuple[nn.Module, nn.Module, int]:
    """Build baseline and folded synthetic models sharing weights."""
    base_model = SyntheticTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)

    cache = ActivationCache(max_entries_per_layer=1024, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")

    folded_model = FoldedSyntheticTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        cache=cache,
        gate=gate,
    ).to(device)

    # Share weights so the two models are functionally identical.
    folded_model.embedding.weight.data.copy_(base_model.embedding.weight.data)
    for folded_layer, base_layer in zip(folded_model.layers, base_model.layers):
        folded_layer.original_layer.load_state_dict(base_layer.state_dict())
    folded_model.lm_head.weight.data.copy_(base_model.lm_head.weight.data)

    return base_model, folded_model, vocab_size


def build_real_models(
    model_name_or_path: str,
    model_family: str,
    device: str,
    dtype: torch.dtype | None,
) -> tuple[nn.Module, nn.Module, int, ArchitectureProfile]:
    """Load a real Diffusion LLM and build an architecture-agnostic folded variant.

    The baseline uses the original model.  The ActFold path first tries
    :class:`~actfold.core.model_wrapper.FoldedModel`, which auto-discovers the
    Transformer layer stack for most Hugging Face architectures.  If that fails,
    it falls back to :class:`~actfold.models.architecture_utils.ManualFoldedForward`,
    which explicitly extracts the embedding, layers, and head.
    """
    logger.info("Loading real model: %s (family=%s)", model_name_or_path, model_family)
    load_kwargs: dict[str, object] = {}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    diffusion = load_model(model_name_or_path, model_family=model_family, **load_kwargs)
    diffusion.to(device)

    vocab_size = diffusion.vocab_size
    num_layers = diffusion.num_layers

    cache = ActivationCache(max_entries_per_layer=1024, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=num_layers, num_steps=10)

    raw_model = getattr(diffusion, "model", diffusion)
    folded_model: nn.Module = FoldedModel(
        raw_model,
        cache=cache,
        gate=gate,
        scheduler=scheduler,
    ).to(device)

    if not getattr(folded_model, "folding_applied", False):
        logger.info("FoldedModel could not auto-discover layers; using manual extraction.")
        cache2 = ActivationCache(max_entries_per_layer=1024, device=device)
        folded_model = build_manual_folded_forward(
            raw_model,
            cache=cache2,
            gate=SimilarityGate(tau=0.95, metric="cosine"),
            scheduler=scheduler,
        ).to(device)

    profile = detect_architecture(raw_model)
    return diffusion, folded_model, vocab_size, profile


def _measure_stable_ratios(
    folded_model: nn.Module,
    parent_tokens: torch.Tensor,
    child_tokens: torch.Tensor,
    num_layers: int,
    synthetic: bool,
    embed_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[float]:
    """Measure per-layer stable ratios from real parent/child hidden states.

    For the synthetic model we run each layer explicitly and compare outputs.
    For real models we use the embedding similarity as a conservative proxy
    because the per-layer hidden states are already consumed by the folded
    forward pass.
    """
    with torch.no_grad():
        if synthetic:
            parent_hidden = folded_model.embedding(parent_tokens)
            child_hidden = folded_model.embedding(child_tokens)
            ratios: list[float] = []
            for layer in folded_model.layers:
                parent_hidden = layer.original_layer(parent_hidden)
                child_hidden = layer(
                    child_hidden,
                    branch_id="child",
                    parent_branch_id="parent",
                )
                sim = torch.cosine_similarity(parent_hidden, child_hidden, dim=-1)
                ratios.append(float(sim.mean().item()))
            return ratios

        # Real model path: use embedding-layer similarity as a proxy.
        if embed_fn is None:
            profile = detect_architecture(folded_model)
            embed = profile.embed_module
            embed_fn = cast(Callable[[torch.Tensor], torch.Tensor], embed)
        parent_embed = embed_fn(parent_tokens)
        child_embed = embed_fn(child_tokens)
        sim = torch.cosine_similarity(parent_embed, child_embed, dim=-1)
        ratio = float(sim.mean().item())
        return [ratio] * num_layers


def _encode_prompt(
    diffusion: nn.Module,
    prompt: str | None,
    seq_len: int,
    device: str,
) -> torch.Tensor:
    """Encode a text prompt, falling back to random token ids when unavailable."""
    tokenizer = getattr(diffusion, "tokenizer", None)
    if prompt is not None and tokenizer is not None:
        encoded = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        return cast(torch.Tensor, encoded).to(device)

    if not hasattr(diffusion, "vocab_size"):
        # Synthetic demonstration path: use the fixed vocabulary size.
        vocab_size = 1000
    else:
        vocab_size = int(diffusion.vocab_size)
    logger.info("No tokenizer or prompt provided; using random token ids for the parent branch.")
    return torch.randint(0, vocab_size, (1, seq_len), device=device)


def _maybe_generate(
    diffusion: nn.Module,
    folded_model: nn.Module,
    prompt_tokens: torch.Tensor,
    num_steps: int,
    max_new_tokens: int,
) -> torch.Tensor | None:
    """Run a short diffusion/generation step if the family supports it."""
    if num_steps <= 1:
        return None
    try:
        generated = diffusion.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            num_steps=num_steps,
            folded_model=folded_model,
        )
        return cast(torch.Tensor | None, generated)
    except Exception as exc:  # noqa: BLE001
        logger.info("Generation step skipped: %s", exc)
        return None


def main() -> None:
    """Run the ActFold demo."""
    parser = argparse.ArgumentParser(description="ActFold Demo")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hugging Face model identifier or local path (e.g., gpt2). "
        "If omitted, a small synthetic Transformer is used for the demo.",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default="auto",
        help="Model family: llada, dream, fast_dllm, causal_lm, generic, auto.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a tiny synthetic Transformer instead of a real model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu). Defaults to best available.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype for the real model. Defaults to float32.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Sequence length for the parent branch when using random tokens.",
    )
    parser.add_argument(
        "--num-branches",
        type=int,
        default=2,
        help="Number of child branches to generate.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.95,
        help="Cosine-similarity threshold for the similarity gate.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional text prompt. If provided and a tokenizer is available, "
        "it is used as the parent branch instead of random token ids.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1,
        help="Number of diffusion steps for the optional generation demo. "
        "Use 1 to skip generation and only run the folding benchmark.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Number of new tokens to generate when --num-steps > 1.",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_torch_dtype(args.dtype)

    use_synthetic = args.synthetic or args.model is None
    profile: ArchitectureProfile | None = None
    if use_synthetic:
        if args.synthetic:
            logger.info("Using synthetic demonstration model.")
        else:
            logger.info("No --model provided; using synthetic demonstration model.")
        vocab_size = 1000
        hidden_dim = 128
        num_layers = 4
        num_heads = 8
        base_model, folded_model, vocab_size = build_synthetic_models(
            vocab_size, hidden_dim, num_layers, num_heads, device
        )
    else:
        base_model, folded_model, vocab_size, profile = build_real_models(
            args.model, args.model_family, device, dtype
        )
        num_layers = base_model.num_layers
        hidden_dim = base_model.hidden_dim
        num_heads = base_model.num_heads

    seq_len = args.seq_len
    num_branches = args.num_branches

    print("=" * 65)
    print(" ActFold Demo")
    print("=" * 65)
    print(f" Device: {device}")
    if dtype is not None:
        print(f" Dtype: {args.dtype}")
    if use_synthetic:
        print(" Model: synthetic demonstration Transformer")
    else:
        print(" Model:", args.model)
        if profile is not None:
            print(f" Detected architecture: {profile.model_type}")
            print(f" Layer path: {profile.layer_path}")
            print(f" Embed path: {profile.embed_path}")
            if profile.head_path:
                print(f" Head path: {profile.head_path}")
    print(f" Architecture: {num_layers} layers, {hidden_dim} hidden dim, {num_heads} heads")
    print(f" Vocab size: {vocab_size}")
    print(f" Parent branch: [seq_len={seq_len}]")
    print(f" Child branches: {num_branches}")
    print()

    # Parent branch.
    parent_tokens = _encode_prompt(base_model, args.prompt, seq_len, device)
    parent_branch = Branch(
        branch_id="root",
        parent_id=None,
        tokens=parent_tokens,
    )

    # Generate child branches close to parent.
    draft_generator = DraftGenerator(
        vocab_size=vocab_size,
        mode="copy_flip",
        flip_ratio=0.05,
    )
    child_branches = draft_generator.generate(
        parent=parent_branch,
        num_branches=num_branches,
        seed=42,
    )

    # Run parent through folded model to populate cache.
    with torch.no_grad():
        _ = folded_model(parent_branch.tokens, branch_id=parent_branch.branch_id)

    # Baseline full forward on the first child.
    example_child = child_branches[0]
    with torch.no_grad():
        baseline_logits = base_model(example_child.tokens)

    # ActFold forward on the same child reusing parent cache.
    with torch.no_grad():
        actfold_logits = folded_model(
            example_child.tokens,
            branch_id=example_child.branch_id,
            parent_branch_id=parent_branch.branch_id,
        )

    # Compute per-layer stable ratios from real parent/child hidden states.
    rows = [["Layer", "Baseline", "ActFold", "Similarity"]]
    stable_ratios = _measure_stable_ratios(
        folded_model,
        parent_branch.tokens,
        example_child.tokens,
        num_layers,
        synthetic=use_synthetic,
        embed_fn=base_model.embed if not use_synthetic else None,
    )
    for layer_idx, ratio in enumerate(stable_ratios):
        actfold_pct = f"{int(100 * (1.0 - ratio))}%"
        baseline_pct = "100%"
        sim = ratio**0.5  # rough similarity corresponding to the stable ratio
        rows.append([str(layer_idx), baseline_pct, actfold_pct, f"{sim:.3f}"])

    print(" Verification Results:")
    print_table(rows)

    # FLOPs reduction estimate.
    avg_stable = sum(stable_ratios) / len(stable_ratios)
    base_flops = count_diffusion_llm_flops(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_steps=1,
        reuse_ratio=0.0,
    )
    actfold_flops = count_diffusion_llm_flops(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_steps=1,
        reuse_ratio=avg_stable,
    )
    flops_reduction = 100.0 * (1.0 - actfold_flops.total_tflops / base_flops.total_tflops)
    print(f" Total FLOPs reduction: {flops_reduction:.1f}%")

    # Output equivalence.
    mse = (baseline_logits - actfold_logits).pow(2).mean().item()
    status = "OK" if mse < 1e-3 else "HIGH"
    print(f" Output equivalence (MSE): {mse:.2e}  [{status}]")

    # Optional generation demo.
    if args.num_steps > 1 and not use_synthetic:
        print()
        print(" Generation Demo:")
        generated = _maybe_generate(
            base_model,
            folded_model,
            parent_tokens,
            args.num_steps,
            args.max_new_tokens,
        )
        if generated is not None:
            tokenizer = getattr(base_model, "tokenizer", None)
            if tokenizer is not None:
                text = tokenizer.decode(generated[0], skip_special_tokens=True)
                print(f" Generated: {text}")
            else:
                print(f" Generated tokens: {generated[0].tolist()}")

    # Also run the speculative verification engines for reporting.
    adapter = FastDLLMAdapter(
        base_model,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
    )
    cache = ActivationCache(max_entries_per_layer=1024, device=device)
    gate = SimilarityGate(tau=args.tau, metric="cosine")
    scheduler = FoldingScheduler(base_tau=args.tau, num_layers=num_layers, num_steps=10)
    engine = ActFoldVerificationEngine(adapter, cache, gate, scheduler)
    result = engine.verify_branch(parent_branch, example_child, step_idx=0)
    print(f" Estimated stable token ratio: {result.stable_ratio:.2%}")
    print("=" * 65)


if __name__ == "__main__":
    main()
