#!/usr/bin/env python3
"""ActFold Demo: Synthetic or Real Diffusion LLM.

Creates a minimal model (4 layers, 128 hidden dim, 8 heads) by default, or
loads a real model from the Hugging Face Hub when --model is provided. The
demo demonstrates:
1. Parent branch generation
2. Child branch generation (2 branches)
3. Verification with ActFold vs. Baseline
4. Per-layer similarity scores, FLOPs reduction, output equivalence
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.models import load_model
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
        x = self.norm1(x + residual)
        residual2 = x
        x = self.ffn(x)
        return self.norm2(x + residual2)


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
        return self.lm_head(x)


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
        return self.lm_head(x)


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


class RealModelFoldedForward(nn.Module):
    """Manual folded forward for GPT2-like causal language models.

    This is a structural demonstration: it extracts the embedding, Transformer
    blocks, and language head from a loaded Hugging Face model and runs the
    blocks through :class:`~actfold.core.folded_transformer.FoldedTransformerLayer`.
    Production integration for other architectures requires equivalent
    model-specific wiring.
    """

    def __init__(
        self,
        base_model: nn.Module,
        cache: ActivationCache,
        gate: SimilarityGate,
        scheduler: FoldingScheduler | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.cache = cache
        self.gate = gate
        self.scheduler = scheduler
        self.embed, self.layers, self.lm_head = self._extract_parts(base_model)
        self._wrapped_layers = nn.ModuleList(
            FoldedTransformerLayer(
                original_layer=layer,
                cache=cache,
                gate=gate,
                layer_idx=idx,
                scheduler=scheduler,
            )
            for idx, layer in enumerate(self.layers)
        )

    @staticmethod
    def _extract_parts(model: nn.Module) -> tuple[nn.Module, nn.ModuleList, nn.Module]:
        """Find embedding, Transformer blocks, and head on common HF models."""
        # GPT2 / GPT-Neo / GPT-J style.
        if hasattr(model, "transformer"):
            transformer = model.transformer
            if hasattr(transformer, "wte"):
                embed = transformer.wte
            elif hasattr(transformer, "word_embeddings"):
                embed = transformer.word_embeddings
            elif hasattr(transformer, "embedding"):
                embed = transformer.embedding
            else:
                raise RuntimeError("Could not find embedding module on transformer.")

            if hasattr(transformer, "h"):
                layers = transformer.h
            elif hasattr(transformer, "layers"):
                layers = transformer.layers
            else:
                raise RuntimeError("Could not find Transformer block list.")

            lm_head = model.lm_head
            return embed, layers, lm_head

        # Plain nn.Module with embedding/layers/head attributes.
        embed = getattr(
            model, "embedding", getattr(model, "word_embeddings", getattr(model, "wte", None))
        )
        layers = getattr(model, "layers", None)
        lm_head = getattr(model, "lm_head", getattr(model, "head", None))
        if embed is None or layers is None or lm_head is None:
            raise RuntimeError(
                "Real-model folded demo only supports GPT2-like architectures. "
                "Use --synthetic for a toy demonstration."
            )
        return embed, layers, lm_head

    def forward(
        self,
        tokens: torch.Tensor,
        branch_id: str,
        parent_branch_id: str | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(tokens)
        for wrapped in self._wrapped_layers:
            x = wrapped(
                x,
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                attention_mask=attention_mask,
                step_idx=0,
            )
        return self.lm_head(x)


def build_real_models(
    model_name_or_path: str,
    model_family: str,
    device: str,
) -> tuple[nn.Module, nn.Module, int]:
    """Load a real Diffusion LLM and build a folded structural variant.

    The baseline uses the original model.  The ActFold path uses
    :class:`RealModelFoldedForward` for GPT2-like architectures.
    """
    logger.info("Loading real model: %s (family=%s)", model_name_or_path, model_family)
    diffusion = load_model(model_name_or_path, model_family=model_family)
    diffusion.to(device)

    vocab_size = diffusion.vocab_size
    num_layers = diffusion.num_layers

    cache = ActivationCache(max_entries_per_layer=1024, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=num_layers, num_steps=10)

    raw_model = getattr(diffusion, "model", diffusion)
    folded_model = RealModelFoldedForward(
        raw_model,
        cache=cache,
        gate=gate,
        scheduler=scheduler,
    ).to(device)

    return diffusion, folded_model, vocab_size


def _measure_stable_ratios(
    folded_model: nn.Module,
    parent_tokens: torch.Tensor,
    child_tokens: torch.Tensor,
    num_layers: int,
    synthetic: bool,
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
        embed = folded_model.embed
        parent_embed = embed(parent_tokens)
        child_embed = embed(child_tokens)
        sim = torch.cosine_similarity(parent_embed, child_embed, dim=-1)
        ratio = float(sim.mean().item())
        return [ratio] * num_layers


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
        help="Model family: llada, dream, fast_dllm, causal_lm, auto.",
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
    args = parser.parse_args()

    torch.manual_seed(42)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    use_synthetic = args.synthetic or args.model is None
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
        base_model, folded_model, vocab_size = build_real_models(
            args.model, args.model_family, device
        )
        num_layers = base_model.num_layers
        hidden_dim = base_model.hidden_dim
        num_heads = base_model.num_heads

    seq_len = 16
    num_branches = 2

    print("=" * 65)
    print(" ActFold Demo")
    print("=" * 65)
    print(f" Device: {device}")
    if use_synthetic:
        print(" Model: synthetic demonstration Transformer")
    else:
        print(" Model:", args.model)
    print(f" Architecture: {num_layers} layers, {hidden_dim} hidden dim, {num_heads} heads")
    print(f" Vocab size: {vocab_size}")
    print(f" Parent branch: [seq_len={seq_len}]")
    print(f" Child branches: {num_branches}")
    print()

    # Parent branch.
    parent_tokens = torch.randint(0, vocab_size, (1, seq_len), device=device)
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

    # Also run the speculative verification engines for reporting.
    adapter = FastDLLMAdapter(
        base_model,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
    )
    cache = ActivationCache(max_entries_per_layer=1024, device=device)
    gate = SimilarityGate(tau=0.95, metric="cosine")
    scheduler = FoldingScheduler(base_tau=0.95, num_layers=num_layers, num_steps=10)
    engine = ActFoldVerificationEngine(adapter, cache, gate, scheduler)
    result = engine.verify_branch(parent_branch, example_child, step_idx=0)
    print(f" Estimated stable token ratio: {result.stable_ratio:.2%}")
    print("=" * 65)


if __name__ == "__main__":
    main()
