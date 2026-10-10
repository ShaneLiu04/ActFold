"""Red tests for T024: measured (non-synthetic) ablation studies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.model_wrapper import FoldedModel
from actfold.eval.ablation_study import AblationStudy, FoldedMeasurement
from actfold.speculative import DraftGenerator, FastDLLMAdapter


def _make_model(
    vocab_size: int = 100,
    hidden_dim: int = 64,
    num_layers: int = 4,
    seed: int = 1234,
) -> nn.Module:
    """Build a tiny deterministic Transformer stack for CPU-only tests.

    Args:
        vocab_size: Size of the embedding table and output head.
        hidden_dim: Hidden dimension of the encoder layers.
        num_layers: Number of Transformer encoder layers.
        seed: Torch seed so weight initialization is deterministic.

    Returns:
        A raw ``nn.Module`` with ``.embedding``, ``.layers`` and ``.head`` whose
        forward maps ``tokens`` of shape ``[batch, seq]`` to logits.
    """

    class TinyModel(nn.Module):
        """Minimal encoder-only model auto-discoverable by ``FoldedModel``."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=1,
                        dim_feedforward=hidden_dim * 4,
                        batch_first=True,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.head = nn.Linear(hidden_dim, vocab_size)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            hidden = self.embedding(tokens)
            for layer in self.layers:
                hidden = layer(hidden)
            return self.head(hidden)

    torch.manual_seed(seed)
    return TinyModel()


def _make_study(
    vocab_size: int = 100,
    hidden_dim: int = 64,
    num_layers: int = 4,
    seq_len: int = 16,
    seed: int = 1234,
) -> AblationStudy:
    """Build an ``AblationStudy`` around a tiny real Transformer.

    Args:
        vocab_size: Vocabulary shared by model, adapter and draft generator.
        hidden_dim: Hidden dimension of the underlying tiny model.
        num_layers: Layer count of the underlying tiny model.
        seq_len: Sequence length used for ablation inputs.
        seed: Torch seed for deterministic model initialization.

    Returns:
        An ``AblationStudy`` wired with a suffix-append draft generator.
    """
    raw = _make_model(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seed=seed,
    )
    adapter = FastDLLMAdapter(
        raw,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=1,
        vocab_size=vocab_size,
    )
    draft_generator = DraftGenerator(
        vocab_size=vocab_size,
        mode="suffix_append",
        # Only the suffix flips; the 8-token prompt prefix (half of
        # seq_len=16) is never resampled, so layer 0 keeps stable tokens.
        flip_ratio=0.5,
        prompt_length=8,
    )
    return AblationStudy(
        model=adapter,
        baseline=None,
        draft_generator=draft_generator,
        vocab_size=vocab_size,
        seq_len=seq_len,
        device="cpu",
    )


def test_default_draft_generator_is_suffix_append() -> None:
    """A study built without a draft generator defaults to suffix_append."""
    raw = _make_model()
    adapter = FastDLLMAdapter(
        raw, num_layers=4, hidden_dim=64, num_heads=1, vocab_size=100
    )
    study = AblationStudy(
        model=adapter,
        baseline=None,
        draft_generator=None,
        vocab_size=100,
        seq_len=16,
        device="cpu",
    )
    assert study.draft_generator.mode == "suffix_append"


def test_measure_folding_returns_real_per_layer_profile() -> None:
    """measure_folding reports measured per-layer ratios for folded layers."""
    study = _make_study(num_layers=4)
    m: FoldedMeasurement = study.measure_folding(tau=0.90)
    # The final layer (index 3) is never folded by FoldingScheduler.
    assert set(m.per_layer_stable.keys()) == {0, 1, 2}
    assert m.folded_layer_count == 3
    assert all(0.0 <= ratio <= 1.0 for ratio in m.per_layer_stable.values())
    assert m.baseline_tflops > m.actfold_tflops > 0
    assert 0.0 <= m.reduction_pct <= 100.0
    # suffix_append protects an 8-token prefix, so layer 0 is genuinely stable.
    assert m.stable_ratio > 0


def test_layerwise_disabled_layers_restricts_measured_layers() -> None:
    """disabled_layers restricts measured folding to the enabled layers."""
    study = _make_study(num_layers=4)
    m = study.measure_folding(tau=0.90, disabled_layers={1, 2, 3})
    assert set(m.per_layer_stable.keys()) == {0}
    assert m.folded_layer_count == 1


def test_layerwise_measured_differs_from_linear_extrapolation() -> None:
    """Layerwise sweeps report measured values distinct from linear estimates."""
    study = _make_study(num_layers=4)
    df = study.run_layerwise_folding(layer_ranges=[(0, 0), (0, 3)])

    row_first = df.loc[(df["start_layer"] == 0) & (df["end_layer"] == 0)].iloc[0]
    row_full = df.loc[(df["start_layer"] == 0) & (df["end_layer"] == 3)].iloc[0]
    assert int(row_first["folded_layer_count"]) == 1
    assert int(row_full["folded_layer_count"]) == 3

    # Real per-layer ratios are non-uniform, so at least one partial range
    # must differ from the old linear extrapolation.
    assert ((df["estimated_reduction_pct"] - df["linear_estimate_pct"]).abs() > 1e-9).any()
    # Folding more layers can only increase the measured reduction.
    assert row_full["estimated_reduction_pct"] >= row_first["estimated_reduction_pct"]


def test_cache_size_sweep_creates_real_eviction() -> None:
    """A cache budget of seq_len forces real eviction during the child pass."""
    study = _make_study(num_layers=4, seq_len=16)
    df = study.run_cache_size_impact(cache_sizes=[16, 32])

    # With budget == seq_len the child's layer-0 store evicts the parent's
    # layer-0 group, so deeper layers recompute without recording.
    count_16 = int(df.loc[df.cache_size == 16, "folded_layer_count"].iloc[0])
    count_32 = int(df.loc[df.cache_size == 32, "folded_layer_count"].iloc[0])
    assert count_16 == 1
    # With budget >= 2*seq_len no eviction occurs.
    assert count_32 == 3

    reduction_16 = float(df.loc[df.cache_size == 16, "tflops_reduction_pct"].iloc[0])
    reduction_32 = float(df.loc[df.cache_size == 32, "tflops_reduction_pct"].iloc[0])
    assert reduction_16 < reduction_32

    stable_16 = float(df.loc[df.cache_size == 16, "stable_ratio"].iloc[0])
    stable_32 = float(df.loc[df.cache_size == 32, "stable_ratio"].iloc[0])
    assert stable_16 <= stable_32


def test_cache_size_default_scales_with_seq_len() -> None:
    """Default cache sizes scale with seq_len so eviction actually happens."""
    study = _make_study(num_layers=4, seq_len=16)
    df = study.run_cache_size_impact()
    assert df["cache_size"].tolist() == [16, 32, 64]


def test_study_rejects_adapter_with_folded_model() -> None:
    """measure_folding refuses adapters that already carry a FoldedModel."""
    raw = _make_model()
    folded = FoldedModel(
        raw,
        ActivationCache(max_entries_per_layer=64, device="cpu"),
        SimilarityGate(tau=0.95),
    )
    adapter = FastDLLMAdapter(raw, num_layers=4, hidden_dim=64, folded_model=folded)
    study = AblationStudy(
        model=adapter,
        baseline=None,
        draft_generator=None,
        vocab_size=100,
        seq_len=16,
        device="cpu",
    )
    with pytest.raises(ValueError):
        study.measure_folding(tau=0.9)


def test_study_rejects_adapter_without_underlying_module() -> None:
    """measure_folding refuses adapters that expose no raw nn.Module."""

    class _NoUnderlyingAdapter:
        """Stub adapter with model metadata but no 'underlying_model'."""

        num_layers = 4
        hidden_dim = 64
        num_heads = 1
        vocab_size = 100

    study = AblationStudy(
        model=_NoUnderlyingAdapter(),  # type: ignore[arg-type]
        baseline=None,
        draft_generator=None,
        vocab_size=100,
        seq_len=16,
        device="cpu",
    )
    with pytest.raises(TypeError):
        study.measure_folding(tau=0.9)


def test_study_rejects_model_without_layer_stack() -> None:
    """measure_folding fails fast when no Transformer layer stack exists."""
    # A bare Linear has no discoverable layer ModuleList, so FoldedModel
    # cannot wrap anything and the measurement must not silently degrade.
    adapter = FastDLLMAdapter(
        nn.Linear(100, 64), num_layers=4, hidden_dim=64, num_heads=1, vocab_size=100
    )
    study = AblationStudy(
        model=adapter,
        baseline=None,
        draft_generator=None,
        vocab_size=100,
        seq_len=16,
        device="cpu",
    )
    with pytest.raises(RuntimeError):
        study.measure_folding(tau=0.9)


def test_threshold_sensitivity_measured_columns() -> None:
    """Threshold sensitivity exposes measured per-layer columns."""
    study = _make_study(num_layers=4)
    df = study.run_threshold_sensitivity(taus=[0.90, 0.95])

    expected_columns = {
        "tau",
        "stable_ratio",
        "folded_layer_count",
        "actfold_tflops",
        "baseline_tflops",
        "tflops_reduction_pct",
    }
    assert expected_columns.issubset(set(df.columns))
    assert len(df) == 2
    assert df["stable_ratio"].between(0, 1).all()
    assert (df["tflops_reduction_pct"] >= 0).all()
    # Full-range measurement folds exactly layers {0, 1, 2} of a 4-layer model.
    assert (df["folded_layer_count"] == 3).all()


def test_run_all_saves_measured_csvs(tmp_path: Path) -> None:
    """run_all persists the three measured CSV reports."""
    study = _make_study(num_layers=4)
    study.run_all(output_dir=tmp_path)

    for name in ("threshold_sensitivity", "layerwise_folding", "cache_size_impact"):
        assert (tmp_path / f"{name}.csv").exists()

    layerwise = pd.read_csv(tmp_path / "layerwise_folding.csv")
    # The linear extrapolation column is kept alongside the measured values.
    assert "linear_estimate_pct" in layerwise.columns


def test_t005_ablation_uses_manual_not_folded_model() -> None:
    """After the T005 switch, AblationStudy folds via ManualFoldedForward.

    The module-level ``FoldedModel`` import is removed from
    ``actfold.eval.ablation_study`` once ``measure_folding`` builds its
    internal stack with ``ManualFoldedForward``, and the measurement itself
    keeps reporting real per-layer stability.
    """
    import actfold.eval.ablation_study as ab_mod

    assert not hasattr(ab_mod, "FoldedModel")

    study = _make_study(num_layers=4)
    m: FoldedMeasurement = study.measure_folding(tau=0.90)
    assert m.per_layer_stable


def test_it412a_ablation_flops_budget_uses_config_geometry() -> None:
    """IT-412a (AR005 design 6.2): ``_flops_budget`` consumes real config geometry.

    The adapter wraps a DiffusionLLM stub whose geometry lives only on
    ``model.config`` (the real checkpoint path); the budget must equal the
    hand-computed estimate with the real SwiGLU/MoE geometry and differ from
    the 4h-MLP default.
    """
    from types import SimpleNamespace

    from actfold.models.base import DiffusionLLM
    from actfold.utils.flops_counter import count_diffusion_llm_flops, model_ffn_flops_kwargs

    class _GeometryOnlyModel(DiffusionLLM):
        """DiffusionLLM stub exposing dims and geometry via ``model.config``."""

        def __init__(self, config: SimpleNamespace) -> None:
            super().__init__("geometry-stub")
            self.model = SimpleNamespace(config=config)

        def forward(self, tokens, attention_mask=None, **kwargs):
            raise NotImplementedError

        def embed(self, tokens):
            raise NotImplementedError

        @property
        def num_layers(self):
            return 2

        @property
        def hidden_dim(self):
            return 16

        @property
        def num_heads(self):
            return 2

        @property
        def vocab_size(self):
            return 100

    config = SimpleNamespace(
        intermediate_size=48,
        hidden_act="silu",
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=32,
        num_hidden_layers=2,
        first_k_dense_replace=1,
    )
    adapter = FastDLLMAdapter(_GeometryOnlyModel(config))
    study = AblationStudy(model=adapter, baseline=None, vocab_size=100, seq_len=16)

    baseline_total, per_layer_reusable = study._flops_budget()

    kwargs = model_ffn_flops_kwargs(adapter)
    expected = count_diffusion_llm_flops(
        num_layers=2,
        hidden_dim=16,
        num_heads=2,
        seq_len=16,
        vocab_size=100,
        num_steps=1,
        reuse_ratio=0.0,
        **kwargs,
    )
    assert baseline_total == pytest.approx(expected.total_tflops, rel=1e-12)
    assert per_layer_reusable == pytest.approx(
        (expected.attention_tflops + expected.ffn_tflops) / 2, rel=1e-12
    )

    default_total = count_diffusion_llm_flops(
        num_layers=2,
        hidden_dim=16,
        num_heads=2,
        seq_len=16,
        vocab_size=100,
        num_steps=1,
        reuse_ratio=0.0,
    ).total_tflops
    assert baseline_total != pytest.approx(default_total)
