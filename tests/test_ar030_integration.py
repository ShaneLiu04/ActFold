"""AR030 factory/config/integration tests (UT-3063..3068)."""

from __future__ import annotations

import warnings

import pytest
import torch

from actfold.core import (
    FoldPagePlan,
    PagedFoldCache,
    PageTable,
    PlanStaleError,
    RolloutBudgetError,
    RolloutSet,
    make_activation_cache,
)
from actfold.core.pyramid_plan import build_pyramid_plan
from actfold.utils.config_manager import ActFoldConfig, load_config


def test_ut3063_paged_takes_priority_over_other_caches() -> None:
    cache = make_activation_cache(
        device="cpu",
        use_paged=True,
        paged_depth=2,
        paged_total_tokens=256,
        paged_page_tokens=16,
        use_pyramid=True,
        pyramid_depth=2,
        pyramid_total_tokens=256,
        use_vectorized=True,
        use_chunked=True,
    )
    assert isinstance(cache, PagedFoldCache)
    assert cache.plan.depth == 2


def test_ut3064_paged_requires_real_depth() -> None:
    with pytest.raises(ValueError):
        make_activation_cache(device="cpu", use_paged=True, paged_depth=None)


def test_ut3065_paged_rejects_page_token_mismatch() -> None:
    plan = build_pyramid_plan(
        depth=2, total_tokens=256, block_size=16, max_ratio=1.0, min_ratio=1.0
    )
    with pytest.raises(ValueError):
        PagedFoldCache(plan, 32, device="cpu")


def test_ut3066_paged_is_graph_incompatible() -> None:
    assert PagedFoldCache.graph_incompatible is True
    assert PagedFoldCache.missing_value != PagedFoldCache.missing_value


def test_ut3067_config_consumes_all_paged_keys_without_warning(tmp_path) -> None:
    path = tmp_path / "paged.yaml"
    path.write_text(
        "\n".join(
            [
                "use_paged_cache: true",
                "paged_depth: 2",
                "paged_page_tokens: 16",
                "paged_total_tokens: 256",
                "paged_max_ratio: 1.0",
                "paged_min_ratio: 1.0",
                "paged_schedule: linear",
                "paged_recent_pages: 2",
                "paged_anchor_pages: 1",
                "paged_anchor_stride: 2",
            ]
        ),
        encoding="utf-8",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load_config(path)
    assert not [w for w in caught if "unknown config keys" in str(w.message)]
    assert config.use_paged_cache is True
    assert config.paged_depth == 2
    assert config.paged_page_tokens == 16


def test_ut3068_exports_are_public_and_default_off() -> None:
    for symbol in (
        PagedFoldCache,
        FoldPagePlan,
        PageTable,
        RolloutSet,
        PlanStaleError,
        RolloutBudgetError,
    ):
        assert symbol is not None
    config = ActFoldConfig()
    assert config.use_paged_cache is False
    cache = make_activation_cache(device="cpu")
    assert not isinstance(cache, PagedFoldCache)


def test_ut3097_paged_graph_rejected_at_mff_init_and_setter() -> None:
    """UT-3097: the folded stack refuses paged caches under CUDA-graph folding."""
    import torch.nn as nn

    from actfold.core import ActivationCache
    from actfold.core.similarity_gate import SimilarityGate
    from actfold.models.architecture_utils import ManualFoldedForward

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type(
                "Config", (), {"model_type": "llama", "vocab_size": 8, "num_hidden_layers": 2}
            )()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(8, 4)
            self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            x = self.model.embed_tokens(tokens)
            for layer in self.model.layers:
                x = layer(x)
            return x

    model = _Tiny()
    paged = make_activation_cache(
        device="cpu", use_paged=True, paged_depth=2, paged_total_tokens=256, paged_page_tokens=16
    )
    with pytest.raises(ValueError, match="CUDA-graph"):
        ManualFoldedForward(model, cache=paged, gate=SimilarityGate(tau=0.9), use_cuda_graph=True)
    # The cache setter validates before propagation: a failed swap keeps the
    # old cache in place.
    dense = ActivationCache(max_entries_per_layer=1024)
    mff = ManualFoldedForward(model, cache=dense, gate=SimilarityGate(tau=0.9), use_cuda_graph=True)
    with pytest.raises(ValueError, match="CUDA-graph"):
        mff.cache = paged
    assert mff.cache is dense


def test_ut3098_runner_paged_depth_and_graph_conflict() -> None:
    """UT-3098: the runner resolves only real layer counts and rejects graph use."""
    from types import SimpleNamespace

    from actfold.eval.benchmark_runner import BenchmarkRunner

    def _runner_config(**overrides):
        values = {
            "paged_depth": 0,
            "use_paged_cache": False,
            "use_cuda_graph": False,
            "paged_total_tokens": 2048,
            "paged_page_tokens": 16,
            "paged_max_ratio": 1.0,
            "paged_min_ratio": 0.25,
            "paged_schedule": "linear",
            "paged_recent_pages": 2,
            "paged_anchor_pages": 1,
            "paged_anchor_stride": 2,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _bare_runner(config, diffusion_model=None):
        runner = object.__new__(BenchmarkRunner)
        runner.config = config
        runner._diffusion_model = diffusion_model
        runner.model = None
        return runner

    assert _bare_runner(_runner_config(paged_depth=6))._resolve_paged_depth() == 6
    adapter = SimpleNamespace(num_layers=4)
    assert _bare_runner(_runner_config(), adapter)._resolve_paged_depth() == 4
    nested = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=3))
    assert _bare_runner(_runner_config(), nested)._resolve_paged_depth() == 3
    with pytest.raises(ValueError, match="real layer count"):
        _bare_runner(_runner_config())._resolve_paged_depth()
    assert _bare_runner(_runner_config())._paged_cache_kwargs() == {}
    kwargs = _bare_runner(_runner_config(use_paged_cache=True, paged_depth=4))._paged_cache_kwargs()
    assert kwargs["use_paged"] is True
    assert kwargs["paged_depth"] == 4
    assert kwargs["paged_total_tokens"] == 2048
    with pytest.raises(ValueError, match="use_cuda_graph"):
        _bare_runner(
            _runner_config(use_paged_cache=True, paged_depth=4, use_cuda_graph=True)
        )._paged_cache_kwargs()


def _tokens() -> dict[str, torch.Tensor]:
    return {"ffn_out": torch.randn(1, 16, 4)}
