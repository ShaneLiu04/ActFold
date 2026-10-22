"""AR031 wiring/config/export tests (UT-3145..3152)."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.layer_budget import LayerRecomputeBudgetPolicy
from actfold.eval.benchmark_runner import BenchmarkRunner
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.utils.config_manager import ActFoldConfig, load_config


class _Tiny(nn.Module):
    """Synthetic two-layer decoder (the UT-3097 fixture pattern)."""

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


def _tiny_model(seed: int = 3151) -> _Tiny:
    torch.manual_seed(seed)
    return _Tiny()


def _fresh_mff(model: nn.Module, **kwargs: Any) -> ManualFoldedForward:
    """Fresh MFF with its own cache + gate; ``kwargs`` carry the budget face."""
    return ManualFoldedForward(
        model,
        cache=ActivationCache(max_entries_per_layer=1024),
        gate=SimilarityGate(tau=0.9),
        **kwargs,
    )


def _runner_config(**overrides: Any) -> SimpleNamespace:
    """SimpleNamespace config carrying every field ``_build_folded_model`` reads."""
    values: dict[str, Any] = {
        "max_entries_per_layer": 1024,
        "use_chunked_cache": False,
        "cache_chunk_size": 64,
        "use_vectorized_cache": False,
        "use_paged_cache": False,
        "use_pyramid_cache": False,
        "tau": 0.9,
        "metric": "cosine",
        "use_cuda_graph": False,
        "num_steps": 4,
        "use_split_layers": False,
        "split_min_tokens": 512,
        "graph_capacity_ratio": 0.5,
        "graph_adaptive_capacity": False,
        "graph_runner_pool_size": 0,
        "layer_budget_enabled": False,
        "layer_budget_ratio": 0.15,
        "layer_budget_warmup": 5,
        "layer_budget_window": 10,
        "uncertainty_validation_order": False,
        "uncertainty_margin": 0.9,
        "uncertainty_margin_gap": 0.05,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _bare_runner(config: SimpleNamespace, diffusion_model: Any = None) -> BenchmarkRunner:
    """Bare BenchmarkRunner (the UT-3098 pattern), no ``__init__`` side effects."""
    runner = object.__new__(BenchmarkRunner)
    runner.config = config
    runner._diffusion_model = diffusion_model
    runner.model = None
    runner.device = "cpu"
    return runner


def test_ut3145_config_consumes_seven_budget_keys_without_warning(tmp_path: Path) -> None:
    """UT-3145: load_config consumes the seven AR031 keys with no unknown-key warning."""
    path = tmp_path / "ar031_budget.yaml"
    path.write_text(
        "\n".join(
            [
                "layer_budget_enabled: true",
                "layer_budget_ratio: 0.25",
                "layer_budget_warmup: 3",
                "layer_budget_window: 7",
                "uncertainty_validation_order: true",
                "uncertainty_margin: 0.8",
                "uncertainty_margin_gap: 0.1",
            ]
        ),
        encoding="utf-8",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load_config(path)
    assert not [w for w in caught if "unknown config keys" in str(w.message)]
    assert config.layer_budget_enabled is True
    assert config.layer_budget_ratio == 0.25
    assert config.layer_budget_warmup == 3
    assert config.layer_budget_window == 7
    assert config.uncertainty_validation_order is True
    assert config.uncertainty_margin == 0.8
    assert config.uncertainty_margin_gap == 0.1
    defaults = ActFoldConfig()
    assert defaults.layer_budget_enabled is False
    assert defaults.layer_budget_ratio == 0.15
    assert defaults.layer_budget_warmup == 5
    assert defaults.layer_budget_window == 10
    assert defaults.uncertainty_validation_order is False
    assert defaults.uncertainty_margin == 0.9
    assert defaults.uncertainty_margin_gap == 0.05


def test_ut3146_public_exports() -> None:
    """UT-3146: the AR031 policy/validator classes are public package exports."""
    import actfold.core as core_module
    import actfold.speculative as speculative_module
    from actfold.core import LayerRecomputeBudgetPolicy as CorePolicy
    from actfold.speculative import UncertaintyRolloutValidator as SpecValidator

    assert CorePolicy is not None
    assert SpecValidator is not None
    assert "LayerRecomputeBudgetPolicy" in core_module.__all__
    assert "UncertaintyRolloutValidator" in speculative_module.__all__


def test_ut3147_default_config_zero_regression() -> None:
    """UT-3147: default config keeps every AR031 knob off / at its frozen default."""
    defaults = ActFoldConfig()
    assert defaults.layer_budget_enabled is False
    assert defaults.layer_budget_ratio == 0.15
    assert defaults.layer_budget_warmup == 5
    assert defaults.layer_budget_window == 10
    assert defaults.uncertainty_validation_order is False
    assert defaults.uncertainty_margin == 0.9
    assert defaults.uncertainty_margin_gap == 0.05
    runner = _bare_runner(_runner_config(layer_budget_enabled=False))
    assert runner._layer_budget_policy() is None
    mff = _fresh_mff(_tiny_model())
    assert mff._budget_policy is None


def test_ut3148_runner_builds_policy_and_attaches_mff() -> None:
    """UT-3148: enabled runners build the policy, resolve depth explicitly, wire the MFF."""
    config = _runner_config(
        layer_budget_enabled=True,
        layer_budget_ratio=0.25,
        layer_budget_warmup=3,
        layer_budget_window=7,
    )
    explicit_owner = SimpleNamespace(num_layers=4, config=SimpleNamespace(num_hidden_layers=3))
    assert _bare_runner(config, explicit_owner)._resolve_budget_depth() == 4
    with pytest.raises(ValueError, match="real layer count"):
        _bare_runner(config, SimpleNamespace())._resolve_budget_depth()

    model = _tiny_model()
    adapter = SimpleNamespace(num_layers=2, model=model)
    mff = _bare_runner(config, adapter)._build_folded_model(adapter)
    assert mff is not None
    assert isinstance(mff._budget_policy, LayerRecomputeBudgetPolicy)
    assert mff._budget_policy.num_layers == 2

    mismatched = SimpleNamespace(num_layers=3, model=model)
    with pytest.raises(ValueError):
        _bare_runner(config, mismatched)._build_folded_model(mismatched)


def test_ut3149_runner_default_builds_no_policy() -> None:
    """UT-3149: disabled runners build no policy; enabled-but-unresolvable raises."""
    runner = _bare_runner(_runner_config(layer_budget_enabled=False))
    assert runner._layer_budget_policy() is None

    model = _tiny_model()
    adapter = SimpleNamespace(num_layers=2, model=model)
    mff = _bare_runner(_runner_config(layer_budget_enabled=False), adapter)._build_folded_model(
        adapter
    )
    assert mff is not None
    assert mff._budget_policy is None

    unresolvable = _bare_runner(_runner_config(layer_budget_enabled=True), SimpleNamespace())
    with pytest.raises(ValueError, match="real layer count"):
        unresolvable._layer_budget_policy()


def test_ut3150_scores_forward_bit_equivalence() -> None:
    """UT-3150: forward is bitwise ``scores > tau`` per metric; NaN stays divergent."""
    gen = torch.Generator().manual_seed(3150)
    h_child = torch.randn(2, 9, 5, generator=gen)
    h_parent = h_child.clone()
    h_parent[:, 6:] += torch.randn(2, 3, 5, generator=gen) * 2.0
    h_child[0, 3] = float("nan")
    for metric in ("cosine", "l2", "pearson"):
        gate = SimilarityGate(tau=0.7, metric=metric)
        scores = gate.scores(h_child, h_parent)
        mask = gate.forward(h_child, h_parent)
        assert torch.equal(mask, scores > gate.tau)
        assert not bool(mask[0, 3])
        assert bool(mask.any())
        assert not bool(mask.all())


def test_ut3151_budget_none_forward_byte_identical() -> None:
    """UT-3151: ``budget_policy=None`` (and a never-binding policy) stay byte-identical."""
    model = _tiny_model()
    gen = torch.Generator().manual_seed(3151)
    parent = torch.randint(0, 8, (1, 8), generator=gen)
    child = parent.clone()
    child[0, 4] = (child[0, 4] + 1) % 8
    child[0, 6] = (child[0, 6] + 3) % 8

    mff_a = _fresh_mff(model)
    mff_b = _fresh_mff(model, budget_policy=None)
    mff_c = _fresh_mff(
        model, budget_policy=LayerRecomputeBudgetPolicy(budget_ratio=1.0, num_layers=2)
    )
    assert mff_a._budget_policy is None
    assert mff_b._budget_policy is None
    assert mff_c._budget_policy is not None

    outputs: list[torch.Tensor] = []
    for mff in (mff_a, mff_b, mff_c):
        with torch.no_grad():
            mff(parent, branch_id="3151-parent")
            outputs.append(mff(child, branch_id="3151-child", parent_branch_id="3151-parent"))
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


def test_ut3152_cross_device_observation_rejected() -> None:
    """UT-3152: cross-device observations raise ValueError, never a silent move."""
    policy = LayerRecomputeBudgetPolicy(budget_ratio=0.5, num_layers=2)
    policy.observe_layer(0, torch.tensor(3.0, dtype=torch.float32, device="cpu"), 8)
    with pytest.raises(ValueError, match="device"):
        policy.observe_layer(0, torch.tensor(1.0, dtype=torch.float32, device="meta"), 8)
    policy.observe_layer(0, torch.tensor(1.0, dtype=torch.float32, device="cpu"), 8)


def test_ut3167_position_level_forced_one_dim() -> None:
    """UT-3167 (renumbered from the review-round collision with the T006
    validator face's UT-3164): the #43 production forced tensor is ``[T]``.

    The eager budget face broadcasts a 1-D forced to the mask shape at the
    cap choke point (the frozen ``apply_layer_budget`` same-shape contract
    holds); the 1-D form is bit-identical to the explicit ``[1, T]`` form
    (a silently-ignored or mis-broadcast forced would break the equality).
    """
    model = _tiny_model(seed=3164)
    gen = torch.Generator().manual_seed(3164)
    parent = torch.randint(0, 8, (1, 8), generator=gen)
    child = parent.clone()
    child[0, 5] = (child[0, 5] + 1) % 8
    child[0, 6] = (child[0, 6] + 3) % 8
    forced = torch.zeros(8, dtype=torch.bool)
    forced[5:] = True

    outs = []
    for form in (forced, forced.unsqueeze(0)):
        mff = _fresh_mff(
            model,
            budget_policy=LayerRecomputeBudgetPolicy(budget_ratio=0.15, num_layers=2),
        )
        with torch.no_grad():
            mff(parent, branch_id="3164-parent")
            outs.append(
                mff(
                    child,
                    branch_id="3164-child",
                    parent_branch_id="3164-parent",
                    forced_divergent=form,
                )
            )
    assert torch.equal(outs[0], outs[1])


def test_ut3168_budget_requires_exact_similarity_gate() -> None:
    """UT-3168: the eager budget face rejects non-exact-type gates (design E9).

    The cap semantics are defined on ``scores > tau``; a top-k gate
    (``AdaptiveQuantileGate``, a SimilarityGate subclass) has no threshold
    mask, so the combination fails fast at the scores face with a
    ``ValueError`` naming the gate type — never a silently wrong cap.
    """
    from actfold.core.adaptive_gate import AdaptiveQuantileGate

    model = _tiny_model(seed=3168)
    gen = torch.Generator().manual_seed(3168)
    parent = torch.randint(0, 8, (1, 8), generator=gen)
    child = parent.clone()
    child[0, 5] = (child[0, 5] + 1) % 8

    mff = ManualFoldedForward(
        model,
        cache=ActivationCache(max_entries_per_layer=1024),
        gate=AdaptiveQuantileGate(target_stable_ratio=0.5),
        budget_policy=LayerRecomputeBudgetPolicy(budget_ratio=0.15, num_layers=2),
    )
    with torch.no_grad():
        mff(parent, branch_id="3168-parent")
        with pytest.raises(ValueError, match="SimilarityGate"):
            mff(child, branch_id="3168-child", parent_branch_id="3168-parent")
