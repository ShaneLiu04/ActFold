"""AR029 integration tests: folded-stack guard, protocol roles, factory/config."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import actfold.core.folded_transformer as folded_module
from actfold.core.activation_cache import ActivationCache
from actfold.core.cache_factory import make_activation_cache
from actfold.core.folded_transformer import FoldedTransformerLayer, _apply_cache_availability
from actfold.core.pyramid_cache import PyramidActivationCache
from actfold.core.pyramid_plan import build_pyramid_plan
from actfold.core.similarity_gate import SimilarityGate
from actfold.core.split_layer import SplitFoldedTransformerLayer
from actfold.eval.benchmark_runner import BenchmarkRunner
from actfold.models.architecture_utils import ManualFoldedForward
from actfold.utils.config_manager import load_config

HIDDEN = 8
VOCAB = 16


class TinyModel(nn.Module):
    """A real 2-layer decoder-shaped module (no mock outputs)."""

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.config = type(
            "Config",
            (),
            {"model_type": "llama", "vocab_size": VOCAB, "num_hidden_layers": num_layers},
        )()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.model.layers = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN) for _ in range(num_layers)])
        self.model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.model.lm_head(x)


def _pyramid(depth: int = 2, total: int = 64, **kwargs) -> PyramidActivationCache:
    plan = build_pyramid_plan(
        depth=depth,
        total_tokens=total,
        block_size=16,
        max_ratio=1.0,
        min_ratio=0.25,
        recent_blocks=0,
        anchor_blocks=1,
        anchor_stride=100,
        **kwargs,
    )
    return PyramidActivationCache(plan)


def _complete_root(cache, seq_len: int = 32, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    emb = torch.randn(1, seq_len, HIDDEN, generator=gen)
    par0 = torch.randn(1, seq_len, HIDDEN, generator=gen)
    par1 = torch.randn(1, seq_len, HIDDEN, generator=gen)
    cache.put("root", 0, {"embedding": emb, "ffn_out": par0})
    cache.put("root", 1, {"ffn_out": par1})
    return par0


def _layer(original: nn.Module, cache, idx: int = 1) -> FoldedTransformerLayer:
    return FoldedTransformerLayer(
        original_layer=original,
        cache=cache,
        gate=SimilarityGate(tau=0.9, metric="cosine"),
        layer_idx=idx,
    )


def _child_input(seq_len: int = 32, batch: int = 1, seed: int = 3) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq_len, HIDDEN, generator=gen)


def _stable_child(
    cache: PyramidActivationCache, layer_idx: int = 1, batch: int = 1
) -> torch.Tensor:
    """Child input equal to the parent's gate-layer activation (all-stable)."""
    if layer_idx == 0:
        base = cache.fetch("root", 0)["embedding"]
    else:
        base = cache.fetch("root", layer_idx - 1)["ffn_out"]
    return base.repeat(batch, 1, 1).clone()


def test_ut2935_target_layer_missing_matches_forced_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-2935: a target-only hole forces recompute equal to a forced reference."""
    original = nn.Linear(HIDDEN, HIDDEN)

    pyramid = _pyramid()
    _complete_root(pyramid)
    # Layer 0 (budget 48) keeps both blocks; layer 1 (budget 16) keeps the
    # anchor block only -> positions [16, 32) are missing at the target layer.
    assert not pyramid.availability("root", 1, 32)[16:].any()

    dense = ActivationCache(max_entries_per_layer=64)
    _complete_root(dense)

    # A gate-stable child (equal to the parent's layer-0 ffn_out) makes the
    # target-guard the only reason the tail recomputes.
    x = _stable_child(pyramid, 1)
    got_layer = _layer(original, pyramid)
    got = got_layer(x, branch_id="child", parent_branch_id="root")

    missing = torch.zeros(1, 32, dtype=torch.bool)
    missing[:, 16:] = True
    ref = _layer(original, dense)(
        x, branch_id="ref", parent_branch_id="root", forced_divergent=missing
    )
    assert torch.equal(got, ref)
    assert got[:, :16].any()

    # Mutation: with the target guard disabled the stored NaN leaks into the
    # merge (the gate alone would mark the tail stable).
    monkeypatch.setattr(
        folded_module,
        "_apply_cache_availability",
        lambda layer, parent, mask: (mask, False),
    )
    fresh = _pyramid()
    _complete_root(fresh)
    got_layer.cache = fresh
    unguarded = got_layer(x, branch_id="child", parent_branch_id="root")
    assert not torch.equal(unguarded, got)
    assert not torch.isfinite(unguarded).all()


def test_ut2936_uniform_missing_matches_forced_reference() -> None:
    """UT-2936: a uniform hole set over all layers equals the forced reference."""
    original = nn.Linear(HIDDEN, HIDDEN)
    plan = build_pyramid_plan(
        depth=2,
        total_tokens=2 * 16,
        block_size=16,
        max_ratio=1.0,
        min_ratio=1.0,
        recent_blocks=0,
        anchor_blocks=1,
        anchor_stride=100,
    )
    pyramid = PyramidActivationCache(plan)
    _complete_root(pyramid)
    assert not pyramid.availability("root", 0, 32)[16:].any()
    assert not pyramid.availability("root", 1, 32)[16:].any()

    dense = ActivationCache(max_entries_per_layer=64)
    _complete_root(dense)
    x = dense.fetch("root", 0)["embedding"].clone()
    got = _layer(original, pyramid)(x, branch_id="child", parent_branch_id="root")
    missing = torch.zeros(1, 32, dtype=torch.bool)
    missing[:, 16:] = True
    ref = _layer(original, dense)(
        x, branch_id="ref", parent_branch_id="root", forced_divergent=missing
    )
    assert torch.equal(got, ref)


def test_ut2938_profiler_sees_guarded_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2938: the profiler records the post-guard mask."""
    captured: list[torch.Tensor] = []
    original_record = folded_module.GLOBAL_STABILITY_PROFILER.record

    def spy(**kwargs):
        captured.append(kwargs["stable_mask"].clone())
        return original_record(**kwargs)

    monkeypatch.setattr(folded_module.GLOBAL_STABILITY_PROFILER, "record", spy)
    pyramid = _pyramid()
    par0 = _complete_root(pyramid)
    _layer(nn.Linear(HIDDEN, HIDDEN), pyramid)(par0, branch_id="child", parent_branch_id="root")
    assert captured
    assert not captured[-1][:, 16:].any()
    assert captured[-1][:, :16].any()


def test_ut2940b_layer_zero_guard_uses_target_ffn() -> None:
    """UT-2940b: layer 0 gates on embedding but guards on target ffn_out."""
    original = nn.Linear(HIDDEN, HIDDEN)
    pyramid = _pyramid(depth=1, total=16)
    gen = torch.Generator().manual_seed(7)
    acts = {
        "embedding": torch.randn(1, 32, HIDDEN, generator=gen),
        "ffn_out": torch.randn(1, 32, HIDDEN, generator=gen),
    }
    pyramid.put("root", 0, acts)
    assert not pyramid.availability("root", 0, 32)[16:].any()
    dense = ActivationCache(max_entries_per_layer=64)
    dense.put("root", 0, acts)
    x = acts["embedding"].clone()
    got = _layer(original, pyramid, idx=0)(x, branch_id="child", parent_branch_id="root")
    missing = torch.zeros(1, 32, dtype=torch.bool)
    missing[:, 16:] = True
    ref = _layer(original, dense, idx=0)(
        x, branch_id="ref", parent_branch_id="root", forced_divergent=missing
    )
    assert torch.equal(got, ref)


def test_ut2941_same_parent_batch_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2941: batched rows share the parent availability mask."""
    original = nn.Linear(HIDDEN, HIDDEN)
    pyramid = _pyramid()
    _complete_root(pyramid)
    dense = ActivationCache(max_entries_per_layer=64)
    _complete_root(dense)
    x = _stable_child(pyramid, 1, batch=2)
    got = _layer(original, pyramid)(
        x,
        branch_ids=["c1", "c2"],
        parent_branch_id="root",
    )
    missing = torch.zeros(2, 32, dtype=torch.bool)
    missing[:, 16:] = True
    ref = _layer(original, dense)(
        x,
        branch_ids=["c1", "c2"],
        parent_branch_id="root",
        forced_divergent=missing,
    )
    assert torch.equal(got, ref)

    monkeypatch.setattr(
        folded_module,
        "_apply_cache_availability",
        lambda layer, parent, mask: (mask, False),
    )
    fresh = _pyramid()
    _complete_root(fresh)
    unguarded = _layer(original, fresh)(
        x,
        branch_ids=["c1", "c2"],
        parent_branch_id="root",
    )
    assert not torch.equal(unguarded, got)


def test_ut2942_multi_parent_row_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2942: each multi-parent row uses its own parent availability."""
    original = nn.Linear(HIDDEN, HIDDEN)

    def fill(cache) -> None:
        for seed, pid in ((1, "r1"), (2, "r2")):
            gen = torch.Generator().manual_seed(seed)
            vals = {
                "embedding": torch.randn(1, 32, HIDDEN, generator=gen),
                "ffn_out": torch.randn(1, 32, HIDDEN, generator=gen),
            }
            cache.put(pid, 0, vals)
            v1 = torch.randn(1, 32, HIDDEN, generator=gen)
            cache.put(pid, 1, {"ffn_out": v1})

    pyramid = _pyramid()
    fill(pyramid)
    dense = ActivationCache(max_entries_per_layer=64)
    fill(dense)
    # The shared per-layer budget evicts r1's whole layer-1 entry first; r2
    # keeps its prefix and loses only the tail.  The two rows therefore have
    # DIFFERENT availability, which is exactly what the per-row guard must
    # honour.
    avail = torch.stack([pyramid.availability("r1", 1, 32), pyramid.availability("r2", 1, 32)])
    assert not avail[0].any()
    assert avail[1, :16].all() and not avail[1, 16:].any()
    x = torch.cat([dense.fetch("r1", 0)["ffn_out"], dense.fetch("r2", 0)["ffn_out"]], dim=0)
    got = _layer(original, pyramid)(
        x,
        branch_ids=["c1", "c2"],
        parent_branch_ids=["r1", "r2"],
    )
    missing = ~avail
    ref = _layer(original, dense)(
        x,
        branch_ids=["c1", "c2"],
        parent_branch_ids=["r1", "r2"],
        forced_divergent=missing,
    )
    assert torch.equal(got, ref)

    monkeypatch.setattr(
        folded_module,
        "_apply_cache_availability",
        lambda layer, parent, mask: (mask, False),
    )
    fresh = _pyramid()
    fill(fresh)
    unguarded = _layer(original, fresh)(
        x,
        branch_ids=["c1", "c2"],
        parent_branch_ids=["r1", "r2"],
    )
    assert not torch.equal(unguarded, got)


def test_ut2943_graph_pyramid_init_rejected() -> None:
    """UT-2943: graph folding with a pyramid cache is rejected at construction."""
    model = TinyModel()
    cache = _pyramid()
    with pytest.raises(ValueError, match="CUDA-graph"):
        ManualFoldedForward(
            model,
            cache=cache,
            gate=SimilarityGate(tau=0.9),
            use_cuda_graph=True,
        )


def test_ut2967_setter_swap_rejected_without_mutation() -> None:
    """UT-2967: swapping a pyramid cache into a graph-enabled MFF is rejected."""
    model = TinyModel()
    dense = ActivationCache(max_entries_per_layer=128)
    mff = ManualFoldedForward(model, cache=dense, gate=SimilarityGate(tau=0.9))
    mff.use_cuda_graph = True
    pyramid = _pyramid()
    with pytest.raises(ValueError, match="CUDA-graph"):
        mff.cache = pyramid
    assert mff.cache is dense


def test_ut2944_legacy_cache_guard_is_noop() -> None:
    """UT-2944: caches without availability keep the legacy path unchanged."""
    model = TinyModel()
    cache = ActivationCache(max_entries_per_layer=128)
    mff = ManualFoldedForward(model, cache=cache, gate=SimilarityGate(tau=0.9))
    tokens = torch.randint(0, VOCAB, (1, 16))
    with torch.no_grad():
        mff(tokens, branch_id="root")
        out = mff(tokens, branch_id="child", parent_branch_id="root")
    assert torch.isfinite(out).all()


def test_ut2953_factory_default_is_activation_cache() -> None:
    """UT-2953: the default factory path is unchanged."""
    assert isinstance(make_activation_cache(), ActivationCache)


def test_ut2954_factory_pyramid_requires_depth() -> None:
    """UT-2954: use_pyramid builds a pyramid cache with a real depth."""
    cache = make_activation_cache(use_pyramid=True, pyramid_depth=4)
    assert isinstance(cache, PyramidActivationCache)
    assert cache.plan.depth == 4
    with pytest.raises(ValueError):
        make_activation_cache(use_pyramid=True, pyramid_depth=0)


def test_ut2955_pyramid_takes_precedence() -> None:
    """UT-2955: use_pyramid wins over vectorized/chunked."""
    cache = make_activation_cache(
        use_pyramid=True,
        pyramid_depth=2,
        use_vectorized=True,
        use_chunked=True,
    )
    assert isinstance(cache, PyramidActivationCache)


def test_ut2956_yaml_pyramid_keys_have_no_unknown_warning(tmp_path) -> None:
    """UT-2956: YAML pyramid keys are consumed without unknown-key warnings."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "use_pyramid_cache: true",
                "pyramid_depth: 4",
                "pyramid_total_tokens: 2048",
                "pyramid_block_size: 16",
                "pyramid_max_ratio: 1.0",
                "pyramid_min_ratio: 0.25",
                "pyramid_schedule: linear",
                "pyramid_recent_blocks: 2",
                "pyramid_anchor_blocks: 1",
                "pyramid_anchor_stride: 2",
            ]
        ),
        encoding="utf-8",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load_config(path)
    assert config.use_pyramid_cache is True
    assert config.pyramid_depth == 4
    assert not [w for w in caught if "unknown" in str(w.message).lower()]


def _runner_config(**overrides):
    values = {
        "pyramid_depth": 0,
        "use_pyramid_cache": False,
        "use_cuda_graph": False,
        "pyramid_total_tokens": 2048,
        "pyramid_block_size": 16,
        "pyramid_max_ratio": 1.0,
        "pyramid_min_ratio": 0.25,
        "pyramid_schedule": "linear",
        "pyramid_recent_blocks": 2,
        "pyramid_anchor_blocks": 1,
        "pyramid_anchor_stride": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _bare_runner(config, diffusion_model=None):
    runner = object.__new__(BenchmarkRunner)
    runner.config = config
    runner._diffusion_model = diffusion_model
    runner.model = None
    return runner


def test_ut2957_runner_resolves_only_real_layer_counts() -> None:
    """UT-2957: the depth comes from config or the real model, never a guess."""
    assert _bare_runner(_runner_config(pyramid_depth=6))._resolve_pyramid_depth() == 6
    adapter = SimpleNamespace(num_layers=4)
    assert _bare_runner(_runner_config(), adapter)._resolve_pyramid_depth() == 4
    nested = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=3))
    assert _bare_runner(_runner_config(), nested)._resolve_pyramid_depth() == 3
    with pytest.raises(ValueError, match="real layer count"):
        _bare_runner(_runner_config())._resolve_pyramid_depth()


def test_ut2973_runner_pyramid_kwargs_and_graph_conflict() -> None:
    """UT-2973: the runner wires the real pyramid kwargs and rejects graph use."""
    assert _bare_runner(_runner_config())._pyramid_cache_kwargs() == {}
    runner = _bare_runner(_runner_config(use_pyramid_cache=True, pyramid_depth=4))
    kwargs = runner._pyramid_cache_kwargs()
    assert kwargs["use_pyramid"] is True
    assert kwargs["pyramid_depth"] == 4
    assert kwargs["pyramid_total_tokens"] == 2048
    graph_runner = _bare_runner(
        _runner_config(use_pyramid_cache=True, pyramid_depth=4, use_cuda_graph=True)
    )
    with pytest.raises(ValueError, match="use_cuda_graph"):
        graph_runner._pyramid_cache_kwargs()


def test_ut2958_graph_pyramid_rejected_at_both_layers() -> None:
    """UT-2958: both the folded stack and the runner refuse graph+pyramid."""
    model = TinyModel()
    with pytest.raises(ValueError, match="CUDA-graph"):
        ManualFoldedForward(
            model,
            cache=_pyramid(),
            gate=SimilarityGate(tau=0.9),
            use_cuda_graph=True,
        )
    runner = _bare_runner(
        _runner_config(use_pyramid_cache=True, pyramid_depth=2, use_cuda_graph=True)
    )
    with pytest.raises(ValueError, match="use_cuda_graph"):
        runner._pyramid_cache_kwargs()


def test_ut2974_one_dimensional_mask_keeps_rank() -> None:
    """UT-2974: a per-row [T] mask stays 1-D after the availability guard."""
    cache = _pyramid()
    _complete_root(cache)
    layer = SimpleNamespace(cache=cache, layer_idx=1)
    mask = torch.ones(32, dtype=torch.bool)
    guarded, applied = _apply_cache_availability(layer, "root", mask)
    assert applied is True
    assert guarded.shape == (32,)
    assert guarded[16:].sum().item() == 0


def test_ut2975_two_dimensional_mask_broadcasts_parent_mask() -> None:
    """UT-2975: a [1, T] parent mask broadcasts against an [N, T] child mask."""
    cache = _pyramid()
    _complete_root(cache)
    layer = SimpleNamespace(cache=cache, layer_idx=1)
    mask = torch.ones(3, 32, dtype=torch.bool)
    guarded, applied = _apply_cache_availability(layer, "root", mask)
    assert applied is True
    assert guarded.shape == (3, 32)
    assert guarded[:, 16:].sum().item() == 0


def test_ut2939_split_layer_inherits_guarded_forward() -> None:
    """UT-2939: the split-FFN subclass inherits the guarded forwarded path."""
    assert issubclass(SplitFoldedTransformerLayer, FoldedTransformerLayer)
    assert "forward" not in SplitFoldedTransformerLayer.__dict__


def test_ut2946_batched_full_budget_matches_dense() -> None:
    """UT-2946: batched full-budget output equals ActivationCache bit-for-bit."""
    original = nn.Linear(HIDDEN, HIDDEN)
    plan = build_pyramid_plan(
        depth=2,
        total_tokens=256,
        block_size=16,
        max_ratio=1.0,
        min_ratio=1.0,
        recent_blocks=0,
        anchor_blocks=1,
        anchor_stride=100,
    )
    pyramid = PyramidActivationCache(plan)
    _complete_root(pyramid)
    dense = ActivationCache(max_entries_per_layer=64)
    _complete_root(dense)
    x = _stable_child(pyramid, 1, batch=2)
    got = _layer(original, pyramid)(x, branch_ids=["c1", "c2"], parent_branch_id="root")
    ref = _layer(original, dense)(x, branch_ids=["c1", "c2"], parent_branch_id="root")
    assert torch.equal(got, ref)


def test_ut2949_stats_block_counts() -> None:
    """UT-2949: stats blocks equal the stored block counts per layer."""
    pyramid = _pyramid()
    _complete_root(pyramid)
    stats = pyramid.stats()
    assert stats["blocks"] == {0: 2, 1: 1}
    assert sum(stats["blocks"].values()) == 3


def test_ut2952_legacy_cache_guard_is_identity() -> None:
    """UT-2952: a legacy cache makes the guard a byte-for-byte no-op."""
    dense = ActivationCache(max_entries_per_layer=64)
    layer = SimpleNamespace(cache=dense, layer_idx=1)
    mask = torch.ones(1, 32, dtype=torch.bool)
    guarded, applied = _apply_cache_availability(layer, "root", mask)
    assert applied is False
    assert guarded is mask


def test_ut2957b_core_exports_pyramid_symbols() -> None:
    """UT-2957b: the public package exports the pyramid plan and cache."""
    import actfold.core as core

    assert core.PyramidActivationCache is PyramidActivationCache
    assert core.PyramidPlan is build_pyramid_plan(depth=1, total_tokens=16).__class__
    assert "PyramidActivationCache" in core.__all__
    assert "PyramidPlan" in core.__all__
    assert "build_pyramid_plan" in core.__all__


def test_ut2971_minimum_feasible_budget() -> None:
    """UT-2971: depth*block is the smallest accepted plan."""
    plan = build_pyramid_plan(depth=2, total_tokens=32, block_size=16)
    assert plan.layer_budgets == (16, 16)
    assert plan.total_budget_tokens == 32


def test_ut2972_pyramid_has_no_fetch_flat() -> None:
    """UT-2972: the sentinel cache deliberately exposes no fetch_flat."""
    cache = _pyramid()
    assert not hasattr(cache, "fetch_flat")
    assert getattr(cache, "fetch_flat", None) is None


def test_ut2934_serial_full_budget_matches_dense() -> None:
    """UT-2934: serial folded output under a full budget equals ActivationCache."""
    original = nn.Linear(HIDDEN, HIDDEN)
    plan = build_pyramid_plan(
        depth=2,
        total_tokens=256,
        block_size=16,
        max_ratio=1.0,
        min_ratio=1.0,
        recent_blocks=0,
        anchor_blocks=1,
        anchor_stride=100,
    )
    pyramid = PyramidActivationCache(plan)
    _complete_root(pyramid)
    dense = ActivationCache(max_entries_per_layer=64)
    _complete_root(dense)
    x = _stable_child(pyramid, 1)
    got = _layer(original, pyramid)(x, branch_id="child", parent_branch_id="root")
    ref = _layer(original, dense)(x, branch_id="child", parent_branch_id="root")
    assert torch.equal(got, ref)


def test_ut2940_serial_mff_guard_single_layer() -> None:
    """UT-2940: the serial MFF entry applies the target-layer guard."""
    model = TinyModel(num_layers=1)
    plan = build_pyramid_plan(depth=1, total_tokens=16, block_size=16, recent_blocks=0)
    pyramid = PyramidActivationCache(plan)
    tokens = torch.randint(0, VOCAB, (1, 32), generator=torch.Generator().manual_seed(11))
    folded = ManualFoldedForward(model, cache=pyramid, gate=SimilarityGate(tau=0.9))
    with torch.no_grad():
        folded(tokens, branch_id="root")
    assert not pyramid.availability("root", 0, 32)[16:].any()
    with torch.no_grad():
        got = folded(tokens, branch_id="child", parent_branch_id="root")

    dense = ActivationCache(max_entries_per_layer=64)
    ref_folded = ManualFoldedForward(model, cache=dense, gate=SimilarityGate(tau=0.9))
    with torch.no_grad():
        ref_folded(tokens, branch_id="root")
    missing = torch.zeros(1, 32, dtype=torch.bool)
    missing[:, 16:] = True
    with torch.no_grad():
        ref = ref_folded(
            tokens,
            branch_id="ref",
            parent_branch_id="root",
            forced_divergent=missing,
        )
    assert torch.equal(got, ref)
