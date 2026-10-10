"""Red-phase tests for AR002/T007 (+T008 budget semantics): ``FoldedGraphRunner``.

Contract: ``specs/changes/AR002-m4-graph-capture/design.md`` §4.3.4 — a CUDA
graph capture/replay runner for the Manual folded verification forward with
static buffers, a single per-step count readback, and divergent-budget
validation (D == C passes, D == C + 1 fails).

The runner class is imported lazily inside each test (and inside the shared
helpers, which every runner test reaches) so that the Red state — the module
``actfold.core.cuda_graph`` not existing yet — fails tests individually with
``ModuleNotFoundError`` instead of aborting module collection.

Self-contained per repo convention: the small LLaMA-layout decoder below is
modeled on ``tests/test_architecture_utils.py::LlamaLikeModel`` /
``tests/test_split_layer.py::LlamaLikeLayer`` but its layers carry the
``post_attention_layernorm -> mlp`` chain that
``SplitFoldedTransformerLayer`` hooks (required for the capture design
§4.2.2: split hooks engaged via ``layer._split_state``) and a maskable
attention so ``attention_mask`` forms are numerically observable.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.models.architecture_utils import ManualFoldedForward, detect_architecture
from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

_EPS = 1e-8
_TAU = 0.95
_VOCAB = 100
_HIDDEN = 32
_LAYERS = 2
_SEED = 2024


def _load_runner_cls() -> Any:
    """Import the runner lazily (fails per-test in the Red state)."""
    from actfold.core.cuda_graph import FoldedGraphRunner

    return FoldedGraphRunner


# ---------------------------------------------------------------------------
# Self-contained synthetic decoder (LlamaLikeModel layout)
# ---------------------------------------------------------------------------


class _MaskableAttention(nn.Module):
    """Token-wise attention stand-in that consumes the attention mask.

    The mask is applied MULTIPLICATIVELY: an additive per-token constant
    would be cancelled by the final LayerNorm's shift invariance and thus be
    unobservable at the logits level.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        out = hidden_states * 0.5
        if attention_mask is not None:
            out = out * attention_mask.to(out.dtype).unsqueeze(-1)
        return out, None


class _LlamaLikeLayer(nn.Module):
    """Pre-norm decoder layer with a detectable ``post_attention_layernorm->mlp`` chain."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = _MaskableAttention()
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class LlamaLikeModel(nn.Module):
    """Mock LLaMA-layout decoder: embed -> layers -> model.norm -> lm_head."""

    def __init__(
        self,
        vocab_size: int = _VOCAB,
        hidden_dim: int = _HIDDEN,
        num_layers: int = _LAYERS,
    ) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": vocab_size})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model.layers = nn.ModuleList([_LlamaLikeLayer(hidden_dim) for _ in range(num_layers)])
        self.model.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------


def _build_model(
    device: str,
    num_layers: int = _LAYERS,
    hidden_dim: int = _HIDDEN,
    seed: int = _SEED,
) -> tuple[nn.Module, Any]:
    """Deterministically build the decoder and its architecture profile."""
    torch.manual_seed(seed)
    model = LlamaLikeModel(_VOCAB, hidden_dim, num_layers).to(device)
    model.eval()
    profile = detect_architecture(model)
    return model, profile


def _make_mff(
    model: nn.Module,
    device: str,
    gate: SimilarityGate | None = None,
    scheduler: FoldingScheduler | None = None,
    cache_budget: int = 64,
    use_cuda_graph: bool = False,
    graph_capacity_ratio: float = 0.5,
) -> ManualFoldedForward:
    """Fresh ManualFoldedForward (own cache + gate) over the shared model.

    ``cache_budget`` is the per-layer token budget; scenarios with
    ``batch * seq > 64`` must raise it so the parent activations are not
    truncated by eviction. ``use_cuda_graph`` / ``graph_capacity_ratio`` opt
    in to the AR002/T008 graph wiring (constructor flags only; the
    forward-path wiring itself is what the T008 tests below pin).
    """
    cache = ActivationCache(max_entries_per_layer=cache_budget, device=device)
    if gate is None:
        gate = SimilarityGate(tau=_TAU, metric="cosine", eps=_EPS)
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        scheduler=scheduler,
        split_layers=True,
        split_min_tokens=1,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=graph_capacity_ratio,
    )


def _make_runner(
    profile: Any,
    mff: ManualFoldedForward,
    capacity_ratio: float = 0.5,
    attention_mask_static: torch.Tensor | None = None,
) -> Any:
    """Construct a FoldedGraphRunner with the exact §4.3.4 constructor."""
    runner_cls = _load_runner_cls()
    return runner_cls(
        wrapped_layers=mff._wrapped_layers,
        embed_fn=profile.embed_module,
        final_norm_fn=profile.final_norm,
        head_fn=profile.head_module,
        cache=mff.cache,
        gate_tau=mff.gate.tau,
        gate_eps=mff.gate.eps,
        capacity_ratio=capacity_ratio,
        attention_mask_static=attention_mask_static,
    )


def _captured_runner(
    model: nn.Module,
    profile: Any,
    parent_tokens: torch.Tensor,
    child_tokens: torch.Tensor,
    capacity_ratio: float = 0.5,
    attention_mask: torch.Tensor | None = None,
    cache_budget: int = 64,
) -> tuple[Any, ManualFoldedForward]:
    """Parent eager pass (populates the cache) + child capture; returns (runner, mff)."""
    mff = _make_mff(model, parent_tokens.device.type, cache_budget=cache_budget)
    with torch.no_grad():
        mff(parent_tokens, branch_id="parent", attention_mask=attention_mask)
    runner = _make_runner(
        profile, mff, capacity_ratio=capacity_ratio, attention_mask_static=attention_mask
    )
    with torch.no_grad():
        runner.capture(child_tokens, branch_id="child", parent_branch_id="parent")
    return runner, mff


def _change_tokens(tokens: torch.Tensor, positions: list[int], offset: int = 1) -> torch.Tensor:
    """Clone ``tokens`` and change the flat ``positions`` to a different id."""
    child = tokens.clone()
    flat = child.reshape(-1)
    for pos in positions:
        flat[pos] = (flat[pos] + offset) % _VOCAB
    return child


# ---------------------------------------------------------------------------
# Red trigger: module surfacing
# ---------------------------------------------------------------------------


def test_module_import_surfacing() -> None:
    """``actfold.core.cuda_graph.FoldedGraphRunner`` must be importable."""
    from actfold.core.cuda_graph import FoldedGraphRunner

    assert callable(FoldedGraphRunner)


# ---------------------------------------------------------------------------
# Constructor validation (CPU tensors are fine for this check)
# ---------------------------------------------------------------------------


def test_capacity_ratio_validation() -> None:
    """capacity_ratio must satisfy 0 < r <= 1 (ValueError otherwise)."""
    _load_runner_cls()
    model, profile = _build_model("cpu")
    mff = _make_mff(model, "cpu")
    with pytest.raises(ValueError):
        _make_runner(profile, mff, capacity_ratio=0.0)
    with pytest.raises(ValueError):
        _make_runner(profile, mff, capacity_ratio=1.5)


# ---------------------------------------------------------------------------
# capture() preconditions (CUDA required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_requires_cuda_and_cosine() -> None:
    """capture() rejects CPU inputs, non-cosine gates, and scheduler layers."""
    _load_runner_cls()
    device = "cuda"

    # (a) CPU model/layers/tokens -> RuntimeError.
    cpu_model, cpu_profile = _build_model("cpu")
    cpu_mff = _make_mff(cpu_model, "cpu")
    cpu_runner = _make_runner(cpu_profile, cpu_mff)
    cpu_tokens = torch.randint(0, _VOCAB, (1, 8))
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            cpu_runner.capture(cpu_tokens, branch_id="child", parent_branch_id="parent")

    model, profile = _build_model(device)
    tokens = torch.randint(0, _VOCAB, (1, 8), device=device)

    # (b) CUDA model but a layer gate with metric="l2" -> RuntimeError.
    gate_l2 = SimilarityGate(tau=_TAU, metric="l2", eps=_EPS)
    mff_l2 = _make_mff(model, device, gate=gate_l2)
    with torch.no_grad():
        mff_l2(tokens, branch_id="parent")
    runner_l2 = _make_runner(profile, mff_l2)
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            runner_l2.capture(tokens, branch_id="child", parent_branch_id="parent")

    # (c) CUDA model but layers constructed with a FoldingScheduler -> RuntimeError.
    scheduler = FoldingScheduler(base_tau=_TAU, num_layers=_LAYERS, num_steps=2)
    mff_sched = _make_mff(model, device, scheduler=scheduler)
    with torch.no_grad():
        mff_sched(tokens, branch_id="parent")
    runner_sched = _make_runner(profile, mff_sched)
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            runner_sched.capture(tokens, branch_id="child", parent_branch_id="parent")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_missing_parent_cache_raises() -> None:
    """capture() with an empty parent cache must raise RuntimeError."""
    _load_runner_cls()
    device = "cuda"
    model, profile = _build_model(device)
    mff = _make_mff(model, device)
    runner = _make_runner(profile, mff)
    tokens = torch.randint(0, _VOCAB, (1, 8), device=device)
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            runner.capture(tokens, branch_id="child", parent_branch_id="parent")


# ---------------------------------------------------------------------------
# replay() semantics
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_replay_missing_parent_returns_none() -> None:
    """replay() returns None (no raise) when the parent cache is incomplete."""
    _load_runner_cls()
    device = "cuda"
    model, profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (1, 8), device=device)
    child_tokens = _change_tokens(parent_tokens, [3])
    runner, mff = _captured_runner(model, profile, parent_tokens, child_tokens)
    mff.cache.clear_branch("parent")
    with torch.no_grad():
        out = runner.replay(child_tokens, parent_branch_id="parent", branch_id="child")
    assert out is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_replay_shape_mismatch_raises() -> None:
    """replay() raises ValueError when the token shape differs from capture."""
    _load_runner_cls()
    device = "cuda"
    model, profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (1, 8), device=device)
    child_tokens = _change_tokens(parent_tokens, [3])
    runner, _ = _captured_runner(model, profile, parent_tokens, child_tokens)
    bad_tokens = torch.randint(0, _VOCAB, (1, 9), device=device)
    with pytest.raises(ValueError):
        with torch.no_grad():
            runner.replay(bad_tokens, parent_branch_id="parent", branch_id="child")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_replay_bit_exact_vs_eager() -> None:
    """Replay matches the eager folded forward and is deterministic."""
    _load_runner_cls()
    device = "cuda"
    batch, seq, hidden, num_layers = 1, 8, _HIDDEN, _LAYERS
    model, profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (batch, seq), device=device)
    child_tokens = _change_tokens(parent_tokens, [3, 6])  # tiny divergence: D=2 << C=4

    # Eager folded reference (fresh cache/gate).
    mff_eager = _make_mff(model, device)
    with torch.no_grad():
        mff_eager(parent_tokens, branch_id="parent")
        logits_eager = mff_eager(child_tokens, branch_id="child", parent_branch_id="parent").clone()

    # Graph runner over a fresh cache/gate with the same weights.
    runner, _ = _captured_runner(model, profile, parent_tokens, child_tokens, capacity_ratio=0.5)

    # White-box static-buffer contract (design §4.3.4 additional facts).
    assert tuple(runner.tokens_static.shape) == (batch, seq)
    assert tuple(runner.parent_static.shape) == (num_layers + 1, batch, seq, hidden)
    assert tuple(runner.mask_buf.shape) == (num_layers, batch, seq)
    assert runner.mask_buf.dtype == torch.bool
    assert tuple(runner.count_buf.shape) == (num_layers,)
    assert runner.count_buf.dtype == torch.int32
    assert tuple(runner.child_buf.shape) == (num_layers, batch, seq, hidden)
    assert isinstance(runner.capacity, int)
    assert runner.capacity == math.ceil(0.5 * batch * seq)

    with torch.no_grad():
        out1 = runner.replay(child_tokens, parent_branch_id="parent", branch_id="child")
    assert isinstance(out1, torch.Tensor)
    out1_copy = out1.clone()  # replay returns the static buffer; caller must clone
    with torch.no_grad():
        out2 = runner.replay(child_tokens, parent_branch_id="parent", branch_id="child")

    # Padded capacity-C gathers vs eager exact-D gathers: allclose, not equal.
    assert torch.allclose(out1_copy, logits_eager, atol=1e-4, rtol=1e-4)
    # Determinism: two consecutive replays of identical inputs are bit-equal.
    assert torch.equal(out1_copy, out2)
    assert runner.validate_budgets() is True
    assert runner.budget_exceeded is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_replay_mask_none_and_static_forms() -> None:
    """attention_mask_static=None and a fixed mask both match their eager runs."""
    _load_runner_cls()
    device = "cuda"
    batch, seq = 1, 8
    model, profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (batch, seq), device=device)
    child_tokens = _change_tokens(parent_tokens, [3])

    # (a) attention_mask_static=None vs eager with mask=None.
    mff_none = _make_mff(model, device)
    with torch.no_grad():
        mff_none(parent_tokens, branch_id="parent")
        logits_eager_none = mff_none(
            child_tokens, branch_id="child", parent_branch_id="parent"
        ).clone()
    runner_none, _ = _captured_runner(
        model, profile, parent_tokens, child_tokens, attention_mask=None
    )
    with torch.no_grad():
        out_none = runner_none.replay(child_tokens, parent_branch_id="parent", branch_id="child")
    assert isinstance(out_none, torch.Tensor)
    assert torch.allclose(out_none, logits_eager_none, atol=1e-4, rtol=1e-4)

    # (b) fixed [1, T] 0.5 mask vs eager with the same mask (a ones mask is
    # numerically identical to None under multiplicative masking).
    mask = 0.5 * torch.ones(1, seq, device=device)
    mff_mask = _make_mff(model, device)
    with torch.no_grad():
        mff_mask(parent_tokens, branch_id="parent", attention_mask=mask)
        logits_eager_mask = mff_mask(
            child_tokens, branch_id="child", parent_branch_id="parent", attention_mask=mask
        ).clone()
    runner_mask, _ = _captured_runner(
        model, profile, parent_tokens, child_tokens, attention_mask=mask
    )
    with torch.no_grad():
        out_mask = runner_mask.replay(child_tokens, parent_branch_id="parent", branch_id="child")
    assert isinstance(out_mask, torch.Tensor)
    assert torch.allclose(out_mask, logits_eager_mask, atol=1e-4, rtol=1e-4)

    # Sanity: the mask actually flows through the layer (references differ).
    assert not torch.allclose(logits_eager_none, logits_eager_mask)


# ---------------------------------------------------------------------------
# validate_budgets() semantics
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_validate_budgets_boundary() -> None:
    """D == C passes, D == C + 1 is rejected, all-stable passes again."""
    _load_runner_cls()
    device = "cuda"
    batch, seq = 1, 8
    num_tokens = batch * seq
    model, profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (batch, seq), device=device)
    child_tokens = parent_tokens.clone()  # all-stable child -> D = 0

    runner, _ = _captured_runner(model, profile, parent_tokens, child_tokens, capacity_ratio=0.5)
    capacity = runner.capacity
    assert capacity == math.ceil(0.5 * num_tokens) == 4

    with torch.no_grad():
        out = runner.replay(child_tokens, parent_branch_id="parent", branch_id="child")
    assert out is not None

    # count_buf holds per-layer stable counts: D_l = N - count_l.
    runner.count_buf.fill_(num_tokens - capacity)  # D == C -> passes
    assert runner.validate_budgets() is True
    runner.count_buf.fill_(num_tokens - capacity - 1)  # D == C + 1 -> rejected
    assert runner.validate_budgets() is False
    assert runner.budget_exceeded is True
    runner.count_buf.fill_(num_tokens)  # all-stable restored -> passes
    assert runner.validate_budgets() is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fully_divergent_child_exceeds_budget() -> None:
    """D = N at every layer (N > C): replay returns logits, validation fails."""
    _load_runner_cls()
    device = "cuda"
    model, profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (1, 8), device=device)
    child_tokens = (parent_tokens + 37) % _VOCAB  # every token id differs

    runner, _ = _captured_runner(model, profile, parent_tokens, child_tokens, capacity_ratio=0.5)
    with torch.no_grad():
        out = runner.replay(child_tokens, parent_branch_id="parent", branch_id="child")
    assert isinstance(out, torch.Tensor)
    assert runner.validate_budgets() is False
    assert runner.budget_exceeded is True


# ---------------------------------------------------------------------------
# Verification-loop integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_two_step_verification_loop() -> None:
    """Parent -> child1 (graph) -> publish -> child2 (graph, parent=child1)."""
    _load_runner_cls()
    device = "cuda"
    num_layers = _LAYERS
    model, profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (1, 8), device=device)
    child1_tokens = _change_tokens(parent_tokens, [2])
    child2_tokens = _change_tokens(child1_tokens, [5])

    # Eager two-step folded reference.
    mff_eager = _make_mff(model, device)
    with torch.no_grad():
        mff_eager(parent_tokens, branch_id="parent")
        mff_eager(child1_tokens, branch_id="child1", parent_branch_id="parent")
        logits_ref = mff_eager(child2_tokens, branch_id="child2", parent_branch_id="child1").clone()

    # Graph loop: parent pass eagerly, child1 via capture + replay.
    mff_graph = _make_mff(model, device)
    with torch.no_grad():
        mff_graph(parent_tokens, branch_id="parent")
    runner = _make_runner(profile, mff_graph, capacity_ratio=0.5)
    with torch.no_grad():
        runner.capture(child1_tokens, branch_id="child1", parent_branch_id="parent")
        out1 = runner.replay(child1_tokens, parent_branch_id="parent", branch_id="child1")
    assert out1 is not None
    assert runner.validate_budgets() is True

    # Caller-side publish of child1 layer outputs into the cache:
    # layer 0 stores {"ffn_out", "embedding"} (embedding = embed(child1)),
    # deeper layers store {"ffn_out"} only.
    with torch.no_grad():
        child1_embedding = profile.embed_module(child1_tokens)
    for layer_idx in range(num_layers):
        activations: dict[str, torch.Tensor] = {"ffn_out": runner.child_buf[layer_idx].clone()}
        if layer_idx == 0:
            activations["embedding"] = child1_embedding
        mff_graph.cache.put(branch_id="child1", layer_idx=layer_idx, activations=activations)

    # Step 2: child2 with parent_branch_id="child1".
    with torch.no_grad():
        out2 = runner.replay(child2_tokens, parent_branch_id="child1", branch_id="child2")
    assert isinstance(out2, torch.Tensor)
    assert runner.validate_budgets() is True
    assert torch.allclose(out2, logits_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_path_in_graph() -> None:
    """B*T >= 1024 dispatches the Triton fused gate inside the captured graph."""
    from actfold.core import fused_ops

    _load_runner_cls()
    device = "cuda"
    num_layers, hidden = 2, 64
    model, profile = _build_model(device, num_layers=num_layers, hidden_dim=hidden)
    batch, seq = 2, 512  # B*T = 1024 >= _FUSED_GATE_MIN_TOKENS
    parent_tokens = torch.randint(0, _VOCAB, (batch, seq), device=device)
    child_tokens = _change_tokens(parent_tokens, [100, 500, 900])  # D=3 << C=256

    mff_eager = _make_mff(model, device, cache_budget=4096)
    with torch.no_grad():
        mff_eager(parent_tokens, branch_id="parent")
        logits_eager = mff_eager(child_tokens, branch_id="child", parent_branch_id="parent").clone()

    runner, _ = _captured_runner(
        model, profile, parent_tokens, child_tokens, capacity_ratio=0.25, cache_budget=4096
    )
    with torch.no_grad():
        out = runner.replay(child_tokens, parent_branch_id="parent", branch_id="child")
    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, logits_eager, atol=1e-3, rtol=1e-3)
    assert runner.validate_budgets() is True
    # No silent Triton fallback was triggered during capture/replay.
    assert fused_ops._TRITON_GATE_DISABLED is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_no_host_readback_during_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """replay() performs at most ONE host readback (the budget count check)."""
    _load_runner_cls()
    device = "cuda"
    model, profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (1, 8), device=device)
    child_tokens = _change_tokens(parent_tokens, [3])
    runner, _ = _captured_runner(model, profile, parent_tokens, child_tokens)

    counts = {"tolist": 0, "int": 0, "item": 0}
    orig_tolist = torch.Tensor.tolist
    orig_int = torch.Tensor.__int__
    orig_item = torch.Tensor.item

    def _count_tolist(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        counts["tolist"] += 1
        return orig_tolist(self, *args, **kwargs)  # type: ignore[arg-type]

    def _count_int(self: torch.Tensor, *args: object, **kwargs: object) -> int:
        counts["int"] += 1
        return orig_int(self, *args, **kwargs)  # type: ignore[arg-type,return-value]

    def _count_item(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        counts["item"] += 1
        return orig_item(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.Tensor, "tolist", _count_tolist)
    monkeypatch.setattr(torch.Tensor, "__int__", _count_int, raising=False)
    monkeypatch.setattr(torch.Tensor, "item", _count_item)

    with torch.no_grad():
        out = runner.replay(child_tokens, parent_branch_id="parent", branch_id="child")

    total = counts["tolist"] + counts["int"] + counts["item"]
    assert out is not None
    assert total <= 1, (
        f"replay() must perform at most one host readback (the budget count "
        f"readback), observed {total}: tolist={counts['tolist']}, "
        f"int={counts['int']}, item={counts['item']}"
    )


# --- AR002/T008: ManualFoldedForward graph wiring ---


def _count_user_warnings(record: list[warnings.WarningMessage]) -> int:
    """Count ``UserWarning`` entries in a ``catch_warnings(record=True)`` log."""
    return sum(1 for entry in record if issubclass(entry.category, UserWarning))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_t008_parent_pass_eager_and_cache_populated() -> None:
    """Parent passes stay eager under ``use_cuda_graph=True`` and fill the cache.

    The T008 wiring only ever tries the graph path for child passes, so a
    parent pass must (a) return the raw model logits, (b) leave
    ``graph_runner`` at ``None`` (no capture on parent passes), and (c)
    populate the complete parent cache (layer-0 ``embedding`` plus every
    layer's ``ffn_out``) that a later child capture requires.
    """
    _load_runner_cls()
    device = "cuda"
    batch, seq = 1, 8
    model, _profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (batch, seq), device=device)

    mff = _make_mff(model, device, use_cuda_graph=True, graph_capacity_ratio=0.5)
    with torch.no_grad():
        logits = mff(parent_tokens, branch_id="parent")
        raw_logits = model(parent_tokens)

    assert torch.allclose(logits, raw_logits, atol=1e-4, rtol=1e-4)
    assert mff.graph_runner is None
    for layer_idx in range(_LAYERS):
        entry = mff.cache.fetch(branch_id="parent", layer_idx=layer_idx)
        assert "ffn_out" in entry
        assert tuple(entry["ffn_out"].shape) == (batch, seq, _HIDDEN)
        if layer_idx == 0:
            assert "embedding" in entry
            assert tuple(entry["embedding"].shape) == (batch, seq, _HIDDEN)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_t008_graph_child_pass_end_to_end() -> None:
    """The wired graph child pass matches eager folding, publishes, profiles.

    After the parent eager pass, the first child call lazily creates the
    runner (at most one instance), replays, and publishes the child branch
    exactly like the eager path would: layer-0 ``embedding`` + per-layer
    ``ffn_out`` cache entries and per-layer stability records in the global
    profiler. A second child with different changed positions replays the
    SAME runner.
    """
    _load_runner_cls()
    device = "cuda"
    batch, seq = 1, 8
    model, profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (batch, seq), device=device)
    child_tokens = _change_tokens(parent_tokens, [3, 6])
    child2_tokens = _change_tokens(parent_tokens, [1, 5])

    # Eager folded references (fresh cache/gate, same shared model weights).
    mff_eager = _make_mff(model, device)
    with torch.no_grad():
        mff_eager(parent_tokens, branch_id="parent")
        ref1 = mff_eager(child_tokens, branch_id="child", parent_branch_id="parent").clone()
        ref2 = mff_eager(child2_tokens, branch_id="child2", parent_branch_id="parent").clone()
        child_embedding = profile.embed_module(child_tokens)

    mff = _make_mff(model, device, use_cuda_graph=True, graph_capacity_ratio=0.5)
    with torch.no_grad():
        mff(parent_tokens, branch_id="parent")

    GLOBAL_STABILITY_PROFILER.reset_branch("child")
    with torch.no_grad():
        out1 = mff(child_tokens, branch_id="child", parent_branch_id="parent")

    assert mff.graph_runner is not None
    assert torch.allclose(out1, ref1, atol=1e-4, rtol=1e-4)

    entry0 = mff.cache.fetch(branch_id="child", layer_idx=0)
    assert "embedding" in entry0
    assert "ffn_out" in entry0
    assert torch.allclose(entry0["embedding"], child_embedding, atol=1e-6)
    for layer_idx in range(_LAYERS):
        entry = mff.cache.fetch(branch_id="child", layer_idx=layer_idx)
        assert "ffn_out" in entry
        assert tuple(entry["ffn_out"].shape) == (batch, seq, _HIDDEN)

    prof = GLOBAL_STABILITY_PROFILER.get_profile("child")
    assert prof is not None
    assert prof.parent_branch_id == "parent"
    assert len(prof.layer_stats) == _LAYERS
    for expected_idx, stats in enumerate(prof.layer_stats):
        assert stats.layer_idx == expected_idx
        assert stats.metric == "cosine"
        assert stats.tau_used == _TAU
        assert 0.0 <= stats.stable_ratio <= 1.0

    runner = mff.graph_runner
    with torch.no_grad():
        out2 = mff(child2_tokens, branch_id="child2", parent_branch_id="parent")
    assert mff.graph_runner is runner
    assert torch.allclose(out2, ref2, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_t008_budget_exceeded_discards_and_warns() -> None:
    """D > C discards the replay output, warns once, and recomputes eagerly."""
    _load_runner_cls()
    device = "cuda"
    model, _profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (1, 8), device=device)
    div_a = (parent_tokens + 37) % _VOCAB  # every token id differs: D = N = 8
    div_b = (parent_tokens + 61) % _VOCAB

    mff_eager = _make_mff(model, device)
    with torch.no_grad():
        mff_eager(parent_tokens, branch_id="parent")
        ref_a = mff_eager(div_a, branch_id="div_a", parent_branch_id="parent").clone()
        ref_b = mff_eager(div_b, branch_id="div_b", parent_branch_id="parent").clone()

    # capacity_ratio=0.25 -> C = ceil(0.25 * 8) = 2 < D = 8 at every layer:
    # the replay output is corrupt and must be discarded in favour of an
    # eager recompute of this step.
    mff = _make_mff(model, device, use_cuda_graph=True, graph_capacity_ratio=0.25)
    with torch.no_grad():
        mff(parent_tokens, branch_id="parent")

    with pytest.warns(UserWarning):
        with torch.no_grad():
            out_a = mff(div_a, branch_id="div_a", parent_branch_id="parent")
    assert mff.graph_runner is not None
    assert torch.allclose(out_a, ref_a, atol=1e-4, rtol=1e-4)

    # The budget warning is one-time: a second fully-divergent call stays
    # silent but still returns the correct (eager) result.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with torch.no_grad():
            out_b = mff(div_b, branch_id="div_b", parent_branch_id="parent")
    assert _count_user_warnings(record) == 0
    assert torch.allclose(out_b, ref_b, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_t008_shape_mismatch_falls_back_eager() -> None:
    """A tokens shape change after capture warns once, then stays silent."""
    _load_runner_cls()
    device = "cuda"
    model, _profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (1, 8), device=device)
    child_tokens = _change_tokens(parent_tokens, [3])

    mff = _make_mff(model, device, use_cuda_graph=True, graph_capacity_ratio=0.5)
    with torch.no_grad():
        mff(parent_tokens, branch_id="parent")
        mff(child_tokens, branch_id="child", parent_branch_id="parent")
    assert mff.graph_runner is not None

    # A batch-2 call no longer matches the pinned (B, T) = (1, 8) shape.
    # parent_branch_id=None keeps the eager fallback on the clean full
    # recompute path (no shape-mismatched parent cache is consulted), so the
    # only expected side effect of the graph wiring is the one-time warning.
    wide_tokens = torch.randint(0, _VOCAB, (2, 8), device=device)
    with pytest.warns(UserWarning):
        with torch.no_grad():
            wide_out = mff(wide_tokens, branch_id="parent_b2")
    with torch.no_grad():
        wide_raw = model(wide_tokens)
    assert tuple(wide_out.shape) == (2, 8, _VOCAB)
    assert torch.allclose(wide_out, wide_raw, atol=1e-4, rtol=1e-4)

    # Second shape-mismatched call: silent, still the correct eager output.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with torch.no_grad():
            wide_out2 = mff(wide_tokens, branch_id="parent_b3")
    assert _count_user_warnings(record) == 0
    assert torch.allclose(wide_out2, wide_raw, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_t008_scheduler_and_cpu_degrade_with_warning() -> None:
    """Scheduler or CPU inputs degrade the graph path with a one-time warning."""
    _load_runner_cls()
    device = "cuda"
    model, _profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (1, 8), device=device)
    child_tokens = _change_tokens(parent_tokens, [3])
    child2_tokens = _change_tokens(parent_tokens, [5])

    # (a) CUDA model, but a FoldingScheduler is attached: the graph path is
    # disabled forever after a single UserWarning on the first child call.
    eager_scheduler = FoldingScheduler(base_tau=_TAU, num_layers=_LAYERS, num_steps=2)
    graph_scheduler = FoldingScheduler(base_tau=_TAU, num_layers=_LAYERS, num_steps=2)
    mff_eager = _make_mff(model, device, scheduler=eager_scheduler)
    mff_sched = _make_mff(model, device, scheduler=graph_scheduler, use_cuda_graph=True)
    with torch.no_grad():
        mff_eager(parent_tokens, branch_id="parent")
        ref1 = mff_eager(child_tokens, branch_id="child", parent_branch_id="parent").clone()
        ref2 = mff_eager(child2_tokens, branch_id="child2", parent_branch_id="parent").clone()

        mff_sched(parent_tokens, branch_id="parent")
    with pytest.warns(UserWarning):
        with torch.no_grad():
            out1 = mff_sched(child_tokens, branch_id="child", parent_branch_id="parent")
    assert mff_sched.graph_runner is None
    assert torch.allclose(out1, ref1, atol=1e-4, rtol=1e-4)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with torch.no_grad():
            out2 = mff_sched(child2_tokens, branch_id="child2", parent_branch_id="parent")
    assert _count_user_warnings(record) == 0
    assert torch.allclose(out2, ref2, atol=1e-4, rtol=1e-4)

    # (b) CPU model with use_cuda_graph=True: same one-time degradation.
    cpu_model, _cpu_profile = _build_model("cpu")
    cpu_parent = torch.randint(0, _VOCAB, (1, 8))
    cpu_child = _change_tokens(cpu_parent, [3])
    cpu_child2 = _change_tokens(cpu_parent, [6])

    cpu_eager = _make_mff(cpu_model, "cpu")
    cpu_graph = _make_mff(cpu_model, "cpu", use_cuda_graph=True)
    with torch.no_grad():
        cpu_eager(cpu_parent, branch_id="parent")
        cpu_ref1 = cpu_eager(cpu_child, branch_id="child", parent_branch_id="parent").clone()
        cpu_ref2 = cpu_eager(cpu_child2, branch_id="child2", parent_branch_id="parent").clone()

        cpu_graph(cpu_parent, branch_id="parent")
    with pytest.warns(UserWarning):
        with torch.no_grad():
            cpu_out1 = cpu_graph(cpu_child, branch_id="child", parent_branch_id="parent")
    assert cpu_graph.graph_runner is None
    assert torch.allclose(cpu_out1, cpu_ref1, atol=1e-4, rtol=1e-4)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with torch.no_grad():
            cpu_out2 = cpu_graph(cpu_child2, branch_id="child2", parent_branch_id="parent")
    assert _count_user_warnings(record) == 0
    assert torch.allclose(cpu_out2, cpu_ref2, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_t008_mask_pinned_at_capture() -> None:
    """The capture-time attention mask is pinned; other masks fall back silently.

    The mask passed on the first (capturing) child call is baked into the
    runner. A later graph-eligible call with a DIFFERENT mask object must
    fall back to eager SILENTLY (no warning) and genuinely use the new mask.
    """
    _load_runner_cls()
    device = "cuda"
    batch, seq = 1, 8
    model, _profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (batch, seq), device=device)
    child_tokens = _change_tokens(parent_tokens, [3])
    mask_a = 0.5 * torch.ones(1, seq, device=device)
    mask_b = torch.zeros(1, seq, device=device)

    # Eager references: parent warmed with mask A; child with mask A vs B.
    mff_eager = _make_mff(model, device)
    with torch.no_grad():
        mff_eager(parent_tokens, branch_id="parent", attention_mask=mask_a)
        ref_a = mff_eager(
            child_tokens, branch_id="child", parent_branch_id="parent", attention_mask=mask_a
        ).clone()
        ref_b = mff_eager(
            child_tokens, branch_id="child_b", parent_branch_id="parent", attention_mask=mask_b
        ).clone()

    mff = _make_mff(model, device, use_cuda_graph=True, graph_capacity_ratio=0.5)
    with torch.no_grad():
        mff(parent_tokens, branch_id="parent", attention_mask=mask_a)
        out_a = mff(
            child_tokens, branch_id="child", parent_branch_id="parent", attention_mask=mask_a
        )

    assert mff.graph_runner is not None
    assert torch.allclose(out_a, ref_a, atol=1e-4, rtol=1e-4)

    # Different mask object: silent eager fallback that really consumes B.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with torch.no_grad():
            out_b = mff(
                child_tokens,
                branch_id="child_b",
                parent_branch_id="parent",
                attention_mask=mask_b,
            )
    assert _count_user_warnings(record) == 0
    assert torch.allclose(out_b, ref_b, atol=1e-4, rtol=1e-4)
    # Sanity: mask B genuinely changes the output (B was used, not pinned A).
    assert not torch.allclose(out_b, ref_a)


# ---------------------------------------------------------------------------
# EX-002: capture failure permanently disables the graph path (no retry)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ex002_capture_failure_permanently_disables_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An injected capture failure warns once, runs eager, and never retries."""
    _load_runner_cls()
    from actfold.core import cuda_graph

    device = "cuda"
    model, _profile = _build_model(device)
    parent_tokens = torch.randint(0, _VOCAB, (1, 8), device=device)
    child_tokens = parent_tokens.clone()
    child_tokens[0, 3] = (child_tokens[0, 3] + 1) % _VOCAB
    child2_tokens = parent_tokens.clone()
    child2_tokens[0, 5] = (child2_tokens[0, 5] + 1) % _VOCAB

    mff_eager = _make_mff(model, device)
    with torch.no_grad():
        mff_eager(parent_tokens, branch_id="parent")
        ref = mff_eager(child_tokens, branch_id="child", parent_branch_id="parent").clone()
        ref2 = mff_eager(child2_tokens, branch_id="child2", parent_branch_id="parent").clone()

    capture_calls = {"count": 0}

    def _failing_capture(self: Any, *args: object, **kwargs: object) -> None:
        capture_calls["count"] += 1
        raise RuntimeError("injected capture failure")

    monkeypatch.setattr(cuda_graph.FoldedGraphRunner, "capture", _failing_capture)

    mff = _make_mff(model, device, use_cuda_graph=True)
    with torch.no_grad():
        mff(parent_tokens, branch_id="parent")

    with pytest.warns(UserWarning, match="capture failed"):
        with torch.no_grad():
            out = mff(child_tokens, branch_id="child", parent_branch_id="parent")
    # That step still returns the correct eager result; the loop is unbroken.
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    assert mff.graph_runner is None
    assert mff._graph_capture_failed is True
    assert capture_calls["count"] == 1

    # A second eligible child call must NOT retry the capture (EX-002) and
    # must stay silent (the failure already warned once).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with torch.no_grad():
            out2 = mff(child2_tokens, branch_id="child2", parent_branch_id="parent")
    assert capture_calls["count"] == 1
    assert torch.allclose(out2, ref2, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# BS-001: folded_generate end-to-end with the graph path (T008 integration)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bs001_folded_generate_graph_end_to_end() -> None:
    """folded_generate + Manual(use_cuda_graph=True) matches eager tokens.

    ``folded_generate`` appends one token per step, so every child is one
    token longer than its parent: with variable-length folding unsupported
    (README limitation #7) the parent activations can never be reused and
    every step recomputes (the same semantics as the legacy FoldedModel
    path, where folding silently never activated). The generated tokens
    must equal the all-eager run and generation must not be interrupted or
    crash on the shape mismatch.
    """
    from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter
    from actfold.speculative.folded_generation import folded_generate

    _load_runner_cls()
    device = "cuda"
    model, _profile = _build_model(device)
    prompt = torch.tensor([[3, 17, 42, 7]], device=device)

    mff_eager = _make_mff(model, device)
    adapter_eager = FastDLLMAdapter(
        model, folded_model=mff_eager, num_layers=_LAYERS, hidden_dim=_HIDDEN
    )
    adapter_eager.underlying_model.eval()
    result_eager = folded_generate(adapter_eager, prompt, max_new_tokens=4, folded_model=mff_eager)

    mff_graph = _make_mff(model, device, use_cuda_graph=True)
    adapter_graph = FastDLLMAdapter(
        model, folded_model=mff_graph, num_layers=_LAYERS, hidden_dim=_HIDDEN
    )
    adapter_graph.underlying_model.eval()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result_graph = folded_generate(
            adapter_graph, prompt, max_new_tokens=4, folded_model=mff_graph
        )

    # Identical generated tokens across all >= 4 steps; loop unbroken.
    assert result_graph.tokens.shape == (1, 8)
    assert torch.equal(result_graph.tokens, result_eager.tokens)
    assert result_graph.num_folded_steps == 4
    assert result_eager.num_folded_steps == 4
    # AR length growth means the graph never captures here (no warnings).
    assert mff_graph.graph_runner is None
    assert _count_user_warnings(record) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bs001_fixed_shape_verification_loop_via_adapter() -> None:
    """BS-001 fixed-shape form: adapter-routed verification loop uses the graph.

    The diffusion verification workload repeats the SAME-shape folded child
    forward each step (unlike ``folded_generate``'s AR growth). Routing
    through ``FastDLLMAdapter`` (which forwards ``branch_id`` to the folded
    model) must capture on the first step and replay afterwards, with logits
    matching the all-eager run and the loop uninterrupted.
    """
    from actfold.speculative.fast_dllm_adapter import FastDLLMAdapter

    _load_runner_cls()
    device = "cuda"
    model, _profile = _build_model(device)
    tokens = torch.randint(0, _VOCAB, (1, 8), device=device)

    def _step_tokens(step: int) -> torch.Tensor:
        child = tokens.clone()
        child[0, (3 * step + 1) % 8] = (child[0, (3 * step + 1) % 8] + 1) % _VOCAB
        return child

    # Eager reference: same chained branches through an all-eager Manual.
    mff_eager = _make_mff(model, device)
    adapter_eager = FastDLLMAdapter(
        model, folded_model=mff_eager, num_layers=_LAYERS, hidden_dim=_HIDDEN
    )
    adapter_eager.underlying_model.eval()
    with torch.no_grad():
        adapter_eager.forward(tokens, branch_id="vp0")
        ref_logits = []
        parent = "vp0"
        for step in range(4):
            branch = f"vref{step}"
            ref_logits.append(
                adapter_eager.forward(
                    _step_tokens(step), branch_id=branch, parent_branch_id=parent
                ).clone()
            )
            parent = branch

    # Graph path: capture on the first child step, replay afterwards.
    mff_graph = _make_mff(model, device, use_cuda_graph=True)
    adapter_graph = FastDLLMAdapter(
        model, folded_model=mff_graph, num_layers=_LAYERS, hidden_dim=_HIDDEN
    )
    adapter_graph.underlying_model.eval()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with torch.no_grad():
            adapter_graph.forward(tokens, branch_id="vp0")
            parent = "vp0"
            for step in range(4):
                branch = f"vgraph{step}"
                out = adapter_graph.forward(
                    _step_tokens(step), branch_id=branch, parent_branch_id=parent
                )
                assert torch.allclose(out, ref_logits[step], atol=1e-4, rtol=1e-4)
                parent = branch

    assert mff_graph.graph_runner is not None
    assert _count_user_warnings(record) == 0
