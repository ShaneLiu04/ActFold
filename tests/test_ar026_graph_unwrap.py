"""AR026 T003: tuple-unwrap caveat re-audit (UT-2603/UT-2604).

Contract: ``specs/changes/AR026-real-graph-e2e/design.md`` #3 FP3 / srs.md
§3.3 FR3 — the AGENTS #45 caveat claimed the graph static body does NOT
unwrap tuple layer returns; the code audit found the opposite (the static
body ``FoldedGraphRunner._static_forward`` calls ``layer._recompute_all``,
which unwraps tuple outputs per the #13 convention), so the caveat is stale
and the unwrap semantics are pinned here:

* UT-2603 (CPU): a tuple-returning mini decoder (layers return
  ``(hidden, None)`` — the real LLaDA remote-code shape) driven through the
  static body DIRECTLY — the runner's static buffers are allocated manually
  (mirroring ``capture()``'s allocation block) because ``capture()`` itself
  hard-requires CUDA; the parent prefill uses the real ``_prefill_parent``.
  The static-body logits must be BIT-EXACT against the eager MFF child
  forward on the same parent cache (CPU non-split parity, #41).
* UT-2604 (CUDA-gated): the same tuple stack through the FULL production
  path — ``ManualFoldedForward(use_cuda_graph=True)`` serial capture on the
  child call — must engage the graph runner and stay bit-exact against a
  graph-off MFF (the AR002 replay-parity contract, now exercised with
  tuple-returning layers). Skipped when CUDA is unavailable.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.cuda_graph import FoldedGraphRunner
from actfold.models.architecture_utils import ManualFoldedForward, detect_architecture

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TAU = 0.95
_EPS = 1e-8
_SEED = 2603
_VOCAB = 32
_HIDDEN = 16
_LAYERS = 2


class _TupleLayer(nn.Module):
    """LLaDA-remote-code-shaped block: returns ``(hidden, None)`` (#13)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        return (self.linear(x), None)


class _TupleLlamaModel(nn.Module):
    """Minimal LLaDA-layout decoder whose layers return tuples.

    ``embed_tokens -> tuple layers -> norm -> lm_head``; the layout is the
    standard ``model.model.*`` LLaMA shape so ``detect_architecture`` wires
    the embed/final-norm/head profile the way the production MFF does.
    """

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
        self.model.layers = nn.ModuleList([_TupleLayer(hidden_dim) for _ in range(num_layers)])
        self.model.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)[0]
        x = self.model.norm(x)
        return self.lm_head(x)


def _build(device: str) -> tuple[nn.Module, Any]:
    torch.manual_seed(_SEED)
    model = _TupleLlamaModel().to(device)
    model.eval()
    profile = detect_architecture(model)
    return model, profile


def _make_mff(model: nn.Module, device: str, use_cuda_graph: bool = False) -> ManualFoldedForward:
    cache = ActivationCache(max_entries_per_layer=64, device=device)
    gate = SimilarityGate(tau=_TAU, metric="cosine", eps=_EPS)
    return ManualFoldedForward(
        model,
        cache=cache,
        gate=gate,
        use_cuda_graph=use_cuda_graph,
        graph_capacity_ratio=0.5,
    )


def _parent_child_tokens(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(_SEED + 1)
    parent = torch.randint(0, _VOCAB, (2, 6), device=device)
    child = parent.clone()
    flat = child.reshape(-1)
    flat[3] = (flat[3] + 1) % _VOCAB
    flat[7] = (flat[7] + 1) % _VOCAB
    return parent, child


def _allocate_static_buffers(runner: FoldedGraphRunner, child: torch.Tensor) -> None:
    """Mirror ``capture()``'s allocation block (the CUDA-only face skipped).

    ``capture()`` hard-requires CUDA tokens (its stream/graph machinery), so
    the CPU pinning test allocates the five static buffers with the exact
    shapes/dtypes ``capture()`` uses and then pre-fills the parent through
    the REAL ``_prefill_parent`` — the static body under test is unchanged
    production code.
    """
    batch, seq_len = child.shape
    num_layers = len(runner._layers)
    runner._num_tokens = batch * seq_len
    runner._capacity = max(
        1, min(math.ceil(runner._capacity_ratio * runner._num_tokens), runner._num_tokens)
    )
    runner._tokens_static = child.clone()
    probe = runner._embed_fn(runner._tokens_static)
    hidden_dim = probe.shape[-1]
    runner._parent_static = torch.empty(
        num_layers + 1, batch, seq_len, hidden_dim, dtype=probe.dtype, device=child.device
    )
    runner._mask_buf = torch.empty(
        num_layers, batch, seq_len, dtype=torch.bool, device=child.device
    )
    runner._count_buf = torch.zeros(num_layers, dtype=torch.int32, device=child.device)
    runner._child_buf = torch.empty(
        num_layers, batch, seq_len, hidden_dim, dtype=probe.dtype, device=child.device
    )


def test_ut2603_static_body_unwraps_tuple_layers_cpu(device: str) -> None:
    """UT-2603: the graph static body inherits ``_recompute_all``'s unwrap.

    Tuple-returning layers driven through ``_static_forward`` directly must
    produce BIT-EXACT logits against the eager MFF child forward on the same
    parent cache — the stale AGENTS #45 claim ("the static body does NOT
    unwrap") is refuted on the exact code path a real capture would record.
    """
    model, profile = _build(device)
    parent, child = _parent_child_tokens(device)
    mff = _make_mff(model, device)
    with torch.no_grad():
        mff(parent, branch_id="parent")
        eager_out = mff(child, branch_id="child", parent_branch_id="parent")

    runner = FoldedGraphRunner(
        wrapped_layers=mff._wrapped_layers,
        embed_fn=profile.embed_module,
        final_norm_fn=profile.final_norm,
        head_fn=profile.head_module,
        cache=mff.cache,
        gate_tau=mff.gate.tau,
        gate_eps=mff.gate.eps,
        capacity_ratio=0.5,
        attention_mask_static=None,
    )
    _allocate_static_buffers(runner, child)
    assert runner._prefill_parent("parent"), "parent cache must be complete"

    with torch.no_grad():
        static_out = runner._static_forward()

    assert static_out.shape == eager_out.shape
    assert torch.equal(static_out, eager_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ut2604_production_capture_replay_tuple_layers_cuda() -> None:
    """UT-2604: full production graph capture with tuple layers (CUDA).

    ``ManualFoldedForward(use_cuda_graph=True)`` must capture the serial
    child window over tuple-returning layers (the runner install is asserted
    through ``graph_pool_stats``) and the graph-path child logits must be
    BIT-EXACT against a graph-off MFF on the same model — the AR002
    replay-parity contract, now exercised with the tuple layer shape (the
    caveat's "unexercised to date" half until a real checkpoint runs, AR026
    T006).
    """
    device = "cuda"
    model, profile = _build(device)
    parent, child = _parent_child_tokens(device)
    eager_mff = _make_mff(model, device)
    with torch.no_grad():
        eager_mff(parent, branch_id="parent")
        eager_out = eager_mff(child, branch_id="child", parent_branch_id="parent")

    graph_mff = _make_mff(model, device, use_cuda_graph=True)
    with torch.no_grad():
        graph_mff(parent, branch_id="parent")
        graph_out = graph_mff(child, branch_id="child", parent_branch_id="parent")

    serial_stats = graph_mff.graph_pool_stats["serial"]
    assert serial_stats["pool_size"] == 0
    assert graph_mff.graph_runner is not None, "serial graph capture must engage"
    assert torch.equal(graph_out, eager_out)
