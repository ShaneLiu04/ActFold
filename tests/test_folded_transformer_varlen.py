"""Tests for variable-length prefix folding (AR003)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.core.folded_transformer import FoldedTransformerLayer
from actfold.core.split_layer import SplitFoldedTransformerLayer

B = 2
T_PARENT = 4  # T_p: cached parent prefix length, strictly shorter than T_CHILD
T_CHILD = 7  # T_c: child sequence length (suffix length = T_CHILD - T_PARENT >= 1)
HIDDEN_DIM = 16


class DummyLayer(nn.Module):
    """Identity-like layer with small transformation for testing."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.linear(hidden_states)


def _seed_parent_cache(
    cache: ActivationCache,
    h_parent: torch.Tensor,
    ffn_seed: torch.Tensor,
) -> None:
    """Hand-seed the parent cache at layer 0 without running a parent forward.

    ``ffn_seed`` is an arbitrary tensor (``torch.randn``) intentionally
    different from any ``layer(...)`` output so that folding (reusing the
    parent FFN) and full recompute (taking the layer output) are
    distinguishable in assertions.
    """
    cache.put(
        branch_id="parent",
        layer_idx=0,
        activations={"embedding": h_parent, "ffn_out": ffn_seed},
    )


def _seed_parent_cache_branch(
    cache: ActivationCache,
    branch_id: str,
    h_parent: torch.Tensor,
    ffn_seed: torch.Tensor,
) -> None:
    """Hand-seed a branch cache entry at layer 0 (branch-id aware)."""
    cache.put(
        branch_id=branch_id,
        layer_idx=0,
        activations={"embedding": h_parent, "ffn_out": ffn_seed},
    )


def test_varlen_prefix_mixed_reference(device: str) -> None:
    """UT-201: a mixed-stability var-len prefix matches the reference semantics.

    Given ``T_p < T_c`` and a partially stable prefix (some positions equal to
    the cached parent embedding, others far away), the folded child output
    must be bit-exact against the reference synthesis semantics:
    ``where(concat(gate(child[:, :T_p], h_parent), False_suffix),
    cat([parent_ffn, layer(child)[:, T_p:]]), layer(child))``.
    """
    layer = DummyLayer(HIDDEN_DIM).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    h_parent = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    ffn_seed = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    _seed_parent_cache(cache, h_parent, ffn_seed)

    # Child: prefix positions 0..1 identical to the parent embedding (stable),
    # prefix positions 2..3 heavily perturbed (divergent), suffix arbitrary
    # (always divergent by construction).
    child = torch.randn(B, T_CHILD, HIDDEN_DIM, device=device)
    child[:, :T_PARENT] = h_parent
    child[:, T_PARENT - 2 : T_PARENT] += 100.0 * torch.randn(
        B, 2, HIDDEN_DIM, device=device
    )

    out = folded(child, branch_id="child", parent_branch_id="parent")

    # Scenario sanity: the prefix mask is genuinely mixed.
    ref_prefix_mask = gate(child[:, :T_PARENT], h_parent)
    assert 0 < int(ref_prefix_mask.sum()) < ref_prefix_mask.numel()

    # Reference synthesis semantics (design §6.1), recomputed with the same
    # gate instance.
    layer_child = layer(child)
    ref_mask = torch.cat(
        [
            ref_prefix_mask,
            torch.zeros(B, T_CHILD - T_PARENT, dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    ref_aligned = torch.cat([ffn_seed, layer_child[:, T_PARENT:]], dim=1)
    expected = torch.where(ref_mask.unsqueeze(-1), ref_aligned, layer_child)
    assert torch.equal(out, expected)

    # Child activations are stored at the child length T_c.
    stored = cache.fetch(branch_id="child", layer_idx=0)
    assert stored["ffn_out"].shape == (B, T_CHILD, HIDDEN_DIM)
    assert stored["embedding"].shape == (B, T_CHILD, HIDDEN_DIM)


def test_varlen_suffix_always_divergent(device: str) -> None:
    """UT-202: an all-stable prefix folds; the suffix always recomputes.

    Given ``T_p < T_c`` with the child prefix identical to the parent
    embedding, the prefix positions must equal the hand-seeded parent FFN
    output (proving folding happened rather than a full recompute) and the
    suffix positions must equal ``layer(child)[:, T_p:]`` (always divergent).
    """
    layer = DummyLayer(HIDDEN_DIM).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    h_parent = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    ffn_seed = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    _seed_parent_cache(cache, h_parent, ffn_seed)

    child = torch.randn(B, T_CHILD, HIDDEN_DIM, device=device)
    child[:, :T_PARENT] = h_parent

    layer_child = layer(child)
    # Test validity: the seeded parent FFN differs from a full recompute.
    assert not torch.equal(ffn_seed, layer_child[:, :T_PARENT])
    # Scenario sanity: every prefix position is stable.
    ref_prefix_mask = gate(child[:, :T_PARENT], h_parent)
    assert bool(ref_prefix_mask.all())

    out = folded(child, branch_id="child", parent_branch_id="parent")

    # Prefix: folding must have taken the seeded parent FFN values.
    assert torch.equal(out[:, :T_PARENT], ffn_seed)
    # Suffix: always divergent, must be the recomputed child values.
    assert torch.equal(out[:, T_PARENT:], layer_child[:, T_PARENT:])


def test_varlen_parent_longer_full_recompute(device: str) -> None:
    """UT-203: ``T_p > T_c`` is treated as no parent; full recompute, no error.

    A parent cached at a longer sequence length than the child cannot donate
    a prefix; the forward must not raise and must return ``layer(child)``.
    """
    layer = DummyLayer(HIDDEN_DIM).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    t_parent = T_CHILD + 2
    h_parent = torch.randn(B, t_parent, HIDDEN_DIM, device=device)
    ffn_seed = torch.randn(B, t_parent, HIDDEN_DIM, device=device)
    _seed_parent_cache(cache, h_parent, ffn_seed)

    child = torch.randn(B, T_CHILD, HIDDEN_DIM, device=device)
    out = folded(child, branch_id="child", parent_branch_id="parent")

    assert torch.equal(out, layer(child))


@pytest.mark.parametrize("mismatch", ["batch", "hidden"])
def test_varlen_parent_mismatch_full_recompute(device: str, mismatch: str) -> None:
    """UT-204: batch/hidden-dim parent mismatches fall back to full recompute.

    A parent entry whose batch or hidden dimension differs from the child's
    is treated as absent even when ``T_p < T_c``: no exception, output equals
    ``layer(child)``.
    """
    layer = DummyLayer(HIDDEN_DIM).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    if mismatch == "batch":
        parent_shape = (B + 1, T_PARENT, HIDDEN_DIM)
    else:
        parent_shape = (B, T_PARENT, HIDDEN_DIM * 2)
    h_parent = torch.randn(*parent_shape, device=device)
    ffn_seed = torch.randn(*parent_shape, device=device)
    _seed_parent_cache(cache, h_parent, ffn_seed)

    child = torch.randn(B, T_CHILD, HIDDEN_DIM, device=device)
    out = folded(child, branch_id="child", parent_branch_id="parent")

    assert torch.equal(out, layer(child))


def test_varlen_parent_zero_len_full_recompute(device: str) -> None:
    """UT-210: a parent entry with a zero-length seq dim means no parent.

    Seeding the cache with ``T_p == 0`` must not raise during the child
    forward; the child is fully recomputed and the output equals
    ``layer(child)``.
    """
    layer = DummyLayer(HIDDEN_DIM).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    h_parent = torch.randn(B, 0, HIDDEN_DIM, device=device)
    ffn_seed = torch.randn(B, 0, HIDDEN_DIM, device=device)
    _seed_parent_cache(cache, h_parent, ffn_seed)

    child = torch.randn(B, T_CHILD, HIDDEN_DIM, device=device)
    out = folded(child, branch_id="child", parent_branch_id="parent")

    assert torch.equal(out, layer(child))


# ---------------------------------------------------------------------------
# AR003 / T003: three-way prefix decision, chain recursion, split layers
# ---------------------------------------------------------------------------


def test_varlen_prefix_all_stable_takes_mixed_path(device: str) -> None:
    """UT-205: an all-stable var-len prefix takes the mixed path, not the fast path.

    With ``T_p < T_c`` and the child prefix identical to the parent
    embedding, every prefix token is stable but the always-divergent suffix
    keeps ``stable_count < num_tokens``, so the forward must take the mixed
    merge semantics rather than the all-stable fast path (which is only
    reachable at equal lengths).  The seeded parent FFN is strictly shorter
    than the child: a fast-path misfire would fetch that shape-mismatched
    FFN and silently degrade to a full recompute, so the prefix values
    distinguish the two — they must equal the seeded ``ffn_seed`` (folding
    happened) and the whole output must match the reference synthesis
    semantics.
    """
    layer = DummyLayer(HIDDEN_DIM).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    h_parent = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    ffn_seed = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    _seed_parent_cache(cache, h_parent, ffn_seed)

    child = torch.randn(B, T_CHILD, HIDDEN_DIM, device=device)
    child[:, :T_PARENT] = h_parent

    # Scenario sanity: the seeded parent FFN is shorter than the child (the
    # all-stable fast path requires an exact-shape parent FFN), every prefix
    # token is stable, and the seed differs from a full recompute.
    assert ffn_seed.shape[1] < child.shape[1]
    ref_prefix_mask = gate(child[:, :T_PARENT], h_parent)
    assert bool(ref_prefix_mask.all())
    layer_child = layer(child)
    assert not torch.equal(ffn_seed, layer_child[:, :T_PARENT])

    out = folded(child, branch_id="child", parent_branch_id="parent")

    # Mixed-path folding: the prefix carries the seeded parent FFN values.
    assert torch.equal(out[:, :T_PARENT], ffn_seed)
    # The always-divergent suffix carries the recomputed child values.
    assert torch.equal(out[:, T_PARENT:], layer_child[:, T_PARENT:])
    # The full output matches the reference synthesis semantics.
    ref_mask = torch.cat(
        [
            ref_prefix_mask,
            torch.zeros(B, T_CHILD - T_PARENT, dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    ref_aligned = torch.cat([ffn_seed, layer_child[:, T_PARENT:]], dim=1)
    expected = torch.where(ref_mask.unsqueeze(-1), ref_aligned, layer_child)
    assert torch.equal(out, expected)


def test_varlen_prefix_all_divergent_full_recompute(device: str) -> None:
    """UT-206: an all-divergent var-len prefix full-recomputes.

    A child prefix heavily perturbed away from the parent embedding makes
    every prefix token divergent; together with the always-divergent suffix
    the stable count is zero, so the forward must take the full-recompute
    branch and return exactly ``layer(child)``.
    """
    layer = DummyLayer(HIDDEN_DIM).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    h_parent = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    ffn_seed = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    _seed_parent_cache(cache, h_parent, ffn_seed)

    child = torch.randn(B, T_CHILD, HIDDEN_DIM, device=device)
    child[:, :T_PARENT] = h_parent + 100.0 * torch.randn(
        B, T_PARENT, HIDDEN_DIM, device=device
    )

    # Scenario sanity: no prefix token is stable, so stable_count == 0.
    ref_prefix_mask = gate(child[:, :T_PARENT], h_parent)
    assert int(ref_prefix_mask.sum()) == 0

    out = folded(child, branch_id="child", parent_branch_id="parent")

    assert torch.equal(out, layer(child))


def test_varlen_chain_recursion_three_generations(device: str) -> None:
    """UT-207: chained recursion across three generations T4 -> T5 -> T6.

    A hand-seeded gen0 parent cache at T=4 donates a prefix to a child at
    T=5; the child's own cache entry (written by ``_store_activations``:
    ``embedding`` = child input, ``ffn_out`` = child merged output) then
    donates a prefix to a grandchild at T=6.  Both folded forwards must
    match the reference synthesis semantics computed against the cache
    entries actually written by the previous generation (the gate reads the
    child's stored embedding prefix; the merge reads the child's stored
    ``ffn_out``), and each generation must store its activations at its own
    sequence length.
    """
    b = 2
    hidden = 16
    t_gen0, t_child, t_grandchild = 4, 5, 6

    layer = DummyLayer(hidden).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    # gen0: hand-seeded parent cache at T=4.
    h_gen0 = torch.randn(b, t_gen0, hidden, device=device)
    ffn_gen0 = torch.randn(b, t_gen0, hidden, device=device)
    _seed_parent_cache(cache, h_gen0, ffn_gen0)

    # Child at T=5: prefix positions 0..1 inherit the gen0 input (stable),
    # positions 2..3 are heavily perturbed (divergent), the suffix is
    # always divergent — a genuinely mixed reference mask.
    child = torch.randn(b, t_child, hidden, device=device)
    child[:, :t_gen0] = h_gen0
    child[:, t_gen0 - 2 : t_gen0] += 100.0 * torch.randn(b, 2, hidden, device=device)

    ref_prefix_mask = gate(child[:, :t_gen0], h_gen0)
    assert 0 < int(ref_prefix_mask.sum()) < ref_prefix_mask.numel()

    out_child = folded(child, branch_id="child", parent_branch_id="parent")

    layer_child = layer(child)
    ref_mask = torch.cat(
        [
            ref_prefix_mask,
            torch.zeros(b, t_child - t_gen0, dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    ref_aligned = torch.cat([ffn_gen0, layer_child[:, t_gen0:]], dim=1)
    expected_child = torch.where(ref_mask.unsqueeze(-1), ref_aligned, layer_child)
    assert torch.equal(out_child, expected_child)

    # The child stored its activations at its own length T=5; these entries
    # are the parent donation for the grandchild.
    stored = cache.fetch(branch_id="child", layer_idx=0)
    assert stored["ffn_out"].shape == (b, t_child, hidden)
    assert stored["embedding"].shape == (b, t_child, hidden)

    # Grandchild at T=6: the prefix partially inherits the child input and
    # is partially perturbed, so the T6 reference mask is mixed as well.
    grandchild = torch.randn(b, t_grandchild, hidden, device=device)
    grandchild[:, :t_child] = child
    grandchild[:, t_child - 2 : t_child] += 100.0 * torch.randn(
        b, 2, hidden, device=device
    )

    child_embedding = stored["embedding"]
    child_ffn = stored["ffn_out"]
    ref_prefix_mask_gc = gate(grandchild[:, :t_child], child_embedding)
    assert 0 < int(ref_prefix_mask_gc.sum()) < ref_prefix_mask_gc.numel()

    out_grandchild = folded(grandchild, branch_id="grandchild", parent_branch_id="child")

    layer_grandchild = layer(grandchild)
    ref_mask_gc = torch.cat(
        [
            ref_prefix_mask_gc,
            torch.zeros(b, t_grandchild - t_child, dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    ref_aligned_gc = torch.cat([child_ffn, layer_grandchild[:, t_child:]], dim=1)
    expected_grandchild = torch.where(
        ref_mask_gc.unsqueeze(-1), ref_aligned_gc, layer_grandchild
    )
    assert torch.equal(out_grandchild, expected_grandchild)

    stored_gc = cache.fetch(branch_id="grandchild", layer_idx=0)
    assert stored_gc["ffn_out"].shape == (b, t_grandchild, hidden)
    assert stored_gc["embedding"].shape == (b, t_grandchild, hidden)


class _SplitTestAttention(nn.Module):
    """Token-wise stand-in attention returning ``(hidden * 0.5, None)``."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        return hidden_states * 0.5, None


class _FFNChainLayer(nn.Module):
    """Pre-norm decoder layer exposing the post_attention_layernorm->mlp chain."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_attn = _SplitTestAttention()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)

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


def test_varlen_split_layer_divergent_rows(device: str) -> None:
    """UT-208: the split layer slices FFN rows on the full var-len mask.

    With ``B * T_c >= min_split_tokens`` (default 512, ``>=`` comparison)
    and a mixed var-len prefix, ``SplitFoldedTransformerLayer`` must feed
    the FFN chain exactly the divergent rows of the *complete* stability
    mask — every divergent prefix row plus all ``B * (T_c - T_p)`` suffix
    rows — and the merged output must match the reference synthesis
    semantics.
    """
    b, t_parent, t_child = 8, 60, 64
    hidden = 16

    layer = _FFNChainLayer(hidden).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    split = SplitFoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)
    assert split.split_enabled
    # Scenario sanity: B * T_c meets the default min_split_tokens boundary.
    assert split.min_split_tokens == 512
    assert b * t_child >= split.min_split_tokens

    h_parent = torch.randn(b, t_parent, hidden, device=device)
    ffn_seed = torch.randn(b, t_parent, hidden, device=device)
    _seed_parent_cache(cache, h_parent, ffn_seed)

    # Child: the first half of the prefix inherits the parent embedding
    # (stable), the rest of the prefix and the whole suffix are divergent.
    child = torch.randn(b, t_child, hidden, device=device)
    stable_prefix = t_parent // 2
    child[:, :stable_prefix] = h_parent[:, :stable_prefix]

    ref_prefix_mask = gate(child[:, :t_parent], h_parent)
    assert bool(ref_prefix_mask[:, :stable_prefix].all())
    assert 0 < int(ref_prefix_mask.sum()) < ref_prefix_mask.numel()

    # Record every FFN (mlp) input shape via a forward pre-hook.
    mlp_input_shapes: list[tuple[int, ...]] = []

    def _record_mlp_input(module: nn.Module, args: tuple) -> None:
        mlp_input_shapes.append(tuple(args[0].shape))

    handle = layer.mlp.register_forward_pre_hook(_record_mlp_input)

    # Reference synthesis semantics, computed before the folded forward
    # (the resident split hooks are transparent while no folding state is
    # set, so this call sees the full 3-D FFN input).
    layer_child = layer(child)
    ref_mask = torch.cat(
        [
            ref_prefix_mask,
            torch.zeros(b, t_child - t_parent, dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    ref_aligned = torch.cat([ffn_seed, layer_child[:, t_parent:]], dim=1)
    expected = torch.where(ref_mask.unsqueeze(-1), ref_aligned, layer_child)

    out = split(child, branch_id="child", parent_branch_id="parent")
    handle.remove()

    # The split engaged on the FULL mask: the FFN saw a 2-D batch of
    # exactly the divergent rows — prefix divergent plus all suffix rows.
    prefix_divergent = int((~ref_prefix_mask).sum())
    expected_rows = prefix_divergent + b * (t_child - t_parent)
    assert int((~ref_mask).sum()) == expected_rows
    assert len(mlp_input_shapes[-1]) == 2
    assert mlp_input_shapes[-1][0] == expected_rows
    assert mlp_input_shapes[-1][-1] == hidden

    # Merged output matches the reference semantics (the row-sliced FFN
    # GEMM may differ from the full 3-D GEMM in the last bits).
    assert torch.allclose(out, expected, atol=1e-6)


def test_equal_length_unchanged_reference(device: str) -> None:
    """UT-209: the equal-length path is bit-unchanged after the prefix rework.

    With ``T_p == T_c`` the forward must keep the historical equal-length
    semantics exactly: the gate compares the full child against the parent
    embedding and the output equals
    ``where(gate(child, parent_emb), ffn_seed, layer(child))`` with no
    suffix concatenation.
    """
    layer = DummyLayer(HIDDEN_DIM).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    t_eq = T_CHILD
    h_parent = torch.randn(B, t_eq, HIDDEN_DIM, device=device)
    ffn_seed = torch.randn(B, t_eq, HIDDEN_DIM, device=device)
    _seed_parent_cache(cache, h_parent, ffn_seed)

    # Mixed child: the leading tokens equal the parent embedding (stable),
    # the trailing ones are fresh random values (divergent).
    child = torch.randn(B, t_eq, HIDDEN_DIM, device=device)
    child[:, : t_eq // 2] = h_parent[:, : t_eq // 2]

    ref_mask = gate(child, h_parent)
    assert 0 < int(ref_mask.sum()) < ref_mask.numel()

    out = folded(child, branch_id="child", parent_branch_id="parent")

    layer_child = layer(child)
    expected = torch.where(ref_mask.unsqueeze(-1), ffn_seed, layer_child)
    assert torch.equal(out, expected)


class _DictActivationCache:
    """Minimal dict-backed activation cache for hand-seeding raw entries.

    The production caches (``ActivationCache`` etc.) validate a consistent
    leading shape per entry and share one sequence-length counter across
    names, so an ``ffn_out`` whose leading shape differs from the
    ``embedding`` cannot be represented through their public ``put``.  This
    stub implements the raw ``put``/``fetch`` protocol consumed by
    ``FoldedTransformerLayer`` and returns exactly what was seeded, which
    is what the shape-validation guard in ``_get_parent_ffn_output`` must
    tolerate as a hard error.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, int], dict[str, torch.Tensor]] = {}

    def put(
        self,
        branch_id: str,
        layer_idx: int,
        activations: dict[str, torch.Tensor],
    ) -> None:
        self._entries.setdefault((branch_id, layer_idx), {}).update(activations)

    def fetch(self, branch_id: str, layer_idx: int) -> dict[str, torch.Tensor]:
        return self._entries[(branch_id, layer_idx)]


def test_varlen_parent_ffn_shape_mismatch_raises(device: str) -> None:
    """EX-505: a shape-mismatched parent FFN on the mixed var-len path raises.

    On the mixed variable-length path the cached parent ``ffn_out`` must
    match ``[batch, prefix_len]``.  Seeding an ``ffn_out`` one token longer
    than the prefix (with a well-formed ``embedding``) must make the child
    forward raise ``RuntimeError`` instead of silently mis-merging.
    """
    layer = DummyLayer(HIDDEN_DIM).to(device)
    cache = _DictActivationCache()
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    h_parent = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    bad_ffn_seed = torch.randn(B, T_PARENT + 1, HIDDEN_DIM, device=device)
    cache.put(
        branch_id="parent",
        layer_idx=0,
        activations={"embedding": h_parent, "ffn_out": bad_ffn_seed},
    )

    # Mixed prefix so the forward reaches the mixed merge path where the
    # parent FFN shape is validated.
    child = torch.randn(B, T_CHILD, HIDDEN_DIM, device=device)
    child[:, :T_PARENT] = h_parent
    child[:, T_PARENT - 2 : T_PARENT] += 100.0 * torch.randn(
        B, 2, HIDDEN_DIM, device=device
    )
    ref_prefix_mask = gate(child[:, :T_PARENT], h_parent)
    assert 0 < int(ref_prefix_mask.sum()) < ref_prefix_mask.numel()

    with pytest.raises(RuntimeError):
        folded(child, branch_id="child", parent_branch_id="parent")


def test_varlen_parent_ffn_missing_raises(device: str) -> None:
    """EX-501: a missing parent FFN on the mixed var-len path raises.

    The gate reads the parent ``embedding`` (layer 0 contract) while the
    merge reads the parent ``ffn_out``; when the FFN entry is absent but the
    embedding exists, the mixed merge must raise ``RuntimeError`` instead of
    silently mis-merging — the same no-swallow contract as the equal-length
    mixed path.
    """
    layer = DummyLayer(HIDDEN_DIM).to(device)
    cache = _DictActivationCache()
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    h_parent = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    cache.put(
        branch_id="parent",
        layer_idx=0,
        activations={"embedding": h_parent},  # ffn_out deliberately absent
    )

    child = torch.randn(B, T_CHILD, HIDDEN_DIM, device=device)
    child[:, :T_PARENT] = h_parent
    child[:, T_PARENT - 2 : T_PARENT] += 100.0 * torch.randn(
        B, 2, HIDDEN_DIM, device=device
    )
    ref_prefix_mask = gate(child[:, :T_PARENT], h_parent)
    assert 0 < int(ref_prefix_mask.sum()) < ref_prefix_mask.numel()

    with pytest.raises(RuntimeError):
        folded(child, branch_id="child", parent_branch_id="parent")


class MaskSensitiveLayer(nn.Module):
    """Layer whose output depends on the attention mask (EX-503)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.linear(hidden_states)
        if attention_mask is not None:
            out = out * attention_mask.to(out.dtype).unsqueeze(-1)
        return out


def test_varlen_attention_mask_passthrough(device: str) -> None:
    """EX-503: a child-length attention mask flows to the original layer.

    On variable-length steps the mask (child length) is forwarded to the
    original layer for the recompute exactly as on the equal-length path;
    the gate/merge are mask-agnostic.  The output must equal the reference
    synthesis semantics computed with the masked recompute.
    """
    layer = MaskSensitiveLayer(HIDDEN_DIM).to(device)
    cache = ActivationCache(device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    h_parent = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    ffn_seed = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    _seed_parent_cache(cache, h_parent, ffn_seed)

    child = torch.randn(B, T_CHILD, HIDDEN_DIM, device=device)
    child[:, :T_PARENT] = h_parent
    child[:, T_PARENT - 2 : T_PARENT] += 100.0 * torch.randn(
        B, 2, HIDDEN_DIM, device=device
    )
    attention_mask = torch.ones(B, T_CHILD, dtype=torch.bool, device=device)
    attention_mask[:, -1] = False  # mask out the last suffix position

    out = folded(
        child,
        branch_id="child",
        parent_branch_id="parent",
        attention_mask=attention_mask,
    )

    ref_prefix_mask = gate(child[:, :T_PARENT], h_parent)
    assert 0 < int(ref_prefix_mask.sum()) < ref_prefix_mask.numel()
    layer_child = layer(child, attention_mask=attention_mask)
    ref_mask = torch.cat(
        [
            ref_prefix_mask,
            torch.zeros(B, T_CHILD - T_PARENT, dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    ref_aligned = torch.cat([ffn_seed, layer_child[:, T_PARENT:]], dim=1)
    expected = torch.where(ref_mask.unsqueeze(-1), ref_aligned, layer_child)
    assert torch.equal(out, expected)


def test_varlen_cache_eviction_breaks_and_rebuilds_chain(device: str) -> None:
    """EX-506: eviction breaks the reuse chain for one step, then rebuilds.

    With a token budget that cannot hold two branches, a sibling ``put``
    evicts the parent entry after the parent forward.  The next child step
    treats the evicted parent as a cache miss: full recompute, no exception.
    The chain then rebuilds — the following generation folds again against
    the freshly stored child.
    """
    layer = DummyLayer(HIDDEN_DIM).to(device)
    # Budget of T_PARENT tokens per layer: two groups of T_PARENT cannot
    # coexist, so the oldest group is evicted on every second put.
    cache = ActivationCache(max_entries_per_layer=T_PARENT, device=device)
    gate = SimilarityGate(tau=0.95)
    folded = FoldedTransformerLayer(layer, cache, gate, layer_idx=0).to(device)

    h_parent = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    ffn_seed = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    _seed_parent_cache(cache, h_parent, ffn_seed)
    # Sibling put evicts the parent group (oldest-first under the budget).
    sibling_emb = torch.randn(B, T_PARENT, HIDDEN_DIM, device=device)
    _seed_parent_cache_branch(cache, "sibling", sibling_emb, ffn_seed)
    with pytest.raises((KeyError, RuntimeError)):
        cache.fetch(branch_id="parent", layer_idx=0)

    # Generation 1: parent evicted -> cache miss -> full recompute, no error.
    child = torch.randn(B, T_CHILD, HIDDEN_DIM, device=device)
    child[:, :T_PARENT] = h_parent
    child[:, T_PARENT - 2 : T_PARENT] += 100.0 * torch.randn(
        B, 2, HIDDEN_DIM, device=device
    )
    out_child = folded(child, branch_id="child", parent_branch_id="parent")
    assert torch.equal(out_child, layer(child))

    # Generation 2: the chain rebuilt — grandchild folds against the stored
    # child (its cache entry is the newest group and always survives).
    grandchild = torch.randn(B, T_CHILD + 1, HIDDEN_DIM, device=device)
    grandchild[:, :T_CHILD] = child
    grandchild[:, T_CHILD - 2 : T_CHILD] += 100.0 * torch.randn(
        B, 2, HIDDEN_DIM, device=device
    )
    out_grandchild = folded(
        grandchild, branch_id="grandchild", parent_branch_id="child"
    )

    # Generation 2: the chain rebuilt — grandchild folds against the stored
    # child.  The reference parent states are the gen-1 input/output pair
    # (out_child == layer(child) on the miss path, matching what
    # _store_activations stored); the child entry is read before the
    # grandchild's own put evicts it under the tight budget.
    ref_prefix_mask = gate(grandchild[:, :T_CHILD], child)
    assert 0 < int(ref_prefix_mask.sum()) < ref_prefix_mask.numel()
    layer_gc = layer(grandchild)
    ref_mask = torch.cat(
        [
            ref_prefix_mask,
            torch.zeros(B, 1, dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    ref_aligned = torch.cat([out_child, layer_gc[:, T_CHILD:]], dim=1)
    expected = torch.where(ref_mask.unsqueeze(-1), ref_aligned, layer_gc)
    assert torch.equal(out_grandchild, expected)
