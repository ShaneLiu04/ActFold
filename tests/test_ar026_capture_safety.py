"""Red-phase tests for AR026: capture-root-cause fidelity (UT-2601/UT-2602 + faces).

Contract: ``specs/changes/AR026-real-graph-e2e/design.md`` §4.3 (``safe_capture``
+ ``CaptureDiagnostics`` contracts), §6.1 (UT-2601, UT-2602), §6.2 (interface:
passthrough + is-identity re-raise), §6.4 (exception faces: OOM variant,
reset-failure/poisoned-state variant) and ``srs.md`` §3.1 FR1 — the production
capture boundary must never mask the primary exception behind the
capture-unwinding ``captures_underway.empty() INTERNAL ASSERT`` secondary
failure, and must re-raise the original exception unchanged so the existing
call-site degradation semantics stay byte-identical.

Covered points:

* UT-2601 (FR1 typed degradation): a mock runner whose ``capture`` raises the
  sample ``captures_underway.empty() INTERNAL ASSERT FAILED ...
  (CUDACachingAllocator.cpp:2967)`` RuntimeError with NO exception context →
  ``safe_capture`` re-raises it and the recorded diagnostics (via the
  module-level snapshot accessor) carry ``ok=False``, a ``secondary_error``
  containing ``captures_underway`` and ``primary_error=None``; on the CPU face
  (``torch.cuda.is_available()`` False) the VRAM fields are ``None`` and no
  CUDA call is ever made (a forbidden-attribute guard);
* UT-2601 OOM variant (§6.4): the sample ``CUDA out of memory. Tried to
  allocate ...`` secondary lands in ``secondary_error``;
* UT-2602 (FR1 primary fidelity): a chained mock — the primary
  ``ValueError("shape mismatch: size of tensor a (64) must match tensor b
  (4) ...")`` is raised first inside ``capture``, caught by a handler, and the
  INTERNAL-ASSERT RuntimeError is raised from that handler (``__context__`` /
  ``__cause__`` chain set) → diagnostics separate ``primary_error`` (contains
  ``shape mismatch``) from ``secondary_error`` (contains
  ``captures_underway``); ``CaptureDiagnostics.to_json_dict()`` round-trips
  through ``json.dumps`` with every field key present;
* UT-2602 reset-failure variant (§6.4 poisoned state): with
  ``torch.cuda.is_available`` monkeypatched True and
  ``torch.cuda.synchronize`` monkeypatched to raise ``RuntimeError("poisoned")``,
  the diagnostics ``reset_error`` field contains ``poisoned`` while the
  RE-RAISED exception is still the exact original capture failure;
* §6.2 interface passthrough: the success path forwards ``*args``/``**kwargs``
  verbatim to ``runner.capture`` (positional identity included) and returns a
  ``CaptureDiagnostics`` with ``ok=True`` / ``primary_error=None`` /
  ``secondary_error=None`` / the caller's ``stage``;
* §6.2 is-identity re-raise: the exception object caught by ``pytest.raises``
  IS the exact object the mock raised (``excinfo.value is original``);
* accumulator contract: the module-level diagnostics accumulator in
  ``actfold.core.cuda_graph`` keeps the FIRST 8 failure entries; later
  failures only increment an overflow counter (10 failures → snapshot length
  8, first-8 stages in order, overflow == 2); the snapshot returns copies —
  mutating a returned dict never leaks into the next snapshot; the test-only
  ``_reset_capture_diagnostics()`` helper restores the pristine state;
* probe key: a minimal ``ManualFoldedForward`` (fresh, CPU-constructible) has
  ``graph_pool_stats()["capture_diagnostics"] == []`` and
  ``graph_pool_stats()["capture_diagnostics_overflow"] == 0``.

Red state (why every test fails today) — each failure is a production-missing
face, never a test-own bug: ``actfold.core.cuda_graph`` has no
``safe_capture`` / ``CaptureDiagnostics`` / ``capture_diagnostics_snapshot`` /
``capture_diagnostics_overflow`` / ``_reset_capture_diagnostics`` names — the
lazy per-test import (the ``test_cuda_graph.py`` pattern, so module collection
always succeeds) fails with ``ImportError``; ``ManualFoldedForward.graph_pool_stats``
carries no ``capture_diagnostics`` keys (KeyError).

All faces are pure CPU + mocks (a ``_FakeRunner`` double defined in this
module); the CPU face is pinned by monkeypatching ``torch.cuda.is_available``
to False with forbidden-attribute guards on the CUDA entry points, so no CUDA
hardware is required anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate
from actfold.models.architecture_utils import ManualFoldedForward

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TAU = 0.95
_EPS = 1e-8
_VOCAB = 32
_HIDDEN = 16
_NUM_LAYERS = 2

# Sample secondary texts (srs.md §1 / design.md §4.1 — the two real AR023
# probe failure signatures).
_INTERNAL_ASSERT_MSG = (
    "captures_underway.empty() INTERNAL ASSERT FAILED at "
    "../c10/cuda/CUDACachingAllocator.cpp:2967 "
    "(Please file a bug report with the minimal reproduction script)"
)
_OOM_MSG = (
    "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has 15.93 GiB "
    "capacity of which 1.02 GiB is free."
)
_PRIMARY_MSG = (
    "shape mismatch: size of tensor a (64) must match tensor b (4) at " "non-singleton dimension 2"
)

# The frozen to_json_dict field set (design.md §4.3 CaptureDiagnostics).
_DIAGNOSTICS_KEYS = {
    "ok",
    "stage",
    "primary_error",
    "secondary_error",
    "reset_error",
    "vram_free_mb",
    "vram_total_mb",
    "platform",
}


def _load_capture_api() -> Any:
    """Lazily import the AR026 capture-safety API (per-test Red failures).

    The ``from ... import`` form raises ``ImportError`` per test while the
    names are missing from ``actfold.core.cuda_graph`` (the Red state), so
    module collection always succeeds.
    """
    from actfold.core.cuda_graph import (
        CaptureDiagnostics,
        _reset_capture_diagnostics,
        capture_diagnostics_overflow,
        capture_diagnostics_snapshot,
        safe_capture,
    )

    return SimpleNamespace(
        safe_capture=safe_capture,
        diagnostics_cls=CaptureDiagnostics,
        snapshot=capture_diagnostics_snapshot,
        overflow=capture_diagnostics_overflow,
        reset=_reset_capture_diagnostics,
    )


@pytest.fixture()
def capture_api() -> Any:
    """The AR026 capture API with an isolated module-level accumulator."""
    api = _load_capture_api()
    api.reset()
    yield api
    api.reset()


def _force_cpu_face(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the CPU face: ``is_available`` False and any CUDA call is a failure.

    The forbidden guards pin the contract "on CPU no CUDA calls are made" —
    if the Green implementation reaches ``synchronize``/``mem_get_info``
    while ``is_available()`` is False, the guard raises and the test fails.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _forbidden_cuda_call(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("CUDA call made on the CPU face")

    monkeypatch.setattr(torch.cuda, "synchronize", _forbidden_cuda_call)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _forbidden_cuda_call, raising=False)


# ---------------------------------------------------------------------------
# Runner doubles (pure mocks; no CUDA anywhere)
# ---------------------------------------------------------------------------


class _FakeRunner:
    """Minimal runner double: records ``capture`` args, raises the preset error."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._error = error

    def capture(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append((args, kwargs))
        if self._error is not None:
            raise self._error
        return "captured"


class _ChainedFakeRunner(_FakeRunner):
    """Raise the INTERNAL-ASSERT secondary FROM a handler that caught the primary.

    The primary ``ValueError`` is raised first inside ``capture``, caught, and
    the INTERNAL-ASSERT ``RuntimeError`` is raised from that handler — so the
    secondary's ``__context__``/``__cause__`` chain carries the primary (the
    production capture-unwinding shape, design.md §4.1).
    """

    def capture(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append((args, kwargs))
        try:
            raise ValueError(_PRIMARY_MSG)
        except ValueError as exc:
            raise RuntimeError(_INTERNAL_ASSERT_MSG) from exc


# ---------------------------------------------------------------------------
# Self-contained minimal LLaMA-layout decoder (probe-key fixture)
# ---------------------------------------------------------------------------


class _TinyLayer(nn.Module):
    """Single linear layer (the lightest detectable LLaMA-layout block)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _TinyLlamaModel(nn.Module):
    """Minimal LLaMA-layout decoder: embed -> layers -> lm_head (CPU-friendly)."""

    def __init__(
        self,
        vocab_size: int = _VOCAB,
        hidden_dim: int = _HIDDEN,
        num_layers: int = _NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "vocab_size": vocab_size})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model.layers = nn.ModuleList([_TinyLayer(hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# UT-2601: typed degradation (INTERNAL-ASSERT secondary, no context)
# ---------------------------------------------------------------------------


def test_ut2601_internal_assert_reraise_and_diagnostics(
    capture_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2601: bare INTERNAL-ASSERT failure re-raised + typed diagnostics record.

    The mock raises the sample secondary with NO exception context → the
    re-raised exception is the same failure text, and the recorded
    diagnostics carry ``ok=False`` / ``secondary_error`` with the sample /
    ``primary_error=None``; CPU face: VRAM fields None, platform stamped.
    """
    _force_cpu_face(monkeypatch)
    runner = _FakeRunner(RuntimeError(_INTERNAL_ASSERT_MSG))

    with pytest.raises(RuntimeError, match="captures_underway") as excinfo:
        capture_api.safe_capture(runner, "serial")

    assert "CUDACachingAllocator.cpp:2967" in str(excinfo.value)

    snapshot = capture_api.snapshot()
    assert len(snapshot) == 1
    entry = snapshot[0]
    assert _DIAGNOSTICS_KEYS <= set(entry)
    assert entry["ok"] is False
    assert entry["stage"] == "serial"
    assert entry["primary_error"] is None
    assert entry["secondary_error"] is not None
    assert "captures_underway" in entry["secondary_error"]
    assert entry["reset_error"] is None
    # CPU face: no VRAM numbers, platform still stamped as a string.
    assert entry["vram_free_mb"] is None
    assert entry["vram_total_mb"] is None
    assert isinstance(entry["platform"], str)
    assert entry["platform"]


def test_ut2601_oom_variant_secondary_error(
    capture_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.4 OOM variant: the CUDA-OOM sample lands in ``secondary_error``."""
    _force_cpu_face(monkeypatch)
    runner = _FakeRunner(RuntimeError(_OOM_MSG))

    with pytest.raises(RuntimeError, match="out of memory"):
        capture_api.safe_capture(runner, "batched")

    entry = capture_api.snapshot()[-1]
    assert entry["ok"] is False
    assert entry["secondary_error"] is not None
    assert "out of memory" in entry["secondary_error"]


# ---------------------------------------------------------------------------
# UT-2602: primary-chain extraction + JSON round-trip + poisoned reset
# ---------------------------------------------------------------------------


def test_ut2602_primary_chain_extraction(capture_api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-2602: the ``__context__`` chain primary is not masked by the secondary."""
    _force_cpu_face(monkeypatch)
    runner = _ChainedFakeRunner()

    with pytest.raises(RuntimeError, match="captures_underway"):
        capture_api.safe_capture(runner, "serial")

    entry = capture_api.snapshot()[-1]
    assert entry["ok"] is False
    assert entry["primary_error"] is not None
    assert "shape mismatch" in entry["primary_error"]
    assert "tensor a (64)" in entry["primary_error"]
    assert entry["secondary_error"] is not None
    assert "captures_underway" in entry["secondary_error"]
    assert entry["primary_error"] != entry["secondary_error"]
    # The chained snapshot entry itself is JSON-serializable.
    json.loads(json.dumps(entry))


def test_ut2602_three_link_chain_oldest_is_primary(
    capture_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UT-2602 hardening: a 3-link chain extracts the OLDEST context as primary.

    Chain shape: primary ``ValueError`` -> intermediate ``RuntimeError`` (the
    unwinding face) -> boundary ``RuntimeError`` (INTERNAL ASSERT).  A walk
    that stops at the FIRST context would report the intermediate unwinding
    failure as primary — this test pins the deepest-walk semantics (the
    mutation-kill face for ``_deepest_context``).
    """
    _force_cpu_face(monkeypatch)

    class _ThreeLinkRunner(_FakeRunner):
        def capture(self, *args: Any, **kwargs: Any) -> str:
            self.calls.append((args, kwargs))
            try:
                raise ValueError(_PRIMARY_MSG)
            except ValueError:
                try:
                    raise RuntimeError("unwinding intermediate failure")
                except RuntimeError as intermediate:
                    raise RuntimeError(_INTERNAL_ASSERT_MSG) from intermediate

    runner = _ThreeLinkRunner()
    with pytest.raises(RuntimeError, match="captures_underway"):
        capture_api.safe_capture(runner, "batched")

    entry = capture_api.snapshot()[-1]
    assert entry["primary_error"] is not None
    assert "shape mismatch" in entry["primary_error"]
    assert "intermediate failure" not in entry["primary_error"]
    assert entry["secondary_error"] is not None
    assert "captures_underway" in entry["secondary_error"]


def test_ut2602_to_json_dict_roundtrip(capture_api: Any) -> None:
    """UT-2602: ``CaptureDiagnostics.to_json_dict()`` round-trips via json.dumps."""
    diagnostics_cls = capture_api.diagnostics_cls
    diagnostics = diagnostics_cls(
        ok=False,
        stage="serial",
        primary_error=_PRIMARY_MSG,
        secondary_error=_INTERNAL_ASSERT_MSG,
        reset_error=None,
        vram_free_mb=None,
        vram_total_mb=None,
        platform="win32/torch2.5.1+cu121",
    )
    as_dict = diagnostics.to_json_dict()
    assert _DIAGNOSTICS_KEYS <= set(as_dict)
    assert as_dict["ok"] is False
    assert as_dict["stage"] == "serial"
    assert as_dict["primary_error"] == _PRIMARY_MSG
    assert as_dict["secondary_error"] == _INTERNAL_ASSERT_MSG
    assert as_dict["reset_error"] is None
    assert as_dict["vram_free_mb"] is None
    assert as_dict["vram_total_mb"] is None
    assert as_dict["platform"] == "win32/torch2.5.1+cu121"

    round_tripped = json.loads(json.dumps(as_dict))
    assert round_tripped == as_dict


def test_ut2602_reset_failure_poisoned_state(
    capture_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.4 poisoned state: a failing best-effort reset lands in ``reset_error``.

    ``torch.cuda.is_available`` is monkeypatched True so the reset path
    engages; ``torch.cuda.synchronize`` is monkeypatched to raise
    ``RuntimeError("poisoned")``. The diagnostics record the reset failure,
    and the re-raised exception is still the ORIGINAL capture failure — the
    poisoned reset never replaces or wraps it.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _poisoned_synchronize(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("poisoned")

    monkeypatch.setattr(torch.cuda, "synchronize", _poisoned_synchronize)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (512, 1024), raising=False)

    original = RuntimeError(_INTERNAL_ASSERT_MSG)
    runner = _FakeRunner(original)

    with pytest.raises(RuntimeError) as excinfo:
        capture_api.safe_capture(runner, "adaptive")

    assert excinfo.value is original
    assert "poisoned" not in str(excinfo.value)

    entry = capture_api.snapshot()[-1]
    assert entry["ok"] is False
    assert entry["reset_error"] is not None
    assert "poisoned" in entry["reset_error"]
    assert entry["secondary_error"] is not None
    assert "captures_underway" in entry["secondary_error"]


# ---------------------------------------------------------------------------
# §6.2 interface contract: passthrough + is-identity re-raise
# ---------------------------------------------------------------------------


def test_interface_passthrough_success(capture_api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """§6.2: success path forwards args verbatim and returns ``ok=True`` diagnostics."""
    _force_cpu_face(monkeypatch)
    runner = _FakeRunner()
    tokens = torch.randint(0, _VOCAB, (1, 4))

    result = capture_api.safe_capture(
        runner,
        "batched",
        tokens,
        branch_id="child",
        parent_branch_id="parent",
        forced_divergent=None,
    )

    # Verbatim passthrough: exactly one call, positional identity preserved.
    assert len(runner.calls) == 1
    args, kwargs = runner.calls[0]
    assert args == (tokens,)
    assert args[0] is tokens
    assert kwargs == {
        "branch_id": "child",
        "parent_branch_id": "parent",
        "forced_divergent": None,
    }

    assert isinstance(result, capture_api.diagnostics_cls)
    assert result.ok is True
    assert result.stage == "batched"
    assert result.primary_error is None
    assert result.secondary_error is None
    assert result.reset_error is None
    # CPU face: no VRAM numbers on the success artifact either.
    assert result.vram_free_mb is None
    assert result.vram_total_mb is None
    # A success records no failure entry.
    assert not [entry for entry in capture_api.snapshot() if entry["ok"] is False]


def test_interface_is_identity_reraise(capture_api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """§6.2: the exception caught by the caller IS the object the mock raised."""
    _force_cpu_face(monkeypatch)
    original = RuntimeError(_INTERNAL_ASSERT_MSG)
    runner = _FakeRunner(original)

    with pytest.raises(RuntimeError) as excinfo:
        capture_api.safe_capture(runner, "serial")

    assert excinfo.value is original


# ---------------------------------------------------------------------------
# Accumulator contract: first-8 cap, overflow counter, copy snapshots
# ---------------------------------------------------------------------------


def test_accumulator_cap_overflow_and_snapshot_copies(
    capture_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """10 failures → the snapshot keeps the FIRST 8 (in order), overflow == 2.

    The snapshot returns copies: mutating a returned dict must not affect the
    next snapshot.
    """
    _force_cpu_face(monkeypatch)
    for i in range(10):
        runner = _FakeRunner(RuntimeError(_INTERNAL_ASSERT_MSG))
        with pytest.raises(RuntimeError, match="captures_underway"):
            capture_api.safe_capture(runner, f"s{i}")

    snapshot = capture_api.snapshot()
    assert isinstance(snapshot, list)
    assert len(snapshot) == 8
    assert [entry["stage"] for entry in snapshot] == [f"s{i}" for i in range(8)]
    assert capture_api.overflow() == 2

    # Copy semantics: mutate the returned dicts, re-snapshot, originals intact.
    snapshot[0]["stage"] = "mutated"
    snapshot[0]["ok"] = "mutated"
    snapshot_again = capture_api.snapshot()
    assert snapshot_again is not snapshot
    assert snapshot_again[0] is not snapshot[0]
    assert snapshot_again[0]["stage"] == "s0"
    assert snapshot_again[0]["ok"] is False
    assert len(snapshot_again) == 8
    assert capture_api.overflow() == 2


# ---------------------------------------------------------------------------
# Probe key: ManualFoldedForward.graph_pool_stats extension
# ---------------------------------------------------------------------------


def test_mff_probe_capture_diagnostics_keys(capture_api: Any, device: str) -> None:
    """A fresh minimal MFF exposes the accumulator probe keys at their zero state.

    Construction follows ``tests/test_architecture_utils.py`` (the lightest
    ``ManualFoldedForward(model, cache=..., gate=...)`` pattern); the
    module-level accumulator is reset first so the fresh-instance zero state
    is observable regardless of test ordering.
    """
    model = _TinyLlamaModel().to(device)
    cache = ActivationCache(max_entries_per_layer=16, device=device)
    gate = SimilarityGate(tau=_TAU, metric="cosine", eps=_EPS)
    mff = ManualFoldedForward(model, cache=cache, gate=gate).to(device)

    stats = mff.graph_pool_stats  # property (AR021 probe; no call)
    assert "capture_diagnostics" in stats
    assert stats["capture_diagnostics"] == []
    assert isinstance(stats["capture_diagnostics"], list)
    assert "capture_diagnostics_overflow" in stats
    assert stats["capture_diagnostics_overflow"] == 0
    assert isinstance(stats["capture_diagnostics_overflow"], int)
