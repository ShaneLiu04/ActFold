"""Red-phase tests for AR014/FR4: per-(kernel, dtype, hidden-bucket) Triton fallback.

UT-1301/1302/1303 (srs 3.4, design 3.2) pin the isolation semantics of the
failure registry ``_TRITON_FAILED_KEYS``: a failed Triton key only silences
the same (kernel, dtype, hidden-bucket) class — other dtypes, other buckets
and other kernels keep attempting the real kernel — and the failed class
silently falls back on subsequent calls.  UT-1304 pins the manual
kill-switch compatibility, UT-1305 the diagnosable warning content, and
UT-1306 the ``_hidden_bucket`` pure function (CPU, no gate).

Red-phase isolation note: the autouse fixture below calls
``monkeypatch.setattr(fused_ops, "_TRITON_FAILED_KEYS", set())`` with the
default ``raising=True``.  On the pre-AR014 code the attribute does not
exist, so the fixture raises ``AttributeError`` and every test in this file
is stably red.  ``raising=False`` was rejected because monkeypatch would
then *create* the attribute: the registry assertions would compare against
an always-empty set and UT-1304 (zero registry growth under the manual
switch) would spuriously pass against the old code — exactly the
"attribute missing masked as green" failure mode.  Once the attribute
exists (Green), the same ``setattr`` swaps in a fresh ``set`` per test and
the monkeypatch teardown restores the original binding (design D5).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import pytest
import torch

from actfold.core import fused_ops
from actfold.core.fused_ops import _HAS_TRITON, merge_stable_divergent


@pytest.fixture(autouse=True)
def _fresh_failed_keys_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the process-level failure registry with a fresh set per test (design D5)."""
    monkeypatch.setattr(fused_ops, "_TRITON_FAILED_KEYS", set())


def _merge_reference(parent: torch.Tensor, child: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Hand-written torch mirror of the merge (never the production fallback)."""
    return torch.where(mask.unsqueeze(-1), parent, child)


def _merge_inputs(
    batch: int, seq: int, hidden: int, dtype: torch.dtype, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic CUDA inputs ``(parent, child, mask)`` for the merge."""
    torch.manual_seed(seed)
    parent = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    child = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    mask = torch.rand(batch, seq, device="cuda") > 0.5
    return parent, child, mask


def _forbidden_fallback(*args: object, **kwargs: object) -> torch.Tensor:
    """Stand-in for the PyTorch merge fallback that must never be called."""
    raise AssertionError("fallback used")


class _BoomKernel:
    """Fake Triton kernel whose launch always raises RuntimeError('boom')."""

    def __getitem__(self, grid: object) -> Any:
        """Pretend the kernel launch fails, like ``kernel[grid](...)`` would."""

        def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError("boom")

        return _raise


class _ConditionalBoomKernel:
    """Fake Triton merge kernel: booms for matching inputs, delegates otherwise.

    ``should_boom`` inspects the first launch argument (the parent tensor), so
    one monkeypatch provides both a failing key (dtype- or shape-conditioned)
    and a real-kernel path for every other key (design 3.1).  ``real_kernel``
    must be captured from ``fused_ops`` before any kernel monkeypatching.
    """

    def __init__(self, real_kernel: Any, should_boom: Callable[[torch.Tensor], bool]) -> None:
        self._real_kernel = real_kernel
        self._should_boom = should_boom

    def __getitem__(self, grid: object) -> Any:
        def _launch(*args: object, **kwargs: object) -> None:
            parent = args[0]
            assert isinstance(parent, torch.Tensor)
            if self._should_boom(parent):
                raise RuntimeError("boom")
            self._real_kernel[grid](*args, **kwargs)

        return _launch


class _LaunchCountSpyKernel:
    """Counts launches and delegates to the real kernel (design 3.1)."""

    def __init__(self, real_kernel: Any) -> None:
        self.launch_count = 0
        self._real_kernel = real_kernel

    def __getitem__(self, grid: object) -> Any:
        def _launch(*args: object, **kwargs: object) -> None:
            self.launch_count += 1
            self._real_kernel[grid](*args, **kwargs)

        return _launch


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ut1301_dtype_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1301 (srs 3.4): a failed dtype key does not silence other dtypes.

    Three steps (design 3.2, revised forbidden scoping): (1) an fp32-conditioned
    boom kernel fails the fp32 key with a warning and a *legal* fallback;
    (2) with the fallback forbidden in between, the fp16 call (same bucket)
    must go through the real Triton kernel; (3) after restoring the fallback,
    an fp16-conditioned boom kernel registers the second, independent failure
    key.  The registry ends up holding exactly the two dtype keys.
    """
    monkeypatch.setattr(fused_ops, "_TRITON_MERGE_DISABLED", False)
    real_kernel = fused_ops._merge_kernel
    batch, seq, hidden = 2, 8, 128
    parent32, child32, mask32 = _merge_inputs(batch, seq, hidden, torch.float32, 1311)
    parent16, child16, mask16 = _merge_inputs(batch, seq, hidden, torch.float16, 1312)

    monkeypatch.setattr(
        fused_ops,
        "_merge_kernel",
        _ConditionalBoomKernel(real_kernel, lambda p: p.dtype == torch.float32),
    )
    with pytest.warns(RuntimeWarning, match="merge kernel unavailable"):
        out32 = merge_stable_divergent(parent32, child32, mask32)
    assert torch.equal(out32, _merge_reference(parent32, child32, mask32))

    original_fallback = fused_ops._merge_stable_divergent_torch
    monkeypatch.setattr(fused_ops, "_merge_stable_divergent_torch", _forbidden_fallback)
    out16 = merge_stable_divergent(parent16, child16, mask16)
    assert torch.equal(out16, _merge_reference(parent16, child16, mask16))

    monkeypatch.setattr(fused_ops, "_merge_stable_divergent_torch", original_fallback)
    monkeypatch.setattr(
        fused_ops,
        "_merge_kernel",
        _ConditionalBoomKernel(real_kernel, lambda p: p.dtype == torch.float16),
    )
    with pytest.warns(RuntimeWarning, match="merge kernel unavailable"):
        out16b = merge_stable_divergent(parent16, child16, mask16)
    assert torch.equal(out16b, _merge_reference(parent16, child16, mask16))

    assert fused_ops._TRITON_FAILED_KEYS == {
        ("merge", "torch.float32", 128),
        ("merge", "torch.float16", 128),
    }


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ut1302_same_key_silent_second_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1302 (srs 3.4): a registered key silently falls back without re-attempting.

    The first call with an unconditional boom kernel warns once, returns the
    correct fallback result and registers exactly one key.  Before the second
    call the kernel is replaced by a launch-counting spy that delegates to the
    real kernel: the second call must not attempt the kernel at all (spy
    count == 0), must not warn, and must still return the correct result.
    """
    monkeypatch.setattr(fused_ops, "_TRITON_MERGE_DISABLED", False)
    real_kernel = fused_ops._merge_kernel
    parent, child, mask = _merge_inputs(2, 8, 128, torch.float32, 1321)
    expected = _merge_reference(parent, child, mask)

    monkeypatch.setattr(fused_ops, "_merge_kernel", _BoomKernel())
    with pytest.warns(RuntimeWarning, match="merge kernel unavailable"):
        out1 = merge_stable_divergent(parent, child, mask)
    assert torch.equal(out1, expected)
    assert fused_ops._TRITON_FAILED_KEYS == {("merge", "torch.float32", 128)}

    spy = _LaunchCountSpyKernel(real_kernel)
    monkeypatch.setattr(fused_ops, "_merge_kernel", spy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out2 = merge_stable_divergent(parent, child, mask)
    assert spy.launch_count == 0
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert torch.equal(out2, expected)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ut1303_bucket_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1303 (srs 3.4): a failed hidden bucket does not silence other buckets.

    Same dtype (fp32): the H=512 call fails via a shape-conditioned boom kernel
    and falls back legally; the H=2048 call (different bucket) must still take
    the real Triton kernel, proven with the forbidden fallback active only for
    that call (design 3.2: failure-path fallbacks must stay legal).  The
    registry ends up holding exactly the 512 key.
    """
    monkeypatch.setattr(fused_ops, "_TRITON_MERGE_DISABLED", False)
    real_kernel = fused_ops._merge_kernel
    parent512, child512, mask512 = _merge_inputs(2, 8, 512, torch.float32, 1331)
    parent2048, child2048, mask2048 = _merge_inputs(2, 8, 2048, torch.float32, 1332)

    monkeypatch.setattr(
        fused_ops,
        "_merge_kernel",
        _ConditionalBoomKernel(real_kernel, lambda p: p.shape[2] == 512),
    )
    with pytest.warns(RuntimeWarning, match="merge kernel unavailable"):
        out512 = merge_stable_divergent(parent512, child512, mask512)
    assert torch.equal(out512, _merge_reference(parent512, child512, mask512))

    original_fallback = fused_ops._merge_stable_divergent_torch
    monkeypatch.setattr(fused_ops, "_merge_stable_divergent_torch", _forbidden_fallback)
    out2048 = merge_stable_divergent(parent2048, child2048, mask2048)
    assert torch.equal(out2048, _merge_reference(parent2048, child2048, mask2048))
    monkeypatch.setattr(fused_ops, "_merge_stable_divergent_torch", original_fallback)

    assert fused_ops._TRITON_FAILED_KEYS == {("merge", "torch.float32", 512)}


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ut1304_manual_kill_switch_forces_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1304 (srs 3.4): the manual kill-switch forces the fallback for unregistered keys.

    With ``_TRITON_MERGE_DISABLED=True`` the merge returns the correct PyTorch
    result, emits no warning, and leaves the failure registry untouched.  The
    teardown restoration of the flag is monkeypatch behavior and is
    deliberately not asserted (design 3.2).
    """
    monkeypatch.setattr(fused_ops, "_TRITON_MERGE_DISABLED", True)
    parent, child, mask = _merge_inputs(2, 8, 128, torch.float32, 1341)
    registry_before = set(fused_ops._TRITON_FAILED_KEYS)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = merge_stable_divergent(parent, child, mask)
    assert torch.equal(out, _merge_reference(parent, child, mask))
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert fused_ops._TRITON_FAILED_KEYS == registry_before


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ut1305_warning_is_diagnosable(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-1305 (srs 3.4): the first-failure warning names kernel, dtype and bucket.

    The warning raised for a first fp32 failure at H=128 must contain the
    substrings "merge kernel unavailable", "torch.float32" and
    "hidden-class=128" (design 3.2, reusing the UT-1301 step-1 capture shape).
    """
    monkeypatch.setattr(fused_ops, "_TRITON_MERGE_DISABLED", False)
    real_kernel = fused_ops._merge_kernel
    parent, child, mask = _merge_inputs(2, 8, 128, torch.float32, 1351)

    monkeypatch.setattr(
        fused_ops,
        "_merge_kernel",
        _ConditionalBoomKernel(real_kernel, lambda p: p.dtype == torch.float32),
    )
    with pytest.warns(RuntimeWarning, match="merge kernel unavailable") as record:
        merge_stable_divergent(parent, child, mask)
    message = str(record[0].message)
    assert "torch.float32" in message
    assert "hidden-class=128" in message


@pytest.mark.parametrize(
    ("hidden_dim", "expected"),
    [(4096, 4096), (5000, 8192), (512, 512), (1, 1), (2, 2)],
)
def test_ut1306_hidden_bucket(hidden_dim: int, expected: int) -> None:
    """UT-1306 (srs 3.4): ``_hidden_bucket`` is the pow2 ceiling of the hidden dim.

    Pure function pinned on CPU with no CUDA/Triton gate (design 3.2).
    """
    assert fused_ops._hidden_bucket(hidden_dim) == expected
