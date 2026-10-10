"""Red tests for T021: hardware-parameterized cost model + attention T^2/KV fix.

These tests encode the GREEN contract and are expected to fail against the
current implementation:

1. A module-level ``_DEVICE_TABLE`` mapping GPU name substrings to
   (bf16 dense TFLOPS, memory GB/s).
2. ``HardwareProfile.from_device(device, calibrate=False)`` resolving real GPU
   names instead of hardcoded conservative defaults.
3. The attention quadratic (T^2) term in compute FLOPs, KV read / gate read /
   merge traffic counted in memory bytes, and gate/merge no longer modelled as
   compute FLOPs.
4. ``estimate_total_time`` keeps its layer-count / step-count scaling.
"""

from __future__ import annotations

import pytest
import torch

from actfold.utils.cost_model import ComputeBandwidthCostModel, HardwareProfile


def _expected_compute_flops(seq_len: int, hidden_dim: int, stable_ratio: float) -> float:
    """Green contract: linear attention+FFN term plus the quadratic T^2 term."""
    t = float(seq_len)
    h = float(hidden_dim)
    r = float(stable_ratio)
    return (1.0 - r) * t * h * h * (4.0 + 16.0) + 2.0 * (1.0 - r) * t * t * h


def _expected_gate_bytes(seq_len: int, hidden_dim: int, bytes_per_element: int) -> float:
    """Green contract: gate reads parent + child hidden states."""
    return 2.0 * float(seq_len) * float(hidden_dim) * float(bytes_per_element)


def _expected_merge_bytes(seq_len: int, hidden_dim: int, bytes_per_element: int) -> float:
    """Green contract: read parent FFN + read child FFN + write merged output."""
    return 3.0 * float(seq_len) * float(hidden_dim) * float(bytes_per_element)


def _expected_memory_bytes(
    seq_len: int,
    hidden_dim: int,
    stable_ratio: float,
    bytes_per_element: int,
) -> float:
    """Green contract: stable traffic + KV read + gate reads + merge traffic."""
    t = float(seq_len)
    h = float(hidden_dim)
    r = float(stable_ratio)
    b = float(bytes_per_element)
    stable_traffic = r * t * h * b * 4.0
    kv_read = 2.0 * t * h * b
    gate_reads = _expected_gate_bytes(seq_len, hidden_dim, bytes_per_element)
    merge_traffic = _expected_merge_bytes(seq_len, hidden_dim, bytes_per_element)
    return stable_traffic + kv_read + gate_reads + merge_traffic


def _make_model() -> ComputeBandwidthCostModel:
    hw = HardwareProfile(compute_tflops=100.0, memory_bw_gb_s=600.0, bytes_per_element=2)
    return ComputeBandwidthCostModel(hw)


# ---------------------------------------------------------------------------
# 1. _DEVICE_TABLE
# ---------------------------------------------------------------------------


def test_device_table_contains_known_gpu_entries() -> None:
    """_DEVICE_TABLE maps uppercase GPU name substrings to (TFLOPS, GB/s)."""
    from actfold.utils.cost_model import _DEVICE_TABLE

    expected: dict[str, tuple[float, float]] = {
        "A100": (312.0, 1555.0),
        "H100": (989.0, 3350.0),
        "RTX 6000": (136.0, 1280.0),
        "RTX PRO 6000": (400.0, 1600.0),
        "QUADRO RTX 5000": (60.0, 448.0),
        "4090": (165.0, 1008.0),
    }
    for name, (tflops, gbps) in expected.items():
        assert name in _DEVICE_TABLE, f"missing _DEVICE_TABLE entry for {name!r}"
        assert _DEVICE_TABLE[name][0] == pytest.approx(tflops)
        assert _DEVICE_TABLE[name][1] == pytest.approx(gbps)


# ---------------------------------------------------------------------------
# 2. HardwareProfile.from_device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device_name", "tflops", "gbps"),
    [
        ("NVIDIA Quadro RTX 5000", 60.0, 448.0),
        ("NVIDIA A100-SXM4-80GB", 312.0, 1555.0),
        ("NVIDIA GeForce RTX 4090", 165.0, 1008.0),
    ],
)
def test_from_device_resolves_cuda_device_name(
    monkeypatch: pytest.MonkeyPatch, device_name: str, tflops: float, gbps: float
) -> None:
    """CUDA devices are profiled via torch.cuda.get_device_name lookups."""
    monkeypatch.setattr("torch.cuda.get_device_name", lambda *a, **k: device_name)
    profile = HardwareProfile.from_device("cuda")
    assert profile.compute_tflops == pytest.approx(tflops)
    assert profile.memory_bw_gb_s == pytest.approx(gbps)
    assert profile.bytes_per_element == 2


def test_from_device_unknown_cuda_name_uses_conservative_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown GPU names fall back to conservative defaults; calibrate kwarg exists."""
    monkeypatch.setattr("torch.cuda.get_device_name", lambda *a, **k: "NVIDIA MYSTERY GPU 9000")
    profile = HardwareProfile.from_device("cuda", calibrate=False)
    assert profile.compute_tflops == pytest.approx(100.0)
    assert profile.memory_bw_gb_s == pytest.approx(600.0)
    assert profile.bytes_per_element == 2


def test_from_device_cpu_defaults_ignore_calibrate() -> None:
    """CPU defaults are unchanged and calibrate=True is a no-op on CPU."""
    default_profile = HardwareProfile.from_device("cpu")
    assert default_profile.compute_tflops == pytest.approx(1.0)
    assert default_profile.memory_bw_gb_s == pytest.approx(50.0)
    assert default_profile.bytes_per_element == 4

    calibrated = HardwareProfile.from_device("cpu", calibrate=True)
    assert calibrated.compute_tflops == pytest.approx(1.0)
    assert calibrated.memory_bw_gb_s == pytest.approx(50.0)
    assert calibrated.bytes_per_element == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_from_device_cuda_calibrate_runs_micro_benchmark() -> None:
    """calibrate=True on CUDA micro-benchmarks and returns a positive profile."""
    profile = HardwareProfile.from_device("cuda", calibrate=True)
    assert profile.compute_tflops > 0.0
    assert profile.memory_bw_gb_s > 0.0
    assert profile.bytes_per_element == 2


# ---------------------------------------------------------------------------
# 3. Attention T^2 term, KV bandwidth, gate/merge as bytes
# ---------------------------------------------------------------------------


def test_compute_flops_includes_quadratic_attention_term() -> None:
    """compute_flops = linear term + 2*(1-r)*t*t*h for divergent attention."""
    model = _make_model()
    for t, h, r in [(256, 64, 0.25), (1024, 128, 0.9)]:
        cost = model.layer_cost(seq_len=t, hidden_dim=h, stable_ratio=r)
        assert cost.compute_flops == pytest.approx(_expected_compute_flops(t, h, r))


def test_memory_bytes_includes_kv_gate_and_merge_traffic() -> None:
    """memory_bytes = stable traffic + KV read + gate reads + merge traffic."""
    model = _make_model()
    t, h, r, b = 128, 64, 0.5, 2
    cost = model.layer_cost(seq_len=t, hidden_dim=h, stable_ratio=r)
    assert cost.memory_bytes == pytest.approx(_expected_memory_bytes(t, h, r, b))


def test_layer_cost_reports_gate_and_merge_in_bytes() -> None:
    """LayerCost exposes gate_bytes/merge_bytes and drops gate_flops/merge_flops."""
    model = _make_model()
    t, h, b = 128, 64, 2
    cost = model.layer_cost(seq_len=t, hidden_dim=h, stable_ratio=0.5)
    assert cost.gate_bytes == pytest.approx(_expected_gate_bytes(t, h, b))
    assert cost.merge_bytes == pytest.approx(_expected_merge_bytes(t, h, b))
    assert not hasattr(cost, "gate_flops")
    assert not hasattr(cost, "merge_flops")
    assert cost.estimated_time_ms == pytest.approx(
        model.estimate_layer_time(seq_len=t, hidden_dim=h, stable_ratio=0.5) * 1000.0
    )


def test_estimate_layer_time_sums_compute_and_memory_only() -> None:
    """estimate_layer_time = compute/(TFLOPS) + memory/(GB/s); no gate/merge compute."""
    model = _make_model()
    t, h, r = 128, 64, 0.5
    expected = _expected_compute_flops(t, h, r) / (100.0 * 1e12) + _expected_memory_bytes(
        t, h, r, 2
    ) / (600.0 * 1e9)
    got = model.estimate_layer_time(seq_len=t, hidden_dim=h, stable_ratio=r)
    assert got == pytest.approx(expected)


def test_quadratic_term_dominates_at_long_sequences() -> None:
    """At t=4096 the T^2 term makes compute_flops >10x the t=64 value."""
    model = _make_model()
    h, r = 1024, 0.5
    long_cost = model.layer_cost(seq_len=4096, hidden_dim=h, stable_ratio=r)
    short_cost = model.layer_cost(seq_len=64, hidden_dim=h, stable_ratio=r)
    assert long_cost.compute_flops > 10.0 * short_cost.compute_flops
    assert long_cost.compute_flops == pytest.approx(_expected_compute_flops(4096, h, r))
    assert short_cost.compute_flops == pytest.approx(_expected_compute_flops(64, h, r))


def test_monotonicity_in_stable_ratio_preserved() -> None:
    """compute_flops decreases and memory_bytes increases with stable_ratio."""
    model = _make_model()
    t, h, b = 128, 64, 2
    ratios = (0.0, 0.5, 0.9)
    costs = [model.layer_cost(seq_len=t, hidden_dim=h, stable_ratio=r) for r in ratios]
    assert costs[0].compute_flops > costs[1].compute_flops > costs[2].compute_flops
    assert costs[0].memory_bytes < costs[1].memory_bytes < costs[2].memory_bytes
    for r, cost in zip(ratios, costs):
        assert cost.compute_flops == pytest.approx(_expected_compute_flops(t, h, r))
        assert cost.memory_bytes == pytest.approx(_expected_memory_bytes(t, h, r, b))


# ---------------------------------------------------------------------------
# 4. estimate_total_time semantics
# ---------------------------------------------------------------------------


def test_estimate_total_time_scales_with_layers_and_steps() -> None:
    """estimate_total_time = estimate_layer_time * num_layers * num_steps."""
    model = _make_model()
    t, h, r = 128, 64, 0.5
    layer_time = model.estimate_layer_time(seq_len=t, hidden_dim=h, stable_ratio=r)
    assert layer_time == pytest.approx(
        _expected_compute_flops(t, h, r) / (100.0 * 1e12)
        + _expected_memory_bytes(t, h, r, 2) / (600.0 * 1e9)
    )
    total = model.estimate_total_time(
        num_layers=12,
        seq_len=t,
        hidden_dim=h,
        stable_ratio=r,
        num_steps=3,
        num_heads=1,
    )
    assert total == pytest.approx(layer_time * 12 * 3)
