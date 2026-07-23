from __future__ import annotations

import torch

import gpu_support


def test_resolve_device_prefers_mps_when_available(monkeypatch):
    monkeypatch.setattr(gpu_support, "cuda_available", lambda: False)
    monkeypatch.setattr(gpu_support, "mps_available", lambda: True)
    monkeypatch.setattr(gpu_support, "mps_built", lambda: True)
    monkeypatch.setattr(gpu_support.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu_support.platform, "mac_ver", lambda: ("14.5.0", ("", "", ""), ""))
    monkeypatch.setattr(gpu_support, "_smoke_test", lambda device: (True, None))

    resolution = gpu_support.resolve_device()

    assert resolution.selected == "mps"
    assert resolution.backend == "mps"
    assert resolution.fallback_reason is None


def test_resolve_device_falls_back_when_forced_mps_unavailable(monkeypatch):
    monkeypatch.setattr(gpu_support, "cuda_available", lambda: False)
    monkeypatch.setattr(gpu_support, "mps_available", lambda: False)
    monkeypatch.setattr(gpu_support, "mps_built", lambda: False)
    monkeypatch.setattr(gpu_support.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu_support.platform, "mac_ver", lambda: ("14.5.0", ("", "", ""), ""))
    monkeypatch.setenv("OCTO_DEVICE", "mps")

    resolution = gpu_support.resolve_device()

    assert resolution.selected == "cpu"
    assert "MPS" in resolution.fallback_reason


def test_prepare_input_tensor_uses_long_dtype():
    tensor = gpu_support.prepare_input_tensor([1, 2, 3], "cpu")

    assert tensor.dtype == torch.long
    assert tensor.shape == (1, 3)
    assert tensor.device.type == "cpu"


def test_benchmark_inference_returns_expected_metrics():
    result = gpu_support.benchmark_inference(device="cpu", runs=2, warmup_runs=1)

    assert result["device"] == "cpu"
    assert result["latency_ms_avg"] >= 0
    assert result["latency_ms_min"] >= 0
    assert result["latency_ms_max"] >= result["latency_ms_min"]
    assert result["throughput_tokens_per_second"] > 0
    assert "allocated_mb" in result["memory"]


def test_render_benchmark_table_includes_speedup_column():
    table = gpu_support.render_benchmark_table(
        [
            {
                "device": "cpu",
                "latency_ms_avg": 10.0,
                "throughput_tokens_per_second": 100.0,
                "memory": {"allocated_mb": None},
                "speedup_vs_cpu": 1.0,
            }
        ]
    )

    assert "Speedup vs CPU" in table
    assert "1.00x" in table
