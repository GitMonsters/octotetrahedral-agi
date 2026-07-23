"""Tests for gpu_support device detection and utilities."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import gpu_support


class TestResolveDevice:
    def test_returns_dict_with_device_key(self):
        info = gpu_support.resolve_device()
        assert "device" in info
        assert info["device"] in ("cpu", "cuda", "mps")

    def test_env_override_cuda(self, monkeypatch):
        monkeypatch.setenv("OCTO_DEVICE", "cuda")
        with patch("gpu_support.os.getenv", side_effect=lambda k, *a: "cuda" if k == "OCTO_DEVICE" else None):
            info = gpu_support.resolve_device()
        assert info["device"] == "cuda"
        assert info["backend"] == "env-override"

    def test_env_override_cpu(self, monkeypatch):
        monkeypatch.setenv("OCTOTETRAHEDRAL_DEVICE", "cpu")
        with patch("gpu_support.os.getenv", side_effect=lambda k, *a: "cpu" if k == "OCTOTETRAHEDRAL_DEVICE" else None):
            info = gpu_support.resolve_device()
        assert info["device"] == "cpu"
        assert info["backend"] == "env-override"

    def test_fallback_to_cpu_when_no_accelerators(self):
        with (
            patch("gpu_support.os.getenv", return_value=None),
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch("gpu_support._smoke_test_cuda", return_value=False),
        ):
            info = gpu_support.resolve_device()
            assert info["device"] == "cpu"

    def test_mps_chosen_when_available(self):
        with (
            patch("gpu_support.os.getenv", return_value=None),
            patch.object(torch.backends.mps, "is_available", return_value=True),
            patch("gpu_support._smoke_test_cuda", return_value=False),
            patch("gpu_support._smoke_test_mps", return_value=True),
        ):
            info = gpu_support.resolve_device()
            assert info["device"] == "mps"
            assert info["backend"] == "torch-mps"

    def test_mps_fallback_when_smoke_test_fails(self):
        with (
            patch("gpu_support.os.getenv", return_value=None),
            patch.object(torch.backends.mps, "is_available", return_value=True),
            patch("gpu_support._smoke_test_cuda", return_value=False),
            patch("gpu_support._smoke_test_mps", return_value=False),
        ):
            info = gpu_support.resolve_device()
            assert info["device"] == "cpu"


class TestClearDeviceCache:
    def test_clear_cpu_is_noop(self):
        gpu_support.clear_device_cache("cpu")  # should not raise

    def test_clear_cuda_calls_empty_cache(self):
        with patch.object(torch.cuda, "empty_cache") as mock_ec:
            gpu_support.clear_device_cache("cuda")
            mock_ec.assert_called_once()

    def test_clear_mps_calls_empty_cache(self):
        with patch.object(torch.mps, "empty_cache") as mock_ec:
            gpu_support.clear_device_cache("mps")
            mock_ec.assert_called_once()


class TestBenchmarkDevice:
    def test_cpu_benchmark_returns_stats(self):
        stats = gpu_support.benchmark_device("cpu", n_iter=5)
        assert stats["latency_ms_mean"] > 0
        assert stats["latency_ms_min"] <= stats["latency_ms_mean"]
        assert stats["latency_ms_max"] >= stats["latency_ms_mean"]
        assert stats["throughput_ops_per_sec"] > 0
