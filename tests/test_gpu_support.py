"""Tests for Metal/MPS GPU support (gpu_metal_support.py).

These tests are hardware-agnostic and run on CPU-only CI environments.
"""

from __future__ import annotations

import torch
import pytest

from gpu_metal_support import (
    device_info,
    get_metal_device,
    is_metal_available,
    select_device,
)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def test_is_metal_available_returns_bool():
    result = is_metal_available()
    assert isinstance(result, bool)


def test_get_metal_device_returns_none_or_mps():
    dev = get_metal_device()
    if is_metal_available():
        assert dev is not None
        assert dev.type == "mps"
    else:
        assert dev is None


def test_select_device_returns_torch_device():
    dev = select_device()
    assert isinstance(dev, torch.device)


def test_select_device_falls_back_to_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    dev = select_device()
    assert dev.type == "cpu"


def test_select_device_respects_env_override(monkeypatch):
    monkeypatch.setenv("OCTO_DEVICE", "cpu")
    dev = select_device()
    assert dev.type == "cpu"


def test_device_info_returns_expected_keys():
    info = device_info()
    assert "cuda_available" in info
    assert "mps_available" in info
    assert "selected_device" in info
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["mps_available"], bool)


def test_device_info_mps_field_matches_availability():
    info = device_info()
    assert info["mps_available"] == is_metal_available()


# ---------------------------------------------------------------------------
# Optimization helpers (import-only checks on non-Metal environments)
# ---------------------------------------------------------------------------


def test_optimize_for_metal_importable():
    from gpu_metal_support import optimize_for_metal  # noqa: F401

    assert callable(optimize_for_metal)


def test_compare_cpu_vs_metal_importable():
    from gpu_metal_support import compare_cpu_vs_metal  # noqa: F401

    assert callable(compare_cpu_vs_metal)
