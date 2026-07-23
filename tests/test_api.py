from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

EXPECTED_TOKEN_ID = 444
TEST_VOCAB_SIZE = 512


def _load_api_module(monkeypatch: pytest.MonkeyPatch, *, resolved_device: str = "cpu"):
    fake_model = types.ModuleType("model")

    class FakeOctoTetrahedralModel:
        def load_state_dict(self, *_args, **_kwargs):
            return None

        def eval(self):
            return self

        def to(self, device):
            self.device = str(device)
            return self

        def __call__(self, input_ids, return_confidences=False):
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros((batch_size, seq_len, TEST_VOCAB_SIZE), device=input_ids.device)
            logits[..., EXPECTED_TOKEN_ID] = 1.0
            return {"logits": logits}

    fake_model.OctoTetrahedralModel = FakeOctoTetrahedralModel

    fake_config = types.ModuleType("config")
    fake_config.get_config = lambda: SimpleNamespace(device="auto")

    fake_gpu = types.ModuleType("gpu_support")
    fake_gpu.detect_device = lambda _preferred=None: SimpleNamespace(
        requested="auto",
        resolved=resolved_device,
        accelerator=None if resolved_device == "cpu" else resolved_device,
        fallback_used=False,
        reason="test",
    )
    fake_gpu.build_benchmark_comparison = lambda: {"speedup_factor": 10.0}

    monkeypatch.setitem(sys.modules, "model", fake_model)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "gpu_support", fake_gpu)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {"model_state_dict": {}})
    monkeypatch.delitem(sys.modules, "api", raising=False)

    return importlib.import_module("api")


def test_predict_returns_predictions_for_valid_input(monkeypatch: pytest.MonkeyPatch):
    api = _load_api_module(monkeypatch)
    client = TestClient(api.app)

    response = client.post("/predict", json={"input_ids": [1, 2, 3]})

    assert response.status_code == 200
    assert response.json()["predictions"] == [[EXPECTED_TOKEN_ID, EXPECTED_TOKEN_ID, EXPECTED_TOKEN_ID]]
    assert response.json()["device"] == "cpu"


def test_predict_rejects_empty_input(monkeypatch: pytest.MonkeyPatch):
    api = _load_api_module(monkeypatch)
    client = TestClient(api.app)

    response = client.post("/predict", json={"input_ids": []})

    assert response.status_code == 400
    assert response.json()["detail"] == "input_ids must contain at least 1 token."


def test_predict_rejects_oversized_batch(monkeypatch: pytest.MonkeyPatch):
    api = _load_api_module(monkeypatch)
    client = TestClient(api.app)

    response = client.post("/predict", json={"input_ids": list(range(257))})

    assert response.status_code == 413
    assert response.json()["detail"] == "input_ids must contain no more than 256 tokens."


def test_predict_rejects_invalid_token_types(monkeypatch: pytest.MonkeyPatch):
    api = _load_api_module(monkeypatch)
    client = TestClient(api.app)

    response = client.post("/predict", json={"input_ids": [1, True, 3]})

    assert response.status_code == 400
    assert response.json()["detail"] == "input_ids[1] must be an integer."


def test_predict_rejects_out_of_range_tokens(monkeypatch: pytest.MonkeyPatch):
    api = _load_api_module(monkeypatch)
    client = TestClient(api.app)

    response = client.post("/predict", json={"input_ids": [50_001]})

    assert response.status_code == 400
    assert response.json()["detail"] == "input_ids[0] must be between 0 and 50000."


def test_health_reports_resolved_device(monkeypatch: pytest.MonkeyPatch):
    api = _load_api_module(monkeypatch, resolved_device="mps")
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["device"] == "mps"
    assert response.json()["expected_speedup_factor"] == 10.0


def test_detect_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch):
    import gpu_support

    monkeypatch.setattr(gpu_support, "_cuda_available", lambda: True)
    monkeypatch.setattr(gpu_support, "_mps_available", lambda: True)
    monkeypatch.setattr(gpu_support, "_smoke_test", lambda name: name.startswith("cuda"))

    info = gpu_support.detect_device()

    assert info.resolved == "cuda"
    assert info.accelerator == "cuda"
    assert info.fallback_used is False


def test_detect_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch):
    import gpu_support

    monkeypatch.setattr(gpu_support, "_cuda_available", lambda: False)
    monkeypatch.setattr(gpu_support, "_mps_available", lambda: False)
    monkeypatch.setattr(gpu_support, "_smoke_test", lambda _name: False)

    info = gpu_support.detect_device("cuda")

    assert info.resolved == "cpu"
    assert info.fallback_used is True
    assert "fallback" in info.reason.lower()


def test_build_benchmark_comparison_reports_speedup():
    from gpu_support import build_benchmark_comparison

    comparison = build_benchmark_comparison()

    assert comparison["speedup_factor"] > 9.0
    assert comparison["accelerator_throughput_rps"] > comparison["cpu_throughput_rps"]
