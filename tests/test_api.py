from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

import torch
from fastapi.testclient import TestClient

import api
from gpu_support import DeviceResolution


class DummyModel(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor, return_confidences: bool = False):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 4, device=input_ids.device)
        logits[..., 2] = 1.0
        return {"logits": logits}


def _make_app(device_resolution: DeviceResolution | None = None):
    return api.create_app(
        model_instance=DummyModel(),
        device_resolution=device_resolution
        or DeviceResolution(requested=None, selected="cpu", backend="cpu"),
        initialize=False,
    )


def test_health_reports_device_metadata():
    app = _make_app(
        DeviceResolution(
            requested="mps",
            selected="cpu",
            backend="cpu",
            fallback_reason="MPS backend is unavailable on this machine.",
            mps_available=False,
            mps_built=False,
        )
    )

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["device"] == "cpu"
    assert payload["device_backend"] == "cpu"
    assert payload["device_fallback_reason"] == "MPS backend is unavailable on this machine."


def test_predict_returns_predictions_and_device():
    client = TestClient(_make_app())

    response = client.post("/predict", json={"input_ids": [1, 2, 3]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["predictions"] == [[2, 2, 2]]
    assert payload["device"] == "cpu"
    assert payload["success"] is True


def test_predict_returns_503_when_model_unavailable():
    app = api.create_app(
        model_instance=None,
        device_resolution=DeviceResolution(requested=None, selected="cpu", backend="cpu"),
        initialize=False,
        model_error="checkpoint missing",
    )
    client = TestClient(app)

    response = client.post("/predict", json={"input_ids": [1]})

    assert response.status_code == 503
    assert response.json()["detail"] == "checkpoint missing"
