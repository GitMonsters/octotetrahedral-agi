from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


class _DummyModel:
    def load_state_dict(self, state_dict, strict=False):
        return None

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids, return_confidences=False):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, 2), dtype=torch.float32)
        logits[..., 1] = 1.0
        return {"logits": logits}


@pytest.fixture
def client(monkeypatch) -> TestClient:
    original_model_module = sys.modules.get("model")

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})

    fake_model_module = types.ModuleType("model")
    fake_model_module.OctoTetrahedralModel = _DummyModel
    monkeypatch.setitem(sys.modules, "model", fake_model_module)

    sys.modules.pop("api", None)
    api = importlib.import_module("api")

    try:
        yield TestClient(api.app)
    finally:
        sys.modules.pop("api", None)
        if original_model_module is not None:
            sys.modules["model"] = original_model_module
        else:
            sys.modules.pop("model", None)


@pytest.mark.parametrize(
    ("payload", "expected_length"),
    [
        ({"input_ids": [1]}, 1),
        ({"input_ids": [100, 200, 300]}, 3),
        ({"input_ids": list(range(256))}, 256),
    ],
)
def test_predict_accepts_valid_requests(client: TestClient, payload: dict, expected_length: int):
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["predictions"][0]) == expected_length


@pytest.mark.parametrize(
    ("payload", "expected_detail"),
    [
        ({"input_ids": []}, "at least 1 token"),
        ({"input_ids": [-1]}, "between 0 and 50000"),
        ({"input_ids": [50001]}, "between 0 and 50000"),
    ],
)
def test_predict_rejects_bad_request_inputs(
    client: TestClient,
    payload: dict,
    expected_detail: str,
):
    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert expected_detail in response.json()["detail"]


def test_predict_rejects_large_payload(client: TestClient):
    response = client.post("/predict", json={"input_ids": list(range(1000))})

    assert response.status_code == 413
    assert "256 tokens" in response.json()["detail"]
