"""Tests for the /predict and /health API endpoints.

Uses FastAPI's TestClient with the model-loading code patched out so
these tests run without a real checkpoint.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies so the test suite never needs a real checkpoint
# ---------------------------------------------------------------------------


def _make_mock_model():
    """Return a tiny mock that mimics OctoTetrahedralModel forward output."""
    import torch

    mock = MagicMock()

    def _forward(**kwargs):
        input_ids = kwargs.get("input_ids")
        seq_len = input_ids.shape[-1] if input_ids is not None else 1
        logits = torch.zeros(1, seq_len, 100)
        return {"logits": logits}

    mock.return_value = _forward  # model() returns a callable
    mock.side_effect = None
    mock.__call__ = lambda self, **kw: _forward(**kw)
    return mock


# Patch model loading at the module level before importing api
_mock_model_instance = MagicMock()
_mock_model_instance.__call__ = MagicMock(
    side_effect=lambda **kw: {
        "logits": __import__("torch").zeros(
            1, max(len(kw.get("input_ids", [[1]])[0]), 1), 100
        )
    }
)
_mock_model_instance.eval = MagicMock(return_value=_mock_model_instance)
_mock_model_instance.to = MagicMock(return_value=_mock_model_instance)


@pytest.fixture(scope="module", autouse=True)
def _patch_api_deps():
    """Patch model, checkpoint loading, and gpu_support before importing api."""
    import torch

    with (
        patch("builtins.__import__", _safe_import),
        patch.dict(
            "sys.modules",
            {
                "gpu_support": _make_gpu_support_stub(),
            },
        ),
    ):
        # Patch torch.load to avoid needing a checkpoint file
        original_load = torch.load
        torch.load = MagicMock(return_value={"dummy_key": torch.zeros(1)})

        # We need OctoTetrahedralModel to be importable and return a mock
        _inject_model_stub()

        # Now import (or re-import) the api module
        if "api" in sys.modules:
            del sys.modules["api"]

        import importlib

        api_module = importlib.import_module("api")
        # Patch the global model in the api module
        api_module.model = _mock_model_instance

        yield api_module

        torch.load = original_load


def _make_gpu_support_stub():
    """Return a minimal gpu_support stub module."""
    stub = types.ModuleType("gpu_support")
    stub.resolve_device = lambda: {"device": "cpu", "accelerator": "cpu", "backend": "stub"}
    return stub


def _inject_model_stub():
    """Ensure model.OctoTetrahedralModel is importable as a mock."""
    if "model" not in sys.modules:
        stub = types.ModuleType("model")
        stub.OctoTetrahedralModel = MagicMock(return_value=_mock_model_instance)
        sys.modules["model"] = stub
    else:
        sys.modules["model"].OctoTetrahedralModel = MagicMock(
            return_value=_mock_model_instance
        )


_ORIGINAL_IMPORT = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _safe_import(name, *args, **kwargs):
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(_patch_api_deps):
    from fastapi.testclient import TestClient

    return TestClient(_patch_api_deps.app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "device" in data


class TestPredictValidInput:
    def test_single_token(self, client):
        resp = client.post("/predict", json={"input_ids": [1]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert isinstance(data["predictions"], list)

    def test_normal_batch(self, client):
        resp = client.post("/predict", json={"input_ids": list(range(10))})
        assert resp.status_code == 200

    def test_max_allowed_batch(self, client):
        resp = client.post("/predict", json={"input_ids": list(range(256))})
        assert resp.status_code == 200

    def test_boundary_token_ids(self, client):
        resp = client.post("/predict", json={"input_ids": [0, 50000]})
        assert resp.status_code == 200


class TestPredictInvalidInput:
    def test_empty_input_returns_400(self, client):
        resp = client.post("/predict", json={"input_ids": []})
        assert resp.status_code == 400
        assert "at least 1 token" in resp.json()["detail"]

    def test_oversized_batch_returns_413(self, client):
        resp = client.post("/predict", json={"input_ids": list(range(1000))})
        assert resp.status_code == 413
        assert "no more than 256 tokens" in resp.json()["detail"]

    def test_negative_token_id_returns_400(self, client):
        resp = client.post("/predict", json={"input_ids": [-1]})
        assert resp.status_code == 400
        assert "between 0 and 50000" in resp.json()["detail"]

    def test_out_of_range_token_id_returns_400(self, client):
        resp = client.post("/predict", json={"input_ids": [99999]})
        assert resp.status_code == 400
        assert "between 0 and 50000" in resp.json()["detail"]

    def test_non_integer_token_id_returns_400(self, client):
        resp = client.post("/predict", json={"input_ids": [1, "hello", 3]})
        assert resp.status_code == 400

    def test_boolean_token_id_returns_400(self, client):
        resp = client.post("/predict", json={"input_ids": [1, True, 3]})
        assert resp.status_code == 400
        assert "integer" in resp.json()["detail"]
