import importlib
import sys
import types
from contextlib import nullcontext

from fastapi.testclient import TestClient
import pytest


class _DummyMonitor:
    def __init__(self):
        self.total_requests = 0
        self.error_count = 0

    def record_request(self, _latency_ms, error=False):
        self.total_requests += 1
        if error:
            self.error_count += 1

    def get_stats(self):
        return {
            "total_requests": self.total_requests,
            "avg_latency_ms": 0,
            "error_count": self.error_count,
            "memory_mb": 0,
        }


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def to(self, _device):
        return self


class _FakeLogits:
    def argmax(self, dim=-1):
        class _Pred:
            @staticmethod
            def tolist():
                return [[0]]

        return _Pred()


class _FakeModel:
    def load_state_dict(self, _state_dict, strict=False):
        return None

    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, input_ids=None, return_confidences=False):
        return {"logits": _FakeLogits()}


@pytest.fixture
def api_module(monkeypatch):
    # api.py imports torch/model and loads checkpoints at module import time, so
    # we provide lightweight stubs to isolate Ollama endpoint behavior in tests.
    fake_torch = types.ModuleType("torch")
    fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda name: types.SimpleNamespace(type=name, __str__=lambda self: name)
    fake_torch.tensor = lambda value: _FakeTensor(value)
    fake_torch.no_grad = lambda: nullcontext()
    fake_torch.load = lambda *args, **kwargs: {}

    fake_model = types.ModuleType("model")
    fake_model.OctoTetrahedralModel = _FakeModel

    fake_auth = types.ModuleType("auth")
    fake_auth.validate_api_key = lambda key: True

    fake_monitoring = types.ModuleType("monitoring")
    fake_monitoring.monitor = _DummyMonitor()

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "model", fake_model)
    monkeypatch.setitem(sys.modules, "auth", fake_auth)
    monkeypatch.setitem(sys.modules, "monitoring", fake_monitoring)
    sys.modules.pop("api", None)

    return importlib.import_module("api")


@pytest.fixture
def client(api_module):
    api_module.app.dependency_overrides[api_module.verify_api_key] = lambda: "test-api-key"
    return TestClient(api_module.app)


def test_ask_endpoint_uses_ollama(client, api_module, monkeypatch):
    monkeypatch.setattr(api_module, "_run_ollama_chat", lambda messages, **kwargs: ("Real AGI answer", "mistral"))

    response = client.post("/ask", json={"question": "What is AGI?"})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["answer"] == "Real AGI answer"
    assert data["model"] == "mistral"


def test_prompt_endpoint_uses_ollama(client, api_module, monkeypatch):
    monkeypatch.setattr(api_module, "_run_ollama_chat", lambda messages, **kwargs: ("Prompt output", "mistral"))

    response = client.post(
        "/prompt",
        json={"prompt": "Explain transformers", "mode": "technical", "max_length": 64, "temperature": 0.2, "top_p": 0.8},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["response"] == "Prompt output"
    assert data["mode"] == "technical"
    assert data["model"] == "mistral"


def test_chat_endpoint_uses_ollama(client, api_module, monkeypatch):
    monkeypatch.setattr(api_module, "_run_ollama_chat", lambda messages, **kwargs: ("Chat output", "llama3.2"))

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hello"}], "system_prompt": "Be concise"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["response"] == "Chat output"
    assert data["model"] == "llama3.2"


def test_chat_endpoint_invalid_role_returns_400(client):
    response = client.post(
        "/chat",
        json={"messages": [{"role": "tool", "content": "hi"}]},
    )

    assert response.status_code == 400
    assert "Invalid message role" in response.json()["detail"]


def test_command_endpoint_uses_ollama(client, api_module, monkeypatch):
    monkeypatch.setattr(api_module, "_run_ollama_chat", lambda messages, **kwargs: ("Short summary", "mistral"))

    response = client.post(
        "/command",
        json={"command": "summarize", "input_text": "Long text to summarize"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["output"] == "Short summary"
    assert data["model"] == "mistral"


def test_command_unknown_still_returns_400(client):
    response = client.post(
        "/command",
        json={"command": "unknown", "input_text": "text"},
    )
    assert response.status_code == 400
    assert "Unknown command" in response.json()["detail"]


def test_ollama_unavailable_returns_503(client, api_module, monkeypatch):
    def _raise_unavailable(*args, **kwargs):
        raise api_module.OllamaUnavailableError("Unable to connect to Ollama")

    monkeypatch.setattr(api_module, "_run_ollama_chat", _raise_unavailable)
    response = client.post("/ask", json={"question": "Hi"})

    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_health_includes_ollama_status(client, api_module, monkeypatch):
    monkeypatch.setattr(api_module, "_ollama_health", lambda: {"status": "healthy", "model": "mistral", "fallback_models": []})

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "ollama" in data
    assert data["ollama"]["status"] == "healthy"
