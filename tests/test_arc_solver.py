"""Tests for the ARC-AGI solver engine (src/arc_solver_engine.py).

All tests are self-contained and require no external services (no Ollama,
no model checkpoint, no GPU).  They exercise:

- Individual rule detectors
- RuleLearner for known transformations
- CatalogLookup (graceful no-op without catalog)
- NeuralInference (graceful no-op without a real model)
- MistralReasoning (graceful no-op without Ollama)
- ARCSolverEngine auto-routing and fallback chain
- /solve-arc API endpoint via FastAPI TestClient
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import nullcontext

import pytest

from src.arc_solver_engine import (
    ARCSolverEngine,
    CatalogLookup,
    MistralReasoning,
    NeuralInference,
    RuleLearner,
    _all_pairs_match,
    _detect_color_map,
    _detect_geometric,
    _detect_gravity,
    _detect_scale_up,
    _detect_tiling,
    _format_arc_prompt,
    _grid_to_tokens,
    _tokens_to_grid,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROT90_TASK = {
    "train": [
        {"input": [[1, 2], [3, 4]], "output": [[2, 4], [1, 3]]},
        {"input": [[5, 6], [7, 8]], "output": [[6, 8], [5, 7]]},
    ],
    "test": [{"input": [[9, 0], [1, 2]]}],
}

COLOR_MAP_TASK = {
    "train": [
        {"input": [[1, 2], [2, 1]], "output": [[3, 4], [4, 3]]},
        {"input": [[1, 1], [2, 2]], "output": [[3, 3], [4, 4]]},
    ],
    "test": [{"input": [[2, 1], [1, 2]]}],
}

SCALE_UP_TASK = {
    "train": [
        {"input": [[1, 2]], "output": [[1, 1, 2, 2], [1, 1, 2, 2]]},
    ],
    "test": [{"input": [[3, 4]]}],
}

TILING_TASK = {
    "train": [
        {"input": [[1, 2]], "output": [[1, 2, 1, 2]]},
    ],
    "test": [{"input": [[5, 6]]}],
}

GRAVITY_TASK = {
    "train": [
        {
            "input": [[0, 1, 0], [0, 0, 2], [0, 0, 0]],
            "output": [[0, 0, 0], [0, 0, 0], [0, 1, 2]],
        },
    ],
    "test": [{"input": [[3, 0, 0], [0, 0, 0], [0, 0, 0]]}],
}

EMPTY_TASK: dict = {"train": [], "test": [{"input": [[0]]}]}


# ---------------------------------------------------------------------------
# _all_pairs_match
# ---------------------------------------------------------------------------


def test_all_pairs_match_true():
    import numpy as np

    fn = lambda x: np.rot90(x, 1)  # noqa: E731
    assert _all_pairs_match(fn, ROT90_TASK["train"])


def test_all_pairs_match_false():
    import numpy as np

    fn = lambda x: x.copy()  # noqa: E731
    assert not _all_pairs_match(fn, ROT90_TASK["train"])


# ---------------------------------------------------------------------------
# Rule Detectors
# ---------------------------------------------------------------------------


def test_detect_geometric_rot90():
    result = _detect_geometric(ROT90_TASK["train"])
    assert result is not None
    name, _ = result
    assert name == "rot90"


def test_detect_geometric_none_for_identity():
    # identity transform: input == output, should NOT match a geometric rule
    task_train = [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}]
    # all geometric transforms change the grid, so this should return None
    result = _detect_geometric(task_train)
    assert result is None


def test_detect_color_map():
    result = _detect_color_map(COLOR_MAP_TASK["train"])
    assert result is not None
    name, fn = result
    assert name == "color_map"
    import numpy as np
    out = fn(np.array([[1, 2], [2, 1]]))
    assert out.tolist() == [[3, 4], [4, 3]]


def test_detect_color_map_none_for_identity():
    train = [{"input": [[1, 2]], "output": [[1, 2]]}]
    assert _detect_color_map(train) is None


def test_detect_scale_up():
    result = _detect_scale_up(SCALE_UP_TASK["train"])
    assert result is not None
    name, fn = result
    assert "scale_up" in name


def test_detect_tiling():
    result = _detect_tiling(TILING_TASK["train"])
    assert result is not None
    name, _ = result
    assert "tiling" in name


def test_detect_gravity_down():
    result = _detect_gravity(GRAVITY_TASK["train"])
    assert result is not None
    name, _ = result
    assert "gravity" in name


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def test_grid_to_tokens_roundtrip():
    grid = [[1, 2, 3], [4, 5, 6]]
    tokens = _grid_to_tokens(grid)
    recovered = _tokens_to_grid(tokens, rows=2, cols=3)
    assert recovered == grid


def test_tokens_to_grid_clamps_values():
    tokens = [100, 200, 300, 400]  # all out of range
    grid = _tokens_to_grid(tokens, rows=1, cols=4)
    assert all(0 <= v <= 9 for v in grid[0])


# ---------------------------------------------------------------------------
# RuleLearner
# ---------------------------------------------------------------------------


class TestRuleLearner:
    def test_solves_rotation(self):
        rl = RuleLearner()
        result = rl.solve(ROT90_TASK)
        assert result is not None
        assert result["method"] == "rule_learner"
        assert result["rule"] == "rot90"
        assert result["confidence"] >= 0.9
        assert result["verified_on_training"] is True
        # Verify prediction has the right structure
        preds = result["predictions"]
        assert len(preds) == 1
        assert isinstance(preds[0], list)

    def test_solves_color_map(self):
        rl = RuleLearner()
        result = rl.solve(COLOR_MAP_TASK)
        assert result is not None
        assert result["rule"] == "color_map"

    def test_solves_scale_up(self):
        rl = RuleLearner()
        result = rl.solve(SCALE_UP_TASK)
        assert result is not None
        assert "scale_up" in result["rule"]

    def test_returns_identity_for_unknown_task(self):
        # Force identity: same input color (1) maps to two different output colors
        # so the color-map detector rejects it, and no geometric/scale rule applies
        task = {
            "train": [
                {"input": [[1, 2], [1, 3]], "output": [[4, 5], [6, 7]]},
            ],
            "test": [{"input": [[0, 1], [2, 3]]}],
        }
        rl = RuleLearner()
        result = rl.solve(task)
        assert result is not None
        assert result["rule"] == "identity"
        assert result["confidence"] < 0.9

    def test_returns_none_for_empty_task(self):
        rl = RuleLearner()
        assert rl.solve(EMPTY_TASK) is None

    def test_prediction_shape_matches_input(self):
        rl = RuleLearner()
        task = {
            "train": [
                {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},
            ],
            "test": [{"input": [[4, 5, 6]]}],
        }
        result = rl.solve(task)
        assert result is not None
        assert result["predictions"][0] == [[6, 5, 4]]


# ---------------------------------------------------------------------------
# CatalogLookup
# ---------------------------------------------------------------------------


class TestCatalogLookup:
    def test_returns_none_without_task_id(self):
        cl = CatalogLookup()
        result = cl.solve(ROT90_TASK, task_id=None)
        assert result is None

    def test_returns_none_for_unknown_task_id(self):
        cl = CatalogLookup()
        result = cl.solve(ROT90_TASK, task_id="nonexistent_puzzle_id")
        assert result is None

    def test_loads_index_lazily(self):
        cl = CatalogLookup()
        assert cl._index is None
        cl._load_index()
        assert cl._index is not None

    def test_graceful_on_missing_catalog_path(self, tmp_path):
        cl = CatalogLookup(catalog_path=tmp_path / "does_not_exist")
        result = cl.solve(ROT90_TASK, task_id="anything")
        assert result is None


# ---------------------------------------------------------------------------
# NeuralInference
# ---------------------------------------------------------------------------


class TestNeuralInference:
    def test_returns_none_without_model(self):
        nn = NeuralInference(model=None)
        assert nn.solve(ROT90_TASK) is None

    def test_returns_predictions_with_fake_model(self):
        import numpy as np

        class _FakeLogits:
            def argmax(self, dim=-1):
                class _P:
                    def squeeze(self, dim):
                        return self

                    def tolist(self):
                        return [1, 2, 11, 3, 4, 11]  # two rows + separators

                return _P()

        class _FakeModel:
            def __call__(self, input_ids=None, return_confidences=False):
                return {"logits": _FakeLogits()}

        fake_torch = types.ModuleType("_fake_torch")
        fake_torch.tensor = lambda x: types.SimpleNamespace(to=lambda d: x)
        fake_torch.no_grad = lambda: nullcontext()

        class _FakeDevice:
            pass

        nn = NeuralInference(model=_FakeModel(), device=_FakeDevice())

        # Patch torch inside the module
        import src.arc_solver_engine as eng

        orig = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch  # type: ignore[assignment]
        try:
            result = nn.solve(ROT90_TASK)
        finally:
            if orig is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = orig

        assert result is not None
        assert result["method"] == "neural"
        assert isinstance(result["predictions"], list)


# ---------------------------------------------------------------------------
# MistralReasoning
# ---------------------------------------------------------------------------


class TestMistralReasoning:
    def test_returns_none_without_ollama(self):
        mr = MistralReasoning(run_ollama_chat_fn=None)
        assert mr.solve(ROT90_TASK) is None

    def test_parses_valid_json_response(self):
        expected_pred = [[3, 1], [4, 2]]
        response_json = (
            '{"predictions": [' + str(expected_pred) + '], "reasoning": "rotation"}'
        )

        def fake_chat(messages, **kwargs):
            return response_json, "mistral"

        mr = MistralReasoning(run_ollama_chat_fn=fake_chat)
        result = mr.solve(ROT90_TASK)
        assert result is not None
        assert result["method"] == "mistral"
        assert result["predictions"] == [expected_pred]
        assert result["reasoning"] == "rotation"

    def test_returns_none_on_invalid_json(self):
        def fake_chat(messages, **kwargs):
            return "not json at all", "mistral"

        mr = MistralReasoning(run_ollama_chat_fn=fake_chat)
        assert mr.solve(ROT90_TASK) is None

    def test_returns_none_on_empty_predictions(self):
        def fake_chat(messages, **kwargs):
            return '{"predictions": [], "reasoning": "none"}', "mistral"

        mr = MistralReasoning(run_ollama_chat_fn=fake_chat)
        assert mr.solve(ROT90_TASK) is None

    def test_format_arc_prompt_contains_training_data(self):
        prompt = _format_arc_prompt(ROT90_TASK)
        assert "Training examples:" in prompt
        assert "Test inputs" in prompt
        assert "predictions" in prompt


# ---------------------------------------------------------------------------
# ARCSolverEngine
# ---------------------------------------------------------------------------


class TestARCSolverEngine:
    def test_auto_uses_rule_learner_for_rotation(self):
        engine = ARCSolverEngine()
        result = engine.solve(ROT90_TASK, method="auto")
        assert result["success"] is True
        assert result["method"] == "rule_learner"
        assert result["rule"] == "rot90"
        assert result["confidence"] >= 0.9

    def test_explicit_rule_learner_method(self):
        engine = ARCSolverEngine()
        result = engine.solve(COLOR_MAP_TASK, method="rule_learner")
        assert result["method"] == "rule_learner"

    def test_identity_fallback_for_no_detectable_rule(self):
        engine = ARCSolverEngine()
        # A task where no simple rule exists
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[9, 8], [7, 6]]},
                {"input": [[5, 6], [7, 8]], "output": [[4, 3], [2, 1]]},
            ],
            "test": [{"input": [[0, 1], [2, 3]]}],
        }
        result = engine.solve(task, method="rule_learner")
        assert result["success"] is True
        # Either a rule was found (unlikely) or identity fallback
        assert "method" in result
        assert "predictions" in result

    def test_fallback_for_unknown_method(self):
        engine = ARCSolverEngine()
        result = engine.solve(ROT90_TASK, method="unknown_method")
        assert result["success"] is True
        assert result["method"] == "identity_fallback"
        assert result["predictions"] == [ROT90_TASK["test"][0]["input"]]

    def test_result_always_has_required_keys(self):
        engine = ARCSolverEngine()
        result = engine.solve(ROT90_TASK, method="auto")
        for key in ("success", "method", "rule", "confidence", "predictions",
                    "verified_on_training", "reasoning", "latency_ms"):
            assert key in result, f"Missing key: {key}"

    def test_latency_ms_is_positive(self):
        engine = ARCSolverEngine()
        result = engine.solve(ROT90_TASK)
        assert result["latency_ms"] >= 0

    def test_mistral_fallback_when_configured(self):
        def fake_chat(messages, **kwargs):
            return '{"predictions": [[[1,2],[3,4]]], "reasoning": "test"}', "mistral"

        engine = ARCSolverEngine(run_ollama_chat_fn=fake_chat)
        # An unsolvable task that hits all fallbacks
        task = {
            "train": [{"input": [[0]], "output": [[9]]}],  # color map should catch this
            "test": [{"input": [[5]]}],
        }
        result = engine.solve(task, method="mistral")
        assert result["success"] is True

    def test_predictions_is_list_of_grids(self):
        engine = ARCSolverEngine()
        result = engine.solve(ROT90_TASK)
        preds = result["predictions"]
        assert isinstance(preds, list)
        assert len(preds) == len(ROT90_TASK["test"])
        for grid in preds:
            assert isinstance(grid, list)
            for row in grid:
                assert isinstance(row, list)


# ---------------------------------------------------------------------------
# /solve-arc API endpoint via TestClient
# ---------------------------------------------------------------------------


class _DummyMonitor:
    def __init__(self):
        self.total_requests = 0

    def record_request(self, _latency_ms, error=False):
        self.total_requests += 1

    def get_stats(self):
        return {"total_requests": self.total_requests, "avg_latency_ms": 0,
                "error_count": 0, "memory_mb": 0}


class _FakeTensor:
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
    def load_state_dict(self, _sd, strict=False):
        pass

    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, input_ids=None, return_confidences=False):
        return {"logits": _FakeLogits()}


@pytest.fixture
def api_module(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    device_name = "cpu"
    fake_device = types.SimpleNamespace(type=device_name, __str__=lambda self: device_name)
    fake_torch.device = lambda name: fake_device
    fake_torch.tensor = lambda value: _FakeTensor()
    fake_torch.no_grad = lambda: nullcontext()
    fake_torch.load = lambda *args, **kwargs: {}

    fake_model_mod = types.ModuleType("model")
    fake_model_mod.OctoTetrahedralModel = _FakeModel

    fake_auth = types.ModuleType("auth")
    fake_auth.validate_api_key = lambda key: True

    fake_monitoring = types.ModuleType("monitoring")
    fake_monitoring.monitor = _DummyMonitor()

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "model", fake_model_mod)
    monkeypatch.setitem(sys.modules, "auth", fake_auth)
    monkeypatch.setitem(sys.modules, "monitoring", fake_monitoring)
    sys.modules.pop("api", None)

    return importlib.import_module("api")


@pytest.fixture
def client(api_module):
    from fastapi.testclient import TestClient

    api_module.app.dependency_overrides[api_module.verify_api_key] = lambda: "test-key"
    return TestClient(api_module.app)


def test_solve_arc_endpoint_rotation(client):
    response = client.post(
        "/solve-arc",
        json={"task": ROT90_TASK, "method": "rule_learner"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["method"] == "rule_learner"
    assert data["rule"] == "rot90"
    assert len(data["predictions"]) == 1


def test_solve_arc_endpoint_auto_mode(client):
    response = client.post(
        "/solve-arc",
        json={"task": ROT90_TASK, "method": "auto"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predictions" in data
    assert "confidence" in data


def test_solve_arc_endpoint_invalid_method(client):
    response = client.post(
        "/solve-arc",
        json={"task": ROT90_TASK, "method": "invalid_method"},
    )
    assert response.status_code == 400
    assert "Unknown method" in response.json()["detail"]


def test_solve_arc_endpoint_missing_train_key(client):
    response = client.post(
        "/solve-arc",
        json={"task": {"test": [{"input": [[1]]}]}, "method": "auto"},
    )
    assert response.status_code == 422
    assert "train" in response.json()["detail"]


def test_solve_arc_endpoint_missing_test_key(client):
    response = client.post(
        "/solve-arc",
        json={"task": {"train": [{"input": [[1]], "output": [[2]]}]}, "method": "auto"},
    )
    assert response.status_code == 422
    assert "test" in response.json()["detail"]


def test_health_endpoint_includes_solve_arc(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "solve-arc" in data["features"]
