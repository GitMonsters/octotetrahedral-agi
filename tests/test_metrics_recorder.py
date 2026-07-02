"""Tests for monitoring/metrics_recorder.py."""

from __future__ import annotations

import csv
import os
import tempfile
import threading
import time
from unittest.mock import MagicMock

import pytest

from monitoring.config import MonitoringConfig
from monitoring.metrics_recorder import MetricsRecorder, _percentile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(coherence: float = 0.95, limb_count: int = 8) -> dict:
    return {
        "limb_states": [0.8] * 4 + [0.2] * (limb_count - 4),
        "shared_component": 0.5,
        "residuals": [0.05] * limb_count,
        "coherence": coherence,
        "coupling_strength": 0.7,
        "phase": 0.3,
        "bias": 0.1,
        "action_channel": 3,
    }


def _make_mock_model(coherence: float = 0.95, limb_count: int = 8):
    model = MagicMock()
    model.limb_count = limb_count
    model.forward.return_value = _make_result(coherence, limb_count)
    return model


def _run_inferences(recorder: MetricsRecorder, model, n: int = 10) -> None:
    limb_states = [0.5] * model.limb_count
    for _ in range(n):
        model.forward(limb_states, task_signal="test")


# ---------------------------------------------------------------------------
# _percentile helper
# ---------------------------------------------------------------------------

def test_percentile_empty_returns_zero():
    assert _percentile([], 50) == 0.0


def test_percentile_single_element():
    assert _percentile([42.0], 50) == 42.0


def test_percentile_p50():
    data = list(range(1, 101))  # 1..100
    assert _percentile(data, 50) == pytest.approx(50.5, abs=1.0)


def test_percentile_p99():
    data = list(range(1, 101))
    assert _percentile(data, 99) >= 98.0


# ---------------------------------------------------------------------------
# Basic recording
# ---------------------------------------------------------------------------

def test_recorder_starts_and_stops():
    recorder = MetricsRecorder()
    model = _make_mock_model()
    recorder.start_recording(model)
    assert recorder._model is model
    recorder.stop_recording()
    assert recorder._model is None


def test_recorder_wraps_forward():
    recorder = MetricsRecorder()
    model = _make_mock_model()
    original_forward = model.forward

    recorder.start_recording(model)
    # forward should now be wrapped
    assert model.forward is not original_forward
    recorder.stop_recording()
    # forward should be restored
    assert model.forward is original_forward


def test_recorder_captures_inferences():
    recorder = MetricsRecorder()
    model = _make_mock_model(coherence=0.97)
    recorder.start_recording(model)

    limb_states = [0.5] * 8
    model.forward(limb_states, task_signal="reasoning")
    model.forward(limb_states, task_signal="planning")

    recorder.stop_recording()
    inferences = recorder.get_all_inferences()
    assert len(inferences) == 2
    assert inferences[0]["coherence"] == pytest.approx(0.97)
    assert inferences[0]["task_signal"] == "reasoning"
    assert inferences[1]["task_signal"] == "planning"


def test_recorder_latency_is_positive():
    recorder = MetricsRecorder()
    model = _make_mock_model()
    recorder.start_recording(model)
    model.forward([0.5] * 8)
    recorder.stop_recording()

    inferences = recorder.get_all_inferences()
    assert inferences[0]["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Rolling stats
# ---------------------------------------------------------------------------

def test_get_rolling_stats_empty():
    recorder = MetricsRecorder()
    stats = recorder.get_rolling_stats()
    assert stats["total_inferences"] == 0
    assert stats["current"] == {}


def test_get_rolling_stats_populated():
    recorder = MetricsRecorder()
    model = _make_mock_model(coherence=0.90)
    recorder.start_recording(model)
    for _ in range(20):
        model.forward([0.5] * 8, task_signal="test")
    recorder.stop_recording()

    stats = recorder.get_rolling_stats()
    assert stats["total_inferences"] == 20
    assert stats["throughput_rps"] > 0
    assert "coherence" in stats["current"]
    assert stats["current"]["coherence"] == pytest.approx(0.90)
    all_s = stats["all"]
    assert "latency_p50" in all_s
    assert "latency_p99" in all_s
    assert "latency_p999" in all_s
    assert "coherence_mean" in all_s


def test_rolling_windows_populated_immediately():
    recorder = MetricsRecorder()
    model = _make_mock_model()
    recorder.start_recording(model)
    model.forward([0.5] * 8)
    recorder.stop_recording()

    stats = recorder.get_rolling_stats()
    # The single inference should appear in all windows
    for window in ("1min", "5min", "15min"):
        assert stats["windows"][window].get("count", 0) >= 1


# ---------------------------------------------------------------------------
# Circular buffer
# ---------------------------------------------------------------------------

def test_circular_buffer_overflow():
    config = MonitoringConfig(circular_buffer_size=5)
    recorder = MetricsRecorder(config=config)
    model = _make_mock_model()
    recorder.start_recording(model)
    for _ in range(10):
        model.forward([0.5] * 8)
    recorder.stop_recording()

    inferences = recorder.get_all_inferences()
    assert len(inferences) == 5  # capped at buffer size


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_buffer():
    recorder = MetricsRecorder()
    model = _make_mock_model()
    recorder.start_recording(model)
    model.forward([0.5] * 8)
    recorder.stop_recording()

    recorder.reset()
    assert recorder.get_all_inferences() == []
    assert recorder.get_rolling_stats()["total_inferences"] == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_thread_safe_concurrent_recording():
    """Multiple threads recording simultaneously should not corrupt state."""
    recorder = MetricsRecorder(config=MonitoringConfig(circular_buffer_size=2000))
    model = _make_mock_model()
    recorder.start_recording(model)

    errors: list[Exception] = []

    def worker():
        try:
            for _ in range(50):
                model.forward([0.5] * 8)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    recorder.stop_recording()
    assert errors == [], f"Thread errors: {errors}"
    inferences = recorder.get_all_inferences()
    assert len(inferences) <= 2000
    assert len(inferences) > 0


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def test_export_csv_creates_file():
    recorder = MetricsRecorder()
    model = _make_mock_model()
    recorder.start_recording(model)
    model.forward([0.5] * 8, task_signal="test")
    recorder.stop_recording()

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        path = tmp.name
    try:
        recorder.export_csv(path)
        assert os.path.exists(path)
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert "coherence" in rows[0]
        assert "latency_ms" in rows[0]
        assert "task_signal" in rows[0]
        assert rows[0]["task_signal"] == "test"
    finally:
        os.unlink(path)


def test_export_csv_empty_does_not_create_file():
    recorder = MetricsRecorder()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        path = tmp.name
    os.unlink(path)
    recorder.export_csv(path)
    assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

def test_context_manager_stops_on_exit():
    model = _make_mock_model()
    with MetricsRecorder() as recorder:
        recorder.start_recording(model)
        model.forward([0.5] * 8)
    # After __exit__, stop_recording should have been called
    assert recorder._model is None


# ---------------------------------------------------------------------------
# Double start guard
# ---------------------------------------------------------------------------

def test_double_start_raises():
    recorder = MetricsRecorder()
    model = _make_mock_model()
    recorder.start_recording(model)
    with pytest.raises(RuntimeError, match="Already recording"):
        recorder.start_recording(model)
    recorder.stop_recording()
