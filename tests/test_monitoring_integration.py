"""Integration tests combining MetricsRecorder, CLIMonitor, and web dashboard."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from monitoring.config import MonitoringConfig
from monitoring.integration import MonitoringSystem
from monitoring.metrics_recorder import MetricsRecorder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(coherence: float = 0.93, limb_count: int = 8):
    model = MagicMock()
    model.limb_count = limb_count
    model.forward.return_value = {
        "limb_states": [0.8] * 4 + [0.2] * (limb_count - 4),
        "shared_component": 0.5,
        "residuals": [0.05] * limb_count,
        "coherence": coherence,
        "coupling_strength": 0.7,
        "phase": 0.3,
        "bias": 0.1,
        "action_channel": 2,
    }
    return model


# ---------------------------------------------------------------------------
# Recorder → stats flow
# ---------------------------------------------------------------------------

def test_recorder_captures_all_inferences():
    model = _make_model()
    config = MonitoringConfig(circular_buffer_size=200)
    recorder = MetricsRecorder(config=config)
    recorder.start_recording(model)

    for _ in range(100):
        model.forward([0.5] * 8, task_signal="integration_test")

    recorder.stop_recording()
    stats = recorder.get_rolling_stats()

    assert stats["total_inferences"] == 100
    assert stats["throughput_rps"] > 0
    assert stats["current"]["coherence"] == pytest.approx(0.93)
    assert stats["all"]["coherence_mean"] == pytest.approx(0.93)


def test_recorder_rolling_window_consistency():
    model = _make_model()
    recorder = MetricsRecorder()
    recorder.start_recording(model)

    for _ in range(50):
        model.forward([0.5] * 8)

    recorder.stop_recording()
    stats = recorder.get_rolling_stats()

    # All-window count should equal 1-min count (we just ran them)
    w1 = stats["windows"]["1min"]
    assert w1.get("count", 0) == 50


# ---------------------------------------------------------------------------
# MonitoringSystem context manager
# ---------------------------------------------------------------------------

def test_monitoring_system_context_manager():
    model = _make_model()
    config = MonitoringConfig()

    with MonitoringSystem(model, config=config) as monitor:
        for _ in range(10):
            model.forward([0.5] * 8, task_signal="ctx_test")
        stats = monitor.get_stats()

    assert stats["total_inferences"] == 10
    assert stats["current"]["action_channel"] == 2
    # recorder should be detached after exit
    assert monitor.recorder._model is None


def test_monitoring_system_manual_start_stop():
    model = _make_model()
    monitor = MonitoringSystem(model)
    monitor.start()

    for _ in range(5):
        model.forward([0.5] * 8)

    monitor.stop()
    stats = monitor.get_stats()
    assert stats["total_inferences"] == 5


# ---------------------------------------------------------------------------
# Concurrent access
# ---------------------------------------------------------------------------

def test_concurrent_inferences_thread_safe():
    model = _make_model()
    config = MonitoringConfig(circular_buffer_size=5000)
    recorder = MetricsRecorder(config=config)
    recorder.start_recording(model)

    errors: list[Exception] = []

    def worker(n: int) -> None:
        try:
            for _ in range(n):
                model.forward([0.5] * 8, task_signal="thread_test")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(25,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    recorder.stop_recording()
    assert errors == [], f"Errors during concurrent recording: {errors}"
    stats = recorder.get_rolling_stats()
    assert stats["total_inferences"] <= 5000


# ---------------------------------------------------------------------------
# MonitoringSystem with CLI monitor (no web, no terminal draw)
# ---------------------------------------------------------------------------

def test_monitoring_system_with_cli_no_crash(capsys):
    model = _make_model()
    config = MonitoringConfig(cli_update_frequency_sec=0.05)

    with MonitoringSystem(model, config=config, enable_cli=True) as monitor:
        for _ in range(5):
            model.forward([0.5] * 8)
        time.sleep(0.1)  # let the CLI thread run at least once
        stats = monitor.get_stats()

    assert stats["total_inferences"] == 5


# ---------------------------------------------------------------------------
# Web dashboard + recorder integration (FastAPI TestClient)
# ---------------------------------------------------------------------------

def test_web_dashboard_reflects_recorder_state():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from monitoring.web_dashboard import create_app

    model = _make_model(coherence=0.88)
    config = MonitoringConfig()
    recorder = MetricsRecorder(config=config)
    app = create_app(recorder=recorder, config=config)

    recorder.start_recording(model)
    for _ in range(20):
        model.forward([0.5] * 8, task_signal="web_test")
    recorder.stop_recording()

    client = TestClient(app)
    resp = client.get("/api/metrics/current")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_inferences"] == 20
    assert data["current"]["coherence"] == pytest.approx(0.88)


def test_prometheus_export_reflects_inference_count():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from monitoring.web_dashboard import create_app

    model = _make_model()
    config = MonitoringConfig()
    recorder = MetricsRecorder(config=config)
    app = create_app(recorder=recorder, config=config)

    recorder.start_recording(model)
    for _ in range(7):
        model.forward([0.5] * 8)
    recorder.stop_recording()

    client = TestClient(app)
    resp = client.get("/metrics")
    assert "unified_inference_count 7" in resp.text


# ---------------------------------------------------------------------------
# Graceful shutdown (recorder detaches cleanly)
# ---------------------------------------------------------------------------

def test_graceful_shutdown_restores_forward():
    model = _make_model()
    original_forward = model.forward

    monitor = MonitoringSystem(model)
    monitor.start()
    model.forward([0.5] * 8)
    monitor.stop()

    # After stop, forward should be restored
    assert model.forward is original_forward
