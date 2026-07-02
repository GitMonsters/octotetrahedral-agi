"""Tests for monitoring/web_dashboard.py."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from monitoring.config import MonitoringConfig
from monitoring.metrics_recorder import MetricsRecorder
from monitoring.web_dashboard import build_prometheus_output, create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recorder_with_data(n: int = 5) -> MetricsRecorder:
    from unittest.mock import MagicMock

    config = MonitoringConfig()
    recorder = MetricsRecorder(config=config)
    model = MagicMock()
    model.limb_count = 8
    model.forward.return_value = {
        "limb_states": [0.8] * 4 + [0.2] * 4,
        "shared_component": 0.5,
        "residuals": [0.05] * 8,
        "coherence": 0.92,
        "coupling_strength": 0.7,
        "phase": 0.3,
        "bias": 0.1,
        "action_channel": 3,
    }
    recorder.start_recording(model)
    for _ in range(n):
        model.forward([0.5] * 8, task_signal="test")
    recorder.stop_recording()
    return recorder


def _make_client(recorder: MetricsRecorder | None = None) -> TestClient:
    rec = recorder or _make_recorder_with_data()
    app = create_app(recorder=rec)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    client = _make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

def test_dashboard_returns_html():
    client = _make_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Unified Cognitive Stack" in resp.text


# ---------------------------------------------------------------------------
# /api/metrics/current
# ---------------------------------------------------------------------------

def test_metrics_current_returns_stats():
    client = _make_client()
    resp = client.get("/api/metrics/current")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_inferences" in data
    assert data["total_inferences"] == 5


def test_metrics_current_includes_current_fields():
    client = _make_client()
    resp = client.get("/api/metrics/current")
    data = resp.json()
    cur = data.get("current", {})
    assert "coherence" in cur
    assert "latency_ms" in cur


# ---------------------------------------------------------------------------
# /api/metrics/history
# ---------------------------------------------------------------------------

def test_metrics_history_default_minutes():
    client = _make_client()
    resp = client.get("/api/metrics/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "minutes" in data
    assert data["minutes"] == 5
    assert "data" in data


def test_metrics_history_custom_minutes():
    client = _make_client()
    resp = client.get("/api/metrics/history?minutes=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["minutes"] == 1


def test_metrics_history_count_matches_data():
    client = _make_client()
    resp = client.get("/api/metrics/history")
    data = resp.json()
    assert data["count"] == len(data["data"])


# ---------------------------------------------------------------------------
# /api/metrics/export
# ---------------------------------------------------------------------------

def test_metrics_export_json():
    client = _make_client()
    resp = client.get("/api/metrics/export?format=json")
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]


def test_metrics_export_prometheus_format():
    client = _make_client()
    resp = client.get("/api/metrics/export?format=prometheus")
    assert resp.status_code == 200
    body = resp.text
    assert "unified_coherence" in body
    assert "unified_latency_ms" in body
    assert "unified_inference_count" in body


# ---------------------------------------------------------------------------
# /metrics (Prometheus endpoint)
# ---------------------------------------------------------------------------

def test_prometheus_endpoint():
    client = _make_client()
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "# HELP unified_coherence" in body
    assert "# TYPE unified_coherence gauge" in body
    assert "# EOF" in body


def test_prometheus_endpoint_content_type():
    client = _make_client()
    resp = client.get("/metrics")
    assert "text/plain" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------

def test_cors_headers_present():
    client = _make_client()
    resp = client.options(
        "/api/metrics/current",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    # CORS middleware should add access-control headers
    headers = resp.headers
    assert "access-control-allow-origin" in headers


# ---------------------------------------------------------------------------
# build_prometheus_output helper
# ---------------------------------------------------------------------------

def test_build_prometheus_output_format():
    stats = {
        "total_inferences": 100,
        "throughput_rps": 10.5,
        "current": {
            "coherence": 0.95,
            "limbs_active": 5,
        },
        "all": {
            "coherence_mean": 0.93,
            "latency_p50": 12.3,
            "latency_p99": 24.6,
            "latency_p999": 35.0,
        },
    }
    output = build_prometheus_output(stats)
    assert 'unified_coherence{quantile="current"} 0.950000' in output
    assert 'unified_latency_ms{quantile="p50"} 12.3000' in output
    assert "unified_inference_count 100" in output
    assert "unified_limbs_active 5" in output
    assert "unified_throughput_rps" in output


def test_build_prometheus_output_empty_stats():
    output = build_prometheus_output({})
    assert "unified_coherence" in output
    assert "unified_inference_count 0" in output
