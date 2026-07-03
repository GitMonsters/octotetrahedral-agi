from monitoring.metrics_recorder import MetricsRecorder
from monitoring.web_dashboard import create_app
from unified.forward_model import UnifiedForwardModel


def test_index_page_renders_limb_count():
    app = create_app(limb_count=8)
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"unified-stack (8 limbs)" in response.data


def test_metrics_endpoint_returns_empty_snapshot_and_history_initially():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/metrics")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["snapshot"]["sample_count"] == 0
    assert payload["history"] == []


def test_metrics_endpoint_reflects_recorded_samples():
    recorder = MetricsRecorder(window_size=5)
    model = UnifiedForwardModel()
    result = model.forward([0.3] * 8, task_signal="spatial")
    recorder.record(result, latency_ms=9.0, task_signal="spatial")

    app = create_app(recorder=recorder)
    client = app.test_client()

    response = client.get("/api/metrics")
    payload = response.get_json()

    assert payload["snapshot"]["sample_count"] == 1
    assert payload["snapshot"]["task_signal"] == "spatial"
    assert len(payload["history"]) == 1
    assert payload["history"][0]["latency_ms"] == 9.0
