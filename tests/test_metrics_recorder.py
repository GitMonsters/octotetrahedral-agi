from monitoring.metrics_recorder import MetricsRecorder
from unified.forward_model import UnifiedForwardModel


def test_recorder_snapshot_is_empty_before_any_samples():
    recorder = MetricsRecorder(window_size=10)
    snapshot = recorder.snapshot()

    assert snapshot["sample_count"] == 0
    assert snapshot["total_requests"] == 0
    assert snapshot["coherence_latest"] is None


def test_record_tracks_coherence_latency_and_limb_activity():
    model = UnifiedForwardModel()
    recorder = MetricsRecorder(window_size=10)

    result = model.forward([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], task_signal="reasoning")
    sample = recorder.record(result, latency_ms=12.5, task_signal="reasoning")

    assert sample.coherence == result["coherence"]
    assert sample.latency_ms == 12.5
    assert sample.task_signal == "reasoning"
    assert sample.limb_count == 8
    assert 0 <= sample.limbs_active <= 8

    snapshot = recorder.snapshot()
    assert snapshot["sample_count"] == 1
    assert snapshot["total_requests"] == 1
    assert snapshot["coherence_latest"] == result["coherence"]
    assert snapshot["coherence_previous"] is None


def test_rolling_window_evicts_oldest_samples():
    model = UnifiedForwardModel()
    recorder = MetricsRecorder(window_size=3)

    for i in range(5):
        result = model.forward([0.1 * i] * 8, task_signal="reasoning")
        recorder.record(result, latency_ms=float(i))

    snapshot = recorder.snapshot()
    assert snapshot["sample_count"] == 3
    assert snapshot["total_requests"] == 5
    # Only the last 3 latencies (2.0, 3.0, 4.0) remain in the window.
    assert snapshot["latency_latest_ms"] == 4.0


def test_task_signal_is_normalized_and_defaults():
    model = UnifiedForwardModel()
    recorder = MetricsRecorder()

    result = model.forward([0.1] * 8, task_signal=None)
    recorder.record(result, latency_ms=1.0, task_signal=None)
    result2 = model.forward([0.1] * 8, task_signal="Reasoning")
    recorder.record(result2, latency_ms=1.0, task_signal="  Reasoning  ")

    history = recorder.history()
    assert history[0]["task_signal"] == "default"
    assert history[1]["task_signal"] == "reasoning"


def test_instrument_transparently_records_forward_calls():
    model = UnifiedForwardModel()
    recorder = MetricsRecorder(window_size=5)
    instrumented = recorder.instrument(model)

    result = instrumented.forward([0.2] * 8, task_signal="language")

    assert len(result["limb_states"]) == 8
    snapshot = recorder.snapshot()
    assert snapshot["sample_count"] == 1
    assert snapshot["task_signal"] == "language"
    assert snapshot["latency_latest_ms"] >= 0


def test_instrument_delegates_unknown_attributes_to_wrapped_model():
    model = UnifiedForwardModel(limb_count=8)
    recorder = MetricsRecorder()
    instrumented = recorder.instrument(model)

    assert instrumented.limb_count == 8


def test_snapshot_reports_percentile_latencies():
    model = UnifiedForwardModel()
    recorder = MetricsRecorder(window_size=100)

    for latency in range(1, 101):
        result = model.forward([0.1] * 8, task_signal="reasoning")
        recorder.record(result, latency_ms=float(latency))

    snapshot = recorder.snapshot()
    assert snapshot["latency_p50_ms"] <= snapshot["latency_p95_ms"] <= snapshot["latency_p99_ms"]


def test_reset_clears_samples_and_counters():
    model = UnifiedForwardModel()
    recorder = MetricsRecorder()
    result = model.forward([0.1] * 8, task_signal="reasoning")
    recorder.record(result, latency_ms=5.0)

    recorder.reset()

    snapshot = recorder.snapshot()
    assert snapshot["sample_count"] == 0
    assert snapshot["total_requests"] == 0


def test_window_size_must_be_positive():
    try:
        MetricsRecorder(window_size=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
