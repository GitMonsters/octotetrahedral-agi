from monitoring.cli_monitor import generate_demo_limb_states, render_snapshot, run
from monitoring.metrics_recorder import MetricsRecorder
from unified.forward_model import UnifiedForwardModel


def test_render_snapshot_shows_waiting_message_when_empty():
    recorder = MetricsRecorder()
    output = render_snapshot(recorder.snapshot(), limb_count=8)

    assert "Model: unified-stack (8 limbs)" in output
    assert "Waiting for inference samples" in output


def test_render_snapshot_formats_recorded_metrics():
    model = UnifiedForwardModel()
    recorder = MetricsRecorder(window_size=10)
    result = model.forward([0.1] * 8, task_signal="language")
    recorder.record(result, latency_ms=14.2, task_signal="language")

    output = render_snapshot(recorder.snapshot(), limb_count=8)

    assert "Coherence:" in output
    assert "Latency:      14.2ms" in output
    assert "Limbs Active:" in output
    assert "Action Ch:" in output
    assert "(language)" in output
    assert "Requests:     1 total" in output


def test_render_snapshot_trend_arrow_reflects_coherence_change():
    recorder = MetricsRecorder(window_size=10)
    model = UnifiedForwardModel()

    increasing = [
        {"coherence": 0.5, "coupling_strength": 0.4, "residuals": [0.1] * 8, "limb_states": [0.5] * 8, "action_channel": 0},
        {"coherence": 0.9, "coupling_strength": 0.4, "residuals": [0.1] * 8, "limb_states": [0.5] * 8, "action_channel": 0},
    ]
    for result in increasing:
        recorder.record(result, latency_ms=1.0, task_signal="reasoning")

    output = render_snapshot(recorder.snapshot(), limb_count=8)
    assert "↑" in output


def test_generate_demo_limb_states_returns_bounded_values():
    import random

    rng = random.Random(42)
    states = generate_demo_limb_states(rng, limb_count=8)

    assert len(states) == 8
    assert all(0.0 <= value <= 1.0 for value in states)


def test_run_stops_after_duration_and_records_samples(capsys):
    run(interval=0.01, window_size=10, task_signals=["reasoning"], duration=0.05, seed=1, clear_screen=False)

    captured = capsys.readouterr()
    assert "Model: unified-stack (8 limbs)" in captured.out
