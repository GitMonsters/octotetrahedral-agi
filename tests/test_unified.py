from benchmarks.unified_perf import run_benchmark
from unified.feedback_loop import UnifiedFeedbackLoop
from unified.forward_model import LegacyForwardAdapter, UnifiedForwardModel


def test_unified_forward_model_outputs_expected_shape_and_metrics():
    model = UnifiedForwardModel()
    result = model.forward([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], task_signal="reasoning")

    assert len(result["limb_states"]) == 8
    assert len(result["residuals"]) == 8
    assert 0.0 <= result["coherence"] <= 1.0
    assert 0.0 <= result["coupling_strength"] <= 1.0
    assert 0 <= result["action_channel"] < 8


def test_feedback_loop_couples_all_limbs_toward_shared_state():
    loop = UnifiedFeedbackLoop(limb_count=8)
    integrated = loop.integrate([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], coupling_strength=0.5)

    assert min(integrated) > 0.0
    assert max(integrated) < 1.0


def test_legacy_adapter_preserves_backward_compatible_api():
    adapter = LegacyForwardAdapter()
    output = adapter.run([0.2] * 8, task_type="language")

    assert isinstance(output, list)
    assert len(output) == 8


def test_unified_benchmark_reports_positive_metrics():
    metrics = run_benchmark(samples=10)

    assert metrics["unified_latency_ms"] > 0
    assert metrics["legacy_adapter_latency_ms"] > 0
    assert metrics["efficiency_gain"] > 0


def test_action_channel_selects_dominant_limb():
    model = UnifiedForwardModel()
    result = model.forward([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], task_signal="action")

    assert isinstance(result["action_channel"], int)
    expected = max(range(len(result["limb_states"])), key=result["limb_states"].__getitem__)
    assert result["action_channel"] == expected
