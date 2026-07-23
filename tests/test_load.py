from __future__ import annotations

import pytest

from load_testing import (
    LoadMetrics,
    generate_load_test_report,
    required_load_scenarios,
    summarize_latencies,
    validate_acceptance,
)


@pytest.mark.parametrize("concurrency", [10, 50, 100, 500])
def test_concurrent_request_scenarios_cover_required_levels(concurrency: int):
    scenarios = [scenario for scenario in required_load_scenarios() if scenario.kind == "concurrent"]

    assert any(scenario.concurrency == concurrency for scenario in scenarios)


@pytest.mark.parametrize("target_rps", [30, 50, 100])
def test_sustained_load_scenarios_cover_required_rates(target_rps: int):
    scenarios = [scenario for scenario in required_load_scenarios() if scenario.kind == "sustained"]

    assert any(
        scenario.target_rps == target_rps and scenario.duration_seconds == 60
        for scenario in scenarios
    )


def test_burst_and_long_running_scenarios_cover_required_targets():
    scenarios = required_load_scenarios()

    assert any(
        scenario.kind == "burst" and scenario.target_rps == 1000 and scenario.duration_seconds == 10
        for scenario in scenarios
    )
    assert any(
        scenario.kind == "long-running" and scenario.duration_seconds == 600
        for scenario in scenarios
    )


def test_metrics_capture_latency_distribution_throughput_errors_and_memory():
    metrics = LoadMetrics(
        latency_samples_ms=[63.36, 64.54, 65.29, 66.81, 70.09],
        throughput_rps=16.2,
        error_rate=0.0,
        memory_usage_mb=512.0,
        successful_requests=50,
        total_requests=50,
    )

    summary = summarize_latencies(metrics.latency_samples_ms)

    assert summary["avg_ms"] == pytest.approx(66.018, abs=0.001)
    assert summary["min_ms"] == 63.36
    assert summary["max_ms"] == 70.09
    assert metrics.throughput_rps == 16.2
    assert metrics.error_rate == 0.0
    assert metrics.memory_usage_mb == 512.0


def test_acceptance_validation_passes_for_healthy_baseline():
    metrics = LoadMetrics(
        latency_samples_ms=[63.36, 64.54, 65.29, 66.81, 70.09],
        throughput_rps=16.2,
        error_rate=0.0,
        memory_usage_mb=512.0,
        successful_requests=50,
        total_requests=50,
    )

    result = validate_acceptance(metrics)

    assert result == {
        "throughput_ok": True,
        "error_rate_ok": True,
        "latency_ok": True,
        "accepted": True,
    }


def test_acceptance_validation_detects_regressions():
    metrics = LoadMetrics(
        latency_samples_ms=[280.0, 300.0, 340.0, 360.0, 400.0],
        throughput_rps=9.5,
        error_rate=0.05,
        memory_usage_mb=1536.0,
        successful_requests=95,
        total_requests=100,
    )

    result = validate_acceptance(metrics)

    assert result["throughput_ok"] is False
    assert result["error_rate_ok"] is False
    assert result["latency_ok"] is False
    assert result["accepted"] is False


def test_load_test_report_generation_includes_scaling_recommendations():
    report = generate_load_test_report(
        {
            "concurrent-50": LoadMetrics(
                latency_samples_ms=[63.36, 64.54, 65.29, 66.81, 70.09],
                throughput_rps=16.2,
                error_rate=0.0,
                memory_usage_mb=512.0,
                successful_requests=50,
                total_requests=50,
            ),
            "burst-1000-rps": LoadMetrics(
                latency_samples_ms=[280.0, 300.0, 340.0, 360.0, 400.0],
                throughput_rps=9.5,
                error_rate=0.05,
                memory_usage_mb=1536.0,
                successful_requests=95,
                total_requests=100,
            ),
        }
    )

    assert "# Load Test Report" in report
    assert "## Recommendations for scaling" in report
    assert "GPU acceleration" in report
    assert "memory pooling" in report
