"""Integration tests for the production inference pipeline."""

from __future__ import annotations

import time

import pytest

from api_types import (
    deserialize_request,
    deserialize_response,
    make_batch_request,
    make_request,
    serialize_request,
    serialize_response,
)
from health_check import run_health_check
from inference_service import InferenceService
from monitoring import InferenceMonitor
from production_config import (
    BATCH_SIZE_MAX,
    BATCH_SIZE_MIN,
    COHERENCE_ALERT_THRESHOLD,
    MODEL_LIMB_COUNT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(pool_size: int = 2, timeout_ms: float = 500.0) -> InferenceService:
    """Build a fresh service with isolated monitor for each test."""
    monitor = InferenceMonitor()
    return InferenceService(
        pool_size=pool_size,
        limb_count=MODEL_LIMB_COUNT,
        timeout_ms=timeout_ms,
        max_retries=2,
        monitor=monitor,
    )


# ---------------------------------------------------------------------------
# Test 1: Load model and run single inference
# ---------------------------------------------------------------------------


def test_load_model_and_run_inference():
    service = _make_service()
    req = make_request([0.1] * MODEL_LIMB_COUNT, task_signal="reasoning")
    resp = service.infer(req)

    assert resp["error"] is None
    assert len(resp["limb_states"]) == MODEL_LIMB_COUNT
    assert 0.0 <= resp["coherence"] <= 1.0
    assert 0 <= resp["action_channel"] < MODEL_LIMB_COUNT
    assert resp["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Test 2: Coherence monitoring records and alerts
# ---------------------------------------------------------------------------


def test_coherence_monitoring_records_and_alerts():
    monitor = InferenceMonitor(coherence_threshold=COHERENCE_ALERT_THRESHOLD)
    service = InferenceService(pool_size=1, monitor=monitor)

    req = make_request([0.5] * MODEL_LIMB_COUNT, task_signal="language")
    resp = service.infer(req)

    assert resp["error"] is None
    stats = monitor.stats()
    assert stats["total_inferences"] >= 1

    # Inject a synthetic low-coherence observation to trigger an alert
    monitor.record(
        request_id="synthetic-low",
        coherence=0.50,
        action_channel=0,
        limb_states=[0.5] * MODEL_LIMB_COUNT,
        latency_ms=1.0,
    )
    alerts = monitor.recent_alerts()
    assert len(alerts) >= 1
    assert alerts[-1]["coherence"] == pytest.approx(0.50)
    assert alerts[-1]["threshold"] == COHERENCE_ALERT_THRESHOLD


# ---------------------------------------------------------------------------
# Test 3: Batch inference (N=1 to small N)
# ---------------------------------------------------------------------------


def test_batch_inference():
    service = _make_service()
    requests = [
        make_request([float(i) / 10.0] * MODEL_LIMB_COUNT, task_signal="batch-test")
        for i in range(1, 6)  # 5 requests
    ]
    batch = make_batch_request(requests)
    result = service.infer_batch(batch)

    assert result["batch_id"] == batch["batch_id"]
    assert len(result["responses"]) == 5
    assert result["total_latency_ms"] >= 0.0
    for resp in result["responses"]:
        assert resp["error"] is None
        assert len(resp["limb_states"]) == MODEL_LIMB_COUNT


# ---------------------------------------------------------------------------
# Test 4: Batch size boundaries
# ---------------------------------------------------------------------------


def test_batch_size_validation():
    service = _make_service()

    # Empty batch should raise
    with pytest.raises(ValueError, match="batch size"):
        service.infer_batch(make_batch_request([]))

    # Batch exceeding max should raise
    oversized = [make_request([0.1] * MODEL_LIMB_COUNT) for _ in range(BATCH_SIZE_MAX + 1)]
    with pytest.raises(ValueError, match="batch size"):
        service.infer_batch(make_batch_request(oversized))

    # Single request batch (minimum) should succeed
    single = make_batch_request([make_request([0.1] * MODEL_LIMB_COUNT)])
    result = service.infer_batch(single)
    assert len(result["responses"]) == BATCH_SIZE_MIN


# ---------------------------------------------------------------------------
# Test 5: Error handling / fallback behaviour
# ---------------------------------------------------------------------------


def test_error_handling_and_fallback():
    service = _make_service()

    # Provide wrong limb count to trigger a ValueError inside forward()
    bad_req = make_request(
        [0.1] * (MODEL_LIMB_COUNT + 5),  # too many limbs
        task_signal="error-test",
        request_id="err-001",
    )
    resp = service.infer(bad_req)
    # Service should return an error response, not raise
    assert resp["request_id"] == "err-001"
    assert resp["error"] is not None

    # After a successful inference, the fallback should carry last-known-good state
    good_req = make_request([0.3] * MODEL_LIMB_COUNT, task_signal="ok", request_id="ok-001")
    good_resp = service.infer(good_req)
    assert good_resp["error"] is None

    # Trigger error again — fallback should return last good limb_states
    bad_req2 = make_request([0.1] * (MODEL_LIMB_COUNT + 2), request_id="err-002")
    fallback_resp = service.infer(bad_req2)
    assert fallback_resp["error"] is not None
    assert "fallback" in fallback_resp["error"]
    assert fallback_resp["limb_states"] == good_resp["limb_states"]


# ---------------------------------------------------------------------------
# Test 6: Timing constraints (p99 < threshold)
# ---------------------------------------------------------------------------


def test_timing_constraints():
    service = _make_service(pool_size=4)
    n = 50
    latencies: list[float] = []

    for i in range(n):
        req = make_request([float(i % 9) / 8.0] * MODEL_LIMB_COUNT, task_signal="timing")
        t0 = time.perf_counter()
        resp = service.infer(req)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assert resp["error"] is None

    latencies.sort()
    p99_ms = latencies[int(0.99 * n) - 1]
    # Allow generous headroom in CI; the hard requirement is <20 ms on production hardware.
    assert p99_ms < 500.0, f"p99 latency {p99_ms:.1f} ms too high in test environment"


# ---------------------------------------------------------------------------
# Test 7: Health check passes
# ---------------------------------------------------------------------------


def test_health_check_passes():
    service = _make_service()
    status = run_health_check(service=service, num_tests=3)

    assert status["model_loaded"] is True
    assert status["self_test_passed"] is True
    assert status["coherence_baseline"] >= COHERENCE_ALERT_THRESHOLD
    assert len(status["self_test_details"]) == 3


# ---------------------------------------------------------------------------
# Test 8: API types serialise/deserialise round-trip
# ---------------------------------------------------------------------------


def test_api_types_round_trip():
    original_req = make_request(
        [float(i + 1) / 10.0 for i in range(MODEL_LIMB_COUNT)],
        task_signal="round-trip",
    )
    assert deserialize_request(serialize_request(original_req)) == original_req

    service = _make_service()
    resp = service.infer(original_req)
    raw = serialize_response(resp)
    decoded = deserialize_response(raw)
    assert decoded["request_id"] == resp["request_id"]
    assert decoded["coherence"] == pytest.approx(resp["coherence"])
    assert decoded["action_channel"] == resp["action_channel"]
