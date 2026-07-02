"""Health check and diagnostics for the unified cognitive stack."""

from __future__ import annotations

import time
from typing import TypedDict

import production_config as cfg
from api_types import make_request
from inference_service import InferenceService
from monitoring import InferenceMonitor


class HealthStatus(TypedDict):
    healthy: bool
    model_loaded: bool
    coherence_baseline: float
    limb_symmetry_ok: bool
    self_test_passed: bool
    self_test_details: list[dict[str, object]]
    diagnostics: dict[str, object]


_SELF_TEST_INPUTS: list[tuple[list[float], str]] = [
    ([0.1] * 8, "reasoning"),
    ([0.5] * 8, "language"),
    ([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1], "spatial"),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "action"),
    ([0.3, 0.5, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1], "compound"),
]


def run_health_check(
    service: InferenceService | None = None,
    num_tests: int = 5,
) -> HealthStatus:
    """
    Verify the inference service is healthy and report diagnostics.

    Parameters
    ----------
    service:
        An existing InferenceService to test; one is created if not provided.
    num_tests:
        Number of self-test cases to run (1–5).
    """
    num_tests = max(1, min(num_tests, len(_SELF_TEST_INPUTS)))
    model_loaded = False
    self_test_passed = False
    self_test_details: list[dict[str, object]] = []
    coherence_baseline = 0.0
    limb_symmetry_ok = False
    diagnostics: dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # 1. Verify model loads
    # ------------------------------------------------------------------ #
    try:
        if service is None:
            monitor = InferenceMonitor()
            service = InferenceService(monitor=monitor)
        model_loaded = True
    except Exception as exc:  # noqa: BLE001
        diagnostics["load_error"] = str(exc)
        return {
            "healthy": False,
            "model_loaded": False,
            "coherence_baseline": 0.0,
            "limb_symmetry_ok": False,
            "self_test_passed": False,
            "self_test_details": [],
            "diagnostics": diagnostics,
        }

    # ------------------------------------------------------------------ #
    # 2. Run self-test suite
    # ------------------------------------------------------------------ #
    coherences: list[float] = []
    all_passed = True

    for i, (limb_states, task_signal) in enumerate(_SELF_TEST_INPUTS[:num_tests]):
        t0 = time.perf_counter()
        req = make_request(limb_states, task_signal=task_signal, request_id=f"healthcheck-{i}")
        resp = service.infer(req)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        passed = (
            resp["error"] is None
            and len(resp["limb_states"]) == cfg.MODEL_LIMB_COUNT
            and 0.0 <= resp["coherence"] <= 1.0
        )
        all_passed = all_passed and passed
        coherences.append(resp["coherence"])

        self_test_details.append(
            {
                "test_index": i,
                "task_signal": task_signal,
                "passed": passed,
                "coherence": resp["coherence"],
                "action_channel": resp["action_channel"],
                "latency_ms": round(elapsed_ms, 3),
                "error": resp["error"],
            }
        )

    self_test_passed = all_passed
    coherence_baseline = sum(coherences) / len(coherences) if coherences else 0.0

    # ------------------------------------------------------------------ #
    # 3. Limb activation symmetry check
    # ------------------------------------------------------------------ #
    # Use a uniform input; a symmetric model should produce nearly equal activations.
    sym_req = make_request([0.5] * cfg.MODEL_LIMB_COUNT, task_signal="symmetry-check")
    sym_resp = service.infer(sym_req)
    if sym_resp["error"] is None and sym_resp["limb_states"]:
        states = sym_resp["limb_states"]
        mean_val = sum(states) / len(states)
        max_deviation = max(abs(v - mean_val) for v in states)
        limb_symmetry_ok = max_deviation < 0.5  # allow reasonable asymmetry
        diagnostics["limb_symmetry_max_deviation"] = round(max_deviation, 4)
    else:
        diagnostics["limb_symmetry_error"] = sym_resp.get("error", "unknown")

    # ------------------------------------------------------------------ #
    # 4. Build diagnostics report
    # ------------------------------------------------------------------ #
    monitor_stats = service.monitor.stats()
    diagnostics.update(
        {
            "env": cfg.ENV,
            "model_version": cfg.MODEL_VERSION,
            "limb_count": cfg.MODEL_LIMB_COUNT,
            "coherence_threshold": cfg.COHERENCE_ALERT_THRESHOLD,
            "pool_size": cfg.EFFECTIVE_POOL_SIZE,
            "monitor_stats": monitor_stats,
        }
    )

    healthy = model_loaded and self_test_passed and coherence_baseline >= cfg.COHERENCE_ALERT_THRESHOLD

    return {
        "healthy": healthy,
        "model_loaded": model_loaded,
        "coherence_baseline": round(coherence_baseline, 4),
        "limb_symmetry_ok": limb_symmetry_ok,
        "self_test_passed": self_test_passed,
        "self_test_details": self_test_details,
        "diagnostics": diagnostics,
    }
