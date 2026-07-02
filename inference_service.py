"""Production inference service for the unified cognitive stack."""

from __future__ import annotations

import json
import logging
import queue
import time
from typing import Any

import production_config as cfg
from api_types import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    InferenceRequest,
    InferenceResponse,
)
from monitoring import InferenceMonitor
from unified.forward_model import UnifiedForwardModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_error_response(request_id: str, error: str, latency_ms: float) -> InferenceResponse:
    return {
        "request_id": request_id,
        "limb_states": [0.0] * cfg.MODEL_LIMB_COUNT,
        "shared_component": 0.0,
        "residuals": [0.0] * cfg.MODEL_LIMB_COUNT,
        "coherence": 0.0,
        "coupling_strength": 0.0,
        "phase": 0.0,
        "bias": 0.0,
        "action_channel": 0,
        "latency_ms": latency_ms,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Model pool
# ---------------------------------------------------------------------------


class _ModelPool:
    """Fixed-size pool of UnifiedForwardModel instances for concurrent use."""

    def __init__(self, size: int, limb_count: int) -> None:
        self._pool: queue.Queue[UnifiedForwardModel] = queue.Queue(maxsize=size)
        for _ in range(size):
            self._pool.put(UnifiedForwardModel(limb_count=limb_count))

    def acquire(self, timeout: float | None = None) -> UnifiedForwardModel:
        try:
            return self._pool.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("model pool exhausted") from exc

    def release(self, model: UnifiedForwardModel) -> None:
        try:
            self._pool.put_nowait(model)
        except queue.Full:
            pass  # pool is full; discard surplus instance

    def size(self) -> int:
        return self._pool.qsize()


# ---------------------------------------------------------------------------
# Production inference service
# ---------------------------------------------------------------------------


class InferenceService:
    """
    Production-ready inference service wrapping the unified forward model.

    Features
    --------
    - Connection pooling (configurable pool size)
    - Per-request timeout with error fallback
    - Automatic retry on transient failures
    - Coherence monitoring and alerting
    - JSON-structured logging with request IDs
    """

    def __init__(
        self,
        pool_size: int = cfg.EFFECTIVE_POOL_SIZE,
        limb_count: int = cfg.MODEL_LIMB_COUNT,
        timeout_ms: float = cfg.EFFECTIVE_TIMEOUT_MS,
        max_retries: int = cfg.EFFECTIVE_MAX_RETRIES,
        monitor: InferenceMonitor | None = None,
    ) -> None:
        self._pool = _ModelPool(size=pool_size, limb_count=limb_count)
        self._timeout_ms = timeout_ms
        self._max_retries = max_retries
        self._monitor = monitor or InferenceMonitor()
        self._last_good_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Single inference
    # ------------------------------------------------------------------

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run a single forward pass; returns an error response on failure."""
        request_id = request["request_id"]
        limb_states = request["limb_states"]
        task_signal = request.get("task_signal")

        logger.debug(
            json.dumps({"event": "inference_start", "request_id": request_id})
        )

        if not (1 <= len(limb_states) <= cfg.LIMB_STATES_MAX_LENGTH):
            return _make_error_response(
                request_id,
                f"limb_states length {len(limb_states)} out of expected range",
                0.0,
            )

        last_error: str = ""
        for attempt in range(1, self._max_retries + 1):
            t0 = time.perf_counter()
            try:
                model = self._pool.acquire(timeout=self._timeout_ms / 1000.0)
                try:
                    result = model.forward(limb_states, task_signal=task_signal)
                finally:
                    self._pool.release(model)

                latency_ms = (time.perf_counter() - t0) * 1000.0

                response: InferenceResponse = {
                    "request_id": request_id,
                    "limb_states": result["limb_states"],
                    "shared_component": result["shared_component"],
                    "residuals": result["residuals"],
                    "coherence": result["coherence"],
                    "coupling_strength": result["coupling_strength"],
                    "phase": result["phase"],
                    "bias": result["bias"],
                    "action_channel": result["action_channel"],
                    "latency_ms": latency_ms,
                    "error": None,
                }

                self._monitor.record(
                    request_id=request_id,
                    coherence=result["coherence"],
                    action_channel=result["action_channel"],
                    limb_states=result["limb_states"],
                    latency_ms=latency_ms,
                )

                self._last_good_result = dict(response)  # type: ignore[arg-type]

                logger.debug(
                    json.dumps(
                        {
                            "event": "inference_complete",
                            "request_id": request_id,
                            "coherence": result["coherence"],
                            "latency_ms": round(latency_ms, 3),
                            "attempt": attempt,
                        }
                    )
                )
                return response

            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                logger.warning(
                    json.dumps(
                        {
                            "event": "inference_error",
                            "request_id": request_id,
                            "error": last_error,
                            "attempt": attempt,
                        }
                    )
                )

        # All retries exhausted — return fallback
        return self._fallback_response(request_id, last_error)

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def infer_batch(self, batch: BatchInferenceRequest) -> BatchInferenceResponse:
        """Run inference on a batch of 1–100 requests."""
        requests = batch["requests"]
        batch_id = batch["batch_id"]

        if not (cfg.BATCH_SIZE_MIN <= len(requests) <= cfg.BATCH_SIZE_MAX):
            raise ValueError(
                f"batch size {len(requests)} outside allowed range "
                f"[{cfg.BATCH_SIZE_MIN}, {cfg.BATCH_SIZE_MAX}]"
            )

        t_batch_start = time.perf_counter()
        responses: list[InferenceResponse] = [self.infer(req) for req in requests]
        total_latency_ms = (time.perf_counter() - t_batch_start) * 1000.0

        logger.debug(
            json.dumps(
                {
                    "event": "batch_complete",
                    "batch_id": batch_id,
                    "count": len(responses),
                    "total_latency_ms": round(total_latency_ms, 3),
                }
            )
        )

        return {
            "batch_id": batch_id,
            "responses": responses,
            "total_latency_ms": total_latency_ms,
        }

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_response(self, request_id: str, error: str) -> InferenceResponse:
        """Return last-known-good state or a zeroed error response."""
        if self._last_good_result is not None:
            fallback = dict(self._last_good_result)
            fallback["request_id"] = request_id
            fallback["error"] = f"fallback (original error: {error})"
            return fallback  # type: ignore[return-value]
        return _make_error_response(request_id, error, 0.0)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def monitor(self) -> InferenceMonitor:
        return self._monitor
