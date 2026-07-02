"""Request/response protocol types for the unified cognitive stack inference API."""

from __future__ import annotations

import json
import uuid
from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class InferenceRequest(TypedDict):
    request_id: str
    limb_states: list[float]
    task_signal: str | None


class BatchInferenceRequest(TypedDict):
    batch_id: str
    requests: list[InferenceRequest]


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class InferenceResponse(TypedDict):
    request_id: str
    limb_states: list[float]
    shared_component: float
    residuals: list[float]
    coherence: float
    coupling_strength: float
    phase: float
    bias: float
    action_channel: int
    latency_ms: float
    error: str | None


class BatchInferenceResponse(TypedDict):
    batch_id: str
    responses: list[InferenceResponse]
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request(
    limb_states: list[float],
    task_signal: str | None = None,
    request_id: str | None = None,
) -> InferenceRequest:
    """Build a single inference request with an auto-generated ID if none supplied."""
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "limb_states": limb_states,
        "task_signal": task_signal,
    }


def make_batch_request(
    requests: list[InferenceRequest],
    batch_id: str | None = None,
) -> BatchInferenceRequest:
    """Wrap a list of individual requests into a batch."""
    return {
        "batch_id": batch_id or str(uuid.uuid4()),
        "requests": requests,
    }


def serialize_request(request: InferenceRequest) -> str:
    return json.dumps(request)


def deserialize_request(raw: str) -> InferenceRequest:
    data: Any = json.loads(raw)
    return {
        "request_id": str(data["request_id"]),
        "limb_states": [float(v) for v in data["limb_states"]],
        "task_signal": data.get("task_signal"),
    }


def serialize_response(response: InferenceResponse) -> str:
    return json.dumps(response)


def deserialize_response(raw: str) -> InferenceResponse:
    data: Any = json.loads(raw)
    return {
        "request_id": str(data["request_id"]),
        "limb_states": [float(v) for v in data["limb_states"]],
        "shared_component": float(data["shared_component"]),
        "residuals": [float(v) for v in data["residuals"]],
        "coherence": float(data["coherence"]),
        "coupling_strength": float(data["coupling_strength"]),
        "phase": float(data["phase"]),
        "bias": float(data["bias"]),
        "action_channel": int(data["action_channel"]),
        "latency_ms": float(data["latency_ms"]),
        "error": data.get("error"),
    }
