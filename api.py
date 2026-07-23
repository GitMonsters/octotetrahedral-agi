from __future__ import annotations

import logging
import os
import sys
import time
from collections import defaultdict, deque
from typing import Dict, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from auth import RateLimiter, hash_key, validate_api_key, verify_token
from gpu_metal_support import device_info, select_device
from model import OctoTetrahedralModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OctoTetrahedral AGI Inference")

# ---------------------------------------------------------------------------
# Device selection (Metal > CUDA > CPU)
# ---------------------------------------------------------------------------

device = select_device()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

try:
    model = OctoTetrahedralModel()
    checkpoint = torch.load('checkpoints/arc/arc_final.pt', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    logger.info(f"✅ Model loaded on {device}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

# ---------------------------------------------------------------------------
# Per-key rate limiter and usage tracker
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter()
_key_usage: Dict[str, int] = defaultdict(int)

# ---------------------------------------------------------------------------
# Request-level performance metrics
# ---------------------------------------------------------------------------

_metrics: Dict[str, object] = {
    "requests_total": 0,
    "requests_predict": 0,
    "requests_failed": 0,
    "total_latency_ms": 0.0,
    "start_time": time.time(),
}
_latency_window: deque = deque(maxlen=1000)

# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------

_AUTH_ENABLED = os.getenv("OCTO_AUTH_ENABLED", "true").lower() != "false"


def _extract_bearer(request: Request) -> Optional[str]:
    """Extract the ****** or API key from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return None


async def require_auth(request: Request) -> str:
    """FastAPI dependency: validate ****** or API key and enforce rate limit.

    Returns the key hash on success; raises HTTP 401/429 on failure.
    Authentication can be disabled by setting ``OCTO_AUTH_ENABLED=false``.
    """
    if not _AUTH_ENABLED:
        return "anonymous"

    credential = _extract_bearer(request)
    if not credential:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Use: Authorization: ******",
        )

    # Accept raw API keys or JWT tokens
    if "." in credential:
        # JWT-style token
        valid, key_hash = verify_token(credential)
        if not valid or key_hash is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    else:
        # Raw API key
        valid, key_hash = validate_api_key(credential)
        if not valid or key_hash is None:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # Rate limiting
    if not _rate_limiter.is_allowed(key_hash):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    _key_usage[key_hash] += 1
    return key_hash


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_memory_stats() -> Optional[dict]:
    """Return memory stats dict from psutil, or None if unavailable."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_mb": round(mem.total / 1e6, 1),
            "available_mb": round(mem.available / 1e6, 1),
            "used_mb": round(mem.used / 1e6, 1),
            "used_pct": mem.percent,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class InferenceRequest(BaseModel):
    input_ids: list


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/predict")
async def predict(request: InferenceRequest, _key: str = Depends(require_auth)):
    """Run inference on input tokens."""
    _metrics["requests_total"] = int(_metrics["requests_total"]) + 1
    _metrics["requests_predict"] = int(_metrics["requests_predict"]) + 1
    t0 = time.perf_counter()
    try:
        input_ids = torch.tensor([request.input_ids]).to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, return_confidences=False)

        predictions = output['logits'].argmax(dim=-1).tolist()
        latency_ms = (time.perf_counter() - t0) * 1000
        _metrics["total_latency_ms"] = float(_metrics["total_latency_ms"]) + latency_ms
        _latency_window.append(latency_ms)

        logger.info("✅ Prediction successful (%.1f ms)", latency_ms)

        return {
            "predictions": predictions,
            "device": str(device),
            "latency_ms": round(latency_ms, 3),
            "success": True,
        }
    except Exception as e:
        _metrics["requests_failed"] = int(_metrics["requests_failed"]) + 1
        logger.error("❌ Inference error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check with device information."""
    return {
        "status": "healthy",
        "model": "OctoTetrahedralModel",
        "device": str(device),
        "device_info": device_info(),
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    reqs = int(_metrics["requests_total"])
    failed = int(_metrics["requests_failed"])
    total_lat = float(_metrics["total_latency_ms"])
    predict_reqs = int(_metrics["requests_predict"])
    avg_lat = total_lat / predict_reqs if predict_reqs > 0 else 0.0

    uptime = time.time() - float(_metrics["start_time"])

    lines = [
        "# HELP octoagi_requests_total Total API requests",
        "# TYPE octoagi_requests_total counter",
        f"octoagi_requests_total {reqs}",
        "",
        "# HELP octoagi_requests_failed_total Total failed requests",
        "# TYPE octoagi_requests_failed_total counter",
        f"octoagi_requests_failed_total {failed}",
        "",
        "# HELP octoagi_avg_latency_ms Average predict latency (ms)",
        "# TYPE octoagi_avg_latency_ms gauge",
        f"octoagi_avg_latency_ms {avg_lat:.3f}",
        "",
        "# HELP octoagi_uptime_seconds Server uptime in seconds",
        "# TYPE octoagi_uptime_seconds gauge",
        f"octoagi_uptime_seconds {uptime:.1f}",
    ]

    mem = _get_memory_stats()
    if mem is not None:
        lines += [
            "",
            "# HELP octoagi_memory_used_mb Memory used (MB)",
            "# TYPE octoagi_memory_used_mb gauge",
            f"octoagi_memory_used_mb {mem['used_mb']:.1f}",
        ]

    lines.append("")
    return "\n".join(lines)


@app.get("/performance")
async def performance():
    """Performance summary endpoint."""
    reqs = int(_metrics["requests_total"])
    predict_reqs = int(_metrics["requests_predict"])
    failed = int(_metrics["requests_failed"])
    total_lat = float(_metrics["total_latency_ms"])
    uptime = time.time() - float(_metrics["start_time"])

    avg_lat = total_lat / predict_reqs if predict_reqs > 0 else 0.0
    throughput = predict_reqs / uptime if uptime > 0 else 0.0
    error_rate = failed / reqs if reqs > 0 else 0.0

    lats = list(_latency_window)
    p50 = p99 = 0.0
    if lats:
        sorted_lats = sorted(lats)
        n = len(sorted_lats)
        p50 = sorted_lats[int(n * 0.50)]
        p99 = sorted_lats[min(int(n * 0.99), n - 1)]

    return {
        "requests_total": reqs,
        "requests_predict": predict_reqs,
        "requests_failed": failed,
        "error_rate": round(error_rate, 4),
        "avg_latency_ms": round(avg_lat, 3),
        "p50_latency_ms": round(p50, 3),
        "p99_latency_ms": round(p99, 3),
        "throughput_rps": round(throughput, 3),
        "uptime_seconds": round(uptime, 1),
        "device": str(device),
    }


@app.get("/stats")
async def stats():
    """Real-time statistics endpoint."""
    uptime = time.time() - float(_metrics["start_time"])
    reqs = int(_metrics["requests_total"])
    mem = _get_memory_stats()

    return {
        "uptime_seconds": round(uptime, 1),
        "requests_total": reqs,
        "active_keys": len(_key_usage),
        "device": str(device),
        "metal_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "memory": mem or {},
    }


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port)
