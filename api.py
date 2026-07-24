from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import time
import torch
from model import OctoTetrahedralModel
from auth import verify_api_key, get_key_stats
from monitoring import PerformanceMonitor
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OctoTetrahedral AGI Inference")

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load model once at startup
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

perf_monitor = PerformanceMonitor()


class InferenceRequest(BaseModel):
    input_ids: list


@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(request: InferenceRequest):
    """Run inference on input tokens (requires API key)."""
    t0 = time.monotonic()
    try:
        input_ids = torch.tensor([request.input_ids]).to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, return_confidences=False)

        predictions = output['logits'].argmax(dim=-1).tolist()

        latency_ms = (time.monotonic() - t0) * 1000
        perf_monitor.record(latency_ms, error=False)
        logger.info(f"✅ Prediction successful ({latency_ms:.1f} ms)")

        return {
            "predictions": predictions,
            "device": str(device),
            "latency_ms": round(latency_ms, 2),
            "success": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        perf_monitor.record(latency_ms, error=True)
        logger.error(f"❌ Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check including device info."""
    device_type = device.type
    device_info: dict = {"type": device_type}
    if device_type == "mps":
        device_info["backend"] = "Apple Metal"
    elif device_type == "cuda":
        device_info["backend"] = "NVIDIA CUDA"
        try:
            device_info["name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    else:
        device_info["backend"] = "CPU"
    return {
        "status": "healthy",
        "model": "OctoTetrahedralModel",
        "device": str(device),
        "device_info": device_info,
    }


@app.get("/stats")
async def stats():
    """Performance statistics for the inference API."""
    return {
        "performance": perf_monitor.get_stats(),
        "api_keys": get_key_stats(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics."""
    return PlainTextResponse(
        perf_monitor.get_prometheus_metrics(),
        media_type="text/plain; version=0.0.4",
    )


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port)
