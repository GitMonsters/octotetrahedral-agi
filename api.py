from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import time
import torch
from model import OctoTetrahedralModel
from auth import validate_api_key
from monitoring import monitor
import logging
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OctoTetrahedral AGI Inference")

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

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


async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    try:
        scheme, token = authorization.split(" ")
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    if not validate_api_key(token):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return token


class InferenceRequest(BaseModel):
    input_ids: list


@app.post("/predict")
async def predict(request: InferenceRequest, api_key: str = Depends(verify_api_key)):
    """Run inference on input tokens (requires valid API key)"""
    t0 = time.time()
    try:
        input_ids = torch.tensor([request.input_ids]).to(device)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, return_confidences=False)
        
        predictions = output['logits'].argmax(dim=-1).tolist()
        
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=False)
        
        logger.info(f"✅ Prediction successful ({latency_ms:.1f}ms)")
        
        return {
            "predictions": predictions,
            "device": str(device),
            "latency_ms": round(latency_ms, 2),
            "success": True
        }
    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check with device information"""
    device_type = device.type
    device_info = {"type": device_type}
    if device_type == "mps":
        device_info["backend"] = "Apple Metal"
    elif device_type == "cuda":
        device_info["backend"] = "NVIDIA CUDA"
    else:
        device_info["backend"] = "CPU"
    
    return {
        "status": "healthy",
        "model": "OctoTetrahedralModel",
        "device": str(device),
        "device_info": device_info
    }


@app.get("/stats")
async def stats():
    """Performance statistics"""
    return monitor.get_stats()


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    stats = monitor.get_stats()
    return {
        "api_requests_total": stats.get("total_requests", 0),
        "api_latency_ms_avg": stats.get("avg_latency_ms", 0),
        "api_errors_total": stats.get("error_count", 0),
        "process_memory_mb": stats.get("memory_mb", 0)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port)
