from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from config import get_config
from gpu_support import build_benchmark_comparison, detect_device
from model import OctoTetrahedralModel
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OctoTetrahedral AGI Inference")

MAX_INPUT_TOKENS = 256
MAX_TOKEN_ID = 50_000

config = get_config()
device_info = detect_device(config.device)
device = torch.device(device_info.resolved)
benchmark_comparison = build_benchmark_comparison()

# Load model once at startup
try:
    model = OctoTetrahedralModel()
    checkpoint = torch.load('checkpoints/arc/arc_final.pt', map_location='cpu', weights_only=True)

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


class InferenceRequest(BaseModel):
    input_ids: list


def _validate_input_ids(input_ids: list) -> list[int]:
    if not isinstance(input_ids, list):
        raise HTTPException(status_code=400, detail="input_ids must be a JSON array of integers.")

    if not input_ids:
        raise HTTPException(status_code=400, detail="input_ids must contain at least 1 token.")

    if len(input_ids) > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=413,
            detail=f"input_ids must contain no more than {MAX_INPUT_TOKENS} tokens.",
        )

    validated: list[int] = []
    for index, token_id in enumerate(input_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise HTTPException(status_code=400, detail=f"input_ids[{index}] must be an integer.")
        if token_id < 0 or token_id > MAX_TOKEN_ID:
            raise HTTPException(
                status_code=400,
                detail=f"input_ids[{index}] must be between 0 and {MAX_TOKEN_ID}.",
            )
        validated.append(token_id)

    return validated


@app.post("/predict")
async def predict(request: InferenceRequest):
    """Run inference on input tokens"""
    try:
        validated_ids = _validate_input_ids(request.input_ids)
        input_ids = torch.tensor([validated_ids], device=device)

        with torch.no_grad():
            output = model(input_ids=input_ids, return_confidences=False)

        predictions = output['logits'].argmax(dim=-1).tolist()

        logger.info("✅ Prediction successful")

        return {
            "predictions": predictions,
            "device": str(device),
            "success": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model": "OctoTetrahedralModel",
        "device": str(device),
        "requested_device": device_info.requested,
        "accelerator": device_info.accelerator,
        "fallback_used": device_info.fallback_used,
        "expected_speedup_factor": round(benchmark_comparison["speedup_factor"], 2),
    }

if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port)
