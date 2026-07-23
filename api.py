from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import OctoTetrahedralModel
import logging
import sys

from gpu_support import resolve_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OctoTetrahedral AGI Inference")

_device_info = resolve_device()
device = torch.device(_device_info["device"])

# Input validation constants
MAX_INPUT_TOKENS = 256
MIN_TOKEN_ID = 0
MAX_TOKEN_ID = 50000

# Load model once at startup
try:
    model = OctoTetrahedralModel()
    checkpoint = torch.load('checkpoints/arc/arc_final.pt', weights_only=True)

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


@app.post("/predict")
async def predict(request: InferenceRequest):
    """Run inference on input tokens.

    Returns predictions as a list of token IDs.

    Raises:
        400 Bad Request: if input_ids is empty or contains out-of-range/invalid token IDs.
        413 Payload Too Large: if input_ids exceeds MAX_INPUT_TOKENS (256).
        500 Internal Server Error: on unexpected inference failures.
    """
    # Validate: non-empty
    if len(request.input_ids) == 0:
        raise HTTPException(status_code=400, detail="input_ids must contain at least 1 token.")

    # Validate: not too large
    if len(request.input_ids) > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=413,
            detail=f"input_ids must contain no more than {MAX_INPUT_TOKENS} tokens.",
        )

    # Validate: each token ID must be an integer in the valid range
    for i, token_id in enumerate(request.input_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise HTTPException(
                status_code=400,
                detail=f"input_ids[{i}] must be an integer, got {type(token_id).__name__}.",
            )
        if token_id < MIN_TOKEN_ID or token_id > MAX_TOKEN_ID:
            raise HTTPException(
                status_code=400,
                detail=f"input_ids[{i}] must be between {MIN_TOKEN_ID} and {MAX_TOKEN_ID}.",
            )

    try:
        input_tensor = torch.tensor([request.input_ids]).to(device)

        with torch.no_grad():
            try:
                output = model(input_ids=input_tensor, return_confidences=False)
            except RuntimeError as mps_err:
                # Graceful MPS/Metal fallback: retry on CPU
                if device.type == "mps":
                    logger.warning(f"⚠️ Metal inference failed, retrying on CPU: {mps_err}")
                    cpu_tensor = input_tensor.to("cpu")
                    cpu_model = model.to("cpu")
                    output = cpu_model(input_ids=cpu_tensor, return_confidences=False)
                    model.to(device)
                else:
                    raise

        predictions = output['logits'].argmax(dim=-1).tolist()
        logger.info("✅ Prediction successful")

        return {
            "predictions": predictions,
            "device": str(device),
            "success": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check — returns model and device information."""
    return {
        "status": "healthy",
        "model": "OctoTetrahedralModel",
        "device": str(device),
        "device_type": _device_info["device"],
        "accelerator": _device_info.get("accelerator", "cpu"),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port)
