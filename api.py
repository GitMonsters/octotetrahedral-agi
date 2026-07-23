from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from model import OctoTetrahedralModel
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OctoTetrahedral AGI Inference")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_INPUT_TOKENS = 256
MIN_TOKEN_ID = 0
MAX_TOKEN_ID = 50000

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

class InferenceRequest(BaseModel):
    # Keep raw values here so invalid payloads can return the API's explicit
    # 400/413 contract instead of FastAPI/Pydantic's default 422 response.
    input_ids: list[Any] = Field(..., description="List of token IDs to run inference on")


def validate_input_ids(input_ids: list[Any]) -> list[int]:
    """Validate request token IDs before they reach the model.

    Args:
        input_ids: Raw token IDs from the request payload.

    Returns:
        A validated list of integer token IDs.

    Raises:
        HTTPException: 400 when the payload is empty, contains non-integers,
            or includes token IDs outside the 0-50000 range.
        HTTPException: 413 when the payload contains more than 256 tokens.
    """
    if not input_ids:
        raise HTTPException(
            status_code=400,
            detail="input_ids must contain at least 1 token.",
        )

    if len(input_ids) > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=413,
            detail=f"input_ids must contain no more than {MAX_INPUT_TOKENS} tokens.",
        )

    validated_input_ids: list[int] = []
    for index, token_id in enumerate(input_ids):
        # bool is a subclass of int in Python, so reject it explicitly.
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise HTTPException(
                status_code=400,
                detail=f"input_ids[{index}] must be an integer.",
            )
        if token_id < MIN_TOKEN_ID or token_id > MAX_TOKEN_ID:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"input_ids[{index}] must be between "
                    f"{MIN_TOKEN_ID} and {MAX_TOKEN_ID}."
                ),
            )
        validated_input_ids.append(token_id)

    return validated_input_ids

@app.post("/predict")
async def predict(request: InferenceRequest):
    """Run inference on input tokens"""
    try:
        validated_input_ids = validate_input_ids(request.input_ids)
        # Wrap the validated token list in an outer list to create the
        # single-request batch dimension expected by the model: [1, seq_len].
        input_ids = torch.tensor([validated_input_ids], dtype=torch.long, device=device)
        
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
        raise HTTPException(status_code=500, detail="Inference failed.")

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "model": "OctoTetrahedralModel", "device": str(device)}

if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port)
