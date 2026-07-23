from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import OctoTetrahedralModel
import logging
import sys
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OctoTetrahedral AGI Inference")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    input_ids: list

@app.post("/predict")
async def predict(request: InferenceRequest):
    """Run inference on input tokens"""
    try:
        input_ids = torch.tensor([request.input_ids]).to(device)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, return_confidences=False)
        
        predictions = output['logits'].argmax(dim=-1).tolist()
        
        logger.info(f"✅ Prediction successful")
        
        return {
            "predictions": predictions,
            "device": str(device),
            "success": True
        }
    except Exception as e:
        logger.error(f"❌ Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "model": "OctoTetrahedralModel", "device": str(device)}

if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port)
