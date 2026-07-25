from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import time
import torch
from model import OctoTetrahedralModel
from auth import validate_api_key
from monitoring import monitor
import logging
import sys
from typing import Optional, List, Dict

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


# ============================================================================
# Request/Response Models
# ============================================================================

class InferenceRequest(BaseModel):
    """Token-based inference request"""
    input_ids: list


class PromptRequest(BaseModel):
    """Natural language prompt request"""
    prompt: str
    mode: str = "answer"  # answer, code, creative, technical
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9


class ChatMessage(BaseModel):
    """Chat message"""
    role: str  # user, assistant, system
    content: str


class ChatRequest(BaseModel):
    """Chat request"""
    messages: List[ChatMessage]
    system_prompt: Optional[str] = None
    max_length: int = 200


class CommandRequest(BaseModel):
    """Command request"""
    command: str  # analyze, summarize, translate, expand, simplify
    input_text: str
    options: Optional[Dict] = None


class AskRequest(BaseModel):
    """Simple ask request"""
    question: str


# ============================================================================
# Token-Based Inference (Original)
# ============================================================================

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


# ============================================================================
# Natural Language Endpoints
# ============================================================================

@app.post("/prompt")
async def handle_prompt(
    request: PromptRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Natural language prompt endpoint
    
    Modes: answer, code, creative, technical
    """
    t0 = time.time()
    try:
        # Simulate prompt processing
        response_text = f"Response to '{request.prompt}': This is a generated response based on mode '{request.mode}'"
        
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=False)
        
        return {
            "success": True,
            "prompt": request.prompt,
            "response": response_text,
            "mode": request.mode,
            "device": str(device),
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Prompt error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def handle_chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Conversational chat endpoint
    """
    t0 = time.time()
    try:
        # Build conversation context
        conversation = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in request.messages
        ])
        
        # Simulate chat response
        response_text = f"Chat response based on conversation: {conversation[:100]}..."
        
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=False)
        
        return {
            "success": True,
            "response": response_text,
            "device": str(device),
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(
    request: AskRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Simple question answering endpoint
    """
    t0 = time.time()
    try:
        # Simulate Q&A
        answer = f"Answer to '{request.question}': This is a generated answer based on your question."
        
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=False)
        
        return {
            "success": True,
            "question": request.question,
            "answer": answer,
            "device": str(device),
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/command")
async def handle_command(
    request: CommandRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Execute natural language commands
    
    Commands: summarize, analyze, translate, expand, simplify
    """
    t0 = time.time()
    try:
        command = request.command.lower()
        
        # Map commands to responses
        if command == "summarize":
            response = f"Summary: {request.input_text[:100]}..."
        elif command == "analyze":
            response = f"Analysis of: {request.input_text[:100]}..."
        elif command == "translate":
            target_lang = request.options.get("target_language", "Spanish") if request.options else "Spanish"
            response = f"Translation to {target_lang}: {request.input_text}"
        elif command == "expand":
            response = f"Expanded: {request.input_text}...[expanded content]"
        elif command == "simplify":
            response = f"Simplified: {request.input_text}...[simplified content]"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown command: {command}")
        
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=False)
        
        return {
            "success": True,
            "command": command,
            "input": request.input_text,
            "output": response,
            "device": str(device),
            "latency_ms": round(latency_ms, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Command error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health & Monitoring
# ============================================================================

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
        "device_info": device_info,
        "features": ["predict", "prompt", "chat", "ask", "command"]
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
