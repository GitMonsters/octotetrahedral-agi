from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import time
import torch
from model import OctoTetrahedralModel
from auth import validate_api_key
from monitoring import monitor
import logging
import sys
import os
from typing import Any, Optional, List, Dict

try:
    from ollama import Client as OllamaClient, ResponseError as OllamaResponseError
except ImportError:  # pragma: no cover - guarded in runtime checks
    OllamaClient = None
    OllamaResponseError = Exception

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


DEFAULT_OLLAMA_MODEL = "mistral"
DEFAULT_OLLAMA_TEMPERATURE = 0.7
DEFAULT_OLLAMA_TOP_P = 0.9

MODE_SYSTEM_PROMPTS = {
    "answer": "Provide a clear and accurate answer.",
    "code": "Provide production-quality code with brief explanation.",
    "creative": "Respond creatively while staying relevant.",
    "technical": "Provide precise technical detail and actionable guidance.",
}

SUPPORTED_COMMANDS = {"summarize", "analyze", "translate", "expand", "simplify"}


class OllamaServiceError(Exception):
    """Raised when Ollama request fails."""


class OllamaUnavailableError(OllamaServiceError):
    """Raised when Ollama server is unavailable."""


def _parse_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _ollama_model_candidates() -> list[str]:
    primary = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
    fallback_models = os.getenv("OLLAMA_FALLBACK_MODELS", "")

    models = [primary]
    for model_name in fallback_models.split(","):
        candidate = model_name.strip()
        if candidate and candidate not in models:
            models.append(candidate)
    return models


def _ollama_client() -> Any:
    if OllamaClient is None:
        raise OllamaUnavailableError(
            "Ollama dependency is not installed. Run: pip install -r requirements.txt"
        )

    host = os.getenv("OLLAMA_HOST")
    if host:
        return OllamaClient(host=host)
    return OllamaClient()


def _ollama_options(
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "temperature": temperature
        if temperature is not None
        else _parse_float(os.getenv("OLLAMA_TEMPERATURE"), DEFAULT_OLLAMA_TEMPERATURE),
        "top_p": top_p if top_p is not None else _parse_float(os.getenv("OLLAMA_TOP_P"), DEFAULT_OLLAMA_TOP_P),
    }
    if max_length is not None and max_length > 0:
        options["num_predict"] = max_length
    return options


def _is_connection_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("connection", "refused", "timed out", "failed to connect"))


def _run_ollama_chat(
    messages: list[dict[str, str]],
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_length: Optional[int] = None,
) -> tuple[str, str]:
    client = _ollama_client()
    options = _ollama_options(temperature=temperature, top_p=top_p, max_length=max_length)
    models = _ollama_model_candidates()

    last_error: Optional[Exception] = None
    for idx, model_name in enumerate(models):
        try:
            response = client.chat(
                model=model_name,
                messages=messages,
                options=options,
            )
            content = (response.get("message") or {}).get("content", "").strip()
            if not content:
                raise OllamaServiceError(f"Ollama returned an empty response for model '{model_name}'")
            return content, model_name
        except OllamaResponseError as exc:
            last_error = exc
            if getattr(exc, "status_code", None) == 404 and idx < len(models) - 1:
                logger.warning(f"⚠️ Ollama model '{model_name}' unavailable, trying fallback model")
                continue
            raise OllamaServiceError(f"Ollama request failed for model '{model_name}': {exc}") from exc
        except Exception as exc:
            last_error = exc
            if _is_connection_error(exc):
                raise OllamaUnavailableError(
                    "Unable to connect to Ollama. Start it with `ollama serve` and ensure the model is installed."
                ) from exc
            if "not found" in str(exc).lower() and idx < len(models) - 1:
                logger.warning(f"⚠️ Ollama model '{model_name}' unavailable, trying fallback model")
                continue
            raise OllamaServiceError(f"Ollama request failed for model '{model_name}': {exc}") from exc

    raise OllamaServiceError(f"Ollama request failed for all configured models: {last_error}")


def _ollama_health() -> dict[str, Any]:
    models = _ollama_model_candidates()
    try:
        client = _ollama_client()
        client.list()
        return {
            "status": "healthy",
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "model": models[0],
            "fallback_models": models[1:],
        }
    except OllamaUnavailableError as exc:
        return {"status": "unavailable", "error": str(exc), "model": models[0], "fallback_models": models[1:]}
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "model": models[0], "fallback_models": models[1:]}


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
        system_prompt = MODE_SYSTEM_PROMPTS.get(request.mode.lower(), MODE_SYSTEM_PROMPTS["answer"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt},
        ]
        response_text, used_model = _run_ollama_chat(
            messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_length=request.max_length,
        )
        
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=False)
        
        return {
            "success": True,
            "prompt": request.prompt,
            "response": response_text,
            "mode": request.mode,
            "model": used_model,
            "device": str(device),
            "latency_ms": round(latency_ms, 2)
        }
    except OllamaUnavailableError as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Ollama unavailable for /prompt: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except OllamaServiceError as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Ollama error for /prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))
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
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for msg in request.messages:
            role = msg.role if msg.role in {"system", "user", "assistant"} else "user"
            messages.append({"role": role, "content": msg.content})

        response_text, used_model = _run_ollama_chat(
            messages,
            max_length=request.max_length,
        )
        
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=False)
        
        return {
            "success": True,
            "response": response_text,
            "model": used_model,
            "device": str(device),
            "latency_ms": round(latency_ms, 2)
        }
    except OllamaUnavailableError as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Ollama unavailable for /chat: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except OllamaServiceError as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Ollama error for /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
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
        messages = [{"role": "user", "content": request.question}]
        answer, used_model = _run_ollama_chat(messages)
        
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=False)
        
        return {
            "success": True,
            "question": request.question,
            "answer": answer,
            "model": used_model,
            "device": str(device),
            "latency_ms": round(latency_ms, 2)
        }
    except OllamaUnavailableError as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Ollama unavailable for /ask: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except OllamaServiceError as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Ollama error for /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))
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

        if command not in SUPPORTED_COMMANDS:
            raise HTTPException(status_code=400, detail=f"Unknown command: {command}")

        options_text = f"\nOptions: {request.options}" if request.options else ""
        prompt = (
            f"Command: {command}\n"
            f"Input:\n{request.input_text}{options_text}\n\n"
            "Execute this command and return only the processed result."
        )
        messages = [{"role": "user", "content": prompt}]
        response, used_model = _run_ollama_chat(messages)
        
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=False)
        
        return {
            "success": True,
            "command": command,
            "input": request.input_text,
            "output": response,
            "model": used_model,
            "device": str(device),
            "latency_ms": round(latency_ms, 2)
        }
    except HTTPException:
        raise
    except OllamaUnavailableError as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Ollama unavailable for /command: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except OllamaServiceError as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ Ollama error for /command: {e}")
        raise HTTPException(status_code=500, detail=str(e))
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
        "features": ["predict", "prompt", "chat", "ask", "command"],
        "ollama": _ollama_health(),
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
