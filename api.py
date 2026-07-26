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
from typing import Any, Optional, List, Dict, Tuple

try:
    from ollama import Client as OllamaClient, ResponseError as OllamaResponseError
except ImportError:  # pragma: no cover - guarded in runtime checks
    OllamaClient = None
    OllamaResponseError = Exception

from src.arc_solver_engine import ARCSolverEngine

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

# ARC solver engine – initialised after the model loads
arc_solver = ARCSolverEngine(model=model, device=device)


DEFAULT_OLLAMA_MODEL = "mistral:latest"
DEFAULT_OLLAMA_TEMPERATURE = 0.7
DEFAULT_OLLAMA_TOP_P = 0.9

MODE_SYSTEM_PROMPTS = {
    "answer": "Provide a clear and accurate answer.",
    "code": "Provide production-quality code with brief explanation.",
    "creative": "Respond creatively while staying relevant.",
    "technical": "Provide precise technical detail and actionable guidance.",
}

# Supported natural-language command operations for /command endpoint.
SUPPORTED_COMMANDS = {"summarize", "analyze", "translate", "expand", "simplify"}
VALID_CHAT_ROLES = {"system", "user", "assistant"}


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


def _ollama_model_candidates() -> List[str]:
    primary = (os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()
    if not primary:
        primary = DEFAULT_OLLAMA_MODEL
    fallback_models = os.getenv("OLLAMA_FALLBACK_MODELS", "")

    models = [primary]
    if fallback_models:
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
    messages: List[Dict[str, str]],
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_length: Optional[int] = None,
) -> Tuple[str, str]:
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
            message_obj = response.get("message") or {}
            content = message_obj.get("content", "").strip()
            if not content:
                has_fallbacks = idx < len(models) - 1
                if has_fallbacks:
                    logger.warning(
                        f"⚠️ Ollama model '{model_name}' returned empty content, trying fallback model"
                    )
                    continue
                raise OllamaServiceError(
                    f"Ollama returned empty message content for model '{model_name}'. "
                    "Check the selected model configuration and Ollama logs."
                )
            return content, model_name
        except OllamaResponseError as exc:
            last_error = exc
            if getattr(exc, "status_code", None) == 404:
                if idx < len(models) - 1:
                    logger.warning(f"⚠️ Ollama model '{model_name}' unavailable, trying fallback model")
                continue
            raise OllamaServiceError(f"Ollama request failed for model '{model_name}': {exc}") from exc
        except Exception as exc:
            last_error = exc
            if _is_connection_error(exc):
                raise OllamaUnavailableError(
                    "Unable to connect to Ollama. Start it with `ollama serve` and ensure the model is installed."
                ) from exc
            raise OllamaServiceError(f"Ollama request failed for model '{model_name}': {exc}") from exc

    raise OllamaServiceError(f"Ollama request failed for all configured models: {last_error}")


def _ollama_health() -> Dict[str, Any]:
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


class ARCSolveRequest(BaseModel):
    """ARC-AGI puzzle solve request"""
    task: Dict  # ARC task with 'train' and 'test' keys
    method: str = "auto"  # auto | rule_learner | catalog | neural | mistral
    task_id: Optional[str] = None  # puzzle ID for catalog lookup


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
    except HTTPException:
        raise
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
            if msg.role not in VALID_CHAT_ROLES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid message role: {msg.role}. Supported roles: system, user, assistant.",
                )
            messages.append({"role": msg.role, "content": msg.content})

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
    except HTTPException:
        raise
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
    except HTTPException:
        raise
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
# ARC-AGI Puzzle Solver
# ============================================================================

_VALID_SOLVE_METHODS = {"auto", "rule_learner", "catalog", "neural", "mistral"}


@app.post("/solve-arc")
async def solve_arc(
    request: ARCSolveRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Solve an ARC-AGI puzzle using the strongest applicable method.

    The solver tries four strategies in priority order (when method='auto'):
      1. **Catalog Lookup** – exact match against 514 pre-solved puzzles
      2. **Rule Learner** – geometric/color/scale rules from training pairs
      3. **Neural Inference** – OctoTetrahedralModel token-level predictions
      4. **Mistral Reasoning** – Ollama LLM for novel/complex puzzles

    Returns structured predictions with confidence scores and reasoning.
    """
    t0 = time.time()
    try:
        method = request.method.lower()
        if method not in _VALID_SOLVE_METHODS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown method '{method}'. Valid options: {sorted(_VALID_SOLVE_METHODS)}",
            )

        task = request.task
        if "train" not in task or "test" not in task:
            raise HTTPException(
                status_code=422,
                detail="Task must contain both 'train' and 'test' keys.",
            )

        # Wire Ollama into the solver on each request so it picks up any
        # runtime config changes without restarting the server.
        arc_solver._mistral._run_ollama_chat = _run_ollama_chat  # type: ignore[attr-defined]

        result = arc_solver.solve(task, method=method, task_id=request.task_id)

        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=False)
        logger.info(
            f"✅ /solve-arc method={result['method']} "
            f"confidence={result['confidence']:.2f} ({latency_ms:.1f}ms)"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        monitor.record_request(latency_ms, error=True)
        logger.error(f"❌ /solve-arc error: {e}", exc_info=True)
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
        "features": ["predict", "prompt", "chat", "ask", "command", "solve-arc"],
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
