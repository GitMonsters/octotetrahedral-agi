"""
OctoTetrahedral AGI — FastAPI Inference Server

Serves MoE models via REST API with streaming support.
Compatible with NVIDIA NIM deployment patterns.

Usage:
    # Local dev
    python serve.py --config 7b --device cuda:0

    # With specific checkpoint
    python serve.py --config 7b --checkpoint checkpoints/best.pt

    # Docker (see deploy/Dockerfile.gpu)
    docker run --gpus all -p 8000:8000 octotetrahedral:7b-moe
"""

import argparse
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------
model = None
config = None
device = None
model_info = {}


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_tokens: int = Field(128, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stream: bool = Field(False, description="Stream tokens as SSE")


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float


class ModelInfoResponse(BaseModel):
    model_name: str
    total_params: str
    active_params: str
    config_name: str
    device: str
    dtype: str
    moe_enabled: bool
    num_experts: int
    top_k: int
    max_seq_len: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config_name: str, checkpoint_path: Optional[str], target_device: str):
    """Load the OctoTetrahedral model."""
    global model, config, device, model_info

    from train_distributed import load_config
    from model import OctoTetrahedralModel

    config = load_config(config_name)
    config.device = target_device
    device = torch.device(target_device)

    logger.info(f"Loading {config_name} config (d={config.model.hidden_dim}, "
                f"L={config.model.num_layers}, MoE={config.moe.enabled})")

    # Use bf16 for GPU inference
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = OctoTetrahedralModel(config, use_geometric_physics=False)

    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"], strict=False)

    model = model.to(device=device, dtype=dtype)
    model.eval()

    total = model.get_num_params()
    active = model.get_active_params()

    model_info = {
        "model_name": f"OctoTetrahedral-{config_name}-MoE",
        "total_params": f"{total / 1e9:.1f}B",
        "active_params": f"{active / 1e9:.1f}B",
        "config_name": config_name,
        "device": str(device),
        "dtype": str(dtype),
        "moe_enabled": config.moe.enabled,
        "num_experts": config.moe.num_experts,
        "top_k": config.moe.top_k,
        "max_seq_len": config.model.max_seq_len,
    }

    logger.info(f"Model loaded: {total / 1e9:.1f}B total, {active / 1e9:.1f}B active, {dtype}")


def tokenize(text: str) -> torch.Tensor:
    """Tokenize text using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
    except ImportError:
        tokens = [ord(c) % config.model.vocab_size for c in text]
    return torch.tensor([tokens], device=device)


def detokenize(token_ids: List[int]) -> str:
    """Detokenize token IDs back to text."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.decode(token_ids)
    except ImportError:
        return "".join(chr(t % 128) for t in token_ids)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

# Parse args before creating app so lifespan can use them
_args = None


def parse_args():
    parser = argparse.ArgumentParser(description="OctoTetrahedral Inference Server")
    parser.add_argument("--config", type=str, default="7b",
                        choices=["default", "7b", "70b", "1.72t"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _args
    if _args is None:
        _args = parse_args()
    target_device = _args.device
    if target_device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    load_model(_args.config, _args.checkpoint, target_device)
    yield


app = FastAPI(
    title="OctoTetrahedral AGI API",
    description="1.72T MoE model inference API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_mem_used = gpu_mem_total = None
    if torch.cuda.is_available():
        gpu_mem_used = torch.cuda.memory_allocated() / 1e9
        gpu_mem_total = torch.cuda.get_device_properties(0).total_mem / 1e9
    return HealthResponse(
        status="ok" if model is not None else "loading",
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available(),
        gpu_memory_used_gb=gpu_mem_used,
        gpu_memory_total_gb=gpu_mem_total,
    )


@app.get("/info", response_model=ModelInfoResponse)
async def info():
    if not model_info:
        raise HTTPException(503, "Model not loaded")
    return ModelInfoResponse(**model_info)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    input_ids = tokenize(req.prompt)

    if input_ids.shape[1] > config.model.max_seq_len:
        input_ids = input_ids[:, -config.model.max_seq_len:]

    t0 = time.perf_counter()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k if req.top_k > 0 else None,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    new_tokens = generated[0, input_ids.shape[1]:].tolist()
    text = detokenize(new_tokens)
    tps = len(new_tokens) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

    return GenerateResponse(
        text=text,
        tokens_generated=len(new_tokens),
        generation_time_ms=round(elapsed_ms, 1),
        tokens_per_second=round(tps, 1),
    )


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Stream tokens via Server-Sent Events."""
    if model is None:
        raise HTTPException(503, "Model not loaded")

    input_ids = tokenize(req.prompt)
    if input_ids.shape[1] > config.model.max_seq_len:
        input_ids = input_ids[:, -config.model.max_seq_len:]

    async def event_stream():
        generated = input_ids.clone()
        with torch.no_grad():
            for i in range(req.max_tokens):
                if generated.size(1) > config.model.max_seq_len:
                    context = generated[:, -config.model.max_seq_len:]
                else:
                    context = generated

                output = model(input_ids=context)
                next_logits = output["logits"][:, -1, :] / max(req.temperature, 1e-8)

                if req.top_k > 0:
                    v, _ = torch.topk(next_logits, req.top_k)
                    next_logits[next_logits < v[:, [-1]]] = -float("inf")

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=1)

                token_text = detokenize([next_token[0].item()])
                yield f"data: {token_text}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _args = parse_args()
    uvicorn.run(app, host=_args.host, port=_args.port)
