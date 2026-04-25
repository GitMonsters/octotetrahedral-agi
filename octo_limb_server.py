#!/usr/bin/env python3
"""
OctoTetrahedral Parallel Limb Server
=====================================
FastAPI microserver that exposes the 11 cognitive limbs as independent HTTP
endpoints so Rust (or any language) can call them in parallel.

Endpoints
---------
POST /encode          → shared memory_enhanced representation [B, L, D]
POST /limb/{name}     → run one named limb, returns output tensor + confidence
POST /braid           → run KimiCognitiveBraid over a set of limb outputs
GET  /limbs           → list available limb names
GET  /healthz         → liveness probe

Wire-format
-----------
Tensors are transmitted as little-endian float32 raw bytes, base64-encoded.
Shape is sent as a separate JSON array.  This keeps serialisation overhead
minimal and avoids JSON floating-point precision loss.

Usage
-----
    python octo_limb_server.py              # port 8765, 1 worker
    python octo_limb_server.py --workers 4  # 4 parallel workers
    python octo_limb_server.py --port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger("octo-limb-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

app = FastAPI(title="OctoTetrahedral Limb Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None   # loaded once at startup

LIMB_NAMES = [
    "memory", "planning", "language", "spatial",
    "reasoning", "metacognition", "action",
    "visualization", "imagination", "empathy", "emotion", "ethics",
]


# ──────────────────────────────────────────────────────────────────────────────
# Tensor wire helpers
# ──────────────────────────────────────────────────────────────────────────────

def tensor_to_wire(t: torch.Tensor) -> tuple[str, list[int]]:
    # Explicit little-endian float32 so Rust (which assumes LE) is always correct
    arr = t.detach().cpu().to(torch.float32).numpy().astype(np.dtype("<f4"))
    return base64.b64encode(arr.tobytes()).decode(), list(arr.shape)


def tensor_from_wire(data: str, shape: list[int]) -> torch.Tensor:
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())


# ──────────────────────────────────────────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────────────────────────────────────────

class EncodeRequest(BaseModel):
    input_ids: List[int]          # flat token list, single sequence
    batch_size: int = 1           # repeat for batch (testing only)


class EncodeResponse(BaseModel):
    data: str                     # base64 float32 bytes
    shape: List[int]              # [B, L, D]
    elapsed_ms: float


class LimbRequest(BaseModel):
    data: str                     # base64 float32 memory_enhanced bytes
    shape: List[int]              # [B, L, D]


class LimbResponse(BaseModel):
    data: str                     # base64 float32 output bytes
    shape: List[int]
    confidence: float
    limb: str
    elapsed_ms: float


class BraidRequest(BaseModel):
    streams: List[dict]           # list of {data, shape} dicts, one per limb


class BraidResponse(BaseModel):
    streams: List[dict]           # updated streams after Block AttnRes
    elapsed_ms: float
    n_blocks: int


# ──────────────────────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_model() -> None:
    global _model
    logger.info("Loading OctoTetrahedralModel…")
    t0 = time.perf_counter()
    from model import OctoTetrahedralModel
    _model = OctoTetrahedralModel().eval()
    elapsed = time.perf_counter() - t0
    params = sum(p.numel() for p in _model.parameters())
    logger.info(f"Model loaded in {elapsed:.1f}s  ({params:,} params)")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/limbs")
async def list_limbs() -> dict:
    return {"limbs": LIMB_NAMES}


@app.post("/encode", response_model=EncodeResponse)
async def encode(req: EncodeRequest) -> EncodeResponse:
    if _model is None:
        raise HTTPException(503, "Model not ready")
    ids = torch.tensor([req.input_ids] * req.batch_size, dtype=torch.long)
    t0 = time.perf_counter()
    mem_enh = _model.encode_for_parallel(ids)          # [B, L, D]
    elapsed = (time.perf_counter() - t0) * 1000
    data, shape = tensor_to_wire(mem_enh)
    return EncodeResponse(data=data, shape=shape, elapsed_ms=round(elapsed, 2))


@app.post("/limb/{name}", response_model=LimbResponse)
async def run_limb(name: str, req: LimbRequest) -> LimbResponse:
    if _model is None:
        raise HTTPException(503, "Model not ready")
    if name not in LIMB_NAMES:
        raise HTTPException(404, f"Unknown limb '{name}'. Available: {LIMB_NAMES}")
    mem_enh = tensor_from_wire(req.data, req.shape)
    t0 = time.perf_counter()
    out, conf, _ = _model.run_limb(name, mem_enh)
    elapsed = (time.perf_counter() - t0) * 1000
    conf_val = conf.mean().item() if torch.is_tensor(conf) else float(conf)
    data, shape = tensor_to_wire(out)
    return LimbResponse(data=data, shape=shape, confidence=conf_val, limb=name, elapsed_ms=round(elapsed, 2))


@app.post("/braid", response_model=BraidResponse)
async def braid(req: BraidRequest) -> BraidResponse:
    """
    Run KimiCognitiveBraid (Block AttnRes) over the received limb streams.
    This endpoint exists for Python-side braid; the Rust server implements the
    same operation natively using the exported pseudo-query weights.
    """
    if _model is None:
        raise HTTPException(503, "Model not ready")
    if _model.kimi_braid is None:
        raise HTTPException(400, "KimiCognitiveBraid not enabled in config")

    streams = [tensor_from_wire(s["data"], s["shape"]) for s in req.streams]
    t0 = time.perf_counter()
    updated, info = _model.kimi_braid(streams)
    elapsed = (time.perf_counter() - t0) * 1000

    out_streams = [{"data": d, "shape": sh} for d, sh in (tensor_to_wire(u) for u in updated)]
    return BraidResponse(streams=out_streams, elapsed_ms=round(elapsed, 2), n_blocks=info["n_blocks"])


@app.post("/export_kimi_weights")
async def export_kimi_weights(path: str = "kimi_weights.json") -> dict:
    """Export KimiCognitiveBraid pseudo-query weights for the Rust server."""
    if _model is None:
        raise HTTPException(503, "Model not ready")
    _model.export_kimi_weights(path)
    return {"exported": path}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OctoTetrahedral Parallel Limb Server")
    parser.add_argument("--port",    type=int, default=8765)
    parser.add_argument("--host",    type=str, default="0.0.0.0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload",  action="store_true")
    args = parser.parse_args()

    logger.info(f"Starting limb server on {args.host}:{args.port}  workers={args.workers}")
    uvicorn.run(
        "octo_limb_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
