"""
OctoTetrahedral AGI — NVIDIA NIM Integration

Provides TensorRT-LLM export and NIM-compatible model packaging.

Two modes:
1. Export: Convert PyTorch model to ONNX → TensorRT engine for NIM deployment
2. Benchmark: Compare PyTorch vs TensorRT inference speed

Requires (on GPU machine):
    pip install onnx tensorrt nvidia-modelopt

Usage:
    # Export to ONNX (first step, can run on CPU)
    python nim_export.py --config 7b --checkpoint checkpoints/best.pt --export onnx

    # Build TensorRT engine (requires GPU + TensorRT)
    python nim_export.py --config 7b --checkpoint checkpoints/best.pt --export tensorrt

    # Benchmark comparison
    python nim_export.py --config 7b --checkpoint checkpoints/best.pt --benchmark

    # Package for NIM deployment
    python nim_export.py --config 7b --checkpoint checkpoints/best.pt --package-nim
"""

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# ONNX Export
# ────────────────────────────────────────────────────────────────

def export_onnx(
    model: nn.Module,
    config,
    output_dir: str = "nim_export",
    max_batch: int = 1,
    max_seq_len: int = 512,
    opset_version: int = 17,
) -> str:
    """Export model to ONNX format."""
    import onnx

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_path = str(out_path / "model.onnx")

    model.eval()
    model.cpu()

    # Create dummy input
    dummy_input = torch.randint(0, config.model.vocab_size, (max_batch, max_seq_len))

    logger.info(f"Exporting ONNX: batch={max_batch}, seq_len={max_seq_len}, opset={opset_version}")

    # Wrap forward to return only logits
    class LogitsWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model

        def forward(self, input_ids):
            output = self.model(input_ids=input_ids)
            if isinstance(output, dict):
                return output["logits"]
            return output

    wrapper = LogitsWrapper(model)

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Verify
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(onnx_path) / 1e6
    logger.info(f"ONNX export complete: {onnx_path} ({size_mb:.0f} MB)")
    return onnx_path


# ────────────────────────────────────────────────────────────────
# TensorRT Build
# ────────────────────────────────────────────────────────────────

def build_tensorrt_engine(
    onnx_path: str,
    output_dir: str = "nim_export",
    max_batch: int = 1,
    max_seq_len: int = 512,
    fp16: bool = True,
) -> str:
    """Build TensorRT engine from ONNX model."""
    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT not installed. Install with: pip install tensorrt")
        raise

    out_path = Path(output_dir)
    engine_path = str(out_path / "model.engine")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    logger.info(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"TensorRT parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")

    # Set optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input_ids",
        (1, 1),                        # min
        (max_batch, max_seq_len // 2),  # opt
        (max_batch, max_seq_len),       # max
    )
    config.add_optimization_profile(profile)

    logger.info("Building TensorRT engine (this may take several minutes)...")
    t0 = time.time()
    engine = builder.build_serialized_network(network, config)
    build_time = time.time() - t0

    if engine is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(engine)

    size_mb = os.path.getsize(engine_path) / 1e6
    logger.info(f"TensorRT engine built in {build_time:.0f}s: {engine_path} ({size_mb:.0f} MB)")
    return engine_path


# ────────────────────────────────────────────────────────────────
# Benchmark
# ────────────────────────────────────────────────────────────────

def benchmark_pytorch(
    model: nn.Module,
    config,
    device: str = "cuda",
    num_runs: int = 20,
    warmup: int = 3,
    seq_len: int = 128,
    gen_tokens: int = 64,
) -> Dict:
    """Benchmark PyTorch inference speed."""
    import tiktoken

    model.eval()
    model.to(device)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    prompt = "Task: Learn the pattern. [1 2 3 | 4 5 6]->[7 8 9 | 0 1 2] [3 4 5 | 6 7 8]->["
    tokens = tokenizer.encode(prompt)[:seq_len]
    input_ids = torch.tensor([tokens], device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(input_ids=input_ids, max_new_tokens=gen_tokens, temperature=0.0)

    # Timed runs
    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=gen_tokens, temperature=0.0)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    tokens_generated = gen_tokens * num_runs
    total_time_ms = sum(latencies)
    avg_latency = total_time_ms / num_runs
    throughput = tokens_generated / (total_time_ms / 1000)

    return {
        "framework": "pytorch",
        "device": device,
        "num_runs": num_runs,
        "seq_len": seq_len,
        "gen_tokens": gen_tokens,
        "avg_latency_ms": round(avg_latency, 1),
        "p50_latency_ms": round(sorted(latencies)[num_runs // 2], 1),
        "p99_latency_ms": round(sorted(latencies)[int(num_runs * 0.99)], 1),
        "throughput_tokens_per_sec": round(throughput, 1),
    }


# ────────────────────────────────────────────────────────────────
# NIM Package
# ────────────────────────────────────────────────────────────────

def package_nim(
    config,
    checkpoint_path: str,
    output_dir: str = "nim_package",
    model_name: str = "octotetrahedral-7b-moe",
) -> str:
    """Create NIM-compatible model package directory structure."""
    pkg = Path(output_dir)
    pkg.mkdir(parents=True, exist_ok=True)

    # model_config.json — NIM metadata
    nim_config = {
        "model": {
            "name": model_name,
            "version": "1.0.0",
            "framework": "pytorch",
            "architecture": "OctoTetrahedralMoE",
            "precision": "bf16",
            "max_batch_size": 8,
            "max_input_len": config.model.max_seq_len,
            "max_output_len": 4096,
        },
        "tokenizer": {
            "type": "tiktoken",
            "encoding": "cl100k_base",
        },
        "inference": {
            "engine": "pytorch",
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
        },
        "moe": {
            "num_experts": config.moe.num_experts,
            "top_k": config.moe.top_k,
            "expert_ffn_dim": config.moe.expert_ffn_dim,
        },
        "endpoints": [
            {"path": "/generate", "method": "POST"},
            {"path": "/generate/stream", "method": "POST"},
            {"path": "/health", "method": "GET"},
            {"path": "/info", "method": "GET"},
            {"path": "/metrics", "method": "GET"},
        ],
    }

    config_path = pkg / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(nim_config, f, indent=2)
    logger.info(f"Wrote {config_path}")

    # Copy checkpoint
    ckpt_dst = pkg / "model.pt"
    if Path(checkpoint_path).exists():
        shutil.copy2(checkpoint_path, ckpt_dst)
        logger.info(f"Copied checkpoint to {ckpt_dst}")

    # Dockerfile for NIM
    dockerfile = pkg / "Dockerfile"
    dockerfile.write_text(f"""\
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model code
COPY model.py config.py serve.py train_distributed.py ./
COPY core/ ./core/
COPY limbs/ ./limbs/
COPY adaptation/ ./adaptation/
COPY sync/ ./sync/
COPY physics/ ./physics/
COPY cognition.py ./
COPY data/ ./data/

# Copy model artifacts
COPY {pkg.name}/model.pt /models/model.pt
COPY {pkg.name}/model_config.json /models/model_config.json

ENV MODEL_CONFIG=/models/model_config.json
ENV CHECKPOINT=/models/model.pt
ENV OCTO_CONFIG=7b

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "serve.py", "--config", "$OCTO_CONFIG", \\
     "--checkpoint", "$CHECKPOINT", "--device", "cuda:0"]
""")
    logger.info(f"Wrote {dockerfile}")

    # docker-compose for NIM deployment
    compose = pkg / "docker-compose.yml"
    compose.write_text(f"""\
version: '3.8'
services:
  {model_name}:
    build: ..
    image: {model_name}:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - OCTO_API_KEYS=${{OCTO_API_KEYS}}
      - OCTO_CONFIG=7b
      - CHECKPOINT=/models/model.pt
    ports:
      - "8000:8000"
    volumes:
      - ./model.pt:/models/model.pt
      - ./model_config.json:/models/model_config.json
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
""")
    logger.info(f"Wrote {compose}")

    logger.info(f"\nNIM package created: {pkg}/")
    logger.info(f"To deploy: cd {pkg} && docker compose up --build")
    return str(pkg)


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NVIDIA NIM Export & Benchmark")
    parser.add_argument("--config", type=str, default="7b", choices=["default", "7b", "70b", "1.72t"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="nim_export")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", choices=["onnx", "tensorrt"], help="Export format")
    group.add_argument("--benchmark", action="store_true", help="Benchmark PyTorch speed")
    group.add_argument("--package-nim", action="store_true", help="Create NIM deployment package")

    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument("--gen-tokens", type=int, default=64)
    args = parser.parse_args()

    from train_distributed import load_config
    from model import OctoTetrahedralModel

    config = load_config(args.config)
    device = args.device or config.device

    logger.info(f"Loading {args.config} model from {args.checkpoint}...")
    model = OctoTetrahedralModel(config, use_geometric_physics=False)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_state = model.state_dict()
    pretrained = ckpt["model_state_dict"]
    filtered = {k: v for k, v in pretrained.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)

    total = model.get_num_params()
    logger.info(f"Model loaded: {total / 1e9:.2f}B params")

    if args.export == "onnx":
        export_onnx(model, config, args.output_dir, args.max_batch, args.max_seq_len)

    elif args.export == "tensorrt":
        onnx_path = Path(args.output_dir) / "model.onnx"
        if not onnx_path.exists():
            logger.info("ONNX not found, exporting first...")
            export_onnx(model, config, args.output_dir, args.max_batch, args.max_seq_len)
        build_tensorrt_engine(str(onnx_path), args.output_dir, args.max_batch, args.max_seq_len, args.fp16)

    elif args.benchmark:
        results = benchmark_pytorch(
            model, config, device,
            num_runs=args.benchmark_runs,
            gen_tokens=args.gen_tokens,
        )
        print("\n" + "=" * 50)
        print("PYTORCH BENCHMARK RESULTS")
        print("=" * 50)
        for k, v in results.items():
            print(f"  {k:30s}  {v}")

    elif args.package_nim:
        package_nim(config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
