"""Metal (Apple Silicon MPS) GPU support for OctoTetrahedral AGI.

Provides device selection with Metal/MPS auto-detection and CPU fallback,
tensor optimization utilities for Metal, and CPU vs Metal performance
comparison.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def is_metal_available() -> bool:
    """Return True when Metal Performance Shaders (MPS) is available."""
    return torch.backends.mps.is_available()


def get_metal_device() -> Optional[torch.device]:
    """Return a Metal device or None if unavailable."""
    if is_metal_available():
        return torch.device("mps")
    return None


def select_device(prefer_metal: bool = True) -> torch.device:
    """Select the best available device.

    Priority order (when *prefer_metal* is True):
    1. CUDA (if available and working)
    2. Metal / MPS (Apple Silicon, macOS)
    3. CPU (universal fallback)

    The environment variable ``OCTO_DEVICE`` / ``OCTOTETRAHEDRAL_DEVICE``
    overrides auto-selection.
    """
    forced = os.getenv("OCTO_DEVICE") or os.getenv("OCTOTETRAHEDRAL_DEVICE")
    if forced:
        logger.info("Device forced via env: %s", forced)
        return torch.device(forced)

    # CUDA
    if torch.cuda.is_available():
        try:
            probe = torch.zeros(4, device="cuda")
            _ = (probe + 1).sum().item()
            logger.info("Selected device: cuda")
            return torch.device("cuda")
        except Exception as exc:  # pragma: no cover
            logger.warning("CUDA probe failed (%s), trying next device", exc)

    # Metal / MPS
    if prefer_metal and is_metal_available():
        try:
            probe = torch.zeros(4, device="mps")
            _ = (probe + 1).sum().item()
            logger.info("Selected device: mps (Apple Metal)")
            return torch.device("mps")
        except Exception as exc:  # pragma: no cover
            logger.warning("MPS probe failed (%s), falling back to cpu", exc)

    logger.info("Selected device: cpu")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Metal optimization helpers
# ---------------------------------------------------------------------------


def optimize_for_metal(model: nn.Module) -> nn.Module:
    """Move *model* to the MPS device and set it to eval mode.

    If MPS is unavailable the model is returned unchanged.
    """
    device = get_metal_device()
    if device is None:
        logger.debug("optimize_for_metal: MPS not available, skipping")
        return model
    model = model.to(device)
    model.eval()
    logger.info("Model optimized for Metal (MPS)")
    return model


def metal_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run a single forward pass on Metal with no_grad context.

    Args:
        model: The neural network model (already on *device*).
        input_tensor: Input tensor (will be moved to *device*).
        device: Target device; defaults to ``select_device()``.

    Returns:
        Output tensor on CPU.
    """
    if device is None:
        device = select_device()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_ids=input_tensor)
    if isinstance(output, dict):
        logits = output.get("logits", next(iter(output.values())))
    else:
        logits = output
    return logits.cpu()


# ---------------------------------------------------------------------------
# Performance comparison
# ---------------------------------------------------------------------------


def _time_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int = 2,
    runs: int = 10,
) -> float:
    """Return mean inference latency (ms) over *runs* iterations."""
    model = model.to(device)
    model.eval()
    inp = input_tensor.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids=inp)

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_ids=inp)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms / runs


def compare_cpu_vs_metal(
    model: nn.Module,
    seq_len: int = 32,
    batch_size: int = 1,
    runs: int = 10,
) -> Dict[str, object]:
    """Benchmark *model* on CPU and Metal, returning a comparison dict.

    Returns a dict with keys:
        cpu_latency_ms, metal_latency_ms (or None), speedup (or None),
        metal_available, device_used.
    """
    input_ids = torch.randint(0, 64, (batch_size, seq_len))

    cpu_latency = _time_inference(model, input_ids, torch.device("cpu"), runs=runs)
    logger.info("CPU latency: %.2f ms", cpu_latency)

    metal_latency: Optional[float] = None
    speedup: Optional[float] = None
    device_used = "cpu"

    if is_metal_available():
        try:
            metal_latency = _time_inference(
                model, input_ids, torch.device("mps"), runs=runs
            )
            speedup = cpu_latency / metal_latency if metal_latency > 0 else None
            device_used = "mps"
            logger.info(
                "Metal latency: %.2f ms  (speedup %.2fx)",
                metal_latency,
                speedup or 0,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Metal benchmark failed: %s", exc)

    return {
        "cpu_latency_ms": round(cpu_latency, 3),
        "metal_latency_ms": round(metal_latency, 3) if metal_latency is not None else None,
        "speedup": round(speedup, 2) if speedup is not None else None,
        "metal_available": is_metal_available(),
        "device_used": device_used,
    }


# ---------------------------------------------------------------------------
# Device info helper
# ---------------------------------------------------------------------------


def device_info() -> Dict[str, object]:
    """Return a dict with device capabilities for the /health endpoint."""
    info: Dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": is_metal_available(),
        "selected_device": str(select_device()),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    if is_metal_available():
        info["metal_backend"] = "Apple MPS"
    return info
