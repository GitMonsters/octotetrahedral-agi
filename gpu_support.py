"""GPU/Metal device support for OctoTetrahedral AGI.

Provides device auto-detection with the following priority chain:
  CUDA (NVIDIA) → MPS/Metal (Apple Silicon) → CPU

Each accelerator is smoke-tested before being accepted so that
environments that report availability but fail at runtime still
fall back safely to CPU.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_device() -> dict[str, Any]:
    """Return the best available compute device with metadata.

    Returns a dict with at minimum the key ``"device"`` (str) plus
    optional keys ``"accelerator"`` and ``"backend"``.

    Examples::

        >>> info = resolve_device()
        >>> info["device"] in ("cuda", "mps", "cpu")
        True
    """
    forced = os.getenv("OCTO_DEVICE") or os.getenv("OCTOTETRAHEDRAL_DEVICE")
    if forced:
        logger.info(f"🔧 Device forced via environment variable: {forced}")
        return {"device": forced, "accelerator": forced, "backend": "env-override"}

    # --- CUDA ---
    if os.getenv("CUDA_VISIBLE_DEVICES", None) != "":
        if torch.cuda.is_available():
            if _smoke_test_cuda():
                name = torch.cuda.get_device_name(0)
                logger.info(f"🟢 CUDA available: {name}")
                return {"device": "cuda", "accelerator": "cuda", "backend": "torch-cuda", "name": name}

    # --- Metal (MPS) ---
    if torch.backends.mps.is_available():
        if _smoke_test_mps():
            logger.info("🍎 Metal (MPS) available — using Apple Silicon GPU")
            return {"device": "mps", "accelerator": "mps", "backend": "torch-mps"}
        logger.warning("⚠️ MPS reported available but smoke-test failed; falling back to CPU")

    logger.info("🖥️  Using CPU")
    return {"device": "cpu", "accelerator": "cpu", "backend": "torch-cpu"}


def clear_device_cache(device: str | None = None) -> None:
    """Free cached memory on the given device (no-op on CPU)."""
    if device is None:
        device = resolve_device()["device"]
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def benchmark_device(device: str, n_iter: int = 50) -> dict[str, float]:
    """Run a small matrix-multiply benchmark and return timing statistics.

    Args:
        device: ``"cpu"``, ``"cuda"``, or ``"mps"``.
        n_iter: Number of iterations to average.

    Returns:
        Dict with keys ``latency_ms_mean``, ``latency_ms_min``,
        ``latency_ms_max``, ``throughput_ops_per_sec``.
    """
    dev = torch.device(device)
    size = 256
    a = torch.randn(size, size, device=dev)
    b = torch.randn(size, size, device=dev)

    # Warm-up
    for _ in range(5):
        _ = torch.matmul(a, b)
    _sync(device)

    times: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = torch.matmul(a, b)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times) / len(times)
    return {
        "latency_ms_mean": round(mean_ms, 3),
        "latency_ms_min": round(min(times), 3),
        "latency_ms_max": round(max(times), 3),
        "throughput_ops_per_sec": round(1000.0 / mean_ms, 1),
    }


def benchmark_comparison_table(n_iter: int = 50) -> str:
    """Return a Markdown table comparing CPU vs available accelerators."""
    rows: list[tuple[str, dict[str, float]]] = []

    rows.append(("CPU", benchmark_device("cpu", n_iter)))

    if torch.cuda.is_available() and _smoke_test_cuda():
        rows.append(("CUDA", benchmark_device("cuda", n_iter)))

    if torch.backends.mps.is_available() and _smoke_test_mps():
        rows.append(("Metal (MPS)", benchmark_device("mps", n_iter)))

    lines = [
        "| Device | Mean (ms) | Min (ms) | Max (ms) | Throughput (ops/s) |",
        "|--------|-----------|----------|----------|--------------------|",
    ]
    for name, stats in rows:
        lines.append(
            f"| {name} | {stats['latency_ms_mean']} | {stats['latency_ms_min']} "
            f"| {stats['latency_ms_max']} | {stats['throughput_ops_per_sec']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _smoke_test_cuda() -> bool:
    try:
        emb = torch.nn.Embedding(16, 8).to("cuda")
        ids = torch.tensor([1, 2, 3, 4], device="cuda")
        _ = emb(ids).sum().item()
        return True
    except Exception as exc:
        logger.debug(f"CUDA smoke-test failed: {exc}")
        return False


def _smoke_test_mps() -> bool:
    try:
        a = torch.tensor([1.0, 2.0, 3.0], device="mps")
        b = torch.tensor([4.0, 5.0, 6.0], device="mps")
        _ = (a + b).sum().item()
        return True
    except Exception as exc:
        logger.debug(f"MPS smoke-test failed: {exc}")
        return False


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


if __name__ == "__main__":
    info = resolve_device()
    print(f"Device: {info['device']}  (backend: {info.get('backend', 'n/a')})")
    print("\nBenchmark comparison:")
    print(benchmark_comparison_table(n_iter=20))
