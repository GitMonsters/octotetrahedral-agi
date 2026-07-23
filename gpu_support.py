"""GPU detection helpers and benchmark comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import torch

CPU_BASELINE_LATENCY_MS = 65.29
GPU_TARGET_LATENCY_MS = 6.53
BASELINE_CONCURRENCY = 50
PRIMARY_DEVICE_ENV = "OCTO_DEVICE"
LEGACY_DEVICE_ENV = "OCTOTETRAHEDRAL_DEVICE"


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved runtime device selection."""

    requested: str
    resolved: str
    accelerator: str | None
    fallback_used: bool
    reason: str


def _cuda_available() -> bool:
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices == "":
        return False
    return torch.cuda.is_available()


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _smoke_test(device_name: str) -> bool:
    try:
        probe = torch.tensor([1.0], device=device_name)
        _ = (probe + 1).sum().item()
        return True
    except Exception:
        return False


def _requested_device(preferred: str | None = None) -> str:
    return preferred or os.getenv(PRIMARY_DEVICE_ENV) or os.getenv(LEGACY_DEVICE_ENV) or "auto"


def _candidate_devices(preferred: str | None) -> list[str]:
    requested = _requested_device(preferred).lower()
    if requested == "auto":
        return ["cuda", "mps", "cpu"]
    if requested.startswith("cuda"):
        return [preferred or "cuda", "cpu"]
    if requested in {"mps", "metal"}:
        return ["mps", "cpu"]
    if requested == "cpu":
        return ["cpu"]
    return [preferred or requested, "cpu"]


def detect_device(preferred: str | None = None) -> DeviceInfo:
    """Resolve the best available device with accelerator smoke tests."""
    requested = _requested_device(preferred)
    requested_lower = requested.lower()

    for candidate in _candidate_devices(preferred):
        if candidate.startswith("cuda"):
            if not _cuda_available():
                continue
            if _smoke_test(candidate):
                return DeviceInfo(
                    requested,
                    candidate,
                    "cuda",
                    requested_lower not in {"auto", candidate.lower()},
                    "CUDA available",
                )
            continue

        if candidate == "mps":
            if not _mps_available():
                continue
            if _smoke_test(candidate):
                return DeviceInfo(
                    requested,
                    candidate,
                    "mps",
                    requested_lower not in {"auto", candidate},
                    "Metal available",
                )
            continue

        if candidate == "cpu":
            if requested == "cpu":
                reason = "CPU explicitly requested"
            elif requested == "auto":
                reason = "No accelerator available; using CPU fallback"
            else:
                reason = f"Requested device '{requested}' unavailable; using CPU fallback"
            return DeviceInfo(requested, "cpu", None, requested != "cpu", reason)

        if _smoke_test(candidate):
            accelerator = candidate.split(":", 1)[0]
            return DeviceInfo(
                requested,
                candidate,
                accelerator,
                requested_lower not in {"auto", candidate.lower()},
                "Requested device available",
            )

    return DeviceInfo(requested, "cpu", None, requested != "cpu", "Using CPU fallback")


def build_benchmark_comparison(
    cpu_latency_ms: float = CPU_BASELINE_LATENCY_MS,
    accelerator_latency_ms: float = GPU_TARGET_LATENCY_MS,
    concurrent_requests: int = BASELINE_CONCURRENCY,
) -> dict[str, float]:
    """Return a simple CPU vs. accelerator comparison summary."""
    if accelerator_latency_ms <= 0:
        raise ValueError("accelerator_latency_ms must be positive")

    cpu_throughput = 1000.0 / cpu_latency_ms if cpu_latency_ms > 0 else 0.0
    accelerator_throughput = 1000.0 / accelerator_latency_ms
    speedup = cpu_latency_ms / accelerator_latency_ms if cpu_latency_ms > 0 else math.inf

    return {
        "cpu_latency_ms": cpu_latency_ms,
        "accelerator_latency_ms": accelerator_latency_ms,
        "cpu_throughput_rps": cpu_throughput,
        "accelerator_throughput_rps": accelerator_throughput,
        "speedup_factor": speedup,
        "concurrent_requests": concurrent_requests,
    }
