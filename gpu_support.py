from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)

_MINIMUM_METAL_VERSION = (12, 3)


@dataclass(frozen=True)
class DeviceResolution:
    """Resolved runtime device plus fallback metadata."""

    requested: str | None
    selected: str
    backend: str
    fallback_reason: str | None = None
    cuda_available: bool = False
    mps_available: bool = False
    mps_built: bool = False

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.selected)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _SyntheticBenchmarkModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4096, 64)
        self.proj = torch.nn.Linear(64, 256)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.embedding(input_ids)
        logits = self.proj(hidden)
        return {"logits": logits}


def _normalize_device_name(device: str | None) -> str | None:
    if device is None:
        return None
    return str(device).strip().lower()


def _mps_backend() -> Any:
    return getattr(torch.backends, "mps", None)


def mps_built() -> bool:
    backend = _mps_backend()
    return bool(backend and backend.is_built())


def mps_available() -> bool:
    backend = _mps_backend()
    return bool(backend and backend.is_available())


def cuda_available() -> bool:
    if os.getenv("CUDA_VISIBLE_DEVICES", None) == "":
        return False
    return torch.cuda.is_available()


def _parse_macos_version(version: str) -> tuple[int, int] | None:
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def validate_mps_environment() -> tuple[bool, str | None]:
    if platform.system() != "Darwin":
        return False, "Metal Performance Shaders requires macOS."

    version = _parse_macos_version(platform.mac_ver()[0])
    if version is not None and version < _MINIMUM_METAL_VERSION:
        return (
            False,
            "Metal Performance Shaders requires macOS 12.3 or newer.",
        )

    if not mps_built():
        return False, "Installed PyTorch build does not include MPS support."

    if not mps_available():
        return False, "MPS backend is unavailable on this machine."

    return True, None


def synchronize_device(device: str | torch.device) -> None:
    device_name = str(device)
    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        return

    if device_name == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "synchronize"):
            mps_module.synchronize()


def clear_device_cache(device: str | torch.device) -> None:
    device_name = str(device)
    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return

    if device_name == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "empty_cache"):
            mps_module.empty_cache()


def _smoke_test(device: str) -> tuple[bool, str | None]:
    try:
        probe = torch.arange(8, dtype=torch.float32, device=device)
        _ = (probe.square().sum()).item()
        synchronize_device(device)
        return True, None
    except Exception as exc:  # pragma: no cover - hardware specific
        return False, str(exc)


def resolve_device(requested: str | None = None) -> DeviceResolution:
    requested_name = _normalize_device_name(
        requested or os.getenv("OCTO_DEVICE") or os.getenv("OCTOTETRAHEDRAL_DEVICE")
    )
    cuda_ok = cuda_available()
    mps_ok = mps_available()
    mps_ready, mps_reason = validate_mps_environment()

    def _cpu(reason: str | None = None) -> DeviceResolution:
        return DeviceResolution(
            requested=requested_name,
            selected="cpu",
            backend="cpu",
            fallback_reason=reason,
            cuda_available=cuda_ok,
            mps_available=mps_ok,
            mps_built=mps_built(),
        )

    if requested_name:
        if requested_name == "cpu":
            return _cpu()

        if requested_name.startswith("cuda"):
            if not cuda_ok:
                return _cpu("CUDA was requested but is unavailable.")
            passed, smoke_reason = _smoke_test(requested_name)
            if passed:
                return DeviceResolution(
                    requested=requested_name,
                    selected=requested_name,
                    backend="cuda",
                    cuda_available=cuda_ok,
                    mps_available=mps_ok,
                    mps_built=mps_built(),
                )
            return _cpu(f"CUDA smoke test failed: {smoke_reason}")

        if requested_name == "mps":
            if not mps_ready:
                return _cpu(mps_reason)
            passed, smoke_reason = _smoke_test("mps")
            if passed:
                return DeviceResolution(
                    requested=requested_name,
                    selected="mps",
                    backend="mps",
                    cuda_available=cuda_ok,
                    mps_available=mps_ok,
                    mps_built=mps_built(),
                )
            return _cpu(f"Metal smoke test failed: {smoke_reason}")

        return _cpu(f"Unsupported device override: {requested_name}")

    if cuda_ok:
        passed, smoke_reason = _smoke_test("cuda")
        if passed:
            return DeviceResolution(
                requested=None,
                selected="cuda",
                backend="cuda",
                cuda_available=cuda_ok,
                mps_available=mps_ok,
                mps_built=mps_built(),
            )
        logger.warning("CUDA advertised but smoke test failed: %s", smoke_reason)

    if mps_ready:
        passed, smoke_reason = _smoke_test("mps")
        if passed:
            return DeviceResolution(
                requested=None,
                selected="mps",
                backend="mps",
                cuda_available=cuda_ok,
                mps_available=mps_ok,
                mps_built=mps_built(),
            )
        return _cpu(f"Metal smoke test failed: {smoke_reason}")

    return _cpu(mps_reason)


def prepare_input_tensor(
    input_ids: Sequence[int],
    device: str | torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor([list(input_ids)], dtype=torch.long)
    if str(device) == "mps":
        tensor = tensor.contiguous()
    return tensor.to(torch.device(device))


def warmup_device(
    device: str | torch.device,
    sequence_length: int = 16,
    iterations: int = 2,
) -> None:
    device_name = str(device)
    if device_name == "cpu":
        return

    for _ in range(iterations):
        probe = prepare_input_tensor(range(sequence_length), device_name)
        values = probe.to(dtype=torch.float32)
        _ = values.mul(2.0).sum().item()
        synchronize_device(device_name)


def get_memory_stats(device: str | torch.device) -> dict[str, float | None]:
    device_name = str(device)
    allocated_mb = driver_allocated_mb = reserved_mb = None

    if device_name.startswith("cuda") and torch.cuda.is_available():
        allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
    elif device_name == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None:
            if hasattr(mps_module, "current_allocated_memory"):
                allocated_mb = (
                    mps_module.current_allocated_memory() / (1024 * 1024)
                )
            if hasattr(mps_module, "driver_allocated_memory"):
                driver_allocated_mb = (
                    mps_module.driver_allocated_memory() / (1024 * 1024)
                )

    return {
        "allocated_mb": allocated_mb,
        "driver_allocated_mb": driver_allocated_mb,
        "reserved_mb": reserved_mb,
    }


def benchmark_inference(
    model: torch.nn.Module | None = None,
    input_ids: Sequence[int] | None = None,
    device: str = "cpu",
    runs: int = 10,
    warmup_runs: int = 3,
) -> dict[str, Any]:
    resolution = resolve_device(device)
    working_model = model or _SyntheticBenchmarkModel()
    working_model = working_model.to(resolution.torch_device)
    # Keep eval mode for deterministic inference; inference_mode disables autograd only.
    working_model.eval()

    tokens = list(input_ids or range(32))
    tensor = prepare_input_tensor(tokens, resolution.selected)
    warmup_device(resolution.selected, sequence_length=len(tokens))
    clear_device_cache(resolution.selected)

    if resolution.selected.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = working_model(tensor)
            synchronize_device(resolution.selected)

        latencies_ms: list[float] = []
        start_total = time.perf_counter()
        for _ in range(runs):
            start = time.perf_counter()
            result = working_model(tensor)
            if isinstance(result, dict) and "logits" in result:
                _ = result["logits"].sum().item()
            synchronize_device(resolution.selected)
            latencies_ms.append((time.perf_counter() - start) * 1000)
        total_seconds = time.perf_counter() - start_total

    memory = get_memory_stats(resolution.selected)
    total_tokens = len(tokens) * runs
    throughput = total_tokens / total_seconds if total_seconds else 0.0

    return {
        "device": resolution.selected,
        "backend": resolution.backend,
        "requested": resolution.requested,
        "fallback_reason": resolution.fallback_reason,
        "runs": runs,
        "sequence_length": len(tokens),
        "latency_ms_avg": sum(latencies_ms) / len(latencies_ms),
        "latency_ms_min": min(latencies_ms),
        "latency_ms_max": max(latencies_ms),
        "throughput_tokens_per_second": throughput,
        "memory": memory,
    }


def compare_benchmarks(
    model: torch.nn.Module | None = None,
    input_ids: Sequence[int] | None = None,
    runs: int = 10,
) -> list[dict[str, Any]]:
    results = [benchmark_inference(model=model, input_ids=input_ids, device="cpu", runs=runs)]

    for candidate in ("mps", "cuda"):
        result = benchmark_inference(
            model=model,
            input_ids=input_ids,
            device=candidate,
            runs=runs,
        )
        if result["device"] != "cpu":
            results.append(result)

    cpu_latency = results[0]["latency_ms_avg"]
    for result in results:
        result["speedup_vs_cpu"] = (
            cpu_latency / result["latency_ms_avg"]
            if result["latency_ms_avg"] > 0
            else None
        )
    return results


def render_benchmark_table(results: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Device | Avg latency (ms) | Throughput (tokens/s) | Allocated memory (MB) | Speedup vs CPU |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        memory = result["memory"].get("allocated_mb")
        speedup = result.get("speedup_vs_cpu")
        lines.append(
            "| {device} | {latency:.2f} | {throughput:.2f} | {memory} | {speedup} |".format(
                device=result["device"],
                latency=result["latency_ms_avg"],
                throughput=result["throughput_tokens_per_second"],
                memory=f"{memory:.2f}" if memory is not None else "n/a",
                speedup=f"{speedup:.2f}x" if speedup is not None else "n/a",
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="OctoTetrahedral GPU/Metal benchmark")
    parser.add_argument("--runs", type=int, default=20, help="Number of timed runs")
    parser.add_argument("--tokens", type=int, default=128, help="Token count per run")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a markdown table",
    )
    args = parser.parse_args()

    input_ids = list(range(args.tokens))
    results = compare_benchmarks(input_ids=input_ids, runs=args.runs)
    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(render_benchmark_table(results))


if __name__ == "__main__":
    main()
