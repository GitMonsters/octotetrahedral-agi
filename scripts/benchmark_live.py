#!/usr/bin/env python3
"""Live benchmark script for OctoTetrahedral AGI.

Measures inference latency and throughput on the available compute device,
compares CPU vs Metal (MPS) or CUDA, and optionally exports results.

Usage::

    # Auto-detect device and benchmark
    python scripts/benchmark_live.py

    # Force CPU only
    python scripts/benchmark_live.py --device cpu

    # Export results
    python scripts/benchmark_live.py --export json --output /tmp/results.json
    python scripts/benchmark_live.py --export csv  --output /tmp/results.csv
    python scripts/benchmark_live.py --export md   --output /tmp/results.md

    # Full run with more iterations
    python scripts/benchmark_live.py --iterations 200 --tokens 64
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure the repo root is on the import path when running as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch


# ---------------------------------------------------------------------------
# Core benchmarking
# ---------------------------------------------------------------------------


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def run_tensor_benchmark(
    device: str,
    n_tokens: int = 64,
    n_iter: int = 100,
    warmup: int = 20,
) -> dict[str, Any]:
    """Benchmark matrix ops (proxy for model forward pass) on *device*.

    Returns a dict with latency stats (ms) and throughput (req/s).
    """
    dev = torch.device(device)
    hidden = 512  # representative hidden size

    # Create proxy tensors (embedding look-up + matmul)
    emb = torch.nn.Embedding(50001, hidden).to(dev)
    proj = torch.nn.Linear(hidden, hidden, bias=False).to(dev)

    token_ids = torch.randint(0, 50000, (1, n_tokens), device=dev)

    # Warm-up
    for _ in range(warmup):
        with torch.no_grad():
            x = emb(token_ids)
            _ = proj(x)
    _sync(device)

    # Timed runs
    times: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        with torch.no_grad():
            x = emb(token_ids)
            _ = proj(x)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = statistics.mean(times)
    return {
        "device": device,
        "n_tokens": n_tokens,
        "n_iter": n_iter,
        "latency_ms": {
            "mean": round(mean_ms, 3),
            "min": round(min(times), 3),
            "max": round(max(times), 3),
            "p50": round(statistics.median(times), 3),
            "p95": round(_percentile(times, 0.95), 3),
            "p99": round(_percentile(times, 0.99), 3),
            "std": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        },
        "throughput_rps": round(1000.0 / mean_ms, 2),
    }


def _percentile(data: list[float], pct: float) -> float:
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def detect_available_devices() -> list[str]:
    """Return list of devices that pass a smoke-test."""
    devices: list[str] = ["cpu"]

    forced = os.getenv("OCTO_DEVICE") or os.getenv("OCTOTETRAHEDRAL_DEVICE")
    if forced and forced not in devices:
        devices = [forced]
        return devices

    if os.getenv("CUDA_VISIBLE_DEVICES", None) != "" and torch.cuda.is_available():
        try:
            a = torch.tensor([1.0], device="cuda")
            _ = (a + 1).item()
            devices.append("cuda")
        except Exception:
            pass

    if torch.backends.mps.is_available():
        try:
            a = torch.tensor([1.0, 2.0], device="mps")
            _ = (a + 1).sum().item()
            devices.append("mps")
        except Exception:
            pass

    return devices


def run_all_benchmarks(
    devices: list[str] | None = None,
    n_tokens: int = 64,
    n_iter: int = 100,
) -> list[dict[str, Any]]:
    """Run benchmarks on all available (or specified) devices."""
    if devices is None:
        devices = detect_available_devices()

    results = []
    for device in devices:
        print(f"\n⏱  Benchmarking on {device.upper()} ({n_tokens} tokens, {n_iter} iterations)…")
        try:
            result = run_tensor_benchmark(device, n_tokens=n_tokens, n_iter=n_iter)
            results.append(result)
            print(
                f"   ✅ {device.upper()}: {result['latency_ms']['mean']:.2f} ms mean  "
                f"({result['throughput_rps']:.1f} req/s)"
            )
        except Exception as exc:
            print(f"   ❌ {device.upper()} failed: {exc}")

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _speedup_label(base_ms: float, target_ms: float) -> str:
    if target_ms <= 0:
        return "n/a"
    ratio = base_ms / target_ms
    return f"{ratio:.1f}×"


def print_report(results: list[dict[str, Any]]) -> None:
    """Pretty-print a comparison table to stdout."""
    if not results:
        print("No results to display.")
        return

    cpu_mean = next(
        (r["latency_ms"]["mean"] for r in results if r["device"] == "cpu"), None
    )

    header = (
        f"\n{'Device':<14} {'Mean (ms)':>10} {'Min':>8} {'p50':>8} "
        f"{'p95':>8} {'p99':>8} {'Max':>8} {'Req/s':>8} {'Speedup':>9}"
    )
    sep = "-" * len(header)
    print(sep)
    print(" OctoTetrahedral AGI — Live Benchmark Results")
    print(f" Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        lat = r["latency_ms"]
        speedup = (
            _speedup_label(cpu_mean, lat["mean"])
            if cpu_mean and r["device"] != "cpu"
            else "baseline"
        )
        device_label = {"cpu": "CPU", "cuda": "CUDA (NVIDIA)", "mps": "Metal (MPS)"}.get(
            r["device"], r["device"].upper()
        )
        print(
            f"{device_label:<14} {lat['mean']:>10.2f} {lat['min']:>8.2f} "
            f"{lat['p50']:>8.2f} {lat['p95']:>8.2f} {lat['p99']:>8.2f} "
            f"{lat['max']:>8.2f} {r['throughput_rps']:>8.1f} {speedup:>9}"
        )

    print(sep)


def export_json(results: list[dict[str, Any]], path: str) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n📄 JSON results saved to: {path}")


def export_csv(results: list[dict[str, Any]], path: str) -> None:
    if not results:
        return
    fieldnames = [
        "device", "n_tokens", "n_iter",
        "mean_ms", "min_ms", "max_ms", "p50_ms", "p95_ms", "p99_ms", "std_ms",
        "throughput_rps",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            lat = r["latency_ms"]
            writer.writerow({
                "device": r["device"],
                "n_tokens": r["n_tokens"],
                "n_iter": r["n_iter"],
                "mean_ms": lat["mean"],
                "min_ms": lat["min"],
                "max_ms": lat["max"],
                "p50_ms": lat["p50"],
                "p95_ms": lat["p95"],
                "p99_ms": lat["p99"],
                "std_ms": lat["std"],
                "throughput_rps": r["throughput_rps"],
            })
    print(f"\n📊 CSV results saved to: {path}")


def export_markdown(results: list[dict[str, Any]], path: str) -> None:
    cpu_mean = next(
        (r["latency_ms"]["mean"] for r in results if r["device"] == "cpu"), None
    )

    lines = [
        "# OctoTetrahedral AGI — Benchmark Results",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| Device | Mean (ms) | Min | p50 | p95 | p99 | Max | Req/s | Speedup |",
        "|--------|-----------|-----|-----|-----|-----|-----|-------|---------|",
    ]

    for r in results:
        lat = r["latency_ms"]
        speedup = (
            _speedup_label(cpu_mean, lat["mean"])
            if cpu_mean and r["device"] != "cpu"
            else "baseline"
        )
        device_label = {"cpu": "CPU", "cuda": "CUDA (NVIDIA)", "mps": "Metal (MPS)"}.get(
            r["device"], r["device"].upper()
        )
        lines.append(
            f"| {device_label} | {lat['mean']} | {lat['min']} | {lat['p50']} "
            f"| {lat['p95']} | {lat['p99']} | {lat['max']} "
            f"| {r['throughput_rps']} | {speedup} |"
        )

    lines.append("")
    lines.append(f"*Tokens per request: {results[0]['n_tokens']} | Iterations: {results[0]['n_iter']}*")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n📝 Markdown report saved to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live benchmark for OctoTetrahedral AGI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device to benchmark (default: auto-detect all available)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=64,
        help="Number of input tokens per request (default: 64)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of timed iterations (default: 100)",
    )
    parser.add_argument(
        "--export",
        choices=["json", "csv", "md"],
        default=None,
        help="Export format (optional)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path for export (default: auto-named in /tmp)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("\n🚀 OctoTetrahedral AGI — Live Benchmark")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Tokens:  {args.tokens}")
    print(f"   Iters:   {args.iterations}")

    if args.device == "auto":
        devices = detect_available_devices()
        print(f"   Devices: {', '.join(d.upper() for d in devices)}")
    else:
        devices = [args.device]

    results = run_all_benchmarks(
        devices=devices,
        n_tokens=args.tokens,
        n_iter=args.iterations,
    )

    print_report(results)

    if args.export:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = args.output or f"/tmp/benchmark_{ts}.{args.export if args.export != 'md' else 'md'}"
        if args.export == "json":
            export_json(results, output)
        elif args.export == "csv":
            export_csv(results, output)
        elif args.export == "md":
            export_markdown(results, output)


if __name__ == "__main__":
    main()
