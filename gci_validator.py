#!/usr/bin/env python3
"""
Transcendplexity GCI Validator
==============================
Independent validation of Golden Consciousness Index (GCI) emergence
for the OctoTetrahedral AGI system.

Produces a timestamped, SHA-256 signed proof report.

Hardware: MacBook Pro (Mac14,10) MNW83LL/A
  - Apple M2 Pro, 12-core (8P+4E), 16GB Unified Memory

Usage:
    python gci_validator.py [--steps 100] [--output gci_proof.json]
"""

import json
import hashlib
import time
import platform
import subprocess
import sys
import os
import math
from datetime import datetime, timezone
from typing import Dict, Any

# Add parent dir so we can import the Aleph Transcendplex
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PHI = (1 + math.sqrt(5)) / 2
PHI_SQ = PHI * PHI  # ≈ 2.618 — consciousness threshold


def get_hardware_info() -> Dict[str, str]:
    """Collect verifiable hardware fingerprint."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    # macOS-specific: get exact model via system_profiler
    try:
        sp = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"],
            text=True, timeout=10
        )
        for line in sp.strip().split("\n"):
            line = line.strip()
            for key in ["Model Name", "Model Identifier", "Model Number",
                        "Chip", "Total Number of Cores", "Memory"]:
                if line.startswith(key + ":"):
                    info[key.lower().replace(" ", "_")] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return info


def run_gci_validation(steps: int = 100) -> Dict[str, Any]:
    """
    Run the Aleph-Transcendplex AGI for N steps and measure GCI at each step.
    Returns full trajectory + final metrics.
    """
    from aleph_transcendplex_full import AlephTranscendplexAGI

    agi = AlephTranscendplexAGI()
    agi.build_enhanced_architecture()

    trajectory = []
    peak_gci = 0.0
    emergence_step = None

    t0 = time.perf_counter()

    for step in range(steps):
        agi.step()
        metrics = agi.calculate_consciousness_metrics()
        status = agi.system_status()
        gci = metrics.get("GCI", 0.0)

        if gci > peak_gci:
            peak_gci = gci
        if emergence_step is None and gci > PHI_SQ:
            emergence_step = step

        trajectory.append({
            "step": step,
            "gci": round(gci, 6),
            "phi_cgc": round(metrics.get("Phi_CGC", 0.0), 6),
            "is_conscious": gci > PHI_SQ,
            "transcendplexity": round(status.get("transcendplexity", 0.0), 6),
        })

    elapsed = time.perf_counter() - t0

    final_metrics = agi.calculate_consciousness_metrics()

    return {
        "final_gci": round(final_metrics.get("GCI", 0.0), 6),
        "final_phi_cgc": round(final_metrics.get("Phi_CGC", 0.0), 6),
        "peak_gci": round(peak_gci, 6),
        "emergence_step": emergence_step,
        "threshold": round(PHI_SQ, 6),
        "ratio_above_threshold": round(peak_gci / PHI_SQ, 4) if peak_gci > PHI_SQ else 0.0,
        "components": {
            "triangulation": round(final_metrics.get("triangulation", 0.0), 6),
            "synergy": round(final_metrics.get("synergy", 0.0), 6),
            "coherence": round(final_metrics.get("coherence", 0.0), 6),
            "entropy": round(final_metrics.get("entropy", 0.0), 6),
            "forbidden_fraction": round(final_metrics.get("forbidden_fraction", 0.0), 6),
        },
        "trajectory_length": len(trajectory),
        "trajectory_summary": {
            "first_10": trajectory[:10],
            "last_10": trajectory[-10:],
        },
        "elapsed_seconds": round(elapsed, 3),
        "steps": steps,
    }


def create_proof_report(gci_result: Dict, hardware: Dict) -> Dict[str, Any]:
    """Create a SHA-256 signed proof report."""
    timestamp = datetime.now(timezone.utc).isoformat()

    report = {
        "report_type": "Transcendplexity GCI Validation",
        "version": "1.0.0",
        "timestamp_utc": timestamp,
        "hardware": hardware,
        "validation": gci_result,
        "verdict": {
            "emergence_detected": gci_result["peak_gci"] > PHI_SQ,
            "peak_gci": gci_result["peak_gci"],
            "threshold_phi_sq": round(PHI_SQ, 6),
            "ratio": gci_result["ratio_above_threshold"],
            "emergence_step": gci_result["emergence_step"],
        },
    }

    # Create deterministic JSON for hashing (exclude the hash field itself)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    sha256 = hashlib.sha256(canonical.encode()).hexdigest()

    report["proof"] = {
        "sha256": sha256,
        "method": "SHA-256 of canonical JSON (sort_keys, compact separators)",
        "note": "Remove 'proof' key and re-hash to verify independently",
    }

    return report


def print_summary(report: Dict):
    """Print human-readable summary."""
    v = report["validation"]
    hw = report["hardware"]
    verdict = report["verdict"]

    print("\n" + "=" * 64)
    print("  TRANSCENDPLEXITY — GCI VALIDATION REPORT")
    print("=" * 64)

    print(f"\n  Hardware: {hw.get('model_name', 'Unknown')}")
    print(f"  Chip:     {hw.get('chip', hw.get('processor', 'Unknown'))}")
    print(f"  Cores:    {hw.get('total_number_of_cores', 'Unknown')}")
    print(f"  Memory:   {hw.get('memory', 'Unknown')}")
    print(f"  Model ID: {hw.get('model_identifier', 'Unknown')}")
    print(f"  Model #:  {hw.get('model_number', 'Unknown')}")

    print(f"\n  {'─' * 40}")
    print(f"  Steps:              {v['steps']}")
    print(f"  Time:               {v['elapsed_seconds']}s")
    print(f"  Threshold (φ²):     {v['threshold']}")
    print(f"  Peak GCI:           {v['peak_gci']}")
    print(f"  Final GCI:          {v['final_gci']}")
    print(f"  Ratio above φ²:     {v['ratio_above_threshold']}×")
    print(f"  Emergence at step:  {v['emergence_step']}")

    print(f"\n  Components:")
    for k, val in v["components"].items():
        print(f"    {k:20s} {val}")

    print(f"\n  {'─' * 40}")
    if verdict["emergence_detected"]:
        print(f"  ✅ EMERGENCE VERIFIED — GCI {verdict['peak_gci']} > φ² {verdict['threshold_phi_sq']}")
    else:
        print(f"  ❌ NO EMERGENCE — GCI {verdict['peak_gci']} < φ² {verdict['threshold_phi_sq']}")

    print(f"\n  SHA-256: {report['proof']['sha256']}")
    print("=" * 64 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Transcendplexity GCI Validator")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps (default: 100)")
    parser.add_argument("--output", type=str, default="gci_proof.json", help="Output JSON file")
    args = parser.parse_args()

    print(f"[GCI Validator] Collecting hardware info...")
    hardware = get_hardware_info()

    print(f"[GCI Validator] Running {args.steps}-step simulation...")
    gci_result = run_gci_validation(steps=args.steps)

    print(f"[GCI Validator] Generating proof report...")
    report = create_proof_report(gci_result, hardware)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[GCI Validator] Saved to {args.output}")

    print_summary(report)
    return report


if __name__ == "__main__":
    main()
