#!/usr/bin/env python3
"""Standalone CLI to generate the OctoTetrahedral cognitive cohesion HTML dashboard."""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

# Ensure project root is on the path regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from model import OctoTetrahedralModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a cognitive cohesion HTML dashboard for the OctoTetrahedral AGI model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file (arc_step_*.pt). If omitted, uses a freshly initialised model.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=10,
        help="Number of random forward passes to run before generating the report (default: 10).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for the HTML report. Defaults to logs/cohesion/cohesion_report.html.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the report in the default browser after generation (macOS: uses 'open').",
    )
    return parser.parse_args()


def load_model(checkpoint: str | None) -> OctoTetrahedralModel:
    model = OctoTetrahedralModel()
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            print(f"[cohesion_dashboard] ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
            sys.exit(1)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # checkpoints may be raw state_dicts or wrapped dicts
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict, strict=False)
        print(f"[cohesion_dashboard] Loaded checkpoint: {ckpt_path}")
    else:
        print("[cohesion_dashboard] No checkpoint provided — using freshly initialised model.")
    model.eval()
    return model


def run_forward_passes(model: OctoTetrahedralModel, n: int) -> None:
    print(f"[cohesion_dashboard] Running {n} forward pass(es) with random input…")
    with torch.no_grad():
        for i in range(n):
            x = torch.randint(0, 1000, (1, 16))
            model(x)
            if (i + 1) % max(1, n // 5) == 0:
                print(f"  pass {i + 1}/{n}")


def main() -> None:
    args = parse_args()

    model = load_model(args.checkpoint)
    run_forward_passes(model, args.passes)

    path = model.export_cohesion_report(args.out)
    score = model.cohesion_score()

    cohesion = score.get("cohesion_score", "n/a")
    limbs_active = score.get("limbs_active", "n/a")
    total_events = score.get("braid_stats", {}).get("total_events", "n/a")

    print(f"\n{'─'*55}")
    print(f"  Report written : {path}")
    print(f"  cohesion_score : {cohesion}")
    print(f"  limbs_active   : {limbs_active}")
    print(f"  total_events   : {total_events}")
    print(f"{'─'*55}")

    if args.open:
        subprocess.run(["open", path], check=False)


if __name__ == "__main__":
    main()
