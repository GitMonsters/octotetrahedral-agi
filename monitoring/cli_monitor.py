#!/usr/bin/env python3
"""Real-time terminal monitor for the unified 8-limb forward model.

Run with:

    python -m monitoring.cli_monitor

Drives synthetic (but real) inference traffic through `UnifiedForwardModel`,
records it with `MetricsRecorder`, and redraws a live-updating summary panel
every `--interval` seconds until interrupted with Ctrl+C.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Ensure project root is on the path regardless of cwd, matching tools/cohesion_dashboard.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from monitoring.metrics_recorder import MetricsRecorder  # noqa: E402
from unified.forward_model import UnifiedForwardModel  # noqa: E402

DIVIDER = "─" * 48


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live terminal monitor for the unified-stack (8 limb) forward model."
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between refreshes (default: 1.0).")
    parser.add_argument("--window", type=int, default=100, help="Rolling window size in samples (default: 100).")
    parser.add_argument(
        "--task-signals",
        type=str,
        default="reasoning,language,spatial",
        help="Comma-separated task signals to cycle through (default: reasoning,language,spatial).",
    )
    parser.add_argument("--duration", type=float, default=None, help="Stop after this many seconds (default: run until Ctrl+C).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for synthetic limb states.")
    return parser.parse_args(argv)


def _trend_arrow(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return "→"
    if current > previous:
        return "↑"
    if current < previous:
        return "↓"
    return "→"


def render_snapshot(snapshot: dict, limb_count: int = 8) -> str:
    """Render a metrics snapshot as the multi-line monitor panel."""
    header = f"Model: unified-stack ({limb_count} limbs)"

    if snapshot["sample_count"] == 0:
        return "\n".join([DIVIDER, header, "Waiting for inference samples...", DIVIDER])

    arrow = _trend_arrow(snapshot["coherence_latest"], snapshot["coherence_previous"])
    lines = [
        DIVIDER,
        header,
        f"Coherence:    {snapshot['coherence_latest']:.3f} {arrow} (avg: {snapshot['coherence_mean']:.3f})",
        f"Latency:      {snapshot['latency_latest_ms']:.1f}ms (p99: {snapshot['latency_p99_ms']:.1f}ms)",
        f"Coupling:     {snapshot['coupling_latest']:.3f}",
        f"Limbs Active: {snapshot['limbs_active']}/{snapshot['limb_count']}",
        f"Action Ch:    {snapshot['action_channel']} ({snapshot['task_signal']})",
        f"Requests:     {snapshot['total_requests']:,} total | {snapshot['requests_per_second']:.1f} req/s",
        DIVIDER,
    ]
    return "\n".join(lines)


def generate_demo_limb_states(rng: random.Random, limb_count: int) -> list[float]:
    """Produce synthetic (but plausible) limb state input for demo traffic."""
    return [rng.uniform(0.0, 1.0) for _ in range(limb_count)]


def run(
    interval: float,
    window_size: int,
    task_signals: list[str],
    duration: float | None,
    seed: int | None,
    clear_screen: bool = True,
) -> None:
    model = UnifiedForwardModel()
    recorder = MetricsRecorder(window_size=window_size)
    instrumented = recorder.instrument(model)
    rng = random.Random(seed)

    start = time.monotonic()
    tick = 0
    try:
        while duration is None or (time.monotonic() - start) < duration:
            task_signal = task_signals[tick % len(task_signals)]
            instrumented.forward(generate_demo_limb_states(rng, model.limb_count), task_signal=task_signal)

            if clear_screen:
                print("\033c", end="")
            print(render_snapshot(recorder.snapshot(), limb_count=model.limb_count))

            tick += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task_signals = [signal.strip() for signal in args.task_signals.split(",") if signal.strip()]
    run(
        interval=args.interval,
        window_size=args.window,
        task_signals=task_signals or ["default"],
        duration=args.duration,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
