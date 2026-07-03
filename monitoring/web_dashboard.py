#!/usr/bin/env python3
"""Real-time web dashboard for the unified 8-limb forward model.

Run with:

    python -m monitoring.web_dashboard

Serves a small auto-refreshing HTML page (polling `/api/metrics` once a
second) plus the JSON endpoint itself, backed by the same `MetricsRecorder`
used by `monitoring.cli_monitor`. A background thread drives synthetic (but
real) inference traffic through `UnifiedForwardModel` so the dashboard has
live data to display.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on the path regardless of cwd, matching tools/cohesion_dashboard.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, jsonify, render_template_string  # noqa: E402

from monitoring.cli_monitor import generate_demo_limb_states  # noqa: E402
from monitoring.metrics_recorder import MetricsRecorder  # noqa: E402
from unified.forward_model import UnifiedForwardModel  # noqa: E402

_PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>unified-stack real-time analytics</title>
  <style>
    body { background: #0d1117; color: #c9d1d9; font-family: ui-monospace, Menlo, monospace; padding: 2rem; }
    h1 { font-size: 1.1rem; font-weight: 600; color: #58a6ff; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; }
    .label { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; }
    .value { font-size: 1.5rem; margin-top: 0.25rem; }
    canvas { background: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-top: 1rem; }
  </style>
</head>
<body>
  <h1>Model: unified-stack ({{ limb_count }} limbs)</h1>
  <div class="grid" id="metrics-grid"></div>
  <canvas id="coherence-chart" width="600" height="120"></canvas>

  <script>
    const REFRESH_MS = {{ refresh_ms }};

    function fmt(value, digits) {
      return (value === null || value === undefined) ? "–" : Number(value).toFixed(digits);
    }

    function renderCards(snapshot) {
      const grid = document.getElementById("metrics-grid");
      const cards = [
        ["Coherence", fmt(snapshot.coherence_latest, 3) + " (avg " + fmt(snapshot.coherence_mean, 3) + ")"],
        ["Latency", fmt(snapshot.latency_latest_ms, 1) + "ms (p99 " + fmt(snapshot.latency_p99_ms, 1) + "ms)"],
        ["Coupling", fmt(snapshot.coupling_latest, 3)],
        ["Limbs Active", (snapshot.limbs_active ?? "–") + "/" + (snapshot.limb_count ?? "–")],
        ["Action Channel", (snapshot.action_channel ?? "–") + " (" + (snapshot.task_signal ?? "–") + ")"],
        ["Requests", snapshot.total_requests.toLocaleString() + " total, " + fmt(snapshot.requests_per_second, 1) + " req/s"],
      ];
      grid.innerHTML = cards.map(([label, value]) =>
        `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`
      ).join("");
    }

    function renderChart(history) {
      const canvas = document.getElementById("coherence-chart");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (history.length < 2) return;

      ctx.beginPath();
      ctx.strokeStyle = "#58a6ff";
      ctx.lineWidth = 2;
      history.forEach((sample, index) => {
        const x = (index / (history.length - 1)) * canvas.width;
        const y = canvas.height - (sample.coherence * canvas.height);
        index === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    async function poll() {
      try {
        const response = await fetch("/api/metrics");
        const data = await response.json();
        renderCards(data.snapshot);
        renderChart(data.history);
      } catch (err) {
        console.error("metrics poll failed", err);
      }
    }

    poll();
    setInterval(poll, REFRESH_MS);
  </script>
</body>
</html>
"""


def create_app(recorder: MetricsRecorder | None = None, limb_count: int = 8, refresh_ms: int = 1000) -> Flask:
    """Build the Flask app. Injectable `recorder` makes this test-friendly."""
    app = Flask(__name__)
    app.config["METRICS_RECORDER"] = recorder or MetricsRecorder()
    app.config["LIMB_COUNT"] = limb_count
    app.config["REFRESH_MS"] = refresh_ms

    @app.get("/")
    def index():
        return render_template_string(
            _PAGE_TEMPLATE,
            limb_count=app.config["LIMB_COUNT"],
            refresh_ms=app.config["REFRESH_MS"],
        )

    @app.get("/api/metrics")
    def metrics():
        current_recorder: MetricsRecorder = app.config["METRICS_RECORDER"]
        return jsonify(
            {
                "snapshot": current_recorder.snapshot(),
                "history": current_recorder.history(),
            }
        )

    return app


def _run_demo_traffic(
    recorder: MetricsRecorder,
    model: UnifiedForwardModel,
    task_signals: list[str],
    interval: float,
    seed: int | None,
    stop_event: threading.Event,
) -> None:
    rng = random.Random(seed)
    instrumented = recorder.instrument(model)
    tick = 0
    while not stop_event.is_set():
        task_signal = task_signals[tick % len(task_signals)]
        instrumented.forward(generate_demo_limb_states(rng, model.limb_count), task_signal=task_signal)
        tick += 1
        stop_event.wait(interval)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time web dashboard for the unified-stack (8 limb) forward model."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between synthetic inferences (default: 1.0).")
    parser.add_argument("--window", type=int, default=100, help="Rolling window size in samples (default: 100).")
    parser.add_argument(
        "--task-signals",
        type=str,
        default="reasoning,language,spatial",
        help="Comma-separated task signals to cycle through (default: reasoning,language,spatial).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for synthetic limb states.")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug/reload mode.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task_signals = [signal.strip() for signal in args.task_signals.split(",") if signal.strip()] or ["default"]

    model = UnifiedForwardModel()
    recorder = MetricsRecorder(window_size=args.window)
    stop_event = threading.Event()

    traffic_thread = threading.Thread(
        target=_run_demo_traffic,
        args=(recorder, model, task_signals, args.interval, args.seed, stop_event),
        daemon=True,
    )
    traffic_thread.start()

    app = create_app(recorder=recorder, limb_count=model.limb_count, refresh_ms=int(args.interval * 1000))
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
