"""Terminal-based real-time live monitor for UnifiedForwardModel metrics.

Run directly::

    python -m monitoring.cli_monitor

Or as a background process::

    python -m monitoring.cli_monitor &
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Any

from monitoring.config import MonitoringConfig
from monitoring.metrics_recorder import MetricsRecorder

# ANSI escape codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_CLEAR_SCREEN = "\033[2J\033[H"


def _color(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def _coherence_color(value: float, config: MonitoringConfig) -> str:
    if value >= config.coherence_green:
        return _GREEN
    if value >= config.coherence_yellow:
        return _YELLOW
    return _RED


def _latency_color(value_ms: float, config: MonitoringConfig) -> str:
    if value_ms < config.latency_green_ms:
        return _GREEN
    if value_ms < config.latency_yellow_ms:
        return _YELLOW
    return _RED


def _sla_status(stats: dict[str, Any], config: MonitoringConfig) -> str:
    current = stats.get("current", {})
    coherence = current.get("coherence", 1.0)
    latency = current.get("latency_ms", 0.0)
    if coherence >= config.coherence_green and latency < config.latency_green_ms:
        return _color("● GREEN  (SLA OK)", _GREEN)
    if coherence >= config.coherence_yellow and latency < config.latency_yellow_ms:
        return _color("● YELLOW (SLA WARNING)", _YELLOW)
    return _color("● RED    (SLA BREACH)", _RED)


def _limb_bar(limbs_active: int, total: int = 8) -> str:
    filled = min(limbs_active, total)
    bar = "█" * filled + "░" * (total - filled)
    return f"[{bar}] {filled}/{total}"


def _trend_arrow(current: float, previous: float) -> str:
    if current > previous + 0.001:
        return "↑"
    if current < previous - 0.001:
        return "↓"
    return "→"


_PANEL_WIDTH = 58


def render_stats(
    stats: dict[str, Any],
    prev_coherence: float,
    config: MonitoringConfig,
    detail_level: int,
) -> str:
    """Render stats as a formatted terminal string."""
    lines = []
    sep = "─" * _PANEL_WIDTH

    lines.append(_color(f"{'═' * _PANEL_WIDTH}", _CYAN))
    lines.append(_color("  🧠  UNIFIED COGNITIVE STACK — LIVE MONITOR", _BOLD))
    lines.append(_color(sep, _CYAN))

    if not stats.get("current"):
        lines.append("  Waiting for inferences…")
        lines.append(_color(f"{'═' * _PANEL_WIDTH}", _CYAN))
        return "\n".join(lines)

    cur = stats["current"]
    all_s = stats.get("all", {})

    # SLA status line
    lines.append(f"  {_sla_status(stats, config)}")
    lines.append(f"  {_color(sep, _CYAN)}")

    # Coherence
    coh = cur.get("coherence", 0.0)
    coh_avg = all_s.get("coherence_mean", coh)
    arrow = _trend_arrow(coh, prev_coherence)
    coh_str = _color(f"{coh:.4f}", _coherence_color(coh, config))
    avg_str = _color(f"{coh_avg:.4f}", _coherence_color(coh_avg, config))
    lines.append(f"  Coherence    : {coh_str}  avg={avg_str}  {arrow}")

    # Latency
    p50 = all_s.get("latency_p50", cur.get("latency_ms", 0.0))
    p99 = all_s.get("latency_p99", 0.0)
    p999 = all_s.get("latency_p999", 0.0)
    lat_color = _latency_color(p50, config)
    lines.append(
        f"  Latency (ms) : p50={_color(f'{p50:.1f}', lat_color)}"
        f"  p99={_color(f'{p99:.1f}', lat_color)}"
        f"  p99.9={_color(f'{p999:.1f}', lat_color)}"
    )

    # Coupling strength
    lines.append(f"  Coupling     : {cur.get('coupling_strength', 0.0):.4f}")

    # Limbs active
    limbs = cur.get("limbs_active", 0)
    lines.append(f"  Limbs Active : {_limb_bar(limbs)}")

    # Action channel
    lines.append(f"  Action Chan  : {cur.get('action_channel', 0)}")

    # Throughput
    total = stats.get("total_inferences", 0)
    rps = stats.get("throughput_rps", 0.0)
    lines.append(f"  Inferences   : {total}  ({rps:.1f} req/s)")

    if detail_level >= 1:
        # Rolling windows
        lines.append(f"  {_color(sep, _CYAN)}")
        for window, ws in stats.get("windows", {}).items():
            if ws:
                lines.append(
                    f"  [{window:5s}] coh={ws.get('coherence_mean', 0):.4f}"
                    f"  lat_p50={ws.get('latency_p50', 0):.1f}ms"
                    f"  n={ws.get('count', 0)}"
                )

    if detail_level >= 2:
        lines.append(f"  {_color(sep, _CYAN)}")
        lines.append(
            f"  Task Signal  : {cur.get('task_signal', '(none)') or '(none)'}"
        )
        lines.append(
            f"  Phase        : {cur.get('phase', 0.0):.4f}"
            f"   Bias={cur.get('bias', 0.0):.4f}"
        )

    lines.append(_color(f"{'═' * 58}", _CYAN))
    lines.append(
        _color("  [q] quit  [r] reset  [s] detail level", _CYAN)
    )
    return "\n".join(lines)


class CLIMonitor:
    """Live terminal monitor that reads from a MetricsRecorder."""

    def __init__(
        self,
        recorder: MetricsRecorder,
        config: MonitoringConfig | None = None,
    ) -> None:
        self._recorder = recorder
        self._config = config or MonitoringConfig()
        self._running = False
        self._detail_level = 0
        self._prev_coherence = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the monitor update loop in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitor update loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        while self._running:
            stats = self._recorder.get_rolling_stats()
            output = render_stats(
                stats, self._prev_coherence, self._config, self._detail_level
            )
            cur = stats.get("current", {})
            self._prev_coherence = cur.get("coherence", self._prev_coherence)

            sys.stdout.write(_CLEAR_SCREEN + output + "\n")
            sys.stdout.flush()
            time.sleep(self._config.cli_update_frequency_sec)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live CLI monitor for UnifiedForwardModel metrics"
    )
    parser.add_argument(
        "--coherence-alert",
        type=float,
        default=0.90,
        metavar="FLOAT",
        help="Coherence threshold for green SLA (default: 0.90)",
    )
    parser.add_argument(
        "--latency-alert",
        type=float,
        default=20.0,
        metavar="MS",
        help="Latency threshold (ms) for green SLA (default: 20)",
    )
    parser.add_argument(
        "--update-freq",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Update frequency in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Circular buffer size (default: 1000)",
    )
    return parser.parse_args(argv)


def _setup_terminal_input() -> "Any":
    """Set terminal to raw mode and return old settings (Unix only)."""
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setraw(fd)
        return old
    except Exception:
        return None


def _restore_terminal(old_settings: "Any") -> None:
    try:
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI monitor."""
    args = _parse_args(argv)

    config = MonitoringConfig(
        cli_update_frequency_sec=args.update_freq,
        cli_coherence_threshold=args.coherence_alert,
        cli_latency_threshold_ms=args.latency_alert,
        coherence_green=args.coherence_alert,
        latency_green_ms=args.latency_alert,
        circular_buffer_size=args.buffer_size,
    )

    # Use a standalone recorder — caller must attach it to a model
    recorder = MetricsRecorder(config=config)
    monitor = CLIMonitor(recorder=recorder, config=config)

    print("Starting CLI monitor (no model attached — waiting for recorder data).")
    print("Press Ctrl-C to exit.")

    monitor.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
