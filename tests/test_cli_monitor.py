"""Tests for monitoring/cli_monitor.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from monitoring.cli_monitor import (
    CLIMonitor,
    _coherence_color,
    _color,
    _latency_color,
    _limb_bar,
    _sla_status,
    _trend_arrow,
    render_stats,
    _GREEN,
    _YELLOW,
    _RED,
)
from monitoring.config import MonitoringConfig
from monitoring.metrics_recorder import MetricsRecorder


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def test_coherence_color_green():
    config = MonitoringConfig(coherence_green=0.90)
    assert _coherence_color(0.95, config) == _GREEN


def test_coherence_color_yellow():
    config = MonitoringConfig(coherence_green=0.90, coherence_yellow=0.80)
    assert _coherence_color(0.85, config) == _YELLOW


def test_coherence_color_red():
    config = MonitoringConfig(coherence_green=0.90, coherence_yellow=0.80)
    assert _coherence_color(0.70, config) == _RED


def test_latency_color_green():
    config = MonitoringConfig(latency_green_ms=20.0)
    assert _latency_color(15.0, config) == _GREEN


def test_latency_color_yellow():
    config = MonitoringConfig(latency_green_ms=20.0, latency_yellow_ms=50.0)
    assert _latency_color(35.0, config) == _YELLOW


def test_latency_color_red():
    config = MonitoringConfig(latency_green_ms=20.0, latency_yellow_ms=50.0)
    assert _latency_color(60.0, config) == _RED


# ---------------------------------------------------------------------------
# Limb bar
# ---------------------------------------------------------------------------

def test_limb_bar_full():
    bar = _limb_bar(8, 8)
    assert "8/8" in bar
    assert "░" not in bar


def test_limb_bar_empty():
    bar = _limb_bar(0, 8)
    assert "0/8" in bar
    assert "█" not in bar


def test_limb_bar_partial():
    bar = _limb_bar(4, 8)
    assert "4/8" in bar
    assert "█" in bar
    assert "░" in bar


# ---------------------------------------------------------------------------
# Trend arrow
# ---------------------------------------------------------------------------

def test_trend_arrow_up():
    assert _trend_arrow(0.95, 0.90) == "↑"


def test_trend_arrow_down():
    assert _trend_arrow(0.80, 0.90) == "↓"


def test_trend_arrow_stable():
    assert _trend_arrow(0.90, 0.90) == "→"


# ---------------------------------------------------------------------------
# SLA status
# ---------------------------------------------------------------------------

def _stats_with(coherence: float, latency_ms: float) -> dict:
    return {"current": {"coherence": coherence, "latency_ms": latency_ms}}


def test_sla_green():
    config = MonitoringConfig(coherence_green=0.90, latency_green_ms=20.0)
    status = _sla_status(_stats_with(0.95, 15.0), config)
    assert "GREEN" in status


def test_sla_yellow():
    config = MonitoringConfig(
        coherence_green=0.90, coherence_yellow=0.80,
        latency_green_ms=20.0, latency_yellow_ms=50.0
    )
    status = _sla_status(_stats_with(0.85, 30.0), config)
    assert "YELLOW" in status


def test_sla_red():
    config = MonitoringConfig(
        coherence_green=0.90, coherence_yellow=0.80,
        latency_green_ms=20.0, latency_yellow_ms=50.0
    )
    status = _sla_status(_stats_with(0.70, 60.0), config)
    assert "RED" in status


# ---------------------------------------------------------------------------
# render_stats output
# ---------------------------------------------------------------------------

def _mock_stats(
    coherence: float = 0.95,
    latency_ms: float = 10.0,
    total: int = 42,
) -> dict:
    return {
        "total_inferences": total,
        "throughput_rps": 5.0,
        "current": {
            "coherence": coherence,
            "latency_ms": latency_ms,
            "coupling_strength": 0.7,
            "phase": 0.3,
            "bias": 0.1,
            "limbs_active": 5,
            "action_channel": 3,
            "task_signal": "reasoning",
        },
        "all": {
            "latency_p50": latency_ms,
            "latency_p99": latency_ms * 1.5,
            "latency_p999": latency_ms * 2,
            "coherence_mean": coherence,
        },
        "windows": {
            "1min": {"coherence_mean": coherence, "latency_p50": latency_ms, "count": 5},
            "5min": {},
            "15min": {},
        },
    }


def test_render_stats_contains_coherence():
    config = MonitoringConfig()
    output = render_stats(_mock_stats(), 0.90, config, detail_level=0)
    assert "Coherence" in output
    assert "0.9500" in output


def test_render_stats_contains_latency():
    config = MonitoringConfig()
    output = render_stats(_mock_stats(latency_ms=15.0), 0.90, config, detail_level=0)
    assert "Latency" in output
    assert "15.0" in output


def test_render_stats_contains_limb_bar():
    config = MonitoringConfig()
    output = render_stats(_mock_stats(), 0.90, config, detail_level=0)
    assert "5/8" in output


def test_render_stats_detail_level_1_shows_windows():
    config = MonitoringConfig()
    output = render_stats(_mock_stats(), 0.90, config, detail_level=1)
    assert "1min" in output


def test_render_stats_detail_level_2_shows_task():
    config = MonitoringConfig()
    output = render_stats(_mock_stats(), 0.90, config, detail_level=2)
    assert "reasoning" in output
    assert "Phase" in output


def test_render_stats_empty_shows_waiting():
    config = MonitoringConfig()
    empty_stats = {"total_inferences": 0, "current": {}, "all": {}, "windows": {}}
    output = render_stats(empty_stats, 0.0, config, detail_level=0)
    assert "Waiting" in output


# ---------------------------------------------------------------------------
# CLIMonitor lifecycle
# ---------------------------------------------------------------------------

def test_cli_monitor_starts_and_stops():
    recorder = MagicMock(spec=MetricsRecorder)
    recorder.get_rolling_stats.return_value = {
        "total_inferences": 0, "current": {}, "all": {}, "windows": {}
    }
    config = MonitoringConfig(cli_update_frequency_sec=0.1)
    monitor = CLIMonitor(recorder=recorder, config=config)
    monitor.start()
    assert monitor._running is True
    monitor.stop()
    assert monitor._running is False
