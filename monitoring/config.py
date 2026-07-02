"""Centralized configuration for all monitoring components."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MonitoringConfig:
    # Recorder
    circular_buffer_size: int = 1000
    enable_per_inference_logging: bool = False

    # CLI Monitor
    cli_update_frequency_sec: float = 1.0
    cli_coherence_threshold: float = 0.90
    cli_latency_threshold_ms: float = 20.0

    # Web Dashboard
    web_port: int = 8000
    web_history_minutes: int = 10
    web_update_frequency_sec: float = 1.0

    # Prometheus
    prometheus_enabled: bool = True
    prometheus_port: int = 8001

    # SLA thresholds
    coherence_green: float = 0.90
    coherence_yellow: float = 0.80
    latency_green_ms: float = 20.0
    latency_yellow_ms: float = 50.0
