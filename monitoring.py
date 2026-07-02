"""Real-time coherence monitoring for the unified cognitive stack."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from typing import TypedDict

import production_config as cfg

logger = logging.getLogger(__name__)

# Maximum number of recent observations kept in memory per metric
_WINDOW = 1000


class _InferenceRecord(TypedDict):
    request_id: str
    coherence: float
    action_channel: int
    limb_states: list[float]
    latency_ms: float
    timestamp: float


class CoherenceAlert(TypedDict):
    request_id: str
    coherence: float
    threshold: float
    timestamp: float


class MonitoringStats(TypedDict):
    total_inferences: int
    mean_coherence: float
    min_coherence: float
    alert_count: int
    limb_activation_histogram: list[float]
    action_channel_distribution: dict[str, int]
    mean_latency_ms: float
    p99_latency_ms: float


class InferenceMonitor:
    """Tracks coherence, limb utilization, action channels, and timing."""

    def __init__(
        self,
        coherence_threshold: float = cfg.COHERENCE_ALERT_THRESHOLD,
        latency_warn_ms: float = cfg.LATENCY_WARN_MS,
        limb_count: int = cfg.MODEL_LIMB_COUNT,
    ) -> None:
        self.coherence_threshold = coherence_threshold
        self.latency_warn_ms = latency_warn_ms
        self.limb_count = limb_count

        self._records: deque[_InferenceRecord] = deque(maxlen=_WINDOW)
        self._alerts: deque[CoherenceAlert] = deque(maxlen=_WINDOW)
        self._action_channel_counts: dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        request_id: str,
        coherence: float,
        action_channel: int,
        limb_states: list[float],
        latency_ms: float,
    ) -> list[CoherenceAlert]:
        """Record one inference and return any new alerts raised."""
        record: _InferenceRecord = {
            "request_id": request_id,
            "coherence": coherence,
            "action_channel": action_channel,
            "limb_states": limb_states,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        }
        self._records.append(record)
        self._action_channel_counts[action_channel] += 1

        new_alerts: list[CoherenceAlert] = []

        if coherence < self.coherence_threshold:
            alert: CoherenceAlert = {
                "request_id": request_id,
                "coherence": coherence,
                "threshold": self.coherence_threshold,
                "timestamp": record["timestamp"],
            }
            self._alerts.append(alert)
            new_alerts.append(alert)
            logger.warning(
                json.dumps(
                    {
                        "event": "coherence_alert",
                        "request_id": request_id,
                        "coherence": coherence,
                        "threshold": self.coherence_threshold,
                    }
                )
            )

        if latency_ms > self.latency_warn_ms:
            logger.warning(
                json.dumps(
                    {
                        "event": "latency_warning",
                        "request_id": request_id,
                        "latency_ms": latency_ms,
                        "threshold_ms": self.latency_warn_ms,
                    }
                )
            )

        return new_alerts

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> MonitoringStats:
        """Return aggregated monitoring statistics over the current window."""
        records = list(self._records)
        n = len(records)

        if n == 0:
            return {
                "total_inferences": 0,
                "mean_coherence": 0.0,
                "min_coherence": 0.0,
                "alert_count": 0,
                "limb_activation_histogram": [0.0] * self.limb_count,
                "action_channel_distribution": {},
                "mean_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
            }

        coherences = [r["coherence"] for r in records]
        latencies = sorted(r["latency_ms"] for r in records)

        # Limb activation histogram: mean activation per limb across window
        limb_sums = [0.0] * self.limb_count
        for r in records:
            if len(r["limb_states"]) > self.limb_count:
                logger.warning(
                    json.dumps(
                        {
                            "event": "limb_states_truncated",
                            "request_id": r["request_id"],
                            "received": len(r["limb_states"]),
                            "expected": self.limb_count,
                        }
                    )
                )
            for i, v in enumerate(r["limb_states"][: self.limb_count]):
                limb_sums[i] += v
        limb_histogram = [s / n for s in limb_sums]

        p99_index = max(0, int(0.99 * n) - 1)

        return {
            "total_inferences": n,
            "mean_coherence": sum(coherences) / n,
            "min_coherence": min(coherences),
            "alert_count": len(self._alerts),
            "limb_activation_histogram": limb_histogram,
            "action_channel_distribution": {
                str(ch): cnt for ch, cnt in sorted(self._action_channel_counts.items())
            },
            "mean_latency_ms": sum(latencies) / n,
            "p99_latency_ms": latencies[p99_index],
        }

    def recent_alerts(self, limit: int = 10) -> list[CoherenceAlert]:
        """Return the most recent coherence alerts."""
        alerts = list(self._alerts)
        return alerts[-limit:]

    def reset(self) -> None:
        """Clear all accumulated records and alerts (e.g. between test runs)."""
        self._records.clear()
        self._alerts.clear()
        self._action_channel_counts.clear()
