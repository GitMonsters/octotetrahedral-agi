#!/usr/bin/env python3
"""
SIEM-grade audit logging for the Trident browser.

Every action is logged as structured JSON with:
- Timestamp (ISO 8601 UTC)
- Event type (fetch, scan, block, quarantine, failover, etc.)
- Instance ID
- Full request/response metadata
- Threat details
- Exportable to Splunk, ELK, or any SIEM

Compliant with:
- NIST SP 800-92 (Log Management)
- NIST SP 800-53 AU controls (Audit and Accountability)
"""

import json
import os
import time
import uuid
import hashlib
import logging
import threading
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("browser.audit")


class AuditEventType(Enum):
    # Browser lifecycle
    BROWSER_START = "browser.start"
    BROWSER_STOP = "browser.stop"
    INSTANCE_SPAWN = "instance.spawn"
    INSTANCE_DESTROY = "instance.destroy"
    INSTANCE_QUARANTINE = "instance.quarantine"
    INSTANCE_RESPAWN = "instance.respawn"

    # Fetch operations
    FETCH_START = "fetch.start"
    FETCH_COMPLETE = "fetch.complete"
    FETCH_ERROR = "fetch.error"
    FETCH_TIMEOUT = "fetch.timeout"

    # Security events
    THREAT_DETECTED = "security.threat_detected"
    THREAT_BLOCKED = "security.threat_blocked"
    CONSENSUS_FAILURE = "security.consensus_failure"
    DNS_BLOCKED = "security.dns_blocked"
    CERT_FAILURE = "security.cert_failure"
    CERT_PIN_MISMATCH = "security.cert_pin_mismatch"
    HSTS_VIOLATION = "security.hsts_violation"
    TLS_DOWNGRADE = "security.tls_downgrade"

    # Navigation
    NAVIGATE = "nav.navigate"
    NAVIGATE_BACK = "nav.back"
    NAVIGATE_FORWARD = "nav.forward"
    LINK_FOLLOW = "nav.link_follow"


@dataclass
class AuditEvent:
    event_id: str
    timestamp: str
    event_type: str
    instance_id: Optional[int] = None
    url: Optional[str] = None
    status_code: Optional[int] = None
    threat_level: Optional[str] = None
    threat_count: int = 0
    blocked: bool = False
    content_hash: Optional[str] = None
    response_time_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        # Remove None values for cleaner logs
        d = {k: v for k, v in d.items() if v is not None}
        return json.dumps(d, ensure_ascii=False)

    def to_syslog(self) -> str:
        """Format as syslog-compatible CEF (Common Event Format)."""
        severity = 0
        if self.threat_level == "HIGH":
            severity = 7
        elif self.threat_level == "CRITICAL":
            severity = 10
        elif self.blocked:
            severity = 5

        return (
            f"CEF:0|QuantumShield|TridentBrowser|1.0|{self.event_type}|"
            f"{self.event_type}|{severity}|"
            f"src={self.url or ''} "
            f"outcome={'blocked' if self.blocked else 'allowed'} "
            f"rt={self.timestamp}"
        )


class AuditLogger:
    """SIEM-grade structured audit logger."""

    def __init__(self, log_dir: str, session_id: Optional[str] = None,
                 max_file_size: int = 50 * 1024 * 1024) -> None:
        self.log_dir = log_dir
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.max_file_size = max_file_size
        self._lock = threading.Lock()
        self._events: List[AuditEvent] = []
        self._file_idx = 0

        os.makedirs(log_dir, exist_ok=True)
        self._current_file = self._open_log_file()

        # Log session start
        self.log(AuditEventType.BROWSER_START, details={
            "session_id": self.session_id,
            "log_dir": log_dir,
        })

    def _open_log_file(self) -> str:
        """Open a new audit log file."""
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        fname = f"audit_{ts}_{self._file_idx:04d}.jsonl"
        return os.path.join(self.log_dir, fname)

    def log(self, event_type: AuditEventType,
            instance_id: Optional[int] = None,
            url: Optional[str] = None,
            status_code: Optional[int] = None,
            threat_level: Optional[str] = None,
            threat_count: int = 0,
            blocked: bool = False,
            content_hash: Optional[str] = None,
            response_time_ms: Optional[float] = None,
            details: Optional[Dict[str, Any]] = None) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            event_id=uuid.uuid4().hex[:16],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            event_type=event_type.value,
            instance_id=instance_id,
            url=url,
            status_code=status_code,
            threat_level=threat_level,
            threat_count=threat_count,
            blocked=blocked,
            content_hash=content_hash,
            response_time_ms=response_time_ms,
            details=details or {},
            session_id=self.session_id,
        )

        with self._lock:
            self._events.append(event)
            self._write_event(event)

        # Also log high-severity events to Python logger
        if blocked or (threat_level and threat_level in ("HIGH", "CRITICAL")):
            logger.warning(f"AUDIT [{event.event_type}] {event.url} — {threat_level} blocked={blocked}")

        return event

    def _write_event(self, event: AuditEvent) -> None:
        """Write event to JSONL audit file."""
        try:
            # Rotate if needed
            if os.path.exists(self._current_file):
                if os.path.getsize(self._current_file) > self.max_file_size:
                    self._file_idx += 1
                    self._current_file = self._open_log_file()

            with open(self._current_file, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def log_fetch(self, instance_id: int, url: str, result) -> None:
        """Convenience: log a complete fetch with threat data."""
        if result.error:
            self.log(
                AuditEventType.FETCH_ERROR,
                instance_id=instance_id,
                url=url,
                details={"error": result.error},
            )
        else:
            threat_level = None
            threat_count = 0
            blocked = False
            content_hash = None

            if result.threat_report:
                threat_level = result.threat_report.level.name
                threat_count = len(result.threat_report.threats)
                blocked = result.threat_report.blocked
                content_hash = result.threat_report.content_hash

            self.log(
                AuditEventType.FETCH_COMPLETE,
                instance_id=instance_id,
                url=url,
                status_code=result.status_code,
                threat_level=threat_level,
                threat_count=threat_count,
                blocked=blocked,
                content_hash=content_hash,
                response_time_ms=result.fetch_time_ms,
            )

            if blocked:
                self.log(
                    AuditEventType.THREAT_BLOCKED,
                    instance_id=instance_id,
                    url=url,
                    threat_level=threat_level,
                    threat_count=threat_count,
                    blocked=True,
                    details={
                        "threats": result.threat_report.threats if result.threat_report else [],
                    },
                )

    def log_consensus_failure(self, url: str, details: Dict) -> None:
        """Log content divergence between instances."""
        self.log(
            AuditEventType.CONSENSUS_FAILURE,
            url=url,
            threat_level="HIGH",
            blocked=False,
            details=details,
        )

    def get_events(self, event_type: Optional[str] = None,
                   since: Optional[str] = None,
                   limit: int = 100) -> List[AuditEvent]:
        """Query logged events."""
        events = self._events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events[-limit:]

    def get_security_summary(self) -> Dict:
        """Get summary of security events for this session."""
        threats = [e for e in self._events if e.threat_level in ("HIGH", "CRITICAL")]
        blocks = [e for e in self._events if e.blocked]
        fetches = [e for e in self._events if e.event_type == "fetch.complete"]
        consensus = [e for e in self._events if e.event_type == "security.consensus_failure"]

        return {
            "session_id": self.session_id,
            "total_events": len(self._events),
            "total_fetches": len(fetches),
            "threats_detected": len(threats),
            "pages_blocked": len(blocks),
            "consensus_failures": len(consensus),
            "unique_urls": len(set(e.url for e in fetches if e.url)),
            "log_file": self._current_file,
        }

    def export_cef(self, output_path: str) -> int:
        """Export all events in CEF format for SIEM ingestion."""
        count = 0
        with open(output_path, "w") as f:
            for event in self._events:
                f.write(event.to_syslog() + "\n")
                count += 1
        return count

    def shutdown(self) -> None:
        """Log session end and flush."""
        self.log(AuditEventType.BROWSER_STOP, details={
            "summary": self.get_security_summary(),
        })
