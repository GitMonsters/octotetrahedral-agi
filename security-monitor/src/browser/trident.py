#!/usr/bin/env python3
"""
TridentBrowser — Three parallel sandboxed browser instances with failover.

Architecture:
  ┌─────────────────────────────────────────────┐
  │              Trident Orchestrator            │
  │                                              │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
  │  │ Instance  │ │ Instance  │ │ Instance  │    │
  │  │    α      │ │    β      │ │    γ      │    │
  │  │ (primary) │ │ (mirror)  │ │ (mirror)  │    │
  │  │  sandbox  │ │  sandbox  │ │  sandbox  │    │
  │  └──────────┘ └──────────┘ └──────────┘    │
  │       │            │            │            │
  │       ▼            ▼            ▼            │
  │  ┌─────────────────────────────────────┐    │
  │  │     Consensus + Threat Voting       │    │
  │  └─────────────────────────────────────┘    │
  └─────────────────────────────────────────────┘

Consensus:
- All 3 instances fetch the same URL independently
- Content hashes compared — if one differs, it's quarantined
- If 2+ threat scans agree content is dangerous → block
- If one instance dies, the remaining 2 continue with failover
- Dead instances are automatically respawned
"""

import os
import time
import logging
import threading
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from .sandbox import SandboxedInstance, FetchResult, InstanceState
from .scanner import ThreatScanner, ThreatReport, ThreatLevel
from .audit_log import AuditLogger, AuditEventType
from .yara_scanner import YaraScanner
from .secure_memory import SecureMemory, EncryptedStateStore

logger = logging.getLogger("browser.trident")


class TridentBrowser:
    """Three-instance parallel browser with consensus and failover."""

    INSTANCE_NAMES = ["α", "β", "γ"]
    MIN_ALIVE = 1  # Minimum instances needed to continue operating

    def __init__(self, verbose: bool = False,
                 audit_dir: Optional[str] = None) -> None:
        self.verbose = verbose
        self.scanner = ThreatScanner()
        self.yara_scanner = YaraScanner()
        self.instances: List[SandboxedInstance] = []
        self.history: List[str] = []
        self.history_pos: int = -1
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=3)
        self.total_fetches = 0
        self.total_blocks = 0
        self.consensus_failures = 0

        # Gov-level: SIEM audit logging
        import tempfile
        log_dir = audit_dir or os.path.join(tempfile.gettempdir(), "trident_audit")
        self.audit = AuditLogger(log_dir)

        # Gov-level: encrypted state store
        self.secure_mem = SecureMemory()
        self.state_store = EncryptedStateStore()

        # Spawn 3 instances
        for i in range(3):
            inst = SandboxedInstance(instance_id=i, scanner=self.scanner)
            self.instances.append(inst)
            self.audit.log(AuditEventType.INSTANCE_SPAWN, instance_id=i,
                          details={"sandbox_dir": inst.sandbox_dir})
            if verbose:
                print(f"  🔧 Instance {self.INSTANCE_NAMES[i]} spawned → {inst.sandbox_dir}")

    def fetch(self, url: str) -> FetchResult:
        """
        Fetch URL through all 3 instances in parallel.
        Uses consensus to verify content integrity.
        Returns the verified result.
        """
        alive = [inst for inst in self.instances if inst.is_alive()]
        if len(alive) < self.MIN_ALIVE:
            self._respawn_dead()
            alive = [inst for inst in self.instances if inst.is_alive()]
            if not alive:
                result = FetchResult(url=url)
                result.error = "All instances dead — cannot fetch"
                return result

        # Fetch in parallel across all alive instances
        futures: Dict[Future, SandboxedInstance] = {}
        for inst in alive:
            f = self._pool.submit(inst.fetch, url)
            futures[f] = inst

        results: List[FetchResult] = []
        instance_results: Dict[int, FetchResult] = {}

        for f in as_completed(futures, timeout=30):
            inst = futures[f]
            try:
                result = f.result(timeout=20)
                results.append(result)
                instance_results[inst.instance_id] = result
            except Exception as e:
                logger.error(f"Instance {self.INSTANCE_NAMES[inst.instance_id]} failed: {e}")
                inst.quarantine()

        if not results:
            r = FetchResult(url=url)
            r.error = "All instances failed to fetch"
            return r

        # Consensus: compare content hashes
        verified = self._consensus_check(results, instance_results)

        # Update navigation history
        if verified.ok:
            if self.history_pos < len(self.history) - 1:
                self.history = self.history[:self.history_pos + 1]
            self.history.append(url)
            self.history_pos = len(self.history) - 1

        self.total_fetches += 1
        if verified.threat_report and verified.threat_report.blocked:
            self.total_blocks += 1

        # Gov-level: YARA deep scan on verified content
        if verified.ok and verified.raw_html:
            yara_matches = self.yara_scanner.scan_content(verified.raw_html, url)
            if yara_matches:
                yara_detail = [{"rule": m.rule, "severity": m.severity} for m in yara_matches]
                self.audit.log(AuditEventType.THREAT_DETECTED, url=url,
                              threat_level=yara_matches[0].severity.upper(),
                              threat_count=len(yara_matches),
                              details={"yara_matches": yara_detail})
                critical = any(m.severity == "critical" for m in yara_matches)
                if critical and verified.threat_report:
                    verified.threat_report.blocked = True
                    verified.threat_report.threats.extend(
                        [f"YARA:{m.rule}" for m in yara_matches]
                    )
                    self.total_blocks += 1

        # Gov-level: audit log the fetch
        self.audit.log(
            AuditEventType.FETCH_COMPLETE, url=url,
            threat_level=verified.threat_report.level.name if verified.threat_report else None,
            threat_count=len(verified.threat_report.threats) if verified.threat_report else 0,
            blocked=verified.threat_report.blocked if verified.threat_report else False,
            content_hash=verified.threat_report.content_hash if verified.threat_report else None,
        )

        return verified

    def _consensus_check(self, results: List[FetchResult],
                         instance_results: Dict[int, FetchResult]) -> FetchResult:
        """Compare results across instances for integrity verification."""
        # Separate successful vs failed
        ok_results = [r for r in results if r.ok and r.threat_report]
        if not ok_results:
            # All failed — return best error
            return results[0]

        # Check content hashes
        hashes = {}
        for r in ok_results:
            h = r.threat_report.content_hash if r.threat_report else ""
            hashes.setdefault(h, []).append(r)

        if len(hashes) > 1:
            # Content divergence detected — find the majority
            self.consensus_failures += 1
            majority_hash = max(hashes, key=lambda h: len(hashes[h]))
            minority_results = {r for h, rs in hashes.items() if h != majority_hash for r in rs}

            # Gov-level: audit consensus failure
            self.audit.log_consensus_failure(
                url=ok_results[0].url if ok_results else "",
                details={"hash_count": len(hashes), "majority": majority_hash[:16]},
            )

            # Quarantine instances that returned different content
            for inst_id, result in instance_results.items():
                if result in minority_results:
                    logger.warning(
                        f"Instance {self.INSTANCE_NAMES[inst_id]} returned divergent content — QUARANTINED"
                    )
                    self.instances[inst_id].quarantine()
                    self.audit.log(AuditEventType.INSTANCE_QUARANTINE, instance_id=inst_id,
                                  url=result.url)
                    self._respawn_instance(inst_id)

            ok_results = hashes[majority_hash]

        # Threat voting: if majority says blocked → block
        block_votes = sum(1 for r in ok_results if r.threat_report and r.threat_report.blocked)
        if block_votes > len(ok_results) / 2:
            # Majority says block
            blocked = next(r for r in ok_results if r.threat_report and r.threat_report.blocked)
            return blocked

        # Return the first clean result
        clean = [r for r in ok_results if r.threat_report and not r.threat_report.blocked]
        return clean[0] if clean else ok_results[0]

    def _respawn_dead(self) -> None:
        """Respawn any dead/quarantined instances."""
        for i, inst in enumerate(self.instances):
            if not inst.is_alive():
                self._respawn_instance(i)

    def _respawn_instance(self, idx: int) -> None:
        """Destroy and recreate a specific instance."""
        old = self.instances[idx]
        old.destroy()
        new_inst = SandboxedInstance(instance_id=idx, scanner=self.scanner)
        self.instances[idx] = new_inst
        if self.verbose:
            print(f"  🔄 Instance {self.INSTANCE_NAMES[idx]} respawned → {new_inst.sandbox_dir}")

    def go_back(self) -> Optional[str]:
        """Navigate back in history."""
        if self.history_pos > 0:
            self.history_pos -= 1
            return self.history[self.history_pos]
        return None

    def go_forward(self) -> Optional[str]:
        """Navigate forward in history."""
        if self.history_pos < len(self.history) - 1:
            self.history_pos += 1
            return self.history[self.history_pos]
        return None

    def status(self) -> Dict:
        """Full trident status."""
        return {
            "instances": [inst.status() for inst in self.instances],
            "alive": sum(1 for inst in self.instances if inst.is_alive()),
            "total_fetches": self.total_fetches,
            "total_blocks": self.total_blocks,
            "consensus_failures": self.consensus_failures,
            "history_depth": len(self.history),
            "audit": self.audit.get_security_summary(),
            "yara": self.yara_scanner.stats(),
            "state_store": self.state_store.stats(),
        }

    def shutdown(self) -> None:
        """Gracefully destroy all instances."""
        for i, inst in enumerate(self.instances):
            inst.destroy()
            self.audit.log(AuditEventType.INSTANCE_DESTROY, instance_id=i)
        self._pool.shutdown(wait=False)
        self.state_store.destroy()
        self.audit.shutdown()
        logger.info("Trident browser shutdown complete")
