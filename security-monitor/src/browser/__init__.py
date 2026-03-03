"""
QuantumShield Sandboxed Browser — Trident Architecture
=====================================================
Three identical isolated browser instances running in parallel
with failover, consensus verification, and malware scanning.

Each instance runs in its own sandbox:
- Isolated temp filesystem (destroyed on exit)
- No persistent cookies/state
- Content scanned before rendering
- Resource-limited subprocess execution

If any instance detects malware or becomes compromised,
it is killed immediately while the others continue serving.
"""

from .trident import TridentBrowser
from .sandbox import SandboxedInstance
from .scanner import ThreatScanner

__all__ = ["TridentBrowser", "SandboxedInstance", "ThreatScanner"]
__version__ = "1.0.0"
