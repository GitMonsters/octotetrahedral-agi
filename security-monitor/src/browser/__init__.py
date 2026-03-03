"""
QuantumShield Sandboxed Browser — Trident Architecture (Gov-Level)
==================================================================
Three identical isolated browser instances running in parallel
with failover, consensus verification, and multi-layer threat scanning.

Security layers:
- L1: Content regex scanning (50+ patterns)
- L2: YARA rule engine + ClamAV/VirusTotal integration
- L3: DNS-over-HTTPS with threat intel blocklists
- L4: TLS 1.3 enforcement + certificate verification
- L5: Encrypted scratch space (AES-256-GCM)
- L6: SIEM-grade audit logging (JSON + CEF export)
- L7: Secure memory handling (mlock + zero-on-free)
- L8: OS-level sandbox (macOS Seatbelt / Linux seccomp)
"""

from .trident import TridentBrowser
from .sandbox import SandboxedInstance
from .scanner import ThreatScanner
from .audit_log import AuditLogger
from .yara_scanner import YaraScanner
from .secure_memory import SecureMemory, EncryptedStateStore
from .hardened_sandbox import HardenedSandbox
from .encrypted_dns import EncryptedDNS
from .crypto_layer import CryptoLayer, EncryptedScratchSpace

__all__ = [
    "TridentBrowser", "SandboxedInstance", "ThreatScanner",
    "AuditLogger", "YaraScanner", "SecureMemory", "EncryptedStateStore",
    "HardenedSandbox", "EncryptedDNS", "CryptoLayer", "EncryptedScratchSpace",
]
__version__ = "2.0.0"
