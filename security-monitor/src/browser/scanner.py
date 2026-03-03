#!/usr/bin/env python3
"""
ThreatScanner — Scans fetched web content for malicious patterns.

Checks for:
- JavaScript injection / XSS payloads
- Phishing indicators (fake login forms, credential harvesting)
- Drive-by download triggers (auto-download iframes, obfuscated JS)
- Crypto miners (WebAssembly miners, coinhive-style scripts)
- Suspicious redirects and meta-refresh attacks
- Malicious file downloads (.exe, .scr, .bat, .ps1 links)
- Data exfiltration beacons
"""

import re
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("browser.scanner")


class ThreatLevel(Enum):
    CLEAN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThreatReport:
    url: str
    level: ThreatLevel
    threats: List[Dict[str, str]] = field(default_factory=list)
    content_hash: str = ""
    blocked: bool = False

    @property
    def is_safe(self) -> bool:
        return self.level.value <= ThreatLevel.LOW.value

    def summary(self) -> str:
        if self.is_safe:
            return f"✅ CLEAN — {self.url}"
        icon = {ThreatLevel.MEDIUM: "⚠️", ThreatLevel.HIGH: "🔴", ThreatLevel.CRITICAL: "🚫"}
        return f"{icon.get(self.level, '⚠️')} {self.level.name} — {len(self.threats)} threat(s) — {self.url}"


class ThreatScanner:
    """Scans HTML/text content for malicious patterns before rendering."""

    # Dangerous JS patterns
    JS_INJECTION = [
        r'<script[^>]*>.*?(eval|document\.write|innerHTML|outerHTML)\s*[\(=]',
        r'javascript\s*:',
        r'on(load|error|click|mouseover|focus)\s*=\s*["\']',
        r'<script[^>]*src\s*=\s*["\']data:',
        r'String\.fromCharCode',
        r'\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}',
        r'atob\s*\(',
        r'\\u00[0-9a-fA-F]{2}.*\\u00[0-9a-fA-F]{2}',
    ]

    # Phishing indicators
    PHISHING = [
        r'<form[^>]*action\s*=\s*["\']https?://[^"\']*(?:login|signin|verify|secure|account|update)',
        r'<input[^>]*type\s*=\s*["\']password["\']',
        r'(?:verify|confirm|update)\s+(?:your\s+)?(?:account|password|identity|billing)',
        r'(?:suspended|locked|limited|unusual\s+activity)',
        r'(?:click\s+here|act\s+now|immediately|urgent)',
    ]

    # Crypto miners
    MINERS = [
        r'coinhive',
        r'cryptonight',
        r'minero\.cc',
        r'coin-hive',
        r'jsecoin',
        r'cryptoloot',
        r'webmine',
        r'deepminer',
        r'WebAssembly\.instantiate.*mining',
    ]

    # Drive-by downloads
    DRIVEBYPASS = [
        r'<iframe[^>]*src\s*=\s*["\'][^"\']*\.(exe|scr|bat|ps1|msi|cmd)',
        r'<meta[^>]*http-equiv\s*=\s*["\']refresh["\'][^>]*url\s*=',
        r'window\.location\s*=\s*["\'][^"\']*\.(exe|scr|bat|ps1)',
        r'Content-Disposition.*attachment.*\.(exe|scr|bat|ps1|msi)',
        r'<a[^>]*download\s*=\s*[^>]*\.(exe|scr|bat|ps1|msi|cmd)',
    ]

    # Suspicious file links
    DANGEROUS_EXTENSIONS = [
        r'href\s*=\s*["\'][^"\']*\.(exe|scr|bat|ps1|msi|cmd|vbs|wsf|hta|cpl|jar|pif)',
        r'src\s*=\s*["\'][^"\']*\.(exe|scr|bat|cmd)',
    ]

    # Data exfiltration
    EXFILTRATION = [
        r'navigator\.(clipboard|credentials)',
        r'document\.cookie',
        r'localStorage\.',
        r'indexedDB\.',
        r'new\s+WebSocket\s*\(',
        r'fetch\s*\([^)]*\{[^}]*method\s*:\s*["\']POST',
        r'XMLHttpRequest.*\.send\(',
    ]

    # Known malicious domains (sample — extend in production)
    MALICIOUS_DOMAINS = {
        "malware-traffic-analysis.net",
        "urlhaus.abuse.ch",
        "phishing.army",
    }

    def __init__(self) -> None:
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile all regex patterns for performance."""
        self._patterns: List[Tuple[str, re.Pattern, ThreatLevel]] = []
        for p in self.JS_INJECTION:
            self._patterns.append(("JS Injection", re.compile(p, re.I | re.S), ThreatLevel.HIGH))
        for p in self.PHISHING:
            self._patterns.append(("Phishing", re.compile(p, re.I | re.S), ThreatLevel.HIGH))
        for p in self.MINERS:
            self._patterns.append(("Crypto Miner", re.compile(p, re.I), ThreatLevel.CRITICAL))
        for p in self.DRIVEBYPASS:
            self._patterns.append(("Drive-by Download", re.compile(p, re.I | re.S), ThreatLevel.CRITICAL))
        for p in self.DANGEROUS_EXTENSIONS:
            self._patterns.append(("Dangerous File", re.compile(p, re.I), ThreatLevel.MEDIUM))
        for p in self.EXFILTRATION:
            self._patterns.append(("Data Exfiltration", re.compile(p, re.I), ThreatLevel.HIGH))

    def scan(self, url: str, content: str, headers: Optional[Dict[str, str]] = None) -> ThreatReport:
        """Scan content and return a threat report."""
        report = ThreatReport(
            url=url,
            level=ThreatLevel.CLEAN,
            content_hash=hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest(),
        )

        # Check URL against known malicious domains
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).hostname or ""
            if domain in self.MALICIOUS_DOMAINS:
                report.threats.append({
                    "type": "Known Malicious Domain",
                    "detail": f"Domain {domain} is on the blocklist",
                    "level": "CRITICAL",
                })
                report.level = ThreatLevel.CRITICAL
                report.blocked = True
                return report
        except Exception:
            pass

        # Check header-based attacks
        if headers:
            ct = headers.get("content-type", "")
            if "application/x-msdownload" in ct or "application/x-msdos-program" in ct:
                report.threats.append({
                    "type": "Binary Download",
                    "detail": f"Server returned executable content-type: {ct}",
                    "level": "CRITICAL",
                })
                report.level = ThreatLevel.CRITICAL
                report.blocked = True
                return report

        # Pattern scanning
        max_level = ThreatLevel.CLEAN
        for name, pattern, level in self._patterns:
            matches = pattern.findall(content)
            if matches:
                detail = matches[0] if isinstance(matches[0], str) else str(matches[0])
                if len(detail) > 120:
                    detail = detail[:120] + "..."
                report.threats.append({
                    "type": name,
                    "detail": detail,
                    "level": level.name,
                })
                if level.value > max_level.value:
                    max_level = level

        report.level = max_level
        if max_level.value >= ThreatLevel.HIGH.value:
            report.blocked = True

        return report

    def sanitize(self, content: str) -> str:
        """Strip all scripts and dangerous elements from HTML."""
        # Remove script tags entirely
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.I | re.S)
        # Remove event handlers
        content = re.sub(r'\s+on\w+\s*=\s*["\'][^"\']*["\']', '', content, flags=re.I)
        # Remove iframes
        content = re.sub(r'<iframe[^>]*>.*?</iframe>', '', content, flags=re.I | re.S)
        # Remove object/embed
        content = re.sub(r'<(object|embed|applet)[^>]*>.*?</\1>', '', content, flags=re.I | re.S)
        # Remove style tags (CSS-based attacks)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.I | re.S)
        return content
