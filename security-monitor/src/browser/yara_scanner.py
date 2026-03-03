#!/usr/bin/env python3
"""
YARA-based threat scanner + VirusTotal hash lookup.

Extends the regex scanner with:
- YARA rules for structured malware detection
- ClamAV integration (if available) 
- VirusTotal API hash lookup for downloaded files
- Custom rule authoring support
"""

import os
import re
import hashlib
import logging
import subprocess
import json
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("browser.yara")


# Built-in YARA rules (compiled at runtime if yara-python available)
BUILTIN_YARA_RULES = """
rule CryptoMiner {
    meta:
        description = "Detects cryptocurrency mining scripts"
        severity = "critical"
    strings:
        $miner1 = "coinhive" nocase
        $miner2 = "CoinImp" nocase
        $miner3 = "cryptonight" nocase
        $miner4 = "stratum+tcp://" nocase
        $miner5 = "hashrate" nocase
        $wasm1 = "WebAssembly.instantiate" nocase
        $wasm2 = "WebAssembly.compile" nocase
    condition:
        any of ($miner*) or (any of ($wasm*) and any of ($miner*))
}

rule PhishingKit {
    meta:
        description = "Detects phishing page indicators"
        severity = "high"
    strings:
        $form = "<form" nocase
        $pass = "type=\"password\"" nocase
        $pass2 = "type='password'" nocase
        $urgent1 = "verify your account" nocase
        $urgent2 = "confirm your identity" nocase
        $urgent3 = "update your payment" nocase
        $urgent4 = "suspended" nocase
        $urgent5 = "unusual activity" nocase
    condition:
        $form and ($pass or $pass2) and any of ($urgent*)
}

rule JavaScriptObfuscation {
    meta:
        description = "Detects heavily obfuscated JavaScript"
        severity = "high"
    strings:
        $eval = "eval(" nocase
        $fromchar = "String.fromCharCode" nocase
        $atob = "atob(" nocase
        $unescape = "unescape(" nocase
        $hex_chain = /\\\\x[0-9a-fA-F]{2}(\\\\x[0-9a-fA-F]{2}){10,}/ 
        $unicode_chain = /\\\\u00[0-9a-fA-F]{2}(\\\\u00[0-9a-fA-F]{2}){10,}/
    condition:
        ($eval and ($fromchar or $atob or $unescape)) or $hex_chain or $unicode_chain
}

rule DriveByDownload {
    meta:
        description = "Detects drive-by download attempts"
        severity = "critical"
    strings:
        $iframe_exe = /iframe[^>]*src[^>]*\\.(exe|scr|bat|ps1|msi)/ nocase
        $meta_refresh = /meta[^>]*http-equiv[^>]*refresh[^>]*url/ nocase
        $auto_dl = /Content-Disposition.*attachment.*\\.(exe|scr|bat|ps1)/ nocase
        $js_redirect = /window\\.location[^;]*\\.(exe|scr|bat|ps1)/ nocase
    condition:
        any of them
}

rule DataExfiltration {
    meta:
        description = "Detects data theft attempts"
        severity = "high"  
    strings:
        $cookie = "document.cookie" nocase
        $storage = "localStorage" nocase
        $indexed = "indexedDB" nocase
        $clipboard = "navigator.clipboard" nocase
        $creds = "navigator.credentials" nocase
        $ws = "new WebSocket(" nocase
        $beacon = "navigator.sendBeacon(" nocase
    condition:
        2 of them
}

rule MaliciousRedirect {
    meta:
        description = "Detects suspicious redirect chains"
        severity = "medium"
    strings:
        $meta_redir = /meta[^>]*refresh[^>]*content[^>]*url=/i
        $js_redir1 = "window.location.replace(" nocase
        $js_redir2 = "window.location.href=" nocase
        $js_redir3 = "document.location=" nocase
        $hidden_form = /form[^>]*style[^>]*display\\s*:\\s*none/i
    condition:
        2 of them
}
"""


@dataclass
class YaraMatch:
    rule: str
    severity: str
    description: str
    matched_strings: List[str] = field(default_factory=list)


class YaraScanner:
    """YARA-based content scanner with ClamAV and VirusTotal integration."""

    def __init__(self, custom_rules_dir: Optional[str] = None,
                 virustotal_key: Optional[str] = None) -> None:
        self._yara_available = False
        self._yara_rules = None
        self._clamav_available = False
        self._vt_key = virustotal_key
        self._fallback_patterns: List[Tuple[str, str, str, re.Pattern]] = []

        # Try to import yara
        try:
            import yara
            self._yara_available = True
            self._yara_rules = yara.compile(source=BUILTIN_YARA_RULES)
            logger.info("YARA engine loaded with built-in rules")
        except ImportError:
            logger.info("yara-python not installed — using regex fallback")
            self._compile_fallback_patterns()

        # Check ClamAV availability
        try:
            result = subprocess.run(["clamscan", "--version"],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self._clamav_available = True
                logger.info(f"ClamAV available: {result.stdout.strip()}")
        except Exception:
            logger.info("ClamAV not installed — skipping AV integration")

    def _compile_fallback_patterns(self) -> None:
        """Compile regex fallback patterns from YARA rules."""
        # Parse YARA rule structure into regex patterns
        rules = [
            ("CryptoMiner", "critical", "Cryptocurrency mining",
             re.compile(r'coinhive|coinimp|cryptonight|stratum\+tcp://|hashrate', re.I)),
            ("PhishingKit", "high", "Phishing page",
             re.compile(r'(?=.*<form)(?=.*type=.password.)(?=.*(?:verify|confirm|suspended|unusual))', re.I | re.S)),
            ("JavaScriptObfuscation", "high", "Obfuscated JS",
             re.compile(r'eval\s*\(.*(?:fromCharCode|atob|unescape)|\\x[0-9a-f]{2}(?:\\x[0-9a-f]{2}){10,}', re.I | re.S)),
            ("DriveByDownload", "critical", "Drive-by download",
             re.compile(r'iframe[^>]*\.(exe|scr|bat|ps1)|meta[^>]*refresh[^>]*url|Content-Disposition.*\.(exe|scr)', re.I)),
            ("DataExfiltration", "high", "Data theft",
             re.compile(r'(?:document\.cookie|localStorage|indexedDB|navigator\.clipboard|navigator\.credentials|sendBeacon)', re.I)),
            ("MaliciousRedirect", "medium", "Suspicious redirect",
             re.compile(r'(?:window\.location\.replace|document\.location\s*=|display\s*:\s*none.*form)', re.I | re.S)),
        ]
        self._fallback_patterns = rules

    def scan_content(self, content: str, url: str = "") -> List[YaraMatch]:
        """Scan content with YARA rules (or regex fallback)."""
        matches = []

        if self._yara_available and self._yara_rules:
            try:
                yara_matches = self._yara_rules.match(data=content.encode("utf-8", errors="replace"))
                for m in yara_matches:
                    severity = m.meta.get("severity", "medium")
                    description = m.meta.get("description", m.rule)
                    matched = [str(s) for s in m.strings[:5]]
                    matches.append(YaraMatch(
                        rule=m.rule,
                        severity=severity,
                        description=description,
                        matched_strings=matched,
                    ))
            except Exception as e:
                logger.error(f"YARA scan error: {e}")
        else:
            # Regex fallback
            for rule_name, severity, description, pattern in self._fallback_patterns:
                found = pattern.findall(content)
                if found:
                    matches.append(YaraMatch(
                        rule=rule_name,
                        severity=severity,
                        description=description,
                        matched_strings=[str(f)[:80] for f in found[:3]],
                    ))

        return matches

    def scan_file_clamav(self, filepath: str) -> Optional[str]:
        """Scan a file with ClamAV. Returns threat name or None if clean."""
        if not self._clamav_available:
            return None
        try:
            result = subprocess.run(
                ["clamscan", "--no-summary", filepath],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 1:
                # Virus found
                for line in result.stdout.split("\n"):
                    if "FOUND" in line:
                        return line.strip()
            return None
        except Exception as e:
            logger.error(f"ClamAV scan error: {e}")
            return None

    def check_virustotal(self, content_hash: str) -> Optional[Dict]:
        """Check a SHA256 hash against VirusTotal (requires API key)."""
        if not self._vt_key:
            return None
        try:
            result = subprocess.run(
                [
                    "curl", "-sS", "--max-time", "10",
                    "-H", f"x-apikey: {self._vt_key}",
                    f"https://www.virustotal.com/api/v3/files/{content_hash}",
                ],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                attrs = data.get("data", {}).get("attributes", {})
                stats = attrs.get("last_analysis_stats", {})
                return {
                    "hash": content_hash,
                    "malicious": stats.get("malicious", 0),
                    "suspicious": stats.get("suspicious", 0),
                    "harmless": stats.get("harmless", 0),
                    "undetected": stats.get("undetected", 0),
                }
        except Exception as e:
            logger.error(f"VirusTotal lookup error: {e}")
        return None

    def stats(self) -> Dict:
        return {
            "yara_available": self._yara_available,
            "clamav_available": self._clamav_available,
            "virustotal_configured": bool(self._vt_key),
            "fallback_rules": len(self._fallback_patterns) if not self._yara_available else 0,
        }
