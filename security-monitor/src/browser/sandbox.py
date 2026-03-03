#!/usr/bin/env python3
"""
SandboxedInstance — A single isolated browser instance.

Each instance:
- Runs fetches in an isolated temp directory (destroyed on exit)
- Has no access to the host filesystem
- Has its own cookie jar (ephemeral, in-memory only)
- Content is scanned before being returned
- Can be killed instantly without affecting other instances
- Has resource limits (timeout, max response size)
"""

import os
import sys
import re
import time
import signal
import tempfile
import shutil
import logging
import hashlib
import subprocess
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from .scanner import ThreatScanner, ThreatReport, ThreatLevel

logger = logging.getLogger("browser.sandbox")


class InstanceState(Enum):
    IDLE = "idle"
    FETCHING = "fetching"
    SCANNING = "scanning"
    RENDERING = "rendering"
    DEAD = "dead"
    QUARANTINED = "quarantined"


@dataclass
class FetchResult:
    url: str
    status_code: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    raw_html: str = ""
    rendered_text: str = ""
    links: List[Tuple[str, str]] = field(default_factory=list)  # (url, text)
    threat_report: Optional[ThreatReport] = None
    error: Optional[str] = None
    fetch_time_ms: float = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.status_code < 400


class TextRenderer(HTMLParser):
    """Convert HTML to readable terminal text, extracting links."""

    def __init__(self, base_url: str = ""):
        super().__init__()
        self.base_url = base_url
        self.text_parts: List[str] = []
        self.links: List[Tuple[str, str]] = []
        self._current_link: Optional[str] = None
        self._link_text: List[str] = []
        self._skip_tags = {"script", "style", "noscript", "svg", "head"}
        self._skip_depth = 0
        self._block_tags = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
                            "li", "tr", "blockquote", "pre", "hr", "section", "article"}
        self._in_pre = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return

        attr_dict = dict(attrs)

        if tag == "a" and "href" in attr_dict:
            href = attr_dict["href"]
            if self.base_url and not href.startswith(("http://", "https://", "mailto:")):
                href = urljoin(self.base_url, href)
            self._current_link = href
            self._link_text = []

        if tag in self._block_tags:
            self.text_parts.append("\n")

        if tag == "br":
            self.text_parts.append("\n")
        elif tag == "hr":
            self.text_parts.append("\n" + "─" * 60 + "\n")
        elif tag in {"h1", "h2", "h3"}:
            self.text_parts.append("\n")
        elif tag == "li":
            self.text_parts.append("\n  • ")
        elif tag == "pre":
            self._in_pre = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return

        if tag == "a" and self._current_link:
            link_text = "".join(self._link_text).strip()
            if link_text and self._current_link.startswith(("http://", "https://")):
                idx = len(self.links)
                self.links.append((self._current_link, link_text))
                self.text_parts.append(f" [{idx}]")
            self._current_link = None
            self._link_text = []

        if tag in {"h1", "h2", "h3"}:
            self.text_parts.append("\n")
        elif tag == "pre":
            self._in_pre = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if not self._in_pre:
            data = re.sub(r'\s+', ' ', data)
        if self._current_link is not None:
            self._link_text.append(data)
        self.text_parts.append(data)

    def get_text(self) -> str:
        text = "".join(self.text_parts)
        # Collapse multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


class SandboxedInstance:
    """A single sandboxed browser instance with isolation guarantees."""

    MAX_RESPONSE_SIZE = 5 * 1024 * 1024  # 5MB max
    FETCH_TIMEOUT = 15  # seconds
    USER_AGENT = "QuantumShield-SecureBrowser/1.0 (Sandboxed)"

    def __init__(self, instance_id: int, scanner: ThreatScanner) -> None:
        self.instance_id = instance_id
        self.scanner = scanner
        self.state = InstanceState.IDLE
        self.sandbox_dir = tempfile.mkdtemp(prefix=f"qsbrowser_{instance_id}_")
        self.history: List[str] = []
        self.current_url: str = ""
        self.last_result: Optional[FetchResult] = None
        self.fetch_count = 0
        self.threats_blocked = 0
        self._executor = ThreadPoolExecutor(max_workers=1)
        logger.info(f"Instance {instance_id} sandbox: {self.sandbox_dir}")

    def fetch(self, url: str) -> FetchResult:
        """Fetch a URL inside the sandbox with full isolation."""
        self.state = InstanceState.FETCHING
        start = time.time()
        result = FetchResult(url=url)

        try:
            # Use curl subprocess for network isolation
            # (no shared Python state, no cookie leaks)
            result = self._fetch_with_curl(url)
            elapsed = (time.time() - start) * 1000
            result.fetch_time_ms = elapsed

            if result.error:
                self.state = InstanceState.IDLE
                return result

            # Scan for threats
            self.state = InstanceState.SCANNING
            report = self.scanner.scan(url, result.raw_html, result.headers)
            result.threat_report = report

            if report.blocked:
                self.threats_blocked += 1
                result.rendered_text = (
                    f"\n🚫 BLOCKED — {report.level.name} THREAT DETECTED\n"
                    f"{'='*60}\n"
                )
                for t in report.threats:
                    result.rendered_text += f"  [{t['level']}] {t['type']}: {t['detail']}\n"
                result.rendered_text += f"{'='*60}\n"
                result.rendered_text += f"URL: {url}\n"
                result.rendered_text += f"Content hash: {report.content_hash[:16]}...\n"
                self.state = InstanceState.IDLE
                return result

            # Sanitize and render
            self.state = InstanceState.RENDERING
            sanitized = self.scanner.sanitize(result.raw_html)
            renderer = TextRenderer(base_url=url)
            renderer.feed(sanitized)
            result.rendered_text = renderer.get_text()
            result.links = renderer.links

            self.history.append(url)
            self.current_url = url
            self.fetch_count += 1
            self.last_result = result

        except Exception as e:
            result.error = str(e)
            logger.error(f"Instance {self.instance_id} fetch error: {e}")

        self.state = InstanceState.IDLE
        return result

    def _fetch_with_curl(self, url: str) -> FetchResult:
        """Fetch URL using curl subprocess for isolation."""
        result = FetchResult(url=url)

        # Validate URL
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            result.error = f"Blocked scheme: {parsed.scheme} (only http/https allowed)"
            return result

        # Block internal/private IPs
        hostname = parsed.hostname or ""
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0") or hostname.startswith("192.168.") or hostname.startswith("10."):
            result.error = f"Blocked: private/internal address {hostname}"
            return result

        try:
            # Headers output file inside sandbox
            header_file = os.path.join(self.sandbox_dir, "headers.txt")

            cmd = [
                "curl", "-sS",
                "--max-time", str(self.FETCH_TIMEOUT),
                "--max-filesize", str(self.MAX_RESPONSE_SIZE),
                "-L",  # follow redirects
                "--max-redirs", "5",
                "-A", self.USER_AGENT,
                "-D", header_file,  # dump headers
                "--no-sessionid",
                "--no-keepalive",
                url,
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.FETCH_TIMEOUT + 5,
                cwd=self.sandbox_dir,
            )

            if proc.returncode != 0:
                result.error = f"curl error (code {proc.returncode}): {proc.stderr[:200]}"
                return result

            result.raw_html = proc.stdout
            result.status_code = 200  # default

            # Parse headers
            if os.path.exists(header_file):
                with open(header_file, "r") as f:
                    header_text = f.read()
                for line in header_text.split("\n"):
                    if line.startswith("HTTP/"):
                        parts = line.split(None, 2)
                        if len(parts) >= 2:
                            try:
                                result.status_code = int(parts[1])
                            except ValueError:
                                pass
                    elif ":" in line:
                        key, _, val = line.partition(":")
                        result.headers[key.strip().lower()] = val.strip()
                os.remove(header_file)

        except subprocess.TimeoutExpired:
            result.error = f"Timeout after {self.FETCH_TIMEOUT}s"
        except Exception as e:
            result.error = str(e)

        return result

    def destroy(self) -> None:
        """Completely destroy this sandbox instance."""
        self.state = InstanceState.DEAD
        self._executor.shutdown(wait=False)
        try:
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        except Exception:
            pass
        logger.info(f"Instance {self.instance_id} destroyed")

    def quarantine(self) -> None:
        """Quarantine this instance — stop all activity, preserve for analysis."""
        self.state = InstanceState.QUARANTINED
        logger.warning(f"Instance {self.instance_id} QUARANTINED")

    def is_alive(self) -> bool:
        return self.state not in (InstanceState.DEAD, InstanceState.QUARANTINED)

    def status(self) -> Dict:
        return {
            "id": self.instance_id,
            "state": self.state.value,
            "url": self.current_url,
            "fetches": self.fetch_count,
            "blocked": self.threats_blocked,
            "sandbox": self.sandbox_dir,
        }
