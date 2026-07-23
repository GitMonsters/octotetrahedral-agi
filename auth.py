"""API authentication and security for OctoTetrahedral AGI.

Provides:
- Secure API key generation and validation
- JWT-style token authentication (HMAC-SHA256)
- Per-key rate limiting
- Request signing validation
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_KEYS_FILE = Path.home() / ".octotetrahedral" / "api_keys.json"
_DEFAULT_RATE_LIMIT = int(os.getenv("OCTO_RATE_LIMIT", "1000"))  # req/min per key
_JWT_SECRET = os.getenv("OCTO_JWT_SECRET", secrets.token_hex(32))
_KEY_PREFIX = "octo_"
_KEY_BYTES = 32  # 256-bit keys


# ---------------------------------------------------------------------------
# API key generation
# ---------------------------------------------------------------------------


def generate_api_key() -> str:
    """Generate a cryptographically secure random API key.

    Returns a string of the form ``octo_<base64url-encoded-random-bytes>``.
    """
    raw = secrets.token_bytes(_KEY_BYTES)
    encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    return f"{_KEY_PREFIX}{encoded}"


def hash_key(api_key: str) -> str:
    """Return the SHA-256 hex digest of an API key (for storage)."""
    return hashlib.sha256(api_key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Persistent key store
# ---------------------------------------------------------------------------


def _load_keys() -> Dict[str, Dict]:
    """Load API keys from the JSON store; return empty dict on any error."""
    if not _API_KEYS_FILE.exists():
        return {}
    try:
        with _API_KEYS_FILE.open() as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not load API key store: %s", exc)
        return {}


def _save_keys(keys: Dict[str, Dict]) -> None:
    """Persist API keys to the JSON store."""
    _API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _API_KEYS_FILE.open("w") as fh:
            json.dump(keys, fh, indent=2)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not save API key store: %s", exc)


def store_api_key(api_key: str, label: str = "") -> str:
    """Hash and persist *api_key*.  Returns the key hash."""
    key_hash = hash_key(api_key)
    keys = _load_keys()
    keys[key_hash] = {
        "label": label,
        "created_at": time.time(),
        "request_count": 0,
    }
    _save_keys(keys)
    logger.info("API key stored (hash prefix: %s…)", key_hash[:8])
    return key_hash


def validate_api_key(api_key: str) -> Tuple[bool, Optional[str]]:
    """Check whether *api_key* exists in the store.

    Returns ``(True, key_hash)`` if valid, ``(False, None)`` otherwise.
    """
    key_hash = hash_key(api_key)
    keys = _load_keys()
    if key_hash in keys:
        # Increment usage counter (best-effort; no lock needed for monitoring)
        keys[key_hash]["request_count"] = keys[key_hash].get("request_count", 0) + 1
        _save_keys(keys)
        return True, key_hash
    return False, None


# ---------------------------------------------------------------------------
# JWT-style tokens
# ---------------------------------------------------------------------------


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sign(payload: str, secret: str) -> str:
    return _b64url(
        hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    )


def create_token(api_key: str, expires_in: int = 3600) -> str:
    """Create a signed JWT-style token for *api_key*.

    Args:
        api_key: The raw API key to embed in the token.
        expires_in: Token validity in seconds (default 1 hour).

    Returns:
        A ``header.payload.signature`` token string.
    """
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    claims = {
        "sub": hash_key(api_key),
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_in,
    }
    payload = _b64url(json.dumps(claims).encode())
    signature = _sign(f"{header}.{payload}", _JWT_SECRET)
    return f"{header}.{payload}.{signature}"


def verify_token(token: str) -> Tuple[bool, Optional[str]]:
    """Verify a JWT-style token.

    Returns ``(True, key_hash)`` if the token is valid and unexpired,
    ``(False, None)`` otherwise.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False, None
        header, payload_b64, provided_sig = parts
        expected_sig = _sign(f"{header}.{payload_b64}", _JWT_SECRET)
        if not hmac.compare_digest(expected_sig, provided_sig):
            return False, None
        # Decode payload
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
        if time.time() > claims.get("exp", 0):
            return False, None
        return True, claims.get("sub")
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# Request signing
# ---------------------------------------------------------------------------


def sign_request(body: bytes, api_key: str) -> str:
    """Return an HMAC-SHA256 hex digest for request body signing."""
    return hmac.new(api_key.encode(), body, hashlib.sha256).hexdigest()


def verify_request_signature(body: bytes, api_key: str, provided: str) -> bool:
    """Return True when *provided* matches the expected request signature."""
    expected = sign_request(body, api_key)
    return hmac.compare_digest(expected, provided)


# ---------------------------------------------------------------------------
# Per-key rate limiter (in-memory sliding window)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window rate limiter keyed by API key hash.

    Args:
        limit: Maximum allowed requests per *window_seconds*.
        window_seconds: Duration of the sliding window (default 60 s).
    """

    def __init__(self, limit: int = _DEFAULT_RATE_LIMIT, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window = window_seconds
        self._windows: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key_hash: str) -> bool:
        """Return True if *key_hash* is within its rate limit."""
        now = time.time()
        cutoff = now - self.window
        dq = self._windows[key_hash]
        # Evict expired timestamps
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= self.limit:
            return False
        dq.append(now)
        return True

    def remaining(self, key_hash: str) -> int:
        """Return how many requests remain in the current window."""
        now = time.time()
        cutoff = now - self.window
        dq = self._windows[key_hash]
        while dq and dq[0] < cutoff:
            dq.popleft()
        return max(0, self.limit - len(dq))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

rate_limiter = RateLimiter()
