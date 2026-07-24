"""API key authentication for OctoTetrahedral AGI.

Provides secure API key generation, hashing, validation, and usage tracking.
Keys are stored as SHA-256 hashes in ~/.octotetrahedral/api_keys.json.
"""

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_KEY_STORE_PATH = Path.home() / ".octotetrahedral" / "api_keys.json"
_KEY_PREFIX = "octo_"
_KEY_BYTES = 32

security = HTTPBearer(auto_error=False)

# In-memory cache: {hashed_key: usage_info}
_key_cache: dict[str, dict] = {}
_auth_enabled: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_keys() -> dict[str, dict]:
    """Load hashed keys from disk, returning empty dict if not present."""
    if not _KEY_STORE_PATH.exists():
        return {}
    try:
        with open(_KEY_STORE_PATH) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_keys(keys: dict[str, dict]) -> None:
    """Persist hashed keys to disk."""
    _KEY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_KEY_STORE_PATH, "w") as fh:
        json.dump(keys, fh, indent=2)


def _hash_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of a raw API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _refresh_cache() -> None:
    """Reload the in-memory cache from disk."""
    global _key_cache
    _key_cache = _load_keys()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_api_key(label: str = "") -> str:
    """Generate a new secure API key, store its hash, and return the raw key.

    The raw key is returned **once** and never stored; only the hash is kept.
    """
    raw_key = _KEY_PREFIX + secrets.token_urlsafe(_KEY_BYTES)
    hashed = _hash_key(raw_key)
    keys = _load_keys()
    keys[hashed] = {
        "label": label,
        "created_at": time.time(),
        "request_count": 0,
        "last_used": None,
    }
    _save_keys(keys)
    _refresh_cache()
    return raw_key


def validate_key(raw_key: str) -> bool:
    """Return True if *raw_key* matches a stored hash, and record usage."""
    if not _auth_enabled:
        return True
    if not _key_cache:
        _refresh_cache()
    hashed = _hash_key(raw_key)
    if hashed not in _key_cache:
        return False
    # Update usage stats in-memory (best effort – flush lazily)
    _key_cache[hashed]["request_count"] = _key_cache[hashed].get("request_count", 0) + 1
    _key_cache[hashed]["last_used"] = time.time()
    return True


def disable_auth() -> None:
    """Disable authentication (useful for local dev / testing)."""
    global _auth_enabled
    _auth_enabled = False


def enable_auth() -> None:
    """Re-enable authentication."""
    global _auth_enabled
    _auth_enabled = True


def get_key_stats() -> dict:
    """Return summary statistics for all registered API keys."""
    if not _key_cache:
        _refresh_cache()
    return {
        "total_keys": len(_key_cache),
        "keys": [
            {
                "label": info.get("label", ""),
                "created_at": info.get("created_at"),
                "request_count": info.get("request_count", 0),
                "last_used": info.get("last_used"),
            }
            for info in _key_cache.values()
        ],
    }


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> None:
    """FastAPI dependency that enforces API key authentication.

    Pass as ``dependencies=[Depends(verify_api_key)]`` on a route.
    When authentication is disabled the check is skipped entirely.
    """
    if not _auth_enabled:
        return
    if credentials is None or not validate_key(credentials.credentials):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass it as: Authorization: ******",
        )
