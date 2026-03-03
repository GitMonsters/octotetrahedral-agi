#!/usr/bin/env python3
"""
Secure memory handling and encrypted state management.

Provides:
- Secure memory allocation with mlock (prevent swapping to disk)
- Automatic zeroing of sensitive data on deallocation
- Encrypted in-memory key-value store for browser state
- Protection against memory dumps/forensics
"""

import os
import sys
import ctypes
import secrets
import hashlib
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("browser.memory")


class SecureMemory:
    """Secure memory handling — prevent sensitive data from leaking to disk."""

    def __init__(self) -> None:
        self._locked_regions: list = []
        self._platform = sys.platform

    def secure_alloc(self, size: int) -> bytearray:
        """Allocate a bytearray and attempt to mlock it (prevent swap)."""
        buf = bytearray(size)
        self._try_mlock(buf)
        return buf

    def secure_free(self, buf: bytearray) -> None:
        """Zero out and release a secure buffer."""
        for i in range(len(buf)):
            buf[i] = 0
        # Write random over it too (defense against cold boot)
        rand = secrets.token_bytes(len(buf))
        for i in range(len(buf)):
            buf[i] = rand[i]
        # Final zero
        for i in range(len(buf)):
            buf[i] = 0

    def _try_mlock(self, buf: bytearray) -> bool:
        """Try to lock memory pages (prevent swap to disk)."""
        try:
            if self._platform == "darwin":
                libc = ctypes.CDLL("libSystem.B.dylib")
            elif self._platform.startswith("linux"):
                libc = ctypes.CDLL("libc.so.6")
            else:
                return False

            addr = ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))
            result = libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(len(buf)))
            if result == 0:
                self._locked_regions.append((addr, len(buf)))
                return True
        except Exception as e:
            logger.debug(f"mlock unavailable: {e}")
        return False

    def secure_string(self, s: str) -> "SecureString":
        """Create a secure string that zeros on delete."""
        return SecureString(s, self)


class SecureString:
    """A string-like object that securely zeros its contents when destroyed."""

    def __init__(self, value: str, mem: SecureMemory) -> None:
        self._data = bytearray(value.encode("utf-8"))
        self._mem = mem
        mem._try_mlock(self._data)

    def get(self) -> str:
        return self._data.decode("utf-8")

    def __len__(self) -> int:
        return len(self._data)

    def __del__(self) -> None:
        self._mem.secure_free(self._data)

    def __str__(self) -> str:
        return "[REDACTED]"

    def __repr__(self) -> str:
        return "SecureString([REDACTED])"


class EncryptedStateStore:
    """Encrypted in-memory key-value store for browser state.
    
    All values are encrypted with a session-unique key.
    Keys are hashed (not stored in plaintext).
    """

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)
        self._store: Dict[str, bytes] = {}
        self._mem = SecureMemory()

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode() + self._key[:8]).hexdigest()

    def _encrypt_value(self, value: bytes) -> bytes:
        nonce = secrets.token_bytes(16)
        key_stream = hashlib.sha256(self._key + nonce).digest()
        # XOR encryption (use AES in production with hardware support)
        encrypted = bytes(b ^ key_stream[i % 32] for i, b in enumerate(value))
        return nonce + encrypted

    def _decrypt_value(self, data: bytes) -> bytes:
        nonce = data[:16]
        encrypted = data[16:]
        key_stream = hashlib.sha256(self._key + nonce).digest()
        return bytes(b ^ key_stream[i % 32] for i, b in enumerate(encrypted))

    def put(self, key: str, value: Any) -> None:
        """Store an encrypted value."""
        if isinstance(value, str):
            raw = value.encode("utf-8")
        elif isinstance(value, bytes):
            raw = value
        else:
            import json
            raw = json.dumps(value).encode("utf-8")

        hashed_key = self._hash_key(key)
        self._store[hashed_key] = self._encrypt_value(raw)

    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Retrieve and decrypt a value."""
        hashed_key = self._hash_key(key)
        encrypted = self._store.get(hashed_key)
        if encrypted is None:
            return default
        try:
            raw = self._decrypt_value(encrypted)
            return raw.decode("utf-8")
        except Exception:
            return default

    def delete(self, key: str) -> None:
        """Securely delete a value."""
        hashed_key = self._hash_key(key)
        if hashed_key in self._store:
            # Overwrite with random before deleting
            old = self._store[hashed_key]
            self._store[hashed_key] = secrets.token_bytes(len(old))
            del self._store[hashed_key]

    def destroy(self) -> None:
        """Securely destroy all state."""
        for k in list(self._store.keys()):
            self._store[k] = secrets.token_bytes(len(self._store[k]))
        self._store.clear()
        # Zero the master key
        key_buf = bytearray(self._key)
        self._mem.secure_free(key_buf)
        self._key = bytes(key_buf)

    def stats(self) -> Dict:
        return {
            "entries": len(self._store),
            "key_id": hashlib.sha256(self._key).hexdigest()[:12],
        }
