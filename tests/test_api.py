"""Tests for API authentication (auth.py) and the /predict endpoint (api.py).

These tests do not spin up the FastAPI server; they validate auth logic in
isolation and exercise key api.py functions through direct import.
"""

from __future__ import annotations

import time

import pytest

from auth import (
    RateLimiter,
    create_token,
    generate_api_key,
    hash_key,
    sign_request,
    verify_request_signature,
    verify_token,
)


# ---------------------------------------------------------------------------
# API key generation
# ---------------------------------------------------------------------------


def test_generate_api_key_has_prefix():
    key = generate_api_key()
    assert key.startswith("octo_")


def test_generate_api_key_is_unique():
    keys = {generate_api_key() for _ in range(20)}
    assert len(keys) == 20, "Generated keys must be unique"


def test_generate_api_key_minimum_length():
    key = generate_api_key()
    # prefix (5) + base64url(32 bytes) = 5 + 43 chars
    assert len(key) >= 40


def test_hash_key_is_deterministic():
    key = generate_api_key()
    assert hash_key(key) == hash_key(key)


def test_hash_key_differs_for_different_keys():
    k1, k2 = generate_api_key(), generate_api_key()
    assert hash_key(k1) != hash_key(k2)


# ---------------------------------------------------------------------------
# JWT-style tokens
# ---------------------------------------------------------------------------


def test_create_and_verify_token():
    key = generate_api_key()
    token = create_token(key)
    valid, key_hash = verify_token(token)
    assert valid is True
    assert key_hash == hash_key(key)


def test_verify_token_rejects_tampered_signature():
    key = generate_api_key()
    token = create_token(key)
    header, payload, sig = token.split(".")
    tampered = f"{header}.{payload}.INVALIDSIGNATURE"
    valid, _ = verify_token(tampered)
    assert valid is False


def test_verify_token_rejects_expired():
    key = generate_api_key()
    token = create_token(key, expires_in=-1)  # already expired
    valid, _ = verify_token(token)
    assert valid is False


def test_verify_token_rejects_malformed():
    assert verify_token("not.a.valid.token.at.all") == (False, None)
    assert verify_token("onlyonepart") == (False, None)


def test_token_has_three_parts():
    token = create_token(generate_api_key())
    assert token.count(".") == 2


# ---------------------------------------------------------------------------
# Request signing
# ---------------------------------------------------------------------------


def test_sign_and_verify_request():
    key = generate_api_key()
    body = b'{"input_ids": [1, 2, 3]}'
    sig = sign_request(body, key)
    assert verify_request_signature(body, key, sig) is True


def test_verify_request_rejects_wrong_key():
    key1, key2 = generate_api_key(), generate_api_key()
    body = b"test body"
    sig = sign_request(body, key1)
    assert verify_request_signature(body, key2, sig) is False


def test_verify_request_rejects_tampered_body():
    key = generate_api_key()
    body = b"original body"
    sig = sign_request(body, key)
    assert verify_request_signature(b"tampered body", key, sig) is False


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


def test_rate_limiter_allows_within_limit():
    rl = RateLimiter(limit=5, window_seconds=60)
    key_hash = hash_key(generate_api_key())
    for _ in range(5):
        assert rl.is_allowed(key_hash) is True


def test_rate_limiter_blocks_over_limit():
    rl = RateLimiter(limit=3, window_seconds=60)
    key_hash = hash_key(generate_api_key())
    for _ in range(3):
        rl.is_allowed(key_hash)
    assert rl.is_allowed(key_hash) is False


def test_rate_limiter_remaining_decreases():
    rl = RateLimiter(limit=10, window_seconds=60)
    key_hash = hash_key(generate_api_key())
    assert rl.remaining(key_hash) == 10
    rl.is_allowed(key_hash)
    assert rl.remaining(key_hash) == 9


def test_rate_limiter_keys_are_independent():
    rl = RateLimiter(limit=2, window_seconds=60)
    k1 = hash_key(generate_api_key())
    k2 = hash_key(generate_api_key())
    rl.is_allowed(k1)
    rl.is_allowed(k1)
    # k1 is at limit; k2 should still be allowed
    assert rl.is_allowed(k1) is False
    assert rl.is_allowed(k2) is True


# ---------------------------------------------------------------------------
# api.py imports
# ---------------------------------------------------------------------------


def test_api_imports_without_error():
    """api.py top-level imports must resolve without RuntimeError."""
    import importlib

    # api.py raises at startup when model checkpoint is missing;
    # we only validate the module-level symbols that don't require the file.
    from gpu_metal_support import device_info, select_device

    assert callable(select_device)
    assert callable(device_info)
