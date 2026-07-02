"""Production configuration for the unified cognitive stack inference pipeline."""

from __future__ import annotations

import os
from typing import Literal

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

ENV = os.environ.get("OCTOAGI_ENV", "dev")
_VALID_ENVS: tuple[str, ...] = ("dev", "staging", "prod")
if ENV not in _VALID_ENVS:
    ENV = "dev"

Environment = Literal["dev", "staging", "prod"]

# ---------------------------------------------------------------------------
# Model paths and versions
# ---------------------------------------------------------------------------

MODEL_VERSION = os.environ.get("OCTOAGI_MODEL_VERSION", "1.0.0")
MODEL_PATH = os.environ.get("OCTOAGI_MODEL_PATH", "")          # empty = in-memory (no disk load)
MODEL_LIMB_COUNT = int(os.environ.get("OCTOAGI_LIMB_COUNT", "8"))

# ---------------------------------------------------------------------------
# Inference parameters
# ---------------------------------------------------------------------------

BATCH_SIZE_MIN = 1
BATCH_SIZE_MAX = 100
LIMB_STATES_MAX_LENGTH = 1000
INFERENCE_TIMEOUT_MS = float(os.environ.get("OCTOAGI_INFERENCE_TIMEOUT_MS", "20.0"))
MAX_RETRIES = int(os.environ.get("OCTOAGI_MAX_RETRIES", "3"))
POOL_SIZE = int(os.environ.get("OCTOAGI_POOL_SIZE", "4"))

# ---------------------------------------------------------------------------
# Monitoring thresholds
# ---------------------------------------------------------------------------

COHERENCE_ALERT_THRESHOLD = float(os.environ.get("OCTOAGI_COHERENCE_THRESHOLD", "0.90"))
LATENCY_WARN_MS = float(os.environ.get("OCTOAGI_LATENCY_WARN_MS", "20.0"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_LEVEL_MAP = {
    "dev": "DEBUG",
    "staging": "INFO",
    "prod": "WARNING",
}
LOG_LEVEL = os.environ.get("OCTOAGI_LOG_LEVEL", _LOG_LEVEL_MAP.get(ENV, "INFO"))
LOG_FORMAT = "json"

# ---------------------------------------------------------------------------
# Environment-specific overrides
# ---------------------------------------------------------------------------

_ENV_DEFAULTS: dict[str, dict[str, object]] = {
    "dev": {
        "pool_size": 1,
        "max_retries": 1,
        "inference_timeout_ms": 100.0,
    },
    "staging": {
        "pool_size": 2,
        "max_retries": 2,
        "inference_timeout_ms": 30.0,
    },
    "prod": {
        "pool_size": POOL_SIZE,
        "max_retries": MAX_RETRIES,
        "inference_timeout_ms": INFERENCE_TIMEOUT_MS,
    },
}

_active = _ENV_DEFAULTS.get(ENV, _ENV_DEFAULTS["dev"])
EFFECTIVE_POOL_SIZE: int = int(_active["pool_size"])  # type: ignore[arg-type]
EFFECTIVE_MAX_RETRIES: int = int(_active["max_retries"])  # type: ignore[arg-type]
EFFECTIVE_TIMEOUT_MS: float = float(_active["inference_timeout_ms"])  # type: ignore[arg-type]
