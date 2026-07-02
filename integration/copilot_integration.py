"""Integration glue: wire the model registry and Copilot adapter into the
task execution bootstrap sequence.

This module is imported during Copilot agent initialisation.  It:
1. Loads the ``.copilot/config.yml`` configuration file (if present).
2. Registers any models declared in the config.
3. Sets up the CopilotModelAdapter with the configured default model.
4. Exposes ``process_request`` for the Copilot inference pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(".copilot") / "config.yml"

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load and return the YAML config.  Returns ``{}`` if file is absent."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import]
        with path.open() as fh:
            data = yaml.safe_load(fh)
        return data or {}
    except ImportError:
        logger.warning("PyYAML not installed; config file ignored.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load config from %s: %s", path, exc)
    return {}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap(config: Optional[Dict[str, Any]] = None) -> "CopilotIntegration":
    """Initialise and return a ready CopilotIntegration instance."""
    if config is None:
        config = load_config()
    return CopilotIntegration(config)


# ---------------------------------------------------------------------------
# CopilotIntegration
# ---------------------------------------------------------------------------

class CopilotIntegration:
    """Wires the model registry and adapter into the Copilot pipeline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        from model_registry import get_registry
        from integration.copilot_model_adapter import CopilotModelAdapter

        self._config = config
        self._registry = get_registry()
        self._apply_config(config)

        default_model: str = (
            config.get("default_model")
            or config.get("user_preferences", {}).get("preferred_model")
            or "gpt-4"
        )
        self._adapter = CopilotModelAdapter(
            registry=self._registry,
            default_model=default_model,
        )
        logger.info("CopilotIntegration bootstrapped with default model '%s'.", default_model)

    # ------------------------------------------------------------------
    # Config application
    # ------------------------------------------------------------------

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Register models declared in the config file."""
        from model_registry import ModelMetadata

        models_cfg: Dict[str, Any] = config.get("models", {})
        for name, attrs in models_cfg.items():
            if not isinstance(attrs, dict):
                continue
            try:
                meta = ModelMetadata(
                    name=name,
                    description=attrs.get("description", ""),
                    limbs=int(attrs.get("limbs", 0)),
                    coherence_threshold=float(attrs.get("coherence_threshold", 0.90)),
                    batch_size=int(attrs.get("batch_size", 32)),
                    timeout_ms=int(attrs.get("timeout_ms", 30000)),
                    capabilities=list(attrs.get("capabilities", [])),
                    default_parameters={},
                )
                # Only register if not already present (avoid overwriting defaults)
                if name not in self._registry._metadata:
                    self._registry.register(meta)
                    logger.debug("Registered model from config: %s", name)
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed model config for '%s': %s", name, exc)

        # Fallback chain from config
        fallback_chain = config.get("fallback_chain")
        if fallback_chain and isinstance(fallback_chain, list):
            self._registry.set_fallback_chain(fallback_chain)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a Copilot request and return a structured response.

        The *request* dict may contain:
          prompt     : str   — required
          model      : str   — optional model override
          context    : dict  — optional additional context
          trace_id   : str   — optional trace identifier
        """
        trace_id = request.get("trace_id", _new_trace_id())
        request = {**request, "trace_id": trace_id}

        t0 = time.monotonic()
        response = self._adapter.process(request)
        response["integration_latency_ms"] = (time.monotonic() - t0) * 1000

        # Structured JSON log entry
        log_entry = {
            "trace_id": trace_id,
            "model": response.get("model"),
            "coherence": response.get("coherence"),
            "action_channel": response.get("action_channel"),
            "latency_ms": response.get("latency_ms"),
            "error": response.get("error"),
        }
        logger.info("model_selection %s", json.dumps(log_entry))
        return response

    @property
    def registry(self):
        return self._registry

    @property
    def adapter(self):
        return self._adapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_trace_id() -> str:
    import uuid
    return str(uuid.uuid4())
