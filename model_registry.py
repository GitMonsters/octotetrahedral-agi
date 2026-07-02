"""Model registry for the unified cognitive stack and external models.

Supports model-name and model-name:variant syntax, fallback chains,
capability discovery, and lazy loading.
"""

from __future__ import annotations

import importlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    name: str
    description: str
    limbs: int = 0
    coherence_threshold: float = 0.90
    batch_size: int = 32
    timeout_ms: int = 30000
    capabilities: List[str] = field(default_factory=list)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    provider: str = "local"


@dataclass
class ModelInstance:
    metadata: ModelMetadata
    instance: Any  # lazy-loaded model object
    loaded_at: float = 0.0


# ---------------------------------------------------------------------------
# Default model definitions
# ---------------------------------------------------------------------------

DEFAULT_MODELS: Dict[str, ModelMetadata] = {
    "unified-stack": ModelMetadata(
        name="unified-stack",
        description="8-limb quantum-biological unified model (production)",
        limbs=8,
        coherence_threshold=0.90,
        batch_size=32,
        timeout_ms=30000,
        capabilities=["reasoning", "language", "spatial", "planning"],
        default_parameters={"limb_count": 8},
        version="1.0",
        provider="local",
    ),
    "unified-stack-16": ModelMetadata(
        name="unified-stack-16",
        description="16-limb extended model (experimental)",
        limbs=16,
        coherence_threshold=0.92,
        batch_size=16,
        timeout_ms=50000,
        capabilities=["reasoning", "language", "spatial", "planning", "multi-domain"],
        default_parameters={"limb_count": 16},
        version="1.0",
        provider="local",
    ),
    "gpt-4": ModelMetadata(
        name="gpt-4",
        description="OpenAI GPT-4",
        limbs=0,
        coherence_threshold=0.0,
        batch_size=64,
        timeout_ms=60000,
        capabilities=["reasoning", "language", "planning"],
        default_parameters={},
        version="1.0",
        provider="openai",
    ),
    "claude-3-opus": ModelMetadata(
        name="claude-3-opus",
        description="Anthropic Claude 3 Opus",
        limbs=0,
        coherence_threshold=0.0,
        batch_size=64,
        timeout_ms=60000,
        capabilities=["reasoning", "language", "planning"],
        default_parameters={},
        version="1.0",
        provider="anthropic",
    ),
}

# Fallback chain: preferred → fallback order
FALLBACK_CHAIN: List[str] = ["unified-stack", "gpt-4", "claude-3-opus"]

# Version aliases: model:variant → canonical model name
VERSION_ALIASES: Dict[str, str] = {
    "unified-stack:16-limb": "unified-stack-16",
    "unified-stack:8-limb": "unified-stack",
    "unified-stack:v1.0": "unified-stack",
    "unified-stack-16:v1.0": "unified-stack-16",
}


# ---------------------------------------------------------------------------
# Loader registry (lazy)
# ---------------------------------------------------------------------------

def _load_unified_stack(limb_count: int = 8) -> Any:
    """Lazy-load the UnifiedForwardModel."""
    module = importlib.import_module("unified.forward_model")
    cls = getattr(module, "UnifiedForwardModel")
    return cls(limb_count=limb_count)


_LOADERS: Dict[str, Callable[..., Any]] = {
    "unified-stack": lambda: _load_unified_stack(8),
    "unified-stack-16": lambda: _load_unified_stack(16),
    "gpt-4": lambda: None,       # external — no local object
    "claude-3-opus": lambda: None,
}


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Central registry for all available models.

    Supports:
    - Registration of custom models at runtime
    - Lazy loading (model loaded on first use)
    - Capability discovery
    - Fallback chain when preferred model is unavailable
    - ``model-name:variant`` / ``model-name:version`` syntax
    """

    def __init__(self) -> None:
        self._metadata: Dict[str, ModelMetadata] = dict(DEFAULT_MODELS)
        self._instances: Dict[str, ModelInstance] = {}
        self._loaders: Dict[str, Callable[..., Any]] = dict(_LOADERS)
        self._fallback_chain: List[str] = list(FALLBACK_CHAIN)
        self._version_aliases: Dict[str, str] = dict(VERSION_ALIASES)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        metadata: ModelMetadata,
        loader: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Register a model with optional custom loader."""
        self._metadata[metadata.name] = metadata
        if loader is not None:
            self._loaders[metadata.name] = loader
        else:
            self._loaders.setdefault(metadata.name, lambda: None)
        logger.debug("Registered model: %s", metadata.name)

    # ------------------------------------------------------------------
    # Name resolution
    # ------------------------------------------------------------------

    def resolve_name(self, spec: str) -> str:
        """Resolve a model spec (possibly ``name:variant``) to a canonical name.

        Raises ``ValueError`` if the spec is not recognised.
        """
        spec = spec.strip()

        # Exact match in aliases first
        if spec in self._version_aliases:
            return self._version_aliases[spec]

        # Exact match in metadata
        if spec in self._metadata:
            return spec

        # Pattern: name:version (e.g. unified-stack:v1.2)
        m = re.match(r"^([a-zA-Z0-9_-]+):(.+)$", spec)
        if m:
            base, variant = m.group(1), m.group(2)
            alias_key = f"{base}:{variant}"
            if alias_key in self._version_aliases:
                return self._version_aliases[alias_key]
            if base in self._metadata:
                logger.warning(
                    "Unknown variant '%s' for model '%s'; using base model.", variant, base
                )
                return base

        raise ValueError(f"Unknown model specification: '{spec}'")

    def is_available(self, name: str) -> bool:
        """Return True if the model name is registered."""
        try:
            canonical = self.resolve_name(name)
        except ValueError:
            return False
        return canonical in self._metadata

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, spec: str) -> Any:
        """Lazy-load and return the model instance for *spec*."""
        canonical = self.resolve_name(spec)
        if canonical in self._instances:
            return self._instances[canonical].instance

        loader = self._loaders.get(canonical)
        if loader is None:
            raise RuntimeError(f"No loader registered for model '{canonical}'")

        logger.info("Loading model '%s' …", canonical)
        t0 = time.monotonic()
        instance = loader()
        elapsed = (time.monotonic() - t0) * 1000

        self._instances[canonical] = ModelInstance(
            metadata=self._metadata[canonical],
            instance=instance,
            loaded_at=time.time(),
        )
        logger.info("Model '%s' loaded in %.1f ms", canonical, elapsed)
        return instance

    # ------------------------------------------------------------------
    # Metadata access
    # ------------------------------------------------------------------

    def get_metadata(self, spec: str) -> ModelMetadata:
        """Return metadata for *spec* (resolved)."""
        canonical = self.resolve_name(spec)
        return self._metadata[canonical]

    def list_models(self) -> List[ModelMetadata]:
        """Return metadata for all registered models."""
        return list(self._metadata.values())

    # ------------------------------------------------------------------
    # Capability discovery
    # ------------------------------------------------------------------

    def find_by_capability(self, capability: str) -> List[str]:
        """Return model names that support *capability*."""
        return [
            name
            for name, meta in self._metadata.items()
            if capability in meta.capabilities
        ]

    # ------------------------------------------------------------------
    # Fallback chain
    # ------------------------------------------------------------------

    def with_fallback(self, preferred: str) -> str:
        """Return *preferred* if available, else the first available fallback.

        Raises ``RuntimeError`` if no model in the chain is available.
        """
        try:
            canonical = self.resolve_name(preferred)
            if canonical in self._metadata:
                return canonical
        except ValueError:
            pass

        for fallback in self._fallback_chain:
            if fallback in self._metadata:
                logger.warning(
                    "Model '%s' not available; falling back to '%s'.", preferred, fallback
                )
                return fallback

        raise RuntimeError(
            f"No available model found for '{preferred}' and fallback chain is exhausted."
        )

    def set_fallback_chain(self, chain: List[str]) -> None:
        """Override the default fallback chain."""
        self._fallback_chain = list(chain)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Return the global model registry singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
