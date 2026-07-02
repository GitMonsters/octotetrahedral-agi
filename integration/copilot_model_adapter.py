"""Adapter pattern: Copilot request/response ↔ unified model format.

Converts Copilot-style requests (prompt + context dict) into the
8-dimensional limb-state + task-signal format expected by UnifiedForwardModel,
and converts the unified model output back into a Copilot-compatible response
dict that preserves coherence and limb metadata.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response type aliases
# ---------------------------------------------------------------------------

CopilotRequest = Dict[str, Any]
"""
Expected keys (all optional except ``prompt``):
  prompt          str   — user / task prompt
  context         dict  — optional additional context
  model           str   — model specification override
  trace_id        str   — request trace identifier
  session_id      str   — session identifier
"""

CopilotResponse = Dict[str, Any]
"""
Returned keys:
  model           str   — canonical model name used
  response        Any   — model output (list[float] for unified, str for external)
  coherence       float — coherence score (0.0 for external models)
  limb_metadata   dict  — per-limb activation data (empty for external models)
  trace_id        str   — echoed from request
  latency_ms      float — total adapter latency
  action_channel  int   — dominant action channel (−1 for external)
"""


# ---------------------------------------------------------------------------
# Helper: encode a text prompt as limb activation states
# ---------------------------------------------------------------------------

def _prompt_to_limb_states(prompt: str, limb_count: int = 8) -> List[float]:
    """Deterministically map *prompt* → list of *limb_count* floats in [0, 1]."""
    digest = hashlib.sha256(prompt.encode()).digest()
    activations: List[float] = []
    for i in range(limb_count):
        byte_val = digest[i % len(digest)]
        activations.append(byte_val / 255.0)
    return activations


# ---------------------------------------------------------------------------
# CopilotModelAdapter
# ---------------------------------------------------------------------------

class CopilotModelAdapter:
    """Adapts Copilot requests to the unified model and back.

    Parameters
    ----------
    registry :
        Model registry to resolve and load models from.  When *None* the
        global singleton registry is used.
    default_model :
        Model specification to use when the request does not specify one.
    """

    def __init__(
        self,
        registry: Optional[Any] = None,
        default_model: str = "gpt-4",
    ) -> None:
        """Create the adapter.

        Parameters
        ----------
        registry :
            Model registry to use.  Defaults to the global singleton.
        default_model :
            Model spec used when a request does not include a ``model`` key.
            Defaults to ``"gpt-4"`` for backward compatibility (no breaking
            changes).  Projects should set ``default_model`` in
            ``.copilot/config.yml`` to override this at the integration level.
        """
        if registry is None:
            from model_registry import get_registry
            registry = get_registry()
        self._registry = registry
        self._default_model = default_model

    # ------------------------------------------------------------------
    # Request conversion
    # ------------------------------------------------------------------

    def to_unified_format(
        self, request: CopilotRequest, limb_count: int = 8
    ) -> Dict[str, Any]:
        """Convert a Copilot request to unified model input format.

        Returns a dict with keys:
          limb_states : list[float]
          task_signal : str
          trace_id    : str | None
        """
        prompt: str = request.get("prompt", "")
        context: Dict[str, Any] = request.get("context", {})

        # Enrich prompt with context keys for better signal diversity
        combined = prompt
        if context:
            combined += " " + " ".join(str(v) for v in context.values())

        limb_states = _prompt_to_limb_states(combined, limb_count)
        task_signal = prompt[:64] if prompt else "generic"

        return {
            "limb_states": limb_states,
            "task_signal": task_signal,
            "trace_id": request.get("trace_id"),
        }

    # ------------------------------------------------------------------
    # Response conversion
    # ------------------------------------------------------------------

    def to_copilot_format(
        self,
        unified_result: Dict[str, Any],
        request: CopilotRequest,
        model_name: str,
        latency_ms: float = 0.0,
    ) -> CopilotResponse:
        """Convert unified model output to a Copilot-compatible response."""
        limb_states: List[float] = unified_result.get("limb_states", [])
        coherence: float = unified_result.get("coherence", 0.0)
        action_channel: int = unified_result.get("action_channel", -1)

        limb_metadata: Dict[str, Any] = {}
        if limb_states:
            limb_metadata = {
                "limb_states": limb_states,
                "active_limbs": sum(1 for v in limb_states if v > 0.5),
                "dominant_limb": action_channel,
                "coupling_strength": unified_result.get("coupling_strength", 0.0),
                "phase": unified_result.get("phase", 0.0),
                "bias": unified_result.get("bias", 0.0),
            }

        return {
            "model": model_name,
            "response": limb_states or unified_result.get("response", ""),
            "coherence": coherence,
            "limb_metadata": limb_metadata,
            "trace_id": request.get("trace_id"),
            "session_id": request.get("session_id"),
            "latency_ms": latency_ms,
            "action_channel": action_channel,
        }

    # ------------------------------------------------------------------
    # End-to-end processing
    # ------------------------------------------------------------------

    def process(self, request: CopilotRequest) -> CopilotResponse:
        """Process a Copilot request end-to-end through the selected model.

        Handles:
        - model resolution (including fallback)
        - request conversion
        - model invocation
        - response conversion
        - error handling + graceful degradation
        """
        t0 = time.monotonic()
        model_spec: str = request.get("model") or self._default_model

        try:
            canonical = self._registry.with_fallback(model_spec)
        except (ValueError, RuntimeError) as exc:
            logger.error("Model resolution failed: %s", exc)
            return self._error_response(request, str(exc), time.monotonic() - t0)

        try:
            meta = self._registry.get_metadata(canonical)
            limb_count = meta.limbs if meta.limbs > 0 else 8
            unified_input = self.to_unified_format(request, limb_count)

            model_obj = self._registry.load(canonical)
            if model_obj is not None and hasattr(model_obj, "forward"):
                forward_result = model_obj.forward(
                    unified_input["limb_states"],
                    task_signal=unified_input["task_signal"],
                )
            else:
                # External / stub model
                forward_result = {
                    "limb_states": [],
                    "coherence": 0.0,
                    "action_channel": -1,
                    "response": f"[{canonical}] {request.get('prompt', '')}",
                }

            latency_ms = (time.monotonic() - t0) * 1000
            return self.to_copilot_format(forward_result, request, canonical, latency_ms)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference error for model '%s': %s", canonical, exc)
            return self._error_response(request, str(exc), time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Error response helper
    # ------------------------------------------------------------------

    @staticmethod
    def _error_response(
        request: CopilotRequest,
        error: str,
        elapsed_s: float,
    ) -> CopilotResponse:
        return {
            "model": request.get("model", "unknown"),
            "response": None,
            "coherence": 0.0,
            "limb_metadata": {},
            "trace_id": request.get("trace_id"),
            "session_id": request.get("session_id"),
            "latency_ms": elapsed_s * 1000,
            "action_channel": -1,
            "error": error,
        }
