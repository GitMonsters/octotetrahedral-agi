"""Unified forward model with tensor decomposition, quantum gates, and RNA adaptation."""

from __future__ import annotations

from cognitive.functions import aggregate_cognitive_state
from cognitive.integration import bidirectional_integrate
from quantum.operators import apply_unified_quantum_operator, tensor_decompose
from rna.adaptation import adapt_for_task
from rna.regulatory import RNARegulatoryNetwork
from unified.feedback_loop import UnifiedFeedbackLoop
from unified.state_transitions import UnifiedStateTransitions


class UnifiedForwardModel:
    """Coherent quantum-biological model coordinating 8 cognitive limbs."""

    def __init__(self, limb_count: int = 8) -> None:
        self.limb_count = limb_count
        self.feedback_loop = UnifiedFeedbackLoop(limb_count=limb_count)
        self.state_transitions = UnifiedStateTransitions()
        self.regulatory_network = RNARegulatoryNetwork()

    def forward(self, limb_states: list[float], task_signal: str | None = None) -> dict[str, object]:
        if len(limb_states) != self.limb_count:
            raise ValueError(f"expected {self.limb_count} limb states, got {len(limb_states)}")

        shared_component, residuals = tensor_decompose(limb_states)
        adaptation = adapt_for_task(self.regulatory_network, task_signal)

        quantum_states, coherence = apply_unified_quantum_operator(
            limb_states,
            phase=adaptation["phase"],
            bias=adaptation["bias"],
            coupling_strength=adaptation["coupling_strength"],
        )

        feedback = self.feedback_loop.integrate(quantum_states, adaptation["coupling_strength"])
        quantum_evolved = self.state_transitions.evolve_quantum(feedback, adaptation["phase"])
        biological_state = self.state_transitions.adapt_biological(
            quantum_evolved, adaptation["coupling_strength"]
        )
        cognitive_state = aggregate_cognitive_state(biological_state, feedback)
        unified_state = bidirectional_integrate(cognitive_state, feedback)

        return {
            "limb_states": unified_state,
            "shared_component": shared_component,
            "residuals": residuals,
            "coherence": coherence,
            "coupling_strength": adaptation["coupling_strength"],
            "phase": adaptation["phase"],
            "bias": adaptation["bias"],
        }


class LegacyForwardAdapter:
    """Backward-compatible adapter for legacy modular callers."""

    def __init__(self, limb_count: int = 8) -> None:
        self._model = UnifiedForwardModel(limb_count=limb_count)

    def run(self, limb_states: list[float], task_type: str | None = None) -> list[float]:
        """Legacy API: returns only the integrated limb state list."""
        return self._model.forward(limb_states, task_signal=task_type)["limb_states"]
