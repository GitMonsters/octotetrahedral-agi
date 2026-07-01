"""Unified tensor and gate operators for quantum limb coupling."""

from __future__ import annotations

from quantum.coherence import coherence_score, synchronize
from quantum.gates import apply_bias_gate, apply_phase_gate


def tensor_decompose(limb_states: list[float]) -> tuple[float, list[float]]:
    """Decompose limb states into shared component and residuals."""
    if not limb_states:
        return 0.0, []

    shared = sum(limb_states) / len(limb_states)
    residuals = [state - shared for state in limb_states]
    return shared, residuals


def apply_unified_quantum_operator(
    limb_states: list[float],
    phase: float,
    bias: float,
    coupling_strength: float,
) -> tuple[list[float], float]:
    """Apply coherence-preserving quantum transforms across all limbs."""
    gated = [apply_bias_gate(apply_phase_gate(value, phase), bias) for value in limb_states]
    coherent = synchronize(gated, coupling_strength)
    return coherent, coherence_score(coherent)
