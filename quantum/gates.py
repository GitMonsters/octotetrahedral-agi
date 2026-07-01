"""Simple quantum-inspired gates for scalar limb states."""

from __future__ import annotations

import math


def apply_phase_gate(value: float, phase: float) -> float:
    """Rotate a scalar state with a phase-based cosine projection."""
    return value * math.cos(phase)


def apply_bias_gate(value: float, bias: float) -> float:
    """Shift a scalar state by a gate bias term."""
    return value + bias
