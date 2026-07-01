"""RNA splicing utilities for gate parameter adaptation."""

from __future__ import annotations


def splice_gate_parameters(phase: float, bias: float, adaptation_gain: float) -> tuple[float, float]:
    """Apply splicing gain to phase and bias gate parameters."""
    gain = max(0.2, min(2.0, adaptation_gain))
    return phase * gain, bias * (1.0 + (gain - 1.0) * 0.5)
