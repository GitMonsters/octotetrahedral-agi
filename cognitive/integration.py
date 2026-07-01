"""Bidirectional cognitive integration with feedback-aware updates."""

from __future__ import annotations


def bidirectional_integrate(forward_state: list[float], backward_feedback: list[float]) -> list[float]:
    """Integrate top-down and bottom-up cognitive streams."""
    if len(forward_state) != len(backward_feedback):
        raise ValueError("forward_state and backward_feedback must have identical lengths")

    return [
        (feedforward * 0.6) + (feedback * 0.4)
        for feedforward, feedback in zip(forward_state, backward_feedback)
    ]
