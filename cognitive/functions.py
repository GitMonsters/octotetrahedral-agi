"""Cognitive helper functions aligned with the unified architecture."""

from __future__ import annotations


def aggregate_cognitive_state(limb_states: list[float], feedback_states: list[float]) -> list[float]:
    """Blend limb outputs with synchronized feedback."""
    if not limb_states:
        return []

    if len(limb_states) != len(feedback_states):
        raise ValueError("limb_states and feedback_states must have identical lengths")

    return [state * 0.7 + feedback * 0.3 for state, feedback in zip(limb_states, feedback_states)]


def select_action_channel(unified_state: list[float]) -> int:
    """Return the strongest action channel index."""
    if not unified_state:
        return 0
    return max(range(len(unified_state)), key=unified_state.__getitem__)
