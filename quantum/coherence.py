"""Quantum coherence helpers for unified limb synchronization."""

from __future__ import annotations


def coherence_score(values: list[float]) -> float:
    """Return a bounded coherence score where 1.0 is perfect coherence."""
    if not values:
        return 1.0

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return 1.0 / (1.0 + variance)


def synchronize(values: list[float], strength: float) -> list[float]:
    """Blend values toward a shared quantum manifold."""
    if not values:
        return []

    strength = max(0.0, min(1.0, strength))
    mean_value = sum(values) / len(values)
    return [value * (1.0 - strength) + mean_value * strength for value in values]
