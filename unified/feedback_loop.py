"""Cross-modal feedback integration with 8-limb coupling."""

from __future__ import annotations

from quantum.coherence import synchronize


class UnifiedFeedbackLoop:
    """Feedback loop that synchronizes all limbs through quantum coupling."""

    def __init__(self, limb_count: int = 8) -> None:
        self.limb_count = limb_count

    def integrate(self, limb_states: list[float], coupling_strength: float) -> list[float]:
        if len(limb_states) != self.limb_count:
            raise ValueError(f"expected {self.limb_count} limb states, got {len(limb_states)}")

        return synchronize(limb_states, coupling_strength)
