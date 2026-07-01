"""RNA regulatory network controlling coupling and gate dynamics."""

from __future__ import annotations


class RNARegulatoryNetwork:
    """Task-adaptive regulator over modality coupling strengths."""

    def __init__(self) -> None:
        self._coupling = {
            "default": 0.35,
            "reasoning": 0.55,
            "language": 0.45,
            "spatial": 0.50,
        }

    def update_for_task(self, task_signal: str) -> None:
        """Adjust internal coupling priors for a task context."""
        key = task_signal.lower().strip() if task_signal else "default"
        if key not in self._coupling:
            self._coupling[key] = self._coupling["default"]

        for modality in list(self._coupling):
            if modality == key:
                self._coupling[modality] = min(0.9, self._coupling[modality] + 0.05)
            else:
                self._coupling[modality] = max(0.15, self._coupling[modality] - 0.01)

    def coupling_strength(self, task_signal: str | None = None) -> float:
        key = task_signal.lower().strip() if task_signal else "default"
        return self._coupling.get(key, self._coupling["default"])
