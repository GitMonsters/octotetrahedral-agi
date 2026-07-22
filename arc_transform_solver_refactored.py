#!/usr/bin/env python3
"""
Refactored Transform ARC Solver

Trait-based transform solver that applies geometric transformations to derive
ARC-AGI solutions.  This implementation provides a portable, repository-local
replacement for the previous developer-local module.
"""

from __future__ import annotations

from pathlib import Path
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

# Ensure the repo root is on sys.path so src.solver_abstractions resolves.
sys.path.insert(0, str(Path(__file__).parent))

from src.solver_abstractions import (
    BBoxTrait,
    GridUtils,
    Solver,
    TransformTrait,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concrete trait implementations
# ---------------------------------------------------------------------------

class _TransformTraitImpl(TransformTrait):
    def can_rotate(self) -> bool:
        return True

    def can_flip(self) -> bool:
        return True

    def can_scale(self) -> bool:
        return False

    def apply_transform(
        self, grid: List[List[int]], transform_type: str, **kwargs: Any
    ) -> List[List[int]]:
        ops = {
            "rotate_90": GridUtils.rotate_90,
            "rotate_180": GridUtils.rotate_180,
            "rotate_270": GridUtils.rotate_270,
            "flip_h": GridUtils.flip_horizontal,
            "flip_v": GridUtils.flip_vertical,
        }
        return ops.get(transform_type, lambda g: g)(grid)


class _BBoxTraitImpl(BBoxTrait):
    def extract_bounding_box(
        self, grid: List[List[int]]
    ) -> Tuple[int, int, int, int]:
        bg = GridUtils.find_background_color(grid)
        bbox = GridUtils.find_bounding_box(grid, bg)
        return bbox if bbox else (0, 0, 0, 0)

    def extract_region(
        self, grid: List[List[int]], bbox: Tuple[int, int, int, int]
    ) -> List[List[int]]:
        min_r, max_r, min_c, max_c = bbox
        return GridUtils.extract_region(grid, min_r, max_r, min_c, max_c)

    def detect_background_color(self, grid: List[List[int]]) -> int:
        return GridUtils.find_background_color(grid)


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------

class TransformSolverRefactored(Solver):
    """
    Trait-based transform solver for ARC-AGI.

    Explores all standard geometric transformations (rotations, flips) and
    selects the candidate that best matches training output patterns.
    """

    _TRANSFORMS = ("rotate_90", "rotate_180", "rotate_270", "flip_h", "flip_v")

    def __init__(self) -> None:
        super().__init__(name="TransformSolverRefactored", traits=[])
        self.traits["TransformTrait"] = _TransformTraitImpl()
        self.traits["BBoxTrait"] = _BBoxTraitImpl()

        self.transform_trait = self.get_trait(TransformTrait)
        self.bbox_trait = self.get_trait(BBoxTrait)

    def solve(self, task_data: Any) -> Optional[List[Any]]:
        """
        Solve a task dict with 'train' and 'test' keys.

        Returns a list of prediction dicts (one per test input), each with
        'attempt_1' and 'attempt_2' keys, or None on failure.
        """
        if not isinstance(task_data, dict):
            return None

        train = task_data.get("train", [])
        tests = task_data.get("test", [])

        if not train or not tests:
            return None

        # Determine target output shape from first training example.
        try:
            target = train[0].get("output", [])
            target_shape = (len(target), len(target[0])) if target and target[0] else (0, 0)
        except (IndexError, KeyError):
            target_shape = None

        predictions = []
        for test_example in tests:
            inp = test_example.get("input", [])
            if not inp:
                predictions.append({"attempt_1": [], "attempt_2": []})
                continue

            best: List[List[int]] = inp
            second_best: List[List[int]] = inp

            if self.transform_trait and target_shape:
                candidates: List[Tuple[List[List[int]], float]] = [(inp, 0.3)]
                for t_name in self._TRANSFORMS:
                    try:
                        t = self.transform_trait.apply_transform(inp, t_name)
                        t_shape = (len(t), len(t[0]) if t else 0)
                        score = 0.8 if t_shape == target_shape else 0.4
                        candidates.append((t, score))
                    except Exception as exc:
                        logger.debug("Transform %s failed: %s", t_name, exc)

                candidates.sort(key=lambda x: -x[1])
                best = candidates[0][0]
                second_best = candidates[1][0] if len(candidates) >= 2 else best

            predictions.append({"attempt_1": best, "attempt_2": second_best})

        return predictions


def main() -> None:
    """Smoke-test the transform solver."""
    solver = TransformSolverRefactored()
    task = {
        "train": [
            {
                "input": [[1, 2, 3], [4, 5, 6]],
                "output": [[4, 5, 6], [1, 2, 3]],
            }
        ],
        "test": [{"input": [[7, 8, 9], [0, 1, 2]]}],
    }
    print(f"Solver: {solver}")
    result = solver.solve(task)
    print(f"Predictions: {result}")


if __name__ == "__main__":
    main()
