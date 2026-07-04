#!/usr/bin/env python3
"""
Refactored Compound ARC Solver

Trait-based compound solver that layers multiple solving strategies with
ensemble voting.  This implementation provides a portable, repository-local
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
    CompoundTrait,
    GridUtils,
    Solver,
    TransformTrait,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concrete trait implementations
# ---------------------------------------------------------------------------

class _CompoundTraitImpl(CompoundTrait):
    def __init__(self) -> None:
        self.subsolvers: List[Tuple[Solver, int]] = []

    def add_subsolver(self, solver: Solver, priority: int = 0) -> None:
        if solver is None:
            raise ValueError("solver cannot be None")
        self.subsolvers.append((solver, priority))

    def compose_solutions(
        self, solutions: List[Tuple[List[List[int]], float]]
    ) -> List[List[int]]:
        if not solutions:
            return [[]]
        best_grid, _ = max(solutions, key=lambda x: x[1])
        return best_grid


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

class CompoundArcSolverRefactored(Solver):
    """
    Trait-based compound solver for ARC-AGI.

    Combines CompoundTrait, TransformTrait, and BBoxTrait to apply multiple
    transformation strategies and vote on the best candidate output.
    """

    def __init__(self) -> None:
        super().__init__(name="CompoundArcSolverRefactored", traits=[])
        self.traits["CompoundTrait"] = _CompoundTraitImpl()
        self.traits["TransformTrait"] = _TransformTraitImpl()
        self.traits["BBoxTrait"] = _BBoxTraitImpl()

        self.compound_trait = self.get_trait(CompoundTrait)
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

        predictions = []
        for test_example in tests:
            inp = test_example.get("input", [])
            if not inp:
                predictions.append({"attempt_1": [], "attempt_2": []})
                continue

            candidates: List[Tuple[List[List[int]], float]] = [(inp, 0.3)]

            try:
                bg = GridUtils.find_background_color(inp)
                bbox = GridUtils.find_bounding_box(inp, bg)
                if bbox:
                    extracted = GridUtils.extract_region(
                        inp, bbox[0], bbox[1], bbox[2], bbox[3]
                    )
                    if extracted and extracted != inp:
                        candidates.append((extracted, 0.5))
            except Exception as exc:
                logger.debug("BBox strategy failed: %s", exc)

            for transform in ("rotate_90", "rotate_180", "flip_h", "flip_v"):
                try:
                    if self.transform_trait:
                        t = self.transform_trait.apply_transform(inp, transform)
                        if t and t != inp:
                            candidates.append((t, 0.4))
                except Exception as exc:
                    logger.debug("Transform %s failed: %s", transform, exc)

            if self.compound_trait and len(candidates) > 1:
                best = self.compound_trait.compose_solutions(candidates)
            else:
                best = candidates[0][0]

            attempt_2 = candidates[1][0] if len(candidates) >= 2 else best
            predictions.append({"attempt_1": best, "attempt_2": attempt_2})

        return predictions


def main() -> None:
    """Smoke-test the compound solver."""
    solver = CompoundArcSolverRefactored()
    task = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[4, 3], [2, 1]],
            }
        ],
        "test": [{"input": [[1, 1], [2, 2]]}],
    }
    print(f"Solver: {solver}")
    result = solver.solve(task)
    print(f"Predictions: {result}")


if __name__ == "__main__":
    main()
