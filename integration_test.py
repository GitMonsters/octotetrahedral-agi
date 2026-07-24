"""
integration_test.py — pytest integration tests for ARC task a32d8b75 solver.

Verifies that the solver produces the exact expected output for every
training example in the official task JSON, and that test-case predictions
are structurally valid grids.

Run:
    python -m pytest integration_test.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

Grid = List[List[int]]

_REPO_ROOT = Path(__file__).resolve().parent
_TASK_JSON = _REPO_ROOT / "arc-puzzle-catalog" / "dataset" / "tasks" / "a32d8b75.json"


@pytest.fixture(scope="module")
def task_data() -> Dict[str, Any]:
    with open(_TASK_JSON) as fh:
        return json.load(fh)


def _is_valid_grid(grid: Any) -> bool:
    """Return True if *grid* is a non-empty list of equal-length int rows."""
    if not isinstance(grid, list) or len(grid) == 0:
        return False
    # grid is confirmed non-empty here, so grid[0] is safe to access
    row_len = len(grid[0])
    if row_len == 0:
        return False
    return all(
        isinstance(row, list) and len(row) == row_len and all(isinstance(v, int) for v in row)
        for row in grid
    )


# ---------------------------------------------------------------------------
# Training-example tests
# ---------------------------------------------------------------------------

class TestTrainingExamples:
    """The solver must exactly reproduce every training output."""

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_training_exact_match(self, task_data: Dict[str, Any], idx: int) -> None:
        from arc_task_a32d8b75_solver import solve  # type: ignore

        examples = task_data["train"]
        assert idx < len(examples), f"Training example {idx} does not exist"

        ex = examples[idx]
        actual = solve(ex["input"])
        expected = ex["output"]

        assert actual == expected, (
            f"Training example {idx}: output mismatch.\n"
            f"  Expected shape: {len(expected)}×{len(expected[0])}\n"
            f"  Actual shape:   {len(actual)}×{len(actual[0])}\n"
            f"  Differing cells: "
            + str(
                sum(
                    1
                    for r in range(len(expected))
                    for c in range(len(expected[0]))
                    if r >= len(actual)
                    or c >= len(actual[0])
                    or actual[r][c] != expected[r][c]
                )
            )
        )

    def test_all_training_examples_pass(self, task_data: Dict[str, Any]) -> None:
        """Convenience test – fails if any single training example fails."""
        from arc_task_a32d8b75_solver import solve  # type: ignore

        failures = []
        for i, ex in enumerate(task_data["train"]):
            actual = solve(ex["input"])
            if actual != ex["output"]:
                failures.append(i)

        assert not failures, f"Training examples failed: {failures}"


# ---------------------------------------------------------------------------
# Test-case structure tests
# ---------------------------------------------------------------------------

class TestTestCasePredictions:
    """Solver must return valid grids for the test inputs."""

    @pytest.mark.parametrize("idx", [0, 1])
    def test_prediction_is_valid_grid(self, task_data: Dict[str, Any], idx: int) -> None:
        from arc_task_a32d8b75_solver import solve  # type: ignore

        test_cases = task_data["test"]
        assert idx < len(test_cases), f"Test case {idx} does not exist"

        result = solve(test_cases[idx]["input"])
        assert _is_valid_grid(result), f"Test {idx}: solver returned an invalid grid"

    def test_prediction_values_in_arc_range(self, task_data: Dict[str, Any]) -> None:
        """All predicted cell values must be integers 0-9."""
        from arc_task_a32d8b75_solver import solve  # type: ignore

        for i, ex in enumerate(task_data["test"]):
            result = solve(ex["input"])
            bad = [
                (r, c, v)
                for r, row in enumerate(result)
                for c, v in enumerate(row)
                if not (0 <= v <= 9)
            ]
            assert not bad, f"Test {i}: out-of-range values {bad[:5]}"


# ---------------------------------------------------------------------------
# Solver unit tests
# ---------------------------------------------------------------------------

class TestSolverHelpers:
    """Unit tests for internal helpers."""

    def test_rotate_grid_identity(self) -> None:
        from arc_task_a32d8b75_solver import _rotate_grid  # type: ignore

        g = [[1, 2], [3, 4]]
        assert _rotate_grid(g, 0) == g
        assert _rotate_grid(g, 4) == g  # 4 × 90° = identity

    def test_rotate_grid_90(self) -> None:
        from arc_task_a32d8b75_solver import _rotate_grid  # type: ignore

        g = [[1, 2, 3], [4, 5, 6]]
        rotated = _rotate_grid(g, 1)
        # 90° CCW of [[1,2,3],[4,5,6]] → [[3,6],[2,5],[1,4]]
        assert rotated == [[3, 6], [2, 5], [1, 4]]

    def test_solve_returns_list_of_lists(self, task_data: Dict[str, Any]) -> None:
        from arc_task_a32d8b75_solver import solve  # type: ignore

        for i, ex in enumerate(task_data["train"]):
            result = solve(ex["input"])
            assert isinstance(result, list), f"Train {i}: result is not a list"
            assert all(isinstance(row, list) for row in result), (
                f"Train {i}: result rows are not lists"
            )
