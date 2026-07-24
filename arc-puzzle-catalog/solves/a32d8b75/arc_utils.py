from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TASK_ID = "a32d8b75"
TASK_PATH = Path(__file__).resolve().parents[2] / "dataset" / "tasks" / f"{TASK_ID}.json"


def load_task(task_path: str | Path = TASK_PATH) -> dict[str, Any]:
    path = Path(task_path)
    with path.open() as handle:
        return json.load(handle)


def ensure_rectangular(grid: list[list[int]]) -> None:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid must be rectangular")


def count_differences(actual: list[list[int]], expected: list[list[int]]) -> int:
    if len(actual) != len(expected):
        return -1
    if actual and expected and len(actual[0]) != len(expected[0]):
        return -1
    return sum(
        1
        for row_a, row_b in zip(actual, expected)
        for value_a, value_b in zip(row_a, row_b)
        if value_a != value_b
    )


def save_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(payload, handle, indent=2)
    return path
