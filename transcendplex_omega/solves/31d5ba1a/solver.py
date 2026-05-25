"""
Task 31d5ba1a: XOR of two halves.

The input is a 6×5 grid split into two 3×5 halves:
  - Top half uses 9 for "on" and 0 for "off"
  - Bottom half uses 4 for "on" and 0 for "off"

The output is a 3×5 grid where each cell is 6 if exactly one of the
corresponding top/bottom cells is "on" (XOR), and 0 otherwise.
"""
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    half = len(grid) // 2
    cols = len(grid[0])
    result = []
    for r in range(half):
        row = []
        for c in range(cols):
            top_on = grid[r][c] != 0
            bot_on = grid[r + half][c] != 0
            row.append(6 if top_on ^ bot_on else 0)
        result.append(row)
    return result


if __name__ == "__main__":
    import json, pathlib

    task_path = pathlib.Path(__file__).resolve().parents[2] / "dataset" / "tasks" / "31d5ba1a.json"
    with open(task_path) as f:
        task = json.load(f)

    for split in ("train", "test"):
        for i, ex in enumerate(task[split]):
            result = solve(ex["input"])
            status = "PASS" if result == ex["output"] else "FAIL"
            print(f"{split} {i}: {status}")
            if status == "FAIL":
                print(f"  Expected: {ex['output']}")
                print(f"  Got:      {result}")
