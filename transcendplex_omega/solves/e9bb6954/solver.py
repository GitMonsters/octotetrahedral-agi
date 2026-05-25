"""Solver for ARC-AGI puzzle e9bb6954.

Pattern: Each 3x3 monochrome block emits a cross (full row + full column)
through its center, using the block's color. At intersections of lines
from different blocks, the cell is set to 0.
"""

import json
import copy


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)

    # Find all 3x3 monochrome blocks by their center
    centers: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for r in range(rows - 2):
        for c in range(cols - 2):
            color = grid[r][c]
            if color == 0:
                continue
            if all(grid[r + dr][c + dc] == color for dr in range(3) for dc in range(3)):
                center = (r + 1, c + 1, color)
                if center not in seen:
                    seen.add(center)
                    centers.append(center)

    # Draw cross lines from each block center
    for cr, cc, color in centers:
        for c in range(cols):
            output[cr][c] = color
        for r in range(rows):
            output[r][cc] = color

    # At intersections of lines from different blocks, set cell to 0
    for i, (cr1, cc1, _) in enumerate(centers):
        for j, (cr2, cc2, _) in enumerate(centers):
            if i != j:
                output[cr1][cc2] = 0
                output[cr2][cc1] = 0

    return output


if __name__ == "__main__":
    import sys

    task_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/arc_task_e9bb6954.json"
    with open(task_path) as f:
        task = json.load(f)

    # Verify on training examples
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")

    # Solve test inputs
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"\nTest {i} output:")
        for row in result:
            print(row)
