"""
Solver for ARC-AGI puzzle e5c44e8f.

Pattern: A rectangular spiral emanates from a center cell (value 3).
- Directions cycle: Up, Right, Down, Left
- Segment lengths: 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, ...
  (each length used for 2 consecutive segments, then +2)
- Cells with value 2 are obstacles: hitting one stops the entire spiral.
- Off-grid cells are skipped but the spiral continues with virtual position tracking.
"""

import json
from pathlib import Path


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find the center cell (value 3)
    cr, cc = -1, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                cr, cc = r, c
                break
        if cr != -1:
            break

    out = [row[:] for row in grid]

    # Spiral directions: Up, Right, Down, Left
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    pr, pc = cr, cc  # virtual cursor position
    dir_idx = 0
    length = 2
    stopped = False

    while length <= 100 and not stopped:
        for _ in range(2):  # two segments share the same length
            dr, dc = dirs[dir_idx % 4]
            for _ in range(length):
                pr += dr
                pc += dc
                if 0 <= pr < rows and 0 <= pc < cols:
                    if grid[pr][pc] == 2:
                        stopped = True
                        break
                    out[pr][pc] = 3
            if stopped:
                break
            dir_idx += 1
        length += 2

    return out


if __name__ == "__main__":
    data_path = Path.home() / "ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/e5c44e8f.json"
    with open(data_path) as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Training example {i}: PASS ✓")
        else:
            all_pass = False
            print(f"Training example {i}: FAIL ✗")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")

    if all_pass:
        print("\nAll training examples passed!")
        # Solve test
        for i, ex in enumerate(data["test"]):
            result = solve(ex["input"])
            print(f"\nTest {i} output:")
            for row in result:
                print(row)
    else:
        print("\nSome examples FAILED.")
