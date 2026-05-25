"""
ARC-AGI puzzle f83cb3f6 solver.

Rule: A line of 8s (horizontal or vertical, possibly with gaps) divides the grid.
For each 8-cell on the line, look perpendicular in both directions.
If any colored (non-0, non-8) cell exists in that direction, place a colored cell
adjacent to the 8 on that side. Everything else becomes 0.
"""
import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])

    # Find all 8-positions
    eights = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
    eight_rows = set(r for r, c in eights)
    eight_cols = set(c for r, c in eights)

    # Determine the non-background color
    color = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in (0, 8):
                color = grid[r][c]
                break
        if color:
            break

    # Build output (all zeros, same size)
    out = [[0] * cols for _ in range(rows)]

    # Place the 8-line
    for r, c in eights:
        out[r][c] = 8

    if len(eight_rows) == 1:
        # Horizontal line at row R
        R = list(eight_rows)[0]
        for c in range(cols):
            if grid[R][c] != 8:
                continue
            # Check above (rows 0..R-1)
            if any(grid[r][c] not in (0, 8) for r in range(R)):
                out[R - 1][c] = color
            # Check below (rows R+1..end)
            if any(grid[r][c] not in (0, 8) for r in range(R + 1, rows)):
                out[R + 1][c] = color
    else:
        # Vertical line at col C
        C = list(eight_cols)[0]
        for r in range(rows):
            if grid[r][C] != 8:
                continue
            # Check left (cols 0..C-1)
            if any(grid[r][c] not in (0, 8) for c in range(C)):
                out[r][C - 1] = color
            # Check right (cols C+1..end)
            if any(grid[r][c] not in (0, 8) for c in range(C + 1, cols)):
                out[r][C + 1] = color

    return out


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/f83cb3f6.json"))
    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        match = result == ex["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(result)):
                if result[r] != ex["output"][r]:
                    print(f"  Row {r}: got {result[r]}")
                    print(f"       exp {ex['output'][r]}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i} output: {result}")
        if "output" in ex:
            match = result == ex["output"]
            print(f"Test {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
