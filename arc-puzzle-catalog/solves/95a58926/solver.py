"""
ARC task 95a58926 solver.

Pattern: Grid of 5s with a marker color (non-0, non-5) scattered as noise.
- Grid lines (rows/cols) are identified as those containing ONLY 5s and the marker color (no 0s).
- Output: clean grid where grid lines are 5, intersections are the marker color, rest is 0.
"""
import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])

    # Find the marker color (non-0, non-5)
    marker = 0
    for r in grid:
        for v in r:
            if v != 0 and v != 5:
                marker = v
                break
        if marker:
            break

    # A row is a grid line if every cell is 5 or marker (no 0s)
    h_lines = set()
    for r in range(rows):
        if all(grid[r][c] in (5, marker) for c in range(cols)):
            h_lines.add(r)

    # A col is a grid line if every cell is 5 or marker (no 0s)
    v_lines = set()
    for c in range(cols):
        if all(grid[r][c] in (5, marker) for r in range(rows)):
            v_lines.add(c)

    # Build output
    out = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r in h_lines and c in v_lines:
                out[r][c] = marker
            elif r in h_lines or c in v_lines:
                out[r][c] = 5
    return out


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/95a58926.json"))

    ok = True
    for i, p in enumerate(task["train"]):
        result = solve(p["input"])
        match = result == p["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            ok = False
            for r, (got, exp) in enumerate(zip(result, p["output"])):
                if got != exp:
                    print(f"  Row {r}: got {got}")
                    print(f"       exp {exp}")

    for i, p in enumerate(task["test"]):
        result = solve(p["input"])
        if "output" in p:
            match = result == p["output"]
            print(f"Test  {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                ok = False
        else:
            print(f"Test  {i}: produced {len(result)}x{len(result[0])} output")

    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")
