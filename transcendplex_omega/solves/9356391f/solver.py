"""Solver for ARC-AGI task 9356391f.

Pattern: Row 0 contains a color legend (innermost→outermost ring colors).
Row 1 is a separator of 5s. Below that, a single colored pixel marks the center.
Output draws concentric square rings (Chebyshev distance) around the center,
colored by the legend sequence.
"""

import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Extract legend from row 0: values from index 0 to last non-zero index
    last_nz = -1
    for i in range(cols - 1, -1, -1):
        if grid[0][i] != 0:
            last_nz = i
            break
    legend = list(grid[0][: last_nz + 1]) if last_nz >= 0 else []

    # Find the single non-zero pixel below row 1 (the center)
    cr, cc = -1, -1
    for r in range(2, rows):
        for c in range(cols):
            if grid[r][c] != 0:
                cr, cc = r, c
                break
        if cr >= 0:
            break

    # Draw concentric Chebyshev-distance rings below row 1
    for r in range(2, rows):
        for c in range(cols):
            dist = max(abs(r - cr), abs(c - cc))
            if dist < len(legend):
                out[r][c] = legend[dist]
            else:
                out[r][c] = 0

    # Row 0 fix: replace isolated single non-zero values (size-1 groups) with 5
    i = 0
    while i <= last_nz:
        if grid[0][i] != 0:
            start = i
            while i <= last_nz and grid[0][i] != 0:
                i += 1
            if i - start == 1:  # single isolated value
                out[0][start] = 5
        else:
            i += 1

    return out


if __name__ == "__main__":
    task = json.load(
        open(
            "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/9356391f.json"
        )
    )

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        match = result == pair["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(result)):
                if result[r] != pair["output"][r]:
                    print(f"  Row {r}: got    {result[r]}")
                    print(f"  Row {r}: expect {pair['output'][r]}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            match = result == pair["output"]
            print(f"Test  {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
                for r in range(len(result)):
                    if result[r] != pair["output"][r]:
                        print(f"  Row {r}: got    {result[r]}")
                        print(f"  Row {r}: expect {pair['output'][r]}")
        else:
            print(f"Test  {i}: (no expected output)")

    print(f"\nAll pass: {all_pass}")
