"""
Solver for ARC task 8b28cd80.

Rule: The 3x3 input has exactly one non-zero cell at (r, c) with color value.
Map to a 9x9 source position (sr, sc) = (r*4, c*4).
Each output cell is colored based on Chebyshev distance from the source,
with a spiral seam diagonal that flips the parity.
"""

import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    # Find the non-zero cell
    sr, sc, color = 0, 0, 0
    for r in range(3):
        for c in range(3):
            if grid[r][c] != 0:
                sr, sc, color = r * 4, c * 4, grid[r][c]

    out = [[0] * 9 for _ in range(9)]
    for i in range(9):
        for j in range(9):
            d = max(abs(i - sr), abs(j - sc))
            # Spiral seam: a diagonal line above the source row
            on_seam = (j - i == sc - sr + 1) and (i < sr)
            if on_seam:
                out[i][j] = color if d % 2 == 1 else 0
            else:
                out[i][j] = color if d % 2 == 0 else 0
    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/8b28cd80.json") as f:
        task = json.load(f)

    for idx, example in enumerate(task["train"]):
        result = solve(example["input"])
        expected = example["output"]
        status = "PASS" if result == expected else "FAIL"
        print(f"Train {idx}: {status}")
        if result != expected:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")

    for idx, example in enumerate(task["test"]):
        result = solve(example["input"])
        expected = example["output"]
        status = "PASS" if result == expected else "FAIL"
        print(f"Test  {idx}: {status}")
        if result != expected:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
