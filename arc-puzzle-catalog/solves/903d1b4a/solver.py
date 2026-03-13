"""Solver for ARC-AGI task 903d1b4a.

Pattern: The grid has D2 symmetry (horizontal + vertical reflection).
Color 3 marks corrupted cells. Replace each 3 with the value from
a non-corrupted symmetric counterpart.
"""

import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    n = len(grid)
    m = len(grid[0])
    result = [row[:] for row in grid]

    for r in range(n):
        for c in range(m):
            if grid[r][c] == 3:
                # D2 symmetry mirrors
                mirrors = [
                    (n - 1 - r, m - 1 - c),  # 180° rotation
                    (r, m - 1 - c),            # horizontal reflection
                    (n - 1 - r, c),            # vertical reflection
                ]
                for mr, mc in mirrors:
                    if grid[mr][mc] != 3:
                        result[r][c] = grid[mr][mc]
                        break

    return result


def main():
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/903d1b4a.json") as f:
        task = json.load(f)

    all_pass = True

    for i, pair in enumerate(task["train"]):
        predicted = solve(pair["input"])
        expected = pair["output"]
        match = predicted == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if predicted[r][c] != expected[r][c]:
                        print(f"  Diff at ({r},{c}): got {predicted[r][c]}, expected {expected[r][c]}")

    for i, pair in enumerate(task["test"]):
        predicted = solve(pair["input"])
        if "output" in pair:
            expected = pair["output"]
            match = predicted == expected
            print(f"Test {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
                for r in range(len(expected)):
                    for c in range(len(expected[0])):
                        if predicted[r][c] != expected[r][c]:
                            print(f"  Diff at ({r},{c}): got {predicted[r][c]}, expected {expected[r][c]}")
        else:
            print(f"Test {i}: output produced (no ground truth to verify)")
            for row in predicted:
                print(row)

    print(f"\nAll train passed: {all_pass}")


if __name__ == "__main__":
    main()
