"""Solver for ARC-AGI task 929ab4e9.

The grid has D4 symmetry (full square symmetry: 4 rotations + 4 reflections).
A rectangular region of 2s masks part of the pattern.
Restore masked cells using their symmetric counterparts.
"""

from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    N = len(grid)
    result = [row[:] for row in grid]

    for r in range(N):
        for c in range(N):
            if result[r][c] == 2:
                M = N - 1
                # All 8 D4-symmetric positions
                mirrors = [
                    (r, M - c),         # horizontal flip
                    (M - r, c),         # vertical flip
                    (M - r, M - c),     # 180° rotation
                    (c, M - r),         # 90° CW rotation
                    (M - c, r),         # 270° CW rotation
                    (c, r),             # main diagonal reflection
                    (M - c, M - r),     # anti-diagonal reflection
                ]
                for mr, mc in mirrors:
                    if grid[mr][mc] != 2:
                        result[r][c] = grid[mr][mc]
                        break

    return result


if __name__ == "__main__":
    import json, sys

    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/929ab4e9.json"))

    all_pass = True
    for i, pair in enumerate(task["train"]):
        output = solve(pair["input"])
        match = output == pair["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(output)):
                for c in range(len(output[0])):
                    if output[r][c] != pair["output"][r][c]:
                        print(f"  diff at ({r},{c}): got {output[r][c]}, expected {pair['output'][r][c]}")

    for i, pair in enumerate(task["test"]):
        output = solve(pair["input"])
        if "output" in pair:
            match = output == pair["output"]
            print(f"Test  {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
        else:
            has_2 = any(2 in row for row in output)
            print(f"Test  {i}: produced (no 2s remaining: {not has_2})")

    print(f"\nAll passed: {all_pass}")
