"""
ARC-AGI Task ca8f78db Solver

The grid contains a repeating 2D tile pattern with rectangular regions
of 0s punched out. The transformation restores the original tile pattern
by filling all 0-holes with the correct repeating values.

Algorithm:
1. Search for the smallest tile (pr × pc) consistent with all non-zero cells
2. Fill every 0 cell with tile[r % pr][c % pc]
"""

from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])

    for pr in range(1, rows + 1):
        for pc in range(1, cols + 1):
            tile = [[None] * pc for _ in range(pr)]
            consistent = True
            for r in range(rows):
                if not consistent:
                    break
                for c in range(cols):
                    v = grid[r][c]
                    if v == 0:
                        continue
                    tr, tc = r % pr, c % pc
                    if tile[tr][tc] is None:
                        tile[tr][tc] = v
                    elif tile[tr][tc] != v:
                        consistent = False
                        break

            if consistent and all(
                tile[r][c] is not None for r in range(pr) for c in range(pc)
            ):
                result = [row[:] for row in grid]
                for r in range(rows):
                    for c in range(cols):
                        if result[r][c] == 0:
                            result[r][c] = tile[r % pr][c % pc]
                return result

    return [row[:] for row in grid]


if __name__ == "__main__":
    import json

    path = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/ca8f78db.json"
    with open(path) as f:
        task = json.load(f)

    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")

    test_result = solve(task["test"][0]["input"])
    zeros = sum(1 for row in test_result for v in row if v == 0)
    print(f"Test: {len(test_result)}x{len(test_result[0])}, zeros remaining: {zeros}")
