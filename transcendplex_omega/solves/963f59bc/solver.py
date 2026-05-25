"""
ARC-AGI Puzzle 963f59bc Solver

Rule: The grid contains a blue (1) shape and one or more colored dots.
Each dot shares exactly one row or column with a single blue cell (the "anchor").
- If same row: reflect the blue shape horizontally about the anchor, place at dot with dot's color.
- If same column: reflect vertically about the anchor, place at dot with dot's color.
"""
import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    # Find blue (1) cells
    blue_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                blue_cells.add((r, c))

    # Find colored dots (non-0, non-1)
    dots = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in (0, 1):
                dots.append((r, c, grid[r][c]))

    # Index blue cells by row and column
    blue_by_row: dict[int, list] = {}
    blue_by_col: dict[int, list] = {}
    for r, c in blue_cells:
        blue_by_row.setdefault(r, []).append((r, c))
        blue_by_col.setdefault(c, []).append((r, c))

    for dot_r, dot_c, color in dots:
        if dot_r in blue_by_row:
            # Anchor is the blue cell on the same row; reflect horizontally
            anchor_r, anchor_c = blue_by_row[dot_r][0]
            for br, bc in blue_cells:
                dr = br - anchor_r
                dc = bc - anchor_c
                nr, nc = dot_r + dr, dot_c - dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    output[nr][nc] = color
        elif dot_c in blue_by_col:
            # Anchor is the blue cell on the same column; reflect vertically
            anchor_r, anchor_c = blue_by_col[dot_c][0]
            for br, bc in blue_cells:
                dr = br - anchor_r
                dc = bc - anchor_c
                nr, nc = dot_r - dr, dot_c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    output[nr][nc] = color

    return output


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/963f59bc.json"))

    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Train {i}: PASS ✓")
        else:
            all_pass = False
            print(f"Train {i}: FAIL ✗")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")

    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        if "output" in ex:
            expected = ex["output"]
            if result == expected:
                print(f"Test  {i}: PASS ✓")
            else:
                all_pass = False
                print(f"Test  {i}: FAIL ✗")
                for r in range(len(expected)):
                    for c in range(len(expected[0])):
                        if result[r][c] != expected[r][c]:
                            print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
        else:
            print(f"Test  {i}: (no expected output)")
            for row in result:
                print(row)

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
