"""ARC-AGI solver for task 5ffb2104.

Pattern: Each connected component (4-connected, same color) is treated as a
rigid body and pushed rightward ("gravity to the right"). Objects furthest
right land first; leftward objects stack against them, preserving relative order.
"""

import json
from collections import defaultdict
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])

    # Find 4-connected components grouped by same color
    visited = [[False] * cols for _ in range(rows)]
    objects: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                cells: list[tuple[int, int]] = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if 0 <= cr < rows and 0 <= cc < cols and not visited[cr][cc] and grid[cr][cc] == color:
                        visited[cr][cc] = True
                        cells.append((cr, cc))
                        stack.extend([(cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)])
                objects.append((color, cells))

    # Sort by rightmost column descending (rightmost objects placed first)
    objects.sort(key=lambda obj: max(c for _, c in obj[1]), reverse=True)

    # Per-row frontier: the leftmost column already claimed from the right
    frontier = [cols] * rows
    output = [[0] * cols for _ in range(rows)]

    for color, cells in objects:
        # For each row, find the object's rightmost cell
        row_max_col: dict[int, int] = defaultdict(lambda: -1)
        row_min_col: dict[int, int] = defaultdict(lambda: cols)
        for r, c in cells:
            row_max_col[r] = max(row_max_col[r], c)
            row_min_col[r] = min(row_min_col[r], c)

        # Max shift: each row's rightmost cell must stay left of its frontier
        shift = min(frontier[r] - row_max_col[r] - 1 for r in row_max_col)

        for r, c in cells:
            output[r][c + shift] = color

        # Update frontiers to the leftmost cell placed in each row
        for r in row_min_col:
            frontier[r] = min(frontier[r], row_min_col[r] + shift)

    return output


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/5ffb2104.json"))

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        match = result == pair["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r, (got, exp) in enumerate(zip(result, pair["output"])):
                if got != exp:
                    print(f"  Row {r}: got {got}")
                    print(f"       exp {exp}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            match = result == pair["output"]
            print(f"Test  {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
                for r, (got, exp) in enumerate(zip(result, pair["output"])):
                    if got != exp:
                        print(f"  Row {r}: got {got}")
                        print(f"       exp {exp}")
        else:
            print(f"Test  {i}: (no expected output)")
            for r in result:
                print(f"  {r}")

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
