"""Solver for ARC-AGI task ce8d95cc.

Pattern: The grid has divider rows (all non-zero) and divider columns (all non-zero).
These partition the grid into rectangular regions. The output compresses each region
to a single cell, preserving divider rows/columns as single rows/columns.
"""

import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    divider_rows = [r for r in range(rows) if all(grid[r][c] != 0 for c in range(cols))]
    divider_cols = [c for c in range(cols) if all(grid[r][c] != 0 for r in range(rows))]

    def make_groups(dividers: list[int], total: int) -> list[list[int]]:
        groups = []
        prev = 0
        for d in dividers:
            if prev < d:
                groups.append([prev])  # representative index for region
            groups.append([d])         # divider kept as-is
            prev = d + 1
        if prev < total:
            groups.append([prev])
        return groups

    row_groups = make_groups(divider_rows, rows)
    col_groups = make_groups(divider_cols, cols)

    return [[grid[rg[0]][cg[0]] for cg in col_groups] for rg in row_groups]


if __name__ == "__main__":
    task = json.loads(r'''{"train": [{"input": [[0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0], [3, 3, 3, 8, 3, 3, 3, 3, 6, 3, 3], [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0], [5, 5, 5, 8, 5, 5, 5, 5, 6, 5, 5], [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0]], "output": [[0, 8, 0, 6, 0], [3, 8, 3, 6, 3], [0, 8, 0, 6, 0], [5, 8, 5, 6, 5], [0, 8, 0, 6, 0]]}, {"input": [[0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0], [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0], [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0], [2, 2, 1, 2, 2, 8, 2, 3, 2, 2, 2], [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0], [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0], [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0], [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0], [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0], [5, 5, 1, 5, 5, 8, 5, 3, 5, 5, 5], [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0], [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0]], "output": [[0, 1, 0, 8, 0, 3, 0], [2, 1, 2, 8, 2, 3, 2], [0, 1, 0, 8, 0, 3, 0], [5, 1, 5, 8, 5, 3, 5], [0, 1, 0, 8, 0, 3, 0]]}, {"input": [[0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0], [3, 3, 4, 3, 3, 3, 3, 3, 3], [0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0], [8, 8, 8, 8, 8, 8, 8, 8, 8], [0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0]], "output": [[0, 4, 0], [3, 4, 3], [0, 4, 0], [8, 8, 8], [0, 4, 0]]}, {"input": [[0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0], [7, 7, 3, 7, 7, 7, 7, 1, 7, 7, 7], [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2], [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0]], "output": [[0, 3, 0, 1, 0], [7, 3, 7, 1, 7], [0, 3, 0, 1, 0], [2, 2, 2, 1, 2], [0, 3, 0, 1, 0], [8, 8, 8, 8, 8], [0, 3, 0, 1, 0]]}], "test": [{"input": [[0, 0, 3, 0, 0, 2, 0, 7, 0, 0, 4, 0, 0], [0, 0, 3, 0, 0, 2, 0, 7, 0, 0, 4, 0, 0], [0, 0, 3, 0, 0, 2, 0, 7, 0, 0, 4, 0, 0], [6, 6, 6, 6, 6, 2, 6, 7, 6, 6, 4, 6, 6], [0, 0, 3, 0, 0, 2, 0, 7, 0, 0, 4, 0, 0], [0, 0, 3, 0, 0, 2, 0, 7, 0, 0, 4, 0, 0], [1, 1, 1, 1, 1, 2, 1, 7, 1, 1, 4, 1, 1], [0, 0, 3, 0, 0, 2, 0, 7, 0, 0, 4, 0, 0], [0, 0, 3, 0, 0, 2, 0, 7, 0, 0, 4, 0, 0], [8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 4, 8, 8], [0, 0, 3, 0, 0, 2, 0, 7, 0, 0, 4, 0, 0], [0, 0, 3, 0, 0, 2, 0, 7, 0, 0, 4, 0, 0]]}]}''')

    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        match = result == ex["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            print(f"  Expected: {ex['output']}")
            print(f"  Got:      {result}")

    print()
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i} output: {result}")

    if all_pass:
        print("\nAll training examples passed!")
