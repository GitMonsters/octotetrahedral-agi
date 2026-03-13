"""Solver for ARC-AGI task c64f1187.

Pattern:
- Top section has color markers with 2x2 shape templates (1s) below-right.
- Bottom section has a grid of 2x2 cells (5s) with color tags.
- Output replaces each tagged cell with its color's shape template.
"""

from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])

    # Step 1: Find color-shape mapping from the template section
    # Shapes are made of 1s; the color marker sits one row above, one col left
    one_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    shape_min_r = min(r for r, _ in one_positions)
    color_marker_row = shape_min_r - 1

    color_shapes: dict[int, list[list[int]]] = {}
    for c in range(cols):
        val = grid[color_marker_row][c]
        if val not in (0, 1, 5):
            shape = [
                [grid[shape_min_r + dr][c + 1 + dc] for dc in range(2)]
                for dr in range(2)
            ]
            color_shapes[val] = shape

    # Step 2: Locate the grid of 2x2 cells made of 5s
    five_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    min_r = min(r for r, _ in five_positions)
    min_c = min(c for _, c in five_positions)
    max_r = max(r for r, _ in five_positions)
    max_c = max(c for _, c in five_positions)

    # Cells are 2x2 with 1-wide gaps → stride of 3
    cell_rows = list(range(min_r, max_r + 1, 3))
    cell_cols = list(range(min_c, max_c + 1, 3))

    # Step 3: Read color from each cell (non-5, non-0 value, or 0 if blank)
    cell_colors = []
    for cr in cell_rows:
        row_colors = []
        for cc in cell_cols:
            color = 0
            for dr in range(2):
                for dc in range(2):
                    v = grid[cr + dr][cc + dc]
                    if v not in (0, 5):
                        color = v
            row_colors.append(color)
        cell_colors.append(row_colors)

    # Step 4: Build output — same cell grid but shapes instead of 5-blocks
    nr, nc = len(cell_rows), len(cell_cols)
    out_h = nr * 2 + (nr - 1)
    out_w = nc * 2 + (nc - 1)
    output = [[0] * out_w for _ in range(out_h)]

    for ri in range(nr):
        for ci in range(nc):
            color = cell_colors[ri][ci]
            if color == 0:
                continue
            shape = color_shapes[color]
            or_ = ri * 3
            oc = ci * 3
            for dr in range(2):
                for dc in range(2):
                    if shape[dr][dc] == 1:
                        output[or_ + dr][oc + dc] = color

    return output


if __name__ == "__main__":
    import json

    task = json.loads(r'''{"train": [{"input": [[2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 2, 0, 5, 5, 0, 5, 5, 0, 5, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 3, 0, 5, 2, 0, 5, 3, 0, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 3, 0, 5, 5, 0, 5, 5, 0, 5, 8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[2, 2, 0, 0, 0, 0, 0, 0, 0, 4, 4], [2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 0, 2, 2, 0, 3, 3, 0, 0, 0], [0, 3, 0, 2, 0, 0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 0, 0, 0, 0, 0, 0, 0, 8, 0], [0, 3, 0, 0, 0, 0, 0, 0, 0, 8, 8]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0], [0, 0, 5, 3, 0, 5, 3, 0, 5, 7, 0, 5, 5, 0, 5, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0], [0, 0, 5, 2, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0], [0, 0, 5, 2, 0, 5, 2, 0, 5, 3, 0, 5, 3, 0, 5, 7, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[3, 3, 0, 3, 3, 0, 7, 7, 0, 0, 0, 0, 2, 2], [0, 3, 0, 0, 3, 0, 7, 0, 0, 0, 0, 0, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 0, 2, 2, 0, 3, 3, 0, 3, 3, 0, 7, 7], [2, 2, 0, 2, 2, 0, 0, 3, 0, 0, 3, 0, 7, 0]]}], "test": [{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0, 0], [0, 5, 2, 0, 5, 2, 0, 5, 2, 0, 5, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0, 0], [0, 5, 5, 0, 5, 3, 0, 5, 3, 0, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0, 0], [0, 5, 4, 0, 5, 3, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 0, 0, 0], [0, 5, 4, 0, 5, 5, 0, 5, 5, 0, 5, 8, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[2, 2, 0, 2, 2, 0, 2, 2, 0, 8, 8], [2, 0, 0, 2, 0, 0, 2, 0, 0, 8, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 3, 0, 3, 3, 0, 0, 0], [0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 3, 3, 0, 0, 0, 0, 0, 0], [4, 4, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 0, 8, 8], [4, 4, 0, 0, 0, 0, 0, 0, 0, 8, 8]]}]}''')

    # Verify on all examples
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        ok = result == expected
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")

    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        expected = ex["output"]
        ok = result == expected
        print(f"Test  {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
