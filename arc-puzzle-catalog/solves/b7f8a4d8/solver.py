"""Solver for ARC-AGI task b7f8a4d8.

Pattern: Grid of tiled cells with borders and interiors. Some cells have a
special (non-default) interior color. Connect same-colored special cells by
filling gap pixels between them horizontally and vertically.
"""

from collections import Counter
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    grid = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find first non-zero row to detect column tiling
    border_row = next(r for r in range(rows) if any(v != 0 for v in grid[r]))

    # Find column segment starts
    col_starts = []
    in_seg = False
    for c in range(cols):
        if grid[border_row][c] != 0 and not in_seg:
            col_starts.append(c)
            in_seg = True
        elif grid[border_row][c] == 0:
            in_seg = False

    # Cell width = length of first non-zero segment
    cell_w = 0
    c = col_starts[0]
    while c < cols and grid[border_row][c] != 0:
        cell_w += 1
        c += 1

    # Find row segment starts using first cell column
    border_col = col_starts[0]
    row_starts = []
    in_seg = False
    for r in range(rows):
        if grid[r][border_col] != 0 and not in_seg:
            row_starts.append(r)
            in_seg = True
        elif grid[r][border_col] == 0:
            in_seg = False

    # Cell height
    cell_h = 0
    r = row_starts[0]
    while r < rows and grid[r][border_col] != 0:
        cell_h += 1
        r += 1

    interior_h = cell_h - 2
    interior_w = cell_w - 2

    num_cr = len(row_starts)
    num_cc = len(col_starts)

    # Extract cell interior values (use top-left interior pixel)
    cell_vals = {}
    for i in range(num_cr):
        for j in range(num_cc):
            r0 = row_starts[i] + 1
            c0 = col_starts[j] + 1
            if r0 < rows and c0 < cols:
                cell_vals[(i, j)] = grid[r0][c0]

    default_color = Counter(cell_vals.values()).most_common(1)[0][0]

    # Group special cells by color
    special: dict[int, list] = {}
    for (i, j), color in cell_vals.items():
        if color != default_color:
            special.setdefault(color, []).append((i, j))

    # Connect special cells
    for color, positions in special.items():
        # Horizontal connections per cell-row
        by_row: dict[int, list] = {}
        for i, j in positions:
            by_row.setdefault(i, []).append(j)
        for i, js in by_row.items():
            if len(js) < 2:
                continue
            j_min, j_max = min(js), max(js)
            for jj in range(j_min, j_max):
                if jj + 1 >= num_cc:
                    break
                gap_c0 = col_starts[jj] + cell_w
                gap_c1 = col_starts[jj + 1]
                for dr in range(interior_h):
                    pr = row_starts[i] + 1 + dr
                    for gc in range(gap_c0, gap_c1):
                        if pr < rows and gc < cols:
                            grid[pr][gc] = color

        # Vertical connections per cell-column
        by_col: dict[int, list] = {}
        for i, j in positions:
            by_col.setdefault(j, []).append(i)
        for j, is_ in by_col.items():
            if len(is_) < 2:
                continue
            i_min, i_max = min(is_), max(is_)
            for ii in range(i_min, i_max):
                if ii + 1 >= num_cr:
                    break
                gap_r0 = row_starts[ii] + cell_h
                gap_r1 = row_starts[ii + 1]
                for dc in range(interior_w):
                    pc = col_starts[j] + 1 + dc
                    for gr in range(gap_r0, gap_r1):
                        if gr < rows and pc < cols:
                            grid[gr][pc] = color

    return grid


if __name__ == "__main__":
    import json, sys

    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/b7f8a4d8.json"))

    ok = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        match = result == pair["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            ok = False
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != pair["output"][r][c]:
                        print(f"  diff at ({r},{c}): got {result[r][c]} expected {pair['output'][r][c]}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        print(f"Test {i}: solved ({len(result)}x{len(result[0])})")
        outpath = f"/Users/evanpieser/arc-puzzle-catalog/solves/b7f8a4d8/test_{i}.json"
        json.dump(result, open(outpath, "w"))
        print(f"  -> {outpath}")

    if ok:
        print("\nAll training examples PASS!")
    sys.exit(0 if ok else 1)
