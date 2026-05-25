"""
ARC-AGI Task 8a371977 Solver

Pattern: Grid of identical rectangular cells separated by blue (1) borders.
- Border cells (on the edge of the cell grid) → filled with 2 (red)
- Interior cells (not on edge) → filled with 3 (green)
- Blue (1) borders remain unchanged
"""
import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Find rows/cols that are entirely 1s (borders)
    border_rows = {r for r in range(rows) if all(grid[r][c] == 1 for c in range(cols))}
    border_cols = {c for c in range(cols) if all(grid[r][c] == 1 for r in range(rows))}

    # Group consecutive non-border rows into cell row bands
    cell_row_bands = []
    band = []
    for r in range(rows):
        if r in border_rows:
            if band:
                cell_row_bands.append(band)
                band = []
        else:
            band.append(r)
    if band:
        cell_row_bands.append(band)

    # Group consecutive non-border cols into cell col bands
    cell_col_bands = []
    band = []
    for c in range(cols):
        if c in border_cols:
            if band:
                cell_col_bands.append(band)
                band = []
        else:
            band.append(c)
    if band:
        cell_col_bands.append(band)

    num_cell_rows = len(cell_row_bands)
    num_cell_cols = len(cell_col_bands)

    # Fill cells: border cells → 2, interior cells → 3
    for ri, rband in enumerate(cell_row_bands):
        for ci, cband in enumerate(cell_col_bands):
            is_border = (ri == 0 or ri == num_cell_rows - 1 or
                         ci == 0 or ci == num_cell_cols - 1)
            fill = 2 if is_border else 3
            for r in rband:
                for c in cband:
                    if grid[r][c] == 0:
                        out[r][c] = fill

    return out


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/8a371977.json"))

    # Verify train examples
    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        match = result == pair["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(result)):
                if result[r] != pair["output"][r]:
                    print(f"  Row {r} got:    {result[r]}")
                    print(f"  Row {r} expect: {pair['output'][r]}")

    # Run test
    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            match = result == pair["output"]
            print(f"Test {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
        else:
            print(f"Test {i}: computed (no expected output to check)")
            # Print first few rows to inspect
            for r in range(min(6, len(result))):
                print(f"  {result[r]}")

    print(f"\nAll train passed: {all_pass}")
