"""
Task aa300dc3: Draw diagonal line through cave.

A 5-bordered grid contains a cave of 0-cells. An 8 is placed along the
longest diagonal (slope +1 or -1) through the cave, starting from the
topmost row's leftmost or rightmost 0.
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find topmost row with 0s
    top_row = None
    for r in range(rows):
        if 0 in grid[r]:
            top_row = r
            break
    if top_row is None:
        return result

    # Find leftmost and rightmost 0 in that row
    left_col = None
    right_col = None
    for c in range(cols):
        if grid[top_row][c] == 0:
            if left_col is None:
                left_col = c
            right_col = c

    # Trace down-right diagonal from (top_row, left_col)
    dr_cells = []
    r, c = top_row, left_col
    while 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
        dr_cells.append((r, c))
        r += 1
        c += 1

    # Trace down-left diagonal from (top_row, right_col)
    dl_cells = []
    r, c = top_row, right_col
    while 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
        dl_cells.append((r, c))
        r += 1
        c -= 1

    cells = dr_cells if len(dr_cells) >= len(dl_cells) else dl_cells

    for r, c in cells:
        result[r][c] = 8

    return result


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/aa300dc3.json") as f:
        task = json.load(f)

    for split in ("train", "test"):
        for i, ex in enumerate(task[split]):
            output = solve(ex["input"])
            status = "PASS" if output == ex["output"] else "FAIL"
            print(f"{split} {i}: {status}")
            if status == "FAIL":
                for r, (got, exp) in enumerate(zip(output, ex["output"])):
                    if got != exp:
                        print(f"  row {r}: got {got}, expected {exp}")
