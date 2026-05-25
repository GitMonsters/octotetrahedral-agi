def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    The input is divided by 5s into a colored pattern and a key region with 4s.
    The 4s mark intersection points of vertical/horizontal stripes.
    The output fills the key region with the pattern's colored stripes.
    """
    rows, cols = len(grid), len(grid[0])

    # Find divider column (entire column is 5)
    div_col = None
    for c in range(cols):
        if all(grid[r][c] == 5 for r in range(rows)):
            div_col = c
            break

    if div_col is not None:
        left_has_4 = any(grid[r][c] == 4 for r in range(rows) for c in range(div_col))
        if left_has_4:
            key = [[grid[r][c] for c in range(div_col)] for r in range(rows)]
            pat_cols = list(range(div_col + 1, cols))
        else:
            key = [[grid[r][c] for c in range(div_col + 1, cols)] for r in range(rows)]
            pat_cols = list(range(div_col))
        pat_rows = [r for r in range(rows) if not all(grid[r][c] == 5 for c in pat_cols)]
        pattern = [[grid[r][c] for c in pat_cols] for r in pat_rows]
    else:
        # Divider row fallback
        div_row = None
        for r in range(rows):
            if all(grid[r][c] == 5 for c in range(cols)):
                div_row = r
                break
        top_has_4 = any(grid[r][c] == 4 for r in range(div_row) for c in range(cols))
        if top_has_4:
            key = [grid[r][:] for r in range(div_row)]
            pr_range = list(range(div_row + 1, rows))
        else:
            key = [grid[r][:] for r in range(div_row + 1, rows)]
            pr_range = list(range(div_row))
        pc = [c for c in range(cols) if not all(grid[r][c] == 5 for r in pr_range)]
        pattern = [[grid[r][c] for c in pc] for r in pr_range]

    pat_h, pat_w = len(pattern), len(pattern[0])
    key_h, key_w = len(key), len(key[0])

    # Vertical line columns in pattern (non-zero on every row)
    vert_cols = [c for c in range(pat_w) if all(pattern[r][c] != 0 for r in range(pat_h))]
    # Horizontal line rows in pattern (non-zero on every column)
    horiz_rows = [r for r in range(pat_h) if all(pattern[r][c] != 0 for c in range(pat_w))]

    # Line colors from non-intersection reference cells
    ref_row = next((r for r in range(pat_h) if r not in horiz_rows), 0)
    vert_colors = [pattern[ref_row][c] for c in vert_cols]
    ref_col = next((c for c in range(pat_w) if c not in vert_cols), 0)
    horiz_colors = [pattern[r][ref_col] for r in horiz_rows]

    # Key analysis: rows/cols containing 4s
    k4_rows = sorted({r for r in range(key_h) for c in range(key_w) if key[r][c] == 4})
    k4_cols = sorted({c for r in range(key_h) for c in range(key_w) if key[r][c] == 4})

    # In the sub-matrix, all-4 columns = vertical lines, all-4 rows = horizontal lines
    out_vert = [c for c in k4_cols if all(key[r][c] == 4 for r in k4_rows)]
    out_horiz = [r for r in k4_rows if all(key[r][c] == 4 for c in k4_cols)]

    # Build output
    output = [[0] * key_w for _ in range(key_h)]

    for vi, oc in enumerate(out_vert):
        for r in range(key_h):
            output[r][oc] = vert_colors[vi]

    for hi, orow in enumerate(out_horiz):
        for c in range(key_w):
            output[orow][c] = horiz_colors[hi]

    # Intersections use actual pattern values
    for hi, orow in enumerate(out_horiz):
        for vi, oc in enumerate(out_vert):
            output[orow][oc] = pattern[horiz_rows[hi]][vert_cols[vi]]

    return output


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/b0f4d537.json") as f:
        task = json.load(f)
    ok_all = True
    for i, p in enumerate(task["train"]):
        res = solve(p["input"])
        ok = res == p["output"]
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            ok_all = False
    for i, p in enumerate(task["test"]):
        res = solve(p["input"])
        if "output" in p:
            ok = res == p["output"]
            print(f"Test  {i}: {'PASS' if ok else 'FAIL'}")
            if not ok: ok_all = False
        else:
            print(f"Test  {i}: computed")
    print(f"\n{'ALL PASSED' if ok_all else 'SOME FAILED'}")