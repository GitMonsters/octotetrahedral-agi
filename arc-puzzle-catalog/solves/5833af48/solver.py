#!/usr/bin/env python3
"""
Solver for ARC-AGI task 5833af48.

Rule:
- Pattern1: small template using color 2 as background and 8 as the shape.
- Pattern2: tile-map template using the canvas color as background and 8 as
  filled-tile markers.
- Canvas: large rectangle filled uniformly with the canvas color.

The output equals the canvas dimensions, filled with the canvas color.
The canvas is divided into (canvas_h/p1_h) × (canvas_w/p1_w) tiles.
Pattern2 is likewise divided into the same grid of macro-blocks.
Wherever a macro-block in pattern2 contains an 8, stamp pattern1's 8-pixels
into the corresponding tile of the output.
"""

import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # ── 1. Find the separator row (first all-zero row after some non-zero rows) ──
    in_pattern = False
    sep_row = None
    for r in range(rows):
        if any(grid[r][c] != 0 for c in range(cols)):
            in_pattern = True
        elif in_pattern:
            sep_row = r
            break

    if sep_row is None:
        return grid

    # ── 2. Locate the canvas (large uniform rectangle below the separator) ──
    canvas_start_row = canvas_end_row = None
    canvas_color = None
    for r in range(sep_row + 1, rows):
        nz = [grid[r][c] for c in range(cols) if grid[r][c] != 0]
        if not nz:
            if canvas_start_row is not None:
                canvas_end_row = r - 1
            break
        unique = set(nz)
        if len(unique) == 1:
            if canvas_start_row is None:
                canvas_start_row = r
                canvas_color = nz[0]
    if canvas_start_row is not None and canvas_end_row is None:
        canvas_end_row = rows - 1

    if canvas_start_row is None or canvas_color is None:
        return grid

    canvas_cols = [c for c in range(cols) if grid[canvas_start_row][c] == canvas_color]
    canvas_start_col = min(canvas_cols)
    canvas_end_col = max(canvas_cols)
    canvas_h = canvas_end_row - canvas_start_row + 1
    canvas_w = canvas_end_col - canvas_start_col + 1

    # ── 3. Extract pattern rows (non-zero rows above the separator) ──
    pat_rows = [r for r in range(sep_row) if any(grid[r][c] != 0 for c in range(cols))]
    if not pat_rows:
        return grid
    pat_r0, pat_r1 = min(pat_rows), max(pat_rows)

    # Find the column gap separating pattern1 from pattern2
    data_cols = [c for c in range(cols)
                 if any(grid[r][c] != 0 for r in range(pat_r0, pat_r1 + 1))]
    gap_col = None
    for i in range(len(data_cols) - 1):
        if data_cols[i + 1] - data_cols[i] > 1:
            gap_col = (data_cols[i], data_cols[i + 1])
            break

    if gap_col is None:
        return grid

    p1_c0 = min(c for c in data_cols if c <= gap_col[0])
    p1_c1 = max(c for c in data_cols if c <= gap_col[0])
    p2_c0 = min(c for c in data_cols if c >= gap_col[1])
    p2_c1 = max(c for c in data_cols if c >= gap_col[1])

    # Build pattern sub-grids, keeping only non-zero rows
    p1_raw = [grid[r][p1_c0:p1_c1 + 1] for r in range(pat_r0, pat_r1 + 1)]
    p1 = [row for row in p1_raw if any(v != 0 for v in row)]

    p2_raw = [grid[r][p2_c0:p2_c1 + 1] for r in range(pat_r0, pat_r1 + 1)]
    p2 = [row for row in p2_raw if any(v != 0 for v in row)]

    if not p1 or not p2:
        return grid

    p1_h, p1_w = len(p1), len(p1[0])
    p2_h, p2_w = len(p2), len(p2[0])

    # ── 4. Compute tile grid and macro-block size ──
    tiles_v = canvas_h // p1_h   # number of tile rows
    tiles_h_n = canvas_w // p1_w  # number of tile cols

    if tiles_v == 0 or tiles_h_n == 0:
        return grid

    block_h = p2_h // tiles_v    # macro-block height in pattern2
    block_w = p2_w // tiles_h_n  # macro-block width in pattern2

    # ── 5. Build output: canvas-sized, canvas-color fill ──
    out = [[canvas_color] * canvas_w for _ in range(canvas_h)]

    for ti in range(tiles_v):
        for tj in range(tiles_h_n):
            # Check if the macro-block at (ti, tj) in pattern2 has any 8
            has_8 = False
            for bi in range(block_h if block_h > 0 else 1):
                for bj in range(block_w if block_w > 0 else 1):
                    ri = ti * block_h + bi
                    cj = tj * block_w + bj
                    if ri < p2_h and cj < p2_w and p2[ri][cj] == 8:
                        has_8 = True
                        break
                if has_8:
                    break

            if has_8:
                out_r0 = ti * p1_h
                out_c0 = tj * p1_w
                for pi in range(p1_h):
                    for pj in range(p1_w):
                        if p1[pi][pj] == 8:
                            out[out_r0 + pi][out_c0 + pj] = 8

    return out


if __name__ == "__main__":
    with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/5833af48.json') as f:
        task = json.load(f)

    all_pass = True
    for i, example in enumerate(task['train']):
        result = solve(example['input'])
        expected = example['output']
        passed = result == expected
        print(f"Train {i + 1}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False
            print(f"  expected {len(expected)}x{len(expected[0])}, got {len(result)}x{len(result[0])}")

    print()
    test = task['test'][0]
    result = solve(test['input'])
    expected = test['output']
    passed = result == expected
    print(f"Test: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  expected {len(expected)}x{len(expected[0])}, got {len(result)}x{len(result[0])}")
