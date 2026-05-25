"""
Solver for ARC task 9aaea919.

The grid contains plus-shaped patterns arranged in a 4-column layout.
The last row is a "key row" with color codes:
  - Color 2 at a column: change all shapes in that column to color 5
  - Color 3 at a column: extend/replicate shapes upward
The number of new blocks added = total blocks in "change" columns.
The key row is cleared to background in the output.
"""
import json
from collections import Counter


def solve(grid):
    R, C = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(R) for c in range(C)).most_common(1)[0][0]
    out = [row[:] for row in grid]

    # Find column blocks (groups of 5 cols with non-bg cells)
    col_has_color = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                col_has_color.add(c)
    sorted_cols = sorted(col_has_color)
    col_blocks = []
    i = 0
    while i < len(sorted_cols):
        start = sorted_cols[i]
        while i < len(sorted_cols) and sorted_cols[i] - start < 5:
            i += 1
        col_blocks.append(start)

    key_row = R - 1

    # Row blocks: 3-row groups spaced 4 apart above key row
    row_block_tops = []
    r = key_row - 4
    while r >= 0:
        row_block_tops.append(r)
        r -= 4
    row_block_tops.reverse()

    # Parse key row entries per column block
    key_entries = {}
    for ci, c_start in enumerate(col_blocks):
        mid_c = c_start + 2
        if mid_c < C and grid[key_row][mid_c] != bg:
            key_entries[ci] = grid[key_row][mid_c]

    def get_shape_color(ci, rb_top):
        c_start = col_blocks[ci]
        mid_r, mid_c = rb_top + 1, c_start + 2
        if 0 <= mid_r < R and 0 <= mid_c < C:
            v = grid[mid_r][mid_c]
            return v if v != bg else None
        return None

    blocks_per_col = {}
    color_per_col = {}
    for ci in range(len(col_blocks)):
        count = 0
        color = None
        for rb in row_block_tops:
            sc = get_shape_color(ci, rb)
            if sc is not None:
                count += 1
                color = sc
        blocks_per_col[ci] = count
        color_per_col[ci] = color

    extend_cols = [ci for ci, kc in key_entries.items() if kc == 3]
    change_cols = [ci for ci, kc in key_entries.items() if kc == 2]
    n_add = sum(blocks_per_col.get(ci, 0) for ci in change_cols)

    def draw_plus(rb_top, c_start, color):
        for dr in range(3):
            for dc in range(5):
                r, c = rb_top + dr, c_start + dc
                if 0 <= r < R and 0 <= c < C:
                    if dr == 1 or (1 <= dc <= 3):
                        out[r][c] = color

    for ci in change_cols:
        c_start = col_blocks[ci]
        for r in range(R):
            for c in range(c_start, min(c_start + 5, C)):
                if out[r][c] != bg and r != key_row:
                    out[r][c] = 5

    for ci in extend_cols:
        ext_color = color_per_col[ci]
        c_start = col_blocks[ci]
        topmost_rb = None
        for rb in row_block_tops:
            if get_shape_color(ci, rb) is not None:
                topmost_rb = rb
                break
        if topmost_rb is not None and ext_color is not None:
            new_top = topmost_rb - 4
            for _ in range(n_add):
                if new_top < 0:
                    break
                draw_plus(new_top, c_start, ext_color)
                new_top -= 4

    for c in range(C):
        if out[key_row][c] != bg:
            out[key_row][c] = bg

    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/9aaea919.json") as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            R, C = len(expected), len(expected[0])
            diffs = sum(
                1 for r in range(R) for c in range(C) if result[r][c] != expected[r][c]
            )
            print(f"Train {i}: FAIL ({diffs} mismatches)")
            all_pass = False

    for i, ex in enumerate(data["test"]):
        if "output" in ex:
            result = solve(ex["input"])
            expected = ex["output"]
            if result == expected:
                print(f"Test  {i}: PASS")
            else:
                R, C = len(expected), len(expected[0])
                diffs = sum(
                    1
                    for r in range(R)
                    for c in range(C)
                    if result[r][c] != expected[r][c]
                )
                print(f"Test  {i}: FAIL ({diffs} mismatches)")
                all_pass = False
        else:
            result = solve(ex["input"])
            print(f"Test  {i}: produced {len(result)}x{len(result[0])} grid")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")
