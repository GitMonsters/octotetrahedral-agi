"""
Solver for ARC-AGI task a32d8b75.

Pattern:
- Input has key section(s) and a puzzle section separated by vertical strip(s) of color 6.
- Single-key: one key on the left, puzzle on the right.
- Dual-key: keys on BOTH sides, puzzle in the middle. The puzzle itself has an
  embedded separator splitting it into left and right sub-grids. Each key's mask
  (crossed) determines tiling on the opposite sub-grid; stamps are also crossed.

Key structure:
  1. Stamp section: bordered by 0s, inner NxN stamp with two colors
  2. Middle section: contains a shape (mask) in a third color
  3. Bottom sections (bordered by 6s):
     - Section with color-4 marker -> determines mask rotation
     - Section with color-7 marker -> (directional hint)
"""

import json


def extract_key(grid, rows, col_start, col_end, sep_col_val=6):
    """Extract key components from a vertical key section."""
    key_width = col_end - col_start
    N = key_width - 2

    # Find actual stamp inner rows (first/last rows with non-zero inner values)
    stamp_start = None
    stamp_end = None
    for r in range(N + 2):
        inner = [grid[r][col_start + 1 + c] for c in range(N)]
        if any(v != 0 for v in inner):
            if stamp_start is None:
                stamp_start = r
            stamp_end = r
    if stamp_start is None:
        stamp_start = 1
        stamp_end = N

    stamp = [grid[stamp_start + r][col_start + 1:col_start + 1 + N] for r in range(N)]
    stamp_colors = sorted(set(c for row in stamp for c in row))
    inv_stamp = None
    if len(stamp_colors) == 2:
        cA, cB = stamp_colors
        inv_stamp = [[cB if v == cA else cA for v in row] for row in stamp]

    # Find bottom key separator (first all-6 row within key columns)
    bot_sep_row = None
    for r in range(N + 2, rows):
        if all(grid[r][c] == 6 for c in range(col_start, col_end)):
            bot_sep_row = r
            break

    # Extract bottom sections (always 3x3 inner)
    marker4_pos = None
    marker7_positions = []
    if bot_sep_row:
        def find_inner_start(r_start):
            for c in range(col_start, col_end):
                if grid[r_start][c] != 6:
                    return c
            return col_start + 1

        bs1_row = bot_sep_row + 1
        bs1_cs = find_inner_start(bs1_row)
        bs1 = [grid[bs1_row + r][bs1_cs:bs1_cs + 3] for r in range(3)]

        bs2_sep = None
        for r in range(bs1_row + 3, rows):
            if all(grid[r][c] == 6 for c in range(col_start, col_end)):
                bs2_sep = r
                break

        if bs2_sep:
            bs2_row = bs2_sep + 1
            bs2_cs = find_inner_start(bs2_row)
            bs2 = [grid[bs2_row + r][bs2_cs:bs2_cs + 3] for r in range(3)]
        else:
            bs2 = [[0]*3 for _ in range(3)]

        for r in range(3):
            for c in range(3):
                if bs1[r][c] == 4:
                    marker4_pos = (r, c)
                if bs2[r][c] == 7:
                    marker7_positions.append((r, c))

    # Determine rotation from 4-marker
    rot_k = 0
    if marker4_pos:
        r4, c4 = marker4_pos
        if r4 == 0 and c4 == 0:
            rot_k = 0
        elif r4 == 0 and c4 == 2:
            rot_k = 3
        elif r4 == 2 and c4 == 0:
            rot_k = 1
        elif r4 == 2 and c4 == 2:
            rot_k = 2

    # Extract mask from middle section
    mid_start = N + 2
    mid_end = bot_sep_row if bot_sep_row else rows
    mask_full = [grid[r][col_start:col_end] for r in range(mid_start, mid_end)]

    # Find non-zero bounding box
    mrc = len(mask_full)
    r_min, r_max, c_min, c_max = mrc, -1, key_width, -1
    for r in range(mrc):
        for c in range(key_width):
            if mask_full[r][c] != 0:
                r_min = min(r_min, r)
                r_max = max(r_max, r)
                c_min = min(c_min, c)
                c_max = max(c_max, c)

    mask_core = None
    if r_max >= 0:
        mask_core = []
        for r in range(r_min, r_max + 1):
            row = [1 if mask_full[r][c] != 0 else 0 for c in range(c_min, c_max + 1)]
            mask_core.append(row)

    return {
        'N': N,
        'stamp': stamp,
        'inv_stamp': inv_stamp,
        'marker4_pos': marker4_pos,
        'rot_k': rot_k,
        'mask_core': mask_core,
    }


def apply_tile(output, mask, inv_stamp, N, rot_k, marker4_pos, out_rows, out_cols, invert_anchor=False):
    """Apply inverted stamp tile to output where mask covers."""
    if mask is None or inv_stamp is None:
        return

    rotated = rotate_grid(mask, rot_k)
    rot_rows = len(rotated)
    rot_cols = len(rotated[0])

    # Compute cell-level start position based on 4-marker corner
    if marker4_pos:
        r4, c4 = marker4_pos
        if invert_anchor:
            r4, c4 = 2 - r4, 2 - c4
        start_r = 0 if r4 == 0 else out_rows - rot_rows * N
        start_c = 0 if c4 == 0 else out_cols - rot_cols * N
    else:
        start_r, start_c = 0, 0

    dr = (-start_r) % N
    dc = (-start_c) % N

    for r in range(out_rows):
        for c in range(out_cols):
            cell_r = r - start_r
            cell_c = c - start_c
            if cell_r < 0 or cell_c < 0:
                continue
            mr = cell_r // N
            mc = cell_c // N
            if mr < 0 or mr >= rot_rows or mc < 0 or mc >= rot_cols:
                continue
            if rotated[mr][mc]:
                output[r][c] = inv_stamp[(r + dr) % N][(c + dc) % N]


def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find separator columns (all-6 columns)
    sep_cols = []
    for c in range(cols):
        if all(grid[r][c] == 6 for r in range(rows)):
            sep_cols.append(c)
    if not sep_cols:
        return grid

    first_sep = sep_cols[0]
    last_sep = sep_cols[-1] if len(sep_cols) > 1 else cols
    dual_key = len(sep_cols) > 1

    if not dual_key:
        # Single-key mode
        left_key = extract_key(grid, rows, 0, first_sep)
        N = left_key['N']
        right_start = first_sep + 1
        out_cols = cols - right_start
        output = [row[right_start:] for row in grid]

        apply_tile(output, left_key['mask_core'], left_key['inv_stamp'],
                   N, left_key['rot_k'], left_key['marker4_pos'],
                   rows, out_cols)
        return output

    # Dual-key mode
    left_key = extract_key(grid, rows, 0, first_sep)
    right_key = extract_key(grid, rows, last_sep + 1, cols)
    N = left_key['N']

    # Puzzle section = between the two separators
    puzzle_start = first_sep + 1
    puzzle_end = last_sep
    puzzle_cols = puzzle_end - puzzle_start
    output = [row[puzzle_start:puzzle_end] for row in grid]

    # Find embedded separator in puzzle section (contiguous columns of one uniform value)
    # The embedded key has width = N + 2 (same as outer keys)
    embed_width = N + 2
    embed_start = None
    for cs in range(puzzle_cols - embed_width + 1):
        # Check if columns cs..cs+embed_width-1 are all the same value per row
        is_uniform = True
        for c in range(cs, cs + embed_width):
            vals = set(output[r][c] for r in range(rows))
            if len(vals) > 1:
                is_uniform = False
                break
        if is_uniform:
            embed_start = cs
            break

    if embed_start is None:
        # Fallback: just return puzzle section
        return output

    embed_end = embed_start + embed_width
    left_sub_cols = embed_start
    right_sub_start = embed_end
    right_sub_cols = puzzle_cols - right_sub_start

    # LEFT sub-grid: use RIGHT key's mask, rotated by LEFT key's rotation,
    # anchor inverted (dual-key crossing), tile = inverted RIGHT stamp
    left_sub = [row[:left_sub_cols] for row in output]
    apply_tile(left_sub, right_key['mask_core'], right_key['inv_stamp'],
               N, left_key['rot_k'], left_key['marker4_pos'],
               rows, left_sub_cols, invert_anchor=True)

    # RIGHT sub-grid: use LEFT key's mask, rotated by RIGHT key's rotation,
    # anchor inverted, tile = inverted LEFT stamp
    right_sub = [row[right_sub_start:] for row in output]
    apply_tile(right_sub, left_key['mask_core'], left_key['inv_stamp'],
               N, right_key['rot_k'], right_key['marker4_pos'],
               rows, right_sub_cols, invert_anchor=True)

    # Reassemble output
    for r in range(rows):
        output[r] = left_sub[r] + output[r][embed_start:embed_end] + right_sub[r]

    return output


def rotate_grid(grid, k):
    """Rotate grid by k*90° counterclockwise (same as numpy.rot90)."""
    k = k % 4
    result = [row[:] for row in grid]
    for _ in range(k):
        r = len(result)
        c = len(result[0])
        new = [[result[row][c - 1 - col] for row in range(r)] for col in range(c)]
        result = new
    return result


def main():
    with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/a32d8b75.json') as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data['train']):
        inp = ex['input']
        expected = ex['output']
        actual = solve(inp)

        if actual == expected:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            all_pass = False
            # Show first few differences
            diffs = 0
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if r < len(actual) and c < len(actual[0]):
                        if actual[r][c] != expected[r][c]:
                            if diffs < 10:
                                print(f"  ({r},{c}): expected {expected[r][c]}, got {actual[r][c]}")
                            diffs += 1
                    else:
                        diffs += 1
            print(f"  Total diffs: {diffs}")

    if all_pass:
        print("\nAll training examples PASS!")
    else:
        print("\nSome training examples FAILED.")


if __name__ == '__main__':
    main()
