"""Solver for 42202aa7: D4-symmetric grid with a uniform rectangular overlay.
Find the overlay and output the hidden content using symmetry prediction."""

from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    cnt = Counter(c for row in grid for c in row)
    bg = cnt.most_common(1)[0][0]

    def get_expected(r, c):
        partners = [grid[r][W-1-c], grid[H-1-r][c], grid[H-1-r][W-1-c]]
        if H == W:
            partners += [grid[c][r], grid[c][H-1-r], grid[H-1-c][r], grid[H-1-c][H-1-r]]
        vals = [grid[r][c]] + partners
        return Counter(vals).most_common(1)[0][0]

    diffs = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != get_expected(r, c)]

    if not diffs:
        return [[bg]]

    fill = grid[diffs[0][0]][diffs[0][1]]
    r1 = min(r for r, c in diffs)
    r2 = max(r for r, c in diffs)
    c1 = min(c for r, c in diffs)
    c2 = max(c for r, c in diffs)

    def all_fill_row(row, ca, cb):
        return all(grid[row][c] == fill for c in range(ca, cb + 1))

    def all_fill_col(col, ra, rb):
        return all(grid[r][col] == fill for r in range(ra, rb + 1))

    # Compute max raw vertical/horizontal extension from bbox
    tr1, tr2 = r1, r2
    while tr1 > 0 and all_fill_row(tr1 - 1, c1, c2):
        tr1 -= 1
    while tr2 < H - 1 and all_fill_row(tr2 + 1, c1, c2):
        tr2 += 1
    vert_ext = (r1 - tr1) + (tr2 - r2)

    tc1, tc2 = c1, c2
    while tc1 > 0 and all_fill_col(tc1 - 1, r1, r2):
        tc1 -= 1
    while tc2 < W - 1 and all_fill_col(tc2 + 1, r1, r2):
        tc2 += 1
    horiz_ext = (c1 - tc1) + (tc2 - c2)

    if vert_ext <= horiz_ext:
        # Extend vertically first (fully), then horizontally with perp border check
        while r1 > 0 and all_fill_row(r1 - 1, c1, c2):
            r1 -= 1
        while r2 < H - 1 and all_fill_row(r2 + 1, c1, c2):
            r2 += 1

        while c1 > 0 and all_fill_col(c1 - 1, r1, r2):
            top_ok = r1 == 0 or grid[r1 - 1][c1 - 1] != fill
            bot_ok = r2 == H - 1 or grid[r2 + 1][c1 - 1] != fill
            if top_ok or bot_ok:
                c1 -= 1
            else:
                break

        while c2 < W - 1 and all_fill_col(c2 + 1, r1, r2):
            top_ok = r1 == 0 or grid[r1 - 1][c2 + 1] != fill
            bot_ok = r2 == H - 1 or grid[r2 + 1][c2 + 1] != fill
            if top_ok or bot_ok:
                c2 += 1
            else:
                break
    else:
        # Extend horizontally first, then vertically with perp border check
        while c1 > 0 and all_fill_col(c1 - 1, r1, r2):
            c1 -= 1
        while c2 < W - 1 and all_fill_col(c2 + 1, r1, r2):
            c2 += 1

        while r1 > 0 and all_fill_row(r1 - 1, c1, c2):
            left_ok = c1 == 0 or grid[r1 - 1][c1 - 1] != fill
            right_ok = c2 == W - 1 or grid[r1 - 1][c2 + 1] != fill
            if left_ok or right_ok:
                r1 -= 1
            else:
                break

        while r2 < H - 1 and all_fill_row(r2 + 1, c1, c2):
            left_ok = c1 == 0 or grid[r2 + 1][c1 - 1] != fill
            right_ok = c2 == W - 1 or grid[r2 + 1][c2 + 1] != fill
            if left_ok or right_ok:
                r2 += 1
            else:
                break

    return [[get_expected(r, c) for c in range(c1, c2 + 1)] for r in range(r1, r2 + 1)]
