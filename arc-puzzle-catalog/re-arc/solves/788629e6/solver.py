"""
Solver for ARC-AGI puzzle 788629e6

Rule: Input is a TABLE structure (grid lines of one color, bg filling cells).
At intersections of horizontal and vertical grid lines, some have SPECIAL colors.
The output encodes the structure of these intersections.

Algorithm:
1. Detect horizontal lines, vertical lines, background color, and grid color.
2. Build intersection grid (values at h-line x v-line crossings).
3. Trim trailing all-gc h-lines and leading all-gc v-lines.
4. If intersection grid has 2x2 block structure -> use block-level rules
   with boundary/corner logic for gc placement.
5. Otherwise -> use standard 4-corners rule: output = corner value when all 4
   intersection corners of a cell agree (gc->bg), else bg. With special handling
   for uniform columns with boundary transitions.
"""

import json
from collections import Counter
from typing import List


def solve(grid_input: List[List[int]]) -> List[List[int]]:
    grid = grid_input
    H, W = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(H) for c in range(W)]
    bg = Counter(flat).most_common(1)[0][0]

    # Detect horizontal grid lines (rows that are mostly non-bg)
    all_h = sorted(
        r for r in range(H)
        if sum(1 for c in range(W) if grid[r][c] != bg) > W * 0.5
    )
    grid_c = Counter(grid[all_h[0]][c] for c in range(W)).most_common(1)[0][0]

    # Detect vertical grid lines (cols that are non-bg between h-lines)
    band_rows = [r for r in range(H) if r not in set(all_h)]
    v_lines = (
        sorted(c for c in range(W) if all(grid[r][c] != bg for r in band_rows))
        if band_rows else []
    )

    # Build intersection grid
    int_grid = [[grid[r][c] for c in v_lines] for r in all_h]
    nR, nC = len(all_h), len(v_lines)

    # Remove trailing all-gc h-lines
    while nR > 0 and all(int_grid[nR - 1][j] == grid_c for j in range(nC)):
        nR -= 1
    # Remove leading all-gc v-lines
    start_c = 0
    while start_c < nC and all(int_grid[i][start_c] == grid_c for i in range(nR)):
        start_c += 1

    G = [int_grid[i][start_c:nC] for i in range(nR)]
    rR = nR
    rC = nC - start_c
    oR = rR - 1
    oC = rC - 1

    # Detect block structure (consecutive identical rows/cols forming >=2-wide groups)
    row_groups: List[tuple] = []
    i = 0
    while i < rR:
        j = i + 1
        while j < rR and G[j] == G[i]:
            j += 1
        row_groups.append((i, j - 1))
        i = j

    col_groups: List[tuple] = []
    j = 0
    while j < rC:
        k = j + 1
        col_j = [G[r][j] for r in range(rR)]
        while k < rC and [G[r][k] for r in range(rR)] == col_j:
            k += 1
        col_groups.append((j, k - 1))
        j = k

    has_block = (
        all(e - s + 1 >= 2 for s, e in row_groups)
        and all(e - s + 1 >= 2 for s, e in col_groups)
        and len(row_groups) >= 2
        and len(col_groups) >= 2
    )

    if has_block:
        bR = len(row_groups)
        bC = len(col_groups)
        exp_oR = 2 * bR - 1
        exp_oC = 2 * bC - 1

        if exp_oR == oR and exp_oC == oC:
            return _solve_block(G, rR, rC, oR, oC, bg, grid_c,
                                row_groups, col_groups, bR, bC)

    return _solve_four_corners(G, rR, rC, oR, oC, bg, grid_c)


def _solve_block(G, rR, rC, oR, oC, bg, gc, row_groups, col_groups, bR, bC):
    """Solve using block-level logic for grids with 2x2 block structure."""
    BG = [
        [G[row_groups[ri][0]][col_groups[ci][0]] for ci in range(bC)]
        for ri in range(bR)
    ]

    def row_seg(R: int, C: int) -> bool:
        return BG[R][C] != BG[R + 1][C]

    def col_seg(R: int, C: int) -> bool:
        return BG[R][C] != BG[R][C + 1]

    result = [[bg] * oC for _ in range(oR)]

    for out_r in range(oR):
        for out_c in range(oC):
            is_br = (out_r % 2 == 0)
            is_bc = (out_c % 2 == 0)
            br = out_r // 2
            bc = out_c // 2

            if is_br and is_bc:
                val = BG[br][bc]
                if val != gc:
                    result[out_r][out_c] = val
                else:
                    n_sp = sum(1 for v in BG[br] if v != gc)
                    if n_sp > len(BG[br]) // 2:
                        result[out_r][out_c] = gc

            elif not is_br and is_bc:
                # V-boundary between block rows br and br+1, at block col bc
                has_other = any(row_seg(br, c) for c in range(bC) if c != bc)
                no_self = not row_seg(br, bc)
                has_col_seg = any(
                    row_seg(r, bc) for r in range(bR - 1) if r != br
                )
                if has_other and no_self and has_col_seg:
                    result[out_r][out_c] = gc

            elif is_br and not is_bc:
                # H-boundary between block cols bc and bc+1, at block row br
                has_other = any(col_seg(r, bc) for r in range(bR) if r != br)
                no_self = not col_seg(br, bc)
                above = [
                    (BG[r][bc], BG[r][bc + 1])
                    for r in range(br) if col_seg(r, bc)
                ]
                below = [
                    (BG[r][bc], BG[r][bc + 1])
                    for r in range(br + 1, bR) if col_seg(r, bc)
                ]
                same_type = bool(set(above) & set(below)) if above and below else False
                if has_other and no_self and same_type:
                    result[out_r][out_c] = gc

    return result


def _solve_four_corners(G, rR, rC, oR, oC, bg, gc):
    """Standard 4-corners rule with uniform-column correction."""
    result = []
    for i in range(oR):
        row = []
        for j in range(oC):
            tl, tr, bl, br = G[i][j], G[i][j + 1], G[i + 1][j], G[i + 1][j + 1]
            if tl == tr == bl == br:
                val = tl
                if val == gc:
                    row.append(bg)
                else:
                    col_j_uniform = len(set(G[r][j] for r in range(rR))) == 1
                    col_j1_uniform = len(set(G[r][j + 1] for r in range(rR))) == 1
                    if col_j_uniform and col_j1_uniform and j + 2 < rC:
                        trans_top = (i > 0 and G[i - 1][j + 2] != G[i][j + 2])
                        trans_bot = (i + 2 < rR and G[i + 1][j + 2] != G[i + 2][j + 2])
                        if trans_top and trans_bot:
                            row.append(bg)
                        else:
                            row.append(val)
                    else:
                        row.append(val)
            else:
                row.append(bg)
        result.append(row)
    return result


def test():
    import sys
    task_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rearc45/788629e6.json'
    with open(task_path) as f:
        task = json.load(f)

    all_ok = True
    for ti, pair in enumerate(task['train']):
        pred = solve(pair['input'])
        exp = pair['output']
        oR, oC = len(exp), len(exp[0])
        match = sum(
            1 for i in range(oR) for j in range(oC)
            if pred[i][j] == exp[i][j]
        )
        total = oR * oC
        status = 'PASS' if match == total else 'FAIL'
        print(f"Train {ti}: {match}/{total} {status}")
        if match != total:
            all_ok = False
            for i in range(oR):
                for j in range(oC):
                    if pred[i][j] != exp[i][j]:
                        print(f"  ({i},{j}): got {pred[i][j]}, expected {exp[i][j]}")

    print(f"\nAll training pairs pass: {all_ok}")

    for ti, pair in enumerate(task['test']):
        pred = solve(pair['input'])
        print(f"\nTest {ti} ({len(pred)}x{len(pred[0])}):")
        for row in pred:
            print(f"  {row}")

    return all_ok


if __name__ == '__main__':
    test()

transform = solve

