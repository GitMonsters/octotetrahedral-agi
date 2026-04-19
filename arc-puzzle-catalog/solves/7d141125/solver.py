"""
7d141125: Extract the unique patch from a grid with broken 4-fold rotational symmetry.

The grid has near-4-fold rotational symmetry. Find cells where the actual value
differs from the majority value across the 4 rotational copies. Extract those cells
(with their majority/expected values) as the output. If the bounding box starts at
an odd column, extend left by one.
"""
from collections import Counter

def transform(grid):
    R, C = len(grid), len(grid[0])
    freq = Counter(c for row in grid for c in row)
    bg = freq.most_common(1)[0][0]

    def rot90cw(g):
        R2, C2 = len(g), len(g[0])
        return [[g[R2-1-c][r] for c in range(R2)] for r in range(C2)]

    rot1 = rot90cw(grid)
    rot2 = rot90cw(rot1)
    rot3 = rot90cw(rot2)

    def majority_val(r, c):
        vals = [grid[r][c], rot1[r][c], rot2[r][c], rot3[r][c]]
        return Counter(vals).most_common(1)[0][0]

    # Find cells where actual != majority
    unique_cells = {}
    for r in range(R):
        for c in range(C):
            mv = majority_val(r, c)
            if grid[r][c] != mv:
                unique_cells[(r, c)] = mv

    if not unique_cells:
        return [[bg]]

    ur = sorted(set(r for r, c in unique_cells))
    uc = sorted(set(c for r, c in unique_cells))
    r_min, r_max = min(ur), max(ur)
    c_min, c_max = min(uc), max(uc)

    # Extend c_min left if odd, so output always has even-aligned start column
    if c_min % 2 == 1:
        c_min -= 1

    # Build output: majority value for every cell in the bounding box
    out = []
    for r in range(r_min, r_max + 1):
        row = []
        for c in range(c_min, c_max + 1):
            row.append(majority_val(r, c))
        out.append(row)

    return out
