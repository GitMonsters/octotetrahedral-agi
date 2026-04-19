"""
ARC-AGI puzzle 38d44dbd solver.

Rule:
  - Grid has 2 dominant colors (maze pattern) and optionally a rare template color
  - The rare template marks one arm of cross shapes (arm length HALF=2)
  - In the output, dominant background cells within each cross shape become red (2)
  - 3-color case: find cross centers from pairs of template cells at distance 2*HALF in same row/col
  - 2-color case: find cross centers from majority arm + minority outer pair pattern
"""
from collections import Counter
from itertools import combinations


HALF = 2
RED = 2


def _cross_cells(r, c, R, C):
    cells = []
    for d in range(-HALF, HALF + 1):
        if 0 <= r + d < R:
            cells.append((r + d, c))
        if d != 0 and 0 <= c + d < C:
            cells.append((r, c + d))
    return cells


def _find_centers_3color(inp, bgc, template, R, C):
    """Find cross centers using template cell pairs at distance 2*HALF."""
    tcells = [(r, c) for r in range(R) for c in range(C) if inp[r][c] == template]
    centers = set()
    for i, (r1, c1) in enumerate(tcells):
        for r2, c2 in tcells[i + 1:]:
            if r1 == r2 and abs(c1 - c2) == 2 * HALF:
                centers.add((r1, (c1 + c2) // 2))
            elif c1 == c2 and abs(r1 - r2) == 2 * HALF:
                centers.add(((r1 + r2) // 2, c1))
    return centers


def _find_centers_2color(inp, bgc, noisec, R, C):
    """
    Find cross centers for the 2-color case.
    Primary arm = majority (bgc), secondary arm outer pair = minority (noisec).
    Algorithm:
      1. One candidate per column (V-arm) and per row (H-arm), using bridge priority.
      2. Remove direct overlaps (Chebyshev <= 1) by bridge > solid.
      3. Resolve conflicts (Chebyshev 2-5): higher score wins; tie -> V/V prefer lower col, H/H prefer lower row.
    """
    def score_cand(h_arm_5, v_arm_5, is_bridge):
        h_min = sum(1 for x in h_arm_5 if x == noisec) / len(h_arm_5)
        v_maj = sum(1 for x in v_arm_5 if x == bgc) / len(v_arm_5)
        return min(h_min, v_maj) * (1.0 if is_bridge else 0.9)

    # Collect all raw candidates per column (V-arm) and per row (H-arm)
    col_cands = {}  # col -> list of (r, score, is_bridge)
    for c in range(HALF, C - HALF):
        lst = []
        for r in range(HALF, R - HALF):
            # V-arm: ±1 and ±HALF in col direction = majority, H outer pair = minority
            if (inp[r - 1][c] == bgc and inp[r + 1][c] == bgc and
                    inp[r - HALF][c] == bgc and inp[r + HALF][c] == bgc and
                    inp[r][c - HALF] == noisec and inp[r][c + HALF] == noisec):
                h5 = [inp[r][c + d] for d in range(-HALF, HALF + 1)]
                v5 = [inp[r + d][c] for d in range(-HALF, HALF + 1)]
                is_bridge = (inp[r][c] == noisec)
                s = score_cand(h5, v5, is_bridge)
                lst.append((r, s, is_bridge))
        if lst:
            bridges = [x for x in lst if x[2]]
            solids = [x for x in lst if not x[2]]
            if bridges:
                best = max(bridges, key=lambda x: (x[1], x[0]))
                col_cands[c] = (best[0], best[1], True, 'v')
            else:
                best = min(solids, key=lambda x: x[0])
                col_cands[c] = (best[0], best[1], False, 'v')

    row_cands = {}  # row -> (c, score, is_bridge, 'h')
    for r in range(HALF, R - HALF):
        lst = []
        for c in range(HALF, C - HALF):
            if (inp[r][c - 1] == bgc and inp[r][c + 1] == bgc and
                    inp[r][c - HALF] == bgc and inp[r][c + HALF] == bgc and
                    inp[r - HALF][c] == noisec and inp[r + HALF][c] == noisec):
                h5 = [inp[r][c + d] for d in range(-HALF, HALF + 1)]
                v5 = [inp[r + d][c] for d in range(-HALF, HALF + 1)]
                is_bridge = (inp[r][c] == noisec)
                h_maj = sum(1 for x in h5 if x == bgc) / len(h5)
                v_min = sum(1 for x in v5 if x == noisec) / len(v5)
                s = min(h_maj, v_min) * (1.0 if is_bridge else 0.9)
                lst.append((c, s, is_bridge))
        if lst:
            bridges = [x for x in lst if x[2]]
            solids = [x for x in lst if not x[2]]
            if bridges:
                best = max(bridges, key=lambda x: (x[1], x[0]))
                row_cands[r] = (best[0], best[1], True, 'h')
            else:
                best = min(solids, key=lambda x: x[0])
                row_cands[r] = (best[0], best[1], False, 'h')

    # Build selected set: (r, c) -> (score, is_bridge, arm_dir)
    selected = {}
    for c, (r, s, br, d) in col_cands.items():
        selected[(r, c)] = (s, br, d)
    for r, (c, s, br, d) in row_cands.items():
        key = (r, c)
        if key not in selected or s > selected[key][0]:
            selected[key] = (s, br, d)

    def linf(p1, p2):
        return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

    # Step 2: remove Chebyshev-1 overlaps (bridge beats solid)
    changed = True
    while changed:
        changed = False
        pts = list(selected.keys())
        for i, p1 in enumerate(pts):
            if p1 not in selected:
                continue
            for p2 in pts[i + 1:]:
                if p2 not in selected:
                    continue
                if linf(p1, p2) <= 1:
                    s1, br1, d1 = selected[p1]
                    s2, br2, d2 = selected[p2]
                    if br1 and not br2:
                        del selected[p2]; changed = True
                    elif br2 and not br1:
                        del selected[p1]; changed = True; break
                    elif s1 >= s2:
                        del selected[p2]; changed = True
                    else:
                        del selected[p1]; changed = True; break

    # Step 3: resolve conflicts (Chebyshev 2-5) iteratively
    changed = True
    while changed:
        changed = False
        pts = list(selected.keys())
        for i, p1 in enumerate(pts):
            if p1 not in selected:
                continue
            for p2 in pts[i + 1:]:
                if p2 not in selected:
                    continue
                d = linf(p1, p2)
                if 2 <= d <= 5:
                    s1, br1, d1 = selected[p1]
                    s2, br2, d2 = selected[p2]
                    # Determine loser
                    if s1 > s2:
                        loser = p2
                    elif s2 > s1:
                        loser = p1
                    elif br1 and not br2:
                        loser = p2
                    elif br2 and not br1:
                        loser = p1
                    elif d1 == 'v' and d2 == 'v':
                        # V vs V: prefer lower column
                        loser = p1 if p1[1] > p2[1] else p2
                    elif d1 == 'h' and d2 == 'h':
                        # H vs H: prefer lower row
                        loser = p1 if p1[0] > p2[0] else p2
                    else:
                        # V vs H: prefer lower row
                        loser = p1 if p1[0] > p2[0] else p2
                    del selected[loser]
                    changed = True
                    break

    return set(selected.keys())


def transform(grid):
    inp = grid
    R, C = len(inp), len(inp[0])
    counts = Counter(c for row in inp for c in row)
    mc = sorted(counts.items(), key=lambda x: -x[1])

    bgc = mc[0][0]
    total = sum(v for _, v in mc)

    if len(mc) >= 3 and mc[2][1] / total < 0.20:
        # 3-color case: rare template
        template = mc[2][0]
        centers = _find_centers_3color(inp, bgc, template, R, C)
    else:
        # 2-color case
        if len(mc) < 2:
            return [list(row) for row in inp]
        noisec = mc[1][0]
        centers = _find_centers_2color(inp, bgc, noisec, R, C)

    out = [list(row) for row in inp]
    for cr, cc in centers:
        for r, c in _cross_cells(cr, cc, R, C):
            if inp[r][c] == bgc:
                out[r][c] = RED
    return out
