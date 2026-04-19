"""
ARC puzzle 5aa85779 solver.

The pattern has D4 symmetry (4-fold rotation + bilateral reflections).
A mask color covers parts of the pattern. We reconstruct the pattern by:
1. Finding the mask color and bilateral symmetry axes
2. Using all 8 D4 symmetry operations to propagate known values into masked cells
3. Handling remaining cells via edge reflection about mask segment boundaries
"""
from collections import Counter


def find_mask_and_axes(grid):
    H, W = len(grid), len(grid[0])
    colors = Counter()
    for r in range(H):
        for c in range(W):
            colors[grid[r][c]] += 1
    best = None
    for mask_color in colors:
        if colors[mask_color] < 3:
            continue
        is_mask = [[grid[r][c] == mask_color for c in range(W)] for r in range(H)]
        valid_v, valid_h = [], []
        for v2 in range(0, 2 * H + 1):
            ok, pairs = True, 0
            for r in range(H):
                mr = v2 - r
                if 0 <= mr < H and mr != r:
                    for c in range(W):
                        if not is_mask[r][c] and not is_mask[mr][c]:
                            pairs += 1
                            if grid[r][c] != grid[mr][c]:
                                ok = False
                                break
                if not ok:
                    break
            if ok and pairs > 0:
                valid_v.append((v2, pairs))
        for h2 in range(0, 2 * W + 1):
            ok, pairs = True, 0
            for r in range(H):
                for c in range(W):
                    mc = h2 - c
                    if 0 <= mc < W and mc != c:
                        if not is_mask[r][c] and not is_mask[r][mc]:
                            pairs += 1
                            if grid[r][c] != grid[r][mc]:
                                ok = False
                                break
                if not ok:
                    break
            if ok and pairs > 0:
                valid_h.append((h2, pairs))
        if not valid_v or not valid_h:
            continue
        best_v = max(valid_v, key=lambda x: x[1])
        best_h = max(valid_h, key=lambda x: x[1])
        score = best_v[1] + best_h[1]
        if best is None or score > best[0]:
            best = (score, mask_color, best_v[0], best_h[0])
    return best[1], best[2], best[3]


def transform(input_grid):
    grid = [row[:] for row in input_grid]
    H, W = len(grid), len(grid[0])
    mask_color, v2, h2 = find_mask_and_axes(grid)
    orig_mask = [[grid[r][c] == mask_color for c in range(W)] for r in range(H)]
    is_mask = [[orig_mask[r][c] for c in range(W)] for r in range(H)]
    bg_counter = Counter()
    for r in range(H):
        for c in range(W):
            if not orig_mask[r][c]:
                bg_counter[grid[r][c]] += 1
    bg_color = bg_counter.most_common(1)[0][0]

    def get_symmetry_positions(r, c):
        """All D4 symmetry partners of (r,c) about center (v2/2, h2/2)."""
        positions = []
        dr2 = 2 * r - v2
        dc2 = 2 * c - h2
        transforms = [
            (dr2, dc2), (dc2, -dr2), (-dr2, -dc2), (-dc2, dr2),
            (-dr2, dc2), (dr2, -dc2), (dc2, dr2), (-dc2, -dr2),
        ]
        for tdr2, tdc2 in transforms:
            tr2 = tdr2 + v2
            tc2 = tdc2 + h2
            if tr2 % 2 != 0 or tc2 % 2 != 0:
                continue
            tr, tc = tr2 // 2, tc2 // 2
            if 0 <= tr < H and 0 <= tc < W and (tr, tc) != (r, c):
                positions.append((tr, tc))
        return positions

    def d4_fill():
        changed = True
        while changed:
            changed = False
            for r in range(H):
                for c in range(W):
                    if not is_mask[r][c]:
                        continue
                    for tr, tc in get_symmetry_positions(r, c):
                        if not is_mask[tr][tc]:
                            grid[r][c] = grid[tr][tc]
                            is_mask[r][c] = False
                            changed = True
                            break

    def get_orbits():
        remaining = set(
            (r, c) for r in range(H) for c in range(W) if is_mask[r][c]
        )
        visited = set()
        orbits = []
        for r, c in sorted(remaining):
            if (r, c) in visited:
                continue
            orbit = set()
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                if (rr, cc) in orbit:
                    continue
                orbit.add((rr, cc))
                visited.add((rr, cc))
                for tr, tc in get_symmetry_positions(rr, cc):
                    if is_mask[tr][tc] and (tr, tc) not in orbit:
                        stack.append((tr, tc))
            orbits.append(orbit)
        return orbits

    d4_fill()

    # Handle remaining orbits via edge reflection
    for iteration in range(500):
        orbits = get_orbits()
        if not orbits:
            break
        best = None
        for orbit in orbits:
            refs = []
            min_seg = float('inf')
            for r, c in orbit:
                cl, cr = c, c
                while cl > 0 and orig_mask[r][cl - 1]:
                    cl -= 1
                while cr < W - 1 and orig_mask[r][cr + 1]:
                    cr += 1
                rs = cr - cl + 1
                rt, rb = r, r
                while rt > 0 and orig_mask[rt - 1][c]:
                    rt -= 1
                while rb < H - 1 and orig_mask[rb + 1][c]:
                    rb += 1
                cs = rb - rt + 1
                min_seg = min(min_seg, rs, cs)
                if cs >= 2:
                    tm = 2 * rt - 1 - r
                    if 0 <= tm < H and not orig_mask[tm][c]:
                        refs.append((cs, input_grid[tm][c]))
                    bm = 2 * rb + 1 - r
                    if 0 <= bm < H and not orig_mask[bm][c]:
                        refs.append((cs, input_grid[bm][c]))
                if rs >= 2:
                    lm = 2 * cl - 1 - c
                    if 0 <= lm < W and not orig_mask[r][lm]:
                        refs.append((rs, input_grid[r][lm]))
                    rm = 2 * cr + 1 - c
                    if 0 <= rm < W and not orig_mask[r][rm]:
                        refs.append((rs, input_grid[r][rm]))

            # Genuine isolated mask cells
            if min_seg <= 1:
                priority = (100, 0, 0)
                if best is None or priority > best[0]:
                    best = (priority, orbit, mask_color)
                continue
            if min_seg <= 2 and refs:
                sl = min(r[0] for r in refs)
                sv = [r[1] for r in refs if r[0] == sl]
                if all(v == bg_color for v in sv):
                    priority = (90, 0, 0)
                    if best is None or priority > best[0]:
                        best = (priority, orbit, mask_color)
                    continue

            # Non-bg at shortest segment
            if refs:
                sl = min(r[0] for r in refs)
                sv = [r[1] for r in refs if r[0] == sl]
                non_bg = [v for v in sv if v != bg_color]
                if non_bg:
                    val = Counter(non_bg).most_common(1)[0][0]
                    priority = (50, -sl, len(non_bg))
                    if best is None or priority > best[0]:
                        best = (priority, orbit, val)
                    continue

            priority = (10, 0, 0)
            if best is None or priority > best[0]:
                best = (priority, orbit, bg_color)

        if best is None:
            break
        _, orbit, val = best
        for r, c in orbit:
            grid[r][c] = val
            is_mask[r][c] = False
        d4_fill()

    for r in range(H):
        for c in range(W):
            if is_mask[r][c]:
                grid[r][c] = bg_color
    return grid
