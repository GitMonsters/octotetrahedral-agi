"""Solver for ARC puzzle b9e38dc0.

The grid has a background, a border shape (one color), and a fill marker inside.
The interior gets filled, and the fill extends through any opening in the border
following the border line slopes. Non-bg/non-fill/non-border cells act as walls
that split the fill into channels.
"""
import json
import math
from collections import Counter, deque


def _get_slope(pts: list[tuple[int, int]], end: str = 'last') -> float:
    """Get slope from boundary points. Try nearest 2, fallback to half endpoints."""
    if len(pts) < 2:
        return 0
    if end == 'last':
        p0, p1 = pts[-2], pts[-1]
        half_pts = pts[len(pts) // 2:]
    else:
        p0, p1 = pts[0], pts[1]
        half_pts = pts[:max(2, (len(pts) + 1) // 2)]

    dr = p1[0] - p0[0]
    if dr != 0:
        slope = (p1[1] - p0[1]) / dr
        if abs(slope) > 1e-9:
            return slope

    if len(half_pts) >= 2:
        p0, p1 = half_pts[0], half_pts[-1]
        dr = p1[0] - p0[0]
        if dr != 0:
            slope = (p1[1] - p0[1]) / dr
            if abs(slope) > 1e-9:
                return slope
    return 0


def transform(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(H) for c in range(W)]
    bg = Counter(flat).most_common(1)[0][0]

    non_bg: dict[int, list[tuple[int, int]]] = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                non_bg.setdefault(grid[r][c], []).append((r, c))

    # Border = largest connected non-bg component (8-connectivity)
    border_color, border_cells, max_sz = None, set(), 0
    for color, cells in non_bg.items():
        cs = set(cells)
        vis: set[tuple[int, int]] = set()
        for s in cells:
            if s in vis:
                continue
            q = deque([s])
            vis.add(s)
            comp = {s}
            while q:
                rr, cc = q.popleft()
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if (nr, nc) in cs and (nr, nc) not in vis:
                            vis.add((nr, nc))
                            q.append((nr, nc))
                            comp.add((nr, nc))
            if len(comp) > max_sz:
                max_sz = len(comp)
                border_color = color
                border_cells = comp

    brs = [r for r, c in border_cells]
    bcs = [c for r, c in border_cells]
    center_r = sum(brs) / len(brs)
    center_c = sum(bcs) / len(bcs)
    min_br, max_br = min(brs), max(brs)
    min_bc, max_bc = min(bcs), max(bcs)

    # Fill color = non-bg non-border color closest to border center
    fill_color, md = None, float('inf')
    for color, cells in non_bg.items():
        if color == border_color:
            continue
        for r, c in cells:
            d2 = abs(r - center_r) + abs(c - center_c)
            if d2 < md:
                md = d2
                fill_color = color
    if fill_color is None:
        return [row[:] for row in grid]

    walls = set(border_cells)

    # Per-row border extremes
    rl: dict[int, int] = {}
    rr_: dict[int, int] = {}
    for r in range(min_br, max_br + 1):
        cols = [c for rr, c in border_cells if rr == r]
        if cols:
            rl[r] = min(cols)
            rr_[r] = max(cols)

    # Per-col border extremes
    ct_: dict[int, int] = {}
    cb_: dict[int, int] = {}
    for c in range(min_bc, max_bc + 1):
        rows = [r for r, cc in border_cells if cc == c]
        if rows:
            ct_[c] = min(rows)
            cb_[c] = max(rows)

    top_o = min_br > 0
    bot_o = max_br < H - 1
    left_o = min_bc > 0
    right_o = max_bc < W - 1

    sr = sorted(rl.keys())
    left_pts = [(r, rl[r]) for r in sr]
    right_pts = [(r, rr_[r]) for r in sr]

    # Extend border lines through openings as virtual walls
    if top_o and len(sr) >= 2:
        l_slope = _get_slope(left_pts, 'first')
        r_slope = _get_slope(right_pts, 'first')
        r0 = sr[0]
        cl, crr = float(rl[r0]), float(rr_[r0])
        for er in range(r0 - 1, -1, -1):
            cl -= l_slope
            crr -= r_slope
            lc, rc = math.floor(cl), math.ceil(crr)
            if lc >= rc:
                break
            if 0 <= lc < W:
                walls.add((er, lc))
            if 0 <= rc < W:
                walls.add((er, rc))

    if bot_o and len(sr) >= 2:
        l_slope = _get_slope(left_pts, 'last')
        r_slope = _get_slope(right_pts, 'last')
        r0 = sr[-1]
        cl, crr = float(rl[r0]), float(rr_[r0])
        for er in range(r0 + 1, H):
            cl += l_slope
            crr += r_slope
            lc, rc = math.floor(cl), math.ceil(crr)
            if lc >= rc:
                break
            if 0 <= lc < W:
                walls.add((er, lc))
            if 0 <= rc < W:
                walls.add((er, rc))

    sc = sorted(ct_.keys())
    top_pts_c = [(c, ct_[c]) for c in sc]
    bot_pts_c = [(c, cb_[c]) for c in sc]

    if left_o and len(sc) >= 2:
        t_slope = _get_slope(top_pts_c, 'first')
        b_slope = _get_slope(bot_pts_c, 'first')
        c0 = sc[0]
        ctt, cbb = float(ct_[c0]), float(cb_[c0])
        for ec in range(c0 - 1, -1, -1):
            ctt -= t_slope
            cbb -= b_slope
            tr, br = math.floor(ctt), math.ceil(cbb)
            if tr >= br:
                break
            if 0 <= tr < H:
                walls.add((tr, ec))
            if 0 <= br < H:
                walls.add((br, ec))

    if right_o and len(sc) >= 2:
        t_slope = _get_slope(top_pts_c, 'last')
        b_slope = _get_slope(bot_pts_c, 'last')
        c0 = sc[-1]
        ctt, cbb = float(ct_[c0]), float(cb_[c0])
        for ec in range(c0 + 1, W):
            ctt += t_slope
            cbb += b_slope
            tr, br = math.floor(ctt), math.ceil(cbb)
            if tr >= br:
                break
            if 0 <= tr < H:
                walls.add((tr, ec))
            if 0 <= br < H:
                walls.add((br, ec))

    # Marker cells: walls that extend toward the nearest opening
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != bg and v != fill_color and v != border_color:
                walls.add((r, c))
                if r < min_br and top_o:
                    for er in range(r - 1, -1, -1):
                        walls.add((er, c))
                elif r > max_br and bot_o:
                    for er in range(r + 1, H):
                        walls.add((er, c))
                elif c < min_bc and left_o:
                    for ec in range(c - 1, -1, -1):
                        walls.add((r, ec))
                elif c > max_bc and right_o:
                    for ec in range(c + 1, W):
                        walls.add((r, ec))
                else:
                    dists: list[tuple[str, int]] = []
                    if top_o:
                        dists.append(('top', r - min_br))
                    if bot_o:
                        dists.append(('bot', max_br - r))
                    if left_o:
                        dists.append(('left', c - min_bc))
                    if right_o:
                        dists.append(('right', max_bc - c))
                    if dists:
                        nearest = min(dists, key=lambda x: x[1])[0]
                        if nearest == 'top':
                            for er in range(r - 1, -1, -1):
                                walls.add((er, c))
                        elif nearest == 'bot':
                            for er in range(r + 1, H):
                                walls.add((er, c))
                        elif nearest == 'left':
                            for ec in range(c - 1, -1, -1):
                                walls.add((r, ec))
                        elif nearest == 'right':
                            for ec in range(c + 1, W):
                                walls.add((r, ec))

    # Flood fill from fill-colored cells
    seeds = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == fill_color]
    filled: set[tuple[int, int]] = set()
    q: deque[tuple[int, int]] = deque()
    for s in seeds:
        if s not in walls:
            filled.add(s)
            q.append(s)
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in filled and (nr, nc) not in walls:
                filled.add((nr, nc))
                q.append((nr, nc))

    out = [row[:] for row in grid]
    for r, c in filled:
        if grid[r][c] == bg:
            out[r][c] = fill_color
    return out


if __name__ == "__main__":
    import os

    task_path = os.path.join(
        os.path.expanduser("~"),
        "ARC_AMD_TRANSFER", "data", "ARC-AGI-2", "data", "evaluation", "b9e38dc0.json",
    )
    with open(task_path) as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = transform(ex["input"])
        expected = ex["output"]
        ok = result == expected
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            for r in range(len(expected)):
                if result[r] != expected[r]:
                    print(f"  Row {r}: got    {result[r]}")
                    print(f"  Row {r}: expect {expected[r]}")

    for i, ex in enumerate(data["test"]):
        result = transform(ex["input"])
        if "output" in ex:
            ok = result == ex["output"]
            print(f"Test {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
        else:
            print(f"Test {i}: produced {len(result)}x{len(result[0])} output")

    if all_pass:
        print("\nALL PASS")


solve = transform
