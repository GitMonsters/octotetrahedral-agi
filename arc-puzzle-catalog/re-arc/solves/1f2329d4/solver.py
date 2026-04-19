from collections import Counter
from itertools import combinations


def transform(grid):
    H = len(grid)
    W = len(grid[0])

    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    colors = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                colors.setdefault(grid[r][c], []).append((r, c))

    # Find the rectangle defined by 4 same-color corner pixels
    rect = None
    for v, pts in colors.items():
        if len(pts) >= 4:
            for combo in combinations(pts, 4):
                rows = sorted(set(p[0] for p in combo))
                cols = sorted(set(p[1] for p in combo))
                if len(rows) == 2 and len(cols) == 2:
                    r1, r2 = rows
                    c1, c2 = cols
                    if all((r, c) in combo for r in rows for c in cols):
                        rect = (r1, r2, c1, c2, v)
                        break
            if rect:
                break

    r1, r2, c1, c2, cv = rect
    oH = r2 - r1 + 1
    oW = c2 - c1 + 1
    corners = {(r1, c1), (r1, c2), (r2, c1), (r2, c2)}

    border_set = set()
    for r in range(oH):
        for c in range(oW):
            if r == 0 or r == oH - 1 or c == 0 or c == oW - 1:
                if not ((r == 0 or r == oH - 1) and (c == 0 or c == oW - 1)):
                    border_set.add((r, c))

    scatter_colors = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and (r, c) not in corners:
                v = grid[r][c]
                scatter_colors.setdefault(v, []).append((r, c))

    def find_valid_offsets(pts):
        valid = []
        min_r = min(r for r, c in pts)
        max_r = max(r for r, c in pts)
        min_c = min(c for r, c in pts)
        max_c = max(c for r, c in pts)
        for a in range(min_r - oH + 1, max_r + 1):
            for b in range(min_c - oW + 1, max_c + 1):
                if all((r - a, c - b) in border_set for r, c in pts):
                    valid.append((a, b))
        return valid

    out = [[bg] * oW for _ in range(oH)]
    out[0][0] = cv
    out[0][oW - 1] = cv
    out[oH - 1][0] = cv
    out[oH - 1][oW - 1] = cv

    occupied = set()
    deferred = {}

    for v, pts in scatter_colors.items():
        offsets = find_valid_offsets(pts)
        if len(offsets) == 1:
            a, b = offsets[0]
            for r, c in pts:
                pos = (r - a, c - b)
                out[pos[0]][pos[1]] = v
                occupied.add(pos)
        else:
            deferred[v] = pts

    for v, pts in deferred.items():
        n = len(pts)
        found = False
        for size1 in range(n // 2, 0, -1):
            if found:
                break
            for combo in combinations(range(n), size1):
                g1 = [pts[i] for i in combo]
                g2 = [pts[i] for i in range(n) if i not in combo]
                o1 = find_valid_offsets(g1)
                o2 = find_valid_offsets(g2)
                if o1 and o2:
                    for a1, b1 in o1:
                        for a2, b2 in o2:
                            pos1 = set((r - a1, c - b1) for r, c in g1)
                            pos2 = set((r - a2, c - b2) for r, c in g2)
                            if (not pos1.intersection(pos2)
                                    and not pos1.intersection(occupied)
                                    and not pos2.intersection(occupied)):
                                for r, c in g1:
                                    p = (r - a1, c - b1)
                                    out[p[0]][p[1]] = v
                                    occupied.add(p)
                                for r, c in g2:
                                    p = (r - a2, c - b2)
                                    out[p[0]][p[1]] = v
                                    occupied.add(p)
                                found = True
                                break
                        if found:
                            break
                if found:
                    break

    return out
