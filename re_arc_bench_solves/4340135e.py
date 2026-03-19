def transform(grid):
    rows, cols = len(grid), len(grid[0])
    from collections import Counter
    flat = [c for r in grid for c in r]
    bg = Counter(flat).most_common(1)[0][0]
    out = [row[:] for row in grid]

    # Find the largest solid rectangle of a single non-bg color
    best = None
    for r1 in range(rows):
        for c1 in range(cols):
            if grid[r1][c1] == bg:
                continue
            rc = grid[r1][c1]
            for r2 in range(r1, rows):
                for c2 in range(c1, cols):
                    ok = True
                    for rr in range(r1, r2 + 1):
                        for cc in range(c1, c2 + 1):
                            if grid[rr][cc] != rc:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if best is None or area > best[0]:
                            best = (area, r1, c1, r2, c2, rc)

    if best is None:
        return grid

    _, r1, c1, r2, c2, rc = best

    # Extend non-bg, non-rect cells toward the rectangle
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == bg or grid[r][c] == rc:
                continue
            color = grid[r][c]
            if r1 <= r <= r2 and c < c1:
                for cc in range(c + 1, c1):
                    if out[r][cc] == bg:
                        out[r][cc] = color
            elif r1 <= r <= r2 and c > c2:
                for cc in range(c2 + 1, c):
                    if out[r][cc] == bg:
                        out[r][cc] = color
            if c1 <= c <= c2 and r < r1:
                for rr in range(r + 1, r1):
                    if out[rr][c] == bg:
                        out[rr][c] = color
            elif c1 <= c <= c2 and r > r2:
                for rr in range(r2 + 1, r):
                    if out[rr][c] == bg:
                        out[rr][c] = color

    # Connect same-colored cells on rectangle rows
    for r in range(r1, r2 + 1):
        cells = {}
        for c in range(cols):
            if out[r][c] != bg:
                cells.setdefault(out[r][c], []).append(c)
        for color, cs in cells.items():
            cs.sort()
            for i in range(len(cs) - 1):
                for cc in range(cs[i] + 1, cs[i + 1]):
                    if out[r][cc] == bg:
                        out[r][cc] = color

    # Connect same-colored cells on rectangle columns
    for c in range(c1, c2 + 1):
        cells = {}
        for r in range(rows):
            if out[r][c] != bg:
                cells.setdefault(out[r][c], []).append(r)
        for color, rs in cells.items():
            rs.sort()
            for i in range(len(rs) - 1):
                for rr in range(rs[i] + 1, rs[i + 1]):
                    if out[rr][c] == bg:
                        out[rr][c] = color

    return out
