def transform(grid):
    from collections import Counter
    H, W = len(grid), len(grid[0])
    counts = Counter(v for row in grid for v in row)
    bg = counts.most_common(1)[0][0]

    color_pos = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != bg:
                color_pos.setdefault(v, []).append((r, c))

    # Find marker: exactly 4 positions forming rectangle corners
    marker = None
    r0 = c0 = r1 = c1 = 0
    for color, positions in color_pos.items():
        if len(positions) == 4:
            rows = sorted(set(r for r, c in positions))
            cols = sorted(set(c for r, c in positions))
            if len(rows) == 2 and len(cols) == 2:
                corners = {(rows[0], cols[0]), (rows[0], cols[1]),
                           (rows[1], cols[0]), (rows[1], cols[1])}
                if set(map(tuple, positions)) == corners:
                    marker = color
                    r0, r1 = rows
                    c0, c1 = cols
                    break

    if marker is not None:
        pattern = [c for c in color_pos if c != marker][0] if len(color_pos) > 1 else None
        out = []
        for r in range(r0 + 1, r1):
            row = []
            for c in range(c0 + 1, c1):
                v = grid[r][c]
                row.append(marker if v == pattern else v)
            out.append(row)
        return out
    else:
        all_pos = [(r, c) for positions in color_pos.values() for r, c in positions]
        min_r = min(r for r, c in all_pos)
        max_r = max(r for r, c in all_pos)
        min_c = min(c for r, c in all_pos)
        max_c = max(c for r, c in all_pos)
        return [[bg] * (max_c - min_c + 1) for _ in range(max_r - min_r + 1)]
