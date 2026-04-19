from collections import Counter


def most_common_color(grid):
    counts = Counter()
    for row in grid:
        counts.update(row)
    return counts.most_common(1)[0][0]


def in_corner_pattern(r, c, h, w, corner):
    if corner == "tl":
        rr, cc = r, c
    elif corner == "tr":
        rr, cc = r, w - 1 - c
    elif corner == "bl":
        rr, cc = h - 1 - r, c
    else:
        rr, cc = h - 1 - r, w - 1 - c
    if cc > max(0, 2 * (h - 2)):
        return False
    return (cc % 2 == 0 and cc >= rr) or (rr % 2 == 0 and cc <= rr)


def transform(grid):
    h, w = len(grid), len(grid[0])
    background = most_common_color(grid)
    seeds = []
    for corner, r, c in (
        ("tl", 0, 0),
        ("tr", 0, w - 1),
        ("bl", h - 1, 0),
        ("br", h - 1, w - 1),
    ):
        color = grid[r][c]
        if color != background:
            seeds.append((corner, r, c, color))

    present = {corner for corner, _, _, _ in seeds}
    out = [[background for _ in range(w)] for _ in range(h)]

    for r in range(h):
        for c in range(w):
            best = None
            tie = False
            for corner, sr, sc, color in seeds:
                dist = abs(r - sr) + abs(c - sc)
                if best is None or dist < best[0]:
                    best = (dist, corner, color)
                    tie = False
                elif dist == best[0]:
                    tie = True
            if best is None or tie:
                continue

            _, corner, color = best
            if present == {"tl", "br"}:
                if corner == "tl" and r >= h // 2:
                    continue
                if corner == "br" and c < w - h + 1:
                    continue

            if in_corner_pattern(r, c, h, w, corner):
                out[r][c] = color

    return out
