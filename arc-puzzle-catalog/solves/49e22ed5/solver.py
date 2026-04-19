def transform(grid):
    from collections import Counter

    H, W = len(grid), len(grid[0])
    counts = Counter(v for row in grid for v in row)
    bg = counts.most_common(1)[0][0]

    dots = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                dots.add((r, c))

    if len(dots) < 2:
        return [row[:] for row in grid]

    # Find 180-degree rotation center via most common pairwise sum
    dot_list = list(dots)
    pair_sums = Counter()
    for i in range(len(dot_list)):
        for j in range(i + 1, len(dot_list)):
            r1, c1 = dot_list[i]
            r2, c2 = dot_list[j]
            pair_sums[(r1 + r2, c1 + c2)] += 1

    (sr, sc), _ = pair_sums.most_common(1)[0]

    out = [row[:] for row in grid]
    for r, c in dots:
        rr, rc = sr - r, sc - c
        if 0 <= rr < H and 0 <= rc < W and (rr, rc) not in dots:
            out[rr][rc] = 0

    return out
