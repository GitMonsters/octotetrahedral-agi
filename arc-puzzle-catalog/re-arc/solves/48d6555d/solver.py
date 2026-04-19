def transform(input_grid: list[list[int]]) -> list[list[int]]:
    from collections import Counter, defaultdict

    grid = input_grid
    nrows, ncols = len(grid), len(grid[0])

    # Background = most common color
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Non-background pixels
    pixels = [(r, c, grid[r][c]) for r in range(nrows) for c in range(ncols) if grid[r][c] != bg]
    if not pixels:
        return [[bg] * 3 for _ in range(3)]

    # 8-connected clusters via union-find
    parent: dict = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for r, c, _ in pixels:
        parent[(r, c)] = (r, c)
    coords = [(r, c) for r, c, _ in pixels]
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            r1, c1 = coords[i]
            r2, c2 = coords[j]
            if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                union((r1, c1), (r2, c2))

    cmap = defaultdict(list)
    for r, c, v in pixels:
        cmap[find((r, c))].append((r, c, v))
    clusters = list(cmap.values())

    # Merge clusters whose combined bounding box fits within 3x3
    def bb(members):
        rs = [r for r, c, v in members]
        cs = [c for r, c, v in members]
        return min(rs), max(rs), min(cs), max(cs)

    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                combo = clusters[i] + clusters[j]
                r0, r1, c0, c1 = bb(combo)
                if r1 - r0 <= 2 and c1 - c0 <= 2:
                    clusters[i] = combo
                    del clusters[j]
                    changed = True
                    break
            if changed:
                break

    # Find anchor color: appears exactly once in every cluster
    all_colors = sorted(set(v for _, _, v in pixels))
    anchor_color = None
    for color in all_colors:
        if all(sum(1 for _, _, v in cl if v == color) == 1 for cl in clusters):
            anchor_color = color
            break

    if anchor_color is not None:
        # Collect offsets from each cluster's anchor to other pixels
        offsets = []
        for cl in clusters:
            apos = next((r, c) for r, c, v in cl if v == anchor_color)
            for r, c, v in cl:
                if (r, c) != apos:
                    offsets.append((r - apos[0], c - apos[1]))

        if offsets:
            min_dr = min(dr for dr, _ in offsets)
            max_dr = max(dr for dr, _ in offsets)
            min_dc = min(dc for _, dc in offsets)
            max_dc = max(dc for _, dc in offsets)
        else:
            min_dr = max_dr = min_dc = max_dc = 0

        ar, ac = -min_dr, -min_dc  # anchor output position

        out = [[bg] * 3 for _ in range(3)]
        out[ar][ac] = anchor_color
        for cl in clusters:
            apos = next((r, c) for r, c, v in cl if v == anchor_color)
            for r, c, v in cl:
                dr, dc = r - apos[0], c - apos[1]
                orr, occ = ar + dr, ac + dc
                if 0 <= orr < 3 and 0 <= occ < 3:
                    out[orr][occ] = v
        return out

    # Fallback: no anchor – overlay clusters greedily (largest BB first)
    def bb_area(cl):
        r0, r1, c0, c1 = bb(cl)
        return (r1 - r0 + 1) * (c1 - c0 + 1)

    clusters.sort(key=bb_area, reverse=True)
    out = [[bg] * 3 for _ in range(3)]

    for cl in clusters:
        r0, r1, c0, c1 = bb(cl)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        best, best_conflicts = None, float('inf')
        for sr in range(4 - h):
            for sc in range(4 - w):
                conflicts = sum(
                    1 for r, c, v in cl
                    if out[sr + r - r0][sc + c - c0] != bg
                    and out[sr + r - r0][sc + c - c0] != v
                )
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best = (sr, sc)
        if best is not None:
            sr, sc = best
            for r, c, v in cl:
                out[sr + r - r0][sc + c - c0] = v

    return out
